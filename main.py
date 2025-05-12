import os
import json
import base64
import time
import threading
import io
import logging
import queue
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
import websocket
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("realtime")

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
load_dotenv()  # makes OPENAI_API_KEY available
logger.info("Environment variables loaded")

# --------------------------------------------------------------------------- #
# App-wide constants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    SAMPLE_RATE: int = 16_000
    CHANNELS: int = 1
    CHUNK_SIZE: int = 4096               # frames per sounddevice callback
    API_MODEL: str = "gpt-4o-realtime-preview-2024-10-01"
    WS_ENDPOINT: str = f"wss://api.openai.com/v1/realtime?model={API_MODEL}"
    OUTPUT_SAMPLE_RATE: int = 24_000     # PCM-16 mono from the API
    WS_TIMEOUT: int = 60                 # seconds to wait for session.created


# --------------------------------------------------------------------------- #
# CLI Tool for command execution
# --------------------------------------------------------------------------- #
class UseCli:
    """Tool for executing CLI commands on macOS"""

    @staticmethod
    def execute(command: str) -> Dict[str, Any]:
        """Execute a command line instruction and return the result"""
        logger.info(f"Executing CLI command: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )

            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode

            logger.debug(f"Command output (return code {return_code})")

            return {
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                "error": str(e),
                "stdout": "",
                "stderr": f"Error: {str(e)}",
                "return_code": -1,
            }


# --------------------------------------------------------------------------- #
# Realtime conversation helper
# --------------------------------------------------------------------------- #
class RealtimeConversation:
    """
    Encapsulates a single, full-duplex voice session with the OpenAI realtime
    beta API.  A dedicated playback thread is used so that audio chunks are
    rendered strictly in order, preventing overlapping “double talking”.
    """

    # --------------------------- initialisation --------------------------- #
    def __init__(self) -> None:
        logger.debug("Initialising RealtimeConversation")
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment variables")

        # OpenAI client – kept around for potential future use
        self.client = OpenAI(api_key=self.api_key)

        # Runtime state
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.session_ready = threading.Event()  # becomes true after session.created
        self.closed = threading.Event()         # becomes true after on_close

        self.stream: Optional[sd.InputStream] = None
        self.is_recording = False

        # Initialize the CLI tool
        self.cli_tool = UseCli()

        # Define tool configuration
        self.tools = [{
            "type": "function",
            "description": "Execute a CLI command on macOS",
            "name": "usecli",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute on the macOS terminal"
                    }
                },
                "required": ["command"]
            }
        }]

        # Buffers
        self.output_audio_chunks: List[bytes] = []

        # Serialised audio playback
        self._audio_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._player_thread = threading.Thread(
            target=self._audio_player_loop,
            name="audio-player",
            daemon=True,
        )
        self._player_thread.start()

        # Tracking whether we have already printed the “Assistant:” prefix
        self._printed_prefix = False

    # ----------------------------- websocket ------------------------------ #
    def _on_open(self, ws):
        logger.info("WebSocket connection established")

        # Keep the payload minimal – superfluous fields can trigger silent errors
        session_update = {
            "type": "session.update",
            "session": {
                "model": Settings.API_MODEL,
                "voice": "alloy",
                "instructions": (
                    "You are a helpful AI assistant. Respond thoughtfully "
                    "and concisely. You can run commands on the user's system using the usecli tool."
                ),
                "tools": self.tools,
            },
        }
        ws.send(json.dumps(session_update))
        logger.debug("session.update sent")

    # Helper: tolerate schema changes where `delta` may be str or dict
    @staticmethod
    def _extract_delta(delta: Union[str, dict, None], key: str | None = None) -> Optional[str]:
        """
        Safely pull a value from `delta`, which may be either:

        • a bare string  – return it directly
        • a dict         – return `delta[key]`
        • None           – return None
        """
        if delta is None:
            return None
        if isinstance(delta, str):
            # If the caller provided a key, only return the string when the key is
            # the implicit value we are interested in (text/chunk).  Otherwise None.
            if key is None or key in ("text", "chunk"):
                return delta
            return None
        if isinstance(delta, dict):
            return delta.get(key) if key else None
        return None

    def _on_message(self, ws, message):
        try:
            event = json.loads(message)
            print("⚠️ ", event)
        except json.JSONDecodeError as exc:  # pragma: no cover
            logger.error("Failed to decode WS message: %s", exc, exc_info=True)
            return

        etype = event.get("type", "")
        logger.debug("WS → %s", etype)

        if etype == "session.created":
            logger.info("Session created – ready to stream audio")
            self.session_ready.set()

        elif etype == "input_audio_buffer.speech_started":
            logger.info("💜 Speech detected")
            self.output_audio_chunks = []

        elif etype == "input_audio_buffer.speech_stopped":
            logger.info("Speech ended – waiting for assistant…")
        elif etype == "conversation.item.created":
            logger.info("💜 Conversation item created")
            logger.info(event)

        elif etype == "response.audio.delta":
            # In recent builds `delta` is the base64 string directly – older
            # versions wrapped it inside {"chunk": "<b64>"}.  Support both.
            chunk_b64 = self._extract_delta(event.get("delta"), "chunk")
            if chunk_b64:
                try:
                    chunk = base64.b64decode(chunk_b64)
                except Exception as exc:  # pragma: no cover
                    logger.error("Failed to decode audio chunk: %s", exc, exc_info=True)
                    return
                self.output_audio_chunks.append(chunk)
                self._audio_queue.put(chunk)  # enqueue for serial playback

        elif etype == "response.audio_transcript.delta":
            # Print transcript once, streaming text without repeating prefix
            text = self._extract_delta(event.get("delta"), "text")
            if text:
                if not self._printed_prefix:
                    print("Assistant: ", end="", flush=True)
                    self._printed_prefix = True
                print(text, end="", flush=True)

        elif etype == "message.tool_calls":
            # Handle tool calls from the assistant
            logger.info("Received tool call request")
            tool_calls = event.get("tool_calls", [])

            for tool_call in tool_calls:
                tool_type = tool_call.get("type")
                if tool_type != "function":
                    logger.warning(f"Unsupported tool type: {tool_type}")
                    continue

                function_call = tool_call.get("function", {})
                name = function_call.get("name")
                arguments = function_call.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse arguments: {arguments}")
                        continue

                # Process the usecli tool call
                if name == "usecli":
                    command = arguments.get("command")
                    if command:
                        logger.info(f"Executing command: {command}")
                        result = self.cli_tool.execute(command)

                        # Send the tool call result back
                        tool_result = {
                            "type": "message.tool_call_results",
                            "tool_call_results": [{
                                "id": tool_call.get("id"),
                                "content": json.dumps(result)
                            }]
                        }
                        ws.send(json.dumps(tool_result))
                        logger.debug("Tool call result sent")

        elif etype == "response.done":
            logger.info("Assistant finished speaking")

            print()  # newline after transcript
            self._printed_prefix = False  # ready for next exchange

        elif etype == "error":
            logger.error("API error: %s", event.get("message"))

    def _on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)

    def _on_close(self, ws, status_code, msg):
        logger.info("WebSocket closed (%s): %s", status_code, msg)
        self.closed.set()
        self.session_ready.clear()

    # ------------------------- connection helpers ------------------------- #
    def connect(self) -> None:
        """Open the websocket and block until session.created or timeout."""
        logger.info("Connecting to OpenAI realtime endpoint…")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "realtime=v1",
        }

        self.ws = websocket.WebSocketApp(
            Settings.WS_ENDPOINT,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()

        if not self.session_ready.wait(Settings.WS_TIMEOUT):
            raise TimeoutError("Timed-out waiting for session.created event")

    # --------------------- microphone / audio streaming ------------------- #
    def _audio_callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            logger.warning("InputStream status: %s", status)

        if not (
            self.is_recording
            and self.ws
            and self.ws.sock
            and self.ws.sock.connected
        ):
            return

        # Convert to 16-bit PCM mono → base64
        pcm16 = np.int16(indata[:, 0] * 32767).tobytes()
        payload = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
        }

        try:
            self.ws.send(json.dumps(payload))
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to send audio frame: %s", exc, exc_info=True)

    def start_recording(self) -> None:
        if not self.session_ready.is_set():
            raise RuntimeError("WebSocket session not ready")

        logger.info("Starting microphone capture – speak now (Ctrl-C to stop)")
        self.is_recording = True

        self.stream = sd.InputStream(
            channels=Settings.CHANNELS,
            samplerate=Settings.SAMPLE_RATE,
            blocksize=Settings.CHUNK_SIZE,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        logger.info("Stopping microphone capture")
        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                time.sleep(0.2)
                self.ws.send(json.dumps({"type": "response.create"}))
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to finalise conversation: %s", exc, exc_info=True)

    # ------------------------ audio playback helpers ---------------------- #
    def _audio_player_loop(self) -> None:
        """
        Dedicated thread that pulls raw PCM chunks from the queue and plays
        them serially.  Because only one thread performs playback, chunks
        cannot overlap – eliminating the “double talking” issue.
        """
        while True:
            chunk = self._audio_queue.get()
            if chunk is None:  # shutdown sentinel
                break
            try:
                seg = AudioSegment.from_file(
                    io.BytesIO(chunk),
                    format="raw",
                    sample_width=2,
                    channels=1,
                    frame_rate=Settings.OUTPUT_SAMPLE_RATE,
                )
                play(seg)  # blocking until `seg` has finished playing
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to play audio chunk: %s", exc, exc_info=True)

    # ----------------------------- shutdown ------------------------------ #
    def close(self) -> None:
        logger.info("Shutting down conversation")
        self.stop_recording()

        # Trigger graceful shutdown of audio player
        self._audio_queue.put(None)

        if self.ws:
            try:
                self.ws.close()
            except Exception:  # pragma: no cover
                pass

        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2)

        if self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=2)

        logger.info("Session ended")


# --------------------------------------------------------------------------- #
# entry-point
# --------------------------------------------------------------------------- #
def main() -> None:
    logger.info("== Real-time voice conversation – press Ctrl-C to quit ==")
    convo = RealtimeConversation()

    try:
        convo.connect()
        convo.start_recording()

        while not convo.closed.is_set():
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        convo.close()
        logger.info("Good-bye!")
        exit()


if __name__ == "__main__":
    main()