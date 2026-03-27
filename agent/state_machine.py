"""
Core state machine — the brain of the voice agent.

Coordinates VAD, STT, LLM, TTS, backchannels, and fillers into a
coherent full-duplex conversation loop.

States:
    IDLE      — waiting for user to speak
    LISTENING — user is speaking; STT running, backchannels injected
    THINKING  — user stopped; LLM generating, filler may play
    SPEAKING  — TTS audio streaming to user; watching for barge-in

Usage:
    agent = VoiceAgent(websocket)
    await agent.run()  # main loop — runs until connection closes
"""

import asyncio
import enum
import time
from dataclasses import dataclass

import numpy as np

from config import config
from pipeline.vad import VADProcessor, VadEvent, VadEventType
from pipeline.stt import StreamingSTT
from pipeline.llm import LLMClient
from pipeline.tts import StreamingTTS
from pipeline.audio_io import AudioIO
from agent.backchannel import BackchannelInjector
from agent.filler import FillerManager
from agent.conversation import Conversation
from voice.cache import AudioCache


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceAgent:
    """
    Full-duplex voice agent with backchanneling and filler support.

    Each WebSocket connection gets its own VoiceAgent instance.
    """

    def __init__(
        self,
        websocket,
        tts: StreamingTTS,
        audio_cache: AudioCache,
    ):
        """
        Args:
            websocket: The WebSocket connection (fastapi.WebSocket)
            tts: Shared StreamingTTS instance (model loaded, voice cloned)
            audio_cache: Shared AudioCache (fillers + backchannels pre-generated)
        """
        self.ws = websocket
        self.tts = tts
        self.audio_cache = audio_cache

        # Pipeline components (per-connection)
        self.vad = VADProcessor()
        self.stt = StreamingSTT()
        self.llm = LLMClient()
        self.audio_io = AudioIO()

        # Agent logic
        self.backchannel = BackchannelInjector(audio_cache)
        self.filler = FillerManager(audio_cache, tts)
        self.conversation = Conversation()

        # State
        self.state = State.IDLE
        self._speaking_task: asyncio.Task | None = None
        self._cancel_speaking = asyncio.Event()
        self._running = True

        # Prefix warming task handle
        self._warm_task: asyncio.Task | None = None

    async def run(self):
        """
        Main loop. Reads audio from WebSocket, processes through pipeline,
        sends audio responses back.
        """
        try:
            while self._running:
                # Receive any message type from WebSocket
                message = await self.ws.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                # Handle binary audio data
                if "bytes" in message and message["bytes"]:
                    data = message["bytes"]
                elif "text" in message:
                    # Text message — ignore (could be control message from client)
                    continue
                else:
                    continue

                # Decode and resample
                raw_audio = self.audio_io.decode_ws_audio(data)
                vad_audio = self.audio_io.resample_for_vad(raw_audio)
                stt_audio = self.audio_io.resample_for_stt(raw_audio)

                # Run VAD
                is_agent_speaking = self.state == State.SPEAKING
                event = self.vad.process_chunk(vad_audio, is_agent_speaking)

                # Route event to current state handler
                await self._handle_event(event, stt_audio)

        except Exception as e:
            # WebSocket closed or error — clean up
            import traceback
            print(f"Agent error: {e}")
            traceback.print_exc()
            self._running = False

    async def _handle_event(self, event: VadEvent, stt_audio: np.ndarray):
        """Route a VAD event to the appropriate state handler."""

        if event.type == VadEventType.BARGE_IN:
            await self._handle_barge_in()
            return

        if self.state == State.IDLE:
            await self._handle_idle(event, stt_audio)
        elif self.state == State.LISTENING:
            await self._handle_listening(event, stt_audio)
        elif self.state == State.THINKING:
            await self._handle_thinking(event, stt_audio)
        elif self.state == State.SPEAKING:
            await self._handle_speaking(event)

    # ── State handlers ─────────────────────────────────────────────

    async def _handle_idle(self, event: VadEvent, stt_audio: np.ndarray):
        """IDLE: waiting for user to start talking."""
        if event.type == VadEventType.SPEECH_START:
            self.state = State.LISTENING
            self.stt.reset()
            self.stt.add_audio(stt_audio)
            await self._send_state("listening")

    async def _handle_listening(self, event: VadEvent, stt_audio: np.ndarray):
        """LISTENING: user is talking. Feed STT, inject backchannels."""

        # Always feed audio to STT
        self.stt.add_audio(stt_audio)

        if event.type == VadEventType.SHORT_PAUSE:
            # Backchannel opportunity
            result = self.backchannel.maybe_inject(event.speech_duration_ms)
            if result is not None:
                phrase, audio = result
                await self._send_audio(audio)

        elif event.type == VadEventType.END_OF_TURN:
            # User is done — transition to THINKING
            self.state = State.THINKING
            await self._send_state("thinking")

            # Get final transcript
            transcript = self.stt.finalize()

            if not transcript.text.strip():
                # Empty transcript — go back to idle
                self.state = State.IDLE
                await self._send_state("idle")
                return

            # Record user turn
            self.conversation.add_user_turn(transcript.text)

            # Start filler timer
            self.filler.start()

            # Start generating response (runs concurrently)
            self._speaking_task = asyncio.create_task(
                self._generate_and_speak(transcript.text)
            )

        elif event.type in (VadEventType.SPEECH_ONGOING, VadEventType.SPEECH_START):
            # Still talking — try to get a partial transcript for KV warming
            partial = self.stt.get_partial()
            if partial is not None and partial.text.strip():
                # Fire-and-forget: warm the LLM KV cache with partial text
                if self._warm_task is None or self._warm_task.done():
                    history = self.conversation.get_history()
                    self._warm_task = asyncio.create_task(
                        self.llm.warm_prefix(partial.text, history)
                    )

    async def _handle_thinking(self, event: VadEvent, stt_audio: np.ndarray):
        """THINKING: LLM generating, maybe play filler."""
        # Check if filler should play
        filler_result = self.filler.get_filler_if_needed()
        if filler_result is not None:
            phrase, audio = filler_result
            await self._send_audio(audio)

        # If user starts talking again during THINKING, treat as new turn
        if event.type == VadEventType.SPEECH_START:
            # Cancel the in-progress response
            if self._speaking_task and not self._speaking_task.done():
                self._cancel_speaking.set()
                self._speaking_task.cancel()
            self.filler.reset()
            self.state = State.LISTENING
            self.stt.reset()
            self.stt.add_audio(stt_audio)
            await self._send_state("listening")

    async def _handle_speaking(self, event: VadEvent):
        """SPEAKING: streaming TTS to user. Watch for barge-in."""
        # Barge-in is handled separately in _handle_barge_in
        pass

    async def _handle_barge_in(self):
        """User started talking while agent is speaking — stop immediately."""
        if self._speaking_task and not self._speaking_task.done():
            self._cancel_speaking.set()
            self._speaking_task.cancel()

        self.filler.reset()
        self.state = State.LISTENING
        self.stt.reset()
        await self._send_state("listening")

    # ── Response generation ────────────────────────────────────────

    async def _generate_and_speak(self, user_text: str):
        """
        Generate LLM response and stream TTS audio to the user.

        Handles filler-to-response crossfade on the first chunk.
        """
        self._cancel_speaking.clear()
        history = self.conversation.get_history()
        full_response = ""
        first_chunk = True

        try:
            async for sentence in self.llm.generate_stream(user_text, history):
                if self._cancel_speaking.is_set():
                    return

                full_response += sentence + " "

                # Synthesize this sentence (run sync TTS in executor)
                loop = asyncio.get_event_loop()
                chunks = await loop.run_in_executor(
                    None, lambda s=sentence: list(self.tts.synthesize_stream(s))
                )

                for chunk in chunks:
                    if self._cancel_speaking.is_set():
                        return

                    audio = chunk.audio

                    if first_chunk:
                        # Crossfade filler -> real response
                        audio = self.filler.blend_with_response(audio)
                        self.state = State.SPEAKING
                        await self._send_state("speaking")
                        first_chunk = False

                    await self._send_audio(audio)

            # Done speaking — record assistant turn
            self.conversation.add_assistant_turn(full_response.strip())
            self.filler.reset()
            self.state = State.IDLE
            await self._send_state("idle")

        except asyncio.CancelledError:
            # Barge-in or shutdown — record partial response
            if full_response.strip():
                self.conversation.add_assistant_turn(full_response.strip())

    # ── Audio/message sending ──────────────────────────────────────

    async def _send_audio(self, audio: np.ndarray):
        """Normalize, encode, chunk, and send audio over WebSocket."""
        audio = self.audio_io.normalize_volume(audio)
        chunks = self.audio_io.chunk_audio(audio, self.tts.sample_rate)

        for chunk in chunks:
            if self._cancel_speaking.is_set():
                return
            encoded = self.audio_io.encode_ws_audio(chunk)
            await self.ws.send_bytes(encoded)

    async def _send_state(self, state_name: str):
        """Send a JSON state update to the client (for UI indicators)."""
        try:
            await self.ws.send_json({"type": "state", "state": state_name})
        except Exception:
            pass
