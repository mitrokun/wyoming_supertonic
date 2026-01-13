import argparse
import asyncio
import logging
import json

from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .supertonic_engine import SupertonicEngine

_LOGGER = logging.getLogger(__name__)

class SupertonicEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        engine: SupertonicEngine,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reader=reader, writer=writer, *args, **kwargs)
        
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.engine = engine
        
        self.sbd = SentenceBoundaryDetector()
        self._is_streaming = False
        self._audio_started = False
        self._sentence_buffer = ""
        self._current_voice = None 
        self._current_language = self.cli_args.language


    async def run_raw(self) -> None:
        """Reads raw data from the socket and logs it before processing."""
        try:
            while True:
                header_line = await self.reader.readline()
                if not header_line:
                    _LOGGER.debug("Client disconnected")
                    break
                
                header_line_str = header_line.decode().strip()
                _LOGGER.debug("Raw header received: %s", header_line_str)

                if not header_line_str:
                    continue

                header_dict = json.loads(header_line_str)
                data_length = header_dict.get("data_length")
                
                data_dict = header_dict.get("data", {})

                if data_length and data_length > 0:
                    data_bytes = await self.reader.readexactly(data_length)
                    data_bytes_str = data_bytes.decode().strip()
                    _LOGGER.debug("Raw data payload received: %s", data_bytes_str)
                    data_dict.update(json.loads(data_bytes_str))
                
                event_type = header_dict["type"]
                event = Event(type=event_type, data=data_dict)
                
                if not (await self.handle_event(event)):
                    break

        except (ConnectionResetError, asyncio.IncompleteReadError) as e:
            _LOGGER.debug(f"Connection closed: {e}")
        except Exception:
            _LOGGER.exception("Unexpected error in raw handler loop")
        finally:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            # --- 1. Synthesize (Single) ---
            if Synthesize.is_type(event.type):
                if self._is_streaming: return True
                
                syn = Synthesize.from_event(event)
                voice_data = event.data.get("voice", {})
                lang = voice_data.get("language")

                if syn.voice and syn.voice.name: self._current_voice = syn.voice.name
                
                if lang: self._current_language = lang
                
                return await self._handle_synthesize_full(syn.text)

            # --- 2. Start Streaming ---
            if SynthesizeStart.is_type(event.type):
                if self.cli_args.no_streaming: return True

                start = SynthesizeStart.from_event(event)
                voice_data = event.data.get("voice", {})
                lang = voice_data.get("language")
                
                self._is_streaming = True
                self._audio_started = False
                self.sbd = SentenceBoundaryDetector()
                self._sentence_buffer = ""
                
                if start.voice and start.voice.name:
                    self._current_voice = start.voice.name

                if lang:
                    self._current_language = lang
                
                _LOGGER.debug(f"Stream started. Voice: {self._current_voice}, Language: {self._current_language}")
                return True

            # --- 3. Chunk ---
            if SynthesizeChunk.is_type(event.type):
                if not self._is_streaming: return True
                chunk = SynthesizeChunk.from_event(event)
                for sentence in self.sbd.add_chunk(chunk.text):
                    await self._process_sentence(sentence)
                return True

            # --- 4. Stop ---
            if SynthesizeStop.is_type(event.type):
                if not self._is_streaming: return True
                remaining = self.sbd.finish()
                if remaining:
                    await self._process_sentence(remaining)
                await self._flush_buffer()
                
                if self._audio_started:
                    await self.write_event(AudioStop().event())
                await self.write_event(SynthesizeStopped().event())
                self._is_streaming = False
                return True

        except Exception as err:
            _LOGGER.exception("Handler error")
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            self._is_streaming = False

        return True

    async def _handle_synthesize_full(self, text: str) -> bool:
        self._audio_started = False
        self._sentence_buffer = ""
        self.sbd = SentenceBoundaryDetector()
        
        sentences = list(self.sbd.add_chunk(text))
        remaining = self.sbd.finish()
        if remaining: sentences.append(remaining)
        
        for s in sentences:
            await self._process_sentence(s)
        await self._flush_buffer()
        
        if self._audio_started:
            await self.write_event(AudioStop().event())
        return True

    async def _process_sentence(self, sentence: str):
        s = sentence.strip()
        if not s: return
        
        if self._sentence_buffer:
            self._sentence_buffer += " " + s
        else:
            self._sentence_buffer = s

        if len(self._sentence_buffer) >= 20:
            await self._flush_buffer()

    async def _flush_buffer(self):
        text = self._sentence_buffer.strip()
        self._sentence_buffer = ""
        if not text: return

        voice_name = "M1"
        if self._current_voice and self._current_voice in self.engine.available_voices:
             voice_name = self._current_voice
        elif self.engine.available_voices:
             voice_name = self.engine.available_voices[0]

        _LOGGER.debug(f"Requesting synthesis for: '{text[:40]}...'")

        loop = asyncio.get_running_loop()
        try:
            pcm_bytes, rate = await loop.run_in_executor(
                None, self.engine.synthesize, text, voice_name, self._current_language
            )
        except Exception as e:
            _LOGGER.error(f"Engine error: {e}")
            return

        if not self._audio_started:
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
            self._audio_started = True

        chunk_size = 2048
        for i in range(0, len(pcm_bytes), chunk_size):
            await self.write_event(
                AudioChunk(audio=pcm_bytes[i : i + chunk_size], rate=rate, width=2, channels=1).event()
            )