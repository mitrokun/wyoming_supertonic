import importlib
import inspect
import logging
from typing import Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Registry of available normalizers: "lang_code": ("module_name", "ClassName")
# To add a new language, simply add a new line to this dictionary.
NORMALIZERS_REGISTRY = {
    "ru": (".ru_norm", "RussianTextNormalizer"),
    "de": (".de_norm", "GermanTextNormalizer"),
}


class SupertonicEngine:
    """
    Main TTS engine class for Supertone V3.
    """

    def __init__(
        self,
        steps: int = 5,
        speed: float = 1.0,
        model_path: str = None,
        crop_silence_ms: int = 300,
    ) -> None:
        """
        Initialize engine settings.
        
        Args:
            steps: Denoising steps (3-5 for speed, 10-20 for quality).
            speed: Speech rate multiplier.
            model_path: Optional path for models.
            crop_silence_ms: Silence to remove from both ends.
        """
        self.steps = steps
        self.speed = speed
        self.model_path = model_path
        self.crop_silence_ms = crop_silence_ms
        self.tts = None
        self.sample_rate = 44100

        # Lazy cache for voice style objects (loaded on demand)
        self._style_cache = {}

        # Official V3 voices
        self.available_voices = [
            "M1", "M2", "M3", "M4", "M5",
            "F1", "F2", "F3", "F4", "F5"
        ]

        # Supported languages
        self.supported_langs = [
            "en", "ko", "ja", "ar", "bg", "cs", "da", "de",
            "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
            "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
            "ru", "sk", "sl", "sv", "tr", "uk", "vi"
        ]

        # Register normalizers dynamically
        self.normalizers = {}
        self._load_normalizers()

    def _load_normalizers(self) -> None:
        """
        Dynamically loads available text normalizers based on the registry.
        Fails safely if a module's dependencies are missing.
        """
        for lang, (module_name, class_name) in NORMALIZERS_REGISTRY.items():
            try:
                module = importlib.import_module(module_name, package=__package__)
                normalizer_class = getattr(module, class_name)
                self.normalizers[lang] = normalizer_class()
                _LOGGER.debug("[%s] Text normalizer loaded successfully.", lang)
            except ImportError as exc:
                _LOGGER.warning(
                    "[%s] Text normalizer won't be available (missing dependency): %s", 
                    lang, exc
                )
            except Exception as exc:
                _LOGGER.error(
                    "[%s] Failed to initialize normalizer: %s", 
                    lang, exc
                )

    def load(self) -> None:
        """
        Initializes the TTS library. 
        Models are auto-downloaded if missing.
        """
        _LOGGER.info("Loading engine (Supertonic V3)...")
        try:
            from supertonic import TTS
            # The library handles HF model caching automatically
            self.tts = TTS(model_dir=self.model_path, auto_download=True)
            _LOGGER.info(
                "Engine loaded. Sample Rate: %dHz. Lazy caching enabled.",
                self.sample_rate
            )
        except ImportError as exc:
            raise RuntimeError(
                "supertonic package not found! Please run 'pip install supertonic'."
            ) from exc

    def _sanitize_text(self, text: str) -> str:
        """
        Removes hidden control characters (like soft hyphen \xad)
        that cause synthesis errors.
        """
        # \xad: Soft Hyphen, \u200b: Zero Width Space, \ufeff: BOM
        bad_chars = ["\xad", "\u200b", "\ufeff", "\u200c", "\u200d"]
        for char in bad_chars:
            text = text.replace(char, "")
        
        # Cleanup whitespace
        return " ".join(text.split())

    def synthesize(
        self, text: str, voice_name: str, lang_code: str = "en"
    ) -> Tuple[bytes, int]:
        """
        Synthesize text into PCM audio.
        """
        if self.tts is None:
            raise RuntimeError("Engine not loaded! Call load() first.")

        # 0. Global sanitization
        text = self._sanitize_text(text)

        # 1. Language processing
        if not lang_code:
            lang_code = "en"
        short_lang = lang_code[:2].lower()
        if short_lang not in self.supported_langs:
            short_lang = "en"

        # 2. Text Normalization
        if short_lang in self.normalizers:
            original_text = text
            text = self.normalizers[short_lang].normalize(text)
            
            # Log the prepared text so we can see the result of the normalizer
            if original_text != text:
                _LOGGER.debug(
                    "[%s] Synth: %s", 
                    short_lang, text
                )

        # 3. Resolve Voice Style (Lazy Caching)
        if voice_name not in self._style_cache:
            try:
                # Load style from disk once
                style = self.tts.get_voice_style(voice_name=voice_name)
                self._style_cache[voice_name] = style
                _LOGGER.debug("Style '%s' added to cache.", voice_name)
            except Exception as exc:
                _LOGGER.warning(
                    "Style '%s' not found, falling back to 'M1'. (%s)",
                    voice_name, exc
                )
                # Ensure M1 is cached for future fallbacks
                if "M1" not in self._style_cache:
                    self._style_cache["M1"] = self.tts.get_voice_style(
                        voice_name="M1"
                    )
                style = self._style_cache["M1"]
        else:
            style = self._style_cache[voice_name]

        # 4. Perform Synthesis
        try:
            kwargs = {"lang": short_lang}
            sig = inspect.signature(self.tts.synthesize)

            # Safely handle library API parameters
            if "speed" in sig.parameters:
                kwargs["speed"] = self.speed
            if "total_step" in sig.parameters:
                kwargs["total_step"] = self.steps
            elif "total_steps" in sig.parameters:
                kwargs["total_steps"] = self.steps

            wav, _ = self.tts.synthesize(text, voice_style=style, **kwargs)

            # Squeeze (1, N) -> (N,)
            if hasattr(wav, "squeeze"):
                wav = wav.squeeze()

            # 5. Silence Cropping
            if self.crop_silence_ms > 0:
                crop_samples = int(
                    (self.crop_silence_ms / 1000.0) * self.sample_rate
                )

                if len(wav) > crop_samples * 2:
                    wav = wav[crop_samples : -crop_samples]
                else:
                    _LOGGER.warning(
                        "Audio too short (%d samples) for cropping.", len(wav)
                    )

        except Exception as exc:
            _LOGGER.error("In-library synthesis failure: %s", exc)
            raise

        # 6. Convert to int16 PCM (Wyoming requirement)
        audio_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)

        return audio_int16.tobytes(), self.sample_rate