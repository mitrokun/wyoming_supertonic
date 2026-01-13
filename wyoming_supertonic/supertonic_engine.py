import logging
import numpy as np
import os
import sys

# Add current directory to path to find helper.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_LOGGER = logging.getLogger(__name__)

class SupertonicEngine:
    def __init__(self, steps: int = 5, speed: float = 1.05, model_path: str = None):
        self.steps = steps
        self.speed = speed
        self.model_path = model_path
        self.tts = None
        self.sample_rate = 44100 
        self.available_voices = []
        
        self._load_style_func = None
        
        # Supported V2 model languages
        self.supported_langs = ["en", "ko", "es", "pt", "fr"]

    def load(self):
        """Load via local helper.py"""
        _LOGGER.info("Loading engine (standalone mode)...")
        
        try:
            from helper import load_text_to_speech, load_voice_style
            self._load_style_func = load_voice_style
            
            # Locate folders
            base_dir = self.model_path if self.model_path else os.getcwd()
            onnx_dir = os.path.join(base_dir, "onnx")
            styles_dir = os.path.join(base_dir, "voice_styles")
            
            # Check alternative paths
            if not os.path.exists(onnx_dir):
                if os.path.exists(os.path.join(base_dir, "assets", "onnx")):
                    onnx_dir = os.path.join(base_dir, "assets", "onnx")
                    styles_dir = os.path.join(base_dir, "assets", "voice_styles")
                else:
                    raise FileNotFoundError(f"Folder 'onnx' not found in {base_dir}")

            _LOGGER.info(f"Loading ONNX models from: {onnx_dir}")
            
            # Load (use_gpu=False)
            self.tts = load_text_to_speech(onnx_dir, use_gpu=False)
            
            if hasattr(self.tts, 'sample_rate'):
                self.sample_rate = self.tts.sample_rate
            
            # Scan voices
            if os.path.exists(styles_dir):
                self.available_voices = sorted([f.replace(".json", "") for f in os.listdir(styles_dir) if f.endswith(".json")])
                self.styles_dir = styles_dir
            else:
                _LOGGER.warning("Styles folder not found")
                self.available_voices = ["M1"]
                self.styles_dir = None

            _LOGGER.info(f"Engine ready. Rate: {self.sample_rate}Hz. Voices: {len(self.available_voices)}")
            
        except ImportError as e:
            raise RuntimeError(f"helper.py not found! {e}")

    def synthesize(self, text: str, voice_name: str, lang_code: str = "en") -> tuple[bytes, int]:
        if self.tts is None:
            raise RuntimeError("Engine not loaded!")

        # 1. Process language
        if not lang_code: lang_code = "en"
        short_lang = lang_code[:2].lower()
        if short_lang not in self.supported_langs:
            short_lang = "en"

        # 2. Load style
        style_path = os.path.join(self.styles_dir, f"{voice_name}.json")
        style = self._load_style_func([style_path])

        # 3. Synthesize via helper
        try:
            _LOGGER.debug(f"Synthesizing: (Voice: {voice_name}, Lang: {short_lang}, Speed: {self.speed}, Steps: {self.steps})")
            
            wav, duration = self.tts(
                text,       
                short_lang, 
                style, 
                self.steps, 
                self.speed
            )
            
            # Remove batch dimensions (1, N) -> (N,)
            wav = wav.squeeze()
            
        except Exception as e:
            _LOGGER.error(f"Synthesis error in helper: {e}")
            raise e

        # 4. Simple conversion to int16 without trimming
        audio_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
        
        return audio_int16.tobytes(), self.sample_rate