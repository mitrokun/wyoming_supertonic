# Wyoming [Supertonic](https://github.com/supertone-inc/supertonic)

Wyoming server for Supertonic TTS (V2).

### Model Setup

You must download the model files manually.

1. Download the contents of the `assets` folder from the [Supertonic HuggingFace repository](https://huggingface.co/Supertone/supertonic-2/tree/main).
2. Organize them in a directory (e.g., `/home/username/supertonic-data`) exactly as shown below:

```text
supertonic-data/
├── onnx/
│   ├── duration_predictor.onnx
│   ├── text_encoder.onnx
│   ├── tts.json
│   ├── unicode_indexer.json
│   ├── vector_estimator.onnx
│   └── vocoder.onnx
└── voice_styles/
    ├── F1.json
    ├── M1.json
    └── ...
```

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/mitrokun/wyoming_supertonic.git
cd wyoming_supertonic

python3 -m venv venv
source venv/bin/activate
pip install wyoming sentence-stream numpy onnxruntime
```

## Usage

Run the server pointing to your model directory:

```bash
python3 -m wyoming_supertonic --data-dir ~/supertonic-data --uri 'tcp://0.0.0.0:10209'
```

### Arguments

*   `--data-dir`: **Required**. Path to the folder containing `onnx` and `voice_styles` directories.
*   `--uri`: Server URI (default: `tcp://0.0.0.0:10209`).
*   `--speed`: Speech speed, 0.5 to 2.0 (default: `1.0`).
*   `--steps`: Denoising steps. Higher is better quality but slower (default: `5`).
*   `--threads`: Number of CPU threads to use (default: `4`).
*   `--no-streaming`: Disable sentence-by-sentence streaming.
*   `--debug`: Enable debug logging.

## Quick start with uv

```
git clone https://github.com/mitrokun/wyoming_supertonic.git
cd wyoming_supertonic
UV_CACHE_DIR=.uv_cache uv run -m wyoming_supertonic --data-dir ~/supertonic-data --uri 'tcp://0.0.0.0:10209'
```
