# Wyoming [Supertonic](https://github.com/supertone-inc/supertonic)

Wyoming server for Supertonic TTS (V3).

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/mitrokun/wyoming_supertonic.git
cd wyoming_supertonic

python3 -m venv venv
source venv/bin/activate
pip install supertonic wyoming sentence-stream num2words onnxruntime numpy
```


## Usage

Run the server pointing to your model directory:

```bash
python3 -m wyoming_supertonic --uri 'tcp://0.0.0.0:10209'
```

### Arguments

*   `--language` Default voice language (default: `en`).      
*   `--uri`: Server URI (default: `tcp://0.0.0.0:10209`).
*   `--speed`: Speech speed, 0.5 to 2.0 (default: `1.0`).
*   `--steps`: Denoising steps. Higher is better quality but slower (default: `5`).
*   `--threads`: Number of CPU threads to use (default: `4`).
*   `--no-streaming`: Disable sentence-by-sentence streaming.
*   `--debug`: Enable debug logging.

The initial synthesized audio for each sentence is framed by silence on both sides. I chose a default trim value of 300ms; if the pauses between sentences are unsatisfactory for you, try adjusting `--crop-silence` parameter.

### Supported Languages:
| Code | Language | Code | Language | Code | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ar` | Arabic | `fr` | French | `pt` | Portuguese |
| `bg` | Bulgarian | `hi` | Hindi | `ro` | Romanian |
| `cs` | Czech | `hr` | Croatian | `ru` | Russian |
| `da` | Danish | `hu` | Hungarian | `sk` | Slovak |
| `de` | German | `id` | Indonesian | `sl` | Slovenian |
| `el` | Greek | `it` | Italian | `sv` | Swedish |
| `en` | English | `ja` | Japanese | `tr` | Turkish |
| `es` | Spanish | `ko` | Korean | `uk` | Ukrainian |
| `et` | Estonian | `lt` | Lithuanian | `vi` | Vietnamese |
| `fi` | Finnish | `lv` | Latvian | `nl` | Dutch |

## Quick start with uv

```
git clone https://github.com/mitrokun/wyoming_supertonic.git
cd wyoming_supertonic
uv --cache-dir .uv_cache run -m wyoming_supertonic --uri 'tcp://0.0.0.0:10209'
```
for auto-stress for the Russian language, add the extra parameter `--extra ru` (additional libraries will be downloaded).

## Run using Docker

```
git clone https://github.com/mitrokun/wyoming_supertonic.git
cd wyoming_supertonic
sudo docker build . -t "wyoming-supertonic:latest" --build-arg WITH_RU=false # for auto-stress for the Russian language, set this to true
mkdir data # models will be downloaded here
sudo docker compose up
```
Edit [docker-compose-yml](docker-compose.yml) as needed.

---

### Tip: Multi-Language Setup

Due to a [limitation](https://github.com/OHF-Voice/wyoming/blob/main/wyoming/tts.py#L39) of the Wyoming standard library, passing the `voice` parameter disables the `language` parameter.

If you are setting up a multi-language assistant with different voice pipelines and need dynamic language switching, it is recommended to use this alternative client: **[Streaming tts proxy](https://github.com/mitrokun/streaming_tts_proxy)**

It automatically forwards the language of the currently active voice assistant pipeline, allowing the Wyoming-Supertonic server to dynamically apply the correct language normalizer and synthesis settings on the fly. The server-side fix has already been implemented.

