# VoxCPMANE2

VoxCPMANE2 is the VoxCPM2 version of [VoxCPMANE](../VoxCPMANE). It provides a
pure numpy/CoreML runtime and FastAPI HTTP server for running VoxCPM2 TTS on
Apple Silicon with Apple Neural Engine acceleration.

CoreML model assets are downloaded from
[seba/VoxCPM2-ANE](https://huggingface.co/seba/VoxCPM2-ANE) by default.

- VoxCPM2 text-to-speech generation
- OpenAI-compatible `/v1/audio/speech` endpoint
- Streaming audio generation
- Server-side playback
- Web playground at `/`
- Custom cached voices

## Requirements

- macOS on Apple Silicon
- Python `>=3.10,<3.13`
- `uv` or `pip`
- CoreML runtime support through `coremltools`
- Optional: `pydub` for `mp3`, `opus`, `ogg`, and `aac` responses

## Installation

From this directory:

```bash
uv sync
```

For editable development:

```bash
uv pip install -e .
```

## Run The Server

```bash
uv run voxcpmane-server
```

The server starts on `http://localhost:8000` by default. Open the root URL for
the web playground.

Common options:

```bash
uv run voxcpmane-server \
  --host 0.0.0.0 \
  --port 8000 \
  --model-dir /path/to/VoxCPM2-ANE \
  --cache-dir ~/.cache/ane_tts
```

If `--model-dir` is omitted, the server downloads the CoreML model directory
from Hugging Face.

## Working Modes

`--lm-mode` controls how multifunction LM prefill and decode handles are kept in
memory. The default prefill length is `128`. Available prefill lengths are `1`,
`8`, `16`, `32`, `64`, and `128`; any of these can be used with
`single-length` mode.

| Mode | Behavior | Tradeoff |
| --- | --- | --- |
| `hot-swap` | Keeps the prefill function loaded while idle, loads length `1` for decode, unloads inactive functions, then swaps back after generation. | Default. Lower idle memory, with function load/unload cost around generation. |
| `preload` | Preloads length `1` and the selected prefill size at startup, keeps decode resident, unloads prefill during decode, reloads prefill when idle. | Avoids cold decode load while still reducing decode-time memory. |
| `always-loaded` | Preloads length `1` and the selected prefill size and never unloads either function. | Fastest transitions, highest memory use. |
| `single-length` | Uses only the selected prefill length and restricts LM calls to that function. | Simplest resident set; decode also uses the selected length instead of length `1`. |

Examples:

```bash
# Default hot-swap behavior with prefill length 128.
uv run voxcpmane-server --lm-mode hot-swap

# Keep both prefill and decode functions resident.
uv run voxcpmane-server --lm-mode always-loaded

# Preload decode and prefill, but unload prefill during decode.
uv run voxcpmane-server --lm-mode preload

# Use only one LM function length.
uv run voxcpmane-server --lm-prefill-chunk-size 128 --lm-mode single-length
```

## Model Path Options

You can point the server at a complete model directory:

```bash
uv run voxcpmane-server --model-dir /path/to/VoxCPM2-ANE
```

Or override individual CoreML packages:

```bash
uv run voxcpmane-server \
  --base-lm-path /path/to/base_lm_s4_part0_of_2.mlpackage /path/to/base_lm_s4_part1_of_2.mlpackage \
  --residual-lm-path /path/to/residual_lm_fused_s4.mlpackage \
  --locdit-path /path/to/locdit_p4_c4.mlpackage \
  --vae-encoder-path /path/to/audio_vae_encoder.mlpackage \
  --feat-encoder-path /path/to/feat_encoder.mlpackage \
  --vae-decoder-path /path/to/audio_vae_decoder_lf4.mlpackage \
  --fsq-path /path/to/fsq_s4.mlpackage \
  --projections-path /path/to/projections.mlpackage
```

Use `--compile-and-save` to compile `.mlpackage` directories into sibling
`.mlmodelc` directories when compiled versions are missing.

## API

### Generate Full Audio

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2",
    "input": "Hello from VoxCPM2 on Apple Neural Engine.",
    "response_format": "wav",
    "max_length": 2048,
    "cfg_value": 2.0,
    "inference_timesteps": 10
  }' \
  --output speech.wav
```

Supported `response_format` values are `wav`, `flac`, `mp3`, `opus`, `ogg`,
and `aac`. Non-`wav`/`flac` formats require `pydub`.

### Stream Raw PCM16 Audio

```bash
curl http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"model":"voxcpm2","input":"Streaming speech."}' \
  --output stream.pcm
```

The stream response is raw PCM16 at the sample rate exposed in the
`X-Sample-Rate` header.

### Other Endpoints

- `GET /`: web playground
- `GET /health`: server status
- `GET /voices`: available cached voices
- `POST /v1/voices`: create a cached custom voice
- `POST /v1/audio/speech/playback`: generate and play on the server audio device
- `POST /v1/audio/speech/cancel`: cancel the current job

## Custom Voices

Custom voices are stored in `--cache-dir` (`~/.cache/ane_tts` by default).

You can create a voice through the web UI or API:

```bash
curl http://localhost:8000/v1/voices \
  -H "Content-Type: application/json" \
  -d '{
    "voice_name": "myvoice",
    "prompt_wav_path": "/path/to/reference.wav",
    "prompt_text": "The exact transcript of the reference audio.",
    "replace": false
  }'
```

You can also place matching audio and transcript files in the cache directory,
for example `myvoice.wav` and `myvoice.txt`. On startup, the server compiles
missing `.npy` voice caches automatically.

## Metrics And Tuning

Use `--live-rtf` to print real-time-factor metrics:

```bash
uv run voxcpmane-server --live-rtf live
uv run voxcpmane-server --live-rtf final
```

VAE streaming latency can be tuned with:

- `--vae-early-decode-steps`: number of initial AR steps decoded immediately
- `--vae-batch-decode-steps`: number of AR steps to batch after the early phase

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) for the original VoxCPM model family
- [VoxCPMANE](../VoxCPMANE) for the earlier Apple Neural Engine server/runtime
