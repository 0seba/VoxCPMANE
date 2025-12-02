# VoxCPMANE

[VoxCPM TTS](https://github.com/OpenBMB/VoxCPM) model with Apple Neural Engine (ANE) backend server. CoreML models available in [Huggingface repository](https://huggingface.co/seba/VoxCPM-ANE).


- 🎤 **Voice Cloning**: Support for custom voice prompts and cached voices
- 📡 **Streaming Support**: Real-time audio streaming for low latency
- 🎧 **Server-side Playback**: Direct audio playback on the server
- 🌐 **Web Interface**: Interactive playground for testing

## Voice Cloning

https://github.com/user-attachments/assets/02ffa400-b2fd-422e-a3ad-a0ea232a55aa

## Included Voices [Listen samples](https://gregr.org/tts-samples/)


https://github.com/user-attachments/assets/28880ed2-2e21-4eb4-b0ce-18a100403e87


## Installation

### Prerequisites

- macOS with Apple Silicon for ANE acceleration
- Python 3.9-3.12
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- `pydub` required for audio formats other than `wav` in `/speech` endpoint

### Install with `pip` or `uv`

```bash
uv pip install voxcpmane
```

```bash
pip install voxcpmane
```

The server will start on `http://localhost:8000` by default. You can access the web playground at the root URL.

## Configuration

### Command Line Options

```bash
uv run voxcpmane-server --help
```

- `--host`: Host to bind the server to (default: `0.0.0.0`)
- `--port`: Port to run the server on (default: `8000`)
- `--cache-dir`: Directory for custom voice caches (default: `~/.cache/ane_tts`)

## Custom Voices

You can create reusable cached voices in two ways:

1.  **Via the Web Playground/API**: Use the "Create Voice" tab or `POST /v1/voices` endpoint.
2.  **Startup Compilation**: Place pairs of audio files (e.g., `.wav`, `.mp3`) and transcriptions (`.txt`) in the custom cache directory. The server will automatically compile them into voice caches (`.npy`) on startup.

Example:
If you place `myvoice.mp3` and `myvoice.txt` in the cache directory, the server will generate `myvoice.npy` on start, making "myvoice" available for generation.

## API Reference

The full API documentation is available in [docs/API.md](docs/API.md).

## Changelog

### Version 0.0.3
- Added support for creation of custom voices

## Roadmap
  - [ ] Automatic prompt caching
  - [ ] Chunked long audio generation
  - [x] Custom voices

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) - Original TTS model
