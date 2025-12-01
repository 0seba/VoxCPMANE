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


## API Reference

The full API documentation is available in [docs/API.md](docs/API.md).

## Roadmap

  - [ ] Creating custom voices and automatic prompt caching
  - [ ] Chunked long audio generation

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) - Original TTS model
