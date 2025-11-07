# VoxCPMANE

[VoxCPM TTS](https://github.com/OpenBMB/VoxCPM) model with Apple Neural Engine (ANE) backend server. CoreML models available in [Huggingface repository](https://github.com/0seba/VoxCPMANE).


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

The server provides OpenAI-compatible endpoints for text-to-speech generation.

### Base URL

```
http://localhost:8000
``` 

### Request Model

All TTS endpoints accept the following request parameters:

```json
{
  "model": "voxcpm-0.5b",           // Model identifier (fixed)
  "input": "Text to synthesize",     // Required: Text to generate speech for
  "voice": "voice_name",            // Optional: Use cached voice
  "prompt_wav_path": "/path/to/audio.wav",  // Optional: Path to prompt audio file
  "prompt_text": "Transcription of prompt audio",  // Optional: Text matching prompt audio
  "response_format": "wav",         // Optional: Audio format (wav, mp3, flac, opus, aac, pcm)
  "max_length": 2048,               // Optional: Max generated sequence length (1-2048)
  "cfg_value": 2.0,                 // Optional: Classifier-free guidance (0.0-10.0)
  "inference_timesteps": 10         // Optional: Diffusion steps (1-100)
}
```

### Voice Selection

You have two options for voice control:

1. **Cached Voices**: Use pre-computed voice embeddings
   - Set `voice` parameter to a cached voice name
   - Available voices can be listed via `/voices` endpoint
   - Ignores `prompt_wav_path` and `prompt_text` parameters

2. **Custom Voice Cloning**: Provide your own audio prompt
   - Set `prompt_wav_path` to the path of local WAV file
   - Set `prompt_text` to the exact transcription of the audio
   - If `prompt_wav_path` is empty, generates with random voice

### Parameters

- **max_length**: Controls maximum generated audio length (each unit ≈ 0.04 seconds)
- **cfg_value**: Classifier-free guidance strength.
- **inference_timesteps**: Number of diffusion steps, defaults to 10.

## Endpoints

### 1. Generate Speech (File)

**POST** `/v1/audio/speech`

Generates a complete audio file and returns it for download.

**Request:**
```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm-0.5b",
    "input": "Hello, this is a test of the VoxCPM TTS system.",
    "response_format": "wav"
  }'
```

**Response:** Binary audio file with appropriate `Content-Type` header

**Supported formats:** `wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`

### 2. Stream Speech (Real-time)

**POST** `/v1/audio/speech/stream`

Streams audio chunks in real-time for low-latency playback.

**Request:**
```bash
curl -X POST "http://localhost:8000/v1/audio/speech/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: application/octet-stream" \
  -d '{
    "model": "voxcpm-0.5b",
    "input": "This speech will be streamed in real-time.",
    "response_format": "pcm"
  }'
```

**Response:** Streaming binary audio data (16-bit PCM, 16kHz)

**Headers:**
- `X-Sample-Rate`: Sample rate of the audio (typically 16000)

### 3. Play on Server

**POST** `/v1/audio/speech/playback`

Generates speech and plays it directly on the server with progress indicators.

**Request:**
```json
{
  "model": "voxcpm-0.5b",
  "input": "This will play on the server"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Audio playback completed on server",
  "duration_seconds": 5.23,
}
```

### 4. Cancel Generation

**POST** `/v1/audio/speech/cancel`

Cancels the currently running audio generation.

**Request:**
```bash
curl -X POST "http://localhost:8000/v1/audio/speech/cancel"
```

**Response:**
```json
{
  "status": "success",
  "message": "Cancellation signal sent to Job 123"
}
```

### 5. List Available Voices

**GET** `/voices`

Returns a list of available cached voice names.

**Request:**
```bash
curl -X GET "http://localhost:8000/voices"
```

**Response:**
```json
{
  "voices": ["voice1", "voice2", "voice3"],
  "count": 3,
  "cache_directory": "assets/caches"
}
```

### 6. Health Check

**GET** `/health`

Returns server status and current processing state.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "is_processing": true,
  "current_job_id": 123,
  "queue_pending": false,
  "model": "voxcpm-0.5b"
}
```

### 7. Web Playground

**GET** `/`

Interactive web interface for testing the TTS functionality.

Access at: `http://localhost:{PORT}`

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) - Original TTS model
