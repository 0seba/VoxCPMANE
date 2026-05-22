# VoxCPMANE2

Install the VoxCPM2 package:

```bash
uv pip install -U voxcpmane2
```

VoxCPMANE2 is the VoxCPM2 version of [VoxCPMANE](../VoxCPMANE). It provides a
pure numpy/CoreML runtime and FastAPI HTTP server for running VoxCPM2 TTS on
Apple Silicon with Apple Neural Engine acceleration.

CoreML model assets are downloaded from
[seba/VoxCPM2ANE-Preview](https://huggingface.co/seba/VoxCPM2ANE-Preview)
by default.

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

Install with `uv`:

```bash
uv pip install -U voxcpmane2
```

Or install with `pip`:

```bash
pip install -U voxcpmane2
```

For editable development from a source checkout, run `uv pip install -e .` from
this directory.

## Run The Server

```bash
voxcpmane2-server
```

The server starts on `http://localhost:8000` by default. Open the root URL for
the web playground.

Common options:

```bash
voxcpmane2-server \
  --host 0.0.0.0 \
  --port 8000 \
  --repo-id seba/VoxCPM2ANE-Preview \
  --cache-dir ~/.cache/ane_tts
```

If `--model-dir` is omitted, the server downloads the CoreML model directory
from `--repo-id`. If individual package paths are not supplied, components are
loaded from that downloaded directory. The default repo layout includes
`config.json`, `embed_tokens.npy`, a small `.mlpackage` marker for CoreML repo
recognition, and the runtime packages:
`base_lm_multifunction.mlmodelc`, `residual_lm_fused_multifunction.mlmodelc`,
and the compiled component packages at the repo root. Included voice caches live
under `caches/`.

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
voxcpmane2-server --lm-mode hot-swap

# Keep both prefill and decode functions resident.
voxcpmane2-server --lm-mode always-loaded

# Preload decode and prefill, but unload prefill during decode.
voxcpmane2-server --lm-mode preload

# Use only one LM function length.
voxcpmane2-server --lm-prefill-chunk-size 128 --lm-mode single-length
```

## Model Path Options

You can point the server at a complete model directory:

```bash
voxcpmane2-server --model-dir /path/to/VoxCPM2-ANE
voxcpmane2-server --repo-id seba/VoxCPM2ANE-Preview
```

Or override individual CoreML packages:

```bash
voxcpmane2-server \
  --base-lm-path /path/to/base_lm_multifunction.mlpackage \
  --residual-lm-path /path/to/residual_lm_fused_multifunction.mlpackage \
  --locdit-path /path/to/locdit_p4_c4.mlpackage \
  --vae-encoder-path /path/to/audio_vae_encoder.mlpackage \
  --feat-encoder-path /path/to/feat_encoder.mlpackage \
  --vae-decoder-path /path/to/audio_vae_decoder_lf4.mlpackage \
  --fsq-path /path/to/fsq_s4.mlpackage \
  --projections-path /path/to/projections.mlpackage
```

Use `--compile-and-save` to compile `.mlpackage` directories into sibling
`.mlmodelc` directories when compiled versions are missing.

When running with local package paths, included voices are loaded from
`--included-voice-cache-dir` if provided, then from `<model-dir>/caches` if it
exists. If neither is available, the server downloads only `caches/*` from
`--repo-id`, so the bundled voices still appear without downloading the model
packages again.

```bash
voxcpmane2-server \
  --model-dir /path/to/local-models \
  --included-voice-cache-dir /path/to/local-models/caches
```

## API

### Generate Full Audio

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2",
    "input": "Hello from VoxCPM2 on Apple Neural Engine.",
    "voice": "af_alloy",
    "voice_mode": "reference",
    "response_format": "wav",
    "max_length": 2048,
    "cfg_value": 2.0,
    "inference_timesteps": 10
  }' \
  --output speech.wav
```

When `voice` is set, `voice_mode` controls preset voice conditioning:
`reference` uses the cached reference audio only and has lower first-byte
latency; `reference_plus_prompt` uses the cached reference voice plus a supplied
`prompt_wav_path` and matching `prompt_text`; `high_similarity` uses cached
prompt embeddings, transcript, and VAE decoder warmup context when available.

Supported `response_format` values are `wav`, `flac`, `mp3`, `opus`, `ogg`,
and `aac`. Non-`wav`/`flac` formats require `pydub`.

`max_length` is bounded by the available LM KV cache after prompt prefill. If
the generated length exceeds the cache capacity, the server caps generation to
the remaining cache length.

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
- `DELETE /v1/voices/{voice_name}`: delete a cached custom voice
- `POST /v1/audio/speech/playback`: generate and play on the server audio device
- `POST /v1/audio/speech/cancel`: cancel the current job

## Custom Voices

Included voices are stored in the model `caches/` directory or the directory
provided with `--included-voice-cache-dir`. Custom voices created at runtime are
stored in `--cache-dir` (`~/.cache/ane_tts` by default).

Voice caches use feature-encoder outputs, not full VAE encoder latents:

- `name.embed.npy`: reference voice embeddings
- `name.prompt.embed.npy`: optional continuation prompt embeddings
- `name.prompt.cond.npy`: optional final prompt VAE patch used to seed
  high-similarity continuation decoding
- `name.prompt.decode_context.npy`: optional tail prompt VAE patches used to
  warm the streaming VAE decoder for high-similarity audio continuity

Included voices may also ship LM prefix KV caches as `caches/name.lm_prefix.npz`.
Matching caches restore the base/residual LM prefix on the first request. Missing
or custom voices build a local copy under `--cache-dir` on first use.

Old VAE-latent voice caches are not migrated at startup. A valid cache file is
`(T, hidden_size)` feature-encoder output; if an older cache shape is present,
delete and recreate that custom voice.

You can create a voice through the web UI or API:

```bash
curl http://localhost:8000/v1/voices \
  -H "Content-Type: application/json" \
  -d '{
    "voice_name": "myvoice",
    "reference_wav_path": "/path/to/reference.wav",
    "replace": false
  }'
```

For higher-similarity continuation cloning, include the exact transcript of the
same audio. The server then caches feature embeddings for both the VoxCPM2
reference and the prompt continuation:

```bash
curl http://localhost:8000/v1/voices \
  -H "Content-Type: application/json" \
  -d '{
    "voice_name": "myvoice",
    "reference_wav_path": "/path/to/reference.wav",
    "prompt_text": "The exact transcript of the reference audio.",
    "replace": true
  }'
```

A transcript is optional for VoxCPM2 reference-only cloning, but required for
prompt-continuation cloning.

## Metrics And Tuning

Use `--live-rtf` to print real-time-factor metrics:

```bash
voxcpmane2-server --live-rtf live
voxcpmane2-server --live-rtf final
```

VAE streaming latency can be tuned with:

- `--vae-early-decode-steps`: number of initial AR steps decoded immediately
- `--vae-batch-decode-steps`: number of AR steps to batch after the early phase

Defaults are `--vae-early-decode-steps 16` and `--vae-batch-decode-steps 4`.

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) for the original VoxCPM model family
- [VoxCPMANE](../VoxCPMANE) for the earlier Apple Neural Engine server/runtime
