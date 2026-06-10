# VoxCPMANE2

Install the VoxCPM2 package:

```bash
uv tool install --python '>=3.10,<3.13' voxcpmane2
```

VoxCPMANE2 is the VoxCPM2 version of [VoxCPMANE](../VoxCPMANE). It provides a
pure numpy/CoreML runtime and FastAPI HTTP server for running VoxCPM2 TTS on
Apple Silicon with Apple Neural Engine acceleration.

The package includes a browser-based playground at `http://localhost:8000/` for
trying voices, streaming, playback, and custom voice creation without writing
client code.

CoreML model assets are downloaded from
[seba/VoxCPM2ANE-Preview](https://huggingface.co/seba/VoxCPM2ANE-Preview)
by default.

- VoxCPM2 text-to-speech generation
- OpenAI-compatible `/v1/audio/speech` endpoint
- Streaming audio generation
- Server-side playback
- Web playground for generation, voice management, streaming, and playback
- Custom cached voices

## Requirements

- macOS on Apple Silicon
- Python `>=3.10,<3.13` is required
- `uv` or `pip`
- CoreML runtime support through `coremltools`
- Optional: `pydub` for `mp3`, `opus`, `ogg`, and `aac` responses

## Installation

Install as a `uv` tool:

```bash
uv tool install --python '>=3.10,<3.13' voxcpmane2
```

Or install into an environment with `uv pip`:

```bash
uv pip install --python '>=3.10,<3.13' -U voxcpmane2
```

Or install with `pip`:

```bash
pip install -U voxcpmane2
```

For editable development from a source checkout, run
`uv pip install --python '>=3.10,<3.13' -e .` from this directory.

If you need to load raw VoxCPM2 `.safetensors` weights during development,
install the optional development extra:

```bash
uv pip install --python '>=3.10,<3.13' -e '.[development]'
```

## Run The Server

```bash
voxcpmane2-server
```

The server starts on `http://localhost:8000` by default. Open
`http://localhost:8000/` to use the included web playground. It exposes the main
workflows from the browser: generate speech, stream audio, play audio on the
server, create custom voices, and inspect available voices.

### M1 BaseLM Load Workaround

Some M1 Macs fail while loading the full BaseLM CoreML package on ANE. If server
startup fails with a BaseLM traceback and an error like this:

```text
ANE model load has failed for on-device compiled macho. Must re-compile the E5 bundle.
RuntimeError: `MLModelConfiguration`'s `.functionName` property must be `nil`
unless the model type is ML Program.
```

try the `0.1.3b1` beta and start the server with the split BaseLM package:

```bash
uv tool install --python '>=3.10,<3.13' --prerelease allow -U 'voxcpmane2==0.1.3b1'
voxcpmane2-server --split-base-lm
```

## Web Playground

Most users can start with the playground instead of writing API requests. After
starting the server, open `http://localhost:8000/` to generate speech with the
included voices, test streaming behavior, use server-side playback, create
custom voices, and switch between voice modes from the browser.

![VoxCPMANE2 web playground](assets/voxcpmane_frontend.png)

Common options:

```bash
voxcpmane2-server \
  --host 0.0.0.0 \
  --port 8000 \
```

If `--model-dir` is omitted, the server downloads the CoreML model directory
from `--repo-id`, restricted to only the packages needed for the selected
configuration. If individual package paths are not supplied, components are
loaded from that downloaded directory. The default repo layout includes
`config.json`, `embed_tokens.npy`, a small `.mlpackage` marker for CoreML repo
recognition, and the runtime packages:
`base_lm_multifunction.mlmodelc`, `residual_lm_fused_multifunction.mlmodelc`,
and the compiled component packages at the repo root. Included voice caches live
under `caches/`.

On machines where the full BaseLM package fails to load on ANE, use the split
BaseLM package:

```bash
voxcpmane2-server --split-base-lm
```

With `--split-base-lm`, the standard BaseLM package is not downloaded. The
server downloads only the two split BaseLM packages and `config.json` from
`seba/VoxCPMANE2-Debug-Models`, and downloads the remaining runtime components
from the default model repo.

## Working Modes

`--lm-mode` controls how multifunction LM prefill and decode handles are kept in
memory. The default mode is fixed-length `16`, exposed as `single-length` with
prefill/decode length `16`. Available prefill lengths are `1`, `8`, `16`, `32`,
`64`, and `128`; any of these can be used with `single-length` mode. If
`--lm-prefill-chunk-size` is omitted, `preload` and `hot-swap` default to
prefill length `128`; other modes default to `16`.

| Mode | Behavior | Tradeoff |
| --- | --- | --- |
| `hot-swap` | Keeps the selected prefill function loaded while idle, then swaps to length `1` for decode and back after generation. | Lower idle memory, with function load/unload cost around generation. |
| `preload` | Keeps length `1` and the selected prefill size resident for both BaseLM and ResidualLM, unloads prefill during decode, then reloads prefill when idle. | Avoids cold decode load, but roughly doubles BaseLM and ResidualLM resident memory. |
| `always-loaded` | Keeps length `1` and the selected prefill size resident and never unloads either function. | Fastest transitions, highest memory use. |
| `single-length` | Uses only the selected prefill length and restricts LM calls to that function. | Default at length `16`. Good TTFB/RTF tradeoff; decode also uses the selected length instead of length `1`. |

If memory use is not a concern and you want the best steady-state performance,
`preload` is usually the best option by RTF. Otherwise `single-length` tends to
provide the best latency/performance tradeoff. In informal power observations,
`single-length` 8 and 16 add very little power draw, under about 1 W; length 32
is around 1 W higher; larger lengths cost progressively more energy.

Lower TTFB and RTF are better. `Hot-swap` is only reported for `hot-swap` mode;
other modes do not perform a decode function swap.

## Memory Notes

The default compiled CoreML model bundle is about 3.2 GB by apparent file size.
This is a useful floor for estimating memory pressure because CoreML must load
the model programs and weights, and runtime state/KV caches add more memory on
top. Actual resident memory varies by macOS/CoreML version, compute unit
placement, active function handles, and request shape.

Approximate compiled model sizes in the default bundle:

| Component | Size |
| --- | ---: |
| BaseLM multifunction | 1.71 GB |
| ResidualLM multifunction | 615 MB |
| Feat encoder | 420 MB |
| LocDiT | 260 MB |
| Audio VAE encoder | 96 MB |
| Audio VAE decoder | 92 MB |
| Projections | 17 MB |
| FSQ | 8 MB |
| Total compiled models | 3.2 GB |

`preload` mode is intentionally memory-heavy. BaseLM and ResidualLM are separate
CoreML multifunction packages, and each loaded function is a separate CoreML
model handle. Keeping both length `1` and the selected prefill length resident
therefore roughly doubles the BaseLM and ResidualLM memory footprint compared
with modes that keep only one LM function resident. Based on the default bundle
sizes, the extra resident model memory for that second LM function is about
2.3 GB: roughly 1.71 GB for BaseLM plus 615 MB for ResidualLM, before CoreML
runtime overhead and KV/state buffers.

| Scenario | Mode | TTFB | RTF | Prefill | Hot-swap |
| --- | --- | ---: | ---: | ---: | ---: |
| Text only | `preload` | 0.745s | 0.591 | 0.664s | - |
| Text only | `hot-swap` | 1.095s | 0.672 | 0.322s | 0.721s |
| Text only | `single-length-1` | 2.589s | 0.612 | 2.529s | - |
| Text only | `single-length-8` | 0.474s | 0.664 | 0.413s | - |
| Text only | `single-length-16` | 0.306s | 0.712 | 0.246s | - |
| Text only | `single-length-32` | 0.224s | 0.723 | 0.170s | - |
| Text only | `single-length-64` | 0.197s | 0.809 | 0.143s | - |
| Text only | `single-length-128` | 0.163s | 1.029 | 0.110s | - |
| Text + reference wav | `preload` | 0.745s | 0.593 | 0.652s | - |
| Text + reference wav | `hot-swap` | 1.419s | 0.718 | 0.547s | 0.815s |
| Text + reference wav | `single-length-1` | 3.681s | 0.646 | 3.605s | - |
| Text + reference wav | `single-length-8` | 0.763s | 0.668 | 0.689s | - |
| Text + reference wav | `single-length-16` | 0.538s | 0.686 | 0.473s | - |
| Text + reference wav | `single-length-32` | 0.408s | 0.726 | 0.351s | - |
| Text + reference wav | `single-length-64` | 0.382s | 0.812 | 0.329s | - |
| Text + reference wav | `single-length-128` | 0.338s | 1.022 | 0.285s | - |
| Text + prompt wav + transcript | `preload` | 0.394s | 0.590 | 0.306s | - |
| Text + prompt wav + transcript | `hot-swap` | 1.225s | 0.662 | 0.361s | 0.811s |
| Text + prompt wav + transcript | `single-length-1` | 4.100s | 0.632 | 4.023s | - |
| Text + prompt wav + transcript | `single-length-8` | 0.841s | 0.664 | 0.781s | - |
| Text + prompt wav + transcript | `single-length-16` | 0.583s | 0.699 | 0.527s | - |
| Text + prompt wav + transcript | `single-length-32` | 0.468s | 0.728 | 0.412s | - |
| Text + prompt wav + transcript | `single-length-64` | 0.382s | 0.813 | 0.328s | - |
| Text + prompt wav + transcript | `single-length-128` | 0.341s | 1.045 | 0.287s | - |
| Text + reference + prompt | `preload` | 0.520s | 0.596 | 0.445s | - |
| Text + reference + prompt | `hot-swap` | 1.628s | 0.721 | 0.779s | 0.796s |
| Text + reference + prompt | `single-length-1` | 5.000s | 0.637 | 4.934s | - |
| Text + reference + prompt | `single-length-8` | 1.308s | 0.685 | 1.226s | - |
| Text + reference + prompt | `single-length-16` | 0.749s | 0.688 | 0.686s | - |
| Text + reference + prompt | `single-length-32` | 0.572s | 0.729 | 0.518s | - |
| Text + reference + prompt | `single-length-64` | 0.486s | 0.812 | 0.434s | - |
| Text + reference + prompt | `single-length-128` | 0.484s | 1.083 | 0.431s | - |
| Preset voice reference | `preload` | 1.171s | 0.608 | 1.058s | - |
| Preset voice reference | `hot-swap` | 1.423s | 0.673 | 0.560s | 0.802s |
| Preset voice reference | `single-length-1` | 2.660s | 0.634 | 2.603s | - |
| Preset voice reference | `single-length-8` | 0.503s | 0.682 | 0.445s | - |
| Preset voice reference | `single-length-16` | 0.334s | 0.681 | 0.278s | - |
| Preset voice reference | `single-length-32` | 0.262s | 0.722 | 0.208s | - |
| Preset voice reference | `single-length-64` | 0.230s | 0.809 | 0.177s | - |
| Preset voice reference | `single-length-128` | 0.198s | 1.066 | 0.146s | - |
| Preset voice high similarity | `preload` | 1.182s | 0.592 | 1.045s | - |
| Preset voice high similarity | `hot-swap` | 1.659s | 0.677 | 0.766s | 0.810s |
| Preset voice high similarity | `single-length-1` | 10.845s | 0.622 | 10.728s | - |
| Preset voice high similarity | `single-length-8` | 1.739s | 0.681 | 1.610s | - |
| Preset voice high similarity | `single-length-16` | 0.969s | 0.678 | 0.863s | - |
| Preset voice high similarity | `single-length-32` | 0.648s | 0.724 | 0.552s | - |
| Preset voice high similarity | `single-length-64` | 0.494s | 0.806 | 0.390s | - |
| Preset voice high similarity | `single-length-128` | 0.457s | 1.056 | 0.359s | - |

Examples:

```bash
# Default single-length behavior with length 16.
voxcpmane2-server

# Keep both prefill and decode functions resident.
voxcpmane2-server --lm-mode always-loaded

# Preload decode and prefill length 128, but unload prefill during decode.
voxcpmane2-server --lm-mode preload

# Use only one LM function length.
voxcpmane2-server --lm-mode single-length --lm-prefill-chunk-size 16
```

## Model Path Options


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
prompt embeddings, transcript, and VAE decoder context when available.

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

- `GET /`: browser-based web playground for generation and voice management
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
- `name.prompt.decode_context.npy`: optional tail prompt VAE patches used for
  high-similarity audio continuity

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
