# VoxCPMANE2

Install the VoxCPM2 package:

```bash
uv pip install -U 'voxcpmane2[tokenizer]'
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
- Python `>=3.10,<3.15` for the normal GIL runtime
- `uv` or `pip`
- CoreML runtime support through `coremltools`
- Optional: `pydub` for `mp3`, `opus`, `ogg`, and `aac` responses
- Optional: `voxcpmane2[tokenizer]` for text tokenization through
  `tokenizers`
- Optional: `voxcpmane2[normalize]` for `normalize=true` text normalization

The async no-GIL branch requires free-threaded CPython `3.14t`, plus a locally
patched CoreMLTools native bridge because official CoreMLTools wheels are not
published for this Python ABI yet. Do not install `voxcpmane2[normalize]` in the
`3.14t` environment unless you already have a compatible `kaldifst` build;
`voxcpmane2[nogil]` installs the tokenizer and optional audio-format dependency
without the normalization stack.

## Installation

### Normal PyPI Runtime

Install with `uv`:

```bash
uv pip install -U 'voxcpmane2[tokenizer]'
```

Or install with `pip`:

```bash
python -m pip install -U 'voxcpmane2[tokenizer]'
```

For editable development from a source checkout, run this from the repository:

```bash
uv pip install -e '.[tokenizer]'
```

### Async No-GIL Branch

The async/no-GIL runtime lives on the `voxcpmane2-async-no-gil` branch. Install
from a source checkout so the CoreMLTools patch under `patches/` is available.

Clone the branch:

```bash
git clone --branch voxcpmane2-async-no-gil https://github.com/0seba/VoxCPMANE.git
cd VoxCPMANE
export VOXCPMANE2_DIR="$PWD"
```

#### Install With uv

Create a free-threaded Python `3.14t` environment and install VoxCPMANE2:

```bash
uv python install cpython-3.14.2+freethreaded
uv venv --python cpython-3.14.2+freethreaded .venv314t

RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
  uv pip install --python .venv314t/bin/python -e '.[nogil]'
```

#### Install With pip

For pure `pip`, provide an existing free-threaded Python `3.14t` executable,
then create the virtual environment and install VoxCPMANE2:

```bash
python3.14t -m venv .venv314t
source .venv314t/bin/activate
python -m pip install -U pip setuptools wheel cmake

RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
  python -m pip install -e '.[nogil]'
```

`pip` does not install Python itself; only the `uv` path above provisions the
free-threaded interpreter.

#### Build CoreMLTools For Python 3.14t

Both installation paths need the same patched CoreMLTools native modules. Build
them from CoreMLTools `9.0` and copy the native libraries into the `.venv314t`
environment:

```bash
export PY314T="$VOXCPMANE2_DIR/.venv314t/bin/python"
export COREMLTOOLS_SRC="/tmp/coremltools-9.0-cp314t"

git clone --depth 1 --branch 9.0 https://github.com/apple/coremltools.git "$COREMLTOOLS_SRC"
cd "$COREMLTOOLS_SRC"
git apply "$VOXCPMANE2_DIR/patches/coremltools-9.0-py314t-array-lifetime.patch"

"$PY314T" -m pip install -U setuptools wheel cmake
PATH="$VOXCPMANE2_DIR/.venv314t/bin:$PATH" cmake -S . -B build-cp314t \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYTHON_EXECUTABLE="$PY314T" \
  -DPYTHON_INCLUDE_DIR="$("$PY314T" -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
  -DPYTHON_INCLUDE_DIRS="$("$PY314T" -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
  -DPYTHON_LIBRARY="$("$PY314T" -c 'import sysconfig, pathlib; print(pathlib.Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY"))')" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cp314t --target coremlpython milstoragepython modelpackage --parallel

SITE_PACKAGES="$("$PY314T" -c 'import site; print(site.getsitepackages()[0])')"
cp coremltools/libcoremlpython.so \
  coremltools/libmilstoragepython.so \
  coremltools/libmodelpackage.so \
  "$SITE_PACKAGES/coremltools/"
```

The patch keeps the NumPy-backed `MLMultiArray` owner alive and decrements its
reference under the GIL. Without it, CoreML can release Python arrays from a
worker thread and crash under free-threaded Python.

Verify the install:

```bash
PYTHON_GIL=0 "$PY314T" -c "import sys, coremltools, tokenizers; print(sys.version); print('gil=', sys._is_gil_enabled())"
PYTHON_GIL=0 "$VOXCPMANE2_DIR/.venv314t/bin/voxcpmane2-server" --help
```

Start the async hot-swap server:

```bash
cd "$VOXCPMANE2_DIR"
PYTHON_GIL=0 .venv314t/bin/voxcpmane2-server \
  --lm-mode hot-swap \
  --lm-prefill-chunk-size 128 \
  --lm-async-decode-load \
  --lm-async-prefill-unload \
  --prefill-audio-async \
  --vae-async-decode \
  --live-rtf final
```

Open `http://localhost:8000/` and set the playground's initial stream buffer to
`0.00` to feel backend TTFB directly.

#### Direct Git Install

After the CoreMLTools native bridge is built, the branch can also be installed
directly by URL:

```bash
RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
  "$PY314T" -m pip install \
  'voxcpmane2[nogil] @ git+https://github.com/0seba/VoxCPMANE.git@voxcpmane2-async-no-gil'
```

This direct install does not provide the local patch file, so use the clone path
when setting up a machine from scratch.

#### PyPI Extras And Installer Limits

`voxcpmane2[nogil]` is a normal Python extra. It can request Python packages
such as `tokenizers` and `pydub`, and if a no-GIL release is published to PyPI,
installs such as the following count as `voxcpmane2` PyPI downloads:

```bash
uv pip install --python .venv314t/bin/python 'voxcpmane2[nogil]'
python -m pip install 'voxcpmane2[nogil]'
```

Installing from GitHub with `voxcpmane2 @ git+https://...` does not count as a
PyPI download for `voxcpmane2`; only packages actually downloaded from PyPI are
counted by PyPI download analytics.

Python packaging does not provide a safe pip/uv flag that both counts as a PyPI
install and runs arbitrary post-install commands to patch, compile, and copy
CoreMLTools native libraries. Extras can only declare dependencies. The practical
options are:

- publish this branch to PyPI as a pre-release or normal release, then document
  the CoreMLTools build step above;
- ship a helper console command in `voxcpmane2` that builds/copies the patched
  CoreMLTools native modules, but users still run that command after install;
- publish a separate patched CoreMLTools wheel when redistribution is acceptable.

This branch uses the first approach and keeps CoreMLTools as an explicit local
build step.

## Run The Server

```bash
voxcpmane2-server
```

The server starts on `http://localhost:8000` by default. Open
`http://localhost:8000/` to use the included web playground. It exposes the main
workflows from the browser: generate speech, stream audio, play audio on the
server, create custom voices, and inspect available voices.

## Web Playground

Most users can start with the playground instead of writing API requests. After
starting the server, open `http://localhost:8000/` to generate speech with the
included voices, test streaming behavior, use server-side playback, create
custom voices, and switch between voice modes from the browser.

Common options:

```bash
voxcpmane2-server \
  --host 0.0.0.0 \
  --port 8000 \
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

Experimental overlap flags are opt-in so baseline and async runs are comparable:

- `--lm-async-decode-load`: in hot-swap mode, start decode handle loading in the
  background as each LM finishes prefill, and wait only when decode is needed.
- `--lm-async-prefill-unload`: unload inactive prefill handles on a background
  thread, mainly for preload mode.
- `--prefill-audio-async`: for prompt/reference WAV inputs, run AudioVAE encoder
  chunk production on a background thread while feature encoder and BaseLM prefill
  consume previous chunks.
- `--vae-async-decode`: run AudioVAE decoder calls on a background thread after
  `--vae-early-decode-steps`, bounded by `--vae-decode-max-pending`.

Measure before and after on the same machine and model assets:

```bash
uv run voxcpmane2-benchmark-overlap \
  --variant both \
  --model-dir /path/to/local-models \
  --voice af_alloy \
  --lm-mode hot-swap \
  --runs 3 \
  --warmup-runs 1
```

For a minimal Python 3.14t environment without the tokenizer extra, pass
pre-tokenized IDs:

```bash
uv run --python cpython-3.14.2+freethreaded voxcpmane2-benchmark-overlap \
  --variant both \
  --model-dir /path/to/local-models \
  --token-ids 1,2,3,4,5 \
  --runs 3 \
  --warmup-runs 1
```

For a prompt/reference WAV case that exercises AudioVAE encoder overlap, pass
`--reference-wav-path` or `--prompt-wav-path` instead of a cached `--voice`.

Compare hot-swap under the same free-threaded Python with and without the GIL:

```bash
scripts/benchmark_gil_vs_nogil_hot_swap.py \
  --runs 3 \
  --warmup-runs 1 \
  -- \
  --model-dir /path/to/local-models \
  --prompt-wav-path /path/to/prompt.wav \
  --prompt-text "Transcript for the prompt WAV." \
  --text "Text to synthesize."
```

The wrapper runs `voxcpmane2-benchmark-overlap` twice with identical hot-swap
arguments, first with `PYTHON_GIL=1` and then with `PYTHON_GIL=0`, and writes raw
logs plus `comparison.json` under `benchmarks/`.

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) for the original VoxCPM model family
- [VoxCPMANE](../VoxCPMANE) for the earlier Apple Neural Engine server/runtime
