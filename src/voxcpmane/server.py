"""VoxCPM2 TTS HTTP server — OpenAI-compatible API.

Adapted from VoxCPMANE server.py. Uses VoxCPM2Generator for inference.
"""

import faulthandler
faulthandler.enable()

import os
import io
import re
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import uvicorn
import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    HTMLResponse,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import soundfile
import soxr
from pydantic import BaseModel
from typing import Optional
import aiofiles
from huggingface_hub import snapshot_download
import asyncio
from dataclasses import dataclass
import argparse
import ftfy

from .generator import VoxCPM2Generator

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model loading (deferred — called from main() or startup)
# ---------------------------------------------------------------------------

REPO_ID = "seba/VoxCPM2-ANE"
MODEL_PATH_PREFIX = ""
VOICE_CACHE_DIR = ""
CUSTOM_VOICE_CACHE_DIR = os.path.expanduser("~/.cache/ane_tts")

generator: VoxCPM2Generator = None  # type: ignore[assignment]
text_normalizer = None


def load_model(
    model_dir: str | None = None,
    quantized_model_dir: str | None = None,
    embedding_path: str | None = None,
    hot_swap: bool = True,
    lm_prefill_chunk_size: int | None = None,
    base_lm_splits: int = 2,
    quantized_lm_mode: str = "auto",
    quantized_components: str = "auto",
    compiled_fallback_dir: str | None = None,
    debug: bool = False,
):
    """Load VoxCPM2Generator. Called once from main().

    Args:
        model_dir: Local path to CoreML model directory. If None, downloads
            from HuggingFace ``REPO_ID``.
        quantized_model_dir: Optional path to a directory containing quantized
            W8 overrides for LM and LocDiT models. Files found here override
            the corresponding models from ``model_dir``. Generator-ready W8
            directories may contain split BaseLM packages
            ``base_lm_s4_part*_of_*.mlpackage`` plus
            ``residual_lm_fused_s4.mlpackage`` and ``locdit_p4_c4.mlpackage``.
            Older multifunction W8 directories may contain
            ``base_lm_multifunction.mlpackage`` and
            ``residual_lm_fused_multifunction.mlpackage``.
        embedding_path: Path to the LM embedding table. Accepts:
            - A ``.npy`` file (direct load)
            - A directory containing ``embed_tokens.npy``
            - A directory containing ``.safetensors`` files
            - ``None`` to search ``model_dir``, then local HF cache
        hot_swap: If True (default), unload inactive LM functions to save
            memory when fast LM prefill uses a separate function. If False,
            keep both prefill and decode LM functions loaded simultaneously.
        lm_prefill_chunk_size: Optional LM chunk length used during prompt
            prefill. ``None`` lets fixed fp16 split packages use their native
            chunk size. W8 multifunction overrides default to ``1`` because
            MLState reuse across separate CoreML function handles has been
            observed to segfault. ``128`` enables experimental fast W8 prefill.
        base_lm_splits: Number of fp16 BaseLM split packages to load from
            ``model_dir`` when no single BaseLM override is provided.
        debug: Enable verbose debug logging in the generator.
    """
    global generator, MODEL_PATH_PREFIX, VOICE_CACHE_DIR

    if model_dir is not None:
        MODEL_PATH_PREFIX = os.path.abspath(model_dir)
        print(f"📂 Using local model directory: {MODEL_PATH_PREFIX}")
    else:
        print(f"🚀 Downloading models from HuggingFace: {REPO_ID}")
        MODEL_PATH_PREFIX = snapshot_download(repo_id=REPO_ID)

    VOICE_CACHE_DIR = os.path.join(MODEL_PATH_PREFIX, "caches")

    # Build generator kwargs for quantized overrides
    generator_model_dir = MODEL_PATH_PREFIX
    gen_kwargs = {"base_lm_splits": int(base_lm_splits)}
    if quantized_model_dir:
        qdir = os.path.abspath(quantized_model_dir)
        print(f"⚡ Using quantized models from: {qdir}")
        all_components = {
            "base_lm", "residual_lm", "locdit",
            "vae_encoder", "feat_encoder", "vae_decoder", "fsq", "projections",
        }
        if quantized_components == "auto":
            selected_components = {"base_lm", "residual_lm", "locdit"}
        else:
            parsed = {
                c.strip() for c in quantized_components.split(",") if c.strip()
            }
            unknown = parsed - all_components
            if unknown:
                raise ValueError(
                    f"unknown components in --quantized-components: {sorted(unknown)}"
                )
            selected_components = parsed
        print(f"   ✓ quantized components: {sorted(selected_components)}")

        base_lm_q = os.path.join(qdir, "base_lm_multifunction.mlpackage")
        residual_lm_q = os.path.join(qdir, "residual_lm_fused_multifunction.mlpackage")
        locdit_q = os.path.join(qdir, "locdit_p4_c4.mlpackage")
        residual_lm_split_q = os.path.join(qdir, "residual_lm_fused_s4.mlpackage")
        split_paths = sorted(
            pathlib.Path(qdir).glob("base_lm_s4_part*_of_*.mlpackage")
        )

        if quantized_lm_mode not in {"auto", "split", "multifunction"}:
            raise ValueError(
                f"quantized_lm_mode must be one of auto|split|multifunction, got {quantized_lm_mode}"
            )

        split_available = bool(split_paths)
        multifunction_available = os.path.exists(base_lm_q)
        if quantized_lm_mode == "split":
            use_split = True
        elif quantized_lm_mode == "multifunction":
            use_split = False
        else:
            use_split = split_available

        if use_split and "base_lm" in selected_components:
            if not split_available:
                raise FileNotFoundError(
                    f"requested quantized_lm_mode=split but no split BaseLM packages found in {qdir}"
                )
            total_splits = len(split_paths)
            expected = [
                pathlib.Path(qdir) / f"base_lm_s4_part{i}_of_{total_splits}.mlpackage"
                for i in range(total_splits)
            ]
            if not all(path.exists() for path in expected):
                raise FileNotFoundError(
                    "quantized split BaseLM directory is incomplete; expected "
                    + ", ".join(str(path) for path in expected)
                )
            if "residual_lm" in selected_components and not os.path.exists(residual_lm_split_q):
                raise FileNotFoundError(residual_lm_split_q)
            print(f"   ✓ base_lm split overrides: {total_splits} packages")
            gen_kwargs["base_lm_split_model_paths"] = [str(path) for path in expected]
            gen_kwargs["base_lm_splits"] = total_splits
            if "residual_lm" in selected_components and os.path.exists(residual_lm_split_q):
                print(f"   ✓ residual_lm override: {residual_lm_split_q}")
                gen_kwargs["residual_lm_model_path"] = residual_lm_split_q
            if "locdit" in selected_components and os.path.exists(locdit_q):
                print(f"   ✓ locdit override: {locdit_q}")
            if lm_prefill_chunk_size is not None:
                gen_kwargs["lm_prefill_chunk_size"] = int(lm_prefill_chunk_size)
            else:
                gen_kwargs["lm_prefill_chunk_size"] = None
            print("   ✓ W8 split mode for base_lm")
        elif multifunction_available and "base_lm" in selected_components:
            if lm_prefill_chunk_size is None:
                lm_prefill_chunk_size = 1
            lm_prefill_chunk_size = int(lm_prefill_chunk_size)
            if lm_prefill_chunk_size not in (1, 128):
                raise ValueError(
                    "lm_prefill_chunk_size must be 1 or 128 for the current "
                    f"VoxCPM2 LM multifunction packages; got {lm_prefill_chunk_size}"
                )
            print(f"   ✓ base_lm override: {base_lm_q}")
            print("   ⚠ W8 LM override is experimental and may degrade speech quality")
            gen_kwargs["base_lm_model_path"] = base_lm_q
            gen_kwargs["base_lm_splits"] = 1  # multifunction = single model

            residual_multifunction_selected = (
                "residual_lm" in selected_components and os.path.exists(residual_lm_q)
            )
            if residual_multifunction_selected:
                print(f"   ✓ residual_lm override: {residual_lm_q}")
                print("   ⚠ W8 residual LM override is experimental and may degrade speech quality")
                gen_kwargs["residual_lm_model_path"] = residual_lm_q
            if "locdit" in selected_components and os.path.exists(locdit_q):
                print(f"   ✓ locdit override: {locdit_q}")
                gen_kwargs["locdit_model_path"] = locdit_q

            if lm_prefill_chunk_size == 1:
                # Keep all LM calls on the same CoreML function handle when both
                # LMs are multifunction-capable. If residual LM is still fp16 s4,
                # it does not expose length_1; forcing preload chunk size 1 would
                # fail during model init.
                gen_kwargs["lm_unload_inactive_functions"] = False
                if residual_multifunction_selected:
                    gen_kwargs["lm_preload_chunk_sizes"] = [1]
                    print("   ✓ LM stable state mode: using length_1 for prefill + generate")
                else:
                    print("   ✓ Mixed LM mode: base_lm length_1 prefill enabled; residual_lm keeps native chunking")
            elif hot_swap:
                # Enable hot-swapping: keep length-128 loaded while idle (for
                # fast prefill), swap to length-1 for AR generation, then swap
                # back to length-128 when the request finishes.
                gen_kwargs["lm_unload_inactive_functions"] = True
                gen_kwargs["lm_idle_prefill_chunk_size"] = lm_prefill_chunk_size
                print("   ✓ LM hot-swap enabled: length-128 (prefill) ↔ length-1 (generate)")
            else:
                # Keep both functions loaded — uses more memory but avoids
                # any function loading/unloading during generation.
                gen_kwargs["lm_unload_inactive_functions"] = False
                gen_kwargs["lm_preload_chunk_sizes"] = [1, lm_prefill_chunk_size]
                print("   ✓ LM hot-swap DISABLED: preloading length_1 + length_128")

            gen_kwargs["lm_prefill_chunk_size"] = lm_prefill_chunk_size
        else:
            if quantized_lm_mode == "multifunction" and "base_lm" in selected_components:
                raise FileNotFoundError(
                    f"requested quantized_lm_mode=multifunction but missing {base_lm_q}"
                )
            if "residual_lm" in selected_components and os.path.exists(residual_lm_q):
                print(f"   ✓ residual_lm override: {residual_lm_q}")
                gen_kwargs["residual_lm_model_path"] = residual_lm_q
            if "locdit" in selected_components and os.path.exists(locdit_q):
                print(f"   ✓ locdit override: {locdit_q}")
                gen_kwargs["locdit_model_path"] = locdit_q

        # Optional non-LM component overrides
        maybe_non_lm = [
            ("vae_encoder", "audio_vae_encoder.mlpackage", "vae_encoder_model_path"),
            ("feat_encoder", "feat_encoder.mlpackage", "feat_encoder_model_path"),
            ("vae_decoder", "audio_vae_decoder_lf4.mlpackage", "vae_decoder_model_path"),
            ("fsq", "fsq_s4.mlpackage", "fsq_model_path"),
            ("projections", "projections.mlpackage", "projections_model_path"),
        ]
        for component, filename, kwarg in maybe_non_lm:
            if component not in selected_components:
                continue
            qpath = os.path.join(qdir, filename)
            if not os.path.exists(qpath):
                raise FileNotFoundError(
                    f"requested quantized component '{component}' but missing {qpath}"
                )
            print(f"   ✓ {component} override: {qpath}")
            gen_kwargs[kwarg] = qpath

    if embedding_path:
        gen_kwargs["embedding_safetensors_path"] = embedding_path
    if compiled_fallback_dir:
        gen_kwargs["compiled_fallback_dir"] = os.path.abspath(compiled_fallback_dir)

    gen_kwargs["debug"] = debug

    print("Loading CoreML models via VoxCPM2Generator...")
    generator = VoxCPM2Generator(generator_model_dir, **gen_kwargs)
    generator.preload_tokenizer()
    print("✅ Models loaded successfully.")


# ---------------------------------------------------------------------------
# Job queue (single-worker threading model)
# ---------------------------------------------------------------------------

@dataclass
class GenerationJob:
    request: "SpeechRequest"
    output_queue: queue.Queue
    cancel_event: threading.Event
    job_id: int


GENERATION_QUEUE = queue.Queue(maxsize=1)
CURRENT_JOB: Optional[GenerationJob] = None
JOB_COUNTER = 0


def generation_worker():
    """Single thread that touches the CoreML model. Runs forever."""
    global CURRENT_JOB
    import traceback as _tb
    import sys as _sys

    while True:
        try:
            job = GENERATION_QUEUE.get()
            CURRENT_JOB = job
            try:
                print(f"🔄 Job {job.job_id}: starting generation", flush=True)
                audio_gen = generate_audio_chunks(
                    text_to_generate=job.request.input,
                    control_instruction=job.request.control_instruction,
                    reference_wav_path=job.request.reference_wav_path,
                    prompt_wav_path=job.request.prompt_wav_path,
                    prompt_text=job.request.prompt_text,
                    voice=job.request.voice,
                    max_length=job.request.max_length,
                    cfg_value=job.request.cfg_value,
                    inference_timesteps=job.request.inference_timesteps,
                    normalize=job.request.normalize,
                    cancellation_event=job.cancel_event,
                )
                chunk_count = 0
                for chunk in audio_gen:
                    if job.cancel_event.is_set():
                        print(f"🛑 Job {job.job_id}: cancelled after {chunk_count} chunks", flush=True)
                        break
                    job.output_queue.put(chunk)
                    chunk_count += 1
                print(f"✅ Job {job.job_id}: done, {chunk_count} chunks produced", flush=True)
            except Exception as e:
                print(f"❌ Job {job.job_id}: exception in worker:", flush=True)
                _tb.print_exc()
                _sys.stderr.flush()
                job.output_queue.put(e)
            finally:
                job.output_queue.put(None)
                CURRENT_JOB = None
                GENERATION_QUEUE.task_done()
        except Exception:
            print("❌ generation_worker: outer exception:", flush=True)
            _tb.print_exc()
            _sys.stderr.flush()
            CURRENT_JOB = None
            if "job" in locals() and isinstance(job, GenerationJob):
                job.output_queue.put(Exception("Worker failed"))
                GENERATION_QUEUE.task_done()
            time.sleep(1)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
app = FastAPI(title="VoxCPM2 TTS Server")
CACHED_VOICE_TEXT = """Jittery Jack's jam jars jiggled jauntily, jolting Jack's jumbled jelly-filled jars joyously.
Cindy's circular cymbals clanged cheerfully, clashing crazily near Carla's crashing crockery.
You think you can just waltz in here and cause chaos? Well, I've got news for you."""

APP_DIR = pathlib.Path(__file__).parent
FRONTEND_FILE = APP_DIR / "frontend" / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "voxcpm2"
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    control_instruction: Optional[str] = ""
    reference_wav_path: Optional[str] = None
    prompt_wav_path: Optional[str] = None
    prompt_text: Optional[str] = ""
    max_length: Optional[int] = 2048
    cfg_value: Optional[float] = 2.0
    inference_timesteps: Optional[int] = 10
    normalize: Optional[bool] = False


class PlaybackRequest(SpeechRequest):
    show_progress: Optional[bool] = True


class CreateVoiceRequest(BaseModel):
    voice_name: str
    prompt_wav_path: str
    prompt_text: str
    replace: Optional[bool] = False


# ---------------------------------------------------------------------------
# Voice cache helpers
# ---------------------------------------------------------------------------

def load_available_voices():
    voices = set()
    for d in [VOICE_CACHE_DIR, CUSTOM_VOICE_CACHE_DIR]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".npy"):
                    voices.add(f[:-4])
    return sorted(list(voices))


def is_default_voice(voice_name: str) -> bool:
    return os.path.exists(os.path.join(VOICE_CACHE_DIR, f"{voice_name}.npy"))


def get_voice_prompt_text(voice_name: str) -> str:
    if is_default_voice(voice_name):
        return CACHED_VOICE_TEXT
    txt_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{voice_name}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(
        status_code=500,
        detail=f"Voice '{voice_name}' found but transcription file missing at: {txt_path}",
    )


def load_voice_cache(voice_name: str):
    for d in [VOICE_CACHE_DIR, CUSTOM_VOICE_CACHE_DIR]:
        path = os.path.join(d, f"{voice_name}.npy")
        if os.path.exists(path):
            return np.load(path)
    raise HTTPException(
        status_code=404,
        detail=f"Voice '{voice_name}' not found. Available: {load_available_voices()}",
    )


def validate_voice_parameters(max_length, cfg_value, inference_timesteps):
    if not (0 < max_length <= 4096):
        raise HTTPException(status_code=400, detail="max_length must be between 1 and 4096")
    if not (0.0 <= cfg_value <= 10.0):
        raise HTTPException(status_code=400, detail="cfg_value must be between 0.0 and 10.0")
    if not (0 < inference_timesteps <= 100):
        raise HTTPException(status_code=400, detail="inference_timesteps must be between 1 and 100")


def normalize_apple_punctuation(text):
    """Convert Apple smart/typographic punctuation to ASCII equivalents."""
    table = str.maketrans({
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        '\u2013': '-', '\u2014': '-', '\u2026': '...',
        '\u2022': '*', '\u00a0': ' ', '\u201a': ',', '\u201e': '"',
        '\u2039': '<', '\u203a': '>',
    })
    return text.translate(table)


def audio_float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float audio to PCM16 without wrapping out-of-range samples."""
    audio_arr = np.asarray(audio, dtype=np.float32)
    return (np.clip(audio_arr, -1.0, 1.0) * 32767.0).astype(np.int16)


# ---------------------------------------------------------------------------
# Generation core — VoxCPM2 conditioning dispatch
# ---------------------------------------------------------------------------

def generate_audio_chunks(
    text_to_generate,
    control_instruction="",
    reference_wav_path=None,
    prompt_wav_path=None,
    prompt_text=None,
    voice=None,
    max_length=2048,
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=False,
    cancellation_event=None,
):
    global text_normalizer

    if cancellation_event is None:
        cancellation_event = threading.Event()

    validate_voice_parameters(max_length, cfg_value, inference_timesteps)

    # Text normalization (lazy, matching upstream)
    text = text_to_generate
    if normalize:
        if text_normalizer is None:
            from .text_normalize import TextNormalizer
            text_normalizer = TextNormalizer()
        text = text_normalizer.normalize(text)

    text = normalize_apple_punctuation(text)
    text = ftfy.fix_text(text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    # Control instruction → "(control)text" (upstream app.py line 293)
    control = (control_instruction or "").strip()
    control = re.sub(r"[()（）]", "", control).strip()
    final_text = f"({control}){text}" if control else text

    reference_path = (reference_wav_path or "").strip() or None
    prompt_path = (prompt_wav_path or "").strip() or None
    prompt_text_clean = (prompt_text or "").strip()

    # Voice cache mode. Cached voices are continuation prompts and can be
    # combined with a separate reference_wav_path, matching upstream's
    # reference+continuation packing.
    prompt_audio_feat = None
    voice_name = (voice or "").strip() or None
    if voice_name is not None:
        if prompt_path is not None:
            raise ValueError("voice and prompt_wav_path are both continuation prompts; use only one")
        audio_cache = load_voice_cache(voice_name)
        voice_prompt_text = get_voice_prompt_text(voice_name)
        # Cached voices are continuation prompts: feed both the prompt
        # transcript and the pre-encoded prompt audio features. Prepending
        # the transcript without audio conditioning breaks the trained
        # text/audio mask contract and produces poor generations.
        prompt_text_clean = voice_prompt_text.strip()
        prompt_audio_feat = audio_cache.astype(np.float32, copy=False)

    for label, path in (
        ("reference_wav_path", reference_path),
        ("prompt_wav_path", prompt_path),
    ):
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")

    target_text_length = len(generator._encode_text(final_text))
    effective_max_length = min(int(target_text_length * 6.0 + 10), int(max_length))

    gen_kwargs = dict(
        target_text=final_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        max_len=effective_max_length,
    )
    if reference_path is not None:
        gen_kwargs["reference_wav_path"] = reference_path
    if prompt_path is not None:
        gen_kwargs["prompt_wav_path"] = prompt_path
        gen_kwargs["prompt_text"] = prompt_text_clean
    elif prompt_audio_feat is not None:
        gen_kwargs["prompt_audio_feat"] = prompt_audio_feat
        gen_kwargs["prompt_text"] = prompt_text_clean

    has_reference = reference_path is not None
    has_prompt = prompt_path is not None or prompt_audio_feat is not None
    if has_reference and has_prompt:
        mode = "reference_plus_continuation"
    elif has_reference:
        mode = "reference_only"
    elif has_prompt:
        mode = "continuation"
    else:
        mode = "zero_shot"
    print(f"🎤 generate_audio_chunks: mode={mode}, text={final_text[:80]!r}..., "
          f"max_len={effective_max_length}, cfg={cfg_value}, steps={inference_timesteps}",
          flush=True)
    import sys; sys.stdout.flush(); sys.stderr.flush()

    try:
        chunk_idx = 0
        for audio_chunk in generator.generate_streaming(**gen_kwargs):
            if cancellation_event.is_set():
                print(f"🛑 generate_audio_chunks: cancelled at chunk {chunk_idx}", flush=True)
                break
            chunk_idx += 1
            yield audio_chunk.astype(np.float32)
    except GeneratorExit:
        print("generate_audio_chunks: GeneratorExit", flush=True)
    except Exception as e:
        import traceback
        print(f"❌ generate_audio_chunks: exception: {e}", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        cancellation_event.set()


# ---------------------------------------------------------------------------
# Startup scan for voice caches
# ---------------------------------------------------------------------------

def scan_and_compile_audio_cache():
    """Scan custom cache dir for audio+txt pairs and compile missing .npy caches."""
    if not os.path.exists(CUSTOM_VOICE_CACHE_DIR):
        return

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac", ".m4a"}
    try:
        files = os.listdir(CUSTOM_VOICE_CACHE_DIR)
    except Exception as e:
        print(f"⚠️  Failed to list custom cache dir: {e}")
        return

    file_map = {}
    for f in files:
        name, ext = os.path.splitext(f)
        file_map.setdefault(name, set()).add(ext.lower())

    for name, extensions in file_map.items():
        if ".npy" in extensions:
            continue
        has_txt = ".txt" in extensions
        audio_ext = next((ext for ext in extensions if ext in AUDIO_EXTENSIONS), None)

        if has_txt and audio_ext:
            print(f"🔄 Compiling cache for voice: '{name}' from {audio_ext}...")
            audio_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}{audio_ext}")
            txt_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}.txt")

            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt_text_val = f.read()
            except Exception as e:
                print(f"❌ Failed to read text for '{name}': {e}")
                continue

            tmp_wav_path = None
            try:
                processing_path = audio_path
                if PYDUB_AVAILABLE and audio_ext not in [".wav", ".flac", ".ogg"]:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_wav_path = tmp.name
                    AudioSegment.from_file(audio_path).export(tmp_wav_path, format="wav")
                    processing_path = tmp_wav_path

                # Encode with VAE to create cache
                patches = generator.vae_encoder.encode_wav(processing_path)
                npy_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}.npy")
                np.save(npy_path, patches)
                print(f"✅ Compiled voice: '{name}'")
            except Exception as e:
                print(f"❌ Failed to compile voice '{name}': {e}")
            finally:
                if tmp_wav_path and os.path.exists(tmp_wav_path):
                    try:
                        os.unlink(tmp_wav_path)
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    scan_and_compile_audio_cache()
    threading.Thread(target=generation_worker, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    try:
        async with aiofiles.open(FRONTEND_FILE, mode="r") as f:
            return HTMLResponse(content=await f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


async def poll_queue_for_chunks(output_queue, poll_interval=0.005):
    while True:
        try:
            item = output_queue.get_nowait()
            if item is None:
                break
            elif isinstance(item, Exception):
                raise item
            else:
                yield item
        except queue.Empty:
            await asyncio.sleep(poll_interval)


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    audio_format = request.response_format.lower()

    if audio_format in ["wav", "flac"]:
        pass
    elif not PYDUB_AVAILABLE:
        raise HTTPException(status_code=501, detail=f"Format '{audio_format}' requires pydub.")
    elif audio_format not in ["mp3", "opus", "ogg", "aac"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {audio_format}")

    global JOB_COUNTER
    JOB_COUNTER += 1
    job = GenerationJob(request, queue.Queue(maxsize=1024), threading.Event(), JOB_COUNTER)

    try:
        GENERATION_QUEUE.put_nowait(job)
    except queue.Full:
        raise HTTPException(status_code=429, detail="Server is busy")

    all_chunks = []
    try:
        async for chunk in poll_queue_for_chunks(job.output_queue):
            all_chunks.append(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        job.cancel_event.set()

    if not all_chunks:
        raise HTTPException(status_code=500, detail="No audio produced")

    full_audio = np.concatenate(all_chunks)
    buffer = io.BytesIO()

    if audio_format in ["wav", "flac"]:
        soundfile.write(buffer, full_audio, SAMPLE_RATE, format=audio_format)
        media_type = f"audio/{audio_format}"
    else:
        pcm = audio_float_to_int16(full_audio)
        seg = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=SAMPLE_RATE, channels=1)
        fmt_map = {"mp3": ("mp3", "audio/mpeg"), "opus": ("opus", "audio/opus"),
                    "ogg": ("ogg", "audio/ogg"), "aac": ("adts", "audio/aac")}
        exp_fmt, media_type = fmt_map[audio_format]
        seg.export(buffer, format=exp_fmt)

    buffer.seek(0)
    return Response(content=buffer.getvalue(), media_type=media_type)


@app.post("/v1/audio/speech/stream")
async def stream_speech(request: SpeechRequest):
    global JOB_COUNTER
    JOB_COUNTER += 1
    job = GenerationJob(request, queue.Queue(maxsize=1024), threading.Event(), JOB_COUNTER)

    try:
        GENERATION_QUEUE.put_nowait(job)
    except queue.Full:
        raise HTTPException(status_code=429, detail="Server is busy")

    async def audio_stream():
        try:
            async for chunk in poll_queue_for_chunks(job.output_queue):
                yield audio_float_to_int16(chunk).tobytes()
        finally:
            job.cancel_event.set()

    return StreamingResponse(
        audio_stream(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE)},
    )


@app.post("/v1/audio/speech/playback")
async def playback_speech(request: PlaybackRequest):
    global JOB_COUNTER, CURRENT_JOB
    JOB_COUNTER += 1
    job = GenerationJob(request, queue.Queue(maxsize=1024), threading.Event(), JOB_COUNTER)

    try:
        GENERATION_QUEUE.put_nowait(job)
        CURRENT_JOB = job
    except queue.Full:
        raise HTTPException(status_code=429, detail="Server is busy")

    playback_start = time.time()
    chunk_count = 0

    try:
        if not sd.query_devices():
            raise HTTPException(status_code=500, detail="No audio output devices")

        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
                             latency="low", blocksize=1024) as stream:
            last_chunk = None
            async for chunk in poll_queue_for_chunks(job.output_queue):
                if time.time() - playback_start > 300:
                    raise HTTPException(status_code=500, detail="Playback timeout")
                chunk_count += 1
                last_chunk = chunk
                await asyncio.to_thread(stream.write, chunk)

            # Fade-out to prevent click/pop
            if last_chunk is not None and len(last_chunk) > 100:
                fade_samples = min(int(SAMPLE_RATE * 0.05), len(last_chunk))
                faded = last_chunk[-fade_samples:].copy()
                faded *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                await asyncio.to_thread(stream.write, faded)
            await asyncio.to_thread(stream.write, np.zeros(128, dtype=np.float32))

        return JSONResponse({
            "status": "success", "job_id": job.job_id,
            "chunks_played": chunk_count,
            "duration_seconds": round(time.time() - playback_start, 2),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Playback failed: {e}")
    finally:
        job.cancel_event.set()
        if CURRENT_JOB is job:
            CURRENT_JOB = None


@app.post("/v1/audio/speech/cancel")
async def cancel_generation():
    if CURRENT_JOB is None:
        return JSONResponse({"status": "success", "message": "No generation in progress"})
    CURRENT_JOB.cancel_event.set()
    return JSONResponse({"status": "success", "message": f"Cancelled Job {CURRENT_JOB.job_id}"})


@app.post("/v1/voices")
async def create_voice(request: CreateVoiceRequest):
    name = request.voice_name
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid voice name")
    if is_default_voice(name):
        raise HTTPException(status_code=403, detail=f"'{name}' is a system voice")

    npy_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}.npy")
    if os.path.exists(npy_path) and not request.replace:
        raise HTTPException(status_code=409, detail=f"'{name}' exists. Set replace=True.")

    prompt_text_val = request.prompt_text
    if os.path.isfile(prompt_text_val):
        with open(prompt_text_val, "r", encoding="utf-8") as f:
            prompt_text_val = f.read()

    if not os.path.exists(request.prompt_wav_path):
        raise HTTPException(status_code=400, detail=f"Audio not found: {request.prompt_wav_path}")

    try:
        patches = generator.vae_encoder.encode_wav(request.prompt_wav_path)
        os.makedirs(CUSTOM_VOICE_CACHE_DIR, exist_ok=True)
        np.save(npy_path, patches)
        txt_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text_val)
        return {"status": "success", "message": f"Voice '{name}' created."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {e}")


@app.get("/voices")
async def get_available_voices():
    voices = load_available_voices()
    return {"voices": voices, "count": len(voices),
            "custom_cache_directory": CUSTOM_VOICE_CACHE_DIR}


@app.get("/health")
async def health_check():
    is_processing = CURRENT_JOB is not None
    return {
        "status": "healthy", "is_processing": is_processing,
        "current_job_id": CURRENT_JOB.job_id if is_processing else None,
        "model": "voxcpm2",
    }


def main():
    parser = argparse.ArgumentParser(description="VoxCPM2 TTS Server")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--cache-dir", type=str, default=os.path.expanduser("~/.cache/ane_tts"))
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local path to CoreML model directory. If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--quantized-model-dir",
        type=str,
        default=None,
        help="Path to quantized W8 model overrides (base_lm, residual_lm, locdit).",
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        default=None,
        help="Path to safetensors embedding file/dir. Defaults to model-dir.",
    )
    parser.add_argument(
        "--no-hot-swap",
        action="store_true",
        default=False,
        help=(
            "With experimental fast LM prefill, keep both prefill and decode "
            "functions loaded instead of hot-swapping them."
        ),
    )
    parser.add_argument(
        "--lm-prefill-chunk-size",
        type=int,
        choices=(1, 4, 128),
        default=None,
        help=(
            "LM chunk length for prompt prefill. By default fp16 split packages "
            "use their native s4 chunking, while W8 multifunction packages use "
            "stable length_1. Use 128 only for experimental W8 fast prefill."
        ),
    )
    parser.add_argument(
        "--base-lm-splits",
        type=int,
        default=2,
        help=(
            "Number of fp16 BaseLM split packages under --model-dir. "
            "VoxCPM2 fp16 should use 2."
        ),
    )
    parser.add_argument(
        "--quantized-lm-mode",
        type=str,
        choices=("auto", "split", "multifunction"),
        default="auto",
        help=(
            "Quantized LM selection policy for --quantized-model-dir: "
            "'auto' prefers split when present, 'split' forces split packages, "
            "'multifunction' forces multifunction LM packages."
        ),
    )
    parser.add_argument(
        "--quantized-components",
        type=str,
        default="auto",
        help=(
            "Comma-separated model components to source from --quantized-model-dir. "
            "Supported: base_lm,residual_lm,locdit,vae_encoder,feat_encoder,"
            "vae_decoder,fsq,projections. Default 'auto' = base_lm,residual_lm,locdit."
        ),
    )
    parser.add_argument(
        "--compiled-fallback-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing .mlmodelc runtime artifacts. "
            "When a selected model path resolves to .mlpackage, loader will "
            "try <compiled-fallback-dir>/<basename>.mlmodelc before fallback."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debug logging for generation pipeline.",
    )
    args = parser.parse_args()

    global CUSTOM_VOICE_CACHE_DIR
    CUSTOM_VOICE_CACHE_DIR = args.cache_dir
    os.makedirs(CUSTOM_VOICE_CACHE_DIR, exist_ok=True)

    load_model(
        model_dir=args.model_dir,
        quantized_model_dir=args.quantized_model_dir,
        embedding_path=args.embedding_path,
        hot_swap=not args.no_hot_swap,
        lm_prefill_chunk_size=args.lm_prefill_chunk_size,
        base_lm_splits=args.base_lm_splits,
        quantized_lm_mode=args.quantized_lm_mode,
        quantized_components=args.quantized_components,
        compiled_fallback_dir=args.compiled_fallback_dir,
        debug=args.debug,
    )

    print(f"🚀 Starting VoxCPM2 server on {args.host}:{args.port}")
    print(f"   Custom cache: {CUSTOM_VOICE_CACHE_DIR}")
    print(f"   Voices: {len(load_available_voices())}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
