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
import argparse
import ftfy

from .generator import VoxCPM2Generator
from .metrics import (
    GenerationJob,
    GenerationMetricEvent,
    _handle_metric_event,
    _update_live_rtf,
    _update_final_rtf,
    _print_final_rtf_summary,
    _mark_first_byte,
    _print_final_metrics,
)
import voxcpmane.metrics as metrics

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False


REPO_ID = "seba/VoxCPM2-ANE"
MODEL_PATH_PREFIX = ""
VOICE_CACHE_DIR = ""
CUSTOM_VOICE_CACHE_DIR = os.path.expanduser("~/.cache/ane_tts")
LM_MULTIFUNCTION_PREFILL_LENGTHS = (1, 8, 16, 32, 64, 128)
RAW_AUDIO_FORMATS = {"wav", "flac"}
PYDUB_AUDIO_FORMATS = {
    "mp3": ("mp3", "audio/mpeg"),
    "opus": ("opus", "audio/opus"),
    "ogg": ("ogg", "audio/ogg"),
    "aac": ("adts", "audio/aac"),
}

generator: VoxCPM2Generator = None  # type: ignore[assignment]
text_normalizer = None


def resolve_and_compile_path(
    path_str: str | None, compile_and_save: bool
) -> str | None:
    """Helper to resolve a model package path, prioritize compiled counterparts,
    and optionally compile and save models dynamically on the fly.
    """
    if not path_str:
        return None
    path = pathlib.Path(path_str)
    if path.suffix == ".mlpackage":
        compiled_path = path.with_suffix(".mlmodelc")
        if compiled_path.exists():
            print(f"   ✓ Prioritizing compiled model: {compiled_path}")
            return str(compiled_path)
        if compile_and_save:
            try:
                import shutil
                import coremltools as ct

                print(f"📦 Compiling and saving model: {path} -> {compiled_path}")
                model = ct.models.MLModel(str(path))
                temp_compiled = model.get_compiled_model_path()
                shutil.copytree(temp_compiled, str(compiled_path), dirs_exist_ok=True)
                print(f"✅ Saved compiled model to {compiled_path}")
                return str(compiled_path)
            except Exception as e:
                print(f"⚠️ Failed to compile and save {path}: {e}")
    return path_str


def load_model(
    model_dir: str | None = None,
    embedding_path: str | None = None,
    lm_mode: str = "hot-swap",
    lm_prefill_chunk_size: int | None = 128,
    lm_async_decode_load: bool = False,
    base_lm_splits: int = 2,
    compiled_fallback_dir: str | None = None,
    vae_early_decode_steps: int = 0,
    vae_batch_decode_steps: int = 1,
    base_lm_path: list[str] | None = None,
    residual_lm_path: str | None = None,
    locdit_path: str | None = None,
    vae_encoder_path: str | None = None,
    feat_encoder_path: str | None = None,
    vae_decoder_path: str | None = None,
    fsq_path: str | None = None,
    projections_path: str | None = None,
    compile_and_save: bool = False,
):
    """Load VoxCPM2Generator. Called once from main()."""
    global generator, MODEL_PATH_PREFIX, VOICE_CACHE_DIR

    if model_dir is not None:
        MODEL_PATH_PREFIX = os.path.abspath(model_dir)
        print(f"📂 Using local model directory: {MODEL_PATH_PREFIX}")
    else:
        print(f"🚀 Downloading models from HuggingFace: {REPO_ID}")
        MODEL_PATH_PREFIX = snapshot_download(repo_id=REPO_ID)

    VOICE_CACHE_DIR = os.path.join(MODEL_PATH_PREFIX, "caches")

    # Helper to resolve / compile paths
    def get_path(custom_path: str | None, default_filename: str) -> str:
        target = (
            custom_path
            if custom_path
            else os.path.join(MODEL_PATH_PREFIX, default_filename)
        )
        return resolve_and_compile_path(target, compile_and_save)

    # 1. Resolve BaseLM package(s) and splits
    base_lm_split_paths = None
    resolved_base_lm_path = None

    if base_lm_path:
        # Check for split package sibling auto-detection if only one path is provided
        if len(base_lm_path) == 1:
            p_obj = pathlib.Path(base_lm_path[0])
            stem = p_obj.name.rsplit(".", 1)[0]
            splits = sorted(
                p_obj.parent.glob(f"{stem}_part*_of_*.mlpackage")
            ) or sorted(p_obj.parent.glob(f"{stem}_part*_of_*.mlmodelc"))
            if splits:
                base_lm_path = [str(x) for x in splits]

        # Resolve all split paths
        resolved_paths = [
            resolve_and_compile_path(p, compile_and_save) for p in base_lm_path
        ]
        if len(resolved_paths) > 1:
            base_lm_split_paths = resolved_paths
            base_lm_splits = len(resolved_paths)
            print(f"   ✓ base_lm split overrides: {base_lm_splits} packages")
        else:
            resolved_base_lm_path = resolved_paths[0]
            base_lm_splits = 1
            print(f"   ✓ base_lm override: {resolved_base_lm_path}")
    elif base_lm_splits > 1:
        base_lm_split_paths = [
            resolve_and_compile_path(
                os.path.join(
                    MODEL_PATH_PREFIX,
                    f"base_lm_s4_part{i}_of_{base_lm_splits}.mlpackage",
                ),
                compile_and_save,
            )
            for i in range(base_lm_splits)
        ]
    else:
        resolved_base_lm_path = resolve_and_compile_path(
            os.path.join(MODEL_PATH_PREFIX, "base_lm_s4.mlpackage"), compile_and_save
        )

    # 2. Resolve other model components
    resolved_residual_lm_path = get_path(
        residual_lm_path, "residual_lm_fused_s4.mlpackage"
    )
    resolved_locdit_path = get_path(locdit_path, "locdit_p4_c4.mlpackage")
    resolved_vae_encoder_path = get_path(
        vae_encoder_path, "audio_vae_encoder.mlpackage"
    )
    resolved_feat_encoder_path = get_path(feat_encoder_path, "feat_encoder.mlpackage")
    resolved_vae_decoder_path = get_path(
        vae_decoder_path, "audio_vae_decoder_lf4.mlpackage"
    )
    resolved_fsq_path = get_path(fsq_path, "fsq_s4.mlpackage")
    resolved_projections_path = get_path(projections_path, "projections.mlpackage")

    # 3. Setup generator kwargs
    generator_model_dir = MODEL_PATH_PREFIX
    gen_kwargs = {
        "base_lm_splits": int(base_lm_splits),
        "residual_lm_model_path": resolved_residual_lm_path,
        "locdit_model_path": resolved_locdit_path,
        "vae_encoder_model_path": resolved_vae_encoder_path,
        "feat_encoder_model_path": resolved_feat_encoder_path,
        "vae_decoder_model_path": resolved_vae_decoder_path,
        "fsq_model_path": resolved_fsq_path,
        "projections_model_path": resolved_projections_path,
        "vae_early_decode_steps": vae_early_decode_steps,
        "vae_batch_decode_steps": vae_batch_decode_steps,
    }

    if resolved_base_lm_path:
        gen_kwargs["base_lm_model_path"] = resolved_base_lm_path
    if base_lm_split_paths:
        gen_kwargs["base_lm_split_model_paths"] = base_lm_split_paths

    if embedding_path:
        gen_kwargs["embedding_safetensors_path"] = embedding_path
    if compiled_fallback_dir:
        gen_kwargs["compiled_fallback_dir"] = os.path.abspath(compiled_fallback_dir)

    # 4. Configure LM prefill and decode mode behavior
    if lm_prefill_chunk_size is not None:
        chunk_size = int(lm_prefill_chunk_size)
        if chunk_size not in LM_MULTIFUNCTION_PREFILL_LENGTHS:
            raise ValueError(
                f"lm_prefill_chunk_size must be one of {LM_MULTIFUNCTION_PREFILL_LENGTHS}; "
                f"got {chunk_size}"
            )
        gen_kwargs["lm_prefill_chunk_size"] = chunk_size

        if chunk_size == 1:
            gen_kwargs["lm_unload_inactive_functions"] = False
            gen_kwargs["lm_restrict_to_preload"] = False
            is_residual_multifunction = (
                residual_lm_path
                and "multifunction" in os.path.basename(residual_lm_path)
            )
            if is_residual_multifunction:
                gen_kwargs["lm_preload_chunk_sizes"] = [1]
        elif lm_mode == "single-length":
            gen_kwargs["lm_unload_inactive_functions"] = False
            gen_kwargs["lm_restrict_to_preload"] = True
            gen_kwargs["lm_preload_chunk_sizes"] = [chunk_size]
            gen_kwargs["lm_async_decode_load"] = False
        elif lm_mode == "preload":
            gen_kwargs["lm_unload_inactive_functions"] = True
            gen_kwargs["lm_restrict_to_preload"] = False
            gen_kwargs["lm_idle_prefill_chunk_size"] = chunk_size
            gen_kwargs["lm_preload_chunk_sizes"] = [1, chunk_size]
            gen_kwargs["lm_keep_decode_function_loaded"] = True
            gen_kwargs["lm_async_decode_load"] = False
        elif lm_mode == "always-loaded":
            gen_kwargs["lm_unload_inactive_functions"] = False
            gen_kwargs["lm_restrict_to_preload"] = False
            gen_kwargs["lm_preload_chunk_sizes"] = [1, chunk_size]
            gen_kwargs["lm_keep_decode_function_loaded"] = True
            gen_kwargs["lm_async_decode_load"] = False
        elif lm_mode == "hot-swap":
            gen_kwargs["lm_unload_inactive_functions"] = True
            gen_kwargs["lm_restrict_to_preload"] = False
            gen_kwargs["lm_idle_prefill_chunk_size"] = chunk_size
            gen_kwargs["lm_preload_chunk_sizes"] = [chunk_size]
            gen_kwargs["lm_keep_decode_function_loaded"] = False
            gen_kwargs["lm_async_decode_load"] = bool(lm_async_decode_load)
        else:
            raise ValueError(f"Unknown lm_mode: {lm_mode}")

    # Convert string paths to pathlib.Path objects for generator initialization
    path_keys = [
        "embedding_safetensors_path",
        "base_lm_model_path",
        "residual_lm_model_path",
        "locdit_model_path",
        "vae_encoder_model_path",
        "feat_encoder_model_path",
        "vae_decoder_model_path",
        "fsq_model_path",
        "projections_model_path",
        "compiled_fallback_dir",
    ]
    for key in path_keys:
        if gen_kwargs.get(key) is not None:
            gen_kwargs[key] = pathlib.Path(gen_kwargs[key])

    if gen_kwargs.get("base_lm_split_model_paths") is not None:
        gen_kwargs["base_lm_split_model_paths"] = [
            pathlib.Path(p) for p in gen_kwargs["base_lm_split_model_paths"]
        ]

    print("Loading CoreML models via VoxCPM2Generator...")
    generator = VoxCPM2Generator(pathlib.Path(generator_model_dir), **gen_kwargs)
    generator.preload_tokenizer()
    print("✅ Models loaded successfully.")


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

                def emit_metric(kind, values):
                    job.output_queue.put(GenerationMetricEvent(kind, values))

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
                    seed=job.request.seed,
                    normalize=job.request.normalize,
                    cancellation_event=job.cancel_event,
                    metrics_callback=emit_metric,
                )
                chunk_count = 0
                for chunk in audio_gen:
                    if job.cancel_event.is_set():
                        print(
                            f"🛑 Job {job.job_id}: cancelled after {chunk_count} chunks",
                            flush=True,
                        )
                        break
                    job.output_queue.put(chunk)
                    chunk_count += 1
                print(
                    f"✅ Job {job.job_id}: done, {chunk_count} chunks produced",
                    flush=True,
                )
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
    seed: Optional[int] = None
    normalize: Optional[bool] = False


class PlaybackRequest(SpeechRequest):
    show_progress: Optional[bool] = True


class CreateVoiceRequest(BaseModel):
    voice_name: str
    prompt_wav_path: str
    prompt_text: str
    replace: Optional[bool] = False


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
        raise HTTPException(
            status_code=400, detail="max_length must be between 1 and 4096"
        )
    if not (0.0 <= cfg_value <= 10.0):
        raise HTTPException(
            status_code=400, detail="cfg_value must be between 0.0 and 10.0"
        )
    if not (0 < inference_timesteps <= 100):
        raise HTTPException(
            status_code=400, detail="inference_timesteps must be between 1 and 100"
        )


def normalize_apple_punctuation(text):
    """Convert Apple smart/typographic punctuation to ASCII equivalents."""
    table = str.maketrans(
        {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\u2022": "*",
            "\u00a0": " ",
            "\u201a": ",",
            "\u201e": '"',
            "\u2039": "<",
            "\u203a": ">",
        }
    )
    return text.translate(table)


def audio_float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float audio to PCM16 without wrapping out-of-range samples."""
    audio_arr = np.asarray(audio, dtype=np.float32)
    return (np.clip(audio_arr, -1.0, 1.0) * 32767.0).astype(np.int16)


def validate_audio_format(audio_format: str) -> None:
    if audio_format in RAW_AUDIO_FORMATS:
        return
    if not PYDUB_AVAILABLE:
        raise HTTPException(
            status_code=501, detail=f"Format '{audio_format}' requires pydub."
        )
    if audio_format not in PYDUB_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format: {audio_format}"
        )


def encode_audio_response(audio: np.ndarray, audio_format: str) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    if audio_format in RAW_AUDIO_FORMATS:
        soundfile.write(buffer, audio, SAMPLE_RATE, format=audio_format)
        media_type = f"audio/{audio_format}"
    else:
        export_format, media_type = PYDUB_AUDIO_FORMATS[audio_format]
        pcm = audio_float_to_int16(audio)
        segment = AudioSegment(
            data=pcm.tobytes(), sample_width=2, frame_rate=SAMPLE_RATE, channels=1
        )
        segment.export(buffer, format=export_format)
    return buffer.getvalue(), media_type


def submit_generation_job(request: "SpeechRequest") -> GenerationJob:
    global JOB_COUNTER
    JOB_COUNTER += 1
    job = GenerationJob(
        request, queue.Queue(maxsize=1024), threading.Event(), JOB_COUNTER
    )
    try:
        GENERATION_QUEUE.put_nowait(job)
    except queue.Full:
        raise HTTPException(status_code=429, detail="Server is busy")
    return job


def record_audio_chunk(job: GenerationJob, chunk: np.ndarray) -> int:
    chunk_samples = int(chunk.shape[0])
    job.audio_samples_sent += chunk_samples
    job.inference_loops_sent += 1
    _update_live_rtf(job, chunk_samples)
    _update_final_rtf(job, chunk_samples)
    return chunk_samples


def finish_generation_job(job: GenerationJob, status: str = "stopped") -> None:
    job.cancel_event.set()
    _print_final_rtf_summary(job)
    if not job.final_printed:
        _print_final_metrics(job, status)


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
    seed=None,
    normalize=False,
    cancellation_event=None,
    metrics_callback=None,
):
    global text_normalizer

    if cancellation_event is None:
        cancellation_event = threading.Event()

    validate_voice_parameters(max_length, cfg_value, inference_timesteps)

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

    control = (control_instruction or "").strip()
    control = re.sub(r"[()（）]", "", control).strip()
    final_text = f"({control}){text}" if control else text

    reference_path = (reference_wav_path or "").strip() or None
    prompt_path = (prompt_wav_path or "").strip() or None
    prompt_text_clean = (prompt_text or "").strip()

    prompt_audio_feat = None
    voice_name = (voice or "").strip() or None
    if voice_name is not None:
        if prompt_path is not None:
            raise ValueError(
                "voice and prompt_wav_path are both continuation prompts; use only one"
            )
        audio_cache = load_voice_cache(voice_name)
        voice_prompt_text = get_voice_prompt_text(voice_name)
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
        seed=seed,
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
    print(
        f"🎤 generate_audio_chunks: mode={mode}, text={final_text[:80]!r}..., "
        f"max_len={effective_max_length}, cfg={cfg_value}, steps={inference_timesteps}",
        flush=True,
    )
    import sys

    sys.stdout.flush()
    sys.stderr.flush()

    try:
        chunk_idx = 0
        for audio_chunk in generator.generate_streaming(
            **gen_kwargs,
            metrics_callback=metrics_callback,
        ):
            if cancellation_event.is_set():
                print(
                    f"🛑 generate_audio_chunks: cancelled at chunk {chunk_idx}",
                    flush=True,
                )
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
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp:
                        tmp_wav_path = tmp.name
                    AudioSegment.from_file(audio_path).export(
                        tmp_wav_path, format="wav"
                    )
                    processing_path = tmp_wav_path

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


async def poll_queue_for_chunks(output_queue, poll_interval=0.005, on_metric=None):
    while True:
        try:
            item = output_queue.get_nowait()
            if item is None:
                break
            elif isinstance(item, Exception):
                raise item
            elif isinstance(item, GenerationMetricEvent):
                if on_metric is not None:
                    on_metric(item)
            else:
                yield item
        except queue.Empty:
            await asyncio.sleep(poll_interval)


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    audio_format = (request.response_format or "wav").lower()
    validate_audio_format(audio_format)

    job = submit_generation_job(request)
    all_chunks = []
    try:
        async for chunk in poll_queue_for_chunks(
            job.output_queue,
            on_metric=lambda event: _handle_metric_event(job, event),
        ):
            all_chunks.append(chunk)
            record_audio_chunk(job, chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        job.cancel_event.set()
        _print_final_rtf_summary(job)

    if not all_chunks:
        raise HTTPException(status_code=500, detail="No audio produced")

    full_audio = np.concatenate(all_chunks)
    content, media_type = encode_audio_response(full_audio, audio_format)
    _mark_first_byte(job)
    if job.generation_seconds is not None:
        _print_final_metrics(job)
    return Response(content=content, media_type=media_type)


@app.post("/v1/audio/speech/stream")
async def stream_speech(request: SpeechRequest):
    job = submit_generation_job(request)

    async def audio_stream():
        try:
            async for chunk in poll_queue_for_chunks(
                job.output_queue,
                on_metric=lambda event: _handle_metric_event(job, event),
            ):
                payload = audio_float_to_int16(chunk).tobytes()
                _mark_first_byte(job)
                record_audio_chunk(job, chunk)
                yield payload
        finally:
            finish_generation_job(job)

    return StreamingResponse(
        audio_stream(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE)},
    )


@app.post("/v1/audio/speech/playback")
async def playback_speech(request: PlaybackRequest):
    global CURRENT_JOB
    job = submit_generation_job(request)
    CURRENT_JOB = job

    playback_start = time.time()
    chunk_count = 0

    try:
        if not sd.query_devices():
            raise HTTPException(status_code=500, detail="No audio output devices")

        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            latency="low",
            blocksize=1024,
        ) as stream:
            last_chunk = None
            async for chunk in poll_queue_for_chunks(
                job.output_queue,
                on_metric=lambda event: _handle_metric_event(job, event),
            ):
                if time.time() - playback_start > 300:
                    raise HTTPException(status_code=500, detail="Playback timeout")
                chunk_count += 1
                last_chunk = chunk
                _mark_first_byte(job)
                record_audio_chunk(job, chunk)
                await asyncio.to_thread(stream.write, chunk)

            # Fade-out to prevent click/pop
            if last_chunk is not None and len(last_chunk) > 100:
                fade_samples = min(int(SAMPLE_RATE * 0.05), len(last_chunk))
                faded = last_chunk[-fade_samples:].copy()
                faded *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                await asyncio.to_thread(stream.write, faded)
            await asyncio.to_thread(stream.write, np.zeros(128, dtype=np.float32))

        return JSONResponse(
            {
                "status": "success",
                "job_id": job.job_id,
                "chunks_played": chunk_count,
                "duration_seconds": round(time.time() - playback_start, 2),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Playback failed: {e}")
    finally:
        finish_generation_job(job)
        if CURRENT_JOB is job:
            CURRENT_JOB = None


@app.post("/v1/audio/speech/cancel")
async def cancel_generation():
    if CURRENT_JOB is None:
        return JSONResponse(
            {"status": "success", "message": "No generation in progress"}
        )
    CURRENT_JOB.cancel_event.set()
    return JSONResponse(
        {"status": "success", "message": f"Cancelled Job {CURRENT_JOB.job_id}"}
    )


@app.post("/v1/voices")
async def create_voice(request: CreateVoiceRequest):
    name = request.voice_name
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid voice name")
    if is_default_voice(name):
        raise HTTPException(status_code=403, detail=f"'{name}' is a system voice")

    npy_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}.npy")
    if os.path.exists(npy_path) and not request.replace:
        raise HTTPException(
            status_code=409, detail=f"'{name}' exists. Set replace=True."
        )

    prompt_text_val = request.prompt_text
    if os.path.isfile(prompt_text_val):
        with open(prompt_text_val, "r", encoding="utf-8") as f:
            prompt_text_val = f.read()

    if not os.path.exists(request.prompt_wav_path):
        raise HTTPException(
            status_code=400, detail=f"Audio not found: {request.prompt_wav_path}"
        )

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
    return {
        "voices": voices,
        "count": len(voices),
        "custom_cache_directory": CUSTOM_VOICE_CACHE_DIR,
    }


@app.get("/health")
async def health_check():
    is_processing = CURRENT_JOB is not None
    return {
        "status": "healthy",
        "is_processing": is_processing,
        "current_job_id": CURRENT_JOB.job_id if is_processing else None,
        "model": "voxcpm2",
    }


def main():
    parser = argparse.ArgumentParser(description="VoxCPM2 TTS Server")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--cache-dir", type=str, default=os.path.expanduser("~/.cache/ane_tts")
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local path to CoreML model directory. If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        default=None,
        help="Path to safetensors embedding file/dir. Defaults to model-dir.",
    )
    parser.add_argument(
        "--lm-mode",
        type=str,
        choices=["single-length", "preload", "always-loaded", "hot-swap"],
        default="hot-swap",
        help=(
            "Unified LM prefill and decode mode behavior: 'single-length' (use same length for "
            "prefill and decode, restrict to selected prefill size), 'preload' (preload length 1 "
            "and prefill chunk size at startup, unload prefill size during decode, reload on idle), "
            "'always-loaded' (preload length 1 and prefill chunk size and keep both resident), "
            "or 'hot-swap' (preload chunk size while idle, swap to length 1 for decode, swap back "
            "after completion)."
        ),
    )
    parser.add_argument(
        "--lm-prefill-chunk-size",
        type=int,
        choices=LM_MULTIFUNCTION_PREFILL_LENGTHS,
        default=128,
        help=(
            "LM chunk length for prompt prefill (default: 128). Available "
            "values are 1, 8, 16, 32, 64, and 128; these lengths can also "
            "be used with --lm-mode single-length."
        ),
    )
    parser.add_argument(
        "--lm-async-decode-load",
        action="store_true",
        default=False,
        help=(
            "In LM hot-swap mode, start loading length_1 decode handles after "
            "prefill and overlap that load with projections, LocDiT, and VAE "
            "work before the first LM decode step."
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
        "--live-rtf",
        choices=("off", "live", "final"),
        default="off",
        help=(
            "RTF metrics display mode: 'off' disables, 'live' prints an "
            "in-place updating line during generation, 'final' prints a "
            "single summary line when the request completes."
        ),
    )
    parser.add_argument(
        "--vae-early-decode-steps",
        type=int,
        default=0,
        help=(
            "Number of initial AR steps where the VAE decoder runs immediately "
            "(one chunk per step) for low TTFB. After this many steps, "
            "decoding switches to batch mode. 0 = always immediate (default)."
        ),
    )
    parser.add_argument(
        "--vae-batch-decode-steps",
        type=int,
        default=1,
        help=(
            "Number of AR steps to accumulate before batch-decoding audio "
            "after the early-decode phase. Requires a RangeDim VAE decoder "
            "model. 1 = no batching (default)."
        ),
    )
    # New individual package paths
    parser.add_argument(
        "--base-lm-path",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to BaseLM model package(s). Can accept multiple split parts.",
    )
    parser.add_argument(
        "--residual-lm-path",
        type=str,
        default=None,
        help="Path to ResidualLM model package.",
    )
    parser.add_argument(
        "--locdit-path",
        type=str,
        default=None,
        help="Path to LocDiT model package.",
    )
    parser.add_argument(
        "--vae-encoder-path",
        type=str,
        default=None,
        help="Path to Audio VAE Encoder model package.",
    )
    parser.add_argument(
        "--feat-encoder-path",
        type=str,
        default=None,
        help="Path to Feature Encoder model package.",
    )
    parser.add_argument(
        "--vae-decoder-path",
        type=str,
        default=None,
        help="Path to Audio VAE Decoder model package.",
    )
    parser.add_argument(
        "--fsq-path",
        type=str,
        default=None,
        help="Path to FSQ model package.",
    )
    parser.add_argument(
        "--projections-path",
        type=str,
        default=None,
        help="Path to Projections model package.",
    )
    parser.add_argument(
        "--compile-and-save",
        action="store_true",
        default=False,
        help="Compile CoreML .mlpackage files into .mlmodelc on the fly if they do not exist.",
    )

    args = parser.parse_args()

    global CUSTOM_VOICE_CACHE_DIR
    CUSTOM_VOICE_CACHE_DIR = args.cache_dir
    metrics.LIVE_RTF_METRICS = str(args.live_rtf)
    os.makedirs(CUSTOM_VOICE_CACHE_DIR, exist_ok=True)

    load_model(
        model_dir=args.model_dir,
        embedding_path=args.embedding_path,
        lm_mode=args.lm_mode,
        lm_prefill_chunk_size=args.lm_prefill_chunk_size,
        lm_async_decode_load=args.lm_async_decode_load,
        base_lm_splits=args.base_lm_splits,
        compiled_fallback_dir=args.compiled_fallback_dir,
        vae_early_decode_steps=args.vae_early_decode_steps,
        vae_batch_decode_steps=args.vae_batch_decode_steps,
        base_lm_path=args.base_lm_path,
        residual_lm_path=args.residual_lm_path,
        locdit_path=args.locdit_path,
        vae_encoder_path=args.vae_encoder_path,
        feat_encoder_path=args.feat_encoder_path,
        vae_decoder_path=args.vae_decoder_path,
        fsq_path=args.fsq_path,
        projections_path=args.projections_path,
        compile_and_save=args.compile_and_save,
    )

    print(f"🚀 Starting VoxCPM2 server on {args.host}:{args.port}")
    print(f"   Custom cache: {CUSTOM_VOICE_CACHE_DIR}")
    print(f"   Voices: {len(load_available_voices())}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
