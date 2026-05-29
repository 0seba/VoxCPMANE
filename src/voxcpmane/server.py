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
import soundfile
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


REPO_ID = "seba/VoxCPM2ANE-Preview"
MODEL_PATH_PREFIX = ""
VOICE_CACHE_DIR = ""
VOICE_CACHE_DIRS: list[str] = []
CUSTOM_VOICE_CACHE_DIR = os.path.expanduser("~/.cache/ane_tts")
PROMPT_DECODE_CONTEXT_PATCHES = 3
LM_MULTIFUNCTION_PREFILL_LENGTHS = (1, 8, 16, 32, 64, 128)
RAW_AUDIO_FORMATS = {"wav", "flac"}
PYDUB_AUDIO_FORMATS = {
    "mp3": ("mp3", "audio/mpeg"),
    "opus": ("opus", "audio/opus"),
    "ogg": ("ogg", "audio/ogg"),
    "aac": ("adts", "audio/aac"),
}


def default_lm_prefill_chunk_size(lm_mode: str) -> int:
    return 128 if lm_mode in {"preload", "hot-swap"} else 16


COMPONENT_PATH_SPECS = (
    (
        "residual_lm_model_path",
        "residual_lm_path",
        ("residual_lm_fused_multifunction.mlpackage", "residual_lm_fused_s4.mlpackage"),
    ),
    ("locdit_model_path", "locdit_path", ("locdit_p4_c4.mlpackage",)),
    (
        "vae_encoder_model_path", "vae_encoder_path", ("audio_vae_encoder.mlpackage",)
    ),
    ("feat_encoder_model_path", "feat_encoder_path", ("feat_encoder.mlpackage",)),
    (
        "vae_decoder_model_path", "vae_decoder_path", ("audio_vae_decoder_lf4.mlpackage",)
    ),
    ("fsq_model_path", "fsq_path", ("fsq_s4.mlpackage",)),
    ("projections_model_path", "projections_path", ("projections.mlpackage",)),
)

generator: VoxCPM2Generator = None  # type: ignore[assignment]
text_normalizer = None
VOICE_FEATURE_CACHE_MEMORY: dict[tuple[str, str], np.ndarray] = {}


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


def default_model_path(model_dir: str, *filenames: str) -> str:
    for filename in filenames:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            return path
        if filename.endswith(".mlpackage"):
            compiled = str(pathlib.Path(path).with_suffix(".mlmodelc"))
            if os.path.exists(compiled):
                return path
    return os.path.join(model_dir, filenames[0])


def resolve_component_path(
    model_dir: str,
    override: str | None,
    filenames: tuple[str, ...],
    compile_and_save: bool,
) -> str | None:
    return resolve_and_compile_path(
        override or default_model_path(model_dir, *filenames),
        compile_and_save,
    )


def resolve_base_lm_paths(
    model_dir: str,
    base_lm_path: list[str] | None,
    base_lm_splits: int,
    compile_and_save: bool,
) -> tuple[list[str] | None, str | None, int]:
    split_paths = None
    resolved_path = None
    default_base_lm = default_model_path(model_dir, "base_lm_multifunction.mlpackage")

    if base_lm_path:
        paths = list(base_lm_path)
        if len(paths) == 1:
            p_obj = pathlib.Path(paths[0])
            stem = p_obj.name.rsplit(".", 1)[0]
            splits = sorted(p_obj.parent.glob(f"{stem}_part*_of_*.mlpackage")) or sorted(
                p_obj.parent.glob(f"{stem}_part*_of_*.mlmodelc")
            )
            if splits:
                paths = [str(x) for x in splits]

        resolved_paths = [resolve_and_compile_path(p, compile_and_save) for p in paths]
        if len(resolved_paths) > 1:
            split_paths = resolved_paths
            base_lm_splits = len(resolved_paths)
            print(f"   ✓ base_lm split overrides: {base_lm_splits} packages")
        else:
            resolved_path = resolved_paths[0]
            base_lm_splits = 1
            print(f"   ✓ base_lm override: {resolved_path}")
    elif (
        pathlib.Path(default_base_lm).exists()
        or pathlib.Path(default_base_lm).with_suffix(".mlmodelc").exists()
    ):
        resolved_path = resolve_and_compile_path(default_base_lm, compile_and_save)
        base_lm_splits = 1
        print(f"   ✓ base_lm default: {resolved_path}")
    elif base_lm_splits > 1:
        split_paths = [
            resolve_and_compile_path(
                os.path.join(
                    model_dir,
                    f"base_lm_s4_part{i}_of_{base_lm_splits}.mlpackage",
                ),
                compile_and_save,
            )
            for i in range(base_lm_splits)
        ]
    else:
        resolved_path = resolve_and_compile_path(
            default_model_path(model_dir, "base_lm_s4.mlpackage"),
            compile_and_save,
        )

    return split_paths, resolved_path, base_lm_splits


def lm_mode_kwargs(
    lm_mode: str,
    chunk_size: int,
    *,
    residual_lm_path: str | None,
) -> dict:
    if chunk_size == 1:
        kwargs = {
            "lm_unload_inactive_functions": False,
            "lm_restrict_to_preload": False,
        }
        if residual_lm_path and "multifunction" in os.path.basename(residual_lm_path):
            kwargs["lm_preload_chunk_sizes"] = [1]
        return kwargs

    modes = {
        "single-length": {
            "lm_unload_inactive_functions": False,
            "lm_restrict_to_preload": True,
            "lm_preload_chunk_sizes": [chunk_size],
        },
        "preload": {
            "lm_unload_inactive_functions": True,
            "lm_restrict_to_preload": False,
            "lm_idle_prefill_chunk_size": chunk_size,
            "lm_preload_chunk_sizes": [1, chunk_size],
            "lm_keep_decode_function_loaded": True,
        },
        "always-loaded": {
            "lm_unload_inactive_functions": False,
            "lm_restrict_to_preload": False,
            "lm_preload_chunk_sizes": [1, chunk_size],
            "lm_keep_decode_function_loaded": True,
        },
        "hot-swap": {
            "lm_unload_inactive_functions": True,
            "lm_restrict_to_preload": False,
            "lm_idle_prefill_chunk_size": chunk_size,
            "lm_preload_chunk_sizes": [chunk_size],
            "lm_keep_decode_function_loaded": False,
        },
    }
    try:
        return dict(modes[lm_mode])
    except KeyError as exc:
        raise ValueError(f"Unknown lm_mode: {lm_mode}") from exc


def pathify_gen_kwargs(gen_kwargs: dict) -> None:
    path_keys = {
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
    }
    for key in path_keys:
        if gen_kwargs.get(key) is not None:
            gen_kwargs[key] = pathlib.Path(gen_kwargs[key])
    if gen_kwargs.get("base_lm_split_model_paths") is not None:
        gen_kwargs["base_lm_split_model_paths"] = [
            pathlib.Path(p) for p in gen_kwargs["base_lm_split_model_paths"]
        ]


def load_model(
    model_dir: str | None = None,
    repo_id: str = REPO_ID,
    included_voice_cache_dir: str | None = None,
    embedding_path: str | None = None,
    lm_mode: str = "single-length",
    lm_prefill_chunk_size: int | None = None,
    base_lm_splits: int = 2,
    compiled_fallback_dir: str | None = None,
    vae_early_decode_steps: int = 16,
    vae_batch_decode_steps: int = 4,
    base_lm_path: list[str] | None = None,
    residual_lm_path: str | None = None,
    locdit_path: str | None = None,
    vae_encoder_path: str | None = None,
    feat_encoder_path: str | None = None,
    vae_decoder_path: str | None = None,
    fsq_path: str | None = None,
    projections_path: str | None = None,
    compile_and_save: bool = False,
    startup_warmup_repeats: int = 5,
):
    """Load VoxCPM2Generator. Called once from main()."""
    global generator, MODEL_PATH_PREFIX, VOICE_CACHE_DIR, VOICE_CACHE_DIRS

    if model_dir is not None:
        MODEL_PATH_PREFIX = os.path.abspath(model_dir)
        print(f"📂 Using local model directory: {MODEL_PATH_PREFIX}")
    else:
        print(f"🚀 Downloading models from HuggingFace: {repo_id}")
        MODEL_PATH_PREFIX = snapshot_download(repo_id=repo_id)

    model_voice_cache_dir = os.path.join(MODEL_PATH_PREFIX, "caches")
    if included_voice_cache_dir is not None:
        VOICE_CACHE_DIR = os.path.abspath(included_voice_cache_dir)
        print(f"🎙️ Using included voice cache directory: {VOICE_CACHE_DIR}")
    elif os.path.isdir(model_voice_cache_dir):
        VOICE_CACHE_DIR = model_voice_cache_dir
    else:
        print(f"🎙️ Downloading included voice caches from HuggingFace: {repo_id}")
        voice_snapshot_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns="caches/*",
        )
        VOICE_CACHE_DIR = os.path.join(voice_snapshot_dir, "caches")
    VOICE_CACHE_DIRS = [VOICE_CACHE_DIR]

    try:
        hf_voice_snapshot_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns="caches/*",
        )
        hf_voice_cache_dir = os.path.join(hf_voice_snapshot_dir, "caches")
        if os.path.isdir(hf_voice_cache_dir) and os.path.abspath(
            hf_voice_cache_dir
        ) not in {os.path.abspath(d) for d in VOICE_CACHE_DIRS}:
            VOICE_CACHE_DIRS.append(hf_voice_cache_dir)
            print(f"🎙️ Using HF voice cache fallback: {hf_voice_cache_dir}")
    except Exception as exc:
        print(f"⚠️ Could not initialize HF voice cache fallback: {exc}")

    base_lm_split_paths, resolved_base_lm_path, base_lm_splits = resolve_base_lm_paths(
        MODEL_PATH_PREFIX,
        base_lm_path,
        base_lm_splits,
        compile_and_save,
    )
    overrides = {
        "residual_lm_path": residual_lm_path,
        "locdit_path": locdit_path,
        "vae_encoder_path": vae_encoder_path,
        "feat_encoder_path": feat_encoder_path,
        "vae_decoder_path": vae_decoder_path,
        "fsq_path": fsq_path,
        "projections_path": projections_path,
    }

    generator_model_dir = MODEL_PATH_PREFIX
    gen_kwargs = {
        "base_lm_splits": int(base_lm_splits),
        "vae_early_decode_steps": vae_early_decode_steps,
        "vae_batch_decode_steps": vae_batch_decode_steps,
    }
    for kwarg, override_key, filenames in COMPONENT_PATH_SPECS:
        gen_kwargs[kwarg] = resolve_component_path(
            MODEL_PATH_PREFIX,
            overrides[override_key],
            filenames,
            compile_and_save,
        )

    if resolved_base_lm_path:
        gen_kwargs["base_lm_model_path"] = resolved_base_lm_path
    if base_lm_split_paths:
        gen_kwargs["base_lm_split_model_paths"] = base_lm_split_paths

    if embedding_path:
        gen_kwargs["embedding_safetensors_path"] = embedding_path
    if compiled_fallback_dir:
        gen_kwargs["compiled_fallback_dir"] = os.path.abspath(compiled_fallback_dir)

    # 4. Configure LM prefill and decode mode behavior
    if lm_prefill_chunk_size is None:
        lm_prefill_chunk_size = default_lm_prefill_chunk_size(lm_mode)
    if lm_prefill_chunk_size is not None:
        chunk_size = int(lm_prefill_chunk_size)
        if chunk_size not in LM_MULTIFUNCTION_PREFILL_LENGTHS:
            raise ValueError(
                f"lm_prefill_chunk_size must be one of {LM_MULTIFUNCTION_PREFILL_LENGTHS}; "
                f"got {chunk_size}"
            )
        gen_kwargs["lm_prefill_chunk_size"] = chunk_size

        gen_kwargs.update(
            lm_mode_kwargs(
                lm_mode,
                chunk_size,
                residual_lm_path=residual_lm_path,
            )
        )
        if lm_mode == "hot-swap" and chunk_size != 1:
            gen_kwargs["lm_startup_decode_warmup_repeats"] = int(
                startup_warmup_repeats
            )

    pathify_gen_kwargs(gen_kwargs)

    print("Loading CoreML models via VoxCPM2Generator...")
    generator = VoxCPM2Generator(pathlib.Path(generator_model_dir), **gen_kwargs)
    generator.preload_tokenizer()
    generator.warmup_startup_models(repeats=startup_warmup_repeats)
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
                    job.request,
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
                if isinstance(e, HTTPException):
                    job.output_queue.put(e)
                elif isinstance(e, (ValueError, FileNotFoundError)):
                    job.output_queue.put(HTTPException(status_code=400, detail=str(e)))
                else:
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
    voice_mode: Optional[str] = "reference"
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


class CreateVoiceRequest(BaseModel):
    voice_name: str
    reference_wav_path: Optional[str] = None
    prompt_text: Optional[str] = ""
    replace: Optional[bool] = False


class VoiceStore:
    @property
    def all_dirs(self) -> list[str]:
        return [*VOICE_CACHE_DIRS, CUSTOM_VOICE_CACHE_DIR]

    def validate(self, voice_name: str) -> str:
        name = voice_name.strip()
        if not name or ".." in name or "/" in name or "\\" in name:
            raise HTTPException(status_code=400, detail="Invalid voice name")
        return name

    def names_from_dir(self, directory: str) -> list[str]:
        if not os.path.exists(directory):
            return []
        return sorted(
            f[: -len(".embed.npy")]
            for f in os.listdir(directory)
            if f.endswith(".embed.npy")
            and not f.endswith(".prompt.embed.npy")
            and os.path.isfile(os.path.join(directory, f))
        )

    def available(self) -> list[str]:
        return sorted({
            name
            for directory in self.all_dirs
            for name in self.names_from_dir(directory)
        })

    def path(self, voice_name: str, suffix: str, directory: str) -> str:
        return os.path.join(directory, f"{voice_name}{suffix}")

    def find(self, voice_name: str, suffix: str) -> str | None:
        return next(
            (
                path
                for directory in self.all_dirs
                for path in [self.path(voice_name, suffix, directory)]
                if os.path.exists(path)
            ),
            None,
        )

    def embed_path(self, voice_name: str, *, prompt: bool = False) -> str | None:
        return self.find(voice_name, ".prompt.embed.npy" if prompt else ".embed.npy")

    def is_default(self, voice_name: str) -> bool:
        return os.path.exists(self.path(voice_name, ".embed.npy", VOICE_CACHE_DIR))

    def is_custom(self, voice_name: str) -> bool:
        return os.path.exists(
            self.path(voice_name, ".embed.npy", CUSTOM_VOICE_CACHE_DIR)
        )

    def prompt_text(self, voice_name: str) -> str | None:
        if self.is_default(voice_name):
            return CACHED_VOICE_TEXT
        txt_path = self.path(voice_name, ".txt", CUSTOM_VOICE_CACHE_DIR)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def lm_prefix_paths(self, voice_name: str) -> tuple[list[str], str]:
        filename = f"{voice_name}.lm_prefix.npz"
        read_paths = []
        for directory in VOICE_CACHE_DIRS:
            path = os.path.join(directory, filename)
            if os.path.exists(path) and path not in read_paths:
                read_paths.append(path)

        custom_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, filename)
        if custom_path not in read_paths:
            read_paths.append(custom_path)
        return read_paths, custom_path

    def remove_lm_prefix_caches(self, voice_name: str) -> list[str]:
        if not os.path.isdir(CUSTOM_VOICE_CACHE_DIR):
            return []
        prefix = f"{voice_name}."
        removed = []
        for filename in os.listdir(CUSTOM_VOICE_CACHE_DIR):
            if not (filename.startswith(prefix) and filename.endswith(".lm_prefix.npz")):
                continue
            path = os.path.join(CUSTOM_VOICE_CACHE_DIR, filename)
            try:
                os.unlink(path)
                removed.append(filename)
            except OSError:
                pass
        return removed


VOICE_STORE = VoiceStore()


def lm_prefix_cache_paths(voice_name: str) -> tuple[list[str], str]:
    read_paths, custom_path = VOICE_STORE.lm_prefix_paths(voice_name)
    included_roots = tuple(os.path.abspath(d) + os.sep for d in VOICE_CACHE_DIRS)
    included_hit = any(
        os.path.abspath(path).startswith(included_roots) for path in read_paths
    )
    print(
        f"🧠 LM prefix cache lookup voice={voice_name} "
        f"included={'hit' if included_hit else 'miss'} paths={read_paths}",
        flush=True,
    )
    return read_paths, custom_path


def encode_voice_feature_cache(audio_feat: np.ndarray) -> np.ndarray:
    audio_feat = np.asarray(audio_feat, dtype=np.float32)
    if audio_feat.ndim != 3:
        raise ValueError(f"voice cache must have rank 3, got {audio_feat.shape}")
    return generator.feat_encoder.encode_patches(audio_feat[None, ...])[0].astype(
        np.float32, copy=False
    )


def load_voice_feature_cache(voice_name: str, *, prompt: bool = False) -> np.ndarray:
    cache_kind = "prompt" if prompt else "reference"
    memory_key = (voice_name, cache_kind)
    cached = VOICE_FEATURE_CACHE_MEMORY.get(memory_key)
    if cached is not None:
        return cached

    embed_path = VOICE_STORE.embed_path(voice_name, prompt=prompt)
    if embed_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found. Available: {VOICE_STORE.available()}",
        )

    embed = np.load(embed_path).astype(np.float32, copy=False)
    if embed.ndim != 2 or embed.shape[1] != generator.hidden_size:
        cache_name = os.path.basename(embed_path)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Voice cache '{cache_name}' is not in the current feature-cache "
                f"format. Expected (T, {generator.hidden_size}), got {embed.shape}. "
                "Regenerate the voice cache with this VoxCPMANE2 version."
            ),
        )
    VOICE_FEATURE_CACHE_MEMORY[memory_key] = embed
    return embed


def load_voice_prompt_cond(voice_name: str) -> np.ndarray:
    cond_path = VOICE_STORE.find(voice_name, ".prompt.cond.npy")
    if cond_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' has no high-similarity prompt condition cache",
        )
    cond = np.load(cond_path).astype(np.float32, copy=False)
    expected = (generator.patch_size, generator.latent_dim)
    if cond.shape not in {expected, (generator.latent_dim, generator.patch_size)}:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Voice '{voice_name}' prompt condition cache has shape {cond.shape}; "
                f"expected {expected}."
            ),
        )
    return cond


def load_voice_prompt_decode_context(voice_name: str) -> np.ndarray:
    context_path = VOICE_STORE.find(voice_name, ".prompt.decode_context.npy")
    if context_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' has no high-similarity VAE decode context cache",
        )
    context = np.load(context_path).astype(np.float32, copy=False)
    if context.ndim != 3 or context.shape[1:] not in {
        (generator.patch_size, generator.latent_dim),
        (generator.latent_dim, generator.patch_size),
    }:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Voice '{voice_name}' prompt decode context cache has shape "
                f"{context.shape}; expected (N, {generator.patch_size}, "
                f"{generator.latent_dim})."
            ),
        )
    return context


def compile_voice_feature_cache_from_audio(
    voice_name: str,
    audio_path: str,
    *,
    prompt: bool = False,
) -> None:
    padding_mode = "left" if prompt else "right"
    audio_feat = generator.vae_encoder.encode_wav(audio_path, padding_mode=padding_mode)
    embed = encode_voice_feature_cache(audio_feat)
    embed_suffix = ".prompt.embed.npy" if prompt else ".embed.npy"
    np.save(os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{voice_name}{embed_suffix}"), embed)
    if prompt and audio_feat.shape[0] > 0:
        np.save(
            os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{voice_name}.prompt.cond.npy"),
            audio_feat[-1],
        )
        context = audio_feat[-PROMPT_DECODE_CONTEXT_PATCHES:]
        np.save(
            os.path.join(
                CUSTOM_VOICE_CACHE_DIR, f"{voice_name}.prompt.decode_context.npy"
            ),
            context,
        )


def get_lm_cache_length() -> int | None:
    return (
        int(generator.lm_cache_length)
        if generator.lm_cache_length is not None
        else None
    )


def validate_voice_parameters(max_length, cfg_value, inference_timesteps):
    cache_length = get_lm_cache_length()
    if max_length <= 0:
        raise HTTPException(status_code=400, detail="max_length must be positive")
    if cache_length is not None and max_length > cache_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_length must be between 1 and {cache_length}; "
                "the upper bound is the LM KV cache length"
            ),
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


def validate_speech_request_preflight(request: "SpeechRequest") -> None:
    if (
        request.max_length is None
        or request.cfg_value is None
        or request.inference_timesteps is None
    ):
        raise HTTPException(
            status_code=400,
            detail="max_length, cfg_value, and inference_timesteps are required",
        )
    validate_voice_parameters(
        request.max_length,
        request.cfg_value,
        request.inference_timesteps,
    )

    reference_path = (request.reference_wav_path or "").strip() or None
    prompt_path = (request.prompt_wav_path or "").strip() or None
    prompt_text = (request.prompt_text or "").strip()
    voice_name = (request.voice or "").strip() or None
    if voice_name is not None:
        voice_name = VOICE_STORE.validate(voice_name)
        voice_mode = (request.voice_mode or "reference").strip().replace("-", "_")
        if voice_mode not in {"reference", "reference_plus_prompt", "high_similarity"}:
            raise HTTPException(status_code=400, detail="Invalid voice_mode")
        if reference_path is not None:
            raise HTTPException(
                status_code=400,
                detail="voice and reference_wav_path are both voice references; use only one",
            )
        if VOICE_STORE.embed_path(voice_name) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_name}' not found. Available: {VOICE_STORE.available()}",
            )
        if prompt_path is not None and voice_mode != "reference_plus_prompt":
            raise HTTPException(
                status_code=400,
                detail=(
                    "prompt_wav_path with a preset voice requires "
                    "voice_mode='reference_plus_prompt'"
                ),
            )
        if voice_mode == "reference_plus_prompt" and (prompt_path is None or not prompt_text):
            raise HTTPException(
                status_code=400,
                detail=(
                    "voice_mode='reference_plus_prompt' requires "
                    "prompt_wav_path and prompt_text"
                ),
            )
        if voice_mode == "high_similarity" and not VOICE_STORE.prompt_text(voice_name):
            raise HTTPException(
                status_code=400,
                detail=f"voice '{voice_name}' has no prompt transcription for high_similarity mode",
            )
        if voice_mode == "high_similarity":
            missing_suffix = next(
                (
                    suffix
                    for suffix in (
                        ".prompt.embed.npy",
                        ".prompt.cond.npy",
                        ".prompt.decode_context.npy",
                    )
                    if VOICE_STORE.find(voice_name, suffix) is None
                ),
                None,
            )
            if missing_suffix is not None:
                raise HTTPException(
                    status_code=404,
                    detail=f"voice '{voice_name}' is missing {missing_suffix}",
                )
    for label, path in (("reference_wav_path", reference_path), ("prompt_wav_path", prompt_path)):
        if path is not None and not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"{label} does not exist: {path}")
    if prompt_path is not None and not prompt_text:
        raise HTTPException(
            status_code=400,
            detail="prompt_text is required when prompt_wav_path is provided",
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
    validate_speech_request_preflight(request)
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
    _update_live_rtf(job, chunk_samples)
    _update_final_rtf(job, chunk_samples)
    return chunk_samples


def finish_generation_job(job: GenerationJob, status: str = "stopped") -> None:
    job.cancel_event.set()
    _print_final_rtf_summary(job)
    if not job.final_printed:
        _print_final_metrics(job, status)


def generate_audio_chunks(
    request: SpeechRequest,
    cancellation_event=None,
    metrics_callback=None,
):
    global text_normalizer

    if cancellation_event is None:
        cancellation_event = threading.Event()

    max_length = int(request.max_length)
    cfg_value = float(request.cfg_value)
    inference_timesteps = int(request.inference_timesteps)

    text = request.input
    if request.normalize:
        if text_normalizer is None:
            from .text_normalize import TextNormalizer

            text_normalizer = TextNormalizer()
        text = text_normalizer.normalize(text)

    text = normalize_apple_punctuation(text)
    text = ftfy.fix_text(text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    control = (request.control_instruction or "").strip()
    control = re.sub(r"[()（）]", "", control).strip()
    final_text = f"({control}){text}" if control else text

    reference_path = (request.reference_wav_path or "").strip() or None
    prompt_path = (request.prompt_wav_path or "").strip() or None
    prompt_text_clean = (request.prompt_text or "").strip()

    prompt_audio_embed = None
    reference_audio_embed = None
    prompt_prefix_feat_cond = None
    prompt_decode_context = None
    gen_kwargs_lm_prefix_cache_path = None
    gen_kwargs_lm_prefix_cache_read_paths = None
    voice_name = (request.voice or "").strip() or None
    if voice_name is not None:
        voice_name = VOICE_STORE.validate(voice_name)
        voice_mode_clean = (request.voice_mode or "reference").strip().replace("-", "_")
        reference_audio_embed = load_voice_feature_cache(voice_name).astype(
            np.float32, copy=False
        )
        (
            gen_kwargs_lm_prefix_cache_read_paths,
            gen_kwargs_lm_prefix_cache_path,
        ) = lm_prefix_cache_paths(voice_name)
        voice_prompt_text = (
            VOICE_STORE.prompt_text(voice_name)
            if voice_mode_clean == "high_similarity"
            else None
        )
        if voice_mode_clean == "high_similarity" and not voice_prompt_text:
            raise ValueError(
                f"voice '{voice_name}' has no prompt transcription for high_similarity mode"
            )
        if voice_prompt_text:
            prompt_text_clean = voice_prompt_text.strip()
            prompt_audio_embed = load_voice_feature_cache(
                voice_name, prompt=True
            ).astype(np.float32, copy=False)
            prompt_prefix_feat_cond = load_voice_prompt_cond(voice_name)
            prompt_decode_context = load_voice_prompt_decode_context(voice_name)

    target_text_length = len(generator._encode_text(final_text))
    effective_max_length = min(int(target_text_length * 6.0 + 10), int(max_length))

    gen_kwargs = dict(
        target_text=final_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        max_len=effective_max_length,
        seed=request.seed,
    )
    if voice_name is not None:
        gen_kwargs["lm_prefix_cache_path"] = gen_kwargs_lm_prefix_cache_path
        gen_kwargs["lm_prefix_cache_read_paths"] = (
            gen_kwargs_lm_prefix_cache_read_paths
        )
    if reference_path is not None:
        gen_kwargs["reference_wav_path"] = reference_path
    elif reference_audio_embed is not None:
        gen_kwargs["reference_audio_embed"] = reference_audio_embed
    if prompt_path is not None:
        gen_kwargs["prompt_wav_path"] = prompt_path
        gen_kwargs["prompt_text"] = prompt_text_clean
    elif prompt_audio_embed is not None:
        gen_kwargs["prompt_audio_embed"] = prompt_audio_embed
        gen_kwargs["prompt_prefix_feat_cond"] = prompt_prefix_feat_cond
        gen_kwargs["prompt_decode_context"] = prompt_decode_context
        gen_kwargs["prompt_text"] = prompt_text_clean

    has_reference = reference_path is not None or reference_audio_embed is not None
    has_prompt = prompt_path is not None or prompt_audio_embed is not None
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


@app.on_event("startup")
async def startup_event():
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
    except HTTPException:
        raise
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
async def playback_speech(request: SpeechRequest):
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
    name = VOICE_STORE.validate(request.voice_name)
    if VOICE_STORE.is_default(name):
        raise HTTPException(status_code=403, detail=f"'{name}' is a system voice")

    embed_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}.embed.npy")
    if os.path.exists(embed_path) and not request.replace:
        raise HTTPException(
            status_code=409, detail=f"'{name}' exists. Set replace=True."
        )

    audio_path = (request.reference_wav_path or "").strip()
    if not audio_path:
        raise HTTPException(status_code=400, detail="reference_wav_path is required")

    prompt_text_val = (request.prompt_text or "").strip()
    if prompt_text_val and os.path.isfile(prompt_text_val):
        with open(prompt_text_val, "r", encoding="utf-8") as f:
            prompt_text_val = f.read().strip()

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail=f"Audio not found: {audio_path}")

    try:
        os.makedirs(CUSTOM_VOICE_CACHE_DIR, exist_ok=True)
        VOICE_FEATURE_CACHE_MEMORY.pop((name, "reference"), None)
        VOICE_FEATURE_CACHE_MEMORY.pop((name, "prompt"), None)
        for stale_suffix in (
            ".embed.npy",
            ".prompt.embed.npy",
            ".prompt.cond.npy",
            ".prompt.decode_context.npy",
            ".npy",
        ):
            stale_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}{stale_suffix}")
            if os.path.exists(stale_path):
                os.unlink(stale_path)
        VOICE_STORE.remove_lm_prefix_caches(name)

        compile_voice_feature_cache_from_audio(name, audio_path)
        txt_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}.txt")
        if prompt_text_val:
            compile_voice_feature_cache_from_audio(name, audio_path, prompt=True)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text_val)
            mode = "reference_plus_continuation"
        else:
            if os.path.exists(txt_path):
                os.unlink(txt_path)
            mode = "reference"

        return {
            "status": "success",
            "message": f"Voice '{name}' created.",
            "mode": mode,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {e}")


@app.get("/voices")
async def get_available_voices():
    voices = VOICE_STORE.available()
    system_voices = VOICE_STORE.names_from_dir(VOICE_CACHE_DIR)
    custom_voices = VOICE_STORE.names_from_dir(CUSTOM_VOICE_CACHE_DIR)
    return {
        "voices": voices,
        "count": len(voices),
        "system_voices": system_voices,
        "custom_voices": custom_voices,
        "included_voice_cache_directory": VOICE_CACHE_DIR,
        "included_voice_cache_directories": VOICE_CACHE_DIRS,
        "custom_cache_directory": CUSTOM_VOICE_CACHE_DIR,
    }


@app.delete("/v1/voices/{voice_name}")
async def delete_voice(voice_name: str):
    name = VOICE_STORE.validate(voice_name)
    if not VOICE_STORE.is_custom(name):
        if VOICE_STORE.is_default(name):
            raise HTTPException(status_code=403, detail=f"'{name}' is a system voice")
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    removed = []
    for suffix in (
        ".embed.npy",
        ".prompt.embed.npy",
        ".prompt.cond.npy",
        ".prompt.decode_context.npy",
        ".npy",
        ".txt",
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".m4a",
        ".aac",
    ):
        path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}{suffix}")
        if not os.path.exists(path):
            continue
        try:
            os.unlink(path)
            removed.append(os.path.basename(path))
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to remove '{os.path.basename(path)}': {exc}",
            ) from exc

    removed.extend(VOICE_STORE.remove_lm_prefix_caches(name))
    VOICE_FEATURE_CACHE_MEMORY.pop((name, "reference"), None)
    VOICE_FEATURE_CACHE_MEMORY.pop((name, "prompt"), None)
    return {
        "status": "success",
        "message": f"Voice '{name}' deleted.",
        "removed": removed,
    }


@app.get("/health")
async def health_check():
    is_processing = CURRENT_JOB is not None
    return {
        "status": "healthy",
        "is_processing": is_processing,
        "current_job_id": CURRENT_JOB.job_id if is_processing else None,
        "model": "voxcpm2",
        "lm_cache_length": get_lm_cache_length(),
        "included_voice_cache_directories": VOICE_CACHE_DIRS,
    }


def main():
    parser = argparse.ArgumentParser(description="VoxCPM2 TTS Server")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--cache-dir", type=str, default=os.path.expanduser("~/.cache/ane_tts")
    )
    parser.add_argument(
        "--included-voice-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory containing bundled voice caches. Defaults to "
            "<model-dir>/caches when present; otherwise downloads caches/* "
            "from --repo-id."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local path to CoreML model directory. If not set, downloads from --repo-id.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help=f"Hugging Face model repo to download when --model-dir is omitted. Default: {REPO_ID}.",
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
        default="single-length",
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
        default=None,
        help=(
            "LM chunk length for prompt prefill. Defaults to 128 for preload "
            "and hot-swap, otherwise 16. Available values are 1, 8, 16, 32, "
            "64, and 128."
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
        default=16,
        help=(
            "Number of initial AR steps where the VAE decoder runs immediately "
            "(one chunk per step) for low TTFB. After this many steps, "
            "decoding switches to batch mode. Default: 16. Use 0 to always "
            "decode immediately."
        ),
    )
    parser.add_argument(
        "--vae-batch-decode-steps",
        type=int,
        default=4,
        help=(
            "Number of AR steps to accumulate before batch-decoding audio "
            "after the early-decode phase. Requires a RangeDim VAE decoder "
            "model. Default: 4. Use 1 to disable batching."
        ),
    )
    parser.add_argument(
        "--base-lm-path",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to BaseLM model package(s). Can accept multiple split parts.",
    )
    for option, description in (
        ("residual-lm-path", "ResidualLM"),
        ("locdit-path", "LocDiT"),
        ("vae-encoder-path", "Audio VAE Encoder"),
        ("feat-encoder-path", "Feature Encoder"),
        ("vae-decoder-path", "Audio VAE Decoder"),
        ("fsq-path", "FSQ"),
        ("projections-path", "Projections"),
    ):
        parser.add_argument(
            f"--{option}",
            type=str,
            default=None,
            help=f"Path to {description} model package.",
        )
    parser.add_argument(
        "--compile-and-save",
        action="store_true",
        default=False,
        help="Compile CoreML .mlpackage files into .mlmodelc on the fly if they do not exist.",
    )
    parser.add_argument(
        "--startup-warmup-repeats",
        type=int,
        default=5,
        help=(
            "Number of synthetic predict calls to run per CoreML model/function "
            "during startup warmup. In hot-swap mode, the idle prefill "
            "functions are warmed and unloaded, the length_1 decode functions "
            "are warmed and unloaded, then the idle prefill functions are "
            "loaded. Use 0 to disable startup warmup."
        ),
    )

    args = parser.parse_args()

    global CUSTOM_VOICE_CACHE_DIR
    CUSTOM_VOICE_CACHE_DIR = args.cache_dir
    metrics.LIVE_RTF_METRICS = str(args.live_rtf)
    os.makedirs(CUSTOM_VOICE_CACHE_DIR, exist_ok=True)

    load_model(
        **{
            k: v
            for k, v in vars(args).items()
            if k not in {"cache_dir", "host", "port", "live_rtf"}
        }
    )

    print(f"🚀 Starting VoxCPM2 server on {args.host}:{args.port}")
    print(f"   Included voices: {VOICE_CACHE_DIR}")
    print(f"   Custom cache: {CUSTOM_VOICE_CACHE_DIR}")
    print(f"   Voices: {len(VOICE_STORE.available())}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
