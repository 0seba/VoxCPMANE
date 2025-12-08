import os
import io
import time
import threading
import traceback
import queue
import numpy as np
import sounddevice as sd
from tqdm import tqdm
import uvicorn
import pathlib
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from starlette.background import BackgroundTask
from fastapi.responses import (
    StreamingResponse,
    FileResponse,
    JSONResponse,
    HTMLResponse,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import soundfile
import soxr
import coremltools as ct
from pydantic import BaseModel
from typing import Optional, Any
import aiofiles
from huggingface_hub import snapshot_download
import asyncio
from dataclasses import dataclass
import argparse
import ftfy

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False


REPO_ID = "seba/VoxCPM-ANE"
MODEL_PATH_PREFIX = ""
VOICE_CACHE_DIR = ""
CUSTOM_VOICE_CACHE_DIR = os.path.expanduser("~/.cache/ane_tts")

try:
    lm_length = 8
    print(f"🚀 Downloading/loading model files from Hugging Face Hub repo: {REPO_ID}")
    MODEL_PATH_PREFIX = snapshot_download(repo_id=REPO_ID)
    VOICE_CACHE_DIR = os.path.join(MODEL_PATH_PREFIX, "caches")

    locdit_mlmodel_path = os.path.join(MODEL_PATH_PREFIX, "locdit_f16.mlmodelc")
    projections_mlmodel_path = os.path.join(
        MODEL_PATH_PREFIX, "projections_1_t.mlmodelc"
    )
    feature_encoder_mlmodel_path = os.path.join(
        MODEL_PATH_PREFIX, "voxcpm_feat_encoder_ane_enum_12.mlmodelc"
    )
    fsq_mlmodel_path = os.path.join(MODEL_PATH_PREFIX, "fsq_layer.mlmodelc")
    audio_vae_decoder_mlmodel_path = os.path.join(
        MODEL_PATH_PREFIX, "voxcpm_audio_vae_decoder_length_24.mlmodelc"
    )
    audio_vae_encoder_mlmodel_path = os.path.join(
        MODEL_PATH_PREFIX, "voxcpm_audio_vae_encoder_enum_length_17920.mlmodelc"
    )
    base_lm_embed_tokens_path = os.path.join(MODEL_PATH_PREFIX, "base_lm_embeds.npy")
    base_lm_mf_path = os.path.join(MODEL_PATH_PREFIX, "base_lm_mf_f16.mlmodelc/")
    residual_lm_mf_path = os.path.join(
        MODEL_PATH_PREFIX, "residual_lm_mf_f16.mlmodelc/"
    )

    required_files = [
        locdit_mlmodel_path,
        projections_mlmodel_path,
        feature_encoder_mlmodel_path,
        fsq_mlmodel_path,
        audio_vae_decoder_mlmodel_path,
        audio_vae_encoder_mlmodel_path,
        base_lm_embed_tokens_path,
        base_lm_mf_path,
        residual_lm_mf_path,
        VOICE_CACHE_DIR,
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Error: Missing model files in downloaded snapshot:")
        for f in missing_files:
            print(f"  - {f}")
        exit()

    print("Loading CoreML models...")
    locdit_mlmodel = ct.models.CompiledMLModel(
        locdit_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    projections_mlmodel = ct.models.CompiledMLModel(
        projections_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    feature_encoder_mlmodel = ct.models.CompiledMLModel(
        feature_encoder_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    fsq_mlmodel = ct.models.CompiledMLModel(
        fsq_mlmodel_path, compute_units=ct.ComputeUnit.CPU_ONLY
    )
    audio_vae_decoder_mlmodel = ct.models.CompiledMLModel(
        audio_vae_decoder_mlmodel_path, compute_units=ct.ComputeUnit.CPU_ONLY
    )
    audio_vae_encoder_mlmodel = ct.models.CompiledMLModel(
        audio_vae_encoder_mlmodel_path, compute_units=ct.ComputeUnit.CPU_ONLY
    )
    base_lm_embed_tokens = np.load(base_lm_embed_tokens_path)

    from .voxcpm import VoxCPMANE

    model = VoxCPMANE(
        "openbmb/VoxCPM-0.5B",
        base_lm_mf_path,
        residual_lm_mf_path,
        fsq_mlmodel,
        locdit_mlmodel,
        projections_mlmodel,
        audio_vae_decoder_mlmodel,
        audio_vae_encoder_mlmodel,
        feature_encoder_mlmodel,
        base_lm_embed_tokens,
        enable_denoiser=False,
        base_lm_chunk_size=lm_length,
        residual_lm_chunk_size=lm_length,
        audio_vae_encoder_chunk_size=23040,
        feature_encoder_chunk_size=16,
    )
    print("✅ Models loaded successfully.")

except Exception as e:
    print(f"❌ An unexpected error occurred during model setup: {e}")
    raise


@dataclass
class GenerationJob:
    request: "SpeechRequest"
    output_queue: queue.Queue  # Worker puts audio chunks here
    cancel_event: threading.Event  # Endpoint sets this on disconnect
    job_id: int


# The central job queue. maxsize=1 means only one job can be
# "pending". This acts as our "is_processing" flag.
GENERATION_QUEUE = queue.Queue(maxsize=1)

# This will hold a reference to the job the worker is *currently* processing
# Used by the /cancel endpoint
CURRENT_JOB: Optional[GenerationJob] = None

# A counter for unique job IDs, just for logging
JOB_COUNTER = 0


def generation_worker():
    """
    This is the *only* thread that touches the CoreML model.
    It runs forever, waiting for jobs from GENERATION_QUEUE.
    """
    global CURRENT_JOB

    while True:
        try:
            job = GENERATION_QUEUE.get()
            CURRENT_JOB = job

            try:
                audio_generator = generate_audio_chunks(
                    text_to_generate=job.request.input,
                    prompt_wav_path=job.request.prompt_wav_path,
                    prompt_text=job.request.prompt_text,
                    voice=job.request.voice,
                    max_length=job.request.max_length,
                    cfg_value=job.request.cfg_value,
                    inference_timesteps=job.request.inference_timesteps,
                    cancellation_event=job.cancel_event,
                )

                for chunk in audio_generator:
                    if job.cancel_event.is_set():
                        break

                    job.output_queue.put(chunk)

            except Exception as e:
                job.output_queue.put(e)

            finally:
                job.output_queue.put(None)
                CURRENT_JOB = None
                GENERATION_QUEUE.task_done()

        except Exception as e:
            CURRENT_JOB = None
            if "job" in locals() and isinstance(job, GenerationJob):
                job.output_queue.put(Exception("Worker failed"))
                GENERATION_QUEUE.task_done()
            time.sleep(1)


SAMPLE_RATE = 16000
app = FastAPI(title="OpenAI Compatible TTS Server")
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
)


class SpeechRequest(BaseModel):
    model: str = "voxcpm-0.5b"
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    prompt_wav_path: Optional[str] = None
    prompt_text: Optional[str] = ""
    max_length: Optional[int] = 2048
    cfg_value: Optional[float] = 2.0
    inference_timesteps: Optional[int] = 10


class PlaybackRequest(SpeechRequest):
    show_progress: Optional[bool] = True


class CreateVoiceRequest(BaseModel):
    voice_name: str
    prompt_wav_path: str
    prompt_text: str
    replace: Optional[bool] = False


def load_available_voices():
    voices = set()

    # Default cache
    if os.path.exists(VOICE_CACHE_DIR):
        for file in os.listdir(VOICE_CACHE_DIR):
            if file.endswith(".npy"):
                voices.add(file[:-4])

    # Custom cache
    if os.path.exists(CUSTOM_VOICE_CACHE_DIR):
        for file in os.listdir(CUSTOM_VOICE_CACHE_DIR):
            if file.endswith(".npy"):
                voices.add(file[:-4])

    return sorted(list(voices))


def is_default_voice(voice_name: str) -> bool:
    path = os.path.join(VOICE_CACHE_DIR, f"{voice_name}.npy")
    return os.path.exists(path)


def get_voice_prompt_text(voice_name: str) -> str:
    # Check default first
    if is_default_voice(voice_name):
        return CACHED_VOICE_TEXT

    # Check custom
    txt_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{voice_name}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    # Fallback/Error
    raise HTTPException(
        status_code=500,
        detail=f"Voice '{voice_name}' found in custom cache but transcription file missing at: {txt_path}",
    )


def load_voice_cache(voice_name: str):

    # Check default first
    cache_path = os.path.join(VOICE_CACHE_DIR, f"{voice_name}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    # Check custom
    cache_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{voice_name}.npy")
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load custom voice '{voice_name}': {e}",
            )

    raise HTTPException(
        status_code=404,
        detail=f"Voice '{voice_name}' not found. Available: {load_available_voices()}",
    )


def validate_voice_parameters(
    max_length: int, cfg_value: float, inference_timesteps: int
):
    if not (0 < max_length <= 2048):
        raise HTTPException(
            status_code=400, detail="max_length must be between 1 and 2048"
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
    """
    Convert Apple smart/typographic punctuation to ASCII equivalents.
    """
    # Create translation table mapping Unicode smart chars to ASCII
    translation_table = str.maketrans({
        # Smart quotes
        '\u201c': '"',  # “ (left double quote)
        '\u201d': '"',  # ” (right double quote)
        '\u2018': "'",  # ‘ (left single quote)
        '\u2019': "'",  # ’ (right single quote)
        
        # Dashes
        '\u2013': '-',  # – (en dash)
        '\u2014': '-',  # — (em dash)
        
        # Ellipsis
        '\u2026': '...',  # … (horizontal ellipsis)
        
        # Bullets and other symbols
        '\u2022': '*',  # • (bullet)
        '\u00a0': ' ',  # (non-breaking space)
        
        # Other common smart punctuation
        '\u201a': ',',  # ‚ (single low-9 quotation mark)
        '\u201e': '"',  # „ (double low-9 quotation mark)
        '\u2039': '<',  # ‹ (single left-pointing angle quotation)
        '\u203a': '>',  # › (single right-pointing angle quotation)
    })
    
    return text.translate(translation_table)

def generate_audio_chunks(
    text_to_generate,
    prompt_wav_path,
    prompt_text,
    voice=None,
    max_length=2048,
    cfg_value=2.0,
    inference_timesteps=10,
    cancellation_event: threading.Event = None,
):
    import re

    if cancellation_event is None:
        cancellation_event = threading.Event()

    validate_voice_parameters(max_length, cfg_value, inference_timesteps)

    audio_cache = None
    audio = None

    if voice is not None:
        audio_cache = load_voice_cache(voice)
        prompt_text = get_voice_prompt_text(voice)
        text = prompt_text + " " + text_to_generate
    else:
        if prompt_wav_path and prompt_wav_path.strip():
            if not os.path.exists(prompt_wav_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt WAV file not found: {prompt_wav_path}",
                )
            try:
                audio, sr = soundfile.read(prompt_wav_path)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1, keepdims=False)
                if sr != 16_000:
                    audio = soxr.resample(audio, sr, 16_000, "HQ")
                if audio.ndim == 1:
                    audio = audio[None, :]
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Error loading prompt WAV: {e}"
                )
        else:
            audio = None
        text = prompt_text + " " + text_to_generate

    text = normalize_apple_punctuation(text)
    text = ftfy.fix_text(text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text_token = np.array(
        # model.tts_model.text_tokenizer(model.text_normalizer.normalize(text)),
        model.tts_model.text_tokenizer(text),
        dtype=np.int32,
    )[None, :]

    if audio is not None:
        patch_len = model.tts_model.patch_size * model.tts_model.chunk_size
        if audio.shape[1] % patch_len != 0:
            pad_width = patch_len - audio.shape[1] % patch_len
            audio = np.pad(audio, ((0, 0), (0, pad_width)))
        audio = audio[None, :]

    try:
        if voice is not None:
            generator = model.tts_model._generate_threaded_prompt_processing(
                text_token,
                audio=None,
                audio_cache=audio_cache,
                max_length=max_length,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
        else:
            generator = model.tts_model._generate_threaded_prompt_processing(
                text_token,
                audio,
                max_length=max_length,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )

        for (chunk,) in generator:
            if cancellation_event.is_set():
                break
            audio_chunk_float32 = chunk.astype(np.float32)
            yield audio_chunk_float32

    except GeneratorExit:
        pass
    except Exception as e:
        raise
    finally:
        cancellation_event.set()


def scan_and_compile_audio_cache():
    """
    Scans the custom cache directory for audio and .txt files.
    If both exist and .npy is missing, creates the voice cache.
    Supported audio formats: wav, mp3, flac, ogg, opus, aac, m4a.
    If partial files exist without .npy, warns the user.
    """
    if not os.path.exists(CUSTOM_VOICE_CACHE_DIR):
        return

    try:
        files = os.listdir(CUSTOM_VOICE_CACHE_DIR)
    except Exception as e:
        print(f"⚠️  Failed to list custom cache dir: {e}")
        return

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac", ".m4a"}

    # Map basename -> {extensions present}
    file_map = {}
    for f in files:
        name, ext = os.path.splitext(f)
        ext = ext.lower()
        if name not in file_map:
            file_map[name] = set()
        file_map[name].add(ext)

    for name, extensions in file_map.items():
        if ".npy" in extensions:
            continue

        has_txt = ".txt" in extensions
        audio_ext = None
        for ext in extensions:
            if ext in AUDIO_EXTENSIONS:
                audio_ext = ext
                break

        if has_txt and audio_ext:
            print(
                f"🔄 Compiling cache for new voice: '{name}' from {audio_ext} and .txt..."
            )

            audio_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}{audio_ext}")
            txt_path = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{name}.txt")

            # Read text
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read()
            except Exception as e:
                print(f"❌ Failed to read text for '{name}': {e}")
                continue

            tmp_wav_path = None
            try:
                processing_path = None
                # Convert audio using pydub if available (robust)
                if PYDUB_AVAILABLE:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp_wav:
                        tmp_wav_path = tmp_wav.name

                    # AudioSegment.from_file handles various formats (ffmpeg)
                    audio = AudioSegment.from_file(audio_path)
                    audio.export(tmp_wav_path, format="wav")
                    processing_path = tmp_wav_path
                else:
                    # If pydub missing, rely on what soundfile supports directly (usually wav, flac, ogg)
                    if audio_ext in [".wav", ".flac", ".ogg"]:
                        processing_path = audio_path
                    else:
                        print(
                            f"❌ Cannot compile '{name}': pydub is not installed (required for {audio_ext})"
                        )
                        continue

                # Create voice
                # model is global in server.py
                model.create_custom_voice(
                    voice_name=name,
                    prompt_wav_path=processing_path,
                    prompt_text=prompt_text,
                    cache_dir=CUSTOM_VOICE_CACHE_DIR,
                )
                print(f"✅ Successfully compiled voice: '{name}'")

            except Exception as e:
                print(f"❌ Failed to compile voice '{name}': {e}")
            finally:
                if tmp_wav_path and os.path.exists(tmp_wav_path):
                    try:
                        os.unlink(tmp_wav_path)
                    except:
                        pass

        elif has_txt or audio_ext:
            # Only partial match found (and no npy)
            missing = []
            if not has_txt:
                missing.append(".txt")
            if not audio_ext:
                missing.append("audio file")
            print(
                f"⚠️  Incomplete voice files for '{name}': Missing {', '.join(missing)}. .npy cache not generated."
            )


@app.on_event("startup")
async def startup_event():
    scan_and_compile_audio_cache()
    worker_thread = threading.Thread(target=generation_worker, daemon=True)
    worker_thread.start()


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    try:
        async with aiofiles.open(FRONTEND_FILE, mode="r") as f:
            content = await f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: <code>index.html</code> not found.</h1>",
            status_code=404,
        )


async def poll_queue_for_chunks(
    output_queue: queue.Queue, poll_interval: float = 0.005
):
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

    # Fast path: Use soundfile for WAV and FLAC
    if audio_format in ["wav", "flac"]:
        # These formats are always supported
        pass
    # Conversion path: Use pydub for other formats (mp3, ogg, opus, etc.)
    else:
        if not PYDUB_AVAILABLE:
            raise HTTPException(
                status_code=501,
                detail=f"Format '{audio_format}' is not supported. "
                f"Install 'pydub' to enable conversion.",
            )

        # Check if the format is specifically supported
        if audio_format not in ["mp3", "opus", "ogg", "aac"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {audio_format}. Supported: wav, flac, mp3, opus, ogg, aac",
            )

    global JOB_COUNTER
    JOB_COUNTER += 1
    job_id = JOB_COUNTER

    output_queue = queue.Queue(maxsize=1024)
    cancel_event = threading.Event()
    job = GenerationJob(request, output_queue, cancel_event, job_id)

    try:
        GENERATION_QUEUE.put_nowait(job)
    except queue.Full:
        raise HTTPException(
            status_code=429, detail="Server is busy processing another request"
        )

    all_chunks = []
    try:
        async for chunk in poll_queue_for_chunks(output_queue):
            all_chunks.append(chunk)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Audio generation failed: {str(e)}"
        )
    finally:
        cancel_event.set()

    if not all_chunks:
        raise HTTPException(
            status_code=500, detail="Audio generation failed (no chunks produced)"
        )

    try:
        full_audio_float32 = np.concatenate(all_chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to concatenate audio chunks: {str(e)}"
        )

    audio_format = request.response_format.lower()
    buffer = io.BytesIO()

    if audio_format in ["wav", "flac"]:
        try:
            soundfile.write(
                buffer, full_audio_float32, SAMPLE_RATE, format=audio_format
            )
            media_type = f"audio/{audio_format}"
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode audio to {audio_format}: {str(e)}",
            )

    else:
        try:
            pcm_data_16bit = (full_audio_float32 * 32767).astype(np.int16)

            segment = AudioSegment(
                data=pcm_data_16bit.tobytes(),
                sample_width=2,  # 2 bytes = 16-bit
                frame_rate=SAMPLE_RATE,
                channels=1,
            )

            if audio_format == "mp3":
                media_type = "audio/mpeg"
                segment.export(buffer, format="mp3")
            elif audio_format == "opus":
                media_type = "audio/opus"
                segment.export(buffer, format="opus")
            elif audio_format == "ogg":
                media_type = "audio/ogg"
                segment.export(buffer, format="ogg")
            elif audio_format == "aac":
                media_type = "audio/aac"
                segment.export(
                    buffer, format="adts"
                )  # ADTS is a common container for raw AAC

        except Exception as e:
            # This can happen if ffmpeg is not installed!
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode audio to {audio_format}. "
                f"Ensure 'ffmpeg' is installed and accessible. Error: {str(e)}",
            )

    buffer.seek(0)
    return Response(content=buffer.getvalue(), media_type=media_type)


@app.post("/v1/audio/speech/stream")
async def stream_speech(request: SpeechRequest):
    global JOB_COUNTER
    JOB_COUNTER += 1
    job_id = JOB_COUNTER

    output_queue = queue.Queue(maxsize=1024)
    cancel_event = threading.Event()
    job = GenerationJob(request, output_queue, cancel_event, job_id)

    try:
        GENERATION_QUEUE.put_nowait(job)
    except queue.Full:
        raise HTTPException(
            status_code=429, detail="Server is busy processing another request"
        )

    async def audio_stream_generator():
        try:
            async for chunk in poll_queue_for_chunks(output_queue):
                # Convert to 16-bit PCM bytes
                chunk_16bit = (chunk * 32767).astype(np.int16)
                yield chunk_16bit.tobytes()
        except Exception as e:
            pass
        finally:
            cancel_event.set()

    return StreamingResponse(
        audio_stream_generator(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE)},
    )


@app.post("/v1/audio/speech/playback")
async def playback_speech(request: PlaybackRequest):
    global JOB_COUNTER, CURRENT_JOB
    JOB_COUNTER += 1
    job_id = JOB_COUNTER

    output_queue = queue.Queue(maxsize=1024)
    cancel_event = threading.Event()
    job = GenerationJob(request, output_queue, cancel_event, job_id)

    try:
        GENERATION_QUEUE.put_nowait(job)
        CURRENT_JOB = job
    except queue.Full:
        raise HTTPException(
            status_code=429, detail="Server is busy processing another request"
        )

    client_disconnected = False
    playback_start_time = time.time()
    TIMEOUT_SECONDS = 300

    try:
        if not sd.query_devices():
            raise HTTPException(
                status_code=500, detail="No audio output devices available"
            )

        chunks = poll_queue_for_chunks(output_queue)

        # Progress bar (optional)
        pbar = None
        # if request.show_progress:
        #     pbar = tqdm(desc=f"Job {job_id}: Playing audio", unit="chunk")

        chunk_count = 0
        last_chunk = None

        # Use context manager for proper stream lifecycle
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            latency="low",
            blocksize=1024,
        ) as stream:
            async for chunk in chunks:
                elapsed = time.time() - playback_start_time
                if elapsed > TIMEOUT_SECONDS:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Playback timeout after {TIMEOUT_SECONDS} seconds",
                    )

                chunk_count += 1
                last_chunk = chunk

                if pbar:
                    pbar.update(1)

                await asyncio.to_thread(stream.write, chunk)

            # CRITICAL FIX: Apply fade-out to prevent click/pop at end
            if last_chunk is not None and len(last_chunk) > 100:
                # Take the last 50ms of audio and fade it to zero
                fade_duration_ms = 50
                fade_samples = int(SAMPLE_RATE * fade_duration_ms / 1000)
                fade_samples = min(fade_samples, len(last_chunk))

                # Create a copy to avoid modifying the original
                faded_chunk = last_chunk[-fade_samples:].copy()
                fade_window = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                faded_chunk *= fade_window

                # Write the faded tail (this replaces the original tail)
                # We write it separately to ensure it's the very last thing played
                await asyncio.to_thread(stream.write, faded_chunk)

            # Optional: add a tiny silence buffer to ensure clean stop
            # This gives the hardware time to finish the fade
            silence_buffer = np.zeros(128, dtype=np.float32)  # ~8ms of silence
            await asyncio.to_thread(stream.write, silence_buffer)

        # Close progress bar
        if pbar:
            pbar.close()

        if chunk_count == 0:
            raise HTTPException(
                status_code=500, detail="Failed to generate audio (no chunks)"
            )

        total_duration = time.time() - playback_start_time

        status = "cancelled" if client_disconnected else "success"
        message = f"Audio playback {'cancelled' if client_disconnected else 'completed'} for Job {job_id}"

        return JSONResponse(
            {
                "status": status,
                "message": message,
                "job_id": job_id,
                "chunks_played": chunk_count,
                "duration_seconds": round(total_duration, 2),
            }
        )

    except asyncio.CancelledError:
        client_disconnected = True
        raise HTTPException(status_code=499, detail="Client disconnected")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Playback failed: {e}")
    finally:
        cancel_event.set()
        if "pbar" in locals() and pbar is not None:
            pbar.close()
        if CURRENT_JOB is job:
            CURRENT_JOB = None


@app.post("/v1/audio/speech/cancel")
async def cancel_generation():
    if CURRENT_JOB is None:
        return JSONResponse(
            {"status": "success", "message": "No generation in progress"}
        )

    try:
        CURRENT_JOB.cancel_event.set()
        return JSONResponse(
            {
                "status": "success",
                "message": f"Cancellation signal sent to Job {CURRENT_JOB.job_id}",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel: {e}")


@app.post("/v1/voices")
async def create_voice(request: CreateVoiceRequest):
    voice_name = request.voice_name

    if ".." in voice_name or "/" in voice_name or "\\" in voice_name:
        raise HTTPException(
            status_code=400, detail="Invalid voice name: must be a safe filename"
        )

    # Check if exists in default (Forbidden)
    if is_default_voice(voice_name):
        raise HTTPException(
            status_code=403,
            detail=f"Voice '{voice_name}' exists in system voices and cannot be overwritten.",
        )

    # Check if exists in custom (Conflict/Replace)
    custom_npy = os.path.join(CUSTOM_VOICE_CACHE_DIR, f"{voice_name}.npy")
    if os.path.exists(custom_npy) and not request.replace:
        raise HTTPException(
            status_code=409,
            detail=f"Voice '{voice_name}' already exists in custom cache. Set replace=True to overwrite.",
        )

    # Handle prompt_text (file path or text)
    prompt_text = request.prompt_text
    if os.path.exists(prompt_text) and os.path.isfile(prompt_text):
        try:
            with open(prompt_text, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to read prompt text file: {e}"
            )

    # Validate audio path
    if not os.path.exists(request.prompt_wav_path):
        raise HTTPException(
            status_code=400,
            detail=f"Prompt audio file not found: {request.prompt_wav_path}",
        )

    try:
        model.create_custom_voice(
            voice_name=voice_name,
            prompt_wav_path=request.prompt_wav_path,
            prompt_text=prompt_text,
            cache_dir=CUSTOM_VOICE_CACHE_DIR,
        )
        return {
            "status": "success",
            "message": f"Voice '{voice_name}' created successfully.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create voice: {e}")


@app.get("/voices")
async def get_available_voices():
    """Get list of available cached voices"""
    try:
        voices = load_available_voices()
        return {
            "voices": voices,
            "count": len(voices),
            "cache_directory": "assets/caches",
            "custom_cache_directory": CUSTOM_VOICE_CACHE_DIR,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load voices: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_processing = CURRENT_JOB is not None
    return {
        "status": "healthy",
        "is_processing": is_processing,
        "current_job_id": CURRENT_JOB.job_id if is_processing else None,
        "queue_pending": not GENERATION_QUEUE.empty(),
        "model": "voxcpm-0.5b",
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible TTS Server")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expanduser("~/.cache/ane_tts"),
        help="Directory for custom voice caches",
    )
    args = parser.parse_args()

    global CUSTOM_VOICE_CACHE_DIR
    CUSTOM_VOICE_CACHE_DIR = args.cache_dir
    if not os.path.exists(CUSTOM_VOICE_CACHE_DIR):
        os.makedirs(CUSTOM_VOICE_CACHE_DIR, exist_ok=True)

    print("🚀 Starting server...")
    print(f"   Access the frontend playground at: http://{args.host}:{args.port}")
    print(f"   Custom cache dir: {CUSTOM_VOICE_CACHE_DIR}")
    print(f"   Available voices: {len(load_available_voices())}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
