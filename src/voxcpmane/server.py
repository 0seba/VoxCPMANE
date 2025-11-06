import os
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

# ===================================================================
# 🚀 1. Model Setup (Unchanged)
# ===================================================================

REPO_ID = "seba/VoxCPM-ANE"
MODEL_PATH_PREFIX = ""
VOICE_CACHE_DIR = ""

try:
    lm_length = 8
    print(f"🚀 Downloading/loading model files from Hugging Face Hub repo: {REPO_ID}")
    MODEL_PATH_PREFIX = snapshot_download(repo_id=REPO_ID)
    print(f"✅ Model files are available at: {MODEL_PATH_PREFIX}")
    VOICE_CACHE_DIR = os.path.join(MODEL_PATH_PREFIX, "caches")

    # ... [All your model path and file existence checks] ...

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

    # --- Model Loading ---
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
    # Re-raise to stop the server from starting
    raise

# ===================================================================
# ⚙️ 2. Generation Actor (Worker) Setup
# ===================================================================


# This dataclass defines a "job" for the generation worker
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
    This is the *only* thread that touches the C++ model.
    It runs forever, waiting for jobs from GENERATION_QUEUE.
    """
    global CURRENT_JOB
    print(
        f"[Worker] 🚀 Generation worker thread started (TID: {threading.current_thread().ident})"
    )

    while True:
        try:
            # 1. Wait for a job
            # This call blocks until a job is available
            job = GENERATION_QUEUE.get()
            print(f"[Worker] 🟢 Picked up Job {job.job_id}")
            CURRENT_JOB = job

            # 2. Run the generation
            try:
                # Get the generator
                audio_generator = generate_audio_chunks(
                    text_to_generate=job.request.input,
                    prompt_wav_path=job.request.prompt_wav_path,
                    prompt_text=job.request.prompt_text,
                    voice=job.request.voice,
                    max_length=job.request.max_length,
                    cfg_value=job.request.cfg_value,
                    inference_timesteps=job.request.inference_timesteps,
                    cancellation_event=job.cancel_event,  # Pass the per-job event
                )

                # 3. Feed chunks to the output queue
                for chunk in audio_generator:
                    if job.cancel_event.is_set():
                        print(f"[Worker] 🟡 Job {job.job_id} cancelled by event.")
                        break

                    # Put chunk in the output queue.
                    # This might block if the client isn't consuming,
                    # which is what we want (backpressure).
                    job.output_queue.put(chunk)

                if not job.cancel_event.is_set():
                    print(f"[Worker] ✅ Job {job.job_id} finished normally.")

            except Exception as e:
                # Handle errors *during* generation
                print(f"[Worker] ❌ Error during generation for Job {job.job_id}: {e}")
                # Send the error to the client via the queue
                job.output_queue.put(e)

            finally:
                # 4. Signal completion (or error/cancellation)
                # Put the "None" sentinel to tell the client we're done
                job.output_queue.put(None)
                print(f"[Worker] 🏁 Job {job.job_id} complete. Cleaning up.")
                CURRENT_JOB = None
                GENERATION_QUEUE.task_done()

        except Exception as e:
            # Handle errors in the worker loop itself
            print(f"[Worker] ❌ FATAL WORKER ERROR: {e}. Restarting loop.")
            # Clear state to be safe
            CURRENT_JOB = None
            if "job" in locals() and isinstance(job, GenerationJob):
                job.output_queue.put(Exception("Worker failed"))
                GENERATION_QUEUE.task_done()
            time.sleep(1)


# ===================================================================
# 🎵 3. Audio Generation & Helper Functions
# ===================================================================

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


# --- Request Models ---
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


# --- Voice Cache ---
def load_available_voices():
    cache_dir = "assets/caches"  # Note: Your code used VOICE_CACHE_DIR, but logic used "assets/caches". Sticking to this.
    voices = []
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith(".npy"):
                voices.append(file[:-4])
    return sorted(voices)


def load_voice_cache(voice_name: str):
    cache_path = f"assets/caches/{voice_name}.npy"
    if not os.path.exists(cache_path):
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found. Available: {load_available_voices()}",
        )
    try:
        return np.load(cache_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load voice '{voice_name}': {e}"
        )


# --- Param Validation ---
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


# --- Core Generation Function (Modified) ---
def generate_audio_chunks(
    text_to_generate,
    prompt_wav_path,
    prompt_text,
    voice=None,
    max_length=2048,  # Default was 4096, but validation says 2048
    cfg_value=2.0,
    inference_timesteps=10,
    cancellation_event: threading.Event = None,  # <-- Accepts the per-job event
):
    """Generator that yields audio chunks and respects the cancellation_event"""
    import re

    # Ensure a default event if none is provided
    if cancellation_event is None:
        cancellation_event = threading.Event()

    # Validate parameters
    validate_voice_parameters(max_length, cfg_value, inference_timesteps)

    # ... [Rest of your audio/text setup logic is unchanged] ...
    audio_cache = None
    audio = None

    if voice is not None:
        audio_cache = load_voice_cache(voice)
        text = CACHED_VOICE_TEXT + " " + text_to_generate
    else:
        if prompt_wav_path and prompt_wav_path.strip():
            if not os.path.exists(prompt_wav_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt WAV file not found: {prompt_wav_path}",
                )
            try:
                audio, sr = soundfile.read(prompt_wav_path)
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

    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text_token = np.array(
        model.tts_model.text_tokenizer(model.text_normalizer.normalize(text)),
        dtype=np.int32,
    )[None, :]

    if audio is not None:
        patch_len = model.tts_model.patch_size * model.tts_model.chunk_size
        if audio.shape[1] % patch_len != 0:
            pad_width = patch_len - audio.shape[1] % patch_len
            audio = np.pad(audio, ((0, 0), (0, pad_width)))
        audio = audio[None, :]

    # Generate audio
    try:
        # Determine the correct generation method
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

        # Iterate and yield, checking for cancellation
        for (chunk,) in generator:
            if cancellation_event.is_set():
                print("Generation cancelled by event.")
                break
            audio_chunk_float32 = chunk.astype(np.float32)
            yield audio_chunk_float32

    except GeneratorExit:
        print("Generator exited (likely due to client disconnect)")
    except Exception as e:
        print(f"Error in generate_audio_chunks: {e}")
        raise  # Re-raise to be caught by the worker
    finally:
        # IMPORTANT: Set the event on *any* exit (normal, error, or cancel)
        # This tells the underlying model thread to stop.
        cancellation_event.set()
        print("generate_audio_chunks finalizing.")


# ===================================================================
# 🖥️ 4. FastAPI Endpoints (Rewritten)
# ===================================================================


@app.on_event("startup")
async def startup_event():
    """Starts the background generation worker thread."""
    worker_thread = threading.Thread(target=generation_worker, daemon=True)
    worker_thread.start()


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serves the frontend playground HTML file"""
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
    """
    Asynchronous generator that polls a thread-safe
    queue for data.
    """
    while True:
        try:
            # Get item without blocking the event loop
            item = output_queue.get_nowait()

            if item is None:
                # Sentinel: generation is done
                break
            elif isinstance(item, Exception):
                # An error occurred in the worker
                print(f"Propagating worker error to client: {item}")
                raise item
            else:
                # Yield the audio chunk
                yield item

        except queue.Empty:
            # Queue was empty, sleep asynchronously
            await asyncio.sleep(poll_interval)


@app.post("/v1/audio/speech/stream")
async def stream_speech(request: SpeechRequest):
    """OpenAI-compatible streaming TTS endpoint"""
    global JOB_COUNTER
    JOB_COUNTER += 1
    job_id = JOB_COUNTER

    # 1. Create resources for this job
    output_queue = queue.Queue(maxsize=1024)  # Small buffer
    cancel_event = threading.Event()
    job = GenerationJob(request, output_queue, cancel_event, job_id)

    # 2. Try to queue the job
    try:
        GENERATION_QUEUE.put_nowait(job)
        print(f"[Endpoint] ➡️ Queued Job {job_id} for streaming.")
    except queue.Full:
        print(f"[Endpoint] 🚦 Queue is full. Rejecting Job {job_id}.")
        raise HTTPException(
            status_code=429, detail="Server is busy processing another request"
        )

    # 3. Create the streaming generator
    async def audio_stream_generator():
        """
        This async generator polls the job's output queue
        and yields data to the client.
        """
        try:
            async for chunk in poll_queue_for_chunks(output_queue):
                # Convert to 16-bit PCM bytes
                chunk_16bit = (chunk * 32767).astype(np.int16)
                yield chunk_16bit.tobytes()
            print(f"[Endpoint] Streaming finished for Job {job_id}.")
        except Exception as e:
            # This catches errors from the worker
            print(f"[Endpoint] ❌ Error in stream for Job {job_id}: {e}")
            # We can't raise an HTTPException here as headers are already sent
            # The stream will just be terminated
        finally:
            # THIS IS THE MOST IMPORTANT PART
            # When the client disconnects (or stream finishes),
            # this *always* runs.
            print(
                f"[Endpoint] ⏹️ Stream generator cleanup for Job {job_id}. Setting cancel event."
            )
            cancel_event.set()

    # 4. Return the StreamingResponse
    return StreamingResponse(
        audio_stream_generator(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE)},
    )


@app.post("/v1/audio/speech/playback")
async def playback_speech(request: PlaybackRequest):
    """
    Endpoint that plays audio on the server *as it is generated* (streaming)
    and waits for the entire audio to be played before returning.
    """
    global JOB_COUNTER, CURRENT_JOB
    JOB_COUNTER += 1
    job_id = JOB_COUNTER

    print(f"[Playback Endpoint] 🎵 Starting playback request for Job {job_id}")

    # 1. Create resources for this job
    output_queue = queue.Queue(maxsize=1024)
    cancel_event = threading.Event()
    job = GenerationJob(request, output_queue, cancel_event, job_id)

    # 2. Try to queue the job
    try:
        GENERATION_QUEUE.put_nowait(job)
        CURRENT_JOB = job  # Set this so /cancel can find it
        print(f"[Playback Endpoint] ➡️ Queued Job {job_id} for playback.")
    except queue.Full:
        print(f"[Playback Endpoint] 🚦 Queue is full. Rejecting Job {job_id}.")
        raise HTTPException(
            status_code=429, detail="Server is busy processing another request"
        )

    client_disconnected = False
    playback_start_time = time.time()
    TIMEOUT_SECONDS = 300  # 5 minute timeout

    # 3. Play audio chunks as they arrive
    try:
        # Verify audio device availability
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
            channels=1,  # Mono
            dtype=np.float32,
            latency="low",  # Low latency for real-time playback
            # IMPORTANT: blocksize should be smaller than chunk size for smooth playback
            blocksize=1024,  # Adjust based on your typical chunk size
        ) as stream:

            # Process all chunks
            async for chunk in chunks:
                # Timeout check
                elapsed = time.time() - playback_start_time
                if elapsed > TIMEOUT_SECONDS:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Playback timeout after {TIMEOUT_SECONDS} seconds",
                    )

                chunk_count += 1
                last_chunk = chunk  # Keep reference to the last chunk

                if pbar:
                    pbar.update(1)

                # Write chunk in a thread to avoid blocking the event loop
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

        # 5. Validate and finalize
        if chunk_count == 0:
            raise HTTPException(
                status_code=500, detail="Failed to generate audio (no chunks)"
            )

        total_duration = time.time() - playback_start_time
        print(
            f"[Playback Endpoint] ✅ Playback complete for Job {job_id} ({chunk_count} chunks, {total_duration:.2f}s)."
        )

        # 6. Return success response only after full playback
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
        print(f"[Playback Endpoint] 🚫 Client disconnected for Job {job_id}")
        raise HTTPException(status_code=499, detail="Client disconnected")
    except HTTPException:
        raise
    except Exception as e:
        print(
            f"[Playback Endpoint] ❌ Unexpected error during playback for Job {job_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Playback failed: {e}")
    finally:
        print(f"[Playback Endpoint] 🏁 Cleanup for Job {job_id}. Setting cancel event.")
        cancel_event.set()
        # Ensure progress bar is closed on error
        if "pbar" in locals() and pbar is not None:
            pbar.close()
        # Clear CURRENT_JOB if it was this job
        if CURRENT_JOB is job:
            CURRENT_JOB = None


@app.post("/v1/audio/speech/cancel")
async def cancel_generation():
    """Cancel the current audio generation"""
    if CURRENT_JOB is None:
        return JSONResponse(
            {"status": "success", "message": "No generation in progress"}
        )

    try:
        print(f"[Endpoint] 🔴 Received /cancel request for Job {CURRENT_JOB.job_id}")
        CURRENT_JOB.cancel_event.set()
        return JSONResponse(
            {
                "status": "success",
                "message": f"Cancellation signal sent to Job {CURRENT_JOB.job_id}",
            }
        )
    except Exception as e:
        print(f"Error cancelling generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel: {e}")


@app.get("/voices")
async def get_available_voices():
    """Get list of available cached voices"""
    try:
        voices = load_available_voices()
        return {
            "voices": voices,
            "count": len(voices),
            "cache_directory": "assets/caches",
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


# ===================================================================
# 🏁 5. Server Run
# ===================================================================


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OpenAI-compatible TTS Server")
    parser.add_argument(
        "--port", "-p",
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
    args = parser.parse_args()

    print("🚀 Starting OpenAI-compatible TTS server...")
    print(f"   Access the frontend playground at: http://{args.host}:{args.port}")
    print("   Architecture: Single-worker thread model")
    print(f"   Available voices: {len(load_available_voices())}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
