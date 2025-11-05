import os
import time
import threading
import queue
import numpy as np
import sounddevice as sd
from tqdm import tqdm
import uvicorn
from fastapi import FastAPI, HTTPException, Request
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
from typing import Optional
import aiofiles
from huggingface_hub import snapshot_download


REPO_ID = "seba/VoxCPM-ANE"

MODEL_PATH_PREFIX = ""
VOICE_CACHE_DIR = ""

try:
    lm_length = 8

    print(f"🚀 Downloading/loading model files from Hugging Face Hub repo: {REPO_ID}")
    MODEL_PATH_PREFIX = snapshot_download(repo_id=REPO_ID)
    print(f"✅ Model files are available at: {MODEL_PATH_PREFIX}")

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
        print(
            f"Please check your Hugging Face repo '{REPO_ID}' and ensure all files are present."
        )
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

    from src.voxcpm import VoxCPMANE

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

except (FileNotFoundError, IOError) as e:
    print(f"❌ Error: Model file not found.")
    print(f"Details: {e}")
    print(
        "Please ensure all .mlmodelc, .mlpackage, and .npy files are in the correct location."
    )
    exit()
except ImportError as e:
    print(f"❌ Error: Missing required Python package.")
    print(f"Details: {e}")
    print(
        "Please ensure you have installed all dependencies (e.g., `pip install -r requirements.txt`)"
    )
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred during model setup: {e}")
    exit()


SAMPLE_RATE = 16000
app = FastAPI(title="OpenAI Compatible TTS Server")
request_lock = threading.Lock()
is_processing = False
CACHED_VOICE_TEXT = """Jittery Jack's jam jars jiggled jauntily, jolting Jack's jumbled jelly-filled jars joyously.
Cindy's circular cymbals clanged cheerfully, clashing crazily near Carla's crashing crockery.
You think you can just waltz in here and cause chaos? Well, I've got news for you."""

# Add a global reference to track the current generation task
current_generation_task = None
cancellation_event = threading.Event()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# ------------------------------------


# OpenAI-compatible request model
class SpeechRequest(BaseModel):
    model: str = "voxcpm-0.5b"
    input: str
    voice: Optional[str] = None  # Voice name for cached audio features
    response_format: Optional[str] = (
        "wav"  # Supported: mp3, opus, aac, flac, wav, pcm [[3]]
    )
    prompt_wav_path: Optional[str] = None
    prompt_text: Optional[str] = "Get in line troublemakers, and I'll take care of you."
    # Generation parameters
    max_length: Optional[int] = 2048
    cfg_value: Optional[float] = 2.0
    inference_timesteps: Optional[int] = 10


class PlaybackRequest(SpeechRequest):
    show_progress: Optional[bool] = True


# ===========================
# Voice Cache Management
# ===========================
def load_available_voices():
    """Load available voice names from the caches directory"""
    cache_dir = "assets/caches"
    voices = []

    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith(".npy"):
                voice_name = file[:-4]  # Remove .npy extension
                voices.append(voice_name)

    return sorted(voices)


def load_voice_cache(voice_name: str):
    """Load cached audio features for a specific voice"""
    cache_path = f"assets/caches/{voice_name}.npy"

    if not os.path.exists(cache_path):
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found. Available voices: {load_available_voices()}",
        )

    try:
        cache_data = np.load(cache_path)
        return cache_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load voice cache for '{voice_name}': {str(e)}",
        )


def validate_voice_parameters(
    max_length: int, cfg_value: float, inference_timesteps: int
):
    """Validate generation parameters"""
    if max_length <= 0 or max_length > 2048:
        raise HTTPException(
            status_code=400, detail="max_length must be between 1 and 8192"
        )

    if cfg_value < 0.0 or cfg_value > 10.0:
        raise HTTPException(
            status_code=400, detail="cfg_value must be between 0.0 and 10.0"
        )

    if inference_timesteps <= 0 or inference_timesteps > 100:
        raise HTTPException(
            status_code=400, detail="inference_timesteps must be between 1 and 100"
        )


# ===========================
# Audio Generation Functions
# ===========================
def generate_audio_chunks(
    text_to_generate,
    prompt_wav_path,
    prompt_text,
    voice=None,
    max_length=4096,
    cfg_value=2.0,
    inference_timesteps=10,
):
    """Generator that yields audio chunks during generation"""
    import re

    # Validate parameters
    validate_voice_parameters(max_length, cfg_value, inference_timesteps)

    # Handle voice caching vs prompt audio
    audio_cache = None
    audio = None

    if voice is not None:
        # Use cached voice features
        audio_cache = load_voice_cache(voice)
        text = CACHED_VOICE_TEXT + " " + text_to_generate
    else:
        # Use traditional prompt audio approach
        if prompt_wav_path is not None and prompt_wav_path.strip():
            # Prompt path is provided, validate file exists
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
            # No prompt path provided, use fallback silence
            audio = np.zeros((1, 16000), dtype=np.float32)

        text = prompt_text + " " + text_to_generate

    # Normalize text
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # Tokenize text
    text_token = np.array(
        model.tts_model.text_tokenizer(model.text_normalizer.normalize(text)),
        dtype=np.int32,
    )
    text_token = text_token[None, :]  # Add batch dimension

    # Handle audio padding if using prompt audio
    if audio is not None:
        patch_len = model.tts_model.patch_size * model.tts_model.chunk_size
        if audio.shape[1] % patch_len != 0:
            pad_width = patch_len - audio.shape[1] % patch_len
            audio = np.pad(audio, ((0, 0), (0, pad_width)))

    # Generate audio using the appropriate method
    try:
        if voice is not None:
            # Use cached voice method
            for (chunk,) in model.tts_model._generate_threaded_prompt_processing(
                text_token,
                audio=None,
                audio_cache=audio_cache,
                max_length=max_length,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            ):
                # Check for cancellation before yielding each chunk
                if cancellation_event.is_set():
                    print("Generation cancelled by user interrupt.")
                    break
                audio_chunk_float32 = chunk.astype(np.float32)
                yield audio_chunk_float32
        else:
            # Use traditional prompt method
            for (chunk,) in model.tts_model._generate_threaded_prompt_processing(
                text_token,
                audio[None, :],
                max_length=max_length,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            ):
                # Check for cancellation before yielding each chunk
                if cancellation_event.is_set():
                    print("Generation cancelled by user interrupt.")
                    break
                audio_chunk_float32 = chunk.astype(np.float32)
                yield audio_chunk_float32
    except GeneratorExit:
        print("Generator exited (likely due to cancellation)")
        raise


def generate_complete_audio(
    text_to_generate,
    prompt_wav_path,
    prompt_text,
    voice=None,
    max_length=4096,
    cfg_value=2.0,
    inference_timesteps=10,
):
    """Generates complete audio and returns as numpy array"""
    chunks = []
    for chunk in generate_audio_chunks(
        text_to_generate,
        prompt_wav_path,
        prompt_text,
        voice,
        max_length,
        cfg_value,
        inference_timesteps,
    ):
        chunks.append(chunk)

    if chunks:
        return np.concatenate(chunks)
    return None


# ===========================
# Server Endpoints
# ===========================


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serves the frontend playground HTML file"""
    try:
        async with aiofiles.open("index.html", mode="r") as f:
            content = await f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error</h1><p><code>index.html</code> not found. Please make sure it's in the same directory as this Python script.</p>",
            status_code=404,
        )


# -----------------------------------------


@app.post("/v1/audio/speech/stream")
async def stream_speech(request: SpeechRequest):
    """OpenAI-compatible streaming TTS endpoint"""
    global is_processing

    # Acquire lock to ensure only one request at a time
    if not request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429, detail="Server is busy processing another request"
        )

    is_processing = True
    try:

        def audio_stream():
            try:
                for chunk in generate_audio_chunks(
                    request.input,
                    request.prompt_wav_path,
                    request.prompt_text,
                    voice=request.voice,
                    max_length=request.max_length,
                    cfg_value=request.cfg_value,
                    inference_timesteps=request.inference_timesteps,
                ):
                    # Convert to 16-bit PCM for streaming
                    # This is what the frontend AudioContext will expect
                    chunk_16bit = (chunk * 32767).astype(np.int16)
                    yield chunk_16bit.tobytes()
            except Exception as e:
                print(f"Error during audio generation: {e}")
                # You might want to yield an error message or handle differently
            finally:
                global is_processing
                is_processing = False
                if request_lock.locked():
                    request_lock.release()
                print("Streaming finished, lock released.")

        # For the frontend, we are streaming raw 16-bit PCM.
        # The 'response_format' from the request is less relevant here
        # as the frontend logic is built to handle this specific format.
        # We'll set the media type to 'application/octet-stream'
        # to indicate raw binary data.
        return StreamingResponse(
            audio_stream(),
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(SAMPLE_RATE)},  # Send sample rate as a header
        )

    except Exception as e:
        if request_lock.locked():
            request_lock.release()
        is_processing = False
        print(f"Error in stream_speech endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech")
async def generate_speech(request: SpeechRequest):
    """OpenAI-compatible TTS endpoint that returns complete audio"""
    global is_processing

    # Acquire lock to ensure only one request at a time
    if not request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429, detail="Server is busy processing another request"
        )

    is_processing = True
    try:
        # Generate complete audio
        audio_data = generate_complete_audio(
            request.input,
            request.prompt_wav_path,
            request.prompt_text,
            voice=request.voice,
            max_length=request.max_length,
            cfg_value=request.cfg_value,
            inference_timesteps=request.inference_timesteps,
        )

        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # Save to temporary file based on response_format [[3]]
        with tempfile.NamedTemporaryFile(
            suffix=f".{request.response_format}", delete=False
        ) as tmp_file:
            if request.response_format.lower() == "pcm":
                # Raw PCM data (as float32, which is soundfile's default)
                soundfile.write(
                    tmp_file.name,
                    audio_data,
                    SAMPLE_RATE,
                    format="RAW",
                    subtype="PCM_16",
                )
            else:
                # Use soundfile for other formats
                soundfile.write(
                    tmp_file.name,
                    audio_data,
                    SAMPLE_RATE,
                    format=request.response_format.lower(),
                )

            tmp_path = tmp_file.name

        # Return file response
        media_type = {
            "wav": "audio/wav",
            "pcm": "audio/pcm",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "opus": "audio/opus",
            "aac": "audio/aac",
        }.get(request.response_format.lower(), "audio/wav")

        return FileResponse(
            tmp_path,
            media_type=media_type,
            filename=f"speech.{request.response_format}",
            background=lambda: os.unlink(tmp_path),  # Clean up temp file
        )

    except Exception as e:
        print(f"Error in generate_speech endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        is_processing = False
        if request_lock.locked():
            request_lock.release()


@app.post("/v1/audio/speech/playback")
async def playback_speech(request: PlaybackRequest, req: Request):
    """Endpoint that plays audio on the server with progress bars"""
    global is_processing

    # Acquire lock to ensure only one request at a time
    if not request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429, detail="Server is busy processing another request"
        )

    is_processing = True

    try:
        # Setup progress tracking
        generated_frames_count = 0
        played_frames_count = 0
        generation_complete = False
        audio_queue = queue.Queue(maxsize=100)
        count_lock = threading.Lock()

        # Thread for generation
        def generation_thread():
            nonlocal generated_frames_count, generation_complete
            try:
                for chunk in generate_audio_chunks(
                    request.input,
                    request.prompt_wav_path,
                    request.prompt_text,
                    voice=request.voice,
                    max_length=request.max_length,
                    cfg_value=request.cfg_value,
                    inference_timesteps=request.inference_timesteps,
                ):
                    audio_queue.put(chunk)
                    with count_lock:
                        generated_frames_count += len(chunk)
            except Exception as e:
                print(f"Error in generation_thread: {e}")
                audio_queue.put(None)  # Ensure sentinel is put on error
            finally:
                with count_lock:
                    generation_complete = True
                audio_queue.put(None)  # Sentinel value

        # Start generation thread
        gen_thread = threading.Thread(target=generation_thread)
        gen_thread.start()

        # Setup progress bars if requested
        pbar_gen = pbar_play = None
        if request.show_progress:
            initial_total_frames = int(30 * SAMPLE_RATE)
            pbar_gen = tqdm(
                total=initial_total_frames,
                desc="📢 Generation",
                unit="frame",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_frames} frames",
            )
            pbar_play = tqdm(
                total=initial_total_frames,
                desc="🎧 Playback ",
                unit="frame",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_frames} frames",
            )

        # Playback loop
        try:
            with sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32"
            ) as stream:
                while True:
                    chunk = audio_queue.get()
                    if chunk is None:  # End of generation
                        break

                    stream.write(chunk)

                    # Update counters
                    with count_lock:
                        gen_frames = generated_frames_count
                        played_frames_count += len(chunk)
                        play_frames = played_frames_count

                    # Update progress bars if enabled
                    if request.show_progress:
                        pbar_gen.n = gen_frames
                        pbar_play.n = play_frames

                        # Dynamically increase total if needed
                        if gen_frames >= pbar_gen.total:
                            new_total = int(gen_frames * 1.5)
                            pbar_gen.total = new_total
                            pbar_play.total = new_total

                        pbar_gen.refresh()
                        pbar_play.refresh()

            # Final progress bar updates
            if request.show_progress:
                with count_lock:
                    final_gen_frames = generated_frames_count
                    final_play_frames = played_frames_count

                pbar_gen.n = final_gen_frames
                pbar_play.n = final_play_frames

                if final_gen_frames > 0:
                    pbar_gen.total = final_gen_frames
                    pbar_play.total = final_gen_frames

                pbar_gen.refresh()
                pbar_play.refresh()
                pbar_gen.close()
                pbar_play.close()

        except Exception as e:
            if request.show_progress:
                if pbar_gen:
                    pbar_gen.close()
                if pbar_play:
                    pbar_play.close()
            print(f"Playback error: {e}")
            raise HTTPException(status_code=500, detail=f"Playback error: {str(e)}")

        finally:
            gen_thread.join()
            print("Playback thread joined.")

        # Return success response
        duration = played_frames_count / SAMPLE_RATE
        return JSONResponse(
            {
                "status": "success",
                "message": "Audio playback completed on server",
                "duration_seconds": round(duration, 2),
                "frames_played": played_frames_count,
                "text": request.input,
            }
        )

    except Exception as e:
        print(f"Error in playback_speech endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        is_processing = False
        if request_lock.locked():
            request_lock.release()
        print("Playback endpoint finished, lock released.")


@app.post("/v1/audio/speech/cancel")
async def cancel_generation():
    """Cancel the current audio generation"""
    global current_generation_task, is_processing

    if not is_processing:
        return JSONResponse(
            {"status": "success", "message": "No generation in progress"}
        )

    try:
        # Set the cancellation event to signal ongoing generation to stop
        cancellation_event.set()

        # Release the lock immediately
        if request_lock.locked():
            request_lock.release()

        is_processing = False
        current_generation_task = None

        print("Generation cancelled by user, lock released.")

        # Reset the cancellation event after a short delay to allow new generations
        threading.Timer(1.0, lambda: cancellation_event.clear()).start()

        return JSONResponse(
            {"status": "success", "message": "Audio generation cancelled successfully"}
        )

    except Exception as e:
        print(f"Error cancelling generation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel generation: {str(e)}"
        )


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
        raise HTTPException(status_code=500, detail=f"Failed to load voices: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "is_processing": is_processing,
        "sample_rate": SAMPLE_RATE,
        "model": "voxcpm-0.5b",
        "available_voices": len(load_available_voices()),
    }


if __name__ == "__main__":
    print("🚀 Starting OpenAI-compatible TTS server...")
    print("   Access the frontend playground at: http://localhost:8000")
    print("Endpoints:")
    print("  GET / - Frontend Playground")
    print("  POST /v1/audio/speech/stream - Streaming audio generation (for frontend)")
    print("  POST /v1/audio/speech - Complete audio file generation")
    print("  POST /v1/audio/speech/playback - Server-side playback with progress bars")
    print("  POST /v1/audio/speech/cancel - Cancel current audio generation")
    print("  GET /voices - List available cached voices")
    print("  GET /health - Health check")
    print(f"\n📢 Available voices: {len(load_available_voices())}")
    print("💡 New features:")
    print("  - Voice caching: Use 'voice' parameter to select cached voices")
    print("  - Generation parameters: max_length, cfg_value, inference_timesteps")
    print(
        "  - Lock release on stop: Fixed issue where locks weren't released when stopping playback"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
