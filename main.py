import os


os.environ['GLOG_minloglevel'] = '2'
import logging

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)


import numpy as np
import sounddevice as sd
import time
from tqdm import tqdm
import queue
import threading
import coremltools as ct
from src.voxcpm import VoxCPMANE
# from netts.models.voxcpm.voxcpm_parallel import VoxCPMANE

import os
import soxr
import soundfile

# sys.path.insert(0, "../ne-ts")

# ===========================
# Model Setup (Assuming this part is correct and unchanged)
# ===========================
try:
    lm_length = 8

    # base_lm_mlmodel = ct.models.CompiledMLModel("base_lm_4_w8.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    # base_lm_mlmodel = ct.models.CompiledMLModel("voxcpm_base_lm_4_channels_first.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    # base_lm_mlmodel = ct.models.CompiledMLModel("voxcpm_base_lm_1_channels_first.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    # base_lm_mlmodel = ct.models.CompiledMLModel("base_lm_mf_w8.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"length_{lm_length}")
    # residual_lm_mlmodel = ct.models.CompiledMLModel("residual_lm_mf_f16.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"length_{lm_length}")
    # locdit_mlmodel = ct.models.CompiledMLModel("locdit_w8.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    # locdit_mlmodel = ct.models.CompiledMLModel("locdit_w8.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    locdit_mlmodel = ct.models.CompiledMLModel("../ne-ts/locdit_f16.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    # projections_mlmodel = ct.models.CompiledMLModel("../ne-ts/projections_1.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    projections_mlmodel = ct.models.MLModel("../ne-ts/projections_1_t.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
    feature_encoder_mlmodel = ct.models.CompiledMLModel("../ne-ts/voxcpm_feat_encoder_ane_enum_12.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    # residual_lm_mlmodel = ct.models.CompiledMLModel("../ne-ts/voxcpm_residual_lm_4_channels_first.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    # residual_lm_mlmodel = ct.models.CompiledMLModel("../ne-ts/voxcpm_residual_lm_1_channels_first.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    # residual_lm_mlmodel = ct.models.CompiledMLModel("../ne-ts/voxcpm_residual_lm_8_channels_first.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
    fsq_mlmodel = ct.models.CompiledMLModel("../ne-ts/fsq_layer.mlmodelc", compute_units=ct.ComputeUnit.CPU_ONLY)
    audio_vae_decoder_mlmodel = ct.models.CompiledMLModel("../ne-ts/voxcpm_audio_vae_decoder_length_24.mlmodelc", compute_units=ct.ComputeUnit.CPU_ONLY)
    audio_vae_encoder_mlmodel = ct.models.CompiledMLModel("../ne-ts/voxcpm_audio_vae_encoder_enum_length_17920.mlmodelc", compute_units=ct.ComputeUnit.CPU_ONLY)
    base_lm_embed_tokens = np.load("../ne-ts/base_lm_embeds.npy")

        
    model = VoxCPMANE(
        "openbmb/VoxCPM-0.5B",
        # base_lm_mlmodel,
        # residual_lm_mlmodel,
        # "base_lm_mf_w8.mlmodelc/",
        "../ne-ts/base_lm_mf_f16.mlmodelc/",
        "../ne-ts/residual_lm_mf_f16.mlmodelc/",
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


except FileNotFoundError as e:
    print(f"❌ Error: Model file not found: {e.filename}")
    print("Please ensure all .mlmodelc and .npy files are in the same directory as the script.")
    exit()

text = """
Jittery Jack's jam jars jiggled jauntily, jolting Jack's jumbled jelly-filled jars joyously.
Cindy's circular cymbals clanged cheerfully, clashing crazily near Carla's crashing crockery.
You think you can just waltz in here and cause chaos? Well, I've got news for you.
"""
text = """Jittery Jack's jam jars jiggled jauntily, jolting Jack's jumbled jelly-filled jars joyously.\nCindy's circular cymbals clanged cheerfully, clashing crazily near Carla's crashing crockery.\nYou think you can just waltz in here and cause chaos? Well, I've got news for you."""

# These are now only used if the model requires them.
prompt_text = "Get in line troublemakers, and I'll take care of you."
prompt_speech_path = "../ne-ts/en_013.wav"

# ===========================
# Streaming & Playback Configuration
# ===========================
SAMPLE_RATE = 16000
INITIAL_BUFFER_SECONDS = 0.0
INITIAL_BUFFER_FRAMES = int(INITIAL_BUFFER_SECONDS * SAMPLE_RATE)

# Thread-safe queue to pass audio from the generator to the main thread.
audio_queue = queue.Queue(maxsize=100)

# Thread-safe counters for progress bars.
generated_frames_count = 0
played_frames_count = 0
generation_complete = False
count_lock = threading.Lock()

ttft = None
# ===========================
# Audio Generation Thread
# ===========================
def generation_thread_func(text_to_generate):
    """Generates audio in a separate thread and puts it into the queue."""
    global generated_frames_count, generation_complete, ttft


    import re
    text = prompt_text + " " + text_to_generate
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)

    text_token = np.array(model.tts_model.text_tokenizer(model.text_normalizer.normalize(text)), dtype=np.int32)

    audio, sr = soundfile.read(prompt_speech_path)
    audio = soxr.resample(audio, sr, 16_000, "HQ")
    if audio.ndim == 1:
        audio = audio[None, :]
    patch_len = model.tts_model.patch_size * model.tts_model.chunk_size
    if audio.shape[1] % patch_len != 0:
        pad_width = patch_len - audio.shape[1] % patch_len
        audio = np.pad(audio, ((0, 0), (0, pad_width)))

    print(patch_len, audio.shape)

    try:
        chunks = []
        for chunk, in model.tts_model._generate_threaded_prompt_processing(
            text_token[None, :],
            audio[None, :],
            # None,
        ):
            
            # chunks.append(chunk)
            if ttft is not None:
                print("\n\n\nTTFT:", time.time() - ttft)
                ttft = None
            audio_chunk_float32 = chunk.astype(np.float32)
            audio_queue.put(audio_chunk_float32)
            with count_lock:
                generated_frames_count += len(audio_chunk_float32)
    
# model.tts_model._process_prompt_fully_pipelined(text_token[None, :], None);

    # try:
        # for chunk in model.generate_streaming(
        #     text=text_to_generate,
        #     prompt_wav_path=prompt_speech_path,
        #     prompt_text=prompt_text,
        #     cfg_value=2.0,
        #     inference_timesteps=10,
        #     normalize=True,
        #     denoise=False,
        #     retry_badcase=False,
        # ):
        #     if ttft is not None:
        #         print("\n\n\nTTFT:", time.time() - ttft)
        #         ttft = None
        #     print("\n\n new outer time", time.time() - new_outer)
        #     new_outer = time.time()
        #     audio_chunk_float32 = chunk.astype(np.float32)
        #     audio_queue.put(audio_chunk_float32)
        #     with count_lock:
        #         generated_frames_count += len(audio_chunk_float32)

        

    finally:
        # Mark generation as complete and put a sentinel value to signal the end of generation.
        with count_lock:
            generation_complete = True
        audio_queue.put(None)

# ===========================
# Main Execution Logic
# ===========================

# Get user input before starting anything.
print("Please enter the text you want the model to say:")
user_text = input("> ")
ttft = time.time()

if not user_text.strip():
    print("No input provided. Exiting.")
    exit()

# 1. Start the audio generation in the background.
gen_thread = threading.Thread(target=generation_thread_func, args=(user_text,))
gen_thread.start()

# 2. Setup progress bars.
initial_total_frames = int(30 * SAMPLE_RATE)
pbar_gen = tqdm(total=initial_total_frames, desc="📢 Generation", unit="frame", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} frames')
pbar_play = tqdm(total=initial_total_frames, desc="🎧 Playback ", unit="frame", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} frames')

# 3. Pre-buffer audio.
while generated_frames_count < INITIAL_BUFFER_FRAMES and gen_thread.is_alive():
    pbar_gen.n = generated_frames_count
    pbar_gen.refresh()
    time.sleep(0.05)
pbar_gen.n = generated_frames_count
pbar_gen.refresh()
print("✅ Buffer ready! Starting playback...\n")


# 4. Main playback loop using stream.write()
try:
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
        # chunks = []
        # for chunk in chunks:
        while True:
            # Get the next chunk from the queue. This blocks until a chunk is available.
            chunk = audio_queue.get()
            # chunk = chunks.take(0)

            # The sentinel value 'None' indicates the end of the stream.
            if chunk is None:
                # chunks = np.concatenate(chunks)
                # soundfile.write("out.wav", chunks, SAMPLE_RATE)
                # sd.play(chunks, samplerate=SAMPLE_RATE)
                break

            # chunks.append(chunk)
            # Write the audio chunk to the stream.
            stream.write(chunk)

            # Update counters and progress bars
            with count_lock:
                gen_frames = generated_frames_count
                played_frames_count += len(chunk)
                play_frames = played_frames_count

            pbar_gen.n = gen_frames
            pbar_play.n = play_frames

            # Dynamically increase the total size of the progress bars.
            if gen_frames >= pbar_gen.total:
                new_total = int(gen_frames * 1.5)
                pbar_gen.total = new_total
                pbar_play.total = new_total

            pbar_gen.refresh()
            pbar_play.refresh()
        print("✅ Audio playback completed")


except KeyboardInterrupt:
    print("\n⏹️ User interrupted. Stopping...")
except Exception as e:
    print(f"\nAn error occurred: {e}")

finally:
    # 5. Final cleanup.
    # Wait for the generation thread to complete first
    gen_thread.join()
    
    # Ensure progress bars show the final numbers.
    with count_lock:
        final_gen_frames = generated_frames_count
        final_play_frames = played_frames_count
    
    pbar_gen.n = final_gen_frames
    pbar_play.n = final_play_frames
    
    # Set both progress bars to the same total (total generated)
    if final_gen_frames > 0:
        pbar_gen.total = final_gen_frames
        pbar_play.total = final_gen_frames
    
    pbar_gen.refresh()
    pbar_play.refresh()

    pbar_gen.close()
    pbar_play.close()
    
    # Display final statistics
    duration_gen = final_gen_frames / SAMPLE_RATE
    duration_play = final_play_frames / SAMPLE_RATE
    print(f"\n✅ Playback finished!")
    print(f"   Generated: {duration_gen:.2f}s ({final_gen_frames} frames)")
    print(f"   Played: {duration_play:.2f}s ({final_play_frames} frames)")
