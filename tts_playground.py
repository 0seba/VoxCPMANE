import gradio as gr
import requests
import numpy as np
import soundfile as sf
import tempfile
import os
import threading
import queue
import time
from io import BytesIO
import json

def stream_audio_from_server(host, text, prompt_text, prompt_wav_path, response_format="wav"):
    """
    Stream audio from the TTS server and return playable audio data
    """
    if not host.startswith(('http://', 'https://')):
        host = f"http://{host}"
    
    url = f"{host.rstrip('/')}/v1/audio/speech/stream"
    
    payload = {
        "input": text,
        "prompt_text": prompt_text,
        "prompt_wav_path": prompt_wav_path,
        "response_format": response_format
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the streaming request
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        # Create a temporary file to store the audio
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False)
        temp_path = temp_file.name
        temp_file.close()  # Close so we can write to it
        
        # Read and write chunks as they arrive
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
        
        # Return the file path for Gradio to play
        return temp_path
        
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API request failed: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error processing audio: {str(e)}")

def generate_and_play(host, text, prompt_text, prompt_wav_path, response_format):
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize")
    
    if not host.strip():
        raise gr.Error("Please enter a valid host URL")
    
    # Show loading state
    yield gr.update(interactive=False, value="Generating audio..."), None, gr.update(visible=True)
    
    try:
        # Stream the audio from server
        audio_path = stream_audio_from_server(host, text, prompt_text, prompt_wav_path, response_format)
        
        # Return the audio file for playback
        yield gr.update(interactive=True, value="Generate Speech"), audio_path, gr.update(visible=False)
        
    except Exception as e:
        yield gr.update(interactive=True, value="Generate Speech"), None, gr.update(visible=False)
        raise gr.Error(str(e))

# Create the Gradio interface
with gr.Blocks(title="VoxCPM TTS Playground", css=".footer {text-align: center; margin-top: 20px;}") as demo:
    gr.Markdown("# 🎵 VoxCPM TTS Playground")
    gr.Markdown("Connect to your TTS server and generate speech with streaming playback")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔌 Server Configuration")
            host_input = gr.Textbox(
                label="Server Host",
                placeholder="http://localhost:8000",
                value="http://localhost:8000",
                info="Your TTS server URL"
            )
            
            gr.Markdown("### 🎙️ Generation Parameters")
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter your text here...",
                lines=4,
                value="""Jittery Jack's jam jars jiggled jauntily, jolting Jack's jumbled jelly-filled jars joyously.
Cindy's circular cymbals clanged cheerfully, clashing crazily near Carla's crashing crockery.
You think you can just waltz in here and cause chaos? Well, I've got news for you.""",
                info="The text you want to convert to speech"
            )
            
            prompt_text_input = gr.Textbox(
                label="Prompt Text",
                placeholder="Get in line troublemakers, and I'll take care of you.",
                value="Get in line troublemakers, and I'll take care of you.",
                info="Text that describes the prompt audio"
            )
            
            prompt_wav_input = gr.Textbox(
                label="Prompt WAV Path",
                placeholder="../ne-ts/en_013.wav",
                value="../ne-ts/en_013.wav",
                info="Path to the prompt audio file on the server"
            )
            
            format_dropdown = gr.Dropdown(
                label="Response Format",
                choices=["wav", "mp3", "flac", "opus", "aac", "pcm"],
                value="wav",
                info="Audio format for streaming"
            )
            
            generate_btn = gr.Button(
                "Generate Speech",
                variant="primary",
                size="lg"
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎧 Output Audio")
            audio_output = gr.Audio(
                label="Generated Speech",
                type="filepath",
                interactive=False,
                autoplay=True
            )
    
    gr.Markdown("### ℹ️ Instructions")
    gr.Markdown("""
    1. **Configure your server**: Enter the host URL where your TTS server is running
    2. **Set parameters**: Enter the text to synthesize and prompt configuration
    3. **Generate**: Click the button to start streaming audio generation
    4. **Listen**: The audio will play automatically as it's received from the server
    
    **Note**: This uses the streaming endpoint for real-time audio generation and playback.
    """)
    
    gr.HTML("""
    <div class="footer">
        <p>🎵 VoxCPM TTS Playground | Streaming Audio Generation</p>
        <p style="font-size: 0.8em; color: #666;">Make sure your TTS server is running and accessible</p>
    </div>
    """)
    
    # Event handlers
    generate_btn.click(
        fn=generate_and_play,
        inputs=[
            host_input,
            text_input,
            prompt_text_input,
            prompt_wav_input,
            format_dropdown
        ],
        outputs=[
            generate_btn,
            audio_output,
            status_text
        ]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        favicon_path=None  # Remove or replace with actual path if you have a favicon
    )