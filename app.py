import os
import torch
from diffusers import DiffusionPipeline
import streamlit as st
from collections import deque

# Initialize the pipeline
@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
    return pipe

pipe = load_pipeline()

# Function to generate an image based on a prompt
def generate_image(prompt, guidance_scale):
    try:
        guidance_scale = max(1, min(guidance_scale, 20))
        image = pipe(prompt, guidance_scale=guidance_scale).images[0]
        return image
    except IndexError as e:
        st.error(f"An error occurred during image generation: {e}")
        return None

# Streamlit app
st.title("Stable Diffusion Image Generator")
st.write("Generate stunning images using the Stable Diffusion model.")

# Initialize session state for the queue and image
if 'task_queue' not in st.session_state:
    st.session_state.task_queue = deque()
if 'image' not in st.session_state:
    st.session_state.image = None

prompt = st.text_area("Enter your prompt:")
guidance_scale = st.slider("Guidance Scale:", min_value=1, max_value=20, value=10)

if st.button("Generate Image"):
    st.session_state.task_queue.append((prompt, guidance_scale))

# Process the queue
if st.session_state.task_queue:
    current_task = st.session_state.task_queue.popleft()
    prompt, guidance_scale = current_task
    with st.spinner("Generating image..."):
        st.session_state.image = generate_image(prompt, guidance_scale)

if st.session_state.image:
    st.image(st.session_state.image, caption="Generated Image", use_column_width=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        st.success("CUDA cache cleared.")
