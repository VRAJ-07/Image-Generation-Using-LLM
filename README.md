# Image Generation Using LLM - Stable Diffusion Image Generator with Streamlit

This project is a web application that allows users to generate images using the Stable Diffusion model from Hugging Face. Built with Streamlit, it provides an interactive interface where users can input prompts and adjust parameters to generate custom images.

## **Features**

- **Interactive Prompt Input**: Enter custom text prompts to generate images.
- **Adjustable Guidance Scale**: Control the creativity of the generated images by adjusting the guidance scale slider.
- **GPU Acceleration**: Utilizes CUDA if available for faster image generation.
- **Queue Management**: Handles multiple image generation tasks using a queue system.
- **Cache Optimization**: Clears CUDA cache after image generation to optimize GPU memory usage.

## **Technologies Used**

- **Python 3.8+**
- **PyTorch**
- **Hugging Face Diffusers**
- **Streamlit**
- **CUDA (Optional for GPU acceleration)**

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/VRAJ-07/Image-Generation-Using-LLM.git
cd Image-Generation-Using-LLM
```

### **2. Create a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### **3. Install Dependencies**

Ensure you have PyTorch installed with CUDA support if you plan to use GPU acceleration.

```bash
pip install -r requirements.txt
```

**Note:** For CUDA support with PyTorch, you might need to install it separately based on your CUDA version. Visit the [PyTorch Getting Started](https://pytorch.org/get-started/locally/) page for more details.

### **4. Run the Streamlit App**

```bash
streamlit run app.py
```

## **Usage**

1. **Open the Web App**: After running the Streamlit command, a local URL will be provided (usually `http://localhost:8501`). Open it in your web browser.

2. **Enter a Prompt**: In the text area labeled "Enter your prompt," type a description of the image you want to generate.

3. **Adjust Guidance Scale**: Use the slider to set the guidance scale between 1 and 20. A higher value encourages the model to follow the prompt more closely.

4. **Generate Image**: Click the "Generate Image" button. The app will process your request and display the generated image.

5. **View and Save**: Once generated, the image will be displayed. You can right-click and save the image if desired.

## **Example**

- **Prompt**: "A serene landscape with mountains during sunset."
- **Guidance Scale**: 7

**Result**: The app generates an image of a peaceful mountain landscape illuminated by the colors of the sunset.

https://github.com/user-attachments/assets/cb57b9d4-51d5-4886-b0e6-ef76bbe92487

## **Notes**

- **CUDA Support**: If you have a compatible NVIDIA GPU, the app will use CUDA for faster image generation. Ensure that the correct version of PyTorch with CUDA support is installed.

- **Model Download**: The first time you run the app, it will download the Stable Diffusion model from Hugging Face, which may take some time.

- **Queue System**: The app uses a queue (`collections.deque`) to manage multiple image generation requests.

## **Troubleshooting**

- **IndexError during Image Generation**: If you encounter an `IndexError`, it may be due to an issue with the prompt or model. Try modifying your prompt or check your installation.

- **CUDA Out of Memory**: If you run out of GPU memory, try lowering the `guidance_scale` or ensure no other processes are using the GPU.

- **Clearing Cache**: The app clears the CUDA cache after image generation to free up memory.

## **Dependencies Explanation**

- **torch**: PyTorch framework for tensor computations and deep learning.

- **diffusers**: Provides pre-trained diffusion models and pipelines.

- **transformers**: Required by `diffusers` for model loading and tokenization.

- **safetensors**: Enables safe and fast loading of models.

- **streamlit**: Framework for building interactive web applications.
