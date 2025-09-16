import os

# Suppress TensorFlow INFO & WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import load_image

# -------------------------
# Configuration constants
# -------------------------
CSV_PATH = Path("image_captions.csv")
IMAGE_DIR = Path("./images")
MAX_IMAGE_DIM = 800

CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0-mid"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
BASE_MODEL = "Lykon/dreamshaper-xl-lightning"

LOW_THRESHOLD = 100
HIGH_THRESHOLD = 200


# -------------------------
# Core functions
# -------------------------
def select_best_image(prompt: str, csv_path: Path, image_dir: Path) -> Path:
    """
    Select the image whose caption is most similar to the given prompt.

    Similarity is computed using TF-IDF vectorization and cosine similarity.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    if {"image_name", "image_caption"} - set(df.columns):
        raise ValueError("CSV must contain 'image_name' and 'image_caption' columns")

    captions = df["image_caption"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(captions + [prompt])

    # Compute cosine similarity between prompt and captions
    prompt_vec = tfidf[-1]
    similarities = cosine_similarity(tfidf[:-1], prompt_vec).ravel()

    best_idx = similarities.argmax()
    best_image_name = df.iloc[best_idx]["image_name"]

    image_path = image_dir / best_image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def preprocess_image(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """
    Load the image, resize if below `max_dim`, apply Canny edge detection,
    and return as a PIL image suitable for ControlNet.
    """
    original_image = load_image(str(image_path))
    width, height = original_image.size

    if width < max_dim and height < max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_image = original_image

    image_np = np.array(resized_image)
    edges = cv2.Canny(image_np, LOW_THRESHOLD, HIGH_THRESHOLD)
    # This method is fine, but using the exact one from Script A for consistency
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)

    return Image.fromarray(edges)


def build_pipeline() -> StableDiffusionXLControlNetPipeline:
    """
    Construct and return a Stable Diffusion XL pipeline with ControlNet and VAE.
    """
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, torch_dtype=torch.float16, use_safetensors=True
    )
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL, torch_dtype=torch.float16, use_safetensors=True
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    
    # Lightning models require a specific scheduler for fast inference.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe

def generate_images(
    pipe: StableDiffusionXLControlNetPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    conditioning_image: Image.Image,
    scale: float = 0.5,
) -> list[Image.Image]:
    """
    Generate images using the pipeline and return them as a list of PIL Images.
    """
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=scale,
        image=conditioning_image,
        guess_mode=True,
        guidance_scale=3,
        num_inference_steps=7, 
    )
    return result.images


# -------------------------
# Entrypoint
# -------------------------
def main() -> None:
    prompt = input("Describe your image: ")
    
    negative_prompt = "low quality, bad quality, sketches"

    # Assuming 'image_captions.csv' and the 'images' directory are set up correctly
    try:
        best_image_path = select_best_image(prompt, CSV_PATH, IMAGE_DIR)
        print(f"Selected reference image: {best_image_path}")
        conditioning_image = preprocess_image(best_image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error selecting reference image: {e}")
        print("Using a placeholder image instead.")
        # Create a dummy blank image if the selection process fails
        conditioning_image = Image.new('RGB', (1024, 1024), 'gray')


    pipe = build_pipeline()

    outputs = generate_images(pipe, prompt, negative_prompt, conditioning_image)

    for idx, img in enumerate(outputs):
        out_path = Path(f"output_{idx}.png")
        img.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()