# simple-custom-image-generation
Simple custom image generation without retraining a new model

Process Description:

1️⃣ Build an image gallery with captions.

2️⃣ Take a prompt (e.g., “Komtar in LEGO”).

3️⃣ Find the closest base image through caption search.

4️⃣ Use that base with ControlNet SDXL, AutoencoderKL, and StableDiffusionXLControlNetPipeline.

5️⃣ Generate an image that aligned closely with the user’s request.
