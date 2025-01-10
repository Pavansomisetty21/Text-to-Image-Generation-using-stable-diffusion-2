# Text-to-Image-Generation-using-stable-diffusion-2
In this we generate Images with a prompt(Text)
Here’s a detailed explanation of the functions used in your project

---

### **Explanation of Functions**

#### **1. StableDiffusionPipeline**
The `StableDiffusionPipeline` is the main function from the `diffusers` library used to generate images from text prompts. 

**Purpose**: 
It integrates the components of a Stable Diffusion model (text encoder, UNet, scheduler, and VAE decoder) into a single pipeline for streamlined text-to-image generation.

**Key Features**:
- Converts natural language prompts into visual outputs.
- Supports precision optimizations like FP16 for GPU acceleration.

**Important Parameters**:
- `from_pretrained()`:
  - **`model_id`**: The identifier for the pre-trained Stable Diffusion model (e.g., `stabilityai/stable-diffusion-2`).
  - **`torch_dtype`**: Data type (e.g., `torch.float16`) to reduce memory usage.
  - **`revision`**: Model weights version, such as "fp16" for half-precision.
  - **`use_auth_token`**: Token for accessing Hugging Face models.
- `.to()`:
  - Specifies the device (`cuda` or `cpu`) where the pipeline will run.

**Usage**:
```python
image_gen_model = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2", torch_dtype=torch.float16,
    revision="fp16", use_auth_token='YOUR_HF_TOKEN'
).to("cuda")
```

---

#### **2. generate_image(prompt, model)**
This function wraps the pipeline to generate an image based on a user-provided text prompt.

**Workflow**:
1. Calls the `StableDiffusionPipeline` with the text prompt.
2. Configures:
   - **`num_inference_steps`**: Number of diffusion steps, controlling quality and generation time.
   - **`guidance_scale`**: Controls adherence to the prompt.
   - **`generator`**: Ensures reproducibility with a fixed seed.
3. The output image is resized to a specified dimension (default: 400x400 pixels).

**Example**:
```python
image = generate_image("A futuristic city at night", image_gen_model)
```

---

#### **3. CFG Class**
The `CFG` class stores configuration parameters to make the code modular and customizable.

**Attributes**:
- **Device**: GPU (`cuda`) or CPU.
- **Seed**: Ensures reproducibility of results.
- **Generator**: A PyTorch random generator seeded for consistency.
- **Image Generation**:
  - Steps (`image_gen_steps`): Number of diffusion iterations.
  - Model ID (`image_gen_model_id`): Specifies the Stable Diffusion version.
  - Size (`image_gen_size`): Image dimensions.
  - Guidance Scale (`image_gen_guidance_scale`): Prompt adherence.

---



---

# Text-to-Image Generation using Stable Diffusion v2

## Overview
This project demonstrates the use of **Stable Diffusion v2**, a powerful text-to-image generation model. By providing textual descriptions, users can create visually compelling images that reflect their inputs.

## Features
- Generate high-quality images from text prompts.
- Configurable settings for image resolution, quality, and creativity.
- GPU-accelerated performance with FP16 precision.
- Modular and reproducible design for consistent results.

---

## How It Works

### **1. Stable Diffusion Pipeline**
The project uses `StableDiffusionPipeline` from the `diffusers` library:
- Text prompts are encoded and processed through a pre-trained model.
- Images are generated iteratively using diffusion techniques.
- Post-processed images are returned for visualization or storage.

### **2. Key Functions**
#### **StableDiffusionPipeline**
Loads and initializes the pre-trained Stable Diffusion v2 model.

#### **generate_image(prompt, model)**
Generates an image from a given text prompt:
- Input: Text prompt and Stable Diffusion model.
- Output: Resized PIL image.

#### **CFG Class**
Holds project settings for customization:
- Device (CPU/GPU).
- Model ID.
- Image generation steps, size, and guidance scale.

---


## Usage

1. **Load the Model**:
   Initialize the pipeline:
   ```python
   from diffusers import StableDiffusionPipeline

   image_gen_model = StableDiffusionPipeline.from_pretrained(
       "stabilityai/stable-diffusion-2", torch_dtype=torch.float16,
       revision="fp16", use_auth_token="YOUR_HF_TOKEN"
   ).to("cuda")
   ```

2. **Generate Images**:
   Use the `generate_image` function:
   ```python
   image = generate_image("A forest with glowing mushrooms", image_gen_model)
   ```

3. **Display or Save**:
   Visualize the image:
   ```python
   import matplotlib.pyplot as plt

   plt.imshow(image)
   plt.axis("off")
   plt.show()
   ```

---

## Configuration

Modify the `CFG` class to change default settings:
- **Seed**: Ensure consistent results.
- **Inference Steps**: Adjust image quality.
- **Guidance Scale**: Control prompt adherence.

---

## Example Output

| **Prompt**                                    | **Generated Image**                |
|-----------------------------------------------|------------------------------------|
| "spiderman and batman fighting each other"    | ![City](examples/futuristic_city.png) |
| "A forest with glowing mushrooms"             | ![Forest](examples/forest.png)     |

---


## License
This project is licensed under the [MIT License](LICENSE).

--- 


