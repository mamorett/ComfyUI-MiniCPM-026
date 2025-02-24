# MiniCPM-o-26 Image Captioning Node for ComfyUI

This repository provides an **Image Captioning Node** for **ComfyUI**, utilizing the **MiniCPM-o-2_6** model to generate captions for input images.

## Features
- Automatically loads the **MiniCPM-o-2_6** model from **Hugging Face**.
- Accepts **image input** and a **text prompt**.
- Generates a **caption** describing the image.
- Fully integrated as a **custom node** for ComfyUI.

## Installation
### Prerequisites
Ensure you have **ComfyUI** installed and configured.

### Steps to Install
1. **Clone or Download** this repository and place it inside the `custom_nodes` directory of your ComfyUI installation.
   ```bash
   cd path/to/ComfyUI/custom_nodes
   git clone https://github.com/your-repo/ComfyUI-MiniCPM-Captioning.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install torch transformers pillow numpy
   ```

## Usage
1. **Start ComfyUI**:
   ```bash
   python main.py
   ```
2. **Add the Node in ComfyUI**:
   - Locate `Image Captioning (MiniCPM-o-2_6)` in the **Trithemius/MiniCPM-o-2_6** category.
   - Connect an **image input** and **string prompt**.
   - Run the workflow to generate a caption.

## Node Details
### Inputs
| Parameter | Type  | Description |
|-----------|------|-------------|
| `image`  | IMAGE | Input image for captioning |
| `prompt` | STRING | Custom prompt to influence caption generation |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `caption` | STRING | Generated image caption |

### Example Workflow

![Example Workflow](https://github.com/ComfyUI-MiniCPM-Captioning/ComfyUI-MiniCPM-Captioning/blob/main/Screenshot_2023-09-17_19-31-35.png)


## Model Details
- **Model Used:** `MiniCPM-o-2_6-nf4`
- **Source:** [Hugging Face](https://huggingface.co/2dameneko/MiniCPM-o-2_6-nf4)
- **Device Support:** CPU & CUDA

## Troubleshooting
- **Image Format Issues**: Ensure the image is in a standard format (RGB, 3-channel tensor, or NumPy array).
- **CUDA Errors**: If running on **CPU**, modify `device_map='cpu'` in the `load_model()` method.

## License
This project follows the **MIT License**. See the `LICENSE` file for more details.

## Acknowledgments
- Inspired by [2dameneko's MiniCPM model](https://huggingface.co/2dameneko/MiniCPM-o-2_6-nf4).
- Developed for **ComfyUI** community enhancements.

