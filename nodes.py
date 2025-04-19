import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import folder_paths
import comfy.model_management as mm
import gc
import math

class FluxResolutionNode:
    def __init__(self):
        self.add_input('image', 'Image', 'Input image for resolution calculation')
        self.add_input('megapixel', 'Megapixel', 1.0, optional=True)
        self.add_output('width', 'Width', 0)
        self.add_output('height', 'Height', 0)
        
        # Predefined FLUX resolution standards (add more as needed)
        self.flux_resolutions = [
            (1024, 768),   # 4:3
            (1280, 720),   # 16:9
            (1920, 1080),  # 16:9
            (2048, 1536),  # 4:3
            (2560, 1440),  # 16:9
            (3840, 2160),  # 16:9 (4K)
            (4096, 3072),  # 4:3
        ]

    def compute(self):
        image = self.get_input('image')
        if not image:
            raise ValueError("Input image is required")
        
        megapixel = self.get_input('megapixel', default=1.0)
        target_area = megapixel * 1e6
        
        original_width = image.width
        original_height = image.height
        original_ratio = original_width / original_height
        
        # Filter FLUX resolutions with closest aspect ratio
        closest_ratio = None
        valid_resolutions = []
        for w, h in self.flux_resolutions:
            ratio = w / h
            if closest_ratio is None or abs(ratio - original_ratio) < abs(closest_ratio - original_ratio):
                closest_ratio = ratio
                valid_resolutions = [(w, h)]
            elif abs(ratio - original_ratio) == abs(closest_ratio - original_ratio):
                valid_resolutions.append((w, h))
        
        # Find best resolution matching target area
        best_resolution = None
        min_diff = float('inf')
        for w, h in valid_resolutions:
            area = w * h
            diff = abs(area - target_area)
            if diff < min_diff:
                min_diff = diff
                best_resolution = (w, h)
        
        if not best_resolution:
            # Fallback to original scaling if no FLUX resolution matches
            scaling_factor = math.sqrt(target_area / (original_width * original_height))
            new_width = max(1, round(original_width * scaling_factor))
            new_height = max(1, round(original_height * scaling_factor))
            best_resolution = (new_width, new_height)
        
        self.set_output('width', best_resolution[0])
        self.set_output('height', best_resolution[1])


class ImageCaptioningNode:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
        # Register model folder in ComfyUI
        self.model_dir = os.path.join(folder_paths.models_dir, "minicpm")
        os.makedirs(self.model_dir, exist_ok=True)
        folder_paths.folder_names_and_paths["minicpm"] = ([self.model_dir], folder_paths.supported_pt_extensions)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "What do you see in this image? Describe the image accurately."}),
                "unload_after_generation": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_caption"
    CATEGORY = "Trithemius/MiniCPM-o-2_6"

    def load_model(self):
        if self.model is None:
            print("Loading model...")
            model_path = "2dameneko/MiniCPM-o-2_6-nf4"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Free VRAM before loading model
            mm.soft_empty_cache()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                local_files_only=False,
                cache_dir=self.model_dir
            )
            
            # Use the appropriate dtype based on hardware capability
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            self.model = AutoModel.from_pretrained(
                model_path,
                attn_implementation='eager',
                torch_dtype=dtype,
                trust_remote_code=True,
                # Only initialize what's needed
                init_vision=True,
                init_audio=False,  # Not needed for captioning
                init_tts=False,    # Not needed for captioning
                device_map=device,
                local_files_only=False,
                cache_dir=self.model_dir
            )
            
            self.model.eval()
            print("Model loaded successfully.")
        else:
            print("Model already loaded.")

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                self.tokenizer = None
            torch.cuda.empty_cache()
            gc.collect()
            print("Model unloaded to free memory.")

    def generate_caption(self, image, prompt, unload_after_generation=True):
        # Unload other models to free VRAM
        mm.unload_all_models()
        self.load_model()
        
        try:
            # Debug: Check input image type
            print(f"Received image type: {type(image)}")
            
            if isinstance(image, torch.Tensor):
                # Convert to numpy safely
                image = image.cpu().numpy()
            
            if isinstance(image, np.ndarray):
                print(f"Original image shape: {image.shape}")
                
                # Ensure image is in (H, W, C) format
                if image.ndim == 4:
                    image = image.squeeze(0)  # Remove batch dimension if exists
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)
                
                if image.dtype != np.uint8:
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)
                
                image = Image.fromarray(image).convert('RGB')
            elif isinstance(image, Image.Image):
                print("Image is already a PIL Image.")
            else:
                raise ValueError(f"Unsupported image format: {type(image)}")
            
            # Use the provided prompt directly without combining with system prompt
            print(f"Using prompt: {prompt}")
            
            # Prepare the message with only the custom prompt
            msgs = [{'role': 'user', 'content': [image, prompt]}]
            
            # Use inference mode to reduce memory usage
            with torch.inference_mode():
                # Generate caption
                caption = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer
                )
            
            print(f"Generated Caption: {caption}")
            return (caption,)
        
        finally:
            # Unload model if requested to free memory
            if unload_after_generation:
                self.unload_model()
                mm.soft_empty_cache()

NODE_CLASS_MAPPINGS = {
    "ImageCaptioning": ImageCaptioningNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCaptioning": "Image Captioning (MiniCPM-o-2_6)"
}
