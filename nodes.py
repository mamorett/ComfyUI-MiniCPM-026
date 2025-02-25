import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import folder_paths
import comfy.model_management as mm

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
                "prompt": ("STRING", {"default": "What do you see in this image? Describe the image accurately."})
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
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                local_files_only=False,
                cache_dir=self.model_dir
            )
            
            self.model = AutoModel.from_pretrained(
                model_path,
                attn_implementation='eager',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                init_vision=True,
                init_audio=True,
                init_tts=True,
                device_map=device,
                local_files_only=False,
                cache_dir=self.model_dir
            )
            
            self.model.eval()
            self.model.init_tts()
            self.model.tts.float()
            print("Model loaded successfully.")
        else:
            print("Model already loaded.")

    def generate_caption(self, image, prompt):
        self.load_model()
        
        # Debug: Check input image type
        print(f"Received image type: {type(image)}")
        
        if isinstance(image, torch.Tensor):
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
        
        # Generate caption
        caption = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        
        print(f"Generated Caption: {caption}")
        return (caption,)


NODE_CLASS_MAPPINGS = {
    "ImageCaptioning": ImageCaptioningNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCaptioning": "Image Captioning (MiniCPM-o-2_6)"
}
