import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

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
                "prompt": ("STRING", {"default": "What do you see in this image?"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_caption"
    CATEGORY = "Trithemius/MiniCPM-o-2_6"

    def load_model(self):
        if self.model is None:
            model_path = "2dameneko/MiniCPM-o-2_6-nf4"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Download and load using ComfyUI's model directory
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

    def generate_caption(self, image, prompt):
        # Load model if not already loaded
        self.load_model()
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8)).convert('RGB')
        
        # Prepare the message
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        
        # Generate caption
        caption = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        
        return (caption,)

NODE_CLASS_MAPPINGS = {
    "ImageCaptioning": ImageCaptioningNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCaptioning": "Image Captioning (MiniCPM-o-2_6)"
}