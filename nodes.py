import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import folder_paths
import comfy.model_management as mm
import gc

class ImageCaptioningNode:
    """
    ComfyUI node for image captioning using MiniCPM-o-2_6 model.
    """
    
    # Class variables to share model across instances
    _model = None
    _tokenizer = None
    _device = None
    _dtype = None
    
    def __init__(self):
        # Register model folder in ComfyUI
        self.model_dir = os.path.join(folder_paths.models_dir, "minicpm")
        os.makedirs(self.model_dir, exist_ok=True)
        
        if "minicpm" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["minicpm"] = ([self.model_dir], folder_paths.supported_pt_extensions)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "What do you see in this image? Describe the image accurately.", 
                                     "multiline": True}),
                "unload_after_generation": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_caption"
    CATEGORY = "Trithemius/MiniCPM-o-2_6"
    
    @classmethod
    def load_model(cls):
        """Load the model if not already loaded, using class variables for sharing."""
        if cls._model is None:
            print("Loading MiniCPM-o-2_6 model...")
            model_path = "2dameneko/MiniCPM-o-2_6-nf4"
            
            # Determine device and optimize memory
            cls._device = mm.get_torch_device()
            mm.soft_empty_cache()
            
            # Use the appropriate dtype based on hardware capability
            cls._dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            # Load tokenizer
            cls._tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                local_files_only=False
            )
            
            # Load model with optimized settings
            cls._model = AutoModel.from_pretrained(
                model_path,
                attn_implementation='eager',
                torch_dtype=cls._dtype,
                trust_remote_code=True,
                # Only initialize what's needed
                init_vision=True,
                init_audio=False,
                init_tts=False,
                device_map=cls._device,
                local_files_only=False
            )
            
            cls._model.eval()
            print("MiniCPM-o-2_6 model loaded successfully.")
        else:
            print("MiniCPM-o-2_6 model already loaded.")

    @classmethod
    def unload_model(cls):
        """Unload the model to free memory."""
        if cls._model is not None:
            print("Unloading MiniCPM-o-2_6 model to free memory.")
            del cls._model
            cls._model = None
            
            if cls._tokenizer is not None:
                del cls._tokenizer
                cls._tokenizer = None
            
            torch.cuda.empty_cache()
            gc.collect()
            mm.soft_empty_cache()

    def prepare_image(self, image):
        """Convert ComfyUI image to PIL format for the model."""
        if isinstance(image, torch.Tensor):
            # Handle tensor image (ComfyUI standard format)
            if image.ndim == 4:  # Batch of images
                image = image[0]  # Take first image from batch
            
            # Convert from BCHW/CHW format to HWC
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)
            
            # Convert to numpy and then to PIL
            image_np = image.cpu().numpy()
            
            # Convert to uint8 if needed
            if image_np.dtype != np.uint8:
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            
            return Image.fromarray(image_np).convert('RGB')
        
        elif isinstance(image, np.ndarray):
            # Handle numpy array
            if image.ndim == 4:
                image = image[0]  # Take first image from batch
            
            # Ensure correct channel order
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            return Image.fromarray(image).convert('RGB')
        
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")

    def generate_caption(self, image, prompt, unload_after_generation=True):
        """Generate caption for the provided image."""
        try:
            # Temporarily unload other models to free VRAM
            mm.unload_all_models()
            self.load_model()
            
            # Prepare the image
            pil_image = self.prepare_image(image)
            
            # Prepare the message
            msgs = [{'role': 'user', 'content': [pil_image, prompt]}]
            
            # Generate caption with memory optimization
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=self._dtype):
                caption = self._model.chat(
                    image=None,  # Image is already in msgs
                    msgs=msgs,
                    tokenizer=self._tokenizer
                )
            
            return (caption,)
        
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return (f"Error: {str(e)}",)
        
        finally:
            # Unload model if requested to free memory
            if unload_after_generation:
                self.unload_model()

NODE_CLASS_MAPPINGS = {
    "ImageCaptioning": ImageCaptioningNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCaptioning": "Image Captioning (MiniCPM-o-2_6)"
}
