import spaces
import os
import json
import time
import torch
from PIL import Image
from tqdm import tqdm
import gradio as gr

from safetensors.torch import save_file
from .pipeline import FluxPipeline
from .transformer_flux import FluxTransformer2DModel
from .lora_helper import set_single_lora, set_multi_lora, unset_lora, update_model_with_lora_v2

class ImageProcessor:
    def __init__(self, base_path, lora_path, base_spatial_image):
        device = "cuda"
        self.pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
        transformer = FluxTransformer2DModel.from_pretrained(base_path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device)
        self.pipe.transformer = transformer
        self.pipe.to(device)
        self.lora_path = lora_path
        self.process_image("Ghibli.safetensors", subject_imgs=[base_spatial_image])
        
    def clear_cache(self, transformer):
        for name, attn_processor in transformer.attn_processors.items():
            attn_processor.bank_kv.clear()
            
    @spaces.GPU()
    def process_image(self, lora_name, prompt='', subject_imgs=[], spatial_imgs=[], height=512, width=512, output_path=None, seed=42, zero_steps=1):
        update_model_with_lora_v2(self.lora_path, lora_name, [1], self.pipe.transformer, 512)
        image = self.pipe(
            prompt,
            height=int(height),
            width=int(width),
            guidance_scale=3.5,
            num_inference_steps=25,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed), 
            subject_images=subject_imgs,
            spatial_images=spatial_imgs,
            cond_size=512,
            use_zero_init=True,
            zero_steps=int(zero_steps)
        ).images[0]
        self.clear_cache(self.pipe.transformer)
        if output_path:
            image.save(output_path)
        return image
