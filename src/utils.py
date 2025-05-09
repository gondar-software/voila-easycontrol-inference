from PIL import Image, ImageSequence, ImageOps
import torch
from io import BytesIO
import os
import numpy as np
import requests
from typing import Tuple, Union

def pil2tensor(img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert PIL Image to PyTorch tensor(s)
    
    Args:
        img: PIL Image object (can be animated/multi-frame)
    
    Returns:
        Tuple of (image_tensor, mask_tensor)
    """
    output_images = []
    output_masks = []
    
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)

def load_image(image_source: str) -> Tuple[Image.Image, str]:
    """
    Load image from URL or local path
    
    Args:
        image_source: Either URL or local file path
        
    Returns:
        Tuple of (PIL Image, filename)
    """
    try:
        if image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            file_name = image_source.split('/')[-1].split('?')[0]
        else:
            img = Image.open(image_source)
            file_name = os.path.basename(image_source)
        return img, file_name
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_source}: {str(e)}")

def load_image_from_url(url: str) -> torch.Tensor:
    """
    Load image from URL and convert to tensor
    
    Args:
        url: Image URL
        
    Returns:
        Image tensor
    """
    img, _ = load_image(url)
    img_out, _ = pil2tensor(img)
    return img_out

def tensor2pil(image_tensor: torch.Tensor, mask_tensor: torch.Tensor = None) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
    """
    Convert PyTorch tensor(s) back to PIL Image(s)
    
    Args:
        image_tensor: Image tensor in shape (C, H, W) or (B, C, H, W)
        mask_tensor: Optional mask tensor in shape (H, W) or (B, H, W)
        
    Returns:
        PIL Image or tuple of (PIL Image, PIL Mask) if mask is provided
    """
    # Handle batch dimension
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]  # Take first image in batch
    
    # Convert to numpy and scale to 0-255
    image_np = image_tensor.cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    # Convert to PIL Image
    if image_np.shape[0] == 1:  # Grayscale
        image_pil = Image.fromarray(image_np[0], mode='L')
    else:  # RGB
        image_pil = Image.fromarray(np.transpose(image_np, (1, 2, 0)), mode='RGB')
    
    # Process mask if provided
    if mask_tensor is not None:
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor[0]  # Take first mask in batch
            
        mask_np = (1. - mask_tensor).cpu().numpy() * 255  # Invert and scale
        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode='L')
        return image_pil, mask_pil
    
    return image_pil