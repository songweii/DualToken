import os
import os.path as osp
import numpy as np
import cv2

from huggingface_hub import snapshot_download, repo_exists
from huggingface_hub.utils import HFValidationError
from transformers import  PretrainedConfig

import torch
from PIL import Image
from io import BytesIO
from torchvision.transforms import CenterCrop


def save_images(images, path):
    images = images.to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)
    # os.makedirs(path, exist_ok=True)
    for i in range(images.shape[0]):
        image = images[i].permute(1, 2, 0)
        # image = image.cpu().numpy().astype(np.uint8)
        image = image.detach().to(torch.float).cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)


def process_images(image_files, image_processor):
    new_images = [process_image(image_file, image_processor, None) for image_file in image_files]

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def process_image(image_file, image_processor, generation_mode=False, image_aspect_ratio="resize", crop_size={'height': 256, 'width': 256}):
    processor = image_processor
    
    if isinstance(image_file, str):
        image = Image.open(image_file).convert("RGB")
    elif isinstance(image_file, BytesIO):
        image = Image.open(image_file).convert("RGB")
    else:
        image = image_file

    if generation_mode:
        if image.size[0] < image.size[1]:
            image = image.crop((0, 0, min(image.size), min(image.size)))
        else:
            ccrop = CenterCrop(min(image.size))
            image = ccrop(image)
    elif image_aspect_ratio == "resize":
        crop_size = crop_size
        image = image.resize((crop_size["height"], crop_size["width"]))
    elif image_aspect_ratio == "pad":
        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
    else:
        raise NotImplementedError()
        
    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image


def expand2square(pil_img, background_color):
    """
    Expand the given PIL image to a square shape by adding padding.

    Parameters:
    - pil_img: The PIL image to be expanded.
    - background_color: The color of the padding to be added.

    Returns:
    - The expanded PIL image.

    If the image is already square, it is returned as is.
    If the image is wider than it is tall, padding is added to the top and bottom.
    If the image is taller than it is wide, padding is added to the left and right.
    """
    width, height = pil_img.size
    if pil_img.mode == 'L':
        background_color = background_color[0]
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_model_config(config):
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_projector_cfg"]
    
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path 

    if root_path is not None and not osp.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError as e:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)

    return_list = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_list.append(os.path.join(root_path, key[:-4]))
            except:
                raise ValueError(f"Cannot find resume path in config for {key}!")
        elif isinstance(cfg, PretrainedConfig):
            return_list.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_list.append(cfg)
        
    return return_list


def normalize(img):
    return (img - 0.5) * 2


def denormalize(img):
    return (img + 1) / 2
