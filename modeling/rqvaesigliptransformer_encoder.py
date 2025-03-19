import torch
import torch.nn as nn
import os
import os.path as osp
from collections import OrderedDict

from transformers import CLIPImageProcessor, AutoProcessor, PreTrainedModel, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor

from .rqvaesigliptransformer import RQVAESIGLIPTransformerConfig, RQVAESIGLIPTransformer
# from src.open_clip.factory import *
import src.open_clip as openclip
from PIL import Image
from io import BytesIO


class RQVAESIGLIPTransformerVisionTower(nn.Module):
    def __init__(self, model_name_or_path, device, input_dtype):
        super().__init__()
        # model_dtype = "torch.bfloat16"
        self.model_dtype = input_dtype
        self.device = device

        self.config = RQVAESIGLIPTransformerConfig.from_pretrained(model_name_or_path)
        self.vision_tower = RQVAESIGLIPTransformer._from_config(self.config, torch_dtype=self.model_dtype).to(self.device)
        self.is_loaded = True

        encoder_path = self.config.rqvaesiglip["pretrained_model"]
        print(f"Successfully loaded pretrained encoder's weight from: {encoder_path}")

        self.processor = AutoProcessor.from_pretrained(encoder_path)
        self.image_processor = None

        if "siglip-large-patch16-256" in encoder_path:
            encoder_name = 'ViT-L-16-SigLIP-256'
        elif "siglip-so400m-patch14-384" in encoder_path:
            encoder_name = 'ViT-SO400M-14-SigLIP-384'
        elif "siglip-large-patch16-384" in encoder_path:
            encoder_name = 'ViT-L-16-SigLIP-384'
        else:
            raise NotImplementedError()
        
        self.tokenizer = openclip.factory.get_tokenizer(model_name=encoder_name, 
                                                        local=encoder_path, 
                                                        cache_dir=None)

        if "siglip-so400m-patch14-384" in encoder_path:  # SigLIP-SO400M-patch14-384
            self.image_processor = CLIPImageProcessor(
                size={"height": 384, "width": 384}, 
                crop_size={"height": 384, "width": 384}, 
                image_mean=[0.5, 0.5, 0.5], 
                image_std=[0.5, 0.5, 0.5]
            )
            self.image_tokens = 729
            # self.config.hidden_size == 1152
        elif "siglip-large-patch16-384" in encoder_path:  # SigLIP-Large-patch16-384
            self.image_processor = CLIPImageProcessor(
                size={"height": 384, "width": 384}, 
                crop_size={"height": 384, "width": 384}, 
                image_mean=[0.5, 0.5, 0.5], 
                image_std=[0.5, 0.5, 0.5]
            )
            self.image_tokens = 576
            # self.config.hidden_size == 1024
        elif "siglip-large-patch16-256" in encoder_path:  # SigLIP-Large-patch16-256
            self.image_processor = CLIPImageProcessor(
                size={"height": 256, "width": 256}, 
                crop_size={"height": 256, "width": 256}, 
                image_mean=[0.5, 0.5, 0.5], 
                image_std=[0.5, 0.5, 0.5]
            )
            self.image_tokens = 256
            # self.config.hidden_size == 1024
        else:
            raise NotImplementedError()
    
    
    def get_clip_loss(self, image_files, texts):
        image_file = image_files[0]
        if isinstance(image_file, str):
            image = Image.open(image_file).convert("RGB")
        elif isinstance(image_file, BytesIO):
            image = Image.open(image_file).convert("RGB")
        else:
            image = image_file
        # image = Image.open(requests.get(url, stream=True).raw)
        # image = process_image(image_files, self.image_processor).to(self.device, dtype=self.model_dtype)
        processor = self.processor
        inputs = processor(text=texts, images=image, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device, dtype=self.model_dtype)
        
        with torch.no_grad():
            clip_loss = self.vision_tower.rqvaesiglip.calc_clip_loss(**inputs)

        return clip_loss

    
    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            state_dict = self.state_dict()

        if self.get_vision_tower() and "radio" not in self.get_vision_tower().__class__.__name__.lower():
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vision_tower.config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            self.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.image_processor.save_pretrained(os.path.join(output_dir, "vision_tower"))