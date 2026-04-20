import torch
import torch.nn as nn
import os
import os.path as osp
from collections import OrderedDict

from transformers import AutoImageProcessor, AutoProcessor, PreTrainedModel, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor

from .rqvaesigliptransformer import RQVAESIGLIPTransformerConfig, RQVAESIGLIPTransformer
import src.open_clip as openclip
from PIL import Image
from io import BytesIO


class RQVAESIGLIPTransformerVisionTower(nn.Module):
    def __init__(self, model_name_or_path, device, input_dtype):
        super().__init__()
        self.model_dtype = input_dtype
        self.device = device

        self.config = RQVAESIGLIPTransformerConfig.from_pretrained(model_name_or_path)
        self.vision_tower = RQVAESIGLIPTransformer._from_config(self.config, torch_dtype=self.model_dtype).to(self.device)
        self.is_loaded = True

        encoder_path = self.config.rqvaesiglip["pretrained_model"]
        print(f"Successfully loaded pretrained encoder's weight from: {encoder_path}")

        if "siglip2-so400m-patch16-256" in encoder_path:
            encoder_name = 'ViT-SO400M-16-SigLIP2-256'
            self.image_tokens = 256
            # self.config.hidden_size == 1152
        elif "siglip-large-patch16-256" in encoder_path:
            encoder_name = 'ViT-L-16-SigLIP-256'
            self.image_tokens = 256
            # self.config.hidden_size == 1024
        
        elif "siglip2-so400m-patch14-384" in encoder_path:
            encoder_name = 'ViT-SO400M-14-SigLIP2-384'
            self.image_tokens = 729
            # self.config.hidden_size == 1152
        elif "siglip-so400m-patch14-384" in encoder_path:
            encoder_name = 'ViT-SO400M-14-SigLIP-384'
            self.image_tokens = 729
            # self.config.hidden_size == 1152
        
        elif "siglip2-so400m-patch16-384" in encoder_path:
            encoder_name = 'ViT-SO400M-16-SigLIP2-384'
            self.image_tokens = 576
            # self.config.hidden_size == 1152
        elif "siglip-large-patch16-384" in encoder_path:
            encoder_name = 'ViT-L-16-SigLIP-384'
            self.image_tokens = 576
            # self.config.hidden_size == 1024
        else:
            raise NotImplementedError()

        self.processor = AutoProcessor.from_pretrained(encoder_path)
        self.image_processor = AutoImageProcessor.from_pretrained(encoder_path)
        self.tokenizer = openclip.factory.get_tokenizer(model_name=encoder_name, 
                                                        model_path=encoder_path, 
                                                        cache_dir=None)
