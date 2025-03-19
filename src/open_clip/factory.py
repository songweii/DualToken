import json
import logging
import os
import re
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .convert import convert_state_dict
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
from .coca_model import CoCa
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from .transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from .tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH

from modeling.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    """ Fetch model config from builtin (local library) configs.
    """
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def _get_hf_config(
        model_id: str,
        cache_dir: Optional[str] = None,
):
    """ Fetch model config from HuggingFace Hub.
    """
    config_path = download_pretrained_from_hf(
        model_id,
        filename='open_clip_config.json',
        cache_dir=cache_dir,
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_tokenizer(
        model_name: str = '',
        local: str = None,
        context_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
):
    if model_name.startswith(HF_HUB_PREFIX):
        model_name = model_name[len(HF_HUB_PREFIX):]
        try:
            config = _get_hf_config(model_name, cache_dir=cache_dir)['model_cfg']
        except Exception:
            tokenizer = HFTokenizer(
                model_name,
                context_length=context_length or DEFAULT_CONTEXT_LENGTH,
                cache_dir=cache_dir,
                **kwargs,
            )
            return tokenizer
    else:
        config = get_model_config(model_name)
        assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get('text_cfg', {})
    if 'tokenizer_kwargs' in text_config:
        tokenizer_kwargs = dict(text_config['tokenizer_kwargs'], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    if 'hf_tokenizer_name' in text_config and local:
        tokenizer = HFTokenizer(
            tokenizer_name=local,
            context_length=context_length,
            cache_dir=cache_dir,
            **tokenizer_kwargs,
        )
    elif 'hf_tokenizer_name' in text_config:
        tokenizer = HFTokenizer(
            text_config['hf_tokenizer_name'],
            context_length=context_length,
            cache_dir=cache_dir,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer


def load_state_dict(
        checkpoint_path: str,
        device='cpu',
        weights_only=True,
):
    # Check if safetensors or not and load weights accordingly
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(
        model: Union[CLIP, CustomTextCLIP],
        checkpoint_path: str,
        strict: bool = True,
        weights_only: bool = True,
        device='cpu',
):
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        # Separate path loading numpy big_vision (SigLIP) weights
        from open_clip.convert import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path, device=device, weights_only=weights_only)

    # Detect & convert 3rd party state_dicts -> open_clip
    state_dict = convert_state_dict(model, state_dict)

    # Detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if 'logit_bias' not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])

    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]

    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)

    # Finally, load the massaged state_dict into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        config_path: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu'
):
    if isinstance(device, str):
        device = torch.device(device)

    # loading rqsiglip (the full model class)
    dtype = None
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16

    rqvaesiglip_path = config_path
    model_tokenizer_processor = RQVAESIGLIPTransformerVisionTower(rqvaesiglip_path,
                                                                  device=device, 
                                                                  input_dtype=dtype)
    model = model_tokenizer_processor.vision_tower.rqvaesiglip
    tokenizer = model_tokenizer_processor.tokenizer
    config = model_tokenizer_processor.config
    image_processor = model_tokenizer_processor.image_processor
    
    return model, tokenizer, config, image_processor


def create_loss(args):
    if "coca" in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.siglip:
        assert not args.horovod, "Horovod not currently supported for SigLip"
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
        )
    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def create_model_and_transforms(
        config_path: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    model, tokenizer, config, image_processor = create_model(
        config_path,
        precision=precision,
        device=device
    )
    pp_cfg = PreprocessCfg(
        size=image_processor.size["height"], mode='RGB',
        mean=tuple(image_processor.image_mean),
        std=tuple(image_processor.image_std),
        interpolation='bicubic',  # CLIP's default, see CLIPImageProcessor and BaseImageProcessor.resize
        resize_mode='shortest',   # CLIP's default, see CLIPImageProcessor and BaseImageProcessor.resize
        fill_color=0,             # CLIP's default, see CLIPImageProcessor and BaseImageProcessor.center_crop
    )

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, tokenizer, config, preprocess_train, preprocess_val
