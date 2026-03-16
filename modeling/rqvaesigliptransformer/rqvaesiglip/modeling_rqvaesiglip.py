import re
import logging
import torch

from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoProcessor

from .configuration_rqvaesiglip import RQVAESiglipConfig
from .modules import Decoder, PostQuantResnetBlock, ProjectResnetBlock
from .quantizations import RQBottleneck
from .quantizations_ema import RQBottleneck_ema
# from .siglip import SiglipModel
from transformers import SiglipModel

from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from functools import partial
from torch.nn.functional import scaled_dot_product_attention
from timm.models.layers import get_norm_layer


class AttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        act_layer = None,
        drop = 0.0,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer('layernorm'), eps=1e-6)
        self.norm = norm_layer(in_features)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = AttentionBlock(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def build_projector(dim_in, dim_out, projector_type='mlp2x_gelu'):
    if projector_type == 'linear':
        linear = nn.Linear(dim_in, dim_out)
        linear.reset_parameters()
        return linear
    elif projector_type == 'nonlinear':
        linear = nn.Linear(dim_in, dim_out)
        linear.reset_parameters()
        modules = [linear, nn.GELU()]
        return nn.Sequential(*modules)
    elif projector_type == 'conv':
        return nn.Conv2d(dim_in, dim_out, 1)
    else:  # mlp2x_gelu
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), 
            nn.GELU(), 
            nn.Linear(dim_out, dim_out)
        )


class RQVAESiglipModel(PreTrainedModel):
    config_class = RQVAESiglipConfig
    def __init__(self, config: RQVAESiglipConfig):
        super().__init__(config)

        # siglip_config = SiglipModel.config_class.from_pretrained(config.pretrained_model)
        # self.siglip_model = SiglipModel._from_config(siglip_config)
        self.siglip_model = SiglipModel.from_pretrained(config.pretrained_model)

        self.layer_total = len(self.siglip_model.vision_model.encoder.layers)

        self.layer_recon = config.layer_recon
        self.layer_sem = config.layer_sem

        assert self.layer_sem >= self.layer_recon, \
            f"layer_recon ({self.layer_recon}) is greater than layer_sem ({self.layer_sem})"

        vq_checkpoint = None
        vq_config_semantic = config.vq_semantic
        vq_config_pixel = config.vq_pixel

        # self.prequant_semantic = AttnProjection(
        #     in_dim=config.hidden_size, 
        #     out_dim=vq_config_semantic["embed_dim"], 
        #     num_heads=config.hidden_size // vq_config_semantic["embed_dim"]
        # )
        # self.prequant_semantic = build_projector(
        #     dim_in=config.hidden_size, 
        #     dim_out=vq_config_semantic["embed_dim"], 
        #     projector_type='mlp2x_gelu'
        # )
        self.prequant_pixel = AttnProjection(
            in_dim=config.hidden_size, 
            out_dim=vq_config_pixel["embed_dim"], 
            num_heads=config.hidden_size // vq_config_pixel["embed_dim"]
        )
        # self.layer_norm_pixel = nn.LayerNorm(vq_config_pixel["embed_dim"])
        
        if vq_config_semantic["bottleneck_type"] == "rq_ema":
            logging.info("quantizer_semantic: rq_ema")
            self.quantizer_semantic = RQBottleneck_ema(
                latent_shape=vq_config_semantic["latent_shape"],
                code_shape=vq_config_semantic["code_shape"],
                n_embed=vq_config_semantic["n_embed"],
                decay=vq_config_semantic["decay"],
                shared_codebook=vq_config_semantic["shared_codebook"],
                restart_unused_codes=vq_config_semantic["restart_unused_codes"],
                checkpoint=vq_checkpoint
            )
        elif vq_config_semantic["bottleneck_type"] == "rq":
            logging.info("quantizer_semantic: rq")
            self.quantizer_semantic = RQBottleneck(
                latent_shape=vq_config_semantic["latent_shape"],
                code_shape=vq_config_semantic["code_shape"],
                n_embed=vq_config_semantic["n_embed"],
                decay=vq_config_semantic["decay"],
                shared_codebook=vq_config_semantic["shared_codebook"],
                restart_unused_codes=vq_config_semantic["restart_unused_codes"],
                checkpoint=vq_checkpoint
            )

        if vq_config_pixel["bottleneck_type"] == "rq_ema":
            logging.info("quantizer_pixel: rq_ema")
            self.quantizer_pixel = RQBottleneck_ema(
                latent_shape=vq_config_pixel["latent_shape"],
                code_shape=vq_config_pixel["code_shape"],
                n_embed=vq_config_pixel["n_embed"],
                decay=vq_config_pixel["decay"],
                shared_codebook=vq_config_pixel["shared_codebook"],
                restart_unused_codes=vq_config_pixel["restart_unused_codes"],
                checkpoint=vq_checkpoint
            )
        elif vq_config_pixel["bottleneck_type"] == "rq":
            logging.info("quantizer_pixel: rq")
            self.quantizer_pixel = RQBottleneck(
                latent_shape=vq_config_pixel["latent_shape"],
                code_shape=vq_config_pixel["code_shape"],
                n_embed=vq_config_pixel["n_embed"],
                decay=vq_config_pixel["decay"],
                shared_codebook=vq_config_pixel["shared_codebook"],
                restart_unused_codes=vq_config_pixel["restart_unused_codes"],
                checkpoint=vq_checkpoint
            )

        self.postquant_semantic = AttnProjection(
            in_dim=vq_config_semantic["embed_dim"], 
            out_dim=config.hidden_size, 
            num_heads=config.hidden_size // vq_config_semantic["embed_dim"]
        )
        self.postquant_pixel = AttnProjection(
            in_dim=vq_config_pixel["embed_dim"], 
            out_dim=config.hidden_size, 
            num_heads=config.hidden_size // vq_config_pixel["embed_dim"]
        )

        self.post_quant_conv = PostQuantResnetBlock(
            in_channels=config.hidden_size,
            out_channels=config.ddconfig["decoder_in_channels"],
            dropout=0.0
        )

        self.decoder = Decoder(**config.ddconfig)
        try:
            self.decoder_latent_shape = config.decoder_latent_shape
        except:
            self.decoder_latent_shape = None
        
        self.shortcut = config.layer_shortcut is not None
        if self.shortcut:
            self.layer_shortcut = config.layer_shortcut
        
            assert self.layer_sem >= self.layer_shortcut, \
                f"layer_shortcut ({self.layer_shortcut}) is greater than layer_sem ({self.layer_sem})"
            
            self.weight_shortcut = config.weight_shortcut

            self.prequant_shortcut = AttnProjection(
                in_dim=config.hidden_size, 
                out_dim=vq_config_pixel["embed_dim"], 
                num_heads=config.hidden_size // vq_config_pixel["embed_dim"]
            )
        

    def encode_text(self, text):
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions, output_hidden_states, return_dict = None, None, None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_model = self.siglip_model.text_model
        text_outputs = text_model(
            input_ids=text,
            attention_mask=None,
            position_ids=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_embeds = text_outputs[1]
        return text_embeds


    def encode_image(self, image):
        vision_model = self.siglip_model.vision_model
        hidden_state = vision_model.embeddings(image)

        hidden_states = [hidden_state]

        attention_mask = None
        output_attentions = None

        hidden_state_pixel = None
        hidden_state_shortcut = None
        hidden_state_semantic = None
        
        for i, encoder_layer in enumerate(vision_model.encoder.layers):
            if vision_model.encoder.gradient_checkpointing and vision_model.encoder.training:
                layer_outputs = vision_model.encoder._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_state,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_state,
                    attention_mask,
                    output_attentions=output_attentions,
                )
            hidden_state = layer_outputs[0]
            hidden_states.append(hidden_state)

            if (i + 1) == self.layer_recon:
                hidden_state_pixel = self.prequant_pixel(hidden_state)
            
            if self.shortcut:
                if (i + 1) == self.layer_shortcut:
                    hidden_state_shortcut = self.prequant_shortcut(hidden_state)

            if (i + 1) == self.layer_sem:
                # pixel
                if self.shortcut:
                    hidden_state_pixel += self.weight_shortcut * hidden_state_shortcut
                B, L, C = hidden_state_pixel.shape
                
                hidden_state_pixel = hidden_state_pixel.reshape(B, int(L**0.5), int(L**0.5), C)
                # hidden_state_pixel = self.layer_norm_pixel(hidden_state_pixel)
                z_q_pixel, quant_loss_pixel, code_pixel = self.quantizer_pixel(hidden_state_pixel)
                z_q_pixel = z_q_pixel.reshape(B, L, -1)

                # semantic
                # hidden_state_semantic = self.prequant_semantic(hidden_state)
                hidden_state_semantic = hidden_state
                B, L, C = hidden_state_semantic.shape

                hidden_state_semantic = hidden_state_semantic.reshape(B, int(L**0.5), int(L**0.5), C)
                z_q_semantic, quant_loss_semantic, code_semantic = self.quantizer_semantic(hidden_state_semantic)
                z_q_semantic = z_q_semantic.reshape(B, L, -1)
                
                hidden_state = self.postquant_semantic(z_q_semantic)

        last_hidden_state = hidden_state
        last_hidden_state_norm = vision_model.post_layernorm(last_hidden_state)
        pooler_output = vision_model.head(last_hidden_state_norm) if vision_model.use_head else None

        return z_q_pixel, quant_loss_pixel, code_pixel, z_q_semantic, quant_loss_semantic, code_semantic, hidden_states, pooler_output

    
    def decode(self, z_q):
        B, L, C = z_q.shape
        z_q = self.postquant_pixel(z_q)
        z_q = z_q.reshape(B, int(L**0.5), int(L**0.5), -1)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        if self.decoder_latent_shape is not None:
            z_q = F.interpolate(z_q, size=tuple(self.decoder_latent_shape), mode='bilinear')
        z_q = self.post_quant_conv(z_q)
        out = self.decoder(z_q)
        return out
    

    @torch.no_grad()
    def get_code_emb_with_depth(self, code):
        return self.quantizer_pixel.embed_code_with_depth(code)
    

    def forward(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None):

        # return z_q_pixel, quant_loss_pixel, code_pixel, z_q_semantic, quant_loss_semantic, code_semantic, hidden_states, pooler_output
        vision_output = self.encode_image(image) if image is not None else None
        zq_pixel, quant_loss_pixel, code_pixel = vision_output[0], vision_output[1], vision_output[2]
        zq_semantic, quant_loss_semantic, code_semantic = vision_output[3], vision_output[4], vision_output[5]
        hidden_states, pooler_output = vision_output[-2], vision_output[-1]

        # normalized features
        image_embeds = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
        if text is not None:
            text_embeds = self.encode_text(text)  # tokenized tokens
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        else:
            text_embeds = None

        images_recon = self.decode(z_q=zq_pixel)

        clip_loss_dict = {
            "image_features": image_embeds,
            "text_features": text_embeds,
            "logit_scale": self.siglip_model.logit_scale
        }
        if self.siglip_model.logit_bias is not None:
            clip_loss_dict['logit_bias'] = self.siglip_model.logit_bias

        return clip_loss_dict, zq_semantic, hidden_states, pooler_output, images_recon, code_pixel, code_semantic, quant_loss_pixel, quant_loss_semantic
    

AutoConfig.register("rqvaesiglip_model", RQVAESiglipConfig)
AutoModel.register(RQVAESiglipConfig, RQVAESiglipModel)