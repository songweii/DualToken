from transformers import PretrainedConfig


class RQVAESiglipConfig(PretrainedConfig):
    model_type = "rqvaesiglip_model"
    def __init__(
        self,
        hidden_size=None,
        layer_recon=None,
        layer_sem=None,
        layer_shortcut=None,
        weight_shortcut=None,
        vq_sem=None,
        vq_pix=None,
        ddconfig=None,
        decay=0.99,
        architectures=None,
        decoder_latent_shape=None,
        pretrained_model="google/siglip-large-patch16-256",
        **kwargs,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.layer_recon = layer_recon
        self.layer_sem = layer_sem
        self.layer_shortcut = layer_shortcut
        self.weight_shortcut = weight_shortcut

        self.vq_sem = vq_sem
        self.vq_pix = vq_pix
        self.ddconfig = ddconfig
        self.decay = decay
        self.architectures = architectures
        self.decoder_latent_shape = decoder_latent_shape
        self.pretrained_model = pretrained_model
