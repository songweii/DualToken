from transformers import PreTrainedModel, AutoConfig, AutoModel

from .configuration_rqvaesigliptransformer import RQVAESIGLIPTransformerConfig
from .rqvaesiglip import RQVAESiglipModel


class RQVAESIGLIPTransformer(PreTrainedModel):
    config_class = RQVAESIGLIPTransformerConfig
    def __init__(self, config: RQVAESIGLIPTransformerConfig):
        super().__init__(config)

        rqvaesiglip_config = RQVAESiglipModel.config_class.from_dict(config.rqvaesiglip)
        self.rqvaesiglip = RQVAESiglipModel._from_config(rqvaesiglip_config)


AutoConfig.register("rqvaesigliptransformer_model", RQVAESIGLIPTransformerConfig)
AutoModel.register(RQVAESIGLIPTransformerConfig, RQVAESIGLIPTransformer)