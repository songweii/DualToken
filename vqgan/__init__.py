# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .scheduler import create_scheduler
from .optimizer import create_resnet_optimizer

from .discriminator import NLayerDiscriminator, weights_init
from .gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from .lpips import LPIPS

import torch


def create_vqgan_loss():

    disc_loss_type = "hinge"
    # disc_loss_type = "vanilla"
    if disc_loss_type == "hinge":
        disc_loss = hinge_d_loss
    elif disc_loss_type == "vanilla":
        disc_loss = vanilla_d_loss
    else:
        raise ValueError(f"Unknown GAN loss '{disc_loss_type}'.")

    gen_loss_type = "vanilla"
    if gen_loss_type == 'vanilla':
        gen_loss = vanilla_g_loss
    else:
        raise ValueError(f"Unknown GAN loss '{gen_loss_type}'.")

    perceptual_loss = LPIPS()

    return disc_loss, gen_loss, perceptual_loss


def create_discriminator_with_optimizer_scheduler(steps_per_epoch, max_epoch, lr=7.2e-5, distenv=None):
    model = NLayerDiscriminator(input_nc=3,
                                n_layers=3,
                                use_actnorm=False,
                                ndf=64,
                                ).apply(weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0.0, betas=[0.5, 0.9]
    )
    scheduler = create_scheduler(optimizer,
                                 steps_per_epoch=steps_per_epoch,
                                 max_epoch=max_epoch,
                                 distenv=distenv)

    return model, optimizer, scheduler