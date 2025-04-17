import os
import sys
import cv2

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
from src.open_clip.factory import create_model_and_transforms
import requests


def save_images(images, path):
    images = images.to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)
    # os.makedirs(path, exist_ok=True)
    for i in range(images.shape[0]):
        image = images[i].permute(1, 2, 0)
        # image = image.cpu().numpy().astype(np.uint8)
        image = image.detach().to(torch.float).cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path_new = path.split(".j")[0] + "_" + str(i) + ".jpg"
        cv2.imwrite(path_new, image)


def load_pretrained(model, path, ignore_keys=list()):
    sd = torch.load(path, map_location="cpu")["state_dict"]
    keys = list(sd.keys())
    for k in keys:
        print(k)
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    try:
        model.load_state_dict(sd, strict=True)
        print("Successfully loaded pretrained weights!")
    except RuntimeError as e:
        print("Error loading weights.")
        print(e)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dtype = torch.bfloat16

    vision_tower_path = "model_config"
    model, tokenizer, config, preprocess_train, preprocess_val = create_model_and_transforms(config_path=vision_tower_path, precision="bf16", device=device)
    # model.siglip_model.logit_scale.data = torch.tensor(4.7215, dtype=torch.float32)
    
    weights_path = "/your/path/to/checkpoints/epoch_n.pt"
    load_pretrained(model, weights_path)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image  = preprocess_val(image)
    image = image.to(device=device, dtype=input_dtype, non_blocking=True)
    image = image.unsqueeze(0)

    with torch.no_grad():
        clip_loss_dict, hidden_state_26, pooler_output, img_recon, code_visual, code_semantic, quant_loss_visual, quant_loss_semantic = model(image, None)
    
    image_save_path = f'cat.jpg'
    save_images(image, image_save_path)
    image_save_path = f'cat_recon.jpg'
    save_images(img_recon, image_save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
