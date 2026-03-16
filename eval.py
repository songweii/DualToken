import glob
import logging
import os
import sys
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
import pyiqa

try:
    import wandb
except ImportError:
    wandb = None

from PIL import Image
from datasets import Dataset
import matplotlib.pyplot as plt

from src.open_clip.factory import create_model_and_transforms
from src.open_clip import get_input_dtype
from src.open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from src.open_clip_train.data import get_data
from src.open_clip_train.logger import setup_logging
from src.open_clip_train.train import evaluate
from src.open_clip_train.params import parse_args
from tqdm import tqdm


def compute_kl_divergence(event_counts):
    # 归一化得到 P 分布
    P = event_counts / event_counts.sum()
    # 均匀分布 Q
    K = len(event_counts)
    Q = torch.full_like(P, 1 / K)
    # 计算 KL 散度，避免 log(0) 和 division by zero
    mask = P > 0  # 只对 P > 0 的项计算
    kl_divergence = (P[mask] * (P[mask] / Q[mask]).log()).sum()
    return kl_divergence


def save_images(images, path):
    images = images.to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)
    # os.makedirs(path, exist_ok=True)
    for i in range(images.shape[0]):
        image = images[i].permute(1, 2, 0)
        image = image.detach().to(torch.float32).cpu().numpy().astype(np.uint8)
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


def evaluate_recon(model, original_state, data, device, input_dtype, args):
    dataloader = data["imagenet-val"].dataloader
    # dataloader = data['train'].dataloader

    save_path = f"./{args.name}/recon_results"
    os.makedirs(save_path, exist_ok=True)
    save_path_gt = f"./{args.name}/gt_results"
    os.makedirs(save_path_gt, exist_ok=True)

    from vqgan.lpips import LPIPS
    from pytorch_msssim import ssim

    perceptual_loss = LPIPS().to(device).eval()
    cnt, maxnum = 0, 500
    # psnr_computer = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
    psnr_computer = pyiqa.create_metric('psnr', device=device)
    psnr_total = 0.0
    sem_loss, sem_cos = 0, 0
    L1_loss, L2_loss, pcpt_loss, ssim_sum = 0, 0, 0, 0
    codes = []

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader)):
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            # texts = texts.to(device=device, non_blocking=True)
            batch_size = images.shape[0]

            with torch.no_grad():
                clip_loss_dict, zq_semantic, hidden_states, pooler_output, img_recon, code_pixel, code_semantic, quant_loss_pixel, quant_loss_semantic = model(images, None)

                outputs_gt = original_state(pixel_values=images, output_hidden_states=True)
                z_semantic_gt = outputs_gt.hidden_states[-2]
            
            # 1*27*27 --> 1*729
            codes.append(code_pixel.view(code_pixel.size(0), -1).cpu())

            L2 = nn.MSELoss()(img_recon, images)
            L1 = torch.mean(torch.abs(images.contiguous() - img_recon.contiguous()))
            p_loss = torch.mean(perceptual_loss(images.contiguous(), img_recon.contiguous()))

            semantic_loss = nn.MSELoss()(zq_semantic, z_semantic_gt)
            semantic_loss_cos = -F.cosine_similarity(zq_semantic, z_semantic_gt, dim=-1).mean()
            # semantic_loss = nn.MSELoss()(hidden_states[-2], z_semantic_gt)
            # semantic_loss_cos = -F.cosine_similarity(hidden_states[-2], z_semantic_gt, dim=-1).mean()

            L2_loss += L2.item() * batch_size
            L1_loss += L1.item() * batch_size
            pcpt_loss += p_loss.item() * batch_size

            sem_loss += semantic_loss.item() * batch_size
            sem_cos += semantic_loss_cos.item() * batch_size

            img_recon[img_recon > 1] = 1
            img_recon[img_recon < -1] = -1

            save_x = (images + 1) / 2
            save_xrec = (img_recon + 1) / 2

            ssim_val = ssim(save_xrec, save_x, data_range=1, size_average=True)
            ssim_sum += ssim_val.item() * batch_size
            
            psnr_score = psnr_computer(save_x, save_xrec)
            psnr_total += torch.sum(psnr_score)

            cnt += batch_size

            if is_master(args) and (i % 50) == 0:
                logging.info(
                    f"L2: {L2_loss / cnt:.6f}\t"
                    f"L1: {L1_loss / cnt:.6f}\t"
                    f"LPIPS: {pcpt_loss / cnt:.6f}\t"
                    f"SSIM: {ssim_sum / cnt:.6f}\t"
                    f"PSNR: {psnr_total / cnt:.6f}\t"
                    f"sem_loss: {sem_loss / cnt:.6f}\t"
                    f"sem_cos: {sem_cos / cnt:.6f}\t")
                    
            image_save_path = f'{save_path}/{i}.jpg'
            save_images(img_recon, image_save_path)
            image_save_path_gt = f'{save_path_gt}/{i}_gt.jpg'
            save_images(images, image_save_path_gt)

    if is_master(args):
        logging.info(
            f"Average L2: {L2_loss / cnt:.6f}\t"
            f"Average L1: {L1_loss / cnt:.6f}\t"
            f"Average LPIPS: {pcpt_loss / cnt:.6f}\t"
            f"Average SSIM: {ssim_sum / cnt:.6f}\t"
            f"Average PSNR: {psnr_total / cnt:.6f}\t"
            f"sem_loss: {sem_loss / cnt:.6f}\t"
            f"sem_cos: {sem_cos / cnt:.6f}\t")

    #### FID Score
    print ("Calculating FID Score...")
    from cleanfid import fid
    # fid_value_clip = fid.compute_fid(save_path_gt, save_path, model_name="clip_vit_b_32", mode="clean")
    # fid_value_clip = fid.compute_fid(save_path_gt, save_path, model_name="clip_vit_l_14", mode="clean")
    fid_value = fid.compute_fid(save_path_gt, save_path, mode="clean")
    
    if is_master(args):
        logging.info(
            # f"rFID-clip: {fid_value_clip:.6f}\t"
            f"rFID: {fid_value:.6f}\t")
    
    # 1*729 --> 500*729
    codes = torch.cat(codes, dim=0).to(torch.int32)
    entropy = []
    K = 16384
    codebook_usage = torch.zeros(K).to(torch.int32)
    # ideal_usage = torch.ones(K).to(torch.int32)

    for i in range(codes.shape[0]):
        codes_ = codes[i, :]
        # print(codes_.shape)
        codebook_usage.index_add_(0, codes_, torch.ones_like(codes_).to(torch.int32))

        counts = torch.bincount(codes_)
        counts = (counts / counts.sum()).clamp(1e-10)
        entropy.append(-(counts * counts.log2()).sum().item())

    used_code_num = torch.count_nonzero(codebook_usage)
    print(codebook_usage)
    logging.info(f"Number of embeddings used: {used_code_num}")

    kl_divergence = compute_kl_divergence(codebook_usage)
    logging.info(f"KL Divergence: {kl_divergence.item()}")

    P = (codebook_usage / codebook_usage.sum())
    mask = P != 0
    print(P)

    E = -(P[mask] * P[mask].log2()).sum().item()
    pct = E / np.log2(K)
    logging.info(f"Entropy percentage: {pct * 100}%")

    import matplotlib.pyplot as plt
    idxs = torch.arange(len(codebook_usage))
    plt.bar(idxs.numpy(), P.numpy(), color='blue', alpha=0.7)
    plt.savefig("bar.png", dpi=100)
    plt.show()

    pct = sum(entropy) / (np.log2(K) * len(entropy))
    entropy_avg = sum(entropy) / len(entropy)
    print(f"Average entropy of codes: {entropy_avg}")
    print(f"Effective percentage: {pct * 100}%")


def main(args):
    args = parse_args(args)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = init_distributed_device(args)

    # input_dtype = torch.bfloat16
    input_dtype = None

    # Setup text logger
    log_base_path = os.path.join(args.logdir, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        
    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)
    # args.save_logs = args.logdir and args.logdir.lower() != 'none' and is_master(args)
    args.save_logs = False
    args.wandb = False
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")

    vision_tower_path = args.model
    # model, tokenizer, config, preprocess_train, preprocess_val = create_model_and_transforms(config_path=vision_tower_path, precision="bf16", device=device)
    model, tokenizer, config, preprocess_train, preprocess_val = create_model_and_transforms(config_path=vision_tower_path, device=device)
    print(f"logit_scale: {model.siglip_model.logit_scale}")
    
    from transformers import SiglipVisionModel
    original_state = SiglipVisionModel.from_pretrained(config.rqvaesiglip["pretrained_model"])
    original_state = original_state.to(device=device)
    for param in original_state.parameters():
        param.requires_grad_(False)

    weights_path = args.weights_path
    load_pretrained(model, weights_path)
    logging.info(f"loaded from {weights_path}")
    
    print(f"logit_scale (loaded): {model.siglip_model.logit_scale}")
    print(f"model.dtype: {model.dtype}")

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    model.eval()
    # model.train()

    data = get_data(
        args=args,
        preprocess_fns=(preprocess_train, preprocess_val),
        epoch=0,
        tokenizer=tokenizer,
    )

    writer = None
    if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        evaluate(model, data, 0, args, tb_writer=writer, tokenizer=tokenizer)
    
    evaluate_recon(model, original_state, data, device, input_dtype, args)


if __name__ == "__main__":
    main(sys.argv[1:])
