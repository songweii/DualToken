export OMP_NUM_THREADS=8

torchrun --nproc_per_node 1 -m eval \
    --precision "fp32" \
    --name eval-siglip-384-rvq8 \
    --model "model_config_siglip_384_rvq8" \
    --weights_path "$PATH_TO_YOUR_CHECKPOINTS/epoch_xxx.pt" \
    --imagenet-val "$PATH_TO_YOUR_IMAGENET/imagenet-1k/val" \
    --batch-size=200 \
    --workers=1 \