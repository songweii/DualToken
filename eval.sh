export OMP_NUM_THREADS=8

torchrun --nproc_per_node 1 -m eval \
    --precision "fp32" \
    --name eval-siglip2-384-rvq8-32d-9 \
    --model "model_config_siglip_384_rvq8_32d" \
    --weights_path "/mnt/public/users/songwei/code/DualToken-causal-downdim/logs/siglip2-384-rvq8-32d-stage1/checkpoints/epoch_9.pt" \
    --imagenet-val "/mnt/public/users/songwei/data_zoo/data_tokenizer/imagenet-1k/val" \
    --batch-size=200 \
    --workers=1 \