export OMP_NUM_THREADS=8

torchrun --nproc_per_node 1 -m eval \
    --precision "fp32" \
    --name eval-siglip2-256-pixrq-semrqema-zloss-7 \
    --model "model_config_siglip_256_pixrq_semrqema" \
    --weights_path "/inspire/hdd/project/deepgen/songwei-240108120100/DualToken/epoch_7.pt" \
    --imagenet-val "/inspire/hdd/global_user/songwei-240108120100/data_tokenizer/imagenet-1k/val" \
    --batch-size=200 \
    --workers=1 \