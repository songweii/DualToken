export OMP_NUM_THREADS=8
torchrun --nproc_per_node 1 -m main \
    --sem_weight 1 \
    --stage 0 \
    --name debug1 \
    --model "model_config_siglip_256" \
    --save-frequency 1 \
    --train-data="/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/cc12/cc12m-train-{0000..2175}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/ImageNet-1K-web/train_web/train-{00000..00256}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/Imagenet21K-web/{00000..07769}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/laion-aesthetics-12m-web/{00000..02419}.tar" \
    --train-num-samples 30000000 \
    --dataset-type "webdataset" \
    --warmup=100000 \
    --batch-size=32 \
    --lr=1e-5 \
    --beta1=0.9 \
    --beta2=0.98 \
    --eps=1e-8 \
    --wd=1e-4 \
    --epochs=10 \
    --gan_start_epoch=0 \
    --restart_gan=20 \
    --workers=1

# torchrun --nproc_per_node 1 -m main \
#     --sem_weight 1 \
#     --stage 1 \
#     --name debug1 \
#     --model "model_config_siglip_256" \
#     --save-frequency 1 \
#     --train-data="/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/cc12/cc12m-train-{0000..2175}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/ImageNet-1K-web/train_web/train-{00000..00256}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/Imagenet21K-web/{00000..07769}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/laion-aesthetics-12m-web/{00000..02419}.tar" \
#     --train-num-samples 30000000 \
#     --dataset-type "webdataset" \
#     --warmup=100000 \
#     --batch-size=16 \
#     --lr=5e-4 \
#     --beta1=0.5 \
#     --beta2=0.9 \
#     --wd=0.0001 \
#     --epochs=20 \
#     --gan_start_epoch=0 \
#     --restart_gan=20 \
#     --workers=1

# torchrun --nproc_per_node 8 -m main \
#     --sem_weight 1 \
#     --stage 2 \
#     --load_from /inspire/hdd/project/deepgen/songwei-240108120100/DeepGen/dualtoken-tokenizer/DualToken-siglip2-so400m-16-256/logs/bs64x16-lr1e-5/checkpoints/epoch_15.pt \
#     --name debug2 \
#     --model "model_config_siglip_256" \
#     --save-frequency 1 \
#     --train-data="/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/cc12/cc12m-train-{0000..2175}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/ImageNet-1K-web/train_web/train-{00000..00256}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/Imagenet21K-web/{00000..07769}.tar::/inspire/hdd/project/deepgen/songwei-240108120100/data_tokenizer/laion-aesthetics-12m-web/{00000..02419}.tar" \
#     --train-num-samples 30000000 \
#     --dataset-type "webdataset" \
#     --warmup=100000 \
#     --batch-size=128 \
#     --lr=5e-4 \
#     --beta1=0.5 \
#     --beta2=0.9 \
#     --wd=0.0001 \
#     --epochs=20 \
#     --gan_start_epoch=0 \
#     --restart_gan=20 \
#     --workers=1
