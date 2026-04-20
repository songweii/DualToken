export OMP_NUM_THREADS=8
torchrun --nproc_per_node 8 -m main \
    --sem_weight 1 \
    --stage 1 \
    --name siglip-384-rvq8 \
    --model "model_config_siglip_384_rvq8_32d" \
    --save-frequency 1 \
    --train-data="$YOUR_DATA_PATH/cc12/cc12m-train-{0000..2175}.tar" \
    --train-num-samples 10000000 \
    --dataset-type "webdataset" \
    --warmup=10000 \
    --batch-size=32 \
    --lr=7.2e-5 \
    --beta1=0.5 \
    --beta2=0.9 \
    --wd=0.0001 \
    --epochs=20 \
    --gan_start_epoch=0 \
    --restart_gan=20 \
    --workers=1
