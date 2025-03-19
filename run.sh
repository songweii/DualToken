export OMP_NUM_THREADS=8
torchrun --nproc_per_node 2 -m main \
    --sem_weight 1 \
    --stage 1 \
    --name baseline \
    --model "model_config" \
    --save-frequency 1 \
    --train-data="$YOUR_DATA_PATH/cc12/cc12m-train-{0000..0255}.tar" \
    --train-num-samples 1290496 \
    --dataset-type "webdataset" \
    --warmup=10000 \
    --batch-size=16 \
    --lr=7.2e-5 \
    --beta1=0.5 \
    --beta2=0.9 \
    --wd=0.0001 \
    --epochs=20 \
    --gan_start_epoch=0 \
    --restart_gan=20 \
    --workers=1 \
