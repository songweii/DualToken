export OMP_NUM_THREADS=8
torchrun --nproc_per_node 1 -m main \
    --sem_weight 1 \
    --stage 1 \
    --name siglip2-256-rvq44-16k-nolast-postquant-pixrq-semrqema-zloss \
    --model "model_config_siglip_256_pixrq_semrqema" \
    --save-frequency 1 \
    --train-data="/inspire/hdd/global_user/songwei-240108120100/data_tokenizer/cc12/cc12m-train-{0000..2175}.tar" \
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