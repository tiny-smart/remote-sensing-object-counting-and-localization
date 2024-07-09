CUDA_VISIBLE_DEVICES='6' \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=10821 \
    --use_env main.py \
    --lr=0.00045 \
    --backbone="vgg16_bn" \
    --ce_loss_coef=1.0 \
    --point_loss_coef=5.0 \
    --eos_coef=0.5 \
    --dec_layers=2 \
    --hidden_dim=256 \
    --dim_feedforward=512 \
    --nheads=8 \
    --dropout=0.0 \
    --epochs=2000 \
    --dataset_file="CARPK" \
    --eval_freq=5 \
    --output_dir='pet_model'\
    --sch_type=1 \
    --gauss=1 \
    --clahe=1 \
    --saltpepper=1 \
    --fourier=1 \
    --order=1234
    
