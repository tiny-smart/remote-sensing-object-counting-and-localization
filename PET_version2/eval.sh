CUDA_VISIBLE_DEVICES='1' \
python eval.py \
    --dataset_file="CARPK" \

    --resume="./outputs/CARPK/pet_model/best_checkpoint.pth" \
    --vis_dir="VIS" \
    --gauss=1 \
    --clahe=1 \
    --saltpepper=1 \
    --fourier=1 \
    --order=1234