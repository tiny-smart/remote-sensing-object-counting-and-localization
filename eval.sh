CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --backbone="vgg16_bn"\
    --dataset_file="Car" \
    --data_path="/root/autodl-tmp/Car_train_test_total" \
    --pretrained="../store/pretrained/vgg16_bn-6c64b313.pth" \
    --resume="outputs/Car/pet_model_Car/best_checkpoint.pth" \
    --vis_dir="vis_dir"

