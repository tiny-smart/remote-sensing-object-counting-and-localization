# Remote-Sensing-Target-Localization


## Highlights


We express the localization problem as a decomposable point querying process, concurrently addressing the counting issue. In this process, sparse input points can bifurcate into four new points when necessary. Our project is designed for aerial scenes in remote sensing, with the current focus on locating objects such as ships, crowds, and aircraft.

  

## Installation

- Our project has no extra compiled components, minimal package dependencies, making the code straightforward and easy to use. Instructions for installing dependencies via conda are provided. First, clone the repository locally:
  
```
git clone https://github.com/babadaiwo/Remote-Sensing-Target-Localization.git
```
- Required packages:
  
```
torch
torchvision
numpy
opencv-python
scipy
matplotlib
```

- Then, install packages as:

```
pip install -r requirements.txt
```


## Data Preparation

- Download crowd-counting datasets, e.g., [ShanghaiTech](https://github.com/desenzhou/ShanghaiTechDataset).
  
- We expect the directory structure to be as follows:
  

```
PET
├── data
│    ├── ShanghaiTech
├── datasets
├── models
├── ...
```

- Download crowd-counting datasets, e.g., [ShanghaiTech](https://github.com/desenzhou/ShanghaiTechDataset).
  
- We expect the directory structure to be as follows:
  

```
PET
├── data
│    ├── CARPK
├── datasets
├── models
├── ...
```


- Download CARPK datasets, e.g., [CARPK](https://github.com/babadaiwo/CARPK.git).
- I have transformed the CARPK dataset into a universal VOC dataset format. For unannotated images, you can use [LabelImg](https://pan.baidu.com/s/1hB-WxbBhhRDVYOBs7h961w) (ps:c8lc) to standardize the annotations for the images. For annotated images, you can refer to 'Remote-Sensing-Target-Localization/util/convert_bbox_to_points.py' to convert existing data into the VOC dataset format.
  
- We expect the directory structure to be as follows:

- Alternatively, you can define the path of the dataset in [datasets/__init__.py](datasets/__init__.py)


## Training

- Download ImageNet pretrained [vgg16_bn](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth), and put it in ```pretrained``` folder. Or you can define your pre-trained model path in [models/backbones/vgg.py](models/backbones/vgg.py)
  

- To train our model on ShanghaiTech PartA, run
  
  ```
  sh train.sh
  ```

  - To train our model on CARPK, run
  
  ```
  sh train.sh
  or
CUDA_VISIBLE_DEVICES='0' \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=10001 \
    --use_env main.py \
    --lr=0.0001 \
    --backbone="vgg16_bn" \
    --ce_loss_coef=1.0 \
    --point_loss_coef=5.0 \
    --eos_coef=0.5 \
    --dec_layers=2 \
    --hidden_dim=256 \
    --dim_feedforward=512 \
    --nheads=8 \
    --dropout=0.0 \
    --epochs=1500 \
    --dataset_file="CARPK" \
    --eval_freq=5 \
    --output_dir='pet_model'
  ```
  

## Evaluation

- Modify [eval.sh](eval.sh)
  - change ```--resume``` to your local model path
- Run

```
sh eval.sh
```




## Permission

This code is for academic purposes only. Contact: Yujia Liang (yjlianghust.edu.cn)


## Acknowledgement

We thank the authors of [PET](https://github.com/cxliu0/PET) for open-sourcing their work.


