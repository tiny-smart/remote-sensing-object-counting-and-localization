import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .CARPK import build as build_car
from .CARPK import build as build_Car

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'CARPK': '/home/rscount/lhx/Remote-Sensing-Target-Localization/data/CARPK/',
    'Car': '/root/autodl-tmp/Car_train_test_total'
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    if args.dataset_file == 'CARPK':
        return build_car(image_set, args)
    if args.dataset_file == 'Car':
        return build_Car(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')




