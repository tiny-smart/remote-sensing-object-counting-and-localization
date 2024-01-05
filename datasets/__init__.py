import torch.utils.data
import torchvision

from .SHA import build as build_sha


from .CARPK import build as build_car
data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'CARPK': '/data/yjliang/code/SAC/PET/data/CARPK/',
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    if args.dataset_file == 'CARPK':
        return build_car(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


