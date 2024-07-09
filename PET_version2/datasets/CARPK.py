import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob
import scipy.io as io
import torchvision.transforms as standard_transforms
import warnings
import json
import xml.etree.ElementTree as ET

warnings.filterwarnings('ignore')


class SHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False,arg=None):
        self.root_path = data_root

        prefix = "train_data" if train else "test_data"
        #prefix = "train_data"
        self.prefix = prefix
        self.img_list = os.listdir(f"{data_root}/{prefix}/images")

        # get image and ground-truth list
        self.gt_list = {}
        for img_name in self.img_list:
            img_path = f"/data/ctf/workfile/CORN_DATA/DATA/{prefix}/images/{img_name}"##
            gt_path = f"/data/ctf/workfile/CORN_DATA/DATA/{prefix}/VGG_anotation_truth/{img_name}"##
            self.gt_list[img_path] = gt_path.replace("png", "xml")
        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = 256
        self.arg=arg

    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # load image and gt points
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        img, points,bboxs = load_data((img_path, gt_path), self.train,self.arg)
        points = points.astype(float)
        bboxs = bboxs.astype(float)
        # image transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        # random scale
        # if self.train:
        #     scale_range = [0.8, 1.2]
        #     min_size = min(img.shape[1:])
        #     scale = random.uniform(*scale_range)
        #
        #     # interpolation
        #     if scale * min_size > self.patch_size:
        #         img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
        #         points *= scale


        if self.train:
            scale_range =[0.3,0.7]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            # interpolation
            if scale * min_size > self.patch_size:
                 img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                 points *= scale
                 bboxs*=scale
        img_ = img.numpy()  # FloatTensor转为ndarray
        img_ = np.transpose(img_, (1, 2, 0))  # 把channel那一维放到最后
        # plt.imshow(img_)
        # plt.axis('on')  # 关掉坐标轴为 off
        # plt.title('image')  # 图像题目
        # plt.show()
        # random crop patch
        if self.train:
            img, points,bboxs = random_crop(img, points,bboxs, patch_size=self.patch_size)
        else:
            img, points,bboxs =resize(img, points,bboxs)
        # img_ = img.numpy()  # FloatTensor转为ndarray
        # img_ = np.transpose(img_, (1, 2, 0))  # 把channel那一维放到最后
        # plt.imshow(img_)
        # plt.axis('on')  # 关掉坐标轴为 offS
        # plt.title('image')  # 图像题目
        # plt.show()
        #random flip
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]

        # target
        target = {}
        target['points'] = torch.Tensor(points)
        target['bboxs'] = torch.Tensor(bboxs)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = img_path

        return img, target
###########################################################################################################


def median_filter(image, ksize=5):
    """
    Apply median filter to the image.

    :param image: Input image
    :param ksize: Kernel size for the median filter (must be odd)
    :return: Image after applying median filter
    """
    return cv2.medianBlur(image, ksize)


def fourier_edge_enhancement(image):
    """
    Apply Fourier Transform to the image for edge enhancement.

    :param image: Input image
    :return: Image with enhanced edges using Fourier Transform
    """
    # Convert image to float32
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create a mask with high-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 100 # Radius for high-pass filter
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # Apply the mask and inverse FFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize to uint8
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_back


def process_image_fourier(image, ksize=3):
    """
    Process a color image with median filter, Fourier edge enhancement, and additional median filtering.

    :param image_path: Path to the input color image
    :param ksize: Kernel size for the median filter (must be odd)
    :return: Processed color image
    """
    # Load image


    if image is None:
        raise FileNotFoundError(f"No image found ")

    # Split the image into its color channels
    channels = cv2.split(image)
    processed_channels = []

    # Process each color channel
    for channel in channels:
        # Apply median filter
        median_filtered_channel = median_filter(channel, 3)

        # Apply Fourier edge enhancement
        edge_enhanced_channel = fourier_edge_enhancement(median_filtered_channel)

        # Combine the original and edge enhanced images
        combined_channel = cv2.addWeighted(channel, 1.3/1.6, edge_enhanced_channel, 0.7/1.6, 0)
        #combined_channel=combined_channel/1.6
        # Collect the processed channel
        processed_channels.append(combined_channel)

    # Merge the processed channels back into a color image
    final_image= cv2.merge(processed_channels)

    # Apply median filter again on the processed image
    #final_image = median_filter(final_image, ksize)

    return final_image

#############################################################################################
        
        
        
        
def apply_CLAHE_to_color_image(image):

    if image is None:
        print("null")
        return
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return image_clahe


def add_salt_and_pepper_noise(image, prob=0.006):
    """
    Add salt and pepper noise to an image.

    Parameters:
    image (numpy.ndarray): Input image in cv2 format.
    prob (float): Probability of the noise.

    Returns:
    numpy.ndarray: Image with salt and pepper noise added.
    """
    if random.random() > 0.5:
        noisy_image = image.copy()
        rand_matrix = np.random.rand(*image.shape[:2])

        # Add salt noise (white pixels)
        noisy_image[rand_matrix < (prob / 20)] = 255

        # Add pepper noise (black pixels)
        noisy_image[(rand_matrix >= (prob / 2)) & (rand_matrix < prob)] = 0
        
    else:
        noisy_image=image

    return noisy_image


def add_gaussian_noise(image, mean=0, stddev=25, prob=0.1):
    """
    Add Gaussian noise to an image.

    Parameters:
    image (numpy.ndarray): Input image in cv2 format.
    mean (float): Mean of the Gaussian noise.
    stddev (float): Standard deviation of the Gaussian noise.
    prob (float): Probability of applying the noise to each pixel.

    Returns:
    numpy.ndarray: Image with Gaussian noise added.
    
    """
    if random.random() > 0.5:
          noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
          mask = np.random.rand(*image.shape[:2]) < prob
          noisy_image = image.copy().astype(np.float32)
      
          # Apply Gaussian noise only to the selected pixels
          noisy_image[mask] = noisy_image[mask] + noise[mask]
          noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    else:
          noisy_image=image

    return noisy_image
##############################################################
def multi_process(img,arg):
    #print(arg)
    a=[]
    
    order=arg.order
    #print(a)
    while(order>0):
        a.append(order % 10)
        order=order//10
    a.reverse()
    #print(a)
    #print(arg.clahe)
    for i in range(len(a)):
        if (arg.clahe == 1 and a[i]==1):
            #print('clahe_init')
            img = apply_CLAHE_to_color_image(img)
            
        if (arg.gauss == 1 and a[i]==2):
            img = add_salt_and_pepper_noise(img)
            #print('saltpeppernoise_init')
        if (arg.saltpepper == 1 and a[i]==3):
            img = add_gaussian_noise(img)
            #print('gaussian_init')
        if (arg.fourier == 1 and a[i]==4):
            img = process_image_fourier(img)
            #print('fourier_init')
    return img
#########################################################################

def load_data(img_gt_path, train, arg):
    img_path, gt_path = img_gt_path
    #############################
    # processing image
    img = cv2.imread(img_path)


    img=multi_process(img,arg)
   ################################
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #################################
    tree = ET.parse(gt_path)
    root = tree.getroot()
    #print(gt_path)
    elements = root.findall('object')
    points=[]
    bboxs=[]
    for element in elements:
      x=int(element.findall('point_2d')[0].find('center_x').text)
      y =int(element.findall('point_2d')[0].find('center_y').text)

      xmin=int(element.findall('bndbox')[0].find('xmin').text)
      ymin =int(element.findall('bndbox')[0].find('ymin').text)
      xmax=int(element.findall('bndbox')[0].find('xmax').text)
      ymax =int(element.findall('bndbox')[0].find('ymax').text)
      points.append([y,x])
      bboxs.append([xmin, ymin,xmax, ymax])
    points = np.array(points)
    bboxs = np.array(bboxs)
    return img, points,bboxs


def random_crop(img, points,bboxs, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size
    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_bboxs=bboxs[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w
    result_bboxs[:, 0] -= start_w
    result_bboxs[:, 1] -= start_h
    result_bboxs[:, 2] -= start_w
    result_bboxs[:, 3] -= start_h

    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    result_bboxs[:, 0] *= fW
    result_bboxs[:, 1] *= fH
    result_bboxs[:, 2] *= fW
    result_bboxs[:, 3] *= fH
    return result_img, result_points,result_bboxs




def resize(img, points,bboxs):
    patch_h = 512
    patch_w = 1024


    # resize to patchsize
    imgH, imgW = img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_bboxs = bboxs.copy()
    result_points = points.copy()
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    result_bboxs[:, 0] *= fW
    result_bboxs[:, 1] *= fH
    result_bboxs[:, 2] *= fW
    result_bboxs[:, 3] *= fH
    return result_img, result_points,result_bboxs


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])

    data_root = args.data_path
    if image_set == 'train':
        train_set = SHA(data_root, train=True, transform=transform, flip=True,arg=args)
        return train_set
    elif image_set == 'val':
        val_set = SHA(data_root, train=False, transform=transform,arg=args)
        return val_set