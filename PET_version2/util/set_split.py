# 根据给定的train.txt 和 test.txt提取图片
import os
import shutil

def select_and_copy_images(txt_file, source_folder, destination_folder_img,destination_folder_ann):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder_img):
        os.makedirs(destination_folder_img)
    if not os.path.exists(destination_folder_ann):
        os.makedirs(destination_folder_ann)
    # 读取txt文件中的图片名
    with open(txt_file, 'r') as file:
        image_names = file.read().splitlines()

    # 从源文件夹中选择并复制图片到目标文件夹
    for image_name in image_names:
        source_path = os.path.join(os.path.join(source_folder,'Images'), image_name+'.png')
        print(source_path)
        destination_path = os.path.join(destination_folder_img, image_name+'.png')
        if os.path.exists(source_path):
            shutil.copyfile(source_path, destination_path)
            print("已复制图片: "+image_name)
        else:
            print("找不到图片: "+image_name)

    # 从源文件夹中选择并复制标注文件到目标文件夹
    for image_name in image_names:
        source_path = os.path.join(os.path.join(source_folder,'Point_Annotations'), image_name+'.xml')
        print(source_path)
        destination_path = os.path.join(destination_folder_ann, image_name+'.xml')
        if os.path.exists(source_path):
            shutil.copyfile(source_path, destination_path)
            print("已复制标注文件: "+image_name)
        else:
            print("找不到标注文件: "+image_name)


file=['train','test']
for i in file:#遍历训练集和测试集
    txt_file = os.path.join(r'/data/yjliang/code/SAC/PET/data/CARPK/ImageSets',i+'.txt') # 训练集=测试集txt的存放地址
    source_folder = os.path.join(r'/data/yjliang/code/SAC/PET/data/CARPK') # 源路径
    destination_folder_img = os.path.join(r'/data/yjliang/code/SAC/PET/data/CARPK', i+'_data','images') # 目的路径-图片
    destination_folder_ann = os.path.join(r'/data/yjliang/code/SAC/PET/data/CARPK', i+'_data','ground_truth')# 目的路径-标注文件
    select_and_copy_images(txt_file, source_folder, destination_folder_img,destination_folder_ann)
