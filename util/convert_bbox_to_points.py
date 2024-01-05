import os
import shutil
import numpy as np
import json
def convert_bbox_to_points(bbox_file_path):
    # 读取框标注文件
    with open(bbox_file_path, 'r') as file:
        lines = file.readlines()

    points = []
    for line in lines:
        # 解析x1 y1 x2 y2信息并计算框的中心点坐标
        x1, y1, x2, y2,l1 = map(int, line.strip().split())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        points.append((center_y, center_x))

    return points

def main():
    # 原始文件夹路径和目标文件夹路径
    source_folder = '/data/yjliang/code/SAC/PET/data/CARPK/Annotations/'
    destination_folder = '/data/yjliang/code/SAC/PET/data/CARPK/Point_Annotations/'

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历原始文件夹中的所有框标注文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            # 构建完整路径
            bbox_file_path = os.path.join(source_folder, filename)
            image_info={'2d_point':None,'bbox':None,'size':None}
            # 将框标注文件转换为点标注列表
            points = convert_bbox_to_points(bbox_file_path)
            #ndarray1 = np.array(points)
            image_info['2d_point']=points
            # 构建新的点标注文件名
            point_file_name = filename[:-4]+'.xml'

            # 构建新的点标注文件路径
            point_file_path = os.path.join(destination_folder, point_file_name)

            # 将点标注写入新的文件
            with open(point_file_path, 'w') as point_file:
                point_file.write(json.dumps(image_info))


            # 移动原始框标注文件到目标文件夹


if __name__ == "__main__":
    main()