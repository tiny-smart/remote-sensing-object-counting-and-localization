#encoding=utf-8
import json
import os
import xml.etree.ElementTree as ET

#新建xml文件
import numpy as np


def buildNewsXmlFile(source_folder):
    for file in os.listdir(source_folder):
        #设置一个新节点，并设置其标签为root
        root = ET.Element("annotation")

        #在root下新建n子节点,设置其名称分别为..

        #这些如果有的话可以填上...
        folder = ET.SubElement(root, "folder")

        filename= ET.SubElement(root, "filename")

        path = ET.SubElement(root, "path")

        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")


        size = ET.SubElement(root, "size")
        #如果有的话把长宽写进来
        width = ET.SubElement(size, "width")
        height = ET.SubElement(size, "height")
        depth = ET.SubElement(size, "depth")

        segmented= ET.SubElement(root, "segmented",)
        segmented.text=str(0)
        gt_path = source_folder + file
        data = open(gt_path, "r")
        text = json.load(data)
        points = np.array(text['2d_point'])
        boxx = np.array(text['bbox'])
        for i in  range (len(points)):
               object= ET.SubElement(root, "object")
               name =ET.SubElement(object, "name") #类别名 car,people, plane
               pose= ET.SubElement(object, "pose")
               truncated = ET.SubElement(object, "truncated")
               difficult = ET.SubElement(object, "difficult")

               bndbox = ET.SubElement(object, "bndbox")
               #有的话填上
               xmin=ET.SubElement(bndbox, "xmin")
               ymin=ET.SubElement(bndbox, "ymin")
               xmax = ET.SubElement(bndbox, "xmax")
               ymax = ET.SubElement(bndbox, "ymax")
               xmin.text=str(boxx[i][0])
               ymin.text=str(boxx[i][1])
               xmax.text=str(boxx[i][2])
               ymax.text=str(boxx[i][3])
               point_2d = ET.SubElement(object, "point_2d")
               #必要的！！！
               center_x=ET.SubElement(point_2d, "center_x")
               center_y=ET.SubElement(point_2d, "center_y")


               center_y.text=str(points[i][0])
               center_x.text=str(points[i][1])
        #将节点数信息保存在ElementTree中，并且保存为XML格式文件
        tree = ET.ElementTree(root)
        tree.write('/data/yjliang/code/SAC/PET/data/CARPK/test_data/VGG_anotation_truth/'+file)

buildNewsXmlFile('/data/yjliang/code/SAC/PET/data/CARPK/test_data/ground_truth/')