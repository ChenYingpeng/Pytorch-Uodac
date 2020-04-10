# -*- coding: utf-8 -*-
####################
#####   Author:Chen
#####   Date:2020-02-29
#######################

import sys
import os
import cv2
import json
import argparse
import xml.dom.minidom
import xml.etree.ElementTree as ET

from tqdm import tqdm

def get_segmentation(points):

    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]

def main():
    parser = argparse.ArgumentParser(description='Convert xml to json for object detection.')
    parser.add_argument('--xml-dir',type=str, default='../underwater/optics/data/train/box/', help='xml file dir path') 
    parser.add_argument('--json',type=str, default='../underwater/optics/data/train/train_data_annotations.json', help='json file save dir path') 
    args = parser.parse_args()
    print(args)

    #### xml 2 json
    xml_dir_path = args.xml_dir
    img_dir_path = args.xml_dir.replace('box','image')
    save_json_name = args.json

    annotations_info = {'images': [], 'annotations': [], 'categories': []}
    categories_map = {'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}

    for key in categories_map:
        categoriy_info = {"id":categories_map[key], "name":key}
        annotations_info['categories'].append(categoriy_info)
    
    ann_id = 1
    for i, xml_name in tqdm(enumerate(sorted(os.listdir(xml_dir_path)))):
        #print(i,xml_name)
        if "xml" not in xml_name:
            continue
        image_name = xml_name.split('.')[0]+ '.jpg'
        xml_path = xml_dir_path + xml_name
        image_path = img_dir_path + image_name

        if os.path.exists(xml_path) and os.path.exists(image_path):
            #read image and get img_h,img_w
            img = cv2.imread(image_path)
            img_h,img_w,img_c = img.shape

            image_info = {'file_name': image_name, 'height': img_h, 'width': img_w,'id': i + 1}
            annotations_info['images'].append(image_info)
           
            DOMTree = xml.dom.minidom.parse(xml_path)
            collection = DOMTree.documentElement

            names = collection.getElementsByTagName('name')
            names = [name.firstChild.data for name in names]

            xmins = collection.getElementsByTagName('xmin')
            xmins = [xmin.firstChild.data for xmin in xmins]
            ymins = collection.getElementsByTagName('ymin')
            ymins = [ymin.firstChild.data for ymin in ymins]
            xmaxs = collection.getElementsByTagName('xmax')
            xmaxs = [xmax.firstChild.data for xmax in xmaxs]
            ymaxs = collection.getElementsByTagName('ymax')
            ymaxs = [ymax.firstChild.data for ymax in ymaxs]

            object_num = len(names)

            for j in range(object_num):
                if names[j] in categories_map:
                    image_id = i + 1
                    xmin,ymin,xmax,ymax = int(xmins[j]),int(ymins[j]),int(xmaxs[j]),int(ymaxs[j])
                    xmin,ymin,xmax,ymax = xmin - 1,ymin - 1,xmax - 1,ymax - 1

                    if xmin >= xmax or ymin >= ymax:
                        print('This image {} include invalid bbox.'.format(image_path))
                        continue

                    if xmax >= img_w:
                        xmax = img_w - 1
                    if ymax >= img_h:
                        ymax = img_h - 1

                    x,y = xmin,ymin
                    w,h = xmax - xmin + 1,ymax - ymin + 1
                    category_id = categories_map[names[j]]
                    area = w * h
                    segmentation = get_segmentation([x, y, w, h])
                    annotation_info = {
                        "id": ann_id, 
                        "image_id":image_id, 
                        "bbox":[x, y, w, h], 
                        "category_id": category_id, 
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0}
                    annotations_info['annotations'].append(annotation_info)
                    ann_id += 1

    with  open(save_json_name, 'w')  as f:
        json.dump(annotations_info, f, indent=4)

    print('---Done annotations file---')
    print('all images is',  len(annotations_info['images']))
    print('all annotations is ',  len(annotations_info['annotations']))
    print('all categories is ',  len(annotations_info['categories']))

if __name__ == '__main__':
    main()

