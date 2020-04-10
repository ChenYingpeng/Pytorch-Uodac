import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from mmdet.core import underwater_classes
import argparse

label_ids = {name: i + 1 for i, name in enumerate(underwater_classes())}


def save(images, annotations,save_json_path):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    ann['categories'] = categories
    json.dump(ann, open(save_json_path, 'w'))


def generate_test_json(image_dir,save_json_path):
    im_list = glob(os.path.join(image_dir, '*.jpg'))
    idx = 1
    image_id = 0
    images = []
    annotations = []
    for im_path in tqdm(im_list):
        image_id += 1
        im = Image.open(im_path)
        w, h = im.size
        image = {'file_name': os.path.basename(im_path), 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    save(images, annotations,save_json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate test json file for submitting')
    parser.add_argument('--test-image-dir', help='test image dir', type=str,default='../underwater/optics/data/test-A-image/')
    parser.add_argument('--save-json-path', help='save json path', type=str,default='../underwater/optics/data/annotations/test-A-image.json')
    args = parser.parse_args()
    print(args)
    print("generate test json label file.")
    generate_test_json(args.test_image_dir,args.save_json_path)
