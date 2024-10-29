import json
from collections import defaultdict
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
class COCOParser:
    def __init__(self, anns_file, imgs_dir, input_size=(224, 224)):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
        
        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        self.names_dict = {}
        self.imgs_dir = imgs_dir
        self.input_size = input_size

        for ann in coco['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']] = ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
            self.names_dict[img['id']] = img['file_name']
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat

    def get_imgIds(self):
        return list(self.im_dict.keys())
    
    def get_imgName(self, key):
        return self.names_dict.get(key)
    
    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        ann_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]        

    def load_cats(self, class_ids):
        class_ids = class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def load_image_and_annotations(self, image_id):
        img_path = f"{self.imgs_dir}/{self.get_imgName(image_id)}"
        image = load_img(img_path, target_size=self.input_size)
        img_array = img_to_array(image) / 255.0
        
        ann_ids = self.get_annIds(image_id)
        annotations = self.load_anns(ann_ids)
        
        bboxes = []
        class_labels = []
        for ann in annotations:
            bbox = ann['bbox']
            class_id = ann['category_id']
            
            img_info = self.im_dict[image_id]
            img_w, img_h = img_info['width'], img_info['height']
            x, y, w, h = bbox
            bbox_normalized = [x / img_w, y / img_h, w / img_w, h / img_h]
            bboxes.append(bbox_normalized)
            class_labels.append(class_id)

        return img_array, np.array(bboxes), np.array(class_labels)

coco_parser = COCOParser(anns_file='instances_default.json', imgs_dir='TrainingImages/')

