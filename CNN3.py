import tensorflow as tf
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import layers, models
import numpy as np
import json
from collections import defaultdict
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array

# Ensure eager execution is enabled (default in TensorFlow 2.x)
tf.config.run_functions_eagerly(True)
TF_ENABLE_ONEDNN_OPTS=0
# COCOParser Class to load COCO-like dataset
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

# Dataset Generator Function
def dataset_generator(coco_parser):
    img_ids = coco_parser.get_imgIds()
    for img_id in img_ids:
        img, bboxes, labels = coco_parser.load_image_and_annotations(img_id)
        
        max_bbox_count = 50
        padded_bboxes = np.pad(bboxes, ((0, max_bbox_count - len(bboxes)), (0, 0)), mode='constant')
        padded_labels = np.pad(labels, (0, max_bbox_count - len(labels)), mode='constant', constant_values=-1)
        
        masks = np.zeros((max_bbox_count,), dtype=np.float32)
        masks[:len(labels)] = 1.0

        yield img, (padded_bboxes, padded_labels, masks)

# Create a TensorFlow dataset from the COCOParser
def create_tf_dataset(coco_parser, batch_size=8):
    output_signature = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(coco_parser),
        output_signature=output_signature
    )
    
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            (224, 224, 3),
            (tf.TensorShape([None, 4]), tf.TensorShape([None]), tf.TensorShape([None]))
        )
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

# Build a CNN Model
def build_cnn_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(x)
    class_output = layers.Dense(1, activation='sigmoid', name='class')(x)

    model = models.Model(inputs=inputs, outputs=[bbox_output, class_output])
    return model

# Compile and train the model
coco_parser = COCOParser(anns_file='instances_default.json', imgs_dir='TrainingImages/')

# Create dataset
dataset = create_tf_dataset(coco_parser, batch_size=8)
print(dataset)
dataset_size = len(coco_parser.get_imgIds())
split_index = int(0.8 * dataset_size)
train_dataset = dataset.take(split_index)
val_dataset = dataset.skip(split_index)

# Build and compile model
model = build_cnn_model(num_classes=1)
model.compile(optimizer='adam', loss='binary_crossentropy')
# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
