from collections import defaultdict
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

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
        self.input_size = input_size  # Input size for the CNN

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
        """Load and preprocess an image and its annotations."""
        img_path = f"{self.imgs_dir}/{self.get_imgName(image_id)}"
        image = load_img(img_path, target_size=self.input_size)
        img_array = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
        
        ann_ids = self.get_annIds(image_id)
        annotations = self.load_anns(ann_ids)
        
        bboxes = []
        class_labels = []
        for ann in annotations:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            class_id = ann['category_id']
            
            # Normalize the bounding box
            img_info = self.im_dict[image_id]
            img_w, img_h = img_info['width'], img_info['height']
            x, y, w, h = bbox
            bbox_normalized = [x / img_w, y / img_h, w / img_w, h / img_h]  # Normalize to [0, 1]
            bboxes.append(bbox_normalized)
            class_labels.append(class_id)

        return img_array, np.array(bboxes), np.array(class_labels)

# Dataset Generator Function
def dataset_generator(coco_parser):
    img_ids = coco_parser.get_imgIds()
    for img_id in img_ids:
        img, bboxes, labels = coco_parser.load_image_and_annotations(img_id)
        
        # Pad bounding boxes and labels
        max_bbox_count = 50  # Choose a suitable number based on your dataset
        padded_bboxes = np.pad(bboxes, ((0, max_bbox_count - len(bboxes)), (0, 0)), mode='constant')
        padded_labels = np.pad(labels, (0, max_bbox_count - len(labels)), mode='constant', constant_values=-1)
        
        # Create masks
        masks = np.zeros((max_bbox_count,), dtype=np.float32)
        masks[:len(labels)] = 1.0  # Mark valid entries as 1

        yield img, (padded_bboxes, padded_labels, masks)

# Create a TensorFlow dataset from the COCOParser
def create_tf_dataset(coco_parser, batch_size=8):
    output_signature = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # Image shape
        (
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),  # Bounding boxes
            tf.TensorSpec(shape=(None,), dtype=tf.int32),      # Labels
            tf.TensorSpec(shape=(None,), dtype=tf.float32)     # Masks
        )
    )
    
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(coco_parser),
        output_signature=output_signature
    )
    
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            (224, 224, 3),  # Image shape
            (tf.TensorShape([None, 4]), tf.TensorShape([None]), tf.TensorShape([None]))  # Padded bounding boxes, labels, and masks
        )
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

def custom_bbox_loss(y_true, y_pred, mask):
    mask = tf.cast(mask, tf.float32)
    y_true = tf.multiply(y_true, tf.expand_dims(mask, axis=-1))
    y_pred = tf.multiply(y_pred, tf.expand_dims(mask, axis=-1))
    loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return tf.reduce_mean(loss)

# CNN Model Definition
from tensorflow.keras import layers, models
def build_cnn_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))

    # CNN feature extractor
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    
    # Output for bounding boxes
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(x)  # Sigmoid for normalized output

    # Output for binary class labels
    class_output = layers.Dense(1, activation='sigmoid', name='class')(x)  # Sigmoid for binary classification

    model = models.Model(inputs=inputs, outputs=[bbox_output, class_output])
    return model

# Loss function including mask
@tf.function
def custom_loss(y_true, y_pred):
    bboxes_true, labels_true, mask = y_true
    bboxes_pred, labels_pred = y_pred
    
    # Convert tensors to float32
    mask = tf.cast(mask, tf.float32)
    
    # Calculate the bounding box loss
    bbox_loss = tf.reduce_sum(tf.square(bboxes_true - bboxes_pred), axis=-1)
    bbox_loss = tf.reduce_mean(bbox_loss * mask)
    
    # Calculate binary classification loss
    class_loss = tf.keras.losses.binary_crossentropy(labels_true, labels_pred)
    class_loss = tf.reduce_mean(class_loss * mask)
    
    return bbox_loss + class_loss

# Example usage
# Initialize COCOParser
coco_parser = COCOParser(anns_file='instances_default.json', imgs_dir='TrainingImages/')

# Create training and validation datasets
dataset = create_tf_dataset(coco_parser, batch_size=8)
# Split dataset into training and validation sets if necessary
dataset_size = len(coco_parser.get_imgIds())
split_index = int(0.8 * dataset_size)
train_dataset = dataset.take(split_index)
val_dataset = dataset.skip(split_index)

# Build and compile model
model = build_cnn_model(num_classes=1)
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
