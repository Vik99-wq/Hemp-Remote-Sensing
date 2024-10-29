import tensorflow as tf
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os

def load_image(image_id, img_dir):
    img_path = os.path.join(img_dir, coco.imgs[image_id]['file_name'])
    image = Image.open(img_path).convert('RGB')
    return np.array(image)

def convert_polygon_to_bbox(polygon):
    """Convert a polygon to a bounding box."""
    polygon = np.array(polygon).reshape(-1, 2)
    x_min = np.min(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    x_max = np.max(polygon[:, 0])
    y_max = np.max(polygon[:, 1])
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def load_annotations(image_id, coco):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    bboxes = []
    class_ids = []
    for ann in anns:
        if 'segmentation' in ann:
            for polygon in ann['segmentation']:
                bbox = convert_polygon_to_bbox(polygon)
                bboxes.append(bbox)
        else:
            bbox = ann['bbox']
            bboxes.append(bbox)
        class_ids.append(ann['category_id'])
    return bboxes, class_ids

def preprocess_image(image):
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalize to [0,1]
    return image

def create_tf_dataset(coco, img_dir, batch_size=4):
    img_ids = coco.getImgIds()
    
    def generator():
        for img_id in img_ids:
            image = load_image(img_id, img_dir)
            bboxes, class_ids = load_annotations(img_id, coco)
            image = preprocess_image(image)
            yield image, bboxes, class_ids
    
    def pad_annotations(bboxes, class_ids):
        max_bbox_len = max(len(b) for b in bboxes)
        padded_bboxes = [np.pad(b, ((0, max_bbox_len - len(b)), (0, 0)), mode='constant') for b in bboxes]
        padded_bboxes = np.array(padded_bboxes, dtype=np.float32)
        padded_class_ids = [np.pad(c, (0, max_bbox_len - len(c)), mode='constant') for c in class_ids]
        padded_class_ids = np.array(padded_class_ids, dtype=np.int32)
        return padded_bboxes, padded_class_ids

    def map_fn(image, bboxes, class_ids):
        bboxes, class_ids = pad_annotations(bboxes, class_ids)
        return image, (bboxes, class_ids)
    
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, None), dtype=tf.int32)
                                             ))
    dataset = dataset.map(lambda img, bboxes, class_ids: map_fn(img, bboxes, class_ids))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Path to your COCO dataset
annotations_file = 'instances_default.json'
img_dir = 'TrainingImages/'

# Load COCO annotations
coco = COCO(annotations_file)

# Create TensorFlow dataset
tf_dataset = create_tf_dataset(coco, img_dir)

# Example usage
for images, (bboxes, class_ids) in tf_dataset.take(1):
    print(images.shape)  # Shape: (batch_size, height, width, channels)
    print(bboxes)  # Padded bounding boxes
    print(class_ids)  # Padded class IDs
