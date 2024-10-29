from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
            
        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        self.names_dict = {}

        # Store annotations per image, categories, and annotation IDs
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

coco_annotations_file = "instances_default.json"
coco_images_dir = "TrainingImages/"
coco = COCOParser(coco_annotations_file, coco_images_dir)

# Define a list of colors for bounding boxes (repeat colors if there are many categories)
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta", "green", "aqua"] * 10

# Number of images to display
num_imgs_to_disp = 4
total_images = len(coco.get_imgIds())  # Total number of images
sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]
img_ids = coco.get_imgIds()
selected_img_ids = [img_ids[i] for i in sel_im_idxs]

# Setup for plotting
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
ax = ax.ravel()

# Iterate over selected images
for i, im_id in enumerate(selected_img_ids):
    image = Image.open(f"{coco_images_dir}/{coco.get_imgName(im_id)}")
    ann_ids = coco.get_annIds(im_id)
    annotations = coco.load_anns(ann_ids)
    
    ax[i].imshow(image)  # Display the image
    ax[i].axis('off')  # Remove axis

    for ann in annotations:
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        x, y, w, h = [int(b) for b in bbox]
        class_id = ann["category_id"]
        class_name = coco.load_cats(class_id)[0]["name"]
        
        # Draw the bounding box
        color_ = color_list[class_id % len(color_list)]  # Modulo to loop through colors
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')
        ax[i].add_patch(rect)

        # Display the class name on the bounding box
        t_box = ax[i].text(x, y - 10, class_name, color='white', fontsize=10, weight='bold', 
                           bbox=dict(facecolor=color_, alpha=0.6, edgecolor='blue', boxstyle='round,pad=0.3'))

plt.tight_layout()
plt.show()
