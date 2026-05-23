import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class DetectionDataset(Dataset):
    """
    Custom Dataset for Object Detection.
    Supports advanced data augmentations (flipping, cropping, color jitter) and Multi-Scale training.
    """
    def __init__(self, json_path, image_dir, resolution=448, is_train=True):
        self.image_dir = image_dir
        self.is_train = is_train
        self.resolution = resolution
        self.grid_size = resolution // 16
        
        # Load classes
        # The 5 default classes are: "person", "car", "dog", "cat", "chair"
        self.classes = ["person", "car", "dog", "cat", "chair"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Load annotations JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create mapping from image ID to image info and annotations
        images_info = {img['id']: img for img in data['images']}
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
            
        # Parse into examples list
        self.examples = []
        for img_id, img_info in images_info.items():
            anns = annotations_by_image.get(img_id, [])
            
            bboxes = []
            labels = []
            for ann in anns:
                # bbox format: [xmin, ymin, xmax, ymax]
                bbox = ann['bbox']
                class_name = ann['class']
                if class_name in self.class_to_idx:
                    bboxes.append(bbox)
                    labels.append(self.class_to_idx[class_name])
                    
            self.examples.append({
                'id': img_id,
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'bboxes': bboxes,
                'labels': labels
            })
            
        # Standard normalization for ImageNet pre-trained models
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Color Jitter transform for data augmentation
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)

    def set_resolution(self, resolution):
        """
        Dynamically update the target resolution and grid size for Multi-Scale training.
        """
        self.resolution = resolution
        self.grid_size = resolution // 16

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        img_path = os.path.join(self.image_dir, os.path.basename(example['file_name']))
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img.size
        
        bboxes = list(example['bboxes']) # List of [xmin, ymin, xmax, ymax]
        labels = list(example['labels'])
        
        # Apply data augmentations if in training mode
        if self.is_train:
            # 1. Random Color Jitter
            if random.random() < 0.6:
                img = self.color_jitter(img)
                
            # 2. Random Horizontal Flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_bboxes = []
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax = bbox
                    new_xmin = w_orig - xmax
                    new_xmax = w_orig - xmin
                    new_bboxes.append([new_xmin, ymin, new_xmax, ymax])
                bboxes = new_bboxes

            # 3. Random Crop (keep objects whose center is inside the crop region)
            if random.random() < 0.5 and len(bboxes) > 0:
                crop_scale = random.uniform(0.8, 1.0)
                crop_w = int(w_orig * crop_scale)
                crop_h = int(h_orig * crop_scale)
                
                # Pick a random top-left corner
                cx1 = random.randint(0, w_orig - crop_w)
                cy1 = random.randint(0, h_orig - crop_h)
                cx2 = cx1 + crop_w
                cy2 = cy1 + crop_h
                
                # Filter bboxes based on their center
                new_bboxes = []
                new_labels = []
                for bbox, label in zip(bboxes, labels):
                    xmin, ymin, xmax, ymax = bbox
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    
                    if cx1 <= x_center <= cx2 and cy1 <= y_center <= cy2:
                        # Shift box and clip inside crop area
                        new_xmin = max(0.0, xmin - cx1)
                        new_ymin = max(0.0, ymin - cy1)
                        new_xmax = min(float(crop_w), xmax - cx1)
                        new_ymax = min(float(crop_h), ymax - cy1)
                        if new_xmax > new_xmin and new_ymax > new_ymin:
                            new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
                            new_labels.append(label)
                            
                # If we have at least one valid object left, execute the crop
                if len(new_bboxes) > 0:
                    img = img.crop((cx1, cy1, cx2, cy2))
                    w_orig, h_orig = img.size
                    bboxes = new_bboxes
                    labels = new_labels

        # Resize image and scale bboxes
        img_resized = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img_tensor = TF.to_tensor(img_resized)
        img_tensor = self.normalize(img_tensor)
        
        # Grid target generation
        # Shape: (10, grid_size, grid_size)
        # 10 channels = [objectness, c0, c1, c2, c3, c4, x_offset, y_offset, w, h]
        S = self.grid_size
        target = torch.zeros((10, S, S), dtype=torch.float32)
        
        # To handle overlapping centers in a cell, keep the one with smaller area
        cell_areas = torch.full((S, S), float('inf'))
        
        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            
            # Normalize bounding box coordinates to [0, 1] relative to current image size
            x1 = xmin / w_orig
            y1 = ymin / h_orig
            x2 = xmax / w_orig
            y2 = ymax / h_orig
            
            # Bounding box center, width, and height in [0, 1]
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            
            # Safe boundary check
            xc = min(0.9999, max(0.0, xc))
            yc = min(0.9999, max(0.0, yc))
            w = min(1.0, max(0.0, w))
            h = min(1.0, max(0.0, h))
            
            # Determine grid cell row and column
            col = int(xc * S)
            row = int(yc * S)
            
            area = w * h
            # If this cell is empty or has a larger object, assign this smaller object to it
            if area < cell_areas[row, col]:
                cell_areas[row, col] = area
                
                # Clear former classes if any
                target[0:6, row, col] = 0.0
                
                # 0. Objectness = 1.0
                target[0, row, col] = 1.0
                # 1-5. One-hot class label
                target[1 + label, row, col] = 1.0
                # 6-7. Bounding box center relative to cell top-left (range [0, 1])
                target[6, row, col] = xc * S - col
                target[7, row, col] = yc * S - row
                # 8-9. Bounding box width and height relative to image size (range [0, 1])
                target[8, row, col] = w
                target[9, row, col] = h
                
        # Return image, target grid, and metadata for inference/eval
        meta = {
            'image_id': example['id'],
            'width_orig': example['width'],
            'height_orig': example['height'],
            'bboxes_orig': torch.tensor(example['bboxes'], dtype=torch.float32),
            'labels_orig': torch.tensor(example['labels'], dtype=torch.long)
        }
        
        return img_tensor, target, meta
