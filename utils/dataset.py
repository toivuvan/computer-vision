import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DetectionDataset(Dataset):
    """
    Custom Dataset for Object Detection.
    Fully integrated with Albumentations for advanced augmentation (Cutout, ShiftScaleRotate, crops)
    and Multi-Scale Stride 16 grid target generation.
    """
    def __init__(self, json_path, image_dir, resolution=448, is_train=True, mosaic_prob=0.5):
        self.image_dir = image_dir
        self.is_train = is_train
        self.resolution = resolution
        self.grid_size = resolution // 16  # Stable Stride 16 Resolution Grid
        self.mosaic_prob = mosaic_prob if is_train else 0.0
        self.strong_aug_disabled = False
        
        # Load classes
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
            
        # Define Albumentations pipelines
        self._build_transforms()
    
    def _build_transforms(self):
        """Build or rebuild all Albumentations transform pipelines for the current resolution."""
        if self.is_train:
            if getattr(self, 'strong_aug_disabled', False):
                # Light transform pipeline for no-augment epoch fine-tuning phase
                # Retains HorizontalFlip and Normalize, but keeps bbox_params to avoid error.
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Resize(self.resolution, self.resolution),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3))
                self.mosaic_transform = self.transform
                return
                
            # Standard single-image augmentation pipeline
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(
                    size=(self.resolution, self.resolution),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.11),
                    p=0.5
                ),
                A.Resize(self.resolution, self.resolution),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), shear=(-10, 10), border_mode=0, p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.GaussNoise(p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3))
            
            # Mosaic-specific pipeline: image is already at correct resolution after mosaic,
            # so skip RandomResizedCrop and Resize to avoid double-resizing distortion.
            self.mosaic_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), shear=(-10, 10), border_mode=0, p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.GaussNoise(p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.2))
        else:
            self.transform = A.Compose([
                A.Resize(self.resolution, self.resolution),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.0))

    def set_resolution(self, resolution):
        """
        Dynamically update the target resolution and grid size for Multi-Scale training.
        Rebuilds all transform pipelines (standard + mosaic) for the new resolution.
        """
        self.resolution = resolution
        self.grid_size = resolution // 16  # Stable Stride 16 Resolution Grid
        self._build_transforms()
        
    def disable_strong_augmentations(self):
        """
        Disable Mosaic and strong geometric/noise augmentations for the final training stage.
        Swaps transform to a lighter pipeline (only HorizontalFlip and Normalize) to refine bbox predictions.
        """
        if getattr(self, 'strong_aug_disabled', False):
            return
            
        self.mosaic_prob = 0.0
        self.strong_aug_disabled = True
        self._build_transforms()
        print("\n=== [Dataset] Disabling Mosaic and strong augmentations (Affine, Cutout, Blur, Noise) for fine-tuning ===")
    
    def _load_image_and_boxes(self, idx):
        """Load a single image and its bounding boxes/labels by index."""
        example = self.examples[idx]
        img_path = os.path.join(self.image_dir, os.path.basename(example['file_name']))
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        bboxes = list(example['bboxes'])
        labels = list(example['labels'])
        return img_np, bboxes, labels
    
    def _mosaic_augment(self, idx):
        """
        Mosaic Augmentation: Compose 4 training images into a single mosaic image.
        Returns the mosaic image (numpy), bboxes (pascal_voc), and labels.
        """
        s = self.resolution  # target output size
        # Random cut point within [s*0.25, s*0.75] for more variation
        xc = int(random.uniform(s * 0.25, s * 0.75))
        yc = int(random.uniform(s * 0.25, s * 0.75))
        
        # Select 3 additional random indices (plus the current one)
        indices = [idx] + [random.randint(0, len(self.examples) - 1) for _ in range(3)]
        
        # Create output canvas
        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)  # gray fill
        mosaic_bboxes = []
        mosaic_labels = []
        
        for i, img_idx in enumerate(indices):
            img_np, bboxes, labels = self._load_image_and_boxes(img_idx)
            h_orig, w_orig = img_np.shape[:2]
            
            # Determine placement region on canvas for each of the 4 quadrants
            if i == 0:    # top-left
                x1a, y1a, x2a, y2a = 0, 0, xc, yc
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, 0, s, yc
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = 0, yc, xc, s
            else:         # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, s, s
            
            # Target region dimensions
            region_w = x2a - x1a
            region_h = y2a - y1a
            if region_w <= 0 or region_h <= 0:
                continue
            
            # Resize image to fit the region
            img_resized = cv2.resize(img_np, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
            mosaic_img[y1a:y2a, x1a:x2a] = img_resized
            
            # Transform bounding boxes to canvas coordinates
            scale_x = region_w / w_orig
            scale_y = region_h / h_orig
            
            for bbox, label in zip(bboxes, labels):
                # Original bbox in pascal_voc: [xmin, ymin, xmax, ymax]
                xmin = bbox[0] * scale_x + x1a
                ymin = bbox[1] * scale_y + y1a
                xmax = bbox[2] * scale_x + x1a
                ymax = bbox[3] * scale_y + y1a
                
                # Clip to canvas boundaries
                xmin = max(0.0, min(float(s), xmin))
                ymin = max(0.0, min(float(s), ymin))
                xmax = max(0.0, min(float(s), xmax))
                ymax = max(0.0, min(float(s), ymax))
                
                # Filter out boxes that are too small after clipping
                clipped_w = xmax - xmin
                clipped_h = ymax - ymin
                if clipped_w > 4 and clipped_h > 4:  # minimum 4px
                    mosaic_bboxes.append([xmin, ymin, xmax, ymax])
                    mosaic_labels.append(label)
        
        return mosaic_img, mosaic_bboxes, mosaic_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        use_mosaic = self.is_train and self.mosaic_prob > 0 and random.random() < self.mosaic_prob
        
        if use_mosaic:
            # Mosaic path: compose 4 images, then apply mosaic-specific augmentations
            img_np, bboxes, labels = self._mosaic_augment(idx)
            try:
                transformed = self.mosaic_transform(image=img_np, bboxes=bboxes, category_ids=labels)
                img_tensor = transformed['image']
                bboxes = transformed['bboxes']
                labels = transformed['category_ids']
            except Exception as e:
                fallback_transform = A.Compose([
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
                img_tensor = fallback_transform(image=img_np)['image']
                bboxes = []
                labels = []
        else:
            # Standard single-image path
            img_np, bboxes, labels = self._load_image_and_boxes(idx)
            try:
                transformed = self.transform(image=img_np, bboxes=bboxes, category_ids=labels)
                img_tensor = transformed['image']
                bboxes = transformed['bboxes']
                labels = transformed['category_ids']
            except Exception as e:
                fallback_transform = A.Compose([
                    A.Resize(self.resolution, self.resolution),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
                img_tensor = fallback_transform(image=img_np)['image']
                bboxes = []
                labels = []
        
        example = self.examples[idx]
            
        # Grid target generation (Stride 16)
        S = self.grid_size
        target = torch.zeros((10, S, S), dtype=torch.float32)
        
        # To handle overlapping centers in a cell, keep the one with smaller area
        cell_areas = torch.full((S, S), float('inf'))
        
        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            
            # Normalize bounding box coordinates to [0, 1] relative to the post-transformation resolution
            x1 = xmin / self.resolution
            y1 = ymin / self.resolution
            x2 = xmax / self.resolution
            y2 = ymax / self.resolution
            
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
                target[1 + int(label), row, col] = 1.0
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
