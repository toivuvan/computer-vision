import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import DetectionDataset
from models.detector import ResNetYOLO
from utils.nms import decode_predictions, non_maximum_suppression, bbox_iou

def compute_ap(recalls, precisions):
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    metas = [item[2] for item in batch]
    return torch.stack(images, 0), torch.stack(targets, 0), metas

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Validation Dataset
    val_dataset = DetectionDataset(
        json_path="./public/annotations/val.json", 
        image_dir="./public/val/images", 
        resolution=448, 
        is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    # 2. Instantiate and Load Model
    checkpoint_path = "./models/best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at '{checkpoint_path}'. Make sure you train the model first.")
        return
        
    print(f"Loading model checkpoint from: {checkpoint_path}")
    model = ResNetYOLO(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # 3. Extract all raw outputs and ground truths once
    print("Running validation inference to extract raw model logits...")
    all_raw_outputs = []
    all_metas = []
    
    with torch.no_grad():
        for images, _, metas in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)  # (batch, 10, S, S)
            
            for b in range(images.shape[0]):
                # Move prediction to CPU to free GPU memory
                all_raw_outputs.append(outputs[b].cpu())
                all_metas.append({
                    'image_id': metas[b]['image_id'],
                    'width_orig': metas[b]['width_orig'],
                    'height_orig': metas[b]['height_orig'],
                    'bboxes_orig': metas[b]['bboxes_orig'].tolist(),
                    'labels_orig': metas[b]['labels_orig'].tolist()
                })

    classes = ["person", "car", "dog", "cat", "chair"]
    
    # Grid search ranges
    conf_thresholds = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]
    iou_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    
    best_map = 0.0
    best_conf = 0.0
    best_iou = 0.0
    
    print("\nStarting Grid Search over Thresholds...")
    
    for conf_t in conf_thresholds:
        for iou_t in iou_thresholds:
            
            gt_boxes_by_class = {cls: {} for cls in classes}
            pred_boxes_by_class = {cls: [] for cls in classes}
            gt_counts = {cls: 0 for cls in classes}
            
            # Process each image prediction in memory
            for out, meta in zip(all_raw_outputs, all_metas):
                img_id = meta['image_id']
                w_orig = meta['width_orig']
                h_orig = meta['height_orig']
                
                # Group GTs by class
                bboxes_gt = meta['bboxes_orig']
                labels_gt = meta['labels_orig']
                
                for bbox, label in zip(bboxes_gt, labels_gt):
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    cls_name = classes[label]
                    if img_id not in gt_boxes_by_class[cls_name]:
                        gt_boxes_by_class[cls_name][img_id] = []
                    gt_boxes_by_class[cls_name][img_id].append({
                        "bbox": bbox,
                        "matched": False
                    })
                    gt_counts[cls_name] += 1
                
                # Decode predictions using current conf_t
                raw_predictions = decode_predictions(out, w_orig, h_orig, conf_threshold=conf_t)
                # Apply class-wise NMS using current iou_t
                final_predictions = non_maximum_suppression(raw_predictions, iou_threshold=iou_t)
                
                for pred in final_predictions:
                    cls_name = pred["class"]
                    pred_boxes_by_class[cls_name].append({
                        "image_id": img_id,
                        "confidence": pred["confidence"],
                        "bbox": pred["bbox"]
                    })
            
            # Calculate mAP
            aps = []
            for cls_name in classes:
                num_gt = gt_counts[cls_name]
                class_preds = sorted(pred_boxes_by_class[cls_name], key=lambda x: x["confidence"], reverse=True)
                class_gts = gt_boxes_by_class[cls_name]
                
                if num_gt == 0:
                    continue
                    
                tp_flags = []
                fp_flags = []
                
                for pred in class_preds:
                    img_id = pred["image_id"]
                    candidates = class_gts.get(img_id, [])
                    
                    best_iou = 0.0
                    best_idx = -1
                    
                    for idx, gt in enumerate(candidates):
                        if gt["matched"]:
                            continue
                        iou = bbox_iou(pred["bbox"], gt["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                            
                    if best_idx >= 0 and best_iou >= 0.5:
                        candidates[best_idx]["matched"] = True
                        tp_flags.append(1)
                        fp_flags.append(0)
                    else:
                        tp_flags.append(0)
                        fp_flags.append(1)
                        
                cumulative_tp = []
                cumulative_fp = []
                tp_sum = 0
                fp_sum = 0
                for tp, fp in zip(tp_flags, fp_flags):
                    tp_sum += tp
                    fp_sum += fp
                    cumulative_tp.append(tp_sum)
                    cumulative_fp.append(fp_sum)
                    
                recalls = [tp / num_gt if num_gt else 0.0 for tp in cumulative_tp]
                precisions = [tp / max(tp + fp, 1) for tp, fp in zip(cumulative_tp, cumulative_fp)]
                
                ap = compute_ap(recalls, precisions)
                aps.append(ap)
                
            mAP = np.mean(aps) if aps else 0.0
            
            print(f"Conf Threshold: {conf_t:.2f} | IoU Threshold: {iou_t:.2f} | mAP@0.5 = {mAP:.4f}")
            
            if mAP > best_map:
                best_map = mAP
                best_conf = conf_t
                best_iou = iou_t
                
    print("\n" + "="*50)
    print(f"⭐ GRID SEARCH COMPLETE!")
    print(f"Best Configuration:")
    print(f"  --conf_threshold {best_conf:.2f}")
    print(f"  --iou_threshold  {best_iou:.2f}")
    print(f"  Max mAP@0.5      = {best_map:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
