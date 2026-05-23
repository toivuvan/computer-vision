import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.dataset import DetectionDataset
from models.detector import ResNetYOLO
from utils.loss import DetectionLoss
from utils.nms import decode_predictions, non_maximum_suppression, bbox_iou

def parse_args():
    parser = argparse.ArgumentParser(description="Train custom ResNet-50 FPN Object Detector.")
    parser.add_argument("--train_data", required=True, type=str, help="Path to train.json")
    parser.add_argument("--val_data", required=True, type=str, help="Path to val.json")
    parser.add_argument("--image_dir", required=True, type=str, help="Path to train images")
    parser.add_argument("--val_image_dir", required=True, type=str, help="Path to val images")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="Directory to save checkpoints")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--multi_scale", action="store_true", default=True, help="Enable multi-scale training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    """
    Custom collate function for DataLoader.
    Prevents PyTorch default_collate from trying to stack variable-size tensors (bboxes/labels) in metadata.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    metas = [item[2] for item in batch]
    return torch.stack(images, 0), torch.stack(targets, 0), metas

def evaluate_map(model, val_loader, device):
    """
    Computes exact mAP@0.5 on the validation set using the grading logic.
    """
    model.eval()
    
    # Store all ground truths and predictions
    classes = ["person", "car", "dog", "cat", "chair"]
    
    gt_boxes_by_class = {cls: {} for cls in classes}
    pred_boxes_by_class = {cls: [] for cls in classes}
    
    # Total ground truth boxes counter
    gt_counts = {cls: 0 for cls in classes}
    
    with torch.no_grad():
        for images, _, metas in val_loader:
            images = images.to(device)
            outputs = model(images)  # (batch, 10, S, S)
            
            for b in range(images.shape[0]):
                img_id = metas[b]['image_id']
                w_orig = metas[b]['width_orig']
                h_orig = metas[b]['height_orig']
                
                # Extract ground truth boxes for this image
                bboxes_gt = metas[b]['bboxes_orig']  # tensor
                labels_gt = metas[b]['labels_orig']
                
                # Group GTs by class
                for bbox, label in zip(bboxes_gt, labels_gt):
                    # Filter padded bboxes if any (width/height = 0)
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    cls_name = classes[label.item()]
                    if img_id not in gt_boxes_by_class[cls_name]:
                        gt_boxes_by_class[cls_name][img_id] = []
                    gt_boxes_by_class[cls_name][img_id].append({
                        "bbox": bbox.tolist(),
                        "matched": False
                    })
                    gt_counts[cls_name] += 1
                
                # Decode model predictions
                # Outputs[b] has shape (10, S, S)
                raw_predictions = decode_predictions(outputs[b], w_orig, h_orig, conf_threshold=0.05)
                # Apply class-wise NMS
                final_predictions = non_maximum_suppression(raw_predictions, iou_threshold=0.5)
                
                # Group predictions by class
                for pred in final_predictions:
                    cls_name = pred["class"]
                    pred_boxes_by_class[cls_name].append({
                        "image_id": img_id,
                        "confidence": pred["confidence"],
                        "bbox": pred["bbox"]
                    })
                    
    # Calculate AP for each class
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
                
        # Calculate precision-recall curves
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
    return mAP

def train(args):
    set_seed(args.seed)
    
    # 1. Setup Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Create Datasets
    # Default resolution is 448 (yielding 14x14 grid size)
    train_dataset = DetectionDataset(args.train_data, args.image_dir, resolution=448, is_train=True)
    val_dataset = DetectionDataset(args.val_data, args.val_image_dir, resolution=448, is_train=False)
    
    print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    
    # Validation uses num_workers=2
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    
    # 4. Instantiate Model, Loss, Optimizer, and Cosine Scheduler
    model = ResNetYOLO(pretrained=True).to(device)
    
    # Compute inverse frequency class weights to combat extreme class imbalance
    # Frequency counts: person: 5829, car: 1339, dog: 1028, cat: 833, chair: 1613
    # Absolute counts sum to 10642 annotations. Inverse frequency weights are:
    class_weights = torch.tensor([1.83, 7.95, 10.35, 12.78, 6.60], dtype=torch.float32).to(device)
    # Normalize weights so that their mean is 1.0 (sums to num_classes = 5)
    class_weights = class_weights / class_weights.sum() * 5.0
    
    criterion = DetectionLoss(class_weights=class_weights).to(device)
    
    # Differential Learning Rates: fine-tune backbone 10x slower than the head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1}, # 10x smaller learning rate for backbone parameters
        {"params": head_params, "lr": args.lr}            # normal learning rate for head parameters
    ], weight_decay=args.weight_decay)
    
    # Cosine learning rate decay for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Version-safe modern GradScaler for Mixed Precision (AMP)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    except (TypeError, ValueError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
    best_map = 0.0
    scales = [384, 416, 448, 480] # Multi-scale resolutions (multiples of 32)
    
    for epoch in range(args.epochs):
        model.train()
        
        # 5. Multi-Scale Training: pick a random resolution at the start of each epoch
        if args.multi_scale and torch.cuda.is_available():
            new_res = random.choice(scales)
            train_dataset.set_resolution(new_res)
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | Multi-scale target resolution set to: {new_res}x{new_res} ---")
        else:
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | Target resolution: 448x448 ---")
            
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for images, targets, _ in progress_bar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast context helper for version safety in PyTorch 2.6+
            try:
                autocast_context = torch.amp.autocast('cuda', enabled=torch.cuda.is_available())
            except (TypeError, ValueError, AttributeError):
                autocast_context = torch.cuda.amp.autocast(enabled=torch.cuda.is_available())
                
            # Forward pass under Mixed Precision autocast
            with autocast_context:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            # Backward and Optimizer step using GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        
        # 6. Evaluation Phase
        print(f"Calculating Validation mAP@0.5...")
        val_map = evaluate_map(model, val_loader, device)
        
        print(f"Epoch {epoch+1} Summary: Avg Train Loss = {avg_train_loss:.4f} | Val mAP@0.5 = {val_map:.4f}")
        
        # 7. Checkpoint Saving (Save best checkpoint)
        if val_map > best_map:
            best_map = val_map
            best_path = os.path.join(args.checkpoint_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': val_map,
            }, best_path)
            print(f"⭐ New Best Model saved with mAP@0.5 = {val_map:.4f} at {best_path}")
            
        # Also save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "latest.pth")
        torch.save(model.state_dict(), latest_path)

    print(f"\nTraining completed! Best Validation mAP@0.5 = {best_map:.4f}")

if __name__ == "__main__":
    args = parse_args()
    train(args)
