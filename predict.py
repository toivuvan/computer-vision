import os
import argparse
import json
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models.detector import ResNetYOLO
from utils.nms import decode_predictions, non_maximum_suppression

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and generate object detection predictions.")
    parser.add_argument("--image_dir", required=True, type=str, help="Directory containing images to predict")
    parser.add_argument("--output", required=True, type=str, help="Path to save predictions predictions.json")
    parser.add_argument("--checkpoint", type=str, nargs="+", default=["./models/best.pth"], help="Path(s) to checkpoint(s) to load or average")
    parser.add_argument("--resolution", type=int, default=448, help="Inference resolution")
    parser.add_argument("--conf_threshold", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.50, help="IoU threshold for NMS")
    return parser.parse_args()

def average_checkpoints(checkpoint_paths, device):
    """
    Loads multiple PyTorch checkpoints and averages their model state dicts.
    """
    print(f"Loading and averaging {len(checkpoint_paths)} checkpoints...")
    avg_state_dict = None
    count = 0
    
    for path in checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at '{path}'")
        
        print(f"  -> Loading: {path}")
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
            
        state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
        
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k, v in state_dict.items():
                avg_state_dict[k] += v.float()
        count += 1
        
    for k in avg_state_dict.keys():
        avg_state_dict[k] = (avg_state_dict[k] / count).to(state_dict[k].dtype)
        
    print("Checkpoint averaging completed successfully.")
    return avg_state_dict

def main():
    args = parse_args()
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Instantiate and Load Model
    model = ResNetYOLO(pretrained=False)
    
    # Load and average state dicts
    try:
        avg_state_dict = average_checkpoints(args.checkpoint, device)
        model.load_state_dict(avg_state_dict)
    except Exception as e:
        print(f"Error loading checkpoints: {e}")
        raise e
        
    model = model.to(device)
    model.eval()
    
    # Define exact same transform pipeline as val_dataset in train.py
    transform = A.Compose([
        A.Resize(args.resolution, args.resolution),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # 3. Locate Images
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(valid_exts)]
    print(f"Found {len(img_files)} images in '{args.image_dir}' for prediction.")
    
    predictions_json = []
    
    # 4. Inference loop
    with torch.no_grad():
        for filename in tqdm(img_files, desc="Inferring"):
            img_path = os.path.join(args.image_dir, filename)
            
            try:
                # Load image with OpenCV in BGR, convert to RGB
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    raise ValueError(f"Could not read image: {img_path}")
                
                h_orig, w_orig = img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Apply transform
                transformed = transform(image=img_rgb)
                img_tensor = transformed['image'].unsqueeze(0).to(device)
                
                # Forward pass
                output = model(img_tensor)  # shape (1, 10, S, S)
                
                # Decode predictions
                raw_boxes = decode_predictions(
                    output[0], 
                    w_orig, 
                    h_orig, 
                    conf_threshold=args.conf_threshold
                )
                
                # Apply class-wise NMS
                final_boxes = non_maximum_suppression(
                    raw_boxes, 
                    iou_threshold=args.iou_threshold
                )
                
                # Append result
                predictions_json.append({
                    "image_id": filename,
                    "boxes": final_boxes
                })
                
            except Exception as e:
                print(f"Error predicting image {filename}: {e}")
                # Ensure the entry is still generated even if failed (empty list)
                predictions_json.append({
                    "image_id": filename,
                    "boxes": []
                })
                
    # 5. Save output predictions JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(predictions_json, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated predictions file: '{args.output}'")

if __name__ == "__main__":
    main()
