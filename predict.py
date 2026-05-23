import os
import argparse
import json
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models.detector import ResNetYOLO
from utils.nms import decode_predictions, non_maximum_suppression

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and generate object detection predictions.")
    parser.add_argument("--image_dir", required=True, type=str, help="Directory containing images to predict")
    parser.add_argument("--output", required=True, type=str, help="Path to save predictions predictions.json")
    parser.add_argument("--checkpoint", type=str, default="./models/best.pth", help="Path to best.pth checkpoint")
    parser.add_argument("--conf_threshold", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.45, help="IoU threshold for NMS")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Instantiate and Load Model
    model = ResNetYOLO(pretrained=False)
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'. Make sure you train the model first.")
        
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Robustly handle different checkpoint save formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    # Image Net normalization transforms
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
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
                img = Image.open(img_path).convert("RGB")
                w_orig, h_orig = img.size
                
                # Resize and pre-process image
                # Standard input size for optimal GPU model is 448x448
                img_resized = img.resize((448, 448), Image.BILINEAR)
                img_tensor = TF.to_tensor(img_resized)
                img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
                
                # Forward pass
                output = model(img_tensor)  # (1, 10, 14, 14)
                
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
