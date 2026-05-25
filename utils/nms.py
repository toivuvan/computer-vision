import torch
import numpy as np

def bbox_iou(box_a, box_b):
    """
    Compute IoU between two bounding boxes [xmin, ymin, xmax, ymax].
    Used during NMS post-processing.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    
    if union <= 0:
        return 0.0
    return intersection / union

def decode_predictions(prediction, img_width, img_height, conf_threshold=0.15):
    """
    Decode raw network grid output tensor into bounding boxes in original image resolution.
    prediction: tensor of shape (10, S, S)
    img_width, img_height: original image size
    conf_threshold: confidence threshold for filtering
    """
    S = prediction.shape[1]
    classes = ["person", "car", "dog", "cat", "chair"]
    
    # Apply activations to logits
    # Channel 0: objectness (sigmoid)
    pred_obj = torch.sigmoid(prediction[0, :, :]) # (S, S)
    # Channels 1-5: class scores (softmax)
    pred_class_probs = torch.softmax(prediction[1:6, :, :], dim=0) # (5, S, S)
    # Channels 6-9: coordinates (sigmoid offsets and sizes)
    pred_coords = torch.sigmoid(prediction[6:10, :, :]) # (4, S, S)
    
    # For each cell, find the class with the maximum probability (argmax)
    max_class_probs, max_class_indices = torch.max(pred_class_probs, dim=0) # (S, S)
    
    # Combined scores for the best class in each cell
    scores = pred_obj * max_class_probs # (S, S)
    
    # Find grid locations exceeding threshold
    rows, cols = torch.where(scores >= conf_threshold)
    
    decoded_boxes = []
    for row, col in zip(rows, cols):
        row = row.item()
        col = col.item()
        c_idx = max_class_indices[row, col].item()
        
        score = scores[row, col].item()
        class_name = classes[c_idx]
        
        # Decode center x, y and width, height relative to cell and image
        tx = pred_coords[0, row, col].item()
        ty = pred_coords[1, row, col].item()
        tw = pred_coords[2, row, col].item()
        th = pred_coords[3, row, col].item()
        
        xc = (col + tx) / S
        yc = (row + ty) / S
        w = tw
        h = th
        
        # Convert to corner bounding boxes [xmin, ymin, xmax, ymax]
        xmin = (xc - w / 2.0) * img_width
        ymin = (yc - h / 2.0) * img_height
        xmax = (xc + w / 2.0) * img_width
        ymax = (yc + h / 2.0) * img_height
        
        # Clip bounding box to fit inside original image boundaries
        xmin = max(0.0, min(float(img_width), xmin))
        ymin = max(0.0, min(float(img_height), ymin))
        xmax = max(0.0, min(float(img_width), xmax))
        ymax = max(0.0, min(float(img_height), ymax))
        
        # Require a valid area bounding box
        if xmax > xmin and ymax > ymin:
            decoded_boxes.append({
                "class": class_name,
                "confidence": round(score, 4),
                "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]
            })
            
    return decoded_boxes

def non_maximum_suppression(boxes, iou_threshold=0.5):
    """
    Perform class-wise Non-Maximum Suppression (NMS) to eliminate overlapping redundant boxes.
    boxes: list of dicts: [{"class": str, "confidence": float, "bbox": [4 values]}]
    iou_threshold: threshold above which overlapping boxes are suppressed
    """
    if not boxes:
        return []
        
    # Group boxes by their predicted class
    boxes_by_class = {}
    for box in boxes:
        cls = box["class"]
        if cls not in boxes_by_class:
            boxes_by_class[cls] = []
        boxes_by_class[cls].append(box)
        
    keep_boxes = []
    
    # Apply NMS for each class independently
    for cls, class_boxes in boxes_by_class.items():
        # Sort boxes descending by confidence score
        sorted_boxes = sorted(class_boxes, key=lambda b: b["confidence"], reverse=True)
        
        while sorted_boxes:
            best_box = sorted_boxes.pop(0)
            keep_boxes.append(best_box)
            
            # Keep only boxes that do not overlap significantly with the best box
            sorted_boxes = [
                box for box in sorted_boxes
                if bbox_iou(best_box["bbox"], box["bbox"]) < iou_threshold
            ]
            
    return keep_boxes
