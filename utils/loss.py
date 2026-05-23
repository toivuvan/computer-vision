import torch
import torch.nn as nn
import numpy as np

class DetectionLoss(nn.Module):
    """
    Custom Loss Function for Anchor-Free Object Detector.
    Combines:
    - Binary Cross Entropy (BCE) with logits for Objectness (positive/negative cells).
    - Cross Entropy for class probabilities.
    - Custom CIoU (Complete IoU) Loss + Smooth L1 Loss for bounding boxes.
    """
    def __init__(self, lambda_obj=5.0, lambda_noobj=0.2, lambda_class=1.0, lambda_box=2.0):
        super(DetectionLoss, self).__init__()
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.lambda_box = lambda_box
        
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

    def forward(self, predictions, targets):
        """
        predictions: tensor of shape (batch_size, 10, S, S)
        targets: tensor of shape (batch_size, 10, S, S)
        S = grid_size
        """
        batch_size, _, S, _ = predictions.shape
        
        # Split predicted logits and target values
        pred_obj = predictions[:, 0, :, :]           # (batch, S, S)
        pred_class = predictions[:, 1:6, :, :]        # (batch, 5, S, S)
        pred_coords = predictions[:, 6:10, :, :]      # (batch, 4, S, S)
        
        target_obj = targets[:, 0, :, :]             # (batch, S, S)
        target_class = targets[:, 1:6, :, :]          # (batch, 5, S, S)
        target_coords = targets[:, 6:10, :, :]        # (batch, 4, S, S)
        
        # Boolean masks for positive (object) and negative (no object) grid cells
        obj_mask = (target_obj == 1.0)
        noobj_mask = (target_obj == 0.0)
        
        # 1. Objectness Loss (BCE Loss with logits)
        loss_obj_all = self.bce_logits(pred_obj, target_obj)
        loss_obj = loss_obj_all[obj_mask].sum() if obj_mask.sum() > 0 else 0.0
        loss_noobj = loss_obj_all[noobj_mask].sum()
        
        total_obj_loss = self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj
        
        # Check if there are any objects in this batch
        num_pos = obj_mask.sum().item()
        if num_pos == 0:
            # If no objects, return only background classification loss
            return total_obj_loss / batch_size
            
        # 2. Classification Loss (Cross Entropy Loss)
        # Extract class logits for grid cells containing objects
        # Transpose class dimensions to facilitate masking
        pred_class_flat = pred_class.permute(0, 2, 3, 1)[obj_mask] # (num_pos, 5)
        target_class_flat = target_class.permute(0, 2, 3, 1)[obj_mask].argmax(dim=-1) # (num_pos)
        
        total_class_loss = self.ce_loss(pred_class_flat, target_class_flat)
        
        # 3. Bounding Box Loss (CIoU Loss + Smooth L1 Loss)
        # Extract coordinate logits
        p_coords = pred_coords.permute(0, 2, 3, 1)[obj_mask] # (num_pos, 4)
        t_coords = target_coords.permute(0, 2, 3, 1)[obj_mask] # (num_pos, 4)
        
        # Find cell indices (batch, row, col) for decoding
        batch_idx, rows, cols = torch.nonzero(obj_mask, as_tuple=True)
        
        # Decode predicted bounding boxes in normalized coordinates [0, 1]
        px_c = (cols.float() + torch.sigmoid(p_coords[:, 0])) / S
        py_c = (rows.float() + torch.sigmoid(p_coords[:, 1])) / S
        pw = torch.sigmoid(p_coords[:, 2])
        ph = torch.sigmoid(p_coords[:, 3])
        
        # Decode target bounding boxes in normalized coordinates [0, 1]
        tx_c = (cols.float() + t_coords[:, 0]) / S
        ty_c = (rows.float() + t_coords[:, 1]) / S
        tw = t_coords[:, 2]
        th = t_coords[:, 3]
        
        # Convert centers + sizes to corner coordinates [x1, y1, x2, y2]
        pred_x1 = px_c - pw / 2.0
        pred_y1 = py_c - ph / 2.0
        pred_x2 = px_c + pw / 2.0
        pred_y2 = py_c + ph / 2.0
        
        target_x1 = tx_c - tw / 2.0
        target_y1 = ty_c - th / 2.0
        target_x2 = tx_c + tw / 2.0
        target_y2 = ty_c + th / 2.0
        
        # Intersection bounding boxes
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Union area
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + 1e-7
        
        # IoU
        iou = inter_area / union_area
        
        # CIoU terms: distance regularization + aspect ratio similarity
        # 1. Square distance of box centers
        center_dist_sq = (px_c - tx_c)**2 + (py_c - ty_c)**2
        
        # 2. Smallest enclosing bounding box diagonal squared
        convex_x1 = torch.min(pred_x1, target_x1)
        convex_y1 = torch.min(pred_y1, target_y1)
        convex_x2 = torch.max(pred_x2, target_x2)
        convex_y2 = torch.max(pred_y2, target_y2)
        convex_diag_sq = (convex_x2 - convex_x1)**2 + (convex_y2 - convex_y1)**2 + 1e-7
        
        # 3. Distance penalty term
        d_term = center_dist_sq / convex_diag_sq
        
        # 4. Aspect Ratio penalty term
        # Aspect Ratio difference v
        v = (4.0 / (np.pi**2)) * (torch.atan(tw / (th + 1e-7)) - torch.atan(pw / (ph + 1e-7)))**2
        
        # Aspect Ratio balance term alpha
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + 1e-7)
            
        ciou = iou - d_term - alpha * v
        loss_ciou = (1.0 - ciou).sum()
        
        # Bbox smooth L1 regularization on raw grid logits to stabilize early training
        # We target sigmoid-inverse targets in grid-coordinates
        # We can map predicted grid logits directly to targets in grid-coordinates
        loss_l1 = self.smooth_l1(p_coords, t_coords)
        
        total_box_loss = loss_ciou + 0.3 * loss_l1
        
        # Combine all losses
        loss = (total_obj_loss + self.lambda_class * total_class_loss + self.lambda_box * total_box_loss) / batch_size
        return loss
