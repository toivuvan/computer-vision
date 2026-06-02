import torch
import torch.nn as nn
import torchvision.models as models

class ResNetYOLO(nn.Module):
    """
    Object Detection Model using ConvNeXt-Small Backbone with FPN (Feature Pyramid Network),
    PANet (Path Aggregation Network), and Decoupled Detection Heads (separate Classification & Regression branches).
    ConvNeXt-Small has deeper stages (depth [3,3,27,3]) compared to ConvNeXt-Tiny ([3,3,9,3]),
    providing richer feature representations while maintaining the same channel layout (192/384/768).
    Outputs a fine-grained grid (S x S x 10) where S = resolution // 16.
    """
    def __init__(self, pretrained=True):
        super(ResNetYOLO, self).__init__()
        
        # Safe loading of torchvision convnext_small backbone
        try:
            if pretrained:
                from torchvision.models import ConvNeXt_Small_Weights
                backbone = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            else:
                backbone = models.convnext_small(weights=None)
        except (ImportError, TypeError):
            # Fallback for older torchvision versions
            backbone = models.convnext_small(pretrained=pretrained)
            
        # Store features submodule so PyTorch registers and tracks all its parameters
        self.backbone_features = backbone.features
        
        # FPN Projection layers (reduce channels to 256)
        # ConvNeXt stage 3 outputs 768 channels; stage 2 outputs 384 channels; stage 1 outputs 192 channels (stride 8)
        self.proj_l4 = nn.Conv2d(768, 256, kernel_size=1, bias=False)
        self.proj_l3 = nn.Conv2d(384, 256, kernel_size=1, bias=False)
        self.proj_l2 = nn.Conv2d(192, 256, kernel_size=1, bias=False)
        
        # FPN Bilinear Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # PANet Bottom-Up layers
        # Downsample from N2 (stride 8) to N3 (stride 16): Conv 3x3 Stride 2
        self.downsample_n2_to_n3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_n3 = nn.BatchNorm2d(256)
        self.silu = nn.SiLU()
                
        # 1. Decoupled Classification Branch (6 channels: 1 objectness + 5 class probabilities)
        self.cls_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(), # Modern swish activation function
            nn.Dropout(0.3),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Conv2d(128, 6, kernel_size=1)
        )
        
        # 2. Decoupled Regression Branch (4 channels: x, y, w, h bbox coordinates)
        self.reg_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Conv2d(128, 4, kernel_size=1)
        )

    def forward(self, x):
        # Extract features using ConvNeXt-Tiny stage submodules
        # index 0: Stem (stride 4, outputs 96)
        # index 1: Stage 0 (stride 4, outputs 96)
        # index 2: Downsample (stride 8, outputs 192)
        # index 3: Stage 1 (stride 8, outputs 192)
        # index 4: Downsample (stride 16, outputs 384)
        # index 5: Stage 2 (stride 16, outputs 384)
        # index 6: Downsample (stride 32, outputs 768)
        # index 7: Stage 3 (stride 32, outputs 768)
        x = self.backbone_features[0](x)
        x = self.backbone_features[1](x)
        x = self.backbone_features[2](x)
        c2 = self.backbone_features[3](x) # stride 8: (batch, 192, H/8, W/8)
        x = self.backbone_features[4](c2)
        c3 = self.backbone_features[5](x) # stride 16: (batch, 384, H/16, W/16)
        x = self.backbone_features[6](c3)
        c4 = self.backbone_features[7](x) # stride 32: (batch, 768, H/32, W/32)
        
        # --- Top-Down Path (FPN) ---
        # Project layers to equal channels (256)
        p4 = self.proj_l4(c4) # shape: (batch, 256, H/32, W/32)
        p3 = self.proj_l3(c3) + self.upsample(p4) # shape: (batch, 256, H/16, W/16)
        p2 = self.proj_l2(c2) + self.upsample(p3) # shape: (batch, 256, H/8, W/8)
        
        # --- Bottom-Up Path (PANet) ---
        n2 = p2 # shape: (batch, 256, H/8, W/8)
        # Downsample n2 to stride 16 and fuse with p3
        n3 = p3 + self.silu(self.bn_n3(self.downsample_n2_to_n3(n2))) # shape: (batch, 256, H/16, W/16)
        
        # --- Feature Fusion for Single-Scale Head ---
        p4_upsampled = self.upsample(p4) # shape: (batch, 256, H/16, W/16)
        
        # Concatenate features along the channel dimension (256 + 256 = 512 channels)
        fused = torch.cat([n3, p4_upsampled], dim=1) # shape: (batch, 512, H/16, W/16)
        
        # Predict grid outputs using independent specialized Decoupled Heads
        cls_out = self.cls_head(fused) # shape: (batch, 6, H/16, W/16)
        reg_out = self.reg_head(fused) # shape: (batch, 4, H/16, W/16)
        
        # Concatenate outputs back to (batch, 10, H/16, W/16) with original channel layout
        # Channels 0-5 are classification (objectness + class scores), 6-9 are regression coordinates
        out = torch.cat([cls_out, reg_out], dim=1)
        
        return out
