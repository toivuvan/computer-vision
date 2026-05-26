import torch
import torch.nn as nn
import torchvision.models as models

class ResNetYOLO(nn.Module):
    """
    Object Detection Model using ConvNeXt-Tiny Backbone with FPN (Feature Pyramid Network).
    Fuses semantic stage 3 (stride 32, 768 channels) and high-resolution stage 2 (stride 16, 384 channels) features.
    Outputs a fine-grained grid (S x S x 10) where S = resolution // 16.
    """
    def __init__(self, pretrained=True):
        super(ResNetYOLO, self).__init__()
        
        # Safe loading of torchvision convnext_tiny backbone
        try:
            if pretrained:
                from torchvision.models import ConvNeXt_Tiny_Weights
                backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            else:
                backbone = models.convnext_tiny(weights=None)
        except (ImportError, TypeError):
            # Fallback for older torchvision versions
            backbone = models.convnext_tiny(pretrained=pretrained)
            
        # Store features submodule so PyTorch registers and tracks all its parameters
        self.backbone_features = backbone.features
        
        # FPN Projection layers (reduce channels to 256)
        # ConvNeXt stage 3 outputs 768 channels; stage 2 outputs 384 channels
        self.proj_l4 = nn.Conv2d(768, 256, kernel_size=1, bias=False)
        self.proj_l3 = nn.Conv2d(384, 256, kernel_size=1, bias=False)
        
        # FPN Bilinear Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                
        # Custom Detection Head
        # Processes the fused map (batch, 512, H/16, W/16) and predicts (batch, 10, H/16, W/16)
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(), # Modern swish activation function
            nn.Dropout(0.3),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            # Predict 10 output channels: 1 (objectness) + 5 (classes) + 4 (bbox coordinates)
            nn.Conv2d(128, 10, kernel_size=1)
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
        x = self.backbone_features[3](x)
        x = self.backbone_features[4](x)
        c3 = self.backbone_features[5](x) # stride 16: (batch, 384, H/16, W/16)
        x = self.backbone_features[6](c3)
        c4 = self.backbone_features[7](x) # stride 32: (batch, 768, H/32, W/32)
        
        # Project layers to equal channels (256)
        p4 = self.proj_l4(c4) # shape: (batch, 256, H/32, W/32)
        p3 = self.proj_l3(c3) # shape: (batch, 256, H/16, W/16)
        
        # Bilinear Upsample p4 to match shape of p3
        p4_upsampled = self.upsample(p4) # shape: (batch, 256, H/16, W/16)
        
        # Concatenate features along the channel dimension (256 + 256 = 512 channels)
        fused = torch.cat([p3, p4_upsampled], dim=1) # shape: (batch, 512, H/16, W/16)
        
        # Predict grid outputs
        out = self.head(fused) # shape: (batch, 10, H/16, W/16)
        
        return out
