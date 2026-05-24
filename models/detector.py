import torch
import torch.nn as nn
import torchvision.models as models

class ResNetYOLO(nn.Module):
    """
    Object Detection Model using ResNet-50 Backbone with FPN (Feature Pyramid Network).
    Fuses semantic layer4 (stride 32) and high-resolution layer3 (stride 16) features.
    Outputs a fine-grained grid (S x S x 10) where S = resolution // 16.
    """
    def __init__(self, pretrained=True):
        super(ResNetYOLO, self).__init__()
        
        # Safe loading of torchvision resnet50 backbone
        try:
            if pretrained:
                from torchvision.models import ResNet50_Weights
                backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                backbone = models.resnet50(weights=None)
        except (ImportError, TypeError):
            # Fallback for older torchvision versions
            backbone = models.resnet50(pretrained=pretrained)
            
        # Extract features at multiple levels of ResNet-50
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1 # Output stride 4: channels 256
        self.layer2 = backbone.layer2 # Output stride 8: channels 512
        self.layer3 = backbone.layer3 # Output stride 16: channels 1024
        self.layer4 = backbone.layer4 # Output stride 32: channels 2048
        
        # Keep all backbone layers trainable to support Differential Learning Rates
        # (Fine-tuning backbone parameters at a smaller learning rate on GPU)
        
        # FPN Projection layers for stride 8 resolution
        self.proj_l4 = nn.Conv2d(2048, 128, kernel_size=1, bias=False)
        self.proj_l3 = nn.Conv2d(1024, 128, kernel_size=1, bias=False)
        self.proj_l2 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        
        # FPN Bilinear Upsampling layers
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
                
        # Custom Detection Head
        # Processes the fused map (batch, 512, H/8, W/8) and predicts (batch, 10, H/8, W/8)
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
        # Initial stages
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)
        c2 = self.layer2(c1) # stride 8: (batch, 512, H/8, W/8)
        c3 = self.layer3(c2) # stride 16: (batch, 1024, H/16, W/16)
        c4 = self.layer4(c3) # stride 32: (batch, 2048, H/32, W/32)
        
        # Project layers to stride 8 size
        p4 = self.proj_l4(c4) # shape: (batch, 128, H/32, W/32)
        p3 = self.proj_l3(c3) # shape: (batch, 128, H/16, W/16)
        p2 = self.proj_l2(c2) # shape: (batch, 256, H/8, W/8)
        
        # Upsample p4 and p3 to stride 8
        p4_upsampled = self.upsample4x(p4) # shape: (batch, 128, H/8, W/8)
        p3_upsampled = self.upsample2x(p3) # shape: (batch, 128, H/8, W/8)
        
        # Concatenate features along the channel dimension (256 + 128 + 128 = 512 channels)
        fused = torch.cat([p2, p3_upsampled, p4_upsampled], dim=1) # shape: (batch, 512, H/8, W/8)
        
        # Predict grid outputs
        out = self.head(fused) # shape: (batch, 10, H/8, W/8)
        
        return out
