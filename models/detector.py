import torch
import torch.nn as nn
import torchvision.models as models

class ResNetYOLO(nn.Module):
    """
    Object Detection Model using ResNet-34 Backbone.
    Fully convolutional design, allowing dynamic input resolution.
    Predicts 10 channels per grid cell:
    - Channel 0: Objectness (logit)
    - Channels 1-5: Class logits (5 classes)
    - Channels 6-9: Bounding box offsets [tx, ty, tw, th] (logits)
    """
    def __init__(self, pretrained=True):
        super(ResNetYOLO, self).__init__()
        
        # Safe loading of torchvision resnet34 backbone
        try:
            if pretrained:
                from torchvision.models import ResNet34_Weights
                backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            else:
                backbone = models.resnet34(weights=None)
        except (ImportError, TypeError):
            # Fallback for older torchvision versions
            backbone = models.resnet34(pretrained=pretrained)
            
        # Extract layers except the global average pooling and fully connected layers
        # Input shape: (batch_size, 3, H, W)
        # Output shape after backbone: (batch_size, 512, H/32, W/32)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Freeze initial layers of ResNet-34 to speed up training on CPU/GPU
        # and preserve pre-trained low-level features
        # ResNet-34 children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        # We freeze conv1, bn1, relu, maxpool, layer1, and layer2
        children_list = list(self.backbone.children())
        for i in range(6):  # Freeze up to layer2
            for param in children_list[i].parameters():
                param.requires_grad = False
                
        # Custom Detection Head
        # Processes the (batch, 512, S, S) feature map and predicts (batch, 10, S, S)
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
        # Extract features
        features = self.backbone(x)
        
        # Predict grid outputs
        out = self.head(features)
        
        return out
