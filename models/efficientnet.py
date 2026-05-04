import torch
import torch.nn as nn
import timm


class DRModel(nn.Module):
    """
    EfficientNet-B4 for Diabetic Retinopathy grading.

    Key design decisions:
    - EfficientNet-B4: best accuracy/efficiency tradeoff for medical imaging
    - Ordinal output head: respects that DR grades are ordered (0 < 1 < 2 < 3 < 4)
    - Dropout before classifier: reduces overfitting on small datasets
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        backbone_out = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)