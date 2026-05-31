import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import LambdaLR


class BNNeck(nn.Module):
    """
    Trick 3.5 – BNNeck
      ft  ──────────────────────► Triplet loss + Center loss
      ft ──► BN ──► fi ──► FC ──► logits ──► ID loss (Label Smooth CE)
    • BN: bias.requires_grad=False
    • FC: no bias, Kaiming normal init
    Inference: dùng fi + cosine distance
    """
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias,   0)

        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_out")

    def forward(self, ft):
        fi     = self.bn(ft)
        logits = self.classifier(fi)
        return ft, fi, logits


class ReIDModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Trick 3.4 – Last stride = 1: 8×4 → 16×8, không thêm params
        backbone.layer4[0].conv2.stride         = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.bnneck = BNNeck(2048, num_classes)

    def forward(self, x):
        x  = self.backbone(x)
        # print("Feature map shape:", x.shape)
        ft = self.gap(x).view(x.size(0), -1)   # (B, 2048)
        ft, fi, logits = self.bnneck(ft)
        return ft, fi, logits

    @torch.no_grad()
    def inference(self, x):
        """
        Dùng lúc test/retrieval, KHÔNG dùng lúc train.
        Chỉ trả về fi đã L2-normalize (sau BNNeck, bỏ qua classifier).
        Sau khi normalize: cosine_similarity(a, b) = dot(a, b).

        Parameters
        ----------
        x : torch.Tensor (B, 3, H, W)

        Returns
        -------
        fi : torch.Tensor (B, 2048) – L2-normalized, dùng để retrieval
        """
        self.eval()
        x  = self.backbone(x)
        ft = self.gap(x).view(x.size(0), -1)  # (B, 2048)
        fi = self.bnneck.bn(ft)                # chỉ qua BN, bỏ classifier
        fi = F.normalize(fi, p=2, dim=1)       # L2-normalize
        return fi

