import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import LambdaLR


class LabelSmoothingCE(nn.Module):
    """
    Trick 3.3 – Label Smoothing  (paper eq.3)
      q_i = 1 - (N-1)/N * ε    nếu i == y   →  1 - ε + ε/N
      q_i = ε / N              ngược lại
    Tổng xác suất: (1 - ε + ε/N) + (N-1)*(ε/N) = 1 ✓
    """
    def __init__(self, num_classes, eps=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eps         = eps
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets):
        log_probs = self.log_softmax(logits)
        # ε/N cho tất cả class
        eps_per_class = self.eps / self.num_classes
        with torch.no_grad():
            smooth = torch.full_like(log_probs, eps_per_class)           # q_i = ε/N
            smooth.scatter_(1, targets.unsqueeze(1),
                            1.0 - self.eps + eps_per_class)              # q_y = 1 - ε + ε/N
        return -(smooth * log_probs).sum(dim=1).mean()


class TripletLossHardMining(nn.Module):
    """
    Triplet Loss hard mining  (paper eq.4)
      L_Tri = [d_p - d_n + α]+
      Hard positive : max pairwise dist cùng identity (loại self-pair)
      Hard negative : min pairwise dist khác identity
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, ft, labels):
        dist = torch.cdist(ft, ft, p=2)                       # (N,N) Euclidean
        same = labels.unsqueeze(1).eq(labels.unsqueeze(0))    # (N,N) bool

        # Hard positive – mask self-pair bằng -inf trước khi lấy max
        # (tránh bug: nhân 0 cho dist=0 vẫn ra 0, không phải hard positive thật)
        dist_ap = dist.clone()
        dist_ap[~same] = -float("inf")                        # ẩn khác class
        dist_ap.fill_diagonal_(-float("inf"))                  # ẩn self-pair
        dist_ap = dist_ap.max(dim=1)[0]                       # (N,) hard positive

        # Hard negative – mask same-class bằng +inf trước khi lấy min
        dist_an = dist.clone()
        dist_an[same] = float("inf")                          # ẩn cùng class
        dist_an = dist_an.min(dim=1)[0]                       # (N,) hard negative

        loss = torch.clamp(dist_ap - dist_an + self.margin, min=0.0).mean()
        return loss


class CenterLoss(nn.Module):
    """
    Trick 3.6 – Center Loss  (paper eq.5)
      L_C = 1/2 * Σ_j ||ft_j - c_{y_j}||²₂
      centers học qua optimizer riêng SGD lr=0.5
    """
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, ft, labels):
        c    = self.centers[labels]
        loss = 0.5 * ((ft - c) ** 2).sum(dim=1).mean() 
        return loss