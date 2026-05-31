import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class CustomReIDFeatureExtractor:
    """
    Feature extractor wrapper cho model ReID custom (ResNet50 + BNNeck).
    API tương tự OSNetFeatureExtractor:
      - extract(frame_bgr, bboxes) -> np.ndarray (N, D), L2-normalized
      - extract_single(frame_bgr, bbox) -> np.ndarray (D,)
    """

    def __init__(self,
                 checkpoint_path,
                 num_classes,
                 model_cls,              # truyền ReIDModel class của bạn vào đây
                 input_size=(256, 128),  # (H, W)
                 feature_dim=2048,
                 device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.feature_dim = feature_dim
        self.input_size = input_size

        print("Loading custom ReID model ...")
        self.model = model_cls(num_classes).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # ưu tiên key "model_state", fallback nếu checkpoint lưu thẳng state_dict
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),  # (H, W)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"  Custom ReID ready — feature_dim={self.feature_dim}, device={self.device}")
        if isinstance(ckpt, dict):
            print(f"  checkpoint epoch={ckpt.get('epoch', '?')}")

    @torch.no_grad()
    def extract(self, frame_bgr, bboxes):
        """
        Args:
            frame_bgr: np.ndarray (H, W, 3) BGR
            bboxes: np.ndarray (N, 4) [x1, y1, x2, y2]

        Returns:
            np.ndarray (N, feature_dim), L2-normalized
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        crops = []

        for bbox in bboxes:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            if x2 <= x1 or y2 <= y1:
                # crop rỗng -> ảnh đen fallback
                crop = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            else:
                crop = frame_bgr[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crops.append(self.transform(crop))

        batch = torch.stack(crops, dim=0).to(self.device)  # (N, 3, H, W)

        # dùng hàm inference của model custom (đã BN + L2 normalize)
        feats = self.model.inference(batch)  # (N, D)

        # an toàn: normalize lại 1 lần (idempotent nếu đã normalize rồi)
        feats = F.normalize(feats, p=2, dim=1)

        return feats.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_single(self, frame_bgr, bbox):
        return self.extract(frame_bgr, np.array([bbox], dtype=np.float32))[0]