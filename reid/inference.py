import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms, datasets, models
from reid.model import ReIDModel
from reid.sampler import RandomIdentitySampler


def load_model_for_inference(checkpoint_path, num_classes, device):
    '''
    Load model for inference
    '''
    
    model = ReIDModel(num_classes).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}  (epoch={ckpt.get('epoch', '?')})")
    return model


# Transform dùng lúc inference (không augment, chỉ resize + normalize)
def build_inference_transform(img_h=256, img_w=128):
    '''
    Load transform for inference
    
    Inference transform:
    - Resize (H, W) = (256, 128)
    - ToTensor
    - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    '''
    
    return transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    