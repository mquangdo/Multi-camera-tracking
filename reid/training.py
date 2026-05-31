import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import LambdaLR
from reid.loss import LabelSmoothingCE, TripletLossHardMining, CenterLoss
from reid.model import ReIDModel
from reid.sampler import RandomIdentitySampler
import time
import random

CFG = {
    "data_dir"         : "/kaggle/input/datasets/mrfreddy0209/quang-quan3/kaggle/working/quang_quan",       # thư mục chứa 702 subfolder
    "save_dir"         : "./checkpoints_quangquan",
    "num_epochs"       : 120,
    "P"                : 16,              # số identity / batch  (paper: P=16)
    "K"                : 4,               # số ảnh / identity    (paper: K=4)
    "img_h"            : 256,
    "img_w"            : 128,
    "base_lr"          : 3.5e-4,          # paper: initial lr = 3.5e-4
    "weight_decay"     : 5e-4,
    "triplet_margin"   : 0.3,             # paper: m = 0.3
    "label_smooth_eps" : 0.1,             # paper: ε = 0.1
    "beta_center"      : 0.0005,          # paper: β = 0.0005
    "center_lr"        : 0.1,
    "num_workers"      : 4,
    "seed"             : 42,
}

def build_dataloader(cfg):
    """
    Augment theo paper sec.2 (không có Random Erasing – bỏ trick 3.2):
      resize → pad 10px → random crop → hflip p=0.5 → normalize
    """
    transform = transforms.Compose([
        transforms.Resize((cfg["img_h"], cfg["img_w"])),
        transforms.Pad(10),
        transforms.RandomCrop((cfg["img_h"], cfg["img_w"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(cfg["data_dir"], transform=transform)
    sampler = RandomIdentitySampler(dataset, P=cfg["P"], K=cfg["K"])
    loader  = DataLoader(
        dataset,
        batch_size  = cfg["P"] * cfg["K"],
        sampler     = sampler,
        num_workers = cfg["num_workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    return loader, len(dataset.classes)


def build_model(num_classes, device):
    model = ReIDModel(num_classes).to(device)
    return model


def build_losses(num_classes, cfg, device):
    criterion_id     = LabelSmoothingCE(num_classes, eps=cfg["label_smooth_eps"]).to(device)
    criterion_tri    = TripletLossHardMining(margin=cfg["triplet_margin"]).to(device)
    criterion_center = CenterLoss(num_classes, feat_dim=2048).to(device)
    return criterion_id, criterion_tri, criterion_center


def build_optimizers(model, criterion_center, cfg):
    """
    Paper sec.2 step 8:
      - Adam, lr=3.5e-4, weight_decay=5e-4
      - Center loss: SGD lr=0.5 (optimizer riêng, không dùng scheduler)
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["base_lr"],
        weight_decay=cfg["weight_decay"],
    )
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=cfg["center_lr"])
    return optimizer, optimizer_center


def build_scheduler(optimizer):
    """
    Trick 3.1 – Warmup LR  (paper eq.1, epoch 1-indexed)
      t ≤ 10        : lr(t) = 3.5e-5 * t/10  → tuyến tính warmup
      10 < t ≤ 40   : lr    = 3.5e-4          (factor = 1.0)
      40 < t ≤ 70   : lr    = 3.5e-5          (factor = 0.1)
      70 < t ≤ 120  : lr    = 3.5e-6          (factor = 0.01)

    LambdaLR nhận epoch 0-indexed → cộng 1 để khớp paper.
    """
    def lr_lambda(epoch):          # epoch: 0-indexed từ LambdaLR
        t = epoch + 1              # chuyển sang 1-indexed như paper
        if t <= 10:
            return t / 10          # warmup: base_lr * (t/10), bắt đầu từ base_lr/10
        elif t <= 40:
            return 1.0
        elif t <= 70:
            return 0.1
        else:
            return 0.01

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, criterion_id, criterion_tri,
                    criterion_center, optimizer, optimizer_center,
                    cfg, device):
    """
    Forward + backward cho 1 epoch.
    Total loss = L_ID + L_Triplet + β * L_Center  (paper eq.6)
      - L_ID     : Label Smooth CE trên fi  (sau BNNeck)
      - L_Triplet: Hard-mining Triplet trên ft (trước BNNeck)
      - L_Center : Center Loss trên ft
    Returns dict metrics.
    """
    model.train()
    sum_loss = sum_id = sum_tri = sum_ctr = 0.0
    correct  = total  = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        ft, fi, logits = model(imgs)

        loss_id  = criterion_id(logits, labels)
        loss_tri = criterion_tri(ft, labels)
        loss_ctr = criterion_center(ft, labels)
        loss     = loss_id + loss_tri + cfg["beta_center"] * loss_ctr

        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()

        # Rescale gradient của centers trước update
        # (backward đã nhân beta_center vào gradient của centers,
        #  cần chia lại để optimizer_center update đúng tốc độ)
        for param in criterion_center.parameters():
            if param.grad is not None:
                param.grad.data *= 1.0 / cfg["beta_center"]
        optimizer_center.step()

        sum_loss += loss.item()
        sum_id   += loss_id.item()
        sum_tri  += loss_tri.item()
        sum_ctr  += loss_ctr.item()
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)

    n = len(loader)
    return {
        "loss"    : sum_loss / n,
        "loss_id" : sum_id   / n,
        "loss_tri": sum_tri  / n,
        "loss_ctr": sum_ctr  / n,
        "acc"     : 100.0 * correct / total,
    }


def save_checkpoint(path, epoch, model, optimizer, scheduler,
                    criterion_center, metrics, cfg):
    torch.save({
        "epoch"           : epoch,
        "model_state"     : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "scheduler_state" : scheduler.state_dict(),
        "center_state"    : criterion_center.state_dict(),
        "loss"            : metrics["loss"],
        "acc"             : metrics["acc"],
        "cfg"             : cfg,
    }, path)
    print(f"Checkpoint saved: {path}")
    
def load_checkpoint(resume_path, model, optimizer, scheduler,
                    criterion_center, device):
    """
    Restore model / optimizer / scheduler / center weights từ file .pth.
    Trả về (start_epoch, best_loss, best_epoch).
    """
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {resume_path}")

    print(f"  → Loading checkpoint : {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    criterion_center.load_state_dict(ckpt["center_state"])

    start_epoch = ckpt["epoch"]                      # epoch đã hoàn thành
    best_loss   = ckpt.get("loss",  float("inf"))
    best_epoch  = ckpt.get("epoch", -1)

    print(f"  → Resumed  : epoch={start_epoch}  loss={best_loss:.4f}  acc={ckpt.get('acc', 0):.2f}%")
    return start_epoch, best_loss, best_epoch

def load_model_for_inference(checkpoint_path, num_classes, device):
    """
    Load ReIDModel từ checkpoint, chuyển sang eval mode.

    Parameters
    ----------
    checkpoint_path : str
    num_classes     : int – phải khớp với lúc train (702 với DukeMTMC)
    device          : torch.device

    Returns
    -------
    model : ReIDModel ở eval mode, sẵn sàng extract_features
    """
    model = ReIDModel(num_classes).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}  (epoch={ckpt.get('epoch', '?')})")
    return model


# Transform dùng lúc inference (không augment, chỉ resize + normalize)
def build_inference_transform(img_h=256, img_w=128):
    
    return transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    
def train(cfg=CFG, resume=None):
    """
    Parameters
    ----------
    cfg    : dict config (xem CFG ở đầu file)
    resume : str | None
        - None            → train từ đầu
        - "path/ckpt.pth" → tiếp tục từ checkpoint đó
        Ví dụ:
          train(CFG)                                        # train mới
          train(CFG, resume="./checkpoints/epoch_050.pth") # resume epoch 50
          train(CFG, resume="./checkpoints/best_model.pth")# resume best
    """
    # --- seed ---
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    print(f"Tricks     : Warmup LR | Label Smoothing | Last Stride=1 | BNNeck | Center Loss")
    print(f"Skip       : Random Erasing Augmentation (trick 3.2)")
    print("=" * 72)

    # --- build tất cả components ---
    loader, num_classes             = build_dataloader(cfg)
    model                           = build_model(num_classes, device)
    criterion_id, criterion_tri, \
        criterion_center            = build_losses(num_classes, cfg, device)
    optimizer, optimizer_center     = build_optimizers(model, criterion_center, cfg)
    scheduler                       = build_scheduler(optimizer)

    print(f"Identities : {num_classes}")
    print(f"Batch      : P={cfg['P']} × K={cfg['K']} = {cfg['P']*cfg['K']}")
    print(f"Steps/epoch: {len(loader)}")

    # --- resume (nếu có) ---
    start_epoch = 0
    best_loss   = float("inf")
    best_epoch  = -1

    if resume is not None:
        start_epoch, best_loss, best_epoch = load_checkpoint(
            resume, model, optimizer, scheduler, criterion_center, device
        )

    remaining = cfg["num_epochs"] - start_epoch
    if remaining <= 0:
        print(f"[!] Checkpoint đã train đủ {cfg['num_epochs']} epoch. "
              f"Tăng cfg['num_epochs'] để tiếp tục.")
        return []

    mode = f"Resume từ epoch {start_epoch+1}" if resume else "Train từ đầu"
    print(f"Mode       : {mode}  →  epoch {cfg['num_epochs']}  ({remaining} epochs)")
    print("=" * 72)

    history = []

    for epoch in range(start_epoch, cfg["num_epochs"]):
        t0 = time.time()

        metrics = train_one_epoch(
            model, loader,
            criterion_id, criterion_tri, criterion_center,
            optimizer, optimizer_center,
            cfg, device,
        )
        scheduler.step()

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]
        history.append(metrics)

        print(
            f"[{epoch+1:3d}/{cfg['num_epochs']}] "
            f"Loss {metrics['loss']:.4f} "
            f"(ID {metrics['loss_id']:.4f} | "
            f"Tri {metrics['loss_tri']:.4f} | "
            f"Ctr {metrics['loss_ctr']:.4f})  "
            f"Acc {metrics['acc']:.2f}%  "
            f"LR {cur_lr:.2e}  "
            f"{elapsed:.0f}s"
        )

        # ── lưu best checkpoint ──────────────────────────────
        if metrics["loss"] < best_loss:
            best_loss  = metrics["loss"]
            best_epoch = epoch + 1
            save_checkpoint(
                path             = os.path.join(cfg["save_dir"], "best_model.pth"),
                epoch            = best_epoch,
                model            = model,
                optimizer        = optimizer,
                scheduler        = scheduler,
                criterion_center = criterion_center,
                metrics          = metrics,
                cfg              = cfg,
            )
            print(f"  ✓ Best saved → epoch={best_epoch}  loss={best_loss:.4f}")

        # ── lưu periodic checkpoint mỗi 10 epoch ────────────
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(cfg["save_dir"], f"epoch_{epoch+1:03d}.pth")
            save_checkpoint(
                path             = ckpt_path,
                epoch            = epoch + 1,
                model            = model,
                optimizer        = optimizer,
                scheduler        = scheduler,
                criterion_center = criterion_center,
                metrics          = metrics,
                cfg              = cfg,
            )
            print(f"  → Checkpoint : {ckpt_path}")

    print("=" * 72)
    print(f"Done.  Best epoch={best_epoch}  best loss={best_loss:.4f}")
    print(f"Best model : {cfg['save_dir']}/best_model.pth")
    print("[Inference] load fi (sau BNNeck) + cosine distance để retrieval.")

    return history