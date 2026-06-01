import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import glob
import numpy as np
import torch
from PIL import Image


def parse_imname_subfolder(impath, id_from="folder"):
    """
    Parse person id (pid) and camera id (cam) from a path.

    If id_from="folder", pid is read from the parent folder name.
    If id_from="filename", pid is read from leading digits of filename.
    Camera id is parsed from pattern "_c{cam}" in filename, default 0.
    """
    parts = impath.replace("\\", "/").split("/")
    filename = parts[-1]

    if id_from == "folder":
        try:
            pid = int(parts[-2])
        except Exception:
            m = re.match(r"(\d+)", filename)
            pid = int(m.group(1)) if m else -1
    else:
        m = re.match(r"(\d+)", filename)
        pid = int(m.group(1)) if m else -1

    mcam = re.search(r"_c(\d+)", filename)
    cam = int(mcam.group(1)) if mcam else 0
    return pid, cam


def collect_images_with_subfolder(root_dir, extensions=(".jpg", ".jpeg", ".png"), id_from="folder"):
    """
    Recursively collect images and return list of (path, pid, cam).
    """
    images = []
    for ext in extensions:
        pattern = os.path.join(root_dir, "**", f"*{ext}")
        for p in glob.glob(pattern, recursive=True):
            pid, cam = parse_imname_subfolder(p, id_from=id_from)
            if pid >= 0:
                images.append((p, pid, cam))
    return images


@torch.no_grad()
def extract_features(model, image_paths, transform, device, batch_size=64):
    """
    Extract features for image_paths. Corrupted images are skipped.
    Returns a tensor of shape (N, D).
    """
    model.eval()
    all_feats = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]

        imgs_list = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs_list.append(transform(img))
            except Exception as e:
                print(f"Skip image: {p} ({e})")

        if not imgs_list:
            continue

        imgs = torch.stack(imgs_list).to(device)
        feats = model.inference(imgs)
        all_feats.append(feats.cpu())

    if not all_feats:
        raise ValueError("No valid images to extract features.")
    return torch.cat(all_feats, dim=0)


def evaluate(query_feats, gallery_feats, query_info, gallery_info, max_rank=50):
    """
    Compute mAP and CMC for person re-identification.

    Protocol:
    - Sort gallery by distance for each query
    - Positive: same pid, different cam
    - Junk: same pid, same cam (ignored)
    """
    num_q = query_feats.shape[0]
    distmat = 1 - torch.mm(query_feats, gallery_feats.t())
    distmat = distmat.cpu().numpy()

    q_pids = np.array([x["pid"] for x in query_info])
    g_pids = np.array([x["pid"] for x in gallery_info])
    q_cams = np.array([x["cam"] for x in query_info])
    g_cams = np.array([x["cam"] for x in gallery_info])

    indices = np.argsort(distmat, axis=1)

    matches = np.zeros_like(indices, dtype=np.int32)
    for i in range(num_q):
        for j, gal_idx in enumerate(indices[i]):
            if g_pids[gal_idx] == q_pids[i]:
                if g_cams[gal_idx] != q_cams[i]:
                    matches[i, j] = 1
                else:
                    matches[i, j] = -1

    cmc = np.zeros(max_rank, dtype=np.float32)
    num_valid_q = 0

    for i in range(num_q):
        valid_indices = np.where(matches[i] != -1)[0]
        if len(valid_indices) == 0:
            continue
        if not np.any(matches[i][valid_indices] == 1):
            continue

        num_valid_q += 1
        first_match_idx = -1
        for rank, idx in enumerate(valid_indices):
            if matches[i, idx] == 1:
                first_match_idx = rank
                break

        if 0 <= first_match_idx < max_rank:
            cmc[first_match_idx:] += 1

    if num_valid_q > 0:
        cmc = cmc / num_valid_q * 100.0
    rank1 = cmc[0] if max_rank > 0 else 0.0

    ap_sum = 0.0
    num_valid_q_map = 0

    for i in range(num_q):
        if not np.any(matches[i] != -1):
            continue
        if not np.any(matches[i] == 1):
            continue

        num_valid_q_map += 1
        num_gt = np.sum(matches[i] == 1)
        num_correct = 0
        ap = 0.0

        for rank in range(len(matches[i])):
            if matches[i, rank] == -1:
                continue
            if matches[i, rank] == 1:
                num_correct += 1
                num_retrieved = np.sum(matches[i, : rank + 1] != -1)
                ap += num_correct / num_retrieved

        ap /= num_gt
        ap_sum += ap

    mAP = (ap_sum / num_valid_q_map * 100.0) if num_valid_q_map > 0 else 0.0
    return mAP, rank1, cmc


def debug_evaluation_setup(query_dir, gallery_dir, id_from="folder"):
    """
    Print a short summary to verify query/gallery structure and pid overlap.
    """
    query_data = collect_images_with_subfolder(query_dir, id_from=id_from)
    gallery_data = collect_images_with_subfolder(gallery_dir, id_from=id_from)

    print("Query images:", len(query_data))
    print("Gallery images:", len(gallery_data))

    query_pids = set(d[1] for d in query_data)
    gallery_pids = set(d[1] for d in gallery_data)
    overlap = query_pids & gallery_pids
    print("Query unique pids:", len(query_pids))
    print("Gallery unique pids:", len(gallery_pids))
    print("PID overlap:", len(overlap))