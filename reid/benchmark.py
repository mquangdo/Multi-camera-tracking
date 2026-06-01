import torch

from inference import load_model_for_inference, build_inference_transform
from reid.metrics import (
    collect_images_with_subfolder,
    extract_features,
    evaluate,
    debug_evaluation_setup,
)
from configs.load_config import load_config


def run_benchmark(
    model_path,
    query_dir,
    gallery_dir,
    num_classes=702,
    img_h=256,
    img_w=128,
    batch_size=64,
    id_from="folder",
    max_rank=50,
    debug=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if debug:
        debug_evaluation_setup(query_dir, gallery_dir, id_from=id_from)

    model = load_model_for_inference(model_path, num_classes, device)
    transform = build_inference_transform(img_h, img_w)

    query_data = collect_images_with_subfolder(query_dir, id_from=id_from)
    gallery_data = collect_images_with_subfolder(gallery_dir, id_from=id_from)

    query_paths = [d[0] for d in query_data]
    gallery_paths = [d[0] for d in gallery_data]

    if len(query_paths) == 0 or len(gallery_paths) == 0:
        raise ValueError("No images found in query or gallery directory.")

    query_info = [{"pid": d[1], "cam": d[2]} for d in query_data]
    gallery_info = [{"pid": d[1], "cam": d[2]} for d in gallery_data]

    query_feats = extract_features(model, query_paths, transform, device, batch_size)
    gallery_feats = extract_features(model, gallery_paths, transform, device, batch_size)

    mAP, rank1, cmc = evaluate(
        query_feats, gallery_feats, query_info, gallery_info, max_rank=max_rank
    )

    print("mAP:", f"{mAP:.2f}")
    print("Rank-1:", f"{rank1:.2f}")
    if len(cmc) >= 5:
        print("Rank-5:", f"{cmc[4]:.2f}")
    if len(cmc) >= 10:
        print("Rank-10:", f"{cmc[9]:.2f}")

    return mAP, rank1, cmc


if __name__ == "__main__":
    cfg = load_config("configs/benchmark_config.yaml")
    b = cfg["benchmark"]

    run_benchmark(
        model_path=b["model_path"],
        query_dir=b["query_dir"],
        gallery_dir=b["gallery_dir"],
        num_classes=b["num_classes"],
        img_h=b["img_h"],
        img_w=b["img_w"],
        batch_size=b["batch_size"],
        id_from=b["id_from"],
        max_rank=b["max_rank"],
        debug=b.get("debug", False),
    )