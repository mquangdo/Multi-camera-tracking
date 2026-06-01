import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np


def load_run_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    
def load_map_config(config_path="config_map.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minimap_path = cfg["minimap"]
    
    results_path = cfg["results"]["jsonl"]

    cam1_src_pts = np.array(cfg["cam1"]["src_pts"], dtype=np.float32)
    cam1_dst_pts = np.array(cfg["cam1"]["dst_pts"], dtype=np.float32)
    cam2_src_pts = np.array(cfg["cam2"]["src_pts"], dtype=np.float32)
    cam2_dst_pts = np.array(cfg["cam2"]["dst_pts"], dtype=np.float32)

    return minimap_path, results_path, cam1_src_pts, cam1_dst_pts, cam2_src_pts, cam2_dst_pts