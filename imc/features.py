"""Stage 1 (PyTorch): image retrieval, local features and matching.

* DINOv2 global descriptors -> shortlist of image pairs worth matching.
* ALIKED keypoints + LightGlue matcher on every shortlisted pair.

Results are written to an .npz-per-stage cache in ``workdir`` so the SfM stage
(pycolmap, separate process) never needs to import torch.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# lightglue/sift.py imports pycolmap if present; on macOS pycolmap and torch ship
# two copies of libomp that abort when loaded together. We never use SIFT here,
# so hide pycolmap from this process.
sys.modules.setdefault("pycolmap", None)
from lightglue import ALIKED, LightGlue  # noqa: E402
from lightglue.utils import load_image, rbd  # noqa: E402


@dataclass
class FeatureConfig:
    device: str = "auto"
    dino_model: str = "facebook/dinov2-base"  # path to a local copy on Kaggle
    dino_size: int = 448
    exhaustive_below: int = 20   # match all pairs if the dataset is this small
    topk: int = 20               # neighbours per image from global descriptors
    sim_threshold: float = 0.5   # ... plus every pair above this cosine similarity
    max_keypoints: int = 4096
    detection_threshold: float = 0.01
    resize: int = 1024
    min_matches: int = 15


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- retrieval
@torch.inference_mode()
def global_descriptors(paths: list[Path], cfg: FeatureConfig, device: torch.device) -> torch.Tensor:
    """L2-normalised DINOv2 descriptor per image (CLS token concat GeM of patch tokens)."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(cfg.dino_model).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    descs = []
    for p in paths:
        img = load_image(p).to(device)  # 3xHxW in [0, 1]
        img = F.interpolate(img[None], size=(cfg.dino_size, cfg.dino_size), mode="bilinear", align_corners=False)[0]
        out = model(pixel_values=((img - mean) / std)[None]).last_hidden_state[0]
        cls, patches = out[0], out[1:]
        gem = patches.clamp(min=1e-6).pow(3).mean(0).pow(1 / 3)
        d = torch.cat([F.normalize(cls, dim=0), F.normalize(gem, dim=0)])
        descs.append(F.normalize(d, dim=0).cpu())
    del model
    return torch.stack(descs)


def shortlist_pairs(desc: torch.Tensor, cfg: FeatureConfig) -> list[tuple[int, int]]:
    """Top-k neighbours per image plus all pairs above the similarity threshold."""
    n = len(desc)
    if n <= cfg.exhaustive_below:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    sim = (desc @ desc.T).numpy()
    np.fill_diagonal(sim, -1)
    pairs = set()
    k = min(cfg.topk, n - 1)
    for i in range(n):
        for j in np.argsort(-sim[i])[:k]:
            pairs.add((min(i, j), max(i, j)))
    for i, j in zip(*np.nonzero(sim > cfg.sim_threshold)):
        if i < j:
            pairs.add((int(i), int(j)))
    return sorted(pairs)


# ---------------------------------------------------------------- matching
@torch.inference_mode()
def extract_and_match(paths: list[Path], pairs: list[tuple[int, int]], cfg: FeatureConfig,
                      device: torch.device):
    """Returns (keypoints per image, {(i, j): (M, 2) index array}, image sizes)."""
    # ALIKED uses deformable convolutions, which MPS does not implement.
    feat_dev = torch.device("cpu") if device.type == "mps" else device
    extractor = ALIKED(max_num_keypoints=cfg.max_keypoints, detection_threshold=cfg.detection_threshold).eval().to(feat_dev)
    matcher = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval().to(device)

    feats, kpts, sizes = [], [], []
    for p in paths:
        img = load_image(p)
        f = extractor.extract(img.to(feat_dev), resize=cfg.resize)
        feats.append({k: v.to(device) for k, v in f.items()})
        kpts.append(f["keypoints"][0].cpu().numpy())
        sizes.append(tuple(img.shape[-2:][::-1]))  # (W, H)

    matches = {}
    for i, j in pairs:
        out = rbd(matcher({"image0": feats[i], "image1": feats[j]}))
        m = out["matches"].cpu().numpy()
        if len(m) >= cfg.min_matches:
            matches[(i, j)] = m.astype(np.uint32)
    return kpts, matches, sizes


def run_stage(image_dir: Path, image_names: list[str], workdir: Path, cfg: FeatureConfig) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    paths = [image_dir / n for n in image_names]
    t0 = time.time()
    desc = global_descriptors(paths, cfg, device)
    pairs = shortlist_pairs(desc, cfg)
    t1 = time.time()
    kpts, matches, sizes = extract_and_match(paths, pairs, cfg, device)
    t2 = time.time()

    np.savez(workdir / "keypoints.npz", **{str(i): k for i, k in enumerate(kpts)})
    np.savez(workdir / "matches.npz", **{f"{i}_{j}": m for (i, j), m in matches.items()})
    (workdir / "images.txt").write_text("\n".join(image_names))
    return {"n_images": len(paths), "n_pairs": len(pairs), "n_matched_pairs": len(matches),
            "t_retrieval": t1 - t0, "t_matching": t2 - t1, "device": str(device)}
