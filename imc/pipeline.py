"""End-to-end: one dataset folder -> submission rows."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .data import clusters_to_rows
from .features import FeatureConfig, run_stage


def run_dataset(dataset: str, image_dir: Path, image_names: list[str], workdir: Path,
                cfg: FeatureConfig | None = None, min_num_matches: int = 200,
                reuse_features: bool = False) -> tuple[list[dict], dict]:
    cfg = cfg or FeatureConfig()
    workdir = Path(workdir) / dataset
    t0 = time.time()
    cached = workdir / "images.txt"
    if reuse_features and cached.exists() and cached.read_text().split("\n") == list(image_names):
        stats = {"features": "cached"}
    else:
        stats = run_stage(Path(image_dir), image_names, workdir, cfg)
    # SfM in a child process: keeps torch and pycolmap (two libomp copies on macOS) apart,
    # and a COLMAP crash on one dataset cannot kill the whole submission.
    proc = subprocess.run([sys.executable, "-m", "imc.sfm", str(workdir), str(image_dir),
                           "--min-num-matches", str(min_num_matches)],
                          cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
    clusters = []
    if proc.returncode == 0:
        res = json.loads((workdir / "clusters.json").read_text())
        stats.update({k: v for k, v in res.items() if k != "clusters"})
        clusters = [{n: (np.array(p["R"]), np.array(p["t"])) for n, p in c.items()} for c in res["clusters"]]
    else:
        stats["sfm_error"] = proc.stderr[-2000:]
    stats["n_clusters"] = len(clusters)
    stats["t_total"] = time.time() - t0
    return clusters_to_rows(dataset, image_names, clusters), stats
