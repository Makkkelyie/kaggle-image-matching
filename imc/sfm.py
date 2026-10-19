"""Stage 2 (pycolmap): geometric verification + incremental SfM -> clusters.

Run as ``python -m imc.sfm <workdir> <image_dir>``; it reads the cache written
by ``imc.features`` and writes ``clusters.json``. Kept torch-free on purpose
(see the libomp note in features.py).

Every reconstruction COLMAP returns becomes one scene cluster; images that are
not registered anywhere are labelled outliers. This is how the IMC 2025 host
baseline and most top teams did the clustering.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap


def build_database(db_path: Path, image_dir: Path, image_names: list[str],
                   kpts: dict, matches: dict) -> Path:
    if db_path.exists():
        db_path.unlink()
    pycolmap.Database.open(str(db_path)).close()  # creates the empty schema
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = "SIMPLE_RADIAL"
    pycolmap.import_images(str(db_path), str(image_dir), pycolmap.CameraMode.PER_IMAGE, image_names, opts)

    db = pycolmap.Database.open(str(db_path))
    name_to_id = {img.name: img.image_id for img in db.read_all_images()}
    for i, name in enumerate(image_names):
        # COLMAP's pixel convention puts the centre of the top-left pixel at (0.5, 0.5).
        db.write_keypoints(name_to_id[name], (kpts[str(i)] + 0.5).astype(np.float32))
    pair_lines = []
    for key, m in matches.items():
        i, j = map(int, key.split("_"))
        db.write_matches(name_to_id[image_names[i]], name_to_id[image_names[j]], m.astype(np.uint32))
        pair_lines.append(f"{image_names[i]} {image_names[j]}")
    db.close()
    pairs_path = db_path.with_name("pairs.txt")
    pairs_path.write_text("\n".join(pair_lines))
    return pairs_path


def reconstruct(workdir: Path, image_dir: Path, min_model_size: int = 3, min_num_matches: int = 200,
                num_threads: int = -1) -> dict:
    image_names = (workdir / "images.txt").read_text().split("\n")
    kpts = dict(np.load(workdir / "keypoints.npz"))
    matches = dict(np.load(workdir / "matches.npz"))
    db_path = workdir / "colmap.db"
    t0 = time.time()
    pairs_path = build_database(db_path, image_dir, image_names, kpts, matches)

    ver = pycolmap.TwoViewGeometryOptions()
    ver.ransac.max_num_trials = 10000
    pycolmap.verify_matches(str(db_path), str(pairs_path), ver)
    t1 = time.time()

    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = min_model_size
    # Verified inliers needed before COLMAP uses an image pair. The default (15) lets a few
    # chance matches glue look-alike scenes (St Peter's + St Paul's) into one model; 200
    # keeps them apart (local clustering score 0.43-0.5 -> 1.0 on ETs / pt_stpeters_stpauls).
    opts.min_num_matches = min_num_matches
    opts.num_threads = num_threads
    opts.random_seed = 0  # reproducible runs (COLMAP's RANSAC otherwise varies run to run)
    opts.mapper.init_min_num_inliers = 50  # LightGlue gives fewer, cleaner matches than SIFT
    out_dir = workdir / "sparse"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()
    recs = pycolmap.incremental_mapping(str(db_path), str(image_dir), str(out_dir), opts)
    t2 = time.time()

    clusters = []
    for rec in sorted(recs.values(), key=lambda r: r.num_reg_images(), reverse=True):
        poses = {}
        for img in rec.images.values():
            if not img.has_pose:
                continue
            pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            poses[img.name] = {"R": pose.rotation.matrix().tolist(), "t": np.asarray(pose.translation).tolist()}
        clusters.append(poses)
    result = {"clusters": clusters, "t_verify": t1 - t0, "t_mapping": t2 - t1,
              "n_registered": sum(len(c) for c in clusters)}
    (workdir / "clusters.json").write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", type=Path)
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("--min-model-size", type=int, default=3)
    ap.add_argument("--min-num-matches", type=int, default=200)
    a = ap.parse_args()
    reconstruct(a.workdir, a.image_dir, a.min_model_size, a.min_num_matches)
