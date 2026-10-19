"""Local re-implementation of the IMC 2025 score (mAA x clustering, harmonic mean).

Follows the evaluation section of the competition page:

1. For every ground-truth scene S, greedily pick the user cluster C that
   maximises mAA (ties broken by the clustering score |S n C| / |C|).
   Images the user labelled ``outliers`` never take part in this assignment.
2. Per dataset, aggregate:  clustering = sum|S n C| / sum|C|,
   mAA = sum(accurate cameras) / sum(scene cameras), and the dataset score is
   their harmonic mean (an F1 between "recall" mAA and "precision" clustering).
3. The final score is the mean over datasets.

mAA follows the IMC 2024 recipe: predicted camera centres are aligned to the
ground truth with a similarity transform estimated from camera triplets
(Horn/Umeyama), and a camera counts as accurate if its centre lies within the
scene's distance threshold; the score is averaged over all thresholds.  The 3
cameras used for the minimal alignment are subtracted (``to_dec=3``) so that a
cluster with three random poses does not score for free.

This is NOT the organisers' code: the triplet search is capped at
``max_triplets`` random triplets for big scenes, so numbers can differ slightly
from the leaderboard. It is meant for relative comparisons on training data.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .geometry import apply_similarity, camera_center, parse_matrix, umeyama_similarity

OUTLIER_LABEL = "outliers"


def count_accurate_cameras(pred_c: np.ndarray, gt_c: np.ndarray, thresholds, max_triplets: int = 2000,
                           refine_iters: int = 10, seed: int = 0) -> np.ndarray:
    """For each threshold, the max number of cameras whose aligned centre is within it.

    pred_c, gt_c: (N, 3) corresponding camera centres. Returns int array (len(thresholds),).
    """
    thresholds = np.asarray(thresholds, float)
    n = len(pred_c)
    best = np.zeros(len(thresholds), dtype=int)
    if n < 3:
        return best

    n_comb = n * (n - 1) * (n - 2) // 6
    if n_comb <= max_triplets:
        triplets = np.array(list(combinations(range(n), 3)))
    else:
        rng = np.random.default_rng(seed)
        triplets = np.array([rng.choice(n, 3, replace=False) for _ in range(max_triplets)])

    best_inliers = [None] * len(thresholds)
    for tri in triplets:
        s, R, t = umeyama_similarity(pred_c[tri], gt_c[tri])
        if not np.isfinite(s) or s <= 0:
            continue
        err = np.linalg.norm(apply_similarity(s, R, t, pred_c) - gt_c, axis=1)
        inl = err[None, :] < thresholds[:, None]  # (T, N)
        cnt = inl.sum(1)
        better = cnt > best
        for k in np.flatnonzero(better):
            best[k], best_inliers[k] = cnt[k], inl[k]

    # Least-squares refinement on the inlier set (like a final LO-RANSAC step).
    for k, th in enumerate(thresholds):
        inl = best_inliers[k]
        for _ in range(refine_iters):
            if inl is None or inl.sum() < 3:
                break
            s, R, t = umeyama_similarity(pred_c[inl], gt_c[inl])
            new = np.linalg.norm(apply_similarity(s, R, t, pred_c) - gt_c, axis=1) < th
            if new.sum() <= inl.sum():
                break
            inl = new
        if inl is not None:
            best[k] = max(best[k], int(inl.sum()))
    return best


@dataclass
class SceneMatch:
    scene: str
    cluster: str | None
    maa_num: float  # mean over thresholds of max(accurate - 3, 0)
    maa_den: float  # |S| - 3
    inter: int      # |S n C|
    cluster_size: int  # |C|


def _centres(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for img, r, t in zip(df["image"], df["rotation_matrix"], df["translation_vector"]):
        R, tv = parse_matrix(r, (3, 3)), parse_matrix(t, (3,))
        if np.all(np.isfinite(R)) and np.all(np.isfinite(tv)):
            out[img] = camera_center(R, tv)
    return out


def score_dataset(gt: pd.DataFrame, pred: pd.DataFrame, thresholds: dict[str, list[float]],
                  max_triplets: int = 2000, to_dec: int = 3) -> tuple[float, float, float, list[SceneMatch]]:
    """Score one dataset. gt/pred have columns image, scene, rotation_matrix, translation_vector."""
    gt = gt[gt["scene"] != OUTLIER_LABEL]
    pred = pred[pred["scene"] != OUTLIER_LABEL]
    pred_centres = _centres(pred)
    clusters = {c: set(g["image"]) for c, g in pred.groupby("scene")}

    matches = []
    for scene, g in gt.groupby("scene"):
        scene_imgs = set(g["image"])
        gt_centres = _centres(g)
        ths = thresholds[scene]
        den = max(len(scene_imgs) - to_dec, 1)
        best = SceneMatch(scene, None, 0.0, den, 0, 0)
        best_key = (-1.0, -1.0)
        for c, members in clusters.items():
            inter = scene_imgs & members
            if not inter:
                continue
            common = sorted(i for i in inter if i in pred_centres and i in gt_centres)
            num = 0.0
            if len(common) >= 3:
                cnt = count_accurate_cameras(np.array([pred_centres[i] for i in common]),
                                             np.array([gt_centres[i] for i in common]), ths, max_triplets)
                num = float(np.mean(np.maximum(cnt - to_dec, 0)))
            key = (num / den, len(inter) / len(members))
            if key > best_key:
                best_key = key
                best = SceneMatch(scene, c, num, den, len(inter), len(members))
        matches.append(best)

    maa = sum(m.maa_num for m in matches) / sum(m.maa_den for m in matches)
    csize = sum(m.cluster_size for m in matches)
    clus = sum(m.inter for m in matches) / csize if csize else 0.0
    f = 2 * maa * clus / (maa + clus) if (maa + clus) > 0 else 0.0
    return f, maa, clus, matches


def score(gt: pd.DataFrame, pred: pd.DataFrame, thresholds_df: pd.DataFrame, datasets=None,
          max_triplets: int = 2000) -> tuple[float, pd.DataFrame]:
    """Competition-style score. Returns (final_score, per-dataset table)."""
    ths = {(d, s): [float(v) for v in str(t).split(";")]
           for d, s, t in zip(thresholds_df["dataset"], thresholds_df["scene"], thresholds_df["thresholds"])}
    datasets = sorted(gt["dataset"].unique()) if datasets is None else datasets
    rows = []
    for d in datasets:
        g, p = gt[gt["dataset"] == d], pred[pred["dataset"] == d]
        scene_ths = {s: ths[(d, s)] for s in g["scene"].unique() if s != OUTLIER_LABEL}
        f, maa, clus, _ = score_dataset(g, p, scene_ths, max_triplets)
        rows.append({"dataset": d, "score": f, "mAA": maa, "clustering": clus,
                     "n_images": len(g), "n_scenes": len(scene_ths)})
    table = pd.DataFrame(rows)
    return float(table["score"].mean()), table
