"""Reading the competition files and writing submission.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import format_matrix

NAN_R = ";".join(["nan"] * 9)
NAN_T = ";".join(["nan"] * 3)


def load_labels(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "train_labels.csv")


def load_thresholds(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "train_thresholds.csv")


def test_datasets(data_dir: Path) -> dict[str, list[str]]:
    """dataset -> image names, read from sample_submission.csv (what Kaggle gives at scoring time)."""
    sub = pd.read_csv(Path(data_dir) / "sample_submission.csv")
    return {d: sorted(g["image"]) for d, g in sub.groupby("dataset")}


def subsample_dataset(labels: pd.DataFrame, dataset: str, max_per_scene: int, seed: int = 0) -> pd.DataFrame:
    """Keep at most ``max_per_scene`` images of each scene (and of the outliers) for a quick local run."""
    rng = np.random.default_rng(seed)
    g = labels[labels["dataset"] == dataset]
    parts = []
    for _, s in g.groupby("scene"):
        idx = rng.choice(len(s), size=min(len(s), max_per_scene), replace=False)
        parts.append(s.iloc[np.sort(idx)])
    return pd.concat(parts).sort_values("image").reset_index(drop=True)


def clusters_to_rows(dataset: str, image_names: list[str], clusters: list[dict]) -> list[dict]:
    """clusters: list of {image_name: (R 3x3, t 3)}; images in none of them become outliers."""
    rows, seen = [], set()
    for k, poses in enumerate(clusters):
        for name, (R, t) in poses.items():
            if name in seen:
                continue
            seen.add(name)
            rows.append(dict(dataset=dataset, scene=f"cluster{k}", image=name,
                             rotation_matrix=format_matrix(R), translation_vector=format_matrix(t)))
    for name in image_names:
        if name not in seen:
            rows.append(dict(dataset=dataset, scene="outliers", image=name,
                             rotation_matrix=NAN_R, translation_vector=NAN_T))
    return rows


def write_submission(rows: list[dict], path: Path, sample_submission: Path | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["dataset", "scene", "image", "rotation_matrix", "translation_vector"])
    if sample_submission is not None:  # keep Kaggle's image_id column and row order
        ss = pd.read_csv(sample_submission)[["image_id", "dataset", "image"]]
        df = ss.merge(df, on=["dataset", "image"], how="left")
        df["scene"] = df["scene"].fillna("outliers")
        df["rotation_matrix"] = df["rotation_matrix"].fillna(NAN_R)
        df["translation_vector"] = df["translation_vector"].fillna(NAN_T)
        df = df[["image_id", "dataset", "scene", "image", "rotation_matrix", "translation_vector"]]
    df.to_csv(path, index=False)
    return df
