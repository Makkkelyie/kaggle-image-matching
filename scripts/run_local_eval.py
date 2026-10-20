"""Validate the pipeline on a few training datasets and compute the competition score.

    python scripts/run_local_eval.py --data-dir data --datasets imc2024_lizard_pond --max-per-scene 40

Writes results/<tag>/{submission.csv, scores.csv, stats.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imc.data import load_labels, load_thresholds, subsample_dataset, write_submission  # noqa: E402
from imc.features import FeatureConfig  # noqa: E402
from imc.metric import score  # noqa: E402
from imc.pipeline import run_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--max-per-scene", type=int, default=40)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--topk", type=int, default=FeatureConfig.topk)
    ap.add_argument("--exhaustive-below", type=int, default=FeatureConfig.exhaustive_below)
    ap.add_argument("--max-keypoints", type=int, default=FeatureConfig.max_keypoints)
    ap.add_argument("--resize", type=int, default=FeatureConfig.resize)
    ap.add_argument("--min-num-matches", type=int, default=200, help="SfM: min verified inliers per pair")
    ap.add_argument("--fallback-min-num-matches", type=int, default=50,
                    help="SfM leftover pass on unregistered images with this looser threshold (0 = off)")
    ap.add_argument("--reuse-features", type=Path, default=None,
                    help="results/<tag> whose cached matches to reuse (re-runs only the SfM stage)")
    a = ap.parse_args()

    cfg = FeatureConfig(topk=a.topk, exhaustive_below=a.exhaustive_below,
                        max_keypoints=a.max_keypoints, resize=a.resize)
    labels, ths = load_labels(a.data_dir), load_thresholds(a.data_dir)
    out = Path("results") / a.tag
    out.mkdir(parents=True, exist_ok=True)

    rows, gts, stats = [], [], {}
    for d in a.datasets:
        gt = subsample_dataset(labels, d, a.max_per_scene)
        print(f"== {d}: {len(gt)} images, scenes: {gt['scene'].value_counts().to_dict()}", flush=True)
        work = (a.reuse_features or out) / "work"
        r, s = run_dataset(d, a.data_dir / "train" / d, list(gt["image"]), work, cfg,
                           a.min_num_matches, a.fallback_min_num_matches,
                           reuse_features=a.reuse_features is not None)
        print(json.dumps({k: v for k, v in s.items() if k != "sfm_error"}, default=str), flush=True)
        rows += r
        gts.append(gt)
        stats[d] = s

    pred = write_submission(rows, out / "submission.csv")
    final, table = score(pd.concat(gts), pred, ths, datasets=a.datasets)
    table.to_csv(out / "scores.csv", index=False)
    (out / "stats.json").write_text(json.dumps({"config": vars(cfg), "max_per_scene": a.max_per_scene,
                                                "min_num_matches": a.min_num_matches,
                                                "fallback_min_num_matches": a.fallback_min_num_matches,
                                                "final_score": final, "per_dataset": stats}, indent=1, default=str))
    print(table.to_string(index=False, float_format="%.3f"))
    print(f"FINAL (mean over datasets): {final:.4f}")


if __name__ == "__main__":
    main()
