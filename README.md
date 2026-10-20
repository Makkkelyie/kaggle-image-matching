# Image Matching Challenge 2025: scene clustering + camera pose estimation

My entry for the Kaggle [Image Matching Challenge 2025 (ongoing)](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing),
the open benchmark that the CVPR Image Matching workshop uses.

**The task:** you get a folder of photos that mixes several places (for example two sides of the same
building) plus unrelated "outlier" shots. The model has to (1) work out which photos show the same scene
and (2) recover the 3D position and orientation (rotation `R`, translation `t`) of the camera for every
photo, i.e. a sparse 3D reconstruction (Structure-from-Motion).

**What this repo shows:** multi-view 3D geometry and coordinate frames, camera pose estimation,
pretrained PyTorch vision models, and a reproducible pipeline: tested geometry code, a local copy of the
competition metric, and a Kaggle notebook generated from the same package.

## Pipeline

```
images ──► DINOv2 global descriptor ──► pair shortlist (top-k + similarity threshold)
                                              │
          ALIKED keypoints ◄──────────────────┘
                 │
          LightGlue matching ──► COLMAP geometric verification (RANSAC, F/E/H)
                                              │
                          incremental SfM (pycolmap) ──► 1 reconstruction = 1 scene cluster
                                                         unregistered image = outlier
```

| Stage | Component | What it does and why |
|---|---|---|
| Retrieval | [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 (`imc/features.py`) | One embedding per image (CLS token + GeM-pooled patch tokens). Only image pairs that look alike get matched, so cost grows ~linearly instead of O(n²) |
| Local features | [ALIKED](https://github.com/Shiaoming/ALIKED) | Up to 4096 keypoints + descriptors per image at 1024 px. BSD-licensed, which the competition rules need (SuperPoint's licence is non-commercial) |
| Matching | [LightGlue](https://github.com/cvg/LightGlue) | Transformer matcher that uses the context of both images; pairs with < 15 matches are dropped |
| Verification | COLMAP `verify_matches` | RANSAC two-view geometry removes matches that break epipolar constraints |
| SfM | [pycolmap](https://github.com/colmap/colmap) incremental mapper (`imc/sfm.py`) | Registers images one by one, triangulates 3D points, bundle adjustment. COLMAP's multiple models become the clusters |
| Metric | `imc/metric.py` | Re-implementation of the competition score (see below) for local validation |

Engineering notes:
- **Two-process design.** The PyTorch stage writes keypoints/matches to `.npz`, and the SfM stage runs as
  `python -m imc.sfm` in a child process. On macOS, torch and pycolmap ship two copies of `libomp` that
  abort the process when loaded together. Splitting them also means a COLMAP crash on one dataset cannot
  take down a 9-hour Kaggle submission (that dataset falls back to "all outliers").
- **Pose conventions** are COLMAP's `cam_from_world` (`x_cam = R X + t`), which is also what the
  submission CSV expects. Camera centres are `C = -Rᵀt`. Keypoints get +0.5 px to match COLMAP's pixel
  convention.
- **Offline Kaggle run.** Kaggle code competitions run with internet off, so `scripts/build_kaggle_deps.sh`
  packs Linux wheels (pycolmap, LightGlue, kornia), model weights and the `imc` package into one dataset.
  `scripts/build_notebook.py` generates the submission notebook from the same code.

## The metric (from the competition evaluation page)

For each dataset:
1. Each ground-truth scene is matched to the predicted cluster that gives it the best **mAA**.
2. **mAA** (≈ recall): predicted camera centres are aligned to ground truth with a similarity transform
   (Horn/Umeyama on camera triplets, RANSAC-style). A camera counts as accurate if its centre is within the
   scene's distance threshold, averaged over several thresholds.
3. **Clustering score** (≈ precision): |scene ∩ cluster| / |cluster|.
4. Dataset score = harmonic mean of the two (like F1). Final score = mean over datasets.

`imc/metric.py` is my own implementation from that description, not the organisers' code (big scenes use
2,000 random triplets rather than all of them), so treat local numbers as relative, not leaderboard-exact.
`tests/test_metric.py` checks the edge cases: perfect = 1, invariance to the arbitrary SfM world frame,
merged scenes hurt only clustering, unregistered images hurt only mAA.

## Results

Local validation on 4 training datasets, max 30 images per scene (189 images, 8 scenes plus outliers),
run on an M-series MacBook (LightGlue on MPS, ALIKED on CPU, 2048 keypoints). The scores come from my
re-implemented metric, so compare them with each other, not with the leaderboard.

**Effect of the SfM pair threshold** (`min_num_matches`: verified inliers COLMAP needs before it uses a pair)

| Setting | ETs | church + theater | St Peter's + St Paul's | stairs | **Mean** |
|---|---|---|---|---|---|
| COLMAP default (15), run 1 | 0.301 | 0.544 | 0.508 | 0.036 | **0.347** |
| COLMAP default (15), run 2 | 0.278 | 0.526 | 0.495 | 0.022 | **0.330** |
| 50 | 0.361 | 0.450 | 0.460 | 0.000 | 0.318 |
| 100 | 0.440 | 0.507 | 0.491 | 0.029 | 0.367 |
| **200 (chosen)**, run 1 | 0.424 | 0.458 | 0.647 | 0.015 | **0.386** |
| **200 (chosen)**, run 2 | 0.424 | 0.469 | 0.644 | 0.029 | **0.392** |
| 400 | 0.340 | 0.295 | 0.644 | 0.000 | 0.320 |

Per-dataset breakdown for the chosen setting (`results/seed_minmatch200/scores.csv`):

| Dataset | mAA | Clustering | Score |
|---|---|---|---|
| ETs (two near-identical figurines + outliers) | 0.269 | 1.000 | 0.424 |
| imc2023_theather_imc2024_church | 0.307 | 1.000 | 0.469 |
| pt_stpeters_stpauls | 0.475 | 1.000 | 0.644 |
| stairs | 0.015 | 0.647 | 0.029 |

**Harder datasets and two follow-up changes (Oct 2026).** After the leaderboard came in at 19.46, I added two
harder training datasets (`fbk_vineyard`: three rows of near-identical vines; `imc2023_haiper`: bike, chairs and a
fountain shot in the same yard) and re-ran the threshold sweep on all six, reusing the cached matches:

| `min_num_matches` | 4 original (mean) | fbk_vineyard | imc2023_haiper | **6-dataset mean** |
|---|---|---|---|---|
| 15 | 0.330–0.347 | 0.155 | 0.499 | – |
| 50 | 0.329 | 0.147 | 0.496 | 0.326 |
| 100 | 0.360 | 0.215 | 0.462 | 0.353 |
| **200** | 0.395 | 0.289 | 0.400 | **0.378** |
| 400 | 0.320 | 0.163 | 0.144 | – |
| **200 + leftover pass at 50 (new default)** | 0.397 | 0.298 | 0.397 | **0.381** |

- **200 still wins on average.** Looser thresholds help haiper (bike and chairs get more cameras) but glue the
  vineyard rows and St Peter's / St Paul's together, which costs more.
- **Leftover pass** (`imc/sfm.py`): images the strict pass leaves unregistered get a second incremental-mapping run
  at 50 inliers, restricted to those images so they cannot merge into the strict models. Locally it is neutral
  (+0.003, inside the ±0.01 run-to-run noise). I kept it because it is a safety net for hidden datasets where a whole
  scene falls below 200 inliers and would otherwise score 0.
- **4096 keypoints** (ETs + stairs only, since it took 8× longer on the laptop): ETs 0.440 → 0.485, stairs clustering
  0.84 → 1.00 (its mAA stays ≈0). The Kaggle notebook now uses 4096 when a GPU is present; laptop runs stay at 2048.
  The notebook's 8-hour budget check marks any remaining dataset as outliers if it runs long.
- Errors that remain: haiper's bike and chairs share one model (clustering 0.72) because the background is the same
  yard, and every vineyard row splits into 2–4 small models (mAA 0.17), so only the largest piece counts.

| Version | Change | Local (6 datasets) | Leaderboard |
|---|---|---|---|
| 3 | 200 inliers, 2048 keypoints | 0.378 | 19.46 |
| 4 | + leftover pass, 4096 keypoints on GPU | 0.381 (2048 kp) | **27.72** |

The leaderboard went up much more (+8.3) than the local score (+0.003). Two reasons are likely: local runs used
2048 keypoints, so the 4096 gain isn't counted there, and the hidden test probably has scenes that fall entirely
below 200 inliers, which the leftover pass now reconstructs instead of scoring 0. Both changes went into one
submission, so I can't split the +8.3 between them.

What I learned from the errors (`results/*/submission.csv` vs ground truth):
- **Look-alike scenes get glued together.** With COLMAP's default of 15 inliers, a handful of chance matches
  merged St Peter's Square and St Paul's Cathedral into one model (clustering 0.5). Requiring 200 inliers
  splits them (clustering 1.0) and lifts the mean score by about 0.05. Too strict (400) breaks real scenes apart.
- **Symmetric buildings flip.** The church is nearly symmetric. About half its cameras are placed on the
  wrong side (relative rotation off by ~176°), so mAA stays at ~0.3 even though every image is in the right
  cluster. This is the "doppelgänger" problem in SfM.
- **stairs** (repetitive steps, narrow baseline) barely registers. Repetitive texture needs better matching.
- COLMAP is multi-threaded, so runs are not bit-identical even with a fixed seed. Expect about ±0.01 on the mean.
- Speed: matching is the bottleneck (~0.4–0.8 s per pair on the laptop). Global descriptors pruned only 0–9% of pairs
  at a 0.5 similarity threshold. This suits recall (top teams also used loose pairing) but costs time.

Kaggle: the notebook runs end to end on a Kaggle GPU (internet off). On the public example test set it took
2.8 min, clustered the ETs dataset perfectly (10 / 9 images, 3 outliers) and wrote a valid `submission.csv`.
On GPU, LightGlue takes ~0.065 s per pair, so the ~1,300 hidden test images should fit well inside the 9-hour limit.
**Leaderboard score: 27.72** (Version 4; Version 3 scored 19.46). Both were submitted in Oct 2026 after the board
froze, so they aren't ranked. For scale, the frozen board's 1st place is 65.56 and 10th is 42.58. The hidden test
still scores lower than my local subset (≈0.38 on a 0–1 scale, i.e. ≈38), probably because it includes harder
datasets than the six I validated on. Version 4 takes 7.5 min on the public example test (4096 keypoints on GPU,
~0.17 s per pair). The "Next steps" below are where I'd expect the gains.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                      # geometry + metric unit tests
# competition data (needs a Kaggle token and accepted rules)
.venv/bin/python scripts/download_subset.py --list
.venv/bin/python scripts/download_subset.py --datasets ETs imc2023_theather_imc2024_church pt_stpeters_stpauls stairs fbk_vineyard imc2023_haiper   # ~645 MB
.venv/bin/python scripts/run_local_eval.py --datasets ETs imc2023_theather_imc2024_church pt_stpeters_stpauls stairs --max-per-scene 30 --tag mine
# re-run only the SfM stage on cached matches (about 1 min instead of ~40)
.venv/bin/python scripts/run_local_eval.py --datasets ETs stairs --max-per-scene 30 --reuse-features results/mine --min-num-matches 100 --tag mine_100
```

Kaggle submission:
```bash
bash scripts/build_kaggle_deps.sh && kaggle datasets create -p kaggle_deps   # later: kaggle datasets version -p kaggle_deps -m "..."
python scripts/build_notebook.py <kaggle-username> && kaggle kernels push -p notebooks
# then press "Submit" on the notebook's page (internet is off, GPU on)
```

## Repo layout

```
imc/geometry.py   pose maths: camera centre, compose/invert/relative pose, rotation & translation angle errors, Umeyama
imc/metric.py     competition score (mAA + clustering, harmonic mean)
imc/features.py   DINOv2 retrieval, ALIKED + LightGlue matching (PyTorch)
imc/sfm.py        COLMAP database, geometric verification, incremental mapping (pycolmap)
imc/pipeline.py   one dataset -> submission rows
imc/data.py       CSV I/O, subsampling, submission writer
scripts/          data download, local evaluation, Kaggle packaging, notebook generator
tests/            pytest unit tests
```

## Credits

This builds on public work. I did not invent the method:
- IMC 2025 host baseline notebook (DINOv2 shortlist → ALIKED + LightGlue → COLMAP, reconstructions as clusters);
  most top teams kept this skeleton.
- Bellavia et al., *Image Matching Challenge 2024: Methods & Solutions Review*, [arXiv:2407.03172](https://arxiv.org/abs/2407.03172),
  for the metric details (mAA on camera centres) and common tricks.
- Community summaries of the IMC 2025 top solutions (MASt3R-based matching and retrieval, image splitting/cropping,
  looser pairing thresholds), listed under "Next steps".
- LightGlue (Lindenberger et al., ICCV 2023), ALIKED (Zhao et al., 2023), DINOv2 (Oquab et al., 2023),
  COLMAP (Schönberger & Frahm, CVPR 2016).

## Next steps

Ideas from the IMC 2025 top solutions that I have not done yet:
- Stronger retrieval for clustering: MASt3R-ASMK instead of DINOv2 (used by the 1st and 5th place teams).
- Rotation-aware matching (upright vs rotated photos) and keypoint-density crops for high-res images.
- Merging or splitting COLMAP models with a match-graph check before writing clusters (would fix the vineyard
  rows that split into several small models).
- Doppelgänger filtering for symmetric buildings (drop pairs whose matches cover only a small, symmetric part of the image).
- A larger retrieval cut (stricter similarity threshold) once matching runs on a GPU, to trade a little recall for a lot of speed.
