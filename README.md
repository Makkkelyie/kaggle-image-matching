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

Local validation on a subset of the training data (RESULTS PENDING: see `results/`).

| Run | Datasets | Images | mAA | Clustering | Score |
|---|---|---|---|---|---|
| baseline | – | – | – | – | – |

Kaggle leaderboard (ongoing competition, single public+private board): pending.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                      # geometry + metric unit tests
# competition data (needs a Kaggle token and accepted rules)
.venv/bin/python scripts/download_subset.py --list
.venv/bin/python scripts/download_subset.py --datasets imc2024_lizard_pond
.venv/bin/python scripts/run_local_eval.py --datasets imc2024_lizard_pond --max-per-scene 40
```

Kaggle submission:
```bash
bash scripts/build_kaggle_deps.sh && kaggle datasets create -p kaggle_deps
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
- Merging or splitting COLMAP models with a match-graph check before writing clusters.
