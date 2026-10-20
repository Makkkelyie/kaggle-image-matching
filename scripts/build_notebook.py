"""Generate notebooks/imc25_submission.ipynb (+ kernel-metadata.json for `kaggle kernels push`).

The notebook is a thin driver: all logic lives in the `imc` package, which is
attached through the "imc25-offline-deps" dataset (see build_kaggle_deps.sh).
Keeping the notebook generated from code means the Kaggle entry and the local
validation always run the same pipeline.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = sys.argv[1] if len(sys.argv) > 1 else "YOUR_KAGGLE_USERNAME"

cells = [
    ("markdown", """# IMC 2025: DINOv2 retrieval + ALIKED/LightGlue + COLMAP

Groups each dataset's images into scenes and estimates camera poses.
Code and write-up: see the GitHub repo (`kaggle-image-matching`). Internet is off; wheels, weights
and the `imc` package come from the attached `imc25-offline-deps` dataset."""),
    ("code", """import glob, os, shutil, subprocess, sys
# Mount paths differ between Kaggle images, so find the attached inputs instead of hard-coding them.
DEPS = os.path.dirname(glob.glob('/kaggle/input/**/wheels/lightglue-*.whl', recursive=True)[0]).rsplit('/wheels', 1)[0]
print('deps:', DEPS)
print(sys.version)
for pkg in ['kornia_rs', 'kornia', 'lightglue', 'pycolmap']:  # one at a time so one miss can't block the rest
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                    '--find-links', f'{DEPS}/wheels', pkg], check=True)
# LightGlue/ALIKED load weights through the torch hub cache.
hub = os.path.expanduser('~/.cache/torch/hub/checkpoints'); os.makedirs(hub, exist_ok=True)
for f in glob.glob(f'{DEPS}/weights/*.pth'):
    shutil.copy(f, hub)
# Copy the package somewhere writable; the SfM stage runs as `python -m imc.sfm` from there.
shutil.copytree(f'{DEPS}/src/imc', '/kaggle/working/imc', dirs_exist_ok=True)
os.chdir('/kaggle/working'); sys.path.insert(0, '/kaggle/working')"""),
    ("code", """import time, traceback
from pathlib import Path
import pandas as pd
import torch
from imc.data import NAN_R, NAN_T, test_datasets, write_submission
from imc.features import FeatureConfig
from imc.pipeline import run_dataset

DATA = Path(sorted(glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True))[0]).parent
print('data:', DATA)
# 4096 keypoints on GPU: helped locally (ETs 0.44 -> 0.49, stairs clustering 0.84 -> 1.0) but is
# ~4-8x slower on a laptop, so the CPU/MPS default stays 2048.
cfg = FeatureConfig(dino_model=f'{DEPS}/weights/dinov2-base', max_keypoints=4096 if torch.cuda.is_available() else 2048)
print(torch.__version__, torch.cuda.is_available())
datasets = test_datasets(DATA)
print({d: len(v) for d, v in datasets.items()})"""),
    ("code", """BUDGET_S = 8 * 3600  # Kaggle limit is 9 h; leave room to write the file
start, rows = time.time(), []
for d, names in datasets.items():
    # The public example folder holds only some of sample_submission's datasets; at scoring time all exist.
    missing = not (DATA / 'test' / d).is_dir()
    if missing or time.time() - start > BUDGET_S:
        print(d, 'not in test folder' if missing else 'skipped: time budget hit', '-> outliers')
        rows += [dict(dataset=d, scene='outliers', image=n, rotation_matrix=NAN_R, translation_vector=NAN_T) for n in names]
        continue
    try:
        r, stats = run_dataset(d, DATA / 'test' / d, names, Path('/tmp/imc_work'), cfg)
        rows += r
        print(d, {k: v for k, v in stats.items() if k != 'sfm_error'})
        if 'sfm_error' in stats: print(stats['sfm_error'][-500:])
    except Exception:
        traceback.print_exc()
        rows += [dict(dataset=d, scene='outliers', image=n, rotation_matrix=NAN_R, translation_vector=NAN_T) for n in names]
sub = write_submission(rows, Path('/kaggle/working/submission.csv'), DATA / 'sample_submission.csv')
shutil.rmtree('/kaggle/working/imc', ignore_errors=True)
print(sub.shape, f'{(time.time() - start) / 60:.1f} min'); sub.head()"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "cells": [
        {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, src in cells
    ],
}
out = ROOT / "notebooks"
out.mkdir(exist_ok=True)
(out / "imc25_submission.ipynb").write_text(json.dumps(nb, indent=1))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{USER}/imc25-dinov2-aliked-lightglue-colmap",
    "title": "IMC25 DINOv2 ALIKED LightGlue COLMAP",
    "code_file": "imc25_submission.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False,
    "dataset_sources": [f"{USER}/imc25-offline-deps"],
    "competition_sources": ["image-matching-challenge-2025-ongoing"],
}, indent=1))
print("wrote", out)
