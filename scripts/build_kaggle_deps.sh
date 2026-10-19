#!/usr/bin/env bash
# Build the folder that gets uploaded as the Kaggle dataset "imc25-offline-deps".
# The submission notebook runs with internet OFF, so wheels, model weights and
# this repo's `imc` package all have to be attached as a dataset.
#
#   bash scripts/build_kaggle_deps.sh            # -> kaggle_deps/
#   kaggle datasets create -p kaggle_deps        # (Kelly) first upload
#   kaggle datasets version -p kaggle_deps -m "update"   # later updates
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=kaggle_deps
rm -rf "$OUT"; mkdir -p "$OUT/wheels" "$OUT/weights" "$OUT/src"

# Linux wheels for the Kaggle image (Python version differs between images, so grab several).
for v in 3.10 3.11 3.12 3.13; do
  for pkg in pycolmap==3.13.0 kornia_rs; do
    $PY -m pip download -q --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 \
        --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
        --python-version "$v" -d "$OUT/wheels" "$pkg"
  done
done
$PY -m pip download -q --no-deps -d "$OUT/wheels" kornia==0.8.2
$PY -m pip wheel -q --no-deps -w "$OUT/wheels" "git+https://github.com/cvg/LightGlue.git"

# ALIKED + LightGlue weights (torch hub cache) and DINOv2 (Hugging Face format).
cp ~/.cache/torch/hub/checkpoints/aliked-n16.pth "$OUT/weights/"
cp ~/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth "$OUT/weights/"
$PY - <<'EOF'
from transformers import AutoModel, AutoImageProcessor
m = AutoModel.from_pretrained("facebook/dinov2-base"); m.save_pretrained("kaggle_deps/weights/dinov2-base")
EOF

cp -r imc "$OUT/src/"
find "$OUT/src" -name __pycache__ -prune -exec rm -rf {} +

USER=$($PY -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])" 2>/dev/null || echo YOUR_KAGGLE_USERNAME)
cat > "$OUT/dataset-metadata.json" <<EOF
{"title": "imc25-offline-deps", "id": "$USER/imc25-offline-deps", "licenses": [{"name": "apache-2.0"}]}
EOF
du -sh "$OUT"; ls "$OUT/wheels"
