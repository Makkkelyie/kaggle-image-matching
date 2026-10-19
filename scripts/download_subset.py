"""Download only part of the competition data (CSV files + a few training datasets).

The full download is large; for a laptop-sized validation run we only need the
label/threshold CSVs and the images of a handful of training datasets.

    python scripts/download_subset.py --list                 # sizes per dataset folder, no download
    python scripts/download_subset.py --datasets imc2024_lizard_pond pt_stpeters_stpauls

Needs a Kaggle API token in ~/.kaggle/kaggle.json and the competition rules accepted.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

COMP = "image-matching-challenge-2025-ongoing"


def kaggle(*args: str) -> str:
    return subprocess.run([str(Path(sys.executable).parent / "kaggle"), "competitions", *args], check=True, capture_output=True, text=True).stdout


def list_files() -> list[tuple[str, int]]:
    files, token = [], None
    while True:
        args = ["files", "-c", COMP, "--csv", "--page-size", "200"]
        if token:
            args += ["--page-token", token]
        out = kaggle(*args)
        lines = [l for l in out.splitlines() if l and not l.startswith("Next Page Token")]
        token = next((l.split("=", 1)[1].strip() for l in out.splitlines() if l.startswith("Next Page Token")), None)
        for row in csv.DictReader(io.StringIO("\n".join(lines))):
            files.append((row["name"], int(row.get("totalBytes") or row.get("size") or 0)))
        if not token:
            return files


def download(name: str, out_dir: Path) -> None:
    target = out_dir / name
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    kaggle("download", "-c", COMP, "-f", name, "-p", str(target.parent), "-q")
    zipped = target.parent / (Path(name).name + ".zip")
    if zipped.exists():  # the API zips some files
        with zipfile.ZipFile(zipped) as z:
            z.extractall(target.parent)
        zipped.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    files = list_files()
    by_folder = defaultdict(lambda: [0, 0])
    for name, size in files:
        key = "/".join(name.split("/")[:2]) if "/" in name else name
        by_folder[key][0] += 1
        by_folder[key][1] += size
    if a.list:
        for k, (n, b) in sorted(by_folder.items()):
            print(f"{k:60s} {n:5d} files {b / 1e6:9.1f} MB")
        print(f"TOTAL {len(files)} files {sum(s for _, s in files) / 1e9:.2f} GB")
        return

    wanted = [n for n, _ in files if "/" not in n]  # CSVs at the top level
    wanted += [n for n, _ in files if any(n.startswith(f"train/{d}/") for d in a.datasets)]
    for i, name in enumerate(wanted, 1):
        download(name, a.out)
        print(f"[{i}/{len(wanted)}] {name}", end="\r")
    print(f"\ndone: {len(wanted)} files in {a.out}")


if __name__ == "__main__":
    main()
