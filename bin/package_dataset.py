#!/usr/bin/env python3
"""Package existing NPZ clips for Hub upload without recomputing them."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, default=Path("data/seed-processed-full"))
  parser.add_argument(
    "--output", type=Path, default=Path("data/seed-processed-full-hub")
  )
  parser.add_argument("--clips-per-shard", type=int, default=1000)
  args = parser.parse_args()
  source, destination = args.input.resolve(), args.output.resolve()
  if args.clips_per_shard < 1:
    parser.error("--clips-per-shard must be positive")
  if source == destination or source in destination.parents:
    parser.error("--output must be outside --input")
  if (source / ".processing").exists():
    parser.error("Dataset is still being processed (.processing exists)")
  metadata = ["manifest.jsonl", "preprocess.json", "summary.json"]
  for name in metadata:
    if not (source / name).is_file():
      parser.error(f"Missing metadata: {name}")
  clips = {
    split: sorted((source / split).rglob("*.npz")) for split in ("train", "validation")
  }
  if not all(clips.values()):
    parser.error("Expected nonempty train and validation splits")
  # A new destination prevents silently mixing packages from different datasets.
  destination.mkdir(parents=True, exist_ok=False)
  (destination / ".gitignore").write_text(".unpacked/\n.cache/\n")
  shards = destination / "shards"
  shards.mkdir()
  index = {"format": "tar", "splits": {}, "shards": []}
  for split, paths in clips.items():
    index["splits"][split] = len(paths)
    for offset in range(0, len(paths), args.clips_per_shard):
      batch = paths[offset : offset + args.clips_per_shard]
      number = offset // args.clips_per_shard
      folder = shards / split / f"{number // 100:03d}"
      folder.mkdir(parents=True, exist_ok=True)
      target = folder / f"{split}-{number:05d}.tar"
      with tarfile.open(target.with_suffix(".tmp"), "w") as archive:
        for path in batch:
          archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
      target.with_suffix(".tmp").replace(target)
      with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
      index["shards"].append(
        {
          "path": str(target.relative_to(destination)),
          "clips": len(batch),
          "bytes": target.stat().st_size,
          "sha256": digest,
        }
      )
      print(f"{target.name}: {len(batch)} clips", flush=True)
  for name in metadata:
    shutil.copy2(source / name, destination / name)
  (destination / "shards.json").write_text(json.dumps(index, indent=2) + "\n")
  (destination / "README.md").write_text(
    "# Processed BONES-SEED SMPL–G1 pairs\n\n"
    "Existing processed NPZ clips packaged into uncompressed tar shards. "
    "No motion conversion is performed during packaging.\n\n"
    "Download and extract from the project root:\n\n```bash\n"
    "hf download NONHUMAN-RESEARCH/seed-processed-full --repo-type dataset "
    '--include "shards/*" --include "shards.json" --include "manifest.jsonl" --include "preprocess.json" '
    '--include "summary.json" --local-dir data/seed-processed-full-download\n'
    "mkdir -p data/seed-processed-full\n"
    "for shard in data/seed-processed-full-download/shards/*/*/*.tar; do\n"
    '  tar -xf "$shard" -C data/seed-processed-full\ndone\n'
    "cp data/seed-processed-full-download/{manifest.jsonl,preprocess.json,summary.json} "
    "data/seed-processed-full/\n```\n\n"
    "Extraction restores train/ and validation/ for the existing loaders. "
    "shards.json records clip counts and SHA-256 checksums. "
    "Shards are grouped into directories containing at most 100 archives. "
    "Training and auditing accept the downloaded package root via --data; "
    "evaluation accepts that root or its validation/ path. The first use verifies "
    "checksums and extracts both splits into .unpacked/ (additional disk space "
    "required); later uses reuse that copy. "
    "The original preprocessing metadata and manifest are preserved; source "
    "paths in them describe the machine that processed the data.\n"
  )
  print(json.dumps(index["splits"]), flush=True)
  print(f"Upload directory: {destination}", flush=True)


if __name__ == "__main__":
  main()
