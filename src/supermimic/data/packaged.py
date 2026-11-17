"""Restore byte-preserving tar packages for the existing motion loaders."""

import fcntl
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def resolve_dataset(root: Path) -> Path:
  """Accept either an NPZ dataset or a downloaded package; cache extraction."""
  root = root.expanduser().resolve()
  if (root / ".processing").exists():
    raise ValueError(f"Dataset preprocessing incomplete: {root}")
  if not (root / "shards.json").is_file():
    return root
  index = json.loads((root / "shards.json").read_text())
  if index.get("format") != "tar":
    raise ValueError("Unsupported dataset package format")
  cache = root / ".unpacked"
  cache.mkdir(exist_ok=True)
  with (cache / ".lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    marker = cache / ".complete.json"
    if marker.exists() and json.loads(marker.read_text()) == index:
      return cache
    for shard in index["shards"]:
      path = (root / shard["path"]).resolve()
      if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid or missing shard: {shard['path']}")
      with path.open("rb") as stream:
        if (
          path.stat().st_size != shard["bytes"]
          or hashlib.file_digest(stream, "sha256").hexdigest() != shard["sha256"]
        ):
          raise ValueError(f"Corrupt shard: {path}")
      print(f"Extracting {path.name}", flush=True)
      with tarfile.open(path) as archive:
        members = archive.getmembers()
        if len(members) != shard["clips"]:
          raise ValueError(f"Incorrect clip count: {path}")
        for member in members:
          parts = Path(member.name).parts
          if (
            not member.isfile()
            or len(parts) != 2
            or parts[0] not in ("train", "validation")
            or not parts[1].endswith(".npz")
          ):
            raise ValueError(f"Invalid archive member: {member.name}")
        archive.extractall(cache, members=members, filter="data")
    for split, count in index["splits"].items():
      if len(list((cache / split).glob("*.npz"))) != count:
        raise ValueError(f"Incorrect extracted split count: {split}")
    for name in ("manifest.jsonl", "summary.json", "preprocess.json"):
      shutil.copy2(root / name, cache / name)
    temp = marker.with_suffix(".tmp")
    temp.write_text(json.dumps(index))
    temp.replace(marker)
  return cache
