"""Audit paired NPZ timing/heading, splits and channels before long training.

Diagnostics flag suspicious pairs; retargeting can legitimately change height
and phase. They do not prove alignment and never silently shift training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.data.human import SCHEMA
from supermimic.data.prepare_seed import family_id


def lag_diagnostic(human, robot, max_lag=10):
  human, robot = np.diff(human), np.diff(robot)
  if min(np.std(human), np.std(robot)) < 1e-5 or len(human) < 2 * max_lag + 10:
    return None, None  # Idle/short clips do not identify temporal alignment.
  scores = []
  for lag in range(-max_lag, max_lag + 1):
    a = human[max_lag:-max_lag]
    b = robot[max_lag + lag : len(robot) - max_lag + lag]
    scores.append(float(np.corrcoef(a, b)[0, 1]))
  index = int(np.nanargmax(scores))
  return index - max_lag, scores[index]


def audit_clip(path):
  with np.load(path) as clip:
    frames = clip["human_frames"]
    if str(clip["human_schema"]) != SCHEMA or float(clip["fps"]) != 50:
      raise ValueError(f"Unsupported schema/fps: {path}")
    n = len(frames)
    shapes = {
      "human_frames": (n, 136),
      "human_root_quat_w": (n, 4),
      "root_pos_w": (n, 3),
      "root_quat_w": (n, 4),
      "joint_pos": (n, 29),
      "joint_vel": (n, 29),
    }
    for key, shape in shapes.items():
      if clip[key].shape != shape or not np.isfinite(clip[key]).all():
        raise ValueError(f"Invalid {key}: {path}")
    human_yaw = np.unwrap(
      Rotation.from_quat(clip["human_root_quat_w"][:, [1, 2, 3, 0]]).as_euler("xyz")[
        :, 2
      ]
    )
    robot_yaw = np.unwrap(
      Rotation.from_quat(clip["root_quat_w"][:, [1, 2, 3, 0]]).as_euler("xyz")[:, 2]
    )
    heading_error = np.angle(np.exp(1j * (human_yaw - robot_yaw)))
    height = frames[:, 84].copy()
    robot_height = clip["root_pos_w"][:, 2].copy()
  lag, correlation = lag_diagnostic(height, robot_height)
  heading = float(np.degrees(np.median(np.abs(heading_error))))
  warnings = []
  if heading > 30:
    warnings.append("median_heading_error_over_30deg")
  if lag is not None and abs(lag) > 2 and correlation > 0.5:
    warnings.append("height_phase_offset_over_40ms")
  if height.min() <= 0 or height.max() > 2:
    warnings.append("implausible_scaled_pelvis_height")
  row = dict(
    name=path.stem,
    split=path.parent.name,
    frames=n,
    family=family_id(path.stem),
    heading_error_median_deg=heading,
    height_lag_frames=lag,
    height_lag_correlation=correlation,
    warnings=warnings,
  )
  return row, (height, robot_height, human_yaw, robot_yaw)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--data", type=Path, required=True)
  p.add_argument("--plots", type=int, default=12)
  p.add_argument(
    "--strict", action="store_true", help="Exit nonzero on diagnostic warnings"
  )
  args = p.parse_args()
  from supermimic.data.packaged import resolve_dataset

  args.data = resolve_dataset(args.data)
  if (args.data / ".processing").exists():
    p.error("Preprocessing incomplete")
  paths = sorted(args.data.glob("train/*.npz")) + sorted(
    args.data.glob("validation/*.npz")
  )
  if (
    not paths
    or not list(args.data.glob("validation/*.npz"))
    or not list(args.data.glob("train/*.npz"))
  ):
    p.error("Both train and validation must contain clips")
  out = args.data / "audit"
  out.mkdir(exist_ok=True)
  rows, families = [], {"train": set(), "validation": set()}
  for path in paths:
    row, series = audit_clip(path)
    families[row["split"]].add(row["family"])
    if len(rows) < args.plots:
      import matplotlib

      matplotlib.use("Agg")
      import matplotlib.pyplot as plt

      fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
      times = np.arange(row["frames"]) / 50
      for axis, pair in zip(axes, (series[:2], series[2:]), strict=True):
        axis.plot(times, pair[0], label="human")
        axis.plot(times, pair[1], label="G1")
        axis.legend()
      axes[0].set_ylabel("Pelvis height (m)")
      axes[1].set_ylabel("Unwrapped yaw (rad)")
      axes[1].set_xlabel("Time (s)")
      fig.suptitle(path.stem)
      fig.tight_layout()
      fig.savefig(out / f"{path.stem}.png")
      plt.close(fig)
    rows.append(row)
  overlap = sorted(families["train"] & families["validation"])
  report = dict(
    clips=len(rows),
    split_family_overlap=overlap,
    flagged=sum(bool(r["warnings"]) for r in rows),
    rows=rows,
    note="Heuristic diagnostics; inspect plots and flagged pairs. No automatic realignment.",
  )
  (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
  print(f"Report and plots: {out}")
  if overlap or (args.strict and report["flagged"]):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
