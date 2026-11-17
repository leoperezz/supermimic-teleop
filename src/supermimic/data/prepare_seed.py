"""Convert paired BONES-SEED / SONIC SMPL to SuperMimic NPZ at 50 Hz.

Reuses GR00T's CSV reader and filename filter. Writes each accepted clip once,
with a manifest (including rejected clips), deterministic family split and provenance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

import joblib
import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from scipy.spatial.transform import Rotation, Slerp

from supermimic.data.human import FRAME_DIM, SCHEMA


def load_script(path: Path):
  spec = importlib.util.spec_from_file_location(path.stem, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load GR00T script: {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def filter_keywords(path: Path) -> list[str]:
  """Read the upstream CLI's literal defaults without duplicating its blacklist."""
  for node in ast.walk(ast.parse(path.read_text())):
    if isinstance(node, ast.Call) and node.args:
      if (
        isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--filter-keywords"
      ):
        return ast.literal_eval(
          next(k.value for k in node.keywords if k.arg == "default")
        )
  raise ValueError("Upstream filter list not found")


def family_id(name: str) -> str:
  # Group actors, take numbers and mirrored variants of the same motion family.
  return re.sub(r"_\d+$", "", name.split("__")[0]).lower()


def split_for(name: str, fraction: float, seed: int) -> str:
  value = int(hashlib.sha256(f"{seed}:{family_id(name)}".encode()).hexdigest()[:8], 16)
  return "validation" if value / 2**32 < fraction else "train"


def interpolate(values, old_times, times):
  flat = values.reshape(len(values), -1)
  out = np.stack([np.interp(times, old_times, col) for col in flat.T], axis=-1)
  return out.reshape(len(times), *values.shape[1:])


def interpolate_rotation(rot: Rotation, old_times, times) -> Rotation:
  return Slerp(old_times, rot)(times)


def backward_velocity(values, dt):
  return np.concatenate([np.zeros_like(values[:1]), np.diff(values, axis=0) / dt])


def initial_support(feet_z, fps, seconds=0.12, tolerance=0.04):
  """Reject ambiguous initial floor calibration (heuristic, not contact force).

  Future samples are used only to accept/reject the clip, never to construct
  its causal features. Require an initially low foot to remain near its plane.
  """
  count = min(len(feet_z), max(2, int(seconds * fps) + 1))
  initial = feet_z[0]
  floor = float(initial.min())
  excursion = np.max(np.abs(feet_z[:count] - initial), axis=0)
  supported = (initial - floor <= tolerance) & (excursion <= tolerance)
  if not supported.any():
    raise ValueError("initial human support ambiguous; cannot calibrate floor")
  return floor, float(excursion[initial - floor <= tolerance].min())


def wrists_from_smpl(pose):
  """SONIC PICO elbow swing -> wrist mapping, including its final L pitch sign.

  Decomposes elbow about Y; transfers remaining roll/yaw to the wrist.
  Source: pico_manager_thread_server.py lines 1420–1477.
  """
  output = []
  for elbow, wrist, sign in [(18, 20, 1), (19, 21, -1)]:
    r = Rotation.from_rotvec(pose[:, elbow])
    q = r.as_quat()
    twist = q.copy()
    twist[:, [0, 2]] = 0
    norm = np.linalg.norm(twist, axis=-1, keepdims=True)
    twist = np.where(norm > 1e-8, twist / np.maximum(norm, 1e-8), [0, 0, 0, 1])
    swing = (Rotation.from_quat(twist).inv() * r).as_euler("XYZ")
    w = Rotation.from_rotvec(pose[:, wrist]).as_euler("XYZ")
    output.extend(
      [
        sign * (swing[:, 0] + w[:, 0]),
        w[:, 1] if sign == 1 else -w[:, 1],
        swing[:, 2] + w[:, 2],
      ]
    )
  return np.stack(output, axis=-1)


def convert_pair(csv_path, smpl_path, reader, model, args):
  robot = reader.load_bones_csv(str(csv_path))
  smpl = joblib.load(smpl_path)
  pose = np.asarray(smpl["pose_aa"]).reshape(-1, 24, 3)
  sj = np.asarray(smpl["smpl_joints"])
  trans = np.asarray(smpl["transl"])
  if sj.shape != (len(pose), 24, 3) or trans.shape != (len(pose), 3):
    raise ValueError("SMPL channels have incompatible shapes")
  rt = np.arange(len(robot["joint_pos"])) / args.source_fps
  ht = np.arange(len(pose)) / float(smpl["fps"])
  mismatch = abs(rt[-1] - ht[-1])
  if mismatch > args.duration_tolerance:
    raise ValueError(f"duration mismatch {mismatch:.3f}s")
  duration = min(rt[-1], ht[-1])
  if duration < args.min_seconds:
    raise ValueError("clip too short")
  times = np.arange(int(np.floor(duration * 50)) + 1) / 50
  q = interpolate_rotation(
    Rotation.from_quat(robot["body_quat_w"][:, 0, [1, 2, 3, 0]]), rt, times
  )
  root = interpolate(robot["body_pos_w"][:, 0], rt, times)
  jp = interpolate(robot["joint_pos"], rt, times)
  # Reorder explicitly against the installed G1 model; never trust actuator order.
  names = [model.joint(i).name for i in range(1, model.njnt)]
  order = [reader.BONES_CSV_JOINT_NAMES.index(n + "_dof") for n in names]
  jp = jp[:, order]
  angular = np.zeros((len(times), 3))
  angular[1:] = (q[1:] * q[:-1].inv()).as_rotvec() * 50
  result = dict(
    root_pos_w=root,
    root_quat_w=q.as_quat()[:, [3, 0, 1, 2]],
    root_lin_vel_w=backward_velocity(root, 0.02),
    root_ang_vel_w=angular,
    joint_pos=jp,
    joint_vel=backward_velocity(jp, 0.02),
  )
  if not all(np.isfinite(v).all() for v in result.values()):
    raise ValueError("nonfinite robot channels")
  ranges = model.jnt_range[1:]
  if np.any(jp < ranges[:, 0] - args.joint_limit_tolerance) or np.any(
    jp > ranges[:, 1] + args.joint_limit_tolerance
  ):
    raise ValueError("joint limit exceeded")
  # Cheap FK support check, NOT the paper's full feasibility pass.
  data = mujoco.MjData(model)
  feet = [model.body(n).id for n in ["left_ankle_roll_link", "right_ankle_roll_link"]]
  support = []
  for i in range(0, len(times), 5):
    data.qpos[:3] = root[i]
    data.qpos[3:7] = result["root_quat_w"][i]
    data.qpos[7:] = jp[i]
    mujoco.mj_kinematics(model, data)
    support.append(data.xpos[feet, 2].min())
  support = np.asarray(support)
  if (
    support.min() < -args.max_penetration
    or np.mean(support > args.max_support_height) > args.max_floating_fraction
  ):
    raise ValueError(
      f"FK support rejected: min={support.min():.3f}, floating={np.mean(support > args.max_support_height):.3f}"
    )
  # SONIC convention: stored joints are Z-up, root-centered before local rotation.
  ytoz = Rotation.from_euler("x", 90, degrees=True)
  base_inv = Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
  human_rot = ytoz * Rotation.from_rotvec(pose[:, 0]) * base_inv
  hrot = interpolate_rotation(human_rot, ht, times)
  joints = interpolate(sj - sj[:, :1], ht, times)
  local = np.einsum("tij,tkj->tki", hrot.inv().as_matrix(), joints) * args.human_scale
  # Initial standing calibration only: never use future samples to set the floor.
  pelvis = ytoz.apply(trans) + sj[:, 0]
  feet_z = (ytoz.apply(trans)[:, None] + sj)[:, [7, 8, 10, 11], 2]
  floor, initial_excursion = initial_support(feet_z, float(smpl["fps"]))
  pelvis = interpolate(pelvis, ht, times)
  height = (pelvis[:, 2] - floor) * args.human_scale
  body_vel = hrot.inv().apply(backward_velocity(pelvis, 0.02) * args.human_scale)
  yaw = np.unwrap(hrot.as_euler("xyz")[:, 2])
  velocity = np.column_stack([body_vel[:, :2], backward_velocity(yaw, 0.02)])
  # Identical causal one-pole velocity filter can run online, reset each clip.
  for i in range(1, len(velocity)):
    velocity[i] = (
      args.velocity_alpha * velocity[i] + (1 - args.velocity_alpha) * velocity[i - 1]
    )
  wrist = interpolate(wrists_from_smpl(pose), ht, times)
  frames = np.concatenate(
    [
      local.reshape(len(times), 72),
      np.zeros((len(times), 6)),
      wrist,
      height[:, None],
      velocity,
      np.ones((len(times), 24)),
      np.zeros((len(times), 24)),
    ],
    axis=-1,
  )
  if frames.shape[1] != FRAME_DIM or not np.isfinite(frames).all():
    raise ValueError("invalid human features")
  result.update(human_frames=frames, human_root_quat_w=hrot.as_quat()[:, [3, 0, 1, 2]])
  result = {k: v.astype(np.float32) for k, v in result.items()}
  result.update(
    joint_names=np.array(names), fps=np.array(50), human_schema=np.array(SCHEMA)
  )
  return result, dict(
    frames=len(times),
    seconds=float(times[-1]),
    duration_mismatch=mismatch,
    support_min=float(support.min()),
    human_floor=float(floor),
    initial_support_excursion=initial_excursion,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, default=Path("data/seed"))
  parser.add_argument("--output", type=Path, default=Path("data/seed-processed"))
  parser.add_argument("--gr00t", type=Path, default=Path("../GR00T-WholeBodyControl"))
  parser.add_argument(
    "--max-clips", type=int, default=16, help="Accepted clips; 0 = full corpus"
  )
  parser.add_argument("--include", nargs="*", default=["walk", "idle"])
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--validation-fraction", type=float, default=0.2)
  parser.add_argument("--source-fps", type=float, default=120)
  parser.add_argument("--duration-tolerance", type=float, default=0.1)
  parser.add_argument("--min-seconds", type=float, default=1.0)
  parser.add_argument("--human-scale", type=float, default=0.8)
  parser.add_argument("--velocity-alpha", type=float, default=0.25)
  parser.add_argument("--joint-limit-tolerance", type=float, default=0.3)
  parser.add_argument("--max-penetration", type=float, default=0.08)
  parser.add_argument("--max-support-height", type=float, default=0.25)
  parser.add_argument("--max-floating-fraction", type=float, default=0.2)
  args = parser.parse_args()
  if (
    not 0 < args.validation_fraction < 1 or args.max_clips < 0 or args.human_scale <= 0
  ):
    parser.error("Invalid split, clip count or human scale")
  scripts = args.gr00t / "gear_sonic/data_process"
  reader = load_script(scripts / "convert_soma_csv_to_motion_lib.py")
  filter_path = scripts / "filter_and_copy_bones_data.py"
  filter_module = load_script(filter_path)
  blacklist = filter_keywords(filter_path)
  model = get_spec().compile()
  if not (args.input / "smpl_filtered").is_dir():
    parser.error("SMPL input missing")
  args.output.mkdir(parents=True, exist_ok=True)
  settings = {
    k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
  }
  # Refuse mixing incompatible preprocessing recipes in an existing destination.
  config_path = args.output / "preprocess.json"
  recipe = {"schema": SCHEMA, "settings": settings, "filter_keywords": blacklist}
  if config_path.exists() and json.loads(config_path.read_text()) != recipe:
    raise ValueError("Output recipe differs; use another --output directory")
  config_path.write_text(json.dumps(recipe, indent=2) + "\n")
  counts, accepted = Counter(), []
  manifest = args.output / "manifest.jsonl"
  previous = []
  if manifest.exists():
    previous = [json.loads(line) for line in manifest.read_text().splitlines()]
  processing = args.output / ".processing"
  processing.touch()
  candidates = sorted((args.input / "g1/csv").rglob("*.csv"))
  random.Random(args.seed).shuffle(candidates)
  manifest_temp = manifest.with_suffix(".jsonl.tmp")
  with manifest_temp.open("w") as log:
    for csv in candidates:
      name = csv.stem
      if filter_module.should_filter_out(name, blacklist, args.include or None):
        counts["name_filter"] += 1
        continue
      smpl = args.input / "smpl_filtered" / f"{name}.pkl"
      if not smpl.exists():
        counts["missing_smpl"] += 1
        continue
      split = split_for(name, args.validation_fraction, args.seed)
      target = args.output / split / f"{name}.npz"
      row = dict(
        name=name, family=family_id(name), split=split, csv=str(csv), smpl=str(smpl)
      )
      try:
        result, details = convert_pair(csv, smpl, reader, model, args)
        target.parent.mkdir(exist_ok=True)
        temp = target.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **result)
        temp.replace(target)
        row.update(status="accepted", path=str(target), **details)
        accepted.append(row)
        counts[split] += 1
        print(
          f"[{len(accepted)}] {split}: {name} ({details['frames']} frames)", flush=True
        )
      except (ValueError, KeyError, IndexError) as exc:
        row.update(status="rejected", reason=str(exc))
        counts["rejected"] += 1
      log.write(json.dumps(row) + "\n")
      log.flush()
      if args.max_clips and len(accepted) >= args.max_clips:
        break
  summary = dict(
    counts=counts,
    accepted=len(accepted),
    schema=SCHEMA,
    fps=50,
    frames=sum(x["frames"] for x in accepted),
    note="Filename + joint-limit + sampled FK support checks; not the paper's full filter",
  )
  summary["selection"] = "seeded shuffle; grouped family split"
  summary["implementation_sha256"] = hashlib.sha256(
    Path(__file__).read_bytes()
  ).hexdigest()
  summary["gr00t_sources"] = {
    name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
    for name in ["convert_soma_csv_to_motion_lib.py", "filter_and_copy_bones_data.py"]
  }
  current = {(row["split"], row["name"]) for row in accepted}
  for row in previous:
    if row.get("status") == "accepted" and (row["split"], row["name"]) not in current:
      (args.output / row["split"] / f"{row['name']}.npz").unlink(missing_ok=True)
  (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  manifest_temp.replace(manifest)
  print(json.dumps(summary, indent=2))
  if not accepted:
    raise RuntimeError("No valid pairs; inspect manifest.jsonl")
  processing.unlink()


if __name__ == "__main__":
  main()
