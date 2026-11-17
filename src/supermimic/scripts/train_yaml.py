"""Validated YAML entry point for local / cloud training."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from dotenv import dotenv_values


def merge_config(target: Any, values: dict, prefix="") -> Any:
  """Strict recursive dataclass overlay: typos fail before building the simulator."""
  fields = {f.name for f in dataclasses.fields(target)}
  for key, value in values.items():
    if key not in fields:
      raise ValueError(f"Unknown config key: {prefix}{key}")
    current = getattr(target, key)
    if dataclasses.is_dataclass(current):
      if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at {prefix}{key}")
      # Frozen task configs need dataclasses.replace rather than setattr.
      if vars(type(current))["__dataclass_params__"].frozen:
        # Reconstruct frozen nested specs using the same strict merge validation.
        current = merge_frozen(current, value, f"{prefix}{key}.")
      else:
        merge_config(current, value, f"{prefix}{key}.")
      setattr(target, key, current)
    elif isinstance(current, tuple):
      setattr(target, key, tuple(value))
    else:
      setattr(target, key, value)
  return target


def merge_frozen(target, values, prefix):
  fields = {f.name for f in dataclasses.fields(target)}
  updates = {}
  for key, value in values.items():
    if key not in fields:
      raise ValueError(f"Unknown config key: {prefix}{key}")
    old = getattr(target, key)
    updates[key] = (
      merge_frozen(old, value, f"{prefix}{key}.")
      if dataclasses.is_dataclass(old)
      else value
    )
  return dataclasses.replace(target, **updates)


def set_override(config, expression):
  name, sep, raw = expression.partition("=")
  if not sep:
    raise ValueError("Use --set dotted.key=value")
  keys = name.split(".")
  node = config
  for key in keys[:-1]:
    node = node.setdefault(key, {})
  node[keys[-1]] = yaml.safe_load(raw)


def apply_stage(config, stage, distillation, checkpoint=None):
  """Explicit stages; user --set overrides are applied afterward."""
  runner = config.setdefault("runner", {})
  human = stage != "baseline"
  runner.setdefault("tasks", {})["human_body"] = human
  runner.update(
    human_adaptation=stage == "adaptation",
    checkpoint_dir=str(checkpoint) if checkpoint else None,
    slot_mix={
      "motion": 0.0 if stage == "adaptation" else (0.5 if human else 1.0),
      "velocity": 0.0,
      **({"human_body": 1.0 if stage == "adaptation" else 0.5} if human else {}),
    },
    latent_bc_weight=0.1 if human and distillation != "off" else 0.0,
    action_bc_weight=1.0 if human and distillation != "off" else 0.0,
    bc_decay_iterations=runner.get("max_iterations", 10000) // 2
    if human and distillation == "decay"
    else 0,
    run_name=f"seed-{stage}-{distillation}" if human else "seed-motion-baseline",
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
  parser.add_argument("--set", action="append", default=[])
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--print-config", action="store_true")
  parser.add_argument(
    "--stage", choices=["baseline", "adaptation", "joint"], default="baseline"
  )
  parser.add_argument(
    "--distillation", choices=["off", "fixed", "decay"], default="decay"
  )
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument(
    "--output-dir", type=Path, help="New checkpoint directory (refuses overwrite)"
  )
  parser.add_argument(
    "--data", type=Path, help="Processed dataset root containing train/ and validation/"
  )
  args = parser.parse_args()
  config = yaml.safe_load(args.config.read_text())
  apply_stage(config, args.stage, args.distillation, args.checkpoint)
  if args.output_dir:
    config["runner"]["log_directory"] = str(args.output_dir)
  if args.data:
    from supermimic.data.packaged import resolve_dataset

    args.data = resolve_dataset(args.data)
    config["motion_file"] = str(args.data / "train/*.npz")
    config["runner"]["human_data_dir"] = str(args.data / "train")
  if args.smoke:
    for key, value in {
      "num_envs": 4,
      "runtime.compile": False,
      "runtime.cuda_graphs": False,
      "runner.max_iterations": 8,
      "runner.buffer_size": 32,
      "runner.log_interval": 1,
      "runner.save_interval": 8,
      "runner.replay_ratio": 1,
      "runner.dynamics_hidden_dims": [64, 32],
      "runner.actor_hidden_dims": [64, 32],
      "runner.critic_hidden_dims": [32, 16],
      "runner.policy.batch_size": 8,
      "runner.policy.horizon": 3,
      "runner.wm.batch_size": 8,
      "runner.wm.num_batches": 1,
      "runner.wm.horizon": 3,
      "runner.run_name": "seed-human-smoke",
      # Keep BC nonzero when the replay first becomes large enough to train.
      "runner.bc_decay_iterations": 8 if args.distillation == "decay" else 0,
    }.items():
      set_override(config, f"{key}={json.dumps(value)}")
  for expression in args.set:
    set_override(config, expression)
  if config["runner"].get("human_adaptation") and not config["runner"].get(
    "checkpoint_dir"
  ):
    parser.error(
      "Adaptation requires --checkpoint PATH to an evaluated motion baseline"
    )
  if args.print_config:
    print(yaml.safe_dump(config, sort_keys=False))
    return
  recipe = copy.deepcopy(config)
  runtime = config.pop("runtime", {})
  allowed_runtime = {"compile", "cuda_graphs", "seed", "wandb_mode", "wandb_entity"}
  if set(runtime) - allowed_runtime:
    raise ValueError(f"Unknown runtime keys: {set(runtime) - allowed_runtime}")
  os.environ["TORCHDYNAMO_DISABLE"] = "0" if runtime.get("compile", True) else "1"
  os.environ["SUPERMIMIC_CUDA_GRAPHS"] = (
    "1" if runtime.get("cuda_graphs", True) else "0"
  )
  # Environment > .env.local > .env. Never source shell code or print secrets.
  for filename in [".env.local", ".env"]:
    for key, value in dotenv_values(filename).items():
      if value and not os.environ.get(key):
        os.environ[key] = value
  mode = str(runtime.get("wandb_mode", "online"))
  os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE") or mode
  if runtime.get("wandb_entity"):
    os.environ["WANDB_ENTITY"] = runtime["wandb_entity"]
  import torch
  import wandb

  if os.environ["WANDB_MODE"] == "online" and not wandb.api.api_key:
    print("[INFO] W&B sin credenciales: guardando offline; usa wandb sync después.")
    os.environ["WANDB_MODE"] = "offline"
  recipe.setdefault("runtime", {})["wandb_mode"] = os.environ["WANDB_MODE"]
  seed = runtime.get("seed", 42)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  from supermimic.runners.supermimic_runner import SuperMimicRunner
  from supermimic.scripts.train import TrainConfig
  from supermimic.scripts.train_utils import create_env, setup_device

  task = config.pop("task")
  cfg = merge_config(TrainConfig.from_task(task), config)
  cfg.runner.experiment_config = recipe
  if not cfg.motion_file:
    raise ValueError("YAML training requires explicit local motion_file")
  dataset_root = Path(cfg.motion_file).parent.parent
  if (dataset_root / ".processing").exists():
    raise ValueError("Dataset preprocessing is incomplete; rerun preprocess_dataset.sh")
  if cfg.num_envs < 1 or cfg.runner.max_iterations < 1:
    raise ValueError("num_envs and max_iterations must be positive")
  if cfg.runner.policy.batch_size < 4 or cfg.runner.wm.horizon < 2:
    raise ValueError("policy batch_size >= 4 and wm horizon >= 2 required")
  if cfg.runner.buffer_size < cfg.runner.wm.horizon:
    raise ValueError("buffer_size must cover wm.horizon")
  if cfg.runner.log_directory and Path(cfg.runner.log_directory).exists():
    raise ValueError("Output directory already exists; choose a new --output-dir")
  if (
    not 0 <= cfg.runner.human_dropout_probability <= 1
    or cfg.runner.human_position_noise < 0
    or cfg.runner.human_max_dropout_frames < 1
  ):
    raise ValueError("Invalid human sensor augmentation parameters")
  if cfg.runner.human_adaptation and cfg.runner.slot_mix.get("human_body", 0) != 1:
    raise ValueError("Adaptation must allocate all slots to human_body")
  if (
    min(
      cfg.runner.latent_bc_weight,
      cfg.runner.action_bc_weight,
      cfg.runner.bc_decay_iterations,
    )
    < 0
  ):
    raise ValueError("Distillation weights and decay iterations must be nonnegative")
  cfg.registry_name = None
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

  command = cast(MotionCommandCfg, cfg.env.commands["motion"])
  command.motion_file = cfg.motion_file
  command.sampling_mode = (
    "external" if cfg.runner.sampling_mode == "return" else cfg.runner.sampling_mode
  )
  device = setup_device(cfg.device)
  env = create_env(cfg.env, cfg.num_envs, device)
  try:
    runner = SuperMimicRunner(env, cfg.runner, task, device=device)
    runner.run()
  finally:
    env.close()
    wandb.finish()


if __name__ == "__main__":
  main()
