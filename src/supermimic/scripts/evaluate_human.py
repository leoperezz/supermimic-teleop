"""Evaluate every held-out clip, with optional causal simulated human transport."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import torch
import wandb
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

from supermimic.data.human import HumanMotionTable, HumanStream, StaleHumanSignal
from supermimic.runners.exporter import HumanActor
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train import TrainConfig
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.task_spec import build_actor_obs_td


def episode_outcome(terminated, clip_end, timed_out):
  # A physical failure on the last frame is still a failure.
  if terminated:
    return "failure"
  if clip_end:
    return "completed"
  return "time_limit" if timed_out else "step_limit"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", required=True, type=Path)
  p.add_argument("--data", type=Path, default=Path("data/seed-processed/validation"))
  p.add_argument("--num-envs", type=int, default=1, help="Repeated trials per clip")
  p.add_argument(
    "--steps", type=int, default=0, help="Cap per clip; 0 evaluates to clip end/failure"
  )
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--branch", choices=["human_body", "motion"], default="human_body")
  p.add_argument(
    "--wandb-mode", choices=["online", "offline", "disabled"], default="offline"
  )
  p.add_argument("--output", type=Path)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--delay-ms", type=float, default=0)
  p.add_argument(
    "--jitter-ms",
    type=float,
    default=0,
    help="Additional uniform [0,jitter] packet delay",
  )
  p.add_argument("--position-noise", type=float, default=0)
  p.add_argument("--dropout", type=float, default=0)
  p.add_argument("--max-dropout-frames", type=int, default=4)
  p.add_argument("--max-age-ms", type=float, default=200)
  p.add_argument(
    "--outage-at", type=float, default=1, help="Seconds from start of each clip"
  )
  p.add_argument("--outage-seconds", type=float, default=0)
  args = p.parse_args()
  from supermimic.data.packaged import resolve_dataset

  if args.data.name in ("train", "validation"):
    args.data = resolve_dataset(args.data.parent) / args.data.name
  elif (args.data / "shards.json").is_file() or (args.data / "validation").is_dir():
    args.data = resolve_dataset(args.data) / "validation"
  paths = sorted(args.data.glob("*.npz"))
  if args.steps < 0 or args.num_envs < 1 or not paths:
    p.error("steps >= 0, num-envs >= 1 and a nonempty dataset are required")
  if (args.data.parent / ".processing").exists():
    p.error("Dataset preprocessing incomplete")
  if not 0 <= args.dropout <= 1 or args.max_dropout_frames < 1 or args.max_age_ms <= 0:
    p.error("Invalid dropout/max-age parameters")
  if (
    min(
      args.delay_ms,
      args.jitter_ms,
      args.position_noise,
      args.outage_at,
      args.outage_seconds,
    )
    < 0
  ):
    p.error("Sensor parameters must be nonnegative")
  if args.branch == "motion" and any(
    [
      args.delay_ms,
      args.jitter_ms,
      args.position_noise,
      args.dropout,
      args.outage_seconds,
    ]
  ):
    p.error("Human sensor perturbations require --branch human_body")
  os.environ["TORCHDYNAMO_DISABLE"] = "1"
  torch.manual_seed(args.seed)
  raw = json.loads((args.checkpoint / "runner_config.json").read_text())
  cfg = load_runner_cfg({"runner_cfg": raw})
  if args.branch == "human_body" and not cfg.tasks.human_body:
    p.error("Checkpoint has no human encoder")
  cfg.slot_mix = {args.branch: 1.0}
  task = raw.get("experiment_config", {}).get("task", "SuperMimic-GenCoord-G1")
  env_cfg = TrainConfig.from_task(task).env
  command_cfg = cast(MotionCommandCfg, env_cfg.commands["motion"])
  command_cfg.motion_file = str(args.data / "*.npz")
  command_cfg.sampling_mode = "start"
  # Each clip must have enough wall-clock simulation time to finish.
  max_frames = 0
  for path in paths:
    with np.load(path) as clip:
      max_frames = max(max_frames, len(clip["joint_pos"]))
  env_cfg.episode_length_s = max_frames / 50 + 1
  device = setup_device(args.device)
  env = create_env(env_cfg, args.num_envs, device)
  try:
    wm = build_world_model(cfg, env, device)
    actor = build_actor(cfg, wm, device)
    actor.load_state_dict(
      torch.load(args.checkpoint / "actor.pt", map_location=device, weights_only=True)
    )
    actor.eval()
    command = env.unwrapped.command_manager.get_term("motion")
    motion = command.motion
    table = (
      HumanMotionTable(
        motion,
        str(args.data),
        device,
        noise_std=args.position_noise,
        dropout_prob=args.dropout,
        max_dropout=args.max_dropout_frames,
        seed=args.seed,
      )
      if args.branch == "human_body"
      else None
    )
    deploy = HumanActor(actor).eval() if table is not None else None
    ids = torch.arange(args.num_envs, device=device)
    ends = motion.motion_ends.cpu().tolist()
    starts = [0, *ends[:-1]]
    trials = []
    wandb.init(
      project=cfg.wandb_project,
      name=f"evaluate-{args.branch}-{args.checkpoint.name}",
      mode=args.wandb_mode,
      config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    with torch.inference_mode():
      for clip_index, (name, start, end) in enumerate(
        zip(motion.motion_names, starts, ends, strict=True)
      ):
        # Loading a differently sized actor or failing early on a previous clip
        # must not change the next clip's initial state or transport realization.
        torch.manual_seed(args.seed + clip_index)
        rng = np.random.default_rng(args.seed + clip_index)
        env.reset()
        command.reset_to_frame(ids, start)
        env.unwrapped.sim.forward()
        command.update_relative_body_poses()
        env.unwrapped.sim.sense()
        obs = env.get_observations()
        history = None
        active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
        records = [
          dict(clip=name, trial=i, steps=0, outcome="step_limit", metrics={})
          for i in range(args.num_envs)
        ]
        stream = HumanStream(args.max_age_ms / 1000)
        if table is not None:
          # Initial calibrated pose is available before the control episode starts.
          stream.receive(table.frames[start], table.quats[start], 0.0)
        limit = min(end - start, args.steps) if args.steps else end - start
        for step in range(limit):
          state = obs["full_state"]
          rep = wm.representation.represent(state)
          aux = wm.representation.extract_aux_state(state)
          sensor = wm.representation.compute_actor_sensor(
            rep,
            wm.sim_context.read(env.unwrapped),
            aux_state=aux,
            sim_actor_obs=obs["actor"] if cfg.actor_obs == "proprio" else None,
            noisy=False,
          )
          history = (
            sensor[:, None].expand(-1, cfg.history_length, -1).clone()
            if history is None
            else torch.cat([history[:, 1:], sensor[:, None]], dim=1)
          )
          if table is not None:
            now = step * 0.02
            delay = args.delay_ms + rng.uniform(0, args.jitter_ms)
            source = int(np.floor(step - delay / 20))
            outage = args.outage_at <= now < args.outage_at + args.outage_seconds
            if source >= 0 and not outage:
              stream.receive(
                table.frames[start + source], table.quats[start + source], source * 0.02
              )
            try:
              stream.tick(now)
              goal = stream.goal(aux[:, 3:7], now)
            except StaleHumanSignal:
              for i in active.nonzero().flatten().tolist():
                records[i]["outcome"] = "sensor_timeout"
              break  # Stop simulation trial; do not invent a hardware fallback.
            action = deploy(goal, history)
          else:
            spec = actor.task_specs[0]
            action = actor(
              build_actor_obs_td(
                {"motion": spec.extract_training_goal(motion, command.time_steps, aux)},
                history,
              )
            )
          if not torch.isfinite(action[active]).all():
            raise RuntimeError("Nonfinite evaluation actions")
          # Measure before step: auto-reset must not contaminate terminal metrics.
          errors = (command.robot_body_pos_w - command.body_pos_relative_w).norm(dim=-1)
          values = {"body_position_error_m": errors.mean(-1)}
          names = command.cfg.body_names
          wrists = [i for i, n in enumerate(names) if "wrist" in n]
          feet = [i for i, n in enumerate(names) if "ankle_roll" in n]
          if wrists:
            values["wrist_position_error_m"] = errors[:, wrists].mean(-1)
          if feet:
            # Kinematic proxy, not a contact-force measurement.
            low = command.robot_body_pos_w[:, feet, 2] < 0.12
            speed = command.robot_body_lin_vel_w[:, feet, :2].norm(dim=-1)
            values["low_foot_speed_proxy_mps"] = (speed * low).sum(-1) / low.sum(
              -1
            ).clamp(min=1)
          values["pelvis_position_error_m"] = (
            command.robot_anchor_pos_w - command.anchor_pos_w
          ).norm(dim=-1)
          obs, reward, done, _ = env.step(action)
          values["reward"] = reward
          manager = env.unwrapped.termination_manager
          for i in active.nonzero().flatten().tolist():
            record = records[i]
            record["steps"] += 1
            for key, value in values.items():
              record["metrics"][key] = record["metrics"].get(key, 0.0) + float(value[i])
            if done[i]:
              record["outcome"] = episode_outcome(
                bool(manager.terminated[i]),
                bool(manager.get_term("motion_end")[i]),
                bool(manager.time_outs[i]),
              )
              active[i] = False
          if not active.any():
            break
        for record in records:
          count = record["steps"]
          record["seconds"] = count * env.unwrapped.step_dt
          record["clip_frames"] = end - start
          record["metrics"] = {
            k: v / max(count, 1) for k, v in record["metrics"].items()
          }
          trials.append(record)
        wandb.log(
          {
            "Eval/completion_rate": sum(r["outcome"] == "completed" for r in records)
            / len(records)
          },
          step=clip_index,
        )
        print(
          f"[{clip_index + 1}/{len(starts)}] {name}: {[r['outcome'] for r in records]}",
          flush=True,
        )
    counts = dict(Counter(r["outcome"] for r in trials))
    result = dict(
      branch=args.branch,
      dataset=str(args.data),
      num_clips=len(starts),
      trials=len(trials),
      outcomes=counts,
      completion_rate=counts.get("completed", 0) / len(trials),
      mean_seconds=float(np.mean([r["seconds"] for r in trials])),
      settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
      per_clip=trials,
    )
    output = args.output or args.checkpoint / f"evaluation_{args.branch}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_clip"}, indent=2))
  finally:
    env.close()
    wandb.finish()


if __name__ == "__main__":
  main()
