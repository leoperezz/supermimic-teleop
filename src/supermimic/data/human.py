"""Shared causal human command contract, independent of MotionLoader at inference.

Frame: local SMPL joints(72), relative root rot6d(6), G1 wrists(6),
scaled pelvis height(1), body planar velocity/yaw rate(3), validity(24), age(24).
Rot6d uses the first two matrix columns in row-major order, like SONIC.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul

FRAME_DIM = 136
HISTORY = 10
GOAL_DIM = FRAME_DIM * HISTORY
SCHEMA = "human_body_v1"


def augment_sensor(frames, quats, rng, noise_std, dropout_prob, max_dropout):
  """Causal global dropout: hold last estimate, flag invalid, increase its age.

  Mocks a sensor with global validity, without inventing per-joint confidence.
  Draw once per clip so collection and WM imagination see the same signal.
  """
  frames = frames.copy()
  quats = quats.copy()
  frames[:, :72] += rng.normal(0, noise_std, frames[:, :72].shape)
  remaining, age = 0, 0.0
  for i in range(1, len(frames)):
    if remaining == 0 and rng.random() < dropout_prob:
      remaining = int(rng.integers(1, max_dropout + 1))
    if remaining:
      frames[i, :88] = frames[i - 1, :88]
      quats[i] = quats[i - 1]
      age += 0.02
      frames[i, 88:112] = 0
      frames[i, 112:] = min(age, 1.0)
      remaining -= 1
    else:
      age = 0.0
  return frames, quats


def build_human_goal(
  frames: torch.Tensor, root_quats: torch.Tensor, robot_quat: torch.Tensor
) -> torch.Tensor:
  """Frame-major history -> goal; all orientations relative to CURRENT robot.

  Inputs: (B,10,136), (B,10,4 wxyz), (B,4 wxyz). No clip or teacher required.
  """
  relative = quat_mul(quat_inv(robot_quat)[:, None].expand_as(root_quats), root_quats)
  rot6d = matrix_from_quat(relative)[..., :2].reshape(*frames.shape[:2], 6)
  return torch.cat([frames[..., :72], rot6d, frames[..., 78:]], dim=-1).flatten(1)


class HumanHistory:
  """Online history for already calibrated frames at 50 Hz.

  Sensor adapters own calibration, timestamp resampling, and last-valid holding.
  This buffer rejects out-of-order data and resets across gaps/recentering.
  """

  def __init__(self):
    self.samples = deque(maxlen=HISTORY)
    self.last_timestamp: float | None = None

  def reset(self):
    self.samples.clear()
    self.last_timestamp = None

  def push(self, frame: torch.Tensor, root_quat: torch.Tensor, timestamp: float):
    if frame.shape != (FRAME_DIM,) or root_quat.shape != (4,):
      raise ValueError("Expected calibrated frame[136] and root_quat[4 wxyz]")
    if not torch.isfinite(frame).all() or not torch.isfinite(root_quat).all():
      raise ValueError("Nonfinite human sample")
    if not math.isfinite(timestamp) or root_quat.norm() < 1e-8:
      raise ValueError("Invalid timestamp or root orientation")
    root_quat = root_quat / root_quat.norm()
    if self.last_timestamp is not None:
      dt = timestamp - self.last_timestamp
      if dt <= 0:
        raise ValueError("Human timestamps must increase")
      if dt > 0.03:
        self.reset()
      elif abs(dt - 0.02) > 0.002:
        raise ValueError("Resample the sensor stream to 50 Hz before buffering")
    if not self.samples:
      self.samples.extend([(frame.clone(), root_quat.clone())] * (HISTORY - 1))
    self.samples.append((frame.clone(), root_quat.clone()))
    self.last_timestamp = timestamp

  def goal(self, robot_quat: torch.Tensor) -> torch.Tensor:
    if not self.samples:
      raise RuntimeError("No human samples")
    frames, quats = zip(*self.samples, strict=True)
    return build_human_goal(
      torch.stack(frames)[None], torch.stack(quats)[None], robot_quat.reshape(1, 4)
    )


class HumanMotionTable:
  """One frame per motion frame; gather past windows lazily within each clip."""

  def __init__(
    self,
    motion,
    root: str,
    device: str,
    noise_std=0.0,
    dropout_prob=0.0,
    max_dropout=4,
    seed=42,
  ):
    arrays, quats, ends = [], [], []
    rng = np.random.default_rng(seed)
    total = 0
    for name in motion.motion_names:
      path = Path(root) / f"{name}.npz"
      with np.load(path) as clip:
        if str(clip["human_schema"]) != SCHEMA or float(clip["fps"]) != 50:
          raise ValueError(f"Unsupported human schema/fps: {path}")
        frames, orientations = augment_sensor(
          clip["human_frames"],
          clip["human_root_quat_w"],
          rng,
          noise_std,
          dropout_prob,
          max_dropout,
        )
        arrays.append(frames)
        quats.append(orientations)
        total += len(arrays[-1])
        ends.append(total)
    if ends != motion.motion_ends.cpu().tolist():
      raise ValueError("Human/robot clip lengths or ordering do not match")
    self.frames = torch.as_tensor(np.concatenate(arrays), device=device)
    self.quats = torch.as_tensor(np.concatenate(quats), device=device)
    starts = np.repeat([0, *ends[:-1]], np.diff([0, *ends]))
    self.starts = torch.as_tensor(starts, device=device)

  def extract(self, times: torch.Tensor, robot_quat: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(-HISTORY + 1, 1, device=times.device)
    indices = torch.maximum(times[:, None] + offsets, self.starts[times, None])
    return build_human_goal(self.frames[indices], self.quats[indices], robot_quat)


class StaleHumanSignal(RuntimeError):
  """The simulator must end/pause the episode instead of using an expired goal."""


class HumanStream:
  """Causal latest-sample resampling of calibrated packets onto a 50 Hz clock.

  Timestamps are capture times in the control clock's domain (seconds). No future
  interpolation or PICO dependency. Call tick even when no packet arrives. On
  expiry, the caller must stop the simulation episode; no fallback policy is
  assumed. Call reset explicitly on recentering or a new episode.
  """

  def __init__(self, max_age: float = 0.2):
    if not math.isfinite(max_age) or max_age <= 0:
      raise ValueError("max_age must be positive and finite")
    self.max_age = max_age
    self.history = HumanHistory()
    self.sample = None
    self.last_tick = None

  def reset(self):
    self.history.reset()
    self.sample = None
    self.last_tick = None

  def receive(self, frame: torch.Tensor, root_quat: torch.Tensor, timestamp: float):
    # Reuse validation without imposing a sensor frequency.
    validator = HumanHistory()
    validator.push(frame, root_quat, timestamp)
    if self.sample is not None and timestamp <= self.sample[2]:
      return False  # Late/duplicate packet; never rewind the causal stream.
    if (
      (frame[88:112] < 0).any() or (frame[88:112] > 1).any() or (frame[112:] < 0).any()
    ):
      raise ValueError("Invalid human validity/age channels")
    self.sample = (frame.clone(), root_quat.clone(), timestamp)
    return True

  def tick(self, timestamp: float):
    if not math.isfinite(timestamp):
      raise ValueError("Invalid control timestamp")
    if self.last_tick is not None and abs(timestamp - self.last_tick - 0.02) > 0.002:
      raise ValueError("Control ticks must advance at 50 Hz; reset after a pause")
    if self.sample is None:
      raise StaleHumanSignal("No human packet received")
    frame, quat, captured = self.sample
    elapsed = timestamp - captured
    if elapsed < -1e-6:
      raise ValueError("Packet capture timestamp is in the future")
    frame = frame.clone()
    frame[112:] += max(0.0, elapsed)
    # Conservative global timeout; partial tracking remains explicitly encoded.
    if float(frame[112:].min()) > self.max_age + 1e-6:
      raise StaleHumanSignal("Human signal exceeded max_age")
    if elapsed > 0.022:
      frame[88:112] = 0
    self.history.push(frame, quat, timestamp)
    self.last_tick = timestamp

  def goal(self, robot_quat: torch.Tensor, timestamp: float):
    # Require an explicit current tick; goal() cannot silently serve old history.
    if self.last_tick is None or abs(timestamp - self.last_tick) > 1e-6:
      raise StaleHumanSignal("Call tick at the current control timestamp first")
    frames, quats = zip(*self.history.samples, strict=True)
    batch = robot_quat.reshape(-1, 4).shape[0]
    return build_human_goal(
      torch.stack(frames)[None].expand(batch, -1, -1),
      torch.stack(quats)[None].expand(batch, -1, -1),
      robot_quat.reshape(-1, 4),
    )
