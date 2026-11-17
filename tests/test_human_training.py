"""Causality, preservation and deployment contracts for human adaptation."""

from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from supermimic.data.human import SCHEMA, HumanHistory, HumanMotionTable
from supermimic.data.prepare_seed import family_id, split_for, wrists_from_smpl
from supermimic.learners.learner_utils import build_optimizers
from supermimic.models import MultiEncoderAE
from supermimic.runners.exporter import HumanActor, export_human_actor
from supermimic.scripts.train_yaml import merge_config
from supermimic.tasks.task_spec import TaskSpec


def make_actor():
  return MultiEncoderAE(
    obs_dim=68,
    action_dim=29,
    task_specs=[
      TaskSpec("motion", 640, lambda *a: None),
      TaskSpec("human_body", 1360, lambda *a: None),
    ],
    slot_mix={"motion": 0.0, "human_body": 1.0},
    latent_dim=32,
    hidden_dims=(32, 16),
    latent_bc_weight=0.1,
    history_length=1,
  )


def test_causal_windows_do_not_cross_clips_and_match_online(tmp_path):
  for name, values in [("a", [1, 2, 3]), ("b", [101, 102, 103])]:
    frame = np.zeros((3, 136), np.float32)
    frame[:, 0] = values
    np.savez(
      tmp_path / f"{name}.npz",
      human_frames=frame,
      human_root_quat_w=np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32),
      human_schema=SCHEMA,
      fps=50,
    )
  motion = SimpleNamespace(motion_names=["a", "b"], motion_ends=torch.tensor([3, 6]))
  table = HumanMotionTable(motion, str(tmp_path), "cpu")
  robot = torch.tensor([[1.0, 0, 0, 0]])
  goal = table.extract(torch.tensor([4]), robot).reshape(10, 136)
  assert goal[:, 0].tolist() == [101.0] * 9 + [102.0]
  before = table.extract(torch.tensor([4]), robot)
  table.frames[5, 0] = -999
  torch.testing.assert_close(table.extract(torch.tensor([4]), robot), before)
  stream = HumanHistory()
  for i in [3, 4]:
    stream.push(table.frames[i], table.quats[i], (i - 3) * 0.02)
  torch.testing.assert_close(stream.goal(robot), before)
  with pytest.raises(ValueError, match="increase"):
    stream.push(table.frames[4], table.quats[4], 0.02)


def test_history_orientation_uses_current_robot_and_retains_gradient(tmp_path):
  frame = np.zeros((2, 136), np.float32)
  q = np.tile([1.0, 0, 0, 0], (2, 1)).astype(np.float32)
  np.savez(
    tmp_path / "a.npz",
    human_frames=frame,
    human_root_quat_w=q,
    human_schema=SCHEMA,
    fps=50,
  )
  motion = SimpleNamespace(motion_names=["a"], motion_ends=torch.tensor([2]))
  table = HumanMotionTable(motion, str(tmp_path), "cpu")
  robot = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], requires_grad=True)
  goal = table.extract(torch.tensor([1]), robot).reshape(10, 136)
  torch.testing.assert_close(
    goal[:, 72:78],
    torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]).expand(10, -1),
    atol=1e-6,
    rtol=0,
  )
  goal[:, 72].sum().backward()
  assert robot.grad is not None and robot.grad.abs().sum() > 0


def test_adaptation_keeps_original_weights_statistics_and_actions():
  torch.manual_seed(42)
  actor = make_actor()
  actor.action_bc_weight = 1.0
  actor.freeze_for_human_adaptation()
  obs = TensorDict(
    dict(
      history=torch.randn(8, 1, 68),
      goal_motion=torch.randn(8, 640),
      goal_human_body=torch.randn(8, 1360),
    ),
    [8],
  )
  original = {k: v.clone() for k, v in actor.state_dict().items()}
  actor.set_slot_mix({"motion": 1.0})
  actor.eval()
  motion_before = actor(obs).detach()
  actor.set_slot_mix({"human_body": 1.0})
  actor.train()
  actor.update_normalization(obs)
  optimizers = build_optimizers([actor], 0.001, device="cpu")
  params = {
    id(p) for opt in optimizers for group in opt.param_groups for p in group["params"]
  }
  assert params == {id(p) for p in actor.encoders["human_body"].parameters()}
  loss = actor(obs).square().mean() + actor.auxiliary_loss().mean()
  loss.backward()
  for opt in optimizers:
    opt.step()
  changed = [
    k for k, v in actor.state_dict().items() if not torch.equal(v, original[k])
  ]
  assert any(k.startswith("encoders.human_body.") for k in changed)
  assert all(
    k.startswith(("encoders.human_body.", "goal_normalizers.human_body."))
    for k in changed
  )
  actor.eval()
  actor.set_slot_mix({"motion": 1.0})
  torch.testing.assert_close(actor(obs), motion_before, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 3])
def test_human_onnx_matches_pytorch_without_motion_inputs(tmp_path, batch):
  actor = make_actor().eval()
  path = export_human_actor(actor, str(tmp_path))
  graph = onnx.load(path)
  assert [i.name for i in graph.graph.input] == ["goal_human", "history_robot"]
  assert all("motion" not in item.name for item in graph.graph.initializer)
  inputs = {
    "goal_human": np.random.default_rng(42)
    .normal(size=(batch, 1360))
    .astype(np.float32),
    "history_robot": np.ones((batch, 1, 68), np.float32),
  }
  expected = (
    HumanActor(actor)(*(torch.from_numpy(x) for x in inputs.values())).detach().numpy()
  )
  actual = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
    None, inputs
  )[0]
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_split_groups_actors_takes_and_mirrors():
  names = [
    "walk_forward_001__A001",
    "walk_forward_001__A001_M",
    "walk_forward_002__A002",
  ]
  assert len({family_id(n) for n in names}) == 1
  assert len({split_for(n, 0.2, 42) for n in names}) == 1


def test_wrist_mapping_handles_identity_and_elbow_twist():
  pose = np.zeros((2, 24, 3))
  pose[1, 18:20, 1] = 0.4
  np.testing.assert_allclose(wrists_from_smpl(pose), 0, atol=1e-7)


def test_yaml_rejects_unknown_runner_key():
  from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

  with pytest.raises(ValueError, match="Unknown config key"):
    merge_config(SuperMimicRunnerCfg(), {"polciy": {"batch_size": 8}})


def test_dropout_holds_estimates_and_reports_age():
  from supermimic.data.human import augment_sensor

  frames = np.zeros((4, 136), np.float32)
  frames[:, 0] = [1, 2, 3, 4]
  frames[:, 88:112] = 1
  quats = np.eye(4, dtype=np.float32)
  held, orientations = augment_sensor(
    frames, quats, np.random.default_rng(42), 0.0, 1.0, 2
  )
  np.testing.assert_array_equal(held[:, 0], [1, 1, 1, 1])
  np.testing.assert_array_equal(held[1:, 88:112], 0)
  np.testing.assert_allclose(held[:, 112], [0, 0.02, 0.04, 0.06])
  np.testing.assert_array_equal(orientations, np.tile(quats[0], (4, 1)))
  np.testing.assert_array_equal(frames[:, 0], [1, 2, 3, 4])


def test_preprocess_resamples_120_to_50_and_uses_only_initial_floor(tmp_path):
  import joblib
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

  from supermimic.data.prepare_seed import convert_pair

  model = get_spec().compile()
  names = [model.joint(i).name + "_dof" for i in range(1, model.njnt)]
  robot = dict(
    joint_pos=np.zeros((121, 29)),
    body_pos_w=np.zeros((121, 1, 3)),
    body_quat_w=np.tile([1.0, 0, 0, 0], (121, 1, 1)),
  )
  robot["body_pos_w"][:, 0, 0] = np.arange(121) / 120
  robot["body_pos_w"][:, 0, 2] = 1
  reader = SimpleNamespace(load_bones_csv=lambda _: robot, BONES_CSV_JOINT_NAMES=names)
  args = SimpleNamespace(
    source_fps=120,
    duration_tolerance=0.1,
    min_seconds=1.0,
    joint_limit_tolerance=0.3,
    max_penetration=10.0,
    max_support_height=10.0,
    max_floating_fraction=1.0,
    human_scale=0.8,
    velocity_alpha=0.25,
  )
  joints = np.zeros((51, 24, 3))
  joints[:, [7, 8, 10, 11], 2] = -1
  trans = np.zeros((51, 3))
  trans[:, 1] = 1
  smpl = dict(pose_aa=np.zeros((51, 72)), smpl_joints=joints, transl=trans, fps=50.0)
  path = tmp_path / "smpl.pkl"
  joblib.dump(smpl, path)
  result, details = convert_pair("dummy", path, reader, model, args)
  assert details["frames"] == 51
  np.testing.assert_allclose(result["root_pos_w"][:, 0], np.arange(51) / 50, atol=1e-6)
  np.testing.assert_allclose(result["root_lin_vel_w"][1:, 0], 1, atol=1e-6)
  np.testing.assert_allclose(result["human_frames"][:, 84], 0.8, atol=1e-6)
  smpl["smpl_joints"][-1, :, 2] -= 100
  joblib.dump(smpl, path)
  changed, _ = convert_pair("dummy", path, reader, model, args)
  np.testing.assert_array_equal(
    changed["human_frames"][:-1], result["human_frames"][:-1]
  )


def test_stream_resamples_irregular_packets_and_expires_without_receives():
  from supermimic.data.human import HumanStream, StaleHumanSignal

  stream = HumanStream(max_age=0.05)
  frame = torch.zeros(136)
  frame[88:112] = 1
  quat = torch.tensor([1.0, 0, 0, 0])
  stream.receive(frame, quat, 0)
  stream.tick(0)
  frame[0] = 7
  assert stream.receive(frame, quat, 0.011)
  assert not stream.receive(frame, quat, 0.005)
  stream.tick(0.02)
  goal = stream.goal(quat[None], 0.02).reshape(10, 136)
  assert goal[-1, 0] == 7
  assert goal[-1, 112].item() == pytest.approx(0.009)
  with pytest.raises(StaleHumanSignal):
    stream.goal(quat[None], 0.04)
  stream.tick(0.04)
  assert stream.goal(quat[None], 0.04).reshape(10, 136)[-1, 88] == 0
  stream.tick(0.06)
  with pytest.raises(StaleHumanSignal):
    stream.tick(0.08)
  stream.reset()
  with pytest.raises(StaleHumanSignal):
    stream.tick(1)


def test_stream_zero_delay_matches_offline_and_resets_history(tmp_path):
  from supermimic.data.human import HumanStream

  frames = np.zeros((12, 136), np.float32)
  frames[:, 0] = np.arange(12)
  frames[:, 88:112] = 1
  quats = np.tile([1, 0, 0, 0], (12, 1)).astype(np.float32)
  np.savez(
    tmp_path / "a.npz",
    human_frames=frames,
    human_root_quat_w=quats,
    human_schema=SCHEMA,
    fps=50,
  )
  motion = SimpleNamespace(motion_names=["a"], motion_ends=torch.tensor([12]))
  table = HumanMotionTable(motion, str(tmp_path), "cpu")
  stream = HumanStream()
  robot = torch.tensor([[1.0, 0, 0, 0]])
  for i in range(12):
    stream.receive(table.frames[i], table.quats[i], i * 0.02)
    stream.tick(i * 0.02)
    torch.testing.assert_close(
      stream.goal(robot, i * 0.02), table.extract(torch.tensor([i]), robot)
    )
  stream.reset()
  stream.receive(table.frames[0], table.quats[0], 2)
  stream.tick(2)
  assert stream.goal(robot, 2).reshape(10, 136)[:, 0].sum() == 0


def test_bc_schedule_zero_removes_auxiliary_gradient():
  actor = make_actor().train()
  actor.action_bc_weight = 1
  obs = TensorDict(
    dict(
      history=torch.randn(4, 1, 68),
      goal_motion=torch.randn(4, 640),
      goal_human_body=torch.randn(4, 1360),
    ),
    [4],
  )
  actor(obs)
  assert actor.auxiliary_loss().sum() > 0
  actor.bc_scale.zero_()
  actor(obs)
  actor.auxiliary_loss().sum().backward()
  assert all(
    p.grad is None or torch.count_nonzero(p.grad) == 0 for p in actor.parameters()
  )
  assert "bc_scale" not in actor.state_dict()  # old checkpoints stay compatible


def test_stages_and_initial_support_validation():
  from supermimic.data.prepare_seed import initial_support
  from supermimic.scripts.train_yaml import apply_stage

  cfg = {"runner": {"max_iterations": 100}}
  apply_stage(cfg, "baseline", "decay")
  assert cfg["runner"]["tasks"]["human_body"] is False
  assert cfg["runner"]["action_bc_weight"] == 0
  apply_stage(cfg, "adaptation", "decay", "checkpoint")
  assert cfg["runner"]["human_adaptation"]
  assert cfg["runner"]["slot_mix"]["human_body"] == 1
  assert cfg["runner"]["bc_decay_iterations"] == 50
  apply_stage(cfg, "adaptation", "off", "checkpoint")
  assert cfg["runner"]["action_bc_weight"] == 0
  assert cfg["runner"]["latent_bc_weight"] == 0
  feet = np.zeros((10, 4))
  initial_support(feet, 50)
  feet[:] = np.arange(10)[:, None] * 0.03
  with pytest.raises(ValueError, match="support"):
    initial_support(feet, 50)


def test_evaluation_outcomes_and_phase_diagnostic():
  from supermimic.scripts.audit_dataset import lag_diagnostic
  from supermimic.scripts.evaluate_human import episode_outcome

  assert episode_outcome(True, True, True) == "failure"
  assert episode_outcome(False, True, True) == "completed"
  assert episode_outcome(False, False, True) == "time_limit"
  assert episode_outcome(False, False, False) == "step_limit"
  assert lag_diagnostic(np.ones(100), np.ones(100)) == (None, None)
  rng = np.random.default_rng(42)
  signal = rng.normal(size=200)
  lag, correlation = lag_diagnostic(signal, np.roll(signal, 3))
  assert lag == 3 and correlation == pytest.approx(1)
