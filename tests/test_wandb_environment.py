"""W&B destination and credential source precedence without network access."""

import os
import sys
from types import SimpleNamespace

import pytest

from supermimic.scripts.train_yaml import configure_wandb_environment, main


@pytest.fixture(autouse=True)
def isolated_wandb_environment(monkeypatch, tmp_path):
  monkeypatch.chdir(tmp_path)
  for key in (
    "WANDB_MODE",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
    "WANDB_APP_URL",
    "WANDB_API_KEY",
  ):
    monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("key", ["WANDB_MODE", "WANDB_ENTITY", "WANDB_BASE_URL"])
def test_wandb_environment_precedence(monkeypatch, tmp_path, key):
  monkeypatch.chdir(tmp_path)
  monkeypatch.delenv(key, raising=False)
  runtime = {key.lower(): "recipe"}
  configure_wandb_environment(runtime)
  assert os.environ[key] == "recipe"

  monkeypatch.delenv(key)
  (tmp_path / ".env").write_text(f"{key}=dotenv\n")
  configure_wandb_environment(runtime)
  assert os.environ[key] == "dotenv"

  monkeypatch.delenv(key)
  (tmp_path / ".env.local").write_text(f"{key}=local\n")
  configure_wandb_environment(runtime)
  assert os.environ[key] == "local"

  monkeypatch.setenv(key, "shell")
  configure_wandb_environment(runtime)
  assert os.environ[key] == "shell"


def test_wandb_api_key_is_loaded_without_printing(monkeypatch, tmp_path, capsys):
  monkeypatch.chdir(tmp_path)
  monkeypatch.delenv("WANDB_API_KEY", raising=False)
  (tmp_path / ".env.local").write_text("WANDB_API_KEY=test-key\n")
  configure_wandb_environment({})
  assert os.environ["WANDB_API_KEY"] == "test-key"
  assert capsys.readouterr().out == ""


def test_forge_run_links():
  import wandb

  configure_wandb_environment({"wandb_base_url": "https://api.forge.coreweave.com"})
  settings = wandb.Settings()
  settings.update_from_env_vars(os.environ)
  assert settings.app_url == "https://forge.coreweave.com/wandb"


def test_online_without_credentials_fails_before_simulator(monkeypatch, tmp_path):
  config = tmp_path / "train.yaml"
  config.write_text("runner: {}\nruntime: {wandb_mode: online}\n")
  monkeypatch.setattr(sys, "argv", ["train_yaml", "--config", str(config)])
  monkeypatch.setitem(
    sys.modules,
    "wandb",
    SimpleNamespace(
      login=lambda *, prompt: False,
      Settings=lambda: SimpleNamespace(base_url="https://api.forge.coreweave.com"),
    ),
  )
  with pytest.raises(RuntimeError, match="W&B online requiere credenciales"):
    main()
  assert os.environ["WANDB_MODE"] == "online"
