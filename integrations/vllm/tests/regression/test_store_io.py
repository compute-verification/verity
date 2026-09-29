"""`store_io`'s environment for the store CLI inside a `research run --custody-r2` job: the run's custody key reaches the remote when the
machine has no store config of its own (a pod), and nothing else changes (laptop-safe, no store)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.regression import store_io

KEY = {"AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "secret", "AWS_SESSION_TOKEN": "token", "expires_utc": "2026-09-29T09:00:00Z"}


@pytest.fixture
def pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A pod's research root with one custody run: `runs/<id>/` and `requests/<id>/custody/{store.toml,cred.json}`; an empty home."""
    root = tmp_path / "research"
    (root / "runs" / "r1").mkdir(parents=True)
    custody = root / "requests" / "r1" / "custody"
    custody.mkdir(parents=True)
    (custody / "store.toml").write_text('[remote]\ntype = "s3"\nendpoint = "https://r2.example"\nbucket = "b"\n')
    (custody / "cred.json").write_text(json.dumps(KEY))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(root / "runs" / "r1"))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    for k in ("RESEARCH_STORE_CONFIG", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "CHECK_STORE_CUSTODY",
              "VERITY_STORE_REQUIRED"):
        monkeypatch.delenv(k, raising=False)
    return custody


def test_a_custody_run_without_store_config_uses_its_key(pod: Path) -> None:
    env = store_io._env()
    assert env["RESEARCH_STORE_CONFIG"] == str(pod / "store.toml")
    assert {k: env[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")} == {
        "AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "secret", "AWS_SESSION_TOKEN": "token"}
    assert "expires_utc" not in env


def test_an_explicit_store_config_wins(pod: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", "/etc/store.toml")
    env = store_io._env()
    assert env["RESEARCH_STORE_CONFIG"] == "/etc/store.toml" and "AWS_SESSION_TOKEN" not in env


def test_a_machine_with_its_own_store_config_is_left_alone(pod: Path) -> None:
    home = Path(store_io.os.environ["HOME"])
    (home / ".research").mkdir()
    (home / ".research" / "store.toml").write_text("")
    env = store_io._env()
    assert "RESEARCH_STORE_CONFIG" not in env and "AWS_SESSION_TOKEN" not in env


@pytest.mark.parametrize("missing", ["cred.json", "store.toml"])
def test_no_key_no_change(pod: Path, missing: str) -> None:
    (pod / missing).unlink()
    env = store_io._env()
    assert "RESEARCH_STORE_CONFIG" not in env and "AWS_SESSION_TOKEN" not in env


def test_outside_a_run_no_change(pod: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_RUN_ID")
    assert "RESEARCH_STORE_CONFIG" not in store_io._env()


def test_inside_checks_steps_the_custody_dir_comes_by_name(pod: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`check.py` drops RESEARCH_* from its steps and names the custody dir in CHECK_STORE_CUSTODY instead."""
    monkeypatch.delenv("RESEARCH_RUN_ID")
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    monkeypatch.setenv("CHECK_STORE_CUSTODY", str(pod))
    env = store_io._env()
    assert env["RESEARCH_STORE_CONFIG"] == str(pod / "store.toml") and env["AWS_SESSION_TOKEN"] == "token"


def test_no_remote_is_unreachable_outside_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A laptop or dev pod with a local store and no remote: the store-backed tests skip."""
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    for k in ("RESEARCH_STORE_CONFIG", "RESEARCH_RUN_ID", "RESEARCH_RUN_DIR", "CHECK_STORE_CUSTODY", "R2_ENDPOINT", "R2_BUCKET",
              "VERITY_STORE_REQUIRED"):
        monkeypatch.delenv(k, raising=False)
    assert store_io.reachable(need_remote=False)
    assert not store_io.reachable()
    monkeypatch.setenv("VERITY_STORE_REQUIRED", "1")
    assert store_io.reachable(), "a recorded check never skips the store-backed tests"
