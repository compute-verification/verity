"""A cloud VM's long-lived shell can lose the Cursor secrets a fresh tmux login shell still has: the commands that need them say so,
and fail, instead of reading as a machine that was never configured; everywhere else they behave as before."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from research import secret_hint
from research.pods import runpod
from research.store.cli import main
from research.store.local import LocalStore, StoreError
from research.store.remote import ConfigError, load_remote

INJECTED = "AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,R2_ENDPOINT,R2_BUCKET,RUNPOD_SSH_KEY_B64,NEBIUS_SA_ID"
HINT = "rerun from a fresh tmux login shell"


def test_only_a_name_the_vm_injects_and_this_shell_lacks_is_lost(monkeypatch):
    assert secret_hint.lost("R2_ENDPOINT") == ""                        # no injected list: a laptop, a pod, a node
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", INJECTED)
    monkeypatch.setenv("R2_ENDPOINT", "https://r2.invalid")
    assert secret_hint.lost("R2_ENDPOINT", "SLACK_BOT_TOKEN") == ""     # present here, or never injected on this VM
    assert secret_hint.lost("R2_ENDPOINT", "R2_BUCKET") == f"R2_BUCKET not in this shell though this VM injects it: {HINT}"


def test_a_label_or_put_in_a_shell_that_lost_the_store_secrets_fails_with_the_hint(tmp_path, monkeypatch, capsys):
    f = tmp_path / "a.bin"
    f.write_bytes(b"a")
    art = LocalStore().put_artifact("fixture/v1", {"n": 1}, file=f)
    assert main(["label", art, "note", "x", "--by", "x", "--off-vocab"]) == 0
    assert "local only (write-through off)" in capsys.readouterr().err
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", INJECTED)
    assert main(["label", art, "note", "y", "--by", "x", "--off-vocab"]) == 1
    assert f"NOT preserved (R2_ENDPOINT, R2_BUCKET not in this shell though this VM injects them: {HINT})" in capsys.readouterr().err
    assert main(["put", "--kind", "fixture/v1", "--meta", '{"n": 2}', "--file", str(f)]) == 1
    assert HINT in capsys.readouterr().err
    with pytest.raises(StoreError, match=HINT):
        LocalStore()._require_remote()
    cfg = tmp_path / "store.toml"
    cfg.write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://r2.invalid"\n'
                   'access_key_id_env = "AWS_ACCESS_KEY_ID"\nsecret_access_key_env = "AWS_SECRET_ACCESS_KEY"\n')
    with pytest.raises(ConfigError, match=f"are not set \\(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY not in this shell.*{HINT}"):
        load_remote(cfg)


def test_node_ssh_in_a_shell_that_lost_the_key_fails_with_the_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(runpod, "SSH_KEY", tmp_path / "absent-key")
    assert runpod.ssh_key() == runpod.SSH_KEY
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", INJECTED)
    with pytest.raises(runpod.PodError, match=f"RUNPOD_SSH_KEY_B64 not in this shell though this VM injects it: {HINT}"):
        runpod.ssh_command("192.0.2.1", 22)


def test_nebius_auth_in_a_shell_that_lost_its_key_says_so():
    common = Path(runpod.__file__).parent / "nebius" / "common.sh"
    env = {"PATH": os.environ["PATH"], "NEBIUS_BIN": "/bin/true"}
    r = subprocess.run(["bash", "-c", f'. "{common}"; nebius_auth'], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 1 and "NEBIUS_SA_ID is not set (a Cursor secret" in r.stderr
    env["CLOUD_AGENT_INJECTED_SECRET_NAMES"] = INJECTED
    r = subprocess.run(["bash", "-c", f'. "{common}"; nebius_auth'], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 1 and f"NEBIUS_SA_ID not in this shell though this VM injects it: {HINT}" in r.stderr
