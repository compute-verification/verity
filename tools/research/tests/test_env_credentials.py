"""Credentials from the environment where the laptop's files are absent (a cloud VM with only the Cursor secrets); a file that
exists always wins, so the laptop's choice and behaviour are unchanged."""
from __future__ import annotations

import base64
import hashlib
import re
import subprocess
import threading
import tomllib
from pathlib import Path

import pytest

from research.pods import runpod
from research.store import remote_s3
from research.store import remote
from research.store.remote import POD_CONFIG, ConfigError, load_remote

BUCKET = "store-bucket"                                            # a dummy: the tracked tree never names the store's bucket


@pytest.fixture
def rp(tmp_path, monkeypatch):
    """runpod's laptop paths under a tmp home that has none of the files yet."""
    home = tmp_path / "home"
    monkeypatch.setattr(runpod, "CONFIG", home / ".runpod" / "config.toml")
    monkeypatch.setattr(runpod, "SSH_KEY", home / ".runpod" / "ssh" / "runpodctl-ssh-key")
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", home / ".runpod" / "ssh" / "veritor-campaign-known_hosts")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return home


def _keypair(d: Path) -> tuple[bytes, str]:
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(d / "k")], check=True)
    return (d / "k").read_bytes(), (d / "k.pub").read_text().strip()


def test_api_key_file_wins_then_env_then_a_refusal(rp, monkeypatch):
    with pytest.raises(runpod.PodError, match="RUNPOD_API_KEY is not set"):
        runpod._config()
    monkeypatch.setenv("RUNPOD_API_KEY", "env-key")
    assert runpod._config() == {"apikey": "env-key", "apiurl": runpod.GRAPHQL}
    runpod.CONFIG.parent.mkdir(parents=True)
    runpod.CONFIG.write_text('apikey = "file-key"\napiurl = "https://example.invalid/graphql"\n')
    assert runpod._config() == {"apikey": "file-key", "apiurl": "https://example.invalid/graphql"}


def test_ssh_key_from_env_is_written_once_mode_600_and_the_laptop_file_wins(rp, tmp_path, monkeypatch):
    assert runpod.ssh_key() == runpod.SSH_KEY                    # neither: ssh names the missing laptop file, as before
    priv, pub = _keypair(tmp_path / "gen")
    monkeypatch.setenv("RUNPOD_SSH_KEY_B64", base64.b64encode(priv.rstrip(b"\n")).decode())
    k = runpod.ssh_key()
    assert k.parent == tmp_path / "state" / "research" / "runpod" and k.read_bytes() == priv
    assert k.stat().st_mode & 0o777 == 0o600
    mtime = k.stat().st_mtime_ns
    assert runpod.ssh_key() == k and k.stat().st_mtime_ns == mtime
    assert runpod.public_key() == pub
    argv = runpod.ssh_command("1.2.3.4", 2222)
    assert argv[argv.index("-i") + 1] == str(k)
    assert f"UserKnownHostsFile={k.parent / 'known_hosts'}" in argv
    assert base64.b64encode(priv).decode()[:40] not in " ".join(argv)

    runpod.SSH_KEY.parent.mkdir(parents=True)
    runpod.SSH_KEY.write_bytes(b"laptop key\n")
    runpod.SSH_KEY.with_suffix(".pub").write_text("ssh-ed25519 LAPTOP\n")
    argv = runpod.ssh_command("1.2.3.4", 2222)
    assert argv[argv.index("-i") + 1] == str(runpod.SSH_KEY)
    assert f"UserKnownHostsFile={runpod.KNOWN_HOSTS}" in argv and runpod.public_key() == "ssh-ed25519 LAPTOP"


def test_threads_writing_the_key_at_once_each_get_the_whole_file(rp, tmp_path, monkeypatch):
    # `research doctor` resolves ssh argv in several probe threads at once: with one temp name per process, on a fresh state dir
    # two threads failed 481 of 500 times (FileNotFoundError from os.replace) and three read zero-length keys
    priv = b"dummy-ssh-key " * 40 + b"\n"
    monkeypatch.setenv("RUNPOD_SSH_KEY_B64", base64.b64encode(priv).decode())
    for i in range(20):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / f"state-{i}"))
        start, got, failed = threading.Barrier(4), [], []

        def resolve():
            start.wait()
            try:
                k = runpod.ssh_key()
                got.append((k, k.read_bytes()))
            except OSError as e:
                failed.append(e)
        threads = [threading.Thread(target=resolve) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert failed == [] and len({k for k, _ in got}) == 1 and all(b == priv for _, b in got)
        assert [p.name for p in got[0][0].parent.iterdir()] == [got[0][0].name]       # no temp file left behind


def test_a_malformed_key_secret_is_refused_without_echoing_it(rp, monkeypatch):
    monkeypatch.setenv("RUNPOD_SSH_KEY_B64", "not*base64*secretvalue")
    with pytest.raises(runpod.PodError) as e:
        runpod.ssh_key()
    assert "secretvalue" not in str(e.value)


@pytest.fixture
def r2(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.delenv("RESEARCH_STORE_CONFIG")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    pod = tmp_path / "store.pod.toml"
    pod.write_text(POD_CONFIG.read_text().replace(tomllib.loads(POD_CONFIG.read_text())["remote"]["bucket_sha256"],
                                                  hashlib.sha256(BUCKET.encode()).hexdigest()))
    monkeypatch.setattr(remote, "POD_CONFIG", pod)                 # pins the dummy's digest, as the tracked file pins the store's
    return tmp_path / "home" / ".research" / "store.toml"


def test_store_remote_from_env_when_no_store_toml(r2, monkeypatch):
    assert load_remote() is None                                  # no file and no R2_* at all: no remote, as before
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    with pytest.raises(ConfigError, match="R2_BUCKET is not set"):
        load_remote()
    monkeypatch.setenv("R2_BUCKET", "some-other-bucket")
    with pytest.raises(ConfigError, match="not the research store's bucket"):
        load_remote()
    monkeypatch.setenv("R2_BUCKET", BUCKET)
    r = load_remote()
    assert isinstance(r, remote_s3.S3Remote) and r.bucket == BUCKET and r.host == "acct.r2.cloudflarestorage.com"
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
    with pytest.raises(ConfigError, match="AWS_SECRET_ACCESS_KEY"):
        load_remote()


def test_store_toml_wins_and_an_explicit_config_path_is_never_replaced_by_env(r2, tmp_path, monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_BUCKET", BUCKET)
    r2.parent.mkdir(parents=True)
    r2.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "fs"}"\n')
    assert load_remote().describe().startswith("fs")               # the laptop's file, not the env
    r2.unlink()
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    assert load_remote() is None
    assert load_remote(tmp_path / "absent.toml") is None


def test_the_tracked_pod_config_pins_the_bucket_by_digest_and_names_neither_bucket_nor_endpoint():
    assert POD_CONFIG == Path(__file__).resolve().parents[1] / "store.pod.toml"
    r = tomllib.loads(POD_CONFIG.read_text())["remote"]
    assert re.fullmatch(r"[0-9a-f]{64}", r["bucket_sha256"])
    assert "bucket" not in r and "endpoint" not in r and (r["bucket_env"], r["endpoint_env"]) == ("R2_BUCKET", "R2_ENDPOINT")
    assert "r2.cloudflarestorage.com" not in POD_CONFIG.read_text()


def test_the_pod_config_takes_bucket_and_endpoint_from_the_environment_and_checks_the_digest(r2, monkeypatch):
    pod = remote.POD_CONFIG
    monkeypatch.delenv("R2_BUCKET", raising=False)
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    with pytest.raises(ConfigError, match="bucket_env names R2_BUCKET, which is not set"):
        load_remote(pod)
    monkeypatch.setenv("R2_BUCKET", "some-other-bucket")
    with pytest.raises(ConfigError, match="not the one .remote. bucket_sha256 pins"):
        load_remote(pod)
    monkeypatch.setenv("R2_BUCKET", BUCKET)
    r = load_remote(pod)
    assert isinstance(r, remote_s3.S3Remote) and r.bucket == BUCKET and r.host == "acct.r2.cloudflarestorage.com"
