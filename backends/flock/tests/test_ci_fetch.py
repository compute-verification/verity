"""`ci.fetch` reaches the evidence store inside a recorded check. `check`'s steps carry no store credentials, only the run's
custody dir (`$CHECK_STORE_CUSTODY`), so the store CLI is handed that key, as integrations/vllm's `store_io` is. Without it the
Lean tests that fetch their stages through `ci.fetch` skipped in every recorded check (#757's red team)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

VERIFIER = Path(__file__).resolve().parents[1] / "verifier"


@pytest.fixture
def ci(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(VERIFIER))
    import ci as mod  # noqa: PLC0415

    for k in ("RESEARCH_STORE_CONFIG", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "CHECK_STORE_CUSTODY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return mod


def _custody(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "store.toml").write_text("[remote]\n")
    (d / "cred.json").write_text(json.dumps({"AWS_ACCESS_KEY_ID": "dummy-id", "AWS_SECRET_ACCESS_KEY": "dummy",
                                             "AWS_SESSION_TOKEN": "dummy-token"}))
    return d


def test_fetch_hands_the_store_cli_the_checks_custody_key(ci, monkeypatch, tmp_path):
    seen = []

    def run(argv, **kw):
        seen.append((argv, kw["env"]))
        to = Path(argv[argv.index("--to") + 1])
        to.mkdir(parents=True)
        (to / "f.bin").write_bytes(b"x")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ci.subprocess, "run", run)
    custody = _custody(tmp_path / "custody")
    monkeypatch.setenv("CHECK_STORE_CUSTODY", str(custody))
    assert ci.fetch("art:0123abcd", tmp_path / "store") == tmp_path / "store" / "0123abcd" / "f.bin"
    (argv, env), = seen
    assert argv[-4:] == ["fetch", "art:0123abcd", "--to", str(tmp_path / "store" / "0123abcd")]
    assert env["RESEARCH_STORE_CONFIG"] == str(custody / "store.toml")
    assert (env["AWS_ACCESS_KEY_ID"], env["AWS_SECRET_ACCESS_KEY"], env["AWS_SESSION_TOKEN"]) == ("dummy-id", "dummy", "dummy-token")


def test_the_custody_key_goes_only_where_no_store_is_configured(ci, monkeypatch, tmp_path):
    custody = _custody(tmp_path / "custody")
    assert "RESEARCH_STORE_CONFIG" not in ci.store_env()                       # outside a recorded check
    monkeypatch.setenv("CHECK_STORE_CUSTODY", str(tmp_path / "no-key"))
    assert "RESEARCH_STORE_CONFIG" not in ci.store_env()                       # a custody dir that holds no key
    monkeypatch.setenv("CHECK_STORE_CUSTODY", str(custody))
    (tmp_path / "home" / ".research").mkdir(parents=True)
    (tmp_path / "home" / ".research" / "store.toml").write_text("[remote]\n")
    assert "RESEARCH_STORE_CONFIG" not in ci.store_env()                       # this machine's own store config
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", "/elsewhere/store.toml")
    env = ci.store_env()
    assert env["RESEARCH_STORE_CONFIG"] == "/elsewhere/store.toml" and "AWS_ACCESS_KEY_ID" not in env
