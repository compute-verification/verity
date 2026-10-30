"""The evidence store from inside a recorded run: its custody key, handed to the store CLI's environment only.

A run launched with ``research run --on`` carries no store credentials; its custody dir (``<root>/requests/<run id>/custody/``,
beside ``<root>/runs/<run id>/``) holds a delete-free key while it runs. As Flock's ``verifier/ci.py`` and
``tools/verity/check/check.py`` do, the key goes into the ``research data`` subprocess's environment and nowhere else.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RESEARCH_SRC = Path(__file__).resolve().parents[3] / "tools" / "research" / "src"


def custody_dir(environ: dict[str, str] | None = None) -> Path | None:
    env = os.environ if environ is None else environ
    run_dir, run_id = env.get("RESEARCH_RUN_DIR"), env.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return None
    d = Path(run_dir).resolve().parent.parent / "requests" / run_id / "custody"
    return d if (d / "store.toml").is_file() and (d / "cred.json").is_file() else None


def store_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The store CLI's environment: this one, with the run's custody key when no store is configured here."""
    env = dict(os.environ if environ is None else environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(RESEARCH_SRC), env.get("PYTHONPATH")) if p)
    d = custody_dir(env)
    if d is None or env.get("RESEARCH_STORE_CONFIG") or (Path.home() / ".research" / "store.toml").is_file():
        return env
    cred = json.loads((d / "cred.json").read_text())
    env["RESEARCH_STORE_CONFIG"] = str(d / "store.toml")
    env.update({k: cred[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN") if cred.get(k)})
    return env


def research_data(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "research", "data", *args], capture_output=True, text=True, env=store_env(),
                          check=False)
