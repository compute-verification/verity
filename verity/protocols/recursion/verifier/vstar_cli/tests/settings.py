"""The environment variables vstar's difftests read, each declared once."""

from __future__ import annotations

import json
import os
from pathlib import Path


def recursive_data() -> str:
    """``VERITY_VSTAR_RECURSIVE_DATA``: a recursion_e2e run's kept data/TAG (KEEP_DATA=1, after STEP=lean), which the
    whole-session recursive tests read and skip without, or ``fetch``: the kept run `recursive_data.json` names, fetched
    from the evidence store when they run (about 2 GB); else empty. ``1`` is ``fetch``: a cold check sets an opt-in
    test's variable to 1 (check.py's `OPT_IN`)."""
    v = os.environ.get("VERITY_VSTAR_RECURSIVE_DATA", "")
    return "fetch" if v == "1" else v


def store_env() -> dict[str, str]:
    """The store CLI's environment. A recorded check's steps and a suite inside a `research run` carry no store config,
    only the run's custody key, which ``CHECK_STORE_CUSTODY`` names (check.py's "The evidence store"): it goes into that
    subprocess's environment alone, when no store config is set here, as Flock's `agreement/ci.py` hands it to its own."""
    env = dict(os.environ)
    custody = Path(env["CHECK_STORE_CUSTODY"]) if env.get("CHECK_STORE_CUSTODY") else None
    if (custody is None or env.get("RESEARCH_STORE_CONFIG") or (Path.home() / ".research" / "store.toml").is_file()
            or not ((custody / "store.toml").is_file() and (custody / "cred.json").is_file())):
        return env
    cred = json.loads((custody / "cred.json").read_text())
    env["RESEARCH_STORE_CONFIG"] = str(custody / "store.toml")
    env.update({k: cred[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN") if cred.get(k)})
    return env


def lean_jobs() -> int:
    """``LEAN_JOBS``: how many flock-verify processes a whole-session recursive test runs at once (default 4)."""
    return int(os.environ.get("LEAN_JOBS") or 4)


def flock_trace() -> Path | None:
    """``FLOCK_TRACE``, a build of ``flock-trace``; else the ``flock-trace`` beside ``FLOCK_CIRCUIT``'s ``flock-circuit``."""
    if os.environ.get("FLOCK_TRACE"):
        return Path(os.environ["FLOCK_TRACE"])
    if os.environ.get("FLOCK_CIRCUIT"):
        return Path(os.environ["FLOCK_CIRCUIT"]).with_name("flock-trace")
    return None


def irdep_tree() -> Path | None:
    """``VERITY_IRDEP_TREE``, a deployed recursive run's tree (`stage/`, `inner/s0/`, `v/`)."""
    return Path(os.environ["VERITY_IRDEP_TREE"]) if os.environ.get("VERITY_IRDEP_TREE") else None


def irdep_lean() -> Path | None:
    """``VERITY_IRDEP_LEAN``, the output of `IrDepSession.lean` at that tree."""
    return Path(os.environ["VERITY_IRDEP_LEAN"]) if os.environ.get("VERITY_IRDEP_LEAN") else None
