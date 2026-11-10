"""The auditor's challenger for a `flock-circuit serve` process in the sweeps' own loopback and verifier-pod runs: the pods'
helper (`pod/challenger.sh`, whose Lean the pod builds before the run) driven from Python. Under FLOCK_CHALLENGER=1 every
server takes its sessions' coin trees from its own challenger (`serve --tree-from`), so it is bounded (`--sessions`, or a
`--queue` that ends): the challenger's session closes after that server's last, and the challenger exits 0 once it closes.
``end`` then holds the server's session records to the challenger's release log (`challenger_audit`: Python's
`sessions_tree_reason` and the Lean's `challenger_session_trees`). The challenger's key and secret never enter a caller's
tree (`Challenger.work`)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

#: `challenger_start`, `challenger_end` and `challenger_audit`
SH = Path(__file__).resolve().parents[2] / "pod" / "challenger.sh"


def wanted(env: Mapping[str, str]) -> bool:
    """Whether ``env``, the environment a sweep's servers run in, puts each server on its own challenger
    (FLOCK_CHALLENGER=1)."""
    return env.get("FLOCK_CHALLENGER") == "1"


class Challenger(subprocess.Popen):
    """A challenger's process. ``work`` holds its LMS key and session secret: a private directory (`mkdtemp`, mode 0700)
    outside every tree a caller keeps, so no caller's custody can publish them, and ``end`` deletes it."""
    work: Path


def _env(d: Path, work: Path | None = None) -> dict:
    """The environment `challenger.sh` runs in: this process's, where CHALLENGER_REPO and CHALLENGER_PY default to this
    tree and this interpreter, and the challenger's work directory ``work``."""
    return ({"CHALLENGER_REPO": str(SH.parents[4]), "CHALLENGER_PY": sys.executable} | dict(os.environ)
            | ({"CHALLENGER_WORK": str(work)} if work is not None else {}))


def start(d: Path) -> tuple[Challenger, list[str]]:
    """A challenger whose record goes under ``d`` (its key and secret in its own ``work``), and the flags `serve` takes so
    that it samples and opens every session's coin tree. The process exits with the challenger's status."""
    work = Path(tempfile.mkdtemp(prefix="challenger-"))
    p = Challenger(["bash", "-c", '. "$0" && challenger_start "$1" || exit 1; echo "$CHALLENGER $CHALLENGER_SECRET"; '
                    'challenger_end "$1"', str(SH), str(d)], stdout=subprocess.PIPE, text=True, env=_env(d, work))
    p.work = work
    got = (p.stdout.readline() if p.stdout else "").split()
    if len(got) != 2:
        p.wait()
        shutil.rmtree(work, ignore_errors=True)
        err = d / "challenger.err"
        raise RuntimeError(f"no challenger under {d}: {err.read_text()[-300:] if err.exists() else 'it never listened'}")
    return p, ["--tree-from", got[0], "--tree-secret-file", got[1]]


def end(d: Path, p: subprocess.Popen, sessions: list[str]) -> dict:
    """``d``'s challenger's exit status once its server has closed, and its audit of the ``sessions`` directories; ``ok`` when
    it exited 0 and the audit passed every record, of which there is at least one. A challenger still open a minute after its
    server is gone (a server killed before its last session) is killed, and fails. Its key and secret are deleted."""
    work = getattr(p, "work", None)
    try:
        rc = p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            if work is None:
                p.kill()
            else:
                os.kill(int((work / d.name / "pid").read_text()), 9)
        except (OSError, ValueError):
            p.kill()
        rc = p.wait()
    a = subprocess.run(["bash", "-c", '. "$0" && challenger_audit "$@"', str(SH), str(d), *sessions],
                       capture_output=True, text=True, env=_env(d, work))
    if work is not None:
        shutil.rmtree(work, ignore_errors=True)
    return ({"exit": rc, "sessions": len(sessions), "audit": a.returncode == 0, "ok": rc == 0 and bool(sessions) and a.returncode == 0}
            | ({} if a.returncode == 0 else {"audit_tail": (a.stdout + a.stderr)[-400:]}))


def records(sessions: Path, pattern: str = "*") -> list[str]:
    """The session directories under ``sessions`` whose name matches ``pattern`` and that hold a record."""
    return sorted(str(x.parent) for x in sessions.glob(f"{pattern}/session.json"))
