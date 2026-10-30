"""`research status RUN` on a check run (`tools/verity/check/check.py`): its live state from the run dir's own files, and an
ETA against a basis, a finished check whose files say how long each step and circuit-check target took.

The files are the ones the check already writes, read and never written here (`CHECK_FILES`):

- `spans.jsonl` (trace/v1, the last line per `span_id` wins): the root is the check; a child with `end_utc` null is a step
  running since its `start_utc`, and a completed child is an ended step with its status.
- `circuit-check.json.progress.jsonl`: one line per circuit-check target as it ends (its `target`, `seconds`, `failures`).

A running step's ETA: for circuit-check, the basis's targets this run hasn't checked yet, at this run's own pace (the
target-seconds it has done per second it has run); for another step, the basis's time for it less this one's so far. The
check's ETA is the longest of those, and only when every running step has one and no step the basis ran is still to start:
otherwise it says why it is unknown, never a guess.
"""
from __future__ import annotations

import json
import re
import shlex
import tarfile
from datetime import UTC, datetime
from pathlib import Path

SPANS = "spans.jsonl"
PROGRESS = "circuit-check.json.progress.jsonl"
CHECK_FILES = (SPANS, "steps.jsonl", PROGRESS)
CIRCUIT_CHECK = "circuit-check"
#: an ended step that passed within this many seconds is counted, not named
QUICK_S = 60
#: how many of a machine's newest run dirs `remote_basis` looks through for a finished check
BASIS_SCAN = 200


def _jsonl(path: Path) -> list[dict]:
    """The file's JSON objects, one a line, but a last line a writer hasn't finished yet."""
    if not path.is_file():
        return []
    text = path.read_text(errors="replace")
    lines = text.splitlines() if text.endswith("\n") else text.splitlines()[:-1]
    return [doc for line in lines if line.strip() and isinstance(doc := json.loads(line), dict)]


def spans(run_dir: Path) -> list[dict]:
    """The run's spans, the last line of each `span_id`, in the order each first appeared."""
    last: dict[str, dict] = {}
    for s in _jsonl(run_dir / SPANS):
        if isinstance(s.get("span_id"), str):
            last[s["span_id"]] = s
    return list(last.values())


def is_check(run_dir: Path) -> bool:
    return any(s.get("parent_span_id") is None and s.get("name") == "check" for s in spans(run_dir))


def finished(run_dir: Path) -> bool:
    """Whether the run's check ended: its root span is completed."""
    return any(s.get("parent_span_id") is None and s.get("end_utc") for s in spans(run_dir))


def _t(utc: object) -> datetime | None:
    if not (isinstance(utc, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", utc)):
        return None
    return datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _took(s: dict) -> float | None:
    a, b = _t(s.get("start_utc")), _t(s.get("end_utc"))
    return (b - a).total_seconds() if a and b else None


def _span(seconds: float) -> str:
    m = max(0, round(seconds / 60))
    return f"{m} min" if m < 90 else f"{m // 60} h {m % 60:02d} m"


def view(run_dir: Path, now: datetime, basis: Path | None = None) -> list[str]:
    """The check's state in lines, `run_dir`'s files against `basis`'s; [] for a run that isn't a check."""
    sp = spans(run_dir)
    root = next((s for s in sp if s.get("parent_span_id") is None and s.get("name") == "check"), None)
    if root is None:
        return []
    kids = [s for s in sp if s.get("parent_span_id") == root.get("span_id")]
    ended = [s for s in kids if s.get("end_utc")]
    running = [s for s in kids if not s.get("end_utc")]
    base = [s for s in spans(basis) if s.get("parent_span_id") and s.get("end_utc")] if basis else []
    took = {s["name"]: t for s in base if (t := _took(s)) is not None}
    out = []
    shown = [s for s in ended if s.get("status") != "ok" or (_took(s) or 0) >= QUICK_S]
    if ended:
        rest = len(ended) - len(shown)
        out.append("ended   " + "; ".join([*(f"{s['name']} {s.get('status')} in {_span(_took(s) or 0)}" for s in shown),
                                           *([f"{rest} more ok in under a minute"] if rest else [])]))
    etas: list[float | None] = []
    for s in running:
        start = _t(s.get("start_utc"))
        so_far = (now - start).total_seconds() if start else 0.0
        eta: float | None = None
        if s.get("name") == CIRCUIT_CHECK:
            text, eta = _circuit_check(run_dir, basis, so_far)
        elif s.get("name") in took:
            eta = max(0.0, took[s["name"]] - so_far)
            text = f"the basis took {_span(took[s['name']])}, " + (f"about {_span(eta)} left" if eta else "past it already")
        else:
            text = "no time for it in the basis" if basis else "no basis to time it against"
        etas.append(eta)
        out.append(f"running {s.get('name')} for {_span(so_far)}: {text}")
    if root.get("end_utc"):
        out.append(f"check   ended {root.get('status')} at {root['end_utc']} after {_span(_took(root) or 0)}")
        return out
    seen = {s.get("name") for s in kids}
    later = sorted(n for n in took if n not in seen)
    if basis is None:
        out.append("ETA     unknown: no finished check to time it against (`--basis RUN`)")
    elif not running:
        out.append("ETA     unknown: no step is running")
    elif later:
        out.append(f"ETA     unknown: the basis's {', '.join(later)} hasn't started here yet")
    elif any(e is None for e in etas):
        out.append("ETA     unknown: a running step has no time in the basis")
    else:
        out.append(f"ETA     about {_span(max(etas))} from now, by {basis.name}")
    return out


def _circuit_check(run_dir: Path, basis: Path | None, so_far: float) -> tuple[str, float | None]:
    """circuit-check's progress in words, and its ETA: the basis's targets not yet done here at this run's pace."""
    done = [d for d in _jsonl(run_dir / PROGRESS) if isinstance(d.get("target"), str)]
    failed = sum(1 for d in done if d.get("failures"))
    text = f"{len(done)} target(s) done" + (f", {failed} with failures" if failed else ", none failed")
    if basis is None:
        return text, None
    ref = [d for d in _jsonl(basis / PROGRESS) if isinstance(d.get("target"), str)]
    if not ref:
        return f"{text}; the basis has no circuit-check targets", None
    seen = {d["target"] for d in done}
    left = [d for d in ref if d["target"] not in seen]
    work = sum(float(d.get("seconds") or 0) for d in done)
    if not done or so_far <= 0 or work <= 0:
        return f"{text}; {len(left)} of the basis's {len(ref)} to go, no pace yet", None
    eta = sum(float(d.get("seconds") or 0) for d in left) / (work / so_far)
    return f"{text}; {len(left)} of the basis's {len(ref)} to go, about {_span(eta)} at this run's pace", eta


def remote_basis(rec: dict, ssh: list[str], dest_root: Path, read) -> Path | None:
    """The newest other finished check on the machine of a remote run (``rec``, its remote.json): its CHECK_FILES copied into
    ``dest_root``/<run id>, or None when the machine has none. ``read`` is `remote.ssh_read`; a stream that isn't a tar raises
    tarfile.TarError."""
    rd = rec.get("remote_run_dir")
    if not rd:
        return None
    parent, me = str(Path(rd).parent), Path(rd).name
    files = " ".join(shlex.quote(f) for f in CHECK_FILES)
    script = (f"cd {shlex.quote(parent)} || exit 0; for d in $(ls -td r*/ 2>/dev/null | head -n {BASIS_SCAN}); do d=${{d%/}}; "
              f"[ \"$d\" = {shlex.quote(me)} ] && continue; [ -s \"$d/{PROGRESS}\" ] || continue; "
              f"grep -q '\"parent_span_id\": null.*\"end_utc\": \"' \"$d/{SPANS}\" 2>/dev/null || continue; "
              f"cd \"$d\" && ls {files} 2>/dev/null | sed \"s|^|$d/|\" | (cd .. && tar -cf - -T -); exit 0; done; "
              f"tar -cf - -T /dev/null")
    got: list[str] = []

    def extract(stream) -> None:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for m in tf:
                parts = Path(m.name).parts
                if len(parts) != 2 or parts[1] not in CHECK_FILES or ".." in parts or m.name.startswith("/"):
                    continue
                tf.extract(m, dest_root, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
                got.append(parts[0])

    r = read(ssh, ["sh", "-c", script], extract, timeout=120)
    return dest_root / got[0] if r.returncode == 0 and got else None
