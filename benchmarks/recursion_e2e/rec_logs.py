"""What the recursion bench reads from one record file of a job's ``out/`` (``job.sh``): ``/usr/bin/time -v``'s
wall, peak RSS and CPU seconds, the prover's ``LIVE`` lines, tagged JSON lines, Lean's verdict, the lean step's
composition of V*'s parts, the GPU's peak, a step's exit status, the run's parameters and its process samples. The
readers of Flock's own output formats (its phase buckets, Lean's time split, the staged statements' keys) stay in
``rec_profile``. A line or file that doesn't parse is read as absent, and each one is
logged at debug level. Stdlib only, as ``rec_profile``.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, TypedDict

log = logging.getLogger(__name__)

#: the lean step's record of its composition of V*'s parts, in its ``out/lean`` (lean_verdicts.py's ``RECORD``)
COMPOSITION = "rec-compose.json"


def text(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


_WALL = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)")


_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")


def wall_seconds(s: str) -> float:
    """``/usr/bin/time -v``'s elapsed time, ``h:mm:ss`` or ``m:ss.ss``, in seconds."""
    parts = [float(p) for p in s.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


class TimeV(TypedDict):
    wall_s: float
    peak_rss_gb: float | None
    cpu_s: float


def timev(p: Path) -> TimeV | None:
    """Wall seconds, peak RSS (GB) and CPU seconds of one ``/usr/bin/time -v`` file; None when it has no wall line."""
    t = text(p)
    m = _WALL.search(t)
    if not m:
        return None
    rss = _RSS.search(t)
    cpu = sum(float(x) for x in re.findall(r"(?:User|System) time \(seconds\): ([0-9.]+)", t))
    return {"wall_s": wall_seconds(m.group(1)), "peak_rss_gb": int(rss.group(1)) / 2**20 if rss else None, "cpu_s": cpu}


def live_lines(p: Path) -> list[tuple[float | None, dict[str, Any]]]:
    """The prover's ``LIVE`` records with the time each arrived: ``prove.tsv`` is 85's stamped stdout
    (``time<TAB>LIVE<TAB>{json}``, printed when the run's verdict is in); a plain stdout has no times."""
    out = []
    for ln in text(p).splitlines():
        t = None
        head, _, rest = ln.partition("\t")
        if rest.startswith("LIVE"):
            try:
                t, ln = float(head), rest
            except ValueError as e:
                log.debug("%s: a LIVE line's stamp %r is no time (%s), so the line is read unstamped", p, head, e)
        if ln.startswith("LIVE"):
            try:
                out.append((t, json.loads(ln[5:])))
            except ValueError as e:
                log.debug("%s: a LIVE line that isn't JSON is skipped (%s)", p, e)
    return out


def tagged(p: Path, tag: str) -> list[dict[str, Any]]:
    """``TAG<TAB>{json}`` lines (serve's ``SESSION``, the replay's ``UPSTREAM``)."""
    out = []
    for ln in text(p).splitlines():
        if ln.startswith(tag + "\t"):
            try:
                out.append(json.loads(ln.split("\t", 1)[1]))
            except ValueError as e:
                log.debug("%s: a %s line that isn't JSON is skipped (%s)", p, tag, e)
    return out


def lean_verdict(p: Path) -> dict[str, Any] | None:
    t = text(p)
    at = t.find("VERDICT ")
    if at < 0:
        return None
    try:
        return json.JSONDecoder(strict=False).raw_decode(t[at + 8:])[0]
    except ValueError as e:
        log.debug("%s: its VERDICT isn't JSON, read as no verdict (%s)", p, e)
        return None


_COMPOSE = re.compile(r"^lean compose(?: \(RETRY=([^,)]*)[^)]*\))?: (.*?)(?:  <- FAILS)?$", re.M)


def composition(o: Path) -> dict[str, Any] | None:
    """What one invocation of the lean step says of its composition of V*'s parts: ``composed`` (whether Lean was asked),
    ``accepted`` (its answer; None when it gave none or was not asked) and ``why``. From the step's record
    (`COMPOSITION`), and for an invocation from before that record from its log (`logged_composition`). None when it
    left neither, which is not a composition."""
    path = o / "lean" / COMPOSITION
    try:
        rec = json.loads(text(path) or "null")
    except ValueError as e:
        log.debug("%s isn't JSON, read as no record (%s)", path, e)
        rec = None
    if not (isinstance(rec, dict) and isinstance(rec.get("composed"), bool)):
        rec = logged_composition(text(o.parent / "step.log"))
        if rec is None:
            return None
    accepted = rec.get("accepted") if rec["composed"] and isinstance(rec.get("accepted"), bool) else None
    return {"composed": rec["composed"], "accepted": accepted, "why": str(rec.get("why") or "")}


def logged_composition(step_log: str) -> dict[str, Any] | None:
    """A lean step's composition from its log (job.sh's ``step.log``): the step's ``lean compose`` line on the honest
    parts, which says they were not composed, or carries Lean's ``REC-COMPOSE`` line, or says why Lean gave none; a
    forced retry's line (85's RETRY) names its part, which the reason then starts with, as the step's record does. None
    without that line."""
    m = _COMPOSE.search(step_log)
    if not m:
        return None
    retried, line = m.group(1), m.group(2)
    forced = f"a forced retry of {retried} (RETRY={retried}): " if retried else ""
    if line.startswith("REC-COMPOSE "):
        try:
            v = json.loads(line.removeprefix("REC-COMPOSE "))
        except ValueError as e:
            log.debug("a step log's REC-COMPOSE line isn't JSON, read as no answer (%s)", e)
            v = None
        if isinstance(v, dict):
            return {"composed": True, "accepted": v.get("accepted"), "why": forced + str(v.get("why") or "")}
    return {"composed": "is not composed" not in line, "why": forced + line}


def gpu_peak(p: Path) -> float | None:
    xs = [int(x) for x in text(p).split() if x.isdigit()]
    return float(max(xs)) if xs else None


def exit_status(p: Path) -> int | None:
    m = re.search(r"^\s*Exit status: (\d+)\s*$", text(p), re.M)
    return int(m.group(1)) if m else None


def processes(p: Path, comm: str) -> list[dict[str, Any]]:
    """Each ``comm`` process the run's sampler saw (``research run``'s ``resources.jsonl`` beside ``out/``): its CPU seconds and
    peak RSS at its last sample, and when it was first and last seen; the run's own telemetry, not 85's records."""
    seen: dict[int, dict[str, Any]] = {}
    for ln in text(p).splitlines():
        if '"kind":"process"' not in ln or comm not in ln:
            continue
        try:
            d = json.loads(ln)
        except ValueError as err:
            log.debug("%s: a process sample that isn't JSON is skipped (%s)", p, err)
            continue
        for q in d.get("processes", []):
            if q.get("comm") == comm:
                e = seen.setdefault(q["pid"], {"pid": q["pid"], "first": d["t_utc"]})
                e |= {"last": d["t_utc"], "cpu_s": q.get("cpu_s"), "rss_hwm_gb": round((q.get("rss_hwm_bytes") or 0) / 2**30, 2),
                      "threads": q.get("threads")}
    return [e for e in seen.values() if (e.get("cpu_s") or 0) >= 1]


def params(root: Path) -> dict[str, str]:
    out = {}
    for tok in text(root / "params.txt").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out
