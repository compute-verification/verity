"""The recursion bench's rows (``Row``, each time a ``Stat``) and how the rows that read no Flock output format
are built: a tool's repeats (``tool_rows``, ``step_row``), the lean step's composition (``compose_row``), forge_after.py's
cases (``forge_after_rows``), and the forgeries judged as lean_verdicts.py judges them (``forgeries``, ``forged``). A
prover's runs and the flipped inner coins read Flock's output, so their rows stay in ``rec_profile``. Stdlib only, as ``rec_profile``.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rec_logs import COMPOSITION, text, timev

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class Stat:
    median: float
    min: float
    max: float
    values: tuple[float, ...]

    @classmethod
    def of(cls, xs: Iterable[float | None]) -> Stat | None:
        v = tuple(float(x) for x in xs if x is not None)
        return cls(statistics.median(v), min(v), max(v), v) if v else None


@dataclass
class Row:
    stage: str
    step: str
    what: str
    kind: str                       # gpu (runs) | cpu (repeats) | setup (once)
    wall_s: Stat | None = None      # gpu: prove_total_s per timed run; cpu and setup: the tool's wall
    e2e_s: Stat | None = None       # gpu: each timed run's wall, its waits on the live verifier included
    peak_rss_gb: float | None = None
    gpu_peak_mib: float | None = None
    proof_bytes: int | None = None
    wire_bytes: dict[str, int | None] | None = None
    dominant: dict[str, Any] | None = None
    ok: bool | None = None
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Row:
        st = {k: Stat(d[k]["median"], d[k]["min"], d[k]["max"], tuple(d[k]["values"])) if d.get(k) else None
              for k in ("wall_s", "e2e_s")}
        return cls(**{**{k: v for k, v in d.items() if k in cls.__dataclass_fields__}, **st})


def compose_row(cs: list[dict[str, Any] | None]) -> Row:
    """The lean step's composition of V*'s parts over its invocations (`composition`). ok only when every invocation
    composed them and Lean accepted; not ok when Lean refused one; and no verdict, which is not an accepted recursive
    run, when one was not composed, left no record or got no answer. Its note is the first reason."""
    r = Row("compose", "lean", "V*'s parts composed into the recursive statement by Lean (rec-compose over the run's record)",
            "cpu")
    missing = not cs or any(c is None for c in cs)
    got = [c for c in cs if c]
    refused = next((c for c in got if c["composed"] and c["accepted"] is False), None)
    apart = next((c for c in got if not c["composed"]), None)
    silent = next((c for c in got if c["composed"] and c["accepted"] is None), None)
    if refused:
        r.ok, r.note = False, "refused: " + refused["why"]
    elif apart:
        r.note = "not composed: " + apart["why"]
    elif missing:
        r.note = f"no record of the composition ({COMPOSITION})"
    elif silent:
        r.note = "no answer: " + silent["why"]
    else:
        r.ok = True
    r.note = r.note[:300]
    r.extra = {"composed": False if apart else None if missing else True, "accepted": r.ok}
    return r


def tool_rows(outs: list[Path], step: str, rel: dict[str, tuple[str, str]], verdict=None) -> list[Row]:
    """One row per tool: its ``/usr/bin/time -v`` file (``rel``: name -> (path under out/, what)) over every repeat; none
    for a step that never ran."""
    rows: list[Row] = []
    if not outs:
        return rows
    for name, (path, what) in rel.items():
        ts = [timev(o / path) for o in outs]
        got = [t for t in ts if t]
        r = Row(name, step, what, "cpu", wall_s=Stat.of(t["wall_s"] for t in got),
                peak_rss_gb=max((t["peak_rss_gb"] for t in got if t["peak_rss_gb"] is not None), default=None))
        if not got:
            r.note = "not run"
        if verdict:
            # only repeats whose tool finished (``time -v`` opens its file at the start and writes it at the end)
            vs = [v for o in outs if timev(o / path) if (v := verdict(o, name)) is not None]
            r.ok = all(v is True for v in vs) if vs else None
        rows.append(r)
    return rows


def step_row(outs: list[Path], step: str, what: str) -> Row:
    """A step's whole wall over its repeats (job.sh's ``time-step.txt`` around each invocation of 85)."""
    ts = [t for t in (timev(o.parent / "time-step.txt") for o in outs) if t]
    return Row(f"{step} step", step, what, "cpu", wall_s=Stat.of(t["wall_s"] for t in ts),
               peak_rss_gb=max((t["peak_rss_gb"] for t in ts if t["peak_rss_gb"] is not None), default=None))


def refused(r: Row) -> None:
    """A forged statement's outer row: refused when no run of it was accepted (the ZK server stopped it before release, or
    its releases were refused), the stop or the refusal as its note; `forgeries` judges an accepted one."""
    r.extra["forged"] = True
    r.extra["refused"] = r.ok is not True
    r.ok = r.extra["refused"]


def kind_of(part: str, honest: list[str]) -> str:
    """The forgery ``forged-KIND-PART`` is KIND, PART the longest honest part it ends with; else the name without ``forged-``
    (lean_verdicts.py's rule)."""
    name = part.removeprefix("forged-")
    ends = [h for h in honest if name.endswith("-" + h)]
    return name[: -len(max(ends, key=len)) - 1] if ends else name


def forgeries(rows: list[Row]) -> None:
    """85's forged statements, judged per forgery as lean_verdicts.py judges them: V* accepts only when every part does, so a
    forgery is refused when one of its parts is, and a part it leaves as the honest one (forged-message's alg-p1) is accepted.
    An accepted part's row is ok when another part of its forgery, at the same step, was refused."""
    groups: dict[tuple[str, str], list[Row]] = {}
    for step in ("oprove", "lean"):
        named = [(r, r.stage.split(" ", 1)[1]) for r in rows if r.step == step and r.stage.startswith(("outer ", "lean "))]
        honest = [n for r, n in named if not r.extra.get("forged")]
        for r, n in named:
            if "refused" in r.extra:
                r.extra["kind"] = kind_of(n, honest)
                groups.setdefault((step, r.extra["kind"]), []).append(r)
    for rs in groups.values():
        held = any(r.extra["refused"] for r in rs)
        for r in rs:
            if not r.extra["refused"]:
                r.ok = held
                r.note = ("accepted: a part this forgery leaves as the honest one; " if held
                          else "ACCEPTED: no part of this forgery was refused; ") + r.note


def forge_after_rows(p: Path) -> list[Row]:
    """One row per forge_after.py case (``<bin>/<kind>-<case>``): ok when a forged case was refused (exit 1 and no accepting
    VERDICT) or the honest control accepted, Lean's why as its note; none when the file is missing."""
    try:
        doc = json.loads(text(p) or "null")
    except ValueError as e:
        log.debug("%s isn't JSON, read as no case (%s)", p, e)
        doc = None
    if not doc:
        return []
    rows = []
    for name, v in doc.get("verify", {}).items():
        kind, case = name.split("/")[-1].rsplit("-", 1)
        honest = case == "honest"
        r = Row(f"forge-after {name}", "forge-after",
                f"{kind}'s honest control" if honest else f"{kind} forged after proving ({case})", "cpu")
        accepted = v.get("rc") == 0 or bool(v.get("accepted"))
        r.ok = accepted if honest else not accepted
        r.note = ("ACCEPTED: " if accepted and not honest else "") + str(v.get("why") or "")[:200]
        r.extra = {"rc": v.get("rc"), "forged": not honest}
        rows.append(r)
    if not doc.get("untouched", True):
        r = Row("forge-after inputs", "forge-after", "the honest releases and statements, hashed before and after", "cpu")
        r.ok, r.note = False, "an input changed under forge_after.py"
        rows.append(r)
    return rows


def forged(rows: list[Row]) -> dict[str, bool]:
    """Each forgery kind's verdict: 85's forged statements of it refused at each step (`forgeries`), forge-after's cases of
    it every one refused, and forge-after's honest controls of it accepted."""
    out: dict[str, bool] = {}
    for r in rows:
        if r.ok is None:
            continue
        m = re.fullmatch(r"forge-after [^/]+/([a-z]+)-[a-z]+", r.stage) if r.step == "forge-after" else None
        kind = m.group(1) if m else r.extra.get("kind") if r.step in ("oprove", "lean") else None
        if kind:
            out[kind] = out.get(kind, True) and r.ok
    return out
