"""Every recorded network-traces run's audit through the Lean verifier beside the Python one, link-window by link-window,
with negative controls.

The verifier of record moves from `audit.Audit` to `lean_audit.Audit` (each verdict computed by the Lean check's
`network-records` executable) only if both give the same verdict on every link-window of every recorded run. Two kinds of
run carry an audit verdict:
- **trace runs** (`run.sh`, `record.py`'s traces): `replay.py` over the trace at a calibration report, which audits every
  link-window raw, committed and certified (`traces`). Their recorded verdicts are a replay's `verdicts.jsonl`, the rows
  of the trace's `labels.run` (`--recorded`; a trace sweep recorded raw and committed only);
- **active runs** (`active_live.sh`, `active_live_cells.sh`, `active_replay.py`): `active_replay.audit_configuration` over
  each configuration directory's published records, raw and committed (`active`). The run's own `verdicts.jsonl`
  beside its `runs/` holds the verdicts it recorded.

Each run's audit path runs twice, once with each verifier (`replay.replay_configuration`'s and
`active_replay.audit_configuration`'s `audit`), and every link-window's verdict is compared, for each audit kind, with
the other verifier's and with the recorded one. A trace replay's certified audit (`replay.ReplaySigner`'s toy LMS key)
is compared between the verifiers only: the recorded runs had no certified audit.

**Negative controls.** The natural ones are the rejections the runs already contain, counted per violation. The injected
ones mutate one published link-window of each run (the first link-window both verifiers accepted, of the first audit
that checked an egress link) and check the mutated link again with a fresh audit of each verifier: drop its record
(`no-verdict`), duplicate it (`duplicate`), flip a byte of its committed digest, change one slot of its raw grid
(`mismatch`), give its certified head a certificate signing the next window or one of another of the certifier's sessions
(`signature`), make its status incomplete
(`status:…`), declare a sync outside Σ_sync (`advice`) or one other than the spatial unit's (`advice`, the unit's clock
set by a link checked before it with no records), and on the ingress link drop, duplicate or fault its record. A control
passes when both verifiers give the same verdict on every link-window and reject the mutated one; one both accept, or on
which they differ, fails.

    python lean_replay.py traces --calibration REPORT [--recorded VERDICTS.jsonl] --out DIR [--jobs N] TRACE ...
    python lean_replay.py active --out DIR [--jobs N] [--max-frames F] [--budget-gb G] RUN_DIR ...
    python lean_replay.py summarize OUT PROGRESS.jsonl ...

Each task (a trace, an active configuration) runs in a process of its own. The Lean check holds each frame's payload as
a list of bits, about 1 MB of memory per 4 KiB frame, so a real active configuration's audit peaks at tens to hundreds of
GB: `--budget-gb` bounds the estimate of what runs at once, and `--max-frames` leaves out (and lists) the larger ones.
The injected controls of an active run are made on its configuration with the fewest Program frames. A RUN_DIR
is an active run's record tree: `inputs/calibration.json`, and `out/[CELL/]runs/CONFIG/STRATEGY/` with the
recorded `out/[CELL/]verdicts.jsonl`. It writes `progress.jsonl` (each task's result as it ends), `runs.jsonl` (one
line per task), `diffs.jsonl` (one line per link-window whose verdicts differ anywhere) and `summary.json`, and prints
the summary line.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import importlib
import itertools
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import replay  # noqa: E402
from verity.core.protocols.certifier import audit as py_audit  # noqa: E402
from verity.core.protocols.certifier.certificate import (Certificate, CertifiedRecord, Direction, Issuer,  # noqa: E402
                                                          certificate_message, parameters_digest)
from verity.core.protocols.certifier.commitment import commit  # noqa: E402
from verity.core.protocols.certifier.grid import COMPLETE, Reason, Record, Status  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync, expected_grids  # noqa: E402

TRACE_AUDITS = ("raw", "committed", "certified")
ACTIVE_AUDITS = ("raw", "committed")


LEAN = "verity.core.protocols.certifier.lean_audit:Audit"


def verifier(name: str | type) -> type:
    """The class `module:Name` names (or the class itself): the verifier compared with `audit.Audit` (default `LEAN`)."""
    if isinstance(name, type):
        return name
    module, _, cls = name.partition(":")
    return getattr(importlib.import_module(module), cls)


class AuditDone(Exception):
    """The audit path's audits are done; what it computes after them isn't the verifier's."""


_serial = itertools.count()


def recording(base: type, sink: list, stop_after_ingress: int = 0) -> type:
    """`base` with every check's arguments and verdicts (a copy: the caller may extend the list) appended to `sink`, keyed
    by the audit object (a serial, since a freed audit's `id` is reused); with `stop_after_ingress` n, `AuditDone` is raised once the n-th
    `check_ingress_link` has returned."""

    def tag(audit) -> int:
        return audit.__dict__.setdefault("_recording", next(_serial))

    class Recording(base):
        def check_link(self, link, records, frames, syncs, sync_set, params, key=None, session=None):
            v = super().check_link(link, records, frames, syncs, sync_set, params, key, session)
            sink.append((tag(self), "egress", (link, list(records), frames, list(syncs), sync_set, params, key, session),
                         list(v)))
            return v

        def check_ingress_link(self, link, records, windows, key=None, params=None, session=None):
            v = super().check_ingress_link(link, records, windows, key, params, session)
            sink.append((tag(self), "ingress", (link, list(records), windows, key, params, session), list(v)))
            if stop_after_ingress and sum(c[1] == "ingress" for c in sink) == stop_after_ingress:
                raise AuditDone
            return v

    return Recording


def violation_counts(rows: Sequence[dict], audits: Sequence[str]) -> dict[str, dict[str, int]]:
    """Per audit kind, its violations; under `link_windows`, the rejected link-windows by their first kind's violation."""
    first = (next((r[k] for k in audits if r[k] is not None), None) for r in rows)
    return {k: dict(Counter(r[k] for r in rows if r[k] is not None)) for k in audits} | \
        {"link_windows": dict(Counter(v for v in first if v is not None))}


def row_key(row: dict) -> tuple:
    return row.get("config"), row.get("run"), row.get("strategy"), row["direction"], row["link"], row["window"]


def compare_rows(py: Sequence[dict], lean: Sequence[dict], recorded: dict[tuple, dict] | None,
                 audits: Sequence[str]) -> tuple[list[dict], dict]:
    """The link-windows whose verdicts differ between the verifiers (any audit kind) or from the recorded ones (the kinds
    recorded), and the tallies."""
    diffs, stats = [], Counter(link_windows=0, same=0)
    if [row_key(r) for r in py] != [row_key(r) for r in lean]:
        raise AssertionError("the two audit paths gave different link-windows")
    for a, b in zip(py, lean, strict=True):
        stats["link_windows"] += 1
        d = {}
        if any(a[k] != b[k] for k in audits):
            d["python"], d["lean"] = {k: a[k] for k in audits}, {k: b[k] for k in audits}
        else:
            stats["same"] += 1
        if recorded is not None:
            rec = recorded.get(row_key(a))
            if rec is None:
                stats["recorded_missing"] += 1
                d["recorded"] = None
            else:
                kinds = [k for k in audits if k in rec]
                stats["recorded_compared"] += 1
                py_ok, lean_ok = all(a[k] == rec[k] for k in kinds), all(b[k] == rec[k] for k in kinds)
                stats["recorded_same_python"] += py_ok
                stats["recorded_same_lean"] += lean_ok
                if not (py_ok and lean_ok):
                    d["recorded"] = {k: rec[k] for k in kinds}
                    d.setdefault("python", {k: a[k] for k in audits})
                    d.setdefault("lean", {k: b[k] for k in audits})
        if d:
            diffs.append({"config": a.get("config"), "run": a.get("run"), "strategy": a.get("strategy"),
                          "direction": a["direction"], "link": a["link"], "window": a["window"]} | d)
    if recorded is not None:
        stats["recorded_rows"] = len(recorded)
    return diffs, dict(stats)


# -- negative controls -------------------------------------------------------------------------------------------------

def _flip_hex(s: str) -> str:
    last = s[-1]
    return s[:-1] + ("0" if last != "0" else "1")


def _change_slot(record: Record, frame_bytes: int) -> Record:
    grid = [list(row) for row in record.grid]
    for row in grid:
        for k, slot in enumerate(row):
            if slot is not None:
                row[k] = bytes([slot[0] ^ 1]) + slot[1:]
                return dataclasses.replace(record, grid=tuple(tuple(r) for r in grid))
    grid[0][0] = bytes(frame_bytes)
    return dataclasses.replace(record, grid=tuple(tuple(r) for r in grid))


def _other_window_certificate(c: CertifiedRecord, signer: replay.ReplaySigner) -> CertifiedRecord:
    h, w = c.head, c.head.window + 1
    sig = signer.sign(certificate_message(signer.key_id, replay.REPLAY_SESSION, h.link, h.direction, w, h.digest()))
    return CertifiedRecord(h, Certificate(signer.key_id, h.link, h.direction, w, h.digest(), sig))


def _other_session_certificate(c: CertifiedRecord, signer: replay.ReplaySigner) -> CertifiedRecord:
    """The certificate the same key gives the same head in another of the certifier's sessions."""
    h = c.head
    other = bytes(b ^ 0xFF for b in replay.REPLAY_SESSION)
    sig = signer.sign(certificate_message(signer.key_id, other, h.link, h.direction, h.window, h.digest()))
    return CertifiedRecord(h, Certificate(signer.key_id, h.link, h.direction, h.window, h.digest(), sig))


def _verdicts(cls: type, calls: Sequence[tuple[str, tuple]]) -> list[tuple[str, int, str | None]]:
    a = cls()
    out = []
    for kind, args in calls:
        vs = a.check_link(*args) if kind == "egress" else a.check_ingress_link(*args)
        out += [(v.link, v.window, v.violation) for v in vs]
    return out


def injected_controls(calls: list, lean_cls: type) -> list[dict]:
    """The injected mutations of one run (the module docstring), each checked by both verifiers. `calls` is a
    `recording` sink of the Python audit path."""
    egress = [c for c in calls if c[1] == "egress" and c[3]]
    egress = [c for c in egress if any(v.accepted for v in c[3])] or egress
    if not egress:
        return []
    owner, _, (link, records, frames, syncs, sync_set, params, key, session), vs = egress[0]
    w = next((v.window for v in vs if v.accepted), None)
    clean = "accepted"
    if w is None:
        # No window was accepted, so a check behind the status one would see the natural violation first: window 0 is
        # made clean, its sync one of Σ_sync and its record the Program's grid with a complete status.
        w, clean, syncs = 0, "made", list(syncs)
        if syncs[0] not in sync_set:
            syncs[0] = ClockSync(min(sync_set.deltas), syncs[0].rho if syncs[0].rho in sync_set.rhos else
                                 min(sync_set.rhos))
        expected = expected_grids(frames, syncs, params)
        records = [r for r in records if not (r.link == link and r.window == 0)] + \
            [Record(link, 0, expected[0], COMPLETE)]
    raw = [r for r in records if r.link == link and isinstance(r, Record)]
    signer = replay.ReplaySigner()
    parameters = parameters_digest(params, sync_set)
    tests: list[tuple[str, str, list]] = []
    target = next((r for r in raw if r.window == w), None)
    honest = [("egress", (link, records, frames, syncs, sync_set, params, key, session))]
    baseline = {side: next((v for l, x, v in _verdicts(cls, honest) if l == link and x == w), "absent")
                for side, cls in (("python", py_audit.Audit), ("lean", lean_cls))}
    if raw and target is not None:
        others = [r for r in raw if r is not target]
        tests += [
            ("drop", "no-verdict", [("egress", (link, others, frames, syncs, sync_set, params, None))]),
            ("duplicate", "duplicate", [("egress", (link, raw + [target], frames, syncs, sync_set, params, None))]),
            ("raw-slot", "mismatch", [("egress", (link, others + [_change_slot(target, params.frame_bytes)], frames,
                                                  syncs, sync_set, params, None))]),
            ("status", "status:dropped", [("egress", (link, others + [dataclasses.replace(target, status=Status(
                Reason.DROPPED))], frames, syncs, sync_set, params, None))]),
        ]
        committed = [commit(r, params) for r in raw]
        flipped = [dataclasses.replace(c, digest=_flip_hex(c.digest)) if c.window == w else c for c in committed]
        tests.append(("committed-digest-byte", "mismatch",
                      [("egress", (link, flipped, frames, syncs, sync_set, params, None))]))
        issuer = Issuer(signer, replay.REPLAY_SESSION)
        certified = [issuer.certify(r, Direction.EGRESS, params, parameters) for r in raw]
        for name, change in (("certificate-other-window", _other_window_certificate),
                             ("certificate-other-session", _other_session_certificate)):
            swapped = [change(c, signer) if c.window == w else c for c in certified]
            tests.append((name, "signature", [("egress", (link, swapped, frames, syncs, sync_set, params,
                                                          signer.public_key(), replay.REPLAY_SESSION))]))
    outside = ClockSync(max(sync_set.deltas) + 1, syncs[w].rho)
    tests.append(("sync-outside-set", "advice", [("egress", (link, records, frames,
                                                             [outside if i == w else s for i, s in enumerate(syncs)],
                                                             sync_set, params, key, session))]))
    alternative = next((ClockSync(d, r) for d in sync_set.deltas for r in sync_set.rhos
                        if ClockSync(d, r) != syncs[w]), None)
    if alternative is not None:
        unit = [alternative if i == w else s for i, s in enumerate(syncs)]
        tests.append(("sync-not-units", "advice", [("egress", ("unit-clock", [], [], unit, sync_set, params, None)),
                                                   ("egress", (link, records, frames, syncs, sync_set, params, key,
                                                               session))]))
    ingress = [c for c in calls if c[0] == owner and c[1] == "ingress" and c[3]]
    ilink, w_in = None, 0
    if ingress:
        ilink, irecords, windows, ikey, iparams, isession = ingress[0][2]
        w_in = next((v.window for v in ingress[0][3] if v.accepted), 0)
        iraw = [r for r in irecords if r.link == ilink and isinstance(r, Record)]
        itarget = next((r for r in iraw if r.window == w_in), None)
        if itarget is not None:
            iothers = [r for r in iraw if r is not itarget]
            tests += [
                ("ingress-drop", "no-verdict", [("ingress", (ilink, iothers, windows, None, iparams))]),
                ("ingress-duplicate", "duplicate", [("ingress", (ilink, iraw + [itarget], windows, None, iparams))]),
                ("ingress-status", "status:overflow", [("ingress", (ilink, iothers + [dataclasses.replace(
                    itarget, status=Status(Reason.OVERFLOW))], windows, None, iparams))]),
            ]
    out = []
    for name, expected, mutated in tests:
        tw, tlink = (w_in, ilink) if name.startswith("ingress") else (w, link)
        where = {} if name.startswith("ingress") else {"clean": clean, "baseline": baseline}
        py = _verdicts(py_audit.Audit, mutated)
        try:
            le = _verdicts(lean_cls, mutated)
        except (ValueError, RuntimeError) as e:
            out.append({"control": name, "expected": expected, "window": tw, "python": None, "lean": None,
                        "error": str(e)[:300], "passed": False} | where)
            continue
        pv = next((v for l, x, v in py if l == tlink and x == tw), "absent")
        lv = next((v for l, x, v in le if l == tlink and x == tw), "absent")
        out.append({"control": name, "expected": expected, "window": tw, "python": pv, "lean": lv,
                    "same": py == le, "passed": py == le and pv is not None and lv is not None and pv != "absent",
                    "expected_kind": pv == expected and lv == expected} | where)
    return out


# -- the two audit paths -----------------------------------------------------------------------------------------------

def _replay_trace(configs: Sequence[dict], setup: replay.Setup, syncs: str, fresh: bool, audit: type) -> list[dict]:
    """`replay.replay_file`'s loop over one trace's configurations, with `audit` as the verifier."""
    rows, history, run = [], [], None
    for cfg in configs:
        same = not fresh and run is not None and cfg.get("labels", {}).get("run") == run
        run = cfg.get("labels", {}).get("run")
        r, _, history = replay.replay_configuration(cfg, setup, syncs, history if same else (), audit=audit)
        rows += r
    return rows


def trace_job(args: tuple) -> dict:
    path, report, syncs, fresh, recorded, controls, other = args
    setup = replay.Setup(report)
    doc = replay.read_trace(Path(path))
    grid = {k: doc[k] for k in ("bucket_ms", "window_buckets", "frame_bytes", "queue_capacity")}
    if grid != setup.grid:
        return {"run": Path(path).name, "kind": "trace", "error": f"grid {grid} differs from the calibration's"}
    configs = doc["configurations"]
    calls: list = []
    t0 = time.monotonic()
    py = _replay_trace(configs, setup, syncs, fresh, recording(py_audit.Audit, calls))
    t1 = time.monotonic()
    lean = verifier(other)
    le = _replay_trace(configs, setup, syncs, fresh, lean)
    t2 = time.monotonic()
    rec = None if recorded is None else {row_key(r): r for r in recorded}
    diffs, stats = compare_rows(py, le, rec, TRACE_AUDITS)
    runs = sorted({r["run"] for r in py if r.get("run")})
    name = Path(path).parent.name if Path(path).name == "trace.json" else Path(path).name
    return {"run": runs[0] if len(runs) == 1 else name, "kind": "trace", "trace": str(path),
            "configurations": len(configs), **stats, "violations": {"python": violation_counts(py, TRACE_AUDITS),
                                                                    "lean": violation_counts(le, TRACE_AUDITS)},
            "seconds": {"python_path": round(t1 - t0, 2), "lean_path": round(t2 - t1, 2)},
            "controls": injected_controls(calls, lean) if controls else [], "diffs": diffs}


def _recorded_active(root: Path, run: Path) -> dict[tuple, dict] | None:
    """The verdicts the run recorded for this configuration directory (`…/runs/CONFIG/STRATEGY`), keyed as its rows."""
    path = run.parent.parent.parent / "verdicts.jsonl"
    if not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("strategy", run.name) == run.name:
            out[(r["config"], None, None, r["direction"], r["link"], r["window"])] = r
    return out


def _audited_rows(plan: dict, report: dict, run: Path, audit: type, calls: list) -> list[dict]:
    """`active_replay.audit_configuration`'s rows with `audit` as the verifier, read off its two audits (raw, then
    committed) as they return; it is stopped there, before the comparisons that don't read a verdict."""
    import active_replay
    try:
        active_replay.audit_configuration(plan, report, run, audit=recording(audit, calls, stop_after_ingress=2))
    except AuditDone:
        pass
    audits = list(dict.fromkeys(c[0] for c in calls))
    if len(audits) != 2:
        raise AssertionError(f"audit_configuration ran {len(audits)} audits, not raw and committed")
    egress = next(c[2][0] for c in calls if c[1] == "egress")
    raw, committed = ([v for c in calls if c[0] == a for v in c[3]] for a in audits)
    return [{"config": plan["name"], "link": a.link, "window": a.window,
             "direction": "egress" if a.link == egress else "ingress", "raw": a.violation, "committed": b.violation}
            for a, b in zip(raw, committed, strict=True)]


def active_frames(run: Path) -> int:
    """The Program frames an active configuration's audit reads: the live shim's events, else the plan's frames."""
    if (run / "server.events.gz").exists():
        with gzip.open(run / "server.events.gz", "rb") as fh:
            return sum(1 for _ in fh)
    with gzip.open(run / "plan.json.gz", "rt") as fh:
        return sum(len(ds) for _, _, ds in json.load(fh)["sessions"])


def active_job(args: tuple) -> dict:
    root, run, controls, other = args
    root, run = Path(root), Path(run)
    cal = root / "inputs" / "calibration.json"
    if not cal.exists():
        cal, = (root / "inputs").glob("*calibration.json")
    report = json.loads(cal.read_text())
    with gzip.open(run / "plan.json.gz", "rt") as fh:
        plan = json.load(fh)
    calls: list = []
    t0 = time.monotonic()
    py = _audited_rows(plan, report, run, py_audit.Audit, calls)
    t1 = time.monotonic()
    lean = verifier(other)
    le = _audited_rows(plan, report, run, lean, [])
    t2 = time.monotonic()
    rec = _recorded_active(root, run)
    if rec is not None:
        mine = {(r["config"], r["direction"], r["link"], r["window"]) for r in py}
        rec = {k: v for k, v in rec.items() if (k[0], k[3], k[4], k[5]) in mine}
    diffs, stats = compare_rows(py, le, rec, ACTIVE_AUDITS)
    for d in diffs:
        d["strategy"] = run.name
    return {"run": root.name, "kind": "active", "configuration": str(run.relative_to(root)),
            "frames": len(calls[0][2][2]), **stats,
            "violations": {"python": violation_counts(py, ACTIVE_AUDITS), "lean": violation_counts(le, ACTIVE_AUDITS)},
            "seconds": {"python_path": round(t1 - t0, 2), "lean_path": round(t2 - t1, 2)},
            "controls": injected_controls(calls, lean) if controls else [], "diffs": diffs}


JOBS = {"trace": trace_job, "active": active_job}


def _label(kind: str, args: tuple) -> dict:
    if kind == "active":
        return {"run": Path(args[0]).name, "kind": kind, "configuration": str(Path(args[1]).relative_to(args[0]))}
    return {"run": str(args[0]), "kind": kind}


def safe_job(kind: str, args: tuple) -> dict:
    try:
        return JOBS[kind](args)
    except Exception as e:  # a run that can't be replayed is reported, never dropped
        return _label(kind, args) | {"error": f"{type(e).__name__}: {e}"[:2000]}


# -- the sweep ---------------------------------------------------------------------------------------------------------

def summarize(results: Sequence[dict]) -> dict:
    """Runs, runs whose every link-window has the same verdict on both sides (and the recorded one, where recorded),
    link-windows compared, violations on each side and the negative controls by kind. A run with no link-window to
    compare (a trace that kept no window) is listed apart, not counted."""
    by_run: dict[str, list[dict]] = {}
    for r in results:
        by_run.setdefault(r["run"], []).append(r)
    s = Counter()
    natural: dict[str, Counter] = {"python": Counter(), "lean": Counter()}
    injected, failed, own_kind, clean, unclean = Counter(), Counter(), Counter(), Counter(), []
    errors, left_out, empty = [], [], []
    same_runs = 0
    for run, parts in list(by_run.items()):
        left_out += [{"run": run, "configuration": p.get("configuration"), "why": p["left_out"]}
                     for p in parts if "left_out" in p]
        parts = [p for p in parts if "left_out" not in p]
        if parts and all("error" not in p and not p["link_windows"] for p in parts):
            empty.append(run)
        if not parts or run in empty:
            del by_run[run]
            continue
        ok = True
        for p in parts:
            if "error" in p:
                errors.append({"run": run, "error": p["error"]})
                ok = False
                continue
            s["link_windows"] += p["link_windows"]
            s["same"] += p["same"]
            s["recorded_compared"] += p.get("recorded_compared", 0)
            s["recorded_same_python"] += p.get("recorded_same_python", 0)
            s["recorded_same_lean"] += p.get("recorded_same_lean", 0)
            s["recorded_missing"] += p.get("recorded_missing", 0)
            ok &= p["same"] == p["link_windows"] and not p["diffs"]
            for side in ("python", "lean"):
                natural[side].update(p["violations"][side]["link_windows"])
            for c in p["controls"]:
                (injected if c["passed"] else failed)[c["control"]] += 1
                own_kind[c["control"]] += bool(c.get("expected_kind"))
            egress = [c for c in p["controls"] if "clean" in c]
            if egress:
                clean[egress[0]["clean"]] += 1
                if any(v is not None for v in egress[0]["baseline"].values()):
                    unclean.append({"run": run, "configuration": p.get("configuration"), "baseline": egress[0]["baseline"]})
        same_runs += ok
    k_natural = sum(natural["python"].values())
    return {"runs": len(by_run), "runs_same_verdict": same_runs, **dict(s),
            "natural_controls": {k: dict(v) for k, v in natural.items()},
            "injected_controls": dict(injected), "injected_failed": dict(failed), "injected_own_kind": dict(own_kind),
            "clean_windows": dict(clean), "baselines_not_accepted": unclean,
            "K": {"natural": k_natural, "injected": sum(injected.values()), "failed": sum(failed.values())},
            "errors": errors, "left_out": left_out,
            "runs_left_out": sorted({x["run"] for x in left_out} - set(by_run)), "runs_without_link_windows": empty}


def write(out: Path, results: Sequence[dict]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.jsonl", "w", encoding="utf-8") as fr, open(out / "diffs.jsonl", "w", encoding="utf-8") as fd:
        for r in results:
            for d in r.get("diffs", []):
                fd.write(json.dumps({"run": r["run"]} | d, separators=(",", ":")) + "\n")
            fr.write(json.dumps({k: v for k, v in r.items() if k != "diffs"} | {"diffs": len(r.get("diffs", []))},
                                separators=(",", ":")) + "\n")
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def active_configurations(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("out/**/runs/*/*/plan.json.gz"))


def run_tasks(kind: str, tasks: Sequence[tuple], jobs: int, costs: Sequence[float] | None = None,
              budget: float | None = None):
    """Each task's `safe_job` in a process of its own (`lean_replay.py task`, its arguments pickled to a file), in the
    order given, at most `jobs` at once and, with `budget`, at most that much of `costs` (estimated peak GB) at once,
    though one task always runs. A process that dies without its result (the OOM killer's) is that task's error."""
    costs = costs or [0.0] * len(tasks)
    pending, running = list(range(len(tasks))), {}
    with tempfile.TemporaryDirectory(prefix="lean-replay-") as tmp:
        while pending or running:
            used = sum(costs[i] for i, _ in running.values())
            for i in list(pending):
                if len(running) >= jobs:
                    break
                if running and budget is not None and used + costs[i] > budget:
                    continue
                pending.remove(i)
                used += costs[i]
                src, dst = Path(tmp) / f"{i}.in", Path(tmp) / f"{i}.out"
                src.write_bytes(pickle.dumps((kind, tasks[i])))
                p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "task", str(src), str(dst)])
                running[p.pid] = (i, p)
            pid, status, usage = os.wait4(-1, 0)
            if pid not in running:
                continue
            i, p = running.pop(pid)
            p.returncode = os.waitstatus_to_exitcode(status)
            dst = Path(tmp) / f"{i}.out"
            peak = {"max_rss_gb": round(usage.ru_maxrss / 2 ** 20, 3)}
            if dst.exists():
                yield json.loads(dst.read_text()) | peak
            else:
                yield _label(kind, tasks[i]) | peak | {"error": f"its process exited {p.returncode} without a result"}


def _exe_rss(log: str, exe: str, *args: str) -> int:
    """`exe args`, its standard streams this process's, then a line of its peak memory and its time appended to `log`
    under the task's label (`LEAN_REPLAY_TASK`): what `VERITY_NETWORK_RECORDS_EXE` names to measure the Lean check
    apart from the audit around it."""
    t0 = time.monotonic()
    p = subprocess.Popen([exe, *args])
    _, status, usage = os.wait4(p.pid, 0)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"task": json.loads(os.environ.get("LEAN_REPLAY_TASK", "null")), "exe": exe,
                             "max_rss_gb": round(usage.ru_maxrss / 2 ** 20, 3), "user_s": round(usage.ru_utime, 2),
                             "sys_s": round(usage.ru_stime, 2), "wall_s": round(time.monotonic() - t0, 2)}) + "\n")
    return os.waitstatus_to_exitcode(status)


def _task(src: str, dst: str) -> int:
    kind, args = pickle.loads(Path(src).read_bytes())
    os.environ["LEAN_REPLAY_TASK"] = json.dumps(_label(kind, args))
    result = safe_job(kind, args)
    Path(dst + ".part").write_text(json.dumps(result))
    os.replace(dst + ".part", dst)
    return 0


LEAN_GB_PER_FRAME = 1.2e-3


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["task"]:
        return _task(*argv[1:])
    if argv[:1] == ["exe-rss"]:  # exe-rss LOG EXE ARGS...: EXE's peak memory and time, appended to LOG
        return _exe_rss(*argv[1:])
    if argv[:1] == ["summarize"]:  # summarize OUT PROGRESS.jsonl ...: the outputs again from tasks' results
        return print_summary(write(Path(argv[1]), [json.loads(line) for f in argv[2:]
                                                   for line in Path(f).read_text().splitlines()]))
    ap = argparse.ArgumentParser(prog="lean_replay.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    t = sub.add_parser("traces")
    t.add_argument("traces", nargs="+", type=Path)
    t.add_argument("--calibration", required=True, type=Path)
    t.add_argument("--recorded", type=Path, help="a replay's verdicts.jsonl over these traces at this calibration")
    t.add_argument("--syncs", default="hindsight")
    t.add_argument("--fresh", action="store_true")
    a = sub.add_parser("active")
    a.add_argument("runs", nargs="+", type=Path)
    a.add_argument("--max-frames", type=int,
                   help="leave out (and list) the configurations whose audit reads more Program frames than this")
    a.add_argument("--each-run-up-to", type=int, default=0, metavar="FRAMES",
                   help="keep each run's smallest configuration over --max-frames when it reads at most this many")
    a.add_argument("--controls-up-to", type=int, metavar="FRAMES",
                   help="inject a run's controls only if its smallest configuration reads at most this many frames "
                        "(each control checks the whole link again)")
    a.add_argument("--only", action="append", metavar="RUN/CONFIGURATION",
                   help="only the configurations whose RUN/CONFIGURATION (as progress.jsonl names them) ends with this; "
                        "repeatable")
    a.add_argument("--budget-gb", type=float,
                   help="run configurations at once only while their estimated peak memory (--gb-per-frame) stays under this")
    a.add_argument("--gb-per-frame", type=float, default=LEAN_GB_PER_FRAME,
                   help=f"a configuration's estimated peak memory per Program frame, past 2 GB (default {LEAN_GB_PER_FRAME:g})")
    for p in (t, a):
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--no-controls", action="store_true")
        p.add_argument("--verifier", default=LEAN, help=f"the verifier compared with audit.Audit (default {LEAN})")
    args = ap.parse_args(argv)
    controls = not args.no_controls
    if args.kind == "traces":
        report = json.loads(args.calibration.read_text())
        recorded: dict[str, list] | None = None
        if args.recorded:
            recorded = {}
            for line in args.recorded.read_text().splitlines():
                r = json.loads(line)
                recorded.setdefault(r.get("run"), []).append(r)
        tasks = []
        for p in sorted(args.traces, key=lambda p: -p.stat().st_size):
            run = p.name.removesuffix(".gz").removesuffix(".json")
            if p.is_dir() or run == "trace":
                run = (p if p.is_dir() else p.parent).name
            tasks.append((p, report, replay.parse_sync_policy(args.syncs), args.fresh,
                          None if recorded is None else recorded.get(run, []), controls, args.verifier))
        kind, costs, budget, results = "trace", None, None, []
    else:
        sized, results = [], []
        for root in args.runs:
            configurations = {c: active_frames(c) for c in active_configurations(root)
                              if not args.only or any(f"{root.name}/{c.relative_to(root)}".endswith(o) for o in args.only)}
            kept = {c: n for c, n in configurations.items() if args.max_frames is None or n <= args.max_frames}
            least = min(configurations, key=configurations.get, default=None)
            if least is not None and not kept and configurations[least] <= args.each_run_up_to:
                kept[least] = configurations[least]
            results += [_label("active", (root, c)) | {"frames": n, "left_out": f"{n} frames, over --max-frames"}
                        for c, n in configurations.items() if c not in kept]
            smallest = min(kept, key=kept.get, default=None)
            if smallest is not None and args.controls_up_to is not None and kept[smallest] > args.controls_up_to:
                smallest = None
            sized += [(n, (root, c, controls and c == smallest, args.verifier)) for c, n in kept.items()]
        sized.sort(key=lambda x: -x[0])
        tasks = [t for _, t in sized]
        kind, costs, budget = "active", [2 + n * args.gb_per_frame for n, _ in sized], args.budget_gb
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "progress.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        for r in run_tasks(kind, tasks, args.jobs, costs, budget):
            results.append(r)
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            fh.flush()
            print(json.dumps({k: r.get(k) for k in ("run", "configuration", "frames", "link_windows", "same", "seconds",
                                                    "error") if k in r}), flush=True)
    return print_summary(write(args.out, results))


def print_summary(s: dict) -> int:
    print(f"{s['runs']} runs, {s['runs_same_verdict']} same verdict, {s['K']['natural'] + s['K']['injected']} negative "
          f"controls ({s['K']['natural']} natural, {s['K']['injected']} injected, {s['K']['failed']} failed); "
          f"{s.get('link_windows', 0)} link-windows compared, {s.get('same', 0)} same; errors {len(s['errors'])}; "
          f"{len(s['left_out'])} configurations left out; {len(s['runs_without_link_windows'])} runs without link-windows")
    return 0 if s["runs"] == s["runs_same_verdict"] and not s["K"]["failed"] and not s["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
