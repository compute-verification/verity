"""Sweep driver: ``(algorithm, model, workload) x F x X`` -> lower/upper bounds, written incrementally as JSONL.

Safety discipline (mandatory):

* every case (``build + extract + ADW + bounds for all its (F, X)``) runs in ONE guarded child process
  (:func:`accumulation.sweeps.guard.run_guarded`: wall-clock timeout, RSS watchdog killing at
  ``--max-rss-mb`` (3 GB), a cross-process lock so at most one child runs at a time); the parent only
  reads/writes JSONL and stays small;
* every bound call inside the child has its own wall-clock limit (``--lower-time-limit``,
  ``--upper-time-limit``; a hit is recorded as ``lower_error`` / ``upper_error = "StepTimeout: ..."``);
* when the watchdog kills a child, the parent reads the child's *in-flight marker* (which ``(F, X)`` and
  which stage it was in), records that stage as killed for that row and **re-spawns the child for the
  remaining ``(F, X)``**, so one runaway ``upper_bound`` does not lose the other rows of the case;
* cases outside :data:`accumulation.sweeps.guard.DRIVER_LIMITS` are recorded as ``skipped`` unless ``--unsafe``.

Row schema (:data:`ROW_KEYS`; every row has every key, ``None`` = not available):

    alg, family, model, tokens, seq, population, rank, adapted, adapted_layers, local_steps, gen_tokens, chunk,
    F, X, params_total, params_active, n_ops,
    P (state bytes = accumulated + carried), token_seed_bytes,
    ADW_total, ADW_matmul (work), ADW_matmul_macs, ADW_credited_macs (= |M(C)|: ADW matmul MACs minus the
    ``recompute`` ops), ADW_recompute_macs, all_work, all_macs,
    L, L_source, L_cap, L_infeasible, lower_notes, lower_error, lower_src,
    U, U_units, U_kappa (kappa reported by upper_bound), upper_notes, upper_error, upper_src,
    L_per_token, U_per_token,
    kappa_L = ADW_credited_macs / L, kappa_U = ADW_credited_macs / U            (training circuits: |M(C)| per byte),
    kappa_L_matmul = ADW_matmul_macs / L, kappa_U_matmul = ADW_matmul_macs / U  (recomputes included, second column),
    kappa_L_allmacs = all_macs / L, kappa_U_allmacs = all_macs / U              (inference circuits: ADW = 0 there),
    inv_eta_honest = all_work / ADW_credited_macs (1 / eta of the honest circuit, eta = |M(C)| / Work(C)),
    U_over_L, soundness_violation,
    build_s, extract_s, adw_s, lower_s, upper_s, peak_rss_mb, errors (joined summary), error (hard), ts

``*_src`` are short hashes of the bound module sources at the time the value was computed, so ``--redo lower
--resume`` recomputes exactly the rows whose ``L`` came from an older ``bounds/lower.py``.

Resume / redo semantics:

* default: (re)compute every stage of every ``(case, F, X)`` (later JSONL lines supersede earlier ones;
  the file is compacted after each case);
* ``--resume``: skip triples whose row is complete (a hard error/skip, or a value or a *recorded error* for
  both bounds).  ``--retry-errors`` additionally re-runs stages whose error was a kill
  (``timeout`` / ``oom-guard`` / ``child failed`` / ``StepTimeout``);
* ``--redo {lower,upper,adw,all}``: recompute only those stages for every triple, keeping the other fields
  of the existing row; with ``--resume`` a stage is skipped when its ``*_src`` already matches the current
  module source.

CLI (from the repo root)::

    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.run --preset smoke
    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.run --preset p1 --resume
    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.run --preset p1 --redo lower --resume   # after L v2
    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.run --preset p3

Presets: ``smoke`` (tiny2 / tiny-moe, all algorithms), ``p1`` (Q scaling at 8B: pretrain-dense, lora r=16,
inference-dense; X = 5 MB, F = 1e12), ``p2`` (all algorithms x models at Q = seq = 4096, X x F grid; on
hold until L v2 is validated), ``p2-8b-point`` (the 8B, X = 5 MB, F = 1e12 point of p2 for the P3 table),
``p3`` (efficiency translation from the p1 / p2 files; see :mod:`accumulation.sweeps.analysis`).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import signal
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, NamedTuple

from accumulation.algorithms.registry import ALGORITHMS, build
from accumulation.bounds.adw import adw as compute_adw
from accumulation.configs import MODELS, Workload
from accumulation.graph import extract
from accumulation.sweeps.guard import DRIVER_LIMITS, check_limits
from accumulation.sweeps.util import (RESULTS_DIR, StepTimeout, append_jsonl, dedupe, dumps, read_jsonl, time_limit,
                                      write_json)

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("ACCSWEEP_SCRATCH", "/tmp/accsweep"))
STAGES = ("adw", "lower", "upper")
KILL_ERRORS = ("timeout", "oom-guard", "child failed", "StepTimeout", "not run")


class Case(NamedTuple):
    alg: str
    model: str
    wl: Workload


# ---------------------------------------------------------------------------------------------------------
# module versions (so a redo can tell which rows are stale)
# ---------------------------------------------------------------------------------------------------------

_SRC_FILES = {"lower": "bounds/lower.py", "upper": "bounds/upper.py", "adw": "bounds/adw.py", "extract": "graph/opgraph.py"}


def module_versions() -> dict[str, str]:
    """Short content hashes of the bound / extractor sources."""
    out = {}
    for k, rel in _SRC_FILES.items():
        p = REPO / "accumulation" / rel
        out[k] = hashlib.sha1(p.read_bytes()).hexdigest()[:10] if p.exists() else "missing"
    return out


# ---------------------------------------------------------------------------------------------------------
# optional bound modules
# ---------------------------------------------------------------------------------------------------------

class Bounds(NamedTuple):
    lower: Callable | None
    upper: Callable | None
    lower_error: str | None
    upper_error: str | None


def load_bounds() -> Bounds:
    """Import ``lower_bound`` / ``upper_bound`` if they exist; never raises."""
    lo = up = None
    lo_err = up_err = None
    try:
        from accumulation.bounds.lower import lower_bound as lo  # type: ignore
    except ImportError as e:
        lo_err = f"ImportError: {e}"
    except Exception as e:  # noqa: BLE001 -- a half-written module must not kill the sweep
        lo_err = f"{type(e).__name__}: {e}"
    try:
        from accumulation.bounds.upper import upper_bound as up  # type: ignore
    except ImportError as e:
        up_err = f"ImportError: {e}"
    except Exception as e:  # noqa: BLE001
        up_err = f"{type(e).__name__}: {e}"
    return Bounds(lo, up, lo_err, up_err)


def call_bound(fn: Callable, g, F: int, X: int, **context):
    """Call ``fn`` with whatever subset of ``(g, F, X, roles, adw, workload, program, ...)`` its signature
    asks for (SPEC §5 lists ``lower_bound(g, roles_ok, F, X, *, adw=None)`` and ``upper_bound(g, F, X)``;
    the current modules take ``lower_bound(g, F, X, *, adw=None)`` and ``upper_bound(g, F, X, *, program=...)``).
    Matching by parameter name keeps the driver agnostic to the v2 signature."""
    pool = {"g": g, "graph": g, "F": F, "X": X, **context}
    sig = inspect.signature(fn)
    args, kwargs = [], {}
    first = True
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if first and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            args.append(g)  # the graph is always the first positional parameter, whatever it is called
            first = False
            continue
        first = False
        if p.name in pool:
            if p.kind == p.KEYWORD_ONLY:
                kwargs[p.name] = pool[p.name]
            else:
                args.append(pool[p.name])
        elif p.default is p.empty:
            raise TypeError(f"{fn.__module__}.{fn.__name__}: cannot supply required parameter {p.name!r}")
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------------------------

WL_KEYS = ("tokens", "seq", "population", "rank", "adapted", "adapted_layers", "local_steps", "gen_tokens", "chunk")

ROW_KEYS = (
    "alg", "family", "model", *WL_KEYS, "F", "X", "G", "params_total", "params_active", "n_ops",
    "P", "token_seed_bytes",
    "ADW_total", "ADW_matmul", "ADW_matmul_macs", "ADW_credited_macs", "ADW_recompute_macs", "all_work", "all_macs",
    "L", "L_source", "L_cap", "L_infeasible", "L_at_G", "L_worst_ru", "lower_notes", "lower_error", "lower_src",
    "U", "U_units", "U_kappa", "U_by_class", "U_recompute", "U_n_ru", "U_n_ru_per_token", "U_ru_input_mean", "U_ru_input_max",
    "U_ru_work_mean", "U_ru_work_max", "U_ru_up_max", "upper_notes", "upper_error", "upper_src",
    "M", "M_per_token", "kappa_low", "kappa_high",
    "L_per_token", "U_per_token", "kappa_L", "kappa_U", "kappa_L_matmul", "kappa_U_matmul", "kappa_L_allmacs",
    "kappa_U_allmacs", "inv_eta_honest", "U_over_L",
    "soundness_violation",
    "build_s", "extract_s", "adw_s", "lower_s", "upper_s", "peak_rss_mb", "adw_src", "extract_src",
    "errors", "error", "ts",
)

#: old column name -> new (rows written by the previous driver version are upgraded on read)
LEGACY_KEYS = {"ADW": "ADW_total", "t_build": "build_s", "t_extract": "extract_s", "t_lower": "lower_s",
               "t_upper": "upper_s", "U_n_units": "U_units"}


def wl_fields(wl: Workload) -> dict:
    d = asdict(wl)
    d["adapted"] = "".join(wl.adapted)
    return d


def row_key(r: dict) -> tuple:
    return (r["alg"], r["model"], int(r["tokens"]), int(r["seq"]), r.get("population", 1), r.get("rank", 16),
            r.get("adapted"), r.get("adapted_layers"), r.get("local_steps", 1), r.get("gen_tokens"),
            int(r["F"]), int(r["X"]), _gkey(r.get("G")))


def _gkey(G) -> int:
    """``G`` in a row key: ``None`` (unbounded) -> 0."""
    return 0 if G is None else int(G)


def parse_G(s) -> int | None:
    if s is None or (isinstance(s, str) and s.lower() in ("none", "inf", "-")):
        return None
    return int(float(s))


def is_training(alg: str) -> bool:
    return not alg.startswith("inference")


def case_key(c: Case, F: int, X: int, G=None) -> tuple:
    return row_key({"alg": c.alg, "model": c.model, **wl_fields(c.wl), "F": F, "X": X, "G": G})


def new_row(c: Case, F: int, X: int, G=None) -> dict:
    A = ALGORITHMS[c.alg]
    cfg = MODELS[c.model]
    r = {k: None for k in ROW_KEYS}
    r.update({"alg": c.alg, "family": A.family, "model": c.model, **wl_fields(c.wl), "F": int(F), "X": int(X),
              "G": None if G is None else int(G),
              "params_total": cfg.total_params(), "params_active": cfg.active_params(), "ts": time.time()})
    return r


def _opt_int(x):
    return None if x is None else int(x)


def _jsonable(x):
    """Best-effort JSON view of a bound-module object (dataclass / mapping / sequence / scalar)."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    return str(x)


def _e(x) -> str:
    return "-" if x is None else f"{x:.3e}"


def finalize_row(r: dict) -> dict:
    """Derived columns (idempotent; also upgrades rows written by the previous driver version)."""
    for old, new in LEGACY_KEYS.items():
        if r.get(new) is None and r.get(old) is not None:
            r[new] = r[old]
    for k in ROW_KEYS:
        r.setdefault(k, None)
    Q = r.get("tokens") or 0
    L, U, macs, allm = r.get("L"), r.get("U"), r.get("ADW_matmul_macs"), r.get("all_macs")
    cred = r.get("ADW_credited_macs")
    r["U_over_L"] = (U / L) if (L and U is not None) else None
    r["L_per_token"] = (L / Q) if (L is not None and Q) else None
    r["U_per_token"] = (U / Q) if (U is not None and Q) else None
    r["kappa_L"] = (cred / L) if (L and cred is not None) else None
    r["kappa_U"] = (cred / U) if (U and cred is not None) else None
    r["kappa_L_matmul"] = (macs / L) if (L and macs is not None) else None
    r["kappa_U_matmul"] = (macs / U) if (U and macs is not None) else None
    r["inv_eta_honest"] = (r["all_work"] / cred) if (cred and r.get("all_work") is not None) else None
    r["kappa_L_allmacs"] = (allm / L) if (L and allm is not None) else None
    r["kappa_U_allmacs"] = (allm / U) if (U and allm is not None) else None
    # SPEC §6: M(C) = credited_macs for training circuits, all_macs for inference (matmul_macs is 0 there)
    M = cred if is_training(r.get("alg") or "") else allm
    r["M"] = M
    r["M_per_token"] = (M / Q) if (M is not None and Q) else None
    r["kappa_low"] = (M / U) if (U and M is not None) else None
    r["kappa_high"] = (M / L) if (L and M is not None) else None
    r["U_n_ru_per_token"] = (r["U_n_ru"] / Q) if (r.get("U_n_ru") is not None and Q) else None
    r["soundness_violation"] = True if (L is not None and U is not None and U < L) else None  # SPEC §3: U >= L
    bits = []
    if r.get("error"):
        bits.append(r["error"])
    if r.get("lower_error"):
        bits.append(f"lower: {r['lower_error']}")
    if r.get("upper_error"):
        bits.append(f"upper: {r['upper_error']}")
    if r.get("soundness_violation"):
        bits.append("U < L")
    r["errors"] = "; ".join(bits) or None
    return r


def is_kill_error(msg: str | None) -> bool:
    return bool(msg) and any(msg.startswith(k) for k in KILL_ERRORS)


def needed_stages(existing: dict | None, *, redo: set[str], resume: bool, retry_errors: bool,
                  versions: dict[str, str], bounds: Bounds) -> set[str]:
    """Which of :data:`STAGES` to (re)compute for a triple given its existing row (``None`` = absent)."""
    avail = {"adw", *(["lower"] if bounds.lower else []), *(["upper"] if bounds.upper else [])}
    if existing is None:
        return set(avail)
    if existing.get("error"):                      # hard error / skip on the whole case
        if resume and not (retry_errors and is_kill_error(existing["error"])):
            return set()
        return set(avail)
    if redo:
        out = set()
        for s in redo & avail:
            if resume and existing.get(f"{s}_src") == versions.get(s) and _stage_value_present(existing, s):
                continue
            out.add(s)
        return out
    if not resume:
        return set(avail)
    out = set()
    for s in avail:
        if _stage_value_present(existing, s):
            continue
        err = existing.get(f"{s}_error") if s != "adw" else None
        if err and not (retry_errors and is_kill_error(err)):
            continue
        out.add(s)
    return out


def _stage_value_present(r: dict, s: str) -> bool:
    if s == "adw":  # every graph-stage column of the current schema must be on the row
        return all(r.get(k) is not None for k in ("ADW_matmul_macs", "ADW_credited_macs", "ADW_recompute_macs"))
    if s == "lower":
        return r.get("L") is not None or bool(r.get("L_infeasible"))
    return r.get("U") is not None


# ---------------------------------------------------------------------------------------------------------
# parent: sweep over cases, one guarded child per case
# ---------------------------------------------------------------------------------------------------------

def _case_to_json(c: Case) -> dict:
    return {"alg": c.alg, "model": c.model, "wl": asdict(c.wl)}


def _case_from_json(d: dict) -> Case:
    wl = dict(d["wl"])
    if isinstance(wl.get("adapted"), list):
        wl["adapted"] = tuple(wl["adapted"])
    return Case(d["alg"], d["model"], Workload(**wl))


def case_cost(c: Case) -> float:
    """Rough extraction + bound cost: active params x tokens (x3 for training-style circuits)."""
    cfg = MODELS[c.model]
    mult = 1.0 if c.alg.startswith("inference") or c.alg in ("forward-nonfixed", "es") else 3.0
    return cfg.active_params() * c.wl.tokens * mult


def compact(out_path: Path, patch: dict | None = None, since: int = 0) -> list[dict]:
    """Rewrite ``out_path`` with one row per key (last wins).  ``patch`` is merged into rows at index >= ``since``
    (the rows a just-finished child wrote, e.g. to add ``peak_rss_mb``)."""
    rows = read_jsonl(out_path)
    if patch:
        for r in rows[since:]:
            r.update(patch)
    rows = dedupe(rows, row_key)
    for r in rows:  # a checkpoint line that was never superseded: the child died before finishing the row
        if r.pop("_partial", None) and r.get("U") is None and not r.get("upper_error"):
            r["upper_error"] = "not run: child exited before the upper_bound call finished"
        finalize_row(r)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(dumps(r) + "\n")
    return rows


def sweep(cases: list[Case], F_values, X_values, out_path: Path, *, G_values=(None,), resume: bool = False,
          redo: set[str] | None = None, retry_errors: bool = False, time_limit_s: float = 900.0,
          lower_time_limit_s: float = 600.0, upper_time_limit_s: float = 600.0, unsafe: bool = False, guarded: bool = True,
          max_rss_mb: float = 3072.0, case_timeout_s: float | None = None, log=print) -> list[dict]:
    """Run every case (cheapest first, as given) for every policy point ``(F, X, G)``; append one JSONL row per
    point to ``out_path`` (``G = None`` = no per-RU work cap).

    See the module docstring for the ``resume`` / ``redo`` / ``retry_errors`` semantics.  Returns the rows
    written or rewritten by this call."""
    out_path = Path(out_path)
    redo = set(redo or ())
    if "all" in redo:
        redo = set(STAGES)
    bounds = load_bounds()
    versions = module_versions()
    if bounds.lower is None:
        log(f"[sweep] lower_bound unavailable: {bounds.lower_error}")
    if bounds.upper is None:
        log(f"[sweep] upper_bound unavailable: {bounds.upper_error}")
    existing = {row_key(r): r for r in compact(out_path)} if (resume or redo) and out_path.exists() else {}
    F_values = [int(f) for f in F_values]
    X_values = [int(x) for x in X_values]
    G_values = [parse_G(gv) for gv in G_values]
    written: list[dict] = []
    for c in cases:
        cfg = MODELS[c.model]
        items = []
        for F in F_values:
            for X in X_values:
                for G in G_values:
                    ex = existing.get(case_key(c, F, X, G))
                    st = needed_stages(ex, redo=redo, resume=resume, retry_errors=retry_errors, versions=versions, bounds=bounds)
                    if st:
                        items.append({"F": F, "X": X, "G": G, "stages": sorted(st), "base": ex})
        tag = f"{c.alg}/{c.model}/Q={c.wl.tokens}" + (f"/S={c.wl.seq}" if c.wl.seq != 4096 else "")
        if not items:
            log(f"[sweep] {tag}: all {len(F_values) * len(X_values) * len(G_values)} rows up to date, skipping")
            continue
        why = ALGORITHMS[c.alg].supports(cfg, c.wl)
        unsafe_why = None if unsafe else check_limits(c.alg, c.model, c.wl, DRIVER_LIMITS)
        if why or unsafe_why:
            msg = f"unsupported: {why}" if why else f"skipped: outside extractor safe limits ({unsafe_why})"
            for it in items:
                r = dict(it["base"] or new_row(c, it["F"], it["X"], it.get("G")), error=msg)
                append_jsonl(out_path, finalize_row(r))
                written.append(r)
            compact(out_path)
            log(f"[sweep] {tag}: {msg}")
            continue
        kw = dict(time_limit_s=time_limit_s, lower_time_limit_s=lower_time_limit_s, upper_time_limit_s=upper_time_limit_s, log=log)
        if guarded:
            rows = run_case_guarded(c, items, out_path, max_rss_mb=max_rss_mb, case_timeout_s=case_timeout_s, **kw)
        else:
            rows = run_case(c, items, out_path, bounds=bounds, **kw)
            compact(out_path)
        written.extend(rows)
    return written


def _spec_path(name: str) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return SCRATCH / f"{name}_{os.getpid()}_{uuid.uuid4().hex[:8]}.json"


def _budget(items: list[dict], time_limit_s: float, lower_s: float, upper_s: float) -> float:
    b = time_limit_s + 120.0
    for it in items:
        b += (lower_s if "lower" in it["stages"] else 0) + (upper_s if "upper" in it["stages"] else 0) + 5
    return b


def run_case_guarded(c: Case, items: list[dict], out_path: Path, *, time_limit_s: float, lower_time_limit_s: float,
                     upper_time_limit_s: float, max_rss_mb: float = 3072.0, case_timeout_s: float | None = None,
                     log=print) -> list[dict]:
    """Run :func:`run_case` in guarded child processes (see the module docstring for the kill recovery)."""
    from accumulation.sweeps.guard import run_guarded
    tag = f"{c.alg}/{c.model}/Q={c.wl.tokens}"
    pending = list(items)
    before = len(read_jsonl(out_path))
    attempts = 0
    peak = 0.0
    elapsed = 0.0
    while pending and attempts <= len(items) + 1:
        attempts += 1
        spec_path = _spec_path("spec")
        inflight_path = _spec_path("inflight")
        spec = dict(_case_to_json(c), items=[{"F": int(it["F"]), "X": int(it["X"]), "G": it.get("G"), "stages": it["stages"],
                                              "base": it["base"]} for it in pending],
                    out=str(out_path), inflight=str(inflight_path), time_limit_s=time_limit_s,
                    lower_time_limit_s=lower_time_limit_s, upper_time_limit_s=upper_time_limit_s)
        write_json(spec_path, spec)
        budget = case_timeout_s if case_timeout_s is not None else _budget(pending, time_limit_s, lower_time_limit_s, upper_time_limit_s)
        cmd = [sys.executable, "-u", "-m", "accumulation.sweeps.run", "--child-spec", str(spec_path)]
        env = {"PYTHONPATH": ".", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        for k in ("ACCSWEEP_TEST_ABORT", "ACCSWEEP_SCRATCH"):
            if k in os.environ:
                env[k] = os.environ[k]
        res = run_guarded(cmd, timeout_s=budget, max_rss_mb=max_rss_mb, cwd=REPO, env=env, log=log)
        peak = max(peak, res.peak_rss_mb)
        elapsed += res.elapsed_s
        for line in (res.stdout or "").splitlines():
            log(line)
        new_rows = read_jsonl(out_path)[before:]
        done = {_pt(r) for r in new_rows if not r.get("_partial")}
        partial = {_pt(r): r for r in new_rows if r.get("_partial")}
        pending = [it for it in pending if _pt(it) not in done]
        try:
            inflight = json.loads(Path(inflight_path).read_text()) if Path(inflight_path).exists() else None
        except ValueError:
            inflight = None
        for p in (spec_path, inflight_path):
            try:
                Path(p).unlink()
            except OSError:
                pass
        if res.status == "ok":
            for it in pending:  # child exited cleanly but wrote nothing for these: a driver bug, record it
                r = dict(it["base"] or new_row(c, it["F"], it["X"], it.get("G")), error="child failed: exited without writing the row")
                append_jsonl(out_path, finalize_row(r))
            pending = []
            break
        msg = f"{res.status}: {res.detail}" if res.status in ("timeout", "oom-guard") else f"child failed: {res.detail}"
        log(f"[sweep] {tag}: {msg} (peak RSS {res.peak_rss_mb:.0f} MB, {res.elapsed_s:.0f}s; in flight: {inflight})")
        stage = (inflight or {}).get("stage")
        # a guard kill or a signal death (returncode < 0: SIGKILL/SIGSEGV) inside one bound call is recoverable
        killed = res.status in ("timeout", "oom-guard") or (res.returncode is not None and res.returncode < 0)
        if not killed or stage in (None, "extract", "adw", "write", "done") or not pending:
            for it in pending:  # nothing salvageable for this case
                r = dict(it["base"] or new_row(c, it["F"], it["X"], it.get("G")), error=msg, peak_rss_mb=round(res.peak_rss_mb, 1),
                         build_s=res.elapsed_s)
                append_jsonl(out_path, finalize_row(r))
            pending = []
            break
        # killed inside a bound call of one (F, X, G): record it, drop that point, respawn for the rest
        pt0 = _pt(inflight)
        victim = [it for it in pending if _pt(it) == pt0]
        pending = [it for it in pending if _pt(it) != pt0]
        for it in victim:
            r = dict(partial.get(pt0) or it["base"] or new_row(c, it["F"], it["X"], it.get("G")))
            r.pop("_partial", None)
            r[f"{stage}_error"] = f"{msg} (killed during {stage})"
            r[f"{stage}_s"] = None
            for s in it["stages"]:
                if s != stage and s != "adw" and not _stage_value_present(r, s):
                    r[f"{s}_error"] = f"not run: child killed during {stage} of this row"
            r["peak_rss_mb"] = round(res.peak_rss_mb, 1)
            append_jsonl(out_path, finalize_row(r))
        if pending:
            log(f"[sweep] {tag}: re-spawning for the remaining {len(pending)} (F, X, G)")
    rows = compact(out_path, patch={"peak_rss_mb": round(peak, 1)}, since=before)
    keys = {case_key(c, it["F"], it["X"], it.get("G")) for it in items}
    return [r for r in rows if row_key(r) in keys]


def _pt(d: dict) -> tuple:
    """``(F, X, G)`` policy point of an item / row / inflight marker."""
    return (int(d["F"]), int(d["X"]), _gkey(d.get("G")))


# ---------------------------------------------------------------------------------------------------------
# child: one case in this process
# ---------------------------------------------------------------------------------------------------------

def _mark(inflight_path: Path | None, **kw) -> None:
    if inflight_path is None:
        return
    try:
        write_json(inflight_path, dict(kw, ts=time.time()))
    except OSError:
        pass


def _test_abort(F: int | None, X: int | None, stage: str) -> None:
    """Test hook: ``ACCSWEEP_TEST_ABORT="F,X,stage"`` kills the child at that point (parent recovery test)."""
    spec = os.environ.get("ACCSWEEP_TEST_ABORT")
    if not spec:
        return
    f, x, s = spec.split(",")
    if s == stage and (s in ("extract", "adw") or (int(f) == F and int(x) == X)):
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGKILL)


def run_case(c: Case, items: list[dict], out_path: Path, *, bounds: Bounds | None = None, time_limit_s: float = 900.0,
             lower_time_limit_s: float = 600.0, upper_time_limit_s: float = 600.0, inflight_path: Path | None = None,
             log=print) -> list[dict]:
    """Build + extract ``c`` once (in this process), then for every item ``{"F", "X", "stages", "base"}`` compute the
    requested stages (keeping the other fields of ``base``) and append one row to ``out_path``."""
    bounds = bounds or load_bounds()
    versions = module_versions()
    cfg = MODELS[c.model]
    rows: list[dict] = []
    tag = f"{c.alg}/{c.model}/Q={c.wl.tokens}"
    _mark(inflight_path, stage="extract")
    _test_abort(None, None, "extract")
    t0 = time.perf_counter()
    try:
        with time_limit(time_limit_s):
            bp = build(c.alg, cfg, c.wl)
            t1 = time.perf_counter()
            g = extract(bp)
            t2 = time.perf_counter()
    except (StepTimeout, Exception) as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        for it in items:
            r = dict(it["base"] or new_row(c, it["F"], it["X"], it.get("G")), error=f"build/extract failed: {msg}", build_s=time.perf_counter() - t0)
            append_jsonl(out_path, finalize_row(r))
            rows.append(r)
        log(f"[sweep] {tag}: build/extract failed after {time.perf_counter() - t0:.0f}s: {msg}")
        return rows
    _mark(inflight_path, stage="adw")
    _test_abort(None, None, "adw")
    ta = time.perf_counter()
    a = compute_adw(g)
    adw_s = time.perf_counter() - ta
    P = bp.state_bytes()
    tok_seed = bp.param_bytes("token") + bp.param_bytes("seed")
    Q = c.wl.tokens
    log(f"[sweep] {tag}: ops={len(g.ops)} build={t1 - t0:.1f}s extract={t2 - t1:.1f}s adw={adw_s:.1f}s "
        f"ADW_macs={a.matmul_macs:.3e} credited={getattr(a, 'credited_macs', 0):.3e} P={P:.3e}")
    ctx = {"roles": bp.roles, "roles_ok": bp.roles, "adw": a, "workload": c.wl, "wl": c.wl, "bp": bp,
           "program": bp.program, "tokens": Q, "P": P}
    graph_fields = dict(ADW_total=a.total, ADW_matmul=a.matmul, ADW_matmul_macs=a.matmul_macs,
                        ADW_credited_macs=getattr(a, "credited_macs", None), ADW_recompute_macs=getattr(a, "recompute_macs", None),
                        all_work=a.all_work, all_macs=a.all_macs, P=P, token_seed_bytes=tok_seed, n_ops=len(g.ops), build_s=t1 - t0,
                        extract_s=t2 - t1, adw_s=adw_s, adw_src=versions["adw"], extract_src=versions["extract"])
    lower_takes_G = bounds.lower is not None and "G" in inspect.signature(bounds.lower).parameters
    lower_cache: dict[tuple, dict] = {}   # (F, X) -> lower fields, reused across G while lower_bound ignores G
    for it in items:
        F, X, G, stages = int(it["F"]), int(it["X"]), parse_G(it.get("G")), set(it["stages"])
        r = dict(it["base"]) if it["base"] else new_row(c, F, X, G)
        r["G"] = G
        r["error"] = None
        ctx_pt = dict(ctx, G=G)
        if "adw" in stages or not _stage_value_present(r, "adw"):
            r.update(graph_fields)
        else:  # keep the recorded ADW but refresh the timings of this run
            r.update(build_s=t1 - t0, extract_s=t2 - t1)
        LOWER_FIELDS = ("L", "L_source", "L_cap", "L_infeasible", "L_at_G", "L_worst_ru", "lower_notes", "lower_error", "lower_s")
        if "lower" in stages:
            r.update({k: None for k in LOWER_FIELDS})
            if bounds.lower is None:
                r["lower_error"] = bounds.lower_error
            elif not lower_takes_G and (F, X) in lower_cache:
                r.update(lower_cache[(F, X)])
                r["lower_s"] = 0.0
            else:
                _mark(inflight_path, stage="lower", F=F, X=X, G=G)
                _test_abort(F, X, "lower")
                tl = time.perf_counter()
                try:
                    with time_limit(lower_time_limit_s):
                        lb = call_bound(bounds.lower, g, F, X, **ctx_pt)
                    r["L_source"] = _opt_int(getattr(lb, "source", None))
                    r["L_cap"] = _opt_int(getattr(lb, "cap", None))
                    r["lower_notes"] = list(getattr(lb, "notes", []) or [])[:8]
                    # the certificate's most constraining legal RU (THEORY §0.1 worst-RU diagnostic), when provided
                    wr = getattr(lb, "worst_ru", None)
                    r["L_worst_ru"] = _jsonable(wr) if wr is not None else None
                    # until lower_bound takes G, L is computed at G = None: valid (looser) for every G
                    r["L_at_G"] = ("G=" + ("None" if G is None else f"{G:.0e}")) if lower_takes_G else "G=None"
                    if getattr(lb, "infeasible", False):
                        r["L_infeasible"] = True
                        r["lower_error"] = "infeasible: some op cannot execute a single MAC inside an RU of X bytes"
                    else:
                        r["L"] = int(lb.total)
                except (StepTimeout, Exception) as e:  # noqa: BLE001
                    r["lower_error"] = f"{type(e).__name__}: {e}"
                    log(f"[sweep]   lower_bound failed F={F:.0e} X={X:.0e} G={G}: {r['lower_error']}"
                        + ("" if isinstance(e, (ValueError, StepTimeout)) else "\n" + traceback.format_exc(limit=3)))
                r["lower_s"] = time.perf_counter() - tl
                if not lower_takes_G:
                    lower_cache[(F, X)] = {k: r[k] for k in LOWER_FIELDS}
            r["lower_src"] = versions["lower"]
            if "upper" in stages:  # checkpoint: if the upper call is killed, the parent recovers L from this line
                append_jsonl(out_path, dict(finalize_row(dict(r)), _partial="upper pending"))
        if "upper" in stages:
            r.update(U=None, U_units=None, U_kappa=None, U_by_class=None, U_recompute=None, U_n_ru=None, U_ru_input_mean=None,
                     U_ru_input_max=None, U_ru_work_mean=None, U_ru_work_max=None, U_ru_up_max=None, upper_notes=None,
                     upper_error=None, upper_s=None)
            if bounds.upper is None:
                r["upper_error"] = bounds.upper_error
            else:
                _mark(inflight_path, stage="upper", F=F, X=X, G=G)
                _test_abort(F, X, "upper")
                tu = time.perf_counter()
                try:
                    with time_limit(upper_time_limit_s):
                        ub = call_bound(bounds.upper, g, F, X, **ctx_pt)
                    r["U"] = int(ub.total)
                    r["U_units"] = _opt_int(getattr(ub, "n_units", None))
                    k = getattr(ub, "kappa", None)
                    r["U_kappa"] = None if k is None or k == float("inf") else float(k)
                    r["upper_notes"] = list(getattr(ub, "notes", []) or [])[:8]
                    bc = getattr(ub, "by_class", None)
                    r["U_by_class"] = {str(k2): int(v) for k2, v in bc.items()} if bc else None
                    plan = getattr(ub, "plan", None)
                    if plan is not None:
                        r["U_recompute"] = bool(getattr(plan, "recompute", None))
                        if hasattr(plan, "ru_stats"):
                            st = plan.ru_stats()
                            r["U_n_ru"] = _opt_int(st.get("n_ru"))
                            r["U_ru_input_mean"] = st.get("input_mean")
                            r["U_ru_input_max"] = _opt_int(st.get("input_max"))
                            r["U_ru_work_mean"] = st.get("work_mean")
                            r["U_ru_work_max"] = _opt_int(st.get("work_max"))
                            r["U_ru_up_max"] = _opt_int(st.get("up_max"))
                except (StepTimeout, Exception) as e:  # noqa: BLE001
                    r["upper_error"] = f"{type(e).__name__}: {e}"
                    log(f"[sweep]   upper_bound failed F={F:.0e} X={X:.0e} G={G}: {r['upper_error']}"
                        + ("" if isinstance(e, (ValueError, StepTimeout)) else "\n" + traceback.format_exc(limit=3)))
                r["upper_s"] = time.perf_counter() - tu
                r["upper_src"] = versions["upper"]
        r["ts"] = time.time()
        finalize_row(r)
        _mark(inflight_path, stage="write", F=F, X=X, G=G)
        append_jsonl(out_path, r)
        rows.append(r)
        log(f"[sweep]   F={F:.0e} X={X:.0e} G={'None' if G is None else f'{G:.0e}'}  L={_e(r['L'])} U={_e(r['U'])} "
            f"U/L={_e(r['U_over_L'])} kappa_low={_e(r['kappa_low'])} kappa_high={_e(r['kappa_high'])} "
            f"n_RU={r['U_n_ru']} (lower {r['lower_s'] or 0:.1f}s, upper {r['upper_s'] or 0:.1f}s)")
    _mark(inflight_path, stage="done")
    return rows


def child_main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    c = _case_from_json(spec)
    run_case(c, spec["items"], Path(spec["out"]), time_limit_s=spec["time_limit_s"],
             lower_time_limit_s=spec["lower_time_limit_s"], upper_time_limit_s=spec["upper_time_limit_s"],
             inflight_path=Path(spec["inflight"]) if spec.get("inflight") else None)
    return 0


# ---------------------------------------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------------------------------------

REAL_DENSE = ("llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b")
REAL_MOE = ("mixtral-8x7b", "deepseek-v3", "qwen3-235b-a22b")
DENSE_ALGS = ("inference-dense", "forward-nonfixed", "pretrain-dense", "local-sgd", "lora", "hidden-adapter",
              "rl-policy-gradient", "es")
MOE_ALGS = ("inference-moe", "pretrain-moe")
ALL_ADAPTED = ("q", "k", "v", "o", "g", "u", "d")

MB = 1_000_000
P1_TOKENS = (4096, 32768, 262144, 1_048_576, 4_194_304)
P1_F, P1_X = int(1e12), 5 * MB
P2_X = (1 * MB, 5 * MB, 20 * MB, 100 * MB, 1000 * MB, 64_000 * MB)
P2_F = (int(1e10), int(1e12), int(1e14), 1 << 62)


def p2_workload(alg: str, tokens: int = 4096, seq: int = 4096, **kw) -> Workload:
    """``Q = seq = 4096`` for everything except the split families: ES needs ``tokens = population x seq`` and
    local-SGD ``tokens = local_steps x seq`` (``Algorithm.supports``); both use 2 members / 2 steps -> 8192 tokens
    (1 member / 1 step would degenerate to forward-nonfixed / pretrain-dense)."""
    if alg == "es":
        return Workload(tokens=2 * seq, seq=seq, population=2, **kw)
    if alg == "local-sgd":
        return Workload(tokens=2 * seq, seq=seq, local_steps=2, **kw)
    if alg in ("lora", "hidden-adapter"):
        return Workload(tokens=tokens, seq=seq, rank=kw.pop("rank", 16), **kw)
    if alg == "rl-policy-gradient":
        return Workload(tokens=tokens, seq=seq, gen_tokens=kw.pop("gen_tokens", 2048), **kw)
    return Workload(tokens=tokens, seq=seq, **kw)


def smoke_cases() -> list[Case]:
    cases = []
    for alg in DENSE_ALGS:
        wl = {"es": Workload(tokens=8, seq=4, population=2), "local-sgd": Workload(tokens=8, seq=4, local_steps=2),
              "lora": Workload(tokens=8, seq=4, rank=2), "hidden-adapter": Workload(tokens=8, seq=4, rank=2),
              "rl-policy-gradient": Workload(tokens=8, seq=4, gen_tokens=2)}.get(alg, Workload(tokens=8, seq=4))
        cases.append(Case(alg, "tiny2", wl))
    for alg in MOE_ALGS:
        cases.append(Case(alg, "tiny-moe", Workload(tokens=8, seq=4)))
    return cases


def p1_cases(max_tokens: int = 4_194_304, model: str = "llama3-8b", seq: int = 4096) -> list[Case]:
    """Q scaling of the three headline circuits: pretrain-dense, lora r = 16 (all projections), inference-dense."""
    cases = []
    for t in (t for t in P1_TOKENS if t <= max_tokens):
        cases.append(Case("inference-dense", model, Workload(tokens=t, seq=seq)))
        cases.append(Case("lora", model, Workload(tokens=t, seq=seq, rank=16, adapted=ALL_ADAPTED)))
        cases.append(Case("pretrain-dense", model, Workload(tokens=t, seq=seq)))
    return sorted(cases, key=case_cost)


def p2_cases(seq_variants: bool = True, lora_variants: bool = True) -> list[Case]:
    cases = []
    for m in REAL_DENSE:
        for alg in DENSE_ALGS:
            cases.append(Case(alg, m, p2_workload(alg)))
        if lora_variants:
            for r in (1, 16, 64):
                for ad in (("q", "v"), ALL_ADAPTED):
                    if r == 16 and ad == ALL_ADAPTED:
                        continue  # the default row
                    cases.append(Case("lora", m, Workload(tokens=4096, seq=4096, rank=r, adapted=ad)))
    for m in REAL_MOE:
        for alg in MOE_ALGS:
            cases.append(Case(alg, m, p2_workload(alg)))
    if seq_variants:
        cases.append(Case("pretrain-dense", "llama3-8b", Workload(tokens=4096, seq=1024)))
        cases.append(Case("pretrain-dense", "llama3-8b", Workload(tokens=8192, seq=8192)))
    return sorted(cases, key=case_cost)


def p2_8b_point_cases() -> list[Case]:
    """All dense algorithms on llama3-8b at the (X = 5 MB, F = 1e12) grid point (feeds the P3 table at 8B)."""
    return sorted([Case(alg, "llama3-8b", p2_workload(alg)) for alg in DENSE_ALGS], key=case_cost)


# --- policy-frontier presets (SPEC §6): the primary object is I*(C; F, G, X) ------------------------------

FRONTIER_F = (int(1e10), int(1e11), int(1e12), 1 << 62)           # 1 << 62 = "inf-ish"
FRONTIER_G = (None, int(1e10), int(3e9), int(1e9), int(3e8), int(1e8), int(3e7), int(1e7))
FRONTIER_X = (5 * MB,)
QSCALING_TOKENS = (4096, 16384, 65536, 262144, 1_048_576)
SCALE_DENSE = ("llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b")


def frontier_cases(model: str = "llama3-8b", tokens: int = 4096, seq: int = 4096) -> list[Case]:
    """Matched inference-dense / pretrain-dense pair for the (F, G) policy frontier."""
    return [Case("inference-dense", model, Workload(tokens=tokens, seq=seq)),
            Case("pretrain-dense", model, Workload(tokens=tokens, seq=seq))]


def scale_cases() -> list[Case]:
    """Artifact B: full-parameter training and inference, dense 1B..405B and the three MoE models (no LoRA)."""
    cases = []
    for m in SCALE_DENSE:
        cases += [Case("inference-dense", m, Workload(4096, 4096)), Case("pretrain-dense", m, Workload(4096, 4096))]
    for m in REAL_MOE:
        cases += [Case("inference-moe", m, Workload(4096, 4096)), Case("pretrain-moe", m, Workload(4096, 4096))]
    return sorted(cases, key=case_cost)


def qscaling_cases(max_tokens: int = 1_048_576, models=("llama3-8b", "llama3-1b"), seq: int = 4096) -> list[Case]:
    """Artifact C: Q scaling of inference-dense / pretrain-dense at the operating points (F, G given on the CLI)."""
    cases = []
    for m in models:
        for t in (t for t in QSCALING_TOKENS + (4_194_304,) if t <= max_tokens):
            cases += [Case("inference-dense", m, Workload(tokens=t, seq=seq)), Case("pretrain-dense", m, Workload(tokens=t, seq=seq))]
    return sorted(cases, key=case_cost)


PRESETS = {
    "smoke": dict(cases=smoke_cases, F=(1 << 20, 1 << 40), X=(64, 4096), file="sweep_smoke.jsonl"),
    "p1": dict(cases=p1_cases, F=(P1_F,), X=(P1_X,), file="sweep_p1.jsonl"),
    "p2": dict(cases=p2_cases, F=P2_F, X=P2_X, file="sweep_p2.jsonl"),
    "p2-8b-point": dict(cases=p2_8b_point_cases, F=(P1_F,), X=(P1_X,), file="sweep_p2.jsonl"),
    "frontier-8b": dict(cases=frontier_cases, F=FRONTIER_F, X=FRONTIER_X, G=FRONTIER_G, file="frontier_8b.jsonl"),
    "frontier-8b-64k": dict(cases=lambda: frontier_cases(tokens=65536), F=FRONTIER_F, X=FRONTIER_X, G=FRONTIER_G,
                            file="frontier_8b.jsonl"),
    "smoke-G": dict(cases=lambda: [Case("inference-dense", "tiny2", Workload(8, 4)), Case("pretrain-dense", "tiny2", Workload(8, 4))],
                    F=(1 << 40,), X=(4096,), G=(None, 4096, 256), file="sweep_smoke.jsonl"),
    "scale": dict(cases=scale_cases, F=FRONTIER_F, X=FRONTIER_X, G=FRONTIER_G, file="scale.jsonl"),
    "qscaling": dict(cases=qscaling_cases, F=FRONTIER_F, X=FRONTIER_X, G=FRONTIER_G, file="qscaling.jsonl"),
}


# --- p1 fits and p3: efficiency translation (the math lives in accumulation.sweeps.analysis) --------------

def load_sweep_rows(out_dir: Path, names=("sweep_p1.jsonl", "sweep_p2.jsonl")) -> list[dict]:
    rows = []
    for name in names:
        rows += read_jsonl(Path(out_dir) / name)
    return dedupe([finalize_row(r) for r in rows], row_key)


def run_p1_fit(out_dir: Path = RESULTS_DIR, log=print) -> dict:
    from accumulation.sweeps.analysis import p1_fits
    rows = [finalize_row(r) for r in read_jsonl(Path(out_dir) / "sweep_p1.jsonl")]
    res = p1_fits(dedupe(rows, row_key))
    write_json(Path(out_dir) / "p1_fit.json", res)
    log(f"[p1-fit] {len(res['fits'])} fits -> {Path(out_dir) / 'p1_fit.json'}")
    for f in res["fits"]:
        log(f"[p1-fit]   {f['alg']:18s} {f['model']} {f['bound']}: r={f['slope']:.4g} B/token intercept={f['intercept']:.4g} "
            f"(P={f['P']:.4g}) n={f['n']} resid_max={f['resid_rel_max']:.2%}")
    return res


def run_p3(out_dir: Path = RESULTS_DIR, log=print) -> dict:
    from accumulation.sweeps.analysis import p1_fits, p3_from_rows
    out_dir = Path(out_dir)
    rows = load_sweep_rows(out_dir)
    p1 = [finalize_row(r) for r in read_jsonl(out_dir / "sweep_p1.jsonl")]
    fits = p1_fits(dedupe(p1, row_key)) if p1 else {"fits": []}
    res = p3_from_rows(rows, fits)
    write_json(out_dir / "p3_efficiency.json", res)
    log(f"[p3] {len(res['rows'])} rows -> {out_dir / 'p3_efficiency.json'}")
    return res


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="bounded-accumulation sweeps (JSONL, resumable, guarded)")
    ap.add_argument("--preset", choices=(*PRESETS, "p1-fit", "p3"), action="append",
                    help="one or more presets (run in order); p1-fit / p3 are analysis passes over the JSONL files")
    ap.add_argument("--resume", action="store_true", help="skip triples whose row is complete")
    ap.add_argument("--redo", choices=("lower", "upper", "adw", "all"), action="append", default=[],
                    help="recompute only these stages (keep the rest of each existing row)")
    ap.add_argument("--retry-errors", action="store_true", help="with --resume: re-run stages killed by the guard")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--time-limit", type=float, default=900.0, help="build+extract wall-clock budget per case (s)")
    ap.add_argument("--lower-time-limit", type=float, default=600.0, help="wall-clock budget per lower_bound call (s)")
    ap.add_argument("--upper-time-limit", type=float, default=600.0, help="wall-clock budget per upper_bound call (s)")
    ap.add_argument("--max-rss-mb", type=float, default=3072.0, help="RSS ceiling for each guarded case child")
    ap.add_argument("--p1-max-tokens", type=int, default=4_194_304, help="largest Q in the p1 / qscaling presets")
    ap.add_argument("--F", nargs="*", default=None, help="override the preset's F values (floats ok, e.g. 1e12)")
    ap.add_argument("--X", nargs="*", default=None, help="override the preset's X values (bytes)")
    ap.add_argument("--G", nargs="*", default=None, help="override the preset's per-RU work caps ('none' = unbounded)")
    ap.add_argument("--unsafe", action="store_true", help="run cases outside guard.DRIVER_LIMITS (still guarded)")
    ap.add_argument("--models", nargs="*", default=None, help="restrict to these models")
    ap.add_argument("--algs", nargs="*", default=None, help="restrict to these algorithms")
    ap.add_argument("--tokens", nargs="*", type=int, default=None, help="restrict to these token counts")
    ap.add_argument("--in-process", action="store_true", help="run cases in this process (tests only: no watchdog)")
    ap.add_argument("--child-spec", help="internal: run one case from a JSON spec file in this process")
    args = ap.parse_args(argv)
    if args.child_spec:
        return child_main(args.child_spec)
    if not args.preset:
        ap.error("choose at least one --preset")
    for name in args.preset:
        if name == "p1-fit":
            run_p1_fit(args.out_dir)
            continue
        if name == "p3":
            run_p3(args.out_dir)
            continue
        spec = PRESETS[name]
        cases = spec["cases"](args.p1_max_tokens) if name in ("p1", "qscaling") else spec["cases"]()
        F_vals = [int(float(f)) for f in args.F] if args.F else spec["F"]
        X_vals = [int(float(x)) for x in args.X] if args.X else spec["X"]
        G_vals = [parse_G(gv) for gv in args.G] if args.G else spec.get("G", (None,))
        if args.models:
            cases = [c for c in cases if c.model in args.models]
        if args.algs:
            cases = [c for c in cases if c.alg in args.algs]
        if args.tokens:
            cases = [c for c in cases if c.wl.tokens in args.tokens]
        print(f"[sweep] preset {name}: {len(cases)} cases x {len(F_vals)} F x {len(X_vals)} X x {len(G_vals)} G -> "
              f"{args.out_dir / spec['file']} (resume={args.resume} redo={args.redo or '-'} limits: extract {args.time_limit:.0f}s, "
              f"lower {args.lower_time_limit:.0f}s, upper {args.upper_time_limit:.0f}s, rss {args.max_rss_mb:.0f} MB)", flush=True)
        sweep(cases, F_vals, X_vals, args.out_dir / spec["file"], G_values=G_vals, resume=args.resume, redo=set(args.redo),
              retry_errors=args.retry_errors, time_limit_s=args.time_limit, lower_time_limit_s=args.lower_time_limit,
              upper_time_limit_s=args.upper_time_limit, unsafe=args.unsafe, guarded=not args.in_process,
              max_rss_mb=args.max_rss_mb, log=lambda *a: print(*a, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
