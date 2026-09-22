"""bench/profile_check.py — where the CPU checker's time and memory go on ONE saved run (perf owner, window 2026-09-17).

    python -m verity_capture.bench.profile_check --run /workspace/cp/cov/runs/<run> --out <run>/profile \\
        [--stages fold,compare,wiring,replay_chain] [--k 3] [--steps 0-3] [--stage-timeout 900] [--max-rss-gb 38] \\
        [--total-budget-s 1500] [--inprocess-budget-s 600] [--python PY] [--no-inprocess] [--no-cprofile]

Two measurements, both read-only on the run directory (every output goes to --out):

1. **Stage subprocesses.**  The CPU stages of `bench/run_config.py` (`fold` worker, `compare`, `wiring` = provenance, `replay_chain`,
   optionally `replay_a`) are launched one at a time with the same command lines the driver builds, under `/usr/bin/time -v` (peak RSS,
   wall, user/sys, page faults, file-system inputs) plus an RSS watchdog (the process tree is polled; a stage above `--max-rss-gb` or
   `--stage-timeout` is killed and recorded as such, its peak so far kept).  Each stage's own report (`fold_seconds`, `seconds`,
   `max_rss_gb`) is read back so the fold is separated from the check inside the stage.

2. **In-process profile of tier A.**  `read_log` (log parsing), `fold` (Program), `SnapshotStore` (index), `WeightSource` (checkpoint) are
   timed one after the other with the process RSS after each; then `replay.tier_a(k, steps)` runs under `cProfile` with instrumentation
   wrapped around (not edited into) `Replayer.evaluate_instance` / `evaluate_child` (time and count per *family* = spec id before `{`, and
   per attention T), `SnapshotStore._raw` (snapshot files read: count, bytes, seconds — the store caches every file's words for the life
   of the process) and `Values.of_run` / `Replayer.expected_words`.  A `BaseException` alarm (`--inprocess-budget-s`) bounds the run: tier A
   catches `Exception` per evaluation, so an ordinary exception would be swallowed as an "error" entry.  From the fold alone the tool also
   counts, per family, the specializations and the evaluations a full tier A at `k` would need (sum over (spec, regime) strata of
   min(k, instances)), and projects the family's full tier-A seconds from the measured per-evaluation cost (attention: a linear fit of
   seconds on T).

Outputs: `<out>/profile.json` (everything), `<out>/PROFILE.md` (the summary tables), `<out>/tier_a.prof` (pstats), `<out>/logs/<stage>.log`,
`<out>/time_<stage>.txt`, and the stage reports `<out>/<stage>.json`.  Nothing in the run directory is written or modified.  Torch-free.
"""
from __future__ import annotations

import argparse
import cProfile
import glob
import io
import json
import os
import pstats
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------

_T_RE = re.compile(r"\bT=(\d+)")


def family_of(spec: str) -> str:
    return spec.split("{")[0]


def attention_T(spec: str) -> int | None:
    m = _T_RE.search(spec)
    return int(m.group(1)) if m else None


def rss_now() -> dict[str, float]:
    """Current and high-water RSS of this process in GB (Linux /proc; macOS falls back to ru_maxrss for both)."""
    try:
        cur = hwm = None
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 2 ** 20
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 2 ** 20
        if cur is not None:
            return {"rss_gb": round(cur, 3), "hwm_gb": round(hwm or cur, 3)}
    except OSError:
        pass
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gb = r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20
    return {"rss_gb": round(gb, 3), "hwm_gb": round(gb, 3)}


def _tree_rss_gb(root_pid: int) -> float | None:
    """Summed RSS (GB) of `root_pid` and its descendants (one `ps` call)."""
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    kids: dict[int, list[int]] = defaultdict(list)
    rss: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, r = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        kids[ppid].append(pid)
        rss[pid] = r
    if root_pid not in rss:
        return None
    total, stack = 0, [root_pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, ()))
    return total / 2 ** 20


def parse_time_v(text: str) -> dict[str, Any]:
    """The fields of `/usr/bin/time -v` we report."""
    out: dict[str, Any] = {}
    keys = {"Maximum resident set size (kbytes)": ("max_rss_gb", lambda v: round(int(v) / 2 ** 20, 3)),
            "User time (seconds)": ("user_s", float), "System time (seconds)": ("sys_s", float),
            "Elapsed (wall clock) time (h:mm:ss or m:ss)": ("wall_s", None),
            "Major (requiring I/O) page faults": ("major_faults", int), "Minor (reclaiming a frame) page faults": ("minor_faults", int),
            "File system inputs": ("fs_inputs", int), "File system outputs": ("fs_outputs", int),
            "Exit status": ("exit_status", int), "Percent of CPU this job got": ("cpu_percent", lambda v: int(v.rstrip("%")))}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        k, _, v = line.rpartition(": ")
        k = k.strip()
        if k in keys:
            name, conv = keys[k]
            try:
                if name == "wall_s":
                    parts = [float(x) for x in v.strip().split(":")]
                    out[name] = round(sum(p * 60 ** i for i, p in enumerate(reversed(parts))), 3)
                else:
                    out[name] = conv(v.strip())
            except ValueError:
                out[name] = v.strip()
        elif "terminated by signal" in line:
            out["signal"] = line
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return repr(x)


def parse_steps(s: str | None, all_steps: list[int]) -> list[int] | None:
    if s is None or s.strip().lower() == "all":
        return None
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return sorted(out & set(all_steps)) if all_steps else sorted(out)


# ---------------------------------------------------------------------------------------------------------
# 1. stage subprocesses
# ---------------------------------------------------------------------------------------------------------

STAGES = ("fold", "compare", "wiring", "replay_chain", "replay_a")


def stage_cmd(stage: str, *, py: str, log: Path, snaps: Path, profile: str, out: Path, k: int, seed: int, safetensors: str | None,
              chains: int) -> list[str]:
    """The driver's command line for `stage` (bench/run_config.Plan.cmd), outputs redirected to `out`."""
    if stage == "fold":
        return [py, "-m", "verity_capture.bench.run_config", "--worker", "fold", "--log", str(log), "--profile", profile, "--out", str(out / "fold_summary.json")]
    if stage == "compare":
        c = [py, "-m", "verity_capture.compare", "--log", str(log), "--snapshots", str(snaps), "--profile", profile, "--out", str(out / "compare.json")]
        return c + (["--safetensors", safetensors] if safetensors else [])
    if stage == "wiring":
        return [py, "-m", "verity_capture.provenance", "--log", str(log), "--profile", profile, "--out", str(out / "wiring.json")]
    if stage == "replay_chain":
        c = [py, "-m", "verity_capture.replay", "--log", str(log), "--snapshots", str(snaps), "--profile", profile, "--tier", "chain",
             "--chains", str(chains), "--seed", str(seed), "--out", str(out / "replay_chain.json")]
        return c + (["--safetensors", safetensors] if safetensors else [])
    if stage == "replay_a":
        c = [py, "-m", "verity_capture.replay", "--log", str(log), "--snapshots", str(snaps), "--profile", profile, "--tier", "a",
             "--k", str(k), "--seed", str(seed), "--out", str(out / "replay_a.json")]
        return c + (["--safetensors", safetensors] if safetensors else [])
    raise ValueError(stage)


def run_stage(stage: str, cmd: list[str], out: Path, *, env: dict, max_rss_gb: float | None, timeout_s: float | None) -> dict:
    """One stage under `/usr/bin/time -v` (when present) with an RSS / wall watchdog over the process tree."""
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tfile = out / f"time_{stage}.txt"
    time_bin = shutil.which("time", path="/usr/bin:/bin:/usr/local/bin")
    gnu_time = False
    if time_bin:
        try:
            probe = subprocess.run([time_bin, "-v", "true"], capture_output=True, text=True, timeout=10)
            gnu_time = "Maximum resident set size" in probe.stderr
        except (OSError, subprocess.TimeoutExpired):
            gnu_time = False
    if gnu_time:
        full = [time_bin, "-v", "-o", str(tfile)] + cmd
    else:   # the pods have no /usr/bin/time: the same numbers from getrusage(RUSAGE_CHILDREN) in a tiny wrapper (`--time-wrap`)
        full = [cmd[0], "-m", "verity_capture.bench.profile_check", "--time-wrap", str(tfile), "--"] + cmd
    rec: dict[str, Any] = {"stage": stage, "cmd": shlex.join(cmd), "gnu_time": gnu_time, "status": "ok", "returncode": None, "reason": None,
                           "log": str(logs / f"{stage}.log")}
    t0 = time.time()
    peak = 0.0
    samples: list[tuple[float, float]] = []
    with open(logs / f"{stage}.log", "w") as lf:
        lf.write(f"$ {shlex.join(full)}\n")
        lf.flush()
        proc = subprocess.Popen(full, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        last_sample = 0.0
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            now = time.time()
            if now - last_sample >= 1.0:
                last_sample = now
                r = _tree_rss_gb(proc.pid)
                if r is not None:
                    peak = max(peak, r)
                    if len(samples) < 4000:
                        samples.append((round(now - t0, 1), round(r, 3)))
                    if max_rss_gb is not None and r > max_rss_gb:
                        os.killpg(proc.pid, signal.SIGKILL)
                        rec.update(status="killed_memory", reason=f"tree RSS {r:.2f} GB > {max_rss_gb} GB")
                        proc.wait()
                        break
            if timeout_s is not None and now - t0 > timeout_s:
                os.killpg(proc.pid, signal.SIGKILL)
                rec.update(status="timeout", reason=f"wall > {timeout_s:.0f} s")
                proc.wait()
                break
            time.sleep(0.25)
    rec["returncode"] = proc.returncode
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["peak_tree_rss_gb"] = round(peak, 3)
    rec["rss_samples"] = samples[:: max(1, len(samples) // 120)]      # <= ~120 points for the report
    if tfile.exists():
        try:
            rec["time_v"] = parse_time_v(tfile.read_text()) if gnu_time else json.loads(tfile.read_text())
        except ValueError as e:
            rec["time_v_error"] = f"{type(e).__name__}: {e}"
    if rec["status"] == "ok" and proc.returncode != 0:
        rec.update(status="failed", reason=f"exit {proc.returncode}; see {rec['log']}")
    # the stage's own numbers (fold vs check inside the process)
    report_file = {"fold": "fold_summary.json", "compare": "compare.json", "wiring": "wiring.json", "replay_chain": "replay_chain.json",
                   "replay_a": "replay_a.json"}[stage]
    p = out / report_file
    if p.exists():
        try:
            d = json.loads(p.read_text())
            rec["report"] = {kk: d.get(kk) for kk in ("status", "fold_seconds", "seconds", "total_seconds", "max_rss_gb", "digest", "n_specs",
                                                      "n_specs_covered", "strata", "summary", "n_slices", "n_edges_chained")}
            if stage == "fold":
                rec["report"].update({"n_instances": d.get("n_instances"), "n_steps": d.get("n_steps"), "timings": d.get("timings"),
                                      "memory": {kk: v for kk, v in (d.get("memory") or {}).items() if not isinstance(v, (list, dict))}})
            if stage == "wiring":
                rec["report"].update({kk: d.get(kk) for kk in ("n_checked", "n_operands", "checked", "mismatches", "n_instances")})
            if stage == "replay_a":
                rec["report"]["per_family"] = per_family_from_report(d)
        except (OSError, ValueError) as e:
            rec["report_error"] = f"{type(e).__name__}: {e}"
    return rec


def per_family_from_report(rep: dict) -> dict:
    fam: dict[str, dict] = {}
    for e in rep.get("instances", []):
        f = fam.setdefault(family_of(e["spec"]), {"evaluations": 0, "seconds": 0.0, "status": Counter()})
        f["evaluations"] += 1
        f["seconds"] += float(e.get("seconds") or 0.0)
        f["status"][e["status"]] += 1
    for s in rep.get("coverage", {}):
        fam.setdefault(family_of(s), {"evaluations": 0, "seconds": 0.0, "status": Counter()}).setdefault("specs", 0)
        fam[family_of(s)]["specs"] = fam[family_of(s)].get("specs", 0) + 1
    for f in fam.values():
        f["seconds"] = round(f["seconds"], 3)
        f["status"] = dict(f["status"])
    return fam


# ---------------------------------------------------------------------------------------------------------
# 2. in-process profile of tier A
# ---------------------------------------------------------------------------------------------------------


class _Budget(BaseException):
    """Raised by the alarm: NOT an Exception, so `tier_a`'s per-evaluation `except Exception` does not swallow it."""


class Instrument:
    """Wrappers around the checker's hot methods (installed on the classes for this process only; never edits the modules)."""

    def __init__(self) -> None:
        self.family_seconds: Counter = Counter()
        self.family_evals: Counter = Counter()
        self.family_status: dict[str, Counter] = defaultdict(Counter)
        self.attention: list[tuple[int, float, str]] = []          # (T, seconds, status)
        self.snapshot_loads = 0
        self.snapshot_bytes = 0
        self.snapshot_seconds = 0.0
        self.snapshot_hits = 0
        self.of_run_calls = 0
        self.of_run_seconds = 0.0
        self.expected_calls = 0
        self.expected_seconds = 0.0
        self.operand_calls = 0
        self.operand_seconds = 0.0
        self.evaluate_calls = 0
        self.evaluate_seconds = 0.0
        self._orig: list[tuple[Any, str, Any]] = []

    def install(self) -> None:
        from verity_capture import compare as cmp
        from verity_capture import replay as rp
        inst = self

        def wrap_eval(name):
            orig = getattr(rp.Replayer, name)

            def w(self_, m, *a, **kw):
                spec = m["spec"]
                fam = family_of(spec)
                t0 = time.time()
                st = "error"
                try:
                    e = orig(self_, m, *a, **kw)
                    st = e.get("status", "?")
                    return e
                except rp.Skipped:
                    st = "skipped"
                    raise
                finally:
                    dt = time.time() - t0
                    inst.family_seconds[fam] += dt
                    inst.family_evals[fam] += 1
                    inst.family_status[fam][st] += 1
                    T = attention_T(spec)
                    if fam.startswith("Attention") and T is not None:
                        inst.attention.append((T, dt, st))
            self._orig.append((rp.Replayer, name, orig))
            setattr(rp.Replayer, name, w)

        wrap_eval("evaluate_instance")
        wrap_eval("evaluate_child")

        orig_raw = cmp.SnapshotStore._raw

        def raw(self_, s):
            if s._words is not None:
                inst.snapshot_hits += 1
                return s._words
            t0 = time.time()
            w = orig_raw(self_, s)
            inst.snapshot_seconds += time.time() - t0
            inst.snapshot_loads += 1
            inst.snapshot_bytes += int(w.nbytes)
            return w
        self._orig.append((cmp.SnapshotStore, "_raw", orig_raw))
        cmp.SnapshotStore._raw = raw

        def timed(cls, name, counter, seconds):
            orig = getattr(cls, name)

            def w(self_, *a, **kw):
                t0 = time.time()
                try:
                    return orig(self_, *a, **kw)
                finally:
                    setattr(inst, counter, getattr(inst, counter) + 1)
                    setattr(inst, seconds, getattr(inst, seconds) + time.time() - t0)
            self._orig.append((cls, name, orig))
            setattr(cls, name, w)

        timed(cmp.Values, "of_run", "of_run_calls", "of_run_seconds")
        timed(rp.Replayer, "expected_words", "expected_calls", "expected_seconds")
        timed(rp.Replayer, "operand_words", "operand_calls", "operand_seconds")
        timed(rp.Replayer, "evaluate", "evaluate_calls", "evaluate_seconds")

    def uninstall(self) -> None:
        for cls, name, orig in reversed(self._orig):
            setattr(cls, name, orig)
        self._orig.clear()

    def summary(self) -> dict:
        att = sorted(self.attention)
        fit = linear_fit_T([(T, s) for T, s, _ in att])
        return {"per_family": {f: {"evaluations": self.family_evals[f], "seconds": round(self.family_seconds[f], 3),
                                   "seconds_per_evaluation": round(self.family_seconds[f] / max(self.family_evals[f], 1), 4),
                                   "status": dict(self.family_status[f])} for f in sorted(self.family_evals)},
                "attention_evals": [{"T": T, "seconds": round(s, 3), "status": st} for T, s, st in att],
                "attention_fit": fit,
                "snapshots": {"files_loaded": self.snapshot_loads, "bytes_loaded": self.snapshot_bytes, "gb_loaded": round(self.snapshot_bytes / 2 ** 30, 3),
                              "seconds": round(self.snapshot_seconds, 3), "cache_hits": self.snapshot_hits},
                "values_of_run": {"calls": self.of_run_calls, "seconds": round(self.of_run_seconds, 3)},
                "expected_words": {"calls": self.expected_calls, "seconds": round(self.expected_seconds, 3)},
                "operand_words": {"calls": self.operand_calls, "seconds": round(self.operand_seconds, 3)},
                "evaluate_call": {"calls": self.evaluate_calls, "seconds": round(self.evaluate_seconds, 3)}}


def linear_fit_T(pairs: list[tuple[int, float]]) -> dict | None:
    """Least-squares `seconds = a*T + b` over (T, seconds) pairs (None below 3 points / 2 distinct T)."""
    if len(pairs) < 3 or len({T for T, _ in pairs}) < 2:
        return None
    import numpy as np
    Ts = np.array([T for T, _ in pairs], dtype=float)
    ss = np.array([s for _, s in pairs], dtype=float)
    a, b = np.polyfit(Ts, ss, 1)
    pred = a * Ts + b
    return {"seconds_per_T": float(a), "intercept_s": float(b),
            "r2": float(1 - ((ss - pred) ** 2).sum() / max(((ss - ss.mean()) ** 2).sum(), 1e-12)), "n": int(len(pairs)),
            "T_range": [int(Ts.min()), int(Ts.max())], "mean_T": float(Ts.mean()), "mean_seconds": float(ss.mean())}


def calibration(inst_summary: dict, legacy: dict) -> dict | None:
    """cProfile inflates Python-call-heavy code several-fold.  When the run's own uninstrumented `replay_a.json` carries per-instance
    seconds, the ratio of the two attention fits (profiled / legacy, at the legacy sample's mean T) is the overhead factor; the
    projection divided by it is the calibrated estimate."""
    fit = inst_summary.get("attention_fit")
    leg = ((legacy.get("replay_a.json") or {}).get("attention_fit")) if legacy else None
    if not fit or not leg:
        return None
    T = leg["mean_T"]
    prof_at_T = fit["seconds_per_T"] * T + fit["intercept_s"]
    leg_at_T = leg["seconds_per_T"] * T + leg["intercept_s"]
    if leg_at_T <= 0 or prof_at_T <= 0:
        return None
    return {"at_T": round(T, 1), "profiled_seconds_per_eval": round(prof_at_T, 4), "legacy_seconds_per_eval": round(leg_at_T, 4),
            "overhead_factor": round(prof_at_T / leg_at_T, 3), "legacy_fit": leg}


def population(result, k: int, snap_steps: list[int]) -> dict:
    """Per family, from the fold alone: specializations, instances, strata and the evaluations a full tier A at `k` needs."""
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for m in result.instances:
        by_spec[m["spec"]].append(m)
    fam: dict[str, dict] = {}
    for spec, metas in by_spec.items():
        f = fam.setdefault(family_of(spec), {"specs": 0, "instances": 0, "strata": 0, "required_evaluations": 0, "T_min": None, "T_max": None,
                                              "specs_without_snapshot_step": 0})
        f["specs"] += 1
        f["instances"] += len(metas)
        cands = [m for m in metas if int(m["step"]) in snap_steps]
        if not cands:
            f["specs_without_snapshot_step"] += 1
        regimes: Counter = Counter("prefill" if int(m["step"]) == 0 else "decode" for m in cands)
        f["strata"] += len(regimes)
        f["required_evaluations"] += sum(min(k, n) for n in regimes.values())
        T = attention_T(spec)
        if T is not None:
            f["T_min"] = T if f["T_min"] is None else min(f["T_min"], T)
            f["T_max"] = T if f["T_max"] is None else max(f["T_max"], T)
    tot = {"specs": sum(f["specs"] for f in fam.values()), "instances": len(result.instances),
           "required_evaluations": sum(f["required_evaluations"] for f in fam.values())}
    return {"k": k, "snapshot_steps": snap_steps, "families": dict(sorted(fam.items())), "total": tot}


def project(pop: dict, inst_summary: dict, k: int, result) -> dict:
    """Projected full tier-A seconds per family from the measured per-evaluation cost (attention: the T fit over every spec's T)."""
    out: dict[str, dict] = {}
    fit = inst_summary.get("attention_fit")
    per_T_specs: dict[str, list[int]] = defaultdict(list)
    by_spec_regimes: dict[str, Counter] = defaultdict(Counter)
    for m in result.instances:
        by_spec_regimes[m["spec"]]["prefill" if int(m["step"]) == 0 else "decode"] += 1
    for spec, regs in by_spec_regimes.items():
        T = attention_T(spec)
        if T is not None and family_of(spec).startswith("Attention"):
            per_T_specs[family_of(spec)].extend([T] * sum(min(k, n) for n in regs.values()))
    for f, p in pop["families"].items():
        meas = inst_summary["per_family"].get(f)
        rec: dict[str, Any] = {"required_evaluations": p["required_evaluations"], "measured_evaluations": meas["evaluations"] if meas else 0}
        if meas and meas["evaluations"]:
            if f in per_T_specs and fit is not None:
                secs = sum(max(fit["seconds_per_T"] * T + fit["intercept_s"], 0.0) for T in per_T_specs[f])
                rec.update(projected_seconds=round(secs, 1), method="linear fit in T over every required (spec, regime) evaluation")
            else:
                rec.update(projected_seconds=round(meas["seconds_per_evaluation"] * p["required_evaluations"], 1), method="mean seconds per evaluation x required")
        else:
            rec.update(projected_seconds=None, method="no measured evaluation of this family in the bounded run")
        out[f] = rec
    total = sum(r["projected_seconds"] or 0.0 for r in out.values())
    return {"families": out, "projected_total_seconds": round(total, 1),
            "projected_attention_share": round(sum(r["projected_seconds"] or 0.0 for f, r in out.items() if f.startswith("Attention")) / total, 4) if total else None}


def inprocess_profile(run: Path, out: Path, *, profile_name: str, log: Path, snaps: Path, safetensors: str | None, k: int, steps_arg: str | None,
                      budget_s: float, seed: int, use_cprofile: bool = True) -> dict:
    """Phase timings / RSS for load -> fold -> store -> weights, then tier A with the instrumentation wrappers, under cProfile unless
    `use_cprofile=False` (cProfile inflates this Python-call-heavy code several-fold: it gives the attribution, not the absolute cost;
    see `calibration`)."""
    rec: dict[str, Any] = {"phases": [], "k": k, "seed": seed, "cprofile": use_cprofile}
    t_all = time.time()

    def phase(name: str, t0: float, **extra) -> None:
        rec["phases"].append({"phase": name, "seconds": round(time.time() - t0, 3), **rss_now(), **extra})

    t0 = time.time()
    from verity_capture import replay as rp
    from verity_capture.compare import SnapshotStore, WeightSource, load_profile
    from verity_capture.fold import fold
    from verity_capture.log import read_log
    phase("imports", t0)
    t0 = time.time()
    profile = load_profile(profile_name)
    phase("load_profile", t0)
    t0 = time.time()
    events = list(read_log(log))
    phase("read_log", t0, n_events=len(events), log_bytes=log.stat().st_size)
    t0 = time.time()
    result = fold(events, profile, strict=None, with_program=True)
    phase("fold_with_program", t0, n_instances=len(result.instances), n_steps=len(result.steps), digest=result.digest,
          fold_timings=_jsonable(result.timings))
    del events
    t0 = time.time()
    store = SnapshotStore(snaps)
    phase("snapshot_store_index", t0, n_entries=len(store.entries), n_skipped=len(store.skipped), n_steps=len(store.steps),
          index_bytes=(snaps / "snapshots.json").stat().st_size)
    t0 = time.time()
    repo = (profile.config or {}).get("model")
    cos_sin = snaps / "cos_sin_cache.npy" if (snaps / "cos_sin_cache.npy").exists() else None
    weights = WeightSource(result.params[-1][1], [Path(p) for p in sorted(glob.glob(safetensors))] if safetensors else None, cos_sin, repo=repo,
                           config=dict(profile.config or {}))
    phase("weight_source", t0, available=weights.available(), n_tensors=len(weights.tensors), paths=[str(p) for p in weights.paths])

    all_steps = sorted(store.steps)
    steps = parse_steps(steps_arg, all_steps)
    snap_steps = all_steps if steps is None else steps
    rec["steps"] = {"snapshot_steps": all_steps, "profiled_steps": snap_steps}
    rec["population"] = population(result, k, all_steps)
    rec["population_profiled_steps"] = population(result, k, snap_steps) if steps is not None else rec["population"]

    inst = Instrument()
    inst.install()
    pr = cProfile.Profile() if use_cprofile else None
    old = signal.getsignal(signal.SIGALRM)

    def _alarm(signum, frame):
        raise _Budget(f"in-process tier A budget of {budget_s:.0f} s exhausted")
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, budget_s))
    t0 = time.time()
    report = None
    aborted = None
    if pr is not None:
        pr.enable()
    try:
        report = rp.tier_a(result, store, weights, k=k, seed=seed, steps=steps)
    except _Budget as b:
        aborted = str(b)
    finally:
        if pr is not None:
            pr.disable()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        inst.uninstall()
    tier_s = time.time() - t0
    phase("tier_a", t0, aborted=aborted)
    cache_bytes = sum(int(s._words.nbytes) for s in store.entries if s._words is not None)
    rec["tier_a"] = {"seconds": round(tier_s, 3), "aborted": aborted,
                     "report": None if report is None else {kk: report.get(kk) for kk in ("status", "status_reason", "n_specs", "n_specs_covered", "strata",
                                                                                         "summary", "seconds", "max_rss_gb", "snapshot_steps")},
                     "vu_population_seconds": (report or {}).get("vu_population", {}).get("seconds") if report else None,
                     "tokens": {kk: (report or {}).get("tokens", {}).get(kk) for kk in ("evaluated", "equal", "skipped")} if report else None,
                     "uncovered_specs": len((report or {}).get("uncovered_specs", [])) if report else None,
                     "snapshot_cache_after": {"files_resident": sum(1 for s in store.entries if s._words is not None), "gb": round(cache_bytes / 2 ** 30, 3)},
                     "replayer_caches": None}
    rec["instrumentation"] = inst.summary()
    rec["projection_full_tier_a"] = project(rec["population"], rec["instrumentation"], k, result)
    if report is not None:
        rec["tier_a"]["per_family_from_report"] = per_family_from_report(report)
        (out / "replay_a_inprocess.json").write_text(json.dumps(_jsonable(report), indent=1))
    # cProfile: top by cumulative and by internal time
    hot: dict[str, str] = {}
    if pr is not None:
        pr.dump_stats(str(out / "tier_a.prof"))
        for key in ("cumulative", "tottime"):
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats(key).print_stats(45)
            hot[key] = "\n".join(line for line in s.getvalue().splitlines() if line.strip())[:12000]
    rec["cprofile_top"] = hot
    rec["seconds_total"] = round(time.time() - t_all, 3)
    rec["rss_end"] = rss_now()
    return rec


# ---------------------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------------------


def legacy_numbers(run: Path) -> dict:
    """What the run directory already records about its own CPU phase (stages.json + the stage reports): the side-by-side baseline."""
    out: dict[str, Any] = {}
    p = run / "stages.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            out["stages"] = [{kk: s.get(kk) for kk in ("stage", "status", "seconds", "peak_rss_gb", "reason")} for s in d.get("stages", d)]
        except (OSError, ValueError) as e:
            out["stages_error"] = f"{type(e).__name__}: {e}"
    for name in ("fold_summary.json", "compare.json", "wiring.json", "replay_chain.json", "replay_a.json"):
        q = run / name
        if not q.exists():
            out[name] = None
            continue
        try:
            d = json.loads(q.read_text())
            rec = {kk: d.get(kk) for kk in ("status", "fold_seconds", "seconds", "total_seconds", "max_rss_gb", "k", "n_specs", "n_specs_covered", "strata",
                                            "summary", "n_slices", "n_instances", "n_steps", "timings")}
            if name == "replay_a.json":
                rec["per_family"] = per_family_from_report(d)
                pairs = [(attention_T(e.get("spec", "")), float(e.get("seconds") or 0.0)) for e in d.get("instances", [])
                         if family_of(e.get("spec", "")).startswith("Attention") and e.get("status") in ("pass", "fail") and e.get("seconds") is not None]
                rec["attention_fit"] = linear_fit_T([(T, s) for T, s in pairs if T is not None])   # uninstrumented per-eval cost vs T
                rec["tokens"] = {kk: (d.get("tokens") or {}).get(kk) for kk in ("evaluated", "equal", "skipped")}
                rec["vu_population"] = {kk: (d.get("vu_population") or {}).get(kk) for kk in ("count", "instances", "seconds")}
            out[name] = rec
        except (OSError, ValueError) as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    try:
        man = json.loads((run / "run.json").read_text())
        out["run"] = {kk: man.get(kk) for kk in ("config_id", "profile", "parallel_replay", "sampling", "snapshot_steps")}
        real = man.get("realized") or {}
        out["realized"] = {kk: real.get(kk) for kk in ("prompt_lens", "n_generated", "engine_steps_capture", "max_concurrent_requests")}
    except (OSError, ValueError):
        pass
    return out


def markdown(doc: dict) -> str:
    L: list[str] = [f"# profile_check — `{doc['run']}`", "",
                    f"utc {doc['utc']} · host `{doc.get('host')}` · cgroup limit {doc.get('cgroup_mem_max_gb')} GB · python `{doc['python']}` · "
                    f"profile `{doc['profile']}` · budget {doc['args']['total_budget_s']} s / {doc['args']['max_rss_gb']} GB", ""]
    L += ["## 1. Stage subprocesses (each alone; `/usr/bin/time -v` + process-tree RSS watchdog)", "",
          "| stage | status | wall s | peak RSS GB (time -v) | peak tree RSS GB (poll) | user s | sys s | inner: fold s | inner: check s | inner max_rss GB | notes |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in doc.get("stages", []):
        tv, rp_ = s.get("time_v") or {}, s.get("report") or {}
        notes = s.get("reason") or ""
        if rp_.get("status"):
            notes = (notes + " " if notes else "") + f"report {rp_['status']}"
        for kk in ("n_specs", "strata", "n_slices", "n_edges_chained", "n_instances"):
            if rp_.get(kk) is not None:
                notes += f" {kk}={rp_[kk]}"
        L.append(f"| {s['stage']} | {s['status']} | {s.get('wall_s')} | {tv.get('max_rss_gb', '')} | {s.get('peak_tree_rss_gb')} | {tv.get('user_s', '')} | "
                 f"{tv.get('sys_s', '')} | {rp_.get('fold_seconds', '')} | {rp_.get('seconds', '')} | {rp_.get('max_rss_gb', '')} | {notes.strip()} |")
    if not doc.get("stages"):
        L.append("| (none run) | | | | | | | | | | |")
    ip = doc.get("inprocess")
    if ip:
        L += ["", "## 2. In-process: load → fold → store → weights → tier A", "",
              "| phase | seconds | RSS after GB | high-water GB | detail |", "|---|---|---|---|---|"]
        for p in ip["phases"]:
            det = {kk: v for kk, v in p.items() if kk not in ("phase", "seconds", "rss_gb", "hwm_gb", "fold_timings", "paths")}
            L.append(f"| {p['phase']} | {p['seconds']} | {p['rss_gb']} | {p['hwm_gb']} | {json.dumps(det)[:160]} |")
        ta = ip["tier_a"]
        L += ["", f"tier A (k={ip['k']}, steps {ip['steps']['profiled_steps'][:8]}{'…' if len(ip['steps']['profiled_steps']) > 8 else ''} of "
              f"{len(ip['steps']['snapshot_steps'])} snapshot steps): {ta['seconds']} s" + (f", **aborted**: {ta['aborted']}" if ta["aborted"] else "")
              + (f"; report {ta['report']}" if ta.get("report") else ""),
              f"snapshot cache after tier A: {ta['snapshot_cache_after']['files_resident']} files resident, {ta['snapshot_cache_after']['gb']} GB "
              f"(`SnapshotStore._raw` keeps every loaded file for the life of the process)", ""]
        ins = ip["instrumentation"]
        L += ["### Per family (measured in the bounded run)", "",
              "| family | specs (whole run) | instances | strata | required evals (full tier A, k) | measured evals | measured s | s / eval | projected full tier-A s |",
              "|---|---|---|---|---|---|---|---|---|"]
        pop, proj = ip["population"]["families"], ip["projection_full_tier_a"]["families"]
        for f in sorted(pop, key=lambda x: -(proj[x].get("projected_seconds") or 0)):
            m = ins["per_family"].get(f, {})
            L.append(f"| {f} | {pop[f]['specs']} | {pop[f]['instances']} | {pop[f]['strata']} | {pop[f]['required_evaluations']} | {m.get('evaluations', 0)} | "
                     f"{m.get('seconds', 0)} | {m.get('seconds_per_evaluation', '')} | {proj[f].get('projected_seconds')} |")
        pj = ip["projection_full_tier_a"]
        under = "under cProfile (attribution, not absolute cost)" if ip.get("cprofile", True) else "wrappers only (no cProfile)"
        L += ["", f"Projected full tier A at k={ip['k']}, {under}: **{pj['projected_total_seconds']} s** ({(pj['projected_total_seconds'] or 0) / 3600:.2f} h), "
              f"attention share {pj['projected_attention_share']}.  Attention fit: {json.dumps(ins.get('attention_fit'))}"]
        cal = ip.get("calibration")
        if cal:
            tot = pj["projected_total_seconds"] or 0
            L.append(f"Calibration against the run's own uninstrumented `replay_a.json`: at T ≈ {cal['at_T']} profiled {cal['profiled_seconds_per_eval']} s/eval vs "
                     f"legacy {cal['legacy_seconds_per_eval']} s/eval → **cProfile overhead ×{cal['overhead_factor']}**; calibrated projection ≈ "
                     f"**{tot / cal['overhead_factor']:.0f} s**.  Legacy attention fit: {json.dumps(cal['legacy_fit'])}")
        L.append("")
        L += ["### Where the tier-A seconds go (instrumented wrappers, inclusive)", "",
              "| what | calls | seconds |", "|---|---|---|"]
        for kk in ("evaluate_call", "operand_words", "expected_words", "values_of_run"):
            L.append(f"| {kk} | {ins[kk]['calls']} | {ins[kk]['seconds']} |")
        sn = ins["snapshots"]
        L.append(f"| snapshot files read (`SnapshotStore._raw` misses) | {sn['files_loaded']} ({sn['gb_loaded']} GB; {sn['cache_hits']} cache hits) | {sn['seconds']} |")
        if ip.get("cprofile_top"):
            L += ["", "### cProfile — top by cumulative time", "", "~~~", ip["cprofile_top"]["cumulative"][:6000], "~~~", "",
                  "### cProfile — top by internal time", "", "~~~", ip["cprofile_top"]["tottime"][:6000], "~~~"]
    lg = doc.get("legacy") or {}
    if lg.get("stages"):
        L += ["", "## 3. Legacy (the run directory's own CPU phase records)", "",
              f"run.json: {json.dumps(lg.get('run'))}; realized: {json.dumps(lg.get('realized'))}", "",
              "| stage | status | seconds | reason |", "|---|---|---|---|"]
        for s in lg["stages"]:
            L.append(f"| {s['stage']} | {s['status']} | {s.get('seconds')} | {(s.get('reason') or '')[:70]} |")
        for name in ("fold_summary.json", "compare.json", "wiring.json", "replay_chain.json", "replay_a.json"):
            r = lg.get(name)
            if r:
                L.append(f"- `{name}`: " + json.dumps({kk: v for kk, v in r.items() if kk not in ("per_family",) and v is not None})[:400])
                if r.get("per_family"):
                    L.append("  per family: " + json.dumps(r["per_family"])[:800])
    if doc.get("errors"):
        L += ["", "## errors", ""] + [f"- {e}" for e in doc["errors"]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------


def time_wrap(out_file: str, cmd: list[str]) -> int:
    """`/usr/bin/time -v` stand-in: run `cmd`, write getrusage(RUSAGE_CHILDREN) as JSON (peak RSS, user/sys, wall, exit status)."""
    import resource
    t0 = time.time()
    proc = subprocess.run(cmd)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    maxrss_gb = ru.ru_maxrss / 2 ** 30 if sys.platform == "darwin" else ru.ru_maxrss / 2 ** 20
    rec = {"max_rss_gb": round(maxrss_gb, 3), "user_s": round(ru.ru_utime, 3), "sys_s": round(ru.ru_stime, 3), "wall_s": round(time.time() - t0, 3),
           "major_faults": ru.ru_majflt, "minor_faults": ru.ru_minflt, "fs_inputs": ru.ru_inblock, "fs_outputs": ru.ru_oublock,
           "exit_status": proc.returncode, "source": "getrusage(RUSAGE_CHILDREN)"}
    if proc.returncode < 0:
        rec["signal"] = f"Command terminated by signal {-proc.returncode}"
    Path(out_file).write_text(json.dumps(rec))
    return proc.returncode if proc.returncode >= 0 else 128 - proc.returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--time-wrap"]:
        sep = argv.index("--")
        return time_wrap(argv[1], argv[sep + 1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (run.json, capture/log.jsonl.gz, capture/snapshots.json)")
    ap.add_argument("--out", required=True, help="output directory (created; nothing is written into --run)")
    ap.add_argument("--profile", default=None, help="profile module (default: run.json['profile'])")
    ap.add_argument("--checkpoint-dir", default=None, help="HF snapshot dir / model*.safetensors glob (default: run.json['checkpoint_dir'])")
    ap.add_argument("--python", default=None, help="interpreter for the stage subprocesses (default: this one)")
    ap.add_argument("--stages", default="fold,compare,wiring,replay_chain", help=f"comma list from {STAGES} ('' = none)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--steps", default="0-3", help="in-process tier A: snapshot steps to sample from (e.g. 0-3, 0,1,5, all)")
    ap.add_argument("--stage-timeout", type=float, default=900.0)
    ap.add_argument("--max-rss-gb", type=float, default=38.0)
    ap.add_argument("--total-budget-s", type=float, default=1500.0, help="stop launching stages beyond this (in-process part keeps its own budget)")
    ap.add_argument("--inprocess-budget-s", type=float, default=600.0)
    ap.add_argument("--no-inprocess", action="store_true")
    ap.add_argument("--no-cprofile", action="store_true", help="in-process tier A with the wrappers only (uninflated per-evaluation seconds, no hot-spot table)")
    a = ap.parse_args(argv)

    t_start = time.time()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": "verity-perf/profile_check/v1", "run": str(run), "out": str(out), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "host": os.uname().nodename, "python": sys.executable, "args": vars(a), "errors": [], "stages": []}
    try:
        doc["cgroup_mem_max_gb"] = round(int(Path("/sys/fs/cgroup/memory.max").read_text().strip()) / 1e9, 1)
    except (OSError, ValueError):
        doc["cgroup_mem_max_gb"] = None
    man = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    profile_name = a.profile or man.get("profile")
    if not profile_name:
        raise SystemExit("no profile: give --profile or a run.json with 'profile'")
    doc["profile"] = profile_name
    cap = run / "capture"
    arts = man.get("artifacts") or {}
    log = Path(arts.get("log") or (cap / "log.jsonl.gz"))
    snaps = Path(arts["snapshots"]).parent if arts.get("snapshots") else cap
    doc["inputs"] = {"log": str(log), "snapshots": str(snaps), "log_bytes": log.stat().st_size if log.exists() else None,
                     "snapshots_json_bytes": (snaps / "snapshots.json").stat().st_size if (snaps / "snapshots.json").exists() else None,
                     "values_present": (snaps / "values").is_dir(), "pruned_json": (run / "pruned.json").exists()}
    if not log.exists():
        raise SystemExit(f"{log} missing")
    if not (snaps / "snapshots.json").exists() or (run / "pruned.json").exists():
        doc["status"] = "pruned"
        doc["errors"].append(f"raw evidence absent ({snaps}/snapshots.json missing or pruned.json present): nothing to profile against")
        (out / "profile.json").write_text(json.dumps(_jsonable(doc), indent=1))
        (out / "PROFILE.md").write_text(markdown(doc))
        print(f"[profile_check] PRUNED: {doc['errors'][-1]}", file=sys.stderr)
        return 3
    doc["legacy"] = legacy_numbers(run)
    ck = a.checkpoint_dir or man.get("checkpoint_dir")
    safetensors = None
    if ck:
        safetensors = ck if any(ch in ck for ch in "*?[") else str(Path(ck) / "model*.safetensors")
    doc["inputs"]["safetensors"] = safetensors
    py = a.python or sys.executable
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("PYTHONPATH", ".:vllm-poc")
    if Path("/workspace/hf").is_dir():
        env.setdefault("HF_HOME", "/workspace/hf")
        env.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", "/workspace/hf")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for st in stages:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st}; choose from {STAGES}")
        elapsed = time.time() - t_start
        reserve = 0 if a.no_inprocess else a.inprocess_budget_s * 0.5
        if elapsed + reserve > a.total_budget_s:
            doc["stages"].append({"stage": st, "status": "not_run", "reason": f"total budget {a.total_budget_s} s reached ({elapsed:.0f} s elapsed)"})
            continue
        remaining = max(30.0, min(a.stage_timeout, a.total_budget_s - elapsed - reserve))
        cmd = stage_cmd(st, py=py, log=log, snaps=snaps, profile=profile_name, out=out, k=a.k, seed=a.seed, safetensors=safetensors, chains=a.chains)
        print(f"[profile_check] stage {st}: {shlex.join(cmd)}  (timeout {remaining:.0f} s, rss cap {a.max_rss_gb} GB)", file=sys.stderr, flush=True)
        rec = run_stage(st, cmd, out, env=env, max_rss_gb=a.max_rss_gb, timeout_s=remaining)
        print(f"[profile_check] stage {st}: {rec['status']} wall={rec['wall_s']} s peak_tree={rec['peak_tree_rss_gb']} GB "
              f"time-v={ (rec.get('time_v') or {}).get('max_rss_gb') } GB", file=sys.stderr, flush=True)
        doc["stages"].append(rec)
        (out / "profile.json").write_text(json.dumps(_jsonable(doc), indent=1))
    if not a.no_inprocess:
        try:
            doc["inprocess"] = inprocess_profile(run, out, profile_name=profile_name, log=log, snaps=snaps, safetensors=safetensors, k=a.k,
                                                 steps_arg=a.steps, budget_s=a.inprocess_budget_s, seed=a.seed, use_cprofile=not a.no_cprofile)
            doc["inprocess"]["calibration"] = calibration(doc["inprocess"].get("instrumentation") or {}, doc.get("legacy") or {}) if not a.no_cprofile else None
        except Exception as e:  # noqa: BLE001 - the profile is a measurement; report what broke
            import traceback
            doc["errors"].append(f"inprocess: {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
    doc["seconds_total"] = round(time.time() - t_start, 3)
    doc["rss_end"] = rss_now()
    doc.setdefault("status", "ok" if not doc["errors"] else "partial")
    (out / "profile.json").write_text(json.dumps(_jsonable(doc), indent=1))
    (out / "PROFILE.md").write_text(markdown(doc))
    print(markdown(doc))
    print(str(out / "profile.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
