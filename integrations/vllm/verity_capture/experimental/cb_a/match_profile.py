"""match_profile -- phase profile of the offline Match pipeline (fold -> X-09 batch_decomp -> GM-01 global_match) on a PRESERVED row.

[R17 mprof, BRIEF §5 step 1]  Runs the three offline stages exactly as `verity_capture/sweep/row_pod.sh` runs them (same CLIs, same
arguments, same artifacts), each in its own process, and records per stage: wall, user+sys CPU, peak RSS (ru_maxrss), the represented
object counts (instances, nodes, rows, steps, components) and -- without touching the checker of record -- the time inside the
stage's phases:

  * cheap timers around the stage's phase functions (load_oracle / decode_program / load_derived / read_record / read_steps /
    decompose / compare_steps / _canonical_hash / dag_hashes / live_cone / Projection / sampler_geometry_check / json.dump), installed
    by wrapping the module attributes in the child process (transparent: same return values, exceptions propagate);
  * optionally a `py-spy record --format raw` sample of the child (low overhead, no code change) aggregated into the BRIEF §5 phase
    buckets by frame name + line range (loading/decompression, operation resolution (decode), attribution, component construction
    and comparison, global wiring/interface/chronology/population checks, serialization), so the checker's own loop bodies -- which no
    function boundary isolates -- are attributed too.

The verdicts are untouched: the stages run the code of record; this module only measures.  Nothing here is read by the finalizer.

    python -m verity_capture.experimental.cb_a.match_profile --row-dir <row attempt dir> --out <dir> [--stages fold,x09,gm01]
        [--pyspy] [--within-step dag] [--no-alternate]
    python -m verity_capture.experimental.cb_a.match_profile --report <dir>       # (re)render <dir>/MATCH-PROFILE.md from the json

Row dir layout (row_pod.sh): match/{program.json,instances.jsonl,fold_summary.json,capture/{log.jsonl.gz,tokens.json}},
build_workload/workload_program.json, build_request*/ (derive_step components), manifest.json (the workload manifest, `WL`).
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

_TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))   # the shipped tree (row_pod's WL lives under workloads)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = sys.executable


# ------------------------------------------------------------------------------------------------------ child: timers


class _Timers:
    """Cumulative wall + call counts per label; `wrap(module, name, label)` replaces `module.name` with a timing wrapper."""

    def __init__(self) -> None:
        self.t: dict[str, float] = collections.defaultdict(float)
        self.n: dict[str, int] = collections.defaultdict(int)
        self.depth: dict[str, int] = collections.defaultdict(int)
        self.marks: list[tuple[str, float, float]] = []   # (label, t_start, t_end) for the top-level calls
        self.objects: dict[str, object] = {}
        self.t0 = time.time()

    def wrap(self, mod, name: str, label: str | None = None, on_return=None) -> None:
        label = label or f"{mod.__name__.rsplit('.', 1)[-1]}.{name}"
        fn = getattr(mod, name)

        @functools.wraps(fn)
        def w(*a, **k):
            self.depth[label] += 1
            t = time.time()
            try:
                r = fn(*a, **k)
            finally:
                dt = time.time() - t
                self.depth[label] -= 1
                if self.depth[label] == 0:   # count outer calls only (dag_hashes etc. do not nest; check() does: the alternate run)
                    self.t[label] += dt
                    self.n[label] += 1
                    self.marks.append((label, t - self.t0, t - self.t0 + dt))
            if on_return is not None and self.depth[label] == 0:
                try:
                    on_return(r, *a, **k)
                except Exception:  # noqa: BLE001  measurement only
                    pass
            return r
        setattr(mod, name, w)

    def wrap_method(self, cls, name: str, label: str, on_return=None) -> None:
        self.wrap(cls, name, label, on_return)

    def dump(self) -> dict:
        return {"timers_s": {k: round(v, 3) for k, v in sorted(self.t.items(), key=lambda kv: -kv[1])},
                "calls": dict(self.n), "marks": [(l, round(a, 3), round(b, 3)) for l, a, b in self.marks[:4000]], "objects": self.objects}


class _Sampler:
    """In-process statistical sampler (SIGPROF / setitimer, ITIMER_PROF = CPU time of the process): every tick records the main
    thread's stack as a py-spy-style collapsed line `fn (file:line);...;leaf count` (root first).  Needs no ptrace (py-spy attach is
    refused in the pods' containers); overhead ~1-2 % at 50 Hz.  Measurement only -- the checker's code is not touched."""

    def __init__(self, rate: int) -> None:
        self.stacks: collections.Counter = collections.Counter()
        self.rate = max(1, rate)
        self.n = 0

    def _tick(self, signum, frame) -> None:
        parts = []
        f = frame
        depth = 0
        while f is not None and depth < 60:
            co = f.f_code
            parts.append(f"{co.co_name} ({co.co_filename}:{f.f_lineno})")
            f = f.f_back
            depth += 1
        self.stacks[";".join(reversed(parts))] += 1
        self.n += 1

    def start(self) -> None:
        import signal
        signal.signal(signal.SIGPROF, self._tick)
        signal.setitimer(signal.ITIMER_PROF, 1.0 / self.rate, 1.0 / self.rate)

    def stop(self, path: str) -> None:
        import signal
        signal.setitimer(signal.ITIMER_PROF, 0, 0)
        with open(path, "w") as f:
            for k, v in self.stacks.most_common():
                f.write(f"{k} {v}\n")


def _rusage() -> dict:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    maxrss_gb = ru.ru_maxrss / (1024 ** 3) if sys.platform == "darwin" else ru.ru_maxrss / (1024 ** 2)
    return {"user_s": round(ru.ru_utime, 2), "sys_s": round(ru.ru_stime, 2), "max_rss_gb": round(maxrss_gb, 3)}


def _prog_counts(P) -> dict:
    inst = getattr(P, "inst", None) or []
    runs = 0
    node_runs = 0
    for r in inst:
        for a in r["args"]:
            runs += len(a)
            for t in a:
                if t[0] == "n":
                    node_runs += 1
    return {"instances": len(inst), "root_nodes": getattr(P, "n_nodes", None), "inputs": len(getattr(P, "inputs", {}) or {}),
            "steps": len(getattr(P, "steps", []) or []), "operand_runs": runs, "node_space_runs": node_runs,
            "oracle_rows": len(getattr(P, "oracle_rows", None) or []) if getattr(P, "oracle_rows", None) is not None else None,
            "weight_fields": len(getattr(P, "oracle_wmap", []) or []), "digest": getattr(P, "digest", None)}


def _install_timers(T: _Timers, stage: str) -> None:
    import gzip
    import json as _json

    from verity.ir import codec
    from . import batch_decomp as BD
    from . import compare as C
    from . import global_match as GM

    # loading / decompression / operation resolution (decode)
    T.wrap(codec, "decode_program", "load.decode_program")
    comps: dict = T.objects.setdefault("components", {})

    def _on_oracle(P, d, *a, **k):   # the FIRST load_oracle is the fold; later ones are fold records standing in for components (fixtures)
        if "fold" not in T.objects:
            T.objects["fold"] = _prog_counts(P)
        else:
            comps[os.path.basename(str(d))] = _prog_counts(P)
    T.wrap(C, "load_oracle", "load.load_oracle(fold: program.json+instances.jsonl)", on_return=_on_oracle)
    _on_comp = lambda P, d, *a, **k: comps.__setitem__(os.path.basename(str(d)), _prog_counts(P))  # noqa: E731
    T.wrap(C, "load_derived", "load.load_derived(component)", on_return=_on_comp)
    if hasattr(GM, "_LOAD_DERIVED"):   # global_match._load_component calls the ORIGINAL load_derived through this module binding
        T.wrap(GM, "_LOAD_DERIVED", "load.load_derived(component)", on_return=_on_comp)
    T.wrap(GM, "read_record", "record.read_record(capture log)",
           on_return=lambda rec, *a, **k: T.objects.__setitem__("record", {"engine_steps": len(rec["steps"]), "samples": len(rec["samples"]),
                                                                          "links": len(rec["links"]),
                                                                          "rows": sum(len(s["rows"]) for s in rec["steps"].values())}))
    T.wrap(BD, "read_steps", "record.read_steps(capture log, X-09)")
    T.wrap(GM, "from_gprog", "load.from_gprog(GP-01 artifact)")
    T.wrap(GM, "canonicalize_record", "attribution.canonicalize_record")
    # component structure construction + comparison
    T.wrap(C, "compare_steps", "compare.compare_steps(A,P)")
    T.wrap(C, "_canonical_hash", "compare._canonical_hash")
    T.wrap(BD, "dag_hashes", "compare.dag_hashes")
    T.wrap(BD, "live_cone", "compare.live_cone")
    T.wrap(BD.Projection, "__init__", "compare.Projection(request view)")
    T.wrap(C.Prog, "_segment", "compare.Prog._segment")
    T.wrap(C.Prog, "_bind_splits", "compare.Prog._bind_splits")
    # X-09 inside GM-01 / the whole decompose
    T.wrap(BD, "decompose", "x09.decompose")
    # global checks with a function boundary
    T.wrap(GM, "sampler_geometry_check", "global.sampler_geometry_check(G6)")
    T.wrap(GM, "record_engine_facts", "global.record_engine_facts(G1)")
    T.wrap(GM, "check", "gm01.check (outer; alternate criterion nested inside)")
    # serialization
    T.wrap(_json, "dump", "serialize.json.dump")
    _ = gzip
    if stage == "fold":
        from verity_capture import fold as F
        from verity_capture import log as LOG
        from verity_capture.bench import run_config as RC
        T.wrap(LOG, "read_log", "fold.read_log(capture log -> events)")
        T.wrap(F, "fold", "fold.fold(events -> Program + instances)")
        T.wrap(F._Fold, "_step", "fold._Fold._step (view/write/version folding per engine step)")
        T.wrap(F._Fold, "_finish", "fold._Fold._finish (Program build + digest)")
        T.wrap(RC, "persist_program", "fold.persist_program (program.json + instances.jsonl + accesses)")
        T.wrap(RC, "fold_summary_result", "fold.fold_summary_result (read_log + fold + summary)")


def _child(a: argparse.Namespace) -> int:
    """Run ONE stage in this process (the code of record, same CLI args as row_pod.sh) with timers, then write <out>/<stage>.rusage.json."""
    T = _Timers()
    stage = a.child
    _install_timers(T, stage)
    from . import batch_decomp as BD
    from . import global_match as GM
    argv = json.loads(a.child_argv)
    sampler = None
    if a.sample:
        sampler = _Sampler(a.rate)
        sampler.start()
    t0 = time.time()
    rc = None
    err = None
    try:
        if stage == "x09":
            rc = BD.main(argv)
        elif stage == "gm01":
            rc = GM.main(argv)
        elif stage == "fold":
            from verity_capture.bench import run_config as RC
            fa = argparse.Namespace(log=argv[0], profile=argv[1], out=argv[2], persist_dir=argv[3] if len(argv) > 3 else None, no_program=False)
            rc = RC.fold_worker(fa)
        else:
            raise SystemExit(f"unknown child stage {stage}")
    except SystemExit as e:  # the CLIs raise SystemExit on argparse errors
        rc = e.code
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:400]
        rc = 99
    wall = time.time() - t0
    doc = {"stage": stage, "argv": argv, "rc": rc, "error": err, "wall_s": round(wall, 2), **_rusage(), **T.dump()}
    if sampler is not None:
        sampler.stop(os.path.join(a.out, f"{stage}.sample.raw.txt"))
        doc["samples"] = sampler.n
    json.dump(doc, open(os.path.join(a.out, f"{stage}.rusage.json"), "w"), indent=1, default=str)
    return int(rc or 0)


# ------------------------------------------------------------------------------------------------------ parent: stages


def _pyspy() -> str | None:
    for c in ("py-spy", os.path.expanduser("~/.local/bin/py-spy"), os.path.expanduser("~/.cargo/bin/py-spy")):
        if shutil.which(c) or os.path.exists(c):
            return shutil.which(c) or c
    return None


def _run_stage(name: str, cmd: list[str], out: str, pyspy: bool, rate: int, env: dict | None = None, log_name: str | None = None) -> dict:
    """Run `cmd` detached from this shell's stdin, optionally sampled by py-spy (attach by pid, non-blocking), return timing + rusage."""
    log = open(os.path.join(out, log_name or f"{name}.log"), "w")
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=REPO, env=env or os.environ.copy())
    spy = None
    spy_out = None
    if pyspy:
        exe = _pyspy()
        if exe:
            spy_out = os.path.join(out, f"{name}.pyspy.raw.txt")
            spy = subprocess.Popen([exe, "record", "--pid", str(p.pid), "--format", "raw", "--rate", str(rate), "--nonblocking",
                                    "-o", spy_out], stdout=open(os.path.join(out, f"{name}.pyspy.log"), "w"), stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
    rc = p.wait()
    wall = time.time() - t0
    if spy is not None:
        try:
            spy.wait(timeout=120)
        except subprocess.TimeoutExpired:
            spy.kill()
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    log.close()
    maxrss = ru1.ru_maxrss / (1024 ** 3) if sys.platform == "darwin" else ru1.ru_maxrss / (1024 ** 2)
    return {"stage": name, "cmd": cmd, "rc": rc, "wall_s": round(wall, 2), "children_user_s": round(ru1.ru_utime - ru0.ru_utime, 2),
            "children_sys_s": round(ru1.ru_stime - ru0.ru_stime, 2), "children_max_rss_gb_cumulative": round(maxrss, 3),
            "pyspy_raw": spy_out if spy_out and os.path.exists(spy_out) else None, "log": log.name,
            "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _row(a: argparse.Namespace) -> dict:
    d = a.row_dir
    match = a.match_dir or os.path.join(d, "match")
    # the workload spec `WL` row_pod passes to global_program/global_match: <row>/workload.json (match_diff real fixtures),
    # else the tree's workloads/<row name>.json (the row dir is <row name>/<attempt>), else <row>/manifest.json
    wl = a.workload
    if not wl:
        cands = [os.path.join(d, "workload.json")]
        for nm in (os.path.basename(os.path.dirname(os.path.abspath(d))), os.path.basename(os.path.abspath(d))):
            cands.append(os.path.join(_TREE, "out", "gen", "sweep", "workloads", nm + ".json"))
        cands.append(os.path.join(d, "manifest.json"))
        wl = next((c for c in cands if os.path.exists(c)), cands[-1])
    gprog = a.gprog or os.path.join(d, "build_workload", "workload_program.json")
    derived = [os.path.join(d, "build_request")] if os.path.isdir(os.path.join(d, "build_request")) else []
    derived += sorted(p for p in (os.path.join(d, x) for x in os.listdir(d)) if os.path.isdir(p) and os.path.basename(p).startswith("build_request_LP"))
    # a preserved component without its derive-report.json cannot be an X-09 --derived (the standalone stage would fail by name);
    # GM-01's in-process X-09 resolves its components from the GP-01 artifact itself
    derived = [p for p in derived if os.path.exists(os.path.join(p, "derive-report.json"))]
    if a.derived:
        derived = list(a.derived)
    fs = os.path.join(match, "fold_summary.json")
    prof_name = a.fold_profile
    fold_summary = None
    if os.path.exists(fs):
        fold_summary = json.load(open(fs))
        prof_name = prof_name or fold_summary.get("profile")
    return {"row_dir": d, "match": match, "workload": wl, "gprog": gprog if os.path.exists(gprog) else None, "global_program": a.global_program,
            "derived": derived, "log": os.path.join(match, "capture", "log.jsonl.gz"), "fold_profile": prof_name, "fold_summary": fold_summary}


def _sizes(row: dict) -> dict:
    def sz(p):
        try:
            return os.path.getsize(p)
        except OSError:
            return None
    out = {"program.json": sz(os.path.join(row["match"], "program.json")), "instances.jsonl": sz(os.path.join(row["match"], "instances.jsonl")),
           "capture/log.jsonl.gz": sz(row["log"]), "workload_program.json": sz(row["gprog"]) if row["gprog"] else None,
           "manifest.json": sz(row["workload"])}
    comp = {}
    for dd in row["derived"]:
        comp[os.path.basename(dd)] = {"descriptor.json.gz": sz(os.path.join(dd, "descriptor.json.gz"))}
    out["components"] = comp
    return out


def run(a: argparse.Namespace) -> int:
    os.makedirs(a.out, exist_ok=True)
    row = _row(a)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    doc = {"schema": "cb-a/match-profile/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
           "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "row": {k: v for k, v in row.items() if k != "fold_summary"},
           "artifact_bytes": _sizes(row), "pyspy": bool(a.pyspy) and _pyspy() is not None, "stages": {}}
    fsum = row["fold_summary"]
    if fsum:
        doc["fold_summary_of_record"] = {"digest": fsum.get("digest"), "n_instances": fsum.get("n_instances"), "n_steps": fsum.get("n_steps"),
                                         "seconds": fsum.get("seconds"), "max_rss_gb": fsum.get("max_rss_gb"), "timings": fsum.get("timings"),
                                         "memory": fsum.get("memory"), "profile": fsum.get("profile"),
                                         "per_step_seconds_sum": round(sum(float(s.get("seconds") or 0) for s in fsum.get("per_step") or []), 2)}
    json.dump(doc, open(os.path.join(a.out, "profile.json"), "w"), indent=1, default=str)

    if "fold" in stages:
        if not row["fold_profile"]:
            doc["stages"]["fold"] = {"skipped": "no fold profile name (match/fold_summary.json absent; pass --fold-profile)"}
        else:
            refold = os.path.join(a.out, "refold")
            os.makedirs(refold, exist_ok=True)
            # the same worker row_pod's run_config runs (`--worker fold --persist-dir`), in the sampled child so the fold's phases are bucketed too
            fargv = [row["log"], row["fold_profile"], os.path.join(refold, "fold_summary.json"), refold]
            cmd = [PY, "-m", "verity_capture.experimental.cb_a.match_profile", "--out", a.out, "--rate", str(a.rate)] + (["--sample"] if a.sample else []) + \
                  ["--child", "fold", "--child-argv", json.dumps(fargv)]
            st = _run_stage("fold", cmd, a.out, a.pyspy, a.rate)
            st["child"] = _load(os.path.join(a.out, "fold.rusage.json"))
            rf = os.path.join(refold, "fold_summary.json")
            if os.path.exists(rf):
                r = json.load(open(rf))
                st["refold"] = {"digest": r.get("digest"), "n_instances": r.get("n_instances"), "n_steps": r.get("n_steps"), "seconds": r.get("seconds"),
                                "max_rss_gb": r.get("max_rss_gb"), "timings": r.get("timings"), "memory": r.get("memory"),
                                "digest_equals_record": (fsum or {}).get("digest") == r.get("digest") if fsum else None}
                if not a.keep_refold:
                    for fn in ("program.json", "instances.jsonl", "accesses.jsonl.gz"):
                        try:
                            os.remove(os.path.join(refold, fn))
                        except OSError:
                            pass
            doc["stages"]["fold"] = st
            json.dump(doc, open(os.path.join(a.out, "profile.json"), "w"), indent=1, default=str)

    child_base = [PY, "-m", "verity_capture.experimental.cb_a.match_profile", "--out", a.out, "--rate", str(a.rate)] + (["--sample"] if a.sample else [])
    if "x09" in stages:
        argv = ["--oracle", row["match"]]
        for dd in row["derived"]:
            argv += ["--derived", dd]
        argv += ["--out", os.path.join(a.out, "match_decomp.json"), "--within-step", a.within_step]
        st = _run_stage("x09", child_base + ["--child", "x09", "--child-argv", json.dumps(argv)], a.out, a.pyspy, a.rate)
        st["child"] = _load(os.path.join(a.out, "x09.rusage.json"))
        st["verdict"] = _load(os.path.join(a.out, "match_decomp.json"), keys=("verdict", "reasons", "seconds", "attribution", "xreq_total"))
        doc["stages"]["x09"] = st
        json.dump(doc, open(os.path.join(a.out, "profile.json"), "w"), indent=1, default=str)

    if "gm01" in stages:
        argv = []
        if row["global_program"]:
            argv += ["--global-program", row["global_program"]]
        elif row["gprog"]:
            argv += ["--gprog", row["gprog"], "--workload", row["workload"]]
        else:
            argv += ["--workload", row["workload"]]
            for dd in row["derived"]:
                argv += ["--component", dd]
        argv += ["--oracle", row["match"], "--out", os.path.join(a.out, "global_match.json"), "--within-step", a.within_step]
        if a.no_alternate:
            argv.append("--no-alternate")
        if a.no_x09:
            argv.append("--no-x09")
        st = _run_stage("gm01", child_base + ["--child", "gm01", "--child-argv", json.dumps(argv)], a.out, a.pyspy, a.rate)
        st["child"] = _load(os.path.join(a.out, "gm01.rusage.json"))
        st["verdict"] = _load(os.path.join(a.out, "global_match.json"),
                              keys=("verdict", "failed_checks", "seconds", "within_step_criterion", "achieved_concurrency", "eos_truncated_requests"))
        g = _load(os.path.join(a.out, "global_match.json"))
        if g:
            st["checks"] = [(k["id"], k["status"], k.get("n_problems")) for k in g.get("checks", [])]
            st["x09_inside"] = {k: (g.get("x09") or {}).get(k) for k in ("verdict", "seconds")}
            st["alternate"] = {k: (g.get("alternate_verdict") or {}).get(k) for k in ("criterion", "verdict", "seconds")}
            g7 = next((k for k in g.get("checks", []) if k["id"] == "G7"), {})
            st["population"] = {k: g7.get(k) for k in ("population_query", "population_fold", "boundary_probes", "unexercised_declared")}
            st["requests"] = [{k: e.get(k) for k in ("request_id", "LP", "n_instances", "local_steps", "prefill_chunks", "executed_steps_equal")}
                              for e in g.get("requests", [])]
        doc["stages"]["gm01"] = st
        json.dump(doc, open(os.path.join(a.out, "profile.json"), "w"), indent=1, default=str)

    _aggregate_all(doc, a.out)
    json.dump(doc, open(os.path.join(a.out, "profile.json"), "w"), indent=1, default=str)
    open(os.path.join(a.out, "MATCH-PROFILE.md"), "w").write(render(doc))
    print(render(doc))
    return 0


def _aggregate_all(doc: dict, out: str) -> None:
    for name, st in doc["stages"].items():
        raw = st.get("pyspy_raw")
        smp = os.path.join(out, f"{name}.sample.raw.txt")
        if raw and os.path.exists(raw):
            st["pyspy_buckets"] = aggregate_pyspy(raw)
        elif os.path.exists(smp):
            st["pyspy_buckets"] = aggregate_pyspy(smp)
            st["pyspy_buckets"]["source"] = "in-process SIGPROF sampler (ITIMER_PROF)"


def _load(p: str, keys: tuple | None = None):
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}
    if keys:
        return {k: d.get(k) for k in keys}
    return d


# ------------------------------------------------------------------------------------------------------ py-spy aggregation

_FRAME_RE = re.compile(r"^(?P<fn>.+?) \((?P<file>[^:]+):(?P<line>\d+)\)$")

# BRIEF §5 step 1 buckets.  Each rule: (bucket, predicate over the parsed stack, leaf first).  First matching rule wins; rules are ordered
# from the most specific (innermost function) to the loop bodies of check()/decompose() by line range (revision 52e85363b of the checker).
CHECK_LINES = [  # global_match.check own lines -> sub-phase (52e85363b)
    (548, 611, "gm01.load+x09 (read_record/load_oracle/decompose calls)"),
    (612, 795, "global.G1 requests_declared (manifest/header/applicability)"),
    (796, 821, "attribution.G2 attribute every fold instance (meta channel / rows)"),
    (822, 841, "global.G2 requests_observed"),
    (842, 909, "compare.G3-5 per request: Projection + canon() every instance + operand chronology"),
    (910, 947, "compare.G3 component load + compare_steps (sequence) per request"),
    (948, 1007, "compare.G3 dag: dag_hashes(A)+dag_hashes(P) per request + per-step hash matching"),
    (1008, 1042, "global.G4/G5 identity map + shared state"),
    (1043, 1300, "global.G6 chronology (events/EOS/lag/feed-forward/concurrency/sampler geometry)"),
    (1301, 1341, "global.G7 population + boundary probes"),
    (1342, 1407, "global.G8 layout + segmentation"),
    (1408, 1427, "global.G8 dependency order: canon() every instance again"),
    (1428, 1490, "global.G8 permutations: dag_hashes(A)+dag_hashes(P) per permuted step (<=32)"),
    (1491, 1516, "gm01.verdict + alternate criterion re-run"),
]
DECOMP_LINES = [
    (320, 345, "x09.load (read_steps/load_oracle)"),
    (346, 406, "attribution.X-09 attribute every fold instance (meta/dataflow/rows)"),
    (407, 470, "compare.X-09 per request: Projection + component load"),
    (471, 523, "compare.X-09 compare_steps / live_cone / dag_hashes per request"),
]


@functools.lru_cache(maxsize=None)
def _section_lines(path: str, fn_name: str, fallback: tuple) -> tuple:
    """[R17 mprof] The sub-phase table of `check()` / `decompose()` read from the checker's OWN source: every
    `    # ---------------- <title>` section comment inside the function opens a bucket that runs to the next one (or the function's
    end).  Line numbers move with every edit of global_match.py (stoch's G6 work moved G7 by +70 lines and the static table labelled
    G7's probe loop as G8); the section titles do not.  Falls back to the static table when the source has no section comments."""
    try:
        src = open(path).read().splitlines()
    except OSError:
        return fallback
    start = next((i for i, l in enumerate(src) if l.startswith(f"def {fn_name}(")), None)
    if start is None:
        return fallback
    end = next((i for i in range(start + 1, len(src)) if src[i].startswith("def ") or src[i].startswith("class ")), len(src))
    secs = [(i + 1, re.sub(r"^\s*# -+\s*", "", src[i]).strip()) for i in range(start, end) if re.match(r"^\s{4}# -{20,}", src[i])]
    if not secs:
        return fallback
    out = [(start + 1, secs[0][0] - 1, f"{fn_name}: prologue (loads / calls before the first section)")]
    for j, (ln, title) in enumerate(secs):
        hi = secs[j + 1][0] - 1 if j + 1 < len(secs) else end
        out.append((ln, hi, f"{fn_name} § {title[:90]}"))
    return tuple(out)


def _bucket(stack: list[tuple[str, str, int]]) -> str:
    """stack: leaf-first list of (fn, file, line)."""
    fns = [f for f, _, _ in stack]
    files = [os.path.basename(p) for _, p, _ in stack]
    nested_check = fns.count("check") >= 2
    prefix = "alternate." if nested_check else ""
    under_x09 = "decompose" in fns
    in_gm = "check" in fns
    for i, (fn, file, line) in enumerate(stack):   # leaf first: innermost specific functions
        b = os.path.basename(file)
        if fn == "decode_program" or (b == "codec.py" and fn in ("_decode_refs", "get", "static_of", "_lookup_primitive")):
            return prefix + "load.decode_program (operation resolution: descriptor -> IR)"
        if b == "refs.py" and fn in ("refs_from_json", "runs", "_runs"):
            return prefix + "load.decode_program (operation resolution: descriptor -> IR)"
        if fn == "dag_hashes":
            where = "x09" if under_x09 else "gm01"
            return f"{prefix}compare.dag_hashes [{where}]"
        if fn == "_canonical_hash":
            return f"{prefix}compare._canonical_hash [{'x09' if under_x09 else 'gm01'}]"
        if fn == "compare_steps" or fn == "_canon_step_relative":
            return f"{prefix}compare.compare_steps [{'x09' if under_x09 else 'gm01'}]"
        if fn == "live_cone":
            return prefix + "compare.live_cone [x09]"
        if fn in ("_segment", "_bind_splits") or (fn == "__init__" and b == "batch_decomp.py"):
            return prefix + "compare.Projection/_segment"
        if fn in ("read_record", "read_steps"):
            return "record.read capture log (gzip + json)"
        if fn == "load_derived":
            return prefix + "load.load_derived component (gzip descriptor json + Prog)"
        if fn == "load_oracle":
            return "load.load_oracle fold (program.json json.load + instances.jsonl + wmap)"
        if fn == "__init__" and b == "compare.py":
            return prefix + "load.Prog.__init__ (instances from IR: runs())"
        if fn == "sampler_geometry_check":
            return prefix + "global.G6 sampler geometry"
        if fn == "canonicalize_record":
            return prefix + "attribution.canonicalize_record"
        if fn == "from_gprog":
            return "load.from_gprog (GP-01 artifact + descriptor digest)"
        if fn in ("dump", "dumps", "_iterencode", "_iterencode_dict", "_iterencode_list", "encode") and b in ("__init__.py", "encoder.py") and not in_gm and not under_x09:
            return "serialize.json.dump result"
        if fn in ("dump", "dumps") and b in ("__init__.py",) and ("main" in fns) and not in_gm and not under_x09:
            return "serialize.json.dump result"   # json.dumps INSIDE check()/decompose() belongs to _canonical_hash/dag_hashes (outer frames)
        if fn == "check" and b == "global_match.py":
            for lo, hi, name in _section_lines(file, "check", tuple(CHECK_LINES)):
                if lo <= line <= hi:
                    return prefix + name
            return prefix + "gm01.check other"
        if fn == "decompose" and b == "batch_decomp.py":
            for lo, hi, name in _section_lines(file, "decompose", tuple(DECOMP_LINES)):
                if lo <= line <= hi:
                    return prefix + name
            return prefix + "x09.decompose other"
        if fn == "fold" or (b == "fold.py"):
            if fn in ("_step", "_flush_and_execute", "_execute", "_parts", "_compute_emit", "_compute_reads", "_compute_writes", "_read_operand", "_write_operand"):
                return "fold.view/write/version folding (_step/_execute/_compute_*)"
            if fn in ("_finish",):
                return "fold.finish (Program build + digest)"
            if fn in ("_header", "_root", "_weight_root", "_prompt_root", "_alloc", "_free", "_journal", "_mem_event"):
                return "fold.roots/allocs/journal"
        if b in ("patterns.py", "resolver.py"):
            return "fold.operation resolution (patterns/resolver)"
        if fn == "read_log" or (b == "log.py"):
            return "fold.load capture log (gzip + json -> events)"
        if fn == "persist_program" or fn == "encode_program" or (b == "codec.py" and fn.startswith("_encode")):
            return "fold.serialize program.json/instances.jsonl"
    if under_x09:
        return prefix + "x09.other"
    if in_gm:
        return prefix + "gm01.other"
    return "other/import/startup"


def aggregate_pyspy(path: str) -> dict:
    tot = 0
    by: collections.Counter = collections.Counter()
    fn_self: collections.Counter = collections.Counter()
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        stack_s, _, cnt = line.rpartition(" ")
        try:
            n = int(cnt)
        except ValueError:
            continue
        frames = stack_s.split(";")
        parsed = []
        for fr in reversed(frames):  # py-spy raw: root first -> reverse to leaf first
            m = _FRAME_RE.match(fr.strip())
            if m:
                parsed.append((m["fn"], m["file"], int(m["line"])))
            else:
                parsed.append((fr.strip(), "", 0))
        tot += n
        by[_bucket(parsed)] += n
        if parsed:
            f, p, ln = parsed[0]
            fn_self[f"{f} ({os.path.basename(p)})"] += n
    return {"samples": tot, "buckets": [(k, v, round(100.0 * v / tot, 1)) for k, v in by.most_common()],
            "self_top": [(k, v, round(100.0 * v / tot, 1)) for k, v in fn_self.most_common(25)]}


# ------------------------------------------------------------------------------------------------------ report


def _fmt_s(x) -> str:
    if x is None:
        return "-"
    x = float(x)
    return f"{x:.0f} s" if x < 600 else f"{x / 60:.1f} min"


def render(doc: dict) -> str:
    L = [f"# Match phase profile -- {doc.get('row', {}).get('row_dir')}", "",
         f"host `{doc.get('host')}` ({doc.get('cpu_count')} vCPU), python {doc.get('python')}, {doc.get('utc')}; py-spy sampling: {doc.get('pyspy')}", ""]
    ab = doc.get("artifact_bytes") or {}
    L.append("## Artifacts")
    L.append("")
    L.append("| file | bytes |")
    L.append("|---|---:|")
    for k, v in ab.items():
        if k == "components":
            for c, d in (v or {}).items():
                L.append(f"| {c}/descriptor.json.gz | {d.get('descriptor.json.gz')} |")
        else:
            L.append(f"| {k} | {v} |")
    L.append("")
    fs = doc.get("fold_summary_of_record")
    if fs:
        L.append(f"Fold of record: digest `{(fs.get('digest') or '')[:16]}`, {fs.get('n_instances')} instances, {fs.get('n_steps')} steps, "
                 f"fold {_fmt_s(fs.get('seconds'))} (per-step sum {_fmt_s(fs.get('per_step_seconds_sum'))}), max RSS {fs.get('max_rss_gb')} GB; "
                 f"timings {json.dumps(fs.get('timings'))}")
        L.append("")
    L.append("## Stages (unprofiled wall; separate processes, as row_pod.sh runs them)")
    L.append("")
    L.append("| stage | wall | CPU user+sys | peak RSS (GB) | verdict | rc |")
    L.append("|---|---:|---:|---:|---|---:|")
    for name, st in (doc.get("stages") or {}).items():
        if st.get("skipped"):
            L.append(f"| {name} | skipped: {st['skipped']} | | | | |")
            continue
        ch = st.get("child") or {}
        cpu = (ch.get("user_s") or 0) + (ch.get("sys_s") or 0) if ch else (st.get("children_user_s") or 0) + (st.get("children_sys_s") or 0)
        rss = ch.get("max_rss_gb") if ch else (st.get("refold") or {}).get("max_rss_gb") or st.get("children_max_rss_gb_cumulative")
        v = st.get("verdict") or {}
        vs = v.get("verdict") if isinstance(v, dict) else v
        if name == "fold" and st.get("refold"):
            vs = f"digest==record: {st['refold'].get('digest_equals_record')}"
        L.append(f"| {name} | {_fmt_s(st.get('wall_s'))} | {_fmt_s(cpu)} | {rss} | {vs} | {st.get('rc')} |")
    L.append("")
    for name, st in (doc.get("stages") or {}).items():
        ch = st.get("child") or {}
        if ch.get("timers_s"):
            L.append(f"### {name}: phase timers (wall inside the process; outer calls only)")
            L.append("")
            L.append("| phase function | calls | wall |")
            L.append("|---|---:|---:|")
            for k, v in ch["timers_s"].items():
                L.append(f"| {k} | {ch['calls'].get(k)} | {_fmt_s(v)} |")
            L.append("")
            if ch.get("objects"):
                L.append(f"Represented objects: `{json.dumps(ch['objects'], default=str)[:1500]}`")
                L.append("")
        if st.get("refold"):
            r = st["refold"]
            L.append(f"### fold: re-fold on this host: {r.get('n_instances')} instances, {r.get('n_steps')} steps, {_fmt_s(r.get('seconds'))}, "
                     f"max RSS {r.get('max_rss_gb')} GB, timings {json.dumps(r.get('timings'))}, digest == record: {r.get('digest_equals_record')}")
            L.append("")
        pb = st.get("pyspy_buckets")
        if pb:
            L.append(f"### {name}: sample buckets (BRIEF §5 phases; {pb['samples']} samples; {pb.get('source', 'py-spy')})")
            L.append("")
            L.append("| bucket | samples | % |")
            L.append("|---|---:|---:|")
            for k, v, pct in pb["buckets"]:
                L.append(f"| {k} | {v} | {pct} |")
            L.append("")
            L.append("Top self-time functions:")
            L.append("")
            for k, v, pct in pb["self_top"][:15]:
                L.append(f"- {k}: {pct}%")
            L.append("")
        if name == "gm01" and st.get("checks"):
            L.append(f"GM-01 checks: {st['checks']}; X-09 inside: {st.get('x09_inside')}; alternate: {st.get('alternate')}; population: {st.get('population')}")
            L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--row-dir", help="row attempt dir (row_pod.sh layout)")
    ap.add_argument("--match-dir", help="override <row>/match")
    ap.add_argument("--workload", help="override <row>/manifest.json")
    ap.add_argument("--gprog", help="override <row>/build_workload/workload_program.json")
    ap.add_argument("--global-program", help="a cb-a/global-program/v1 JSON (mock) instead of --gprog (fixtures)")
    ap.add_argument("--derived", action="append", help="override the component dirs (repeatable)")
    ap.add_argument("--fold-profile", help="fold profile name (default: match/fold_summary.json `profile`)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="x09,gm01", help="comma list of fold,x09,gm01 (fold re-folds the capture log on this host)")
    ap.add_argument("--within-step", default="dag")
    ap.add_argument("--no-alternate", action="store_true")
    ap.add_argument("--no-x09", action="store_true")
    ap.add_argument("--pyspy", action="store_true", help="sample each stage with py-spy (attach by pid, --nonblocking; needs ptrace)")
    ap.add_argument("--sample", action="store_true", help="sample x09/gm01 in-process (SIGPROF; no ptrace needed)")
    ap.add_argument("--rate", type=int, default=50, help="samples per second (py-spy or in-process)")
    ap.add_argument("--keep-refold", action="store_true")
    ap.add_argument("--report", help="re-render MATCH-PROFILE.md from <dir>/profile.json")
    ap.add_argument("--child", choices=["fold", "x09", "gm01"], help=argparse.SUPPRESS)
    ap.add_argument("--child-argv", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return _child(a)
    if a.report:
        doc = json.load(open(os.path.join(a.report, "profile.json")))
        _aggregate_all(doc, a.report)
        json.dump(doc, open(os.path.join(a.report, "profile.json"), "w"), indent=1, default=str)
        open(os.path.join(a.report, "MATCH-PROFILE.md"), "w").write(render(doc))
        print(render(doc))
        return 0
    if not a.row_dir:
        ap.error("--row-dir is required")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
