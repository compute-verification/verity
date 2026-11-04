"""``verity bench proof_throughput`` (``python -m verity_proof_throughput``): plan, launch, judge and render the sweep.

    plan     [--workload W] [--lo B] [--hi B] [--json]   each point's identity and its passing prover run, if any
    run      [--workload W] [--lo B] [--hi B] [--dry-run]  one recorded prover run per point without a passing one
    verify   --prove-run RUN [--node vy-nebius-cpu-3|-cpu-2|vy-nebius-1] [--dry-run]   the Lean verifier point on a prover run's bundle
    check    [--workload W] ...                          exit 0 only when every point has a passing prover run
    compare  A B [--allow FIELD]                         accept two results only when their identities agree
    table    [RESULT ...] [--store --workload W [--commit C]] [--baseline] [--json]   the throughput frontier over B
    baseline [--workload W] [--lock RESULT ...]          the pinned hill-climb baseline (baseline.json), or pin one sweep
    instances --workload W [--write]                     regenerate the workload's frozen instance manifest
    workloads                                            the VU families and their B caps
    zk                                                   what configuration 2 measures today (job.sh ZK=1: --zk sessions under the ZK server)

The job scripts call the rest: ``population``, ``check-instances``, ``partition``, ``staged``, ``probe``, ``put-bundle``,
``fetch-bundle``, ``lean-verdict``, ``point-result``, ``zk-point`` and ``profile-phases``.
A RESULT is a ``result.json`` path or a run id (its result fetched from the evidence store).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import instances as I
from . import plan as P
from . import report as R
from . import result as RS
from .workloads import WORKLOADS, workload

DEFAULT_WORKLOAD = "bf16-k6144"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip().startswith("{")]
    except OSError:
        return []


# --- job side ----------------------------------------------------------------------------------------------------------


def cmd_population(a) -> int:
    """The point's population and draw law: ``law.json`` (the verifier's stratified law for B, ``serve --draw-law``) in
    ``--out``; prints the VU Definition and n_max, the class sweep's PT_VU and PT_N."""
    w = workload(a.workload)
    if not 1 <= a.B <= w.n_max:
        print(f"{w.name}: B = {a.B} outside [1, {w.n_max}]", file=sys.stderr)
        return 1
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "law.json").write_text(json.dumps(I.law(w, a.B), sort_keys=True) + "\n")
    print(w.vu, w.n_max)
    return 0


def cmd_check_instances(a) -> int:
    got = I.check(workload(a.workload))
    Path(a.out).write_text(json.dumps(got, indent=1) + "\n")
    print(json.dumps(got))
    return 0 if got["ok"] else 1


def cmd_partition(a) -> int:
    """The verifier's own copies of the partition object the circuit binds (META, its second line) and of the program it
    names (``--program``), from the workload."""
    with open(a.circuit) as f:
        f.readline()
        meta = json.loads(f.readline()[5:])
    w = workload(a.workload)
    try:
        obj = I.partition_object(w, meta)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    Path(a.out).write_text(json.dumps(obj, sort_keys=True))
    if a.program_out:
        Path(a.program_out).write_bytes(I.program_file(w))
    print(meta["partition"]["digest"])
    return 0


def _stage_record(path: Path) -> dict | None:
    recs = [r for r in _jsonl(path) if r.get("shape")]
    return recs[-1] if recs else None


def cmd_staged(a) -> int:
    rec = _stage_record(Path(a.results))
    if rec is None or "stage" not in rec:
        print(f"no staged class in {a.results}: {(rec or {}).get('broke')}", file=sys.stderr)
        return 1
    if a.cache_dir:
        name = rec.get("stage_cached")
        root = os.environ.get("FLOCK_STAGE_CACHE")
        if not name or not root:
            print("the record was not taken from FLOCK_STAGE_CACHE", file=sys.stderr)
            return 1
        print(Path(root) / name)
        return 0
    st = rec["stage"]
    print(json.dumps({"shape": rec["shape"][:16], "n": st.get("n"), "k_log": st.get("k_log"), "ands": rec.get("ands"),
                      "domain": rec.get("domain"), "stage_s": rec.get("stage_s"), "cached": rec.get("stage_cached")}))
    return 0


#: the leased GPU's fixed properties (``job.sh``'s ``gpu.csv``); its uuid and live readings stay out of the identity
LEASED = ("name", "compute_cap", "clocks.max.sm", "power.limit", "memory.total")


def leased_gpu(csv_text: str) -> dict[str, str] | None:
    rows = [[c.strip() for c in ln.split(",")] for ln in csv_text.splitlines() if ln.strip()]
    if len(rows) < 2 or len(rows[0]) != len(rows[1]):
        return None
    got = dict(zip(rows[0], rows[1]))
    return {k: got[k] for k in LEASED if k in got} | {"driver": got.get("driver_version")}


def cmd_probe(a) -> int:
    from verity_numerical.bench import contract

    hw = contract.probe_hardware()
    leased = leased_gpu(Path(a.gpu_csv).read_text()) if a.gpu_csv and Path(a.gpu_csv).exists() else None
    if leased:
        if hw["gpu"] is None:
            mib = leased.get("memory.total") or ""
            hw["gpu"] = {"name": leased["name"], "driver": leased["driver"], "cuda": None, "count": None,
                         "memory_total_bytes": int(mib) << 20 if mib.isdigit() else None}
        hw["gpu"]["leased"] = {k: v for k, v in leased.items() if k != "driver"}
    try:
        hw["affinity_cpus"] = len(os.sched_getaffinity(0))
    except OSError:
        pass
    Path(a.out).write_text(json.dumps(hw, indent=1) + "\n")
    return 0


_MS = re.compile(r"^(inputs|setup(?: \(drawn\))?|.+?): (\d+) ms$")


def cmd_lean_verdict(a) -> int:
    """One JSON line for one session's ``flock-verify verify``: its verdict, time and peak memory."""
    out = Path(a.stdout).read_text() if Path(a.stdout).exists() else ""
    err = Path(a.stderr).read_text() if Path(a.stderr).exists() else ""
    verdict = next((json.loads(ln[len("VERDICT "):]) for ln in out.splitlines() if ln.startswith("VERDICT ")), None)
    line: dict[str, Any] = {"session": a.session, "rc": a.rc, "accepted": bool(verdict and verdict.get("accepted") and a.rc == 0),
                            "why": (verdict or {}).get("why") if verdict else (err.strip().splitlines() or ["no verdict"])[-1][:400]}
    try:
        e, u, s, m = Path(a.time).read_text().split()[-4:]
        line |= {"wall_s": float(e), "cpu_s": float(u) + float(s), "max_rss_kb": int(m)}
    except (OSError, ValueError):
        pass
    for ln in err.splitlines():
        mt = _MS.match(ln.strip())
        if mt:
            key = {"inputs": "inputs_ms"}.get(mt.group(1), "setup_ms" if mt.group(1).startswith("setup") else "session_ms")
            line[key] = int(mt.group(2))
    print(json.dumps(line, sort_keys=True), flush=True)
    return 0


def cmd_profile_phases(a) -> int:
    """``verify.sh PROFILE=1``'s result: the perf samples of one session by stage, phase and kernel (``profile``)."""
    from . import profile as PF

    out = Path(a.stdout).read_text() if a.stdout and Path(a.stdout).exists() else ""
    verdicts = [json.loads(ln[len("VERDICT "):]) for ln in out.splitlines() if ln.startswith("VERDICT ")]
    err = Path(a.stderr).read_text()
    summary = PF.summarize(PF.samples(PF.read_script(Path(a.script))))
    doc = PF.result(summary, Path(a.time).read_text().strip().splitlines()[-1], err, bundle=a.bundle, prove_run=a.prove_run,
                    verdicts=[{k: v.get(k) for k in ("accepted", "why")} for v in verdicts])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    (Path(a.out).parent / "out").mkdir(exist_ok=True)
    (Path(a.out).parent / "out" / "profile.json").write_text(json.dumps(summary | {"times": PF.times(err)}, indent=1))
    Path(a.out).write_text(json.dumps(doc, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ("samples", "phase", "kernel")}))
    return 0 if doc["validation"]["status"] == "passed" else 1


def _timed(lean: list[dict], warm: int) -> list[dict]:
    """The timed sessions' verdicts: sessions are numbered in the order served (l0000-PID, ...), warm ones first."""
    order = sorted(lean, key=lambda v: v.get("session") or "")
    return order[warm:] or order


def cmd_point_result(a) -> int:
    d = Path(a.dir)
    w = workload(a.workload)
    rec = next((r for r in reversed(_jsonl(d / "results.jsonl")) if r.get("live")), None) or {"broke": "no session record"}
    stage = _read_json(d / "stage-rec.json") or {}
    if stage.get("stage_s") is not None:
        rec["stage_setup_s"] = stage["stage_s"]
    pt = _read_json(d / "point.json") or {}
    mem = [float(x) for x in (d / "gpu-mem.txt").read_text().split() if re.fullmatch(r"[0-9.]+", x)] \
        if (d / "gpu-mem.txt").exists() else []
    lean = _jsonl(Path(a.lean)) if a.lean else []
    timed = _timed(lean, int(pt.get("warm", 1))) if lean else []
    hw = _read_json(d / "hardware.json") or {"gpu": None, "cpu": None}
    lean_sha = (d / "lean.sha256").read_text().strip() if (d / "lean.sha256").exists() else None
    lean_key = (d / "lean.key").read_text().strip() if (d / "lean.key").exists() else None
    binsha = (d / "binary.sha256").read_text().strip() if (d / "binary.sha256").exists() else None
    idx = _jsonl(d / "index.jsonl")
    sw = _read_json(d / "software.json")
    if sw is None:
        sw = RS.software(rec, commit=os.environ.get("RESEARCH_SOURCE_SHA") or _git_head(), dirty=None, binary_sha256=binsha)
        if not lean:
            (d / "software.json").write_text(json.dumps(sw, indent=1, sort_keys=True) + "\n")
    if lean:
        sw = {**sw, "verifier": {**sw["verifier"], "sha256": lean_sha, "build_key": lean_key,
                                 "commit": os.environ.get("RESEARCH_SOURCE_SHA") or _git_head()}}
    extra = {"prove_run": a.prove_run or pt.get("run"), "bundle": a.bundle, "sessions": {"warm": pt.get("warm"), "runs": pt.get("runs")},
             "prover_node": "vy-nebius-1", "gpu_lease": "n1-lease (infra/cluster submit.py): one GPU, the sessions only"}
    if lean:
        extra["verifier_hardware"] = _probe_cpu()
    doc = RS.point(w, int(a.B), rec, run_id=os.environ.get("RESEARCH_RUN_ID") or "local", hardware=hw, software=sw,
                   gpu_peak_mib=max(mem) if mem else None, lean=timed if lean else None,
                   evidence=[p for p in ("out/results.jsonl", "out/lean.jsonl", "out/index.jsonl") if (d / Path(p).name).exists()],
                   **extra)
    warm = [v for v in lean if v not in timed]
    if warm and not all(RS.lean_passed(v) for v in warm) and doc["validation"]["status"] == "passed":
        doc["validation"] = {**doc["validation"], "status": "failed",
                             "detail": "a warm session was rejected by the Lean verifier: " +
                                       "; ".join(f"{v['session']}: {v.get('why')}" for v in lean if not RS.lean_passed(v))[:800]}
    rounds = [r["rounds"] for r in idx if isinstance(r.get("rounds"), (int, float))]
    if rounds and doc["validation"]["status"] != "failed":
        doc["measurements"].append({"name": "rounds.protocol", "value": max(rounds), "unit": "count"})
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, a.out)
    print(json.dumps({"status": doc["validation"]["status"], "detail": doc["validation"]["detail"][:300],
                      **{m["name"]: m["value"] for m in doc["measurements"] if m["name"] in R.HEADLINE}}))
    return 0


BUNDLE_KIND = "proof/v1"


def cmd_put_bundle(a) -> int:
    """Put the point's bundle in the evidence store (``--preserve``) and write its ``research data put --json`` answer."""
    from .store import research_data

    p = research_data("put", "--kind", BUNDLE_KIND, "--tree", a.tree, "--preserve", "--json", "--meta", a.meta)
    if p.returncode:
        print(f"research data put: {(p.stderr or p.stdout).strip()[-600:]}", file=sys.stderr)
        return 1
    Path(a.out).write_text(p.stdout)
    doc = json.loads(p.stdout)
    print(doc.get("id"))
    if doc.get("pending"):
        print(f"{doc.get('id')} is not preserved, so the verifier node can't fetch it: {doc['pending']}", file=sys.stderr)
        return 1
    return 0


def cmd_fetch_bundle(a) -> int:
    from .store import research_data

    p = research_data("fetch", a.art, "--to", a.to)
    if p.returncode:
        print(f"research data fetch {a.art}: {(p.stderr or p.stdout).strip()[-600:]}", file=sys.stderr)
    return p.returncode


def _probe_cpu() -> dict:
    from verity_numerical.bench import contract

    return {"cpu": contract.probe_hardware().get("cpu"), "host": os.uname().nodename}


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --- user side ---------------------------------------------------------------------------------------------------------


def _plan(a, lookup=None):
    w = workload(a.workload)
    root = P.repository_root()
    pins = P.pins(root, a.commit)
    kw = {"lookup": lookup} if lookup else {}
    return w, root, pins, P.plan(w, pins, config=a.config, lo=a.lo, hi=a.hi, warm=a.warm, runs=a.runs, **kw)


def cmd_plan(a) -> int:
    w, _root, pins, items = _plan(a, None if a.store else (lambda *_: None))
    if a.json:
        print(json.dumps(P.plan_json(w, pins, items), indent=1))
        return 0
    print(f"{w.name}: {w.vu}, profile {w.profile}, cap B = {w.cap}; commit {pins.commit[:12]}")
    for i in items:
        print(f"  B={i.point.B:<5} {i.identity[:16]}  {i.passing_run or ('missing' if a.store else '-')}")
    return 0


def cmd_check(a) -> int:
    _w, _root, _pins, items = _plan(a)
    missing = [i.point.B for i in items if i.passing_run is None]
    print("complete" if not missing else f"missing B = {missing}")
    return 1 if missing else 0


def cmd_run(a) -> int:
    _w, root, _pins, items = _plan(a, None if a.store else (lambda *_: None))

    def execute(cmd: list[str]) -> int:
        print(" ".join(cmd), flush=True)
        return 0 if a.dry_run else subprocess.run(cmd, cwd=root, check=False).returncode

    return P.dispatch(root, items, execute, mem_gb=a.mem_gb)


def _fetch_run_file(run: str, rel: str) -> Path:
    """A file of a run's directory; ``research fetch`` brings only its records unless ``--all`` (``out/`` included)."""
    path = Path.home() / ".research" / "runs" / run / rel
    subprocess.run(["research", "fetch", "--all", run], check=False, capture_output=True)
    if path.exists():
        return path
    raise SystemExit(f"{run}: no {rel} (research fetch --all {run} found none)")


def cmd_verify(a) -> int:
    bundle = a.bundle or (_read_json(_fetch_run_file(a.prove_run, "out/bundle.json")) or {}).get("id")
    if not bundle:
        raise SystemExit(f"{a.prove_run}: its out/bundle.json names no artifact")
    root = P.repository_root()
    cmd = P.verifier_command(root, bundle if bundle.startswith("art:") else f"art:{bundle}", a.prove_run, node=a.node,
                             mem_gb=a.mem_gb)
    print(" ".join(cmd), flush=True)
    return 0 if a.dry_run else subprocess.run(cmd, cwd=root, check=False).returncode


def _load_result(ref: str) -> dict:
    p = Path(ref)
    if p.exists():
        return json.loads(p.read_text())
    if re.fullmatch(r"r\d{8}-\d{6}-[0-9a-f]{4}", ref):
        return json.loads(_fetch_run_file(ref, "result.json").read_text())
    raise SystemExit(f"{ref}: neither a result.json nor a run id")


def cmd_compare(a) -> int:
    try:
        out = R.compare(_load_result(a.baseline), _load_result(a.candidate), allow=a.allow)
    except R.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0


STORE_QUERY = """
SELECT a.id FROM attempts a
JOIN attempt_params p ON p.attempt = a.id AND p.key = 'workload' AND p.value = '{workload}'
WHERE a.tool = 'proof_throughput_verifier' AND a.validation = 'passed' {commit}
ORDER BY a.start_utc
""".strip()


BASELINE = Path(__file__).resolve().parents[1] / "baseline.json"


def _baselines() -> dict:
    return _read_json(BASELINE) or {}


def cmd_baseline(a) -> int:
    """``--lock RESULT...``: pin those results as the workload's hill-climb baseline; else print the pinned one's run ids."""
    pins = _baselines()
    if not a.lock:
        print(json.dumps(pins.get(a.workload) or {}, indent=1))
        return 0 if pins.get(a.workload) else 1
    try:
        pin = R.baseline([_load_result(r) for r in a.lock], a.commit or _git_head() or "unknown")
    except R.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    BASELINE.write_text(json.dumps({**pins, pin["workload"]: pin}, indent=1, sort_keys=True) + "\n")
    print(f"{BASELINE}: {pin['workload']} at {pin['commit'][:12]}, B = {', '.join(pin['points'])}")
    return 0


def cmd_table(a) -> int:
    refs = list(a.results)
    if a.baseline:
        refs += list((_baselines().get(a.workload) or {}).get("points", {}).values())
    if a.store:
        commit = f"AND a.source_commit = '{a.commit}'" if a.commit else ""
        proc = subprocess.run(["research", "data", "sql", STORE_QUERY.format(workload=a.workload, commit=commit), "--json"],
                              capture_output=True, text=True, check=False)
        if proc.returncode:
            raise SystemExit(f"the evidence index: {proc.stderr.strip()}")
        refs += [r["id"] for r in json.loads(proc.stdout or "[]")]
    try:
        rows = R.frontier([_load_result(r) for r in refs])
    except R.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(R.markdown(rows, a.workload))
        b = R.best(rows)
        if b:
            print(f"best B = {b['B']}: {b['vu.per_second']:.4g} VU/s, overhead {b['overhead.vs_native_peak']:,.0f}x the H200 peak; "
                  f"dominant {b['dominant']} ({b['dominant_share']:.0%} of t.total)")
    return 0


def cmd_instances(a) -> int:
    w = workload(a.workload)
    doc = I.build(w)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if a.write:
        I.manifest_path(w).parent.mkdir(parents=True, exist_ok=True)
        I.manifest_path(w).write_text(text)
        print(f"{I.manifest_path(w)}: sha256 {I.manifest_sha256(w)}")
    else:
        print(json.dumps({k: doc[k] for k in ("tier", "n", "population_digest", "output_words")}, indent=1))
    return 0


def cmd_workloads(_a) -> int:
    for w in WORKLOADS.values():
        print(f"{w.name:<17} {w.vu:<52} cap B = {w.cap:<5} {w.profile}  ({w.stager})")
    return 0


def cmd_zk(_a) -> int:
    from . import zk

    print(zk.describe())
    return 0


def cmd_zk_point(a) -> int:
    """``zk-point``: the ZK point's result (``job.sh ZK=1``) from its records; exit 0 when it passed."""
    from . import zk

    doc = zk.point(Path(a.dir), workload(a.workload), a.B, zk.firewall_on(), os.environ.get("RESEARCH_RUN_ID"))
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": doc["validation"]["status"], "detail": doc["validation"]["detail"]}
                     | {m["name"]: m["value"] for m in doc["measurements"]}))
    return 0 if doc["validation"]["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench proof_throughput", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sweep(p):
        p.add_argument("--workload", default=DEFAULT_WORKLOAD)
        p.add_argument("--config", default="plain")
        p.add_argument("--lo", type=int, default=1)
        p.add_argument("--hi", type=int, default=None)
        p.add_argument("--warm", type=int, default=1)
        p.add_argument("--runs", type=int, default=3)
        p.add_argument("--commit", default=None)

    p = sub.add_parser("plan"); sweep(p); p.add_argument("--json", action="store_true")
    p.add_argument("--store", action="store_true", help="look up each point's passing run in the evidence index")
    p = sub.add_parser("check"); sweep(p)
    p = sub.add_parser("run"); sweep(p); p.add_argument("--dry-run", action="store_true"); p.add_argument("--mem-gb", type=int, default=96)
    p.add_argument("--store", action="store_true", help="skip points that already have a passing run")
    p = sub.add_parser("verify"); p.add_argument("--prove-run", required=True); p.add_argument("--bundle", default=None)
    p.add_argument("--node", default=P.VERIFIER_NODES[0], choices=P.VERIFIER_NODES); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mem-gb", type=int, default=16, help="the Lean verifier peaks near 5 GB on a 2^25-bit block circuit")
    p = sub.add_parser("compare"); p.add_argument("baseline"); p.add_argument("candidate")
    p.add_argument("--allow", action="append", default=[], help="an identity field the comparison deliberately varies")
    p = sub.add_parser("table"); p.add_argument("results", nargs="*"); p.add_argument("--store", action="store_true")
    p.add_argument("--workload", default=DEFAULT_WORKLOAD); p.add_argument("--commit", default=None); p.add_argument("--json", action="store_true")
    p.add_argument("--baseline", action="store_true", help="the workload's pinned baseline (baseline.json)")
    p = sub.add_parser("baseline"); p.add_argument("--workload", default=DEFAULT_WORKLOAD)
    p.add_argument("--lock", nargs="+", default=None, metavar="RESULT", help="pin these results (one sweep) as the baseline")
    p.add_argument("--commit", default=None, help="the commit the sweep ran at (default HEAD)")
    p = sub.add_parser("instances"); p.add_argument("--workload", default=DEFAULT_WORKLOAD); p.add_argument("--write", action="store_true")
    sub.add_parser("workloads")
    sub.add_parser("zk")
    p = sub.add_parser("population"); p.add_argument("--workload", required=True); p.add_argument("--B", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check-instances"); p.add_argument("--workload", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("partition"); p.add_argument("--workload", required=True); p.add_argument("--circuit", required=True)
    p.add_argument("--out", required=True); p.add_argument("--program-out", default=None)
    p = sub.add_parser("staged"); p.add_argument("--results", required=True); p.add_argument("--cache-dir", action="store_true")
    p = sub.add_parser("probe"); p.add_argument("--out", required=True); p.add_argument("--gpu-csv", default=None)
    p = sub.add_parser("put-bundle"); p.add_argument("--tree", required=True); p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fetch-bundle"); p.add_argument("art"); p.add_argument("--to", required=True)
    p = sub.add_parser("lean-verdict")
    for f in ("--session", "--time", "--stdout", "--stderr"):
        p.add_argument(f, required=True)
    p.add_argument("--rc", type=int, required=True)
    p = sub.add_parser("point-result"); p.add_argument("--workload", required=True); p.add_argument("--B", type=int, required=True)
    p.add_argument("--dir", required=True); p.add_argument("--lean", default=None); p.add_argument("--prove-run", default=None)
    p.add_argument("--bundle", default=None); p.add_argument("--out", required=True)
    p = sub.add_parser("zk-point"); p.add_argument("--workload", required=True); p.add_argument("--B", type=int, required=True)
    p.add_argument("--dir", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("profile-phases"); p.add_argument("--script", required=True); p.add_argument("--time", required=True)
    p.add_argument("--stderr", required=True); p.add_argument("--stdout", default=None); p.add_argument("--out", required=True)
    p.add_argument("--bundle", default=None); p.add_argument("--prove-run", default=None)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "check": cmd_check, "run": cmd_run, "verify": cmd_verify, "compare": cmd_compare, "table": cmd_table,
            "baseline": cmd_baseline,
            "instances": cmd_instances, "workloads": cmd_workloads, "zk": cmd_zk, "population": cmd_population,
            "check-instances": cmd_check_instances, "partition": cmd_partition, "staged": cmd_staged, "probe": cmd_probe, "lean-verdict": cmd_lean_verdict,
            "point-result": cmd_point_result, "put-bundle": cmd_put_bundle, "fetch-bundle": cmd_fetch_bundle,
            "profile-phases": cmd_profile_phases, "zk-point": cmd_zk_point}[a.cmd](a)
