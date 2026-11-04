"""``verity bench recursion_e2e COMMAND``: the recursive run through pod 85's steps, timed step by step (README.md).

    plan   [POINT]                 the steps, each statement's lease, and the job's launch command
    run    [POINT] [--e2e] [--dry-run] [--here]
                                   one recorded run of job.sh: a firewalled run (85's CPU prover) or --cpu a plain run on
                                   a CPU node pinned to --cores, a GPU run a GPU node's queue (--here: job.sh in this
                                   shell, e.g. inside a run); --e2e: one pass, every step once, no warm run
    result --dir OUT [--out FILE] [--md FILE]
                                   the result from a job's out/ (the job runs it last, into out/recursion_e2e.json)
    table  RESULT... [--merge] [--json]
                                   the table from results: a recursion_e2e.json or a run id (its records fetched and read)
    store  RESULT... [--merge]     the results and their table as one art in the evidence store; prints its art: id

POINT: --K 4096 --N 2 --steps S,.. --repeat 3 --modes ZK server --oprove-runs 1 --lease-min M --neg challenge,query
       --forge KINDS (85's forgeries; with coef, dirs or message the forge-after step too, unless --no-forge-after)
       --cpu (GPU=0: the CPU prover, no lease) --on MACHINE --cores 0-31 --mem-gb G --cpus 32 --max-min M
       --env KEY=VALUE ...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rec_profile as RP  # noqa: E402
import settings as RS  # noqa: E402

NODE = "vy-nebius-1"
STEPS = RP.ORDER
E2E = (*RP.E2E_STEPS, "negative")
KIND = "bench/recursion-e2e/v1"
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")

# Each V* statement's firewalled release at K = 4096, N = 2, end to end (s), on the CPU prover, one at a time. No one
# ratio to the dev run fits, so each statement has its own. A release's time is the Lean session loop's.
# A lease or a run holds MARGIN times its statements' releases.
FW_S = {"L0": 3738, "L1": 894, "L2": 836, "L3": 469, "alg-p0": 815, "alg-p1": 847,
        "cap": 63, "cap-L1": 86, "cap-L2": 74, "cap-L3": 51}
MARGIN = 1.5
SETUP_MIN = 5  # the job's start, its retention record and its result
OTHER_MIN = {"build": 6, "inner": 3, "icheck": 1, "vstage": 6, "vcap": 1, "oreplay": 2, "lean": 12, "forge-after": 15,
             "negative": 30}
#: the forgeries the ZK server stops before release (FIREWALL-REFUSED, no release for Lean), which forge-after makes on the
#: honest releases after proving instead (forge_after.CASES)
FORGE_AFTER = ("coef", "dirs", "message")
#: rec_vstage.FORGERIES, each with the V* statements it stages under forged-<kind>/ and oprove proves one release each, named
#: as LEVELS names them (`statements`): the levels it forges (L3 is the last at K = 4096; sum's openings either side of the
#: reps' seam), and with the algebra every part, those it leaves as the honest one too (forged-message's alg-p1). Each is
#: sized as its honest statement's firewalled release, which bounds the forged one
FORGED = {"message": "alg", "sum": "L0,L3,alg", "comb": "alg", "top": "L3", "top-salt": "L0", "coef": "L0", "dirs": "L0",
          "filler": "alg"}

# Where a run goes. 85's CPU prover needs no GPU, so a firewalled run (and --cpu) goes straight to a CPU node that takes
# plain runs, pinned to CORES; the SLOT_ONLY nodes take runs only through slots.
CPU_NODE, PLAIN_CPU_NODES = "vy-nebius-cpu-2", ("vy-nebius-cpu-2", "vy-nebius-cpu-3", "vy-nebius-cpu-4", "vy-nebius-cpu-5")
SLOT_ONLY = ("vy-nebius-1", "vy-nebius-2", "vy-nebius-3", "vy-nebius-cpu-1")
CPU_WORK = "/workspace/cache/rec-bench"

STAGES = {
    "build": "the CUDA prover (60-circuit.sh), flock-verify (lake build), the inner statement staged with plain and hm96 leaves",
    "inner": "M0 (ZK off) against the rec_live proxy, WARM + RUNS runs; the non-recursive M0 and --zk on loopback beside it",
    "icheck": "upstream's replay of the inner run and V*'s hashing reference (rec_vstar)",
    "vstage": "V*'s fold points, InnerFold.lean, and each statement staged (RecOpen per level, InnerRepCheck parts)",
    "vcap": "RecCap over each level's committed tops",
    "oprove": "each V* statement proved with --zk on loopback serve, one at a time, per mode (firewall: the Lean session loop)",
    "oreplay": "upstream's replay --zk of each V* release",
    "lean": "flock-verify verify --zk on each V* release with V*'s own inputs, the inner run, the inner statement",
    "forge-after": "coef, dirs and message forged on the honest L0 and alg-p0 releases' statements after proving "
                   "(forge_after.py), each refused by Lean beside its accepted honest control",
    "negative": "the lean step again over each copy of the inner run with one coin of rep 0 flipped, which Lean must refuse",
}


def statements(levels: str = "") -> list[str]:
    """V*'s statements 85 proves under ``LEVELS`` (its ``want``: a name, or the name less its ``-p<N>`` part)."""
    want = {x for x in levels.split(",") if x}
    return [s for s in FW_S if not want or s in want or re.sub(r"-p\d+$", "", s) in want]


def stmt_lease_min(stmt: str, sessions: int = 1) -> int:
    """A lease (minutes) that holds ``sessions`` firewalled releases of ``stmt`` with MARGIN."""
    return math.ceil(MARGIN * sessions * FW_S[stmt] / 60)


def lease_min(sessions: int, firewall: bool = True, levels: str = "") -> int:
    """The job's one lease (85's GPU_MAX_MIN covers every prove of an invocation): its largest statement's; 85's default
    10 without the ZK server."""
    return max((stmt_lease_min(s, sessions) for s in statements(levels)), default=10) if firewall else 10


def levels(a) -> str:
    """LEVELS as --env passes it to 85 (the last one wins)."""
    return next((e.split("=", 1)[1] for e in reversed(a.env) if e.startswith("LEVELS=")), "")


def max_min(a) -> int:
    """The job's limit: its statements' firewalled releases (with MARGIN) plus each other step's minutes per repeat."""
    sessions = a.oprove_warm + a.oprove_runs
    fw = "firewall" in a.modes.split(",")
    forged = [s for k in forge(a) for s in statements(FORGED[k])]
    oprove = MARGIN * (sessions * sum(FW_S[s] for s in statements(levels(a))) + sum(FW_S[s] for s in forged)) / 60 if fw else 15
    rest = sum(m * (a.repeat if s in ("icheck", "vstage", "vcap", "oreplay", "lean") else 1)
               for s, m in OTHER_MIN.items() if s in steps(a.steps))
    # 85's CPU prover built once (oprove under the ZK server, OPROVE_GPU=0), and its proofs: a minute a statement
    cpu = 25 if fw and a.oprove_gpu != "1" else 0
    return math.ceil(SETUP_MIN + (oprove + cpu if "oprove" in steps(a.steps) else 0) + rest)


def cpu_run(a) -> bool:
    """A run with no GPU prove: --cpu, or a firewalled one on 85's CPU prover (the default)."""
    return a.cpu or ("firewall" in a.modes.split(",") and a.oprove_gpu != "1")


def on(a) -> str:
    """The machine: --on, else a plain CPU node for a CPU run and the GPU node's queue for a GPU one."""
    m = a.on or (CPU_NODE if cpu_run(a) else NODE)
    if m in PLAIN_CPU_NODES and not cpu_run(a):
        raise SystemExit(f"--on {m}: a CPU node has no GPU; --oprove-gpu 1 and GPU modes need a GPU node")
    if m in SLOT_ONLY and cpu_run(a):
        raise SystemExit(f"--on {m}: takes runs only through its slots; a CPU run goes to one of {', '.join(PLAIN_CPU_NODES)}")
    return m


def mem_gb(a) -> int:
    """The job's --mem-gb, which a queued node makes its MemoryMax: under the ZK server the Lean session loop and the
    prover together peak near 100 GB on L0, else 64."""
    return a.mem_gb or (128 if "firewall" in a.modes.split(",") and "oprove" in steps(a.steps) else 64)


def steps(arg: str) -> list[str]:
    got = [s for s in arg.split(",") if s]
    bad = [s for s in got if s not in STEPS]
    if bad:
        raise SystemExit(f"--steps: {', '.join(bad)} not among {', '.join(STEPS)}")
    return [s for s in STEPS if s in got]


def preset(a: argparse.Namespace) -> argparse.Namespace:
    """``--e2e``: one pass: the steps one recursive run needs and its negative, each once,
    one release per prove and no warm one."""
    if a.e2e:
        a.steps, a.repeat, a.oprove_runs, a.oprove_warm = ",".join(E2E), 1, 1, 0
        a.env = ["WARM=0", "RUNS=1", *a.env]
    if forge_after(a) and "lean" in steps(a.steps) and not a.no_forge_after:
        a.steps = ",".join([*steps(a.steps), "forge-after"])
    return a


def forge(a) -> list[str]:
    kinds = a.forge.replace(",", " ").split()
    bad = [k for k in kinds if k not in FORGED]
    if bad:
        raise SystemExit(f"--forge: {', '.join(bad)} not among {', '.join(FORGED)}")
    return kinds


def forge_after(a) -> list[str]:
    """The forged statements' kinds forge-after makes after proving: on whenever --forge runs one of them."""
    return [k for k in FORGE_AFTER if k in forge(a)]


def direct(a) -> bool:
    """A plain run on a CPU node (``research run --on``, pinned to --cores), not a queued job."""
    return on(a) in PLAIN_CPU_NODES


def job_env(a: argparse.Namespace) -> list[str]:
    bad = [e for e in a.env if not re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", e)]
    if bad:
        raise SystemExit(f"--env: {', '.join(bad)} is not KEY=VALUE")
    lease = a.lease_min or lease_min(a.oprove_warm + a.oprove_runs, "firewall" in a.modes.split(","), levels(a))
    # a CPU node has no GPU, and /workspace/jobs isn't the research user's there
    node = ["GPU=0", f"FLOCK_WORK={CPU_WORK}"] if direct(a) else ["GPU=0"] if a.cpu else []
    env = [f"K={a.K}", f"N={a.N}", f"STEPS={','.join(steps(a.steps))}", f"REPEAT={a.repeat}", f"MODES={a.modes}",
           f"OPROVE_WARM={a.oprove_warm}", f"OPROVE_RUNS={a.oprove_runs}", f"GPU_MAX_MIN={lease}", f"NEG={','.join(a.neg.split())}",
           f"FORGE={','.join(forge(a))}", *([f"FORGE_AFTER={','.join(forge_after(a))}"] if "forge-after" in steps(a.steps) else []),
           *node, *([f"OPROVE_GPU={a.oprove_gpu}"] if a.oprove_gpu != "auto" else []), *a.env]
    # job.sh refuses the same on the pod (settings.py record); here it fails before anything is launched
    try:
        RS.validate(dict(e.split("=", 1) for e in env), of_record=False)
    except ValueError as e:
        raise SystemExit(f"--env: {e}") from None
    return env


def job_command(a: argparse.Namespace, root: Path) -> list[str]:
    env, m, limit = job_env(a), on(a), a.max_min or max_min(a)
    job = ["bash", "benchmarks/recursion_e2e/job.sh", *env]
    if direct(a):
        return ["research", "run", "--on", m, "--project", "verity", "--campaign", "recursion-e2e", "--source", str(root),
                "--cwd", "source", "--mem-gb", str(mem_gb(a)), "--timeout", f"{limit}m", "--", "taskset", "-c", a.cores, *job]
    return ["research", "run", "--queue", "--on", m, *([] if a.cpu else ["--gpu-host"]),
            *(["--preemptible"] if a.preemptible else []), "--project", "verity",
            "--campaign", "recursion-e2e", "--source", str(root), "--cwd", "source", "--mem-gb", str(mem_gb(a)),
            "--cpus", str(a.cpus), "--max-min", str(limit),
            "--question", f"recursion_e2e: 85-rec-reprice's steps at K = {a.K}, N = {a.N}, oprove {a.modes}, timed and repeated",
            "--", *job]


def root() -> Path:
    for d in (HERE, *HERE.parents):
        if (d / "uv.lock").is_file():
            return d
    raise SystemExit(f"{HERE}: no workspace root")


def cmd_plan(a) -> int:
    a = preset(a)
    env = dict(e.split("=", 1) for e in job_env(a))
    print(f"K = {a.K}, N = {a.N}: 85-rec-reprice.sh's steps on {on(a)}, CPU steps {a.repeat}x, oprove {a.modes} "
          f"with {a.oprove_warm} warm + {a.oprove_runs} timed releases per statement, each prove's lease {env['GPU_MAX_MIN']} min")
    for s in steps(a.steps):
        print(f"  {s:<8} {STAGES[s]}")
    if "oprove" in steps(a.steps) and "firewall" in a.modes.split(","):
        n = a.oprove_warm + a.oprove_runs
        print("  leases (min): " + ", ".join(f"{s} {stmt_lease_min(s, n)}" for s in statements(levels(a))))
    print(" ".join(job_command(a, root())))
    return 0


def cmd_run(a) -> int:
    a = preset(a)
    if a.here:
        cmd = ["bash", str(HERE / "job.sh"), *job_env(a)]
    else:
        cmd = job_command(a, root())
    print(" ".join(cmd), flush=True)
    return 0 if a.dry_run else subprocess.run(cmd, cwd=root(), check=False).returncode


def cmd_result(a) -> int:
    doc = RP.read(Path(a.dir))
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    md = RP.markdown(doc)
    if a.md:
        Path(a.md).write_text(md)
    print(md)
    print(json.dumps(doc["headline"]))
    return 0 if doc["rows"] else 1


def load(ref: str) -> dict:
    p = Path(ref)
    if p.is_file():
        return json.loads(p.read_text())
    if RUN_ID.fullmatch(ref):
        subprocess.run(["research", "fetch", "--all", ref], check=False, capture_output=True)
        out = Path(RS.RESEARCH_RUNS_DIR or Path.home() / ".research" / "runs") / ref / "out"
        # the records themselves, read by this reader (a job's own result is its commit's reader's, and a job stopped
        # before its end wrote none); the job's result only where the records weren't fetched
        res = next((out / f for f in ("recursion_e2e.json", "profile.json") if (out / f).is_file()), None)
        doc = RP.read(out) if (out / "steps").is_dir() else json.loads(res.read_text()) if res else None
        if doc is None:
            raise SystemExit(f"{ref}: neither out/recursion_e2e.json nor out/steps (research fetch --all {ref} found neither)")
        doc.setdefault("run", ref)
        return doc
    raise SystemExit(f"{ref}: neither a result file nor a run id")


def docs_of(refs: list[str]) -> list[dict]:
    docs = [d | {"ref": d.get("ref") or d.get("run") or r} for r, d in ((r, load(r)) for r in refs)]
    for d in docs:
        if d.get("schema") not in (RP.SCHEMA, "recursion-profile/v1"):
            raise SystemExit(f"{d['ref']}: schema {d.get('schema')!r}, not {RP.SCHEMA}")
    return docs


def render(docs: list[dict], merge: bool) -> tuple[dict | None, str]:
    merged = RP.merge(docs) if merge else None
    md = (f"## {', '.join(merged['runs'])}\n\n{RP.markdown(merged)}" if merged
          else "\n".join(f"## {d['ref']}\n\n{RP.markdown(d)}" for d in docs))
    return merged, md


def cmd_table(a) -> int:
    docs = docs_of(a.results)
    merged, md = render(docs, a.merge)
    print(json.dumps(merged or docs, indent=1) if a.json else md)
    return 0


def cmd_store(a) -> int:
    docs = docs_of(a.results)
    merged, md = render(docs, a.merge)
    with tempfile.TemporaryDirectory() as t:
        for d in docs:
            (Path(t) / f"{Path(d['ref']).stem if Path(d['ref']).is_file() else d['ref']}.json").write_text(json.dumps(d, indent=1))
        if merged:
            (Path(t) / "merged.json").write_text(json.dumps(merged, indent=1))
        (Path(t) / "table.md").write_text(md)
        top = merged or docs[-1]
        meta = json.dumps({"benchmark": "recursion_e2e", "schema": RP.SCHEMA, "results": [d["ref"] for d in docs],
                           "commits": sorted({str(d.get("params", {}).get("commit")) for d in docs}),
                           "ok": (top.get("headline") or {}).get("ok"), "complete": (top.get("headline") or {}).get("complete")})
        p = subprocess.run(["research", "data", "put", "--kind", KIND, "--tree", t, "--preserve", "--json", "--meta", meta],
                           capture_output=True, text=True, check=False)
    if p.returncode:
        print(f"research data put: {(p.stderr or p.stdout).strip()[-600:]}", file=sys.stderr)
        return 1
    print(json.loads(p.stdout).get("id"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench recursion_e2e", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def point(p):
        p.add_argument("--K", type=int, default=4096)
        p.add_argument("--N", type=int, default=2)
        p.add_argument("--steps", default=",".join(s for s in STEPS if s != "forge-after"),
                       help="default every step but forge-after, which --forge adds")
        p.add_argument("--repeat", type=int, default=3, help="each CPU step's invocations (REPEAT_<STEP>=n via --env)")
        p.add_argument("--modes", default="firewall", help="oprove's prover sides (ZK server, host, dev); the last one's "
                       "releases are the ones oreplay, lean and negative verify (default ZK server: outer proofs go through "
                       "the ZK server, one at a time)")
        p.add_argument("--oprove-warm", type=int, default=0)
        p.add_argument("--oprove-runs", type=int, default=1, help="timed releases per V* statement (default 1)")
        p.add_argument("--lease-min", type=int, default=0, help="each prove's GPU lease, minutes (85's GPU_MAX_MIN; default "
                       "sized from the ZK server's L3 release: lease_min)")
        p.add_argument("--neg", default="challenge,query", help="inner coins the negative flips, space or comma separated ('' for none)")
        p.add_argument("--forge", default="", help=f"85's forgeries vstage stages and oprove proves (of {', '.join(FORGED)}; "
                       "default none); with any of coef, dirs, message the forge-after step runs after lean")
        p.add_argument("--no-forge-after", action="store_true", help="--forge without the forge-after step")
        p.add_argument("--cpu", action="store_true", help="GPU=0: the CPU prover end to end, no lease (a small K and N)")
        p.add_argument("--oprove-gpu", choices=["auto", "0", "1"], default="auto",
                       help="V*'s prover: 0 the CPU prover (auto under the ZK server: node 1's lease pool stops a lease whose "
                            "GPU idles about 10 min, as a firewalled release's does), 1 the GPU one under a lease")
        p.add_argument("--on", "--node", dest="on", default="", metavar="MACHINE",
                       help=f"default {CPU_NODE} for a CPU run (firewalled on the CPU prover, or --cpu), a plain run there; "
                            f"else {NODE}'s queue")
        p.add_argument("--cores", default="0-31", help="a plain run's cores on its CPU node (taskset -c; default 0-31)")
        p.add_argument("--preemptible", action="store_true",
                       help="queue the run preemptible (a guest on a node that admits guests only so)")
        p.add_argument("--mem-gb", type=int, default=0, help="the run's memory (default mem_gb: 128 with a firewalled oprove, else 64)")
        p.add_argument("--cpus", type=int, default=32)
        p.add_argument("--max-min", type=int, default=0, help="the run's limit (default from the lease model: max_min)")
        p.add_argument("--e2e", action="store_true", help="one pass: build, inner, vstage, vcap, oprove, lean, negative once each, "
                       "one release per prove and no warm one")
        p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                       help="passed to job.sh and on to 85 (LEVELS=L0, REPEAT_LEAN=1, TAG=..., KEEP_DATA=1, ...); repeatable")

    point(sub.add_parser("plan"))
    p = sub.add_parser("run")
    point(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--here", action="store_true", help="job.sh in this shell (inside a run), not a new recorded run")
    p = sub.add_parser("result")
    p.add_argument("--dir", required=True)
    p.add_argument("--out")
    p.add_argument("--md")
    for name in ("table", "store"):
        p = sub.add_parser(name)
        p.add_argument("results", nargs="+")
        p.add_argument("--merge", action="store_true", help="one table: a later result's row replaces an earlier one's "
                       "(same step, stage and mode), each row naming its run")
        if name == "table":
            p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "result": cmd_result, "table": cmd_table, "store": cmd_store}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
