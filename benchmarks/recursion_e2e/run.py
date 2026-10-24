"""``verity bench recursion_e2e COMMAND``: the recursive session through pod 85's steps, timed stage by stage (README.md).

    plan   [POINT]                 the stages, the lease, and the job's launch command
    run    [POINT] [--e2e] [--dry-run] [--here]
                                   one research run of job.sh on vy-nebius-1 (--here: job.sh in this shell, e.g. inside a
                                   run); --e2e: one pass, every stage once, no warm session (RECBENCH's single-run mode)
    result --dir OUT [--out FILE] [--md FILE]
                                   the result from a job's out/ (the job runs it last, into out/recursion_e2e.json)
    table  RESULT... [--merge] [--json]
                                   the table from results: a recursion_e2e.json or a run id (its records fetched and read)
    store  RESULT... [--merge]     the results and their table as one art (research data put); prints its art: id

POINT: --K 4096 --N 2 --steps S,.. --repeat 3 --modes firewall --oprove-runs 1 --lease-min M --neg challenge,query
       --cpu (GPU=0: the CPU prover, no lease) --node vy-nebius-1 --mem-gb 64 --cpus 32 --max-min M --env KEY=VALUE ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rec_profile as RP  # noqa: E402

NODE = "vy-nebius-1"
STEPS = RP.ORDER
E2E = (*RP.E2E_STEPS, "negative")
KIND = "bench/recursion-e2e/v1"
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")

# The lease (GPU_MAX_MIN) and the job's --max-min, from the firewall's one measured session at K = 4096, N = 2: V*'s L3
# took 519 s end to end with the firewall on (r20261024-071656-66cb) against 2.46 s on the device's witness (dev,
# r20261024-063425-1ec5), and the firewall's time follows the dev session's (L1, 3.02 s dev, outran a 10 min lease). The
# largest statement (alg-p1) is 13.99 s dev and the ten together 46.7 s, so one firewalled session of alg-p1 is about
# 2950 s and of all ten about 9850 s; a lease holds MARGIN times its statement's sessions.
FW_L3_S, DEV_L3_S, DEV_MAX_S, DEV_SUM_S, MARGIN = 519.2, 2.464, 13.99, 46.68, 1.5
OTHER_MIN = {"build": 6, "inner": 3, "icheck": 1, "vstage": 6, "vcap": 1, "oreplay": 2, "lean": 10, "negative": 25}

STAGES = {
    "build": "the CUDA prover (60-circuit.sh), flock-verify (lake build), the inner statement staged with plain and hm96 leaves",
    "inner": "M0 (ZK off) against the rec_live proxy, WARM + RUNS sessions; today's M0 and --zk on loopback beside it",
    "icheck": "upstream's replay of the inner session and V*'s hashing reference (rec_vstar)",
    "vstage": "V*'s fold points, InnerFold.lean, and each statement staged (RecOpen per level, InnerRepCheck parts)",
    "vcap": "RecCap over each level's committed tops",
    "oprove": "each V* statement proved with --zk on loopback serve, one at a time, per mode (firewall: the Lean session loop)",
    "oreplay": "upstream's replay --zk of each V* session",
    "lean": "flock-verify verify --zk on each V* session with V*'s own inputs, the inner session, the inner statement",
    "negative": "the lean step again over each copy of the inner session with one coin of rep 0 flipped, which Lean must refuse",
}


def lease_min(sessions: int, firewall: bool = True) -> int:
    """A lease (minutes) that holds the largest statement's ``sessions`` firewalled sessions with MARGIN; 85's default 10
    without the firewall."""
    return math.ceil(MARGIN * sessions * FW_L3_S * DEV_MAX_S / DEV_L3_S / 60) if firewall else 10


def max_min(a) -> int:
    """The job's --max-min: every statement's firewalled sessions (with MARGIN) plus each other step's minutes per repeat."""
    sessions = a.oprove_warm + a.oprove_runs
    oprove = MARGIN * sessions * FW_L3_S * DEV_SUM_S / DEV_L3_S / 60 if "firewall" in a.modes.split(",") else 15
    rest = sum(m * (a.repeat if s in ("icheck", "vstage", "vcap", "oreplay", "lean") else 1)
               for s, m in OTHER_MIN.items() if s in steps(a.steps))
    # 85's CPU prover built once (oprove under the firewall, OPROVE_GPU=0), and its proofs: a minute a statement
    cpu = 25 if "firewall" in a.modes.split(",") and a.oprove_gpu != "1" else 0
    return math.ceil((oprove + cpu if "oprove" in steps(a.steps) else 0) + rest)


def mem_gb(a) -> int:
    """The job's --mem-gb, which a queued node makes its MemoryMax: under the firewall the Lean session loop and the
    prover together peak at 99 GB on L0 (57 GB and 31 GB, r20261024-094937-6572), else 64."""
    return a.mem_gb or (128 if "firewall" in a.modes.split(",") and "oprove" in steps(a.steps) else 64)


def steps(arg: str) -> list[str]:
    got = [s for s in arg.split(",") if s]
    bad = [s for s in got if s not in STEPS]
    if bad:
        raise SystemExit(f"--steps: {', '.join(bad)} not among {', '.join(STEPS)}")
    return [s for s in STEPS if s in got]


def preset(a: argparse.Namespace) -> argparse.Namespace:
    """``--e2e``: one pass, as RECBENCH's draft ran it: the steps one recursive session needs and its negative, each once,
    one session per prove and no warm one."""
    if a.e2e:
        a.steps, a.repeat, a.oprove_runs, a.oprove_warm = ",".join(E2E), 1, 1, 0
        a.env = ["WARM=0", "RUNS=1", *a.env]
    return a


def job_env(a: argparse.Namespace) -> list[str]:
    bad = [e for e in a.env if not re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", e)]
    if bad:
        raise SystemExit(f"--env: {', '.join(bad)} is not KEY=VALUE")
    lease = a.lease_min or lease_min(a.oprove_warm + a.oprove_runs, "firewall" in a.modes.split(","))
    return [f"K={a.K}", f"N={a.N}", f"STEPS={','.join(steps(a.steps))}", f"REPEAT={a.repeat}", f"MODES={a.modes}",
            f"OPROVE_WARM={a.oprove_warm}", f"OPROVE_RUNS={a.oprove_runs}", f"GPU_MAX_MIN={lease}", f"NEG={','.join(a.neg.split())}",
            *(["GPU=0"] if a.cpu else []), *([f"OPROVE_GPU={a.oprove_gpu}"] if a.oprove_gpu != "auto" else []), *a.env]


def job_command(a: argparse.Namespace, root: Path) -> list[str]:
    env = job_env(a)
    return ["research", "run", "--queue", "--on", a.node, *([] if a.cpu else ["--gpu-host"]),
            *(["--preemptible"] if a.preemptible else []), "--project", "verity",
            "--campaign", "recursion-e2e", "--source", str(root), "--cwd", "source", "--mem-gb", str(mem_gb(a)),
            "--cpus", str(a.cpus), "--max-min", str(a.max_min or max_min(a)),
            "--question", f"recursion_e2e: 85-rec-reprice's stages at K = {a.K}, N = {a.N}, oprove {a.modes}, timed and repeated",
            "--", "bash", "benchmarks/recursion_e2e/job.sh", *env]


def root() -> Path:
    for d in (HERE, *HERE.parents):
        if (d / "uv.lock").is_file():
            return d
    raise SystemExit(f"{HERE}: no workspace root")


def cmd_plan(a) -> int:
    a = preset(a)
    env = dict(e.split("=", 1) for e in job_env(a))
    print(f"K = {a.K}, N = {a.N}: 85-rec-reprice.sh's steps on {a.node}, CPU steps {a.repeat}x, oprove {a.modes} "
          f"with {a.oprove_warm} warm + {a.oprove_runs} timed sessions per statement, each prove's lease {env['GPU_MAX_MIN']} min")
    for s in steps(a.steps):
        print(f"  {s:<8} {STAGES[s]}")
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
        out = Path(os.environ.get("RESEARCH_RUNS_DIR", Path.home() / ".research" / "runs")) / ref / "out"
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
        p.add_argument("--steps", default=",".join(STEPS))
        p.add_argument("--repeat", type=int, default=3, help="each CPU step's invocations (REPEAT_<STEP>=n via --env)")
        p.add_argument("--modes", default="firewall", help="oprove's prover sides (firewall, host, dev); the last one's "
                       "sessions are the ones oreplay, lean and negative verify (default firewall: outer proofs go through "
                       "the firewall, one at a time)")
        p.add_argument("--oprove-warm", type=int, default=0)
        p.add_argument("--oprove-runs", type=int, default=1, help="timed sessions per V* statement (default 1)")
        p.add_argument("--lease-min", type=int, default=0, help="each prove's GPU lease, minutes (85's GPU_MAX_MIN; default "
                       "sized from the firewall's L3 session: lease_min)")
        p.add_argument("--neg", default="challenge,query", help="inner coins the negative flips, space or comma separated ('' for none)")
        p.add_argument("--cpu", action="store_true", help="GPU=0: the CPU prover end to end, no lease (a small K and N)")
        p.add_argument("--oprove-gpu", choices=["auto", "0", "1"], default="auto",
                       help="V*'s prover: 0 the CPU prover (auto under the firewall: node 1's lease pool stops a lease whose "
                            "GPU idles about 10 min, as a firewalled session's does), 1 the GPU one under a lease")
        p.add_argument("--node", default=NODE)
        p.add_argument("--preemptible", action="store_true",
                       help="queue the run preemptible (a guest on another lane's node, vy-nebius-2's rule)")
        p.add_argument("--mem-gb", type=int, default=0, help="the run's memory (default mem_gb: 128 with a firewalled oprove, else 64)")
        p.add_argument("--cpus", type=int, default=32)
        p.add_argument("--max-min", type=int, default=0, help="the run's limit (default from the lease model: max_min)")
        p.add_argument("--e2e", action="store_true", help="one pass: build, inner, vstage, vcap, oprove, lean, negative once each, "
                       "one session per prove and no warm one")
        p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                       help="passed to job.sh and on to 85 (LEVELS=L0, REPEAT_LEAN=1, TAG=..., KEEP_DATA=1, ...); repeatable")

    point(sub.add_parser("plan"))
    p = sub.add_parser("run")
    point(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--here", action="store_true", help="job.sh in this shell (inside a run), not a new research run")
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
