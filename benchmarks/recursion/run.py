"""``verity bench recursion profile COMMAND``: the recursion pipeline's stages, profiled one by one (README.md).

    profile plan   [--K 4096] [--N 2] [--steps S,..] [--repeat 3]   the stages, and the job's launch command
    profile run    [same] [--mem-gb 64] [--max-min 300] [--dry-run]   one recorded job on vy-nebius-1 (job.sh); --env
                   [--env KEY=VALUE ...]                              passes 85's own knobs through (LEVELS=L0, ...)
    profile result --dir OUT [--out FILE] [--md FILE]                 the profile from a job's out/ (the job runs it last)
    profile table  RESULT... [--json] [--put]                         the table from results: a profile.json or a run id
                                                                      (its out/profile.json fetched); --put stores the
                                                                      result and table as one art (research data put)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rec_profile as RP  # noqa: E402

NODE = "vy-nebius-1"
STEPS = ("build", "inner", "icheck", "vstage", "vcap", "oprove", "oreplay", "lean")
KIND = "bench/recursion-profile/v1"
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")

STAGES = {
    "build": "the CUDA prover (60-circuit.sh), flock-verify (lake build), the inner statement staged with plain and hm96 leaves",
    "inner": "M0 (ZK off) against the rec_live proxy, WARM + RUNS sessions; today's M0 and --zk on loopback beside it",
    "icheck": "upstream's replay of the inner session and V*'s hashing reference (rec_vstar)",
    "vstage": "V*'s fold points, InnerFold.lean, and each statement staged (RecOpen per level, InnerRepCheck parts)",
    "vcap": "RecCap over each level's committed tops",
    "oprove": "each V* statement proved with --zk on loopback serve, per mode (firewall: the Lean session loop; dev)",
    "oreplay": "upstream's replay --zk of each V* session",
    "lean": "flock-verify verify --zk on each V* session with V*'s own inputs, the inner session, the inner statement",
}


def steps(arg: str) -> list[str]:
    got = [s for s in arg.split(",") if s]
    bad = [s for s in got if s not in STEPS]
    if bad:
        raise SystemExit(f"--steps: {', '.join(bad)} not among {', '.join(STEPS)}")
    return [s for s in STEPS if s in got]


def job_command(a: argparse.Namespace, root: Path) -> list[str]:
    bad = [e for e in a.env if not re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", e)]
    if bad:
        raise SystemExit(f"--env: {', '.join(bad)} is not KEY=VALUE")
    env = [f"K={a.K}", f"N={a.N}", f"STEPS={','.join(steps(a.steps))}", f"REPEAT={a.repeat}", f"MODES={a.modes}", *a.env]
    return ["research", "run", "--queue", "--on", NODE, "--gpu-host", "--project", "verity", "--campaign", "recursion-profile",
            "--source", str(root), "--cwd", "source", "--mem-gb", str(a.mem_gb), "--cpus", str(a.cpus), "--max-min", str(a.max_min),
            "--question", f"recursion profile: 85-rec-reprice's stages at K = {a.K}, N = {a.N}, timed and repeated",
            "--", "bash", "benchmarks/recursion/job.sh", *env]


def root() -> Path:
    for d in (HERE, *HERE.parents):
        if (d / "uv.lock").is_file():
            return d
    raise SystemExit(f"{HERE}: no workspace root")


def cmd_plan(a) -> int:
    print(f"K = {a.K}, N = {a.N}: 85-rec-reprice.sh's steps on {NODE}, CPU steps {a.repeat}x, GPU steps WARM=1 RUNS=3")
    for s in steps(a.steps):
        print(f"  {s:<8} {STAGES[s]}")
    print(" ".join(job_command(a, root())))
    return 0


def cmd_run(a) -> int:
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
    return 0 if doc["rows"] else 1


def load(ref: str) -> dict:
    p = Path(ref)
    if p.is_file():
        return json.loads(p.read_text())
    if RUN_ID.fullmatch(ref):
        subprocess.run(["research", "fetch", "--all", ref], check=False, capture_output=True)
        out = Path.home() / ".research" / "runs" / ref / "out"
        # a job stopped before its end (cancelled, out of time) wrote no profile.json; its steps' records are read as it would
        doc = json.loads((out / "profile.json").read_text()) if (out / "profile.json").is_file() else (
            RP.read(out) if (out / "steps").is_dir() else None)
        if doc is None:
            raise SystemExit(f"{ref}: neither out/profile.json nor out/steps (research fetch --all {ref} found neither)")
        doc.setdefault("run", ref)
        return doc
    raise SystemExit(f"{ref}: neither a profile.json nor a run id")


def cmd_table(a) -> int:
    docs = [d | {"ref": d.get("ref") or d.get("run") or r} for r, d in ((r, load(r)) for r in a.results)]
    for d in docs:
        if d.get("schema") != RP.SCHEMA:
            raise SystemExit(f"{d['ref']}: schema {d.get('schema')!r}, not {RP.SCHEMA}")
    merged = RP.merge(docs) if a.merge else None
    md = (f"## {', '.join(merged['runs'])}\n\n{RP.markdown(merged)}" if merged
          else "\n".join(f"## {d['ref']}\n\n{RP.markdown(d)}" for d in docs))
    print(json.dumps(merged or docs, indent=1) if a.json else md)
    if a.put:
        with tempfile.TemporaryDirectory() as t:
            for d in docs:
                (Path(t) / f"{Path(d['ref']).stem if Path(d['ref']).is_file() else d['ref']}.json").write_text(json.dumps(d, indent=1))
            if merged:
                (Path(t) / "merged.json").write_text(json.dumps(merged, indent=1))
            (Path(t) / "table.md").write_text(md)
            meta = json.dumps({"what": "recursion profile", "results": [d["ref"] for d in docs]})
            p = subprocess.run(["research", "data", "put", "--kind", KIND, "--tree", t, "--preserve", "--json", "--meta", meta],
                               capture_output=True, text=True, check=False)
        if p.returncode:
            print(f"research data put: {(p.stderr or p.stdout).strip()[-600:]}", file=sys.stderr)
            return 1
        print(json.loads(p.stdout).get("id"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench recursion", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fam = ap.add_subparsers(dest="family", required=True)
    sub = fam.add_parser("profile").add_subparsers(dest="cmd", required=True)

    def point(p):
        p.add_argument("--K", type=int, default=4096)
        p.add_argument("--N", type=int, default=2)
        p.add_argument("--steps", default=",".join(STEPS))
        p.add_argument("--repeat", type=int, default=3)
        p.add_argument("--modes", default="firewall,dev", help="oprove's prover sides; the last one's sessions are verified")
        p.add_argument("--mem-gb", type=int, default=64)
        p.add_argument("--cpus", type=int, default=32)
        p.add_argument("--max-min", type=int, default=300)
        p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                       help="passed to job.sh and on to 85 (LEVELS=L0, GPU_MAX_MIN=20, ...); repeatable")

    point(sub.add_parser("plan"))
    p = sub.add_parser("run"); point(p); p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("result"); p.add_argument("--dir", required=True); p.add_argument("--out"); p.add_argument("--md")
    p = sub.add_parser("table"); p.add_argument("results", nargs="+"); p.add_argument("--json", action="store_true")
    p.add_argument("--put", action="store_true")
    p.add_argument("--merge", action="store_true", help="one table: a later result's row replaces an earlier one's "
                   "(same step, stage and mode), each row naming its run")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "result": cmd_result, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
