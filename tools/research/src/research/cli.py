"""`research` command line.

  research run [--project P] [--stage S] [--timeout T] [--runs-dir D] [tele-run flags...] -- CMD...
  research status [--runs-dir D] [--all] [--json]
  research inspect <run-id|prefix|path> [--runs-dir D] [--json]
  research tele <telemetry subcommand...>        # the full telemetry surface (init, event, sample, classify, ...)

`run` allocates a run id, creates its directory under the runs root, and hands off to the telemetry harness, which writes
job.json, status.json, events.jsonl, resources.jsonl, tee'd logs, and failure.json, then exits with the workload's status.
Anything the harness accepts (--tree-sha, --workload-digest, --input, --declared-output, --policy, ...) passes through.
`--on <machine>` (remote execution) is not implemented yet and says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import runs as R


def _fmt(s: dict) -> str:
    cmd = " ".join(s["command"]) if s.get("command") else "?"
    if len(cmd) > 60:
        cmd = cmd[:57] + "..."
    rc = "" if s["rc"] is None else str(s["rc"])
    cls = s["class"] or ""
    return f"{s['id']:<24} {str(s['state']):<10} {rc:>3}  {cls:<22} {str(s['project']):<14} {str(s['stage']):<18} {cmd}"


def cmd_run(argv: list[str]) -> int:
    if "--" not in argv:
        print("research run: the workload command goes after `--`", file=sys.stderr)
        return 2
    i = argv.index("--")
    opts, command = argv[:i], argv[i + 1 :]
    if not command:
        print("research run: empty command after `--`", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="research run", add_help=True, allow_abbrev=False)
    ap.add_argument("--project", default=None, help="resource/ownership boundary; recorded, and required for anything remote")
    ap.add_argument("--stage", default="run", help="stage name recorded in job.json (default: run)")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--runs-dir", default=None, help="registry root (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--on", default=None, metavar="MACHINE", help="run on an existing authorized machine (not implemented yet)")
    ap.add_argument("--id", default=None, help="use this run id instead of allocating one")
    ns, passthrough = ap.parse_known_args(opts)
    if ns.on:
        print(f"research run --on {ns.on}: remote execution is not implemented yet (tools/research/README.md, step 4)", file=sys.stderr)
        return 2
    root = R.runs_root(ns.runs_dir)
    run_id = ns.id or R.new_run_id()
    run_dir = root / run_id
    if run_dir.exists():
        print(f"research run: {run_dir} already exists", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    tele = ["--attempt-dir", str(run_dir), "--stage", ns.stage, "--job", run_id]
    if ns.project:
        tele += ["--project", ns.project]
    if ns.timeout is not None:
        tele += ["--timeout", str(ns.timeout)]
    tele += passthrough + ["--"] + command
    print(f"research: run {run_id} -> {run_dir}", file=sys.stderr)
    from .telemetry.run import main as tele_run

    rc = tele_run(tele)
    s = R.summarize(run_dir)
    print(f"research: run {run_id} {s['state']} rc={s['rc']} class={s['class'] or '-'}  (research inspect {run_id})", file=sys.stderr)
    return rc


def cmd_status(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research status")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--all", action="store_true", help="every run, not just the last 20")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    rows = R.list_runs(root)
    if not ns.all:
        rows = rows[-20:]
    if ns.json:
        print(json.dumps({"runs_root": str(root), "runs": rows}, indent=2, sort_keys=True))
        return 0
    if not rows:
        print(f"no runs under {root}")
        return 0
    print(f"{'run':<24} {'state':<10} {'rc':>3}  {'class':<22} {'project':<14} {'stage':<18} command")
    for s in rows:
        print(_fmt(s))
    active = sum(1 for s in rows if s["state"] in ("submitted", "admitted", "running", "preserving"))
    print(f"\n{len(rows)} shown ({active} active) under {root}")
    return 0


def cmd_inspect(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research inspect")
    ap.add_argument("ref", help="run id, unique prefix, or directory")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    try:
        run_dir = R.resolve(root, ns.ref)
    except (FileNotFoundError, LookupError) as exc:
        print(f"research inspect: {exc}", file=sys.stderr)
        return 1
    s = R.summarize(run_dir)
    if ns.json:
        print(json.dumps(s, indent=2, sort_keys=True))
        return 0
    print(f"run       {s['id']}")
    print(f"dir       {s['dir']}")
    print(f"state     {s['state']}  rc={s['rc']} signal={s['signal']}")
    if s["class"]:
        print(f"class     {s['class']} ({s['certainty']})")
        if s["why"]:
            print(f"why       {s['why']}")
    print(f"project   {s['project']}  job={s['job']}  lane={s['lane']}  stage={s['stage']}")
    print(f"source    tree_sha={s['tree_sha']}  workload_digest={s['workload_digest']}")
    print(f"time      started={s['started_utc']}  ended={s['ended_utc']}")
    print(f"command   {' '.join(s['command']) if s['command'] else '?'}")
    files = sorted(p.name for p in Path(s["dir"]).iterdir() if p.is_file())
    print(f"files     {', '.join(files)}")
    missing = [n for n, ok in (("job.json", s["has_job"]), ("status.json", s["has_status"])) if not ok]
    if missing:
        print(f"missing   {', '.join(missing)}  (the run died before the harness wrote them)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] in ("--version",):
        from . import __version__

        print(__version__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "status":
        return cmd_status(rest)
    if cmd == "inspect":
        return cmd_inspect(rest)
    if cmd == "tele":
        from .telemetry.cli import main as tele_main

        return tele_main(rest)
    print(f"research: unknown command {cmd!r}\n\n{__doc__.strip()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
