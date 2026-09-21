"""`research` command line.

  research run [--project P] [--stage S] [--timeout T] [--expect K=V ...] [--require-result] [--runs-dir D] [harness flags] -- CMD...
  research status [--runs-dir D] [--all] [--json]
  research inspect <run-id|prefix|path> [--runs-dir D] [--json]
  research tele <telemetry subcommand...>        # the full harness surface (init, event, sample, cancel-intent, classify, ...)

`run` allocates a run id, writes the v0.2 job record, gives the workload RESEARCH_RUN_ID / RESEARCH_RUN_DIR, and hands execution
to the telemetry harness (timeout, sampler, cancel evidence, classifier, tee'd logs).  It exits with the workload's status; with
--require-result, a zero exit and a missing or invalid result.json exits 65 instead.  Harness flags pass through unchanged
(--tree-sha, --workload-digest, --input name=path, --declared-output, --policy, --no-sampler, ...).
`--on <machine>` (remote execution) is not implemented yet and says so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import record, result
from . import runs as R

EX_DATAERR = 65


def _harness_parser(prog: str) -> argparse.ArgumentParser:
    from .telemetry.run import add_args

    ap = argparse.ArgumentParser(prog=prog, add_help=False, allow_abbrev=False)
    add_args(ap)
    return ap


def build_job_from(a: argparse.Namespace, command: list[str]) -> dict:
    """Mirror of the job the harness would build for this namespace (telemetry.run.run), so that pre-writing job.json changes
    nothing the harness records.  Kept in one place; if the harness's call changes upstream, change this."""
    from .telemetry.job import build_job, input_ref, sha256_file

    policy_sha = sha256_file(a.policy)[0] if a.policy and Path(a.policy).exists() else None
    policy_version = a.policy_version
    if a.policy and Path(a.policy).exists() and not policy_version:
        try:
            import tomllib

            policy_version = tomllib.loads(Path(a.policy).read_text()).get("policy", {}).get("version")
        except Exception:  # noqa: BLE001
            policy_version = None
    return build_job(project=a.project, task=a.task, job=a.job, attempt=a.attempt, lane=a.lane, generation=a.generation, tree_sha=a.tree_sha,
                     release_tag=a.release_tag, workload_digest=a.workload_digest, stage=a.stage, inputs=[input_ref(x) for x in a.input],
                     profile={k: os.environ.get(k) for k in a.profile_key if k in os.environ}, command=command, declared_outputs=a.declared_output,
                     timeout_s=a.timeout, reservation={"gpu": a.reservation_gpu, "host_memory_bytes": a.reservation_host_memory_bytes},
                     policy_version=policy_version, policy_sha256=policy_sha,
                     instruments=["cgroup", "process", "pss", "gpu", "storage"] if not a.no_sampler else [], cadence_s=a.cadence, pss_every_s=a.pss_every)


def _parse_expect(items: list[str]) -> dict | None:
    if not items:
        return None
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"research run: --expect wants KEY=VALUE, got {it!r}")
        out[k] = v
    return out


def cmd_run(argv: list[str]) -> int:
    if "--" not in argv:
        print("research run: the workload command goes after `--`", file=sys.stderr)
        return 2
    i = argv.index("--")
    opts, command = argv[:i], argv[i + 1 :]
    if not command:
        print("research run: empty command after `--`", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="research run", add_help=True, allow_abbrev=False, description=__doc__.split("\n\n")[1])
    ap.add_argument("--runs-dir", default=None, help="registry root (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--on", default=None, metavar="MACHINE", help="run on an existing authorized machine (not implemented yet)")
    ap.add_argument("--id", default=None, help="use this run id instead of allocating one")
    ap.add_argument("--expect", action="append", default=[], metavar="K=V", help="expected workload fingerprint entry (recorded separately from what the workload observes)")
    ap.add_argument("--require-result", action="store_true", help="a zero exit without a valid result.json exits 65")
    ap.add_argument("--stage", default="run", help="stage name recorded in the job (default: run)")
    ns, harness_opts = ap.parse_known_args(opts)
    if ns.on:
        print(f"research run --on {ns.on}: remote execution is not implemented yet (tools/research/README.md)", file=sys.stderr)
        return 2
    root = R.runs_root(ns.runs_dir)
    run_id = ns.id or R.new_run_id()
    run_dir = root / run_id
    if run_dir.exists():
        print(f"research run: {run_dir} already exists; a rerun gets a fresh id", file=sys.stderr)
        return 2
    hp = _harness_parser("research run")
    try:
        a = hp.parse_args(["--attempt-dir", str(run_dir), "--stage", ns.stage, "--job", run_id, *harness_opts])
    except SystemExit:
        return 2
    run_dir.mkdir(parents=True)
    job = record.extend_job(build_job_from(a, command), run_id=run_id, runs_root=root, expected_fingerprint=_parse_expect(ns.expect))
    from .telemetry.job import write_json_atomic
    from .telemetry.run import run as harness_run

    write_json_atomic(run_dir / "job.json", job)
    os.environ.update(result.env_for(run_id, run_dir))
    print(f"research: run {run_id} -> {run_dir}", file=sys.stderr)
    rc = harness_run(a, command)
    s = R.summarize(run_dir)
    line = f"research: run {run_id} {s['state']} rc={s['rc']} class={s['class'] or '-'} result={s['result']}"
    if s["validation"]:
        line += f" validation={s['validation']}"
    print(line + f"  (research inspect {run_id})", file=sys.stderr)
    if ns.require_result and rc == 0 and not s["result_valid"]:
        print(f"research: --require-result: the workload exited 0 but result.json is {s['result']}; exiting {EX_DATAERR}", file=sys.stderr)
        return EX_DATAERR
    return rc


def _fmt(s: dict) -> str:
    cmd = " ".join(s["command"]).replace("\n", " ") if s.get("command") else "?"
    if len(cmd) > 50:
        cmd = cmd[:47] + "..."
    rc = "" if s["rc"] is None else str(s["rc"])
    res = "-" if not s["result_present"] else ("ok" if s["result_valid"] else "BAD")
    val = s["validation"] or "-"
    return f"{s['id']:<24} {str(s['state']):<10} {rc:>3}  {(s['class'] or ''):<22} {res:<4} {val:<8} {str(s['project']):<12} {str(s['stage']):<16} {cmd}"


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
    print(f"{'run':<24} {'state':<10} {'rc':>3}  {'class':<22} {'res':<4} {'valid':<8} {'project':<12} {'stage':<16} command")
    for s in rows:
        print(_fmt(s))
    active = sum(1 for s in rows if s["state"] in R.ACTIVE)
    print(f"\n{len(rows)} shown ({active} active) under {root}   res: result.json ok/BAD/-absent   valid: the workload's own checker")
    return 0


def _tool_line(tool: dict | None) -> str:
    if tool is None:
        return "unknown (v0.1 record: the harness alone wrote it)"
    sha = tool.get("research_git_sha")
    state = "" if tool.get("research_dirty") is None else (" (dirty)" if tool.get("research_dirty") else " (clean)")
    return f"research {tool.get('research_version')} @ {(sha or 'no git sha')[:12]}{state}; classifier {tool.get('classifier_version')}"


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
    print(f"run       {s['run_id']}   ({s['schema'] or 'no job.json'})")
    print(f"dir       {s['dir']}")
    if not s["readable"]:
        print(f"WARNING   job.json schema {s['schema']!r} is not one this research can read; fields below are best effort")
    print(f"exited    {s['state']}  rc={s['rc']} signal={s['signal']} wall_s={s['wall_s']}")
    if s["class"]:
        print(f"class     {s['class']} ({s['certainty']})" + (f": {s['why']}" if s["why"] else ""))
    print(f"result    {s['result']}")
    print(f"validated {s['validation'] or 'no validation reported'}  (the workload's checker; research does not verify proofs)")
    pres = s["preserved"]
    print(f"preserved {'yes: ' + str(pres.get('destination')) if pres else 'no preservation manifest'}")
    print(f"project   {s['project']}  lane={s['lane']}  stage={s['stage']}")
    print(f"source    tree_sha={s['tree_sha']}  workload_digest={s['workload_digest']}")
    if s["fingerprint_expected"] is not None or s["fingerprint_observed"] is not None:
        print(f"expected  {json.dumps(s['fingerprint_expected'], sort_keys=True)}")
        print(f"observed  {json.dumps(s['fingerprint_observed'], sort_keys=True)}")
    if s["measurements"]:
        print("measured  " + "; ".join(f"{m['name']}={m['value']} {m['unit']}" for m in s["measurements"][:12]))
    tool = s["tool"]
    print(f"tool      {_tool_line(s['tool'])}")
    cost = s["cost"]
    print(f"cost      {('$' + str(cost['usd']) + ' est. (' + cost['basis'] + ')') if cost.get('usd') is not None else 'unknown: ' + str(cost.get('reason'))}")
    print(f"time      started={s['started_utc']}  ended={s['ended_utc']}")
    print(f"command   {' '.join(s['command']) if s['command'] else '?'}")
    if s["remote"]:
        print(f"remote    {json.dumps(s['remote'], sort_keys=True)}")
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
    if argv[0] == "--version":
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
