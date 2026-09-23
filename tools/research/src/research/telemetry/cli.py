"""`research tele <cmd>` / `python3 -m research.telemetry <cmd>` -- the shell-callable surface.

  init          write job.json for an attempt dir            status        set status.json state
  event         spans / marks / holds / queue timeline       sample        Level-0 sidecar
  cancel-intent cancellation marker (+ optional delivery)    classify      failure.json
  run           reference harness (sampler + spans + timeout markers + classify)
  set-timeout   raise/lower/clear a RUNNING attempt's stage timeout (control/timeout_s; the run loop records a mark)
  render-rules  policy.toml -> RULES.md                      check-rules   verify a RULES.md is the render of a policy
  read          dump merged events / span accounting / holds of an attempt dir (for fixtures and humans)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import HOLD_REASONS, STATES
from .events import Emitter, append, events_path, holds, queue_timeline, read_all, spans


def _kv(items: list[str]) -> dict:
    out = {}
    for x in items:
        k, _, v = x.partition("=")
        if v in ("true", "false", "null"):
            out[k] = {"true": True, "false": False, "null": None}[v]
        else:
            try:
                out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                out[k] = v
    return out


def cmd_init(argv: list[str]) -> int:
    from .job import build_job, input_ref, sha256_file, write_json_atomic
    ap = argparse.ArgumentParser(prog="tele init")
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--generation", type=int, default=1)
    ap.add_argument("--tree-sha", default=None)
    ap.add_argument("--release-tag", default=None)
    ap.add_argument("--workload-digest", default=None)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--profile-key", action="append", default=[])
    ap.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--declared-output", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--reservation-gpu", default=None)
    ap.add_argument("--reservation-host-memory-bytes", type=int, default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--policy-version", default=None)
    ap.add_argument("--source-identity", default=None, help="source_identity.json written by the runner's precheck (F-r17-tp-06); embedded in job.json computation.source_identity")
    ap.add_argument("--cadence", type=float, default=1.0)
    ap.add_argument("--pss-every", type=float, default=5.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    p = Path(a.attempt_dir) / "job.json"
    if p.exists() and not a.force:
        print(f"{p} exists (use --force to rewrite)", file=sys.stderr)
        return 1
    pv, ps = a.policy_version, None
    if a.policy and Path(a.policy).exists():
        ps = sha256_file(a.policy)[0]
        if not pv:
            import tomllib
            pv = tomllib.loads(Path(a.policy).read_text()).get("policy", {}).get("version")
    doc = build_job(project=a.project, task=a.task, job=a.job, attempt=a.attempt, lane=a.lane, generation=a.generation, tree_sha=a.tree_sha,
                    release_tag=a.release_tag, workload_digest=a.workload_digest, stage=a.stage, inputs=[input_ref(x) for x in a.input],
                    profile={k: os.environ.get(k) for k in a.profile_key if k in os.environ}, command=a.command, declared_outputs=a.declared_output,
                    timeout_s=a.timeout, reservation={"gpu": a.reservation_gpu, "host_memory_bytes": a.reservation_host_memory_bytes},
                    policy_version=pv, policy_sha256=ps, instruments=["cgroup", "process", "pss", "gpu", "storage"], cadence_s=a.cadence, pss_every_s=a.pss_every,
                    source_identity=(json.loads(Path(a.source_identity).read_text()) if a.source_identity and Path(a.source_identity).exists() else None))
    Path(a.attempt_dir).mkdir(parents=True, exist_ok=True)
    write_json_atomic(p, doc)
    print(p)
    return 0


def cmd_status(argv: list[str]) -> int:
    from .job import set_status
    ap = argparse.ArgumentParser(prog="tele status")
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("state", choices=STATES)
    ap.add_argument("kv", nargs="*")
    a = ap.parse_args(argv)
    set_status(a.attempt_dir, a.state, **_kv(a.kv))
    return 0


def cmd_event(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tele event", description="append one event record")
    ap.add_argument("--attempt-dir", default=os.environ.get("TELE_ATTEMPT_DIR"), help="default $TELE_ATTEMPT_DIR (exported by the runner)")
    ap.add_argument("what", choices=["submitted", "admitted", "started", "span-start", "span-end", "mark", "hold-start", "hold-end"])
    ap.add_argument("--stage", default=None)
    ap.add_argument("--id", default=None, help="span/hold id (span-start prints the generated id when omitted)")
    ap.add_argument("--parent", default=None, help="parent span id (default $TELE_PARENT_SPAN)")
    ap.add_argument("--name", default=None, help="mark name")
    ap.add_argument("--reason", default=None, help=f"hold reason: {'|'.join(HOLD_REASONS)}")
    ap.add_argument("--detail", default=None)
    ap.add_argument("--outcome", default="ok", choices=["ok", "fail", "interrupted"])
    ap.add_argument("--rc", type=int, default=None)
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--proc", default=None, help="process label (default: the calling program's name)")
    ap.add_argument("--stages-txt", default=None, help="also append a `<stage> START|END <utc> ...` line to this file")
    ap.add_argument("kv", nargs="*", help="extra k=v fields")
    # [R18 infra, r18-int-2; arch M-0025] `tele event submitted --stage S row=.. role=..` puts the k=v positionals AFTER the options:
    # Python <= 3.11 argparse.parse_args rejects them ("unrecognized arguments"), 3.12 accepts; the pods run 3.11 (system python3) and
    # 3.12 (venv-py312) side by side.  parse_intermixed_args accepts the shape on every version the pods carry.
    a = ap.parse_intermixed_args(argv)
    if not a.attempt_dir:
        ap.error("--attempt-dir required (or export TELE_ATTEMPT_DIR)")
    em = Emitter(a.attempt_dir, proc=a.proc or os.environ.get("TELEMETRY_PROC") or "shell")
    kv = _kv(a.kv)
    sid = a.id
    if a.what in ("submitted", "admitted", "started"):
        em.queue(a.what, stage=a.stage, **kv)
    elif a.what == "span-start":
        if not a.stage:
            ap.error("--stage required")
        sid = em.span_start(a.stage, parent=a.parent, span_id=a.id, **kv)
        print(sid)
    elif a.what == "span-end":
        if not a.stage or not a.id:
            ap.error("--stage and --id required")
        em.span_end(a.stage, a.id, outcome=a.outcome, rc=a.rc, refs=a.ref, **kv)
    elif a.what == "mark":
        em.mark(a.stage or "", a.name or "mark", **kv)
    elif a.what == "hold-start":
        if not a.reason:
            ap.error("--reason required")
        print(em.hold_start(a.reason, detail=a.detail, hold_id=a.id, **kv))
    elif a.what == "hold-end":
        if not a.id:
            ap.error("--id required")
        em.hold_end(a.id, reason=a.reason, **kv)
    if a.stages_txt and a.what in ("span-start", "span-end") and a.stage:
        import time
        with open(a.stages_txt, "a") as f:
            f.write(f"{a.stage} {'START' if a.what == 'span-start' else 'END'} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                    f"{'rc=' + str(a.rc) + ' ' if a.rc is not None else ''}id={sid or ''}\n")
    return 0


def cmd_read(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tele read")
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--what", choices=["events", "spans", "holds", "queue", "resources", "all"], default="all")
    a = ap.parse_args(argv)
    rep = read_all(a.attempt_dir, "events")
    out = {}
    if a.what in ("events", "all"):
        out["events"] = rep.records
    if a.what in ("spans", "all"):
        out["spans"] = spans(rep.records)
    if a.what in ("holds", "all"):
        out["holds"] = holds(rep.records)
    if a.what in ("queue", "all"):
        out["queue"] = queue_timeline(rep.records)
    if a.what in ("resources", "all"):
        r2 = read_all(a.attempt_dir, "resources")
        out["resources"] = {"count": len(r2.records), "kinds": sorted({r.get("kind") for r in r2.records}), "partial": r2.partial}
    out["partial"] = rep.partial
    out["malformed"] = rep.malformed
    print(json.dumps(out, indent=1, default=str))
    return 0


def cmd_hash_inputs(argv: list[str]) -> int:
    """complete the deferred input hashes in job.json (the preserve.verify step)"""
    from .job import hash_inputs
    ap = argparse.ArgumentParser(prog="tele hash-inputs", description=cmd_hash_inputs.__doc__)
    ap.add_argument("--attempt-dir", default=os.environ.get("TELE_ATTEMPT_DIR"))
    a = ap.parse_args(argv)
    if not a.attempt_dir:
        ap.error("--attempt-dir required")
    done = hash_inputs(a.attempt_dir)
    for r in done:
        print(f"{r['name']}: sha256={r['sha256']} changed_since_launch={r['stat_changed_since_launch']}")
    print(f"{len(done)} input(s) hashed")
    return 0


COMMANDS = {
    "init": cmd_init, "status": cmd_status, "event": cmd_event, "read": cmd_read, "hash-inputs": cmd_hash_inputs,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "sample":
        from .sample import main as m
        return m(rest)
    if cmd == "cancel-intent":
        from .cancel import main as m
        return m(rest)
    if cmd == "classify":
        from .classify import main as m
        return m(rest)
    if cmd == "run":
        from .run import main as m
        return m(rest)
    if cmd == "set-timeout":
        from .run import set_timeout_main as m
        return m(rest)
    if cmd in ("render-rules", "check-rules"):
        from .policy import main as m
        return m([cmd] + rest)
    if cmd in COMMANDS:
        return COMMANDS[cmd](rest)
    print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
    return 2
