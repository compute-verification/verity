"""The Commit's PoUB session (``--remote-verifier HOST:PORT``), from its ``--remote-*`` options to its lines in the Commit's
record: the audited run's row entry, the verdict's ``poub_session`` and the Commit's last line, whose rejection fails the
Commit (`pipeline.commit.main`).  The session is the process registered as `SESSION` (`scheme.PROCESSES`, once the one
declaration that states it is imported by that name, `compose.schemes.named`), so this package imports no PoUB code: it hands the session the exchange of the build about to be
made (`messages.current`), on which the session attaches its scheme, and drives what it gets back."""
from __future__ import annotations

from typing import Any

from verity.protocols.compose import schemes as declarations
from verity.protocols.resource_accounting import scheme

from verity_vllm import messages

#: the process a PoUB scheme starts beside the Commit's engine
SESSION = "poub-session"


def commit_session(opts: Any, out: Any, kv_pages: int | None) -> Any:
    """The session (`SESSION`) of ``opts``' ``--remote-*`` options (`pipeline.commit.Options`'s, or serve-openai's
    `serve_poub.RemoteOptions`), its KV pool at ``kv_pages`` pages (None: the engine's own) and its record under ``out``;
    `NotImplementedError` until a scheme registers it."""
    declarations.get(declarations.named("processes", SESSION))
    if (start := scheme.PROCESSES.get(SESSION)) is None:
        raise NotImplementedError(f"vllm/adapter-imports-no-protocol: memory-accounting registers the Commit's PoUB session "
                                  f"as process {SESSION!r} (Scheme.processes); no scheme has")
    return start(opts.remote_verifier, exchange=messages.current, plan=opts.remote_plan, kv_pages=kv_pages,
                 weight_blocks=opts.remote_weight_blocks, kv_lib=opts.remote_kv_lib,
                 out=out, node=opts.remote_node, step_ms=opts.remote_step_ms,
                 filler_gib=opts.remote_filler_gib, log=lambda m: print(m, flush=True))


def session_of(args: Any) -> Any | None:
    """The Commit's session from its options (`pipeline.commit.Options`), or None without ``--remote-verifier``; refused
    before anything runs where the Commit has no instrumented run to audit."""
    if not args.remote_verifier:
        return None
    rs = commit_session(args, args.out, args.remote_kv_pages or None)
    if args.only_arm == "control" or args.pairs < 1:
        raise SystemExit("--remote-verifier audits the Commit's first instrumented run: not without one (--only-arm control, --pairs 0)")
    return rs


def end_run(rs: Any, pair: int, arm: str) -> dict[str, Any]:
    """After the run the session audited: `CommitSession.end`, logged, and the run's row entry."""
    rec, lock = rs.end(), rs.lockstep
    rs.log(f"[delta] remote session over pair {pair}'s {arm} run: {lock.audited} of its {lock.audited + lock.unaudited} "
           f"engine steps a pass each, {'ACCEPTED' if rec['accepted'] else 'REJECTED'}" + (f" ({rec['error']})" if rec["error"] else ""))
    return {"audited_steps": lock.audited, "unaudited_steps": lock.unaudited, "accepted": rec["accepted"]}


def into_verdict(rs: Any | None, verdict: dict[str, Any]) -> None:
    if rs is not None and rs.record is not None:
        verdict["remote_session"] = {k: rs.record.get(k) for k in ("verifier", "plan", "accepted", "error", "encode", "memory_map", "run")}


def accepted(rs: Any | None) -> bool:
    """The Commit's last line on its session, after its verdict: whether the session accepted (True if none ran)."""
    rec = None if rs is None else rs.record
    if rec is None:
        return True
    rs.log(f"[delta] REMOTE SESSION {'ACCEPTED' if rec['accepted'] else 'REJECTED'} ({rec['plan']} at {rec['verifier']}; "
           f"map agrees {(rec['memory_map'] or {}).get('agrees')}): {rs.out / 'remote_session.json'}")
    return bool(rec["accepted"])
