"""The Commit's PoUB session (``--remote-verifier HOST:PORT``, `store.remote.CommitSession`), from its
``--remote-*`` options to its lines in the Commit's record: the audited run's row entry, the verdict's
``remote_session`` and the Commit's last line, whose rejection fails the Commit (`pipeline.commit.main`)."""
from __future__ import annotations

from typing import Any

from verity_vllm.protocols.remote_bandwidth import remote


def commit_session(opts: Any, out: Any, kv_pages: int | None) -> remote.CommitSession:
    """The session of ``opts``' ``--remote-*`` options (`pipeline.commit.Options`'s, or serve-openai's
    `serve_remote.RemoteOptions`), its KV pool at ``kv_pages`` pages (None: the engine's own) and its record under ``out``."""
    backend_of = None
    if opts.remote_session_lib:
        from verity_vllm.protocols.remote_bandwidth.in_place import InPlaceBackend
        backend_of = InPlaceBackend.for_commit(opts.remote_session_lib)
    return remote.CommitSession(opts.remote_verifier, plan=opts.remote_plan, kv_pages=kv_pages,
                                weight_blocks=opts.remote_weight_blocks, kv_lib=opts.remote_kv_lib, backend_of=backend_of,
                                out=out, step_ms=opts.remote_step_ms, filler_gib=opts.remote_filler_gib,
                                log=lambda m: print(m, flush=True))


def session_of(args: Any) -> remote.CommitSession | None:
    """The Commit's session from its options (`pipeline.commit.Options`), or None without ``--remote-verifier``; refused
    before anything runs where the Commit has no instrumented run to audit."""
    if not args.remote_verifier:
        return None
    rs = commit_session(args, args.out, args.remote_kv_pages or None)
    if args.only_arm == "control" or args.pairs < 1:
        raise SystemExit("--remote-verifier audits the Commit's first instrumented run: not without one (--only-arm control, --pairs 0)")
    return rs


def end_run(rs: remote.CommitSession, pair: int, arm: str) -> dict[str, Any]:
    """After the run the session audited: `CommitSession.end`, logged, and the run's row entry."""
    rec, lock = rs.end(), rs.lockstep
    rs.log(f"[delta] remote session over pair {pair}'s {arm} run: {lock.audited} of its {lock.audited + lock.unaudited} "
           f"engine steps a pass each, {'ACCEPTED' if rec['accepted'] else 'REJECTED'}" + (f" ({rec['error']})" if rec["error"] else ""))
    return {"audited_steps": lock.audited, "unaudited_steps": lock.unaudited, "accepted": rec["accepted"]}


def into_verdict(rs: remote.CommitSession | None, verdict: dict[str, Any]) -> None:
    if rs is not None and rs.record is not None:
        verdict["remote_session"] = {k: rs.record.get(k) for k in ("verifier", "plan", "accepted", "error", "encode", "memory_map", "run")}


def accepted(rs: remote.CommitSession | None) -> bool:
    """The Commit's last line on its session, after its verdict: whether the session accepted (True if none ran)."""
    rec = None if rs is None else rs.record
    if rec is None:
        return True
    rs.log(f"[delta] REMOTE SESSION {'ACCEPTED' if rec['accepted'] else 'REJECTED'} ({rec['plan']} at {rec['verifier']}; "
           f"map agrees {(rec['memory_map'] or {}).get('agrees')}): {rs.out / 'remote_session.json'}")
    return bool(rec["accepted"])
