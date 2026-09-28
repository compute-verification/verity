"""`verity-vllm commit-group SPEC`: the Commits of several rows of one engine in one process, under one GPU lease (a grouped Commit).

Each row's Commit is the one its row would run alone (`row_group` writes SPEC from each row's own command line and environment): this
process runs `commit.main` once per row, in order, in that row's environment (with what the engine's build wrote into the process's,
`row_environ`) and with its output in that row's ``commit.log``, as the hot worker runs a job (`hot.serve`).  The first row's Commit takes the lease and builds the engine; the later rows' keep the lease
(`gpu_lease.acquire`) and reuse the engine when it is exactly the one they would build (`commit.main`: HOT-KEY-MISMATCH otherwise),
and the pinned pool, the weights root and the live weight digests (`group_reuse`); only the last row's last run releases the lease.

A row is refused by name, and runs no Commit here, when its Commit is one the hot worker refuses (`commit.hot_unsafe_reasons`), a
negative test that changes the served weights (``--flip-weight``), a Commit whose replay is not deferred (``--replay-deferred``: its
replay would run here, on the engine), a Commit into another row's output, or one whose engine (the workload's engine blocks, the
checkpoints, the engine arguments, the committer and the native extensions it loads) differs from the first row's.  Between two rows
every committer of the earlier one is closed and its device memory dropped, and the engine must be as it was built (`hot.serve`'s
checks, and no patch left that was not live before the first committer); one that is not refuses every later row by name.

``SPEC`` (``verity-vllm/commit-group/v1``): ``{"lead", "result", "rows": [{"row", "dir", "argv", "env", "log"}]}`` -- each row's
Commit arguments (``verity-vllm commit ARGV``), the environment keys its own Commit has that differ from this process's (``env``),
and its log.  ``result`` (``verity-vllm/commit-group-result/v1``) is rewritten after every row: per row its exit status, whether it
was refused and why, its times, what it reused and the checks after it.  Exit 0 once every row has its result.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.config import UsageError, environ_snapshot, option

SCHEMA = "verity-vllm/commit-group/v1"
RESULT_SCHEMA = "verity-vllm/commit-group-result/v1"
RC_REFUSED = 70                   # as the hot worker's: the row's Commit did not run here
REFUSED = ("HOT-UNSAFE", "HOT-KEY-MISMATCH")
UNSAFE_FLAGS = (("flip_weight", "--flip-weight (changes the served weights the later rows would share)"),)


@dataclass(frozen=True, kw_only=True)
class Options:
    spec: str = option("spec", help="the group's spec (verity-vllm/commit-group/v1; written by verity_vllm.pipeline.row_group)")
    environ: Mapping[str, str] = environ_snapshot()


def load(path: str) -> dict[str, Any]:
    with open(path) as f:
        spec = json.load(f)
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise UsageError(f"{path}: not a {SCHEMA} spec")
    rows = spec.get("rows")
    if not isinstance(rows, list) or not rows:
        raise UsageError(f"{path}: no rows")
    for r in rows:
        if not (isinstance(r, dict) and isinstance(r.get("row"), str) and isinstance(r.get("argv"), list) and isinstance(r.get("env"), dict)
                and isinstance(r.get("log"), str)):
            raise UsageError(f"{path}: a row needs row, argv, env and log: {str(r)[:200]}")
    if not isinstance(spec.get("result"), str):
        raise UsageError(f"{path}: no result path")
    return spec


def engine_key(args: Any, identity: Callable[[list[str]], dict[str, Any]], argv: list[str]) -> dict[str, Any]:
    """What a row's engine and the native code its Commit loads are made of: the hot worker's engine identity (`hot.engine_identity`)
    but the workload's row id and path, and the committer, the attention tap and the other taps' extensions."""
    ident = identity(argv)
    ident.pop("workload_path", None)
    ident["workload"] = {k: v for k, v in (ident.get("workload") or {}).items() if k != "row"}
    taps = getattr(args, "tap_sources", None)
    ident.update(committer=args.committer, hidden_so=args.hidden_so, hidden_ns=args.hidden_ns,
                 taps=taps.given() if hasattr(taps, "given") else None)
    return ident


def _differs(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out += [f"{k}.{s}" for s in _differs(a[k], b[k])]
        elif a.get(k) != b.get(k):
            out.append(k)
    return out


def refusals(rows: list[dict[str, Any]], parse: Callable[[list[str]], Any], unsafe: Callable[[Any], list[str]],
             key: Callable[[Any, list[str]], dict[str, Any]], enter: Callable[[dict[str, Any]], None]) -> dict[str, str]:
    """Why each row this process won't commit is refused (by row id), each row read in its own environment (``enter``)."""
    why: dict[str, str] = {}
    first: tuple[str, dict[str, Any]] | None = None
    outs: dict[str, str] = {}
    for r in rows:
        enter(r)
        try:
            a = parse(list(r["argv"]))
        except SystemExit as e:
            why[r["row"]] = f"its Commit's command line does not parse (exit {e.code})"
            continue
        reasons = [*unsafe(a), *(text for f, text in UNSAFE_FLAGS if getattr(a, f, None))]
        if not getattr(a, "replay_deferred", None):
            reasons.append("its replay is not deferred (no --replay-deferred): it would replay on the engine, inside the group's lease")
        if a.out in outs:
            reasons.append(f"it commits into {a.out}, row {outs[a.out]}'s output")
        if reasons:
            why[r["row"]] = "; ".join(reasons)
            continue
        outs[a.out] = r["row"]
        k = key(a, list(r["argv"]))
        if first is None:
            first = (r["row"], k)
        elif k != first[1]:
            why[r["row"]] = f"its engine differs from row {first[0]}'s ({', '.join(_differs(k, first[1]))})"
    return why


def row_environ(env0: Mapping[str, str], r: dict[str, Any], eng: Any) -> dict[str, str]:
    """A row's process environment: this process's, the row's own keys, then what the engine's build wrote (``Engine.environ``), as
    a row that built the engine in its own process would have it."""
    return {**env0, **r["env"], **(getattr(eng, "environ", None) or {})}


def _result_of(r: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"row": r["row"], "dir": r.get("dir"), "log": r["log"], **fields}


def _write(path: str, doc: dict[str, Any]) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def run_row(r: dict[str, Any], commit: Callable[[list[str]], None]) -> tuple[int, str | None]:
    """``commit(argv)``, the row's Commit, with this process's output in the row's log -> (its exit status, why it was refused)."""
    fd_out, fd_err = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    rc, refused = 0, None
    with open(r["log"], "ab", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            commit(list(r["argv"]))
        except SystemExit as e:
            code = e.code
            if isinstance(code, str):
                print(code, file=sys.stderr, flush=True)
                refused = code if code.startswith(REFUSED) else None
                rc = RC_REFUSED if refused else 1
            else:
                rc = int(code or 0)
        except BaseException:  # noqa: BLE001 -- a crashed Commit is its row's exit 1, as in its own process; the next rows go on
            traceback.print_exc()
            rc = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(fd_out, 1)
            os.dup2(fd_err, 2)
            os.close(fd_out)
            os.close(fd_err)
    return rc, refused


def run(spec: dict[str, Any], *, enter: Callable[[dict[str, Any]], None], parse: Callable[[list[str]], Any],
        unsafe: Callable[[Any], list[str]], key: Callable[[Any, list[str]], dict[str, Any]], commit: Callable[[list[str]], None],
        hot: dict[str, Any], clean: Callable[[], dict[str, Any]], clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """The group's rows in order (`refusals`, then each row's Commit and the checks after it) -> the result document. ``hot`` is
    `commit._HOT`: its ``jobs`` names the row running, ``last`` marks the last row to run, and ``group`` collects `group_reuse`'s
    events."""
    rows = spec["rows"]
    doc: dict[str, Any] = {"schema": RESULT_SCHEMA, "lead": spec.get("lead"), "started": clock(), "rows": {}}
    why = refusals(rows, parse, unsafe, key, enter)
    for r in rows:
        if r["row"] in why:
            doc["rows"][r["row"]] = _result_of(r, rc=RC_REFUSED, refused=f"REFUSED by the grouped Commit: {why[r['row']]}")
    todo = [r for r in rows if r["row"] not in why]
    _write(spec["result"], doc)
    for i, r in enumerate(todo):
        enter(r)
        hot.update(jobs=i + 1, coms=[], last=i == len(todo) - 1)
        t0, built = clock(), hot.get("eng") is None
        rc, refused = run_row(r, commit)
        t1 = clock()
        res = _result_of(r, rc=rc, refused=refused, job=i + 1, started=t0, ended=t1, wall_s=round(t1 - t0, 3),
                         engine_built=built and hot.get("eng") is not None, engine_reused=not built and refused is None,
                         reuse=[e for e in hot["group"]["events"] if e.get("job") == i + 1],
                         pinned_pool_bytes=getattr(hot["group"].get("pool"), "bytes_total", None))
        doc["rows"][r["row"]] = res
        if i < len(todo) - 1:
            res["after"] = after = clean()
            if not after["ok"]:
                for rest in todo[i + 1:]:
                    doc["rows"][rest["row"]] = _result_of(rest, rc=RC_REFUSED, refused=f"REFUSED by the grouped Commit: after row {r['row']} "
                                                                                       f"the engine is not as it was built ({after['why']})")
                _write(spec["result"], doc)
                break
        _write(spec["result"], doc)
    doc["ended"] = clock()
    _write(spec["result"], doc)
    return doc


def clean(cd: Any, torch: Any, release: Callable[[Any], list[str]], live: Callable[[], tuple[Any, ...]]) -> dict[str, Any]:
    """After a row that is not the last: its replay bundles written, every committer it built closed and its device memory dropped
    (``release``, `hot.release_committer_device`), and the engine as it was built (`hot.serve`'s post-job checks; ``live``, the
    patches, holds none that was not live before the first committer) -> ``{"ok", "why", ...}``."""
    t0 = time.perf_counter()
    for t in threading.enumerate():
        if t.name.startswith("replay-bundle") and t is not threading.current_thread():
            t.join()
    hot = cd._HOT
    coms, errors, released = hot.pop("coms", []) or [], [], []
    for com in coms:
        for meth in ("uninstall", "close"):
            try:
                getattr(com, meth)()
            except Exception as e:  # noqa: BLE001 -- recorded; the checks below decide
                errors.append(f"{type(com).__name__}.{meth}: {e!r}"[:300])
        released += release(com)
    n = len(coms)
    del coms
    gc.collect()
    out: dict[str, Any] = {"committers_closed": n, "close_errors": errors, "released": sorted(set(released)), "ok": True, "why": None}
    eng = hot.get("eng")
    reasons = ["a committer's worker thread outlived its close()"] if "SKIPPED:cmt-worker-alive" in released else []
    if eng is not None:
        torch.cuda.synchronize()
        out["unfinished_requests"] = bool(eng.llm.llm_engine.has_unfinished_requests())
        torch.cuda.empty_cache()
        if out["unfinished_requests"]:
            reasons.append("the engine has unfinished requests")
        try:
            cd.assert_pristine(eng, hot["base0"])
        except AssertionError as e:
            reasons.append(str(e))
        if hot.get("cuda0") is not None:
            out["cuda_residue"] = cd.cuda_residue(hot["cuda0"], cd.cuda_census())
    base = (hot.get("group") or {}).get("patches") or ()
    left = [e for e in live() if not any(e is b for b in base)]
    if left:
        reasons.append(f"{len(left)} patch(es) left by the row: {[f'{type(e).__name__}:{getattr(e, 'name', '?')}' for e in left][:6]}")
    out.update(ok=not reasons, why="; ".join(reasons) or None, s=round(time.perf_counter() - t0, 3))
    return out


def main(o: Options) -> int:
    import torch

    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.engine import hooks
    from verity_vllm.pipeline import cli, commit as cd, gpu_lease as GL, hot as HC

    spec = load(o.spec)
    env0 = dict(o.environ)

    def enter(r: dict[str, Any]) -> None:
        cli.set_environment(row_environ(env0, r, (cd._HOT or {}).get("eng")), keep=GL.LEASED)
        prof.apply_env()                   # the pins a row's own Commit process has from its import of `commit`

    def commit(argv: list[str]) -> None:
        sys.argv[:] = [cd.__file__, *argv]
        cd.main(cli.parse("commit", argv))

    cd._HOT = {"jobs": 0, "coms": [], "last": False, "restart_reason": f"grouped Commit of {len(spec['rows'])} rows (commit_group, lead {spec.get('lead')})",
               "group": {"pool": None, "weights": None, "live": None, "patches": None, "events": []}}
    doc = run(spec, enter=enter, parse=lambda argv: cli.parse("commit", argv), unsafe=cd.hot_unsafe_reasons,
              key=lambda a, argv: engine_key(a, HC.engine_identity, argv), commit=commit, hot=cd._HOT,
              clean=lambda: clean(cd, torch, HC.release_committer_device, hooks.live))
    print(json.dumps({r: {"rc": d["rc"], "refused": d.get("refused"), "wall_s": d.get("wall_s")} for r, d in doc["rows"].items()}), flush=True)
    return 0
