"""A TP Commit's serving rows (``tp-commit --serving-rows REQUEST_JSON``, the request `pipeline.serving_rows` reads): one ``served``
window per rank in ``<dir>/rank<r>/``, which the rank's own process commits (`rank_rows`, through the collective RPC) from its
committed store, its registered weight shards and its Programs of record with `pipeline.serving_rows.after_commit`, and ``<dir>/tp.json``
over the world.  The driver calls `commit` in the first instrumented run once every rank's root is final and the TP run root is fixed,
and before any draw: the challenge, the population openings and the cross-rank picks all come after it, and every rank's C2 replay is
deferred (`require`), since an in-process replay draws its seed inside finalize.

Rank ``r``'s window binds ``run = <TP run root hex>/rank<r>`` (`run_of`): the TP run root fixes every rank's vllm-v1 root, and the rank
keeps two ranks' ports, whose population, partition and leaf counts can coincide, in distinct domains.  A rank that writes no window,
or whose root is not the one the TP run root combines, exits the Commit before its challenge."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

SCHEMA = "verity-vllm/tp-serving-rows/v1"


def require(args: Any, replay_cfg: dict[str, Any] | None) -> None:
    """Refuse a ``--serving-rows`` TP Commit whose windows could not be written before every draw."""
    if args.serving_rows and (replay_cfg is None or not args.replay_deferred or args.only_arm == "control"):
        raise SystemExit("--serving-rows on a TP Commit needs --required-manifest with a --build-rank per rank (each rank reads its Programs of "
                         "record from the replay config), --replay-deferred at a world that defers (no rank draws its replay seed before "
                         "the windows) and an instrumented run")


def run_of(tp_root: bytes, rank: int) -> str:
    return f"{tp_root.hex()}/rank{int(rank)}"


def commit(llm: Any, args: Any, root: bytes, claimed: dict[str, str], world: int, root_tag: bytes) -> dict[str, Any]:
    """Every rank's window under the TP run root ``root`` (tagged ``root_tag``) of the ranks' roots ``claimed`` (rank -> hex), and
    ``tp.json``."""
    req = json.loads(Path(args.serving_rows).read_text())
    tp = {"world": world, "tp_run_root": root.hex(), "roots": {str(r): claimed[str(r)] for r in range(world)}, "root_tag": root_tag.decode()}
    got = llm.collective_rpc(rank_rows, timeout=args.rpc_timeout, args=(dict(req, row=args.label), tp))
    return gather(req["dir"], tp, got)


def gather(out_dir: str, tp: dict[str, Any], got: list[dict[str, Any]]) -> dict[str, Any]:
    """``tp.json`` over the ranks' answers; fail closed unless every rank of the world, and only those, wrote its window."""
    world, by = int(tp["world"]), {int(g["rank"]): g for g in got}
    why = [f"rank {r}: {(by.get(r) or {}).get('error') or 'no window'}" for r in range(world) if not (by.get(r) or {}).get("window")]
    why += [f"rank {r} is not a rank of the world of {world}" for r in sorted(set(by) - set(range(world)))]
    why += [f"{len(got)} answers for {len(by)} ranks"] if len(got) != len(by) else []
    root = bytes.fromhex(tp["tp_run_root"])
    doc = {"schema": SCHEMA, **tp, "runs": {str(r): run_of(root, r) for r in range(world)},
           "windows": {str(r): (by.get(r) or {}).get("window") for r in range(world)}, "result": not why, "why": "; ".join(why) or None}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "tp.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    if why:
        raise SystemExit(f"fail-closed: serving rows: {doc['why']}"[:600])
    print(f"[tp2commit] serving rows: {world} rank windows under TP run root {root.hex()[:16]}: "
          + ", ".join(f"rank {r} {doc['windows'][str(r)]['population']} units, registration {str(doc['windows'][str(r)]['registration_digest'])[:16]}"
                      for r in range(world)), flush=True)
    return doc


def rank_rows(worker: Any, request: dict[str, Any], tp: dict[str, Any]) -> dict[str, Any]:
    """On a rank (``LLM.collective_rpc(rank_rows)``): its window (`write`), or the error that kept it from one, written beside it."""
    st = worker._tp2_commit
    rank = int(st["rank"])
    try:
        return {"rank": rank, "window": write(st, request, tp)}
    except Exception as e:  # noqa: BLE001 -- the driver fails the Commit on it
        d = Path(request["dir"]) / f"rank{rank}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "error.txt").write_text(traceback.format_exc())
        return {"rank": rank, "error": repr(e)[:400]}


def write(st: dict[str, Any], request: dict[str, Any], tp: dict[str, Any]) -> dict[str, Any]:
    """Rank ``st["rank"]``'s window over its committed store opened against its root, its live weight shards and its Programs of
    record (`rank_replay.load`: refused unless of record and this rank's), from the worker's commit state."""
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.pipeline.tp import rank_replay as RR

    rank, world, rc = int(st["rank"]), int(tp["world"]), st.get("rc")
    if rc is None or not st.get("bmap"):
        raise LookupError(f"rank {rank}: no finalized run or no binding map (binding skipped or failed)")
    if rc.run_root.hex() != tp["roots"].get(str(rank)):
        raise ValueError(f"rank {rank}: its run root {rc.run_root.hex()[:16]} is not the one the TP run root combines")
    p = RR.load(st["replay"], rank, world)
    res = SRW.after_commit(str(Path(request["dir"]) / f"rank{rank}"), partition_file=request["partition"], programs=p.progs, bmap=st["bmap"],
                           committed=OC.OpenedReader(st["com"], rc), weights=SR.model_weights_provider(st["eng"].model, p.index.prefix),
                           manifest=RR.manifest_of_rank(st.get("manifest_doc"), rank), run_root=run_of(bytes.fromhex(tp["tp_run_root"]), rank),
                           run=request.get("run"), row=request.get("row"), arrivals=p.arrivals, program_dirs=p.dirs, digests=p.digs,
                           committer=request.get("committer"), workers=max(1, min(16, (os.cpu_count() or 1) // world)),
                           tp={"world": world, "rank": rank, "tp_run_root": tp["tp_run_root"], "run_root": rc.run_root.hex(), "root_tag": tp["root_tag"]},
                           index={"attention_paths": p.attn_paths, "tp": p.tp})
    return {k: res.get(k) for k in ("run", "population", "partition_digest", "registration_digest", "seconds")}
