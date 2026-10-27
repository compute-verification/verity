"""A TP Commit's serving rows (``tp-commit --serving-rows REQUEST_JSON``, the request `pipeline.serving_rows` reads): one ``served``
window per rank in ``<dir>/rank<r>/``, which the rank's own process commits (`rank_rows`, through the collective RPC) from its
committed store, its registered weight shards and its Programs of record with `pipeline.serving_rows.after_commit`, and ``<dir>/tp.json``
over the world.  The driver calls `commit` in the first instrumented run once every rank's root is final and the TP run root is fixed,
and before sampling: the challenge, the population openings and the cross-rank picks all come after it, and every rank's C2 replay is
deferred (`require`), since an in-process replay draws its seed inside finalize.

Rank ``r``'s window binds ``run = <TP run root hex>/rank<r>`` (`run_of`): the TP run root fixes every rank's rows-sha512 root, and the rank
keeps two ranks' ports, whose population, partition and leaf counts can coincide, in distinct domains.  A rank that writes no window,
or whose root is not the one the TP run root combines, exits the Commit before its challenge.

The verifier's own window of rank ``r`` (`window_of_record`) comes from the Commit's record alone, no committed value read: the TP run
root and every rank's root of its first instrumented run (`record`: ``runs.jsonl``, refused unless the TP run root combines those
roots under the form the record's ``root_tag`` names, `tp.run_root.combine`), and the population over rank ``r``'s Programs of record
(`rank_replay.load`, under the Build's ``rank<r>/``, refused unless of record for the rank in ``versions.json`` and the rank's) for the
workload the record served.  Neither side evaluates a row, so the Definitions the Build bound are not held to this tree's here: the
one-stage proof is over the verifier's own template.  `source` holds a registration's window to it: ``served``, its run ``<TP run root>/rank<r>`` under a
SHA-512 TP run root, its ``tp`` the record's and its served Programs the ones the verifier loaded.  A served window names a SHA-512
commitment, as `deferred_rows.source` holds a deferred Commit's: a TP Commit's run root is SHA-512 (``verity/tp-run-root/v2``), and a
record under the SHA-256 v1 root (an old record, or ``--commit-scheme vllm-v1``) passes no window as served (``sha512``).

A TP Commit that deferred its ranks' replay to sealed bundles and wrote no window has its windows from those bundles afterwards
(``verity-vllm tp-serving-rows COMMIT_OUT --partition F --out O``, `from_bundle`): rank ``r``'s bundle opened against the record as
its CPU replay opens it (`commit_replay._of_record`: pending or passed, a failed rank refused), its Programs of record, its weight
shards opened against its committed weights root, and the window `pipeline.serving_rows.after_commit` writes under the same run and
``tp``.  Its rows are a fresh commitment, made after the Commit and before sampling, so the window is a ``stand-in`` whose link to
the rank's rows-sha512 root is provenance (`stand_in`: ``of`` the TP run root, the rank, its root, the bundle's manifest digest and
this command, and ``link``).  `source` takes a stand-in only for a rank whose bundle the record names (``replay_deferred.json``) and
whose replay did not fail, with that ``of`` and ``link``."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verity_vllm import config as C
from verity_vllm.config import option

SCHEMA = "verity-vllm/tp-serving-rows/v1"
ROWS_SCHEME = "rows-sha512"
ROWS = "verity-vllm tp-serving-rows: a rank's sealed replay bundle -> rows"
LINK = ("provenance, not proof: verity-vllm tp-serving-rows opened the rank's sealed replay bundle against the TP Commit's record and "
        "committed these rows from openings verified against the rank's rows-sha512 root; no circuit opens the rows-sha512 leaves")


def require(args: Any, replay_cfg: dict[str, Any] | None) -> None:
    """Refuse a ``--serving-rows`` TP Commit whose windows could not be written before sampling."""
    if args.serving_rows and (replay_cfg is None or not args.replay_deferred or args.only_arm == "control"):
        raise SystemExit("--serving-rows on a TP Commit needs --required-manifest with a --build-rank per rank (each rank reads its Programs of "
                         "record from the replay config), --replay-deferred at a world that defers (no rank draws its replay seed before "
                         "the windows) and an instrumented run")


def run_of(tp_root: bytes, rank: int) -> str:
    return f"{tp_root.hex()}/rank{int(rank)}"


def stand_in(tp: dict[str, Any], bundle_manifest_sha256: str) -> dict[str, Any]:
    """A rank's window's link to its TP Commit's commitment (``tp``: the window's world, rank, TP run root, rank root and root tag):
    ``of`` the rows-sha512 commitment of the rank's root under the TP run root, the bundle's manifest digest and the command that
    turned the bundle into rows, and ``link``."""
    return {"of": {"commitment": ROWS_SCHEME, "tp_run_root": tp["tp_run_root"], "rank": int(tp["rank"]), "run_root": tp["run_root"],
                   "bundle_manifest_sha256": bundle_manifest_sha256, "rows": ROWS}, "link": LINK}


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
    """Rank ``st["rank"]``'s window over its committed store opened against its root, the weight shards it registered and its Programs
    of record (`rank_replay.load`: refused unless of record and this rank's), from the worker's commit state."""
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.pipeline.tp import rank_replay as RR

    rank, world, rc = int(st["rank"]), int(tp["world"]), st.get("rc")
    if rc is None or not st.get("bmap"):
        raise LookupError(f"rank {rank}: no finalized run or no binding map (binding skipped or failed)")
    if rc.run_root.hex() != tp["roots"].get(str(rank)):
        raise ValueError(f"rank {rank}: its run root {rc.run_root.hex()[:16]} is not the one the TP run root combines")
    p = RR.load(st["replay"], rank, world, definitions=False)
    res = SRW.after_commit(str(Path(request["dir"]) / f"rank{rank}"), partition_file=request["partition"], programs=p.progs, bmap=st["bmap"],
                           committed=OC.OpenedReader(st["com"], rc), weights=SR.model_weights_provider(st["weights"], p.index.prefix),
                           manifest=RR.manifest_of_rank(st.get("manifest_doc"), rank), run_root=run_of(bytes.fromhex(tp["tp_run_root"]), rank),
                           run=request.get("run"), row=request.get("row"), arrivals=p.arrivals, program_dirs=p.dirs, digests=p.digs,
                           committer=request.get("committer"), workers=1,   # vLLM's TP workers are daemonic: no process pool under them
                           tp={"world": world, "rank": rank, "tp_run_root": tp["tp_run_root"], "run_root": rc.run_root.hex(), "root_tag": tp["root_tag"]},
                           index={"attention_paths": p.attn_paths, "tp": p.tp}, reuse=(p.index,))
    return {k: res.get(k) for k in ("run", "population", "partition_digest", "registration_digest", "seconds")}


def from_bundle(commit_dir: str | os.PathLike, rank: int, partition_file: str | os.PathLike, out_dir: str | os.PathLike, *, pair: int = 0,
                build_root: str | os.PathLike | None = None, checkpoint: Any = None, weights: Any = None,
                required_manifest: str | None = None, workers: int | None = None, programs: Any = None,
                binding: dict[str, Any] | None = None, transfer_gates: Sequence[str] = ()) -> dict[str, Any]:
    """Rank ``rank``'s ``stand-in`` window into ``out_dir`` from its sealed replay bundle of pair ``pair`` (module docstring): the
    bundle opened against the record (`commit_replay._of_record`), the required manifest, the rank's Programs of record
    (`rank_replay.load`; ``build_root``'s ``rank<r>/`` instead of the record's dirs, its digests held either way; ``programs``, the
    rank's `rank_replay.Loaded` a registered compiler made from advice, instead, its ``binding`` in the window's provenance and its
    ``transfer_gates`` proved whole, `pipeline.advice_windows`), its weight shards
    opened against the committed weights root from the bundle's tensors and ``checkpoint`` (a `weights_of_record.CheckpointIndex`;
    without one, only the tensors the bundle carries; ``weights``, tests, replaces them), each on the window's first read of it
    (`cpu_replay.OnRead`), and every value read through an opening verified against the rank's root."""
    from types import SimpleNamespace

    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.pipeline import cpu_replay as CPU
    from verity_vllm.pipeline import replay_bundle as RB
    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.pipeline.tp import c2 as C2
    from verity_vllm.pipeline.tp import commit_replay as CR
    from verity_vllm.pipeline.tp import rank_replay as RR

    d, rank = Path(commit_dir), int(rank)
    if not (d / C2.DEFERRED).is_file():
        raise CR.Refused(f"{d / C2.DEFERRED}: not a deferred TP Commit (tp-commit --replay-deferred): no bundle holds its values")
    dd = json.loads((d / C2.DEFERRED).read_text())
    try:
        b = RB.open_bundle(RB.path(d, pair, rank=rank))
    except (OSError, ValueError) as e:
        raise CR.Refused(f"pair {pair} rank {rank}: {e}") from e
    ctx, bdir = b["context"], Path(b["path"])
    _row, c, store, run, bmap = CR._of_record(d, dd, b, pair, rank, awaiting=False)
    if store.commit_scheme != ROWS_SCHEME:
        raise CR.Refused(f"rank {rank}: its store commits under {store.commit_scheme}, not {ROWS_SCHEME}: no stand-in names a SHA-512 root")
    rec = record(d)
    if rec["tp_run_root"] != ctx["tp_run_root"]:
        raise CR.Refused(f"pair {pair}: the bundle's TP run root {ctx['tp_run_root'][:16]} is not the record's {rec['tp_run_root'][:16]}")
    man = CR._manifest(SimpleNamespace(required_manifest=required_manifest), ctx)
    world = int(dd["world"])
    cfg = dict(ctx["replay"])
    if build_root is not None:
        cfg["programs"] = {str(r): str(Path(build_root) / f"rank{r}") for r in range(world)}
    p = programs if programs is not None else RR.load(cfg, rank, world, definitions=False)
    if weights is None:
        tree = json.loads((bdir / "weights" / "tree.json").read_text())
        try:
            words = CPU.OnRead(bdir, tree, (c.get("weight_registration") or {}).get("root"), checkpoint, tp={"world": world, "rank": rank},
                               roles=CR.registered_roles(p))
        except ValueError as e:
            raise CR.Refused(f"rank {rank}: the bundle's weights: {e}") from e
        weights = CPU.provider(words, p.index.prefix, set())
    tp = {"world": world, "rank": rank, "tp_run_root": rec["tp_run_root"], "run_root": c["run_root"], "root_tag": rec["root_tag"]}
    res = SRW.after_commit(str(out_dir), partition_file=str(partition_file), programs=p.progs, bmap=bmap, committed=OC.OpenedReader(store, run),
                           weights=weights, manifest=RR.manifest_of_rank(man, rank), run_root=run_of(bytes.fromhex(rec["tp_run_root"]), rank),
                           row=ctx.get("label"), arrivals=p.arrivals, program_dirs=p.dirs, digests=p.digs,
                           committer={"from": "replay bundle", "pair": pair, "source": ctx.get("code")}, workers=workers, tp=tp,
                           index={"attention_paths": p.attn_paths, "tp": p.tp}, reuse=(p.index,), stand_in=stand_in(tp, b["manifest_sha256"]),
                           binding=binding, transfer_gates=transfer_gates)
    return {k: res.get(k) for k in ("run", "population", "partition_digest", "registration_digest", "seconds")}


def record(commit_dir: str | os.PathLike) -> dict[str, Any]:
    """The TP Commit's record (its ``--out``): the world, TP run root and every rank's root of its pair-0 instrumented run, the one
    whose windows `commit` wrote (``runs.jsonl``, which each Commit into the same ``--out`` appends to: the last such row, the run
    ``versions.json`` is of), refused unless the TP run root combines exactly those roots; the root tag, each rank's Programs of
    record and the workload (``versions.json``).  The roots combine under the form the record's root tag names (`tp.run_root.combine`:
    v2, or an old record's v1, which `source` refuses as served), and any other tag is refused.  ``bundles``: the manifest sha256 of
    each rank's sealed pair-0 bundle whose replay did not fail (``replay_deferred.json``; none when nothing was deferred)."""
    from verity_vllm.pipeline.tp import c2 as C2
    from verity_vllm.pipeline.tp import run_root as RT

    d = Path(commit_dir)
    rows = [json.loads(x) for x in (d / "runs.jsonl").read_text().splitlines() if x.strip()]
    row = next((r for r in reversed(rows) if r.get("arm") != "control" and r.get("pair") == 0 and (r.get("commit") or {}).get("tp_run_root")),
               None)
    if row is None:
        raise LookupError(f"{d / 'runs.jsonl'}: no pair-0 instrumented run with a TP run root (the Commit failed before its roots)")
    c = row["commit"]
    world = int(c["world"])
    roots = {int(k): str(v["run_root"]) for k, v in (c.get("per_rank") or {}).items()}
    if sorted(roots) != list(range(world)):
        raise ValueError(f"the record's ranks {sorted(roots)} are not the world of {world}")
    v = json.loads((d / "versions.json").read_text())
    if RT.combine(v.get("root_tag"), world, {r: bytes.fromhex(x) for r, x in roots.items()}).hex() != c["tp_run_root"]:
        raise ValueError(f"the record's TP run root {c['tp_run_root'][:16]} does not combine its ranks' roots")
    argv = [str(x) for x in v.get("argv") or []]
    dd = json.loads((d / C2.DEFERRED).read_text()) if (d / C2.DEFERRED).is_file() else {}
    bundles = {str(int(x["rank"])): str(x["manifest_sha256"]) for x in dd.get("bundles") or []
               if int(x["pair"]) == 0 and "replay_deferred" in c["per_rank"].get(str(int(x["rank"])), {})
               and c["per_rank"][str(int(x["rank"]))].get("commit_pass") is not False}
    return {"world": world, "tp_run_root": c["tp_run_root"], "roots": {str(r): roots[r] for r in range(world)}, "root_tag": v["root_tag"],
            "bundles": bundles,
            "of_record": (v.get("sampled_replay") or {}).get("of_record") or {}, "label": row.get("label"),
            "workload": argv[argv.index("--workload") + 1] if "--workload" in argv[:-1] else None, "workload_sha256": row.get("workload_sha256")}


def window_of_record(commit_dir: str | os.PathLike, rank: int, partition_file: str | os.PathLike, build_root: str | os.PathLike,
                     workload_file: str | os.PathLike) -> tuple[Any, dict[str, Any]]:
    """The verifier's own window of rank ``rank`` (`pipeline.serving_rows.population` over the rank's Programs of record under
    ``<build_root>/rank<r>``, its run ``<TP run root>/rank<r>`` of `record`), and what `source` holds the rank's registration to.
    The requests are the workload's in order, ``r<position>``, as ``tp-commit`` names them, from a workload file refused unless the
    one the record served (its ``workload_sha256``, `vllm_adapter.load_workload`'s digest)."""
    from verity_vllm.pipeline.tp import rank_replay as RR

    rec = record(commit_dir)
    rank, world = int(rank), int(rec["world"])
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} is not a rank of the world of {world}")
    doc = json.loads(C.workload_file(workload_file).read_text())
    got = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if got != rec["workload_sha256"]:
        raise ValueError(f"{workload_file}: workload sha256 {got[:16]} is not the record's {str(rec['workload_sha256'])[:16]}")
    reqs = [{"request_id": f"r{i}", "arrive_step": int(q.get("arrive_step") or 0), "prompt_len": len(q["prompt_token_ids"]),
             "max_tokens": int(q["max_tokens"])} for i, q in enumerate(doc["requests"])]
    p = RR.load({"programs": {str(rank): str(Path(build_root) / f"rank{rank}")}, "of_record": rec["of_record"], "requests": reqs}, rank, world,
                definitions=False)
    return window_of(rec, rank, partition_file, p)


def window_of(rec: dict[str, Any], rank: int, partition_file: str | os.PathLike, p: Any,
              binding: dict[str, Any] | None = None, transfer_gates: Sequence[str] = ()) -> tuple[Any, dict[str, Any]]:
    """The verifier's own window of rank ``rank`` of the record ``rec`` (`record`) over the rank's loaded Programs ``p`` (a
    `rank_replay.Loaded`: of record, `window_of_record`, or a registered compiler's from advice, `pipeline.advice_windows`), and what
    `source` holds the rank's registration to (``binding``: the compiler's digest and the advice's hiding root the window must name;
    ``transfer_gates``: the templates the compiler lists as holding transfer gates, every unit of which the window proves)."""
    from verity_vllm.pipeline import serving_rows as SRW

    rank, world = int(rank), int(rec["world"])
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} is not a rank of the world of {world}")
    window, _rows = SRW.population(p.progs, json.loads(Path(partition_file).read_text()), run_of(bytes.fromhex(rec["tp_run_root"]), rank),
                                   p.dirs, {"attention_paths": p.attn_paths, "tp": p.tp}, reuse=(p.index,), transfer_gates=transfer_gates)
    tp = {"world": world, "rank": rank, "tp_run_root": rec["tp_run_root"], "run_root": rec["roots"][str(rank)], "root_tag": rec["root_tag"]}
    sha = rec["bundles"].get(str(rank))
    held = {"tp": tp, "served_programs": {str(k): str(x) for k, x in sorted(p.digs.items())}, "stand_in": stand_in(tp, sha) if sha else None,
            **{k: (binding or {}).get(k) for k in BINDING}}
    return window, held


#: the window provenance's keys that bind the Programs to the registered compiler and the advice it read (`pipeline.advice_windows`)
BINDING = ("compiler", "advice")


def source(registered: dict[str, Any] | None, mine: Any, held: dict[str, Any]) -> dict[str, Any]:
    """A rank's registration's ``window`` held to the verifier's own (`window_of_record`, `window_of`): ``served`` with no stand-in link, or a
    ``stand-in`` of a rank whose bundle the record names with `stand_in`'s ``of`` and ``link`` over it (``held["stand_in"]``); either
    way its run the rank's of record under a SHA-512 TP run root (``sha512``, `deferred_rows.source`'s 64-byte test: an old record's
    SHA-256 v1 root is refused), its ``tp`` the record's, its served Programs the ones the verifier loaded, and its ``compiler`` and
    ``advice`` the verifier's (`BINDING`: none for Programs of record, the digests of the compiler and the advice the verifier
    compiled for an advice window)."""
    from verity_vllm.pipeline.deferred_rows import SERVED_ROOT

    w = registered or {}
    si = held.get("stand_in") if w.get("kind") == "stand-in" else None
    ok = {"kind": w.get("kind") == "served" or si is not None, "run": w.get("run") == mine.run,
          "sha512": len(held["tp"]["tp_run_root"]) == 2 * SERVED_ROOT, "tp": w.get("tp") == held["tp"],
          "served_programs": w.get("served_programs") == held["served_programs"],
          **{k: w.get(k) == held.get(k) for k in BINDING},
          **({"of": w.get("of") == si["of"], "link": w.get("link") == si["link"]} if si else {"no_link": "of" not in w and "link" not in w})}
    return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v],
            "detail": {"kind": w.get("kind"), "run": w.get("run"), "tp": w.get("tp"), **({"of": w.get("of")} if si else {})}}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """verity-vllm tp-serving-rows COMMIT_OUT --partition F --out O [--ranks 0,3] [--build-root B] [--checkpoint-dir CK ...]: a deferred TP
    Commit's ``stand-in`` windows, ``O/rank<r>/`` each, from its ranks' sealed replay bundles (`from_bundle`)."""
    commit_out: str = option("commit_out", help="the deferred TP Commit's --out (runs.jsonl, versions.json, replay_deferred.json, its bundles)")
    partition: str = option("--partition", help="the verifier's partition file: {templates, law, scope?}")
    out: str = option("--out", help="the windows' directory: rank<r>/ each (registration.json, partition.json, index.json, window.json, M0's bodies)")
    ranks: str | None = option("--ranks", default=None, help="comma-separated ranks (default: every rank of the world)")
    build_root: str | None = option("--build-root", default=None, help="each rank's Programs of record in rank<r>/ (default: the record's Build dirs)")
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None,
                                        help="the checkpoint of record (default: none, so only the tensors the bundles carry open)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard sha256 pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    no_verify_shards: bool = option("--no-verify-shards", default=False, action="store_true",
                                    help="skip the shard sha256 (each tensor read is still held to the committed weights root)")
    required_manifest: str | None = option("--required-manifest", default=None, help="the Commit's required manifest (default: its recorded path)")
    workers: int | None = option("--workers", type=int, default=None, help="processes hashing the rows (default: min(16, CPUs))")


def main(o: Options) -> int:
    from verity_vllm.pipeline.tp import commit_replay as CR

    ck = None
    if o.checkpoint_dir:
        from verity_vllm.check import weights_of_record as WR
        ck = WR.CheckpointIndex(o.checkpoint_dir, checkpoints_manifest=o.checkpoints_manifest, model=o.model, revision=o.revision,
                                verify_shards=not o.no_verify_shards)
    try:
        world = int(record(o.commit_out)["world"])
        for r in [int(x) for x in o.ranks.split(",")] if o.ranks else range(world):
            got = from_bundle(o.commit_out, r, o.partition, Path(o.out) / f"rank{r}", build_root=o.build_root, checkpoint=ck,
                              required_manifest=o.required_manifest, workers=o.workers)
            print(f"[tp-serving-rows] rank {r}: stand-in window of {got['population']} units, run {got['run'][:16]} rank {r}, registration "
                  f"{str(got['registration_digest'])[:16]}", flush=True)
    except (CR.Refused, LookupError, ValueError, RuntimeError) as e:
        print(f"[tp-serving-rows] REFUSED: {e}", flush=True)
        return 2
    return 0
