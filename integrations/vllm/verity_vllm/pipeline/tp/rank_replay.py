"""The C2 of record on one rank of a TP Commit, for the rank's committing process (`TP2CaptureWorkerExtension._tp2_sampled_replay`)
and for the CPU replay of the rank's sealed bundle (`pipeline.tp.commit_replay`) alike.

`load` reads the rank's request Programs and refuses, by name, any whose digest is not of record for the rank or whose TP context is
not the rank's; `evaluate` runs manbind's sampled replay and the exhaustive boundary linkage on the rank's binding map, its committed
members (`committed`: an `oracle_compare.OpenedReader` over the live committer, or over the bundle's dumped store) and its weight
shards (`weights_of(prefix)`: the engine's parameters, or the checkpoint's shards opened against the committed weights root).  The
seed is `seed_of`'s: the driver's targeted seed, else the verifier's challenge with the rank's run root, else the run root."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SCHEMA = "verity-sweep/tp-sampled-replay/v1"
METHOD = "sampled-exact-replay per rank (manbind sampled_replay on the rank's map, Programs and shards)"


@dataclass
class Loaded:
    """One rank's request Programs of record: by request label the loaded Program, its instances, digest and dir."""
    rank: int
    world: int
    reqs: list[dict[str, Any]]
    loaded: dict[str, dict[str, Any]]
    progs: dict[str, dict[str, Any]]
    digs: dict[str, str]
    dirs: dict[str, str]
    arrivals: dict[str, int]
    attn_paths: list[str]
    index: Any

    @property
    def tp(self) -> dict[str, int]:
        return {"rank": self.rank, "world": self.world}


def load(cfg: dict[str, Any], rank: int, world: Any) -> Loaded:
    """The rank's request Programs (B1: the request Program at the rank's Build dir; B>=2: one component per (LP, T) under it, by the
    manifest's lookup of record), REFUSED by name unless every loaded digest is of record for the rank (`cfg["of_record"]`) and every
    Program's TP context (result.json `tp`) is this rank's of this world."""
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.engine.rank_worker import _tp2_request_program_dirs
    from verity_vllm.query.manifest import format as RM

    reqs = list(cfg.get("requests") or [])
    root = (cfg.get("programs") or {}).get(str(rank))
    if not root:
        raise RuntimeError(f"no Build dir for rank {rank} in the replay config (driver --build-rank{rank})")
    of_record = {str(x) for x in ((cfg.get("of_record") or {}).get(str(rank)) or [])}
    if not of_record:
        raise RuntimeError(f"no Program digest of record for rank {rank} (GP-01 workload_program components / manifest per-rank digest)")
    dirs = _tp2_request_program_dirs(str(root), reqs, rank)
    by_dir = {d: RM.load_program(d) for d in sorted(set(dirs.values()))}
    loaded: dict[str, dict] = {rid: by_dir[d] for rid, d in dirs.items()}
    digs = {k: str(((v.get("result") or {}).get("program") or {}).get("digest") or (v.get("result") or {}).get("program_digest")
                   or (v.get("instances") or {}).get("digest") or "") for k, v in loaded.items()}
    bad = {k: d for k, d in digs.items() if d not in of_record}
    if bad:
        named = ", ".join(f"{k}: {d[:16]} ({loaded[k].get('dir')})" for k, d in bad.items())
        raise RuntimeError(f"rank {rank}: REFUSED (not the Program of record): {named} vs of-record {sorted(x[:16] for x in of_record)}")
    progs = {k: v["instances"] for k, v in loaded.items()}
    arr = {str(r["request_id"]): int(r.get("arrive_step") or 0) for r in reqs}
    # the export wrapper's prefix is DETERMINED from the Program's attention rows vs the implementation paths result.json observed
    # (`attention_impls_observed`) -- the same source the manifest used, so the replay addresses the map's op_paths (a rank Program
    # names `inner.model.model.layers.*`, the map `model.layers.*`; without the observed paths the heuristic strips nothing)
    attn_paths = sorted({a for v in loaded.values() for a in ((v.get("result") or {}).get("attention_impls_observed") or {}).keys()})
    # [TP-08d] the rank Programs' TP context (result.json `tp` = {"world", "rank"}): their row-parallel Gemm / EmbeddingShard rows
    # produce this rank's PARTIAL members, the collective rows the module's output -- every loaded Program must be THIS rank's
    tps = {k: ((v.get("result") or {}).get("tp") or {}) for k, v in loaded.items()}
    bad_tp = {k: t for k, t in tps.items() if int(t.get("world") or 1) != int(world) or int(t.get("rank") or 0) != rank}
    if bad_tp:
        raise RuntimeError(f"rank {rank}: REFUSED (not this rank's Program): " + ", ".join(f"{k}: tp={t}" for k, t in bad_tp.items()))
    world = int(world)
    rid0 = next(iter(progs))
    index = SR.ProgramIndex(progs[rid0], attention_paths=attn_paths, tp={"rank": rank, "world": world}, program_dir=dirs.get(rid0))
    return Loaded(rank, world, reqs, loaded, progs, digs, dirs, arr, attn_paths, index)


def unreplayable_sites(p: Loaded) -> dict[str, Any]:
    """[TP-12 / rev M-0418] the SITES of the collective-output families this replay cannot evaluate (AllReduce2 / AllGather2 /
    EmbeddingShard rows with a committed member), from the Program itself: the cross-rank pass records its strata against them.
    Never raises: ``{"error": ...}`` names what could not be read."""
    try:
        from verity_vllm.program.registry import b1_tp2 as _TP2
        from verity_vllm.query.manifest import format as RM

        xs: dict[str, set] = {}
        xr_fams = _TP2.cross_rank_families(p.world)     # [lane vllm-tp-n] from the registry's composites at this world, not typed
        for r_ in p.index.rows:
            fam_ = RM.spec_family(r_["spec"])
            if fam_ in xr_fams:
                d_ = p.index.describe(r_)
                if d_ and d_.get("members"):
                    xs.setdefault(fam_, set()).add(str(d_["op_path"]))
        return {f_: sorted(v_) for f_, v_ in xs.items()}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def seed_of(cfg: dict[str, Any], run_root: bytes) -> tuple[int, str]:
    """[O6] seed from the rank's run root AFTER commitment unless the driver targets one; [manbind MAN-06d, M-0608] under a
    VERIFIER-ISSUED post-commitment challenge the seed is sha256(challenge | this rank's run root)[:8] (no grind by recommitting)."""
    from verity_vllm.check.replay import driver as SR

    if cfg.get("seed") is not None:
        return int(cfg["seed"]), "--replay-seed (targeted)"
    if cfg.get("challenge"):
        return SR.challenge_seed(str(cfg["challenge"]), run_root)
    return int.from_bytes(run_root[:8], "big"), "this rank's run root (first 8 bytes, after commitment)"


def manifest_of_rank(doc: Any, rank: int) -> dict[str, Any] | None:
    """[R17-3 / reqval] Q(P) of record for THIS rank: the merged TP manifest's identities of rank r (per-rank populations + the rank's
    own partials); the replay's reach is reconciled against them identity by identity (population.query_total / ...)."""
    if not (isinstance(doc, dict) and isinstance(doc.get("identities"), list)):
        return None
    return dict(doc, identities=[x for x in doc["identities"] if int(x.get("rank", 0) or 0) == rank])


def evaluate(p: Loaded, cfg: dict[str, Any], *, bmap: dict[str, Any], committed: Any, weights_of: Callable[[str], Any],
             weights_check: dict[str, Any], run_root: bytes, manifest_doc: Any, worker_peak: int | None = None) -> tuple[dict, dict]:
    """(the rank's sampled-replay record, its boundary-linkage record) on the rank's map, committed members and weight shards."""
    from verity_vllm.check.replay import driver as SR

    rank = p.rank
    seed, seed_src = seed_of(cfg, run_root)
    ring = cfg.get("ring_slots")
    sr = SR.sampled_replay(p.progs, bmap, committed, weights_of(p.index.prefix), arrivals=p.arrivals, per_stratum=int(cfg.get("per_stratum") or 2),
                           seed=seed, seed_source=seed_src, max_evaluate=int(cfg.get("max_evaluate") or 2000), weights_check=weights_check,
                           targets=list(cfg.get("targets") or []), workers=cfg.get("workers"), worker_peak=worker_peak,
                           ring_slots=(int(ring) if ring else None), attention_paths=p.attn_paths, tp=p.tp,
                           manifest=manifest_of_rank(manifest_doc, rank), program_dirs=p.dirs, replay_k=(cfg.get("k_per_rank") or {}).get(str(rank)))
    sr["wrapper_prefix"] = p.index.prefix
    sr["tp_context"] = p.tp
    sr["programs"] = {k: {"dir": p.loaded[k].get("dir"), "program_digest": p.digs[k]} for k in p.loaded}
    sr["rank"] = rank                    # [MAN-06c] the strata carry this rank's label (a per-rank map: every entry identity.rank == rank)
    arr = p.arrivals
    wreq = [{"request_id": str(r["request_id"]), "arrive_step": arr[str(r["request_id"])], "prompt_token_ids": list(r.get("prompt_token_ids") or []), "seed": r.get("seed"),
             "steps": list(range(arr[str(r["request_id"])], arr[str(r["request_id"])] + int(r.get("max_tokens") or 1)))} for r in p.reqs]
    lk = SR.boundary_linkage(bmap, committed, wreq, embed_op_path=p.index.embed_op_path(), splits_of_record=SR.splits_of_record(p.loaded))
    lk["rank"] = rank
    sr["value_source"] = dict(committed.record(), forked_children=sr.pop("value_reads_in_children", None))
    return sr, lk


def not_run(rank: int, e: BaseException, trace: str) -> tuple[dict, dict]:
    """Fail closed: an unrunnable replay is 'not evaluated', never PASS."""
    return ({"schema": "verity-sweep/sampled-replay/v1", "method": "sampled-exact-replay", "result": None, "grade": "not-evaluated", "rank": rank,
             "why": f"rank {rank}: sampled replay not run: {e!r}"[:400], "error": repr(e)[:400], "trace": trace[-1500:]},
            {"schema": "verity-sweep/boundary-linkage/v1", "result": None, "rank": rank, "why": f"rank {rank}: boundary linkage not run: {e!r}"[:300], "pairs": 0, "linked": 0})


def no_map() -> dict[str, Any]:
    return {"sampled_replay": {"result": None, "grade": "not-evaluated", "why": "no binding map on this rank (binding skipped / failed)"},
            "boundary_linkage": {"result": None, "why": "no binding map on this rank"}}
