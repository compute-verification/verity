"""The C2 of record on one rank of a TP Commit, for the rank's committing process (`in_process`, or `defer` under
`--replay-deferred`) and for the CPU replay of the rank's sealed bundle (`pipeline.tp.commit_replay`) alike.

`load` reads the rank's request Programs and refuses, by name, any whose digest is not of record for the rank or whose TP context is
not the rank's; `evaluate` runs manbind's sampled replay and the exhaustive boundary linkage on the rank's binding map, its committed
members (`committed`: an `oracle_compare.OpenedReader` over the live committer, or over the bundle's dumped store) and its weight
shards (`weights_of(prefix)`: the engine's parameters, or the checkpoint's shards opened against the committed weights root).  The
seed is `seed_of`'s: the driver's targeted seed, else the verifier's challenge with the rank's run root, else the run root.

`in_process` and `defer` take the worker's commit state (`TP2CaptureWorkerExtension._tp2_commit`: rank, world, committer, engine,
binding map, replay config)."""
from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "verity-sweep/tp-sampled-replay/v1"
METHOD = "sampled-exact-replay per rank (manbind sampled_replay on the rank's map, Programs and shards)"


def request_program_dirs(root: str, reqs: list[dict], rank: int) -> dict[str, str]:
    """[TP-13 / moe] This rank's request Program DIR per declared request, by the manifest's lookup of record
    (`required_manifest._program_dirs_for`): a single request Program at `<root>` (B1: `<root>/instances.json.gz`), else under `<root>`
    a `*LP<prompt_len>_T<max_tokens-1>` dir, else any `build_request*/result.json` declaring EXACTLY (LP, T) -- row_pod names the row's
    nominal shape `build_request` and every other shape `build_request_LP<lp>_T<t>`, so a B>=2 mixed row's nominal-shape request has no
    `*LP_T` dir and a bare glob refused it ("no request Program for r0 at <root>/*LP1024_T127": the Qwen3-30B-A3B TP2 B2 and OLMoE TP2 B8
    rows on r16-int-4 537a9362d, sampled replay not run on both ranks and TP-12's unreplayable_sites never set).  Never a near shape;
    the caller still refuses any loaded Program whose digest is not of record."""
    from verity_vllm.query.manifest import format as RM

    if os.path.exists(os.path.join(str(root), "instances.json.gz")):
        return {str(r["request_id"]): str(root) for r in reqs}
    try:
        return {str(k): str(v) for k, v in RM.program_dirs_for({"requests": list(reqs)}, str(root)).items()}
    except SystemExit as e:  # the manifest lookup names the missing (LP, T) and exits; inside a worker that is an error, not an exit
        raise RuntimeError(f"rank {rank}: {e}") from None


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
    from verity_vllm.query.manifest import format as RM

    reqs = list(cfg.get("requests") or [])
    root = (cfg.get("programs") or {}).get(str(rank))
    if not root:
        raise RuntimeError(f"no Build dir for rank {rank} in the replay config (driver --build-rank{rank})")
    of_record = {str(x) for x in ((cfg.get("of_record") or {}).get(str(rank)) or [])}
    if not of_record:
        raise RuntimeError(f"no Program digest of record for rank {rank} (GP-01 workload_program components / manifest per-rank digest)")
    dirs = request_program_dirs(str(root), reqs, rank)
    by_dir = {d: RM.load_program(d, expand=False) for d in sorted(set(dirs.values()))}   # the replay expands the rows it reads
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
    """[TP-12] the SITES of the collective-output families this replay cannot evaluate (AllReduce2 / AllGather2 /
    EmbeddingShard rows with a committed member), from the Program itself: the cross-rank pass records its strata against them.
    Never raises: ``{"error": ...}`` names what could not be read."""
    try:
        from verity_vllm.program.frontend import collective_kinds
        from verity_vllm.query.manifest import format as RM

        xs: dict[str, set] = {}
        xr_fams = collective_kinds.cross_rank_families(p.world)     # [lane vllm-tp-n] from the registry's composites at this world, not typed
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
    """[O6] seed from the rank's run root AFTER commitment unless the driver targets one; [manbind MAN-06d] under a
    VERIFIER-ISSUED post-commitment challenge the seed is sha256(challenge | this rank's run root)[:8] (no grind by recommitting)."""
    from verity_vllm.check.replay import driver as SR

    if cfg.get("seed") is not None:
        return int(cfg["seed"]), "--replay-seed (targeted)"
    if cfg.get("challenge"):
        return SR.challenge_seed(str(cfg["challenge"]), run_root)
    return int.from_bytes(run_root[:8], "big"), "this rank's run root (first 8 bytes, after commitment)"


def manifest_of_rank(doc: Any, rank: int) -> dict[str, Any] | None:
    """[reqval] Q(P) of record for THIS rank: the merged TP manifest's identities of rank r (per-rank populations + the rank's
    own partials); the replay's reach is reconciled against them identity by identity (population.query_total / ...)."""
    if not (isinstance(doc, dict) and isinstance(doc.get("identities"), list)):
        return None
    return dict(doc, identities=[x for x in doc["identities"] if int(x.get("rank", 0) or 0) == rank])


def index_memo(p: Loaded, manifest: dict[str, Any] | None) -> dict:
    """`sampled_replay`'s memo holding `load`'s index of the first Program, when it is the index the replay would build (the manifest's
    query is the default one, no norm-scale policy); else empty, and the replay builds its own."""
    from verity_vllm.check.replay import driver as SR, index as IX
    from verity_vllm.query import norm_scales

    if (SR.manifest_query(manifest) or IX.DEFAULT_QUERY) != IX.DEFAULT_QUERY or norm_scales.policy_of(manifest):
        return {}
    return {("index", id(p.progs[next(iter(p.progs))])): p.index}


def evaluate(p: Loaded, cfg: dict[str, Any], *, bmap: dict[str, Any], committed: Any, weights_of: Callable[[str], Any],
             weights_check: dict[str, Any], run_root: bytes, manifest_doc: Any, worker_peak: int | None = None) -> tuple[dict, dict]:
    """(the rank's sampled-replay record, its boundary-linkage record) on the rank's map, committed members and weight shards."""
    from verity_vllm.check.replay import driver as SR

    rank = p.rank
    seed, seed_src = seed_of(cfg, run_root)
    ring = cfg.get("ring_slots")
    man = manifest_of_rank(manifest_doc, rank)
    sr = SR.sampled_replay(p.progs, bmap, committed, weights_of(p.index.prefix), arrivals=p.arrivals, per_stratum=int(cfg.get("per_stratum") or 2),
                           seed=seed, seed_source=seed_src, max_evaluate=int(cfg.get("max_evaluate") or 2000), weights_check=weights_check,
                           targets=list(cfg.get("targets") or []), workers=cfg.get("workers"), worker_peak=worker_peak,
                           ring_slots=(int(ring) if ring else None), attention_paths=p.attn_paths, tp=p.tp,
                           manifest=man, program_dirs=p.dirs, replay_k=(cfg.get("k_per_rank") or {}).get(str(rank)), memo=index_memo(p, man))
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


def in_process(st: dict[str, Any], rc: Any) -> dict[str, Any]:
    """The rank's replay record from its live committer, the engine's parameters and the rank's Programs."""
    cfg = st.get("replay") or {}
    rank = int(st["rank"])
    t0 = time.perf_counter()
    base = {"schema": SCHEMA, "rank": rank, "method": METHOD}
    bmap = st.get("bmap")
    if not bmap:
        return base | no_map()
    try:
        from verity_vllm.check import oracle_compare as OC
        from verity_vllm.check.replay import driver as SR

        com = st["com"]
        p = load(cfg, rank, st.get("world"))
        st["xrank_unreplayable_sites"] = unreplayable_sites(p)
        rd = OC.OpenedReader(com, rc)
        # [O5] the shards the relations consume must be the COMMITTED shards: re-register the live parameters now vs the run's weights root
        try:
            w_saved = getattr(com, "weights", None)
            w_now = com.register_weights()
            if w_saved is not None:
                com.weights = w_saved
            w_ref = (rc.weight_registration or {}).get("root") if isinstance(rc.weight_registration, dict) else (w_saved or {}).get("root")
            eq = (str(w_now.get("root")) == str(w_ref)) if w_ref else None
            # this is SELF-CONSISTENCY (the shards the relations consumed are the shards the run committed), NOT a pin: nothing here
            # compares the root with a digest OF RECORD derived from the checkpoint bytes (per rank = H(shard bytes in Program order) via
            # the rank Program's shard map).  `pin_of_record` stays None until the Build emits it.
            wc = {"equal": eq, "self_consistent": eq, "pin_of_record": None, "pinned": None, "committed_root": w_ref, "replay_root": w_now.get("root"),
                  "tensors": w_now.get("tensors"), "method": f"rank {rank}: live parameters (weight SHARDS) re-registered at replay == the run's committed weights root "
                                                              "(self-consistency; not a weights-of-record pin, M-0614)",
                  "why": None if eq else ("no committed weights root on the run" if not w_ref else "root differs")}
        except Exception as we:  # noqa: BLE001
            wc = {"equal": None, "self_consistent": None, "pin_of_record": None, "pinned": None, "note": f"by reference, unverified at replay: {we!r}"[:200]}
        sr, lk = evaluate(p, cfg, bmap=bmap, committed=rd, weights_of=lambda prefix: SR.model_weights_provider(st["eng"].model, prefix),
                          weights_check=wc, run_root=rc.run_root, manifest_doc=st.get("manifest_doc"))
    except Exception as e:  # noqa: BLE001  -- fail closed: an unrunnable replay is 'not evaluated', never PASS
        sr, lk = not_run(rank, e, traceback.format_exc())
    return base | {"sampled_replay": sr, "boundary_linkage": lk, "wall_s": round(time.perf_counter() - t0, 3)}


def defer(st: dict[str, Any], rc: Any, to: dict[str, Any]) -> dict[str, Any]:
    """`--replay-deferred`: in place of `in_process`, the rank's replay bundle is written into its partial dir under ``to`` ({"out",
    "pair"}; `replay_bundle.write_store`: the dumped store, the committed weights tree and the tensors no checkpoint holds; the binding
    map; the engine's facts of record) for the driver to seal, and the records are the deferred ones (`replay_bundle.deferred_records`).
    The rank's Programs are still loaded and checked here, so a Program that is not of record fails the rank now, and the cross-rank
    pass has the unreplayable sites it reads (`tp2_commit_xrank_dump`).  A rank that cannot defer is the in-process failure record."""
    from verity_vllm.pipeline import replay_bundle as RB

    cfg = st.get("replay") or {}
    rank, pair, out = int(st["rank"]), int(to["pair"]), Path(to["out"])
    t0 = time.perf_counter()
    base = {"schema": SCHEMA, "rank": rank, "method": METHOD}
    bmap = st.get("bmap")
    if not bmap:
        return base | no_map()
    try:
        from verity_vllm.check.replay.driver import engine_facts_of_record

        p = load(cfg, rank, st.get("world"))
        st["xrank_unreplayable_sites"] = unreplayable_sites(p)
        hashed: dict[str, str] = {}
        part = RB.write_store(out, pair, st["com"], rc, digests=hashed, rank=rank)
        (part / "binding_map.json").write_text(json.dumps(bmap, default=str) + "\n")
        sr, lk = RB.deferred_records(RB.path(out, pair, rank=rank))
        sr["rank"], lk["rank"] = rank, rank
        bundle = {"partial": str(part), "hashed": hashed, "engine_of_record": engine_facts_of_record(),
                  "programs": {k: {"dir": p.loaded[k].get("dir"), "program_digest": p.digs[k]} for k in p.loaded}}
    except Exception as e:  # noqa: BLE001  -- fail closed, as the in-process replay that could not run
        RB.discard(out, pair, rank=rank)
        (sr, lk), bundle = not_run(rank, e, traceback.format_exc()), None
    return base | {"sampled_replay": sr, "boundary_linkage": lk, "bundle": bundle, "wall_s": round(time.perf_counter() - t0, 3)}
