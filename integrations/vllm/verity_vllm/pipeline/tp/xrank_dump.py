"""[TP-12] One rank's picks for the cross-rank collective-output pass (`check.match.xrank_collectives`), from the COMMITTED bytes.

Every member is opened and verified against this rank's run root (the reader of the value check and the sampled replay); a member
that does not verify is a pick without bytes, named, which the driver's check fails.  Per pick: the rank's partial rows and the rows
of the received collective output as this rank committed it.  An all-reduce pick on a captured route also names the routed Definition
the rank Program binds at the call (`route_of`: `registry.tp_allreduce`'s route for the call's M and N under the engine's all-reduce
switches at the rank's world and GPU, the fused norm's when `fused_allreduce_rms_norm` takes FlashInfer's path), held to the specs the
rank's Programs bind
(`program_binds`); its received member is the norm's `<norm>/all_reduce` on the other path, and the fused norm's are `<norm>/0` and
`<norm>/1` with its residual (the previous call's `<norm>/1`, the embedding for the first) and the norm's weight.  Picks are drawn
identically on every rank from the pair's combined root (fixed after commitment) -> xrank_pair<k>_rank<r>.pt.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from verity_vllm.acquire.sources.allreduce_norm import FUSED, REDUCED_MEMBER
from verity_vllm.check.match import xrank_collectives as XC

ROUTED_WORLD = 8


def route_of(settings: dict[str, Any], M: int, N: int, world: int, norm: Any = None, device: Any = None) -> dict[str, Any] | None:
    """The routed Definition of one bf16 [M, N] all-reduce: {"spec", "family", "statics", "fused"}, {"why"} when no captured route
    covers it at the routed world, or None at world 2 and, off the routed world, where no captured route covers it (the AllReduce_v2
    chain).  `norm`: the RMSNorm a `fused_allreduce_rms_norm` call passes, which takes the fused route exactly where the export shim
    authors it (FlashInfer on, M <= `fused_max_tokens`, a captured entry).  `device`: the rank's GPU (`tp_allreduce.device_of`), which
    the routes keyed on a device read with the switches' NCCL transport."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import tp_allreduce as T

    world = int(world)
    if world <= 2:
        return None
    setting, route = T.served_route(settings, M, N)
    if setting is None:
        return {"why": f"all-reduce switches {settings!r} name no captured setting"} if world == ROUTED_WORLD else None
    facts = {"world": world, "device": device, "transport": settings.get("nccl_transport")}
    fused = None
    if norm is not None and settings.get("flashinfer") and M <= T.fused_max_tokens(N, world):
        fused = T.covered(setting, N, "fused", M, **facts)
    if fused is not None:
        defn, statics = T.AllReduceRmsNormMnnvl, {"WORLD": world, "N": N, "SHOT": T.mnnvl_shot(M, N, world), "CLUSTER": T.mnnvl_grid(M, N)[1],
                                                  "EPS": float(norm.variance_epsilon), "FMA": bool(fused.statics["FMA"])}
    else:
        c = T.covered(setting, N, route, M, **facts)
        if c is None:
            if world != ROUTED_WORLD:
                return None
            return {"why": f"tp_allreduce.covered() has no entry for setting={setting!r}, N={N}, route={route!r}, M={M}, world={world}, "
                           f"device={facts['device']!r}, transport={facts['transport']!r}"}
        defn, statics = T.route_binding(c, world, M, N)
    spec = bind(defn, **statics).id
    return {"spec": spec, "family": spec.split("{", 1)[0], "statics": statics, "fused": fused is not None}


def residual_member(order: list[str] | None, site: str, embedding: str | None) -> str | None:
    """The committed member a fused norm call reads as its residual: the previous call's new residual `<norm>/1`, the embedding's
    output for the first (DeepSeek-V3.2's layer 0 keeps the embedding as the residual)."""
    if not order or site not in order:
        return None
    k = order.index(site)
    return f"{order[k - 1]}/1" if k else embedding


def path_disagrees(route: dict[str, Any], took: str | None, order: list[str] | None, site: str) -> dict[str, Any] | None:
    """{"why"} when the norm source recorded this call taking the other path than `route` names (`AllReduceNorm.paths`: one
    character per call, in `order`); None when they agree or the step's paths were not kept."""
    if not took or not order or site not in order or len(took) != len(order):
        return None
    ch = took[order.index(site)]
    if (ch == FUSED) == bool(route["fused"]):
        return None
    return {"why": f"{site}: the call took path {ch}, the route names {'the fused norm' if route['fused'] else 'an all-reduce then the norm'}"}


def program_specs(progs: dict[str, dict[str, Any]]) -> list[str]:
    """The routed all-reduce specs the rank's Programs bind (`rank_replay.Loaded.progs`)."""
    from verity_vllm.query.manifest import format as RM

    return sorted({str(r["spec"]) for inst in progs.values() for r in inst.get("rows") or ()
                   if RM.spec_family(str(r.get("spec") or "")) in XC.ROUTED_FAMILIES})


def _routed(pk: dict[str, Any], route: dict[str, Any], site_mod: Any, order: list[str] | None, embedding: str | None,
            known: set | None) -> dict[str, Any]:
    """The pick's routed fields: the spec and what the driver reads beside the partial."""
    site = pk["site"]
    out = {"spec": route["spec"], "family": route["family"], "statics": route["statics"], "fused": route["fused"],
           "program_binds": (None if known is None else XC.spec_key(route["spec"]) in known)}
    if route["fused"]:
        out.update(recv_candidates=[f"{site}/0"], recv1_member=f"{site}/1", res_member=residual_member(order, site, embedding))
    elif site_mod is not None and hasattr(site_mod, "variance_epsilon"):
        out["recv_candidates"] = [f"{site}/{REDUCED_MEMBER}"]
    return out


def dump(st: dict[str, Any], *, settings: dict[str, Any], out_dir: str, seed_hex: str, pair: int, per_stratum: int = 2,
         max_rows: int = 16, device: Any = None) -> dict[str, Any]:
    """THIS rank's picks -> `out_dir/xrank_pair<pair>_rank<r>.pt`; the record the driver reads beside it.  `device`: `route_of`'s."""
    import torch

    from verity_vllm.acquire.sources.partial_source import _committed
    from verity_vllm.check import oracle_compare as OC

    com, rank, world = st["com"], int(st["rank"]), int(st["world"])
    t0 = time.perf_counter()
    layouts = {int(s): [(m.name, tuple(m.shape)) for m in ms] for s, ms in getattr(com, "_layouts", {}).items()}
    plan = XC.pick_strata(layouts, rank, bytes.fromhex(seed_hex), per_stratum=int(per_stratum), max_rows=int(max_rows), world=world)
    picks, missing, nbytes = [], 0, 0
    rd = OC.OpenedReader(com, st["rc"]) if st.get("rc") is not None else None
    unverified: dict[tuple[int, str], str] = {}
    mods = dict(st["eng"].model.named_modules())
    far = getattr(st.get("tps"), "far", None)
    order = list(getattr(far, "order", None) or []) or None
    paths = {int(s): str(p) for s, p in (getattr(far, "paths", None) or ())}
    embedding = next((n for n, m in mods.items() if XC.is_embedding_site(n) and hasattr(m, "shard_indices")), None)
    specs = st.get("xrank_program_specs")
    known = None if specs is None else {XC.spec_key(s) for s in specs}

    def tensor_of(step: int, meta) -> Any:
        if rd is None:
            unverified[(step, meta.name)] = "no run commitment on this rank (not finalized)"
            return None
        try:
            b = _committed(rd, step, meta)
        except OC.OpeningNotVerified as e:
            unverified[(step, meta.name)] = f"opened value not verified: {e}"[:200]
            return None
        if b is None:
            return None
        dt = getattr(torch, str(meta.dtype).replace("torch.", ""))
        return torch.frombuffer(bytearray(b), dtype=dt).reshape(tuple(meta.shape)).clone()

    for pk in plan["picks"]:
        step = int(pk["step"])
        metas = {m.name: m for m in com._layouts.get(step, [])}
        pm = metas.get(pk["member"])
        extra: dict[str, Any] = {}
        if pk["kind"] == "all_reduce" and pm is not None and len(pm.shape) != 2 and world == ROUTED_WORLD:
            extra = {"route_why": f"partial {pm.name} of shape {tuple(pm.shape)}, not [M, N]"}
        elif pk["kind"] == "all_reduce" and pm is not None and len(pm.shape) == 2:
            route = route_of(settings, int(pm.shape[0]), int(pm.shape[1]), world, norm=mods.get(pk["site"]) if order and pk["site"] in order else None,
                             device=device)
            if route is not None and "why" not in route:
                route = path_disagrees(route, paths.get(step), order, pk["site"]) or route
            if route is not None and "why" in route:
                extra = {"route_why": route["why"]}
            elif route is not None:
                extra = _routed(pk, route, mods.get(pk["site"]), order, embedding, known)
        pk = dict(pk, **extra)
        rname = next((n for n in pk["recv_candidates"] if n in metas), None)
        rm = metas.get(rname) if rname else None
        rows = torch.tensor(pk["rows"], dtype=torch.long)
        part = tensor_of(step, pm) if pm is not None else None
        recv = tensor_of(step, rm) if rm is not None else None
        why = None
        if part is None or recv is None:
            def _gone(meta) -> str:
                return unverified.get((step, meta.name), "bytes not retained")
            why = ("partial " + ("not in this step's layout" if pm is None else _gone(pm)) if part is None else "") + \
                  ("; " if part is None and recv is None else "") + \
                  (("received output " + ("not committed under any of " + "/".join(pk["recv_candidates"]) if rm is None else _gone(rm))) if recv is None else "")
            missing += 1
        else:
            if part.dim() >= 2 and recv.dim() >= 2 and part.shape[0] == recv.shape[0] == pk["rows_total"]:
                part, recv = part[rows], recv[rows]
            elif part.dim() >= 2 and recv.dim() >= 2 and part.shape[0] != recv.shape[0]:
                why = f"row counts differ: partial {tuple(part.shape)} vs received {tuple(recv.shape)}"
            nbytes += part.numel() * part.element_size() + recv.numel() * recv.element_size()
        if pk.get("fused"):
            for key in ("recv1", "res"):
                mn = pk.get(f"{key}_member")
                m_ = metas.get(mn) if mn else None
                t_ = tensor_of(step, m_) if m_ is not None else None
                pk[key] = t_[rows] if (t_ is not None and t_.dim() >= 2 and t_.shape[0] == pk["rows_total"]) else t_
                if t_ is None:
                    why = (why + "; " if why else "") + f"{key} {mn or '(no residual source)'} not committed"
            w_ = getattr(mods.get(pk["site"]), "weight", None)
            pk["w"] = w_.detach().to("cpu").clone() if w_ is not None else None
        emb = embedding_leg(pk, metas, step, rows, part, mods, tensor_of, unverified) if pk["kind"] == "all_reduce" and XC.is_embedding_site(pk["site"]) else None
        picks.append(dict(pk, part=part, recv=recv, recv_member=rname, why=why, emb=emb))
    rec = {"schema": "verity/tp2/xrank-picks/v1", "rank": rank, "pair": int(pair), "seed": seed_hex, "strata": plan["strata"], "picks": picks,
           "per_stratum": int(per_stratum), "max_rows": int(max_rows), "steps_in_layout": len(layouts)}
    fn = Path(out_dir) / f"xrank_pair{int(pair)}_rank{rank}.pt"
    torch.save(rec, fn)
    return {"rank": rank, "file": str(fn), "picks": len(picks), "strata": len(plan["strata"]), "missing": missing, "bytes": nbytes,
            "value_source": rd.record() if rd is not None else {"source": "opened", "reads": 0, "why": "no run commitment on this rank"},
            "unreplayable_sites": st.get("xrank_unreplayable_sites"), "strata_sites": sorted({v["site"] for v in plan["strata"].values()}),
            "families": sorted({p["family"] for p in picks}), "flipped_collective_inputs": int(getattr(st.get("tps"), "flipped", 0) or 0),
            "routed_specs": sorted({p["spec"] for p in picks if p.get("spec")}), "program_specs_n": (None if specs is None else len(specs)),
            "wall_s": round(time.perf_counter() - t0, 3)}


def embedding_leg(pk: dict[str, Any], metas: dict[str, Any], step: int, rows: Any, part: Any, mods: dict[str, Any], tensor_of: Any,
                  unverified: dict[tuple[int, str], str]) -> dict[str, Any]:
    """EmbeddingShard_v1 leg (ruling 27): this rank's committed partial rows == masked gather of the committed ids from ITS vocab shard
    (the live parameter, which the per-rank weights pin ties to the checkpoint of record); rows outside the rank's range are zero."""
    import torch

    try:
        im = metas.get(f"{pk['site']}/in")
        ids_t = tensor_of(step, im) if im is not None else None
        mod = mods.get(pk["site"])
        si = getattr(mod, "shard_indices", None)
        if ids_t is None or mod is None or si is None or part is None:
            return {"ok": None, "why": (((unverified.get((step, im.name)) if im is not None else None) or "no committed ids member <site>/in") if ids_t is None
                                        else "no vocabulary shard_indices on the module" if si is None else "partial not committed")}
        lo, hi = int(si.org_vocab_start_index), int(si.org_vocab_end_index)
        idx = ids_t.reshape(-1)
        idx = idx[rows] if (idx.numel() == pk["rows_total"] and part.shape[0] == len(pk["rows"])) else idx
        mask = (idx >= lo) & (idx < hi)
        local = torch.where(mask, idx - lo, torch.zeros_like(idx)).long()
        W = mod.weight
        exp = W[local.to(W.device)].masked_fill((~mask).unsqueeze(-1).to(W.device), 0).to("cpu")
        ok = bool(exp.shape == part.shape and exp.dtype == part.dtype and torch.equal(exp, part))
        return {"ok": ok, "n_diff": (int((exp.view(torch.int16) != part.view(torch.int16)).sum()) if (exp.shape == part.shape and exp.dtype == torch.bfloat16) else None),
                "vocab_range": [lo, hi], "masked_rows": int((~mask).sum()), "rows": int(idx.numel()), "why": None if ok else "committed partial != masked gather from this rank's shard"}
    except Exception as e:  # noqa: BLE001
        return {"ok": None, "why": f"embedding-shard leg error: {e!r}"[:200]}


__all__ = ["dump", "route_of", "residual_member", "path_disagrees", "program_specs", "embedding_leg", "ROUTED_WORLD"]
