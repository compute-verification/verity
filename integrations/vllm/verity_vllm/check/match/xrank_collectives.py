"""[R16 tp TP-12, rev F-r16-18] CROSS-RANK REPLAY OF THE COLLECTIVE OUTPUTS from the two ranks' COMMITTED partials.

What TP adds to a Commit is the all-reduce / all-gather OUTPUT every layer consumes; at TP2 the per-rank sampled replay evaluates the rank
partial (`Gemm_v1{K/W}` on the shard) but the collective output itself had no registered evaluator (`AllReduce2_v1`, `AllGather2_v1`,
`EmbeddingShard_v1` = 30 % of the VU population at B1) -- in Commit it was checked only by the same-process hook-time clone and by
tp2_match on Match's stored steps.  This pass evaluates it bit-exactly from what the RANKS COMMITTED (bound + opened members, the bytes
under the run roots):

    all_reduce : committed <site>/0 | /out (rank r)  ==  bf16(f32(committed part_rank0) + f32(committed part_rank1))   for BOTH ranks, and equal across ranks
    all_gather : committed <site>/0 (rank r)          ==  concat(committed shard_rank0, committed shard_rank1) along the last dim
    embedding  : the VocabParallelEmbedding all-reduce of the masked partials -- the all_reduce rule on the `embed_tokens` site (EmbeddingShard_v1)

Picks are STRATIFIED per (site, kind, phase) and drawn from a seed fixed AFTER commitment (the pair's combined root), identically on every
rank (the rank suffix is stripped from the stratum key), with a row subset per pick so a prefill tensor costs a few rows, not 32 MB.  Each
rank dumps its picks (partial rows + received rows, from the committed bytes) to `xrank_rank<r>.pt`; the driver pairs them.  Any pick whose
received member is not committed on a rank, or whose partial is not committed, is a named FAIL (the population of record requires both).
Negative: `--flip-partial <rank>:<site suffix>` makes that rank commit a partial that is not the collective's input -> this pass fails by name.
"""
from __future__ import annotations

import hashlib
from typing import Any

FAMILY = {"all_reduce": "AllReduce2_v1", "all_gather": "AllGather2_v1"}           # the two-rank identities (world 2)


def collective_family(kind: str, world: int) -> str:
    """[lane vllm-tp-n] the received-output Definition family of a `kind` collective in a `world`-rank communicator, from the registry's own
    selector (`b1_tp2.all_reduce` / `all_gather`): `AllReduce2_v1` / `AllGather2_v1` at world 2, `AllReduce_v2` / `AllGather_v1` beyond."""
    from verity.ml.circuits.ops import b1_tp2 as TP2
    sel = {"all_reduce": TP2.all_reduce, "all_gather": TP2.all_gather}.get(kind)
    return sel(int(world), 1).id.split("{", 1)[0] if sel else kind
EMBED_MARKERS = ("embed_tokens", "embedding", "wte")
RECV_MEMBERS = ("out", "0", "")       # the received collective output as the committer names it: MoE block `<site>/out`, dense tuple output `<site>/0`,
                                      # a single-tensor module output (VocabParallelEmbedding, LogitsProcessor) = the BARE module name (binding.split_name -> member "out")
MAX_ROWS = 16


def strip_rank(name: str) -> str:
    """`<site>/part_rank1` -> `<site>/part_rank*` (the stratum key shared by every rank)."""
    head, _, mem = name.rpartition("/")
    if mem.startswith(("part_rank", "shard_rank")):
        mem = mem.split("_rank")[0] + "_rank*"
    return f"{head}/{mem}"


def family_of(site: str, kind: str, world: int = 2) -> str:
    """The received-output family of a site (the all-reduce at the embedding site is an AllReduce2 output too; the EmbeddingShard_v1 leg --
    the rank's PARTIAL == masked gather from its pinned vocab shard -- is evaluated on the rank and carried in the pick as `emb`)."""
    return collective_family(kind, world)


def is_embedding_site(site: str) -> bool:
    return any(m in site for m in EMBED_MARKERS)


def _h(*parts: str | bytes) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big")


def pick_strata(layouts: dict[int, list[tuple[str, tuple[int, ...]]]], rank: int, seed: bytes, per_stratum: int = 2, max_rows: int = MAX_ROWS,
                world: int = 2) -> dict[str, Any]:
    """Deterministic, rank-agnostic picks.  `layouts`: step -> [(member name, sizes)] of THIS rank's committed layout (only the rank's own
    partial members matter here).  Returns {"strata": {key: {...}}, "picks": [{step, site, kind, member, recv_candidates, rows, family, stratum}]}."""
    suf = (f"/part_rank{rank}", f"/shard_rank{rank}")
    by_site: dict[tuple[str, str], list[tuple[int, tuple[int, ...]]]] = {}
    for step, metas in layouts.items():
        for name, sizes in metas:
            if name.endswith(suf):
                site, mem = name.rsplit("/", 1)
                kind = "all_reduce" if mem.startswith("part_rank") else "all_gather"
                by_site.setdefault((site, kind), []).append((int(step), tuple(int(x) for x in sizes)))
    strata: dict[str, dict[str, Any]] = {}
    picks: list[dict[str, Any]] = []
    for (site, kind), lst in sorted(by_site.items()):
        rows_of = {s: (sz[0] if sz else 1) for s, sz in lst}
        mx, mn = max(rows_of.values()), min(rows_of.values())
        groups: dict[str, list[int]] = {}
        for s, _sz in sorted(lst):
            phase = "prefill" if (mx > mn and rows_of[s] == mx) else ("decode" if mx > mn else "all")
            groups.setdefault(phase, []).append(s)
        for phase, steps in groups.items():
            key = f"{site}|{kind}|{phase}"
            steps = sorted(set(steps))
            n = min(per_stratum, len(steps))
            # seeded choice without replacement: rank the steps by a hash of (seed, key, step)
            order = sorted(steps, key=lambda s: _h(seed, key, str(s)))
            chosen = sorted(order[:n])
            strata[key] = {"site": site, "kind": kind, "phase": phase, "family": family_of(site, kind, world), "steps_n": len(steps), "picked": chosen}
            for s in chosen:
                rows = rows_of[s]
                if rows > max_rows:
                    ridx = sorted({_h(seed, key, str(s), str(i)) % rows for i in range(max_rows * 2)})[:max_rows]
                    ridx = sorted(set(ridx))
                else:
                    ridx = list(range(rows))
                picks.append({"step": s, "site": site, "kind": kind, "member": f"{site}/{'part' if kind == 'all_reduce' else 'shard'}_rank{rank}",
                              "recv_candidates": [f"{site}/{m}" if m else site for m in RECV_MEMBERS], "rows": ridx, "rows_total": rows, "family": family_of(site, kind, world), "stratum": key})
    return {"strata": strata, "picks": picks, "per_stratum": per_stratum, "max_rows": max_rows, "seed": seed.hex() if isinstance(seed, (bytes, bytearray)) else str(seed)}


def check_picks(dumps: dict[int, dict[str, Any]], world: int = 2) -> dict[str, Any]:
    """Driver side.  `dumps[rank]` = torch.load(xrank_rank<r>.pt): {"rank", "picks": [{step, site, kind, family, stratum, rows, part: Tensor|None,
    recv: Tensor|None, recv_member, why}]}.  Same pick order on every rank (same seed, same layout names).  Bit-exact rules per kind."""
    import torch

    from verity.ml.circuits.ops.b1_tp2 import allreduce_order

    res: dict[str, Any] = {"result": None, "world": world, "picks": 0, "equal": 0, "mismatches": [], "missing": [], "by_family": {}, "by_stratum": {},
                           "rule": ("per pick (site, step, row subset): all_reduce -> committed received (each rank) == bf16(f32(part_rank0) + f32(part_rank1)) elementwise "
                                    "and equal across ranks; all_gather -> committed received == concat(shard_rank0, shard_rank1) on the last dim; "
                                    "embedding site additionally (EmbeddingShard_v1) -> each rank's committed partial == masked gather of the committed ids (<site>/in) "
                                    "from the rank's pinned vocab shard (rows outside the rank's vocab range are zero); "
                                    "partials and received are the ranks' COMMITTED bytes (bound + under the run roots), never the hook-time clone")}
    ranks = sorted(dumps)
    if len(ranks) != world:
        res["result"], res["why"] = False, f"{len(ranks)}/{world} rank dumps"
        return res
    n = min(len(dumps[r]["picks"]) for r in ranks)
    if any(len(dumps[r]["picks"]) != n for r in ranks):
        res["missing"].append({"why": f"pick lists differ across ranks: {[len(dumps[r]['picks']) for r in ranks]}"})
    for i in range(n):
        ps = [dumps[r]["picks"][i] for r in ranks]
        p0 = ps[0]
        key = (p0["site"], int(p0["step"]), p0["kind"])
        if any((p["site"], int(p["step"]), p["kind"]) != key for p in ps):
            res["missing"].append({"pick": i, "why": f"pick {i} names different (site, step, kind) across ranks", "keys": [(p["site"], p["step"], p["kind"]) for p in ps]})
            continue
        fam = res["by_family"].setdefault(p0["family"], {"picks": 0, "equal": 0, "mismatch": 0, "missing": 0})
        st = res["by_stratum"].setdefault(p0["stratum"], {"picks": 0, "equal": 0})
        fam["picks"] += 1
        st["picks"] += 1
        res["picks"] += 1
        if p0["kind"] == "all_reduce" and is_embedding_site(p0["site"]):
            # EmbeddingShard_v1 leg (ruling 27): each rank's committed partial rows == masked gather of the committed ids from the rank's pinned
            # vocab shard (evaluated on the rank: `emb` = {ok, n_diff, vocab_range, masked_rows, why}); missing on a rank = named FAIL
            ef = res["by_family"].setdefault("EmbeddingShard_v1", {"picks": 0, "equal": 0, "mismatch": 0, "missing": 0})
            ef["picks"] += 1
            res["picks"] += 1
            embs = [p.get("emb") for p in ps]
            if any(e is None or e.get("ok") is None for e in embs):
                ef["missing"] += 1
                res["missing"].append({"pick": i, "site": p0["site"], "step": p0["step"], "kind": "embedding_shard",
                                       "why": "; ".join(f"rank {r}: embedding-shard leg not evaluable ({(embs[j] or {}).get('why') or 'no emb record'})" for j, r in enumerate(ranks) if embs[j] is None or embs[j].get("ok") is None)})
            elif all(e.get("ok") for e in embs):
                ef["equal"] += 1
                res["equal"] += 1
            else:
                ef["mismatch"] += 1
                res["mismatches"].append({"pick": i, "site": p0["site"], "step": p0["step"], "kind": "embedding_shard", "family": "EmbeddingShard_v1", "rows": p0["rows"][:8],
                                          "received_equals_definition_per_rank": {str(r): bool(embs[j].get("ok")) for j, r in enumerate(ranks)}, "received_equal_across_ranks": None,
                                          "n_diff_per_rank": {str(r): embs[j].get("n_diff") for j, r in enumerate(ranks)}, "vocab_range_per_rank": {str(r): embs[j].get("vocab_range") for j, r in enumerate(ranks)}})
        parts = [p.get("part") for p in ps]
        recvs = [p.get("recv") for p in ps]
        if any(x is None for x in parts) or any(x is None for x in recvs):
            fam["missing"] += 1
            res["missing"].append({"pick": i, "site": p0["site"], "step": p0["step"], "kind": p0["kind"],
                                   "why": "; ".join(f"rank {r}: {'partial' if parts[j] is None else ''}{' and ' if parts[j] is None and recvs[j] is None else ''}{'received' if recvs[j] is None else ''} not committed"
                                                    + (f" ({ps[j].get('why')})" if ps[j].get("why") else "") for j, r in enumerate(ranks) if parts[j] is None or recvs[j] is None)})
            continue
        if p0["kind"] == "all_reduce":
            # the Definition's chain: world 2 = one bf16(f32(a) + f32(b)); beyond, one rounding per hop in allreduce_order (AllReduce_v2{WORLD,N})
            order = allreduce_order(len(parts))
            exp = parts[order[0]]
            for j in order[1:]:
                exp = (exp.float() + parts[j].float()).to(parts[0].dtype)
        elif p0["kind"] == "all_gather":
            exp = torch.cat(parts, dim=-1)
        else:
            res["missing"].append({"pick": i, "site": p0["site"], "step": p0["step"], "why": f"unbound collective kind {p0['kind']}"})
            fam["missing"] += 1
            continue
        eq = [bool(r_.shape == exp.shape and torch.equal(r_, exp)) for r_ in recvs]
        same = all(bool(recvs[0].shape == r_.shape and torch.equal(recvs[0], r_)) for r_ in recvs[1:])
        if all(eq) and same:
            fam["equal"] += 1
            st["equal"] += 1
            res["equal"] += 1
        else:
            fam["mismatch"] += 1
            diffs = [int((r_.view(torch.int16) != exp.view(torch.int16)).sum()) if (r_.shape == exp.shape and r_.dtype == torch.bfloat16) else -1 for r_ in recvs]
            res["mismatches"].append({"pick": i, "site": p0["site"], "step": p0["step"], "kind": p0["kind"], "family": p0["family"], "rows": p0["rows"][:8],
                                      "received_equals_definition_per_rank": dict(zip([str(r) for r in ranks], eq)), "received_equal_across_ranks": same,
                                      "n_diff_per_rank": dict(zip([str(r) for r in ranks], diffs)), "elements": int(exp.numel())})
    res["mismatch_n"], res["missing_n"] = len(res["mismatches"]), len(res["missing"])
    res["mismatches"], res["missing"] = res["mismatches"][:20], res["missing"][:20]
    res["result"] = bool(res["picks"] > 0 and res["mismatch_n"] == 0 and res["missing_n"] == 0)
    if not res["result"]:
        first = (res["mismatches"] or res["missing"] or [{}])[0]
        res["why"] = (f"{res['mismatch_n']} pick(s) where a committed received output != the definition over the committed partials, {res['missing_n']} not evaluable"
                      f" -- first: {first.get('site')} step {first.get('step')} {first.get('kind') or ''} {first.get('why') or ''}") if res["picks"] else "no collective picks (no rank partials committed?)"
    return res


def fold_pairs(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    """The component over instrumented pairs: True only when every pair's pass is True."""
    rs = [p.get("result") for p in per_pair]
    fam: dict[str, dict[str, int]] = {}
    for p in per_pair:
        for f, c in (p.get("by_family") or {}).items():
            d = fam.setdefault(f, {"picks": 0, "equal": 0, "mismatch": 0, "missing": 0})
            for k in d:
                d[k] += int(c.get(k) or 0)
    return {"result": (all(r is True for r in rs) if rs and all(r is not None for r in rs) else (False if any(r is False for r in rs) else None)),
            "pairs": len(per_pair), "picks": sum(int(p.get("picks") or 0) for p in per_pair), "equal": sum(int(p.get("equal") or 0) for p in per_pair),
            "mismatch_n": sum(int(p.get("mismatch_n") or 0) for p in per_pair), "missing_n": sum(int(p.get("missing_n") or 0) for p in per_pair),
            "by_family": fam, "families": sorted(fam), "why": next((p.get("why") for p in per_pair if p.get("result") is not True and p.get("why")), None),
            "coverage_of_unreplayable": {"complete": (all(((p.get("coverage_of_unreplayable") or {}).get("complete") is True) for p in per_pair) if per_pair else None),
                                         "per_pair": [(p.get("coverage_of_unreplayable") or {}) for p in per_pair]},
            "rule": (per_pair[0].get("rule") if per_pair else None), "seed": "the pair's combined run root (fixed after commitment)"}


def coverage_of_unreplayable(res: dict[str, Any], dumps_meta: list[dict[str, Any]]) -> dict[str, Any]:
    """[rev M-0418 hardening] the pass's strata SITES against the sites of the replay-unreplayable collective families named by the ranks'
    Programs (`unreplayable_sites` from each rank dump: family -> [op_path]).  A family site with no stratum is a NAMED gap (a received
    output committed without a committed partial, a collective kind the partial hook missed): the pass result becomes False by name.
    Field read for the finalizer: coverage_of_unreplayable.complete."""
    sites_by_fam: dict[str, set] = {}
    errors = []
    for d in dumps_meta:
        us = d.get("unreplayable_sites")
        if not isinstance(us, dict):
            errors.append(f"rank {d.get('rank')}: no unreplayable_sites")
            continue
        if "error" in us:
            errors.append(f"rank {d.get('rank')}: {us['error']}")
            continue
        for f, ss in us.items():
            sites_by_fam.setdefault(f, set()).update(str(x) for x in (ss or []))
    strata_sites = {k.split("|")[0] for k in (res.get("by_stratum") or {})}

    def covered(site: str) -> bool:
        return site in strata_sites or any(x.endswith("." + site) or site.endswith("." + x) for x in strata_sites)

    fams = {}
    for f, ss in sorted(sites_by_fam.items()):
        miss = sorted(x for x in ss if not covered(x))
        fams[f] = {"sites": len(ss), "covered": len(ss) - len(miss), "missing": miss,
                   "picks": ((res.get("by_family") or {}).get(f) or {}).get("picks"), "equal": ((res.get("by_family") or {}).get(f) or {}).get("equal")}
    complete = bool(sites_by_fam) and not errors and all(not v["missing"] for v in fams.values())
    out = {"complete": complete, "families": fams, "strata_sites_n": len(strata_sites), "errors": errors or None,
           "rule": "every site of every replay-unreplayable collective family (from the rank Programs) has >= 1 stratum in this pass; else missing by name"}
    if not complete:
        why = ("; ".join(errors) if errors else ("no unreplayable collective sites named by the Programs" if not sites_by_fam else
               "sites without a stratum: " + ", ".join(f"{f}: {v['missing'][:4]}" for f, v in fams.items() if v["missing"])))
        out["why"] = why
        res["result"] = False
        res["why"] = f"coverage of the unreplayable collective sites incomplete -- {why}" + (f"; also: {res.get('why')}" if res.get("why") else "")
    res["coverage_of_unreplayable"] = out
    return out

