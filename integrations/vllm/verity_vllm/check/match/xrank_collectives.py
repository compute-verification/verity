"""[TP-12] CROSS-RANK REPLAY OF THE COLLECTIVE OUTPUTS from the two ranks' COMMITTED partials.

What TP adds to a Commit is the all-reduce / all-gather OUTPUT every layer consumes; at TP2 the per-rank sampled replay evaluates the rank
partial (`Gemm_v1{K/W}` on the shard) but the collective output itself had no registered evaluator (`AllReduce2_v1`, `AllGather2_v1`,
`EmbeddingShard_v1` = 30 % of the VU population at B1) -- in Commit it was checked only by the same-process hook-time clone and by
tp2_match on Match's stored steps.  This pass evaluates it bit-exactly from what the RANKS COMMITTED (bound + opened members, the bytes
under the run roots):

    all_reduce : committed <site>/0 | /out (rank r)  ==  bf16(f32(committed part_rank0) + f32(committed part_rank1))   for BOTH ranks, and equal across ranks
    all_gather : committed <site>/0 (rank r)          ==  concat(committed shard_rank0, committed shard_rank1) along the last dim
    embedding  : the VocabParallelEmbedding all-reduce of the masked partials -- the all_reduce rule on the `embed_tokens` site (EmbeddingShard_v1)

Picks are STRATIFIED per (site, kind, phase), plus one stratum per routed family a site's calls take that its phase strata missed, and
sampled from a seed fixed AFTER commitment (the pair's combined root), identically on every
rank (the rank suffix is stripped from the stratum key), with a row subset per pick so a prefill tensor costs a few rows, not 32 MB.  Each
rank dumps its picks (partial rows + received rows, from the committed bytes) to `xrank_rank<r>.pt`; the driver pairs them.  Any pick whose
received member is not committed on a rank, or whose partial is not committed, is a named FAIL (the population of record requires both).
Negative: `--flip-partial <rank>:<site suffix>` makes that rank commit a partial that is not the collective's input -> this pass fails by name.

An all-reduce on a captured route is evaluated with the routed Definition the rank Program binds at the call (`registry.tp_allreduce`):
the rank names it per pick (`spec`, `statics`, `pipeline.tp.xrank_dump.route_of`; over several requests the one call per request the
Build binds on it, `calls`) and the driver evaluates it with `tp_allreduce_check`'s models, which tests/program/test_tp_allreduce.py
holds to the Definitions:

    AllReduceMultimem_v1 / AllReduceCustom_v1 : the received rows == each flat element's owner sum (row * N + col of the M * N words)
    AllReduceMnnvl_v1                         : the received rows == the elementwise FTZ sum, ranks ascending
    AllReduceNcclTree_v1                      : the received rows == the elementwise bf16 sum down CHAIN from its leaf
    AllReduceRmsNormMnnvl_v1                  : committed <norm>/0 == normed and <norm>/1 == bf16(a + res), from the committed partials,
                                                the committed residual (`res`) and the norm's weight (`w`), each rank its own
"""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

import numpy as np

FAMILY = {"all_reduce": "AllReduce2_v1", "all_gather": "AllGather2_v1"}           # the two-rank identities (world 2)


def collective_family(kind: str, world: int) -> str:
    """The received-output Definition family of a `kind` collective in a `world`-rank communicator, from the registry's own
    selector (`b1_tp2.all_reduce` / `all_gather`): `AllReduce2_v1` / `AllGather2_v1` at world 2, `AllReduce_v2` / `AllGather_v1` beyond."""
    from verity.ml.definitions.registry import b1_tp2 as TP2
    sel = {"all_reduce": TP2.all_reduce, "all_gather": TP2.all_gather}.get(kind)
    return sel(int(world), 1).id.split("{", 1)[0] if sel else kind
EMBED_MARKERS = ("embed_tokens", "embedding", "wte")
RECV_MEMBERS = ("out", "0", "")       # the received collective output as the committer names it: MoE block `<site>/out`, dense tuple output `<site>/0`,
                                      # a single-tensor module output (VocabParallelEmbedding, LogitsProcessor) = the BARE module name (binding.split_name -> member "out")
MAX_ROWS = 16


def family_of(site: str, kind: str, world: int = 2) -> str:
    """The received-output family of a site (the all-reduce at the embedding site is an AllReduce2 output too; the EmbeddingShard_v1 leg --
    the rank's PARTIAL == masked gather from its pinned vocab shard -- is evaluated on the rank and carried in the pick as `emb`)."""
    return collective_family(kind, world)


def is_embedding_site(site: str) -> bool:
    return any(m in site for m in EMBED_MARKERS)


FUSED_FAMILY = "AllReduceRmsNormMnnvl_v1"
NCCL_TREE_FAMILY = "AllReduceNcclTree_v1"
ROUTED_FAMILIES = ("AllReduceMnnvl_v1", "AllReduceMultimem_v1", "AllReduceCustom_v1", NCCL_TREE_FAMILY, FUSED_FAMILY)


def _canon(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(sorted((str(k), _canon(x)) for k, x in v.items()))
    if isinstance(v, (set, frozenset)):
        return tuple(sorted(_canon(x) for x in v))
    return tuple(_canon(x) for x in v) if isinstance(v, (list, tuple)) else v


def spec_key(spec: str) -> tuple[str, tuple]:
    """(family, sorted (static, value)) of a bound Definition id such as ``AllReduceMultimem_v1{WORLD=8,SIZE=..,TREES=((..),..),FTZ=False}``:
    the statics split at top-level commas, each value the descriptor's JSON (``EPS={"f64":"0x1.4f8b588e368f1p-17"}``, ``FMA=true``,
    as a Program's rows spell it) or else a Python literal (``bind(..).id``'s ``EPS=1e-05``, nested tuples), so both spell one key."""
    from verity.ml.definitions.call_scope import static_value

    fam, _, rest = str(spec).partition("{")
    body = rest[:-1] if rest.endswith("}") else rest
    items, depth, cur = [], 0, ""
    for ch in body:
        if ch == "," and depth == 0:
            items.append(cur)
            cur = ""
            continue
        depth += (ch in "([{") - (ch in ")]}")
        cur += ch
    if cur.strip():
        items.append(cur)
    out = []
    for it in items:
        k, _, v = it.partition("=")
        try:
            val = static_value(json.loads(v.strip()))
        except ValueError:
            try:
                val = ast.literal_eval(v.strip())
            except (ValueError, SyntaxError):
                val = v.strip()
        out.append((k.strip(), _canon(val)))
    return fam.strip(), tuple(sorted(out))


def words(t: Any) -> np.ndarray:
    """The uint16 words of a bf16 CPU tensor."""
    import torch

    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def routed_expected(family: str, statics: dict[str, Any], parts: np.ndarray, rows: list[int], res: np.ndarray | None = None,
                    w: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """The routed Definition's words at the picked rows: `parts` (WORLD, R, N) the ranks' partial rows `rows` of an M x N call (the flat
    M * N words a Multimem / Custom call reads, row-major).  Returns (out,), or (normed, r) for the fused norm (`res` (R, N), `w` (N))."""
    from verity.ml.definitions.numerics import derived_rows as DR
    from verity.ml.definitions.registry import tp_allreduce as T
    from verity.ml.definitions.registry import tp_allreduce_check as TC

    parts = np.asarray(parts, np.uint16)
    W, R, N = parts.shape
    if family == "AllReduceMnnvl_v1":
        return (TC.mnnvl_rows(parts, int(statics["SHOT"])),)
    if family == NCCL_TREE_FAMILY:
        return (TC.nccl_tree_rows(parts, T.nccl_chain(statics["CHAIN"], W)),)
    if family == FUSED_FAMILY:
        if res is None or w is None:
            raise ValueError("the fused all-reduce norm reads the residual and the weight")
        return TC.mnnvl_norm_rows(parts, np.asarray(res, np.uint16), np.asarray(w, np.uint16), int(statics["SHOT"]), int(statics["CLUSTER"]),
                                  float(statics["EPS"]), bool(statics["FMA"]))
    size = int(statics["SIZE"])
    flat = np.asarray(rows, np.int64)[:, None] * N + np.arange(N, dtype=np.int64)[None, :]
    if int(flat.max(initial=0)) >= size:
        raise ValueError(f"rows {max(rows)} x N={N} past SIZE={size}")
    out = np.empty((R, N), np.uint16)
    if family == "AllReduceMultimem_v1":
        trees = T.multimem_trees(statics["TREES"], W)
        for lo, hi, owner in T.multimem_owners(W, size):
            m = (flat >= lo) & (flat < hi)
            if m.any():
                out[m] = DR.f2fp_bf16(TC.tree_sum(parts[:, m], trees[owner], bool(statics["FTZ"])))
        return (out,)
    if family == "AllReduceCustom_v1":
        for lo, hi, owner in T.custom_owners(W, size, int(statics["STAGE"])):
            m = (flat >= lo) & (flat < hi)
            if m.any():
                order = T.custom_order(W, owner)
                acc = DR.bf16_to_f32_words(parts[order[0]][m])
                for r in order[1:]:
                    acc = DR.add_rn(acc, DR.bf16_to_f32_words(parts[r][m]))
                out[m] = DR.f2fp_bf16(acc)
        return (out,)
    raise ValueError(f"{family}: not a routed all-reduce")


def calls_expected(family: str, statics: dict[str, Any], calls: list[dict[str, Any]] | None, parts: np.ndarray,
                   rows: list[int]) -> tuple[np.ndarray, ...]:
    """`routed_expected` over a launch the Build binds as one call per request (`calls` [{lo, rows, statics}], each over its own rows
    from its first); None = one call over the launch."""
    if not calls:
        return routed_expected(family, statics, parts, rows)
    rs, out = np.asarray(rows, np.int64), np.empty(np.shape(parts)[1:], np.uint16)
    done = np.zeros(len(rs), bool)
    for c in calls:
        sel = np.flatnonzero((rs >= c["lo"]) & (rs < c["lo"] + c["rows"]))
        if sel.size:
            out[sel] = routed_expected(family, c["statics"], np.asarray(parts)[:, sel], (rs[sel] - c["lo"]).tolist())[0]
            done[sel] = True
    if not done.all():
        raise ValueError(f"rows {rs[~done][:4].tolist()} are in no request's call {[(c['lo'], c['rows']) for c in calls]}")
    return (out,)


def _check_routed(ps: list[dict[str, Any]], ranks: list[int]) -> tuple[bool | None, dict[str, Any]]:
    """One routed pick over every rank's dump: (True | False | None = not evaluable, detail)."""
    specs = {(str(p.get("spec")), tuple(str(c["spec"]) for c in p.get("calls") or ())) for p in ps}
    if len(specs) != 1:
        return None, {"why": f"the ranks bind different Definitions at the call: {sorted((s[:80], len(c)) for s, c in specs)}"}
    unbound = {r: p for p, r in zip(ps, ranks) if p.get("program_binds") is not True}
    if unbound:
        return None, {"why": "; ".join(f"rank {r}: " + ("the rank Program binds no " + str(p.get("unbound_spec") or p["spec"])[:120]
                                                         if p.get("program_binds") is False else "the rank Programs' routed specs were not read")
                                       for r, p in unbound.items())}
    p0 = ps[0]
    fam, st, fused = spec_key(p0["spec"])[0], dict(p0["statics"]), p0["spec"].startswith(FUSED_FAMILY)
    need = ("part", "recv", "recv1", "res", "w") if fused else ("part", "recv")
    gone = [f"rank {r}: {k}" for p, r in zip(ps, ranks) for k in need if p.get(k) is None]
    if gone:
        return None, {"why": "; ".join(gone[:6]) + " not committed" + "".join(f" ({p['why']})" for p in ps if p.get("why"))[:300]}
    parts = np.stack([words(p["part"]) for p in ps])
    if parts.ndim != 3:
        return None, {"why": f"partials of shape {parts.shape[1:]}; expected [rows, N]"}
    try:
        if fused:
            exps = [routed_expected(fam, st, parts, p0["rows"], words(p["res"]), words(p["w"]).reshape(-1)) for p in ps]
        else:
            exps = [calls_expected(fam, st, p0.get("calls"), parts, p0["rows"])] * len(ps)
    except ValueError as e:
        return None, {"why": f"{fam}: {e}"}
    got = [(words(p["recv"]),) + ((words(p["recv1"]),) if fused else ()) for p in ps]
    eq = [all(g.shape == e.shape and np.array_equal(g, e) for g, e in zip(gs, es)) for gs, es in zip(got, exps)]
    same = all(all(np.array_equal(a, b) for a, b in zip(got[0], g)) for g in got[1:])
    diffs = [sum(int((g != e).sum()) if g.shape == e.shape else -1 for g, e in zip(gs, es)) for gs, es in zip(got, exps)]
    return bool(all(eq) and same), {"received_equals_definition_per_rank": dict(zip([str(r) for r in ranks], eq)), "received_equal_across_ranks": same,
                                     "n_diff_per_rank": dict(zip([str(r) for r in ranks], diffs)), "elements": int(exps[0][0].size), "spec": p0["spec"][:160],
                                     **({"request_calls": [(c["lo"], c["rows"]) for c in p0["calls"]]} if p0.get("calls") else {})}


def _h(*parts: str | bytes) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big")


def pick_strata(layouts: dict[int, list[tuple[str, tuple[int, ...]]]], rank: int, seed: bytes, per_stratum: int = 2, max_rows: int = MAX_ROWS,
                world: int = 2, call_family: Any = None) -> dict[str, Any]:
    """Deterministic, rank-agnostic picks.  `layouts`: step -> [(member name, sizes)] of THIS rank's committed layout (only the rank's own
    partial members matter here).  `call_family(site, kind, sizes)`: the routed family a call takes (`xrank_dump.call_family`); a family
    a site's calls take that none of its phase strata drew gets a stratum of its own (`<site>|<kind>|route=<family>`), so every routed
    Definition a site binds is evaluated there.  Returns {"strata": {key: {...}}, "picks": [{step, site, kind, member, recv_candidates,
    rows, family, stratum}]}."""
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
        fam = family_of(site, kind, world)
        groups: dict[str, list[int]] = {}
        for s, _sz in sorted(lst):
            phase = "prefill" if (mx > mn and rows_of[s] == mx) else ("decode" if mx > mn else "all")
            groups.setdefault(phase, []).append(s)
        draws = {f"{site}|{kind}|{phase}": (phase, fam, steps, _draw(seed, f"{site}|{kind}|{phase}", steps, rows_of, per_stratum, max_rows))
                 for phase, steps in groups.items()}
        if call_family is not None:
            route_at = {s: call_family(site, kind, sz) for s, sz in sorted(lst)}
            taken = {route_at[s] for d in draws.values() for s, _r in d[3]}
            for f in sorted({f for f in route_at.values() if f} - taken):
                key, steps = f"{site}|{kind}|route={f}", [s for s, f_ in route_at.items() if f_ == f]
                draws[key] = (f"route={f}", f, steps, _draw(seed, key, steps, rows_of, per_stratum, max_rows))
        for key, (phase, family, steps, chosen) in draws.items():
            strata[key] = {"site": site, "kind": kind, "phase": phase, "family": family, "steps_n": len(set(steps)), "picked": [s for s, _r in chosen]}
            for s, ridx in chosen:
                picks.append({"step": s, "site": site, "kind": kind, "member": f"{site}/{'part' if kind == 'all_reduce' else 'shard'}_rank{rank}",
                              "recv_candidates": [f"{site}/{m}" if m else site for m in RECV_MEMBERS], "rows": ridx, "rows_total": rows_of[s], "family": fam, "stratum": key})
    return {"strata": strata, "picks": picks, "per_stratum": per_stratum, "max_rows": max_rows, "seed": seed.hex() if isinstance(seed, (bytes, bytearray)) else str(seed)}


def _draw(seed: bytes, key: str, steps: list[int], rows_of: dict[int, int], per_stratum: int, max_rows: int) -> list[tuple[int, list[int]]]:
    """A stratum's picks: up to `per_stratum` of its steps, each with at most `max_rows` of its rows, by the seed's hash of (key, step)."""
    steps = sorted(set(steps))
    # seeded choice without replacement: rank the steps by a hash of (seed, key, step)
    order = sorted(steps, key=lambda s: _h(seed, key, str(s)))
    out = []
    for s in sorted(order[:min(per_stratum, len(steps))]):
        rows = rows_of[s]
        out.append((s, sorted({_h(seed, key, str(s), str(i)) % rows for i in range(max_rows * 2)})[:max_rows] if rows > max_rows else list(range(rows))))
    return out


def check_picks(dumps: dict[int, dict[str, Any]], world: int = 2) -> dict[str, Any]:
    """Driver side.  `dumps[rank]` = torch.load(xrank_rank<r>.pt): {"rank", "picks": [{step, site, kind, family, stratum, rows, part: Tensor|None,
    recv: Tensor|None, recv_member, why}]}.  Same pick order on every rank (same seed, same layout names).  Bit-exact rules per kind."""
    import torch

    from verity.ml.definitions.registry.b1_tp2 import allreduce_order

    res: dict[str, Any] = {"result": None, "world": world, "picks": 0, "equal": 0, "mismatches": [], "missing": [], "by_family": {}, "by_stratum": {},
                           "rule": ("per pick (site, step, row subset): all_reduce -> committed received (each rank) == bf16(f32(part_rank0) + f32(part_rank1)) elementwise "
                                    "and equal across ranks (world > 2: the AllReduce_v2 chain); a routed world-8 all-reduce (the pick's `spec`) -> "
                                    "the routed Definition's model over the committed partials (fused norm: <norm>/0 and <norm>/1 from the partials, "
                                    "the committed residual and the norm weight); all_gather -> committed received == concat(shard_rank0, shard_rank1) on the last dim; "
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
        st = res["by_stratum"].setdefault(p0["stratum"], {"picks": 0, "equal": 0, "families": []})
        for f_ in (p0["family"],) + (("EmbeddingShard_v1",) if p0["kind"] == "all_reduce" and is_embedding_site(p0["site"]) else ()):
            if f_ not in st["families"]:
                st["families"].append(f_)
        fam["picks"] += 1
        st["picks"] += 1
        res["picks"] += 1
        if p0["kind"] == "all_reduce" and is_embedding_site(p0["site"]):
            # EmbeddingShard_v1 leg: each rank's committed partial rows == masked gather of the committed ids from the rank's pinned
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
        route_whys = [f"rank {r}: {p['route_why']}" for p, r in zip(ps, ranks) if p.get("route_why")]
        if p0["kind"] == "all_reduce" and route_whys:
            fam["missing"] += 1
            res["missing"].append({"pick": i, "site": p0["site"], "step": p0["step"], "kind": p0["kind"], "family": p0["family"],
                                   "why": ("no routed Definition for the call: " + "; ".join(route_whys[:4]))[:400]})
            continue
        if p0["kind"] == "all_reduce" and p0.get("spec"):
            ok, d = _check_routed(ps, ranks)
            if ok is None:
                fam["missing"] += 1
                res["missing"].append({"pick": i, "site": p0["site"], "step": p0["step"], "kind": p0["kind"], "family": p0["family"], "why": d["why"]})
            elif ok:
                fam["equal"] += 1
                st["equal"] += 1
                res["equal"] += 1
            else:
                fam["mismatch"] += 1
                res["mismatches"].append({"pick": i, "site": p0["site"], "step": p0["step"], "kind": p0["kind"], "family": p0["family"], "rows": p0["rows"][:8], **d})
            continue
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
    """The pass's strata SITES against the sites of the replay-unreplayable collective families named by the ranks'
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
    strata = {k.split("|")[0]: set() for k in (res.get("by_stratum") or {})}
    for k, v in (res.get("by_stratum") or {}).items():
        strata[k.split("|")[0]].update(v.get("families") or ())
    strata_sites = set(strata)

    def covered(site: str, fam: str) -> bool:
        # a Program site is covered by a stratum of its family (strata recorded without families: any) at the site, a suffix of it, or a
        # module under it (the fused all-reduce norms of a decoder layer are named by the norm, the Program's rows by the layer)
        xs = [x for x, fs in strata.items() if not fs or fam in fs]
        return any(x == site or x.endswith("." + site) or site.endswith("." + x) or x.startswith(site + ".") for x in xs)

    fams = {}
    for f, ss in sorted(sites_by_fam.items()):
        miss = sorted(x for x in ss if not covered(x, f))
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

