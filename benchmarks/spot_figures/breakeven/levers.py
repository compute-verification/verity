"""Levers on Figure 1's request-grain lines, computed with the figure's own planner path (total_cost._point):
each lever rewrites the priced strata, the replay per sampled request or the scope's budget, and the bound and the
break-even beta follow from optimize.two_stage_design."""
from dataclasses import replace
from functools import lru_cache
from math import sqrt

from common import SCOPES, c, computed, confs, cuts, f1, inputs, lines, sc, tc

PEAK_MAC_S = sc["proof_overhead"]["value"]["peak_mac_s"]
#: the measured compressed size of the FP8 checkpoint, a share of its 8 bits a weight: what an FP8 theft must recover
FP8_COMPRESSED = 0.83
#: the bits a break-even reaches: FP4 at 4 bits a parameter, FP8 at its compressed size
WEIGHTS = tc.weights(c) | {"FP8": FP8_COMPRESSED * tc.weights(c)["FP8"]}
MACHINE_GPUS = c["machine_gpus"].value
WINDOW = c["window"].value


def refit(name, strata):
    """The inputs and line for `name` with its priced strata replaced: h and each stratum's price_macs recomputed as
    total_cost.priced does."""
    i = inputs[name]
    part = i["partition"]
    seconds = sum(x["per_token"] * x["seconds_per_block"] for x in strata)
    h = seconds * PEAK_MAC_S / sum(x["per_token"] * x["macs"] for x in strata)
    strata = [x | {"price_macs": x["seconds_per_block"] * PEAK_MAC_S / h} for x in strata]
    computed_ = sum(x["per_token"] * x["seconds_per_block"] for x in strata if x["priced"] == "computed") / seconds
    i2 = i | {"partition": replace(part, value=part.value | {"strata": strata, "computed_share": computed_}),
              "proof_overhead": replace(i["proof_overhead"], value=h)}
    line = (f1.TokenBlocks.of(strata, part.value["served_macs_per_token"]), f1.Fill.of(strata, i["proof_blocks"].value), h)
    comp = tuple(x["priced"] == "computed" for x in strata if x["per_token"] > 0)
    return i2, line, comp


@lru_cache(maxsize=None)
def _cut(name, partition):
    """`name`'s inputs with its circuit also cut by the `partition` card `partition`, priced as the line's own are."""
    if partition not in cuts:
        raise ValueError(f"GLM-5.3 has no partition card `{partition}`: its cards are {sorted(cuts)}")
    own = confs[name].partition
    if own and (both := set(cuts[own]["value"]["replaces"]) & set(cuts[partition]["value"]["replaces"])):
        raise ValueError(f"the partition cards `{own}` and `{partition}` both replace {sorted(both)}: a line takes one "
                         "cut of a kind")
    return tc.priced(tc.cut(tc.cut(c, cuts[own]) if own else c, cuts[partition]), sc["proof_overhead"])


def priced_strata(name, partition=""):
    """The line's priced strata, its circuit also cut by the `partition` card of that name where one is given."""
    i = _cut(name, partition) if partition else inputs[name]
    return [dict(x) for x in i["partition"].value["strata"]]


# ---- strata transforms -------------------------------------------------------------------------------------------
def fuse_reads(strata, fixed="read"):
    """Each per-expert table read fused into the GEMM that consumes it, one block a row read: the weight row stays
    inside the block, which commits only the GEMM's outputs.  Proving: the two measured marginals summed; the fixed
    cost a statement the read's (or both summed, `fixed="sum"`); a statement at 2^35 witness bits over the fused block's
    k_log (the larger plus one)."""
    reads = {3072: next(x for x in strata if x["block"].startswith("EmbeddingRead_v1{V=256,H=3072}")),
             4096: next(x for x in strata if x["block"].startswith("EmbeddingRead_v1{V=256,H=4096}"))}
    gemms = {3072: next(x for x in strata if x["block"].startswith("MoeExpertFp8BlockGemm_v4{K=6144,N=1")),
             4096: next(x for x in strata if x["block"].startswith("MoeExpertFp8BlockGemmW_v4{K=256,N=32"))}
    out = [x for x in strata if x not in reads.values() and x not in gemms.values()]
    for h in (3072, 4096):
        r, g = reads[h], gemms[h]
        assert abs(r["per_token"] - g["per_token"]) < 1e-6, (r["per_token"], g["per_token"])
        k_log = max(r["k_log"], g["k_log"]) + 1
        out.append(g | {"block": f"Fused[{r['block']} + {g['block']}]", "kind": "fused", "k_log": k_log,
                        "seconds_per_block": r["seconds_per_block"] + g["seconds_per_block"],
                        "fixed_seconds": r["fixed_seconds"] + (g["fixed_seconds"] if fixed == "sum" else 0.0),
                        "statement_blocks": max(1, tc.STATEMENT_BITS >> k_log),
                        "table_blocks": tc.table_blocks(k_log), "priced": "modeled"})
    return out


def set_bits(strata, prefix, bits):
    return [x | {"out_bits": bits} if x["block"].startswith(prefix) else x for x in strata]


def scale_proving(strata, marginal=1.0, fixed=1.0, cap=1.0, where=None):
    """Each stratum's marginal and fixed seconds scaled, its statement's cap in blocks multiplied by `cap`."""
    return [x | {"seconds_per_block": x["seconds_per_block"] * marginal, "fixed_seconds": x["fixed_seconds"] * fixed,
                 "statement_blocks": max(1, int(x["statement_blocks"] * cap))}
            if where is None or where(x) else x for x in strata]


# ---- bound and break-even ------------------------------------------------------------------------------------------
def make_line(name, strata=None, replay_gpu_s=None, extra_budget_share=0.0, registration=None):
    """A line: (name, inputs, line, conf, replay per sampled request in GPU-s or None for the figure's)."""
    conf = confs[name] if registration is None else replace(confs[name], registration=registration)
    if strata is None:
        return (name, inputs[name], lines[name], computed[name], conf, replay_gpu_s, extra_budget_share)
    i2, line, comp = refit(name, strata)
    return (name, i2, line, comp, conf, replay_gpu_s, extra_budget_share)


def bound(L, beta, scope="world week", scale=1.0, full=False, lean=False):
    """The line's exfiltration bound at beta (bits), as total_cost._point computes it, with its row if `full` and
    Lean's check of its design if `lean`.  `extra_budget_share` is a share of the scope's compute taken out of beta
    before the audit (a serve-time cost counted in beta)."""
    name, i, (tb, fl, oh), comp, conf, replay_gpu_s, extra, *more = L
    opts = more[0] if more else {}
    s = opts.get("scopes", SCOPES)[scope]
    name = opts.get("point", name)
    m = f1.model(i, tb, s, oh * scale, fl)
    replay = (m.replay.cost * tc.batch(conf, s, c)) if replay_gpu_s is None else replay_gpu_s
    g = tc.Grain(m.replay.replay_blocks, m.cells, replay, m.served)
    span = g.replay_blocks * tc.registration_gpu_seconds(conf.registration, c) / s.gpu_seconds
    budget = (beta - extra) * s.gpu_seconds
    if budget <= 0:
        return f1.served_total(m)
    if form := opts.get("budget_form"):
        import zkstack
        return zkstack.bound(m, g, budget, span, full, per_machine=form == "zkstack-machine")
    pt = tc._point((name, m, g, budget, comp, conf.form, span, lean))["exfiltration"]
    return (pt["y"], pt) if full else pt["y"]


def solve_beta(L, target, scope="world week", scale=1.0, lo=1e-6, hi=1e6, rel=0.01):
    """The least beta at which the bound reaches `target` bits (bisection in log beta), None if not by `hi`."""
    if bound(L, hi, scope, scale) > target:
        return None
    if bound(L, lo, scope, scale) <= target:
        return lo
    while hi / lo > 1 + rel:
        mid = sqrt(lo * hi)
        lo, hi = (mid, hi) if bound(L, mid, scope, scale) > target else (lo, mid)
    return sqrt(lo * hi)


def floor_bits(L, beta=0.01, scope="world week"):
    """The bound as proving's cost goes to nothing (h times 1e-12, as total_cost.breakeven's floor)."""
    return bound(L, beta, scope, scale=1e-12)


def tb(bits):
    return bits / 8e12
