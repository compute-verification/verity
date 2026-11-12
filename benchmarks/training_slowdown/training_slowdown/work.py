"""The work a step's record states for the theorem of record: M, the step's matrix products whose two operands both
depend on the update, with each operand's β and each output's width.

A step's record is a job's trace (`run.trace_step`'s: the capture's raw product rows at each k of the console's dial,
which `priced` prices), or the products stated from one over a run's steps (`products_of_trace`, which
`formalism.measure` keeps as `Measured.products`). `he_of_rows` states M from a step's priced rows at a k: the
products it leaves out, each operand's β and each output's width. `classified_at` is a record's products at a
policy's k. The reasons a record states no M are here too.

Nothing here computes a slowdown or reads the slowdown calculator: `calculator.slowdown` reads this work and asks it.
"""
from __future__ import annotations

import math

from .gates import gates

#: what every string a panel publishes calls this code, which names no module, function or constant of it
#: (tests/test_formalism.py's code-name check)
THE_CODE = "training's slowdown code"
#: the premise the products term takes on the step's elementwise ops, as published strings state it
P_MATCH = "elementwise steps don't fan out (P-match)"
#: the first training_slowdown whose capture writes a traced step's product rows (training_slowdown.capture.products' `CAPTURE`)
PRODUCTS_FROM = "4.8"
#: why the products kind is unmeasured on a record with no product rows
PRODUCTS_UNMEASURED = (f"unmeasured: the record has no product rows (a step captured on its own has them, and a "
                       f"recorded program's trace has them from {THE_CODE} {PRODUCTS_FROM} on)")
#: the first training_slowdown whose capture writes everything the term reads: which operands read a gather's rows
#: (`gathered`) and the trace's `p_match`
SYSTEM_FROM = "4.12.1"
#: the positions at the start of each sequence that M leaves out of every token-indexed product (`he_of_rows`):
#: position 0 reads only its own token, so sequences that share their first token (a BOS, or the stream's own where
#: windows are cut from it) compute it identically through every layer. A longer shared prefix needs the batch's
#: tokens, which no trace keeps
SHARED_PREFIX = 1
#: why the products kind is unmeasured where a trace isn't of the run's own program (`formalism.other_program`): a
#: products term is priced only on its run's own program
OTHER_PROGRAM = ("unmeasured: the trace isn't of the run's own program: {unmatched} of the {shapes} matrix-product "
                 "shapes its traced step runs are none of the {ran} the run's op log ran (the largest by "
                 "multiplications: {largest}), so its products aren't the run's and are never priced; retrace the "
                 "run so that its traced step runs the run's own sizes")
#: a product row's kinds (`product_kind`)
PRODUCT_KINDS = ("fresh", "theta", "expansion", "attention", "self", "unaffected")
#: each GEMM Definition's operand and output widths in bits, by its gates a gadget (a product row's `cell_gates`):
#: bf16's and f16's 16 and 16, f32's 32 and 32, fp8's and fp4's outputs at their operands' width, the least a circuit
#: writes them at
HE_BITS = {float(gates(d)): bits for d, bits in (("HopperBF16WgmmaDot16_v2", (16, 16)), ("F32Fma_v4", (32, 32)),
                                                  ("BlackwellE4m3QmmaDot32_v2", (8, 8)),
                                                  ("BlackwellE5m2QmmaDot32_v2", (8, 8)),
                                                  ("BlackwellNvf4OmmaDot64_v2", (4, 4)))}
#: why the products term is unmeasured on rows that don't say which operands read a gather's rows
HE_RETRACE = (f"unmeasured: the trace's product rows predate {THE_CODE} {SYSTEM_FROM}, so they don't say which "
              "products read an embedding's rows before any op makes them token-specific, which M leaves out; retrace "
              "the run")
#: why the products term is unmeasured on products that come without M (a hand-written set)
HE_NO_ROWS = ("unmeasured: the products carry no operand rates or output widths, which M reads (a step's product rows "
              "give them, and so does a model)")
#: a product row's dims each operand spans: A is batch x m x k, B batch x k x n
SIDES = (("batch", "m", "k"), ("batch", "k", "n"))
#: the k from which the head's input and weight gradients leave M: their operand δ_logits, a token's vocabulary-wide
#: row, comes from its 512 words and two scalars, and with them in M no β(k) holds there
HEAD_K = 8e7
#: why the products term is unmeasured from `HEAD_K` on products that don't say which are the head's
HE_HEAD = (f"unmeasured: from k = {HEAD_K:.0e} M leaves out the head's input and weight gradients, and the "
           "model's products sum them with the layers'").replace("e+0", "e")


def is_paid(rate: float, width: float) -> bool:
    """Whether an operand a block pays `rate` bits an entry of is paid at its passed width `width`: a cast keeps its
    argument's rate, and a block can cut before it, so a widened operand is paid at the narrower width."""
    return rate >= 0.999 * width


def product_kind(row: dict) -> str:
    """A product row's kind (training_slowdown.capture.products' rows): unaffected, attention (two token-indexed dims
    among m, k and n), self (both operands one array), theta (one operand doesn't depend on θ′), fresh (both do and
    both are paid at their passed width, `is_paid` on the row's rates and widths) or expansion. The row's own `paid`
    isn't read: before training_slowdown 4.10 a capture compared an upcast operand's rate with the cast's width, so a
    trace's rates and widths decide."""
    if not any(row["affected"]):
        return "unaffected"
    if sum(x in row["token_dims"] for x in ("m", "k", "n")) >= 2:
        return "attention"
    if row["self"]:
        return "self"
    if not all(row["affected"]):
        return "theta"
    return "fresh" if all(is_paid(r, w) for r, w in zip(row["rates"], row["widths"])) else "expansion"


def products_of_trace(block: dict, steps: float | None = None) -> dict:
    """`Measured.products` from a trace's product rows (training_slowdown.capture.products' `rows`, the trace's
    `products`) over `steps` traced steps: each kind's MACs (`product_kind`, each row's times its count and `steps`),
    the trace's P-match check, and the products term's M over the steps of the block's tokens (`he`, `he_of_rows`), at
    the k the capture classified them at, and at each k of the console's dial where the trace keeps rows there (`at`).
    The steps are ones after the run's first update (`formalism.of_record` counts no step before it, whose products
    read only registered weights; None: a step captured on its own, which stands for one after it)."""
    kinds = dict.fromkeys(PRODUCT_KINDS, 0.0)
    for r in block["rows"]:
        kinds[product_kind(r)] += r["count"] * (1.0 if steps is None else steps) * r["macs"]
    got = {"k": float(block["k"]), "capture": block["capture"], "capture_code": block.get("capture_code"),
           "kinds_macs": kinds, "p_match": block.get("p_match"),
           "he": he_of_rows(block["rows"], block["k"], steps, block.get("tokens"))}
    if "at" in block:
        rest = {x: v for x, v in block.items() if x != "at"}
        got["at"] = {x: products_of_trace(rest | b, steps) for x, b in block["at"].items()}
    return got


def classified_at(p: dict, k: float) -> dict | None:
    """`Measured.products` `p` as classified at k: itself where it was classified at k, else its block at k (`at`: a
    trace's rows at each k of the console's dial, training_slowdown.capture.products.raw_on_dial's) with `p`'s
    stand_in and other_program; None where it has neither."""
    if math.isclose(p["k"], k, rel_tol=1e-12):
        return p
    return next((b | {x: p[x] for x in ("stand_in", "other_program") if x in p} for b in p.get("at", {}).values()
                 if math.isclose(b["k"], k, rel_tol=1e-12)), None)


def k_star(k: float, w: float) -> float:
    """K* = ½√(k/w): the least contraction whose output a block of k bits can't make below its width w from the
    contraction's two operands (a rows by b columns of them make ab entries for (a + b)·K·w bits)."""
    return 0.5 * math.sqrt(k / w)


def beta_eff(w: float, r: float, k: float) -> float:
    """An operand's width in (H) where the update reaches it through a bottleneck of rank r a row or column (an
    adapter's Ax, a broadcast vector, a frozen edge): β/(1 + (K*/r)²), K* at its width (`k_star`)."""
    return w / (1 + (k_star(k, w) / r) ** 2) if r > 0 else 0.0


def left_out(row: dict) -> str | None:
    """Which of `he_of_rows`' left-out MACs a product row whose two operands both depend on the update is counted in,
    None where M holds it: `self_macs` for one array times itself, `gathered_macs` for a product with an operand that
    reads a gather's rows, `left_out_macs` for the optimizer's products and attention's (two token-indexed dims among
    m, k and n)."""
    if row["self"]:
        return "self_macs"
    if any(row["gathered"]):
        return "gathered_macs"
    if row["phase"] == "optimizer" or sum(x in row["token_dims"] for x in ("m", "k", "n")) >= 2:
        return "left_out_macs"
    return None


def in_m(row: dict) -> bool:
    """Whether the products term's M holds a product row, a trace's raw one or a priced one: its two operands both
    depend on the update, and nothing leaves it out (`left_out`)."""
    return all(row["affected"]) and left_out(row) is None


def he_of_rows(rows: list[dict], k: float, steps: float | None = None, tokens: float | None = None) -> dict | None:
    """The products term's M from a step's product rows at k (training_slowdown.capture.products' rows) over `steps` steps
    after the run's first (None: one step captured on its own, which stands for one after the first ones), each of
    `tokens` tokens (+1: the grader chooses M): every product whose two operands both depend on the update, except
    one array times itself (W and X disjoint), a product with an operand that reads a gather's rows before any op
    makes them token-specific (repeated rows aren't distinct words), attention's products (two token-indexed dims
    among m, k and n) and the optimizer's (Newton-Schulz's B·X, whose operands come from one X), which the red-team
    of (H) leaves out. Each operand's β is its passed width, capped at its
    Definition's operand width (`HE_BITS`), where the row's rate pays it (`is_paid`); else its rate ρ_a
    below the width implies the bottleneck rank r = √(ρ_a k)/(2w) (a K-long product made whole at 4K²w²/k an entry),
    so β = `beta_eff`(w, r, k) = w·ρ_a/(w + ρ_a). On a step with an adapter (`r_step`, the least contraction K of a
    forward product of two affected operands, one token-free, that widens it 4K-fold or more on a step where a
    frozen weight multiplies an operand the update reaches (a LoRA's frozen layers, not a distilling teacher's, which
    read only tokens), or that builds a weight with both sides 4K or more: a LoRA's B·(Ax), a DoRA's B·A),
    every token-indexed operand with more than r_step entries a token, or with no dim but tokens (attention's
    scores), is at most `beta_eff`(w, r_step, k): the update reaches it only through the adapters, and its rate
    stops showing that once r_step passes K*. The adapters' own operands (A, B, Ax and the rank-wide gradients) keep
    their widths. Each output is at its Definition's output width b_y. Token-indexed products keep 1 −
    `SHARED_PREFIX`/n of their MACs, and every output. An operand no token indexes is a
    weight (`weight`), which `calculator.slowdown` holds to the record's code bits a trained weight at most, the
    others activations, which it holds to β(k) at most; a backward product contracting over tokens is a weight
    gradient (`dw`); the head's are the
    backward products over its vocabulary width V, the widest output of a forward product indexed by tokens on m that
    the update reaches, a frozen head's too (`head`). The trained weights (`trained_weights`, None without `tokens`) are each forward product's weight
    operands' entries, a product of one token-indexed operand counted as count·(its tokens)/`tokens` weights (to the
    nearest whole one, the weights a step's forward reads once a token), a product of two weights as count of each.
    Grouped by (β_a, β_b, the widths, which is a weight, b_y, forward or not, its contraction K, `dw`, `head`): {"k",
    "groups": [{"beta", "width", "weight", "b_y", "forward", "k_dim", "dw", "head", "macs", "wires"}], "r_step",
    "candidates_macs", "self_macs", "gathered_macs", "left_out_macs" (attention's and the optimizer's), "steps",
    "step_tokens", "trained_weights"}; None for rows captured before `SYSTEM_FROM`, which don't flag a gather's
    rows."""
    if any("gathered" not in r for r in rows):
        return None
    seq = max((r["dims"][d] for r in rows if sum(x in r["token_dims"] for x in ("m", "k", "n")) >= 2
               for d in ("m", "k", "n") if d in r["token_dims"]), default=None)
    keep = 1 - SHARED_PREFIX / seq if seq else 1.0
    sides = SIDES

    def token(r: dict, i: int) -> bool:
        return any(x in r["token_dims"] for x in sides[i])

    def entries(r: dict, i: int) -> float:
        return math.prod(r["dims"][x] for x in sides[i])

    frozen = any(r["phase"] == "forward" and not r["affected"][i] and not token(r, i) and r["affected"][1 - i]
                 for r in rows for i in (0, 1))
    ranks = []
    for r in rows:
        d, td = r["dims"], r["token_dims"]
        if r["phase"] != "forward" or not all(r["affected"]):
            continue
        widened = frozen and ((td == ["m"] and 4 * d["k"] <= d["n"]) or (td == ["n"] and 4 * d["k"] <= d["m"]))
        if widened or (not td and 4 * d["k"] <= min(d["m"], d["n"])):
            ranks.append(d["k"])
    r_step = min(ranks, default=None)
    vocab = max((r["dims"]["n"] for r in rows if r["phase"] == "forward" and r["token_dims"] == ["m"]
                 and any(r["affected"])), default=None)
    groups: dict = {}
    total = trained = 0.0
    out = {"self_macs": 0.0, "gathered_macs": 0.0, "left_out_macs": 0.0}
    for r in rows:
        if not all(r["affected"]):
            continue
        n = r["count"] * (1.0 if steps is None else steps)
        macs = n * r["macs"]
        total += macs
        d, td = r["dims"], r["token_dims"]
        why = left_out(r)
        if why:
            out[why] += macs
            continue
        w_op, b_y = HE_BITS.get(r["cell_gates"], (min(r["widths"]),) * 2)
        betas, widths, weight = [], [], []
        for i, (w, rate) in enumerate(zip(r["widths"], r["rates"])):
            w = min(w, w_op)
            b = w if is_paid(rate, w) else w * rate / (w + rate) if rate > 0 else 0.0
            if r_step is not None and token(r, i):
                wide = math.prod(d[x] for x in sides[i] if x not in td)
                if wide > r_step or all(x in td for x in sides[i] if x != "batch"):
                    b = min(b, beta_eff(w, r_step, k))
            betas.append(round(b, 9))
            widths.append(w)
            weight.append(not token(r, i))
        if r["phase"] == "forward" and tokens:
            if all(weight):
                trained += r["count"] * (entries(r, 0) + entries(r, 1))
            elif any(weight):
                uses = r["count"] * math.prod(d[x] for x in td) / tokens
                trained += (round(uses) or uses) * entries(r, weight.index(True))
        dw = r["phase"] == "backward" and td == ["k"]
        head = vocab is not None and r["phase"] == "backward" and (
            (td == ["m"] and d["k"] == vocab) or (dw and vocab in (d["m"], d["n"])))
        key = (*betas, *widths, *weight, b_y, r["phase"] == "forward", d["k"], dw, head)
        g = groups.setdefault(key, {"beta": betas, "width": widths, "weight": weight, "b_y": b_y,
                                    "forward": r["phase"] == "forward", "k_dim": d["k"], "dw": dw, "head": head,
                                    "macs": 0.0, "wires": 0.0})
        g["macs"] += (keep if td else 1.0) * macs
        g["wires"] += n * r["outputs"] * b_y
    return {"k": float(k), "groups": [groups[x] for x in sorted(groups)], "r_step": r_step, "candidates_macs": total,
            **out, "steps": steps, "step_tokens": tokens, "trained_weights": trained if tokens and trained else None}


def p_match_reason(broken: list[dict]) -> str:
    """Why a products figure is unmeasured on a step whose elementwise ops fan out into a product's operand (its
    `p_match`): the rows' rates take P-match, so an operand's entries may be fewer distinct words than its rate says."""
    ops = ", ".join(f"{x['op']} (fan-out {x['fan_out']:.3g}, {x['count']}×)" for x in broken)
    return (f"unmeasured: the traced step breaks the premise that {P_MATCH}: {ops} hand one entry to several on the "
            f"way into a product's operand")


def repriced(trace: dict) -> list[dict]:
    """The raw rows of a job's trace whose product is priced in another dtype than the trace's circuit counts it in
    (`capture.products.priced_dtype` and `counted_dtype`), the largest first by its MACs. A trace of this capture
    has none; one of a capture before `capture.products.PRICED_AS_TRACED_FROM` has its float32 attention products,
    which that capture's rule prices in bf16 (`capture.products.AUTOCAST`) and its circuit counted in float32."""
    from .capture import products

    found = [r for r in trace["rows"] if products.priced_dtype(r, trace.get("capture")) != products.counted_dtype(r)]
    return sorted(found, key=lambda r: -r["count"] * math.prod(r["dims"].values()))


def repriced_reason(trace: dict, found: list[dict]) -> str:
    """Why there is no slowdown over a trace's own circuit where `repriced` finds rows: the work would be priced in
    one dtype and the circuit it is divided by in another."""
    from .capture import products

    r, d = found[0], found[0]["dims"]
    return (f"unmeasured: {sum(x['count'] for x in found)} of the traced step's matrix products are priced in "
            f"another dtype than its circuit counts them in (the largest: {r['count']} {r['phase']} {r['op']} of "
            f"{d['m']}×{d['k']} by {d['k']}×{d['n']}, counted in {products.counted_dtype(r)} and priced in "
            f"{products.priced_dtype(r, trace.get('capture'))}), and one slowdown has one price; retrace the "
            f"record with {THE_CODE} {products.CAPTURE} or later, which prices each product as its circuit counts it")


def priced(trace: dict) -> dict:
    """A job's trace (`run.trace_step`'s) as the block `products_of_trace` reads: each raw product row priced
    (`capture.products.priced`, by its capture's rule), at the k the job classified the rows at and at each other k
    of the console's dial (`at`), with the capture's version (None on a trace that states none), the traced step's
    tokens and, where the trace has it, the version of the code that took it (`capture_code`)."""
    from .capture import products

    def block(b: dict) -> dict:
        return {"k": b["k"], "rows": [products.priced(r, trace.get("capture")) for r in b["rows"]],
                "p_match": b["p_match"]}

    return block(trace) | {"capture": trace.get("capture"), "tokens": trace["tokens"],
                           "at": {k: block(b) for k, b in trace["at"].items()}} | (
        {"capture_code": trace["capture_code"]} if "capture_code" in trace else {})
