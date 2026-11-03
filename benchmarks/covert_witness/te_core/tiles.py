"""Reading A's matrix-product tiles (te_core.formalism.packer): witness blocks that do part of one matrix product,
reading only the operand coordinates they use.

The ops are the record's own count: a ledger row's `macs_log` (`of_log`), or a traced step's GEMMs (`of_graph`). An op
is one or more products of dims (b, m, K, n) (`products`): b independent products of an m×K and a K×n operand, each
output a sum over K. A tile is a box of them, b' products, m' rows, k' inner coordinates and n' columns, holding
b'·m'·k'·n' MACs at the op's gates a MAC (te_core.gates). It reads its slices of the two operands at their widths
(`widths`: the narrowest width the record shows each passing, a log's op dtype, `WIDTH`, and a trace's through its
views and casts), b'·m'·k' and b'·k'·n' values, except an operand reading A takes as free; its outputs are free. A
weight-gradient product δᵀX is one of them: its inner coordinates are the tokens, so its tiles are δ and X slices over
a token set.

Which operand is free: a value computed from θ and token data alone is free under reading A (the block computes it
from the token's id), and neither the log nor the trace's GEMM records say which of a product's operands are, so a
tile takes either one free, the more favourable to the block. At least one is not: the GEMM gadget accumulates from a
zero word, so a product whose operands are both free has no affected gate. The tile reads the other operand's slice.
Taking either free also covers a block that reads the update code (Δ bits, once) and then has a θ' operand free: it
holds no more than the same tile with that operand free and all k bits for the other.

A tile whose sums continue ones begun outside it also reads their b'·m'·n' partial sums at the accumulator's width;
the same box beginning its sums reads less, so a beginning box is the most a tile holds. Which coordinates can begin a
sum is the summation order's, and the trace keeps a matmul as one op with no order, so every inner coordinate is taken
as one a sum may begin at (the order most favourable to a block; under a chain, coordinate j begins no tile shorter
than j + 1, which a known order would charge).

The most MACs a tile reading at most R bits holds (`volume`) spans every row and column: its reads grow with m' and n'
more slowly than its MACs. So, α and β the operands' bits a value (0 for the free one):
- where a one-coordinate slice of every row and column fits (α·m + β·n ≤ R), V = m·n·min(b·K, R/(α·m + β·n));
- otherwise V is the most m'·n' with α·m' + β·n' ≤ R (k' = b' = 1), within the dims.
A tile's cap is min(block gates, V × gates a MAC) (`cap`). An op `products` doesn't place (an elementwise
multiplication, a gather, an op outside these names, the rest of a phase's gates) has no tile: its gates ride free, at
the block's gates, and count in `unplaced`.
"""
from __future__ import annotations

import json
import math
import operator

from .gates import kind_of, per_mac, split_key

#: a GEMM operand's bits a value, by the op's kind (te_core.gates)
WIDTH = {"gemm/bf16": 16, "gemm/f16": 16, "gemm/e4m3": 8, "gemm/e5m2": 8, "gemm/fp4": 4, "gemm/f32": 32}
#: the phases whose ops are in token blocks (te_core.formalism's `_token_log`)
TOKEN_PHASES = ("fwd", "bwd")
SDPA = ("_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_cudnn_attention")
GRAPH_PHASE = {"forward": "fwd", "backward": "bwd"}


def products(op: str, shapes: list) -> list[tuple[int, int, int, int]] | None:
    """An op's matrix products as (b, m, K, n), from its tensor operands' shapes in te_core.flops's order; None for an
    op this doesn't place. Attention (scaled dot product, [B, H, L, E] operands) is its products as torch's FLOP formula
    counts them, the query heads sharing a key head stacked as rows of one product."""
    s = [tuple(x) for x in shapes]
    try:
        if op in ("mm", "_int_mm", "_scaled_mm"):
            (m, K), (_, n) = s[0], s[1]
        elif op in ("addmm", "addmm_", "_addmm_activation"):
            (m, K), (_, n) = s[1], s[2]
        elif op == "bmm":
            (b, m, K), (_, _, n) = s[0], s[1]
            return [(b, m, K, n)]
        elif op in ("baddbmm", "baddbmm_"):
            (b, m, K), (_, _, n) = s[1], s[2]
            return [(b, m, K, n)]
        elif op in ("addbmm", "addbmm_"):
            (b, m, K), (_, _, n) = s[1], s[2]
            K *= b
        elif op == "mv":
            (m, K), n = s[0], 1
        elif op in ("addmv", "addmv_"):
            (m, K), n = s[1], 1
        elif op in ("dot", "vdot"):
            (K,), m, n = s[0], 1, 1
        elif op in ("addr", "addr_"):
            (m,), (n,), K = s[1], s[2], 1
        elif op in SDPA or op.removesuffix("_backward") in SDPA:
            back = op.endswith("_backward")
            (B, hq, L, E), (_, hkv, S, _), (*_, Ev) = s[1:4] if back else s[0:3]
            b, rows = B * hkv, hq // hkv * L
            scores = (b, rows, E, S)
            if not back:
                return [scores, (b, rows, S, Ev)]
            return [scores, (b, rows, Ev, S), (b, S, rows, Ev), (b, rows, S, E), (b, S, rows, E)]
        else:
            return None
    except (ValueError, TypeError, IndexError, ZeroDivisionError):
        return None
    return [(1, m, K, n)]


def _op(phase: str, cls: str, op: str, kind: str, macs: float, count: float, prods, widths=None) -> dict:
    placed = prods is not None and kind in WIDTH and sum(math.prod(p) for p in prods) * count == macs
    w = WIDTH.get(kind)
    return {"phase": phase, "class": cls, "op": op, "gates": macs * per_mac(kind), "macs": macs,
            "products": [list(p) for p in prods] if placed else [],
            "widths": list(widths) if widths is not None else [w, w] if w else None}


def of_log(log: dict) -> list[dict]:
    """A `macs_log`'s ops as the tiles read them: each op's phase, class, gates and MACs, its products where `products`
    places it and their MACs times its count are its MACs (else none: it rides free), and its operands' widths (its
    dtype's); the MACs of each `by` class that no op row holds are one more op, with no products."""
    out, listed = [], {}
    for key, (count, macs) in log.get("ops", {}).items():
        phase, cls, op, shapes, dtype = split_key(key)
        kind, _ = kind_of(phase, cls, op, dtype)
        try:
            prods = products(op, json.loads(shapes)) if kind in WIDTH else None
        except json.JSONDecodeError:
            prods = None
        out.append(_op(phase, cls, op, kind, macs, count, prods))
        listed[f"{phase}/{cls}"] = listed.get(f"{phase}/{cls}", 0) + macs
    for by, macs in log.get("by", {}).items():
        rest = macs - listed.get(by, 0)
        if rest > 0:
            phase, _, cls = by.partition("/")
            out.append(_op(phase, cls, "unlisted", "elementwise/bf16", rest, 1, None))
    return out


def _passed_width(a) -> int:
    """The narrowest dtype a GEMM operand has on its way in through views, casts and scalings by a constant
    (te_core.capture.structure's `_WALK` and `_SCALE` with one tensor argument), which a block can apply itself."""
    from .capture.structure import _SCALE, _WALK, _arg_nodes, _dtype_bits, _pk

    b = 64
    for _ in range(10_000):
        v = a.meta.get("val") if hasattr(a, "meta") else None
        if hasattr(v, "dtype"):
            b = min(b, _dtype_bits(v.dtype))
        if not (hasattr(a, "op") and a.op == "call_function"):
            break
        if a.target is operator.getitem or _pk(a.target) in _WALK and a.args and hasattr(a.args[0], "op"):
            a = a.args[0]
            continue
        ins = [x for x in _arg_nodes(a) if hasattr(x.meta.get("val"), "dtype")]
        if _pk(a.target) in _SCALE and len(ins) == 1:
            a = ins[0]
            continue
        break
    return b


def of_graph(graph, circuit) -> list[dict]:
    """A traced step's ops as `of_log` gives a log's: each GEMM of `graph` (te_core.capture.structure.StepGraph) with
    its dims, its weight operand's class and its operands' passed widths (`_passed_width`, at most the GEMM kind's),
    and the rest of each phase's gates in `circuit` (its StepCircuit) as one op with no products."""
    from .capture.structure import _gemms

    nodes = {n.name: n for n in graph.graph.graph.nodes}
    out, gemm = [], {}
    for g in graph.matmuls:
        if g.part == "elementwise":
            continue
        phase = GRAPH_PHASE.get(g.phase, "other")
        op = g.op.removeprefix("aten.").split(".")[0]
        kind, _ = kind_of(g.phase, "", op, g.dtypes[0] if g.dtypes else None)
        d = g.dims
        part = next(x for x in _gemms(nodes[g.node]) if x.part == g.part)
        widths = [min(_passed_width(a), WIDTH.get(kind, 64)) for a in part.operands[:2]]
        cls = "grad" if phase == "bwd" else g.weight
        out.append(_op(phase, cls, op, kind, g.macs, 1, [(d["batch"], d["m"], d["k"], d["n"])], widths))
        gemm[phase] = gemm.get(phase, 0.0) + out[-1]["gates"]
    for code, phase in ((0, "fwd"), (1, "bwd"), (2, "other")):
        rest = float(circuit.gates[circuit.phase == code].sum()) - gemm.get(phase, 0.0)
        if rest > 0:
            out.append({"phase": phase, "class": "rest", "op": "rest", "gates": rest, "macs": None, "products": [],
                        "widths": None})
    return out


def volume(R: float, alpha: float, beta: float, b: int, m: int, K: int, n: int) -> float:
    """The most MACs of a (b, m, K, n) product a tile reading at most R bits holds (module docstring), α and β its
    operands' bits a value."""
    if R <= 0:
        return 0.0
    row = alpha * m + beta * n
    if row <= R:
        return m * n * min(b * K, R / row if row else math.inf)
    if not alpha:
        return R / beta * m
    if not beta:
        return R / alpha * n
    mm, nn = min(m, R / (2 * alpha)), min(n, R / (2 * beta))
    if mm == m:
        nn = min(n, (R - alpha * m) / beta)
    elif nn == n:
        mm = min(m, (R - beta * n) / alpha)
    return mm * nn


def cap(op: dict, prod, k: float, size: float) -> float:
    """The most of an op's gates a tile of `prod` holds within k bits and `size` gates: the better of its two operands
    free (module docstring)."""
    g = op["gates"] / op["macs"]
    wa, wb = op["widths"]
    return min(size, g * max(volume(k, 0, wb, *prod), volume(k, wa, 0, *prod)))


def caps(ops: list[dict], phase: str, k: float, size: float) -> list[tuple[float, float]]:
    """(gates, cap) over the ops of `phase`: each product's share of its op's gates (by its MACs) at its tile's cap,
    and an op with no products at `size` (it rides free)."""
    out = []
    for op in ops:
        if op["phase"] != phase or not op["gates"]:
            continue
        if not op["products"]:
            out.append((op["gates"], size))
            continue
        total = sum(math.prod(p) for p in op["products"])
        out += [(op["gates"] * math.prod(p) / total, cap(op, p, k, size)) for p in op["products"]]
    return out


def unplaced(ops: list[dict]) -> float:
    """The share of the token ops' gates with no tile (they ride free)."""
    tok = [op for op in ops if op["phase"] in TOKEN_PHASES]
    total = sum(op["gates"] for op in tok)
    return sum(op["gates"] for op in tok if not op["products"]) / total if total else 0.0
