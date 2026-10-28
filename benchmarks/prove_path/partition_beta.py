"""Committed bytes a token (β) and each proof block's boundary, for partitions of the served Llama-3.2-1B int8 decode step.

The served step (`verity_vllm.program.predict.build`, ``step``: one decode row, ``int8_exact`` linears on the H200, FA3
attention) is one root Call an op: a layer is Int8RowQuant, ScaledMmS8 (qkv), RoPE (q, k), Attention, Int8RowQuant,
ScaledMmS8 (o), RMSNormFused, Int8RowQuant, ScaledMmS8 (gate_up), SiluMul, Int8RowQuant, ScaledMmS8 (down), RMSNormFused.
Each partition groups those ops into proof blocks and commits exactly the values a block other than their writer's reads (and
k and v, which later tokens read). β counts each committed value once, at its writer.

* ``a0``: every root Call's output, today's Commit stream (COMMIT-BYTES' 1.66 MB a token). Its blocks are whole Calls, so it
  is no proof partition: gate_up's Call is 4.4 G ANDs.
* ``a``: `cut.Q_word` v1 (X = 16) over each Call, the IR's own answer (`evaluate_definition`): a unit an output word, and
  the derived committed set (`derived_committed`), which adds the norms' rsqrt and attention's per-key scores and states.
* ``w``: the window circuit's grain (`window_builder.Grain`): GEMM slices of 32 768 weight words (``ScaledMmS8_v1{2048,32}``,
  ``{8192,8}``), each other op a unit (RoPE a head, attention a head), committing what ``a0`` commits.
* ``b``: the norm and the quantize fused into the GEMM tile that reads them (a prologue each tile recomputes), SiluMul into
  down's.
* ``c``: only dot-product inputs (the bf16 activations a GEMM or attention reads) and outputs, quantize fused into the GEMM
  tile's prologue; the residual stream kept.
* ``d``: ``c`` with int8 commits for the GEMM inputs (the codes and their scale, not the bf16 row): the norm's and SiluMul's
  blocks quantize. Attention's output stays bf16, since its quantize spans the 32 head blocks.
* ``e1``: epilogue fusion: RoPE into the qkv tile (a tile of whole rotary pairs), SiluMul into gate_up's (a tile of g, u
  pairs), the norms quantize, o quantized in o's prologue, the residual once a layer (norm1 adds a back before its own
  residual add). A head's q columns, RoPE and attention in one block would drop q' too, but its 64 coordinates alone are
  17.4 M ANDs, past the cap.
* ``e2``: ``e1`` with max-first recompute (cut.py's ``Q_word`` v2 recompute): a gate_up tile commits its partial max only,
  and a code block recomputes the tile and writes its int8 codes, so s is never committed; attention likewise writes oq.
* ``e3``: ``e1`` at the finest GEMM units: a unit a coordinate, or a rotary or (g, u) pair where its epilogue pairs columns,
  each reading its own weight column and its slice's shared rows, and o quantized in a block of its own (a coordinate
  can't afford the prologue). A unit is drawn by its work and costs its block, so smaller units prove less a window.

Every number is the IR's: widths from the Calls' signatures, MACs from their tensor-core steps, ANDs and two-input gates from
each primitive's Boolean version (census ``boolean=boolean_version``), summed over a unit's gates (`Q_word` units, tile
slices) or the whole Call's (a block holding whole ops). A block's opening ANDs are `optimize.row_ands` of each row it reads or
writes (a value slice a row, a weight column a row, a key's k and v a row), an estimate of the in-circuit hashing. Its
``k_log_est`` is `class_size.block_k_log` of its ANDs and those rows, and ``over_cap`` marks one past C-Flock's 2^27-bit
block. Attention's key blocks, where the window's FA3 BN doesn't set them, are as few as keep a block's ANDs under
``CAP_ANDS``.

    python partition_beta.py [--keys 18,35,144,288] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/vllm"))
from verity_vllm.program.model import INT8_EXACT_QUANT  # noqa: E402
from verity_vllm.program.predict.build import build_program  # noqa: E402
from verity_vllm.program.predict.phi import Phi  # noqa: E402

from verity.core.protocols.sampled_proofs.one_stage.partition import unit_class  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import row_ands  # noqa: E402
from verity.ml.circuits.census import census  # noqa: E402
from verity.ml.circuits.codec import descriptor_id  # noqa: E402
from verity.ml.circuits.cut import CallGraph, boundary_widths, call_scope, derived_committed, evaluate_definition  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.ml.circuits.program import Program, is_input  # noqa: E402
from verity.ml.definitions.boolean import boolean_version  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import class_size as CS  # noqa: E402

#: the window's embedding unit: a registered table row read
EMBED_READ = "EmbeddingRead_v1{V=128256,H=2048}"
CFG = ROOT / "integrations/vllm/tests/program/predict/fixtures/llama32-1b.config.json"
#: MACs of one tensor-core step, by primitive
STEP_MACS = {"BlackwellS8ImmaDot32_v1": 32, "HopperBF16WgmmaDot16_v1": 16}
#: Gemm_v2{2048,16}'s ANDs, at k_log 27 (class_size.MEASURED): about where a block passes C-Flock's 2^27 bits
CAP_ANDS = 16 * 1_053_595
#: Λ/ε at δ = 2^-40, ε = 1%: the expected draws a window
DRAWS = math.log(2**40) / 0.01
LAYERS = 16
HEADS, KV_HEADS, HEAD_DIM = 32, 8, 64
#: FA3's key block at a decode launch on the H200 (`window_cost.L1`), and a key block's running state (m, l and O in f32):
#: an estimate of its committed bytes
FA3_BN, STATE = 192, 4 + 4 + 4 * 64
#: the window grain's GEMM columns a slice (`Grain.gemm_words` 32 768 weight words, s8 twice the columns)
TILE = {2048: 32, 8192: 8}


def phi() -> Phi:
    cfg = {**json.loads(CFG.read_text()), "quantization_config": dict(INT8_EXACT_QUANT)}
    return Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg},
               target={"compute_capability": [9, 0], "num_sms": 132, "int8_exact": "linears"})


def _one(fn):
    return CompositeDefinition("OneCall", 1, (), lambda S: (tuple(fn.signature()[0]), fn.signature()[1]),
                               lambda B, S, *args: B.call(fn, *args), register=False)


_WORK: dict[str, tuple[int, int]] = {}


def work(fn) -> tuple[int, int]:
    """(two-input gates, ANDs) of `fn` through its Boolean version, every facet."""
    if fn.id not in _WORK:
        if not getattr(fn, "params", True):
            _WORK[fn.id] = (0, 0)
        else:
            e = census(Program(_one(fn)), boolean=boolean_version)["classes"][descriptor_id(fn)]
            w = e.get("work", {}).values()
            _WORK[fn.id] = (sum(x["gates"] for x in w), sum(x["and"] for x in w))
    return _WORK[fn.id]


def bits(t) -> int:
    return sum(t.leaf_widths())


@dataclass(frozen=True)
class Unit:
    """A `Q_word` unit kind: its role (``output`` or ``committed``), out bits, IR gates, MACs, two-input gates and ANDs."""
    kind: str
    out_bits: int
    ir_gates: int
    macs: int
    gates2: int
    ands: int


def qword(fn, memo: dict, cache: dict) -> dict:
    """One activation of `fn` under Q_word{16}: ``units`` Counter[Unit], ``committed`` (derived committed bits), ``returned``
    and ``params`` bits."""
    if fn.id in cache:
        return cache[fn.id]
    cut = evaluate_definition(fn, 16, memo)
    params, ret = fn.signature()
    out = {"units": Counter(), "committed": 0, "returned": bits(ret), "params": sum(bits(t) for _, t in params)}
    if cut.parts is not None:
        for (c, m), _node in zip(cut.parts, fn.body.nodes):
            if c is None:
                continue
            sub = qword(c.fn, memo, cache)
            for k, n in sub["units"].items():
                out["units"][k] += n * m
            out["committed"] += sub["committed"] * m
    elif not hasattr(fn, "body"):
        if cut.units:
            w = fn.signature()[1].w
            g2, a = work(fn)
            out["units"][Unit("output", w, 1, STEP_MACS.get(fn.id, 0), g2, a)] += 1
            out["committed"] = w
    else:
        g = CallGraph(fn)
        owner = cut.flat
        U = len(cut.kinds)
        ob, _og = boundary_widths(g, owner, U)
        ir, macs, g2s, ands = [0] * U, [0] * U, [0] * U, [0] * U
        c, call = call_scope(fn)
        for i, j in enumerate(g.keep):
            p = c.gate(call.offset + j).prim
            u = owner[i]
            g2, a = work(p)
            ir[u] += 1
            macs[u] += STEP_MACS.get(p.id, 0)
            g2s[u] += g2
            ands[u] += a
        for u, kind in enumerate(cut.kinds):
            out["units"][Unit(kind, ob[u], ir[u], macs[u], g2s[u], ands[u])] += 1
        out["committed"] = sum(g.width[i] for i in derived_committed(g, owner))
    cache[fn.id] = out
    return out


def totals(units: Counter) -> dict:
    return {"units": sum(units.values()), "macs": sum(u.macs * n for u, n in units.items()),
            "gates2": sum(u.gates2 * n for u, n in units.items()), "ands": sum(u.ands * n for u, n in units.items()),
            "ir_gates": sum(u.ir_gates * n for u, n in units.items())}


# ------------------------------------------------------------------------------------------------------------- the IR


def measure(keys: list[int]) -> dict:
    """The served step's ops at Llama-3.2-1B: each distinct root Call's Q_word units and Call-level Boolean work, and
    Attention_v4 at each of ``keys``."""
    r = build_program(phi(), "step", M=1, pos0=8, with_desc=False)
    calls, fns, order = Counter(), {}, []
    for n in r.program.root.body.nodes:
        if is_input(n.fn):
            continue
        calls[n.fn.id] += 1
        fns[n.fn.id] = n.fn
        order.append(n.fn.id)
    memo, cache, ops = {}, {}, {}
    for fid, fn in fns.items():
        t0 = time.time()
        q = qword(fn, memo, cache)
        g2, a = work(fn)
        ops[fid] = {"family": fid.split("{")[0], "calls": calls[fid], "returned_bits": q["returned"],
                    "returned_leaves": list(fn.signature()[1].leaf_widths()), "params_bits": q["params"],
                    "qword_committed_bits": q["committed"], "call_gates2": g2, "call_ands": a,
                    "qword": totals(q["units"]), "units": [[*asdict(u).values(), n] for u, n in q["units"].most_common()],
                    "s": round(time.time() - t0, 1)}
        print(f"{fid[:80]}: units {ops[fid]['qword']['units']} call ANDs {a:,} ({ops[fid]['s']} s)", flush=True)
    req = build_program(phi(), "request", LP=1, T=2, with_desc=False)
    base = next(n.fn for n in req.program.root.body.nodes if n.fn.id.startswith("Attention_v4{T=3"))
    attn = {}
    for T in keys:
        t0 = time.time()
        fn = bind(base.defn, **{**base.bindings, "T": T})
        q = qword(fn, {}, {})
        g2, a = work(fn)
        attn[T] = {"returned_bits": q["returned"], "params_bits": q["params"], "qword_committed_bits": q["committed"],
                   "call_gates2": g2, "call_ands": a, "qword": totals(q["units"]),
                   "units": [[*asdict(u).values(), n] for u, n in q["units"].most_common()], "s": round(time.time() - t0, 1)}
        print(f"Attention_v4 T={T}: Q_word units {attn[T]['qword']['units']}, call ANDs {a:,} ({attn[T]['s']} s)", flush=True)
    return {"ops": ops, "order": order, "attention": attn}


# ------------------------------------------------------------------------------------------------------- the partitions


@dataclass
class Block:
    """A proof block kind: ``n`` a token (``per`` "layer": a token-layer, ×16), its MACs, two-input gates and ANDs (compute),
    the rows it reads (bytes each, or ``(count, bytes)`` for ``count`` rows; ``weights`` the registered weight bytes it reads,
    in ``weight_rows`` rows) and the bytes it commits (one row)."""
    name: str
    n: float
    macs: int
    gates2: int
    ands: int
    commits: int
    reads: list[int | tuple[int, int]] = field(default_factory=list)
    weights: int = 0
    per: str = "layer"
    note: str = ""
    weight_rows: int = 1

    def rows(self) -> list[int]:
        out = [b for r in self.reads for b in ([r[1]] * r[0] if isinstance(r, tuple) else [r])]
        return (out + ([self.weights // self.weight_rows] * self.weight_rows if self.weights else [])
                + ([self.commits] if self.commits else []))

    def opening_ands(self) -> int:
        return sum(row_ands(b) for b in self.rows())

    def row(self) -> dict:
        boundary = self.commits * 8
        k_log = CS.block_k_log(self.ands, self.rows())
        return {"block": self.name, "per": self.per, "n_per_token": self.n * (LAYERS if self.per == "layer" else 1),
                "macs": self.macs, "gates2": self.gates2, "ands": self.ands, "opening_ands_est": self.opening_ands(),
                "boundary_bits": boundary, "size_class": unit_class(boundary) if boundary else None,
                "reads_bytes": sum(r[0] * r[1] if isinstance(r, tuple) else r for r in self.reads), "weight_bytes": self.weights,
                "k_log_est": k_log, "over_cap": k_log > CS.K_MAX, "note": self.note}


class Facts:
    """Per-op constants read off `measure`'s IR record."""

    def __init__(self, m: dict, T: int) -> None:
        ops = m["ops"]

        def op(prefix: str) -> dict:
            hit = [v for k, v in ops.items() if k.startswith(prefix)]
            if len(hit) != 1:
                raise LookupError(f"{prefix}: {len(hit)} ops")
            return hit[0]

        def unit(o: dict, kind: str, out_bits: int | None = None) -> dict:
            hit = [u for u in o["units"] if u[0] == kind and (out_bits is None or u[1] == out_bits)]
            u = max(hit, key=lambda u: u[-1])
            return {"ir_gates": u[2], "macs": u[3], "gates2": u[4], "ands": u[5]}

        self.ops = ops
        self.mm = {(2048, 3072): op("ScaledMmS8_v1{K=2048,N=3072"), (2048, 2048): op("ScaledMmS8_v1{K=2048,N=2048"),
                   (2048, 16384): op("ScaledMmS8_v1{K=2048,N=16384"), (8192, 2048): op("ScaledMmS8_v1{K=8192")}
        #: an output coordinate of the int8 GEMM at K: the Q_word unit
        self.coord = {2048: unit(self.mm[(2048, 2048)], "output"), 8192: unit(self.mm[(8192, 2048)], "output")}
        self.quant = {2048: op("Int8RowQuant_v1{K=2048"), 8192: op("Int8RowQuant_v1{K=8192")}
        self.norm, self.norm0 = op("RMSNormFusedCuda_v2"), op("RMSNormTriton_v1")
        self.rope_q, self.rope_k = op("RoPE_v1{NHEADS=32"), op("RoPE_v1{NHEADS=8")
        self.silu, self.embed = op("SiluMul_v2"), op("Embedding_v1")
        self.head, self.select = op("Gemm_v2"), op("TokenSelect_v1")
        self.attn = m["attention"][str(T)] if str(T) in m["attention"] else m["attention"][T]
        self.T = T
        #: the served MACs a token (the LM head apart): every root Call's tensor-core MACs, attention's at T keys
        self.served_macs = LAYERS * self.attn["qword"]["macs"] + sum(
            o["calls"] * o["qword"]["macs"] for k, o in ops.items() if not k.startswith(("Gemm_v2", "Attention_v4")))
        #: Int8RowQuant's per-code cost (a unit holds two codes) and its scale unit's
        self.code = {K: {k: v // 2 for k, v in unit(self.quant[K], "output", 16).items()} for K in (2048, 8192)}
        self.scale = {K: unit(self.quant[K], "output", 32) for K in (2048, 8192)}
        #: RoPE a rotated word, SiluMul an output word, the norm's x·r·γ word and its residual-add word
        self.rope_word = unit(self.rope_q, "output")
        self.silu_word = unit(self.silu, "output")
        nu = sorted((u for u in self.norm["units"] if u[0] == "output"), key=lambda u: -u[5])
        self.norm_word = {"gates2": nu[0][4], "ands": nu[0][5]}
        self.add_word = {"gates2": nu[1][4], "ands": nu[1][5]}
        self.rsqrt = unit(self.norm, "committed", 32)
        H = 2048
        self.bf16_row, self.int8_row = 2 * H, H + 4  # a hidden row in bf16; int8 codes and their f32 scale
        self.mlp_bf16, self.mlp_int8 = 2 * 8192, 8192 + 4
        self.kv_row = 2 * HEAD_DIM  # a head's k or v, bf16

    def call(self, o: dict) -> tuple[int, int]:
        return o["call_gates2"], o["call_ands"]

    def tile(self, K: int, n: int, prologue: tuple[int, int] = (0, 0), epilogue: tuple[int, int] = (0, 0)) -> tuple:
        c = self.coord[K]
        return n * c["macs"], n * c["gates2"] + prologue[0] + epilogue[0], n * c["ands"] + prologue[1] + epilogue[1]

    def attn_head(self) -> tuple[int, int, int]:
        """One query head of the decode token over T keys: Attention_v4's Call-level work / 32 (its heads are alike)."""
        return self.attn["qword"]["macs"] // HEADS, self.attn["call_gates2"] // HEADS, self.attn["call_ands"] // HEADS


def gemm_blocks(F: Facts, name: str, K: int, N: int, n: int, reads: list[int], out_bytes: int, prologue=(0, 0),
                epilogue=(0, 0), note: str = "") -> Block:
    macs, g2, a = F.tile(K, n, prologue, epilogue)
    return Block(name, N / n, macs, g2, a, out_bytes, reads, n * (K + 4), note=note, weight_rows=n)


def partitions(F: Facts) -> dict[str, list[Block]]:
    T = F.T
    X, XQ, S, SQ, GU = F.bf16_row, F.int8_row, F.mlp_bf16, F.mlp_int8, 2 * F.mlp_bf16
    KV = F.kv_row
    t2, t8 = TILE[2048], TILE[8192]
    hm, hg, ha = F.attn_head()
    norm, quant2, quant8, silu = F.call(F.norm), F.call(F.quant[2048]), F.call(F.quant[8192]), F.call(F.silu)
    rope_head = (HEAD_DIM * F.rope_word["gates2"], HEAD_DIM * F.rope_word["ands"])
    keys = (T, 2 * KV)  # a q head reads its KV head's T keys and values, a row a key
    plus = lambda *ws: (sum(w[0] for w in ws), sum(w[1] for w in ws))  # noqa: E731
    out: dict[str, list[Block]] = {}

    def heads(name: str, by_window: bool, extra=(0, 0, 0), commits: int = KV, reads: list[int] | None = None,
              last_only: bool = False) -> list[Block]:
        """A layer's 32 query heads over T keys, each cut into key blocks (the window's FA3 BN of 192, else as few as keep a
        block under the cap), a non-last key block committing its running state (`STATE`); ``last_only``: only the last
        key block, reading the states an earlier pass committed."""
        kb = math.ceil(T / FA3_BN) if by_window else max(1, math.ceil((ha + extra[2]) / CAP_ANDS))
        reads = [KV, keys] if reads is None else reads
        if kb == 1:
            return [Block(f"{name} T={T}", HEADS, hm + extra[0], hg + extra[1], ha + extra[2], commits, reads)]
        per_block = (math.ceil(T / kb), 2 * KV)
        last = Block(f"{name} T={T}, last key block of {kb}", HEADS, hm // kb + extra[0], hg // kb + extra[1],
                     ha // kb + extra[2], commits, [STATE, per_block] + reads[2:])
        return [last] if last_only else [
            Block(f"{name} T={T}, key block 1..{kb - 1} of {kb}", HEADS * (kb - 1), hm // kb, hg // kb, ha // kb, STATE,
                  [KV, per_block], note="commits the running state (m, l, O): an estimate of its bytes"), last]

    # today's: each Call a block, every Call output committed
    out["a0"] = [
        Block("Int8RowQuant{2048} x3", 3, 0, *quant2, XQ, [X]),
        Block("ScaledMmS8 qkv", 1, *F.tile(2048, 3072), 2 * 3072, [XQ], 3072 * 2052),
        Block("RoPE q", 1, 0, *F.call(F.rope_q), X, [X]),
        Block("RoPE k", 1, 0, *F.call(F.rope_k), 2 * 512, [2 * 512]),
        Block(f"Attention T={T}", 1, F.attn["qword"]["macs"], F.attn["call_gates2"], F.attn["call_ands"], X,
              [X, 2 * 512, 2 * 512, (T - 1) * 4 * 512]),
        Block("ScaledMmS8 o", 1, *F.tile(2048, 2048), X, [XQ], 2048 * 2052),
        Block("RMSNormFused x2", 2, 0, *norm, 2 * X, [X, X]),
        Block("ScaledMmS8 gate_up", 1, *F.tile(2048, 16384), GU, [XQ], 16384 * 2052),
        Block("SiluMul", 1, 0, *silu, S, [GU]),
        Block("Int8RowQuant{8192}", 1, 0, *quant8, SQ, [S]),
        Block("ScaledMmS8 down", 1, *F.tile(8192, 2048), X, [SQ], 2048 * 8196),
    ]

    # the window circuit's grain: GEMM slices, the rest a unit an op (RoPE a head, attention a head's key block)
    out["w"] = [
        Block("Int8RowQuant{2048}", 3, 0, *quant2, XQ, [X]),
        gemm_blocks(F, "qkv slice {2048,32}", 2048, 3072, t2, [XQ], 2 * t2),
        Block("RoPE head", HEADS + KV_HEADS, 0, *rope_head, KV, [KV]),
        *heads("attention head", True),
        gemm_blocks(F, "o slice {2048,32}", 2048, 2048, t2, [XQ], 2 * t2),
        Block("RMSNormFused", 2, 0, *norm, 2 * X, [X, X]),
        gemm_blocks(F, "gate_up slice {2048,32}", 2048, 16384, t2, [XQ], 2 * t2),
        Block("SiluMul", 1, 0, *silu, S, [GU]),
        Block("Int8RowQuant{8192}", 1, 0, *quant8, SQ, [S]),
        gemm_blocks(F, "down slice {8192,8}", 8192, 2048, t8, [SQ], 2 * t8),
    ]

    # (b) norm and quantize into the GEMM tile that reads them, SiluMul into down's; one tile also commits the residual
    out["b"] = [
        gemm_blocks(F, "qkv tile + norm1 + quant", 2048, 3072, t2, [X, X], 2 * t2, prologue=plus(norm, quant2)),
        Block("qkv tile 0's residual r1", 1, 0, 0, 0, X, note="the norm's residual output, committed by one qkv tile"),
        Block("RoPE head", HEADS + KV_HEADS, 0, *rope_head, KV, [KV]),
        *heads("attention head", False),
        gemm_blocks(F, "o tile + quant", 2048, 2048, t2, [X], 2 * t2, prologue=quant2),
        gemm_blocks(F, "gate_up tile + norm2 + quant", 2048, 16384, t2, [X, X], 2 * t2, prologue=plus(norm, quant2)),
        Block("gate_up tile 0's residual r2", 1, 0, 0, 0, X),
        gemm_blocks(F, "down tile + SiluMul + quant", 8192, 2048, t8, [GU], 2 * t8, prologue=plus(silu, quant8)),
    ]

    # (c) dot-product inputs (bf16) and outputs only, quantize in the GEMM's prologue, the residual stream kept
    out["c"] = [
        Block("RMSNormFused (x, r)", 2, 0, *norm, 2 * X, [X, X]),
        gemm_blocks(F, "qkv tile + quant", 2048, 3072, t2, [X], 2 * t2, prologue=quant2),
        Block("RoPE head", HEADS + KV_HEADS, 0, *rope_head, KV, [KV]),
        *heads("attention head", False),
        gemm_blocks(F, "o tile + quant", 2048, 2048, t2, [X], 2 * t2, prologue=quant2),
        gemm_blocks(F, "gate_up tile + quant", 2048, 16384, t2, [X], 2 * t2, prologue=quant2),
        Block("SiluMul", 1, 0, *silu, S, [GU]),
        gemm_blocks(F, "down tile + quant", 8192, 2048, t8, [S], 2 * t8, prologue=quant8),
    ]

    # (d) c with the GEMM inputs committed as int8 codes: the norm and SiluMul blocks quantize; attention's o stays bf16
    out["d"] = [
        Block("RMSNormFused + quant (xq, r)", 2, 0, *plus(norm, quant2), XQ + X, [X, X]),
        gemm_blocks(F, "qkv tile", 2048, 3072, t2, [XQ], 2 * t2),
        Block("RoPE head", HEADS + KV_HEADS, 0, *rope_head, KV, [KV]),
        *heads("attention head", False),
        gemm_blocks(F, "o tile + quant", 2048, 2048, t2, [X], 2 * t2, prologue=quant2),
        gemm_blocks(F, "gate_up tile", 2048, 16384, t2, [XQ], 2 * t2),
        Block("SiluMul + quant", 1, 0, *plus(silu, quant8), SQ, [GU]),
        gemm_blocks(F, "down tile", 8192, 2048, t8, [SQ], 2 * t8),
    ]

    # (e1) epilogue fusion and the residual once a layer, no recompute
    rope_ep = (F.rope_word["gates2"] * t2, F.rope_word["ands"] * t2)  # a q or k tile: every output word rotated
    silu_ep = (F.silu_word["gates2"] * t2 // 2, F.silu_word["ands"] * t2 // 2)  # a tile of t2/2 (g, u) pairs
    add = (2048 * F.add_word["gates2"], 2048 * F.add_word["ands"])
    e_common = [
        Block("norm1 + quant (r += a + d; xq1, r)", 1, 0, *plus(norm, quant2, add), XQ + X, [X, X, X],
              note="adds the previous layer's a and d to the residual it committed"),
        gemm_blocks(F, "q/k tile + RoPE", 2048, 2560, t2, [XQ], 2 * t2, epilogue=rope_ep,
                    note="a tile of whole rotary pairs; commits q' and k'"),
        gemm_blocks(F, "v tile", 2048, 512, t2, [XQ], 2 * t2),
    ]
    out["e1"] = e_common + [
        *heads("attention head", False),
        gemm_blocks(F, "o tile + quant", 2048, 2048, t2, [X], 2 * t2, prologue=quant2),
        Block("norm2 + quant (xq2)", 1, 0, *plus(norm, quant2), XQ, [X, X], note="r2 = r1 + a, not committed"),
        gemm_blocks(F, "gate_up tile + SiluMul", 2048, 16384, t2, [XQ], t2, epilogue=silu_ep,
                    note="g and u columns paired; commits s, half of gu"),
        Block("Int8RowQuant{8192} (sq)", 1, 0, *quant8, SQ, [S]),
        gemm_blocks(F, "down tile", 8192, 2048, t8, [SQ], 2 * t8),
    ]

    # (e2) e1 with max-first recompute: o and s never committed, only their blocks' partial maxima, then int8 codes
    code8 = (F.code[8192]["gates2"] * t2 // 2, F.code[8192]["ands"] * t2 // 2)
    code2_head = (F.code[2048]["gates2"] * HEAD_DIM, F.code[2048]["ands"] * HEAD_DIM)
    mx2, mx8 = (F.scale[2048]["gates2"], F.scale[2048]["ands"]), (F.scale[8192]["gates2"], F.scale[8192]["ands"])
    out["e2"] = e_common + [
        *heads("attention pass 1 (partial max)", False, commits=4),
        Block("o scale (max of 32 partial maxima)", 1, 0, *mx2, 4, [4] * HEADS,
              note="ANDs: Int8RowQuant{2048}'s scale unit, an upper bound"),
        *heads("attention pass 2 (recompute, oq codes)", False, extra=(0, *code2_head), commits=HEAD_DIM,
               reads=[KV, keys, 4], last_only=True),
        gemm_blocks(F, "o tile", 2048, 2048, t2, [XQ], 2 * t2),
        out["e1"][-4],
        gemm_blocks(F, "gate_up pass 1: tile + SiluMul, partial max", 2048, 16384, t2, [XQ], 4, epilogue=silu_ep),
        Block("s scale (max of 512 partial maxima)", 1, 0, *mx8, 4, [4 * 16384 // t2],
              note="ANDs: Int8RowQuant{8192}'s scale unit, an upper bound"),
        gemm_blocks(F, "gate_up pass 2: recompute + sq codes", 2048, 16384, t2, [XQ, 4], t2 // 2,
                    epilogue=plus(silu_ep, code8)),
        out["e1"][-1],
    ]

    # (e3) e1 at the finest GEMM units, each reading its weight column and sharing its slice's rows
    out["e3"] = [
        e_common[0],
        gemm_blocks(F, "q/k rotary pair + RoPE", 2048, 2560, 2, [XQ], 4, epilogue=(2 * F.rope_word["gates2"], 2 * F.rope_word["ands"])),
        gemm_blocks(F, "v coordinate", 2048, 512, 1, [XQ], 2),
        *heads("attention head", False),
        Block("Int8RowQuant{2048} (oq)", 1, 0, *quant2, XQ, [X]),
        gemm_blocks(F, "o coordinate", 2048, 2048, 1, [XQ], 2),
        out["e1"][-4],
        gemm_blocks(F, "gate_up (g, u) pair + SiluMul", 2048, 16384, 2, [XQ], 2, epilogue=(F.silu_word["gates2"], F.silu_word["ands"])),
        out["e1"][-2],
        gemm_blocks(F, "down coordinate", 8192, 2048, 1, [SQ], 2),
    ]
    return out


def first_last(F: Facts, name: str) -> list[Block]:
    """Blocks outside the layers, a token: the embedding read, and in ``a0`` and ``w`` the input norm (elsewhere layer 0's
    norm1 is it; the last layer's end-of-layer norm is the final norm, counted in the layer)."""
    ands, _k, src = CS.MEASURED[EMBED_READ]
    emb = [Block(EMBED_READ, 1, 0, 0, ands, F.bf16_row, [4], 2 * 2048, per="token",
                 note=f"ANDs measured ({src}); two-input gates unmeasured. The served Embedding_v1 Call's Boolean version "
                      f"muxes the whole table: {F.embed['call_ands']:,} ANDs")]
    if name in ("a0", "w"):
        emb.append(Block("RMSNormTriton (input norm)", 1, 0, *F.call(F.norm0), F.bf16_row, [F.bf16_row], per="token"))
    return emb


def head_blocks(F: Facts, name: str) -> list[Block]:
    """The LM head and the token select, a sampled token (every decode position samples): in ``a0`` the two Calls; in
    ``e1`` and ``e2`` each head slice ``Gemm_v2{2048,16}`` selects its best logit in its epilogue and commits it and its
    index, and one block combines them; elsewhere the window's slices commit the logits and its select units read them."""
    V, n = 128256, 16
    c = [u for u in F.head["units"] if u[0] == "output"][0]
    co = {"macs": c[3], "gates2": c[4], "ands": c[5]}
    slices = V // n
    if name == "a0":
        return [Block("LM head Gemm_v2{2048,128256}", 1, co["macs"] * V, F.head["call_gates2"], F.head["call_ands"],
                      2 * V, [F.bf16_row], V * 2 * 2048, per="sampled"),
                Block("TokenSelect_v1", 1, 0, *F.call(F.select), 4, [2 * V], per="sampled")]
    groups_ands, _k, gsrc = CS.MEASURED["TokenSelectGroups_v1{C=2672,G=16}"]
    comb_ands, _k, csrc = CS.MEASURED["TokenSelectCombine_v2{V=128256,C=16}"]
    per_group = groups_ands // 167
    if name in ("e1", "e2", "e3"):
        return [Block("LM head slice {2048,16} + best of 16", slices, n * co["macs"], n * co["gates2"],
                      n * co["ands"] + per_group, 2 + 4, [F.bf16_row], n * 2 * 2048, per="sampled",
                      note="epilogue ANDs: TokenSelectGroups' a group of 16 (measured / 167); the index's 4 bytes an estimate"),
                Block("token select over the slices' bests", 1, 0, 0, comb_ands, 4, [6 * slices], per="sampled",
                      note=f"ANDs: TokenSelectCombine_v2's ({csrc}), over 167 x 48 bests")]
    return [Block("LM head slice {2048,16}", slices, n * co["macs"], n * co["gates2"], n * co["ands"], 2 * n, [F.bf16_row],
                  n * 2 * 2048, per="sampled"),
            Block("TokenSelectGroups_v1{C=2672,G=16}", 48, 0, 0, groups_ands, 167 * 6, [2 * 2672], per="sampled",
                  note=f"ANDs measured ({gsrc}); commits 167 bests and indices, the index's 4 bytes an estimate"),
            Block("TokenSelectCombine_v2", 1, 0, 0, comb_ands, 4, [167 * 6] * 48, per="sampled")]


def summarize(F: Facts, name: str, blocks: list[Block]) -> dict:
    head = [b.row() for b in head_blocks(F, name)]
    head_beta = sum(r["n_per_token"] * r["boundary_bits"] / 8 for r in head)
    rows = [b.row() for b in blocks + first_last(F, name)]
    beta = sum(r["n_per_token"] * r["boundary_bits"] / 8 for r in rows)
    macs = sum(r["n_per_token"] * r["macs"] for r in rows)
    n = sum(r["n_per_token"] for r in rows)
    ands = sum(r["n_per_token"] * r["ands"] for r in rows)
    w = sum(r["n_per_token"] * r["ands"] ** 2 for r in rows)
    wo = sum(r["n_per_token"] * (r["ands"] + r["opening_ands_est"]) ** 2 for r in rows)
    ao = sum(r["n_per_token"] * (r["ands"] + r["opening_ands_est"]) for r in rows)
    # a unit is drawn by its work (its ANDs) and costs its block, openings and padding included
    cost_ands = sum(r["n_per_token"] * r["ands"] * (r["ands"] + r["opening_ands_est"]) for r in rows) / ands
    cost_bits = sum(r["n_per_token"] * r["ands"] * 2 ** r["k_log_est"] for r in rows) / ands
    return {"partition": name, "keys": F.T, "beta_bytes_per_token": beta, "beta_bytes_per_kmac": 1000 * beta / F.served_macs,
            "served_macs_per_token": F.served_macs, "proved_macs_per_token": macs, "blocks_per_token": n,
            "ands_per_token": ands, "blocks_over_cap_per_token": sum(r["n_per_token"] for r in rows if r["over_cap"]),
            "and_weighted_mean_block_ands": w / ands, "with_openings_and_weighted_mean": wo / ao,
            "expected_proved_ands_per_window_est": DRAWS * w / ands,
            "expected_proved_ands_with_openings_per_window_est": DRAWS * cost_ands,
            "expected_proved_bits_per_window_est": DRAWS * cost_bits, "head_beta_bytes_per_sampled_token": head_beta,
            "head_ands_per_sampled_token": sum(r["n_per_token"] * r["ands"] for r in head),
            "beta_with_head_bytes_per_token": beta + head_beta, "blocks": rows, "head_blocks": head}


def qword_partition(m: dict, T: int) -> dict:
    """``a``: Q_word{16} over each root Call, the IR's derived committed set, attention at T keys."""
    ops, attn = m["ops"], m["attention"][str(T)] if str(T) in m["attention"] else m["attention"][T]
    units, beta_bits, macs = Counter(), 0, 0
    for fid, o in ops.items():
        if fid.startswith(("Gemm_v2", "TokenSelect", "Attention_v4")):
            continue
        n = o["calls"] if not fid.startswith("Embedding") else 1
        src = o["units"]
        beta_bits += n * o["qword_committed_bits"]
        for u in src:
            units[tuple(u[:6]) + (fid.split("{")[0],)] += n * u[6]
    beta_bits += LAYERS * attn["qword_committed_bits"]
    for u in attn["units"]:
        units[tuple(u[:6]) + ("Attention_v4",)] += LAYERS * u[6]
    rows = [{"op": k[6], "kind": k[0], "boundary_bits": k[1], "size_class": unit_class(k[1]), "ir_gates": k[2], "macs": k[3],
             "gates2": k[4], "ands": k[5], "n_per_token": n} for k, n in sorted(units.items(), key=lambda kv: -kv[1])]
    macs = sum(r["macs"] * r["n_per_token"] for r in rows)
    ands = sum(r["ands"] * r["n_per_token"] for r in rows)
    w = sum(r["n_per_token"] * r["ands"] ** 2 for r in rows)
    return {"partition": "a", "keys": T, "beta_bytes_per_token": beta_bits / 8, "beta_bytes_per_kmac": 1000 * beta_bits / 8 / macs,
            "served_macs_per_token": macs, "proved_macs_per_token": macs,
            "blocks_per_token": sum(r["n_per_token"] for r in rows), "ands_per_token": ands,
            "and_weighted_mean_block_ands": w / ands, "expected_proved_ands_per_window_est": DRAWS * w / ands,
            "head_beta_bytes_per_sampled_token": sum(o["qword_committed_bits"] for k, o in ops.items()
                                                     if k.startswith(("Gemm_v2", "TokenSelect"))) / 8,
            "blocks": rows}


def lm_head(m: dict) -> dict:
    ops = m["ops"]
    head = next(v for k, v in ops.items() if k.startswith("Gemm_v2"))
    sel = next(v for k, v in ops.items() if k.startswith("TokenSelect"))
    return {"logits_bytes_per_sampled_token": head["returned_bits"] / 8, "macs": head["qword"]["macs"],
            "head_call_ands": head["call_ands"], "select_call_ands": sel["call_ands"],
            "coordinate": [u for u in head["units"] if u[0] == "output"][0]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="18,35,144,288")
    ap.add_argument("--ir", default=None, help="reuse the IR record of an earlier --out")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    keys = [int(k) for k in a.keys.split(",")]
    m = json.loads(Path(a.ir).read_text())["ir"] if a.ir else measure(keys)
    res = {"ir": m, "lm_head": lm_head(m), "at": {}}
    for T in keys:
        F = Facts(m, T)
        res["at"][T] = [qword_partition(m, T)] + [summarize(F, k, v) for k, v in partitions(F).items()]
    for T in keys:
        print(f"\nkeys T={T}")
        for p in res["at"][T]:
            print(f"  {p['partition']:>3}: beta {p['beta_bytes_per_token']:>12,.0f} B/token {p['beta_bytes_per_kmac']:7.2f} B/kMAC"
                  f"  head {p.get('head_beta_bytes_per_sampled_token', 0):>9,.0f} B/sampled token"
                  f"  blocks/token {p['blocks_per_token']:>10,.0f}  AND-weighted mean block {p['and_weighted_mean_block_ands']:>13,.0f}")
    for T in keys:
        for p in res["at"][T]:
            print(f"\n[{p['partition']}] T={T}: beta {p['beta_bytes_per_token']:,.0f} B/token; blocks/token "
                  f"{p['blocks_per_token']:,.0f}; ANDs/token {p['ands_per_token']:,.0f}; AND-weighted mean "
                  f"{p['and_weighted_mean_block_ands']:,.0f}")
            for r in p["blocks"][:14]:
                print("   ", {k: r.get(k) for k in ("block", "op", "kind", "n_per_token", "macs", "gates2", "ands",
                                                    "opening_ands_est", "boundary_bits", "size_class", "k_log_est", "over_cap")
                              if k in r})
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
