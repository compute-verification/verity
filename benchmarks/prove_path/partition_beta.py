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
  17.4 M ANDs and 64 weight columns, k_log 28.
* ``e2``: ``e1`` with max-first recompute (cut.py's ``Q_word`` v2 recompute): a gate_up tile commits its partial max only,
  and a code block recomputes the tile and writes its int8 codes, so s is never committed; attention likewise writes oq.
* ``e3``: ``e1`` at the finest GEMM units: a unit a coordinate, or a rotary or (g, u) pair where its epilogue pairs columns,
  each reading its own weight column and its slice's shared rows, and o quantized in a block of its own (a coordinate
  can't afford the prologue). A unit is drawn by its work and costs its block, so smaller units prove less a window. Its
  2-byte rows are gathered for their consumers, and a head's q and a KV head's k|v packed into one row each.

Every number is the IR's: widths from the Calls' signatures, MACs from their tensor-core steps, ANDs and two-input gates from
each primitive's Boolean version (census ``boolean=boolean_version``), summed over a unit's gates (`Q_word` units, tile
slices) or the whole Call's (a block holding whole ops). A block commits one row and reads each value as the rows its
writers committed (a GEMM output a row a tile, a key's k and v the rows of the tiles or the RoPE head that wrote them, a
weight column a row). C-Flock gives each row a block reads or writes its own 2^18-bit slot, so a read of more than
`GATHER_FAN` rows goes through gather blocks, copies that re-commit `GATHER_FAN` rows as one. A block's opening ANDs are
`class_size.opening_ands` of its rows, an estimate of the in-circuit hashing; its ``k_log_est`` is `class_size.block_k_log`
of its ANDs and those rows, and ``over_cap`` marks one past C-Flock's 2^27-bit block. A unit is drawn by its compute's ANDs,
a copy (gather, pack) by its openings'.

Attention's unit is a knob: ``group`` query heads of a KV head (1 or 4) over ``key_block`` keys (0: the fewest key blocks
that fit 2^27 bits; the window's FA3 BN of 192 in ``w``), each non-last key block committing its running state; and in
(e1) and (e2), optionally, the packs (e3) has. ``--key-blocks`` and ``--groups`` sweep it.

    python partition_beta.py [--keys 18,35,144,288,512] [--key-blocks 0,32,64,128] [--groups 1,4] [--out FILE.json]
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
#: rows a gather block reads: a consumer of more rows than this reads gathered rows of this many, since C-Flock gives each row
#: a unit reads its own 2^18-bit slot (one level: 2,048 rows read as 16, the LM head's 8,016 bests as 63)
GATHER_FAN = 128
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
    #: drawn by its openings' ANDs rather than its compute's: a copy, which a law by work would never draw
    drawn_by_openings: bool = False

    def rows(self) -> list[int]:
        out = [b for r in self.reads for b in ([r[1]] * r[0] if isinstance(r, tuple) else [r])]
        return (out + ([self.weights // self.weight_rows] * self.weight_rows if self.weights else [])
                + ([self.commits] if self.commits else []))

    def opening_ands(self) -> int:
        return sum(CS.opening_ands(b) for b in self.rows())

    def row(self) -> dict:
        boundary = self.commits * 8
        k_log = CS.block_k_log(self.ands, self.rows())
        opening = self.opening_ands()
        return {"block": self.name, "per": self.per, "n_per_token": self.n * (LAYERS if self.per == "layer" else 1),
                "macs": self.macs, "gates2": self.gates2, "ands": self.ands, "opening_ands_est": opening,
                "drawn_ands": opening if self.drawn_by_openings else self.ands, "rows": len(self.rows()),
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


def gather(name: str, rows: int, nbytes: int, per: str = "layer") -> tuple[list[Block], list[tuple[int, int]]]:
    """``rows`` rows of ``nbytes`` bytes, past `GATHER_FAN`, gathered `GATHER_FAN` to a row (the last the rest): the gather
    blocks (copies, drawn by their openings) and the rows a consumer reads, ``(count, bytes)`` each."""
    if rows <= GATHER_FAN:
        return [], [(rows, nbytes)]
    full, rest = divmod(rows, GATHER_FAN)
    if full + bool(rest) > GATHER_FAN:
        raise ValueError(f"{name}: {rows} rows need a second gather level")
    parts = [(full, GATHER_FAN)] + ([(1, rest)] if rest else [])
    return ([Block(f"{name} gather ({k} rows of {nbytes} B)", n, 0, 0, 0, k * nbytes, [(k, nbytes)], per=per,
                   drawn_by_openings=True) for n, k in parts], [(n, k * nbytes) for n, k in parts])


def gemm_blocks(F: Facts, name: str, K: int, N: int, n: int, reads: list[int], out_bytes: int, prologue=(0, 0),
                epilogue=(0, 0), note: str = "") -> Block:
    macs, g2, a = F.tile(K, n, prologue, epilogue)
    return Block(name, N / n, macs, g2, a, out_bytes, reads, n * (K + 4), note=note, weight_rows=n)


def partitions(F: Facts, key_block: int = 0, group: int = 1, pack: bool | None = None) -> dict[str, list[Block]]:
    """The partitions at keys ``F.T``, attention's unit ``group`` query heads of a KV head over at most ``key_block`` keys
    (0: the fewest key blocks that fit 2^27 bits), and, in (e1) and (e2) (``pack``, default off) and (e3) (default on), a
    head's q and a KV head's k|v re-committed as one row each for the key blocks to read."""
    T = F.T
    X, XQ, S, SQ, GU = F.bf16_row, F.int8_row, F.mlp_bf16, F.mlp_int8, 2 * F.mlp_bf16
    KV = F.kv_row
    t2, t8 = TILE[2048], TILE[8192]
    hm, hg, ha = F.attn_head()
    norm, quant2, quant8, silu = F.call(F.norm), F.call(F.quant[2048]), F.call(F.quant[8192]), F.call(F.silu)
    rope_head = (HEAD_DIM * F.rope_word["gates2"], HEAD_DIM * F.rope_word["ands"])
    #: a head's 64 bf16 words as the writers commit them: a RoPE head's one row, or a 32-column GEMM tile's two
    by_head, by_tile = [KV], [2 * t2] * (HEAD_DIM // t2)
    plus = lambda *ws: (sum(w[0] for w in ws), sum(w[1] for w in ws))  # noqa: E731
    out: dict[str, list[Block]] = {}

    def heads(name: str, q_rows: list, kv_rows: list, extra=(0, 0, 0), commits: int = KV, extra_reads: tuple = (),
              last_only: bool = False, by_window: bool = False) -> list[Block]:
        """A layer's 32 query heads over T keys, ``group`` query heads of a KV head a unit, each cut into kb key blocks of
        ceil(T / kb) keys (``by_window``: the window's FA3 BN of 192, one head a unit). A block reads its heads' q
        (``q_rows`` a head), the running state but in the first, and its keys' k and v as their writers commit them
        (``kv_rows`` a key); a non-last block commits the running state (`STATE` a head). ``last_only``: only the last
        key block, reading the states an earlier pass committed."""
        g = 1 if by_window else group
        work, ext = [x * g for x in (hm, hg, ha)], [x * g for x in extra]
        tag = f" ({g} q heads)" if g > 1 else ""

        def blocks(kb: int) -> list[Block]:
            keys, q, st = [(math.ceil(T / kb), b) for b in kv_rows], [(g, b) for b in q_rows], [g * STATE]
            m, g2, a = (x // kb for x in work)
            if kb == 1:
                return [Block(f"{name}{tag} T={T}", HEADS // g, m + ext[0], g2 + ext[1], a + ext[2], g * commits,
                              q + keys + list(extra_reads))]
            last = Block(f"{name}{tag} T={T}, last key block of {kb}", HEADS // g, m + ext[0], g2 + ext[1], a + ext[2],
                         g * commits, q + st + keys + list(extra_reads))
            if last_only:
                return [last]
            note = "commits the running state (m, l, O): an estimate of its bytes"
            mid = [Block(f"{name}{tag} T={T}, key blocks 2..{kb - 1} of {kb}", HEADS // g * (kb - 2), m, g2, a,
                         g * STATE, q + st + keys, note=note)] if kb > 2 else []
            return [Block(f"{name}{tag} T={T}, first key block of {kb}", HEADS // g, m, g2, a, g * STATE, q + keys,
                          note=note), *mid, last]

        if by_window:
            return blocks(math.ceil(T / FA3_BN))
        if key_block:
            return blocks(math.ceil(T / key_block))
        for kb in range(1, T + 1):
            bs = blocks(kb)
            if max(CS.block_k_log(b.ands, b.rows()) for b in bs) <= CS.K_MAX:
                return bs
        return blocks(T)

    def packs(q_rows: list, kv_rows: list) -> list[Block]:
        """A head's q, and a KV head's k and v, re-committed as one row each: copies, drawn by their openings."""
        return [Block("q pack (one row a head)", HEADS, 0, 0, 0, KV, q_rows, drawn_by_openings=True),
                Block("k|v pack (one row a KV head)", KV_HEADS, 0, 0, 0, 2 * KV, kv_rows, drawn_by_openings=True)]

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

    # what a consumer reads of a GEMM's or attention's output: a row a tile (or a unit) that wrote it
    A2, D8, GU_T, S_T = (2048 // t2, 2 * t2), (2048 // t8, 2 * t8), (16384 // t2, 2 * t2), (16384 // t2, t2)
    O_H = (HEADS // group, KV * group)
    #: the reads past `GATHER_FAN` rows, gathered once a layer: d, gu and s as their tiles write them
    (gd, D8g), (ggu, GUg), (gs, Sg) = gather("d", *D8), gather("gu", *GU_T), gather("s", *S_T)

    # the window circuit's grain: GEMM slices, the rest a unit an op (RoPE a head, attention a head's key block)
    out["w"] = [
        Block("Int8RowQuant{2048} (xq1, xq2)", 2, 0, *quant2, XQ, [X]),
        Block("Int8RowQuant{2048} (oq)", 1, 0, *quant2, XQ, [(HEADS, KV)]),
        gemm_blocks(F, "qkv slice {2048,32}", 2048, 3072, t2, [XQ], 2 * t2),
        Block("RoPE head", HEADS + KV_HEADS, 0, *rope_head, KV, by_tile),
        *heads("attention head", by_head, by_head + by_tile, by_window=True),
        gemm_blocks(F, "o slice {2048,32}", 2048, 2048, t2, [XQ], 2 * t2),
        Block("RMSNormFused after attention", 1, 0, *norm, 2 * X, [X, A2]),
        *gd,
        Block("RMSNormFused after the MLP", 1, 0, *norm, 2 * X, [X, *D8g]),
        gemm_blocks(F, "gate_up slice {2048,32}", 2048, 16384, t2, [XQ], 2 * t2),
        *ggu,
        Block("SiluMul", 1, 0, *silu, S, GUg),
        Block("Int8RowQuant{8192}", 1, 0, *quant8, SQ, [S]),
        gemm_blocks(F, "down slice {8192,8}", 8192, 2048, t8, [SQ], 2 * t8),
    ]

    # (b) norm and quantize into the GEMM tile that reads them, SiluMul into down's; one tile also commits the residual
    out["b"] = [
        *gd,
        gemm_blocks(F, "qkv tile + norm1 + quant", 2048, 3072, t2, [X, *D8g], 2 * t2, prologue=plus(norm, quant2)),
        Block("qkv tile 0's residual r1", 1, 0, 0, 0, X, note="the norm's residual output, committed by one qkv tile"),
        Block("RoPE head", HEADS + KV_HEADS, 0, *rope_head, KV, by_tile),
        *heads("attention head", by_head, by_head + by_tile),
        gemm_blocks(F, "o tile + quant", 2048, 2048, t2, [O_H], 2 * t2, prologue=quant2),
        gemm_blocks(F, "gate_up tile + norm2 + quant", 2048, 16384, t2, [X, A2], 2 * t2, prologue=plus(norm, quant2)),
        Block("gate_up tile 0's residual r2", 1, 0, 0, 0, X),
        *ggu,
        gemm_blocks(F, "down tile + SiluMul + quant", 8192, 2048, t8, GUg, 2 * t8, prologue=plus(silu, quant8)),
    ]

    # (c) dot-product inputs (bf16) and outputs only, quantize in the GEMM's prologue, the residual stream kept
    out["c"] = [
        Block("RMSNormFused (x, r) after attention", 1, 0, *norm, 2 * X, [X, A2]),
        *gd,
        Block("RMSNormFused (x, r) after the MLP", 1, 0, *norm, 2 * X, [X, *D8g]),
        gemm_blocks(F, "qkv tile + quant", 2048, 3072, t2, [X], 2 * t2, prologue=quant2),
        Block("RoPE head", HEADS + KV_HEADS, 0, *rope_head, KV, by_tile),
        *heads("attention head", by_head, by_head + by_tile),
        gemm_blocks(F, "o tile + quant", 2048, 2048, t2, [O_H], 2 * t2, prologue=quant2),
        gemm_blocks(F, "gate_up tile + quant", 2048, 16384, t2, [X], 2 * t2, prologue=quant2),
        *ggu,
        Block("SiluMul", 1, 0, *silu, S, GUg),
        gemm_blocks(F, "down tile + quant", 8192, 2048, t8, [S], 2 * t8, prologue=quant8),
    ]

    # (d) c with the GEMM inputs committed as int8 codes: the norm and SiluMul blocks quantize; attention's o stays bf16
    out["d"] = [
        Block("RMSNormFused + quant (xq, r) after attention", 1, 0, *plus(norm, quant2), XQ + X, [X, A2]),
        *gd,
        Block("RMSNormFused + quant (xq, r) after the MLP", 1, 0, *plus(norm, quant2), XQ + X, [X, *D8g]),
        gemm_blocks(F, "qkv tile", 2048, 3072, t2, [XQ], 2 * t2),
        Block("RoPE head", HEADS + KV_HEADS, 0, *rope_head, KV, by_tile),
        *heads("attention head", by_head, by_head + by_tile),
        gemm_blocks(F, "o tile + quant", 2048, 2048, t2, [O_H], 2 * t2, prologue=quant2),
        gemm_blocks(F, "gate_up tile", 2048, 16384, t2, [XQ], 2 * t2),
        *ggu,
        Block("SiluMul + quant", 1, 0, *plus(silu, quant8), SQ, GUg),
        gemm_blocks(F, "down tile", 8192, 2048, t8, [SQ], 2 * t8),
    ]

    # (e1) epilogue fusion and the residual once a layer, no recompute
    rope_ep = (F.rope_word["gates2"] * t2, F.rope_word["ands"] * t2)  # a q or k tile: every output word rotated
    silu_ep = (F.silu_word["gates2"] * t2 // 2, F.silu_word["ands"] * t2 // 2)  # a tile of t2/2 (g, u) pairs
    add = (2048 * F.add_word["gates2"], 2048 * F.add_word["ands"])

    def norm1(a: list, d: list) -> Block:
        return Block("norm1 + quant (r += a + d; xq1, r)", 1, 0, *plus(norm, quant2, add), XQ + X, [X, *a, *d],
                     note="adds the previous layer's a and d to the residual it committed")

    def norm2(a: list) -> Block:
        return Block("norm2 + quant (xq2)", 1, 0, *plus(norm, quant2), XQ, [X, *a], note="r2 = r1 + a, not committed")

    qk_tiles = [gemm_blocks(F, "q/k tile + RoPE", 2048, 2560, t2, [XQ], 2 * t2, epilogue=rope_ep,
                            note="a tile of whole rotary pairs; commits q' and k'"),
                gemm_blocks(F, "v tile", 2048, 512, t2, [XQ], 2 * t2)]
    q_e, kv_e = (by_head, [2 * KV]) if pack else (by_tile, by_tile * 2)
    pk = packs(by_tile, by_tile * 2) if pack else []
    out["e1"] = [*gd, norm1([A2], D8g), *qk_tiles, *pk,
        *heads("attention head", q_e, kv_e),
        gemm_blocks(F, "o tile + quant", 2048, 2048, t2, [O_H], 2 * t2, prologue=quant2),
        norm2([A2]),
        gemm_blocks(F, "gate_up tile + SiluMul", 2048, 16384, t2, [XQ], t2, epilogue=silu_ep,
                    note="g and u columns paired; commits s, half of gu"),
        *gs,
        Block("Int8RowQuant{8192} (sq)", 1, 0, *quant8, SQ, Sg),
        gemm_blocks(F, "down tile", 8192, 2048, t8, [SQ], 2 * t8),
    ]

    # (e2) e1 with max-first recompute: o and s never committed, only their blocks' partial maxima, then int8 codes
    code8 = (F.code[8192]["gates2"] * t2 // 2, F.code[8192]["ands"] * t2 // 2)
    code2_head = (F.code[2048]["gates2"] * HEAD_DIM, F.code[2048]["ands"] * HEAD_DIM)
    mx2, mx8 = (F.scale[2048]["gates2"], F.scale[2048]["ands"]), (F.scale[8192]["gates2"], F.scale[8192]["ands"])
    gmx, MXg = gather("s partial maxima", 16384 // t2, 4)
    gsq, SQg = gather("sq codes", 16384 // t2, t2 // 2)
    out["e2"] = [*gd, norm1([A2], D8g), *qk_tiles, *pk,
        *heads("attention pass 1 (partial max)", q_e, kv_e, commits=4),
        Block("o scale (max of 32 partial maxima)", 1, 0, *mx2, 4, [(HEADS // group, 4 * group)],
              note="ANDs: Int8RowQuant{2048}'s scale unit, an upper bound"),
        *heads("attention pass 2 (recompute, oq codes)", q_e, kv_e, extra=(0, *code2_head), commits=HEAD_DIM,
               extra_reads=(4,), last_only=True),
        gemm_blocks(F, "o tile", 2048, 2048, t2, [(HEADS // group, HEAD_DIM * group), 4], 2 * t2),
        norm2([A2]),
        gemm_blocks(F, "gate_up pass 1: tile + SiluMul, partial max", 2048, 16384, t2, [XQ], 4, epilogue=silu_ep),
        *gmx,
        Block("s scale (max of 512 partial maxima)", 1, 0, *mx8, 4, MXg,
              note="ANDs: Int8RowQuant{8192}'s scale unit, an upper bound"),
        gemm_blocks(F, "gate_up pass 2: recompute + sq codes", 2048, 16384, t2, [XQ, 4], t2 // 2,
                    epilogue=plus(silu_ep, code8)),
        *gsq,
        gemm_blocks(F, "down tile", 8192, 2048, t8, [*SQg, 4], 2 * t8),
    ]

    # (e3) e1 at the finest GEMM units, each reading its weight column and sharing its slice's rows; the 2-byte rows they
    # write gathered for their consumers
    pairs, coords = [4] * (HEAD_DIM // 2), [2] * HEAD_DIM
    p3 = pack is None or pack
    ga, a3 = gather("o", 2048, 2)
    gd, d3 = gather("down", 2048, 2)
    gs, s3 = gather("s", 8192, 2)
    out["e3"] = [norm1(a3, d3),
        gemm_blocks(F, "q/k rotary pair + RoPE", 2048, 2560, 2, [XQ], 4, epilogue=(2 * F.rope_word["gates2"], 2 * F.rope_word["ands"])),
        gemm_blocks(F, "v coordinate", 2048, 512, 1, [XQ], 2),
        *(packs(pairs, pairs + coords) if p3 else []),
        *heads("attention head", *((by_head, [2 * KV]) if p3 else (pairs, pairs + coords))),
        Block("Int8RowQuant{2048} (oq)", 1, 0, *quant2, XQ, [O_H]),
        gemm_blocks(F, "o coordinate", 2048, 2048, 1, [XQ], 2),
        *ga,
        norm2(a3),
        gemm_blocks(F, "gate_up (g, u) pair + SiluMul", 2048, 16384, 2, [XQ], 2, epilogue=(F.silu_word["gates2"], F.silu_word["ands"])),
        *gs,
        Block("Int8RowQuant{8192} (sq)", 1, 0, *quant8, SQ, s3),
        gemm_blocks(F, "down coordinate", 8192, 2048, 1, [SQ], 2),
        *gd,
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
        gb, best_rows = gather("head bests", slices, 6, per="sampled")
        return [Block("LM head slice {2048,16} + best of 16", slices, n * co["macs"], n * co["gates2"],
                      n * co["ands"] + per_group, 2 + 4, [F.bf16_row], n * 2 * 2048, per="sampled",
                      note="epilogue ANDs: TokenSelectGroups' a group of 16 (measured / 167); the index's 4 bytes an estimate"),
                *gb,
                Block("token select over the slices' bests", 1, 0, 0, comb_ands, 4, best_rows, per="sampled",
                      note=f"ANDs: TokenSelectCombine_v2's ({csrc}), over 167 x 48 bests")]
    gl, logit_rows = gather("logit slices", 2672 // n, 2 * n, per="sampled")
    return [Block("LM head slice {2048,16}", slices, n * co["macs"], n * co["gates2"], n * co["ands"], 2 * n, [F.bf16_row],
                  n * 2 * 2048, per="sampled"),
            *[Block(b.name + " (a group's)", 48 * b.n, 0, 0, 0, b.commits, b.reads, per="sampled", drawn_by_openings=True)
              for b in gl],
            Block("TokenSelectGroups_v1{C=2672,G=16}", 48, 0, 0, groups_ands, 167 * 6, logit_rows, per="sampled",
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

    def cost(rs: list[dict]) -> tuple[float, float]:
        """Expected ANDs (openings included) and bits a draw: a unit drawn by its `drawn_ands`, costing its block, and each
        row's share of the bits (``bits_share``)."""
        drawn = sum(r["n_per_token"] * r["drawn_ands"] for r in rs)
        a = sum(r["n_per_token"] * r["drawn_ands"] * (r["ands"] + r["opening_ands_est"]) for r in rs) / drawn
        b = sum(r["n_per_token"] * r["drawn_ands"] * 2 ** r["k_log_est"] for r in rs) / drawn
        for r in rs:
            r["bits_share"] = r["n_per_token"] * r["drawn_ands"] * 2 ** r["k_log_est"] / drawn / b
        return a, b

    cost_ah, cost_bh = cost([dict(r) for r in rows + head])
    cost_ands, cost_bits = cost(rows)
    return {"partition": name, "keys": F.T, "beta_bytes_per_token": beta, "beta_bytes_per_kmac": 1000 * beta / F.served_macs,
            "served_macs_per_token": F.served_macs, "proved_macs_per_token": macs, "blocks_per_token": n,
            "ands_per_token": ands, "blocks_over_cap_per_token": sum(r["n_per_token"] for r in rows if r["over_cap"]),
            "and_weighted_mean_block_ands": w / ands, "with_openings_and_weighted_mean": wo / ao,
            "expected_proved_ands_per_window_est": DRAWS * w / ands,
            "expected_proved_ands_with_openings_per_window_est": DRAWS * cost_ands,
            "expected_proved_bits_per_window_est": DRAWS * cost_bits,
            "with_head_expected_proved_ands_with_openings_per_window_est": DRAWS * cost_ah,
            "with_head_expected_proved_bits_per_window_est": DRAWS * cost_bh,
            "head_beta_bytes_per_sampled_token": head_beta,
            "head_ands_per_sampled_token": sum(r["n_per_token"] * r["ands"] for r in head),
            "head_blocks_over_cap_per_sampled_token": sum(r["n_per_token"] for r in head if r["over_cap"]),
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
    ap.add_argument("--keys", default="18,35,144,288,512")
    ap.add_argument("--key-blocks", default="0,32,64,128", help="attention's keys a unit to sweep (0: the fewest that fit)")
    ap.add_argument("--groups", default="1,4", help="query heads of a KV head a unit to sweep")
    ap.add_argument("--ir", default=None, help="reuse the IR record of an earlier --out")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    keys = [int(k) for k in a.keys.split(",")]
    m = json.loads(Path(a.ir).read_text())["ir"] if a.ir else measure(keys)
    res = {"ir": m, "lm_head": lm_head(m), "at": {}, "attention_unit": {}}
    for T in keys:
        F = Facts(m, T)
        res["at"][T] = [qword_partition(m, T)] + [summarize(F, k, v) for k, v in partitions(F).items()]
    slim = ("partition", "beta_bytes_per_token", "beta_with_head_bytes_per_token", "blocks_per_token",
            "blocks_over_cap_per_token", "expected_proved_ands_with_openings_per_window_est",
            "expected_proved_bits_per_window_est", "with_head_expected_proved_bits_per_window_est")
    print("\nattention's unit: keys a unit (0: the fewest that fit) x query heads a unit x packs; proved bits a window, β")
    for T in keys:
        F, res["attention_unit"][T] = Facts(m, T), []
        for kb in (int(k) for k in a.key_blocks.split(",")):
            for g in (int(x) for x in a.groups.split(",")):
                for pack in (False, True):
                    for name, blocks in partitions(F, kb, g, pack).items():
                        if name in ("a0", "w") or (not pack and name not in ("e1", "e2", "e3")):
                            continue
                        s = summarize(F, name, blocks)
                        att = [r for r in s["blocks"] if r["block"].startswith("attention")]
                        packed = (pack or None) if name in ("e1", "e2", "e3") else None
                        res["attention_unit"][T].append({
                            "key_block": kb, "group": g, "pack": packed, **{k: s[k] for k in slim},
                            "attention_bits_share": sum(r["bits_share"] for r in att), "attention_blocks": att})
                        print(f"  T={T:>3} kb={kb:>3} g={g} pack={'-' if packed is None and name not in ('e1', 'e2', 'e3') else int(pack)}"
                              f" {name:>2}: bits/window 2^"
                              f"{math.log2(s['expected_proved_bits_per_window_est']):.2f} (attention "
                              f"{sum(r['bits_share'] for r in att):.0%}, k {sorted({r['k_log_est'] for r in att})}) "
                              f"β {s['beta_bytes_per_token']:>10,.0f} over cap {s['blocks_over_cap_per_token']:,.0f}")
    for T in keys:
        print(f"\nkeys T={T}")
        for p in res["at"][T]:
            print(f"  {p['partition']:>3}: beta {p['beta_bytes_per_token']:>12,.0f} B/token {p['beta_bytes_per_kmac']:7.2f} B/kMAC"
                  f"  head {p.get('head_beta_bytes_per_sampled_token', 0):>9,.0f} B/sampled token"
                  f"  blocks/token {p['blocks_per_token']:>10,.0f}  AND-weighted mean block {p['and_weighted_mean_block_ands']:>13,.0f}"
                  + (f"  bits/window 2^{math.log2(p['expected_proved_bits_per_window_est']):.2f} (with head 2^"
                     f"{math.log2(p['with_head_expected_proved_bits_per_window_est']):.2f}) over cap "
                     f"{p['blocks_over_cap_per_token']:,.0f}" if "expected_proved_bits_per_window_est" in p else ""))
    for T in keys:
        for p in res["at"][T]:
            print(f"\n[{p['partition']}] T={T}: beta {p['beta_bytes_per_token']:,.0f} B/token; blocks/token "
                  f"{p['blocks_per_token']:,.0f}; ANDs/token {p['ands_per_token']:,.0f}; AND-weighted mean "
                  f"{p['and_weighted_mean_block_ands']:,.0f}")
            for r in p["blocks"][:24]:
                print("   ", {k: r.get(k) for k in ("block", "op", "kind", "n_per_token", "macs", "gates2", "ands",
                                                    "opening_ands_est", "rows", "boundary_bits", "size_class", "k_log_est",
                                                    "over_cap", "bits_share")
                              if k in r})
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
