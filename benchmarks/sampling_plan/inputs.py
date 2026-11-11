"""The int8 rows' inputs to the sampling planner (Llama-3.1-8B and Qwen3-8B): each a value, a low and a high, its source and
its status (M measured, A assumed, D derived from the others), in one currency, GPU-seconds of an RTX PRO 6000 (sm_120);
the models' shapes; what every scale commits at serve time, a token a layer, by PoUW scheme (the scales
table); and the cost model's observations by card (`CARDS`: the RTX PRO 6000's, and the `h200` card's in H200
GPU-seconds)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, log2

from verity.commitments.windows.interface import K_MAX, K_MIN, LAST_BLOCK_BYTES, MASK_LOG, PER_SLOT, SLOT_LOG
from verity.protocols.spot.costfit import Observation
from verity.protocols.spot.optimize import SHA512_ANDS


@dataclass(frozen=True)
class Input:
    value: float
    low: float
    high: float
    source: str
    status: str


I = {
    "cpu": Input(1 / 55, 1 / 110, 1 / 27, "vCPU $0.033/h against sm_120 $1.80/h (X = 55)", "M"),
    "cpu_h200": Input(0.033 / 5.29, 0.033 / 10.58, 0.033 / 2.65, "vCPU $0.033/h against one H200 SXM's $5.29/h on "
                      "RunPod, the h200 card's pod (X = 160)", "M"),
    "round": Input(12.6, 12.6, 12.6, "a round (the budget model)", "A"),
    "rounds": Input(48_000, 48_000, 48_000, "a week of rounds (the budget model's S)", "A"),
    "tokens_per_s": Input(20_000, 8_000, 40_000, "int8 Llama-3.1-8B on one RTX PRO 6000: 7.2G MACs a token at 2K context "
                          "against 5e14 MAC/s dense int8 at 30-40% (prefill) down to decode's weight-bandwidth bound", "A"),
    "context": Input(2048, 2048, 2048, "prefill at 2K, positions uniform (the budget model's binding cell): a request is "
                     "one prompt of it, re-run whole from its committed tokens and the weights to rebuild a drawn proof "
                     "block's witness, at `tokens_per_s`", "A"),
    "commit_per_byte": Input(4.34e-12, 2.2e-12, 1.3e-11, "SHA-512 rows on H200: 3.62 ps a byte plus 2.94 ns a 4 KiB leaf; "
                             "sm_120 taken at 0.5-3x", "M"),
    "commit_floor": Input(0.060, 0.045, 0.101, "the one serve-time term of the 'floor' scheme: the measured "
                          "hashing of the int8 linears' s8 A operands in R = 128 x 32 leaves beside the GEMM on "
                          "sm_120 (0.101 at R = 64), the leaves the Commit holds and PoUW's "
                          "seeds read under compile: Llama-3.1-8B's 26,624 B a token a layer, which the "
                          "serve-time total prices with every other scale's bytes", "M"),
    "remote_receipts": Input(268e6, 268e6, 268e6, "PoUB's receipts a GPU a pass: 64 B a 4 KiB block over the "
                             "registered store, 128 GiB on the 8-GPU Llama-3.1-8B node", "D"),
    "passes_per_s": Input(65, 1, 82, "PoUB passes a GPU-second: about 60-70 at 12.1 ms a pass; 82 back to back; "
                          "whether every served step is a pass is open", "A"),
    "decode_tokens": Input(32, 32, 32, "tokens a decode step (B32, the scales table's decode)", "A"),
    "compress_latency": Input(4.3e-6, 2.0e-6, 6.0e-6, "seconds a dependent SHA-512 compression on decode's critical path: "
                              "Pearl-C's s512_rows chain, 28 deep in 121 us", "M"),
    "noise_bits": Input(1, 1, 2, "noise bits an entry", "A"),
    "zero_work_templates": Input(30, 15, 60, "the layer's templates that carry no work (quantizers, softmax, norms, "
                                 "the epilogues' pieces), each held at its floor", "A"),
}

#: ANDs of the pieces (their pins and hm96 PROTOCOL §5)
S8_STEP = 4_190
EPILOGUE = 5_000
SHA512 = SHA512_ANDS
HM96 = 116_240



@dataclass(frozen=True)
class Model:
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    hidden: int
    intermediate: int

    @property
    def linears(self) -> dict[int, int]:
        """Output coordinates a token by K: qkv, o_proj and gate_up read the hidden state, down the activation."""
        qkv = (self.heads + 2 * self.kv_heads) * self.head_dim
        return {self.hidden: qkv + self.heads * self.head_dim + 2 * self.intermediate, self.intermediate: self.hidden}

    @property
    def out_cells(self) -> int:
        return sum(self.linears.values())


#: 32 query heads and 8 KV heads of 128, hidden 4,096; Llama-3.1-8B 32 layers, intermediate 14,336; Qwen3-8B 36 and 12,288
MODELS = {"llama-3.1-8b": Model(32, 32, 8, 128, 4096, 14336), "qwen3-8b": Model(36, 32, 8, 128, 4096, 12288)}


@dataclass(frozen=True)
class Row:
    """What one scale commits of one tensor at serve time: `nbytes` a token a layer at prefill (`decode`, a token a layer
    at decode, when it differs), in leaves of `leaf` bytes at decode (`fmt` row-seg/v1 or row/v2), one tree a call when a
    seed reads its root, which gates the step today when `gates`; a `tile` leaf is hashed as its call ends, its tree being
    the pass's transcript."""

    scale: str
    tensor: str
    nbytes: float
    leaf: int = 4096
    fmt: str = "row-seg"
    gates: bool = False
    tile: bool = False
    decode: float | None = None


#: one tensor's name under every scale that commits it, so a repeat is an overlap
NORM1, ATTN, NORM2, ACT = ("norm1 (qkv's input)", "attention output (o_proj's input)", "norm2 (gate_up's input)",
                           "activation (down's input)")
KV = "K and V"
SCHEMES = ("floor", "h1", "h1-bf16", "pearl-c")


def serve_rows(m: Model, scheme: str, *, kv: bool = False) -> list[Row]:
    """Every serve-time commitment a token a layer under PoUW `scheme` (the scales table's section 1 and 4):

    - the Commit's taps, today's per-Call fused units (hash's layout, 71,680 B on Qwen3-8B: qkv's output, rope_q, kv_k,
      the attention output, both residuals, the activation), with V inside qkv's output and K as kv_k;
    - `pearl-c` (served today): PoUW's FP32 A rows (row-seg/v1, a row a token), its tile leaves (8 B an output cell,
      64 x 64 tiles at prefill, 32 x 64 at decode) and the pass transcript's tile tree; nothing binds the A rows to the
      Commit's copies;
    - `h1` (compile, W8A8, H1 + fx1): A from the Commit's roots, its rows s8, the norm outputs pinned into
      the Commit (`operand-inside`), rope_q pinned as QK^T's A; the CP-W receipts (2 (K/32 + m) u32 a 32 x 32 tile; m 2 at
      K = 4,096 and 3 at 12,288, taken as 3 at 14,336); `h1-bf16` the same with the A rows BF16;
    - `kv`: the remote KV append's payload, one 4 KiB block a token-layer, when live KV is credited (not for chat).

    PoUB's receipts are a GPU's, not a token's (`I['remote_receipts']`), and `floor` is the measured term
    alone (`I['commit_floor']`)."""
    if scheme not in SCHEMES:
        raise ValueError(f"no PoUW scheme {scheme!r}: one of {', '.join(SCHEMES)}")
    if scheme == "floor":
        return []
    q, kvd, h, f = m.heads * m.head_dim, m.kv_heads * m.head_dim, m.hidden, m.intermediate
    compile_ = scheme.startswith("h1")
    a = 1 if scheme == "h1" else 2
    rows = [Row("commit", "q and k before RoPE", 2 * (q + kvd)), Row("commit", KV, 2 * 2 * kvd),
            Row("commit", "q after RoPE", 2 * q, gates=compile_),
            Row("commit", ATTN, a * q if compile_ else 2 * q, gates=compile_),
            Row("commit", "post-attention residual", 2 * h), Row("commit", ACT, a * f if compile_ else 2 * f,
                                                                  gates=compile_),
            Row("commit", "post-MLP residual", 2 * h)]
    if compile_:
        rows += [Row("pin:h1", NORM1, a * h, gates=True), Row("pin:h1", NORM2, a * h, gates=True)]
        receipts = sum(cells * 2 * (k // 32 + (2 if k <= 4096 else 3)) * 4 / 1024 for k, cells in m.linears.items())
        rows.append(Row("commit", "CP-W receipts", receipts))
    else:
        rows += [Row("pearl-c:A", t, 4 * k, leaf=4 * k, gates=True)
                 for t, k in ((NORM1, h), (ATTN, q), (NORM2, h), (ACT, f))]
        tiles = m.out_cells
        rows.append(Row("pearl-c:tiles", "C~ and U (tile leaves)", 8 * tiles, leaf=32 * 64 * 8, gates=True, tile=True))
        rows.append(Row("pearl-c:transcript", "tile tree", 3 * 128 * tiles / 4096, decode=3 * 128 * tiles / 2048))
    if kv:
        rows.append(Row("kv-append", KV, 4096, fmt="row"))
    return rows


def leaf_chain(nbytes: int, fmt: str = "row-seg") -> int:
    """Dependent SHA-512 compressions to hash one hiding leaf of `nbytes` (frame_v3 PROTOCOL): row-seg/v1 hashes 1 KiB
    segments side by side, 9 deep each past a constant prefix, then its outer digest over their 64 B digests; row/v2 is one
    chain over the row; then hm96's hiding leaf and the frame's row leaf, 2 each."""
    if fmt == "row":
        inner = ceil((nbytes + 17) / 128) + 1
    else:
        segments = max(1, ceil(nbytes / 1024))
        inner = ceil((min(nbytes, 1024) + 17) / 128) + ceil((64 * segments + 17) / 128)
    return inner + 4


def tree_chain(leaves: float) -> int:
    """A frame-v3 tree's dependent compressions over `leaves` leaves: 3 a node level."""
    return 3 * max(0, ceil(log2(max(leaves, 1))))



def block_k_log(ands: int, rows: Sequence[int]) -> int:
    """The ``k_log`` of the Flock block that holds one proof block of ``ands`` ANDs opening ``rows`` (bytes each), as
    Flock's composer lays it out (`verity_flock.circuit.compose`'s ranges): the unit's slot, the least power of two above
    its ANDs and its rows' bits (a row an AND and a column a port bit, as `UnitStager`'s early check counts them), beside an
    hm96 slot a row (a power of two in all) and the mask, then a sha512x3 slot per `PER_SLOT` compressions before the rows'
    last SHA-512 blocks. A row's opening is those slots, not ANDs in the unit's slot."""
    hm = (1 << (len(rows) - 1).bit_length()) << SLOT_LOG
    slot = 1 << max(7, (ands + 8 * sum(rows)).bit_length())
    comps = sum(-(-(b - LAST_BLOCK_BYTES) // 128) for b in rows if b > LAST_BLOCK_BYTES)
    bits = (-(-(hm + slot + (1 << MASK_LOG)) >> SLOT_LOG) + -(-comps // PER_SLOT)) << SLOT_LOG
    return max(K_MIN, (bits - 1).bit_length())


def cap(ands: int, rows: Sequence[int]) -> int:
    """A statement's proof blocks of ``ands`` ANDs opening ``rows``: one a Flock block of 2^`block_k_log` under
    m = k_log + nbl <= 35, at most 2^20 (below 2^22 the composer packs several a block, about as many a statement); 0, not a
    proof block, past Flock's 2^`K_MAX`."""
    k = block_k_log(ands, rows)
    return 0 if k > K_MAX else min(1 << (35 - k), 1 << 20)


def s8_coord_ands(k: int) -> int:
    """One s8 coordinate's ANDs with its rows' openings, as `costfit` counts them: K/32 steps, the epilogue, two K-byte
    operand rows and three one-block rows (the scales and the output), each at `optimize.row_ands`."""
    from verity.protocols.spot.optimize import row_ands

    return k // 32 * S8_STEP + EPILOGUE + 2 * row_ands(k) + 3 * row_ands(1)


#: the measurements the cost model is fitted from (`costfit`): benchmarks/one_stage/zk_cost.py's sessions of
#: `ScaledMmS8Coordinate_v1{K}` by `flock-circuit prove --zk --gpu` with the ZK server's send check, per (K, staged units,
#: sampled proof blocks): the mean wall of its two cold prover processes (GPU-s: one touched statement's prover today, its statement
#: build included), the mean CPU time of Lean `verify --zk` on its three sessions and its statement's staging (CPU-s)
ZK_RUN = "r20261018-060433-b55e"
ZK: dict[tuple[int, int, int], tuple[float, float, float]] = {
    (4096, 1, 1): (22.34, 57.63, 4.7),
    (4096, 8, 1): (21.57, 54.98, 4.5),
    (4096, 8, 8): (19.33, 50.17, 4.5),
    (4096, 64, 1): (18.55, 49.91, 7.2),
    (4096, 64, 8): (25.67, 47.32, 7.2),
    (4096, 64, 64): (27.75, 42.71, 7.2),
    (4096, 512, 1): (21.68, 41.03, 10.2),
    (4096, 512, 8): (21.22, 43.25, 10.2),
    (4096, 512, 64): (30.57, 46.27, 10.2),
    (4096, 512, 512): (62.17, 45.50, 10.2),
    (14336, 1, 1): (50.50, 109.09, 9.6),
    (14336, 8, 1): (54.69, 113.45, 9.9),
    (14336, 8, 8): (54.68, 145.56, 9.9),
    (14336, 64, 1): (53.68, 137.31, 16.1),
    (14336, 64, 8): (54.06, 124.98, 16.1),
    (14336, 64, 64): (65.44, 121.23, 16.1),
    (14336, 512, 1): (54.45, 114.67, 41.6),
    (14336, 512, 8): (54.39, 109.83, 41.6),
    (14336, 512, 64): (66.22, 107.66, 41.6),
    (14336, 512, 512): (131.07, 119.78, 41.6)}


#: PoUB's opening unit (`verity.protocols.poub.verifier.commitments.gates.block_check`): a sampled receipt opened in
#: circuit, its 4 KiB block decoded to its payload, its sketch receipt recomputed, and the receipt leaf and the payload
#: opened to their roots (its ANDs are the whole unit's, every opening included)
OPENING = "BlockOpeningCheck_v1{N=512,C=569,K=2,MASKED=0,BYTES=4096,ROWS=16,LEAF=16,P=64,DR=2,DP=12,DV=1}"
OPENING_ANDS = 29_617_630

#: benchmarks/one_stage/a8.py's sessions of it by `flock-circuit prove --zk --gpu` with the ZK server's send check, per run
#: and (staged units, sampled proof blocks): the GPU-seconds its cold prover process held the lease (its wall less the lease's
#: wait; the mean of two where it ran two). Their device phase is 79 ms an opening at a full statement, 16 ms without the
#: send check, the GPU's own algebra about 1 ms of it. They
#: give no `verify`: a8.py times Lean's by wall (478-488 s, flock-verify lean-4e521bba8e1bc519), not CPU. Nor `stage`:
#: A8's staging is about 130 s at 64 staged units and at 512, a fixed cost a statement of this template pays that the stage
#: fit's terms, shared with the s8 coordinates, don't carry.
OPENING_SESSIONS: dict[str, dict[tuple[int, int], float]] = {
    "r20261018-195718-d00b": {(64, 1): 139.5, (64, 8): 138.0, (64, 32): 161.0, (64, 64): 163.5},
    "r20261018-211523-93ec": {(512, 64): 156.0, (512, 128): 161.0, (512, 256): 197.0, (512, 512): 234.0}}


#: the `h200` card: GLM-5.3's proof blocks at TP8 (benchmarks/served_units/stage.py's types) proved on one H200 by
#: benchmarks/one_stage/zk_cost.py with `flock-circuit prove --zk --gpu`, the device building the witness and no send check
#: (arm dev, the deployed prover), per run and (Definition, ANDs a proof block as staged, staged proof blocks, sampled): the
#: median end-to-end seconds of a warm prover's timed sessions (GPU-s, the live verifier's round trips in them), the CPU
#: time of Lean `verify --zk` where it verified that point (None elsewhere) and the statement's staging (CPU-s); its
#: `overhead` step's observations.json
ZK_H200: dict[str, dict[tuple[str, int, int, int], tuple[float, float | None, float]]] = {
    "r20261106-074634-13c4": {
        ("MoeExpertFp8BlockCoordinateW_v1{K=256,G=128,KB=128,SWAP=True}", 73969, 4096, 4096): (2.409, None, 28.6),
        ("MoeExpertFp8BlockCoordinateW_v1{K=256,G=128,KB=128,SWAP=True}", 73969, 8192, 8): (1.144, None, 32.0),
        ("MoeExpertFp8BlockCoordinateW_v1{K=256,G=128,KB=128,SWAP=True}", 73969, 8192, 64): (1.311, None, 32.0),
        ("MoeExpertFp8BlockCoordinateW_v1{K=256,G=128,KB=128,SWAP=True}", 73969, 8192, 8192): (3.956, 12.82, 32.0),
        ("MoeExpertFp8BlockCoordinate_v1{K=6144,G=128,KB=128,SWAP=True}", 1737375, 512, 512): (7.542, None, 93.2),
        ("MoeExpertFp8BlockCoordinate_v1{K=6144,G=128,KB=128,SWAP=True}", 1737375, 1024, 8): (6.341, None, 103.2),
        ("MoeExpertFp8BlockCoordinate_v1{K=6144,G=128,KB=128,SWAP=True}", 1737375, 1024, 64): (6.528, None, 103.2),
        ("MoeExpertFp8BlockCoordinate_v1{K=6144,G=128,KB=128,SWAP=True}", 1737375, 1024, 1024): (8.794, 43.32, 103.2),
        ("ScaledMmFp8BlockCoordinate_v2{K=2048,G=128}", 539177, 2048, 2048): (3.489, None, 58.4),
        ("ScaledMmFp8BlockCoordinate_v2{K=2048,G=128}", 539177, 4096, 8): (2.312, None, 79.7),
        ("ScaledMmFp8BlockCoordinate_v2{K=2048,G=128}", 539177, 4096, 64): (2.195, None, 79.7),
        ("ScaledMmFp8BlockCoordinate_v2{K=2048,G=128}", 539177, 4096, 4096): (4.542, 17.97, 79.7),
        ("ScaledMmFp8BlockCoordinate_v2{K=256,G=128}", 66607, 8192, 8192): (2.407, None, 33.8),
        ("ScaledMmFp8BlockCoordinate_v2{K=256,G=128}", 66607, 16384, 8): (1.212, None, 45.9),
        ("ScaledMmFp8BlockCoordinate_v2{K=256,G=128}", 66607, 16384, 64): (1.266, None, 45.9),
        ("ScaledMmFp8BlockCoordinate_v2{K=256,G=128}", 66607, 16384, 16384): (4.086, 12.14, 45.9),
        ("ScaledMmFp8BlockCoordinate_v2{K=6144,G=128}", 1619337, 1024, 8): (4.281, None, 108.8),
        ("ScaledMmFp8BlockCoordinate_v2{K=6144,G=128}", 1619337, 1024, 1024): (5.497, None, 108.8),
        ("ScaledMmFp8BlockCoordinate_v2{K=6144,G=128}", 1619337, 2048, 8): (4.732, None, 137.8),
        ("ScaledMmFp8BlockCoordinate_v2{K=6144,G=128}", 1619337, 2048, 64): (4.724, None, 137.8),
        ("ScaledMmFp8BlockCoordinate_v2{K=6144,G=128}", 1619337, 2048, 2048): (6.764, 42.37, 137.8)},
    "r20261106-092817-df0b": {
        ("MoeExpertFp8CoordinateChunk_v1{E=256,K=256,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1,FIRST=False,LAST=False}", 619034, 128, 128): (19.884, None, 62.4),
        ("MoeExpertFp8CoordinateChunk_v1{E=256,K=256,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1,FIRST=False,LAST=False}", 619034, 256, 8): (16.883, None, 68.8),
        ("MoeExpertFp8CoordinateChunk_v1{E=256,K=256,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1,FIRST=False,LAST=False}", 619034, 256, 64): (17.583, None, 68.8),
        ("MoeExpertFp8CoordinateChunk_v1{E=256,K=256,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1,FIRST=False,LAST=False}", 619034, 256, 256): (18.481, 21.54, 68.8)},
    "r20261106-074918-e6ff": {
        ("DsaIndexerKRow_v1{D=128,ROPE=64,EPS=1e-06}", 126769, 8192, 8192): (5.776, None, 107.5),
        ("DsaIndexerKRow_v1{D=128,ROPE=64,EPS=1e-06}", 126769, 16384, 8): (3.813, None, 149.1),
        ("DsaIndexerKRow_v1{D=128,ROPE=64,EPS=1e-06}", 126769, 16384, 64): (4.429, None, 149.1),
        ("DsaIndexerKRow_v1{D=128,ROPE=64,EPS=1e-06}", 126769, 16384, 16384): (8.054, 60.62, 149.1),
        ("DsaIndexerQHead_v1{D=128,ROPE=64,NH=32}", 122673, 8192, 8192): (5.976, None, 84.6),
        ("DsaIndexerQHead_v1{D=128,ROPE=64,NH=32}", 122673, 16384, 8): (4.582, None, 104.8),
        ("DsaIndexerQHead_v1{D=128,ROPE=64,NH=32}", 122673, 16384, 64): (4.938, None, 104.8),
        ("DsaIndexerQHead_v1{D=128,ROPE=64,NH=32}", 122673, 16384, 16384): (7.996, 59.87, 104.8),
        ("DsaMqaLogit_v1{H=32,D=128}", 1087058, 1024, 1024): (4.577, None, 71.8),
        ("DsaMqaLogit_v1{H=32,D=128}", 1087058, 2048, 8): (3.907, None, 75.6),
        ("DsaMqaLogit_v1{H=32,D=128}", 1087058, 2048, 64): (3.702, None, 75.6),
        ("DsaMqaLogit_v1{H=32,D=128}", 1087058, 2048, 2048): (6.252, 28.04, 75.6),
        ("DsaTopKShortRow_v1{TOPK=2048}", 28, 16384, 16384): (2.473, None, 91.3),
        ("DsaTopKShortRow_v1{TOPK=2048}", 28, 32768, 8): (0.984, None, 166.9),
        ("DsaTopKShortRow_v1{TOPK=2048}", 28, 32768, 64): (1.072, None, 166.9),
        ("DsaTopKShortRow_v1{TOPK=2048}", 28, 32768, 32768): (4.197, 12.92, 166.9),
        ("MoeRouterSigmoidBias_v1{E=256,TOPK=8,SCALE=2.5,KERNEL='block'}", 164320, 4096, 4096): (2.763, None, 77.7),
        ("MoeRouterSigmoidBias_v1{E=256,TOPK=8,SCALE=2.5,KERNEL='block'}", 164320, 8192, 8): (1.496, None, 120.8),
        ("MoeRouterSigmoidBias_v1{E=256,TOPK=8,SCALE=2.5,KERNEL='block'}", 164320, 8192, 64): (1.550, None, 120.8),
        ("MoeRouterSigmoidBias_v1{E=256,TOPK=8,SCALE=2.5,KERNEL='block'}", 164320, 8192, 8192): (4.138, 17.82, 120.8),
        ("MoeSumCoordinate_v1{TOPK=8}", 6494, 16384, 16384): (2.641, None, 23.6),
        ("MoeSumCoordinate_v1{TOPK=8}", 6494, 32768, 8): (1.234, None, 25.5),
        ("MoeSumCoordinate_v1{TOPK=8}", 6494, 32768, 64): (1.035, None, 25.5),
        ("MoeSumCoordinate_v1{TOPK=8}", 6494, 32768, 32768): (4.033, 13.21, 25.5),
        ("RMSNormFusedCuda_v2{N=6144,EPS=1e-05}", 12778458, 512, 512): (26.543, None, 559.3),
        ("RMSNormFusedCuda_v2{N=6144,EPS=1e-05}", 12778458, 1024, 8): (27.654, None, 633.5),
        ("RMSNormFusedCuda_v2{N=6144,EPS=1e-05}", 12778458, 1024, 1024): (30.936, 652.35, 633.5),
        ("RMSNormTriton_v1{N=2048,EPS=1e-06}", 4368140, 1024, 1024): (21.757, None, 301.7),
        ("RMSNormTriton_v1{N=2048,EPS=1e-06}", 4368140, 2048, 8): (16.018, None, 347.5),
        ("RMSNormTriton_v1{N=2048,EPS=1e-06}", 4368140, 2048, 2048): (17.793, 461.70, 347.5),
        ("RopeOut_v1", 2998, 4096, 4096): (2.261, None, 22.3),
        ("RopeOut_v1", 2998, 8192, 8): (1.087, None, 23.8),
        ("RopeOut_v1", 2998, 8192, 64): (1.197, None, 23.8),
        ("RopeOut_v1", 2998, 8192, 8192): (3.985, 10.06, 23.8),
        ("SiluMulBf16_v2", 1416, 8192, 8192): (2.432, None, 23.0),
        ("SiluMulBf16_v2", 1416, 16384, 8): (1.111, None, 23.0),
        ("SiluMulBf16_v2", 1416, 16384, 64): (1.001, None, 23.0),
        ("SiluMulBf16_v2", 1416, 16384, 16384): (3.664, 10.98, 23.0),
        ("SiluMulFp8BlockQuantGroup_v1{G=128}", 48230, 8192, 8192): (2.247, None, 126.4),
        ("SiluMulFp8BlockQuantGroup_v1{G=128}", 48230, 16384, 8): (1.260, None, 237.2),
        ("SiluMulFp8BlockQuantGroup_v1{G=128}", 48230, 16384, 64): (0.996, None, 237.2),
        ("SiluMulFp8BlockQuantGroup_v1{G=128}", 48230, 16384, 16384): (3.558, 10.77, 237.2)},
    "r20261106-092821-266a": {
        ("MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1}", 620575, 128, 128): (17.365, None, 64.0),
        ("MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1}", 620575, 256, 8): (19.276, None, 68.3),
        ("MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1}", 620575, 256, 64): (17.762, None, 68.3),
        ("MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1}", 620575, 256, 256): (20.799, 21.94, 68.3)}}


def observations(card: str = "rtx-pro-6000") -> list[Observation]:
    if card not in CARDS:
        raise ValueError(f"no card {card!r}: one of {', '.join(CARDS)}")
    if card == "h200":
        return [Observation(run, t, a, c, d, prove=p, verify=v, stage=s)
                for run, points in ZK_H200.items() for (t, a, c, d), (p, v, s) in points.items()]
    return [Observation(ZK_RUN, f"ScaledMmS8Coordinate_v1{{K={k}}}", s8_coord_ands(k), c, d, prove=p, verify=v, stage=s)
            for (k, c, d), (p, v, s) in ZK.items()] + \
        [Observation(run, OPENING, OPENING_ANDS, c, d, prove=p)
         for run, points in OPENING_SESSIONS.items() for (c, d), p in points.items()]


KEY = {"prover": "flock-circuit bin-dec484d021e461a0-g1-sm120 (binary 21944fa201b5, CUDA 13.3.1), --zk --gpu, "
                  "FC_HOST_WITNESS=1, the firewall's send check on; the opening unit's sessions "
                  "bin-b4e798e6b2590a24-g1-sm120, the same flags",
       "verifier": "flock-verify lean-3a9d4dc43048fdae (binary b3b257d5ac0e), verify --zk, one CPU",
       "hardware": "RTX PRO 6000 Blackwell Server Edition (driver 580.173.02), Xeon 6776P, vy-nebius-2; the opening "
                   "unit's, the same card on vy-nebius-1",
       "templates": f"ScaledMmS8Coordinate_v1{{K=4096|14336}}, {OPENING}"}

KEY_H200 = {"prover": "flock-circuit bin-32a698fff862d3db-g1-sm90 (binary 09fbd3d8c1d3, CUDA 13.3), --zk --gpu, "
                       "FC_HOST_WITNESS and FC_FIREWALL unset",
            "verifier": "flock-verify lean-3e804125132e928c (binary f399b8425bd6), verify --zk, one CPU",
            "hardware": "H200 SXM 141 GB (driver 595.91.07), Xeon Platinum 8568Y+, vy-spot-h1",
            "templates": "GLM-5.3's served proof-block types at TP8 (benchmarks/served_units/stage.py)"}

#: each card's observations, the builds and hardware they key, and its price ratio (`I`'s key): GPU-seconds of that card
CARDS = {"rtx-pro-6000": (KEY, "cpu"), "h200": (KEY_H200, "cpu_h200")}
