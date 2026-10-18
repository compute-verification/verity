"""The int8 rows' inputs to the sampling planner (Llama-3.1-8B and Qwen3-8B): each a value, a low and a high, its source and
its status (M measured, A assumed, D derived from the others), in one currency, GPU-seconds of an RTX PRO 6000 (sm_120);
the models' shapes; and what every scale commits at serve time, a token a layer, by PoUW scheme (the scales table,
note:proofs/20261018T1720Z-draft-scales-table v3)."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

from verity.core.protocols.sampled_proofs.costfit import Observation


@dataclass(frozen=True)
class Input:
    value: float
    low: float
    high: float
    source: str
    status: str


I = {
    "cpu": Input(1 / 55, 1 / 110, 1 / 27, "vCPU $0.033/h against sm_120 $1.80/h (note:compute-accounting/"
                 "20261018T0349Z-draft-budget, X = 55)", "M"),
    "round": Input(12.6, 12.6, 12.6, "a round (BUDGET)", "A"),
    "rounds": Input(48_000, 48_000, 48_000, "a week of rounds (BUDGET's S)", "A"),
    "tokens_per_s": Input(20_000, 8_000, 40_000, "int8 Llama-3.1-8B on one RTX PRO 6000: 7.2G MACs a token at 2K context "
                          "against 5e14 MAC/s dense int8 at 30-40% (prefill) down to decode's weight-bandwidth bound", "A"),
    "context": Input(2048, 2048, 2048, "prefill at 2K, positions uniform (BUDGET's binding cell)", "A"),
    "step_tokens": Input(256, 64, 2048, "tokens a forward step (a replay unit is one layer of one step)", "A"),
    "replay_layer_step": Input(5.2e-4, 1.0e-4, 2.0e-3, "one layer of one step re-executed from its committed input: its "
                               "weights' read (218 MB at 1.8 TB/s) plus 256 tokens at the plain rate; Qwen3-8B on H200 "
                               "is 0.18 ms a layer-step at B32 (note:circuits/20261018T0315Z-report-commit-defaults-"
                               "overlap)", "A"),
    "commit_per_byte": Input(4.34e-12, 2.2e-12, 1.3e-11, "SHA-512 rows on H200: 3.62 ps a byte plus 2.94 ns a 4 KiB leaf "
                             "(note:circuits/20261017T2200Z-report-commit-speed); sm_120 taken at 0.5-3x", "M"),
    "commit_floor": Input(0.060, 0.045, 0.101, "#2457's one serve-time term, now the 'floor' scheme only: COMMIT-GPU r5's "
                          "measured hashing of the int8 linears' s8 A operands in R = 128 x 32 leaves beside the GEMM on "
                          "sm_120 (r20261016-183051-a379 via BUDGET; 0.101 at R = 64), the leaves the Commit holds and PoUW's "
                          "seeds read under compile (decision 101): Llama-3.1-8B's 26,624 B a token a layer, which the "
                          "serve-time total prices with every other scale's bytes", "M"),
    "remote_receipts": Input(268e6, 268e6, 268e6, "the remote audit's receipts a GPU a pass: 64 B a 4 KiB block over the "
                             "registered store, 128 GiB on the 8-GPU Llama-3.1-8B node (memory-accounting, 17:02Z)", "D"),
    "passes_per_s": Input(65, 1, 82, "remote-audit passes a GPU-second: about 60-70 at 12.1 ms a pass; 82 back to back; "
                          "whether every served step is a pass is open (memory-accounting)", "A"),
    "decode_tokens": Input(32, 32, 32, "tokens a decode step (B32, the scales table's decode)", "A"),
    "compress_latency": Input(4.3e-6, 2.0e-6, 6.0e-6, "seconds a dependent SHA-512 compression on decode's critical path: "
                              "Pearl-C's s512_rows chain, 28 deep in 121 us (r20261015-061930-89bf)", "M"),
    "zk_unit": Input(2.0, 1.0, 4.0, "a zk unit's proving over the non-zk sweep's (A6 is zk, the sweep is not)", "A"),
    "noise_bits": Input(1, 1, 2, "noise bits an entry (compute accounting, decision 105)", "A"),
    "zero_work_templates": Input(30, 15, 60, "the layer's templates that carry no work (quantizers, softmax, norms, "
                                 "the epilogues' pieces), each held at its floor", "A"),
}

#: ANDs of the pieces (#2391's pins and hm96 PROTOCOL §5)
S8_STEP = 4_190
EPILOGUE = 5_000
SHA512 = 58_120
BF16_STEP = 8_233



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
    - `h1` (compile, W8A8, H1 + fx1): A from the Commit's roots (decision 101), its rows s8, the norm outputs pinned into
      the Commit (`operand-inside`), rope_q pinned as QK^T's A; the CP-W receipts (2 (K/32 + m) u32 a 32 x 32 tile; m 2 at
      K = 4,096 and 3 at 12,288, taken as 3 at 14,336); `h1-bf16` the same with the A rows BF16;
    - `kv`: the remote KV append's payload, one 4 KiB block a token-layer, when live KV is credited (ASK-19: not for chat).

    The remote audit's receipts are a GPU's, not a token's (`I['remote_receipts']`), and `floor` is #2457's measured term
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

#: the measurements the cost model is fitted from (`costfit`); ands are a unit's with its rows' openings
SWEEP_ANDS = 384 * BF16_STEP + 77 + 2 * (116_240 + SHA512 * 97) + (116_240 + SHA512)
SWEEP = {1: (0.6769, 224.84), 2: (0.6319, 224.58), 4: (0.6406, 225.2), 8: (0.6400, 222.6), 16: (0.6849, 220.6),
         32: (0.6288, 219.0), 64: (0.6609, 200.2), 128: (1.048, 193.6), 256: (1.019, 193.6), 512: (1.369, 190.5),
         1024: (2.081, 169.4)}
A6_ANDS = {2048: 64 * S8_STEP + EPILOGUE + 2 * (116_240 + SHA512 * 17) + 3 * (116_240 + SHA512),
           8192: 256 * S8_STEP + EPILOGUE + 2 * (116_240 + SHA512 * 65) + 3 * (116_240 + SHA512)}


def observations() -> list[Observation]:
    """benchmarks/proof_throughput's bf16-k6144 sweep (non-zk, a 1,024-unit statement, runs in its baseline.json, prove
    run r20261017-103527-eb06); A6's int8 statements (zk, run r20261017-202742-da01: 33 s and 69 s staging, 32 s and 83 s
    Lean verify, 1.4 s and 2.2 s a session of about 11 and 6 drawn units); PRF-57's small CPU sessions (about 1.15 s Lean
    verify, note:proofs/20261017T0552Z-report-proof-service-cost-model)."""
    out = [Observation(f"proof_throughput:B{b}", "GemmCoordinate_v2{K=6144}", SWEEP_ANDS, 1024, b, prove=p)
           for b, (p, _) in SWEEP.items()]
    out += [Observation("r20261017-202742-da01", "ScaledMmS8Coordinate{K=2048}", A6_ANDS[2048], 4096, 11, prove=1.4,
                        verify=32.0, stage=33.0),
            Observation("r20261017-202742-da01", "ScaledMmS8Coordinate{K=8192}", A6_ANDS[8192], 2048, 6, prove=2.2,
                        verify=83.0, stage=69.0),
            Observation("note:proofs/20261017T0552Z-report-proof-service-cost-model", "Pc4Tile_v1{K=4096}",
                        A6_ANDS[2048] * 2, 16, 16, verify=1.15)]
    return out


KEY = {"prover": "flock-circuit 1216db812a (binary dff9f6fbc8bb) for the per-unit slope; source 7107bab611 (A6) for the "
                 "zk session",
       "verifier": "flock-verify 7107bab611 (A6); build_key 4163b0b7fc3d3cfc in the sweep (not fitted: an older parse)",
       "hardware": "RTX PRO 6000 Blackwell Server Edition; Xeon 6776P (sweep), EPYC-Genoa (A6)",
       "templates": "GemmCoordinate_v2{K=6144,DOT=HopperBF16WgmmaDot16_v1} (statement 6b661a3d), "
                    "ScaledMmS8Coordinate_v1{K=2048|8192} (A6)"}
