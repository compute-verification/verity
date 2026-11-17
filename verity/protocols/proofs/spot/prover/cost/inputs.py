"""The cost model's rows and their inputs (Llama-3.1-8B and Qwen3-8B): each input a value, a low and a high, its source
and its status (M measured, A assumed, D derived from the others), in one currency, GPU-seconds of an RTX PRO 6000
(sm_120); the models' shapes; what every scale commits at serve time, a token a layer, by PoUW scheme (the scales table);
and the cost model's observations by card (`CARDS`: the RTX PRO 6000's, and the `h200` card's in H200 GPU-seconds).

The measured values are data under `calibration/sampling_plan/` (`inputs.json`, `MEASURED`), handed in by `load`; the
names below are the default file's."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, log2
from pathlib import Path

from verity.commitments.periods.interface import K_MAX, proof_block_k_log
from verity.protocols.proofs.spot.prover.costfit import Observation
from verity.protocols.proofs.spot.prover.optimize import SHA512_ANDS

#: the measured inputs, kept with the measurements that made them
MEASURED = Path(__file__).resolve().parents[6] / "calibration" / "sampling_plan" / "inputs.json"


@dataclass(frozen=True)
class Input:
    value: float
    low: float
    high: float
    source: str
    status: str


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
    """The ``k_log`` of the Flock block that holds one proof block of ``ands`` ANDs opening ``rows`` (bytes each, the
    output's row last) before the composer packs it: `proof_block_k_log` unpacked. A row's opening is its hm96 and
    sha512x3 slots, not ANDs in the unit's slot."""
    return proof_block_k_log(rows[:-1], rows[-1:], ands, packed=False)


def cap(ands: int, rows: Sequence[int]) -> int:
    """A statement's proof blocks of ``ands`` ANDs opening ``rows``: one a Flock block of 2^`block_k_log` under
    m = k_log + nbl <= 35, at most 2^20 (below 2^22 the composer packs several a block, about as many a statement); 0, not a
    proof block, past Flock's 2^`K_MAX`."""
    k = block_k_log(ands, rows)
    return 0 if k > K_MAX else min(1 << (35 - k), 1 << 20)


def s8_coord_ands(k: int) -> int:
    """One s8 coordinate's ANDs with its rows' openings, as `costfit` counts them: K/32 steps, the epilogue, two K-byte
    operand rows and three one-block rows (the scales and the output), each at `optimize.row_ands`."""
    from verity.protocols.proofs.spot.prover.optimize import row_ands

    return k // 32 * S8_STEP + EPILOGUE + 2 * row_ands(k) + 3 * row_ands(1)


@dataclass(frozen=True)
class Measured:
    """One inputs file: `inputs` by name, the `models`' shapes, and each card's observations, the builds and hardware
    they key (`keys`) and its price ratio, an input's name (`cards`)."""

    inputs: dict[str, Input]
    models: dict[str, Model]
    s8_run: str
    s8: dict[tuple[int, int, int], tuple[float, float, float]]
    opening: str
    opening_ands: int
    opening_sessions: dict[str, dict[tuple[int, int], float]]
    h200: dict[str, dict[tuple[str, int, int, int], tuple[float, float | None, float]]]
    cards: dict[str, tuple[dict[str, str], str]]

    def observations(self, card: str = "rtx-pro-6000") -> list[Observation]:
        if card not in self.cards:
            raise ValueError(f"no card {card!r}: one of {', '.join(self.cards)}")
        if card == "h200":
            return [Observation(run, t, a, c, d, prove=p, verify=v, stage=s)
                    for run, points in self.h200.items() for (t, a, c, d), (p, v, s) in points.items()]
        return [Observation(self.s8_run, f"ScaledMmS8Coordinate_v1{{K={k}}}", s8_coord_ands(k), c, d, prove=p, verify=v,
                            stage=s) for (k, c, d), (p, v, s) in self.s8.items()] + \
            [Observation(run, self.opening, self.opening_ands, c, d, prove=p)
             for run, points in self.opening_sessions.items() for (c, d), p in points.items()]


def load(path: Path = MEASURED) -> Measured:
    """The inputs file at ``path`` (`MEASURED` by default)."""
    d = json.loads(Path(path).read_text())
    return Measured(
        inputs={k: Input(**v) for k, v in d["inputs"].items()},
        models={k: Model(**v) for k, v in d["models"].items()},
        s8_run=d["s8"]["run"],
        s8={(k, c, n): (p, v, s) for k, c, n, p, v, s in d["s8"]["points"]},
        opening=d["opening"]["template"], opening_ands=d["opening"]["ands"],
        opening_sessions={run: {(c, n): p for c, n, p in pts} for run, pts in d["opening"]["sessions"].items()},
        h200={run: {(t, a, c, n): (p, v, s) for t, a, c, n, p, v, s in pts} for run, pts in d["h200"].items()},
        cards={card: (c["key"], c["price"]) for card, c in d["cards"].items()})


DEFAULT = load()
I = DEFAULT.inputs
MODELS = DEFAULT.models
CARDS = DEFAULT.cards
KEY = CARDS["rtx-pro-6000"][0]
KEY_H200 = CARDS["h200"][0]
observations = DEFAULT.observations
