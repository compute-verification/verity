"""Kernel-tapped fixtures: B-kernel's P0 diagnostic dumps of the pinned FA2 kernel (R8 lane B-kernel,
`out/gen/r8/fa2-kernel/results/p0/*.npz`, GPU-measured on cp-fa2 with the instrumented rebuild of vllm-flash-attn
`506341a1`; outputs of that build are bit-identical to the installed `.so`).

A dump holds, per kernel head `hq` and tile row `r`, the LIVE REGISTER VALUES of every stage the census's Definition
names — `s_raw` (acc_s before masking, per absolute key column), `P_f32`, `rP_bf16`, `row_max` / `rescale` /
`row_sum_partial[4]` / `acc_o[64]` after every visited key block, `fin_inv_sum` — plus the bf16 inputs (as exact fp32
copies), the paged block table and the kernel's HBM `out`.  This module turns one dump into

  * an :class:`AttentionFixture` whose q/k/v are the kernel's input words (``values="recorded"``), and
  * a *kernel replay*: a `replay_row`-compatible producer that returns, for a (head, qrow), the committed stream of
    `encoding.RowLayout` filled with the tapped register words instead of a recomputation.

So a commitment built with the kernel replay is a commitment to what the GPU held in registers, and opening it with the
oracle replay (`oracle.oracle_row`, the registry Definition) is the end-to-end statement "the Definition regenerates the
kernel's words" — checked here word for word, not only at the bf16 output.

Row / head identity (from `p0_validate.py` and B-sem `correspondence.md` §0):
  non-swapped: kernel head hq = query head h, tile row r = query position (sk - sq) + r (bottom-right aligned causal);
  swapped decode (sq == 1, H > Hk): kernel head hq = KV head, tile row r = head-in-group -> h = hq * (H // Hk) + r.

Kernel tile schedule vs Definition row schedule: the tile visits `n_block_max` blocks for ALL its rows; a row whose
visible keys end before the tile's last block sees that block fully masked (kernel: m = -inf, P = 0, l = 0, O = 0,
then rescale = 0 into the next block).  The Definition `AttentionHead_v3{T}` has no versions for such a block, and
`RowLayout` has no positions: the kernel replay drops it (and the following block's ALPHA) by predicate and reports the
count in :attr:`KernelDump.skipped_blocks`.  In the accepted run this never happens (kBlockM 64 | BN 128 with 64-aligned
tiles and sq == sk); it does in chunked prefill with (sk - sq) % 64 != 0 or sq % 64 != 0 (dumps 40x200, 64x300).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .encoding import NLANES, Geometry, RowLayout
from .fixture import AttentionFixture, Context

_BKERNEL_P0 = Path("/private/tmp/w-perf/out/gen/r8/fa2-kernel/results/p0")          # B-kernel's worktree (R8 only)
_LOCAL_P0 = Path(__file__).with_name("fixtures") / "p0"                                # unmodified copies + PROVENANCE.md
P0_DIR = _LOCAL_P0 if _LOCAL_P0.is_dir() else _BKERNEL_P0


def _bf16_words(x32: np.ndarray) -> np.ndarray:
    """Exact bf16 words of fp32 values that ARE bf16 values (the dump stores bf16 inputs widened to fp32)."""
    bits = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise ValueError("dump input is not bf16-representable")
    return (bits >> 16).astype(np.uint16)


def _f32_words(x32: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32)


@dataclass
class KernelDump:
    path: Path
    sha256: str
    sq: int
    sk: int
    H: int
    Hk: int
    causal: bool
    swapped: bool
    BN: int
    NB: int                                  # blocks the tile visits (n_block_max)
    fixture: AttentionFixture
    z: dict = field(repr=False)
    skipped_blocks: int = 0                  # fully-masked kernel blocks dropped by predicate (see module doc)

    # ---- identity ---------------------------------------------------------------------------------------------------
    def kernel_coords(self, head: int, qrow: int) -> tuple[int, int]:
        """(hq, r) of the dump arrays for a (query head, qrow) of the fixture."""
        ratio = self.H // self.Hk
        if self.swapped:
            return head // ratio, head % ratio
        return head, qrow - (self.sk - self.sq)

    def T(self, qrow: int) -> int:
        return qrow + 1

    # ---- the kernel replay ------------------------------------------------------------------------------------------
    def kernel_words(self, layout: RowLayout) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """(committed words in layout order, bf16 `out` row from HBM, P16 per visited block) from the tapped registers."""
        z = self.z
        hq, r = self.kernel_coords(layout.head, layout.qrow)
        T, D, BN, nb = layout.T, layout.D, layout.BN, layout.nb
        if nb > self.NB:
            raise ValueError(f"row T={T} needs {nb} blocks, tile visited {self.NB}")
        if self.NB > nb:
            self.skipped_blocks += self.NB - nb
        s_raw = _f32_words(z["s_raw"][hq, r])
        P32 = _f32_words(z["P_f32"][hq, r])
        rP = _bf16_words(z["rP_bf16"][hq, r])
        row_max = _f32_words(z["row_max"][hq, r])
        rescale = _f32_words(z["rescale"][hq, r])
        lsum = _f32_words(z["row_sum_partial"][hq, r])
        acc_o = _f32_words(z["acc_o"][hq, r])
        words = np.zeros(layout.count, dtype=np.uint32)
        p16s: list[np.ndarray] = []
        for v in range(nb):
            b = nb - 1 - v
            lo, hi = layout.block_span(v)
            c0, c1 = b * BN, min((b + 1) * BN, T)
            nvis = c1 - c0
            cur = lo
            words[cur:cur + nvis] = s_raw[c0:c1]; cur += nvis
            words[cur] = row_max[b]; cur += 1
            if v > 0:
                words[cur] = rescale[b]; cur += 1
            words[cur:cur + nvis] = P32[c0:c1]; cur += nvis
            words[cur:cur + NLANES] = lsum[b]; cur += NLANES
            words[cur:cur + D] = acc_o[b]; cur += D
            if cur != hi:
                raise RuntimeError(f"kernel words: block v={v} filled {cur - lo}, span {hi - lo}")
            p16s.append(rP[c0:c1])
        words[layout.count - 1] = int(_f32_words(np.asarray([z["fin_inv_sum"][hq, r]]))[0])
        ratio = self.H // self.Hk
        out = z["out"][0, hq * ratio + r] if self.swapped else z["out"][r, hq]
        return words, _bf16_words(out), p16s

    def kernel_replay(self, layout: RowLayout, q16, k16, v16, *, upto_version: int | None = None):
        """`replay_row`-compatible: ignores q/k/v (the words are tapped, not recomputed)."""
        words, out16, _ = self.kernel_words(layout)
        if upto_version is not None and upto_version < layout.nb - 1:
            return words, None
        return words, out16


def load_dump(path: str | Path, *, layer: int = 0, request_id: str | None = None,
              program_digest: bytes | None = None) -> KernelDump:
    path = Path(path)
    raw = path.read_bytes()
    z = dict(np.load(path))
    sq, sk, H, Hk = int(z["sq"]), int(z["sk"]), int(z["H"]), int(z["Hk"])
    Hq, Mq = int(z["Hq"]), int(z["Mq"])
    causal, BN, PAGE = bool(z["causal"]), int(z["kBlockN"]), int(z["PAGE"])
    swapped = sq == 1 and H > Hk
    ratio = H // Hk
    NB = int(z["row_max"].shape[-1])
    if not swapped and (Hq != H or Mq != sq):
        raise ValueError(f"unexpected dump geometry Hq={Hq} Mq={Mq} for H={H} sq={sq}")
    if swapped and (Hq != Hk or Mq != ratio):
        raise ValueError(f"unexpected swapped-decode geometry Hq={Hq} Mq={Mq} for Hk={Hk} ratio={ratio}")
    if not causal and sq != 1:
        raise ValueError("non-causal multi-row dumps are outside the fixture's Context model")
    # K/V rows of the sequence through the paged block table
    bt = z["block_table"][0]
    seqused = int(z["seqused_k"][0])
    if seqused != sk:
        raise ValueError(f"seqused_k {seqused} != sk {sk}")
    kv_rows = np.concatenate([np.arange(PAGE) + int(b) * PAGE for b in bt])[:sk]
    k16 = _bf16_words(z["k"].reshape(-1, Hk, 64)[kv_rows])          # (sk, Hk, D)
    v16 = _bf16_words(z["v"].reshape(-1, Hk, 64)[kv_rows])
    geo = Geometry(D=64, BN=BN, QB=64, NH=H, KVH=Hk)
    if sq == 1:                                                     # decode: one query row at position sk - 1
        ctx = Context(program_digest or hashlib.sha256(b"fa2-kernel-p0").digest(), request_id or path.stem, 1, sk - 1, layer)
        q16 = _bf16_words(z["q"]).reshape(1, H, 64)                 # (1, H, D): q[0, h]
    else:                                                           # causal chunk: rows [sk - sq, sk)
        ctx = Context(program_digest or hashlib.sha256(b"fa2-kernel-p0").digest(), request_id or path.stem, 0, sk, layer,
                      chunk_lo=sk - sq)
        q16 = _bf16_words(z["q"]).reshape(sq, H, 64)                # q[r, h]
    fx = AttentionFixture(ctx, geo, q16, k16, v16, "recorded",
                          {"kind": "fa2-kernel-p0-dump", "file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                           "sq": sq, "sk": sk, "H": H, "Hk": Hk, "causal": causal, "kBlockN": BN, "swapped": swapped})
    return KernelDump(path, fx.source["sha256"], sq, sk, H, Hk, causal, swapped, BN, NB, fx, z)


def available_dumps(directory: Path = P0_DIR) -> list[Path]:
    return sorted(directory.glob("p0_*.npz")) if directory.is_dir() else []


def n_block_max(sq: int, sk: int, causal: bool, BN: int, QB: int = 64) -> int:
    nb = math.ceil(sk / BN)
    if causal:
        nb = min(nb, math.ceil((QB + sk - sq) / BN))
    return nb
