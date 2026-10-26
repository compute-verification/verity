"""Verity-semantics replay producer: the registry Definition `AttentionHead_v3{T, D, BN}` (`verity_vllm/program/registry/b1.py`)
evaluated gate by gate by the IR oracle (`verity.ml.circuits.evaluate.evaluate_call`), with the committed stream of
`encoding.py` read out of the execution transcript by walking the call tree.

This is the "deterministic Verity-semantics reference" of the fixture in the strict sense: the values are the pinned
Definition's, not a re-implementation (the numpy `reference.replay_row` approximates MUFU.EX2, the tensor-core adder and
the FFMA contraction and is for mechanics only; `compare_reference` counts where they differ).  Cost: Python per gate —
tens of milliseconds per row, so tests use it on a few rows.

Correspondence used (Definition -> stream), all read from `AttnBlock_v3` / `AttentionHead_v3` bodies:
    S[c]   = return of the c-th `DotBf16{K=D}` child of the block call (columns ascending)
    M      = block return 0 (`m_new`)
    ALPHA  = return of the FIRST `MufuEx2Ftz` child of a non-first block (`scores_scale`; the body creates it before the P's)
    P32[c] = returns of the remaining `MufuEx2Ftz` children (columns ascending)
    L[ln]  = block returns 1..4, O[d] = block returns 5..5+D
    INV    = return of the `Fa2InvSum` child of the head call
    P16[c] = returns of the `F2fpBf16` children of the block (derived; used to check `derived.conv_f2fp_bf16`)
"""

from __future__ import annotations

import hashlib
import lzma
from functools import lru_cache
from pathlib import Path

import numpy as np

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluate import evaluate_call
from verity.ml.circuits.program import Program
from verity_vllm.program.registry import b1

from .encoding import NLANES, RowLayout

# The MUFU.EX2 / MUFU.RCP tables (2 x 2^23 u32 = 64 MB) are the pinned semantics of `MufuEx2Ftz` / `Fa2InvSum`.
# `verity_vllm.program.kernels.fa2_model.MufuTables.load` reconstructs them through int64 temporaries (peak ~210 MB traced,
# ~250 MB resident afterwards on macOS because the freed pages stay mapped).  The laptop's `~/.veritor/mem_guardian.py`
# kills any lane python above 256 MiB while the data volume is below its 6-GB disk floor (it was at 1.7 GB free on
# 2026-09-18), so this module fills the numerics lane's table cache with the SAME arrays reconstructed chunkwise
# (identical int64 arithmetic, u32 wrap, peak ~75 MB).  Equality with the original loader is pinned by these digests
# (computed once with `MufuTables.load` on this laptop; `test_oracle.py::test_low_peak_tables_match_the_original_loader`).
MUFU_TABLE_SHA256 = {
    "ex2": "b2a42c4a626469c986e33e43f16f41bde9d84de94347cd9ceaa1bd68d36bcdf0",
    "rcp": "c408381446132e07fa5581e930dc33fb790bea7ec4276ef3621e322df2492a55",
}


def preload_mufu_tables_low_peak() -> None:
    from verity_vllm.program.kernels import fa2_model as FA
    from verity_vllm.program.kernels import fa2_relation

    d = fa2_relation.tables_dir()
    if d in fa2_relation._TABLES:
        return
    tabs, src = [], {}
    for name, t0 in (("ex2", FA.EX2_T0), ("rcp", FA.RCP_R0)):
        p = Path(d) / f"mufu_{name}_delta_int8.xz"
        raw = p.read_bytes()
        delta = np.frombuffer(lzma.decompress(raw), dtype=np.int8)
        if delta.shape[0] != FA.TABLE_N - 1:
            raise ValueError(f"{p}: expected {FA.TABLE_N - 1} deltas, got {delta.shape[0]}")
        t = np.empty(FA.TABLE_N, dtype=np.uint32)
        t[0] = t0
        run, CH = int(t0), 1 << 20
        for lo in range(0, FA.TABLE_N - 1, CH):
            c = np.cumsum(delta[lo:lo + CH], dtype=np.int64)
            c += run
            t[1 + lo:1 + lo + c.size] = c.astype(np.uint32)   # the original's `int64 -> astype(uint32)` wrap
            run = int(c[-1])
        got = hashlib.sha256(t.tobytes()).hexdigest()
        if got != MUFU_TABLE_SHA256[name]:
            raise RuntimeError(f"low-peak MUFU {name} table digest {got} != pinned {MUFU_TABLE_SHA256[name]}")
        tabs.append(t)
        src[name] = {"file": p.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    fa2_relation._TABLES[d] = FA.MufuTables(tabs[0], tabs[1], src)


preload_mufu_tables_low_peak()

_STANDALONE = CompositeDefinition("Standalone", 1, ("FN",), lambda S: S.FN.signature(),
                                  lambda B, S, *args: B.call(S.FN, *args),
                                  doc="root wrapper: one call of F over Input gates (as `verity_vllm.program.registry.crosscheck`)",
                                  register=False)


@lru_cache(maxsize=256)
def _head_call(T: int, D: int, BN: int):
    prog = Program(bind(_STANDALONE, FN=bind(b1.AttentionHeadV3, T=T, D=D, BN=BN)))
    calls = [c for c in prog.circuit.root_calls() if not c.is_primitive]
    assert len(calls) == 1, calls
    return prog, calls[0]


def oracle_row_full(layout: RowLayout, q16: np.ndarray, k16: np.ndarray, v16: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(committed words in layout order, bf16 output row, P16 per visited block) from the IR oracle."""
    T, D, BN = layout.T, layout.D, layout.BN
    prog, call = _head_call(T, D, BN)
    args = [int(x) for x in np.asarray(q16, dtype=np.uint16)] \
        + [int(x) for x in np.asarray(k16[:T], dtype=np.uint16).ravel()] \
        + [int(x) for x in np.asarray(v16[:T], dtype=np.uint16).ravel()]
    t: dict[int, int] = {}
    out = evaluate_call(prog.circuit, call, args, transcript=t)
    words = np.zeros(layout.count, dtype=np.uint32)
    blocks = [c for c in call.children() if c.matches("AttnBlock")]
    if len(blocks) != layout.nb:
        raise RuntimeError(f"oracle: {len(blocks)} AttnBlock calls, layout expects {layout.nb}")
    p16s: list[np.ndarray] = []
    for v, blk in enumerate(blocks):
        kids = list(blk.children())
        S = [t[c.returns[0]] for c in kids if c.matches("DotBf16")]
        ex2 = [t[c.returns[0]] for c in kids if c.matches("MufuEx2Ftz")]
        p16 = [t[c.returns[0]] for c in kids if c.matches("F2fpBf16")]
        first = v == 0
        alpha, P = (None, ex2) if first else (ex2[0], ex2[1:])
        nvis = len(S)
        if len(P) != nvis or len(p16) != nvis:
            raise RuntimeError(f"oracle: block {v}: {nvis} S, {len(P)} P, {len(p16)} P16")
        ret = blk.returns
        lo, hi = layout.block_span(v)
        cur = lo
        words[cur:cur + nvis] = S; cur += nvis
        words[cur] = t[ret[0]]; cur += 1
        if not first:
            words[cur] = alpha; cur += 1
        words[cur:cur + nvis] = P; cur += nvis
        words[cur:cur + NLANES] = [t[g] for g in ret[1:1 + NLANES]]; cur += NLANES
        words[cur:cur + D] = [t[g] for g in ret[1 + NLANES:1 + NLANES + D]]; cur += D
        if cur != hi:
            raise RuntimeError(f"oracle: block {v} filled {cur - lo} words, layout span {hi - lo}")
        p16s.append(np.array(p16, dtype=np.uint16))
    inv = [c for c in call.children() if c.matches("Fa2InvSum")]
    if len(inv) != 1:
        raise RuntimeError("oracle: expected one Fa2InvSum in the head call")
    words[layout.count - 1] = t[inv[0].returns[0]]
    return words, np.array(out, dtype=np.uint16), p16s


def oracle_row(layout: RowLayout, q16: np.ndarray, k16: np.ndarray, v16: np.ndarray, *, upto_version: int | None = None
               ) -> tuple[np.ndarray, np.ndarray | None]:
    """`reference.replay_row`-compatible producer.  ``upto_version`` only truncates what is returned (the oracle always
    evaluates the whole row; the layouts read `[:block_span(upto_version)[1]]`)."""
    words, out16, _ = oracle_row_full(layout, q16, k16, v16)
    if upto_version is not None and upto_version < layout.nb - 1:
        return words, None
    return words, out16


def compare_reference(layout: RowLayout, q16: np.ndarray, k16: np.ndarray, v16: np.ndarray) -> dict[str, tuple[int, int]]:
    """Per-stage (mismatched, total) word counts between the numpy `reference.replay_row` and the oracle on one row."""
    from .reference import replay_row

    ref, ref_out = replay_row(layout, q16, k16, v16)
    orc, orc_out, _ = oracle_row_full(layout, q16, k16, v16)
    counts: dict[str, list[int]] = {}
    for i in range(layout.count):
        st = layout.unrank(i).stage
        c = counts.setdefault(st, [0, 0])
        c[1] += 1
        c[0] += int(ref[i] != orc[i])
    counts["OUT16"] = [int(np.count_nonzero(ref_out != orc_out)), int(orc_out.size)]
    return {k: (v[0], v[1]) for k, v in counts.items()}
