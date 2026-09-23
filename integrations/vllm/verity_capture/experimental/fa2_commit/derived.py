"""Derived-value openings: a value that is a declared deterministic function of ONE committed word is not committed
again; it is opened by opening its source and recomputing the conversion.  All consumers are thereby forced to the same
derived value (there is no second leaf that could carry a different one).

Two conversions are declared:
    F2fpBf16  fp32 -> bf16, RNE (`F2FP.BF16.PACK_AB`; NaN -> 0x7fff).  Not injective: the fp32 P32 is the committed source.
    Bf16ToF32 bf16 -> fp32 (`x << 16`).  Injective: the bf16 native word (e.g. the `fused_add_rms_norm` residual output,
              a committed Instance output) is the source; the fp32 lift the norm kernel consumes is derived.

Rule (verifier side): for a derived identity d with `DERIVED[d.stage] = (src_stage, conv)`, accept a claimed word w iff the
source identity s (same layer/head/qrow/block/index/version, stage src_stage) opens to a word u under the ORIGINAL root and
`conv(u) == w`.  The derived gate stays in the Program and in the resource accounting (it is a gate whose output is
checked by recomputation, not a gate that disappears); only its commitment position is elided.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .encoding import DERIVED, STAGE_WIDTH, ValueId
from .layouts import HierCommitment, HierLayout, PackedCommitment, PackedLayout, ValueOpening, verify_value_opening
from .reference import f32_to_bf16_rne


def conv_f2fp_bf16(u32_word: bytes) -> bytes:
    x = np.frombuffer(u32_word, dtype=np.uint32).view(np.float32)
    return f32_to_bf16_rne(x).tobytes()


def conv_bf16_to_f32(u16_word: bytes) -> bytes:
    return (int.from_bytes(u16_word, "little") << 16).to_bytes(4, "little")


CONVERSIONS = {"F2fpBf16": conv_f2fp_bf16, "Bf16ToF32": conv_bf16_to_f32}


@dataclass(frozen=True, slots=True)
class DerivedOpening:
    vid: ValueId               # the derived identity (stage in DERIVED)
    word: bytes                # the claimed derived word
    conversion: str
    source: ValueOpening       # the committed source, opened under the original root

    def replace(self, **changes):  # noqa: ANN001, ANN201
        return dataclasses.replace(self, **changes)


def source_of(vid: ValueId) -> tuple[ValueId, str]:
    src_stage, conv = DERIVED[vid.stage]
    return dataclasses.replace(vid, stage=src_stage), conv


def open_derived(commitment: HierCommitment | PackedCommitment, fx, vid: ValueId) -> DerivedOpening:  # noqa: ANN001
    src, conv = source_of(vid)
    src_open = commitment.open(fx, src)
    return DerivedOpening(vid, CONVERSIONS[conv](src_open.word), conv, src_open)


def verify_derived(R: bytes, layout: HierLayout | PackedLayout, d: DerivedOpening) -> bool:
    try:
        if d.vid.stage not in DERIVED:
            return False
        src_stage, conv = DERIVED[d.vid.stage]
        if d.conversion != conv or d.source.vid != dataclasses.replace(d.vid, stage=src_stage):
            return False
        if len(d.word) != STAGE_WIDTH[d.vid.stage]:
            return False
        if not verify_value_opening(R, layout, d.source):
            return False
        return CONVERSIONS[conv](d.source.word) == d.word
    except (KeyError, ValueError, TypeError):
        return False
