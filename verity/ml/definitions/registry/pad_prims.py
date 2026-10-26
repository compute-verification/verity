"""Primitives of the padded serving circuit (`Serve@3`) — lane PAD-IMPL, §9 of PAD-SPEC v1,
GUIDANCE §29.

The padded circuit carries one live/finished bit per decode step and selects, at decode-step granularity,
between the step's computed outputs and the canonical pad values.  The gates this takes are boolean algebra
on `Value<1>` plus typed two-way selects, all of them core's basic primitives (`verity.ml.circuits.ops.scalar`), cited here,
not redefined (spec §9: "Existing, reused — not new, do not re-register"):

  spec name (§9)   registered id   where
  --------------   -------------   -----
  I32Eq_v1         I32Eq_v1        verity.ml.circuits.ops.scalar (1 iff the two 32-bit words are equal)
  SelectI32_v1     SelectI32_v1    verity.ml.circuits.ops.scalar (c ? a : b on encodings)
  SelectBf16_v1    SelectBf16_v1   verity.ml.circuits.ops.scalar
  SelectF32_v1     SelectF32_v1    verity.ml.circuits.ops.scalar (unused by Serve@3 v1 itself, spec §15 Q5)
  BitAnd_v1        BitAnd_v1       verity.ml.circuits.ops.scalar (the spec's new gate)
  BitNot_v1        BitNot_v1       verity.ml.circuits.ops.scalar (the spec's new gate)

Bit-exact semantics (total on encodings; `Value<1>` has exactly the encodings {0, 1}):
  * `BitAnd_v1(a, b) = (a & b) & 1` — absorbing for the finished chain: `BitAnd(0, x) = 0` for every `x`,
    so `live_{g+1} = BitAnd(live_g, BitNot(I32Eq(y_g, EOS_ID)))` is non-increasing (spec §4).
  * `BitNot_v1(a) = (a ^ 1) & 1`.

Typed selects over *collections* are `batch(Select<T>_v1, …)` — the existing structural form, one select
gate per word, the constant side a zero-stride broadcast of one zero gate.  No new collection-level
primitive exists or is needed (spec §9); the row muxes of `PadDecodeStep_v1` are authored that way in
`serve3.py`.

Backend/spec mirrors (the F2fpBf16 wiring pattern, one row per gate kind):
  * SP1/Rust statement table: gate set `veritor.pad-serve@1` (`archive/sp1/common/src/gateset.rs`, semantics in
    `archive/sp1/common/src/pad.rs`) — gates `select_bf16`, `select_f32`, `select_i32`, `eq_i32`, `bit_and`,
    `bit_not` (+ `in`/`weight` sources at width 32); its own tests pin the gate set's digest and evaluations.
  * Lean primitive table: the predecessor implementation's `PadServe.family` (`Lawful`, with the
    select/equality/boolean semantics and the absorbing step as theorems); not in this repository.
`tests/program/test_pad_prims.py` pins the primitives on vectors; existing primitive semantics are not
modified anywhere.
"""

from __future__ import annotations

from verity.ml.circuits.ops.scalar import BitAnd, BitNot
from verity.ml.definitions.registry import prims as P

ALL_NEW = [BitAnd, BitNot]
REUSED = [P.I32Eq, P.SelectBf16, P.SelectF32, P.SelectI32]

__all__ = ["BitAnd", "BitNot", "ALL_NEW", "REUSED"]
