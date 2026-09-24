"""Primitives of the padded serving circuit (`Serve@3`) — lane PAD-IMPL, spec §9 of
`docs/vllm-poc/serve3-spec.md` (PAD-SPEC v1), GUIDANCE §29.

The padded circuit carries one live/finished bit per decode step and selects, at decode-step granularity,
between the step's computed outputs and the canonical pad values.  The gates this takes are boolean algebra
on `Value<1>` plus typed two-way selects; the scalar selects and the i32 equality already exist in
`prims.py` and are REUSED, not redefined (spec §9: "Existing, reused — not new, do not re-register"):

  spec name (§9)   registered id   where
  --------------   -------------   -----
  I32Eq_v1         I32Eq_v1        prims.py (reused; 1 iff the two 32-bit words are equal)
  SelectI32_v1     SelectI32_v1    prims.py (reused; c ? a : b on encodings)
  SelectBf16_v1    SelectBf16_v1   prims.py (reused)
  SelectF32_v1     SelectF32_v1    prims.py (reused; unused by Serve@3 v1 itself, spec §15 Q5)
  BitAnd_v1        BitAnd_v1       NEW (this file)
  BitNot_v1        BitNot_v1       NEW (this file)

Bit-exact semantics (total on encodings; `Value<1>` has exactly the encodings {0, 1}):
  * `BitAnd_v1(a, b) = (a & b) & 1` — absorbing for the finished chain: `BitAnd(0, x) = 0` for every `x`,
    so `live_{g+1} = BitAnd(live_g, BitNot(I32Eq(y_g, EOS_ID)))` is non-increasing (spec §4).
  * `BitNot_v1(a) = (a ^ 1) & 1`.

Typed selects over *collections* are `batch(Select<T>_v1, …)` — the existing structural form, one select
gate per word, the constant side a zero-stride broadcast of one zero gate.  No new collection-level
primitive exists or is needed (spec §9); the row muxes of `PadDecodeStep_v1` are authored that way in
`serve3.py`.

Backend/spec mirrors (the F2fpBf16 wiring pattern, one row per gate kind):
  * SP1/Rust statement table: gate set `veritor.pad-serve@1` (`zk/sp1/common/src/gateset.rs`, semantics in
    `zk/sp1/common/src/pad.rs`) — gates `select_bf16`, `select_f32`, `select_i32`, `eq_i32`, `bit_and`,
    `bit_not` (+ `in`/`weight` sources at width 32); Python twin `src/veritor/core/pad_serve.py`.
  * Lean primitive table: `lean/VeritorSpec/PadServe.lean` (`PadServe.family`), `Lawful`, with the
    select/equality/boolean semantics and the absorbing step as theorems.
`tests/program/test_pad_prims.py` pins the three-way agreement on vectors; existing primitive semantics
are not modified anywhere.
"""

from __future__ import annotations

from verity.ir.defs import primitive
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BIT

# ---------------------------------------------------------------------------------------------------------
# boolean gates on Value<1> (the per-step live/finished flag algebra, spec §4/§9)
# ---------------------------------------------------------------------------------------------------------


@primitive("BitAnd", 1, [("a", BIT), ("b", BIT)], BIT, conformance="protocol-defined")
def BitAnd(a: int, b: int) -> int:
    """Logical and on 1-bit encodings: `(a & b) & 1` (spec §9).  Absorbing for the finished chain:
    `BitAnd(0, x) = 0` for every `x` — once a request's live bit falls, it never rises."""
    return a & b & 1


@primitive("BitNot", 1, [("a", BIT)], BIT, conformance="protocol-defined")
def BitNot(a: int) -> int:
    """Logical not on 1-bit encodings: `(a ^ 1) & 1` (spec §9)."""
    return (a ^ 1) & 1


ALL_NEW = [BitAnd, BitNot]
REUSED = [P.I32Eq, P.SelectBf16, P.SelectF32, P.SelectI32]

__all__ = ["BitAnd", "BitNot", "ALL_NEW", "REUSED"]
