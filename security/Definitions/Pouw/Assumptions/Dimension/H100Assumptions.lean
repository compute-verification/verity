module

public import Mathlib.Data.Finset.Card
public import Mathlib.Data.Rat.Init
public import Mathlib.Tactic.ToAdditive
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! `MH100`'s price table `Prices100`, in units of one dense int8 MAC at 4,096 MACs per SM per clock, and the named hypotheses on it (`H32` and `FragLanes`: draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.H100

/-- A `wgmma` shape of the H100 (`sm_90`) at its least `k`, by operand type and accumulator. The instruction's `N` (8
to 256) sets how many accumulator registers it writes. The model does not price `N`: every `N` costs `regPrice` per
register. On hardware that is the price at `N ≥ 64` only. Round 7 measured m64n8k32 E4M3 → F32 at 144 per register,
and sparse n8 at 160. The H200 charges 144 too. The lower bounds are unaffected, since a real price is at least the
model's. But
the pins that exhibit programs at the nominal price (`MH100Attains`, `HonestSplitK100`, `DistinctWritesAttained`) hold
on hardware only at `N ≥ 64`. -/
inductive Shape100 where
  /-- E4M3 m64nNk32, FP32 accumulate: the honest atom `HOPPER_E4M3_K32`. -/
  | e4m3F32
  /-- E5M2 m64nNk32, FP32 accumulate. -/
  | e5m2F32
  /-- E4M3 m64nNk32, FP16 accumulate. -/
  | e4m3F16
  /-- E5M2 m64nNk32, FP16 accumulate. -/
  | e5m2F16
  /-- s8 or u8 m64nNk32, s32 accumulate. -/
  | s8
  /-- FP16 m64nNk16, FP32 accumulate. -/
  | f16F32
  /-- FP16 m64nNk16, FP16 accumulate. -/
  | f16F16
  /-- BF16 m64nNk16, FP32 accumulate. -/
  | bf16F32
  /-- TF32 m64nNk8, FP32 accumulate. -/
  | tf32

/-- The shape's `k`: MACs per accumulator element. -/
def Shape100.k : Shape100 → ℕ
  | .e4m3F32 | .e5m2F32 | .e4m3F16 | .e5m2F16 | .s8 => 32
  | .f16F32 | .f16F16 | .bf16F32 => 16
  | .tf32 => 8

/-- Accumulator elements per 32-bit register: 2 for FP16 accumulate (`.f16x2`), else 1. -/
def Shape100.elems : Shape100 → ℕ
  | .e4m3F16 | .e5m2F16 | .f16F16 => 2
  | _ => 1

/-- `MH100`'s prices, in units of one dense int8 MAC. -/
structure Prices100 where
  /-- The price per MAC of each `wgmma` shape. -/
  mac : Shape100 → ℕ
  /-- The least price per accumulator register written of every other tensor-core instruction. -/
  tc : ℕ
  /-- FFMA, FADD or FMUL, one register. -/
  ffma : ℕ
  /-- An INT32 instruction, one register. -/
  int32 : ℕ
  /-- An FP64 instruction, per 32-bit register of its result. -/
  fp64 : ℕ
  /-- Every other writer of a 32-bit word, in a register or in memory (data-dependent accesses, sub-word operations,
  memory-side reductions and atomics included), per word written. -/
  other : ℕ

/-- The price of a `wgmma` shape per 32-bit accumulator register written. -/
def Shape100.regPrice (pr : Prices100) (s : Shape100) : ℕ := s.k * s.elems * pr.mac s

/-- **Hypothesis `Floor c pr`** (named): every register writer of `MH100` costs at least `c` per
32-bit register written. -/
def Floor (c : ℕ) (pr : Prices100) : Prop :=
  (∀ s : Shape100, c ≤ s.regPrice pr) ∧ c ≤ pr.tc ∧ c ≤ pr.ffma ∧ c ≤ pr.int32 ∧ c ≤ pr.fp64 ∧ c ≤ pr.other

/-- **Hypothesis `H32 pr`** (named; draft semantics): every register writer of `MH100` costs at least 32 per 32-bit
register written.

It is a floor per instruction class, not per register in time. Round 8 found
no single writer below 32 loop-free (`h32LoopFree`, `h32Floor`). On the H200, none of 963 `sm_90` SASS forms costs
under 32 alone, and 26 are unpriced (the module's review). But writers on separate pipes co-issue, and mixes go
below 32 per register: as low as 21.45 loop-free (1:1 DADD + FADD with an immediate in place of one source each), and
21.33 at full dispatch (the FP64 pipe's bound). On the H200, 2:1:1 DADD + IMAD + FADD writes one every 21.54. So a
bound on `cost100` under `H32` bounds additive-W1 work, not time. A time bound needs a maximum over resources. -/
def H32 (pr : Prices100) : Prop := Floor 32 pr

/-- The data sheet's per-MAC prices. -/
def nominalMac : Shape100 → ℕ
  | .e4m3F32 | .e5m2F32 | .e4m3F16 | .e5m2F16 | .s8 => 1
  | .f16F32 | .f16F16 | .bf16F32 => 2
  | .tf32 => 4

/-- **The nominal H100 table**, the data sheet's and unmeasured; `tc` and `other`, which the data sheet does not give,
are 32, the least `H32` allows. -/
def h100Nominal : Prices100 where
  mac := nominalMac
  tc := 32
  ffma := 32
  int32 := 64
  fp64 := 32
  other := 32

/-- **Hypothesis `FloorTc a c pr`** (named): `a` per register, except `c` for the tensor-core instructions outside the
`wgmma` shapes. `Floor c pr` is `FloorTc c c pr`, so `H32 pr` is `FloorTc 32 32 pr`. -/
def FloorTc (a c : ℕ) (pr : Prices100) : Prop :=
  (∀ s : Shape100, a ≤ s.regPrice pr) ∧ c ≤ pr.tc ∧ a ≤ pr.ffma ∧ a ≤ pr.int32 ∧ a ≤ pr.fp64 ∧ a ≤ pr.other

/-- **Hypothesis `FragLanes S p q W`** (named; draft semantics): every multiplicand register in `W` depends on at most
`q` lanes on `S`. It is not a fact about the hardware and can fail (`FragShareGain`); its witness is `FragLanesSat`. -/
def FragLanes {G Λ : Type} (S : Set G) (p : Λ → G → ℚ) (q : ℕ) (W : Set (G → ℚ)) : Prop :=
  ∀ w ∈ W, ∃ J : Finset Λ, J.card ≤ q ∧ ∀ g ∈ S, ∀ g' ∈ S, (∀ j ∈ J, p j g = p j g') → w g = w g'

end Pouw.Dimension.H100
