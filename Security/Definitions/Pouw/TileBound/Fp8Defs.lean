import Definitions.Pouw.TileBound.Defs

/-!
# The FP8-tile model at sm_120's E4M3 atom (trusted definitions)

The TT_OUT restatements lane (bc-b58c6093), on bc-d9842080's FP4-tile model (`Pouw.TileBound`, reused as is). The honest
source is the E4M3 atom `mma.sync ... kind::f8f6f4 ... m16n8k32` (`BLACKWELL_SM120_E4M3_M16N8K32`, one group of 32 on a
26-bit adder), priced by its whole fragment: 16·8·32 = 4,096 slots at `price .fp8 = 1` each. The adversary may use every
multiplier in `Fmt` at `price`, the FP4-tile model's measured table: NVFP4 0.5 per slot, E4M3 or int8 1, BF16 or FP16 2,
TF32 4, FFMA 8.46 (measured since at 8.0–8.15, `price`'s docstring). Unlike the FP4 case, the honest source is not the
cheapest multiplier; NVFP4 is.

* **Blocks.** E4M3 has no block scale (the row scale α is folded into the codes), so a block is one entry (`e4m3X`,
  `e4m3Y`).
* **Capacities** (`capacity8`): how many E4M3 entries one exactly representable value of a format can combine, for
  full-width 4-bit significands at one exponent.
  - E4M3 holds 1 in the model's per-slot rule. That rule fails per pair: on in-domain codes the sum and the difference
    of two codes are both E4M3 for 28–30% of pairs. What holds is its fragment form: 0 of 160 whole 16 × 32 operand
    fragments merge (`fp8-merge-rate/sm120`, `pearlc_merge_rate.py`). A tensor-core pre-add feeds a whole fragment,
    whose inputs are shared across 8 columns or 16 rows, so the closure of E4M3 pre-adds rests on that fragment rate,
    not on this per-slot capacity.
  - BF16 holds 16 (8 bits), FP16 128 (11 bits), TF32 128 (11 bits) and FFMA 2²⁰ (24 bits).
  - int8, and `dp4a` on int8 operands, hold 8 E4M3 significands at one scale (15·8 ≤ 127).
  - NVFP4 holds 0 on in-domain operands. A block-scaled E2M1 value has 6 significant bits, but the 16 entries of a block
    share one UE4M3 scale, and no in-domain 16-block of E4M3 codes is one NVFP4 limb (0%, `r20260930-055203-10e8`).
    That zero is a measured fact about the operands, not a format width: the hypothesis `fp8-limb-rate/sm120`. So
    NVFP4 is closed only through the operands, and limb splitting is nonlinear forming, the residue `a2/fp8-tile`.

* **int8 and `dp4a`** (both now in `Fmt`). At full width, int8 at E4M3's price holds 8 entries, so the count closes it
  only at 1 entry (`int8_closed_upto8`). `dp4a` at 4.05 is closed up to 4 (`dp4a_closed_upto8`). On blocks sharing an
  exponent window, int8 Strassen at depth 1 would cost 7/8 of the honest MACs before its combine. What excludes int8 is
  the operands (≤ 0.29% of in-domain slice pairs are one int8 limb on both sides, `fp8-limb-rate/sm120`) and each
  32-slice's rescale (I2F plus FFMA, +1.27 per MAC at the measured prices), not the count.
* **The honest program** `honest8` is stated here, self-contained, so that the pins reading it read only trusted
  definitions.
-/

namespace Pouw.TileBound

/-- E4M3 entries per exactly representable value of each format, for full-width 4-bit significands at one exponent:
NVFP4 0 on in-domain blocks, E4M3 1, BF16 16, FP16 128, TF32 128, FFMA 2²⁰, int8 and `dp4a`'s int8 operands 8
(15·8 ≤ 127). -/
def capacity8 : Fmt → ℕ
  | .fp4 => 0
  | .fp8 => 1
  | .bf16 => 16
  | .tf32 => 128
  | .fp32 => 2 ^ 20
  | .int8 => 8
  | .fp16 => 128
  | .dp4a => 8

variable {K : Type*} {m k n : ℕ} {τ : Type*}

/-- E4M3's X-blocks: single entries. -/
def e4m3X (m k : ℕ) : XBlocks m k (Fin m × Fin k) := ⟨id, fun _ _ h => by rw [id, id] at h; rw [h]⟩

/-- E4M3's Y-blocks: single entries. -/
def e4m3Y (k n : ℕ) : YBlocks k n (Fin k × Fin n) := ⟨id, fun _ _ h => by rw [id, id] at h; rw [h]⟩

/-- **The representability rule at E4M3:** each input of each slot is one exactly representable value of its format, so
it reads at most `capacity8` blocks, X-blocks and Y-blocks together. `capacity8` counts entries, so the intended blocks
are the single entries `e4m3X` and `e4m3Y`. The pins hold for every partition, which is stronger than needed. -/
def SlotProg.Representable8 [Field K] {β γ : Type*} [Fintype β] [Fintype γ] (P : SlotProg K m k n τ)
    (BX : XBlocks m k β) (BY : YBlocks k n γ) : Prop :=
  ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ capacity8 (P.fmt t) ∧
    BX.touches (P.a' t) + BY.touches (P.b' t) ≤ capacity8 (P.fmt t)

/-- **The honest E4M3 program:** one E4M3 slot per triple `(i, l, j)`, the unit form of `X (i, l)` times the unit form of
`Y (l, j)`, into output `(i, j)`. -/
def honest8 (K : Type*) [Field K] (m k n : ℕ) : SlotProg K m k n (Fin m × Fin k × Fin n) where
  a t := fun e => if e = (t.1, t.2.1) then 1 else 0
  b _ := 0
  a' _ := 0
  b' t := fun e => if e = (t.2.1, t.2.2) then 1 else 0
  w t := fun e => if e = (t.1, t.2.2) then 1 else 0
  fmt _ := .fp8

end Pouw.TileBound
