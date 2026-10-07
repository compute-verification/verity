import Models.Pouw.Assumptions.Dimension.Embed

/-!
# The RTX 4090's tensor-core shapes in `M4090`, FP8 included (trusted statements; draft semantics)

**Draft semantics, S-1 to S-11 pending Daniel's review** (`barrier-assets/embed/DESIGN.md` §5).

`Op.mma outs p` is `M4090`'s tensor-core class: any functions `outs`, one per 32-bit accumulator register written, at
the full instruction's price `p`, well-formed when `16·len(outs) ≤ p`. `Shape4090` names the 4090's `mma.sync` shapes
with their measured prices (`internal/pouw/gpu-constants.md` §1, Round 1; units of one int8 MAC), and
`mmaOp s outs = Op.mma outs (regPrice s · len(outs))` is the instruction of shape `s`. Nothing in `Op` changes: the
shapes are instances of `Op.mma`, so every `M4090` theorem (`EmbedM4090`, `PerWordM4090`, `PerWordM4090Gap`) covers
programs that use them.

* **Price per register** `regPrice s = k · elements per register · price per MAC`. An FP16 accumulator register is
  `.f16x2`, two elements, so its price is twice the per-element price.

  | Shape | Price per MAC | Per element | Per 32-bit register |
  |---|---|---|---|
  | FP8 E4M3 or E5M2 m16n8k32, FP32 accumulate | 2 | 64 | 64 |
  | FP8 E4M3 or E5M2 m16n8k32, FP16 accumulate | 1 | 32 | 64 |
  | int8 m16n8k16, s32 | 1 | 16 | 16 |
  | int8 m16n8k32, s32 | 1 | 32 | 32 |
  | int4 m16n8k64, s32 | 1/2 | 32 | 32 |
  | FP16 m16n8k16, FP32 accumulate | 4 | 64 | 64 |
  | FP16 m16n8k16, FP16 accumulate | 2 | 32 | 64 |
  | BF16 m16n8k16, FP32 accumulate | 4 | 64 | 64 |
  | FP16 m8n8k4 (legacy, S-4), FP32 or FP16 accumulate | 4 or 2 | 16 or 8 | 16 |

* **Semantics.** Each shape computes any function, which is generous to the adversary. Semantics matter only for the
  honest side: the honest atom is `sm89.mma.m16n8k32.e4m3` with FP32 accumulate, `verity.ml.tc`'s
  `ADA_E4M3_M16N8K32` (not formalized here), at 64 per checked FP32 word.
* **The consequence.** The cheapest price per register written in `M4090` stays 16 (int8 m16n8k16, legacy m8n8k4 and
  `int`), and 8 on the FP32 pipe. A dimension bound in `M4090` therefore gives at most 16 per independent target
  word, and that is attained (`Ceiling4090`): on FP8 atom words, whose honest cost is 64, at most 16/64 = 25%.
-/

namespace Pouw.Dimension.Fp8Mma

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Embed

/-- A tensor-core shape of the RTX 4090 (`mma.sync`, `sm_89`) with a measured price (Round 1), or the legacy m8n8k4
(S-4). -/
inductive Shape4090 where
  /-- FP8 E4M3 m16n8k32, FP32 accumulate: the honest atom. -/
  | e4m3F32
  /-- FP8 E5M2 m16n8k32, FP32 accumulate. -/
  | e5m2F32
  /-- FP8 E4M3 m16n8k32, FP16 accumulate. -/
  | e4m3F16
  /-- FP8 E5M2 m16n8k32, FP16 accumulate. -/
  | e5m2F16
  /-- int8 m16n8k16, s32 accumulate. -/
  | s8k16
  /-- int8 m16n8k32, s32 accumulate. -/
  | s8k32
  /-- int4 m16n8k64, s32 accumulate. -/
  | s4k64
  /-- FP16 m16n8k16, FP32 accumulate. -/
  | f16F32
  /-- FP16 m16n8k16, FP16 accumulate. -/
  | f16F16
  /-- BF16 m16n8k16, FP32 accumulate. -/
  | bf16F32
  /-- FP16 m8n8k4 (legacy), FP32 accumulate. -/
  | f16k4F32
  /-- FP16 m8n8k4 (legacy), FP16 accumulate. -/
  | f16k4F16

/-- The shape's `k`: MACs per accumulator element. -/
def Shape4090.k : Shape4090 → ℕ
  | .e4m3F32 | .e5m2F32 | .e4m3F16 | .e5m2F16 | .s8k32 => 32
  | .s8k16 | .f16F32 | .f16F16 | .bf16F32 => 16
  | .s4k64 => 64
  | .f16k4F32 | .f16k4F16 => 4

/-- Accumulator elements per 32-bit register: 2 for FP16 accumulate (`.f16x2`), else 1. -/
def Shape4090.elems : Shape4090 → ℕ
  | .e4m3F16 | .e5m2F16 | .f16F16 | .f16k4F16 => 2
  | _ => 1

/-- Twice the measured price per MAC (Round 1): int4 1; int8 and FP8 with FP16 accumulate 2; FP8 with FP32
accumulate and FP16 with FP16 accumulate 4; FP16 and BF16 with FP32 accumulate 8. -/
def Shape4090.mac2 : Shape4090 → ℕ
  | .s4k64 => 1
  | .s8k16 | .s8k32 | .e4m3F16 | .e5m2F16 => 2
  | .e4m3F32 | .e5m2F32 | .f16F16 | .f16k4F16 => 4
  | .f16F32 | .bf16F32 | .f16k4F32 => 8

/-- The price per 32-bit accumulator register written: `k · elems · (price per MAC)`. -/
def Shape4090.regPrice (s : Shape4090) : ℕ := s.k * s.elems * s.mac2 / 2

/-- The `M4090` instruction of shape `s` writing the accumulator registers `outs` (any functions). -/
def mmaOp {G : Type} (s : Shape4090) (outs : List (G → ℚ)) : Op G := Op.mma outs (s.regPrice * outs.length)

/-! ## Pinned statements (draft semantics, S-1 to S-11 pending Daniel's review) -/

/-- **The shape table** (Round 1): every shape costs at least 16 per register written, int8 m16n8k16 costs exactly
16, and FP8 m16n8k32 with FP32 accumulate costs 64. -/
def ShapeTable4090 : Prop :=
  (∀ s : Shape4090, 16 ≤ s.regPrice) ∧ Shape4090.s8k16.regPrice = 16 ∧ Shape4090.e4m3F32.regPrice = 64 ∧
    Shape4090.e5m2F32.regPrice = 64

/-- **Every shape is an `M4090` instruction**: `mmaOp s outs` is well-formed, so `EmbedM4090` and the per-word
theorems cover programs that use the FP8 classes. -/
def ShapeWF4090 : Prop :=
  ∀ (N m n k : ℕ) (s : Shape4090) (outs : List (Noise N m n k → ℚ)), (mmaOp s outs).WF

/-- **No bound in `M4090` exceeds 16 per target word** (draft semantics, S-1 to S-11 pending Daniel's review): for
every FP32 semantics and every finite set `W` of functions of the noise, a lower bound `b` on the cost of every
`M4090` program that outputs all of `W` satisfies `b ≤ 16·|W|`. On FP8 atom words, whose honest cost is 64 each,
that is at most a quarter: `4·b ≤ 64·|W|`. -/
def Ceiling4090 : Prop :=
  ∀ (N m n k : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (W : Finset (Noise N m n k → ℚ)) (b : ℝ),
    (∀ q : (m4090 N m n k fpSem).Prog, (∀ w ∈ W, w ∈ (m4090 N m n k fpSem).outputs q) →
      b ≤ (m4090 N m n k fpSem).cost q) →
    b ≤ 16 * W.card ∧ 4 * b ≤ Shape4090.e4m3F32.regPrice * W.card

end Pouw.Dimension.Fp8Mma
