module

public import Definitions.Pouw.Assumptions.Dimension.Embed

/-! The RTX 4090's `mma.sync` shapes, FP8 included, as instances of `M4090Instr.mma` (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.Fp8Mma

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Embed

/-- An `mma.sync` shape of the RTX 4090 (`sm_89`). -/
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

/-- Accumulator elements per 32-bit register. -/
def Shape4090.elems : Shape4090 → ℕ
  | .e4m3F16 | .e5m2F16 | .f16F16 | .f16k4F16 => 2
  | _ => 1

/-- Twice the price per MAC, in units of one int8 MAC. -/
def Shape4090.mac2 : Shape4090 → ℕ
  | .s4k64 => 1
  | .s8k16 | .s8k32 | .e4m3F16 | .e5m2F16 => 2
  | .e4m3F32 | .e5m2F32 | .f16F16 | .f16k4F16 => 4
  | .f16F32 | .bf16F32 | .f16k4F32 => 8

def Shape4090.regPrice (s : Shape4090) : ℕ := s.k * s.elems * s.mac2 / 2

def mmaOp {G : Type} (s : Shape4090) (outs : List (G → ℚ)) : M4090Instr G :=
  M4090Instr.mma outs (s.regPrice * outs.length)

/-! ## Guarantees (draft semantics, pending review) -/

def ShapeTable4090 : Prop :=
  (∀ s : Shape4090, 16 ≤ s.regPrice) ∧ Shape4090.s8k16.regPrice = 16 ∧ Shape4090.e4m3F32.regPrice = 64 ∧
    Shape4090.e5m2F32.regPrice = 64

/-- So `EmbedM4090` and the per-word theorems cover programs that use the FP8 classes. -/
def ShapeWF4090 : Prop :=
  ∀ (N m n k : ℕ) (s : Shape4090) (outs : List (Noise N m n k → ℚ)), (mmaOp s outs).WF

end Pouw.Dimension.Fp8Mma
