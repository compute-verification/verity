module

public import Definitions.Pouw.TileBound.Defs
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.ContinuousFunctionalCalculus
import Mathlib.Tactic.NormNum.GCD

/-! The FP8-tile model at sm_120's E4M3 atom, on the FP4-tile model, where NVFP4's closure rests on the measured hypothesis `fp8-limb-rate/sm120`, not on a format width. -/

@[expose] public section

namespace Pouw.TileBound

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

def e4m3X (m k : ℕ) : XBlocks m k (Fin m × Fin k) := ⟨id, fun _ _ h => by rw [id, id] at h; rw [h]⟩

def e4m3Y (k n : ℕ) : YBlocks k n (Fin k × Fin n) := ⟨id, fun _ _ h => by rw [id, id] at h; rw [h]⟩

/-- `capacity8` counts entries, so the intended blocks are the single entries `e4m3X` and `e4m3Y`; the pins hold for every
partition. -/
def SlotProg.Representable8 [Field K] {β γ : Type*} [Fintype β] [Fintype γ] (P : SlotProg K m k n τ)
    (BX : XBlocks m k β) (BY : YBlocks k n γ) : Prop :=
  ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ capacity8 (P.fmt t) ∧
    BX.touches (P.a' t) + BY.touches (P.b' t) ≤ capacity8 (P.fmt t)

def honest8 (K : Type*) [Field K] (m k n : ℕ) : SlotProg K m k n (Fin m × Fin k × Fin n) where
  a t := fun e => if e = (t.1, t.2.1) then 1 else 0
  b _ := 0
  a' _ := 0
  b' t := fun e => if e = (t.2.1, t.2.2) then 1 else 0
  w t := fun e => if e = (t.1, t.2.2) then 1 else 0
  fmt _ := .fp8

end Pouw.TileBound
