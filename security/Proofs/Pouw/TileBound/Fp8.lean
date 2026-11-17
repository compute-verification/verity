module

public import Definitions.Pouw.TileBound.Fp8Defs
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.ContinuousFunctionalCalculus
import Mathlib.Tactic.NormNum.GCD
import Proofs.Pouw.TileBound.Proofs

/-! The FP8-tile model: the E4M3 cost bound from the FP4-tile support count `quad_support`, which closes NVFP4 and E4M3
and per-slot supports up to the `_closed_upto8` pins. -/

@[expose] public section

namespace Pouw.TileBound

open Finset

variable {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*}
  [Fintype γ]

theorem cost_ge8_support (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (s : τ → ℕ)
    (hs : ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ s t ∧ BX.touches (P.a' t) + BY.touches (P.b' t) ≤ s t)
    (hf : ∀ t, (s t : ℚ) * price .fp8 ≤ price (P.fmt t)) :
    price .fp8 * nLive liveX liveY ≤ P.cost :=
  cost_ge_supportAt P.toQuadProg h BX BY s hs (fun t => price (P.fmt t)) (price .fp8) (by norm_num [price]) hf

theorem cost_ge8 (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (hr : P.Representable8 BX BY)
    (hf : ∀ t, (capacity8 (P.fmt t) : ℚ) * price .fp8 ≤ price (P.fmt t)) :
    price .fp8 * nLive liveX liveY ≤ P.cost :=
  cost_ge8_support P h BX BY (fun t => capacity8 (P.fmt t)) hr hf

/-- NVFP4 closes only through its capacity 0, which holds on in-domain operands. -/
theorem fmt8_closed_iff (f : Fmt) : (capacity8 f : ℚ) * price .fp8 ≤ price f ↔ f = .fp4 ∨ f = .fp8 := by
  cases f <;> norm_num [capacity8, price]

theorem bf16_closed_upto8 : (2 : ℚ) * price .fp8 ≤ price .bf16 := by norm_num [price]

theorem tf32_closed_upto8 : (4 : ℚ) * price .fp8 ≤ price .tf32 := by norm_num [price]

theorem fp32_closed_upto8 : (8 : ℚ) * price .fp8 ≤ price .fp32 := by norm_num [price]

theorem bf16_open_at8 : price .bf16 < (3 : ℕ) * price .fp8 := by norm_num [price]

theorem tf32_open_at8 : price .tf32 < (5 : ℕ) * price .fp8 := by norm_num [price]

theorem fp32_open_at8 : price .fp32 < (9 : ℕ) * price .fp8 := by norm_num [price]

theorem int8_closed_upto8 : (1 : ℚ) * price .fp8 ≤ price .int8 := by norm_num [price]

theorem int8_open_at8 : price .int8 < (2 : ℕ) * price .fp8 := by norm_num [price]

theorem fp16_closed_upto8 : (2 : ℚ) * price .fp8 ≤ price .fp16 := by norm_num [price]

theorem fp16_open_at8 : price .fp16 < (3 : ℕ) * price .fp8 := by norm_num [price]

theorem dp4a_closed_upto8 : (4 : ℚ) * price .fp8 ≤ price .dp4a := by norm_num [price]

theorem dp4a_open_at8 : price .dp4a < (5 : ℕ) * price .fp8 := by norm_num [price]

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem honest8_toQuad : (honest8 K m k n).toQuadProg = (honest K m k n).toQuadProg := rfl

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem honest8_crossOn : (honest8 K m k n).toQuadProg.CrossOn (fun _ => True) (fun _ => True) := by
  rw [honest8_toQuad]
  exact honest_crossOn

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem honest8_cost : (honest8 K m k n).cost = price .fp8 * (m * k * n : ℕ) := by
  simp [SlotProg.cost, honest8, Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin]
  ring

omit [Fintype τ] in
theorem honest8_representable [DecidableEq β] [DecidableEq γ] (BX : XBlocks m k β) (BY : YBlocks k n γ) :
    (honest8 K m k n).Representable8 BX BY := by
  intro t
  have ha : (honest8 K m k n).a t = unitForm (t.1, t.2.1) := rfl
  have hb : (honest8 K m k n).b t = 0 := rfl
  have ha' : (honest8 K m k n).a' t = 0 := rfl
  have hb' : (honest8 K m k n).b' t = unitForm (t.2.1, t.2.2) := rfl
  have hf : (honest8 K m k n).fmt t = .fp8 := rfl
  rw [ha, hb, ha', hb', hf, XBlocks.touches_unit, YBlocks.touches_unit, XBlocks.touches_zero, YBlocks.touches_zero]
  simp [capacity8]

end Pouw.TileBound
