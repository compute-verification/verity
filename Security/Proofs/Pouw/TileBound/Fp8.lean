import Proofs.Pouw.TileBound.Proofs
import Definitions.Pouw.TileBound.Fp8Defs

/-!
# The FP8-tile model: the E4M3 count from the FP4-tile core (proofs)

Every bound here comes from `quad_support`, the slice-rank count
that holds for any quadratic program correct on the live entries, for any blocks, over any field. Nothing in it is
format-specific.

* The FP8 bounds are the shared cost lemma `cost_ge_supportAt` (in `Proofs.lean`) at `p₀ = price .fp8`,
  as the FP4 model's `cost_ge_support` is at `p₀ = price .fp4`.
* **What it closes at E4M3:** `cost_ge8`, under `Representable8` and a format with `capacity8 f · price .fp8 ≤ price f`.
  That holds exactly for NVFP4 and E4M3 (`fmt8_closed_iff`). NVFP4's case holds only through the operands (its capacity
  0 is `fp8-limb-rate/sm120`), and E4M3's 1 is the per-slot rule, whose in-domain form holds per fragment, not per pair
  (`fp8-merge-rate/sm120`). Per-slot
  supports reach further (`cost_ge8_support`):
  - BF16 or FP16 pre-adds of up to 2 entries: Strassen depth 1 (`bf16_closed_upto8`);
  - TF32 up to 4: depth 2 (`tf32_closed_upto8`);
  - FFMA up to 8: depth 3 (`fp32_closed_upto8`);
  - int8 at 1 entry (`int8_closed_upto8`), FP16 up to 2 (`fp16_closed_upto8`) and `dp4a` up to 4 (`dp4a_closed_upto8`).

  Each closure is a tie at best.
* **Where it stops:** BF16 or FP16 at 3 entries, TF32 at 5, FFMA at 9, int8 at 2 and `dp4a` at 5 (the `_open_at8`
  theorems). There, `support_admits_openAt` shows that the support count cannot rule out a cheaper cost
  profile. In the FP4 case, BF16
  was closed outright, TF32 up to 8 and FFMA up to 16, because a scaled E2M1 summand is 6 bits wide and an E4M3 summand
  only 4.
* **Tightness:** `honest8_crossOn`, `honest8_cost` and `honest8_representable`. The honest E4M3 program is correct,
  representable and costs `price .fp8 · m·k·n`.
-/

namespace Pouw.TileBound

open Finset

variable {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*}
  [Fintype γ]

/-- **The E4M3 cost bound, per-slot supports.** A slot program correct on the live entries whose slot `t` has both
inputs reading at most `s t` blocks costs at least E4M3's price per live product, provided each slot's price covers
`s t` E4M3 slots. -/
theorem cost_ge8_support (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (s : τ → ℕ)
    (hs : ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ s t ∧ BX.touches (P.a' t) + BY.touches (P.b' t) ≤ s t)
    (hf : ∀ t, (s t : ℚ) * price .fp8 ≤ price (P.fmt t)) :
    price .fp8 * nLive liveX liveY ≤ P.cost :=
  cost_ge_supportAt P.toQuadProg h BX BY s hs (fun t => price (P.fmt t)) (price .fp8) (by norm_num [price]) hf

/-- **The E4M3 cost bound.** Under the representability rule at E4M3, a slot program correct on the live entries costs
at least E4M3's price per live product, provided every format it uses has `capacity8 f · price .fp8 ≤ price f`. -/
theorem cost_ge8 (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (hr : P.Representable8 BX BY)
    (hf : ∀ t, (capacity8 (P.fmt t) : ℚ) * price .fp8 ≤ price (P.fmt t)) :
    price .fp8 * nLive liveX liveY ≤ P.cost :=
  cost_ge8_support P h BX BY (fun t => capacity8 (P.fmt t)) hr hf

/-- **Which formats the E4M3 bound closes:** exactly NVFP4 (capacity 0 on in-domain operands) and E4M3. -/
theorem fmt8_closed_iff (f : Fmt) : (capacity8 f : ℚ) * price .fp8 ≤ price f ↔ f = .fp4 ∨ f = .fp8 := by
  cases f <;> norm_num [capacity8, price]

/-- **BF16 or FP16 pre-adds of up to 2 entries are closed** (a tie): Strassen depth 1. -/
theorem bf16_closed_upto8 : (2 : ℚ) * price .fp8 ≤ price .bf16 := by norm_num [price]

/-- **TF32 pre-adds of up to 4 entries are closed** (a tie): Strassen depth 2. -/
theorem tf32_closed_upto8 : (4 : ℚ) * price .fp8 ≤ price .tf32 := by norm_num [price]

/-- **FFMA pre-adds of up to 8 entries are closed:** Strassen depth 3. -/
theorem fp32_closed_upto8 : (8 : ℚ) * price .fp8 ≤ price .fp32 := by norm_num [price]

/-- BF16 or FP16 at 3 entries: the count admits a cheaper program. -/
theorem bf16_open_at8 : price .bf16 < (3 : ℕ) * price .fp8 := by norm_num [price]

/-- TF32 at 5 entries. -/
theorem tf32_open_at8 : price .tf32 < (5 : ℕ) * price .fp8 := by norm_num [price]

/-- FFMA at 9 entries. -/
theorem fp32_open_at8 : price .fp32 < (9 : ℕ) * price .fp8 := by norm_num [price]

/-- **int8 is closed only at 1 entry** (a tie): at E4M3's price it holds 8 at full width. -/
theorem int8_closed_upto8 : (1 : ℚ) * price .fp8 ≤ price .int8 := by norm_num [price]

/-- int8 at 2 entries. -/
theorem int8_open_at8 : price .int8 < (2 : ℕ) * price .fp8 := by norm_num [price]

/-- **FP16 pre-adds of up to 2 entries are closed** (a tie). -/
theorem fp16_closed_upto8 : (2 : ℚ) * price .fp8 ≤ price .fp16 := by norm_num [price]

/-- FP16 at 3 entries. -/
theorem fp16_open_at8 : price .fp16 < (3 : ℕ) * price .fp8 := by norm_num [price]

/-- **`dp4a` pre-adds of up to 4 entries are closed:** 4 E4M3 slots cost 4, `dp4a` 4.05. -/
theorem dp4a_closed_upto8 : (4 : ℚ) * price .fp8 ≤ price .dp4a := by norm_num [price]

/-- `dp4a` at 5 entries. -/
theorem dp4a_open_at8 : price .dp4a < (5 : ℕ) * price .fp8 := by norm_num [price]

omit [Fintype τ] [Fintype β] [Fintype γ] in
/-- `honest8` has `honest`'s products: only the format differs. -/
theorem honest8_toQuad : (honest8 K m k n).toQuadProg = (honest K m k n).toQuadProg := rfl

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem honest8_crossOn : (honest8 K m k n).toQuadProg.CrossOn (fun _ => True) (fun _ => True) := by
  rw [honest8_toQuad]
  exact honest_crossOn

omit [Fintype τ] [Fintype β] [Fintype γ] in
/-- The honest E4M3 program costs exactly E4M3's price per product: the E4M3 bound is tight. -/
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
