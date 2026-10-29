import Proofs.Core.Game.Basic
import Proofs.Pouw.Lead.Tail.Bennett
import Proofs.Pouw.Lead.Tail.Hoeffding

/-!
# The tail bounds over the coins, with shared coins

The coins `C` are a shared part `R` and blocks `∀ i, X i` (an equivalence `C ≃ R × ∀ i, X i`), and each term
`f r i` reads the shared part and block `i` alone. At each shared value the blocks are a uniform product, so a count
bound that holds at every shared value holds over all the coins (`card_filter_le_of_equiv`); the blocks' means are
bounded at every shared value. `prCoin_sum_ge_le_hoeffding`, `prCoin_sum_ge_le_chernoff`,
`prCoin_sum_ge_le_chernoff_one` and `prCoin_sum_ge_le_bennett` state the bounds as `prCoin` over `C`. With no shared
coins, `R` is `PUnit`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Finset Verity Verity.Game

/-- **Conditioning on shared coins**: if the coins are a shared part and a rest, and at every shared value the event
holds on at most `#Y · B` of the rest, it holds on at most `#Ω · B` of the coins. -/
theorem card_filter_le_of_equiv {Ω R Y : Type*} [Fintype Ω] [Fintype R] [Fintype Y] (e : Ω ≃ R × Y)
    (P : R → Y → Prop) [∀ r, DecidablePred (P r)] {B : ℝ}
    (h : ∀ r, (#{y | P r y} : ℝ) ≤ Fintype.card Y * B) :
    (#{ω : Ω | P (e ω).1 (e ω).2} : ℝ) ≤ Fintype.card Ω * B := by
  have hcard : #{ω : Ω | P (e ω).1 (e ω).2} = #{q : R × Y | P q.1 q.2} := card_equiv e fun ω => by simp
  calc (#{ω : Ω | P (e ω).1 (e ω).2} : ℝ) = #{q : R × Y | P q.1 q.2} := by rw [hcard]
    _ = ∑ r, (#{y | P r y} : ℝ) := by
        rw [card_filter, Nat.cast_sum, Fintype.sum_prod_type]
        simp only [card_filter, Nat.cast_sum]
    _ ≤ ∑ _r : R, (Fintype.card Y * B) := sum_le_sum fun r _ => h r
    _ = Fintype.card Ω * B := by
        rw [sum_const, card_univ, nsmul_eq_mul, Fintype.card_congr e, Fintype.card_prod, Nat.cast_mul]
        ring

/-- A count bound as a probability: if `p` holds on at most `#C · B` coins, `prCoin p ≤ B`. -/
theorem prCoin_le_ofReal {C : Type} [Fintype C] (p : C → Prop) [DecidablePred p] {B : ℝ}
    (h : (#{ω | p ω} : ℝ) ≤ Fintype.card C * B) : prCoin p ≤ ENNReal.ofReal B := by
  rw [Game.prCoin_eq_card]
  rcases isEmpty_or_nonempty C with hC | hC
  · simp
  · have hn : (0 : ℝ) < Fintype.card C := Nat.cast_pos.2 Fintype.card_pos
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_div_of_pos hn]
    exact ENNReal.ofReal_le_ofReal ((div_le_iff₀' hn).2 h)

variable {C R ι : Type} [Fintype C] [Fintype R] [Fintype ι] [DecidableEq ι] {X : ι → Type} [∀ i, Fintype (X i)]

/-- **Hoeffding over the coins**: if each `f r i` reads the shared coins and block `i`, lies in `[0, c i]`, and
averages at most `m i` over its block at every shared value, then `∑ i, f` reaches `∑ i, m i + t` (`t ≥ 0`) with
probability at most `exp (−2t² / ∑ i, c i²)`. -/
theorem prCoin_sum_ge_le_hoeffding (e : C ≃ R × ∀ i, X i) (f : R → ∀ i, X i → ℝ) (c m : ι → ℝ)
    (hf : ∀ r i x, 0 ≤ f r i x ∧ f r i x ≤ c i) (hm : ∀ r i, (∑ x, f r i x) / Fintype.card (X i) ≤ m i)
    {t : ℝ} (ht : 0 ≤ t) :
    prCoin (fun ω => ∑ i, m i + t ≤ ∑ i, f (e ω).1 i ((e ω).2 i)) ≤
      ENNReal.ofReal (Real.exp (-2 * t ^ 2 / ∑ i, c i ^ 2)) :=
  prCoin_le_ofReal _ (card_filter_le_of_equiv e (fun r y => ∑ i, m i + t ≤ ∑ i, f r i (y i))
    fun r => card_sum_ge_le_hoeffding_of_mean_le (f r) c m (hf r) (hm r) ht)

/-- **Chernoff over the coins**: if each `f r i` reads the shared coins and block `i`, lies in `[0, c]`, and averages
at most `m i` over its block at every shared value, then for `l ≥ 0`, `∑ i, f` reaches `a` with probability at most
`exp (−l·a + (e^{l c} − 1)/c · ∑ i, m i)`. -/
theorem prCoin_sum_ge_le_chernoff (e : C ≃ R × ∀ i, X i) (f : R → ∀ i, X i → ℝ) {c : ℝ} (m : ι → ℝ)
    (hf : ∀ r i x, 0 ≤ f r i x ∧ f r i x ≤ c) (hm : ∀ r i, (∑ x, f r i x) / Fintype.card (X i) ≤ m i)
    {l : ℝ} (hl : 0 ≤ l) (a : ℝ) :
    prCoin (fun ω => a ≤ ∑ i, f (e ω).1 i ((e ω).2 i)) ≤
      ENNReal.ofReal (Real.exp (-(l * a) + (Real.exp (l * c) - 1) / c * ∑ i, m i)) :=
  prCoin_le_ofReal _ (card_filter_le_of_equiv e (fun r y => a ≤ ∑ i, f r i (y i))
    fun r => card_sum_ge_le_chernoff (f r) m (hf r) (hm r) hl a)

/-- **Chernoff over the coins at `l = 1/c`**: `∑ i, f` reaches `(e − 1) · ∑ i, m i + s` with probability at most
`exp (−s/c)`; the slack for a deviation probability `η` is `s = c · ln (1/η)`. -/
theorem prCoin_sum_ge_le_chernoff_one (e : C ≃ R × ∀ i, X i) (f : R → ∀ i, X i → ℝ) {c : ℝ} (hc : 0 < c)
    (m : ι → ℝ) (hf : ∀ r i x, 0 ≤ f r i x ∧ f r i x ≤ c)
    (hm : ∀ r i, (∑ x, f r i x) / Fintype.card (X i) ≤ m i) (s : ℝ) :
    prCoin (fun ω => (Real.exp 1 - 1) * ∑ i, m i + s ≤ ∑ i, f (e ω).1 i ((e ω).2 i)) ≤
      ENNReal.ofReal (Real.exp (-(s / c))) :=
  prCoin_le_ofReal _ (card_filter_le_of_equiv e (fun r y => (Real.exp 1 - 1) * ∑ i, m i + s ≤ ∑ i, f r i (y i))
    fun r => card_sum_ge_le_chernoff_one (f r) hc m (hf r) (hm r) s)

/-- **Bennett over the coins**: for `M = ∑ i, m i > 0` and `s ≥ 0`, `∑ i, f` reaches `M + s` with probability at
most `exp (−(M/c) · ((1 + s/M) · ln (1 + s/M) − s/M))`. -/
theorem prCoin_sum_ge_le_bennett (e : C ≃ R × ∀ i, X i) (f : R → ∀ i, X i → ℝ) {c : ℝ} (hc : 0 < c)
    (m : ι → ℝ) (hf : ∀ r i x, 0 ≤ f r i x ∧ f r i x ≤ c)
    (hm : ∀ r i, (∑ x, f r i x) / Fintype.card (X i) ≤ m i) (hM : 0 < ∑ i, m i) {s : ℝ} (hs : 0 ≤ s) :
    prCoin (fun ω => ∑ i, m i + s ≤ ∑ i, f (e ω).1 i ((e ω).2 i)) ≤
      ENNReal.ofReal (Real.exp (-((∑ i, m i) / c *
        ((1 + s / ∑ i, m i) * Real.log (1 + s / ∑ i, m i) - s / ∑ i, m i)))) :=
  prCoin_le_ofReal _ (card_filter_le_of_equiv e (fun r y => ∑ i, m i + s ≤ ∑ i, f r i (y i))
    fun r => card_sum_ge_le_bennett (f r) hc m (hf r) (hm r) hM hs)

end Pouw.SecurityProofs.Lead.Tail
