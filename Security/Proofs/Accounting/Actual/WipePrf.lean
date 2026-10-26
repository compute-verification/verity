import Mathlib
import Definitions.Accounting.Guarantees.Actual.WipePrf
import Proofs.Accounting.Actual.WipeBridge

/-!
# The store held read by read, on the deployed labels

`Accounting.Guarantees.Actual.WipeReadHeldPrf`. A uniform lifetime table meets `UniformLabels` given any coins of the
rest (`uniformLabels_table`): at any read, the tables that take given values on a set `A` of the read's wipe's blocks
(its `slots`) are a `2^{-|A| b}` share (`card_tables_le`), since a label's value fixes it (`bitsVal_injective`) and
distinct blocks are distinct slots. So `WipeReadHeld` bounds the event's count
on the system's coins `((ω, F), c)`, which is its probability with the table drawn last (`pr_reshuffle_le`), and the
deployed probability, at `F = prf s`, is that plus the PRF's advantage against the event read off one run
(`pr_le_add_prfAdvantage`), as in `PousStreamedWipe.streamed_wipe`.
-/

namespace Accounting.SecurityProofs.Actual

open Finset
open scoped ENNReal

/-- A label's value fixes the label. -/
theorem bitsVal_injective (w : ℕ) : Function.Injective (Pous.bitsVal (w := w)) :=
  fun _ _ h => (Pous.bitsEquiv w).injective (Fin.ext h)

/-- **Wipe `w`'s slots** in the lifetime table: block `i` at `finProdFinEquiv (w, i)`. -/
def slots {W N : ℕ} (w : Fin W) : Fin N ↪ Fin (W * N) :=
  ⟨fun i => finProdFinEquiv (w, i), fun _ _ h => congrArg Prod.snd (finProdFinEquiv.injective h)⟩

theorem slots_apply {W N : ℕ} (w : Fin W) (i : Fin N) : slots w i = finProdFinEquiv (w, i) := rfl

open Classical in
/-- **The tables that take given values on `A`** at the slots `e` are at most a `2^{-|A| b}` share: a table is fixed by
its labels off `e`'s image of `A`. -/
theorem card_tables_le {M N b : ℕ} (e : Fin N ↪ Fin M) (A : Finset (Fin N)) (v : Fin N → ℕ) :
    (univ.filter fun F : Fin M → Pous.Bits b => ∀ i ∈ A, Pous.bitsVal (F (e i)) = v i).card * 2 ^ (A.card * b) ≤
      Fintype.card (Fin M → Pous.Bits b) := by
  have hinj : Set.InjOn (fun (F : Fin M → Pous.Bits b) (k : ↥((A.map e)ᶜ)) => F k.1)
      ↑(univ.filter fun F : Fin M → Pous.Bits b => ∀ i ∈ A, Pous.bitsVal (F (e i)) = v i) := by
    intro F hF G hG hFG
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hF hG
    funext k
    by_cases hk : k ∈ A.map e
    · obtain ⟨i, hi, rfl⟩ := mem_map.1 hk
      exact bitsVal_injective b ((hF i hi).trans (hG i hi).symm)
    · exact congrFun hFG ⟨k, mem_compl.2 hk⟩
  have h1 := card_le_card_of_injOn _ (fun F _ => mem_univ _) hinj
  rw [card_univ, Fintype.card_fun, Fintype.card_coe, card_compl, card_map, Fintype.card_fun, Fintype.card_bool,
    Fintype.card_fin, Fintype.card_fin] at h1
  rw [Fintype.card_fun, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Fintype.card_fin]
  have hA : A.card ≤ M := by simpa using card_le_univ (A.map e)
  calc _ ≤ (2 ^ b) ^ (M - A.card) * 2 ^ (A.card * b) := Nat.mul_le_mul_right _ h1
    _ = (2 ^ b) ^ M := by
      rw [mul_comm A.card b, pow_mul, ← pow_add, Nat.sub_add_cancel hA]

open Classical in
/-- **A uniform table's labels are uniform** given any coins `ζ` of the rest of the system: `UniformLabels` holds of
`Actual.tableLabel wipe` at `ζ` read off `ω`, for any reads' wipes `wipe`. -/
theorem uniformLabels_table {Ω Z : Type*} [Fintype Ω] {n W N b : ℕ} (wipe : Fin n → Fin W) (ζ : Ω → Z) :
    Assumptions.UniformLabels (Actual.tableLabel (Ω := Ω) (N := N) (b := b) wipe) b (fun ω₀ => ζ ω₀.1) := by
  intro j A v z
  have hprod : (univ.filter fun ω₀ : Ω × (Fin (W * N) → Pous.Bits b) =>
      ζ ω₀.1 = z ∧ ∀ i ∈ A, Actual.tableLabel wipe ω₀ j i = v i) =
        (univ.filter fun ω => ζ ω = z) ×ˢ
          (univ.filter fun F : Fin (W * N) → Pous.Bits b =>
            ∀ i ∈ A, Pous.bitsVal (F (slots (wipe j) i)) = v i) := by
    ext ⟨ω, F⟩
    rw [mem_filter, mem_product, mem_filter, mem_filter]
    simp [Actual.tableLabel, slots_apply]
  have hprod' : (univ.filter fun ω₀ : Ω × (Fin (W * N) → Pous.Bits b) => ζ ω₀.1 = z) =
      (univ.filter fun ω => ζ ω = z) ×ˢ (univ : Finset (Fin (W * N) → Pous.Bits b)) := by
    ext ⟨ω, F⟩
    simp
  have key : ((univ.filter fun ω₀ : Ω × (Fin (W * N) → Pous.Bits b) =>
      ζ ω₀.1 = z ∧ ∀ i ∈ A, Actual.tableLabel wipe ω₀ j i = v i).card : ℝ) * 2 ^ (A.card * b) ≤
        (univ.filter fun ω₀ : Ω × (Fin (W * N) → Pous.Bits b) => ζ ω₀.1 = z).card := by
    rw [hprod, hprod', card_product, card_product, card_univ]
    have h := card_tables_le (b := b) (slots (wipe j)) A v
    rw [Nat.cast_mul, mul_assoc]
    exact_mod_cast Nat.mul_le_mul_left _ h
  convert key using 4
  ext x
  simp

/-- The system's coins `((ω, F), c)` with the table drawn last, `((ω, c), F)`, as a distinguisher reads them. -/
def reshuffle (Ω I F : Type*) : (Ω × I) × F ≃ (Ω × F) × I where
  toFun x := ((x.1.1, x.2), x.1.2)
  invFun r := ((r.1.1, r.2), r.1.2)
  left_inv _ := rfl
  right_inv _ := rfl

open Classical in
/-- **An event's count on the system's coins is its probability with the table drawn last**: at most `β` of the coins
`((ω, F), c)` makes the distinguisher reading `((ω, c), F)` output true with probability at most `β`. -/
theorem pr_reshuffle_le {Ω I F : Type} [Fintype Ω] [Fintype I] [Fintype F] (E : (Ω × F) × I → Prop) {β : ℝ}
    (hβ : 0 ≤ β) (h : ((univ.filter E).card : ℝ) ≤ β * Fintype.card ((Ω × F) × I)) :
    Pous.pr (fun x : (Ω × I) × F => E ((x.1.1, x.2), x.1.2)) ≤ ENNReal.ofReal β := by
  unfold Pous.pr
  have hcard : (univ.filter fun x : (Ω × I) × F => E ((x.1.1, x.2), x.1.2)).card = (univ.filter E).card :=
    card_equiv (reshuffle Ω I F) fun x => by simp [reshuffle]
  have hU : Fintype.card ((Ω × I) × F) = Fintype.card ((Ω × F) × I) := Fintype.card_congr (reshuffle Ω I F)
  rw [hcard, hU]
  apply ENNReal.div_le_of_le_mul
  rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul hβ]
  exact ENNReal.ofReal_le_ofReal h

/-- **The real game is the ideal one plus the PRF's advantage** against the distinguisher, for any bound on the ideal
one. -/
theorem pr_le_add_prfAdvantage {N b κ : ℕ} {ω : Type} [Fintype ω] (prf : Pous.Bits κ → Fin N → Pous.Bits b)
    (D : ω × (Fin N → Pous.Bits b) → Prop) {β : ℝ≥0∞} (h : Pous.pr D ≤ β) :
    Pous.pr (fun x : ω × Pous.Bits κ => D (x.1, prf x.2)) ≤ β + Pous.StreamedWipe.prfAdvantage prf D :=
  le_add_tsub.trans (add_le_add h le_rfl)

open Classical in
theorem WipeReadHeldPrf : Guarantees.Actual.WipeReadHeldPrf := by
  intro Ω L Z _ _ n W N b κ prf wipe S ℓ X Y V Ch D ℓmax a m K arrive ζ quiet T hS hlat hlive hlv hq hlate
  refine pr_le_add_prfAdvantage prf (Actual.heldDistinguisher S X V Ch arrive T m K) ?_
  refine pr_reshuffle_le (Actual.heldEvent S X V Ch arrive T m K) (by positivity) ?_
  have h := WipeReadHeld S ℓ X Y V Ch (Actual.tableLabel wipe) b D ℓmax a m K arrive (fun ω₀ => ζ ω₀.1) quiet T hS
    hlat hlive hlv hq hlate (uniformLabels_table wipe ζ)
  refine le_trans (le_of_eq ?_) (h.trans (le_of_eq ?_))
  · congr 1
    exact card_equiv (Equiv.refl _) fun r => by
      simp only [mem_filter, mem_univ, true_and, Actual.heldEvent, Equiv.refl_apply]
  · simp only [Fintype.card_fin]

end Accounting.SecurityProofs.Actual
