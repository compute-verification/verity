module

public import Definitions.Pous.Claims.StreamedWipeServer
public import Proofs.Pous.StreamedWipe.Honest
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Data.Fintype.BigOperators
import Proofs.Pous.Prob

/-! The honest and nontrivial lemmas of the streamed-label wipe's per-server claims. -/

@[expose] public section

namespace Pous.Claims

open Finset Pous Pous.StreamedWipe
open scoped ENNReal

theorem wipeDistinguisherBy_accepts {N k b q M E : ℕ} (P : Prover N k b q M E) :
    wipeDistinguisherBy Accepts P = wipeDistinguisher P := rfl

theorem winRealBy_accepts {N k b q M E κ : ℕ} (prf : Bits κ → Fin N → Bits b) (P : Prover N k b q M E) :
    winRealBy Accepts prf P = winReal prf P := rfl

def keepAll (N k b q M E : ℕ) (h : q ≤ M) : Prover N k b q M E where
  fill z _ := (fun j => if hj : (j : ℕ) < q then z ⟨j, hj⟩ else false, fun _ => false)
  answer _ _ _ _ := fun _ => false
  keep σ _ _ := fun i => σ (Fin.castLE h i)

theorem keepAll_keeps {N k b q M E : ℕ} (h : q ≤ M) (z : Bits q) (L : Fin N → Bits b) (c : Fin k → Fin N) :
    Keeps (keepAll N k b q M E h) z L c := by
  funext i
  simp [keepAll]

theorem pr_of_forall {α : Type} [Fintype α] [Nonempty α] {P : α → Prop} (h : ∀ x, P x) : pr P = 1 := by
  classical
  simp only [Pous.pr_eq_card]
  rw [Finset.filter_true_of_mem fun x _ => h x, Finset.card_univ]
  exact ENNReal.div_self (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

theorem winRealBy_acceptAll_keepAll {N k b q M E κ : ℕ} (hN : 0 < N) (h : q ≤ M) (prf : Bits κ → Fin N → Bits b) :
    winRealBy acceptAll prf (keepAll N k b q M E h) = 1 := by
  have : Nonempty (Fin N) := ⟨⟨0, hN⟩⟩
  exact pr_of_forall fun x => ⟨trivial, keepAll_keeps h _ _ _⟩

theorem keepAll_adv_zero {N k b q M E κ : ℕ} (hN : 0 < N) (h : q ≤ M) (prf : Bits κ → Fin N → Bits b) :
    prfAdvantage prf (wipeDistinguisherBy acceptAll (keepAll N k b q M E h)) = 0 := by
  have : Nonempty (Fin N) := ⟨⟨0, hN⟩⟩
  have hreal := winRealBy_acceptAll_keepAll (E := E) (k := k) hN h prf
  have hideal : pr (wipeDistinguisherBy acceptAll (keepAll N k b q M E h)) = 1 :=
    pr_of_forall fun x => ⟨trivial, keepAll_keeps h _ _ _⟩
  unfold prfAdvantage
  rw [show pr (fun x : (Bits q × (Fin k → Fin N)) × Bits κ =>
      wipeDistinguisherBy acceptAll (keepAll N k b q M E h) (x.1, prf x.2)) = 1 from hreal, hideal, tsub_self]

theorem keepAll_not_within {N k b q M E κ : ℕ} (hN : 0 < N) (h : q ≤ M) (prf : Bits κ → Fin N → Bits b) :
    ¬ winRealBy acceptAll prf (keepAll N k b q M E h) ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisherBy acceptAll (keepAll N k b q M E h)) := by
  rw [winRealBy_acceptAll_keepAll hN h prf, keepAll_adv_zero hN h prf, add_zero, not_le]
  exact pow_lt_one₀ zero_le (ENNReal.inv_lt_one.2 ENNReal.one_lt_two) (by norm_num)

theorem winRealBy_le_of_keep_const (V : Acceptance) {N k b q M E κ : ℕ} (prf : Bits κ → Fin N → Bits b)
    (P : Prover N k b q M E) (w : Bits q) (hw : ∀ σ ξ c, P.keep σ ξ c = w) :
    winRealBy V prf P ≤ (2⁻¹ : ℝ≥0∞) ^ q := by
  classical
  have hcard : (univ.filter fun x : (Bits q × (Fin k → Fin N)) × Bits κ =>
      wipeDistinguisherBy V P (x.1, prf x.2)).card ≤ Fintype.card (Fin k → Fin N) * Fintype.card (Bits κ) :=
    calc _ ≤ (({w} ×ˢ univ) ×ˢ (univ : Finset (Bits κ))).card := card_le_card fun x hx => by
          have hk : P.keep (P.fill x.1.1 (prf x.2)).1 (P.fill x.1.1 (prf x.2)).2 x.1.2 = x.1.1 :=
            (mem_filter.1 hx).2.2
          rw [mem_product, mem_product, mem_singleton]
          exact ⟨⟨hk.symm.trans (hw _ _ _), mem_univ _⟩, mem_univ _⟩
      _ = _ := by rw [card_product, card_product, card_singleton, one_mul, card_univ, card_univ]
  have htot : Fintype.card ((Bits q × (Fin k → Fin N)) × Bits κ) =
      2 ^ q * (Fintype.card (Fin k → Fin N) * Fintype.card (Bits κ)) := by
    simp [Fintype.card_prod, Fintype.card_bool, Fintype.card_fin]
    ring
  have h2 : (2⁻¹ : ℝ≥0∞) ^ q * 2 ^ q = 1 := by
    rw [← ENNReal.inv_pow]
    exact ENNReal.inv_mul_cancel (pow_ne_zero _ two_ne_zero) (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)
  unfold winRealBy; simp only [Pous.pr_eq_card]
  apply ENNReal.div_le_of_le_mul
  rw [htot, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat, ← mul_assoc, h2, one_mul]
  exact_mod_cast hcard

theorem keepNothing_within (V : Acceptance) {N k b q M E κ : ℕ} (hq : 128 ≤ q) (prf : Bits κ → Fin N → Bits b)
    (P : Prover N k b q M E) (w : Bits q) (hw : ∀ σ ξ c, P.keep σ ξ c = w) :
    winRealBy V prf P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisherBy V P) :=
  ((winRealBy_le_of_keep_const V prf P w hw).trans
    (pow_le_pow_of_le_one zero_le (ENNReal.inv_le_one.2 ENNReal.one_lt_two.le) hq)).trans le_self_add

theorem honest_within {N k b q κ : ℕ} (hq : 128 ≤ q) (prf : Bits κ → Fin N → Bits b) :
    winRealBy acceptAll prf (PousStreamedWipe.honest N k b q) ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisherBy acceptAll (PousStreamedWipe.honest N k b q)) :=
  keepNothing_within acceptAll hq prf _ (fun _ => false) fun _ _ _ => rfl

def noKey (N : ℕ) : Bits 0 → Fin N → Bits 512 := fun _ _ _ => false

/-- **`StreamedWipeMeetsReadsTight`'s honest lemma**: its antecedents hold at the deployed H200 as measured
(`h200`): `r = 62 ≥ 1`, `61^5500 · 2^129 ≤ 62^5500`, `n = 2,319,974,400 ≥ 1` labels, and its memory
`M_s = 1,206,038,560,768` bits is `n · 512 + R`. -/
theorem StreamedWipeMeetsReadsTight_meets :
    0 < h200.r ∧ (h200.r - 1) ^ h200.k * 2 ^ 129 ≤ h200.r ^ h200.k ∧ 0 < h200.n ∧
      h200.Ms = h200.n * 512 + h200.R := by
  decide +kernel

/-- **`StreamedWipeMeetsReadsTight`'s nontrivial lemma**: it is false at `acceptAll`, at the deployed H200's numbers,
for `keepAll` keeping `4·10^10` bits. -/
theorem StreamedWipeMeetsReadsTight_fails : ¬ StreamedWipeMeetsReadsTight acceptAll := fun h =>
  keepAll_not_within (by decide +kernel) (by decide +kernel) (noKey h200.n)
    (h h200.r h200.k (by decide +kernel) (by decide +kernel) h200.n (by decide +kernel) h200.R 40000000000 0 0
      (noKey h200.n) (keepAll _ _ _ _ _ _ (by decide +kernel)) (by decide +kernel))

/-- **`StreamedWipeMeetsServer`'s honest lemma**: its antecedents hold at the deployed H200 as measured
(`h200`): `M_s = 1,206,038,560,768 = 512 · 2,319,974,400 + R` bits, `n ≥ 1`, `r = 62 ≥ 1` and
`61^5500 · 2^129 ≤ 62^5500`. -/
theorem StreamedWipeMeetsServer_meets :
    h200.Ms = 512 * h200.n + h200.R ∧ 0 < h200.n ∧ 0 < h200.r ∧
      (h200.r - 1) ^ h200.k * 2 ^ 129 ≤ h200.r ^ h200.k := by
  decide +kernel

/-- **`StreamedWipeMeetsServer`'s nontrivial lemma**: it is false at `acceptAll`, at the deployed H200's numbers, for
`keepAll` keeping `B(n, r, k, R)` bits. -/
theorem StreamedWipeMeetsServer_fails : ¬ StreamedWipeMeetsServer acceptAll := fun h =>
  keepAll_not_within (by decide +kernel) (by decide +kernel) (noKey h200.n)
    (h h200.Ms h200.n h200.R (by decide +kernel) (by decide +kernel) h200.r h200.k (by decide +kernel)
      (by decide +kernel) (serverBound h200.n h200.r h200.k h200.R) 0 0 (noKey h200.n)
      (keepAll _ _ _ _ _ _ (by decide +kernel)) le_rfl)

/-- **`StreamedWipeMeetsServer5GB`'s honest lemma**: `StreamedWipeMeetsServer_meets` and the 5 GB point,
`B(2,319,974,400, 62, 5500, 18,211,667,968) = 37,669,517,430 ≤ 4·10^10`, at the deployed H200 as measured
(`h200`). -/
theorem StreamedWipeMeetsServer5GB_meets :
    h200.Ms = 512 * h200.n + h200.R ∧ 0 < h200.n ∧ 0 < h200.r ∧
      (h200.r - 1) ^ h200.k * 2 ^ 129 ≤ h200.r ^ h200.k ∧ OperatingPoint h200.n h200.r h200.k h200.R := by
  unfold OperatingPoint
  decide +kernel

/-- **`StreamedWipeMeetsServer5GB`'s nontrivial lemma**: it is false at `acceptAll`, at the deployed H200's numbers,
for `keepAll` keeping `4·10^10` bits. -/
theorem StreamedWipeMeetsServer5GB_fails : ¬ StreamedWipeMeetsServer5GB acceptAll := fun h =>
  keepAll_not_within (by decide +kernel) (by decide +kernel) (noKey h200.n)
    (h h200.Ms h200.n h200.R (by decide +kernel) (by decide +kernel) h200.r h200.k (by decide +kernel)
      (by decide +kernel) (by unfold OperatingPoint; decide +kernel) 40000000000 0 0 (noKey h200.n)
      (keepAll _ _ _ _ _ _ (by decide +kernel)) le_rfl)

/-- **`StreamedWipeMeetsServerH200`'s honest lemma**: the premises of the 5 GB form hold at the deployed H200 as
measured (`h200`): its fill of `n = 2,319,974,400` labels and `R = 18,211,667,968` bits (2.28 GB) outside it, at
the deployed `k = 5500` and `r = 62`. -/
theorem StreamedWipeMeetsServerH200_meets :
    h200.Ms = 512 * h200.n + h200.R ∧ 0 < h200.n ∧ 0 < h200.r ∧
      (h200.r - 1) ^ h200.k * 2 ^ 129 ≤ h200.r ^ h200.k ∧ OperatingPoint h200.n h200.r h200.k h200.R := by
  unfold OperatingPoint
  decide +kernel

/-- **`StreamedWipeMeetsServerH200`'s nontrivial lemma**: it is false at `acceptAll`, for `keepAll` keeping `4·10^10`
of the measured card's bits. -/
theorem StreamedWipeMeetsServerH200_fails : ¬ StreamedWipeMeetsServerH200 acceptAll := fun h =>
  keepAll_not_within (by decide +kernel) (by decide +kernel) (noKey h200.n)
    (h 40000000000 0 0 (noKey h200.n) (keepAll _ _ _ _ _ _ (by decide +kernel)) le_rfl)

end Pous.Claims
