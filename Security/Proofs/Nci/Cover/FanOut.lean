import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Finset.Prod
import Proofs.Nci.Cover.FanOutRootCost
import Proofs.Nci.Cover.Additive

/-!
# The fan-out bounds

The whole products of an inference block split by token (`fanOut_split`). A token whose root the block holds spends at
least `d − E` values (`rootCost`, `lookupCost`), and its whole products number at most `c` per updated parameter. Any
other token's are at most `aOut` per value it spends (`charge`). The spent values of distinct tokens are distinct, so
they number at most the block's reads (`readSet_card_le`), or `ValueReadHardness` prices them at `f` bits each beside
the parameters. The rest is arithmetic: linear for `FanOutIntensity`, AM–GM (`four_mul_le_sq`) and the larger endpoint
for `FanOutReadIntensity`. The cover bounds are `cover_sum` on the intensities.
-/

namespace Nci.Cover.SecurityProofs

/-- **The split by token**: an inference block spends values `X`, `x` of them of tokens whose roots it doesn't hold
and `y` of tokens whose roots it holds, `h` tokens of the second kind, with `(d − E) h ≤ y` and
`#whole ≤ aOut · x + c · h · #(updated parameters of whole)`. -/
theorem fanOut_split {C : Circuit} {J L Tok Val : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Val]
    {M : MulBlocks C J L} {d aOut c V f : ℕ} (K : Keyed M Tok Val d aOut c V f) {s k E : ℕ} {bcol : ℝ}
    (hCI : Assumptions.ColumnsIncompressible K s bcol) (hk : (k : ℝ) < bcol * V * (E + 1))
    {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) :
    ∃ X : Finset Val, (∀ u ∈ X, K.Spent I u) ∧ ∃ x y h : ℕ, x + y = X.card ∧ (d - E) * h ≤ y ∧
      (M.whole I).card ≤ aOut * x + c * h * ((M.whole I).image Prod.snd).card := by
  classical
  set W := M.whole I
  set Lw := W.image Prod.snd
  set Th : Finset Tok := (K.roots ∩ K.heldSet I).image K.tok
  have hX : ∀ i : Tok, ∃ Xi : Finset Val, (∀ u ∈ Xi, K.Spent I u ∧ K.tok u = i) ∧ (i ∈ Th → d - E ≤ Xi.card) ∧
      (i ∉ Th → (W.filter fun p => K.tok (K.act p) = i).card ≤ aOut * Xi.card) := by
    intro i
    by_cases hi : i ∈ Th
    · obtain ⟨r, hr, rfl⟩ := Finset.mem_image.1 hi
      obtain ⟨X, Y, hXs, hYs, hd⟩ := rootCost C M d aOut c V f K I r hr
      have hY := lookupCost C M d aOut c V f K s k E bcol hCI hk I hI Y (fun u hu => (hYs u hu).1)
        fun u hu u' hu' => (hYs u hu).2.trans (hYs u' hu').2.symm
      exact ⟨X, hXs, fun _ => by omega, fun h => absurd hi h⟩
    · have hnr : ∀ r ∈ K.roots ∩ K.heldSet I, K.tok r ≠ i := fun r hr hri =>
        hi (hri ▸ Finset.mem_image_of_mem _ hr)
      obtain ⟨X, hXs, hcard⟩ := charge C M d aOut c V f K I i hnr (W.filter fun p => K.tok (K.act p) = i)
        (Finset.filter_subset _ _) fun p hp => (Finset.mem_filter.1 hp).2
      exact ⟨X, hXs, fun h => absurd h hi, fun _ => hcard⟩
  choose Xt hXt using hX
  set Tl : Finset Tok := (W.image fun p => K.tok (K.act p)).filter (· ∉ Th)
  have hTdisj : Disjoint Tl Th := Finset.disjoint_left.2 fun i hi hi' => (Finset.mem_filter.1 hi).2 hi'
  have hXdisj : ∀ T : Finset Tok, (T : Set Tok).PairwiseDisjoint Xt := by
    intro T i _ j _ hij
    refine Finset.disjoint_left.2 fun u hu hu' => hij ?_
    exact ((hXt i).1 u hu).2.symm.trans ((hXt j).1 u hu').2
  refine ⟨(Tl ∪ Th).biUnion Xt, fun u hu => ?_, ∑ i ∈ Tl, (Xt i).card, ∑ i ∈ Th, (Xt i).card, Th.card, ?_, ?_, ?_⟩
  · obtain ⟨i, -, hu⟩ := Finset.mem_biUnion.1 hu
    exact ((hXt i).1 u hu).1
  · rw [Finset.card_biUnion (hXdisj _), Finset.sum_union hTdisj]
  · calc (d - E) * Th.card = ∑ _i ∈ Th, (d - E) := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
      _ ≤ ∑ i ∈ Th, (Xt i).card := Finset.sum_le_sum fun i hi => (hXt i).2.1 hi
  · have hsplit := Finset.card_filter_add_card_filter_not (s := W) fun p => K.tok (K.act p) ∈ Th
    have hlow : (W.filter fun p => ¬ K.tok (K.act p) ∈ Th).card ≤ aOut * ∑ i ∈ Tl, (Xt i).card := by
      have hsub : (W.filter fun p => ¬ K.tok (K.act p) ∈ Th) ⊆
          Tl.biUnion fun i => W.filter fun p => K.tok (K.act p) = i := by
        intro p hp
        obtain ⟨hpW, hpT⟩ := Finset.mem_filter.1 hp
        exact Finset.mem_biUnion.2 ⟨K.tok (K.act p), Finset.mem_filter.2 ⟨Finset.mem_image_of_mem _ hpW, hpT⟩,
          Finset.mem_filter.2 ⟨hpW, rfl⟩⟩
      calc _ ≤ (Tl.biUnion fun i => W.filter fun p => K.tok (K.act p) = i).card := Finset.card_le_card hsub
        _ ≤ ∑ i ∈ Tl, (W.filter fun p => K.tok (K.act p) = i).card := Finset.card_biUnion_le
        _ ≤ ∑ i ∈ Tl, aOut * (Xt i).card := Finset.sum_le_sum fun i hi => (hXt i).2.2 (Finset.mem_filter.1 hi).2
        _ = aOut * ∑ i ∈ Tl, (Xt i).card := (Finset.mul_sum _ _ _).symm
    have hhigh : (W.filter fun p => K.tok (K.act p) ∈ Th).card ≤ c * Th.card * Lw.card := by
      set Wh := W.filter fun p => K.tok (K.act p) ∈ Th
      have h1 := Finset.card_le_mul_card_image (f := fun p => (K.tok (K.act p), p.2)) Wh c fun q _ => by
        refine K.mult_le q.1 q.2 _ (fun p hp => ?_) fun p hp => ?_
        · exact (Finset.mem_filter.1 (Finset.mem_filter.1 (Finset.mem_filter.1 hp).1).1).1
        · have hq := (Finset.mem_filter.1 hp).2
          exact ⟨congrArg Prod.fst hq, congrArg Prod.snd hq⟩
      have h2 : (Wh.image fun p => (K.tok (K.act p), p.2)) ⊆ Th ×ˢ Lw := by
        intro q hq
        obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hq
        exact Finset.mem_product.2 ⟨(Finset.mem_filter.1 hp).2, Finset.mem_image_of_mem _ (Finset.mem_filter.1 hp).1⟩
      calc Wh.card ≤ c * (Wh.image fun p => (K.tok (K.act p), p.2)).card := h1
        _ ≤ c * (Th ×ˢ Lw).card := Nat.mul_le_mul_left _ (Finset.card_le_card h2)
        _ = c * Th.card * Lw.card := by rw [Finset.card_product, mul_assoc]
    rw [← hsplit]
    exact (Nat.add_le_add hhigh hlow).trans_eq (Nat.add_comm _ _)

theorem FanOutIntensity : Guarantees.FanOutIntensity := by
  intro C J L Tok Val _ _ _ M d aOut c V f K s k E b bcol hb hH hCI hk hE I hI
  obtain ⟨X, hXs, x, y, h, hxy, hy, hW⟩ := fanOut_split K hCI hk hI
  have hcap : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  have hXk : (x : ℝ) + y ≤ k := by
    have h1 := (Finset.card_le_card fun u hu => (hXs u hu).1).trans
      ((readSet_card_le C M d aOut c V f K I).trans hI.2)
    rw [← hxy] at h1
    exact_mod_cast h1
  have hL : b * ((M.whole I).image Prod.snd).card ≤ k :=
    (hH I hI.1 (M.whole I) (fun p hp => M.P_subset (Finset.mem_filter.1 hp).1)
      fun p hp => (Finset.mem_filter.1 hp).2).trans hcap
  have hrest : ((M.finished I \ M.whole I).card : ℝ) ≤ k := by
    exact_mod_cast (finished_sdiff_whole_card_le M).trans hI.2
  have hfin : ((M.finished I).card : ℝ) ≤ (M.finished I \ M.whole I).card + (M.whole I).card := by
    exact_mod_cast Finset.card_le_card_sdiff_add_card
  have hW' : ((M.whole I).card : ℝ) ≤ aOut * x + c * h * ((M.whole I).image Prod.snd).card := by
    exact_mod_cast hW
  generalize d - E = δ at hy ⊢
  have hy' : (δ : ℝ) * h ≤ y := by exact_mod_cast hy
  set w : ℝ := (((M.whole I).image Prod.snd).card : ℝ)
  set m := max ((δ : ℝ) * b * aOut) ((c : ℝ) * k)
  have hx0 : (0 : ℝ) ≤ x := Nat.cast_nonneg _
  have hy0 : (0 : ℝ) ≤ y := Nat.cast_nonneg _
  have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg _
  have hδb : 0 ≤ (δ : ℝ) * b := by positivity
  have hm0 : 0 ≤ m := le_trans (by positivity) (le_max_right _ _)
  have e1 : (δ : ℝ) * b * (M.finished I).card ≤ δ * b * (k + (aOut * x + c * h * w)) :=
    mul_le_mul_of_nonneg_left (hfin.trans (add_le_add hrest hW')) hδb
  have e2 : (c : ℝ) * ((δ * h) * (b * w)) ≤ c * (y * k) :=
    mul_le_mul_of_nonneg_left (mul_le_mul hy' hL (by positivity) hy0) hc0
  have e3 : (δ : ℝ) * b * aOut * x ≤ m * x := mul_le_mul_of_nonneg_right (le_max_left _ _) hx0
  have e4 : (c : ℝ) * k * y ≤ m * y := mul_le_mul_of_nonneg_right (le_max_right _ _) hy0
  have e5 : m * (x + y) ≤ m * k := mul_le_mul_of_nonneg_left hXk hm0
  linarith

theorem FanOutCoverBound : Guarantees.FanOutCoverBound := by
  intro C J L Tok Val _ _ _ M d aOut c V f K s k E b bcol 𝓘 hb hH hCI hk hE h𝓘 hcov
  exact cover_sum M hcov (by positivity) fun I hI =>
    FanOutIntensity C M d aOut c V f K s k E b bcol hb hH hCI hk hE I (h𝓘 I hI)

theorem FanOutReadIntensity : Guarantees.FanOutReadIntensity := by
  intro C J L Tok Val _ _ _ M d aOut c V f K s k E b bcol hb hVR hCI hk hE I hI
  obtain ⟨X, hXs, x, y, h, hxy, hy, hW⟩ := fanOut_split K hCI hk hI
  have hcap : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  have hpay := (hVR I hI.1 X hXs).trans hcap
  rw [← hxy, Nat.cast_add] at hpay
  have hrest : ((M.finished I \ M.whole I).card : ℝ) ≤ k := by
    exact_mod_cast (finished_sdiff_whole_card_le M).trans hI.2
  have hfin : ((M.finished I).card : ℝ) ≤ (M.finished I \ M.whole I).card + (M.whole I).card := by
    exact_mod_cast Finset.card_le_card_sdiff_add_card
  have hW' : ((M.whole I).card : ℝ) ≤ aOut * x + c * h * ((M.whole I).image Prod.snd).card := by
    exact_mod_cast hW
  generalize d - E = δ at hy ⊢
  have hy' : (δ : ℝ) * h ≤ y := by exact_mod_cast hy
  set w : ℝ := (((M.whole I).image Prod.snd).card : ℝ)
  set m := max (4 * (δ : ℝ) * b * aOut) ((c : ℝ) * k)
  have hx0 : (0 : ℝ) ≤ x := Nat.cast_nonneg _
  have hy0 : (0 : ℝ) ≤ y := Nat.cast_nonneg _
  have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg _
  have hf0 : (0 : ℝ) ≤ f := Nat.cast_nonneg _
  have hw0 : 0 ≤ w := Nat.cast_nonneg _
  have hbw : 0 ≤ b * w := by positivity
  have hfx : (f : ℝ) * x ≤ k := by nlinarith
  have hcoef : 0 ≤ 4 * (δ : ℝ) * f * b := by positivity
  have e1 : 4 * (δ : ℝ) * f * b * (M.finished I).card ≤ 4 * δ * f * b * (k + (aOut * x + c * h * w)) :=
    mul_le_mul_of_nonneg_left (hfin.trans (add_le_add hrest hW')) hcoef
  have e2 : 4 * (f * y) * (b * w) ≤ (k - f * x) * (k - f * x) :=
    four_mul_le_sq (by positivity) hbw (by linarith)
  have e3 : 4 * (f * (δ * h)) * (b * w) ≤ 4 * (f * y) * (b * w) :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hy' hf0) (by norm_num)) hbw
  have e4 : ((k : ℝ) - f * x) * (k - f * x) ≤ k * (k - f * x) :=
    mul_le_mul_of_nonneg_right (by linarith [mul_nonneg hf0 hx0]) (by linarith)
  have e5 : (c : ℝ) * (4 * (f * (δ * h)) * (b * w)) ≤ c * (k * (k - f * x)) :=
    mul_le_mul_of_nonneg_left (e3.trans (e2.trans e4)) hc0
  have e6 : 4 * (δ : ℝ) * b * aOut * (f * x) ≤ m * (f * x) :=
    mul_le_mul_of_nonneg_right (le_max_left _ _) (mul_nonneg hf0 hx0)
  have e7 : (c : ℝ) * k * (k - f * x) ≤ m * (k - f * x) :=
    mul_le_mul_of_nonneg_right (le_max_right _ _) (by linarith)
  linarith

theorem FanOutReadCoverBound : Guarantees.FanOutReadCoverBound := by
  intro C J L Tok Val _ _ _ M d aOut c V f K s k E b bcol 𝓘 hb hVR hCI hk hE h𝓘 hcov
  exact cover_sum M hcov (by positivity) fun I hI =>
    FanOutReadIntensity C M d aOut c V f K s k E b bcol hb hVR hCI hk hE I (h𝓘 I hI)

end Nci.Cover.SecurityProofs
