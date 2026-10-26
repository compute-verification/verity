import Mathlib
import Definitions.Accounting.Guarantees.Actual.WipeBridge
import Proofs.Accounting.Actual.Confidence
import Proofs.Accounting.Actual.WipeReadPool

/-!
# What an answered block costs, and the store held read by read

`Accounting.Guarantees.Actual.AnsweredOfQuiet` and `WipeReadHeld`.

The binding (`label_eq_of_late`): under a latency ceiling of `ℓ⁺` from `X` to `Y` and causality, an answer the
challenger sees at `Y` during `[t + ℓ⁺, t + D)` left `X` during `[t, t + D)` (`crossing_subset_sent`), so it is among
what `X` sent then; two runs alike in what `X` sent then, each accepted, are accepted on the same label
(`verdictBinds_of_lateAnswers`). With live reads and quiet reads, two coins whose read `j` arrives at the same tick and
finds the same content, with the same devices' coins `ζ`, are alike in what crossed into `X` during the deadline for
any index both accept, so in what `X` did and sent then (`CausalFrom`), so in that index's label
(`label_eq_of_answered`).

The count (`AnsweredOfQuiet`): the coins of the event, read `j` arriving at `t ∈ T` with content `κ ∈ C` and `a`
answered blocks `A`, fall in the fibres of `(t, κ, ζ ω₀, A)` (`fibre`). On a fibre the labels of `A` are those of any one
of its coins, so by `UniformLabels` it has at most `2^{-a b}` of the coins with that `ζ` (`fibre_card_le`); summing over
`T`, `C`, the values of `ζ` and the `C(|B|, a)` sets `A` gives `|T| |C| C(|B|, a) 2^{-a b}` of all the coins.

The guarantee (`WipeReadHeld`): a read whose state is narrower than `m` bits has a content among the fewer than `2^m`
strings of fewer than `m` bits (`content_length_le`), so outside `AnsweredOfQuiet`'s event for those contents, at most
`|T| 2^m C(|B|, a) 2^{-a b}` of the coins a read, it answers fewer than `a` blocks and its un-answered fraction is at
least `1 − (a − 1)/|B|`. `K` such reads make the fractions sum to at least `K (1 − (a − 1)/|B|)`, which
`WipeReadPool` bounds by `exp(−K (1 − (a − 1)/|B|))`.
-/

namespace Accounting.SecurityProofs.Actual

open Finset

section Binding

variable {L : Type*}

/-- **A late answer left `X` during the deadline**: under causality and a latency ceiling of `ℓ⁺` from `X` to `Y`, apart
from `X`, what crossed from `X` into `Y` during `[t + ℓ⁺, t + D)` is among what `X` sent during `[t, t + D)`. -/
theorem crossing_subset_sent {R : Actual.Run L} {ℓ : L → L → ℕ} (hc : R.A.Causal ℓ) {X Y : Finset L}
    {ℓmax D t : ℕ} (hlat : R.A.LatencyAtMost X Y ℓmax) (hXY : Disjoint X Y) :
    R.crossing ↑X ↑Y (Set.Ico (t + ℓmax) (t + D)) ⊆ R.sent ↑X (Set.Ico t (t + D)) := by
  rintro _ ⟨g, a, hr, ha, hg, ht, rfl⟩
  obtain ⟨ht1, ht2⟩ := ht
  have h1 := hc g a hr
  have h2 := hlat g a hr ha hg
  exact ⟨g, a, hr, ha, fun h => Finset.disjoint_left.1 hXY h hg, Set.mem_Ico.2 ⟨by omega, by omega⟩, rfl⟩

/-- **One label a window**: if `R`'s late answers are not empty and each `v`, `R'`'s are each `v'`, and `X` sent the same
during the deadline in both, then `v = v'`. -/
theorem label_eq_of_late {R R' : Actual.Run L} {ℓ : L → L → ℕ} (hc : R.A.Causal ℓ) {X Y : Finset L}
    {ℓmax D t v v' : ℕ} (hlat : R.A.LatencyAtMost X Y ℓmax) (hXY : Disjoint X Y)
    (hr : (R.crossing ↑X ↑Y (Set.Ico (t + ℓmax) (t + D))).Nonempty ∧
      ∀ p ∈ R.crossing ↑X ↑Y (Set.Ico (t + ℓmax) (t + D)), p.2.2.2.2.2.2 = v)
    (hr' : ∀ p ∈ R'.crossing ↑X ↑Y (Set.Ico (t + ℓmax) (t + D)), p.2.2.2.2.2.2 = v')
    (hsent : R.sent ↑X (Set.Ico t (t + D)) = R'.sent ↑X (Set.Ico t (t + D))) : v = v' := by
  obtain ⟨p, hp⟩ := hr.1
  have hp' : p ∈ R'.crossing ↑X ↑Y (Set.Ico (t + ℓmax) (t + D)) := by
    have hs : p ∈ R'.sent ↑X (Set.Ico t (t + D)) := by
      rw [← hsent]
      exact crossing_subset_sent hc hlat hXY hp
    obtain ⟨g, a, hr, ha, -, -, rfl⟩ := hs
    obtain ⟨g', a', -, -, hg', ht', he⟩ := hp
    have h1 : R'.A.t g = R.A.t g' := congrArg Prod.fst he
    have h2 : R'.A.x g = R.A.x g' := congrArg (fun q => q.2.1) he
    exact ⟨g, a, hr, ha, by rw [h2]; exact hg', by rw [h1]; exact ht', rfl⟩
  exact (hr.2 p hp).symm.trans (hr' p hp')

variable {Ω₀ B : Type*} {n : ℕ}

/-- **The verdict binds one label to what `X` sent** (red team 3's `VerdictBinds`, now a consequence): under causality, a
latency ceiling from `X` to `Y` and the late-answer rule, two coins on which read `j` is accepted at the same arrival
tick, with what `X` sent during the deadline alike, are accepted on the same label. -/
theorem verdictBinds_of_lateAnswers {S : Ω₀ × (Fin n → B) → Actual.Run L} {ℓ : L → L → ℕ} {X Y : Finset L}
    {lab : Ω₀ → Fin n → B → ℕ} {arrive : Ω₀ × (Fin n → B) → Fin n → ℕ} {ℓmax D : ℕ}
    {acc : Ω₀ × (Fin n → B) → Fin n → Prop} (hc : ∀ r, (S r).A.Causal ℓ)
    (hlat : ∀ r, (S r).A.LatencyAtMost X Y ℓmax) (hlate : Assumptions.LateAnswers S X Y lab arrive ℓmax D acc)
    (j : Fin n) (r r' : Ω₀ × (Fin n → B)) (hr : acc r j) (hr' : acc r' j) (ht : arrive r j = arrive r' j)
    (hsent : (S r).sent ↑X (Set.Ico (arrive r j) (arrive r j + D)) =
      (S r').sent ↑X (Set.Ico (arrive r' j) (arrive r' j + D))) :
    lab r.1 j (r.2 j) = lab r'.1 j (r'.2 j) := by
  have h' := (hlate.2 j r' hr').2
  rw [← ht] at h' hsent
  exact label_eq_of_late (hc r) (hlat r) hlate.1 (hlate.2 j r hr) h' hsent

end Binding

variable {Ω₀ B L Z : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
  {S : Ω₀ × (Fin n → B) → Actual.Run L} {ℓ : L → L → ℕ} {X Y : Finset L} {lab : Ω₀ → Fin n → B → ℕ}
  {b D ℓmax : ℕ} {arrive : Ω₀ × (Fin n → B) → Fin n → ℕ} {acc : Ω₀ × (Fin n → B) → Fin n → Prop} {ζ : Ω₀ → Z}
  {quiet : Fin n → ℕ → B → Z → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)}

omit [Fintype Ω₀] in
theorem mem_answered {j : Fin n} {r : Ω₀ × (Fin n → B)} {i : B} :
    i ∈ Actual.answered acc j r ↔ acc (r.1, Function.update r.2 j i) j := by
  simp only [Actual.answered, mem_filter, mem_univ, true_and]

omit [Fintype Ω₀] [Fintype B] in
/-- **The label of an accepted index is fixed by the window**: two coins whose read `j` is accepted with the same
index, arrival tick, content and devices' coins have the same label there. -/
theorem label_eq_of_window (hS : CausalStrategy S ℓ) (hlat : ∀ r, (S r).A.LatencyAtMost X Y ℓmax)
    (hq : Assumptions.QuietReads S X arrive D acc ζ quiet)
    (hlate : Assumptions.LateAnswers S X Y lab arrive ℓmax D acc) {j : Fin n} {q q' : Ω₀ × (Fin n → B)}
    (ha : acc q j) (ha' : acc q' j) (ht : arrive q j = arrive q' j)
    (hκ : (S q).content X (arrive q j) = (S q').content X (arrive q' j)) (hz : ζ q.1 = ζ q'.1)
    (hi : q.2 j = q'.2 j) : lab q.1 j (q.2 j) = lab q'.1 j (q'.2 j) := by
  have hcr := hq j q ha
  have hcr' := hq j q' ha'
  rw [← ht, ← hi, ← hz, ← hcr] at hcr'
  rw [← ht] at hκ
  have hacts := hS.1 X (arrive q j) (arrive q j + D) q q' hκ hcr'.symm
  refine verdictBinds_of_lateAnswers hS.2 hlat hlate j q q' ha ha' ht ?_
  rw [← ht]
  exact congrArg (fun p => p.2.1) hacts

omit [Fintype Ω₀] in
/-- **An answered block's label is fixed before its index**: two coins whose read `j` arrives at the same tick and
finds the same content, with the same devices' coins, have the same label at every block both would accept. -/
theorem label_eq_of_answered (hS : CausalStrategy S ℓ) (hlat : ∀ r, (S r).A.LatencyAtMost X Y ℓmax)
    (hlive : Assumptions.LiveReads S X arrive) (hq : Assumptions.QuietReads S X arrive D acc ζ quiet)
    (hlate : Assumptions.LateAnswers S X Y lab arrive ℓmax D acc) {j : Fin n} {r r' : Ω₀ × (Fin n → B)} {i : B}
    (hi : i ∈ Actual.answered acc j r) (hi' : i ∈ Actual.answered acc j r') (ht : arrive r j = arrive r' j)
    (hκ : (S r).content X (arrive r j) = (S r').content X (arrive r' j)) (hz : ζ r.1 = ζ r'.1) :
    lab r.1 j i = lab r'.1 j i := by
  have hagree : ∀ (q : Ω₀ × (Fin n → B)) (k : Fin n), k < j → Function.update q.2 j i k = q.2 k :=
    fun q k hk => Function.update_of_ne hk.ne _ _
  obtain ⟨ht1, hκ1⟩ := hlive j (r.1, Function.update r.2 j i) r rfl (hagree r)
  obtain ⟨ht2, hκ2⟩ := hlive j (r'.1, Function.update r'.2 j i) r' rfl (hagree r')
  have := label_eq_of_window hS hlat hq hlate (mem_answered.1 hi) (mem_answered.1 hi')
    (ht1.trans (ht.trans ht2.symm)) (hκ1.trans (hκ.trans hκ2.symm)) hz (by simp)
  simpa using this

open Classical in
/-- **A fibre**: the coins whose read `j` arrives at `q.1`, finds `X`'s content `q.2.1` there, has the devices' coins
`q.2.2.1`, and would accept every block of `q.2.2.2`. -/
noncomputable def fibre (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (ζ : Ω₀ → Z) (j : Fin n)
    (q : ℕ × List Bool × Z × Finset B) : Finset (Ω₀ × (Fin n → B)) :=
  univ.filter fun r => arrive r j = q.1 ∧ (S r).content X (arrive r j) = q.2.1 ∧ ζ r.1 = q.2.2.1 ∧
    q.2.2.2 ⊆ Actual.answered acc j r

theorem mem_fibre {j : Fin n} {q : ℕ × List Bool × Z × Finset B} {r : Ω₀ × (Fin n → B)} :
    r ∈ fibre S X arrive acc ζ j q ↔ arrive r j = q.1 ∧ (S r).content X (arrive r j) = q.2.1 ∧ ζ r.1 = q.2.2.1 ∧
      q.2.2.2 ⊆ Actual.answered acc j r := by
  simp only [fibre, mem_filter, mem_univ, true_and]

open Classical in
/-- **A fibre has at most `2^{-|A| b}` of the coins with its `ζ`**: its coins agree on the labels of `A`. -/
theorem fibre_card_le (hS : CausalStrategy S ℓ) (hlat : ∀ r, (S r).A.LatencyAtMost X Y ℓmax)
    (hlive : Assumptions.LiveReads S X arrive) (hq : Assumptions.QuietReads S X arrive D acc ζ quiet)
    (hlate : Assumptions.LateAnswers S X Y lab arrive ℓmax D acc) (hu : Assumptions.UniformLabels lab b ζ)
    (j : Fin n) (q : ℕ × List Bool × Z × Finset B) :
    ((fibre S X arrive acc ζ j q).card : ℝ) * 2 ^ (q.2.2.2.card * b) ≤
      (Fintype.card (Fin n → B) : ℝ) * (univ.filter fun ω => ζ ω = q.2.2.1).card := by
  obtain ⟨t, κ, z, A⟩ := q
  rcases (fibre S X arrive acc ζ j (t, κ, z, A)).eq_empty_or_nonempty with he | ⟨r0, hr0⟩
  · rw [he, card_empty, Nat.cast_zero, zero_mul]
    positivity
  rw [mem_fibre] at hr0
  have hsub : fibre S X arrive acc ζ j (t, κ, z, A) ⊆
      (univ.filter fun ω => ζ ω = z ∧ ∀ i ∈ A, lab ω j i = lab r0.1 j i) ×ˢ (univ : Finset (Fin n → B)) := by
    intro r hr
    rw [mem_fibre] at hr
    refine mem_product.2 ⟨mem_filter.2 ⟨mem_univ _, hr.2.2.1, fun i hi => ?_⟩, mem_univ _⟩
    exact label_eq_of_answered hS hlat hlive hq hlate (hr.2.2.2 hi) (hr0.2.2.2 hi) (hr.1.trans hr0.1.symm)
      (hr.2.1.trans hr0.2.1.symm) (hr.2.2.1.trans hr0.2.2.1.symm)
  have h1 := hu j A (lab r0.1 j) z
  have h2 : ((fibre S X arrive acc ζ j (t, κ, z, A)).card : ℝ) ≤
      (Fintype.card (Fin n → B) : ℝ) *
        (univ.filter fun ω => ζ ω = z ∧ ∀ i ∈ A, lab ω j i = lab r0.1 j i).card := by
    have := card_le_card hsub
    rw [card_product, card_univ] at this
    rw [mul_comm]
    exact_mod_cast this
  calc ((fibre S X arrive acc ζ j (t, κ, z, A)).card : ℝ) * 2 ^ (A.card * b)
      ≤ (Fintype.card (Fin n → B) : ℝ) *
          (((univ.filter fun ω => ζ ω = z ∧ ∀ i ∈ A, lab ω j i = lab r0.1 j i).card : ℝ) * 2 ^ (A.card * b)) := by
        rw [← mul_assoc]
        exact mul_le_mul_of_nonneg_right h2 (by positivity)
    _ ≤ (Fintype.card (Fin n → B) : ℝ) * (univ.filter fun ω => ζ ω = z).card := by
        convert mul_le_mul_of_nonneg_left h1 (Nat.cast_nonneg (α := ℝ) (Fintype.card (Fin n → B)))

theorem sum_product4 {α β γ δ : Type*} (T : Finset α) (C : Finset β) (I : Finset γ) (W : Finset δ) (f : γ → ℕ) :
    ∑ q ∈ T ×ˢ C ×ˢ I ×ˢ W, f q.2.2.1 = T.card * C.card * W.card * ∑ z ∈ I, f z := by
  simp only [sum_product, sum_const, smul_eq_mul, ← mul_sum]
  ring

open Classical in
theorem AnsweredOfQuiet : Guarantees.Actual.AnsweredOfQuiet := by
  intro Ω₀ B L Z _ _ _ n S ℓ X Y lab b D ℓmax a arrive acc ζ quiet T C j hS hlat hlive hq hlate hu
  have hcover : (univ.filter fun r => arrive r j ∈ T ∧ (S r).content X (arrive r j) ∈ C ∧
      a ≤ (Actual.answered acc j r).card) ⊆
        (T ×ˢ C ×ˢ univ.image ζ ×ˢ powersetCard a (univ : Finset B)).biUnion (fibre S X arrive acc ζ j) := by
    intro r hr
    obtain ⟨hT, hC, ha⟩ := (mem_filter.1 hr).2
    obtain ⟨A, hA, hcard⟩ := exists_subset_card_eq ha
    refine mem_biUnion.2 ⟨(arrive r j, (S r).content X (arrive r j), ζ r.1, A), ?_, ?_⟩
    · exact mem_product.2 ⟨hT, mem_product.2 ⟨hC, mem_product.2 ⟨mem_image_of_mem ζ (mem_univ _),
        mem_powersetCard.2 ⟨subset_univ _, hcard⟩⟩⟩⟩
    · exact mem_fibre.2 ⟨rfl, rfl, rfl, hA⟩
  have hfib : ∀ q ∈ T ×ˢ C ×ˢ univ.image ζ ×ˢ powersetCard a (univ : Finset B),
      ((fibre S X arrive acc ζ j q).card : ℝ) ≤
        ((Fintype.card (Fin n → B) * (univ.filter fun ω => ζ ω = q.2.2.1).card : ℕ) : ℝ) / 2 ^ (a * b) := by
    intro q hq'
    have hA : q.2.2.2.card = a :=
      (mem_powersetCard.1 (mem_product.1 (mem_product.1 (mem_product.1 hq').2).2).2).2
    rw [le_div_iff₀ (by positivity), Nat.cast_mul, ← hA]
    exact fibre_card_le hS hlat hlive hq hlate hu j q
  have hz : ∑ z ∈ univ.image ζ, (univ.filter fun ω => ζ ω = z).card = Fintype.card Ω₀ :=
    (card_eq_sum_card_image ζ univ).symm.trans card_univ
  have hnat : ∑ q ∈ T ×ˢ C ×ˢ univ.image ζ ×ˢ powersetCard a (univ : Finset B),
      Fintype.card (Fin n → B) * (univ.filter fun ω => ζ ω = q.2.2.1).card =
        T.card * C.card * (Fintype.card B).choose a * Fintype.card (Ω₀ × (Fin n → B)) := by
    rw [sum_product4 T C (univ.image ζ) (powersetCard a univ)
      (fun z => Fintype.card (Fin n → B) * (univ.filter fun ω => ζ ω = z).card), ← mul_sum, hz,
      card_powersetCard, card_univ, Fintype.card_prod]
    ring
  refine le_trans (Nat.cast_le.2 (card_le_card hcover)) ?_
  refine le_trans (Nat.cast_le.2 card_biUnion_le) ?_
  rw [Nat.cast_sum]
  refine le_trans (sum_le_sum hfib) (le_of_eq ?_)
  rw [← sum_div, ← Nat.cast_sum, hnat]
  push_cast
  ring

/-- Bit strings of fewer than `m` bits: `∑_{k < m} 2^k < 2^m`. -/
theorem sum_pow_two_lt (m : ℕ) : ∑ k ∈ range m, 2 ^ k < 2 ^ m := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [sum_range_succ, pow_succ]
    omega

/-- A set of coins each in one of the sets `Bad j` or in `Good` has at most their sizes summed. -/
theorem card_le_of_cover {α ι : Type*} [DecidableEq α] [Fintype ι] {s : Finset α} (Bad : ι → Finset α)
    (Good : Finset α) (h : ∀ r ∈ s, (∃ j, r ∈ Bad j) ∨ r ∈ Good) :
    (s.card : ℝ) ≤ ∑ j, ((Bad j).card : ℝ) + Good.card := by
  have hsub : s ⊆ univ.biUnion Bad ∪ Good := fun r hr => by
    rcases h r hr with ⟨j, hj⟩ | hg
    · exact mem_union_left _ (mem_biUnion.2 ⟨j, mem_univ _, hj⟩)
    · exact mem_union_right _ hg
  calc (s.card : ℝ) ≤ ((univ.biUnion Bad ∪ Good).card : ℝ) := Nat.cast_le.2 (card_le_card hsub)
    _ ≤ ((univ.biUnion Bad).card : ℝ) + Good.card := by exact_mod_cast card_union_le _ _
    _ ≤ ∑ j, ((Bad j).card : ℝ) + Good.card := by
      gcongr
      exact_mod_cast card_biUnion_le

open Classical in
theorem WipeReadHeld : Guarantees.Actual.WipeReadHeld := by
  intro Ω₀ B L Z _ _ _ n S ℓ X Y V Ch lab b D ℓmax a m K arrive ζ quiet T hS hlat hlive hlv hq hlate hu
  set acc : Ω₀ × (Fin n → B) → Fin n → Prop := fun r j => Ch j ((S r).view V) with hacc
  set c : ℝ := 1 - ((a : ℝ) - 1) / Fintype.card B with hc
  obtain ⟨Cm, hCm, hCmc⟩ : ∃ Cm : Finset (List Bool), (∀ κ : List Bool, κ.length < m → κ ∈ Cm) ∧
      (Cm.card : ℝ) ≤ 2 ^ m := by
    refine ⟨(range m).biUnion fun k => univ.image (List.ofFn : (Fin k → Bool) → List Bool), fun κ hκ => ?_, ?_⟩
    · exact mem_biUnion.2 ⟨κ.length, mem_range.2 hκ, mem_image.2 ⟨κ.get, mem_univ _, List.ofFn_get κ⟩⟩
    · have h1 := card_biUnion_le (s := range m) (t := fun k => univ.image (List.ofFn : (Fin k → Bool) → List Bool))
      have h2 : ∑ k ∈ range m, (univ.image (List.ofFn : (Fin k → Bool) → List Bool)).card ≤
          ∑ k ∈ range m, 2 ^ k :=
        sum_le_sum fun k _ => card_image_le.trans (by simp)
      exact_mod_cast h1.trans (h2.trans (sum_pow_two_lt m).le)
  obtain ⟨Bad, hBad, hBadc⟩ : ∃ Bad : Fin n → Finset (Ω₀ × (Fin n → B)),
      (∀ j r, arrive r j ∈ T → (S r).content X (arrive r j) ∈ Cm → a ≤ (Actual.answered acc j r).card →
        r ∈ Bad j) ∧
      ∀ j, ((Bad j).card : ℝ) ≤
        T.card * 2 ^ m * ((Fintype.card B).choose a : ℝ) / 2 ^ (a * b) * Fintype.card (Ω₀ × (Fin n → B)) := by
    refine ⟨fun j => univ.filter fun r => arrive r j ∈ T ∧ (S r).content X (arrive r j) ∈ Cm ∧
        a ≤ (Actual.answered acc j r).card, fun j r h1 h2 h3 => mem_filter.2 ⟨mem_univ _, h1, h2, h3⟩,
      fun j => ?_⟩
    refine (AnsweredOfQuiet S ℓ X Y lab b D ℓmax a arrive acc ζ quiet T Cm j hS hlat hlive hq hlate hu).trans ?_
    gcongr
  obtain ⟨Good, hGood, hGoodc⟩ : ∃ Good : Finset (Ω₀ × (Fin n → B)),
      (∀ r, (∀ j, acc r j) → (K : ℝ) * c ≤ ∑ j, Actual.unanswered acc j r → r ∈ Good) ∧
      (Good.card : ℝ) ≤ Real.exp (-((K : ℝ) * c)) * Fintype.card (Ω₀ × (Fin n → B)) :=
    ⟨univ.filter fun r => (∀ j, acc r j) ∧ (K : ℝ) * c ≤ ∑ j, Actual.unanswered acc j r,
      fun r h1 h2 => mem_filter.2 ⟨mem_univ _, h1, h2⟩, WipeReadPool acc _ hlv⟩
  refine le_trans (card_le_of_cover Bad Good fun r hr => ?_) ?_
  · obtain ⟨hall, hK⟩ := (mem_filter.1 hr).2
    by_cases hb : ∃ j, r ∈ Bad j
    · exact Or.inl hb
    simp only [not_exists] at hb
    refine Or.inr (hGood r (fun j => (hall j).1) ?_)
    have hx : ∀ j, (S r).mem X (arrive r j) < m → c ≤ Actual.unanswered acc j r := by
      intro j hj
      have hlt : (Actual.answered acc j r).card < a := by
        by_contra h
        exact hb j (hBad j r (hall j).2 (hCm _ ((content_length_le _ _ _).trans_lt hj)) (not_lt.1 h))
      have : Nonempty B := ⟨r.2 j⟩
      have hB : (0 : ℝ) < Fintype.card B := by exact_mod_cast Fintype.card_pos
      have hle : (Actual.answered acc j r).card ≤ Fintype.card B := card_le_univ _
      have hlt' : ((Actual.answered acc j r).card : ℝ) + 1 ≤ a := by
        have h := Nat.add_one_le_iff.2 hlt
        exact_mod_cast h
      rw [hc, Actual.unanswered, Nat.cast_sub hle, le_div_iff₀ hB, sub_mul, one_mul, div_mul_cancel₀ _ hB.ne']
      linarith
    have hx0 : ∀ j, 0 ≤ Actual.unanswered acc j r := fun j => by
      unfold Actual.unanswered
      positivity
    rcases le_or_gt 0 c with hc0 | hc0
    · calc (K : ℝ) * c ≤ ((univ.filter fun j => (S r).mem X (arrive r j) < m).card : ℝ) * c :=
            mul_le_mul_of_nonneg_right (by exact_mod_cast hK) hc0
        _ = ∑ j ∈ univ.filter fun j => (S r).mem X (arrive r j) < m, c := by rw [sum_const, nsmul_eq_mul]
        _ ≤ ∑ j ∈ univ.filter fun j => (S r).mem X (arrive r j) < m, Actual.unanswered acc j r :=
            sum_le_sum fun j hj => hx j (mem_filter.1 hj).2
        _ ≤ ∑ j, Actual.unanswered acc j r :=
            sum_le_sum_of_subset_of_nonneg (subset_univ _) fun j _ _ => hx0 j
    · calc (K : ℝ) * c ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (Nat.cast_nonneg _) hc0.le
        _ ≤ ∑ j, Actual.unanswered acc j r := sum_nonneg fun j _ => hx0 j
  · have h1 : ∑ j, ((Bad j).card : ℝ) ≤ n * (T.card * 2 ^ m * ((Fintype.card B).choose a : ℝ) / 2 ^ (a * b) *
        Fintype.card (Ω₀ × (Fin n → B))) := by
      refine (sum_le_sum fun j _ => hBadc j).trans (le_of_eq ?_)
      rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    refine (add_le_add h1 hGoodc).trans (le_of_eq ?_)
    ring

end Accounting.SecurityProofs.Actual
