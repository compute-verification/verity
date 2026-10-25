import Definitions.Accounting.Guarantees.TS.Steer
import Proofs.Accounting.TS.Stages

/-!
# Steering under one σ: proofs

A segmented consumer's noise-active coordinates split by segment (`Steer.card_act`: `|act| = Σ_g segCount g` over the
active segments). Zeroing `Z` saves exactly the active segments whose active coordinates all lie in `Z`; their counts
sum to at most `|Z|` (`Steer.saved_mul_le`), which is `SegmentSteer`. Zeroing one whole segment `g` saves one segment
for `segCount g` coordinates (`Steer.segActive_seg`), so LIN-STEER needs `segCount g · G_act ≥ |act|` at every active
segment, which with the sum forces equal counts (`Steer.linSteer_fail`, the least count against the mean).

The union bound runs over `κ ∈ K` and the `C(G, n)` sets of `n` segments; for each, the segments' fresh noises are
independent of each other and of `ω₀` (`Steer.prob_pi_mem`), so all `n` hit their targets with probability at most
`μ^n`. The grinding bound is that union over the `T` queries: split query `q`'s seed off the coins (`Steer.splitAt`),
and since its target doesn't read its own seed, `SteerUnion` with one configuration bounds it by `C(G, n)·μ^n`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Accounting.Guarantees.TS Pouw.RelativeSecurity

namespace Steer

variable {m G : ℕ}

theorem mem_segActive_empty (seg : Fin m → Fin G) (act : Finset (Fin m)) (g : Fin G) :
    g ∈ segActive seg act ∅ ↔ 0 < segCount seg act g := by
  simp only [segActive, segCount, Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty,
    not_false_eq_true, and_true, Finset.card_pos, Finset.filter_nonempty_iff]

theorem segActive_sub (seg : Fin m → Fin G) (act Z : Finset (Fin m)) :
    segActive seg act Z ⊆ segActive seg act ∅ := by
  intro g hg
  simp only [segActive, Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty, not_false_eq_true,
    and_true] at hg ⊢
  obtain ⟨i, hi, hgi, -⟩ := hg
  exact ⟨i, hi, hgi⟩

theorem card_act (seg : Fin m → Fin G) (act : Finset (Fin m)) :
    act.card = ∑ g ∈ segActive seg act ∅, segCount seg act g := by
  classical
  refine Finset.card_eq_sum_card_fiberwise fun i hi => ?_
  refine Finset.mem_coe.mpr ((mem_segActive_empty seg act _).mpr ?_)
  rw [segCount, Finset.card_pos]
  exact ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_coe.mp hi, rfl⟩⟩

/-- The active segments `Z` saves hold, together, at most `|Z|` active coordinates. -/
theorem saved_sum_le (seg : Fin m → Fin G) (act Z : Finset (Fin m)) :
    ∑ g ∈ segActive seg act ∅ \ segActive seg act Z, segCount seg act g ≤ Z.card := by
  classical
  set T := segActive seg act ∅ \ segActive seg act Z
  have hfib : (act.filter fun i => seg i ∈ T).card = ∑ g ∈ T, segCount seg act g := by
    rw [Finset.card_eq_sum_card_fiberwise (s := act.filter fun i => seg i ∈ T) (f := seg) (t := T)
      fun i hi => Finset.mem_coe.mpr (Finset.mem_filter.mp (Finset.mem_coe.mp hi)).2]
    refine Finset.sum_congr rfl fun g hg => ?_
    rw [segCount, Finset.filter_filter]
    congr 1
    ext i
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨hi, -, h⟩; exact ⟨hi, h⟩
    · rintro ⟨hi, h⟩; exact ⟨hi, h ▸ hg, h⟩
  rw [← hfib]
  refine Finset.card_le_card fun i hi => ?_
  obtain ⟨hi, hT⟩ := Finset.mem_filter.mp hi
  have hT' := (Finset.mem_sdiff.mp hT).2
  by_contra hZ
  exact hT' (by simp only [segActive, Finset.mem_filter, Finset.mem_univ, true_and]; exact ⟨i, hi, rfl, hZ⟩)

theorem saving_eq (seg : Fin m → Fin G) (act Z : Finset (Fin m)) (c : ℝ) :
    (segConsumer seg act c).cost ∅ - (segConsumer seg act c).cost Z =
      c * (segActive seg act ∅ \ segActive seg act Z).card := by
  have h := Finset.card_sdiff_add_card_eq_card (segActive_sub seg act Z)
  simp only [segConsumer]
  rw [← h, Nat.cast_add]
  ring

theorem segment_steer (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) (a : ℕ) (hc : 0 ≤ c) (ha : 0 < a)
    (hmin : ∀ g, 0 < segCount seg act g → a ≤ segCount seg act g) (Z : Finset (Fin m)) :
    (segConsumer seg act c).cost ∅ - (segConsumer seg act c).cost Z ≤ (Z.card : ℝ) / a * c := by
  rw [saving_eq]
  set T := segActive seg act ∅ \ segActive seg act Z
  have hnat : a * T.card ≤ Z.card := by
    calc a * T.card = ∑ _g ∈ T, a := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
      _ ≤ ∑ g ∈ T, segCount seg act g := Finset.sum_le_sum fun g hg =>
          hmin g ((mem_segActive_empty seg act g).mp (Finset.mem_sdiff.mp hg).1)
      _ ≤ Z.card := saved_sum_le seg act Z
  have ha' : (0 : ℝ) < a := by exact_mod_cast ha
  have hT : (T.card : ℝ) ≤ Z.card / a := by
    rw [le_div_iff₀ ha', mul_comm]; exact_mod_cast hnat
  calc c * T.card ≤ c * (Z.card / a) := mul_le_mul_of_nonneg_left hT hc
    _ = Z.card / a * c := mul_comm _ _

/-- Zeroing one whole segment's active coordinates leaves every other active segment. -/
theorem segActive_seg (seg : Fin m → Fin G) (act : Finset (Fin m)) (g : Fin G) :
    segActive seg act (act.filter fun i => seg i = g) = (segActive seg act ∅).erase g := by
  ext g'
  simp only [segActive, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_erase, Finset.notMem_empty,
    not_false_eq_true, and_true, not_and]
  constructor
  · rintro ⟨i, hi, rfl, hne⟩
    exact ⟨fun h => hne hi h, i, hi, rfl⟩
  · rintro ⟨hne, i, hi, rfl⟩
    exact ⟨i, hi, rfl, fun _ h => hne h⟩

theorem linSteer_fail (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) (hc : 0 < c) (g : Fin G)
    (hg : 0 < segCount seg act g) (hlt : segCount seg act g * (segActive seg act ∅).card < act.card) :
    ¬ LinSteer (segConsumer seg act c) := by
  classical
  intro h
  have hs := h _ (Finset.filter_subset (fun i => seg i = g) act)
  have hmem : g ∈ segActive seg act ∅ := (mem_segActive_empty seg act g).mpr hg
  rw [saving_eq, segActive_seg, Finset.sdiff_erase_self hmem, Finset.card_singleton, Nat.cast_one, mul_one] at hs
  have hact : (0 : ℝ) < act.card := by exact_mod_cast (Nat.zero_le _).trans_lt hlt
  have hlt' : (segCount seg act g : ℝ) * (segActive seg act ∅).card < act.card := by exact_mod_cast hlt
  have hcost : (segConsumer seg act c).cost ∅ = c * (segActive seg act ∅).card := rfl
  rw [hcost] at hs
  change c ≤ (segCount seg act g : ℝ) / act.card * (c * (segActive seg act ∅).card) at hs
  have : c * act.card ≤ c * ((segCount seg act g : ℝ) * (segActive seg act ∅).card) := by
    have := mul_le_mul_of_nonneg_right hs hact.le
    rw [div_mul_eq_mul_div, div_mul_cancel₀ _ hact.ne'] at this
    linarith
  nlinarith

theorem linSteer_of_even (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) (hc : 0 ≤ c)
    (heq : ∀ g g', 0 < segCount seg act g → 0 < segCount seg act g' → segCount seg act g = segCount seg act g') :
    LinSteer (segConsumer seg act c) := by
  intro Z hZ
  rcases act.eq_empty_or_nonempty with h0 | ⟨i₀, hi₀⟩
  · have hZ0 : Z = ∅ := Finset.subset_empty.mp (h0 ▸ hZ)
    subst hZ0
    simp
  set a := segCount seg act (seg i₀)
  have ha : 0 < a := Finset.card_pos.mpr ⟨i₀, Finset.mem_filter.mpr ⟨hi₀, rfl⟩⟩
  have hall : ∀ g ∈ segActive seg act ∅, segCount seg act g = a := fun g hg =>
    heq g (seg i₀) ((mem_segActive_empty seg act g).mp hg) ha
  have hcard : act.card = a * (segActive seg act ∅).card := by
    rw [card_act seg act, Finset.sum_congr rfl hall, Finset.sum_const, smul_eq_mul, mul_comm]
  have hA : 0 < (segActive seg act ∅).card :=
    Finset.card_pos.mpr ⟨seg i₀, (mem_segActive_empty seg act _).mpr ha⟩
  have hs := segment_steer seg act c a hc ha (fun g hg => (heq g (seg i₀) hg ha).ge) Z
  have hA' : ((segActive seg act ∅).card : ℝ) ≠ 0 := by exact_mod_cast hA.ne'
  have ha' : (a : ℝ) ≠ 0 := by exact_mod_cast ha.ne'
  have hrhs : (Z.card : ℝ) / (segConsumer seg act c).act.card * (segConsumer seg act c).cost ∅ =
      Z.card / a * c := by
    change (Z.card : ℝ) / act.card * (c * (segActive seg act ∅).card) = _
    rw [hcard, Nat.cast_mul]
    field_simp
  rw [hrhs]
  exact hs

/-- Unequal counts put the least count below the mean. -/
theorem exists_below_mean (seg : Fin m → Fin G) (act : Finset (Fin m))
    (hne : ¬ ∀ g g', 0 < segCount seg act g → 0 < segCount seg act g' → segCount seg act g = segCount seg act g') :
    ∃ g, 0 < segCount seg act g ∧ segCount seg act g * (segActive seg act ∅).card < act.card := by
  push Not at hne
  obtain ⟨g₁, g₂, h₁, h₂, h12⟩ := hne
  have hne : (segActive seg act ∅).Nonempty := ⟨g₁, (mem_segActive_empty seg act g₁).mpr h₁⟩
  obtain ⟨g₀, hg₀, hmin⟩ := Finset.exists_min_image _ (segCount seg act) hne
  refine ⟨g₀, (mem_segActive_empty seg act g₀).mp hg₀, ?_⟩
  rw [card_act seg act, mul_comm, ← smul_eq_mul, ← Finset.sum_const]
  have hm₁ := hmin g₁ ((mem_segActive_empty seg act g₁).mpr h₁)
  have hm₂ := hmin g₂ ((mem_segActive_empty seg act g₂).mpr h₂)
  rcases lt_or_gt_of_ne h12 with h | h
  · exact Finset.sum_lt_sum (fun g hg => hmin g hg) ⟨g₂, (mem_segActive_empty seg act g₂).mpr h₂, by omega⟩
  · exact Finset.sum_lt_sum (fun g hg => hmin g hg) ⟨g₁, (mem_segActive_empty seg act g₁).mpr h₁, by omega⟩

theorem segUntouched_empty (seg : Fin m → Fin G) (act : Finset (Fin m)) :
    segUntouched seg act ∅ = segActive seg act ∅ := by
  ext g
  simp [segUntouched, segActive]

theorem segUntouched_single (seg : Fin m → Fin G) (act : Finset (Fin m)) (i : Fin m) :
    segUntouched seg act {i} = (segUntouched seg act ∅).erase (seg i) := by
  ext g
  simp only [segUntouched, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton, forall_eq,
    Finset.mem_erase, Finset.notMem_empty, IsEmpty.forall_iff, implies_true, and_true]
  exact ⟨fun ⟨h, hne⟩ => ⟨Ne.symm hne, h⟩, fun ⟨hne, h⟩ => ⟨h, Ne.symm hne⟩⟩

/-- `s` coordinates drawn independently, each landing in its own event: the probability is the product. -/
theorem prob_pi_mem {B : Type} [Fintype B] [Nonempty B] (S : Finset (Fin G)) (q : Fin G → B → Prop) :
    prob (fun b : Fin G → B => ∀ g ∈ S, q g (b g)) = ∏ g ∈ S, prob (q g) := by
  classical
  unfold prob
  have hB : (Fintype.card B : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have key : ∀ (inst : DecidablePred fun b : Fin G → B => ∀ g ∈ S, q g (b g)),
      (@Finset.filter _ (fun b : Fin G → B => ∀ g ∈ S, q g (b g)) inst Finset.univ).card =
        ∏ g, (if g ∈ S then (Finset.univ.filter (q g)).card else Fintype.card B) := by
    intro inst
    have h : @Finset.filter _ (fun b : Fin G → B => ∀ g ∈ S, q g (b g)) inst Finset.univ =
        Fintype.piFinset fun g => if g ∈ S then Finset.univ.filter (q g) else Finset.univ := by
      ext b
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
      refine ⟨fun h g => ?_, fun h g hg => ?_⟩
      · by_cases hg : g ∈ S
        · simp only [hg, ↓reduceIte]; exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, h g hg⟩
        · simp only [hg, ↓reduceIte]; exact Finset.mem_univ _
      · have := h g
        simp only [hg, ↓reduceIte] at this
        exact (Finset.mem_filter.mp this).2
    rw [h, Fintype.card_piFinset]
    refine Finset.prod_congr rfl fun g _ => ?_
    split_ifs <;> rfl
  rw [key, Fintype.card_fun, Fintype.card_fin, Nat.cast_prod, Nat.cast_pow, ← Fin.prod_const G,
    ← Finset.prod_div_distrib, ← Fintype.prod_ite_mem S]
  refine Finset.prod_congr rfl fun g _ => ?_
  split_ifs with hg
  · rfl
  · exact div_self hB

theorem prob_pi_le {B : Type} [Fintype B] (S : Finset (Fin G)) (q : Fin G → B → Prop) (μ : ℝ) (hμ : 0 ≤ μ)
    (hq : ∀ g, prob (q g) ≤ μ) : prob (fun b : Fin G → B => ∀ g ∈ S, q g (b g)) ≤ μ ^ S.card := by
  rcases isEmpty_or_nonempty B with hB | hB
  · rcases S.eq_empty_or_nonempty with hS | ⟨g, hg⟩
    · rw [hS, Finset.card_empty, pow_zero]; exact prob_le_one _
    · have : IsEmpty (Fin G → B) := ⟨fun f => hB.false (f g)⟩
      have h0 : prob (fun b : Fin G → B => ∀ g ∈ S, q g (b g)) = 0 := by unfold prob; simp
      rw [h0]; positivity
  · rw [prob_pi_mem, ← Finset.prod_const]
    exact Finset.prod_le_prod₀ (fun g _ => prob_nonneg _) fun g _ => hq g

/-- The coins with query `q`'s seed split off. -/
def splitAt {Ω₀ X : Type} {T : ℕ} (q : Fin T) : Ω₀ × (Fin T → X) ≃ (Ω₀ × ({j // j ≠ q} → X)) × X where
  toFun ω := ((ω.1, fun j => ω.2 j), ω.2 q)
  invFun z := (z.1.1, fun j => if h : j = q then z.2 else z.1.2 ⟨j, h⟩)
  left_inv ω := by
    refine Prod.ext rfl (funext fun j => ?_)
    by_cases h : j = q
    · subst h; simp
    · simp [h]
  right_inv z := by
    refine Prod.ext (Prod.ext rfl (funext fun j => ?_)) (by simp)
    simp [j.2]

end Steer

open Steer

theorem LinSteerSegments : Guarantees.TS.LinSteerSegments := by
  intro m G seg act c hc
  refine ⟨fun h => ?_, linSteer_of_even seg act c hc.le⟩
  by_contra hne
  obtain ⟨g, hg, hlt⟩ := exists_below_mean seg act hne
  exact linSteer_fail seg act c hc g hg hlt h

theorem LinSteerFalse : Guarantees.TS.LinSteerFalse := by
  refine ⟨3, 2, ![0, 0, 1], Finset.univ, fun h => ?_⟩
  have := (LinSteerSegments ![0, 0, 1] Finset.univ 1 one_pos).mp h 0 1 (by decide) (by decide)
  exact absurd this (by decide)

theorem SegmentSteer : Guarantees.TS.SegmentSteer := by
  intro m G seg act c a hc ha hmin Z _
  exact segment_steer seg act c a hc ha hmin Z

theorem AdaptSteerFail : Guarantees.TS.AdaptSteerFail := by
  classical
  intro m G seg act c hc hlt h
  obtain ⟨i, hi⟩ : act.Nonempty := Finset.card_pos.mp ((Nat.zero_le _).trans_lt hlt)
  have hs := h {i} (Finset.singleton_subset_iff.mpr hi)
  have hmem : seg i ∈ segUntouched seg act ∅ := by
    rw [segUntouched_empty]
    exact (mem_segActive_empty seg act _).mpr (Finset.card_pos.mpr ⟨i, Finset.mem_filter.mpr ⟨hi, rfl⟩⟩)
  have hsave : (adaptConsumer seg act c).cost ∅ - (adaptConsumer seg act c).cost {i} = c := by
    simp only [adaptConsumer]
    rw [segUntouched_single, Finset.card_erase_of_mem hmem,
      Nat.cast_sub (Finset.card_pos.mpr ⟨_, hmem⟩), Nat.cast_one]
    ring
  have hcost : (adaptConsumer seg act c).cost ∅ = c * (segActive seg act ∅).card := by
    simp only [adaptConsumer]; rw [segUntouched_empty]
  rw [hsave, hcost, Finset.card_singleton, Nat.cast_one] at hs
  change c ≤ 1 / (act.card : ℝ) * (c * (segActive seg act ∅).card) at hs
  have hact : (0 : ℝ) < act.card := by exact_mod_cast (Nat.zero_le _).trans_lt hlt
  have hlt' : ((segActive seg act ∅).card : ℝ) < act.card := by exact_mod_cast hlt
  rw [one_div, inv_mul_eq_div, le_div_iff₀ hact] at hs
  nlinarith

theorem SteerUnion : Guarantees.TS.SteerUnion := by
  classical
  intro Ω₀ B N K _ _ _ G η v μ n hμ hpt
  set P : K → Finset (Fin G) → Ω₀ × (Fin G → B) → Prop := fun κ S ω => ∀ g ∈ S, η g (ω.2 g) = v κ ω.1 g
  have hP : ∀ κ S, S.card = n → prob (P κ S) ≤ μ ^ n := by
    intro κ S hS
    rw [Stages.prob_prod]
    refine expect_le_of_le (pow_nonneg hμ n) fun ω₀ => ?_
    rw [← hS]
    exact prob_pi_le S (fun g b => η g b = v κ ω₀ g) μ hμ fun g => hpt g _
  calc prob (fun ω : Ω₀ × (Fin G → B) => ∃ κ, ∃ S : Finset (Fin G), S.card = n ∧ P κ S ω)
      ≤ prob (fun ω => ∃ κ ∈ Finset.univ, ∃ S ∈ Finset.powersetCard n Finset.univ, P κ S ω) :=
        prob_mono fun ω ⟨κ, S, hS, h⟩ => ⟨κ, Finset.mem_univ _, S, Finset.mem_powersetCard.mpr
          ⟨Finset.subset_univ _, hS⟩, h⟩
    _ ≤ ∑ κ ∈ Finset.univ, prob (fun ω => ∃ S ∈ Finset.powersetCard n Finset.univ, P κ S ω) :=
        prob_exists_le _ _
    _ ≤ ∑ κ ∈ (Finset.univ : Finset K), ∑ S ∈ Finset.powersetCard n (Finset.univ : Finset (Fin G)), μ ^ n :=
        Finset.sum_le_sum fun κ _ => (prob_exists_le _ _).trans
          (Finset.sum_le_sum fun S hS => hP κ S (Finset.mem_powersetCard.mp hS).2)
    _ = Fintype.card K * G.choose n * μ ^ n := by
        rw [Finset.sum_const, Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, Finset.card_univ,
          Fintype.card_fin, nsmul_eq_mul, nsmul_eq_mul]
        ring

theorem GrindUnion : Guarantees.TS.GrindUnion := by
  classical
  intro Ω₀ B N _ _ T G η v μ n hμ hpt hv
  set E : Fin T → Ω₀ × (Fin T → Fin G → B) → Prop := fun q ω =>
    ∃ S : Finset (Fin G), S.card = n ∧ ∀ g ∈ S, η g (ω.2 q g) = v q ω.1 ω.2 g
  have hq : ∀ q, prob (E q) ≤ G.choose n * μ ^ n := by
    intro q
    rcases isEmpty_or_nonempty (Fin G → B) with hX | hX
    · have : IsEmpty (Ω₀ × (Fin T → Fin G → B)) := ⟨fun ω => hX.false (ω.2 q)⟩
      have h0 : prob (E q) = 0 := by unfold prob; simp
      rw [h0]; positivity
    obtain ⟨x₀⟩ := hX
    set Φ := splitAt (Ω₀ := Ω₀) (X := Fin G → B) q
    set w : Unit → Ω₀ × ({j // j ≠ q} → Fin G → B) → Fin G → N := fun _ z => v q z.1 (Φ.symm (z, x₀)).2
    rw [prob_equiv Φ (E q)]
    calc prob (fun z => E q (Φ.symm z))
        ≤ prob (fun z : (Ω₀ × ({j // j ≠ q} → Fin G → B)) × (Fin G → B) =>
            ∃ κ, ∃ S : Finset (Fin G), S.card = n ∧ ∀ g ∈ S, η g (z.2 g) = w κ z.1 g) := by
          refine prob_mono fun z ⟨S, hS, h⟩ => ⟨(), S, hS, fun g hg => ?_⟩
          have hz : (Φ.symm z).2 q = z.2 := by simp [Φ, splitAt]
          have hw : v q (Φ.symm z).1 (Φ.symm z).2 = w () z.1 := by
            refine hv q _ _ _ fun q' hq' => ?_
            simp [Φ, splitAt, hq']
          rw [← hz, ← hw]
          exact h g hg
      _ ≤ Fintype.card Unit * G.choose n * μ ^ n := SteerUnion η w μ n hμ hpt
      _ = G.choose n * μ ^ n := by rw [Fintype.card_unit, Nat.cast_one, one_mul]
  calc prob (fun ω : Ω₀ × (Fin T → Fin G → B) => ∃ q, E q ω)
      ≤ prob (fun ω => ∃ q ∈ Finset.univ, E q ω) := prob_mono fun ω ⟨q, h⟩ => ⟨q, Finset.mem_univ _, h⟩
    _ ≤ ∑ q ∈ (Finset.univ : Finset (Fin T)), prob (E q) := prob_exists_le _ _
    _ ≤ ∑ _q ∈ (Finset.univ : Finset (Fin T)), (G.choose n * μ ^ n : ℝ) := Finset.sum_le_sum fun q _ => hq q
    _ = T * G.choose n * μ ^ n := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring

end Accounting.SecurityProofs.TS
