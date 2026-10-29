import Definitions.NetworkCertifier.Guarantees.PooledLeak
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Soundness.Audit.Periods
import Proofs.NetworkCertifier.Period
import Proofs.NetworkCertifier.Pooled
import Proofs.NetworkCertifier.Tap

/-!
# The leak through the network certifier over a monitoring period of sampled audits

The probability half: each window's audit, given the earlier windows' coins, accepts with probability at most
`(1 - q)^|W_i| + e_i`, and the escapes multiply over the windows with the errors adding (`pr_adaptive_add_le`). That is
the monitoring-period step (`FlockSoundness.Audit.Periods.prCoin_forall_prod_le`) at each window's bound capped at `1`,
whose product is at most the product of the escapes plus the errors' sum (`prod_min_add_le`). The tied audits'
incorrect proof blocks are sampled proofs' pooled bound, an input taken as it is, and a union bound adds it and the leak
relations' errors `e' i`.

What a tied proof block reaches: a gate outside the downstream cone of the gates holding free values computes the
value it computes with none free (`valOver_congr_free`, gate by gate as `valOver` reads), so a returned value of a tied
audit wired to a scan input outside what its incorrect proof blocks reach is the value its Program computes
(`TiedAudit.valOver_eq_of_notMem_reach`). That is why the leak relation pins every other registered input.

The count half: a family of sets of incorrect proof blocks with fewer than `Fmax` in total is the proof blocks listed by
some `Fin Fmax → Option (Σ i, Fin (n i))` (`exists_famOf_eq`), so there are at most `(N + 1)^Fmax` of them, and a family
of fewer than `F_t` tied proof blocks one of at most `(N_t + 1)^F_t`. At one pair of families, each window's
observations number at most `2^#(freeInputs \ (reg \ A)) · 2^(faultyBits) · Dj^k` (`encard_leakAt_le`, from
`encard_acceptedFaulty_le` at `AsReg`), `A` what the window's incorrect tied proof blocks reach, at most
`2^#(freeInputs \ reg) · 2^#A` (`card_sdiff_sdiff_le`); the incorrect proof blocks' bits sum to at most `b · Fmax`, and
the `A` to at most `R · F_t` inputs (`card_tiedReach_le`). Every `A` is within the window's span
(`tiedReach_subset_span`), and a leak relation with fewer inputs pinned holds whenever one with more does
(`leakAt_mono`), so the same observations are also among those at the spans unpinned, `2^(Σ_i #span_i)` in place of
the tied families. Under `trusted-host` the cells the client receives read back from the network certifier's record
under the schedule, one its record format holds (`tap_cells_read`), so they number at most `#Ss` times those records,
which bounds the secrets they decode to (`encard_pooledGood_le`); `∏_c (cap_c + 1)` is room, as in the period's bound.

At no tied audit (`horizonLeak_untied`) the statement is the one without them: nothing is unpinned, and the tied factor
and error are `1` and `0`.
-/

namespace NetTiming

open Claims Claims.Sampling Finset

/-! ### The escapes multiply, each with an additive error -/

/-- Bounds capped at `1`, each an escape `a i ≤ 1` plus an error `e i`, multiply to at most the escapes' product plus
the errors' sum. -/
theorem prod_min_add_le : ∀ {m : ℕ} (a e : Fin m → ℝ), (∀ i, 0 ≤ a i) → (∀ i, a i ≤ 1) → (∀ i, 0 ≤ e i) →
    ∏ i, min 1 (a i + e i) ≤ ∏ i, a i + ∑ i, e i
  | 0, _, _, _, _, _ => by simp
  | m + 1, a, e, ha0, ha1, he => by
    have ih : ∏ i : Fin m, min 1 (a i.succ + e i.succ) ≤ ∏ i : Fin m, a i.succ + ∑ i : Fin m, e i.succ :=
      prod_min_add_le _ _ (fun i => ha0 _) (fun i => ha1 _) (fun i => he _)
    have hm0 : ∀ i : Fin m, 0 ≤ min 1 (a i.succ + e i.succ) := fun i => le_min zero_le_one (add_nonneg (ha0 _) (he _))
    have hX0 : 0 ≤ ∏ i : Fin m, min 1 (a i.succ + e i.succ) := Finset.prod_nonneg fun i _ => hm0 i
    have hX1 : ∏ i : Fin m, min 1 (a i.succ + e i.succ) ≤ 1 :=
      Finset.prod_le_one₀ (fun i _ => hm0 i) fun i _ => min_le_left _ _
    have hE0 : 0 ≤ ∑ i : Fin m, e i.succ := Finset.sum_nonneg fun i _ => he _
    rw [Fin.prod_univ_succ, Fin.prod_univ_succ, Fin.sum_univ_succ]
    calc min 1 (a 0 + e 0) * ∏ i : Fin m, min 1 (a i.succ + e i.succ)
        ≤ (a 0 + e 0) * ∏ i : Fin m, min 1 (a i.succ + e i.succ) := mul_le_mul_of_nonneg_right (min_le_right _ _) hX0
      _ = a 0 * ∏ i : Fin m, min 1 (a i.succ + e i.succ) + e 0 * ∏ i : Fin m, min 1 (a i.succ + e i.succ) :=
          add_mul _ _ _
      _ ≤ a 0 * (∏ i : Fin m, a i.succ + ∑ i : Fin m, e i.succ) + e 0 * 1 :=
          add_le_add (mul_le_mul_of_nonneg_left ih (ha0 0)) (mul_le_mul_of_nonneg_left hX1 (he 0))
      _ ≤ a 0 * ∏ i : Fin m, a i.succ + (e 0 + ∑ i : Fin m, e i.succ) := by
          rw [mul_add, mul_one]
          linarith [mul_le_of_le_one_left hE0 (ha1 0)]

open FlockSoundness.Game in
/-- **The escapes multiply with additive errors**: windows whose audits, given the earlier windows' coins, accept with
probability at most `r^(f i) + e i`, `f i` fixed before window `i`'s coins, all accept with `Σ_i f i ≥ F` with
probability at most `r^F + Σ_i e i`: the monitoring-period step at the bounds capped at `1`. -/
theorem pr_adaptive_add_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1)
    (m : ℕ) (A : Fin m → (Fin m → Ω) → Ω → Prop) (f : Fin m → (Fin m → Ω) → ℕ) (e : Fin m → ℝ)
    (hA : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → A i ω = A i ω')
    (hf : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → f i ω = f i ω') (he : ∀ i, 0 ≤ e i)
    (hesc : ∀ i ω, Pr (A i ω) ≤ r ^ f i ω + e i) (F : ℕ) :
    Pr (fun ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤ r ^ F + ∑ i, e i := by
  have hle1 : ∀ p : Ω → Prop, prCoin p ≤ 1 := fun p =>
    (prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans_eq (by simp [prCoin, avg_const])
  have hstep : ∀ (t : Fin m) (ω : Fin m → Ω), prCoin (fun a : Ω => A t (Function.update ω t a)
      (Function.update ω t a t)) ≤ ENNReal.ofReal (min 1 (r ^ f t ω + e t)) := fun t ω => by
    have hupd : (fun a : Ω => A t (Function.update ω t a) (Function.update ω t a t)) = A t ω := funext fun a => by
      rw [Function.update_self, hA t (Function.update ω t a) ω fun j hj => Function.update_of_ne (ne_of_lt hj) _ _]
    rw [hupd, ← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (hle1 _)),
      ← Claims.Sampling.Pr_eq_toReal_prCoin]
    exact ENNReal.ofReal_le_ofReal (le_min (Pr_le_one _) (hesc t ω))
  have key := FlockSoundness.Audit.Periods.prCoin_forall_prod_le (Ω := fun _ : Fin m => Ω)
    (fun t ω => A t ω (ω t)) (fun t ω => ENNReal.ofReal (min 1 (r ^ f t ω + e t)))
    (fun t ω ω' h => by
      show A t ω (ω t) = A t ω' (ω' t)
      rw [hA t ω ω' fun j hj => h j hj.le, h t le_rfl])
    (fun t ω ω' h => by simp only [hf t ω ω' h]) hstep (ENNReal.ofReal (r ^ F + ∑ i, e i))
  have he0 : 0 ≤ r ^ F + ∑ i, e i := add_nonneg (pow_nonneg hr0 F) (Finset.sum_nonneg fun i _ => he i)
  have hprod : ∀ ω : Fin m → Ω, F ≤ ∑ i, f i ω →
      ∏ t, ENNReal.ofReal (min 1 (r ^ f t ω + e t)) ≤ ENNReal.ofReal (r ^ F + ∑ i, e i) := fun ω hF => by
    rw [← ENNReal.ofReal_prod_of_nonneg fun i _ => le_min zero_le_one (add_nonneg (pow_nonneg hr0 _) (he i))]
    refine ENNReal.ofReal_le_ofReal ((prod_min_add_le (fun i => r ^ f i ω) e (fun i => pow_nonneg hr0 _)
      (fun i => pow_le_one₀ hr0 hr1) he).trans ?_)
    rw [Finset.prod_pow_eq_pow_sum]
    exact add_le_add (pow_le_pow_of_le_one hr0 hr1 hF) le_rfl
  have hmono : prCoin (fun ω : Fin m → Ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤
      ENNReal.ofReal (r ^ F + ∑ i, e i) :=
    (prCoin_mono (q := fun ω : Fin m → Ω => (∀ t, A t ω (ω t)) ∧
      ∏ t, ENNReal.ofReal (min 1 (r ^ f t ω + e t)) ≤ ENNReal.ofReal (r ^ F + ∑ i, e i))
      fun ω hω => ⟨hω.1, hprod ω hω.2⟩).trans key
  rw [Claims.Sampling.Pr_eq_toReal_prCoin]
  exact (ENNReal.toReal_mono ENNReal.ofReal_ne_top hmono).trans_eq (ENNReal.toReal_ofReal he0)

/-! ### The families of faulty sets -/

/-- The units that `g` lists in window `i`. -/
def famOf {m : ℕ} {n : Fin m → ℕ} {F : ℕ} (g : Fin F → Option (Σ i : Fin m, Fin (n i))) (i : Fin m) : Finset ℕ :=
  (univ.filter fun v : Fin (n i) => ∃ t, g t = some ⟨i, v⟩).image Fin.val

theorem famOf_subset {m : ℕ} {n : Fin m → ℕ} {F : ℕ} (g : Fin F → Option (Σ i : Fin m, Fin (n i))) (i : Fin m) :
    famOf g i ⊆ range (n i) := by
  intro x hx
  simp only [famOf, mem_image] at hx
  obtain ⟨v, -, rfl⟩ := hx
  exact mem_range.2 v.2

theorem exists_famOf_eq {m : ℕ} (n : Fin m → ℕ) (Fmax : ℕ) (W : Fin m → Finset ℕ) (hW : ∀ i, W i ⊆ range (n i))
    (hF : ∑ i, (W i).card ≤ Fmax) : ∃ g : Fin Fmax → Option (Σ i : Fin m, Fin (n i)), famOf g = W := by
  classical
  let T : Finset (Σ i : Fin m, Fin (n i)) := univ.sigma fun i => univ.filter fun v : Fin (n i) => v.val ∈ W i
  have hfib : ∀ i, (univ.filter fun v : Fin (n i) => v.val ∈ W i).card = (W i).card := by
    intro i
    rw [← Finset.card_map Fin.valEmbedding]
    congr 1
    ext x
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply]
    constructor
    · rintro ⟨v, hv, rfl⟩
      exact hv
    · intro hx
      exact ⟨⟨x, Finset.mem_range.1 (hW i hx)⟩, hx, rfl⟩
  have hTc : T.card ≤ Fmax := by
    rw [Finset.card_sigma]
    simp only [hfib]
    exact hF
  let g : Fin Fmax → Option (Σ i : Fin m, Fin (n i)) := fun t =>
    if h : t.val < T.card then some (T.equivFin.symm ⟨t.val, h⟩).val else none
  have hg : ∀ p, (∃ t, g t = some p) ↔ p ∈ T := by
    intro p
    constructor
    · rintro ⟨t, ht⟩
      by_cases h : t.val < T.card
      · simp only [g, h, dite_true, Option.some.injEq] at ht
        rw [← ht]
        exact (T.equivFin.symm _).2
      · simp only [g, h, dite_false, reduceCtorEq] at ht
    · intro hp
      refine ⟨⟨(T.equivFin ⟨p, hp⟩).val, lt_of_lt_of_le (T.equivFin ⟨p, hp⟩).2 hTc⟩, ?_⟩
      simp only [g, (T.equivFin ⟨p, hp⟩).2, dite_true, Fin.eta, Equiv.symm_apply_apply]
  refine ⟨g, funext fun i => ?_⟩
  ext x
  simp only [famOf, mem_image, mem_filter, mem_univ, true_and, hg]
  constructor
  · rintro ⟨v, hv, rfl⟩
    simpa [T] using hv
  · intro hx
    exact ⟨⟨x, mem_range.1 (hW i hx)⟩, by simpa [T] using hx, rfl⟩

theorem faultyBits_le_card_mul (C : Prog) (P : Partition) {n b : ℕ} {F : Finset ℕ} (hF : F ⊆ range n)
    (hb : ∀ v < n, faultyBits C P {v} ≤ b) : faultyBits C P F ≤ F.card * b := by
  have h : faultyBits C P F = ∑ v ∈ F, faultyBits C P {v} := by simp only [faultyBits, Finset.sum_singleton]
  rw [h]
  calc ∑ v ∈ F, faultyBits C P {v} ≤ ∑ _v ∈ F, b := sum_le_sum fun v hv => hb v (mem_range.1 (hF hv))
    _ = F.card * b := by rw [sum_const, smul_eq_mul]

/-! ### The tied audits -/

theorem asReg_mono {L S : ℕ} (C : Prog) (lay : BoundaryGates L S) (k : ℕ) {R R' : Finset ℕ} (ρ : ℕ → Bool)
    (h : R' ⊆ R) : AsReg C lay k R ρ ⊆ AsReg C lay k R' ρ := by
  rintro a ⟨hR, hF⟩
  exact ⟨fun g hg hf => hR g (h hg) hf, hF⟩

/-- A leak relation with fewer registered inputs pinned holds whenever one with more does. -/
theorem leakAt_mono {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    {reg reg' : AuditStatement → Finset ℕ} (Dj : ℕ) (s : AuditStatement) (P : Partition) (ρ : ℕ → Bool)
    (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ) (h : reg' s ⊆ reg s) {o : List (Obs L K S)}
    (hl : LeakAt lay k reg Dj s P ρ op F o) : LeakAt lay k reg' Dj s P ρ op F o := by
  obtain ⟨h1, h2, h3, a, ha, free, hfree, hacc⟩ := hl
  exact ⟨h1, h2, h3, a, asReg_mono _ _ _ ρ h ha, free, hfree, hacc⟩

theorem card_sdiff_sdiff_le (X R A : Finset ℕ) : (X \ (R \ A)).card ≤ (X \ R).card + A.card := by
  have h : X \ (R \ A) ⊆ X \ R ∪ A := by
    intro x hx
    simp only [Finset.mem_sdiff, Finset.mem_union] at hx ⊢
    tauto
  exact (Finset.card_le_card h).trans (Finset.card_union_le _ _)

theorem getD_tiedReaches_card_le (T : List TiedAudit) (u : ℕ) : ((tiedReaches T).getD u ∅).card ≤ tiedWidth T := by
  unfold tiedWidth
  by_cases hu : u < (tiedReaches T).length
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hu, Option.getD_some]
    exact Finset.le_sup (f := Finset.card) (List.mem_toFinset.2 (List.getElem_mem hu))
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (not_lt.1 hu), Option.getD_none, Finset.card_empty]
    exact Nat.zero_le _

/-- The tied proof blocks `F` reach at most `R` scan inputs each. -/
theorem card_tiedReach_le (T : List TiedAudit) (F : Finset ℕ) : (tiedReach T F).card ≤ F.card * tiedWidth T :=
  Finset.card_biUnion_le_card_mul _ _ _ fun u _ => getD_tiedReaches_card_le T u

theorem tiedReach_subset_span (T : List TiedAudit) (F : Finset ℕ) : tiedReach T F ⊆ tiedSpan T := by
  intro x hx
  unfold tiedReach at hx
  unfold tiedSpan tiedReach
  obtain ⟨u, -, hxu⟩ := Finset.mem_biUnion.1 hx
  by_cases hu : u < (tiedReaches T).length
  · exact Finset.mem_biUnion.2 ⟨u, Finset.mem_range.2 hu, hxu⟩
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (not_lt.1 hu), Option.getD_none] at hxu
    exact absurd hxu (Finset.notMem_empty x)

/-- An index past `tiedReaches T` reaches nothing, so `F` reaches what its indices into it reach. -/
theorem tiedReach_inter_range (T : List TiedAudit) (F : Finset ℕ) :
    tiedReach T (F ∩ range (tiedReaches T).length) = tiedReach T F := by
  ext x
  unfold tiedReach
  constructor
  · intro hx
    obtain ⟨u, hu, hxu⟩ := Finset.mem_biUnion.1 hx
    exact Finset.mem_biUnion.2 ⟨u, (Finset.mem_inter.1 hu).1, hxu⟩
  · intro hx
    obtain ⟨u, hu, hxu⟩ := Finset.mem_biUnion.1 hx
    by_cases hl : u < (tiedReaches T).length
    · exact Finset.mem_biUnion.2 ⟨u, Finset.mem_inter.2 ⟨hu, Finset.mem_range.2 hl⟩, hxu⟩
    · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (not_lt.1 hl), Option.getD_none] at hxu
      exact absurd hxu (Finset.notMem_empty x)

theorem tiedReach_nil (F : Finset ℕ) : tiedReach [] F = ∅ := by
  ext x
  simp [tiedReach, tiedReaches]

theorem tiedSpan_nil : tiedSpan [] = ∅ := tiedReach_nil _

theorem tiedWidth_nil : tiedWidth [] = 0 := by simp [tiedWidth, tiedReaches]

theorem sup_tiedWidth_nil {m : ℕ} : ((univ : Finset (Fin m)).sup fun _ => tiedWidth []) = 0 :=
  Nat.eq_zero_of_le_zero (Finset.sup_le fun _ _ => tiedWidth_nil.le)

/-! ### What a tied proof block reaches -/

/-- **Free values reach only their downstream cone**: free values that agree off the gates `Sv v` of the proof blocks
`v ∈ F` give one value at every gate outside each one's cone (`inCone`). -/
theorem valOver_congr_free (C : Prog) (inp : ℕ → Bool) (F : Finset ℕ) (Sv : ℕ → List ℕ)
    {free free' : ℕ → Option Bool} (h : ∀ g, (∀ v ∈ F, g ∉ Sv v) → free g = free' g) (g : ℕ)
    (hg : ∀ v ∈ F, inCone C (Sv v) g = false) : valOver C inp free g = valOver C inp free' g := by
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    have hgS : ∀ v ∈ F, g ∉ Sv v := fun v hv => by
      have hv' := hg v hv
      rw [inCone] at hv'
      simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not] at hv'
      exact hv'.1
    have hfg := h g hgS
    rcases hf : free g with _ | b
    · have hf' : free' g = none := by rw [← hfg]; exact hf
      rcases hx : C.gates[g]? with _ | x
      · rw [valOver_of_none_none C inp hf hx, valOver_of_none_none C inp hf' hx]
      · rw [valOver_of_none_some C inp hf hx, valOver_of_none_some C inp hf' hx]
        congr 1
        apply bgate_val_congr x
        intro i hi
        split_ifs with hig
        · have hi' : ∀ v ∈ F, inCone C (Sv v) i = false := fun v hv => by
            have hv' := hg v hv
            rw [inCone] at hv'
            simp only [Bool.or_eq_false_iff, hx, List.any_eq_false] at hv'
            simpa [hig] using hv'.2 i hi
          rw [ih i hig hi']
        · rfl
    · have hf' : free' g = some b := by rw [← hfg]; exact hf
      rw [valOver_of_some C inp hf, valOver_of_some C inp hf']

/-- **What a tied audit's incorrect proof blocks reach**: with free values only at the committed gates of the proof
blocks `F` (as `AcceptedFaulty` takes them), a returned value wired to a scan input outside what `F` reaches
(`TiedAudit.reach`) is the value the audit's Program computes with no free value. -/
theorem TiedAudit.valOver_eq_of_notMem_reach (a : TiedAudit) (inp : ℕ → Bool) {free : ℕ → Option Bool}
    {F : Finset ℕ} (hfree : ∀ g b, free g = some b → FaultyGate a.prog a.part F g) {w o : ℕ}
    (hwo : (w, o) ∈ a.wire.zip a.prog.outs) (hw : w ∉ F.biUnion a.reach) :
    valOver a.prog inp free o = valOver a.prog inp (fun _ => none) o := by
  refine valOver_congr_free a.prog inp F (fun v => RU.gates (a.part.rus.getD v [])) (fun g hg => ?_) o
    fun v hv => ?_
  · rcases hf : free g with _ | b
    · rfl
    · obtain ⟨v, hv, r, hr, hgr, -⟩ := hfree g b hf
      have hgv : g ∈ RU.gates (a.part.rus.getD v []) := by
        rw [List.getD_eq_getElem?_getD, hr, Option.getD_some]
        exact hgr
      exact (hg v hv hgv).elim
  · by_contra hc
    simp only [Bool.not_eq_false] at hc
    exact hw (Finset.mem_biUnion.2 ⟨v, hv, List.mem_toFinset.2
      (List.mem_map.2 ⟨(w, o), List.mem_filter.2 ⟨hwo, hc⟩, rfl⟩)⟩)

/-! ### The count -/

theorem encard_leakAt_le {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    (s : AuditStatement) (P : Partition) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ) :
    {o | LeakAt lay k reg Dj s P ρ op F o}.encard ≤
      ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card * (2 ^ faultyBits s.prog P F * Dj ^ k s) : ℕ) : ℕ∞) := by
  by_cases hfits : (lay s).Fits s.prog (k s)
  · calc _ ≤ {o : List (Obs L K S) | o.length = k s ∧ (∀ x ∈ o, x.jitter < Dj) ∧
            AcceptedFaulty s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) F op o}.encard :=
          Set.encard_le_encard fun o h => h.2
      _ ≤ (AsReg s.prog (lay s) (k s) (reg s) ρ).encard * ((2 ^ faultyBits s.prog P F * Dj ^ k s : ℕ) : ℕ∞) :=
          encard_acceptedFaulty_le _ _ _ _ hfits _ _ _ _
      _ ≤ ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card : ℕ) : ℕ∞) *
            ((2 ^ faultyBits s.prog P F * Dj ^ k s : ℕ) : ℕ∞) :=
          mul_le_mul' (encard_asReg_le _ _ _ _ _) le_rfl
      _ = _ := (Nat.cast_mul _ _).symm
  · have : {o | LeakAt lay k reg Dj s P ρ op F o} = ∅ :=
      Set.eq_empty_iff_forall_notMem.2 fun o h => hfits h.1
    rw [this, Set.encard_empty]
    exact zero_le

theorem encard_pooledGood_le {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → ℕ → List (Obs L K S) → List Bool) (T : Fin m → List TiedAudit) (n : Fin m → ℕ) (b : ℕ)
    (hb : ∀ i, ∀ v < n i, faultyBits (sw i).prog (Pw i) {v} ≤ b)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (sched : Z → WardenSchedule Conn) (srv : Z → Srv) (hs : ∀ z ∈ Zs, sched z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (sched z, srv z)).1 = z)
    (W Wt : Z → Fin m → Finset ℕ) (hWn : ∀ z i, W z i ⊆ range (n i)) (Fmax Ft : ℕ) :
    {z | z ∈ Zs ∧ PooledGood lay k reg Dj sw Pw ρ op T Fmax Ft (W z) (Wt z) (run tap (sched z, srv z)).2}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
        (∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i))) *
        min (((∑ i, (tiedReaches (T i)).length) + 1) ^ Ft * 2 ^ ((univ.sup fun i => tiedWidth (T i)) * Ft))
          (2 ^ ∑ i, (tiedSpan (T i)).card) : ℕ) : ℕ∞) := by
  classical
  set G := {z | z ∈ Zs ∧ PooledGood lay k reg Dj sw Pw ρ op T Fmax Ft (W z) (Wt z) (run tap (sched z, srv z)).2}
    with hG
  set c : Fin m → ℕ := fun i => (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card with hc
  set R : ℕ := univ.sup fun i => tiedWidth (T i)
  set P0 : ℕ := ∏ i, (2 ^ c i * Dj ^ k (sw i)) with hP0
  set A1 : ℕ := ((∑ i, (tiedReaches (T i)).length) + 1) ^ Ft * 2 ^ (R * Ft) with hA1
  set A2 : ℕ := 2 ^ ∑ i, (tiedSpan (T i)).card
  let s : Finset (Fin Fmax → Option (Σ i : Fin m, Fin (n i))) := univ.filter fun g => ∑ i, (famOf g i).card < Fmax
  let st : Finset (Fin Ft → Option (Σ i : Fin m, Fin ((tiedReaches (T i)).length))) :=
    univ.filter fun g => ∑ i, (famOf g i).card < Ft
  let X : (Fin Fmax → Option (Σ i : Fin m, Fin (n i))) → (Fin m → Finset ℕ) → Set (Fin m → List (Obs L K S)) :=
    fun g A => Set.pi Set.univ fun i =>
      {o | LeakAt lay k (fun t => reg t \ A i) Dj (sw i) (Pw i) (ρ i) (op i) (famOf g i) o}
  let Y : Set (Fin m → List (Obs L K S)) := ⋃ g ∈ s, ⋃ gt ∈ st, X g fun i => tiedReach (T i) (famOf gt i)
  let Y2 : Set (Fin m → List (Obs L K S)) := ⋃ g ∈ s, X g fun i => tiedSpan (T i)
  let V : Set (Conn → List (ℕ × CellView Cell)) :=
    (fun p : WardenSchedule Conn × (Fin m → List (Obs L K S)) => f.read p.1 p.2) '' (Ss ×ˢ Y)
  have hgood : ∀ z ∈ G, (run tap (sched z, srv z)).2 ∈ Y := by
    rintro z ⟨-, hlt, hle, hleak⟩
    obtain ⟨g, hg⟩ := exists_famOf_eq n Fmax (W z) (hWn z) hlt.le
    have hle' : ∑ i, (Wt z i ∩ range (tiedReaches (T i)).length).card < Ft :=
      (sum_le_sum fun i _ => Finset.card_le_card Finset.inter_subset_left).trans_lt hle
    obtain ⟨gt, hgt⟩ := exists_famOf_eq (fun i => (tiedReaches (T i)).length) Ft
      (fun i => Wt z i ∩ range (tiedReaches (T i)).length) (fun i => Finset.inter_subset_right) hle'.le
    refine Set.mem_biUnion (x := g) ?_ ?_
    · simp only [s, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq, hg]
      exact hlt
    refine Set.mem_biUnion (x := gt) ?_ ?_
    · simp only [st, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq, hgt]
      exact hle'
    · intro i _
      simp only [Set.mem_ofPred_eq, hg, hgt, tiedReach_inter_range]
      exact hleak i
  have hY12 : Y ⊆ Y2 := by
    intro y hy
    simp only [Y, Y2, Set.mem_iUnion] at hy ⊢
    obtain ⟨g, hg, gt, -, hx⟩ := hy
    refine ⟨g, hg, ?_⟩
    intro i _
    exact leakAt_mono lay k (reg := fun t => reg t \ tiedReach (T i) (famOf gt i))
      (reg' := fun t => reg t \ tiedSpan (T i)) Dj (sw i) (Pw i) (ρ i) (op i) (famOf g i)
      (Finset.sdiff_subset_sdiff (Finset.Subset.refl _) (tiedReach_subset_span _ _)) (hx i (Set.mem_univ i))
  have hsub : G ⊆ dec '' V := fun z hz =>
    ⟨(run tap (sched z, srv z)).1, ⟨(sched z, (run tap (sched z, srv z)).2), ⟨hs z hz.1, hgood z hz⟩,
      (tap_cells_read f arr tap run htrust _ (hfit _ (hs z hz.1)) _).symm⟩, hdz z hz.1⟩
  have hX : ∀ g ∈ s, ∀ A : Fin m → Finset ℕ,
      (X g A).encard ≤ ((2 ^ (b * Fmax) * P0 * 2 ^ ∑ i, (A i).card : ℕ) : ℕ∞) := by
    intro g hg A
    have hlt : ∑ i, (famOf g i).card < Fmax := (Finset.mem_filter.1 hg).2
    have hbits : ∑ i, faultyBits (sw i).prog (Pw i) (famOf g i) ≤ b * Fmax := by
      calc ∑ i, faultyBits (sw i).prog (Pw i) (famOf g i) ≤ ∑ i, (famOf g i).card * b :=
            sum_le_sum fun i _ => faultyBits_le_card_mul _ _ (famOf_subset g i) (hb i)
        _ = (∑ i, (famOf g i).card) * b := (sum_mul _ _ _).symm
        _ ≤ Fmax * b := Nat.mul_le_mul_right _ hlt.le
        _ = b * Fmax := mul_comm _ _
    have hwin : ∀ i, 2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) \ A i)).card *
        (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * Dj ^ k (sw i)) ≤
        2 ^ (A i).card * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * (2 ^ c i * Dj ^ k (sw i))) := by
      intro i
      calc 2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) \ A i)).card *
            (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * Dj ^ k (sw i))
          ≤ 2 ^ (c i + (A i).card) * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * Dj ^ k (sw i)) :=
            Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) (card_sdiff_sdiff_le _ _ _))
        _ = 2 ^ (A i).card * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * (2 ^ c i * Dj ^ k (sw i))) := by
            rw [pow_add]; ring
    have hnat : ∏ i, (2 ^ (A i).card * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * (2 ^ c i * Dj ^ k (sw i)))) ≤
        2 ^ (b * Fmax) * P0 * 2 ^ ∑ i, (A i).card := by
      calc ∏ i, (2 ^ (A i).card * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * (2 ^ c i * Dj ^ k (sw i))))
          = (∏ i, 2 ^ faultyBits (sw i).prog (Pw i) (famOf g i)) * P0 * ∏ i, 2 ^ (A i).card := by
            rw [hP0, ← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
            exact Finset.prod_congr rfl fun i _ => by ring
        _ = 2 ^ (∑ i, faultyBits (sw i).prog (Pw i) (famOf g i)) * P0 * 2 ^ ∑ i, (A i).card := by
            rw [Finset.prod_pow_eq_pow_sum, Finset.prod_pow_eq_pow_sum]
        _ ≤ 2 ^ (b * Fmax) * P0 * 2 ^ ∑ i, (A i).card :=
            Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) hbits))
    calc (X g A).encard
        = ∏ i, {o | LeakAt lay k (fun t => reg t \ A i) Dj (sw i) (Pw i) (ρ i) (op i) (famOf g i) o}.encard :=
          Set.encard_pi_eq_prod_encard
      _ ≤ ∏ i, ((2 ^ (A i).card * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * (2 ^ c i * Dj ^ k (sw i))) : ℕ) :
            ℕ∞) :=
          Finset.prod_le_prod fun i _ => (encard_leakAt_le _ _ _ _ _ _ _ _ _).trans (by exact_mod_cast hwin i)
      _ = ((∏ i, (2 ^ (A i).card * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * (2 ^ c i * Dj ^ k (sw i)))) :
            ℕ) : ℕ∞) :=
          (Nat.cast_prod _ _).symm
      _ ≤ _ := by exact_mod_cast hnat
  have hRi : ∀ i, tiedWidth (T i) ≤ R := fun i => Finset.le_sup (f := fun i => tiedWidth (T i)) (Finset.mem_univ i)
  have hX1 : ∀ g ∈ s, ∀ gt ∈ st, (X g fun i => tiedReach (T i) (famOf gt i)).encard ≤
      ((2 ^ (b * Fmax) * P0 * 2 ^ (R * Ft) : ℕ) : ℕ∞) := by
    intro g hg gt hgt
    have hle : ∑ i, (famOf gt i).card < Ft := (Finset.mem_filter.1 hgt).2
    have hA : ∑ i, (tiedReach (T i) (famOf gt i)).card ≤ R * Ft :=
      calc ∑ i, (tiedReach (T i) (famOf gt i)).card ≤ ∑ i, (famOf gt i).card * R :=
            sum_le_sum fun i _ => (card_tiedReach_le _ _).trans (Nat.mul_le_mul_left _ (hRi i))
        _ = (∑ i, (famOf gt i).card) * R := (sum_mul _ _ _).symm
        _ ≤ Ft * R := Nat.mul_le_mul_right _ hle.le
        _ = R * Ft := mul_comm _ _
    refine (hX g hg _).trans ?_
    exact_mod_cast Nat.mul_le_mul_left (2 ^ (b * Fmax) * P0) (Nat.pow_le_pow_right (by norm_num : (0 : ℕ) < 2) hA)
  have hcard_s : s.card ≤ ((∑ i, n i) + 1) ^ Fmax := by
    refine (Finset.card_le_univ s).trans_eq ?_
    rw [Fintype.card_fun, Fintype.card_option, Fintype.card_sigma, Fintype.card_fin]
    simp only [Fintype.card_fin]
  have hcard_st : st.card ≤ ((∑ i, (tiedReaches (T i)).length) + 1) ^ Ft := by
    refine (Finset.card_le_univ st).trans_eq ?_
    rw [Fintype.card_fun, Fintype.card_option, Fintype.card_sigma, Fintype.card_fin]
    simp only [Fintype.card_fin]
  have hY1 : Y.encard ≤ ((((∑ i, n i) + 1) ^ Fmax * (2 ^ (b * Fmax) * P0 * A1) : ℕ) : ℕ∞) := by
    have h : s.card * (st.card * (2 ^ (b * Fmax) * P0 * 2 ^ (R * Ft))) ≤
        ((∑ i, n i) + 1) ^ Fmax * (2 ^ (b * Fmax) * P0 * A1) :=
      calc s.card * (st.card * (2 ^ (b * Fmax) * P0 * 2 ^ (R * Ft)))
          ≤ ((∑ i, n i) + 1) ^ Fmax * (((∑ i, (tiedReaches (T i)).length) + 1) ^ Ft *
              (2 ^ (b * Fmax) * P0 * 2 ^ (R * Ft))) :=
            Nat.mul_le_mul hcard_s (Nat.mul_le_mul_right _ hcard_st)
        _ = ((∑ i, n i) + 1) ^ Fmax * (2 ^ (b * Fmax) * P0 * A1) := by rw [hA1]; ring
    calc Y.encard ≤ ∑ g ∈ s, (⋃ gt ∈ st, X g fun i => tiedReach (T i) (famOf gt i)).encard :=
          encard_biUnion_le s _
      _ ≤ ∑ g ∈ s, ∑ gt ∈ st, (X g fun i => tiedReach (T i) (famOf gt i)).encard :=
          sum_le_sum fun g _ => encard_biUnion_le st _
      _ ≤ ∑ _g ∈ s, ∑ _gt ∈ st, ((2 ^ (b * Fmax) * P0 * 2 ^ (R * Ft) : ℕ) : ℕ∞) :=
          sum_le_sum fun g hg => sum_le_sum fun gt hgt => hX1 g hg gt hgt
      _ = ((s.card * (st.card * (2 ^ (b * Fmax) * P0 * 2 ^ (R * Ft))) : ℕ) : ℕ∞) := by
          simp only [sum_const, nsmul_eq_mul] <;> push_cast <;> ring
      _ ≤ _ := by exact_mod_cast h
  have hY2 : Y2.encard ≤ ((((∑ i, n i) + 1) ^ Fmax * (2 ^ (b * Fmax) * P0 * A2) : ℕ) : ℕ∞) := by
    calc Y2.encard ≤ ∑ g ∈ s, (X g fun i => tiedSpan (T i)).encard := encard_biUnion_le s _
      _ ≤ ∑ _g ∈ s, ((2 ^ (b * Fmax) * P0 * A2 : ℕ) : ℕ∞) := sum_le_sum fun g hg => hX g hg _
      _ = ((s.card * (2 ^ (b * Fmax) * P0 * A2) : ℕ) : ℕ∞) := by
          simp only [sum_const, nsmul_eq_mul] <;> push_cast <;> ring
      _ ≤ _ := by exact_mod_cast Nat.mul_le_mul_right (2 ^ (b * Fmax) * P0 * A2) hcard_s
  have hY : Y.encard ≤ ((((∑ i, n i) + 1) ^ Fmax * (2 ^ (b * Fmax) * P0 * min A1 A2) : ℕ) : ℕ∞) := by
    rcases le_total A1 A2 with h | h
    · rw [min_eq_left h]
      exact hY1
    · rw [min_eq_right h]
      exact (Set.encard_le_encard hY12).trans hY2
  have hcap1 : 0 < ∏ c, (cap c + 1) := Finset.prod_pos fun c _ => Nat.succ_pos _
  have hroom : ((∑ i, n i) + 1) ^ Fmax * (2 ^ (b * Fmax) * P0 * min A1 A2) ≤
      (∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
        (∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i))) * min A1 A2 := by
    calc ((∑ i, n i) + 1) ^ Fmax * (2 ^ (b * Fmax) * P0 * min A1 A2)
        ≤ (∏ c, (cap c + 1)) * (((∑ i, n i) + 1) ^ Fmax * (2 ^ (b * Fmax) * P0 * min A1 A2)) :=
          Nat.le_mul_of_pos_left _ hcap1
      _ = _ := by simp only [hP0, hc]; ring
  calc G.encard ≤ (dec '' V).encard := Set.encard_le_encard hsub
    _ ≤ V.encard := Set.encard_image_le _ _
    _ ≤ (Ss ×ˢ Y).encard := Set.encard_image_le _ _
    _ = Ss.encard * Y.encard := Set.encard_prod
    _ ≤ _ := by
        gcongr
        exact hY.trans (by exact_mod_cast hroom)

/-! ### The monitoring period -/

/-- **The leak through the network certifier over a monitoring period**: `Guarantees.PooledLeakAccepted`, from
`pr_adaptive_add_le` at the scan's conditional escapes, a union bound over the tied audits' pooled bound and the leak
relations, and `encard_pooledGood_le` at each draw of the coins. -/
theorem horizonLeak : Guarantees.PooledLeakAccepted := by
  intro L K S lay k reg Dj m sw Pw ρ op T n b hb Ω _ _ Conn Cell Srv Dev Z _ arr f tap run htrust cap Ss hcap hfit Zs
    schedOf srvOf hs dec hdz acc hacc W hW hWn Wt e e' he q hq0 hq1 hesc Ft et hpool hleak δ hδ Fmax hF
  refine ⟨fun z hz => ?_, fun ω => encard_pooledGood_le lay k reg Dj sw Pw ρ op T n b hb arr f tap run htrust cap Ss
    hfit Zs (fun z => schedOf z ω) (fun z => srvOf z ω) (fun z hz => hs z hz ω) (dec ω) (fun z hz => hdz z hz ω)
    (fun z => W z ω) (fun z => Wt z ω) (fun z i => hWn z ω i) Fmax Ft⟩
  classical
  have hmiss := pr_adaptive_add_le (Ω := Ω) (r := 1 - q) (by linarith) (by linarith) m (acc z)
    (fun i ω => (W z ω i).card) e (fun i ω ω' h => hacc z i ω ω' h)
    (fun i ω ω' h => by simp only [hW z i ω ω' h]) he (fun i ω => hesc z hz i ω) Fmax
  have hpow : (1 - q) ^ Fmax ≤ δ := by
    have h1 : 1 - q ≤ Real.exp (-q) := by linarith [Real.add_one_le_exp (-q)]
    calc (1 - q) ^ Fmax ≤ Real.exp (-q) ^ Fmax := pow_le_pow_left₀ (by linarith) h1 Fmax
      _ = Real.exp (Fmax * (-q)) := (Real.exp_nat_mul _ _).symm
      _ ≤ Real.exp (-Real.log (1 / δ)) := Real.exp_le_exp.2 (by nlinarith)
      _ = δ := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
  calc Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
        ¬PooledGood lay k reg Dj sw Pw ρ op T Fmax Ft (W z ω) (Wt z ω) (run tap (schedOf z ω, srvOf z ω)).2)
      ≤ Pr (fun ω => (((∀ i, acc z i ω (ω i)) ∧ Fmax ≤ ∑ i, (W z ω i).card) ∨
            ((∀ i, acc z i ω (ω i)) ∧ Ft ≤ ∑ i, (Wt z ω i).card)) ∨
          ∃ i ∈ (univ : Finset (Fin m)), acc z i ω (ω i) ∧
            ¬LeakAt lay k (fun s => reg s \ tiedReach (T i) (Wt z ω i)) Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i)
              ((run tap (schedOf z ω, srvOf z ω)).2 i)) := by
        refine Pr_mono fun ω h => ?_
        obtain ⟨hall, hng⟩ := h
        unfold PooledGood at hng
        by_cases hlt : ∑ i, (W z ω i).card < Fmax
        · by_cases hle : ∑ i, (Wt z ω i).card < Ft
          · have : ¬∀ i, LeakAt lay k (fun s => reg s \ tiedReach (T i) (Wt z ω i)) Dj (sw i) (Pw i) (ρ i) (op i)
                (W z ω i) ((run tap (schedOf z ω, srvOf z ω)).2 i) := fun hl => hng ⟨hlt, hle, hl⟩
            push Not at this
            obtain ⟨i, hi⟩ := this
            exact Or.inr ⟨i, mem_univ i, hall i, hi⟩
          · exact Or.inl (Or.inr ⟨hall, not_lt.1 hle⟩)
        · exact Or.inl (Or.inl ⟨hall, not_lt.1 hlt⟩)
    _ ≤ _ := Pr_or_le _ _
    _ ≤ ((1 - q) ^ Fmax + ∑ i, e i + et) + ∑ i, e' i :=
        add_le_add ((Pr_or_le _ _).trans (add_le_add hmiss (hpool z hz)))
          ((Pr_exists_mem_le_sum univ _).trans (sum_le_sum fun i _ => hleak z hz i))
    _ ≤ δ + et + ∑ i, e i + ∑ i, e' i := by linarith

/-- **At no tied audit** `Guarantees.PooledLeakAccepted` is the statement without them: nothing is unpinned, no tied
proof block is incorrect, so the pooled bound holds at `F_t = 1` with error `0`, and the tied factor is `1`. -/
theorem horizonLeak_untied {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → ℕ → List (Obs L K S) → List Bool)
    (n : Fin m → ℕ) (b : ℕ) (hb : ∀ i, ∀ v < n i, faultyBits (sw i).prog (Pw i) {v} ≤ b)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → (Fin m → Ω) → WardenSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
    (hs : ∀ z ∈ Zs, ∀ ω, schedOf z ω ∈ Ss) (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (hdz : ∀ z ∈ Zs, ∀ ω, dec ω (run tap (schedOf z ω, srvOf z ω)).1 = z)
    (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (hacc : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → acc z i ω = acc z i ω')
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (hW : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → W z ω i = W z ω' i)
    (hWn : ∀ z ω i, W z ω i ⊆ range (n i)) (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i)
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hesc : ∀ z ∈ Zs, ∀ i ω, Pr (acc z i ω) ≤ (1 - q) ^ (W z ω i).card + e i)
    (hleak : ∀ z ∈ Zs, ∀ i, Pr (fun ω => acc z i ω (ω i) ∧
      ¬LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i) ((run tap (schedOf z ω, srvOf z ω)).2 i)) ≤ e' i)
    (δ : ℝ) (hδ : 0 < δ) (Fmax : ℕ) (hF : Real.log (1 / δ) ≤ q * Fmax) :
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧ ¬(∑ i, (W z ω i).card < Fmax ∧
        ∀ i, LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i) ((run tap (schedOf z ω, srvOf z ω)).2 i))) ≤
        δ + ∑ i, e i + ∑ i, e' i) ∧
      ∀ ω, {z | z ∈ Zs ∧ ∑ i, (W z ω i).card < Fmax ∧
          ∀ i, LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i) ((run tap (schedOf z ω, srvOf z ω)).2 i)}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
          ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  have hgood : ∀ (Wf : Fin m → Finset ℕ) (o : Fin m → List (Obs L K S)),
      PooledGood lay k reg Dj sw Pw ρ op (fun _ => []) Fmax 1 Wf (fun _ => ∅) o ↔
        ∑ i, (Wf i).card < Fmax ∧ ∀ i, LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (Wf i) (o i) := by
    intro Wf o
    simp [PooledGood, tiedReach_nil]
  obtain ⟨h1, h2⟩ := horizonLeak lay k reg Dj sw Pw ρ op (fun _ => []) n b hb arr f tap run htrust cap Ss hcap hfit
    Zs schedOf srvOf hs dec hdz acc hacc W hW hWn (fun _ _ _ => ∅) e e' he
    q hq0 hq1 hesc 1 0 (fun z _ => (Pr_mono fun ω h => by simp at h).trans_eq Pr_false)
    (fun z hz i => by simpa [tiedReach_nil] using hleak z hz i)
    δ hδ Fmax hF
  refine ⟨fun z hz => ?_, fun ω => ?_⟩
  · simpa [hgood] using h1 z hz
  · simpa [hgood, tiedReaches, tiedSpan_nil, sup_tiedWidth_nil] using h2 ω

end NetTiming
