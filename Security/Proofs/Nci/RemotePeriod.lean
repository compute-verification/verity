import Definitions.Nci.Guarantees.RemotePeriod
import Proofs.Nci.Remote
import Proofs.Flock.Soundness.Audit.Periods

/-!
# Proof of the remote overhead bound over a monitoring period

`DampenerRemotePeriod` is `DampenerRemote`'s argument with each protocol's leg read window by window. A per-window bound
is a bound on the period's coins (`prob_le_of_update`: the run's average is, by `avg_update`, the average over the other
windows of window `t`'s own). Three events carry the failure:

* PoUW's wrong windows hold more than `ε' C`: `CertificateSlack` at the period's law, each window's marginal from its
  per-window bound, at most `p/ε'`;
* some window draws a faulty block and detects none: the union over the windows (`prob_exists_le`), at most `∑_t ks t`;
* no window does, and some identity has more than `K` escaped blocks: `escape_tail`, over the identities, at most
  `J (1 − q)^(K+1)`. Up to its first detection an identity's windows each miss all their faulty blocks, so the escaped
  blocks are the faulty blocks of windows that missed them; counted at a weight fixed before each window's coins, that
  of its faulty blocks while the identity is undetected and has at most `K` so far (`truncated_sum_ge`), the tail is
  `FlockSoundness.Audit.Periods.prCoin_forall_sum_le_pow`.

Outside them, `VerdictLag` and the escape share bound the faulty blocks' skipped work, and `remote_core` sums the
sources.

The witnesses, `DampenerRemotePeriod_honest` and `DampenerRemotePeriod_nontrivial`, are at constructed instances of one
window with no faulty proof block and no declared run, where each event holds on every run or on none. The `example` at
the end applies `DampenerRemotePeriod` to the nontrivial lemma's conjuncts, so they are its premises as it states them.
-/

namespace Nci.SecurityProofs

open Verity Nci.Dampener FlockSoundness.Audit.Periods
open FlockSoundness.Game (avg prCoin avg_mono avg_const prCoin_mono)
open scoped ENNReal

/-- Under the uniform law, `prob` is `prCoin` as a real number. -/
theorem prob_uniform_eq {α : Type} [Fintype α] [Nonempty α] (E : α → Prop) :
    prob (PMF.uniformOfFintype α) E = (prCoin E).toReal := by
  classical
  unfold prob prCoin avg
  rw [ENNReal.toReal_sum fun a _ => ?_]
  · refine Finset.sum_congr rfl fun a _ => ?_
    by_cases h : E a <;> simp [h, PMF.uniformOfFintype_apply]
  · by_cases h : E a <;> simp [h]

/-- The union bound over a finite family. -/
theorem prob_exists_le {Ω ι : Type} [Fintype Ω] (μ : PMF Ω) (s : Finset ι) (E : ι → Ω → Prop) :
    prob μ (fun ω => ∃ i ∈ s, E i ω) ≤ ∑ i ∈ s, prob μ (E i) := by
  classical
  unfold prob
  rw [Finset.sum_comm]
  refine Finset.sum_le_sum fun ω _ => ?_
  have hq : 0 ≤ (μ ω).toReal := ENNReal.toReal_nonneg
  by_cases h : ∃ i ∈ s, E i ω
  · obtain ⟨i, hi, hE⟩ := h
    have h' : ∃ i ∈ s, E i ω := ⟨i, hi, hE⟩
    simp only [h', ↓reduceIte]
    refine le_trans ?_ (Finset.single_le_sum (f := fun i => if E i ω then (μ ω).toReal else 0)
      (fun i _ => by split_ifs <;> simp [hq]) hi)
    simp [hE]
  · simp only [h, ↓reduceIte]
    exact Finset.sum_nonneg fun i _ => by split_ifs <;> simp [hq]

/-- A uniform coin's event has probability at most 1. -/
theorem prCoin_le_one {α : Type} [Fintype α] [Nonempty α] (E : α → Prop) : prCoin E ≤ 1 := by
  refine (prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans_eq ?_
  simp [prCoin, avg_const]

/-- Changing coordinate `t` of a run split at `t` is changing the split's first part. -/
theorem update_piSplitAt_symm {T : ℕ} {Ω : Fin T → Type} (t : Fin T) (a₀ a : Ω t) (ρ : ∀ j : {j // j ≠ t}, Ω j) :
    Function.update ((Equiv.piSplitAt t Ω).symm (a₀, ρ)) t a = (Equiv.piSplitAt t Ω).symm (a, ρ) := by
  funext j
  by_cases h : j = t
  · subst h
    simp [Equiv.piSplitAt]
  · simp [Equiv.piSplitAt, h]

section Coords

variable {T : ℕ} {Ω : Fin T → Type} [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]

/-- **One coordinate's tower rule**: averaging over the run is averaging, over the run, coordinate `t`'s average with
the others fixed. -/
theorem avg_update (t : Fin T) (F : ((t : Fin T) → Ω t) → ℝ≥0∞) :
    avg F = avg fun ξ => avg fun a : Ω t => F (Function.update ξ t a) := by
  classical
  have hΩ0 : (Fintype.card (Ω t) : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hΩt : (Fintype.card (Ω t) : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have h1 := ENNReal.mul_inv_cancel hΩ0 hΩt
  unfold avg
  rw [← (Equiv.piSplitAt t Ω).symm.sum_comp, ← (Equiv.piSplitAt t Ω).symm.sum_comp, Fintype.sum_prod_type,
    Fintype.sum_prod_type]
  simp only [update_piSplitAt_symm]
  conv_rhs => rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  calc (Fintype.card ((t : Fin T) → Ω t) : ℝ≥0∞)⁻¹ * F ((Equiv.piSplitAt t Ω).symm (x, y))
      = (Fintype.card ((t : Fin T) → Ω t) : ℝ≥0∞)⁻¹ *
          ((Fintype.card (Ω t) * (Fintype.card (Ω t) : ℝ≥0∞)⁻¹) * F ((Equiv.piSplitAt t Ω).symm (x, y))) := by
        rw [h1, one_mul]
    _ = _ := by ring

/-- **A per-window bound is a marginal bound**: an event of the run whose slice along coordinate `t` has probability at
most `p` at every value of the other coordinates has probability at most `p`. -/
theorem prCoin_le_of_update (t : Fin T) (E : ((t : Fin T) → Ω t) → Prop) {x : ℝ≥0∞}
    (h : ∀ ξ, prCoin (fun a : Ω t => E (Function.update ξ t a)) ≤ x) : prCoin E ≤ x := by
  unfold prCoin
  rw [avg_update t]
  refine (avg_mono fun ξ => h ξ).trans_eq (avg_const x)

/-- `prCoin_le_of_update` for `prob` under the uniform laws. -/
theorem prob_le_of_update (t : Fin T) (E : ((t : Fin T) → Ω t) → Prop) {p : ℝ}
    (h : ∀ ξ, prob (PMF.uniformOfFintype (Ω t)) (fun a => E (Function.update ξ t a)) ≤ p) :
    prob (PMF.uniformOfFintype ((t : Fin T) → Ω t)) E ≤ p := by
  have hp : 0 ≤ p := by
    obtain ⟨ξ⟩ : Nonempty ((t : Fin T) → Ω t) := inferInstance
    exact (by rw [prob_uniform_eq]; exact ENNReal.toReal_nonneg : (0 : ℝ) ≤ _).trans (h ξ)
  rw [prob_uniform_eq]
  refine ENNReal.toReal_le_of_le_ofReal hp (prCoin_le_of_update t E fun ξ => ?_)
  have := h ξ
  rw [prob_uniform_eq] at this
  exact (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top
    (prCoin_le_one _)) hp).2 this


/-- **The truncated count**: keeping each term while the sum before it is at most `K` keeps at least `K + 1` when the
total is at least `K + 1`. -/
theorem truncated_sum_ge {T : ℕ} (x : Fin T → ℕ) (K : ℕ) (hx : K + 1 ≤ ∑ t, x t) :
    K + 1 ≤ ∑ t, (if ∑ t' ∈ Finset.univ.filter (· < t), x t' ≤ K then x t else 0) := by
  classical
  by_cases hall : ∀ t, ∑ t' ∈ Finset.univ.filter (· < t), x t' ≤ K
  · refine hx.trans_eq (Finset.sum_congr rfl fun t _ => ?_)
    simp only [hall t, ↓reduceIte]
  · push Not at hall
    have hS : (Finset.univ.filter fun t => K < ∑ t' ∈ Finset.univ.filter (· < t), x t').Nonempty := by
      obtain ⟨t, ht⟩ := hall
      exact ⟨t, Finset.mem_filter.2 ⟨Finset.mem_univ _, ht⟩⟩
    obtain ⟨t0, ht0S, hmin⟩ := Finset.exists_min_image _ (fun t => t) hS
    have ht0 : K < ∑ t' ∈ Finset.univ.filter (· < t0), x t' := (Finset.mem_filter.1 ht0S).2
    have hlt : ∀ t, t < t0 → ∑ t' ∈ Finset.univ.filter (· < t), x t' ≤ K := fun t ht => by
      by_contra h
      exact absurd (hmin t (Finset.mem_filter.2 ⟨Finset.mem_univ _, not_le.1 h⟩)) (not_le.2 ht)
    calc K + 1 ≤ ∑ t' ∈ Finset.univ.filter (· < t0), x t' := ht0
      _ = ∑ t ∈ Finset.univ.filter (· < t0),
            (if ∑ t' ∈ Finset.univ.filter (· < t), x t' ≤ K then x t else 0) :=
          Finset.sum_congr rfl fun t ht => by simp only [hlt t (Finset.mem_filter.1 ht).2, ↓reduceIte]
      _ ≤ _ := Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)

/-- **The escape tail, one identity.** Under `DampenerRemotePeriod`'s sampled-proofs legs, the runs where no window
draws a faulty block without detecting one, and identity `j` has more than `K` escaped blocks (faulty, and accepted
before its first detection), have probability at most `(1 − q)^(K+1)`. Up to its first detection, the identity's windows
each miss all their faulty blocks, which, at a weight fixed before each window's coins (its faulty blocks while the
identity is undetected and has at most `K` escaped so far), is `prCoin_forall_sum_le_pow`. -/
theorem escape_tail {β : Type} (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T) {ι : Type}
    [DecidableEq ι] (idW : Fin T → ι) (τw : Fin T → ℝ) (q : ℝ) (K : ℕ) (j : ι)
    (hbadP : ∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → ∀ b, win b = t → (b ∈ bad ξ ↔ b ∈ bad ξ'))
    (hdrawnA : ∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s ≤ t → ξ s = ξ' s) → ∀ b, win b = t →
      (b ∈ drawn ξ ↔ b ∈ drawn ξ') ∧ (b ∈ detected ξ ↔ b ∈ detected ξ'))
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hmiss : ∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) ≤
        (1 - q) ^ ((bad ξ).filter fun b => win b = t).card)
    (hord : ∀ t t', idW t = idW t' → t < t' → τw t < τw t') :
    prob (PMF.uniformOfFintype ((t : Fin T) → Ω t)) (fun ξ =>
        (∀ t, ¬ ((∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn ξ) ∧ ∀ b ∈ bad ξ, win b = t → b ∉ detected ξ)) ∧
          K + 1 ≤ ((escaped (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b))).filter
            fun b => idW (win b) = j).card) ≤
      (1 - q) ^ (K + 1) := by
  classical
  -- window `t` detects one of its faulty blocks
  let D : Fin T → ((t : Fin T) → Ω t) → Prop := fun t ξ => ∃ b ∈ bad ξ, win b = t ∧ b ∈ detected ξ
  -- window `t`'s faulty blocks
  let m : Fin T → ((t : Fin T) → Ω t) → ℕ := fun t ξ => ((bad ξ).filter fun b => win b = t).card
  -- identity `j` is undetected before window `t`
  let st : Fin T → ((t : Fin T) → Ω t) → Prop := fun t ξ => ∀ t', t' < t → idW t' = j → ¬ D t' ξ
  let m' : Fin T → ((t : Fin T) → Ω t) → ℕ := fun t ξ => if idW t = j ∧ st t ξ then m t ξ else 0
  let cum : Fin T → ((t : Fin T) → Ω t) → ℕ := fun t ξ => ∑ t' ∈ Finset.univ.filter (· < t), m' t' ξ
  let wt : Fin T → ((t : Fin T) → Ω t) → ℕ := fun t ξ => if cum t ξ ≤ K then m' t ξ else 0
  let Miss : Fin T → ((t : Fin T) → Ω t) → Prop := fun t ξ => ∀ b ∈ bad ξ, win b = t → b ∉ drawn ξ
  let E : Fin T → ((t : Fin T) → Ω t) → Prop := fun t ξ => wt t ξ ≠ 0 → Miss t ξ
  -- what reads what
  have hm : ∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → m t ξ = m t ξ' := by
    intro t ξ ξ' h
    refine congrArg Finset.card (Finset.ext fun b => ?_)
    simp only [Finset.mem_filter]
    exact ⟨fun ⟨hb, hw⟩ => ⟨(hbadP t ξ ξ' h b hw).1 hb, hw⟩, fun ⟨hb, hw⟩ => ⟨(hbadP t ξ ξ' h b hw).2 hb, hw⟩⟩
  have hD : ∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s ≤ t → ξ s = ξ' s) → (D t ξ ↔ D t ξ') := by
    intro t ξ ξ' h
    have hb := hbadP t ξ ξ' fun s hs => h s hs.le
    constructor
    · rintro ⟨b, hb', hw, hd⟩
      exact ⟨b, (hb b hw).1 hb', hw, ((hdrawnA t ξ ξ' h b hw).2).1 hd⟩
    · rintro ⟨b, hb', hw, hd⟩
      exact ⟨b, (hb b hw).2 hb', hw, ((hdrawnA t ξ ξ' h b hw).2).2 hd⟩
  have hst : ∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → (st t ξ ↔ st t ξ') := by
    intro t ξ ξ' h
    have h' : ∀ t', t' < t → (D t' ξ ↔ D t' ξ') := fun t' ht' => hD t' ξ ξ' fun s hs => h s (hs.trans_lt ht')
    exact ⟨fun H t' ht' hj hd => H t' ht' hj ((h' t' ht').2 hd),
      fun H t' ht' hj hd => H t' ht' hj ((h' t' ht').1 hd)⟩
  have hm' : ∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → m' t ξ = m' t ξ' := by
    intro t ξ ξ' h
    simp only [m', hm t ξ ξ' h, hst t ξ ξ' h]
  have hwt : Predictable (β := fun _ => ℕ) wt := by
    intro t ξ ξ' h
    have hc : cum t ξ = cum t ξ' := Finset.sum_congr rfl fun t' ht' =>
      hm' t' ξ ξ' fun s hs => h s (hs.trans (Finset.mem_filter.1 ht').2)
    simp only [wt, hc, hm' t ξ ξ' h]
  have hE : Adapted (β := fun _ => Prop) E := by
    intro t ξ ξ' h
    have hw : wt t ξ = wt t ξ' := hwt t ξ ξ' fun s hs => h s hs.le
    have hb := hbadP t ξ ξ' fun s hs => h s hs.le
    have hd := hdrawnA t ξ ξ' h
    refine propext ⟨fun H hne b hb' hwb => ?_, fun H hne b hb' hwb => ?_⟩
    · exact fun hdr => H (hw ▸ hne) b ((hb b hwb).2 hb') hwb (((hd b hwb).1).2 hdr)
    · exact fun hdr => H (hw.symm ▸ hne) b ((hb b hwb).1 hb') hwb (((hd b hwb).1).1 hdr)
  -- each window's bound, given the past
  have hρ : ENNReal.ofReal (1 - q) ≤ 1 := ENNReal.ofReal_le_one.2 (by linarith)
  have hstep : ∀ t (ξ : (t : Fin T) → Ω t),
      prCoin (fun a : Ω t => E t (Function.update ξ t a)) ≤ ENNReal.ofReal (1 - q) ^ wt t ξ := by
    intro t ξ
    have hagree : ∀ a s, s < t → Function.update ξ t a s = ξ s := fun a s hs =>
      Function.update_of_ne (ne_of_lt hs) _ _
    have hwa : ∀ a, wt t (Function.update ξ t a) = wt t ξ := fun a => hwt t _ _ (hagree a)
    by_cases h0 : wt t ξ = 0
    · rw [h0, pow_zero]
      exact prCoin_le_one _
    · have hsub : prCoin (fun a : Ω t => E t (Function.update ξ t a)) ≤
          prCoin (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) := by
        refine prCoin_mono fun a ha b hb hwb => ?_
        exact ha ((hwa a).symm ▸ h0) b ((hbadP t _ _ (fun s hs => (hagree a s hs).symm) b hwb).1 hb) hwb
      have hle : wt t ξ ≤ m t ξ := by
        simp only [wt, m']
        split_ifs <;> simp
      refine hsub.trans ?_
      rw [← ENNReal.ofReal_pow (by linarith),
        ← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (prCoin_le_one _)), ← prob_uniform_eq]
      exact ENNReal.ofReal_le_ofReal ((hmiss t ξ).trans (pow_le_pow_of_le_one (by linarith) (by linarith) hle))
  have htail := prCoin_forall_sum_le_pow E wt hE hwt hρ hstep (K + 1)
  -- the event lies inside the tail's
  have hin : ∀ ξ : (t : Fin T) → Ω t,
      ((∀ t, ¬ ((∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn ξ) ∧ ∀ b ∈ bad ξ, win b = t → b ∉ detected ξ)) ∧
        K + 1 ≤ ((escaped (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b))).filter
          fun b => idW (win b) = j).card) → (∀ t, E t ξ) ∧ K + 1 ≤ ∑ t, wt t ξ := by
    rintro ξ ⟨hG, hesc⟩
    -- identity `j` is undetected up to window `t`
    let A : Fin T → Prop := fun t => idW t = j ∧ ∀ t', t' ≤ t → idW t' = j → ¬ D t' ξ
    have hfil : (escaped (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b))).filter
        (fun b => idW (win b) = j) = (bad ξ).filter fun b => A (win b) := by
      ext b
      simp only [Finset.mem_filter, mem_escaped, A]
      constructor
      · rintro ⟨⟨hb, hτ⟩, hj⟩
        refine ⟨hb, hj, fun t' ht' hj' ⟨b', hb', hw', hd'⟩ => ?_⟩
        have h1 := hτ b' hb' hd' (by rw [hw', hj', hj])
        rw [hw'] at h1
        rcases ht'.lt_or_eq with hlt | heq
        · exact lt_asymm h1 (hord t' (win b) (by rw [hj', hj]) hlt)
        · rw [heq] at h1
          exact lt_irrefl _ h1
      · rintro ⟨hb, hj, hA⟩
        refine ⟨⟨hb, fun b' hb' hd' hid => ?_⟩, hj⟩
        by_contra hle
        rcases lt_or_ge (win b) (win b') with hlt | hge
        · exact hle (hord (win b) (win b') hid.symm hlt)
        · exact hA (win b') hge (hid.trans hj) ⟨b', hb', rfl, hd'⟩
    have hcard : ((bad ξ).filter fun b => A (win b)).card = ∑ t, if A t then m t ξ else 0 := by
      rw [Finset.card_eq_sum_card_fiberwise (f := win) (t := Finset.univ) fun _ _ => Finset.mem_univ _]
      refine Finset.sum_congr rfl fun t _ => ?_
      by_cases hA : A t
      · simp only [hA, ↓reduceIte, m]
        congr 1
        ext b
        simp only [Finset.mem_filter]
        exact ⟨fun ⟨⟨hb, _⟩, hw⟩ => ⟨hb, hw⟩, fun ⟨hb, hw⟩ => ⟨⟨hb, hw ▸ hA⟩, hw⟩⟩
      · simp only [hA, ↓reduceIte]
        refine Finset.card_eq_zero.2 (Finset.filter_eq_empty_iff.2 fun b hb hw => ?_)
        exact hA (hw ▸ (Finset.mem_filter.1 hb).2)
    have hAm : ∀ t, A t → m' t ξ = m t ξ := fun t hA => by
      have h : idW t = j ∧ st t ξ := ⟨hA.1, fun t' ht' hj => hA.2 t' ht'.le hj⟩
      simp only [m', h, and_self, ↓reduceIte]
    have hsum : K + 1 ≤ ∑ t, m' t ξ := by
      rw [hfil, hcard] at hesc
      refine hesc.trans (Finset.sum_le_sum fun t _ => ?_)
      by_cases hA : A t
      · simp only [hA, ↓reduceIte, hAm t hA, le_refl]
      · simp only [hA, ↓reduceIte, zero_le]
    refine ⟨fun t hne => ?_, truncated_sum_ge (fun t => m' t ξ) K hsum⟩
    have hcum : cum t ξ ≤ K := by
      by_contra h
      exact hne (by simp only [wt, h, ↓reduceIte])
    have hm'0 : m' t ξ ≠ 0 := by
      intro h
      exact hne (by simp only [wt, h, ite_self])
    have hjt : idW t = j ∧ st t ξ := by
      by_contra h
      exact hm'0 (by simp only [m', h, ↓reduceIte])
    by_contra hmiss'
    have hDt : D t ξ := by
      have hdr : ∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn ξ := by
        by_contra h
        exact hmiss' fun b hb hw hd => h ⟨b, hb, hw, hd⟩
      by_contra hnd
      exact hG t ⟨hdr, fun b hb hw hd => hnd ⟨b, hb, hw, hd⟩⟩
    have hlow : ∑ t'', (if A t'' then m t'' ξ else 0) ≤ cum t ξ := by
      rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (· < t)]
      have hz : ∑ t'' ∈ Finset.univ.filter (fun t'' => ¬ t'' < t), (if A t'' then m t'' ξ else 0) = 0 := by
        refine Finset.sum_eq_zero fun t'' ht'' => ?_
        have hge : t ≤ t'' := not_lt.1 (Finset.mem_filter.1 ht'').2
        have hA : ¬ A t'' := fun hA => hA.2 t hge hjt.1 hDt
        simp only [hA, ↓reduceIte]
      rw [hz, add_zero]
      refine Finset.sum_le_sum fun t'' _ => ?_
      by_cases hA : A t''
      · simp only [hA, ↓reduceIte, hAm t'' hA, le_refl]
      · simp only [hA, ↓reduceIte, zero_le]
    rw [hfil, hcard] at hesc
    omega
  calc prob (PMF.uniformOfFintype ((t : Fin T) → Ω t)) _
      ≤ prob (PMF.uniformOfFintype ((t : Fin T) → Ω t)) (fun ξ => (∀ t, E t ξ) ∧ K + 1 ≤ ∑ t, wt t ξ) :=
        prob_mono _ hin
    _ = (prCoin (fun ξ : (t : Fin T) → Ω t => (∀ t, E t ξ) ∧ K + 1 ≤ ∑ t, wt t ξ)).toReal := prob_uniform_eq _
    _ ≤ (ENNReal.ofReal (1 - q) ^ (K + 1)).toReal :=
        ENNReal.toReal_mono (ENNReal.pow_ne_top ENNReal.ofReal_ne_top) htail
    _ = (1 - q) ^ (K + 1) := by rw [ENNReal.toReal_pow, ENNReal.toReal_ofReal (by linarith)]

end Coords

/-- **The remote overhead bound over a monitoring period.** Three events of the period's coins carry the failure:
PoUW's wrong windows hold more than `ε' C` (`CertificateSlack` on the run's law, each window's marginal from its
per-window bound, `prob_le_of_update`: at most `p/ε'`); some window draws a faulty block and detects none (the union
over windows: at most `∑_t ks t`); or no window does and some identity has more than `K` escaped blocks
(`escape_tail`, over the identities: at most `J (1 − q)^(K+1)`). Outside them the wrong windows hold at most `ε' C`,
each identity's escaped blocks skip at most `K B`, so `VerdictLag` puts the faulty blocks' skipped work at most
`(JΔ/H) C + J K B`, and the residual's hypothesis is `DampenerThreaded`'s at
`(1 − ε_W) W_cert − ε' C − (JΔ/H) C − J K B`; the declared runs sum their `Throttled` bounds under `Λ` and the
monitored compute `C`; `remote_core` gives both forms. -/
theorem DampenerRemotePeriod : Guarantees.DampenerRemotePeriod := by
  intro T Ω _ _ c wrong p ε β ι _ bad drawn detected win I idW τw w ks K Φ Δ H q B εW κ C U χ mfu W M Mu Mr Mx R _
    r d Th Γ α b_w b_a Λ dd e hp hε hc hcC hpouw hbadP hdrawnA hq0 hq1 hmiss hks hord hI hH hΔ hΦ hTD hcap hw hB0 hwB
    hκ hC hU hmfu hmfu1 hα hba hΛ hthr hbound hsize hM0 hM hu hres hx
  classical
  set μ := PMF.uniformOfFintype ((t : Fin T) → Ω t)
  -- window `t` draws a faulty block and detects none
  let Bad : Fin T → ((t : Fin T) → Ω t) → Prop := fun t ξ =>
    (∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn ξ) ∧ ∀ b ∈ bad ξ, win b = t → b ∉ detected ξ
  -- no window does, and identity `j` has more than `K` escaped blocks
  let Esc : ι → ((t : Fin T) → Ω t) → Prop := fun j ξ => (∀ t, ¬ Bad t ξ) ∧
    K + 1 ≤ ((escaped (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b))).filter
      fun b => idW (win b) = j).card
  have h1 : prob μ (fun ξ => ε * C < wrongCapacity c (fun t => wrong t ξ)) ≤ p / ε :=
    CertificateSlack _ μ T c (fun ξ t => wrong t ξ) p ε C hp hε hC hc hcC fun t =>
      prob_le_of_update t (fun ξ => wrong t ξ) (hpouw t)
  have h2 : prob μ (fun ξ => ∃ t ∈ Finset.univ, Bad t ξ) ≤ ∑ t, ks t := by
    refine (prob_exists_le μ Finset.univ Bad).trans (Finset.sum_le_sum fun t _ => prob_le_of_update t _ fun ξ => ?_)
    refine (prob_mono _ fun a ha => ?_).trans (hks t ξ)
    have hb := hbadP t (Function.update ξ t a) ξ fun s hs => Function.update_of_ne (ne_of_lt hs) _ _
    obtain ⟨⟨b, hb', hw', hd'⟩, hnd⟩ := ha
    exact ⟨⟨b, (hb b hw').1 hb', hw', hd'⟩, fun b' hb'' hw'' => hnd b' ((hb b' hw'').2 hb'') hw''⟩
  have h3 : prob μ (fun ξ => ∃ j ∈ I, Esc j ξ) ≤ I.card * (1 - q) ^ (K + 1) := by
    refine (prob_exists_le μ I Esc).trans ?_
    calc ∑ j ∈ I, prob μ (Esc j) ≤ ∑ _j ∈ I, (1 - q) ^ (K + 1) := Finset.sum_le_sum fun j _ =>
          escape_tail bad drawn detected win idW τw q K j hbadP hdrawnA hq0 hq1 hmiss hord
      _ = I.card * (1 - q) ^ (K + 1) := by rw [Finset.sum_const, nsmul_eq_mul]
  refine prob_ge_of_not μ (F := fun ξ => (ε * C < wrongCapacity c (fun t => wrong t ξ) ∨
    ∃ t ∈ Finset.univ, Bad t ξ) ∨ ∃ j ∈ I, Esc j ξ) (fun ξ hbad => ?_) ?_
  · by_contra hgood
    apply hbad
    have hwrong : wrongCapacity c (fun t => wrong t ξ) ≤ ε * C := not_lt.1 fun h => hgood (Or.inl (Or.inl h))
    have hG : ∀ t, ¬ Bad t ξ := fun t h => hgood (Or.inl (Or.inr ⟨t, Finset.mem_univ _, h⟩))
    have hesc : ∑ b ∈ escaped (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b)), w b ≤
        I.card * K * B := by
      have hmaps : ∀ b ∈ escaped (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b)), idW (win b) ∈ I :=
        fun b hb => hI ξ b (mem_escaped.1 hb).1
      rw [← Finset.sum_fiberwise_of_maps_to hmaps]
      let Ej : ι → Finset β := fun j =>
        (escaped (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b))).filter fun b => idW (win b) = j
      calc ∑ j ∈ I, ∑ b ∈ Ej j, w b
          ≤ ∑ _j ∈ I, (K : ℝ) * B := Finset.sum_le_sum fun j hj => by
            have hcard : (Ej j).card ≤ K := by
              by_contra h
              exact hgood (Or.inr ⟨j, hj, hG, Nat.succ_le_of_lt (not_le.1 h)⟩)
            calc ∑ b ∈ Ej j, w b ≤ ∑ _b ∈ Ej j, B :=
                  Finset.sum_le_sum fun b hb => hwB ξ b (mem_escaped.1 (Finset.mem_filter.1 hb).1).1
              _ = (Ej j).card * B := by rw [Finset.sum_const, nsmul_eq_mul]
              _ ≤ K * B := mul_le_mul_of_nonneg_right (by exact_mod_cast hcard) hB0
        _ = I.card * K * B := by rw [Finset.sum_const, nsmul_eq_mul, mul_assoc]
    have hlag := VerdictLag β ι (bad ξ) (detected ξ) I (fun b => idW (win b)) (fun b => τw (win b)) w Φ Δ H C hH hΔ
      hΦ (hI ξ) (hTD ξ) (hcap ξ)
    have hθ : (I.card * K * B / C : ℝ) * C = I.card * K * B := div_mul_cancel₀ _ hC.ne'
    have hr : (Mr ξ : ℝ) ≤ κ * (C - ((1 - εW) * W ξ - ε * C - lagShare I.card Δ H * C -
        (I.card * K * B / C) * C)) / 2 := by
      have h1 := hres ξ
      have h2 : κ * (C - (1 - εW) * W ξ + wrongCapacity c (fun t => wrong t ξ) + ∑ b ∈ bad ξ, w b) ≤
          κ * (C - ((1 - εW) * W ξ - ε * C - lagShare I.card Δ H * C - (I.card * K * B / C) * C)) :=
        mul_le_mul_of_nonneg_left (by linarith) hκ.le
      linarith
    have hdecl : ((∑ i, (Th i).M.card : ℕ) : ℝ) ≤ C / rhoT α b_a Λ := by
      unfold rhoT
      rw [div_div_eq_mul_div, le_div_iff₀ hba, Nat.cast_sum, Finset.sum_mul]
      calc ∑ i, ((Th i).M.card : ℝ) * b_a ≤ ∑ i, α * Λ * ((d i).D.size (d i).D.computed : ℝ) :=
            Finset.sum_le_sum fun i _ => by
              have h1 := hthr i
              have h2 : α * (Th i).bound (α * Γ) b_w dd e * (d i).D.size (d i).D.computed ≤
                  α * Λ * (d i).D.size (d i).D.computed :=
                mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (hbound i) hα) (Nat.cast_nonneg _)
              linarith
        _ = α * Λ * ∑ i, ((d i).D.size (d i).D.computed : ℝ) := by rw [Finset.mul_sum]
        _ ≤ α * Λ * C := mul_le_mul_of_nonneg_left hsize (mul_nonneg hα hΛ)
        _ = C * (α * Λ) := by ring
    have hMc : (M ξ : ℝ) ≤ Mu ξ + Mr ξ + ((∑ i, (Th i).M.card : ℕ) : ℝ) + Mx ξ := by exact_mod_cast hM ξ
    obtain ⟨h1, h2⟩ := remote_core hκ hC hU hmfu hmfu1 (Nat.cast_pos.2 (hM0 ξ)) hMc (hu ξ) hr hdecl (hx ξ)
    rw [residualShare_sub hC] at h1 h2
    exact ⟨h1, h2⟩
  · calc prob μ (fun ξ => (ε * C < wrongCapacity c (fun t => wrong t ξ) ∨ ∃ t ∈ Finset.univ, Bad t ξ) ∨
          ∃ j ∈ I, Esc j ξ)
        ≤ prob μ (fun ξ => ε * C < wrongCapacity c (fun t => wrong t ξ) ∨ ∃ t ∈ Finset.univ, Bad t ξ) +
            prob μ (fun ξ => ∃ j ∈ I, Esc j ξ) := prob_or_le _ _ _
      _ ≤ prob μ (fun ξ => ε * C < wrongCapacity c (fun t => wrong t ξ)) +
            prob μ (fun ξ => ∃ t ∈ Finset.univ, Bad t ξ) + prob μ (fun ξ => ∃ j ∈ I, Esc j ξ) :=
          add_le_add_left (prob_or_le _ _ _) _
      _ ≤ p / ε + I.card * (1 - q) ^ (K + 1) + ∑ t, ks t := by linarith

/-! ## The witnesses -/

/-- An event that holds on no run has probability 0. -/
theorem prob_of_forall_not {Ω : Type} [Fintype Ω] (μ : PMF Ω) {E : Ω → Prop} (h : ∀ ω, ¬ E ω) : prob μ E = 0 := by
  unfold prob
  exact Finset.sum_eq_zero fun ω _ => by simp [h ω]

theorem prob_uniform_le_one {α : Type} [Fintype α] [Nonempty α] (E : α → Prop) :
    prob (PMF.uniformOfFintype α) E ≤ 1 := by
  rw [prob_uniform_eq]
  exact ENNReal.toReal_le_of_le_ofReal zero_le_one (by simpa using prCoin_le_one E)

/-- **`DampenerRemotePeriod`'s premises hold where its conclusion says something** (`DampenerRemotePeriod_honest`): one
window run by one identity, its certificate right (`wrong` empty) at capacity `C = 2`, all of the monitored compute
certified (`W_cert = 2`), no faulty proof block and no declared run, and the run's one counted multiplication on its
unmonitored compute `U = 2` at `MFU₀ = 1/2`, `κ = 2`, at `p = 3 · 2⁻⁴⁰`, `ε' = 2p/10⁻⁶`, `q = 1` and `K = 0`. Then
`δ = 1/(2 · 10⁶)`, and the overhead bound on the hardware multiple is `1/(1/2 + ε' + 1/100)`, in `(1, 2]`, the run's
own multiple being 2. -/
theorem DampenerRemotePeriod_honest : Guarantees.DampenerRemotePeriod_honest := by
  refine ⟨1, fun _ => Bool, inferInstance, inferInstance, fun _ => 2, fun _ _ => False, 3 / 2 ^ 40,
    2 * (3 / 2 ^ 40) * 10 ^ 6, Empty, Unit, inferInstance, fun _ => ∅, fun _ => ∅, fun _ => ∅, fun b => b.elim,
    Finset.univ, fun _ => (), fun _ => 0, fun _ => 0, fun _ => 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 0, 1 / 2,
    fun _ => 2, fun _ => 1, fun _ => 1, fun _ => 0, fun _ => 0, Empty, inferInstance, fun i => i.elim,
    fun i => i.elim, fun i => i.elim, 0, 1, 0, 100, 1, 0, 0,
    by norm_num, by positivity, fun _ => by norm_num, by simp, fun t ξ => ?_,
    fun _ _ _ _ b => b.elim, fun _ _ _ _ b => b.elim, by norm_num, le_rfl,
    fun t ξ => by simpa using prob_uniform_le_one _,
    fun t ξ => by rw [prob_of_forall_not _ fun _ h => by simp at h],
    fun t t' _ h => absurd h (by rw [Subsingleton.elim t t']; exact lt_irrefl _), by simp, one_pos, le_rfl,
    by norm_num, by simp [Assumptions.TerminalDetection], by simp, fun _ => le_rfl, le_rfl, by simp,
    two_pos, two_pos, zero_le_two, by norm_num, by norm_num, zero_le_one, by norm_num, zero_le_one,
    fun i => i.elim, fun i => i.elim, by simp,
    fun _ => Nat.one_pos, fun _ => by simp, fun _ => by norm_num, fun _ => by simp [wrongCapacity],
    fun _ => by norm_num, by norm_num [Finset.card_univ], fun ξ => ?_⟩
  · rw [prob_of_forall_not _ fun _ h => h]
    positivity
  · simp only [remoteDampener, hardwareMultiple, efficiency, unmonitoredShare, residualShare, lagShare,
      declaredShare, rhoT, Finset.card_univ, Fintype.card_unit]
    norm_num

/-- **`DampenerRemotePeriod` is false without PoUW's per-window bound** (`DampenerRemotePeriod_nontrivial`), at
`AcceptAll`: the window's certificate is wrong on every run, so PoUW's per-window probability is 1, above `p = 0`. The
residual counts the wrong capacity, so the run's one counted multiplication (`M_r = 1`, `κ = 2`, `MFU₀ = 1`) has compute
efficiency 1, above the bound `51/100`, and hardware multiple 1, below the overhead bound `100/51`, on every run, while
`δ = 0`. -/
theorem DampenerRemotePeriod_nontrivial : Guarantees.DampenerRemotePeriod_nontrivial := by
  refine ⟨le_rfl, by norm_num, fun _ => zero_le_one, by simp, fun _ _ _ _ b => b.elim, fun _ _ _ _ b => b.elim,
    zero_le_one, le_rfl, fun t ξ => by simpa using prob_uniform_le_one _,
    fun t ξ => by rw [prob_of_forall_not _ fun _ h => by simp at h],
    fun t t' _ h => absurd h (by rw [Subsingleton.elim t t']; exact lt_irrefl _), by simp, one_pos, le_rfl,
    by norm_num, by simp [Assumptions.TerminalDetection], by simp, fun _ => le_rfl, le_rfl, by simp,
    two_pos, one_pos, le_rfl, one_pos, le_rfl, zero_le_one, by norm_num, zero_le_one, fun i => i.elim,
    fun i => i.elim, by simp, fun _ => Nat.one_pos, fun _ => by simp, fun _ => by norm_num,
    fun _ => by simp [wrongCapacity], fun _ => by norm_num, fun h => ?_, ?_⟩
  · have h1 := h 0 fun _ => true
    have h2 := prob_add_not (PMF.uniformOfFintype Bool)
      fun a => Guarantees.AcceptAll.wrong 0 (Function.update (fun _ => true) 0 a)
    have h3 : prob (PMF.uniformOfFintype Bool)
        (fun a => ¬ Guarantees.AcceptAll.wrong 0 (Function.update (fun _ => true) 0 a)) = 0 :=
      prob_of_forall_not _ fun _ h => h trivial
    linarith
  · rw [prob_of_forall_not _ fun ξ hG => by
      have h2 := hG.2
      simp only [remoteDampener, hardwareMultiple, efficiency, unmonitoredShare, residualShare, lagShare,
        declaredShare, rhoT, Guarantees.AcceptAll.M, Guarantees.AcceptAll.W] at h2
      norm_num at h2]
    norm_num [Guarantees.AcceptAll.ks]

/-! `DampenerRemotePeriod_nontrivial`'s conjuncts are `DampenerRemotePeriod`'s premises as it states them at
`AcceptAll`: with PoUW's per-window bound they give the conclusion the last conjunct refutes. -/

open Guarantees.AcceptAll in
example (h5 : ∀ (t : Fin 1) (ξ : Fin 1 → Bool),
    prob (PMF.uniformOfFintype Bool) (fun a => wrong t (Function.update ξ t a)) ≤ 0) : False := by
  obtain ⟨h1, h2, h3, h4, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23,
    h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, -, hnot⟩ := DampenerRemotePeriod_nontrivial
  exact hnot (DampenerRemotePeriod 1 (fun _ => Bool) c wrong 0 (1 / 2) Empty Unit bad drawn detected win ∅ idW τw w
    ks 0 0 0 1 1 0 0 2 1 0 0 1 W M Mu Mr Mx Empty r d Th 0 1 0 100 1 0 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13
    h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24 h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37)

end Nci.SecurityProofs
