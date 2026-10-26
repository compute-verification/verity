import Proofs.Flock.EndToEnd.PerWindow.Statement
import Proofs.Flock.EndToEnd.Period
import Proofs.Nci.Remote

/-!
# Sampled proofs, window by window: proof

`Flock.SecurityProofs.SamplingBlockPerWindow : Flock.Guarantees.SamplingBlockPerWindow`. A window's play reads only the
earlier windows' coins, so it is one play at every run that agrees before the window, and at every value of the
window's own coins (`past`, `Function.update_of_ne`): that gives the first two conclusions through the identifications.
For the other two the play is fixed, and each event over the window's coins lies inside an event of the play's outcome:
the draw misses the image of the window's faulty blocks under `unit` (of their number, `unit` injective on them),
bounded by `SamplingBlock.drawMiss_le` and `SamplingBlock.one_sub_le`; and the sampling block is accepted with a drawn
wrong unit, since a drawn faulty block that is not detected is accepted, bounded by
`SamplingBlock.accepted_drawnWrong_le`. `PlayOn.law` carries each to the game. A window with no block has no faulty
one: its bounds are `1` and `0`.

`Flock.SecurityProofs.SamplingBlockPerWindow_nontrivial`: with nothing detected and every block drawn, the last leg's
event holds at every value of window `t₀`'s coins, so its probability is `1` (`prob_add_not`), above `ks t₀`.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkExec Flock.Guarantees
open scoped ENNReal

private theorem dprob_nonneg {Ω : Type} [Fintype Ω] (μ : PMF Ω) (E : Ω → Prop) : 0 ≤ Nci.Dampener.prob μ E := by
  unfold Nci.Dampener.prob
  refine Finset.sum_nonneg fun ω _ => ?_
  split_ifs
  · exact ENNReal.toReal_nonneg
  · exact le_rfl

private theorem dprob_le_one {Ω : Type} [Fintype Ω] (μ : PMF Ω) (E : Ω → Prop) : Nci.Dampener.prob μ E ≤ 1 := by
  linarith [Nci.SecurityProofs.prob_add_not μ E, dprob_nonneg μ fun ω => ¬ E ω]

private theorem dprob_eq_zero {Ω : Type} [Fintype Ω] (μ : PMF Ω) {E : Ω → Prop} (h : ∀ ω, ¬ E ω) :
    Nci.Dampener.prob μ E = 0 := by
  unfold Nci.Dampener.prob
  exact Finset.sum_eq_zero fun ω _ => by simp [h ω]

private theorem dprob_eq_one {Ω : Type} [Fintype Ω] (μ : PMF Ω) {E : Ω → Prop} (h : ∀ ω, E ω) :
    Nci.Dampener.prob μ E = 1 := by
  linarith [Nci.SecurityProofs.prob_add_not μ E, dprob_eq_zero μ fun ω (hn : ¬ E ω) => hn (h ω)]

/-- **Sampled proofs, window by window** (`Flock.Guarantees.SamplingBlockPerWindow`). -/
theorem SamplingBlockPerWindow : Flock.Guarantees.SamplingBlockPerWindow := by
  intro T Ω _ _ β bad drawn detected win q ks hq hks hD
  have past : ∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) →
      (fun s : {s : Fin T // s < t} => ξ s.1) = fun s : {s : Fin T // s < t} => ξ' s.1 :=
    fun t ξ ξ' h => funext fun s => h s.1 s.2
  have upd : ∀ t (ξ : (t : Fin T) → Ω t) (a : Ω t) s, s < t → Function.update ξ t a s = ξ s :=
    fun t ξ a s hs => Function.update_of_ne (ne_of_lt hs) a ξ
  have hbad : ∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → ∀ b, win b = t →
      (b ∈ bad ξ ↔ b ∈ bad ξ') := by
    intro t ξ ξ' h b hb
    obtain ⟨D, hDt⟩ := hD t ⟨b, hb⟩
    obtain ⟨-, h1, -⟩ := hDt ξ
    obtain ⟨-, h1', -⟩ := hDt ξ'
    have hp : (D fun s => ξ s.1) = D fun s => ξ' s.1 := congrArg D (past t ξ ξ' h)
    rw [h1 b hb, h1' b hb, hp]
  refine ⟨hbad, ?_, ?_, ?_⟩
  · intro t ξ ξ' h b hb
    obtain ⟨D, hDt⟩ := hD t ⟨b, hb⟩
    obtain ⟨-, -, h2, h3, -⟩ := hDt ξ
    obtain ⟨-, -, h2', h3', -⟩ := hDt ξ'
    have hp : (D fun s => ξ s.1) = D fun s => ξ' s.1 := congrArg D (past t ξ ξ' fun s hs => h s hs.le)
    have ht : ξ t = ξ' t := h t le_rfl
    have hb' := hbad t ξ ξ' (fun s hs => h s hs.le) b hb
    constructor
    · rw [h2 b hb, h2' b hb, hp, ht]
    · rw [h3 b hb, h3' b hb, hp, ht, hb']
  · intro t ξ
    by_cases hex : ∃ b, win b = t
    · obtain ⟨D, hDt⟩ := hD t hex
      obtain ⟨P, hP⟩ : ∃ P, (D fun s => ξ s.1) = P := ⟨_, rfl⟩
      have hPa : ∀ a, (D fun s => Function.update ξ t a s.1) = P := fun a =>
        (congrArg D (past t (Function.update ξ t a) ξ (upd t ξ a))).trans hP
      obtain ⟨hinj, -, -, -, hA3, -, hqk, -⟩ := hDt ξ
      rw [hP] at hinj hA3 hqk
      set B := ((bad ξ).filter fun b => win b = t).image P.unit
      have hB : B.card = ((bad ξ).filter fun b => win b = t).card :=
        Finset.card_image_of_injOn fun b hb b' hb' he =>
          hinj b b' (Finset.mem_filter.1 (Finset.mem_coe.1 hb)).2 (Finset.mem_filter.1 (Finset.mem_coe.1 hb')).2 he
      have hev : ∀ a, (∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) →
          ¬ ∃ u ∈ (P.out a).2.1, u ∈ B := by
        rintro a ha ⟨u, hu, huB⟩
        obtain ⟨b, hbf, rfl⟩ := Finset.mem_image.1 huB
        obtain ⟨hbξ, hbt⟩ := Finset.mem_filter.1 hbf
        obtain ⟨-, -, h2, -⟩ := hDt (Function.update ξ t a)
        rw [hPa a, Function.update_self] at h2
        exact ha b hbξ hbt ((h2 b hbt).2 hu)
      rcases hq.eq_or_lt with hq0 | hq0
      · rw [← hq0, sub_zero, one_pow]
        exact dprob_le_one _ _
      have h1q : 0 ≤ 1 - q := by
        have : (P.w.kd : ℝ) / P.w.n ≤ 1 := by
          rcases Nat.eq_zero_or_pos P.w.n with h0 | h0
          · rw [h0, Nat.cast_zero, div_zero]
            exact zero_le_one
          · exact (div_le_one (by exact_mod_cast h0)).2 (by exact_mod_cast P.w.hkd)
        linarith
      calc Nci.Dampener.prob (PMF.uniformOfFintype (Ω t))
            (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a))
          ≤ Nci.Dampener.prob (PMF.uniformOfFintype (Ω t)) (fun a => ¬ ∃ u ∈ (P.out a).2.1, u ∈ B) :=
            Nci.SecurityProofs.prob_mono _ hev
        _ = (prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ B) P.w.game P.τ).toReal := P.law fun o => ¬ ∃ u ∈ o.2.1, u ∈ B
        _ ≤ (1 - q) ^ ((bad ξ).filter fun b => win b = t).card := by
            refine ENNReal.toReal_le_of_le_ofReal (pow_nonneg h1q _) ?_
            rw [ENNReal.ofReal_pow h1q, ← hB]
            exact (SamplingBlock.drawMiss_le P.w hA3 B P.τ).trans
              (pow_le_pow_left₀ zero_le (SamplingBlock.one_sub_le P.w hq0 hqk) _)
    · have h0 : ((bad ξ).filter fun b => win b = t).card = 0 := by
        rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
        intro b _ hb
        exact hex ⟨b, hb⟩
      rw [h0, pow_zero]
      exact dprob_le_one _ _
  · intro t ξ
    by_cases hex : ∃ b, win b = t
    · obtain ⟨D, hDt⟩ := hD t hex
      obtain ⟨P, hP⟩ : ∃ P, (D fun s => ξ s.1) = P := ⟨_, rfl⟩
      have hPa : ∀ a, (D fun s => Function.update ξ t a s.1) = P := fun a =>
        (congrArg D (past t (Function.update ξ t a) ξ (upd t ξ a))).trans hP
      obtain ⟨-, h1ξ, -, -, -, hcr, -, hks'⟩ := hDt ξ
      rw [hP] at h1ξ hcr hks'
      have hev : ∀ a, ((∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn (Function.update ξ t a)) ∧
          ∀ b ∈ bad ξ, win b = t → b ∉ detected (Function.update ξ t a)) →
          LiveAcceptsZK P.w.I (P.out a) ∧ ∃ u ∈ (P.out a).2.1, u ∈ P.w.wrongSet P.τ := by
        rintro a ⟨⟨b, hbξ, hbt, hbd⟩, hnd⟩
        obtain ⟨-, h1, h2, h3, -⟩ := hDt (Function.update ξ t a)
        rw [hPa a] at h1 h2 h3
        rw [Function.update_self] at h2 h3
        have hbw : P.unit b ∈ P.w.wrongSet P.τ := (h1ξ b hbt).1 hbξ
        refine ⟨?_, P.unit b, (h2 b hbt).1 hbd, hbw⟩
        by_contra hrej
        exact hnd b hbξ hbt ((h3 b hbt).2 ⟨(h1 b hbt).2 hbw, hrej⟩)
      calc Nci.Dampener.prob (PMF.uniformOfFintype (Ω t))
            (fun a => (∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn (Function.update ξ t a)) ∧
              ∀ b ∈ bad ξ, win b = t → b ∉ detected (Function.update ξ t a))
          ≤ Nci.Dampener.prob (PMF.uniformOfFintype (Ω t))
            (fun a => LiveAcceptsZK P.w.I (P.out a) ∧ ∃ u ∈ (P.out a).2.1, u ∈ P.w.wrongSet P.τ) :=
            Nci.SecurityProofs.prob_mono _ hev
        _ = (prob (fun o => LiveAcceptsZK P.w.I o ∧ ∃ u ∈ o.2.1, u ∈ P.w.wrongSet P.τ) P.w.game P.τ).toReal :=
            P.law fun o => LiveAcceptsZK P.w.I o ∧ ∃ u ∈ o.2.1, u ∈ P.w.wrongSet P.τ
        _ ≤ ks t := ENNReal.toReal_le_of_le_ofReal (hks t)
            ((SamplingBlock.accepted_drawnWrong_le P.w P.τ hcr).trans hks')
    · refine le_trans (le_of_eq (dprob_eq_zero _ ?_)) (hks t)
      rintro a ⟨⟨b, -, hbt, -⟩, -⟩
      exact hex ⟨b, hbt⟩

/-- **`SamplingBlockPerWindow`'s nontrivial lemma** (`Flock.Guarantees.SamplingBlockPerWindow_nontrivial`). -/
theorem SamplingBlockPerWindow_nontrivial : Flock.Guarantees.SamplingBlockPerWindow_nontrivial := by
  rintro T Ω _ _ β bad drawn detected win q ks t₀ hdet hdr hbad hks ⟨-, -, -, h4⟩
  let ξ : (t : Fin T) → Ω t := fun _ => Classical.arbitrary _
  obtain ⟨b, hb, hbt⟩ := hbad ξ
  have h := h4 t₀ ξ
  rw [dprob_eq_one _ fun a => ⟨⟨b, hb, hbt, hdr _ b⟩, fun b' _ _ => by simp [hdet]⟩] at h
  linarith

end Flock.SecurityProofs
