import Proofs.Accounting.PoubHeld.OverTime

/-!
# Held over time from any start in the first group's window

`heldOverLinks_of` (`Proofs.Accounting.PoubHeld.OverTime`) bounds what a paced session held over its credited span
`[e_0, s_C)`. Its argument bounds the absent ticks over `[s_0, s_C)`, so it bounds what was held over `[u, s_C)` for any
`u` in `[s_0, e_0]`: `heldOverWindow_of`, the same argument with `u` for `e_0`, at one outcome and with the paced
session's facts given there rather than through `Paced` and `Covers`. The memory line's stretches after the first start
at their first group's coin `s_lo`, before their paced session's `e_lo` (`Properties.PoubCheck.Line.LineHeld`).
-/

namespace Accounting.SecurityProofs.Poub

open Finset Accounting.Poub.Held Accounting.Poub.Held.Assumptions Pouw Absence

variable {Ω : Type} {T : Trace Ω}

/-- **Held over time from `u ∈ [s_0, e_0]`**: `heldOverLinks_of`'s bound over `[u, s_C)` at one outcome, from the
paced session's facts there (`Paced`'s, but the ticks' order, and `Covers`'). -/
theorem heldOverWindow_of [Fintype Ω] (K : Pace Ω) (ω : Ω) {S : ℕ} {s E β βl Br Bd Bl W : ℝ} (hE : 0 < E)
    (hs : 0 ≤ s) (hβ : 0 ≤ β) (hβl : 0 ≤ βl) (hl : Links T βl Bl) (hsq : SquaringCeiling T s E Br)
    (hd : Disks T β Bd) (hS : ChipCeiling T S) (hr : Routed T)
    (hfP : K.first ω (K.credited + 1) ≤ T.passes) (hfg : ∀ g ≤ K.credited, K.first ω g < K.first ω (g + 1))
    (hrel : Monotone (K.released ω)) (hre : ∀ g ≤ K.credited, K.released ω (g + 1) ≤ K.ended ω g)
    (hpass : ∀ g ≤ K.credited, ∀ p, K.first ω g ≤ p → p < K.first ω (g + 1) →
      K.released ω g ≤ T.tick ω p ∧ T.tick ω (p + 1) ≤ K.ended ω g)
    (hplan : ∀ p < T.passes, T.plan ω p = univ) {u : ℕ} (hu0 : K.released ω 0 ≤ u) (hu1 : u ≤ K.ended ω 0)
    (hk : ∀ p < T.passes, ∀ j ∈ T.correct ω p,
      ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t)
    (hw : (T.wrong ω : ℝ) ≤ W) :
    ((K.stop ω - u : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω *
              ((s / E + β + βl) * (((K.stop ω - u : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd + Bl))
          - K.fill ω T.blocks W
        ≤ T.heldOver ω u (K.stop ω) := by
  rw [show K.stop ω = K.released ω K.credited from rfl]
  set C := K.credited with hCdef
  set L := K.longest ω with hLdef
  set rel := K.released ω with hreldef
  set en := K.ended ω with hendef
  set N := T.blocks with hNdef
  have hL : ∀ g ≤ C, en g ≤ rel g + L := by
    intro g hg
    have h1 : en g - rel g ≤ L :=
      le_sup (f := fun g => K.ended ω g - K.released ω g) (mem_range.2 (Nat.lt_succ_of_le hg))
    have h2 := (hrel (Nat.le_succ g)).trans (hre g hg)
    omega
  have hse : ∀ g < C, rel (g + 1) ≤ en g := fun g hg => hre g hg.le
  have h01 : rel 0 ≤ en 0 := (hrel (Nat.zero_le 1)).trans (hre 0 (Nat.zero_le C))
  have h0C : rel 0 ≤ rel C := hrel (Nat.zero_le C)
  -- each block's absence
  have hsum := sum_le_sum fun j (_ : j ∈ (univ : Finset (Fin N))) =>
    absent_le (fun t => j ∈ T.present ω t) C L rel en hrel hse hL
  rw [sum_add_distrib, sum_add_distrib, ← mul_sum] at hsum
  -- the entries, summed over the blocks
  have hent : ∑ j : Fin N, (entries (fun t => j ∈ T.present ω t) (rel 0) (rel C + L)).card =
      ∑ τ ∈ Ico (rel 0) (rel C + L), T.entered ω τ := by
    simp only [entries, card_filter]
    rw [sum_comm]
    refine sum_congr rfl fun τ _ => ?_
    have : T.present ω (τ + 1) \ T.present ω τ =
        univ.filter fun j => ¬ j ∈ T.present ω τ ∧ j ∈ T.present ω (τ + 1) := by
      ext j
      simp [and_comm]
    rw [Trace.entered, this, card_filter]
  have hin : ∑ τ ∈ Ico (rel 0) (rel C + L), T.entered ω τ ≤
      over T.rebuilt ω (rel 0) (rel C + L) + over T.disk ω (rel 0) (rel C + L) +
        over T.link ω (rel 0) (rel C + L) := by
    have h := sum_le_sum fun τ (_ : τ ∈ Ico (rel 0) (rel C + L)) => hr ω τ
    simp only [sum_add_distrib] at h
    exact h
  -- the groups whose windows a block is absent from throughout
  set nw : ℕ → ℕ := fun g =>
    (univ.filter fun j : Fin N => ¬ ∃ w ∈ Ico (rel g) (en g), j ∈ T.present ω w).card with hnw
  have hcost1 : ∑ j : Fin N, ∑ g ∈ range C,
      (if ∃ w ∈ Ico (rel g) (en g), j ∈ T.present ω w then 0 else rel (g + 1) - rel g) =
      ∑ g ∈ range C, nw g * (rel (g + 1) - rel g) := by
    rw [sum_comm]
    refine sum_congr rfl fun g _ => ?_
    rw [sum_ite, sum_const_zero, zero_add, sum_const, smul_eq_mul]
  have hcost2 : ∑ j : Fin N, (if ∃ w ∈ Ico (rel C) (en C), j ∈ T.present ω w then 0 else L) = nw C * L := by
    rw [sum_ite, sum_const_zero, zero_add, sum_const, smul_eq_mul]
  rw [hcost1, hcost2, hent] at hsum
  -- their wrong receipts
  have hnw_le : ∀ g ≤ C, ∀ p, K.first ω g ≤ p → p < K.first ω (g + 1) →
      nw g ≤ (T.plan ω p \ T.correct ω p).card := by
    intro g hg p hp1 hp2
    have hpP : p < T.passes :=
      lt_of_lt_of_le hp2 ((le_of_steps (K.first ω) (fun g hg => (hfg g (by omega)).le) (by omega) le_rfl).trans hfP)
    refine card_le_card fun j hj => ?_
    have hj' := (mem_filter.1 hj).2
    rw [hplan p hpP]
    refine mem_sdiff.2 ⟨mem_univ j, fun hc => hj' ?_⟩
    obtain ⟨t, ht1, ht2, ht⟩ := hk p hpP j hc
    obtain ⟨hp3, hp4⟩ := hpass g hg p hp1 hp2
    exact ⟨t, mem_Ico.2 ⟨by omega, by omega⟩, ht⟩
  have hwrong : ∑ g ∈ range (C + 1), K.size ω g * nw g ≤ T.wrong ω := by
    calc ∑ g ∈ range (C + 1), K.size ω g * nw g
        ≤ ∑ g ∈ range (C + 1), ∑ p ∈ Ico (K.first ω g) (K.first ω (g + 1)), (T.plan ω p \ T.correct ω p).card := by
          refine sum_le_sum fun g hg => ?_
          rw [Pace.size, ← Nat.card_Ico, ← smul_eq_mul, ← sum_const]
          exact sum_le_sum fun p hp =>
            hnw_le g (Nat.lt_succ_iff.1 (mem_range.1 hg)) p (mem_Ico.1 hp).1 (mem_Ico.1 hp).2
      _ = ∑ p ∈ Ico (K.first ω 0) (K.first ω (C + 1)), (T.plan ω p \ T.correct ω p).card :=
          sum_groups _ _ fun g hg => (hfg g (by omega)).le
      _ ≤ T.wrong ω := sum_le_sum_of_subset fun p hp => mem_range.2 ((mem_Ico.1 hp).2.trans_le hfP)
  -- the knapsack's point
  have hrate : ∀ g, 0 ≤ K.rate ω g := fun g => by
    unfold Pace.rate
    split_ifs <;> positivity
  have hsize : ∀ g ≤ C, (K.size ω g : ℝ) ≠ 0 := fun g hg => by
    have := hfg g hg
    rw [Pace.size, Nat.cast_ne_zero]
    omega
  have hfill : ((∑ g ∈ range C, nw g * (rel (g + 1) - rel g) + nw C * L : ℕ) : ℝ) ≤ K.fill ω N W := by
    refine le_csSup ⟨∑ g ∈ range (C + 1), K.rate ω g * (K.size ω g * N : ℕ), ?_⟩
      ⟨fun g => ((K.size ω g * nw g : ℕ) : ℝ), ⟨fun g => ⟨Nat.cast_nonneg _, ?_⟩, ?_⟩, ?_⟩
    · rintro v ⟨y, ⟨hy, -⟩, rfl⟩
      exact sum_le_sum fun g _ => mul_le_mul_of_nonneg_left (hy g).2 (hrate g)
    · exact Nat.cast_le.2 (Nat.mul_le_mul_left _ ((card_filter_le _ _).trans (card_fin N).le))
    · rw [← Nat.cast_sum]
      exact (Nat.cast_le.2 hwrong).trans hw
    · simp only
      rw [sum_range_succ]
      push_cast
      congr 1
      · refine sum_congr rfl fun g hg => ?_
        have hgC := mem_range.1 hg
        rw [Pace.rate, ite_eq_left hgC]
        exact div_mul_cancel_mid (hsize g hgC.le)
      · rw [Pace.rate, ite_eq_right (lt_irrefl C)]
        exact div_mul_cancel_mid (hsize C le_rfl)
  -- each tick: its blocks in node memory, on a chip and absent
  have hpt : ∀ t, N ≤ (T.mem ω t).card + S + (univ \ T.present ω t).card := by
    intro t
    have h1 : (univ : Finset (Fin N)).card ≤ (univ \ T.present ω t).card + (T.present ω t).card :=
      card_le_card_sdiff_add_card
    have h2 : (T.present ω t).card ≤ (T.mem ω t).card + (T.chip ω t).card := card_union_le _ _
    have h3 := hS ω t
    rw [card_fin] at h1
    omega
  have hheld : (rel C - u) * N ≤ T.heldOver ω u (rel C) + (rel C - u) * S +
      ∑ t ∈ Ico u (rel C), (univ \ T.present ω t).card := by
    have h := sum_le_sum fun t (_ : t ∈ Ico u (rel C)) => hpt t
    rw [sum_add_distrib, sum_add_distrib, sum_const, sum_const, Nat.card_Ico, smul_eq_mul, smul_eq_mul] at h
    exact h
  have habs : ∑ t ∈ Ico u (rel C), (univ \ T.present ω t).card ≤
      ∑ j : Fin N, ((Ico (rel 0) (rel C)).filter fun t => ¬ j ∈ T.present ω t).card := by
    refine (sum_le_sum_of_subset (Ico_subset_Ico hu0 le_rfl)).trans_eq ?_
    simp only [card_filter]
    rw [sum_comm]
    refine sum_congr rfl fun t _ => ?_
    rw [show univ \ T.present ω t = univ.filter fun j => ¬ j ∈ T.present ω t by ext j; simp, card_filter]
  -- the window the entries are counted over
  have hlen : ((rel C + L - rel 0 : ℕ) : ℝ) ≤ ((rel C - u : ℕ) : ℝ) + 2 * L := by
    have := hL 0 (Nat.zero_le C)
    have h : rel C + L - rel 0 ≤ (rel C - u) + 2 * L := by omega
    exact_mod_cast h
  have h0L : rel 0 ≤ rel C + L := h0C.trans (Nat.le_add_right _ _)
  have hreb := hsq ω (rel 0) (rel C + L) h0L
  have hdisk := hd ω (rel 0) (rel C + L) h0L
  have hlink := hl ω (rel 0) (rel C + L) h0L
  -- in the reals
  have hheldR : ((rel C - u : ℕ) : ℝ) * N ≤ T.heldOver ω u (rel C) + ((rel C - u : ℕ) : ℝ) * S +
      ((∑ t ∈ Ico u (rel C), (univ \ T.present ω t).card : ℕ) : ℝ) := by
    exact_mod_cast hheld
  have hA : ((∑ t ∈ Ico u (rel C), (univ \ T.present ω t).card : ℕ) : ℝ) ≤
      ((∑ g ∈ range C, nw g * (rel (g + 1) - rel g) + nw C * L : ℕ) : ℝ) +
        2 * L * ((over T.rebuilt ω (rel 0) (rel C + L) : ℝ) + over T.disk ω (rel 0) (rel C + L) +
          over T.link ω (rel 0) (rel C + L)) := by
    have h := habs.trans (hsum.trans (Nat.add_le_add_left (Nat.mul_le_mul_left _ hin) _))
    exact_mod_cast h
  have hX : 0 ≤ s / E + β + βl := add_nonneg (add_nonneg (div_nonneg hs hE.le) hβ) hβl
  have hL0 : (0 : ℝ) ≤ 2 * L := by positivity
  have hO : (over T.rebuilt ω (rel 0) (rel C + L) : ℝ) + over T.disk ω (rel 0) (rel C + L) +
      over T.link ω (rel 0) (rel C + L) ≤
      (s / E + β + βl) * (((rel C - u : ℕ) : ℝ) + 2 * L) + (Br + Bd + Bl) := by
    have := mul_le_mul_of_nonneg_left hlen hX
    nlinarith
  have hO2 := mul_le_mul_of_nonneg_left hO hL0
  linarith

end Accounting.SecurityProofs.Poub
