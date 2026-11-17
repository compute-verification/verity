module

public import Proofs.Pous.Lemmas.PeriodDeployed
public import Proofs.Pous.Period
public import Proofs.Pous.P2.RingTrunc

/-! The proofs of PoUS's per-period guarantee at the deployed ring budget, which lies outside both regions where
`P2RingAuditSecure` is refuted, and of its honest and nontrivial lemmas. -/

@[expose] public section

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties Pous.P2.Assumptions
open scoped ENNReal

set_option exponentiation.threshold 20000

theorem periodHeldDeployed : PeriodHeldDeployed := by
  intro hP
  exact periodHeld RING_D RING_Q RING_L hP

theorem limit_ns : LIMIT_NS = 854000 := rfl

theorem deadline_rounds : DEADLINE_ROUNDS = 8243 := by
  unfold DEADLINE_ROUNDS
  rw [limit_ns, Nat.floor_eq_iff (by norm_num [ROUND_NS])]
  norm_num [ROUND_NS]

theorem rootProg_not_ringBounded (e : ℕ) (a : Bits 16448) (Q L : ℕ) :
    ¬ (PousP2Ring.rootProg 16446 e a : OracleAlgorithm (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)) _).RingBounded RING_D Q L := by
  intro h
  obtain ⟨ω⟩ := (inferInstance : Nonempty (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω)
  have h1 := (h ω).1
  rw [(PousP2Ring.rootProg_cost (B := 2 ^ 19) (p := 2 ^ 16448 - 21065) _ 16446 e a).1] at h1
  have h2 : 16446 + 1 ≤ 16446 := h1
  omega

theorem not_p2RingAuditSecure_trunc (hP : PrimeP16448) (D Q L : ℕ) (hD : 2 ≤ D) (hQ : 2 ≤ Q) (hL : 2 ^ 169 ≤ L) :
    ¬ P2RingAuditSecure D Q L := by
  intro hsec
  have hS : 501819 * (16448 - 168) ≤ (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ := by
    rw [PousP2Ring.stateBound_p2ring_b19]
    norm_num
  let A₂ : SeqResponderOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432)
      ((p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) D Q L :=
    ⟨(PousP2Ring.truncSeq (b := 16432) 168 hS).step, fun W pp j σ i =>
      PousP2Ring.ringBounded_mono ((PousP2Ring.truncSeq (b := 16432) 168 hS).bounded W pp j σ i) hD hQ
        (le_trans (by norm_num) hL)⟩
  have h := (PousP2Ring.trunc_seq_accepted (b := 16432) hP PousP2Ring.p16448_mod_four PousP2p.hyps_16448.2.1 105
    (by norm_num) (by norm_num) hS).trans (hsec _ _ A₂)
  have hL' : (((501819 : ℕ) : ℝ≥0∞) / ((2 ^ 19 : ℕ) : ℝ≥0∞)) ^ 105 =
      ENNReal.ofReal (((501819 : ℝ) / 2 ^ 19) ^ 105) := by
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast (2 ^ 19), ← ENNReal.ofReal_div_of_pos (by norm_num),
      ← ENNReal.ofReal_pow (by positivity)]
    norm_num
  have hR : ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) =
      ENNReal.ofReal (1 / 100 + ((2⁻¹ : ℝ) ^ 129 + (2⁻¹ : ℝ) ^ 130)) := by
    rw [ENNReal.ofReal_add (by norm_num) (by positivity), ENNReal.ofReal_add (by positivity) (by positivity),
      ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_inv_of_pos (by norm_num),
      ENNReal.ofReal_ofNat]
    simp only [Params.δ]
    norm_num
  rw [hL', hR, ENNReal.ofReal_le_ofReal_iff (by positivity)] at h
  have hlt : 1 / 100 + ((2⁻¹ : ℝ) ^ 129 + (2⁻¹ : ℝ) ^ 130) < ((501819 : ℝ) / 2 ^ 19) ^ 105 := by norm_num
  exact absurd h (not_le.2 hlt)

theorem S_eq : Pous.EndToEnd.S = 8169621180 := PousP2Ring.stateBound_p2ring_b19 _ _

theorem held_periodRun (m : ℕ) (hm : m ≤ FULL)
    (ω : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) :
    (periodRun m hm).held ω = K * m := by
  simp [ResourceAccounting.Run.held, ResourceAccounting.Schedule.held, ResourceAccounting.Window.ticks, periodRun]

theorem periodHeldHonest : PeriodHeldHonest := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro P hs
    have h0 : (0 : ℕ) ∈ (honestRun.win .capacity).ticks := by
      simp [honestRun, periodRun, ResourceAccounting.Window.ticks, K]
    exact periodHeld_honest honestRun P RING_D RING_Q RING_L 1 hs fun _ => ⟨0, h0, le_rfl⟩
  · show (0 : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ)
    norm_num [HONEST_S, K]
  · rw [S_eq]
    show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
      ((8169621180 : ℕ) : ℝ)
    norm_num [HONEST_S, K]
  · rw [deadline_rounds]
    show 8243 ≤ 16446
    omega
  · rw [limit_ns]
    norm_num [RING_D, ROUND_NS]

theorem εMax_toReal : Params.εMax.toReal = (2⁻¹ : ℝ) ^ 128 := by
  simp [Params.εMax, Params.secParam, ENNReal.toReal_pow, ENNReal.toReal_inv]

theorem periodHeldNontrivial : PeriodHeldNontrivial := by
  refine ⟨fun P o hb h => ?_, fun P o x ⟨_, _, hh⟩ => ?_⟩
  · have h1 : Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
        True ∧ ¬ P.Break (o.1, o.2, x) ∧
          (emptyRun.held (o.1, o.2, x) : ℝ) < HONEST_S * (emptyRun.win .capacity).len) = 1 := by
      refine Pouw.pr_eq_one fun x => ⟨trivial, hb x, ?_⟩
      have he : emptyRun.held (o.1, o.2, x) = K * 0 := held_periodRun 0 _ _
      rw [he]
      show ((K * 0 : ℕ) : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ)
      norm_num [HONEST_S, K]
    have h2 := h1.symm.le.trans h
    rw [εMax_toReal] at h2
    norm_num [Params.δ] at h2
  · have he : honestRun.held (o.1, o.2, x) = K * FULL := held_periodRun FULL le_rfl _
    rw [he] at hh
    change ((K * FULL : ℕ) : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ) at hh
    norm_num [HONEST_S, K, FULL] at hh

end Pous.Period.SecurityProofs
