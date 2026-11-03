import Proofs.Pouw.DitherWitness

/-!
# `WorkSound`, `WorkMoment` and `FleetSound` say something

Each guarantee's honest and nontrivial witnesses, at `Definitions.Pouw.Witness`'s machine `tw`, protocol `twRound`
and period `twPd` (`T = 8`, `μ = (2, 1, 1)`, `C = 14`, `η = 1/7`), with `k = 2` sampling blocks:

* `WorkSound_honest`: every premise of `WorkSound` holds there, `AnswersFit` among them, and the verdict accepts the
  honest prover on every oracle and coins, within the cap with its answers' calls. `WorkSound_nontrivial`: the
  conclusion there (`witness_freed`) is false at the verifier that accepts everything, since the idle prover's other
  work is worth 46 against at most `32 + ε_A C < 46`.
* `WorkMoment_honest`: every premise of `WorkMoment` outside the named assumptions holds there (the dual price, the
  cost split, the cap, G1–G3, P1, P2), so do the named assumptions, and the verdict `twVerdict` (the rounds' checks)
  accepts the honest prover on every oracle and coins. `WorkMoment` at these premises is `MomentSound` with
  `ε = 1/7` (the slack `(7 − 4 − 2)/7`, no luck, no walk), `C = 14` and `κ = 0` (`witness_moment`).
* `WorkMoment_nontrivial`: that conclusion, at the same period and parameters, is false at the verifier that accepts
  everything. A prover that does nothing (`idle`) leaves the period's capacity to other work worth 46, against at most
  32 for any honest strategy (`value_le`), so wherever it is accepted its shortfall is at least 12 (`shortfall_ge`) and
  its moment at least `e^{12/7} > 1`, which no moment of average at most 1 is everywhere.
* `FleetSound_honest_at n`: at a fleet of `n + 1` replicas of the witness, each with its own coins (the challenger's
  randomness is every replica's coins, `coin = id`), each replica's moment holds at the honest verdict, the coins are
  the replicas' own (`OwnDither`), and the honest fleet reads no replica's coins (`CoinsConfidential`).
* `FleetSound_nontrivial_at n`: at that fleet, the verifier that accepts everything, `δ_coin = 1/2` and `κ_max = 0`,
  the idle fleet is accepted with every replica's shortfall at least 12, past the fleet's luck `7 (n + 1) ln 2`, on
  every oracle and coins, so `FleetSound`'s event has probability 1.
* `FleetSound_honest` and `FleetSound_nontrivial`, the witnesses the lock lists: both at a fleet of two replicas
  (`n = 1`), closed at that one instance.
-/

namespace Pouw.SecurityProofs.DitherWitness

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Witness Finset

section Idle

/-- **The idle prover**: every sampling block commits the zero instance and the zero root and answers zero, at no
use. -/
noncomputable def idle : Prover tw twRound 2 where
  St := Unit
  start := ()
  commit _ := constAlg ((), twInst)
  work _ := constAlg ((), fun _ _ _ => 0)
  absorb _ _ _ := ()
  answer _ := constAlg fun _ => (0, ())

theorem idle_timedUse (o : Unit) (x : Fin 2 → TDraw × Fin 2) (r : Fin 3) : idle.timedUse o x r = 0 := by
  show ∑ _b : Fin 2, ((0 : ℝ) + 0) = 0
  simp

/-- The idle prover's sampling blocks leave the whole period to other work: 8 units of each job. -/
theorem idle_fits (o : Unit) (x : Fin 2 → TDraw × Fin 2) : twPd.Fits (idle.timedUse o x) fun _ => 8 := by
  refine ⟨fun _ => by norm_num, fun r => ?_⟩
  rw [idle_timedUse]
  show (0 : ℝ) + ∑ j : Fin 3, (8 : ℝ) * (if j = r then 1 else 0) ≤ 1 * 8
  simp

end Idle

section Value

/-- Every honest strategy on the witness period is worth at most 32: `μ · cap`, the period's capacity at the dual
price. -/
theorem value_le (W' : ℝ) (z' : twPd.J → ℝ) (h : twPd.Fits (fun r => W' * twPd.hon r) z') : twPd.value W' z' ≤ 32 := by
  have hr : ∀ r : Fin 3, W' * twHon r + ∑ j : Fin 3, z' j * (if j = r then 1 else 0) ≤ 1 * 8 := h.2
  have h0 := hr 0
  have h1 := hr 1
  have h2 := hr 2
  change W' * 14 + ∑ j : Fin 3, twμ j * z' j ≤ 32
  norm_num [twHon, twμ, Fin.sum_univ_three] at h0 h1 h2 ⊢
  linarith

/-- The idle prover's other work, 8 units of each job, is worth `14 + 32 = 46`. -/
theorem value_eight : twPd.value 1 (fun _ => 8) = 46 := by
  change 1 * 14 + ∑ j : Fin 3, twμ j * 8 = 46
  norm_num [twμ, Fin.sum_univ_three]

/-- Where the idle prover's other work is freed within `ε = 1/7` plus its shortfall `x` over `C = 14`, the shortfall
is at least 12. -/
theorem shortfall_ge (x : ℝ) (h : twPd.FreedWithin 1 (fun _ => 8) (1 / 7 + x / 14) 14) : 12 ≤ x := by
  obtain ⟨W', z', -, hfit, hval⟩ := h
  have h32 := value_le W' z' hfit
  have h46 := value_eight
  have : (1 / 7 + x / 14) * 14 = 2 + x := by ring
  linarith

end Value

section Probability

theorem not_of_prob_le_zero {Ω : Type} [Fintype Ω] {p : Ω → Prop} (h : prob p ≤ 0) (ω : Ω) : ¬ p ω := by
  classical
  intro hp
  rw [prob_eq_card] at h
  have hc : (0 : ℝ) < (univ.filter p).card := by exact_mod_cast card_pos.2 ⟨ω, mem_filter.2 ⟨mem_univ _, hp⟩⟩
  have hn : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos_iff.2 ⟨ω⟩
  linarith [div_pos hc hn]

theorem prob_of_forall {Ω : Type} [Fintype Ω] [Nonempty Ω] {p : Ω → Prop} (h : ∀ ω, p ω) : prob p = 1 := by
  have hs := prob_add_not p
  have h0 := prob_le_zero (p := fun ω => ¬ p ω) fun ω hn => hn (h ω)
  linarith [prob_nonneg fun ω => ¬ p ω]

theorem one_lt_avg {Ω : Type} [Fintype Ω] [Nonempty Ω] {M : Ω → ℝ} (h : ∀ ω, 1 < M ω) : 1 < avg M := by
  have hn : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  unfold avg
  rw [one_lt_div hn]
  calc (Fintype.card Ω : ℝ) = ∑ _ω : Ω, (1 : ℝ) := by simp
    _ < ∑ ω, M ω := sum_lt_sum_of_nonempty univ_nonempty fun ω _ => h ω

end Probability

/-- `WorkMoment`'s conclusion at the witness: the moment at `ε = 1/7`, `C = 14`, `η = 1/7` and `κ = 0`. -/
theorem witness_moment :
    MomentSound tw twRound 2 twVerdict twDev twPd 1 (fun _ => 1 / 7) 14 twη (fun _ => 0) := by
  obtain ⟨hdual, hinst, hC, -, -, -, hβ, hK, hξ, hdeg, hcap, hG1, hG2, hG3, hP1, hP2, hsak, hsha, hout,
    hsvc, himp, hans, hext, -, -⟩ := witness
  have h := Pouw.SecurityProofs.Dither.WorkMoment tw twRound 2 twReg Prop twVerdict twDev twPd twμ twHmm
    twHsha twHo twHmm twHsha twη 0 0 0 twQ (fun _ => 0) (fun _ => 0) twSel twBadv twξ (fun _ => 0) (fun _ => 0) hdual
    hinst hC hβ hK hξ hdeg hcap hG1 hG2 hG3 hP1 hP2 hsak hsha hout hsvc himp hans hext
  convert h using 1
  · funext A
    norm_num [epsWorkSound, twHmm, twHsha, twHo, twξ, Fin.sum_univ_two]
  · change (14 : ℝ) = ∑ r : Fin 3, twμ r * (1 * twHon r)
    norm_num [Fin.sum_univ_three, twμ, twHon]
  · funext A
    norm_num

end Pouw.SecurityProofs.DitherWitness

namespace Pouw.SecurityProofs.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Witness Pouw.SecurityProofs.DitherWitness Finset

/-- **`WorkMoment`'s honest witness**: at `Definitions.Pouw.Witness`'s machine, protocol and period, every
premise of `WorkMoment` that is no named assumption holds (the dual price, the cost split, the cap on calls, G1–G3, P1
and P2), the named assumptions hold too, and the verdict accepts the honest prover on every oracle and coins. -/
theorem WorkMoment_honest :
    twPd.DualOpt twμ 1 ∧
      ∑ w, (twHmm w + twHsha w + twHo w) = ∑ r, twμ r * twPd.hon r ∧
      (∃ r, 0 < tw.callUse r ∧ twPd.cap r ≤ twQ * tw.callUse r) ∧
      tw.G1 ∧ tw.G2 ∧ tw.G3 ∧ tw.P1 ∧ tw.P2 ∧
      (∀ w, SAKMatmul tw twRound twμ (twReg w) (fun _ => twHmm w)
        (fun _ => twη * (twHmm w + twHsha w + twHo w) / twHmm w) (fun _ => 0)) ∧
      SAKSha tw twRound 2 twReg twμ twHsha (fun w => twη * (twHmm w + twHsha w + twHo w) / twHsha w)
        (fun _ => 0) (fun _ => 0) ∧
      OutsourceWithinEgress tw twμ 0 ∧ ProofSystemSound tw twRound 2 twReg twVerdict twDev 0 ∧
      ImportWithinCertifier tw twRound 2 twVerdict twDev 0 ∧ AnswersFit tw twRound 2 twVerdict twDev twSel twPd ∧
      ExtractAdv tw twRound 2 twReg twSel twQ twBadv twξ (fun _ => 0) ∧
      ∀ o x, honest.Passes twVerdict twDev o x := by
  obtain ⟨hdual, -, hC, -, -, -, -, -, -, -, hcap, hG1, hG2, hG3, hP1, hP2, hsak, hsha, hout, hsvc, himp, hans, hext,
    -, hhon⟩ := witness
  exact ⟨hdual, hC, hcap, hG1, hG2, hG3, hP1, hP2, hsak, hsha, hout, hsvc, himp, hans, hext,
    fun o x => (hhon o x).1⟩

/-- **`WorkMoment`'s nontrivial witness**: its conclusion at the witness (`witness_moment`'s period and
parameters, `ε = 1/7`, `C = 14`, `η = 1/7`, `κ = 0`) is false at the verifier that accepts everything. -/
theorem WorkMoment_nontrivial :
    ¬ MomentSound tw twRound 2 (fun _ _ _ _ => True) twDev twPd 1 (fun _ => 1 / 7) 14 twη (fun _ => 0) := by
  intro h
  obtain ⟨D, M, Bad, -, -, hav, hbad, hfree⟩ := h idle
  have : Nonempty (twRound.Orc × (Fin 2 → twRound.Draw × twRound.Chk)) :=
    (inferInstance : Nonempty (Unit × (Fin 2 → TDraw × Fin 2)))
  refine absurd hav (not_le.2 (one_lt_avg fun ox => ?_))
  obtain ⟨hM, hfw⟩ := hfree ox (fun _ => 8) trivial (not_of_prob_le_zero hbad ox) (idle_fits ox.1 ox.2)
  have hD := shortfall_ge (D ox) hfw
  have hpos : 0 < twη * D ox := mul_pos (by norm_num [twη]) (by linarith)
  linarith [Real.one_lt_exp_iff.2 hpos]

/-- **`WorkSound`'s honest witness**: at `Definitions.Pouw.Witness`'s machine, protocol and period,
every premise of `WorkSound` that is no named assumption holds (the dual price, the costs and their split, the rate
and confidence, the cap on calls, G1–G3, P1 and P2), the named assumptions hold too, `AnswersFit` among them, and the
verdict accepts the honest prover on every oracle and coins, with its sampling blocks' timed use the period's honest
use and its run, its answers' calls included, within the cap. -/
theorem WorkSound_honest :
    twPd.DualOpt twμ 1 ∧
      (∀ w, 0 < twHmm w ∧ 0 < twHsha w ∧ 0 ≤ twHo w) ∧
      ∑ w, (twHmm w + twHsha w + twHo w) = ∑ r, twμ r * twPd.hon r ∧
      0 < twη ∧ 0 < twδ ∧ twδ ≤ 1 ∧
      (∃ r, 0 < tw.callUse r ∧ twPd.cap r ≤ twQ * tw.callUse r) ∧
      tw.G1 ∧ tw.G2 ∧ tw.G3 ∧ tw.P1 ∧ tw.P2 ∧
      (∀ w, SAKMatmul tw twRound twμ (twReg w) (fun _ => twHmm w)
        (fun _ => twη * (twHmm w + twHsha w + twHo w) / twHmm w) (fun _ => 0)) ∧
      SAKSha tw twRound 2 twReg twμ twHsha (fun w => twη * (twHmm w + twHsha w + twHo w) / twHsha w)
        (fun _ => 0) (fun _ => 0) ∧
      OutsourceWithinEgress tw twμ 0 ∧ ProofSystemSound tw twRound 2 twReg twVerdict twDev 0 ∧
      ImportWithinCertifier tw twRound 2 twVerdict twDev 0 ∧ AnswersFit tw twRound 2 twVerdict twDev twSel twPd ∧
      ExtractAdv tw twRound 2 twReg twSel twQ twBadv twξ (fun _ => 0) ∧
      ∀ o x, honest.Passes twVerdict twDev o x ∧ honest.timedUse o x = twPd.hon ∧
        honest.WithinCap twSel twQ o x := by
  obtain ⟨hdual, hinst, hC, hη, hδ, hδ1, -, -, -, -, hcap, hG1, hG2, hG3, hP1, hP2, hsak, hsha, hout, hsvc, himp,
    hans, hext, -, hhon⟩ := witness
  exact ⟨hdual, fun w => ⟨(hinst w).1, (hinst w).2.1, (hinst w).2.2.1⟩, hC, hη, hδ, hδ1, hcap, hG1, hG2, hG3, hP1,
    hP2, hsak, hsha, hout, hsvc, himp, hans, hext, hhon⟩

/-- **`WorkSound`'s nontrivial witness**: its conclusion at the witness (`witness_freed`'s period
and bounds, `ε_A = 1/7 + ln 2 / 2`, `C = 14`, `κ_A = 1/2`) is false at the verifier that accepts everything. The
idle prover is accepted on every oracle and coins and leaves the period to other work worth 46, against at most
`32 + ε_A C < 46` for any honest strategy (`value_le`), so the event has probability 1. -/
theorem WorkSound_nontrivial :
    ¬ FreedSound tw twRound 2 (fun _ _ _ _ => True) twDev twPd 1 twε (∑ r, twμ r * (1 * twPd.hon r)) twκ := by
  intro h
  have : Nonempty (twRound.Orc × (Fin 2 → twRound.Draw × twRound.Chk)) :=
    (inferInstance : Nonempty (Unit × (Fin 2 → TDraw × Fin 2)))
  have hC : ∑ r, twμ r * (1 * twPd.hon r) = 14 := by
    change ∑ r : Fin 3, twμ r * (1 * twHon r) = 14
    norm_num [Fin.sum_univ_three, twμ, twHon]
  have h1 := h idle
  rw [prob_of_forall] at h1
  · linarith [twκ_lt idle]
  intro ox
  refine ⟨trivial, fun _ => 8, idle_fits ox.1 ox.2, ?_⟩
  rintro ⟨W', z', -, hfit, hval⟩
  rw [hC, value_eight] at hval
  linarith [value_le W' z' hfit, twε_lt idle]

/-- `FleetSound`'s honest witness at a fleet of `n + 1` replicas of the witness: the challenger's randomness every
replica's coins (`coin = id`), each replica's moment holds at the honest verdict (`witness_moment`), each replica's
coins are its own, the honest fleet reads no replica's coins, and the verdict accepts it. -/
theorem FleetSound_honest_at (n : ℕ) :
    MomentSound tw twRound 2 twVerdict twDev twPd 1 (fun _ => 1 / 7) 14 twη (fun _ => 0) ∧
      OwnDither twRound 2 (n + 1) (fun ω : Fin (n + 1) → Fin 2 → TDraw × Fin 2 => ω) ∧
      CoinsConfidential (G := tw) (P := twRound) (k := 2) (N := n + 1)
        (fun ω : Fin (n + 1) → Fin 2 → TDraw × Fin 2 => ω) (fun _ _ => honest) ∧
      ∀ (u : Fin (n + 1)) (ω : Fin (n + 1) → Fin 2 → TDraw × Fin 2) (o : Unit), honest.Passes twVerdict twDev o (ω u) :=
  ⟨witness_moment, fun _ => rfl, fun _ _ _ _ => rfl, fun u ω o => honest_passes o (ω u)⟩

/-- `FleetSound`'s nontrivial witness at the fleet of `FleetSound_honest_at n`: at the verifier that accepts
everything, `δ_coin = 1/2` and `κ_max = 0`, some prover's fleet breaks `FleetSound`'s bound: its event has
probability 1. -/
theorem FleetSound_nontrivial_at (n : ℕ) :
    ∃ A : Prover tw twRound 2, ¬ prob (fun oω : Unit × (Fin (n + 1) → Fin 2 → TDraw × Fin 2) =>
      ∃ S : Finset (Fin (n + 1)), S.Nonempty ∧ ∃ z : Fin (n + 1) → twPd.J → ℝ,
        (∀ u ∈ S, A.Passes (fun _ _ _ _ => True) twDev oω.1 (oω.2 u) ∧ twPd.Fits (A.timedUse oω.1 (oω.2 u)) (z u)) ∧
        ¬ ∃ d : Fin (n + 1) → ℝ, (∀ u ∈ S, 0 ≤ d u ∧ twPd.FreedWithin 1 (z u) (1 / 7 + d u / 14) 14) ∧
          ∑ u ∈ S, d u ≤ fleetLuck (fun _ => twη) S twδ) ≤ twδ + ∑ _u : Fin (n + 1), (0 : ℝ) := by
  refine ⟨idle, ?_⟩
  rw [not_le, prob_of_forall]
  · norm_num [twδ]
  intro oω
  refine ⟨univ, univ_nonempty, fun _ _ => 8, fun u _ => ⟨trivial, idle_fits _ _⟩, ?_⟩
  rintro ⟨d, hd, hsum⟩
  have h12 : ∀ u, 12 ≤ d u := fun u => shortfall_ge (d u) (hd u (mem_univ u)).2
  have hlow : 12 * ((n : ℝ) + 1) ≤ ∑ u, d u := by
    calc 12 * ((n : ℝ) + 1) = ∑ _u : Fin (n + 1), (12 : ℝ) := by
          rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
          push_cast
          ring
      _ ≤ ∑ u, d u := sum_le_sum fun u _ => h12 u
  have hx : (∑ _u : Fin (n + 1), 1 / twη) = 7 * ((n : ℝ) + 1) := by
    rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    simp only [twη, one_div_one_div]
    push_cast
    ring
  have hluck : fleetLuck (N := n + 1) (fun _ => twη) univ twδ = 7 * ((n : ℝ) + 1) * Real.log 2 := by
    simp only [fleetLuck]
    rw [hx, div_self (by positivity : 7 * ((n : ℝ) + 1) ≠ 0), Real.log_one, zero_add]
    norm_num [twδ]
  rw [hluck] at hsum
  have hn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hl : Real.log 2 ≤ 1 := by linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)]
  nlinarith [mul_le_mul_of_nonneg_left hl hn.le]

/-- **`FleetSound`'s honest witness**: `FleetSound_honest_at` at a fleet of two replicas of the witness. -/
theorem FleetSound_honest :
    MomentSound tw twRound 2 twVerdict twDev twPd 1 (fun _ => 1 / 7) 14 twη (fun _ => 0) ∧
      OwnDither twRound 2 (1 + 1) (fun ω : Fin (1 + 1) → Fin 2 → TDraw × Fin 2 => ω) ∧
      CoinsConfidential (G := tw) (P := twRound) (k := 2) (N := 1 + 1)
        (fun ω : Fin (1 + 1) → Fin 2 → TDraw × Fin 2 => ω) (fun _ _ => honest) ∧
      ∀ (u : Fin (1 + 1)) (ω : Fin (1 + 1) → Fin 2 → TDraw × Fin 2) (o : Unit), honest.Passes twVerdict twDev o (ω u) :=
  FleetSound_honest_at 1

/-- **`FleetSound`'s nontrivial witness**: `FleetSound_nontrivial_at` at the fleet of two replicas of
`FleetSound_honest`. -/
theorem FleetSound_nontrivial :
    ∃ A : Prover tw twRound 2, ¬ prob (fun oω : Unit × (Fin (1 + 1) → Fin 2 → TDraw × Fin 2) =>
      ∃ S : Finset (Fin (1 + 1)), S.Nonempty ∧ ∃ z : Fin (1 + 1) → twPd.J → ℝ,
        (∀ u ∈ S, A.Passes (fun _ _ _ _ => True) twDev oω.1 (oω.2 u) ∧ twPd.Fits (A.timedUse oω.1 (oω.2 u)) (z u)) ∧
        ¬ ∃ d : Fin (1 + 1) → ℝ, (∀ u ∈ S, 0 ≤ d u ∧ twPd.FreedWithin 1 (z u) (1 / 7 + d u / 14) 14) ∧
          ∑ u ∈ S, d u ≤ fleetLuck (fun _ => twη) S twδ) ≤ twδ + ∑ _u : Fin (1 + 1), (0 : ℝ) :=
  FleetSound_nontrivial_at 1

end Pouw.SecurityProofs.Dither
