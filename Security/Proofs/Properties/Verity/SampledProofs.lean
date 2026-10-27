import Proofs.Properties.Verity.Soundness
import Proofs.Flock.Bridge.SampledSound

/-!
# Sampled proofs' target at the verifier of record, from C-Flock's live session

`Vocab.SampledProofsSound` (`Target .sampled`, `Accounting.PeriodLive.Sound` at `Vocab.live`) holds at every vocabulary
coupled, strategy by strategy, to a C-Flock live session (`SampledIn`): the session's setting
(`Flock.Guarantees.LiveAt`), the strategy of it the play induces, a break `Brk` of the play, and five facts relating
them.

* `accept_le`: outside the break, sampled proofs' verdict in the play is at most as likely as flock-verify's acceptance
  in the session (ONE-4's projection of the device's releases onto the session's coins, and ONE-3t's
  `acc_of_accepts`).
* `break_le`: the break (ONE-4's `Properties.Challenger.Break`: a forged release, a collision) is at most the loss at
  the probability that the vocabulary's finder outputs a collision, as `Accounting.Live.sound_of_break` charges it.
* `unbound_le`: the units the vocabulary's extractor leaves unbound are at most those faulty under the session
  extractor's witness (`LiveAt.faultyE`: wrong, or with a committed value unopened; R4′; ONE-7's `place` and ONE78's
  `circuitOfFile`).
* `cr`: where the vocabulary's finder meets its quantitative premise (`Vocab.FinderCR`), the finders C-Flock builds
  from the induced strategy meet theirs (`LiveAt.CR`).
* `delta_le`: the published error at the strategy's cost is at least C-Flock's (`Bridge.deltaPub` at `t'`).

`sampledProofsSound` composes them with `LiveAt.soundF`. Its error is the vocabulary's (`Vocab.error`, the conditional
form `Accounting.condError`), so it asks nothing of the strategy. `sampledProofsSound_nontrivial` is
`soundness_of_subprotocols_nontrivial_of`'s hypothesis at a commitment that opens no committed value of any unit, ONE19's
instance (#4867); `honestHolds_sampled` is `Vocab.HonestHolds .sampled` at the honest commitment.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **Sampled proofs at `θ`, `F`, `c` and the strategy `σ`, coupled to a C-Flock live session.** The session's setting
`A` at `n` units, the draw size `kd`, `K₀` and the descriptor `desc`; the strategy `τ` of the session the play induces;
the sampling query `q`; the play's break `Brk`; and the five facts the target reads off them: outside the break, the
play's verdict at most as likely as the session's acceptance (`accept_le`), the break at most the loss at the finder
(`break_le`), the vocabulary's unbound units at most the session's faulty ones (`unbound_le`), the finders'
premises (`cr`) and the errors (`delta_le`). `K₀` covers the claim's `ε` share (`hε`). -/
structure SampledIn (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) where
  n : ℕ
  D : Flock.Desc
  hU : FlockSoundness.Built D.unroll.unit
  kd : ℕ
  K₀ : ℕ
  desc : Claims.Msg
  A : Flock.Guarantees.LiveAt n D hU kd K₀ desc
  τ : FlockSoundness.Game.Strategy A.G
  q : Claims.Sampling.Query
  units_eq : V.units F = n
  hε : (K₀ : ℝ) - 1 ≤ (V.certOf θ).ε * n
  Brk : Outcome → Prop
  accept_le : Game.prob (fun o => V.Verdict .sampled o ∧ ¬ Brk o) (V.game θ F c) σ ≤
    FlockSoundness.Game.prob (fun o => FlockSoundness.Discharge.ZkExec.LiveAcceptsZK A.I o) A.G τ
  break_le : Game.prob Brk (V.game θ F c) σ ≤ V.loss .sampled θ ((V.live θ F c).collides sha512Msg .sampled σ)
  unbound_le : (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card ≤ (A.faultyE q τ).card
  cr : V.FinderCR .sampled θ F c σ → A.CR τ
  delta_le : ENNReal.ofReal (Flock.Bridge.deltaPub n kd K₀ A.k (A.qF τ) (A.qS τ) (A.t' τ) A.M A.Rw
    (Flock.Bridge.ioSum A.P) A.ρ) ≤ V.delta θ .sampled (V.cost θ F c σ)

namespace SampledIn

variable {V : Vocab d α} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}

/-- **The session's bound and the break's loss are at most the vocabulary's error**: where the vocabulary's finder
meets its premise, the session's finders meet theirs and `deltaPub` is at most the published `δ`; elsewhere the error
is `⊤`. -/
theorem bound_le (S : SampledIn V θ F c σ) :
    S.A.bound S.τ + V.loss .sampled θ ((V.live θ F c).collides sha512Msg .sampled σ) ≤
      V.error .sampled θ F c σ := by
  by_cases hcr : V.FinderCR .sampled θ F c σ
  · rw [Flock.Guarantees.LiveAt.bound, ite_eq_left (S.cr hcr), Vocab.error, Accounting.condError, ite_eq_left hcr]
    dsimp only [PeriodLive.error]
    exact add_le_add_right S.delta_le _
  · rw [Vocab.error, Accounting.condError, ite_eq_right hcr]
    exact le_top

/-- **The target's event at the coupled strategy**: sampled proofs' verdict accepts while its claim fails with
probability at most the vocabulary's error. -/
theorem le_error (S : SampledIn V θ F c σ) :
    Game.prob (fun o => V.Verdict .sampled o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤
      V.error .sampled θ F c σ := by
  by_cases hH : ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
      (V.certOf θ).ε * V.units F
  · refine (Game.prob_mono (q := fun _ => False) (fun o ho => ho.2 hH) _ σ).trans ?_
    rw [Game.prob_false]
    exact zero_le
  · have hA : ¬ S.A.HoldsF S.q (V.certOf θ).ε S.τ := fun h => hH (by
      have h' : ((S.A.faultyE S.q S.τ).card : ℝ) ≤ (V.certOf θ).ε * S.n := h
      have hu : ((V.units F : ℕ) : ℝ) = S.n := by exact_mod_cast S.units_eq
      have hw : ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
          (S.A.faultyE S.q S.τ).card := by exact_mod_cast S.unbound_le
      rw [hu]
      exact hw.trans h')
    calc Game.prob (fun o => V.Verdict .sampled o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ
        ≤ Game.prob (fun o => (V.Verdict .sampled o ∧ ¬ S.Brk o) ∨ S.Brk o) (V.game θ F c) σ :=
          Game.prob_mono (fun o ho => (Classical.em (S.Brk o)).elim Or.inr fun hb => Or.inl ⟨ho.1, hb⟩) _ σ
      _ ≤ Game.prob (fun o => V.Verdict .sampled o ∧ ¬ S.Brk o) (V.game θ F c) σ +
            Game.prob S.Brk (V.game θ F c) σ := Game.prob_or_le _ _ _ σ
      _ ≤ FlockSoundness.Game.prob
            (fun o => FlockSoundness.Discharge.ZkExec.LiveAcceptsZK S.A.I o ∧ ¬ S.A.HoldsF S.q (V.certOf θ).ε S.τ)
            S.A.G S.τ + V.loss .sampled θ ((V.live θ F c).collides sha512Msg .sampled σ) :=
          add_le_add (S.accept_le.trans (FlockSoundness.Game.prob_mono (fun _ ho => ⟨ho, hA⟩) _ _)) S.break_le
      _ ≤ S.A.bound S.τ + V.loss .sampled θ ((V.live θ F c).collides sha512Msg .sampled σ) :=
          add_le_add_right (S.A.soundF S.q (V.certOf θ).ε S.hε S.τ) _
      _ ≤ V.error .sampled θ F c σ := S.bound_le

end SampledIn

/-- **Sampled proofs' target** (`Vocab.SampledProofsSound`, ONE-8): at every vocabulary coupled to C-Flock's live
session at every `θ`, `F`, `c` and strategy (`SampledIn`), for every strategy, the probability that sampled proofs'
verdict accepts while more than an `ε` share of the units are unbound in the extractor's transcript is at most
`Vocab.error .sampled` there: the published `δ` at the strategy's cost plus its loss, where the finder built from the
strategy meets its quantitative premise (`Vocab.FinderCR`), and `⊤` elsewhere. -/
theorem sampledProofsSound (V : Vocab d α) (C : ∀ θ F c (σ : Game.Strategy (V.game θ F c)), SampledIn V θ F c σ) :
    V.SampledProofsSound :=
  fun θ F c σ => (C θ F c σ).le_error

/-- **Sampled proofs' claim fails where the commitment opens no committed value of any unit**: every unit has a
committed input or output the commitment opens to no value, so every unit is unbound, and the claim fails at any
`ε < 1` once there is a unit. ONE19's instance (#4867, `sampledKSLivePub_nontrivial`: a commitment with no rows, one
unit holding a returned computed gate) is this case. -/
theorem not_holds_sampled_of_unopened (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (o : Outcome) (hio : ∀ u, ∃ g ∈ (V.partsOf F).io u, ∀ v, ¬ V.Opens θ F c g v) (hε : (V.certOf θ).ε < 1)
    (hn : 0 < V.units F) : ¬ V.Holds θ F c σ .sampled o := by
  have hu : Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ) = Finset.univ := by
    classical
    refine Finset.eq_univ_of_forall fun u => Finset.mem_filter.2 ⟨Finset.mem_univ u, Or.inr ?_⟩
    obtain ⟨g, hg, hno⟩ := hio u
    exact ⟨g, hg, hno _⟩
  intro h
  change ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
    (V.certOf θ).ε * V.units F at h
  rw [hu, Finset.card_univ, Fintype.card_fin] at h
  have hn' : (0 : ℝ) < V.units F := by exact_mod_cast hn
  have := mul_lt_mul_of_pos_right hε hn'
  linarith

/-- **Sampled proofs' target is not trivial** (`soundness_of_subprotocols_nontrivial_of`'s hypothesis at `.sampled`):
at a strategy whose commitment opens no committed value of any unit (ONE19's instance), with `ε < 1`, a unit, and the
errors summing below one, the errors sum below the probability that sampled proofs' claim fails. -/
theorem sampledProofsSound_nontrivial (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hio : ∀ u, ∃ g ∈ (V.partsOf F).io u, ∀ v, ¬ V.Opens θ F c g v) (hε : (V.certOf θ).ε < 1) (hn : 0 < V.units F)
    (hlt : ∑ b, V.error b θ F c σ < 1) :
    ∑ b, V.error b θ F c σ < Game.prob (fun o => ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ := by
  exact hlt.trans_eq (prob_play_eq_one (V.rounds θ) θ F c (fun o => ¬ V.Holds θ F c σ .sampled o)
    (fun _ _ => not_holds_sampled_of_unopened V θ F c σ _ hio hε hn) σ).symm

/-- **Sampled proofs' honest lemma** (`Vocab.HonestHolds .sampled`): where, at registered files, the extractor's
transcript at the honest prover is correct on every unit and the commitment opens every gate to it (the honest
commitment, `Vocab.holds_sampled_of_opens`), at any `ε ≥ 0`. -/
theorem honestHolds_sampled (V : Vocab d α) (hε : ∀ θ, 0 ≤ (V.certOf θ).ε)
    (h : ∀ θ F c, V.Registered θ F →
      (∀ u, (V.partsOf F).Correct (V.extract θ F c (V.honest θ F c)) u) ∧
        ∀ g, V.Opens θ F c g (V.extract θ F c (V.honest θ F c) g)) :
    V.HonestHolds .sampled := fun θ F c hreg =>
  prob_play_eq_zero (V.rounds θ) θ F c _
    (fun _ _ hn => hn (V.holds_sampled_of_opens θ F c _ _ (h θ F c hreg).1 (h θ F c hreg).2 (hε θ))) _

end Verity
