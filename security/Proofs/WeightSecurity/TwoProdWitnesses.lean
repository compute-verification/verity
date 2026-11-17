module

public import Proofs.WeightSecurity.TwoProd
public import Proofs.WeightSecurity.TwoRegWitnesses
public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.Witnesses.SamplingTwo
import Proofs.Core.Game.Basic
import Proofs.Core.ValueSalts
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Witnesses.SamplingTwoReg
import Proofs.WeightSecurity.TwoProdMonitoringPeriod

/-! The two-stage period's product form says something: its premises hold at an instance, and its statements fail
where they don't. -/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open Claims.Witness (silentP run_accV_out)
open Claims.Witness.SamplingGeneric
open Claims.Witness.Sampling (toyStmt toyPart)
open FlockSoundness.Game (monitoringPeriodG)
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

theorem twoAtRun_twoStrat (H : Msg → Msg) (L : CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (m : (O → V) × Tick × Adv)
    (x : Msg) (P : Prover Unit) :
    twoAtRun H (fun _ => L) val 1 (twoStrat L m x P) =
      .coin (Fin (L.V.k x) → Coin) fun r =>
        .ret [twoAtRec H L (val 0 m x P) (regCountAt L x (val 0 m x P)) (escAt L x (val 0 m x P)) (m, ⟨x, P, r⟩)] :=
  rfl

theorem undetectedBelow_one {w : ServedEsc O V Tick Adv} {δ : ℝ} (hδ : 0 < δ) (hs : ¬ w.sampledIncorrect)
    (he : w.missProb = 0) : ServedEsc.UndetectedBelow δ [w] := by
  refine Or.inr ?_
  have h0 : ServedEsc.leadMissProb [w] = 0 := by simp [ServedEsc.leadMissProb, hs, he]
  rw [h0]
  exact ENNReal.ofReal_pos.2 hδ

theorem prodBelow_one {w : ServedEsc O V Tick Adv} {δ : ℝ} (hδ : 0 < δ) (ha : w.accepted) (he : w.missProb = 0) :
    (∀ r ∈ [w], r.accepted) ∧ ([w].map ServedEsc.missProb).prod < ENNReal.ofReal δ := by
  refine ⟨fun r hr => ?_, ?_⟩
  · rw [List.mem_singleton.1 hr]
    exact ha
  · rw [List.map_singleton, List.prod_singleton, he]
    exact ENNReal.ofReal_pos.2 hδ

theorem value_undetectedBelow_one (H : Msg → Msg) (L : CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (m : (O → V) × Tick × Adv)
    (x : Msg) (P : Prover Unit) {δ : ℝ} (hδ : 0 < δ)
    (hs : ∀ r : Fin (L.V.k x) → Coin, ¬ ∃ u ∈ L.checked x r, u ∈ L.FregTwoAt H P x () (val 0 m x P) r)
    (he : escAt L x (val 0 m x P) = 0) :
    Verity.Game.value (ServedEsc.UndetectedBelow δ) (twoAtRun H (fun _ => L) val 1 (twoStrat L m x P)) = 1 := by
  rw [twoAtRun_twoStrat, Verity.Game.value_coin_ret]
  exact Verity.ValueSalts.prCoin_of_forall fun r => undetectedBelow_one hδ (hs r) he

theorem value_prodBelow_one (H : Msg → Msg) (L : CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (m : (O → V) × Tick × Adv)
    (x : Msg) (P : Prover Unit) {δ : ℝ} (hδ : 0 < δ)
    (ha : ∀ r : Fin (L.V.k x) → Coin, (run H P L.V x () r).out = .acc)
    (he : escAt L x (val 0 m x P) = 0) :
    Verity.Game.value
        (fun l : List (ServedEsc O V Tick Adv) =>
          (∀ r ∈ l, r.accepted) ∧ (l.map ServedEsc.missProb).prod < ENNReal.ofReal δ)
        (twoAtRun H (fun _ => L) val 1 (twoStrat L m x P)) = 1 := by
  rw [twoAtRun_twoStrat, Verity.Game.value_coin_ret]
  exact Verity.ValueSalts.prCoin_of_forall fun r => prodBelow_one hδ (ha r) he

/-- The witness is `twoLTwo`'s verification phases (`compilerKSTwo_honest`). -/
theorem monitoringPeriodTwoProd_honest :
    (twoA Bool).KS FlockDeploy.sha512 ∧
      (∀ _t : ℕ, (twoLTwo Bool).commit.Binds FlockDeploy.sha512 ∧ (twoLTwo Bool).Sessions FlockDeploy.sha512) ∧
      ∀ _t : ℕ, (twoLTwo Bool).Sessions FlockDeploy.sha512 := by
  obtain ⟨hA, hB, hS, -, -⟩ := compilerKSTwo_honest
  exact ⟨hA, fun _ => ⟨hB, hS⟩, fun _ => hS⟩

theorem toyTwo_paBestAt {val : ℕ → Option Bool} (hv : val 0 = none) :
    toyTwo.PaBestAt toyStmt toyPart val = 0 := by
  unfold Claims.Sampling.Design.Two.PaBestAt
  exact toyTwo_Pa (toy_footprintAt hv _)

open Classical in
theorem escAt_sksLTwo (x : Msg) {val : ℕ → Option Bool} (hv : val 0 = none) : escAt (sksLTwo Bool) x val = 0 := by
  have hval : escAt (sksLTwo Bool) x val = ENNReal.ofReal (toyTwo.PaBestAt toyStmt toyPart val) :=
    ite_eq_left inScopeTwo_toy
  rw [hval, toyTwo_paBestAt hv, ENNReal.ofReal_zero]

theorem sksLTwo_samples_none (H : Msg → Msg) (x : Msg) (P : Prover Unit) (val : ℕ → Option Bool)
    (r : Fin ((sksLTwo Bool).V.k x) → Bool) :
    ¬ ∃ u ∈ (sksLTwo Bool).checked x r, u ∈ (sksLTwo Bool).FregTwoAt H P x () val r :=
  fun ⟨u, hu, _⟩ => absurd (show u ∈ ([] : List ℕ) from hu) (by simp)

theorem acceptedSampledAt_sksLTwo (H : Msg → Msg) (x : Msg) (P : Prover Unit) (val : ℕ → Option Bool) :
    acceptedSampledAt H (sksLTwo Bool) x P val = 0 := by
  unfold acceptedSampledAt
  exact Eq.trans (Pr_congr fun r => iff_false_intro fun h => sksLTwo_samples_none H x P val r h.2) Pr_false

theorem escAt_sksLTwo_regVal :
    escAt (sksLTwo Bool) [] ((sksLTwo Bool).regVal FlockDeploy.sha512 silentP [] ()) = 0 :=
  escAt_sksLTwo [] (val := (sksLTwo Bool).regVal FlockDeploy.sha512 silentP [] ()) (sksL_regVal _ silentP [] () 0)

/-- At `sksLTwo`'s verification phases, the committed boundary trace and `δ = 1/2`, one phase at the empty statement
and the silent prover samples nothing and has missProb `0`. -/
theorem monitoringPeriodTwoLead_nontrivial :
    ¬ MonitoringPeriodTwoLead (O := Unit) (V := Unit) (Tick := Unit) (Adv := Unit) FlockDeploy.sha512 (fun _ => sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) := by
  intro h
  let m : (Unit → Unit) × Unit × Unit := (fun _ => (), (), ())
  have hv : Verity.Game.value (ServedEsc.UndetectedBelow (1 / 2))
      (twoAtRun FlockDeploy.sha512 (fun _ => sksLTwo Bool)
        (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) 1
        (twoStrat (sksLTwo Bool) m [] silentP)) = 1 :=
    value_undetectedBelow_one FlockDeploy.sha512 (sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) m [] silentP (by norm_num)
      (fun r => sksLTwo_samples_none _ _ _ _ r) escAt_sksLTwo_regVal
  have h1 := h 1 (fun _ => 0) (twoStrat (sksLTwo Bool) m [] silentP)
    (fun _ _ _ _ => (acceptedSampledAt_sksLTwo _ _ _ _).le) (1 / 2)
  rw [hv] at h1
  norm_num at h1

/-- At `sksLTwo`'s verification phases, the committed boundary trace and `δ = 1/2`, one phase at the empty statement
and the silent prover is accepted surely with missProb `0`. -/
theorem monitoringPeriodTwoProd_nontrivial :
    ¬ MonitoringPeriodTwoProd (O := Unit) (V := Unit) (Tick := Unit) (Adv := Unit) FlockDeploy.sha512 (fun _ => sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) := by
  intro h
  let m : (Unit → Unit) × Unit × Unit := (fun _ => (), (), ())
  have hv : Verity.Game.value
      (fun l : List (ServedEsc Unit Unit Unit Unit) =>
        (∀ r ∈ l, r.accepted) ∧ (l.map ServedEsc.missProb).prod < ENNReal.ofReal (1 / 2))
      (twoAtRun FlockDeploy.sha512 (fun _ => sksLTwo Bool)
        (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) 1
        (twoStrat (sksLTwo Bool) m [] silentP)) = 1 :=
    value_prodBelow_one FlockDeploy.sha512 (sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) m [] silentP (by norm_num)
      (fun r => run_accV_out FlockDeploy.sha512 silentP [] () r) escAt_sksLTwo_regVal
  have h1 := h 1 (fun _ => 0) (twoStrat (sksLTwo Bool) m [] silentP)
    (fun _ _ _ _ => (acceptedSampledAt_sksLTwo _ _ _ _).le) (1 / 2)
  rw [hv] at h1
  norm_num at h1

theorem sksLTwo_not_sessions : ¬ (sksLTwo Bool).Sessions FlockDeploy.sha512 := fun hS =>
  monitoringPeriodTwoLead_nontrivial
    (compilerMonitoringPeriodTwoProd Bool (toyA Bool) Unit Unit Unit Unit FlockDeploy.sha512 (fun _ => sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) fun _ => hS).1

end WeightSecurity.SecurityProofs
