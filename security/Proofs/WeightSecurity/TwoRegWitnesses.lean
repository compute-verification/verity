module

public import Proofs.WeightSecurity.TwoReg
public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.Witnesses.SamplingTwo
import Proofs.Core.Game.Basic
import Proofs.Flock.Claims.Witnesses.SamplingTwoReg
import Proofs.Flock.Claims.Witnesses.SamplingTwoRegMiss

/-! The two-stage period's statements say something: `CompilerMonitoringPeriodTwoReg`'s premises hold together, and
`MonitoringPeriodTwoReg` fails where they don't. -/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open Claims.Witness (silentP prB_accV)
open Claims.Witness.SamplingGeneric
open Claims.Witness.Sampling (toyStmt toyPart)
open FlockSoundness.Game (monitoringPeriodG)
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

def twoStrat (L : CompiledTwo Coin A) {Rls : Type} (m : Rls) (x : Msg) (P : Prover Unit) :
    Verity.Game.Strategy (monitoringPeriodG (fun _ : ℕ => twoGame L Rls) 0 1) :=
  ⟨m, (x, P), fun _ => PUnit.unit⟩

theorem twoRegRun_twoStrat (H : Msg → Msg) (L : CompiledTwo Coin A) (m : (O → V) × Tick × Adv) (x : Msg)
    (P : Prover Unit) :
    twoRegRun H (fun _ => L) 1 (twoStrat L m x P) =
      .coin (Fin (L.V.k x) → Coin) fun r => .ret [twoRegRec H L (regCount H L x P) (m, ⟨x, P, r⟩)] :=
  rfl

/-- The witness is `twoLTwo`'s windows at both rates `1` (`compilerKSTwo_honest`), whose design `twoTwo` catches each
replay block and samples each cell surely. -/
theorem monitoringPeriodTwoReg_honest :
    (twoA Bool).KS FlockDeploy.sha512 ∧
      (∀ _t : ℕ, (twoLTwo Bool).commit.Binds FlockDeploy.sha512 ∧ (twoLTwo Bool).Sessions FlockDeploy.sha512) ∧
      (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
      ∀ (_t : ℕ) x s Pt d, (twoLTwo Bool).parse x = some s → (twoLTwo Bool).partition s = some Pt →
        (twoLTwo Bool).two s = some d → InScopeTwo Pt d s → d.FirstRate 1 ∧ d.SecondRate Pt 1 := by
  obtain ⟨hA, hB, hS, -, -⟩ := compilerKSTwo_honest
  refine ⟨hA, fun _ => ⟨hB, hS⟩, zero_le_one, le_rfl, zero_le_one, le_rfl, fun _ x s Pt d hs hP hd _ => ?_⟩
  obtain rfl : s = twoStmt := (Option.some.inj hs).symm
  obtain rfl : Pt = twoPart := (Option.some.inj hP).symm
  obtain rfl : d = twoTwo := (Option.some.inj hd).symm
  exact ⟨twoTwo_firstRate, twoTwo_secondRate⟩

open Classical in
/-- At `sksLTwo`'s windows and both rates `1`, one window at the empty statement and the silent prover leaves its
replay block wrong at the committed boundary undetected surely. -/
theorem monitoringPeriodTwoReg_nontrivial :
    ¬ MonitoringPeriodTwoReg (O := Unit) (V := Unit) (Tick := Unit) (Adv := Unit) FlockDeploy.sha512 (fun _ => sksLTwo Bool)
      1 1 := by
  intro h
  let m : (Unit → Unit) × Unit × Unit := (fun _ => (), (), ())
  have hε : ∀ (x : Msg) (P : Prover Unit), (sksLTwo Bool).εregTwo FlockDeploy.sha512 x P = 0 := by
    intro x P
    have hb : PrB FlockDeploy.sha512 (sksL Bool).V P x = 0 := prB_accV _ P x
    have h1 : (sksL Bool).εreg FlockDeploy.sha512 x P = 0 := by
      unfold Compiled.εreg
      rw [sksL_κs, hb, bconf_pr _ (sksL Bool) (fun _ _ _ => rfl) Unit P x]
      simp
    unfold CompiledTwo.εregTwo
    rw [sksLTwo_bconfTwo, Real.sqrt_zero, mul_zero, add_zero]
    exact h1
  have hcnt : 1 ≤ regCount FlockDeploy.sha512 (sksLTwo Bool) [] silentP := by
    have hval : regCount FlockDeploy.sha512 (sksLTwo Bool) [] silentP =
        (incorrectAtBoundaryAt toyStmt toyPart toyTwo.part
          ((sksLTwo Bool).regVal FlockDeploy.sha512 silentP [] ())).card :=
      ite_eq_left inScopeTwo_toy
    rw [hval]
    exact Finset.card_pos.2 ⟨0, toy_incorrectAtBoundaryAt (val := (sksLTwo Bool).regVal FlockDeploy.sha512 silentP [] ())
      (sksL_regVal _ silentP [] () 0)⟩
  have hund : ∀ w : Served Unit Unit Unit Unit, ¬ w.sampledIncorrect → 1 ≤ w.incorrect →
      Served.Undetected 1 [w] :=
    fun w hs hn => Or.inr (by simpa [Served.lead, hs] using hn)
  have hU : ∀ r : Fin ((sksLTwo Bool).V.k []) → Bool, Served.Undetected 1
      [twoRegRec FlockDeploy.sha512 (sksLTwo Bool) (regCount FlockDeploy.sha512 (sksLTwo Bool) [] silentP)
        (m, ⟨[], silentP, r⟩)] := fun r =>
    hund _ (fun ⟨u, hu, _⟩ => absurd (show u ∈ ([] : List ℕ) from hu) (by simp)) hcnt
  have hv : Verity.Game.value (Served.Undetected 1)
      (twoRegRun FlockDeploy.sha512 (fun _ => sksLTwo Bool) 1 (twoStrat (sksLTwo Bool) m [] silentP)) = 1 := by
    rw [twoRegRun_twoStrat, Verity.Game.value_coin_ret]
    simp only [Verity.Game.prCoin, hU, if_true]
    exact Verity.Game.avg_const 1
  have h1 := h 1 (fun _ => 0) (twoStrat (sksLTwo Bool) m [] silentP) (fun _ _ _ => (hε _ _).le) 1
  rw [hv] at h1
  norm_num at h1

end WeightSecurity.SecurityProofs
