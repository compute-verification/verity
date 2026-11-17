module

public import Proofs.WeightSecurity.Reg
public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
import Proofs.Core.Game.Basic
import Proofs.Flock.Claims.Witnesses.SamplingGenericCompiler

/-! The one-stage period's statements say something: `CompilerMonitoringPeriodReg`'s premises hold together, and
`MonitoringPeriodReg` fails where they don't. -/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open Claims.Witness (silentP prB_accV)
open Claims.Witness.SamplingGeneric
open FlockSoundness.Game (monitoringPeriodG)
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

def oneStrat (L : Compiled Coin A) {Rls : Type} (m : Rls) (x : Msg) (P : Prover Unit) :
    Verity.Game.Strategy (monitoringPeriodG (fun _ : ℕ => oneGame L Rls) 0 1) :=
  ⟨m, (x, P), fun _ => PUnit.unit⟩

theorem oneRegRun_oneStrat (H : Msg → Msg) (L : Compiled Coin A) (m : (O → V) × Tick × Adv) (x : Msg)
    (P : Prover Unit) :
    oneRegRun H (fun _ => L) 1 (oneStrat L m x P) =
      .coin (Fin (L.V.k x) → Coin) fun r => .ret [oneRegRec H L (oneRegCount H L x P) (m, ⟨x, P, r⟩)] :=
  rfl

theorem undetected_one {w : Served O V Tick Adv} (hs : ¬ w.sampledIncorrect) (hn : 1 ≤ w.incorrect) :
    Served.Undetected 1 [w] :=
  Or.inr (by simpa [Served.lead, hs] using hn)

theorem sksL_εreg_all (H : Msg → Msg) (x : Msg) (P : Prover Unit) : (sksL Coin).εreg H x P = 0 := by
  have hb : PrB H (sksL Coin).V P x = 0 := prB_accV H P x
  unfold Compiled.εreg
  rw [sksL_κs, hb, bconf_pr H (sksL Coin) (fun _ _ _ => rfl) Unit P x]
  simp

/-- The witness is `twoL`'s windows at `x = 0`, whose design misses every nonempty set of its proof blocks surely
(`two_Pa`). -/
theorem monitoringPeriodReg_honest :
    (twoA Bool).KS FlockDeploy.sha512 ∧
      (∀ _t : ℕ, (twoL Bool).commit.Binds FlockDeploy.sha512 ∧ (twoL Bool).Sessions FlockDeploy.sha512) ∧
      (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
      ∀ (_t : ℕ) y s P, (twoL Bool).parse y = some s → (twoL Bool).partition s = some P → InScope P s →
        ∀ B ⊆ Finset.range P.blocks.length, s.design.Pa P B ≤ (0 : ℝ) ^ B.card := by
  obtain ⟨hA, hB, hS, -, -⟩ := compilerKS_honest
  refine ⟨hA, fun _ => ⟨hB, hS⟩, le_rfl, zero_le_one, fun _ y s P hs hP _ B hBsub => ?_⟩
  obtain rfl := twoL_parse hs
  obtain rfl := twoL_partition hP
  rw [two_Pa]
  split_ifs with hw
  · have hB0 : B = ∅ := by
      refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
      have hlen : twoPart.blocks.length = 2 := rfl
      have hv2 : v < 2 := by
        have := Finset.mem_range.1 (hBsub hv)
        omega
      have hvw : v ∈ window 0 2 B := by
        simp [window, hv]
        omega
      rw [hw] at hvw
      simp at hvw
    subst hB0
    simp
  · exact pow_nonneg le_rfl _

open Classical in
/-- At `sksL`'s windows and `x = 0`, one window at the empty statement and the silent prover leaves its committed
incorrect set undetected surely. -/
theorem monitoringPeriodReg_nontrivial :
    ¬ MonitoringPeriodReg (O := Unit) (V := Unit) (Tick := Unit) (Adv := Unit) FlockDeploy.sha512 (fun _ => sksL Bool) 0 := by
  intro h
  let m : (Unit → Unit) × Unit × Unit := (fun _ => (), (), ())
  have hcnt : 1 ≤ oneRegCount FlockDeploy.sha512 (sksL Bool) [] silentP := by
    rw [oneRegCount, ite_eq_left (sksL_scope [])]
    exact Finset.card_pos.2 ⟨0, sksL_Freg _ _ _ _⟩
  have hU : ∀ r : Fin ((sksL Bool).V.k []) → Bool, Served.Undetected 1
      [oneRegRec FlockDeploy.sha512 (sksL Bool) (oneRegCount FlockDeploy.sha512 (sksL Bool) [] silentP)
        (m, ⟨[], silentP, r⟩)] := fun r =>
    undetected_one (fun ⟨u, hu, _⟩ => absurd (show u ∈ ([] : List ℕ) from hu) (by simp)) hcnt
  have hv : Verity.Game.value (Served.Undetected 1)
      (oneRegRun FlockDeploy.sha512 (fun _ => sksL Bool) 1 (oneStrat (sksL Bool) m [] silentP)) = 1 := by
    rw [oneRegRun_oneStrat, Verity.Game.value_coin_ret]
    simp only [Verity.Game.prCoin, hU, if_true]
    exact Verity.Game.avg_const 1
  have h1 := h 1 (fun _ => 0) (oneStrat (sksL Bool) m [] silentP) (fun _ _ _ => (sksL_εreg_all _ _ _).le) 1
  rw [hv] at h1
  simp at h1

end WeightSecurity.SecurityProofs
