module

public import Proofs.Flock.Claims.Witnesses.SamplingTwoRegX
public import Proofs.Flock.SamplingGeneric.TwoOneWaitKS
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ZK.NonVacuous

/-! The two-stage committed form at the one-wait extraction's boundary trace says something: it fails at `sksLTwo`. -/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (footprintAt)
open Claims.Witness.Sampling (toyStmt toyPart rejCm)

theorem sksL_oneWait_opensAt {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((sksL Coin).oneWait.V.k x) → Coin)
    (c : (sksL Coin).oneWait.SlotEC x) (g : ℕ) : (sksL Coin).oneWait.opensAt H P x ω r c g = none := by
  have hp : (sksL Coin).parse x = some toyStmt := rfl
  have hc : (sksL Coin).commit = rejCm := rfl
  simp only [Compiled.opensAt, hp]
  simp [Claims.Sampling.Wit.val, Claims.Sampling.CommitmentScheme.Opens, hc, rejCm, OC.eval]

theorem sksL_oneWait_regValX {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) (g : ℕ) : (sksL Coin).oneWait.regValX H P x g = none := by
  simp [Compiled.regValX, sksL_oneWait_opensAt, Pr_false]

/-- **`CompiledTwo.OneWaitTwoXAt` fails** at `sksLTwo`'s silent prover, at the error `0`: it accepts always, while
`P̄_a` at the footprint of its one-wait boundary trace read at the statement is `0` at every interior. -/
theorem compiledTwoOneWaitXAt_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).OneWaitTwoXAt H [] toyStmt toyPart toyTwo silentP 0 := by
  intro h
  unfold CompiledTwo.OneWaitTwoXAt at h
  have h1 := h.2.2 0 fun ι => (toyTwo_Pa
    (toy_footprintAt (val := (sksLTwo Coin).oneWait.regValX H silentP [])
      (sksL_oneWait_regValX H silentP [] 0) ι)).le
  have hp : Pr (fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      (run H silentP (sksLTwo Coin).V [] p.1 p.2).out = .acc) = 1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      iff_true_intro (run_accV_out H silentP [] p.1 p.2)) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + 0 := hp.symm.trans_le h1
  norm_num at h2

end Claims.Witness.SamplingGeneric

namespace Claims.SamplingGeneric

open Claims.Witness (silentP)
open Claims.Witness.SamplingGeneric
open Claims.Witness.Sampling (toyStmt toyPart)

/-- **`SpotOneWaitTwo`'s conclusion fails** at `sksLTwo`, at its silent prover and the error `0`: it accepts with
`P̄_a` at the footprint of its one-wait boundary trace read at the statement `0` at every interior. -/
theorem compilerOneWaitTwoXAt_nontrivial :
    ¬ (sksLTwo Bool).OneWaitTwoXAt FlockDeploy.sha512 [] toyStmt toyPart toyTwo silentP 0 :=
  compiledTwoOneWaitXAt_fails _

end Claims.SamplingGeneric
