import Proofs.Flock.Recursive.BitsK
import Proofs.Flock.Claims.Recursion.KappaEpsTight
import Proofs.Flock.Claims.Recursion.CallsClosed

/-!
# `Pr[acc ∧ ¬R] ≤ 2^{−140.4} + t·2^{−138.3}` at the deployed reader

`recursiveBitsK_tight`: `recursiveBitsK` (`Flock.SecurityProofs.RecursiveKS` through
`Claims.Recursion.bitsLin2KOn_of_ksLin2On`) evaluated at `lamLin2_deployedB_ge_tight`'s caps, every one of them, and
both of `BitsLin2KOn`'s call bounds, read from a run the compiled verifier accepts. Where none accepts at `x`,
`Pr[acc ∧ ¬R] = 0`. Where one does:

* the inner statement is in `PlainIn` (`plainIn_of_run_acc`), each staged V\* part in `PartIn` (`partIn_of_run_acc`),
  so at most `RecBudget.MAX_STATEMENTS = 16` parts (`nS_le_of_partIn`), within the caps' `2^16`, and the inner setup
  has at most `2^56` inputs (`innerSetupIns_le_acc`);
* the inner verifier's calls are at most `nR = 2^28 + (c_B + 2^24)` on every run (`n`) and the compiled verifier's own
  at its V\* statements at most `2^55` (`v`), `lamLin2_R_ge_in`'s counts of the verifier's code;
* at those caps and `n, v ≤ 2^55`, `κ ≤ 2^{−140.4}` and `ε ≤ 2^{−138.3}` (`kappa_eps_deployed_dep_tight`).

What is left is what the statement names: the decoders' list-decoding spec (`decOK`), the prover's calls `t ≥ 1` per
run, and SHA-512's named collision assumptions, `StrictBirthday` at `B` and `ExpectedCR` at `B_I`, `Fdev`, `FCI` and
`F0I`, each finder charged the verifier's own calls at `n = nR` and `v = 2^55`. `λ₂ ≥ 135.7` charges each of its six
terms the least one's bits less `log₂ 6`; the split form charges `κ` once and sums the five per-call terms.
-/

namespace RecDeploy

open Claims (Msg Out)

/-- **The inner verifier's calls at an accepted statement**, `lamLin2_R_ge_in`'s count: `2^28`, the inner setup's
`c_B = SetupCount.innerB` and a decision's `c_V = 2^24`. -/
abbrev nR : ℕ := 2 ^ 28 + (SetupCount.innerB + 2 ^ 24)

theorem nR_le : nR ≤ 2 ^ 55 :=
  le_trans (Nat.add_le_add_left (Nat.add_le_add_right SetupCount.innerB_le (2 ^ 24)) (2 ^ 28)) (by norm_num)

/-- **The deployed bound record** `Flock.Guarantees.RecursiveKS` is stated at: the deployed reader's steps and
trimmed columns, V\*'s last message's reader and the deployed weights. -/
noncomputable abbrev bdKS :=
  bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER FlockDeploy.joinR
    rowInDeployed DeployedR.finalR DeployedR.wR

/-- **The compiled record** `Flock.Guarantees.RecursiveKSAt` is stated at, at the brings `b` and the coin server's
record, its inner view running `bdKS`'s `IVStar`. -/
noncomputable abbrev recKS (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    Claims.Recursion.Deploy FlockDeploy.Coin :=
  (recBoundAt _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
    (DeployedR.layR OpenRows.elemR)).record bdKS.IVStar

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−140.4} + t·2^{−138.3}` at the deployed reader**, at the brings `b` and the coin server's
record (`aux`, `nOf`, `hd`, `recRest`): for every inner and outer decoder meeting the list-decoding spec (`decOK`),
every dense statement `x`, every prover `P` making at most `t ≥ 1` SHA-512 calls per run, under `StrictBirthday` at
`B` and `ExpectedCR` at `B_I`, `Fdev`, `FCI` and `F0I`, each charged `|S(x_I)|` and the verifier's own calls at
`n = nR` and `v = 2^55`. -/
def RecursiveBitsKTightAt (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  ∀ decI, Claims.Recursion.decOK bdKS.deploy decI →
  ∀ decO, Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) decO →
  ∀ x : Msg, DenseAt x →
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (Claims.run FlockDeploy.sha512 P (recKS b aux nOf hd recRest).V x ω r).pLog.length +
      (Claims.run FlockDeploy.sha512 P (recKS b aux nOf hd recRest).V x ω r).vLog.length ≤ t) →
    (∀ hk : 0 < (recKS b aux nOf hd recRest).V.k x,
      Claims.StrictBirthday FlockDeploy.sha512
        (Claims.B FlockDeploy.sha512 (recKS b aux nOf hd recRest).V P x hk)) →
    (∀ hk : 0 < (Claims.Recursion.Inner.ofFlock bdKS.deploy decI).V.k
        (Claims.Recursion.xI (recKS b aux nOf hd recRest) x),
      Claims.ExpectedCR FlockDeploy.sha512
        ((Claims.Recursion.BinnerV FlockDeploy.sha512 (recKS b aux nOf hd recRest)
          (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO P x
          hk).charge (Claims.Recursion.setupLen (recKS b aux nOf hd recRest) x + 2 * (nR + 2 ^ 55)))) →
    (∀ (M : ℕ) (hk : 0 < (Claims.Recursion.Inner.ofFlock bdKS.deploy decI).V.k
        (Claims.Recursion.xI (recKS b aux nOf hd recRest) x)),
      Claims.ExpectedCR FlockDeploy.sha512
        ((Claims.Recursion.FdevV FlockDeploy.sha512 (recKS b aux nOf hd recRest)
          (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO P x M
          hk).charge (Claims.Recursion.setupLen (recKS b aux nOf hd recRest) x + 2 * (nR + 2 ^ 55)))) →
    (∀ (M : ℕ) (hW : 0 < FlockDeploy.KS.CpDep.width (Claims.Recursion.xI (recKS b aux nOf hd recRest) x)),
      Claims.ExpectedCR FlockDeploy.sha512
        ((Claims.Recursion.FCI FlockDeploy.sha512 (recKS b aux nOf hd recRest)
          (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO
          FlockDeploy.KS.CpDep P x M hW).charge
          (Claims.Recursion.setupLen (recKS b aux nOf hd recRest) x + 2 * (nR + 2 ^ 55)))) →
    (∀ R : ℕ, Claims.ExpectedCR FlockDeploy.sha512
      ((Claims.Recursion.F0I FlockDeploy.sha512 (recKS b aux nOf hd recRest)
        (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO P x
        R).charge (Claims.Recursion.setupLen (recKS b aux nOf hd recRest) x +
          ((Claims.Recursion.Inner.ofFlock bdKS.deploy decI).k
            (Claims.Recursion.xI (recKS b aux nOf hd recRest) x) + 1) * (nR + 2 ^ 55)))) →
    Claims.PrBad FlockDeploy.sha512 (recKS b aux nOf hd recRest).V
        (Claims.Recursion.R (recKS b aux nOf hd recRest) (Claims.Recursion.Inner.ofFlock bdKS.deploy decI))
        (Claims.Recursion.E FlockDeploy.sha512 (recKS b aux nOf hd recRest)
          (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO) x P ≤
      2 ^ (-(140.4 : ℝ)) + t * 2 ^ (-(138.3 : ℝ))

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−140.4} + t·2^{−138.3}` at the deployed reader**, at the deployed brings `deployedB b`, every
other field of the brings and every record the coin server keeps: `recursiveBitsK` at `n = nR` and `v = 2^55`, whose
call premises `lamLin2_R_ge_in` proves at a run the compiled verifier accepts, with `kappa_eps_deployed_dep_tight` at
the caps that run gives. -/
theorem recursiveBitsK_tight (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    RecursiveBitsKTightAt (deployedB b) aux nOf hd recRest := by
  intro decI hdecI decO hdecO x hx Ω _ _ P t ht1 hrun hSB hBI hFdev hFCI hF0I
  have hpos : (0 : ℝ) ≤ 2 ^ (-(140.4 : ℝ)) + t * 2 ^ (-(138.3 : ℝ)) := by positivity
  by_cases hex : ∃ ω r,
      (Claims.run FlockDeploy.sha512 P (recKS (deployedB b) aux nOf hd recRest).V x ω r).out = .acc
  · obtain ⟨ω, r, hacc⟩ := hex
    have hacc' : (Claims.run FlockDeploy.sha512 P
        (V bdKS.IVStar (deployedB { b with tr := recTr aux nOf hd recRest }).st) x ω r).out = .acc := hacc
    obtain ⟨s, spec, sts, hin⟩ := plainIn_of_run_acc hacc'
    have hO := partIn_of_run_acc hacc'
    have hc := lamLin2_R_ge_in _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
      (DeployedR.layR OpenRows.elemR) DeployedR.rdR
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
      FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR decI FlockDeploy.sha512 hacc'
    obtain ⟨hκ, hε⟩ := kappa_eps_deployed_dep_tight _root_.Flock.Readable.relation
      (deployedB { b with tr := recTr aux nOf hd recRest }) (DeployedR.layR OpenRows.elemR)
      (FlockDeploy.nextR DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
      FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR decI hin hO
      ((nS_le_of_partIn hO).trans (by decide)) (innerSetupIns_le_acc hacc') nR_le (le_refl (2 ^ 55))
    exact (recursiveBitsK b aux nOf hd recRest decI hdecI decO hdecO x hx Ω P t ht1 hrun nR (hc []).1 (2 ^ 55)
      (fun ps => (hc ps).2.2) hSB hBI hFdev hFCI hF0I).trans
      (add_le_add hκ (mul_le_mul_of_nonneg_left hε (Nat.cast_nonneg t)))
  · unfold Claims.PrBad
    refine le_trans (Claims.Pr_mono (q := fun _ => False) fun q hq => ?_) (Claims.Pr_false.le.trans hpos)
    exact hex ⟨q.1, q.2.1, hq.1⟩

end RecDeploy
