module

public import Proofs.Flock.Claims.Recursion.DeployedBrings
public import Proofs.Flock.Claims.Recursion.InnerSetupCount
public import Proofs.Flock.Claims.Recursion.KappaEpsDeployed
public import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.AcceptedScope
import Proofs.Flock.Claims.Recursion.CallsClosed
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.Recursion.DeployedCounts
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Recursive.BitsK

/-!
# `Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at every statement the compiled verifier accepts

`recursiveBitsK_tight`: `recursiveBitsK` (`Flock.SecurityProofs.RecursiveKS` through
`Claims.Recursion.bitsLin2KOn_of_ksLin2On`) with `kappa_eps_deployed` at the deployment's sizes, every one of them, and
both of `BitsLin2KOn`'s call bounds read from a run the compiled verifier accepts. Where none accepts at `x`,
`Pr[acc ∧ ¬R] = 0`. Where one does:

* `x` is at the deployment's sizes (`sizesAt_of_run_acc`): the inner statement is one plain table (`plainIn_of_run_acc`,
  `innerSetupCalls_le_acc`), each staged V\* part one `--zk` table (`partIn_of_run_acc`, `RecSetup.oneFull`), so at
  most `RecBudget.MAX_STATEMENTS = 16` parts (`nS_le_of_partIn`), and the inner setup has at most `2^32` inputs
  (`innerSetupIns_le_acc`);
* the inner verifier's calls are at most `nR = 2^28 + (c_B + 2^24) ≤ 2^32` on every run (`n`, `lamLin2_R_ge_in`) and
  the compiled verifier's own at its V\* statements at most `16 · 138543104 ≤ 2^32` (`v`,
  `setupCalls_deployedB_le_32`), counts of the verifier's code at `RecBudget`'s caps;
* at those sizes and `n, v ≤ 2^32`, `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}` (`kappa_eps_deployed`).

What is left is what the statement names: the decoders' list-decoding spec (`decOK`), the prover's calls `t ≥ 1` per
run, and SHA-512's named collision assumptions, `StrictBirthday` at `B` and `ExpectedCR` at `B_I`, `Fdev`, `FCI` and
`F0I`, each finder charged the verifier's own calls at `n = nR` and `v = 2^32`.
-/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)

/-- **The inner verifier's calls at an accepted statement**, `lamLin2_R_ge_in`'s count: `2^28`, the inner setup's
`c_B = SetupCount.innerB` and a decision's `c_V = 2^24`. -/
abbrev nR : ℕ := 2 ^ 28 + (SetupCount.innerB + 2 ^ 24)

theorem nR_le : nR ≤ 2 ^ 32 := by
  show 2 ^ 28 + (SetupCount.innerB + 2 ^ 24) ≤ 2 ^ 32
  rw [SetupCount.innerB_eq]
  norm_num

/-- **Every statement the compiled verifier accepts is at the deployment's sizes**: on a run of the recursive verifier
at the brings `b` that accepts `x`, the inner statement is one plain table, each V\* part `x` stages is one `--zk`
table, and the inner setup has at most `2^32` inputs (`SizesAt b x`). Each is a refusal of the verifier's: the inner
setup's lookup tables (`RecSetup.checkForm`), a part's tables past one (`RecSetup.oneFull`), and the inner public
file's instances and shared rows past `RecBudget.MAX_INNER_ROWS` (`RecBudget.innerRows`). -/
theorem sizesAt_of_run_acc {H : Msg → Msg} {IV : Claims.Verifier FlockDeploy.Coin} {b : Brings} {x : Msg} {Ω : Type}
    {P : Claims.Prover Ω} {ω : Ω} {r : Fin ((V IV b.st).k x) → FlockDeploy.Coin}
    (hacc : (Claims.run H P (V IV b.st) x ω r).out = .acc) : SizesAt b x := by
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_run_acc hacc
  obtain ⟨s', spec', sts', hset, h1, -⟩ := innerSetupCalls_le_acc hacc
  have e : sts' = sts := congrArg (fun v => v.2.2) (Option.some.inj (hset.symm.trans hin.setup))
  refine ⟨⟨s, spec, sts, hin, by rw [← e]; exact h1⟩, fun j hj => ?_, innerSetupIns_le_acc hacc⟩
  obtain ⟨i, spec₀, sts₀, hp⟩ := partIn_of_run_acc hacc j hj
  obtain ⟨hins, hsu⟩ := setupO_some hp.setup
  obtain ⟨xr, -, hinner, -⟩ := insO_staged hins
  have hO := RecIns.setup_oneFull hsu
  rw [hinner] at hO
  exact ⟨i, spec₀, sts₀, hp, (oneFull_ok hO).1⟩

/-- **The deployed bound record** `Flock.Guarantees.RecursiveKS` is stated at: the deployed reader's steps and
trimmed columns, V\*'s last message's reader and the deployed weights. -/
noncomputable abbrev bdKS :=
  bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER FlockDeploy.joinR
    rowInDeployed DeployedR.finalR DeployedR.wR

/-- **The compiled record** `Flock.Guarantees.RecursiveKSAt` is stated at, at the brings `b` and the challenger's live-coin part's
record, its inner view running `bdKS`'s `IVStar`. -/
noncomputable abbrev recKS (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    Claims.Recursion.Deploy FlockDeploy.Coin :=
  (recBoundAt _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
    (DeployedR.layR OpenRows.elemR)).record bdKS.IVStar

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at every statement the compiled verifier accepts**, at the brings
`b` and the challenger's live-coin part's record (`aux`, `nOf`, `hd`, `recRest`): for every inner and outer decoder meeting the
list-decoding spec (`decOK`), every dense statement `x`, every prover `P` making at most `t ≥ 1` SHA-512 calls per run,
under `StrictBirthday` at `B` and `ExpectedCR` at `B_I`, `Fdev`, `FCI` and `F0I`, each charged `|S(x_I)|` and the
verifier's own calls at `n = nR` and `v = 2^32`. -/
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
          hk).charge (Claims.Recursion.setupLen (recKS b aux nOf hd recRest) x + 2 * (nR + 2 ^ 32)))) →
    (∀ (M : ℕ) (hk : 0 < (Claims.Recursion.Inner.ofFlock bdKS.deploy decI).V.k
        (Claims.Recursion.xI (recKS b aux nOf hd recRest) x)),
      Claims.ExpectedCR FlockDeploy.sha512
        ((Claims.Recursion.FdevV FlockDeploy.sha512 (recKS b aux nOf hd recRest)
          (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO P x M
          hk).charge (Claims.Recursion.setupLen (recKS b aux nOf hd recRest) x + 2 * (nR + 2 ^ 32)))) →
    (∀ (M : ℕ) (hW : 0 < FlockDeploy.KS.CpDep.width (Claims.Recursion.xI (recKS b aux nOf hd recRest) x)),
      Claims.ExpectedCR FlockDeploy.sha512
        ((Claims.Recursion.FCI FlockDeploy.sha512 (recKS b aux nOf hd recRest)
          (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO
          FlockDeploy.KS.CpDep P x M hW).charge
          (Claims.Recursion.setupLen (recKS b aux nOf hd recRest) x + 2 * (nR + 2 ^ 32)))) →
    (∀ R : ℕ, Claims.ExpectedCR FlockDeploy.sha512
      ((Claims.Recursion.F0I FlockDeploy.sha512 (recKS b aux nOf hd recRest)
        (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO P x
        R).charge (Claims.Recursion.setupLen (recKS b aux nOf hd recRest) x +
          ((Claims.Recursion.Inner.ofFlock bdKS.deploy decI).k
            (Claims.Recursion.xI (recKS b aux nOf hd recRest) x) + 1) * (nR + 2 ^ 32)))) →
    Claims.PrBad FlockDeploy.sha512 (recKS b aux nOf hd recRest).V
        (Claims.Recursion.R (recKS b aux nOf hd recRest) (Claims.Recursion.Inner.ofFlock bdKS.deploy decI))
        (Claims.Recursion.E FlockDeploy.sha512 (recKS b aux nOf hd recRest)
          (Claims.Recursion.Inner.ofFlock bdKS.deploy decI) (deployZkRec _root_.Flock.Readable.relation) decO) x P ≤
      2 ^ (-(167.7 : ℝ)) + t * 2 ^ (-(174.95 : ℝ))

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at every statement the compiled verifier accepts**, at the deployed
brings `deployedB b`, every other field of the brings and every record the challenger's live-coin part keeps: `recursiveBitsK` at
`n = nR` and `v = 2^32`, whose call premises `lamLin2_R_ge_in` and `setupCalls_deployedB_le_32` prove at a run the
compiled verifier accepts, with `kappa_eps_deployed` at the sizes that run gives (`sizesAt_of_run_acc`). -/
theorem recursiveBitsK_tight (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    RecursiveBitsKTightAt (deployedB b) aux nOf hd recRest := by
  intro decI hdecI decO hdecO x hx Ω _ _ P t ht1 hrun hSB hBI hFdev hFCI hF0I
  have hpos : (0 : ℝ) ≤ 2 ^ (-(167.7 : ℝ)) + t * 2 ^ (-(174.95 : ℝ)) := by positivity
  by_cases hex : ∃ ω r,
      (Claims.run FlockDeploy.sha512 P (recKS (deployedB b) aux nOf hd recRest).V x ω r).out = .acc
  · obtain ⟨ω, r, hacc⟩ := hex
    have hacc' : (Claims.run FlockDeploy.sha512 P
        (V bdKS.IVStar (deployedB { b with tr := recTr aux nOf hd recRest }).st) x ω r).out = .acc := hacc
    obtain ⟨⟨s, spec, sts, hin, h1⟩, hO, hS⟩ :=
      sizesAt_of_run_acc (b := deployedB { b with tr := recTr aux nOf hd recRest }) hacc'
    have hn := nS_le_sixteen hO
    have hc := lamLin2_R_ge_in _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
      (DeployedR.layR OpenRows.elemR) DeployedR.rdR
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
      FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR decI FlockDeploy.sha512 hacc'
    obtain ⟨hκ, hε⟩ := kappa_eps_deployed _root_.Flock.Readable.relation
      (deployedB { b with tr := recTr aux nOf hd recRest }) (DeployedR.layR OpenRows.elemR)
      (FlockDeploy.nextR DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
      FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR decI hin h1 hO hS nR_le (le_refl (2 ^ 32))
    exact (recursiveBitsK b aux nOf hd recRest decI hdecI decO hdecO x hx Ω P t ht1 hrun nR (hc []).1 (2 ^ 32)
      (fun ps => setupCalls_deployedB_le_32 _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
        (DeployedR.layR OpenRows.elemR) bdKS.IVStar FlockDeploy.sha512 x ps hn)
      hSB hBI hFdev hFCI hF0I).trans
      (add_le_add hκ (mul_le_mul_of_nonneg_left hε (Nat.cast_nonneg t)))
  · unfold Claims.PrBad
    refine le_trans (Claims.Pr_mono (q := fun _ => False) fun q hq => ?_) (Claims.Pr_false.le.trans hpos)
    exact hex ⟨q.1, q.2.1, hq.1⟩

end RecDeploy
