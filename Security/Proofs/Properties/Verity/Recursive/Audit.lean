module

public import Proofs.Properties.Verity.Recursive.Target
public import Proofs.Flock.Claims.Recursion.InnerSetupCount
public import Proofs.Flock.Recursive.Firewall
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
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
import Proofs.Flock.Claims.Recursion.DeployedCounts
import Proofs.Flock.Claims.Recursion.LamDeployedTight
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.KSProved

/-!
# The recursive audit over sampled proofs' recursive form

`Flock.Guarantees.RecursiveAuditC3` is sampled proofs at the recursive configuration, the sample checked by V\*'s
compiled check (`verity/sampled-proofs/recursive/v1`, the recursive form of onsite's and remote's sampled-proofs
component), at every composite, with zero knowledge (`RecursiveFirewallPrivate`).

* `SpotRecursive Acc` is `spotRecursive`'s bound at every deployment `RecursiveKS` is stated at
  (`Dep.deployed`: the deployed weights, and any brings, live-coin record and decoders meeting the list-decoding spec)
  and every composite `k`, at the verdict `Acc`, with the call bounds `nI₀`, `vS₀` and the level `λ = 135.7` fixed. The
  guarantee takes it at the recursive configuration's verdict, `RecVerdict`.
* Its knowledge-soundness term is `Dep.ksAt`: `t'·2^{-135.7}`, and at a sample whose statement is at the deployment's
  tables (`Dep.OneTable`: one plain inner table, and each V\* part one `--zk` table) the least of it and
  `2^{-167.7} + t'·2^{-174.95}` (`RecDeploy.kappa_eps_oneTable_cap`), where the inner setup's calls and `nI₀` are at
  most `2^32` and `vS₀ = nS·138543104` (`RecDeploy.setupCalls_deployedB_le_cap`) is at most `16·138543104 ≤ 2^32`,
  at the verifier's cap of `16` V\* parts (`RecDeploy.nS_le_sixteen`).
* It follows from `RecursiveKS`, read by name (`spotRecursive_of_ks`): an accepted play shows an accepted run of
  the deployed recursive verifier (`verdict_run`), off which `RecDeploy.lamLin2_R_ge_in` and
  `RecDeploy.setupCalls_deployedB_le_cap` read the call bounds and `RecDeploy.lamLin2_deployed_ge_dep_tight` reads
  `λ₂ ≥ 135.7` (`calls_lam`).

The form is dense only: `recursive/v1` refuses every other, and `RecursiveKSAt` covers the dense statements
(`DenseAt`). Candidate 3's Holo statement is `Flock.Guarantees.RecursiveAuditHolo`.
-/

@[expose] public section

namespace Verity.Recursive

open scoped ENNReal
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy
open Accounting (Device)

/-- **The deployment `RecursiveKS` is stated at**: the deployed weights `DeployedR.wER` and `DeployedR.wR`, with the
brings `b`, the challenger's live-coin part's record (`aux`, `nOf`, `hd`, `recRest`) and the decoders `decI` and `decO`. -/
noncomputable def Dep.deployed (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (decO : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace) : Dep :=
  ⟨DeployedR.wER, DeployedR.wR, b, aux, nOf, hd, recRest, decI, decO⟩

/-- **The inner verifier's SHA-512 calls on every run**, at most, at a statement some run accepts: `2^28 + (c_B + c_V)`,
with `c_B = SetupCount.innerB` the inner setup's and `c_V = 2^24` a decision's. -/
def nI₀ : Msg → ℕ := fun _ => 2 ^ 28 + (SetupCount.innerB + 2 ^ 24)

/-- **V\*'s setup calls**, at most, at a statement some run accepts: `138543104` a staged part
(`setupCalls_deployedB_le_cap`), at most `16 · 138543104` since the verifier refuses more than `16` parts
(`nS_le_of_partIn`). -/
def vS₀ : Msg → ℕ := fun x => nS x * 138543104

/-- **The call bounds and the level at an accepted statement**: where some run of the deployed recursive verifier
accepts `x`, the inner verifier's calls are at most `nI₀ x` on every run (`RecDeploy.lamLin2_R_ge_in`), V\*'s setup
calls at most `vS₀ x` (`RecDeploy.setupCalls_deployedB_le_cap`), route 2C's `λ₂` at those counts is at least `135.7`
(`RecDeploy.lamLin2_deployed_ge_dep_tight`), and at the deployment's tables the inner setup's calls and `nI₀ x` are at
most `2^32` (`RecDeploy.innerSetupIns_le_acc`, `SetupCount.innerB_eq`) and `vS₀ x` is `setupCalls_deployedB_le_cap`'s
`nS x · 138543104`, every premise read from the run. -/
theorem calls_lam (q : Dep) {x : Msg} (h : q.RunAccepts x) :
    (∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI₀ x) ∧
      (135.7 : ℝ) ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI₀ x) (vS₀ x) ∧
      (∀ ps, (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS₀ x) ∧
      (q.OneTable x →
        (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 ∧ nI₀ x ≤ 2 ^ 32 ∧ vS₀ x ≤ nS x * 138543104) := by
  obtain ⟨Ω, P, ω, r, hacc⟩ := h
  have hacc' : (Claims.run FlockDeploy.sha512 P (RecDeploy.V (boundAt q.wE q.w).IVStar
      (deployedB { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest }).st) x ω r).out = .acc := hacc
  have H := fun ps => lamLin2_R_ge_in _root_.Flock.Readable.relation
    { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest } (DeployedR.layR OpenRows.elemR) DeployedR.rdR
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols q.wE FlockDeploy.joinR
    rowInDeployed DeployedR.finalR q.w q.decI FlockDeploy.sha512 hacc' ps
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_run_acc hacc'
  have hO := partIn_of_run_acc hacc'
  have hnS : nS x ≤ 2 ^ 16 := (nS_le_of_partIn hO).trans (by decide)
  have hc : 2 ^ 28 + (SetupCount.innerB + 2 ^ 24) ≤ 2 ^ 32 := by
    rw [SetupCount.innerB_eq]
    norm_num
  have hV : vS₀ x ≤ 2 ^ 55 := by
    show nS x * 138543104 ≤ 2 ^ 55
    exact (Nat.mul_le_mul_right _ hnS).trans (by norm_num)
  have hset : (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 := innerSetupIns_le_acc hacc'
  have hcap : q.OneTable x →
      (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 ∧ nI₀ x ≤ 2 ^ 32 ∧ vS₀ x ≤ nS x * 138543104 :=
    fun _ => ⟨hset, hc, le_rfl⟩
  exact ⟨(H []).1, lamLin2_deployed_ge_dep_tight _root_.Flock.Readable.relation
    (deployedB { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest }) (DeployedR.layR OpenRows.elemR)
    (FlockDeploy.nextR DeployedR.rdR) (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols
    q.wE FlockDeploy.joinR rowInDeployed DeployedR.finalR q.w q.decI hin hO hnS (hset.trans (by norm_num))
    (hc.trans (by norm_num)) hV,
    fun ps => setupCalls_deployedB_le_cap _root_.Flock.Readable.relation
      { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest } (DeployedR.layR OpenRows.elemR)
      (boundAt q.wE q.w).IVStar FlockDeploy.sha512 x ps, hcap⟩

/-- **Sampled proofs at the recursive configuration, at the deployment `q` and the verdict `Acc`**: for decoders meeting
the list-decoding spec, at every composite `k` (as `Verity.SoundnessOf` quantifies it), the parameters `θ`, the
committed files `F`, the commitment `c`, a strategy `σ` and its recursive session `S` at `k` (`SessionIn`), the
probability that `Acc` at `k` holds of the play's outcome while sampled proofs' claim at `k` fails on the extracted
transcript is at most `Dep.errAt` at `S`, the call bounds `nI₀`, `vS₀` and `λ = 135.7`. Its
hypotheses are `spotRecursive`'s: `K₀` covers the claim's `ε` share and the sample misses any `K₀` units with
probability at most `δs`; the vocabulary's extractor is the recursive configuration's (`Dep.extract`); the reading
`readX` of a witness in the recursive relation is correct at, and opened on, every sampled proof block, and `collOf` turns two
readings that differ at a wire sampled in both into a SHA-512 collision; and the records hold the run's live coins
(`LiveCoins`, `live-verifier`). -/
def SpotRecursiveAt
    (Acc : ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α], Vocab d α → Verity.Registration.Composite →
      Verity.Exchange → Prop) (q : Dep) : Prop :=
  Claims.Recursion.decOK q.Bd.deploy q.decI →
  Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO →
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) (k : Verity.Registration.Composite)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (S : SessionIn V q k θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl K₀ : ℕ) (δs : ℝ),
    (∀ m : ℕ, (V.certOf θ).ε * V.units θ F < m → K₀ ≤ m) →
    (∀ U : Finset (Fin (V.units θ F)), K₀ ≤ U.card → Claims.Pr (fun ξ : S.Ξ => Disjoint (S.drawOf ξ) U) ≤ δs) →
    V.extract θ F c σ = q.extract V S readX →
    (∀ ξ wt, V.Registered θ F → q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g)) →
    (∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m') →
    (∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) →
    Game.prob (fun o => Acc V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      q.errAt V S collOf dColl nI₀ vS₀ δs 135.7

/-- **Sampled proofs at the recursive configuration, at the verdict `Acc`**: `SpotRecursiveAt` at every
deployment `RecursiveKS` is stated at, at every composite. -/
def SpotRecursive
    (Acc : ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α], Vocab d α → Verity.Registration.Composite →
      Verity.Exchange → Prop) : Prop :=
  ∀ (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (decO : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace),
    SpotRecursiveAt Acc (Dep.deployed b aux nOf hd recRest decI decO)

/-- `SpotRecursiveAt` at the recursive configuration's verdict, from `RecursiveKSAt` at `q`'s brings:
`spotRecursive` with the call bounds and the level `calls_lam` reads off an accepted run. -/
theorem spotRecursive_at (q : Dep)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest) :
    SpotRecursiveAt RecVerdict q := by
  intro hdecI hdecO ι d α _ V k θ F c σ S readX collOf dColl K₀ δs hK₀ hdraw hX hread hcoll hlive
  exact spotRecursive q hKS hdecI hdecO nI₀ vS₀ 135.7 (fun _ h => (calls_lam q h).1)
    (fun _ h => (calls_lam q h).2.2.1) (fun _ h => (calls_lam q h).2.1) (fun _ h => (calls_lam q h).2.2.2) V k θ F c
    σ S readX collOf dColl K₀ δs hK₀ hdraw hX hread hcoll hlive

/-- **`RecursiveKS` gives sampled proofs at the recursive configuration.** -/
theorem spotRecursive_of_ks (hKS : Flock.Guarantees.RecursiveKS) : SpotRecursive RecVerdict :=
  fun b aux nOf hd recRest decI decO =>
    spotRecursive_at (Dep.deployed b aux nOf hd recRest decI decO) (hKS b aux nOf hd recRest)

end Verity.Recursive

namespace Flock.Guarantees

/-- **The recursive audit**: (i) sampled proofs at the recursive configuration (`SpotRecursive` at the
recursive configuration's verdict `RecVerdict`), at every composite `k`: wherever sampled proofs' case in what the
prover sent `k` names the recursive form, V\*'s compiled check (`verity/sampled-proofs/recursive/v1`), the probability
that the play is accepted while sampled proofs' claim fails on the extracted transcript is at most
`δs + E_ξ[ksAt] + √(min(bits, 2n)·E[bindBudget]/2^256)`, `n` the units, where the finders built from the strategy
meet `FinderCR`, and `⊤` elsewhere; and (ii) zero knowledge, `RecursiveFirewallPrivate`.

Its knowledge-soundness term `ksAt` (`Verity.Recursive.Dep.ksAt`) is `t'·2^{-λ}` with `λ = λ₂ = 135.7`, route 2C's
level (`Claims.Recursion.lamLin2`), and at a sample whose statement is at the deployment's tables (`Dep.OneTable`) the
least of it and route 2C's split bound `κ + t'·ε` with `κ = 2^{-167.7}` and `ε = 2^{-174.95}`
(`RecDeploy.kappa_eps_oneTable_cap`, at the inner setup and the call bounds the run shows there). The finders are
charged the inner calls `nI₀ = 2^28 + (c_B + 2^24)` and V\*'s setup calls `vS₀ = nS·138543104`, read off the
accepted run with the deployed caps (`RecDeploy.lamLin2_R_ge_in`, `RecDeploy.setupCalls_deployedB_le_cap`,
`RecDeploy.lamLin2_deployed_ge_dep_tight`). The bound record's level (`Claims.Recursion.lam`), which no theorem
evaluates at the deployment, is one (i) doesn't use. -/
def RecursiveAuditC3 : Prop :=
  Verity.Recursive.SpotRecursive Verity.Recursive.RecVerdict ∧ RecursiveFirewallPrivate

end Flock.Guarantees

namespace Flock.SecurityProofs

/-- **The recursive audit**, from `RecursiveKS` and `RecursiveFirewallPrivate`. -/
theorem RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3 :=
  ⟨Verity.Recursive.spotRecursive_of_ks RecursiveKS, RecursiveFirewallPrivate⟩

end Flock.SecurityProofs
