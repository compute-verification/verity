import Proofs.Properties.Verity.Recursive.Target
import Proofs.Flock.Claims.Recursion.CallsClosed
import Proofs.Flock.Recursive.KSProved
import Proofs.Flock.Recursive.FirewallZK

/-!
# The recursive audit over sampled proofs' recursive form

`Flock.Guarantees.RecursiveAuditC3` is sampled proofs at the recursive configuration, the sample checked by V\*'s
compiled check (`verity/sampled-proofs/recursive/v1`, the recursive form of onsite's and remote's sampled-proofs
component), with zero knowledge (`RecursiveFirewallPrivate`).

* `SampledProofsRecursive Acc` is `sampledProofsRecursive`'s bound at every deployment `RecursiveKS` is stated at
  (`Dep.deployed`: the deployed weights, and any brings, coin-server record and decoders meeting the list-decoding spec),
  at the verdict `Acc`, with the call bounds `nI₀`, `vS₀` and the level `λ = 100` fixed. The guarantee takes it at the
  recursive configuration's verdict, `RecVerdict`.
* It follows from `RecursiveKS`, read by name (`sampledProofsRecursive_of_ks`): an accepted play shows an accepted run of
  the deployed recursive verifier (`verdict_run`), and `RecDeploy.lamLin2_R_ge_in` reads the call bounds and `λ₂ ≥ 100`
  off that run (`calls_lam`).

The form is dense only: `recursive/v1` refuses every other, and `RecursiveKSAt` covers the dense statements
(`DenseAt`). Candidate 3's Holo statement is `Flock.Guarantees.RecursiveAuditHolo`.
-/

namespace Verity.Recursive

open scoped ENNReal
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy
open Accounting (Device)

/-- **The deployment `RecursiveKS` is stated at**: the deployed weights `DeployedR.wER` and `DeployedR.wR`, with the
brings `b`, the coin server's record (`aux`, `nOf`, `hd`, `recRest`) and the decoders `decI` and `decO`. -/
noncomputable def Dep.deployed (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (decO : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace) : Dep :=
  ⟨DeployedR.wER, DeployedR.wR, b, aux, nOf, hd, recRest, decI, decO⟩

/-- **The inner verifier's SHA-512 calls on every run**, at most, at a statement some run accepts: `2^28 + (c_B + c_V)`,
with `c_B = SetupCount.innerB` the inner setup's and `c_V = 2^24` a decision's. -/
def nI₀ : Msg → ℕ := fun _ => 2 ^ 28 + (SetupCount.innerB + 2 ^ 24)

/-- **V\*'s setup calls**, at most, at a statement some run accepts. -/
def vS₀ : Msg → ℕ := fun _ => 2 ^ 55

/-- **The call bounds and the level at an accepted statement**: where some run of the deployed recursive verifier
accepts `x`, the inner verifier's calls are at most `nI₀ x` on every run, V\*'s setup calls at most `vS₀ x`, and route
2C's `λ₂` at those counts is at least `100` (`RecDeploy.lamLin2_R_ge_in`, every premise read from the run). -/
theorem calls_lam (q : Dep) {x : Msg} (h : q.RunAccepts x) :
    (∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI₀ x) ∧
      (100 : ℝ) ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI₀ x) (vS₀ x) ∧
      ∀ ps, (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS₀ x := by
  obtain ⟨Ω, P, ω, r, hacc⟩ := h
  have hacc' : (Claims.run FlockDeploy.sha512 P (RecDeploy.V (boundAt q.wE q.w).IVStar
      (deployedB { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest }).st) x ω r).out = .acc := hacc
  have H := fun ps => lamLin2_R_ge_in _root_.Flock.Readable.relation
    { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest } (DeployedR.layR OpenRows.elemR) DeployedR.rdR
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols q.wE FlockDeploy.joinR
    rowInDeployed DeployedR.finalR q.w q.decI FlockDeploy.sha512 hacc' ps
  exact ⟨(H []).1, (H []).2.1, fun ps => (H ps).2.2⟩

/-- **Sampled proofs at the recursive configuration, at the deployment `q` and the verdict `Acc`**: for decoders meeting
the list-decoding spec, at the parameters `θ`, the circuit file `F`, the commitment `c`, a strategy `σ` and its
recursive session `S` (`SessionIn`), the probability that `Acc` holds of the play's outcome while sampled proofs' claim
fails on the extracted transcript is at most `Dep.errAt` at `S`, the call bounds `nI₀`, `vS₀` and `λ = 100`. Its
hypotheses are `sampledProofsRecursive`'s: `K₀` covers the claim's `ε` share and the draw misses any `K₀` units with
probability at most `δs`; the vocabulary's extractor is the recursive configuration's (`Dep.extract`); the reading
`readX` of a witness in the recursive relation is correct at, and opened on, every drawn unit, and `collOf` turns two
readings that differ at a wire drawn in both into a SHA-512 collision; and the records hold the run's live coins
(`LiveCoins`, `live-verifier`). -/
def SampledProofsRecursiveAt
    (Acc : ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α], Vocab d α → Verity.Ending → Prop) (q : Dep) : Prop :=
  Claims.Recursion.decOK q.Bd.deploy q.decI →
  Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO →
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) (θ F c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (S : SessionIn V q θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl K₀ : ℕ) (δs : ℝ),
    (∀ m : ℕ, (V.certOf θ).ε * V.units F < m → K₀ ≤ m) →
    (∀ U : Finset (Fin (V.units F)), K₀ ≤ U.card → Claims.Pr (fun ξ : S.Ξ => Disjoint (S.drawOf ξ) U) ≤ δs) →
    V.extract θ F c σ = q.extract V S readX →
    (∀ ξ wt, V.Registered θ F → q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf F).Correct (readX (S.x ξ) wt) u ∧ ∀ g ∈ (V.partsOf F).io u, V.Opens θ F c g (readX (S.x ξ) wt g)) →
    (∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m') →
    (∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) →
    Game.prob (fun o => Acc V o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤
      q.errAt V S collOf dColl nI₀ vS₀ δs 100

/-- **Sampled proofs at the recursive configuration, at the verdict `Acc`**: `SampledProofsRecursiveAt` at every
deployment `RecursiveKS` is stated at. -/
def SampledProofsRecursive
    (Acc : ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α], Vocab d α → Verity.Ending → Prop) : Prop :=
  ∀ (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (decO : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace),
    SampledProofsRecursiveAt Acc (Dep.deployed b aux nOf hd recRest decI decO)

/-- `SampledProofsRecursiveAt` at the recursive configuration's verdict, from `RecursiveKSAt` at `q`'s brings:
`sampledProofsRecursive` with the call bounds and the level `calls_lam` reads off an accepted run. -/
theorem sampledProofsRecursive_at (q : Dep)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest) :
    SampledProofsRecursiveAt RecVerdict q := by
  intro hdecI hdecO ι d α _ V θ F c σ S readX collOf dColl K₀ δs hK₀ hdraw hX hread hcoll hlive
  exact sampledProofsRecursive q hKS hdecI hdecO nI₀ vS₀ 100 (fun _ h => (calls_lam q h).1)
    (fun _ h => (calls_lam q h).2.2) (fun _ h => (calls_lam q h).2.1) V θ F c σ S readX collOf dColl K₀ δs hK₀ hdraw
    hX hread hcoll hlive

/-- **`RecursiveKS` gives sampled proofs at the recursive configuration.** -/
theorem sampledProofsRecursive_of_ks (hKS : Flock.Guarantees.RecursiveKS) : SampledProofsRecursive RecVerdict :=
  fun b aux nOf hd recRest decI decO =>
    sampledProofsRecursive_at (Dep.deployed b aux nOf hd recRest decI decO) (hKS b aux nOf hd recRest)

end Verity.Recursive

namespace Flock.Guarantees

/-- **The recursive audit**: (i) sampled proofs at the recursive configuration (`SampledProofsRecursive` at the
recursive configuration's verdict `RecVerdict`): wherever sampled proofs' case in what the prover sent names the
recursive form, V\*'s compiled check (`verity/sampled-proofs/recursive/v1`), the probability that the play is accepted
while sampled proofs' claim fails on the extracted transcript is at most `δs + t'·2^{-λ} + bits·√(E[bindBudget]/2^256)`
where the finders built from the strategy meet `FinderCR`, and `⊤` elsewhere; and (ii) zero knowledge,
`RecursiveFirewallPrivate`.

Its `λ` is `λ₂ = 100`: route 2C's level (`Claims.Recursion.lamLin2`) at the inner calls `nI₀ = 2^28 + (c_B + 2^24)` and
V\*'s setup calls `vS₀ = 2^55`, both read off the accepted run (`RecDeploy.lamLin2_R_ge_in`). The bound record's level
(`Claims.Recursion.lam`, about 65 bits per SHA-512 call at the deployed shape) is below the 100-bit floor under every
weight, and (i) doesn't use it. -/
def RecursiveAuditC3 : Prop :=
  Verity.Recursive.SampledProofsRecursive Verity.Recursive.RecVerdict ∧ RecursiveFirewallPrivate

end Flock.Guarantees

namespace Flock.SecurityProofs

/-- **The recursive audit**, from `RecursiveKS` and `RecursiveFirewallPrivate`. -/
theorem RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3 :=
  ⟨Verity.Recursive.sampledProofsRecursive_of_ks RecursiveKS, RecursiveFirewallPrivate⟩

end Flock.SecurityProofs
