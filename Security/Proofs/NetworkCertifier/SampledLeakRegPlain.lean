module

public import Definitions.NetworkCertifier.SampledCharge
public import Definitions.NetworkCertifier.SampledReg
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.SamplingGeneric
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
import Proofs.Flock.Claims.Sampling.DeployedSha512
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.SampledLeakReg

/-!
# The leak bound at the registration's values, at the deployed plain record

`sampledLeak_registered` is stated at any record. At the deployed sampling record over any Flock record `F`
(`SamplingDeploy.deployOf F c`) its trace commitment is `F`-free (`SamplingDeploy.deployOf_commit`), hm96 leaves under the
verifier's own registered path check, and binds at every `H` with no hypothesis (`binds_deployOf`, from
`Binding_of_regPath`). So the masked and plain records share one commitment (`deployPlain_commit`), and at the plain
record (`SamplingDeploy.deployPlain`) the bound takes only the knowledge soundness of the record's compiled sampled
verifier (`Compiled.KS` at `ofDeploy`, `sampledLeak_registered_plain`), which `Claims.SamplingGeneric.ks_ofDeploy` gives
at SHA-512 at the deployed relation from Flock's plain KS (`sampledLeak_registered_plain_sha512`).
`SampledAuditAccepted` is discharged by its proof.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling
open Claims.SamplingGeneric (ofDeploy)

/-- **The deployed commitment binds, at any Flock record**: hm96 leaves under the registered path check, at every `H`. -/
theorem binds_deployOf {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (c : SamplingDeploy.Code) (H : Msg → Msg) : (SamplingDeploy.deployOf F c).commit.Binds H :=
  Binding_of_regPath _ c.own c.at_ _ c.π c.layer (SamplingDeploy.deployOf_commit F c)
    (fun H => hm96Leaf_binds H (SamplingDeploy.row_places c.place c.bits) c.salt) H

/-- The masked and plain records' trace commitment is one. -/
theorem deployPlain_commit (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (c : SamplingDeploy.Code) : (SamplingDeploy.deployPlain rel c).commit = (SamplingDeploy.deploy rel c).commit :=
  rfl

section Plain

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : SamplingDeploy.Code)

local notation "Dp" => SamplingDeploy.deployPlain rel c

set_option synthInstance.maxSize 1024 in
/-- **The leak bound at the registration's values, at the deployed plain record**: `sampledLeak_registered` at
`SamplingDeploy.deployPlain rel c`, its binding `binds_deployOf` and `SampledAuditAccepted` its proof, given the
knowledge soundness of the record's compiled sampled verifier. -/
theorem sampledLeak_registered_plain (H : Msg → Msg) (iroots : Msg → Seen → ℕ → List Msg)
    (hks : (ofDeploy Dp iroots).KS H) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (δ : ℝ) (hδ : 0 ≤ δ) (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ)
    (x : Msg) (hx : ScopeAcc Dp lay k x)
    (hreg : ∀ st, (Dp).parse x = some st → RegOpens H (Dp).commit st (reg st) (ops st)) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    PrBad H (Dp).V
        (RLeak H (Dp).commit (Dp).parse (Dp).partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj) (ofDeploy Dp iroots).E x P ≤
      (ofDeploy Dp iroots).κ δ x + Real.sqrt ((ofDeploy Dp iroots).γ x * PrB H (Dp).V P x) +
        (Bext H (Dp).V (ofDeploy Dp iroots).E Ω P x).Pr H +
        (Breg H (Dp).V (ofDeploy Dp iroots).E (Dp).commit (Dp).parse ops Ω P x).Pr H :=
  sampledLeak_registered SecurityProofs.SampledAuditAccepted H Dp iroots hks (binds_deployOf _ c H) L K S lay k δ hδ
    reg ops Dj x hx hreg Ω P

end Plain

section Sha512

local notation "relD" => _root_.Flock.Readable.relation
local notation "shaD" => FlockDeploy.sha512

variable (c : SamplingDeploy.Code)

local notation "Dp" => SamplingDeploy.deployPlain relD c

set_option synthInstance.maxSize 1024 in
/-- **The same at SHA-512 at the deployed relation**: the record's `Compiled.KS` from Flock's plain KS
(`FlockDeploy.KS.plain_KS`), its decoder's list bound, its binding and its frozen facts
(`Claims.SamplingGeneric.ks_ofDeploy`). -/
theorem sampledLeak_registered_plain_sha512
    (hfixed : DrawFixed Dp) (hsessions : SessionRuns shaD Dp) (hnum : SamplingDeploy.SessionsInPlain c)
    (hslots : DrawSlots Dp) (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens shaD Dp iroots)
    (hfix : InteriorFixed shaD Dp iroots) (hprog : ProgramScope Dp) (hchecks : ChecksScope shaD Dp) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ) (x : Msg)
    (hx : ScopeAcc Dp lay k x)
    (hreg : ∀ st, (Dp).parse x = some st → RegOpens shaD (Dp).commit st (reg st) (ops st)) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    PrBad shaD (Dp).V
        (RLeak shaD (Dp).commit (Dp).parse (Dp).partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj) (ofDeploy Dp iroots).E x P ≤
      (ofDeploy Dp iroots).κ δ x + Real.sqrt ((ofDeploy Dp iroots).γ x * PrB shaD (Dp).V P x) +
        (Bext shaD (Dp).V (ofDeploy Dp iroots).E Ω P x).Pr shaD +
        (Breg shaD (Dp).V (ofDeploy Dp iroots).E (Dp).commit (Dp).parse ops Ω P x).Pr shaD :=
  sampledLeak_registered_plain relD c shaD iroots
    (Claims.SamplingGeneric.ks_ofDeploy iroots FlockDeploy.KS.plain_KS (listWithin_deployPlainTrim relD)
      (binds_deployOf _ c shaD) hopens hfix (drawLaw_of_slots _ hslots)
      (refuses_of shaD _ (scopeMatches_of _ hslots hprog) hchecks) hfixed hsessions
      (SamplingDeploy.innerNumbers_deployPlain relD c hnum))
    L K S lay k δ hδ reg ops Dj x hx hreg Ω P

end Sha512

end NetTiming
