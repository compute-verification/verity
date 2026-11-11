module

public import Definitions.NetworkCertifier.SampledCharge
public import Definitions.NetworkCertifier.SampledReg
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.SamplingGeneric.Deployed
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
import Proofs.Flock.Claims.KS.PlainLeafKS
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Claims.Sampling.Select.DrawLaw
import Proofs.Flock.Claims.Sampling.Select.Scope
import Proofs.Flock.SamplingGeneric
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.SampledLeakReg

/-!
# The leak bound at the registration's values, at the deployed plain record

`sampledLeak_registered` is stated at any compiled sampled verifier. At the deployed sampling record over any Flock
record `F` (`SamplingDeploy.deployOf F c`) its trace commitment is `F`-free (`SamplingDeploy.deployOf_commit`), hm96
leaves under the verifier's own registered path check, and binds at every `H` with no hypothesis (`binds_deployOf`, from
`SamplingDeploy.commit_binds`). So the masked and plain records share one commitment (`deployPlain_commit`), and at the
plain record as a compiled verifier (`SamplingDeploy.compiled` at `deployPlainTrim`) the bound takes only its
`Compiled.KS` (`sampledLeak_registered_plain`), which the compiler at Flock gives at SHA-512 at the deployed relation
from Flock's plain KS (`SamplingDeploy.ks_compiled`, `FlockDeploy.KS.plain_KS`) and the code's facts
(`sampledLeak_registered_plain_sha512`). `SampledAuditAccepted` is discharged by its proof.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

/-- **The deployed commitment binds, at any Flock record**: hm96 leaves under the registered path check, at every `H`. -/
theorem binds_deployOf {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (c : SamplingDeploy.Code) (H : Msg → Msg) : (SamplingDeploy.deployOf F c).commit.Binds H :=
  SamplingDeploy.commit_binds c H

/-- The masked and plain records' trace commitment is one. -/
theorem deployPlain_commit (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (c : SamplingDeploy.Code) : (SamplingDeploy.deployPlain rel c).commit = (SamplingDeploy.deploy rel c).commit :=
  rfl

section Plain

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : SamplingDeploy.Code)
  (iroots : Msg → Seen → ℕ → List Msg)

local notation "Lp" => SamplingDeploy.compiled (FlockDeploy.deployPlainTrim rel) c iroots

set_option synthInstance.maxSize 1024 in
/-- **The leak bound at the registration's values, at the deployed plain record**: `sampledLeak_registered` at
`SamplingDeploy.deployPlain rel c` as a compiled verifier, its binding `binds_deployOf` and `SampledAuditAccepted` its
proof, given its `Compiled.KS`. -/
theorem sampledLeak_registered_plain (H : Msg → Msg) (hks : (Lp).KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ) (x : Msg)
    (hx : ScopeAcc Lp lay k x) (hreg : ∀ st, (Lp).parse x = some st → RegOpens H (Lp).commit st (reg st) (ops st))
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBad H (Lp).V
        (RLeak H (Lp).commit (Lp).parse (Lp).partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj) (Lp).E x P ≤
      (Lp).κ δ x + Real.sqrt ((Lp).γ x * PrB H (Lp).V P x) + (Bext H (Lp).V (Lp).E Ω P x).Pr H +
        (Breg H (Lp).V (Lp).E (Lp).commit (Lp).parse ops Ω P x).Pr H :=
  sampledLeak_registered SecurityProofs.SampledAuditAccepted H Lp hks
    (binds_deployOf (FlockDeploy.deployPlainTrim rel) c H) L K S lay k δ hδ reg ops Dj x hx hreg Ω P

end Plain

section Sha512

local notation "relD" => _root_.Flock.Readable.relation
local notation "shaD" => FlockDeploy.sha512

variable (c : SamplingDeploy.Code) (iroots : Msg → Seen → ℕ → List Msg)

local notation "Dp" => SamplingDeploy.deployPlain relD c
local notation "Lp" => SamplingDeploy.compiled (FlockDeploy.deployPlainTrim relD) c iroots

set_option synthInstance.maxSize 1024 in
/-- **The same at SHA-512 at the deployed relation**: the plain record's `Compiled.KS` from the compiler at Flock
(`SamplingDeploy.ks_compiled`), at Flock's plain KS (`FlockDeploy.KS.plain_KS`), its decoder within the radius
(`listWithin_deployPlainTrim`), the commitment's binding and the code's facts. -/
theorem sampledLeak_registered_plain_sha512
    (hfixed : DrawFixed Dp) (hsessions : SessionRuns shaD Dp) (hnum : SamplingDeploy.SessionsInPlain c)
    (hslots : DrawSlots Dp) (hopens : SessionOpens shaD Dp iroots)
    (hfix : InteriorFixed shaD Dp iroots) (hprog : ProgramScope Dp) (hchecks : ChecksScope shaD Dp) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ) (x : Msg)
    (hx : ScopeAcc Lp lay k x) (hreg : ∀ st, (Lp).parse x = some st → RegOpens shaD (Lp).commit st (reg st) (ops st))
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBad shaD (Lp).V
        (RLeak shaD (Lp).commit (Lp).parse (Lp).partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj) (Lp).E x P ≤
      (Lp).κ δ x + Real.sqrt ((Lp).γ x * PrB shaD (Lp).V P x) + (Bext shaD (Lp).V (Lp).E Ω P x).Pr shaD +
        (Breg shaD (Lp).V (Lp).E (Lp).commit (Lp).parse ops Ω P x).Pr shaD :=
  sampledLeak_registered_plain relD c iroots shaD
    (SamplingDeploy.ks_compiled (FlockDeploy.deployPlainTrim relD) c iroots shaD FlockDeploy.KS.plain_KS
      (listWithin_deployPlainTrim relD) (binds_deployOf (FlockDeploy.deployPlainTrim relD) c shaD)
      (drawLaw_of_slots Dp hslots)
      (refuses_of shaD Dp (scopeMatches_of Dp hslots hprog) hchecks) hfix hfixed hsessions
      (SamplingDeploy.innerNumbers_deployPlain relD c hnum) hopens)
    L K S lay k δ hδ reg ops Dj x hx hreg Ω P

end Sha512

end NetTiming
