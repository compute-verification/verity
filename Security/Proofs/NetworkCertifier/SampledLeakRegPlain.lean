import Proofs.NetworkCertifier.SampledLeakReg
import Proofs.Flock.Claims.Sampling.DeployedSha512

/-!
# The leak bound at the registration's values, at the deployed plain record

`sampledLeak_registered` is stated at any record. At the deployed sampling record over any Flock record `F`
(`SamplingDeploy.deployOf F c`) its trace commitment is `F`-free (`SamplingDeploy.deployOf_commit`), hm96 leaves under the
verifier's own registered path check, and binds at every `H` with no hypothesis (`binds_deployOf`, from
`Binding_of_regPath`). So the masked and plain records share one commitment (`deployPlain_commit`), and at the plain
record (`SamplingDeploy.deployPlain`) the bound takes only the record's `KS` (`sampledLeak_registered_plain`), which
`SamplingDeploy.sampling_deployed_plain_sha512` gives at SHA-512 at the deployed relation
(`sampledLeak_registered_plain_sha512`). `SampledAuditAccepted` is discharged by its proof.
-/

namespace NetTiming

open Claims Claims.Sampling

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
`SamplingDeploy.deployPlain rel c`, its binding `binds_deployOf` and `SampledAuditAccepted` its proof, given the record's
`KS`. -/
theorem sampledLeak_registered_plain (H : Msg → Msg) (hks : KS H Dp) (L K S : ℕ) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin (Dp).flock.len (Dp).flock.enc (Dp).flock.θ decode) (reg : Stmt → Finset ℕ)
    (ops : Stmt → List (Opening Bool)) (Dj : ℕ) (x : Msg) (hx : ScopeAcc Dp lay k x)
    (hreg : ∀ st, (Dp).parse x = some st → RegOpens H (Dp).commit st (reg st) (ops st)) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    PrBad H (Dp).V
        (RLeak H (Dp).commit (Dp).parse (Dp).partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj) (E Dp decode) x P ≤
      κ Dp δ x + Real.sqrt (γ Dp x * PrB H (Dp).V P x) + (Bext H (Dp).V (E Dp decode) Ω P x).Pr H +
        (Breg H (Dp).V (E Dp decode) (Dp).commit (Dp).parse ops Ω P x).Pr H :=
  sampledLeak_registered SecurityProofs.SampledAuditAccepted H Dp hks (binds_deployOf _ c H) L K S lay k δ decode hδ
    hdec reg ops Dj x hx hreg Ω P

end Plain

section Sha512

local notation "relD" => _root_.Flock.Readable.relation
local notation "shaD" => FlockDeploy.sha512

variable (c : SamplingDeploy.Code)

local notation "Dp" => SamplingDeploy.deployPlain relD c

set_option synthInstance.maxSize 1024 in
/-- **The same at SHA-512 at the deployed relation**: the plain record's `KS` from
`SamplingDeploy.sampling_deployed_plain_sha512`, on its hypotheses. -/
theorem sampledLeak_registered_plain_sha512
    (hfixed : DrawFixed Dp) (hsessions : SessionRuns shaD Dp) (hnum : SamplingDeploy.SessionsInPlain c)
    (hslots : DrawSlots Dp) (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens shaD Dp iroots)
    (hfix : InteriorFixed shaD Dp iroots) (hprog : ProgramScope Dp) (hchecks : ChecksScope shaD Dp)
    (hreveals : SamplingDeploy.RevealsLe c) (hComplete : Complete shaD Dp) (L K S : ℕ) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin (Dp).flock.len (Dp).flock.enc (Dp).flock.θ decode) (reg : Stmt → Finset ℕ)
    (ops : Stmt → List (Opening Bool)) (Dj : ℕ) (x : Msg) (hx : ScopeAcc Dp lay k x)
    (hreg : ∀ st, (Dp).parse x = some st → RegOpens shaD (Dp).commit st (reg st) (ops st)) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    PrBad shaD (Dp).V
        (RLeak shaD (Dp).commit (Dp).parse (Dp).partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj) (E Dp decode) x P ≤
      κ Dp δ x + Real.sqrt (γ Dp x * PrB shaD (Dp).V P x) + (Bext shaD (Dp).V (E Dp decode) Ω P x).Pr shaD +
        (Breg shaD (Dp).V (E Dp decode) (Dp).commit (Dp).parse ops Ω P x).Pr shaD :=
  sampledLeak_registered_plain relD c shaD
    (SamplingDeploy.sampling_deployed_plain_sha512 c hfixed hsessions hnum hslots iroots hopens hfix hprog hchecks
      hreveals hComplete).2.2.2.2.1 L K S lay k δ decode hδ hdec reg ops Dj x hx hreg Ω P

end Sha512

end NetTiming
