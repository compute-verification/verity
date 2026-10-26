import Proofs.Flock.Claims.Sampling.Tables
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.KS.PlainLeafKS

/-!
# The sampling claims at the deployed record, at SHA-512 at the deployed relation

Every deployed form of the sampling claims takes Flock's KS at the untrimmed masked instance as `hKS`
(`Claims.Flock.KS H (FlockDeploy.deployZk rel)`). At `H := FlockDeploy.sha512` and `rel := Flock.Readable.relation` it
holds with no premise (`FlockDeploy.KS.zk_KS_untrim`), so there each form reads with no KS binder:

* `sampling_deployed_sha512`: `sampling_deployed`;
* `sampling_deployed_models_sha512`: `sampling_deployed_models`;
* `sampling_deployed_sampled_sha512`: `sampling_deployed_sampled`;
* `sampling_deployed_tables_sha512`: `sampling_deployed_tables`.

At the deployed plain record, Flock's plain KS at the trimmed instance holds there with no premise
(`FlockDeploy.KS.plain_KS`):

* `sampling_deployed_plain_sha512`: `sampling_deployed_plain`.

Each is its form at that hash and relation, with `hKS` discharged; none is a guarantee of its own.
-/

namespace SamplingDeploy

open Claims Claims.Sampling Claims.ZKMal
open FlockSoundness.Discharge

local notation "relD" => _root_.Flock.Readable.relation
local notation "shaD" => FlockDeploy.sha512

variable (c : Code)

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record, at SHA-512 at the deployed relation**: `sampling_deployed` with
Flock's KS at the untrimmed masked instance discharged (`FlockDeploy.KS.zk_KS_untrim`). -/
theorem sampling_deployed_sha512
    (hfixed : DrawFixed (deploy relD c)) (hsessions : SessionRuns shaD (deploy relD c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy relD c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens shaD (deploy relD c) iroots) (hfix : InteriorFixed shaD (deploy relD c) iroots)
    (hprog : ProgramScope (deploy relD c)) (hchecks : ChecksScope shaD (deploy relD c)) (hreveals : RevealsLe c)
    (hComplete : Complete shaD (deploy relD c))
    {Nz Ωz : Type} [Fintype Nz] [Nonempty Nz] [Fintype Ωz] [Nonempty Ωz] (tapeOf : Nz × Ωz → Reference.Tape)
    (S : SimLeak) (εzk γzk : Msg → ℝ) (hZK : ZK shaD ((deploy relD c).mask Nz Ωz tapeOf S) εzk γzk) :
    ScopeMatches (deploy relD c) ∧ Binding (deploy relD c) ∧ Universal (deploy relD c) ∧ FaultyBoundary ∧
      KS shaD (deploy relD c) ∧ KSCount shaD (deploy relD c) ∧ Bits shaD (deploy relD c) ∧
      Efficient shaD (deploy relD c) FlockDeploy.zkStepPoly ∧ Complete shaD (deploy relD c) ∧
      ZK shaD ((deploy relD c).mask Nz Ωz tapeOf S) εzk γzk :=
  sampling_deployed relD c shaD FlockDeploy.KS.zk_KS_untrim hfixed hsessions hnum hslots iroots hopens hfix hprog
    hchecks hreveals hComplete tapeOf S εzk γzk hZK

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed plain record, at SHA-512 at the deployed relation**:
`sampling_deployed_plain` with Flock's plain KS at the trimmed instance discharged (`FlockDeploy.KS.plain_KS`). -/
theorem sampling_deployed_plain_sha512
    (hfixed : DrawFixed (deployPlain relD c)) (hsessions : SessionRuns shaD (deployPlain relD c))
    (hnum : SessionsInPlain c) (hslots : DrawSlots (deployPlain relD c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens shaD (deployPlain relD c) iroots) (hfix : InteriorFixed shaD (deployPlain relD c) iroots)
    (hprog : ProgramScope (deployPlain relD c)) (hchecks : ChecksScope shaD (deployPlain relD c))
    (hreveals : RevealsLe c) (hComplete : Complete shaD (deployPlain relD c)) :
    ScopeMatches (deployPlain relD c) ∧ Binding (deployPlain relD c) ∧ Universal (deployPlain relD c) ∧
      FaultyBoundary ∧ KS shaD (deployPlain relD c) ∧ KSCount shaD (deployPlain relD c) ∧
      Bits shaD (deployPlain relD c) ∧ Efficient shaD (deployPlain relD c) FlockDeploy.plainStepPoly ∧
      Complete shaD (deployPlain relD c) :=
  sampling_deployed_plain relD c shaD FlockDeploy.KS.plain_KS hfixed hsessions hnum hslots iroots hopens hfix hprog
    hchecks hreveals hComplete

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record with zero knowledge from the models, at SHA-512 at the deployed
relation**: `sampling_deployed_models` with `hKS` discharged (`FlockDeploy.KS.zk_KS_untrim`). -/
theorem sampling_deployed_models_sha512
    (hfixed : DrawFixed (deploy relD c)) (hsessions : SessionRuns shaD (deploy relD c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy relD c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens shaD (deploy relD c) iroots) (hfix : InteriorFixed shaD (deploy relD c) iroots)
    (hprog : ProgramScope (deploy relD c)) (hchecks : ChecksScope shaD (deploy relD c)) (hreveals : RevealsLe c)
    (hComplete : Complete shaD (deploy relD c)) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [DecidableEq Cc] (My : Salt → B)
    (cs : Salt → Cc) (tapeOf : L.G.Nonce × L.G.Rand → Reference.Tape) (hP : Honest c shaD L.G tapeOf)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink shaD (RAt shaD (deploy relD c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink shaD (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt shaD (deploy relD c) lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ScopeMatches (deploy relD c) ∧ Binding (deploy relD c) ∧ Universal (deploy relD c) ∧ FaultyBoundary ∧
      KS shaD (deploy relD c) ∧ KSCount shaD (deploy relD c) ∧ Bits shaD (deploy relD c) ∧
      Efficient shaD (deploy relD c) FlockDeploy.zkStepPoly ∧ Complete shaD (deploy relD c) ∧
      ZK shaD ((deploy relD c).mask L.G.Nonce L.G.Rand tapeOf (L.simLeakE t (t * 3 ^ t)))
        (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
          8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
        fun _ => 4 * t * 3 ^ t :=
  sampling_deployed_models relD c shaD FlockDeploy.KS.zk_KS_untrim hfixed hsessions hnum hslots iroots hopens hfix
    hprog hchecks hreveals hComplete L t ht hK My cs tapeOf hP Mp Mc nL nB hnL hnB hL hS hN hI hC

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record with zero knowledge of the sampled masked protocol, at SHA-512 at the
deployed relation**: `sampling_deployed_sampled` with `hKS` discharged (`FlockDeploy.KS.zk_KS_untrim`). -/
theorem sampling_deployed_sampled_sha512
    (hfixed : DrawFixed (deploy relD c)) (hsessions : SessionRuns shaD (deploy relD c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy relD c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens shaD (deploy relD c) iroots) (hfix : InteriorFixed shaD (deploy relD c) iroots)
    (hprog : ProgramScope (deploy relD c)) (hchecks : ChecksScope shaD (deploy relD c)) (hreveals : RevealsLe c)
    (hComplete : Complete shaD (deploy relD c)) (S : Sampled c) (hR : HelloRounds c) (t : ℕ) (ht : 0 < t)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] [DecidableEq Cc] (My : Salt → B) (cs : Salt → Cc)
    (tapeOf : S.leaky.G.Nonce × S.leaky.G.Rand → Reference.Tape) (hP : Honest c shaD S.leaky.G tapeOf)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).CoinModel x Salt B Cc)
    (nL nB : Msg → ℕ) (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x)
    (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (S.leaky.proto lx).MsgLink shaD (RAt shaD (deploy relD c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (S.leaky.proto lx).SimLink shaD (Mc lx)) (hN : ∀ lx, (S.leaky.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (S.leaky.proto lx).InnerHolds (RAt shaD (deploy relD c) lx) (Mc lx)) :
    ScopeMatches (deploy relD c) ∧ Binding (deploy relD c) ∧ Universal (deploy relD c) ∧ FaultyBoundary ∧
      KS shaD (deploy relD c) ∧ KSCount shaD (deploy relD c) ∧ Bits shaD (deploy relD c) ∧
      Efficient shaD (deploy relD c) FlockDeploy.zkStepPoly ∧ Complete shaD (deploy relD c) ∧
      ZK shaD ((deploy relD c).mask S.leaky.G.Nonce S.leaky.G.Rand tapeOf (S.leaky.simLeakE t (t * 3 ^ t)))
        (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
          8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
        fun _ => 4 * t * 3 ^ t :=
  sampling_deployed_sampled relD c shaD FlockDeploy.KS.zk_KS_untrim hfixed hsessions hnum hslots iroots hopens hfix
    hprog hchecks hreveals hComplete S hR t ht My cs tapeOf hP Mp Mc nL nB hnL hnB hL hS hN hI

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record with zero knowledge of the sampled masked protocol at the sessions'
tables, at SHA-512 at the deployed relation**: `sampling_deployed_tables` with `hKS` discharged
(`FlockDeploy.KS.zk_KS_untrim`). -/
theorem sampling_deployed_tables_sha512
    (hfixed : DrawFixed (deploy relD c)) (hsessions : SessionRuns shaD (deploy relD c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy relD c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens shaD (deploy relD c) iroots) (hfix : InteriorFixed shaD (deploy relD c) iroots)
    (hprog : ProgramScope (deploy relD c)) (hchecks : ChecksScope shaD (deploy relD c)) (hreveals : RevealsLe c)
    (hComplete : Complete shaD (deploy relD c)) (S : Sampled c) (hR : HelloRounds c) (t : ℕ) (ht : 0 < t)
    (tapeOf : S.leaky.G.Nonce × S.leaky.G.Rand → Reference.Tape) (hP : Honest c shaD S.leaky.G tapeOf)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).TPreModel x)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).TCoinModel x)
    (nL nB nS : Msg → ℕ) (hnL : ∀ lx x ν, (Mp lx x ν).size ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hnS : ∀ lx x ν c, Fintype.card (Mc lx x ν c).SimAt ≤ nS x)
    (hL : ∀ lx, (S.leaky.proto lx).TMsgLink shaD (RAt shaD (deploy relD c) lx) (Mp lx) (Mc lx))
    (hOK : ∀ lx, (S.leaky.proto lx).TablesOK (Mp lx) (Mc lx)) :
    ScopeMatches (deploy relD c) ∧ Binding (deploy relD c) ∧ Universal (deploy relD c) ∧ FaultyBoundary ∧
      KS shaD (deploy relD c) ∧ KSCount shaD (deploy relD c) ∧ Bits shaD (deploy relD c) ∧
      Efficient shaD (deploy relD c) FlockDeploy.zkStepPoly ∧ Complete shaD (deploy relD c) ∧
      ZK shaD ((deploy relD c).mask S.leaky.G.Nonce S.leaky.G.Rand tapeOf
          ((S.leaky.withTabSim Mc nS).simLeakE t (t * 3 ^ t)))
        (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
          8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
          2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  sampling_deployed_tables relD c shaD FlockDeploy.KS.zk_KS_untrim hfixed hsessions hnum hslots iroots hopens hfix
    hprog hchecks hreveals hComplete S hR t ht tapeOf hP Mp Mc nL nB nS hnL hnB hnS hL hOK

end SamplingDeploy
