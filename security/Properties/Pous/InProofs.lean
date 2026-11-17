module

public import MemoryChallenger.Verdict
public import Properties.Pous
public import PousCheck.SetupInProofs
public import Definitions.Core.Interaction
public import Definitions.Core.Boolean

/-! PoUS's setup and session checked inside proofs, as deployed (`SetupInProofsDeployed`, `RunInProofsDeployed`):
stated and not proved. -/

@[expose] public section

namespace MemoryChallenger.Properties.SessionDeployed

open MemoryChallenger

/-- **PoUS's session checked inside proofs, as deployed: stated and unproved**, with `sessionInProofs` and
`UnitOutcome` as parameters: a fixed extractor recovers a signed session that `verdict` accepts, except
`(c.k + 2) · (εKs + εBind)` and `Claims.KS`'s collision term at `γ`. -/
def RunInProofsDeployed (H : Claims.Msg → Claims.Msg) (UnitOutcome : Type)
    (sessionInProofs : Schemes → List ByteArray → (c : ChallengePhaseCommitment) → Array CoinOpening → String → ByteArray →
      (Fin (c.k + 2) → UnitOutcome) → Bool) (εKs εBind γ : ℝ) : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (c : ChallengePhaseCommitment)
    (openings : Array CoinOpening) (scheme : String) (pk : ByteArray)
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Claims.Verifier Coin)
    (outsOf : List Claims.Msg → Option (Fin (c.k + 2) → UnitOutcome)),
    (∀ (x : Claims.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin),
      (Claims.run H P V x ω r).out = .acc →
        ∃ outs, outsOf ((Claims.run H P V x ω r).msgs.map Prod.fst) = some outs ∧
          sessionInProofs schemes trusted c openings scheme pk outs = true) →
    ∃ E : Claims.Extractor Coin SignedChallengePhase V,
      Claims.KS H V (fun _ s => s.commitment = c ∧ s.scheme = scheme ∧ s.publicKey = pk ∧
          (verdict schemes trusted expected s c openings).accepted = true) E
        (fun _ => (c.k + 2) * (εKs + εBind)) fun _ => γ

end MemoryChallenger.Properties.SessionDeployed

namespace PousCheck.Properties.SetupDeployed

open PousCheck PousCheck.Properties

open Classical in
/-- **The blocks of `C*` that do not decode** to their slice of `W` (its `payloadBytes` bytes at `u · payloadBytes`)
under the salt `salt`, as `BlockDecodes` reads block `u % blocksPerSegment` of segment `u / blocksPerSegment`. -/
noncomputable def undecoded (q : P2.Params) (salt W : ByteArray) (cs : Fin q.blocks → ByteArray) :
    Finset (Fin q.blocks) :=
  Finset.univ.filter fun u => ¬ BlockDecodes q salt (u.val / q.blocksPerSegment) (u.val % q.blocksPerSegment) (cs u)
    (W.extract (u.val * q.payloadBytes) (u.val * q.payloadBytes + q.payloadBytes))

/-- **Named assumption: the registered setup circuit computes the per-block check**: an assignment of `read` carrying
the verifier's `setup/base` and `setup/slot` rows for block `u` whose `o0` is the `setup/ok` row decodes to a
block `BlockDecodes` accepts. This tree names no reading `read`, so it holds of one that registers nothing. -/
def SetupComputesBlockOk (q : P2.Params) (salt : ByteArray)
    (read : Option ((C : Verity.Boolean.Circuit) × (String → List (Fin C.N)) × List (Fin C.N)))
    (supplied : Nat → String → List Bool) (decode : (String → List Bool) → ByteArray × ByteArray) : Prop :=
  ∀ r ∈ read, ∀ (u : Fin q.blocks) (x : Fin r.1.N → Bool),
    (∀ name ∈ ["setup/base", "setup/slot"], ∀ j (h : j < (r.2.1 name).length),
      x ((r.2.1 name).get ⟨j, h⟩) = (supplied u.val name).getD j false) →
    r.2.2.map (r.1.eval x) = supplied u.val "setup/ok" →
    BlockDecodes q salt (u.val / q.blocksPerSegment) (u.val % q.blocksPerSegment)
      (decode fun name => (r.2.1 name).map (r.1.eval x)).1 (decode fun name => (r.2.1 name).map (r.1.eval x)).2

/-- **PoUS's setup checked inside proofs, as deployed: stated and unproved.** Under `SetupComputesBlockOk`, a fixed
extractor recovers a store `C*` of which at most `⌊B/100⌋` blocks fail to decode to their slice of `W`, except
`εPa + εKs + εBind` and `Claims.KS`'s collision term at `γ`. -/
def SetupInProofsDeployed (H : Claims.Msg → Claims.Msg) (εPa εKs εBind γ : ℝ) : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (W : ByteArray)
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Claims.Verifier Coin)
    (claimOf : List Claims.Msg → Option _root_.PousCheck.SetupInProofs.Claim)
    (read : Option ((C : Verity.Boolean.Circuit) × (String → List (Fin C.N)) × List (Fin C.N)))
    (supplied : Nat → String → List Bool) (decode : (String → List Bool) → ByteArray × ByteArray),
    (∀ (x : Claims.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin),
      (Claims.run H P V x ω r).out = .acc →
        ∃ cl pub, claimOf ((Claims.run H P V x ω r).msgs.map Prod.fst) = some cl ∧
          _root_.PousCheck.SetupInProofs.check q c cl = .accept pub ()) →
    SetupComputesBlockOk q c.salt read supplied decode →
    ∃ E : Claims.Extractor Coin (Fin q.blocks → ByteArray) V,
      Claims.KS H V (fun _ cs => (undecoded q c.salt W cs).card ≤ q.blocks / 100) E (fun _ => εPa + εKs + εBind)
        fun _ => γ

end PousCheck.Properties.SetupDeployed
