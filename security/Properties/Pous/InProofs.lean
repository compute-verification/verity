module

public import MemoryAttester.Verdict
public import Properties.Pous
public import PousVerifier.SetupInProofs
public import Definitions.Core.Interaction
public import Definitions.Core.Boolean

/-!
# PoUS's setup and session checked inside proofs, as deployed: stated, unproved

**The session.** The memory challenger's signed record never reaches the auditor. The auditor's session decides by
SESSUNITS' `MemoryChallenger.sessionInProofs` (`MemoryChallenger/InProofs.lean`, memory-accounting/pous-session-inside-proof):
the clear checks on what the device releases (the commitment, the coin openings, the scheme and key), the carried rows'
leaves agreeing across the session's `c.k + 2` proof blocks, and every block's proof accepted. `RunInProofsDeployed`
says that a fixed extractor, rerunning any prover, recovers a signed session that today's `verdict` accepts, except
`(c.k + 2) · (ε_ks + ε_bind)` and the collision term `Claims.KS` carries. `sessionInProofs` and its `UnitOutcome` are not
on this branch yet, so the statement takes them as parameters at the signature its issue gives; the deployed instance is
`RunInProofsDeployed H UnitOutcome sessionInProofs`.

**The setup.** `C*` and `W` never reach the auditor. The auditor decides by SETUPUNIT's `PousCheck.SetupInProofs.check`
(`PousCheck/SetupInProofs.lean`, memory-accounting/pous-setup-as-proof) on the setup's claim: the sampled audit at
`rate:8828/B` of the setup's C, one `PousP2SetupCheck` unit a block of `C*`. `SetupInProofsDeployed` says that a fixed
extractor recovers a store `C*` of which at most `⌊B/100⌋` blocks fail to decode to their slice of the registered `W`,
except `ε_Pa + ε_ks + ε_bind` (`ε_Pa ≤ 0.99^8828`, the chance the sample misses `⌊B/100⌋ + 1` wrong blocks) and the
collision term. It takes the named assumption `SetupComputesBlockOk`, that the registered setup circuit computes the
per-block check: the one value the verifier supplies is the salt (`pous.setup.salt/v1`), each unit builds its base from
it and its block's constants, its slot is a constant of the unit in C, and its `ok` output is held to `1`. It is in
`OpeningComputesCheck`'s form, over a reading of the circuit record this tree does not name yet, so until it does the
premise holds of a reading that registers nothing.

Both are stated and not proved. What closes them is proofs' two forms, through the lemmas of
`Proofs.Properties.Pous.InProofs` and `Proofs.Properties.Pous.SetupInProofs`:
- for the session, `Flock.Guarantees.IntegrityAllProvedCarried` at `I` the session's statements (every unit proved),
  whose conclusion is `Verity.InProofs.Gives` at `K₀ = 1` with each carried leaf (the digest `m`, each round's record)
  read at its one value in every statement that shares it, which is `Carried`;
- for the setup, the sampled period form with carried leaves, `Flock.Guarantees.IntegrityPooledCarried` at
  `q = 8828/B`, through `SetupGives` (the shared sampling theorem's count, `Spot.Sampling.extraction_audit_count`,
  with `W` held to its registration by `RegisteredW`).
-/

@[expose] public section

namespace MemoryAttester.Properties.SessionDeployed

open MemoryAttester

/-- **PoUS's session checked inside proofs, as deployed: stated and unproved**; proofs' `IntegrityAllProvedCarried`
closes it through `sessionGives`. For a session check `sessionInProofs` over one outcome a proof block (SESSUNITS', not on
this branch yet), the accepted schemes, the trusted keys, the expected digests `expected` (the store's registered rows,
as the round units `PousRound_v1` read them), the commitment `c`, the coin openings, the scheme and the key, and a session
verifier `V` that accepts only outcomes (`outsOf`) that `sessionInProofs` accepts: a fixed extractor `E` recovers, by
rerunning any prover, a signed session under `c`, `scheme` and `pk` that `verdict` accepts, except
`(c.k + 2) · (εKs + εBind)` and `Claims.KS`'s collision term at `γ`. -/
def RunInProofsDeployed (H : Hash.Msg → Hash.Msg) (UnitOutcome : Type)
    (sessionInProofs : Schemes → List ByteArray → (c : ChallengePhaseCommitment) → Array CoinOpening → String → ByteArray →
      (Fin (c.k + 2) → UnitOutcome) → Bool) (εKs εBind γ : ℝ) : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (c : ChallengePhaseCommitment)
    (openings : Array CoinOpening) (scheme : String) (pk : ByteArray)
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Claims.Verifier Coin)
    (outsOf : List Hash.Msg → Option (Fin (c.k + 2) → UnitOutcome)),
    (∀ (x : Hash.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin),
      (Claims.run H P V x ω r).out = .acc →
        ∃ outs, outsOf ((Claims.run H P V x ω r).msgs.map Prod.fst) = some outs ∧
          sessionInProofs schemes trusted c openings scheme pk outs = true) →
    ∃ E : Claims.Extractor Coin SignedChallengePhase V,
      Claims.KS H V (fun _ s => s.commitment = c ∧ s.scheme = scheme ∧ s.publicKey = pk ∧
          (verdict schemes trusted expected s c openings).accepted = true) E
        (fun _ => (c.k + 2) * (εKs + εBind)) fun _ => γ

end MemoryAttester.Properties.SessionDeployed

namespace PousVerifier.Properties.SetupDeployed

open PousVerifier PousVerifier.Properties

open Classical in
/-- **The blocks of `C*` that do not decode** to their slice of `W` (its `payloadBytes` bytes at `u · payloadBytes`)
under the salt `salt`, as `BlockDecodes` reads block `u % blocksPerSegment` of segment `u / blocksPerSegment`. -/
noncomputable def undecoded (q : P2.Params) (salt W : ByteArray) (cs : Fin q.blocks → ByteArray) :
    Finset (Fin q.blocks) :=
  Finset.univ.filter fun u => ¬ BlockDecodes q salt (u.val / q.blocksPerSegment) (u.val % q.blocksPerSegment) (cs u)
    (W.extract (u.val * q.payloadBytes) (u.val * q.payloadBytes + q.payloadBytes))

/-- **Named assumption: the registered setup circuit computes the per-block check.** `read` is the verifier's parsed
circuit file of the setup's C, read as a Boolean circuit with unit `u`'s wires by port (`r.2.1 u name`, in its
signature's order) and its `ok` output's wires (`r.2.2 u`), the output the circuit record holds to the supplied constant
`1` (`const.u16/v1`); `none` where no such circuit is registered; this tree does not name that reading yet. The one
value the verifier supplies is the salt (`pous.setup.salt/v1`), which every unit reads at its `salt` port, its bits in
byte order (`le`); unit `u` builds its base `expand_prefix(params, salt) ‖ u64be(u / S) ‖ u64be(u mod S)` inside from the
salt and its block's constants, and its slot `u mod n` is a constant of the unit in C, so no assignment reaches either.
`decode` reads a block and its payload from the unit's port rows. For every block `u` and every assignment whose salt
wires carry `salt`, an `ok` of `1` (least significant bit first) decodes to a block that decodes canonically under `salt`
to its payload, as `BlockDecodes` checks it. -/
def SetupComputesBlockOk (q : P2.Params) (salt : ByteArray)
    (read : Option ((C : Verity.Boolean.Circuit) × (Fin q.blocks → String → List (Fin C.N)) ×
      (Fin q.blocks → List (Fin C.N))))
    (decode : (String → List Bool) → ByteArray × ByteArray) : Prop :=
  ∀ r ∈ read, ∀ (u : Fin q.blocks) (x : Fin r.1.N → Bool),
    (∀ j (h : j < (r.2.1 u "salt").length),
      x ((r.2.1 u "salt").get ⟨j, h⟩) = (salt.get! (j / 8)).toNat.testBit (j % 8)) →
    (r.2.2 u).map (r.1.eval x) = (List.range (r.2.2 u).length).map (· == 0) →
    BlockDecodes q salt (u.val / q.blocksPerSegment) (u.val % q.blocksPerSegment)
      (decode fun name => (r.2.1 u name).map (r.1.eval x)).1 (decode fun name => (r.2.1 u name).map (r.1.eval x)).2

/-- **PoUS's setup checked inside proofs, as deployed: stated and unproved**; proofs' `IntegrityPooledCarried` at
`q = 8828/B` closes it through `setupGives`, its `right` from `SetupComputesBlockOk`. For the parameters `q`,
the commitment `c`, the registered weights `W` (each row padded to its slots, SETUPUNIT's `padded`), a setup verifier `V`
that accepts only a claim (`claimOf`) that `SetupInProofs.check` accepts, and the named assumption
`SetupComputesBlockOk` at the circuit record's reading `read` and the decoding `decode`: a
fixed extractor `E` recovers, by rerunning any prover, a store `C*` of which at most `⌊B/100⌋` blocks fail to decode to
their slice of `W`, except `εPa + εKs + εBind` and `Claims.KS`'s collision term at `γ`. -/
def SetupInProofsDeployed (H : Hash.Msg → Hash.Msg) (εPa εKs εBind γ : ℝ) : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (W : ByteArray)
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Claims.Verifier Coin)
    (claimOf : List Hash.Msg → Option _root_.PousVerifier.SetupInProofs.Claim)
    (read : Option ((C : Verity.Boolean.Circuit) × (Fin q.blocks → String → List (Fin C.N)) ×
      (Fin q.blocks → List (Fin C.N))))
    (decode : (String → List Bool) → ByteArray × ByteArray),
    (∀ (x : Hash.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin),
      (Claims.run H P V x ω r).out = .acc →
        ∃ cl pub, claimOf ((Claims.run H P V x ω r).msgs.map Prod.fst) = some cl ∧
          _root_.PousVerifier.SetupInProofs.check q c cl = .accept pub ()) →
    SetupComputesBlockOk q c.salt read decode →
    ∃ E : Claims.Extractor Coin (Fin q.blocks → ByteArray) V,
      Claims.KS H V (fun _ cs => (undecoded q c.salt W cs).card ≤ q.blocks / 100) E (fun _ => εPa + εKs + εBind)
        fun _ => γ

end PousVerifier.Properties.SetupDeployed
