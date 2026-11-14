module

public import MemoryChallenger.Verdict
public import Properties.Pous
public import PousCheck.SetupInProofs
public import Definitions.Core.Interaction

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
`rate:8828/B` of one `PousP2SetupCheck_v1` instance a block of `C*`. `SetupInProofsDeployed` says that a fixed extractor
recovers a store `C*` of which at most `⌊B/100⌋` blocks fail to decode to their slice of the registered `W`, except
`ε_Pa + ε_ks + ε_bind` (`ε_Pa ≤ 0.99^8828`, the chance the sample misses `⌊B/100⌋ + 1` wrong blocks) and the collision
term.

Both are stated and not proved. What closes them is proofs' two forms, through the lemmas of
`Proofs.Properties.Pous.InProofs` and `Proofs.Properties.Pous.SetupInProofs`:
- for the session, `Flock.Guarantees.IntegrityAllProvedCarried` at `I` the session's statements (every unit proved),
  whose conclusion is `Verity.InProofs.Gives` at `K₀ = 1` with each carried leaf (the digest `m`, each round's record)
  read at its one value in every statement that shares it, which is `Carried`;
- for the setup, the sampled period form with carried leaves, `Flock.Guarantees.IntegrityPooledCarried` at
  `q = 8828/B`, whose conclusion gives `Gives` at `K₀ = ⌊B/100⌋ + 1` with `W`'s rows carried, which is `CarriedW`.
-/

@[expose] public section

namespace MemoryChallenger.Properties.SessionDeployed

open MemoryChallenger

/-- **PoUS's session checked inside proofs, as deployed: stated and unproved**; proofs' `IntegrityAllProvedCarried`
closes it through `sessionGives`. For a session check `sessionInProofs` over one outcome a proof block (SESSUNITS', not on
this branch yet), the accepted schemes, the trusted keys, the expected digests `expected` (the store's registered rows,
as the round units `PousRound_v1` read them), the commitment `c`, the coin openings, the scheme and the key, and a session
verifier `V` that accepts only outcomes (`outsOf`) that `sessionInProofs` accepts: a fixed extractor `E` recovers, by
rerunning any prover, a signed session under `c`, `scheme` and `pk` that `verdict` accepts, except
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

/-- **PoUS's setup checked inside proofs, as deployed: stated and unproved**; proofs' `IntegrityPooledCarried` at
`q = 8828/B` closes it through `setupGives`. For the parameters `q`, the commitment `c`, the registered weights `W` (each row
padded to its slots, SETUPUNIT's `padded`), and a setup verifier `V` that accepts only a claim (`claimOf`) that
`SetupInProofs.check` accepts: a fixed
extractor `E` recovers, by rerunning any prover, a store `C*` of which at most `⌊B/100⌋` blocks fail to decode to their
slice of `W`, except `εPa + εKs + εBind` and `Claims.KS`'s collision term at `γ`. -/
def SetupInProofsDeployed (H : Claims.Msg → Claims.Msg) (εPa εKs εBind γ : ℝ) : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (W : ByteArray)
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Claims.Verifier Coin)
    (claimOf : List Claims.Msg → Option _root_.PousCheck.SetupInProofs.Claim),
    (∀ (x : Claims.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin),
      (Claims.run H P V x ω r).out = .acc →
        ∃ cl pub, claimOf ((Claims.run H P V x ω r).msgs.map Prod.fst) = some cl ∧
          _root_.PousCheck.SetupInProofs.check q c cl = .accept pub ()) →
    ∃ E : Claims.Extractor Coin (Fin q.blocks → ByteArray) V,
      Claims.KS H V (fun _ cs => (undecoded q c.salt W cs).card ≤ q.blocks / 100) E (fun _ => εPa + εKs + εBind)
        fun _ => γ

end PousCheck.Properties.SetupDeployed
