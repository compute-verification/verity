import MemoryChallenger.Judge

/-!
# The memory challenger's verdict's property, stated against the Lean code

What `MemoryChallenger.verdict` (package `core_memory_challenger`, `memory_challenger.verdict` in the Python) accepts,
stated over its inputs: the accepted schemes, the trusted keys, the expected digests, the signed session, the commitment
published before it and the coin openings. No statement takes an assumption: the verdict is a function of those inputs.

* `VerdictAccepts`: an accepted verdict's scheme is accepted, its key trusted and `lmsVerify` passes on the session's
  message; the signed header is the commitment; the session ended complete after the commitment's `k` rounds; and each
  round `j`'s index is its committed coin (an opening of round `j` to that index verifies under the coin tree), its
  digest is the expected digest of that block of `C`, and it is on time, within `Δ + RTT`.
* `JudgeAccepts`: the same for `Judge.judge` on a record as its caller holds it, under the schemes the record names.

Neither says the signer signed: the conclusion is that the signature verifies, not that it was made by the key's holder
(no LMS-break form yet). Both are proved (`Properties.Proofs.Pous.MemoryChallenger`).
-/

namespace MemoryChallenger.Properties

/-- What an accepted verdict establishes about its inputs. -/
def Accepted (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedSession)
    (c : SessionCommitment) (openings : Array CoinOpening) : Prop :=
  (∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true) ∧
    s.commitment = c ∧ s.ended = COMPLETE ∧ s.rounds.size = c.k ∧
    ∀ j (hj : j < s.rounds.size),
      s.rounds[j].j = j ∧ (∃ o ∈ openings, o.j = j ∧ o.index = s.rounds[j].index ∧ verifyCoin c o = true) ∧
        s.rounds[j].index < c.blocks ∧ expected s.rounds[j].index = some s.rounds[j].digest ∧
        0 ≤ s.rounds[j].elapsedNs ∧ s.rounds[j].elapsedNs ≤ c.limitNs

def VerdictAccepts : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedSession)
    (c : SessionCommitment) (openings : Array CoinOpening),
    (verdict schemes trusted expected s c openings).accepted = true → Accepted schemes trusted expected s c openings

def JudgeAccepts : Prop :=
  ∀ (r : Judge.Record), Verdict.accepted <$> Judge.judge r = .ok true →
    ∃ schemes, Judge.schemesOf r.schemes = .ok schemes ∧
      Accepted schemes r.trusted.toList r.expected.find r.signed r.commitment r.openings

end MemoryChallenger.Properties
