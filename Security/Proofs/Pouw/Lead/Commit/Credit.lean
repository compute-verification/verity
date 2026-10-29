import Definitions.Pouw.Assumptions.Lead

/-!
# The commitment's hashing credit

Without a collision, an execution that reaches each of the honest commitment's digests hashed each of its messages,
so its ShaWrites credit is at least the honest commitment's distinct messages'.
-/

namespace Pouw.SecurityProofs.Lead.Commit

open Pouw.Lead Pouw.Assumptions.Lead

/-- **Hashing credit**: if the execution at `ω` hashes, for each message of the honest commitment, a
message with its digest (`HonestHashRuled`'s conclusion) and hashed no collision with them (`¬ Break`), its messages'
credit is at least the honest commitment's distinct messages'. Reads `Hash.CollisionIn`, `shaCredit` and `Break`. -/
theorem hashed_credit_ge {L : Pouw.Lead.Layout} (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (isA : Hash.Msg → Bool) (σ : Strategy L) (ω : Coins L)
    (hhash : ∀ y ∈ C.msgs H (word L κ ω) ++ C.aMsgs, ∃ x ∈ (σ.exec ω).hashed, H x = H y)
    (hnb : ¬ Break L H C κ σ ω) :
    (((C.msgs H (word L κ ω) ++ C.aMsgs).dedup).map (shaCredit isA)).sum ≤
      ((σ.exec ω).hashed.map (shaCredit isA)).sum := by
  sorry

end Pouw.SecurityProofs.Lead.Commit
