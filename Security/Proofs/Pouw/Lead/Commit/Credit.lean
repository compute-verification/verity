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
  have hsub : (C.msgs H (word L κ ω) ++ C.aMsgs).dedup ⊆ (σ.exec ω).hashed := fun y hy => by
    rw [List.mem_dedup] at hy
    obtain ⟨x, hx, hxy⟩ := hhash y hy
    by_cases hne : x = y
    · exact hne ▸ hx
    · exact absurd ⟨x, List.mem_append_left _ hx, y, List.mem_append_right _ hy, hne,
        congrArg Verity.Digest.bytes hxy⟩ hnb
  obtain ⟨l, hp, hl⟩ := (List.nodup_dedup _).subperm hsub
  rw [← (hp.map (shaCredit isA)).sum_eq]
  exact (hl.map (shaCredit isA)).sum_le_sum fun a ha => by
    obtain ⟨b, -, rfl⟩ := List.mem_map.1 ha
    unfold shaCredit shaARegs shaWordsRegs
    split_ifs <;> positivity

end Pouw.SecurityProofs.Lead.Commit
