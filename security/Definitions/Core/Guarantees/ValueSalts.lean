module

public import Definitions.Core.ValueSalts
public import Definitions.Core.Assumptions.ZkServer

/-! What a registration leaks, at most `log₂ m` bits for `m` sessions its epoch opened; neither statement counts a leak
through a stack's choice of `m`. -/

@[expose] public section

namespace Verity.Guarantees

open Verity Verity.ValueSalts Verity.Assumptions

/-- **A registration leaks at most `log₂ m` bits through the stack's choice of its session.** For every value service
`sv` with finite keys, sessions, salts, tags and record tags, registration number `n`, number `m` of sessions the stack
commits before it registers, values `v`, stack `st`, event `A` on a record and key `k₀`: if the stack reads no key
(`ZkServerOwnUser`), the probability that the loop admits the record it hands and the record meets `A` is at most `m`
times the probability of `A` on an ideal record plus the PRF gap, plus the probability that the stack, run at the key
`k₀`, forges a record tag. The value service of record fixes `m` at 1, so the stack does not choose it and the factor
is 1. -/
def RegisteredLeakage : Prop :=
  ∀ {K S Y V O T R : Type} [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R]
    (sv : Service K S Y V O T R) (n m : ℕ) (v : V) (st : Stack K V S O T R m)
    (A : RegistrationRecord S O T R → Prop) (k₀ : K),
    ZkServerOwnUser st →
      sv.real n m v st A ≤ m * (sv.ideal n m v A + sv.prfAdvantage n m v A) + sv.forgeAdvantage n m v (st k₀ v)

/-- **A stream's registrations leak at most the stack's choices among them.** For every value service `sv` with finite
keys, sessions, salts, tags and record tags, stream of `r` registrations, registration `i` of an epoch of `m i` sessions,
number `b` of registrations the loop may pass over (at most the stream's failure budget), values `v i` of each, stack
`st`, event `A` on what the loop admits of the stream and key `k₀`: if the stack reads no key (`ZkServerOwnUser`), the
probability that the loop admits the stream the stack hands it and the stream meets `A` is at most the sum, over the
stack's choices `d` (`streamChoices r m b`: a session of each registration's epoch, or none, at most `b` none), of the
probability of `A` on an ideal stream passing over `d`'s none plus the PRF gap at `d`, plus the probability that the
stack, run at the key `k₀`, forges a record tag at some registration. The value service of record fixes every `m i` at
1, so the stack chooses no `m`, adaptively or otherwise, and `d` ranges over its choices of which to pass over alone. -/
def RegisteredStreamLeakage : Prop :=
  ∀ {K S Y V O T R : Type} [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R]
    (sv : Service K S Y V O T R) (r : ℕ) (m : Fin r → ℕ) (b : ℕ) (v : Fin r → V) (st : StreamStack K V S O T R r m)
    (A : StreamRecord r S O T R → Prop) (k₀ : K),
    ZkServerOwnUser st →
      sv.streamReal b v st A ≤
        (∑ d ∈ streamChoices r m b, (sv.streamIdeal m v (passed d) A + sv.streamPrfAdvantage v d A)) +
          sv.streamForgeAdvantage v (st k₀ v)

end Verity.Guarantees
