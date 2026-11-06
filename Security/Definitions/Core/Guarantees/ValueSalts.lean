module

public import Definitions.Core.ValueSalts
public import Definitions.Core.Assumptions.ZkServer

/-!
# What a registration leaks: at most `log₂ m` bits, `m` the sessions its epoch opened

`RegisteredLeakage`: a serving stack that commits its values in `m` sessions of the ZK server's value service and
registers the one whose public commitments it likes chooses among `m` records, so an event on the record the loop admits
holds at most `m` times as often as on one record of uniform salts and tags: at most `log₂ m` bits leak through the
choice, a term of the guarantee, as PoUW's noise seeds' grinding is. The record makes `m` public
(`Verity.ValueSalts.RegistrationRecord`), so the term is computable from what the verifier sees. Beside it stand how far
a genuine record is from an ideal one (`prfAdvantage`) and a forged record tag (`forgeAdvantage`), each an explicit term,
under the key-separation premise (`Verity.Assumptions.ZkServerOwnUser`).

`RegisteredStreamLeakage`: over a stream of `r` registrations the loop admits each once, oldest first, and a stack that
registers more than once also chooses which ones its statements read: each one passed over is a failure of the stream, so
with at most `b` passed over it chooses among `streamChoices r m b`, `∑_{|P| ≤ b} ∏_{i ∉ P} m i` choices. Every epoch
opened the session its registration registered, so `m i ≥ 1` and that is at most `∑_{i ≤ b} C(r, i) · ∏_i m i`: at most
`log₂ ∑_{i ≤ b} C(r, i) + ∑_i log₂ m i` bits.

**`m` is the service's, not the stack's.** Both statements hold for every `m`, but a stack that chose `m` from its values
would leak through `m` itself as well, which neither counts. The ZK server's value service of record
(`core-firewall values`, `Flock.Firewall.Values.answer`) fixes `m` at 1: it refuses a commit while its epoch's session is
not yet registered. So the stack chooses no `m` and no session, every record says `m = 1`, `RegisteredLeakage` at
`m = 1` is the whole of what a registration leaks (no `log₂ m` term), and over a stream, at every `m i = 1`, the stack's
only choice is which registrations it passes over: at most `log₂ ∑_{i ≤ b} C(r, i)` bits.
-/

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
