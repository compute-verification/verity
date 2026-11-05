module

public import Definitions.Core.Assumptions.ZkServer
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# What a registration leaks: at most `log₂ m` bits, `m` the sessions its epoch opened

`RegisteredLeakage`: a serving stack that commits its values in `m` sessions of the ZK server's value service and
registers the one whose public commitments it likes chooses among `m` records, so an event on the record the loop admits
holds at most `m` times as often as on one record of uniform salts and tags: at most `log₂ m` bits leak through the
choice, a term of the guarantee, as PoUW's noise seeds' grinding is. The record makes `m` public
(`Verity.ValueSalts.RegistrationRecord`), so the term is computable from what the verifier sees. Beside it stand how far
a genuine record is from an ideal one (`prfAdvantage`) and a forged record tag (`forgeAdvantage`), each an explicit term,
under the key-separation premise (`Verity.Assumptions.ZkServerOwnUser`).
-/

@[expose] public section

namespace Verity.Guarantees

open Verity Verity.ValueSalts Verity.Assumptions

/-- **A registration leaks at most `log₂ m` bits through the stack's choice of its session.** For every value service
`sv` with finite keys, sessions, salts, tags and record tags, number `m` of sessions the stack commits before it
registers, values `v`, stack `st`, event `A` on a record and key `k₀`: if the stack reads no key (`ZkServerOwnUser`),
the probability that the loop admits the record it hands and the record meets `A` is at most `m` times the probability
of `A` on an ideal record plus the PRF gap, plus the probability that the stack, run at the key `k₀`, forges a record
tag. -/
def RegisteredLeakage : Prop :=
  ∀ {K S Y V O T R : Type} [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R]
    (sv : Service K S Y V O T R) (m : ℕ) (v : V) (st : Stack K V S O T R m)
    (A : RegistrationRecord S O T R → Prop) (k₀ : K),
    ZkServerOwnUser st →
      sv.real m v st A ≤ m * (sv.ideal m v A + sv.prfAdvantage m v A) + sv.forgeAdvantage m v (st k₀ v)

end Verity.Guarantees
