module

public import Definitions.Accounting.Guarantees.RateAudit
public import Definitions.Accounting.Guarantees.RateAuditPartial

/-!
# The rate audit's honest devices: lemmas

The honest devices of `Definitions.Accounting.Guarantees.RateAudit` and `RateAuditPartial`: for every store, stream and
tag size, a device accepted on every coin, which shows the guarantee's event is not vacuous. Neither is a guarantee, nor
a witness of one: a witness constructs its instance, and each quantifies over every `N`, `b`, `h` and `ρ`. Each is a
`def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same name, and audited
with the proofs: no record pins it.

* `RateHonest`: an honest device that reads every word once is accepted on every coin, with footprint and ingress
  exactly `N·b`.
* `RatePartialHonest`: `RateHonest`'s device as a partial device, accepted on every coin, completing nothing.
-/

@[expose] public section

namespace Accounting.Guarantees

open Accounting Accounting.Rate
open Pous (Bits)

/-- **Completeness of `RateAudit`: an honest device that reads every word once is accepted on every coin.** For every
`N`, `b`, `h` and `ρ` there is a device with no on-chip state and no free side sources, whose link carries exactly the
certified stream (`L = ρ`) and which makes `N` oracle queries (a `Prover N b h ρ 0 ρ N`), that is accepted on every
store, stream, oracle and sample, so with probability 1; its footprint and its ingress are exactly the store, `N·b`
bits, and its compute is `N·c_tag`, with nothing rebuilt. It reads the stream, then reads every word once and queries
the oracle at it under the stream. Against `RateAudit` at `S = 0`, `L = ρ` (so `c_free = 0`), `q = N` and
`F = N·c_tag`, the honest `N·b` sits above the floor `k·b − λ − (N − k)·c_tag·b/c_enc` by the sample's slack
`(N − k)·b`, about `ε_s·N·b`, plus `λ` and the tagging the sample's slack leaves unspent. Proof:
`Accounting.SecurityProofs.RateHonest`. -/
def RateHonest : Prop :=
  ∀ N b h ρ : ℕ, ∃ A : Rate.Prover N b h ρ 0 ρ N,
    ∀ (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (s : ℕ) (I : Fin s → Fin N),
      A.Accepts E R O I ∧ A.footprint E R O = N * b ∧ A.ingress E R O = N * b ∧ A.compute cTag cEnc E R O = N * cTag

/-- **Completeness of `RateAuditPartial`: an honest device that reads every word once is accepted on every coin.** For
every `N`, `b`, `h` and `ρ` there is a partial device with no on-chip state and no free side sources, whose link
carries exactly the certified stream (`L = ρ`) and which makes `N` oracle queries, that is accepted on every store,
stream, oracle and sample; its footprint and its ingress are exactly the store, `N·b` bits, it completes nothing, and
its compute is `N·c_tag` at every price of a completion. It is `RateHonest`'s device. Proof:
`Accounting.SecurityProofs.RatePartialHonest`. -/
def RatePartialHonest : Prop :=
  ∀ N b h ρ : ℕ, ∃ A : Rate.PartialDevice N b h ρ 0 ρ N,
    ∀ (cTag : ℕ) (cost : ℕ → ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (s : ℕ) (I : Fin s → Fin N),
      A.Accepts E R O I ∧ A.footprint E R O = N * b ∧ A.ingress E R O = N * b ∧ A.completed E R O = 0 ∧
        A.compute cTag cost E R O = N * cTag

end Accounting.Guarantees
