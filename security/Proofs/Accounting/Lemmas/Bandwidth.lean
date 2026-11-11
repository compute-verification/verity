module

public import Definitions.Accounting.Guarantees.Bandwidth

/-!
# The bandwidth audit's honest device: a lemma

`HonestPasses`, the honest device of `Definitions.Accounting.Guarantees.Bandwidth`'s `AuditRO`: for every store and tag
size, a device accepted on every coin, which shows `AuditRO`'s event is not vacuous. It is no guarantee, nor a witness
of one: a witness constructs its instance, and it quantifies over every `N`, `b` and `h`. It is a `def G : Prop` under
the guarantees' namespace, proved by the `SecurityProofs` theorem of the same name, and audited with the proofs: no
record pins it.
-/

@[expose] public section

namespace Bandwidth.Guarantees

/-- **Completeness of `AuditRO`: an honest device is accepted, and its ingress is exactly the store.** For every `N`,
`b` and `h` there is a device with no on-chip state, no side sources and no regeneration that makes `N` oracle queries
(a `Device N b h 0 0 0 N`) whose tags are accepted on every store, oracle and sample, and whose ingress is exactly `N·b`
bits. The device reads every block once and queries the oracle at it. Against `AuditRO` at `S = c = G = 0` and `q = N`,
the honest `N·b` sits above the floor `k·b − λ` by the sample's slack `(N − k)·b`, about `ε_s·N·b`, plus `λ`. Proof:
`Bandwidth.SecurityProofs.HonestPasses`. -/
def HonestPasses : Prop :=
  ∀ N b h : ℕ, ∃ A : Device N b h 0 0 0 N,
    ∀ (E : Store N b) (O : Oracle N b h) (s : ℕ) (I : Fin s → Fin N), Accepts A E O I ∧ A.ingress E O = N * b

end Bandwidth.Guarantees
