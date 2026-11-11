module

public import Properties.Pous.EndToEnd

/-!
# PoUS end to end in the ring model: the honest server's lemmas

The two non-vacuity lemmas beside `Properties.Pous.EndToEnd`'s ring-model properties: the honest full-state server
passes P2's audit, and `EndToEndRing`'s event, with probability 1. Neither is a property, nor a witness of one: a witness
constructs its instance, and each quantifies over every `W` (and the session's id). Each is a `def G : Prop`
under the properties' namespace, proved by the `SecurityProofs` theorem of its name in lower camel case, and audited
with the proofs: no record pins it.

* `HonestPassesRing`: the honest full-state server passes P2's ring audit (`seqAcceptsOn`) with probability 1.
* `HonestPassesAccepts`: it passes the session with probability 1, for every 32-byte session id and `W`.
-/

@[expose] public section

open MemoryChallenger MemoryChallenger.Properties
open PousCheck PousCheck.Properties

namespace Pous.EndToEnd

open scoped ENNReal

/-- **Non-vacuity in the ring model, with `ω` uniform: the honest full-state server passes P2's audit.** At the full
state (`FULL = 2^19 · 16,448` bits, more than `S`), a responder that keeps all of `C` and answers each index from it
with no query at all (`RingBounded 0 0 0`) is accepted by `seqAcceptsOn` with probability 1, for every `W`. -/
def HonestPassesRing : Prop :=
  S < FULL ∧ ∃ (A₁ : Preprocessor sch FULL) (A₂ : SeqResponderOn sch FULL 0 0 0),
    ∀ W, pr (seqAcceptsOn sch 105 W A₁ A₂) = 1

/-- **Non-vacuity of `EndToEndRing`'s event, in the ring model, with `ω` uniform**: the honest full-state server, a
`Server FULL 0 0 0`, passes the session with probability 1 for every 32-byte session id and `W`: the code always
builds the coin tree, and the sample never fails (`ε_rs = 0`, `DrawExact`). -/
def HonestPassesAccepts : Prop :=
  ∃ A : Server FULL 0 0 0, ∀ session : ByteArray, session.size = 32 → ∀ W : Bits sch.n,
    pr (Passes session W A) = 1

end Pous.EndToEnd
