import Definitions.Accounting.RateAuditSketchDeployed

/-!
# A value check bounds what a carrier stacks by the contents it accepts (trusted)

A carrier stacks when memory the audit credits holds other blocks' encodings, read back by a decoder at audit time
(`Guarantees.CarrierBits`). An audit that also checks the carrier's contents, accepting only those in a set `Acc` (the
honest image of the model on some inputs, or the values a statistical test passes), bounds the sectors it can carry by
`log₂|Acc|`, whatever the circuit that fills it costs:

* `CheckedCarrierBits`: for any contents chosen as a function of the store (every prover's choice, any circuit, inputs
  and encodes) that lie in `Acc` at every store, and any decoder that reads the store sectors `carried` back from
  them, whole and as they are, at every store, `2^(256·|carried|) ≤ |Acc|`.

With `CarrierBits` a checked carrier holds at most `min(I + X, log₂|Acc|)` bits about the store. A check that pins the
contents to the honest model's values on the run inputs a carrier's life admits has `log₂|Acc|` at most those inputs'
bits, so it removes the encode term; a statistical check accepts nearly every content, so it removes nearly nothing.
-/

namespace Accounting.Guarantees

open Accounting.RateSketch

/-- **A checked carrier holds at most `log₂|Acc|` bits about the store.** For contents `v E` of type `α` chosen as any
function of the store `E` (`N` tiles of `m` sectors), each in the set `Acc` the audit's value check accepts, and a
decoder `dec` that reads the sectors `carried` back from them at every store:

  `2^(256·|carried|) ≤ |Acc|`. -/
def CheckedCarrierBits : Prop :=
  ∀ (N m : ℕ) {α : Type} (v : SketchStore N m → α) (carried : Finset (Fin N × Fin m)) (dec : α → carried → Sector)
    (Acc : Finset α),
    (∀ E, v E ∈ Acc) → (∀ E, dec (v E) = fun p => E p.1) → 2 ^ (carried.card * 256) ≤ Acc.card

end Accounting.Guarantees
