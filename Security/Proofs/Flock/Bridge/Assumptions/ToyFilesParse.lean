import Flock.Test.ToyPubFiles
import SampledProofs.Slots

/-!
# The toy session's files parse

`SampledKSLivePub`'s nontrivial lemma is at a toy `verify --zk` session (`Flock.Test.ToyPubFiles`): one unit of
`toy-copy`, a draw of that unit, and a registration of one unit. The verifier reads those files through
`Lean.Json.parse` and its own readers, which loop with `partial` and `while`, so the kernel can't evaluate them on any
file and no proof exhibits their results. `ToyFilesParse` is the one named premise that they give what
`Flock.Test.ToyPubFiles` writes out, stated with the verifier's own functions on those files.
-/

namespace Flock.Bridge.Assumptions

open Flock Flock.Test.ToyPubFiles

/-- **The toy session's files parse to what `Flock.Test.ToyPubFiles` writes** (interim, until the verifier's readers
are ones the kernel evaluates). Under the tags `verify --zk` sets the toy session up with (`tags`), the circuit file
parses to the descriptor of `circuit` (`Desc.parse`), which is in the hidden-output headlines' scope
(`ProvedScope.scopeOkH`); the public file loads at it to `pub` (`HmOut.loadPublic`); the two ports' registered roots
over the one row differ (`root`); the draw `draw` proves unit 0 alone (`Draw.UnitDraw.ofJson`), and the drawn public
file's instances are unit 0 (`HmOut.drawn`, `Registered.unitIndices`); and the registration's program `progDesc` parses
(`Program.parse`) and its population under `progQuery` is unit 0 alone (`Slots.popOk`).

Assumption record. It assumes a computation's result: `Flock.Test.ToyPub.check`, compiled into `flock-witness toy-pub`
(run by C-Flock's `test_toy_pub.py`), evaluates the circuit, public file, scope, roots and draw conjuncts with these
functions on these files, comparing every field `circuit` and `pub` write out (the rest of each is defined from the
parse itself); `SampledProofs.ToyPop`'s `#guard` evaluates the registration's. A change to any of these functions or
files makes it a new premise, checked by those again. -/
def ToyFilesParse : Prop :=
  Desc.parse tags circuitFile [] = .ok (.single circuit) ∧
    ProvedScope.scopeOkH circuit = true ∧
    HmOut.loadPublic tags circuit publicFile = .ok pub ∧
    (root 0).data ≠ (root 1).data ∧
    (∃ D : Draw.UnitDraw, Draw.UnitDraw.ofJson draw = .ok D ∧ D.proved = #[0]) ∧
    (∃ dp : Public, HmOut.drawn circuit pub draw = .ok dp ∧ Registered.unitIndices dp = .ok #[0]) ∧
    ∃ p : SampledProofs.Program, SampledProofs.Program.parse ⟨progDesc.data.toList.toArray⟩ = .ok p ∧
      SampledProofs.Slots.popOk p progQuery 1 = true

end Flock.Bridge.Assumptions
