module

public import Definitions.WeightSecurity.Guarantees

/-!
# Model weight security's tightness: a lemma

`TheftTight`, beside `Definitions.WeightSecurity.Guarantees`' `Theft` and its witnesses: at the toy instance the factor
`|S|` of `Theft`'s bound is attained. It is no guarantee and neither of `Theft`'s witnesses (`TheftHonest`,
`TheftNontrivial`). It is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the
same name, and audited with the proofs: no record pins it.
-/

@[expose] public section

namespace WeightSecurity.Guarantees

open WeightSecurity WeightSecurity.Assumptions

/-- **`Theft`'s factor `|S|` is attained.** The cheating toy datacenter's view goes through the one-bit channel (two
covert strings) with `ε = 0`, and the thief steals the weights with probability `1`, the bound `2 · 1/2 + 0`: each
covert bit can double the theft's probability, so it is only unlikely while the covert capacity stays below
`log₂ (1/p)`. -/
def TheftTight : Prop :=
  LeaksThrough Toy.C Toy.gen Toy.cheatDC Toy.thief 1 Toy.oneBit 0 ∧
    Claims.Pr (fun z : (Bool × Bool) × Unit =>
        Toy.Stolen (Toy.gen z.1) (Toy.thief.model z.2 (Toy.cheatDC (Toy.gen z.1) z.2) 1)) =
      Fintype.card Toy.oneBit.S * 2⁻¹ + 0

end WeightSecurity.Guarantees
