module

public import Definitions.WeightSecurity.Guarantees

/-! `TheftTight`, a lemma beside `Theft` that no record pins: at the toy instance the factor `|S|` of `Theft`'s bound
is attained. -/

@[expose] public section

namespace WeightSecurity.Guarantees

open WeightSecurity WeightSecurity.Assumptions

/-- Each covert bit can double the theft's probability, so it is only unlikely while the covert capacity stays below
`log₂ (1/p)`. -/
def TheftTight : Prop :=
  LeaksThrough Toy.C Toy.gen Toy.cheatDC Toy.thief 1 Toy.oneBit 0 ∧
    Claims.Pr (fun z : (Bool × Bool) × Unit =>
        Toy.Stolen (Toy.gen z.1) (Toy.thief.model z.2 (Toy.cheatDC (Toy.gen z.1) z.2) 1)) =
      Fintype.card Toy.oneBit.S * 2⁻¹ + 0

end WeightSecurity.Guarantees
