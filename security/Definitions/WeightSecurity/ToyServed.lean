module

public import Definitions.WeightSecurity.Served

/-! The toy instance `TheftServed`'s witnesses are stated at: two uniform 4-bit weight values, of which honest use reveals nothing. -/

@[expose] public section

namespace WeightSecurity.ToyServed

/-- The weights: two 4-bit values. -/
abbrev W := Fin 16 × Fin 16

/-- The served function: every query is answered with `0`. -/
def C (_ : W) (_ : Unit × Unit) (_ : Unit) : Fin 16 := 0

/-- The weights are the coins. -/
def gen : W → W := id

/-- Stealing is outputting the weights. -/
def Stolen (θ θ' : W) : Prop := θ' = θ

/-- Every honest-use party, on one coin: the class of all parties. -/
def anyone : Set (Outsider Unit (Unit × Unit) (Unit → Fin 16) W) := Set.univ

/-- The honest deployment's run: two windows, each accepted on an empty sample, with no incorrect proof block, releasing
the honest answer. -/
def honestRun (_ : W) (_ : Unit) : Verity.Game (List (Served Unit (Fin 16) Unit Unit)) :=
  .ret [⟨True, False, 0, fun _ => 0, (), ()⟩, ⟨True, False, 0, fun _ => 0, (), ()⟩]

/-- A user of the honest deployment: it asks twice and outputs `(0, 0)`. -/
def user : Outsider Unit (Unit × Unit) (Option ((Unit → Fin 16) × Unit)) W where
  query _ _ := ((), ())
  out _ _ := (0, 0)

/-- The leaking deployment's run at the verifier that accepts everything: window `i` writes the weights' `i`-th value
into its slot, with one incorrect proof block, and is accepted on an empty sample. -/
def leakRun (θ : W) (_ : Unit) : Verity.Game (List (Served Unit (Fin 16) Unit Unit)) :=
  .ret [⟨True, False, 1, fun _ => θ.1, (), ()⟩, ⟨True, False, 1, fun _ => θ.2, (), ()⟩]

/-- The thief: it asks twice and reads the weights off the two releases (its view is newest first). -/
def thief : Outsider Unit (Unit × Unit) (Option ((Unit → Fin 16) × Unit)) W where
  query _ _ := ((), ())
  out _ h := match h with
    | [(_, some (b, _)), (_, some (a, _))] => (a (), b ())
    | _ => (0, 0)

end WeightSecurity.ToyServed
