module

public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Interaction

/-!
# Model weight security: the setting

The weights `θ` are the values of the declaration's registered inputs, and the circuit serves a function
`C : Θ → X → Y` (`served`): a query `x` is the values of every other input gate (a prompt, a sampling seed, anything
that isn't the weights) and the answer `C θ x` the output gates' values. The statements take any `C`, so they hold at
`served d` for every declaration `d`. The weights come from coins nobody outside the datacenter sees,
`θ = gen ω` with `ω` uniform (training's randomness), and every other coin of the world, the outsiders' and the
datacenter's, is one uniform `r` independent of `ω`. Probability is Core's (`Claims.Pr`).

* An `Outsider` is everyone outside the datacenter at once (users, an accomplice, anyone who pools what they saw): it
  picks each query from its coins and what it has seen so far (adaptively), and at the end outputs a model.
  `Outsider.view` is what it has seen after `n` rounds against a responder, `Outsider.model` what it outputs.
* `honest C θ` is honest use: each query is answered with `C θ x`.
* A `Channel` with covert strings `S` is a way for the datacenter to deviate from honest use that a covert string
  `s ∈ S` describes, `log₂ |S|` bits: with `s`, after the outsider's view `h` and its query `x`, the datacenter evaluates
  the served function at the input `ask s r h x` (it may change the query, set the inputs the user doesn't, or ignore
  it), and the outsider receives `tell s r h x y`, `y` the honest answer there (it may change the answer, add timing,
  refuse, or append chunks of `s`). Neither reads `θ`: everything about the weights beyond honest answers is in `s`,
  which the datacenter may choose knowing `θ` and everything else.
* `Outsider.withString O ch s` is `O` with the covert string `s` written in: an honest-use party that asks the served
  function what the datacenter would evaluate and feeds `O` what the channel would tell it (`Outsider.rebuild`).
-/

@[expose] public section

namespace WeightSecurity

/-- **The function the declaration `d` serves**: the weights are the values of its registered inputs `d.θ`, a query
the values of its other input gates, and the answer the values of its output gates on the honest trace. -/
def served (d : Verity.RegisteredCircuit) (θ : (g : {g // g ∈ d.θ}) → BitVec (d.D.w g.1))
    (x : (g : {g // d.D.isInput g = true ∧ g ∉ d.θ}) → BitVec (d.D.w g.1)) :
    (o : {o // o ∈ d.D.outputs}) → BitVec (d.D.w o.1) :=
  fun o => d.D.eval (fun g => if h : g ∈ d.θ then θ ⟨g, h⟩ else
    if h' : d.D.isInput g = true then x ⟨g, h', h⟩ else 0) o.1

/-- **An outsider**, everyone outside the datacenter at once: from its coins `r` and what it has seen (the queries and
the replies, newest first) its next query, and from its coins and everything it saw its output, a model `Θ'`. -/
structure Outsider (R X M Θ' : Type) where
  query : R → List (X × M) → X
  out : R → List (X × M) → Θ'

variable {R X Y M Θ Θ' : Type}

/-- **What the outsider has seen** after `n` rounds against the responder `resp`, which replies to the query `x` after
the view `h` with `resp h x`: the queries and replies, newest first. -/
def Outsider.view (O : Outsider R X M Θ') (r : R) (resp : List (X × M) → X → M) : ℕ → List (X × M)
  | 0 => []
  | n + 1 => (O.query r (O.view r resp n), resp (O.view r resp n) (O.query r (O.view r resp n))) :: O.view r resp n

/-- **The model the outsider outputs** after `q` rounds against `resp`. -/
def Outsider.model (O : Outsider R X M Θ') (r : R) (resp : List (X × M) → X → M) (q : ℕ) : Θ' :=
  O.out r (O.view r resp q)

/-- **Honest use**: each query `x` is answered with the served function's value `C θ x`. -/
def honest (C : Θ → X → Y) (θ : Θ) : List (X × Y) → X → Y := fun _ x => C θ x

/-- **A covert channel** from the datacenter to the outsider, with covert strings `S`, `log₂ |S|` bits. With the covert
string `s`, at the coins `r`, after the outsider's view `h` and its query `x`, the datacenter evaluates the served
function at `ask s r h x` and the outsider receives `tell s r h x y`, `y` the honest answer there. -/
structure Channel (R X Y M : Type) where
  S : Type
  [fin : Fintype S]
  ask : S → R → List (X × M) → X → X
  tell : S → R → List (X × M) → X → Y → M

attribute [instance] Channel.fin

/-- **The datacenter that deviates through the channel** with the covert string `s`, at the weights `θ` and coins `r`. -/
def Channel.resp (ch : Channel R X Y M) (C : Θ → X → Y) (s : ch.S) (θ : Θ) (r : R) : List (X × M) → X → M :=
  fun h x => ch.tell s r h x (C θ (ch.ask s r h x))

/-- **The outsider's view rebuilt from an honest-use view** (newest first) with the covert string `s`: each round's
query is the outsider's at the rebuilt view so far, and its reply what the channel tells it from the honest answer. -/
def Outsider.rebuild (O : Outsider R X M Θ') (ch : Channel R X Y M) (s : ch.S) (r : R) : List (X × Y) → List (X × M)
  | [] => []
  | (_, y) :: h =>
    (O.query r (O.rebuild ch s r h), ch.tell s r (O.rebuild ch s r h) (O.query r (O.rebuild ch s r h)) y) ::
      O.rebuild ch s r h

/-- **The outsider with the covert string `s` written in**: an honest-use party that, at each round, asks the served
function what the datacenter would evaluate for the outsider's query, and at the end outputs the outsider's model on
the rebuilt view. It makes as many queries as the outsider, and needs no weights. -/
def Outsider.withString (O : Outsider R X M Θ') (ch : Channel R X Y M) (s : ch.S) : Outsider R X Y Θ' where
  query r h := ch.ask s r (O.rebuild ch s r h) (O.query r (O.rebuild ch s r h))
  out r h := O.out r (O.rebuild ch s r h)

end WeightSecurity
