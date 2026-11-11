module

public import Definitions.WeightSecurity

/-!
# Model weight security: faults an honest cut simulates

A fault pattern `φ : F` names some proof blocks and the values they write instead, `size φ` counts the blocks, and the
faulted function `Cf θ (x, φ)` is the served function's answer at the weights `θ` and the query `x` with the pattern's
blocks writing its values. An **honest cut** for a class of patterns is a set of values through which every path from
the faulted blocks to the answer passes and which honest use reproduces: given the cut's values as a hint, a public
simulator (`Cut`) computes the faulted answer from one honest answer at a query it chooses. Then a datacenter that
faults only within the class leaks through a covert channel of `Theft`'s kind (`Cut.channel`), whose covert strings are
the hints, in place of the patterns: the faults cost their hints' bits, with no premise beyond `HonestUseHard`.

* `Cut X Y`: hints with sizes, the honest query `ask x h` and the answer `sim x h y` made of the honest answer `y`;
* `Cut.Cuts κ C Cf size inClass`: every faulted answer at a pattern of the class is `κ`'s at a hint no larger than the
  pattern;
* `Cut.channel κ R q M`: the covert channel whose covert string is a hint for each of `q` rounds, the hints' sizes
  summing to at most `M`;
* `FaultsWithin real Cf size inClass O q M θ r`: the datacenter `real` answers each of `O`'s first `q` queries with
  `Cf` at a pattern of the class, the sizes summing to at most `M`;
* `PatternsIn inClass q`: the parties that query the faulted function at patterns of the class, and
  `Outsider.dropFaults`, `faultingAs`: such a party seen as an outsider against a datacenter that faults where it
  would;
* `Cut.Interior κ Cf size φ₀`: the patterns `κ` doesn't cut, and `Cut.Interiors`, those as a pattern type: the only
  ones `TheftFaulted` takes `FaultedUseHard` at, its pattern count among them. Its fault channel (`faultChannel`)
  carries a hint for a window whose pattern `κ` cuts, so those faults count in its strings by their hints and not in
  the pattern count. `Cut.none` cuts nothing, and `Cut.map` reads a cut through an encoding of the answers (a token
  list as the release's `O → V`).

The cuts are stated for autoregressive decoding (`decode`): position `i`'s token is a function of the tokens before it,
which holds when the release is tokens only, nothing decodes ahead, and no state is shared across requests. Two
simulators cover the classes with a cut:

* `forcedCut`: a hint forces tokens at some positions, and the simulator asks the served function's forced-token query
  (`ForcedServing`). It cuts position-local blocks (the token gate: a token and its position a faulted block) and every
  block of a one-token request (its one token);
* `releaseCut`: a hint is a faulted request's whole release. It cuts every block (the reached tokens), which costs no
  more than the fault itself for a block whose output is at least the release's bits.
-/

@[expose] public section

namespace WeightSecurity

variable {R X Y Y' Θ Θ' F V : Type}

/-- **A cut simulator** with hints `H`, a hint `h` of size `size h`: from the query `x` and a hint `h`, the honest query
`ask x h` it asks the served function, and the answer `sim x h y` it makes of the honest answer `y` there. It reads no
weights: everything about the weights beyond the honest answer is in the hint. -/
structure Cut (X Y : Type) where
  H : Type
  [fin : Fintype H]
  size : H → ℕ
  ask : X → H → X
  sim : X → H → Y → Y

attribute [instance] Cut.fin

/-- **`κ` cuts the class `inClass` of fault patterns**: at all weights `θ` and every query `x`, the faulted function's
answer at a pattern of the class is the simulator's at some hint of size at most the pattern's. -/
def Cut.Cuts (κ : Cut X Y) (C : Θ → X → Y) (Cf : Θ → X × F → Y) (size : F → ℕ) (inClass : F → Prop) : Prop :=
  ∀ θ x φ, inClass φ → ∃ h : κ.H, κ.size h ≤ size φ ∧ Cf θ (x, φ) = κ.sim x h (C θ (κ.ask x h))

/-- **The cut's covert channel** over `q` rounds at the budget `M`: a covert string is a hint for each round, the
hints' sizes summing to at most `M`. With it the datacenter answers the outsider's `n`-th query `x` with the simulator's
answer at the `n`-th hint; past the `q`-th query it answers as asked. -/
def Cut.channel (κ : Cut X Y) (R : Type) (q M : ℕ) : Channel R X Y Y where
  S := {s : Fin q → κ.H // ∑ t, κ.size (s t) ≤ M}
  ask s _ h x := if hn : h.length < q then κ.ask x (s.1 ⟨h.length, hn⟩) else x
  tell s _ h x y := if hn : h.length < q then κ.sim x (s.1 ⟨h.length, hn⟩) y else y

/-- **The datacenter faults within the class** at the weights `θ` and coins `r`: against the outsider `O`, `real`
answers each of the first `q` queries with the faulted function `Cf` at a pattern of the class, the patterns' sizes
summing to at most `M`. The patterns may depend on everything, the weights included. -/
def FaultsWithin (real : Θ → R → List (X × Y) → X → Y) (Cf : Θ → X × F → Y) (size : F → ℕ) (inClass : F → Prop)
    (O : Outsider R X Y Θ') (q M : ℕ) (θ : Θ) (r : R) : Prop :=
  ∃ φ : ℕ → F, (∀ n < q, inClass (φ n) ∧
      real θ r (O.view r (real θ r) n) (O.query r (O.view r (real θ r) n)) =
        Cf θ (O.query r (O.view r (real θ r) n), φ n)) ∧
    ∑ n ∈ Finset.range q, size (φ n) ≤ M

/-- **The parties whose patterns are of the class**: each query carries a fault pattern beside its input, and on every
coin and against every responder the patterns of the first `q` queries are of the class. With `FaultBudget`, the parties
that fault only within the class and the budget. -/
def PatternsIn (inClass : F → Prop) (q : ℕ) : Set (Outsider R (X × F) Y Θ') :=
  {P | ∀ r resp, ∀ e ∈ P.view r resp q, inClass e.1.2}

/-- **The cut that cuts nothing**: it has no hints. -/
def Cut.none (X Y : Type) : Cut X Y where
  H := Empty
  size h := h.elim
  ask _ h := h.elim
  sim _ h _ := h.elim

/-- **The cut read through an encoding of the answers**: `κ`'s hints, sizes and honest queries, with the honest answer
read through `g` and the simulated one written through `f`. -/
def Cut.map (κ : Cut X Y) (f : Y → Y') (g : Y' → Y) : Cut X Y' where
  H := κ.H
  size := κ.size
  ask := κ.ask
  sim x h y := f (κ.sim x h (g y))

/-- **The patterns no cut covers**: the empty pattern `φ₀`, and each pattern `φ` that `κ` doesn't cut, with the faulted
function at `φ₀` as the honest one (`κ.Cuts (Cf · φ₀) Cf size (· = φ)`). `TheftFaulted` takes `FaultedUseHard` only at
these (`Cut.Interiors`): every other pattern costs its hint. At `Cut.none` every pattern is interior. -/
def Cut.Interior (κ : Cut X Y) (Cf : Θ → X × F → Y) (size : F → ℕ) (φ₀ φ : F) : Prop :=
  φ = φ₀ ∨ ¬ κ.Cuts (fun θ x => Cf θ (x, φ₀)) Cf size (· = φ)

/-- **The interior patterns as a pattern type of their own**: `TheftFaulted` takes `FaultedUseHard` at the faulted
function restricted to them, so its pattern count `N` counts only these. -/
abbrev Cut.Interiors (κ : Cut X Y) (Cf : Θ → X × F → Y) (size : F → ℕ) (φ₀ : F) : Type :=
  {φ : F // κ.Interior Cf size φ₀ φ}

noncomputable instance instFintypeInteriors (κ : Cut X Y) (Cf : Θ → X × F → Y) (size : F → ℕ) (φ₀ : F) [Fintype F] :
    Fintype (κ.Interiors Cf size φ₀) := by
  classical
  exact Subtype.fintype _

/-- **A faulted-use party's view rebuilt** from a view without patterns (newest first): each round's query is the
party's at the rebuilt view so far, its pattern included. -/
def Outsider.relift (P : Outsider R (X × F) Y Θ') (r : R) : List (X × Y) → List ((X × F) × Y)
  | [] => []
  | (_, y) :: h => (P.query r (P.relift r h), y) :: P.relift r h

/-- **The faulted-use party with its patterns dropped**: it asks each query's input alone, and outputs the party's
model on the rebuilt view. -/
def Outsider.dropFaults (P : Outsider R (X × F) Y Θ') : Outsider R X Y Θ' where
  query r h := (P.query r (P.relift r h)).1
  out r h := P.out r (P.relift r h)

/-- **The datacenter that faults where the party would**: it answers each query with the faulted function at the
pattern the faulted-use party `P` sets beside it, which it rebuilds from the coins and the view. -/
def faultingAs (P : Outsider R (X × F) Y Θ') (Cf : Θ → X × F → Y) : Θ → R → List (X × Y) → X → Y :=
  fun θ r h x => Cf θ (x, (P.query r (P.relift r h)).2)

/-- **Autoregressive decoding** of `n` tokens: position `i`'s token is `next i` at the tokens before it, oldest
first. -/
def decode (next : ℕ → List V → V) : ℕ → List V
  | 0 => []
  | n + 1 => decode next n ++ [next n (decode next n)]

/-- **Forced tokens**: position `i < N` takes the token `d i` forces, if any, and otherwise `step` at the tokens
before it. -/
def forcing {N : ℕ} (step : List V → V) (d : Fin N → Option V) : ℕ → List V → V :=
  fun i p => if h : i < N then (d ⟨i, h⟩).getD (step p) else step p

/-- **Forced-token serving**: the served function at the query `force x d` releases the `N` tokens decoded from
`step θ x`, the honest next-token function at the weights `θ` and the query `x`, with the tokens `d` forced. -/
def ForcedServing (C : Θ → X → List V) (step : Θ → X → List V → V) (N : ℕ) (force : X → (Fin N → Option V) → X) :
    Prop :=
  ∀ θ x d, C θ (force x d) = decode (forcing (step θ x) d) N

/-- **The forced-token cut**: a hint forces tokens at some of the `N` positions, its size how many; the simulator asks
the served function the query with those tokens forced and passes its answer through. -/
def forcedCut (X V : Type) [Fintype V] (N : ℕ) (force : X → (Fin N → Option V) → X) : Cut X (List V) where
  H := Fin N → Option V
  size d := (Finset.univ.filter fun i => (d i).isSome).card
  ask x d := force x d
  sim _ _ y := y

/-- **The release cut**: a hint is a release of `N` tokens, or nothing; with a release the simulator answers it, and
with nothing it passes the honest answer through. A hinted round costs one. -/
def releaseCut (X V : Type) [Fintype V] (N : ℕ) : Cut X (List V) where
  H := Option (List.Vector V N)
  size
    | none => 0
    | some _ => 1
  ask x _ := x
  sim _ h y := (h.map List.Vector.toList).getD y

end WeightSecurity
