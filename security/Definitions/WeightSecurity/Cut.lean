module

public import Definitions.WeightSecurity

/-! Faults an honest cut simulates, which cost their hints' bits in a covert channel, with no premise beyond `HonestUseHard`. -/

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

/-- Past the `q`-th query the datacenter answers as asked. -/
def Cut.channel (κ : Cut X Y) (R : Type) (q M : ℕ) : Channel R X Y Y where
  S := {s : Fin q → κ.H // ∑ t, κ.size (s t) ≤ M}
  ask s _ h x := if hn : h.length < q then κ.ask x (s.1 ⟨h.length, hn⟩) else x
  tell s _ h x y := if hn : h.length < q then κ.sim x (s.1 ⟨h.length, hn⟩) y else y

/-- The patterns may depend on everything, the weights included. -/
def FaultsWithin (real : Θ → R → List (X × Y) → X → Y) (Cf : Θ → X × F → Y) (size : F → ℕ) (inClass : F → Prop)
    (O : Outsider R X Y Θ') (q M : ℕ) (θ : Θ) (r : R) : Prop :=
  ∃ φ : ℕ → F, (∀ n < q, inClass (φ n) ∧
      real θ r (O.view r (real θ r) n) (O.query r (O.view r (real θ r) n)) =
        Cf θ (O.query r (O.view r (real θ r) n), φ n)) ∧
    ∑ n ∈ Finset.range q, size (φ n) ≤ M

def PatternsIn (inClass : F → Prop) (q : ℕ) : Set (Outsider R (X × F) Y Θ') :=
  {P | ∀ r resp, ∀ e ∈ P.view r resp q, inClass e.1.2}

/-- **The cut that cuts nothing**: it has no hints. -/
def Cut.none (X Y : Type) : Cut X Y where
  H := Empty
  size h := h.elim
  ask _ h := h.elim
  sim _ h _ := h.elim

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

def Outsider.relift (P : Outsider R (X × F) Y Θ') (r : R) : List (X × Y) → List ((X × F) × Y)
  | [] => []
  | (_, y) :: h => (P.query r (P.relift r h), y) :: P.relift r h

def Outsider.dropFaults (P : Outsider R (X × F) Y Θ') : Outsider R X Y Θ' where
  query r h := (P.query r (P.relift r h)).1
  out r h := P.out r (P.relift r h)

def faultingAs (P : Outsider R (X × F) Y Θ') (Cf : Θ → X × F → Y) : Θ → R → List (X × Y) → X → Y :=
  fun θ r h x => Cf θ (x, (P.query r (P.relift r h)).2)

/-- Position `i`'s token is a function of the tokens before it, which holds when the release is tokens only, nothing
decodes ahead, and no state is shared across requests. -/
def decode (next : ℕ → List V → V) : ℕ → List V
  | 0 => []
  | n + 1 => decode next n ++ [next n (decode next n)]

def forcing {N : ℕ} (step : List V → V) (d : Fin N → Option V) : ℕ → List V → V :=
  fun i p => if h : i < N then (d ⟨i, h⟩).getD (step p) else step p

def ForcedServing (C : Θ → X → List V) (step : Θ → X → List V → V) (N : ℕ) (force : X → (Fin N → Option V) → X) :
    Prop :=
  ∀ θ x d, C θ (force x d) = decode (forcing (step θ x) d) N

def forcedCut (X V : Type) [Fintype V] (N : ℕ) (force : X → (Fin N → Option V) → X) : Cut X (List V) where
  H := Fin N → Option V
  size d := (Finset.univ.filter fun i => (d i).isSome).card
  ask x d := force x d
  sim _ _ y := y

/-- A hinted round costs one, no more than the fault itself for a block whose output is at least the release's bits. -/
def releaseCut (X V : Type) [Fintype V] (N : ℕ) : Cut X (List V) where
  H := Option (List.Vector V N)
  size
    | none => 0
    | some _ => 1
  ask x _ := x
  sim _ h y := (h.map List.Vector.toList).getD y

end WeightSecurity
