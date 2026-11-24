module

public import Definitions.Certifier.Bridge
public import Definitions.Poub.Bandwidth.Causal

/-!
# The chip-and-store model (trusted)

A candidate for the feasible set, stated over a strategy's runs on the actual circuit: a chip (a location), its store
(a location, reached over the store's ports) and the outside (every other leaf, reached over the outside ports), with
the limits the auditor supplies. Each property is an existing definition read at the chip or the store, or is new
where none exists (`FixedBefore`, `Wired`, `Listed`).

The two rates on ports have two forms, and neither gives the other. `PortBits` counts the bits a link delivers in an
interval (`Placed.Links`), which is what the device model's schedule charges. `Alphabet` counts the signals a link can
deliver at a tick, a wire's arrival tick and its source's position included, which is what a counting argument over a
strategy's runs needs: when a bit arrives is information.
-/

@[expose] public section

namespace ResourceAccounting.ChipStore

open Bandwidth.Causal

/-- The object: the chip, the store, and the ports wires into the chip run on (a link and the leaf of the chip it
delivers to). -/
structure Ports (L K : Type*) where
  chip : Finset L
  store : Finset L
  storePorts : List (K × L)
  outsidePorts : List (K × L)

variable {Ω L K : Type*}

/-- **Three channels, the wiring**: every wire into the chip from outside it runs on a port of the store or of the
outside. -/
def Wired (S : Ω → Actual.Run L) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (P : Ports L K) : Prop :=
  ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ P.chip → (S r).A.x g ∈ P.chip →
    (link r g a, (S r).A.x g) ∈ P.storePorts ++ P.outsidePorts

/-- **Fixed before the coins**: the coins `ρ` exist from `t₀`, so what the chip holds at `t₀` is the same at every
value of them. -/
def FixedBefore {ε ρ : Type*} [LinearOrder L] (S : ε × ρ → Actual.Run L) (X : Finset L) (t₀ : ℕ) : Prop :=
  ∀ e o o', (S (e, o)).content X t₀ = (S (e, o')).content X t₀

/-- **Chip state**: the chip holds at most `s` bits at every tick. -/
def ChipState (S : Ω → Actual.Run L) (P : Ports L K) (s : ℕ) : Prop := ∀ r t, (S r).mem P.chip t ≤ s

/-- **Store size**: the store holds at most `m` bits at every tick. -/
def StoreSize (S : Ω → Actual.Run L) (P : Ports L K) (m : ℕ) : Prop := ∀ r t, (S r).mem P.store t ≤ m

/-- **Port rates, in bits**: in every run the links join the leaves they carry wires between, and each delivers at
most `β e` bits a tick over every interval. -/
def PortBits (S : Ω → Actual.Run L) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (ends : K → L × L)
    (β : K → ℕ) : Prop :=
  ∀ r, (S r).A.Links ends β (link r)

open Classical in
/-- **Work rate**: in every run, the chip and its store together run at most `w o` ops of class `o` a tick, under the
labelling `cls` of gates by op class. -/
def WorkRate (S : Ω → Actual.Run L) (P : Ports L K) (cls : ∀ r, Fin (S r).A.N → Option OpClass)
    (w : OpClass → ℕ) : Prop :=
  ∀ r o, (S r).A.ComputeRate {P.chip ∪ P.store} ((S r).A.classPrice (cls r) o) (fun _ => w o)

/-- **Every chip operation is of a listed type**: `cls` labels every computed gate placed at the chip, so none runs
free of `WorkRate`. -/
def Listed (S : Ω → Actual.Run L) (P : Ports L K) (cls : ∀ r, Fin (S r).A.N → Option OpClass) : Prop :=
  ∀ r g, (S r).A.isInput g = false → (S r).A.x g ∈ P.chip → cls r g ≠ none

/-- **The structure and the limits a certificate over a window reads**, for a strategy over an input `ε` fixed before
the window and coins `ρ` drawn at `t₀`: causality from every tick, the wiring, a chip fixed before the coins and
holding at most `s` bits, and each link an alphabet of `2^{β e}` signals a tick. -/
structure Certifies {ε ρ : Type*} [LinearOrder L] (S : ε × ρ → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (P : Ports L K) (t₀ s : ℕ) (β : K → ℕ) : Prop where
  causal : CausalFrom S
  wired : Wired S link P
  fixed : FixedBefore S P.chip t₀
  chip : ChipState S P s
  alphabet : Alphabet S β link

open Classical in
/-- **The limits a subtraction reads**, in every run: the chip's and the store's states, the links' bits and the work
rates. -/
structure Limits (S : Ω → Actual.Run L) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K)
    (P : Ports L K) (cls : ∀ r, Fin (S r).A.N → Option OpClass) (ends : K → L × L) (s m : ℕ) (β : K → ℕ)
    (w : OpClass → ℕ) : Prop where
  chip : ChipState S P s
  store : StoreSize S P m
  bits : PortBits S link ends β
  work : WorkRate S P cls w

end ResourceAccounting.ChipStore
