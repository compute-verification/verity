module

public import Definitions.Pouw.PearlC.Tile
public import Definitions.Core.Partition

/-! The tile check as a circuit and L3, that it computes the good-tile predicate, which says something only of a tile check tied to the deployed Program by its descriptor digest and a test. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-- A tile check: the tile-check Program as a core circuit for each unit shape, its partition into proof blocks, the
wires a tile is checked at, the public wires with the anchors the verifier derives and its check of them, the registered
wires with the values a commitment gives them, and the event outside which those values bind the circuit's words (see
the module docstring). -/
structure TileCheck (Q R S : Type) where
  C : Shape → Verity.Circuit
  n : Shape → ℕ
  part : (sh : Shape) → Verity.Partition (C sh) (n sh)
  out : (sh : Shape) → Finset ℕ → Finset ℕ → Finset (Fin (C sh).N)
  pub : (sh : Shape) → Finset (Fin (C sh).N)
  reg : (sh : Shape) → Finset (Fin (C sh).N)
  anchorsOf : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes → (C (U.layout.shape u)).Trace
  anchorsOK : (Q → R) → S → Workload → ℕ → Codes → Prop
  rowsOf : (U : Workload) → (u : ℕ) → Commit → (C (U.layout.shape u)).Trace
  Collides : (U : Workload) → (u : ℕ) → Finset ℕ → Finset ℕ → Commit → (C (U.layout.shape u)).Trace → Prop

def TileCheck.valueAt (TC : TileCheck Q R S) (U : Workload) (τ : Trace) (u : ℕ) :
    (TC.C (U.layout.shape u)).Trace :=
  match τ u with
  | some c => TC.rowsOf U u c
  | none => fun _ => 0

/-- **Tile `g`'s check accepts `X`** for the commitment `c` of its unit: every unit of the cone of the tile's checked
wires is correct in `X`, `X` holds the anchors on the public wires, the anchors pass the verifier's check
(`TC.anchorsOK`), `X` holds `c`'s values on the registered wires, and each of the tile's weight rows passes the rules
(`P.fixedOK`). The check decides the anchors' check from the public seeds and the last from the weight rows it opens,
which are `U`'s, never from `X`. -/
def TileCheck.Accepts (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (H : Q → R) (s : S)
    (U : Workload) (g : ℕ) (c : Commit) (X : (TC.C (U.layout.shape ((TR.tiling U.layout).unit g))).Trace) : Prop :=
  (∀ v ∈ (TC.part _).cone (TC.out _ ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g)),
      (TC.part _).Correct X v) ∧
    (∀ w ∈ TC.pub _, X w = TC.anchorsOf H s U ((TR.tiling U.layout).unit g) c.act w) ∧
    TC.anchorsOK H s U ((TR.tiling U.layout).unit g) c.act ∧
    (∀ w ∈ TC.reg _, X w = TC.rowsOf U ((TR.tiling U.layout).unit g) c w) ∧
    ∀ j ∈ (TR.tiling U.layout).cols g,
      P.fixedOK (U.layout.shape ((TR.tiling U.layout).unit g)) (U.fixed (U.layout.fixedId ((TR.tiling U.layout).unit g))) j

/-- The words of a good tile: each ticket and useful word of a credited row has the reference value. `TileGood` is the
cap, the tile's weight rows and these. -/
def TileWordsGood (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S) (g : ℕ) (c : Commit) :
    Prop :=
  ∀ i ∈ (TR.tiling U.layout).rows g, TR.rowOK (U.layout.shape ((TR.tiling U.layout).unit g)) c.act i →
    ∀ j ∈ (TR.tiling U.layout).cols g,
      wordQ (c.ticket i j) = P.ticket H s U ((TR.tiling U.layout).unit g) c.act i j ∧
        wordQ (c.useful i j) = P.useful H s U ((TR.tiling U.layout).unit g) c.act i j

def TileCheck.Computes (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X →
      TileGood P TR U H s g c ∨
        TC.Collides U ((TR.tiling U.layout).unit g) ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g) c X

/-- The Program has no debit or cap, so its check leaves the activations' debit free: this holds of it only once a cap
unit returns the cap's verdict on a public wire. -/
def TileCheck.ComputesInertBound (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X → TR.inertBoundOK H s U g c.act

def TileCheck.ComputesWords (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X →
      TileWordsGood P TR U H s g c ∨
        TC.Collides U ((TR.tiling U.layout).unit g) ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g) c X

end Pouw.PearlC
