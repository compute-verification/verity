module

public import Definitions.ResourceAccounting.Served
public import Proofs.Flock.C3.Compose.Honest
public import Proofs.Flock.Soundness.Discharge.HiddenDec.Assemble
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteHP
public import Proofs.Flock.VBridge.Family.Registered
public import Proofs.Flock.EndToEndRegistered
public import Definitions.Pouw.TS.ProofLayer
public import Proofs.Flock.SoundAtBridge.Session
public import Proofs.Flock.C3.RegLevels.OpensRegs
public import Proofs.Flock.C3.RegLevels.ValueTie
public import Proofs.Flock.Recursive.ValueSrc
public import Proofs.Flock.Recursive.Registrants
public import Flock.KnownRead
public import PouwWindow.OwnPorts
public import PouwWindow.FullSample
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# What a lane of a leaf's option A statements reads

Under option A a sampled 32 × 32 H1 leaf is proved as 33 statements: its 32 lanes `PouwH1LaneZy_v1{TILE}` and their
combine `PouwH1Combine_v1{TILE}`.

* `Row`, `LaneView`: an A or B row, and what lane `l`'s statement reads;
* `PortRow`, `portRead`: a wire assignment at every bit of every unit's port rows, registered or not.
-/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open ResourceAccounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

/-- An A or B row: its `K` codes, then its scale. -/
structure Row (K : ℕ) where
  codes : Fin K → ℤ
  scale : ℤ

/-- **What lane `l`'s statement reads**: its A rows by row slot, its B rows by column slot, the leaf's record, its
partials row, the public inputs (admit, salt, position, roots), its index and its `ok` word. -/
structure LaneView (K nw : ℕ) (Pub : Type) where
  a : Fin 4 → Row K
  b : Fin 8 → Row K
  z : Fin 32 → Fin 32 → ℤ
  y : Fin 32 → Fin 32 → ℤ
  part : Fin nw → BitVec 32
  pub : Pub
  lane : Fin 32
  ok : Bool

/-- **Wire `g` is a bit of one of unit `u`'s port rows**: its commit string is `u`'s row of some port `q` of the
circuit, an input port or an output row, not a single gate's own string (`Readers.ofRows`). -/
def PortRow (s : RowsSession) (u : Fin s.n) (g : Fin s.Circ.N) : Prop :=
  ∃ q : Fin s.c.ports.size, s.vb.pos g = Sum.inl (u, q)

open Classical in
/-- **An assignment's port rows**: `vals` at every wire of unit `a` that is a bit of one of `a`'s port rows (`PortRow`),
registered or not; `none` elsewhere. -/
noncomputable def portRead (s : RowsSession) (vals : Fin s.Circ.N → Bool) : ℕ → ℕ → Option Bool :=
  fun a b =>
    if h : a < s.n ∧ b < s.Circ.N then
      if (⟨b, h.2⟩ : Fin s.Circ.N) ∈ s.P.io ⟨a, h.1⟩ ∧ PortRow s ⟨a, h.1⟩ ⟨b, h.2⟩ then some (vals ⟨b, h.2⟩)
      else none
    else none

end ResourceAccounting.TS.Lanes
