module

public import Definitions.Pouw.TS.ProofLayer
public import Proofs.Pouw.TS.Session.Assumptions
public import Proofs.Flock.SoundAtBridge.Session
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

/-! A checked unit's proof layer through its Flock sessions: accepted where verity-verify-proof `--zk --registered`
accepts, opening the committed rows at the registered reads, its relation `CheckedRel`. -/

@[expose] public section

namespace ResourceAccounting.TS.Checked

open ResourceAccounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

open Classical in
/-- **A session's registered reads, as a layer value**: at unit `a` and wire `b`, `f` of them where `b` is a wire of
unit `a` (`P.io`) at a registered read of `rdU a`; `none` elsewhere. -/
noncomputable def regRead (s : RowsSession) (f : Fin s.n → Fin s.Circ.N → Bool) : ℕ → ℕ → Option Bool :=
  fun a b =>
    if h : a < s.n ∧ b < s.Circ.N then
      if (⟨b, h.2⟩ : Fin s.Circ.N) ∈ s.P.io ⟨a, h.1⟩ ∧ (s.rdU ⟨a, h.1⟩).isRead (s.vb.pos ⟨b, h.2⟩) then
        some (f ⟨a, h.1⟩ ⟨b, h.2⟩)
      else none
    else none

/-- **An assignment's registered reads**: `vals` at every registered read of every unit. -/
noncomputable def rowsReg (s : RowsSession) (vals : Fin s.Circ.N → Bool) : ℕ → ℕ → Option Bool :=
  regRead s fun _ g => vals g

/-- **The committed rows at the registered reads**: the bit of the registrant's opening of unit `u`'s row at each of
its wires there (`RowsSession.wo`, the commit-time opening the service registers with the root). -/
noncomputable def openedReg (s : RowsSession) : ℕ → ℕ → Option Bool :=
  regRead s fun u g => s.vb.bit (s.vb.val (s.wo u (s.vb.pos g))) g

/-- **The checked unit's relation at `ok`**: the circuit has a port named `ok`, `own` registers a port of that name,
and the layer value `v` is the registered reads of one wire assignment the circuit's evaluation agrees with on every
wire of every sampled proof block of the outcome `o`. -/
def CheckedRel (s : RowsSession) (o : s.Out) (ok : String) (v : ℕ → ℕ → Option Bool) : Prop :=
  ((∃ q : Fin s.c.ports.size, s.c.ports[q].name = ok) ∧ ∃ pt ∈ s.own, pt.name = ok) ∧
    ∃ vals : Fin s.Circ.N → Bool, v = rowsReg s vals ∧ ∀ u ∈ o.2.1, ∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g

/-- **The checked layer through its Flock sessions**: proof `i` of round `w` is accepted where verity-verify-proof
`--zk --registered` accepts its session, opens the committed rows at the registered reads (`openedReg`), and its
relation is the unit's Definition with its check word `ok` registered (`CheckedRel`). -/
noncomputable def checkedLayer {Ω : Type} {S nπ : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) :
    ProofLayer Ω S nπ (ℕ → ℕ → Option Bool) where
  accepted x w i := LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)
  opened x w i := openedReg (sess x w i)
  rel x w i v := CheckedRel (sess x w i) (out x w i) ok v
  right := right

end ResourceAccounting.TS.Checked
