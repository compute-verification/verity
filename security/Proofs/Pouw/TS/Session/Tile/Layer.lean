module

public import Proofs.Pouw.TS.Session.Held.Layer
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

/-! A tile unit's proof layer through its Flock sessions, whose relation `TileRel` states on words what the tile's
Definition checks. -/

@[expose] public section

namespace ResourceAccounting.TS.Tile

open ResourceAccounting.TS ResourceAccounting.TS.Checked ResourceAccounting.TS.Held Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

/-- **What a tile unit reads**: its `R` A rows and `C` B rows, each `K` codes and a scale, and its z/y record,
row-major. -/
structure TileView (R C K : ℕ) where
  a : Fin R → Fin K → ℤ
  sa : Fin R → ℤ
  b : Fin C → Fin K → ℤ
  sb : Fin C → ℤ
  z : Fin R → Fin C → ℤ
  y : Fin R → Fin C → ℤ

/-- **The tile's checks at `ok = 1`**: every cell's `z` is the dot of its A row's and B row's codes, and its `y` is
`dq` of its `z` at the A row's and the B row's scales. -/
def TileOk {R C K : ℕ} (dq : List ℤ → ℤ) (v : TileView R C K) : Prop :=
  ∀ i j, v.z i j = ∑ k, v.a i k * v.b j k ∧ v.y i j = dq [v.z i j, v.sa i, v.sb j]

/-- **The tile unit's relation with its Definition's checks**: `CheckedRel`, and every sampled proof block `u`'s view decoded
from the layer value `v` (`dec v u`) meets `TileOk`; or the `ok` tie collides (`OkColl`) at the `ok` port's known rows
in `known`. -/
def TileRel {R C K : ℕ} (s : RowsSession) (o : s.Out) (ok : String) (dq : List ℤ → ℤ)
    (dec : (ℕ → ℕ → Option Bool) → ℕ → TileView R C K) (known : Array (String × Array (Array ℕ × ByteArray)))
    (v : ℕ → ℕ → Option Bool) : Prop :=
  CheckedRel s o ok v ∧ ((∀ u ∈ o.2.1, TileOk dq (dec v u.val)) ∨ ∃ ks, (ok, ks) ∈ known ∧ OkColl s ok ks)

/-- **The tile layer through its Flock sessions**: proof `i` of round `w` is accepted where verity-verify-proof
`--zk --registered` accepts its session at the registered file's known content `known x w i` (`LiveAcceptsZKR`, and
`Registered.checkKnownAll`) and under its complete check (`Registered.checkComplete`, run whenever `--registered` is
given: every port of the circuit is a port of `own`), opens the committed rows at the registered reads
(`openedReg`), and its relation is `TileRel` at the decode `dec x w i`. -/
noncomputable def tileLayer {Ω : Type} {S nπ R C K : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (known : Paths Ω S → Fin S → Fin nπ → Array (String × Array (Array ℕ × ByteArray))) (dq : List ℤ → ℤ)
    (dec : Paths Ω S → Fin S → Fin nπ → (ℕ → ℕ → Option Bool) → ℕ → TileView R C K)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) :
    ProofLayer Ω S nπ (ℕ → ℕ → Option Bool) where
  accepted x w i := LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) ∧
    Flock.Registered.checkKnownAll (sess x w i).own (known x w i) = .ok () ∧
    Flock.Registered.checkComplete (sess x w i).c (sess x w i).own = .ok ()
  opened x w i := openedReg (sess x w i)
  rel x w i v := TileRel (sess x w i) (out x w i) ok dq (dec x w i) (known x w i) v
  right := right

end ResourceAccounting.TS.Tile
