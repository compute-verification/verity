module

public import Proofs.Pouw.TS.Session.Lanes.Lane
public import Proofs.Pouw.TS.Session.Tile.Assumptions
public import Proofs.Pouw.H1Zy.Tile
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
import Proofs.Pouw.H1Zy.Lane

/-! A tile unit's `TileChecks` from its Definition: its `ok = 1` means `TileOk` at every input and a `K` the tile binds
at, which leaves `TileComputes`, with `rd`, the v3 noise and `dqW`'s float roundings as parameters. -/

@[expose] public section

namespace ResourceAccounting.TS.Tile

open ResourceAccounting.TS ResourceAccounting.TS.Checked ResourceAccounting.TS.Held Flock.SoundAtBridge

/-- **The tile's words as `TileView` reads them**: codes and z their s32 and s8 values, scales and y their encodings
as unsigned numbers. -/
def tileView {R C K : ℕ} {P : Type} (x : Pouw.H1Zy.TileIn R C K P) : TileView R C K where
  a i k := (x.a i k).toInt
  sa i := ((x.sa i).toNat : ℤ)
  b j k := (x.b j k).toInt
  sb j := ((x.sw j).toNat : ℤ)
  z i j := (x.zr i j).toInt
  y i j := ((x.yr i j).toNat : ℤ)

private theorem ofInt_toNat32 (v : BitVec 32) : BitVec.ofInt 32 (v.toNat : ℤ) = v := by
  rw [BitVec.ofInt_natCast, BitVec.ofNat_toNat, BitVec.setWidth_eq]

theorem tileOk_of_def {R C K : ℕ} {P : Type} (form : Pouw.H1Zy.TileForm R C K P)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : Pouw.H1Zy.TileIn R C K P) (hK32 : 32 ∣ K)
    (hK : K < 2 ^ 17) (h : Pouw.H1Zy.tileOk form dqW x = true) : TileOk (Lanes.dqOf dqW) (tileView x) := by
  intro i j
  obtain ⟨hz, hy⟩ := Pouw.H1Zy.tileOk_spec form dqW x hK32 hK h i j
  refine ⟨?_, ?_⟩
  · show (x.zr i j).toInt = ∑ k, (x.a i k).toInt * (x.b j k).toInt
    rw [hz]
    exact Pouw.H1Zy.toInt_ofInt_dot _ _ hK32 hK
  · show ((x.yr i j).toNat : ℤ) = ((dqW (BitVec.ofInt 32 (x.zr i j).toInt) (BitVec.ofInt 32 ((x.sa i).toNat : ℤ))
      (BitVec.ofInt 32 ((x.sw j).toNat : ℤ))).toNat : ℤ)
    rw [BitVec.ofInt_toInt, ofInt_toNat32, ofInt_toNat32, hy, hz]

/-- **A tile session's circuit computes its Definition at its registered reads** (claims 1 and 2 of `TileChecks`):
at the session `s`, on every wire assignment `vals` the key program's evaluation agrees with on every wire of unit `u`,
whose `ok` wires of `u` each carry the bit of a value that is the `ok` row at its position (`OkVal`, the one word `1`),
`PouwH1TileZy_v1{TILE}`'s `ok` (`Pouw.H1Zy.tileOk` at TILE's form `form` and the dequant `dqW`) is `true` on the words
`rd` reads from `vals`' registered reads at `u`. -/
def TileComputes {R C K : ℕ} {P : Type} (form : Pouw.H1Zy.TileForm R C K P)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (s : RowsSession) (ok : String)
    (rd : (ℕ → ℕ → Option Bool) → ℕ → Pouw.H1Zy.TileIn R C K P) : Prop :=
  ∀ (vals : Fin s.Circ.N → Bool) (u : Fin s.n), (∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g) →
    (∀ g, OkWire s ok u g → ∃ w, OkVal s (s.vb.pos g) w ∧ vals g = s.vb.bit w g) →
    Pouw.H1Zy.tileOk form dqW (rd (rowsReg s vals) u.val) = true

theorem tileChecks_of_computes {R C K : ℕ} {P : Type} (form : Pouw.H1Zy.TileForm R C K P)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (s : RowsSession) (ok : String)
    (rd : (ℕ → ℕ → Option Bool) → ℕ → Pouw.H1Zy.TileIn R C K P) (h : TileComputes form dqW s ok rd) :
    ResourceAccounting.Assumptions.TS.TileChecks (Lanes.dqOf dqW) s ok (fun v u => tileView (rd v u)) :=
  fun hK32 hK vals u hv hok => tileOk_of_def form dqW _ hK32 hK (h vals u hv hok)

end ResourceAccounting.TS.Tile
