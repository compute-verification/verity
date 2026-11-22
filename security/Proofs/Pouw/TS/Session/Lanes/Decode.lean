module

public import Proofs.Pouw.TS.Session.Lanes.Layer
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
# A lane's decode, over its statement's port layout

* `bitsN`, `sgnBits`: a field's bits as a natural number, least significant first, and its two's complement;
* `decLane T rd`: lane `l`'s view from a reader `rd` of ports' row bits, each field at its port and first bit in the
  layout (`fieldBit`, `PouwWindow.Lanes.fieldOf?`): an A row's codes (s8) and scale (its f32 encoding), the record's z
  (s32) and y (its bf16 encoding) cell by cell, a partials row's u32 words, the admit bits, salt, position and roots
  (`LanePub`), the index and `ok`.

A field's place in its port's row is the same in every lane (`lanePorts`' fields don't read the lane), so `decLane`
reads them at lane 0's ports.
-/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open PouwWindow.Lanes Flock.SoundAtBridge
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW)

/-- The natural number a reader's bits `[lo, lo + n)` spell, least significant first, a missing bit read as 0. -/
def bitsN (rd : ℕ → Option Bool) (lo n : ℕ) : ℕ :=
  ∑ i ∈ Finset.range n, if rd (lo + i) = some true then 2 ^ i else 0

/-- A field's `n` bits spell a number below `2^n`. -/
theorem bitsN_lt (rd : ℕ → Option Bool) (lo : ℕ) : ∀ n, bitsN rd lo n < 2 ^ n
  | 0 => by simp [bitsN]
  | n + 1 => by
    have ih := bitsN_lt rd lo n
    have e : bitsN rd lo (n + 1) = bitsN rd lo n + (if rd (lo + n) = some true then 2 ^ n else 0) := by
      unfold bitsN
      rw [Finset.sum_range_succ]
    rw [e, pow_succ]
    split <;> omega

/-- Two's complement at `n` bits. -/
def sgnBits (n x : ℕ) : ℤ := if x < 2 ^ (n - 1) then (x : ℤ) else (x : ℤ) - 2 ^ n

/-- The first bit of the field at `path` among the ports `ps` (`fieldOf?`), 0 where none holds it. -/
def fieldBit (ps : List Port) (path : String) : ℕ := ((fieldOf? ps path).map fun pf => pf.2.bit).getD 0

/-- **A lane's public inputs** as its statement reads them: its admit bits and the salt, position and roots, each
within its field's width (a digest's 512 bits, a u64, two digests), as the decode reads them from those bits. -/
structure LanePub where
  admit : Fin 32 → Bool
  salt : ℕ
  position : ℕ
  roots : ℕ
  salt_lt : salt < 2 ^ digestBits
  position_lt : position < 2 ^ positionBits
  roots_lt : roots < 2 ^ (2 * digestBits)

/-- **Lane `l`'s view from a reader of its ports' row bits**, each field at its port and first bit in the tile's layout:
an A row at row slot `s` from port `a<s>`, a B row at column slot `t` from `b<t>` (its `K` s8 codes, then its scale's
encoding), the record from `zy` (cell `(i, j)`'s s32 z and its y's encoding), the partials row's `nw` u32 words from
`partials`, the admit bits from `admit`, the salt, position, roots and index from `pub`, and `ok` from `o0`. -/
noncomputable def decLane (K nw : ℕ) (T : Tile) (rd : String → ℕ → Option Bool) : LaneView K nw LanePub where
  a s := ⟨fun k => sgnBits 8 (bitsN (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].codes" + 8 * k.val) 8),
    (bitsN (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].scale") 32 : ℤ)⟩
  b t := ⟨fun k => sgnBits 8 (bitsN (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].codes" + 8 * k.val) 8),
    (bitsN (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].scale") 32 : ℤ)⟩
  z i j := sgnBits 32 (bitsN (rd "zy") (fieldBit (lanePorts T 0) "zy.z" + 32 * (i.val * T.c + j.val)) 32)
  y i j := (bitsN (rd "zy") (fieldBit (lanePorts T 0) "zy.y" + 16 * (i.val * T.c + j.val)) 16 : ℤ)
  part w := BitVec.ofNat 32 (bitsN (rd "partials") (fieldBit (lanePorts T 0) "partials" + 32 * w.val) 32)
  pub := ⟨fun i => rd "admit" (fieldBit (lanePorts T 0) "admit" + i.val) == some true,
    bitsN (rd "pub") (fieldBit (lanePorts T 0) "salt") digestBits,
    bitsN (rd "pub") (fieldBit (lanePorts T 0) "position") positionBits,
    bitsN (rd "pub") (fieldBit (lanePorts T 0) "roots") (2 * digestBits), bitsN_lt _ _ _, bitsN_lt _ _ _,
    bitsN_lt _ _ _⟩
  lane := ⟨bitsN (rd "pub") (fieldBit (lanePorts T 0) "lane") laneBits % 32, Nat.mod_lt _ (by norm_num)⟩
  ok := rd "o0" 0 == some true

end ResourceAccounting.TS.Lanes
