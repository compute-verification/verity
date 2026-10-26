import Properties.Flock.Level3.Seam
import Flock.Piop
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Prod

/-!
C-Flock's executable arithmetic as `Arith`'s fields: `GF128` and `GF256` finite (`f128Equiv`, `f256Equiv`), the packing
basis `x^v` (`pack`, `unpack`), the query positions' low word (`lo`), the Reed–Solomon domain points (`omega`) and the
pinned zerocheck coordinates (`pinnedF`). Their facts, each in the shape of its `Arith.Correct` field, are
`Proofs.Flock.Level3.ArithFacts`.
-/

open Polynomial Flock

set_option maxRecDepth 20000

namespace FlockLevel3

/-! ### The fields: finite, of order `2^128` and `2^256`, characteristic 2 -/

def f128Equiv : GF128 ≃ Fin (2 ^ 64) × Fin (2 ^ 64) :=
  ⟨fun a => (GF128.val a |>.lo.toFin, GF128.val a |>.hi.toFin), fun p => GF128.of ⟨UInt64.ofFin p.1, UInt64.ofFin p.2⟩,
    fun _ => rfl, fun _ => rfl⟩

instance instFintypeGF128 : Fintype GF128 := Fintype.ofEquiv _ f128Equiv.symm

instance instDecidableEqGF128 : DecidableEq GF128 := inferInstanceAs (DecidableEq F128)

def f256Equiv : GF256 ≃ GF128 × GF128 :=
  ⟨fun a => ((a : F256).c0, (a : F256).c1), fun p => GF256.of ⟨p.1.val, p.2.val⟩, fun _ => rfl, fun _ => rfl⟩

instance instFintypeGF256 : Fintype GF256 := Fintype.ofEquiv _ f256Equiv.symm

instance instDecidableEqGF256 : DecidableEq GF256 := inferInstanceAs (DecidableEq F256)

/-! ### Packing: bit `v` is the coefficient of `x^v` (§12.1) -/

/-- The bits of an element in the packing basis `x^v`. -/
noncomputable def unpack (a : GF128) : Fin 128 → ZMod 2 := fun v => (F128.poly a.val).coeff v

/-- `Σ_v b_v · x^v`. -/
noncomputable def pack (b : Fin 128 → ZMod 2) : GF128 :=
  GF128.of (equivAdj.symm (AdjoinRoot.mk G (∑ v : Fin 128, C (b v) * X ^ (v : ℕ))))

/-! ### Query positions: the low bits of `lo` are balanced (§13.5) -/

/-- The low 64-bit word. -/
def lo (w : GF128) : ℕ := (GF128.val w).lo.toNat

/-! ### The Reed–Solomon domain points (§13.1) -/

/-- `ω_q`: the element whose low word is `q`. -/
def omega (q : ℕ) : GF128 := GF128.of ⟨q.toUInt64, 0⟩

/-! ### The pinned zerocheck coordinates (§9.2) -/

/-- The seven pinned coordinates (the executable's `Flock.pinned`). -/
def pinnedF (i : Fin 7) : GF128 := GF128.of (Flock.pinned[i.val]!)

end FlockLevel3
