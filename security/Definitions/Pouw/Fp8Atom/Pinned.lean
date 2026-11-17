module

public import Definitions.Pouw.Fp8Atom.Vectors
public import Definitions.Pouw.Fp8Atom.Vectors.Ada
public import Definitions.Pouw.Fp8Atom.Vectors.Add
public import Mathlib.Data.Finset.Card
public import Mathlib.Data.Fintype.Basic

/-! The guarantees of the bit-exact `HOPPER_E4M3_K32` step and of the FP32 add `rnAdd`; conformance rests on pinned vectors, and the larger differential runs are a compiled test, not a proof. -/

@[expose] public section

namespace Pouw.Fp8Atom

def Conformance : Prop :=
  631 ≤ vectors.length ∧ ∀ v ∈ vectors, hopperStep v.acc (codes v.a) (codes v.b) = v.d

def AdaCaptures : Prop :=
  Vectors.ada.length = 360 ∧ ∀ v ∈ Vectors.ada, adaStep v.acc (codes v.a) (codes v.b) = v.d

def CodeClasses : Prop :=
  (∀ c : Code, c.isNaN = true ↔ c = 0x7F ∨ c = 0xFF) ∧
  (∀ c : Code, decode c = none ↔ c.isNaN = true) ∧
  (∀ c : Code, decode c = some 0 ↔ c = 0x00 ∨ c = 0x80) ∧
  (Finset.univ.filter fun c : Code => c.isNaN = false).card = 254

def DecodeInjective : Prop :=
  ∀ a b : Code, a.isNaN = false → decode a = decode b → a = b ∨ (a.isZero = true ∧ b.isZero = true)

def ProductExact : Prop :=
  ∀ (a b : Code) (x y : ℚ), decode a = some x → decode b = some y → (product a b).value = x * y

def AlignedMono : Prop :=
  ∀ (width : Nat) (E : Int) (s t : Term), width ≤ 24 → s.exp ≤ E → t.exp ≤ E → s.value ≤ t.value →
    aligned width E s ≤ aligned width E t

def AddConformance : Prop :=
  1090 ≤ Vectors.add.length ∧ ∀ v ∈ Vectors.add,
    ∃ x y z : ℚ, wordQ v.a = some x ∧ wordQ v.b = some y ∧ wordQ v.c = some z ∧ rnAdd x y = z

def RnAddExact : Prop := ∀ x y : ℚ, Representable (x + y) → rnAdd x y = x + y

def RnAddMono : Prop := ∀ x y x' y' : ℚ, x + y ≤ x' + y' → rnAdd x y ≤ rnAdd x' y'

def RnAddAbsorb : Prop :=
  ∀ r i : ℚ, Representable r → r ≠ 0 → 2 * |i| < ulp r → (2 ^ binade |r| < |r| ∨ 0 ≤ r * i) → rnAdd r i = r

def RnAddMoves : Prop := ∀ r i : ℚ, Representable r → r ≠ 0 → ulp r ≤ |i| → rnAdd r i ≠ r

def WordsRepresentable : Prop := ∀ (w : Nat) (x : ℚ), wordQ w = some x → Representable x

end Pouw.Fp8Atom
