module

public import Proofs.Poub.Lemmas.Bandwidth
public import Mathlib.RingTheory.WittVector.IsPoly
public import Mathlib.Tactic.ReduceModChar
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
import Mathlib.Tactic.ENatToNat
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
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# The honest device: the guarantee is not vacuous

A device with no on-chip state, no side sources and no regeneration, whose device memory holds the store as it is,
reads every block once and queries the oracle at it. It is accepted on every store, oracle and sample, and its
ingress is exactly the store, `N·b` bits (`honest_accepts`; the guarantee `Bandwidth.SecurityProofs.HonestPasses`).
-/

@[expose] public section

namespace Bandwidth.Completeness

open Pous (Bits)
open scoped ENNReal

variable {N b h : ℕ} {α : Type}

/-- Device memory holding the store as it is: bit `i` of block `j` at address `j·b + i`. -/
def layout (E : Store N b) (a : ℕ) : Bool :=
  if ha : a < N * b then
    E ⟨a / b, Nat.div_lt_of_lt_mul (by rwa [mul_comm] at ha)⟩
      ⟨a % b, Nat.mod_lt _ (Nat.pos_of_ne_zero (by rintro rfl; simp at ha))⟩
  else false

theorem layout_block (E : Store N b) (j : Fin N) (i : Fin b) : layout E (j * b + i) = E j i := by
  have hb : 0 < b := Nat.lt_of_le_of_lt (Nat.zero_le _) i.2
  have hlt : (j : ℕ) * b + i < N * b := by
    have := j.2; have := i.2
    calc (j : ℕ) * b + i < j * b + b := by omega
      _ = (j + 1) * b := by ring
      _ ≤ N * b := Nat.mul_le_mul_right _ (by omega)
  have hdiv : ((j : ℕ) * b + i) / b = j := by
    rw [add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt i.2, zero_add]
  have hmod : ((j : ℕ) * b + i) % b = i := by
    rw [add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt i.2]
  simp only [layout, hlt, ↓reduceDIte]
  congr 1
  · exact Fin.ext hdiv
  · exact Fin.ext hmod

/-- Read `n` bits of device memory from address `a`, then continue with them. -/
def readBits (a : ℕ) : (n : ℕ) → (Bits n → Chip N b h α) → Chip N b h α
  | 0, k => k Fin.elim0
  | n + 1, k => .ingress a (fun x => readBits (a + 1) n (fun v => k (Fin.cons x v)))


/-- Read each listed block and query the oracle at it, recording the tag. -/
def loop : List (Fin N) → (Fin N → Bits h) → Chip N b h (Fin N → Bits h)
  | [], acc => .done acc
  | j :: js, acc => readBits (j * b) b (fun x => .tag (j, x) (fun y => loop js (Function.update acc j y)))

/-- Block `j` as device memory holds it at `j·b`. -/
def blockAt (w : World N b h) (j : Fin N) : Bits b := fun i => w.mem (j * b + i)


/-- The honest device: no on-chip state, no side sources, no regeneration; device memory holds the store as it is,
and the chip reads every block once and queries the oracle at it. -/
def honest (N b h : ℕ) : Device N b h 0 0 0 N where
  state _ := Fin.elim0
  mem := layout
  side _ _ := false
  chip _ := loop (List.finRange N) (fun _ _ => false)
  bounded st w := by
    obtain ⟨-, -, h3, h4, h5⟩ := loop_eq w (List.finRange N) (fun _ _ => false)
    simp [h3, h4, h5]

theorem honest_tags (E : Store N b) (O : Oracle N b h) (j : Fin N) :
    (honest N b h).tags E O j = O (j, E j) := by
  have := congrFun (loop_eq ((honest N b h).world E O) (List.finRange N) (fun _ _ => false)).1 j
  simp only [List.mem_finRange, ↓reduceIte] at this
  refine this.trans (congrArg _ (Prod.ext rfl (funext (fun i => ?_))))
  exact layout_block E j i

/-- **Completeness.** The honest device is accepted on every store, oracle and sample, and reads exactly the store. -/
theorem honest_accepts {s : ℕ} (E : Store N b) (O : Oracle N b h) (I : Fin s → Fin N) :
    Accepts (honest N b h) E O I ∧ (honest N b h).ingress E O = N * b := by
  refine ⟨fun i => honest_tags E O (I i), ?_⟩
  have := (loop_eq ((honest N b h).world E O) (List.finRange N) (fun _ _ => false)).2.1
  simpa [Device.ingress, honest] using this

end Bandwidth.Completeness

theorem Bandwidth.SecurityProofs.HonestPasses : Bandwidth.Guarantees.HonestPasses :=
  fun N b h => ⟨Bandwidth.Completeness.honest N b h, fun E O _ I => Bandwidth.Completeness.honest_accepts E O I⟩
