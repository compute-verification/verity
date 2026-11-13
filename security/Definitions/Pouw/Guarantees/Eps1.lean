module

public import Definitions.Pouw.Eps1.M3
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
# Dither's CUDA-core lower bounds in model M3 (trusted)

The proved lower bounds over `Pouw.Eps1`'s model M3, each proved as `Pouw.SecurityProofs.Eps1.G` in
`Proofs.Pouw.Eps1`:

* `WordLocalZero`: Theorem B at the zero instance, a word-local program costs at least 4 instructions a word;
* `WordLocalGeneral`: Theorem B at a general instance, at least 8 a word (met by dp4a's 8 IDP.4A);
* `ClassPSlots`: Theorem C's count over issue slots, with each writer's creations `w` a parameter, at least
  `1 + (32 − w)/(65·d)` slots a word when a slot writes at most `d` words.

Theorem C's packed form, `(1 + 28/27)·L·m·n` instructions, is retired: its paper proof caps a writer's creations at 4,
and a writer can create 8. `ClassPSlots` takes the cap as its parameter `w`.

Free words (functions of one side alone) cost nothing, and the fold is not counted. M3 has no data-dependent
addressing and each instruction writes one 32-bit word from at most three, so these bound the gather-free `q*` in
M3; whether the ISA's instructions are such is not these statements' business.
-/

@[expose] public section

namespace Pouw.Guarantees.Eps1

open Pouw.Eps1

/-- **Theorem B, zero instance.** A word-local M3 program that computes Dither's prefix words `T_ℓ[i,j]` at the zero
instance has at least `4·L·m·n` instructions. -/
def WordLocalZero : Prop :=
  ∀ (m n Lseg : ℕ) (FR FU : Type) (G : ℕ) (P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (blk : Fin G → Tau m n Lseg),
    P.Computes (fun _ U => ColsValid U) zeroOut → P.WordLocal blk → 4 * (Lseg * m * n) ≤ G

/-- **Theorem B, general instance.** A word-local M3 program that computes the prefix words `T_ℓ[i,j]` of `Ã·B̃` on
every int8 instance has at least `8·L·m·n` instructions. -/
def WordLocalGeneral : Prop :=
  ∀ (m n Lseg : ℕ) (FR FU : Type) (G : ℕ) (P : M3Algorithm (Rows m Lseg) (GCols n Lseg) m n Lseg FR FU G)
    (blk : Fin G → Tau m n Lseg),
    P.Computes (fun A B => (∀ i l p, Int8Valid (A i l p)) ∧ ∀ j l p, Int8Valid (B j l p)) genOut →
    P.WordLocal blk → 8 * (Lseg * m * n) ≤ G

/-- **Theorem C's count over issue slots.** A class-P M3 program that computes Dither's prefix words at the zero
instance, whose output writers each create at most `w` incidences, run in `S` issue slots (`slot`) that each write at
most `d` words, a writer's slot writing nothing else, has `(32 − w)·L·m·n ≤ 65·d·(S − L·m·n)`. A tuple word creates at
most 65 incidences (two 32-entry fields and a one-entry field fill its `2^32` values), so a slot creates at most
`κ = 65·d` and a word costs at least `1 + (32 − w)/κ` slots: `1 + 24/130 ≈ 1.185` at `w = 8`, `d = 2`. -/
def ClassPSlots : Prop :=
  ∀ (m n Lseg : ℕ) (FR FU : Type) (G : ℕ) (P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)) (w d S : ℕ) (slot : Fin G → Fin S),
    P.Computes (fun _ U => ColsValid U) zeroOut → ClassP P pk →
    (∀ τ, created P pk (P.out τ) ≤ w) →
    (∀ τ t, slot t = slot (P.out τ) → t = P.out τ) →
    (∀ s, (Finset.univ.filter fun t => slot t = s).card ≤ d) →
    ((32 : ℝ) - w) * (Lseg * m * n) ≤ 65 * d * ((S : ℝ) - Lseg * m * n)

end Pouw.Guarantees.Eps1
