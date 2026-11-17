module

public import Definitions.Pouw.PearlC.Sha512
import all Init.Data.List.Basic
import all Init.Data.String.Defs
public import Proofs.Hash.HistoryTable
public import Proofs.Hash.Witnesses
public import Mathlib.Tactic.Basic
import Std.Tactic.BVDecide.Normalize.Prop
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

/-! The honest and nontrivial witnesses of `Hash.SecurityProofs.HistoryReadBinding`, under the spec's SHA-512. -/

@[expose] public section

namespace Hash.SecurityProofs

open Hash

local notation "SHA512" => fun (x : Hash.Msg) (i : Fin 64) =>
  Nat.toUInt8 (List.getD (Pouw.PearlC.Sha512.digest (List.map UInt8.toNat x)) (Fin.val i) 0)

local notation "key₀" => List.replicate 64 (0 : UInt8)

local notation "D₀" => Tree.Domain.mk (fun _ => 4) (2 ^ 17) (fun r => r)

local notation "row₀" => List.replicate 16 true
local notation "salt₀" => List.replicate 192 (2 : UInt8)

local notation "s₀" => HistoryTable.string 1
  (fun _ => Hm96.commitString SHA512 key₀ (RowLeaf.Format.seg.digest SHA512 row₀) salt₀)
  (HistoryTable.placeholder SHA512 SHA512 key₀ 1)

/-- **`HistoryReadBinding`'s honest witness**: in a table of `2^17` ranks holding one row, the read at rank 0 of the
row behind its string, under its salt and with the tree's own path, returns ok. -/
theorem HistoryReadBinding_honest :
    EmbeddingRead.ok SHA512 SHA512 D₀ key₀ (HistoryTable.root SHA512 D₀ key₀ s₀) (D₀).count 0
        (HistoryTable.opening SHA512 D₀ key₀ s₀ row₀ salt₀ 0) ∧
      Hm96.commitString SHA512 key₀ (RowLeaf.Format.seg.digest SHA512 row₀) salt₀ = s₀ 0 := by
  have hs : Hm96.commitString SHA512 key₀ (RowLeaf.Format.seg.digest SHA512 row₀) salt₀ = s₀ 0 := by
    rfl
  have hrow : (row₀).length < 2 ^ 64 := by
    rw [List.length_replicate]
    decide
  have hsalt : (salt₀).length = Hm96.SALT_BYTES := by
    simp only [List.length_replicate]
    rfl
  exact ⟨Or.inr (HistoryTable.accepts_opening SHA512 SHA512 D₀ key₀ s₀ row₀ salt₀ 0 (by decide) hrow hsalt hs), hs⟩

local notation "D₁" => Tree.Domain.mk (fun _ => 4) 1 (fun r => r)

local notation "o₁" => RowLeaf.Opening.mk [false] [] []

local notation "op₁" => HistoryTable.opening SHA512 D₁ [] (fun _ => []) [true] [0] 0

/-- **`HistoryReadBinding`'s nontrivial witness**: with every read ok, the row `[false]` read at rank 0 is not the row
`[true]` behind that rank's string, and the messages the two openings hashed have pairwise different SHA-512 digests,
under `H` and under `G`. -/
theorem HistoryReadBinding_nontrivial :
    ¬ (EmbeddingRead.h (D₁).count 1 0 o₁ = [true] ∨
      CollisionIn SHA512 (RowLeaf.outerHashed SHA512 SHA512 (EmbeddingRead.verifier D₁ []) 0 o₁ ++
        RowLeaf.outerHashed SHA512 SHA512 (EmbeddingRead.verifier D₁ []) 0 op₁) ∨
      CollisionIn SHA512 (RowLeaf.innerHashed SHA512 (EmbeddingRead.verifier D₁ []) o₁ ++
        RowLeaf.innerHashed SHA512 (EmbeddingRead.verifier D₁ []) op₁)) := by
  rintro (h | h | h)
  · exact absurd h (by decide)
  · refine not_collisionIn (List.Subset.refl _) ?_ h
    simp only [RowLeaf.outerHashed, RowLeaf.leafMsg, Hm96.hashed, Hm96.leaf, Hm96.commitString,
      HistoryTable.opening, EmbeddingRead.verifier, mask_zero [] [] (by simp), mask_zero [] [0] (by simp)]
    decide +kernel
  · refine not_collisionIn (List.Subset.refl _) ?_ h
    decide +kernel

end Hash.SecurityProofs
