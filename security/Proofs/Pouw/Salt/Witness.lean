module

public import Definitions.Pouw.PearlC.Sha512
import all Init.Data.List.Basic
import all Init.Data.String.Defs
public import Definitions.Pouw.Assumptions.LeadCommit
public import Proofs.Hash.Witnesses
public import Mathlib.Tactic.Basic
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
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
import Mathlib.RingTheory.WittVector.IsPoly
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# `keyed_leaf_binds` says something

The honest and nontrivial witnesses (the lock's `witnesses`) of `Pouw.Salt.keyed_leaf_binds`, whose statement reads the
keyed verifier's acceptance (`Hash.Keyed.accepts`), at one instance under the spec's SHA-512
(`Pouw.PearlC.Sha512.digest`): a window key of 64 bytes, a live salt, a key leaf over 192 salt bytes, and one instance,
a two-leaf frame-v3 subtree whose public digest is the keyed hash of its root node.

* `keyed_leaf_binds_honest`: when a sampled proof block's proofs open exactly the honest committer's values (rank 0's and rank
  1's, each with its sibling, under the window's key and key-leaf salt), both premises hold (`SubtreeLayout`,
  `SaltDerivation`), and the keyed verifier accepts the honest opening at rank 0.
* `keyed_leaf_binds_nontrivial`: when every opening is accepted (the premises dropped), two different values at rank 0
  of one instance, under one key and with no siblings, are opened, and none of the messages those openings hashed
  collide under SHA-512, so `Binds`' conclusion fails there.
-/

@[expose] public section

namespace Pouw.Salt

open Hash

/-- The spec's SHA-512 (`Pouw.PearlC.Sha512.digest`) on bytes, its digest's 64 bytes. -/
local notation "SHA512" => fun (x : Hash.Msg) (i : Fin 64) =>
  Nat.toUInt8 (List.getD (Pouw.PearlC.Sha512.digest (List.map UInt8.toNat x)) (Fin.val i) 0)

/-- The window's key. -/
local notation "ks₀" => (fun _ => 1 : Digest)

/-- The window's live salt. -/
local notation "salt₀" => (fun _ => 3 : Digest)

/-- The verifier's hm96 key. -/
local notation "key₀" => List.replicate 64 (0 : UInt8)

/-- The key leaf's 192 salt bytes. -/
local notation "y₀" => List.replicate 192 (2 : UInt8)

/-- The instance's frame-v3 domain: two leaves, at positions 0 and 1. -/
local notation "D₀" => Tree.Domain.mk (fun _ => 4) 2 (fun r => r)

/-- The instance's two values' frame leaves, at ranks 0 and 1. -/
local notation "ℓ₀" => Tree.leafMsg D₀ 0 "row" [5]
local notation "ℓ₁" => Tree.leafMsg D₀ 1 "row" [6]

/-- The instance's public digest: the keyed hash of its root node over its two leaves. -/
local notation "root₀" =>
  (fun _ => Keyed.digest SHA512 ks₀ salt₀ (Keyed.Top.msg D₀ (Keyed.Top.node 0 (SHA512 ℓ₀) (SHA512 ℓ₁))) :
    ℕ → Digest)

/-- The honest committer's openings: instance 0's two values, each with its sibling, under the window's key and the
key leaf's salt. -/
local notation "Opens₀" => (fun pos rank v ks y sibs =>
  pos = 0 ∧ ks = ks₀ ∧ y = y₀ ∧
    (rank = 0 ∧ v = [5] ∧ sibs = [SHA512 ℓ₁] ∨ rank = 1 ∧ v = [6] ∧ sibs = [SHA512 ℓ₀]) :
  ℕ → ℕ → Msg → Digest → Msg → List Digest → Prop)

theorem depth_two : Tree.depth 2 = 1 := by
  have h : Nat.log2 1 = 0 := by
    have := (Nat.log2_lt (k := 1) one_ne_zero).2 (by norm_num)
    omega
  simp [Tree.depth, h]

/-- **`Binds`' honest witness**: the honest committer's openings meet `SubtreeLayout` and `SaltDerivation`, under the
key leaf over the window's key, and the keyed verifier accepts the honest opening at rank 0. -/
theorem keyed_leaf_binds_honest :
    Assumptions.SubtreeLayout SHA512 salt₀ (fun _ => D₀) "row" root₀ Opens₀ ∧
      Assumptions.SaltDerivation SHA512 key₀ (Hm96.leaf SHA512 key₀ ks₀ y₀) Opens₀ ∧
      Keyed.accepts SHA512 ks₀ salt₀ D₀ (root₀ 0) 0 ℓ₀ [SHA512 ℓ₁] := by
  refine ⟨?_, ?_, ?_⟩
  · rintro pos rank v ks y sibs ⟨rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩⟩
    · unfold Keyed.accepts
      exact ⟨by decide, by simp [depth_two], rfl⟩
    · unfold Keyed.accepts
      exact ⟨by decide, by simp [depth_two], rfl⟩
  · rintro pos rank v ks y sibs ⟨-, rfl, rfl, -⟩
    rfl
  · unfold Keyed.accepts
    exact ⟨by decide, by simp [depth_two], rfl⟩

/-- **`Binds`' nontrivial witness**: with every opening accepted, the values `[0]` and `[1]` opened at rank 0 of one
instance, under one key and key-leaf salt and with no siblings, are different, and the messages those openings hashed
(the key leaf's salt message and leaf message, and each value's keyed message) have pairwise different SHA-512
digests. -/
theorem keyed_leaf_binds_nontrivial :
    ¬ (([0] : Msg) = [1] ∨ CollisionIn SHA512 (Hm96.hashed SHA512 [] ks₀ [] ++ Hm96.hashed SHA512 [] ks₀ [] ++
        Keyed.hashed SHA512 ks₀ salt₀ D₀ 0 (Tree.leafMsg D₀ 0 "row" [0]) [] ++
        Keyed.hashed SHA512 ks₀ salt₀ D₀ 0 (Tree.leafMsg D₀ 0 "row" [1]) [])) := by
  rintro (h | h)
  · exact absurd h (by decide)
  refine Hash.SecurityProofs.not_collisionIn (ys := [Hm96.saltPrefix ++ [],
    Hm96.leafPrefix SHA512 [] ++ Hm96.commitString SHA512 [] ks₀ [],
    Keyed.msg ks₀ salt₀ (Tree.leafMsg D₀ 0 "row" [0]), Keyed.msg ks₀ salt₀ (Tree.leafMsg D₀ 0 "row" [1])]) ?_ ?_ h
  · intro z hz
    simp only [Hm96.hashed, Keyed.hashed, Keyed.climb, List.mem_append, List.mem_cons, List.nil_append,
      List.not_mem_nil, or_false] at hz ⊢
    tauto
  · simp only [Hm96.commitString, Hash.SecurityProofs.mask_zero [] [] (by simp)]
    decide +kernel

end Pouw.Salt
