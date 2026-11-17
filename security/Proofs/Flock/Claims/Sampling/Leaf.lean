module

public import Proofs.Flock.Claims.Sampling.Path
public import Proofs.Flock.Soundness.Binding.Leaf
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
import Std.Tactic.BVDecide.Normalize.Prop

/-! The hm96 leaf (hm96-sha512/row/v1) binds at every `H`, at a row that places the value. -/

@[expose] public section

namespace Claims.Sampling

open FlockSoundness.Refine FlockSoundness.Binding

def hm96Outer (H : Msg → Msg) (row salt : Opening Bool → Msg) (o : Opening Bool) : Msg :=
  K512.leafPrefix ++ (FlockSoundness.Merkle.xorBytes (H (row o)) (K512.mask (salt o)) ++ H (K512.saltPrefix ++ salt o))

/-- **hm96's leaf as an oracle program**: the row's digest, the salt's, then the outer hash, refusing digests that
aren't 64 bytes. -/
def hm96Leaf (row salt : Opening Bool → Msg) (o : Opening Bool) : OC (Option Msg) :=
  .query (row o) fun x =>
    .query (K512.saltPrefix ++ salt o) fun c =>
      .query (K512.leafPrefix ++ (FlockSoundness.Merkle.xorBytes x (K512.mask (salt o)) ++ c)) fun l =>
        .ret (if x.length = 64 ∧ c.length = 64 then some l else none)

def RowPlaces (row : Opening Bool → Msg) : Prop :=
  ∀ o o' : Opening Bool, o.pos = o'.pos → o.val ≠ o'.val → row o ≠ row o'

theorem hm96Leaf_fst (H : Msg → Msg) (row salt : Opening Bool → Msg) (o : Opening Bool) :
    ((hm96Leaf row salt o).eval H).1 =
      if (H (row o)).length = 64 ∧ (H (K512.saltPrefix ++ salt o)).length = 64 then some (H (hm96Outer H row salt o))
      else none :=
  rfl

theorem hm96Leaf_log (H : Msg → Msg) (row salt : Opening Bool → Msg) (o : Opening Bool) :
    ((hm96Leaf row salt o).eval H).2 =
      [(row o, H (row o)), (K512.saltPrefix ++ salt o, H (K512.saltPrefix ++ salt o)),
        (hm96Outer H row salt o, H (hm96Outer H row salt o))] :=
  rfl

theorem hm96Leaf_binds (H : Msg → Msg) {row : Opening Bool → Msg} (hrow : RowPlaces row) (salt : Opening Bool → Msg) :
    LeafBinds H (hm96Leaf row salt) := by
  intro o o' l hpos hne h h'
  rw [hm96Leaf_fst] at h h'
  split_ifs at h with hg
  split_ifs at h' with hg'
  obtain ⟨hx, hc⟩ := hg
  obtain ⟨hx', hc'⟩ := hg'
  rw [hm96Leaf_log, hm96Leaf_log]
  by_cases hq : hm96Outer H row salt o = hm96Outer H row salt o'
  · have hcs := List.append_cancel_left hq
    have hm : ∀ y, (K512.mask y).length = 64 := K512.mask_len
    have hlen : (FlockSoundness.Merkle.xorBytes (H (row o)) (K512.mask (salt o))).length =
        (FlockSoundness.Merkle.xorBytes (H (row o')) (K512.mask (salt o'))).length := by
      simp only [FlockSoundness.Merkle.xorBytes, List.length_zipWith, hx, hx', hm]
    obtain ⟨hb, hcc⟩ := List.append_inj hcs hlen
    by_cases hy : salt o = salt o'
    · rw [hy] at hb
      have hxx := FlockSoundness.Merkle.xorBytes_inj (hx.trans (hm _).symm) (hx'.trans (hm _).symm) hb
      exact ⟨(row o, H (row o)), by simp, (row o', H (row o')), by simp, hrow o o' hpos hne, hxx⟩
    · exact ⟨(K512.saltPrefix ++ salt o, _), by simp, (K512.saltPrefix ++ salt o', _), by simp,
        fun e => hy (List.append_cancel_left e), hcc⟩
  · exact ⟨(hm96Outer H row salt o, _), by simp, (hm96Outer H row salt o', _), by simp, hq,
      (Option.some.inj h).trans (Option.some.inj h').symm⟩

theorem hm96Leaf_sha512 (row salt : Opening Bool → Msg) (o : Opening Bool) :
    ((hm96Leaf row salt o).eval sha512Msg).1 = some (digBytes (H512 (hm96Input (row o) (salt o)))) := by
  have e : ∀ m, sha512Msg m = digBytes (H512 m) := fun m => by
    unfold sha512Msg bytes
    rw [digBytes, dig_eq, H512_val]
    rfl
  have hlen : ∀ m, (sha512Msg m).length = 64 := fun m => by
    rw [e]
    exact hL512 _
  rw [hm96Leaf_fst, ite_eq_left ⟨hlen _, hlen _⟩]
  simp only [hm96Outer, e, hm96Input, commitString]

theorem Binding_of_hm96 {Coin Col T : Type} (D : Deploy Coin Col T) (own : Array _root_.Flock.Registered.Port)
    (at_ : List Msg → ℕ → Option (Msg × ℕ)) (row salt π : Opening Bool → Msg)
    (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hD : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row) : Binding D :=
  Binding_of_regPath D own at_ (hm96Leaf row salt) π layer hD fun H => hm96Leaf_binds H hrow salt

end Claims.Sampling
