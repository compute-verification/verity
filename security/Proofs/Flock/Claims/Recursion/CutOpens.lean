module

public import Proofs.Flock.Claims.Recursion.DeployedR
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

/-! The layout-free half of `OpensSlots`: `finalR`'s cut over openings laid row then 192-byte salt, and 128-byte commit
strings laid end to end. -/

@[expose] public section

namespace RecDeploy.CutOpens

open Claims (Msg)

theorem saltBytes_eq : Flock.Hm96.sha512.saltBytes = 192 := rfl

theorem cutRound_pairs (rest : Msg) : ∀ l : List (Msg × Msg), (∀ p ∈ l, p.2.length = 192) →
    DeployedR.cutRound (l.map (·.1.length)) (l.flatMap (fun p => p.1 ++ p.2) ++ rest) =
      some (l.map fun p => (p.1, DeployedR.saltOf p.2), rest)
  | [], _ => rfl
  | p :: l, h => by
    have h2 : p.2.length = 192 := h p List.mem_cons_self
    have ih := cutRound_pairs rest l fun q hq => h q (List.mem_cons_of_mem _ hq)
    simp only [List.map_cons, List.flatMap_cons, DeployedR.cutRound, List.append_assoc, saltBytes_eq]
    rw [List.drop_left, List.take_left, List.drop_left' h2, List.take_left' h2, ih]
    have hc : p.1.length + 192 ≤ (p.1 ++ (p.2 ++ (l.flatMap (fun p => p.1 ++ p.2) ++ rest))).length := by
      simp only [List.length_append, h2]
      omega
    simp only [hc, ↓reduceIte]
    rfl

theorem cutRounds_groups (rest : Msg) : ∀ L : List (List (Msg × Msg)), (∀ l ∈ L, ∀ p ∈ l, p.2.length = 192) →
    DeployedR.cutRounds (L.map fun l => l.map (·.1.length)) (L.flatMap (·.flatMap fun p => p.1 ++ p.2) ++ rest) =
      some (L.map fun l => l.map fun p => (p.1, DeployedR.saltOf p.2), rest)
  | [], _ => rfl
  | l :: L, h => by
    simp only [List.map_cons, List.flatMap_cons, DeployedR.cutRounds, List.append_assoc,
      cutRound_pairs (L.flatMap (·.flatMap fun p => p.1 ++ p.2) ++ rest) l (h l List.mem_cons_self), Option.bind_some,
      cutRounds_groups rest L fun l' hl' => h l' (List.mem_cons_of_mem _ hl'), Option.map_some]

theorem commitAt_flatten : ∀ (l : List Msg), (∀ m ∈ l, m.length = 128) → ∀ i (hi : i < l.length),
    Claims.Recursion.VStarB.commitAt l.flatten i = l[i]
  | [], _, _, hi => absurd hi (by simp)
  | m :: l, h, 0, _ => by
    have hm : m.length = 128 := h m List.mem_cons_self
    simp only [Claims.Recursion.VStarB.commitAt, List.flatten_cons, Nat.mul_zero, List.drop_zero,
      List.getElem_cons_zero]
    exact List.take_left' hm
  | m :: l, h, i + 1, hi => by
    have hm : m.length = 128 := h m List.mem_cons_self
    have ih := commitAt_flatten l (fun m' hm' => h m' (List.mem_cons_of_mem _ hm')) i
      (by simpa using hi)
    simp only [Claims.Recursion.VStarB.commitAt, List.flatten_cons, List.getElem_cons_succ] at ih ⊢
    rw [show 128 * (i + 1) = m.length + 128 * i by omega, ← List.drop_drop, List.drop_left]
    exact ih

end RecDeploy.CutOpens
