module

public import Proofs.Flock.Claims.Steps.FlockKSDeployK
public import Proofs.Flock.Claims.DeployVK
public import Proofs.Flock.Claims.KS.ConflictWithin
public import Proofs.Flock.Claims.Recursion.ConflictTiedE
public import Proofs.Flock.Claims.Recursion.FlockKSRecord
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.KS.ConflictRun
import Proofs.Flock.Claims.KS.Width
import Proofs.Flock.Claims.Recursion.FlockKSDeploy

/-!
# The code facts and the accepted run's decision at `FlockDeploy.VK keep`

`FlockKSDeployE`'s code facts and `ConflictTiedE`'s accepted-run decision, at every `keep`, by the same proofs at
`nextK keep` (`nextK_decide`): `VK keep` shows each round's coin (`coinsShown_VK`), accepts only at its last round
(`accLast_VK`, `streams_size_plainK` holding at every `keep`), decides without the last coin (`decideFree_VK`) and never
accepts the empty message (`refusable_VK`); an accepted run of it decides within its own verifier log (`accK_decide`),
so its verdict accepts with SHA-512 answered only within that log (`decisionK_runs`, `verdictK_within`).

At `keep := fun _ => true` these are the facts at `VE` (`VK_true`), and at `keep := id` the ones at `V false`
(`VK_id`), so a proof that reads them at `VK keep` holds at both.
-/

@[expose] public section

namespace FlockDeploy

open Flock

open Claims.Recursion

/-- **Where `VK keep` answers, the deployed plain verifier answers alike.** -/
theorem next_msg_of_nextK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {m : Claims.Msg} (h : ((nextK keep x as cs).eval H).1 = .msg m) :
    ((next false x as cs).eval H).1 = .msg m := by
  cases ha : act false x as cs with
  | rej => rw [nextK_rej keep H ha] at h; cases h
  | coin c => rw [nextK_coin keep H ha] at h; rw [next_coin H ha]; exact h
  | decide s spec sts link c₁ mids f =>
    simp only [nextK, ha, outOf, Claims.OC.eval_bind, Claims.OC.eval] at h
    split at h <;> simp at h

theorem refusable_VK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) : Refusable H (VK keep) := fun x =>
  ⟨[], fun as cs => by
    show ((nextK keep x (as ++ [[]]) cs).eval H).1 ≠ .acc
    rw [nextK_rej keep H (act_snoc_nil false x as cs)]
    exact fun h => Claims.Out.noConfusion h⟩

theorem coinsShown_VK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) : CoinsShown H (VK keep) decCoin :=
  fun x as cs c m hl h => coinsShown_V H false x as cs c m hl (next_msg_of_nextK (cs := cs ++ [c]) keep H h)

theorem decideFree_VK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) : DecideFree H (VK keep) := by
  intro x as cs c c' hl hacc
  show (nextK keep x as (cs ++ [c'])).eval H = (nextK keep x as (cs ++ [c])).eval H
  change ((nextK keep x as (cs ++ [c])).eval H).1 = .acc at hacc
  cases hact : act false x as (cs ++ [c]) with
  | rej =>
    rw [nextK_rej keep H hact] at hacc
    cases hacc
  | coin c₂ =>
    rw [nextK_coin keep H hact] at hacc
    cases hacc
  | decide s spec sts link c₁ mids f =>
    have hlen := act_decide_length (by simp [hl]) hact
    obtain ⟨c₀, cs', rfl⟩ : ∃ c₀ cs', cs = c₀ :: cs' := by
      rcases cs with _ | ⟨c₀, cs'⟩
      · simp only [List.length_nil, zero_add] at hl
        omega
      · exact ⟨c₀, cs', rfl⟩
    have ha : ∀ t, as.getLast? ≠ some (0 :: t) := fun t h => by
      rcases act_snoc_zero (zk := false) (x := x) (cs := c₀ :: cs') (c := c) h with h' | h' <;>
        rw [hact] at h' <;> cases h'
    unfold nextK
    rw [act_cons_snoc false x as c₀ cs' c c' ha]

end FlockDeploy

namespace FlockDeploy.KS

open Flock

open Claims (OC OCV)

end FlockDeploy.KS
