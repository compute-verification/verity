module

public import Proofs.Flock.Soundness.Discharge.ZkView.Leaf
public import Proofs.Flock.Soundness.Discharge.ZkView.Statements
import ArkLib.Data.CodingTheory.JohnsonBound.Family
import ArkLib.Data.CodingTheory.ReedSolomon.MutualCorrelatedAgreement.Johnson.Probability
import Batteries.Data.UInt
import Flock.Sizes
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Algebra.Polynomial.SpecificDegree
import Mathlib.Data.Nat.BitIndices
import Mathlib.Data.Nat.Size
import Mathlib.FieldTheory.Finite.Extension
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.RingTheory.Henselian
import Mathlib.RingTheory.RegularLocalRing.Defs
import Mathlib.RingTheory.SimpleRing.Principal
import Mathlib.Tactic
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
import Proofs.Flock.Level3
import Proofs.Flock.Soundness.Accounting.Fast100
import Proofs.Flock.Soundness.Accounting.Numbers
import Proofs.Flock.Soundness.Audit.Flock
import Proofs.Flock.Soundness.Binding.E2E
import Proofs.Flock.Soundness.Binding.Weights
import Proofs.Flock.Soundness.CROnly
import Proofs.Flock.Soundness.CROnly.ZK
import Proofs.Flock.Soundness.CompiledAlign
import Proofs.Flock.Soundness.Defs
import Proofs.Flock.Soundness.Discharge.Hm.Commit
import Proofs.Flock.Soundness.Discharge.Hm.Slot
import Proofs.Flock.Soundness.Extract
import Proofs.Flock.Soundness.Game.Basic
import Proofs.Flock.Soundness.Game.Batch
import Proofs.Flock.Soundness.Game.Interleave
import Proofs.Flock.Soundness.Game.Lock
import Proofs.Flock.Soundness.Game.Poly
import Proofs.Flock.Soundness.Game.Seq
import Proofs.Flock.Soundness.Game.Union
import Proofs.Flock.Soundness.Knowledge
import Proofs.Flock.Soundness.LigeritoCoins
import Proofs.Flock.Soundness.LigeritoWeights
import Proofs.Flock.Soundness.LincheckPhase
import Proofs.Flock.Soundness.Merkle
import Proofs.Flock.Soundness.Model.Compiled
import Proofs.Flock.Soundness.OpeningPhase
import Proofs.Flock.Soundness.Refine.Arith
import Proofs.Flock.Soundness.Refine.Basis
import Proofs.Flock.Soundness.Refine.Final
import Proofs.Flock.Soundness.Refine.Lincheck
import Proofs.Flock.Soundness.Refine.Opening
import Proofs.Flock.Soundness.Refine.Plays
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Refine.Zerocheck
import Proofs.Flock.Soundness.Registered.Binding
import Proofs.Flock.Soundness.Registered.E2E
import Proofs.Flock.Soundness.RepSound
import Proofs.Flock.Soundness.Rewinding
import Proofs.Flock.Soundness.Rounds
import Proofs.Flock.Soundness.SessionAcc
import Proofs.Flock.Soundness.Soundness
import Proofs.Flock.Soundness.SparseRows
import Proofs.Flock.Soundness.TmplFoldDef
import Proofs.Flock.Soundness.TmplFoldLayout
import Proofs.Flock.Soundness.TmplFoldPos
import Proofs.Flock.Soundness.TmplFoldRef
import Proofs.Flock.Soundness.TmplLayOk
import Proofs.Flock.Soundness.TmplSplitEq
import Proofs.Flock.Soundness.TmplSplitWalk
import Proofs.Flock.Soundness.TmplUnit
import Proofs.Flock.Soundness.TmplUnrollAt
import Proofs.Flock.Soundness.TmplWf
import Proofs.Flock.Soundness.Types.ProgramE2E
import Proofs.Flock.Soundness.ZK.Blocks
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.RealLeaves
import Proofs.Flock.Soundness.ZK.Refusal
import Proofs.Flock.Soundness.ZK.SHVZK
import Proofs.Flock.Verifier.Transcript
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas
import VCVio.Prelude.Core

/-!
# Lemma C at the `--zk` wire

`CROnly.adaptive_prefinal_gap` at the `--zk` tables (`Coined.tab`), the honest prover's messages `msgsAt` and the leaves'
structure `H(x ⊕ M·y, c(y))` (`leafSplit`, through `HonestMsg`):

* `hP` holds by definition: `pairProver` is `msgsAt` at the leaves `H(dig + B, c)`.
* `hI` holds by definition: at ideal leaves `H(U, c(y))`, `Coined.preWith` is `Table.preView` at the randomness
  `Coined.randOf` builds from the base and the leaves' seeds (`Coined.preWith_ideal`), since `Table.preView` reads the
  seeds only in its leaves.
* `he` with `k = 1`: `randAt T c` is a bijection (`randAt_bijective`), each table's randomness being its base and its
  pre-final leaves' seeds placed by `T.s0`, `T.sP` (`Coined.unRand` reads them back).
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkView

open Flock ZK
open FlockSoundness.Game (prCoin)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

variable {K W Hi VL : Type} [Field K] {z : Sizes}

/-- At ideal leaves `H(U, c(y))`, the pre-final view with its leaves given by position is `Table.preView`. -/
theorem Coined.preWith_ideal (T : Coined K W Hi VL z) (w : W) (b : Base z Hi) (u : LeafPos z → Bits × Salt) :
    T.preWith w b (fun p => leafH ((u p).1, saltHash (u p).2)) = T.tab.preView w (T.randOf b u) := rfl

/-- The base and each pre-final leaf's seed by position, read back from a table's randomness. -/
def Coined.unRand (T : Coined K W Hi VL z) (ω : T.tab.Rand) : Base z Hi × (LeafPos z → Bits × Salt) :=
  let s := ω.2.2.2.2.2.2
  ((ω.1, ω.2.1, ω.2.2.1, ω.2.2.2.1, ω.2.2.2.2.1, ω.2.2.2.2.2.1, s.1, s.2.2.2.1, s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2),
    Sum.elim (fun p => Sum.elim s.2.1 s.2.2.1 (T.s0.symm p))
      (Sum.elim (fun rp => Sum.elim (s.2.2.2.2.1 rp.1) (s.2.2.2.2.2.1 rp.1) ((T.sP rp.1).symm rp.2)) s.2.2.2.2.2.2.2.1))

theorem Coined.unRand_randOf (T : Coined K W Hi VL z) (b : Base z Hi) (u : LeafPos z → Bits × Salt) :
    T.unRand (T.randOf b u) = (b, u) := by
  refine Prod.ext rfl (funext fun p => ?_)
  rcases p with p | ⟨r, p⟩ | r
  · obtain ⟨q, rfl⟩ := T.s0.surjective p
    change Sum.elim _ _ (T.s0.symm (T.s0 q)) = _
    rw [Equiv.symm_apply_apply]
    rcases q with q | q <;> rfl
  · obtain ⟨q, rfl⟩ := (T.sP r).surjective p
    change Sum.elim _ _ ((T.sP r).symm ((T.sP r) q)) = _
    rw [Equiv.symm_apply_apply]
    rcases q with q | q <;> rfl
  · rfl

theorem Coined.randOf_injective (T : Coined K W Hi VL z) :
    Function.Injective fun x : Base z Hi × (LeafPos z → Bits × Salt) => T.randOf x.1 x.2 :=
  Function.LeftInverse.injective (g := T.unRand) fun x => T.unRand_randOf x.1 x.2

theorem Coined.randOf_surjective (T : Coined K W Hi VL z) :
    Function.Surjective fun x : Base z Hi × (LeafPos z → Bits × Salt) => T.randOf x.1 x.2 := by
  rintro ⟨m, ⟨μ0, R, μR⟩, h, padh, padμ, μ, s1, s2, s3, s4, s5, s6, s7, s8, s9⟩
  refine ⟨((m, (μ0, R, μR), h, padh, padμ, μ, s1, s4, s7, s9), fun
    | .inl p => Sum.elim s2 s3 (T.s0.symm p)
    | .inr (.inl (r, p)) => Sum.elim (s5 r) (s6 r) ((T.sP r).symm p)
    | .inr (.inr r) => s8 r), ?_⟩
  simp only [Coined.randOf, Equiv.symm_apply_apply]
  rfl

variable {J n : ℕ} {Cn : Type} {zs : Fin J → Sizes}

/-- **The session's tables' randomness at coins `c`** (Lemma C's `e c`): each table's base, and its pre-final leaves'
seeds by position. -/
noncomputable def randAt (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j)) (c : Fin n → Cn)
    (x : (∀ j, Base (zs j) Hi) × (Leaves zs → Bits × Salt)) : ∀ j, (T c j).tab.Rand :=
  fun j => (T c j).randOf (x.1 j) fun p => x.2 ⟨j, p⟩

theorem randAt_bijective (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j)) (c : Fin n → Cn) :
    Function.Bijective (randAt T c) := by
  constructor
  · intro x x' h
    have hj : ∀ j, ((x.1 j, fun p => x.2 ⟨j, p⟩) : Base (zs j) Hi × (LeafPos (zs j) → Bits × Salt))
        = (x'.1 j, fun p => x'.2 ⟨j, p⟩) := fun j =>
      (T c j).randOf_injective (a₁ := (x.1 j, fun p => x.2 ⟨j, p⟩)) (a₂ := (x'.1 j, fun p => x'.2 ⟨j, p⟩))
        (congrFun h j)
    refine Prod.ext (funext fun j => (Prod.ext_iff.1 (hj j)).1) (funext fun l => ?_)
    exact congrFun (Prod.ext_iff.1 (hj l.1)).2 l.2
  · intro r
    choose x hx using fun j => (T c j).randOf_surjective (r j)
    exact ⟨(fun j => (x j).1, fun l => (x l.1).2 l.2), funext fun j => hx j⟩

theorem card_fiber_of_bijective {α β : Type} {f : α → β} (hf : Function.Bijective f) (b : β) :
    Nat.card {x // f x = b} = 1 :=
  Nat.card_eq_one_iff_unique.2
    ⟨⟨fun x y => Subtype.ext (hf.1 (x.2.trans y.2.symm))⟩, ⟨⟨_, (hf.2 b).choose_spec⟩⟩⟩

end FlockSoundness.Discharge.ZkView
