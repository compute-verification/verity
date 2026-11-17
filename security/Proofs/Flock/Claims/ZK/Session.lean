module

public import Proofs.Flock.Claims.ZK.Capacity
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

/-! The session's premises at the deployed masked record, round by round. -/

@[expose] public section

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

def RefSession.HonestPreOn (S : RefSession) (A : Msg → Prop) (R : Rel Flock.Wit) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x, A x → ∀ w, R x w → ∀ ν t (ops : List (Msg × Msg)), ops.length < refK x →
    S.enc x (S.pad (ops.map fun p => S.decOf p.2)) ops.length
        (S.cutOf x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w) t) =
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

def RefSession.HonestFinOn (S : RefSession) (A : Msg → Prop) (R : Rel Flock.Wit) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x, A x → ∀ w, R x w → ∀ ν t (ops : List (Msg × Msg)), ops.length = refK x →
    S.Admits x (S.pad (ops.map fun p => S.decOf p.2)) →
    S.fin x (S.pad (ops.map fun p => S.decOf p.2))
        (S.viewsAt x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w) t) =
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

def RefSession.HonestOffOn (S : RefSession) (A : Msg → Prop) (R : Rel Flock.Wit) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x, A x → ∀ w, R x w → ∀ ν t (ops : List (Msg × Msg)), ops.length = refK x →
    ¬ S.Admits x (S.pad (ops.map fun p => S.decOf p.2)) →
    S.off x (S.pad (ops.map fun p => S.decOf p.2))
        (S.cutOf x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w) t) =
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

theorem RefSession.honestMsgsOn_of (S : RefSession) (A : Msg → Prop) (R : Rel Flock.Wit) (H : Msg → Msg)
    (coinOf : Msg → Msg) (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk)
    (hpre : S.HonestPreOn A R H coinOf tapeOf) (hfin : S.HonestFinOn A R H coinOf tapeOf)
    (hoff : S.HonestOffOn A R H coinOf tapeOf) : S.HonestMsgsOn A R H coinOf tapeOf := by
  intro x hx w hR ν t ops hops
  have e : (ops.map fun p => S.decOf p.2).length = ops.length := by simp
  unfold RefSession.msgAt
  split_ifs with h1 h2
  · rw [e]
    exact hpre x hx w hR ν t ops (by simpa using h1)
  · exact hfin x hx w hR ν t ops (by rw [e] at h1; omega) h2
  · exact hoff x hx w hR ν t ops (by rw [e] at h1; omega) h2

theorem Flock.zk_deployZk_ref_on_rounds (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ) (T : TableData)
    (A : Msg → Prop) [DecidablePred A] (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : (session nr T).Tables)
    (hV : (session nr T).Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hD : (session nr T).DummyReg) (hc : (session nr T).Causal) (hT : ∀ x, A x → FibersEq (T.tape x))
    (hpre : (session nr T).HonestPreOn A (Flock.R (FlockDeploy.deployZk rel).parse) H coinOfD fun ω => T.tapeZk ω.2)
    (hfin : (session nr T).HonestFinOn A (Flock.R (FlockDeploy.deployZk rel).parse) H coinOfD fun ω => T.tapeZk ω.2)
    (hoff : (session nr T).HonestOffOn A (Flock.R (FlockDeploy.deployZk rel).parse) H coinOfD fun ω => T.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed rel nr T t)
      (fun x => if A x then 8 * (session nr T).nLeaves x * δ₁.toReal +
        8 * t * (session nr T).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_ref_on H rel nr T A t ht hT1 hδ₁ hS hV hD hc hT
    ((session nr T).honestMsgsOn_of A _ H coinOfD _ hpre hfin hoff)

end Claims.ZKMal
