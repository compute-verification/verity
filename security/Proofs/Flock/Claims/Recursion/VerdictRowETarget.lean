module

public import Proofs.Flock.Claims.Recursion.NonVacuous
public import Proofs.Flock.Claims.Recursion.VerdictRowEDeployed
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
import Proofs.Flock.Claims.Recursion.VerdictLoop
import Proofs.Flock.Claims.Recursion.VerdictRowG

/-! Row E's target at the decision (`targetE`), at the accumulators `accR` read off the decision, which is unique. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP pointsOf)
open FlockVBridge.AccChain (UnitSums ReadsEnds)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt nOpen accHolds)

namespace Rows.DecAt

variable {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}

theorem eq (D D' : DecAt rd x ms cs) : D = D' := by
  obtain ⟨d, hd, spec, sts, hs, sess, hdec⟩ := D
  obtain ⟨d', hd', spec', sts', hs', sess', hdec'⟩ := D'
  obtain rfl : d = d' := Option.some.inj (hd.symm.trans hd')
  have e : (spec, sts) = (spec', sts') := Except.ok.inj (hs.symm.trans hs')
  simp only [Prod.mk.injEq] at e
  obtain ⟨rfl, rfl⟩ := e
  obtain rfl : sess = sess' := Except.ok.inj (hdec.symm.trans hdec')
  rfl

end Rows.DecAt

open Classical in
noncomputable def accR (rd : Msg → Option FinalP) (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (x : Msg)
    (ms : List Msg) (cs : List Coin) (k : ℕ) : Array F128 :=
  if h : Nonempty (DecAt rd x ms cs) then (Classical.choice h).accE chainAt k else #[0, 0, 0, 0]

theorem accR_eq {rd : Msg → Option FinalP} (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) {x : Msg}
    {ms : List Msg} {cs : List Coin} (D : DecAt rd x ms cs) : accR rd chainAt x ms cs = D.accE chainAt := by
  funext k
  unfold accR
  rw [dif_pos ⟨D⟩, Rows.DecAt.eq (Classical.choice ⟨D⟩) D]

theorem accR_size (rd : Msg → Option FinalP) (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (x : Msg)
    (ms : List Msg) (cs : List Coin) (k : ℕ) : (accR rd chainAt x ms cs k).size = 4 := by
  unfold accR
  split
  · exact Rows.DecAt.accE_size _ chainAt k
  · rfl

namespace Top

theorem targetE (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop)
    (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (pieces : Msg → List OpenPiece)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (coefAt : Msg → List Coin → OpenPiece → Array F128)
    (hholds : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), P x ms cs →
      ∀ q ∈ pieces x, accHolds sideAt coefAt chainAt q x ms cs)
    (hside : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size → ∀ r : Fin 2,
        ∀ q ∈ pieces x, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hcoef : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size → ∀ r : Fin 2,
        ∀ q ∈ pieces x, q.table = i → q.rep = r.val →
          coefAt x cs q = D.coefAt cls (accR rd chainAt x ms cs) i r q.level q.query)
    (hcover : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size → ∀ r : Fin 2,
        ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
          ∃ q ∈ pieces x, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j) :
    TargetE rd P (accR rd chainAt) := by
  intro x ms cs hA D cls i hi
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, hr, hb, hP⟩ := id hA
  have hco := hcoef x ms cs hA D cls i hi
  rw [accR_eq chainAt D] at hco ⊢
  have hG := D.rowG cls (D.accE chainAt) hst hf hx hi
  refine ⟨fun _ => chainAt x ms i, fun r => r * nOpen (D.files cls i).sch, fun r => ?_, fun r => ?_⟩
  · exact D.unitSums_of_pieces cls (D.accE chainAt) i r (pieces x) sideAt coefAt chainAt (hholds x ms cs hP)
      (hside x ms cs hA D cls i hi r) (hco r) (hcover x ms cs hA D cls i hi r)
  · exact D.readsEnds_accE cls (D.accE chainAt) i r chainAt hG D.points_eq (idx_lt D hi r)
      (shaped_of_base hr hb D _ (idx_lt D hi r))

end Top

end RecDeploy
