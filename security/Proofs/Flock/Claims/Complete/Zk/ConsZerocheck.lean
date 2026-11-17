module

public import Proofs.Flock.Claims.Complete.Zk.PiopGo
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
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra

/-! The zerocheck's two constraints vanish at the masked prover's pads (`cEval_zero`, `finalAB_zero`). -/

@[expose] public section

namespace FlockComplete.Replay

open Flock FlockSoundness.Discharge.Composed

theorem zcZ_masks (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (cs : Zc.Challenges) (tr : Trace) :
    ZerocheckMasked.Masks lay h (Zc.proof m cs tr) (ZcZ.proof lay h m cs tr) (m - 6) where
  round1ab i hi := by
    simp only [ZcZ.proof, ZcZ.r1ab, Zc.proof]
    rw [getElem!_pos _ i (by simpa using hi)]
    simp
  round1c i hi := by
    simp only [ZcZ.proof, ZcZ.r1c, Zc.proof]
    rw [getElem!_pos _ i (by simpa using hi)]
    simp
  rounds j hj := by
    simp only [ZcZ.proof, Zc.proof]
    rw [pushes_get _ _ j hj, pushes_get _ _ j hj]
    rfl
  finalA := rfl
  finalB := rfl
  finalC := rfl

theorem cEval_zero (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (cs : Zc.Challenges) (tr : Trace) :
    ZerocheckMasked.evalL ((ZerocheckMasked.computedC lay (ZcZ.proof lay h m cs tr) cs.z).add
      (ZerocheckMasked.fcL lay (ZcZ.proof lay h m cs tr))) h = 0 := by
  rw [ZerocheckMasked.c_eval_eval lay h _ _ (m - 6) (zcZ_masks lay h m cs tr) cs.z]
  exact ZerocheckMasked.add_self' _

theorem finalAB_zero (lay : Zk.PadLayout) (h : Array F128) {m : ℕ} (hm : 14 ≤ m) (hn : m - 6 ≤ lay.nMlv)
    (cs : Zc.Challenges) (tr : Trace) (hR1CS : ∀ i < 2 ^ m, (tr.az[i]! && tr.bz[i]!) = tr.z[i]!)
    (hc1 : ∀ k < m - 13, cs.c1 k ≠ 1) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) :
    ZerocheckMasked.evalL ((ZerocheckMasked.running lay (ZcZ.proof lay h m cs tr) (cs.r m) (pushes cs.ρ (m - 6)) cs.z
      (m - 6)).add (ZerocheckMasked.prodL lay (ZcZ.proof lay h m cs tr))) h = 0 := by
  have hf := Zc.fold_final hm cs tr hR1CS hc1
  rw [Zc.zcFold_fst (cs.r m) (Zc.proof m cs tr) (fun i => #[cs.ρ i]) (pushes cs.ρ (m - 6)) (m - 6)
    (fun k hk => (pushes_get cs.ρ (m - 6) k hk).symm) cs.z (m - 6) le_rfl] at hf
  rw [ZerocheckMasked.final_eval lay h _ _ (m - 6) hn (zcZ_masks lay h m cs tr) hab, hf]
  exact ZerocheckMasked.add_self' _

end FlockComplete.Replay
