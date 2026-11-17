module

public import Proofs.Flock.Claims.Complete.Replay.RingSwitchAlgebra
public import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
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
import Proofs.Flock.Claims.Complete.Zk.ConsLincheck
import Proofs.Flock.Claims.Complete.Zk.ConsZerocheck
import Proofs.Flock.Soundness.Refine.Arith
import Std.Tactic.BVDecide.Normalize.Prop

/-! The ring switch's two constraints vanish at the masked prover's pads (`rsAb_zero_of`, `rsC_zero_of`). -/

@[expose] public section

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Composed
open ZerocheckMasked (evalL)

local notation "𝔽" => FlockLevel3.GF128

theorem claimCheck_eq (c c' : Claim) (hs : c'.skip = c.skip) (hx : c'.x[0]! = c.x[0]!) (sv : Array F128) :
    Zk.claimCheck c' sv = dot (wOf c) sv := by
  simp only [Zk.claimCheck, wOf, hs, hx]

theorem lcAbValue_eval (lay : Zk.PadLayout) (h : Array F128) (zp zp' : Array F128)
    (hm : ∀ i < 64, zp'[i]! = zp[i]! + h[lay.zp i]!) (rPrime : F128) :
    evalL (Lincheck.lcAbValue lay zp' rPrime) h = dot (skipWeights rPrime) zp := by
  unfold Lincheck.lcAbValue
  rw [ZerocheckMasked.evalL_combo _ _ lay.len h fun l hl => ?_]
  · refine ZerocheckMasked.dot_congr _ _ _ 64 (skipWeights_size rPrime) fun i hi => ?_
    rw [ZerocheckMasked.get_map' _ _ i (by simpa [Lincheck.lcZp] using hi), Lincheck.lcZp,
      ZerocheckMasked.get_map_lin _ hi, ZerocheckMasked.evalL_masked _ _ _ (zp_lt_len lay i hi), hm i hi,
      ZerocheckMasked.add_add_self]
  · obtain ⟨i, _, rfl⟩ := ZerocheckMasked.mem_map_range hl
    simp

def abClaim (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) : Claim :=
  { value := lc.w, skip := skipWeights lc.rPrime, x := lc.rIn ++ zc.rho.extract (kLog - 6) (m - 6) }

def cClaim (zc : ZcOut) : Claim := { value := zc.vC, skip := skipWeights zc.z, x := zc.rRest }

theorem abClaim_mem (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) (extras : Array Claim) :
    abClaim m kLog zc lc ∈ abcClaims m kLog zc lc ++ extras :=
  Array.mem_append.2 (Or.inl (by simp [abcClaims, abClaim]))

theorem cClaim_mem (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) (extras : Array Claim) :
    cClaim zc ∈ abcClaims m kLog zc lc ++ extras :=
  Array.mem_append.2 (Or.inl (by simp [abcClaims, cClaim]))

theorem rsAb_zero (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut)
    (full : Array F128) (cs : Lc.Challenges) (c' : Claim) (sv : Array F128)
    (hs : c'.skip = skipWeights (cs.cf 0))
    (hx : c'.x[0]! = (abClaim m c.kLog zc (Lc.out c m tr zc cs)).x[0]!)
    (hsw : dot (wOf (abClaim m c.kLog zc (Lc.out c m tr zc cs))) sv = (Lc.out c m tr zc cs).w) :
    evalL ((Lincheck.lcAbValue lay (LcZ.proof lay h c m tr zc full cs).zPartial (cs.cf 0)).addConst
      (Zk.claimCheck c' sv)) h = 0 := by
  rw [evalL_addConst, lcAbValue_eval lay h (Lc.proof c m tr zc full cs).zPartial _
      (lcZ_masks lay h c m tr zc full cs).zp, claimCheck_eq _ c' hs hx sv, hsw]
  exact ZerocheckMasked.add_self' _

theorem rsC_zero (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (zcs : Zc.Challenges) (tr : Trace) (c' : Claim)
    (sv : Array F128) (hs : c'.skip = skipWeights zcs.z) (hx : c'.x[0]! = (zcs.r m)[0]!)
    (hsw : dot (wOf (cClaim (Zc.out m zcs tr))) sv = (Zc.out m zcs tr).vC) :
    evalL ((ZerocheckMasked.fcL lay (ZcZ.proof lay h m zcs tr)).addConst (Zk.claimCheck c' sv)) h = 0 := by
  unfold ZerocheckMasked.fcL
  rw [evalL_addConst, ZerocheckMasked.evalL_masked _ _ _ (ZerocheckMasked.fc_lt_len lay),
    (zcZ_masks lay h m zcs tr).finalC, ZerocheckMasked.add_add_self,
    claimCheck_eq (cClaim (Zc.out m zcs tr)) c' hs hx sv, hsw]
  exact ZerocheckMasked.add_self' _

theorem rsAb_zero_of (st : Stmt) (hwf : StmtWF st) (hrw : RegionsWF st) (hm : 22 ≤ st.m) (w : Model.Witness st.m)
    (hsat : (stmtOf st hwf (regionsOf st)).Satisfies w) (tr : Trace)
    (hz : ∀ a : Fin (2 ^ st.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (zcs : Zc.Challenges) (cs : Lc.Challenges)
    (points : Array (Array F128)) (hpts : ∀ p ∈ points.toList, p.size = st.mPts)
    (lay : Zk.PadLayout) (h : Array F128) (full : Array F128) (c' : Claim)
    (hs : c'.skip = skipWeights (cs.cf 0))
    (hx : c'.x[0]! = (abClaim st.m st.c.kLog (Zc.out st.m zcs tr)
      (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) cs)).x[0]!) :
    evalL ((Lincheck.lcAbValue lay
      (LcZ.proof lay h st.circuitFold st.m tr (Zc.out st.m zcs tr) full cs).zPartial (cs.cf 0)).addConst
      (Zk.claimCheck c' (switchVec tr (abClaim st.m st.c.kLog (Zc.out st.m zcs tr)
        (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) cs))))) h = 0 :=
  rsAb_zero lay h st.circuitFold st.m tr (Zc.out st.m zcs tr) full cs c' _ hs hx
    (claims_switch st hwf hrw hm w hsat tr hz zcs cs points hpts _ (abClaim_mem _ _ _ _ _))

theorem rsC_zero_of (st : Stmt) (hwf : StmtWF st) (hrw : RegionsWF st) (hm : 22 ≤ st.m) (w : Model.Witness st.m)
    (hsat : (stmtOf st hwf (regionsOf st)).Satisfies w) (tr : Trace)
    (hz : ∀ a : Fin (2 ^ st.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (zcs : Zc.Challenges) (cs : Lc.Challenges)
    (points : Array (Array F128)) (hpts : ∀ p ∈ points.toList, p.size = st.mPts)
    (lay : Zk.PadLayout) (h : Array F128) (c' : Claim) (hs : c'.skip = skipWeights zcs.z)
    (hx : c'.x[0]! = (zcs.r st.m)[0]!) :
    evalL ((ZerocheckMasked.fcL lay (ZcZ.proof lay h st.m zcs tr)).addConst
      (Zk.claimCheck c' (switchVec tr (cClaim (Zc.out st.m zcs tr))))) h = 0 :=
  rsC_zero lay h st.m zcs tr c' _ hs hx
    (claims_switch st hwf hrw hm w hsat tr hz zcs cs points hpts _ (cClaim_mem st.m st.c.kLog _
      (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) cs) _))

end FlockComplete.Replay
