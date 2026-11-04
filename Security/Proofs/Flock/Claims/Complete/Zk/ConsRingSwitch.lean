import Proofs.Flock.Claims.Complete.Zk.ConsLincheck
import Proofs.Flock.Claims.Complete.Replay.RingSwitchAlgebra

/-!
# The ring switch's two constraints vanish at the masked prover's pads

The masked prover runs the plain ring switch on the clear claims (`repPZk`'s `ringSwitchP` on `abcClaims` of the clear
zerocheck and lincheck outputs), so its `ŝ` meet the clear values. The masked replay hands the opening the claims'
values as affine forms in the pads (`Lincheck.lcAbValue`, `ZerocheckMasked.fcL`), and `Zk.replay`'s rows
"ring switch ab" and "ring switch c" are those forms plus the per-claim check `Zk.claimCheck`. By the transfer
(`ConsLincheck`'s three steps):

* `lcAbValue_eval`: on a `z_partial` masked at `lay.zp`, the `ab` form evaluates to the clear `⟨L(r′), z_partial⟩`,
  the plain lincheck's `w`;
* `claimCheck_eq`: `Zk.claimCheck` is the plain check `⟨wOf c, ŝ⟩`, which reads only the skip weights and `x[0]`;
* `rsAb_zero`, `rsC_zero`: the rows vanish wherever the plain check meets the clear value;
* `rsAb_zero_of`, `rsC_zero_of`: at the honest `ŝ` of a statement's table, by `claims_switch`.
-/

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Composed
open ZerocheckMasked (evalL)

local notation "𝔽" => FlockLevel3.GF128

/-- **`Zk.claimCheck` is the plain per-claim check**: it reads only the claim's skip weights and first coordinate. -/
theorem claimCheck_eq (c c' : Claim) (hs : c'.skip = c.skip) (hx : c'.x[0]! = c.x[0]!) (sv : Array F128) :
    Zk.claimCheck c' sv = dot (wOf c) sv := by
  simp only [Zk.claimCheck, wOf, hs, hx]

/-- **The masked `ab` value at the pads is the clear one**: on a `z_partial` masked at `lay.zp`, `Zk.replay`'s
`abValue` evaluates to `⟨L(r′), z_partial⟩` of the clear `z_partial`. -/
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

/-- The plain `ab` claim of the clear lincheck output (`abcClaims`' first). -/
def abClaim (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) : Claim :=
  { value := lc.w, skip := skipWeights lc.rPrime, x := lc.rIn ++ zc.rho.extract (kLog - 6) (m - 6) }

/-- The plain `c` claim of the clear zerocheck output (`abcClaims`' second). -/
def cClaim (zc : ZcOut) : Claim := { value := zc.vC, skip := skipWeights zc.z, x := zc.rRest }

theorem abClaim_mem (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) (extras : Array Claim) :
    abClaim m kLog zc lc ∈ abcClaims m kLog zc lc ++ extras :=
  Array.mem_append.2 (Or.inl (by simp [abcClaims, abClaim]))

theorem cClaim_mem (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) (extras : Array Claim) :
    cClaim zc ∈ abcClaims m kLog zc lc ++ extras :=
  Array.mem_append.2 (Or.inl (by simp [abcClaims, cClaim]))

/-- **`ring switch ab` vanishes at the masked prover's pads**, wherever the plain check of the clear `ab` claim meets
its value: on the masked lincheck proof, for any claim `c'` at the clear claim's skip weights and first coordinate
(the replay's, whose value it leaves at `0`), at any pads. -/
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

/-- **`ring switch c` vanishes at the masked prover's pads**, wherever the plain check of the clear `c` claim meets
its value: on the masked zerocheck proof, for any claim `c'` at the clear claim's skip weights and first coordinate,
at any pads. -/
theorem rsC_zero (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (zcs : Zc.Challenges) (tr : Trace) (c' : Claim)
    (sv : Array F128) (hs : c'.skip = skipWeights zcs.z) (hx : c'.x[0]! = (zcs.r m)[0]!)
    (hsw : dot (wOf (cClaim (Zc.out m zcs tr))) sv = (Zc.out m zcs tr).vC) :
    evalL ((ZerocheckMasked.fcL lay (ZcZ.proof lay h m zcs tr)).addConst (Zk.claimCheck c' sv)) h = 0 := by
  unfold ZerocheckMasked.fcL
  rw [evalL_addConst, ZerocheckMasked.evalL_masked _ _ _ (ZerocheckMasked.fc_lt_len lay),
    (zcZ_masks lay h m zcs tr).finalC, ZerocheckMasked.add_add_self,
    claimCheck_eq (cClaim (Zc.out m zcs tr)) c' hs hx sv, hsw]
  exact ZerocheckMasked.add_self' _

/-! ## At the honest `ŝ` -/

/-- **`ring switch ab` vanishes at the masked prover's pads and its honest `ŝ`**, for a statement's table, on a trace
whose bits satisfy its statement (`claims_switch`'s hypotheses, verbatim). -/
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

/-- **`ring switch c` vanishes at the masked prover's pads and its honest `ŝ`**, for a statement's table, on a trace
whose bits satisfy its statement (`claims_switch`'s hypotheses, verbatim). -/
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
