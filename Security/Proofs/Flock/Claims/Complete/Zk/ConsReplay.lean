import Proofs.Flock.Claims.Complete.Zk.ConsRingSwitch
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckReplay
import Proofs.Flock.Soundness.Discharge.Composed.Inner.LincheckReplay

/-!
# The masked replay's rows vanish wherever the plain checks pass at its coins

`Zk.replay` hands `verifyInner` five message-dependent rows: its own three (`zerocheck c_eval`,
`zerocheck final a·b`, `lincheck final`) and the opening's two (`ring switch ab`, `ring switch c`, `Zk.verify`'s
`abValue + ⟨w_ab, ŝ_ab⟩` and `cValue + ⟨w_c, ŝ_c⟩`). Live coins make the masked run draw plain's coins at plain's
sends, so each row is a plain check on the clear proof at the replay's coins, moved by the pads:

* `replay_rows_zero`: on a proof that is the clear one masked by the pads (`ZerocheckMasked.Masks`, `LcMasks`) at
  `h_ab = h_a·h_b`, the five rows evaluate to `0` wherever the clear proof passes the plain checks at the replay's
  coins: `P_c(z) = final_c`, the residual claim `= final_a·final_b`, the lincheck fold `= ⟨comb, z_partial⟩`, and the
  two claim checks `⟨w, ŝ⟩` meeting the clear `ab` and `c` values.

It is `replay_zerocheck` and `replay_lincheck` (the rows' shapes) composed with the transfer's evaluation lemmas
(`c_eval_eval`, `final_eval`, `lcFinal_eval`, `lcAbValue_eval`); no row needs its own walk of the masked run.
-/

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Composed
open ZerocheckMasked (evalL zAt rOuterAt rhoAt)
open Lincheck (coinAt rhoOf tsOf)

/-- **`Zk.replay`'s five rows vanish at the pads wherever the clear proof passes the plain checks at its coins.** -/
theorem replay_rows_zero (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (p : Proof) (root : ByteArray)
    (hm : 14 ≤ st.m) {t t' : Transcript} {rp : Zk.Replayed} (hrun : Zk.replay st lay p root t = .ok (rp, t'))
    (zc : ZerocheckProof) (lc : LincheckProof)
    (hzc : ZerocheckMasked.Masks lay h zc p.zerocheck (st.m - 6)) (hlc : LcMasks lay h lc p.lincheck (st.kLog - 6))
    (hn : st.m - 6 ≤ lay.nMlv) (hn' : st.kLog - 6 ≤ lay.nLc) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!)
    (svAb svC : Array F128)
    (hC : dot (lagrange nodesL den6Inv (zAt t)) zc.round1c = zc.finalC)
    (hF : Zerocheck.claim zc (pinned ++ rOuterAt t) (rhoAt t (st.m - 6)) (zAt t) (st.m - 6) = zc.finalA * zc.finalB)
    (hL : ∀ combv, st.fold.folded (coinAt t (3 + (st.m - 6))) (coinAt t (4 + (st.m - 6))) (skipWeights (coinAt t 2))
        ((rhoOf st.m t).extract 0 (st.kLog - 6)) (tsOf st.m st.kLog t) = .ok combv →
        combv.size = 64 ∧ (Flock.RecAlgebra.lcFold lc (fun j => #[(tsOf st.m st.kLog t)[j]!]) (st.kLog - 6)
          (coinAt t (3 + (st.m - 6)) * zc.finalA + zc.finalB + coinAt t (4 + (st.m - 6)))).1 = dot combv lc.zPartial)
    (hRab : dot (wOf (Lincheck.lcAbClaim st.m st.kLog (rhoOf st.m t) (tsOf st.m st.kLog t)
        (coinAt t (5 + (st.m - 6) + (st.kLog - 6))))) svAb =
      dot (skipWeights (coinAt t (5 + (st.m - 6) + (st.kLog - 6)))) lc.zPartial)
    (hRc : dot (wOf { value := 0, skip := skipWeights (zAt t), x := pinned ++ rOuterAt t }) svC = zc.finalC) :
    (∀ i < 3, evalL rp.cons[i]!.2 h = 0) ∧
      evalL (rp.abValue.addConst (Zk.claimCheck rp.claims[0]! svAb)) h = 0 ∧
      evalL (rp.cValue.addConst (Zk.claimCheck rp.claims[1]! svC)) h = 0 := by
  obtain ⟨h0, h1, hc1, hcv⟩ := ZerocheckMasked.replay_zerocheck st lay p root hm hrun
  obtain ⟨combv, hcomb, h2, habv, hc0⟩ := Lincheck.replay_lincheck st lay p root hm hrun
  obtain ⟨hsz, hfold⟩ := hL combv hcomb
  refine ⟨fun i hi => ?_, ?_, ?_⟩
  · rcases (by omega : i = 0 ∨ i = 1 ∨ i = 2) with rfl | rfl | rfl
    · rw [h0, ZerocheckMasked.c_eval_eval lay h zc p.zerocheck _ hzc, hC]
      exact ZerocheckMasked.add_self' _
    · rw [h1, ZerocheckMasked.final_eval lay h zc p.zerocheck _ hn hzc hab, hF]
      exact ZerocheckMasked.add_self' _
    · rw [h2]
      dsimp only
      rw [lcFinal_eval lay h lc p.lincheck _ hn' hlc zc.finalA zc.finalB _ _ _ _ hzc.finalA hzc.finalB _ combv hsz,
        hfold]
      exact ZerocheckMasked.add_self' _
  · rw [habv, hc0, evalL_addConst, lcAbValue_eval lay h lc.zPartial _ hlc.zp, claimCheck_eq _ _ rfl rfl, hRab]
    exact ZerocheckMasked.add_self' _
  · rw [hcv, hc1, evalL_addConst, ZerocheckMasked.fcL,
      ZerocheckMasked.evalL_masked _ _ _ (ZerocheckMasked.fc_lt_len lay), hzc.finalC, ZerocheckMasked.add_add_self,
      claimCheck_eq _ _ rfl rfl, hRc]
    exact ZerocheckMasked.add_self' _

end FlockComplete.Replay
