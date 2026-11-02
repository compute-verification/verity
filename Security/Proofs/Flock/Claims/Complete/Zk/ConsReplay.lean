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
* `replay_rows_zero_honest`: the same at the honest masked prover (`ZcZ.proof`, `LcZ.proof`), from the plain lemmas'
  conclusions verbatim (`Zc.fold_final`, `Lc.lcAlgebra_of`, `claims_switch`) and seven coin identities: the replay's
  `z`, `r_outer`, `ρ`, `α`, `β`, `t_j` and `r′` are the prover's.

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

/-! ## At the honest masked prover -/

theorem arr_eq_of_get {a b : Array F128} (n : ℕ) (ha : a.size = n) (hb : b.size = n)
    (h : ∀ i < n, a[i]! = b[i]!) : a = b :=
  Array.ext (ha.trans hb.symm) fun i h1 h2 => by
    have := h i (ha ▸ h1)
    rwa [getElem!_pos a i h1, getElem!_pos b i h2] at this

theorem rhoAt_eq (t : Transcript) (n : ℕ) (f : ℕ → F128) (hf : ∀ j < n, coinAt t (3 + j) = f j) :
    rhoAt t n = pushes f n :=
  arr_eq_of_get n (by simp [rhoAt]) (pushes_size f n) fun i hi => by
    rw [pushes_get f n i hi, ← hf i hi, getElem!_pos _ i (by simpa [rhoAt] using hi)]
    simp [rhoAt, coinAt, Nat.add_assoc]

theorem rhoOf_eq (m : ℕ) (t : Transcript) (f : ℕ → F128) (hf : ∀ j < m - 6, coinAt t (3 + j) = f j) :
    rhoOf m t = pushes f (m - 6) :=
  arr_eq_of_get (m - 6) (by simp [rhoOf]) (pushes_size f _) fun i hi => by
    rw [pushes_get f _ i hi, ← hf i hi, getElem!_pos _ i (by simpa [rhoOf] using hi)]
    simp [rhoOf]

theorem tsOf_eq (m kLog : ℕ) (t : Transcript) (f : ℕ → F128)
    (hf : ∀ j < kLog - 6, coinAt t (5 + (m - 6) + j) = f j) : tsOf m kLog t = pushes f (kLog - 6) :=
  arr_eq_of_get (kLog - 6) (by simp [tsOf]) (pushes_size f _) fun i hi => by
    rw [pushes_get f _ i hi, ← hf i hi, getElem!_pos _ i (by simpa [tsOf] using hi)]
    simp [tsOf]

theorem wOf_congr (c c' : Claim) (hs : c.skip = c'.skip) (hx : c.x[0]! = c'.x[0]!) : wOf c = wOf c' := by
  simp only [wOf, hs, hx]

/-- **At the honest masked prover, `Zk.replay`'s five rows vanish wherever its coins are the prover's.** The masked
proof is the plain one masked by the pads (`ZcZ.proof`, `LcZ.proof`); the plain facts are `Zc.fold_final`'s,
`Lc.lcAlgebra_of`'s and `claims_switch`'s conclusions verbatim; and the seven coin identities say the replay's rounds
carry the prover's coins at plain's sends, which live coins give (the coin at a send is the send's reply). -/
theorem replay_rows_zero_honest (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (p : Proof) (root : ByteArray)
    (hm : 14 ≤ st.m) {t t' : Transcript} {rp : Zk.Replayed} (hrun : Zk.replay st lay p root t = .ok (rp, t'))
    (c : CircuitFold) (tr : Trace) (zcs : Zc.Challenges) (cs : Lc.Challenges) (full : Array F128)
    (hfd : st.fold = c) (hk : st.kLog = c.kLog)
    (hpz : p.zerocheck = ZcZ.proof lay h st.m zcs tr)
    (hpl : p.lincheck = LcZ.proof lay h c st.m tr (Zc.out st.m zcs tr) full cs)
    (hn : st.m - 6 ≤ lay.nMlv) (hn' : st.kLog - 6 ≤ lay.nLc) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!)
    (hz : zAt t = zcs.z) (hr : pinned ++ rOuterAt t = zcs.r st.m)
    (hρ : ∀ j < st.m - 6, coinAt t (3 + j) = zcs.ρ j)
    (hα : coinAt t (3 + (st.m - 6)) = cs.ca 0) (hβ : coinAt t (4 + (st.m - 6)) = cs.cb 0)
    (hτ : ∀ j < st.kLog - 6, coinAt t (5 + (st.m - 6) + j) = cs.τ j)
    (hr' : coinAt t (5 + (st.m - 6) + (st.kLog - 6)) = cs.cf 0)
    (hF : (Flock.RecAlgebra.zcFold (zcs.r st.m) (Zc.proof st.m zcs tr) (fun i => #[zcs.ρ i]) (st.m - 6)
      (Flock.RecAlgebra.zcStart (Zc.proof st.m zcs tr) zcs.z)).1 =
        (Zc.proof st.m zcs tr).finalA * (Zc.proof st.m zcs tr).finalB)
    (hL : ∃ comb, c.folded (cs.ca 0) (cs.cb 0) (skipWeights (Zc.out st.m zcs tr).z)
        ((Zc.out st.m zcs tr).rho.extract 0 (c.kLog - 6)) (pushes cs.τ (c.kLog - 6)) = .ok comb ∧ comb.size = 64 ∧
      (Flock.RecAlgebra.lcFold (Lc.proof c st.m tr (Zc.out st.m zcs tr) full cs) (fun j => #[cs.τ j]) (c.kLog - 6)
        (cs.ca 0 * (Zc.out st.m zcs tr).vA + (Zc.out st.m zcs tr).vB + cs.cb 0)).1 =
        dot comb (Lc.proof c st.m tr (Zc.out st.m zcs tr) full cs).zPartial)
    (svAb svC : Array F128)
    (hRab : dot (wOf (abClaim st.m c.kLog (Zc.out st.m zcs tr) (Lc.out c st.m tr (Zc.out st.m zcs tr) cs))) svAb =
      (Lc.out c st.m tr (Zc.out st.m zcs tr) cs).w)
    (hRc : dot (wOf (cClaim (Zc.out st.m zcs tr))) svC = (Zc.out st.m zcs tr).vC) :
    (∀ i < 3, evalL rp.cons[i]!.2 h = 0) ∧
      evalL (rp.abValue.addConst (Zk.claimCheck rp.claims[0]! svAb)) h = 0 ∧
      evalL (rp.cValue.addConst (Zk.claimCheck rp.claims[1]! svC)) h = 0 := by
  have hzc : ZerocheckMasked.Masks lay h (Zc.proof st.m zcs tr) p.zerocheck (st.m - 6) :=
    hpz ▸ zcZ_masks lay h st.m zcs tr
  have hlc : LcMasks lay h (Lc.proof c st.m tr (Zc.out st.m zcs tr) full cs) p.lincheck (st.kLog - 6) := by
    rw [hpl, hk]
    exact lcZ_masks lay h c st.m tr _ full cs
  have hρ' := rhoOf_eq st.m t zcs.ρ hρ
  have hτ' := tsOf_eq st.m st.kLog t cs.τ hτ
  refine replay_rows_zero st lay h p root hm hrun _ _ hzc hlc hn hn' hab svAb svC ?_ ?_ ?_ ?_ ?_
  · rw [hz]
    rfl
  · rw [hr, hz, rhoAt_eq t _ zcs.ρ hρ]
    rwa [Zc.zcFold_fst (zcs.r st.m) (Zc.proof st.m zcs tr) (fun i => #[zcs.ρ i]) (pushes zcs.ρ (st.m - 6))
      (st.m - 6) (fun k hk => (pushes_get zcs.ρ (st.m - 6) k hk).symm) zcs.z (st.m - 6) le_rfl] at hF
  · intro combv hcv
    obtain ⟨comb, hcomb, hsz, halg⟩ := hL
    rw [hfd, hα, hβ, show coinAt t 2 = zcs.z from hz, hρ', hτ', hk] at hcv
    obtain rfl : combv = comb := Except.ok.inj (hcv.symm.trans hcomb)
    refine ⟨hsz, ?_⟩
    rw [hα, hβ, hτ', hk, Lc.lcFold_congr (C := fun j => #[(pushes cs.τ (c.kLog - 6))[j]!])
      (C' := fun j => #[cs.τ j]) _ _ _ fun i hi =>
      show (pushes cs.τ (c.kLog - 6))[i]! = cs.τ i from pushes_get cs.τ (c.kLog - 6) i hi]
    exact halg
  · rw [hr', hρ', hτ', hk]
    exact (congrArg (dot · svAb) (wOf_congr
      (Lincheck.lcAbClaim st.m c.kLog (pushes zcs.ρ (st.m - 6)) (pushes cs.τ (c.kLog - 6)) (cs.cf 0))
      (abClaim st.m c.kLog (Zc.out st.m zcs tr) (Lc.out c st.m tr (Zc.out st.m zcs tr) cs)) rfl rfl)).trans hRab
  · rw [hz, hr]
    exact (congrArg (dot · svC) (wOf_congr { value := 0, skip := skipWeights zcs.z, x := zcs.r st.m }
      (cClaim (Zc.out st.m zcs tr)) rfl rfl)).trans hRc

end FlockComplete.Replay
