import Proofs.Flock.Claims.Complete.Zk.ConsReplay
import Proofs.Flock.Claims.Complete.Zk.RingSwitch
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep

/-!
# The masked replay's coins are the prover's, and its five rows vanish on the honest run

`ConsReplay`'s `replay_rows_zero_honest` takes seven coin identities: the replay's `z`, `r_outer`, `ρ`, `α`, `β`, `t_j`
and `r′` are the masked prover's. Live coins give them: the replay reads each coin off the round the prover's send made.

* `zc_coins_at`, `lc_coins_at`, `replay_coins`: all seven, on rounds carrying the masked prover's zerocheck and
  lincheck rounds (`Agrees`, as `rs_of` takes it), by `zc_coins` and the rounds' positions in `ZcZ.tEnd` and
  `LcZ.tEnd`. The lincheck half is stated from any start of the right length: the kernel then never unfolds
  `ZcZ.tEnd`, which it otherwise times out on;
* `replay_rows_zero_agrees`: `Zk.replay`'s five rows vanish at the pads on such rounds, from the plain lemmas'
  conclusions alone;
* `consHold_of_rows`, `replay_consHold_agrees`: the same as `ConsHold` (`verifyInnerWith_of`'s hypothesis) on the
  rows `Zk.verify` hands `verifyInner` ahead of level 0's lane rows: `rp.cons` (three, `replay_cons_size`) and the two
  ring-switch rows.
-/

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Composed Transcript
open ZerocheckMasked (evalL zAt rOuterAt rhoAt)
open Lincheck (rhoOf tsOf)

theorem coin_of_carries {R : Array Round} {j : ℕ} {T : Transcript} {f : ℕ → F128} (hc : Carries R j (sent T 1 1 f)) :
    (R[j]!).coins.getD 0 0 = f 0 := by
  rw [coins_of hc]
  simp [sent]

/-- **Zerocheck's coins at the stream**: on rounds carrying the masked prover's zerocheck from `t₀`, a replay from
round `t₀.k` reads the prover's `z`, `r_outer` and `ρ`. -/
theorem zc_coins_at {lay : Zk.PadLayout} {hp : Array F128} {m : ℕ} {digest cap root : ByteArray} {cz : Zc.Challenges}
    {tr : Trace} (t₀ : Transcript) (hk : t₀.k = t₀.rounds.size) (t : Transcript) (ht : t.k = t₀.k)
    (hZ : Agrees t.rounds (ZcZ.tEnd lay hp m digest cap root cz tr t₀).rounds 0) :
    zAt t = cz.z ∧ pinned ++ rOuterAt t = cz.r m ∧ ∀ j < m - 6, Lincheck.coinAt t (3 + j) = cz.ρ j := by
  obtain ⟨hρ, hr⟩ := zc_coins t₀ hk hZ
  refine ⟨?_, ?_, fun j hj => ?_⟩
  · have e : (ZcZ.t₄ lay hp m digest cap root cz tr t₀).rounds.size = t₀.k + 2 := by
      simp only [ZcZ.t₄, ZcZ.t₃, ZcZ.t₂, ZcZ.t₁, afterSq, Array.size_push, hk]
    have hc := hZ.at_push (ZcZ.prefix_loop (i := 0) t₀ (Nat.zero_le _)) (Nat.zero_le _)
    rw [e] at hc
    simp only [zAt, ht, coin_of_carries hc]
    rfl
  · simp only [zcAt, Zc.out] at hr
    simp only [rOuterAt, ht, hr]
  · have := congrArg (·[j]!) hρ
    simp only [zcAt, Zc.out, pushes_get _ _ j hj, FlockComplete.coinAt] at this
    simp only [Lincheck.coinAt, ht, ← Nat.add_assoc, this]

/-- **Lincheck's coins at the stream**: on rounds carrying the masked prover's lincheck from `T₁`, which starts
`3 + (m − 6)` rounds after the replay, the replay reads the prover's `α`, `β`, `t_j` and `r′`. -/
theorem lc_coins_at {lay : Zk.PadLayout} {hp : Array F128} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut}
    {full : Array F128} {cl : Lc.Challenges} (T₁ t : Transcript) (hTs : T₁.rounds.size = t.k + 3 + (m - 6))
    (hR : Agrees t.rounds (LcZ.tEnd lay hp c m tr zc full cl T₁).rounds 0) :
    Lincheck.coinAt t (3 + (m - 6)) = cl.ca 0 ∧ Lincheck.coinAt t (4 + (m - 6)) = cl.cb 0 ∧
      (∀ j < c.kLog - 6, Lincheck.coinAt t (5 + (m - 6) + j) = cl.τ j) ∧
      Lincheck.coinAt t (5 + (m - 6) + (c.kLog - 6)) = cl.cf 0 := by
  refine ⟨?_, ?_, fun j hj => ?_, ?_⟩
  · have hp₂ : Prefix ((Lc.t₁ T₁).rounds.push (sent (Lc.t₁ T₁) 1 1 cl.ca))
        (LcZ.tEnd lay hp c m tr zc full cl T₁).rounds :=
      (Prefix.push _ _).trans (LcZ.prefix_loop (i := 0) T₁ (Nat.zero_le _))
    have hc := hR.at_push hp₂ (Nat.zero_le _)
    rw [show (Lc.t₁ T₁).rounds.size = t.k + (3 + (m - 6)) by simp only [Lc.t₁, hTs]; omega] at hc
    exact coin_of_carries hc
  · have hp₃ : Prefix ((Lc.t₂ cl T₁).rounds.push (sent (Lc.t₂ cl T₁) 1 1 cl.cb))
        (LcZ.tEnd lay hp c m tr zc full cl T₁).rounds :=
      LcZ.prefix_loop (i := 0) T₁ (Nat.zero_le _)
    have hc := hR.at_push hp₃ (Nat.zero_le _)
    rw [show (Lc.t₂ cl T₁).rounds.size = t.k + (4 + (m - 6)) by
      simp only [Lc.t₂, Lc.t₁, afterSq, Array.size_push, hTs]; omega] at hc
    exact coin_of_carries hc
  · have hc := hR.at_push (LcZ.prefix_loop (i := j + 1) T₁ (by omega)) (Nat.zero_le _)
    rw [show (LcZ.loopT lay hp c m tr zc full cl T₁ j).rounds.size = t.k + (5 + (m - 6) + j) by
      rw [LcZ.loopT_size, hTs]; omega] at hc
    exact coin_of_carries hc
  · have hc := hR.at_push (Prefix.refl _) (Nat.zero_le _)
    rw [show (LcZ.t₄ lay hp c m tr zc full cl T₁).rounds.size = t.k + (5 + (m - 6) + (c.kLog - 6)) by
      simp only [LcZ.t₄, LcZ.loopT_size, hTs]; omega] at hc
    exact coin_of_carries hc

/-- **The seven coin identities** at a replay from round `t₀.k` of rounds carrying the masked prover's zerocheck and
lincheck from `t₀` (A9's `Agrees`, as `rs_of` takes it). -/
theorem replay_coins {lay : Zk.PadLayout} {hp : Array F128} {c : CircuitFold} {m : ℕ} {digest cap root : ByteArray}
    {cz : Zc.Challenges} {tr : Trace} {full : Array F128} {cl : Lc.Challenges} (t₀ : Transcript) (hk : t₀.k = t₀.rounds.size)
    (t : Transcript) (ht : t.k = t₀.k)
    (hR : Agrees t.rounds (LcZ.tEnd lay hp c m tr (Zc.out m cz tr) full cl
      (ZcZ.tEnd lay hp m digest cap root cz tr t₀)).rounds 0) :
    zAt t = cz.z ∧ pinned ++ rOuterAt t = cz.r m ∧ (∀ j < m - 6, Lincheck.coinAt t (3 + j) = cz.ρ j) ∧
      Lincheck.coinAt t (3 + (m - 6)) = cl.ca 0 ∧ Lincheck.coinAt t (4 + (m - 6)) = cl.cb 0 ∧
      (∀ j < c.kLog - 6, Lincheck.coinAt t (5 + (m - 6) + j) = cl.τ j) ∧
      Lincheck.coinAt t (5 + (m - 6) + (c.kLog - 6)) = cl.cf 0 := by
  obtain ⟨hz, hr, hρ⟩ := zc_coins_at t₀ hk t ht (hR.of_prefix (LcZ.prefix_start _))
  exact ⟨hz, hr, hρ, lc_coins_at _ t (by rw [ZcZ.tEnd_size, ← hk, ← ht]) hR⟩

/-- **`Zk.replay`'s five rows vanish at the pads on rounds carrying the masked prover's zerocheck and lincheck**, from
`Zc.fold_final`'s, `Lc.lcAlgebra_of`'s and `claims_switch`'s conclusions. -/
theorem replay_rows_zero_agrees (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (p : Proof) (root : ByteArray)
    (hm : 14 ≤ st.m) {t t' : Transcript} {rp : Zk.Replayed} (hrun : Zk.replay st lay p root t = .ok (rp, t'))
    (c : CircuitFold) (tr : Trace) (zcs : Zc.Challenges) (cs : Lc.Challenges) (full : Array F128)
    (hfd : st.fold = c) (hk : st.kLog = c.kLog)
    (hpz : p.zerocheck = ZcZ.proof lay h st.m zcs tr)
    (hpl : p.lincheck = LcZ.proof lay h c st.m tr (Zc.out st.m zcs tr) full cs)
    (hn : st.m - 6 ≤ lay.nMlv) (hn' : st.kLog - 6 ≤ lay.nLc) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!)
    (digest cap padsRoot : ByteArray) (t₀ : Transcript) (hk₀ : t₀.k = t₀.rounds.size) (ht : t.k = t₀.k)
    (hR : Agrees t.rounds (LcZ.tEnd lay h c st.m tr (Zc.out st.m zcs tr) full cs
      (ZcZ.tEnd lay h st.m digest cap padsRoot zcs tr t₀)).rounds 0)
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
  obtain ⟨hz, hr, hρ, hα, hβ, hτ, hr'⟩ := replay_coins t₀ hk₀ t ht hR
  rw [← hk] at hτ hr'
  exact replay_rows_zero_honest st lay h p root hm hrun c tr zcs cs full hfd hk hpz hpl hn hn' hab hz hr hρ hα hβ
    hτ hr' hF hL svAb svC hRab hRc

/-- **The replay's rows and the two ring-switch rows hold** (`ConsHold`, as `verifyInnerWith_of` takes it), on pads of
the layout's length, wherever the five rows vanish: `Zk.verify`'s row list ahead of level 0's lane rows. -/
theorem consHold_of_rows (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (p : Proof) (root : ByteArray)
    (hm : 14 ≤ st.m) {t t' : Transcript} {rp : Zk.Replayed} (hrun : Zk.replay st lay p root t = .ok (rp, t'))
    (hsz : h.size = lay.len) (svAb svC : Array F128)
    (hrows : (∀ i < 3, evalL rp.cons[i]!.2 h = 0) ∧
      evalL (rp.abValue.addConst (Zk.claimCheck rp.claims[0]! svAb)) h = 0 ∧
      evalL (rp.cValue.addConst (Zk.claimCheck rp.claims[1]! svC)) h = 0) :
    ConsHold (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! svAb)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! svC))]) h := by
  obtain ⟨hr, hab, hc⟩ := hrows
  have h3 := FlockSoundness.Discharge.ZkExec.replay_cons_size st lay p root hrun
  obtain ⟨h0, h1, -, hcv⟩ := ZerocheckMasked.replay_zerocheck st lay p root hm hrun
  obtain ⟨combv, -, h2, habv, -⟩ := Lincheck.replay_lincheck st lay p root hm hrun
  have hsize : ∀ i < 3, (rp.cons[i]!).2.a.size = lay.len := by
    intro i hi
    rcases (by omega : i = 0 ∨ i = 1 ∨ i = 2) with rfl | rfl | rfl
    · rw [h0]; exact size_rowCEval_form _ _ _
    · rw [h1]; exact size_rowAB_form _ _ _ _ _ _
    · rw [h2]
      simp [Lincheck.lcFinal, lcRunning_size]
  intro q hq
  rcases Array.mem_append.1 hq with hq | hq
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hq
    rw [← getElem!_pos rp.cons i hi, hsz]
    exact ⟨hsize i (h3 ▸ hi), hr i (h3 ▸ hi)⟩
  · simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl
    · refine ⟨?_, hab⟩
      rw [hsz, habv]
      simp [Zk.Lin.addConst, Lincheck.lcAbValue]
    · refine ⟨?_, hc⟩
      rw [hsz, hcv]
      simp [Zk.Lin.addConst, ZerocheckMasked.fcL]

/-- **`Zk.verify`'s rows ahead of the lane rows hold on rounds carrying the masked prover's zerocheck and lincheck**,
from the plain lemmas' conclusions: `replay_rows_zero_agrees` as `ConsHold`. -/
theorem replay_consHold_agrees (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (p : Proof) (root : ByteArray)
    (hm : 14 ≤ st.m) {t t' : Transcript} {rp : Zk.Replayed} (hrun : Zk.replay st lay p root t = .ok (rp, t'))
    (hsz : h.size = lay.len)
    (c : CircuitFold) (tr : Trace) (zcs : Zc.Challenges) (cs : Lc.Challenges) (full : Array F128)
    (hfd : st.fold = c) (hk : st.kLog = c.kLog)
    (hpz : p.zerocheck = ZcZ.proof lay h st.m zcs tr)
    (hpl : p.lincheck = LcZ.proof lay h c st.m tr (Zc.out st.m zcs tr) full cs)
    (hn : st.m - 6 ≤ lay.nMlv) (hn' : st.kLog - 6 ≤ lay.nLc) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!)
    (digest cap padsRoot : ByteArray) (t₀ : Transcript) (hk₀ : t₀.k = t₀.rounds.size) (ht : t.k = t₀.k)
    (hR : Agrees t.rounds (LcZ.tEnd lay h c st.m tr (Zc.out st.m zcs tr) full cs
      (ZcZ.tEnd lay h st.m digest cap padsRoot zcs tr t₀)).rounds 0)
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
    ConsHold (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! svAb)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! svC))]) h :=
  consHold_of_rows st lay h p root hm hrun hsz svAb svC
    (replay_rows_zero_agrees st lay h p root hm hrun c tr zcs cs full hfd hk hpz hpl hn hn' hab digest cap padsRoot t₀
      hk₀ ht hR hF hL svAb svC hRab hRc)

end FlockComplete.Replay
