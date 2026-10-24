import Proofs.Flock.Claims.ZK.FinishSwRep

/-!
# The cut PIOP's end and basis at the deployed rep

`L0Pin`'s `repCut_cutL0` puts level 0's bodies at the coins from the round the rep's PIOP from a source ended at, and the
basis of its switched claims. `l0BodyAt_pads` compares two runs only at one coins, table and basis. Here the PIOP from
any source, answered by the opened coins from rep `2j + r`'s base, is shown to end at `ligP0` with the basis `ligBasis`,
the round and basis `swSrcE_repCut`'s `hB` names: the PIOP's switched claims read their claims' points alone, which are
the coins' (`piopFw_at`'s argument, at the source's switched vectors).

* `FlockComplete.piopCut_split`, **`FlockComplete.piopCut_at`**: a returned PIOP from a source ran zerocheck, lincheck,
  the rank check and the ring switch, and its switched claims are `Rs.sws` of the closed-form claims at its coins.
* **`Masked.piopCut_lig`**: at the deployed rep's base and the opened coins, it ended at `ligP0`, before the cut's
  stop, and its basis is `ligBasis`.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete

open Flock Transcript Replay

/-- **A returned PIOP from a source ran its four phases**, the rank check keeping the transcript and the log. -/
theorem piopCut_split {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} {lay : Zk.PadLayout} {src : RepSrc}
    {cap root : ByteArray} {extras : Array Claim} {prior : Array (List F128)} {words : List ℕ} {t₀ : Transcript}
    {l : Array Claims.Msg} {zp lp ss T sws t₃ l₃}
    (h : ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃)) :
    ∃ zc t₁ l₁ lc t₂ l₂,
      ((zerocheckCut (asksCut coin n) st.m st.digest cap root lay src.masked).run t₀).go l = (.ok ((zp, zc), t₁), l₁) ∧
        ((lincheckCut (asksCut coin n) st.fold st.m lay src.masked zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) ∧
        ((ringSwitchCut (asksCut coin n) src.sv (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ =
          (.ok ((ss, T, sws), t₃), l₃) := by
  unfold piopCut at h
  obtain ⟨⟨zp', zc⟩, t₁, l₁, h₁, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp', lc⟩, t₂, l₂, h₂, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨u, t₂', l₂', hr, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  obtain ⟨ht, hl⟩ := rankCheckP_go hr
  rw [ht, hl] at h
  try dsimp only at h
  obtain ⟨⟨ss', T', sws'⟩, t₃', l₃', h₃, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  rw [pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨⟨rfl, rfl, rfl, rfl, rfl⟩, rfl⟩, rfl⟩ := h
  exact ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, h₃⟩

/-- **The PIOP from a source's switched claims at the coins**: answered by `asksCut` from `l`, where the PIOP from a
source returns, its switched claims are `Rs.sws` of the closed-form claims at any trace `tr` at the ring switch's coins,
and it sent `3 + (m − 6) + 3 + (k_log − 6) + (2 + |extras|) + 1` bodies, all before the cut's stop `n`. -/
theorem piopCut_at {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} {lay : Zk.PadLayout} {src : RepSrc}
    {cap root : ByteArray} {extras : Array Claim} {prior : Array (List F128)} {words : List ℕ} {t₀ : Transcript}
    {l : Array Claims.Msg} {zp lp ss T sws t₃ l₃} (hm : 14 ≤ st.m)
    (h : ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃)) (tr : Trace) :
    sws = Rs.sws (abcClaims st.m st.kLog (Zc.out st.m (zcCoins coin l.size) tr)
          (Lc.out st.fold st.m tr (Zc.out st.m (zcCoins coin l.size) tr)
            (lcCoins coin (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
        ⟨fun k => coin (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + k),
          coin (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + (2 + extras.size))⟩ ∧
      l₃.size = l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + (2 + extras.size) + 1 ∧ l₃.size ≤ n := by
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, h₃⟩ := piopCut_split h
  obtain ⟨hz, hl₁⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁
  obtain ⟨hlc, hl₂⟩ := lincheckCut_at coin n _ _ _ _ _ h₂
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by
    rw [Array.size_append, abcClaims_size]
  obtain ⟨hs, hl₃, hn₃⟩ := ringSwitchCut_sws coin n _ _ (by rw [hsz]; omega) h₃
  rw [hl₂, hl₁, hsz] at hs
  rw [hl₁] at hlc
  refine ⟨hs.trans (Rs.sws_congr (by rw [Array.size_append, Array.size_append, abcClaims_size, abcClaims_size])
    (claims_x (abcClaims_size _ _ _ _) (abcClaims_size _ _ _ _)
      (abc_x hz.2.1 hz.2.2 hlc.2).1 (abc_x hz.2.1 hz.2.2 hlc.2).2) _), ?_, hn₃⟩
  rw [hsz] at hl₃
  omega

end FlockComplete

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **The PIOP from a source at the deployed rep ends at `ligP0` with the basis `ligBasis`**: at a live statement,
the session's setup and table `j`'s extra claims at the opened coins' points, the PIOP of rep `r` from any source,
answered by the opened coins from the rep's base, where it returns, ended at `ligP0`, before the cut's stop, and its
switched claims' basis is `ligBasis` at any witness and tape. -/
theorem piopCut_lig {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (n : ℕ) {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    {extras : Array _root_.Flock.Claim} (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (src : FlockComplete.RepSrc)
    (cap root : ByteArray) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) {t₀ : _root_.Flock.Transcript}
    {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val)) {zp lp ss T sws t₃ l₃}
    (h : ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) n) st
      (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) src cap root extras prior words).run t₀).go l =
        (.ok ((zp, lp, ss, T, sws), t₃), l₃)) :
    l₃.size = ligP0 x j r ∧ ligP0 x j r ≤ n ∧
      FlockComplete.basisVals sws (st.m - 7) = ligBasis x c w τ j r extras := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm25 : 25 ≤ (stAt x j).m := (hall j.val hj).1
  have hSm : st.m = (stAt x j).m := by
    subst hst
    rfl
  have hSk : st.kLog = (stAt x j).c.kLog := by
    subst hst
    rfl
  have hkL : st.fold.kLog = st.kLog := by
    subst hst
    rfl
  have hm14 : 14 ≤ st.m := by
    rw [hSm]
    omega
  have heS : (stAt x j).extraClaims (ptsE c spec) = .ok extras := by
    rw [tabsOf_stAt] at he
    exact he
  obtain ⟨hpts, -⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hpsz : (ptsE c spec).size = 2 := by
    unfold ptsE
    simp [hpts]
  have hesz : extras.size = spec.points * (stAt x j).regions.size := by
    rw [FlockDeploy.extraClaims_size heS, hpsz, hpts, Nat.mul_comm]
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  have hP : ligP0 x j r =
      repBase x (2 * j.val + r.val) + ((stAt x j).m + (stAt x j).c.kLog - 6) + (2 + extras.size + 1) := by
    show repSlot x (2 * j.val + r.val) (fun _ => 0) = _
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val)]
    unfold FlockComplete.piopRoundsZk
    rw [hesz]
    omega
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  obtain ⟨hsws, hl₃, hn₃⟩ := FlockComplete.piopCut_at hm14 h (trsE x w (UE.tapeZk τ))[j.val]!
  have hl3 : l₃.size = ligP0 x j r := by
    rw [hl₃, hl, hP, hkL, hSm, hSk]
    omega
  refine ⟨hl3, hl3 ▸ hn₃, ?_⟩
  have hsE : sws = swsE x c w τ j r extras := by
    rw [hsws]
    unfold swsE
    rw [tabsOf_stAt, ← hst, hl, zcCoins_ligRespE, lcCoins_ligRespE, rsCoins_ligRespE]
  rw [hsE]
  unfold ligBasis
  rw [show ligSt x j = st from (tabsOf_stAt x j).trans hst.symm]

end Masked
end Claims.ZKMal
