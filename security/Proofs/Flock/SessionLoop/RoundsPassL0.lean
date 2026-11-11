module

public import Proofs.Flock.SessionLoop.RoundsPassTaken
public import Proofs.Flock.SessionLoop.RoundL0Frame

/-!
# Level 0's rounds past the binding, taken without liveness

`LeanRelease.SchedTaken` at rep stream `2j + r`'s level-0 round `1 ≤ i ≤ k₀ + 3` (`Masked.repSlot` `i`), wherever the
frame takes its coins (**`LeanRelease.schedTaken_l0`**): the frame is in level 0's masked phase at body `i`'s position
(`RhoKeep.repState_l0`), the scheduled round on the rep's own table is body `i` at zero pads (`ofLoop_sched_unrefused`,
`Bind.sched_l0`, `Loop.loopNr_l0r`), and the frame takes it (**`LeanRelease.l0_taken`**: `l0Sent_at` and
`roundFree_on_fr`, at `k₀ + 3` `ρ₁`'s squeeze alone). This is `l0_frame`'s argument with the frame's answer from
`refusesCoins` in liveness's place, and `release_l0_at`'s without the device round.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree)
open FlockSoundness.Refine (Frame encs)

namespace LeanRelease

/-- **The frame takes level 0's body `i` at zero pads**: where the frame takes the coins before level 0's round
`1 ≤ i ≤ k₀ + 3` in its masked phase at the round's position and asks the round's `l0Nc i` coins (`l0Sent_at`,
`roundFree_on_fr`; at `k₀ + 3`, `ρ₁`'s squeeze alone). -/
theorem l0_taken {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (hh : List (Msg × Msg)) {m : Masker}
    {coin : ℕ → ℕ → F128} {p0 : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray} {sch : _root_.Flock.Schedule}
    {t b : Array F128} {lay : _root_.Flock.Zk.PadLayout} {rep : ℕ} {extra : ℕ → ℕ → F128}
    {nonce : ℕ → ℕ → UInt64} (len : ℕ) {i : ℕ}
    (hlay : L.lay x (L.rep x hh.length) = lay) (hk : 0 < sch.k0) (hlk : lay.k0 = sch.k0) (hood : lay.nOod0 = 1)
    (h1 : 1 ≤ i) (hi : i ≤ sch.k0 + 3) (hon : m.on = true) (hin : m.inner = none)
    (hpos : m.pos = absPos lay (lay.lgOod 0) + l0Off i) (hcn : L.coins x hh.length = l0Nc i) :
    L.RoundTaken x ρ hh m
      (FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b lay (Array.replicate len 0) rep extra nonce i) := by
  subst hlay
  have hc : ∀ q, q < (L.cfg x ρ hh).order.size → (L.cfg x ρ hh).order[q]! < (L.cfg x ρ hh).h.size :=
    Claims.ZKServer.ofDraws_order (L.lay x (L.rep x hh.length)) (L.draws ρ (L.rep x hh.length))
      (L.slot x (L.rep x hh.length))
  have hosz : (L.cfg x ρ hh).order.size =
      (_root_.Flock.Firewall.Frame.absorbed (L.lay x (L.rep x hh.length))).size := rfl
  have hnp : (L.cfg x ρ hh).nPiop =
      absPos (L.lay x (L.rep x hh.length)) ((L.lay x (L.rep x hh.length)).lgOod 0) :=
    nPiop_ood (L.lay x (L.rep x hh.length)) (L.drawsAt x ρ (L.rep x hh.length)) (L.slot x (L.rep x hh.length))
      (by omega)
  unfold RoundTaken
  rw [hcn]
  rcases Nat.lt_or_ge i (sch.k0 + 3) with h3 | h3
  · have hS : FlockComplete.LigLock.L0Sent (L.cfg x ρ hh)
        (absPos (L.lay x (L.rep x hh.length)) ((L.lay x (L.rep x hh.length)).lgOod 0) + l0Off i) (l0N i) (l0K i)
        (l0Nc i)
        (FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b (L.lay x (L.rep x hh.length))
          (Array.replicate len 0) rep extra nonce i)
        (FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b (L.lay x (L.rep x hh.length)) (L.cfg x ρ hh).h rep
          extra nonce i) :=
      FlockComplete.LigLock.l0Sent_at (lay := L.lay x (L.rep x hh.length)) (d := L.drawsAt x ρ (L.rep x hh.length))
        (mk := L.slot x (L.rep x hh.length)) (by omega) hood hlk.symm len h1 (by omega)
    obtain ⟨G, H, hR, hBz, -⟩ := id hS
    obtain ⟨hk2, h0, hnc⟩ := l0K_ok i
    have hsz := l0_size (L.lay x (L.rep x hh.length)) (by omega) hood h1
      (show i ≤ (L.lay x (L.rep x hh.length)).k0 + 2 by omega)
    obtain ⟨o, hro, -⟩ := roundFree_on_fr (L.cfg x ρ hh) { m with stopped := false } G (l0K i) (l0Nc i) hc hon hin
      hR.mid_G hR.plain_G hk2 h0 hnc (show m.pos + nAbs G ≤ _ by rw [hR.abs_G, hpos, hosz]; omega)
      (show _ ≤ m.pos by rw [hpos, hnp]; omega)
    rw [← hBz] at hro
    exact ⟨G, l0K i, o, _, #[], hBz, midAll_of_plain hR.mid_G hR.plain_G, hk2, h0, hnc, hro⟩
  · obtain rfl : i = sch.k0 + 3 := by omega
    have hn2 : l0Nc (sch.k0 + 3) = 2 := by
      unfold l0Nc
      rw [if_neg (by omega)]
    have ho3 : l0Off (sch.k0 + 3) = 4 * sch.k0 + 5 := by
      unfold l0Off
      rw [if_neg (by omega)]
      omega
    have ho2 : l0Off (sch.k0 + 2) = 4 * sch.k0 + 1 := by
      unfold l0Off
      rw [if_neg (by omega)]
      omega
    have hn4 : l0N (sch.k0 + 2) = 4 := by
      unfold l0N
      rw [if_neg (by omega)]
    have hsz := l0_size (L.lay x (L.rep x hh.length)) (by omega) hood (i := sch.k0 + 2) (by omega) (by omega)
    have hb : m.pos ≤ (L.cfg x ρ hh).order.size := by
      rw [hpos, hosz]
      omega
    have hpi : (L.cfg x ρ hh).nPiop ≤ m.pos := by
      rw [hpos, hnp]
      omega
    obtain ⟨o, hro, -⟩ := roundFree_on_fr (L.cfg x ρ hh) { m with stopped := false } [] 2 2 hc hon hin
      (by simp) (by simp) (.inl rfl) (by decide) (by norm_num) (show m.pos + 0 ≤ _ by omega) hpi
    have hB : FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b (L.lay x (L.rep x hh.length))
        (Array.replicate len 0) rep extra nonce (sch.k0 + 3) = encs [] ++ (Frame.sq 2 2).enc :=
      FlockComplete.LigLock.l0BodyAt_r1.trans rfl
    rw [← hB] at hro
    rw [hn2]
    exact ⟨[], 2, o, _, #[], hB, fun f hf => by simp at hf, .inl rfl, by decide, by norm_num, hro⟩

open Loop Ranked in
set_option maxHeartbeats 4000000 in
/-- **The frame takes a rep's level-0 round past the binding where it takes its coins**: at a history whose next round
is rep stream `2j + r`'s level-0 round `1 ≤ i ≤ k₀ + 3` (`Masked.repSlot` `i`), whose coins the loop's frame takes, on
the loop's own tables where `src` samples the tape's level 0. The frame is in level 0's masked phase at body `i`'s first
value (`RhoKeep.repState_l0`), the scheduled round is body `i` at zero pads on the rep's own table
(`ofLoop_sched_unrefused`, `Bind.sched_l0`, `Loop.loopNr_l0r`), and the frame takes it (`l0_taken`). -/
theorem schedTaken_l0 (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Loop.Serves x) (hS0 : Loop.Level0Src src x w)
    (hST : Loop.ScheduleTotal x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hlt : ops.length < refK x)
    (href : (ofLoop Masked.dataE L₁ (Loop.realOf src)).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : 1 ≤ i)
    (hi : i ≤ (Masked.zkTab x j).sch.k0 + 3) :
    (ofLoop Masked.dataE L₁ (Loop.realOf src)).SchedTaken x w (ν, t) ((m₀, ρ₀) :: ops) := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hmk : (Masked.stAt x j).merkle = _root_.Flock.MerkleScheme.hm96Sha512 := (htabs j.val hj).2.2
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  -- the round's place: `A + (3 + P) + i` into the rep's rounds, `A` the algebraic phase's, `2 + P` the ring switch's
  obtain ⟨A, hAd⟩ : ∃ A, A = (Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6 := ⟨_, rfl⟩
  obtain ⟨P, hPd⟩ : ∃ P, P = spec.points * (Masked.stAt x j).regions.size := ⟨_, rfl⟩
  have hRSf : ∀ f : _root_.Flock.Schedule → ℕ, Masked.repSlot x (2 * j.val + r.val) f =
      Masked.repBase x (2 * j.val + r.val) + A + (3 + P) + f (Masked.zkTab x j).sch := by
    intro f
    rw [Masked.repSlot_eq hsx hz, hst, ← Masked.repBase_eq hsx (2 * j.val + r.val), hAd, hPd]
    unfold FlockComplete.piopRoundsZk
    try dsimp only
    omega
  have hRSi : ops.length = Masked.repBase x (2 * j.val + r.val) + A + (3 + P) + i := by rw [hn, hRSf]
  have hbs1 : 1 ≤ Masked.repBase x (2 * j.val + r.val) := by
    rw [Masked.repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hpos : 0 < ops.length := by omega
  -- the extra claims: the table's regions at each link point
  obtain ⟨hmp, hmp0, hpc⟩ := Masked.live_links hx hsx hq2
  have hmp' : (Masked.stAt x j).mPts = spec.mPts := by rw [← hst]; exact hmp
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hW0 : (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0).size = spec.points * spec.mPts := by
    have e := wordsAt_size x ops (s := ops.length) (p := 0) hpos le_rfl
    rw [List.take_length] at e
    rw [e, hl0, Nat.min_eq_left hpc]
  obtain ⟨XS, hXSd⟩ : ∃ XS, XS = _root_.Flock.Firewall.Drive.extras (Masked.stAt x j)
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0)) :=
    ⟨_, rfl⟩
  have hXS : XS.size = P := by
    rw [hXSd, Count.extras_size, Count.linkPoints_size, hW0, hmp', Nat.mul_div_cancel _ hmp0, hPd]
    exact Nat.mul_comm _ _
  obtain ⟨-, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  rw [hst] at hxr
  have hXb : XS.size + 2 < 2 ^ 64 := by
    rw [hXSd, Count.extras_size]
    have := Nat.mul_le_mul_left (Masked.stAt x j).regions.size
      ((Loop.linkPoints_size_le (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0)).trans
        (show _ ≤ _root_.Flock.Firewall.Frames.COIN_MAX by rw [hW0]; exact hpc))
    omega
  -- the located round
  have hbase : Masked.repBase x (2 * j.val + r.val) =
      1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
    rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
  obtain ⟨NR, hNd⟩ : ∃ NR, NR = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! +
      ((Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) := ⟨_, rfl⟩
  have hR4 : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 = NR + 4 := by
    have e1 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + ((Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6) := by
      rw [hst]; omega
    rw [hrk, repRounds_true]
    simp only [hz]
    rw [hNd]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hR0 : A + (3 + P) + i < (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hR4, hNd, hst]
    unfold FlockComplete.piopRoundsZk
    omega
  obtain ⟨u, hp⟩ : ∃ u, ops.length = u + 2 := ⟨ops.length - 2, by omega⟩
  obtain ⟨a, ha⟩ : ∃ a, A + (3 + P) + i = a + 1 := ⟨A + (3 + P) + i - 1, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (u + 1) = (2 * j.val + r.val, a + 1) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) (a + 1)
      (by rw [hrsl]; exact hq2) (by omega)
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (u + 1)).2 = a + 1 := by rw [hloc]
  -- the frame's answer
  have hlast : (ofLoop Masked.dataE L₁ (Loop.realOf src)).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ (Loop.realOf src)) x
  have hkd : (ofLoop Masked.dataE L₁ (Loop.realOf src)).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, hA⟩ := answered_of_unrefused href
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ (Loop.realOf src) x (ν, t) (m₀, ρ₀) ops hsx (q := u + 1) (by omega) hj1
    ho ht] at hA'
  replace hA' := (bind_rankStop_ok hA').1
  -- the loop's placeholder table there
  have hcs : (ops.map fun p => ZKMal.CoinLayer.rd p.2).length = u + 2 := by rw [List.length_map]; omega
  have ho' : (progOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun h => answer_owed_none _ h hA'
  obtain ⟨c', tb, dom, xs, his, hph0⟩ := progOf_placeholder x _ hsx hcs ho'
  rw [locate_pred _ 0 u a hj1, hloc] at hph0
  dsimp only at hph0
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph0
  obtain ⟨-, hk0, hnn, hq⟩ := Count.sched_facts htp
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph0
  rw [hst] at hts' hxs
  have hsch : tb.sch = (Masked.zkTab x j).sch := by
    rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
    exact htp.1.symm
  have htm : tb.st.m = (Masked.stAt x j).m := by rw [hts']; rfl
  have htk : tb.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hts']; rfl
  have hdg : tb.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmkt : tb.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [hts']; exact hmk
  -- the loop's own round there
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_unrefused L₁ src x w hx hSv hS0 hST ν t m₀ ρ₀ ops href hkd
    hsx (q := u + 1) (by omega)
  have hre' : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  obtain ⟨hT1, hT2, -, -, -, hcapR⟩ := Count.real_tabAlike (Loop.realTabs_of src x w (ν, t) hsx) hph0 hre'
  have hs'' : schedSoFar tr ((2 * j.val + r.val) % 2)
      (spec.domain (spec.tables[(2 * j.val + r.val) / 2]!) ((2 * j.val + r.val) % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hir
      ((Array.range (a + 1)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (u + 1 - (a + 1) + q + 1)) = .ok sched' := by
    rw [hloc] at hs'
    exact hs'
  rw [hst, ← hXSd] at hs''
  have hcr : ((Array.range (a + 1)).map fun q =>
      wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (u + 1 - (a + 1) + q + 1)) =
      (Array.range (A + (3 + P) + i)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + q) := by
    rw [ha]
    refine Array.map_congr_left fun q _ => ?_
    congr 1
    omega
  rw [hcr] at hs''
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs''
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, htm]
  have htrk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1, htk]
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hmT' : tr.st.m < 2 ^ 64 := by rw [htrm]; exact lt_of_le_of_lt hm35 (by norm_num)
  have hAA' : alg'.size = A := by
    rw [(alg_cnt hmT (by rw [htrk]; omega) hmT' hal').1, htrm, htrk, hAd]
  have hT1' : tr.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := hT1.trans hts'
  have hT2' : tr.sch = (Masked.zkTab x j).sch := hT2.trans hsch
  -- the schedule: level 0's first `i + 1` bodies at zero pads
  obtain ⟨hssz, T, sws, ⟨_zc, _lc, -, hsws⟩, hbody⟩ := Bind.sched_l0 hal' hs'' hdom' (by rw [hT1]; exact hdg)
    hcapR hmT' (by rw [hT1]; exact hmkt) hXb (by rw [hT2]; exact hk0) (by rw [hT2]; exact hnn)
    (by rw [hT2]; exact hq) (by rw [hT2']; have := (Count.sched_rho hz).1; omega) hmT (j := i) (by rw [hT2']; omega)
    (by rw [Array.size_map, Array.size_range]; omega)
  subst hsws
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (u + 1)).2 < sched'.size := by
    rw [hj1]
    omega
  have hNL : _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j := by
    rw [hT1', hT2']
    all_goals rfl
  -- the round's rep, its layout and its coins
  have hrep : (ofLoop Masked.dataE L₁ (Loop.realOf src)).rep x ((m₀, ρ₀) :: ops).length = 2 * j.val + r.val := by
    show repOf x (ops.length + 1) = _
    rw [show ops.length + 1 = u + 1 + 2 by omega, repOf_add_two hsx, hloc]
  have hso : startOf x (2 * j.val + r.val) = .ok (lay, mask) := by
    unfold startOf
    rw [hsx]
    exact hds
  have hlayE : lay = Masked.layE x j := by
    obtain ⟨sch', tp', hs3, hl3⟩ := start_lay hds
    rw [hz, Except.ok.injEq, Prod.mk.injEq] at hs3
    rw [hl3, ← hs3.1, hst]
    try rfl
  have hLk : (ofLoop Masked.dataE L₁ (Loop.realOf src)).lay x (2 * j.val + r.val) = Masked.layE x j := by
    show ((startOf x (2 * j.val + r.val)).toOption.map (·.1)).getD default = _
    rw [hso]
    show lay = _
    exact hlayE
  have hcn : (ofLoop Masked.dataE L₁ (Loop.realOf src)).coins x ((m₀, ρ₀) :: ops).length = l0Nc i := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel, hn, Masked.repSlot_eq hsx hz]
    exact loopNr_l0r hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6) h1 hi
  -- the frame's state there: in level 0 at body `i`'s first value
  have hph : placeholder spec (Masked.stsOf x) (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (u + 1)).1
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', tb, dom, xs, his) := by
    rw [hloc]
    exact hph0
  have hA'' := hA'
  rw [progOf_at x _ hsx (q := u + 1) (by rw [List.length_map]; omega) hj1 hph, hloc] at hA''
  dsimp only at hA''
  obtain ⟨alg, -, hal, -⟩ := repState_ok hA''
  have hmb : tb.st.m < 2 ^ 64 := by rw [htm]; exact lt_of_le_of_lt hm35 (by norm_num)
  have hAA : alg.size = A := by
    rw [(alg_cnt (by rw [htm]; omega) (by rw [htk]; omega) hmb hal).1, htm, htk, hAd]
  have hxX : xs = XS := hxs.trans hXSd.symm
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have htp' : _root_.Flock.Zk.schedule tb.st.m = .ok (tb.sch, tp) := by
    rw [htm, ← hst]
    exact htp
  have hlay : lay = _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 := by
    rw [hlayE, hts', hsch]
    try rfl
  have hord : ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size =
      c'.order.size := by
    rw [hc']
    show (_root_.Flock.Firewall.Frame.absorbed ((ofLoop Masked.dataE L₁ (Loop.realOf src)).lay x
      ((ofLoop Masked.dataE L₁ (Loop.realOf src)).rep x ((m₀, ρ₀) :: ops).length))).size =
        (_root_.Flock.Firewall.Frame.absorbed lay).size
    rw [hrep, hLk, hlayE]
  rw [show a + 1 = alg.size + (xs.size + 2) + 1 + i by rw [hAA, hxX, hXS]; omega] at hA''
  obtain ⟨hon, -, hin, -, hpm0⟩ := RhoKeep.repState_l0 hcc hal (by rw [htm]; omega) hdg (placeholder_cap hph0)
    (hdm ▸ hdom') hmkt (by rw [hxX]; exact hXb) htp' hlay hc' hord h1 (by rw [hsch]; exact hi)
    (by rw [Array.size_map, Array.size_range]) hA''
  have hkL : 0 < (Masked.layE x j).k0 := by
    show 0 < (Masked.zkTab x j).sch.k0
    rw [← hsch]
    omega
  have hpm : m.pos = absPos (Masked.layE x j) ((Masked.layE x j).lgOod 0) + l0Off i := by
    rw [hpm0, hc', hlayE, nPiop_ood _ _ _ hkL]
  -- the frame takes body `i` at zero pads
  refine schedTaken_of hA hin ?_
  rw [ofLoop_scheduled Masked.dataE L₁ (Loop.realOf src) x w (ν, t) (m₀, ρ₀) ops _ hsx (q := u + 1) (by omega) hre hs'
    hlt', hj1, show a + 1 = alg'.size + (XS.size + 2) + 1 + i by omega, (hbody i le_rfl).1, hNL, hT2']
  exact l0_taken (ofLoop Masked.dataE L₁ (Loop.realOf src)) x (ν, t) ((m₀, ρ₀) :: ops)
    (lay := Masked.layE x j) (sch := (Masked.zkTab x j).sch) (Masked.layE x j).len (by rw [hrep]; exact hLk)
    (by rw [← hsch]; omega) rfl rfl h1 hi hon hin hpm hcn

end LeanRelease

end Claims.ZKServer
