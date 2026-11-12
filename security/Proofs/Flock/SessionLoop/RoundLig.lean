module

public import Proofs.Flock.SessionLoop.SchedAt
public import Proofs.Flock.SessionLoop.SchedUnrefused
public import Proofs.Flock.SessionLoop.Tables
public import Proofs.Flock.SessionLoop.FinHmkLive
public import Proofs.Flock.SessionLoop.RoundLigCoin
public import Proofs.Flock.SessionLoop.RoundLigLock
public import Proofs.Flock.SessionLoop.RoundLigCut
public import Proofs.Flock.SessionLoop.RoundLigBasis
public import Proofs.Flock.SessionLoop.RoundLigFrame

/-!
# A rep's Ligerito rounds past level 0

`LiveSched`'s equation at rep stream `2j + r`'s Ligerito rounds from level 0's closing round (`Masked.repSlot`
`k0 + 4`) to the last before the inner proof (`Masked.repSlot` `k0 + 7 + 11R`, `R` the levels past level 0): level 0's
close, the levels ≥ 1 (each a fold, its commitment at the tape's salts, its sumcheck) and the queries. Past the algebraic phase passing is
byte equality, so the release is the frame's scheduled round, which the opening's feed computes from the frame's table,
the levels' salts and the coins; the switched session sends the same round from its table (`Masked.swSrcE`).

* `Loop.tabs_merkle`: a live statement's tables commit with `hm96-sha512`.
* `Loop.swMsgAt_lig`: the switched session's message there is the body its rep's Ligerito opening, cut there, sent
  (`Masked.runCut_lig`, the views' level-0 commitments returning quietly).
* **`LeanRelease.release_sched_lig`**: the release of the worker's round there is the switched session's message
  without its leading domain frame. The frame's round is `ligeritoPZk` at the history's words (`release_lig_frame`),
  which squeezes the opened coins (`LigCoin.ligeritoPZk_go_coin`) at the switch's basis `ligBasis` (`Loop.basis_lig`)
  and the tape's table and salts, and extras and lane pads that are the tape's within level 0's sizes, all the
  restart reads (`Loop.Level0Src.realOf`, `Loop.Level0Src.realOf_lanes`, `Loop.RepSrc.salt`); from round
  `ligP0 + k0 + 4` on its log is the cut opening's (`LigPast.ligeritoCut_lock`, at `Masked.ligY1E`, `Masked.ligYbarE`
  and `Masked.ligSaltE`).
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

namespace Loop

/-- **A live statement's tables commit with `hm96-sha512`** (`ZkTables.tables_of_setup`). -/
theorem tabs_merkle {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {i : ℕ}
    (hi : i < (FlockDeploy.tabsOf (Masked.stsOf x)).size) :
    (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hi' : i < (Masked.stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
  rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) i hi]
  simp only [FlockDeploy.tabsOf, Array.getElem_map]
  exact (hall i hi').2.2

/-- **The switched session at a rep's Ligerito rounds past level 0**: after openings `ops` whose next round is rep
stream `2j + r`'s, from `ligP0 + k0 + 4` to the last before the inner proof, where the coins show no refusal there,
the session sends the body the rep's Ligerito opening at the switched source sent there, cut there and started at
`ligP0` with the basis `ligBasis` (`Masked.runCut_lig`: the views' level-0 commitments return quietly). -/
theorem swMsgAt_lig (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (ops : List (Msg × Msg)) (hlt : ops.length < refK x) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (h1 : Masked.ligP0 x j r + (Masked.zkTab x j).sch.k0 + 4 ≤ ops.length)
    (h2 : ops.length < Masked.ligP0 x j r +
      ((Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)))
    (hN : ¬ Masked.Refused x (finCoins x ops) ops.length) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE (finCoins x ops) spec) = .ok extras) :
    ∃ T cm t₃ l₃ err L, l₃.size = Masked.ligP0 x j r ∧
      ((FlockComplete.ligeritoCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) ops.length) Masked.hsE
          _root_.Flock.MerkleScheme.hm96Sha512 (Masked.zkTab x j).sch T cm
          (Masked.ligBasis x (finCoins x ops) w t j r extras)
          (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0)
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t j.val).rep r.val)).run t₃).go l₃ =
        (.error err, L) ∧
      finS.swMsgAt x (finS.wt x w) (vs x w ν) t (ops.map fun p => finS.decOf p.2) = 0 :: L[ops.length]! := by
  have hj := Masked.lt_of_jOf j
  obtain ⟨schs, hs, -⟩ := Masked.schedsPZk_live hx
  let cmOf : ℕ → FlockComplete.Commit := fun i =>
    Classical.choose (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t i))
  have hcm : ∀ i, (FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
      _root_.Flock.Hashes V) _root_.Flock.MerkleScheme.hm96Sha512 (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t i) : V _) = .ok (cmOf i) := fun i =>
    (Classical.choose_spec (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t i))).1
  have hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t i)).go l =
      (.ok ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf), l) := by
    intro l
    have hq : FlockComplete.quietS.R
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t i))
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
              _root_.Flock.Hashes FlockComplete.PS)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t i)) :=
      _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
        FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
    rw [hq l, FlockComplete.mapM_ok_mem cmOf _ fun i hi => ?_]
    rw [tabs_merkle hx hsx (i := i) (by simpa using hi)]
    exact hcm i
  obtain ⟨T, sws, t₃, l₃, err, L, hl₃, hb, hL, hrl, hrn⟩ := Masked.runCut_lig hx w t (finCoins x ops)
    (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t) hsx hs hcms j r h1 h2 hN hext
  rw [tabs_merkle hx hsx (i := j.val) (by simpa [FlockDeploy.tabsOf] using hj), hb] at hL
  refine ⟨T, _, t₃, l₃, err, L, hl₃, hL, ?_⟩
  have hsrc : Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
        (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) :=
    rfl
  rw [swMsgAt_enc vs w ν t (ops.map fun p => finS.decOf p.2) (by rw [List.length_map]; exact hlt),
    List.length_map]
  show finS.enc x (finCoins x ops) ops.length
    (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) = _
  rw [enc_finS_of (finCoins x ops) (by omega)
    (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) hsx hN
    (by rw [← hsrc]; exact hrl), ← hsrc, hrn]

end Loop

namespace LeanRelease

set_option maxHeartbeats 2000000 in
/-- **Ligerito past level 0**: at rep stream `2j + r`'s round `i` past its PIOP (`Masked.repSlot` `i`), from
`k0 + 4` (level 0's close) to the last before the inner proof, the release of the worker's round is the switched
session's message without its leading domain frame. -/
theorem release_sched_lig (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : (Masked.zkTab x j).sch.k0 + 4 ≤ i)
    (h2 : i < (Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  have hx : Masked.Live x := Masked.live_of hSc hR
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  -- the schedule: `k0 ≥ 1`, `n = m - 7 ≠ 0`, at least two levels, and its sizes fit the coins' counts
  obtain ⟨h25, h35, hsm, hk0, -, ⟨hn7, hlv, -⟩, -, -, -⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (Masked.zkTab x j).hz
  have hsn : (Masked.zkTab x j).sch.n ≠ 0 := by omega
  have hsmall := FlockComplete.JsonRT.zkSchedule_small h25 h35 (Masked.zkTab x j).hz
  have hn62 : (Masked.zkTab x j).sch.n < 2 ^ 62 :=
    lt_of_lt_of_le (Nat.lt_of_succ_le hsmall.1) (by norm_num)
  have hq63 : ∀ l, ((Masked.zkTab x j).sch.levels.getD l default).q < 2 ^ 63 := fun l =>
    lt_of_lt_of_le (Nat.lt_of_succ_le (hsmall.2 l)) (by norm_num)
  -- the rep's rounds: its first, the PIOP, the rank check's, then past the PIOP
  have eS : ∀ f : _root_.Flock.Schedule → ℕ, Masked.repSlot x (2 * j.val + r.val) f =
      Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stAt x j) +
        f (Masked.zkTab x j).sch := by
    intro f
    rw [Masked.repSlot_eq hsx hz, ← Masked.repBase_eq hsx, hst]
  have eR : Masked.rankSlot x j.val r.val =
      Masked.repBase x (2 * j.val + r.val) + ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    unfold Masked.rankSlot
    rw [getElem!_pos (Masked.stsOf x) j.val hj]
    try rfl
  have eP : FlockComplete.piopRoundsZk spec (Masked.stAt x j) =
      ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) +
        (2 + spec.points * (Masked.stAt x j).regions.size + 1) := rfl
  have hjj : (2 * j.val + r.val) / 2 = j.val := by omega
  have hmod : (2 * j.val + r.val) % 2 = r.val := by omega
  have hn' : ops.length = Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stAt x j) +
      i := hn.trans (eS fun _ => i)
  have hP0 : Masked.ligP0 x j r =
      Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stAt x j) :=
    (eS fun _ => 0).trans (Nat.add_zero _)
  -- the coins show no refusal by the round
  have hN : ¬ Masked.Refused x (Loop.finCoins x ops) ops.length :=
    unrefused_sched Masked.dataE (withFiles (withLink L₀ src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops hlt.le hl
      href (q := 2 * j.val + r.val) (by omega) (by omega)
      (by simp only [hjj, hmod, Masked.betaSlot, eS]; omega) (by rw [hjj, hmod, eR]; omega)
  obtain ⟨extras, hext⟩ := Masked.extra_ok hx hsx (Loop.finCoins x ops) hj
  -- the session's round: the cut opening's body there
  obtain ⟨T, cm, t₃, l₃, err, Ls, hl₃, hLs, hmsg⟩ :=
    Loop.swMsgAt_lig vs w hx hsx ν t ops hlt j r (by omega) (by omega) hN hext
  rw [hmsg]
  -- the frame's round: `ligeritoPZk` at the history's words, at the loop's table
  obtain ⟨tb, his, xs, algC, base, cs, hreal', hxs, -, hanchor, halg, hbcs, hcsv, hbase, hfr⟩ :=
    release_lig_frame L₀ src real hreal wk hR hSv hSc ν t κ m₀ ρ₀ ops hpos hlt hl href hv j r hn h1 h2
  obtain ⟨Tf, swf, tr₂, l₂, Lf, hl₂, ⟨zc, lc, hpts, hswf⟩, hLf, hfed, hrel, hdd⟩ :=
    hfr (fun p k => (Loop.wordsAt x (ops.map fun q => CoinLayer.rd q.2) p)[k]!) fun a ha => funext fun k => by
      show (Loop.wordsAt x (ops.map fun q => CoinLayer.rd q.2) (base + a))[k]! = cs[a][k]!
      rw [hcsv a ha]
  rw [hreal] at hreal'
  -- the loop's table is the tape's, at the rep's schedule
  have hS' : Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w := hS
  have htabst : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = tb.st := by
    rw [Loop.realOf_st hsx hreal', hst, Masked.tabsOf_stAt]
  obtain ⟨g0, d0, l0, -, htr⟩ := Loop.realTabs_of src x w (ν, t) hsx (2 * j.val + r.val) tb his hreal'
  have hzz : (Except.ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) : V _) = .ok (tb.sch, tb.tPad) :=
    hz.symm.trans (Count.tabOfLevel0_sched htr).2
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (Except.ok.inj hzz)).1.symm
  have htp : tb.tPad = (Masked.zkTab x j).tPad := (Prod.mk.inj (Except.ok.inj hzz)).2.symm
  obtain ⟨-, -, -, hgt⟩ := Loop.Level0Src.realOf hS'.1 ν t j r hreal'
  obtain ⟨hex, hlp⟩ := Loop.Level0Src.realOf_lanes hS'.1 ν t j r hreal'
  -- the frame's basis is the session's
  subst hswf
  rw [Loop.basis_lig hSc hR hlt.le hsx j r t htabst hext hxs hanchor halg hbcs hcsv hbase (by omega) hpts] at hLf
  rw [hsch, htp, hgt, ← htabst] at hLf
  -- the frame's run squeezes the opened coins
  have hG : FlockComplete.LigCoin.Good (fun p k => (Loop.wordsAt x (ops.map fun q => CoinLayer.rd q.2) p)[k]!)
      (Masked.ligCoin (Loop.finCoins x ops)) ops.length l₂.size Lf := by
    intro p hp hpN t' kind k hk hLp e he
    have hsz := hfed p hp hpN t' kind k hk hLp
    have h := Loop.wordsAt_slot Masked.dataE x ops hlt.le hpN le_rfl (e := e)
      (by rw [List.take_length, hsz]; exact he)
    rw [List.take_length] at h
    exact h
  rw [← hLf] at hG
  have hcoin := FlockComplete.LigCoin.ligeritoPZk_go_coin (hn := hn62) (hq := hq63) (hG := hG)
  have hLf' := (congrArg Prod.snd hcoin).symm.trans hLf
  -- from `ligP0 + k0 + 4` on, the frame's log is the session's
  have hsz : l₃.size = l₂.size := hl₃.trans (hbase.symm.trans hl₂.symm)
  -- the restart reads the loop's extra lanes and lane pads only within level 0's sizes, where they are the tape's
  have hA : ∀ (coin : ℕ → ℕ → F128) (p0 e : ℕ),
      (Array.range (FlockComplete.LigLock.tabS coin p0 (Masked.ligTab x w t j) (Masked.zkTab x j).sch.k0).size).map
          (tb.extra e) =
        (Array.range (FlockComplete.LigLock.tabS coin p0 (Masked.ligTab x w t j) (Masked.zkTab x j).sch.k0).size).map
          ((Masked.UE.tapeZk t).extra j.val e) := fun coin p0 e =>
    Array.map_congr_left fun c hc =>
      hex e c (by rw [← Masked.tabS_ligTab_size coin p0 x w t j]; exact Array.mem_range.1 hc)
  have hlp' : ∀ l i, l < 2 ^ ((Masked.zkTab x j).sch.levels.getD 0 default).lanes + 4 →
      i < (Masked.zkTab x j).tPad → tb.lanePad l i = (Masked.UE.tapeZk t).lanePad j.val l i := hlp
  have hB : ∀ (coin : ℕ → ℕ → F128) (p0 : ℕ),
      (Array.range (Masked.zkTab x j).tPad).map (fun i =>
        (List.range (2 ^ ((Masked.zkTab x j).sch.levels.getD 0 default).lanes)).foldl
          (fun acc e => acc + ((_root_.Flock.eqTableK ((Array.range (Masked.zkTab x j).sch.k0).map
            (FlockComplete.LigLock.chS coin p0)))[e]!).smul (tb.lanePad e i))
          ((FlockComplete.LigLock.rhoS coin p0 (Masked.zkTab x j).sch.k0).1.smul
              (tb.lanePad (2 ^ ((Masked.zkTab x j).sch.levels.getD 0 default).lanes + 2 * r.val) i) +
            (FlockComplete.LigLock.rhoS coin p0 (Masked.zkTab x j).sch.k0).2.smul
              (tb.lanePad (2 ^ ((Masked.zkTab x j).sch.levels.getD 0 default).lanes + 2 * r.val + 1) i))) =
      (Array.range (Masked.zkTab x j).tPad).map (fun i =>
        (List.range (2 ^ ((Masked.zkTab x j).sch.levels.getD 0 default).lanes)).foldl
          (fun acc e => acc + ((_root_.Flock.eqTableK ((Array.range (Masked.zkTab x j).sch.k0).map
            (FlockComplete.LigLock.chS coin p0)))[e]!).smul ((Masked.UE.tapeZk t).lanePad j.val e i))
          ((FlockComplete.LigLock.rhoS coin p0 (Masked.zkTab x j).sch.k0).1.smul ((Masked.UE.tapeZk t).lanePad j.val
              (2 ^ ((Masked.zkTab x j).sch.levels.getD 0 default).lanes + 2 * r.val) i) +
            (FlockComplete.LigLock.rhoS coin p0 (Masked.zkTab x j).sch.k0).2.smul ((Masked.UE.tapeZk t).lanePad j.val
              (2 ^ ((Masked.zkTab x j).sch.levels.getD 0 default).lanes + 2 * r.val + 1) i))) := by
    intro coin p0
    have hr2 := r.isLt
    generalize 2 ^ ((Masked.zkTab x j).sch.levels.getD 0 default).lanes = L at hlp' ⊢
    refine Array.map_congr_left fun i hi => ?_
    have hi' := Array.mem_range.1 hi
    rw [hlp' (L + 2 * r.val) i (by omega) hi', hlp' (L + 2 * r.val + 1) i (by omega) hi']
    exact List.foldl_ext _ _ _ fun acc e he => by
      have := List.mem_range.1 he
      try dsimp only
      rw [hlp' e i (by omega) hi']
  have hk := FlockComplete.LigPast.ligeritoCut_lock _root_.Flock.Sha512.hash (Masked.ligCoin (Loop.finCoins x ops))
    ops.length _root_.Flock.MerkleScheme.hm96Sha512 (Masked.zkTab x j).sch T Tf cm tb.cm0.toC (Masked.ligTab x w t j)
    (Masked.ligBasis x (Loop.finCoins x ops) w t j r extras)
    (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0)
    (Array.replicate (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len 0)
    r.val (Masked.zkTab x j).tPad tb.extra tb.lanePad
    (fun l => _root_.Flock.Firewall.Finish.saltAt (his.getD (l - 1) ByteArray.empty)) (fun _ _ => 0)
    ((Masked.swSrcE x (Loop.finCoins x ops) w (vs x w ν (Loop.finCoins x ops) (Loop.finS.seenOf x t)) t j.val).rep r.val) t₃ tr₂ l₃ l₂
    hsz hk0 hsn hlv
    (fun ha hb => by
      rw [hl₃] at ha hb ⊢
      rw [hA, hA, (Masked.swSrcE_restart x (Loop.finCoins x ops) w (vs x w ν (Loop.finCoins x ops) (Loop.finS.seenOf x t)) t j.val r.val).1]
      exact Masked.ligY1E x hx w hR t (Loop.finCoins x ops) ops.length hN s0 spec hsx j r extras hext ha hb)
    (fun ha hb => by
      rw [hl₃] at ha hb ⊢
      rw [hB, (Masked.swSrcE_restart x (Loop.finCoins x ops) w (vs x w ν (Loop.finCoins x ops) (Loop.finS.seenOf x t)) t j.val r.val).2.1]
      exact Masked.ligYbarE x hx w hR t (Loop.finCoins x ops) ops.length hN s0 spec hsx j r extras hext ha hb)
    (fun ha hb l hl1 hl2 p hp => by
      rw [hl₃] at ha hb
      have es := Masked.ligSaltE Masked.hiSaltE x hx w hR t (Loop.finCoins x ops) ops.length hN s0 spec hsx j r
        extras hext ha hb l hl1 hl2 p hp
      have hmj : (Masked.ligSt x j).merkle = _root_.Flock.MerkleScheme.hm96Sha512 :=
        Loop.tabs_merkle hx hsx (by simpa [FlockDeploy.tabsOf] using hj)
      rw [hmj] at es
      rw [(Masked.swSrcE_restart x (Loop.finCoins x ops) w (vs x w ν (Loop.finCoins x ops) (Loop.finS.seenOf x t)) t j.val r.val).2.2]
      simp only [Loop.RepSrc.salt hS'.2 ν t j r hreal' l hl1 hl2 p hp]
      exact es)
  have hB : l₃.size + (Masked.zkTab x j).sch.k0 + 4 ≤ ops.length := by
    rw [hl₃]
    omega
  have hg : Ls[ops.length]? = Lf[ops.length]? := by
    have e := hk.log.get ops.length hB
    rw [hLs, hLf'] at e
    exact e
  have hLL : Ls[ops.length]! = Lf[ops.length]! := by
    simp only [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, hg]
  rw [hrel, hLL, hdd]

end LeanRelease

end Claims.ZKServer
