import Proofs.Flock.SessionLoop.FinHmkCut
import Proofs.Flock.Claims.ZK.FinishLigEnd

/-!
# `final_c` from the history: the views' masked `final_c`, read off the inner proof's round 0

`Loop.swSrcE_hmk_cut` reads the views' source's masked values below `zp 64` off the history's PIOP rounds, all but
`final_c`'s, which no PIOP round carries. Rep `2j + r` absorbs it first in the inner proof's round 0, after its PIOP and
Ligerito opening: where the rep from the views' source, cut there, stops at that round (`hin`), the history's message
there is the cut session's body (`msg_runCut`, `Masked.runCut_cutRep_of_ret`), which is the inner proof's label frame and
the source's masked `final_c` between their headers (`finalCOf_inner`, the Ligerito opening having ended on a squeeze).
The loop's frame reads `final_c'` there (`finalCOf`) and adds its pad (`valsOf_fc`), so the pads cancel.

* **`Masked.runCut_cutRep`**, **`Masked.runCut_cutRep_of_ret`**: the cut session stopped at `n` in rep `2j + r` sent at
  `n` what the rep, cut at `n`, sent there.
* **`finalCOf_inner`**: `final_c'` of the inner proof's round 0 is its `final_c`, from no pending bytes.
* **`Loop.swSrcE_hmk_cut_fc`**: the views' masked `final_c` is the request's plus its pad.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **The cut session at rep `2j + r`'s stop**: as `runCut_cutPiop`, where rep `2j + r` itself, answered by the cut at
`n` from a history `l` at its first round, stops with log `L` (`hR`), the cut session stopped at `n` (from rep `2j + r`'s
first round, `n ≤ refK x`) sent at `n` what that rep sent there. -/
theorem runCut_cutRep {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2)
    (hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras)
    (hprev : ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((src (k / 2)).rep (k % 2))).go l = (.ok ic, l)) →
      ∀ l : Array Msg, l.size = repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) (refK x)) hsE
          (FlockDeploy.tabsOf (stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic ((src (k / 2)).rep (k % 2))
          (priorE c (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = repBase x (k + 1) ∧
        (k % 2 = 0 → FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog tr.rounds = ptsOpen c (stsOf x)[k / 2]! (2 * (k / 2))))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val))
    {n : ℕ} {e : String} {L : Array Msg}
    (hR : ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) n) hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch r.val (spec.domain spec.tables[j.val]! r.val).toUTF8
        cms[j.val]! extras ic ((src j.val).rep r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l = (.error e, L))
    (hn0 : repBase x (2 * j.val + r.val) ≤ n) (hnN : n ≤ refK x) :
    (runCut x c n spec src).2[n]! = L[n]! := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    have := r.isLt
    omega
  obtain ⟨schs', hs', hschs⟩ := schedsPZk_live hx
  have es : schs' = schs := Except.ok.inj (hs'.symm.trans hs)
  have esch : schs[j.val]! = ((zkTab x j).sch, (zkTab x j).tPad) := es ▸ hschs j
  have h1 : 1 ≤ repBase x (2 * j.val + r.val) := by
    rw [repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hmono : ∀ a b, a ≤ b → repBase x a ≤ repBase x b := fun a b hab => by
    rw [repBase_eq hsx, repBase_eq hsx]
    exact FlockComplete.ZkTables.repStart_mono _ _ hab
  -- rep `k`'s earlier claim points, read from the streams so far
  have hpri : ∀ (k : ℕ) (streams : Array (Array _root_.Flock.Round)),
      (k % 2 = 1 → FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog streams.back! = ptsOpen c (stsOf x)[k / 2]! (2 * (k / 2))) →
      FlockComplete.priorOf (FlockDeploy.tabsOf (stsOf x))[k / 2]! streams (k % 2) = priorE c (k / 2) (k % 2) := by
    intro k b hb
    unfold FlockComplete.priorOf priorE
    by_cases h1 : k % 2 = 1
    · rw [ite_eq_left_of_eq_true _ _ (eq_true h1), ite_eq_left_of_eq_true _ _ (eq_true h1)]
      exact hb h1
    · rw [ite_eq_right_of_eq_false _ _ (eq_false h1), ite_eq_right_of_eq_false _ _ (eq_false h1)]
  unfold runCut
  rw [FlockComplete.finSessionCut_eq, FlockComplete.go_bind, FlockComplete.schedsPZk_quiet, hs]
  dsimp only
  rw [FlockComplete.go_bind, hcms]
  dsimp only
  rw [FlockComplete.go_bind, FlockComplete.finAsksCut_link_go,
    ite_eq_right_of_eq_false _ _ (eq_false (by simp only [Array.size_empty]; omega))]
  try dsimp only
  unfold FlockComplete.afterLinkCut
  try simp only [Array.size_empty]
  try dsimp only
  rw [← Array.forIn_toList, FlockDeploy.repStreams_toList]
  refine Claims.ZKServer.Loop.forIn_bind_err (fun L' => L'[n]! = L[n]!)
    (fun k b l => l.size = repBase x k ∧ (k % 2 = 1 →
      FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog b.1.back! = ptsOpen c (stsOf x)[k / 2]! (2 * (k / 2))))
    (2 * j.val + r.val) (by simpa using hk) ?_ ⟨?_, fun h => absurd h (by norm_num)⟩ ?_
  · -- the reps before `2j + r` return by `n`, as at the firewall's stop
    rintro k hk' ⟨streams, proofs⟩ ls ⟨hls, hpts⟩
    obtain ⟨extras', hxe'⟩ := hexts k hk'
    obtain ⟨ic', hic'⟩ := padsTreeCut_go
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src (k / 2)).rep (k % 2))
    obtain ⟨p', tr', l'', hrep, hsz, hrp⟩ := hprev k hk' extras' hxe' ic' hic' ls hls
    have hle : repBase x (k + 1) ≤ repBase x (2 * j.val + r.val) := hmono _ _ (by omega)
    have hle' : repBase x k ≤ repBase x (k + 1) := hmono _ _ (by omega)
    have hcutn := FlockComplete.CutPrefix.eq_of_agree
      (FlockComplete.CutPrefix.repCut_cut_agree _ hnN _ _ _ _ _ _ _ _ _ _ _) (by omega) hrep (by omega)
    have hp := hpri k streams hpts
    unfold ptsE at hxe'
    refine ⟨?_, l'', ?_, hsz, ?_⟩
    rotate_left
    · simp only [List.getElem_map, List.getElem_range]
      try dsimp only
      rw [FlockComplete.go_bind, FlockComplete.go_liftV, hxe']
      try dsimp only
      rw [FlockComplete.go_bind, hic']
      try dsimp only
      rw [hp, FlockComplete.go_bind]
      try simp only [StateT.run] at hcutn ⊢
      rw [hcutn]
      try dsimp only
      rfl
    · intro h1
      have hk2 : (k + 1) / 2 = k / 2 := by omega
      rw [hk2]
      show FlockComplete.MaskRank.roundPoints _ _ (streams.push tr'.rounds).back! = _
      rw [Array.back!_push]
      exact hrp (by omega)
  · rw [repBase_eq hsx]
    simp [FlockComplete.repStartZk]
  · -- rep `2j + r` stops at `n`
    rintro ⟨streams, proofs⟩ l₀ ⟨hl₀, hpts⟩
    have hp := hpri (2 * j.val + r.val) streams hpts
    have e2 : (2 * j.val + r.val) / 2 = j.val := by omega
    have e3 : (2 * j.val + r.val) % 2 = r.val := by omega
    simp only [e2, e3] at hp
    obtain ⟨bs, hbs, hR0⟩ := FlockComplete.CutPrefix.go_of_shift
      (FlockComplete.CutPrefix.repCut_shift _ _ _ _ _ _ _ _ _ _ _ _ _) (hl.trans hl₀.symm) hR
    refine ⟨e, l₀ ++ bs, ?_, ?_⟩
    · simp only [List.getElem_map, List.getElem_range]
      simp only [e2, e3]
      try dsimp only
      rw [esch]
      try dsimp only
      unfold ptsE at hext
      rw [FlockComplete.go_bind, FlockComplete.go_liftV, hext]
      try dsimp only
      rw [FlockComplete.go_bind, hic]
      try dsimp only
      rw [hp, FlockComplete.go_bind]
      try simp only [StateT.run] at hR0 ⊢
      rw [hR0]
    · rw [hbs]
      exact Claims.ZKServer.Loop.getElem!_append_shift (hl₀.trans hl.symm) (by omega)

/-- **The cut session at rep `2j + r`'s stop, from the earlier reps' returns alone**: `runCut_cutRep` with rep `0`'s
claim points at the opened coins read off its return (`repCut_points`). -/
theorem runCut_cutRep_of_ret {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2)
    (hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras)
    (hprev : ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((src (k / 2)).rep (k % 2))).go l = (.ok ic, l)) →
      ∀ l : Array Msg, l.size = repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) (refK x)) hsE
          (FlockDeploy.tabsOf (stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic ((src (k / 2)).rep (k % 2))
          (priorE c (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = repBase x (k + 1))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val))
    {n : ℕ} {e : String} {L : Array Msg}
    (hR : ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) n) hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch r.val (spec.domain spec.tables[j.val]! r.val).toUTF8
        cms[j.val]! extras ic ((src j.val).rep r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l = (.error e, L))
    (hn0 : repBase x (2 * j.val + r.val) ≤ n) (hnN : n ≤ refK x) :
    (runCut x c n spec src).2[n]! = L[n]! := by
  refine runCut_cutRep hx c src hsx hs hcms j r hexts (fun k hk extras' hxe ic' hic' l' hl' => ?_) hext hic hl
    hR hn0 hnN
  obtain ⟨p, tr, l'', hrep, hsz⟩ := hprev k hk extras' hxe ic' hic' l' hl'
  refine ⟨p, tr, l'', hrep, hsz, fun h0 => ?_⟩
  have := r.isLt
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  exact repCut_points hx c (refK x) (i := k / 2) (by omega) _ _ _ _ _ _ _ _ _
    (by rw [show 2 * (k / 2) = k by omega]; exact hl') hrep

end Masked
end Claims.ZKMal

namespace Claims.ZKServer

open _root_.Flock (V F128)
open FlockSoundness.Refine (Frame encs)
open Claims.ZKMal (dropDom)

/-- **The inner proof's round 0 from no pending bytes**: its label frame, `final_c`'s, then the squeeze. -/
theorem innerBody_enc (v : F128) {t : _root_.Flock.Transcript} (hp : t.pending = ByteArray.empty) :
    Claims.toMsg (_root_.Flock.Transcript.framed (FlockComplete.Replay.InZ.t₁ v t) 1 1) =
      encs [.label _root_.Flock.Zk.LABEL_INNER.toUTF8, .one v] ++ (Frame.sq 1 1).enc := by
  unfold Claims.toMsg
  rw [FlockSoundness.Refine.framed_enc]
  show (FlockComplete.Replay.pF v (FlockComplete.Replay.pLabel _root_.Flock.Zk.LABEL_INNER t.pending)).data.toList ++ _ = _
  rw [FlockComplete.Replay.pF, FlockSoundness.Refine.absorbF_enc, FlockComplete.Replay.pLabel,
    FlockSoundness.Refine.label_enc, hp]
  simp [encs]

/-- **`final_c'` of the inner proof's round 0 is its `final_c`**: past its label frame and `final_c`'s header, where it
began with no pending bytes. -/
theorem finalCOf_inner (v : F128) {t : _root_.Flock.Transcript} (hp : t.pending = ByteArray.empty) :
    Loop.finalCOf ⟨(dropDom (0 :: Claims.toMsg
      (_root_.Flock.Transcript.framed (FlockComplete.Replay.InZ.t₁ v t) 1 1))).tail.toArray⟩ = v := by
  rw [innerBody_enc v hp, dropDom_noDom [.label _, .one v] 1 1 (by
    intro f hf
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> exact True.intro), List.tail_cons]
  show wd (encs [.label _root_.Flock.Zk.LABEL_INNER.toUTF8, .one v] ++ (Frame.sq 1 1).enc)
    ((_root_.Flock.Firewall.Frame.labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) = v
  refine wd_drop (rest := (Frame.sq 1 1).enc) ?_
  have hl : ((Frame.label _root_.Flock.Zk.LABEL_INNER.toUTF8).enc ++
      (_root_.Flock.Transcript.header 3 1 1).data.toList).length =
      (_root_.Flock.Firewall.Frame.labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16 := by
    rw [List.length_append, FlockSoundness.Refine.Frame.header_length, ← labelFrame_data, Array.length_toList]
    rfl
  rw [← hl]
  have e : encs [.label _root_.Flock.Zk.LABEL_INNER.toUTF8, .one v] ++ (Frame.sq 1 1).enc =
      ((Frame.label _root_.Flock.Zk.LABEL_INNER.toUTF8).enc ++ (_root_.Flock.Transcript.header 3 1 1).data.toList) ++
        (v.toBytes.data.toList ++ (Frame.sq 1 1).enc) := by
    simp [encs, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]
  rw [e, List.drop_left]

/-- **The value a rep's frame took at `final_c`'s pad** (`valsOf`): `final_c'` plus its pad. -/
theorem valsOf_fc (lay : _root_.Flock.Zk.PadLayout) (b : ByteArray) (ms : Option _root_.Flock.Firewall.Frame.MaskSlot)
    (M : Array F128) (fc : F128) :
    (Loop.valsOf (_root_.Flock.Firewall.Frame.Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms) M fc)[lay.fc]! =
      fc + (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h[lay.fc]! := by
  have hlt : lay.fc < lay.zp 64 := by
    unfold _root_.Flock.Zk.PadLayout.zp _root_.Flock.Zk.PadLayout.lc
    omega
  have hih : lay.fc < (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h.size := by
    rw [ofBytes_h_size]
    unfold _root_.Flock.Zk.PadLayout.len _root_.Flock.Zk.PadLayout.tw _root_.Flock.Zk.PadLayout.hab
      _root_.Flock.Zk.PadLayout.zp _root_.Flock.Zk.PadLayout.lc
    omega
  have hcl : (_root_.Flock.Firewall.Frame.Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms).lay = lay := rfl
  have hch : (_root_.Flock.Firewall.Frame.Cfg.ofDraws lay (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b) ms).h =
      (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h := rfl
  unfold Loop.valsOf
  simp only [hcl, hch]
  rw [set!_hit]
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hih, Option.getD_some,
      getElem!_pos (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h lay.fc hih]
  · rw [← Array.foldl_toList, foldl_set_size _ (fun v e => by cases e; exact Array.size_set! _ _ _) _ _,
      Array.size_replicate]
    exact hlt

namespace Loop

open Claims Claims.ZKMal
open _root_.Flock (Stmt SessionSpec)

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

theorem getElem!_push_size {L : Array Msg} (b : Msg) : (L.push b)[L.size]! = b := by
  rw [getElem!_pos (L.push b) L.size (by simp), Array.getElem_push_eq]

set_option maxHeartbeats 4000000 in
/-- **`swSrcE_hmk_cut_fc`: the views' masked `final_c` is the request's plus its pad, from the history**: at
(c.3)'s context on the loop's tables, where the earlier reps return from the views' source cut at the opened coins
(`swSrcE_hmk_cut`'s premises) and table `j`'s rep `r` from that source, cut at its inner proof's round 0
(`repSlot … (innerOff 0)`, the rep's scheduled rounds past its first), stops there, after a Ligerito opening that left
no pending bytes, with that round's body (`hin`), the masked `final_c` of the views' source is the request's
`final_c` plus its pad. The history's message at that round is the cut session's body (`msg_runCut`,
`runCut_cutRep_of_ret`), which carries the source's masked `final_c` past the inner proof's label frame
(`finalCOf_inner`); the loop's frame took it as `final_c'` and added its pad (`valsOf_fc`). -/
theorem swSrcE_hmk_cut_fc (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut Masked.hsE (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle
        (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t i)).go l = (.ok cms, l))
    (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (hprev : ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut Masked.hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))).go l =
          (.ok ic, l)) →
      ∀ l : Array Msg, l.size = Masked.repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x)) Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))
          (Masked.priorE (finCoins x ops) (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k + 1))
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray} (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut Masked.hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = Masked.repBase x (2 * j.val + r.val))
    (hin : ∃ (e : String) (L : Array Msg) (t₄ : _root_.Flock.Transcript), t₄.pending = ByteArray.empty ∧
      L.size = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0) ∧
      ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) L.size) Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! (Masked.zkTab x j).sch r.val
          (spec.domain spec.tables[j.val]! r.val).toUTF8 cms[j.val]! extras ic
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
          (Masked.priorE (finCoins x ops) j.val r.val) (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
        (.error e, L.push (Claims.toMsg (_root_.Flock.Transcript.framed (FlockComplete.Replay.InZ.t₁
          (((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked
            (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
              (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).fc) t₄) 1 1)))) :
    ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked
        (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc =
      q.vals[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc]! +
        (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog
          tb.sch.k0).fc]! := by
  obtain ⟨e, L, t₄, hp4, hLs, hrun⟩ := hin
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx := Masked.live_of hsc hR
  have hj := Masked.lt_of_jOf j
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hq : 2 * j.val + r.val < 2 * spec.tables.size := by rw [hspec]; omega
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j.val + r.val) / 2 = j.val := by omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) :=
    Masked.tabsOf_stAt x j
  -- the loop's table and schedule are the session's
  obtain ⟨schs', hschs', hschj⟩ := Masked.schedsPZk_live hx
  obtain ⟨htabst, hschtb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs' hq hRep
  rw [hjj] at htabst hschtb
  have htbsch : tb.sch = (Masked.zkTab x j).sch := by
    have e := hschtb.symm.trans (hschj j)
    simp only [Prod.mk.injEq] at e
    exact e.1
  have hts : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htabst.symm.trans htab
  -- the request
  have hq' := hRep.2.1
  unfold repInOf at hq'
  obtain ⟨⟨lay, msk⟩, hds, hq'⟩ := bok hq'
  obtain ⟨sch', tp', hsch', hlay⟩ := start_lay hds
  rw [hz] at hsch'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hsch'
  obtain ⟨rfl, rfl⟩ := hsch'
  rw [hst] at hlay
  have hlayG : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 = lay := by
    rw [hlay, hts, htbsch]; rfl
  have hlayP : _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0 = lay := by
    rw [hlay, htab]; rfl
  rw [hlayG]
  rw [hlayP] at hrun
  -- where the inner proof's round 0 sits
  have h1 : 1 ≤ Masked.repBase x (2 * j.val + r.val) := by
    rw [Masked.repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hslot : L.size = Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stAt x j) +
      Masked.innerOff 0 (Masked.zkTab x j).sch := by
    rw [hLs, Masked.repSlot_eq hsx hz, Masked.repBase_eq hsx, hst]
  have hnext := Masked.repStart_next (spec := spec) j r
  rw [← Masked.repBase_eq hsx, ← Masked.repBase_eq hsx] at hnext
  have hsucc := FlockComplete.ZkTables.repStart_succ spec (Masked.stsOf x) (2 * j.val + r.val)
  rw [← Masked.repBase_eq hsx, ← Masked.repBase_eq hsx] at hsucc
  have hlastR := FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x)
    (show 2 * j.val + r.val + 1 ≤ 2 * (Masked.stsOf x).size by omega)
  rw [FlockComplete.ZkTables.repStart_last, ← Masked.repBase_eq hsx] at hlastR
  have hK := Masked.refK_eq hsx
  have hLN : L.size < refK x := by omega
  -- the cut session's body there
  have hW := Masked.runCut_cutRep_of_ret hx (finCoins x ops)
    (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t) hsx hs hcms j r hexts hprev hext hic hl hrun
    (by omega) hLN.le
  rw [getElem!_push_size] at hW
  have hlast : (LeanRelease.ofLoopLink L₀ src (realOf src)).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE _ x
  have hsrc : Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
        (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) := rfl
  have hlen : ops.length = refK x := hF.1
  obtain ⟨s', spec', hsx', hm⟩ := msg_runCut vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w
    hE hc hlast ν t κ m₀ ρ₀ ops hF.1 hF.2.1 hF.2.2.1 hF.2.2.2.2.2 (L.size + 1) (by omega) (by omega)
  rw [hsx] at hsx'
  cases hsx'
  rw [← hsrc, Nat.add_sub_cancel, hW] at hm
  -- the round the loop read `final_c'` from
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  have hstream : streamAt (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val) + 1 =
      Masked.repBase x (2 * j.val + r.val) := by
    rw [Masked.repBase_eq hsx]
    unfold streamAt roundsOf FlockComplete.repStartZk
    rw [take_map_range _ (by omega)]
    omega
  have hI : _root_.Flock.Firewall.Frame.INNER_ROUNDS = 4 := rfl
  have hround : ((seenRounds x ((m₀, ρ₀) :: ops) (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val)).getD
      ((seenRounds x ((m₀, ρ₀) :: ops) (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val)).length -
        _root_.Flock.Firewall.Frame.INNER_ROUNDS) (.empty, 0, #[])).1 =
      ⟨(dropDom (0 :: Claims.toMsg (_root_.Flock.Transcript.framed (FlockComplete.Replay.InZ.t₁
        (((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked lay.fc) t₄)
        1 1))).tail.toArray⟩ := by
    have hp : L.size < ops.length := by omega
    unfold seenRounds
    rw [List.length_map, List.length_range, hrk, hI, List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_range (by omega), Option.map_some, Option.getD_some]
    dsimp only
    rw [List.tail_cons, show streamAt (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val) +
      (FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! - 4) + 1 = L.size by omega,
      List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hp, Option.getD_some]
    have e : (ops[L.size]'hp) = ((m₀, ρ₀) :: ops)[L.size + 1]'(by simp only [List.length_cons]; omega) := rfl
    rw [e, hm]
  cases hq'
  dsimp only
  rw [valsOf_fc, hround, finalCOf_inner _ hp4,
    show (_root_.Flock.Firewall.Finish.padsOf (_root_.Flock.Firewall.Frame.Draws.ofBytes lay
      (L₀.draws (ν, t) (2 * j.val + r.val)))).toC.h =
      (_root_.Flock.Firewall.Frame.Draws.ofBytes lay (L₀.draws (ν, t) (2 * j.val + r.val))).h from rfl,
    add_pad_pad]

end Loop

end Claims.ZKServer
