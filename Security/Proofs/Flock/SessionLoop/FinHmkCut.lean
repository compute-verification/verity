import Proofs.Flock.SessionLoop.FinL0Vals
import Proofs.Flock.Claims.ZK.FrameFree
import Proofs.Flock.Claims.ZK.FinishTau

/-!
# `hmk` from the history: the views' masked values below `zp 64`, read off the cut session's PIOP

At (c.3)'s context, below `zp 64` and away from `final_c`'s pad, the views' source (`Masked.swSrcE` at the workers'
values `vs`) holds the request's values plus its pads (`q.vals + pads.h`). Nothing here reads the view rep's return: the
history's rounds of rep `2j + r`'s PIOP are the cut session's bodies there (`msg_runCut`), which are the PIOP's bodies
from the views' source (`runCut_cutPiop`, `piopCut_bodies`): the zerocheck's and lincheck's rounds carrying that source's
masked values. The loop's frame re-runs those rounds and takes back exactly those values (`maskedOf_piop`), so the
request's values there are the source's plus the pads (`valsOf_at`), and the pads cancel.

* **`Masked.runCut_cutPiop`**, **`Masked.runCut_cutPiop_of_ret`**: the cut session stopped at a body `n` of rep
  `2j + r`'s PIOP sent there the PIOP's body `n`, where the earlier reps return (`runCut_cutL0`'s walk).
* **`piopCut_bodies`**: a returned PIOP from a source sent, from its log's start, `piopBody`: the PIOP's frames
  (`piopFr`) at the source's masked values, each with its squeeze, the pending bytes ahead of the first.
* **`Loop.swSrcE_hmk_cut`**: the views' source's masked values below `zp 64`, but `final_c`'s, are `q.vals + pads.h`.
  `final_c`'s is read from the inner proof's round 0, past the PIOP, Ligerito and the replay, and is not stated here.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **The cut session in rep `2j + r`'s PIOP**: as `runCut_cutL0`, where each rep `k < 2j + r` returns from any history
at its first round (with rep `0`'s claim points `ptsOpen`, `hprev`) and rep `2j + r`'s PIOP returns from a history `l`
at its first round with log `l₁` (`hpiop`), the cut session stopped at `n`, for `n` from rep `2j + r`'s first round
below `|l₁|` and `n ≤ refK x`, sent at `n` what that PIOP sent there. -/
theorem runCut_cutPiop {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
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
    {res} {tr₁ : _root_.Flock.Transcript} {l₁ : Array Msg}
    (hpiop : ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) (refK x))
      (FlockDeploy.tabsOf (stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
      ((src j.val).rep r.val) (FlockComplete.flat cms[j.val]!.cap) ic.root extras (priorE c j.val r.val)
      (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
      (.ok (res, tr₁), l₁))
    {n : ℕ} (hn0 : repBase x (2 * j.val + r.val) ≤ n) (hn1 : n < l₁.size) (hnN : n ≤ refK x) :
    (runCut x c n spec src).2[n]! = l₁[n]! := by
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
  refine Claims.ZKServer.Loop.forIn_bind_err (fun L => L[n]! = l₁[n]!)
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
  · -- rep `2j + r` stops at `n` inside its PIOP
    rintro ⟨streams, proofs⟩ l₀ ⟨hl₀, hpts⟩
    have hp := hpri (2 * j.val + r.val) streams hpts
    have e2 : (2 * j.val + r.val) / 2 = j.val := by omega
    have e3 : (2 * j.val + r.val) % 2 = r.val := by omega
    simp only [e2, e3] at hp
    obtain ⟨bs, hbs, hp0⟩ := FlockComplete.CutPrefix.go_of_shift
      (FlockComplete.CutPrefix.piopCut_shift _ _ _ _ _ _ _ _ _ _ _) (hl.trans hl₀.symm) hpiop
    have hsz : (l₀ ++ bs).size = l₁.size := by
      rw [hbs, Array.size_append, Array.size_append, hl, hl₀]
    obtain ⟨e, L, hL, -, hLi⟩ := FlockComplete.CutPrefix.prefix_of_agree'
      (FlockComplete.CutPrefix.piopCut_cut_agree _ hnN _ _ _ _ _ _ _ _ _) (by omega) hp0 (by omega)
    have hR : ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) n) hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch r.val (spec.domain spec.tables[j.val]! r.val).toUTF8
        cms[j.val]! extras ic ((src j.val).rep r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l₀ = (.error e, L) := by
      unfold FlockComplete.repCut
      exact FlockComplete.LigLock.bind_go_err hL
    refine ⟨e, L, ?_, ?_⟩
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
      try simp only [StateT.run] at hR ⊢
      rw [hR]
    · rw [hLi n (by omega), hbs]
      exact Claims.ZKServer.Loop.getElem!_append_shift (hl₀.trans hl.symm) (by omega)

/-- **The cut session in rep `2j + r`'s PIOP, from the earlier reps' returns alone**: `runCut_cutPiop` with rep `0`'s
claim points at the opened coins read off its return (`repCut_points`). -/
theorem runCut_cutPiop_of_ret {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
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
    {res} {tr₁ : _root_.Flock.Transcript} {l₁ : Array Msg}
    (hpiop : ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) (refK x))
      (FlockDeploy.tabsOf (stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
      ((src j.val).rep r.val) (FlockComplete.flat cms[j.val]!.cap) ic.root extras (priorE c j.val r.val)
      (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
      (.ok (res, tr₁), l₁))
    {n : ℕ} (hn0 : repBase x (2 * j.val + r.val) ≤ n) (hn1 : n < l₁.size) (hnN : n ≤ refK x) :
    (runCut x c n spec src).2[n]! = l₁[n]! := by
  refine runCut_cutPiop hx c src hsx hs hcms j r hexts (fun k hk extras' hxe ic' hic' l' hl' => ?_) hext hic hl
    hpiop hn0 hn1 hnN
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

/-- **The PIOP's body `jj`** after the pending bytes `P`: its frames (`piopFr`) and its squeeze, `P` ahead of round
`0`'s. -/
def piopBody (P : List UInt8) (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout)
    (mk : ℕ → F128) (jj : ℕ) : Claims.Msg :=
  (if jj = 0 then P else []) ++
    (encs (piopFr m kLog digest cap root lay mk jj) ++ (Frame.sq (piopK jj) (piopN m jj)).enc)

/-- **The cut's zerocheck and lincheck bodies are the PIOP's bodies** (`zcBodies`, then `lcBodies` after the
zerocheck's finals). -/
theorem piopBodies_eq (P : List UInt8) {m kLog : ℕ} (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) :
    zcBodies m digest cap root lay mk P ++ lcBodies kLog lay mk (encs [.one (mk lay.fa), .one (mk lay.fb)]) =
      (List.range (piopNR m kLog)).map (piopBody P m kLog digest cap root lay mk) := by
  have hsplit : List.range (piopNR m kLog) = List.range' 0 1 ++ (List.range' 1 1 ++ (List.range' 2 1 ++
      (List.range' 3 (m - 6) ++ (List.range' (3 + (m - 6)) 1 ++ (List.range' (4 + (m - 6)) 1 ++
      (List.range' (5 + (m - 6)) (kLog - 6) ++ List.range' (5 + (m - 6) + (kLog - 6)) 1)))))) := by
    rw [range'_join' (5 + (m - 6)) (kLog - 6) _ _ rfl, range'_join' (4 + (m - 6)) 1 _ (5 + (m - 6)) (by omega),
      range'_join' (3 + (m - 6)) 1 _ (4 + (m - 6)) (by omega), range'_join' 3 (m - 6) _ (3 + (m - 6)) rfl,
      range'_join' 2 1 _ 3 rfl, range'_join' 1 1 _ 2 rfl, range'_join' 0 1 _ 1 rfl, List.range_eq_range']
    congr 1
    unfold piopNR
    omega
  have hb0 : piopBody P m kLog digest cap root lay mk 0 =
      P ++ encs (zcBind digest cap root) ++ (Frame.sq 2 6).enc := by
    simp [piopBody, piopFr, piopK, piopN, zcBind, _root_.Flock.Firewall.Frame.LABEL_ZC]
  have hb1 : piopBody P m kLog digest cap root lay mk 1 = (Frame.sq 2 (m - 13)).enc := by
    simp [piopBody, piopFr, piopK, piopN, encs]
  have hb2 : piopBody P m kLog digest cap root lay mk 2 =
      encs [.vec ((List.range 64).map fun i => mk (lay.r1ab i)), .vec ((List.range 64).map fun i => mk (lay.r1c i))] ++
        (Frame.sq 1 1).enc := by
    simp [piopBody, piopFr, piopK, piopN]
  have hml : (List.range' 3 (m - 6)).map (piopBody P m kLog digest cap root lay mk) =
      (List.range (m - 6)).map fun i => encs [.one (mk (lay.ml i 0)), .one (mk (lay.ml i 1))] ++ (Frame.sq 1 1).enc := by
    rw [List.range'_eq_map_range, List.map_map]
    refine List.map_congr_left fun i hi => ?_
    rw [List.mem_range] at hi
    have h0 : ¬ 3 + i = 0 := by omega
    have h1 : ¬ 3 + i = 1 := by omega
    have h2 : ¬ 3 + i = 2 := by omega
    have h3 : 3 + i < 3 + (m - 6) := by omega
    have hK : ¬ 3 + i ≤ 1 := by omega
    simp only [Function.comp_apply, piopBody, piopFr, piopK, piopN, if_neg h0, if_neg h1, if_neg h2, if_pos h3,
      if_neg hK, List.nil_append, Nat.add_sub_cancel_left]
  have hfin : piopBody P m kLog digest cap root lay mk (3 + (m - 6)) =
      encs [.one (mk lay.fa), .one (mk lay.fb)] ++ (Frame.label "flock-lincheck-v0".toUTF8).enc ++
        (Frame.sq 1 1).enc := by
    have h0 : ¬ 3 + (m - 6) = 0 := by omega
    have h1 : ¬ 3 + (m - 6) = 1 := by omega
    have h2 : ¬ 3 + (m - 6) = 2 := by omega
    have h3 : ¬ 3 + (m - 6) < 3 + (m - 6) := by omega
    have hK : ¬ 3 + (m - 6) ≤ 1 := by omega
    simp only [piopBody, piopFr, piopK, piopN, if_neg h0, if_neg h1, if_neg h2, if_neg h3, if_neg hK, if_true,
      List.nil_append]
    simp [encs, List.append_assoc]
  have hbeta : piopBody P m kLog digest cap root lay mk (4 + (m - 6)) = (Frame.sq 1 1).enc := by
    have h0 : ¬ 4 + (m - 6) = 0 := by omega
    have h1 : ¬ 4 + (m - 6) = 1 := by omega
    have h2 : ¬ 4 + (m - 6) = 2 := by omega
    have h3 : ¬ 4 + (m - 6) < 3 + (m - 6) := by omega
    have h4 : ¬ 4 + (m - 6) = 3 + (m - 6) := by omega
    have hK : ¬ 4 + (m - 6) ≤ 1 := by omega
    simp only [piopBody, piopFr, piopK, piopN, if_neg h0, if_neg h1, if_neg h2, if_neg h3, if_neg h4, if_neg hK,
      if_true, List.nil_append]
    simp [encs]
  have hlc : (List.range' (5 + (m - 6)) (kLog - 6)).map (piopBody P m kLog digest cap root lay mk) =
      (List.range (kLog - 6)).map fun j =>
        encs [.one (mk (lay.lc j 0)), .one (mk (lay.lc j 1))] ++ (Frame.sq 1 1).enc := by
    rw [List.range'_eq_map_range, List.map_map]
    refine List.map_congr_left fun i hi => ?_
    rw [List.mem_range] at hi
    have h0 : ¬ 5 + (m - 6) + i = 0 := by omega
    have h1 : ¬ 5 + (m - 6) + i = 1 := by omega
    have h2 : ¬ 5 + (m - 6) + i = 2 := by omega
    have h3 : ¬ 5 + (m - 6) + i < 3 + (m - 6) := by omega
    have h4 : ¬ 5 + (m - 6) + i = 3 + (m - 6) := by omega
    have h5 : ¬ 5 + (m - 6) + i = 4 + (m - 6) := by omega
    have h6 : 5 + (m - 6) + i < 5 + (m - 6) + (kLog - 6) := by omega
    have hK : ¬ 5 + (m - 6) + i ≤ 1 := by omega
    simp only [Function.comp_apply, piopBody, piopFr, piopK, piopN, if_neg h0, if_neg h1, if_neg h2, if_neg h3,
      if_neg h4, if_neg h5, if_pos h6, if_neg hK, List.nil_append, Nat.add_sub_cancel_left]
  have hzp : piopBody P m kLog digest cap root lay mk (5 + (m - 6) + (kLog - 6)) =
      encs [.vec ((List.range 64).map fun i => mk (lay.zp i))] ++ (Frame.sq 1 1).enc := by
    have h0 : ¬ 5 + (m - 6) + (kLog - 6) = 0 := by omega
    have h1 : ¬ 5 + (m - 6) + (kLog - 6) = 1 := by omega
    have h2 : ¬ 5 + (m - 6) + (kLog - 6) = 2 := by omega
    have h3 : ¬ 5 + (m - 6) + (kLog - 6) < 3 + (m - 6) := by omega
    have h4 : ¬ 5 + (m - 6) + (kLog - 6) = 3 + (m - 6) := by omega
    have h5 : ¬ 5 + (m - 6) + (kLog - 6) = 4 + (m - 6) := by omega
    have h6 : ¬ 5 + (m - 6) + (kLog - 6) < 5 + (m - 6) + (kLog - 6) := by omega
    have hK : ¬ 5 + (m - 6) + (kLog - 6) ≤ 1 := by omega
    simp only [piopBody, piopFr, piopK, piopN, if_neg h0, if_neg h1, if_neg h2, if_neg h3, if_neg h4, if_neg h5,
      if_neg h6, if_neg hK, List.nil_append]
  rw [hsplit]
  simp only [List.map_append, List.range'_one, List.map_cons, List.map_nil, hb0, hb1, hb2, hml, hfin, hbeta, hlc,
    hzp]
  unfold zcBodies lcBodies
  simp only [List.cons_append, List.nil_append, List.append_assoc]

/-- A log's body `|l| + jj` where it is `l`, then the bodies `f 0, …`, then any more. -/
theorem getElem!_of_toList_mid {L l : Array Claims.Msg} {N : ℕ} {f : ℕ → Claims.Msg} {rest : List Claims.Msg}
    (h : L.toList = l.toList ++ (List.range N).map f ++ rest) {jj : ℕ} (hj : jj < N) : L[l.size + jj]! = f jj := by
  have hs : l.size + jj < L.size := by
    have e := congrArg List.length h
    simp only [Array.length_toList, List.length_append, List.length_map, List.length_range] at e
    omega
  have e : L.toList[l.size + jj]? = some (f jj) := by
    rw [h, List.append_assoc, List.getElem?_append_right (by simp), Array.length_toList, Nat.add_sub_cancel_left,
      List.getElem?_append_left (by simpa using hj), List.getElem?_map, List.getElem?_range hj]
    rfl
  rw [Array.getElem?_toList] at e
  rw [getElem!_pos L _ hs]
  exact Option.some.inj ((Array.getElem?_eq_getElem hs).symm.trans e)

/-- **A returned PIOP from a source sent the PIOP's bodies** (`piopBody`) from its log's start, at the source's masked
values, the binding `st.digest`, `cap` and `root`, after the pending bytes it started from. -/
theorem piopCut_bodies {coin : ℕ → ℕ → F128} {n : ℕ} {st : _root_.Flock.Setup} {lay : _root_.Flock.Zk.PadLayout}
    {src : FlockComplete.RepSrc} {cap root : ByteArray} {extras : Array _root_.Flock.Claim}
    {prior : Array (List F128)} {words : List ℕ} {t₀ : _root_.Flock.Transcript} {l : Array Claims.Msg} {res t₃ l₃}
    (hm : 14 ≤ st.m)
    (h : ((FlockComplete.piopCut (FlockComplete.asksCut coin n) st lay src cap root extras prior words).run t₀).go l =
      (.ok (res, t₃), l₃)) {jj : ℕ} (hj : jj < piopNR st.m st.fold.kLog) :
    l₃[l.size + jj]! = piopBody t₀.pending.data.toList st.m st.fold.kLog st.digest cap root lay src.masked jj := by
  obtain ⟨zp, lp, ss, T, sws⟩ := res
  obtain ⟨-, hl₃, hn₃⟩ := FlockComplete.piopCut_at hm h default
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, h₃⟩ := FlockComplete.piopCut_split h
  obtain ⟨res₁, t₁', L₁, e₁, hL₁, hP₁⟩ :=
    zerocheckCut_bodies coin n hm st.digest cap root lay src.masked t₀ l (by omega)
  rw [h₁] at e₁
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₁
  obtain ⟨⟨-, rfl⟩, rfl⟩ := e₁
  have hs₁ : l₁.size = l.size + 3 + (st.m - 6) := by
    have e := congrArg List.length hL₁
    simp only [Array.length_toList, List.length_append, zcBodies, List.length_cons, List.length_map,
      List.length_range] at e
    omega
  obtain ⟨res₂, t₂', L₂, e₂, hL₂, -⟩ := lincheckCut_bodies coin n st.fold st.m lay src.masked zc t₁ l₁ (by omega)
    (FlockComplete.RankAt.lincheckCut_fold coin n h₂)
  rw [h₂] at e₂
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₂
  obtain ⟨⟨-, rfl⟩, rfl⟩ := e₂
  obtain ⟨bs, hbs, -⟩ := FlockComplete.CutPrefix.go_of_shift
    (FlockComplete.CutPrefix.ringSwitchCut_shift coin n _ _ _) rfl h₃
  have hL : l₃.toList = l.toList ++
      (List.range (piopNR st.m st.fold.kLog)).map
        (piopBody t₀.pending.data.toList st.m st.fold.kLog st.digest cap root lay src.masked) ++ bs.toList := by
    rw [hbs, Array.toList_append, hL₂, hL₁, hP₁, ← piopBodies_eq]
    simp only [List.append_assoc]
  exact getElem!_of_toList_mid hL hj

/-- A PIOP round's frames past round `0` carry no domain frame. -/
theorem piopFr_noDom {m kLog : ℕ} {digest cap root : ByteArray} {lay : _root_.Flock.Zk.PadLayout} {mk : ℕ → F128}
    {jj : ℕ} (h : jj ≠ 0) : ∀ f ∈ piopFr m kLog digest cap root lay mk jj, noDom f := by
  intro f hf
  unfold piopFr at hf
  rw [if_neg h] at hf
  split_ifs at hf <;> simp only [List.mem_cons, List.not_mem_nil, or_false] at hf <;>
    rcases hf with rfl | rfl | rfl <;> exact True.intro

/-- **The PIOP's round messages without the stream's domain frame**: the round's frames and squeeze. -/
theorem dropDom_piopBody (dom : ByteArray) (hd : dom.size < 2 ^ 64) (m kLog : ℕ) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (jj : ℕ) :
    dropDom (0 :: piopBody (_root_.Flock.Transcript.init #[] dom).pending.data.toList m kLog digest cap root lay mk
        jj) =
      0 :: (encs (piopFr m kLog digest cap root lay mk jj) ++ (Frame.sq (piopK jj) (piopN m jj)).enc) := by
  by_cases h0 : jj = 0
  · subst h0
    have e := dropDom_dom dom (piopFr m kLog digest cap root lay mk 0) ((Frame.sq (piopK 0) (piopN m 0)).enc) hd
    rw [← e]
    congr 2
    simp [piopBody, FlockSoundness.Refine.init_enc, encs]
  · have e : piopBody (_root_.Flock.Transcript.init #[] dom).pending.data.toList m kLog digest cap root lay mk jj =
        encs (piopFr m kLog digest cap root lay mk jj) ++ (Frame.sq (piopK jj) (piopN m jj)).enc := by
      simp only [piopBody, if_neg h0, List.nil_append]
    rw [e]
    exact dropDom_noDom _ _ _ (piopFr_noDom h0)

/-- **The PIOP's coin counts** in the plan's order. -/
theorem piopN_list {m kLog jj : ℕ} (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hj : jj < piopNR m kLog) :
    ((6 :: (m - 13) :: 1 :: (List.replicate (m - 6) 1 ++ 1 :: 1 :: (List.replicate (kLog - 6) 1 ++ [1])))[jj]?).getD
      0 = piopN m jj := by
  have hT : (1 :: (List.replicate (m - 6) 1 ++ 1 :: 1 :: (List.replicate (kLog - 6) 1 ++ [1]))) =
      List.replicate (piopNR m kLog - 2) 1 := by
    rw [List.eq_replicate_iff]
    constructor
    · simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil]
      unfold piopNR
      omega
    · intro b hb
      simp only [List.mem_cons, List.mem_append, List.mem_replicate, List.not_mem_nil, or_false] at hb
      rcases hb with h | ⟨-, h⟩ | h | h | ⟨-, h⟩ | h <;> exact h
  rw [hT]
  match jj, hj with
  | 0, _ => rfl
  | 1, _ => rfl
  | jj + 2, hj =>
    rw [List.getElem?_cons_succ, List.getElem?_cons_succ, List.getElem?_replicate, if_pos (by omega)]
    simp [piopN]

/-- **The masked value at pad `i`'s absorbed position** below `zp 64`, but `final_c`'s, where the masked values are
`mk` at the absorbed order's pads. -/
theorem getElem!_absPos {lay : _root_.Flock.Zk.PadLayout} {M : Array F128} {mk : ℕ → F128} {tail : List F128}
    (h : M.toList = (List.range (lay.fa + 2) ++ List.range' (lay.fa + 3) (2 * lay.nLc + 64)).map mk ++ tail)
    {i : ℕ} (hi : i < lay.zp 64) (hne : i ≠ lay.fc) : M[absPos lay i]! = mk i := by
  have hfc : lay.fc = lay.fa + 2 := rfl
  have hzp : lay.zp 64 = lay.fa + 3 + 2 * lay.nLc + 64 := by
    show lay.fc + 1 + 2 * lay.nLc + 0 + 64 = _
    omega
  have hlen := congrArg List.length h
  simp only [Array.length_toList, List.length_append, List.length_map, List.length_range,
    List.length_range'] at hlen
  have hlt₀ : absPos lay i < lay.fa + 2 + (2 * lay.nLc + 64) := by unfold absPos; split_ifs <;> omega
  have hlt : absPos lay i < M.size := by omega
  have hlt₁ : absPos lay i <
      ((List.range (lay.fa + 2) ++ List.range' (lay.fa + 3) (2 * lay.nLc + 64)).map mk).length := by
    simp only [List.length_map, List.length_append, List.length_range, List.length_range']
    exact hlt₀
  have e : M.toList[absPos lay i]? = some (mk i) := by
    rw [h, List.getElem?_append_left hlt₁, List.getElem?_map]
    unfold absPos
    split_ifs with h1
    · rw [List.getElem?_append_left (by simpa using h1), List.getElem?_range h1]
      rfl
    · rw [List.getElem?_append_right (by simp only [List.length_range]; omega), List.length_range,
        List.range'_eq_map_range, List.getElem?_map, List.getElem?_range (by omega)]
      exact congrArg (fun v => some (mk v)) (show lay.fa + 3 + (i - 1 - (lay.fa + 2)) = i by omega)
  rw [Array.getElem?_toList] at e
  rw [getElem!_pos M _ hlt]
  exact Option.some.inj ((Array.getElem?_eq_getElem hlt).symm.trans e)

/-- A coin other than `1` plus `1` is not zero. -/
theorem isZero_add_one {a : F128} (h : a ≠ 1) : (a + 1).isZero = false := by
  cases hz : (a + 1).isZero
  · rfl
  · exfalso
    apply h
    have e := add_pad_pad a 1
    rw [FlockLevel3.F128.isZero_iff _ hz] at e
    rw [← e]
    rfl

namespace Loop

open Claims Claims.ZKMal
open _root_.Flock (Stmt SessionSpec)

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

set_option maxHeartbeats 4000000 in
/-- **`swSrcE_hmk_cut`: the views' masked values below `zp 64` are the request's plus its pads, from the history**: at
(c.3)'s context on the loop's tables, where the earlier reps return from the views' source cut at the opened coins and
table `j`'s rep `r`'s PIOP returns (`Masked.runCut_cutPiop_of_ret`'s premises, `Loop.swSrcE_l0_live`'s), every masked
value of the views' source below `zp 64` but `final_c`'s is the request's value there plus its pad. The history's rounds
of the rep's PIOP are the cut session's bodies (`msg_runCut`), the PIOP's from the views' source (`runCut_cutPiop_of_ret`,
`piopCut_bodies`); the loop's frame takes back the masked values they carry (`maskedOf_piop`), so the request's value at
pad `i` is the masked value plus pad `i` (`valsOf_at`). `hcap` and `hroot` bound the zerocheck binding's capacity and
pads root, which the frame reads as byte frames. -/
theorem swSrcE_hmk_cut (vs : finS.SwVals)
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
    {res} {tr₁ : _root_.Flock.Transcript} {l₁ : Array Msg}
    (hpiop : ((FlockComplete.piopCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
      (FlockComplete.flat cms[j.val]!.cap) ic.root extras (Masked.priorE (finCoins x ops) j.val r.val)
      (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
      (.ok (res, tr₁), l₁))
    (hcap : (FlockComplete.flat cms[j.val]!.cap).size < 2 ^ 64) (hroot : ic.root.size < 2 ^ 64) :
    ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
      i ≠ (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc →
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := by
  intro i hi hne
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx := Masked.live_of hsc hR
  have hj := Masked.lt_of_jOf j
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hq : 2 * j.val + r.val < 2 * spec.tables.size := by rw [hspec]; omega
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j.val + r.val) / 2 = j.val := by omega
  have hmod : (2 * j.val + r.val) % 2 = r.val := by omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) :=
    Masked.tabsOf_stAt x j
  have hmS : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m = (Masked.stAt x j).m := by rw [htab]; rfl
  have hkS : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.fold.kLog = (Masked.stAt x j).c.kLog := by
    rw [htab]; rfl
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
  have hnM : lay.nMlv = (Masked.stAt x j).m - 6 := by
    have e := start_nMlv hds
    rwa [hst] at e
  rw [hst] at hlay
  have hnL : lay.nLc = (Masked.stAt x j).c.kLog - 6 := by rw [hlay]; rfl
  have hlayG : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 = lay := by
    rw [hlay, hts, htbsch]; rfl
  have hlayP : _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0 = lay := by
    rw [hlay, htab]; rfl
  rw [hlayG] at hi hne
  -- the PIOP's log
  have hm14' : 14 ≤ (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m := by rw [hmS]; omega
  obtain ⟨-, hl₁, hn₁⟩ := FlockComplete.piopCut_at hm14' hpiop default
  rw [hmS, hkS, hl] at hl₁
  have h1 : 1 ≤ Masked.repBase x (2 * j.val + r.val) := by
    rw [Masked.repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hlen : ops.length = refK x := hF.1
  -- the history's rounds of the PIOP are its bodies
  have hlast : (LeanRelease.ofLoopLink L₀ src (realOf src)).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE _ x
  have hsrc : Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
        (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) := rfl
  have hd := (sizesOk_at (sizesOk_of_serves hServe s0 spec _ hsx) hq2).2.1
  rw [hjj, hmod] at hd
  have hbody : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      ((ops.getD (Masked.repBase x (2 * j.val + r.val) + jj) ([], [])).1).tail =
        encs (piopFr (Masked.stAt x j).m (Masked.stAt x j).c.kLog (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.digest
          (FlockComplete.flat cms[j.val]!.cap) ic.root lay
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked jj) ++
          (Frame.sq (piopK jj) (piopN (Masked.stAt x j).m jj)).enc := by
    intro jj hjj'
    have hNR : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog =
        6 + ((Masked.stAt x j).m - 6) + ((Masked.stAt x j).c.kLog - 6) := rfl
    have hp : Masked.repBase x (2 * j.val + r.val) + jj < ops.length := by omega
    obtain ⟨s', spec', hsx', hm⟩ := msg_runCut vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w
      hE hc hlast ν t κ m₀ ρ₀ ops hF.1 hF.2.1 hF.2.2.1 hF.2.2.2.2.2 (Masked.repBase x (2 * j.val + r.val) + jj + 1)
      (by omega) (by omega)
    rw [hsx] at hsx'
    cases hsx'
    have hW1 := Masked.runCut_cutPiop_of_ret hx (finCoins x ops)
      (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t) hsx hs hcms j r hexts hprev hext hic hl hpiop
      (n := Masked.repBase x (2 * j.val + r.val) + jj) (Nat.le_add_right _ _) (by omega) (by omega)
    have hW2 := piopCut_bodies hm14' hpiop (jj := jj) (by rw [hmS, hkS]; exact hjj')
    rw [hl, hlayP, hmS, hkS] at hW2
    rw [← hsrc, Nat.add_sub_cancel, hW1, hW2, dropDom_piopBody _ hd] at hm
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hp, Option.getD_some]
    have e : (ops[Masked.repBase x (2 * j.val + r.val) + jj]'hp) =
        ((m₀, ρ₀) :: ops)[Masked.repBase x (2 * j.val + r.val) + jj + 1]'(by simp only [List.length_cons]; omega) :=
      rfl
    rw [e, hm]
    rfl
  -- the loop's coin counts there
  have hpl := Count.piop_list (kLog := (Masked.stAt x j).c.kLog) (by omega : 14 ≤ (Masked.stAt x j).m)
  have hpL := congrArg List.length hpl
  simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil] at hpL
  have hNR : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      loopNr x (Masked.repBase x (2 * j.val + r.val) + jj) = piopN (Masked.stAt x j).m jj := by
    intro jj hjj'
    unfold piopNR at hjj'
    rw [Masked.repBase_eq hsx, loopNr_plan hsx hq2 hz (by rw [hst]; exact hk6) (i := jj)
      (by unfold FlockComplete.piopRoundsZk; rw [hst]; omega),
      Count.plan_get hz (by rw [List.length_append, hst, hpL]; omega), hst,
      List.getElem?_append_left (by rw [hpL]; omega), hpl]
    exact piopN_list (by omega) hk6 (by unfold piopNR; omega)
  have hwords : ∀ jj, jj + 1 < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog →
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + jj)).size =
        piopN (Masked.stAt x j).m jj := by
    intro jj hjj'
    have hp : Masked.repBase x (2 * j.val + r.val) + jj < ops.length := by unfold piopNR at hjj'; omega
    have hs' := wordsAt_size x ops hp le_rfl
    rw [List.take_length, hNR jj (by omega)] at hs'
    rw [hs']
    exact Nat.min_eq_left (by unfold piopN _root_.Flock.Firewall.Frames.COIN_MAX; split_ifs <;> omega)
  -- the zerocheck's outer coins are not `1`
  have hA := admits_of_finCtx L₀ src (realOf src) wk hServe hF
  have hnr := Masked.not_refused_of_admits hx hA (Masked.innerSlotE x j r + 2)
  have hout : ∀ e < (Masked.stAt x j).m - 13,
      ((wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + 1))[e]! +
        1).isZero = false := by
    intro e he
    have h1' : loopNr x (Masked.repBase x (2 * j.val + r.val) + 1) = (Masked.stAt x j).m - 13 :=
      hNR 1 (by unfold piopNR; omega)
    rw [words_slot hF.1 (by omega) (by rw [h1']; exact he) (by omega)]
    exact isZero_add_one (Masked.outer_ne_one hnr hsx j r (by omega) e he)
  -- the rounds the loop released
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  have hR0 : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
      (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hrk, repRounds_true]
    simp only [hz]
    rw [hst]
    have hA : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
        (Masked.stAt x j).m + ((Masked.stAt x j).c.kLog - 6) := by
      unfold piopNR
      omega
    exact le_trans (le_trans (le_trans hA (Nat.le_add_right _ _)) (Nat.le_add_right _ _)) (Nat.le_add_right _ _)
  have hstream : streamAt (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val) + 1 =
      Masked.repBase x (2 * j.val + r.val) := by
    rw [Masked.repBase_eq hsx]
    unfold streamAt roundsOf FlockComplete.repStartZk
    rw [take_map_range _ (by omega)]
    omega
  have hrs : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      (seenRounds x ((m₀, ρ₀) :: ops) (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val))[jj]? =
        some (ByteArray.mk (encs (piopFr (Masked.stAt x j).m (Masked.stAt x j).c.kLog
            (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.digest (FlockComplete.flat cms[j.val]!.cap) ic.root lay
            ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked jj) ++
          (Frame.sq (piopK jj) (piopN (Masked.stAt x j).m jj)).enc).toArray, piopN (Masked.stAt x j).m jj,
          wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + jj)) := by
    intro jj hjj'
    unfold seenRounds
    rw [List.getElem?_map, List.getElem?_range (by omega)]
    refine congrArg some ?_
    dsimp only
    rw [List.tail_cons, show streamAt (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val) + jj + 1 =
      Masked.repBase x (2 * j.val + r.val) + jj by omega, hbody jj hjj', hNR jj hjj']
  -- the masked values the loop's frame took, and the request's values
  have hdg : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.digest.size < 2 ^ 64 := by
    rw [htab]
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  obtain ⟨tail, htail⟩ := maskedOf_piop lay (L₀.draws (ν, t) (2 * j.val + r.val)) none (Masked.stAt x j).m
    (Masked.stAt x j).c.kLog (by omega) hk6 (lt_of_le_of_lt hm35 (by norm_num)) hnM hnL _ _ _ hdg hcap hroot
    ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked
    (fun jj => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + jj)) _ hrs
    hwords hout
  have hM := congrArg List.length htail
  simp only [Array.length_toList, List.length_append, List.length_map, List.length_range,
    List.length_range'] at hM
  cases hq'
  dsimp only
  rw [valsOf_at lay _ none _ _ hi hne (by omega), getElem!_absPos htail hi hne,
    show (_root_.Flock.Firewall.Finish.padsOf (_root_.Flock.Firewall.Frame.Draws.ofBytes lay
      (L₀.draws (ν, t) (2 * j.val + r.val)))).toC.h =
      (_root_.Flock.Firewall.Frame.Draws.ofBytes lay (L₀.draws (ν, t) (2 * j.val + r.val))).h from rfl,
    add_pad_pad]

end Loop

end Claims.ZKServer
