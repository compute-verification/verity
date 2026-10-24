import Proofs.Flock.Claims.ZK.L0Pin
import Proofs.Flock.Claims.ZK.L0PinCoins
import Proofs.Flock.SessionLoop.FinRounds
import Proofs.Flock.SessionLoop.FinCommits

/-!
# The cut session in rep `2j + r`'s level 0 rounds

`L0Pin`'s `repCut_cutL0` puts level 0's bodies of a rep whose PIOP returns at the coins from the PIOP's end, and
`L0PinCoins`'s `piopCut_lig` puts that end at `ligP0` with the basis `ligBasis`. Here the whole cut session
(`Masked.runCut`), stopped at a round `n` of rep `2j + r`'s level 0 opening, sent there level 0's body `n − ligP0`:
the session runs its schedules, level 0's commitments and the link, then the reps before `2j + r` (each returning by
`n`, so as at the firewall's stop, `eq_of_agree`), then rep `2j + r`, whose PIOP returns as from any history of its size
(`piopCut_shift`) and whose opening stops at `n` (`repCut_cutL0`). `runCut_body` can't serve: it reads the view rep's
return, which is what this feeds.

* `getElem!_of_toList`: the stopped log's body at `|l₁| + j` is the `j`-th body sent after `l₁`.
* `padsTreeCut_go`: a pads commitment returns, sending nothing, from every history.
* **`runCut_cutL0`**: where the reps before `2j + r` return from their first round at the firewall's stop (with rep
  `0`'s claim points at the opened coins, `ptsOpen`) and rep `2j + r`'s PIOP returns, the cut session stopped at any
  `n` in `[ligP0, ligP0 + k₀ + 4)`, `n ≤ refK x`, sent at `n` level 0's body `n − ligP0` at the coins from `ligP0`, the
  PIOP's target and basis `ligBasis`, any table `t`, and the pads `padsL0` of the source.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- The stopped log's body at `|l₁| + j` is the `j`-th body sent after `l₁`. -/
theorem getElem!_of_toList {L l₁ : Array Msg} {f : ℕ → Msg} {j : ℕ}
    (h : L.toList = l₁.toList ++ (List.range (j + 1)).map f) : L[l₁.size + j]! = f j := by
  have hs : L.size = l₁.size + (j + 1) := by
    have e := congrArg List.length h
    simpa using e
  rw [getElem!_pos L _ (by omega), ← Array.getElem_toList, List.getElem_of_eq h,
    List.getElem_append_right (by simp)]
  simp

/-- A pads commitment returns, sending nothing, from every history (`padsTreeCut_ok`, `padsTreeCut_quiet`). -/
theorem padsTreeCut_go (N : ℕ) (src : FlockComplete.RepSrc) :
    ∃ ic : FlockComplete.InnerCommit, ∀ l : Array Msg, (FlockComplete.padsTreeCut hsE N src).go l = (.ok ic, l) := by
  obtain ⟨ic, h⟩ := FlockComplete.padsTreeCut_ok N src
  refine ⟨ic, fun l => ?_⟩
  rw [FlockComplete.CutEq.padsTreeCut_quiet, h]

/-- **The cut session in rep `2j + r`'s level 0 rounds**: at a live statement, the session's setup, schedules and
level 0's commitments from the sources `src`, where each rep `k < 2j + r` returns from any history at its first round,
answered by the opened coins up to the firewall's stop, with rep `0`'s claim points `ptsOpen` (`hprev`), and rep
`2j + r`'s PIOP returns from a history at its first round (`hpiop`), the cut session stopped at `n`, for `n` from `ligP0`
up to level 0's last body and `n ≤ refK x`, sent at `n` level 0's body `n − ligP0`: at the coins from `ligP0`, the
PIOP's transcript and target, level 0's cap, the basis `ligBasis`, any table `t`, and the pads `padsL0` of the source. -/
theorem runCut_cutL0 {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (src : ℕ → FlockComplete.TabSrc) {s0 : _}
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
    (t : Array _root_.Flock.F128) (extra : ℕ → ℕ → _root_.Flock.F128)
    {n : ℕ} (hn0 : ligP0 x j r ≤ n) (hn1 : n < ligP0 x j r + (zkTab x j).sch.k0 + 4) (hnN : n ≤ refK x) :
    (runCut x c n spec src).2[n]! =
      FlockComplete.LigLock.l0BodyAt (ligCoin c) (ligP0 x j r) tr₁.pending.data.toList res.2.2.2.1
        (FlockComplete.flat cms[j.val]!.cap) (zkTab x j).sch t (ligBasis x c w τ j r extras) (layE x j)
        (FlockComplete.LigLock.padsL0 (ligCoin c) (ligP0 x j r) (zkTab x j).sch t (ligBasis x c w τ j r extras)
          (layE x j) extra r.val ((src j.val).rep r.val)) r.val extra (fun _ _ => 0) (n - ligP0 x j r) := by
  obtain ⟨zp, lp, ss, T, sws⟩ := res
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    have := r.isLt
    omega
  obtain ⟨schs', hs', hschs⟩ := schedsPZk_live hx
  obtain rfl : schs' = schs := Except.ok.inj (hs'.symm.trans hs)
  have esch : schs[j.val]! = ((zkTab x j).sch, (zkTab x j).tPad) := hschs j
  obtain ⟨h25, -, hsm, hk0, -, ⟨hn7, hlv, -⟩, -⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  have hsn : (zkTab x j).sch.n ≠ 0 := by
    rw [hn7, hsm]
    omega
  -- the PIOP's end and basis, at the firewall's stop
  obtain ⟨hl₁, hP0N, hbasis⟩ := piopCut_lig hx w τ c (refK x) hsx j r hext _ (tabsOf_stAt x j) _ _ _ _ _ hl hpiop
  obtain ⟨bs₀, hbs₀, -⟩ := FlockComplete.CutPrefix.go_of_shift
    (FlockComplete.CutPrefix.piopCut_shift _ _ _ _ _ _ _ _ _ _ _) rfl hpiop
  have hbase : repBase x (2 * j.val + r.val) ≤ ligP0 x j r := by
    rw [← hl₁, hbs₀, Array.size_append, hl]
    omega
  have h1 : 1 ≤ repBase x (2 * j.val + r.val) := by
    rw [repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hmono : ∀ a b, a ≤ b → repBase x a ≤ repBase x b := fun a b hab => by
    rw [repBase_eq hsx, repBase_eq hsx]
    exact FlockComplete.ZkTables.repStart_mono _ _ hab
  have hlay : _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0 = layE x j := by
    rw [tabsOf_stAt]
    rfl
  -- rep `k`'s earlier claim points, read from the streams so far
  have hpri : ∀ (k : ℕ) (b : Array (Array _root_.Flock.Round) × Array ByteArray),
      (k % 2 = 1 → FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog b.1.back! = ptsOpen c (stsOf x)[k / 2]! (2 * (k / 2))) →
      FlockComplete.priorOf (FlockDeploy.tabsOf (stsOf x))[k / 2]! b.1 (k % 2) = priorE c (k / 2) (k % 2) := by
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
  refine Claims.ZKServer.Loop.forIn_bind_err (fun L => L[n]! = _)
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
    have hp := hpri k (streams, proofs) hpts
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
  · -- rep `2j + r` stops at `n` in level 0's opening
    rintro ⟨streams, proofs⟩ l₀ ⟨hl₀, hpts⟩
    have hp := hpri (2 * j.val + r.val) (streams, proofs) hpts
    have e2 : (2 * j.val + r.val) / 2 = j.val := by omega
    have e3 : (2 * j.val + r.val) % 2 = r.val := by omega
    simp only [e2, e3] at hp
    obtain ⟨bs, rfl, hp0⟩ := FlockComplete.CutPrefix.go_of_shift
      (FlockComplete.CutPrefix.piopCut_shift _ _ _ _ _ _ _ _ _ _ _) (hl.trans hl₀.symm) hpiop
    obtain ⟨hsz1, -, -⟩ := piopCut_lig hx w τ c (refK x) hsx j r hext _ (tabsOf_stAt x j) _ _ _ _ _ hl₀ hp0
    have hpn := FlockComplete.CutPrefix.eq_of_agree
      (FlockComplete.CutPrefix.piopCut_cut_agree _ hnN _ _ _ _ _ _ _ _ _) (by omega) hp0 (by omega)
    have hstop : (l₀ ++ bs).size + (n - ligP0 x j r) = n := by omega
    have hcs := FlockComplete.LigLock.repCut_cutL0 (ligCoin c) hsE (FlockDeploy.tabsOf (stsOf x))[j.val]!
      (zkTab x j).sch r.val (spec.domain spec.tables[j.val]! r.val).toUTF8 cms[j.val]! extras ic
      ((src j.val).rep r.val) (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]! t r.val
      extra (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8) l₀
      (j := n - ligP0 x j r) (by omega) hk0 hsn hlv (by rw [hstop]; exact hpn)
    obtain ⟨L, hL, hQ⟩ := hcs
    have hidx := getElem!_of_toList hQ
    rw [hstop] at hL hidx
    dsimp only at hidx
    rw [hsz1, hbasis, hlay] at hidx
    refine ⟨_, L, ?_, hidx⟩
    simp only [List.getElem_map, List.getElem_range]
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
    try simp only [StateT.run] at hL ⊢
    rw [hL]

end Masked
end Claims.ZKMal
