module

public import Proofs.Flock.ZkAny.RelayLoopCur

/-!
# The frame round's own rep one round on, past its schedule

`loopKept_round`'s `hcur` at a round of a rep past its schedule so far, the inner proof's, where the next round is the
same stream's:

* `roundPlace_own_of`: where `placeOf` places the found frame's next round at the inner proof's own, the readied frame
  is `placeOf`'s and the round's place the frame's own inner round (`Frame.innerSched` on the rounds the loop's finish
  computed), at the inner round its masking state is at.
* `answered_own0`: at the inner proof's round 0 the model's masking state has begun no inner round.
  `innerSched_zero`: the frame's own round 0 is the inner proof's label frame and a 16-byte `final_c'`.
* `round_unstop_own`: so a round the frame releases there, at a state alike the model's and at the model's
  configuration, leaves a state alike the loop's after it (`Loop.progOf_own_step`).
* `roundRun_step_own`, `roundRun_own`: at a `RoundRun` past the rep's schedule the model's coin step takes the
  round's coins, and where the next round is the same rep's the frame the session's next request finds is its run one
  round on, at the model's state for that round but for `stopped`, and keeps `SameInv` one round more of the rep.
* **`repJ_own`**: as `repJ_same`, where the round is past the rep's schedule: the model's coin step takes the round's
  coins, and where the next round is the same stream's the frame the session's next request finds keeps `SameInv` one
  round more of the rep, its opening ended.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Cfg Masker Place Req labelFrame)

set_option maxHeartbeats 1000000 in
/-- **Past the opening the readied frame is `placeOf`'s, at its own inner round**: at the session `s`, its table
`name` and its level 0 `l0`, where `placeOf` places the found frame's next round at the inner proof's own, the frame
`roundPlace` readies is `placeOf`'s, and the round's place is the frame's own inner round at its masking state's, on
the rounds the loop's finish computed. -/
theorem roundPlace_own_of {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hl : f = false) {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hs : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) {k : Nat} {q' : Rep}
    (hv : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok (.own, k, q')) :
    q = q' ∧ pl = .sched (_root_.Flock.Firewall.Frame.innerSched q'.cfg q'.st.m (R.innerOf l0.root r) n
      (q'.st.m.inner.getD 0)) := by
  have hfst : (p.coins.isEmpty && !p.st.begun) = false := (roundPlace_first hf h).symm.trans hl
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  repeat' (first
    | simp only [hs, hn, hl0, hfst, hv, bind, Except.bind, pure, Except.pure, Bool.false_eq_true, ↓reduceIte] at h
    | split at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h; exact ⟨rfl, rfl⟩)

/-- **The frame's own inner round 0** is the inner proof's label frame and a 16-byte `final_c'` between its
headers. -/
theorem innerSched_zero (c : Cfg) (s : Masker) (fin : Array ByteArray) (n : Nat) :
    ∃ w : ByteArray, w.size = 16 ∧ (_root_.Flock.Firewall.Frame.innerSched c s fin n 0).data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w ++ header 4 1 1).data.toList := by
  refine ⟨_root_.Flock.Firewall.Frame.wordBytes (_root_.Flock.Firewall.Frame.words (fin.getD 0 .empty)
    ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1), ?_, rfl⟩
  rw [← ByteArray.size_data, ← Array.length_toList, wordBytes_data, FlockSoundness.Refine.Frame.colBytes_length,
    Array.length_toList, Loop.words_size, Nat.mul_one]

open Loop LeanRelease in
/-- **At the inner proof's round 0 the model has begun no inner round**: where the round after `e0 :: h1` is a rep's
past its first and the placeholder's schedule so far on the rep's coins, its rounds framed, is as long as them, the
model's masking state for it has no inner round (`repState_inner`). -/
theorem answered_own0 (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q j : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) {c' : Cfg}
    {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1
      (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi))
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c') {sched' : Array (ByteArray × Nat)}
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((Array.range (j + 1)).map fun i =>
        wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok sched')
    (hS : sched'.size = j + 1) (hF : ∀ k, k < sched'.size → Sched.Framed (sched'.getD k (.empty, 0)))
    {m : Masker} (hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m) : m.inner = none := by
  have hle' := locate_snd_le (roundsOf spec sts) 0 q
  have hF' : (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) = .ok m := by
    unfold LeanRelease.answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer_same T L₀ real x ρ e0 h1 hx hq hj] at hF'
  obtain ⟨ha, -⟩ := bind_rankStop_ok hF'
  have hlen : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  have hw : coinsAt x (e0 :: h1) = wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) q := by
    obtain ⟨m0, r0⟩ := e0
    have hcr : coinsAt x ((m0, r0) :: h1) = readCoins x (h1.map fun p => ZKMal.CoinLayer.rd p.2) :=
      loop_coinsRead loopNr T L₀ x m0 r0 h1
    rw [hcr]
    exact readCoins_eq x _ hlen
  rw [hw, progOf_at x _ hx hlen hj hph, hj] at ha
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hs'
  obtain ⟨sched, hs⟩ := schedSoFar_pop_ok hs'
  have hg : ((Array.range (j + 1)).map fun i =>
      wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)).getD j #[] =
      wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) q := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by simp), Option.getD_some, Array.getElem_map,
      Array.getElem_range]
    congr 1
    omega
  rw [answer_congr _ _ hc.lay hc.order, ← hg] at ha
  have hcr : ((Array.range (j + 1)).map fun i =>
      wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)).size = j + 1 := by simp
  obtain rfl := schedSoFar_pop_eq hs hs' hcr (by omega)
  exact Sched.answer_inner (repState_inner c' t _ dom xs hi _ _ _ j hal hs (by omega)
    fun k hk => hF k (by omega)) ha

open Loop LeanRelease in
/-- **A round past the rep's schedule, at the model's state for it, leaves the loop's state after it**: as
`round_unstop`, where the placeholder's schedule so far on the rep's coins, its rounds framed, is no longer than them,
so that the round is the inner proof's (`progOf_own_step`): the frame's release of a device round against the
scheduled round `r`, at a state alike the model's and at the model's configuration, leaves a state alike the loop's
after the coins with any next coin, `r` at the inner proof's round 0 being the ZK server's own round 0 at some
`final_c'`. -/
theorem round_unstop_own (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (d : FlockDeploy.Coin) {s0 : _} {spec : SessionSpec}
    {sts : Array Stmt} {q j : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : h1.length = q + 1) (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) {c' : Cfg}
    {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1
      (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi))
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c') (hz : ZeroCfg c')
    (hH : ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)).rowH.size = c'.rowH.size)
    (hM : ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)).rowMu.size = c'.rowMu.size) (hrow : c'.rowH.size < 2 ^ 64)
    {sched' : Array (ByteArray × Nat)}
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((Array.range (j + 1)).map fun i =>
        wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok sched')
    (hle : sched'.size ≤ j + 1) (hF : ∀ k, k < sched'.size → Sched.Framed (sched'.getD k (.empty, 0)))
    {m : Masker} (hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m)
    {st st' : _root_.Flock.Firewall.Frame.State} (hu : unstop st.m = unstop m) {v r out : ByteArray} {vd : String}
    (hr0 : sched'.size = j + 1 → ∃ w : ByteArray, w.size = 16 ∧ r.data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w ++ header 4 1 1).data.toList)
    (h : _root_.Flock.Firewall.Frame.step ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) st
      (.round (loopNr x (q + 1)) v (.sched r)) = (.round out vd, st')) :
    unstop st'.m = unstop (progOf x ((h1.map fun p => ZKMal.CoinLayer.rd p.2) ++ [d])) := by
  have hF' : (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) = .ok m := by
    unfold LeanRelease.answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer_same T L₀ real x ρ e0 h1 hx hq hj] at hF'
  obtain ⟨ha, -⟩ := bind_rankStop_ok hF'
  have hlen : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  have hw : coinsAt x (e0 :: h1) = wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) q := by
    obtain ⟨m0, r0⟩ := e0
    have hcr : coinsAt x ((m0, r0) :: h1) = readCoins x (h1.map fun p => ZKMal.CoinLayer.rd p.2) :=
      loop_coinsRead loopNr T L₀ x m0 r0 h1
    rw [hcr]
    exact readCoins_eq x _ hlen
  rw [hw] at ha
  obtain ⟨why, hck⟩ := _root_.Flock.Firewall.Frame.check_of_step h
  rw [eq_unstop hu.symm] at hck
  rw [progOf_own_step x _ d hx hlen hph hc hz hH hM hrow hj hs' hle hF ha hck hr0]
  all_goals rfl

open Loop LeanRelease in
/-- **A rep's frame at the session's next round, at the model's state, past the rep's schedule**: as
`roundRun_step`, where the placeholder's schedule so far on the rep's coins, its rounds framed, is no longer than them
(`hle`) and the round is in the place of the scheduled round `rr`, the ZK server's own round 0 at some `final_c'` at
the inner proof's round 0 (`hr0`): the model's coin step takes the round's coins (`roundRun_fed`), and where the next
round is the same rep's (`hj`) the frame the session's next round finds is its run one round on, at the model's state
for the history with `(v, a)` but for `stopped` (`round_unstop_own`, `roundRun_carry`). -/
theorem roundRun_step_own (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q₀ j₀ : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hj₀ : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = j₀ + 1) {c' : Cfg}
    {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1
      (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi))
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c') (hz : ZeroCfg c')
    (hH : ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)).rowH.size = c'.rowH.size)
    (hM : ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)).rowMu.size = c'.rowMu.size) (hrow : c'.rowH.size < 2 ^ 64)
    {sched' : Array (ByteArray × Nat)}
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 % 2) dom xs hi
      ((Array.range (j₀ + 1)).map fun i =>
        wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q₀ - (j₀ + 1) + i + 1)) = .ok sched')
    (hle : sched'.size ≤ j₀ + 1) (hF : ∀ k, k < sched'.size → Sched.Framed (sched'.getD k (.empty, 0)))
    {m₀ : Masker} (hA₀ : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m₀)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray} {v : Msg} {e : Env}
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none) (hfr : (frameAt R sid i r).isSome)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} (hR : RepRun pre ev q)
    (hcq : q.cfg = (ofLoop T L₀ real).cfg x ρ (e0 :: h1)) (hu₀ : unstop q.st.m = unstop m₀) {rr : ByteArray}
    (hpl : pl' = .sched rr)
    (hr0 : sched'.size = j₀ + 1 → ∃ w : ByteArray, w.size = 16 ∧ rr.data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w ++ header 4 1 1).data.toList) :
    (ofLoop T L₀ real).fed x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = true ∧
      ∀ j, (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1 → ∀ m,
        (ofLoop T L₀ real).answeredR x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = .ok m →
        ∃ p, frameAt R₃ sid i r = some p ∧
          RepRun pre (ev ++ [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray),
            pl', _root_.Flock.Firewall.Frames.feedOf (loopNr x h1.length) a.data.toList)]) p ∧
          unstop p.st.m = unstop m := by
  have hu : unstop (_root_.Flock.Firewall.Frame.step q.cfg q.st (.round (loopNr x h1.length)
      (if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray)) pl')).2.m =
      unstop (progOf x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2)) := by
    obtain ⟨out, vd, hre, -⟩ := roundStep_out hrp ho
    revert hre
    rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st (.round (loopNr x h1.length)
      (if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray)) pl') with ⟨a', st'⟩
    intro hre
    simp only at hre
    subst hre
    rw [hcq, hpl, hq] at hres
    rw [List.map_append, List.map_cons, List.map_nil]
    exact round_unstop_own T L₀ real x ρ e0 h1 _ hx hq hj₀ hph hc hz hH hM hrow hs' hle hF hA₀ hu₀ hr0 hres
  refine ⟨roundRun_fed T L₀ real x ρ e0 h1 hr hreq hfr hrp ho hput ha hb hs hco hR hcq hu, fun j hj m hA => ?_⟩
  have hcfg : CfgAlike q.cfg ((ofLoop T L₀ real).cfg x ρ (e0 :: (h1 ++ [(v, a.data.toList)]))) := by
    rw [hcq, cfg_same T L₀ real x ρ e0 h1 hx hq hj (v, a.data.toList)]
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  exact roundRun_carry T L₀ real x ρ e0 h1 hx hj hr hreq hfr hrp ho hput ha hb hs hco hR hcfg hu hA

open Loop LeanRelease in
/-- **A rep's frame keeps `SameInv` from one round of the rep past its schedule to the next**: as `roundRun_same`,
where the frame's schedule on its coins (`sched`, of one shape with the placeholder's) ends before the round
(`hle`), so `placeOf` places it at the inner proof's own and the round is in the place of the frame's own inner round
(`roundPlace_own_of`), the ZK server's own round 0 at the inner proof's round 0, where the model has begun no inner round
(`answered_own0`, `innerSched_zero`): the model's coin step takes the round's coins (`roundRun_step_own`), and where
the next round is the same rep's the frame the session's next request finds keeps `SameInv` at the history with the
worker's message and the auditor's answer (`roundRun_step_own`, `roundRun_repInv`, `roundRun_coins`,
`roundRun_cfg`). -/
theorem roundRun_own (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q₀ j₀ : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hj₀ : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = j₀ + 1) {c' : Cfg}
    {t' : _root_.Flock.Firewall.Finish.Tab} {dom' : ByteArray} {xs' : Array (Array F128)} {hi' : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1
      (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t', dom', xs', hi'))
    (hrow : c'.rowH.size < 2 ^ 64) {sched' : Array (ByteArray × Nat)}
    (hs' : schedSoFar t' ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 % 2) dom' xs' hi'
      ((Array.range (j₀ + 1)).map fun i =>
        wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q₀ - (j₀ + 1) + i + 1)) = .ok sched')
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray} {v : Msg} {e : Env}
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {p : Rep} (hfa : frameAt R sid i r = some p) {s : Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} (hss : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    {t : _root_.Flock.Firewall.Finish.Tab} {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt}
    {c : Array F128} {r0 : ByteArray} {q0 : Nat} (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c))
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    (hI : SameInv (ofLoop T L₀ real) x ρ (e0 :: h1) (h1.map fun p => ZKMal.CoinLayer.rd p.2) q₀ j₀ t
      (s.spec.domain name r).toUTF8 xs hi pre ev p)
    {sched : Array (ByteArray × Nat)}
    (hsc : schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched)
    (hrs : Sched.RoundsAt sched sched') (hle : sched.size ≤ j₀ + 1)
    {m₀ : Masker} (hA₀ : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m₀) :
    (ofLoop T L₀ real).fed x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = true ∧
      ∀ j, (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1 →
        ∃ p₃, frameAt R₃ sid i r = some p₃ ∧
          SameInv (ofLoop T L₀ real) x ρ (e0 :: (h1 ++ [(v, a.data.toList)]))
            ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2) (q₀ + 1) (j₀ + 1) t
            (s.spec.domain name r).toUTF8 xs hi pre
            (ev ++ [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray), pl',
              _root_.Flock.Firewall.Frames.feedOf (loopNr x h1.length) a.data.toList)]) p₃ := by
  have hfr : (frameAt R sid i r).isSome := by
    rw [hfa]
    rfl
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  obtain ⟨hl, hRq, hcq, huq, hcw⟩ := sameInv_place hI hfa hrp
  have hsz : p.coins.size = j₀ + 1 := by rw [hI.coins, Array.size_map, Array.size_range]
  obtain ⟨-, P1, hat, hinv⟩ :=
    roundRun_repInv hr hreq hfr hrp ho hput ha hb hs hco hRq hf hl hss hn hl0 hst hlink hsalt htab hxs hI.inv hsc
  obtain ⟨q', hpo, -⟩ :=
    placeOf_sched R.salts l0 s p (s.spec.domain name r).toUTF8 hst hlink hsalt htab hxs hI.inv hsc
  have hpn : placeAt sched p.coins.size = (.own, 0) := placeAt_none (by rw [hsz]; exact hle)
  rw [hpn] at hpo
  obtain ⟨rfl, hpl⟩ := roundPlace_own_of hf hrp hl hss hn hl0 hpo
  have hq2 : (e0 :: h1).length = q₀ + 2 := by simp only [List.length_cons, hq]
  obtain ⟨hc, hH, hM, hz⟩ := ofLoop_cfg_place T L₀ real x ρ (e0 :: h1) hx hq2 (Prod.ext rfl hj₀) hph
  have hsz' := roundsAt_size hrs
  have hF : ∀ k, k < sched'.size → Sched.Framed (sched'.getD k (.empty, 0)) := fun k hk => roundsAt_framed hrs hk
  have hr0 : sched'.size = j₀ + 1 → ∃ w : ByteArray, w.size = 16 ∧
      (_root_.Flock.Firewall.Frame.innerSched q.cfg q.st.m (R.innerOf l0.root r) (loopNr x h1.length)
        (q.st.m.inner.getD 0)).data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w ++ header 4 1 1).data.toList := by
    intro hS
    have h0 := answered_own0 T L₀ real x ρ e0 h1 hx hq hj₀ hph hc hs' hS hF hA₀
    have hin : q.st.m.inner = none := (congrArg Masker.inner (huq m₀ (answeredR_of_answered hA₀))).trans h0
    rw [hin, Option.getD_none]
    exact innerSched_zero _ _ _ _
  have hjq : j₀ + 1 ≤ q₀ := by
    have := locate_snd_le (roundsOf spec sts) 0 q₀
    rw [hj₀] at this
    exact this
  obtain ⟨P2, hat2, hcw2⟩ := roundRun_coins x e0 h1 v hq hjq hr hreq hfr hrp ho hput ha hb hs hco hRq hcw
  obtain rfl := Option.some.inj (hat2.symm.trans hat)
  obtain ⟨P3, hat3, hcf3⟩ := roundRun_cfg hr hreq hfr hf hrp ho hput ha hb hs hco hRq
  obtain rfl := Option.some.inj (hat3.symm.trans hat)
  obtain ⟨st₂, -, hat4, hrun⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hco hRq
  obtain rfl := Option.some.inj (hat4.symm.trans hat)
  obtain ⟨hfd, hstep⟩ := roundRun_step_own T L₀ real x ρ e0 h1 hx hq hj₀ hph hc hz hH hM hrow hs'
    (by rw [← hsz']; exact hle) hF hA₀ hr hreq hfr hrp ho hput ha hb hs hco hRq hcq
    (huq m₀ (answeredR_of_answered hA₀)) hpl hr0
  refine ⟨hfd, fun j hj => ⟨_, hat, hrun, ?_, fun m hA => ?_, hcw2, hinv⟩⟩
  · rw [hcf3, hI.cfg, cfg_same T L₀ real x ρ e0 h1 hx hq hj (v, a.data.toList)]
  · obtain ⟨P5, hat5, -, hu5⟩ := hstep j hj m hA
    obtain rfl := Option.some.inj (hat5.symm.trans hat)
    exact hu5

open Loop LeanRelease in
/-- **The round's own rep one round on, past the rep's schedule**: as `repJ_same`, where the round is past the rep's
schedule so far, the inner proof's (`hout`: the placeholder's schedule so far on the rep's coins no longer than them),
and the placeholder's row of pads is within a `u64`'s count (`hRow`): the model's coin step takes the round's coins
(`roundRun_own`, `ofLoop_fed_snd`), and where the next round is the same stream's (`hsame`) the frame the session's next
request finds keeps `SameInv` one round more of the rep (`roundRun_own`), its schedule of one shape with the
placeholder's (`Sched.schedSoFar_alike`), at the rep's salts, its algebraic phase's rounds no more than its coins. -/
theorem repJ_own (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)}
    (hrel : M.L = ofLoop T L₀ real) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true M.x = some (s0, spec, sts))
    (hS : ∀ (k : Nat) (link : Array F128) (c' : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray)
      (xs : Array (Array F128)) (hi : Array ByteArray), k < spec.repStreams.size →
      link.size ≤ _root_.Flock.Firewall.Frames.COIN_MAX → placeholder spec sts k link = .ok (c', t, dom, xs, hi) →
      dom.size < 2 ^ 64 ∧ xs.size + 2 < 2 ^ 64)
    (hR : Count.RealTabs sts (real M.x M.w M.ρ))
    (hRow : ∀ (k : Nat) (link : Array F128) (c' : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray)
      (xs : Array (Array F128)) (hi : Array ByteArray), k < spec.repStreams.size →
      placeholder spec sts k link = .ok (c', t, dom, xs, hi) → c'.rowH.size < 2 ^ 64)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dom : ByteArray} {v : Claims.Msg} {e : Env}
    {σ σ' : LoopAt} {e0 : Claims.Msg × Claims.Msg} {h1 : List (Claims.Msg × Claims.Msg)} {q₀ j₀ : Nat}
    (hmem : (i, r, stream) ∈ spec.repStreams.toList) (hh : M.hist σ.c = e0 :: h1) (hq : h1.length = q₀ + 1)
    (hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, j₀ + 1)) {m₀ : Masker}
    (hA₀ : M.L.answered M.x M.ρ (M.hist σ.c) = .ok m₀)
    (hK2 : ∀ rel : Release, sessAt σ.R sid = some rel → rel.spec = spec ∧ rel.sts = sts ∧
      rel.link = some (wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2) 0))
    (hK3 : ∀ (rel : Release) (name : String) (l0 : _root_.Flock.Firewall.Level0.Level0) (stm : Stmt)
      (t : _root_.Flock.Firewall.Finish.Tab), sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
      _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
      ∃ hr, real M.x M.w M.ρ (2 * i + r) = .ok (t, hr))
    (hout : ∃ c' t' dom' xs' hi' sched', placeholder spec sts (2 * i + r)
      (wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2) 0) = .ok (c', t', dom', xs', hi') ∧
      schedSoFar t' r dom' xs' hi' ((Array.range (j₀ + 1)).map fun k =>
        wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2)
          ((M.hist σ.c).length - 2 - (j₀ + 1) + k + 1)) = .ok sched' ∧ sched'.size ≤ j₀ + 1)
    (hr : RoundRun sid st pl stream (loopNr M.x h1.length) i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hreq : st.request pl (.round stream (loopNr M.x h1.length)) = none) {q : Rep} {pl' : Place} {f : Bool}
    {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ (loopNr M.x h1.length) =
      .ok (q, pl', f, nd))
    {o : ByteArray} (ho : e.round = .ok o) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr M.x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {m' : Claims.Msg} (hh' : M.hist σ'.c = M.hist σ.c ++ [(m', a.data.toList)]) (hSt : Stays sid σ σ')
    (hJ : RepJ M sid spec sts i r σ) : M.L.fed M.x M.ρ (M.hist σ'.c) = true ∧
      ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).1 = 2 * i + r → RepJ M sid spec sts i r σ') := by
  obtain ⟨p, rel, name, l0, hfa, hss, hn, hl0, hA⟩ := hJ
  have hss' : σ.R.sessions.find? (·.sid == sid) = some rel := hss
  obtain ⟨hsp, hsts, hlk⟩ := hK2 rel hss
  have hlen : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := congrArg Prod.fst hk
    unfold nextAt at this
    rwa [hlen] at this
  have hk2 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = j₀ + 1 := by
    have := congrArg Prod.snd hk
    unfold nextAt at this
    rwa [hlen] at this
  obtain ⟨c', t', dom', xs', hip, sched', hph', hs0, hle'⟩ := hout
  rw [hlen, hh, List.tail_cons] at hs0
  rw [hh, List.tail_cons] at hph' hlk
  rw [hrel, hh] at hA₀
  obtain ⟨j, c, stm, t, xs, hi, pre, ev, hj, hlink, hstm, htab, hcap, hxs, hI, hsa, hord⟩ :=
    hA.1 (by rw [hlen]; exact hk1)
  rw [hlen, hk2] at hj
  obtain rfl : j₀ = j := by omega
  rw [hlen, hh, List.tail_cons] at hI
  have hr2 := rep_lt_two hmem
  have hmod : (2 * i + r) % 2 = r := by omega
  have hdiv : (2 * i + r) / 2 = i := by omega
  rw [hmod] at hI
  rw [hmod] at hsa
  have hstm' : rel.sts[i]? = some stm := by
    have h' := hstm
    rw [hdiv] at h'
    exact h'
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hf)
  have hpr : p.rep = r := congrArg (fun x => x.2.2) (find_key hf)
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm'
  have hsz : p.coins.size = j₀ + 1 := by rw [hI.coins, Array.size_map, Array.size_range]
  have hne' : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  rw [← hpr] at hsa
  obtain ⟨r0, q0, his, hI', hsalt, hsa'⟩ := sameInv_salts_kept hI hfa hrp hss' hn hl0 hsa hord
  obtain ⟨w, hw⟩ := roundPlace_placeOf hf hrp (by rw [hne', Bool.false_and]) hss' hn hl0
  obtain ⟨sched, hsc⟩ := sched_of_placeOf hst' hlink hsalt htab hxs hI'.inv hw
  have hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1
      (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0) = .ok (c', t', dom', xs', hip) := by
    rw [hk1]
    exact hph'
  obtain ⟨lay, mask, -, -, hdm, hxe⟩ := Loop.placeholder_ok hph'
  have hname : spec.tables[i]! = name := by
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hn
    rw [← hsp, getElem!_pos rel.spec.tables i hlt, he]
  have hstmb : sts[i]! = stm := by
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hstm'
    rw [← hsts, getElem!_pos rel.sts i hlt, he]
  have hdom : (rel.spec.domain name r).toUTF8 = dom' := by rw [hdm, hdiv, hmod, hname, hsp]
  have hcw : c = wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0 := Option.some.inj (hlink.symm.trans hlk)
  have hxx : xs = xs' := by rw [hxs, hxe, hdiv, hstmb, hcw]
  have hkl : 2 * i + r < spec.repStreams.size := rep_lt_size hmem
  have hls := wordsAt_zero_size M.x (cs := h1.map fun y => ZKMal.CoinLayer.rd y.2)
    (by rw [List.length_map, hq]; omega)
  obtain ⟨hdm', hxs2⟩ := hS (2 * i + r) _ c' t' dom' xs' hip hkl (by rw [hls]; exact Nat.min_le_right _ _) hph'
  obtain ⟨hr', hre⟩ := hK3 rel name l0 stm t hss hn hl0 hstm' htab
  have hT := Count.real_tabAlike hR hph' hre
  obtain ⟨hdg, hmb, hnb, hlq, hpb, hdb⟩ := Count.real_bounds hph' hT
  have hsc2 : schedSoFar t r dom' xs' his p.coins = .ok sched := by
    rw [← hpr, ← hdom, ← hxx]
    exact hsc
  have hs1 : schedSoFar t' r dom' xs' hip p.coins = .ok sched' := by
    rw [hI.coins]
    exact hs0
  have hal := Sched.schedSoFar_alike t t' r dom' xs' his hip p.coins hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1 hT.2.2.2.2
    hdm' hdg hmb hxs2 hnb hlq hpb hdb
  rw [hsc2, hs1] at hal
  have hle : sched.size ≤ j₀ + 1 := by
    rw [roundsAt_size hal]
    exact hle'
  have halg_le : p.alg.size ≤ j₀ + 1 := by
    by_contra hcon
    have hlt : p.coins.size < p.alg.size := by omega
    have e1 := Except.ok.inj (hsc.symm.trans (schedSoFar_alg t _ xs his p hI'.inv.alg hlt))
    rw [e1] at hle
    omega
  have ho' := (hr.round hreq).symm.trans ho
  obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
  rw [hrel] at hI'
  obtain ⟨hfd, hsm⟩ := roundRun_own T L₀ real M.x M.ρ e0 h1 hx hq hk2 hph
    (hRow _ _ c' t' dom' xs' hip hkl hph') (by rw [hk1, hmod]; exact hs0) hr hreq hrp ho' hput ha hb hs hco hfa
    hss' hn hl0 hst' hlink hsalt htab hxs hI' hsc hal hle hA₀
  have hsnd : (e0 :: (h1 ++ [(v, a.data.toList)])).map Prod.snd = (M.hist σ'.c).map Prod.snd := by
    rw [hh', hh]
    simp
  refine ⟨by rw [hrel]; exact (ofLoop_fed_snd T L₀ real M.x M.ρ hsnd).symm.trans hfd, fun hsame => ?_⟩
  have hj2 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).2 = j₀ + 1 + 1 := by
    rw [Loop.locate_succ _ _ _ (hsame.trans hk1.symm), hk2]
  obtain ⟨p₃, hat, hI3⟩ := hsm _ (by rw [hq]; exact hj2)
  have hI4 := sameInv_snd hsnd hI3
  have hcs : ((h1 ++ [(v, a.data.toList)]).map fun y => ZKMal.CoinLayer.rd y.2) =
      (M.hist σ'.c).tail.map fun y => ZKMal.CoinLayer.rd y.2 := by
    rw [hh', hh, List.cons_append, List.tail_cons]
    simp only [List.map_append, List.map_cons, List.map_nil]
  rw [hcs, ← hrel] at hI4
  have halg : p₃.alg = p.alg := Except.ok.inj (hI4.inv.alg.symm.trans hI'.inv.alg)
  have hc3 : p₃.coins.size = j₀ + 1 + 1 := by rw [hI4.coins, Array.size_map, Array.size_range]
  have hl' : (M.hist σ'.c).length - 2 = q₀ + 1 := by
    rw [hh', hh, List.length_append, List.length_cons, hq, List.length_singleton]
    omega
  refine ⟨p₃, rel, name, l0, hat, ?_, hn, ?_, ?_⟩
  · rw [sessAt_stay (hSt.2 name l0 hl0) (by rw [hss]; rfl)]
    exact hss
  · rw [hSt.1 name]
    exact hl0
  · unfold StreamAt
    rw [hl']
    refine ⟨fun _ => ?_, fun he _ => absurd (hsame.symm.trans he) (by omega)⟩
    rw [hmod]
    refine ⟨j₀ + 1, c, stm, t, xs, his, pre, _, hj2, hlink, hstm, htab, hcap, hxs, hI4, fun hlt3 => ?_,
      fun heq3 => absurd heq3 (by rw [halg, hc3]; omega)⟩
    rw [halg, hc3] at hlt3
    have h1 := hsa' (by omega)
    rw [hpr] at h1
    rw [saltsAt_stay (hSt.2 name l0 hl0) r (by rw [h1]; rfl), h1]

end Claims.ZKServer.Relay
