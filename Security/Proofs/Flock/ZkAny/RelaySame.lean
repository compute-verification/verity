module

public import Proofs.Flock.ZkAny.RelayCount
public import Proofs.Flock.ZkAny.RelayProg

/-!
# A rep's frame at the model's state for its next round, from one round of the rep to the next

What the loop keeps of a rep's frame between its rounds, against the model at the history before the rep's next round
(`SameInv`): the frame's run on the loop's requests to it, the model's configuration and masking state but for
`stopped`, the history's words for the rep's rounds so far as its coins, and its schedule so far (`Loop.RepInv`). One
round of the rep on, the frame the session's next request finds keeps it:

* `placeAt_lt`: within the schedule, `placeAt` places a round at the schedule's round there.
* `sameInv_place`: at a frame keeping `SameInv`, the round is not its stream's first, and the frame `roundPlace`
  readies has the model's configuration, its masking state and the history's words.
* **`roundRun_same`**: so at a `RoundRun` past the rep's first round, where the frame's schedule on its coins holds
  the round and is of one shape with the placeholder's, whose count is the round's, the model's coin step takes the
  round's coins, and where the next round is the same rep's the round is in the frame's schedule's place and the
  frame the session's next request finds keeps `SameInv` at the history with the worker's message and the auditor's
  answer (`roundRun_step`, `roundRun_repInv`, `roundRun_coins`, `roundRun_cfg`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Cfg Masker Req Place)

/-- **A rep's frame at the model's state for the rep's next round**: after the history `h`, whose decoded coins are
`cs`, the rep's next round being its `j + 1`-th, after coin `q`, the frame is its run on the loop's requests (`RepRun`),
at the model's configuration and its masking state before the rank stop (`answeredR`) but for `stopped`, holding the
words of the rep's `j + 1` rounds so far as its coins, and keeping its schedule so far on its table `t`, its stream's
domain `dom`, its extra claims `xs` and its salts `hi` (`Loop.RepInv`). -/
structure SameInv {W Rr : Type} (L : LeanRelease W Rr) (x : Msg) (ρ : Rr) (h : List (Msg × Msg))
    (cs : List FlockDeploy.Coin) (q j : Nat) (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (pre : List Req)
    (ev : List (Nat × ByteArray × Place × Array F128)) (p : Rep) : Prop where
  run : RepRun pre ev p
  cfg : p.cfg = L.cfg x ρ h
  st : ∀ m, L.answeredR x ρ h = .ok m → unstop p.st.m = unstop m
  coins : p.coins = (Array.range (j + 1)).map fun k => Loop.wordsAt x cs (q - (j + 1) + k + 1)
  inv : Loop.RepInv t dom xs hi p

/-- **Within the schedule, `placeAt` places a round at the schedule's round there.** -/
theorem placeAt_lt {sched : Array (ByteArray × Nat)} {k : Nat} (h : k < sched.size) :
    (Loop.placeAt sched k).1 = .sched (sched.getD k (.empty, 0)).1 := by
  unfold Loop.placeAt
  rw [Array.getElem?_eq_getElem h, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h]
  rfl

/-- **At a frame keeping `SameInv` the readied frame is at the model's state**: the round is not its stream's first,
and the frame `roundPlace` readies is the found one's run, at the model's configuration and masking state but for
`stopped`, holding the history's words. -/
theorem sameInv_place {W Rr : Type} {L : LeanRelease W Rr} {x : Msg} {ρ : Rr} {h : List (Msg × Msg)}
    {cs : List FlockDeploy.Coin} {q j : Nat} {t : _root_.Flock.Firewall.Finish.Tab} {dom' : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {pre : List Req}
    {ev : List (Nat × ByteArray × Place × Array F128)} {p : Rep} (hI : SameInv L x ρ h cs q j t dom' xs hi pre ev p)
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat} (hfa : frameAt R sid i r = some p)
    {dom content : ByteArray} {n : Nat} {q' : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q', pl, f, nd)) :
    f = false ∧ RepRun pre ev q' ∧ q'.cfg = L.cfg x ρ h ∧
      (∀ m, L.answeredR x ρ h = .ok m → unstop q'.st.m = unstop m) ∧
      q'.coins = (Array.range (j + 1)).map fun k => Loop.wordsAt x cs (q - (j + 1) + k + 1) := by
  have hsz : p.coins.size = j + 1 := by rw [hI.coins, Array.size_map, Array.size_range]
  have hne : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  obtain ⟨hl, hcf, hst, hco⟩ := roundPlace_next hfa hne hrp
  refine ⟨hl, hI.run.of_eq hcf hst hco, hcf.trans hI.cfg, fun m hA => ?_, hco.trans hI.coins⟩
  rw [hst]
  exact hI.st m hA

open Loop LeanRelease in
/-- **A rep's frame keeps `SameInv` from one round of the rep to the next**: at a `RoundRun` at the model's count whose
content is the worker's message `v`, past the rep's first round (`hj₀`), whose round releases and whose coins the
rep's frame took off the auditor's answer `a`, where the found frame keeps `SameInv` on the table and extras its
session gives it and the salts the loop's salts give it at the opening's first round (`hsalt`), its schedule on its
coins holds the round (`hlt`) and is of one shape with the placeholder's (`hrs`), whose count is the round's (`hn'`),
the model's coin step takes the coins after the history with `(v, a)`, and where the next round is the same rep's
(`hj`) the round is in the frame's schedule's place and the frame the session's next request finds keeps `SameInv`
at that history, one round more of the rep. -/
theorem roundRun_same (T : ZKMal.CoinLayer.TableData)
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
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c') {sched' : Array (ByteArray × Nat)}
    (hs' : schedSoFar t' ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 % 2) dom' xs' hi'
      ((Array.range (j₀ + 1)).map fun i =>
        wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q₀ - (j₀ + 1) + i + 1)) = .ok sched')
    (hn' : loopNr x h1.length = (sched'.getD (j₀ + 1) (.empty, 0)).2)
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
    {p : Rep} (hfa : frameAt R sid i r = some p) {s : _root_.Flock.Firewall.Frames.Release} {name : String}
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
    (hrs : Sched.RoundsAt sched sched') (hlt : j₀ + 1 < sched.size)
    {m₀ : Masker} (hA₀ : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m₀) :
    (ofLoop T L₀ real).fed x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = true ∧
      ∀ j, (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1 →
        pl' = .sched (sched.getD (j₀ + 1) (.empty, 0)).1 ∧
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
  obtain ⟨hpl0, P1, hat, hinv⟩ :=
    roundRun_repInv hr hreq hfr hrp ho hput ha hb hs hco hRq hf hl hss hn hl0 hst hlink hsalt htab hxs hI.inv hsc
  have hpa : (placeAt sched p.coins.size).1 = .sched (sched.getD (j₀ + 1) (.empty, 0)).1 := by
    rw [hsz]
    exact placeAt_lt hlt
  have hpl : pl' = .sched (sched.getD (j₀ + 1) (.empty, 0)).1 := by
    rw [hpa] at hpl0
    rcases hpl0 with h' | h'
    · exact h'
    · cases h'
  have hn2 : loopNr x h1.length = (sched.getD (j₀ + 1) (.empty, 0)).2 := count_roundsAt hrs hlt hn'
  have hjq : j₀ + 1 ≤ q₀ := by
    have := Loop.locate_snd_le (roundsOf spec sts) 0 q₀
    rw [hj₀] at this
    exact this
  obtain ⟨P2, hat2, hcw2⟩ := roundRun_coins x e0 h1 v hq hjq hr hreq hfr hrp ho hput ha hb hs hco hRq hcw
  obtain rfl := Option.some.inj (hat2.symm.trans hat)
  obtain ⟨P3, hat3, hcf3⟩ := roundRun_cfg hr hreq hfr hf hrp ho hput ha hb hs hco hRq
  obtain rfl := Option.some.inj (hat3.symm.trans hat)
  obtain ⟨st₂, -, hat4, hrun⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hco hRq
  obtain rfl := Option.some.inj (hat4.symm.trans hat)
  have hA₀' := answeredR_of_answered hA₀
  obtain ⟨hfd, hstep⟩ := roundRun_step T L₀ real x ρ e0 h1 hx hq hj₀ hph hc hrs hs' hlt hA₀' hr hreq hfr hrp ho hput
    ha hb hs hco hRq hcq (huq m₀ hA₀') hpl hn2
  refine ⟨hfd, fun j hj => ⟨hpl, _, hat, hrun, ?_, fun m hA => ?_, hcw2, hinv⟩⟩
  · rw [hcf3, hI.cfg, cfg_same T L₀ real x ρ e0 h1 hx hq hj (v, a.data.toList)]
  · obtain ⟨P5, hat5, -, hu5⟩ := hstep j hj m hA
    obtain rfl := Option.some.inj (hat5.symm.trans hat)
    exact hu5

end Claims.ZKServer.Relay
