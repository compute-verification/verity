module

public import Proofs.Flock.ZkAny.RelayGap
public import Proofs.Flock.ZkAny.RelayFresh

/-! After its stream's first round a rep's frame keeps its run, words and schedule so far, whichever rep's round comes next (`roundRun_last_first`, `roundRun_same_first`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.ZkServer.Session (State Placed byte?)
open _root_.Flock.ZkServer.Frames (Sessions Rep)
open _root_.Flock.ZkServer.Frame (Cfg Masker Req Place)

open Loop in
theorem roundRun_last_first (x : Msg) (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v : Msg) {q₀ : Nat}
    (hq : h1.length = q₀ + 1) {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray}
    {e : Env} {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {pre : List Req} (hR : RepRun pre [] q) {p : Rep} (hfa : frameAt R sid i r = some p) (hp : FreshRep p)
    {s : _root_.Flock.ZkServer.Frames.Release} {name : String} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.ZkServer.Level0.ofSession L0 sid name = some l0) {p' : Rep}
    (hw : _root_.Flock.ZkServer.Frames.withAlg s l0 name p = .ok p') {t : _root_.Flock.ZkServer.Finish.Tab}
    {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt} {c : Array F128} {r0 : ByteArray} {q0 : Nat}
    (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p'.alg.size = 0 → R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.ZkServer.Finish.flat t.cm0.cap = l0.root)
    (hxs : xs = _root_.Flock.ZkServer.Drive.extras stm (_root_.Flock.ZkServer.Drive.linkPoints stm c))
    {sched : Array (ByteArray × Nat)}
    (hsc : schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched) :
    ∃ p₃, frameAt R₃ sid i r = some p₃ ∧
      RepRun pre [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray), pl',
        _root_.Flock.ZkServer.Frames.feedOf (loopNr x h1.length) a.data.toList)] p₃ ∧
      p₃.coins = (Array.range 1).map (fun k =>
        wordsAt x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2) (q₀ + 1 - 1 + k + 1)) ∧
      RepInv t (s.spec.domain name r).toUTF8 xs hi p₃ := by
  have hfr : (frameAt R sid i r).isSome := by
    rw [hfa]
    rfl
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  obtain ⟨-, P1, hat, hinv⟩ := roundRun_repInv_first hr hreq hrp ho hput ha hb hs hco hR hf hp hss hn hl0 hw hst
    hlink hsalt htab hcap hxs hsc
  obtain ⟨st₂, -, hat2, hrun⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hco hR
  obtain rfl := Option.some.inj (hat2.symm.trans hat)
  refine ⟨_, hat, hrun, ?_, hinv⟩
  rcases hres : _root_.Flock.ZkServer.Frame.step q.cfg q.st
    (.round (loopNr x h1.length) (if nd.isSome then ByteArray.empty else ⟨v.toArray⟩) pl') with ⟨a', st'⟩
  obtain ⟨w, hw'⟩ := roundStep_put hrp hres
  rw [hw', Option.some.injEq] at hput
  subst hput
  have hq0 : q.coins = #[] := Array.toList_inj.mp hR.2
  have hr1 : Array.range 1 = (Array.range 0).push 0 := by
    rw [Array.push_eq_append]
    exact Array.range_succ
  show q.coins.push (_root_.Flock.ZkServer.Frames.feedOf (loopNr x h1.length) a.data.toList) = _
  rw [hq0, feedOf_words x e0 h1 v a.data.toList hq, hr1, Array.map_push, Array.range_eq_empty_iff.mpr rfl,
    Array.map_empty, show q₀ + 1 - 1 + 0 + 1 = q₀ + 1 by omega]

theorem count_first {rs : List Nat} {q : Nat} (hj : (ZKMal.CoinLayer.locate rs 0 q).2 = 0)
    (hnext : (ZKMal.CoinLayer.locate rs 0 (q + 1)).1 = (ZKMal.CoinLayer.locate rs 0 q).1 + 1) :
    rs.getD ((ZKMal.CoinLayer.locate rs 0 (q + 1)).1 - 1) 0 = 1 := by
  have h := locate_last rs 0 q (by rw [hnext]; omega)
  rw [hnext, Nat.add_sub_cancel]
  rw [Nat.sub_zero, hj] at h
  omega

open Loop in
theorem roundRun_handoff_first (x : Msg) (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v : Msg) {rs : List Nat}
    {q₀ : Nat} (hq : h1.length = q₀ + 1) (hj₀ : (ZKMal.CoinLayer.locate rs 0 q₀).2 = 0)
    (hnext : (ZKMal.CoinLayer.locate rs 0 (q₀ + 1)).1 = (ZKMal.CoinLayer.locate rs 0 q₀).1 + 1)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray} {e : Env}
    {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {pre : List Req} (hR : RepRun pre [] q) {p : Rep} (hfa : frameAt R sid i r = some p) (hp : FreshRep p)
    {s : _root_.Flock.ZkServer.Frames.Release} {name : String} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.ZkServer.Level0.ofSession L0 sid name = some l0) {p' : Rep}
    (hw : _root_.Flock.ZkServer.Frames.withAlg s l0 name p = .ok p') {t : _root_.Flock.ZkServer.Finish.Tab}
    {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt} {c : Array F128} {r0 : ByteArray} {q0 : Nat}
    (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p'.alg.size = 0 → R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.ZkServer.Finish.flat t.cm0.cap = l0.root)
    (hxs : xs = _root_.Flock.ZkServer.Drive.extras stm (_root_.Flock.ZkServer.Drive.linkPoints stm c))
    {sched : Array (ByteArray × Nat)}
    (hsc : schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched)
    {R' : Sessions} (hk : Kept sid i r R₃ R') :
    ∃ j0 p0, R'.find sid i r = some (j0, p0) ∧ rs.getD ((ZKMal.CoinLayer.locate rs 0 (q₀ + 1)).1 - 1) 0 = 1 ∧
      p0.coins = (Array.range 1).map (fun k =>
        wordsAt x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2) (q₀ + 1 - 1 + k + 1)) := by
  obtain ⟨p₃, hat, -, hcw, -⟩ := roundRun_last_first x e0 h1 v hq hr hreq hrp ho hput ha hb hs hco hR hfa hp hss
    hn hl0 hw hst hlink hsalt htab hcap hxs hsc
  have hat' : frameAt R' sid i r = some p₃ := by
    rw [frameAt_kept hk (by rw [hat]; rfl), hat]
  obtain ⟨j0, hj0⟩ := find_of_frameAt hat'
  exact ⟨j0, p₃, hj0, count_first hj₀ hnext, hcw⟩

open Loop LeanRelease in
theorem roundRun_step_first (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q₀ : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hj₀ : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = 0) {c' : Cfg}
    {t : _root_.Flock.ZkServer.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1
      (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi))
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c') {rsched sched' : Array (ByteArray × Nat)}
    (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 % 2) dom xs hi #[] = .ok sched')
    (hlt : 0 < rsched.size) {m₀ : Masker} (hm : freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m₀)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray} {v : Msg} {e : Env}
    {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none) (hfr : (frameAt R sid i r).isSome)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} (hR : RepRun pre ev q)
    (hcq : q.cfg = (ofLoop T L₀ real).cfg x ρ (e0 :: h1)) (hu₀ : unstop q.st.m = unstop m₀)
    (hpl : pl' = .sched (rsched.getD 0 (.empty, 0)).1) (hn : loopNr x h1.length = (rsched.getD 0 (.empty, 0)).2) :
    (ofLoop T L₀ real).fed x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = true ∧
      ∀ j, (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1 → ∀ m,
        (ofLoop T L₀ real).answeredR x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = .ok m →
        ∃ p, frameAt R₃ sid i r = some p ∧
          RepRun pre (ev ++ [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray),
            pl', _root_.Flock.ZkServer.Frames.feedOf (loopNr x h1.length) a.data.toList)]) p ∧
          unstop p.st.m = unstop m := by
  have hu : unstop (_root_.Flock.ZkServer.Frame.step q.cfg q.st (.round (loopNr x h1.length)
      (if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray)) pl')).2.m =
      unstop (progOf x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2)) := by
    obtain ⟨out, vd, hre, -⟩ := roundStep_out hrp ho
    revert hre
    rcases hres : _root_.Flock.ZkServer.Frame.step q.cfg q.st (.round (loopNr x h1.length)
      (if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray)) pl') with ⟨a', st'⟩
    intro hre
    simp only at hre
    subst hre
    rw [hn, hpl] at hres
    have hlen : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q₀ + 1 := by rw [List.length_map, hq]
    rw [List.map_append, List.map_cons, List.map_nil]
    exact round_progOf_first x _ _ hx hlen hj₀ hm hph (by rw [hcq]; exact hc) hrs hs' hlt hu₀ hres
  refine ⟨roundRun_fed T L₀ real x ρ e0 h1 hr hreq hfr hrp ho hput ha hb hs hco hR hcq hu, fun j hj m hA => ?_⟩
  have hcfg : CfgAlike q.cfg ((ofLoop T L₀ real).cfg x ρ (e0 :: (h1 ++ [(v, a.data.toList)]))) := by
    rw [hcq, cfg_same T L₀ real x ρ e0 h1 hx hq hj (v, a.data.toList)]
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  exact roundRun_carry T L₀ real x ρ e0 h1 hx hj hr hreq hfr hrp ho hput ha hb hs hco hR hcfg hu hA

open Loop LeanRelease in
theorem roundRun_same_first (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q₀ : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hj₀ : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = 0) {c' : Cfg}
    {t' : _root_.Flock.ZkServer.Finish.Tab} {dom' : ByteArray} {xs' : Array (Array F128)} {hi' : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1
      (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t', dom', xs', hi'))
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c') {sched' : Array (ByteArray × Nat)}
    (hs' : schedSoFar t' ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 % 2) dom' xs' hi' #[] = .ok sched')
    (hn' : loopNr x h1.length = (sched'.getD 0 (.empty, 0)).2)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray} {v : Msg} {e : Env}
    {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {pre : List Req} (hR : RepRun pre [] q) {p : Rep} (hfa : frameAt R sid i r = some p) (hp : FreshRep p)
    {s : _root_.Flock.ZkServer.Frames.Release} {name : String} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.ZkServer.Level0.ofSession L0 sid name = some l0) {p' : Rep}
    (hw : _root_.Flock.ZkServer.Frames.withAlg s l0 name p = .ok p') {t : _root_.Flock.ZkServer.Finish.Tab}
    {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt} {c : Array F128} {r0 : ByteArray} {q0 : Nat}
    (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p'.alg.size = 0 → R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.ZkServer.Finish.flat t.cm0.cap = l0.root)
    (hxs : xs = _root_.Flock.ZkServer.Drive.extras stm (_root_.Flock.ZkServer.Drive.linkPoints stm c))
    {sched : Array (ByteArray × Nat)}
    (hsc : schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched)
    (hrs : Sched.RoundsAt sched sched') (hlt : 0 < sched.size)
    (hcf : p.cfg = (ofLoop T L₀ real).cfg x ρ (e0 :: h1)) {m₀ : Masker}
    (hm : freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m₀) (hu₀ : unstop q.st.m = unstop m₀) :
    (ofLoop T L₀ real).fed x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = true ∧
      ∀ j, (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1 →
        pl' = .sched (sched.getD 0 (.empty, 0)).1 ∧
        ∃ p₃, frameAt R₃ sid i r = some p₃ ∧
          SameInv (ofLoop T L₀ real) x ρ (e0 :: (h1 ++ [(v, a.data.toList)]))
            ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2) (q₀ + 1) 0 t
            (s.spec.domain name r).toUTF8 xs hi pre
            [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray), pl',
              _root_.Flock.ZkServer.Frames.feedOf (loopNr x h1.length) a.data.toList)] p₃ := by
  have hfr : (frameAt R sid i r).isSome := by
    rw [hfa]
    rfl
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  obtain ⟨hpl0, -⟩ := roundRun_repInv_first hr hreq hrp ho hput ha hb hs hco hR hf hp hss hn hl0 hw hst hlink hsalt
    htab hcap hxs hsc
  have hpa : (placeAt sched p.coins.size).1 = .sched (sched.getD 0 (.empty, 0)).1 := by
    rw [hp.coins, Array.size_empty]
    exact placeAt_lt hlt
  have hpl : pl' = .sched (sched.getD 0 (.empty, 0)).1 := by
    rw [hpa] at hpl0
    rcases hpl0 with h' | h'
    · exact h'
    · cases h'
  have hn2 : loopNr x h1.length = (sched.getD 0 (.empty, 0)).2 := count_roundsAt hrs hlt hn'
  obtain ⟨p₃, hat, hrun, hcw, hinv⟩ := roundRun_last_first x e0 h1 v hq hr hreq hrp ho hput ha hb hs hco hR hfa hp
    hss hn hl0 hw hst hlink hsalt htab hcap hxs hsc
  obtain ⟨P3, hat3, hcf3⟩ := roundRun_cfg hr hreq hfr hf hrp ho hput ha hb hs hco hR
  obtain rfl := Option.some.inj (hat3.symm.trans hat)
  have hcq : q.cfg = (ofLoop T L₀ real).cfg x ρ (e0 :: h1) := (roundPlace_cfg hf hrp).trans hcf
  obtain ⟨hfd, hstep⟩ := roundRun_step_first T L₀ real x ρ e0 h1 hx hq hj₀ hph hc hrs hs' hlt hm hr hreq hfr hrp ho
    hput ha hb hs hco hR hcq hu₀ hpl hn2
  refine ⟨hfd, fun j hj => ⟨hpl, _, hat, hrun, ?_, fun m hA => ?_, hcw, hinv⟩⟩
  · rw [hcf3, hcf, cfg_same T L₀ real x ρ e0 h1 hx hq hj (v, a.data.toList)]
  · obtain ⟨P5, hat5, -, hu5⟩ := hstep j hj m hA
    obtain rfl := Option.some.inj (hat5.symm.trans hat)
    exact hu5

end Claims.ZKServer.Relay
