module

public import Proofs.Flock.ZkAny.RelayFold
public import Proofs.Flock.ZkAny.RelayTaken
public import Proofs.Flock.Claims.ZK.FrameProg
public import Proofs.Flock.Claims.ZK.FrameRun

/-! A rep's masking state in the loop's frames is the model's, but for `stopped` (`round_progOf`, `coins_answered`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.ZkServer.Frame (Cfg Masker)
open _root_.Flock.ZkServer.Frames (Rep)

theorem answer_stopped (c : Cfg) (s : Masker) (cs : Array F128) (b : Bool) :
    ({ s with stopped := b } : Masker).answer c cs = (s.answer c cs).map fun a => { a with stopped := b } := by
  obtain ⟨co, pr, ow, rk, po, on, dn, dl, l0, rh, inn, ep, be, st⟩ := s
  unfold Masker.answer
  simp only [bind, Except.bind, Except.map, pure, Except.pure]
  repeat' split
  all_goals first
    | rfl
    | contradiction
    | (simp only [Except.ok.injEq, Except.error.injEq] at *; subst_vars; rfl)
    | (simp_all; done)
    | (simp_all; subst_vars; rfl)
    | (simp_all; subst_vars; simp)

theorem answer_unstop {c c' : Cfg} (hc : CfgAlike c c') {s s' a : Masker} {cs : Array F128}
    (hu : unstop s = unstop s') (h : s.answer c cs = .ok a) :
    ∃ a', s'.answer c' cs = .ok a' ∧ unstop a = unstop a' := by
  have hcc : s'.answer c cs = s'.answer c' cs := by
    unfold Masker.answer
    rw [hc.lay, hc.order]
  rw [eq_unstop hu.symm, answer_stopped, hcc] at h
  cases ha : s'.answer c' cs with
  | error e =>
    rw [ha] at h
    cases h
  | ok a' =>
    rw [ha] at h
    cases h
    exact ⟨a', rfl, rfl⟩

open Loop in
theorem round_progOf (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1) {c c' : Cfg} {t : _root_.Flock.ZkServer.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    (hc : CfgAlike c c') {j : Nat} (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1)
    {rsched sched' : Array (ByteArray × Nat)} (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok sched')
    (hlt : j + 1 < rsched.size) {m : Masker} (ha : (progOf x cs).answer c (wordsAt x cs q) = .ok m)
    {st st' : _root_.Flock.ZkServer.Frame.State} (hu : unstop st.m = unstop m) {v out : ByteArray} {vd : String}
    (h : _root_.Flock.ZkServer.Frame.step c st
      (.round (rsched.getD (j + 1) (.empty, 0)).2 v (.sched (rsched.getD (j + 1) (.empty, 0)).1)) =
        (.round out vd, st')) :
    unstop st'.m = unstop (progOf x (cs ++ [d])) := by
  obtain ⟨why, hck⟩ := _root_.Flock.ZkServer.Frame.check_of_step h
  rw [eq_unstop hu.symm] at hck
  rw [progOf_check_step x cs d hx hq hph hc hj hrs hs' hlt ha hck]
  all_goals rfl

open Loop in
theorem round_progOf_first (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1) (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = 0) {m : Masker}
    (hm : freshOf x cs = .ok m) {c c' : Cfg} {t : _root_.Flock.ZkServer.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    (hc : CfgAlike c c') {rsched sched' : Array (ByteArray × Nat)} (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi #[] = .ok sched')
    (hlt : 0 < rsched.size) {st st' : _root_.Flock.ZkServer.Frame.State} (hu : unstop st.m = unstop m)
    {v out : ByteArray} {vd : String}
    (h : _root_.Flock.ZkServer.Frame.step c st
      (.round (rsched.getD 0 (.empty, 0)).2 v (.sched (rsched.getD 0 (.empty, 0)).1)) = (.round out vd, st')) :
    unstop st'.m = unstop (progOf x (cs ++ [d])) := by
  obtain ⟨why, hck⟩ := _root_.Flock.ZkServer.Frame.check_of_step h
  rw [eq_unstop hu.symm] at hck
  rw [progOf_check_first x cs d hx hq hj hm hph hc hrs hs' hlt hck]
  all_goals rfl

open Loop LeanRelease in
theorem coins_answered (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q j : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) {c : Cfg}
    (hc : CfgAlike c ((ofLoop T L₀ real).cfg x ρ (e0 :: h1))) {st st' : _root_.Flock.ZkServer.Frame.State}
    (hu : unstop st.m = unstop (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)))
    (hco : _root_.Flock.ZkServer.Frame.step c st (.coins (.ok (coinsAt x (e0 :: h1)))) = (.ok, st'))
    {m : Masker} (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: h1) = .ok m) :
    unstop st'.m = unstop m := by
  obtain ⟨-, -, hF⟩ := answeredR_ok hA
  rw [ofLoop_frameAnswerR_same T L₀ real x ρ e0 h1 hx hq hj] at hF
  obtain ⟨a', ha', hua⟩ := answer_unstop hc hu (_root_.Flock.ZkServer.Frame.answer_of_step hco)
  rw [hF] at ha'
  cases ha'
  exact hua

theorem step_prior_alike {c c' : Cfg} (hc : CfgAlike c c') (st : _root_.Flock.ZkServer.Frame.State) (k : Nat)
    (b : ByteArray) :
    _root_.Flock.ZkServer.Frame.step c st (.prior k b) = _root_.Flock.ZkServer.Frame.step c' st (.prior k b) := by
  simp only [_root_.Flock.ZkServer.Frame.step, hc.lay]

theorem repRun_start_fresh {q : Rep} (h : RepRun [] [] q) : q.st = {} :=
  (congrArg Prod.fst h.1).symm

theorem repRun_prior_fresh {k : Nat} {b : ByteArray} {q : Rep} (h : RepRun [.prior k b] [] q) {c' : Cfg}
    (hc : CfgAlike q.cfg c') {st' : _root_.Flock.ZkServer.Frame.State}
    (hp : _root_.Flock.ZkServer.Frame.step c' {} (.prior k b) = (.ok, st')) : q.st = st' := by
  have h1 := h.1
  have e : _root_.Flock.ZkServer.Frame.run q.cfg {} ([.prior k b] ++ evReqs []) = (st', false) := by
    show _root_.Flock.ZkServer.Frame.run q.cfg {} [.prior k b] = (st', false)
    rw [run_cons, step_prior_alike hc, hp]
    all_goals rfl
  rw [e] at h1
  exact (congrArg Prod.fst h1).symm

end Claims.ZKServer.Relay
