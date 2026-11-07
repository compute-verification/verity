module

public import Proofs.Flock.ZkAny.RelayFold
public import Proofs.Flock.ZkAny.RelayTaken
public import Proofs.Flock.Claims.ZK.FrameProg
public import Proofs.Flock.Claims.ZK.FrameRun

/-!
# A rep's masking state in the loop's frames is the model's, but for `stopped`

The loop runs each rep's frame on the device's rounds at the rep's own samples; the model's release reads the frame's
state the loop's own frames leave (`Loop.progOf`, at placeholder samples on the scheduled rounds). The two agree but for
`stopped` (`unstop`) from one round to the next:

* `answer_stopped`, `answer_unstop`: `Masker.answer` neither reads nor writes `stopped`, and reads only the
  configuration's layout and order, so coins taken at states and configurations alike leave states alike.
* `round_progOf`, `round_progOf_first`: a round the frame releases at a state alike the model's, against the round of
  the rep's schedule on its own table in its place, leaves a state alike the loop's after it (`Loop.progOf_check_step`,
  `Loop.progOf_check_first`), whatever the device round and whether the rep is stopped.
* `coins_answered`: and the coins the model hands the frame after it, taken there, leave a state alike the model's
  for the rep's next round before the rank stop (`LeanRelease.answeredR`).
* `step_prior_alike`, `repRun_start_fresh`, `repRun_prior_fresh`: at a rep's first round its frame (`RepRun` on no
  rounds) is the loop's start of it (`Loop.freshOf`): rep 0's the frame's start, rep 1's after rep 0's
  algebraic-phase coins, which the frame takes reading only the layout.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frame (Cfg Masker)
open _root_.Flock.Firewall.Frames (Rep)

/-- **`Masker.answer` neither reads nor writes `stopped`.** -/
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

/-- **Coins taken at states alike** but for `stopped`, at configurations of one layout, leave states alike but for
`stopped`. -/
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
/-- **A round past a rep's first, released at a state alike the model's**: where the frame's state `st` is the loop's
taking of the last coins of `cs` (`m`) but for `stopped`, and the frame releases a device round `v` against round
`j + 1` of the rep's schedule on its own table (`rsched`, of one shape with the placeholder's) at a configuration alike
the placeholder's, its state after it is the loop's after `cs ++ [d]`, but for `stopped`. -/
theorem round_progOf (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1) {c c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    (hc : CfgAlike c c') {j : Nat} (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1)
    {rsched sched' : Array (ByteArray × Nat)} (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok sched')
    (hlt : j + 1 < rsched.size) {m : Masker} (ha : (progOf x cs).answer c (wordsAt x cs q) = .ok m)
    {st st' : _root_.Flock.Firewall.Frame.State} (hu : unstop st.m = unstop m) {v out : ByteArray} {vd : String}
    (h : _root_.Flock.Firewall.Frame.step c st
      (.round (rsched.getD (j + 1) (.empty, 0)).2 v (.sched (rsched.getD (j + 1) (.empty, 0)).1)) =
        (.round out vd, st')) :
    unstop st'.m = unstop (progOf x (cs ++ [d])) := by
  obtain ⟨why, hck⟩ := _root_.Flock.Firewall.Frame.check_of_step h
  rw [eq_unstop hu.symm] at hck
  rw [progOf_check_step x cs d hx hq hph hc hj hrs hs' hlt ha hck]
  all_goals rfl

open Loop in
/-- **A rep's first round, released at a state alike the model's**: as `round_progOf`, at the rep's first round, which
the loop starts at `m` (`freshOf`), against the rep's first scheduled round. -/
theorem round_progOf_first (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1) (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = 0) {m : Masker}
    (hm : freshOf x cs = .ok m) {c c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    (hc : CfgAlike c c') {rsched sched' : Array (ByteArray × Nat)} (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi #[] = .ok sched')
    (hlt : 0 < rsched.size) {st st' : _root_.Flock.Firewall.Frame.State} (hu : unstop st.m = unstop m)
    {v out : ByteArray} {vd : String}
    (h : _root_.Flock.Firewall.Frame.step c st
      (.round (rsched.getD 0 (.empty, 0)).2 v (.sched (rsched.getD 0 (.empty, 0)).1)) = (.round out vd, st')) :
    unstop st'.m = unstop (progOf x (cs ++ [d])) := by
  obtain ⟨why, hck⟩ := _root_.Flock.Firewall.Frame.check_of_step h
  rw [eq_unstop hu.symm] at hck
  rw [progOf_check_first x cs d hx hq hj hm hph hc hrs hs' hlt hck]
  all_goals rfl

open Loop LeanRelease in
/-- **The coins after a round, taken at a state alike the loop's**: where the frame's state `st` after a round is the
loop's after it (`progOf`) but for `stopped`, the round after it is the same rep's, and the frame takes the coins the
model hands it there (`coinsAt`) at a configuration alike the model's, its state is the model's state for that next
round before the rank stop (`answeredR`), but for `stopped`. -/
theorem coins_answered (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q j : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) {c : Cfg}
    (hc : CfgAlike c ((ofLoop T L₀ real).cfg x ρ (e0 :: h1))) {st st' : _root_.Flock.Firewall.Frame.State}
    (hu : unstop st.m = unstop (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)))
    (hco : _root_.Flock.Firewall.Frame.step c st (.coins (.ok (coinsAt x (e0 :: h1)))) = (.ok, st'))
    {m : Masker} (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: h1) = .ok m) :
    unstop st'.m = unstop m := by
  obtain ⟨-, -, hF⟩ := answeredR_ok hA
  rw [ofLoop_frameAnswerR_same T L₀ real x ρ e0 h1 hx hq hj] at hF
  obtain ⟨a', ha', hua⟩ := answer_unstop hc hu (_root_.Flock.Firewall.Frame.answer_of_step hco)
  rw [hF] at ha'
  cases ha'
  exact hua

/-- An earlier rep's coins are taken alike at configurations of one layout. -/
theorem step_prior_alike {c c' : Cfg} (hc : CfgAlike c c') (st : _root_.Flock.Firewall.Frame.State) (k : Nat)
    (b : ByteArray) :
    _root_.Flock.Firewall.Frame.step c st (.prior k b) = _root_.Flock.Firewall.Frame.step c' st (.prior k b) := by
  simp only [_root_.Flock.Firewall.Frame.step, hc.lay]

/-- **At rep 0's first round** its frame is the frame's start, as the loop's (`Loop.freshOf`). -/
theorem repRun_start_fresh {q : Rep} (h : RepRun [] [] q) : q.st = {} :=
  (congrArg Prod.fst h.1).symm

/-- **At rep 1's first round** its frame is the frame's start after rep 0's algebraic-phase coins, taken at any
configuration of its layout, as the loop's (`Loop.freshOf`, at the placeholder's), where the coins are the same. -/
theorem repRun_prior_fresh {k : Nat} {b : ByteArray} {q : Rep} (h : RepRun [.prior k b] [] q) {c' : Cfg}
    (hc : CfgAlike q.cfg c') {st' : _root_.Flock.Firewall.Frame.State}
    (hp : _root_.Flock.Firewall.Frame.step c' {} (.prior k b) = (.ok, st')) : q.st = st' := by
  have h1 := h.1
  have e : _root_.Flock.Firewall.Frame.run q.cfg {} ([.prior k b] ++ evReqs []) = (st', false) := by
    show _root_.Flock.Firewall.Frame.run q.cfg {} [.prior k b] = (st', false)
    rw [run_cons, step_prior_alike hc, hp]
    all_goals rfl
  rw [e] at h1
  exact (congrArg Prod.fst h1).symm

end Claims.ZKServer.Relay
