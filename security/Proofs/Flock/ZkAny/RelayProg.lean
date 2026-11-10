module

public import Proofs.Flock.ZkAny.RelayRoundTook

/-!
# A rep's masking state from one session round of it to the next, the model's

At a rep's round past its first, the model checks the round at the state the rep's last coins left
(`ofLoop_frameAnswerR_same`), the loop's taking of the opening's words (`Loop.progOf`'s answer), so the frame's round in
the place of the rep's next scheduled round leaves the loop's state after it (`round_progOf`):

* `round_unstop`: a round the frame releases at a state alike the model's for it, at the model's configuration, in the
  place of the rep's next scheduled round with its coin count, leaves a state alike the loop's after the round.
* `cfg_of_rep`, `cfg_same`: the model's configuration reads the history only through its round's rep, so it is one for
  a rep's round past its first and the round before it.
* **`roundRun_step`**: so at a `RoundRun` past the rep's first round, where the frame `roundPlace` readied has the
  model's configuration and its state but for `stopped`, in the place of the rep's next scheduled round with its
  count, the model's coin step takes the round's coins (`roundRun_fed`), and where the next round is the same rep's
  the frame the session's next round finds is its run one round on, at the model's state for that round but for
  `stopped` (`roundRun_carry`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Cfg Masker Req Place)

open Loop LeanRelease in
/-- **A round past the rep's first, at the model's state for it, leaves the loop's state after it**: where the model
checks the round after `e0 :: h1` at `m` before the rank stop (`answeredR`), a rep's round past its first, and the
frame releases a device round at a state alike `m`, at the model's configuration, in the place of the rep's next
scheduled round with its coin count, its state after the round is the loop's after the coins with any next coin, but
for `stopped`. -/
theorem round_unstop (T : ZKMal.CoinLayer.TableData)
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
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c') {rsched sched' : Array (ByteArray × Nat)}
    (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((Array.range (j + 1)).map fun i =>
        wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok sched')
    (hlt : j + 1 < rsched.size) {m : Masker} (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: h1) = .ok m)
    {st st' : _root_.Flock.Firewall.Frame.State} (hu : unstop st.m = unstop m) {v out : ByteArray} {vd : String}
    (h : _root_.Flock.Firewall.Frame.step ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) st
      (.round (rsched.getD (j + 1) (.empty, 0)).2 v (.sched (rsched.getD (j + 1) (.empty, 0)).1)) =
        (.round out vd, st')) :
    unstop st'.m = unstop (progOf x ((h1.map fun p => ZKMal.CoinLayer.rd p.2) ++ [d])) := by
  obtain ⟨-, -, ha⟩ := answeredR_ok hA
  rw [ofLoop_frameAnswerR_same T L₀ real x ρ e0 h1 hx hq hj] at ha
  have hlen : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  have hw : coinsAt x (e0 :: h1) = wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) q := by
    obtain ⟨m0, r0⟩ := e0
    have hcr : coinsAt x ((m0, r0) :: h1) = readCoins x (h1.map fun p => ZKMal.CoinLayer.rd p.2) :=
      loop_coinsRead loopNr T L₀ x m0 r0 h1
    rw [hcr]
    exact readCoins_eq x _ hlen
  rw [hw] at ha
  exact round_progOf x _ d hx hlen hph hc hj hrs hs' hlt ha hu h

/-- **The model's configuration reads the history only through its round's rep.** -/
theorem cfg_of_rep {W Rr : Type} (L : LeanRelease W Rr) (x : Msg) (ρ : Rr) {h h' : List (Msg × Msg)}
    (hr : L.rep x h.length = L.rep x h'.length) : L.cfg x ρ h = L.cfg x ρ h' := by
  unfold LeanRelease.cfg
  rw [hr]

open Loop LeanRelease in
/-- **The model's configuration for a rep's round is the one for the same rep's next**: where the round after
`e0 :: h1` and one more message is past its rep's first (`hj`), the round after `e0 :: h1` is the same rep's, so the
model's configuration is one for both histories. -/
theorem cfg_same (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q₀ j : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1) (e : Msg × Msg) :
    (ofLoop T L₀ real).cfg x ρ (e0 :: h1) = (ofLoop T L₀ real).cfg x ρ (e0 :: (h1 ++ [e])) := by
  apply cfg_of_rep
  have e1 : (e0 :: h1).length = q₀ + 2 := by
    simp only [List.length_cons]
    omega
  have e2 : (e0 :: (h1 ++ [e])).length = q₀ + 1 + 2 := by
    simp only [List.length_cons, List.length_append, List.length_nil]
    omega
  have hj' : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).2 = j + 1 := by
    rw [← hq]
    exact hj
  show repOf x (e0 :: h1).length = repOf x (e0 :: (h1 ++ [e])).length
  rw [e1, e2, repOf_add_two hx, repOf_add_two hx]
  have hp := congrArg Prod.fst (locate_pred (roundsOf spec sts) 0 q₀ j hj')
  exact hp

open Loop LeanRelease in
/-- **A rep's frame at the session's next round, at the model's state, from the model's state at this one**: at a
`RoundRun` at the model's count whose content is the worker's message `v`, past the rep's first round (`hj₀`), whose
round releases and whose coins the rep's frame took off the auditor's answer `a`, where the frame `roundPlace` readied
has the model's configuration and its state but for `stopped`, in the place of the rep's next scheduled round on its
own table (`rsched`, of one shape with the placeholder's) with its count, the model's coin step takes the coins after
the history with `(v, a)` (`roundRun_fed`), and where the next round is the same rep's (`hj`) the frame the session's
next round finds is its run one round on (`roundRun_next`), at the model's state for that history but for
`stopped`. -/
theorem roundRun_step (T : ZKMal.CoinLayer.TableData)
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
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c') {rsched sched' : Array (ByteArray × Nat)}
    (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 % 2) dom xs hi
      ((Array.range (j₀ + 1)).map fun i =>
        wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q₀ - (j₀ + 1) + i + 1)) = .ok sched')
    (hlt : j₀ + 1 < rsched.size) {m₀ : Masker} (hA₀ : (ofLoop T L₀ real).answeredR x ρ (e0 :: h1) = .ok m₀)
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
    (hcq : q.cfg = (ofLoop T L₀ real).cfg x ρ (e0 :: h1)) (hu₀ : unstop q.st.m = unstop m₀)
    (hpl : pl' = .sched (rsched.getD (j₀ + 1) (.empty, 0)).1)
    (hn : loopNr x h1.length = (rsched.getD (j₀ + 1) (.empty, 0)).2) :
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
    rw [hcq, hn, hpl] at hres
    rw [List.map_append, List.map_cons, List.map_nil]
    exact round_unstop T L₀ real x ρ e0 h1 _ hx hq hj₀ hph hc hrs hs' hlt hA₀ hu₀ hres
  refine ⟨roundRun_fed T L₀ real x ρ e0 h1 hr hreq hfr hrp ho hput ha hb hs hco hR hcq hu, fun j hj m hA => ?_⟩
  have hcfg : CfgAlike q.cfg ((ofLoop T L₀ real).cfg x ρ (e0 :: (h1 ++ [(v, a.data.toList)]))) := by
    rw [hcq, cfg_same T L₀ real x ρ e0 h1 hx hq hj (v, a.data.toList)]
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  exact roundRun_carry T L₀ real x ρ e0 h1 hx hj hr hreq hfr hrp ho hput ha hb hs hco hR hcfg hu hA

end Claims.ZKServer.Relay
