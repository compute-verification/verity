import Proofs.Flock.Claims.ZK.FrameInner

/-!
# The release at the loop's frames is live past the link

`Flock.Guarantees.SessionLoopZK` bounds what a run releases while it is live (`ZkServer.live`), so a release under which
no run is live past some round makes the bound say nothing past it. The first `LeanRelease.ofLoop` was one: its frame
took each round's coins at the rep of the round after it, so a rep's first round owed none, the frame refused the coins,
and no history past `Hello` was live. This file pins that the corrected release is live there:

* `toSrv_live_honest`: at any release, the honest worker's run is live on a history short of the `Finish` wherever each
  frame round so far passes the frame's own checks at its device round (`roundOk`: `cross`, `pass`, `takes`).
* `roundOk_of_check`: a frame round passes them where the frame takes the stream's coins at a state `m` and the deployed
  `Frame.check` passes the device round there, against the firewall's scheduled round.
* At `ofLoop` the frame takes the coins at the deployed loop's states: the link's at the frame's start, as the `Commit`'s
  coins are the session's (`ofLoop_answered_link`, so the link is the firewall's link and not the stop,
  `ofLoop_ownMsg_link`); rep 0's first round at the frame's start (`ofLoop_answered_first`); a rep's later round at the
  state the rep's own last coins leave (`ofLoop_answered_same`).
* `ofLoop_live_three`: so the history of `Hello`, the link and rep 0's first round is live at the honest worker, through
  rep 0's second round, wherever the deployed frame's check passes rep 0's first two device rounds at those states.
-/

namespace Claims.ZKServer

open _root_.Flock (V Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (Cfg Masker)

namespace LeanRelease

/-- The device round after `g` at honest workers. -/
def devOf {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (g : List (Msg × Msg)) : Msg :=
  L.device x w ρ g (computedOf (L.computed x w ρ g))

/-- The honest worker: each slot's honest value. -/
def honestWk {W R : Type} (L : LeanRelease W R) : L.toSrv.Worker :=
  fun x w ρ h s => some (L.toSrv.honest x h.length s w ρ h)

/-- **The frame round after `g` passes the frame's checks** at its honest device round: the cross-round check, the frame's
pass decision and its taking of the round. -/
def roundOk {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (g : List (Msg × Msg)) : Bool :=
  L.cross x g.length w ρ g (L.devOf x w ρ g) && L.pass x w ρ g (L.devOf x w ρ g) && L.takes x ρ g (L.devOf x w ρ g)

/-- The honest worker's value passes round `g.length`'s check short of the `Finish`, where a frame round there passes
the frame's checks. -/
theorem toSrv_check_honest {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (κ : L.toSrv.Chk)
    (g : List (Msg × Msg)) (cp : ℕ → ℕ → Option L.toSrv.Val)
    (hm : L.kindAt x g.length = .masked → L.roundOk x w ρ g = true) (hl : g.length < L.last x) :
    L.toSrv.check x g.length 0 w ρ g cp κ (L.honestWk x w ρ g 0) = true := by
  have hk : L.toSrv.kind x g.length 0 = L.kindAt x g.length := rfl
  unfold ZkServer.check
  rw [hk]
  cases hkd : L.kindAt x g.length with
  | fixed =>
    exfalso
    unfold kindAt at hkd
    split_ifs at hkd
  | own => rfl
  | recomputed =>
    apply decide_eq_true
    show some (if L.kindAt x g.length = .masked then (L.devOf x w ρ g, false) else (L.ownMsg x w ρ g, false)) =
      some (L.ownMsg x w ρ g, false)
    rw [hkd, if_neg nofun]
  | masked =>
    show L.toSrv.rel x g.length 0 w ρ g cp κ (L.toSrv.honest x g.length 0 w ρ g) = true
    have hv : L.toSrv.honest x g.length 0 w ρ g = (L.devOf x w ρ g, false) := by
      show (if L.kindAt x g.length = .masked then (L.devOf x w ρ g, false) else (L.ownMsg x w ρ g, false)) = _
      rw [if_pos hkd]
    rw [hv]
    show (!false && decide (g.length < L.last x) && L.cross x g.length w ρ g (L.devOf x w ρ g) &&
      L.pass x w ρ g (L.devOf x w ρ g) && L.takes x ρ g (L.devOf x w ρ g)) = true
    rw [decide_eq_true hl]
    exact hm hkd

/-- **The honest worker's run is live short of the `Finish`** wherever every frame round so far passes the frame's
checks: `Hello` is the server's own, and the honest worker's value at the link is the firewall's own. -/
theorem toSrv_live_honest {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (κ : L.toSrv.Chk)
    (h : List (Msg × Msg)) (hl : h.length < L.last x)
    (hr : ∀ i, 2 ≤ i → i ≤ h.length → L.roundOk x w ρ (h.take i) = true) :
    L.toSrv.live L.honestWk x w ρ κ h = true := by
  unfold ZkServer.live
  rw [List.all_eq_true]
  intro i hi
  rw [List.mem_range] at hi
  have hlen : (h.take i).length = i := by rw [List.length_take]; omega
  have hnr : L.toSrv.refuses x w ρ (h.take i) = false := by
    show L.refusesFinish x w ρ (h.take i) = false
    have hne : (h.take i).length ≠ L.last x := by omega
    unfold refusesFinish
    rw [decide_eq_false hne, Bool.false_and]
  have hc := L.toSrv_check_honest x w ρ κ (h.take i) (L.toSrv.copy L.honestWk x w ρ (h.take i))
    (fun hk => by
      obtain ⟨h0, h1, -⟩ := kindAt_masked hk
      exact hr i (by omega) (by omega))
    (by omega)
  show (!L.toSrv.refuses x w ρ (h.take i) && (List.range 1).all fun s =>
      L.toSrv.check x (h.take i).length s w ρ (h.take i) (L.toSrv.copy L.honestWk x w ρ (h.take i)) κ
        (L.honestWk x w ρ (h.take i) s)) = true
  rw [hnr]
  simp only [Bool.not_false, Bool.true_and, List.all_eq_true, List.mem_range, Nat.lt_one_iff, forall_eq]
  exact hc

/-- **A frame round passes the frame's checks** where the frame takes the stream's coins at `m`, the deployed
`Frame.check` passes the device round at `m` against the firewall's scheduled round, and the cross-round check passes. -/
theorem roundOk_of_check {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (g : List (Msg × Msg))
    {m : Masker} (ha : L.answered x ρ g = .ok m) {out : ByteArray} {s' : Masker}
    (hc : _root_.Flock.Firewall.Frame.check (L.cfg x ρ g) { m with stopped := false } ⟨(L.devOf x w ρ g).toArray⟩
      ⟨(L.scheduled x w ρ g (computedOf (L.computed x w ρ g))).toArray⟩ (L.coins x g.length) = .ok (out, s', none))
    (hcr : L.cross x g.length w ρ g (L.devOf x w ρ g) = true) :
    L.roundOk x w ρ g = true := by
  have hp : L.pass x w ρ g (L.devOf x w ρ g) = true := (L.check_release x w ρ g _ ha hc).1
  have ht : L.takes x ρ g (L.devOf x w ρ g) = true := by
    unfold takes step
    rw [ha]
    simp only [bind, Except.bind]
    rw [check_none hc]
    all_goals rfl
  simp only [roundOk, hcr, hp, ht, Bool.and_self]

open Loop in
/-- **The loop's frame takes the link's coins at its start**: the `Commit`'s coins are the session's (`Registry.link`),
and no frame's. -/
theorem ofLoop_answered_link (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) :
    (ofLoop T L₀ real).answered x ρ [e0] = .ok {} := by
  obtain ⟨m0, a0⟩ := e0
  rw [answered_of_opens (ofLoop T L₀ real) x ρ [(m0, a0)] rfl, ofLoop_frameAnswer]
  unfold takeAt
  rw [if_pos (by simp)]
  show freshOf x [] = _
  unfold freshOf
  rw [hx]
  rfl

open Loop in
/-- **The link is the firewall's link**, not the stop, at a session whose `Finish` is past it. -/
theorem ofLoop_ownMsg_link (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hl : lastOf x ≠ 1) :
    (ofLoop T L₀ real).ownMsg x w ρ [e0] = (ofLoop T L₀ real).link x w ρ [e0] := by
  have hr : (ofLoop T L₀ real).refusesCoins x ρ [e0] = false := by
    unfold refusesCoins
    rw [ofLoop_answered_link T L₀ real x ρ e0 hx]
  unfold ownMsg
  rw [hr, if_neg Bool.false_ne_true]
  exact if_neg (fun he => hl he.symm)

open Loop in
/-- **The loop's frame takes rep 0's first round at its start**, as `Frames.roundPlace` hands it, where the transport's
coin check passes the history and the rep's placeholder and algebraic phase are ones. -/
theorem ofLoop_answered_first (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 e1 : Msg × Msg) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (ho : (ofLoop T L₀ real).opens x ρ [e0, e1] = true) {c' : Cfg} {t : _} {dom : ByteArray} {xs : _} {hi : _}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 0).1
      (wordsAt x ([e1].map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi))
    {alg : _}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hk : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 0).1 % 2 = 0) :
    (ofLoop T L₀ real).answered x ρ [e0, e1] = .ok {} := by
  rw [answered_of_opens _ x ρ _ ho, ofLoop_frameAnswer]
  unfold takeAt
  rw [if_pos (by simp)]
  have hq : ([e1].map fun p => ZKMal.CoinLayer.rd p.2).length = 0 + 1 := rfl
  unfold freshOf
  rw [hx, hq]
  simp only
  rw [hph]
  simp only
  rw [hal]
  simp only
  rw [if_pos hk]
  all_goals rfl

open Loop in
/-- **The loop's frame takes a rep's later round at the state the rep's own last coins leave**, where the transport's
coin check passes the history. -/
theorem ofLoop_answered_same (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q j : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1)
    (ho : (ofLoop T L₀ real).opens x ρ (e0 :: h1) = true) :
    (ofLoop T L₀ real).answered x ρ (e0 :: h1) =
      (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).answer ((ofLoop T L₀ real).cfg x ρ (e0 :: h1))
        (coinsAt x (e0 :: h1)) := by
  rw [answered_of_opens _ x ρ _ ho, ofLoop_frameAnswer_same T L₀ real x ρ e0 h1 hx hq hj]

open Loop in
/-- **The loop's release is live past the link**: at a session whose `Finish` is past rep 0's second round, the
history of `Hello`, the link and rep 0's first round is live at the honest worker, its next round (rep 0's second)
included, wherever the transport's coin check passes it, rep 0's placeholder and algebraic phase are ones, and the
deployed frame's check passes rep 0's first device round at the frame's start and its second at the state the first's
coins leave (`Frames.roundStep`, then `Frames.coinsStep`), each against the firewall's scheduled round, with the
cross-round check passing both. -/
theorem ofLoop_live_three (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (e0 e1 e2 : Msg × Msg) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hlast : 3 < lastOf x)
    (ho : (ofLoop T L₀ real).opens x ρ [e0, e1, e2] = true) (ho2 : (ofLoop T L₀ real).opens x ρ [e0, e1] = true)
    {c' : Cfg} {t : _} {dom : ByteArray} {xs : _} {hi : _}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 0).1
      (wordsAt x ([e1].map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi))
    {alg : _}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hk : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 0).1 % 2 = 0)
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 1).2 = 0 + 1)
    {out2 : ByteArray} {s2 : Masker}
    (hc2 : _root_.Flock.Firewall.Frame.check ((ofLoop T L₀ real).cfg x ρ [e0, e1]) {}
      ⟨((ofLoop T L₀ real).devOf x w ρ [e0, e1]).toArray⟩
      ⟨((ofLoop T L₀ real).scheduled x w ρ [e0, e1]
        (computedOf ((ofLoop T L₀ real).computed x w ρ [e0, e1]))).toArray⟩
      ((ofLoop T L₀ real).coins x [e0, e1].length) = .ok (out2, s2, none))
    (hcr2 : (ofLoop T L₀ real).cross x [e0, e1].length w ρ [e0, e1] ((ofLoop T L₀ real).devOf x w ρ [e0, e1]) = true)
    {m3 : Masker}
    (ha3 : (progOf x ([e1, e2].map fun p => ZKMal.CoinLayer.rd p.2)).answer
      ((ofLoop T L₀ real).cfg x ρ [e0, e1, e2]) (coinsAt x [e0, e1, e2]) = .ok m3)
    {out3 : ByteArray} {s3 : Masker}
    (hc3 : _root_.Flock.Firewall.Frame.check ((ofLoop T L₀ real).cfg x ρ [e0, e1, e2]) { m3 with stopped := false }
      ⟨((ofLoop T L₀ real).devOf x w ρ [e0, e1, e2]).toArray⟩
      ⟨((ofLoop T L₀ real).scheduled x w ρ [e0, e1, e2]
        (computedOf ((ofLoop T L₀ real).computed x w ρ [e0, e1, e2]))).toArray⟩
      ((ofLoop T L₀ real).coins x [e0, e1, e2].length) = .ok (out3, s3, none))
    (hcr3 : (ofLoop T L₀ real).cross x [e0, e1, e2].length w ρ [e0, e1, e2]
      ((ofLoop T L₀ real).devOf x w ρ [e0, e1, e2]) = true) :
    (ofLoop T L₀ real).toSrv.live (ofLoop T L₀ real).honestWk x w ρ κ [e0, e1, e2] = true := by
  have hr2 : (ofLoop T L₀ real).roundOk x w ρ [e0, e1] = true :=
    roundOk_of_check _ x w ρ [e0, e1] (ofLoop_answered_first T L₀ real x ρ e0 e1 hx ho2 hph hal hk) hc2 hcr2
  have ha : (ofLoop T L₀ real).answered x ρ [e0, e1, e2] = .ok m3 := by
    rw [ofLoop_answered_same T L₀ real x ρ e0 [e1, e2] hx rfl hj ho, ha3]
  have hr3 : (ofLoop T L₀ real).roundOk x w ρ [e0, e1, e2] = true := roundOk_of_check _ x w ρ _ ha hc3 hcr3
  refine toSrv_live_honest _ x w ρ κ _ hlast ?_
  intro i h2 h3
  obtain rfl | rfl : i = 2 ∨ i = 3 := by simp only [List.length_cons, List.length_nil] at h3; omega
  · exact hr2
  · exact hr3

end LeanRelease

end Claims.ZKServer
