module

public import Proofs.Flock.ZkAny.RelayGap
public import Proofs.Flock.ZkAny.RelayFresh

/-!
# A rep's frame after its stream's first round, whichever rep's round comes next

`roundRun_last` carries a rep's frame past a round after the rep's first. At the stream's first round the frame the
round finds is as `Frames.startRep` pushed it (`FreshRep`), and the round leaves it the same three things: its run, the
history's words for the rep's rounds so far and its schedule so far. So a stream of one round hands its rep's frame to
the next stream's first round as a stream of more does:

* **`roundRun_last_first`**: at a `RoundRun` of a stream's first round on a fresh frame, whose round releases and
  whose coins the rep's frame took, the frame the session's next request finds is its run one round on, holds the
  history's words for the rep's one round and keeps its schedule so far, whichever rep's round comes next.
* `count_first`: where the round after a stream's first is the next stream's, the stream has one round.
* **`roundRun_handoff_first`**: so after a stream of one round, until the next stream's first, the frames hold the
  rep's frame with the history's words for its one round, as many as its count (`roundRun_handoff` at one round).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 Stmt)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Req Place)

open Loop in
/-- **A rep's frame keeps its run, words and schedule so far after its stream's first round**, whichever rep's round
comes next: at a `RoundRun` at the loop's count whose content is the worker's message `v`, on the rep's frame `p` as
`Frames.startRep` pushed it, whose round releases and whose coins the rep's frame took off the auditor's answer `a`,
where the frame `roundPlace` readied is its run on `pre` and no rounds, at the table and extras its session gives it
and the salts the loop's salts give it where `withAlg` scheduled no rounds (`hsalt`, where `placeOf` reads them), the
table's commitment's cap the level-0 root, the frame the session's next request finds is its run one round on, holds
the history's words with `(v, a)` for the rep's one round, and keeps its schedule so far. -/
theorem roundRun_last_first (x : Msg) (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v : Msg) {q₀ : Nat}
    (hq : h1.length = q₀ + 1) {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray}
    {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {pre : List Req} (hR : RepRun pre [] q) {p : Rep} (hfa : frameAt R sid i r = some p) (hp : FreshRep p)
    {s : _root_.Flock.Firewall.Frames.Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) {p' : Rep}
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') {t : _root_.Flock.Firewall.Finish.Tab}
    {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt} {c : Array F128} {r0 : ByteArray} {q0 : Nat}
    (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p'.alg.size = 0 → R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c))
    {sched : Array (ByteArray × Nat)}
    (hsc : schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched) :
    ∃ p₃, frameAt R₃ sid i r = some p₃ ∧
      RepRun pre [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray), pl',
        _root_.Flock.Firewall.Frames.feedOf (loopNr x h1.length) a.data.toList)] p₃ ∧
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
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round (loopNr x h1.length) (if nd.isSome then ByteArray.empty else ⟨v.toArray⟩) pl') with ⟨a', st'⟩
  obtain ⟨w, hw'⟩ := roundStep_put hrp hres
  rw [hw', Option.some.injEq] at hput
  subst hput
  have hq0 : q.coins = #[] := Array.toList_inj.mp hR.2
  have hr1 : Array.range 1 = (Array.range 0).push 0 := by
    rw [Array.push_eq_append]
    exact Array.range_succ
  show q.coins.push (_root_.Flock.Firewall.Frames.feedOf (loopNr x h1.length) a.data.toList) = _
  rw [hq0, feedOf_words x e0 h1 v a.data.toList hq, hr1, Array.map_push, Array.range_eq_empty_iff.mpr rfl,
    Array.map_empty, show q₀ + 1 - 1 + 0 + 1 = q₀ + 1 by omega]

/-- **A stream of one round**: where the round after the `q`th is the next stream's and the `q`th is its stream's
first, the stream before the next one has one round. -/
theorem count_first {rs : List Nat} {q : Nat} (hj : (ZKMal.CoinLayer.locate rs 0 q).2 = 0)
    (hnext : (ZKMal.CoinLayer.locate rs 0 (q + 1)).1 = (ZKMal.CoinLayer.locate rs 0 q).1 + 1) :
    rs.getD ((ZKMal.CoinLayer.locate rs 0 (q + 1)).1 - 1) 0 = 1 := by
  have h := locate_last rs 0 q (by rw [hnext]; omega)
  rw [hnext, Nat.add_sub_cancel]
  rw [Nat.sub_zero, hj] at h
  omega

open Loop in
/-- **A rep's frame from a stream of one round to the next stream's first**: at `roundRun_last_first`'s `RoundRun`, the
round its stream's first (`hj₀`), where the round after it is the next stream's (`hnext`), after updates from the
frames the session's next round reads that keep the rep's frame, the frames find the rep's frame holding the history's
words with `(v, a)` for the stream's one round, as many as the stream's count. -/
theorem roundRun_handoff_first (x : Msg) (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v : Msg) {rs : List Nat}
    {q₀ : Nat} (hq : h1.length = q₀ + 1) (hj₀ : (ZKMal.CoinLayer.locate rs 0 q₀).2 = 0)
    (hnext : (ZKMal.CoinLayer.locate rs 0 (q₀ + 1)).1 = (ZKMal.CoinLayer.locate rs 0 q₀).1 + 1)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray} {e : Env}
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
    {pre : List Req} (hR : RepRun pre [] q) {p : Rep} (hfa : frameAt R sid i r = some p) (hp : FreshRep p)
    {s : _root_.Flock.Firewall.Frames.Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) {p' : Rep}
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') {t : _root_.Flock.Firewall.Finish.Tab}
    {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt} {c : Array F128} {r0 : ByteArray} {q0 : Nat}
    (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p'.alg.size = 0 → R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c))
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

end Claims.ZKServer.Relay
