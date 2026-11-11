module

public import Proofs.Flock.ZkAny.RelayLive
public import Proofs.Flock.ZkAny.RelayCoins

/-!
# The session's stop from one request to the next

The loop's frames and the model stop a session at the same round where they agree at each of its requests: the
session's stop (`Frames.Sessions.stopped`) is whether an earlier round failed (`ZkServer.liveBefore`). Here is that
invariant's start and its step from one round to the session's next request, over the loop's updates in between:

* `stopped_fresh`, `stopped_first`: a session none of whose frames is in the loop's frames has not stopped, nor has it
  after updates that keep its stop (`SessKept`: its `Hello`'s `Sessions.begin`, its `Commit`'s link, the frames port's
  starts, the other sessions' requests). `liveBefore_two`: and at its first frame round every earlier round passed,
  where the worker's link is the ZK server's own; **`stopped_two`**: so there the session's stop is the model's.
* `roundPlace_nd`: a stopped session's round takes no device round, at every place; `roundPlace_nd_eq`: and a round
  of a session that has not stopped takes the device's.
* `stopped_put_kept`: a put after updates that keep the rep's frame and the session's stop stops the session as on the
  frames before them, where a stopped session's put frame is stopped; `roundStep_put_stopped`: a round's put frame
  is. `stopped_round_coins`: so from a round's read to the session's next request, through the round's put on the
  frames it lands on, its coins and the updates in between, the session's stop is the round's put's on the frames the
  round read.
* **`stopped_next`**: at a round that releases, the session's stop at its next request is the round's put's on the
  frames the round read, which `roundStep_nextD` gives as the model's one round on.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **A session none of whose frames is in the loop's frames has not stopped.** -/
theorem stopped_fresh {R : Sessions} {sid : Nat} (h : ∀ p ∈ R.reps, p.sid ≠ sid) : R.stopped sid = false := by
  cases hs : R.stopped sid
  · rfl
  · obtain ⟨p, hp, hps, -⟩ := stopped_iff.mp hs
    exact absurd hps (h p hp)

/-- **A session has not stopped before its first round**: where none of its frames was in the loop's frames before its
`Hello` (the loop's session ids are fresh) and only updates that keep its stop came after. -/
theorem stopped_first {R R₁ : Sessions} {sid : Nat} (h : ∀ p ∈ R.reps, p.sid ≠ sid) (hS : SessKept sid R R₁) :
    R₁.stopped sid = false :=
  (stopped_sessKept hS).trans (stopped_fresh h)

/-- **At the session's first frame round every earlier round passed**, where the worker's link is the ZK server's own:
the `Hello` passes and so does the link (`toSrv_pass_own`, `toSrv_pass_recomp`). -/
theorem liveBefore_two {W Rr : Type} (L : LeanRelease W Rr) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : Rr)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : h.length = 2) (hlt : 1 < L.last x)
    (hv : wk x w ρ (h.take 1) 0 = some (L.ownMsg x w ρ (h.take 1), false)) :
    L.toSrv.liveBefore wk x w ρ κ h = true := by
  have h1 : (h.take 1).length = 1 := by
    rw [List.length_take, hl]
    omega
  unfold ZkServer.liveBefore
  rw [hl]
  simp only [List.all_eq_true, List.mem_range]
  intro j hj
  rcases (by omega : j = 0 ∨ j = 1) with rfl | rfl
  · refine L.toSrv_pass_own wk x w ρ κ (h.take 0) ?_ ?_
    · rw [List.take_zero, List.length_nil]
      exact L.kindAt_zero x
    · rw [List.take_zero, List.length_nil]
      omega
  · refine L.toSrv_pass_recomp wk x w ρ κ (h.take 1) ?_ ?_ hv
    · rw [h1]
      simp [LeanRelease.kindAt]
    · rw [h1]
      exact hlt

/-- **At the session's first frame round its stop is the model's**: neither has stopped, where none of the session's
frames was in the loop's frames before its `Hello`, only updates that keep its stop came after, and the worker's link
is the ZK server's own. -/
theorem stopped_two {W Rr : Type} (L : LeanRelease W Rr) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : Rr)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : h.length = 2) (hlt : 1 < L.last x)
    (hv : wk x w ρ (h.take 1) 0 = some (L.ownMsg x w ρ (h.take 1), false)) {R R₁ : Sessions} {sid : Nat}
    (hfresh : ∀ p ∈ R.reps, p.sid ≠ sid) (hS : SessKept sid R R₁) :
    R₁.stopped sid = !L.toSrv.liveBefore wk x w ρ κ h := by
  rw [stopped_first hfresh hS, liveBefore_two L wk x w ρ κ h hl hlt hv, Bool.not_true]

set_option maxHeartbeats 1000000 in
/-- **A stopped session's round takes no device round**, at every place `roundPlace` readies. -/
theorem roundPlace_nd {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String}
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hR : R.stopped sid = true) : nd.isSome = true := by
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  rcases hf : R.find sid i r with _ | ⟨j, p⟩
  · simp only [hf] at h
    cases h
  simp only [hf, hR, bind, Except.bind, pure, Except.pure, ↓reduceIte] at h
  by_cases hfst : (p.coins.isEmpty && !p.st.begun) = true
  all_goals repeat' (first
    | split at h
    | simp only [eq_true hfst, ↓reduceIte] at h
    | simp only [eq_false hfst, ↓reduceIte] at h
    | simp only at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h; rfl)

set_option maxHeartbeats 1000000 in
/-- **A round takes no device round exactly where its session stopped**, at every place `roundPlace` readies: in a
session that has not stopped it takes the device's round wherever it has a place, the inner proof's rounds included,
since the loop refuses one its finish has not computed. -/
theorem roundPlace_nd_eq {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String}
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    nd.isSome = R.stopped sid := by
  cases hR : R.stopped sid
  · unfold _root_.Flock.Firewall.Frames.roundPlace at h
    rcases hf : R.find sid i r with _ | ⟨j, p⟩
    · simp only [hf] at h
      cases h
    simp only [hf, hR, bind, Except.bind, pure, Except.pure, Bool.false_eq_true, ↓reduceIte] at h
    by_cases hfst : (p.coins.isEmpty && !p.st.begun) = true
    all_goals repeat' (first
      | split at h
      | simp only [eq_true hfst, ↓reduceIte] at h
      | simp only [eq_false hfst, ↓reduceIte] at h
      | simp only at h)
    all_goals first
      | (cases h; done)
      | exact absurd (by assumption) throw_ne_ok
      | (cases h; rfl)
  · exact roundPlace_nd h hR

/-- **A put after updates that keep the rep's frame and the session's stop stops the session as on the frames before
them**, where the frame put at the rep's key is stopped wherever the session had stopped. -/
theorem stopped_put_kept {R R₁ : Sessions} {sid i r : Nat} {p₁ : Rep} (hfr : (frameAt R sid i r).isSome)
    (hk : keyOf p₁ = (sid, i, r)) (hK : Kept sid i r R R₁) (hS : SessKept sid R R₁)
    (hp : R.stopped sid = true → p₁.st.m.stopped = true) :
    (R₁.put p₁).stopped sid = (R.put p₁).stopped sid := by
  simp only [keyOf, Prod.mk.injEq] at hk
  obtain ⟨rfl, rfl, rfl⟩ := hk
  have hfr₁ : (frameAt R₁ p₁.sid p₁.table p₁.rep).isSome := by
    rw [frameAt_kept hK hfr]
    exact hfr
  cases hps : p₁.st.m.stopped
  · have hR : R.stopped p₁.sid = false := by
      cases hR : R.stopped p₁.sid
      · rfl
      · have e := hp hR
        rw [hps] at e
        cases e
    have hR₁ : R₁.stopped p₁.sid = false := (stopped_sessKept hS).trans hR
    rw [stopped_put_live hR₁ hps, stopped_put_live hR hps]
  · rw [stopped_put_self hfr₁ hps, stopped_put_self hfr hps]

/-- **A round of a stopped session puts back a stopped frame**: it reaches the frame as no device round, which the
frame's check fails (`Ranked.check_empty`). -/
theorem roundStep_put_stopped {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q p₁ : Rep} {rr : ByteArray} {f : Bool} {nd : Option String}
    {o : ByteArray} (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, .sched rr, f, nd))
    (hnd : R.stopped sid = true → nd.isSome = true) (hq : R.stopped sid = false → q.st.m.stopped = false)
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) :
    R.stopped sid = true → p₁.st.m.stopped = true := by
  intro hR
  obtain ⟨out, why, hc, -⟩ := roundStep_stopped hfr hrp hnd hq ho hput
  rw [if_pos (hnd hR)] at hc
  exact Ranked.check_empty hc

/-- **From a round's read to the session's next request, the session's stop is the round's put's on the frames the
round read**: where the loop's updates between the round's read and its put (onto `R₁`), between its put and its coins
(to `R₂`) and after the coins' put up to the session's next request (to `R₃`) keep the rep's frame and the session's
stop, and the frame the round puts back is stopped wherever the session had stopped. -/
theorem stopped_round_coins {R R₁ R₂ R₃ : Sessions} {sid i r : Nat} {p₁ : Rep} (hfr : (frameAt R sid i r).isSome)
    (hk : keyOf p₁ = (sid, i, r)) (hK₀₁ : Kept sid i r R R₁) (hS₀₁ : SessKept sid R R₁)
    (hK₁₂ : Kept sid i r (R₁.put p₁) R₂) (hS₁₂ : SessKept sid (R₁.put p₁) R₂) {cs : Array F128}
    {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂))
    (hS₂₃ : SessKept sid (R₂.put (Loop.coinsOf p₁ st₂ cs)) R₃)
    (hp : R.stopped sid = true → p₁.st.m.stopped = true) :
    R₃.stopped sid = (R.put p₁).stopped sid := by
  rw [stopped_sessKept hS₂₃, stopped_coins hfr hk hK₀₁ hK₁₂ hcs sid, stopped_sessKept hS₁₂,
    stopped_put_kept hfr hk hK₀₁ hS₀₁ hp]

/-- **At a round that releases, the session's stop at its next request is the round's put's on the frames the round
read**: at a round of the session's rep `(i, r)` at a scheduled place, where a live session's frame is not stopped,
through the loop's updates that keep the rep's frame and the session's stop, between the round's read and its put,
between its put and its coins, and after them up to the session's next request (`stopped_round_coins`). -/
theorem stopped_next {R R₁ R₂ R₃ : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q p₁ : Rep} {rr : ByteArray} {f : Bool} {nd : Option String}
    {o : ByteArray} (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, .sched rr, f, nd))
    (hq : R.stopped sid = false → q.st.m.stopped = false)
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁)
    (hK₀₁ : Kept sid i r R R₁) (hS₀₁ : SessKept sid R R₁) (hK₁₂ : Kept sid i r (R₁.put p₁) R₂)
    (hS₁₂ : SessKept sid (R₁.put p₁) R₂) {cs : Array F128} {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂))
    (hS₂₃ : SessKept sid (R₂.put (Loop.coinsOf p₁ st₂ cs)) R₃) :
    R₃.stopped sid = (R.put p₁).stopped sid :=
  stopped_round_coins hfr (roundStep_key hput) hK₀₁ hS₀₁ hK₁₂ hS₁₂ hcs hS₂₃
    (roundStep_put_stopped hfr hrp (roundPlace_nd hrp) hq ho hput)

end Claims.ZKServer.Relay
