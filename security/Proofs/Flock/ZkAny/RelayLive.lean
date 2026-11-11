module

public import Proofs.Flock.ZkAny.RelayRelease
public import Proofs.Flock.ZkAny.RelayFirst
public import Proofs.Flock.Claims.ZK.LiveBefore

/-!
# A released frame round is the deferred model's message

The deferred release stands in for the worker once a round of the session failed (`ZkServer.wkD`); the loop's frames
stand in once the session stopped (`Frames.Sessions.stopped`). Where the session's stop is the model's, whether an
earlier round failed (`ZkServer.liveBefore`), a frame round is one message in both:

* **`roundStep_nextD`**: at a frame round below the `Finish` whose worker value is the device round the loop's frame
  takes, where Rust's checks fail only device rounds the frame fails, what the loop's round releases is the deferred
  release's message after `h`, after the stream's domain frame at its first round, and after its put the session's
  stop is again the model's, one round on.
* `roundStep_put_some`: a round at a place `roundPlace` readies puts a frame back. **`envOk_roundStep`**: so where what
  the relay read for a round is the loop's round on its frames, whose release is a message's after the stream's domain
  frame at its first round (as `roundStep_nextD` gives the deferred model's), and the loop's first round of the stream
  and its domain frame are the schedule's and the setup's, what the relay read there is that message's (`EnvOk`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (SessionSpec)
open _root_.Flock.Firewall.Frame (Masker Place)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **A released frame round of the session is the deferred model's message, and keeps the session's stop the
model's**: at a frame round below the `Finish` of the session's rep `(i, r)`, whose worker value is the device round `v`
the loop's frame takes, where Rust's checks fail only device rounds the frame fails and the round's place, configuration
and masking state are the model's (`roundStep_release`'s premises), and where the session has stopped exactly where an
earlier round failed, what the round releases is the deferred release's message after `h`, after the stream's domain
frame at its first round; after its put the session has stopped exactly where a round up to this one failed. -/
theorem roundStep_nextD {W Rr : Type} (L : LeanRelease W Rr) (n : Msg → ℕ) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : Rr) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hn : h.length + 1 < n x) (hlt : h.length < L.last x) {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots}
    {sid i r : Nat} {dom : ByteArray} {q p₁ : Rep} {f : Bool} {nd : Option String} {o : ByteArray}
    (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length) =
      .ok (q, .sched ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩, f, nd))
    (hnd : nd.isSome = R.stopped sid) (hq : R.stopped sid = false → q.st.m.stopped = false)
    (hcfg : q.cfg = L.cfg x ρ h) (hu : unstop q.st.m = unstop m)
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).put =
      some p₁)
    (hst : R.stopped sid = !L.toSrv.liveBefore wk x w ρ κ h) :
    ∃ out, o = (if f then dom ++ out else out) ∧ 0 :: out.data.toList = L.toSrv.nextD L.fillOf n wk x w ρ κ h ∧
      ∀ e, (R.put p₁).stopped sid = !L.toSrv.liveBefore wk x w ρ κ (h ++ [e]) := by
  obtain ⟨out, hoo, hrel, hst'⟩ := roundStep_release L x w ρ h hm v hfr hrp hnd hq hcfg hu ho hput
  refine ⟨out, hoo, ?_, fun e => ?_⟩
  · rw [hrel, L.nextD_frame_stop n wk x w ρ κ h hk hn hlt hv hcr hst]
  · rw [hst']
    exact L.stop_frame_snoc wk x w ρ κ h hk hlt hv hcr hst e

/-- **A round at a place `roundPlace` readies puts a frame back.** -/
theorem roundStep_put_some {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    ∃ p₁, (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁ := by
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl) with ⟨a, st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  exact ⟨_, hv⟩

/-- **What the relay read for a round is the message the loop's round releases**: where the relay's read of the round
(`Env.round`) is the loop's round on its frames at a place `roundPlace` readies, whose release, wherever it puts a frame
back, is the message `M`'s after the stream's domain frame at its first round (`roundStep_nextD`), and the loop's
first round of the stream is the schedule's and its domain frame the setup's. -/
theorem envOk_roundStep {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    {spec : SessionSpec} {seen : List String} {stream : String} {k : Nat} {e : Env} {M : Msg}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (he : e.round = (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out)
    (hrel : ∀ o p₁, (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o →
      (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁ →
      ∃ out, o = (if f then dom ++ out else out) ∧ 0 :: out.data.toList = M)
    (hf : f = !seen.contains stream) (hd : dom = domOf spec stream) :
    EnvOk spec seen (.round stream k) e M := by
  refine envOk_round fun c hc => ?_
  rw [he] at hc
  obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
  obtain ⟨out, hoo, hmsg⟩ := hrel c p₁ hc hput
  exact ⟨out, hmsg, hoo.trans (leaveOf_first hf hd)⟩

end Claims.ZKServer.Relay
