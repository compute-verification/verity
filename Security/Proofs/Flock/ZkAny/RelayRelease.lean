module

public import Proofs.Flock.ZkAny.RelayStop
public import Proofs.Flock.Claims.ZK.FrameProg

/-!
# A released round is the model's, at the session's stop

The model releases a session's frame round from the device's round while every round so far passed, and once one failed
the scheduled round as a stopped rep's (`LeanRelease.check_release`, at the stand-in `fillOf`). The loop's frame does
the same at the session's stop (`Frames.Sessions.stopped`), where a stopped session's frames take no device round:

* **`release_of_check`**: at a frame whose configuration is the model's and whose masking state is the model's
  (`LeanRelease.answered`) but for `stopped`, against the model's scheduled round and coin count, what the frame's
  check releases is the model's release: of the device round where the session has not stopped and the round passes
  (`LeanRelease.pass`), else of the scheduled round as a stopped rep's.
* **`roundStep_release`**: so a round of the session that the loop's frame releases at the model's scheduled round, with
  a device round where the session has not stopped, is the model's release, after the stream's domain frame at its
  first round, and after it the session is stopped where it had stopped or the round fails `LeanRelease.pass`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Frame (Masker)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **What the frame's check releases is the model's release**, where the session stopped (`b`, no device round) or
not (the frame not stopped). -/
theorem release_of_check {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg))
    {m : Masker} (hm : L.answered x ρ h = .ok m) {s : Masker} (hu : unstop s = unstop m) {b : Bool}
    (hb : b = false → s.stopped = false) {v : Msg} {out : ByteArray} {s' : Masker} {why : Option String}
    (hc : _root_.Flock.Firewall.Frame.check (L.cfg x ρ h) s (if b then ByteArray.empty else ⟨v.toArray⟩)
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) = .ok (out, s', why)) :
    0 :: out.data.toList = if (!b && L.pass x w ρ h v) = true then L.release x ρ h false v
      else L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h))) := by
  have hs := eq_unstop hu.symm
  cases b with
  | false =>
    rw [ite_eq_right Bool.false_ne_true] at hc
    rw [hb rfl] at hs
    subst hs
    obtain ⟨hp, hr⟩ := L.check_release x w ρ h v hm hc
    rw [hr, hp]
    try simp only [Bool.not_false, Bool.true_and]
  | true =>
    rw [ite_eq_left rfl] at hc
    have hw : why.isSome = true := by
      cases why with
      | none =>
        have hr := check_none hc
        unfold _root_.Flock.Firewall.Frame.round at hr
        obtain ⟨q, hq, -⟩ := bindV_ok hr
        exact absurd hq (Ranked.roundFree_empty _ _ _ q)
      | some _ => rfl
    obtain ⟨w', rfl⟩ := Option.isSome_iff_exists.mp hw
    have e : ({ s with stopped := true } : Masker) = { m with stopped := true } := by
      have e' := congrArg (fun t : Masker => ({ t with stopped := true } : Masker)) hu
      exact e'
    have hr := check_some hc
    rw [e] at hr
    simp only [Bool.not_true, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
    unfold LeanRelease.release LeanRelease.step
    rw [hm]
    show 0 :: out.data.toList = match _root_.Flock.Firewall.Frame.round (L.cfg x ρ h) { m with stopped := true }
      ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) with
      | .ok (out, _) => 0 :: out.data.toList
      | .error _ => L.stop
    rw [hr]

/-- **A released round of the session is the model's release, and stops the session where the model's does**: at a
round of the session's rep `(i, r)` whose place is the model's scheduled round and coin count, where the session's
frames take no device round exactly once it stopped (`nd`), a frame of a live session is not stopped, and the frame's
configuration and masking state are the model's but for `stopped`, what the round releases is the model's release of the
device round `v` where the session has not stopped and `v` passes, else of the scheduled round as a stopped rep's, after
the stream's domain frame at its first round; after its put the session is stopped where it was or `v` fails. -/
theorem roundStep_release {W Rr : Type} (L : LeanRelease W Rr) (x : Msg) (w : W) (ρ : Rr) (h : List (Msg × Msg))
    {m : Masker} (hm : L.answered x ρ h = .ok m) {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots}
    {sid i r : Nat} {dom : ByteArray} (v : Msg) {q p₁ : Rep} {f : Bool} {nd : Option String} {o : ByteArray}
    (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length) =
      .ok (q, .sched ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩, f, nd))
    (hnd : nd.isSome = R.stopped sid) (hq : R.stopped sid = false → q.st.m.stopped = false)
    (hcfg : q.cfg = L.cfg x ρ h) (hu : unstop q.st.m = unstop m)
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).put =
      some p₁) :
    ∃ out, o = (if f then dom ++ out else out) ∧
      0 :: out.data.toList = (if (!R.stopped sid && L.pass x w ρ h v) = true then L.release x ρ h false v
        else L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h)))) ∧
      (R.put p₁).stopped sid = (R.stopped sid || !L.pass x w ρ h v) := by
  obtain ⟨out, why, hc, hst⟩ := roundStep_stopped hfr hrp (fun e => hnd.trans e) hq ho hput
  have hoo : o = if f then dom ++ out else out := by
    obtain ⟨out', s', why', hc', ho'⟩ := roundStep_check hrp ho
    rw [hc] at hc'
    simp only [Except.ok.injEq, Prod.mk.injEq] at hc'
    rw [ho', hc'.1]
  rw [hnd, hcfg] at hc
  refine ⟨out, hoo, release_of_check L x w ρ h hm hu hq hc, ?_⟩
  rw [hst]
  cases hR : R.stopped sid
  · rw [hR, ite_eq_right Bool.false_ne_true] at hc
    have hs : q.st.m = { m with stopped := false } := by
      have e := eq_unstop hu.symm
      rw [hq hR] at e
      exact e
    rw [hs] at hc
    obtain ⟨hp, -⟩ := L.check_release x w ρ h v hm hc
    rw [hp]
    cases why <;> rfl
  · rfl

end Claims.ZKServer.Relay
