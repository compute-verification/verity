module

public import Proofs.Flock.ZkAny.RelayWords

/-!
# A rep's frame keeps its configuration, and a round's count is its frame's schedule's

* `roundRun_cfg`: at a `RoundRun` whose round releases and whose coins the rep's frame took, the frame the session's
  next round finds has the found frame's configuration (`roundPlace_cfg`, `roundStep_cfg`, and `Loop.coinsOf`).
* `count_roundsAt`: a count that is the placeholder's schedule's at a round is the frame's own schedule's there, the
  two being of one shape (`Sched.RoundAt` holds the counts equal).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Req Place)

/-- **A rep's frame keeps its configuration from one round to the next**: at a `RoundRun` whose round releases and
whose coins the rep's frame took off the auditor's answer, the frame the session's next round finds has the
configuration of the frame the round found. -/
theorem roundRun_cfg {sid : Nat} {st : State} {pl : Placed} {stream : String} {n i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream n i r dom content e R L0 R₃) (hreq : st.request pl (.round stream n) = none)
    (hfr : (frameAt R sid i r).isSome) {j : Nat} {p q p₁ : Rep} (hf : R.find sid i r = some (j, p)) {pl' : Place}
    {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    (hR : RepRun pre ev q) : ∃ p₃, frameAt R₃ sid i r = some p₃ ∧ p₃.cfg = p.cfg := by
  obtain ⟨st₂, -, hat, -⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hc hR
  exact ⟨_, hat, (roundStep_cfg hrp hput).trans (roundPlace_cfg hf hrp)⟩

/-- **A round's count is its frame's schedule's** where it is the placeholder's schedule's, the two of one shape. -/
theorem count_roundsAt {rsched sched' : Array (ByteArray × Nat)} (hrs : Sched.RoundsAt rsched sched') {k n : Nat}
    (hk : k < rsched.size) (hn : n = (sched'.getD k (.empty, 0)).2) : n = (rsched.getD k (.empty, 0)).2 := by
  obtain ⟨h2, -⟩ := Loop.roundsAt_getD hrs hk
  exact hn.trans h2.symm

end Claims.ZKServer.Relay
