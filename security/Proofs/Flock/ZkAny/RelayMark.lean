module

public import Proofs.Flock.ZkAny.RelaySlot

/-!
# The loop's mark of a stream's first round is the schedule's

`Frames.roundPlace` marks a round as its stream's first (`first`: the loop sends its release after the stream's domain
frame) where the frame it found has no coins and took no round yet. The relay's encoding sends the domain frame before
the stream's first round in the schedule (`leaveOf`, the streams whose rounds came before being `seen`):

* `roundPlace_first`: the mark `roundPlace` returns is the found frame's, read before `withAlg` and rep 1's prior step.
* **`roundPlace_mark`**: so on the found frame's run (`RepRun`), where it took no round exactly where no round of its
  stream is in `seen`, the mark is the schedule's (`envOk_round_next`'s `hf`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Req Place)

set_option maxHeartbeats 1000000 in
/-- **The mark `roundPlace` returns is the found frame's**: no coins and no round taken yet, read before `withAlg` and
rep 0's algebraic-phase coins at rep 1's first round change the frame. -/
theorem roundPlace_first {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    f = (p.coins.isEmpty && !p.st.begun) := by
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  by_cases hfst : (p.coins.isEmpty && !p.st.begun) = true
  all_goals repeat' (first
    | split at h
    | simp only [eq_true hfst, ↓reduceIte] at h
    | simp only [eq_false hfst, ↓reduceIte] at h
    | simp only at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h
       first
         | rfl
         | exact hfst.symm
         | (simp only [Bool.not_eq_true] at hfst
            exact hfst.symm))

/-- **The loop's mark of a stream's first round is the schedule's**: on the found frame's run, its requests before its
rounds rep 0's algebraic-phase coins alone, where it took no round exactly where no round of its stream is in
`seen`. -/
theorem roundPlace_mark {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} (hR : RepRun pre ev p)
    (hpre : ∀ x ∈ pre, ∃ k b, x = .prior k b) {seen : List String} {stream : String}
    (hev : ev.isEmpty = !seen.contains stream) : f = !seen.contains stream :=
  (roundPlace_first hf h).trans ((first_repRun hR hpre).trans hev)

end Claims.ZKServer.Relay
