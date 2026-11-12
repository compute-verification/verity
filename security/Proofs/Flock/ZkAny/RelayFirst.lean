module

public import Proofs.Flock.ZkAny.RelayFold
public import Proofs.Flock.ZkAny.RelayFrames

/-!
# A stream's first round in the loop is the schedule's

The loop sends a stream's domain frame before the release of the stream's first round (`Frames.roundStep`), where the
rep's frame has no coins and took no round yet (`roundPlace`'s `first`). The relay's encoding sends it before the
stream's first round in the schedule (`RelayFrames.leaveOf`). On the rep's run (`RepRun`) the two agree:

* `prior_begun`, `run_prior_begun`: rep 0's algebraic-phase coins (`prior`) leave a frame's `begun` as it was, so a
  frame run on them alone took no round.
* `first_repRun`: on its run, the loop's mark of a stream's first round holds exactly where the rep took no round.
* `leave_repRun`: so where the rep took no round exactly where no earlier round of the schedule is its stream's, and
  the loop's domain frame is the setup's, the round leaves as `leaveOf` says.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 SessionSpec)
open _root_.Flock.Firewall.Frames (Rep)
open _root_.Flock.Firewall.Frame (Cfg Req Place)

/-- **Rep 0's algebraic-phase coins keep a frame's `begun`**, whether it takes them or refuses them. -/
theorem prior_begun (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (k : Nat) (b : ByteArray) :
    (_root_.Flock.Firewall.Frame.step c st (.prior k b)).2.begun = st.begun := by
  simp only [_root_.Flock.Firewall.Frame.step]
  split
  · rfl
  · split <;> rfl

/-- **A frame run on rep 0's algebraic-phase coins alone took no round it had not taken at its start.** -/
theorem run_prior_begun (c : Cfg) : ∀ (pre : List Req) (st : _root_.Flock.Firewall.Frame.State),
    (∀ q ∈ pre, ∃ k b, q = .prior k b) → (_root_.Flock.Firewall.Frame.run c st pre).1.begun = st.begun
  | [], _, _ => rfl
  | q :: pre, st, h => by
    obtain ⟨k, b, rfl⟩ := h q List.mem_cons_self
    rw [run_cons]
    split
    · exact prior_begun c st k b
    · rw [run_prior_begun c pre _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
      exact prior_begun c st k b

/-- **On its run, a rep's frame marks its stream's first round** (no coins, no round yet: `roundPlace`'s `first`)
exactly where the rep took no round, where its requests before its rounds are rep 0's algebraic-phase coins alone. -/
theorem first_repRun {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} {p : Rep}
    (h : RepRun pre ev p) (hpre : ∀ q ∈ pre, ∃ k b, q = .prior k b) :
    (p.coins.isEmpty && !p.st.begun) = ev.isEmpty := by
  obtain ⟨h1, h2⟩ := h
  rw [← Array.isEmpty_toList, h2, List.isEmpty_map]
  cases ev with
  | nil =>
    have e0 : evReqs ([] : List (Nat × ByteArray × Place × Array F128)) = [] := rfl
    rw [e0, List.append_nil] at h1
    have e : (_root_.Flock.Firewall.Frame.run p.cfg {} pre).1.begun = p.st.begun := by rw [h1]
    have hb : p.st.begun = false := by
      rw [← e, run_prior_begun _ pre _ hpre]
    simp [hb]
  | cons _ _ => simp

/-- **A round leaves as `leaveOf` says** on the rep's run, where the rep took no round exactly where no earlier round
of the schedule is its stream's (`seen`) and the loop's domain frame is the setup's. -/
theorem leave_repRun {spec : SessionSpec} {seen : List String} {stream : String} {pre : List Req}
    {ev : List (Nat × ByteArray × Place × Array F128)} {p : Rep} {dom c : ByteArray} (h : RepRun pre ev p)
    (hpre : ∀ q ∈ pre, ∃ k b, q = .prior k b) (hev : ev.isEmpty = !seen.contains stream)
    (hd : dom = domOf spec stream) :
    (if (p.coins.isEmpty && !p.st.begun) then dom ++ c else c) = leaveOf spec seen stream c :=
  leaveOf_first ((first_repRun h hpre).trans hev) hd

end Claims.ZKServer.Relay
