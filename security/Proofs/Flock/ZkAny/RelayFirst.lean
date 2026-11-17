module

public import Proofs.Flock.ZkAny.RelayFold
public import Proofs.Flock.ZkAny.RelayFrames

/-! A stream's first round in the loop is the schedule's: its domain frame leaves where `leaveOf` says (`leave_repRun`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 SessionSpec)
open _root_.Flock.ZkServer.Frames (Rep)
open _root_.Flock.ZkServer.Frame (Cfg Req Place)

theorem prior_begun (c : Cfg) (st : _root_.Flock.ZkServer.Frame.State) (k : Nat) (b : ByteArray) :
    (_root_.Flock.ZkServer.Frame.step c st (.prior k b)).2.begun = st.begun := by
  simp only [_root_.Flock.ZkServer.Frame.step]
  split
  · rfl
  · split <;> rfl

theorem run_prior_begun (c : Cfg) : ∀ (pre : List Req) (st : _root_.Flock.ZkServer.Frame.State),
    (∀ q ∈ pre, ∃ k b, q = .prior k b) → (_root_.Flock.ZkServer.Frame.run c st pre).1.begun = st.begun
  | [], _, _ => rfl
  | q :: pre, st, h => by
    obtain ⟨k, b, rfl⟩ := h q List.mem_cons_self
    rw [run_cons]
    split
    · exact prior_begun c st k b
    · rw [run_prior_begun c pre _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
      exact prior_begun c st k b

theorem first_repRun {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} {p : Rep}
    (h : RepRun pre ev p) (hpre : ∀ q ∈ pre, ∃ k b, q = .prior k b) :
    (p.coins.isEmpty && !p.st.begun) = ev.isEmpty := by
  obtain ⟨h1, h2⟩ := h
  rw [← Array.isEmpty_toList, h2, List.isEmpty_map]
  cases ev with
  | nil =>
    have e0 : evReqs ([] : List (Nat × ByteArray × Place × Array F128)) = [] := rfl
    rw [e0, List.append_nil] at h1
    have e : (_root_.Flock.ZkServer.Frame.run p.cfg {} pre).1.begun = p.st.begun := by rw [h1]
    have hb : p.st.begun = false := by
      rw [← e, run_prior_begun _ pre _ hpre]
    simp [hb]
  | cons _ _ => simp

theorem leave_repRun {spec : SessionSpec} {seen : List String} {stream : String} {pre : List Req}
    {ev : List (Nat × ByteArray × Place × Array F128)} {p : Rep} {dom c : ByteArray} (h : RepRun pre ev p)
    (hpre : ∀ q ∈ pre, ∃ k b, q = .prior k b) (hev : ev.isEmpty = !seen.contains stream)
    (hd : dom = domOf spec stream) :
    (if (p.coins.isEmpty && !p.st.begun) then dom ++ c else c) = leaveOf spec seen stream c :=
  leaveOf_first ((first_repRun h hpre).trans hev) hd

end Claims.ZKServer.Relay
