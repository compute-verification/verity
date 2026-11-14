module

public import Proofs.Properties.Verity.Sampled.MemberForm.OpensOwn
public import Proofs.Properties.Verity.Sampled.MemberForm.Consts
public import Proofs.Properties.Verity.Sampled.Window.Instance

/-!
# What the claim reads at the per-member form's windows

The claim's transfer at a window reads the committed files member by member (`WindowInstance.ClaimReads`,
`Flock.Guarantees.ReadsWith`): each member's files pass the verifier's files check, the vocabulary's read is the
member's, the vocabulary's openings hold wherever the member's deployed commitment opens and at its files'
constants, and the extracted transcript is, at the member's gates, the member's transcript. This module proves it
of the windows the verifier sets up at the per-member form (`WindowInstance.windows`, `MemberForm.MembersSetUp`):

* **The seats' traces side by side are each seat's own** (`traceLeft_beside`, `traceRight_beside`,
  `seat_trace_heq`): the first's at the first member's gates and the rest's at theirs, and a seat's trace at its
  member's files is the member's transcript, where those files' circuit is the member's parsed circuit file's.
* **Member by member** (`readsWith_of`, `readsWith_members`): seats whose members' files are the ones side by side
  in order, each `Checked`, are read at the claim at their own traces side by side (`tracesWith`), under openings
  that hold wherever each member's deployed commitment opens (`OpensWith`) and at each member's constants
  (`ConstsWith`).
* **A deployment's windows** (`claimReads`): at each window the verifier sets up, the window is laid out, the
  published share and the units are the window's, and the committed files are read at the claim at every strategy.

The verifier parses parameters in either form (`Spot.OneStageZk.paramsEitherOfJson`), so it sets up a window of
members where they name members, and `claimReads` speaks of those windows. Its proof reads none of that.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Flock.Guarantees (MemberV5 Seat ReadsWith tracesWith JointAuditV5)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **Two traces side by side, at the first circuit's gates, are the first.** -/
theorem traceLeft_beside {C D : Verity.Circuit} (tC : C.Trace) (tD : D.Trace) :
    Verity.Circuit.traceLeft (Verity.Circuit.traceBeside tC tD) = tC := by
  funext i
  unfold Verity.Circuit.traceLeft Verity.Circuit.traceBeside
  rw [Fin.addCases_left]
  exact BitVec.cast_cast ..

/-- **Two traces side by side, at the second circuit's gates, are the second.** -/
theorem traceRight_beside {C D : Verity.Circuit} (tC : C.Trace) (tD : D.Trace) :
    Verity.Circuit.traceRight (Verity.Circuit.traceBeside tC tD) = tD := by
  funext j
  unfold Verity.Circuit.traceRight Verity.Circuit.traceBeside
  rw [Fin.addCases_right]
  exact BitVec.cast_cast ..

/-- **A seat's trace at files whose circuit is its member's parsed circuit file's is the member's transcript** at
the seat's trace. -/
theorem seat_trace_heq {Reg : Type} {nJ : ℕ} (s : Seat Reg nJ) {r : Verity.FileRead}
    (h : r.circuit = Flock.Bridge.circuitOf s.m.D s.m.hU s.m.n) : HEq (s.trace r) (s.m.transcript s.Xv) := by
  unfold Flock.Guarantees.Seat.trace
  rw [dite_eq_left h]
  exact cast_heq _ _

/-- **Seats whose members' files are the ones side by side are read at the claim at their own traces**: a first
member's files and the rest's, and seats whose members' files are those in order, each member `Checked`; under
openings that hold wherever each member's deployed commitment opens and at each member's constants, the files are
read at the claim by the seats, at the seats' traces side by side. -/
theorem readsWith_of {Reg : Type} {nJ : ℕ} : ∀ (r : Verity.FileRead) (rs : List Verity.FileRead)
    (op : (g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop)
    (ss : List (Seat Reg nJ)), ss.map (fun s => s.m.fileRead) = r :: rs → (∀ s ∈ ss, s.m.Checked) →
      OpensWith r rs op (ss.map (·.m)) → ConstsWith r rs op → ReadsWith r rs op (tracesWith r rs ss) ss
  | r, [], op, ss, hf, hc, ho, hk => by
    cases ss with
    | nil => simp at hf
    | cons s ss' =>
      cases ss' with
      | cons s' ss'' => simp at hf
      | nil =>
        obtain rfl : s.m.fileRead = r := List.head_eq_of_cons_eq hf
        obtain ⟨m, hm, hop⟩ := ho
        obtain rfl : s.m = m := List.head_eq_of_cons_eq hm
        have hcs := hc s (List.mem_singleton.2 rfl)
        exact ⟨s, rfl, hcs, rfl, hop, hk, seat_trace_heq s (checked_circuit s.m hcs)⟩
  | r, r' :: rs, op, ss, hf, hc, ho, hk => by
    cases ss with
    | nil => simp at hf
    | cons s ss' =>
      obtain rfl : s.m.fileRead = r := List.head_eq_of_cons_eq hf
      obtain ⟨m, ms', hm, hop, hrest⟩ := ho
      obtain rfl : s.m = m := List.head_eq_of_cons_eq hm
      obtain rfl : ss'.map (·.m) = ms' := List.tail_eq_of_cons_eq hm
      have hcs := hc s List.mem_cons_self
      refine ⟨s, ss', rfl, ⟨hcs, rfl, hop, hk.1, ?_⟩, ?_⟩
      · show HEq (Verity.Circuit.traceLeft
            (Verity.Circuit.traceBeside (s.trace s.m.fileRead) (tracesWith r' rs ss')))
          (s.m.transcript s.Xv)
        rw [traceLeft_beside]
        exact seat_trace_heq s (checked_circuit s.m hcs)
      · show ReadsWith r' rs _
          (Verity.Circuit.traceRight
            (Verity.Circuit.traceBeside (s.trace s.m.fileRead) (tracesWith r' rs ss'))) ss'
        rw [traceRight_beside]
        exact readsWith_of r' rs _ ss' (List.tail_eq_of_cons_eq hf)
          (fun s' h' => hc s' (List.mem_cons_of_mem _ h')) hrest hk.2

/-- **Seats of the members `ms` are read at the claim at their own traces** (`readsWith_of`), at members whose
files are the ones side by side in order, each `Checked`. -/
theorem readsWith_members {Reg : Type} {nJ : ℕ} (r : Verity.FileRead) (rs : List Verity.FileRead)
    (op : (g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop)
    (ss : List (Seat Reg nJ)) {ms : List (MemberV5 Reg nJ)} (hm : ss.map (·.m) = ms)
    (hf : ms.map (·.fileRead) = r :: rs) (hc : ∀ m ∈ ms, m.Checked) (ho : OpensWith r rs op ms)
    (hk : ConstsWith r rs op) : ReadsWith r rs op (tracesWith r rs ss) ss := by
  subst hm
  exact readsWith_of r rs op ss (List.map_map.symm.trans hf)
    (fun s hs => hc s.m (List.mem_map.2 ⟨s, hs, rfl⟩)) ho hk

/-- **A window's seats' members are the window's**, in order. -/
theorem seats_members (w : JointAuditV5) [Nonempty w.EndΩ] (τ : FlockSoundness.Game.Strategy w.game) :
    (w.seats τ).map (·.m) = w.ms := by
  unfold Flock.Guarantees.JointAuditV5.seats
  rw [List.map_ofFn]
  exact List.ofFn_get w.ms

/-- **What the claim reads at the deployment's windows, at the per-member form** (`WindowInstance.ClaimReads`):
each window the verifier sets up is laid out, with the published share and the committed files' units its own
(`MembersSetUp`); and at every strategy the committed files are read at the claim at the projected strategy, by
the window's members in order (`WindowOn.reads_toV5`), each `Checked` (`MembersSetUp.checked`), under the
vocabulary's openings, which hold wherever each member's deployed commitment opens (`MembersSetUp.opensWith`, at
the instance's places) and at each member's constants (`constsWith_read`), at the seats' traces side by side. -/
theorem claimReads (I : WindowInstance d α) (K : WindowConstructions I.base) : I.ClaimReads K where
  laid θ F W hW := (show MembersSetUp I.base θ F W from (I.windows_some hW).2).laid
  share θ F W hW := (show MembersSetUp I.base θ F W from (I.windows_some hW).2).toWindowOn.share
  units θ F W hW := (show MembersSetUp I.base θ F W from (I.windows_some hW).2).toWindowOn.units
  reads θ F W hW hfix c σ _ := by
    have h : MembersSetUp I.base θ F W := (I.windows_some hW).2
    exact readsWith_members (I.base.read θ F).first (I.base.read θ F).rest (I.base.Opens θ F c) _
      (seats_members _ _) (h.toWindowOn.reads_toV5 _) (fun _ hM => h.checked hfix _ hM)
      (h.opensWith c (I.place θ F c) _) (constsWith_read I.base θ c F)

end Verity.Sampled.MemberForm
