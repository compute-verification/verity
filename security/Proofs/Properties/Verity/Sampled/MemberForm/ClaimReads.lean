module

public import Proofs.Properties.Verity.Sampled.MemberForm.OpensOwn
public import Proofs.Properties.Verity.Sampled.MemberForm.Consts
public import Proofs.Properties.Verity.Sampled.Window.Instance

/-! What the claim reads at the per-member form's windows. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Flock.Guarantees (MemberV5 Seat ReadsWith tracesWith JointAuditV5)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem traceLeft_beside {C D : Verity.Circuit} (tC : C.Trace) (tD : D.Trace) :
    Verity.Circuit.traceLeft (Verity.Circuit.traceBeside tC tD) = tC := by
  funext i
  unfold Verity.Circuit.traceLeft Verity.Circuit.traceBeside
  rw [Fin.addCases_left]
  exact BitVec.cast_cast ..

theorem traceRight_beside {C D : Verity.Circuit} (tC : C.Trace) (tD : D.Trace) :
    Verity.Circuit.traceRight (Verity.Circuit.traceBeside tC tD) = tD := by
  funext j
  unfold Verity.Circuit.traceRight Verity.Circuit.traceBeside
  rw [Fin.addCases_right]
  exact BitVec.cast_cast ..

theorem seat_trace_heq {Reg : Type} {nJ : ℕ} (s : Seat Reg nJ) {r : Verity.FileRead}
    (h : r.circuit = Flock.Bridge.circuitOf s.m.D s.m.hU s.m.n) : HEq (s.trace r) (s.m.transcript s.Xv) := by
  unfold Flock.Guarantees.Seat.trace
  rw [dite_eq_left h]
  exact cast_heq _ _

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

theorem readsWith_members {Reg : Type} {nJ : ℕ} (r : Verity.FileRead) (rs : List Verity.FileRead)
    (op : (g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop)
    (ss : List (Seat Reg nJ)) {ms : List (MemberV5 Reg nJ)} (hm : ss.map (·.m) = ms)
    (hf : ms.map (·.fileRead) = r :: rs) (hc : ∀ m ∈ ms, m.Checked) (ho : OpensWith r rs op ms)
    (hk : ConstsWith r rs op) : ReadsWith r rs op (tracesWith r rs ss) ss := by
  subst hm
  exact readsWith_of r rs op ss (List.map_map.symm.trans hf)
    (fun s hs => hc s.m (List.mem_map.2 ⟨s, hs, rfl⟩)) ho hk

theorem seats_members (w : JointAuditV5) [Nonempty w.EndΩ] (τ : FlockSoundness.Game.Strategy w.game) :
    (w.seats τ).map (·.m) = w.ms := by
  unfold Flock.Guarantees.JointAuditV5.seats
  rw [List.map_ofFn]
  exact List.ofFn_get w.ms

theorem claimReads (I : ExecutionPhaseInstance d α) (K : ExecutionPhaseConstructions I.base) : I.ClaimReads K where
  laid θ F W hW := (show MembersSetUp I.base I.pinned θ F W from (I.executionPhases_some hW).2).laid
  share θ F W hW := (show MembersSetUp I.base I.pinned θ F W from (I.executionPhases_some hW).2).toExecutionPhaseOn.share
  units θ F W hW := (show MembersSetUp I.base I.pinned θ F W from (I.executionPhases_some hW).2).toExecutionPhaseOn.units
  reads θ F W hW hfix c σ _ := by
    have h : MembersSetUp I.base I.pinned θ F W := (I.executionPhases_some hW).2
    exact readsWith_members (I.base.read θ F).first (I.base.read θ F).rest (I.base.Opens θ F c) _
      (seats_members _ _) (h.toExecutionPhaseOn.reads_toV5 _) (fun _ hM => h.checked hfix _ hM)
      (h.opensWith c (I.place θ F c) _) (constsWith_read I.base θ c F)

end Verity.Sampled.MemberForm
