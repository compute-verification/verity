import Proofs.Properties.NetworkRecords.Bytes
import Properties.NetworkRecords

/-!
# Every frame of the Program's grid is one of the link's frames

The FIFO releases no more frames than were enqueued (`queue_add`), and no more are enqueued than the link has
(`sum_enqueued_le`), so each slot the Program fills holds the payload of a frame the link has (`expectedRow_frame`): of
`B` bytes when every payload is `8·B` bits (`programGrid_sizes`).
-/

open NetworkCertifier (ClockSync LogicalFrame expectedRow queue released enqueued handingFrom handingBuckets payloads)
open Properties.NetworkCertifier (programGrid bytesGrid)

namespace Properties.NetworkRecords.Program

theorem queue_add (S : Nat) (enq : Nat → Nat) : ∀ t,
    queue S enq t + ((List.range t).map fun u => (released S enq u : Nat)).sum = ((List.range t).map enq).sum
  | 0 => by simp [queue]
  | t + 1 => by
    have ih := queue_add S enq t
    simp only [released] at ih
    simp only [List.range_succ, List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil, queue, released]
    omega

theorem sum_ite (a : Nat) : ∀ n, ((List.range n).map fun u => if a = u then 1 else 0).sum = if a < n then 1 else 0
  | 0 => by simp
  | n + 1 => by
    simp only [List.range_succ, List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil, sum_ite a n]
    split_ifs <;> omega

theorem sum_count_le (n : Nat) : ∀ hs : List Nat, ((List.range n).map fun u => hs.count u).sum ≤ hs.length
  | [] => by simp
  | a :: hs => by
    have ih := sum_count_le n hs
    have e : (fun u => (a :: hs).count u) = fun u => hs.count u + if a = u then 1 else 0 := by
      funext u; rw [List.count_cons]; simp [beq_iff_eq]
    rw [e, List.sum_map_add, sum_ite, List.length_cons]
    split_ifs <;> omega

theorem sum_le {f g : Nat → Nat} (h : ∀ u, f u ≤ g u) :
    ∀ l : List Nat, (l.map f).sum ≤ (l.map g).sum
  | [] => by simp
  | u :: l => by
    simp only [List.map_cons, List.sum_cons]
    have := sum_le h l
    have := h u
    omega

theorem handingFrom_length (T W : Nat) (syncs : Fin W → ClockSync) :
    ∀ (last : Nat) (fs : List LogicalFrame), (handingFrom T W syncs last fs).length = fs.length
  | _, [] => rfl
  | _, _ :: fs => by simp only [handingFrom, List.length_cons, handingFrom_length T W syncs _ fs]

theorem sum_enqueued_le (T W : Nat) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) (n : Nat) :
    ((List.range n).map (enqueued T W syncs frames)).sum ≤ frames.length := by
  have hle : ((List.range n).map (enqueued T W syncs frames)).sum ≤
      ((List.range n).map fun u => (handingBuckets T W syncs frames).count u).sum := by
    apply sum_le
    intro u
    unfold enqueued
    split_ifs <;> omega
  have := sum_count_le n (handingBuckets T W syncs frames)
  rw [handingBuckets, handingFrom_length] at this
  unfold handingBuckets at hle
  omega

theorem offset_last {S t : Nat} (f : Nat → Fin (S + 1)) :
    NetworkCertifier.offset (fun u : Fin (t + 1) => f u.1) (Fin.last t) = ((List.range t).map fun u => (f u : Nat)).sum := by
  unfold NetworkCertifier.offset
  rw [List.finRange_succ_last, List.filter_append]
  have h1 : (List.map Fin.castSucc (List.finRange t)).filter (fun u => decide (u < Fin.last t)) =
      List.map Fin.castSucc (List.finRange t) :=
    List.filter_eq_self.mpr fun u hu => by
      obtain ⟨v, -, rfl⟩ := List.mem_map.mp hu
      simp [Fin.castSucc_lt_last v]
  have h2 : [Fin.last t].filter (fun u => decide (u < Fin.last t)) = [] := by simp
  rw [h1, h2, List.append_nil, List.map_map, ← List.map_coe_finRange_eq_range, List.map_map]
  rfl

/-- A slot the Program fills holds the payload of one of the link's frames. -/
theorem expectedRow_frame {S T W : Nat} {syncs : Fin W → ClockSync} {frames : List LogicalFrame} {t : Nat}
    {s : Fin S} {f : List Bool} (h : expectedRow S T W syncs frames t s = some f) :
    ∃ k < frames.length, f = payloads frames k := by
  unfold expectedRow NetworkCertifier.fifoGrid NetworkCertifier.fill at h
  split at h
  · rename_i hs
    refine ⟨_, ?_, (Option.some.inj h).symm⟩
    rw [offset_last (released S (enqueued T W syncs frames))]
    simp only [Fin.val_last] at hs ⊢
    have hq := queue_add S (enqueued T W syncs frames) (t + 1)
    have he := sum_enqueued_le T W syncs frames (t + 1)
    simp only [List.range_succ, List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil] at hq he
    omega
  · cases h

/-- Every frame of the Program's grid is `B` bytes when every payload of the link is `8·B` bits. -/
theorem programGrid_sizes {p : NetworkCheck.Check.Params} {syncs : Fin p.windows → ClockSync}
    {frames : List LogicalFrame} {w : Nat} (hf : ∀ f ∈ frames, f.payload.length = 8 * p.grid.frameBytes) :
    ∀ row ∈ (programGrid p syncs frames w).toList, ∀ f ∈ (row.filterMap id).toList, f.size = p.grid.frameBytes := by
  intro row hrow f hfr
  simp only [programGrid, bytesGrid, Array.toList_map, List.mem_map, List.mem_range] at hrow
  obtain ⟨i, -, rfl⟩ := hrow
  rw [Array.mem_toList_iff, Array.mem_filterMap] at hfr
  obtain ⟨o, ho, hof⟩ := hfr
  rw [Array.mem_map] at ho
  obtain ⟨o', ho', rfl⟩ := ho
  rw [List.mem_toArray, List.mem_ofFn] at ho'
  obtain ⟨s, hs⟩ := ho'
  cases o' with
  | none => cases hof
  | some x =>
    cases hof
    obtain ⟨k, hk, rfl⟩ := expectedRow_frame hs
    apply Bytes.bytesOfBits_size
    unfold payloads
    rw [List.getElem?_eq_getElem hk]
    exact hf _ (List.getElem_mem hk)

end Properties.NetworkRecords.Program
