import NetworkCheck.Records

/-!
# `Records.expectedGrids` is `Check.expectedGrid`, window by window

`verity/network-records/v2` reads the Program's grids from `expectedGrids`, which computes every window of a link in one
pass, linear in the frames and the buckets, where `Check.expectedGrid` (and `NetworkCertifier.expectedRow`, which
`Security` reads) recomputes the hand times, the queue and the offsets for every bucket. `expectedGrids_eq`: for each
window `w < W`, the two are equal, so every egress verdict of the records check reads the Program `Security` reasons about.
-/

namespace NetworkCheck.RecordsGrid

open NetworkCheck.Records
open NetworkCertifier (ClockSync LogicalFrame)

theorem handingRev_eq (T W : Nat) (syncs : Fin W → ClockSync) (last : Nat) (fs : List LogicalFrame) (acc : List Nat) :
    handingRev T W syncs last fs acc = (NetworkCertifier.handingFrom T W syncs last fs).reverse ++ acc := by
  induction fs generalizing last acc with
  | nil => simp [handingRev, NetworkCertifier.handingFrom]
  | cons f fs ih => simp [handingRev, NetworkCertifier.handingFrom, ih]

theorem foldl_modify_getD (hs : List Nat) (a : Array Nat) (t : Nat) :
    (hs.foldl (fun a h => a.modify h (· + 1)) a).getD t 0 = a.getD t 0 + if t < a.size then hs.count t else 0 := by
  induction hs generalizing a with
  | nil => simp
  | cons h hs ih =>
    simp only [List.foldl_cons, ih, Array.size_modify, List.count_cons]
    by_cases ht : t < a.size
    · by_cases e : h = t
      · subst e; simp [Array.getD, ht, Array.getElem_modify]; omega
      · simp [Array.getD, ht, Array.getElem_modify, e]
    · simp [Array.getD, ht]

theorem enqueueCounts_eq (T W : Nat) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) :
    (fun t => (enqueueCounts (W * T) (handingRev T W syncs 0 frames []).reverse).getD t 0) =
      NetworkCertifier.enqueued T W syncs frames := by
  funext t
  simp only [enqueueCounts, foldl_modify_getD, handingRev_eq, List.append_nil, List.reverse_reverse,
    NetworkCertifier.enqueued, NetworkCertifier.handingBuckets]
  by_cases ht : t < W * T <;> simp [Array.getD, ht]

theorem releases_eq (S : Nat) (enq : Nat → Nat) (t c : Nat) :
    releases S enq t (NetworkCertifier.queue S enq t) c =
      (List.range' t c).map fun u => (NetworkCertifier.released S enq u : Nat) := by
  induction c generalizing t with
  | zero => simp [releases]
  | succ c ih =>
    rw [List.range'_succ, List.map_cons, releases, ← ih (t + 1)]
    rfl

theorem rows_length (S : Nat) (pay : Nat → ByteArray) (k : Nat) (ns : List Nat) : (rows S pay k ns).length = ns.length := by
  induction ns generalizing k with
  | nil => rfl
  | cons n ns ih => simp [rows, ih]

theorem rows_getElem (S : Nat) (pay : Nat → ByteArray) (k : Nat) (ns : List Nat) (i : Nat) (h : i < ns.length) :
    (rows S pay k ns)[i]'(by rw [rows_length]; exact h) =
      Array.ofFn fun s : Fin S => if s.1 < ns[i] then some (pay (k + (ns.take i).sum + s.1)) else none := by
  induction ns generalizing k i with
  | nil => simp at h
  | cons n ns ih =>
    cases i with
    | zero => simp [rows]
    | succ i =>
      simp only [rows, List.getElem_cons_succ, List.take_succ_cons, List.sum_cons]
      rw [ih (k + n) i (by simp at h; omega)]
      simp [Nat.add_assoc]

theorem map_finRange (t : Nat) (f : Nat → Nat) : ((List.finRange t).map fun i => f i.val) = (List.range t).map f := by
  induction t with
  | zero => simp
  | succ t ih => rw [List.finRange_succ_last, List.range_succ]; simp [← ih]

theorem offset_last (S : Nat) (enq : Nat → Nat) (t : Nat) :
    NetworkCertifier.offset (fun u : Fin (t + 1) => NetworkCertifier.released S enq u) (Fin.last t) =
      ((List.range t).map fun u => (NetworkCertifier.released S enq u : Nat)).sum := by
  unfold NetworkCertifier.offset
  rw [List.finRange_succ_last, List.filter_append]
  have : (List.map Fin.castSucc (List.finRange t)).filter (fun u => decide (u < Fin.last t)) =
      List.map Fin.castSucc (List.finRange t) := by
    rw [List.filter_eq_self]; intro u hu; simp at hu; obtain ⟨a, _, rfl⟩ := hu; simp [Fin.lt_def]
  rw [this]
  simp [← map_finRange]
  rfl

theorem releases_length (S : Nat) (enq : Nat → Nat) (t q c : Nat) : (releases S enq t q c).length = c := by
  induction c generalizing t q with
  | zero => rfl
  | succ c ih => simp [releases, ih]

theorem payloads_getD (frames : List LogicalFrame) (k : Nat) :
    (frames.toArray.map fun f => Check.bytesOfBits f.payload).getD k (Check.bytesOfBits []) =
      Check.bytesOfBits (NetworkCertifier.payloads frames k) := by
  unfold NetworkCertifier.payloads
  by_cases h : k < frames.length <;> simp [Array.getD, h]

theorem expectedRow_eq (S T W : Nat) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) (t : Nat) :
    (List.ofFn (NetworkCertifier.expectedRow S T W syncs frames t)).toArray.map (·.map Check.bytesOfBits) =
      Array.ofFn fun s : Fin S =>
        if s.1 < (NetworkCertifier.released S (NetworkCertifier.enqueued T W syncs frames) t : Nat) then
          some (Check.bytesOfBits (NetworkCertifier.payloads frames (((List.range t).map fun u =>
            (NetworkCertifier.released S (NetworkCertifier.enqueued T W syncs frames) u : Nat)).sum + s.1)))
        else none := by
  rw [List.toArray_ofFn, Array.map_ofFn]
  congr 1
  funext s
  simp only [Function.comp, NetworkCertifier.expectedRow, NetworkCertifier.fifoGrid, NetworkCertifier.fill, offset_last,
    Fin.val_last]
  split <;> rfl

/-- Window `w`'s grid of `expectedGrids` is `Check.expectedGrid`'s, slot for slot. -/
theorem expectedGrids_eq (g : Check.Grid) (W : Nat) (l : Check.Link) (w : Fin W) :
    (expectedGrids g W l)[w.val]? = some (Check.expectedGrid g W l w.val) := by
  obtain ⟨w, hw⟩ := w
  dsimp only
  simp only [expectedGrids, enqueueCounts_eq, Array.getElem?_map, Array.getElem?_range, hw, ite_true, Option.map_some]
  congr 1
  simp only [payloads_getD]
  have hrel := releases_eq g.allocation
    (NetworkCertifier.enqueued g.buckets W (fun v => l.syncs.getD (↑v) Check.noSync) l.frames) 0 (W * g.buckets)
  simp only [NetworkCertifier.queue] at hrel
  rw [hrel]
  have hle : w * g.buckets + g.buckets ≤ W * g.buckets := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hw
  apply Array.ext
  · simp [rows_length, Check.expectedGrid]; omega
  · intro i h1 h2
    have hi : i < g.buckets := by simpa [Check.expectedGrid] using h2
    simp only [Array.getElem_extract, List.getElem_toArray]
    rw [rows_getElem _ _ _ _ _ (by simp; omega)]
    simp only [Check.expectedGrid, Array.getElem_map, List.getElem_toArray, List.getElem_range]
    rw [expectedRow_eq]
    congr 1
    funext s
    have hm : min (w * g.buckets + i) (W * g.buckets) = w * g.buckets + i := Nat.min_eq_left (by omega)
    simp [List.getElem_map, ← List.range_eq_range', ← List.map_take, List.take_range, hm]

end NetworkCheck.RecordsGrid
