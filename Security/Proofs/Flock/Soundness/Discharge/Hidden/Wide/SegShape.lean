import Flock.HmRow

/-!
# A segmented row's compressions in closed form

A `sha512/row-seg/v1` port's `segs` lists each segment's bits and compressions (`HmRow.segShape`). Segment `j`'s
compressions start at `firstOf segs j`, the compressions of the segments before it, and the outer message's at
`firstOf segs s`. The executable's loops over `segs` (`HmRow.chainStart`, `HmRow.msgBit`, and `HmRow.byteAt`'s fold) are
these prefix sums: `chainStart_some` and `chainStart_first` say a compression starts a chain exactly at a segment's or
the outer message's first, and `msgBit_seg` and `msgBit_outer` say what each compression's message bit is.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Hidden

open Flock

namespace SegShape

/-- Segment `j`'s first compression: the compressions of the segments before it. -/
def firstOf (segs : Array (Nat × Nat)) (j : Nat) : Nat := ((segs.toList.take j).map Prod.snd).sum

theorem foldl_snd (l : List (Nat × Nat)) (a : Nat) : l.foldl (· + ·.2) a = a + (l.map Prod.snd).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp only [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

theorem foldl_extract (segs : Array (Nat × Nat)) (j : Nat) :
    (segs.extract 0 j).foldl (· + ·.2) 0 = firstOf segs j := by
  rw [← Array.foldl_toList, Array.toList_extract, foldl_snd, Nat.zero_add, firstOf]
  simp [List.extract]

theorem firstOf_zero (segs : Array (Nat × Nat)) : firstOf segs 0 = 0 := by simp [firstOf]

theorem firstOf_succ {segs : Array (Nat × Nat)} {j : Nat} (hj : j < segs.size) :
    firstOf segs (j + 1) = firstOf segs j + segs[j]!.2 := by
  have hj' : j < segs.toList.length := by simpa using hj
  simp only [firstOf, List.take_add_one, List.map_append, List.sum_append, List.getElem?_eq_getElem hj',
    getElem!_pos segs j hj, Option.toList_some, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Array.getElem_toList, Nat.add_zero]

theorem firstOf_mono' {segs : Array (Nat × Nat)} {i j : Nat} (hij : i ≤ j) : firstOf segs i ≤ firstOf segs j := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hij
  simp only [firstOf, List.take_add, List.map_append, List.sum_append]
  omega

variable {segs : Array (Nat × Nat)} (hpos : ∀ j < segs.size, 0 < segs[j]!.2)
include hpos in
theorem firstOf_lt {i j : Nat} (hij : i < j) (hj : j ≤ segs.size) : firstOf segs i < firstOf segs j := by
  induction j with
  | zero => omega
  | succ j ih =>
    rw [firstOf_succ (by omega)]
    have := hpos j (by omega)
    rcases Nat.lt_succ_iff_lt_or_eq.1 hij with h | rfl
    · have := ih h (by omega); omega
    · omega

include hpos in
theorem firstOf_mono {i j : Nat} (hij : i ≤ j) (hj : j ≤ segs.size) : firstOf segs i ≤ firstOf segs j := by
  rcases Nat.lt_or_eq_of_le hij with h | rfl
  · exact Nat.le_of_lt (firstOf_lt hpos h hj)
  · exact Nat.le_refl _

/-! ### `chainStart` -/

/-- `HmRow.chainStart`'s loop body over a segmented row's `segs`. -/
def csBody (segs : Array (Nat × Nat)) (k : Nat) (j : Nat) (st : Option (Option Nat) × Nat) :
    Id (ForInStep (Option (Option Nat) × Nat)) :=
  if (k == st.2) = true then pure (ForInStep.done (some (some j), st.2))
  else if k < st.2 + segs[j]!.2 then pure (ForInStep.done (some none, st.2 + segs[j]!.2))
  else pure (ForInStep.yield (none, st.2 + segs[j]!.2))

section Loop

variable {k : Nat} (F : Nat → Option (Option Nat) × Nat → Id (ForInStep (Option (Option Nat) × Nat)))
  (hF : ∀ j st, F j st = csBody segs k j st)
include hF

theorem cs_loop_some {i m j f : Nat} (him : i + m ≤ segs.size) (hf : f = firstOf segs i)
    {r : Option (Option Nat) × Nat} (hr : Id.run (forIn (List.range' i m) (none, f) F) = r) :
    (r.1 = some (some j) → k = firstOf segs j ∧ i ≤ j ∧ j < i + m) ∧
      (r.1 = none → r.2 = firstOf segs (i + m)) := by
  subst hf
  induction m generalizing i r with
  | zero => subst hr; exact ⟨fun h => (by cases h), fun _ => rfl⟩
  | succ m ih =>
    rw [List.range'_succ, List.forIn_cons, hF] at hr
    simp only [csBody] at hr
    by_cases h1 : (k == firstOf segs i) = true
    · rw [if_pos h1] at hr; simp only [pure_bind] at hr; subst hr
      constructor
      · intro h; cases h; exact ⟨beq_iff_eq.1 h1, Nat.le_refl _, by omega⟩
      · intro h; cases h
    rw [if_neg h1] at hr
    by_cases h2 : k < firstOf segs i + segs[i]!.2
    · rw [if_pos h2] at hr; simp only [pure_bind] at hr; subst hr
      constructor <;> intro h <;> cases h
    rw [if_neg h2, ← firstOf_succ (by omega)] at hr
    simp only [pure_bind] at hr
    have := ih (i := i + 1) (r := r) (by omega) hr
    refine ⟨fun h => ?_, fun h => ?_⟩
    · obtain ⟨a, b, c⟩ := this.1 h; exact ⟨a, by omega, by omega⟩
    · rw [this.2 h]; congr 1; omega

include hpos in
theorem cs_loop_first {i m d f : Nat} (him : i + m ≤ segs.size) (hf : f = firstOf segs i) (hd : d < m)
    (hk : k = firstOf segs (i + d)) {r : Option (Option Nat) × Nat} (hr : Id.run (forIn (List.range' i m) (none, f) F) = r) :
    r.1 = some (some (i + d)) := by
  subst hf hr
  induction m generalizing i d with
  | zero => omega
  | succ m ih =>
    rw [List.range'_succ, List.forIn_cons, hF]
    simp only [csBody]
    rcases d with _ | d
    · rw [Nat.add_zero] at hk ⊢; rw [if_pos (beq_iff_eq.2 hk)]; rfl
    have hlt := firstOf_mono hpos (i := i + 1) (j := i + (d + 1)) (by omega) (by omega)
    rw [firstOf_succ (by omega)] at hlt
    have hne := hpos i (by omega)
    rw [if_neg (by rw [beq_iff_eq]; omega), if_neg (by omega), ← firstOf_succ (by omega)]
    simp only [pure_bind]
    have := ih (i := i + 1) (d := d) (by omega) (by omega) (by rw [hk]; congr 1; omega)
    rw [show i + 1 + d = i + (d + 1) by omega] at this
    exact this

include hpos in
theorem cs_loop_last {i m f : Nat} (him : i + m ≤ segs.size) (hf : f = firstOf segs i)
    (hk : k = firstOf segs (i + m)) {r : Option (Option Nat) × Nat} (hr : Id.run (forIn (List.range' i m) (none, f) F) = r) :
    r = (none, firstOf segs (i + m)) := by
  subst hf hr
  induction m generalizing i with
  | zero => rfl
  | succ m ih =>
    rw [List.range'_succ, List.forIn_cons, hF]
    simp only [csBody]
    have hlt := firstOf_mono hpos (i := i + 1) (j := i + (m + 1)) (by omega) (by omega)
    have hlt0 := hpos i (by omega)
    rw [firstOf_succ (by omega)] at hlt
    rw [if_neg (by rw [beq_iff_eq]; omega), if_neg (by omega), ← firstOf_succ (by omega)]
    simp only [pure_bind]
    have := ih (i := i + 1) (by omega) (by rw [hk]; congr 1; omega)
    rw [show i + 1 + m = i + (m + 1) by omega] at this
    exact this

end Loop

/-- A compression that starts a chain is a segment's first (`j < s`) or the outer message's (`j = s`). -/
theorem chainStart_some {p : Port} (hs : p.segs = some segs) {k j : Nat} (h : HmRow.chainStart p k = some j) :
    j ≤ segs.size ∧ k = firstOf segs j := by
  simp only [HmRow.chainStart, hs, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero,
    Nat.add_sub_cancel, Nat.div_one, Id.run_bind] at h
  generalize hR : Id.run (forIn (List.range' 0 segs.size 1) ((none : Option (Option Nat)), (0 : Nat)) _) = R at h
  have L := cs_loop_some (segs := segs) (k := k) _ (by intro _ _; rfl) (i := 0) (m := segs.size) (j := j) (by omega)
    (firstOf_zero segs).symm hR
  obtain ⟨r1, r2⟩ := R
  rcases r1 with _ | r1
  · simp only [Id.run_pure] at h
    split at h
    · rename_i hk; cases h; exact ⟨Nat.le_refl _, by rw [beq_iff_eq.1 hk]; simpa using L.2 rfl⟩
    · cases h
  · simp only [Id.run_pure] at h
    cases h
    obtain ⟨a, -, c⟩ := L.1 rfl
    exact ⟨by omega, a⟩

include hpos in
/-- Every segment's first compression, and the outer message's, starts a chain. -/
theorem chainStart_first {p : Port} (hs : p.segs = some segs) {j : Nat} (hj : j ≤ segs.size) :
    HmRow.chainStart p (firstOf segs j) = some j := by
  simp only [HmRow.chainStart, hs, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero,
    Nat.add_sub_cancel, Nat.div_one, Id.run_bind]
  generalize hR : Id.run (forIn (List.range' 0 segs.size 1) ((none : Option (Option Nat)), (0 : Nat)) _) = R
  rcases Nat.lt_or_eq_of_le hj with hj | rfl
  · have := cs_loop_first (segs := segs) (k := firstOf segs j) hpos _ (by intro _ _; rfl) (i := 0) (m := segs.size)
      (d := j) (by omega) (firstOf_zero segs).symm hj (by rw [Nat.zero_add]) hR
    rw [Nat.zero_add] at this
    obtain ⟨r1, r2⟩ := R
    cases this; rfl
  · have := cs_loop_last (segs := segs) (k := firstOf segs segs.size) hpos _ (by intro _ _; rfl) (i := 0)
      (m := segs.size) (by omega) (firstOf_zero segs).symm (by rw [Nat.zero_add]) hR
    rw [Nat.zero_add] at this
    subst this
    simp

/-- A compression of a segmented row starts no chain off its segments' and the outer message's firsts. -/
theorem chainStart_none {p : Port} (hs : p.segs = some segs) {k : Nat} (hk : ∀ j ≤ segs.size, k ≠ firstOf segs j) :
    HmRow.chainStart p k = none := by
  rcases h : HmRow.chainStart p k with _ | j
  · rfl
  · obtain ⟨a, b⟩ := chainStart_some hs h; exact absurd b (hk j a)

/-! ### `msgBit` -/

/-- Segment `j`'s message bit at compression `k` (its first `f`), block byte `o`, bit `b`: a row bit while the segment's
`m` bits last, then its fill and SHA-512's padding. -/
def segVal (p : Port) (j k f o b m : Nat) : HmRow.MsgBit :=
  if 8 * (128 * (k - f) + o) + b < m then .row (HmRow.ROW_SEG_BITS * j + 8 * (128 * (k - f) + o) + b)
  else .const (decide ((m + 7) / 8 ≤ 128 * (k - f) + o) &&
    ((HmRow.segTail p j).get! (128 * (k - f) + o - (m + 7) / 8)).toNat >>> b % 2 == 1)

/-- `HmRow.msgBit`'s loop body over a segmented row's `segs`. -/
def mbBody (p : Port) (segs : Array (Nat × Nat)) (k o b : Nat) (j : Nat) (st : Option HmRow.MsgBit × Nat) :
    Id (ForInStep (Option HmRow.MsgBit × Nat)) :=
  if k < st.2 + segs[j]!.2 then
    if 8 * (128 * (k - st.2) + o) + b < segs[j]!.1 then
      pure (ForInStep.done (some (.row (HmRow.ROW_SEG_BITS * j + 8 * (128 * (k - st.2) + o) + b)), st.2))
    else pure (ForInStep.done (some (.const (decide ((segs[j]!.1 + 7) / 8 ≤ 128 * (k - st.2) + o) &&
      ((HmRow.segTail p j).get! (128 * (k - st.2) + o - (segs[j]!.1 + 7) / 8)).toNat >>> b % 2 == 1)), st.2))
  else pure (ForInStep.yield (none, st.2 + segs[j]!.2))

section Loop

variable {p : Port} {k o b : Nat} (F : Nat → Option HmRow.MsgBit × Nat → Id (ForInStep (Option HmRow.MsgBit × Nat)))
  (hF : ∀ j st, F j st = mbBody p segs k o b j st)
include hF

theorem mb_step {j f : Nat} :
    F j (none, f) = if k < f + segs[j]!.2 then pure (ForInStep.done (some (segVal p j k f o b segs[j]!.1), f))
      else pure (ForInStep.yield (none, f + segs[j]!.2)) := by
  rw [hF]
  by_cases h1 : k < f + segs[j]!.2
  · by_cases h2 : 8 * (128 * (k - f) + o) + b < segs[j]!.1
    · simp only [mbBody, segVal, h1, h2, ↓reduceIte]
    · simp only [mbBody, segVal, h1, h2, ↓reduceIte]
  · simp only [mbBody, h1, ↓reduceIte]

theorem mb_loop_seg {i m j f : Nat} (him : i + m ≤ segs.size) (hf : f = firstOf segs i) (hij : i ≤ j)
    (hjm : j < i + m) (hk1 : firstOf segs j ≤ k) (hk2 : k < firstOf segs (j + 1)) {r : Option HmRow.MsgBit × Nat}
    (hr : Id.run (forIn (List.range' i m) (none, f) F) = r) :
    r.1 = some (segVal p j k (firstOf segs j) o b segs[j]!.1) := by
  subst hf hr
  induction m generalizing i with
  | zero => omega
  | succ m ih =>
    rw [List.range'_succ, List.forIn_cons, mb_step F hF]
    rcases Nat.lt_or_eq_of_le hij with h | rfl
    · have hle := firstOf_mono' (segs := segs) (i := i + 1) (j := j) (by omega)
      rw [firstOf_succ (by omega)] at hle
      rw [if_neg (by omega), ← firstOf_succ (by omega)]
      simp only [pure_bind]
      exact ih (i := i + 1) (by omega) (by omega) (by omega)
    · rw [firstOf_succ (by omega)] at hk2
      rw [if_pos hk2]; rfl

theorem mb_loop_past {i m f : Nat} (him : i + m ≤ segs.size) (hf : f = firstOf segs i)
    (hk : firstOf segs (i + m) ≤ k) {r : Option HmRow.MsgBit × Nat}
    (hr : Id.run (forIn (List.range' i m) (none, f) F) = r) : r = (none, firstOf segs (i + m)) := by
  subst hf hr
  induction m generalizing i with
  | zero => rfl
  | succ m ih =>
    rw [List.range'_succ, List.forIn_cons, mb_step F hF]
    have hle := firstOf_mono' (segs := segs) (i := i + 1) (j := i + (m + 1)) (by omega)
    rw [firstOf_succ (by omega)] at hle
    rw [if_neg (by omega), ← firstOf_succ (by omega)]
    simp only [pure_bind]
    have := ih (i := i + 1) (by omega) (by rw [show i + 1 + m = i + (m + 1) by omega]; exact hk)
    rw [show i + 1 + m = i + (m + 1) by omega] at this
    exact this

end Loop

/-- A compression in segment `j` (`firstOf j ≤ k < firstOf (j + 1)`) reads the segment's row bits, then its fill and
padding. -/
theorem msgBit_seg {p : Port} (hs : p.segs = some segs) {j k : Nat} (hj : j < segs.size)
    (hk1 : firstOf segs j ≤ k) (hk2 : k < firstOf segs (j + 1)) (t : Nat) :
    HmRow.msgBit p k t =
      segVal p j k (firstOf segs j) (HmRow.blockBit t).1 (HmRow.blockBit t).2 segs[j]!.1 := by
  rcases hbb : HmRow.blockBit t with ⟨o, b⟩
  simp only [HmRow.msgBit, hbb, hs, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero,
    Nat.add_sub_cancel, Nat.div_one, Id.run_bind]
  generalize hR : Id.run (forIn (List.range' 0 segs.size 1) ((none : Option HmRow.MsgBit), (0 : Nat)) _) = R
  have := mb_loop_seg (segs := segs) (p := p) (k := k) (o := o) (b := b) _ (by intro _ _; rfl) (i := 0)
    (m := segs.size) (j := j) (by omega) (firstOf_zero segs).symm (Nat.zero_le _) (by omega) hk1 hk2 hR
  obtain ⟨r1, r2⟩ := R
  cases this; rfl

/-- A compression from the outer message's first on reads the segments' digests, then the outer padding. -/
theorem msgBit_outer {p : Port} (hs : p.segs = some segs) {k : Nat} (hk : firstOf segs segs.size ≤ k) (t : Nat) :
    HmRow.msgBit p k t =
      if 128 * (k - firstOf segs segs.size) + (HmRow.blockBit t).1 < 64 * segs.size then
        .digest (firstOf segs ((128 * (k - firstOf segs segs.size) + (HmRow.blockBit t).1) / 64 + 1) - 1)
          (64 * ((128 * (k - firstOf segs segs.size) + (HmRow.blockBit t).1) % 64 / 8) + 56 -
            8 * ((128 * (k - firstOf segs segs.size) + (HmRow.blockBit t).1) % 64 % 8) + (HmRow.blockBit t).2)
      else .const ((((HmRow.tailOf p).get! (128 * (k - firstOf segs segs.size) + (HmRow.blockBit t).1 -
        64 * segs.size)).toNat >>> (HmRow.blockBit t).2) % 2 == 1) := by
  rcases hbb : HmRow.blockBit t with ⟨o, b⟩
  simp only [HmRow.msgBit, hbb, hs, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero,
    Nat.add_sub_cancel, Nat.div_one, Id.run_bind]
  generalize hR : Id.run (forIn (List.range' 0 segs.size 1) ((none : Option HmRow.MsgBit), (0 : Nat)) _) = R
  have := mb_loop_past (segs := segs) (p := p) (k := k) (o := o) (b := b) _ (by intro _ _; rfl) (i := 0)
    (m := segs.size) (by omega) (firstOf_zero segs).symm (by rw [Nat.zero_add]; exact hk) hR
  rw [Nat.zero_add] at this
  subst this
  simp only [foldl_extract]
  split <;> rfl

/-! ### `byteAt` -/

theorem byteAt_seg {p : Port} (hs : p.segs = some segs) (byte : Nat) :
    HmRow.byteAt p byte = (firstOf segs (byte / 1024) + byte % 1024 / 128, byte % 128) := by
  simp only [HmRow.byteAt, hs, foldl_extract]

/-- Every segment of `HmRow.segShape` has a compression after its first block. -/
theorem segShape_pos (l : Nat) : ∀ j < (HmRow.segShape l).1.size, 0 < (HmRow.segShape l).1[j]!.2 := by
  intro j hj
  rw [getElem!_pos _ j hj]
  simp only [HmRow.segShape, Array.getElem_map, Array.getElem_range]
  omega

end SegShape

end FlockSoundness.Discharge.Hidden
