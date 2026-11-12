module

public import Definitions.NetworkCertifier.Warden
public import Aesop.BuiltinRules
public import Mathlib.Tactic.Attr.Core
public import Mathlib.Tactic.Basic
public import Mathlib.Tactic.ToAdditive
public import Mathlib.Tactic.ToDual
import Mathlib.Data.Finset.Attr
import Mathlib.Data.Nat.Basic
import Mathlib.Order.Basic
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike

/-!
# The warden's code is the client view

The warden's code on a lane emits, cell by cell, what the client view gives for the lane's first late cell
(`certifierLane_eq`): its data cells and padding before it, the fail cell, padding after. So the code's output is the
client view of the schedule, the data and the fail cells (`certifierCode_eq`), which `Tap` reads the network certifier's
cells from.

On the wire, a bucket's frames are the connection's frames in it (`filter_laneWire`), every cell's bucket in the
connection's span (`bucket_mem_span`). A lane of `k` cells has cell `j` in bucket `t j` (`certifierLane_fst`); among the
cells before `j`, those in `j`'s bucket number `j mod r` (`countP_range_div`), so `j` is the `(j mod r)`-th of its
bucket's frames (`getElem?_filter_countP_take`, `wireFrames_cell`), the admission padding in earlier buckets and the
padding after the lane's last cell after it.
-/

@[expose] public section

namespace NetTiming

/-- A cell before the fail cell is the stream's. -/
theorem cellView_of_lt {Cell : Type} (stream : List Cell) {o : Option ℕ} {j : ℕ} (h : ∀ f, o = some f → j < f) :
    cellView stream o j = streamCell stream j := by
  cases o with
  | none => rfl
  | some f => simp [cellView, h f rfl]

/-- **The network certifier's lane is the client view of its first late cell.** -/
theorem certifierLane_eq {Cell : Type} (a : Arrivals Cell) (t : ℕ → ℕ) (j k : ℕ) :
    certifierLane a t j k = (List.range' j k).map fun i =>
      (t i, cellView a.stream ((List.range' j k).find? fun i => decide (i ≤ a.stream.length ∧ t i < a.ready i)) i) := by
  induction k generalizing j with
  | zero => simp [certifierLane]
  | succ k ih =>
    have hafter : ∀ f, (List.range' (j + 1) k).find? (fun i => decide (i ≤ a.stream.length ∧ t i < a.ready i)) =
        some f → j < f := by
      intro f hf
      have := List.mem_of_find?_eq_some hf
      rw [List.mem_range'_1] at this
      omega
    rw [show List.range' j (k + 1) = j :: List.range' (j + 1) k from rfl, List.map_cons]
    by_cases hend : a.stream.length < j
    · have hp : ¬ (j ≤ a.stream.length ∧ t j < a.ready j) := fun h => by omega
      rw [List.find?_cons_of_neg (by simpa using hp)]
      simp only [certifierLane, hend, ↓reduceIte, ih]
      congr 1
      rw [cellView_of_lt _ hafter]
      simp [streamCell, show ¬ j < a.stream.length by omega]
    · by_cases hlate : t j < a.ready j
      · have hp : j ≤ a.stream.length ∧ t j < a.ready j := ⟨by omega, hlate⟩
        rw [List.find?_cons_of_pos (by simpa using hp)]
        simp only [certifierLane, hend, hlate, ↓reduceIte]
        refine List.cons_eq_cons.mpr ⟨by simp [cellView], ?_⟩
        apply List.map_congr_left
        intro i hi
        rw [List.mem_range'_1] at hi
        simp [cellView, show ¬ i < j by omega, show i ≠ j by omega]
      · have hp : ¬ (j ≤ a.stream.length ∧ t j < a.ready j) := fun h => hlate h.2
        rw [List.find?_cons_of_neg (by simpa using hp)]
        simp only [certifierLane, hend, hlate, ↓reduceIte, ih]
        rw [cellView_of_lt _ hafter]

theorem certifierCode_eq {κ Cell : Type} (s : CertifierSchedule κ) (arr : κ → Arrivals Cell) :
    certifierCode s arr = clientView s (fun c => (arr c).stream) (failCell s arr) := by
  funext c
  simp only [certifierCode, clientView, failCell, certifierLane_eq, List.range_eq_range']

section Wire

variable {κ Cell : Type}

theorem certifierLane_length (a : Arrivals Cell) (t : ℕ → ℕ) : ∀ j k, (certifierLane a t j k).length = k
  | _, 0 => rfl
  | j, k + 1 => by
    simp only [certifierLane]
    split_ifs <;> simp [certifierLane_length a t (j + 1) k]

/-- A lane's cell `i` is in bucket `t i`. -/
theorem certifierLane_fst (a : Arrivals Cell) (t : ℕ → ℕ) :
    ∀ j k, (certifierLane a t j k).map Prod.fst = (List.range' j k).map t
  | _, 0 => rfl
  | j, k + 1 => by
    simp only [certifierLane]
    split_ifs <;> simp [certifierLane_fst a t (j + 1) k, List.range'_succ, Function.comp_def]

theorem certifierCode_fst (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (c : κ) :
    (certifierCode s a c).map Prod.fst = (List.range (s.cells c)).map (s.bucket c) := by
  rw [List.range_eq_range']
  exact certifierLane_fst _ _ 0 _

/-- An element `j` that `p` keeps is the `n`-th that `p` keeps, `n` the number it keeps before `j`. -/
theorem getElem?_filter_countP_take {α : Type} (p : α → Bool) :
    ∀ (L : List α) (j : ℕ) (hj : j < L.length), p L[j] = true → (L.filter p)[(L.take j).countP p]? = some L[j]
  | x :: xs, 0, _, hp => by simp_all
  | x :: xs, j + 1, hj, hp => by
    have ih := getElem?_filter_countP_take p xs j (by simpa using hj) (by simpa using hp)
    by_cases hx : p x = true
    · simpa [List.filter_cons, hx, List.countP_cons] using ih
    · simpa [List.filter_cons, hx, List.countP_cons] using ih

/-- Of the `i < j`, those with `⌊i/r⌋ = ⌊j/r⌋` number `j mod r`. -/
theorem countP_range_div (r j : ℕ) : (List.range j).countP (fun i => i / r == j / r) = j % r := by
  rcases Nat.eq_zero_or_pos r with rfl | hr
  · simp
  · obtain ⟨q, m, hm, rfl⟩ : ∃ q m, m < r ∧ j = r * q + m := ⟨j / r, j % r, Nat.mod_lt _ hr, (Nat.div_add_mod j r).symm⟩
    have hq : (r * q + m) / r = q := by rw [Nat.mul_add_div hr, Nat.div_eq_of_lt hm, Nat.add_zero]
    rw [hq, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hm, List.range_add, List.countP_append, List.countP_map]
    have h1 : (List.range (r * q)).countP (fun i => i / r == q) = 0 := by
      refine List.countP_eq_zero.2 fun i hi => ?_
      have : i / r < q := (Nat.div_lt_iff_lt_mul hr).2 (by rw [Nat.mul_comm]; exact List.mem_range.1 hi)
      simp [Nat.ne_of_lt this]
    have h2 : (List.range m).countP ((fun i => i / r == q) ∘ (r * q + ·)) = m := by
      rw [List.countP_eq_length.2 fun i hi => ?_, List.length_range]
      simp [Nat.mul_add_div hr, Nat.div_eq_of_lt ((List.mem_range.1 hi).trans hm)]
    rw [h1, h2, Nat.zero_add]

theorem filterMap_flatMap_ite [DecidableEq κ] {β : Type} (X : List β) {c : κ} :
    ∀ {conns : List κ}, conns.Nodup → c ∈ conns →
      (conns.flatMap fun d => if d = c then X else []) = X
  | d :: ds, hnd, hc => by
    obtain ⟨hd, hds⟩ := List.nodup_cons.1 hnd
    by_cases hdc : d = c
    · subst hdc
      have : (ds.flatMap fun d' => if d' = d then X else []) = [] :=
        List.flatMap_eq_nil_iff.2 fun d' hd' => by simp [show d' ≠ d from fun h => hd (h ▸ hd')]
      simp [this]
    · have hc' : c ∈ ds := (List.mem_cons.1 hc).resolve_left (Ne.symm hdc)
      simp [hdc, filterMap_flatMap_ite X hds hc']

/-- A lane's cell is in its connection's span. -/
theorem bucket_mem_span (s : CertifierSchedule κ) (c : κ) {j : ℕ} (hj : j < s.cells c) : s.bucket c j ∈ s.span c := by
  rw [CertifierSchedule.span, List.mem_range, CertifierSchedule.bucket]
  have := Nat.div_le_self j s.r
  omega

/-- **A bucket's frames on the wire** are the connection's frames in it. -/
theorem filter_laneWire (s : CertifierSchedule κ) (v : κ → List (ℕ × CellView Cell)) (c : κ) {b : ℕ}
    (hb : b ∈ s.span c) : (laneWire s v c).filter (fun x => x.1 == b) = (wireFrames s v c b).map (b, ·) := by
  have h : ∀ d, ((wireFrames s v c d).map (d, ·)).filter (fun x => x.1 == b) =
      if d = b then (wireFrames s v c b).map (b, ·) else [] := by
    intro d
    by_cases hd : d = b
    · subst hd
      rw [if_pos rfl, List.filter_eq_self]
      simp
    · rw [if_neg hd, List.filter_eq_nil_iff]
      simp [hd]
  rw [laneWire, List.filter_flatMap]
  simp only [h]
  exact filterMap_flatMap_ite _ List.nodup_range hb

/-- **A lane's cell `j` is the `(j mod r)`-th of its connection's frames in its bucket**: the admission padding is in
earlier buckets, and the padding after the lane's last cell follows it. -/
theorem wireFrames_cell (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (c : κ) {j : ℕ} (hj : j < s.cells c) :
    (wireFrames s (certifierCode s a) c (s.bucket c j))[j % s.r]? = ((certifierCode s a c)[j]?).map Prod.snd := by
  have hlen : (certifierCode s a c).length = s.cells c := certifierLane_length _ _ _ _
  have h2 : j < (certifierCode s a c).length := hlen ▸ hj
  have hfst : ((certifierCode s a c)[j]).1 = s.bucket c j := by
    have h := congrArg (fun l => l[j]?) (certifierCode_fst s a c)
    simp only [List.getElem?_map, List.getElem?_eq_getElem h2, List.getElem?_range hj, Option.map_some,
      Option.some.injEq] at h
    exact h
  have hpre : ¬(s.adm c ≤ s.bucket c j ∧ s.bucket c j < s.adm c + s.D) := fun h =>
    absurd h.2 (Nat.not_lt.2 (Nat.le_add_right _ _))
  simp only [wireFrames, hpre, ite_false, List.nil_append]
  have key := getElem?_filter_countP_take (fun x : ℕ × CellView Cell => x.1 == s.bucket c j) (certifierCode s a c) j h2
    (by simp [hfst])
  have hcount : ((certifierCode s a c).take j).countP (fun x => x.1 == s.bucket c j) = j % s.r := by
    have := congrArg (fun l => (l.take j).countP (· == s.bucket c j)) (certifierCode_fst s a c)
    simp only [← List.map_take, List.countP_map, List.take_range, Nat.min_eq_left hj.le] at this
    rw [show (fun x : ℕ × CellView Cell => x.1 == s.bucket c j) = (· == s.bucket c j) ∘ Prod.fst from rfl, this,
      ← countP_range_div s.r j]
    congr 1
    funext i
    simp [CertifierSchedule.bucket]
  rw [hcount] at key
  have hlt : j % s.r < (((certifierCode s a c).filter fun x => x.1 == s.bucket c j).map Prod.snd).length := by
    rw [List.length_map]
    exact (List.getElem?_eq_some_iff.1 key).1
  rw [List.getElem?_append_left hlt, List.getElem?_map, key, List.getElem?_eq_getElem h2]

end Wire

end NetTiming
