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

end Wire

end NetTiming
