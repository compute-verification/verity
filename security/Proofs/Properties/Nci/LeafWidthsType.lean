module

public import Flock.Program
public import Mathlib.Tactic

/-!
# The widths of a type's leaves

What `Flock.Program.Widths` computes from a type's structure, said leaf by leaf (`Widths.wAt`):

* `wAt_mem_rangeWidths`: the width of a leaf of a range is among the range's widths (`Widths.rangeWidths`);
* `homo_wAt`: a type of one width (`Widths.homo`) has it at every leaf;
* `cover_spec`: the part of a type that `Widths.cover` returns for a range holds the range, lies inside the type, and
  its leaves have the widths of the type's leaves from the one it starts at (`Covers`).

A Tuple's and a Record's fields are read by one fold each (`wStep`, `rStep`, `cStep`). The field that holds leaf `j`
is the first that ends past `j` (`field_of_leaf`), and `wAt` reads `j` there (`fold_wAt`).
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

/-- Induction on a type, a Tuple's and a Record's fields taken as lists. -/
theorem ty_induction {motive : Ty → Prop} (value : ∀ w, motive (.value w))
    (array : ∀ n e, motive e → motive (.array n e))
    (tuple : ∀ fs : Array Ty, (∀ f ∈ fs.toList, motive f) → motive (.tuple fs))
    (record : ∀ fs : Array (String × Ty), (∀ f ∈ fs.toList, motive f.2) → motive (.record fs)) (t : Ty) :
    motive t := by
  have key : ∀ (n : ℕ) (t : Ty), sizeOf t < n → motive t := by
    intro n
    induction n with
    | zero => intro t h; omega
    | succ n ih =>
      intro t hs
      cases t with
      | value w => exact value w
      | array k e => exact array k e (ih e (by simp at hs; omega))
      | tuple fs =>
        refine tuple fs fun f hf => ih f ?_
        have := Array.sizeOf_lt_of_mem (Array.mem_def.2 hf)
        simp at hs
        omega
      | record fs =>
        refine record fs fun f hf => ih f.2 ?_
        have := Array.sizeOf_lt_of_mem (Array.mem_def.2 hf)
        have h3 : sizeOf f.2 < sizeOf f := by
          rcases f with ⟨x, y⟩
          simp
        simp at hs
        omega
  exact key _ t (Nat.lt_succ_self _)

/-! ## The folds over fields -/

section Fields

variable {α : Type} (π : α → Ty)

/-- One field of the fold `Widths.wAt` runs over a Tuple's or a Record's fields. -/
def wStep (j : ℕ) (acc : Option ℕ × ℕ) (f : α) : Option ℕ × ℕ :=
  match acc.1 with
  | some _ => acc
  | none =>
    (if j < acc.2 + (π f).leaves then some (Widths.wAt (π f) (j - acc.2)) else none, acc.2 + (π f).leaves)

/-- One field of the fold `Widths.rangeWidths` runs over a Tuple's or a Record's fields. -/
def rStep (lo hi : ℕ) (acc : List ℕ × ℕ) (f : α) : List ℕ × ℕ :=
  (if acc.2 + (π f).leaves ≤ lo || hi ≤ acc.2 then acc.1
    else acc.1 ++ Widths.rangeWidths (π f) (lo - acc.2) (min hi (acc.2 + (π f).leaves) - acc.2),
    acc.2 + (π f).leaves)

/-- One field of the fold `Widths.cover` runs over a Tuple's or a Record's fields. -/
def cStep (lo hi : ℕ) (acc : Option (Ty × ℕ) × ℕ) (f : α) : Option (Ty × ℕ) × ℕ :=
  match acc.1 with
  | some _ => acc
  | none =>
    (if acc.2 ≤ lo && hi ≤ acc.2 + (π f).leaves then
        some ((Widths.cover (π f) (lo - acc.2) (hi - acc.2)).1,
          (Widths.cover (π f) (lo - acc.2) (hi - acc.2)).2 + acc.2)
      else none,
      acc.2 + (π f).leaves)

/-- The leaves of fields, summed from `o`. -/
theorem foldl_leaves (l : List α) (o : ℕ) :
    l.foldl (fun a t => a + (π t).leaves) o = o + (l.map fun f => (π f).leaves).sum := by
  induction l generalizing o with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]
    omega

/-- A leaf of the fields lies in one of them. -/
theorem field_of_leaf (l : List α) (o j : ℕ) (ho : o ≤ j) (hj : j < o + (l.map fun f => (π f).leaves).sum) :
    ∃ pre f post p, l = pre ++ f :: post ∧ p = o + (pre.map fun f => (π f).leaves).sum ∧ p ≤ j ∧
      j < p + (π f).leaves := by
  induction l generalizing o with
  | nil =>
    simp only [List.map_nil, List.sum_nil] at hj
    omega
  | cons x xs ih =>
    by_cases hx : j < o + (π x).leaves
    · exact ⟨[], x, xs, o, rfl, by simp, ho, hx⟩
    · simp only [List.map_cons, List.sum_cons] at hj
      obtain ⟨pre, f, post, p, e, hp, h1, h2⟩ := ih (o + (π x).leaves) (by omega) (by omega)
      refine ⟨x :: pre, f, post, p, by rw [e]; rfl, ?_, h1, h2⟩
      simp only [List.map_cons, List.sum_cons]
      omega

/-- The fields before one, the one and those after it have their leaves between them. -/
theorem field_le (pre : List α) (f : α) (post : List α) :
    (pre.map fun f => (π f).leaves).sum + (π f).leaves ≤ ((pre ++ f :: post).map fun f => (π f).leaves).sum := by
  simp only [List.map_append, List.map_cons, List.sum_append, List.sum_cons]
  omega

/-- `wAt`'s fold reads a leaf in the field that holds it. -/
theorem fold_wAt (j : ℕ) (pre : List α) (f : α) (post : List α) (o p : ℕ)
    (hp : p = o + (pre.map fun f => (π f).leaves).sum) (h1 : p ≤ j) (h2 : j < p + (π f).leaves) :
    ((pre ++ f :: post).foldl (wStep π j) (none, o)).1 = some (Widths.wAt (π f) (j - p)) := by
  have keep : ∀ (l : List α) (w o : ℕ), (l.foldl (wStep π j) (some w, o)).1 = some w := by
    intro l
    induction l with
    | nil => intro w o; rfl
    | cons x xs ih => intro w o; exact ih w o
  induction pre generalizing o with
  | nil =>
    simp only [List.map_nil, List.sum_nil, Nat.add_zero] at hp
    subst hp
    rw [List.nil_append, List.foldl_cons]
    show (post.foldl (wStep π j)
      (if j < p + (π f).leaves then some (Widths.wAt (π f) (j - p)) else none, p + (π f).leaves)).1 = _
    rw [ite_eq_left h2]
    exact keep post _ _
  | cons x pre ih =>
    simp only [List.map_cons, List.sum_cons] at hp
    rw [List.cons_append, List.foldl_cons]
    show ((pre ++ f :: post).foldl (wStep π j)
      (if j < o + (π x).leaves then some (Widths.wAt (π x) (j - o)) else none, o + (π x).leaves)).1 = _
    rw [ite_eq_right (by omega)]
    exact ih (o + (π x).leaves) (by omega)

/-- `rangeWidths`' fold keeps the widths of a field that meets the range. -/
theorem fold_rangeWidths (lo hi : ℕ) (pre : List α) (f : α) (post : List α) (o : ℕ) (acc : List ℕ) (p : ℕ)
    (hp : p = o + (pre.map fun f => (π f).leaves).sum) (h1 : lo < p + (π f).leaves) (h2 : p < hi) (x : ℕ)
    (hx : x ∈ Widths.rangeWidths (π f) (lo - p) (min hi (p + (π f).leaves) - p)) :
    x ∈ ((pre ++ f :: post).foldl (rStep π lo hi) (acc, o)).1 := by
  have keep : ∀ (l : List α) (acc : List ℕ) (o : ℕ), x ∈ acc → x ∈ (l.foldl (rStep π lo hi) (acc, o)).1 := by
    intro l
    induction l with
    | nil => intro acc o h; exact h
    | cons y ys ih =>
      intro acc o h
      rw [List.foldl_cons]
      refine ih _ _ ?_
      show x ∈ if (decide (o + (π y).leaves ≤ lo) || decide (hi ≤ o)) = true then acc else acc ++ _
      split
      · exact h
      · exact List.mem_append_left _ h
  induction pre generalizing o acc with
  | nil =>
    simp only [List.map_nil, List.sum_nil, Nat.add_zero] at hp
    subst hp
    rw [List.nil_append, List.foldl_cons]
    refine keep post _ _ ?_
    show x ∈ if (decide (p + (π f).leaves ≤ lo) || decide (hi ≤ p)) = true then acc
      else acc ++ Widths.rangeWidths (π f) (lo - p) (min hi (p + (π f).leaves) - p)
    rw [ite_eq_right (by simp; omega)]
    exact List.mem_append_right _ hx
  | cons y pre ih =>
    simp only [List.map_cons, List.sum_cons] at hp
    rw [List.cons_append, List.foldl_cons]
    exact ih (o + (π y).leaves) _ (by omega)

/-- `cover`'s fold returns the cover of the range within a field that holds it, where it returns one. -/
theorem fold_cover (lo hi : ℕ) (l : List α) (o : ℕ) (c : Ty × ℕ)
    (h : (l.foldl (cStep π lo hi) (none, o)).1 = some c) :
    ∃ pre f post p, l = pre ++ f :: post ∧ p = o + (pre.map fun f => (π f).leaves).sum ∧ p ≤ lo ∧
      hi ≤ p + (π f).leaves ∧
      c = ((Widths.cover (π f) (lo - p) (hi - p)).1, (Widths.cover (π f) (lo - p) (hi - p)).2 + p) := by
  have keep : ∀ (l : List α) (c : Ty × ℕ) (o : ℕ), (l.foldl (cStep π lo hi) (some c, o)).1 = some c := by
    intro l
    induction l with
    | nil => intro c o; rfl
    | cons x xs ih => intro c o; exact ih c o
  induction l generalizing o with
  | nil => exact absurd h (by simp)
  | cons x xs ih =>
    rw [List.foldl_cons] at h
    by_cases hc : (decide (o ≤ lo) && decide (hi ≤ o + (π x).leaves)) = true
    · have e : cStep π lo hi (none, o) x =
          (some ((Widths.cover (π x) (lo - o) (hi - o)).1, (Widths.cover (π x) (lo - o) (hi - o)).2 + o),
            o + (π x).leaves) := by
        show (if (decide (o ≤ lo) && decide (hi ≤ o + (π x).leaves)) = true then _ else none, _) = _
        rw [ite_eq_left hc]
      rw [e, keep] at h
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      exact ⟨[], x, xs, o, rfl, by simp, hc.1, hc.2, (Option.some.inj h).symm⟩
    · have e : cStep π lo hi (none, o) x = (none, o + (π x).leaves) := by
        show (if (decide (o ≤ lo) && decide (hi ≤ o + (π x).leaves)) = true then _ else none, _) = _
        rw [ite_eq_right hc]
      rw [e] at h
      obtain ⟨pre, f, post, p, e', hp, h1, h2, h3⟩ := ih (o + (π x).leaves) h
      refine ⟨x :: pre, f, post, p, by rw [e']; rfl, ?_, h1, h2, h3⟩
      simp only [List.map_cons, List.sum_cons]
      omega

end Fields

/-! ## A Tuple's and a Record's leaves, by these folds -/

/-- A Tuple's leaves are its fields'. -/
theorem leaves_tuple (fs : Array Ty) : (Ty.tuple fs).leaves = (fs.toList.map fun f => f.leaves).sum := by
  rw [Ty.leaves, ← Array.foldl_toList, foldl_leaves (fun t : Ty => t), Nat.zero_add]

/-- A Record's leaves are its fields'. -/
theorem leaves_record (fs : Array (String × Ty)) :
    (Ty.record fs).leaves = (fs.toList.map fun f => f.2.leaves).sum := by
  rw [Ty.leaves, ← Array.foldl_toList, foldl_leaves (fun f : String × Ty => f.2), Nat.zero_add]

/-- `Widths.wAt` on a Tuple is `wStep` folded over its fields. -/
theorem wAt_tuple (fs : Array Ty) (j : ℕ) :
    Widths.wAt (.tuple fs) j = (fs.toList.foldl (wStep (fun t : Ty => t) j) (none, 0)).1.getD 0 := by
  rw [Widths.wAt, ← Array.foldl_toList]
  rfl

/-- `Widths.wAt` on a Record is `wStep` folded over its fields. -/
theorem wAt_record (fs : Array (String × Ty)) (j : ℕ) :
    Widths.wAt (.record fs) j = (fs.toList.foldl (wStep (fun f : String × Ty => f.2) j) (none, 0)).1.getD 0 := by
  rw [Widths.wAt, ← Array.foldl_toList]
  rfl

/-- `Widths.rangeWidths` on a Tuple is `rStep` folded over its fields. -/
theorem rangeWidths_tuple (fs : Array Ty) (lo hi : ℕ) :
    Widths.rangeWidths (.tuple fs) lo hi =
      (fs.toList.foldl (rStep (fun t : Ty => t) lo hi) ([], 0)).1.eraseDups := by
  rw [Widths.rangeWidths, ← Array.foldl_toList]
  rfl

/-- `Widths.rangeWidths` on a Record is `rStep` folded over its fields. -/
theorem rangeWidths_record (fs : Array (String × Ty)) (lo hi : ℕ) :
    Widths.rangeWidths (.record fs) lo hi =
      (fs.toList.foldl (rStep (fun f : String × Ty => f.2) lo hi) ([], 0)).1.eraseDups := by
  rw [Widths.rangeWidths, ← Array.foldl_toList]
  rfl

/-- `Widths.cover` on a Tuple is `cStep` folded over its fields. -/
theorem cover_tuple (fs : Array Ty) (lo hi : ℕ) :
    Widths.cover (.tuple fs) lo hi =
      (fs.toList.foldl (cStep (fun t : Ty => t) lo hi) (none, 0)).1.getD (.tuple fs, 0) := by
  rw [Widths.cover, ← Array.foldl_toList]
  rfl

/-- `Widths.cover` on a Record is `cStep` folded over its fields. -/
theorem cover_record (fs : Array (String × Ty)) (lo hi : ℕ) :
    Widths.cover (.record fs) lo hi =
      (fs.toList.foldl (cStep (fun f : String × Ty => f.2) lo hi) (none, 0)).1.getD (.record fs, 0) := by
  rw [Widths.cover, ← Array.foldl_toList]
  rfl

/-! ## The width of a leaf is among its range's -/

/-- The fields' case of `wAt_mem_rangeWidths`. -/
theorem mem_fields {α : Type} (π : α → Ty) (l : List α) (lo hi j : ℕ) (hlo : lo ≤ j) (hhi : j < hi)
    (hj : j < (l.map fun f => (π f).leaves).sum)
    (ih : ∀ f ∈ l, ∀ lo hi j, lo ≤ j → j < hi → j < (π f).leaves →
      Widths.wAt (π f) j ∈ Widths.rangeWidths (π f) lo hi) :
    (l.foldl (wStep π j) (none, 0)).1.getD 0 ∈ (l.foldl (rStep π lo hi) ([], 0)).1 := by
  obtain ⟨pre, f, post, p, e, hp, h1, h2⟩ := field_of_leaf π l 0 j (Nat.zero_le _) (by omega)
  subst e
  rw [fold_wAt π j pre f post 0 p hp h1 h2]
  exact fold_rangeWidths π lo hi pre f post 0 [] p hp (by omega) (by omega) _
    (ih f (by simp) (lo - p) (min hi (p + (π f).leaves) - p) (j - p) (by omega) (by omega) (by omega))

/-- **The width of a leaf of a range is among the range's widths.** -/
theorem wAt_mem_rangeWidths (t : Ty) (lo hi j : ℕ) (hlo : lo ≤ j) (hhi : j < hi) (hj : j < t.leaves) :
    Widths.wAt t j ∈ Widths.rangeWidths t lo hi := by
  induction t using ty_induction generalizing lo hi j with
  | value w =>
    rw [Ty.leaves] at hj
    rw [Widths.wAt, Widths.rangeWidths]
    have h0 : lo = 0 := by omega
    subst h0
    rw [ite_eq_left (by simp; omega)]
    exact List.mem_singleton.2 rfl
  | array n e ih =>
    rw [Ty.leaves] at hj
    have hL : 0 < e.leaves := by
      rcases Nat.eq_zero_or_pos e.leaves with h0 | h0
      · rw [h0] at hj
        simp at hj
      · exact h0
    have hjL := Nat.mod_lt j hL
    have hq := Nat.div_add_mod' j e.leaves
    have ha := Nat.div_mul_le_self lo e.leaves
    have hlo' : lo / e.leaves ≤ j / e.leaves := Nat.div_le_div_right hlo
    have hH : j ≤ min hi (n * e.leaves) - 1 := by omega
    have hhi' : j / e.leaves ≤ (min hi (n * e.leaves) - 1) / e.leaves := Nat.div_le_div_right hH
    have hb := Nat.div_mul_le_self (min hi (n * e.leaves) - 1) e.leaves
    have c1 : ¬ ((e.leaves == 0) = true) := by simp; omega
    have c2 : ¬ ((e.leaves == 0 || decide (min hi (n * e.leaves) ≤ lo)) = true) := by simp; omega
    rw [Widths.wAt, Widths.rangeWidths, ite_eq_right c1, ite_eq_right c2]
    show Widths.wAt e (j % e.leaves) ∈
      if (lo / e.leaves == (min hi (n * e.leaves) - 1) / e.leaves) = true then _ else _
    by_cases hab : lo / e.leaves = (min hi (n * e.leaves) - 1) / e.leaves
    · rw [ite_eq_left (by simpa using hab)]
      have hqa : j / e.leaves = lo / e.leaves := by omega
      rw [hqa] at hq
      exact ih _ _ _ (by omega) (by omega) hjL
    · rw [ite_eq_right (by simpa using hab), List.mem_eraseDups]
      simp only [List.mem_append]
      by_cases hqa : j / e.leaves = lo / e.leaves
      · rw [hqa] at hq
        exact Or.inl (Or.inl (ih _ _ _ (by omega) hjL hjL))
      · by_cases hqb : j / e.leaves = (min hi (n * e.leaves) - 1) / e.leaves
        · rw [hqb] at hq
          exact Or.inr (ih _ _ _ (Nat.zero_le _) (by omega) hjL)
        · rw [ite_eq_left (by omega)]
          exact Or.inl (Or.inr (ih _ _ _ (Nat.zero_le _) hjL hjL))
  | tuple fs ih =>
    rw [wAt_tuple, rangeWidths_tuple, List.mem_eraseDups]
    exact mem_fields (fun t : Ty => t) fs.toList lo hi j hlo hhi (by rwa [leaves_tuple] at hj) ih
  | record fs ih =>
    rw [wAt_record, rangeWidths_record, List.mem_eraseDups]
    exact mem_fields (fun f : String × Ty => f.2) fs.toList lo hi j hlo hhi (by rwa [leaves_record] at hj) ih

/-- **A type of one width has it at every leaf.** -/
theorem homo_wAt (t : Ty) (w : ℕ) (h : Widths.homo t = some w) (j : ℕ) (hj : j < t.leaves) :
    Widths.wAt t j = w := by
  have hm := wAt_mem_rangeWidths t 0 t.leaves j (Nat.zero_le _) hj hj
  unfold Widths.homo at h
  split at h
  · rename_i w' hw'
    rw [hw'] at hm
    rw [List.mem_singleton.1 hm]
    exact Option.some.inj h
  · exact absurd h (by simp)

/-! ## The part that covers a range -/

/-- `c` is a part of `t` that holds its leaves `lo … hi - 1`: it starts at leaf `c.2` of `t`, lies inside `t`, and
its leaves have the widths of `t`'s from there. -/
def Covers (t : Ty) (lo hi : ℕ) (c : Ty × ℕ) : Prop :=
  c.2 ≤ lo ∧ hi ≤ c.2 + c.1.leaves ∧ c.2 + c.1.leaves ≤ t.leaves ∧
    ∀ j, c.2 ≤ j → j < c.2 + c.1.leaves → Widths.wAt t j = Widths.wAt c.1 (j - c.2)

/-- A type covers each range of its leaves. -/
theorem covers_self (t : Ty) (lo hi : ℕ) (hh : hi ≤ t.leaves) : Covers t lo hi (t, 0) :=
  ⟨Nat.zero_le _, by simpa using hh, by simp, fun _ _ _ => rfl⟩

/-- The fields' case of `cover_spec`. -/
theorem covers_fields {α : Type} (π : α → Ty) (T : Ty) (l : List α)
    (hT : T.leaves = (l.map fun f => (π f).leaves).sum)
    (hw : ∀ j, Widths.wAt T j = (l.foldl (wStep π j) (none, 0)).1.getD 0)
    (ih : ∀ f ∈ l, ∀ lo hi, lo < hi → hi ≤ (π f).leaves → Covers (π f) lo hi (Widths.cover (π f) lo hi))
    (lo hi : ℕ) (h : lo < hi) (hh : hi ≤ T.leaves) :
    Covers T lo hi ((l.foldl (cStep π lo hi) (none, 0)).1.getD (T, 0)) := by
  rcases hr : (l.foldl (cStep π lo hi) (none, 0)).1 with _ | c
  · exact covers_self T lo hi hh
  · obtain ⟨pre, f, post, p, e, hp, h1, h2, hc⟩ := fold_cover π lo hi l 0 c hr
    rw [Nat.zero_add] at hp
    obtain ⟨i1, i2, i3, i4⟩ := ih f (by rw [e]; simp) (lo - p) (hi - p) (by omega) (by omega)
    have hle := field_le π pre f post
    rw [← e, ← hT, ← hp] at hle
    subst hc
    refine ⟨?_, ?_, ?_, fun j hj1 hj2 => ?_⟩
    · show (Widths.cover (π f) (lo - p) (hi - p)).2 + p ≤ lo
      omega
    · show hi ≤ (Widths.cover (π f) (lo - p) (hi - p)).2 + p + (Widths.cover (π f) (lo - p) (hi - p)).1.leaves
      omega
    · show (Widths.cover (π f) (lo - p) (hi - p)).2 + p + (Widths.cover (π f) (lo - p) (hi - p)).1.leaves ≤ _
      omega
    · have hj1' : (Widths.cover (π f) (lo - p) (hi - p)).2 + p ≤ j := hj1
      have hj2' : j < (Widths.cover (π f) (lo - p) (hi - p)).2 + p +
          (Widths.cover (π f) (lo - p) (hi - p)).1.leaves := hj2
      show Widths.wAt T j = Widths.wAt (Widths.cover (π f) (lo - p) (hi - p)).1
        (j - ((Widths.cover (π f) (lo - p) (hi - p)).2 + p))
      rw [hw j, e, fold_wAt π j pre f post 0 p (by omega) (by omega) (by omega), Option.getD_some,
        i4 (j - p) (by omega) (by omega)]
      congr 1
      omega

/-- The part `Widths.cover` returns covers the range. -/
theorem covers_cover (t : Ty) (lo hi : ℕ) (h : lo < hi) (hh : hi ≤ t.leaves) :
    Covers t lo hi (Widths.cover t lo hi) := by
  induction t using ty_induction generalizing lo hi with
  | value w =>
    rw [Widths.cover]
    exact covers_self _ lo hi hh
  | array n e ih =>
    rw [Widths.cover]
    split
    · exact covers_self _ lo hi hh
    · rename_i hc
      simp only [Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
      obtain ⟨hL0, hab⟩ := hc
      have hL : 0 < e.leaves := Nat.pos_of_ne_zero hL0
      have ha := Nat.div_mul_le_self lo e.leaves
      have hb := Nat.div_add_mod' (hi - 1) e.leaves
      have hbL := Nat.mod_lt (hi - 1) hL
      rw [← hab] at hb
      rw [Ty.leaves] at hh
      have han : lo / e.leaves < n := by
        rw [hab]
        exact (Nat.div_lt_iff_lt_mul hL).2 (by omega)
      have han' : (lo / e.leaves + 1) * e.leaves ≤ n * e.leaves := Nat.mul_le_mul_right _ han
      rw [Nat.add_mul, Nat.one_mul] at han'
      obtain ⟨i1, i2, i3, i4⟩ := ih (lo - lo / e.leaves * e.leaves) (hi - lo / e.leaves * e.leaves)
        (by omega) (by omega)
      refine ⟨?_, ?_, ?_, fun j hj1 hj2 => ?_⟩
      · show (Widths.cover e _ _).2 + lo / e.leaves * e.leaves ≤ lo
        omega
      · show hi ≤ (Widths.cover e _ _).2 + lo / e.leaves * e.leaves + (Widths.cover e _ _).1.leaves
        omega
      · show (Widths.cover e _ _).2 + lo / e.leaves * e.leaves + (Widths.cover e _ _).1.leaves ≤
          (Ty.array n e).leaves
        rw [Ty.leaves]
        omega
      · have hj1' : (Widths.cover e (lo - lo / e.leaves * e.leaves) (hi - lo / e.leaves * e.leaves)).2 +
            lo / e.leaves * e.leaves ≤ j := hj1
        have hj2' : j < (Widths.cover e (lo - lo / e.leaves * e.leaves) (hi - lo / e.leaves * e.leaves)).2 +
            lo / e.leaves * e.leaves +
            (Widths.cover e (lo - lo / e.leaves * e.leaves) (hi - lo / e.leaves * e.leaves)).1.leaves := hj2
        show Widths.wAt (Ty.array n e) j =
          Widths.wAt (Widths.cover e (lo - lo / e.leaves * e.leaves) (hi - lo / e.leaves * e.leaves)).1
            (j - ((Widths.cover e (lo - lo / e.leaves * e.leaves) (hi - lo / e.leaves * e.leaves)).2 +
              lo / e.leaves * e.leaves))
        have hjm : j % e.leaves = j - lo / e.leaves * e.leaves := by
          have e1 : j = (j - lo / e.leaves * e.leaves) + lo / e.leaves * e.leaves := by omega
          rw [e1, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
          omega
        rw [Widths.wAt, ite_eq_right (by simpa using hL0), hjm, i4 _ (by omega) (by omega)]
        congr 1
        omega
  | tuple fs ih =>
    rw [cover_tuple]
    exact covers_fields (fun t : Ty => t) _ fs.toList (leaves_tuple fs) (wAt_tuple fs) ih lo hi h hh
  | record fs ih =>
    rw [cover_record]
    exact covers_fields (fun f : String × Ty => f.2) _ fs.toList (leaves_record fs) (wAt_record fs) ih lo hi h hh

/-- **The part `Widths.cover` returns for a range** starts at or before the range, ends at or after it, lies inside
the type, and its leaves have the widths of the type's leaves from the one it starts at. -/
theorem cover_spec (t : Ty) (lo hi : ℕ) (h : lo < hi) (hh : hi ≤ t.leaves) :
    (Widths.cover t lo hi).2 ≤ lo ∧ hi ≤ (Widths.cover t lo hi).2 + (Widths.cover t lo hi).1.leaves ∧
      (Widths.cover t lo hi).2 + (Widths.cover t lo hi).1.leaves ≤ t.leaves ∧
      ∀ j, (Widths.cover t lo hi).2 ≤ j → j < (Widths.cover t lo hi).2 + (Widths.cover t lo hi).1.leaves →
        Widths.wAt t j = Widths.wAt (Widths.cover t lo hi).1 (j - (Widths.cover t lo hi).2) :=
  covers_cover t lo hi h hh

end NciCheck.Properties.SecurityProofs.LeafWidths
