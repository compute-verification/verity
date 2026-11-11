module

public import Proofs.Properties.Nci.LeafWidths
public import Proofs.Properties.Nci.ResolveWithin
public import Flock.Sigma
public import Mathlib.Tactic

/-!
# Resolution keeps the width

A reference of a body names a leaf of a parameter or of a node's output, and resolution follows it, activation by
activation, to the gate that computes it. This module proves what that needs of a type's leaf widths.

* `wAt_widths`: the width `Widths.wAt` reads at a leaf of a type is that leaf's entry of the type's widths.
* `axis_wide`: along an axis of an array, a leaf of a member is as wide as the leaf of the whole it is read from.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program
open NciCheck.Properties.SecurityProofs.Resolve (Good RefOk BodyOk)

/-! ## A leaf's width is its type's -/

private theorem sizeOf_lt_of_mem {fs : Array Ty} {t : Ty} (h : t ∈ fs) : sizeOf t < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  omega

private theorem sizeOf_snd_lt_of_mem {fs : Array (String × Ty)} {f : String × Ty} (h : f ∈ fs) :
    sizeOf f.2 < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  obtain ⟨k, t⟩ := f
  simp only [Prod.mk.sizeOf_spec] at this ⊢
  omega

private theorem append_left (a b : Array ℕ) (j : ℕ) (hj : j < a.size) : (a ++ b)[j]! = a[j]! := by
  rw [getElem!_pos (a ++ b) j (by rw [Array.size_append]; omega), getElem!_pos a j hj, Array.getElem_append_left hj]

private theorem append_right (a b : Array ℕ) (j : ℕ) (hj : a.size ≤ j) (hb : j < a.size + b.size) :
    (a ++ b)[j]! = b[j - a.size]! := by
  rw [getElem!_pos (a ++ b) j (by rw [Array.size_append]; omega), getElem!_pos b (j - a.size) (by omega),
    Array.getElem_append_right hj]

/-- A fold that only appends keeps what the array already holds. -/
private theorem foldl_append_keep {β : Type} (g : β → Array ℕ) (j : ℕ) :
    ∀ (l : List β) (acc : Array ℕ), j < acc.size → (l.foldl (fun a t => a ++ g t) acc)[j]! = acc[j]! := by
  intro l
  induction l with
  | nil => intro acc _; rfl
  | cons x l ih =>
    intro acc hj
    rw [List.foldl_cons, ih (acc ++ g x) (by rw [Array.size_append]; omega), append_left acc (g x) j hj]

private theorem foldl_add {β : Type} (g : β → ℕ) :
    ∀ (l : List β) (acc : ℕ), l.foldl (fun a t => a + g t) acc = acc + (l.map g).sum := by
  intro l
  induction l with
  | nil => intro acc; rw [List.foldl_nil, List.map_nil, List.sum_nil, Nat.add_zero]
  | cons x l ih => intro acc; rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

/-- Leaf `j` of copies of an element appended one after another is the element's leaf `j` modulo its leaf count. -/
private theorem copies_at {β : Type} (w : Array ℕ) (j : ℕ) :
    ∀ (l : List β) (acc : Array ℕ), acc.size ≤ j → j < acc.size + l.length * w.size →
      (l.foldl (fun a _ => a ++ w) acc)[j]! = w[(j - acc.size) % w.size]! := by
  intro l
  induction l with
  | nil => intro acc h1 h2; rw [List.length_nil, Nat.zero_mul] at h2; omega
  | cons x l ih =>
    intro acc h1 h2
    rw [List.length_cons, Nat.succ_mul] at h2
    rw [List.foldl_cons]
    by_cases h : j < acc.size + w.size
    · rw [foldl_append_keep (fun _ => w) j l (acc ++ w) (by rw [Array.size_append]; exact h),
        append_right acc w j h1 h, Nat.mod_eq_of_lt (by omega)]
    · rw [ih (acc ++ w) (by rw [Array.size_append]; omega) (by rw [Array.size_append]; omega), Array.size_append,
        Nat.mod_eq_sub_mod (show w.size ≤ j - acc.size by omega)]
      congr 2
      omega

/-- The fold `Widths.wAt` runs over a compound type's fields finds the field holding leaf `j` and returns that field's
width at the leaf, which is the entry at `j` of the fields' widths appended in order. -/
private theorem fields_at {β : Type} (lf : β → ℕ) (wf : β → ℕ → ℕ) (g : β → Array ℕ) (j : ℕ)
    (F : Option ℕ × ℕ → β → Option ℕ × ℕ) (hsome : ∀ v n f, F (some v, n) f = (some v, n))
    (hnone : ∀ n f, F (none, n) f = (if j < n + lf f then some (wf f (j - n)) else none, n + lf f)) :
    ∀ (l : List β), (∀ f ∈ l, (g f).size = lf f ∧ ∀ i < lf f, wf f i = (g f)[i]!) → ∀ acc : Array ℕ, acc.size ≤ j →
      j < acc.size + (l.map lf).sum →
      ((l.foldl F (none, acc.size)).1).getD 0 = (l.foldl (fun a t => a ++ g t) acc)[j]! := by
  have keep : ∀ (l : List β) (v n : ℕ), l.foldl F (some v, n) = (some v, n) := by
    intro l
    induction l with
    | nil => intro v n; rfl
    | cons x l ih => intro v n; rw [List.foldl_cons, hsome, ih]
  intro l
  induction l with
  | nil => intro _ acc h1 h2; rw [List.map_nil, List.sum_nil] at h2; omega
  | cons x l ih =>
    intro hl acc h1 h2
    obtain ⟨hx, hwx⟩ := hl x (List.mem_cons_self ..)
    rw [List.map_cons, List.sum_cons] at h2
    rw [List.foldl_cons, List.foldl_cons, hnone]
    by_cases h : j < acc.size + lf x
    · simp only [h, ↓reduceIte]
      rw [keep, foldl_append_keep g j l (acc ++ g x) (by rw [Array.size_append]; omega),
        append_right acc (g x) j h1 (by omega), ← hwx (j - acc.size) (by omega)]
      rfl
    · simp only [h, ↓reduceIte]
      have := ih (fun f hf => hl f (List.mem_cons_of_mem _ hf)) (acc ++ g x) (by rw [Array.size_append]; omega)
        (by rw [Array.size_append]; omega)
      rw [Array.size_append, hx] at this
      exact this

private theorem wAt_widths_all : (t : Ty) → ∀ j, j < t.leaves → Widths.wAt t j = (Ty.widths t)[j]!
  | .value w => by
    intro j hj
    rw [Ty.leaves] at hj
    have : j = 0 := by omega
    subst this
    rw [Widths.wAt, Ty.widths]
    rfl
  | .array n e => by
    intro j hj
    rw [Ty.leaves] at hj
    have hL : 0 < e.leaves := by
      rcases Nat.eq_zero_or_pos e.leaves with h | h
      · rw [h] at hj; omega
      · exact h
    have hs : (Ty.widths e).size = e.leaves := Flock.Sigma.widths_length e
    rw [Widths.wAt, Ty.widths]
    have hne : (e.leaves == 0) = false := by
      rw [beq_eq_false_iff_ne]; omega
    simp only [hne, Bool.false_eq_true, ↓reduceIte]
    rw [wAt_widths_all e (j % e.leaves) (Nat.mod_lt _ hL), ← Array.foldl_toList,
      copies_at (Ty.widths e) j (Array.range n).toList #[] (Nat.zero_le _)
        (by rw [Array.toList_range, List.length_range, hs]; show j < 0 + n * e.leaves; omega), hs]
    rfl
  | .tuple fs => by
    intro j hj
    rw [Ty.leaves, ← Array.foldl_toList, foldl_add] at hj
    rw [Widths.wAt, Ty.widths, ← Array.foldl_toList, ← Array.foldl_toList]
    exact fields_at Ty.leaves Widths.wAt Ty.widths j _ (fun _ _ _ => rfl) (fun _ _ => rfl) fs.toList
      (fun f hf => ⟨Flock.Sigma.widths_length f, fun i hi => wAt_widths_all f i hi⟩) #[] (Nat.zero_le _) hj
  | .record fs => by
    intro j hj
    rw [Ty.leaves, ← Array.foldl_toList, foldl_add] at hj
    rw [Widths.wAt, Ty.widths, ← Array.foldl_toList, ← Array.foldl_toList]
    exact fields_at (fun f : String × Ty => f.2.leaves) (fun f => Widths.wAt f.2) (fun f => Ty.widths f.2) j _
      (fun _ _ _ => rfl) (fun _ _ => rfl) fs.toList
      (fun f hf => ⟨Flock.Sigma.widths_length f.2, fun i hi => wAt_widths_all f.2 i hi⟩) #[] (Nat.zero_le _) hj
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem (Array.mem_def.mpr hf)
  · exact sizeOf_snd_lt_of_mem (Array.mem_def.mpr hf)

theorem wAt_widths (t : Ty) (j : ℕ) (hj : j < t.leaves) : Widths.wAt t j = (Ty.widths t)[j]! := by
  exact wAt_widths_all t j hj

/-! ## Along an axis -/

/-- Along an axis of an array, leaf `j` of member `m` is as wide as the leaf of the whole it is read from. -/
private theorem axis_wide : ∀ (ax : ℕ) (t : Ty) (ln : ℕ) (mt : Ty) (L : ℕ), t.removeAxis ax = some (ln, mt) →
    t.axisDims ax = some (ln, L) → ∀ j < mt.leaves, ∀ m < ln,
      Widths.wAt t ((j / L * ln + m) * L + j % L) = Widths.wAt mt j := by
  intro ax
  induction ax with
  | zero =>
    intro t ln mt L h hd j hj m hm
    cases t <;> simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at h
    rename_i n e
    obtain ⟨hn, he⟩ := h
    subst hn he
    simp only [Ty.axisDims, Option.some.injEq, Prod.mk.injEq, true_and] at hd
    subst hd
    have hne : (e.leaves == 0) = false := by
      rw [beq_eq_false_iff_ne]; omega
    rw [Widths.wAt]
    simp only [hne, Bool.false_eq_true, ↓reduceIte]
    rw [Nat.div_eq_of_lt hj, Nat.mod_eq_of_lt hj, Nat.zero_mul, Nat.zero_add, Nat.add_comm, Nat.add_mul_mod_self_right,
      Nat.mod_eq_of_lt hj]
  | succ ax ih =>
    intro t ln mt L h hd j hj m hm
    cases t <;> simp only [Ty.removeAxis, Option.map_eq_some_iff, Prod.mk.injEq, reduceCtorEq, Prod.exists] at h
    rename_i n e
    obtain ⟨ln', inner, hr, hln, hmt⟩ := h
    subst hln hmt
    simp only [Ty.axisDims] at hd
    obtain ⟨L', hax, hlv, hdvd, hb⟩ := BodiesOk.axis_spec ax e ln' inner hr
    rw [hd] at hax
    simp only [Option.some.injEq, Prod.mk.injEq, true_and] at hax
    subst hax
    rw [BodiesOk.leaves_array] at hj
    have hIpos : 0 < inner.leaves := by
      rcases Nat.eq_zero_or_pos inner.leaves with h0 | h0
      · rw [h0] at hj; omega
      · exact h0
    obtain ⟨s, hs⟩ := hdvd
    have hLpos : 0 < L := by
      rcases Nat.eq_zero_or_pos L with h0 | h0
      · rw [h0, Nat.zero_mul] at hs; omega
      · exact h0
    have hr' : j % inner.leaves < inner.leaves := Nat.mod_lt _ hIpos
    have hjr : j = j % inner.leaves + (j / inner.leaves * s) * L := by
      have h1 := Nat.div_add_mod j inner.leaves
      have h2 : (j / inner.leaves * s) * L = inner.leaves * (j / inner.leaves) := by
        rw [Nat.mul_assoc, Nat.mul_comm s L, ← hs, Nat.mul_comm]
      rw [h2]
      omega
    have hdiv : j / L = j % inner.leaves / L + j / inner.leaves * s := by
      conv => lhs; rw [hjr]
      rw [Nat.add_mul_div_right _ _ hLpos]
    have hmod : j % L = j % inner.leaves % L := by
      conv => lhs; rw [hjr]
      rw [Nat.add_mul_mod_self_right]
    have hA := hb _ hr' m hm
    have e1 : ((j % inner.leaves / L + j / inner.leaves * s) * ln' + m) * L + j % inner.leaves % L =
        ((j % inner.leaves / L * ln' + m) * L + j % inner.leaves % L) + j / inner.leaves * e.leaves := by
      rw [hlv, hs]
      simp only [Nat.mul_add, Nat.mul_comm, Nat.mul_left_comm, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
    have hne : (e.leaves == 0) = false := by
      rw [beq_eq_false_iff_ne]; omega
    have hni : (inner.leaves == 0) = false := by
      rw [beq_eq_false_iff_ne]; omega
    rw [Widths.wAt, Widths.wAt]
    simp only [hne, hni, Bool.false_eq_true, ↓reduceIte]
    rw [hdiv, hmod, e1, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hA]
    exact ih e ln' inner L hr hd (j % inner.leaves) hr' m hm

end NciCheck.Properties.SecurityProofs.LeafWidths
