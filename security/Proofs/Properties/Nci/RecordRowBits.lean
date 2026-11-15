module

public import Proofs.Properties.Nci.RecordTakes
public import Proofs.Properties.Nci.RecordGrid
public import Proofs.Properties.Spot.Credited
public import Mathlib.Tactic

/-!
# A row's bits

A row is its leaves' bits in leaf order, and `bitAt` is the bit at which a leaf starts.

* `bitAt_succ`, `bitAt_size`, `bitAt_mono`: a leaf starts where the leaf before it ends, the leaves end at the row's
  bits, and an earlier leaf ends where a later one starts, or before.
* `outRow_bits`: an output row's bits are its field's.
* `inRow_size`, `inRow_bits`: a row of a parameter cut `k` ways has the parameter's leaves over `k`, and, where `k`
  divides the parameter's outer length, its bits over `k`: an array's leaves are its length's copies of its
  element's (`widths_array`), so a row is whole copies, and any other parameter is one row.
* `outName_inj`: an output row's name says which field it is.
* `unitAt_fun`: a template's unit is one activation. The units before a root node of the template
  (`unitsBefore`) grow by the node's members from one node to the next (`unitsBefore_succ`), so a unit lies in
  one node and is one member of it.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Grid fieldsOf outerOf)
open NciCheck.Record (inSlot)
open NciCheck.Record.SecurityProofs (slotOf OneGrid)
open Spot.Bits (bitsOf)

/-! ## Where a leaf starts -/

/-- Leaf `l + 1` of a row starts where leaf `l` ends. -/
theorem bitAt_succ (ws : Array ℕ) (l : ℕ) (hl : l < ws.size) : bitAt ws (l + 1) = bitAt ws l + ws[l]! := by
  unfold bitAt
  rw [Array.extract_succ_right (Nat.succ_pos l) hl, Array.foldl_push, getElem!_pos ws l hl]

/-- A row's leaves end at the row's bits. -/
theorem bitAt_size (ws : Array ℕ) : bitAt ws ws.size = ws.foldl (· + ·) 0 := by
  unfold bitAt
  rw [Array.extract_eq_self_of_le (Nat.le_refl _)]

/-- A later leaf starts no earlier. -/
theorem bitAt_le (ws : Array ℕ) (l d : ℕ) (h : l + d ≤ ws.size) : bitAt ws l ≤ bitAt ws (l + d) := by
  induction d with
  | zero => exact Nat.le_refl _
  | succ d ih =>
    have e := bitAt_succ ws (l + d) (by omega)
    have h1 := ih (by omega)
    rw [← Nat.add_assoc, e]
    omega

/-- The leaves of a row are apart: an earlier leaf ends where a later one starts, or before. -/
theorem bitAt_mono (ws : Array ℕ) (l l' : ℕ) (h : l < l') (hl : l' ≤ ws.size) : bitAt ws l + ws[l]! ≤ bitAt ws l' := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le (show l + 1 ≤ l' from h)
  have e := bitAt_succ ws l (by omega)
  have h1 := bitAt_le ws (l + 1) d hl
  omega

/-! ## A row's bits -/

/-- An output row's bits are its field's. -/
theorem outRow_bits (d : Defn) (f : ℕ) : (outRow d f).2.foldl (· + ·) 0 = bitsOf (fieldsOf d.ret)[f]! := rfl

/-- The sum the property folds over an array of widths is the sum of its list. -/
private theorem foldl_sum (a : Array ℕ) : a.foldl (· + ·) 0 = a.toList.sum := by
  rw [← Array.foldl_toList, List.sum_eq_foldl]

/-- A type has as many leaf widths as leaves. -/
theorem widths_size (t : Ty) : t.widths.size = t.leaves := by
  rw [← Array.length_toList]
  exact Flock.Sigma.widths_length t

/-- Copies of an array appended one after another, as a list. -/
private theorem foldl_copies (w : Array ℕ) (l : List ℕ) : ∀ acc : Array ℕ,
    (l.foldl (fun a _ => a ++ w) acc).toList = acc.toList ++ (List.replicate l.length w.toList).flatten := by
  induction l with
  | nil => intro acc; exact (List.append_nil _).symm
  | cons x l ih =>
    intro acc
    rw [List.foldl_cons, ih, Array.toList_append, List.length_cons, List.replicate_succ, List.flatten_cons,
      List.append_assoc]

/-- An array's leaf widths are its length's copies of its element's. -/
theorem widths_array (n : ℕ) (e : Ty) :
    (Ty.widths (.array n e)).toList = (List.replicate n e.widths.toList).flatten := by
  have h := foldl_copies e.widths (List.range n) #[]
  rw [List.length_range] at h
  simp only [Ty.widths]
  rw [← Array.foldl_toList, Array.toList_range]
  exact h.trans (List.nil_append _)

/-- Copies of a list have the copies' length. -/
private theorem copies_length (m : ℕ) (w : List ℕ) : (List.replicate m w).flatten.length = m * w.length := by
  induction m with
  | zero => rw [Nat.zero_mul]; rfl
  | succ m ih => rw [List.replicate_succ, List.flatten_cons, List.length_append, ih, Nat.succ_mul, Nat.add_comm]

/-- Copies of a list have the copies' sum. -/
private theorem copies_sum (m : ℕ) (w : List ℕ) : (List.replicate m w).flatten.sum = m * w.sum := by
  induction m with
  | zero => rw [Nat.zero_mul]; rfl
  | succ m ih => rw [List.replicate_succ, List.flatten_cons, List.sum_append, ih, Nat.succ_mul, Nat.add_comm]

/-- `a + b` copies of a list are `a` copies and then `b`. -/
private theorem copies_add (a b : ℕ) (w : List ℕ) :
    (List.replicate (a + b) w).flatten = (List.replicate a w).flatten ++ (List.replicate b w).flatten := by
  induction a with
  | zero => rw [Nat.zero_add]; rfl
  | succ a ih =>
    rw [Nat.succ_add, List.replicate_succ, List.flatten_cons, ih, List.replicate_succ, List.flatten_cons,
      List.append_assoc]

/-- Of `k · c` copies of a list, the leaves from `row · c` copies on, `c` copies long, are `c` copies. -/
private theorem copies_slice (k c row : ℕ) (w : List ℕ) (hrow : row < k) :
    ((List.replicate (k * c) w).flatten.extract (row * (c * w.length)) ((row + 1) * (c * w.length))).sum =
      c * w.sum := by
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_lt hrow
  have e : (row + j + 1) * c = row * c + (c + j * c) := by
    rw [Nat.add_right_comm, Nat.add_mul, Nat.add_mul, Nat.one_mul, Nat.add_assoc]
  have h1 : ((List.replicate (row * c) w).flatten).length = row * (c * w.length) := by
    rw [copies_length, Nat.mul_assoc]
  have h2 : ((List.replicate c w).flatten).length = (row + 1) * (c * w.length) - row * (c * w.length) := by
    rw [copies_length, Nat.succ_mul, Nat.add_sub_cancel_left]
  rw [Nat.mul_comm (row + j + 1) c, Nat.mul_comm c (row + j + 1), e, copies_add, copies_add,
    List.extract_eq_take_drop, List.drop_left' h1, List.take_left' h2, copies_sum]

/-- The leaves `[row · (leaves / k), (row + 1) · (leaves / k))` of a type whose outer length `k` divides have the
type's bits over `k`. -/
private theorem slice_bits (t : Ty) (k row : ℕ) (ho : outerOf t % k = 0) (hrow : row < k) :
    (t.widths.extract (row * (t.leaves / k)) ((row + 1) * (t.leaves / k))).foldl (· + ·) 0 = bitsOf t / k := by
  have hk : 0 < k := by omega
  have whole : outerOf t = 1 →
      (t.widths.extract (row * (t.leaves / k)) ((row + 1) * (t.leaves / k))).foldl (· + ·) 0 = bitsOf t / k := by
    intro h1
    rw [h1] at ho
    have hk1 : k = 1 := by
      rcases Nat.lt_or_ge 1 k with h | h
      · rw [Nat.mod_eq_of_lt h] at ho
        omega
      · omega
    subst hk1
    have hr : row = 0 := by omega
    subst hr
    rw [Nat.zero_mul, Nat.zero_add, Nat.one_mul, Nat.div_one, Nat.div_one,
      Array.extract_eq_self_of_le (Nat.le_of_eq (widths_size t))]
    rfl
  cases t with
  | value w => exact whole rfl
  | tuple fs => exact whole rfl
  | record fs => exact whole rfl
  | array n e =>
    obtain ⟨c, rfl⟩ := Nat.dvd_of_mod_eq_zero ho
    have hl : (Ty.array (k * c) e).leaves = k * c * e.leaves := by simp only [Ty.leaves]
    have hq : k * c * e.leaves / k = c * e.widths.toList.length := by
      rw [Nat.mul_assoc, Nat.mul_div_cancel_left _ hk, Array.length_toList, widths_size]
    have hb : bitsOf (Ty.array (k * c) e) = k * (c * e.widths.toList.sum) := by
      unfold bitsOf
      rw [foldl_sum, widths_array, copies_sum, Nat.mul_assoc]
    rw [foldl_sum, Array.toList_extract, widths_array, hl, hq, copies_slice k c row _ hrow, hb,
      Nat.mul_div_cancel_left _ hk]

/-- A row of a parameter cut `k` ways has the parameter's leaves over `k`. -/
theorem inRow_size (d : Defn) (i k row : ℕ) (hrow : row < k) :
    (inRow d i k row).2.size = d.params[i]!.2.leaves / k := by
  unfold inRow
  rw [Array.size_extract, widths_size]
  generalize d.params[i]!.2.leaves = L
  have h1 := Nat.mul_div_le L k
  have h2 := Nat.mul_le_mul_right (L / k) (show row + 1 ≤ k from hrow)
  rw [Nat.add_one_mul] at h2 ⊢
  generalize L / k = q at h1 h2 ⊢
  generalize row * q = a at h2 ⊢
  generalize k * q = b at h1 h2
  omega

/-- A row of a parameter cut `k` ways, `k` dividing its outer length, has the parameter's bits over `k`. -/
theorem inRow_bits (d : Defn) (i k row : ℕ) (ho : outerOf d.params[i]!.2 % k = 0) (hrow : row < k) :
    (inRow d i k row).2.foldl (· + ·) 0 = bitsOf d.params[i]!.2 / k :=
  slice_bits d.params[i]!.2 k row ho hrow

/-- An output row's name says which field it is. -/
theorem outName_inj (f f' : ℕ) (h : "o" ++ toString f = "o" ++ toString f') : f = f' :=
  Properties.Spot.Credited.toString_inj ((String.append_right_inj "o").1 h)

/-! ## A template's units -/

/-- The units of template `tid` in the root nodes before the `j`-th range. -/
def unitsBefore (rs : Array TRange) (tid : String) (j : ℕ) : ℕ :=
  ((rs.extract 0 j).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0

/-- The units before the next range: one more node's members where the node is of the template. -/
theorem unitsBefore_succ (rs : Array TRange) (tid : String) (j : ℕ) (hj : j < rs.size) :
    unitsBefore rs tid (j + 1) =
      unitsBefore rs tid j + if (rs[j]!.template == tid) = true then rs[j]!.count else 0 := by
  unfold unitsBefore
  rw [Array.extract_succ_right (Nat.succ_pos j) hj, Array.filter_push, getElem!_pos rs j hj]
  by_cases h : (rs[j].template == tid) = true
  · rw [ite_eq_left h, ite_eq_left h, Array.foldl_push]
  · rw [ite_eq_right h, ite_eq_right h, Nat.add_zero]

/-- A later range has no fewer units before it. -/
theorem unitsBefore_le (rs : Array TRange) (tid : String) (j d : ℕ) (h : j + d ≤ rs.size) :
    unitsBefore rs tid j ≤ unitsBefore rs tid (j + d) := by
  induction d with
  | zero => exact Nat.le_refl _
  | succ d ih =>
    have e := unitsBefore_succ rs tid (j + d) (by omega)
    have h1 := ih (by omega)
    rw [← Nat.add_assoc, e]
    omega

/-- A range of the template lies, with all its members, before every later range. -/
theorem unitsBefore_lt (rs : Array TRange) (tid : String) (i i' : ℕ) (h : i < i') (hi' : i' ≤ rs.size)
    (ht : rs[i]!.template = tid) : unitsBefore rs tid i + rs[i]!.count ≤ unitsBefore rs tid i' := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le (show i + 1 ≤ i' from h)
  have e := unitsBefore_succ rs tid i (by omega)
  have h1 := unitsBefore_le rs tid (i + 1) d hi'
  rw [ite_eq_left (beq_iff_eq.2 ht)] at e
  omega

/-- A template's unit is one activation. -/
theorem unitAt_fun (P : Prog) (rs : Array TRange) (tid : String) (u : ℕ) (sc sc' : Scope)
    (h : UnitAt P rs tid u sc) (h' : UnitAt P rs tid u sc') : sc = sc' := by
  obtain ⟨i, m, hi, ht, hm, hu, rfl⟩ := h
  obtain ⟨i', m', hi', ht', hm', hu', rfl⟩ := h'
  have hu1 : u = unitsBefore rs tid i + m := hu
  have hu2 : u = unitsBefore rs tid i' + m' := hu'
  have hii : i = i' := by
    rcases Nat.lt_trichotomy i i' with hlt | heq | hgt
    · have := unitsBefore_lt rs tid i i' hlt (Nat.le_of_lt hi') ht
      omega
    · exact heq
    · have := unitsBefore_lt rs tid i' i hgt (Nat.le_of_lt hi) ht'
      omega
  subst hii
  have hmm : m = m' := by omega
  subst hmm
  rfl

end NciCheck.Properties.Record.SecurityProofs
