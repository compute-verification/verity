import Mathlib
import Definitions.Core.NaiveStrassen

/-!
# Integer bilinear schemes against P1's floor: what is proved of "every scheme loses"

A part of the SAK lower bound (`Pouw.SakLb`), over D4's model (`Definitions.Core.NaiveStrassen`): exact int8
products priced by the tensor units alone, additions free.

* **The cover inequality** (`cover`): every integer bilinear scheme for an `m × k` by `k × n` product (Brent's
  equations, `IntScheme.Brent`) has `Σ_t ‖u_t‖₁ ‖v_t‖₁ ≥ m k n`: each of the `m k n` monomials `A_ij B_jl` is made by
  some product, and a product makes at most `‖u_t‖₁ ‖v_t‖₁` of them. So a scheme whose products are all narrow
  (`‖u_t‖₁, ‖v_t‖₁ ≤ 1`) has at least `m k n` products (`narrow_rank`).
* **P1's floor on a product** (`floor_max`, `floor_wide`): on every one of D4's four atom lists, a depth-0 schedule
  for operands over `a` and `b` values costs at least `log₂ max(a, b)` eighths of an s8 MAC, and at least 16 once one
  operand takes more than 256 values and the other at least 256. Digit splits never help: every atom holds at most
  `2^weight8` values per operand (`CapLog`), and every atom past 256 values weighs at least 16 (`WideHeavy`).
* **One level of any 2 × 2 scheme** (`scheme_floor`): a scheme node over int8 operands whose products are then done
  by atoms costs at least `8 (R + R_wide)` eighths, `R` its nonzero products and `R_wide` those with a combination of
  more than one entry on one side: `κ₁ ≥ (R + R_wide) / (m k n)` of naive.

What is not proved, and why it is open rather than hard to write: the floor grows like the logarithm of an operand's
range while the cover only bounds `Σ ‖u_t‖₁ ‖v_t‖₁`, so a scheme of small rank with moderately wide products is not
excluded by these inequalities; excluding it at the served shapes is a rank lower bound for matrix multiplication
at those sizes. `Pouw.SakLb.SAKResidual` carries it.
-/

namespace Verity.Accel.Bilinear

open Finset Verity.Accel.NaiveStrassen

section Cover

/-- **An integer bilinear scheme** for an `m × k` by `k × n` product with its products indexed by `ι`: product `t`
multiplies `Σ u t i j A_ij` by `Σ v t j l B_jl`, and `C_il = Σ_t w t i l M_t`. Its rank is `|ι|`. -/
structure IntScheme (ι : Type) (m k n : ℕ) where
  u : ι → Fin m → Fin k → ℤ
  v : ι → Fin k → Fin n → ℤ
  w : ι → Fin m → Fin n → ℤ

namespace IntScheme

variable {ι : Type} [Fintype ι] {m k n : ℕ} (S : IntScheme ι m k n)

/-- **Brent's equations**: the scheme computes `C = A B` for every `A` and `B` over every commutative ring. -/
def Brent : Prop :=
  ∀ (i : Fin m) (j : Fin k) (j' : Fin k) (l : Fin n) (i' : Fin m) (l' : Fin n),
    ∑ t, S.u t i j * S.v t j' l * S.w t i' l' = if i = i' ∧ j = j' ∧ l = l' then 1 else 0

/-- `‖u_t‖₁`. -/
def nu (t : ι) : ℕ := ∑ i, ∑ j, (S.u t i j).natAbs

/-- `‖v_t‖₁`. -/
def nv (t : ι) : ℕ := ∑ j, ∑ l, (S.v t j l).natAbs

/-- Each monomial `A_ij B_jl` is made by some product with weight at least 1. -/
theorem monomial_covered (hB : S.Brent) (i : Fin m) (j : Fin k) (l : Fin n) :
    1 ≤ ∑ t, (S.u t i j).natAbs * (S.v t j l).natAbs := by
  have h := hB i j j l i l
  simp only [and_self, ite_true] at h
  by_contra hlt
  have h0 : ∀ t, (S.u t i j).natAbs * (S.v t j l).natAbs = 0 := by
    intro t
    have := Finset.single_le_sum (f := fun t => (S.u t i j).natAbs * (S.v t j l).natAbs)
      (fun _ _ => Nat.zero_le _) (mem_univ t)
    omega
  have : ∑ t, S.u t i j * S.v t j l * S.w t i l = 0 := by
    apply Finset.sum_eq_zero
    intro t _
    rcases Nat.mul_eq_zero.mp (h0 t) with h1 | h1
    · rw [Int.natAbs_eq_zero.mp h1]; ring
    · rw [Int.natAbs_eq_zero.mp h1]; ring
  rw [this] at h
  exact zero_ne_one h

/-- **The cover inequality**: `Σ_t ‖u_t‖₁ ‖v_t‖₁ ≥ m k n`. -/
theorem cover (hB : S.Brent) : m * k * n ≤ ∑ t, S.nu t * S.nv t := by
  calc m * k * n = ∑ _i : Fin m, ∑ _j : Fin k, ∑ _l : Fin n, (1 : ℕ) := by simp [mul_assoc]
    _ ≤ ∑ i, ∑ j, ∑ l, ∑ t, (S.u t i j).natAbs * (S.v t j l).natAbs :=
        sum_le_sum fun i _ => sum_le_sum fun j _ => sum_le_sum fun l _ => S.monomial_covered hB i j l
    _ = ∑ i, ∑ j, ∑ t, ∑ l, (S.u t i j).natAbs * (S.v t j l).natAbs :=
        sum_congr rfl fun i _ => sum_congr rfl fun j _ => Finset.sum_comm
    _ = ∑ i, ∑ t, ∑ j, ∑ l, (S.u t i j).natAbs * (S.v t j l).natAbs :=
        sum_congr rfl fun i _ => Finset.sum_comm
    _ = ∑ t, ∑ i, ∑ j, ∑ l, (S.u t i j).natAbs * (S.v t j l).natAbs := Finset.sum_comm
    _ ≤ ∑ t, S.nu t * S.nv t := by
        refine sum_le_sum fun t _ => ?_
        unfold nu nv
        rw [Finset.sum_mul]
        refine sum_le_sum fun i _ => ?_
        rw [Finset.sum_mul]
        refine sum_le_sum fun j _ => ?_
        calc ∑ l, (S.u t i j).natAbs * (S.v t j l).natAbs
            = (S.u t i j).natAbs * ∑ l, (S.v t j l).natAbs := by rw [Finset.mul_sum]
          _ ≤ ∑ j', (S.u t i j).natAbs * ∑ l, (S.v t j' l).natAbs :=
              Finset.single_le_sum (f := fun j' => (S.u t i j).natAbs * ∑ l, (S.v t j' l).natAbs)
                (fun _ _ => Nat.zero_le _) (mem_univ j)
          _ = (S.u t i j).natAbs * ∑ j', ∑ l, (S.v t j' l).natAbs := by rw [Finset.mul_sum]

/-- **A scheme of narrow products has at least `m k n` of them**: with `‖u_t‖₁, ‖v_t‖₁ ≤ 1` for every product, the
cover gives rank `≥ m k n`, naive's count. -/
theorem narrow_rank (hB : S.Brent) (hn : ∀ t, S.nu t ≤ 1 ∧ S.nv t ≤ 1) : m * k * n ≤ Fintype.card ι := by
  calc m * k * n ≤ ∑ t, S.nu t * S.nv t := S.cover hB
    _ ≤ ∑ _t : ι, 1 := sum_le_sum fun t _ => by
        have := hn t; exact (Nat.mul_le_mul this.1 this.2).trans (le_of_eq (one_mul 1))
    _ = Fintype.card ι := by simp

end IntScheme

/-- **Honest**: naive's scheme, one product for each monomial `A_ij B_jl`. -/
def naive (m k n : ℕ) : IntScheme (Fin m × Fin k × Fin n) m k n where
  u t i j := if t.1 = i ∧ t.2.1 = j then 1 else 0
  v t j l := if t.2.1 = j ∧ t.2.2 = l then 1 else 0
  w t i l := if t.1 = i ∧ t.2.2 = l then 1 else 0

theorem naive_brent (m k n : ℕ) : (naive m k n).Brent := by
  intro i j j' l i' l'
  simp only [naive, Fintype.sum_prod_type]
  by_cases h : i = i' ∧ j = j' ∧ l = l'
  · obtain ⟨rfl, rfl, rfl⟩ := h
    simp [ite_and, Finset.sum_ite_eq']
  · simp only [h, ite_false]
    apply Finset.sum_eq_zero; intro a _
    apply Finset.sum_eq_zero; intro b _
    apply Finset.sum_eq_zero; intro c _
    by_cases h1 : a = i ∧ b = j
    · by_cases h2 : b = j' ∧ c = l
      · by_cases h3 : a = i' ∧ c = l'
        · exact absurd ⟨h1.1 ▸ h3.1, h1.2 ▸ h2.1, h2.2 ▸ h3.2⟩ h
        · simp [h3]
      · simp [h2]
    · simp [h1]

/-- **Nontrivial**: Strassen's 2 × 2 scheme, seven products, satisfies Brent's equations, so the cover inequality is
no rank bound: narrowness is what `narrow_rank` needs. -/
def strassen2 : IntScheme (Fin 7) 2 2 2 where
  u t i j := (strassen.u[t.val]!)[2 * i.val + j.val]!
  v t j l := (strassen.v[t.val]!)[2 * j.val + l.val]!
  w t i l := (strassen.w[2 * i.val + l.val]!)[t.val]!

theorem strassen2_brent : strassen2.Brent := by
  unfold IntScheme.Brent strassen2
  decide

theorem strassen2_rank : Fintype.card (Fin 7) < 2 * 2 * 2 := by decide


end Cover

section Floor

/-- Every atom holds at most `2^weight8` values per operand. -/
def CapLog (fs : List Atom) : Prop := ∀ f ∈ fs, f.cap ≤ 2 ^ f.weight8

/-- Every atom that holds more than 256 values per operand weighs at least 16 eighths. -/
def WideHeavy (fs : List Atom) : Prop := ∀ f ∈ fs, 257 ≤ f.cap → 16 ≤ f.weight8

theorem capLog_instances : ∀ fs ∈ instances, CapLog fs := by
  intro fs hfs f hf
  simp only [instances, List.mem_cons, List.not_mem_nil, or_false] at hfs
  rcases hfs with rfl | rfl | rfl | rfl <;>
    simp only [h200Sheet, h200Measured, rtx6000Sheet, rtx6000Measured, s8, b1, List.mem_cons,
      List.not_mem_nil, or_false] at hf <;>
    rcases hf with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem wideHeavy_instances : ∀ fs ∈ instances, WideHeavy fs := by
  intro fs hfs f hf
  simp only [instances, List.mem_cons, List.not_mem_nil, or_false] at hfs
  rcases hfs with rfl | rfl | rfl | rfl <;>
    simp only [h200Sheet, h200Measured, rtx6000Sheet, rtx6000Measured, s8, b1, List.mem_cons,
      List.not_mem_nil, or_false] at hf <;>
    rcases hf with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem ceil_mul_ge (a d : ℕ) (hd : 0 < d) : a ≤ d * ((a + d - 1) / d) := by
  have := Nat.div_add_mod (a + d - 1) d
  have := Nat.mod_lt (a + d - 1) hd
  rcases Nat.eq_zero_or_pos a with rfl | ha
  · exact Nat.zero_le _
  · nlinarith [Nat.sub_add_cancel (show 1 ≤ a + d by omega)]

theorem atom_eq {f f' : Atom} (h : (f == f') = true) : f = f' := by
  cases f; cases f'
  simp only [BEq.beq] at h
  unfold Verity.Accel.NaiveStrassen.instBEqAtom.beq at h
  simp_all

theorem contains_mem {fs : List Atom} {f : Atom} (h : fs.contains f = true) : f ∈ fs := by
  induction fs with
  | nil => simp at h
  | cons x xs ih =>
    simp only [List.contains_cons, Bool.or_eq_true] at h
    rcases h with h | h
    · exact atom_eq h ▸ List.mem_cons_self
    · exact List.mem_cons_of_mem _ (ih h)

/-- **P1's floor, by range**: a depth-0 schedule (atoms and digit splits) for operands over `a` and `b` values costs
at least `log₂ max(a, b)`: `a, b ≤ 2^cost`. -/
theorem floor_max {fs : List Atom} (hfs : CapLog fs) (ss : List Scheme) (m : ℕ) :
    ∀ (S : Sched) (a b : ℕ), S.valid fs ss m 0 a b = true → a ≤ 2 ^ S.cost 0 ∧ b ≤ 2 ^ S.cost 0 := by
  intro S
  induction S with
  | atom f =>
    intro a b h
    simp only [Sched.valid, Bool.and_eq_true, Atom.holds, decide_eq_true_eq] at h
    obtain ⟨hf, ⟨ha, hb⟩, -⟩ := h
    have hc := hfs f (contains_mem hf)
    simp only [Sched.cost, pow_zero, one_mul]
    exact ⟨ha.trans hc, hb.trans hc⟩
  | splitA d x y ihx ihy =>
    intro a b h
    simp only [Sched.valid, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨hd2, hda⟩, hx⟩, hy⟩ := h
    obtain ⟨hx1, hx2⟩ := ihx _ _ hx
    obtain ⟨hy1, -⟩ := ihy _ _ hy
    simp only [Sched.cost, pow_add]
    refine ⟨(ceil_mul_ge a d (by omega)).trans (Nat.mul_le_mul hx1 hy1), ?_⟩
    exact hx2.trans (Nat.le_mul_of_pos_right _ (Nat.two_pow_pos _))
  | splitB d x y ihx ihy =>
    intro a b h
    simp only [Sched.valid, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨hd2, hdb⟩, hx⟩, hy⟩ := h
    obtain ⟨hx1, hx2⟩ := ihx _ _ hx
    obtain ⟨-, hy2⟩ := ihy _ _ hy
    simp only [Sched.cost, pow_add]
    refine ⟨hx1.trans (Nat.le_mul_of_pos_right _ (Nat.two_pow_pos _)), ?_⟩
    exact (ceil_mul_ge b d (by omega)).trans (Nat.mul_le_mul hx2 hy2)
  | scheme s kids _ =>
    intro a b h
    simp [Sched.valid] at h

theorem le_cost_of_pow {c x : ℕ} (e : ℕ) (h : x ≤ 2 ^ c) (hx : 2 ^ e < x) : e + 1 ≤ c := by
  by_contra hc
  have : 2 ^ c ≤ 2 ^ e := Nat.pow_le_pow_right (by norm_num) (by omega)
  omega

/-- **P1's floor on a wide product**: one operand over more than 256 values and the other over at least 256 cost
at least 16 eighths, twice an s8 MAC. -/
theorem floor_wide {fs : List Atom} (hfs : CapLog fs) (hw : WideHeavy fs) (ss : List Scheme) (m : ℕ) (S : Sched)
    (a b : ℕ) (hv : S.valid fs ss m 0 a b = true) (hab : (257 ≤ a ∧ 256 ≤ b) ∨ (256 ≤ a ∧ 257 ≤ b)) :
    16 ≤ S.cost 0 := by
  cases S with
  | atom f =>
    simp only [Sched.valid, Bool.and_eq_true, Atom.holds, decide_eq_true_eq] at hv
    obtain ⟨hf, ⟨ha, hb⟩, -⟩ := hv
    simp only [Sched.cost, pow_zero, one_mul]
    exact hw f (contains_mem hf) (by omega)
  | splitA d x y =>
    simp only [Sched.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
    obtain ⟨⟨⟨hd2, hda⟩, hx⟩, hy⟩ := hv
    have h1 := le_cost_of_pow 7 (floor_max hfs ss m x _ _ hx).2 (by norm_num; omega)
    have h2 := le_cost_of_pow 7 (floor_max hfs ss m y _ _ hy).2 (by norm_num; omega)
    simp only [Sched.cost]; omega
  | splitB d x y =>
    simp only [Sched.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
    obtain ⟨⟨⟨hd2, hdb⟩, hx⟩, hy⟩ := hv
    have h1 := le_cost_of_pow 7 (floor_max hfs ss m x _ _ hx).1 (by norm_num; omega)
    have h2 := le_cost_of_pow 7 (floor_max hfs ss m y _ _ hy).1 (by norm_num; omega)
    simp only [Sched.cost]; omega
  | scheme s kids => simp [Sched.valid] at hv

/-- P1's floor on a product of one-sided width: `ν_A, ν_B` the `ℓ₁` norms of its combinations of int8 entries
(each over `255 ν + 1` values): 16 when one is at least 2 and the other at least 1, else 8 when one is at least 1. -/
def floorNu (νa νb : ℕ) : ℕ :=
  if (2 ≤ νa ∧ 1 ≤ νb) ∨ (1 ≤ νa ∧ 2 ≤ νb) then 16 else if 1 ≤ νa ∨ 1 ≤ νb then 8 else 0

theorem floor_nu {fs : List Atom} (hfs : CapLog fs) (hw : WideHeavy fs) (ss : List Scheme) (S : Sched)
    (νa νb : ℕ) (hv : S.valid fs ss 0 0 (NaiveStrassen.range 0 256 νa) (NaiveStrassen.range 0 256 νb) = true) :
    floorNu νa νb ≤ S.cost 0 := by
  have ra : NaiveStrassen.range 0 256 νa = 255 * νa + 1 := by simp [NaiveStrassen.range]
  have rb : NaiveStrassen.range 0 256 νb = 255 * νb + 1 := by simp [NaiveStrassen.range]
  rw [ra, rb] at hv
  unfold floorNu
  split_ifs with h1 h2
  · exact floor_wide hfs hw ss 0 S _ _ hv (by omega)
  · obtain ⟨ha, hb⟩ := floor_max hfs ss 0 S _ _ hv
    rcases h2 with h2 | h2
    · exact le_cost_of_pow 7 ha (by norm_num; omega)
    · exact le_cost_of_pow 7 hb (by norm_num; omega)
  · exact Nat.zero_le _

theorem foldl_add_eq (xs : List ℕ) (acc : ℕ) : xs.foldl (· + ·) acc = acc + xs.sum := by
  induction xs generalizing acc with
  | nil => simp
  | cons x xs ih => simp only [List.foldl, ih, List.sum_cons, Nat.add_assoc]

/-- **One level of any 2 × 2 scheme, then atoms**: a scheme node over int8 operands costs at least the sum of its
products' floors, `Σ_t floorNu ‖u_t‖₁ ‖v_t‖₁`, so at least `8 (R + R_wide)` eighths. -/
theorem scheme_floor {fs : List Atom} (hfs : CapLog fs) (hw : WideHeavy fs) (ss : List Scheme) (s : Scheme)
    (kids : Fin s.R → Sched) (hv : (Sched.scheme s kids).valid fs ss 0 1 256 256 = true) :
    ∑ t : Fin s.R, floorNu (l1 s.u[t.val]!) (l1 s.v[t.val]!) ≤ (Sched.scheme s kids).cost 1 := by
  simp only [Sched.valid, Bool.and_eq_true, List.all_eq_true, List.mem_finRange, true_implies] at hv
  simp only [Sched.cost, Nat.sub_self, foldl_add_eq, zero_add]
  rw [← Fin.sum_univ_def]
  exact sum_le_sum fun t _ => floor_nu hfs hw ss (kids t) _ _ (hv.2 t)

end Floor

end Verity.Accel.Bilinear
