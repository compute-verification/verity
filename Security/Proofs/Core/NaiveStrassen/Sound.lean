import Proofs.Core.NaiveStrassen.Check

/-!
# Soundness of the tables (NaiveStrassen)

`Facts` lists, pointwise, what the check establishes about a family of tables `lo l i j` and `has l i j`. `Facts.core`
proves from it, by induction on the schedule, that `lo` bounds every schedule's cost and `has` every fast schedule's,
at any grid cells below the operand ranges.
-/

namespace Verity.Accel.NaiveStrassen.Packed

open Verity.Accel.NaiveStrassen

/-! ## Sums -/

theorem foldl_add (xs : List Nat) (acc : Nat) : xs.foldl (· + ·) acc = acc + xs.sum := by
  induction xs generalizing acc with
  | nil => simp
  | cons x xs ih => simp only [List.foldl, ih, List.sum_cons, Nat.add_assoc]

theorem sum_map_le {α : Type} {f h : α → Nat} : ∀ {xs : List α}, (∀ x ∈ xs, f x ≤ h x) →
    (xs.map f).sum ≤ (xs.map h).sum
  | [], _ => by simp
  | x :: xs, hx => by
    simp only [List.map_cons, List.sum_cons]
    exact Nat.add_le_add (hx x List.mem_cons_self) (sum_map_le fun y hy => hx y (List.mem_cons_of_mem _ hy))

theorem mul_le_sum {c : Nat} : ∀ {xs : List Nat}, (∀ x ∈ xs, c ≤ x) → xs.length * c ≤ xs.sum
  | [], _ => by simp
  | x :: xs, hx => by
    have h1 := mul_le_sum (c := c) (xs := xs) fun y hy => hx y (List.mem_cons_of_mem _ hy)
    have h2 := hx x List.mem_cons_self
    simp only [List.length_cons, List.sum_cons, Nat.succ_mul]
    omega

theorem pred_mul_add_le_sum {c y : Nat} : ∀ {xs : List Nat}, (∀ x ∈ xs, c ≤ x) → y ∈ xs →
    (xs.length - 1) * c + y ≤ xs.sum
  | [], _, hy => absurd hy List.not_mem_nil
  | x :: xs, hx, hy => by
    simp only [List.length_cons, Nat.add_sub_cancel, List.sum_cons]
    have hxs := mul_le_sum (c := c) (xs := xs) fun z hz => hx z (List.mem_cons_of_mem _ hz)
    rcases List.mem_cons.mp hy with rfl | hy
    · omega
    · have h1 := pred_mul_add_le_sum (xs := xs) (fun z hz => hx z (List.mem_cons_of_mem _ hz)) hy
      have h2 := hx x List.mem_cons_self
      have h3 : xs.length * c ≤ (xs.length - 1) * c + c := by
        rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right c (by omega)
      omega

/-! ## Ceiling division -/

theorem ceil_mono {x y d D : Nat} (hd : 0 < d) (hdD : d ≤ D) (hxy : x ≤ y) :
    (x + D - 1) / D ≤ (y + d - 1) / d := by
  have hD : 0 < D := Nat.lt_of_lt_of_le hd hdD
  have h1 := Nat.div_add_mod (y + d - 1) d
  have h2 := Nat.mod_lt (y + d - 1) hd
  have h3 : d * ((y + d - 1) / d) ≤ D * ((y + d - 1) / d) := Nat.mul_le_mul_right _ hdD
  have h4 : (x + D - 1) / D < (y + d - 1) / d + 1 := by
    rw [Nat.div_lt_iff_lt_mul hD]
    have : ((y + d - 1) / d + 1) * D = D * ((y + d - 1) / d) + D := by rw [Nat.succ_mul, Nat.mul_comm]
    omega
  omega

/-! ## Atoms and schemes -/

theorem atom_eq {f f' : Atom} (h : (f == f') = true) : f = f' := by
  cases f; cases f'
  simp only [BEq.beq] at h
  unfold Verity.Accel.NaiveStrassen.instBEqAtom.beq at h
  simp_all

theorem scheme_eq {s s' : Scheme} (h : (s == s') = true) : s = s' := by
  cases s; cases s'
  simp only [BEq.beq] at h
  unfold Verity.Accel.NaiveStrassen.instBEqScheme.beq at h
  simp_all

theorem mem_of_contains {fs : List Atom} {f : Atom} (h : fs.contains f = true) : f ∈ fs := by
  induction fs with
  | nil => simp at h
  | cons x xs ih =>
    simp only [List.contains_cons, Bool.or_eq_true] at h
    rcases h with h | h
    · exact atom_eq h ▸ List.mem_cons_self
    · exact List.mem_cons_of_mem _ (ih h)

theorem scheme_cases {s : Scheme} (h : schemes.contains s = true) :
    s = strassen ∨ s = winograd ∨ s = Verity.Accel.NaiveStrassen.trivial := by
  simp only [schemes, List.contains_cons, List.contains_nil, Bool.or_false, Bool.or_eq_true] at h
  rcases h with h | h | h
  · exact Or.inl (scheme_eq h)
  · exact Or.inr (Or.inl (scheme_eq h))
  · exact Or.inr (Or.inr (scheme_eq h))

theorem holds_mono {f : Atom} {a b a' b' : Nat} (h : f.holds a b = true) (ha : a' ≤ a) (hb : b' ≤ b) :
    f.holds a' b' = true := by
  rcases f with ⟨w, c, _ | ⟨k, x⟩⟩ <;> simp only [Atom.holds, Bool.and_eq_true, decide_eq_true_eq] at h ⊢
  · exact ⟨⟨by omega, by omega⟩, h.2⟩
  · exact ⟨⟨by omega, by omega⟩, Nat.le_trans
      (Nat.mul_le_mul (Nat.mul_le_mul_left k (Nat.div_le_div_right ha)) (Nat.div_le_div_right hb)) h.2⟩

theorem norms_fast {s : Scheme} (hs : s = strassen ∨ s = winograd) :
    ∀ t : Fin s.R, 1 ≤ l1 s.u[t.val]! ∧ l1 s.u[t.val]! ≤ 4 ∧ 1 ≤ l1 s.v[t.val]! ∧ l1 s.v[t.val]! ≤ 4 := by
  rcases hs with rfl | rfl <;> decide

theorem norms_trivial :
    ∀ t : Fin Verity.Accel.NaiveStrassen.trivial.R,
      l1 Verity.Accel.NaiveStrassen.trivial.u[t.val]! = 1 ∧ l1 Verity.Accel.NaiveStrassen.trivial.v[t.val]! = 1 := by
  decide

theorem range_zero (a ν : Nat) : range 0 a ν = (a - 1) * ν + 1 := by simp [range]

/-- Every atom of `fs` that holds is at least the RTX PRO 6000 sheet's table entry. -/
def Dom (fs : List Atom) : Prop :=
  ∀ f ∈ fs, ∀ a b, f.holds a b = true → atomMin rtx6000Sheet a b ≤ f.weight8

/-! ## The facts and the induction -/

/-- What the check establishes about tables `lo` and `has` up to level `Lmax` on a grid `g` of `G` cells. -/
structure Facts (G Lmax : Nat) (g : Nat → Nat) (kc rest : Nat → Nat → Nat) (lo has : Nat → Nat → Nat → Nat) :
    Prop where
  g0 : g 0 = 1
  g1 : g 1 = 2
  two_le : 2 ≤ G
  mono : ∀ k, k + 1 < G → g k < g (k + 1)
  kc_le : ∀ i, i < G → ∀ ν, 1 ≤ ν → ν ≤ 4 → g (kc i ν) ≤ (g i - 1) * ν + 1
  rest_le : ∀ i, i < G → ∀ e, 1 ≤ e → e < i → g (rest i e) ≤ (g i + (g (e + 1) - 1) - 1) / (g (e + 1) - 1)
  out : ∀ l, l ≤ Lmax → ∀ i j, G ≤ i → lo l i j = 0 ∧ has l i j = 0
  symm : ∀ l, l ≤ Lmax → ∀ i j, lo l i j = lo l j i ∧ has l i j = has l j i
  atom : ∀ l, l ≤ Lmax → ∀ i j, lo l i j ≤ 8 ^ l * atomMin rtx6000Sheet (g i) (g j)
  split : ∀ l, l ≤ Lmax → ∀ i j e, 1 ≤ e → e < i →
    lo l i j ≤ lo l e j + lo l (rest i e) j ∧ has l i j ≤ has l e j + lo l (rest i e) j ∧
      has l i j ≤ lo l e j + has l (rest i e) j
  own : ∀ l, l ≤ Lmax → ∀ i j, has l i j ≤ lo l i j + has l 1 j
  triv : ∀ l, l + 1 ≤ Lmax → ∀ i j, lo (l + 1) i j ≤ 8 * lo l i j ∧ has (l + 1) i j ≤ 7 * lo l i j + has l i j
  sch : ∀ l, l + 1 ≤ Lmax → ∀ s, (s = strassen ∨ s = winograd) → ∀ i j,
    lo (l + 1) i j ≤ ((List.finRange s.R).map fun t => lo l (kc i (l1 s.u[t.val]!)) (kc j (l1 s.v[t.val]!))).sum ∧
      has (l + 1) i j ≤ ((List.finRange s.R).map fun t => lo l (kc i (l1 s.u[t.val]!)) (kc j (l1 s.v[t.val]!))).sum

section
variable {G Lmax : Nat} {g : Nat → Nat} {kc rest : Nat → Nat → Nat} {lo has : Nat → Nat → Nat → Nat}

theorem maxCell_aux (g : Nat → Nat) (G v : Nat) : ∀ n k, G - k = n → k < G → g k ≤ v →
    ∃ e, k ≤ e ∧ e < G ∧ g e ≤ v ∧ (e + 1 < G → v < g (e + 1))
  | 0, k, hn, hk, _ => absurd hk (by omega)
  | n + 1, k, hn, hk, hgk => by
    by_cases h : k + 1 < G ∧ g (k + 1) ≤ v
    · obtain ⟨e, h1, h2, h3, h4⟩ := maxCell_aux g G v n (k + 1) (by omega) h.1 h.2
      exact ⟨e, by omega, h2, h3, h4⟩
    · exact ⟨k, Nat.le_refl k, hk, hgk, fun hk1 => Nat.lt_of_not_le fun h' => h ⟨hk1, h'⟩⟩

theorem Facts.g_mono (F : Facts G Lmax g kc rest lo has) : ∀ {i e : Nat}, i ≤ e → e < G → g i ≤ g e
  | i, 0, h, _ => by rw [Nat.le_zero.mp h]; exact Nat.le_refl _
  | i, e + 1, h, he => by
    rcases Nat.lt_or_eq_of_le h with h | h
    · exact Nat.le_trans (F.g_mono (Nat.le_of_lt_succ h) (by omega)) (Nat.le_of_lt (F.mono e he))
    · rw [h]; exact Nat.le_refl _

theorem Facts.zero_right (F : Facts G Lmax g kc rest lo has) {l : Nat} (hl : l ≤ Lmax) (i : Nat) {j : Nat}
    (hj : G ≤ j) : lo l i j = 0 ∧ has l i j = 0 := by
  rw [(F.symm l hl i j).1, (F.symm l hl i j).2]; exact F.out l hl j i hj

/-- A digit split of the row operand at `d`, given bounds on its two subtrees at every cell they cover. -/
theorem Facts.split_bound (F : Facts G Lmax g kc rest lo has) {l : Nat} (hl : l ≤ Lmax) {d A c1 c2 : Nat}
    {fx fy : Bool} (hd : 2 ≤ d) (hdA : d < A) {i : Nat} (hi : g i ≤ A) (j : Nat)
    (hx : ∀ e, g e ≤ d → lo l e j ≤ c1 ∧ (fx = true → has l e j ≤ c1))
    (hy : ∀ r, g r ≤ (A + d - 1) / d → lo l r j ≤ c2 ∧ (fy = true → has l r j ≤ c2)) :
    lo l i j ≤ c1 + c2 ∧ ((fx || fy) = true → has l i j ≤ c1 + c2) := by
  by_cases hG : G ≤ i
  · have h0 := F.out l hl i j hG
    rw [h0.1, h0.2]; exact ⟨Nat.zero_le _, fun _ => Nat.zero_le _⟩
  have hiG : i < G := Nat.lt_of_not_le hG
  have hq : 2 ≤ (A + d - 1) / d := by
    rw [Nat.le_div_iff_mul_le (by omega)]; omega
  by_cases hid : g i ≤ d
  · obtain ⟨h1, h2⟩ := hx i hid
    refine ⟨Nat.le_trans h1 (Nat.le_add_right _ _), fun hf => ?_⟩
    rcases Bool.or_eq_true_iff.mp hf with hf | hf
    · exact Nat.le_trans (h2 hf) (Nat.le_add_right _ _)
    · have h3 := (hy 1 (by rw [F.g1]; exact hq)).2 hf
      have h4 := F.own l hl i j
      omega
  · obtain ⟨e, -, heG, hge, hmax⟩ := maxCell_aux g G d (G - 0) 0 rfl (by have := F.two_le; omega)
      (by rw [F.g0]; omega)
    have hei : e < i := Nat.lt_of_not_le fun h => hid (Nat.le_trans (F.g_mono h heG) hge)
    have he1 : 1 ≤ e := by
      refine Nat.pos_of_ne_zero fun h0 => ?_
      subst h0
      have := hmax (by have := F.two_le; omega)
      rw [F.g1] at this; omega
    have hdlt : d < g (e + 1) := hmax (by omega)
    have hr := Nat.le_trans (F.rest_le i hiG e he1 hei) (ceil_mono (d := d) (by omega) (by omega) hi)
    obtain ⟨hx1, hx2⟩ := hx e hge
    obtain ⟨hy1, hy2⟩ := hy (rest i e) hr
    obtain ⟨s1, s2, s3⟩ := F.split l hl i j e he1 hei
    refine ⟨by omega, fun hf => ?_⟩
    rcases Bool.or_eq_true_iff.mp hf with hf | hf
    · have := hx2 hf; omega
    · have := hy2 hf; omega

/-- **The induction**: `lo` bounds every schedule at cells below its operand ranges, and `has` every fast one. -/
theorem Facts.core (F : Facts G Lmax g kc rest lo has) {fs : List Atom} (hD : Dom fs) :
    ∀ (S : Sched) (l a b : Nat), l ≤ Lmax → S.valid fs schemes 0 l a b = true → ∀ i j, g i ≤ a → g j ≤ b →
      lo l i j ≤ S.cost l ∧ (S.fast = true → has l i j ≤ S.cost l) := by
  intro S
  induction S with
  | atom f =>
    intro l a b hl hv i j hi hj
    simp only [Sched.valid, Bool.and_eq_true] at hv
    refine ⟨?_, fun h => by simp [Sched.fast] at h⟩
    simp only [Sched.cost]
    exact Nat.le_trans (F.atom l hl i j)
      (Nat.mul_le_mul_left _ (hD f (mem_of_contains hv.1) _ _ (holds_mono hv.2 hi hj)))
  | splitA d x y ihx ihy =>
    intro l a b hl hv i j hi hj
    simp only [Sched.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
    obtain ⟨⟨⟨hd, hda⟩, hx⟩, hy⟩ := hv
    simp only [Sched.cost, Sched.fast]
    exact F.split_bound hl hd hda hi j (fun e he => ihx l d b hl hx e j he hj)
      (fun r hr => ihy l _ b hl hy r j hr hj)
  | splitB d x y ihx ihy =>
    intro l a b hl hv i j hi hj
    simp only [Sched.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
    obtain ⟨⟨⟨hd, hdb⟩, hx⟩, hy⟩ := hv
    simp only [Sched.cost, Sched.fast]
    rw [(F.symm l hl i j).1, (F.symm l hl i j).2]
    exact F.split_bound hl hd hdb hj i
      (fun e he => by rw [(F.symm l hl e i).1, (F.symm l hl e i).2]; exact ihx l a d hl hx i e hi he)
      (fun r hr => by rw [(F.symm l hl r i).1, (F.symm l hl r i).2]; exact ihy l a _ hl hy i r hi hr)
  | scheme s kids ih =>
    intro l a b hl hv i j hi hj
    cases l with
    | zero => simp [Sched.valid] at hv
    | succ l =>
      simp only [Sched.valid, Bool.and_eq_true, List.all_eq_true, List.mem_finRange, true_implies] at hv
      obtain ⟨hs, hk⟩ := hv
      have hcost : (Sched.scheme s kids).cost (l + 1) = ((List.finRange s.R).map fun t => (kids t).cost l).sum := by
        simp only [Sched.cost, Nat.add_sub_cancel, foldl_add, Nat.zero_add]
      rw [hcost]
      by_cases hG : G ≤ i
      · have h0 := F.out (l + 1) hl i j hG
        rw [h0.1, h0.2]; exact ⟨Nat.zero_le _, fun _ => Nat.zero_le _⟩
      by_cases hGj : G ≤ j
      · have h0 := F.zero_right hl i hGj
        rw [h0.1, h0.2]; exact ⟨Nat.zero_le _, fun _ => Nat.zero_le _⟩
      have hiG : i < G := Nat.lt_of_not_le hG
      have hjG : j < G := Nat.lt_of_not_le hGj
      have hl' : l ≤ Lmax := by omega
      rcases scheme_cases hs with hsw | hsw | hsw
      all_goals first
        | (have hsf : s = strassen ∨ s = winograd := by first | exact Or.inl hsw | exact Or.inr hsw
           have hsum : ((List.finRange s.R).map fun t => lo l (kc i (l1 s.u[t.val]!)) (kc j (l1 s.v[t.val]!))).sum
               ≤ ((List.finRange s.R).map fun t => (kids t).cost l).sum := by
             refine sum_map_le fun t _ => ?_
             obtain ⟨n1, n2, n3, n4⟩ := norms_fast hsf t
             have hkt := hk t
             rw [range_zero, range_zero] at hkt
             refine (ih t l _ _ hl' hkt _ _ ?_ ?_).1
             · exact Nat.le_trans (F.kc_le i hiG _ n1 n2)
                 (Nat.add_le_add_right (Nat.mul_le_mul_right _ (Nat.sub_le_sub_right hi 1)) 1)
             · exact Nat.le_trans (F.kc_le j hjG _ n3 n4)
                 (Nat.add_le_add_right (Nat.mul_le_mul_right _ (Nat.sub_le_sub_right hj 1)) 1)
           obtain ⟨h1, h2⟩ := F.sch l hl s hsf i j
           exact ⟨Nat.le_trans h1 hsum, fun _ => Nat.le_trans h2 hsum⟩)
        | skip
      subst hsw
      have hga : 1 ≤ g i := by rw [← F.g0]; exact F.g_mono (Nat.zero_le i) hiG
      have hgb : 1 ≤ g j := by rw [← F.g0]; exact F.g_mono (Nat.zero_le j) hjG
      have hkid : ∀ t, lo l i j ≤ (kids t).cost l ∧ ((kids t).fast = true → has l i j ≤ (kids t).cost l) := by
        intro t
        have hkt := hk t
        rw [range_zero, range_zero, (norms_trivial t).1, (norms_trivial t).2, Nat.mul_one, Nat.mul_one,
          Nat.sub_add_cancel (by omega), Nat.sub_add_cancel (by omega)] at hkt
        exact ih t l a b hl' hkt i j hi hj
      have hlen : ((List.finRange Verity.Accel.NaiveStrassen.trivial.R).map fun t => (kids t).cost l).length = 8 := by
        simp only [List.length_map, List.length_finRange]; rfl
      have hall : ∀ x ∈ (List.finRange Verity.Accel.NaiveStrassen.trivial.R).map (fun t => (kids t).cost l),
          lo l i j ≤ x := by
        intro x hx
        obtain ⟨t, -, rfl⟩ := List.mem_map.mp hx
        exact (hkid t).1
      obtain ⟨t1, t2⟩ := F.triv l hl i j
      refine ⟨?_, fun hf => ?_⟩
      · have := mul_le_sum hall
        rw [hlen] at this
        omega
      · have hR : Verity.Accel.NaiveStrassen.trivial.R = 8 := rfl
        simp only [Sched.fast, List.any_eq_true, List.mem_finRange, true_and, Bool.or_eq_true,
          decide_eq_true_eq] at hf
        rcases hf with hf | ⟨t, ht⟩
        · rw [hR] at hf; exact absurd hf (by decide)
        · have := pred_mul_add_le_sum hall (List.mem_map.mpr ⟨t, List.mem_finRange t, rfl⟩)
          rw [hlen] at this
          have := (hkid t).2 ht
          omega

end

/-! ## From the check to the facts -/

theorem allBelow_iff {n : Nat} {f : Nat → Bool} : allBelow n f = true ↔ ∀ k < n, f k = true := by
  induction n with
  | zero => exact ⟨fun _ _ h => absurd h (Nat.not_lt_zero _), fun _ => rfl⟩
  | succ n ih =>
    show (allBelow n f && f n) = true ↔ _
    rw [Bool.and_eq_true, ih]
    constructor
    · rintro ⟨h1, h2⟩ k hk
      rcases Nat.lt_succ_iff_lt_or_eq.mp hk with hk | rfl
      · exact h1 k hk
      · exact h2
    · intro h; exact ⟨fun k hk => h k (Nat.lt_succ_of_lt hk), h n (Nat.lt_succ_self n)⟩

/-- Level `l`'s computed tables. -/
def tab (l : Nat) : Nat × Nat := (run l).2

/-- The tables level `l` is checked over. -/
def prv : Nat → Nat × Nat
  | 0 => (0, 0)
  | l + 1 => tab l

theorem run_zero : run 0 = (checkLevel 0 (level 0 (0, 0)) (0, 0), level 0 (0, 0)) := rfl

theorem run_succ (L : Nat) : run (L + 1) =
    ((run L).1 && checkLevel (L + 1) (level (L + 1) (run L).2) (run L).2, level (L + 1) (run L).2) := rfl

theorem run_checks : ∀ L, (run L).1 = true → ∀ l, l ≤ L → checkLevel l (tab l) (prv l) = true
  | 0, h, l, hl => by
    rw [run_zero] at h
    rw [Nat.le_zero.mp hl, tab, run_zero]
    exact h
  | L + 1, h, l, hl => by
    rw [run_succ, Bool.and_eq_true] at h
    rcases Nat.lt_or_eq_of_le hl with hl | rfl
    · exact run_checks L h.1 l (Nat.le_of_lt_succ hl)
    · rw [tab, run_succ]
      exact h.2

/-- Level `l`'s `lo` at cells `i` and `j`. -/
def LO (l i j : Nat) : Nat := fld (row (tab l).1 (sz l) i) j
/-- Level `l`'s `has` at cells `i` and `j`. -/
def HA (l i j : Nat) : Nat := fld (row (tab l).2 (sz l) i) j

/-! ### Packed rows -/

theorem fld_zero (j : Nat) : fld 0 j = 0 := by rw [fld_eq, getW, Nat.zero_div, Nat.zero_mod]

theorem row_ge {T n i : Nat} (hT : T < 2 ^ (64 * n * n)) (hi : n ≤ i) : row T n i = 0 := by
  rw [row_eq]; exact getW_eq_zero hT hi

theorem packF_lt {f : Nat → Nat} {n : Nat} (hf : ∀ j < n, f j < 2 ^ 64) : packF f n < 2 ^ (64 * n) := by
  rw [packF_eq]; exact packW_lt hf

/-- Every entry of a table whose rows pass `small` is below `2^56`. -/
theorem fld_row_lt {T n : Nat} (hT : T < 2 ^ (64 * n * n)) (hs : ∀ i < n, small n (row T n i) = true)
    (i j : Nat) : fld (row T n i) j < 2 ^ 56 := by
  by_cases hi : i < n
  · by_cases hj : j < n
    · exact (small_fld (hs i hi)).2 j hj
    · rw [fld_row_ge (Nat.le_of_not_lt hj)]; exact Nat.two_pow_pos _
  · rw [row_ge hT (Nat.le_of_not_lt hi), fld_zero]; exact Nat.two_pow_pos _

theorem le_of_leRow {n X Y : Nat} (h : leRow n X Y = true) (hX : small n X = true)
    (hY : ∀ j < n, fld Y j < 2 ^ 63) {j : Nat} (hj : j < n) : fld X j ≤ fld Y j :=
  leRow_fld h (small_fld hX).1 (fun k hk => Nat.lt_trans ((small_fld hX).2 k hk) (by decide)) hY j hj

theorem fld_add' {A B : Nat} (hA : ∀ k, fld A k < 2 ^ 56) (hB : ∀ k, fld B k < 2 ^ 56) (j : Nat) :
    fld (A + B) j = fld A j + fld B j :=
  fld_add (n := j + 1) (fun k _ => by have := hA k; have := hB k; omega) (Nat.lt_succ_self j)

theorem fld_mul' {c X : Nat} (hX : ∀ k, c * fld X k < 2 ^ 64) (j : Nat) : fld (c * X) j = c * fld X j :=
  fld_mul (n := j + 1) (fun k _ => hX k) (Nat.lt_succ_self j)

theorem fld_mask' (X n j : Nat) : fld (X &&& rowMask n) j ≤ fld X j := by
  rw [fld_mask]; split <;> omega

theorem fld_mask_lt {X n j : Nat} (hj : j < n) : fld (X &&& rowMask n) j = fld X j := by
  rw [fld_mask, ite_eq_left hj]

theorem fld_transpose {T n i j : Nat} (hi : i < n) (hj : j < n) :
    fld (row (transpose T n) n i) j = fld (row T n j) i := by
  rw [transpose, row_packRows (fun k _ => packF_lt fun j _ => fld_lt _ _), ite_eq_left hi,
    fld_packF (fun j _ => fld_lt _ _), ite_eq_left hj]

theorem fld_gather {P m n μ k j : Nat} (hP : P < 2 ^ (64 * m * m)) (hj : j < n) :
    fld (row (gatherT P m n μ) n k) j = fld (row P m k) (kc j μ) := by
  rw [gatherT, row_packRows (fun k _ => packF_lt fun j _ => fld_lt _ _)]
  by_cases hk : k < m
  · rw [ite_eq_left hk, fld_packF (fun j _ => fld_lt _ _), ite_eq_left hj]
  · rw [ite_eq_right hk, row_ge hP (Nat.le_of_not_lt hk), fld_zero, fld_zero]

theorem atomMin_le_acc (fs : List Atom) (a b : Nat) : ∀ acc,
    fs.foldl (fun acc f => if f.holds a b then min acc f.weight8 else acc) acc ≤ acc := by
  induction fs with
  | nil => intro acc; exact Nat.le_refl _
  | cons f fs ih =>
    intro acc
    simp only [List.foldl_cons]
    refine Nat.le_trans (ih _) ?_
    split
    · exact Nat.min_le_left _ _
    · exact Nat.le_refl _

theorem atomMin_le_BIG (fs : List Atom) (a b : Nat) : atomMin fs a b ≤ BIG := atomMin_le_acc fs a b BIG

theorem atomMin_le {fs : List Atom} {f : Atom} (hf : f ∈ fs) {a b : Nat} (h : f.holds a b = true) :
    atomMin fs a b ≤ f.weight8 := by
  unfold atomMin
  generalize BIG = acc
  induction fs generalizing acc with
  | nil => exact absurd hf List.not_mem_nil
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rcases List.mem_cons.mp hf with rfl | hf
    · rw [ite_eq_left h]
      exact Nat.le_trans (atomMin_le_acc xs a b _) (Nat.min_le_right _ _)
    · exact ih hf _

theorem fld_atomT {i j : Nat} (hi : i < gsz) (hj : j < gsz) :
    fld (row atomT gsz i) j = atomMin rtx6000Sheet (g i) (g j) := by
  have hB : ∀ i j, atomMin rtx6000Sheet (g i) (g j) < 2 ^ 64 := fun i j =>
    Nat.lt_of_le_of_lt (atomMin_le_BIG _ _ _) (by decide)
  unfold atomT
  rw [row_packRows (fun k _ => packF_lt fun j _ => hB k j), ite_eq_left hi, fld_packF (fun j _ => hB i j),
    ite_eq_left hj]

/-- The sum of a scheme's gathered rows, fieldwise. -/
theorem fld_foldl {j : Nat} {α : Type} (R : α → Nat) (hR : ∀ p k, fld (R p) k < 2 ^ 56)
    (ps : List α) : ∀ acc : Nat, (∀ k, fld acc k + ps.length * 2 ^ 56 < 2 ^ 64) →
      fld (ps.foldl (fun acc p => acc + R p) acc) j = fld acc j + (ps.map fun p => fld (R p) j).sum := by
  induction ps with
  | nil => intro acc _; simp
  | cons p ps ih =>
    intro acc h
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    have hadd : ∀ k, fld (acc + R p) k = fld acc k + fld (R p) k := fun k =>
      fld_add (n := k + 1) (fun k' _ => by
        have h1 := h k'; have h2 := hR p k'; simp only [List.length_cons] at h1; omega) (Nat.lt_succ_self k)
    rw [ih (acc + R p) (fun k => by
      rw [hadd k]; have h1 := h k; have h2 := hR p k; simp only [List.length_cons] at h1; omega), hadd j,
      Nat.add_assoc]

/-! ### One level -/

section level
variable {l : Nat} {prev : Nat × Nat}

theorem lvl_spec (hc : checkLevel l (tab l) prev = true) :
    sz l ≤ gsz ∧ (tab l).1 < 2 ^ (64 * sz l * sz l) ∧ (tab l).2 < 2 ^ (64 * sz l * sz l) ∧
      (tab l).1 = transpose (tab l).1 (sz l) ∧ (tab l).2 = transpose (tab l).2 (sz l) ∧
      ∀ i < sz l, small (sz l) (row (tab l).1 (sz l) i) = true ∧ small (sz l) (row (tab l).2 (sz l) i) = true := by
  simp only [checkLevel, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, allBelow_iff, Bool.or_eq_true] at hc
  obtain ⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, -⟩ := hc
  exact ⟨h1, h2, h3, h4, h5, fun i hi => ⟨(h6 i hi).1.1.1.1.1, (h6 i hi).1.1.1.1.2⟩⟩

theorem lvl_lt (hc : checkLevel l (tab l) prev = true) (i j : Nat) : LO l i j < 2 ^ 56 ∧ HA l i j < 2 ^ 56 := by
  obtain ⟨-, h2, h3, -, -, h6⟩ := lvl_spec hc
  exact ⟨fld_row_lt h2 (fun i hi => (h6 i hi).1) i j, fld_row_lt h3 (fun i hi => (h6 i hi).2) i j⟩

theorem lvl_out (hc : checkLevel l (tab l) prev = true) {i : Nat} (hi : sz l ≤ i) (j : Nat) :
    LO l i j = 0 ∧ HA l i j = 0 := by
  obtain ⟨-, h2, h3, -, -, -⟩ := lvl_spec hc
  exact ⟨by rw [LO, row_ge h2 hi, fld_zero], by rw [HA, row_ge h3 hi, fld_zero]⟩

theorem lvl_col (i : Nat) {j : Nat} (hj : sz l ≤ j) : LO l i j = 0 ∧ HA l i j = 0 :=
  ⟨fld_row_ge hj, fld_row_ge hj⟩

theorem lvl_symm (hc : checkLevel l (tab l) prev = true) (i j : Nat) : LO l i j = LO l j i ∧ HA l i j = HA l j i := by
  obtain ⟨-, -, -, h4, h5, -⟩ := lvl_spec hc
  by_cases hi : i < sz l
  · by_cases hj : j < sz l
    · refine ⟨?_, ?_⟩
      · show fld (row (tab l).1 (sz l) i) j = _
        rw [h4, fld_transpose hi hj]; rfl
      · show fld (row (tab l).2 (sz l) i) j = _
        rw [h5, fld_transpose hi hj]; rfl
    · have := lvl_col (l := l) i (Nat.le_of_not_lt hj)
      have := lvl_out hc (Nat.le_of_not_lt hj) i
      omega
  · have := lvl_out hc (Nat.le_of_not_lt hi) j
    have := lvl_col (l := l) j (Nat.le_of_not_lt hi)
    omega

theorem lvl_rows (hc : checkLevel l (tab l) prev = true) {i : Nat} (hi : i < sz l) :
    leRow (sz l) (row (tab l).1 (sz l) i) (8 ^ l * (row atomT gsz i &&& rowMask (sz l))) = true ∧
    leRow (sz l) (row (tab l).2 (sz l) i) (row (tab l).1 (sz l) i + row (tab l).2 (sz l) 1) = true ∧
    ∀ e < i, e = 0 ∨
      (leRow (sz l) (row (tab l).1 (sz l) i) (row (tab l).1 (sz l) e + row (tab l).1 (sz l) (rest i e)) = true ∧
        leRow (sz l) (row (tab l).2 (sz l) i) (row (tab l).2 (sz l) e + row (tab l).1 (sz l) (rest i e)) = true) ∧
      leRow (sz l) (row (tab l).2 (sz l) i) (row (tab l).1 (sz l) e + row (tab l).2 (sz l) (rest i e)) = true := by
  simp only [checkLevel, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, allBelow_iff, Bool.or_eq_true] at hc
  have h := (hc.1.2 i hi).1
  exact ⟨h.1.1.2, h.1.2, h.2⟩

theorem lvl_atom (hc : checkLevel l (tab l) prev = true) (hl : l ≤ 15) (i j : Nat) :
    LO l i j ≤ 8 ^ l * atomMin rtx6000Sheet (g i) (g j) := by
  have hsz := (lvl_spec hc).1
  by_cases hi : i < sz l
  · by_cases hj : j < sz l
    · have hsm := ((lvl_spec hc).2.2.2.2.2 i hi).1
      have hB : ∀ k, 8 ^ l * fld (row atomT gsz i &&& rowMask (sz l)) k ≤ 8 ^ 15 * 65536 := fun k =>
        Nat.mul_le_mul (Nat.pow_le_pow_right (by decide) hl)
          (Nat.le_trans (fld_mask' _ _ _) (by
            by_cases hk : k < gsz
            · rw [fld_atomT (by omega) hk]; exact atomMin_le_BIG _ _ _
            · rw [fld_row_ge (Nat.le_of_not_lt hk)]; exact Nat.zero_le _))
      have h := le_of_leRow (lvl_rows hc hi).1 hsm (fun k _ => by
        rw [fld_mul' (fun k => Nat.lt_of_le_of_lt (hB k) (by decide))]
        exact Nat.lt_of_le_of_lt (hB k) (by decide)) hj
      rw [fld_mul' (fun k => Nat.lt_of_le_of_lt (hB k) (by decide)), fld_mask_lt hj,
        fld_atomT (by omega) (by omega)] at h
      exact h
    · rw [(lvl_col (l := l) i (Nat.le_of_not_lt hj)).1]; exact Nat.zero_le _
  · rw [(lvl_out hc (Nat.le_of_not_lt hi) j).1]; exact Nat.zero_le _

theorem lvl_split (hc : checkLevel l (tab l) prev = true) (i j e : Nat) (he : 1 ≤ e) (hei : e < i) :
    LO l i j ≤ LO l e j + LO l (rest i e) j ∧ HA l i j ≤ HA l e j + LO l (rest i e) j ∧
      HA l i j ≤ LO l e j + HA l (rest i e) j := by
  by_cases hi : i < sz l
  · by_cases hj : j < sz l
    · obtain ⟨-, h2, h3, -, -, h6⟩ := lvl_spec hc
      have hlo := fld_row_lt h2 (fun i hi => (h6 i hi).1)
      have hha := fld_row_lt h3 (fun i hi => (h6 i hi).2)
      obtain ⟨⟨s1, s2⟩, s3⟩ := ((lvl_rows hc hi).2.2 e hei).resolve_left (by omega)
      have b : ∀ (A B : Nat), (∀ k, fld A k < 2 ^ 56) → (∀ k, fld B k < 2 ^ 56) → ∀ k < sz l,
          fld (A + B) k < 2 ^ 63 := fun A B hA hB k _ => by
        rw [fld_add' hA hB]; have := hA k; have := hB k; omega
      refine ⟨?_, ?_, ?_⟩
      · have := le_of_leRow s1 (h6 i hi).1 (b _ _ (hlo e) (hlo _)) hj
        rwa [fld_add' (hlo e) (hlo _)] at this
      · have := le_of_leRow s2 (h6 i hi).2 (b _ _ (hha e) (hlo _)) hj
        rwa [fld_add' (hha e) (hlo _)] at this
      · have := le_of_leRow s3 (h6 i hi).2 (b _ _ (hlo e) (hha _)) hj
        rwa [fld_add' (hlo e) (hha _)] at this
    · have := lvl_col (l := l) i (Nat.le_of_not_lt hj)
      omega
  · have := lvl_out hc (Nat.le_of_not_lt hi) j
    omega

theorem lvl_own (hc : checkLevel l (tab l) prev = true) (i j : Nat) : HA l i j ≤ LO l i j + HA l 1 j := by
  by_cases hi : i < sz l
  · by_cases hj : j < sz l
    · obtain ⟨-, h2, h3, -, -, h6⟩ := lvl_spec hc
      have hlo := fld_row_lt h2 (fun i hi => (h6 i hi).1)
      have hha := fld_row_lt h3 (fun i hi => (h6 i hi).2)
      have := le_of_leRow (lvl_rows hc hi).2.1 (h6 i hi).2 (fun k _ => by
        rw [fld_add' (hlo i) (hha 1)]; have := hlo i k; have := hha 1 k; omega) hj
      rwa [fld_add' (hlo i) (hha 1)] at this
    · have := lvl_col (l := l) i (Nat.le_of_not_lt hj)
      omega
  · have := lvl_out hc (Nat.le_of_not_lt hi) j
    omega

end level

theorem sum_le_mul {B : Nat} : ∀ {xs : List Nat}, (∀ x ∈ xs, x ≤ B) → xs.sum ≤ xs.length * B
  | [], _ => by simp
  | x :: xs, hx => by
    have h1 := sum_le_mul (B := B) (xs := xs) fun y hy => hx y (List.mem_cons_of_mem _ hy)
    have h2 := hx x List.mem_cons_self
    simp only [List.length_cons, List.sum_cons, Nat.succ_mul]
    omega

theorem prof_strassen (F : Nat → Nat → Nat) :
    (List.finRange strassen.R).map (fun t => F (l1 strassen.u[t.val]!) (l1 strassen.v[t.val]!)) =
      strassenP.map (fun p => F p.1 p.2) := rfl

theorem prof_winograd (F : Nat → Nat → Nat) :
    (List.finRange winograd.R).map (fun t => F (l1 winograd.u[t.val]!) (l1 winograd.v[t.val]!)) =
      winogradP.map (fun p => F p.1 p.2) := rfl

section level'
variable {l : Nat} {prev : Nat × Nat}

theorem lvl_sch_rows (hc : checkLevel (l + 1) (tab (l + 1)) prev = true) {i : Nat} (hi : i < sz (l + 1)) :
    leRow (sz (l + 1)) (row (tab (l + 1)).1 (sz (l + 1)) i)
        (8 * (row prev.1 (sz l) i &&& rowMask (sz (l + 1)))) = true ∧
      leRow (sz (l + 1)) (row (tab (l + 1)).2 (sz (l + 1)) i)
        (7 * (row prev.1 (sz l) i &&& rowMask (sz (l + 1))) + (row prev.2 (sz l) i &&& rowMask (sz (l + 1)))) = true ∧
      leRow (sz (l + 1)) (row (tab (l + 1)).1 (sz (l + 1)) i) (schemeRow prev.1 (sz l) (sz (l + 1)) i strassenP) = true ∧
      leRow (sz (l + 1)) (row (tab (l + 1)).2 (sz (l + 1)) i) (schemeRow prev.1 (sz l) (sz (l + 1)) i strassenP) = true ∧
      leRow (sz (l + 1)) (row (tab (l + 1)).1 (sz (l + 1)) i) (schemeRow prev.1 (sz l) (sz (l + 1)) i winogradP) = true ∧
      leRow (sz (l + 1)) (row (tab (l + 1)).2 (sz (l + 1)) i) (schemeRow prev.1 (sz l) (sz (l + 1)) i winogradP) = true := by
  simp only [checkLevel, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, allBelow_iff, Bool.or_eq_true,
    Nat.add_sub_cancel] at hc
  obtain ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩ := ((hc.1.2 i hi).2).resolve_left (Nat.succ_ne_zero l)
  exact ⟨h1, h2, h3, h4, h5, h6⟩

/-- A scheme's row, fieldwise: the sum of the previous level's `lo` at the kid cells. -/
theorem lvl_sch_fld (hc0 : checkLevel l (tab l) prev = true) (prof : List (Nat × Nat)) (hlen : prof.length ≤ 7)
    {n i k : Nat} (hk : k < n) :
    fld (schemeRow (tab l).1 (sz l) n i prof) k = (prof.map fun p => LO l (kc i p.1) (kc k p.2)).sum := by
  have hP := (lvl_spec hc0).2.1
  have hR : ∀ (p : Nat × Nat) k', fld (row (gatherT (tab l).1 (sz l) n p.2) n (kc i p.1)) k' < 2 ^ 56 := by
    intro p k'
    by_cases hk' : k' < n
    · rw [fld_gather hP hk']; exact (lvl_lt hc0 _ _).1
    · rw [fld_row_ge (Nat.le_of_not_lt hk')]; exact Nat.two_pow_pos _
  have h7 := Nat.mul_le_mul_right (2 ^ 56) hlen
  have e := fld_foldl (j := k) (α := Nat × Nat) (fun p : Nat × Nat => row (gatherT (tab l).1 (sz l) n p.2) n (kc i p.1)) hR prof 0
    (fun k' => by have := fld_zero k'; omega)
  rw [fld_zero, Nat.zero_add] at e
  show fld (prof.foldl (fun acc p => acc + row (gatherT (tab l).1 (sz l) n p.2) n (kc i p.1)) 0) k = _
  rw [e]
  exact congrArg List.sum (List.map_congr_left fun p _ => fld_gather hP hk)

theorem lvl_sch_lt (hc0 : checkLevel l (tab l) prev = true) (prof : List (Nat × Nat)) (hlen : prof.length ≤ 7)
    {n i k : Nat} (hk : k < n) : fld (schemeRow (tab l).1 (sz l) n i prof) k < 2 ^ 63 := by
  rw [lvl_sch_fld hc0 prof hlen hk]
  have := sum_le_mul (B := 2 ^ 56) (xs := prof.map fun p => LO l (kc i p.1) (kc k p.2))
    (fun x hx => by
      obtain ⟨p, -, rfl⟩ := List.mem_map.mp hx
      exact Nat.le_of_lt (lvl_lt hc0 _ _).1)
  rw [List.length_map] at this
  have := Nat.mul_le_mul_right (2 ^ 56) hlen
  omega

theorem lvl_triv (hc1 : checkLevel (l + 1) (tab (l + 1)) (tab l) = true) (hc0 : checkLevel l (tab l) prev = true)
    (i j : Nat) : LO (l + 1) i j ≤ 8 * LO l i j ∧ HA (l + 1) i j ≤ 7 * LO l i j + HA l i j := by
  by_cases hi : i < sz (l + 1)
  · by_cases hj : j < sz (l + 1)
    · have h6 := (lvl_spec hc1).2.2.2.2.2 i hi
      obtain ⟨r1, r2, -⟩ := lvl_sch_rows hc1 hi
      have hX : ∀ k, fld (row (tab l).1 (sz l) i &&& rowMask (sz (l + 1))) k < 2 ^ 56 := fun k =>
        Nat.lt_of_le_of_lt (fld_mask' _ _ _) (lvl_lt hc0 i k).1
      have hZ : ∀ k, fld (row (tab l).2 (sz l) i &&& rowMask (sz (l + 1))) k < 2 ^ 56 := fun k =>
        Nat.lt_of_le_of_lt (fld_mask' _ _ _) (lvl_lt hc0 i k).2
      have e8 := fld_mul' (c := 8) (X := row (tab l).1 (sz l) i &&& rowMask (sz (l + 1)))
        (fun k => by have := hX k; omega)
      have e7 := fld_mul' (c := 7) (X := row (tab l).1 (sz l) i &&& rowMask (sz (l + 1)))
        (fun k => by have := hX k; omega)
      have e7z : ∀ k, fld (7 * (row (tab l).1 (sz l) i &&& rowMask (sz (l + 1))) +
          (row (tab l).2 (sz l) i &&& rowMask (sz (l + 1)))) k =
          7 * fld (row (tab l).1 (sz l) i &&& rowMask (sz (l + 1))) k +
            fld (row (tab l).2 (sz l) i &&& rowMask (sz (l + 1))) k := fun k => by
        have := fld_add (n := k + 1) (X := 7 * (row (tab l).1 (sz l) i &&& rowMask (sz (l + 1))))
          (Y := row (tab l).2 (sz l) i &&& rowMask (sz (l + 1)))
          (fun k' _ => by have := e7 k'; have := hX k'; have := hZ k'; omega) (Nat.lt_succ_self k)
        have := e7 k
        omega
      have a1 := le_of_leRow r1 h6.1 (fun k _ => by have := e8 k; have := hX k; omega) hj
      have a2 := le_of_leRow r2 h6.2 (fun k _ => by have := e7z k; have := hX k; have := hZ k; omega) hj
      rw [e8, fld_mask_lt hj] at a1
      rw [e7z, fld_mask_lt hj, fld_mask_lt hj] at a2
      exact ⟨a1, a2⟩
    · have := lvl_col (l := l + 1) i (Nat.le_of_not_lt hj)
      rw [this.1, this.2]; exact ⟨Nat.zero_le _, Nat.zero_le _⟩
  · have := lvl_out hc1 (Nat.le_of_not_lt hi) j
    rw [this.1, this.2]; exact ⟨Nat.zero_le _, Nat.zero_le _⟩

theorem lvl_sch (hc1 : checkLevel (l + 1) (tab (l + 1)) (tab l) = true) (hc0 : checkLevel l (tab l) prev = true)
    {s : Scheme} (hs : s = strassen ∨ s = winograd) (i j : Nat) :
    LO (l + 1) i j ≤ ((List.finRange s.R).map fun t => LO l (kc i (l1 s.u[t.val]!)) (kc j (l1 s.v[t.val]!))).sum ∧
      HA (l + 1) i j ≤ ((List.finRange s.R).map fun t => LO l (kc i (l1 s.u[t.val]!)) (kc j (l1 s.v[t.val]!))).sum := by
  by_cases hi : i < sz (l + 1)
  · by_cases hj : j < sz (l + 1)
    · have h6 := (lvl_spec hc1).2.2.2.2.2 i hi
      obtain ⟨-, -, r3, r4, r5, r6⟩ := lvl_sch_rows hc1 hi
      rcases hs with rfl | rfl
      · have hp := prof_strassen (fun a b => LO l (kc i a) (kc j b))
        rw [hp, ← lvl_sch_fld hc0 strassenP (by decide) hj]
        exact ⟨le_of_leRow r3 h6.1 (fun k hk => lvl_sch_lt hc0 strassenP (by decide) hk) hj,
          le_of_leRow r4 h6.2 (fun k hk => lvl_sch_lt hc0 strassenP (by decide) hk) hj⟩
      · have hp := prof_winograd (fun a b => LO l (kc i a) (kc j b))
        rw [hp, ← lvl_sch_fld hc0 winogradP (by decide) hj]
        exact ⟨le_of_leRow r5 h6.1 (fun k hk => lvl_sch_lt hc0 winogradP (by decide) hk) hj,
          le_of_leRow r6 h6.2 (fun k hk => lvl_sch_lt hc0 winogradP (by decide) hk) hj⟩
    · have := lvl_col (l := l + 1) i (Nat.le_of_not_lt hj)
      rw [this.1, this.2]; exact ⟨Nat.zero_le _, Nat.zero_le _⟩
  · have := lvl_out hc1 (Nat.le_of_not_lt hi) j
    rw [this.1, this.2]; exact ⟨Nat.zero_le _, Nat.zero_le _⟩

theorem lvl_root (hc : checkLevel l (tab l) prev = true) :
    (l ≤ thmL → 8 ^ (l + 1) + 32 ≤ HA l root root) ∧ (thmL < l → 8 ^ l < LO l root root) := by
  simp only [checkLevel, Bool.and_eq_true] at hc
  have h := hc.2
  simp only [rootOK, Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    decide_eq_true_eq] at h
  refine ⟨fun hl => h.1.resolve_left (fun h' => h' hl), fun hl => h.2.resolve_left (by omega)⟩

end level'

/-! ## The check bounds every schedule -/

theorem facts (h : checkAll = true) : Facts gsz topL g kc rest LO HA := by
  simp only [checkAll, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hg, h15⟩, hr⟩ := h
  have hcl := run_checks topL hr
  simp only [checkGrid, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, allBelow_iff, Bool.or_eq_true] at hg
  obtain ⟨⟨⟨⟨⟨⟨h0, h1⟩, hroot⟩, -⟩, hmono⟩, hkc⟩, hrest⟩ := hg
  have hsz : ∀ l, l ≤ topL → sz l ≤ gsz := fun l hl => (lvl_spec (hcl l hl)).1
  have hr15 : root = 15 := rfl
  exact {
    g0 := h0
    g1 := h1
    two_le := by omega
    mono := fun k hk => hmono k (by omega)
    kc_le := fun i hi ν h1 h4 => by
      have := hkc i hi (ν - 1) (by omega)
      rwa [Nat.sub_add_cancel h1] at this
    rest_le := fun i hi e he1 hei => (hrest i hi e hei).resolve_left (by omega)
    out := fun l hl i j hi => lvl_out (hcl l hl) (Nat.le_trans (hsz l hl) hi) j
    symm := fun l hl i j => lvl_symm (hcl l hl) i j
    atom := fun l hl i j => lvl_atom (hcl l hl) (by omega) i j
    split := fun l hl i j e he hei => lvl_split (hcl l hl) i j e he hei
    own := fun l hl i j => lvl_own (hcl l hl) i j
    triv := fun l hl i j => lvl_triv (hcl (l + 1) hl) (hcl l (by omega)) i j
    sch := fun l hl s hs i j => lvl_sch (hcl (l + 1) hl) (hcl l (by omega)) hs i j }

theorem g_root (h : checkAll = true) : g root = 256 := by
  simp only [checkAll, checkGrid, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, h256⟩, -⟩, -⟩, -⟩, -⟩, -⟩ := h
  exact h256

/-- **The tables bound every schedule** over atoms `fs` that the RTX PRO 6000 sheet's dominate, on int8 operands:
fast ones by naive tiled's `8^(l+1)` plus 32 up to `thmL`, and all of them by `8^l` above it, to `topL`. -/
theorem bound (h : checkAll = true) {fs : List Atom} (hD : Dom fs) {l : Nat} (hl : l ≤ topL) {S : Sched}
    (hv : S.valid fs schemes 0 l 256 256 = true) :
    (l ≤ thmL → S.fast = true → 8 ^ (l + 1) + 32 ≤ S.cost l) ∧ (thmL < l → 8 ^ l < S.cost l) := by
  have hg := Nat.le_of_eq (g_root h)
  obtain ⟨hlo, hhas⟩ := (facts h).core hD S l 256 256 hl hv root root hg hg
  have hc := run_checks topL (by simp only [checkAll, Bool.and_eq_true] at h; exact h.2) l hl
  obtain ⟨r1, r2⟩ := lvl_root hc
  exact ⟨fun h1 hf => Nat.le_trans (r1 h1) (hhas hf), fun h2 => Nat.lt_of_lt_of_le (r2 h2) hlo⟩

end Verity.Accel.NaiveStrassen.Packed
