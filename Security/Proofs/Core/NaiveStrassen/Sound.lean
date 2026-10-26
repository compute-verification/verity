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

end Verity.Accel.NaiveStrassen.Packed
