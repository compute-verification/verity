import Properties.Pouw.Strata

/-!
# Proofs: a served window's work strata tile its units, and what `law`, `epsAt` and `established` answer

On Lean core. `law_ok`, `epsAt_ok` and the `workK` bounds are what `Properties.Proofs.Pouw.Credit`'s bound reads of
`law` and `epsAt`; `credited_eq` and `total_eq` read `established`'s two sums as sums over `callWork`.
-/

namespace Properties.Pouw.Strata

open Lean (Json toJson)
open PouwCheck.Credit (callWork)

theorem nat_toJson (K : Nat) : PouwCheck.nat? (toJson K) = some K := by
  simp [PouwCheck.nat?, PouwWindow.natOf?, PouwWindow.intOf?, toJson, Lean.JsonNumber.fromNat]

/-- Stratum `t` under the work law at `K` over the work `W`. -/
def lawOf (K W : Nat) (t : Flock.Draw.WorkStratum) : Flock.Draw.Stratum :=
  { t.toStratum with k := Flock.Draw.workK K t.work t.size W t.floor }

/-- What `law` answers: `K` is at least one and at most the window's units, and the strata are `ss`'s under `lawOf`. -/
theorem law_ok {ss : Array Flock.Draw.WorkStratum} {K : Nat} {strata : Array Flock.Draw.Stratum}
    (h : PouwCheck.Credit.law ss (toJson K) = .ok strata) :
    1 ≤ K ∧ K ≤ ss.foldl (fun a s => a + s.size) 0 ∧ strata = ss.map (lawOf K (Flock.Draw.totalWork ss)) := by
  unfold PouwCheck.Credit.law at h
  rw [nat_toJson] at h
  by_cases hK : (1 ≤ K && K ≤ PouwCheck.Credit.MAX_K) = true
  · simp only [Option.filter, hK, ite_true, bind, Except.bind, pure, Except.pure] at h
    have hK1 : 1 ≤ K := by simp at hK; omega
    split at h
    · cases h
    · exact ⟨hK1, by omega, (Except.ok.inj h).symm⟩
  · simp only [Option.filter, hK] at h
    cases h

theorem getElem_lawOf (ss : Array Flock.Draw.WorkStratum) (K W : Nat) (s : Fin (ss.map (lawOf K W)).size) :
    (ss.map (lawOf K W))[s] = lawOf K W ss[s.val]! := by
  rw [getElem!_pos ss s.val (by simpa using s.isLt)]
  exact Array.getElem_map ..

theorem leastAbove_true (pred : Nat → Bool) (lo hi : Nat) (h : pred hi = true) :
    pred (PouwWindow.leastAbove pred lo hi) = true := by
  induction lo, hi using PouwWindow.leastAbove.induct pred with
  | case1 lo hi hlt mid hmid ih =>
    rw [PouwWindow.leastAbove, dite_eq_left hlt]
    exact (ite_eq_left hmid).symm ▸ ih hmid
  | case2 lo hi hlt mid hmid ih =>
    rw [PouwWindow.leastAbove, dite_eq_left hlt]
    exact (ite_eq_right hmid).symm ▸ ih h
  | case3 lo hi hlt => rw [PouwWindow.leastAbove, dite_eq_right hlt]; exact h

/-- What `epsAt` answers: an `ε` whose `K′` is at most `K`. -/
theorem epsAt_ok {K : Int} {δ e : Rat} {C W w : Int} (h : PouwWindow.epsAt K δ C W w = .ok e) :
    ∃ k', PouwWindow.kPrime e δ C W w = .ok k' ∧ (k' : Int) ≤ K := by
  have key : e = mkRat (PouwWindow.leastAbove (fun j => match PouwWindow.kPrime (mkRat (j : Int)
        (2 ^ PouwWindow.EPS_BITS)) δ C W w with
      | .ok k' => decide ((k' : Int) ≤ K)
      | .error _ => false) 0 (2 ^ PouwWindow.EPS_BITS) : Int) (2 ^ PouwWindow.EPS_BITS) ∧
      (match PouwWindow.kPrime (mkRat ((2 ^ PouwWindow.EPS_BITS : Nat) : Int) (2 ^ PouwWindow.EPS_BITS)) δ C W w with
        | .ok k' => decide ((k' : Int) ≤ K)
        | .error _ => false) = true := by
    unfold PouwWindow.epsAt at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    split at h
    · split at h
      · cases h
      · rcases hc : PouwWindow.kPrime (mkRat ((2 ^ PouwWindow.EPS_BITS : Nat) : Int) (2 ^ PouwWindow.EPS_BITS)) δ C W w
          with _ | k1 <;> simp only [hc] at h
        · cases h
        · by_cases hk : (k1 : Int) ≤ K
          · simp only [hk, decide_true, ite_true] at h
            exact ⟨(Except.ok.inj h).symm, by simp [hk]⟩
          · simp only [hk, decide_false] at h
            cases h
    · cases h
  have hl := leastAbove_true (fun j => match PouwWindow.kPrime (mkRat (j : Int) (2 ^ PouwWindow.EPS_BITS)) δ C W w with
      | .ok k' => decide ((k' : Int) ≤ K)
      | .error _ => false) 0 (2 ^ PouwWindow.EPS_BITS) key.2
  rw [← key.1] at hl
  revert hl
  cases hk : PouwWindow.kPrime e δ C W w with
  | error _ => simp
  | ok k' => exact fun hl => ⟨k', rfl, of_decide_eq_true hl⟩

theorem workK_le (K w n W fl : Nat) : Flock.Draw.workK K w n W fl ≤ n := Nat.min_le_left _ _

/-- A stratum draws all its units, or at least `K w n / W` of them. -/
theorem workK_ge (K w n W fl : Nat) (hW : 0 < W) :
    Flock.Draw.workK K w n W fl = n ∨ K * w * n ≤ W * Flock.Draw.workK K w n W fl := by
  unfold Flock.Draw.workK
  have h1 := Nat.div_add_mod (K * w * n + W - 1) W
  have h2 := Nat.mod_lt (K * w * n + W - 1) hW
  have hc : K * w * n ≤ W * ((K * w * n + W - 1) / W) := by omega
  rcases Nat.le_total n (max fl ((K * w * n + W - 1) / W)) with h | h
  · exact .inl (Nat.min_eq_left h)
  · rw [Nat.min_eq_right h]
    exact .inr (Nat.le_trans hc (Nat.mul_le_mul_left _ (Nat.le_max_right _ _)))

theorem workK_pos (K w n W fl : Nat) (h : 0 < K * w * n) (hW : 0 < W) : 1 ≤ Flock.Draw.workK K w n W fl := by
  unfold Flock.Draw.workK
  have hn : 0 < n := Nat.pos_of_ne_zero fun h0 => by simp [h0] at h
  have hc : 1 ≤ (K * w * n + W - 1) / W := (Nat.le_div_iff_mul_le hW).2 (by omega)
  exact Nat.le_min.2 ⟨hn, Nat.le_trans hc (Nat.le_max_right _ _)⟩

/-- At `K` at least the window's work, a stratum whose units do work draws all of them. -/
theorem workK_cap (K w n W fl : Nat) (hWK : W ≤ K) (hw : 1 ≤ w) (hW : 0 < W) : Flock.Draw.workK K w n W fl = n := by
  unfold Flock.Draw.workK
  have hKw : W ≤ K * w := Nat.le_trans hWK (Nat.le_mul_of_pos_right K hw)
  have hc : n ≤ (K * w * n + W - 1) / W := by
    refine (Nat.le_div_iff_mul_le hW).2 ?_
    have := Nat.mul_le_mul_right n hKw
    rw [Nat.mul_comm n W]
    omega
  exact Nat.min_eq_left (Nat.le_trans hc (Nat.le_max_right _ _))

/-- A unit doing work is held by a stratum doing that work. -/
theorem unitWork_pos {ss : Array Flock.Draw.WorkStratum} {u : Nat} (hu : 0 < unitWork ss u) :
    ∃ i, ∃ hi : i < ss.size, holds ss[i].toStratum u = true ∧ ss[i].work = unitWork ss u := by
  unfold unitWork at hu ⊢
  cases h : ss.find? (fun s => holds s.toStratum u) with
  | none => simp [h] at hu
  | some t =>
    obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem (Array.mem_of_find?_eq_some h)
    exact ⟨i, hi, (Array.find?_some h :), by simp⟩

theorem unitWork_eq {ss : Array Flock.Draw.WorkStratum} {u w : Nat} (hex : ∃ s ∈ ss, holds s.toStratum u = true)
    (hall : ∀ s ∈ ss, holds s.toStratum u = true → s.work = w) : unitWork ss u = w := by
  unfold unitWork
  cases h : ss.find? (fun s => holds s.toStratum u) with
  | none =>
    obtain ⟨s, hs, hh⟩ := hex
    exact absurd hh (by simpa using Array.find?_eq_none.1 h s hs)
  | some t => simpa using hall t (Array.mem_of_find?_eq_some h) (Array.find?_some h :)

theorem foldl_add {α : Type} (φ : α → Nat) : ∀ (l : List α) (a : Nat),
    l.foldl (fun acc x => acc + φ x) a = a + (l.map φ).sum
  | [], a => by simp
  | x :: l, a => by simp [foldl_add φ l (a + φ x)]; omega

/-- A fold that adds natural numbers, run in `Int`, is the cast of the same fold run in `Nat`. -/
theorem foldl_cast {α : Type} (F : Nat → α → Nat) (G : Int → α → Int) (hFG : ∀ (a : Nat) x, G a x = (F a x : Int)) :
    ∀ (l : List α) (a : Nat), l.foldl G (a : Int) = ((l.foldl F a : Nat) : Int)
  | [], _ => rfl
  | x :: l, a => by rw [List.foldl_cons, List.foldl_cons, hFG]; exact foldl_cast F G hFG l (F a x)

/-- `established`'s window work (an `Int`) is `callWork`'s. -/
theorem total_eq (cw : List (Nat × Nat)) :
    cw.foldl (fun a (x : Nat × Nat) => a + (x.1 : Int) * x.2) 0 = (((cw.map fun p => p.1 * p.2).sum : Nat) : Int) := by
  refine (foldl_cast (fun a (x : Nat × Nat) => a + x.1 * x.2) _ (fun a x => ?_) cw 0).trans ?_
  · push_cast
    rfl
  · exact congrArg _ ((foldl_add (fun (p : Nat × Nat) => p.1 * p.2) cw 0).trans (Nat.zero_add _))

theorem rev_ind {α : Type} {P : List α → Prop} (h0 : P []) (h1 : ∀ l x, P l → P (l ++ [x])) : ∀ l, P l := by
  suffices ∀ r : List α, P r.reverse by intro l; simpa using this l.reverse
  intro r
  induction r with
  | nil => exact h0
  | cons x r ih => simpa [List.reverse_cons] using h1 _ x ih

theorem getD_append_left {α : Type} (l l' : List α) (d : α) {i : Nat} (hi : i < l.length) :
    (l ++ l').getD i d = l.getD i d := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_append_left hi]

theorem credited_nat (got : List Nat) : ∀ (l : List (Nat × Nat)),
    ((List.range l.length).zip l).foldl (fun a (x : Nat × Nat × Nat) => if got.contains x.1 then a + x.2.1 * x.2.2 else a)
      0 =
      ((List.range l.length).map fun i => if i ∈ got then (l.getD i (0, 0)).1 * (l.getD i (0, 0)).2 else 0).sum := by
  refine rev_ind (by simp) fun l x ih => ?_
  rw [List.length_append, List.length_singleton, List.range_succ,
    List.zip_append (by simp), List.foldl_append, ih, List.map_append, List.sum_append]
  have hmap : (List.range l.length).map (fun i => if i ∈ got then ((l ++ [x]).getD i (0, 0)).1 *
      ((l ++ [x]).getD i (0, 0)).2 else 0) =
      (List.range l.length).map (fun i => if i ∈ got then (l.getD i (0, 0)).1 * (l.getD i (0, 0)).2 else 0) := by
    refine List.map_congr_left fun i hi => ?_
    rw [List.mem_range] at hi
    rw [getD_append_left _ _ _ hi]
  rw [hmap]
  by_cases hg : l.length ∈ got <;> simp [hg, List.getD_eq_getElem?_getD]

/-- `established`'s credited work (an `Int`) is `callWork`'s over the credited calls. -/
theorem credited_eq (got : List Nat) (l : List (Nat × Nat)) :
    ((List.range l.length).zip l).foldl
        (fun a (x : Nat × Nat × Nat) => if got.contains x.1 = true then a + (x.2.1 : Int) * x.2.2 else a) 0 =
      ((((List.range l.length).map fun i =>
        if i ∈ got then (l.getD i (0, 0)).1 * (l.getD i (0, 0)).2 else 0).sum : Nat) : Int) := by
  refine (foldl_cast (fun a (x : Nat × Nat × Nat) => if got.contains x.1 then a + x.2.1 * x.2.2 else a) _
    (fun a x => ?_) _ 0).trans (congrArg _ (credited_nat got l))
  by_cases hx : got.contains x.1 = true
  · simp only [hx, ite_true]
    push_cast
    rfl
  · simp only [hx]
    rfl

/-- What `established` answers is `epsAt`'s at `callWork`'s credited and total work. -/
theorem established_ok {layout : Json} {got : List Nat} {K : Nat} {e : Rat}
    (h : PouwCheck.Credit.established layout got K = some e) :
    PouwWindow.epsAt K PouwCheck.Credit.DELTA
      ((((List.range (callWork layout).length).map fun i =>
        if i ∈ got then ((callWork layout).getD i (0, 0)).1 * ((callWork layout).getD i (0, 0)).2 else 0).sum : Nat) :
        Int)
      ((((callWork layout).map fun p => p.1 * p.2).sum : Nat) : Int) 1 = .ok e := by
  unfold PouwCheck.Credit.established at h
  dsimp only at h
  rw [credited_eq, total_eq] at h
  revert h
  cases PouwWindow.epsAt _ _ _ _ _ with
  | error _ => intro h; cases h
  | ok a => intro h; cases h; rfl

theorem callStart_succ (layout : Json) (i : Nat) (hi : i < (callWork layout).length) :
    callStart layout (i + 1) = callStart layout i + ((callWork layout).getD i (0, 0)).1 := by
  unfold callStart
  rw [List.take_add_one, List.map_append, List.sum_append]
  simp [List.getElem?_eq_getElem hi, List.getD_eq_getElem?_getD]

end Properties.Pouw.Strata
