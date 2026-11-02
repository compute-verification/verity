import Properties.Pouw.Strata
import Std.Data.String.ToNat

/-!
# Proofs: a served window's work strata tile its units, and what `law`, `epsAt` and `established` answer

On Lean core. `law_ok`, `epsAt_ok` and the `workK` bounds are what `Proofs.Pouw.Credit`'s bound reads of `law` and
`epsAt`; `credited_eq` and `total_eq` read `established`'s two sums as sums over `servedWork` and `callWork`.

`strataTile`: `strata_ok` peels `strata`'s checks and loops (`forIn_fold`) down to its last loop, a fold of `add` over
the calls' `(K, units)` items (`stepOf`, `itemOf`), and reads `callWork` as the same items; `Inv` is what that fold keeps
true call by call (`inv_step`): every range ends by the units so far, the strata's sizes and work add up, their
templates are distinct and name their K, and each unit so far is held, only by strata of its call's work.
-/

namespace Properties.Pouw.Strata

open Lean (Json toJson)
open PouwCheck.Credit (callWork servedWork)

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

/-- `established`'s credited work (an `Int`) is `servedWork`'s over the credited calls. -/
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

/-- What `established` answers is `epsAt`'s at `servedWork`'s credited work and `callWork`'s total. -/
theorem established_ok {layout : Json} {got served : List Nat} {K : Nat} {e : Rat}
    (h : PouwCheck.Credit.established layout got served K = some e) :
    PouwWindow.epsAt K PouwCheck.Credit.DELTA
      ((((List.range (servedWork layout served).length).map fun i =>
        if i ∈ got then ((servedWork layout served).getD i (0, 0)).1 *
          ((servedWork layout served).getD i (0, 0)).2 else 0).sum : Nat) : Int)
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

open Flock.Draw (WorkStratum)

theorem holds_iff (s : Flock.Draw.Stratum) (u : Nat) :
    holds s u = true ↔ ∃ p ∈ s.units.toList, p.1 ≤ u ∧ u < p.2 := by
  unfold holds
  rw [← Array.any_toList, List.any_eq_true]
  constructor
  · rintro ⟨⟨lo, hi⟩, hp, h⟩
    exact ⟨(lo, hi), hp, by simpa using h⟩
  · rintro ⟨⟨lo, hi⟩, hp, h⟩
    exact ⟨(lo, hi), hp, by simpa using h⟩

theorem size_eq (s : Flock.Draw.Stratum) : s.size = (s.units.toList.map fun p => p.2 - p.1).sum := by
  unfold Flock.Draw.Stratum.size
  rw [← Array.foldl_toList]
  exact (foldl_add (fun p : Nat × Nat => p.2 - p.1) _ 0).trans (Nat.zero_add _)

theorem totalWork_list (ss : Array WorkStratum) :
    Flock.Draw.totalWork ss = (ss.toList.map fun s => s.work * s.size).sum := by
  unfold Flock.Draw.totalWork
  rw [← Array.foldl_toList]
  exact (foldl_add (fun s : WorkStratum => s.work * s.size) _ 0).trans (Nat.zero_add _)

/-- `strata`'s last loop adding call units `[a, a + z)` to the stratum `s` of their template. -/
def merge (a z : Nat) (s : WorkStratum) : WorkStratum :=
  match s.units.back? with
  | some (lo, hi) =>
    if hi == a then { s with units := s.units.pop.push (lo, a + z) } else { s with units := s.units.push (a, a + z) }
  | none => { s with units := s.units.push (a, a + z) }

/-- `strata`'s last loop's step at the strata and units so far `st`, for a call of `q.2` units of a linear of K
`q.1`. -/
def add (r c : Nat) (st : Array WorkStratum × Nat) (q : Nat × Nat) : Array WorkStratum × Nat :=
  match st.1.findIdx? (·.template == "k" ++ toString q.1) with
  | none =>
    (st.1.push
      { template := "k" ++ toString q.1, units := #[(st.2, st.2 + q.2)], k := 0, work := r * c * q.1, floor := 1 },
      st.2 + q.2)
  | some i => (st.1.modify i (merge st.2 q.2), st.2 + q.2)

theorem push_spec (a z : Nat) (s : WorkStratum) (hs : ∀ p ∈ s.units.toList, p.1 ≤ p.2 ∧ p.2 ≤ a) :
    let s' : WorkStratum := { s with units := s.units.push (a, a + z) }
    s'.size = s.size + z ∧ (∀ p ∈ s'.units.toList, p.1 ≤ p.2 ∧ p.2 ≤ a + z) ∧
      ∀ u, holds s'.toStratum u = true ↔ holds s.toStratum u = true ∨ (a ≤ u ∧ u < a + z) := by
  intro s'
  refine ⟨?_, ?_, fun u => ?_⟩
  · simp only [s', size_eq, Array.toList_push, List.map_append, List.sum_append]
    simp
  · simp only [s', Array.toList_push, List.mem_append, List.mem_singleton]
    rintro p (hp | rfl)
    · exact ⟨(hs p hp).1, Nat.le_trans (hs p hp).2 (Nat.le_add_right _ _)⟩
    · exact ⟨Nat.le_add_right _ _, Nat.le_refl _⟩
  · simp only [s', holds_iff, Array.toList_push, List.mem_append, List.mem_singleton]
    constructor
    · rintro ⟨p, hp | rfl, h⟩
      · exact .inl ⟨p, hp, h⟩
      · exact .inr h
    · rintro (⟨p, hp, h⟩ | h)
      · exact ⟨p, .inl hp, h⟩
      · exact ⟨_, .inr rfl, h⟩

theorem merge_spec (a z : Nat) (s : WorkStratum) (hs : ∀ p ∈ s.units.toList, p.1 ≤ p.2 ∧ p.2 ≤ a) :
    (merge a z s).template = s.template ∧ (merge a z s).work = s.work ∧ (merge a z s).size = s.size + z ∧
      (∀ p ∈ (merge a z s).units.toList, p.1 ≤ p.2 ∧ p.2 ≤ a + z) ∧
      ∀ u, holds (merge a z s).toStratum u = true ↔ holds s.toStratum u = true ∨ (a ≤ u ∧ u < a + z) := by
  unfold merge
  split
  · rename_i lo hi hb
    obtain ⟨D, hD⟩ := List.getLast?_eq_some_iff.1 ((Array.getLast?_toList _).trans hb)
    split
    · rename_i hh
      have hha : hi = a := by simpa using hh
      subst hha
      have hlo := (hs (lo, hi) (by rw [hD]; simp)).1
      have hU : (s.units.pop.push (lo, hi + z)).toList = D ++ [(lo, hi + z)] := by
        rw [Array.toList_push, Array.toList_pop, hD, List.dropLast_concat]
      refine ⟨rfl, rfl, ?_, ?_, fun u => ?_⟩
      · show Flock.Draw.Stratum.size { s.toStratum with units := _ } = _
        rw [size_eq, size_eq, hU, hD]
        simp only [List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
        omega
      · show ∀ p ∈ (s.units.pop.push (lo, hi + z)).toList, _
        rw [hU]
        simp only [List.mem_append, List.mem_singleton]
        rintro p (hp | rfl)
        · have := hs p (by rw [hD]; exact List.mem_append_left _ hp)
          exact ⟨this.1, by omega⟩
        · exact ⟨by omega, Nat.le_refl _⟩
      · show holds { s.toStratum with units := _ } u = true ↔ _
        rw [holds_iff, holds_iff, hU, hD]
        simp only [List.mem_append, List.mem_singleton]
        constructor
        · rintro ⟨p, hp | rfl, h⟩
          · exact .inl ⟨p, .inl hp, h⟩
          · by_cases hu : u < hi
            · exact .inl ⟨_, .inr rfl, h.1, hu⟩
            · exact .inr ⟨by omega, h.2⟩
        · rintro (⟨p, hp | rfl, h⟩ | h)
          · exact ⟨p, .inl hp, h⟩
          · exact ⟨_, .inr rfl, h.1, by omega⟩
          · exact ⟨_, .inr rfl, by omega, h.2⟩
    · exact ⟨rfl, rfl, push_spec a z s hs⟩
  · exact ⟨rfl, rfl, push_spec a z s hs⟩


theorem merge_template (a z : Nat) (s : WorkStratum) : (merge a z s).template = s.template := by
  unfold merge
  split
  · split <;> rfl
  · rfl

theorem mem_modify {α : Type} {f : α → α} {l : List α} {j : Nat} {x : α} (hj : j < l.length)
    (h : x ∈ l.modify j f) : x ∈ l ∨ x = f l[j] := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 h
  rw [List.getElem_modify]
  split
  · rename_i hji
    subst hji
    exact .inr rfl
  · exact .inl (List.getElem_mem _)

theorem mem_modify_of {α : Type} {f : α → α} {l : List α} {j : Nat} {x : α} (h : x ∈ l) :
    x ∈ l.modify j f ∨ f x ∈ l.modify j f := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 h
  have hi' : i < (l.modify j f).length := by rw [List.length_modify]; exact hi
  have hm := List.getElem_mem hi'
  rw [List.getElem_modify] at hm
  split at hm
  · exact .inr hm
  · exact .inl hm

theorem modify_mem_at {α : Type} {f : α → α} {l : List α} {j : Nat} (hj : j < l.length) :
    f l[j] ∈ l.modify j f := by
  have hj' : j < (l.modify j f).length := by rw [List.length_modify]; exact hj
  have hm := List.getElem_mem hj'
  rw [List.getElem_modify, ite_eq_left rfl] at hm
  exact hm

theorem sum_modify {α : Type} (φ : α → Nat) (f : α → α) (d : α → Nat) :
    ∀ (l : List α) (j : Nat) (hj : j < l.length), (∀ x ∈ l, φ (f x) = φ x + d x) →
      ((l.modify j f).map φ).sum = (l.map φ).sum + d l[j]
  | [], _, hj, _ => absurd hj (by simp)
  | x :: l, 0, _, hf => by
    simp only [List.modify_zero_cons, List.map_cons, List.sum_cons, List.getElem_cons_zero,
      hf x (List.mem_cons_self ..)]
    omega
  | x :: l, j + 1, hj, hf => by
    simp only [List.modify_succ_cons, List.map_cons, List.sum_cons, List.getElem_cons_succ]
    rw [sum_modify φ f d l j (by simpa using hj) (fun y hy => hf y (List.mem_cons_of_mem _ hy))]
    omega

theorem map_modify {α β : Type} (g : α → β) (f : α → α) (hf : ∀ x, g (f x) = g x) :
    ∀ (l : List α) (j : Nat), (l.modify j f).map g = l.map g
  | [], _ => by simp
  | x :: l, 0 => by simp only [List.modify_zero_cons, List.map_cons, hf]
  | x :: l, j + 1 => by simp only [List.modify_succ_cons, List.map_cons, map_modify g f hf l j]

theorem sum_take_le {α : Type} (φ : α → Nat) (l : List α) (i : Nat) : ((l.take i).map φ).sum ≤ (l.map φ).sum := by
  conv => rhs; rw [← List.take_append_drop i l]
  rw [List.map_append, List.sum_append]
  exact Nat.le_add_right _ _

/-- The first unit of item `i`: the units of the items before it. -/
def start (items : List (Nat × Nat)) (i : Nat) : Nat := ((items.take i).map (·.2)).sum

theorem start_le (items : List (Nat × Nat)) {i j : Nat} (h : i ≤ j) : start items i ≤ start items j := by
  have := sum_take_le (·.2) (items.take j) i
  rwa [List.take_take, Nat.min_eq_left h] at this

theorem start_append (items : List (Nat × Nat)) (q : Nat × Nat) {i : Nat} (h : i ≤ items.length) :
    start (items ++ [q]) i = start items i := by
  unfold start
  rw [List.take_append_of_le_length h]

theorem start_length (items : List (Nat × Nat)) : start items items.length = (items.map (·.2)).sum := by
  unfold start
  rw [List.take_length]

theorem start_last (items : List (Nat × Nat)) (q : Nat × Nat) :
    start (items ++ [q]) (items.length + 1) = (items.map (·.2)).sum + q.2 := by
  unfold start
  rw [List.take_of_length_le (by simp)]
  simp

/-- What `strata`'s last loop keeps true of the strata and units `st` it has after the items `items`, `(K, units)` per
call. -/
structure Inv (r c : Nat) (items : List (Nat × Nat)) (st : Array WorkStratum × Nat) : Prop where
  at_eq : st.2 = (items.map (·.2)).sum
  rng : ∀ s ∈ st.1.toList, ∀ p ∈ s.units.toList, p.1 ≤ p.2 ∧ p.2 ≤ st.2
  size : (st.1.toList.map (·.size)).sum = st.2
  work : (st.1.toList.map fun s => s.work * s.size).sum = (items.map fun q => q.2 * (r * c * q.1)).sum
  tmpl : (st.1.toList.map (·.template)).Nodup
  kk : ∀ s ∈ st.1.toList, ∃ k, s.template = "k" ++ toString k ∧ s.work = r * c * k
  cover : ∀ i (hi : i < items.length) u, start items i ≤ u → u < start items (i + 1) →
    (∃ s ∈ st.1.toList, holds s.toStratum u = true) ∧
      ∀ s ∈ st.1.toList, holds s.toStratum u = true → s.work = r * c * items[i].1

theorem inv_nil (r c : Nat) : Inv r c [] (#[], 0) where
  at_eq := rfl
  rng := by simp
  size := rfl
  work := rfl
  tmpl := List.nodup_nil
  kk := by simp
  cover := fun _ hi => absurd hi (by simp)

theorem not_holds_above {s : WorkStratum} {A u : Nat} (hs : ∀ p ∈ s.units.toList, p.1 ≤ p.2 ∧ p.2 ≤ A) (hu : A ≤ u) :
    holds s.toStratum u = false := by
  cases h : holds s.toStratum u with
  | false => rfl
  | true =>
    obtain ⟨p, hp, h1⟩ := (holds_iff _ _).1 h
    have := (hs p hp).2
    omega

theorem inv_step {r c : Nat} {items : List (Nat × Nat)} {out : Array WorkStratum} {A : Nat} (h : Inv r c items (out, A))
    (q : Nat × Nat) : Inv r c (items ++ [q]) (add r c (out, A) q) := by
  have hA : A = (items.map (·.2)).sum := h.at_eq
  have hrng : ∀ s ∈ out.toList, ∀ p ∈ s.units.toList, p.1 ≤ p.2 ∧ p.2 ≤ A := h.rng
  have hcov := h.cover
  -- the units before `A` keep their strata, and the new call's are `[A, A + q.2)`
  have hcases : ∀ i (hi : i < (items ++ [q]).length) u, start (items ++ [q]) i ≤ u → u < start (items ++ [q]) (i + 1) →
      (∃ (hi' : i < items.length), start items i ≤ u ∧ u < start items (i + 1) ∧ u < A ∧
        (items ++ [q])[i] = items[i]) ∨ (i = items.length ∧ A ≤ u ∧ u < A + q.2 ∧ (items ++ [q])[i] = q) := by
    intro i hi u hlo hup
    rw [List.length_append, List.length_singleton] at hi
    rcases Nat.lt_or_ge i items.length with hlt | hge
    · rw [start_append _ _ (Nat.le_of_lt hlt)] at hlo
      rw [start_append _ _ hlt] at hup
      have := start_le items (Nat.succ_le_of_lt hlt)
      rw [start_length, ← hA] at this
      exact .inl ⟨hlt, hlo, hup, by omega, List.getElem_append_left hlt⟩
    · have hi' : i = items.length := by omega
      subst hi'
      rw [start_append _ _ (Nat.le_refl _), start_length, ← hA] at hlo
      rw [start_last, ← hA] at hup
      exact .inr ⟨rfl, hlo, hup, by simp⟩
  unfold add
  split
  · rename_i hn
    rw [Array.findIdx?_eq_none_iff] at hn
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hA]
    · simp only [Array.toList_push, List.mem_append, List.mem_singleton]
      rintro s (hs | rfl) p hp
      · exact ⟨(hrng s hs p hp).1, Nat.le_trans (hrng s hs p hp).2 (Nat.le_add_right _ _)⟩
      · simp at hp
        subst hp
        exact ⟨Nat.le_add_right _ _, Nat.le_refl _⟩
    · simp only [Array.toList_push, List.map_append, List.sum_append, h.size]
      simp [size_eq]
    · simp only [Array.toList_push, List.map_append, List.sum_append, h.work]
      simp [size_eq, Nat.mul_comm]
    · simp only [Array.toList_push, List.map_append, List.nodup_append]
      refine ⟨h.tmpl, List.nodup_cons.2 ⟨by simp, List.nodup_nil⟩, ?_⟩
      intro a ha b hb
      simp only [List.mem_map] at ha
      obtain ⟨s, hs, rfl⟩ := ha
      simp only [List.map_cons, List.map_nil, List.mem_singleton] at hb
      subst hb
      have := hn s (Array.mem_toList_iff.1 hs)
      simpa using this
    · simp only [Array.toList_push, List.mem_append, List.mem_singleton]
      rintro s (hs | rfl)
      · exact h.kk s hs
      · exact ⟨q.1, rfl, rfl⟩
    · intro i hi u hlo hup
      have hnew : ∀ s : WorkStratum, s.units = #[(A, A + q.2)] → ∀ u,
          holds s.toStratum u = true ↔ A ≤ u ∧ u < A + q.2 := by
        intro s hs u
        rw [holds_iff, hs]
        simp
      simp only [Array.toList_push, List.mem_append, List.mem_singleton]
      rcases hcases i hi u hlo hup with ⟨hi', hlo', hup', hu, hget⟩ | ⟨rfl, hlo', hup', hget⟩
      · obtain ⟨⟨s, hs, hh⟩, hall⟩ := hcov i hi' u hlo' hup'
        refine ⟨⟨s, .inl hs, hh⟩, ?_⟩
        rintro s' (hs' | rfl) hh'
        · rw [hget]
          exact hall s' hs' hh'
        · have := (hnew _ rfl u).1 hh'
          omega
      · refine ⟨⟨_, .inr rfl, (hnew _ rfl u).2 ⟨hlo', hup'⟩⟩, ?_⟩
        rintro s' (hs' | rfl) hh'
        · rw [not_holds_above (hrng s' hs') hlo'] at hh'
          cases hh'
        · rw [hget]
  · rename_i j hj
    obtain ⟨hjlt, hjt, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hj
    have hjl : j < out.toList.length := by simpa using hjlt
    have hget : out.toList[j] = out[j] := Array.getElem_toList hjl
    have hmspec := fun s (hs : s ∈ out.toList) => merge_spec A q.2 s (hrng s hs)
    have hjmem : out[j] ∈ out.toList := hget ▸ List.getElem_mem hjl
    have hwork : out[j].work = r * c * q.1 := by
      obtain ⟨k, hk, hw⟩ := h.kk _ hjmem
      have ht : out[j].template = "k" ++ toString q.1 := by simpa using hjt
      rw [ht, String.append_right_inj] at hk
      rw [hw, Nat.repr_injective hk]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hA]
    · simp only [Array.toList_modify]
      intro s hs p hp
      rcases mem_modify hjl hs with hs | rfl
      · exact ⟨(hrng s hs p hp).1, Nat.le_trans (hrng s hs p hp).2 (Nat.le_add_right _ _)⟩
      · exact (hmspec _ (List.getElem_mem hjl)).2.2.2.1 p hp
    · simp only [Array.toList_modify]
      rw [sum_modify _ _ (fun _ => q.2) _ j hjl (fun x hx => (hmspec x hx).2.2.1), h.size]
    · simp only [Array.toList_modify]
      rw [sum_modify (fun s : WorkStratum => s.work * s.size) _ (fun x => x.work * q.2) _ j hjl
        (fun x hx => by rw [(hmspec x hx).2.1, (hmspec x hx).2.2.1, Nat.mul_add]), h.work, hget, hwork]
      simp [Nat.mul_comm]
    · simp only [Array.toList_modify]
      rw [map_modify _ _ (merge_template A q.2)]
      exact h.tmpl
    · simp only [Array.toList_modify]
      intro s hs
      rcases mem_modify hjl hs with hs | rfl
      · exact h.kk s hs
      · rw [(hmspec _ (List.getElem_mem hjl)).1, (hmspec _ (List.getElem_mem hjl)).2.1]
        exact h.kk _ (List.getElem_mem hjl)
    · intro i hi u hlo hup
      simp only [Array.toList_modify]
      rcases hcases i hi u hlo hup with ⟨hi', hlo', hup', hu, hgi⟩ | ⟨rfl, hlo', hup', hgi⟩
      · obtain ⟨⟨s, hs, hh⟩, hall⟩ := hcov i hi' u hlo' hup'
        refine ⟨?_, ?_⟩
        · rcases mem_modify_of (f := merge A q.2) (j := j) hs with hs' | hs'
          · exact ⟨s, hs', hh⟩
          · exact ⟨_, hs', ((hmspec s hs).2.2.2.2 u).2 (.inl hh)⟩
        · intro s' hs' hh'
          rw [hgi]
          rcases mem_modify hjl hs' with hs' | rfl
          · exact hall s' hs' hh'
          · have hm := hmspec _ (List.getElem_mem hjl)
            rw [hm.2.1]
            rcases (hm.2.2.2.2 u).1 hh' with hh' | hh'
            · exact hall _ (List.getElem_mem hjl) hh'
            · omega
      · refine ⟨⟨_, modify_mem_at hjl, ((hmspec _ (List.getElem_mem hjl)).2.2.2.2 u).2 (.inr ⟨hlo', hup'⟩)⟩, ?_⟩
        intro s' hs' hh'
        rw [hgi]
        rcases mem_modify hjl hs' with hs' | rfl
        · rw [not_holds_above (hrng s' hs') hlo'] at hh'
          cases hh'
        · rw [(hmspec _ (List.getElem_mem hjl)).2.1, hget, hwork]

theorem refuse_ne {α : Type} (w : String) (a : α) : PouwCheck.Credit.layoutRefuse w ≠ .ok a := by
  intro h
  cases h

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem forIn_fold {α σ ε : Type} (g : σ → α → σ) (Q : α → Prop) (f : α → σ → Except ε (ForInStep σ))
    (hf : ∀ x b s, f x b = .ok s → s = .yield (g b x) ∧ Q x) :
    ∀ (xs : List α) (b0 : σ) {b : σ}, forIn xs b0 f = .ok b → b = xs.foldl g b0 ∧ ∀ x ∈ xs, Q x
  | [], b0, b, h => by
    rw [List.forIn_nil] at h
    cases h
    exact ⟨rfl, by simp⟩
  | x :: xs, b0, b, h => by
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := bind_ok h
    obtain ⟨rfl, hq⟩ := hf x b0 s hs
    obtain ⟨hb, hall⟩ := forIn_fold g Q f hf xs (g b0 x) h
    exact ⟨hb, fun y hy => (List.mem_cons.1 hy).elim (· ▸ hq) (hall y)⟩

theorem mapM_pair {α β : Type} {f : α → Option β} {l : List α} {b0 b1 : β} (h : l.mapM f = some [b0, b1]) :
    ∃ a0 a1, l = [a0, a1] ∧ f a0 = some b0 ∧ f a1 = some b1 := by
  match l, h with
  | [], h => simp at h
  | [a0], h => cases h0 : f a0 <;> simp [List.mapM_cons, h0] at h
  | [a0, a1], h =>
    cases h0 : f a0 <;> cases h1 : f a1 <;> simp [List.mapM_cons, h0, h1] at h
    exact ⟨a0, a1, rfl, by rw [h0, h.1], by rw [h1, h.2]⟩
  | a0 :: a1 :: a2 :: l, h =>
    cases h0 : f a0 <;> cases h1 : f a1 <;> cases h2 : f a2 <;> cases h3 : List.mapM f l <;>
      simp [List.mapM_cons, h0, h1, h2, h3] at h

/-- `strata`'s first two loops' step: the row `x` parses to `p x`, which is pushed. -/
def pushSome {α β : Type} (p : α → Option β) (b : Array β) (x : α) : Array β :=
  match p x with
  | some y => b.push y
  | none => b

theorem foldl_pushSome {α β : Type} (p : α → Option β) :
    ∀ (l : List α) (b : Array β), (l.foldl (pushSome p) b).toList = b.toList ++ l.filterMap p
  | [], b => by simp
  | x :: l, b => by
    rw [List.foldl_cons, foldl_pushSome p l, List.filterMap_cons]
    unfold pushSome
    cases p x <;> simp

/-- A row of the layout's linears, as `strata` reads it. -/
def parseL (x : Json) : Option (String × Nat × Nat) :=
  match x with
  | .arr a => match a.toList with
    | [.str s, k, n] => do pure (s, ← PouwCheck.nat? k, ← PouwCheck.nat? n)
    | _ => none
  | _ => none

/-- A row of the layout's calls, as `strata` reads it. -/
def parseC (x : Json) : Option (String × Nat) :=
  match x with
  | .arr a => match a.toList with
    | [.str s, b] => (s, ·) <$> PouwCheck.nat? b
    | _ => none
  | _ => none

/-- `strata`'s last loop's step for the call `x`, its linear's K and n from `ks`. -/
def stepOf (ks : Array (String × Nat × Nat)) (m r c : Nat) (st : Array WorkStratum × Nat) (x : String × Nat) :
    Array WorkStratum × Nat :=
  match ks.find? (·.1 == x.1) with
  | some t => add r c st (t.2.1, m / r * (t.2.2 / c))
  | none => st

/-- The call `x`'s linear's K and its units. -/
def itemOf (ks : Array (String × Nat × Nat)) (m r c : Nat) (x : String × Nat) : Option (Nat × Nat) :=
  (ks.find? (·.1 == x.1)).map fun t => (t.2.1, m / r * (t.2.2 / c))

theorem foldl_stepOf (ks : Array (String × Nat × Nat)) (m r c : Nat) :
    ∀ (l : List (String × Nat)) (st : Array WorkStratum × Nat),
      l.foldl (stepOf ks m r c) st = (l.filterMap (itemOf ks m r c)).foldl (add r c) st
  | [], _ => rfl
  | x :: l, st => by
    rw [List.foldl_cons, List.filterMap_cons, foldl_stepOf ks m r c l]
    cases hx : ks.find? (·.1 == x.1) with
    | none => simp only [stepOf, itemOf, hx, Option.map_none]
    | some t => simp only [stepOf, itemOf, hx, Option.map_some, List.foldl_cons]

theorem filterMap_comp_eq {α β γ : Type} (f : α → Option β) (g : β → Option γ) (F : α → Option γ) (l : List α)
    (h : ∀ x ∈ l, (f x).bind g = F x) : (l.filterMap f).filterMap g = l.filterMap F := by
  rw [List.filterMap_filterMap]
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [List.filterMap_cons, List.filterMap_cons, h x (List.mem_cons_self ..),
      ih fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem strata_ok {layout : Json} {ss : Array WorkStratum} (h : PouwCheck.Credit.strata layout = .ok ss) :
    ∃ (r c : Nat) (items : List (Nat × Nat)),
      callWork layout = items.map (fun q => (q.2, r * c * q.1)) ∧ ss = (items.foldl (add r c) (#[], 0)).1 := by
  unfold PouwCheck.Credit.strata at h
  dsimp only at h
  split at h
  · split at h
    · rename_i tile m linears calls heq
      simp only [Prod.mk.injEq] at heq
      obtain ⟨h1, hm, h3, h4⟩ := heq
      split at h
      · rename_i r c hrc
        obtain ⟨a0, a1, hl, hr, hc⟩ := mapM_pair hrc
        split at h
        · obtain ⟨ks, hks, h⟩ := bind_ok h
          obtain ⟨cs, hcs, h⟩ := bind_ok h
          obtain ⟨_, -, h⟩ := bind_ok h
          split at h
          · obtain ⟨_, h', -⟩ := bind_ok h
            exact absurd h' (refuse_ne _ _)
          · obtain ⟨_, -, h⟩ := bind_ok h
            obtain ⟨_, -, h⟩ := bind_ok h
            obtain ⟨st, hst, h⟩ := bind_ok h
            cases h
            rw [← Array.forIn_toList] at hks hcs hst
            obtain ⟨hks', hLs⟩ := forIn_fold (pushSome parseL) (fun x => (parseL x).isSome) _ (by
              intro x b s hs
              split at hs
              · rename_i name k n heq
                have hp : parseL x = some (name, k, n) := heq
                split at hs
                · obtain ⟨_, h', -⟩ := bind_ok hs
                  exact absurd h' (refuse_ne _ _)
                · cases hs
                  exact ⟨by simp only [pushSome, hp], by simp [hp]⟩
              · obtain ⟨_, h', -⟩ := bind_ok hs
                exact absurd h' (refuse_ne _ _)) linears.toList #[] hks
            obtain ⟨hcs', hCs⟩ := forIn_fold (pushSome parseC) (fun x => (parseC x).isSome) _ (by
              intro x b s hs
              split at hs
              · rename_i lin blk heq
                have hp : parseC x = some (lin, blk) := heq
                split at hs
                · cases hs
                  exact ⟨by simp only [pushSome, hp], by simp [hp]⟩
                · obtain ⟨_, h', -⟩ := bind_ok hs
                  exact absurd h' (refuse_ne _ _)
              · obtain ⟨_, h', -⟩ := bind_ok hs
                exact absurd h' (refuse_ne _ _)) calls.toList #[] hcs
            obtain ⟨hst', -⟩ := forIn_fold (stepOf ks m r c) (fun _ => True) _ (by
              intro x b s hs
              split at hs
              · rename_i fst k n hfind
                split at hs
                · rename_i hidx
                  have hidx' : Array.findIdx? (fun x => x.template == "k" ++ toString k) b.fst = none := hidx
                  cases hs
                  refine ⟨?_, trivial⟩
                  simp only [stepOf, hfind, add, hidx']
                  rfl
                · rename_i i hidx
                  have hidx' : Array.findIdx? (fun x => x.template == "k" ++ toString k) b.fst = some i := hidx
                  cases hs
                  refine ⟨?_, trivial⟩
                  simp only [stepOf, hfind, add, hidx']
                  rfl
              · obtain ⟨_, h', -⟩ := bind_ok hs
                exact absurd h' (refuse_ne _ _)) _ (#[], 0) hst
            refine ⟨r, c, cs.toList.filterMap (itemOf ks m r c), ?_, by rw [hst', foldl_stepOf]⟩
            have hksl : ks.toList = linears.toList.filterMap parseL := by
              rw [hks', foldl_pushSome]
              rfl
            have hcsl : cs.toList = calls.toList.filterMap parseC := by
              rw [hcs', foldl_pushSome]
              rfl
            have hg0 : tile.getD 0 Json.null = a0 := by
              rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hl]
              rfl
            have hg1 : tile.getD 1 Json.null = a1 := by
              rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hl]
              rfl
            simp only [callWork, Id.run, h1, h3, h4, hm, hg0, hg1, hr, hc, Option.getD_some]
            rw [filterMap_comp_eq _ _ parseL linears.toList ?lin, ← hksl,
              filterMap_comp_eq _ _ (fun x => (parseC x).bind fun y =>
                (ks.toList.find? (·.1 == y.1)).map fun t => (m / r * (t.2.2 / c), r * c * t.2.1)) calls.toList ?call,
              ← List.filterMap_filterMap (f := parseC), ← hcsl, List.map_filterMap]
            · show List.filterMap _ _ = List.filterMap _ _
              congr 1
              funext y
              simp only [itemOf, Option.map_map, Array.find?_toList]
              rfl
            case lin =>
              intro x hx
              have hs := hLs x hx
              cases x with
              | arr a =>
                simp only [parseL] at hs ⊢
                split at hs
                · rename_i s k n hal
                  simp only [hal]
                  cases hk : PouwCheck.nat? k <;> cases hn : PouwCheck.nat? n <;> simp [hk, hn] at hs ⊢
                · simp at hs
              | _ => simp [parseL] at hs
            case call =>
              intro x hx
              have hs := hCs x hx
              cases x with
              | arr a =>
                simp only [parseC] at hs ⊢
                split at hs
                · rename_i s b hal
                  simp only [hal]
                  cases hb : PouwCheck.nat? b <;> simp [hb] at hs ⊢
                · simp at hs
              | _ => simp [parseC] at hs
        · obtain ⟨_, h', -⟩ := bind_ok h
          exact absurd h' (refuse_ne _ _)
      · exact absurd h (refuse_ne _ _)
    · exact absurd h (refuse_ne _ _)
  · obtain ⟨_, h', -⟩ := bind_ok h
    exact absurd h' (refuse_ne _ _)

theorem fold_inv (r c : Nat) (items : List (Nat × Nat)) : Inv r c items (items.foldl (add r c) (#[], 0)) :=
  rev_ind (P := fun items => Inv r c items (items.foldl (add r c) (#[], 0))) (inv_nil r c)
    (fun l x ih => by
      rw [List.foldl_append, List.foldl_cons, List.foldl_nil]
      exact inv_step ih x) items

/-- **`strata` tiles the window's units, call by call** (`StrataTile`). -/
theorem strataTile : StrataTile := by
  intro layout ss h
  obtain ⟨r, c, items, hcw, hss⟩ := strata_ok h
  have hinv := fold_inv r c items
  generalize items.foldl (add r c) (#[], 0) = st at hinv hss
  subst hss
  have hstart : ∀ i, callStart layout i = start items i := fun i => by
    unfold callStart start
    rw [hcw, ← List.map_take, List.map_map]
    rfl
  have hlen : (callWork layout).length = items.length := by rw [hcw, List.length_map]
  have hend : callStart layout (callWork layout).length = st.2 := by
    rw [hstart, hlen, start_length, hinv.at_eq]
  refine ⟨fun i hi u hlo hup => ?_, fun u hu s hs => ?_, ?_, ?_, hinv.tmpl⟩
  · have hi' : i < items.length := hlen ▸ hi
    rw [hstart] at hlo hup
    obtain ⟨⟨s, hs, hh⟩, hall⟩ := hinv.cover i hi' u hlo hup
    refine ⟨⟨s, Array.mem_toList_iff.1 hs, hh⟩, fun s hs hh => ?_⟩
    rw [hall s (Array.mem_toList_iff.2 hs) hh]
    simp [hcw]
  · rw [hend] at hu
    exact not_holds_above (hinv.rng s (Array.mem_toList_iff.2 hs)) hu
  · rw [hend, hinv.size]
  · rw [totalWork_list, hinv.work, hcw, List.map_map]
    rfl

end Properties.Pouw.Strata
