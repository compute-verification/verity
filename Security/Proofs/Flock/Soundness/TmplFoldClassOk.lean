import Proofs.Flock.Soundness.TmplFoldClass

/-!
# A class's fold is its copies' reference value (PRF-133, part 3, stage 2)

`refVal`: the slot-`s` value of one copy of a Definition, its own column `y` at `A + B · posIn y` and its inputs at
`ip`, summed over its rows and, copy by copy, its calls' (each read at the position it reads, `readAt`). `foldDef_exp`:
an explicit class's `expVal` is its copies' `refVal`s.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

/-- One entry group's value: references `cc.1` of a row of `E` weight `cc.2 · ex`, each read at `R v`. -/
noncomputable def grpVal (pt : FoldPt) (s : ℕ) (ex : F128) (R : ℕ → ℕ) (cc : Array ℕ × F128) : GF128 :=
  (cc.1.toList.map fun v => of (cc.2 * ex) * Gs pt s (R v)).sum

/-- One row's value, its `A` entries weighted by `α` and its `B` entries by 1. -/
noncomputable def rowVal (pt : FoldPt) (s : ℕ) (ex : F128) (R : ℕ → ℕ) (r : ℕ × Array ℕ × Array ℕ) : GF128 :=
  ([(r.2.1, pt.alpha), (r.2.2, (1 : F128))].map (grpVal pt s ex R)).sum

/-- Where Definition `d`'s copy (own column `y` at `A + B · posIn y`, inputs at `ip`) reads reference `v`. -/
def readAt (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (d A B : ℕ) (ip : List ℕ) (v : ℕ) : ℕ :=
  if v < (t.defs.getD d default).cols then A + B * Flock.Tmpl.posIn t lays (t.defs.size + 1) d v
  else if v < (t.defs.getD d default).cols + (t.defs.getD d default).ins then
    ip.getD (v - (t.defs.getD d default).cols) pt.qc
  else pt.qc

/-- **The reference value** of Definition `d`'s copy at `(A, B)` with inputs at `ip`. -/
noncomputable def refVal (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s : ℕ) : ℕ → ℕ → ℕ → ℕ → List ℕ → GF128
  | 0, _, _, _, _ => 0
  | fuel + 1, d, A, B, ip =>
    ((t.defs.getD d default).rows.toList.map fun r =>
        rowVal pt s (pt.e (A + B * Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)) (readAt t lays pt d A B ip) r).sum +
      ∑ i ∈ Finset.range (t.defs.getD d default).calls.size,
        ∑ j ∈ Finset.range ((t.defs.getD d default).calls.getD i default).count,
          refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_
            (A + B * (((lays.getD d default).places.getD i default).nb +
              j * ((lays.getD d default).places.getD i default).cs))
            (B * ((lays.getD d default).places.getD i default).ms)
            (((t.defs.getD d default).calls.getD i default).bound (readAt t lays pt d A B ip) j)

theorem pure_ok {α ε : Type} {a b : α} (h : (pure a : Except ε α) = .ok b) : a = b := by
  cases h; rfl

theorem getD_replicate {α : Type} (n : ℕ) (a : α) (i : ℕ) : (Array.replicate n a).getD i a = a := by
  by_cases h : i < n <;> simp [h]

theorem getD_replicate₂ (ins nk t k : ℕ) :
    ((Array.replicate ins (Array.replicate nk (0 : F128))).getD t #[]).getD k 0 = 0 := by
  by_cases h : t < ins <;> by_cases hk : k < nk <;> simp [h, hk]

theorem shape_init (ins nk : ℕ) :
    Shape ins nk ⟨Array.replicate 64 0, 0, 0, Array.replicate ins (Array.replicate nk 0), 0⟩ :=
  ⟨by simp, by simp, fun t ht => by simp [ht]⟩

theorem expVal_init (s ins nk : ℕ) (iv : ℕ → ℕ → GF128) :
    expVal s ins nk iv ⟨Array.replicate 64 0, 0, 0, Array.replicate ins (Array.replicate nk 0), 0⟩ = 0 := by
  unfold expVal
  simp only [getD_replicate, getD_replicate₂, of_zero', zero_mul, Finset.sum_const_zero, add_zero]

theorem list_sum_comm {α : Type} (l : List α) (n : ℕ) (f : ℕ → α → GF128) :
    (l.map fun r => ∑ k ∈ Finset.range n, f k r).sum = ∑ k ∈ Finset.range n, (l.map (f k)).sum := by
  induction l with
  | nil => simp
  | cons a l ih => rw [List.map_cons, List.sum_cons, ih, ← Finset.sum_add_distrib]; simp

/-- The reads of copy `k` of an explicit class `(β, h, as)`, its inputs at `ip k`. -/
def readExp (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (d β h : ℕ) (as : Array ℕ) (ip : ℕ → List ℕ) (k v : ℕ) : ℕ :=
  if v < (t.defs.getD d default).cols then as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d v)
  else if v < (t.defs.getD d default).cols + (t.defs.getD d default).ins then
    (ip k).getD (v - (t.defs.getD d default).cols) pt.qc
  else pt.qc

theorem readAt_exp (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (d β h : ℕ) (as : Array ℕ) (ip : ℕ → List ℕ) (k : ℕ) :
    readAt t lays pt d (as.getD k 0 + 2 ^ β * h) (2 ^ β) (ip k) = readExp t lays pt d β h as ip k := by
  funext v
  unfold readAt readExp
  rw [Nat.mul_add, Nat.add_assoc]

/-- An explicit class's own rows: the rows loop of `foldDef` adds each row's value over the class copies. -/
theorem foldDef_exp_rows {t : Tmpl} {lays : Array Lay} {pt : FoldPt} {s : ℕ} (hs : s < 64) {fuel d β h : ℕ}
    {as : Array ℕ} {o : Out} (hd : foldDef t lays pt (fuel + 1) d β h as = .ok o) (h6 : ¬ 6 ≤ β)
    (ip : ℕ → List ℕ) :
    Shape (t.defs.getD d default).ins as.size o ∧
      expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) o =
        ((t.defs.getD d default).rows.toList.map fun r => ∑ k ∈ Finset.range as.size,
          rowVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)))
            (readExp t lays pt d β h as ip k) r).sum + 0 := by
  rw [foldDef] at hd
  simp only [h6, decide_false, ↓reduceIte] at hd
  obtain ⟨oR, hR, hd⟩ := except_bind_ok hd
  have key := forIn_array_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
    (Shape (t.defs.getD d default).ins as.size) (fun r => ∑ k ∈ Finset.range as.size,
      rowVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)))
        (readExp t lays pt d β h as ip k) r) hR (shape_init _ _) ?_
  · obtain ⟨hq, hsum⟩ := key
    rw [expVal_init, zero_add] at hsum
    sorry
  · intro ⟨x, ra, rb⟩ _ b r hr hq
    dsimp only at hr
    obtain ⟨b1, hk, hr⟩ := except_bind_ok hr
    cases hr
    have key := forIn_range_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
      (Shape (t.defs.getD d default).ins as.size) (fun k =>
        rowVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x)))
          (readExp t lays pt d β h as ip k) (x, ra, rb)) hk hq ?_
    · obtain ⟨hq1, hs1⟩ := key
      exact ⟨_, rfl, hq1, hs1⟩
    intro k hk' b r hr hq
    obtain ⟨b2, hc, hr⟩ := except_bind_ok hr
    cases hr
    have key := forIn_list_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
      (Shape (t.defs.getD d default).ins as.size)
      (grpVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x)))
        (readExp t lays pt d β h as ip k)) _ hc hq ?_
    · obtain ⟨hq2, hs2⟩ := key
      exact ⟨_, rfl, hq2, hs2⟩
    intro cc _ b r hr hq
    obtain ⟨b3, hv, hr⟩ := except_bind_ok hr
    cases hr
    have key := forIn_array_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
      (Shape (t.defs.getD d default).ins as.size)
      (fun v => of (cc.2 * pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x))) *
        Gs pt s (readExp t lays pt d β h as ip k v)) hv hq ?_
    · obtain ⟨hq3, hs3⟩ := key
      exact ⟨_, rfl, hq3, hs3⟩
    intro v _ b r hr hq
    cases hr
    refine ⟨_, rfl, ?_⟩
    refine (ref_exp pt hs hk' _ _ _
      (of (cc.2 * pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x))) *
        Gs pt s (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d v))) hq
      (fun _ => own_exp pt hs _ _ _ hq)).imp id (fun e => e.trans ?_)
    unfold readExp
    split_ifs <;> rfl

/-! ## A copy's inputs, bound -/

theorem bound_map (c : Flock.Tmpl.Call) (g : ℕ → ℕ) (j : ℕ) : c.bound g j = (c.bound id j).map g := by
  simp [Flock.Tmpl.Call.bound, List.map_flatMap]

theorem zipIdx_sum (F : ℕ → ℕ → GF128) : ∀ (L : List ℕ) (n : ℕ),
    ((L.zipIdx n).map fun x => F x.1 x.2).sum = ∑ t ∈ Finset.range L.length, F (L.getD t 0) (n + t)
  | [], n => by simp
  | a :: L, n => by
    rw [List.zipIdx_cons, List.map_cons, List.sum_cons, zipIdx_sum F L (n + 1), List.length_cons,
      Finset.sum_range_succ']
    simp only [List.getD_cons_succ, List.getD_cons_zero, Nat.add_zero]
    rw [add_comm (F a n)]
    congr 1
    exact Finset.sum_congr rfl fun t _ => by rw [show n + 1 + t = n + (t + 1) by omega]

theorem getD_bound (c : Flock.Tmpl.Call) (g : ℕ → ℕ) (j t d : ℕ) (ht : t < (c.bound id j).length) :
    (c.bound g j).getD t d = g ((c.bound id j).getD t 0) := by
  rw [bound_map]
  simp [List.getD_eq_getElem?_getD, ht]

/-- **An explicit class binds a callee's copy**: for each bound input `(v, t')` and each class copy `k`, `v` read with
weight `val t' k`. -/
theorem bind_exp (pt : FoldPt) {s ins nk cols : ℕ} (hs : s < 64) (ip : ℕ → List ℕ) (q : ℕ → ℕ → ℕ)
    (val : ℕ → ℕ → F128) : ∀ (L : List (ℕ × ℕ)) (o : Out), Shape ins nk o →
    Shape ins nk (L.foldl (fun o x => (List.range nk).foldl (fun o k =>
        o.ref pt false cols ins x.1 k (val x.2 k) fun y o =>
          { p := addAt o.p (q k y) (val x.2 k * pt.g (q k y)), s := o.s, c := o.c, u := o.u, work := o.work }) o) o) ∧
      expVal s ins nk (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) (L.foldl (fun o x => (List.range nk).foldl
        (fun o k => o.ref pt false cols ins x.1 k (val x.2 k) fun y o =>
          { p := addAt o.p (q k y) (val x.2 k * pt.g (q k y)), s := o.s, c := o.c, u := o.u, work := o.work }) o) o) =
        expVal s ins nk (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) o +
          (L.map fun x => ∑ k ∈ Finset.range nk, of (val x.2 k) * Gs pt s
            (if x.1 < cols then q k x.1 else if x.1 < cols + ins then (ip k).getD (x.1 - cols) pt.qc else pt.qc)).sum := by
  intro L o hsh
  refine foldl_sum _ _ _ L o hsh fun x _ b hb => ?_
  have key := foldl_sum (expVal s ins nk (fun t' k => Gs pt s ((ip k).getD t' pt.qc))) (Shape ins nk)
    (fun k => of (val x.2 k) * Gs pt s
      (if x.1 < cols then q k x.1 else if x.1 < cols + ins then (ip k).getD (x.1 - cols) pt.qc else pt.qc))
    (List.range nk) b hb ?_
  · rw [list_range_sum] at key
    exact key
  intro k hk b hb
  refine (ref_exp pt hs (List.mem_range.mp hk) _ _ _ (of (val x.2 k) * Gs pt s (q k x.1)) hb
    (fun _ => own_exp pt hs _ _ _ hb)).imp id (fun e => e.trans ?_)
  split_ifs <;> rfl

/-! ## Lane bits, and the layout facts the fold reads -/

/-- The most lane bits a copy of Definition `d` adds below it. -/
def lanes (t : Tmpl) (lays : Array Lay) : ℕ → ℕ → ℕ
  | 0, _ => 0
  | fuel + 1, d => (List.range (t.defs.getD d default).calls.size).foldl (fun n i =>
      max n ((if ((lays.getD d default).places.getD i default).ms = 1 then 0
        else ((lays.getD d default).places.getD i default).ms.log2) +
          lanes t lays fuel ((t.defs.getD d default).calls.getD i default).def_)) 0

theorem le_foldl_max (F : ℕ → ℕ) : ∀ (l : List ℕ) (n : ℕ),
    n ≤ l.foldl (fun n i => max n (F i)) n ∧ ∀ x ∈ l, F x ≤ l.foldl (fun n i => max n (F i)) n
  | [], n => ⟨le_refl _, fun x hx => absurd hx List.not_mem_nil⟩
  | a :: l, n => by
    obtain ⟨h1, h2⟩ := le_foldl_max F l (max n (F a))
    refine ⟨(le_max_left _ _).trans h1, fun x hx => ?_⟩
    rcases List.mem_cons.mp hx with rfl | hx
    · exact (le_max_right _ _).trans h1
    · exact h2 x hx

theorem lanes_call (t : Tmpl) (lays : Array Lay) (fuel d i : ℕ) (hi : i < (t.defs.getD d default).calls.size) :
    (if ((lays.getD d default).places.getD i default).ms = 1 then 0
        else ((lays.getD d default).places.getD i default).ms.log2) +
      lanes t lays fuel ((t.defs.getD d default).calls.getD i default).def_ ≤ lanes t lays (fuel + 1) d :=
  (le_foldl_max _ _ 0).2 i (List.mem_range.mpr hi)

/-- What the fold reads of the layout: each copy binds its callee's inputs, and a run of lanes takes `2^m ≥ count`
lanes, its copies side by side (`cs = 1`). -/
structure LayOk (t : Tmpl) (lays : Array Lay) : Prop where
  bound : ∀ d i j, i < (t.defs.getD d default).calls.size →
    (((t.defs.getD d default).calls.getD i default).bound id j).length =
      (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).ins
  lanes : ∀ d i, i < (t.defs.getD d default).calls.size → ((lays.getD d default).places.getD i default).ms ≠ 1 →
    ((lays.getD d default).places.getD i default).cs = 1 ∧
      2 ^ ((lays.getD d default).places.getD i default).ms.log2 = ((lays.getD d default).places.getD i default).ms ∧
      ((t.defs.getD d default).calls.getD i default).count ≤ ((lays.getD d default).places.getD i default).ms

end FlockSoundness.TmplFold
