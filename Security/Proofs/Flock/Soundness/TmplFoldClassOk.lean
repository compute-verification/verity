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

/-! ## A copy's inputs, bound -/

theorem bound_map (c : Flock.Tmpl.Call) (g : ℕ → ℕ) (j : ℕ) : c.bound g j = (c.bound id j).map g := by
  simp [Flock.Tmpl.Call.bound, List.map_flatMap, Function.comp_def]

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
  refine foldl_sum (expVal s ins nk (fun t' k => Gs pt s ((ip k).getD t' pt.qc))) (Shape ins nk)
    (fun x => ∑ k ∈ Finset.range nk, of (val x.2 k) * Gs pt s
      (if x.1 < cols then q k x.1 else if x.1 < cols + ins then (ip k).getD (x.1 - cols) pt.qc else pt.qc))
    (f := fun o x => (List.range nk).foldl (fun o k =>
        o.ref pt false cols ins x.1 k (val x.2 k) fun y o =>
          { p := addAt o.p (q k y) (val x.2 k * pt.g (q k y)), s := o.s, c := o.c, u := o.u, work := o.work }) o)
    L o hsh fun x _ b hb => ?_
  have key := foldl_sum (expVal s ins nk (fun t' k => Gs pt s ((ip k).getD t' pt.qc))) (Shape ins nk)
    (fun k => of (val x.2 k) * Gs pt s
      (if x.1 < cols then q k x.1 else if x.1 < cols + ins then (ip k).getD (x.1 - cols) pt.qc else pt.qc))
    (f := fun o k => o.ref pt false cols ins x.1 k (val x.2 k) fun y o =>
      { p := addAt o.p (q k y) (val x.2 k * pt.g (q k y)), s := o.s, c := o.c, u := o.u, work := o.work })
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
  (le_foldl_max (fun i => (if ((lays.getD d default).places.getD i default).ms = 1 then 0
      else ((lays.getD d default).places.getD i default).ms.log2) +
        lanes t lays fuel ((t.defs.getD d default).calls.getD i default).def_) _ 0).2 i (List.mem_range.mpr hi)

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

theorem getD_lanes (as : Array ℕ) (B n k : ℕ) (hk : k < as.size * n) :
    ((Array.range (as.size * n)).map fun i => as.getD (i / n) 0 + B * (i % n)).getD k 0 =
      as.getD (k / n) 0 + B * (k % n) := by
  simp [Array.getD_eq_getD_getElem?, hk]

theorem divmod_div {k c j : ℕ} (hc0 : 0 < c) (hj : j < c) : (k * c + j) / c = k := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hc0, Nat.div_eq_of_lt hj, Nat.zero_add]

theorem divmod_mod {k c j : ℕ} (hj : j < c) : (k * c + j) % c = j := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]

theorem lane_lt {a β j m : ℕ} (ha : a < 2 ^ β) (hj : j < 2 ^ m) : a + 2 ^ β * j < 2 ^ (β + m) := by
  rw [Nat.pow_add]
  have := Nat.mul_le_mul_left (2 ^ β) (Nat.succ_le_of_lt hj)
  rw [Nat.mul_succ] at this
  omega

/-- A run's lanes, copy by copy: index `k · c + j` is lane `j` of copy `k`. -/
theorem sum_lanes {M : Type} [AddCommMonoid M] (G : ℕ → M) (c : ℕ) : ∀ n,
    ∑ i ∈ Finset.range (n * c), G i = ∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, G (k * c + j)
  | 0 => by simp
  | n + 1 => by rw [Nat.add_mul, Nat.one_mul, Finset.sum_range_add, sum_lanes G c n, Finset.sum_range_succ]

theorem expVal_setP {s ins nk : ℕ} (iv : ℕ → ℕ → GF128) (o : Out) (p' : Array F128) (δ : GF128)
    (h : of (p'.getD s 0) = of (o.p.getD s 0) + δ) (w : ℕ) :
    expVal s ins nk iv { p := p', s := o.s, c := o.c, u := o.u, work := w } = expVal s ins nk iv o + δ := by
  simp only [expVal, h]
  ring

theorem sum3_comm {M : Type} [AddCommMonoid M] (A : ℕ → ℕ → M) (B : ℕ → ℕ → ℕ → M) (n c T : ℕ) :
    ∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, A k j +
        ∑ j ∈ Finset.range c, ∑ t ∈ Finset.range T, ∑ k ∈ Finset.range n, B j t k =
      ∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, (A k j + ∑ t ∈ Finset.range T, B j t k) := by
  have e : ∑ j ∈ Finset.range c, ∑ t ∈ Finset.range T, ∑ k ∈ Finset.range n, B j t k =
      ∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, ∑ t ∈ Finset.range T, B j t k :=
    (Finset.sum_congr rfl fun j _ => Finset.sum_comm).trans Finset.sum_comm
  rw [e, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun k _ => Finset.sum_add_distrib.symm

/-! ## What a class's fold means -/

/-- An explicit class `(d, β, h, as)`: with copy `k`'s inputs at `ip k`, its `expVal` is its copies' `refVal`s. -/
def ExpOk (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s fuel d β h : ℕ) (as : Array ℕ) (o : Out) : Prop :=
  Shape (t.defs.getD d default).ins as.size o ∧ ∀ ip : ℕ → List ℕ,
    expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) o =
      ∑ k ∈ Finset.range as.size, refVal t lays pt s fuel d (as.getD k 0 + 2 ^ β * h) (2 ^ β) (ip k)

/-- A summed class `(d, β, h)`: no `p`, and for every low part `a`, its `sumVal` times `E(a)` is the copy's `refVal`. -/
def SumOk (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s fuel d β h : ℕ) (o : Out) : Prop :=
  Shape (t.defs.getD d default).ins 1 o ∧ PZero o ∧ ∀ a < 2 ^ β, ∀ ip : List ℕ,
    of (eLow pt β a) * sumVal (t.defs.getD d default).ins (gLow pt s β a) (Gs pt s pt.qc)
      (fun t' => Gs pt s (ip.getD t' pt.qc)) o = refVal t lays pt s fuel d (a + 2 ^ β * h) (2 ^ β) ip

/-- Every class's fold at depth `fuel` means what it should, while its lane bits fit the point. -/
def ClassOk (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s fuel : ℕ) : Prop :=
  ∀ d β h (as : Array ℕ) (o : Out), foldDef t lays pt fuel d β h as = .ok o →
    β + lanes t lays fuel d ≤ 6 + pt.rho.size → β + lanes t lays fuel d ≤ 6 + pt.r.size →
    (¬ 6 ≤ β → (∀ k < as.size, as.getD k 0 < 2 ^ β) → ExpOk t lays pt s fuel d β h as o) ∧
      (6 ≤ β → SumOk t lays pt s fuel d β h o)

/-- **An explicit class's step**: its fold at `fuel + 1`, from every class's at `fuel`. -/
theorem exp_step {t : Tmpl} {lays : Array Lay} {pt : FoldPt} {s : ℕ} (hs : s < 64) (hlay : LayOk t lays) {fuel : ℕ}
    (ih : ClassOk t lays pt s fuel) {d β h : ℕ} {as : Array ℕ} {o : Out}
    (hd : foldDef t lays pt (fuel + 1) d β h as = .ok o)
    (hf1 : β + lanes t lays (fuel + 1) d ≤ 6 + pt.rho.size) (hf2 : β + lanes t lays (fuel + 1) d ≤ 6 + pt.r.size)
    (h6 : ¬ 6 ≤ β) (has : ∀ k < as.size, as.getD k 0 < 2 ^ β) : ExpOk t lays pt s (fuel + 1) d β h as o := by
  have key : ∀ ip : ℕ → List ℕ, Shape (t.defs.getD d default).ins as.size o ∧
      expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) o =
        ∑ k ∈ Finset.range as.size, refVal t lays pt s (fuel + 1) d (as.getD k 0 + 2 ^ β * h) (2 ^ β) (ip k) := by
    intro ip
    rw [foldDef] at hd
    simp only [h6, decide_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte] at hd
    obtain ⟨oR, hR, hd⟩ := except_bind_ok hd
    have keyR := forIn_array_sum
      (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
      (Shape (t.defs.getD d default).ins as.size) (fun r => ∑ k ∈ Finset.range as.size,
        rowVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)))
          (readExp t lays pt d β h as ip k) r) hR (shape_init _ _) ?_
    · obtain ⟨hqR, hsR⟩ := keyR
      rw [expVal_init, zero_add] at hsR
      obtain ⟨oC, hC, hd⟩ := except_bind_ok hd
      cases pure_ok hd
      have keyC := forIn_range_sum
        (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
        (Shape (t.defs.getD d default).ins as.size) (fun i => ∑ k ∈ Finset.range as.size,
          ∑ j ∈ Finset.range ((t.defs.getD d default).calls.getD i default).count,
            refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_
              (as.getD k 0 + 2 ^ β * h + 2 ^ β * (((lays.getD d default).places.getD i default).nb +
                j * ((lays.getD d default).places.getD i default).cs))
              (2 ^ β * ((lays.getD d default).places.getD i default).ms)
              (((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip k) j)) hC hqR ?_
      · obtain ⟨hqC, hsC⟩ := keyC
        refine ⟨hqC, ?_⟩
        rw [hsC, hsR]
        simp only [refVal, readAt_exp]
        rw [Finset.sum_add_distrib]
        refine congrArg₂ (· + ·) ?_ Finset.sum_comm
        rw [list_sum_comm]
        refine Finset.sum_congr rfl fun k _ => ?_
        simp only [Nat.mul_add, Nat.add_assoc]
      intro i hi b r hr hq
      by_cases hm : ((lays.getD d default).places.getD i default).ms = 1
      · have hm' : (((lays.getD d default).places.getD i default).ms == 1) = true := by rw [hm]; rfl
        rw [if_pos hm'] at hr
        obtain ⟨bJ, hJ, hr⟩ := except_bind_ok hr
        cases hr
        have keyJ := forIn_range_sum
          (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
          (Shape (t.defs.getD d default).ins as.size) (fun j => ∑ k ∈ Finset.range as.size,
            refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_
              (as.getD k 0 + 2 ^ β * h + 2 ^ β * (((lays.getD d default).places.getD i default).nb +
                j * ((lays.getD d default).places.getD i default).cs))
              (2 ^ β * ((lays.getD d default).places.getD i default).ms)
              (((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip k) j)) hJ hq ?_
        · obtain ⟨hq1, hs1⟩ := keyJ
          exact ⟨_, rfl, hq1, by rw [hs1, Finset.sum_comm]⟩
        intro j _ b r hr hq
        obtain ⟨ch, hch, hr⟩ := except_bind_ok hr
        cases hr
        refine ⟨_, rfl, ?_⟩
        have hl := lanes_call t lays fuel d i hi
        rw [if_pos hm, zero_add] at hl
        obtain ⟨hchS, hchV⟩ := (ih _ β _ as ch hch (by omega) (by omega)).1 h6 has
        refine (bind_exp pt hs ip
          (fun k y => as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y))
          (fun t' k => (ch.u.getD t' #[]).getD k 0)
          (((t.defs.getD d default).calls.getD i default).bound id j).zipIdx _ ?_).imp id (fun e => e.trans ?_)
        · exact (add_exp hs (fun _ _ => 0) 1 1 hq hchS.1).1
        refine (congrArg (· + _) (add_exp hs (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) 1 1 hq hchS.1).2).trans ?_
        refine (add_assoc _ _ _).trans (congrArg (HAdd.hAdd _) ?_)
        have eR : ∀ k, refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_
            (as.getD k 0 + 2 ^ β * h + 2 ^ β * (((lays.getD d default).places.getD i default).nb +
              j * ((lays.getD d default).places.getD i default).cs))
            (2 ^ β * ((lays.getD d default).places.getD i default).ms)
            (((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip k) j) =
          refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_
            (as.getD k 0 + 2 ^ β * (h + ((lays.getD d default).places.getD i default).nb +
              j * ((lays.getD d default).places.getD i default).cs)) (2 ^ β)
            (((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip k) j) := by
          intro k
          rw [hm, Nat.mul_one, show as.getD k 0 + 2 ^ β * h + 2 ^ β * (((lays.getD d default).places.getD i default).nb +
              j * ((lays.getD d default).places.getD i default).cs) = as.getD k 0 + 2 ^ β * (h + ((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs) by ring]
        refine Eq.trans ?_ ((Finset.sum_congr rfl fun k _ => eR k).trans
          (hchV fun k => ((t.defs.getD d default).calls.getD i default).bound
            (readExp t lays pt d β h as ip k) j).symm).symm
        unfold expVal
        refine congrArg (HAdd.hAdd _) ?_
        refine (zipIdx_sum (fun v t' => ∑ k ∈ Finset.range as.size, of ((ch.u.getD t' #[]).getD k 0) *
            Gs pt s (readExp t lays pt d β h as ip k v)) _ 0).trans ?_
        rw [hlay.bound d i j hi]
        refine Finset.sum_congr rfl fun t' ht' => ?_
        rw [zero_add]
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [getD_bound _ _ _ _ _ (by rw [hlay.bound d i j hi]; exact Finset.mem_range.mp ht')]
      · have hm' : ¬ ((((lays.getD d default).places.getD i default).ms == 1) = true) := by rw [beq_iff_eq]; exact hm
        rw [if_neg hm'] at hr
        by_cases hal : ((h + ((lays.getD d default).places.getD i default).nb) % ((lays.getD d default).places.getD i default).ms != 0) = true
        · rw [if_pos hal] at hr; cases hr
        rw [if_neg hal] at hr
        have hal' : (h + ((lays.getD d default).places.getD i default).nb) % ((lays.getD d default).places.getD i default).ms = 0 := by simpa using hal
        obtain ⟨hcs, hpow, hcnt⟩ := hlay.lanes d i hi hm
        have hl := lanes_call t lays fuel d i hi
        rw [if_neg hm] at hl
        have hB : 2 ^ β * ((lays.getD d default).places.getD i default).ms = 2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2) := by rw [Nat.pow_add, hpow]
        have hh : h + ((lays.getD d default).places.getD i default).nb = ((lays.getD d default).places.getD i default).ms * ((h + ((lays.getD d default).places.getD i default).nb) / ((lays.getD d default).places.getD i default).ms) :=
          (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hal')).symm
        by_cases hlt : β + ((lays.getD d default).places.getD i default).ms.log2 < 6
        · rw [if_pos hlt] at hr
          obtain ⟨ch, hch, hr⟩ := except_bind_ok hr
          obtain ⟨b4, hJ, hr⟩ := except_bind_ok hr
          cases hr
          refine ⟨_, rfl, ?_⟩
          have hsz : ((Array.range (as.size * ((t.defs.getD d default).calls.getD i default).count)).map fun i' => as.getD (i' / ((t.defs.getD d default).calls.getD i default).count) 0 + 2 ^ β * (i' % ((t.defs.getD d default).calls.getD i default).count)).size = as.size * ((t.defs.getD d default).calls.getD i default).count := by simp
          obtain ⟨hchS, hchV⟩ := (ih _ _ _ _ ch hch (by omega) (by omega)).1 (by omega) (fun k hk => by
            rw [hsz] at hk
            have hc0 : 0 < ((t.defs.getD d default).calls.getD i default).count := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hk; omega
            rw [getD_lanes _ _ _ _ hk]
            exact lane_lt (has _ (Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hk)))
              (lt_of_lt_of_le (Nat.mod_lt _ hc0) (hcnt.trans_eq hpow.symm)))
          have keyJ := forIn_range_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc))) (Shape (t.defs.getD d default).ins as.size)
            (fun j => ∑ t' ∈ Finset.range (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).ins, ∑ k ∈ Finset.range as.size,
              of ((ch.u.getD t' #[]).getD (k * ((t.defs.getD d default).calls.getD i default).count + j) 0) *
                Gs pt s (readExp t lays pt d β h as ip k ((((t.defs.getD d default).calls.getD i default).bound id j).getD t' 0)))
            hJ (add_exp hs (fun _ _ => 0) 0 0 hq hchS.1).1 ?_
          · obtain ⟨hqJ, hsJ⟩ := keyJ
            refine ⟨hqJ, hsJ.trans ?_⟩
            refine (congrArg₂ (· + ·) (add_exp hs (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) 0 0 hq hchS.1).2 rfl).trans ?_
            refine (add_assoc _ _ _).trans (congrArg (HAdd.hAdd _) ?_)
            have eR : ∀ k ∈ Finset.range as.size, ∀ j ∈ Finset.range ((t.defs.getD d default).calls.getD i default).count,
                refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_ (as.getD k 0 + 2 ^ β * h + 2 ^ β * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs))
                  (2 ^ β * ((lays.getD d default).places.getD i default).ms) (((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip k) j) =
                refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_ (((Array.range (as.size * ((t.defs.getD d default).calls.getD i default).count)).map fun i' => as.getD (i' / ((t.defs.getD d default).calls.getD i default).count) 0 + 2 ^ β * (i' % ((t.defs.getD d default).calls.getD i default).count)).getD (k * ((t.defs.getD d default).calls.getD i default).count + j) 0 +
                  2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2) * ((h + ((lays.getD d default).places.getD i default).nb) / ((lays.getD d default).places.getD i default).ms)) (2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2)) ((fun i' => ((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip (i' / ((t.defs.getD d default).calls.getD i default).count)) (i' % ((t.defs.getD d default).calls.getD i default).count)) (k * ((t.defs.getD d default).calls.getD i default).count + j)) := by
              intro k hk j hj
              have hk := Finset.mem_range.mp hk
              have hj := Finset.mem_range.mp hj
              have hc0 : 0 < ((t.defs.getD d default).calls.getD i default).count := by omega
              have hlt' : k * ((t.defs.getD d default).calls.getD i default).count + j < as.size * ((t.defs.getD d default).calls.getD i default).count :=
                lt_of_lt_of_le (Nat.add_lt_add_left hj _) (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hk)
              have e1 : ((Array.range (as.size * ((t.defs.getD d default).calls.getD i default).count)).map fun i' => as.getD (i' / ((t.defs.getD d default).calls.getD i default).count) 0 + 2 ^ β * (i' % ((t.defs.getD d default).calls.getD i default).count)).getD (k * ((t.defs.getD d default).calls.getD i default).count + j) 0 = as.getD k 0 + 2 ^ β * j := by
                rw [getD_lanes _ _ _ _ hlt', divmod_div hc0 hj, divmod_mod hj]
              have e2 : as.getD k 0 + 2 ^ β * h + 2 ^ β * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs) =
                  as.getD k 0 + 2 ^ β * j + 2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2) * ((h + ((lays.getD d default).places.getD i default).nb) / ((lays.getD d default).places.getD i default).ms) := by
                rw [hcs, Nat.mul_one, ← hB, Nat.mul_assoc, ← hh]; ring
              rw [e1, e2, hB]
              simp only [divmod_div hc0 hj, divmod_mod hj]
            refine Eq.trans ?_ ((Finset.sum_congr rfl fun k hk => Finset.sum_congr rfl fun j hj => eR k hk j hj).trans
              ((sum_lanes (fun i' => refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_ (((Array.range (as.size * ((t.defs.getD d default).calls.getD i default).count)).map fun i' => as.getD (i' / ((t.defs.getD d default).calls.getD i default).count) 0 + 2 ^ β * (i' % ((t.defs.getD d default).calls.getD i default).count)).getD i' 0 +
                  2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2) * ((h + ((lays.getD d default).places.getD i default).nb) / ((lays.getD d default).places.getD i default).ms)) (2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2)) ((fun i' => ((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip (i' / ((t.defs.getD d default).calls.getD i default).count)) (i' % ((t.defs.getD d default).calls.getD i default).count)) i')) _ _).symm.trans
                ((congrArg (fun n => ∑ i' ∈ Finset.range n, refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_ (((Array.range (as.size * ((t.defs.getD d default).calls.getD i default).count)).map fun i' => as.getD (i' / ((t.defs.getD d default).calls.getD i default).count) 0 + 2 ^ β * (i' % ((t.defs.getD d default).calls.getD i default).count)).getD i' 0 +
                  2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2) * ((h + ((lays.getD d default).places.getD i default).nb) / ((lays.getD d default).places.getD i default).ms)) (2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2)) ((fun i' => ((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip (i' / ((t.defs.getD d default).calls.getD i default).count)) (i' % ((t.defs.getD d default).calls.getD i default).count)) i')) hsz).symm.trans
                  (hchV (fun i' => ((t.defs.getD d default).calls.getD i default).bound (readExp t lays pt d β h as ip (i' / ((t.defs.getD d default).calls.getD i default).count)) (i' % ((t.defs.getD d default).calls.getD i default).count))).symm))).symm
            unfold expVal
            refine congrArg (HAdd.hAdd _) ?_
            rw [hsz]
            refine Finset.sum_comm.trans (Finset.sum_congr rfl fun t' ht' => ?_)
            rw [sum_lanes]
            refine Finset.sum_comm.trans (Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun j hj => ?_)
            have hj := Finset.mem_range.mp hj
            simp only [divmod_div (show 0 < ((t.defs.getD d default).calls.getD i default).count by omega) hj, divmod_mod hj]
            rw [getD_bound _ _ _ _ _ (by rw [hlay.bound d i j hi]; exact Finset.mem_range.mp ht')]
          intro j _ b r hr hq
          cases hr
          refine ⟨_, rfl, ?_⟩
          refine (bind_exp pt hs ip
            (fun k y => as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y))
            (fun t' k => (ch.u.getD t' #[]).getD (k * ((t.defs.getD d default).calls.getD i default).count + j) 0)
            (((t.defs.getD d default).calls.getD i default).bound id j).zipIdx _ hq).imp id (fun e => e.trans ?_)
          refine congrArg (HAdd.hAdd _) ((zipIdx_sum (fun v t' => ∑ k ∈ Finset.range as.size,
            of ((ch.u.getD t' #[]).getD (k * ((t.defs.getD d default).calls.getD i default).count + j) 0) * Gs pt s (readExp t lays pt d β h as ip k v)) _ 0).trans ?_)
          rw [hlay.bound d i j hi]
          exact Finset.sum_congr rfl fun t' _ => by rw [zero_add]
        · rw [if_neg hlt] at hr
          obtain ⟨ch, hch, hr⟩ := except_bind_ok hr
          obtain ⟨pP, hP, hr⟩ := except_bind_ok hr
          obtain ⟨b4, hJ, hr⟩ := except_bind_ok hr
          cases hr
          refine ⟨_, rfl, ?_⟩
          obtain ⟨hchS, hchZ, hchV⟩ := (ih _ _ _ _ ch hch (by omega) (by omega)).2 (by omega)
          have hX := add_exp hs (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) 0 0 hq hchS.1
          have keyP := forIn_range_sum (fun p : Array F128 => of (p.getD s 0)) (fun p : Array F128 => p.size = 64)
            (fun k => ∑ j ∈ Finset.range ((t.defs.getD d default).calls.getD i default).count, ((if (as.getD k 0 + 2 ^ β * j) % 64 = s then of ((pt.lw.getD ((as.getD k 0 + 2 ^ β * j) % 64) 0 * Tmpl.eqAt pt.rho 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64)) * Tmpl.eqAt pt.r 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64) * ch.s) else 0) +
                (if pt.qc % 64 = s then of ((pt.lw.getD ((as.getD k 0 + 2 ^ β * j) % 64) 0 * Tmpl.eqAt pt.rho 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64)) * ch.c * pt.g pt.qc) else 0))) hP hX.1.1 ?_
          · obtain ⟨hqP, hsP⟩ := keyP
            have keyJ := forIn_range_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc))) (Shape (t.defs.getD d default).ins as.size)
              (fun j => ∑ t' ∈ Finset.range (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).ins, ∑ k ∈ Finset.range as.size,
                of ((pt.lw.getD ((as.getD k 0 + 2 ^ β * j) % 64) 0 * Tmpl.eqAt pt.rho 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64)) * (ch.u.getD t' #[]).getD 0 0) *
                  Gs pt s (readExp t lays pt d β h as ip k ((((t.defs.getD d default).calls.getD i default).bound id j).getD t' 0)))
              hJ ⟨hqP, hX.1.2⟩ ?_
            · obtain ⟨hqJ, hsJ⟩ := keyJ
              refine ⟨hqJ, hsJ.trans ?_⟩
              refine (congrArg₂ (· + ·) ((expVal_setP _ _ _ _ hsP _).trans
                (congrArg₂ (· + ·) (hX.2.trans (by rw [hchZ s hs, add_zero])) rfl)) rfl).trans ?_
              refine (add_assoc _ _ _).trans (congrArg (HAdd.hAdd _) ?_)
              refine (sum3_comm _ _ _ _ _).trans (Finset.sum_congr rfl fun k hk => Finset.sum_congr rfl fun j hj => ?_)
              have hk' := Finset.mem_range.mp hk
              have hj' := Finset.mem_range.mp hj
              have e2 : as.getD k 0 + 2 ^ β * h + 2 ^ β * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs) =
                  as.getD k 0 + 2 ^ β * j + 2 ^ (β + ((lays.getD d default).places.getD i default).ms.log2) * ((h + ((lays.getD d default).places.getD i default).nb) / ((lays.getD d default).places.getD i default).ms) := by
                rw [hcs, Nat.mul_one, ← hB, Nat.mul_assoc, ← hh]; ring
              rw [e2, hB, ← hchV _ (lane_lt (has k hk') (lt_of_lt_of_le hj' (hcnt.trans_eq hpow.symm))) _]
              unfold sumVal
              rw [mul_add, Finset.mul_sum]
              refine congrArg₂ (· + ·) ?_ (Finset.sum_congr rfl fun t' ht' => ?_)
              · unfold gLow Gs eLow
                rw [mul_add]
                refine congrArg₂ (· + ·) ?_ ?_ <;> split_ifs <;> simp only [of_mul, mul_zero, zero_mul] <;> ring
              · rw [getD_bound _ _ _ _ _ (by rw [hlay.bound d i j hi]; exact Finset.mem_range.mp ht')]
                simp only [eLow, of_mul, mul_assoc]
            intro j _ b r hr hq
            cases hr
            refine ⟨_, rfl, ?_⟩
            refine (bind_exp pt hs ip
              (fun k y => as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y))
              (fun t' k => (pt.lw.getD ((as.getD k 0 + 2 ^ β * j) % 64) 0 * Tmpl.eqAt pt.rho 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64)) * (ch.u.getD t' #[]).getD 0 0)
              (((t.defs.getD d default).calls.getD i default).bound id j).zipIdx _ hq).imp id (fun e => e.trans ?_)
            refine congrArg (HAdd.hAdd _) ((zipIdx_sum (fun v t' => ∑ k ∈ Finset.range as.size,
              of ((pt.lw.getD ((as.getD k 0 + 2 ^ β * j) % 64) 0 * Tmpl.eqAt pt.rho 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64)) * (ch.u.getD t' #[]).getD 0 0) * Gs pt s (readExp t lays pt d β h as ip k v)) _ 0).trans ?_)
            rw [hlay.bound d i j hi]
            exact Finset.sum_congr rfl fun t' _ => by rw [zero_add]
          intro k _ p r hr hq
          obtain ⟨p5, hJ', hr⟩ := except_bind_ok hr
          cases hr
          refine ⟨_, rfl, ?_⟩
          refine forIn_range_sum (fun p : Array F128 => of (p.getD s 0)) (fun p : Array F128 => p.size = 64)
            (fun j => ((if (as.getD k 0 + 2 ^ β * j) % 64 = s then of ((pt.lw.getD ((as.getD k 0 + 2 ^ β * j) % 64) 0 * Tmpl.eqAt pt.rho 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64)) * Tmpl.eqAt pt.r 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64) * ch.s) else 0) +
                (if pt.qc % 64 = s then of ((pt.lw.getD ((as.getD k 0 + 2 ^ β * j) % 64) 0 * Tmpl.eqAt pt.rho 0 ((β + ((lays.getD d default).places.getD i default).ms.log2) - 6) ((as.getD k 0 + 2 ^ β * j) / 64)) * ch.c * pt.g pt.qc) else 0))) hJ' hq fun j _ p r hr hq => ?_
          cases hr
          refine ⟨_, rfl, by rw [addAt_size, addAt_size]; exact hq, ?_⟩
          rw [addAt_get _ _ _ (by rw [addAt_size, hq]; exact hs), addAt_get _ _ _ (by rw [hq]; exact hs), add_assoc]
    · intro ⟨x, ra, rb⟩ _ b r hr hq
      dsimp only at hr
      obtain ⟨b1, hk, hr⟩ := except_bind_ok hr
      cases hr
      have key := forIn_range_sum
        (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
        (Shape (t.defs.getD d default).ins as.size) (fun k =>
          rowVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x)))
            (readExp t lays pt d β h as ip k) (x, ra, rb)) hk hq ?_
      · obtain ⟨hq1, hs1⟩ := key
        exact ⟨_, rfl, hq1, hs1⟩
      intro k hk' b r hr hq
      obtain ⟨b2, hc, hr⟩ := except_bind_ok hr
      cases hr
      have key := forIn_list_sum
        (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
        (Shape (t.defs.getD d default).ins as.size)
        (grpVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x)))
          (readExp t lays pt d β h as ip k)) _ hc hq ?_
      · obtain ⟨hq2, hs2⟩ := key
        exact ⟨_, rfl, hq2, hs2⟩
      intro cc _ b r hr hq
      obtain ⟨b3, hv, hr⟩ := except_bind_ok hr
      cases hr
      have key := forIn_array_sum
        (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
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
  exact ⟨(key fun _ => []).1, fun ip => (key ip).2⟩

end FlockSoundness.TmplFold
