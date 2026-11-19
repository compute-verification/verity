module

public import Proofs.Properties.InferenceOnly.RecordDeclared
public import Proofs.Properties.InferenceOnly.RecordSlopes
public import Mathlib.Tactic

/-!
# What `nodesHeld` holds of the declarations that fill a node's rows

`Record.covers` takes the declarations of one template port that meet a range's units, each cut down to the units and
rows it shares with the range, checks each cut-down rectangle at its corner, and checks that their areas sum to the
range's. Pairwise disjoint declarations whose rectangles sum so tile the range.

* `covers_cell`: every unit of the range and row of the port's cut is in a declaration whose cut-down rectangle was
  checked.
* `withStarts_mem`: the range at index `j` has among its template's units the ones before it, as `UnitAt` counts them.
* `readOk_source`: a registered read whose corner `ReadOk` checked has, at every unit and row of its cut-down
  rectangle, the source `nodesHeld` names for that unit and row: the corner's source and the read's slopes.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open InferenceOnlyVerifier.Record (covers withStarts slopesOf slopesAbout)
open InferenceOnlyVerifier.Record.SecurityProofs (rowSlope unitDigits accum_outer accum_inner sloped_of_slopesOf
  sloped_of_slopesAbout writerAt_rows)
open Spot.RecordText (Decl)
open Spot.Wires (accum prodR)
open Flock.TemplateQuery (TRange)

/-! ## The cut-down rectangles tile the range -/

/-- The declaration `d` cut down to template `tid`'s port `port`, the units `[u0, u0 + M)` and the rows below `kc`:
`(d, ua, ub, ra, rb)`, its units `[ua, ub)` and rows `[ra, rb)`, none where it meets none of them. -/
def tileOf (u0 M kc : Nat) (tid port : String) (d : Decl) : Option (Decl × Nat × Nat × Nat × Nat) :=
  if d.w.reader == tid && d.w.port == port && max u0 d.w.start < min (u0 + M) (d.w.start + d.w.readerUnits) &&
      d.w.cut0 < min kc (d.w.cut0 + d.w.cuts) then
    some (d, max u0 d.w.start, min (u0 + M) (d.w.start + d.w.readerUnits), d.w.cut0, min kc (d.w.cut0 + d.w.cuts))
  else none

theorem covers_eq (ds : List Decl) (tid port : String) (u0 M kc : Nat) (ok : Decl → Nat → Nat → Nat → Nat → Bool) :
    covers ds tid port u0 M kc ok =
      (((ds.filterMap (tileOf u0 M kc tid port)).all fun (d, ua, ub, ra, rb) => ok d ua ub ra rb) &&
        (((ds.filterMap (tileOf u0 M kc tid port)).map fun (_, ua, ub, ra, rb) => (ub - ua) * (rb - ra)).sum ==
          M * kc)) := rfl

/-- The cells of a cut-down rectangle. -/
def cell (t : Decl × Nat × Nat × Nat × Nat) : Finset (Nat × Nat) :=
  (Finset.Ico t.2.1 t.2.2.1) ×ˢ (Finset.Ico t.2.2.2.1 t.2.2.2.2)

/-- The area of a cut-down rectangle. -/
def area (t : Decl × Nat × Nat × Nat × Nat) : Nat := (t.2.2.1 - t.2.1) * (t.2.2.2.2 - t.2.2.2.1)

/-- The cells of a list of cut-down rectangles. -/
def cellsOf (L : List (Decl × Nat × Nat × Nat × Nat)) : Finset (Nat × Nat) :=
  L.foldr (fun t acc => cell t ∪ acc) ∅

theorem mem_cellsOf {L : List (Decl × Nat × Nat × Nat × Nat)} {p : Nat × Nat} :
    p ∈ cellsOf L ↔ ∃ t ∈ L, p ∈ cell t := by
  induction L with
  | nil => simp [cellsOf]
  | cons a L ih =>
    have e : cellsOf (a :: L) = cell a ∪ cellsOf L := rfl
    rw [e, Finset.mem_union, ih]
    constructor
    · rintro (h | ⟨t, ht, h⟩)
      · exact ⟨a, List.mem_cons_self, h⟩
      · exact ⟨t, List.mem_cons_of_mem _ ht, h⟩
    · rintro ⟨t, ht, h⟩
      rcases List.mem_cons.1 ht with rfl | ht'
      · exact Or.inl h
      · exact Or.inr ⟨t, ht', h⟩

theorem card_cell (t : Decl × Nat × Nat × Nat × Nat) : (cell t).card = area t := by
  unfold cell area
  rw [Finset.card_product, Nat.card_Ico, Nat.card_Ico]

theorem card_cellsOf : ∀ L : List (Decl × Nat × Nat × Nat × Nat),
    L.Pairwise (fun a b => Disjoint (cell a) (cell b)) → (cellsOf L).card = (L.map area).sum
  | [], _ => by simp [cellsOf]
  | a :: L, h => by
    rw [List.pairwise_cons] at h
    have e : cellsOf (a :: L) = cell a ∪ cellsOf L := rfl
    have hd : Disjoint (cell a) (cellsOf L) := by
      rw [Finset.disjoint_left]
      intro p hp hq
      obtain ⟨t, ht, hpt⟩ := mem_cellsOf.1 hq
      exact Finset.disjoint_left.1 (h.1 t ht) hp hpt
    rw [e, Finset.card_union_of_disjoint hd, card_cell, card_cellsOf L h.2, List.map_cons, List.sum_cons]

theorem tileOf_some {u0 M kc : Nat} {tid port : String} {d : Decl} {t : Decl × Nat × Nat × Nat × Nat}
    (h : tileOf u0 M kc tid port d = some t) :
    t = (d, max u0 d.w.start, min (u0 + M) (d.w.start + d.w.readerUnits), d.w.cut0, min kc (d.w.cut0 + d.w.cuts)) ∧
      d.w.reader = tid ∧ d.w.port = port ∧ max u0 d.w.start < min (u0 + M) (d.w.start + d.w.readerUnits) ∧
      d.w.cut0 < min kc (d.w.cut0 + d.w.cuts) := by
  unfold tileOf at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
    cases h
    exact ⟨rfl, hc.1.1.1, hc.1.1.2, hc.1.2, hc.2⟩
  · cases h

/-- Two declarations no row of which is both's have disjoint cut-down cells. -/
theorem cells_disjoint {u0 M kc : Nat} {tid port : String} {a b : Decl} {ta tb : Decl × Nat × Nat × Nat × Nat}
    (hd : Spot.Wires.disjoint a.w b.w = true) (ha : tileOf u0 M kc tid port a = some ta)
    (hb : tileOf u0 M kc tid port b = some tb) : Disjoint (cell ta) (cell tb) := by
  obtain ⟨rfl, ra, pa, -⟩ := tileOf_some ha
  obtain ⟨rfl, rb, pb, -⟩ := tileOf_some hb
  unfold Spot.Wires.disjoint at hd
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq] at hd
  rw [Finset.disjoint_left]
  intro p hp hq
  unfold cell at hp hq
  simp only [Finset.mem_product, Finset.mem_Ico] at hp hq
  rcases hd with ((((hd | hd) | hd) | hd) | hd) | hd
  · exact hd (ra.trans rb.symm)
  · exact hd (pa.trans pb.symm)
  · omega
  · omega
  · omega
  · omega

/-- **Every unit of a range and row of its port's cut is in a declaration whose cut-down rectangle `covers` checked**,
among declarations no two of which fill one row. -/
theorem covers_cell {ds : List Decl} {tid port : String} {u0 M kc : Nat}
    {ok : Decl → Nat → Nat → Nat → Nat → Bool} (hdis : Spot.Wires.pairwiseDisjoint (ds.map (·.w)) = true)
    (h : covers ds tid port u0 M kc ok = true) {u c : Nat} (hu0 : u0 ≤ u) (hu : u < u0 + M) (hc : c < kc) :
    ∃ d ∈ ds, d.w.reader = tid ∧ d.w.port = port ∧ d.w.start ≤ u ∧ u < d.w.start + d.w.readerUnits ∧
      d.w.cut0 ≤ c ∧ c < d.w.cut0 + d.w.cuts ∧
      ok d (max u0 d.w.start) (min (u0 + M) (d.w.start + d.w.readerUnits)) d.w.cut0
        (min kc (d.w.cut0 + d.w.cuts)) = true := by
  rw [covers_eq] at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨hall, hsum⟩ := h
  have hpw : (ds.filterMap (tileOf u0 M kc tid port)).Pairwise (fun a b => Disjoint (cell a) (cell b)) := by
    rw [List.pairwise_filterMap]
    have h1 := (Spot.Wires.pairwiseDisjoint_iff _).1 hdis
    rw [List.pairwise_map] at h1
    refine h1.imp fun {a b} hab => ?_
    intro ta hta tb htb
    exact cells_disjoint hab hta htb
  have hcard := card_cellsOf _ hpw
  have hbox : cellsOf (ds.filterMap (tileOf u0 M kc tid port)) ⊆ (Finset.Ico u0 (u0 + M)) ×ˢ (Finset.Ico 0 kc) := by
    intro p hp
    obtain ⟨t, ht, hpt⟩ := mem_cellsOf.1 hp
    obtain ⟨d, hd, hdt⟩ := List.mem_filterMap.1 ht
    obtain ⟨rfl, -, -, -, -⟩ := tileOf_some hdt
    unfold cell at hpt
    simp only [Finset.mem_product, Finset.mem_Ico] at hpt ⊢
    omega
  have hsum' : ((ds.filterMap (tileOf u0 M kc tid port)).map area).sum = M * kc := by
    rw [← hsum]
    unfold area
    rfl
  have hfull : cellsOf (ds.filterMap (tileOf u0 M kc tid port)) = (Finset.Ico u0 (u0 + M)) ×ˢ (Finset.Ico 0 kc) := by
    refine Finset.eq_of_subset_of_card_le hbox ?_
    rw [Finset.card_product, Nat.card_Ico, Nat.card_Ico, hcard, hsum']
    simp
  have hcell : (u, c) ∈ cellsOf (ds.filterMap (tileOf u0 M kc tid port)) := by
    rw [hfull]
    simp only [Finset.mem_product, Finset.mem_Ico]
    omega
  obtain ⟨t, ht, hpt⟩ := mem_cellsOf.1 hcell
  obtain ⟨d, hd, hdt⟩ := List.mem_filterMap.1 ht
  obtain ⟨rfl, hr, hp, -, -⟩ := tileOf_some hdt
  unfold cell at hpt
  simp only [Finset.mem_product, Finset.mem_Ico] at hpt
  refine ⟨d, hd, hr, hp, ?_, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · omega
  · exact List.all_eq_true.1 hall _ ht

/-! ## The range's first unit -/

/-- The units of template `tid` among the ranges `l`. -/
def cumCount (l : List TRange) (tid : String) : Nat :=
  (l.filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0

private theorem foldl_add_count (l : List TRange) (a : Nat) :
    l.foldl (fun a r => a + r.count) a = a + l.foldl (fun a r => a + r.count) 0 := by
  induction l generalizing a with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons, List.foldl_cons, ih (a + x.count), ih (0 + x.count)]
    omega

theorem cumCount_snoc (pre : List TRange) (r : TRange) (t : String) :
    cumCount (pre ++ [r]) t = cumCount pre t + (if r.template == t then r.count else 0) := by
  unfold cumCount
  rw [List.filter_append, List.foldl_append]
  by_cases h : r.template == t
  · simp only [List.filter_cons, h, List.filter_nil, List.foldl_cons, List.foldl_nil, if_true]
  · simp only [List.filter_cons, h, List.filter_nil, List.foldl_nil, Bool.false_eq_true, if_false, Nat.add_zero]

private theorem lookup_filter_ne {α : Type} (l : List (String × α)) (s t : String) (h : (t == s) = false) :
    (l.filter fun x => x.1 != s).lookup t = l.lookup t := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨k, v⟩ := x
    by_cases hk : (k != s) = true
    · simp only [List.filter_cons, hk, if_true, List.lookup_cons]
      by_cases hkt : (t == k) = true
      · simp [hkt]
      · simp [hkt, ih]
    · have hk' : k = s := by simpa using hk
      subst hk'
      have hts : (t == k) = false := h
      simp only [List.filter_cons, hk, Bool.false_eq_true, if_false, List.lookup_cons, hts]
      exact ih

theorem foldW (step : (List (String × Nat) × List (TRange × Nat)) → TRange →
      (List (String × Nat) × List (TRange × Nat)))
    (hstep : ∀ acc r, step acc r = ((r.template, (acc.1.lookup r.template).getD 0 + r.count) ::
      acc.1.filter (fun x => x.1 != r.template), acc.2 ++ [(r, (acc.1.lookup r.template).getD 0)])) :
    ∀ (l pre : List TRange) (acc : List (String × Nat) × List (TRange × Nat)),
      (∀ t, (acc.1.lookup t).getD 0 = cumCount pre t) →
      (∀ x ∈ acc.2, x ∈ (l.foldl step acc).2) ∧
        ∀ (k : Nat) (hk : k < l.length), (l[k], cumCount (pre ++ l.take k) l[k].template) ∈ (l.foldl step acc).2
  | [], pre, acc, _ => ⟨fun x hx => hx, fun k hk => absurd hk (Nat.not_lt_zero _)⟩
  | r :: l, pre, acc, hacc => by
    have hacc' : ∀ t, ((step acc r).1.lookup t).getD 0 = cumCount (pre ++ [r]) t := by
      intro t
      rw [hstep, cumCount_snoc, ← hacc t]
      simp only [List.lookup_cons]
      by_cases ht : (t == r.template) = true
      · have ht' : t = r.template := by simpa using ht
        subst ht'
        simp [Nat.add_comm]
      · have ht1 : (t == r.template) = false := by simpa using ht
        have ht2 : (r.template == t) = false := by
          have : t ≠ r.template := by simpa using ht1
          simpa using this.symm
        rw [ht1, lookup_filter_ne _ _ _ ht1]
        simp [ht2]
    obtain ⟨ih1, ih2⟩ := foldW step hstep l (pre ++ [r]) (step acc r) hacc'
    rw [List.foldl_cons]
    refine ⟨fun x hx => ih1 x ?_, fun k hk => ?_⟩
    · rw [hstep]
      exact List.mem_append_left _ hx
    · cases k with
      | zero =>
        apply ih1
        show (r, cumCount (pre ++ []) r.template) ∈ (step acc r).2
        rw [List.append_nil, ← hacc r.template, hstep]
        exact List.mem_append_right _ (List.mem_singleton_self _)
      | succ k =>
        have := ih2 k (Nat.lt_of_succ_lt_succ hk)
        simpa [List.append_assoc] using this

theorem withStarts_mem_list (rs : List TRange) {j : Nat} (hj : j < rs.length) :
    (rs[j], cumCount (rs.take j) rs[j].template) ∈ withStarts rs.toArray := by
  unfold withStarts
  have := (foldW (fun (acc : List (String × Nat) × List (TRange × Nat)) r =>
    let b := (acc.1.lookup r.template).getD 0
    ((r.template, b + r.count) :: acc.1.filter (·.1 != r.template), acc.2 ++ [(r, b)]))
    (fun _ _ => rfl) rs [] ([], []) (fun t => rfl)).2 j hj
  simpa using this

/-- **The range at index `j` has among its template's units the ones before it**, as `UnitAt` counts them. -/
theorem withStarts_mem (rs : Array TRange) {j : Nat} (hj : j < rs.size) :
    (rs[j]!, ((rs.extract 0 j).filter fun r => r.template == rs[j]!.template).foldl (fun a r => a + r.count) 0) ∈
      withStarts rs := by
  have h := withStarts_mem_list rs.toList (j := j) (by simpa using hj)
  have e1 : rs.toList[j] = rs[j]! := by
    rw [getElem!_pos rs j hj]
    simp
  rw [e1] at h
  have e2 : ((rs.extract 0 j).filter fun r => r.template == rs[j]!.template).foldl (fun a r => a + r.count) 0 =
      cumCount (rs.toList.take j) rs[j]!.template := by
    unfold cumCount
    rw [← Array.foldl_toList, Array.toList_filter, Array.toList_extract]
    simp
  rw [e2]
  simpa using h

/-! ## A registered read's source at a unit and row -/

/-- A registered read of `p` that fills a declared node's rows, checked at the corner `(ua, ra)` of its cut-down
rectangle (the closure `nodesHeld` gives `covers`): the units' slopes agree with the node's, a cut's rows step by one
row's words, and the source at the corner is the node's leaf there, `A / 16` the node's first leaf in the parameter's
16-bit leaves, `so` and `si` its steps over a block of `q` units and over a unit, `wL` a row's 16-bit leaves. -/
def ReadOk (p : String) (q u0 : Nat) (so si A : Int) (wL : Nat) (d : Decl) (ua ub ra rb : Nat) : Bool :=
  let sr : Int := if d.w.cuts > 1 then (d.w.pairs.head?.map (·.2)).getD 0 else 0
  let unitsOk := if q == 1 then (match slopesOf d.w with
      | some (su, _) => ub - ua ≤ 1 || su == so
      | none => false) else
    (ua - d.w.start) % q == 0 && (ua - u0) % q == 0 && match slopesAbout d.w q with
      | some (si', so') => si'.all (· == si) && so'.all (· == so)
      | none => false
  match d.fill with
  | .read p' =>
    p' == p && unitsOk && (rb - ra ≤ 1 || sr == (wL : Int)) &&
      d.w.writerAt ((ua - d.w.start) * d.w.cuts + (ra - d.w.cut0)) ==
        A / 16 + (((ua - u0) / q : Nat) : Int) * so + (((ua - u0) % q : Nat) : Int) * si + ((ra * wL : Nat) : Int)
  | _ => false

/-- **A registered read `ReadOk` holds of at its corner has, at every unit and row of its cut-down rectangle, the
source the node's leaf there names**: `A / 16` and the units' and rows' steps. -/
theorem readOk_source {p p' : String} {q u0 : Nat} {so si A : Int} {wL : Nat} {d : Decl} {ua ub ra rb : Nat}
    (hq : 0 < q) (hfill : d.fill = .read p') (hshape : ReadShaped d.w)
    (hok : ReadOk p q u0 so si A wL d ua ub ra rb = true)
    (hua : ua = max u0 d.w.start) (hra : ra = d.w.cut0) (hub : ub ≤ d.w.start + d.w.readerUnits)
    (hrb : rb ≤ d.w.cut0 + d.w.cuts) (hrlt : ra < rb)
    {u c : Nat} (hu1 : ua ≤ u) (hu2 : u < ub) (hc1 : ra ≤ c) (hc2 : c < rb) :
    p' = p ∧ d.w.writerAt ((u - d.w.start) * d.w.cuts + (c - d.w.cut0)) =
      A / 16 + (((u - u0) / q : Nat) : Int) * so + (((u - u0) % q : Nat) : Int) * si + ((c * wL : Nat) : Int) := by
  unfold ReadOk at hok
  simp only [hfill, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at hok
  obtain ⟨⟨⟨hpp, hunits⟩, hsr⟩, hcorner⟩ := hok
  refine ⟨hpp, ?_⟩
  obtain ⟨hlen, hcut⟩ := hshape
  have hcuts : 0 < d.w.cuts := by omega
  have hstart : d.w.start ≤ ua := by omega
  have hu0ua : u0 ≤ ua := by omega
  obtain ⟨hw1, hw2⟩ := writerAt_rows d.w hlen hcut (v := u - d.w.start) (c := c - d.w.cut0) (by omega)
  obtain ⟨hw1', -⟩ := writerAt_rows d.w hlen hcut (v := ua - d.w.start) (c := 0) hcuts
  have hra0 : ra - d.w.cut0 = 0 := by omega
  rw [hra0, Nat.add_zero] at hcorner
  have hcorner' : d.w.writerAt ((ua - d.w.start) * d.w.cuts + 0) =
      A / 16 + (((ua - u0) / q : Nat) : Int) * so + (((ua - u0) % q : Nat) : Int) * si + ((ra * wL : Nat) : Int) :=
    hcorner
  rw [hw1']  at hcorner'
  rw [hw1]
  have hv : u - d.w.start < d.w.readerUnits := by omega
  have hv0 : ua - d.w.start < d.w.readerUnits := by omega
  have hRU : d.w.readerUnits = Spot.Wires.prodR (unitDigits d.w) := hw2
  have hrowterm : ((c - d.w.cut0 : Nat) : Int) * rowSlope d.w = ((c - d.w.cut0 : Nat) : Int) * (wL : Int) := by
    by_cases h1 : rb - ra ≤ 1
    · have hc0 : c - d.w.cut0 = 0 := by omega
      rw [hc0]
      simp
    · have h2 : (if d.w.cuts > 1 then (d.w.pairs.head?.map (·.2)).getD 0 else 0) = (wL : Int) := by
        rcases hsr with h | h
        · exact absurd h h1
        · exact h
      have h3 : rowSlope d.w = (wL : Int) := h2
      rw [h3]
  have e2 : ((c * wL : Nat) : Int) = ((ra * wL : Nat) : Int) + ((c - d.w.cut0 : Nat) : Int) * (wL : Int) := by
    have : (c : Int) = (ra : Int) + ((c - d.w.cut0 : Nat) : Int) := by omega
    push_cast
    rw [this]
    ring
  by_cases hq1 : q = 1
  · subst hq1
    simp only [beq_self_eq_true, if_true] at hunits
    cases hs : slopesOf d.w with
    | none =>
      rw [hs] at hunits
      simp at hunits
    | some pr =>
      obtain ⟨su, sr'⟩ := pr
      rw [hs] at hunits
      simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hunits
      obtain ⟨-, hsl⟩ := sloped_of_slopesOf hs
      have ha := accum_outer 1 0 su Nat.one_pos (unitDigits d.w) 1 hsl (Nat.one_dvd 1) Nat.one_pos
      have hx : accum (unitDigits d.w) (u - d.w.start) = su * ((u - d.w.start : Nat) : Int) := by
        have := ha (u - d.w.start) (by rw [← hRU]; exact hv)
        simpa using this
      have hx0 : accum (unitDigits d.w) (ua - d.w.start) = su * ((ua - d.w.start : Nat) : Int) := by
        have := ha (ua - d.w.start) (by rw [← hRU]; exact hv0)
        simpa using this
      have hsu : su * (((u - d.w.start : Nat) : Int) - ((ua - d.w.start : Nat) : Int)) =
          so * (((u - d.w.start : Nat) : Int) - ((ua - d.w.start : Nat) : Int)) := by
        rcases hunits with h | h
        · have : u = ua := by omega
          subst this
          simp
        · rw [h]
      have e1 : (((u - u0 : Nat) : Int)) = ((u - d.w.start : Nat) : Int) - ((ua - d.w.start : Nat) : Int) +
          ((ua - u0 : Nat) : Int) := by omega
      simp only [Nat.div_one, Nat.mod_one, Nat.cast_zero, zero_mul, add_zero] at hcorner' ⊢
      linear_combination hcorner' + hx - hx0 + hsu + hrowterm - so * e1 - e2
  · simp only [beq_iff_eq, hq1, if_false, Bool.and_eq_true, decide_eq_true_eq] at hunits
    obtain ⟨⟨hmod1, hmod2⟩, hsl⟩ := hunits
    cases hs : slopesAbout d.w q with
    | none =>
      rw [hs] at hsl
      simp at hsl
    | some pr =>
      obtain ⟨si', so'⟩ := pr
      rw [hs] at hsl
      simp only [Bool.and_eq_true, beq_iff_eq] at hsl
      obtain ⟨-, hslp⟩ := sloped_of_slopesAbout hs (by simpa using hsl.1) (by simpa using hsl.2)
      have ha := accum_inner q si so hq (unitDigits d.w) 1 hslp Nat.one_pos (Nat.one_dvd q)
      have hx := ha (u - d.w.start) (by rw [← hRU]; exact hv)
      have hx0 := ha (ua - d.w.start) (by rw [← hRU]; exact hv0)
      simp only [Nat.one_mul] at hx hx0
      obtain ⟨a, ha1⟩ : ∃ a, ua - d.w.start = q * a :=
        ⟨(ua - d.w.start) / q, (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hmod1)).symm⟩
      obtain ⟨a', ha2⟩ : ∃ a', ua - u0 = q * a' :=
        ⟨(ua - u0) / q, (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hmod2)).symm⟩
      obtain ⟨e, he⟩ : ∃ e, u - ua = e := ⟨_, rfl⟩
      have hv1 : u - d.w.start = q * a + e := by omega
      have hu1' : u - u0 = q * a' + e := by omega
      have m1 : (u - d.w.start) % q = e % q := by rw [hv1]; exact Nat.mul_add_mod q a e
      have d1 : (u - d.w.start) / q = a + e / q := by rw [hv1]; exact Nat.mul_add_div hq a e
      have d0 : (ua - d.w.start) / q = a := by rw [ha1]; exact Nat.mul_div_cancel_left a hq
      have d2 : (ua - u0) / q = a' := by rw [ha2]; exact Nat.mul_div_cancel_left a' hq
      have m3 : (u - u0) % q = e % q := by rw [hu1']; exact Nat.mul_add_mod q a' e
      have d3 : (u - u0) / q = a' + e / q := by rw [hu1']; exact Nat.mul_add_div hq a' e
      rw [m1, d1] at hx
      rw [hmod1, d0] at hx0
      rw [hmod2, d2] at hcorner'
      rw [m3, d3]
      push_cast at hx hx0 hcorner' e2 ⊢
      linear_combination hcorner' + hx - hx0 + hrowterm - e2

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
