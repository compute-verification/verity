module

public import Proofs.Properties.Nci.RecordJson

/-!
# The grids the record check reads of a record's inference blocks

`RecordText.gridsOf` reads, per template, the grids its units' slots are read from: the record's inference entries
of that template, sorted, tiling its units, neighbours of one row length merged. This module states what that gives
(`gridsOf_ok`): the class has at least one slot, there is one list of grids a template, in the templates' order,
and a template's grids are in unit order, each after the one before it, cover its units and no other, and each has a
row length of at least one.

The steps: ranges that tile are in order (`tiles_ordered`), merging neighbours keeps a tiling and every row length
(`merged_tiles`), and an entry's row length is a product of radices of at least one (`gridOf_q`).
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open Lean (Json)
open NciCheck.RecordText (Grid gridOf merged gridsOf Tmpl)
open Spot.Wires (tiles)

/-- The ranges of a list of grids, as `Spot.Wires.tiles` takes them. -/
def rangesOf (gs : List Grid) : List (Nat × Nat) := gs.map fun g => (g.start, g.units)

/-- **Grids that tile are in unit order**: each lies inside the units they tile and has a unit, and each starts
where the one before it ends, so after it. -/
theorem tiles_ordered : ∀ (gs : List Grid) (cur units : Nat), tiles (rangesOf gs) cur units = true →
    cur ≤ units ∧ (∀ g ∈ gs, cur ≤ g.start ∧ g.start + g.units ≤ units ∧ 0 < g.units) ∧
      gs.Pairwise (fun a b => a.start + a.units ≤ b.start)
  | [], cur, units, h => by
    simp only [rangesOf, List.map_nil, tiles, beq_iff_eq] at h
    exact ⟨by omega, fun _ hg => (by cases hg), List.Pairwise.nil⟩
  | g :: gs, cur, units, h => by
    simp only [rangesOf, List.map_cons, tiles, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    obtain ⟨⟨h0, hn⟩, hr⟩ := h
    obtain ⟨h1, h2, h3⟩ := tiles_ordered gs (cur + g.units) units hr
    refine ⟨by omega, fun x hx => ?_, List.pairwise_cons.2 ⟨fun b hb => ?_, h3⟩⟩
    · rcases List.mem_cons.1 hx with rfl | hx
      · exact ⟨by omega, by omega, hn⟩
      · have := h2 x hx
        exact ⟨by omega, this.2.1, this.2.2⟩
    · have := h2 b hb
      omega

/-- **Grids that tile cover the units they tile.** -/
theorem tiles_covers : ∀ (gs : List Grid) (cur units u : Nat), tiles (rangesOf gs) cur units = true → cur ≤ u →
    u < units → ∃ g ∈ gs, g.start ≤ u ∧ u < g.start + g.units
  | [], cur, units, u, h, h1, h2 => by
    simp only [rangesOf, List.map_nil, tiles, beq_iff_eq] at h
    omega
  | g :: gs, cur, units, u, h, h1, h2 => by
    simp only [rangesOf, List.map_cons, tiles, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    obtain ⟨⟨h0, hn⟩, hr⟩ := h
    by_cases hu : u < g.start + g.units
    · exact ⟨g, List.mem_cons_self, by omega, hu⟩
    · obtain ⟨x, hx, hx'⟩ := tiles_covers gs (cur + g.units) units u hr (by omega) h2
      exact ⟨x, List.mem_cons_of_mem _ hx, hx'⟩

/-- Tiling up to a grid, then that grid. -/
theorem tiles_snoc : ∀ (gs : List Grid) (g : Grid) (cur units : Nat),
    tiles (rangesOf (gs ++ [g])) cur units = true ↔
      tiles (rangesOf gs) cur g.start = true ∧ 0 < g.units ∧ g.start + g.units = units
  | [], g, cur, units => by
    simp only [rangesOf, List.nil_append, List.map_cons, List.map_nil, tiles, Bool.and_eq_true, beq_iff_eq,
      decide_eq_true_eq]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3⟩
      exact ⟨h1.symm, h2, by omega⟩
    · rintro ⟨h1, h2, h3⟩
      exact ⟨⟨h1.symm, h2⟩, by omega⟩
  | x :: gs, g, cur, units => by
    have ih := tiles_snoc gs g (cur + x.units) units
    simp only [rangesOf] at ih
    simp only [rangesOf, List.cons_append, List.map_cons, tiles, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, ih]
    constructor
    · rintro ⟨h1, h2, h3, h4⟩
      exact ⟨⟨h1, h2⟩, h3, h4⟩
    · rintro ⟨⟨h1, h2⟩, h3, h4⟩
      exact ⟨h1, h2, h3, h4⟩

/-- One step of `merged`: the grids so far, last first, with the next grid merged into the last or put before it. -/
def mergeStep (acc : List Grid) (g : Grid) : List Grid :=
  match acc with
  | last :: before => if last.q == g.q then { last with units := last.units + g.units } :: before else g :: acc
  | [] => [g]

theorem mergeStep_cons (last : Grid) (before : List Grid) (g : Grid) :
    mergeStep (last :: before) g =
      if last.q == g.q then { last with units := last.units + g.units } :: before else g :: last :: before :=
  rfl

theorem merged_eq (gs : List Grid) : merged gs = (gs.foldl mergeStep []).reverse := rfl

/-- The fold of `merged` keeps a tiling, from where it started to where the grids it has read end, and every row
length it holds is one it was given. -/
theorem mergeStep_fold (Q : Nat → Prop) (cur : Nat) : ∀ (gs : List Grid) (acc : List Grid) (mid units : Nat),
    tiles (rangesOf acc.reverse) cur mid = true → (∀ g ∈ acc, Q g.q) → tiles (rangesOf gs) mid units = true →
    (∀ g ∈ gs, Q g.q) →
    tiles (rangesOf (gs.foldl mergeStep acc).reverse) cur units = true ∧ ∀ g ∈ gs.foldl mergeStep acc, Q g.q
  | [], acc, mid, units, ha, hq, hg, _ => by
    simp only [rangesOf, List.map_nil, tiles, beq_iff_eq] at hg
    subst hg
    exact ⟨ha, hq⟩
  | g :: gs, acc, mid, units, ha, hq, hg, hgq => by
    simp only [rangesOf, List.map_cons, tiles, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hg
    obtain ⟨⟨h0, hn⟩, hr⟩ := hg
    rw [List.foldl_cons]
    refine mergeStep_fold Q cur gs (mergeStep acc g) (mid + g.units) units ?_ ?_ hr
      (fun x hx => hgq x (List.mem_cons_of_mem _ hx))
    · cases acc with
      | nil =>
        simp only [List.reverse_nil, rangesOf, List.map_nil, tiles, beq_iff_eq] at ha
        simp only [mergeStep, List.reverse_cons, List.reverse_nil, List.nil_append, rangesOf, List.map_cons,
          List.map_nil, tiles, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
        exact ⟨⟨by omega, hn⟩, by omega⟩
      | cons last before =>
        rw [List.reverse_cons, tiles_snoc] at ha
        obtain ⟨ha1, ha2, ha3⟩ := ha
        rw [mergeStep_cons]
        split
        · rw [List.reverse_cons, tiles_snoc]
          exact ⟨ha1, by show 0 < last.units + g.units; omega,
            by show last.start + (last.units + g.units) = _; omega⟩
        · rw [List.reverse_cons, tiles_snoc]
          refine ⟨?_, hn, by omega⟩
          rw [List.reverse_cons, tiles_snoc]
          exact ⟨ha1, ha2, by omega⟩
    · intro x hx
      cases acc with
      | nil =>
        simp only [mergeStep, List.mem_singleton] at hx
        subst hx
        exact hgq _ List.mem_cons_self
      | cons last before =>
        rw [mergeStep_cons] at hx
        split at hx
        · rcases List.mem_cons.1 hx with rfl | hx
          · exact hq last List.mem_cons_self
          · exact hq x (List.mem_cons_of_mem _ hx)
        · rcases List.mem_cons.1 hx with rfl | hx
          · exact hgq _ List.mem_cons_self
          · exact hq x hx

/-- **Merging neighbours keeps a tiling**, and every row length of the merged grids is one of the grids'. -/
theorem merged_tiles (Q : Nat → Prop) (gs : List Grid) (cur units : Nat) (h : tiles (rangesOf gs) cur units = true)
    (hq : ∀ g ∈ gs, Q g.q) : tiles (rangesOf (merged gs)) cur units = true ∧ ∀ g ∈ merged gs, Q g.q := by
  have hc := (tiles_ordered gs cur units h).1
  obtain ⟨h1, h2⟩ := mergeStep_fold Q cur gs [] cur units (by simp [rangesOf, tiles]) (fun _ hg => by cases hg) h hq
  exact ⟨by rw [merged_eq]; exact h1, fun g hg => h2 g (by rw [merged_eq] at hg; exact List.mem_reverse.1 hg)⟩

/-- A product of numbers each at least one is at least one. -/
theorem foldl_mul_pos : ∀ (l : List Nat) (a : Nat), 0 < a → (∀ x ∈ l, 1 ≤ x) → 0 < l.foldl (· * ·) a
  | [], _, ha, _ => ha
  | x :: l, a, ha, h => by
    rw [List.foldl_cons]
    exact foldl_mul_pos l (a * x) (Nat.mul_pos ha (h x List.mem_cons_self)) fun y hy =>
      h y (List.mem_cons_of_mem _ hy)

/-- **An inference entry's row length is at least one**: a product of radices, each of which `gridOf` refuses
below one. -/
theorem gridOf_q {S : Nat} {e : Json} {tid : String} {g : Grid} (h : gridOf S e = .ok (tid, g)) : 0 < g.q := by
  unfold gridOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i _ _ _ _ _ _ _ _ _ _ radix hradix _ _ k _ _ _
  refine foldl_mul_pos _ 1 (Nat.le_refl 1) fun y hy => ?_
  obtain ⟨x, -, hx⟩ := (Json.mapM_ok _ _ _ hradix y).1 (List.mem_of_mem_drop hy)
  split at hx
  · split at hx
    · cases hx
      assumption
    · cases hx
  · cases hx

/-- A pair found by `lookup` is in the list. -/
theorem mem_of_lookup {β : Type} : ∀ (l : List (String × β)) (a : String) (b : β), l.lookup a = some b → (a, b) ∈ l
  | [], _, _, h => by cases h
  | (a', b') :: l, a, b, h => by
    rw [List.lookup_cons] at h
    split at h
    · rename_i he
      cases h
      rw [beq_iff_eq.1 he]
      exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ (mem_of_lookup l a b h)

/-- **What `gridsOf` reads**: the class has at least one slot; each list of grids is a template's, tiles its units
and has every row length at least one; and every template has its list. -/
theorem gridsOf_ok {ts : List Tmpl} {doc : Json} {S : Nat} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) :
    1 ≤ S ∧
      (∀ tid gs, (tid, gs) ∈ per →
        ∃ t ∈ ts, t.tid = tid ∧ tiles (rangesOf gs) 0 t.units = true ∧ ∀ g ∈ gs, 0 < g.q) ∧
      ∀ t ∈ ts, ∃ gs, (t.tid, gs) ∈ per := by
  unfold gridsOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i _ _ _ _ _ gs _ _ _ hS hgs hper
  refine ⟨by omega, fun tid ms hm => ?_, fun t ht => ?_⟩
  · obtain ⟨t, ht, he⟩ := (Json.mapM_ok _ _ _ hper (tid, ms)).1 hm
    split at he
    · cases he
    · rename_i hc
      cases he
      have hc' : tiles (rangesOf ((List.map (fun x => x.snd) (List.filter (fun x => x.fst == t.tid) gs)).mergeSort
          fun a b => decide (a.start ≤ b.start))) 0 t.units = true := by
        simpa [rangesOf] using hc
      obtain ⟨h1, h2⟩ := merged_tiles (fun q => 0 < q) _ 0 t.units hc' fun g hg => by
        obtain ⟨x, hx, rfl⟩ := List.mem_map.1 (List.mem_mergeSort.1 hg)
        obtain ⟨e, -, he⟩ := (Json.mapM_ok _ _ _ hgs x).1 (List.mem_filter.1 hx).1
        exact gridOf_q (tid := x.1) he
      exact ⟨t, ht, rfl, h1, h2⟩
  · obtain ⟨y, hy⟩ := Json.mapM_ok_all _ _ _ hper t ht
    have hy' := hy
    split at hy'
    · cases hy'
    · cases hy'
      exact ⟨_, (Json.mapM_ok _ _ _ hper _).2 ⟨t, ht, hy⟩⟩

/-- **The grids a template's units are read by**: in unit order, each after the one before it, so no unit in two of
them, and each with a row length of at least one. -/
theorem gridsOf_lookup {ts : List Tmpl} {doc : Json} {S : Nat} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) (tid : String) :
    ((per.lookup tid).getD []).Pairwise (fun a b => a.start + a.units ≤ b.start) ∧
      ∀ g ∈ (per.lookup tid).getD [], 0 < g.q := by
  cases hl : per.lookup tid with
  | none => exact ⟨List.Pairwise.nil, fun _ hg => by cases hg⟩
  | some gs =>
    obtain ⟨t, -, -, h1, h2⟩ := (gridsOf_ok h).2.1 tid gs (mem_of_lookup _ _ _ hl)
    exact ⟨(tiles_ordered gs 0 t.units h1).2.2, h2⟩

/-- Two members of a list with one image under a map whose images are distinct are one member. -/
theorem eq_of_nodup_map {α β : Type} (f : α → β) : ∀ (l : List α), (l.map f).Nodup → ∀ a ∈ l, ∀ b ∈ l,
    f a = f b → a = b
  | [], _, _, ha, _, _, _ => by cases ha
  | x :: l, h, a, ha, b, hb, he => by
    rw [List.map_cons, List.nodup_cons] at h
    rcases List.mem_cons.1 ha with ha' | ha' <;> rcases List.mem_cons.1 hb with hb' | hb'
    · rw [ha', hb']
    · exact absurd (List.mem_map.2 ⟨b, hb', by rw [← he, ha']⟩) h.1
    · exact absurd (List.mem_map.2 ⟨a, ha', by rw [he, hb']⟩) h.1
    · exact eq_of_nodup_map f l h.2 a ha' b hb' he

/-- **A unit of a template is in one of its grids.** -/
theorem gridsOf_covers {ts : List Tmpl} {doc : Json} {S : Nat} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) (hd : (ts.map (·.tid)).Nodup) {t : Tmpl} (ht : t ∈ ts) {u : Nat}
    (hu : u < t.units) : ∃ g ∈ (per.lookup t.tid).getD [], g.start ≤ u ∧ u < g.start + g.units := by
  obtain ⟨gs, hgs⟩ := (gridsOf_ok h).2.2 t ht
  cases hl : per.lookup t.tid with
  | none =>
    exact absurd hl (by
      intro hn
      have := List.lookup_eq_none_iff.1 hn (t.tid, gs) hgs
      simp at this)
  | some gs' =>
    obtain ⟨t', ht', he, h1, -⟩ := (gridsOf_ok h).2.1 t.tid gs' (mem_of_lookup _ _ _ hl)
    have : t' = t := eq_of_nodup_map (·.tid) ts hd t' ht' t ht he
    subst this
    exact tiles_covers gs' 0 t'.units u h1 (Nat.zero_le u) hu

end NciCheck.Record.SecurityProofs
