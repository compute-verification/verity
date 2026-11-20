module

public import Spot.WiresOrder

/-! **`check` is sound against its unrolled reference.** Whenever `check` accepts a circuit's wires:

* `checkRef`, which expands every reader row the wires cover, accepts them too (`check_sound`): every reader unit and
  writer each wire reads lies in its template, and no reader row is wired twice;
* the writer `Summary.writerOf` gives a reader row by the wires alone is the one the expanded rows give it
  (`writerOf_eq`);
* the wired rows and bits, multiplied through, are the expanded rows' counts (`wiredRows_eq`, `wiredBits_eq`);
* the dataflow between units the expanded rows give is acyclic: no unit reads itself through a chain of wires
  (`check_acyclic`).

The converse, that `check` accepts whatever `checkRef` accepts, is `check_complete`'s. -/

@[expose] public section

namespace Spot.Wires

open Circuits

theorem firstWire_eq_none {c : Wiring} {f : Wire → Option Code} :
    firstWire c f = none ↔ ∀ w ∈ c.wires, f w = none := by
  unfold firstWire
  constructor
  · intro h w hw
    cases hf : c.wires.zipIdx.find? (fun (w, _) => (f w).isSome) with
    | none =>
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hw
      have hm : (c.wires[i], i) ∈ c.wires.zipIdx := List.mem_zipIdx_iff_getElem?.2 (by simp)
      simpa using List.find?_eq_none.1 hf _ hm
    | some p =>
      have hp := List.find?_some hf
      rw [hf] at h
      simp only [Option.bind_some, Option.map_eq_none_iff] at h
      simp [h] at hp
  · intro h
    have : c.wires.zipIdx.find? (fun (w, _) => (f w).isSome) = none := by
      refine List.find?_eq_none.2 fun ⟨w, i⟩ hm => ?_
      have hw : w ∈ c.wires := List.mem_of_getElem? (List.mem_zipIdx_iff_getElem?.1 hm)
      simp [h w hw]
    rw [this]; rfl

theorem orderCode_eq_none {c : Wiring} {f : List UnitRange → Wire → Bool} :
    orderCode c f = none ↔ ∃ o, orderOf c = .ok o ∧ ∀ w ∈ c.wires, f o w = true := by
  unfold orderCode
  split
  · rename_i r hr
    simp [hr]
  · rename_i o ho
    rw [firstWire_eq_none]
    constructor
    · intro hall
      exact ⟨o, ho, fun w hw => by have := hall w hw; split at this <;> simp_all⟩
    · rintro ⟨o', ho', hall⟩ w hw
      rw [ho] at ho'
      cases ho'
      simp [hall w hw]

theorem shape_cut {c : Wiring} {w : Wire} {rt wt : Template} (h : shape c w = .ok (rt, wt)) :
    1 ≤ w.cuts ∧ w.units % w.cuts = 0 := by
  unfold shape at h
  split at h
  · split at h
    · dsimp only at h
      split at h
      · cases h
      · rename_i hc
        simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
        exact ⟨hc.1.1.1, hc.2⟩
    · cases h
  · cases h

theorem boundsRef_of {w : Wire} {rt wt : Template} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0)
    (h : boundsOk w rt wt = true) : boundsRef w rt wt = true := by
  simp only [boundsOk, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  simp only [boundsRef, Wire.entries, List.all_map, List.all_eq_true, List.mem_range, Function.comp_apply,
    Wire.readerAt, Bool.and_eq_true, decide_eq_true_eq]
  intro i hi
  have hu := readerUnit_lt hc hd hi
  have hs := w.writerAt_mem_span i hi
  omega

theorem wireCodeRef_of {c : Wiring} {w : Wire} (h : wireCode c w = none) :
    1 ≤ w.cuts ∧ w.units % w.cuts = 0 ∧ wireCodeRef c w = none := by
  unfold wireCode at h
  unfold wireCodeRef
  split at h
  · cases h
  · rename_i rt wt hs
    have ⟨hc, hd⟩ := shape_cut hs
    refine ⟨hc, hd, ?_⟩
    split at h
    · simp [boundsRef_of hc hd ‹_›]
    · cases h

theorem entries_keys (w : Wire) : w.entries.map (·.1) =
    (List.range w.units).map fun i => (w.reader, w.start + i / w.cuts, w.port, w.cut0 + i % w.cuts) := by
  simp [Wire.entries, Wire.readerAt]

/-- A wire covers each of its reader rows once. -/
theorem entries_nodup (w : Wire) : (w.entries.map (·.1)).Nodup := by
  rw [entries_keys, List.Nodup, List.pairwise_map]
  refine (List.nodup_range).imp ?_
  intro a b hab heq
  simp only [Prod.mk.injEq] at heq
  obtain ⟨-, h1, -, h2⟩ := heq
  apply hab
  have e1 : a / w.cuts = b / w.cuts := by omega
  have e2 : a % w.cuts = b % w.cuts := by omega
  rw [← Nat.div_add_mod a w.cuts, ← Nat.div_add_mod b w.cuts, e1, e2]

/-- Two disjoint wires cover no reader row in common. -/
theorem entries_disjoint {a b : Wire} (hca : 1 ≤ a.cuts) (ha : a.units % a.cuts = 0) (hcb : 1 ≤ b.cuts)
    (hb : b.units % b.cuts = 0) (h : disjoint a b = true) :
    ∀ x ∈ a.entries.map (·.1), ∀ y ∈ b.entries.map (·.1), x ≠ y := by
  rw [entries_keys, entries_keys]
  simp only [List.mem_map, List.mem_range]
  rintro _ ⟨i, hi, rfl⟩ _ ⟨j, hj, rfl⟩ heq
  have ui := readerUnit_lt hca ha hi
  have uj := readerUnit_lt hcb hb hj
  have ri := Nat.mod_lt i (by omega : 0 < a.cuts)
  have rj := Nat.mod_lt j (by omega : 0 < b.cuts)
  simp only [Prod.mk.injEq] at heq
  obtain ⟨e1, e2, e3, e4⟩ := heq
  simp only [disjoint, Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq] at h
  rcases h with ((((h | h) | h) | h) | h) | h
  · exact h e1
  · exact h e3
  all_goals
    generalize i / a.cuts = x at *
    generalize j / b.cuts = y at *
    generalize i % a.cuts = x' at *
    generalize j % b.cuts = y' at *
    omega

theorem pairwiseDisjoint_iff : ∀ ws : List Wire, pairwiseDisjoint ws = true ↔ ws.Pairwise (fun a b => disjoint a b = true)
  | [] => by simp [pairwiseDisjoint]
  | w :: ws => by
    simp [pairwiseDisjoint, List.pairwise_cons, List.all_eq_true, pairwiseDisjoint_iff ws]

/-! ### The fast check is `pairwiseDisjoint` -/

namespace DisjointFast

theorem disjoint_comm (a b : Wire) : disjoint a b = disjoint b a := by
  unfold disjoint
  have h1 : (a.reader != b.reader) = (b.reader != a.reader) := bne_comm
  have h2 : (a.port != b.port) = (b.port != a.port) := bne_comm
  rw [h1, h2]
  generalize (b.reader != a.reader) = x1
  generalize (b.port != a.port) = x2
  generalize decide (a.start + a.readerUnits ≤ b.start) = x3
  generalize decide (b.start + b.readerUnits ≤ a.start) = x4
  generalize decide (a.cut0 + a.cuts ≤ b.cut0) = x5
  generalize decide (b.cut0 + b.cuts ≤ a.cut0) = x6
  cases x1 <;> cases x2 <;> cases x3 <;> cases x4 <;> cases x5 <;> cases x6 <;> rfl

theorem disjoint_of_end_le {a b : Wire} (h : a.start + a.readerUnits ≤ b.start) : disjoint a b = true := by
  simp [disjoint, h]

theorem sweep_iff : ∀ (l act : List Wire), l.Pairwise (fun a b => a.start ≤ b.start) →
    (∀ a ∈ act, ∀ c ∈ l, a.start ≤ c.start) →
    (sweep l act = true ↔ l.Pairwise (fun a b => disjoint a b = true) ∧ ∀ a ∈ act, ∀ c ∈ l, disjoint a c = true)
  | [], act, _, _ => by simp [sweep]
  | b :: bs, act, hs, hact => by
    have hsb : ∀ c ∈ bs, b.start ≤ c.start := (List.pairwise_cons.1 hs).1
    have hbs := (List.pairwise_cons.1 hs).2
    have hact' : ∀ a ∈ b :: act.filter (fun a => decide (b.start < a.start + a.readerUnits)), ∀ c ∈ bs, a.start ≤ c.start := by
      intro a ha c hc
      rcases List.mem_cons.1 ha with rfl | ha
      · exact hsb c hc
      · exact hact a (List.mem_filter.1 ha).1 c (List.mem_cons_of_mem _ hc)
    have ih := sweep_iff bs (b :: act.filter (fun a => decide (b.start < a.start + a.readerUnits))) hbs hact'
    have ev : ∀ a ∈ act, ¬ b.start < a.start + a.readerUnits → ∀ c ∈ b :: bs, disjoint a c = true := by
      intro a _ hn c hc
      rcases List.mem_cons.1 hc with rfl | hc
      · exact disjoint_of_end_le (by omega)
      · exact disjoint_of_end_le (by have := hsb c hc; omega)
    simp only [sweep, Bool.and_eq_true]
    rw [ih, List.all_eq_true]
    constructor
    · rintro ⟨h1, h2, h3⟩
      refine ⟨List.pairwise_cons.2 ⟨fun c hc => h3 b (List.mem_cons_self ..) c hc, h2⟩, fun a ha c hc => ?_⟩
      by_cases hn : b.start < a.start + a.readerUnits
      · rcases List.mem_cons.1 hc with rfl | hc
        · exact h1 a (List.mem_filter.2 ⟨ha, by simpa using hn⟩)
        · exact h3 a (List.mem_cons_of_mem _ (List.mem_filter.2 ⟨ha, by simpa using hn⟩)) c hc
      · exact ev a ha hn c hc
    · rintro ⟨hp, h3⟩
      refine ⟨fun a ha => h3 a (List.mem_filter.1 ha).1 b (List.mem_cons_self ..), (List.pairwise_cons.1 hp).2, ?_⟩
      intro a ha c hc
      rcases List.mem_cons.1 ha with rfl | ha
      · exact (List.pairwise_cons.1 hp).1 c hc
      · exact h3 a (List.mem_filter.1 ha).1 c (List.mem_cons_of_mem _ hc)

theorem groupDisjoint_iff (g : List Wire) :
    groupDisjoint g = true ↔ g.Pairwise (fun a b => disjoint a b = true) := by
  unfold groupDisjoint
  have hperm := List.mergeSort_perm g (fun a b => decide (a.start ≤ b.start))
  have hsorted : (g.mergeSort fun a b => decide (a.start ≤ b.start)).Pairwise (fun a b => a.start ≤ b.start) := by
    have := List.pairwise_mergeSort (le := fun a b => decide (a.start ≤ b.start)) (l := g)
      (by intro a b c hab hbc; simp at *; omega) (by intro a b; simp; omega)
    simpa using this
  rw [sweep_iff _ [] hsorted (by simp)]
  simp only [List.not_mem_nil, false_imp_iff, implies_true, and_true]
  exact List.Perm.pairwise_iff (R := fun a b => disjoint a b = true) (fun {a b} hab => by rw [disjoint_comm]; exact hab) hperm

theorem pairwiseDisjointFast_iff (ws : List Wire) : pairwiseDisjointFast ws = true ↔ ws.Pairwise (fun a b => disjoint a b = true) := by
  unfold pairwiseDisjointFast
  simp only [List.all_eq_true, groupDisjoint_iff, List.mem_eraseDups, List.mem_map]
  constructor
  · intro h
    rw [List.pairwise_iff_forall_sublist]
    intro a b hab
    by_cases hk : (a.reader, a.port) = (b.reader, b.port)
    · have ha : a ∈ ws := hab.subset (by simp)
      have hg := h (a.reader, a.port) ⟨a, ha, rfl⟩
      rw [List.pairwise_iff_forall_sublist] at hg
      apply hg
      have := hab.filter (fun w => (w.reader, w.port) == (a.reader, a.port))
      simpa [List.filter_cons, hk] using this
    · unfold disjoint
      by_cases hr : a.reader = b.reader
      · have hp : a.port ≠ b.port := fun hp => hk (by rw [hr, hp])
        simp [hp]
      · simp [hr]
  · intro h k ⟨a, _, hka⟩
    exact h.sublist (List.filter_sublist)


end DisjointFast

/-- **`pairwiseDisjointFast` is `pairwiseDisjoint`**: the sorted sweep accepts exactly the wire lists whose every pair is
disjoint. -/
theorem pairwiseDisjointFast_eq (ws : List Wire) : pairwiseDisjointFast ws = pairwiseDisjoint ws := by
  rw [Bool.eq_iff_iff, DisjointFast.pairwiseDisjointFast_iff, pairwiseDisjoint_iff]

theorem distinctRows_of {ws : List Wire} (hw : ∀ w ∈ ws, 1 ≤ w.cuts ∧ w.units % w.cuts = 0)
    (h : pairwiseDisjoint ws = true) : distinctRows (ws.flatMap Wire.entries) = true := by
  simp only [distinctRows, decide_eq_true_eq]
  rw [List.map_flatMap, List.Nodup, List.pairwise_flatMap]
  refine ⟨fun w _ => entries_nodup w, ?_⟩
  refine ((pairwiseDisjoint_iff ws).1 h).imp_of_mem fun {a b} ha hb hab => ?_
  exact entries_disjoint (hw a ha).1 (hw a ha).2 (hw b hb).1 (hw b hb).2 hab

/-- Every wire `check` accepts is cut into whole reader units. -/
theorem check_wires {c : Wiring} {S : Summary} (h : check c = .ok S) :
    ∀ w ∈ c.wires, 1 ≤ w.cuts ∧ w.units % w.cuts = 0 := by
  unfold check at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hf
      intro w hw
      have := wireCodeRef_of (firstWire_eq_none.1 hf w hw)
      exact ⟨this.1, this.2.1⟩

/-- **`check` is sound against the unrolled reference**: what it accepts, `checkRef` accepts, its entries those of the
summary's wires, which are the circuit's. -/
theorem check_sound {c : Wiring} {S : Summary} (h : check c = .ok S) :
    S.wires = c.wires ∧ checkRef c = .ok (c.wires.flatMap Wire.entries) := by
  unfold check at h
  split at h
  · cases h
  · rename_i hcut
    split at h
    · cases h
    · rename_i hf
      split at h
      · cases h
      · rename_i hov
        split at h
        · cases h
        · rename_i hord
          cases h
          have hall := firstWire_eq_none.1 hf
          have hw : ∀ w ∈ c.wires, 1 ≤ w.cuts ∧ w.units % w.cuts = 0 := fun w hw =>
            ⟨(wireCodeRef_of (hall w hw)).1, (wireCodeRef_of (hall w hw)).2.1⟩
          have hf' : firstWire c (wireCodeRef c) = none :=
            firstWire_eq_none.2 fun w hw => (wireCodeRef_of (hall w hw)).2.2
          have hd := distinctRows_of hw (by rw [← pairwiseDisjointFast_eq]; simpa using hov)
          have hord' : orderCode c orderedWireRef = none := by
            obtain ⟨o, ho, hin⟩ := orderCode_eq_none.1 (orderCodeIx_eq c ▸ hord)
            exact orderCode_eq_none.2 ⟨o, ho, fun w hw' => orderedWire_eq_ref o w ▸ hin w hw'⟩
          refine ⟨rfl, ?_⟩
          unfold checkRef
          simp [hcut, hf', hd, hord']

/-- The order `checkRef` accepts the wires in. -/
theorem checkRef_order {c : Wiring} {es : List Entry} (h : checkRef c = .ok es) :
    ∃ o, orderOf c = .ok o ∧ ∀ w ∈ c.wires, orderedWireRef o w = true := by
  unfold checkRef at h
  split at h
  · cases h
  · split at h
    · cases h
    · dsimp only at h
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i hord
          exact orderCode_eq_none.1 hord

theorem check_wireCode {c : Wiring} {S : Summary} (h : check c = .ok S) : ∀ w ∈ c.wires, wireCode c w = none := by
  unfold check at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hf
      exact firstWire_eq_none.1 hf

theorem shape_lookup {c : Wiring} {w : Wire} {rt wt : Template} (h : shape c w = .ok (rt, wt)) :
    c.templates.lookup w.reader = some rt ∧ c.templates.lookup w.writer = some wt := by
  unfold shape at h
  split at h
  · rename_i rt' wt' hr hw
    split at h
    · dsimp only at h
      split at h
      · cases h
      · split at h
        · cases h
        · cases h
          exact ⟨hr, hw⟩
    · cases h
  · cases h

theorem wireCode_shape {c : Wiring} {w : Wire} (h : wireCode c w = none) :
    ∃ rt wt, c.templates.lookup w.reader = some rt ∧ c.templates.lookup w.writer = some wt ∧ 1 ≤ w.cuts ∧
      w.units % w.cuts = 0 ∧ boundsOk w rt wt = true := by
  unfold wireCode at h
  split at h
  · cases h
  · rename_i rt wt hs
    split at h
    · rename_i hb
      exact ⟨rt, wt, (shape_lookup hs).1, (shape_lookup hs).2, (shape_cut hs).1, (shape_cut hs).2, hb⟩
    · cases h

/-- Writer unit `x` writes a row that reader unit `y` reads, through one of the wires: an edge of the dataflow between
units the expanded rows give, each unit `(template, unit)`. -/
def Wiring.Reads (c : Wiring) (x y : String × Int) : Prop :=
  ∃ w ∈ c.wires, ∃ e ∈ w.entries, x = (e.2.1, e.2.2.1) ∧ y = (e.1.1, (e.1.2.1 : Int))

/-- **The wires `check` accepts rank the units**: the rank of a unit's range in the order, under which along every wire
the reader unit's exceeds the writer's: the writer lies in the wire's box over the reader indices in the reader unit's
range (`Wire.writerAt_mem_box`), and the order's closed form ranks every range meeting that box below it. -/
theorem check_ranked {c : Wiring} {S : Summary} (h : check c = .ok S) :
    ∃ ρ : String × Int → Nat, ∀ x y, c.Reads x y → ρ x < ρ y := by
  obtain ⟨o, ho, hin⟩ := checkRef_order (check_sound h).2
  have hok := orderOf_ok ho
  refine ⟨fun x => (rangeOf o x.1 x.2.toNat).getD 0, ?_⟩
  · rintro x y ⟨w, hw, e, he, rfl, rfl⟩
    obtain ⟨rt, wt, hr, hwt, hc, hd, hb⟩ := wireCode_shape (check_wireCode h w hw)
    simp only [boundsOk, Bool.and_eq_true, decide_eq_true_eq] at hb
    simp only [Wire.entries, List.mem_map, List.mem_range] at he
    obtain ⟨i, hi, rfl⟩ := he
    simp only [Wire.readerAt]
    have hu := readerUnit_lt hc hd hi
    have hs := w.writerAt_mem_span i hi
    have hcl : orderedWire o w = true := (orderedWire_eq_ref o w).symm ▸ hin w hw
    simp only [orderedWire, List.all_eq_true] at hcl
    obtain ⟨a, ha, ham⟩ := rangeOf_some hok hr (u := w.start + i / w.cuts) (by omega)
    obtain ⟨p0, n, hk, hmeet⟩ := mem_ranksIn.1 ham
    have hblk := hcl ((w.reader, p0, n), a) (List.mem_zipIdx_iff_getElem?.2 hk)
    have hreads : w.readsIn (w.reader, p0, n) = true := by
      simp only [Wire.readsIn, beq_self_eq_true, Bool.true_and, decide_eq_true_eq]
      generalize i / w.cuts = q at hmeet hu
      omega
    simp only [hreads, Bool.not_true, Bool.false_or, List.all_eq_true, decide_eq_true_eq] at hblk
    have hlo : (w.indicesIn p0 n).1 ≤ i := by
      have h1 : max p0 w.start - w.start ≤ i / w.cuts := by
        generalize i / w.cuts = q at hmeet ⊢
        omega
      exact Nat.le_trans (Nat.mul_le_mul_right w.cuts h1) (Nat.div_mul_le_self i w.cuts)
    have hhi : i ≤ (w.indicesIn p0 n).2 := by
      have h1 : i / w.cuts + 1 ≤ min (p0 + n) (w.start + w.readerUnits) - w.start := by
        generalize i / w.cuts = q at hmeet hu ⊢
        omega
      have h2 := Nat.mul_le_mul_right w.cuts h1
      have h3 := Nat.lt_mul_div_succ i (show 0 < w.cuts by omega)
      rw [Nat.mul_comm] at h3
      show i ≤ (min (p0 + n) (w.start + w.readerUnits) - w.start) * w.cuts - 1
      omega
    have hbx := w.writerAt_mem_box hlo hhi
    obtain ⟨b, hb', hbm⟩ := rangeOf_some hok hwt (u := (w.writerAt i).toNat) (by omega)
    have := hblk b (mem_ranksIn_iff.2 ⟨_, Int.toNat_le_toNat hbx.1, by have := Int.toNat_le_toNat hbx.2; omega, hbm⟩)
    simp only [Int.toNat_natCast, ha, hb', Option.getD_some]
    exact this

/-- **The wires `check` accepts are acyclic**: no unit reads itself through a chain of wires, since the wires rank the
units (`check_ranked`). -/
theorem check_acyclic {c : Wiring} {S : Summary} (h : check c = .ok S) (x : String × Int) :
    ¬ Relation.TransGen c.Reads x x := by
  obtain ⟨ρ, step⟩ := check_ranked h
  have chain : ∀ a b, Relation.TransGen c.Reads a b → ρ a < ρ b := by
    intro a b hab
    induction hab with
    | single h => exact step _ _ h
    | tail _ h ih => exact Nat.lt_trans ih (step _ _ h)
  intro hx
  exact Nat.lt_irrefl _ (chain x x hx)

/-! ## The lookup: `Summary.writerOf` is the reference's search -/

theorem find?_of_unique {α : Type} {p : α → Bool} {o : Option α} :
    ∀ {l : List α}, (∀ i ∈ l, p i = true ↔ o = some i) → (∀ k, o = some k → k ∈ l) → l.find? p = o
  | [], h, hk => by
    cases o with
    | none => rfl
    | some k => exact absurd (hk k rfl) List.not_mem_nil
  | x :: l, h, hk => by
    rw [List.find?_cons]
    cases hx : p x with
    | true => exact ((h x (List.mem_cons_self ..)).1 hx).symm
    | false =>
      refine find?_of_unique (fun i hi => h i (List.mem_cons_of_mem _ hi)) fun k hok => ?_
      rcases List.mem_cons.1 (hk k hok) with rfl | hm
      · have := (h k (List.mem_cons_self ..)).2 hok
        simp [hx] at this
      · exact hm

theorem findSome?_congr {α β : Type} {f g : α → Option β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.findSome? f = l.findSome? g
  | [], _ => rfl
  | x :: l, h => by
    rw [List.findSome?_cons, List.findSome?_cons, h x (List.mem_cons_self ..)]
    cases g x with
    | some b => rfl
    | none => exact findSome?_congr fun y hy => h y (List.mem_cons_of_mem _ hy)

theorem map_findSome? {α β γ : Type} (f : β → γ) (g : α → Option β) :
    ∀ l : List α, (l.findSome? g).map f = l.findSome? fun x => (g x).map f
  | [] => rfl
  | x :: l => by
    rw [List.findSome?_cons, List.findSome?_cons]
    cases g x with
    | some b => rfl
    | none => exact map_findSome? f g l

/-- A wire's `indexOf` picks out the one index whose expanded reader row is the query. -/
theorem indexOf_iff {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0) (t : String) (u : Nat) (p : String)
    (r : Nat) (i : Nat) (hi : i < w.units) :
    ((w.reader, w.start + i / w.cuts, w.port, w.cut0 + i % w.cuts) == (t, u, p, r)) = true ↔
      w.indexOf t u p r = some i := by
  have hu := readerUnit_lt hc hd hi
  have hm := Nat.mod_lt i (by omega : 0 < w.cuts)
  have hdm := Nat.div_add_mod' i w.cuts
  simp only [beq_iff_eq, Prod.mk.injEq, Wire.indexOf]
  constructor
  · rintro ⟨rfl, rfl, rfl, rfl⟩
    generalize i / w.cuts = q at *
    generalize i % w.cuts = m at *
    split
    · congr 1
      have e1 : w.start + q - w.start = q := by omega
      have e2 : w.cut0 + m - w.cut0 = m := by omega
      rw [e1, e2]
      omega
    · rename_i hn
      exfalso
      apply hn
      simp only [beq_self_eq_true, Bool.true_and, Bool.and_eq_true, decide_eq_true_eq]
      omega
  · intro h
    split at h
    · rename_i hcnd
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hcnd
      obtain ⟨⟨⟨⟨⟨rfl, rfl⟩, h1⟩, h2⟩, h3⟩, h4⟩ := hcnd
      cases h
      have e1 : ((r - w.cut0) + (u - w.start) * w.cuts) / w.cuts = u - w.start := by
        rw [Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt (by omega), Nat.zero_add]
      have e2 : ((r - w.cut0) + (u - w.start) * w.cuts) % w.cuts = r - w.cut0 := by
        rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
      rw [Nat.add_comm] at e1 e2
      refine ⟨rfl, ?_, rfl, ?_⟩ <;> omega
    · cases h

theorem indexOf_lt {w : Wire} (hd : w.units % w.cuts = 0) {t : String} {u : Nat} {p : String}
    {r k : Nat} (h : w.indexOf t u p r = some k) : k < w.units := by
  unfold Wire.indexOf at h
  split at h
  · rename_i hcnd
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hcnd
    obtain ⟨⟨⟨⟨⟨-, -⟩, h1⟩, h2⟩, h3⟩, h4⟩ := hcnd
    cases h
    have h5 : w.units / w.cuts * w.cuts = w.units := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)
    have h6 : (u - w.start + 1) * w.cuts ≤ w.readerUnits * w.cuts := Nat.mul_le_mul_right _ (by omega)
    unfold Wire.readerUnits at h6
    rw [h5, Nat.succ_mul] at h6
    omega
  · cases h

/-- One wire's search among its expanded rows is its `indexOf`. -/
theorem entries_find? {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0) (t : String) (u : Nat) (p : String)
    (r : Nat) : (w.entries.find? (·.1 == (t, u, p, r))).map (·.2) =
      (w.indexOf t u p r).map fun i => (w.writer, w.writerAt i, w.out) := by
  have key : (List.range w.units).find? (fun i => ((w.reader, (w.readerAt i).1, w.port, (w.readerAt i).2) ==
      (t, u, p, r))) = w.indexOf t u p r :=
    find?_of_unique (fun i hi => indexOf_iff hc hd t u p r i (List.mem_range.1 hi))
      fun k hk => List.mem_range.2 (indexOf_lt hd hk)
  unfold Wire.entries
  rw [List.find?_map, Option.map_map]
  exact (congrArg (Option.map _) key).trans rfl

/-- **The summary's lookup is the reference's**: the writer `writerOf` gives a reader row, by the wires alone, is the one
the expanded entries give it. -/
theorem writerOf_eq {c : Wiring} {S : Summary} (h : check c = .ok S) (t : String) (u : Nat) (p : String) (r : Nat) :
    S.writerOf t u p r = ((c.wires.flatMap Wire.entries).find? (·.1 == (t, u, p, r))).map (·.2) := by
  have ⟨hS, _⟩ := check_sound h
  have hw := check_wires h
  rw [List.find?_flatMap, map_findSome?, Summary.writerOf, hS]
  exact findSome?_congr fun w hw' => (entries_find? (hw w hw').1 (hw w hw').2 t u p r).symm

/-- **A row's writer is a unit its reader's unit reads**: the writer `Summary.writerOf` gives a reader row is an edge of
the dataflow between units, so it ranks below the reader (`check_ranked`). -/
theorem writerOf_reads {c : Wiring} {S : Summary} (h : check c = .ok S) {t : String} {u : Nat} {p : String} {r : Nat}
    {wt : String} {wu : Int} {out : String} (hw : S.writerOf t u p r = some (wt, wu, out)) :
    c.Reads (wt, wu) (t, (u : Int)) := by
  rw [writerOf_eq h] at hw
  obtain ⟨e, he, h2⟩ := Option.map_eq_some_iff.1 hw
  obtain ⟨w, hw', hew⟩ := List.mem_flatMap.1 (List.mem_of_find?_eq_some he)
  have h1 : e.1 = (t, u, p, r) := by simpa using List.find?_some he
  exact ⟨w, hw', e, hew, by rw [h2], by rw [h1]⟩

/-! ## The counts multiplied through -/

theorem check_summary {c : Wiring} {S : Summary} (h : check c = .ok S) : S = ⟨c.wires, c.wires.map (rowWords c)⟩ := by
  unfold check at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · cases h; rfl

theorem entries_length (w : Wire) : w.entries.length = w.units := by simp [Wire.entries]

/-- **The wired rows, multiplied through, are the reference's count.** -/
theorem wiredRows_eq {c : Wiring} {S : Summary} (h : check c = .ok S) :
    S.wiredRows = (c.wires.flatMap Wire.entries).length := by
  rw [check_summary h, Summary.wiredRows, List.length_flatMap]
  simp [entries_length]

theorem wiredBits_list (g : Wire → Nat) : ∀ l : List Wire,
    ((l.zip (l.map g)).map fun (w, n) => w.units * (16 * n)).sum =
      (l.map fun w => (w.entries.map fun _ => 16 * g w).sum).sum
  | [] => rfl
  | w :: l => by
    simp only [List.map_cons, List.zip_cons_cons, List.sum_cons, wiredBits_list g l, List.map_const',
      List.sum_replicate_nat, entries_length]

/-- **The wired bits, multiplied through, are the reference's count**: each covered reader row's words, 16 bits a word. -/
theorem wiredBits_eq {c : Wiring} {S : Summary} (h : check c = .ok S) :
    S.wiredBits = (c.wires.map fun w => (w.entries.map fun _ => 16 * rowWords c w).sum).sum := by
  rw [check_summary h, Summary.wiredBits]
  exact wiredBits_list (rowWords c) c.wires

end Spot.Wires
