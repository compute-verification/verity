import SampledProofs.WiresSpan

/-! **`check` is sound against its unrolled reference.** Whenever `check` accepts a circuit's wires:

* `checkRef`, which expands every reader row the wires cover, accepts them too (`check_sound`): every reader unit and
  writer each wire reads lies in its template, and no reader row is wired twice;
* the writer `Summary.writerOf` gives a reader row by the wires alone is the one the expanded rows give it
  (`writerOf_eq`);
* the wired rows and bits, multiplied through, are the expanded rows' counts (`wiredRows_eq`, `wiredBits_eq`).

The converse, that `check` accepts whatever `checkRef` accepts, is `check_complete`'s. -/

namespace SampledProofs.Wires

theorem firstWire_eq_none {c : Circuit} {f : Wire → Option Code} :
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

theorem shape_cut {c : Circuit} {w : Wire} {rt wt : Template} (h : shape c w = .ok (rt, wt)) :
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

theorem readerUnit_lt {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0) {i : Nat} (hi : i < w.units) :
    i / w.cuts < w.readerUnits := by
  unfold Wire.readerUnits
  have h2 : w.units / w.cuts * w.cuts = w.units := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)
  exact (Nat.div_lt_iff_lt_mul (by omega)).2 (by rw [h2]; exact hi)

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

theorem wireCodeRef_of {c : Circuit} {w : Wire} (h : wireCode c w = none) :
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

theorem distinctRows_of {ws : List Wire} (hw : ∀ w ∈ ws, 1 ≤ w.cuts ∧ w.units % w.cuts = 0)
    (h : pairwiseDisjoint ws = true) : distinctRows (ws.flatMap Wire.entries) = true := by
  simp only [distinctRows, decide_eq_true_eq]
  rw [List.map_flatMap, List.Nodup, List.pairwise_flatMap]
  refine ⟨fun w _ => entries_nodup w, ?_⟩
  refine ((pairwiseDisjoint_iff ws).1 h).imp_of_mem fun {a b} ha hb hab => ?_
  exact entries_disjoint (hw a ha).1 (hw a ha).2 (hw b hb).1 (hw b hb).2 hab

/-- Every wire `check` accepts is cut into whole reader units. -/
theorem check_wires {c : Circuit} {S : Summary} (h : check c = .ok S) :
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
theorem check_sound {c : Circuit} {S : Summary} (h : check c = .ok S) :
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
        cases h
        have hall := firstWire_eq_none.1 hf
        have hw : ∀ w ∈ c.wires, 1 ≤ w.cuts ∧ w.units % w.cuts = 0 := fun w hw =>
          ⟨(wireCodeRef_of (hall w hw)).1, (wireCodeRef_of (hall w hw)).2.1⟩
        have hf' : firstWire c (wireCodeRef c) = none :=
          firstWire_eq_none.2 fun w hw => (wireCodeRef_of (hall w hw)).2.2
        have hd := distinctRows_of hw (by simpa using hov)
        refine ⟨rfl, ?_⟩
        unfold checkRef
        simp [hcut, hf', hd]

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
theorem writerOf_eq {c : Circuit} {S : Summary} (h : check c = .ok S) (t : String) (u : Nat) (p : String) (r : Nat) :
    S.writerOf t u p r = ((c.wires.flatMap Wire.entries).find? (·.1 == (t, u, p, r))).map (·.2) := by
  have ⟨hS, _⟩ := check_sound h
  have hw := check_wires h
  rw [List.find?_flatMap, map_findSome?, Summary.writerOf, hS]
  exact findSome?_congr fun w hw' => (entries_find? (hw w hw').1 (hw w hw').2 t u p r).symm

/-! ## The counts multiplied through -/

theorem check_summary {c : Circuit} {S : Summary} (h : check c = .ok S) : S = ⟨c.wires, c.wires.map (rowWords c)⟩ := by
  unfold check at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · cases h; rfl

theorem entries_length (w : Wire) : w.entries.length = w.units := by simp [Wire.entries]

/-- **The wired rows, multiplied through, are the reference's count.** -/
theorem wiredRows_eq {c : Circuit} {S : Summary} (h : check c = .ok S) :
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
theorem wiredBits_eq {c : Circuit} {S : Summary} (h : check c = .ok S) :
    S.wiredBits = (c.wires.map fun w => (w.entries.map fun _ => 16 * rowWords c w).sum).sum := by
  rw [check_summary h, Summary.wiredBits]
  exact wiredBits_list (rowWords c) c.wires

end SampledProofs.Wires
