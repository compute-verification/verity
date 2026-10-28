import Proofs.Flock.Soundness.ExecPlacement

/-!
# What `HmRow.check` accepted

The layout facts of an `hm96-sha512/row/v1` circuit that `HmRow.check` checks, as `ExecSetup.lean` uses them: the
ranges inside the block, pairwise apart, in slots of their nets' rows; the unit a text net with one range of
`vus_per_block × units_per_vu` slots and a group of 16-bit leaf ports; the row slots' ports; every net range's slots per
template instance; one range per net; the wires' and leaf cuts' bounds; the leaf maps.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-- **The layout facts `HmRow.check` checks.** -/
structure CheckFacts (c : Circuit) : Prop where
  bound_slotLog : ∀ r ∈ c.ranges.toList, r.start + spanH r ≤ 2 ^ c.kLog ∧
    ∀ i, r.kind = .net i → (c.nets.getD i default).2.unitLog = r.slotLog
  disj : c.ranges.toList.Pairwise Apart
  unit_lookup : c.unit.lookup.isSome = false
  sha_in : (portCols (c.nets.getD c.shaNet default).2.inGroups).size = 6 ∧
    ∀ i < 6, ((portCols (c.nets.getD c.shaNet default).2.inGroups)[i]!).2 = if i % 2 == 0 then 512 else 1024
  hm_in : (portCols (c.nets.getD c.hmNet default).2.inGroups).map (·.2) = #[512, 1024, 8 * HmRow.SALT_BYTES]
  perVu_sha : c.perVu.getD c.shaNet 0 = (HmRow.rowSlots c.ports).1
  perVu_hm : c.perVu.getD c.hmNet 0 = nextPow2 c.ports.size
  unit_range : (c.ranges.filter (·.kind == Kind.net c.unitNet)).size = 1 ∧
    (c.ranges.filter (·.kind == Kind.net c.unitNet))[0]!.count = c.g * c.upv
  unit_groups : c.unit.inGroups.size ≠ 0 ∧ ∀ b ∈ (c.unit.inGroups.getD 0 default).bits.toList, b = 16
  count : ∀ r ∈ c.ranges.toList, ∀ n, r.kind = .net n → r.count = c.g * c.perVu.getD n 0
  one_range : ∀ n < c.nets.size, (c.ranges.filter (·.kind == Kind.net n)).size = 1
  wires : ∀ w ∈ c.wires.toList, w.dst.1 < c.nets.size ∧ w.dst.2.1 < c.perVu.getD w.dst.1 0 ∧
    w.dst.2.2 < (portCols (c.nets.getD w.dst.1 default).2.inGroups).size
  /-- each wire's source is an output port of a slot, as wide as the input port it fills -/
  wire_src : ∀ w ∈ c.wires.toList, w.src.1 < c.nets.size ∧ w.src.2.1 < c.perVu.getD w.src.1 0 ∧
    w.src.2.2 < (portCols (c.nets.getD w.src.1 default).2.outGroups).size ∧
    (portCols (c.nets.getD w.dst.1 default).2.inGroups)[w.dst.2.2]!.2 =
      (portCols (c.nets.getD w.src.1 default).2.outGroups)[w.src.2.2]!.2
  cuts : ∀ e ∈ c.leafCuts.toList, e.1 < c.upv ∧ e.2.1 < (portCols c.unit.inGroups).size
  leaves : c.leavesIn.size = c.upv ∧
    ∀ u < c.upv, (c.leavesIn[u]!).size = (c.unit.inGroups.getD 0 default).bits.size
  /-- where the unit returns the outputs, each unit's output leaves are its output group's ports -/
  leavesOut : c.leavesOut.size = c.upv ∧ ∀ u < c.upv, c.outNet = c.unitNet →
    (c.leavesOut[u]!).size = (c.unit.outGroups.getD 0 default).bits.size
  /-- no wire reaches a unit's leaf port -/
  wire_leaf : ∀ w ∈ c.wires.toList, w.dst.1 = c.unitNet → (c.unit.inGroups.getD 0 default).bits.size ≤ w.dst.2.2
  /-- a leaf cut reaches a cut port, past the leaf ports -/
  cut_leaf : ∀ e ∈ c.leafCuts.toList, (c.unit.inGroups.getD 0 default).bits.size ≤ e.2.1
  /-- every other input port of every slot of a circuit but the rows' is wired exactly once, by wires and cuts together -/
  wired : ∀ n < c.nets.size, n ≠ c.shaNet → n ≠ c.hmNet → ∀ i < c.perVu.getD n 0, ∀ p,
    (if n = c.unitNet then (c.unit.inGroups.getD 0 default).bits.size else 0) ≤ p →
    p < (portCols (c.nets.getD n default).2.inGroups).size →
    (c.wires.toList.filter fun w => w.dst = (n, i, p)).length +
      (c.leafCuts.toList.filter fun e => (c.unitNet, e.1, e.2.1) = (n, i, p)).length = 1
  /-- the unit leaves a forced-zero row in its slot -/
  unit_zero : c.unit.useful < 2 ^ c.unit.unitLog
  /-- a unit that returns the outputs has a power of two of slots -/
  unit_pow2 : c.outNet = c.unitNet → isPow2 (c.ranges.filter (·.kind == Kind.net c.unitNet))[0]!.count = true
  /-- a VU holds a unit -/
  upv_pos : 0 < c.upv
  /-- a block holds a power of two of VUs -/
  g_pow2 : isPow2 c.g = true

/-! ## Steps of an accepted check -/

/-- `ite_throw_ok`, keeping the condition that did not throw. -/
private theorem ite_throw_ok' {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : ¬c ∧ g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; cases h
  · rw [ite_eq_right hc] at h; exact ⟨hc, h⟩

/-- An `if` whose taken branch, when it succeeds, ends in the else branch's continuation. -/
theorem ite_join {β : Type} {c : Prop} [Decidable c] {a k : Except String β} {v : β}
    (h : (if c then a else k) = .ok v) (ha : a = .ok v → k = .ok v) : k = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; exact ha h
  · rwa [ite_eq_right hc] at h

/-- A loop in `Except` that finished: every element's step, from some state, yielded and has `P`. -/
theorem forIn_list_forall {α σ ε : Type} {l : List α} {b : σ} {f : α → σ → Except ε (ForInStep σ)} {v : σ}
    {P : α → Prop} (hf : ∀ a ∈ l, ∀ s t, f a s = .ok t → (∃ s', t = .yield s') ∧ P a)
    (h : forIn l b f = .ok v) : ∀ a ∈ l, P a := by
  induction l generalizing b with
  | nil => intro a ha; cases ha
  | cons x l ih =>
    rw [List.forIn_cons] at h
    obtain ⟨t, ht, h⟩ := except_bind_ok h
    obtain ⟨⟨s', rfl⟩, hx⟩ := hf x List.mem_cons_self b t ht
    intro a ha
    rcases List.mem_cons.1 ha with rfl | ha
    · exact hx
    · exact ih (fun a ha' => hf a (List.mem_cons_of_mem _ ha')) h a ha

theorem forIn_array_forall {α σ ε : Type} {xs : Array α} {b : σ} {f : α → σ → Except ε (ForInStep σ)} {v : σ}
    {P : α → Prop} (hf : ∀ a ∈ xs.toList, ∀ s t, f a s = .ok t → (∃ s', t = .yield s') ∧ P a)
    (h : forIn xs b f = .ok v) : ∀ a ∈ xs.toList, P a :=
  forIn_list_forall hf (by rw [Array.forIn_toList]; exact h)

theorem forIn_range_forall {σ ε : Type} {lo hi : ℕ} {b : σ} {f : ℕ → σ → Except ε (ForInStep σ)} {v : σ}
    {P : ℕ → Prop} (hf : ∀ a, lo ≤ a → a < hi → ∀ s t, f a s = .ok t → (∃ s', t = .yield s') ∧ P a)
    (h : forIn [lo:hi] b f = .ok v) : ∀ a, lo ≤ a → a < hi → P a := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hmem : ∀ a, a ∈ List.range' lo [lo:hi].size [lo:hi].step ↔ lo ≤ a ∧ a < hi := by
    intro a
    simp only [Std.Legacy.Range.size, List.mem_range'_1]
    omega
  intro a hlo hhi
  exact forIn_list_forall (fun a ha => hf a ((hmem a).1 ha).1 ((hmem a).1 ha).2) h a ((hmem a).2 ⟨hlo, hhi⟩)

/-- A loop in `Except` that finished, each step pushing `g a`: every element has `P`, and the state is `b` then
the `g a`. -/
theorem forIn_list_push {α β ε : Type} {l : List α} {b : Array β} {f : α → Array β → Except ε (ForInStep (Array β))}
    {v : Array β} {P : α → Prop} {g : α → β}
    (hf : ∀ a ∈ l, ∀ s t, f a s = .ok t → t = .yield (s.push (g a)) ∧ P a)
    (h : forIn l b f = .ok v) : (∀ a ∈ l, P a) ∧ v.toList = b.toList ++ l.map g := by
  induction l generalizing b with
  | nil =>
    rw [List.forIn_nil] at h
    cases h
    exact ⟨fun a ha => (by cases ha), by simp⟩
  | cons x l ih =>
    rw [List.forIn_cons] at h
    obtain ⟨t, ht, h⟩ := except_bind_ok h
    obtain ⟨rfl, hx⟩ := hf x List.mem_cons_self b t ht
    obtain ⟨hl, hv⟩ := ih (fun a ha' => hf a (List.mem_cons_of_mem _ ha')) h
    refine ⟨fun a ha => ?_, by rw [hv]; simp⟩
    rcases List.mem_cons.1 ha with rfl | ha
    · exact hx
    · exact hl a ha

/-! ## `HmRow.check`'s count of each port's wirings -/

/-- `HmRow.check`'s count step: one more wiring of `key`. -/
def bump (wired : Array ((ℕ × ℕ × ℕ) × ℕ)) (key : ℕ × ℕ × ℕ) : Array ((ℕ × ℕ × ℕ) × ℕ) :=
  match wired.findIdx? (·.1 == key) with
  | some i => wired.modify i fun (k, v) => (k, v + 1)
  | none => wired.push (key, 1)

/-- The count `HmRow.check` reads for `key`. -/
def wcount (wired : Array ((ℕ × ℕ × ℕ) × ℕ)) (key : ℕ × ℕ × ℕ) : ℕ :=
  ((wired.find? (·.1 == key)).map (·.2)).getD 0

theorem wcount_bump (wired : Array ((ℕ × ℕ × ℕ) × ℕ)) (key key' : ℕ × ℕ × ℕ) :
    wcount (bump wired key) key' = wcount wired key' + if key = key' then 1 else 0 := by
  unfold wcount bump
  split
  · rename_i i hi
    obtain ⟨hlt, hk, hmin⟩ := Array.findIdx?_eq_some_iff_getElem.1 hi
    have hl : i < wired.toList.length := by simpa using hlt
    have hsplit : wired.toList = wired.toList.take i ++ wired[i] :: wired.toList.drop (i + 1) := by
      rw [← Array.getElem_toList (h := hlt), List.getElem_cons_drop, List.take_append_drop]
    have hmod : (wired.modify i fun (k, v) => (k, v + 1)).toList =
        wired.toList.take i ++ (wired[i].1, wired[i].2 + 1) :: wired.toList.drop (i + 1) := by
      rw [Array.toList_modify, List.modify_eq_take_cons_drop hl]
      rfl
    have hpre : ∀ y ∈ wired.toList.take i, (y.1 == key) = false := by
      intro y hy
      obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hy
      have hj' : j < i := by simp at hj; omega
      rw [List.getElem_take]
      simpa using hmin j hj'
    have hki : wired[i].1 = key := by simpa using hk
    rw [← Array.find?_toList, ← Array.find?_toList, hmod]
    generalize wired.toList.take i = pre at hsplit hpre ⊢
    generalize wired.toList.drop (i + 1) = post at hsplit ⊢
    rw [hsplit]
    simp only [List.find?_append, List.find?_cons]
    by_cases he : key = key'
    · subst he
      rw [List.find?_eq_none.2 (fun y hy => by simp [hpre y hy])]
      simp [hki]
    · have hne : (wired[i].1 == key') = false := by simp [hki, he]
      simp [hne, he]
  · rename_i hn
    rw [Array.findIdx?_eq_none_iff] at hn
    rw [← Array.find?_toList, ← Array.find?_toList, Array.toList_push, List.find?_append]
    by_cases he : key = key'
    · subst he
      rw [List.find?_eq_none.2 (fun y hy => by simp [hn y (Array.mem_toList_iff.1 hy)])]
      simp
    · simp [he]

theorem wcount_empty (key : ℕ × ℕ × ℕ) : wcount #[] key = 0 := rfl

theorem wcount_foldl {α : Type} (key : ℕ × ℕ × ℕ) (kf : α → ℕ × ℕ × ℕ) :
    ∀ (l : List α) (w : Array ((ℕ × ℕ × ℕ) × ℕ)),
      wcount (l.foldl (fun s a => bump s (kf a)) w) key = wcount w key + (l.filter fun a => kf a = key).length
  | [], w => by simp
  | a :: l, w => by
    rw [List.foldl_cons, wcount_foldl key kf l, wcount_bump, List.filter_cons]
    by_cases h : kf a = key
    · simp [h]; omega
    · simp [h]

/-- A loop in `Except` whose every step yields `g a s`: it is a fold. -/
theorem forIn_list_fold {α σ ε : Type} (g : α → σ → σ) {f : α → σ → Except ε (ForInStep σ)} :
    ∀ (l : List α) (b v : σ), (∀ a ∈ l, ∀ s t, f a s = .ok t → t = .yield (g a s)) → forIn l b f = .ok v →
      v = l.foldl (fun s a => g a s) b
  | [], b, v, _, h => by rw [List.forIn_nil] at h; cases h; rfl
  | a :: l, b, v, hf, h => by
    rw [List.forIn_cons] at h
    obtain ⟨t, ht, h⟩ := except_bind_ok h
    have := hf a List.mem_cons_self b t ht
    subst this
    exact forIn_list_fold g l _ v (fun a' ha' => hf a' (List.mem_cons_of_mem _ ha')) h

theorem forIn_array_fold {α σ ε : Type} (g : α → σ → σ) {f : α → σ → Except ε (ForInStep σ)} {xs : Array α} {b v : σ}
    (hf : ∀ a ∈ xs.toList, ∀ s t, f a s = .ok t → t = .yield (g a s)) (h : forIn xs b f = .ok v) :
    v = xs.toList.foldl (fun s a => g a s) b :=
  forIn_list_fold g xs.toList b v hf (by rw [Array.forIn_toList]; exact h)

/-- **What `HmRow.check` accepted.** -/
theorem check_facts {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (h : HmRow.check c tags ls = .ok ()) : CheckFacts c := by
  unfold HmRow.check at h
  obtain ⟨spans, hspans, h⟩ := except_bind_ok h
  rw [← Array.forIn_toList] at hspans
  obtain ⟨hr1, hsp⟩ := forIn_list_push (P := fun r => r.start + spanH r ≤ 2 ^ c.kLog ∧
      ∀ i, r.kind = .net i → (c.nets.getD i default).2.unitLog = r.slotLog)
    (g := fun r => (r.start, r.start + spanH r)) (fun r _ s t hs => by
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      generalize hp : (if r.packed = true then (r.count * 2 ^ r.slotLog, 2 ^ r.slotLog)
        else (nextPow2 r.count * 2 ^ r.slotLog, nextPow2 r.count * 2 ^ r.slotLog)) = p at hs
      have hspan : p.1 = spanH r := by rw [← hp]; unfold spanH; split <;> rfl
      obtain ⟨span, align⟩ := p
      simp only at hspan
      subst hspan
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      obtain ⟨h2, hs⟩ := ite_throw_ok' hs
      simp only [Bool.or_eq_true, not_or, decide_eq_true_eq, not_lt] at h2
      split at hs
      · rename_i i hk
        obtain ⟨h3, hs⟩ := ite_throw_ok' hs
        simp only [bne_iff_ne, ne_eq, not_not] at h3
        cases hs
        refine ⟨rfl, h2.2, fun i' hi' => ?_⟩
        rw [hk] at hi'
        cases hi'
        exact h3
      · rename_i hk
        obtain ⟨-, hs⟩ := ite_throw_ok' hs
        cases hs
        refine ⟨rfl, h2.2, fun i' hi' => ?_⟩
        rw [hk] at hi'
        cases hi'
      · exact absurd hs throw_bind_ne) hspans
  obtain ⟨u, hpair, h⟩ := except_bind_ok h
  have hp := forIn_range_forall (P := fun i => ∀ j, i + 1 ≤ j → j < spans.size →
      ¬ (spans[i]!.1 < spans[j]!.2 ∧ spans[j]!.1 < spans[i]!.2)) (fun i _ _ s t hs => by
    obtain ⟨u', hu, hs⟩ := except_bind_ok hs
    cases hs
    refine ⟨⟨_, rfl⟩, ?_⟩
    exact forIn_range_forall (P := fun j => ¬ (spans[i]!.1 < spans[j]!.2 ∧ spans[j]!.1 < spans[i]!.2))
      (fun j _ _ s t hs => by
        obtain ⟨hc, hs⟩ := ite_throw_ok' hs
        cases hs
        refine ⟨⟨_, rfl⟩, ?_⟩
        simpa using hc) hu) hpair
  have hspans_eq : spans = c.ranges.map (fun r => (r.start, r.start + spanH r)) := by
    apply Array.ext'
    rw [hsp, Array.toList_map]
    simp
  have hg : ∀ i (hi : i < c.ranges.size),
      spans[i]! = (c.ranges[i].start, c.ranges[i].start + spanH c.ranges[i]) := by
    intro i hi
    rw [hspans_eq, getElem!_pos _ _ (by simpa using hi)]
    simp
  have hsz : spans.size = c.ranges.size := by rw [hspans_eq, Array.size_map]
  have disj : c.ranges.toList.Pairwise Apart := by
    rw [List.pairwise_iff_getElem]
    intro i j hi hj hij
    have hi' : i < c.ranges.size := by simpa using hi
    have hj' : j < c.ranges.size := by simpa using hj
    have := hp i (Nat.zero_le _) (by omega) j hij (by omega)
    rw [hg i hi', hg j hj'] at this
    simpa [Apart] using this
  obtain ⟨hlk, h⟩ := ite_throw_ok' h
  obtain ⟨hpv, h⟩ := ite_throw_ok' h
  obtain ⟨hports, h⟩ := ite_throw_ok' h
  simp only [ht] at h
  obtain ⟨hur, h⟩ := ite_throw_ok' h
  obtain ⟨hug, h⟩ := ite_throw_ok' h
  replace h := ite_join h fun ha => (ite_throw_ok' ha).2
  obtain ⟨u1, hcount, h⟩ := except_bind_ok h
  obtain ⟨u2, hone, h⟩ := except_bind_ok h
  obtain ⟨wired, hwires, h⟩ := except_bind_ok h
  obtain ⟨wired2, hcuts, h⟩ := except_bind_ok h
  obtain ⟨u3, hwired, h⟩ := except_bind_ok h
  obtain ⟨hlv, h⟩ := ite_throw_ok' h
  obtain ⟨seen, hseen, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_ok' h
  obtain ⟨hzero, h⟩ := ite_throw_ok' h
  obtain ⟨hupv, h⟩ := ite_throw_ok' h
  obtain ⟨-, h⟩ := ite_throw_ok' h
  obtain ⟨hgp, -⟩ := ite_throw_ok' h
  have wire_leaf := forIn_array_forall (P := fun w : Wire => w.dst.1 = c.unitNet →
      (c.unit.inGroups.getD 0 default).bits.size ≤ w.dst.2.2)
    (fun w _ s t hs => by
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      cases hs
      refine ⟨⟨_, rfl⟩, fun hd => ?_⟩
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, not_and, not_lt] at hc
      exact hc hd) hwires
  have hw2 : wired = c.wires.toList.foldl (fun s w => bump s w.dst) #[] :=
    forIn_array_fold (fun w s => bump s w.dst) (fun w _ s t hs => by
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      cases hs
      rfl) hwires
  have cut_leaf := forIn_array_forall (P := fun e : ℕ × ℕ × ℕ => (c.unit.inGroups.getD 0 default).bits.size ≤ e.2.1)
    (fun e _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      cases hs
      refine ⟨⟨_, rfl⟩, ?_⟩
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_le, not_lt] at hc
      exact hc.1.1.1.2) hcuts
  have hc2 : wired2 = c.leafCuts.toList.foldl (fun s e => bump s (c.unitNet, e.1, e.2.1)) wired :=
    forIn_array_fold (fun e s => bump s (c.unitNet, e.1, e.2.1)) (fun e _ s t hs => by
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      cases hs
      rfl) hcuts
  have wiredOnce := forIn_range_forall (P := fun n => n ≠ c.shaNet → n ≠ c.hmNet → ∀ i < c.perVu.getD n 0, ∀ p,
      (if n = c.unitNet then (c.unit.inGroups.getD 0 default).bits.size else 0) ≤ p →
      p < (portCols (c.nets.getD n default).2.inGroups).size → wcount wired2 (n, i, p) = 1)
    (fun n _ hn s t hs => by
      split at hs
      · rename_i hsk
        cases hs
        refine ⟨⟨_, rfl⟩, fun h1 h2 => absurd hsk ?_⟩
        simp [h1, h2]
      · obtain ⟨u', hu, hs⟩ := except_bind_ok hs
        cases hs
        refine ⟨⟨_, rfl⟩, fun _ _ i hi p hp1 hp2 => ?_⟩
        have hi' := forIn_range_forall (P := fun i => ∀ p,
            (if n = c.unitNet then (c.unit.inGroups.getD 0 default).bits.size else 0) ≤ p →
            p < (portCols (c.nets.getD n default).2.inGroups).size → wcount wired2 (n, i, p) = 1)
          (fun i _ _ s t hs => by
            obtain ⟨u'', hu', hs⟩ := except_bind_ok hs
            cases hs
            refine ⟨⟨_, rfl⟩, fun p hp1 hp2 => ?_⟩
            have := forIn_range_forall (P := fun p => wcount wired2 (n, i, p) = 1)
              (fun p _ _ s t hs => by
                obtain ⟨hc, hs⟩ := ite_throw_ok' hs
                cases hs
                refine ⟨⟨_, rfl⟩, ?_⟩
                simpa [bne_iff_ne, wcount] using hc) hu' p (by simpa [beq_iff_eq] using hp1)
              (by simpa [Array.getD, hn] using hp2)
            exact this) hu
        exact hi' i (Nat.zero_le _) hi p hp1 hp2) hwired
  have count := forIn_array_forall (P := fun r => ∀ n, r.kind = .net n → r.count = c.g * c.perVu.getD n 0)
    (fun r _ s t hs => by
      split at hs
      · rename_i n hk
        obtain ⟨hc, hs⟩ := ite_throw_ok' hs
        cases hs
        refine ⟨⟨_, rfl⟩, fun n' hn' => ?_⟩
        rw [hk] at hn'
        cases hn'
        simp only [bne_iff_ne, ne_eq, not_not] at hc
        exact hc
      · rename_i hk
        cases hs
        exact ⟨⟨_, rfl⟩, fun n' hn' => absurd hn' (hk n')⟩) hcount
  have one_range := forIn_range_forall (P := fun n => (c.ranges.filter (·.kind == Kind.net n)).size = 1)
    (fun n _ _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      cases hs
      simp only [bne_iff_ne, ne_eq, not_not] at hc
      exact ⟨⟨_, rfl⟩, hc⟩) hone
  have wires := forIn_array_forall (P := fun w : Wire => w.dst.1 < c.nets.size ∧ w.dst.2.1 < c.perVu.getD w.dst.1 0 ∧
      w.dst.2.2 < (portCols (c.nets.getD w.dst.1 default).2.inGroups).size)
    (fun w _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      cases hs
      refine ⟨⟨_, rfl⟩, ?_⟩
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_le] at hc
      exact ⟨hc.1.1.1.1.1.1, hc.1.1.1.1.2, hc.1.1.2⟩) hwires
  have wire_src := forIn_array_forall (P := fun w : Wire => w.src.1 < c.nets.size ∧
      w.src.2.1 < c.perVu.getD w.src.1 0 ∧ w.src.2.2 < (portCols (c.nets.getD w.src.1 default).2.outGroups).size ∧
      (portCols (c.nets.getD w.dst.1 default).2.inGroups)[w.dst.2.2]!.2 =
        (portCols (c.nets.getD w.src.1 default).2.outGroups)[w.src.2.2]!.2)
    (fun w _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      cases hs
      refine ⟨⟨_, rfl⟩, ?_⟩
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_le, bne_iff_ne, ne_eq, not_not] at hc
      exact ⟨hc.1.1.1.1.1.2, hc.1.1.1.2, hc.1.2, hc.2⟩) hwires
  have cuts := forIn_array_forall (P := fun e : ℕ × ℕ × ℕ => e.1 < c.upv ∧ e.2.1 < (portCols c.unit.inGroups).size)
    (fun e _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      cases hs
      refine ⟨⟨_, rfl⟩, ?_⟩
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_le] at hc
      exact ⟨hc.1.1.1.1, hc.1.1.2⟩) hcuts
  have leaves := forIn_range_forall
    (P := fun u => (c.leavesIn[u]!).size = (c.unit.inGroups.getD 0 default).bits.size)
    (fun u _ _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      obtain ⟨s', -, hs⟩ := except_bind_ok hs
      cases hs
      refine ⟨⟨_, rfl⟩, ?_⟩
      simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not] at hc
      exact hc.1.1) hseen
  have leavesOut := forIn_range_forall
    (P := fun u => c.outNet = c.unitNet → (c.leavesOut[u]!).size = (c.unit.outGroups.getD 0 default).bits.size)
    (fun u _ _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      obtain ⟨s', -, hs⟩ := except_bind_ok hs
      cases hs
      refine ⟨⟨_, rfl⟩, fun ho => ?_⟩
      simp only [Bool.or_eq_true, not_or] at hc
      simpa [ho] using hc.2) hseen
  have hpv' : ¬(c.perVu.getD c.shaNet 0 != (HmRow.rowSlots c.ports).1 ||
      c.perVu.getD c.hmNet 0 != nextPow2 c.ports.size) = true := hpv
  simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not] at hpv' hur hlv
  simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not, List.any_eq_true, List.mem_range, not_exists,
    not_and] at hports
  simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not, Array.any_eq_true', not_exists, not_and,
    Array.isEmpty_iff_size_eq_zero] at hug
  exact {
    bound_slotLog := hr1
    disj := disj
    unit_lookup := Bool.eq_false_iff.2 hlk
    sha_in := ⟨by rw [hports.1.1.1.1]; rfl, fun i hi => hports.1.1.1.2 i (by rw [hports.1.1.1.1]; exact hi)⟩
    hm_in := hports.1.2
    perVu_sha := hpv'.1
    perVu_hm := hpv'.2
    unit_range := hur.1
    unit_groups := ⟨hug.1.1.1.1.1, fun b hb => hug.1.2 b (Array.mem_def.2 hb)⟩
    count := count
    one_range := fun n hn => one_range n (Nat.zero_le _) hn
    wires := wires
    wire_src := wire_src
    cuts := cuts
    leaves := ⟨hlv.1, fun u hu => leaves u (Nat.zero_le _) hu⟩
    leavesOut := ⟨hlv.2, fun u hu => leavesOut u (Nat.zero_le _) hu⟩
    wire_leaf := wire_leaf
    cut_leaf := cut_leaf
    wired := fun n hn h1 h2 i hi p hp1 hp2 => by
      have := wiredOnce n (Nat.zero_le _) hn h1 h2 i hi p hp1 hp2
      rw [hc2, wcount_foldl, hw2, wcount_foldl, wcount_empty, Nat.zero_add] at this
      exact this
    unit_zero := Nat.lt_of_not_le hzero
    unit_pow2 := fun ho => by simpa [ho] using hur.2
    upv_pos := Nat.pos_of_ne_zero (by simpa using hupv)
    g_pow2 := by simpa using hgp }


/-- **What `HmRow.check` accepted, for a typed template** (`c.typed = some tb`): the ranges' common facts, one range
of `G · per_vu` slots per circuit, the root one slot per template instance, each placed layout's circuit one slot per part, its slot size
its range's, and a forced-zero row in the root's slot. -/
structure CheckFactsT (c : Circuit) (tb : TypedBlock) : Prop where
  bound_slotLog : ∀ r ∈ c.ranges.toList, r.start + spanH r ≤ 2 ^ c.kLog ∧
    ∀ i, r.kind = .net i → (c.nets.getD i default).2.unitLog = r.slotLog
  disj : c.ranges.toList.Pairwise Apart
  unit_lookup : c.unit.lookup.isSome = false
  count : ∀ r ∈ c.ranges.toList, ∀ n, r.kind = .net n → r.count = c.g * c.perVu.getD n 0
  one_range : ∀ n < c.nets.size, (c.ranges.filter (·.kind == Kind.net n)).size = 1
  root_perVu : c.perVu.getD c.unitNet 0 = 1
  rangeLogs : ∀ e ∈ tb.rangeLogs.toList, c.perVu.getD e.1 0 = (tb.parts.filter (·.1 == e.1)).size ∧
    (c.nets.getD e.1 default).2.unitLog = e.2
  root_zero : c.unit.useful < 2 ^ c.unit.unitLog
  sha_in : (portCols (c.nets.getD c.shaNet default).2.inGroups).size = 6 ∧
    ∀ i < 6, ((portCols (c.nets.getD c.shaNet default).2.inGroups)[i]!).2 = if i % 2 == 0 then 512 else 1024
  hm_in : (portCols (c.nets.getD c.hmNet default).2.inGroups).map (·.2) = #[512, 1024, 8 * HmRow.SALT_BYTES]
  perVu_sha : c.perVu.getD c.shaNet 0 = (HmRow.rowSlots c.ports).1
  perVu_hm : c.perVu.getD c.hmNet 0 = nextPow2 c.ports.size
  rows_after : tb.rangeLogs.size < c.shaNet ∧ tb.rangeLogs.size < c.hmNet
  ins : tb.ins = c.ports.map fun p => (p.words, 16)

theorem check_facts_typed {c : Circuit} {tags : Tags} {ls : Lean.Json} {tb : TypedBlock} (ht : c.typed = some tb)
    (h : HmRow.check c tags ls = .ok ()) : CheckFactsT c tb := by
  unfold HmRow.check at h
  obtain ⟨spans, hspans, h⟩ := except_bind_ok h
  rw [← Array.forIn_toList] at hspans
  obtain ⟨hr1, hsp⟩ := forIn_list_push (P := fun r => r.start + spanH r ≤ 2 ^ c.kLog ∧
      ∀ i, r.kind = .net i → (c.nets.getD i default).2.unitLog = r.slotLog)
    (g := fun r => (r.start, r.start + spanH r)) (fun r _ s t hs => by
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      generalize hp : (if r.packed = true then (r.count * 2 ^ r.slotLog, 2 ^ r.slotLog)
        else (nextPow2 r.count * 2 ^ r.slotLog, nextPow2 r.count * 2 ^ r.slotLog)) = p at hs
      have hspan : p.1 = spanH r := by rw [← hp]; unfold spanH; split <;> rfl
      obtain ⟨span, align⟩ := p
      simp only at hspan
      subst hspan
      obtain ⟨-, hs⟩ := ite_throw_ok' hs
      obtain ⟨h2, hs⟩ := ite_throw_ok' hs
      simp only [Bool.or_eq_true, not_or, decide_eq_true_eq, not_lt] at h2
      split at hs
      · rename_i i hk
        obtain ⟨h3, hs⟩ := ite_throw_ok' hs
        simp only [bne_iff_ne, ne_eq, not_not] at h3
        cases hs
        refine ⟨rfl, h2.2, fun i' hi' => ?_⟩
        rw [hk] at hi'
        cases hi'
        exact h3
      · rename_i hk
        obtain ⟨-, hs⟩ := ite_throw_ok' hs
        cases hs
        refine ⟨rfl, h2.2, fun i' hi' => ?_⟩
        rw [hk] at hi'
        cases hi'
      · exact absurd hs throw_bind_ne) hspans
  obtain ⟨u, hpair, h⟩ := except_bind_ok h
  have hp := forIn_range_forall (P := fun i => ∀ j, i + 1 ≤ j → j < spans.size →
      ¬ (spans[i]!.1 < spans[j]!.2 ∧ spans[j]!.1 < spans[i]!.2)) (fun i _ _ s t hs => by
    obtain ⟨u', hu, hs⟩ := except_bind_ok hs
    cases hs
    refine ⟨⟨_, rfl⟩, ?_⟩
    exact forIn_range_forall (P := fun j => ¬ (spans[i]!.1 < spans[j]!.2 ∧ spans[j]!.1 < spans[i]!.2))
      (fun j _ _ s t hs => by
        obtain ⟨hc, hs⟩ := ite_throw_ok' hs
        cases hs
        refine ⟨⟨_, rfl⟩, ?_⟩
        simpa using hc) hu) hpair
  have hspans_eq : spans = c.ranges.map (fun r => (r.start, r.start + spanH r)) := by
    apply Array.ext'
    rw [hsp, Array.toList_map]
    simp
  have hg : ∀ i (hi : i < c.ranges.size),
      spans[i]! = (c.ranges[i].start, c.ranges[i].start + spanH c.ranges[i]) := by
    intro i hi
    rw [hspans_eq, getElem!_pos _ _ (by simpa using hi)]
    simp
  have hsz : spans.size = c.ranges.size := by rw [hspans_eq, Array.size_map]
  have disj : c.ranges.toList.Pairwise Apart := by
    rw [List.pairwise_iff_getElem]
    intro i j hi hj hij
    have hi' : i < c.ranges.size := by simpa using hi
    have hj' : j < c.ranges.size := by simpa using hj
    have := hp i (Nat.zero_le _) (by omega) j hij (by omega)
    rw [hg i hi', hg j hj'] at this
    simpa [Apart] using this
  obtain ⟨hlk, h⟩ := ite_throw_ok' h
  obtain ⟨hpv, h⟩ := ite_throw_ok' h
  obtain ⟨hports, h⟩ := ite_throw_ok' h
  simp only [ht] at h
  obtain ⟨u1, hcount, h⟩ := except_bind_ok h
  obtain ⟨u2, hone, h⟩ := except_bind_ok h
  have count := forIn_array_forall (P := fun r => ∀ n, r.kind = .net n → r.count = c.g * c.perVu.getD n 0)
    (fun r _ s t hs => by
      split at hs
      · rename_i n hk
        obtain ⟨hc, hs⟩ := ite_throw_ok' hs
        cases hs
        refine ⟨⟨_, rfl⟩, fun n' hn' => ?_⟩
        rw [hk] at hn'
        cases hn'
        simp only [bne_iff_ne, ne_eq, not_not] at hc
        exact hc
      · rename_i hk
        cases hs
        exact ⟨⟨_, rfl⟩, fun n' hn' => absurd hn' (hk n')⟩) hcount
  have one_range := forIn_range_forall (P := fun n => (c.ranges.filter (·.kind == Kind.net n)).size = 1)
    (fun n _ _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      cases hs
      simp only [bne_iff_ne, ne_eq, not_not] at hc
      exact ⟨⟨_, rfl⟩, hc⟩) hone
  unfold HmRow.checkTyped at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h)
  obtain ⟨hrows, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jp0] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) root nOut jp1 at h)
  obtain ⟨hpv1, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jp1] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h)
  obtain ⟨hio, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jp2] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h)
  obtain ⟨-, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jp3] at h
  obtain ⟨u3, hrl, h⟩ := except_bind_ok h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h)
  obtain ⟨hz, -⟩ := ite_throw_ok' h
  have rangeLogs := forIn_array_forall (P := fun e : ℕ × ℕ => c.perVu.getD e.1 0 = (tb.parts.filter (·.1 == e.1)).size ∧
      (c.nets.getD e.1 default).2.unitLog = e.2)
    (fun e _ s t hs => by
      obtain ⟨n, sl⟩ := e
      obtain ⟨hc, hs⟩ := ite_throw_ok' hs
      cases hs
      simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not] at hc
      exact ⟨⟨_, rfl⟩, hc⟩) hrl
  simp only [bne_iff_ne, ne_eq, not_not] at hpv1
  have hpv' : ¬(c.perVu.getD c.shaNet 0 != (HmRow.rowSlots c.ports).1 ||
      c.perVu.getD c.hmNet 0 != nextPow2 c.ports.size) = true := hpv
  simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not] at hpv'
  simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not, List.any_eq_true, List.mem_range, not_exists,
    not_and] at hports
  exact {
    bound_slotLog := hr1
    disj := disj
    unit_lookup := by simpa using hlk
    count := count
    one_range := fun n hn => one_range n (Nat.zero_le _) hn
    root_perVu := hpv1
    rangeLogs := rangeLogs
    root_zero := Nat.lt_of_not_le hz
    sha_in := ⟨by rw [hports.1.1.1.1]; rfl, fun i hi => hports.1.1.1.2 i (by rw [hports.1.1.1.1]; exact hi)⟩
    hm_in := hports.1.2
    perVu_sha := hpv'.1
    perVu_hm := hpv'.2
    rows_after := by
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_le] at hrows
      exact hrows
    ins := by
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hio
      exact hio.1 }

end FlockSoundness
