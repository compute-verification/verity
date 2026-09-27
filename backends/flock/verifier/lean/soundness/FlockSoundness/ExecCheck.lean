import FlockSoundness.ExecPlacement

/-!
# What `HmRow.check` accepted

The layout facts of an `hm96-sha512/row/v1` circuit that `HmRow.check` checks, as `ExecSetup.lean` uses them: the
ranges inside the block, pairwise apart, in slots of their nets' rows; the unit a text net with one range of
`vus_per_block × units_per_vu` slots and a group of 16-bit leaf ports; the row slots' ports; every net range's slots per
VU; one range per net; the wires' and leaf cuts' bounds; the leaf maps.
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
  cuts : ∀ e ∈ c.leafCuts.toList, e.1 < c.upv ∧ e.2.1 < (portCols c.unit.inGroups).size
  leaves : c.leavesIn.size = c.upv ∧
    ∀ u < c.upv, (c.leavesIn[u]!).size = (c.unit.inGroups.getD 0 default).bits.size

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

/-- **What `HmRow.check` accepted.** -/
theorem check_facts {c : Circuit} {tags : Tags} {ls : Lean.Json} (h : HmRow.check c tags ls = .ok ()) :
    CheckFacts c := by
  unfold HmRow.check at h
  obtain ⟨spans, hspans, h⟩ := except_bind_ok h
  rw [← Array.forIn_toList] at hspans
  obtain ⟨hr1, hsp⟩ := forIn_list_push (P := fun r => r.start + spanH r ≤ 2 ^ c.kLog ∧
      ∀ i, r.kind = .net i → (c.nets.getD i default).2.unitLog = r.slotLog)
    (g := fun r => (r.start, r.start + spanH r)) (fun r _ s t hs => by
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
  obtain ⟨hur, h⟩ := ite_throw_ok' h
  obtain ⟨hug, h⟩ := ite_throw_ok' h
  replace h := ite_join h fun ha => (ite_throw_ok' ha).2
  obtain ⟨u1, hcount, h⟩ := except_bind_ok h
  obtain ⟨u2, hone, h⟩ := except_bind_ok h
  obtain ⟨wired, hwires, h⟩ := except_bind_ok h
  obtain ⟨wired2, hcuts, h⟩ := except_bind_ok h
  obtain ⟨u3, -, h⟩ := except_bind_ok h
  obtain ⟨hlv, h⟩ := ite_throw_ok' h
  obtain ⟨seen, hseen, -⟩ := except_bind_ok h
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
    cuts := cuts
    leaves := ⟨hlv.1, fun u hu => leaves u (Nat.zero_le _) hu⟩ }

end FlockSoundness
