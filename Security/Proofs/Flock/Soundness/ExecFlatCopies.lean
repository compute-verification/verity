import Proofs.Flock.Soundness.ExecFlatClass
import Proofs.Flock.Soundness.Types.ProgramPlaces

/-!
# A typed flat class's input copies and forced-zero rows (1e for flat classes)

A flat statement's Δ (`HmRow.delta`, no typed template) is its slot constants' pairs, the rows' constant bits, then each
unit copy's leaf bits, cut bits and wired bits (`delta_flat`). `HmRow.check` wires each of a unit's input ports exactly
once (`CheckFacts.wired`, leaves by `leaves_in`), and #434's `Typed.flatNet` makes the unit's inputs its net's input port
bits in order, so the ports' bits are distinct columns. At each of a unit copy's input port bits, Δ then has one copy
pair, over the net's self row there:
- `Layout.delta_port`: the count of Δ's pairs at a port bit is its own and its source's (`flatSrc`);
- `Layout.placed_port`: the block rows there read the source, so they copy it, or are zero when it lies outside the
  block;
- `Layout.placed_zero`: each unit slot's row past its useful rows is a forced-zero row;
- `setupH_flatTableClass`: the flat class's `TableClass`, from the statement `Stmt.setupH` accepted.
-/

open Flock FlockLevel3

namespace FlockSoundness

open Compose Flock.DeriveAll Types.Dag

/-! ## Δ, pair by pair -/

theorem forIn_range_append {γ : Type} (G : ℕ → List γ) (n : ℕ) {f : ℕ → Array γ → Id (ForInStep (Array γ))}
    (hf : ∀ a < n, ∀ d, ∃ d', f a d = ForInStep.yield d' ∧ d'.toList = d.toList ++ G a) (d : Array γ) :
    (forIn [:n] d f).toList = d.toList ++ (List.range n).flatMap G := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := forIn_id_append G (List.range' 0 [:n].size) (fun a ha d => hf a (by
    simp only [Std.Legacy.Range.size, List.mem_range'_1] at ha; omega) d) d
  simpa [Std.Legacy.Range.size, List.range_eq_range'] using this

theorem forIn_array_append {α γ : Type} (G : α → List γ) (xs : Array α) {f : α → Array γ → Id (ForInStep (Array γ))}
    (hf : ∀ a ∈ xs.toList, ∀ d, ∃ d', f a d = ForInStep.yield d' ∧ d'.toList = d.toList ++ G a) (d : Array γ) :
    (forIn xs d f).toList = d.toList ++ xs.toList.flatMap G := by
  rw [← Array.forIn_toList]
  exact forIn_id_append G xs.toList hf d

namespace Flat

/-- Unit copy `k`'s leaf `j`, bit `t`: its column. -/
def leafCol (c : Circuit) (k j t : ℕ) : ℕ :=
  c.unitRange.slot k + ((portCols #[c.unit.inGroups.getD 0 default]).map (·.1)).getD j 0 + t

/-- Unit copy `k`'s leaf `j`, bit `t`: the message bit it copies, or the copy's forced-zero row for a zero leaf or a bit
row's leaf bit past its length. -/
def leafSrc (c : Circuit) (k j t : ℕ) : ℕ :=
  if (c.leavesIn.getD (k % c.upv) #[])[j]! < 0 then c.unitRange.slot k + c.unit.useful else
    let x := ((c.leavesIn.getD (k % c.upv) #[])[j]!).toNat
    let p := ((List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ x)).getD 0
    if 16 * (x - (Typed.starts c)[p]!) + t < c.ports[p]!.bits then
      Typed.msgCol c (k / c.upv) p (2 * (x - (Typed.starts c)[p]!) + t / 8) (t % 8)
    else c.unitRange.slot k + c.unit.useful

/-- `HmRow.delta`'s leaf pairs: each unit copy's leaf bits copy their message bits. -/
def leafPairs (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.unitRange.count).flatMap fun k => (List.range (c.leavesIn.getD (k % c.upv) #[]).size).flatMap fun j =>
    (List.range 16).flatMap fun t => [(leafCol c k j t, leafCol c k j t), (leafCol c k j t, leafSrc c k j t)]

/-- Template instance `g`'s cut `e`: its port's first column. -/
def cutCol (c : Circuit) (g : ℕ) (e : ℕ × ℕ × ℕ) : ℕ := c.slotCol c.unitNet g e.1 + (portCols c.unit.inGroups)[e.2.1]!.1

/-- Template instance `g`'s cut `e`, bit `t`: its row word's message bit, or the forced-zero row past the word or the
row's length. -/
def cutSrc (c : Circuit) (g : ℕ) (e : ℕ × ℕ × ℕ) (t : ℕ) : ℕ :=
  let p := ((List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ e.2.2)).getD 0
  if t < 16 && 16 * (e.2.2 - (Typed.starts c)[p]!) + t < c.ports[p]!.bits then
    Typed.msgCol c g p (2 * (e.2.2 - (Typed.starts c)[p]!) + t / 8) (t % 8)
  else c.unitRange.slot 0 + c.unit.useful

/-- `HmRow.delta`'s cut pairs: each template instance's cut ports copy their row words' bits. -/
def cutPairs (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => c.leafCuts.toList.flatMap fun e =>
    (List.range (portCols c.unit.inGroups)[e.2.1]!.2).flatMap fun t =>
      [(cutCol c g e + t, cutCol c g e + t), (cutCol c g e + t, cutSrc c g e t)]

/-- Template instance `g`'s wire `w`: its destination port's first column. -/
def wireDst (c : Circuit) (g : ℕ) (w : Wire) : ℕ :=
  c.slotCol w.dst.1 g w.dst.2.1 + ((portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).1

/-- Template instance `g`'s wire `w`: its source port's first column. -/
def wireSrc (c : Circuit) (g : ℕ) (w : Wire) : ℕ :=
  c.slotCol w.src.1 g w.src.2.1 + ((portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default).1

/-- A wire's width: its destination port's. -/
def wireWidth (c : Circuit) (w : Wire) : ℕ := ((portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).2

/-- `HmRow.delta`'s wire pairs: each template instance's wired input bits copy their sources. -/
def wirePairs (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => c.wires.toList.flatMap fun w =>
    (List.range (wireWidth c w)).flatMap fun t =>
      [(wireDst c g w + t, wireDst c g w + t), (wireDst c g w + t, wireSrc c g w + t)]

/-- A row circuit's bit that Δ fills: a bit of a row's chain, or of its hm96 slot (`DeltaIn`'s first two forms). -/
def RowBit (c : Circuit) (x : ℕ) : Prop := ShaRowBit c x ∨ HmRowBit c x

/-- Unit copy `s₀`'s input port `p`, bit `t`: the column Δ copies into it. A leaf port's leaf (`leafSrc`), else the cut
into it (`cutSrc`), else the wire into it (`wireSrc`); `HmRow.check` wires each port exactly once. -/
def flatSrc (c : Circuit) (s₀ p t : ℕ) : ℕ :=
  if p < (c.unit.inGroups.getD 0 default).bits.size then leafSrc c s₀ p t else
    match c.leafCuts.toList.filter (fun e => decide (e.1 = s₀ % c.upv ∧ e.2.1 = p)) with
    | e :: _ => cutSrc c (s₀ / c.upv) e t
    | [] => match c.wires.toList.filter (fun w => decide (w.dst = (c.unitNet, s₀ % c.upv, p))) with
      | w :: _ => wireSrc c (s₀ / c.upv) w + t
      | [] => 0

end Flat

/-- Δ, pair by pair, for a flat statement: the slot constants' pairs, the rows' constant bits, then the units' leaf bits,
cut bits and wired bits, in order. -/
def FlatDelta (c : Circuit) (pin : ℕ) (d : Array (ℕ × ℕ)) : Prop :=
  ∃ rows, d.toList = constPairs c pin ++ rows ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c ∧
    ∀ p ∈ rows, Flat.RowBit c p.1

/-- Δ's constants' pairs, then only the rows' constant bits. -/
def FlatRowsQ (c : Circuit) (pin : ℕ) (d : Array (ℕ × ℕ)) : Prop :=
  ∃ rows, d.toList = constPairs c pin ++ rows ∧ ∀ p ∈ rows, Flat.RowBit c p.1

/-- **A flat statement's Δ, pair by pair** (`HmRow.delta`, no typed template). -/
theorem delta_flat (c : Circuit) (pin : ℕ) (ht : c.typed = none) : FlatDelta c pin (HmRow.delta c pin).1 := by
  unfold HmRow.delta
  simp only
  refine run_bind_fst (Q := FlatDelta c pin) (P := fun d => d.toList = constPairs c pin) ?_ ?_
  · rw [← Array.forIn_toList]
    refine (forIn_id_append (slotPairs c pin) _ (fun r _ d => ?_) #[]).trans ?_
    rotate_left
    · simp [constPairs]
    · split
      · rename_i i hk
        refine ⟨_, rfl, ?_⟩
        simp only [slotPairs, hk]
        rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_id_append (fun s =>
          if (r.slot s + (c.nets.getD i default).2.constPos != pin) = true then
            [(r.slot s + (c.nets.getD i default).2.constPos, r.slot s + (c.nets.getD i default).2.constPos),
              (r.slot s + (c.nets.getD i default).2.constPos, pin)] else []) _ (fun s _ d => ?_) d]
        · simp [List.range_eq_range']
        · split
          · exact ⟨_, rfl, by simp⟩
          · exact ⟨_, rfl, by simp⟩
      · rename_i hk
        refine ⟨_, rfl, ?_⟩
        unfold slotPairs
        split
        · rename_i i hi; exact absurd hi (hk i)
        · simp
  intro d hd
  have h0 : FlatRowsQ c pin d := ⟨[], by simp [hd], by simp⟩
  -- the rows' constant bits: each row's first cv, its hm96 pad's, its chain's message bits past its length
  refine run_bind_fst (Q := FlatDelta c pin) (P := FlatRowsQ c pin)
    (forIn_id_inv_range (FlatRowsQ c pin) (fun g _ hg d hd => ⟨_, rfl,
      forIn_id_inv_range (FlatRowsQ c pin) (fun p _ hp d hd => ?_) hd⟩) h0) ?_
  · have push2 : ∀ {d : Array (ℕ × ℕ)} {i a b : ℕ},
        (∃ rows, d.toList = constPairs c pin ++ rows ∧ ∀ p ∈ rows, Flat.RowBit c p.1) → Flat.RowBit c i →
        ∃ rows, ((d.push (i, a)).push (i, b)).toList = constPairs c pin ++ rows ∧ ∀ p ∈ rows, Flat.RowBit c p.1 := by
      rintro d i a b ⟨rows, hr, hq⟩ hi
      refine ⟨rows ++ [(i, a), (i, b)], by simp [hr], fun q hq' => ?_⟩
      simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hq'
      rcases hq' with hq' | rfl | rfl
      · exact hq q hq'
      · exact hi
      · exact hi
    split
    · -- a segmented row: each chain start's midstate, then every constant or digest-copied message bit
      refine ⟨_, rfl, forIn_id_inv_range (FlatRowsQ c pin) (fun k _ hk d hd => ?_) hd⟩
      split
      · refine ⟨_, rfl, forIn_id_inv_range (FlatRowsQ c pin) (fun t _ ht d hd => ?_)
          (forIn_id_inv_range (FlatRowsQ c pin) (fun t _ ht d hd => ⟨_, rfl, push2 hd ?_⟩) hd)⟩
        · split
          · refine ⟨_, rfl, push2 hd ?_⟩
            split
            · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 1, by omega, t, by omega, rfl⟩
            · exact Or.inr ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩
          · refine ⟨_, rfl, push2 hd ?_⟩
            split
            · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 1, by omega, t, by omega, rfl⟩
            · exact Or.inr ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩
          · exact ⟨_, rfl, hd⟩
        · split
          · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 0, by omega, t, by omega, rfl⟩
          · exact Or.inr ⟨g, hg, p, hp, 0, by omega, t, by omega, rfl⟩
      · refine ⟨_, rfl, forIn_id_inv_range (FlatRowsQ c pin) (fun t _ ht d hd => ?_) hd⟩
        split
        · refine ⟨_, rfl, push2 hd ?_⟩
          split
          · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 1, by omega, t, by omega, rfl⟩
          · exact Or.inr ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩
        · refine ⟨_, rfl, push2 hd ?_⟩
          split
          · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 1, by omega, t, by omega, rfl⟩
          · exact Or.inr ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩
        · exact ⟨_, rfl, hd⟩
    · refine ⟨_, rfl, ?_⟩
      refine forIn_id_inv_range (FlatRowsQ c pin) (fun k _ hk d hd => ⟨_, rfl,
          forIn_id_inv_range (FlatRowsQ c pin) (fun t _ ht d hd => ?_) hd⟩)
        (forIn_id_inv_range (FlatRowsQ c pin) (fun t _ ht d hd => ?_)
          (forIn_id_inv_range (FlatRowsQ c pin) (fun t _ ht d hd => ⟨_, rfl, push2 hd ?_⟩) hd))
      · split
        · exact ⟨_, rfl, push2 hd (Or.inl ⟨g, hg, p, hp, k, hk, 1, by omega, t, by omega, rfl⟩)⟩
        · exact ⟨_, rfl, hd⟩
      · split
        · exact ⟨_, rfl, push2 hd (Or.inr ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩)⟩
        · exact ⟨_, rfl, hd⟩
      · split
        · exact Or.inl ⟨g, hg, p, hp, 0, by assumption, 0, by omega, t, by omega, rfl⟩
        · exact Or.inr ⟨g, hg, p, hp, 0, by omega, t, by omega, rfl⟩
  rintro d ⟨rows, hrows, hrq⟩
  refine ⟨rows, ?_, hrq⟩
  -- the units' leaf bits
  refine run_bind_fst (Q := fun d' : Array (ℕ × ℕ) => d'.toList =
      constPairs c pin ++ rows ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c)
    (P := fun d' => d'.toList = d.toList ++ Flat.leafPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d1 hd1
  -- the units' leaf cuts
  refine run_bind_fst (Q := fun d' : Array (ℕ × ℕ) => d'.toList =
      constPairs c pin ++ rows ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c)
    (P := fun d' => d'.toList = d1.toList ++ Flat.cutPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d1
    refine forIn_array_append _ _ (fun e _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d2 hd2
  -- the wires
  refine run_bind_fst (Q := fun d' : Array (ℕ × ℕ) => d'.toList =
      constPairs c pin ++ rows ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c)
    (P := fun d' => d'.toList = d2.toList ++ Flat.wirePairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d2
    refine forIn_array_append _ _ (fun w _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d3 hd3
  simp only [ht]
  show d3.toList = _
  rw [hd3, hd2, hd1, hrows]


/-! ## A net's input port bits -/

/-- A net's input port bits, port by port: each port's columns. -/
def portBits (gs : Array Group) : List ℕ := (portCols gs).toList.flatMap fun q => List.range' q.1 q.2

/-- Distinct input port bits are distinct columns. -/
theorem portBits_inj {gs : Array Group} (hnd : (portBits gs).Nodup) {p p' t t' : ℕ} (hp : p < (portCols gs).size)
    (hp' : p' < (portCols gs).size) (ht : t < (portCols gs)[p]!.2) (ht' : t' < (portCols gs)[p']!.2)
    (he : (portCols gs)[p]!.1 + t = (portCols gs)[p']!.1 + t') : p = p' ∧ t = t' := by
  rw [portBits, List.nodup_flatMap, List.pairwise_iff_getElem] at hnd
  have hq : ∀ i (hi : i < (portCols gs).size), (portCols gs)[i]! = (portCols gs).toList[i]'(by simpa using hi) :=
    fun i hi => by rw [getElem!_pos _ i hi, Array.getElem_toList]
  rw [hq p hp] at ht he
  rw [hq p' hp'] at ht' he
  have key : ∀ i j (hi : i < (portCols gs).size) (hj : j < (portCols gs).size), i < j → ∀ ti tj,
      ti < ((portCols gs).toList[i]'(by simpa using hi)).2 → tj < ((portCols gs).toList[j]'(by simpa using hj)).2 →
      ((portCols gs).toList[i]'(by simpa using hi)).1 + ti ≠ ((portCols gs).toList[j]'(by simpa using hj)).1 + tj := by
    intro i j hi hj hij ti tj hti htj heq
    exact hnd.2 i j (by simpa using hi) (by simpa using hj) hij
      (show ((portCols gs).toList[i]'(by simpa using hi)).1 + ti ∈ List.range' _ _ from
        List.mem_range'_1.2 ⟨by omega, by omega⟩)
      (show ((portCols gs).toList[i]'(by simpa using hi)).1 + ti ∈ List.range' _ _ from
        List.mem_range'_1.2 ⟨by omega, by omega⟩)
  rcases Nat.lt_trichotomy p p' with h | rfl | h
  · exact absurd he (key p p' hp hp' h t t' ht ht')
  · exact ⟨rfl, by omega⟩
  · exact absurd he.symm (key p' p hp' hp h t' t ht' ht)

/-- The port a column of the input rows is a bit of: the first whose columns hold it. -/
def portOf (pc : Array (ℕ × ℕ)) (off : ℕ) : ℕ :=
  ((List.range pc.size).find? fun p => decide (pc[p]!.1 ≤ off ∧ off < pc[p]!.1 + pc[p]!.2)).getD 0

theorem portOf_eq {gs : Array Group} (hnd : (portBits gs).Nodup) {p t : ℕ} (hp : p < (portCols gs).size)
    (ht : t < (portCols gs)[p]!.2) : portOf (portCols gs) ((portCols gs)[p]!.1 + t) = p := by
  unfold portOf
  obtain ⟨p', hf⟩ : ∃ p', (List.range (portCols gs).size).find?
      (fun p' => decide ((portCols gs)[p']!.1 ≤ (portCols gs)[p]!.1 + t ∧
        (portCols gs)[p]!.1 + t < (portCols gs)[p']!.1 + (portCols gs)[p']!.2)) = some p' := by
    rcases hf : (List.range (portCols gs).size).find?
        (fun p' => decide ((portCols gs)[p']!.1 ≤ (portCols gs)[p]!.1 + t ∧
          (portCols gs)[p]!.1 + t < (portCols gs)[p']!.1 + (portCols gs)[p']!.2)) with _ | p'
    · rw [List.find?_eq_none] at hf
      exact absurd (by simp; omega) (hf p (List.mem_range.2 hp))
    · exact ⟨p', hf⟩
  rw [hf, Option.getD_some]
  have h1 := List.find?_some hf
  have h2 := List.mem_range.1 (List.mem_of_find?_eq_some hf)
  simp only [decide_eq_true_eq] at h1
  exact (portBits_inj hnd h2 hp (t := (portCols gs)[p]!.1 + t - (portCols gs)[p']!.1) (t' := t) (by omega) ht
    (by omega)).1

/-- The ports of a net's first input group come first among its ports: port `j` of the group is port `j`. -/
theorem firstGroup_port {gs : Array Group} (hne : gs.size ≠ 0) {j : ℕ} (hj : j < (gs.getD 0 default).bits.size) :
    j < (portCols gs).size ∧ ((portCols #[gs.getD 0 default]).map (·.1)).getD j 0 = (portCols gs)[j]!.1 ∧
      (portCols gs)[j]!.2 = (gs.getD 0 default).bits[j]! := by
  have hsingle := portCols_single (gs.getD 0 default)
  have hlen : (portCols #[gs.getD 0 default]).size = (gs.getD 0 default).bits.size := by
    have := congrArg List.length hsingle
    simpa using this
  have hsplit : (portCols gs).toList = (portCols #[gs.getD 0 default]).toList ++
      (gs.toList.drop 1).flatMap fun g => (portCols #[g]).toList := by
    rw [portCols_toList]
    obtain ⟨g0, rest, hgs⟩ : ∃ g0 rest, gs.toList = g0 :: rest := by
      cases h : gs.toList with
      | nil => exact absurd (by simpa using congrArg List.length h) hne
      | cons g0 rest => exact ⟨g0, rest, rfl⟩
    have hg0 : gs.getD 0 default = g0 := by
      rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hgs]; rfl
    rw [hgs, hg0]
    simp
  have hjs1 : j < (portCols #[gs.getD 0 default]).size := by rw [hlen]; exact hj
  have hjl : j < (portCols #[gs.getD 0 default]).toList.length := by rw [Array.length_toList]; exact hjs1
  have hjs : j < (portCols gs).size := by
    have := congrArg List.length hsplit
    simp only [Array.length_toList, List.length_append] at this
    omega
  have hget : (portCols gs)[j]! = (portCols #[gs.getD 0 default])[j] := by
    rw [getElem!_pos _ j hjs, ← Array.getElem_toList, ← Array.getElem_toList (h := hjs1)]
    simp only [hsplit]
    exact List.getElem_append_left hjl
  refine ⟨hjs, ?_, ?_⟩
  · rw [hget, Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_eq_getElem hjs1]
    rfl
  · rw [hget]
    have := congrArg (fun l : List ℕ => l[j]?) hsingle
    simp only [List.getElem?_map, Array.getElem?_toList] at this
    rw [Array.getElem?_eq_getElem hjs1, Option.map_some, Array.getElem?_eq_getElem hj] at this
    rw [getElem!_pos _ j hj]
    exact Option.some.inj this

theorem count_pair_ne {a b x j : ℕ} (h : a ≠ x) : [(a, a), (a, b)].count (x, j) = 0 := by
  rw [List.count_eq_zero]
  simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq, not_or, not_and]
  exact ⟨fun h' => absurd h'.symm h, fun h' => absurd h'.symm h⟩

theorem count_pair_eq (x b j : ℕ) :
    [(x, x), (x, b)].count (x, j) = (if x = j then 1 else 0) + if b = j then 1 else 0 := by
  simp only [List.count_cons, List.count_nil, beq_iff_eq, Prod.mk.injEq, true_and]
  by_cases h1 : x = j <;> by_cases h2 : b = j <;> simp [h1, h2]

/-- A count over template instances, items and bits whose pairs land at `x` only at template instance `g₀`, bit `t₀`,
for the items `F` picks: each picked item's copy pair there. -/
theorem count_vu_items {α : Type} (n : ℕ) (l : List α) (W : α → ℕ) (D S : ℕ → α → ℕ → ℕ) (F : α → Prop)
    [DecidablePred F] {g₀ t₀ x j : ℕ} (hg₀ : g₀ < n)
    (hD : ∀ g < n, ∀ a ∈ l, ∀ tt < W a, D g a tt = x ↔ g = g₀ ∧ F a ∧ tt = t₀)
    (ht₀ : ∀ a ∈ l, F a → t₀ < W a) :
    ((List.range n).flatMap fun g => l.flatMap fun a => (List.range (W a)).flatMap fun tt =>
        [(D g a tt, D g a tt), (D g a tt, S g a tt)]).count (x, j) =
      ((l.filter fun a => decide (F a)).flatMap fun a => [(x, x), (x, S g₀ a t₀)]).count (x, j) := by
  rw [count_flatMap_range _ _ hg₀ (fun g hg hne => count_flatMap_zero fun a ha => count_flatMap_zero fun tt htt =>
    count_pair_ne fun he => hne ((hD g hg a ha tt (List.mem_range.1 htt)).1 he).1)]
  induction l with
  | nil => rfl
  | cons a l ih =>
    have ih' := ih (fun g hg a' ha' => hD g hg a' (List.mem_cons_of_mem _ ha'))
      (fun a' ha' => ht₀ a' (List.mem_cons_of_mem _ ha'))
    rw [List.flatMap_cons, List.count_append, ih']
    by_cases hF : F a
    · rw [List.filter_cons_of_pos (by simpa using hF), List.flatMap_cons, List.count_append,
        count_flatMap_range _ _ (ht₀ a List.mem_cons_self hF) (fun tt htt hne => count_pair_ne fun he =>
          hne ((hD g₀ hg₀ a List.mem_cons_self tt htt).1 he).2.2),
        ((hD g₀ hg₀ a List.mem_cons_self t₀ (ht₀ a List.mem_cons_self hF)).2 ⟨rfl, hF, rfl⟩)]
    · rw [List.filter_cons_of_neg (by simpa using hF), count_flatMap_zero fun tt htt => count_pair_ne fun he =>
        hF ((hD g₀ hg₀ a List.mem_cons_self tt (List.mem_range.1 htt)).1 he).2.1, Nat.zero_add]

theorem vu_split {upv s₀ g u : ℕ} (hu : u < upv) : g * upv + u = s₀ ↔ g = s₀ / upv ∧ u = s₀ % upv := by
  constructor
  · rintro rfl
    have hpos : 0 < upv := by omega
    refine ⟨?_, ?_⟩
    · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hu, Nat.zero_add]
    · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hu]
  · rintro ⟨rfl, rfl⟩
    exact Nat.div_add_mod' s₀ upv

/-! ## Positions in a unit's range -/

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

/-- A column of a net range's slot that is a unit slot's column is in the unit's range, in the same slot at the same
offset. -/
theorem eq_unit {R : Range} (hR : R ∈ st.c.ranges.toList) {n : ℕ} (hk : R.kind = .net n) {s a s₀ b : ℕ}
    (hs : s < R.count) (ha : a < 2 ^ R.slotLog) (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog)
    (he : R.slot s + a = st.c.unitRange.slot s₀ + b) : n = st.c.unitNet ∧ s = s₀ ∧ a = b := by
  have hb' : b < 2 ^ st.c.unitRange.slotLog := by rw [L.unit_slotLog]; exact hb
  have hx' : InSpan st.c.unitRange (R.slot s + a) := by rw [he]; exact inSpan_of_covers (covers_slot hs₀ hb')
  have hR' := eq_of_inSpan L.disj hR L.unit_mem (inSpan_of_covers (covers_slot hs ha)) hx'
  subst hR'
  exact ⟨Kind.net.inj (hk.symm.trans L.unit_kind), slot_add_inj ha hb' he⟩


/-- A net slot's input port bit, in another net's range than the unit's, is none of the unit slots' columns. -/
theorem ne_unit_at {tags : Tags} (P : ParseFacts tags st.c) (C : CheckFacts st.c) {n : ℕ} (hn : n < st.c.nets.size)
    (hne : n ≠ st.c.unitNet) {g i : ℕ} {q : ℕ × ℕ} {tt : ℕ} (hg : g < st.c.g) (hi : i < st.c.perVu.getD n 0)
    (hq : q ∈ (portCols (st.c.nets.getD n default).2.inGroups).toList) (htt : tt < q.2) {s₀ b : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog) :
    st.c.slotCol n g i + q.1 + tt ≠ st.c.unitRange.slot s₀ + b := by
  intro he
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (C.one_range n hn)
  have hc := C.count _ hm _ hk
  have hs : g * st.c.perVu.getD n 0 + i < (st.c.rangeOf n).count := by rw [hc]; exact Rows.mul_add_lt hg hi
  have hq' := (nets_ok P n).1.2.2 q hq
  have hfit := (L.fit n).2
  have ha : q.1 + tt < 2 ^ (st.c.rangeOf n).slotLog := by rw [L.slotLog _ hm n hk]; omega
  exact hne (L.eq_unit hm hk hs ha hs₀ hb (by rw [← he]; unfold Circuit.slotCol; omega)).1

/-- A port bit's offset is inside the unit's input rows. -/
theorem port_lt {p t : ℕ} (hp : p < (portCols st.c.unit.inGroups).size)
    (ht : t < (portCols st.c.unit.inGroups)[p]!.2) :
    (portCols st.c.unit.inGroups)[p]!.1 + t < st.c.unit.inWords * Flock.WORD := by
  have := built_ports L.unit_built _ (getElem_mem_toList' hp)
  omega

theorem port_lt_slot {p t : ℕ} (hp : p < (portCols st.c.unit.inGroups).size)
    (ht : t < (portCols st.c.unit.inGroups)[p]!.2) :
    (portCols st.c.unit.inGroups)[p]!.1 + t < 2 ^ st.c.unitRange.slotLog := by
  rw [L.unit_slotLog]
  exact lt_of_lt_of_le (L.port_lt hp ht) (L.fit st.c.unitNet).2

/-- **The leaf pairs at a unit port bit**: a leaf port's bit gets its leaf's copy pair, any other port's none. -/
theorem leafPairs_count (C : CheckFacts st.c) (hnd : (portBits st.c.unit.inGroups).Nodup) {s₀ p t : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hp : p < (portCols st.c.unit.inGroups).size)
    (ht : t < (portCols st.c.unit.inGroups)[p]!.2) (j : ℕ) :
    (Flat.leafPairs st.c).count (st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t), j) =
      if p < (st.c.unit.inGroups.getD 0 default).bits.size then
        (if st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t) = j then 1 else 0) +
          if Flat.leafSrc st.c s₀ p t = j then 1 else 0
      else 0 := by
  have hupv : 0 < st.c.upv := by
    rcases Nat.eq_zero_or_pos st.c.upv with h0 | h0
    · rw [L.unit_count, h0, Nat.mul_zero] at hs₀; omega
    · exact h0
  have hli : ∀ k, (st.c.leavesIn.getD (k % st.c.upv) #[]).size = (st.c.unit.inGroups.getD 0 default).bits.size := by
    intro k
    have hmod := Nat.mod_lt k hupv
    have hlt : k % st.c.upv < st.c.leavesIn.size := by rw [C.leaves.1]; exact hmod
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some,
      ← getElem!_pos st.c.leavesIn _ hlt]
    exact C.leaves.2 _ hmod
  have hleaf : ∀ k jj tt, jj < (st.c.unit.inGroups.getD 0 default).bits.size → tt < 16 →
      Flat.leafCol st.c k jj tt = st.c.unitRange.slot k + ((portCols st.c.unit.inGroups)[jj]!.1 + tt) ∧
        jj < (portCols st.c.unit.inGroups).size ∧ tt < (portCols st.c.unit.inGroups)[jj]!.2 := by
    intro k jj tt hjj htt
    obtain ⟨hjs, hcol, hw⟩ := firstGroup_port C.unit_groups.1 hjj
    have h16 : (st.c.unit.inGroups.getD 0 default).bits[jj]! = 16 :=
      C.unit_groups.2 _ (getElem_mem_toList' (xs := (st.c.unit.inGroups.getD 0 default).bits) hjj)
    refine ⟨?_, hjs, by rw [hw, h16]; exact htt⟩
    unfold Flat.leafCol
    rw [hcol]
    omega
  have hx := L.port_lt_slot hp ht
  have hk0 : ∀ k < st.c.unitRange.count, k ≠ s₀ → ∀ jj tt, jj < (st.c.unit.inGroups.getD 0 default).bits.size →
      tt < 16 → Flat.leafCol st.c k jj tt ≠ st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t) := by
    intro k _ hne jj tt hjj htt he
    obtain ⟨hlc, hjs, htw⟩ := hleaf k jj tt hjj htt
    rw [hlc] at he
    exact hne (slot_add_inj (L.port_lt_slot hjs htw) hx he).1
  have hj0 : ∀ jj tt, jj < (st.c.unit.inGroups.getD 0 default).bits.size → tt < 16 → (jj, tt) ≠ (p, t) →
      Flat.leafCol st.c s₀ jj tt ≠ st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t) := by
    intro jj tt hjj htt hne he
    obtain ⟨hlc, hjs, htw⟩ := hleaf s₀ jj tt hjj htt
    rw [hlc] at he
    have h1 := (slot_add_inj (L.port_lt_slot hjs htw) hx he).2
    obtain ⟨rfl, rfl⟩ := portBits_inj hnd hjs hp htw ht h1
    exact hne rfl
  unfold Flat.leafPairs
  rw [count_flatMap_range _ _ hs₀ (fun k hk hne => count_flatMap_zero fun jj hjj => count_flatMap_zero fun tt htt =>
    count_pair_ne (hk0 k hk hne jj tt (by rw [← hli k]; exact List.mem_range.1 hjj) (List.mem_range.1 htt)))]
  rw [hli s₀]
  split
  · rename_i hpl
    have h16 : t < 16 := by
      have := (hleaf s₀ p 0 hpl (by norm_num)).2.2
      obtain ⟨-, -, hw⟩ := firstGroup_port C.unit_groups.1 hpl
      rw [hw, C.unit_groups.2 _ (getElem_mem_toList' (xs := (st.c.unit.inGroups.getD 0 default).bits) hpl)] at ht
      exact ht
    rw [count_flatMap_range _ _ hpl (fun jj hjj hne => count_flatMap_zero fun tt htt =>
      count_pair_ne (hj0 jj tt hjj (List.mem_range.1 htt) (by intro h; exact hne (Prod.mk.inj h).1))),
      count_flatMap_range _ _ h16 (fun tt htt hne => count_pair_ne (hj0 p tt hpl htt (by
        intro h; exact hne (Prod.mk.inj h).2)))]
    rw [(hleaf s₀ p t hpl h16).1, count_pair_eq]
  · rename_i hpl
    exact count_flatMap_zero fun jj hjj => count_flatMap_zero fun tt htt =>
      count_pair_ne (hj0 jj tt (List.mem_range.1 hjj) (List.mem_range.1 htt) (by
        intro h; exact hpl ((Prod.mk.inj h).1 ▸ List.mem_range.1 hjj)))


omit L in
theorem getD_portCols {gs : Array Group} {p : ℕ} (hp : p < (portCols gs).size) :
    (portCols gs).getD p default = (portCols gs)[p]! := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hp, Option.getD_some, getElem!_pos (portCols gs) p hp]

/-- The unit's range slots per template instance: its units per template instance. -/
theorem perVu_eq (C : CheckFacts st.c) {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) :
    st.c.perVu.getD st.c.unitNet 0 = st.c.upv := by
  refine perVu_unit C L.unit_mem L.unit_kind L.unit_count ?_
  rcases Nat.eq_zero_or_pos st.c.g with h0 | h0
  · rw [L.unit_count, h0, Nat.zero_mul] at hs₀; omega
  · exact h0

/-- **The cut pairs at a unit port bit**: the copy pair of each cut into its port, in its template instance. -/
theorem cutPairs_count (C : CheckFacts st.c) (hnd : (portBits st.c.unit.inGroups).Nodup) {s₀ p t : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hp : p < (portCols st.c.unit.inGroups).size)
    (ht : t < (portCols st.c.unit.inGroups)[p]!.2) (j : ℕ) :
    (Flat.cutPairs st.c).count (st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t), j) =
      ((st.c.leafCuts.toList.filter fun e => decide (e.1 = s₀ % st.c.upv ∧ e.2.1 = p)).flatMap fun e =>
        [(st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t),
          st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t)),
          (st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t),
            Flat.cutSrc st.c (s₀ / st.c.upv) e t)]).count
        (st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t), j) := by
  have hpv := L.perVu_eq C hs₀
  have hg₀ : s₀ / st.c.upv < st.c.g := by
    rw [L.unit_count] at hs₀
    exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hs₀)
  have hx := L.port_lt_slot hp ht
  unfold Flat.cutPairs
  refine count_vu_items _ _ _ (fun g e tt => Flat.cutCol st.c g e + tt) _ _ hg₀ (fun g hg e he tt htt => ?_)
    (fun e _ hF => by rw [hF.2]; exact ht)
  obtain ⟨he1, he2⟩ := C.cuts e he
  have hcol : Flat.cutCol st.c g e + tt =
      st.c.unitRange.slot (g * st.c.upv + e.1) + ((portCols st.c.unit.inGroups)[e.2.1]!.1 + tt) := by
    unfold Flat.cutCol Circuit.slotCol
    rw [hpv, Nat.add_assoc]
    rfl
  rw [hcol]
  constructor
  · intro heq
    obtain ⟨h1, h2⟩ := slot_add_inj (L.port_lt_slot he2 htt) hx heq
    obtain ⟨rfl, rfl⟩ := portBits_inj hnd he2 hp htt ht h2
    obtain ⟨rfl, h3⟩ := (vu_split he1).1 h1
    exact ⟨rfl, ⟨h3, rfl⟩, rfl⟩
  · rintro ⟨rfl, ⟨h1, rfl⟩, rfl⟩
    rw [(vu_split he1).2 ⟨rfl, h1⟩]

/-- **The wire pairs at a unit port bit**: the copy pair of each wire into its port, in its template instance. -/
theorem wirePairs_count {tags : Tags} (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hnd : (portBits st.c.unit.inGroups).Nodup) {s₀ p t : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hp : p < (portCols st.c.unit.inGroups).size)
    (ht : t < (portCols st.c.unit.inGroups)[p]!.2) (j : ℕ) :
    (Flat.wirePairs st.c).count (st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t), j) =
      ((st.c.wires.toList.filter fun w => decide (w.dst = (st.c.unitNet, s₀ % st.c.upv, p))).flatMap fun w =>
        [(st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t),
          st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t)),
          (st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t),
            Flat.wireSrc st.c (s₀ / st.c.upv) w + t)]).count
        (st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t), j) := by
  have hpv := L.perVu_eq C hs₀
  have hg₀ : s₀ / st.c.upv < st.c.g := by
    rw [L.unit_count] at hs₀
    exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hs₀)
  have hx := L.port_lt_slot hp ht
  unfold Flat.wirePairs
  refine count_vu_items _ _ _ (fun g w tt => Flat.wireDst st.c g w + tt) (fun g w tt => Flat.wireSrc st.c g w + tt) _
    hg₀ (fun g hg w hw tt htt => ?_) (fun w _ hF => ?_)
  · obtain ⟨hd1, hd2, hd3⟩ := C.wires w hw
    by_cases hdu : w.dst.1 = st.c.unitNet
    · obtain ⟨⟨dn, di, dp⟩, src⟩ := w
      simp only at hdu hd2 hd3 htt ⊢
      subst hdu
      rw [hpv] at hd2
      have hq : Flat.wireWidth st.c ⟨(st.c.unitNet, di, dp), src⟩ = (portCols st.c.unit.inGroups)[dp]!.2 := by
        unfold Flat.wireWidth
        exact congrArg Prod.snd (getD_portCols hd3)
      rw [hq] at htt
      have hcol : Flat.wireDst st.c g ⟨(st.c.unitNet, di, dp), src⟩ + tt =
          st.c.unitRange.slot (g * st.c.upv + di) + ((portCols st.c.unit.inGroups)[dp]!.1 + tt) := by
        unfold Flat.wireDst Circuit.slotCol
        simp only
        rw [hpv, getD_portCols hd3, Nat.add_assoc]
        rfl
      rw [hcol]
      constructor
      · intro heq
        obtain ⟨h1, h2⟩ := slot_add_inj (L.port_lt_slot hd3 htt) hx heq
        obtain ⟨rfl, rfl⟩ := portBits_inj hnd hd3 hp htt ht h2
        obtain ⟨rfl, h3⟩ := (vu_split hd2).1 h1
        exact ⟨rfl, by rw [h3], rfl⟩
      · rintro ⟨rfl, hF, rfl⟩
        simp only [Prod.mk.injEq, true_and] at hF
        obtain ⟨h1, rfl⟩ := hF
        rw [(vu_split hd2).2 ⟨rfl, h1⟩]
    · constructor
      · intro heq
        exact absurd heq (L.ne_unit_at P C hd1 hdu hg hd2 (getD_mem_toList hd3) htt hs₀
          (lt_of_lt_of_le hx (by rw [L.unit_slotLog])))
      · rintro ⟨-, hF, -⟩
        exact absurd (congrArg Prod.fst hF) hdu
  · unfold Flat.wireWidth
    rw [hF]
    show t < ((portCols st.c.unit.inGroups).getD p default).2
    rw [getD_portCols hp]
    exact ht


/-- **Δ at a unit copy's input port bit**: its one copy pair, `(x, x)` and `(x, flatSrc …)`. -/
theorem delta_port {tags : Tags} (P : ParseFacts tags st.c) (C : CheckFacts st.c) (hΔ : FlatDelta st.c st.pin st.da)
    (hsha : st.c.unitNet ≠ st.c.shaNet) (hhm : st.c.unitNet ≠ st.c.hmNet) (hnd : (portBits st.c.unit.inGroups).Nodup)
    {s₀ p t : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hp : p < (portCols st.c.unit.inGroups).size)
    (ht : t < (portCols st.c.unit.inGroups)[p]!.2) (j : ℕ) :
    st.da.toList.count (st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t), j) =
      (if st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t) = j then 1 else 0) +
        if Flat.flatSrc st.c s₀ p t = j then 1 else 0 := by
  obtain ⟨rows, hd, hrows⟩ := hΔ
  have hoff := L.port_lt hp ht
  have hb : (portCols st.c.unit.inGroups)[p]!.1 + t < 2 ^ st.c.unit.unitLog :=
    lt_of_lt_of_le hoff (L.fit st.c.unitNet).2
  have hrow0 : ∀ q ∈ rows, q.1 ≠ st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t) := by
    intro q hq heq
    have hws := rowSlots_spec st.c.ports
    have hall := rowSlots_all st.c.ports
    rcases hrows q hq with ⟨g, hg, pp, hpp, k, hk, i, hi, tt, htt, he⟩ | ⟨g, hg, pp, hpp, i, hi, tt, htt, he⟩
    · obtain ⟨h1, h2⟩ := hall _ (getElem_mem_toList' hpp) _ (getElem_mem_toList' hk)
      have hidx : 2 * ((HmRow.rowSlots st.c.ports).2[pp]!)[k]!.2 + i < 6 := by omega
      refine L.ne_unit_at P C P.shaNet (Ne.symm hsha) hg (by rw [C.perVu_sha]; exact h1)
        (getElem_mem_toList' (by rw [C.sha_in.1]; exact hidx)) ?_ hs₀ hb (he ▸ heq)
      rw [C.sha_in.2 _ hidx, sha_width _ _ hi]
      exact htt
    · obtain ⟨hil, hw⟩ := hm_width C.hm_in hi
      refine L.ne_unit_at P C P.hmNet (Ne.symm hhm) hg ?_ (getElem_mem_toList' hil)
        (by rw [hw]; exact htt) hs₀ hb (he ▸ heq)
      rw [C.perVu_hm, ← hws]
      exact lt_of_lt_of_le hpp (le_nextPow2 _)
  rw [hd]
  simp only [List.count_append]
  rw [List.count_eq_zero.2 (L.constPairs_not_below L.unit_mem L.unit_kind hs₀ (lt_of_lt_of_le hoff L.unit_in) j),
    List.count_eq_zero.2 (fun hm => hrow0 _ hm rfl), L.leafPairs_count C hnd hs₀ hp ht,
    L.cutPairs_count C hnd hs₀ hp ht, L.wirePairs_count P C hnd hs₀ hp ht]
  unfold Flat.flatSrc
  split
  · rename_i hpl
    have hc0 : (st.c.leafCuts.toList.filter fun e => decide (e.1 = s₀ % st.c.upv ∧ e.2.1 = p)) = [] :=
      List.filter_eq_nil_iff.2 fun e he => by
        have := C.cut_leaf e he
        simp only [decide_eq_true_eq, not_and]
        intro _ h; omega
    have hw0 : (st.c.wires.toList.filter fun w => decide (w.dst = (st.c.unitNet, s₀ % st.c.upv, p))) = [] :=
      List.filter_eq_nil_iff.2 fun w hw => by
        simp only [decide_eq_true_eq]
        intro hd'
        have := C.wire_leaf w hw (by rw [hd'])
        rw [hd'] at this
        simp only at this
        omega
    rw [hc0, hw0]
    simp
  · rename_i hpl
    have hpv := L.perVu_eq C hs₀
    have hupv : 0 < st.c.upv := by
      rcases Nat.eq_zero_or_pos st.c.upv with h0 | h0
      · rw [L.unit_count, h0, Nat.mul_zero] at hs₀; omega
      · exact h0
    have hone := C.wired st.c.unitNet P.unitNet hsha hhm (s₀ % st.c.upv) (by rw [hpv]; exact Nat.mod_lt _ hupv) p
      (by rw [ite_eq_left rfl]; omega) hp
    have hcf : (st.c.leafCuts.toList.filter fun e => decide ((st.c.unitNet, e.1, e.2.1) =
        (st.c.unitNet, s₀ % st.c.upv, p))) =
        st.c.leafCuts.toList.filter fun e => decide (e.1 = s₀ % st.c.upv ∧ e.2.1 = p) :=
      List.filter_congr fun e _ => by simp
    rw [hcf] at hone
    simp only [Nat.zero_add]
    rcases hcl : (st.c.leafCuts.toList.filter fun e => decide (e.1 = s₀ % st.c.upv ∧ e.2.1 = p)) with _ | ⟨e, rest⟩
    · rw [hcl, List.length_nil, Nat.add_zero] at hone
      obtain ⟨w, hw⟩ := List.length_eq_one_iff.1 hone
      rw [hcl, hw]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, count_pair_eq, List.count_nil, Nat.zero_add]
    · rw [hcl, List.length_cons] at hone
      have hw0 : (st.c.wires.toList.filter fun w => decide (w.dst = (st.c.unitNet, s₀ % st.c.upv, p))) = [] :=
        List.length_eq_zero_iff.1 (by omega)
      have hr0 : rest = [] := List.length_eq_zero_iff.1 (by omega)
      subst hr0
      rw [hcl, hw0]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, count_pair_eq, List.count_nil, Nat.add_zero]

/-- **The block rows at a unit copy's input port bit read its source**: the net's self row there, and Δ's one copy
pair. -/
theorem placed_port {tags : Tags} (P : ParseFacts tags st.c) (C : CheckFacts st.c) (hΔ : FlatDelta st.c st.pin st.da)
    (hsha : st.c.unitNet ≠ st.c.shaNet) (hhm : st.c.unitNet ≠ st.c.hmNet) (hnd : (portBits st.c.unit.inGroups).Nodup)
    {s₀ p t : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hp : p < (portCols st.c.unit.inGroups).size)
    (ht : t < (portCols st.c.unit.inGroups)[p]!.2)
    (hA : ∀ j', sparseEntry st.c.unit.a ((portCols st.c.unit.inGroups)[p]!.1 + t) j' =
      if j' = (portCols st.c.unit.inGroups)[p]!.1 + t then 1 else 0)
    (hB : ∀ j', sparseEntry st.c.unit.b ((portCols st.c.unit.inGroups)[p]!.1 + t) j' +
      lookupEntry st.c.unit.lookup ((portCols st.c.unit.inGroups)[p]!.1 + t) j' =
        if j' = (portCols st.c.unit.inGroups)[p]!.1 + t then 1 else 0)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.unitRange.slot s₀ + ((portCols st.c.unit.inGroups)[p]!.1 + t)) :
    placedA st r j = (if j.val = Flat.flatSrc st.c s₀ p t then 1 else 0) ∧
      placedB st r j = if j.val = Flat.flatSrc st.c s₀ p t then 1 else 0 := by
  have hoff := L.port_lt hp ht
  have hb : (portCols st.c.unit.inGroups)[p]!.1 + t < 2 ^ st.c.unit.unitLog :=
    lt_of_lt_of_le hoff (L.fit st.c.unitNet).2
  have hW : 0 < 2 ^ st.c.unit.unitLog := Nat.two_pow_pos _
  have hcnt := L.delta_port P C hΔ hsha hhm hnd hs₀ hp ht j.val
  have fin : ∀ (x src : ℕ), ((if j.val = x then (1 : ZMod 2) else 0) +
      (((if x = j.val then 1 else 0) + if src = j.val then 1 else 0 : ℕ) : ZMod 2)) = if j.val = src then 1 else 0 := by
    intro x src
    by_cases h1 : j.val = x
    · rw [ite_eq_left h1, ite_eq_left h1.symm]
      by_cases h2 : src = j.val
      · rw [ite_eq_left h2, ite_eq_left h2.symm]; decide
      · rw [ite_eq_right h2, ite_eq_right (Ne.symm h2)]; decide
    · rw [ite_eq_right h1, ite_eq_right (Ne.symm h1), zero_add, Nat.zero_add]
      by_cases h2 : src = j.val
      · rw [ite_eq_left h2, ite_eq_left h2.symm]; simp
      · rw [ite_eq_right h2, ite_eq_right (Ne.symm h2)]; simp
  constructor
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [hr, L.rangesEntry_unit hs₀ hb, ite_slot_eq (v := slotEntryA st (.net st.c.unitNet) _) hW hb hA,
      deltaEntry_eq_count, hcnt, ← L.slot_eq, ← Nat.add_assoc]
    rw [Nat.add_assoc]
    exact fin _ _
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [L.delta_b, hr, L.rangesEntry_unit hs₀ hb, ite_slot_eq (v := slotEntryB st (.net st.c.unitNet) _) hW hb hB,
      deltaEntry_eq_count, hcnt, ← L.slot_eq, ← Nat.add_assoc]
    rw [Nat.add_assoc]
    exact fin _ _


/-- **A unit slot's row past its useful rows is zero**, in `A` and `B`: the net's row there is empty, no slot constant is
there, and Δ's other pairs are at input bits. -/
theorem placed_zero {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hu : st.c.unit.useful < 2 ^ st.c.unit.unitLog)
    (huse : st.c.unit.useful = st.c.unit.constPos + 1)
    (hA : ∀ j', sparseEntry st.c.unit.a st.c.unit.useful j' = 0)
    (hB : ∀ j', sparseEntry st.c.unit.b st.c.unit.useful j' + lookupEntry st.c.unit.lookup st.c.unit.useful j' = 0)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.unitRange.slot s₀ + st.c.unit.useful) :
    placedA st r j = 0 ∧ placedB st r j = 0 := by
  have hd0 : deltaEntry st.da r.val j.val = 0 := by
    obtain ⟨rest, hsplit, hrest⟩ := L.delta
    rw [hr, deltaEntry_eq_count, hsplit, List.count_append]
    have h1 : (constPairs st.c st.pin).count (st.c.unitRange.slot s₀ + st.c.unit.useful, j.val) = 0 := by
      refine List.count_eq_zero.2 fun hm => ?_
      obtain ⟨r', hr', i, hki, s, hs', heq, -⟩ := constPairs_fst hm
      simp only at heq
      have hc := (L.fit i).1
      rw [← L.slotLog r' hr' i hki] at hc
      obtain ⟨rfl, -, h⟩ := L.eq_unit hr' hki hs' hc hs₀ hu heq.symm
      exact absurd h (by show st.c.unit.constPos ≠ st.c.unit.useful; omega)
    have h2 : rest.count (st.c.unitRange.slot s₀ + st.c.unit.useful, j.val) = 0 := by
      refine List.count_eq_zero.2 fun hm => ?_
      obtain ⟨n, s, off, hm', hk, hs', hoff, heq⟩ := hrest _ hm
      simp only at heq
      have hoffW : off < 2 ^ (st.c.rangeOf n).slotLog := by
        rw [L.slotLog _ hm' n hk]; exact lt_of_lt_of_le hoff (L.fit n).2
      obtain ⟨rfl, -, rfl⟩ := L.eq_unit hm' hk hs' hoffW hs₀ hu heq.symm
      have := L.unit_in
      exact absurd hoff (by show ¬ st.c.unit.useful < st.c.unit.inWords * Flock.WORD; omega)
    rw [h1, h2]
    rfl
  have hW : 0 < 2 ^ st.c.unit.unitLog := Nat.two_pow_pos _
  constructor
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = 0
    rw [hd0, add_zero, hr, L.rangesEntry_unit hs₀ hu]
    split
    · exact hA _
    · rfl
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = 0
    rw [L.delta_b, hd0, add_zero, hr, L.rangesEntry_unit hs₀ hu]
    split
    · exact hB _
    · rfl

end Layout

/-! ## A flat unit's net -/

/-- A flat unit's net holds its rows as they are: its row `c` is the unit's, empty past the unit's rows. -/
theorem netOfD_row {d : D} {name : String} {n : Net} (h : Typed.netOfD d name = .ok n) (hlog : n.unitLog ≤ 32) (c : ℕ) :
    n.a.row c = (if c < d.size then (d.rows.getD c ([], [])).1.toArray else #[]) ∧
      n.b.row c = if c < d.size then (d.rows.getD c ([], [])).2.toArray else #[] := by
  obtain ⟨-, ha, hb, -⟩ := Typed.netOfD_ok h
  have hw : 2 ^ n.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hlog
  rw [ofRows_row ha hw, ofRows_row hb hw]
  unfold Typed.rowsA Typed.rowsB
  by_cases hc : c < d.size
  · simp [Array.getD_eq_getD_getElem?, hc]
  · simp [Array.getD_eq_getD_getElem?, hc]

/-- The column of a unit copy's input `off` that Δ copies into it: `flatSrc` at the port bit `off` is. -/
def flatSrcAt (c : Circuit) (s₀ off : ℕ) : ℕ :=
  Flat.flatSrc c s₀ (portOf (portCols c.unit.inGroups) off)
    (off - (portCols c.unit.inGroups)[portOf (portCols c.unit.inGroups) off]!.1)

/-! ## The accepted flat-class statement -/

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **What a flat class's accepted statement holds beside its `Layout`**: its circuit's parse and check, Δ pair by pair,
and a unit that is neither of the rows' circuits (it is `parseTyped`'s first net, named `unit`). -/
theorem setupH_flatFacts (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net)) :
    ParseFacts tags st.c ∧ CheckFacts st.c ∧ FlatDelta st.c st.pin st.da ∧ st.c.unitNet ≠ st.c.shaNet ∧
      st.c.unitNet ≠ st.c.hmNet := by
  obtain ⟨hp, -, hda, -, -, -⟩ := setupH_spec_typed ht h
  obtain ⟨k', r, hr', hcase⟩ := parseTyped_spec hp
  rw [hr] at hr'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hr')
  rcases hcase with ⟨t, c', hrt, -, -⟩ | ⟨net', c', hrn, hpc, hc⟩
  · cases hrt
  · cases hrn
    obtain ⟨name, hnet, -⟩ := Typed.read_flat hr
    obtain ⟨P, ⟨more, hmore⟩, hunit, hsn, hhn⟩ := parse_facts_pre (fun x hx => by
      simp at hx
      subst hx
      exact netOfD_built hnet) hpc
    have P' : ParseFacts tags st.c := by
      rw [hc]
      exact ⟨P.nets, P.portWords, P.v1Segs, P.rowWires, P.unitNet, P.shaNet, P.hmNet, P.kinds, P.check,
        (by rw [← P.rowNets]; exact HmNets.check_congr rfl rfl rfl),
        (by rw [← P.recNet]; exact RecOpen.check_congr rfl rfl),
        (by rw [← P.recResiduals]; exact RecResiduals.check_congr rfl rfl),
        (by rw [← P.recSparse]; exact RecSparse.check_congr rfl rfl),
        (by rw [← P.recOpening]; exact RecOpening.check_congr rfl rfl), P.typed⟩
    obtain ⟨ls, hcheck⟩ := P'.check
    have h0 : c'.nets[0]? = some ("unit", net) := by
      rw [← Array.getElem?_toList, hmore]
      simp
    have hu0 : c'.unitNet = 0 := by
      obtain ⟨hlt, -, hmin⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
      by_contra hne
      have h1 := hmin 0 (Nat.pos_of_ne_zero hne)
      rw [Array.getElem?_eq_getElem (by omega)] at h0
      simp [Option.some.inj h0] at h1
    have hname0 : (c'.nets.getD 0 default).1 = "unit" := by rw [getD_of_getElem? h0]
    refine ⟨P', check_facts P'.typed hcheck, by rw [hda]; exact delta_flat st.c st.pin P'.typed, ?_, ?_⟩
    · rw [hc]
      show c'.unitNet ≠ c'.shaNet
      intro he
      rw [← he, hu0, hname0] at hsn
      exact absurd hsn (by unfold HmRow.SHA_NET; decide)
    · rw [hc]
      show c'.unitNet ≠ c'.hmNet
      intro he
      rw [← he, hu0, hname0] at hhn
      exact absurd hhn (by unfold HmRow.HM_NET; decide)

/-- **A flat class's input copies, at each unit slot**: input `w`'s block rows read the column Δ copies into it
(`flatSrcAt`). The unit's inputs are its net's input port bits in order (`Typed.flatNet`, #434). -/
theorem setupH_flatInput (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net)) {s₀ w : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hw : w < (k.done.getD k.unit default).inCols.size) (r j : Fin (2 ^ st.c.kLog))
    (hr' : r.val = st.c.unitRange.slot s₀ + (k.done.getD k.unit default).inCols[w]) :
    placedA st r j = (if j.val = flatSrcAt st.c s₀ (k.done.getD k.unit default).inCols[w] then 1 else 0) ∧
      placedB st r j = if j.val = flatSrcAt st.c s₀ (k.done.getD k.unit default).inCols[w] then 1 else 0 := by
  obtain ⟨-, hunit, L⟩ := setupH_flatLayout ht h hr
  obtain ⟨P, C, hΔ, hsha, hhm⟩ := setupH_flatFacts ht h hr
  obtain ⟨-, name, hnet, -, hin⟩ := read_flat' hr
  replace hin : (k.done.getD k.unit default).inCols.toList = portBits net.inGroups := hin
  have hlog : net.unitLog ≤ 32 := hunit ▸ L.unitLog_le hs₀
  obtain ⟨hnd0, -, htake, -, -⟩ := order_sound k.hd 0
  have hndin : (k.done.getD k.unit default).inCols.toList.Nodup := by
    rw [← htake]; exact hnd0.sublist (List.take_sublist _ _)
  generalize hu : k.done.getD k.unit default = u at hin hw hr' hndin htake hnd0 ⊢
  have hnd : (portBits st.c.unit.inGroups).Nodup := by rw [hunit, ← hin]; exact hndin
  -- input `w` is a port bit
  have hmem : u.inCols[w] ∈ portBits net.inGroups := by rw [← hin]; exact Array.getElem_mem_toList hw
  obtain ⟨q, hq, hqr⟩ := List.mem_flatMap.1 hmem
  obtain ⟨p, hp, rfl⟩ := List.getElem_of_mem hq
  rw [List.mem_range'_1] at hqr
  have hp' : p < (portCols st.c.unit.inGroups).size := by rw [hunit]; simpa using hp
  have hq' : (portCols st.c.unit.inGroups)[p]! = (portCols net.inGroups).toList[p] := by
    rw [hunit, getElem!_pos _ _ (by simpa using hp), Array.getElem_toList]
  obtain ⟨t, ht', hoff⟩ : ∃ t, t < (portCols st.c.unit.inGroups)[p]!.2 ∧
      u.inCols[w] = (portCols st.c.unit.inGroups)[p]!.1 + t :=
    ⟨u.inCols[w] - (portCols st.c.unit.inGroups)[p]!.1, by rw [hq']; omega, by rw [hq']; omega⟩
  have hsrc : flatSrcAt st.c s₀ u.inCols[w] = Flat.flatSrc st.c s₀ p t := by
    unfold flatSrcAt
    rw [hoff, portOf_eq hnd hp' ht']
    congr 1
    omega
  rw [hsrc]
  -- the net's self row there
  have hcr : u.rows[u.inCols[w]]? = some ([u.inCols[w]], [u.inCols[w]]) := by
    have := (unit_inputs k.hd u.inCols[w] (by rw [hu]; exact Array.getElem_mem hw)).1
    rwa [hu] at this
  have hcs : u.inCols[w] < u.size := by
    have hmo : u.inCols[w] ∈ order k.done u := by
      rw [← List.take_append_drop u.inCols.size (order k.done u), htake]
      exact List.mem_append_left _ (Array.getElem_mem_toList hw)
    obtain ⟨hflat, -⟩ := read_flat' hr
    rw [hu] at hflat
    simp only [Typed.isFlat, Bool.and_eq_true] at hflat
    rcases (hu ▸ order_cols k.hd) u.inCols[w] hmo with h1 | ⟨pp, hpp, -⟩
    · exact h1
    · rw [Array.isEmpty_iff.1 hflat.1.1] at hpp; simp at hpp
  rw [hu] at hnet
  obtain ⟨hra, hrb⟩ := netOfD_row hnet hlog u.inCols[w]
  have hgd : u.rows.getD u.inCols[w] ([], []) = ([u.inCols[w]], [u.inCols[w]]) := getD_of_getElem? hcr
  have hself : ∀ j', (([u.inCols[w]].toArray : Array ℕ).toList.count j' : ZMod 2) = if j' = u.inCols[w] then 1 else 0 := by
    intro j'
    by_cases hj : j' = u.inCols[w]
    · subst hj; simp
    · simp [hj, Ne.symm hj]
  have hlk : st.c.unit.lookup = none := L.unit_lookup
  refine L.placed_port P C hΔ hsha hhm hnd hs₀ hp' ht' (fun j' => ?_) (fun j' => ?_) r j (by rw [hr', hoff])
  · rw [← hoff, sparseEntry_built_a L.unit_built, hunit, hra, ite_eq_left hcs, hgd]
    exact hself j'
  · rw [← hoff, sparseEntry_built_b L.unit_built, hlk, hunit, hrb, ite_eq_left hcs, hgd]
    show _ + 0 = _
    rw [add_zero]
    exact hself j'

/-- **A flat class's forced-zero rows**: each unit slot's row past the unit's useful rows reads nothing. -/
theorem setupH_flatZero (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net)) {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (r j : Fin (2 ^ st.c.kLog)) (hr' : r.val = st.c.unitRange.slot s₀ + st.c.unit.useful) :
    placedA st r j = 0 ∧ placedB st r j = 0 := by
  obtain ⟨-, hunit, L⟩ := setupH_flatLayout ht h hr
  obtain ⟨-, C, -⟩ := setupH_flatFacts ht h hr
  obtain ⟨-, name, hnet, -⟩ := read_flat' hr
  have hlog : net.unitLog ≤ 32 := hunit ▸ L.unitLog_le hs₀
  obtain ⟨-, -, -, huse, hcst, -⟩ := Typed.netOfD_ok hnet
  have hcu := (ofRows_groups (show Net.ofRows _ _ _ _ _ _ _ = .ok net from hnet)).1
  obtain ⟨hra, hrb⟩ := netOfD_row hnet hlog net.useful
  have hpast : ¬ net.useful < (k.done.getD k.unit default).size := by omega
  rw [ite_eq_right hpast] at hra hrb
  have hlk : st.c.unit.lookup = none := L.unit_lookup
  refine L.placed_zero hs₀ C.unit_zero (by rw [hunit]; omega) (fun j' => ?_) (fun j' => ?_) r j hr'
  · rw [sparseEntry_built_a L.unit_built, hunit, hra]; rfl
  · rw [sparseEntry_built_b L.unit_built, hlk, hunit, hrb]; rfl


/-! ## The flat class's table class -/

section tableClass

variable (ht : tags.typed = true) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
  {k : Typed.Class} {net : Net} (hr : Typed.read tags cf tables = .ok (k, .inl net))
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))

/-- Unit slot `g`'s forced-zero row, past the unit's useful rows. -/
def flatZeroPos (g : Fin st.c.unitRange.count) : Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).kLog) :=
  ⟨st.c.unitRange.slot g + st.c.unit.useful,
    (setupH_flatLayout ht h hr).2.2.unit_slot_lt g.isLt (setupH_flatFacts ht h hr).2.1.unit_zero⟩

/-- Unit slot `g`'s input `i`'s copy position: the column Δ copies into it (`flatSrcAt`), or the slot's forced-zero row
when that column lies outside the block, where the input's rows are zero. -/
def flatCopyPos (g : Fin st.c.unitRange.count) (i : ℕ) :
    Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).kLog) :=
  if hlt : flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) < 2 ^ st.c.kLog then ⟨_, hlt⟩
  else flatZeroPos ht h hr regs g

/-- The rows at a unit slot's forced-zero row carry 0. -/
theorem flatZero_block (g : Fin st.c.unitRange.count) {z : Model.Witness ((setupH_flatLayout ht h hr).2.2.model regs).m}
    (hz : ((setupH_flatLayout ht h hr).2.2.model regs).Satisfies z) (o : Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).nbl)) :
    ((setupH_flatLayout ht h hr).2.2.model regs).block z o (flatZeroPos ht h hr regs g) = 0 := by
  have hZ : ZeroRow ((setupH_flatLayout ht h hr).2.2.model regs) (flatZeroPos ht h hr regs g) :=
    ⟨fun j => (setupH_flatZero ht h hr g.isLt _ j rfl).1, fun j => (setupH_flatZero ht h hr g.isLt _ j rfl).2⟩
  exact hZ.block_eq hz o

/-- **A flat class's input copies**: at unit slot `g`, each input of the unit's composed rows carries what its copy
position carries, in every block of a satisfying witness. -/
theorem setupH_flatCopy (g : Fin st.c.unitRange.count)
    (c : Fin (Rows.compose (ofBlock (Typed.wordsIn k.held) k.done (k.done.getD k.unit default)) (ofBlock_ordered k.hd)).w)
    (hc : c.val < (Rows.compose (ofBlock (Typed.wordsIn k.held) k.done (k.done.getD k.unit default))
      (ofBlock_ordered k.hd)).nIn)
    {z : Model.Witness ((setupH_flatLayout ht h hr).2.2.model regs).m}
    (hz : ((setupH_flatLayout ht h hr).2.2.model regs).Satisfies z)
    (o : Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).nbl)) :
    ((setupH_flatLayout ht h hr).2.2.model regs).block z o
        (composePlace _ _ (ofBlock_ordered k.hd) (fun x => st.c.unitRange.slot g + x)
          (setupH_flatRealizes ht h hr regs g.isLt).lt c) =
      ((setupH_flatLayout ht h hr).2.2.model regs).block z o (flatCopyPos ht h hr regs g c.val) := by
  obtain ⟨-, -, htake, -, -⟩ := order_sound k.hd 0
  have hw : c.val < (k.done.getD k.unit default).inCols.size := hc
  have hlen : (k.done.getD k.unit default).inCols.size ≤ (order k.done (k.done.getD k.unit default)).length := by
    have := congrArg List.length htake
    simp only [List.length_take, Array.length_toList] at this
    omega
  have hcol : (composePlace _ _ (ofBlock_ordered k.hd) (fun x => st.c.unitRange.slot g + x)
      (setupH_flatRealizes ht h hr regs g.isLt).lt c).val =
      st.c.unitRange.slot g + (k.done.getD k.unit default).inCols[c.val] := by
    unfold composePlace
    rw [dite_eq_left (show c.val < (order k.done (k.done.getD k.unit default)).length by omega)]
    show st.c.unitRange.slot g + (order k.done (k.done.getD k.unit default)).getD c.val 0 = _
    rw [order_getD_inCols htake hw, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hw, Option.getD_some]
  have hrow := fun j => setupH_flatInput ht h hr g.isLt hw _ j hcol
  have hgd : (k.done.getD k.unit default).inCols.getD c.val 0 = (k.done.getD k.unit default).inCols[c.val] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hw, Option.getD_some]
  unfold flatCopyPos
  rw [hgd]
  split
  · rename_i hlt
    have hC : CopyRow ((setupH_flatLayout ht h hr).2.2.model regs) _ ⟨_, hlt⟩ :=
      ⟨fun j => by rw [← ite_fin_val j hlt]; exact (hrow j).1, fun j => by rw [← ite_fin_val j hlt]; exact (hrow j).2⟩
    exact hC.block_eq hz o
  · rename_i hlt
    have hZ : ZeroRow ((setupH_flatLayout ht h hr).2.2.model regs)
        (composePlace _ _ (ofBlock_ordered k.hd) (fun x => st.c.unitRange.slot g + x)
          (setupH_flatRealizes ht h hr regs g.isLt).lt c) := by
      refine ⟨fun j => ?_, fun j => ?_⟩
      · rw [show ((setupH_flatLayout ht h hr).2.2.model regs).A₀ = placedA st from rfl, (hrow j).1,
          ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
      · rw [show ((setupH_flatLayout ht h hr).2.2.model regs).B₀ = placedB st from rfl, (hrow j).2,
          ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
    rw [hZ.block_eq hz o, flatZero_block ht h hr regs g hz o]

/-- **A flat class's table class** (the soundness lane's `TableClass`), from the statement `Stmt.setupH` accepted: its
unit slots, the unit's composed rows placed at each (`setupH_flatPlacement`), each input's copy position
(`setupH_flatCopy`) and the slots' forced-zero rows (`setupH_flatZero`). -/
def setupH_flatTableClass : TableClass ((setupH_flatLayout ht h hr).2.2.model regs) where
  types := Typed.typeIn k.types
  ls := k.ls
  info := Typed.infoIn k.held
  words := Typed.wordsIn k.held
  unit := k.unit
  done := k.done
  hd := k.hd
  slots := st.c.unitRange.count
  col g := composePlace _ _ (ofBlock_ordered k.hd) (fun x => st.c.unitRange.slot g + x)
    (setupH_flatRealizes ht h hr regs g.isLt).lt
  placed g := setupH_flatPlacement ht h hr regs g.isLt
  copyPos := flatCopyPos ht h hr regs
  copy g c hc _ hz o := setupH_flatCopy ht h hr regs g c hc hz o
  zeros := Finset.univ.image (flatZeroPos ht h hr regs)
  zero q hq _ hz o := by
    obtain ⟨g, -, rfl⟩ := Finset.mem_image.1 hq
    exact flatZero_block ht h hr regs g hz o

end tableClass

end FlockSoundness
