module

public import Proofs.Flock.SpotFlock.WiresMove
public import Spot.SecondWires

/-!
# The three conditions, as the verifier's package decides them, move every wired source bit

`SecondWiresMove` (`WiresTie`, proved in `WiresMove`) takes three conditions round 2's check of the wires does not
decide: `rowsUnder`, `namesOnce` and `wiresWide`. `Spot.SecondWires` decides each at the size of what it reads. This
file proves that each decider returning true gives its condition as `SecondWiresMove` takes it, and so that the three
together, with round 2's check of the wires, at a registration whose two-stage design evaluates, give `wiresMove`.

* `rowsUnder_of_decided`, `namesOnce_of_decided`: the two that are decided as stated.
* `mem_of_marks`: a bit set in the bitmap (`SecondWires.marks`) is one of the list's, so a bitmap with every bit set
  (`SecondWires.covers`) has every bit under its size in the list (`mem_writes_of_covers`).
* `writesBit_of_mem_writes`: a bit `SecondWires.writes` lists is one the circuit file writes on the output row as
  `writesBit` states it: the leaf's bit has a gate in a key program of one unit (`Bits.outGate`), since its column is
  a computed row of the unit net, and its commitment position is that bit of the row (`Bits.outRowBit`), since the
  output row found among `SecondWires.outRows` is the one `Bits.outPos` finds.
* `wiresWide_of_decided`: so the decider gives `wiresWide`.
* `wiresMove_of_decided`: the three together (`SecondWires.decided`) at the run's parts, the templates' circuit files
  and what the verifier holds of the run, with round 2's check of the wires, give `wiresMove` (`secondWiresMove`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling

/-! ## The two decided as stated -/

/-- **The verifier's `rowsUnder` is the condition.** -/
theorem rowsUnder_of_decided {own : Spot.SecondStage.Own} (hd : Spot.SecondWires.rowsUnder own = true) :
    rowsUnder own = true := hd

/-- **The verifier's `namesOnce` is the condition.** -/
theorem namesOnce_of_decided {ps : Array Spot.SecondStage.Part} (hd : Spot.SecondWires.namesOnce ps = true) :
    namesOnce ps = true := hd

/-! ## The bitmap -/

/-- Setting bits keeps the bitmap's size. -/
theorem size_marks_fold (l : List ℕ) (a : Array Bool) :
    (l.foldl (fun a i => a.setIfInBounds i true) a).size = a.size := by
  induction l generalizing a with
  | nil => rfl
  | cons x xs ih => rw [List.foldl_cons, ih, Array.size_setIfInBounds]

/-- **A bit set after marking a list's was set before or is one of the list's.** -/
theorem mem_of_marks_fold (l : List ℕ) (a : Array Bool) {i : ℕ}
    (h : (l.foldl (fun a i => a.setIfInBounds i true) a)[i]? = some true) : a[i]? = some true ∨ i ∈ l := by
  induction l generalizing a with
  | nil => exact Or.inl h
  | cons x xs ih =>
    rw [List.foldl_cons] at h
    rcases ih _ h with h1 | h1
    · rw [Array.getElem?_setIfInBounds] at h1
      by_cases hx : x = i
      · exact Or.inr (hx ▸ List.mem_cons_self)
      · rw [ite_eq_right hx] at h1
        exact Or.inl h1
    · exact Or.inr (List.mem_cons_of_mem _ h1)

/-- **A bit set in the bitmap is one of the list's.** -/
theorem mem_of_marks {n : ℕ} {l : List ℕ} {i : ℕ} (h : (Spot.SecondWires.marks n l)[i]? = some true) : i ∈ l := by
  rcases mem_of_marks_fold l _ h with h1 | h1
  · rw [Array.getElem?_replicate] at h1
    split at h1
    · cases h1
    · cases h1
  · exact h1

/-- **Where the circuit file covers the row's first `bits` bits, each is one it lists.** -/
theorem mem_writes_of_covers {c : _root_.Flock.Circuit} {port : String} {bits : ℕ}
    (h : Spot.SecondWires.covers c port bits = true) {β : ℕ} (hβ : β < bits) :
    β ∈ Spot.SecondWires.writes c port := by
  have h : (decide (bits ≤ (Spot.SecondWires.writes c port).length) &&
      (Spot.SecondWires.marks bits (Spot.SecondWires.writes c port)).all id) = true := h
  have h := (Bool.and_eq_true_iff.1 h).2
  have hsz : (Spot.SecondWires.marks bits (Spot.SecondWires.writes c port)).size = bits := by
    unfold Spot.SecondWires.marks
    rw [size_marks_fold, Array.size_replicate]
  have hβ' : β < (Spot.SecondWires.marks bits (Spot.SecondWires.writes c port)).size := by rw [hsz]; exact hβ
  have hall := Array.all_eq_true_iff_forall_mem.1 h _ (Array.getElem_mem hβ')
  have hget : (Spot.SecondWires.marks bits (Spot.SecondWires.writes c port))[β]? = some true := by
    rw [Array.getElem?_eq_getElem hβ']
    exact congrArg some hall
  exact mem_of_marks hget

/-! ## A listed bit is a written bit -/

/-- **A bit the verifier's list holds is one the circuit file writes on the output row**, as `writesBit` states it. -/
theorem writesBit_of_mem_writes {c : _root_.Flock.Circuit} {port : String} {β : ℕ}
    (h : β ∈ Spot.SecondWires.writes c port) : writesBit c port β = true := by
  unfold Spot.SecondWires.writes at h
  obtain ⟨k, hk, h⟩ := List.mem_flatMap.1 h
  obtain ⟨j, hj, h⟩ := List.mem_flatMap.1 h
  have hk' : k < c.upv := Nat.lt_of_lt_of_le (List.mem_range.1 hk) (Nat.min_le_left _ _)
  have hj' : j < (c.leavesOut.getD k #[]).size := List.mem_range.1 hj
  generalize hfind : (Spot.SecondWires.outRows c).find? (fun r =>
    decide (r.2.1 ≤ (c.leavesOut.getD k #[]).getD j 0) &&
      decide ((c.leavesOut.getD k #[]).getD j 0 < r.2.1 + r.2.2)) = found at h
  cases found with
  | none => exact absurd h List.not_mem_nil
  | some r =>
    have h : β ∈ (if (r.1 == port) = true then (List.range 16).filterMap fun t =>
        if min (c.unit.inWords * _root_.Flock.WORD) c.unit.constPos ≤
            (c.unit.outGroups.getD 0 default).col * _root_.Flock.WORD + (16 * j + t) ∧
          (c.unit.outGroups.getD 0 default).col * _root_.Flock.WORD + (16 * j + t) < c.unit.constPos
        then some (16 * ((c.leavesOut.getD k #[]).getD j 0 - r.2.1) + t) else none
      else []) := h
    by_cases hport : (r.1 == port) = true
    · rw [ite_eq_left hport] at h
      obtain ⟨t, ht, h⟩ := List.mem_filterMap.1 h
      have ht' := List.mem_range.1 ht
      by_cases hcol : min (c.unit.inWords * _root_.Flock.WORD) c.unit.constPos ≤
            (c.unit.outGroups.getD 0 default).col * _root_.Flock.WORD + (16 * j + t) ∧
          (c.unit.outGroups.getD 0 default).col * _root_.Flock.WORD + (16 * j + t) < c.unit.constPos
      · rw [ite_eq_left hcol] at h
        have hβ : β = 16 * ((c.leavesOut.getD k #[]).getD j 0 - r.2.1) + t := (Option.some.inj h).symm
        have hmod : β % 16 = t := by omega
        have hx : (c.leavesOut.getD k #[])[j]? = some ((c.leavesOut.getD k #[]).getD j 0) := by
          generalize c.leavesOut.getD k #[] = ls at hj'
          rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj', Option.getD_some]
        -- the output row found among the verifier's is the one `Bits.outPos` finds
        have hrows : Spot.SecondWires.outRows c = (List.range c.outRowPorts.size).map fun o =>
            ((c.ports.getD (_root_.Flock.HmOut.nIn c + o) default).name, (_root_.Flock.HmOut.outGroup c o).2.2.1,
              (_root_.Flock.HmOut.outGroup c o).2.2.2) := rfl
        rw [hrows, List.find?_map] at hfind
        obtain ⟨o, ho, hfo⟩ := Option.map_eq_some_iff.1 hfind
        subst hfo
        have hname : (c.ports.getD (_root_.Flock.HmOut.nIn c + o) default).name = port := eq_of_beq hport
        have ho' : ((List.range c.outRowPorts.size).find? fun o =>
            decide ((_root_.Flock.HmOut.outGroup c o).2.2.1 ≤ (c.leavesOut.getD k #[]).getD j 0) &&
              decide ((c.leavesOut.getD k #[]).getD j 0 <
                (_root_.Flock.HmOut.outGroup c o).2.2.1 + (_root_.Flock.HmOut.outGroup c o).2.2.2)) = some o := ho
        have hpos : Spot.Bits.outRowBit c 0 k j t = some (Spot.Bits.RowPos.mk port 0 β) := by
          unfold Spot.Bits.outRowBit
          rw [hx, Option.bind_some]
          show Option.map (fun o => Spot.Bits.RowPos.mk (c.ports.getD (_root_.Flock.HmOut.nIn c + o) default).name 0
            (16 * ((c.leavesOut.getD k #[]).getD j 0 - (_root_.Flock.HmOut.outGroup c o).2.2.1) + t))
            ((List.range c.outRowPorts.size).find? fun o =>
              decide ((_root_.Flock.HmOut.outGroup c o).2.2.1 ≤ (c.leavesOut.getD k #[]).getD j 0) &&
                decide ((c.leavesOut.getD k #[]).getD j 0 <
                  (_root_.Flock.HmOut.outGroup c o).2.2.1 + (_root_.Flock.HmOut.outGroup c o).2.2.2)) = _
          rw [ho', Option.map_some, hname, hβ]
        -- the bit's column is a computed row of the unit net
        have hgate : (Spot.Bits.outGate c 1 0 k j t).isSome = true := by
          have hnin : (Spot.Bits.Rows.ofNet c.unit).nIn = min (c.unit.inWords * _root_.Flock.WORD) c.unit.constPos :=
            rfl
          have hncomp : (Spot.Bits.Rows.ofNet c.unit).nComp =
              c.unit.constPos - min (c.unit.inWords * _root_.Flock.WORD) c.unit.constPos := rfl
          have hoc : Spot.Bits.outCol c j t =
              (c.unit.outGroups.getD 0 default).col * _root_.Flock.WORD + (16 * j + t) := rfl
          have hle := Nat.min_le_right (c.unit.inWords * _root_.Flock.WORD) c.unit.constPos
          have hc : (decide (0 < 1) && decide (k < c.upv) && decide (j < (c.leavesOut.getD k #[]).size) &&
              decide (t < 16) && decide ((Spot.Bits.Rows.ofNet c.unit).nIn ≤ Spot.Bits.outCol c j t) &&
              decide (Spot.Bits.outCol c j t <
                (Spot.Bits.Rows.ofNet c.unit).nIn + (Spot.Bits.Rows.ofNet c.unit).nComp)) = true := by
            simp only [Bool.and_eq_true, decide_eq_true_eq]
            rw [hnin, hncomp, hoc]
            exact ⟨⟨⟨⟨⟨Nat.zero_lt_one, hk'⟩, hj'⟩, ht'⟩, hcol.1⟩, by omega⟩
          have hsome : Spot.Bits.outGate c 1 0 k j t = some (Spot.Bits.rowGate (Spot.Bits.Rows.ofNet c.unit)
              (2 + 1 * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn)
              ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp 0 k (Spot.Bits.outCol c j t)) := by
            unfold Spot.Bits.outGate
            exact ite_eq_left hc
          rw [hsome]
          rfl
        unfold writesBit
        refine List.any_eq_true.2 ⟨k, List.mem_range.2 hk', List.any_eq_true.2 ⟨j, hj, ?_⟩⟩
        rw [hmod, hgate, hpos]
        simp
      · rw [ite_eq_right hcol] at h
        cases h
    · rw [ite_eq_right hport] at h
      exact absurd h List.not_mem_nil

/-! ## The widths, and the three together -/

/-- **The verifier's `wiresWide` gives the condition.** -/
theorem wiresWide_of_decided {cs : Array (String × _root_.Flock.Circuit)} {own : Spot.SecondStage.Own}
    (hd : Spot.SecondWires.wiresWide cs own = true) : wiresWide cs own = true := by
  unfold Spot.SecondWires.wiresWide at hd
  unfold wiresWide
  rw [Array.all_eq_true_iff_forall_mem] at hd ⊢
  intro w hw
  have h1 := hd w hw
  cases hr : cs.find? (·.1 == w.template) with
  | none =>
    rw [hr] at h1
    simp at h1
  | some r =>
    cases hx : cs.find? (·.1 == w.writer) with
    | none =>
      rw [hr, hx] at h1
      simp at h1
    | some x =>
      rw [hr, hx] at h1
      have h2 : (r.2.ports.all fun p => p.name != w.port || Spot.SecondWires.covers x.2 w.wport p.bits) = true := h1
      show (r.2.ports.all fun p => p.name != w.port ||
        (List.range p.bits).all fun β => writesBit x.2 w.wport β) = true
      rw [Array.all_eq_true_iff_forall_mem] at h2 ⊢
      intro p hp
      have h3 := h2 p hp
      rw [Bool.or_eq_true] at h3 ⊢
      rcases h3 with h3 | h3
      · exact Or.inl h3
      · refine Or.inr (List.all_eq_true.2 fun β hβ => ?_)
        exact writesBit_of_mem_writes (mem_writes_of_covers h3 (List.mem_range.1 hβ))

/-- **The three conditions as the verifier's package decides them, with round 2's check of the wires, move every
wired source bit**: at a registration whose two-stage design evaluates, where round 2's check of the wires passes at
what the verifier holds of the run and `SecondWires.decided` holds at the run's parts, the templates' circuit files
and what the verifier holds of the run, every source bit at a port a wire names and no link names has a gate to go to
(`wiresMove`). -/
theorem wiresMove_of_decided (h : Held) (s : AuditStatement) (st : Roots) (rec draw : Lean.Json)
    (c₀ : FlockDeploy.Coin) (htwo : (two h s).isSome) (hchk : wiresCheckedAt h st rec draw c₀ = true)
    (hd : Spot.SecondWires.decided (partsAt h (h.reg.sample (src c₀)).size) (circuitsOf h)
      (ownOf h st rec draw c₀) = true) : wiresMove h = true := by
  unfold Spot.SecondWires.decided at hd
  rw [Bool.and_eq_true, Bool.and_eq_true] at hd
  exact secondWiresMove h s st rec draw c₀ htwo hchk (rowsUnder_of_decided hd.1.1) (namesOnce_of_decided hd.1.2)
    (wiresWide_of_decided hd.2)

end SpotFlock
