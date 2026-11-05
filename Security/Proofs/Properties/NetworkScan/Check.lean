import Properties.NetworkScan.Flattens
import Flock.Sigma

/-!
# What Bits' check of a circuit file gives

`SampledProofs.Bits.check` lays a circuit file out as a template of a Program, or says why it doesn't. Where it finds
nothing wrong (`facts`), the file has one template instance a VU (`Facts.upv`) and one output leaf map an instance
(`Facts.leaves`), each of as many words as the return of the Definition `find?` finds at the template's key
(`Facts.ret`, `SampledProofs.Bits.retWords`), and every returned output bit sits on a computed row of the unit net
(`Facts.outRows`), as `SampledProofs.Bits.outGate` reads the rows (`Rows.ofNet`, whose split of the columns is the
check's: `ofNet_nIn`, `ofNet_nComp`). Its unit's input columns are its input words (`Facts.inWords`), and its input
ports (`Flock.portCols`) are each at least a word wide (`Facts.widths`), in column order with none overlapping the next
(`Facts.ordered`, `PortsOrdered`; so each ends before every later one starts, `ordered_le`), and as many as the
Definition's parameters' words, whose bits `SampledProofs.Bits.paramsWhy` passes (`Facts.ports`).

A port's width is its group's bit (`portCols_snd`, `getD_snd`): the first group's port `p` is its bit `p`
(`portCols_first`). A type's bits (`SampledProofs.Bits.bitsOf`) are its leaves' widths summed, `Flock.Sigma.bits`
(`bits_bitsOf`): an array's its length times its element's, a value's its width (`bits_array`, `bits_value`). A pair's
return words are its two fields' (`retWords_pair`).
-/

open SampledProofs.Bits

namespace Properties.NetworkScan.Check

/-! ## A type's bits -/

theorem foldl_add_eq : ∀ (l : List ℕ) (a : ℕ), l.foldl (· + ·) a = a + l.sum
  | [], a => by simp
  | x :: l, a => by rw [List.foldl_cons, foldl_add_eq l, List.sum_cons]; omega

theorem bits_bitsOf (t : Flock.Program.Ty) : bitsOf t = Flock.Sigma.bits t := by
  unfold bitsOf
  rw [← Array.foldl_toList, foldl_add_eq, Flock.Sigma.widths_sum, Nat.zero_add]

theorem bits_value (w : ℕ) : Flock.Sigma.bits (.value w) = w := by
  simp [Flock.Sigma.bits]

theorem bits_array (n : ℕ) (e : Flock.Program.Ty) : Flock.Sigma.bits (.array n e) = n * Flock.Sigma.bits e := by
  simp [Flock.Sigma.bits]

theorem retWords_pair (a b : Flock.Program.Ty) : retWords (.tuple #[a, b]) = wordsOf a + wordsOf b := rfl

/-! ## The input ports -/

theorem inner_snd : ∀ (l : List ℕ) (acc : Array (ℕ × ℕ)) (s : ℕ),
    (l.foldl (fun b a => (b.1.push (b.2, a), b.2 + a)) (acc, s)).1.toList.map (·.2) = acc.toList.map (·.2) ++ l
  | [], acc, s => by simp
  | a :: l, acc, s => by rw [List.foldl_cons, inner_snd l]; simp

theorem outer_snd : ∀ (gs : List Flock.Group) (acc : Array (ℕ × ℕ)),
    (gs.foldl (fun b g => (g.bits.toList.foldl (fun b a => (b.1.push (b.2, a), b.2 + a)) (b, g.col * Flock.WORD)).1)
      acc).toList.map (·.2) = acc.toList.map (·.2) ++ gs.flatMap fun g => g.bits.toList
  | [], acc => by simp
  | g :: gs, acc => by
    rw [List.foldl_cons, outer_snd gs, inner_snd, List.flatMap_cons, List.append_assoc]

/-- The ports' widths are the groups' bits in order. -/
theorem portCols_snd (gs : Array Flock.Group) :
    (Flock.portCols gs).toList.map (·.2) = gs.toList.flatMap fun g => g.bits.toList := by
  unfold Flock.portCols
  simp only [← Array.forIn_toList, List.forIn_pure_yield_eq_foldl, pure_bind, Id.run_pure]
  rw [outer_snd]
  simp

theorem getD_snd {xs : Array (ℕ × ℕ)} {ys : List ℕ} (h : xs.toList.map (·.2) = ys) (p : ℕ) :
    (xs.getD p (0, 0)).2 = ys.getD p 0 := by
  subst h
  rw [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList]
  cases xs[p]? <;> rfl

/-- The first group's port `p` is its bit `p`. -/
theorem portCols_first {gs : Array Flock.Group} {p : ℕ} (hp : p < (gs.getD 0 default).bits.size) :
    ((Flock.portCols gs).getD p (0, 0)).2 = (gs.getD 0 default).bits.getD p 0 := by
  rw [getD_snd (portCols_snd gs) p]
  rcases hgs : gs.toList with _ | ⟨g, rest⟩
  · have hd : gs.getD 0 default = default := by
      rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hgs]; try rfl
    rw [hd] at hp
    exact absurd hp (Nat.not_lt_zero p)
  · have hg : gs.getD 0 default = g := by
      rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hgs]; try rfl
    rw [hg] at hp ⊢
    have hp' : p < g.bits.toList.length := by rw [Array.length_toList]; exact hp
    rw [List.flatMap_cons, List.getD_eq_getElem?_getD, List.getElem?_append_left hp',
      Array.getD_eq_getD_getElem? (xs := g.bits), ← Array.getElem?_toList]

/-- The unit's input ports in column order, none overlapping the next. -/
def PortsOrdered (c : Flock.Circuit) : Prop :=
  ∀ p, p + 1 < (Flock.portCols c.unit.inGroups).size →
    ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2 ≤
      ((Flock.portCols c.unit.inGroups).getD (p + 1) (0, 0)).1

/-- Of ports in column order, each ends at or before every later one's start. -/
theorem ordered_le {c : Flock.Circuit} (h : PortsOrdered c) {p : ℕ} :
    ∀ q, p < q → q < (Flock.portCols c.unit.inGroups).size →
      ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2 ≤
        ((Flock.portCols c.unit.inGroups).getD q (0, 0)).1
  | 0, hpq, _ => absurd hpq (Nat.not_lt_zero _)
  | q + 1, hpq, hq => by
    have h1 := h q hq
    by_cases hpq' : p = q
    · subst hpq'; exact h1
    · have h2 := ordered_le h (p := p) q (by omega) (by omega)
      have h3 := Nat.le_add_right ((Flock.portCols c.unit.inGroups).getD q (0, 0)).1
        ((Flock.portCols c.unit.inGroups).getD q (0, 0)).2
      omega

/-! ## The check's facts -/

theorem ofNet_nIn (net : Flock.Net) : (Rows.ofNet net).nIn = min (net.inWords * Flock.WORD) net.constPos := rfl

theorem ofNet_nComp (net : Flock.Net) :
    (Rows.ofNet net).nComp = net.constPos - min (net.inWords * Flock.WORD) net.constPos := rfl

theorem orElse_eq_none {α : Type} {x : Option α} {y : Unit → Option α} (h : HOrElse.hOrElse x y = none) :
    x = none ∧ y () = none := by
  cases x with
  | none => exact ⟨rfl, h⟩
  | some a => cases h

theorem ite_eq_none {α : Type} {b : Bool} {x : α} (h : (if b = true then some x else none) = none) : b = false := by
  cases b
  · rfl
  · simp at h

theorem ite_none {α : Type} {p : Prop} {inst : Decidable p} {x : α} {r : Option α}
    (h : @ite _ p inst (some x) r = none) : ¬ p ∧ r = none := by
  split at h
  · cases h
  · rename_i hp
    exact ⟨hp, h⟩

/-- Where each step keeps `Q` or meets an element with `P`, a fold ending in `Q` started in `Q` or met one. -/
theorem fold_ne {α β : Type} {f : β → α → β} {Q : β → Prop} {P : α → Prop}
    (hf : ∀ m e, Q (f m e) → Q m ∨ P e) : ∀ (l : List α) (m : β), Q (l.foldl f m) → Q m ∨ ∃ e ∈ l, P e
  | [], m, h => Or.inl h
  | e :: l, m, h => by
    rw [List.foldl_cons] at h
    rcases fold_ne hf l _ h with h1 | ⟨e', he', hp⟩
    · rcases hf m e h1 with h2 | h2
      · exact Or.inl h2
      · exact Or.inr ⟨e, List.mem_cons.2 (Or.inl rfl), h2⟩
    · exact Or.inr ⟨e', List.mem_cons_of_mem _ he', hp⟩

/-- **What Bits' check gives the proofs** of the circuit file `c` laid out as template `tmpl` of the Program `P`: one
template instance a VU, one output leaf map an instance, each of the words of the return of the Definition `find?`
finds at `tmpl`, and every returned output bit on a computed row of the unit net; the unit's input columns its input
words, and its input ports each at least a word wide, in column order, and as many as that Definition's parameters'
words, whose bits `paramsWhy` passes. -/
structure Facts (c : Flock.Circuit) (P : Flock.Program.Prog) (tmpl : String) : Prop where
  upv : c.upv = 1
  leaves : c.leavesOut.size = c.upv
  ret : ∀ d, P.defs.find? (·.key == tmpl) = some d → ∀ x ∈ c.leavesOut, x.size = retWords d.ret
  outRows : ∀ k < c.upv, ∀ j < (c.leavesOut.getD k #[]).size, ∀ t < 16,
    (Rows.ofNet c.unit).nIn ≤ outCol c j t ∧ outCol c j t < (Rows.ofNet c.unit).nIn + (Rows.ofNet c.unit).nComp
  inWords : (Rows.ofNet c.unit).nIn = c.unit.inWords * Flock.WORD
  widths : ∀ q < (Flock.portCols c.unit.inGroups).size, 16 ≤ ((Flock.portCols c.unit.inGroups).getD q (0, 0)).2
  ordered : PortsOrdered c
  ports : ∀ d, P.defs.find? (·.key == tmpl) = some d →
    (Flock.portCols c.unit.inGroups).size = (d.params.toList.map (wordsOf ·.2)).sum ∧ paramsWhy c d = none

/-- **Bits' check, passed, gives its facts.** A port of the first group is one word wide; a later port is wired
exactly once, by a leaf cut, which the check takes only at a port at least a word wide. -/
theorem facts {c : Flock.Circuit} {P : Flock.Program.Prog} {tmpl : String} (h : check c P tmpl = none) :
    Facts c P tmpl := by
  unfold check at h
  extract_lets nIn nComp gi go outs wired cut pc unwired at h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨h16, h⟩ := ite_none h
  obtain ⟨hleaves, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨hcuts, h⟩ := ite_none h
  obtain ⟨hord, h⟩ := ite_none h
  generalize hu : unwired = uw at h
  rcases uw with _ | ⟨k, p⟩
  rotate_left
  · exact absurd h (Option.some_ne_none _)
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨hnIn, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨hrows, h⟩ := ite_none h
  obtain ⟨hdef, hone⟩ := orElse_eq_none h
  have hupv1 : c.upv = 1 := bne_eq_false_iff_eq.1 (ite_eq_none hone)
  refine ⟨hupv1, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · apply Classical.byContradiction
    intro hne
    apply hleaves
    rw [Bool.or_eq_true, Bool.or_eq_true]
    exact Or.inl (Or.inr (bne_iff_ne.2 hne))
  · intro d hd x hx
    rw [hd] at hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨hsz, -⟩ := ite_none hdef
    exact Classical.byContradiction fun hne => hsz (Array.any_eq_true'.2 ⟨x, hx, bne_iff_ne.2 hne⟩)
  · intro k hk j hj t ht
    rw [ofNet_nIn, ofNet_nComp]
    have h3 := List.any_eq_false.1 (Bool.eq_false_iff.2 hrows) k (List.mem_range.2 hk)
    have h4 := List.any_eq_false.1 (Bool.eq_false_iff.2 h3) j (List.mem_range.2 hj)
    have h5 := List.any_eq_false.1 (Bool.eq_false_iff.2 h4) t (List.mem_range.2 ht)
    simpa only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, decide_eq_true_eq] using h5
  · exact bne_eq_false_iff_eq.1 (Bool.eq_false_iff.2 hnIn)
  · intro q hq
    by_cases hq' : q < cut
    · have hlt : q < (c.unit.inGroups.getD 0 default).bits.size := hq'
      have h16' : (c.unit.inGroups.getD 0 default).bits[q] = 16 := by
        apply Classical.byContradiction
        intro hne
        apply h16
        rw [Bool.or_eq_true]
        exact Or.inl (Array.any_eq_true.2 ⟨q, hlt, bne_iff_ne.2 hne⟩)
      have hgd : (c.unit.inGroups.getD 0 default).bits.getD q 0 = (c.unit.inGroups.getD 0 default).bits[q] := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
      have hpw : ((Flock.portCols c.unit.inGroups).getD q (0, 0)).2 = 16 := by
        rw [portCols_first hlt, hgd, h16']
      omega
    · have hk0 : 0 < c.upv := by omega
      have hq'' : q < pc.size := hq
      have hmem : ((0 : ℕ), q) ∈ (List.range c.upv).flatMap fun k =>
          (List.range (pc.size - cut)).map fun i => (k, cut + i) := by
        rw [List.mem_flatMap]
        refine ⟨0, List.mem_range.2 hk0, ?_⟩
        rw [List.mem_map]
        exact ⟨q - cut, List.mem_range.2 (by omega), Prod.ext rfl (show cut + (q - cut) = q by omega)⟩
      have hu' := List.find?_eq_none.1 hu (0, q) hmem
      have h1 : wired.getD (0, q) 0 = 1 := Classical.byContradiction fun hne => hu' (bne_iff_ne.2 hne)
      have hne : wired.getD (0, q) 0 ≠ 0 := by omega
      have hne2 := hne
      simp only [wired] at hne2
      rw [← Array.foldl_toList] at hne2
      rcases fold_ne (Q := fun (m : Std.HashMap (ℕ × ℕ) ℕ) => m.getD (0, q) 0 ≠ 0)
          (P := fun (e : ℕ × ℕ × ℕ) => (e.1, e.2.1) = (0, q))
          (by
            intro m e hme
            obtain ⟨k', p', w'⟩ := e
            try dsimp only at hme ⊢
            rw [Std.HashMap.getD_insert] at hme
            split at hme
            · rename_i hkp
              exact Or.inr (beq_iff_eq.1 hkp)
            · exact Or.inl hme) _ _ hne2 with h0 | ⟨e, he, hpe⟩
      · simp at h0
      · obtain ⟨k', p', w'⟩ := e
        simp only [Prod.mk.injEq] at hpe
        obtain ⟨hk0', hpq⟩ := hpe
        rw [hk0', hpq] at he
        apply Classical.byContradiction
        intro hlt
        have hlt' : (pc.getD q (0, 0)).2 < 16 := Nat.not_le.1 hlt
        apply hcuts
        refine Array.any_eq_true'.2 ⟨(0, q, w'), Array.mem_def.2 he, ?_⟩
        try dsimp only
        simp only [Bool.or_eq_true, decide_eq_true_eq]
        exact Or.inl (Or.inr hlt')
  · intro p hp
    have hp' : p + 1 < pc.size := hp
    have h1 := List.any_eq_false.1 (Bool.eq_false_iff.2 hord) p (List.mem_range.2 (by omega))
    have h2 : (pc.getD p (0, 0)).1 + (pc.getD p (0, 0)).2 ≤ (pc.getD (p + 1) (0, 0)).1 := by
      simpa only [decide_eq_true_eq, Nat.not_lt] using h1
    exact h2
  · intro d hd
    rw [hd] at hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨hpc, hdef⟩ := ite_none hdef
    exact ⟨bne_eq_false_iff_eq.1 (Bool.eq_false_iff.2 hpc), hdef⟩

end Properties.NetworkScan.Check
