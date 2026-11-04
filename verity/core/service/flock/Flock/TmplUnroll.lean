/-
A templated unit's logical unroll succeeds, checked on its description: `Tmpl.unroll` (its rows, then the checks
`Net.ofRows` runs on a flat unit) refuses nothing that passes `Tmpl.wellFormed`, `Tmpl.apart`, `Tmpl.ordered`,
`Tmpl.inputsT` and `Tmpl.groupsOk` (`unroll_ok`). `groupsOk` is what `Net.ofRows` checks of the port groups and the size,
none of it a row; each row check `Net.ofRows` runs follows from the others: the rows exist (`rows_ok`), the input rows are
self rows or empty (`rows_inputRows`), the constant row is the last write, the order holds (`rows_order`, so
`Net.checkOrder` passes, `checkOrder_ok`), the input port bits have their self rows (`rows_inputsOk`), and every column a
row reads is at most the constant's, inside the slot (`Sparse.ofRows` passes, `sparse_ofRows_ok`). So a reader that checks
the description knows the flat net exists without building it.
-/
import Flock.TmplApart

namespace Flock.Tmpl

/-- **What `Net.ofRows` checks of a templated unit's port groups and size**, on the description: each group an aligned
power of two of words holding its ports, an output group, the input groups inside the input rows and those before the
constant, and each output group inside the unit's slot (D11). -/
def Tmpl.groupsOk (t : Tmpl) : Bool :=
  let cols := (t.defs.getD t.root default).cols
  let inWords := (t.inGroups.back?.map fun g => g.col + g.words).getD 0
  !(t.inGroups ++ t.outGroups).any (fun g => !(isPow2 g.words && g.col % g.words == 0)) && !t.outGroups.isEmpty &&
    !(t.inGroups.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD || g.col + g.words > inWords)
      || t.outGroups.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD)) &&
    decide (inWords * WORD ≤ t.constPos) &&
    !t.outGroups.any (fun g => g.col + g.words > 2 ^ max 7 (log2ceil cols) / WORD)

end Flock.Tmpl

namespace Flock.TmplUnroll

open Flock.Tmpl Flock.TmplOrder Flock.TmplRows Flock.TmplApart

/-! ## Loops that never throw -/

theorem bind_yield {ε α β : Type} {x : Except ε α} {g : α → β} (hx : ∃ a, x = .ok a) :
    ∃ b, (x >>= fun a => pure (ForInStep.yield (g a))) = .ok (.yield b) := by
  obtain ⟨a, rfl⟩ := hx
  exact ⟨g a, rfl⟩

theorem bind_unit {ε α : Type} {x : Except ε α} (hx : ∃ a, x = .ok a) : (x >>= fun _ => pure ()) = .ok () := by
  obtain ⟨a, rfl⟩ := hx
  rfl

theorem bind_yieldUnit {ε α : Type} {x : Except ε α} (hx : ∃ a, x = .ok a) :
    ∃ b, (x >>= fun _ => pure (ForInStep.yield PUnit.unit)) = .ok (.yield b) := by
  obtain ⟨a, rfl⟩ := hx
  exact ⟨_, rfl⟩

theorem bind_ok_of {ε α β : Type} {x : Except ε α} {f : α → Except ε β} (hx : ∃ a, x = .ok a)
    (hf : ∀ a, ∃ b, f a = .ok b) : ∃ b, (x >>= f) = .ok b := by
  obtain ⟨a, rfl⟩ := hx
  exact hf a

/-- A loop each of whose steps continues, from any state, ends. -/
theorem forIn_ok {α β ε : Type} (M : α → Prop) (f : α → β → Except ε (ForInStep β))
    (h : ∀ a b, M a → ∃ b', f a b = .ok (.yield b')) : ∀ (l : List α), (∀ a ∈ l, M a) → ∀ init, ∃ r, forIn l init f = .ok r
  | [], _, init => ⟨init, rfl⟩
  | a :: l, hM, init => by
    obtain ⟨b, hb⟩ := h a init (hM a (List.mem_cons.2 (Or.inl rfl)))
    obtain ⟨r, hr⟩ := forIn_ok M f h l (fun x hx => hM x (List.mem_cons.2 (Or.inr hx))) b
    exact ⟨r, by rw [List.forIn_cons, hb]; exact hr⟩

theorem forIn_okA {α β ε : Type} (M : α → Prop) (f : α → β → Except ε (ForInStep β))
    (h : ∀ a b, M a → ∃ b', f a b = .ok (.yield b')) (xs : Array α) (hM : ∀ a ∈ xs, M a) (init : β) :
    ∃ r, forIn xs init f = .ok r := by
  rw [← Array.forIn_toList]
  exact forIn_ok M f h xs.toList (fun a ha => hM a (Array.mem_toList_iff.1 ha)) init

/-- **`Sparse.ofRows` passes rows whose columns are inside the width.** -/
theorem sparse_ofRows_ok {rows : Array (Array Nat)} {width : Nat} (h : ∀ r ∈ rows, ∀ c ∈ r, c < width) :
    ∃ m, Sparse.ofRows rows width = .ok m := by
  unfold Sparse.ofRows
  refine bind_ok_of (forIn_okA (fun r => ∀ c ∈ r, c < width) _ (fun r b hr => ?_) rows h _) fun _ => ⟨_, rfl⟩
  dsimp only
  refine bind_yield (forIn_okA (fun c => c < width) _ (fun c b hc => ?_) r hr _)
  exact ⟨_, by simp only [ge_iff_le, Nat.not_le.2 hc, ↓reduceIte]; rfl⟩

/-- **`Net.checkOrder` passes rows in its order.** -/
theorem checkOrder_ok {name : String} {inCols constPos : Nat} {ra rb : Array (Array Nat)}
    (h : ∀ r, inCols ≤ r → r < constPos → ∀ c ∈ (ra.getD r #[]).toList ++ (rb.getD r #[]).toList,
      c = constPos ∨ (c < r ∧ ¬ (c < inCols ∧ (ra.getD c #[]).isEmpty = true ∧ (rb.getD c #[]).isEmpty = true))) :
    Net.checkOrder name inCols constPos ra rb = .ok () := by
  unfold Net.checkOrder
  simp only [Std.Legacy.Range.forIn_eq_forIn_range']
  refine bind_unit (forIn_ok (fun r => inCols ≤ r ∧ r < constPos) _ (fun r b hr => ?_) _ (fun r hr => ?_) PUnit.unit)
  · refine bind_yieldUnit (forIn_okA (fun c => c ∈ (ra.getD r #[] ++ rb.getD r #[]).toList) _ (fun c b hc => ?_) _
      (fun c hc => Array.mem_toList_iff.2 hc) _)
    rcases h r hr.1 hr.2 c (by simpa using hc) with e | ⟨h1, h2⟩
    · exact ⟨_, by simp [e]; rfl⟩
    · have hne : (c == constPos) = false := by
        rw [beq_eq_false_iff_ne]; intro e; subst e; omega
      have hpad : (decide (c < inCols) && (ra.getD c #[]).isEmpty && (rb.getD c #[]).isEmpty) = false := by
        cases h3 : decide (c < inCols) <;> cases h4 : (ra.getD c #[]).isEmpty <;> cases h5 : (rb.getD c #[]).isEmpty <;>
          simp_all
      exact ⟨_, by simp only [hne, hpad, ge_iff_le, Nat.not_le.2 h1]; rfl⟩
  · rw [List.mem_range'] at hr
    obtain ⟨i, hi, rfl⟩ := hr
    simp [Std.Legacy.Range.size] at hi
    omega

/-! ## The unit -/

theorem log2ceil_le (n : Nat) : n ≤ 2 ^ log2ceil n := by
  unfold log2ceil
  split
  · simp only [Nat.pow_zero]; omega
  · have := Nat.lt_log2_self (n := n - 1); omega

theorem write_setSize {acc acc' : Acc} {c : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc') :
    acc'.set.size = acc.set.size := by
  obtain ⟨-, rfl⟩ := write_eq h
  simp [Array.set!_eq_setIfInBounds]

theorem place_setSize {defs : Array Def} {k : Nat} (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → acc'.set.size = acc.set.size := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have s1 := foldlM_inv (fun u => u.set.size = acc.set.size) (fun _ _ u u' hu hw => (write_setSize hw).trans hu) rfl h1
      exact foldlM_inv (fun u => u.set.size = acc.set.size) (fun c _ u u' hu hw =>
        foldlM_inv (fun u => u.set.size = acc.set.size) (fun _ _ w w' hw0 hp => (ih hp).trans hw0) hu hw) s1 h2

/-- The unrolled rows: one per column, the constant's `[const]·[const]`. -/
theorem rows_shape {t : Tmpl} {acc : Acc} (hr : t.rows = .ok acc) :
    acc.ra.size = (t.defs.getD t.root default).cols ∧ acc.rb.size = (t.defs.getD t.root default).cols ∧
      acc.ra.getD t.constPos #[] = #[t.constPos] ∧ acc.rb.getD t.constPos #[] = #[t.constPos] := by
  unfold Tmpl.rows at hr
  obtain ⟨acc1, h1, h2⟩ := bind_ok hr
  have hs := place_sized _ h1 (sized_empty _)
  have hz := place_setSize _ h1
  have hself := write_self hs h2
  obtain ⟨-, rfl⟩ := write_eq h2
  refine ⟨?_, ?_, hself.2.1, hself.2.2⟩ <;>
  simp only [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds, hs.1, hs.2, hz, Acc.empty, Array.size_replicate]

theorem wf_cols {t : Tmpl} (hwf : t.wellFormed = true) : t.constPos + 1 = (t.defs.getD t.root default).cols := by
  have hcols : 0 < (t.defs.getD t.root default).cols := by
    unfold Tmpl.wellFormed at hwf
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hwf
    exact hwf.1.2
  unfold Tmpl.constPos; omega

theorem groupsOk_inCols {t : Tmpl} (hg : t.groupsOk = true) : t.inCols ≤ t.constPos := by
  unfold Tmpl.groupsOk at hg
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hg
  unfold Tmpl.inCols; exact hg.1.2

/-- **Every column an unrolled row reads is at most the constant's.** -/
theorem rows_reads {t : Tmpl} {acc : Acc} (hwf : t.wellFormed = true) (ho : t.ordered = true) (hi : t.inputsT = true)
    (hg : t.groupsOk = true) (hr : t.rows = .ok acc) (i : Nat) :
    (∀ c ∈ acc.ra.getD i #[], c ≤ t.constPos) ∧ ∀ c ∈ acc.rb.getD i #[], c ≤ t.constPos := by
  have hin := rows_inputRows hi hr
  have hord := rows_order ho hi hr
  obtain ⟨hsize, hsizeb, hka, hkb⟩ := rows_shape hr
  have hk := wf_cols hwf
  have hic := groupsOk_inCols hg
  by_cases h0 : i < (t.defs.getD t.root default).cols
  · by_cases h1 : i < t.inCols
    · rcases hin i h1 with ⟨ea, eb⟩ | ⟨ea, eb⟩
      · rw [ea, eb]
        exact ⟨fun c e => by simp at e; omega, fun c e => by simp at e; omega⟩
      · rw [ea, eb]; exact ⟨fun c e => by simp at e, fun c e => by simp at e⟩
    · by_cases h2 : i < t.constPos
      · refine ⟨fun c hc => ?_, fun c hc => ?_⟩
        · rcases hord i (by omega) h2 c (List.mem_append.2 (Or.inl (Array.mem_toList_iff.2 hc))) with e | ⟨e, -⟩ <;> omega
        · rcases hord i (by omega) h2 c (List.mem_append.2 (Or.inr (Array.mem_toList_iff.2 hc))) with e | ⟨e, -⟩ <;> omega
      · have e : i = t.constPos := by omega
        subst e
        rw [hka, hkb]
        exact ⟨fun c e => by simp at e; omega, fun c e => by simp at e; omega⟩
  · have ea : acc.ra.getD i #[] = #[] := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl
    have eb : acc.rb.getD i #[] = #[] := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl
    rw [ea, eb]; exact ⟨fun c e => by simp at e, fun c e => by simp at e⟩

/-- **The unrolled rows pass every check `Net.ofRows` runs.** -/
theorem ofRows_of_rows {t : Tmpl} {acc : Acc} (hwf : t.wellFormed = true) (ho : t.ordered = true)
    (hi : t.inputsT = true) (hg : t.groupsOk = true) (hr : t.rows = .ok acc) :
    ∃ n, Net.ofRows t.name (t.defs.getD t.root default).cols t.constPos t.inGroups t.outGroups acc.ra acc.rb = .ok n := by
  have hin := rows_inputRows hi hr
  have hord := rows_order ho hi hr
  have hok := rows_inputsOk hi hr
  obtain ⟨hsize, hsizeb, hka, hkb⟩ := rows_shape hr
  have hk := wf_cols hwf
  have hreads := rows_reads hwf ho hi hg hr
  unfold Tmpl.groupsOk at hg
  simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq, Bool.or_eq_false_iff] at hg
  obtain ⟨⟨⟨⟨hal, hne⟩, hin1, hout1⟩, hic⟩, hd11⟩ := hg
  have hwidth : ∀ (rows : Array (Array Nat)), (∀ i, ∀ c ∈ rows.getD i #[], c ≤ t.constPos) →
      ∀ r ∈ rows, ∀ c ∈ r, c < 2 ^ max 7 (log2ceil (t.defs.getD t.root default).cols) := by
    intro rows hrows r hr c hc
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hr
    have := hrows i c (by rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi]; exact hc)
    have h1 := log2ceil_le (t.defs.getD t.root default).cols
    have h2 : 2 ^ log2ceil (t.defs.getD t.root default).cols ≤ 2 ^ max 7 (log2ceil (t.defs.getD t.root default).cols) :=
      Nat.pow_le_pow_right (by omega) (Nat.le_max_right _ _)
    omega
  obtain ⟨sa, hsa⟩ := sparse_ofRows_ok (hwidth acc.ra fun i => (hreads i).1)
  obtain ⟨sb, hsb⟩ := sparse_ofRows_ok (hwidth acc.rb fun i => (hreads i).2)
  have hrows : (List.range (((t.inGroups.back?.map fun g => g.col + g.words).getD 0) * WORD)).find? (fun i => !((acc.ra.getD i #[] == #[i] && acc.rb.getD i #[] == #[i]) ||
      ((acc.ra.getD i #[]).isEmpty && (acc.rb.getD i #[]).isEmpty))) = none := by
    rw [List.find?_eq_none]
    intro i hi'
    rcases hin i (List.mem_range.1 hi') with ⟨ea, eb⟩ | ⟨ea, eb⟩ <;> simp [ea, eb]
  have hout : t.outGroups.find? (fun g => decide (g.col + g.words > 2 ^ max 7 (log2ceil (t.defs.getD t.root default).cols) / WORD)) = none := by
    rw [Array.find?_eq_none]
    intro g hg'
    rw [Array.any_eq_false'] at hd11
    simpa using hd11 g hg'
  have hco := checkOrder_ok (name := "flock-ir-unit/v2") hord
  have hkb' : (t.constPos + 1 != (t.defs.getD t.root default).cols) = false := by simp [hk]
  unfold Tmpl.inCols at hco
  unfold Net.ofRows
  simp only [bind, Except.bind, pure, Except.pure, hal, hne, hkb', hin1, hout1, hrows, hka, hkb, hco, hok, hsa, hsb,
    hout, Bool.false_eq_true, ↓reduceIte, bne_self_eq_false, Bool.or_self, Nat.not_lt.2 hic, Bool.not_true]
  exact ⟨_, rfl⟩

/-- **A template that passes the description checks unrolls**: `Tmpl.unroll`, the flat net with every check
`Net.ofRows` runs, succeeds. -/
theorem unroll_ok {t : Tmpl} (hwf : t.wellFormed = true) (ha : t.apart = true) (ho : t.ordered = true)
    (hi : t.inputsT = true) (hg : t.groupsOk = true) : ∃ n, unroll t = .ok n := by
  obtain ⟨acc, hr⟩ := rows_ok hwf ha
  obtain ⟨n, hn⟩ := ofRows_of_rows hwf ho hi hg hr
  exact ⟨n, by simp only [unroll, hr, bind, Except.bind]; exact hn⟩

end Flock.TmplUnroll
