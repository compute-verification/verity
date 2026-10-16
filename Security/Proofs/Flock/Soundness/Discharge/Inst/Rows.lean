import Proofs.Flock.Soundness.Discharge.Aliased.Keys

/-!
# A template instance's rows over every net, joined by Δ's wires

`Placed.circuitRows` stacks the unit net's rows `units_per_vu` times, so it is a template instance's rows only when no
wire reaches the unit. `instRows c` is the whole template instance: every slot of every net but the rows' circuits
(`sha512x3`, `hm96`), that is the unit's copies, the stage nets and the lookup slots. Column `off` of net `n`'s slot `i` is
`code n i off`, and its row is the block's there (`row`):
- at a computed row (past the net's input rows, before its constant), the net's local `A` and `B` rows, `B` with a lookup
  slot's folded table side (`entB`);
- at the constant, a copy of `one`;
- at an input bit a wire fills (`feed`), the net's local row there with Δ's two pairs: `(x, x)` (`addSelf`) and
  `(x, src)`, the wire's source bit;
- at any other column, no row.

The logical inputs (`inCols`) are the unit's stacked inputs that no wire fills, in stacked order: its leaf and cut bits
and its input padding rows. `order` appends, pass by pass, every column whose row reads only columns already listed. So
the rows are `Ordered` (`logical_ordered`) whatever the wires are: a column on a cycle of wires, or one that reads a
column with no row, never enters the order. `instRows` is `Rows.compose` of them.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Inst

open Compose

/-! ## Columns of a template instance -/

/-- **Column `off` of net `n`'s slot `i`** of a template instance, as one number. -/
def code (n i off : ℕ) : ℕ := Nat.pair n (Nat.pair i off)

/-- A code's net. -/
def codeNet (x : ℕ) : ℕ := x.unpair.1

/-- A code's slot. -/
def codeSlot (x : ℕ) : ℕ := x.unpair.2.unpair.1

/-- A code's column in its slot. -/
def codeOff (x : ℕ) : ℕ := x.unpair.2.unpair.2

@[simp] theorem codeNet_code (n i off : ℕ) : codeNet (code n i off) = n := by simp [codeNet, code]

@[simp] theorem codeSlot_code (n i off : ℕ) : codeSlot (code n i off) = i := by simp [codeSlot, code]

@[simp] theorem codeOff_code (n i off : ℕ) : codeOff (code n i off) = off := by simp [codeOff, code]

theorem code_parts (x : ℕ) : code (codeNet x) (codeSlot x) (codeOff x) = x := by
  simp [code, codeNet, codeSlot, codeOff]

/-- Net `n` of a circuit. -/
abbrev netOf (c : Circuit) (n : ℕ) : Net := (c.nets.getD n default).2

/-- Net `n`'s slot width. -/
abbrev netW (c : Circuit) (n : ℕ) : ℕ := 2 ^ (netOf c n).unitLog

/-- Net `n`'s input rows. -/
abbrev netIn (c : Circuit) (n : ℕ) : ℕ := (netOf c n).inWords * WORD

/-- **A column's position in the block** at template instance `g`: its slot's column, plus its column in the slot. -/
def pos (c : Circuit) (g x : ℕ) : ℕ := c.slotCol (codeNet x) g (codeSlot x) + codeOff x

theorem pos_code (c : Circuit) (g n i off : ℕ) : pos c g (code n i off) = c.slotCol n g i + off := by
  simp [pos]

/-! ## Each column's row -/

/-- Net `n`'s local `A` entry. -/
def entA (c : Circuit) (n r j : ℕ) : ZMod 2 := sparseEntry (netOf c n).a r j

/-- Net `n`'s local `B` entry, a lookup slot's table side included. -/
def entB (c : Circuit) (n r j : ℕ) : ZMod 2 :=
  sparseEntry (netOf c n).b r j + lookupEntry (netOf c n).lookup r j

/-- A local row with its own column `o` added. -/
def addSelf (e : ℕ → ZMod 2) (o : ℕ) : ℕ → ZMod 2 := fun j => e j + if j = o then 1 else 0

/-- The columns of net `n`'s slot `i` where a local row `e` has a 1. -/
def reads (c : Circuit) (n i : ℕ) (e : ℕ → ZMod 2) : List ℕ :=
  ((List.range (netW c n)).filter fun j => decide (e j = 1)).map (code n i)

/-- The wire into input `o` of net `n`'s slot `i`: the first whose destination is `o`'s port. -/
def wireAt (c : Circuit) (n i o : ℕ) : Option Wire :=
  (c.wires.toList.filter fun w => decide (w.dst = (n, i, portOf (portCols (netOf c n).inGroups) o))).head?

/-- **The column a wire copies into input `o` of net `n`'s slot `i`**, at an input port bit a wire reaches: the bit of
the wire's source port at `o`'s place in its own port. -/
def feed (c : Circuit) (n i o : ℕ) : Option ℕ :=
  if o ∈ portBits (netOf c n).inGroups then
    (wireAt c n i o).map fun w => code w.src.1 w.src.2.1
      (((portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default).1 +
        (o - (portCols (netOf c n).inGroups)[portOf (portCols (netOf c n).inGroups) o]!.1))
  else none

/-- **Column `o` of net `n`'s slot `i`'s row**, outside the rows' circuits: `some none` at the constant (a copy of
`one`), the local rows at a computed row, the local rows with Δ's pair `(x, x)` and the wire's source at an input bit a
wire fills, and `none` (no row) at any other column. -/
def rowAt (c : Circuit) (n i o : ℕ) : Option (Option (List ℕ × List ℕ)) :=
  if n < c.nets.size ∧ n ≠ c.shaNet ∧ n ≠ c.hmNet ∧ i < c.perVu.getD n 0 ∧ o < netW c n then
    if o = (netOf c n).constPos then some none
    else if netIn c n ≤ o ∧ o < (netOf c n).constPos then
      some (some (reads c n i (entA c n o), reads c n i (entB c n o)))
    else match feed c n i o with
      | some s => some (some (reads c n i (addSelf (entA c n o) o) ++ [s],
          reads c n i (addSelf (entB c n o) o) ++ [s]))
      | none => none
  else none

/-- A column's row (`rowAt` at its code's parts). -/
def row (c : Circuit) (x : ℕ) : Option (Option (List ℕ × List ℕ)) :=
  rowAt c (codeNet x) (codeSlot x) (codeOff x)

theorem rowAt_dom {c : Circuit} {n i o : ℕ} (h : rowAt c n i o ≠ none) :
    n < c.nets.size ∧ n ≠ c.shaNet ∧ n ≠ c.hmNet ∧ i < c.perVu.getD n 0 ∧ o < netW c n := by
  unfold rowAt at h
  split at h
  · assumption
  · exact absurd rfl h

theorem rowAt_const {c : Circuit} {n i o : ℕ} (h : rowAt c n i o = some none) : o = (netOf c n).constPos := by
  have hd := rowAt_dom (c := c) (n := n) (i := i) (o := o) (by rw [h]; simp)
  unfold rowAt at h
  rw [ite_eq_left hd] at h
  by_contra hc
  rw [ite_eq_right hc] at h
  split at h
  · cases h
  · split at h <;> cases h

theorem rowAt_comp {c : Circuit} {n i o : ℕ} {p : List ℕ × List ℕ} (h : rowAt c n i o = some (some p)) :
    o ≠ (netOf c n).constPos ∧
      ((netIn c n ≤ o ∧ o < (netOf c n).constPos ∧ p = (reads c n i (entA c n o), reads c n i (entB c n o))) ∨
        (¬ (netIn c n ≤ o ∧ o < (netOf c n).constPos) ∧ ∃ s, feed c n i o = some s ∧
          p = (reads c n i (addSelf (entA c n o) o) ++ [s], reads c n i (addSelf (entB c n o) o) ++ [s]))) := by
  have hd := rowAt_dom (c := c) (n := n) (i := i) (o := o) (by rw [h]; simp)
  unfold rowAt at h
  rw [ite_eq_left hd] at h
  by_cases hc : o = (netOf c n).constPos
  · rw [ite_eq_left hc] at h; cases h
  rw [ite_eq_right hc] at h
  refine ⟨hc, ?_⟩
  by_cases hm : netIn c n ≤ o ∧ o < (netOf c n).constPos
  · rw [ite_eq_left hm] at h
    exact Or.inl ⟨hm.1, hm.2, (Option.some.inj (Option.some.inj h)).symm⟩
  · rw [ite_eq_right hm] at h
    refine Or.inr ⟨hm, ?_⟩
    revert h
    rcases feed c n i o with _ | s
    · simp
    · intro h
      exact ⟨s, rfl, (Option.some.inj (Option.some.inj h)).symm⟩

/-! ## The logical inputs -/

/-- **The unit's stacked inputs that no wire fills**: stacked input `x` is input `x % W` of copy `x / W`. -/
def inX (c : Circuit) : List ℕ :=
  (List.range (c.upv * netIn c c.unitNet)).filter fun x =>
    (feed c c.unitNet (x / netIn c c.unitNet) (x % netIn c c.unitNet)).isNone

/-- A stacked unit input's column. -/
def inCode (c : Circuit) (x : ℕ) : ℕ := code c.unitNet (x / netIn c c.unitNet) (x % netIn c c.unitNet)

/-- **The logical inputs**: the columns of the unit's stacked inputs no wire fills. -/
def inCols (c : Circuit) : List ℕ := (inX c).map (inCode c)

/-! ## The order -/

/-- **A column is ready** once its row reads only listed columns (a constant always is; a column with no row never). -/
def ready (c : Circuit) (acc : List ℕ) (x : ℕ) : Bool :=
  match row c x with
  | some none => true
  | some (some p) => (p.1 ++ p.2).all fun y => decide (y ∈ acc)
  | none => false

/-- Every column of every net's slots in a template instance. -/
def cols (c : Circuit) : List ℕ :=
  (List.range c.nets.size).flatMap fun n => (List.range (c.perVu.getD n 0)).flatMap fun i =>
    (List.range (netW c n)).map (code n i)

/-- One pass: append every unlisted column that is ready. -/
def step (c : Circuit) (acc : List ℕ) : List ℕ :=
  acc ++ (cols c).filter fun x => !decide (x ∈ acc) && ready c acc x

/-- **The logical order**: the inputs, then as many passes as there are columns. -/
def order (c : Circuit) : List ℕ := (step c)^[(cols c).length] (inCols c)

/-- **The template instance's rows in logical order.** -/
def logical (c : Circuit) : Logical := ⟨(inCols c).length, order c, fun x => (row c x).getD none⟩

/-- What every pass keeps: the inputs first, and each later column has a row, which reads only columns before it. -/
def Inv (c : Circuit) (l : List ℕ) : Prop :=
  inCols c <+: l ∧ ∀ ℓ < l.length, (inCols c).length ≤ ℓ → row c (l.getD ℓ 0) ≠ none ∧
    ∀ p, row c (l.getD ℓ 0) = some (some p) → ∀ y ∈ p.1 ++ p.2, y ∈ l.take ℓ

theorem ready_spec {c : Circuit} {acc : List ℕ} {x : ℕ} (h : ready c acc x = true) :
    row c x ≠ none ∧ ∀ p, row c x = some (some p) → ∀ y ∈ p.1 ++ p.2, y ∈ acc := by
  unfold ready at h
  revert h
  rcases row c x with _ | _ | q
  · simp
  · intro _
    exact ⟨by simp, fun p hp => by simp at hp⟩
  · intro h
    refine ⟨by simp, fun p hp y hy => ?_⟩
    simp only [Option.some.injEq] at hp
    subst hp
    simp only [List.all_eq_true, decide_eq_true_eq] at h
    exact h y hy

theorem inv_step {c : Circuit} {l : List ℕ} (h : Inv c l) : Inv c (step c l) := by
  obtain ⟨hpre, hrow⟩ := h
  refine ⟨hpre.trans (List.prefix_append _ _), fun ℓ hℓ hn => ?_⟩
  unfold step at hℓ ⊢
  have hready : ∀ x ∈ (cols c).filter (fun x => !decide (x ∈ l) && ready c l x), ready c l x = true := by
    intro x hx
    have := (List.mem_filter.1 hx).2
    simp only [Bool.and_eq_true] at this
    exact this.2
  generalize (cols c).filter (fun x => !decide (x ∈ l) && ready c l x) = m at hℓ hready ⊢
  by_cases hl : ℓ < l.length
  · have hg : (l ++ m).getD ℓ 0 = l.getD ℓ 0 := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left hl, ← List.getD_eq_getElem?_getD]
    rw [hg, List.take_append_of_le_length hl.le]
    exact hrow ℓ hl hn
  · rw [List.length_append] at hℓ
    have hlm : ℓ - l.length < m.length := by omega
    have hg : (l ++ m).getD ℓ 0 = m[ℓ - l.length] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), List.getElem?_eq_getElem hlm]
      rfl
    rw [hg]
    obtain ⟨h1, h2⟩ := ready_spec (hready _ (List.getElem_mem hlm))
    refine ⟨h1, fun p hp y hy => ?_⟩
    rw [List.take_append, List.take_of_length_le (by omega)]
    exact List.mem_append_left _ (h2 p hp y hy)

theorem inv_order (c : Circuit) : Inv c (order c) := by
  unfold order
  generalize (cols c).length = k
  induction k with
  | zero =>
    show Inv c (inCols c)
    exact ⟨List.prefix_refl _, fun ℓ hℓ hn => absurd hℓ (by omega)⟩
  | succ k ih =>
    rw [Function.iterate_succ_apply']
    exact inv_step ih

theorem row_of_getD {c : Circuit} {x : ℕ} {p : List ℕ × List ℕ} (h : (row c x).getD none = some p) :
    row c x = some (some p) := by
  revert h
  rcases row c x with _ | _ | q <;> simp

/-- **The template instance's rows read only earlier columns**, or `one`. -/
theorem logical_ordered (c : Circuit) : Ordered (logical c) := by
  obtain ⟨hpre, hrow⟩ := inv_order c
  refine ordered_of_take hpre.length_le fun ℓ hℓ hn p hp y hy => ?_
  exact (hrow ℓ hℓ hn).2 p (row_of_getD hp) y hy

theorem order_getD_in {c : Circuit} {ℓ : ℕ} (hℓ : ℓ < (inCols c).length) :
    (order c).getD ℓ 0 = (inCols c).getD ℓ 0 := by
  obtain ⟨t, ht⟩ := (inv_order c).1
  rw [← ht, List.getD_eq_getElem?_getD, List.getElem?_append_left hℓ, ← List.getD_eq_getElem?_getD]

theorem order_row {c : Circuit} {ℓ : ℕ} (hℓ : ℓ < (order c).length) (hn : (inCols c).length ≤ ℓ) :
    row c ((order c).getD ℓ 0) ≠ none :=
  ((inv_order c).2 ℓ hℓ hn).1

/-- **A template instance's rows**: every net slot's but the rows' circuits', in logical order, joined by Δ's wires. -/
def instRows (c : Circuit) : Rows := Rows.compose (logical c) (logical_ordered c)

theorem instRows_nIn (c : Circuit) : (instRows c).nIn = (inCols c).length := rfl

theorem instRows_w (c : Circuit) : (instRows c).w = (order c).length + 1 := Rows.compose_w

end FlockSoundness.Discharge.Inst
