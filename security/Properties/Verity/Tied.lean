module

public import Definitions.Nci

/-!
# The tied circuit with its blocks

The circuit remote audits is cut at its linears: each Pearl+ call's y is an input row of the registered file's circuit
(`LinearOut_v1` takes it and has no gate), so the inference-only policy reads every bit of it as fresh input. The tied
circuit computes it: the registered file's circuit with each cell's y computed by the gates of the product Pearl+
proved, from the call's A row and the weight's row. This module is that circuit, its blocks and its untied gates, as one
definition; that its gates compute Pearl+'s product at W's entries is `Verity.ValueTie.Tying.AtW`.

* **The cut** (`Cut`): the registered file's cells and each one's y, input gates of the circuit, 32 bits a cell, one
  cell a gate. It is the vocabulary's (`Verity.Vocab.cut`), whose registered inputs `θ` hold no y gate
  (`Verity.Vocab.theta_cut`): a served y is a prover value until the tie holds it.
* **The circuit** (`Tying`): the registered file's circuit embedded gate for gate, every y gate computed, every other
  gate as served; the gates it adds are each cell's tie, reading only the cell's A row (served gates), the weights
  (added inputs, registered) and the cell's own tie gates.
* **Its blocks** (`Tying.blocks`): the registered file's blocks, each cell's y gates and its tie's gates in the cell's
  home block (`home`), the block holding the cell's linear output in the registered file (`Cut.Homes`): a block that
  reads the cell's y there, where one does.
* **The untied gates** (`Tying.untiedGates`): the y gates of the cells whose y is not the tie's output, each cell's
  whole word; the placement from below (`Tying.place`, `Tying.FromBelow`) puts each cell at its home block and its y
  gates, so an untied cell's gates are untied gates and distinct cells sit at disjoint gates.
* **The policy on it** (`Tying.CompliesBut`): `Γ` bounds each block's registered rows alone (the tie's gates are PoUW's
  credited work), and the untied y gates count as fresh input, 32 gates a word on a Boolean circuit.
* **A monitoring period** (`Charges`, `PeriodOutsideG`): the blocks outside each window's `G`, those holding more than
  `w` untied words, charged whole, summed over the period's windows.
-/

@[expose] public section

namespace Verity.ValueTie

open Finset

/-- **A cell**: one y word of one Pearl+ call, (the pass, the call's position in its pass, row, column). -/
abbrev Cell : Type := ℕ × ℕ × ℕ × ℕ

/-- **The cut's layout of the committed circuit `C`**: its cells, and their y gates (`y`), input gates of `C`, each the
y of one cell (`cellOf`) and each cell's taking 32 bits. `word` reads a cell's y off its gates as the 32-bit word its
tree stores, and is one-to-one there. -/
structure Cut (C : Verity.Circuit) where
  cells : Finset Cell
  y : Finset (Fin C.N)
  cellOf : Fin C.N → Cell
  y_input : ∀ g ∈ y, C.isInput g = true
  y_cell : ∀ g ∈ y, cellOf g ∈ cells
  width_word : ∀ c ∈ cells, C.width (y.filter fun g => cellOf g = c) = 32
  word : Cell → C.Trace → ℕ
  word_reads : ∀ c (X X' : C.Trace), (∀ g ∈ y, cellOf g = c → X g = X' g) → word c X = word c X'
  word_inj : ∀ c (X X' : C.Trace), word c X = word c X' → ∀ g ∈ y, cellOf g = c → X g = X' g

namespace Cut

variable {C : Verity.Circuit} (L : Cut C)

/-- The y gates of the cell `c`: its word. -/
def yAt (c : Cell) : Finset (Fin C.N) := L.y.filter fun g => L.cellOf g = c

/-- The y gates of the cells `U`. -/
def yOf (U : Finset Cell) : Finset (Fin C.N) := L.y.filter fun g => L.cellOf g ∈ U

/-- **The cell `c`'s home is the block `home c` of the registered file's blocks `Q`**: where a gate of `C` reads one
of the cell's y gates, a gate of block `home c` does; that block holds the cell's linear output. -/
def Homes {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) : Prop :=
  ∀ c ∈ L.cells, (∃ h, ∃ g ∈ L.yAt c, g ∈ C.reads h) → ∃ h, ∃ g ∈ L.yAt c, g ∈ C.reads h ∧ Q.unit h = some (home c)

/-- The layout with no cells, of a circuit that is not cut. -/
def empty (C : Verity.Circuit) : Cut C where
  cells := ∅
  y := ∅
  cellOf _ := (0, 0, 0, 0)
  y_input _ h := absurd h (Finset.notMem_empty _)
  y_cell _ h := absurd h (Finset.notMem_empty _)
  width_word _ h := absurd h (Finset.notMem_empty _)
  word _ _ := 0
  word_reads _ _ _ _ := rfl
  word_inj _ _ _ _ _ h := absurd h (Finset.notMem_empty _)

end Cut

/-- **The tied circuit of the committed circuit `C`, with registered inputs `θ`, at the cut `L`.** The served gates
are gates of the tied circuit of the same width (`emb`); the y gates of the cells it ties (`cells`, the cut's cells
that are the run's calls) are computed there, and every other served gate is as served, another cell's y an input gate
that is not registered: an input gate stays one, registered or not as it was, and a computed gate reads the same gates and
applies the same operation. The gates the tied circuit adds are each cell's tie (`owner`): its computed gates and the
y gates read only the cell's A row (`row`, served gates), the weights' rows (added input gates, registered and holding
`weights`) and the cell's own tie gates. `aRow` and `bRow` are the bytes of the cell's A row and of its weight's row as
their trees store them, the A row read from the served values at `row` alone. -/
structure Tying (C : Verity.Circuit) (θ : Finset (Fin C.N)) (L : Cut C) where
  /-- the cells it ties: the cut's cells that are the run's calls -/
  cells : Finset Cell
  cells_sub : cells ⊆ L.cells
  tied : Verity.RegisteredCircuit
  emb : Fin C.N → Fin tied.D.N
  emb_inj : Function.Injective emb
  emb_w : ∀ g, tied.D.w (emb g) = C.w g
  y_gate : ∀ g ∈ L.yOf cells, tied.D.isInput (emb g) = false
  isInput_emb : ∀ g, g ∉ L.yOf cells → tied.D.isInput (emb g) = C.isInput g
  reads_emb : ∀ g, C.isInput g = false → tied.D.reads (emb g) = (C.reads g).map ⟨emb, emb_inj⟩
  op_emb : ∀ g, C.isInput g = false → ∀ v : tied.D.Trace,
    (tied.D.op (emb g) v).cast (emb_w g) = C.op g fun a => (v (emb a)).cast (emb_w a)
  outputs_emb : ∀ o ∈ C.outputs, emb o ∈ tied.D.outputs
  theta_emb : ∀ g, emb g ∈ tied.θ ↔ g ∈ θ
  /-- an input gate the tied circuit adds is registered: a weight's row -/
  added_registered : ∀ h, (∀ g, emb g ≠ h) → tied.D.isInput h = true → h ∈ tied.θ
  /-- the cell whose tie a computed gate the tied circuit adds is -/
  owner : Fin tied.D.N → Cell
  owner_cell : ∀ h, (∀ g, emb g ≠ h) → tied.D.isInput h = false → owner h ∈ cells
  /-- the served gates of each cell's A row -/
  row : Cell → Finset (Fin C.N)
  /-- a tie's computed gate reads its cell's A row, the weights and its cell's tie gates -/
  reads_tie : ∀ h, (∀ g, emb g ≠ h) → tied.D.isInput h = false → ∀ a ∈ tied.D.reads h,
    (∃ g ∈ row (owner h), emb g = a) ∨ ((∀ g, emb g ≠ a) ∧ (tied.D.isInput a = false → owner a = owner h))
  /-- a y gate reads its cell's A row, the weights and its cell's tie gates -/
  reads_y : ∀ g ∈ L.yOf cells, ∀ a ∈ tied.D.reads (emb g),
    (∃ g' ∈ row (L.cellOf g), emb g' = a) ∨ ((∀ g', emb g' ≠ a) ∧ (tied.D.isInput a = false → owner a = L.cellOf g))
  /-- the registered values of the added input gates, the weights' rows; read at those gates alone -/
  weights : tied.D.Trace
  /-- the bytes of a cell's A row, as its tree stores the row -/
  aRow : C.Trace → Cell → ByteArray
  aRow_reads : ∀ (X X' : C.Trace) (c : Cell), (∀ a ∈ row c, X a = X' a) → aRow X c = aRow X' c
  /-- the bytes of a cell's weight's row, as its tree stores the row -/
  bRow : Cell → ByteArray

namespace Tying

variable {C : Verity.Circuit} {θ : Finset (Fin C.N)} {L : Cut C} (T : Tying C θ L)

/-- The served gates' values in the tied circuit's transcript `X'`. -/
def served (X' : T.tied.D.Trace) : C.Trace := fun g => (X' (T.emb g)).cast (T.emb_w g)

/-- What each served gate's operation in the tied circuit gives on `X'`: at a y gate, its tie's output. -/
def gatesOut (X' : T.tied.D.Trace) : C.Trace := fun g => (T.tied.D.op (T.emb g) X').cast (T.emb_w g)

/-- The tied circuit's transcript `X'` holds the weights' registered rows at the input gates the tied circuit adds. -/
def HoldsWeights (X' : T.tied.D.Trace) : Prop :=
  ∀ h, (∀ g, T.emb g ≠ h) → T.tied.D.isInput h = true → X' h = T.weights h

/-- **`X'` is the served transcript `X` in the tied circuit**: it holds `X`'s value at every served gate, and the
weights' registered rows at the added input gates. -/
def Extends (X : C.Trace) (X' : T.tied.D.Trace) : Prop := T.served X' = X ∧ T.HoldsWeights X'

/-- **The added gates are evaluated in `X'`**: every computed gate the tied circuit adds holds its operation's value on
`X'`. With `Extends` it fixes `X'`, the served transcript with the ties' interiors filled in. -/
def Evaluated (X' : T.tied.D.Trace) : Prop :=
  ∀ h, (∀ g, T.emb g ≠ h) → T.tied.D.isInput h = false → X' h = T.tied.D.op h X'

/-- **The served gate `g` holds the tied gates' output in `X'`**: its value is its tied gate's operation on `X'`. -/
def IsGates (X' : T.tied.D.Trace) (g : Fin C.N) : Prop :=
  X' (T.emb g) = T.tied.D.op (T.emb g) X'

open Classical in
/-- **The untied cells in `X'`**: the cells whose served y is not the tie's output, at some gate of the word. -/
noncomputable def untiedCells (X' : T.tied.D.Trace) : Finset Cell :=
  T.cells.filter fun c => ∃ g ∈ L.yAt c, ¬ T.IsGates X' g

/-- **The untied gates of the cells `U`**: their y gates in the tied circuit, every gate of each one's word. -/
def untiedGates (U : Finset Cell) : Finset (Fin T.tied.D.N) := (L.yOf U).map ⟨T.emb, T.emb_inj⟩

open Classical in
/-- **The tied circuit's blocks** from the registered file's blocks `Q` and the cells' homes `home`: a served computed
gate is in its block of `Q`; a cell's y gates and its tie's computed gates are in its home block; the input gates,
the served ones and the weights, are in none. -/
noncomputable def blocks {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) : Verity.Partition T.tied.D m where
  unit h :=
    if hs : ∃ g, T.emb g = h then
      (if hs.choose ∈ L.yOf T.cells then some (home (L.cellOf hs.choose)) else Q.unit hs.choose)
    else if T.tied.D.isInput h = true then none else some (home (T.owner h))
  input_iff h := by
    by_cases hs : ∃ g, T.emb g = h
    · rw [dif_pos hs]
      have e : T.tied.D.isInput h = T.tied.D.isInput (T.emb hs.choose) := by rw [hs.choose_spec]
      rw [e]
      by_cases hy : hs.choose ∈ L.yOf T.cells
      · rw [if_pos hy, T.y_gate _ hy]
        simp
      · rw [if_neg hy, Q.input_iff, T.isInput_emb _ hy]
    · rw [dif_neg hs]
      by_cases hi : T.tied.D.isInput h = true
      · simp [hi]
      · simp [hi]

/-- **The placement from below**: a cell the circuit ties at its home block and its y gates in the tied circuit; none
elsewhere. Compute-accounting's placement of W's y entries reads this. -/
def place {m : ℕ} (home : Cell → Fin m) (c : Cell) : Option (Fin m × Finset (Fin T.tied.D.N)) :=
  if c ∈ T.cells then some (home c, (L.yAt c).map ⟨T.emb, T.emb_inj⟩) else none

/-- **The placement is from below**, at the registered file's blocks `Q`, the homes `home` and the untied cells `U`:
each placed cell's gates are 32 bits in its block of the tied circuit's blocks, each untied cell's are untied gates,
and distinct cells sit at disjoint gates. -/
def FromBelow {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) (U : Finset Cell) : Prop :=
  (∀ c u O, T.place home c = some (u, O) → O ⊆ (T.blocks Q home).gates u ∧ T.tied.D.width O = 32) ∧
  (∀ c ∈ U, ∀ u O, T.place home c = some (u, O) → O ⊆ T.untiedGates U) ∧
  ∀ c c' u u' O O', c ≠ c' → T.place home c = some (u, O) → T.place home c' = some (u', O') → Disjoint O O'

/-- **The inference-only policy on the tied circuit, the untied y's of `U` counted as fresh input.** The registered
file's circuit divides into blocks `Q` of at most `Γ` registered rows each, the cells at homes that hold their linear
outputs (`Cut.Homes`), and each block of the tied circuit's (`blocks`) but its untied gates reads at most `α` bits
of fresh input a registered row, and the untied gates it holds, 32 bits a word. A block's tie gates count in no row,
and their reads of the cell's A row count as fresh input wherever the block does not hold that row. -/
def CompliesBut (Γ : ℕ) (α : ℝ) (U : Finset Cell) : Prop :=
  ∃ (m : ℕ) (Q : Verity.Partition C m) (home : Cell → Fin m), L.Homes Q home ∧ ∀ u,
    C.size (Q.gates u) ≤ Γ ∧
      (Nci.novelReads T.tied ((T.blocks Q home).gates u \ T.untiedGates U) : ℝ) ≤
        α * C.size (Q.gates u) + T.tied.D.width (T.untiedGates U ∩ (T.blocks Q home).gates u)

end Tying

/-! ## A monitoring period -/

/-- **One window's blocks and untied cells, as the period counts them**: its tied circuit's `m` blocks, each cell's
home, its untied cells, and each block's charge when charged whole. -/
structure Charges where
  m : ℕ
  home : Cell → Fin m
  untied : Finset Cell
  charge : Fin m → ℕ

namespace Charges

variable (W : Charges)

/-- **The blocks outside the window's `G`**: those holding more than `w` untied words. -/
def heavy (w : ℕ) : Finset (Fin W.m) := univ.filter fun u => w < (W.untied.filter fun c => W.home c = u).card

/-- **The window's blocks outside its `G`, charged whole**: their charges' sum. -/
def outside (w : ℕ) : ℕ := ∑ u ∈ W.heavy w, W.charge u

end Charges

/-- **A window of the tied circuit**, at the registered file's blocks `Q`, the homes `home` and the untied cells `U`:
each block charged whole at its registered rows, which `Γ` bounds. -/
def Tying.charges {C : Verity.Circuit} {θ : Finset (Fin C.N)} {L : Cut C} (_ : Tying C θ L) {m : ℕ}
    (Q : Verity.Partition C m) (home : Cell → Fin m) (U : Finset Cell) : Charges :=
  ⟨m, home, U, fun u => C.size (Q.gates u)⟩

/-- **The period's blocks outside each window's `G`**, charged whole and summed over the monitoring period's windows. -/
def periodOutside {n : ℕ} (W : Fin n → Charges) (w : ℕ) : ℕ := ∑ k, (W k).outside w

/-- **The blocks outside `G`, pooled over a monitoring period.** Over any period's windows, each block charged at most
`c` when charged whole, and `G` in each window its blocks holding at most `w` untied words: `w + 1` times the period's
charge for the blocks outside each window's `G` is at most `c` times the period's untied words. With each block
charged at its registered rows (`Tying.charges`), `c` is `Γ`, so the per-window clause's blocks charged whole pool to
`Γ / (w + 1)` a period's untied word, whatever the number of windows. -/
def PeriodOutsideG : Prop :=
  ∀ {n : ℕ} (W : Fin n → Charges) (w c : ℕ), (∀ k u, (W k).charge u ≤ c) →
    (w + 1) * periodOutside W w ≤ c * ∑ k, (W k).untied.card

end Verity.ValueTie
