import Definitions.Flock.Circuit
import Definitions.Flock.Statement

/-!
# The compiled lowering: templates' rows, their placement in a statement, and programs of units

**The audit's circuit is the rows the verifier parses** (PROTOCOL.md §16.5, `Flock/Net.lean`). A
template's unit is `nIn` input columns, `nComp` computed rows and the constant, its last column
(`Rows`). Computed row `i` is column `nIn + i`:

$$z_{nIn+i} = \Big(\bigoplus_{c \in a_i} z_c\Big)\cdot\Big(\bigoplus_{c \in b_i} z_c\Big),$$

reading earlier columns and the constant. In the circuit it is one gate, `Op.row`.

* `Rows` and `Rows.Sat`; `Partition.IsRowsUnit`: a unit of a partition is an instance of a template's rows when its
  gates are those rows, read through a wiring of the unit's columns to the circuit's gates, with its `decode`.
* `Placement`: unit `u`'s rows placed in block `o` of a statement (the block matrices' rows at the unit's computed
  columns read exactly its rows' columns, and its constant's rows read the pin); `PlacedSat`, placed up to the
  matrices, in a statement every witness of the table satisfies. `UnitPlace` bundles a drawn unit's place in a table:
  its rows, its instance, its block and its columns.
* `Prog`: a program of units, each a template's rows appended after the gates it reads, with its circuit
  (`Prog.circuit`), partition and rows, and `Prog.RowsWire`, the wiring that makes each unit an `IsRowsUnit`
  (`rowsWire_snoc`, `rowsWire_last`). The lemmas these constructions carry as proofs (`topo`, `input_iff`, the
  `colWire`, `unit` and `op` equations) are here with them.

The theorem, for every template, is `lowering_sound` in `Proofs/Flock/Soundness/Lowering.lean`: a satisfied table
makes every placed unit correct.
-/

namespace FlockSoundness

open Audit Model

/-- **A template's rows**, as the verifier parses them: columns `0 … nIn − 1` are the unit's input
bits, column `nIn + i` is computed row `i`, and the last column `nIn + nComp` is the constant. Every
row reads earlier columns and the constant. -/
structure Rows where
  nIn : ℕ
  nComp : ℕ
  a : Fin nComp → List (Fin (nIn + nComp + 1))
  b : Fin nComp → List (Fin (nIn + nComp + 1))
  topo : ∀ i, ∀ c ∈ a i ++ b i, c.val < nIn + i.val ∨ c.val = nIn + nComp

namespace Rows

variable (R : Rows)

/-- The columns. -/
abbrev w : ℕ := R.nIn + R.nComp + 1

/-- Computed row `i`'s column. -/
def comp (i : Fin R.nComp) : Fin R.w := ⟨R.nIn + i.val, by show _ < R.nIn + R.nComp + 1; omega⟩

/-- The constant's column. -/
def one : Fin R.w := ⟨R.nIn + R.nComp, by show _ < R.nIn + R.nComp + 1; omega⟩

/-- **The rows are satisfied** by column values `z`. -/
def Sat (z : Fin R.w → Bool) : Prop :=
  ∀ i, z (R.comp i) = (xorSum (R.a i) z && xorSum (R.b i) z)

end Rows

namespace Audit.Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- **Unit `u` is an instance of the rows `R`.** Computed row `i` is the unit's gate
`wire (R.comp i)`, the row gate of its columns' wires. The unit's gates are exactly these, and its
input columns and constant are wires outside it. Two columns share a gate only if both are inputs. -/
structure IsRowsUnit (u : Fin n) (R : Rows) where
  wire : Fin R.w → Fin C.N
  shared : ∀ c c', wire c = wire c' → c = c' ∨ c.val < R.nIn ∧ c'.val < R.nIn
  mem : ∀ i, P.unit (wire (R.comp i)) = some u
  op : ∀ i, C.op (wire (R.comp i)) = .row ((R.a i).map wire) ((R.b i).map wire)
  outside : ∀ c, (∀ i, c ≠ R.comp i) → P.unit (wire c) ≠ some u
  onto : ∀ g, P.unit g = some u → ∃ i, wire (R.comp i) = g

namespace IsRowsUnit

variable {P} {u : Fin n} {R : Rows} (I : P.IsRowsUnit u R)

open Classical in
/-- **The transcript a unit's column values give**: each of the unit's wires carries the value of a
column on it, and every other gate is `false`. -/
noncomputable def decode (z : Fin R.w → Bool) : Fin C.N → Bool :=
  fun g => if h : ∃ c, I.wire c = g then z (Classical.choose h) else false

/-- **Column values agreeing on shared gates**: the unit's input columns on one gate carry one value. -/
def Agree (z : Fin R.w → Bool) : Prop :=
  ∀ c c', I.wire c = I.wire c' → c.val < R.nIn → c'.val < R.nIn → z c = z c'

end IsRowsUnit

end Audit.Partition

/-! ## Rows placed in a statement -/

/-- A witness bit as a Boolean. -/
def bit (x : ZMod 2) : Bool := decide (x = 1)

/-- **A unit's rows placed in a statement**, at the block columns `col`: the block matrices' rows at
the unit's computed columns read exactly the placed columns of its rows, and the rows at its constant
read the pin (every slot's constant is wired to the pin, PROTOCOL.md §16.1). -/
structure Placement (S : Statement) (R : Rows) (col : Fin R.w → Fin (2 ^ S.kLog)) : Prop where
  hA : ∀ i j, S.A₀ (col (R.comp i)) j = (((R.a i).map col).count j : ZMod 2)
  hB : ∀ i j, S.B₀ (col (R.comp i)) j = (((R.b i).map col).count j : ZMod 2)
  hA₁ : ∀ j, S.A₀ (col R.one) j = if j = S.pin then 1 else 0
  hB₁ : ∀ j, S.B₀ (col R.one) j = if j = S.pin then 1 else 0

/-- **A unit's rows placed in a statement up to its matrices**: they are placed (`Placement`) in a statement with the
same blocks, pin and regions and other matrices `A`, `B`, which every witness satisfying `S` satisfies. A pinned
output's row is one: its Δ copies, which no row of the unit holds, are in `S` and not in `A`, `B`. -/
def PlacedSat (S : Statement) (R : Rows) (col : Fin R.w → Fin (2 ^ S.kLog)) : Prop :=
  ∃ A B : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2),
    Placement { S with A₀ := A, B₀ := B } R col ∧
      ∀ z : Witness S.m, S.Satisfies z → ({ S with A₀ := A, B₀ := B } : Statement).Satisfies z

/-- **A drawn unit's place in a table**: its template's rows, its instance in the circuit, and the
block and columns where the table's statement holds its rows. The unit's input columns on one gate
carry one value in every witness satisfying the statement (`aliased`): they copy one source's
position, or forced-zero rows. -/
structure UnitPlace {C : Circuit} {n : ℕ} (P : Audit.Partition C n) (S : Statement) (u : Fin n) where
  R : Rows
  inst : P.IsRowsUnit u R
  o : Fin (2 ^ S.nbl)
  col : Fin R.w → Fin (2 ^ S.kLog)
  placed : PlacedSat S R col
  aliased : ∀ c c', inst.wire c = inst.wire c' → c.val < R.nIn → c'.val < R.nIn →
    ∀ z : Witness S.m, S.Satisfies z → S.block z o (col c) = S.block z o (col c')

namespace UnitPlace

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {S : Statement} {u : Fin n}

/-- The transcript a witness gives for a placed unit, read from its block. -/
noncomputable def decode (pl : UnitPlace P S u) (z : Witness S.m) : Fin C.N → Bool :=
  pl.inst.decode fun c => bit (S.block z pl.o (pl.col c))

end UnitPlace

/-! ## The circuit of a program: its units' rows -/

namespace Audit.Op

variable {N M : ℕ}

/-- An operation read through a map of gates. -/
def lift (f : Fin N → Fin M) : Op N → Op M
  | .input => .input
  | .and a b => .and (f a) (f b)
  | .xor a b => .xor (f a) (f b)
  | .not a => .not (f a)
  | .row a b => .row (a.map f) (b.map f)

theorem args_lift (f : Fin N → Fin M) (o : Op N) : (o.lift f).args = o.args.map f := by
  cases o <;> simp [lift, args]

theorem isInput_lift (f : Fin N → Fin M) (o : Op N) : (o.lift f).isInput = o.isInput := by
  cases o <;> rfl

end Audit.Op

/-- **A program of template units**, unit by unit: `k` input gates, then each unit's computed rows.
A unit's input columns are wired to earlier gates, several possibly to one, and its constant to an
earlier gate none of them is wired to. -/
inductive Prog : ℕ → ℕ → Type
  | inputs (k : ℕ) : Prog k 0
  | snoc {N n : ℕ} (p : Prog N n) (R : Rows) (ins : Fin R.nIn → Fin N) (one : Fin N)
      (hone : ∀ c, ins c ≠ one) : Prog (N + R.nComp) (n + 1)

namespace Prog

/-- A new unit's columns as gates: its inputs and constant where they are wired, its computed rows
after the earlier gates. -/
def colWire {N : ℕ} (R : Rows) (ins : Fin R.nIn → Fin N) (one : Fin N) (c : Fin R.w) :
    Fin (N + R.nComp) :=
  if h : c.val < R.nIn then Fin.castAdd _ (ins ⟨c.val, h⟩)
  else if h' : c.val < R.nIn + R.nComp then ⟨N + (c.val - R.nIn), by omega⟩
  else Fin.castAdd _ one

theorem colWire_comp {N : ℕ} (R : Rows) (ins : Fin R.nIn → Fin N) (one : Fin N)
    (i : Fin R.nComp) : colWire R ins one (R.comp i) = ⟨N + i.val, by omega⟩ := by
  have h1 : ¬ (R.comp i).val < R.nIn := by simp [Rows.comp]
  have h2 : (R.comp i).val < R.nIn + R.nComp := by simp [Rows.comp]
  simp only [colWire, h1, h2, dif_neg, dif_pos, not_false_eq_true]
  exact Fin.ext (by simp [Rows.comp])

/-- An earlier gate or wired column stays below the new unit's gates. -/
theorem colWire_lt_of_not_comp {N : ℕ} (R : Rows) (ins : Fin R.nIn → Fin N) (one : Fin N)
    (c : Fin R.w) (hc : ∀ i, c ≠ R.comp i) : (colWire R ins one c).val < N := by
  unfold colWire
  split_ifs with h h'
  · simp only [Fin.val_castAdd]; exact (ins _).isLt
  · exact absurd (Fin.ext (by simp [Rows.comp]; omega) : c = R.comp ⟨c.val - R.nIn, by omega⟩)
      (hc _)
  · simp only [Fin.val_castAdd]; exact one.isLt

/-- The gates' operations. -/
def op : {N n : ℕ} → Prog N n → Fin N → Audit.Op N
  | _, _, .inputs _, _ => .input
  | _, _, @snoc N₀ _ p R ins one _, g =>
      if h : g.val < N₀ then (p.op ⟨g.val, h⟩).lift (Fin.castAdd R.nComp)
      else .row ((R.a ⟨g.val - N₀, by have := g.isLt; omega⟩).map (colWire R ins one))
        ((R.b ⟨g.val - N₀, by have := g.isLt; omega⟩).map (colWire R ins one))

theorem topo : ∀ {N n : ℕ} (p : Prog N n) (g : Fin N), ∀ a ∈ (p.op g).args, a < g
  | _, _, .inputs _, _, _, ha => by simp [op, Audit.Op.args] at ha
  | _, _, @snoc N₀ _ p R ins one _, g, a, ha => by
      by_cases h : g.val < N₀
      · simp only [op, h, dif_pos, Audit.Op.args_lift] at ha
        obtain ⟨a', ha', rfl⟩ := List.mem_map.1 ha
        have := topo p ⟨g.val, h⟩ a' ha'
        rw [Fin.lt_def] at this ⊢
        simpa using this
      · simp only [op, h, dif_neg, not_false_eq_true, Audit.Op.args, List.mem_append,
          List.mem_map] at ha
        set i : Fin R.nComp := ⟨g.val - N₀, by have := g.isLt; omega⟩ with hi
        have key : ∀ c ∈ R.a i ++ R.b i, colWire R ins one c < g := by
          intro c hc
          rw [Fin.lt_def]
          by_cases hcomp : ∃ i', c = R.comp i'
          · obtain ⟨i', rfl⟩ := hcomp
            rw [colWire_comp]
            rcases R.topo i _ hc with h' | h'
            · simp only [Rows.comp, hi] at h'; simp only; omega
            · simp only [Rows.comp] at h'; have := i'.isLt; omega
          · push Not at hcomp
            have := colWire_lt_of_not_comp R ins one c hcomp
            omega
        rcases ha with ⟨c, hc, rfl⟩ | ⟨c, hc, rfl⟩
        · exact key c (List.mem_append_left _ hc)
        · exact key c (List.mem_append_right _ hc)

/-- The gates' units: the input gates are the input unit, and each unit holds its computed rows. -/
def unit : {N n : ℕ} → Prog N n → Fin N → Option (Fin n)
  | _, _, .inputs _, _ => none
  | _, _, @snoc N₀ n₀ p _ _ _ _, g =>
      if h : g.val < N₀ then (p.unit ⟨g.val, h⟩).map Fin.castSucc else some (Fin.last n₀)

theorem input_iff : ∀ {N n : ℕ} (p : Prog N n) (g : Fin N),
    p.unit g = none ↔ (p.op g).isInput = true
  | _, _, .inputs _, _ => by simp [unit, op, Audit.Op.isInput]
  | _, _, @snoc N₀ _ p _ _ _ _, g => by
      by_cases h : g.val < N₀
      · simp only [unit, op, h, dif_pos, Option.map_eq_none_iff, Audit.Op.isInput_lift]
        exact input_iff p _
      · simp [unit, op, h, Audit.Op.isInput]

/-- **The program's circuit**: its gates are its units' rows. -/
def circuit {N n : ℕ} (p : Prog N n) (outs : Finset (Fin N)) : Audit.Circuit :=
  ⟨N, p.op, p.topo, outs⟩

/-- The program's partition into its units. -/
def partition {N n : ℕ} (p : Prog N n) (outs : Finset (Fin N)) : Audit.Partition (p.circuit outs) n :=
  ⟨p.unit, p.input_iff⟩

/-- Each unit's template rows. -/
def rows : {N n : ℕ} → Prog N n → Fin n → Rows
  | _, _, .inputs _, u => u.elim0
  | _, _, @snoc _ n₀ p R _ _ _, u => if h : u.val < n₀ then p.rows ⟨u.val, h⟩ else R

theorem colWire_val {N : ℕ} (R : Rows) (ins : Fin R.nIn → Fin N) (one : Fin N) (c : Fin R.w) :
    (colWire R ins one c).val =
      if h : c.val < R.nIn then (ins ⟨c.val, h⟩).val
      else if c.val < R.nIn + R.nComp then N + (c.val - R.nIn) else one.val := by
  unfold colWire
  split_ifs <;> rfl

/-- Two of a new unit's columns on one gate are one column, or both inputs. -/
theorem colWire_shared {N : ℕ} (R : Rows) (ins : Fin R.nIn → Fin N) (one : Fin N)
    (hone : ∀ c, ins c ≠ one) (c c' : Fin R.w) (h : colWire R ins one c = colWire R ins one c') :
    c = c' ∨ c.val < R.nIn ∧ c'.val < R.nIn := by
  have hv := congrArg Fin.val h
  rw [colWire_val, colWire_val] at hv
  have hc : c.val < R.nIn + R.nComp + 1 := c.isLt
  have hc' : c'.val < R.nIn + R.nComp + 1 := c'.isLt
  refine (em (c.val < R.nIn ∧ c'.val < R.nIn)).elim Or.inr fun hn => Or.inl ?_
  by_cases h1 : c.val < R.nIn <;> by_cases h2 : c'.val < R.nIn
  · exact absurd ⟨h1, h2⟩ hn
  · rw [dif_pos h1, dif_neg h2] at hv
    have := (ins ⟨c.val, h1⟩).isLt
    split_ifs at hv with h3
    · omega
    · exact absurd (Fin.ext hv) (hone _)
  · rw [dif_neg h1, dif_pos h2] at hv
    have := (ins ⟨c'.val, h2⟩).isLt
    split_ifs at hv with h3
    · omega
    · exact absurd (Fin.ext hv).symm (hone _)
  · rw [dif_neg h1, dif_neg h2] at hv
    have := one.isLt
    split_ifs at hv <;> exact Fin.ext (by omega)

variable {N₀ n₀ : ℕ} (p : Prog N₀ n₀) (R : Rows) (ins : Fin R.nIn → Fin N₀) (one : Fin N₀)
  (hone : ∀ c, ins c ≠ one)

theorem unit_castAdd (g : Fin N₀) :
    (snoc p R ins one hone).unit (Fin.castAdd R.nComp g) = (p.unit g).map Fin.castSucc := by
  simp [unit, g.isLt]

theorem op_castAdd (g : Fin N₀) :
    (snoc p R ins one hone).op (Fin.castAdd R.nComp g) = (p.op g).lift (Fin.castAdd R.nComp) := by
  simp [op, g.isLt]

theorem unit_new (i : Fin R.nComp) :
    (snoc p R ins one hone).unit ⟨N₀ + i.val, by omega⟩ = some (Fin.last n₀) := by
  simp [unit]

theorem op_new (i : Fin R.nComp) :
    (snoc p R ins one hone).op ⟨N₀ + i.val, by omega⟩ =
      .row ((R.a i).map (colWire R ins one)) ((R.b i).map (colWire R ins one)) := by
  have hx : (⟨N₀ + i.val - N₀, by omega⟩ : Fin R.nComp) = i := Fin.ext (by simp)
  simp only [op]
  rw [dif_neg (by simp), hx]

/-- Unit `u` of a program is an instance of `R`: `Partition.IsRowsUnit` for the program's circuit
and partition, stated with the program's gates. -/
structure RowsWire {N n : ℕ} (p : Prog N n) (u : Fin n) (R : Rows) where
  wire : Fin R.w → Fin N
  shared : ∀ c c', wire c = wire c' → c = c' ∨ c.val < R.nIn ∧ c'.val < R.nIn
  mem : ∀ i, p.unit (wire (R.comp i)) = some u
  op : ∀ i, p.op (wire (R.comp i)) = .row ((R.a i).map wire) ((R.b i).map wire)
  outside : ∀ c, (∀ i, c ≠ R.comp i) → p.unit (wire c) ≠ some u
  onto : ∀ g, p.unit g = some u → ∃ i, wire (R.comp i) = g

/-- The program-level instance is the audit's. -/
def RowsWire.toIsRowsUnit {N n : ℕ} {p : Prog N n} {u : Fin n} {R : Rows} (W : p.RowsWire u R)
    (outs : Finset (Fin N)) : (p.partition outs).IsRowsUnit u R :=
  ⟨W.wire, W.shared, W.mem, W.op, W.outside, W.onto⟩

/-- An earlier unit stays an instance of its rows when a unit is added. -/
def rowsWire_snoc {v : Fin n₀} {R' : Rows} (W : p.RowsWire v R') :
    (snoc p R ins one hone).RowsWire (Fin.castSucc v) R' where
  wire := fun c => Fin.castAdd R.nComp (W.wire c)
  shared := fun c c' h => W.shared c c' (Fin.castAdd_injective _ _ h)
  mem := fun i => by rw [unit_castAdd, W.mem i]; rfl
  op := fun i => by
    rw [op_castAdd, W.op i]
    simp only [Audit.Op.lift, List.map_map]
    rfl
  outside := fun c hc => by
    rw [unit_castAdd]
    intro h
    obtain ⟨v', hv', hvv⟩ := Option.map_eq_some_iff.1 h
    exact W.outside c hc (by rw [hv', Fin.castSucc_inj.1 hvv])
  onto := fun g hg => by
    by_cases h : g.val < N₀
    · have hg' : g = Fin.castAdd R.nComp ⟨g.val, h⟩ := Fin.ext rfl
      rw [hg', unit_castAdd] at hg
      obtain ⟨v', hv', hvv⟩ := Option.map_eq_some_iff.1 hg
      rw [Fin.castSucc_inj.1 hvv] at hv'
      obtain ⟨i, hi⟩ := W.onto _ hv'
      exact ⟨i, by rw [hg', hi]⟩
    · have hg' : g = ⟨N₀ + (g.val - N₀), by have := g.isLt; omega⟩ := Fin.ext (by simp; omega)
      rw [hg', unit_new p R ins one hone ⟨g.val - N₀, by have := g.isLt; omega⟩] at hg
      exact absurd (Option.some.inj hg) (Fin.castSucc_ne_last v).symm

/-- The new unit is an instance of its rows. -/
def rowsWire_last : (snoc p R ins one hone).RowsWire (Fin.last n₀) R where
  wire := colWire R ins one
  shared := colWire_shared R ins one hone
  mem := fun i => by rw [colWire_comp, unit_new]
  op := fun i => by rw [colWire_comp, op_new]
  outside := fun c hc => by
    have hlt := colWire_lt_of_not_comp R ins one c hc
    have hc' : colWire R ins one c = Fin.castAdd R.nComp ⟨(colWire R ins one c).val, hlt⟩ :=
      Fin.ext rfl
    rw [hc', unit_castAdd]
    intro h
    obtain ⟨v, -, hv⟩ := Option.map_eq_some_iff.1 h
    exact Fin.castSucc_ne_last v hv
  onto := fun g hg => by
    by_cases h : g.val < N₀
    · have hg' : g = Fin.castAdd R.nComp ⟨g.val, h⟩ := Fin.ext rfl
      rw [hg', unit_castAdd] at hg
      obtain ⟨v, -, hv⟩ := Option.map_eq_some_iff.1 hg
      exact absurd hv (Fin.castSucc_ne_last v)
    · refine ⟨⟨g.val - N₀, by have := g.isLt; omega⟩, ?_⟩
      rw [colWire_comp]
      exact Fin.ext (by simp; omega)

end Prog

end FlockSoundness
