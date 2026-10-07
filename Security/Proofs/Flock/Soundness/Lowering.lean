import Proofs.Flock.Soundness.Audit.Circuit
import Proofs.Flock.Soundness.Model.Statement

/-!
# The compiled lowering: a satisfied table makes every placed unit correct

**The audit's circuit is the rows the verifier parses** (PROTOCOL.md §16.5, `Flock/Net.lean`). A
template's unit is `nIn` input columns, `nComp` computed rows and the constant, its last column
(`Rows`). Computed row `i` is column `nIn + i`:

$$z_{nIn+i} = \Big(\bigoplus_{c \in a_i} z_c\Big)\cdot\Big(\bigoplus_{c \in b_i} z_c\Big),$$

reading earlier columns and the constant. In the circuit it is one gate, `Op.row`. A unit of a
partition is an instance of a template's rows when its gates are those rows, read through a wiring
of the unit's columns to the circuit's gates (`Partition.IsRowsUnit`). Only input columns may share
a gate: a unit may read one source on several inputs.

**The theorem, for every template** (`lowering_sound`). Suppose a witness satisfies a statement in
which unit `u`'s rows are placed in block `o` (`Placement`: the block matrices' rows at the unit's
computed columns read exactly its rows' columns, and its constant's rows read the pin), and the
unit's input columns on one gate carry one value there. Then the transcript decoded from that
block makes `u` correct, and carries `true` on the constant (`UnitPlace.decode_one`). What each
template's rows compute is the certificate's (`IsRowsUnit.computes_of_cert`, `Certificate.lean`),
not a hypothesis.

`UnitPlace` bundles a drawn unit's place in a table: its rows, its instance, its block and its
columns, where every satisfying witness carries one value on the unit's inputs that share a gate.
`UnitPlace.correct` is the compiled lowering for that unit.
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

theorem comp_injective : Function.Injective R.comp := fun i j h => by
  have := congrArg Fin.val h
  simp only [comp] at this
  exact Fin.ext (by omega)

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

theorem decode_wire {z : Fin R.w → Bool} (hz : I.Agree z) (c : Fin R.w) : I.decode z (I.wire c) = z c := by
  have h : ∃ c', I.wire c' = I.wire c := ⟨c, rfl⟩
  simp only [decode, dif_pos h]
  rcases I.shared _ _ (Classical.choose_spec h) with he | ⟨h1, h2⟩
  · rw [he]
  · exact hz _ _ (Classical.choose_spec h) h1 h2

/-- A column a row reads, outside the unit, is one of the unit's committed wires. -/
theorem mem_io_of_read {i : Fin R.nComp} {c : Fin R.w} (hc : c ∈ R.a i ++ R.b i)
    (hout : P.unit (I.wire c) ≠ some u) : I.wire c ∈ P.io u := by
  classical
  have hread : I.wire c ∈ (C.op (I.wire (R.comp i))).args := by
    rw [I.op i]
    simp only [Op.args, List.mem_append, List.mem_map]
    rcases List.mem_append.1 hc with h | h
    · exact Or.inl ⟨c, h, rfl⟩
    · exact Or.inr ⟨c, h, rfl⟩
  have hcom : I.wire c ∈ P.committed :=
    P.mem_committed_of_read hread (by rw [I.mem i]; exact Ne.symm hout)
  simp only [io, Finset.mem_filter]
  exact ⟨hcom, Or.inr ⟨_, I.mem i, hread⟩⟩

/-- **Rows satisfied, unit correct.** If the unit's column values satisfy its rows and the transcript
carries them on the unit's committed wires, the unit is correct. -/
theorem correct {z : Fin R.w → Bool} (hz : R.Sat z) {X : Fin C.N → Bool}
    (hX : ∀ c, I.wire c ∈ P.io u → X (I.wire c) = z c) : P.Correct X u := by
  classical
  have key : ∀ k (i : Fin R.nComp), i.val = k →
      P.localEval X u (I.wire (R.comp i)) = z (R.comp i) := by
    intro k
    induction k using Nat.strong_induction_on with
    | _ k ih =>
    intro i hik
    have hcol : ∀ c ∈ R.a i ++ R.b i, P.localEval X u (I.wire c) = z c := by
      intro c hc
      by_cases hcomp : ∃ i', c = R.comp i'
      · obtain ⟨i', rfl⟩ := hcomp
        have hlt : i'.val < i.val := by
          rcases R.topo i _ hc with h | h
          · simp only [Rows.comp] at h; omega
          · simp only [Rows.comp] at h; have := i'.isLt; omega
        exact ih i'.val (hik ▸ hlt) i' rfl
      · push Not at hcomp
        have hout := I.outside c hcomp
        rw [P.localEval_of_not_mem X hout]
        exact hX c (I.mem_io_of_read hc hout)
    rw [P.localEval_of_mem X (I.mem i), I.op i, hz i]
    simp only [Op.apply, xorSum_map]
    congr 1
    · exact xorSum_congr fun c hc => hcol c (List.mem_append_left _ hc)
    · exact xorSum_congr fun c hc => hcol c (List.mem_append_right _ hc)
  intro g hg hgu
  obtain ⟨i, rfl⟩ := I.onto g hgu
  rw [key i.val i rfl]
  refine hX (R.comp i) ?_
  simp only [io, Finset.mem_filter]
  exact ⟨hg, Or.inl hgu⟩

/-- The decoded transcript makes the unit correct when its column values satisfy its rows and agree on
shared gates. -/
theorem correct_decode {z : Fin R.w → Bool} (hz : R.Sat z) (ha : I.Agree z) : P.Correct (I.decode z) u :=
  I.correct hz fun c _ => I.decode_wire ha c

end IsRowsUnit

end Audit.Partition

/-! ## Rows placed in a statement -/

/-- A witness bit as a Boolean. -/
def bit (x : ZMod 2) : Bool := decide (x = 1)

theorem bit_add (x y : ZMod 2) : bit (x + y) = Bool.xor (bit x) (bit y) := by
  fin_cases x <;> fin_cases y <;> rfl

theorem bit_mul (x y : ZMod 2) : bit (x * y) = (bit x && bit y) := by
  fin_cases x <;> fin_cases y <;> rfl

theorem bit_sum {ι : Type} (l : List ι) (f : ι → ZMod 2) :
    bit (l.map f).sum = xorSum l (bit ∘ f) := by
  induction l with
  | nil => rfl
  | cons x l ih =>
      simp only [List.map_cons, List.sum_cons, bit_add, ih]
      rfl

theorem sum_count_mul {k : ℕ} (l : List (Fin k)) (v : Fin k → ZMod 2) :
    ∑ j, ((l.count j : ℕ) : ZMod 2) * v j = (l.map v).sum := by
  induction l with
  | nil => simp
  | cons c l ih =>
      simp only [List.count_cons, Nat.cast_add, add_mul, Finset.sum_add_distrib, ih, List.map_cons,
        List.sum_cons]
      rw [add_comm]
      congr 1
      rw [Finset.sum_eq_single c]
      · simp
      · intro j _ hj
        have : (c == j) = false := by simpa using fun h => hj h.symm
        simp [this]
      · simp

/-- **A unit's rows placed in a statement**, at the block columns `col`: the block matrices' rows at
the unit's computed columns read exactly the placed columns of its rows, and the rows at its constant
read the pin (every slot's constant is wired to the pin, PROTOCOL.md §16.1). -/
structure Placement (S : Statement) (R : Rows) (col : Fin R.w → Fin (2 ^ S.kLog)) : Prop where
  hA : ∀ i j, S.A₀ (col (R.comp i)) j = (((R.a i).map col).count j : ZMod 2)
  hB : ∀ i j, S.B₀ (col (R.comp i)) j = (((R.b i).map col).count j : ZMod 2)
  hA₁ : ∀ j, S.A₀ (col R.one) j = if j = S.pin then 1 else 0
  hB₁ : ∀ j, S.B₀ (col R.one) j = if j = S.pin then 1 else 0

theorem mulVec_placed {S : Statement} (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2))
    (r : Fin (2 ^ S.kLog)) {ι : Type} (l : List ι) (col : ι → Fin (2 ^ S.kLog))
    (h : ∀ j, M r j = ((l.map col).count j : ZMod 2)) (v : Fin (2 ^ S.kLog) → ZMod 2) :
    M.mulVec v r = (l.map fun c => v (col c)).sum := by
  simp only [Matrix.mulVec, dotProduct, h]
  rw [sum_count_mul, List.map_map]
  rfl

/-- **A satisfied statement satisfies the placed rows**, in every block. -/
theorem rowsSat_of_satisfies {S : Statement} {R : Rows} {col : Fin R.w → Fin (2 ^ S.kLog)}
    (hpl : Placement S R col) {z : Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) :
    R.Sat fun c => bit (S.block z o (col c)) := by
  intro i
  have h := hz.1 o (col (R.comp i))
  rw [mulVec_placed S.A₀ _ (R.a i) col (hpl.hA i), mulVec_placed S.B₀ _ (R.b i) col (hpl.hB i)] at h
  have hb := congrArg bit h
  rw [bit_mul, bit_sum, bit_sum] at hb
  exact hb.symm

/-- **A placed unit's constant is 1** in every block of a satisfying witness. -/
theorem Placement.one_eq {S : Statement} {R : Rows} {col : Fin R.w → Fin (2 ^ S.kLog)}
    (hpl : Placement S R col) {z : Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) :
    S.block z o (col R.one) = 1 := by
  have h := hz.1 o (col R.one)
  simp only [Matrix.mulVec, dotProduct, hpl.hA₁, hpl.hB₁, ite_mul, one_mul, zero_mul,
    Finset.sum_ite_eq', Finset.mem_univ, if_true, hz.2.1 o] at h
  exact h.symm

/-- **The compiled lowering, for every template.** If a witness satisfies a statement in which unit
`u`'s rows are placed in block `o` at the columns `col`, and the unit's inputs on one gate carry one
value there, the transcript decoded from that block makes `u` correct. -/
theorem lowering_sound {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
    (I : P.IsRowsUnit u R) {S : Statement} {col : Fin R.w → Fin (2 ^ S.kLog)}
    (hpl : Placement S R col) (o : Fin (2 ^ S.nbl)) {z : Witness S.m} (hz : S.Satisfies z)
    (ha : I.Agree fun c => bit (S.block z o (col c))) :
    P.Correct (I.decode fun c => bit (S.block z o (col c))) u :=
  I.correct_decode (rowsSat_of_satisfies hpl hz o) ha

/-- **A drawn unit's place in a table**: its template's rows, its instance in the circuit, and the
block and columns where the table's statement holds its rows. The unit's input columns on one gate
carry one value in every witness satisfying the statement (`aliased`): they copy one source's
position, or forced-zero rows. -/
structure UnitPlace {C : Circuit} {n : ℕ} (P : Audit.Partition C n) (S : Statement) (u : Fin n) where
  R : Rows
  inst : P.IsRowsUnit u R
  o : Fin (2 ^ S.nbl)
  col : Fin R.w → Fin (2 ^ S.kLog)
  placed : Placement S R col
  aliased : ∀ c c', inst.wire c = inst.wire c' → c.val < R.nIn → c'.val < R.nIn →
    ∀ z : Witness S.m, S.Satisfies z → S.block z o (col c) = S.block z o (col c')

namespace UnitPlace

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {S : Statement} {u : Fin n}

/-- The transcript a witness gives for a placed unit, read from its block. -/
noncomputable def decode (pl : UnitPlace P S u) (z : Witness S.m) : Fin C.N → Bool :=
  pl.inst.decode fun c => bit (S.block z pl.o (pl.col c))

theorem agree (pl : UnitPlace P S u) {z : Witness S.m} (hz : S.Satisfies z) :
    pl.inst.Agree fun c => bit (S.block z pl.o (pl.col c)) :=
  fun c c' h hc hc' => congrArg bit (pl.aliased c c' h hc hc' z hz)

/-- **The compiled lowering for a placed unit**: every witness satisfying the table's statement
decodes to a transcript on which the unit is correct. -/
theorem correct (pl : UnitPlace P S u) {z : Witness S.m} (hz : S.Satisfies z) :
    P.Correct (pl.decode z) u :=
  lowering_sound pl.inst pl.placed pl.o hz (pl.agree hz)

/-- The decoded transcript carries `true` on the gate the unit's constant reads. -/
theorem decode_one (pl : UnitPlace P S u) {z : Witness S.m} (hz : S.Satisfies z) :
    pl.decode z (pl.inst.wire pl.R.one) = true := by
  simp only [decode]
  rw [pl.inst.decode_wire (pl.agree hz), pl.placed.one_eq hz pl.o]
  rfl

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

omit p R ins one hone in
/-- **Every unit of a program is an instance of its template's rows** (program level). -/
theorem rowsWire : ∀ {N n : ℕ} (p : Prog N n) (u : Fin n), Nonempty (p.RowsWire u (p.rows u))
  | _, _, .inputs _, u => u.elim0
  | _, _, @snoc N₀ n₀ p R ins one hone, u => by
      refine Fin.lastCases ?_ (fun v => ?_) u
      · have hr : (snoc p R ins one hone).rows (Fin.last n₀) = R := by simp [rows]
        rw [hr]
        exact ⟨rowsWire_last p R ins one hone⟩
      · have hr : (snoc p R ins one hone).rows (Fin.castSucc v) = p.rows v := by
          simp [rows, v.isLt]
        rw [hr]
        obtain ⟨W⟩ := rowsWire p v
        exact ⟨rowsWire_snoc p R ins one hone W⟩

omit p R ins one hone in
/-- **Every unit of a program's circuit is an instance of its template's rows.** -/
theorem isRowsUnit {N n : ℕ} (p : Prog N n) (outs : Finset (Fin N)) (u : Fin n) :
    Nonempty ((p.partition outs).IsRowsUnit u (p.rows u)) :=
  let ⟨W⟩ := p.rowsWire u
  ⟨W.toIsRowsUnit outs⟩

end Prog

end FlockSoundness
