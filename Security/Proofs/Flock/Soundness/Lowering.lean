import Definitions.Flock.Lowering
import Proofs.Flock.Soundness.Audit.Circuit

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

namespace Rows

variable (R : Rows)

theorem comp_injective : Function.Injective R.comp := fun i j h => by
  have := congrArg Fin.val h
  simp only [comp] at this
  exact Fin.ext (by omega)

end Rows

namespace Audit.Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

namespace IsRowsUnit

variable {P} {u : Fin n} {R : Rows} (I : P.IsRowsUnit u R)

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

theorem Placement.sat {S : Statement} {R : Rows} {col : Fin R.w → Fin (2 ^ S.kLog)} (h : Placement S R col) :
    PlacedSat S R col :=
  ⟨S.A₀, S.B₀, h, fun _ hz => hz⟩

theorem PlacedSat.rowsSat {S : Statement} {R : Rows} {col : Fin R.w → Fin (2 ^ S.kLog)}
    (hpl : PlacedSat S R col) {z : Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) :
    R.Sat fun c => bit (S.block z o (col c)) := by
  obtain ⟨A, B, hp, ht⟩ := hpl
  exact rowsSat_of_satisfies (S := { S with A₀ := A, B₀ := B }) hp (ht z hz) o

theorem PlacedSat.one_eq {S : Statement} {R : Rows} {col : Fin R.w → Fin (2 ^ S.kLog)}
    (hpl : PlacedSat S R col) {z : Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) :
    S.block z o (col R.one) = 1 := by
  obtain ⟨A, B, hp, ht⟩ := hpl
  exact hp.one_eq (S := { S with A₀ := A, B₀ := B }) (ht z hz) o

/-- **The compiled lowering, for every template.** If a witness satisfies a statement in which unit
`u`'s rows are placed in block `o` at the columns `col`, and the unit's inputs on one gate carry one
value there, the transcript decoded from that block makes `u` correct. -/
theorem lowering_sound {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
    (I : P.IsRowsUnit u R) {S : Statement} {col : Fin R.w → Fin (2 ^ S.kLog)}
    (hpl : Placement S R col) (o : Fin (2 ^ S.nbl)) {z : Witness S.m} (hz : S.Satisfies z)
    (ha : I.Agree fun c => bit (S.block z o (col c))) :
    P.Correct (I.decode fun c => bit (S.block z o (col c))) u :=
  I.correct_decode (rowsSat_of_satisfies hpl hz o) ha

namespace UnitPlace

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {S : Statement} {u : Fin n}

theorem agree (pl : UnitPlace P S u) {z : Witness S.m} (hz : S.Satisfies z) :
    pl.inst.Agree fun c => bit (S.block z pl.o (pl.col c)) :=
  fun c c' h hc hc' => congrArg bit (pl.aliased c c' h hc hc' z hz)

/-- **The compiled lowering for a placed unit**: every witness satisfying the table's statement
decodes to a transcript on which the unit is correct. -/
theorem correct (pl : UnitPlace P S u) {z : Witness S.m} (hz : S.Satisfies z) :
    P.Correct (pl.decode z) u :=
  pl.inst.correct_decode (pl.placed.rowsSat hz pl.o) (pl.agree hz)

/-- The decoded transcript carries `true` on the gate the unit's constant reads. -/
theorem decode_one (pl : UnitPlace P S u) {z : Witness S.m} (hz : S.Satisfies z) :
    pl.decode z (pl.inst.wire pl.R.one) = true := by
  simp only [decode]
  rw [pl.inst.decode_wire (pl.agree hz), pl.placed.one_eq hz pl.o]
  rfl

end UnitPlace

/-! ## The circuit of a program: its units' rows -/

namespace Prog

variable {N₀ n₀ : ℕ} (p : Prog N₀ n₀) (R : Rows) (ins : Fin R.nIn → Fin N₀) (one : Fin N₀)
  (hone : ∀ c, ins c ≠ one)

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
