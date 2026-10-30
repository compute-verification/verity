import Proofs.Flock.Soundness.E2E
import Proofs.Flock.Soundness.ComposeDag
import Proofs.Flock.Soundness.Types.Flatten

/-!
# A partition's units: correct on its rows, correct on its gates (verified-lowering phase 1, S4)

`RowsL1` (`E2E.lean`) says that at values with 1 on the units' constants, a unit wrong on the program's Boolean circuit `CB`
is wrong on its rows, the audit's circuit `C`. `rowsL1_of_pairs` proves it from one `UnitPair` per unit:
- on `C`, the unit is an instance of its composed rows (`Partition.IsRowsUnit`), those `deriveChecked` derives for its type;
- on `CB`, it is an instance of its type's unit gates (`Partition.IsGatesUnit` of `Types.Flatten.unitGates`: the type's
  flattening, then a copy of each output, so that each output is a gate of its own, as each output row is a row);
- `CB`'s values are read off `C`'s through a gate map `m`: the Boolean unit's inputs and `one` read its rows' input columns
  and constant, and each committed gate of it is an output's copy, read off that output's row, committed in `C`.

A unit's local evaluation satisfies its rows (`IsRowsUnit.sat_localEval`), so its output rows carry its type's `evalT`
(`IsRowsUnit.localEval_compose`, from `Rows.compose_eval_unit`). On `CB` its local evaluation is its gates' values
(`IsGatesUnit.localEval_eq`), so its outputs' copies carry the same `evalT` (`IsGatesUnit.localEval_type`, from
`flatten_eval`). The partition enters only through the pairs, so the theorem holds whatever chose the units.
-/

open Flock Flock.DeriveAll

namespace FlockSoundness

open Audit Types.Dag Types.Flatten Compose

/-- A gate as an `Audit.Op`, its arguments renamed by `w`. -/
def Types.Flatten.Gate.lift {N : ℕ} (w : ℕ → Fin N) : Gate → Op N
  | .input => .input
  | .and a b => .and (w a) (w b)
  | .xor a b => .xor (w a) (w b)
  | .not a => .not (w a)

namespace Types.Flatten

variable (types : String → Option CircuitType) (words : String → Option (Array ℕ)) (loOf : ℕ → ℕ) (F : ℕ)
  (t : CircuitType)

/-- **A unit's gates**: type `t`'s flattening, then a copy `g ∧ g` of each output's gate `g`, so that each output is a gate
of its own, as each output of the unit's rows is a row of its own (`form · 1`). Output `o`'s copy is gate
`(flatten …).1.size + o`. -/
def unitGates : Array Gate :=
  (flatten types words loOf F t).1 ++ ((List.range t.outputs.size).map fun o =>
    Gate.and ((flatten types words loOf F t).2.getD o 0) ((flatten types words loOf F t).2.getD o 0)).toArray

theorem unitGates_size :
    (unitGates types words loOf F t).size = (flatten types words loOf F t).1.size + t.outputs.size := by
  simp [unitGates]

theorem unitGates_lt {j : ℕ} (hj : j < (flatten types words loOf F t).1.size) :
    (unitGates types words loOf F t)[j]'(by rw [unitGates_size]; omega) = (flatten types words loOf F t).1[j] :=
  Array.getElem_append_left hj

theorem unitGates_copy {o : ℕ} (ho : o < t.outputs.size) :
    (unitGates types words loOf F t)[(flatten types words loOf F t).1.size + o]'(by rw [unitGates_size]; omega) =
      .and ((flatten types words loOf F t).2.getD o 0) ((flatten types words loOf F t).2.getD o 0) := by
  unfold unitGates
  rw [Array.getElem_append_right (by omega)]
  simp

theorem unitGates_topo : Topo (unitGates types words loOf F t) := by
  intro i hi a ha
  obtain ⟨hT, hO⟩ := flatten_topo types words loOf F t
  by_cases hi' : i < (flatten types words loOf F t).1.size
  · rw [unitGates_lt types words loOf F t hi'] at ha
    exact hT i hi' a ha
  · obtain ⟨o, rfl⟩ : ∃ o, i = (flatten types words loOf F t).1.size + o :=
      ⟨i - (flatten types words loOf F t).1.size, by omega⟩
    have ho : o < t.outputs.size := by rw [unitGates_size] at hi; omega
    rw [unitGates_copy types words loOf F t ho] at ha
    simp only [Gate.args, List.mem_cons, List.not_mem_nil, or_false, or_self] at ha
    subst ha
    have := hO o
    omega

theorem gval_unitGates_lt (inp : ℕ → Bool) {j : ℕ} (hj : j < (flatten types words loOf F t).1.size) :
    gval inp (unitGates types words loOf F t) j = gval inp (flatten types words loOf F t).1 j := by
  have e := (unitGates_topo types words loOf F t).ext_whole (k := (flatten types words loOf F t).1.size)
    (by rw [unitGates_size]; omega)
  have hx : (unitGates types words loOf F t).extract 0 (flatten types words loOf F t).1.size =
      (flatten types words loOf F t).1 := by
    unfold unitGates; rw [Array.extract_append_left, Array.extract_size]
  rw [hx] at e
  exact e.gval inp hj

/-- Output `o`'s copy carries output `o`'s gate. -/
theorem gval_unitGates_copy (inp : ℕ → Bool) {o : ℕ} (ho : o < t.outputs.size) :
    gval inp (unitGates types words loOf F t) ((flatten types words loOf F t).1.size + o) =
      gval inp (flatten types words loOf F t).1 ((flatten types words loOf F t).2.getD o 0) := by
  have hT := unitGates_topo types words loOf F t
  have hlt := (flatten_topo types words loOf F t).2 o
  have hi : (flatten types words loOf F t).1.size + o < (unitGates types words loOf F t).size := by
    rw [unitGates_size]; omega
  rw [gval_step hT inp hi, unitGates_copy types words loOf F t ho]
  simp only [Gate.step, Bool.and_self]
  rw [gval_prefix hT inp (Nat.le_of_lt hi) (by omega), gval_unitGates_lt types words loOf F t inp hlt]

end Types.Flatten

theorem Rows.ne_comp_of_lt (R : Rows) {c : Fin R.w} (hc : c.val < R.nIn) : ∀ i, c ≠ R.comp i := fun i h => by
  have := congrArg Fin.val h
  simp only [Rows.comp] at this
  omega

theorem Rows.one_ne_comp (R : Rows) : ∀ i, R.one ≠ R.comp i := fun i h => by
  have := congrArg Fin.val h
  simp only [Rows.one, Rows.comp] at this
  have := i.isLt
  omega

namespace Audit.Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- **Unit `u` is an instance of the gate array `gs`**, whose first `k` gates are inputs. Gate `j` is the circuit's gate
`wire j`: the inputs are gates outside the unit, and every other gate is the unit's, its operation over the wired
arguments. The unit's gates are exactly these. -/
structure IsGatesUnit (u : Fin n) (gs : Array Gate) (k : ℕ) where
  wire : ℕ → Fin C.N
  input : ∀ j (hj : j < gs.size), j < k → gs[j] = .input
  outside : ∀ j < k, P.unit (wire j) ≠ some u
  mem : ∀ j < gs.size, k ≤ j → P.unit (wire j) = some u
  op : ∀ j (hj : j < gs.size), k ≤ j → C.op (wire j) = gs[j].lift wire
  onto : ∀ g, P.unit g = some u → ∃ j < gs.size, k ≤ j ∧ wire j = g

namespace IsGatesUnit

variable {P} {u : Fin n} {gs : Array Gate} {k : ℕ} (W : P.IsGatesUnit u gs k)

/-- **The unit's local evaluation is its gates' values**, on inputs that agree with the transcript on its input gates. -/
theorem localEval_eq (h : Topo gs) (Xv : Fin C.N → Bool) {inp : ℕ → Bool} (hinp : ∀ j < k, inp j = Xv (W.wire j)) :
    ∀ j < gs.size, P.localEval Xv u (W.wire j) = gval inp gs j := by
  intro j
  induction j using Nat.strongRecOn with
  | ind j ih =>
  intro hj
  rw [gval_step h inp hj]
  have hlt : ∀ x ∈ gs[j].args, x < j := h j hj
  by_cases hjk : j < k
  · have hsz : (evalG inp (gs.extract 0 j)).size = j := by rw [evalG_size]; simp [Array.size_extract]; omega
    rw [W.input j hj hjk, P.localEval_of_not_mem Xv (W.outside j hjk)]
    simp only [Gate.step, hsz]
    exact (hinp j hjk).symm
  · have hmem := W.mem j hj (by omega)
    have hop := W.op j hj (by omega)
    rw [P.localEval_of_mem Xv hmem, hop]
    cases hg : gs[j] with
    | input =>
      rw [hg] at hop
      exact absurd (hmem.symm.trans ((P.input_iff _).2 (by rw [hop]; rfl))) (Option.some_ne_none u)
    | and a b =>
      have ha := hlt a (by rw [hg]; simp [Gate.args])
      have hb := hlt b (by rw [hg]; simp [Gate.args])
      simp only [Gate.lift, Op.apply, Gate.step]
      rw [ih a ha (by omega), ih b hb (by omega), gval_prefix h inp (Nat.le_of_lt hj) ha,
        gval_prefix h inp (Nat.le_of_lt hj) hb]
    | xor a b =>
      have ha := hlt a (by rw [hg]; simp [Gate.args])
      have hb := hlt b (by rw [hg]; simp [Gate.args])
      simp only [Gate.lift, Op.apply, Gate.step]
      rw [ih a ha (by omega), ih b hb (by omega), gval_prefix h inp (Nat.le_of_lt hj) ha,
        gval_prefix h inp (Nat.le_of_lt hj) hb]
    | not a =>
      have ha := hlt a (by rw [hg]; simp [Gate.args])
      simp only [Gate.lift, Op.apply, Gate.step]
      rw [ih a ha (by omega), gval_prefix h inp (Nat.le_of_lt hj) ha]

/-- **The unit computes its type.** An instance of type `t`'s unit gates (its inputs and `one` the first `inBits + 1`)
carries on output `o`'s copy `evalT`'s output `o` of the transcript on its inputs, when `one` reads 1. -/
theorem localEval_type {types : String → Option CircuitType} {words : String → Option (Array ℕ)} {loOf : ℕ → ℕ}
    {F : ℕ} {t : CircuitType} (W : P.IsGatesUnit u (unitGates types words loOf F t) (t.inBits + 1))
    (Xv : Fin C.N → Bool) (h1 : Xv (W.wire t.inBits) = true) {o : ℕ} (ho : o < t.outputs.size) :
    P.localEval Xv u (W.wire ((flatten types words loOf F t).1.size + o)) =
      (evalT types words (F + 1) t fun i => Xv (W.wire i)).getD o false := by
  have hinp : ∀ j < t.inBits + 1, (fun i => if i < t.inBits then Xv (W.wire i) else true) j = Xv (W.wire j) := by
    intro j hj
    by_cases hj' : j < t.inBits
    · simp [hj']
    · obtain rfl : j = t.inBits := by omega
      simp [h1]
  rw [W.localEval_eq (unitGates_topo types words loOf F t) Xv hinp _ (by rw [unitGates_size]; omega),
    gval_unitGates_copy types words loOf F t _ ho]
  exact flatten_eval types words loOf F t (fun i => Xv (W.wire i)) o ho

end IsGatesUnit

namespace IsRowsUnit

variable {P} {u : Fin n} {R : Rows} (I : P.IsRowsUnit u R)

theorem localEval_outside (Xv : Fin C.N → Bool) {c : Fin R.w} (hc : ∀ i, c ≠ R.comp i) :
    P.localEval Xv u (I.wire c) = Xv (I.wire c) :=
  P.localEval_of_not_mem Xv (I.outside c hc)

/-- **The unit's local evaluation satisfies its rows.** -/
theorem sat_localEval (Xv : Fin C.N → Bool) : R.Sat fun c => P.localEval Xv u (I.wire c) := by
  intro i
  show P.localEval Xv u (I.wire (R.comp i)) = _
  rw [P.localEval_of_mem Xv (I.mem i), I.op i]
  simp only [Op.apply, xorSum_map, Function.comp_def]

variable {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}

/-- **The unit computes its type through its composed rows.** With its constant at 1, the local evaluation of an instance
of `deriveChecked`'s composed rows carries on output `o`'s row `evalT`'s output `o` of the transcript on its input
columns. -/
theorem localEval_compose (hd : deriveChecked types ls info words unit = .ok done) {t : CircuitType}
    (ht : (ls.getD unit default).type.bind types = some t) {h : Ordered (ofBlock words done (done.getD unit default))}
    (I : P.IsRowsUnit u (Rows.compose (ofBlock words done (done.getD unit default)) h)) (Xv : Fin C.N → Bool)
    (h1 : Xv (I.wire (Rows.compose (ofBlock words done (done.getD unit default)) h).one) = true)
    (F : ℕ) (hF : unit < F) {o : ℕ} (ho : o < t.outputs.size)
    (ℓ : Fin (Rows.compose (ofBlock words done (done.getD unit default)) h).w)
    (hℓ : ℓ.val = lidx (ofBlock words done (done.getD unit default)) ((done.getD unit default).outCols.getD o 0)) :
    P.localEval Xv u (I.wire ℓ) = (evalT types words F t fun i =>
      if hi : i < (Rows.compose (ofBlock words done (done.getD unit default)) h).w then Xv (I.wire ⟨i, hi⟩)
      else false).getD o false :=
  Rows.compose_eval_unit hd ht (I.sat_localEval Xv)
    (by rw [I.localEval_outside Xv (Rows.one_ne_comp _)]; exact h1)
    (fun i hi => by rw [dite_eq_left i.isLt, I.localEval_outside Xv (Rows.ne_comp_of_lt _ hi)]) F hF ho ℓ hℓ

end IsRowsUnit

end Audit.Partition

open Audit.Partition

/-- **One unit's rows and gates, and how the Boolean circuit's values are read off the audit's** (`m`). On `C` the unit is
an instance of the rows `deriveChecked` composes for layout `idx`, of type `t`; on `CB`, of `t`'s unit gates. Through `m`,
the Boolean unit's inputs and `one` read its rows' input columns and constant, and each of its committed gates is an
output's copy, read off that output's row: a gate of the unit, committed in `C`. -/
structure UnitPair {C CB : Audit.Circuit} {n : ℕ} (P : Partition C n) (PB : Partition CB n) (m : Fin CB.N → Fin C.N)
    (u : Fin n) where
  types : String → Option CircuitType
  ls : Array Layout
  info : String → Option TableInfo
  words : String → Option (Array ℕ)
  idx : ℕ
  done : Array D
  hd : deriveChecked types ls info words idx = .ok done
  t : CircuitType
  ht : (ls.getD idx default).type.bind types = some t
  ord : Ordered (ofBlock words done (done.getD idx default))
  rows : P.IsRowsUnit u (Rows.compose (ofBlock words done (done.getD idx default)) ord)
  loOf : ℕ → ℕ
  F : ℕ
  hF : idx ≤ F
  gates : PB.IsGatesUnit u (unitGates types words loOf F t) (t.inBits + 1)
  ins : ∀ c : Fin (Rows.compose (ofBlock words done (done.getD idx default)) ord).w, c.val < t.inBits →
    m (gates.wire c.val) = rows.wire c
  one : m (gates.wire t.inBits) = rows.wire (Rows.compose (ofBlock words done (done.getD idx default)) ord).one
  out : ∀ g ∈ PB.committed, PB.unit g = some u → ∃ o < t.outputs.size,
    g = gates.wire ((flatten types words loOf F t).1.size + o) ∧
    ∃ ℓ : Fin (Rows.compose (ofBlock words done (done.getD idx default)) ord).w,
      ℓ.val = lidx (ofBlock words done (done.getD idx default)) ((done.getD idx default).outCols.getD o 0) ∧
      m g = rows.wire ℓ ∧ rows.wire ℓ ∈ P.committed ∧ P.unit (rows.wire ℓ) = some u

namespace UnitPair

variable {C CB : Audit.Circuit} {n : ℕ} {P : Partition C n} {PB : Partition CB n} {m : Fin CB.N → Fin C.N} {u : Fin n}
  (U : UnitPair P PB m u)

/-- The audit's gate the unit's constant reads. -/
abbrev oneGate : Fin C.N := U.rows.wire (Rows.compose (ofBlock U.words U.done (U.done.getD U.idx default)) U.ord).one

/-- **Correct on its rows, correct on its gates.** At values carrying 1 on the unit's constant, a unit correct on the
audit's circuit is correct on the Boolean circuit, its values read off the audit's through `m`. -/
theorem correct (vals : Fin C.N → Bool) (h1 : vals U.oneGate = true) (hc : P.Correct vals u) :
    PB.Correct (vals ∘ m) u := by
  intro g hg hgu
  obtain ⟨o, ho, rfl, ℓ, hℓ, hm, hcom, hℓu⟩ := U.out g hg hgu
  have hb := U.gates.localEval_type (vals ∘ m) (by rw [Function.comp_apply, U.one]; exact h1) ho
  have hr := U.rows.localEval_compose U.hd U.ht vals h1 (U.F + 1) (by have := U.hF; omega) ho ℓ hℓ
  rw [hb, Function.comp_apply, hm, hc _ hcom hℓu, hr]
  congr 1
  refine evalT_congr _ _ _ _ fun i hi => ?_
  have hiw : i < (Rows.compose (ofBlock U.words U.done (U.done.getD U.idx default)) U.ord).w := by
    have hin := (checkLayout_spec (deriveChecked_all U.hd U.idx (lt_of_type U.ht)) U.ht).2.2.1
    have hle : (U.done.getD U.idx default).inCols.size ≤ (order U.done (U.done.getD U.idx default)).length := U.ord.le
    rw [Rows.compose_w]
    show i < (order U.done (U.done.getD U.idx default)).length + 1
    omega
  rw [dite_eq_left hiw, Function.comp_apply, U.ins ⟨i, hiw⟩ hi]

end UnitPair

/-- **`RowsL1` from one pair per unit.** At values carrying 1 on `ones`, which holds each unit's constant, every unit wrong
on the Boolean circuit `CB` (its values read off the audit's through `m`) is wrong on its rows. -/
theorem rowsL1_of_pairs {C CB : Audit.Circuit} {n : ℕ} (P : Partition C n) (PB : Partition CB n) (m : Fin CB.N → Fin C.N)
    (ones : Finset (Fin C.N)) (U : ∀ u, UnitPair P PB m u) (hones : ∀ u, (U u).oneGate ∈ ones) :
    P.RowsL1 PB (fun vals => vals ∘ m) ones := by
  intro vals hv u hu
  rw [Partition.mem_wrong] at hu ⊢
  exact fun hc => hu ((U u).correct vals (hv _ (hones u)) hc)

end FlockSoundness
