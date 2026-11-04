import Proofs.Flock.Bridge.BitsLower
import Proofs.Flock.Bridge.BitsRows
import Proofs.Flock.Bridge.BitsKeys
import Proofs.Flock.Bridge.Statement

/-!
# Bits' key program lays out the bridge's (`SampledProofs.Bits.keyProg`)

`Placed.keyProg c hU n`, with outputs `Layout.outsOf c hU n`, is the bridge's program of a circuit file's `n` units, and
`SampledProofs.Bits.keyProg c n` is Bits'. Bits' lays out the bridge's:

* `uniform_op_input`, `uniform_op_row`, `uniform_unit_row`, `rowsWireC_uniform_comp`: `Placed.uniform`'s gates in
  closed form. Its inputs come first, then unit `u`'s computed row `i` at gate `gates k R u + i`, reading its columns'
  gates (`colVal`). `keyProg_op_input`, `keyProg_op_row`, `keyProg_unit_input`, `keyProg_unit_row` and
  `rowsWireC_keyProg_comp` state them at the circuit's program.
* `laysOut_keyProg`, `unitsOut_keyProg`: `LaysOut` and `UnitsOut` at a circuit file's program. Each row reads its
  columns' gates (`keyWire_colVal`, through `keyIdx_eq`), and the outputs are the gates on returned outputs' columns,
  ascending (`outs_keyProg`).
* `progOf_keyProg`, `blockOf_keyProg`: so at the circuit's constants (`constsOf`, Bits' `keyConsts`) the bridge's
  program and proof blocks are Bits'.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

/-! ## `uniform`'s gates -/

section uniform

variable {k : ℕ} {R : Rows} {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one}

theorem gates_val (k : ℕ) (R : Rows) : ∀ n, Placed.gates k R n = k + n * R.nComp
  | 0 => by simp [Placed.gates]
  | n + 1 => by simp only [Placed.gates, gates_val k R n]; ring

theorem gates_mono (k : ℕ) (R : Rows) {u n : ℕ} (h : u ≤ n) : Placed.gates k R u ≤ Placed.gates k R n := by
  rw [gates_val, gates_val]
  exact Nat.add_le_add_left (Nat.mul_le_mul_right _ h) _

theorem uniform_succ (n : ℕ) : Placed.uniform k R ins one hone (n + 1) =
    Prog.snoc (Placed.uniform k R ins one hone n) R (fun c => Fin.castLE (Placed.le_gates k R n) (ins n c))
      (Fin.castLE (Placed.le_gates k R n) one) (fun c h => hone n c (Fin.castLE_injective _ h)) := rfl

variable (k R ins one) in
/-- **Unit `u`'s column `c`'s gate** in `uniform`: input `c`'s gate `ins u c`, computed row `c − nIn`'s gate after the
units before it, or the constant's `one`. -/
def colVal (u : ℕ) (c : Fin R.w) : ℕ :=
  if h : c.val < R.nIn then (ins u ⟨c.val, h⟩).val
  else if c.val < R.nIn + R.nComp then Placed.gates k R u + (c.val - R.nIn) else one.val

/-- `uniform`'s program inputs are input gates. -/
theorem uniform_op_input (n : ℕ) (g : Fin (Placed.gates k R n)) (hg : g.val < k) :
    (Placed.uniform k R ins one hone n).op g = .input := by
  have h := ((Placed.uniform k R ins one hone n).input_iff g).1 (Layout.unit_uniform_input n g hg)
  revert h
  cases (Placed.uniform k R ins one hone n).op g <;> simp [Audit.Op.isInput]

private theorem lt_gates_of_lt {n u : ℕ} (hun : u < n) (i : Fin R.nComp) :
    Placed.gates k R u + i.val < Placed.gates k R n := by
  have h1 := gates_mono k R (show u + 1 ≤ n from hun)
  have h2 : Placed.gates k R (u + 1) = Placed.gates k R u + R.nComp := rfl
  have := i.isLt
  omega

/-- **Unit `u`'s computed row `i` is gate `gates k R u + i`** of `uniform`, a row reading its columns' gates. -/
theorem uniform_op_row : ∀ (n u : ℕ), u < n → ∀ (i : Fin R.nComp) (g : Fin (Placed.gates k R n)),
    g.val = Placed.gates k R u + i.val → ∃ A B, (Placed.uniform k R ins one hone n).op g = .row A B ∧
      A.map Fin.val = (R.a i).map (colVal k R ins one u) ∧ B.map Fin.val = (R.b i).map (colVal k R ins one u)
  | 0, _, hu, _, _, _ => absurd hu (Nat.not_lt_zero _)
  | n + 1, u, hu, i, g, hg => by
    rw [uniform_succ]
    by_cases hun : u < n
    · have hlt : g.val < Placed.gates k R n := hg ▸ lt_gates_of_lt hun i
      obtain ⟨A, B, hop, hA, hB⟩ := uniform_op_row n u hun i ⟨g.val, hlt⟩ hg
      refine ⟨A.map (Fin.castAdd R.nComp), B.map (Fin.castAdd R.nComp), ?_, ?_, ?_⟩
      · rw [show g = Fin.castAdd R.nComp ⟨g.val, hlt⟩ from Fin.ext rfl, Prog.op_castAdd, hop]
        rfl
      · rw [List.map_map]
        exact hA
      · rw [List.map_map]
        exact hB
    · have hun' : u = n := by omega
      rw [hun'] at hg ⊢
      refine ⟨(R.a i).map (Prog.colWire R (fun c => Fin.castLE (Placed.le_gates k R n) (ins n c))
          (Fin.castLE (Placed.le_gates k R n) one)),
        (R.b i).map (Prog.colWire R (fun c => Fin.castLE (Placed.le_gates k R n) (ins n c))
          (Fin.castLE (Placed.le_gates k R n) one)), ?_, ?_, ?_⟩
      · rw [show g = ⟨Placed.gates k R n + i.val, by have := i.isLt; omega⟩ from Fin.ext hg]
        exact Prog.op_new _ R _ _ _ i
      · rw [List.map_map]
        refine List.map_congr_left fun c _ => ?_
        simp only [Function.comp_apply, Prog.colWire_val, colVal, Fin.coe_castLE]
      · rw [List.map_map]
        refine List.map_congr_left fun c _ => ?_
        simp only [Function.comp_apply, Prog.colWire_val, colVal, Fin.coe_castLE]

/-- **Gate `gates k R u + i` is unit `u`'s** in `uniform`. -/
theorem uniform_unit_row : ∀ (n u : ℕ) (hu : u < n) (i : Fin R.nComp) (g : Fin (Placed.gates k R n)),
    g.val = Placed.gates k R u + i.val → (Placed.uniform k R ins one hone n).unit g = some ⟨u, hu⟩
  | 0, _, hu, _, _, _ => absurd hu (Nat.not_lt_zero _)
  | n + 1, u, hu, i, g, hg => by
    rw [uniform_succ]
    by_cases hun : u < n
    · have hlt : g.val < Placed.gates k R n := hg ▸ lt_gates_of_lt hun i
      rw [show g = Fin.castAdd R.nComp ⟨g.val, hlt⟩ from Fin.ext rfl, Prog.unit_castAdd,
        uniform_unit_row n u hun i ⟨g.val, hlt⟩ hg]
      rfl
    · have hun' : u = n := by omega
      rw [show g = ⟨Placed.gates k R n + i.val, by have := i.isLt; omega⟩ from Fin.ext (by rw [hg, hun']),
        Prog.unit_new]
      exact congrArg some (Fin.ext hun'.symm)

/-- **Unit `u`'s computed column `x` is on gate `gates k R u + (x − nIn)`** in `uniform`'s own wiring. -/
theorem rowsWireC_uniform_comp : ∀ (n : ℕ) (u : Fin n) (x : Fin ((Placed.uniform k R ins one hone n).rows u).w),
    R.nIn ≤ x.val → x.val < R.nIn + R.nComp →
    ((Const.Prog.rowsWireC (Placed.uniform k R ins one hone n) u).wire x).val =
      Placed.gates k R u.val + (x.val - R.nIn)
  | 0, u, _, _, _ => u.elim0
  | n + 1, u, x, h1, h2 => by
    induction u using Fin.lastCases with
    | last =>
      change ((Const.Prog.rowsWireC (Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _) (Fin.last n)).wire
        x).val = _
      rw [Const.Prog.rowsWireC_last, Const.Prog.castW_wire, Prog.colWire_val]
      simp only [Fin.coe_cast, Fin.val_last]
      split_ifs <;> first | omega | rfl
    | cast v =>
      change ((Const.Prog.rowsWireC (Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _) v.castSucc).wire
        x).val = _
      rw [Const.Prog.rowsWireC_castSucc, Const.Prog.castW_wire]
      exact rowsWireC_uniform_comp n v _ h1 h2

end uniform

/-! ## Bits' key program -/

section keyProg

private theorem length_flatMap_range_map {α : Type} (F : ℕ → ℕ → α) (n m : ℕ) :
    ((List.range n).flatMap fun u => (List.range m).map (F u)).length = n * m := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ, List.flatMap_append, ih, Nat.add_mul]

private theorem getElem?_flatMap_range_map {α : Type} (F : ℕ → ℕ → α) {n m u i : ℕ} (hu : u < n) (hi : i < m) :
    ((List.range n).flatMap fun u => (List.range m).map (F u))[u * m + i]? = some (F u i) := by
  induction n with
  | zero => exact absurd hu (Nat.not_lt_zero _)
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append]
    by_cases hun : u < n
    · have h := Nat.mul_le_mul_right m (show u + 1 ≤ n from hun)
      rw [Nat.add_mul, Nat.one_mul] at h
      rw [List.getElem?_append_left (by rw [length_flatMap_range_map]; omega)]
      exact ih hun
    · have hun' : u = n := by omega
      subst hun'
      rw [List.getElem?_append_right (by rw [length_flatMap_range_map]; omega)]
      simp [length_flatMap_range_map, hi]

/-- `g` past `k` inputs and below `n` units of `m` gates is unit `u`'s gate `i`. -/
private theorem decomp {k m n g : ℕ} (h1 : k ≤ g) (h2 : g < k + n * m) :
    ∃ u i, u < n ∧ i < m ∧ g = k + (u * m + i) := by
  have hm : 0 < m := Nat.pos_of_ne_zero fun h => by simp [h] at h2; omega
  refine ⟨(g - k) / m, (g - k) % m, (Nat.div_lt_iff_lt_mul hm).2 (by omega), Nat.mod_lt _ hm, ?_⟩
  have := Nat.div_add_mod (g - k) m
  rw [Nat.mul_comm] at this
  omega

variable (c : Flock.Circuit) (n : ℕ)

/-- Bits' stacked unit rows of the circuit file `c`. -/
def bitsRows : SampledProofs.Bits.Rows := (SampledProofs.Bits.Rows.ofNet c.unit).stack c.upv

/-- **Unit `u`'s column `col`'s gate** in Bits' key program: its key's source gate (the zero's where it has none), its
own computed row's gate, or the constant's. -/
def keyWire (u col : ℕ) : ℕ :=
  if col < (bitsRows c).nIn then
    (match ((Array.range (bitsRows c).nIn).map (SampledProofs.Bits.keyIdx c (bitsRows c).nIn)).getD col none with
      | some b => 2 + u * (bitsRows c).nIn + b
      | none => 1)
  else if col < (bitsRows c).nIn + (bitsRows c).nComp then
    2 + n * (bitsRows c).nIn + u * (bitsRows c).nComp + (col - (bitsRows c).nIn)
  else 0

theorem keyProg_inputs : (SampledProofs.Bits.keyProg c n).inputs = 2 + n * (bitsRows c).nIn := rfl

theorem keyProg_units : (SampledProofs.Bits.keyProg c n).units = n := rfl

theorem keyProg_perUnit : (SampledProofs.Bits.keyProg c n).perUnit = (bitsRows c).nComp := rfl

theorem keyProg_rows : (SampledProofs.Bits.keyProg c n).rows = ((List.range n).flatMap fun u =>
    (List.range (bitsRows c).nComp).map fun i =>
      (((bitsRows c).a.getD i []).map (keyWire c n u), ((bitsRows c).b.getD i []).map (keyWire c n u))).toArray :=
  rfl

theorem keyProg_outs : (SampledProofs.Bits.keyProg c n).outs =
    (((List.range n).flatMap fun u => (List.range c.upv).flatMap fun k =>
      (List.range (c.leavesOut.getD k #[]).size).flatMap fun j =>
        (List.range 16).filterMap (SampledProofs.Bits.outGate c n u k j)).mergeSort
      fun x y => decide (x ≤ y)).eraseDups := rfl

theorem keyProg_size :
    (SampledProofs.Bits.keyProg c n).size = 2 + n * (bitsRows c).nIn + n * (bitsRows c).nComp := by
  simp only [SampledProofs.Bits.KeyProg.size, keyProg_inputs, keyProg_rows, List.size_toArray,
    length_flatMap_range_map]

theorem keyProg_row?_input {g : ℕ} (hg : g < 2 + n * (bitsRows c).nIn) :
    (SampledProofs.Bits.keyProg c n).row? g = none := by
  simp [SampledProofs.Bits.KeyProg.row?, keyProg_inputs, hg]

theorem keyProg_row?_row {u i : ℕ} (hu : u < n) (hi : i < (bitsRows c).nComp) :
    (SampledProofs.Bits.keyProg c n).row? (2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + i)) =
      some (((bitsRows c).a.getD i []).map (keyWire c n u), ((bitsRows c).b.getD i []).map (keyWire c n u)) := by
  unfold SampledProofs.Bits.KeyProg.row?
  simp only [keyProg_inputs, show ¬ 2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + i) <
    2 + n * (bitsRows c).nIn by omega, ↓reduceIte, Nat.add_sub_cancel_left, keyProg_rows, List.getElem?_toArray]
  exact getElem?_flatMap_range_map _ hu hi

variable {c n} {hU : Built c.unit}

theorem keyN_eq (hR : RowsAgree (bitsRows c) (Placed.circuitRows c hU)) :
    Layout.keyN c hU n = 2 + n * (bitsRows c).nIn + n * (bitsRows c).nComp := by
  show Placed.gates (2 + n * (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n = _
  rw [gates_val, hR.1, hR.2.1]

/-- **Unit `u`'s computed column `x` is on gate `2 + n·B + u·nComp + (x − B)`** of the circuit's program. -/
theorem rowsWireC_keyProg_comp (u : Fin n) (x : Fin ((Placed.keyProg c hU n).rows u).w)
    (h1 : (Placed.circuitRows c hU).nIn ≤ x.val)
    (h2 : x.val < (Placed.circuitRows c hU).nIn + (Placed.circuitRows c hU).nComp) :
    ((Const.Prog.rowsWireC (Placed.keyProg c hU n) u).wire x).val = 2 + n * (Placed.circuitRows c hU).nIn +
      u.val * (Placed.circuitRows c hU).nComp + (x.val - (Placed.circuitRows c hU).nIn) := by
  have h := rowsWireC_uniform_comp (k := Placed.inGates n (Placed.circuitRows c hU).nIn)
    (R := Placed.circuitRows c hU) (ins := Placed.insOf n _ (Aliased.keyIdx c (Placed.circuitRows c hU).nIn))
    (one := Placed.oneGate n _) (hone := Placed.insOf_ne_one n _ _) n u x h1 h2
  rw [gates_val] at h
  exact h

theorem keyProg_op_input (g : Fin (Layout.keyN c hU n)) (hg : g.val < 2 + n * (Placed.circuitRows c hU).nIn) :
    (Placed.keyProg c hU n).op g = .input :=
  uniform_op_input (k := Placed.inGates n (Placed.circuitRows c hU).nIn) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf n _ (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)) (one := Placed.oneGate n _)
    (hone := Placed.insOf_ne_one n _ _) n g hg

theorem keyProg_unit_input (g : Fin (Layout.keyN c hU n)) (hg : g.val < 2 + n * (Placed.circuitRows c hU).nIn) :
    (Placed.keyProg c hU n).unit g = none :=
  Layout.unit_uniform_input (k := Placed.inGates n (Placed.circuitRows c hU).nIn) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf n _ (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)) (one := Placed.oneGate n _)
    (hone := Placed.insOf_ne_one n _ _) n g hg

theorem keyProg_op_row {u : ℕ} (hu : u < n) (i : Fin (Placed.circuitRows c hU).nComp) (g : Fin (Layout.keyN c hU n))
    (hg : g.val = 2 + n * (Placed.circuitRows c hU).nIn + u * (Placed.circuitRows c hU).nComp + i.val) :
    ∃ A B, (Placed.keyProg c hU n).op g = .row A B ∧
      A.map Fin.val = ((Placed.circuitRows c hU).a i).map (colVal (Placed.inGates n (Placed.circuitRows c hU).nIn)
        (Placed.circuitRows c hU) (Placed.insOf n _ (Aliased.keyIdx c (Placed.circuitRows c hU).nIn))
        (Placed.oneGate n _) u) ∧
      B.map Fin.val = ((Placed.circuitRows c hU).b i).map (colVal (Placed.inGates n (Placed.circuitRows c hU).nIn)
        (Placed.circuitRows c hU) (Placed.insOf n _ (Aliased.keyIdx c (Placed.circuitRows c hU).nIn))
        (Placed.oneGate n _) u) :=
  uniform_op_row (k := Placed.inGates n (Placed.circuitRows c hU).nIn) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf n _ (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)) (one := Placed.oneGate n _)
    (hone := Placed.insOf_ne_one n _ _) n u hu i g (by rw [gates_val]; exact hg)

theorem keyProg_unit_row {u : ℕ} (hu : u < n) (i : Fin (Placed.circuitRows c hU).nComp) (g : Fin (Layout.keyN c hU n))
    (hg : g.val = 2 + n * (Placed.circuitRows c hU).nIn + u * (Placed.circuitRows c hU).nComp + i.val) :
    (Placed.keyProg c hU n).unit g = some ⟨u, hu⟩ :=
  uniform_unit_row (k := Placed.inGates n (Placed.circuitRows c hU).nIn) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf n _ (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)) (one := Placed.oneGate n _)
    (hone := Placed.insOf_ne_one n _ _) n u hu i g (by rw [gates_val]; exact hg)

/-- **Bits' gate of unit `u`'s column `col` is the bridge's.** -/
theorem keyWire_colVal (hR : RowsAgree (bitsRows c) (Placed.circuitRows c hU)) {u : ℕ} (hu : u < n)
    (col : Fin (Placed.circuitRows c hU).w) :
    keyWire c n u col.val = colVal (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU)
      (Placed.insOf n _ (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)) (Placed.oneGate n _) u col := by
  unfold keyWire colVal
  rw [hR.1, hR.2.1]
  by_cases h1 : col.val < (Placed.circuitRows c hU).nIn
  · simp only [h1, ↓reduceIte, ↓reduceDIte, getD_map_range _ h1, keyIdx_eq c h1,
      Placed.insOf_of_lt hu]
    cases Aliased.keyIdx c (Placed.circuitRows c hU).nIn ⟨col.val, h1⟩ <;> rfl
  · by_cases h2 : col.val < (Placed.circuitRows c hU).nIn + (Placed.circuitRows c hU).nComp
    · simp only [h1, h2, ↓reduceIte, ↓reduceDIte, gates_val, Placed.inGates]
    · simp only [h1, h2, ↓reduceIte, ↓reduceDIte]
      rfl

/-- **Bits' key program lays out the bridge's circuit**, without its outputs. -/
theorem keyProg_laysOut_gates (hu : c.unit.unitLog ≤ 32) :
    (SampledProofs.Bits.keyProg c n).size = Layout.keyN c hU n ∧
    (∀ g : Fin (Layout.keyN c hU n), (SampledProofs.Bits.keyProg c n).row? g.val = none →
      (Placed.keyProg c hU n).op g = .input) ∧
    (∀ g : Fin (Layout.keyN c hU n), ∀ a b, (SampledProofs.Bits.keyProg c n).row? g.val = some (a, b) →
      ∃ A B, (Placed.keyProg c hU n).op g = .row A B ∧ a = A.map Fin.val ∧ b = B.map Fin.val) := by
  have hR : RowsAgree (bitsRows c) (Placed.circuitRows c hU) := rowsAgree_circuitRows c hU hu
  have hN := keyN_eq (n := n) hR
  have hrow : ∀ g : Fin (Layout.keyN c hU n), 2 + n * (bitsRows c).nIn ≤ g.val →
      ∃ u i, u < n ∧ i < (bitsRows c).nComp ∧ g.val = 2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + i) :=
    fun g h => decomp h (hN ▸ g.isLt)
  refine ⟨(keyProg_size c n).trans hN.symm, fun g hg => ?_, fun g a b hg => ?_⟩
  · have hlt : g.val < 2 + n * (bitsRows c).nIn := by
      by_contra hge
      obtain ⟨u, i, hu, hi, he⟩ := hrow g (by omega)
      rw [he, keyProg_row?_row c n hu hi] at hg
      exact Option.some_ne_none _ hg
    exact keyProg_op_input g (by rw [← hR.1]; exact hlt)
  · have hge : 2 + n * (bitsRows c).nIn ≤ g.val := by
      by_contra hlt
      rw [keyProg_row?_input c n (by omega)] at hg
      exact Option.noConfusion hg
    obtain ⟨u, i, hu, hi, he⟩ := hrow g hge
    rw [he, keyProg_row?_row c n hu hi, Option.some.injEq, Prod.mk.injEq] at hg
    obtain ⟨rfl, rfl⟩ := hg
    have hi' : i < (Placed.circuitRows c hU).nComp := hR.2.1 ▸ hi
    obtain ⟨A, B, hop, hA, hB⟩ := keyProg_op_row hu ⟨i, hi'⟩ g (by rw [he]; dsimp only; rw [← hR.1, ← hR.2.1]; omega)
    refine ⟨A, B, hop, ?_, ?_⟩
    · rw [hA, (hR.2.2 ⟨i, hi'⟩).1, List.map_map]
      exact List.map_congr_left fun col _ => keyWire_colVal hR hu col
    · rw [hB, (hR.2.2 ⟨i, hi'⟩).2, List.map_map]
      exact List.map_congr_left fun col _ => keyWire_colVal hR hu col

/-! ## Outputs -/

private theorem pairwise_lt_eraseDups : ∀ (m : ℕ) (l : List ℕ), l.length ≤ m → l.Pairwise (· ≤ ·) →
    l.eraseDups.Pairwise (· < ·)
  | _, [], _, _ => by simp
  | 0, _ :: _, hm, _ => absurd hm (by simp)
  | m + 1, a :: l, hm, hp => by
    rw [List.eraseDups_cons]
    rw [List.pairwise_cons] at hp
    refine List.pairwise_cons.2 ⟨fun b hb => ?_, pairwise_lt_eraseDups m _ ?_ (hp.2.filter _)⟩
    · rw [List.mem_eraseDups, List.mem_filter] at hb
      have h1 := hp.1 b hb.1
      have h2 : b ≠ a := by simpa using hb.2
      omega
    · exact (List.length_filter_le _ _).trans (by simpa using hm)

variable (c n) in
private theorem outGate_eq_some {u k j t a : ℕ} : SampledProofs.Bits.outGate c n u k j t = some a ↔
    u < n ∧ k < c.upv ∧ j < (c.leavesOut.getD k #[]).size ∧ t < 16 ∧
      (SampledProofs.Bits.Rows.ofNet c.unit).nIn ≤ SampledProofs.Bits.outCol c j t ∧
      SampledProofs.Bits.outCol c j t <
        (SampledProofs.Bits.Rows.ofNet c.unit).nIn + (SampledProofs.Bits.Rows.ofNet c.unit).nComp ∧
      2 + n * (bitsRows c).nIn + u * (bitsRows c).nComp + (k * ((SampledProofs.Bits.Rows.ofNet c.unit).nComp + 1) +
        (1 + (SampledProofs.Bits.outCol c j t - (SampledProofs.Bits.Rows.ofNet c.unit).nIn))) = a := by
  unfold SampledProofs.Bits.outGate SampledProofs.Bits.rowGate bitsRows
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  split_ifs with h
  · simp only [Option.some.injEq, and_assoc] at h ⊢
    exact ⟨fun he => ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2, he⟩, fun h' => h'.2.2.2.2.2.2⟩
  · simp only [and_assoc] at h
    exact ⟨fun he => Option.noConfusion he,
      fun h' => absurd ⟨h'.1, h'.2.1, h'.2.2.1, h'.2.2.2.1, h'.2.2.2.2.1, h'.2.2.2.2.2.1⟩ h⟩

/-- **Bits' gate on bit `t` of unit `u`'s instance `k`'s returned output leaf `j` is a bridge output.** -/
private theorem outGate_mem (hu : c.unit.unitLog ≤ 32) {u k j t a : ℕ}
    (h : SampledProofs.Bits.outGate c n u k j t = some a) :
    ∃ g : Fin (Layout.keyN c hU n), g ∈ Layout.outsOf c hU n ∧ g.val = a := by
  have hR : RowsAgree (bitsRows c) (Placed.circuitRows c hU) := rowsAgree_circuitRows c hU hu
  obtain ⟨hun, hk, hj, ht, h1, h2, ha⟩ := (outGate_eq_some c n).1 h
  have bI : (bitsRows c).nIn = c.upv * (SampledProofs.Bits.Rows.ofNet c.unit).nIn := rfl
  have bM : (bitsRows c).nComp = c.upv * ((SampledProofs.Bits.Rows.ofNet c.unit).nComp + 1) := rfl
  have hkM := Nat.mul_le_mul_right ((SampledProofs.Bits.Rows.ofNet c.unit).nComp + 1) (show k + 1 ≤ c.upv from hk)
  have huM := Nat.mul_le_mul_right (bitsRows c).nComp (show u + 1 ≤ n from hun)
  rw [Nat.succ_mul] at hkM huM
  have hN := keyN_eq (n := n) hR
  obtain ⟨xv, hxv⟩ : ∃ xv, xv = c.upv * (SampledProofs.Bits.Rows.ofNet c.unit).nIn +
      (k * ((SampledProofs.Bits.Rows.ofNet c.unit).nComp + 1) +
        (1 + (SampledProofs.Bits.outCol c j t - (SampledProofs.Bits.Rows.ofNet c.unit).nIn))) := ⟨_, rfl⟩
  have hx1 : (Placed.circuitRows c hU).nIn ≤ xv := by rw [← hR.1]; omega
  have hx2 : xv < (Placed.circuitRows c hU).nIn + (Placed.circuitRows c hU).nComp := by
    rw [← hR.1, ← hR.2.1]; omega
  have hw : ((Placed.keyProg c hU n).rows ⟨u, hun⟩).w = (Placed.circuitRows c hU).w :=
    congrArg Rows.w (Placed.rows_circuitProg c hU n _ _ ⟨u, hun⟩)
  have hxw : xv < ((Placed.keyProg c hU n).rows ⟨u, hun⟩).w := by rw [hw]; simp only [Rows.w]; omega
  refine ⟨⟨a, by omega⟩, ?_, rfl⟩
  unfold Layout.outsOf
  rw [Finset.mem_filter]
  refine ⟨Finset.mem_univ _, ⟨u, hun⟩, ⟨xv, hxw⟩, Fin.ext ?_, k, j, t, hk, hj, ht, h1, h2, hxv⟩
  rw [rowsWireC_keyProg_comp ⟨u, hun⟩ ⟨xv, hxw⟩ hx1 hx2]
  dsimp only
  rw [← hR.1, ← hR.2.1]
  omega

/-- **A bridge output is Bits' gate on some bit of some unit's instance's returned output leaf.** -/
private theorem outGate_of_mem (hu : c.unit.unitLog ≤ 32) {g : Fin (Layout.keyN c hU n)}
    (hg : g ∈ Layout.outsOf c hU n) : ∃ u < n, ∃ k < c.upv, ∃ j < (c.leavesOut.getD k #[]).size, ∃ t < 16,
      SampledProofs.Bits.outGate c n u k j t = some g.val := by
  have hR : RowsAgree (bitsRows c) (Placed.circuitRows c hU) := rowsAgree_circuitRows c hU hu
  unfold Layout.outsOf at hg
  rw [Finset.mem_filter] at hg
  obtain ⟨-, u, x, hx, k, j, t, hk, hj, ht, h1, h2, hxe⟩ := hg
  have h1' : (SampledProofs.Bits.Rows.ofNet c.unit).nIn ≤ SampledProofs.Bits.outCol c j t := h1
  have h2' : SampledProofs.Bits.outCol c j t <
      (SampledProofs.Bits.Rows.ofNet c.unit).nIn + (SampledProofs.Bits.Rows.ofNet c.unit).nComp := h2
  have hxe' : x.val = c.upv * (SampledProofs.Bits.Rows.ofNet c.unit).nIn +
      (k * ((SampledProofs.Bits.Rows.ofNet c.unit).nComp + 1) +
        (1 + (SampledProofs.Bits.outCol c j t - (SampledProofs.Bits.Rows.ofNet c.unit).nIn))) := hxe
  have bI : (bitsRows c).nIn = c.upv * (SampledProofs.Bits.Rows.ofNet c.unit).nIn := rfl
  have bM : (bitsRows c).nComp = c.upv * ((SampledProofs.Bits.Rows.ofNet c.unit).nComp + 1) := rfl
  have hkM := Nat.mul_le_mul_right ((SampledProofs.Bits.Rows.ofNet c.unit).nComp + 1) (show k + 1 ≤ c.upv from hk)
  rw [Nat.succ_mul] at hkM
  have hx1 : (Placed.circuitRows c hU).nIn ≤ x.val := by rw [← hR.1]; omega
  have hx2 : x.val < (Placed.circuitRows c hU).nIn + (Placed.circuitRows c hU).nComp := by
    rw [← hR.1, ← hR.2.1]; omega
  refine ⟨u.val, u.isLt, k, hk, j, hj, t, ht, (outGate_eq_some c n).2 ⟨u.isLt, hk, hj, ht, h1', h2', ?_⟩⟩
  rw [← hx, rowsWireC_keyProg_comp u x hx1 hx2, ← hR.1, ← hR.2.1]
  omega

/-- **Bits' key program's outputs are the bridge's**, ascending. -/
theorem outs_keyProg (hu : c.unit.unitLog ≤ 32) : (SampledProofs.Bits.keyProg c n).outs =
    ((List.finRange (Layout.keyN c hU n)).filter fun g => decide (g ∈ Layout.outsOf c hU n)).map Fin.val := by
  refine List.Pairwise.eq_of_mem_iff (r := (· < ·)) ?_ ?_ fun a => ?_
  · rw [keyProg_outs]
    exact pairwise_lt_eraseDups _ _ le_rfl ((List.pairwise_mergeSort (fun a b c h1 h2 => by simp at *; omega)
      (fun a b => by simp; omega) _).imp (by simp))
  · exact List.pairwise_map.2 ((List.pairwise_lt_finRange _).filter _)
  · rw [keyProg_outs, List.mem_eraseDups, List.mem_mergeSort]
    simp only [List.mem_flatMap, List.mem_range, List.mem_filterMap, List.mem_map, List.mem_filter,
      List.mem_finRange, true_and, decide_eq_true_eq]
    constructor
    · rintro ⟨u, -, k, -, j, -, t, -, h⟩
      exact outGate_mem hu h
    · rintro ⟨g, hg, rfl⟩
      obtain ⟨u, hun, k, hk, j, hj, t, ht, h⟩ := outGate_of_mem hu hg
      exact ⟨u, hun, k, hk, j, hj, t, ht, h⟩

/-- **Bits' key program lays out the bridge's circuit.** -/
theorem laysOut_keyProg (hu : c.unit.unitLog ≤ 32) :
    LaysOut (SampledProofs.Bits.keyProg c n) ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)) :=
  have h := keyProg_laysOut_gates (n := n) hu
  ⟨h.1, h.2.1, h.2.2, outs_keyProg hu⟩

/-! ## Units -/

/-- **Bits' key program's units are the bridge's**, gate by gate. -/
theorem unitsOut_keyProg (hu : c.unit.unitLog ≤ 32) :
    UnitsOut (SampledProofs.Bits.keyProg c n) ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) := by
  have hR : RowsAgree (bitsRows c) (Placed.circuitRows c hU) := rowsAgree_circuitRows c hU hu
  have hN := keyN_eq (n := n) hR
  refine ⟨rfl, fun (g : Fin (Layout.keyN c hU n)) => ?_⟩
  change SampledProofs.Bits.unitOf (SampledProofs.Bits.keyProg c n) g.val =
    ((Placed.keyProg c hU n).unit g).map Fin.val
  have hg := g.isLt
  unfold SampledProofs.Bits.unitOf
  rw [keyProg_size, keyProg_inputs, keyProg_perUnit]
  by_cases hlt : g.val < 2 + n * (bitsRows c).nIn
  · rw [keyProg_unit_input g (by rw [← hR.1]; exact hlt)]
    simp [hlt]
  · obtain ⟨u, i, hun, hi, he⟩ := decomp (k := 2 + n * (bitsRows c).nIn) (by omega)
      (show g.val < 2 + n * (bitsRows c).nIn + n * (bitsRows c).nComp by omega)
    have hi' : i < (Placed.circuitRows c hU).nComp := hR.2.1 ▸ hi
    rw [keyProg_unit_row hun ⟨i, hi'⟩ g (by rw [he]; dsimp only; rw [← hR.1, ← hR.2.1]; omega)]
    have hc : (decide (g.val < 2 + n * (bitsRows c).nIn) ||
        decide (2 + n * (bitsRows c).nIn + n * (bitsRows c).nComp ≤ g.val) || (bitsRows c).nComp == 0) = false := by
      simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not, beq_eq_false_iff_ne, ne_eq]
      omega
    have hd : (g.val - (2 + n * (bitsRows c).nIn)) / (bitsRows c).nComp = u := by
      rw [he, Nat.add_sub_cancel_left, Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi,
        Nat.zero_add]
    simp [hc, hd]

/-! ## Constants, program and proof blocks -/

/-- **Bits' constants of the key program**: gate 0 is the constant `1`, gate 1 the zero. -/
def keyConsts (g : ℕ) : Option Bool := if g = 0 then some true else if g = 1 then some false else none

theorem constsOf_eq_keyConsts (g : Fin (Layout.keyN c hU n)) : constsOf c hU n g = keyConsts g.val := by
  have h1 : (Layout.oneG (c := c) (hU := hU) (n := n)).val = 0 := rfl
  have h0 : (Layout.zeroG (c := c) (hU := hU) (n := n)).val = 1 := rfl
  unfold constsOf keyConsts
  split_ifs <;> simp_all [Fin.ext_iff]

/-- **The bridge's program of a circuit file is Bits'**, at the circuit's constants. -/
theorem progOf_keyProg (hu : c.unit.unitLog ≤ 32) :
    progOf ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)) (constsOf c hU n) =
      Properties.SampledProofs.bitsProg (SampledProofs.Bits.keyProg c n) keyConsts :=
  (laysOut_keyProg hu).progOf_eq constsOf_eq_keyConsts

/-- **The bridge's positions of a circuit file's gates are Bits'.** -/
theorem posOf_keyProg (hu : c.unit.unitLog ≤ 32)
    (g : Fin (Layout.keyN c hU n)) : posOf ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)) g =
      SampledProofs.Bits.posOf (SampledProofs.Bits.keyProg c n) g.val :=
  (laysOut_keyProg hu).posOf_eq g

/-- **The bridge's proof blocks of a circuit file are Bits'**, at the circuit's constants. -/
theorem blockOf_keyProg (hu : c.unit.unitLog ≤ 32)
    (u : Fin n) : blockOf ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (constsOf c hU n) u =
      SampledProofs.Bits.blockOf (SampledProofs.Bits.keyProg c n) keyConsts u.val :=
  blockOf_eq (laysOut_keyProg hu) (unitsOut_keyProg hu) constsOf_eq_keyConsts u

end keyProg

end Flock.Bridge
