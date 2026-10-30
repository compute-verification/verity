import Proofs.Flock.Soundness.Types.Units
import Proofs.Flock.Soundness.Lowering

/-!
# A program of units over circuit types: its rows circuit, its Boolean circuit, and `RowsL1` between them (phase 1, S4c)

A program (`UProg`) is `k` inputs, input `one` the constant, then units in order. A unit (`UnitSpec`) is the verifier's
derivation for its layout, `deriveChecked` on a type `t`. Each input bit of a unit reads a
source (`Src`): a program input other than the constant, or an earlier unit's output. Several bits of a unit may read one
source. The statement's zero, which its forced-zero rows carry, is a program input like `one`, a constant of the
statement and no unit's gate: a zero leaf's bits, and a leaf cut's bits past its leaf, read it. From it:
- `toRows`, its rows circuit `C` (a `Lowering.Prog`): each unit's composed rows, its input columns on their sources' output
  rows, its constant on `one`;
- `CB`, its Boolean circuit: each unit's unit gates (`Types.Flatten.unitGates`), its inputs on their sources' copies, its
  `one` on `one`;
- `mapG`, reading each input, `one` and copy of `CB` off `C`: an input off the same input, a copy off its output's row.

Every unit output is an output of both circuits. `pairs` gives one `UnitPair` per unit, so `rowsL1`: at values with 1 on
`one`, every unit wrong on the Boolean circuit is wrong on its rows. It carries `E2E.lean`'s theorems, which are about the
rows circuit, to the Boolean circuit (`UProg.flock_e2e_count`, `Types/ProgramE2E.lean`).
-/

open Flock Flock.DeriveAll

namespace FlockSoundness

open Audit Types.Dag Types.Flatten Compose Audit.Partition

/-! ## `flatten`'s input gates are its first `inBits + 1` -/

theorem Types.Flatten.Gate.lift_args {N : ℕ} (w : ℕ → Fin N) (g : Gate) : (g.lift w).args = g.args.map w := by
  cases g <;> rfl

theorem Types.Flatten.Gate.isInput_lift {N : ℕ} (w : ℕ → Fin N) (g : Gate) : (g.lift w).isInput = true ↔ g = .input := by
  cases g <;> simp [Gate.lift, Op.isInput]

theorem Types.Flatten.Gate.lift_lift {N M : ℕ} (w : ℕ → Fin N) (f : Fin N → Fin M) (g : Gate) :
    (g.lift w).lift f = g.lift (f ∘ w) := by
  cases g <;> rfl

namespace Types.Flatten

/-- `gs'` is `gs` followed by gates none of which is an input. -/
def Fresh (gs gs' : Array Gate) : Prop := ∃ more : List Gate, gs' = gs ++ more.toArray ∧ ∀ g ∈ more, g ≠ .input

theorem Fresh.refl (gs : Array Gate) : Fresh gs gs := ⟨[], by simp, by simp⟩

theorem Fresh.trans {a b c : Array Gate} (h1 : Fresh a b) (h2 : Fresh b c) : Fresh a c := by
  obtain ⟨m1, rfl, h1⟩ := h1
  obtain ⟨m2, rfl, h2⟩ := h2
  exact ⟨m1 ++ m2, by simp, fun g hg => (List.mem_append.1 hg).elim (h1 g) (h2 g)⟩

theorem Fresh.push {a b : Array Gate} (h : Fresh a b) (g : Gate) (hg : g ≠ .input) : Fresh a (b.push g) := by
  obtain ⟨m, rfl, hm⟩ := h
  refine ⟨m ++ [g], by simp, fun x hx => ?_⟩
  rcases List.mem_append.1 hx with hx | hx
  · exact hm x hx
  · rw [List.mem_singleton.1 hx]; exact hg

theorem Fresh.size_le {a b : Array Gate} (h : Fresh a b) : a.size ≤ b.size := by
  obtain ⟨m, rfl, -⟩ := h
  simp

theorem Fresh.getElem_lt {a b : Array Gate} (h : Fresh a b) {j : ℕ} (hj : j < a.size) :
    b[j]'(Nat.lt_of_lt_of_le hj h.size_le) = a[j] := by
  obtain ⟨m, rfl, -⟩ := h
  exact Array.getElem_append_left hj

theorem Fresh.not_input {a b : Array Gate} (h : Fresh a b) {j : ℕ} (hj : j < b.size) (hja : a.size ≤ j) :
    b[j] ≠ .input := by
  obtain ⟨m, rfl, hm⟩ := h
  rw [Array.getElem_append_right hja, List.getElem_toArray]
  exact hm _ (List.getElem_mem _)

theorem xorChain_fresh : ∀ (rest : List Nat) (gs : Array Gate) (acc : Nat), Fresh gs (xorChain gs acc rest).1
  | [], gs, _ => Fresh.refl gs
  | g :: rest, gs, acc => ((Fresh.refl gs).push (.xor acc g) (by nofun)).trans (xorChain_fresh rest _ _)

theorem formGate_fresh (one z : Nat) (wg : Array Nat) (gs : Array Gate) (f : Form) :
    Fresh gs (formGate one z wg gs f).1 := by
  unfold formGate
  split
  · split <;> exact Fresh.refl gs
  · rename_i g rest _
    split
    · exact (xorChain_fresh rest gs g).push _ (by nofun)
    · exact xorChain_fresh rest gs g

theorem formsGates_fresh (one z : Nat) (wg : Array Nat) :
    ∀ (fs : List Form) (gs : Array Gate) (acc : Array Nat), Fresh gs (formsGates one z wg gs fs acc).1
  | [], gs, _ => Fresh.refl gs
  | f :: rest, gs, _ => (formGate_fresh one z wg gs f).trans (formsGates_fresh one z wg rest _ _)

theorem minterms_fresh (lo hi : Array Nat) (h : Nat) :
    ∀ (gs : Array Gate) (cnt i : Nat) (acc : Array Nat), Fresh gs (minterms lo hi h gs cnt i acc).1
  | gs, 0, _, _ => Fresh.refl gs
  | gs, cnt + 1, i, _ => ((Fresh.refl gs).push _ (by nofun)).trans (minterms_fresh lo hi h _ cnt (i + 1) _)

theorem decodeGates_fresh (one : Nat) :
    ∀ (n : Nat) (bs : Array Nat) (gs : Array Gate), bs.size = n → Fresh gs (decodeGates one gs bs).1 := by
  intro n
  induction n using Nat.strongRecOn with
  | ind n ih =>
  intro bs gs hn
  unfold decodeGates
  split
  · split
    · exact Fresh.refl gs
    · exact (Fresh.refl gs).push _ (by nofun)
  · rename_i hs
    have h1 := ih (bs.extract 0 (bs.size / 2)).size (by simp [Array.size_extract]; omega) (bs.extract 0 (bs.size / 2))
      gs rfl
    have h2 := ih (bs.extract (bs.size / 2) bs.size).size (by simp [Array.size_extract]; omega)
      (bs.extract (bs.size / 2) bs.size) (decodeGates one gs (bs.extract 0 (bs.size / 2))).1 rfl
    exact (h1.trans h2).trans (minterms_fresh _ _ _ _ _ _ _)

theorem xorGates_fresh (z : Nat) (gs : Array Gate) : ∀ xs : List Nat, Fresh gs (xorGates z gs xs).1
  | [] => Fresh.refl gs
  | g :: rest => xorChain_fresh rest gs g

theorem prodGates_fresh (z : Nat) (ws : Array Nat) (lo : Nat) (loM hiM : Array Nat) (j : Nat) :
    ∀ (cnt : Nat) (gs : Array Gate) (h : Nat) (acc : List Nat),
      Fresh gs (prodGates z ws lo loM hiM j gs cnt h acc).1 := by
  intro cnt
  induction cnt with
  | zero => intro gs _ _; exact Fresh.refl gs
  | succ cnt ih =>
    intro gs h acc
    unfold prodGates
    split
    · exact ih gs (h + 1) acc
    · exact ((xorChain_fresh _ gs _).push _ (by nofun)).trans (ih _ (h + 1) _)

theorem bitsGates_fresh (z : Nat) (ws : Array Nat) (lo : Nat) (loM hiM : Array Nat) :
    ∀ (cnt : Nat) (gs : Array Gate) (j : Nat) (acc : Array Nat), Fresh gs (bitsGates z ws lo loM hiM gs cnt j acc).1 := by
  intro cnt
  induction cnt with
  | zero => intro gs _ _; exact Fresh.refl gs
  | succ cnt ih =>
    intro gs j acc
    unfold bitsGates
    exact ((prodGates_fresh _ _ _ _ _ _ _ gs _ _).trans (xorGates_fresh _ _ _)).trans (ih _ (j + 1) _)

theorem readGates_fresh (one z : Nat) (wg : Array Nat) (gs : Array Gate) (idx : Array Form) (ws : Array Nat)
    (lo vb : Nat) : Fresh gs (readGates one z wg gs idx ws lo vb).1 := by
  unfold readGates
  exact (((formsGates_fresh _ _ _ _ _ _).trans (decodeGates_fresh _ _ _ _ rfl)).trans
    (decodeGates_fresh _ _ _ _ rfl)).trans (bitsGates_fresh _ _ _ _ _ _ _ _ _)

theorem flatItems_fresh (types : String → Option CircuitType) (words : String → Option (Array Nat)) (loOf : Nat → Nat)
    (one z : Nat) :
    ∀ F t (items : List Item) (gs : Array Gate) (wg : Array Nat),
      Fresh gs (flatItems types words loOf one z F t gs wg items).1 := by
  intro F
  induction F using Nat.strongRecOn with
  | ind F ihF =>
  intro t items
  induction items with
  | nil => intro gs wg; simp only [flatItems]; exact Fresh.refl gs
  | cons it rest ih =>
    intro gs wg
    cases it with
    | and a b =>
      simp only [flatItems]
      exact (((formGate_fresh _ _ _ _ a).trans (formGate_fresh _ _ _ _ b)).push _ (by nofun)).trans (ih _ _)
    | call k args =>
      cases hu : (t.calls[k]?).bind types with
      | none => simp only [flatItems, hu]; exact ih gs wg
      | some u =>
        cases F with
        | zero => simp only [flatItems, hu]; exact ih gs _
        | succ F' =>
          simp only [flatItems, hu]
          exact ((formsGates_fresh _ _ _ _ _ _).trans ((ihF F' (by omega) u u.items.toList _ _).trans
            (formsGates_fresh _ _ _ _ _ _))).trans (ih _ _)
    | read k idx =>
      cases ht : t.tables[k]? with
      | none => simp only [flatItems, ht]; exact ih gs wg
      | some e =>
        obtain ⟨sha, n, vb⟩ := e
        simp only [flatItems, ht]
        exact (readGates_fresh _ _ _ _ _ _ _ _).trans (ih _ _)
    | bad x => simp only [flatItems]; exact ih gs wg

theorem flatten_fresh (types : String → Option CircuitType) (words : String → Option (Array Nat)) (loOf : Nat → Nat)
    (F : Nat) (t : CircuitType) :
    Fresh ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits)) (flatten types words loOf F t).1 := by
  unfold flatten
  exact (flatItems_fresh _ _ _ _ _ _ _ _ _ _).trans (formsGates_fresh _ _ _ _ _ _)

variable (types : String → Option CircuitType) (words : String → Option (Array ℕ)) (loOf : ℕ → ℕ) (F : ℕ)
  (t : CircuitType)

theorem flatten_size : t.inBits + 2 ≤ (flatten types words loOf F t).1.size := by
  have := (flatten_fresh types words loOf F t).size_le
  simp at this
  omega

/-- **`flatten`'s input gates are exactly its first `inBits + 1`**: the type's inputs and `one`. -/
theorem flatten_input_iff {j : ℕ} (hj : j < (flatten types words loOf F t).1.size) :
    (flatten types words loOf F t).1[j] = .input ↔ j < t.inBits + 1 := by
  have hf := flatten_fresh types words loOf F t
  by_cases hjs : j < t.inBits + 2
  · rw [hf.getElem_lt (by simp; omega)]
    by_cases hj1 : j < t.inBits + 1
    · simp [Array.getElem_push, hj1]
    · obtain rfl : j = t.inBits + 1 := by omega
      simp [Array.getElem_push]
  · have := hf.not_input hj (by simp; omega)
    constructor
    · intro h; exact absurd h this
    · intro h; omega

/-- **A unit's input gates are exactly its first `inBits + 1`.** -/
theorem unitGates_input_iff {j : ℕ} (hj : j < (unitGates types words loOf F t).size) :
    (unitGates types words loOf F t)[j] = .input ↔ j < t.inBits + 1 := by
  have hs := flatten_size types words loOf F t
  by_cases hjf : j < (flatten types words loOf F t).1.size
  · rw [unitGates_lt types words loOf F t hjf]
    exact flatten_input_iff types words loOf F t hjf
  · obtain ⟨o, rfl⟩ : ∃ o, j = (flatten types words loOf F t).1.size + o :=
      ⟨j - (flatten types words loOf F t).1.size, by omega⟩
    have ho : o < t.outputs.size := by rw [unitGates_size] at hj; omega
    rw [unitGates_copy types words loOf F t ho]
    simp only [reduceCtorEq, false_iff]
    omega

end Types.Flatten

/-! ## A unit, and its output columns -/

/-- **One unit of a program**: the verifier's derivation for its layout (`deriveChecked` on layout `idx`, of type `t`),
and the depth and read split its gates are flattened at. -/
structure UnitSpec where
  types : String → Option CircuitType
  ls : Array Layout
  info : String → Option TableInfo
  words : String → Option (Array ℕ)
  idx : ℕ
  done : Array D
  hd : deriveChecked types ls info words idx = .ok done
  t : CircuitType
  ht : (ls.getD idx default).type.bind types = some t
  loOf : ℕ → ℕ
  F : ℕ
  hF : idx ≤ F

theorem getD_of_lt {a : Array ℕ} {i : ℕ} (h : i < a.size) : a.getD i 0 = a[i] := by
  simp [Array.getD, h]

theorem Audit.Partition.mem_committed_iff {C : Audit.Circuit} {n : ℕ} (P : Partition C n) (g : Fin C.N) :
    g ∈ P.committed ↔ (C.op g).isInput = true ∨ g ∈ C.outputs ∨ ∃ h, g ∈ (C.op h).args ∧ P.unit h ≠ P.unit g := by
  classical
  simp [Partition.committed]

namespace UnitSpec

variable (s : UnitSpec)

/-- **A unit's output columns are distinct**: they end its logical order, which `deriveChecked` checks has no repeat
(`order_sound`). -/
theorem outCols_nodup : (s.done.getD s.idx default).outCols.toList.Nodup := by
  have hn := (order_sound s.hd 0).1
  generalize s.done.getD s.idx default = u at hn ⊢
  unfold order at hn
  rw [DeriveAll.segment] at hn
  exact List.Nodup.of_map _ hn.of_append_right.of_append_right

/-- The unit's rows, in logical order. -/
abbrev R : Rows := Rows.compose (ofBlock s.words s.done (s.done.getD s.idx default)) (ofBlock_ordered s.hd)

/-- The unit's gates. -/
abbrev gs : Array Gate := unitGates s.types s.words s.loOf s.F s.t

/-- The size of its flattening: output `o`'s copy is gate `fsize + o`. -/
abbrev fsize : ℕ := (flatten s.types s.words s.loOf s.F s.t).1.size

/-- Output `o`'s logical column. -/
abbrev out (o : ℕ) : ℕ :=
  lidx (ofBlock s.words s.done (s.done.getD s.idx default)) ((s.done.getD s.idx default).outCols.getD o 0)

theorem inCols_size : (s.done.getD s.idx default).inCols.size = s.t.inBits :=
  (checkLayout_spec (deriveChecked_all s.hd s.idx (lt_of_type s.ht)) s.ht).2.2.1

theorem nIn_eq : s.R.nIn = s.t.inBits := s.inCols_size

theorem w_eq : s.R.w = (order s.done (s.done.getD s.idx default)).length + 1 := Rows.compose_w

theorem nComp_eq : s.R.nComp = (order s.done (s.done.getD s.idx default)).length - s.t.inBits := by
  show (order s.done (s.done.getD s.idx default)).length - (s.done.getD s.idx default).inCols.size = _
  rw [s.inCols_size]

theorem gs_size : s.gs.size = s.fsize + s.t.outputs.size := unitGates_size _ _ _ _ _

theorem fsize_ge : s.t.inBits + 2 ≤ s.fsize := flatten_size _ _ _ _ _

/-- **An output column is not an input column**: an input's row is `c · c` and needs no Δ entry, and an output's row
copies a form times the constant, which has one. -/
theorem outCols_not_inCols {o : ℕ} (ho : o < s.t.outputs.size) :
    (s.done.getD s.idx default).outCols.getD o 0 ∉ (s.done.getD s.idx default).inCols := by
  obtain ⟨-, -, hin, -, hone, hins, wire, -, houts⟩ :=
    checkLayout_spec (deriveChecked_all s.hd s.idx (lt_of_type s.ht)) s.ht
  intro hmem
  obtain ⟨w, hw, hweq⟩ := Array.mem_iff_getElem.1 hmem
  have h1 := hins w (by omega)
  have h2 := houts o ho
  rw [show (s.done.getD s.idx default).inCols.getD w 0 = (s.done.getD s.idx default).outCols.getD o 0 by
    rw [getD_of_lt hw, hweq]] at h1
  simp only [Chk.rowIs, Chk.oneAt, beq_self_eq_true, Bool.and_eq_true, beq_iff_eq, Bool.not_true, Bool.false_or,
    Option.isNone_iff_eq_none, ite_true] at h1 h2 hone
  obtain ⟨⟨hr1, -⟩, hd1⟩ := h1
  obtain ⟨⟨hr2, -⟩, -⟩ := h2
  rw [hr1] at hr2
  have hc := (Prod.mk.inj (Option.some.inj hr2)).2
  rw [List.cons.injEq] at hc
  rw [hc.1, hone] at hd1
  cases hd1

theorem out_lt {o : ℕ} (ho : o < s.t.outputs.size) : s.out o < (order s.done (s.done.getD s.idx default)).length :=
  List.idxOf_lt_length_iff.2 (outCols_mem s.hd s.ht ho)

theorem out_ge {o : ℕ} (ho : o < s.t.outputs.size) : s.t.inBits ≤ s.out o := by
  obtain ⟨-, -, htake, -, -⟩ := order_sound s.hd 0
  by_contra hlt
  push Not at hlt
  have hl : s.out o < (order s.done (s.done.getD s.idx default)).length := s.out_lt ho
  have hi : s.out o < (s.done.getD s.idx default).inCols.size := by rw [s.inCols_size]; exact hlt
  have hg := order_getD_inCols htake hi
  have he : (order s.done (s.done.getD s.idx default)).getD (s.out o) 0 =
      (s.done.getD s.idx default).outCols.getD o 0 := by
    rw [List.getD_eq_getElem _ _ hl]
    exact List.getElem_idxOf hl
  rw [he] at hg
  exact s.outCols_not_inCols ho (by rw [hg, getD_of_lt hi]; exact Array.getElem_mem hi)

theorem out_inj {o o' : ℕ} (ho : o < s.t.outputs.size) (ho' : o' < s.t.outputs.size) (h : s.out o = s.out o') :
    o = o' := by
  obtain ⟨-, -, -, hout, -⟩ := checkLayout_spec (deriveChecked_all s.hd s.idx (lt_of_type s.ht)) s.ht
  have hc := (List.idxOf_inj (outCols_mem s.hd s.ht ho)).1 h
  have hlt : o < (s.done.getD s.idx default).outCols.size := by omega
  have hlt' : o' < (s.done.getD s.idx default).outCols.size := by omega
  rw [getD_of_lt hlt, getD_of_lt hlt'] at hc
  have hc' : (s.done.getD s.idx default).outCols.toList[o]'(by simpa using hlt) =
      (s.done.getD s.idx default).outCols.toList[o']'(by simpa using hlt') := by simpa using hc
  exact (List.Nodup.getElem_inj_iff s.outCols_nodup).1 hc'

end UnitSpec

/-! ## The program -/

/-- Where a unit's input bit reads: program input `a`, or output `o` of unit `v`. -/
inductive Src
  | input (a : ℕ)
  | out (v o : ℕ)

/-- A source a unit may read: a program input other than the constant, or an output of one of the units `outs` lists by
output count. -/
def Src.Valid (k one : ℕ) (outs : List ℕ) : Src → Prop
  | .input a => a < k ∧ a ≠ one
  | .out v o => v < outs.length ∧ o < outs.getD v 0

/-- **A program of units over circuit types**: `k` inputs, input `one` the constant, then units in order. Each input bit
of each unit reads a program input other than the constant or an earlier unit's output, several bits possibly the same
source. The index lists the units' output counts. -/
inductive UProg (k one : ℕ) : List ℕ → Type
  | inputs : UProg k one []
  | snoc {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
      (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) : UProg k one (outs ++ [s.t.outputs.size])

namespace UProg

variable {k one : ℕ}

/-- The number of units. -/
@[reducible] def n : {outs : List ℕ} → UProg k one outs → ℕ
  | _, .inputs => 0
  | _, .snoc p .. => p.n + 1

theorem n_eq : ∀ {outs : List ℕ} (p : UProg k one outs), p.n = outs.length
  | _, .inputs => rfl
  | _, .snoc p .. => by simp [n, n_eq p]

theorem valid_snoc {outs : List ℕ} {x : ℕ} {src : Src} (h : src.Valid k one outs) : src.Valid k one (outs ++ [x]) := by
  cases src with
  | input a => exact h
  | out v o =>
    obtain ⟨hv, ho⟩ := h
    refine ⟨by simp; omega, ?_⟩
    rw [List.getD_append _ _ _ _ hv]
    exact ho

/-! ### The rows circuit -/

/-- The rows circuit's gates. -/
@[reducible] def rowsN : {outs : List ℕ} → UProg k one outs → ℕ
  | _, .inputs => k
  | _, .snoc p s .. => p.rowsN + s.R.nComp

/-- The gate of unit `v`'s output `o`'s row (`one` for none). -/
def outGate : {outs : List ℕ} → UProg k one outs → ℕ → ℕ → ℕ
  | _, .inputs, _, _ => one
  | _, .snoc p s .., v, o => if v = p.n then p.rowsN + (s.out o - s.t.inBits) else p.outGate v o

/-- A source's gate in the rows circuit. -/
def srcGate {outs : List ℕ} (p : UProg k one outs) : Src → ℕ
  | .input a => a
  | .out v o => p.outGate v o

theorem k_le_rowsN : ∀ {outs : List ℕ} (p : UProg k one outs), k ≤ p.rowsN
  | _, .inputs => le_refl k
  | _, .snoc p .. => (k_le_rowsN p).trans (Nat.le_add_right _ _)

theorem outGate_spec : ∀ {outs : List ℕ} (p : UProg k one outs) {v o : ℕ}, v < outs.length → o < outs.getD v 0 →
    k ≤ p.outGate v o ∧ p.outGate v o < p.rowsN
  | _, .inputs, _, _, hv, _ => absurd hv (by simp)
  | _, .snoc (outs := outs) p s .., v, o, hv, ho => by
    simp only [outGate, rowsN]
    have hn := n_eq p
    split
    · rename_i hvn
      rw [hvn, hn, List.getD_append_right _ _ _ _ (le_refl _)] at ho
      simp only [Nat.sub_self, List.getD_cons_zero] at ho
      have h1 := s.out_lt ho
      have h2 := s.out_ge ho
      have h3 := s.nComp_eq
      have := k_le_rowsN p
      constructor <;> omega
    · rename_i hvn
      have hv' : v < outs.length := by simp at hv; omega
      rw [List.getD_append _ _ _ _ hv'] at ho
      have := outGate_spec p hv' ho
      constructor <;> omega

theorem outGate_inj : ∀ {outs : List ℕ} (p : UProg k one outs) {v o v' o' : ℕ},
    v < outs.length → o < outs.getD v 0 → v' < outs.length → o' < outs.getD v' 0 →
    p.outGate v o = p.outGate v' o' → v = v' ∧ o = o'
  | _, .inputs, _, _, _, _, hv, _, _, _, _ => absurd hv (by simp)
  | _, .snoc (outs := outs) p s .., v, o, v', o', hv, ho, hv', ho', h => by
    simp only [outGate] at h
    have hn := n_eq p
    by_cases hvn : v = p.n <;> by_cases hvn' : v' = p.n <;> simp only [hvn, hvn', ite_true, ite_false] at h
    · rw [hvn, hn, List.getD_append_right _ _ _ _ (le_refl _)] at ho
      rw [hvn', hn, List.getD_append_right _ _ _ _ (le_refl _)] at ho'
      simp only [Nat.sub_self, List.getD_cons_zero] at ho ho'
      have := s.out_ge ho
      have := s.out_ge ho'
      exact ⟨hvn.trans hvn'.symm, s.out_inj ho ho' (by omega)⟩
    · have hv'' : v' < outs.length := by simp at hv'; omega
      rw [List.getD_append _ _ _ _ hv''] at ho'
      have := (outGate_spec p hv'' ho').2
      omega
    · have hv'' : v < outs.length := by simp at hv; omega
      rw [List.getD_append _ _ _ _ hv''] at ho
      have := (outGate_spec p hv'' ho).2
      omega
    · have hv1 : v < outs.length := by simp at hv; omega
      have hv2 : v' < outs.length := by simp at hv'; omega
      rw [List.getD_append _ _ _ _ hv1] at ho
      rw [List.getD_append _ _ _ _ hv2] at ho'
      exact outGate_inj p hv1 ho hv2 ho' h

theorem srcGate_lt {outs : List ℕ} (p : UProg k one outs) {s : Src} (h : s.Valid k one outs) : p.srcGate s < p.rowsN := by
  cases s with
  | input a => exact Nat.lt_of_lt_of_le h.1 p.k_le_rowsN
  | out v o => exact (p.outGate_spec h.1 h.2).2

theorem srcGate_ne_one {outs : List ℕ} (p : UProg k one outs) (hk : one < k) {s : Src} (h : s.Valid k one outs) :
    p.srcGate s ≠ one := by
  cases s with
  | input a => exact h.2
  | out v o => have := (p.outGate_spec h.1 h.2).1; simp only [srcGate]; omega

theorem srcGate_inj {outs : List ℕ} (p : UProg k one outs) {s s' : Src} (h : s.Valid k one outs)
    (h' : s'.Valid k one outs) (he : p.srcGate s = p.srcGate s') : s = s' := by
  cases s with
  | input a =>
    cases s' with
    | input a' => simp only [srcGate] at he; rw [he]
    | out v' o' => have := (p.outGate_spec h'.1 h'.2).1; simp only [srcGate] at he; have := h.1; omega
  | out v o =>
    cases s' with
    | input a' => have := (p.outGate_spec h.1 h.2).1; simp only [srcGate] at he; have := h'.1; omega
    | out v' o' =>
      obtain ⟨rfl, rfl⟩ := p.outGate_inj h.1 h.2 h'.1 h'.2 he
      rfl

/-- A new unit's input columns, wired to their sources' gates. -/
def insOf {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) : Fin s.R.nIn → Fin p.rowsN :=
  fun c => ⟨p.srcGate (src c.val), p.srcGate_lt (valid c.val (by rw [← s.nIn_eq]; exact c.isLt))⟩

/-- The constant's gate. -/
def oneOf (hk : one < k) {outs : List ℕ} (p : UProg k one outs) : Fin p.rowsN :=
  ⟨one, Nat.lt_of_lt_of_le hk p.k_le_rowsN⟩

theorem insOf_ne (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) : ∀ c, p.insOf s src valid c ≠ p.oneOf hk :=
  fun c h => p.srcGate_ne_one hk (valid _ (by rw [← s.nIn_eq]; exact c.isLt)) (congrArg Fin.val h)

/-- **The program's rows circuit**, a `Lowering.Prog`: each unit's composed rows, its input columns wired to their
sources' gates and its constant to `one`. -/
def toRows (hk : one < k) : {outs : List ℕ} → (p : UProg k one outs) → Prog p.rowsN p.n
  | _, .inputs => .inputs k
  | _, .snoc p s src valid =>
    .snoc (p.toRows hk) s.R (p.insOf s src valid) (p.oneOf hk) (p.insOf_ne hk s src valid)

/-! ### The Boolean circuit -/

/-- The Boolean circuit's gates: the inputs, then each unit's gates past its inputs and `one`. -/
@[reducible] def boolN : {outs : List ℕ} → UProg k one outs → ℕ
  | _, .inputs => k
  | _, .snoc p s .. => p.boolN + (s.gs.size - (s.t.inBits + 1))

/-- The gate of unit `v`'s output `o`'s copy (`one` for none). -/
def copyGate : {outs : List ℕ} → UProg k one outs → ℕ → ℕ → ℕ
  | _, .inputs, _, _ => one
  | _, .snoc p s .., v, o => if v = p.n then p.boolN + (s.fsize + o - (s.t.inBits + 1)) else p.copyGate v o

/-- A source's gate in the Boolean circuit. -/
def srcB {outs : List ℕ} (p : UProg k one outs) : Src → ℕ
  | .input a => a
  | .out v o => p.copyGate v o

theorem k_le_boolN : ∀ {outs : List ℕ} (p : UProg k one outs), k ≤ p.boolN
  | _, .inputs => le_refl k
  | _, .snoc p .. => (k_le_boolN p).trans (Nat.le_add_right _ _)

theorem copyGate_spec : ∀ {outs : List ℕ} (p : UProg k one outs) {v o : ℕ}, v < outs.length → o < outs.getD v 0 →
    k ≤ p.copyGate v o ∧ p.copyGate v o < p.boolN
  | _, .inputs, _, _, hv, _ => absurd hv (by simp)
  | _, .snoc (outs := outs) p s .., v, o, hv, ho => by
    simp only [copyGate, boolN]
    have hn := n_eq p
    have hs := s.gs_size
    have hf := s.fsize_ge
    split
    · rename_i hvn
      rw [hvn, hn, List.getD_append_right _ _ _ _ (le_refl _)] at ho
      simp only [Nat.sub_self, List.getD_cons_zero] at ho
      have := k_le_boolN p
      constructor <;> omega
    · rename_i hvn
      have hv' : v < outs.length := by simp at hv; omega
      rw [List.getD_append _ _ _ _ hv'] at ho
      have := copyGate_spec p hv' ho
      constructor <;> omega

theorem srcB_lt {outs : List ℕ} (p : UProg k one outs) {s : Src} (h : s.Valid k one outs) : p.srcB s < p.boolN := by
  cases s with
  | input a => exact Nat.lt_of_lt_of_le h.1 p.k_le_boolN
  | out v o => exact (p.copyGate_spec h.1 h.2).2

/-- Gate `j` of unit `s`, appended to `p` and reading `src`, as a gate of the program: an input its source's gate, `one`
the constant, any other its place after `p`'s gates. -/
def wireB {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src) (j : ℕ) : ℕ :=
  if j < s.t.inBits then p.srcB (src j) else if j = s.t.inBits then one else p.boolN + (j - (s.t.inBits + 1))

/-- `wireB` as a gate (`one` past the unit's gates). -/
def wireF (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src) (j : ℕ) :
    Fin (p.boolN + (s.gs.size - (s.t.inBits + 1))) :=
  if h : p.wireB s src j < p.boolN + (s.gs.size - (s.t.inBits + 1)) then ⟨_, h⟩
  else ⟨one, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le hk p.k_le_boolN) (Nat.le_add_right _ _)⟩

theorem wireB_lt (hk : one < k) {outs : List ℕ} (p : UProg k one outs) {s : UnitSpec} {src : ℕ → Src}
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) {j : ℕ} (hj : j < s.gs.size) :
    p.wireB s src j < p.boolN + (s.gs.size - (s.t.inBits + 1)) := by
  unfold wireB
  split
  · rename_i h; have := p.srcB_lt (valid j h); omega
  · split
    · have := p.k_le_boolN; omega
    · omega

theorem wireF_val (hk : one < k) {outs : List ℕ} (p : UProg k one outs) {s : UnitSpec} {src : ℕ → Src}
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) {j : ℕ} (hj : j < s.gs.size) :
    (p.wireF hk s src j).val = p.wireB s src j := by
  unfold wireF
  rw [dite_eq_left (p.wireB_lt hk valid hj)]

/-- The gates' operations: a unit's gates over its gates' wires. -/
def opB (hk : one < k) : {outs : List ℕ} → (p : UProg k one outs) → Fin p.boolN → Op p.boolN
  | _, .inputs, _ => .input
  | _, .snoc p s src .., g =>
    if h : g.val < p.boolN then (p.opB hk ⟨g.val, h⟩).lift (Fin.castAdd _)
    else if h' : g.val - p.boolN + (s.t.inBits + 1) < s.gs.size then
      (s.gs[g.val - p.boolN + (s.t.inBits + 1)]'h').lift (p.wireF hk s src)
    else .input

theorem gs_bound {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) {g : ℕ}
    (hg : g < p.boolN + (s.gs.size - (s.t.inBits + 1))) (h : ¬ g < p.boolN) :
    g - p.boolN + (s.t.inBits + 1) < s.gs.size := by
  have := s.gs_size; have := s.fsize_ge; omega

theorem topoB (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) (g : Fin p.boolN),
    ∀ a ∈ (p.opB hk g).args, a < g
  | _, .inputs, _, _, ha => by simp [opB, Op.args] at ha
  | _, .snoc p s src valid, g, a, ha => by
    by_cases h : g.val < p.boolN
    · simp only [opB, h, dite_true, Audit.Op.args_lift] at ha
      obtain ⟨a', ha', rfl⟩ := List.mem_map.1 ha
      have := topoB hk p ⟨g.val, h⟩ a' ha'
      rw [Fin.lt_def] at this ⊢
      simpa using this
    · have h' := p.gs_bound s g.isLt h
      simp only [opB, h, dite_false, dite_true, h', Types.Flatten.Gate.lift_args] at ha
      obtain ⟨b, hb, rfl⟩ := List.mem_map.1 ha
      have hbj := unitGates_topo _ _ _ _ _ _ h' b hb
      rw [Fin.lt_def, p.wireF_val hk valid (by omega)]
      unfold wireB
      split
      · rename_i hb'; have := p.srcB_lt (valid b hb'); omega
      · split
        · have := p.k_le_boolN; omega
        · omega

/-- The gates' units: a unit's gates past its inputs and `one` are its. -/
def unitB : {outs : List ℕ} → (p : UProg k one outs) → Fin p.boolN → Option (Fin p.n)
  | _, .inputs, _ => none
  | _, .snoc p .., g => if h : g.val < p.boolN then (p.unitB ⟨g.val, h⟩).map Fin.castSucc else some (Fin.last p.n)

theorem input_iffB (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) (g : Fin p.boolN),
    p.unitB g = none ↔ (p.opB hk g).isInput = true
  | _, .inputs, _ => by simp [unitB, opB, Op.isInput]
  | _, .snoc p s src .., g => by
    by_cases h : g.val < p.boolN
    · simp only [unitB, opB, h, dite_true, Option.map_eq_none_iff, Audit.Op.isInput_lift]
      exact input_iffB hk p _
    · have h' := p.gs_bound s g.isLt h
      simp only [unitB, opB, h, dite_false, dite_true, h', reduceCtorEq, false_iff, Types.Flatten.Gate.isInput_lift]
      rw [unitGates_input_iff _ _ _ _ _ h']
      omega

open Classical in
/-- Every unit output's copy. -/
noncomputable def outsB {outs : List ℕ} (p : UProg k one outs) : Finset (Fin p.boolN) :=
  Finset.univ.filter fun g => ∃ v o, v < outs.length ∧ o < outs.getD v 0 ∧ g.val = p.copyGate v o

/-- **The program's Boolean circuit**: each unit's unit gates, its inputs on their sources' copies and its `one` on
`one`; every unit output's copy an output. -/
noncomputable abbrev CB (hk : one < k) {outs : List ℕ} (p : UProg k one outs) : Audit.Circuit :=
  ⟨p.boolN, p.opB hk, p.topoB hk, p.outsB⟩

/-- Its partition into the program's units. -/
noncomputable abbrev PB (hk : one < k) {outs : List ℕ} (p : UProg k one outs) : Partition (p.CB hk) p.n :=
  ⟨p.unitB, p.input_iffB hk⟩

open Classical in
/-- Every unit output's row. -/
noncomputable def outsC {outs : List ℕ} (p : UProg k one outs) : Finset (Fin p.rowsN) :=
  Finset.univ.filter fun g => ∃ v o, v < outs.length ∧ o < outs.getD v 0 ∧ g.val = p.outGate v o

/-- **The program's rows circuit** (`Prog.circuit`), every unit output's row an output. -/
noncomputable abbrev C (hk : one < k) {outs : List ℕ} (p : UProg k one outs) : Audit.Circuit :=
  ⟨p.rowsN, (p.toRows hk).op, (p.toRows hk).topo, p.outsC⟩

/-- Its partition into the program's units (`Prog.partition`). -/
noncomputable abbrev P (hk : one < k) {outs : List ℕ} (p : UProg k one outs) : Partition (p.C hk) p.n :=
  ⟨(p.toRows hk).unit, (p.toRows hk).input_iff⟩

theorem mem_outsC {outs : List ℕ} (p : UProg k one outs) (g : Fin p.rowsN) :
    g ∈ p.outsC ↔ ∃ v o, v < outs.length ∧ o < outs.getD v 0 ∧ g.val = p.outGate v o := by
  classical
  simp [outsC]

theorem mem_outsB {outs : List ℕ} (p : UProg k one outs) (g : Fin p.boolN) :
    g ∈ p.outsB ↔ ∃ v o, v < outs.length ∧ o < outs.getD v 0 ∧ g.val = p.copyGate v o := by
  classical
  simp [outsB]

/-! ### Reading the Boolean circuit off the rows circuit -/

/-- **The gate map**: an input reads the same input, a copy its output's row (`one` for any other gate). -/
def mapG (hk : one < k) : {outs : List ℕ} → (p : UProg k one outs) → Fin p.boolN → Fin p.rowsN
  | _, .inputs, g => g
  | _, .snoc p s .., g =>
    if h : g.val < p.boolN then Fin.castAdd _ (p.mapG hk ⟨g.val, h⟩)
    else if ho : s.fsize ≤ g.val - p.boolN + (s.t.inBits + 1) ∧
        g.val - p.boolN + (s.t.inBits + 1) - s.fsize < s.t.outputs.size then
      ⟨p.rowsN + (s.out (g.val - p.boolN + (s.t.inBits + 1) - s.fsize) - s.t.inBits), by
        have := s.out_lt ho.2; have := s.out_ge ho.2; have := s.nComp_eq
        show _ < p.rowsN + s.R.nComp; omega⟩
    else ⟨one, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le hk p.k_le_rowsN) (Nat.le_add_right _ _)⟩

theorem mapG_input (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) {a : ℕ} (ha : a < p.boolN), a < k →
    (p.mapG hk ⟨a, ha⟩).val = a
  | _, .inputs, _, _, _ => rfl
  | _, .snoc p .., a, _, hak => by
    have hlt := Nat.lt_of_lt_of_le hak p.k_le_boolN
    simp only [mapG, hlt, dite_true, Fin.val_castAdd]
    exact mapG_input hk p hlt hak

theorem mapG_copy (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) {v o : ℕ} (hv : v < outs.length)
    (ho : o < outs.getD v 0), (p.mapG hk ⟨p.copyGate v o, (p.copyGate_spec hv ho).2⟩).val = p.outGate v o
  | _, .inputs, _, _, hv, _ => absurd hv (by simp)
  | _, .snoc (outs := outs) p s src valid, v, o, hv, ho => by
    have hn := n_eq p
    by_cases hvn : v = p.n
    · subst hvn
      have ho' := ho
      rw [hn, List.getD_append_right _ _ _ _ (le_refl _)] at ho'
      simp only [Nat.sub_self, List.getD_cons_zero] at ho'
      have hf := s.fsize_ge
      have hc : (snoc p s src valid).copyGate p.n o = p.boolN + (s.fsize + o - (s.t.inBits + 1)) := by
        simp [copyGate]
      have hj : p.boolN + (s.fsize + o - (s.t.inBits + 1)) - p.boolN + (s.t.inBits + 1) - s.fsize = o := by omega
      simp only [mapG]
      rw [dite_eq_right (by rw [hc]; omega), dite_eq_left (by rw [hc]; constructor <;> omega)]
      simp only [outGate, ite_true, hc, hj]
    · have hv' : v < outs.length := by simp at hv; omega
      rw [List.getD_append _ _ _ _ hv'] at ho
      have hc : (snoc p s src valid).copyGate v o = p.copyGate v o := by simp [copyGate, hvn]
      have hlt := (p.copyGate_spec hv' ho).2
      simp only [mapG]
      rw [dite_eq_left (by rw [hc]; exact hlt)]
      simp only [Fin.val_castAdd, outGate, hvn, ite_false]
      have := mapG_copy hk p hv' ho
      simp only [hc]
      exact this

theorem mapG_srcB (hk : one < k) {outs : List ℕ} (p : UProg k one outs) {s : Src} (h : s.Valid k one outs) :
    (p.mapG hk ⟨p.srcB s, p.srcB_lt h⟩).val = p.srcGate s := by
  cases s with
  | input a => exact p.mapG_input hk _ h.1
  | out v o => exact p.mapG_copy hk h.1 h.2

/-! ### Gates and units, old and new -/

theorem unitB_input (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) {a : ℕ} (ha : a < p.boolN), a < k →
    p.unitB ⟨a, ha⟩ = none
  | _, .inputs, _, _, _ => rfl
  | _, .snoc p .., a, _, hak => by
    have hlt := Nat.lt_of_lt_of_le hak p.k_le_boolN
    simp only [unitB, hlt, dite_true, Option.map_eq_none_iff]
    exact unitB_input hk p hlt hak

theorem unitB_old {outs : List ℕ} {p : UProg k one outs} {s : UnitSpec} {src : ℕ → Src}
    {valid : ∀ i < s.t.inBits, (src i).Valid k one outs}
    {g : Fin (snoc p s src valid).boolN} (h : g.val < p.boolN) :
    (snoc p s src valid).unitB g = (p.unitB ⟨g.val, h⟩).map Fin.castSucc := by
  simp only [unitB, h, dite_true]

theorem unitB_new {outs : List ℕ} {p : UProg k one outs} {s : UnitSpec} {src : ℕ → Src}
    {valid : ∀ i < s.t.inBits, (src i).Valid k one outs}
    {g : Fin (snoc p s src valid).boolN} (h : ¬ g.val < p.boolN) :
    (snoc p s src valid).unitB g = some (Fin.last p.n) := by
  simp only [unitB, h, dite_false]

theorem lt_of_unitB_castSucc {outs : List ℕ} {p : UProg k one outs} {s : UnitSpec} {src : ℕ → Src}
    {valid : ∀ i < s.t.inBits, (src i).Valid k one outs}
    {g : Fin (snoc p s src valid).boolN} {v : Fin p.n} (hg : (snoc p s src valid).unitB g = some v.castSucc) :
    g.val < p.boolN := by
  by_contra h
  rw [unitB_new h] at hg
  exact Fin.castSucc_ne_last v (Option.some.inj hg).symm

theorem opB_old (hk : one < k) {outs : List ℕ} {p : UProg k one outs} {s : UnitSpec} {src : ℕ → Src}
    {valid : ∀ i < s.t.inBits, (src i).Valid k one outs}
    {g : Fin (snoc p s src valid).boolN} (h : g.val < p.boolN) :
    (snoc p s src valid).opB hk g = (p.opB hk ⟨g.val, h⟩).lift (Fin.castAdd _) := by
  simp only [opB, h, dite_true]

theorem opB_new (hk : one < k) {outs : List ℕ} {p : UProg k one outs} {s : UnitSpec} {src : ℕ → Src}
    {valid : ∀ i < s.t.inBits, (src i).Valid k one outs}
    {g : Fin (snoc p s src valid).boolN} (h : ¬ g.val < p.boolN) :
    (snoc p s src valid).opB hk g =
      (s.gs[g.val - p.boolN + (s.t.inBits + 1)]'(p.gs_bound s g.isLt h)).lift (p.wireF hk s src) := by
  simp only [opB, h, dite_false, p.gs_bound s g.isLt h, dite_true]

/-! ### Each unit's instances -/

/-- The new unit is an instance of its unit gates, through `wireF`. -/
def gatesLast (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) :
    ((snoc p s src valid).PB hk).IsGatesUnit (Fin.last p.n) s.gs (s.t.inBits + 1) where
  wire := p.wireF hk s src
  input j hj hjk := (unitGates_input_iff _ _ _ _ _ hj).2 hjk
  outside j hj := by
    have hjs : j < s.gs.size := by have := s.gs_size; have := s.fsize_ge; omega
    have hlt : (p.wireF hk s src j).val < p.boolN := by
      rw [p.wireF_val hk valid hjs]
      unfold wireB
      split
      · rename_i h; exact p.srcB_lt (valid j h)
      · rw [ite_eq_left (by omega)]; have := p.k_le_boolN; omega
    show (snoc p s src valid).unitB _ ≠ _
    rw [unitB_old hlt]
    intro h
    obtain ⟨v, -, hv⟩ := Option.map_eq_some_iff.1 h
    exact Fin.castSucc_ne_last v hv
  mem j hj hkj := by
    have hw : (p.wireF hk s src j).val = p.boolN + (j - (s.t.inBits + 1)) := by
      rw [p.wireF_val hk valid hj]; unfold wireB; rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    show (snoc p s src valid).unitB _ = _
    exact unitB_new (by omega)
  op j hj hkj := by
    have hw : (p.wireF hk s src j).val = p.boolN + (j - (s.t.inBits + 1)) := by
      rw [p.wireF_val hk valid hj]; unfold wireB; rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    show (snoc p s src valid).opB hk _ = _
    rw [opB_new hk (by omega)]
    have hj' : (p.wireF hk s src j).val - p.boolN + (s.t.inBits + 1) = j := by rw [hw]; omega
    congr 1
    simp only [hj']
  onto g hg := by
    change (snoc p s src valid).unitB g = _ at hg
    have hge : p.boolN ≤ g.val := by
      by_contra hlt
      push Not at hlt
      rw [unitB_old hlt] at hg
      obtain ⟨v, -, hv⟩ := Option.map_eq_some_iff.1 hg
      exact Fin.castSucc_ne_last v hv
    have hb := p.gs_bound s g.isLt (by omega)
    refine ⟨g.val - p.boolN + (s.t.inBits + 1), hb, by omega, Fin.ext ?_⟩
    rw [p.wireF_val hk valid hb]
    unfold wireB
    rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    omega

/-- An earlier unit's instance, after a unit is appended. -/
def gatesSnoc (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) {v : Fin p.n} {gs : Array Gate} {kk : ℕ}
    (W : (p.PB hk).IsGatesUnit v gs kk) : ((snoc p s src valid).PB hk).IsGatesUnit v.castSucc gs kk where
  wire j := Fin.castAdd _ (W.wire j)
  input := W.input
  outside j hj := by
    show (snoc p s src valid).unitB _ ≠ _
    rw [unitB_old (by simp)]
    intro h
    obtain ⟨v', hv', hvv⟩ := Option.map_eq_some_iff.1 h
    rw [Fin.castSucc_inj.1 hvv] at hv'
    exact W.outside j hj (by simpa using hv')
  mem j hj hkj := by
    show (snoc p s src valid).unitB _ = _
    rw [unitB_old (by simp)]
    have := W.mem j hj hkj
    simp only [Fin.val_castAdd, Fin.eta]
    exact (congrArg (Option.map Fin.castSucc) this).trans rfl
  op j hj hkj := by
    show (snoc p s src valid).opB hk _ = _
    rw [opB_old hk (by simp)]
    simp only [Fin.val_castAdd, Fin.eta]
    have := W.op j hj hkj
    change p.opB hk (W.wire j) = _ at this
    rw [this, Types.Flatten.Gate.lift_lift]
    rfl
  onto g hg := by
    change (snoc p s src valid).unitB g = _ at hg
    have hlt := lt_of_unitB_castSucc hg
    rw [unitB_old hlt] at hg
    obtain ⟨v', hv', hvv⟩ := Option.map_eq_some_iff.1 hg
    rw [Fin.castSucc_inj.1 hvv] at hv'
    obtain ⟨j, hj, hkj, hw⟩ := W.onto _ hv'
    exact ⟨j, hj, hkj, Fin.ext (by simp [hw])⟩

/-- A `Prog`-level instance from the audit's. -/
def rowsWireOf {N m : ℕ} {q : Prog N m} {os : Finset (Fin N)} {u : Fin m} {R : Rows}
    (I : (q.partition os).IsRowsUnit u R) : q.RowsWire u R :=
  ⟨I.wire, I.shared, I.mem, I.op, I.outside, I.onto⟩

/-! ### Committed gates -/

/-- A committed gate of an earlier unit was committed before the unit was appended. -/
theorem committedB_old (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) {g : Fin (snoc p s src valid).boolN}
    {v : Fin p.n} (hg : g ∈ ((snoc p s src valid).PB hk).committed)
    (hv : (snoc p s src valid).unitB g = some v.castSucc) :
    (⟨g.val, lt_of_unitB_castSucc hv⟩ : Fin p.boolN) ∈ (p.PB hk).committed := by
  classical
  have hlt := lt_of_unitB_castSucc hv
  have hn := n_eq p
  have hin : ∀ a < k, g.val ≠ a := by
    intro a ha he
    have := unitB_input hk p (a := g.val) hlt (by omega)
    rw [unitB_old hlt, this] at hv
    cases hv
  replace hg := (Partition.mem_committed_iff _ _).1 hg
  refine (Partition.mem_committed_iff _ _).2 ?_
  rcases hg with hg | hg | ⟨h, hga, hne⟩
  · exfalso
    have := ((snoc p s src valid).input_iffB hk g).2 hg
    change (snoc p s src valid).unitB g = none at this
    rw [hv] at this
    cases this
  · refine Or.inr (Or.inl ?_)
    change _ ∈ outsB _ at hg
    change _ ∈ outsB _
    rw [mem_outsB] at hg ⊢
    obtain ⟨v', o', hv', ho', he⟩ := hg
    by_cases hvn : v' = p.n
    · have hs := s.fsize_ge
      simp only [copyGate, hvn, ite_true] at he
      omega
    · have hv'' : v' < outs.length := by simp at hv'; omega
      rw [List.getD_append _ _ _ _ hv''] at ho'
      simp only [copyGate, hvn, ite_false] at he
      exact ⟨v', o', hv'', ho', he⟩
  · by_cases hh : h.val < p.boolN
    · refine Or.inr (Or.inr ⟨⟨h.val, hh⟩, ?_, ?_⟩)
      · change g ∈ ((snoc p s src valid).opB hk h).args at hga
        rw [opB_old hk hh, Audit.Op.args_lift] at hga
        obtain ⟨a, ha, hae⟩ := List.mem_map.1 hga
        have hav : a.val = g.val := by have := congrArg Fin.val hae; simpa using this
        have : a = ⟨g.val, hlt⟩ := Fin.ext hav
        rw [← this]
        exact ha
      · change (snoc p s src valid).unitB h ≠ (snoc p s src valid).unitB g at hne
        rw [unitB_old hh, unitB_old hlt] at hne
        exact fun he => hne (congrArg (Option.map Fin.castSucc) he)
    · refine Or.inr (Or.inl ?_)
      change g ∈ ((snoc p s src valid).opB hk h).args at hga
      rw [opB_new hk hh, Types.Flatten.Gate.lift_args] at hga
      obtain ⟨b, hb, hbe⟩ := List.mem_map.1 hga
      have hbj := unitGates_topo _ _ _ _ _ _ (p.gs_bound s h.isLt hh) b hb
      have hbs : b < s.gs.size := by have := p.gs_bound s h.isLt hh; omega
      have hwv := p.wireF_val hk valid hbs
      rw [hbe] at hwv
      unfold wireB at hwv
      change _ ∈ outsB _
      rw [mem_outsB]
      split at hwv
      · rename_i hbi
        have hvb := valid b hbi
        cases hsb : src b with
        | input a =>
          rw [hsb] at hwv hvb
          simp only [srcB] at hwv
          exact absurd hwv (hin a hvb.1)
        | out v' o' =>
          rw [hsb] at hwv hvb
          simp only [srcB] at hwv
          exact ⟨v', o', hvb.1, hvb.2, hwv⟩
      · split at hwv
        · exact absurd hwv (hin one hk)
        · omega

/-- A committed gate of the new unit is one of its copies. -/
theorem committedB_last (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) {g : Fin (snoc p s src valid).boolN}
    (hg : g ∈ ((snoc p s src valid).PB hk).committed)
    (hv : (snoc p s src valid).unitB g = some (Fin.last p.n)) :
    ∃ o < s.t.outputs.size, g.val = p.boolN + (s.fsize + o - (s.t.inBits + 1)) := by
  classical
  have hn := n_eq p
  have hge : p.boolN ≤ g.val := by
    by_contra hlt
    push Not at hlt
    rw [unitB_old hlt] at hv
    obtain ⟨v, -, hv'⟩ := Option.map_eq_some_iff.1 hv
    exact Fin.castSucc_ne_last v hv'
  replace hg := (Partition.mem_committed_iff _ _).1 hg
  rcases hg with hg | hg | ⟨h, hga, hne⟩
  · exfalso
    have := ((snoc p s src valid).input_iffB hk g).2 hg
    change (snoc p s src valid).unitB g = none at this
    rw [hv] at this
    cases this
  · change _ ∈ outsB _ at hg
    rw [mem_outsB] at hg
    obtain ⟨v', o', hv', ho', he⟩ := hg
    by_cases hvn : v' = p.n
    · subst hvn
      rw [hn, List.getD_append_right _ _ _ _ (le_refl _)] at ho'
      simp only [Nat.sub_self, List.getD_cons_zero] at ho'
      simp only [copyGate, ite_true] at he
      exact ⟨o', ho', he⟩
    · have hv'' : v' < outs.length := by simp at hv'; omega
      rw [List.getD_append _ _ _ _ hv''] at ho'
      simp only [copyGate, hvn, ite_false] at he
      have := (p.copyGate_spec hv'' ho').2
      omega
  · exfalso
    by_cases hh : h.val < p.boolN
    · change g ∈ ((snoc p s src valid).opB hk h).args at hga
      rw [opB_old hk hh, Audit.Op.args_lift] at hga
      obtain ⟨a, -, hae⟩ := List.mem_map.1 hga
      have : g.val = a.val := by rw [← hae]; simp
      have := a.isLt
      omega
    · change (snoc p s src valid).unitB h ≠ (snoc p s src valid).unitB g at hne
      rw [unitB_new hh, hv] at hne
      exact hne rfl

/-- A committed gate of the rows circuit stays committed when a unit is appended. -/
theorem committedC_old (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) {g : Fin p.rowsN}
    (hg : g ∈ (p.P hk).committed) : Fin.castAdd s.R.nComp g ∈ ((snoc p s src valid).P hk).committed := by
  classical
  have hn := n_eq p
  replace hg := (Partition.mem_committed_iff _ _).1 hg
  refine (Partition.mem_committed_iff _ _).2 ?_
  rcases hg with hg | hg | ⟨h, hga, hne⟩
  · left
    change (((snoc p s src valid).toRows hk).op _).isInput = true
    change ((p.toRows hk).op g).isInput = true at hg
    simp only [toRows]
    rw [Prog.op_castAdd, Audit.Op.isInput_lift]
    exact hg
  · refine Or.inr (Or.inl ?_)
    change _ ∈ outsC _ at hg
    change _ ∈ outsC _
    rw [mem_outsC] at hg ⊢
    obtain ⟨v, o, hv, ho, he⟩ := hg
    refine ⟨v, o, by simp; omega, by rw [List.getD_append _ _ _ _ hv]; exact ho, ?_⟩
    simp only [Fin.val_castAdd, outGate, show v ≠ p.n by omega, ite_false]
    exact he
  · refine Or.inr (Or.inr ⟨Fin.castAdd s.R.nComp h, ?_, ?_⟩)
    · change _ ∈ (((snoc p s src valid).toRows hk).op _).args
      change g ∈ ((p.toRows hk).op h).args at hga
      simp only [toRows]
      rw [Prog.op_castAdd, Audit.Op.args_lift]
      exact List.mem_map_of_mem hga
    · change ((snoc p s src valid).toRows hk).unit _ ≠ ((snoc p s src valid).toRows hk).unit _
      change (p.toRows hk).unit h ≠ (p.toRows hk).unit g at hne
      simp only [toRows]
      rw [Prog.unit_castAdd, Prog.unit_castAdd]
      intro he
      exact hne (Option.map_injective (Fin.castSucc_injective _) he)

/-! ### One pair per unit -/

/-- **The new unit's pair**: its composed rows, its unit gates, its inputs and `one` on the same sources, and each
committed copy read off its output's row. -/
noncomputable def lastPair (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) :
    UnitPair ((snoc p s src valid).P hk) ((snoc p s src valid).PB hk) ((snoc p s src valid).mapG hk)
      (Fin.last p.n) where
  types := s.types
  ls := s.ls
  info := s.info
  words := s.words
  idx := s.idx
  done := s.done
  hd := s.hd
  t := s.t
  ht := s.ht
  ord := ofBlock_ordered s.hd
  rows := (Prog.rowsWire_last (p.toRows hk) s.R _ _ _).toIsRowsUnit _
  loOf := s.loOf
  F := s.F
  hF := s.hF
  gates := gatesLast hk p s src valid
  ins c hc := by
    apply Fin.ext
    have hcs : c.val < s.gs.size := by have := s.gs_size; have := s.fsize_ge; omega
    have hw : (p.wireF hk s src c.val).val = p.srcB (src c.val) := by
      rw [p.wireF_val hk valid hcs]; simp [wireB, hc]
    have hlt : (p.wireF hk s src c.val).val < p.boolN := by rw [hw]; exact p.srcB_lt (valid _ hc)
    change ((snoc p s src valid).mapG hk (p.wireF hk s src c.val)).val = (Prog.colWire s.R _ _ c).val
    simp only [mapG, hlt, dite_true, Fin.val_castAdd]
    rw [show (⟨(p.wireF hk s src c.val).val, hlt⟩ : Fin p.boolN) = ⟨p.srcB (src c.val), p.srcB_lt (valid _ hc)⟩ from
      Fin.ext hw, p.mapG_srcB hk (valid _ hc), Prog.colWire_val, dite_eq_left (by rw [s.nIn_eq]; exact hc)]
    rfl
  one := by
    apply Fin.ext
    have hs := s.fsize_ge
    have hw : (p.wireF hk s src s.t.inBits).val = one := by
      rw [p.wireF_val hk valid (by have := s.gs_size; omega)]; simp [wireB]
    have hlt : (p.wireF hk s src s.t.inBits).val < p.boolN := by rw [hw]; have := p.k_le_boolN; omega
    change ((snoc p s src valid).mapG hk (p.wireF hk s src s.t.inBits)).val = (Prog.colWire s.R _ _ s.R.one).val
    simp only [mapG, hlt, dite_true, Fin.val_castAdd]
    rw [show (⟨(p.wireF hk s src s.t.inBits).val, hlt⟩ : Fin p.boolN) = ⟨one, Nat.lt_of_lt_of_le hk p.k_le_boolN⟩ from Fin.ext hw,
      p.mapG_input hk _ hk, Prog.colWire_val]
    simp [Rows.one, oneOf]
  out g hg hgu := by
    obtain ⟨o, ho, hgo⟩ := p.committedB_last hk s src valid hg hgu
    have hs := s.fsize_ge
    have hsz := s.gs_size
    have hlo := s.out_lt ho
    have hge := s.out_ge ho
    have hnc := s.nComp_eq
    have hn := n_eq p
    have hwl : s.out o < s.R.w := by rw [s.w_eq]; omega
    have hv : p.n < (outs ++ [s.t.outputs.size]).length := by simp [hn]
    have ho' : o < (outs ++ [s.t.outputs.size]).getD p.n 0 := by
      rw [hn, List.getD_append_right _ _ _ _ (le_refl _)]; simpa using ho
    have hcg : g = ⟨(snoc p s src valid).copyGate p.n o, ((snoc p s src valid).copyGate_spec hv ho').2⟩ :=
      Fin.ext (by simp [copyGate, hgo])
    have hi : (⟨s.out o, hwl⟩ : Fin s.R.w) = s.R.comp ⟨s.out o - s.R.nIn, by rw [s.nIn_eq, hnc]; omega⟩ :=
      Fin.ext (by simp [Rows.comp, s.nIn_eq]; omega)
    have hrow : (Prog.colWire s.R (p.insOf s src valid) (p.oneOf hk) ⟨s.out o, hwl⟩).val =
        p.rowsN + (s.out o - s.t.inBits) := by
      rw [hi, Prog.colWire_comp]
      simp [s.nIn_eq]
    refine ⟨o, ho, Fin.ext ?_, ⟨s.out o, hwl⟩, rfl, Fin.ext ?_, ?_, ?_⟩
    · change g.val = (p.wireF hk s src (s.fsize + o)).val
      rw [p.wireF_val hk valid (by omega), hgo]
      unfold wireB
      rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    · change ((snoc p s src valid).mapG hk g).val = (Prog.colWire s.R _ _ ⟨s.out o, hwl⟩).val
      rw [hcg, (snoc p s src valid).mapG_copy hk hv ho', hrow]
      simp [outGate]
    · classical
      refine Partition.mem_committed_of_output _ ?_
      change _ ∈ outsC _
      rw [mem_outsC]
      refine ⟨p.n, o, hv, ho', ?_⟩
      change (Prog.colWire s.R _ _ ⟨s.out o, hwl⟩).val = _
      rw [hrow]
      simp [outGate]
    · rw [hi]
      exact IsRowsUnit.mem _ _

/-- **An earlier unit's pair, after a unit is appended**: the same derivation, its instances shifted, and its committed
gates as before. -/
noncomputable def snocPair (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) {v : Fin p.n}
    (U : UnitPair (p.P hk) (p.PB hk) (p.mapG hk) v) :
    UnitPair ((snoc p s src valid).P hk) ((snoc p s src valid).PB hk) ((snoc p s src valid).mapG hk)
      v.castSucc where
  types := U.types
  ls := U.ls
  info := U.info
  words := U.words
  idx := U.idx
  done := U.done
  hd := U.hd
  t := U.t
  ht := U.ht
  ord := U.ord
  rows := (Prog.rowsWire_snoc (p.toRows hk) s.R _ _ _ (rowsWireOf U.rows)).toIsRowsUnit _
  loOf := U.loOf
  F := U.F
  hF := U.hF
  gates := gatesSnoc hk p s src valid U.gates
  ins c hc := by
    apply Fin.ext
    change ((snoc p s src valid).mapG hk (Fin.castAdd _ (U.gates.wire c.val))).val = (Fin.castAdd _ (U.rows.wire c)).val
    simp only [mapG, Fin.val_castAdd, (U.gates.wire c.val).isLt, dite_true, Fin.eta]
    rw [U.ins c hc]
  one := by
    apply Fin.ext
    change ((snoc p s src valid).mapG hk (Fin.castAdd _ (U.gates.wire U.t.inBits))).val =
      (Fin.castAdd _ (U.rows.wire (Rows.compose (ofBlock U.words U.done (U.done.getD U.idx default)) U.ord).one)).val
    simp only [mapG, Fin.val_castAdd, (U.gates.wire U.t.inBits).isLt, dite_true, Fin.eta]
    rw [U.one]
  out g hg hgu := by
    have hlt := lt_of_unitB_castSucc hgu
    have hc := p.committedB_old hk s src valid hg hgu
    have hu : (p.PB hk).unit ⟨g.val, hlt⟩ = some v := by
      change (snoc p s src valid).unitB g = _ at hgu
      rw [unitB_old hlt] at hgu
      obtain ⟨v', hv', hvv⟩ := Option.map_eq_some_iff.1 hgu
      rw [Fin.castSucc_inj.1 hvv] at hv'
      exact hv'
    obtain ⟨o, ho, hgo, ℓ, hℓ, hm, hcom, hℓu⟩ := U.out _ hc hu
    refine ⟨o, ho, Fin.ext ?_, ℓ, hℓ, Fin.ext ?_, ?_, ?_⟩
    · change g.val = (U.gates.wire _).val
      rw [← hgo]
    · change ((snoc p s src valid).mapG hk g).val = (U.rows.wire ℓ).val
      simp only [mapG, hlt, dite_true, Fin.val_castAdd]
      rw [hm]
    · exact p.committedC_old hk s src valid hcom
    · change (Prog.snoc (p.toRows hk) s.R _ _ _).unit (Fin.castAdd _ (U.rows.wire ℓ)) = _
      rw [Prog.unit_castAdd]
      change (p.P hk).unit (U.rows.wire ℓ) = some v at hℓu
      change (p.toRows hk).unit (U.rows.wire ℓ) = some v at hℓu
      rw [hℓu]
      rfl

/-- **One pair per unit.** -/
noncomputable def pairs (hk : one < k) : {outs : List ℕ} → (p : UProg k one outs) → (u : Fin p.n) →
    UnitPair (p.P hk) (p.PB hk) (p.mapG hk) u
  | _, .inputs, u => u.elim0
  | _, .snoc p s src valid, u =>
    Fin.lastCases (motive := fun u => UnitPair ((snoc p s src valid).P hk) ((snoc p s src valid).PB hk)
        ((snoc p s src valid).mapG hk) u)
      (lastPair hk p s src valid) (fun v => snocPair hk p s src valid (pairs hk p v)) u

theorem pairs_oneGate (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n),
    ((p.pairs hk u).oneGate).val = one
  | _, .inputs, u => u.elim0
  | _, .snoc p s src valid, u => by
    refine Fin.lastCases ?_ (fun v => ?_) u
    · have e : (snoc p s src valid).pairs hk (Fin.last p.n) = lastPair hk p s src valid := by
        unfold pairs; exact Fin.lastCases_last
      rw [e]
      change (Prog.colWire s.R (p.insOf s src valid) (p.oneOf hk) s.R.one).val = one
      rw [Prog.colWire_val]
      simp [Rows.one, oneOf]
    · have e : (snoc p s src valid).pairs hk v.castSucc = snocPair hk p s src valid (p.pairs hk v) := by
        rw [pairs, Fin.lastCases_castSucc]
      rw [e]
      change (Fin.castAdd _ (UnitPair.oneGate (p.pairs hk v))).val = one
      rw [Fin.val_castAdd]
      exact pairs_oneGate hk p v

/-- **`RowsL1` for every program of units.** At values carrying 1 on `one`, every unit wrong on the program's Boolean
circuit, its values read off the rows circuit through `mapG`, is wrong on its rows. -/
theorem rowsL1 (hk : one < k) {outs : List ℕ} (p : UProg k one outs) :
    (p.P hk).RowsL1 (p.PB hk) (fun vals => vals ∘ p.mapG hk) {⟨one, Nat.lt_of_lt_of_le hk p.k_le_rowsN⟩} :=
  rowsL1_of_pairs _ _ _ _ (p.pairs hk) fun u => Finset.mem_singleton.2 (Fin.ext (p.pairs_oneGate hk u))

end UProg

end FlockSoundness
