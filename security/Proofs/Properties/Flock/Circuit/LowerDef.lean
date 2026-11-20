module

public import Properties.Flock.EvalBool
public import Proofs.Flock.Soundness.TmplValue
public import Proofs.Properties.Flock.Decoder.Walk

/-!
# What `lowerDef` writes computes the Definition it lowers

A `Form` is a GF(2) sum of atoms; at values `V` of the atoms (own column `c` at `2c`, input `i` at `2i + 1`) it is
`fval`. A copy of a finished Definition gives the atoms their values (`atomV`), at which its `Work`'s rows hold
(`rowsHold_finish`). A `Work`'s node outputs are the walk's node values (`FormsOk`) when each reads as `walkStep`'s value
of its node, so the form of every reference is the walk's value of it (`formOf_val`, `opsOf_val`; a reference is read by
its part as the walk reads it in order, `refAt_eq`). Node by node:

* the gate case (`gate_node`, `op_val`): a primitive node whose callee is a §9 gate adds the form `Work.op` returns,
  the gate on its operands' values (`eval_gate`, the walk's), an AND of more than two inputs through the rows
  `Work.and` writes, left to right;
* the call case (`call_node`, `binds_val`, `call_copy`): a call node binds each callee input to an atom holding the
  walk's argument leaf, and its copy of the callee, which satisfies the callee's equations on the bound values,
  gives the node's leaves at its return columns.
* the batch case (`batch_node`, `batch_forms`, `pieces_spec`, `gateTmpl`): a batch node's members are copies one region
  after the work, member `m`'s a copy of the CALL of its piece: a stepping piece's bindings step through one affine part
  of each mapped argument (`pieceEnd_spec`, `step_input`), a single member's commit its references, and an argument not
  mapped is committed once for all (`SharedOk`). Its leaves are the members' returns one after another; a primitive's
  template Definition computes its gate (`gateTmpl`).
* the scan case (`scan_node`, `scan_forms`, `scanIns0_eq`, `scanTail_eq`): a scan of `n` iterations is `n` copies one
  region after the work, iteration 0's a copy of the first CALL, binding the carry to `init` and each other argument to
  its iteration-0 leaf, and iteration `m + 1`'s a copy of the second, binding the carry to iteration `m`'s returns and
  an argument past the carry to its iteration-`m + 1` leaf (`stepping_spec`). Its leaves are the last iteration's carry,
  then each iteration's rest.

`rowsHold_finish` takes each row's column inside its Definition's (`hcol`), a clause of `Def.wellFormed`, which `lower`'s
`selfCheck` holds every template it writes to. The gate, call and batch cases rest on no refusal; the batches `lower`
refuses (a mapped argument stepping through strided or grid references, or a binding stepping across a gate's leaf or a
scan's carry) are outside what these cover. The scan case covers only the scans `Work.scan` accepts: it refuses a scan of
no iterations, a scan without its carry's init, and a scan whose callee returns fewer leaves than its carry, in each of
which the carry's bindings are not the walk's inputs.
-/

@[expose] public section

namespace Flock.Chain.LowerDef

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue
open Flock.Program hiding Form

/-- A form's value: the XOR of its atoms' values, and the constant 1 if it holds it. -/
def fval (V : ℕ → Bool) (f : Form) : Bool := xorAt f.atoms.toList V ^^ f.one

/-- Every row `(c, a, b)` of `w` holds at `V`: own column `c`'s atom is `a`'s value AND `b`'s. -/
def RowsHold (V : ℕ → Bool) (w : Work) : Prop :=
  ∀ r ∈ w.rows.toList, V (2 * r.1) = (fval V r.2.1 && fval V r.2.2)

theorem rowsHold_of_prefix {V : ℕ → Bool} {w w' : Work} (hp : w.rows.toList <+: w'.rows.toList)
    (h : RowsHold V w') : RowsHold V w := fun r hr => h r (hp.subset hr)

theorem val_add (V : ℕ → Bool) (f g : Form) : fval V (f.add g) = (fval V f ^^ fval V g) := by
  simp only [fval, Form.add, xorAt_canon, Array.toList_append, xorAt_append]
  generalize xorAt f.atoms.toList V = x
  generalize xorAt g.atoms.toList V = y
  cases x <;> cases y <;> cases f.one <;> cases g.one <;> rfl

theorem val_const (V : ℕ → Bool) (b : Bool) : fval V (Form.const b) = b := by
  simp [fval, Form.const, xorAt]

theorem val_empty (V : ℕ → Bool) : fval V (({} : Form)) = false := by
  simp [fval, xorAt]

theorem val_own (V : ℕ → Bool) (c : ℕ) : fval V (Form.own c) = V (2 * c) := by
  simp [fval, Form.own, xorAt]

theorem val_of_isEmpty {V : ℕ → Bool} {f : Form} (h : f.atoms.isEmpty = true) : fval V f = f.one := by
  have : f.atoms.toList = [] := by simpa [Array.isEmpty_iff] using h
  simp [fval, this, xorAt]

theorem and_prefix (w : Work) (a b : Form) : w.rows.toList <+: (w.and a b).1.rows.toList := by
  unfold Work.and
  split
  · exact List.prefix_refl _
  split
  · exact List.prefix_refl _
  simp only [Work.row, Array.toList_push]
  exact List.prefix_append _ _

/-- **An AND of two forms**: its form's value, where the rows hold, is the AND of theirs. -/
theorem and_val {V : ℕ → Bool} (w : Work) (a b : Form) (hV : RowsHold V (w.and a b).1) :
    fval V (w.and a b).2 = (fval V a && fval V b) := by
  unfold Work.and at hV ⊢
  split
  · rename_i ha
    rw [val_of_isEmpty ha]
    cases a.one
    · simp [val_empty]
    · simp
  split
  · rename_i _ hb
    rw [val_of_isEmpty hb]
    cases b.one
    · simp [val_empty]
    · simp
  rename_i ha hb
  rw [ite_eq_right ha, ite_eq_right hb] at hV
  simp only [Work.row] at hV ⊢
  rw [val_own]
  exact hV (w.next, a, b) (by simp)

theorem andFold (V : ℕ → Bool) :
    ∀ (l : List Form) (acc : Work × Form),
      RowsHold V (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).1 →
      acc.1.rows.toList <+: (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).1.rows.toList ∧
        fval V (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).2 =
          (fval V acc.2 && (l.map (fval V)).all id)
  | [], acc, _ => ⟨List.prefix_refl _, by simp⟩
  | o :: l, acc, hV => by
    simp only [List.foldl_cons] at hV ⊢
    obtain ⟨hp, hv⟩ := andFold V l _ hV
    have hp1 := and_prefix acc.1 acc.2 o
    refine ⟨hp1.trans hp, ?_⟩
    rw [hv, and_val _ _ _ (rowsHold_of_prefix hp hV)]
    simp [Bool.and_assoc]

theorem xorFold (V : ℕ → Bool) :
    ∀ (l : List Form) (acc : Form),
      fval V (l.foldl Form.add acc) = l.foldl (fun b o => b ^^ fval V o) (fval V acc)
  | [], _ => rfl
  | o :: l, acc => by
    simp only [List.foldl_cons]
    rw [xorFold V l, val_add]

theorem toList_of_getElem?_zero {ops : Array Form} {o : Form} (h : ops[0]? = some o) :
    ops.toList = o :: (ops.extract 1 ops.size).toList := by
  rw [Array.toList_extract, List.extract_eq_take_drop, List.take_of_length_le (by simp)]
  cases hl : ops.toList with
  | nil => simp [← Array.getElem?_toList, hl] at h
  | cons a l =>
    simp only [← Array.getElem?_toList, hl, List.getElem?_cons_zero, Option.some.injEq] at h
    simp [h]

/-- **The gate case**: where `w'`'s rows hold, the form `Work.op` returns is the gate on its operands' values. -/
theorem op_val {V : ℕ → Bool} {w w' : Work} {g : Gate} {ops : Array Form} {f : Form}
    (h : w.op g ops = .ok (w', f)) (hV : RowsHold V w') :
    w.rows.toList <+: w'.rows.toList ∧ fval V f = gateVal g (ops.toList.map (fval V)) := by
  cases g with
  | const_ b =>
    simp only [Work.op, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨List.prefix_refl _, by rw [val_const]; rfl⟩
  | not_ =>
    simp only [Work.op] at h
    cases ho : ops[0]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      simp only at h
      by_cases hs : ops.size = 1
      · rw [ite_eq_left (by simp [hs])] at h
        simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        have hl := toList_of_getElem?_zero ho
        have he : (ops.extract 1 ops.size).toList = [] := by simp [hs]
        refine ⟨List.prefix_refl _, ?_⟩
        rw [val_add, val_const, hl, he]
        simp [gateVal]
      · rw [ite_eq_right (by simp [hs])] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
  | xor_ =>
    simp only [Work.op] at h
    by_cases hs : ops.size < 2
    · rw [ite_eq_left hs] at h
      simp [throw, throwThe, MonadExceptOf.throw] at h
    · rw [ite_eq_right hs] at h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      refine ⟨List.prefix_refl _, ?_⟩
      rw [← Array.foldl_toList, xorFold, val_empty]
      simp [gateVal, List.foldl_map]
  | and_ =>
    simp only [Work.op] at h
    cases ho : ops[0]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      simp only at h
      by_cases hs : ops.size < 2
      · rw [ite_eq_left hs] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
      · rw [ite_eq_right hs] at h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        rw [← Array.foldl_toList] at h
        have hf := andFold V (ops.extract 1 ops.size).toList (w, o) (by rw [h]; exact hV)
        rw [h] at hf
        refine ⟨hf.1, ?_⟩
        rw [hf.2, toList_of_getElem?_zero ho]
        simp [gateVal]

/-! ## A reference sequence, read by its part -/

theorem seqOk_eq {q : Seq} (h : seqOk q = true) : Seq.ofParts q.parts = q := by
  obtain ⟨parts, starts, count⟩ := q
  unfold seqOk at h
  unfold Seq.ofParts
  simp only at h ⊢
  generalize parts.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.count)) (#[], 0) = r at h ⊢
  obtain ⟨s', c'⟩ := r
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨rfl, rfl⟩ := h
  rfl

theorem foldl_push_map {α β : Type} (g : α → β) :
    ∀ (l : List α) (acc : Array β), l.foldl (fun acc k => acc.push (g k)) acc = acc ++ (l.map g).toArray
  | [], acc => by simp
  | a :: l, acc => by
    simp only [List.foldl_cons, List.map_cons]
    rw [foldl_push_map g l]
    apply Array.toList_inj.mp
    simp

theorem toArray_toList (q : Seq) :
    q.toArray.toList = (q.parts.toList.map fun p => (List.range p.count).map p.get).flatten := by
  unfold Seq.toArray
  rw [← Array.foldl_toList]
  generalize q.parts.toList = L
  suffices h : ∀ acc : Array Ref,
      (L.foldl (fun acc p => (Array.range p.count).foldl (fun acc k => acc.push (p.get k)) acc) acc).toList =
        acc.toList ++ (L.map fun p => (List.range p.count).map p.get).flatten by
    simpa using h #[]
  induction L with
  | nil => intro acc; simp
  | cons p L ih =>
    intro acc
    simp only [List.foldl_cons, List.map_cons, List.flatten_cons]
    rw [ih, ← Array.foldl_toList, Array.toList_range, foldl_push_map]
    simp

theorem getElem?_flatten_of {α : Type} :
    ∀ (L : List (List α)) (x k : ℕ) (hx : x < L.length), ((L.take x).map List.length).sum ≤ k →
      k < ((L.take x).map List.length).sum + L[x].length →
      L.flatten[k]? = L[x][k - ((L.take x).map List.length).sum]?
  | [], _, _, hx, _, _ => absurd hx (by simp)
  | l :: L, 0, k, _, _, h2 => by
    simp only [List.take_zero, List.map_nil, List.sum_nil, Nat.zero_add, List.getElem_cons_zero,
      Nat.sub_zero] at h2 ⊢
    rw [List.flatten_cons, List.getElem?_append_left h2]
  | l :: L, x + 1, k, hx, h1, h2 => by
    simp only [List.take_succ_cons, List.map_cons, List.sum_cons, List.getElem_cons_succ] at h1 h2 ⊢
    rw [List.flatten_cons, List.getElem?_append_right (by omega),
      getElem?_flatten_of L x (k - l.length) (by simpa using hx) (by omega) (by omega)]
    congr 1
    omega

theorem partIx_spec (q : Seq) (k : ℕ) (h0 : q.starts[0]! ≤ k) :
    ∀ n, 0 < n → (List.range n).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0 < n ∧
      q.starts[(List.range n).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0]! ≤ k ∧
      ∀ y, (List.range n).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0 < y → y < n → k < q.starts[y]!
  | 0, h => absurd h (lt_irrefl 0)
  | n + 1, _ => by
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp only [List.range_zero, List.foldl_nil, ite_self]
      exact ⟨Nat.one_pos, h0, fun y h1 h2 => by omega⟩
    · obtain ⟨h1, h2, h3⟩ := partIx_spec q k h0 n hn
      by_cases hs : q.starts[n]! ≤ k
      · rw [ite_eq_left hs]
        exact ⟨by omega, hs, fun y h h' => by omega⟩
      · rw [ite_eq_right hs]
        refine ⟨by omega, h2, fun y hy hy' => ?_⟩
        rcases Nat.lt_or_ge y n with hl | hg
        · exact h3 y hy hl
        · have : y = n := by omega
          subst this
          omega

/-- **`lower` reads a reference where the walk does**: reference `k` of a sequence whose starts are its parts' running
sums (`seqOk`), found by its part (`refAt`), is the sequence's `k`-th in order (`Seq.toArray`). -/
theorem refAt_eq {q : Seq} (hq : seqOk q = true) {k : ℕ} (hk : k < q.count) :
    refAt q k = q.toArray.getD k default := by
  have he := seqOk_eq hq
  obtain ⟨_, hsz, hst, hcnt⟩ := InferenceOnlyVerifier.Properties.SecurityProofs.Walk.ofParts_spec q.parts
  rw [he] at hsz hst hcnt
  have hpos : 0 < q.parts.size := by
    rcases Nat.eq_zero_or_pos q.parts.size with h | h
    · have : q.parts.toList = [] := by simpa [← Array.length_toList] using h
      rw [hcnt, this] at hk
      simp at hk
    · exact h
  have h0 : q.starts[0]! = 0 := by rw [hst 0 hpos]; simp
  obtain ⟨r1, r2, r3⟩ := partIx_spec q k (by omega) q.parts.size hpos
  set r := (List.range q.parts.size).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0 with hr
  have hlen : ∀ i < q.parts.size, q.starts[i]! + q.parts[i]!.count = ((q.parts.toList.take (i + 1)).map Part.count).sum :=
    fun i hi => by rw [hst i hi, InferenceOnlyVerifier.Properties.SecurityProofs.Walk.sum_take_succ_map q.parts hi]
  have hin : k < q.starts[r]! + q.parts[r]!.count := by
    rcases Nat.lt_or_ge (r + 1) q.parts.size with hl | hg
    · have := r3 (r + 1) (by omega) hl
      rw [hst (r + 1) hl] at this
      rw [hlen r r1]
      exact this
    · have hre : r + 1 = q.parts.toList.length := by simp; omega
      rw [hlen r r1, hre, List.take_length, ← hcnt]
      exact hk
  have hS : ((((q.parts.toList.map fun p => (List.range p.count).map p.get)).take r).map List.length).sum =
      q.starts[r]! := by
    rw [hst r r1, ← List.map_take, List.map_map]
    congr 1
    apply List.map_congr_left
    intro p _
    simp
  have hrl : r < (q.parts.toList.map fun p => (List.range p.count).map p.get).length := by simpa using r1
  have hpr : q.parts[r]! = q.parts.toList[r]'(by simpa using r1) := by
    rw [getElem!_pos q.parts r r1]; simp
  unfold refAt partIx
  rw [← hr, Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, toArray_toList,
    getElem?_flatten_of _ r k hrl (by rw [hS]; exact r2) (by rw [hS]; simpa [hpr] using hin), hS]
  simp only [List.getElem_map]
  rw [List.getElem?_map, List.getElem?_range (by rw [← hpr]; omega)]
  simp [hpr]

/-! ## A copy of a finished Definition, as atom values -/

/-- A copy of finished Definition `D` as its `Work`'s atom values: atom `2c` its column `c`'s reference, `2i + 1` its
input `i`'s, read through `Def.refV` with the constant 1. -/
def atomV (D : Def) (v : ℕ → Bool) (inp : List Bool) (x : ℕ) : Bool :=
  D.refV v inp true (if x % 2 == 0 then x / 2 else D.cols + x / 2)

theorem finish_cols (w : Work) (ins : ℕ) : (w.finish ins).cols = w.next := rfl

theorem finish_ins (w : Work) (ins : ℕ) : (w.finish ins).ins = ins := rfl

theorem val_input (V : ℕ → Bool) (i : ℕ) : fval V (Form.input i) = V (2 * i + 1) := by
  simp [fval, Form.input, xorAt]

theorem atomV_own {D : Def} {v : ℕ → Bool} {inp : List Bool} {c : ℕ} (hc : c < D.cols) :
    atomV D v inp (2 * c) = v c := by
  simp [atomV, Def.refV, hc]

theorem atomV_input {D : Def} {v : ℕ → Bool} {inp : List Bool} {i : ℕ} (hi : i < D.ins) :
    atomV D v inp (2 * i + 1) = inp.getD i true := by
  have h1 : (2 * i + 1) % 2 = 1 := by omega
  have h2 : (2 * i + 1) / 2 = i := by omega
  simp [atomV, Def.refV, h1, h2, hi]

/-- The references `Work.finish` writes for form `f`, its columns `cols` and inputs `ins`. -/
def refsOf (cols ins : ℕ) (f : Form) : Array ℕ :=
  Tmpl.canon (f.atoms.map (fun x => if x % 2 == 0 then x / 2 else cols + x / 2) ++ (if f.one then #[cols + ins] else #[]))

theorem xorAt_refsOf (w : Work) (ins : ℕ) (v : ℕ → Bool) (inp : List Bool) (f : Form) :
    xorAt (refsOf w.next ins f).toList ((w.finish ins).refV v inp true) = fval (atomV (w.finish ins) v inp) f := by
  unfold refsOf fval
  rw [xorAt_canon, Array.toList_append, xorAt_append, Array.toList_map, xorAt_map]
  congr 1
  cases f.one
  · simp [xorAt]
  · simp [xorAt, Def.refV, finish_cols, finish_ins]

theorem mem_finish_rows {w : Work} {ins : ℕ} {r : ℕ × Form × Form} (hr : r ∈ w.rows.toList) :
    (r.1, refsOf w.next ins r.2.1, refsOf w.next ins r.2.2) ∈ (w.finish ins).rows.toList := by
  obtain ⟨c, a, b⟩ := r
  simp only [Work.finish, Array.toList_map, List.mem_map]
  exact ⟨(c, a, b), hr, rfl⟩

/-- **The rows hold at a copy's atom values**: where the copy's columns satisfy its finished Definition's rows, each
row column inside its columns, the `Work`'s rows hold at `atomV`. -/
theorem rowsHold_finish {w : Work} {ins : ℕ} {v : ℕ → Bool} {inp : List Bool}
    (hcol : ∀ r ∈ (w.finish ins).rows.toList, r.1 < (w.finish ins).cols)
    (hH : ∀ r ∈ (w.finish ins).rows.toList, v r.1 = rowVal ((w.finish ins).refV v inp true) r) :
    RowsHold (atomV (w.finish ins) v inp) w := by
  intro r hr
  have hm := mem_finish_rows (ins := ins) hr
  rw [atomV_own (hcol _ hm), hH _ hm]
  simp only [rowVal, xorAt_refsOf]

/-! ## Node outputs, as the walk's node values -/

/-- Node output `o`'s leaves at atom values `V` are `l`'s, at every leaf `Work.formOf` reads of it. -/
def OutOk (V : ℕ → Bool) : Out → List Bool → Prop
  | .gate f, l => l.getD 0 false = fval V f
  | .copies base stride n rets, l => ∀ j < n * rets.size,
      V (2 * (base + j / rets.size * stride + rets[j % rets.size]!)) = l.getD j false
  | .scan base stride rets c n, l => ∀ j < c + n * (rets.size - c),
      (j < c → V (2 * (base + (n - 1) * stride + rets[j]!)) = l.getD j false) ∧
      (c ≤ j → V (2 * (base + (j - c) / (rets.size - c) * stride + rets[c + (j - c) % (rets.size - c)]!)) =
        l.getD j false)

/-- `w`'s node outputs are the walk's node values `vals`, one each. -/
def FormsOk (V : ℕ → Bool) (w : Work) (vals : Array (List Bool)) : Prop :=
  w.outs.size = vals.size ∧ ∀ i (h : i < w.outs.size), OutOk V w.outs[i] (vals.getD i [])

/-- The atom of each of `ins` inputs is the parameter bit the walk reads. -/
def ParamsOk (V : ℕ → Bool) (ins : ℕ) (xs : List Bool) : Prop := ∀ k < ins, V (2 * k + 1) = xs.getD k false

theorem formsOk_nil (V : ℕ → Bool) (ins : ℕ) : FormsOk V { ins } #[] := ⟨rfl, fun _ h => absurd h (by simp)⟩

theorem getD_push_lt {α : Type} {a : Array α} {x d : α} {i : ℕ} (h : i < a.size) :
    (a.push x).getD i d = a.getD i d := by
  simp [Array.getD_eq_getD_getElem?, Array.getElem?_push, Nat.ne_of_lt h]

theorem getD_push_size {α : Type} (a : Array α) (x d : α) : (a.push x).getD a.size d = x := by
  simp [Array.getD_eq_getD_getElem?]

theorem formsOk_push {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} {o : Out} {l : List Bool}
    (hF : FormsOk V w vals) (hout : w'.outs = w.outs) (ho : OutOk V o l) :
    FormsOk V { w' with outs := w'.outs.push o } (vals.push l) := by
  obtain ⟨hs, hi⟩ := hF
  refine ⟨by simp [hout, hs], fun i hi' => ?_⟩
  simp only [hout, Array.size_push] at hi'
  simp only [hout]
  rcases Nat.lt_or_ge i w.outs.size with hlt | hge
  · rw [Array.getElem_push_lt hlt, getD_push_lt (by omega)]
    exact hi i hlt
  · have : i = w.outs.size := by omega
    subst this
    rw [Array.getElem_push_eq, hs, getD_push_size]
    exact ho

/-- **A reference's form is the walk's value of it**: where the node outputs are the walk's node values and the input
atoms its parameter bits, the form `Work.formOf` gives a reference has the value `refVal` gives it. -/
theorem formOf_val {V : ℕ → Bool} {w : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {r : Ref} {f : Form} (h : w.formOf st r = .ok f) :
    fval V f = refVal xs st vals r := by
  unfold Work.formOf at h
  unfold refVal
  by_cases hp : r.p = true
  · rw [ite_eq_left hp] at h ⊢
    cases hs : st[r.i]? with
    | none => rw [hs] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some s =>
      rw [hs] at h
      simp only at h
      by_cases hl : s + r.j < w.ins
      · rw [ite_eq_left hl] at h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        rw [val_input, hP _ hl, Array.getD_eq_getD_getElem?, hs]
        rfl
      · rw [ite_eq_right hl] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
  · rw [ite_eq_right hp] at h ⊢
    cases ho : w.outs[r.i]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      obtain ⟨hi, heq⟩ := Array.getElem?_eq_some_iff.mp ho
      have hO := hF.2 r.i hi
      rw [heq] at hO
      show _ = (vals.getD r.i []).getD r.j false
      generalize vals.getD r.i [] = l at hO ⊢
      cases o with
      | gate g =>
        simp only at h
        by_cases hj : r.j = 0
        · rw [ite_eq_left (by simp [hj])] at h
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          rw [hj]
          exact hO.symm
        · rw [ite_eq_right (by simp [hj])] at h
          simp [throw, throwThe, MonadExceptOf.throw] at h
      | copies base stride n rets =>
        simp only at h
        by_cases he : rets.isEmpty = true
        · rw [ite_eq_left he] at h
          simp [throw, throwThe, MonadExceptOf.throw] at h
        · rw [ite_eq_right he] at h
          by_cases hl : r.j < n * rets.size
          · rw [ite_eq_left hl] at h
            simp only [pure, Except.pure, Except.ok.injEq] at h
            subst h
            rw [val_own]
            exact hO r.j hl
          · rw [ite_eq_right hl] at h
            simp [throw, throwThe, MonadExceptOf.throw] at h
      | scan base stride rets c n =>
        simp only at h
        by_cases h1 : c + n * (rets.size - c) ≤ r.j
        · rw [ite_eq_left h1] at h
          simp [throw, throwThe, MonadExceptOf.throw] at h
        · rw [ite_eq_right h1] at h
          have hlt : r.j < c + n * (rets.size - c) := by omega
          by_cases h2 : r.j < c
          · rw [ite_eq_left h2] at h
            simp only [pure, Except.pure, Except.ok.injEq] at h
            subst h
            rw [val_own]
            exact (hO r.j hlt).1 h2
          · rw [ite_eq_right h2] at h
            by_cases h3 : (rets.size - c == 0) = true
            · rw [ite_eq_left h3] at h
              simp [throw, throwThe, MonadExceptOf.throw] at h
            · rw [ite_eq_right h3] at h
              simp only [pure, Except.pure, Except.ok.injEq] at h
              subst h
              rw [val_own]
              exact (hO r.j hlt).2 (by omega)

theorem getRef_ok {q : Seq} {t : ℕ} {r : Ref} (h : getRef q t = .ok r) : t < q.count ∧ r = refAt q t := by
  unfold getRef at h
  by_cases ht : t < q.count
  · rw [ite_eq_left ht] at h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact ⟨ht, h.symm⟩
  · rw [ite_eq_right ht] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

theorem opsOf_inner {V : ℕ → Bool} {w : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {q : Seq} (hq : seqOk q = true) :
    ∀ (L : List ℕ) (ops ops' : Array Form),
      L.foldlM (fun ops t => do return ops.push (← w.formOf st (← getRef q t))) ops = .ok ops' →
      ops'.toList.map (fval V) =
        ops.toList.map (fval V) ++ L.map (fun t => refVal xs st vals (q.toArray.getD t default))
  | [], ops, ops', h => by
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | t :: L, ops, ops', h => by
    simp only [List.foldlM_cons] at h
    cases hr : getRef q t with
    | error e => simp [hr, bind, Except.bind] at h
    | ok r =>
      cases hfo : w.formOf st r with
      | error e => simp [hr, hfo, bind, Except.bind] at h
      | ok f0 =>
        simp only [hr, hfo, bind, Except.bind, pure, Except.pure] at h
        rw [opsOf_inner hF hP hq L _ _ h]
        obtain ⟨htc, rfl⟩ := getRef_ok hr
        rw [Array.toList_push, List.map_append, List.map_cons, List.map_nil, formOf_val hF hP hfo,
          refAt_eq hq htc]
        simp

/-- **A primitive node's operands are the walk's**: `Work.opsOf`'s forms have the values of the walk's operands, every
reference of every argument in order. -/
theorem opsOf_val {V : ℕ → Bool} {w : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) :
    ∀ (args : List Arg) (ops ops' : Array Form), (∀ a ∈ args, seqOk a.seq = true) →
      args.foldlM (fun ops a =>
        (List.range a.seq.count).foldlM (fun ops t => do return ops.push (← w.formOf st (← getRef a.seq t))) ops)
        ops = .ok ops' →
      ops'.toList.map (fval V) =
        ops.toList.map (fval V) ++ args.flatMap (fun a => argVals (refVal xs st vals) a a.seq.count id)
  | [], ops, ops', _, h => by
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | a :: args, ops, ops', hs, h => by
    rw [List.foldlM_cons] at h
    cases hi : (List.range a.seq.count).foldlM
        (fun ops t => do return ops.push (← w.formOf st (← getRef a.seq t))) ops with
    | error e => rw [hi] at h; exact absurd h (by simp [bind, Except.bind])
    | ok o1 =>
      rw [hi] at h
      simp only [bind, Except.bind] at h
      rw [opsOf_val hF hP args o1 ops' (fun b hb => hs b (List.mem_cons_of_mem _ hb)) h,
        opsOf_inner hF hP (hs a (by simp)) _ ops o1 hi]
      simp [argVals]

theorem and_outs (w : Work) (a b : Form) : (w.and a b).1.ins = w.ins ∧ (w.and a b).1.outs = w.outs := by
  unfold Work.and
  split
  · exact ⟨rfl, rfl⟩
  split
  · exact ⟨rfl, rfl⟩
  exact ⟨rfl, rfl⟩

theorem andFold_outs :
    ∀ (l : List Form) (acc : Work × Form),
      (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).1.ins = acc.1.ins ∧
        (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).1.outs = acc.1.outs
  | [], _ => ⟨rfl, rfl⟩
  | o :: l, acc => by
    simp only [List.foldl_cons]
    obtain ⟨h1, h2⟩ := andFold_outs l (acc.1.and acc.2 o)
    obtain ⟨h3, h4⟩ := and_outs acc.1 acc.2 o
    exact ⟨h1.trans h3, h2.trans h4⟩

theorem op_outs {w w' : Work} {g : Gate} {ops : Array Form} {f : Form} (h : w.op g ops = .ok (w', f)) :
    w'.ins = w.ins ∧ w'.outs = w.outs := by
  cases g with
  | const_ b =>
    simp only [Work.op, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, rfl⟩
  | not_ =>
    simp only [Work.op] at h
    cases ho : ops[0]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      simp only at h
      split at h
      · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨rfl, rfl⟩
      · simp [throw, throwThe, MonadExceptOf.throw] at h
  | xor_ =>
    simp only [Work.op] at h
    split at h
    · simp [throw, throwThe, MonadExceptOf.throw] at h
    · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨rfl, rfl⟩
  | and_ =>
    simp only [Work.op] at h
    cases ho : ops[0]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      simp only at h
      split at h
      · simp [throw, throwThe, MonadExceptOf.throw] at h
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        rw [← Array.foldl_toList] at h
        have := andFold_outs (ops.extract 1 ops.size).toList (w, o)
        rw [h] at this
        exact this

/-- **The gate case, a node**: a primitive node whose callee is a §9 gate `g` keeps `w`'s node outputs the walk's, its
new output the gate on the walk's operands, wherever the rows the node writes hold. -/
theorem gate_node {V : ℕ → Bool} {w w' : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    {x : Node} {g : Gate} {ops : Array Form} {f : Form}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hops : w.opsOf st x = .ok ops) (hop : w.op g ops = .ok (w', f)) (hV : RowsHold V w') :
    w.rows.toList <+: w'.rows.toList ∧ w'.ins = w.ins ∧
      FormsOk V { w' with outs := w'.outs.push (.gate f) }
        (vals.push [gateVal g (x.args.toList.flatMap fun a => argVals (refVal xs st vals) a a.seq.count id)]) := by
  obtain ⟨hpre, hval⟩ := op_val hop hV
  obtain ⟨hins, hout⟩ := op_outs hop
  have hmap := opsOf_val hF hP x.args.toList #[] ops hseq hops
  simp only [List.nil_append, List.map_nil] at hmap
  refine ⟨hpre, hins, formsOk_push hF hout ?_⟩
  show [_].getD 0 false = _
  rw [hval, hmap]
  rfl

/-- The walk's value of a primitive node whose callee is §9 gate `g`: the gate on its operands. -/
theorem eval_gate {P : Prog} {F fn : ℕ} {callee : Defn} {g : Gate} {args : List Bool}
    (hc : P.defs[fn]? = some callee) (hp : callee.prim = true) (hg : gateOf? callee.id = some g) :
    evalDef? P (F + 1) fn args = some [gateVal g args] := by
  simp [evalDef?, walkStep, hc, hp, hg, Wires.bool]

/-! ## The call case -/

theorem formOf_congr {w w' : Work} (hi : w'.ins = w.ins) (ho : w'.outs = w.outs) (st : Array ℕ) (r : Ref) :
    w'.formOf st r = w.formOf st r := by
  unfold Work.formOf
  rw [hi, ho]

theorem single?_val {V : ℕ → Bool} {f : Form} {x : ℕ} (h : f.single? = some x) : fval V f = V x := by
  unfold Form.single? at h
  by_cases hc : (f.atoms.size == 1 && !f.one) = true
  · rw [ite_eq_left hc] at h
    simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at hc
    obtain ⟨hs, ho⟩ := hc
    have hl : f.atoms.toList = [x] := by
      rcases hl : f.atoms.toList with _ | ⟨a, _ | ⟨b, l⟩⟩
      · simp [← Array.length_toList, hl] at hs
      · simp only [← Array.getElem?_toList, hl, List.getElem?_cons_zero, Option.some.injEq] at h
        rw [h]
      · simp [← Array.length_toList, hl] at hs
    simp [fval, hl, ho, xorAt]
  · rw [ite_eq_right hc] at h
    exact absurd h (by simp)

/-- `Work.commit` keeps its inputs, outputs and rows, and adds at most the row committing `f`. -/
theorem commit_frame (w : Work) (f : Form) :
    (w.commit f).1.ins = w.ins ∧ (w.commit f).1.outs = w.outs ∧ (w.commit f).1.calls = w.calls ∧
      w.rows.toList <+: (w.commit f).1.rows.toList := by
  unfold Work.commit
  split
  · exact ⟨rfl, rfl, rfl, List.prefix_refl _⟩
  · refine ⟨rfl, rfl, rfl, ?_⟩
    simp only [Work.row, Array.toList_push]
    exact List.prefix_append _ _

/-- **A committed form**: its atom's value, where the rows hold, is the form's. -/
theorem commit_val {V : ℕ → Bool} (w : Work) (f : Form) (hV : RowsHold V (w.commit f).1) :
    V (w.commit f).2 = fval V f := by
  unfold Work.commit at hV ⊢
  split
  · rename_i x hx
    exact (single?_val hx).symm
  · rename_i hx
    simp only [hx, Work.row] at hV
    have := hV (w.next, f, Form.const true) (by simp)
    rw [val_const, Bool.and_true] at this
    exact this

theorem bindsInner {V : ℕ → Bool} {w0 : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w0 vals) (hP : ParamsOk V w0.ins xs) {q : Seq} (hq : seqOk q = true) :
    ∀ (L : List ℕ) (w w' : Work) (out out' : Array Bind), w.ins = w0.ins → w.outs = w0.outs →
      L.foldlM (fun (acc : Work × Array Bind) t => do
        let (w, out) := acc
        let (r, s) ← (·, 0) <$> getRef q t
        if s == 0 then
          let (w', at_) := w.commit (← w.formOf st r)
          return (w', out.push ⟨at_, 0⟩)
        else return (w, out.push (← w.stepOf st r s 1))) (w, out) = .ok (w', out') →
      w'.ins = w0.ins ∧ w'.outs = w0.outs ∧ w'.calls = w.calls ∧ w.rows.toList <+: w'.rows.toList ∧
        (RowsHold V w' → out'.toList.map (fun b => (V b.a, b.step)) =
          out.toList.map (fun b => (V b.a, b.step)) ++
            L.map (fun t => (refVal xs st vals (q.toArray.getD t default), 0)))
  | [], w, w', out, out', hi, ho, h => by
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨hi, ho, rfl, List.prefix_refl _, fun _ => by simp⟩
  | t :: L, w, w', out, out', hi, ho, h => by
    simp only [List.foldlM_cons] at h
    cases hr : getRef q t with
    | error e => simp [hr, bind, Except.bind, Functor.map, Except.map] at h
    | ok r =>
      cases hfo : w.formOf st r with
      | error e => simp [hr, hfo, bind, Except.bind, Functor.map, Except.map] at h
      | ok f0 =>
        simp only [hr, hfo, bind, Except.bind, Functor.map, Except.map, pure, Except.pure, beq_self_eq_true,
          ite_true] at h
        obtain ⟨c1, c2, c3, c4⟩ := commit_frame w f0
        obtain ⟨h1, h2, h3, h4, h5⟩ := bindsInner hF hP hq L _ _ _ _ (c1.trans hi) (c2.trans ho) h
        refine ⟨h1, h2, h3.trans c3, c4.trans h4, fun hV => ?_⟩
        rw [h5 hV, Array.toList_push, List.map_append, List.map_cons, List.map_nil]
        obtain ⟨htc, rfl⟩ := getRef_ok hr
        have hv := commit_val w f0 (rowsHold_of_prefix h4 hV)
        rw [formOf_congr hi ho] at hfo
        rw [hv, formOf_val hF hP hfo, refAt_eq hq htc]
        simp

theorem bindsOuter {V : ℕ → Bool} {w0 : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w0 vals) (hP : ParamsOk V w0.ins xs) (x : Node) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true) :
    ∀ (I : List ℕ) (w w' : Work) (out out' : Array Bind), (∀ i ∈ I, i < x.args.size) → w.ins = w0.ins →
      w.outs = w0.outs →
      I.foldlM (fun (acc : Work × Array Bind) (i : ℕ) =>
        (List.range (x.args[i]!).seq.count).foldlM (fun (acc : Work × Array Bind) t => do
          let (w, out) := acc
          let (r, s) ← (·, 0) <$> getRef (x.args[i]!).seq t
          if s == 0 then
            let (w', at_) := w.commit (← w.formOf st r)
            return (w', out.push ⟨at_, 0⟩)
          else return (w, out.push (← w.stepOf st r s 1))) acc) (w, out) = .ok (w', out') →
      w'.ins = w0.ins ∧ w'.outs = w0.outs ∧ w'.calls = w.calls ∧ w.rows.toList <+: w'.rows.toList ∧
        (RowsHold V w' → out'.toList.map (fun b => (V b.a, b.step)) =
          out.toList.map (fun b => (V b.a, b.step)) ++
            (I.flatMap fun i => argVals (refVal xs st vals) x.args[i]! (x.args[i]!).seq.count id).map (·, 0))
  | [], w, w', out, out', _, hi, ho, h => by
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨hi, ho, rfl, List.prefix_refl _, fun _ => by simp⟩
  | i :: I, w, w', out, out', hI, hi, ho, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨⟨w1, out1⟩, h1, h2⟩ := InferenceOnlyVerifier.Properties.SecurityProofs.Walk.except_bind_ok h
    have hiq : x.args[i]! ∈ x.args.toList := by
      rw [getElem!_pos x.args i (hI i (by simp))]
      exact Array.getElem_mem_toList _
    obtain ⟨a1, a2, a3, a4, a5⟩ := bindsInner hF hP (hseq _ hiq) _ w w1 out out1 hi ho h1
    obtain ⟨b1, b2, b3, b4, b5⟩ := bindsOuter hF hP x hseq I w1 w' out1 out' (fun j hj => hI j (by simp [hj])) a1 a2 h2
    refine ⟨b1, b2, b3.trans a3, a4.trans b4, fun hV => ?_⟩
    rw [b5 hV, a5 (rowsHold_of_prefix b4 hV)]
    simp [argVals]

/-- **A call node's bindings are its arguments**: `Work.binds` of a node not a batch binds each of the callee's inputs
to an atom whose value, where the rows it commits hold, is the walk's argument leaf, stepping 0. -/
theorem binds_val {V : ℕ → Bool} {w w' : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    {callee : Defn} {x : Node} {bs : Array Bind}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hform : (x.form == .batch) = false) (hb : w.binds st callee x = .ok (w', bs)) :
    w'.ins = w.ins ∧ w'.outs = w.outs ∧ w'.calls = w.calls ∧ w.rows.toList <+: w'.rows.toList ∧
      (RowsHold V w' → bs.toList.map (fun b => (V b.a, b.step)) =
        (x.args.toList.flatMap fun a => argVals (refVal xs st vals) a a.seq.count id).map (·, 0)) := by
  unfold Work.binds at hb
  simp only [hform, Bool.false_and, Bool.false_eq_true, ite_false] at hb
  obtain ⟨h1, h2, h3, h4, h5⟩ := bindsOuter hF hP x hseq (List.range x.args.size) w w' #[] bs
    (fun i hi => List.mem_range.mp hi) rfl rfl hb
  refine ⟨h1, h2, h3, h4, fun hV => ?_⟩
  rw [h5 hV]
  have hl : x.args.toList = (List.range x.args.size).map (fun i => x.args[i]!) := by
    apply List.ext_getElem (by simp)
    intro k h1 h2
    simp only [List.getElem_map, List.getElem_range]
    rw [getElem!_pos x.args k (by simpa using h1)]
    simp
  rw [hl, List.flatMap_map]
  simp

/-- A segment `Work.finish` writes for binding `b`. -/
def segOf (ref : ℕ → ℕ) (b : Bind) : Seg := { len := 1, at_ := ref b.a, step := b.step / 2, reps := 1, rstride := 0 }

/-- `Work.finish`'s merge of a binding's segment into the segments before it. -/
def mergeSeg (cols : ℕ) (ss : Array Seg) (s : Seg) : Array Seg :=
  match ss.back? with
  | some l =>
    if l.reps == 1 && l.step == s.step && l.at_ + l.len == s.at_ && (decide (l.at_ < cols) == decide (s.at_ < cols)) then
      ss.pop.push { l with len := l.len + 1 }
    else ss.push s
  | none => ss.push s

/-- Segment `s`'s values in `Call.boundV`, for copy `j`. -/
def segVals (g : ℕ → Bool) (j : ℕ) (s : Seg) : List Bool :=
  (List.range s.reps).flatMap fun r => (List.range s.len).map fun t => g (s.at_ + r * s.rstride + j * s.step + t)

theorem toList_of_back? {α : Type} {ss : Array α} {l : α} (h : ss.back? = some l) : ss.toList = ss.pop.toList ++ [l] := by
  have h' : ss.toList.getLast? = some l := by rw [Array.getLast?_toList]; exact h
  rw [Array.toList_pop]
  exact (List.dropLast_append_getLast? l h').symm

theorem merge_bound (g : ℕ → Bool) (j cols : ℕ) (ref : ℕ → ℕ) :
    ∀ (L : List Bind) (acc : Array Seg),
      ((L.map (segOf ref)).foldl (mergeSeg cols) acc).toList.flatMap (segVals g j) =
        acc.toList.flatMap (segVals g j) ++ L.map (fun b => g (ref b.a + j * (b.step / 2)))
  | [], acc => by simp
  | b :: L, acc => by
    simp only [List.map_cons, List.foldl_cons]
    rw [merge_bound g j cols ref L]
    have hstep : (mergeSeg cols acc (segOf ref b)).toList.flatMap (segVals g j) =
        acc.toList.flatMap (segVals g j) ++ [g (ref b.a + j * (b.step / 2))] := by
      unfold mergeSeg
      cases hb : acc.back? with
      | none => simp [segVals, segOf]
      | some l =>
        simp only
        by_cases hc : (l.reps == 1 && l.step == (segOf ref b).step && l.at_ + l.len == (segOf ref b).at_ &&
            (decide (l.at_ < cols) == decide ((segOf ref b).at_ < cols))) = true
        · rw [ite_eq_left hc]
          simp only [Bool.and_eq_true, beq_iff_eq] at hc
          obtain ⟨⟨⟨h1, h2⟩, h3⟩, -⟩ := hc
          simp only [segOf] at h2 h3
          have hl' : segVals g j { l with len := l.len + 1 } = segVals g j l ++ [g (l.at_ + j * l.step + l.len)] := by
            simp [segVals, h1, List.range_succ]
          have key : l.at_ + j * l.step + l.len = ref b.a + j * (b.step / 2) := by
            rw [← h3, h2]
            ring
          rw [toList_of_back? hb, Array.toList_push, List.flatMap_append, List.flatMap_append]
          simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc]
          rw [hl', key]
        · rw [ite_eq_right hc]
          simp [segVals, segOf]
    rw [hstep, List.append_assoc]
    rfl

/-- **A finished call reads its bindings**: each call `Work.finish` writes, copy `j`, binds its callee's inputs to its
bindings' references, each stepping `j` times its step, in order. -/
theorem finish_call (w : Work) (ins : ℕ) {p : Pending} (hp : p ∈ w.calls.toList) :
    ∃ c ∈ (w.finish ins).calls.toList, c.def_ = p.def_ ∧ c.count = p.count ∧ c.base = p.base ∧ c.stride = p.stride ∧
      ∀ g j, c.boundV g j =
        p.binds.toList.map (fun b => g ((if b.a % 2 == 0 then b.a / 2 else w.next + b.a / 2) + j * (b.step / 2))) := by
  refine ⟨⟨p.def_, p.count, p.base, p.stride,
    (p.binds.map (segOf (fun x => if x % 2 == 0 then x / 2 else w.next + x / 2))).foldl (mergeSeg w.next) #[]⟩,
    ?_, rfl, rfl, rfl, rfl, fun g j => ?_⟩
  · simp only [Work.finish, Array.toList_map, List.mem_map]
    exact ⟨p, hp, rfl⟩
  · unfold Call.boundV
    rw [← Array.foldl_toList, Array.toList_map]
    have := merge_bound g j w.next (fun x => if x % 2 == 0 then x / 2 else w.next + x / 2) p.binds.toList #[]
    simp only [List.flatMap_nil, List.nil_append] at this
    rw [← this]
    rfl

/-- **A copy a finished Definition places holds**: where a copy of `w`'s finished Definition satisfies its equations,
copy `j` of each of its calls satisfies its callee's, on its bindings' values. -/
theorem call_copy {defs : Array Def} {fuel e0 ins : ℕ} {w : Work} {v : ℕ → Bool} {inp : List Bool}
    (hD : defs.getD e0 default = w.finish ins) (hH : Holds defs true (fuel + 1) e0 v inp) {p : Pending}
    (hp : p ∈ w.calls.toList) {j : ℕ} (hj : j < p.count) :
    Holds defs true fuel p.def_ (fun y => v (p.base + j * p.stride + y))
      (p.binds.toList.map fun b =>
        (w.finish ins).refV v inp true ((if b.a % 2 == 0 then b.a / 2 else w.next + b.a / 2) + j * (b.step / 2))) := by
  obtain ⟨c, hc, h1, h2, h3, h4, h5⟩ := finish_call w ins hp
  have := hH.2.1
  rw [hD] at this
  have := this c hc j (by omega)
  rw [h1, h3, h4, h5] at this
  exact this

/-- **The call case, a node**: a call node (not a batch) of template Definition `e` keeps `w`'s node outputs the walk's,
its new output the copy's returns `rets`, wherever the rows its bindings commit hold and the copy's return columns hold
the leaves `l`; its bindings are the walk's argument leaves. -/
theorem call_node {V : ℕ → Bool} {w w1 : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    {callee : Defn} {x : Node} {bs : Array Bind} {e cols : ℕ} {rets : Array ℕ} {l : List Bool}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hform : (x.form == .batch) = false) (hb : w.binds st callee x = .ok (w1, bs)) (hV : RowsHold V w1)
    (hl : ∀ k < rets.size, V (2 * (w1.next + rets[k]!)) = l.getD k false) :
    w1.ins = w.ins ∧ w.rows.toList <+: w1.rows.toList ∧ w1.calls = w.calls ∧
      bs.toList.map (fun b => (V b.a, b.step)) =
        (x.args.toList.flatMap fun a => argVals (refVal xs st vals) a a.seq.count id).map (·, 0) ∧
      FormsOk V (w1.place e cols 1 bs rets) (vals.push l) := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := binds_val hF hP hseq hform hb
  refine ⟨h1, h4, h3, h5 hV, ?_⟩
  let pd : Pending := ⟨e, 1, w1.next, cols, bs⟩
  have := formsOk_push (w' := { w1 with calls := w1.calls.push pd, next := w1.next + 1 * cols })
    (o := .copies w1.next cols 1 rets) hF h2 (fun j hj => by
      have hj' : j < rets.size := by omega
      rw [Nat.div_eq_of_lt hj', Nat.mod_eq_of_lt hj', Nat.zero_mul, Nat.add_zero]
      exact hl j hj')
  exact this


/-! ## The batch case: a sequence's part, a stepping binding -/

/-- **A reference by the part holding it**: reference `k` of a sequence whose starts are its parts' running sums is its
part `x`'s element `k - starts[x]`, for any part `x` holding it. -/
theorem toArray_getD_part {q : Seq} (hq : seqOk q = true) {x k : ℕ} (hx : x < q.parts.size)
    (h1 : q.starts[x]! ≤ k) (h2 : k < q.starts[x]! + q.parts[x]!.count) :
    q.toArray.getD k default = q.parts[x]!.get (k - q.starts[x]!) := by
  have he := seqOk_eq hq
  obtain ⟨_, _, hst, _⟩ := InferenceOnlyVerifier.Properties.SecurityProofs.Walk.ofParts_spec q.parts
  rw [he] at hst
  have hS : ((((q.parts.toList.map fun p => (List.range p.count).map p.get)).take x).map List.length).sum =
      q.starts[x]! := by
    rw [hst x hx, ← List.map_take, List.map_map]
    congr 1
    apply List.map_congr_left
    intro p _
    simp
  have hrl : x < (q.parts.toList.map fun p => (List.range p.count).map p.get).length := by simpa using hx
  have hpr : q.parts[x]! = q.parts.toList[x]'(by simpa using hx) := by
    rw [getElem!_pos q.parts x hx]; simp
  rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, toArray_toList,
    getElem?_flatten_of _ x k hrl (by rw [hS]; exact h1) (by rw [hS]; simpa [hpr] using h2), hS]
  simp only [List.getElem_map]
  rw [List.getElem?_map, List.getElem?_range (by rw [← hpr]; omega)]
  simp [hpr]

/-- **`partAt`'s bisection finds the part holding a reference** of a sequence whose starts are its parts' running sums. -/
theorem partAt_spec {q : Seq} (hq : seqOk q = true) {k : ℕ} (hk : k < q.count) :
    partAt q k < q.parts.size ∧ q.starts[partAt q k]! ≤ k ∧ k < q.starts[partAt q k]! + q.parts[partAt q k]!.count := by
  have he := seqOk_eq hq
  obtain ⟨_, hsz, hst, hcnt⟩ := InferenceOnlyVerifier.Properties.SecurityProofs.Walk.ofParts_spec q.parts
  rw [he] at hsz hst hcnt
  have hpos : 0 < q.parts.size := by
    rcases Nat.eq_zero_or_pos q.parts.size with h | h
    · have : q.parts.toList = [] := by simpa [← Array.length_toList] using h
      rw [hcnt, this] at hk
      simp at hk
    · exact h
  have h0 : q.starts[0]! = 0 := by rw [hst 0 hpos]; simp
  have hlen : ∀ i < q.parts.size, q.starts[i]! + q.parts[i]!.count = ((q.parts.toList.take (i + 1)).map Part.count).sum :=
    fun i hi => by rw [hst i hi, InferenceOnlyVerifier.Properties.SecurityProofs.Walk.sum_take_succ_map q.parts hi]
  let f : ℕ → ℕ × ℕ → Id (ForInStep (ℕ × ℕ)) := fun _ (st : ℕ × ℕ) =>
    have lo := st.1; have hi := st.2
    if hi - lo ≤ 1 then pure (ForInStep.done (lo, hi))
    else
      have mid := (lo + hi) / 2
      if q.starts[mid]! ≤ k then pure (ForInStep.yield (mid, hi)) else pure (ForInStep.yield (lo, mid))
  obtain ⟨b1, b2, b3, b4, b5⟩ := InferenceOnlyVerifier.Properties.SecurityProofs.Walk.bisect_spec q.starts k
    q.parts.size f (fun _ _ _ => rfl) (List.range' 0 q.parts.size 1) 0 q.parts.size hpos (le_refl _)
    (by rw [h0]; omega) (fun h => absurd h (lt_irrefl _)) (by simp)
  set r := (forIn (List.range' 0 q.parts.size 1) (0, q.parts.size) f : Id (ℕ × ℕ))
  have hpa : partAt q k = r.1 := by
    unfold partAt
    have hfor : ∀ (g : ℕ → ℕ × ℕ → Id (ForInStep (ℕ × ℕ))) (init : ℕ × ℕ),
        (forIn [:q.parts.size] init g : Id (ℕ × ℕ)) = forIn (List.range' 0 q.parts.size) init g := by
      intro g init
      rw [Std.Legacy.Range.forIn_eq_forIn_range']
      simp [Std.Legacy.Range.size]
    simp only [hfor]
    rfl
  rw [hpa]
  have hr2 : r.2 = r.1 + 1 := by omega
  refine ⟨by omega, b3, ?_⟩
  rw [hlen r.1 (by omega)]
  rcases Nat.lt_or_ge r.2 q.parts.size with hlt | hge
  · have := b4 hlt
    rw [hst r.2 hlt, hr2] at this
    exact this
  · have hre : r.1 + 1 = q.parts.toList.length := by simp; omega
    rw [hre, List.take_length, ← hcnt]
    exact hk

theorem refVal_param {xs : List Bool} {st : Array ℕ} {vals : Array (List Bool)} {r : Ref} (hp : r.p = true) {s0 : ℕ}
    (hs : st[r.i]? = some s0) (j : ℕ) : refVal xs st vals ⟨r.p, r.i, j⟩ = xs.getD (s0 + j) false := by
  simp only [refVal, hp, ite_true, Array.getD_eq_getD_getElem?, hs, Option.getD_some]
  rfl

theorem refVal_node {xs : List Bool} {st : Array ℕ} {vals : Array (List Bool)} {r : Ref} (hp : ¬r.p = true) (j : ℕ) :
    refVal xs st vals ⟨r.p, r.i, j⟩ = (vals.getD r.i []).getD j false := by
  simp only [refVal, hp, Bool.false_eq_true, ite_false]
  rfl

/-- **A stepping binding is the walk's reference stepped**: copy `j < cnt` of the binding `Work.stepOf` gives reference
`r` stepping `s` leaves a copy reads atom `a + j · step`, whose value is the walk's value of `r` moved `j · s` leaves. -/
theorem stepOf_val {V : ℕ → Bool} {w : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {r : Ref} {s cnt : ℕ} {b : Bind}
    (h : w.stepOf st r s cnt = .ok b) :
    b.step % 2 = 0 ∧ ∀ j < cnt, V (b.a + j * b.step) = refVal xs st vals ⟨r.p, r.i, r.j + j * s⟩ := by
  unfold Work.stepOf at h
  by_cases hp : r.p = true
  · rw [ite_eq_left hp] at h
    cases hs : st[r.i]? with
    | none => rw [hs] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some s0 =>
      rw [hs] at h
      simp only at h
      by_cases hl : s0 + r.j + (cnt - 1) * s < w.ins
      · rw [ite_eq_left hl] at h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine ⟨Nat.mul_mod_right 2 _, fun j hj => ?_⟩
        have hjs : j * s ≤ (cnt - 1) * s := Nat.mul_le_mul_right s (by omega)
        have e : 2 * (s0 + r.j) + 1 + j * (2 * s) = 2 * (s0 + (r.j + j * s)) + 1 := by ring
        rw [e, hP _ (by omega), refVal_param hp hs]
      · rw [ite_eq_right hl] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
  · rw [ite_eq_right hp] at h
    cases ho : w.outs[r.i]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      obtain ⟨hi, heq⟩ := Array.getElem?_eq_some_iff.mp ho
      have hO := hF.2 r.i hi
      rw [heq] at hO
      cases o with
      | gate g => simp [throw, throwThe, MonadExceptOf.throw] at h
      | copies base stride m rets =>
        simp only at h
        by_cases hc : (rets.size == 0 || s % rets.size != 0) = true
        · rw [ite_eq_left hc] at h
          simp [throw, throwThe, MonadExceptOf.throw] at h
        · rw [ite_eq_right hc] at h
          simp only [Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
          by_cases hl : m * rets.size ≤ r.j + (cnt - 1) * s
          · rw [ite_eq_left hl] at h
            simp [throw, throwThe, MonadExceptOf.throw] at h
          · rw [ite_eq_right hl] at h
            simp only [pure, Except.pure, Except.ok.injEq] at h
            subst h
            refine ⟨Nat.mul_mod_right 2 _, fun j hj => ?_⟩
            have hL : 0 < rets.size := Nat.pos_of_ne_zero hc.1
            obtain ⟨q, hq⟩ : ∃ q, s = rets.size * q := ⟨s / rets.size, (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hc.2)).symm⟩
            have hjs : j * s ≤ (cnt - 1) * s := Nat.mul_le_mul_right s (by omega)
            have hjl : r.j + j * s < m * rets.size := by omega
            have e0 : j * s = rets.size * (j * q) := by rw [hq]; ring
            have e1 : (r.j + j * s) / rets.size = r.j / rets.size + j * q := by
              rw [e0, Nat.add_mul_div_left _ _ hL]
            have e2 : (r.j + j * s) % rets.size = r.j % rets.size := by rw [e0, Nat.add_mul_mod_self_left]
            have e3 : s / rets.size = q := by rw [hq, Nat.mul_div_cancel_left _ hL]
            have := hO (r.j + j * s) hjl
            rw [e1, e2] at this
            rw [refVal_node hp, ← this, e3]
            congr 1
            ring
      | scan base stride rets c m =>
        simp only at h
        by_cases h1 : c + m * (rets.size - c) ≤ r.j + (cnt - 1) * s
        · rw [ite_eq_left h1] at h
          simp [throw, throwThe, MonadExceptOf.throw] at h
        · rw [ite_eq_right h1] at h
          by_cases h2 : (decide (r.j < c) || rets.size - c == 0 || s % (rets.size - c) != 0) = true
          · rw [ite_eq_left h2] at h
            simp [throw, throwThe, MonadExceptOf.throw] at h
          · rw [ite_eq_right h2] at h
            simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, bne_iff_ne, ne_eq, not_or,
              Decidable.not_not, not_lt] at h2
            obtain ⟨⟨hcj, hy⟩, hsy⟩ := h2
            simp only [pure, Except.pure, Except.ok.injEq] at h
            subst h
            refine ⟨Nat.mul_mod_right 2 _, fun j hj => ?_⟩
            have hL : 0 < rets.size - c := Nat.pos_of_ne_zero hy
            obtain ⟨q, hq⟩ : ∃ q, s = (rets.size - c) * q :=
              ⟨s / (rets.size - c), (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hsy)).symm⟩
            have hjs : j * s ≤ (cnt - 1) * s := Nat.mul_le_mul_right s (by omega)
            have hjl : r.j + j * s < c + m * (rets.size - c) := by omega
            have e0 : r.j + j * s - c = r.j - c + (rets.size - c) * (j * q) := by
              have : j * s = (rets.size - c) * (j * q) := by rw [hq]; ring
              omega
            have e1 : (r.j + j * s - c) / (rets.size - c) = (r.j - c) / (rets.size - c) + j * q := by
              rw [e0, Nat.add_mul_div_left _ _ hL]
            have e2 : (r.j + j * s - c) % (rets.size - c) = (r.j - c) % (rets.size - c) := by
              rw [e0, Nat.add_mul_mod_self_left]
            have e3 : s / (rets.size - c) = q := by rw [hq, Nat.mul_div_cancel_left _ hL]
            have := (hO (r.j + j * s) hjl).2 (by omega)
            rw [e1, e2] at this
            rw [refVal_node hp, ← this, e3]
            congr 1
            ring

/-- A batch's leaves, its members' values one after another, each of `L` leaves. -/
theorem getD_flatten_eq {ms : List (List Bool)} {L : ℕ} (hL : ∀ l ∈ ms, l.length = L) {j : ℕ}
    (hj : j < ms.length * L) : ms.flatten.getD j false = (ms[j / L]?.getD []).getD (j % L) false := by
  have hL0 : 0 < L := by
    rcases Nat.eq_zero_or_pos L with h | h
    · subst h; simp at hj
    · exact h
  have hx : j / L < ms.length := by
    rw [Nat.div_lt_iff_lt_mul hL0]; exact hj
  have hS : ((ms.take (j / L)).map List.length).sum = j / L * L := by
    have : (ms.take (j / L)).map List.length = List.replicate (j / L) L := by
      apply List.eq_replicate_iff.mpr
      refine ⟨by simp; omega, fun b hb => ?_⟩
      obtain ⟨l, hl, rfl⟩ := List.mem_map.mp hb
      exact hL l (List.mem_of_mem_take hl)
    rw [this, List.sum_replicate, smul_eq_mul]
  have hlx := hL _ (List.getElem_mem hx)
  have hm : j % L = j - j / L * L := by
    have := Nat.div_add_mod j L
    rw [Nat.mul_comm] at this
    omega
  rw [List.getD_eq_getElem?_getD, getElem?_flatten_of ms (j / L) j hx (by rw [hS]; exact Nat.div_mul_le_self j L)
    (by rw [hS, hlx]; have := Nat.lt_div_mul_add (a := j) hL0; omega), hS, List.getElem?_eq_getElem hx,
    Option.getD_some, List.getD_eq_getElem?_getD, hm]

/-! ## Bounds: the own columns a form names are written -/

theorem dropPairs_sublist : ∀ l : List ℕ, (dropPairs l).Sublist l
  | [] => List.Sublist.refl _
  | [_] => List.Sublist.refl _
  | a :: b :: rest => by
    simp only [dropPairs]
    split
    · exact ((dropPairs_sublist rest).cons b).cons a
    · exact (dropPairs_sublist (b :: rest)).cons₂ a

theorem mem_canon {xs : Array ℕ} {x : ℕ} (h : x ∈ (Tmpl.canon xs).toList) : x ∈ xs.toList := by
  have h' : x ∈ dropPairs xs.toList.mergeSort := by simpa [Tmpl.canon] using h
  exact (List.mergeSort_perm _ _).mem_iff.mp ((dropPairs_sublist _).subset h')

/-- Every own column form `f` names is below `n`. -/
def FormB (n : ℕ) (f : Form) : Prop := ∀ x ∈ f.atoms.toList, x % 2 = 0 → x / 2 < n

theorem formB_mono {n n' : ℕ} {f : Form} (h : FormB n f) (hn : n ≤ n') : FormB n' f :=
  fun x hx he => Nat.lt_of_lt_of_le (h x hx he) hn

theorem formB_add {n : ℕ} {f g : Form} (hf : FormB n f) (hg : FormB n g) : FormB n (f.add g) := by
  intro x hx he
  have := mem_canon hx
  rw [Array.toList_append, List.mem_append] at this
  rcases this with h | h
  · exact hf x h he
  · exact hg x h he

theorem formB_own {n c : ℕ} (h : c < n) : FormB n (Form.own c) := by
  intro x hx _
  simp only [Form.own, List.mem_singleton] at hx
  subst hx
  omega

theorem formB_const (n : ℕ) (b : Bool) : FormB n (Form.const b) := by
  intro x hx; simp [Form.const] at hx

theorem formB_empty (n : ℕ) : FormB n ({} : Form) := by
  intro x hx; simp at hx

theorem formB_input (n i : ℕ) : FormB n (Form.input i) := by
  intro x hx he
  simp only [Form.input, List.mem_singleton] at hx
  omega

/-- Node output `o`'s own columns are below `n`: a gate's form's, and each copy's return columns inside its copy. -/
def OutB (n : ℕ) : Out → Prop
  | .gate f => FormB n f
  | .copies base stride m rets => 0 < m → (∀ k < rets.size, rets[k]! < stride) ∧ base + m * stride ≤ n
  | .scan base stride rets c m => 0 < m ∧ c ≤ rets.size ∧ (∀ k < rets.size, rets[k]! < stride) ∧ base + m * stride ≤ n

theorem outB_mono {n n' : ℕ} {o : Out} (h : OutB n o) (hn : n ≤ n') : OutB n' o := by
  cases o with
  | gate f => exact formB_mono h hn
  | copies base stride m rets => exact fun hm => ⟨(h hm).1, by have := (h hm).2; omega⟩
  | scan base stride rets c m => exact ⟨h.1, h.2.1, h.2.2.1, by have := h.2.2.2; omega⟩

/-- `w`'s node outputs name only its written own columns. -/
def BoundsOk (w : Work) : Prop := ∀ i (h : i < w.outs.size), OutB w.next w.outs[i]

theorem boundsOk_mono {w w' : Work} (h : BoundsOk w) (ho : w'.outs = w.outs) (hn : w.next ≤ w'.next) : BoundsOk w' :=
  fun i hi => by
    have hi' : i < w.outs.size := by rw [← ho]; exact hi
    have := outB_mono (h i hi') hn
    simp only [ho]
    exact this

theorem boundsOk_push {w w' : Work} {o : Out} (h : BoundsOk w) (ho : w'.outs = w.outs) (hn : w.next ≤ w'.next)
    (hb : OutB w'.next o) : BoundsOk { w' with outs := w'.outs.push o } := by
  intro i hi
  simp only [ho, Array.size_push] at hi
  simp only [ho]
  rcases Nat.lt_or_ge i w.outs.size with hlt | hge
  · rw [Array.getElem_push_lt hlt]
    exact outB_mono (h i hlt) hn
  · have : i = w.outs.size := by omega
    subst this
    rw [Array.getElem_push_eq]
    exact hb

theorem copy_col_lt {n base stride m L j : ℕ} (hj : j < m * L) {r : ℕ} (hr : r < stride) (hb : base + m * stride ≤ n) :
    base + j / L * stride + r < n := by
  have hL : 0 < L := Nat.pos_of_ne_zero (by rintro rfl; simp at hj)
  have : j / L < m := by rw [Nat.div_lt_iff_lt_mul hL]; exact hj
  have : (j / L + 1) * stride ≤ m * stride := Nat.mul_le_mul_right _ this
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- **A form names written own columns**: where `w`'s node outputs do, every form `Work.formOf` gives does. -/
theorem formOf_bound {w : Work} {st : Array ℕ} (hB : BoundsOk w) {r : Ref} {f : Form} (h : w.formOf st r = .ok f) :
    FormB w.next f := by
  unfold Work.formOf at h
  by_cases hp : r.p = true
  · rw [ite_eq_left hp] at h
    cases hs : st[r.i]? with
    | none => rw [hs] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some s =>
      rw [hs] at h
      simp only at h
      by_cases hl : s + r.j < w.ins
      · rw [ite_eq_left hl] at h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact formB_input _ _
      · rw [ite_eq_right hl] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
  · rw [ite_eq_right hp] at h
    cases ho : w.outs[r.i]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      obtain ⟨hi, heq⟩ := Array.getElem?_eq_some_iff.mp ho
      have hO := hB r.i hi
      rw [heq] at hO
      cases o with
      | gate g =>
        simp only at h
        by_cases hj : r.j = 0
        · rw [ite_eq_left (by simp [hj])] at h
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          exact hO
        · rw [ite_eq_right (by simp [hj])] at h
          simp [throw, throwThe, MonadExceptOf.throw] at h
      | copies base stride n rets =>
        simp only at h
        by_cases he : rets.isEmpty = true
        · rw [ite_eq_left he] at h
          simp [throw, throwThe, MonadExceptOf.throw] at h
        · rw [ite_eq_right he] at h
          by_cases hl : r.j < n * rets.size
          · rw [ite_eq_left hl] at h
            simp only [pure, Except.pure, Except.ok.injEq] at h
            subst h
            have hn : 0 < n := Nat.pos_of_ne_zero (by rintro rfl; simp at hl)
            have hL : 0 < rets.size := Nat.pos_of_ne_zero (fun h0 => by rw [h0] at hl; simp at hl)
            exact formB_own (copy_col_lt hl ((hO hn).1 _ (Nat.mod_lt _ hL)) (hO hn).2)
          · rw [ite_eq_right hl] at h
            simp [throw, throwThe, MonadExceptOf.throw] at h
      | scan base stride rets c n =>
        simp only at h
        by_cases h1 : c + n * (rets.size - c) ≤ r.j
        · rw [ite_eq_left h1] at h
          simp [throw, throwThe, MonadExceptOf.throw] at h
        · rw [ite_eq_right h1] at h
          obtain ⟨hn, hc, hr, hb⟩ := hO
          by_cases h2 : r.j < c
          · rw [ite_eq_left h2] at h
            simp only [pure, Except.pure, Except.ok.injEq] at h
            subst h
            apply formB_own
            have := hr r.j (by omega)
            have : (n - 1 + 1) * stride ≤ n * stride := Nat.mul_le_mul_right _ (by omega)
            rw [Nat.add_mul, Nat.one_mul] at this
            omega
          · rw [ite_eq_right h2] at h
            by_cases h3 : (rets.size - c == 0) = true
            · rw [ite_eq_left h3] at h
              simp [throw, throwThe, MonadExceptOf.throw] at h
            · rw [ite_eq_right h3] at h
              simp only [pure, Except.pure, Except.ok.injEq] at h
              subst h
              simp only [beq_iff_eq] at h3
              have hy : 0 < rets.size - c := Nat.pos_of_ne_zero h3
              have := hr (c + (r.j - c) % (rets.size - c)) (by have := Nat.mod_lt (r.j - c) hy; omega)
              exact formB_own (copy_col_lt (m := n) (L := rets.size - c) (j := r.j - c) (by omega) this hb)

theorem and_bound {w : Work} {a b : Form} (ha : FormB w.next a) (hb : FormB w.next b) :
    w.next ≤ (w.and a b).1.next ∧ FormB (w.and a b).1.next (w.and a b).2 := by
  unfold Work.and
  split
  · refine ⟨le_refl _, ?_⟩
    cases a.one
    · exact formB_empty _
    · exact hb
  split
  · refine ⟨le_refl _, ?_⟩
    cases b.one
    · exact formB_empty _
    · exact ha
  simp only [Work.row]
  exact ⟨by omega, formB_own (by omega)⟩

theorem andFold_bound :
    ∀ (l : List Form) (acc : Work × Form), FormB acc.1.next acc.2 → (∀ o ∈ l, FormB acc.1.next o) →
      acc.1.next ≤ (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).1.next ∧
        FormB (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).1.next
          (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).2
  | [], acc, h, _ => ⟨le_refl _, h⟩
  | o :: l, acc, h, hl => by
    simp only [List.foldl_cons]
    obtain ⟨h1, h2⟩ := and_bound h (hl o (by simp))
    obtain ⟨h3, h4⟩ := andFold_bound l _ h2 (fun o' ho' => formB_mono (hl o' (by simp [ho'])) h1)
    exact ⟨h1.trans h3, h4⟩

theorem xorFold_bound {n : ℕ} : ∀ (l : List Form) (acc : Form), FormB n acc → (∀ o ∈ l, FormB n o) →
    FormB n (l.foldl Form.add acc)
  | [], _, h, _ => h
  | o :: l, acc, h, hl => by
    simp only [List.foldl_cons]
    exact xorFold_bound l _ (formB_add h (hl o (by simp))) (fun o' ho' => hl o' (by simp [ho']))

/-- **`Work.op`'s form names written own columns**, where its operands do. -/
theorem op_bound {w w' : Work} {g : Gate} {ops : Array Form} {f : Form} (h : w.op g ops = .ok (w', f))
    (hops : ∀ o ∈ ops.toList, FormB w.next o) : w.next ≤ w'.next ∧ FormB w'.next f := by
  cases g with
  | const_ b =>
    simp only [Work.op, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨le_refl _, formB_const _ _⟩
  | not_ =>
    simp only [Work.op] at h
    cases ho : ops[0]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      simp only at h
      split at h
      · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨le_refl _, formB_add (hops o ?_) (formB_const _ _)⟩
        rw [toList_of_getElem?_zero ho]; simp
      · simp [throw, throwThe, MonadExceptOf.throw] at h
  | xor_ =>
    simp only [Work.op] at h
    split at h
    · simp [throw, throwThe, MonadExceptOf.throw] at h
    · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      refine ⟨le_refl _, ?_⟩
      rw [← Array.foldl_toList]
      exact xorFold_bound _ _ (formB_empty _) hops
  | and_ =>
    simp only [Work.op] at h
    cases ho : ops[0]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      simp only at h
      split at h
      · simp [throw, throwThe, MonadExceptOf.throw] at h
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        rw [← Array.foldl_toList] at h
        have hl := toList_of_getElem?_zero ho
        have := andFold_bound (ops.extract 1 ops.size).toList (w, o) (hops o (by rw [hl]; simp))
          (fun o' ho' => hops o' (by rw [hl]; exact List.mem_cons_of_mem _ ho'))
        rw [h] at this
        exact this

theorem single?_mem {f : Form} {x : ℕ} (h : f.single? = some x) : x ∈ f.atoms.toList := by
  unfold Form.single? at h
  by_cases hc : (f.atoms.size == 1 && !f.one) = true
  · rw [ite_eq_left hc] at h
    rw [← Array.getElem?_toList] at h
    exact List.mem_of_getElem? h
  · rw [ite_eq_right hc] at h
    exact absurd h (by simp)

theorem commit_bound {w : Work} {f : Form} (hf : FormB w.next f) :
    w.next ≤ (w.commit f).1.next ∧ ((w.commit f).2 % 2 = 0 → (w.commit f).2 / 2 < (w.commit f).1.next) := by
  unfold Work.commit
  split
  · rename_i x hx
    exact ⟨le_refl _, hf x (single?_mem hx)⟩
  · simp only [Work.row]
    exact ⟨by omega, fun _ => by omega⟩

/-- `Work.commitOwn`: its column's value is `f`'s where its rows hold, and the column is written. -/
theorem commitOwn_spec {V : ℕ → Bool} (w : Work) (f : Form) (hf : FormB w.next f) :
    (w.commitOwn f).1.ins = w.ins ∧ (w.commitOwn f).1.outs = w.outs ∧ (w.commitOwn f).1.calls = w.calls ∧
      w.rows.toList <+: (w.commitOwn f).1.rows.toList ∧ w.next ≤ (w.commitOwn f).1.next ∧
      (w.commitOwn f).2 < (w.commitOwn f).1.next ∧
      (RowsHold V (w.commitOwn f).1 → V (2 * (w.commitOwn f).2) = fval V f) := by
  unfold Work.commitOwn
  split
  · rename_i x hx
    by_cases he : (x % 2 == 0) = true
    · rw [ite_eq_left he]
      simp only [beq_iff_eq] at he
      refine ⟨rfl, rfl, rfl, List.prefix_refl _, le_refl _, hf x (single?_mem hx) he, fun _ => ?_⟩
      rw [single?_val hx]
      congr 1
      omega
    · rw [ite_eq_right he]
      refine ⟨rfl, rfl, rfl, ?_, by simp [Work.row], by simp [Work.row], fun hV => ?_⟩
      · simp only [Work.row, Array.toList_push]; exact List.prefix_append _ _
      · simp only [Work.row] at hV ⊢
        have := hV (w.next, f, Form.const true) (by simp)
        rw [val_const, Bool.and_true] at this
        exact this
  · refine ⟨rfl, rfl, rfl, ?_, by simp [Work.row], by simp [Work.row], fun hV => ?_⟩
    · simp only [Work.row, Array.toList_push]; exact List.prefix_append _ _
    · simp only [Work.row] at hV ⊢
      have := hV (w.next, f, Form.const true) (by simp)
      rw [val_const, Bool.and_true] at this
      exact this

end Flock.Chain.LowerDef
