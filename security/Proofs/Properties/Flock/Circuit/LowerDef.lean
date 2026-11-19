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


/-! ## The batch case: where a piece of members stops -/

/-- What `pieceEnd`'s end `m1` says of input `p` from member `m0`: a mapped input's reference for member `m0` lies in an
affine part, which holds its references up to member `m1`. -/
def EndOk (m0 m1 : ℕ) (p : Arg × ℕ) : Prop :=
  ∀ d L, p.1.dims = some (d, L) → L ≠ 0 →
    memberIx p.1 m0 p.2 < p.1.seq.count ∧
    (∃ p0 i0 b0 c0, p.1.seq.parts[partAt p.1.seq (memberIx p.1 m0 p.2)]! = .affine p0 i0 b0 c0) ∧
    m1 ≤ m0 + (p.1.seq.starts[partAt p.1.seq (memberIx p.1 m0 p.2)]! +
      p.1.seq.parts[partAt p.1.seq (memberIx p.1 m0 p.2)]!.count - memberIx p.1 m0 p.2 + L - 1) / L

theorem endOk_mono {m0 m1 m1' : ℕ} {p : Arg × ℕ} (h : m1' ≤ m1) (hp : EndOk m0 m1 p) : EndOk m0 m1' p :=
  fun d L hd hL => let ⟨a, b, c⟩ := hp d L hd hL; ⟨a, b, by omega⟩

theorem forIn_end {m0 : ℕ}
    {f : Arg × ℕ → Option (Option ℕ) × ℕ → Except String (ForInStep (Option (Option ℕ) × ℕ))}
    (hf : ∀ p m s, f p (none, m) = .ok s →
      s = .done (some none, m) ∨ ∃ m', s = .yield (none, m') ∧ m' ≤ m ∧ EndOk m0 m' p) :
    ∀ (l : List (Arg × ℕ)) (m : ℕ) (s : Option (Option ℕ) × ℕ), forIn l (none, m) f = .ok s →
      (s.1 = none ∨ s.1 = some none) ∧ (s.1 = none → s.2 ≤ m ∧ ∀ p ∈ l, EndOk m0 s.2 p) := by
  intro l
  induction l with
  | nil =>
    intro m s h
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨Or.inl rfl, fun _ => ⟨le_refl _, fun p hp => absurd hp List.not_mem_nil⟩⟩
  | cons p l ih =>
    intro m s h
    rw [List.forIn_cons] at h
    cases hfp : f p (none, m) with
    | error e => rw [hfp] at h; simp [bind, Except.bind] at h
    | ok st =>
      rw [hfp] at h
      simp only [bind, Except.bind] at h
      rcases hf p m st hfp with hd | ⟨m', hy, hm', hE⟩
      · subst hd
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨Or.inr rfl, fun hs => by simp at hs⟩
      · subst hy
        obtain ⟨h0, h12⟩ := ih m' s h
        refine ⟨h0, fun hs => ?_⟩
        obtain ⟨h1, h2⟩ := h12 hs
        refine ⟨by omega, fun q hq => ?_⟩
        rcases List.mem_cons.mp hq with rfl | hq
        · exact endOk_mono h1 hE
        · exact h2 q hq

/-- **`pieceEnd`'s end**: at most `n`, and each mapped input's references for members `[m0, m1)` in one affine part. -/
theorem pieceEnd_spec {ins : Array (Arg × ℕ)} {n m0 m1 : ℕ} (h : pieceEnd ins n m0 = .ok (some m1)) :
    m1 ≤ n ∧ ∀ p ∈ ins.toList, EndOk m0 m1 p := by
  unfold pieceEnd at h
  dsimp only at h
  simp only [bind, Except.bind] at h
  split at h
  · simp at h
  · rename_i s hfor
    rw [← Array.forIn_toList] at hfor
    obtain ⟨s1, s2⟩ := s
    have key := forIn_end (m0 := m0) ?_ ins.toList n _ hfor
    · cases s1 with
      | some r =>
        rcases key.1 with h0 | h0
        · simp at h0
        · simp only [Option.some.injEq] at h0
          subst h0
          simp [pure, Except.pure] at h
      | none =>
        simp only [pure, Except.pure, Except.ok.injEq, Option.some.injEq] at h
        subst h
        exact key.2 rfl
    · intro p m s hs
      obtain ⟨a, t⟩ := p
      simp only at hs
      rcases hd : a.dims with _ | ⟨d, L⟩
      · simp only [hd, pure, Except.pure, Except.ok.injEq] at hs
        subst hs
        exact Or.inr ⟨m, rfl, le_refl _, fun d' L' hd' _ => by rw [hd] at hd'; simp at hd'⟩
      · simp only [hd] at hs
        by_cases hL : L = 0
        · simp only [hL, beq_self_eq_true, ite_true, pure, Except.pure, Except.ok.injEq] at hs
          subst hs
          exact Or.inr ⟨m, rfl, le_refl _, fun d' L' hd' hL' => by
            rw [hd] at hd'; simp only [Option.some.injEq, Prod.mk.injEq] at hd'; omega⟩
        · have hL' : (L == 0) = false := by simpa using hL
          simp only [hL', Bool.false_eq_true, ite_false] at hs
          by_cases hk : a.seq.count ≤ memberIx a m0 t
          · simp [hk, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
          · simp only [hk, ite_false] at hs
            cases hpart : a.seq.parts[partAt a.seq (memberIx a m0 t)]! with
            | affine p0 i0 b0 c0 =>
              simp only [hpart, pure, Except.pure, Except.ok.injEq] at hs
              subst hs
              refine Or.inr ⟨_, rfl, Nat.min_le_left _ _, fun d' L' hd' _ => ?_⟩
              rw [hd] at hd'
              simp only [Option.some.injEq, Prod.mk.injEq] at hd'
              obtain ⟨-, rfl⟩ := hd'
              refine ⟨by simp only; omega, ⟨p0, i0, b0, c0, hpart⟩, ?_⟩
              simp only [hpart]
              exact Nat.min_le_right _ _
            | explicit items =>
              simp only [hpart, pure, Except.pure, Except.ok.injEq] at hs
              exact Or.inl hs.symm
            | strided => simp [hpart, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
            | grid => simp [hpart, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs


/-! ## The batch case: a piece's bindings -/

/-- A fold that pushes one element a step, each satisfying `P` of its step's element. -/
theorem foldlM_pushes {α β : Type} {F : Array β → α → Except String (Array β)} {P : α → β → Prop}
    {S : List α} (hF : ∀ bs j bs', j ∈ S → F bs j = .ok bs' → ∃ b, bs' = bs.push b ∧ P j b) :
    ∀ (l : List α) (acc bs : Array β), (∀ j ∈ l, j ∈ S) → l.foldlM F acc = .ok bs →
      ∃ ls, bs.toList = acc.toList ++ ls ∧ List.Forall₂ P l ls := by
  intro l
  induction l with
  | nil =>
    intro acc bs _ h
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨[], by simp, List.Forall₂.nil⟩
  | cons j l ih =>
    intro acc bs hS h
    rw [List.foldlM_cons] at h
    cases hj : F acc j with
    | error e => rw [hj] at h; simp [bind, Except.bind] at h
    | ok bs1 =>
      rw [hj] at h
      simp only [bind, Except.bind] at h
      obtain ⟨b, rfl, hb⟩ := hF acc j bs1 (hS j (by simp)) hj
      obtain ⟨ls, h1, h2⟩ := ih _ _ (fun i hi => hS i (by simp [hi])) h
      exact ⟨b :: ls, by rw [h1]; simp, List.Forall₂.cons hb h2⟩

/-- A fold over a work that pushes one element a step, each satisfying `P` of its step's element at every work `R` of
the last. -/
theorem foldlM_pushesW {α β σ : Type} {F : σ × Array β → α → Except String (σ × Array β)} {R : σ → σ → Prop}
    (hR : ∀ a, R a a) (hT : ∀ a b c, R a b → R b c → R a c) {P : α → β → σ → Prop}
    (hP : ∀ j b w w', P j b w → R w w' → P j b w')
    {S : List α} {w₀ : σ}
    (hF : ∀ w bs j r, j ∈ S → R w₀ w → F (w, bs) j = .ok r → ∃ b, r.2 = bs.push b ∧ R w r.1 ∧ P j b r.1) :
    ∀ (l : List α) (w : σ) (acc : Array β) (r : σ × Array β), (∀ j ∈ l, j ∈ S) → R w₀ w →
      l.foldlM F (w, acc) = .ok r →
      R w r.1 ∧ ∃ ls, r.2.toList = acc.toList ++ ls ∧ List.Forall₂ (fun j b => P j b r.1) l ls := by
  intro l
  induction l with
  | nil =>
    intro w acc r _ _ h
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨hR _, [], by simp, List.Forall₂.nil⟩
  | cons j l ih =>
    intro w acc r hS hw0 h
    rw [List.foldlM_cons] at h
    cases hj : F (w, acc) j with
    | error e => rw [hj] at h; simp [bind, Except.bind] at h
    | ok r1 =>
      rw [hj] at h
      simp only [bind, Except.bind] at h
      obtain ⟨b, h1, hw, hb⟩ := hF w acc j r1 (hS j (by simp)) hw0 hj
      obtain ⟨r11, r12⟩ := r1
      simp only at h1 hw hb
      subst h1
      obtain ⟨hw', ls, h2, h3⟩ := ih _ _ _ (fun i hi => hS i (by simp [hi])) (hT _ _ _ hw0 hw) h
      exact ⟨hT _ _ _ hw hw', b :: ls, by rw [h2]; simp, List.Forall₂.cons (hP _ _ _ _ hb hw') h3⟩

theorem forall₂_map {α β γ : Type} {P : α → β → Prop} {g : β → γ} {f : α → γ} (h : ∀ a b, P a b → g b = f a) :
    ∀ {l : List α} {ls : List β}, List.Forall₂ P l ls → ls.map g = l.map f
  | _, _, .nil => rfl
  | _, _, .cons hab hl => by simp only [List.map_cons, h _ _ hab, forall₂_map h hl]

theorem toList_eq_range {α : Type} [Inhabited α] (xs : Array α) :
    xs.toList = (List.range xs.size).map fun j => xs[j]! := by
  apply List.ext_getElem
  · simp
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
    rw [getElem!_pos xs i (by simpa using h1)]

/-- Member `m`'s value of batch input `p`: its argument's reference for `m` (`memberIx`) in the walk. -/
def memberVal (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (m : ℕ) (p : Arg × ℕ) : Bool :=
  refVal xs st vals (p.1.seq.toArray.getD (memberIx p.1 m p.2) default)

theorem memberIx_add {a : Arg} {d L : ℕ} (hd : a.dims = some (d, L)) (m j t : ℕ) :
    memberIx a (m + j) t = memberIx a m t + j * L := by
  simp only [memberIx, hd]
  ring

theorem memberIx_none {a : Arg} (hd : a.dims = none) (m t : ℕ) : memberIx a m t = t := by
  simp only [memberIx, hd]

/-- **A mapped input's stepping binding is each member's reference**: within `pieceEnd`'s end, member `m0 + j`'s. -/
theorem step_input {V : ℕ → Bool} {w : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {a : Arg} {t d L m0 m1 : ℕ} (hq : seqOk a.seq = true)
    (hd : a.dims = some (d, L)) (hE : EndOk m0 m1 (a, t)) {r : Ref} {b : Bind}
    (hr : getRef a.seq (memberIx a m0 t) = .ok r) (hb : w.stepOf st r L (m1 - m0) = .ok b) :
    b.step % 2 = 0 ∧ ∀ j < m1 - m0, V (b.a + j * b.step) = memberVal xs st vals (m0 + j) (a, t) := by
  obtain ⟨hk, rfl⟩ := getRef_ok hr
  obtain ⟨hs, hv⟩ := stepOf_val hF hP hb
  refine ⟨hs, fun j hj => ?_⟩
  rw [hv j hj]
  unfold memberVal
  simp only
  rw [memberIx_add hd]
  by_cases hL : L = 0
  · subst hL
    simp only [Nat.mul_zero, Nat.add_zero]
    rw [refAt_eq hq hk]
  · obtain ⟨-, ⟨p0, i0, b0, c0, hpart⟩, hle⟩ := hE d L hd hL
    simp only at hpart hle
    obtain ⟨hi, h1, h2⟩ := partAt_spec hq hk
    have hjL : memberIx a m0 t + j * L <
        a.seq.starts[partAt a.seq (memberIx a m0 t)]! + a.seq.parts[partAt a.seq (memberIx a m0 t)]!.count := by
      have h3 : j + 1 ≤ (a.seq.starts[partAt a.seq (memberIx a m0 t)]! +
          a.seq.parts[partAt a.seq (memberIx a m0 t)]!.count - memberIx a m0 t + L - 1) / L := by omega
      rw [Nat.le_div_iff_mul_le (Nat.pos_of_ne_zero hL), Nat.add_mul, Nat.one_mul] at h3
      generalize j * L = q at h3 ⊢
      omega
    rw [refAt_eq hq hk, toArray_getD_part hq hi h1 h2, toArray_getD_part hq hi (le_trans h1 (Nat.le_add_right _ _)) hjL, hpart]
    change refVal xs st vals ⟨p0, i0, b0 + (memberIx a m0 t - a.seq.starts[partAt a.seq (memberIx a m0 t)]!) + j * L⟩ =
      refVal xs st vals ⟨p0, i0, b0 + (memberIx a m0 t + j * L - a.seq.starts[partAt a.seq (memberIx a m0 t)]!)⟩
    have e : b0 + (memberIx a m0 t - a.seq.starts[partAt a.seq (memberIx a m0 t)]!) + j * L =
        b0 + (memberIx a m0 t + j * L - a.seq.starts[partAt a.seq (memberIx a m0 t)]!) := by
      generalize j * L = q
      omega
    rw [e]

/-- **An input committed for one member is its reference**: `Work.commit` of member `m0`'s reference's form. -/
theorem commit_input {V : ℕ → Bool} {w : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {a : Arg} {k : ℕ} (hq : seqOk a.seq = true) {r : Ref}
    {f : Form} (hr : getRef a.seq k = .ok r) (hf : w.formOf st r = .ok f) (hV : RowsHold V (w.commit f).1) :
    V (w.commit f).2 = refVal xs st vals (a.seq.toArray.getD k default) := by
  obtain ⟨hk, rfl⟩ := getRef_ok hr
  rw [commit_val w f hV, formOf_val hF hP hf, refAt_eq hq hk]

theorem forall₂_mem {α β : Type} {P : α → β → Prop} :
    ∀ {l : List α} {ls : List β}, List.Forall₂ P l ls → ∀ b ∈ ls, ∃ a ∈ l, P a b
  | _, _, .nil => fun _ hb => absurd hb List.not_mem_nil
  | _, _, .cons hab hl => fun b hb => by
    rcases List.mem_cons.mp hb with rfl | hb
    · exact ⟨_, List.mem_cons_self .., hab⟩
    · obtain ⟨a, ha, h⟩ := forall₂_mem hl b hb
      exact ⟨a, List.mem_cons_of_mem _ ha, h⟩


/-! ## The batch case: its pieces -/

/-- Work `b` extends work `a`: the same inputs, outputs and calls, more rows and columns. -/
def WR (a b : Work) : Prop :=
  b.ins = a.ins ∧ b.outs = a.outs ∧ b.calls = a.calls ∧ a.rows.toList <+: b.rows.toList ∧ a.next ≤ b.next

theorem wr_refl (a : Work) : WR a a := ⟨rfl, rfl, rfl, List.prefix_refl _, le_refl _⟩

theorem wr_trans {a b c : Work} (h1 : WR a b) (h2 : WR b c) : WR a c :=
  ⟨h2.1.trans h1.1, h2.2.1.trans h1.2.1, h2.2.2.1.trans h1.2.2.1, h1.2.2.2.1.trans h2.2.2.2.1,
    le_trans h1.2.2.2.2 h2.2.2.2.2⟩

theorem commit_next (w : Work) (f : Form) : w.next ≤ (w.commit f).1.next := by
  unfold Work.commit
  split
  · exact le_refl _
  · simp [Work.row]

theorem commit_wr (w : Work) (f : Form) : WR w (w.commit f).1 :=
  let ⟨a, b, c, d⟩ := commit_frame w f
  ⟨a, b, c, d, commit_next w f⟩

theorem formsOk_wr {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} (hF : FormsOk V w vals) (h : WR w w') :
    FormsOk V w' vals := by
  unfold FormsOk
  rw [h.2.1]
  exact hF

theorem paramsOk_wr {V : ℕ → Bool} {w w' : Work} {xs : List Bool} (hP : ParamsOk V w.ins xs) (h : WR w w') :
    ParamsOk V w'.ins xs := by
  rw [h.1]
  exact hP

theorem memberVal_none {xs : List Bool} {st : Array ℕ} {vals : Array (List Bool)} {p : Arg × ℕ}
    (hd : p.1.dims = none) (m : ℕ) : memberVal xs st vals m p = memberVal xs st vals 0 p := by
  simp only [memberVal, memberIx_none hd]

/-- What `Work.batch` binds input `p` to for every member: nothing when mapped, else a binding stepping 0 whose atom is
its reference's value once the rows past `W` hold. -/
def SharedOk (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (W : Work) (p : Arg × ℕ)
    (ob : Option Bind) : Prop :=
  (p.1.dims.isSome = true → ob = none) ∧
  (p.1.dims = none → ∃ b, ob = some b ∧ b.step = 0 ∧
    ∀ W' : Work, W.rows.toList <+: W'.rows.toList → RowsHold V W' → V b.a = memberVal xs st vals 0 p)

/-- A piece's bindings: each stepping an even number of atoms, copy `j`'s the batch's inputs for member `m0 + j`. -/
def PieceOk (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (ins : Array (Arg × ℕ))
    (pc : ℕ × ℕ × Array Bind) : Prop :=
  (∀ b ∈ pc.2.2.toList, b.step % 2 = 0) ∧
    ∀ j < pc.2.1, pc.2.2.toList.map (fun b => V (b.a + j * b.step)) = ins.toList.map (memberVal xs st vals (pc.1 + j))

theorem getElem!_mem {α : Type} [Inhabited α] {xs : Array α} {j : ℕ} (hj : j < xs.size) : xs[j]! ∈ xs.toList := by
  rw [getElem!_pos xs j hj]
  exact Array.getElem_mem_toList hj

/-- **`Work.pieces` covers members `[m0, n)`**, each by a piece whose bindings are its members' inputs. -/
theorem pieces_spec {V : ℕ → Bool} {w0 : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w0 vals) (hP : ParamsOk V w0.ins xs) {ins : Array (Arg × ℕ)} {shared : Array (Option Bind)}
    (hq : ∀ p ∈ ins.toList, seqOk p.1.seq = true) {Wsh : Work}
    (hsh : ∀ j < ins.size, SharedOk V xs st vals Wsh ins[j]! shared[j]!) (hWsh : WR Wsh w0) {n : ℕ} :
    ∀ k m0 (w : Work) acc r, n - m0 = k → WR w0 w → w.pieces st ins shared n m0 acc = .ok r →
      WR w r.1 ∧ acc.toList <+: r.2.toList ∧
      ∀ W : Work, r.1.rows.toList <+: W.rows.toList → RowsHold V W → ∀ m, m0 ≤ m → m < n →
        ∃ pc ∈ r.2.toList, pc.1 ≤ m ∧ m < pc.1 + pc.2.1 ∧ PieceOk V xs st vals ins pc := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
  intro m0 w acc r hk hw h
  have hFw := formsOk_wr hF hw
  have hPw := paramsOk_wr hP hw
  rw [Work.pieces.eq_1] at h
  by_cases hn : n ≤ m0
  · rw [dif_pos hn] at h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨wr_refl _, List.prefix_refl _, fun _ _ _ m h1 h2 => absurd (lt_of_le_of_lt h1 h2) (by omega)⟩
  · rw [dif_neg hn] at h
    simp only [bind, Except.bind] at h
    cases hpe : pieceEnd ins n m0 with
    | error e => rw [hpe] at h; simp at h
    | ok o =>
      rw [hpe] at h
      simp only at h
      by_cases h1 : m0 + 1 < o.getD (m0 + 1)
      · rw [dif_pos h1] at h
        cases o with
        | none => simp at h1
        | some m1 =>
          simp only [Option.getD_some] at h1 h
          obtain ⟨hm1n, hE⟩ := pieceEnd_spec hpe
          split at h
          · simp at h
          rename_i bs hbs
          obtain ⟨ih1, ih2, ih3⟩ := ih (n - m1) (by omega) m1 w _ r rfl hw h
          refine ⟨ih1, (List.prefix_append _ _).trans (by simpa [Array.toList_push] using ih2), ?_⟩
          intro W hW hR m hm0 hmn
          by_cases hm : m < m1
          · refine ⟨(m0, m1 - m0, bs), ih2.subset (by simp [Array.toList_push]), hm0,
              by simp only; omega, ?_⟩
            have hWw : w.rows.toList <+: W.rows.toList := ih1.2.2.2.1.trans hW
            obtain ⟨ls, hl1, hl2⟩ := foldlM_pushes
              (P := fun (j : ℕ) (b : Bind) => b.step % 2 = 0 ∧
                ∀ jj < m1 - m0, V (b.a + jj * b.step) = memberVal xs st vals (m0 + jj) ins[j]!)
              (S := List.range ins.size) (fun bs j bs' hj hF' => by
                have hj' : j < ins.size := List.mem_range.mp hj
                cases hs : shared[j]! with
                | some b =>
                  rw [hs] at hF'
                  simp only [pure, Except.pure, Except.ok.injEq] at hF'
                  subst hF'
                  refine ⟨b, rfl, ?_⟩
                  obtain ⟨hs1, hs2⟩ := hsh j hj'
                  have hd : ins[j]!.1.dims = none := by
                    cases hd' : ins[j]!.1.dims with
                    | none => rfl
                    | some _ => rw [hs1 (by simp [hd'])] at hs; simp at hs
                  obtain ⟨b', hb', hst, hv⟩ := hs2 hd
                  rw [hs] at hb'
                  simp only [Option.some.injEq] at hb'
                  subst hb'
                  refine ⟨by omega, fun jj _ => ?_⟩
                  rw [hst, Nat.mul_zero, Nat.add_zero, memberVal_none hd,
                    hv W ((hWsh.2.2.2.1.trans hw.2.2.2.1).trans hWw) hR]
                | none =>
                  rw [hs] at hF'
                  rcases hp : ins[j]! with ⟨a, t⟩
                  try simp only [hp] at hF'
                  try simp only [bind, Except.bind] at hF'
                  split at hF'
                  · simp at hF'
                  rename_i rf hr
                  split at hF'
                  · simp at hF'
                  rename_i b hb
                  simp only [pure, Except.pure, Except.ok.injEq] at hF'
                  subst hF'
                  refine ⟨b, rfl, ?_⟩
                  have hmem : (a, t) ∈ ins.toList := hp ▸ getElem!_mem hj'
                  cases hd : a.dims with
                  | none =>
                    obtain ⟨b', hb', -⟩ := (hsh j hj').2 (by rw [hp]; exact hd)
                    rw [hs] at hb'
                    simp at hb'
                  | some dL =>
                    obtain ⟨d, L⟩ := dL
                    simp only [hd, Option.map_some, Option.getD_some] at hb
                    obtain ⟨he, hv⟩ := step_input hFw hPw (hq _ hmem) hd (hE _ hmem) hr hb
                    exact ⟨he, fun jj hjj => by rw [hv jj hjj]⟩)
              (List.range ins.size) #[] bs (fun _ h => h) hbs
            simp only [List.nil_append] at hl1
            refine ⟨fun b hb => ?_, fun jj hjj => ?_⟩
            · simp only at hb
              rw [hl1] at hb
              obtain ⟨j, -, hj⟩ := forall₂_mem hl2 b hb
              exact hj.1
            · simp only at hjj ⊢
              rw [hl1, forall₂_map (f := fun j => memberVal xs st vals (m0 + jj) ins[j]!)
                (fun j b hb => hb.2 jj hjj) hl2, toList_eq_range ins, List.map_map]
              rfl
          · exact ih3 W hW hR m (by omega) hmn
      · rw [dif_neg h1] at h
        split at h
        · simp at h
        rename_i r1 hr1
        obtain ⟨w', bs⟩ := r1
        simp only at h
        obtain ⟨hww', ls, hl1, hl2⟩ := foldlM_pushesW (R := WR) wr_refl (fun _ _ _ => wr_trans)
          (P := fun (j : ℕ) (b : Bind) (W' : Work) => b.step = 0 ∧ ∀ W : Work, W'.rows.toList <+: W.rows.toList → RowsHold V W →
            V b.a = memberVal xs st vals m0 ins[j]!)
          (fun j b W1 W2 hb h12 => ⟨hb.1, fun W hW hR => hb.2 W (h12.2.2.2.1.trans hW) hR⟩)
          (S := List.range ins.size) (w₀ := w0) (fun w1 bs j r1 hj hw1 hF' => by
            have hj' : j < ins.size := List.mem_range.mp hj
            cases hs : shared[j]! with
            | some b =>
              rw [hs] at hF'
              simp only [pure, Except.pure, Except.ok.injEq] at hF'
              subst hF'
              refine ⟨b, rfl, wr_refl _, ?_⟩
              obtain ⟨hs1, hs2⟩ := hsh j hj'
              have hd : ins[j]!.1.dims = none := by
                cases hd' : ins[j]!.1.dims with
                | none => rfl
                | some _ => rw [hs1 (by simp [hd'])] at hs; simp at hs
              obtain ⟨b', hb', hst, hv⟩ := hs2 hd
              rw [hs] at hb'
              simp only [Option.some.injEq] at hb'
              subst hb'
              exact ⟨hst, fun W hW hR => by
                rw [memberVal_none hd, hv W ((hWsh.2.2.2.1.trans hw1.2.2.2.1).trans hW) hR]⟩
            | none =>
              rw [hs] at hF'
              rcases hp : ins[j]! with ⟨a, t⟩
              try simp only [hp] at hF'
              try simp only [bind, Except.bind] at hF'
              split at hF'
              · simp at hF'
              rename_i rf hr
              split at hF'
              · simp at hF'
              rename_i f hf
              simp only [pure, Except.pure, Except.ok.injEq] at hF'
              subst hF'
              have hmem : (a, t) ∈ ins.toList := hp ▸ getElem!_mem hj'
              refine ⟨⟨(w1.commit f).2, 0⟩, rfl, commit_wr w1 f, rfl, fun W hW hR => ?_⟩
              exact commit_input (formsOk_wr hF hw1) (paramsOk_wr hP hw1) (hq _ hmem) hr hf
                (rowsHold_of_prefix hW hR))
          (List.range ins.size) w #[] (w', bs) (fun _ h => h) hw hr1
        simp only [List.nil_append] at hl1
        obtain ⟨ih1, ih2, ih3⟩ := ih (n - (m0 + 1)) (by omega) (m0 + 1) w' _ r rfl (wr_trans hw hww') h
        refine ⟨wr_trans hww' ih1, (List.prefix_append _ _).trans (by simpa [Array.toList_push] using ih2), ?_⟩
        intro W hW hR m hm0 hmn
        by_cases hm : m = m0
        · refine ⟨(m0, 1, bs), ih2.subset (by simp [Array.toList_push]), by simp only; omega, by simp only; omega, ?_⟩
          have hWw : w'.rows.toList <+: W.rows.toList := ih1.2.2.2.1.trans hW
          refine ⟨fun b hb => ?_, fun jj hjj => ?_⟩
          · simp only at hb
            rw [hl1] at hb
            obtain ⟨j, -, hj⟩ := forall₂_mem hl2 b hb
            rw [hj.1]
          · simp only at hjj ⊢
            have hjj0 : jj = 0 := by omega
            subst hjj0
            rw [hl1, forall₂_map (f := fun j => memberVal xs st vals (m0 + 0) ins[j]!)
              (fun j b hb => by simp only [Nat.zero_mul, Nat.add_zero]; exact hb.2 W hWw hR) hl2,
              toList_eq_range ins, List.map_map]
            rfl
        · exact ih3 W hW hR m (by omega) hmn


/-! ## The batch case: its inputs, its shared bindings, a batch of none -/

/-- A fold over a work whose every step extends it, `R` of the first. -/
theorem foldlM_frame {α β : Type} {F : Work × β → α → Except String (Work × β)} {S : List α} {w₀ : Work}
    (hF : ∀ s a r, a ∈ S → WR w₀ s.1 → F s a = .ok r → WR s.1 r.1) :
    ∀ (l : List α) (s r : Work × β), (∀ a ∈ l, a ∈ S) → WR w₀ s.1 → l.foldlM F s = .ok r → WR s.1 r.1 := by
  intro l
  induction l with
  | nil =>
    intro s r _ _ h
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact wr_refl _
  | cons a l ih =>
    intro s r hS hw h
    rw [List.foldlM_cons] at h
    cases ha : F s a with
    | error e => rw [ha] at h; simp [bind, Except.bind] at h
    | ok s1 =>
      rw [ha] at h
      simp only [bind, Except.bind] at h
      have h1 := hF s a s1 (hS a (by simp)) hw ha
      exact wr_trans h1 (ih s1 r (fun b hb => hS b (by simp [hb])) (wr_trans hw h1) h)

/-- `Work.binds` only extends the work. -/
theorem binds_wr {w w' : Work} {st : Array ℕ} {callee : Defn} {x : Node} {bs : Array Bind}
    (hb : w.binds st callee x = .ok (w', bs)) : WR w w' := by
  unfold Work.binds at hb
  refine foldlM_frame (S := List.range x.args.size) (w₀ := w) (fun s i r _ _ h => ?_) _ (w, #[]) (w', bs)
    (fun _ h => h) (wr_refl _) hb
  refine foldlM_frame (S := List.range _) (w₀ := s.1) (fun s' t r' _ _ h' => ?_) _ s r (fun _ h => h) (wr_refl _) h
  obtain ⟨w1, out⟩ := s'
  simp only [bind, Except.bind, pure, Except.pure] at h'
  repeat' (first | (simp at h'; done) | split at h')
  all_goals
    simp only [Except.ok.injEq] at h'
    subst h'
    first | exact wr_refl _ | exact commit_wr _ _

theorem foldl_push_flat {α β : Type} (g : α → List β) :
    ∀ (l : List α) (acc : Array β), (l.foldl (fun acc i => (g i).foldl (fun acc t => acc.push t) acc) acc).toList =
      acc.toList ++ l.flatMap g
  | [], acc => by simp
  | i :: l, acc => by
    simp only [List.foldl_cons, List.flatMap_cons]
    rw [foldl_push_flat g l]
    have : ((g i).foldl (fun acc t => acc.push t) acc).toList = acc.toList ++ g i := by
      have := foldl_push_map id (g i) acc
      simp only [id] at this
      rw [this]
      simp
    rw [this, List.append_assoc]

/-- `Work.batch`'s inputs: each argument's references, as many as its parameter's leaves when mapped. -/
theorem batchIns_toList (callee : Defn) (x : Node) :
    ((List.range x.args.size).foldl (fun acc i =>
      let a := x.args[i]!
      let cnt := if a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
      (List.range cnt).foldl (fun acc t => acc.push (a, t)) acc) #[]).toList =
    (List.range x.args.size).flatMap fun i =>
      (List.range (if (x.args[i]!).dims.isSome then (callee.params[i]!.2).leaves else (x.args[i]!).seq.count)).map
        fun t => (x.args[i]!, t) := by
  have := foldl_push_flat (fun i => (List.range (if (x.args[i]!).dims.isSome then (callee.params[i]!.2).leaves
    else (x.args[i]!).seq.count)).map fun t => (x.args[i]!, t)) (List.range x.args.size) #[]
  simp only [List.foldl_map] at this
  simpa using this

/-- Member `m`'s inputs as the walk reads them (`walkStep`'s batch). -/
def batchArgs (v : Ref → Bool) (callee : Defn) (x : Node) (m : ℕ) : List Bool :=
  (List.range x.args.size).flatMap fun i =>
    match (x.args.getD i default).dims with
    | none => argVals v (x.args.getD i default) (x.args.getD i default).seq.count id
    | some (dd, L) => argVals v (x.args.getD i default) ((callee.params.getD i default).2.leaves)
        fun t => (t / L * dd) * L + t % L + m * L

theorem getElem!_eq_getD {α : Type} [Inhabited α] (xs : Array α) (i : ℕ) : xs[i]! = xs.getD i default := by
  by_cases h : i < xs.size
  · rw [getElem!_pos xs i h, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h, Option.getD_some]
  · rw [getElem!_neg xs i h, Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none]

theorem batchArgs_eq (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node) (m : ℕ) :
    ((List.range x.args.size).flatMap fun i =>
      (List.range (if (x.args[i]!).dims.isSome then (callee.params[i]!.2).leaves else (x.args[i]!).seq.count)).map
        fun t => (x.args[i]!, t)).map (memberVal xs st vals m) = batchArgs (refVal xs st vals) callee x m := by
  rw [List.map_flatMap]
  unfold batchArgs
  congr 1
  funext i
  rw [getElem!_eq_getD x.args i, getElem!_eq_getD callee.params i]
  cases hd : (x.args.getD i default).dims with
  | none =>
    simp only [hd, Option.isSome_none, Bool.false_eq_true, ite_false, List.map_map, argVals]
    apply List.map_congr_left
    intro t _
    simp only [Function.comp, memberVal, memberIx_none hd, id]
  | some dL =>
    obtain ⟨dd, L⟩ := dL
    simp only [hd, Option.isSome_some, ite_true, List.map_map, argVals]
    apply List.map_congr_left
    intro t _
    simp only [Function.comp, memberVal, memberIx, hd]
    congr 2
    ring

theorem ins_seq {callee : Defn} {x : Node} (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true) :
    ∀ p ∈ ((List.range x.args.size).flatMap fun i =>
      (List.range (if (x.args[i]!).dims.isSome then (callee.params[i]!.2).leaves else (x.args[i]!).seq.count)).map
        fun t => (x.args[i]!, t)), seqOk p.1.seq = true := by
  intro p hp
  obtain ⟨i, hi, hp⟩ := List.mem_flatMap.mp hp
  obtain ⟨t, -, rfl⟩ := List.mem_map.mp hp
  exact hseq _ (getElem!_mem (List.mem_range.mp hi))

theorem formsOk_outs {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} (h : w.outs = w'.outs)
    (hF : FormsOk V w vals) : FormsOk V w' vals := by
  unfold FormsOk at hF ⊢
  rw [← h]
  exact hF

theorem forall₂_getElem! {α β : Type} [Inhabited α] [Inhabited β] {R : α → β → Prop} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ R xs.toList ys.toList) {j : ℕ} (hj : j < xs.size) : R xs[j]! ys[j]! := by
  have hl := h.length_eq
  simp only [Array.length_toList] at hl
  rw [getElem!_pos xs j hj, getElem!_pos ys j (by omega)]
  have := (List.forall₂_iff_get.mp h).2 j (by simpa using hj) (by simp; omega)
  simpa using this

/-- **A batch node**: its members' copies one region after the work, member `m`'s copy a copy of a CALL binding the
walk's inputs for member `m`, and its leaves the members' returns one after another. -/
theorem batch_node {V : ℕ → Bool} {w w' : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    {callee : Defn} {x : Node} {e cols : ℕ} {rets : Array ℕ}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hb : w.batch st callee x e cols rets = .ok w') :
    ∃ base, w'.ins = w.ins ∧ w.rows.toList <+: w'.rows.toList ∧ w.calls.toList <+: w'.calls.toList ∧
      w.next ≤ base ∧ w'.next = base + x.n * cols ∧ w'.outs = w.outs.push (.copies base cols x.n rets) ∧
      (RowsHold V w' → ∀ m < x.n, ∃ p ∈ w'.calls.toList, p.def_ = e ∧ p.stride = cols ∧ ∃ j < p.count,
        p.base + j * cols = base + m * cols ∧ (∀ b ∈ p.binds.toList, b.step % 2 = 0) ∧
        p.binds.toList.map (fun b => V (b.a + j * b.step)) = batchArgs (refVal xs st vals) callee x m) := by
  unfold Work.batch at hb
  by_cases hn0 : (x.n == 0) = true
  · rw [ite_eq_left hn0] at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · simp at hb
    rename_i wb hwb
    obtain ⟨w1, bs⟩ := wb
    simp only [pure, Except.pure, Except.ok.injEq] at hb
    subst hb
    have hW := binds_wr hwb
    have hn : x.n = 0 := by simpa using hn0
    refine ⟨w1.next, hW.1, hW.2.2.2.1, ?_, hW.2.2.2.2, by simp [Work.place, hn], by simp [Work.place, hn, hW.2.1],
      fun _ m hm => absurd hm (by omega)⟩
    simp only [Work.place, Array.toList_push, hW.2.2.1]
    exact List.prefix_append _ _
  · rw [ite_eq_right hn0] at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · simp at hb
    rename_i ws hws
    obtain ⟨w1, shared⟩ := ws
    simp only at hb
    split at hb
    · simp at hb
    rename_i pr hpr
    obtain ⟨w2, ps⟩ := pr
    simp only [pure, Except.pure, Except.ok.injEq] at hb
    subst hb
    have hq := ins_seq (callee := callee) hseq
    rw [← batchIns_toList] at hq
    rw [← Array.foldlM_toList] at hws
    obtain ⟨hW1, ls, hl1, hl2⟩ := foldlM_pushesW (R := WR) wr_refl (fun _ _ _ => wr_trans)
      (P := fun p ob W' => SharedOk V xs st vals W' p ob)
      (fun p ob W1 W2 h h12 => ⟨h.1, fun hd => by
        obtain ⟨b, h1, h2, h3⟩ := h.2 hd
        exact ⟨b, h1, h2, fun W' hW hR => h3 W' (h12.2.2.2.1.trans hW) hR⟩⟩)
      (S := _) (w₀ := w) (fun w0' acc p r hp hw0 hstep => by
        obtain ⟨a, t⟩ := p
        simp only at hstep
        by_cases hs : a.dims.isSome = true
        · rw [ite_eq_left hs] at hstep
          cases hstep
          exact ⟨none, rfl, wr_refl _, fun _ => rfl, fun hd => by simp only at hd; simp [hd] at hs⟩
        · rw [ite_eq_right hs] at hstep
          try simp only [bind, Except.bind] at hstep
          split at hstep
          · simp at hstep
          rename_i rf hr
          split at hstep
          · simp at hstep
          rename_i f hf
          cases hstep
          refine ⟨some ⟨(w0'.commit f).2, 0⟩, rfl, commit_wr _ _, fun h => absurd h hs, fun hd =>
            ⟨_, rfl, rfl, fun W' hW hR => ?_⟩⟩
          simp only [memberVal, memberIx_none hd]
          exact commit_input (formsOk_wr hF hw0) (paramsOk_wr hP hw0) (hq _ hp) hr hf (rowsHold_of_prefix hW hR))
      _ w #[] (w1, shared) (fun _ h => h) (wr_refl _) hws
    simp only [List.nil_append] at hl1
    rw [← hl1] at hl2
    have hsh : ∀ j < _, SharedOk V xs st vals w1 _ _ := fun j hj => forall₂_getElem! hl2 hj
    obtain ⟨hW2, -, hcov⟩ := pieces_spec (formsOk_wr hF hW1) (paramsOk_wr hP hW1) hq hsh (wr_refl _) _ 0 w1 #[]
      (w2, ps) rfl (wr_refl _) hpr
    simp only at hW2 hcov
    have hW := wr_trans hW1 hW2
    refine ⟨w2.next, hW.1, hW.2.2.2.1, ?_, hW.2.2.2.2, rfl, by simp [hW.2.1], fun hR m hm => ?_⟩
    · simp only [Array.toList_append, hW.2.2.1]
      exact List.prefix_append _ _
    obtain ⟨pc, hpc, hle, hlt, hok⟩ := hcov _ (List.prefix_refl _) hR m (Nat.zero_le _) hm
    obtain ⟨m0, cnt, bs⟩ := pc
    have hle' : m0 ≤ m := hle
    have hlt' : m < m0 + cnt := hlt
    refine ⟨⟨e, cnt, w2.next + m0 * cols, cols, bs⟩, ?_, rfl, rfl, m - m0, by show m - m0 < cnt; omega, ?_, hok.1, ?_⟩
    · simp only [Array.toList_append, Array.toList_map, List.mem_append, List.mem_map]
      exact Or.inr ⟨(m0, cnt, bs), hpc, rfl⟩
    · simp only
      rw [Nat.add_assoc, ← Nat.add_mul, Nat.add_sub_cancel' hle']
    · simp only
      rw [hok.2 (m - m0) (by show m - m0 < cnt; omega), Nat.add_sub_cancel' hle', batchIns_toList, batchArgs_eq]

/-- **A batch node's leaves are its members' returns** one after another, `rets.size` each. -/
theorem batch_forms {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} {base n cols : ℕ} {rets : Array ℕ}
    (hF : FormsOk V w vals) (ho : w'.outs = w.outs.push (.copies base cols n rets)) {l : ℕ → List Bool}
    (hlen : ∀ m < n, (l m).length = rets.size)
    (hl : ∀ m < n, ∀ k < rets.size, V (2 * (base + m * cols + rets[k]!)) = (l m).getD k false) :
    FormsOk V w' (vals.push ((List.range n).map l).flatten) := by
  refine formsOk_outs (w := { w with outs := w.outs.push (.copies base cols n rets) }) ho.symm
    (formsOk_push hF rfl ?_)
  intro j hj
  have hL : 0 < rets.size := Nat.pos_of_ne_zero (fun h0 => by rw [h0] at hj; simp at hj)
  have hjn : j / rets.size < n := by rw [Nat.div_lt_iff_lt_mul hL]; exact hj
  rw [getD_flatten_eq (L := rets.size) (fun l' hl' => by
      obtain ⟨m, hm, rfl⟩ := List.mem_map.mp hl'
      exact hlen m (List.mem_range.mp hm)) (by simpa using hj),
    List.getElem?_map, List.getElem?_range hjn]
  simp only [Option.map_some, Option.getD_some]
  rw [← hl _ hjn _ (Nat.mod_lt _ hL)]


/-- **A primitive's template Definition computes its gate** (`gateDef`): its one return column, once its rows hold, is
the gate on its inputs. `hcol` is `Def.wellFormed`'s rows inside the columns, which `selfCheck` holds it to. -/
theorem gateTmpl {g : Gate} {ins : ℕ} {w : Work} {o : Form}
    (hop : ({} : Work).op g ((Array.range ins).map Form.input) = .ok (w, o)) {v : ℕ → Bool} {inp : List Bool}
    (hlen : inp.length = ins)
    (hcol : ∀ r ∈ ((w.commitOwn o).1.finish ins).rows.toList, r.1 < ((w.commitOwn o).1.finish ins).cols)
    (hH : ∀ r ∈ ((w.commitOwn o).1.finish ins).rows.toList,
      v r.1 = rowVal (((w.commitOwn o).1.finish ins).refV v inp true) r) :
    (w.commitOwn o).2 < ((w.commitOwn o).1.finish ins).cols ∧ v (w.commitOwn o).2 = gateVal g inp := by
  have hB := op_bound hop (fun o' ho => by
    rw [Array.toList_map, List.mem_map] at ho
    obtain ⟨k, -, rfl⟩ := ho
    exact formB_input _ _)
  have hR := rowsHold_finish hcol hH
  obtain ⟨-, -, -, hpre, -, hc, hval⟩ := commitOwn_spec (V := atomV ((w.commitOwn o).1.finish ins) v inp) w o hB.2
  refine ⟨hc, ?_⟩
  have h1 := hval hR
  rw [atomV_own hc] at h1
  rw [h1]
  obtain ⟨-, h2⟩ := op_val hop (rowsHold_of_prefix hpre hR)
  rw [h2]
  congr 1
  rw [Array.toList_map, List.map_map, Array.toList_range]
  apply List.ext_getElem
  · simp [hlen]
  · intro i hi1 hi2
    have hi : i < ins := by simpa using hi1
    simp only [List.getElem_map, List.getElem_range, Function.comp, val_input]
    rw [atomV_input (D := (w.commitOwn o).1.finish ins) (show i < ins from hi), List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem hi2, Option.getD_some]


/-! ## The scan case: its loops, a stepping `xs` argument -/

/-- A loop every step of which yields, keeping `Inv` of the elements it has run through. -/
theorem forIn_inv {α σ : Type} {f : α → σ → Except String (ForInStep σ)} {S : List α} {Inv : List α → σ → Prop}
    (hf : ∀ pre a s r, pre ++ [a] <+: S → Inv pre s → f a s = .ok r → ∃ s', r = .yield s' ∧ Inv (pre ++ [a]) s') :
    ∀ (l pre : List α) (s r : σ), pre ++ l = S → Inv pre s → forIn l s f = .ok r → Inv S r := by
  intro l
  induction l with
  | nil =>
    intro pre s r hS hI h
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    rw [List.append_nil] at hS
    subst hS
    exact hI
  | cons a l ih =>
    intro pre s r hS hI h
    rw [List.forIn_cons] at h
    cases ha : f a s with
    | error e => rw [ha] at h; simp [bind, Except.bind] at h
    | ok r1 =>
      rw [ha] at h
      obtain ⟨s', rfl, hI'⟩ := hf pre a s r1 (by rw [← hS]; exact ⟨l, by simp⟩) hI ha
      simp only [bind, Except.bind] at h
      exact ih (pre ++ [a]) s' r (by rw [← hS]; simp) hI' h

theorem forIn_range {σ : Type} (n : ℕ) (s : σ) (f : ℕ → σ → Except String (ForInStep σ)) :
    forIn [:n] s f = forIn (List.range n) s f := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', List.range_eq_range']
  simp [Std.Legacy.Range.size]

theorem prefix_range {l : List ℕ} {n : ℕ} (h : l <+: List.range n) : l = List.range l.length := by
  rw [List.prefix_iff_eq_take.mp h, List.take_range, List.length_range]

theorem prefix_range_last {pre : List ℕ} {i n : ℕ} (h : pre ++ [i] <+: List.range n) :
    pre = List.range pre.length ∧ i = pre.length := by
  have h1 := prefix_range h
  simp only [List.length_append, List.length_singleton, List.range_succ] at h1
  have h2 := List.append_inj h1 (by simp)
  simp only [List.cons.injEq, and_true] at h2
  exact ⟨h2.1, h2.2⟩

theorem forall₂_append {α β : Type} {R : α → β → Prop} :
    ∀ {l1 : List α} {k1 : List β} {l2 : List α} {k2 : List β}, List.Forall₂ R l1 k1 → List.Forall₂ R l2 k2 →
      List.Forall₂ R (l1 ++ l2) (k1 ++ k2)
  | _, _, _, _, .nil, h => h
  | _, _, _, _, .cons hab hl, h => .cons hab (forall₂_append hl h)

/-- The part of `q` that `partIx` names holds reference `k`. -/
theorem partIx_mem {q : Seq} (hq : seqOk q = true) {k : ℕ} (hk : k < q.count) :
    partIx q k < q.parts.size ∧ q.starts[partIx q k]! ≤ k ∧ k < q.starts[partIx q k]! + q.parts[partIx q k]!.count := by
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
  have hlen : ∀ i < q.parts.size, q.starts[i]! + q.parts[i]!.count = ((q.parts.toList.take (i + 1)).map Part.count).sum :=
    fun i hi => by rw [hst i hi, InferenceOnlyVerifier.Properties.SecurityProofs.Walk.sum_take_succ_map q.parts hi]
  have hpi : partIx q k = (List.range q.parts.size).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0 := rfl
  rw [hpi]
  refine ⟨r1, r2, ?_⟩
  set r := (List.range q.parts.size).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0
  rcases Nat.lt_or_ge (r + 1) q.parts.size with hl | hg
  · have := r3 (r + 1) (by omega) hl
    rw [hst (r + 1) hl] at this
    rw [hlen r r1]
    exact this
  · have hre : r + 1 = q.parts.toList.length := by simp; omega
    rw [hlen r r1, hre, List.take_length, ← hcnt]
    exact hk

/-- **A scan's `xs` reference stepping its element's leaves an iteration**: within one affine part, iteration `j`'s
reference is the walk's `k0 + j · L`. -/
theorem stepping_spec {q : Seq} (hq : seqOk q = true) {k0 n L : ℕ} {what : String} {r : Ref} {s : ℕ}
    (h : stepping q k0 n L what = .ok (r, s)) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) :
    (n ≤ 1 → s = 0 ∧ k0 < q.count ∧ r = refAt q k0) ∧
    (1 < n → s = L ∧ ∀ j < n, refVal xs st vals ⟨r.p, r.i, r.j + j * L⟩ =
      refVal xs st vals (q.toArray.getD (k0 + j * L) default)) := by
  unfold stepping at h
  by_cases hn : n ≤ 1
  · rw [ite_eq_left hn] at h
    refine ⟨fun _ => ?_, fun h1 => absurd h1 (by omega)⟩
    cases hg : getRef q k0 with
    | error e => rw [hg] at h; simp [Functor.map, Except.map] at h
    | ok r0 =>
      rw [hg] at h
      simp only [Functor.map, Except.map, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hk, rfl⟩ := getRef_ok hg
      exact ⟨rfl, hk, rfl⟩
  · rw [ite_eq_right hn] at h
    refine ⟨fun h1 => absurd h1 hn, fun _ => ?_⟩
    by_cases hc : q.count ≤ k0 + (n - 1) * L
    · rw [ite_eq_left hc] at h
      simp [throw, throwThe, MonadExceptOf.throw] at h
    · rw [ite_eq_right hc] at h
      by_cases hi : (partIx q (k0 + (n - 1) * L) != partIx q k0) = true
      · rw [ite_eq_left hi] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
      · rw [ite_eq_right hi] at h
        simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hi
        cases hpart : q.parts[partIx q k0]! with
        | affine p0 i0 b0 c0 =>
          simp only [hpart, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          refine ⟨rfl, fun j hj => ?_⟩
          have hjL : j * L ≤ (n - 1) * L := Nat.mul_le_mul_right L (by omega)
          obtain ⟨a1, a2, a3⟩ := partIx_mem hq (k := k0) (by omega)
          obtain ⟨-, -, b3⟩ := partIx_mem hq (k := k0 + (n - 1) * L) (by omega)
          rw [hi] at b3
          rw [toArray_getD_part hq a1 (le_trans a2 (Nat.le_add_right _ _)) (by omega), hpart]
          change refVal xs st vals ⟨p0, i0, b0 + (k0 - q.starts[partIx q k0]!) + j * L⟩ =
            refVal xs st vals ⟨p0, i0, b0 + (k0 + j * L - q.starts[partIx q k0]!)⟩
          have e : b0 + (k0 - q.starts[partIx q k0]!) + j * L = b0 + (k0 + j * L - q.starts[partIx q k0]!) := by
            generalize j * L = u at hjL ⊢
            omega
          rw [e]
        | strided => simp [hpart, throw, throwThe, MonadExceptOf.throw] at h
        | explicit => simp [hpart, throw, throwThe, MonadExceptOf.throw] at h
        | grid => simp [hpart, throw, throwThe, MonadExceptOf.throw] at h

/-- **`Work.bindOne`**: a reference committed when it steps 0, else bound stepping (`stepOf_val`). -/
theorem bindOne_spec {V : ℕ → Bool} {w w1 : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) {r : Ref} {s n : ℕ} {b : Bind}
    (h : w.bindOne st r s n = .ok (w1, b)) :
    WR w w1 ∧ b.step % 2 = 0 ∧
      (s = 0 → b.step = 0 ∧ ∀ W : Work, w1.rows.toList <+: W.rows.toList → RowsHold V W → V b.a = refVal xs st vals r) ∧
      (s ≠ 0 → ∀ j < n, V (b.a + j * b.step) = refVal xs st vals ⟨r.p, r.i, r.j + j * s⟩) := by
  unfold Work.bindOne at h
  by_cases hs : s = 0
  · subst hs
    simp only [beq_self_eq_true, ite_true, bind, Except.bind] at h
    split at h
    · simp at h
    rename_i f hf
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨commit_wr w f, rfl, fun _ => ⟨rfl, fun W hW hR => ?_⟩, fun h => absurd rfl h⟩
    rw [commit_val w f (rowsHold_of_prefix hW hR), formOf_val hF hP hf]
  · have hs' : (s == 0) = false := by simpa using hs
    simp only [hs', Bool.false_eq_true, ite_false, bind, Except.bind] at h
    split at h
    · simp at h
    rename_i b' hb
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨he, hv⟩ := stepOf_val hF hP hb
    exact ⟨wr_refl _, he, fun h => absurd h hs, fun _ j hj => hv j hj⟩


/-! ## The scan case: its bindings -/

/-- Iteration `m`'s value of leaf `t` of a scan's argument `i`, as the walk reads it: an `xs` argument's reference
stepped `m` elements, any other's reference `t`. -/
def scanArg (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node) (i t m : ℕ) : Bool :=
  refVal xs st vals ((x.args[i]!).seq.toArray.getD
    (if 0 < i ∧ i ≤ x.nxs then t + m * (callee.params[i]!.2).leaves else t) default)

/-- What `Work.scan` binds leaf `it.2` of argument `it.1` to: in the first CALL (`fr.1`) its value at iteration 0, in
the rest (`fr.2`, an argument past the carry) its value at iteration `j + 1` in copy `j`. -/
def ScanQ (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (it : ℕ × ℕ) (fr : Bind × Bind) (W' : Work) : Prop :=
  fr.1.step % 2 = 0 ∧ (0 < it.1 → fr.2.step % 2 = 0) ∧
    ∀ W : Work, W'.rows.toList <+: W.rows.toList → RowsHold V W →
      V fr.1.a = scanArg xs st vals callee x it.1 it.2 0 ∧
      (0 < it.1 → ∀ j, j + 1 < x.n → V (fr.2.a + j * fr.2.step) = scanArg xs st vals callee x it.1 it.2 (j + 1))

theorem scanQ_mono {V : ℕ → Bool} {xs : List Bool} {st : Array ℕ} {vals : Array (List Bool)} {callee : Defn}
    {x : Node} {it : ℕ × ℕ} {fr : Bind × Bind} {W1 W2 : Work} (h : ScanQ V xs st vals callee x it fr W1)
    (h12 : WR W1 W2) : ScanQ V xs st vals callee x it fr W2 :=
  ⟨h.1, h.2.1, fun W hW hR => h.2.2 W (h12.2.2.2.1.trans hW) hR⟩

/-- A scan's leaves bound, argument by argument. -/
def scanIdx (callee : Defn) (I : List ℕ) : List (ℕ × ℕ) :=
  I.flatMap fun i => (List.range (callee.params[i]!.2).leaves).map (i, ·)

/-- `Work.scan`'s loop over its arguments, through `pre`: the work extended, the first CALL's and the rest's bindings
`ScanQ` of each leaf so far, and the carry's leaves once argument 0 is through. -/
def ScanInv (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (w : Work) (pre : List ℕ) (s : Work × Array Bind × Array Bind × Array ℕ) : Prop :=
  WR w s.1 ∧ ∃ L : List (Bind × Bind), List.Forall₂ (fun it fr => ScanQ V xs st vals callee x it fr s.1)
    (scanIdx callee pre) L ∧ s.2.1.toList = L.map Prod.fst ∧ s.2.2.1.toList = L.map Prod.snd ∧
    s.2.2.2.toList = (if pre = [] then [] else List.range (callee.params[0]!.2).leaves)

/-- Its loop over argument `i`'s leaves, through `pre`, from state `s0`. -/
def ScanInvT (V : ℕ → Bool) (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (s0 : Work × Array Bind × Array Bind × Array ℕ) (i : ℕ) (pre : List ℕ)
    (s : Work × Array Bind × Array Bind × Array ℕ) : Prop :=
  WR s0.1 s.1 ∧ ∃ L : List (Bind × Bind), List.Forall₂ (fun it fr => ScanQ V xs st vals callee x it fr s.1)
    (pre.map (i, ·)) L ∧ s.2.1.toList = s0.2.1.toList ++ L.map Prod.fst ∧
    s.2.2.1.toList = s0.2.2.1.toList ++ L.map Prod.snd ∧ s.2.2.2.toList = s0.2.2.2.toList ++ (if i = 0 then pre else [])

theorem forall₂_mono {α β : Type} {R S : α → β → Prop} (h : ∀ a b, R a b → S a b) :
    ∀ {l : List α} {k : List β}, List.Forall₂ R l k → List.Forall₂ S l k
  | _, _, .nil => .nil
  | _, _, .cons hab hl => .cons (h _ _ hab) (forall₂_mono h hl)

theorem forall₂_map' {α β γ : Type} {P : α → β → Prop} {g : β → γ} {f : α → γ} :
    ∀ {l : List α} {ls : List β}, List.Forall₂ P l ls → (∀ a ∈ l, ∀ b, P a b → g b = f a) → ls.map g = l.map f
  | _, _, .nil, _ => rfl
  | _, _, .cons hab hl, h => by
    simp only [List.map_cons, h _ List.mem_cons_self _ hab,
      forall₂_map' hl (fun a ha => h a (List.mem_cons_of_mem _ ha))]

theorem forall₂_drop' {α β : Type} {R : α → β → Prop} :
    ∀ (n : ℕ) {l : List α} {k : List β}, List.Forall₂ R l k → List.Forall₂ R (l.drop n) (k.drop n)
  | 0, _, _, h => by simpa using h
  | _ + 1, _, _, .nil => .nil
  | n + 1, _, _, .cons _ h => by simpa using forall₂_drop' n h

theorem scanIdx_append (callee : Defn) (I J : List ℕ) :
    scanIdx callee (I ++ J) = scanIdx callee I ++ scanIdx callee J := by
  simp [scanIdx, List.flatMap_append]

theorem scanIdx_cons (callee : Defn) (i : ℕ) (I : List ℕ) :
    scanIdx callee (i :: I) = (List.range (callee.params[i]!.2).leaves).map (i, ·) ++ scanIdx callee I := by
  simp [scanIdx]

theorem scanIdx_single (callee : Defn) (i : ℕ) :
    scanIdx callee [i] = (List.range (callee.params[i]!.2).leaves).map (i, ·) := by
  simp [scanIdx]

/-- A scan's leaves: argument 0's, the carry's `init`, then the others', each of an argument past 0. -/
theorem scanIdx_range (callee : Defn) {k : ℕ} (hk : 0 < k) :
    ∃ T, scanIdx callee (List.range k) = (List.range (callee.params[0]!.2).leaves).map (0, ·) ++ T ∧
      ∀ it ∈ T, 0 < it.1 := by
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  rw [List.range_succ_eq_map, scanIdx_cons]
  refine ⟨_, rfl, fun it hit => ?_⟩
  simp only [scanIdx, List.mem_flatMap, List.mem_map] at hit
  obtain ⟨i, ⟨j, -, rfl⟩, t, -, rfl⟩ := hit
  simp

theorem restMap_toList {rest : Array Bind} {carries : Array ℕ} {L : List (Bind × Bind)} {c base cols : ℕ}
    {rets : Array ℕ} (hc : c ≤ rest.size) (hcar : carries.toList = List.range c) (hr : rest.toList = L.map Prod.snd) :
    ((Array.range rest.size).map fun k =>
      if k < c then ({ a := 2 * (base + rets[carries[k]!]!), step := 2 * cols } : Bind) else rest[k]!).toList =
      (List.range c).map (fun k => ({ a := 2 * (base + rets[k]!), step := 2 * cols } : Bind)) ++
        (L.drop c).map Prod.snd := by
  have hlen : rest.size = L.length := by rw [← Array.length_toList, hr, List.length_map]
  apply List.ext_getElem
  · simp; omega
  · intro k h1 h2
    simp only [Array.toList_map, Array.toList_range, List.getElem_map, List.getElem_range]
    by_cases hk : k < c
    · rw [ite_eq_left hk, List.getElem_append_left (by simpa using hk)]
      simp only [List.getElem_map, List.getElem_range]
      have hcs : k < carries.size := by rw [← Array.length_toList, hcar, List.length_range]; exact hk
      have : carries[k]! = k := by
        rw [getElem!_pos carries k hcs, ← Array.getElem_toList]
        simp [hcar]
      rw [this]
    · rw [ite_eq_right hk, List.getElem_append_right (by simp; omega)]
      simp only [List.getElem_map, List.getElem_drop, List.length_map, List.length_range]
      have hks : k < rest.size := by simpa using h1
      rw [getElem!_pos rest k hks, ← Array.getElem_toList]
      simp only [hr, List.getElem_map]
      congr 2
      omega

/-- **A scan node**: its `n` iterations one region after the work, iteration 0 the first CALL's copy, binding the
walk's `init` and the other arguments' iteration-0 leaves, and iteration `m ≥ 1` copy `m − 1` of the second, binding
iteration `m − 1`'s carry returns and the other arguments' iteration-`m` leaves. -/
theorem scan_node {V : ℕ → Bool} {w w' : Work} {st : Array ℕ} {xs : List Bool} {vals : Array (List Bool)}
    {callee : Defn} {x : Node} {e cols : ℕ} {rets : Array ℕ}
    (hF : FormsOk V w vals) (hP : ParamsOk V w.ins xs) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hs : w.scan st callee x e cols rets = .ok w') :
    ∃ base, w'.ins = w.ins ∧ w.rows.toList <+: w'.rows.toList ∧ w.calls.toList <+: w'.calls.toList ∧
      w.next ≤ base ∧ w'.next = base + x.n * cols ∧
      w'.outs = w.outs.push (.scan base cols rets (callee.params[0]!.2).leaves x.n) ∧
      0 < x.n ∧ 0 < x.args.size ∧ (callee.params[0]!.2).leaves ≤ rets.size ∧
      (RowsHold V w' → ∀ m < x.n, ∃ p ∈ w'.calls.toList, p.def_ = e ∧ p.stride = cols ∧ ∃ j < p.count,
        p.base + j * cols = base + m * cols ∧ (∀ b ∈ p.binds.toList, b.step % 2 = 0) ∧
        p.binds.toList.map (fun b => V (b.a + j * b.step)) =
          if m = 0 then (scanIdx callee (List.range x.args.size)).map (fun it => scanArg xs st vals callee x it.1 it.2 0)
          else (List.range (callee.params[0]!.2).leaves).map (fun k => V (2 * (base + (m - 1) * cols + rets[k]!))) ++
            ((scanIdx callee (List.range x.args.size)).drop (callee.params[0]!.2).leaves).map
              (fun it => scanArg xs st vals callee x it.1 it.2 m)) := by
  unfold Work.scan at hs
  dsimp only at hs
  split at hs
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
  rename_i hn
  split at hs
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
  rename_i ha
  split at hs
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
  rename_i hc
  simp only [bind, Except.bind] at hs
  split at hs
  · simp at hs
  rename_i s hfor
  cases hs
  rw [forIn_range] at hfor
  have hn' : 0 < x.n := Nat.pos_of_ne_zero (by simpa using hn)
  have ha' : 0 < x.args.size := Nat.pos_of_ne_zero (by simpa using ha)
  have hc' : (callee.params[0]!.2).leaves ≤ rets.size := by omega
  have key := forIn_inv (Inv := ScanInv V xs st vals callee x w) (S := List.range x.args.size) ?step
    (List.range x.args.size) [] _ s (by simp) ⟨wr_refl _, [], .nil, rfl, rfl, by simp⟩ hfor
  case step =>
    intro pre i s0 r hpre hI h'
    obtain ⟨-, hpi⟩ := prefix_range_last hpre
    have hi : i < x.args.size := by
      have := hpre.length_le
      simp at this
      omega
    have hsi := hseq _ (getElem!_mem hi)
    try dsimp only at h'
    try simp only [bind, Except.bind] at h'
    split at h'
    · simp at h'
    rename_i s1 hin
    cases h'
    refine ⟨_, rfl, ?_⟩
    rw [forIn_range] at hin
    have hF0 := formsOk_wr hF hI.1
    have hP0 := paramsOk_wr hP hI.1
    have key2 := forIn_inv (Inv := ScanInvT V xs st vals callee x (s0.1, s0.2.1, s0.2.2.1, s0.2.2.2) i)
      (S := List.range (callee.params[i]!.2).leaves) ?tstep _ [] _ s1 (by simp)
      ⟨wr_refl _, [], .nil, by simp, by simp, by simp⟩ hin
    case tstep =>
      intro preT t s r hpreT hJ h''
      obtain ⟨-, hpt⟩ := prefix_range_last hpreT
      have ht : t < (callee.params[i]!.2).leaves := by
        have := hpreT.length_le
        simp at this
        omega
      have hFs := formsOk_wr hF0 hJ.1
      have hPs := paramsOk_wr hP0 hJ.1
      obtain ⟨hJw, L', hL', hf', hr', hc''⟩ := hJ
      try dsimp only at h''
      split at h''
      · rename_i hi0
        simp only [beq_iff_eq] at hi0
        try simp only [bind, Except.bind] at h''
        split at h''
        · simp at h''
        rename_i rf hrf
        split at h''
        · simp at h''
        rename_i pr hpr
        obtain ⟨w', b⟩ := pr
        cases h''
        obtain ⟨hk, rfl⟩ := getRef_ok hrf
        obtain ⟨hwb, hsb, h0, -⟩ := bindOne_spec hFs hPs hpr
        refine ⟨_, rfl, wr_trans hJw hwb, L' ++ [(b, b)], ?_, ?_, ?_, ?_⟩
        · rw [List.map_append]
          refine forall₂_append (forall₂_mono (fun it fr h => scanQ_mono h hwb) hL') (.cons ?_ .nil)
          refine ⟨hsb, fun h => absurd h (by simp [hi0]), fun W hW hR => ⟨?_, fun h => absurd h (by simp [hi0])⟩⟩
          rw [(h0 rfl).2 W hW hR, refAt_eq hsi hk]
          simp [scanArg, hi0]
        · simp [hf']
        · simp [hr']
        · simp [hc'', hi0]
      · rename_i hi0
        split at h''
        · rename_i hix
          try simp only [bind, Except.bind] at h''
          split at h''
          · simp at h''
          rename_i rs hrs
          obtain ⟨rf, sf⟩ := rs
          simp only at h''
          split at h''
          · simp at h''
          rename_i pr hpr
          obtain ⟨w', b⟩ := pr
          cases h''
          have hip : 0 < i := by simp only [beq_iff_eq] at hi0; omega
          obtain ⟨hsp1, hsp2⟩ := stepping_spec hsi hrs xs st vals
          obtain ⟨hwb, hsb, h0, h1⟩ := bindOne_spec hFs hPs hpr
          refine ⟨_, rfl, wr_trans hJw hwb, L' ++ [(b, ⟨b.a + b.step, b.step⟩)], ?_, ?_, ?_, ?_⟩
          · rw [List.map_append]
            refine forall₂_append (forall₂_mono (fun it fr h => scanQ_mono h hwb) hL') (.cons ?_ .nil)
            refine ⟨hsb, fun _ => hsb, fun W hW hR => ?_⟩
            simp only [scanArg, hip, hix, and_self, ite_true]
            by_cases hn1 : x.n ≤ 1
            · obtain ⟨hs0, hk, rfl⟩ := hsp1 hn1
              refine ⟨?_, fun _ j hj => absurd hj (by omega)⟩
              rw [(h0 hs0).2 W hW hR, refAt_eq hsi hk, Nat.zero_mul, Nat.add_zero]
            · obtain ⟨hsL, hv⟩ := hsp2 (by omega)
              have hs0 : sf ≠ 0 := by rw [hsL]; omega
              refine ⟨?_, fun _ j hj => ?_⟩
              · have := h1 hs0 0 hn'
                rw [Nat.zero_mul, Nat.add_zero] at this
                rw [this, hsL]
                have := hv 0 hn'
                simp only [Nat.zero_mul, Nat.add_zero] at this ⊢
                exact this
              · have := h1 hs0 (j + 1) hj
                rw [show b.a + b.step + j * b.step = b.a + (j + 1) * b.step by ring, this, hsL, hv (j + 1) hj]
          · simp [hf']
          · simp [hr']
          · simp [hc'', show ¬i = 0 by simpa using hi0]
        · rename_i hix
          try simp only [bind, Except.bind] at h''
          split at h''
          · simp at h''
          rename_i rf hrf
          split at h''
          · simp at h''
          rename_i pr hpr
          obtain ⟨w', b⟩ := pr
          cases h''
          obtain ⟨hk, rfl⟩ := getRef_ok hrf
          obtain ⟨hwb, hsb, h0, -⟩ := bindOne_spec hFs hPs hpr
          refine ⟨_, rfl, wr_trans hJw hwb, L' ++ [(b, b)], ?_, ?_, ?_, ?_⟩
          · rw [List.map_append]
            refine forall₂_append (forall₂_mono (fun it fr h => scanQ_mono h hwb) hL') (.cons ?_ .nil)
            have hv : ∀ W : Work, w'.rows.toList <+: W.rows.toList → RowsHold V W →
                V b.a = scanArg xs st vals callee x i t 0 := fun W hW hR => by
              rw [(h0 rfl).2 W hW hR, refAt_eq hsi hk]
              simp [scanArg, hix]
            refine ⟨hsb, fun _ => hsb, fun W hW hR => ⟨hv W hW hR, fun _ j _ => ?_⟩⟩
            rw [(h0 rfl).1, Nat.mul_zero, Nat.add_zero, hv W hW hR]
            simp [scanArg, hix]
          · simp [hf']
          · simp [hr']
          · simp [hc'', show ¬i = 0 by simpa using hi0]
    obtain ⟨hW1, L', hL', hf', hr', hc''⟩ := key2
    obtain ⟨hW0, L, hL, hf, hr, hc0⟩ := hI
    refine ⟨wr_trans hW0 hW1, L ++ L', ?_, ?_, ?_, ?_⟩
    · rw [scanIdx_append, scanIdx_single]
      exact forall₂_append (forall₂_mono (fun it fr h => scanQ_mono h hW1) hL) hL'
    · simp only at hf' ⊢
      rw [hf', hf, List.map_append]
    · simp only at hr' ⊢
      rw [hr', hr, List.map_append]
    · simp only at hc'' ⊢
      rw [hc'', hc0]
      by_cases hp : pre = []
      · subst hp
        simp at hpi
        subst hpi
        simp
      · have : i ≠ 0 := by
          intro h0
          rw [h0] at hpi
          exact hp (List.length_eq_zero_iff.mp hpi.symm)
        simp [hp, this]
  obtain ⟨hW, L, hL, hf, hr, hcar⟩ := key
  have hne : List.range x.args.size ≠ [] := by rw [Ne, List.range_eq_nil]; omega
  simp only [hne, ite_false] at hcar
  obtain ⟨T, hT, hTpos⟩ := scanIdx_range callee ha'
  have hLlen := hL.length_eq
  have hcL : (callee.params[0]!.2).leaves ≤ L.length := by rw [← hLlen, hT]; simp
  have hrsz : (callee.params[0]!.2).leaves ≤ s.2.2.1.size := by
    rw [← Array.length_toList, hr, List.length_map]; exact hcL
  have hdrop : (scanIdx callee (List.range x.args.size)).drop (callee.params[0]!.2).leaves = T := by
    rw [hT, List.drop_left' (by simp)]
  have hLd := forall₂_drop' (callee.params[0]!.2).leaves hL
  rw [hdrop] at hLd
  refine ⟨s.1.next, hW.1, hW.2.2.2.1, ?_, hW.2.2.2.2, rfl, by simp [hW.2.1], hn', ha', hc', fun hR m hm => ?_⟩
  · rw [← hW.2.2.1]
    simp only
    split
    · simp only [Array.toList_push, List.append_assoc]
      exact List.prefix_append _ _
    · simp only [Array.toList_push]
      exact List.prefix_append _ _
  have hR' := fun it fr (h : ScanQ V xs st vals callee x it fr s.1) => h.2.2 _ (List.prefix_refl _) hR
  by_cases hm0 : m = 0
  · refine ⟨⟨e, 1, s.1.next, cols, s.2.1⟩, ?_, rfl, rfl, 0, by simp, by simp [hm0], ?_, ?_⟩
    · simp only
      split <;> simp
    · intro b hb
      simp only at hb
      rw [hf] at hb
      obtain ⟨fr, hfr, rfl⟩ := List.mem_map.mp hb
      obtain ⟨it, -, hq⟩ := forall₂_mem hL fr hfr
      exact hq.1
    · rw [if_pos hm0]
      simp only
      rw [hf, List.map_map]
      exact forall₂_map (fun it fr h => by
        simp only [Function.comp, Nat.zero_mul, Nat.add_zero]
        exact (hR' it fr h).1) hL
  · have hn1 : 1 < x.n := by omega
    obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
    refine ⟨⟨e, x.n - 1, s.1.next + cols, cols, Array.map (fun k => if k < (callee.params[0]!.2).leaves then
      { a := 2 * (s.1.next + rets[s.2.2.2[k]!]!), step := 2 * cols } else s.2.2.1[k]!) (Array.range s.2.2.1.size)⟩, ?_, rfl, rfl, m', by simp; omega, by simp only; ring, ?_, ?_⟩
    · simp [hn1]
    · intro b hb
      simp only at hb
      rw [restMap_toList hrsz hcar hr, List.mem_append] at hb
      rcases hb with hb | hb
      · obtain ⟨k, -, rfl⟩ := List.mem_map.mp hb
        simp
      · obtain ⟨fr, hfr, rfl⟩ := List.mem_map.mp hb
        obtain ⟨it, hit, hq⟩ := forall₂_mem hLd fr hfr
        exact hq.2.1 (hTpos it hit)
    · rw [if_neg hm0]
      simp only
      rw [restMap_toList hrsz hcar hr, List.map_append, List.map_map, List.map_map, hdrop]
      congr 1
      · apply List.map_congr_left
        intro k _
        simp only [Function.comp, Nat.add_sub_cancel]
        congr 1
        ring
      · exact forall₂_map' hLd (fun it hit fr h => by
          simp only [Function.comp]
          exact (hR' it fr h).2 (hTpos it hit) m' (by omega))


/-- **A scan node's leaves are the last carry, then each iteration's `y`**, from its iterations' returns. -/
theorem scan_forms {V : ℕ → Bool} {w w' : Work} {vals : Array (List Bool)} {base n c cols : ℕ} {rets : Array ℕ}
    (hF : FormsOk V w vals) (ho : w'.outs = w.outs.push (.scan base cols rets c n)) (hn : 0 < n)
    (hc : c ≤ rets.size) {l : ℕ → List Bool} (hlen : ∀ m < n, (l m).length = rets.size)
    (hl : ∀ m < n, ∀ k < rets.size, V (2 * (base + m * cols + rets[k]!)) = (l m).getD k false) :
    FormsOk V w' (vals.push ((l (n - 1)).take c ++ ((List.range n).map fun m => (l m).drop c).flatten)) := by
  refine formsOk_outs (w := { w with outs := w.outs.push (.scan base cols rets c n) }) ho.symm
    (formsOk_push hF rfl ?_)
  have htake : ((l (n - 1)).take c).length = c := by
    rw [List.length_take, hlen _ (by omega)]
    omega
  intro j hj
  refine ⟨fun hjc => ?_, fun hjc => ?_⟩
  · rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by omega), List.getElem?_take_of_lt hjc,
      ← List.getD_eq_getElem?_getD, hl _ (by omega) _ (by omega)]
  · have hy : 0 < rets.size - c := Nat.pos_of_ne_zero (fun h0 => by rw [h0] at hj; omega)
    have hjn : (j - c) / (rets.size - c) < n := by rw [Nat.div_lt_iff_lt_mul hy]; omega
    have hr := Nat.mod_lt (j - c) hy
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), htake, ← List.getD_eq_getElem?_getD,
      getD_flatten_eq (L := rets.size - c) (fun l' hl' => by
        obtain ⟨m, hm, rfl⟩ := List.mem_map.mp hl'
        rw [List.length_drop, hlen m (List.mem_range.mp hm)]) (by simp; omega),
      List.getElem?_map, List.getElem?_range hjn]
    simp only [Option.map_some, Option.getD_some]
    rw [List.getD_eq_getElem?_getD, List.getElem?_drop, ← List.getD_eq_getElem?_getD, ← hl _ hjn _ (by omega)]

/-- Iteration `m`'s inputs past the carry, as the walk reads them (`walkStep`'s scan). -/
def scanWalkTail (v : Ref → Bool) (callee : Defn) (x : Node) (m : ℕ) : List Bool :=
  ((List.range x.args.size).drop 1).flatMap fun i =>
    if i ≤ x.nxs then argVals v (x.args.getD i default) ((callee.params.getD i default).2.leaves)
      (fun t => t + m * (callee.params.getD i default).2.leaves)
    else argVals v (x.args.getD i default) ((callee.params.getD i default).2.leaves) id

theorem scanTail_eq (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (ha : 0 < x.args.size) (m : ℕ) :
    ((scanIdx callee (List.range x.args.size)).drop (callee.params[0]!.2).leaves).map
      (fun it => scanArg xs st vals callee x it.1 it.2 m) = scanWalkTail (refVal xs st vals) callee x m := by
  obtain ⟨k, hk⟩ : ∃ k, x.args.size = k + 1 := ⟨x.args.size - 1, by omega⟩
  unfold scanWalkTail
  rw [hk, List.range_succ_eq_map, scanIdx_cons, List.drop_left' (by simp), List.drop_succ_cons, List.drop_zero]
  simp only [scanIdx, List.flatMap_map, List.map_flatMap, List.map_map]
  congr 1
  funext j
  by_cases hj : j + 1 ≤ x.nxs
  · simp [hj, Function.comp, scanArg, argVals, getElem!_eq_getD]
  · simp [hj, Function.comp, scanArg, argVals, getElem!_eq_getD]

theorem scanInit_eq (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node) :
    ((List.range (callee.params[0]!.2).leaves).map (0, ·)).map (fun it => scanArg xs st vals callee x it.1 it.2 0) =
      argVals (refVal xs st vals) (x.args.getD 0 default) ((callee.params.getD 0 default).2.leaves) id := by
  simp [scanArg, argVals, getElem!_eq_getD, Function.comp]

/-- Iteration 0's inputs: `init`, then the rest's. -/
theorem scanIns0_eq (xs : List Bool) (st : Array ℕ) (vals : Array (List Bool)) (callee : Defn) (x : Node)
    (ha : 0 < x.args.size) :
    (scanIdx callee (List.range x.args.size)).map (fun it => scanArg xs st vals callee x it.1 it.2 0) =
      argVals (refVal xs st vals) (x.args.getD 0 default) ((callee.params.getD 0 default).2.leaves) id ++
        scanWalkTail (refVal xs st vals) callee x 0 := by
  obtain ⟨T, hT, -⟩ := scanIdx_range callee ha
  have hd : (scanIdx callee (List.range x.args.size)).drop (callee.params[0]!.2).leaves = T := by
    rw [hT, List.drop_left' (by simp)]
  rw [← scanTail_eq xs st vals callee x ha 0, ← scanInit_eq, hd, hT, List.map_append]

end Flock.Chain.LowerDef
