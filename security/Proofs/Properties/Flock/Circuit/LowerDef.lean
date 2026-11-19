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

`rowsHold_finish` takes each row's column inside its Definition's (`hcol`), a clause of `Def.wellFormed`, which `lower`'s
`selfCheck` holds every template it writes to. Neither case rests on a refusal.
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

end Flock.Chain.LowerDef
