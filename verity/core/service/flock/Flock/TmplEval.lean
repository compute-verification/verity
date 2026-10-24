/-
What a templated unit's copies compute (PRF-133): column values that satisfy the rows `Tmpl.rows` unrolls take, at every
copy placed under one of the root's calls past the input rows, the value of that copy's Definition on its bound inputs
(`rows_eval`). So a session's flat circuit is, to Lean, the composition of its Definitions, and a fact about one
Definition (its value on its inputs) holds at every copy of it, wherever the template places it. Nothing on a verifier
path reads this file.

`eval` is a Definition's columns on its inputs and the constant's value, in column order: a column one of its rows
writes is that row's `(⊕A) ∧ (⊕B)` over its references, a column a copy writes is the callee's on the copy's bound
values, and a column nothing writes is 0, as an empty row forces. `Holds` is the same equations as a relation, every
row, copy and gap at once. The proof has two halves:

- `place_holds`, an induction over the copies `place` writes: a copy whose columns were unwritten before it, and whose
  rows hold after it, satisfies `Holds`. A row's column keeps its rows once written (`write` refuses a written column),
  a copy writes only inside its columns (`place_frame`), and a Definition's rows and copies are apart
  (`Tmpl.apart`), so each callee copy finds its columns unwritten and keeps its rows to the end.
- `holds_eval`: under `Tmpl.ordered`'s flags, a row reads only its inputs, the constant and earlier columns, and a copy
  is bound only to those before it, so `Holds` pins every column to `eval`.

The root is not a Definition copy like the others: its input rows are the unit's inputs and its last column the
constant, so the statement is about the copies under its calls.
-/
import Flock.TmplApart

namespace Flock.Tmpl

/-- The XOR of the values at a row's columns (`Audit.xorSum`). -/
def xorAt (l : List Nat) (z : Nat → Bool) : Bool := (l.map z).foldr Bool.xor false

/-- A row's value: the AND of its `A` and `B` sums, its references read through `g`. -/
def rowVal (g : Nat → Bool) (r : Nat × Array Nat × Array Nat) : Bool := xorAt r.2.1.toList g && xorAt r.2.2.toList g

/-- `Def.ref` on values: Definition `df`'s reference `x`, its own columns' values `v`, its inputs' `inp`, the constant's
`kv`. -/
def Def.refV (df : Def) (v : Nat → Bool) (inp : List Bool) (kv : Bool) (x : Nat) : Bool :=
  if x < df.cols then v x else if x < df.cols + df.ins then inp.getD (x - df.cols) kv else kv

/-- `Call.bound` on values: copy `j`'s input values, through the caller's references' values `g`. -/
def Call.boundV (c : Call) (g : Nat → Bool) (j : Nat) : List Bool :=
  c.segs.toList.flatMap fun s => (List.range s.reps).flatMap fun r =>
    (List.range s.len).map fun t => g (s.at_ + r * s.rstride + j * s.step + t)

/-- The copy of call `c` that holds column `y` of its caller, if one does. -/
def Call.copyOf (defs : Array Def) (c : Call) (y : Nat) : Option Nat :=
  (List.range c.count).find? fun j =>
    decide (c.base + j * c.stride ≤ y) && decide (y < c.base + j * c.stride + (defs.getD c.def_ default).cols)

/-- The first call of `df`, and its copy, that holds column `y`, if one does. -/
def Def.copyAt (defs : Array Def) (df : Def) (y : Nat) : Option (Call × Nat) :=
  df.calls.toList.findSome? fun c => (c.copyOf defs y).map (c, ·)

/-- Column `y` of a copy of `df`, its earlier columns' values `v`: its row's value, else its copy's (`sub`: each callee's
columns on its inputs), else 0. -/
def Def.colAt (defs : Array Def) (kv : Bool) (sub : Nat → List Bool → Nat → Bool) (df : Def) (inp : List Bool)
    (v : Nat → Bool) (y : Nat) : Bool :=
  match df.rows.toList.find? (·.1 == y) with
  | some r => rowVal (df.refV v inp kv) r
  | none =>
    match df.copyAt defs y with
    | some (c, j) => sub c.def_ (c.boundV (df.refV v inp kv) j) (y - (c.base + j * c.stride))
    | none => false

/-- A copy of `df`'s first `n` columns, in order, each from the ones before it; 0 past them. -/
def Def.upTo (defs : Array Def) (kv : Bool) (sub : Nat → List Bool → Nat → Bool) (df : Def) (inp : List Bool) :
    Nat → Nat → Bool
  | 0 => fun _ => false
  | n + 1 => fun x => if x = n then df.colAt defs kv sub inp (df.upTo defs kv sub inp n) n else df.upTo defs kv sub inp n x

/-- **Definition `d`'s columns on its inputs `inp`**, the constant `kv`, calls nested at most `fuel` deep. -/
def eval (defs : Array Def) (kv : Bool) : Nat → Nat → List Bool → Nat → Bool
  | 0, _, _, _ => false
  | fuel + 1, d, inp, y => (defs.getD d default).upTo defs kv (eval defs kv fuel) inp (y + 1) y

/-- **What Definition `d` of `t` computes**: its column `y` on inputs `inp`, the constant `kv`. -/
def Tmpl.evalDef (t : Tmpl) (d : Nat) (inp : List Bool) (kv : Bool) (y : Nat) : Bool :=
  eval t.defs kv t.defs.size d inp y

/-- Column values `v` of a copy of Definition `d`, its inputs `inp`, the constant `kv`, satisfy `d`'s equations: each row,
each copy of each call (on its bound values), and 0 at each column no row or copy writes. -/
def Holds (defs : Array Def) (kv : Bool) : Nat → Nat → (Nat → Bool) → List Bool → Prop
  | 0, _, _, _ => False
  | fuel + 1, d, v, inp =>
    (∀ r ∈ (defs.getD d default).rows.toList, v r.1 = rowVal ((defs.getD d default).refV v inp kv) r) ∧
    (∀ c ∈ (defs.getD d default).calls.toList, ∀ j < c.count,
      Holds defs kv fuel c.def_ (fun y => v (c.base + j * c.stride + y))
        (c.boundV ((defs.getD d default).refV v inp kv) j)) ∧
    ∀ y < (defs.getD d default).cols, (∀ r ∈ (defs.getD d default).rows.toList, r.1 ≠ y) →
      (defs.getD d default).copyAt defs y = none → v y = false

/-- Copy `(e, o', inp')` is copy `(d, o, inp)` or one `place` writes inside it, at fuel `f'` below `f`. -/
inductive Within (defs : Array Def) (k : Nat) : Nat → Nat → Nat → List Nat → Nat → Nat → Nat → List Nat → Prop
  | refl {f d o inp} : Within defs k f d o inp f d o inp
  | call {f d o inp f' e o' inp'} (c : Call) (j : Nat) (hc : c ∈ (defs.getD d default).calls.toList)
      (hj : j < c.count)
      (h : Within defs k f c.def_ (o + c.base + j * c.stride) (c.bound ((defs.getD d default).ref o inp k) j) f' e o' inp') :
      Within defs k (f + 1) d o inp f' e o' inp'

end Flock.Tmpl

namespace Flock.TmplEval

open Flock.Tmpl Flock.TmplOrder Flock.TmplRows Flock.TmplApart

/-! ## Sums -/

theorem xorAt_append (l m : List Nat) (z : Nat → Bool) : xorAt (l ++ m) z = (xorAt l z ^^ xorAt m z) := by
  induction l with
  | nil => simp [xorAt]
  | cons a l ih => simp only [xorAt, List.cons_append, List.map_cons, List.foldr_cons] at ih ⊢; rw [ih, Bool.xor_assoc]

theorem xorAt_perm {l m : List Nat} (h : l.Perm m) (z : Nat → Bool) : xorAt l z = xorAt m z := by
  induction h with
  | nil => rfl
  | cons a _ ih => simp only [xorAt, List.map_cons, List.foldr_cons] at ih ⊢; rw [ih]
  | swap a b l =>
    simp only [xorAt, List.map_cons, List.foldr_cons]
    rw [← Bool.xor_assoc, ← Bool.xor_assoc, Bool.xor_comm (z b)]
  | trans _ _ ih1 ih2 => exact ih1.trans ih2

theorem xorAt_dropPairs : ∀ (l : List Nat) (z : Nat → Bool), xorAt (dropPairs l) z = xorAt l z
  | [], _ => rfl
  | [_], _ => rfl
  | a :: b :: rest, z => by
    simp only [dropPairs]
    split
    · rename_i h
      rw [beq_iff_eq] at h
      subst h
      rw [xorAt_dropPairs rest z]
      simp only [xorAt, List.map_cons, List.foldr_cons]
      rw [← Bool.xor_assoc, Bool.xor_self, Bool.false_xor]
    · have := xorAt_dropPairs (b :: rest) z
      simp only [xorAt, List.map_cons, List.foldr_cons] at this ⊢
      rw [this]

/-- **A canonical row sums as the row does**: `canon` sorts and drops pairs, neither of which moves a sum over GF(2). -/
theorem xorAt_canon (xs : Array Nat) (z : Nat → Bool) : xorAt (Flock.Tmpl.canon xs).toList z = xorAt xs.toList z := by
  show xorAt (dropPairs xs.toList.mergeSort) z = _
  rw [xorAt_dropPairs]
  exact xorAt_perm (List.mergeSort_perm _ _) z

theorem xorAt_map (l : List Nat) (g : Nat → Nat) (z : Nat → Bool) : xorAt (l.map g) z = xorAt l (fun x => z (g x)) := by
  simp [xorAt, List.map_map, Function.comp_def]

theorem xorAt_congr {l : List Nat} {z z' : Nat → Bool} (h : ∀ x ∈ l, z x = z' x) : xorAt l z = xorAt l z' := by
  unfold xorAt
  rw [List.map_congr_left h]

/-! ## Values through references -/

theorem ref_val (df : Def) (o : Nat) (inp : List Nat) (k : Nat) (vals : Nat → Bool) (x : Nat) :
    vals (df.ref o inp k x) = df.refV (fun y => vals (o + y)) (inp.map vals) (vals k) x := by
  unfold Def.ref Def.refV
  split
  · rfl
  · split
    · rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
      cases inp[x - df.cols]? <;> rfl
    · rfl

theorem bound_val (c : Call) (g : Nat → Nat) (j : Nat) (vals : Nat → Bool) :
    (c.bound g j).map vals = c.boundV (fun x => vals (g x)) j := by
  simp [Call.bound, Call.boundV, List.map_flatMap, List.map_map, Function.comp_def]

theorem flatMap_congr {α β : Type} {f g : α → List β} : ∀ {l : List α}, (∀ a ∈ l, f a = g a) →
    l.flatMap f = l.flatMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons, h a (List.mem_cons_self ..),
      flatMap_congr fun b hb => h b (List.mem_cons_of_mem a hb)]

theorem boundV_congr {c : Call} {g g' : Nat → Bool} {j : Nat}
    (h : ∀ s ∈ c.segs.toList, ∀ r < s.reps, ∀ t < s.len,
      g (s.at_ + r * s.rstride + j * s.step + t) = g' (s.at_ + r * s.rstride + j * s.step + t)) :
    c.boundV g j = c.boundV g' j := by
  unfold Call.boundV
  refine flatMap_congr fun s hs => flatMap_congr fun r hr => List.map_congr_left fun t ht => ?_
  exact h s hs r (List.mem_range.1 hr) t (List.mem_range.1 ht)

/-! ## Finding a column's writer -/

theorem copyAt_some {defs : Array Def} {df : Def} {y : Nat} {c : Call} {j : Nat} (h : df.copyAt defs y = some (c, j)) :
    c ∈ df.calls.toList ∧ j < c.count ∧ c.base + j * c.stride ≤ y ∧
      y < c.base + j * c.stride + (defs.getD c.def_ default).cols := by
  unfold Def.copyAt at h
  obtain ⟨c', hc', hm⟩ := List.exists_of_findSome?_eq_some h
  cases hj : c'.copyOf defs y with
  | none => rw [hj] at hm; cases hm
  | some j' =>
    rw [hj] at hm
    cases hm
    unfold Call.copyOf at hj
    have h1 := List.find?_some hj
    have h2 := List.mem_of_find?_eq_some hj
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h1
    exact ⟨hc', List.mem_range.1 h2, h1.1, h1.2⟩

theorem copyAt_none {defs : Array Def} {df : Def} {y : Nat} (h : df.copyAt defs y = none) :
    ∀ c ∈ df.calls.toList, ∀ j < c.count,
      ¬ (c.base + j * c.stride ≤ y ∧ y < c.base + j * c.stride + (defs.getD c.def_ default).cols) := by
  intro c hc j hj hin
  unfold Def.copyAt at h
  rw [List.findSome?_eq_none_iff] at h
  have := h c hc
  rw [Option.map_eq_none_iff] at this
  unfold Call.copyOf at this
  rw [List.find?_eq_none] at this
  exact this j (List.mem_range.2 hj) (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hin)

/-! ## `Holds` pins the columns to `eval` -/

theorem upTo_lt {defs : Array Def} {kv : Bool} {sub : Nat → List Bool → Nat → Bool} {df : Def} {inp : List Bool} :
    ∀ {n x : Nat}, x < n → df.upTo defs kv sub inp n x = df.upTo defs kv sub inp (x + 1) x
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, x, h => by
    by_cases e : x = n
    · subst e; rfl
    · conv => lhs; unfold Def.upTo
      rw [ite_eq_right e]
      exact upTo_lt (by omega)

/-- **An ordered copy's columns are its Definition's `eval`**: `Holds` at a Definition the flags `F` mark ordered pins each
of its columns, column by column, since each row and each copy's binding reads only earlier ones. -/
theorem holds_eval {defs : Array Def} {kv : Bool} {F : Array Bool}
    (hF : ∀ d, F.getD d false = true → (defs.getD d default).ordered F = true) (fuel : Nat) :
    ∀ {d v inp}, F.getD d false = true → Holds defs kv fuel d v inp →
      ∀ y < (defs.getD d default).cols, v y = eval defs kv fuel d inp y := by
  induction fuel with
  | zero => intro d v inp _ h; exact h.elim
  | succ fuel ih =>
    intro d v inp hd hH
    have ho := hF d hd
    generalize hdf : defs.getD d default = df at ho hH
    simp only [Holds, hdf] at hH
    obtain ⟨hrows, hcalls, hgaps⟩ := hH
    unfold Def.ordered at ho
    simp only [Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true, decide_eq_true_eq, List.mem_append] at ho
    obtain ⟨hord, hcord⟩ := ho
    let sub := eval defs kv fuel
    have key : ∀ n, n ≤ df.cols → ∀ x < n, df.upTo defs kv sub inp n x = v x := by
      intro n
      induction n with
      | zero => intro _ x hx; omega
      | succ n ihn =>
        intro hn x hx
        unfold Def.upTo
        by_cases e : x = n
        · subst e
          rw [ite_eq_left rfl]
          have below : ∀ w, df.cols ≤ w ∨ w < x →
              df.refV (df.upTo defs kv sub inp x) inp kv w = df.refV v inp kv w := by
            intro w hw
            unfold Def.refV
            by_cases h1 : w < df.cols
            · rw [ite_eq_left h1, ite_eq_left h1]
              exact ihn (by omega) w (by omega)
            · rw [ite_eq_right h1, ite_eq_right h1]
          unfold Def.colAt
          cases hfind : df.rows.toList.find? (·.1 == x) with
          | some r =>
            have hr := List.mem_of_find?_eq_some hfind
            have hrx : r.1 = x := by simpa using List.find?_some hfind
            simp only
            rw [show v x = v r.1 by rw [hrx], hrows r hr]
            unfold rowVal
            rw [xorAt_congr fun w hw => below w (by rw [← hrx]; exact hord r hr w (Or.inl hw)),
              xorAt_congr fun w hw => below w (by rw [← hrx]; exact hord r hr w (Or.inr hw))]
          | none =>
            simp only
            cases hcopy : df.copyAt defs x with
            | some cj =>
              obtain ⟨c, j⟩ := cj
              simp only
              obtain ⟨hc, hj, h1, h2⟩ := copyAt_some hcopy
              obtain ⟨hFc, hsegs⟩ := hcord c hc
              have hb : c.boundV (df.refV (df.upTo defs kv sub inp x) inp kv) j = c.boundV (df.refV v inp kv) j := by
                refine boundV_congr fun s hs r hr t ht => below _ ?_
                rcases seg_ordered (hsegs s hs) hj hr ht with h | ⟨-, h, -⟩
                · exact Or.inl h
                · exact Or.inr (by omega)
              rw [hb]
              have := ih hFc (hcalls c hc j hj) (x - (c.base + j * c.stride)) (by omega)
              rw [show c.base + j * c.stride + (x - (c.base + j * c.stride)) = x by omega] at this
              exact this.symm
            | none =>
              simp only
              refine (hgaps x (by omega) (fun r hr e => ?_) hcopy).symm
              rw [List.find?_eq_none] at hfind
              exact hfind r hr (by simp [e])
        · rw [ite_eq_right e]
          exact ihn (by omega) x (by omega)
    intro y hy
    show v y = (defs.getD d default).upTo defs kv sub inp (y + 1) y
    rw [hdf]
    exact (key (y + 1) hy y (Nat.lt_succ_self y)).symm

end Flock.TmplEval
