/-
What a templated unit's copies compute: column values that satisfy the rows `Tmpl.rows` unrolls take, at every
copy placed under one of the root's calls past the input rows, the value of that copy's Definition on its bound inputs
(`rows_eval`). So a session's flat circuit is, to Lean, the composition of its Definitions, and a fact about one
Definition (its value on its inputs) holds at every copy of it, wherever the template places it. Nothing on a verifier
path reads this file.

`eval` is a Definition's columns on its inputs and the constant's value, in column order: a column one of its rows
writes is that row's `(⊕A) ∧ (⊕B)` over its references, a column a copy writes is the callee's on the copy's bound
values, and a column nothing writes is 0, as an empty row forces. `Holds` is the same equations as a relation, every
row, copy and gap at once: `Flock.TmplEval.SatAt` (`rows_sat`) is its row and call clauses, and a value needs the gap
clause too, since a row may read a column nothing writes. The proof has two halves:

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

namespace Flock.TmplValue

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

/-! ## `eval` at any fuel past the depth -/

theorem colAt_congr {defs : Array Def} {kv : Bool} {sub sub' : Nat → List Bool → Nat → Bool} {df : Def}
    {inp : List Bool} (h : ∀ c ∈ df.calls.toList, sub c.def_ = sub' c.def_) (v : Nat → Bool) (y : Nat) :
    df.colAt defs kv sub inp v y = df.colAt defs kv sub' inp v y := by
  unfold Def.colAt
  cases df.rows.toList.find? (·.1 == y) with
  | some r => rfl
  | none =>
    simp only
    cases hcopy : df.copyAt defs y with
    | none => rfl
    | some cj => simp only; rw [h cj.1 (copyAt_some hcopy).1]

theorem upTo_congr {defs : Array Def} {kv : Bool} {sub sub' : Nat → List Bool → Nat → Bool} {df : Def}
    {inp : List Bool} (h : ∀ c ∈ df.calls.toList, sub c.def_ = sub' c.def_) :
    ∀ n, df.upTo defs kv sub inp n = df.upTo defs kv sub' inp n
  | 0 => rfl
  | n + 1 => by
    funext x
    simp only [Def.upTo]
    rw [upTo_congr h n, colAt_congr h]

/-- **`eval` is the same at every fuel past a Definition's index**: a well-formed Definition calls only earlier ones. -/
theorem eval_fuel {defs : Array Def} {kv : Bool}
    (hwf : ∀ d, d < defs.size → (defs.getD d default).wellFormed defs d = true) :
    ∀ f f' d, d < f → d < f' → d < defs.size → eval defs kv f d = eval defs kv f' d
  | 0, _, _, h, _, _ => absurd h (Nat.not_lt_zero _)
  | _ + 1, 0, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | f + 1, f' + 1, d, h1, h2, hd => by
    funext inp y
    simp only [eval]
    have hw := hwf d hd
    unfold Def.wellFormed at hw
    simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hw
    rw [upTo_congr fun c hc => eval_fuel hwf f f' c.def_ (by have := (hw.2 c hc).1.1; omega)
      (by have := (hw.2 c hc).1.1; omega) (by have := (hw.2 c hc).1.1; omega)]

/-! ## What `write` and `place` keep -/

/-- Column `x`'s rows and flag are the same in both. -/
abbrev Same3 (x : Nat) (a b : Acc) : Prop :=
  a.ra.getD x #[] = b.ra.getD x #[] ∧ a.rb.getD x #[] = b.rb.getD x #[] ∧ a.set.getD x true = b.set.getD x true

theorem same3_trans {x : Nat} {a b c : Acc} (h1 : Same3 x a b) (h2 : Same3 x b c) : Same3 x a c :=
  ⟨h1.1.trans h2.1, h1.2.1.trans h2.2.1, h1.2.2.trans h2.2.2⟩

theorem write_same3 {acc acc' : Acc} {c x : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc') (e : x ≠ c) :
    Same3 x acc' acc := by
  obtain ⟨-, rfl⟩ := write_eq h
  exact ⟨getD_set!_ne _ _ _ e, getD_set!_ne _ _ _ e, getD_set!_ne _ _ _ e⟩

theorem free_of_same3 {x : Nat} {a b : Acc} (s : Same3 x a b) (h : Free x b) : Free x a := by
  simp only [Free] at h ⊢; rw [s.2.2]; exact h

/-- Every unwritten column has empty rows. -/
def Clean (acc : Acc) : Prop := ∀ x, Free x acc → acc.ra.getD x #[] = #[] ∧ acc.rb.getD x #[] = #[]

theorem clean_empty (n : Nat) : Clean (Acc.empty n) := fun x _ => by
  simp only [Acc.empty, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  constructor <;> split <;> rfl

theorem write_clean {acc acc' : Acc} {c : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc') (hc : Clean acc) :
    Clean acc' := by
  intro x hx
  by_cases e : x = c
  · subst e
    obtain ⟨h0, rfl⟩ := write_eq h
    simp only [Free] at hx
    rw [getD_set!_self _ _ _ (lt_of_getD_false h0)] at hx
    cases hx
  · have s := write_same3 h e
    have := hc x (free_of_same3 ⟨s.1.symm, s.2.1.symm, s.2.2.symm⟩ hx)
    exact ⟨s.1.trans this.1, s.2.1.trans this.2⟩

/-- Column `x` is written, with rows `a` and `b`. -/
abbrev Kept (x : Nat) (a b : Array Nat) (acc : Acc) : Prop :=
  acc.set.getD x true = true ∧ acc.ra.getD x #[] = a ∧ acc.rb.getD x #[] = b

theorem write_kept {acc acc' : Acc} {c x : Nat} {a b a' b' : Array Nat} (h : write acc c a' b' = .ok acc')
    (hk : Kept x a b acc) : Kept x a b acc' := by
  obtain ⟨h0, -⟩ := write_eq h
  have e : x ≠ c := fun e => by subst e; rw [hk.1] at h0; cases h0
  have s := write_same3 h e
  exact ⟨s.2.2.trans hk.1, s.1.trans hk.2.1, s.2.1.trans hk.2.2⟩

theorem write_made {acc acc' : Acc} {c : Nat} {a b : Array Nat} (hs : Sized acc) (h : write acc c a b = .ok acc') :
    Kept c a b acc' := by
  obtain ⟨hc, rfl⟩ := write_eq h
  have hlt := lt_of_getD_false hc
  exact ⟨getD_set!_self _ _ _ hlt, getD_set!_self _ _ _ (by rw [hs.1]; exact hlt),
    getD_set!_self _ _ _ (by rw [hs.2]; exact hlt)⟩

/-- **What every write keeps, every copy keeps.** -/
theorem place_inv {defs : Array Def} {k : Nat} (P : Acc → Prop)
    (hw : ∀ {acc acc' : Acc} {c : Nat} {a b : Array Nat}, write acc c a b = .ok acc' → P acc → P acc') (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → P acc → P acc' := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h hs
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have s1 := foldlM_inv P (fun _ _ u u' hu hw' => hw hw' hu) hs h1
      exact foldlM_inv P (fun c _ u u' hu hw' =>
        foldlM_inv P (fun _ _ w w' hw0 hp => ih hp hw0) hu hw') s1 h2

/-- **A copy leaves every column outside its own as it was**, rows and flag. -/
theorem place_frame {defs : Array Def} {k : Nat}
    (hfits : ∀ (d : Nat) (df : Def), defs[d]? = some df → df.fits defs = true) (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' →
      ∀ x, (x < o ∨ o + (defs.getD d default).cols ≤ x) → Same3 x acc' acc := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h x hx
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨-, hdd⟩ := getD_of_some hdf
      rw [hdd] at hx
      have hf := hfits d df hdf
      unfold Def.fits at hf
      simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hf
      obtain ⟨hrows, hcalls⟩ := hf
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have s1 : Same3 x acc1 acc := foldlM_inv (fun a => Same3 x a acc) (fun r hr u u' hu hw =>
          same3_trans (write_same3 hw (by have := hrows r hr; omega)) hu) ⟨rfl, rfl, rfl⟩ h1
      have s2 : Same3 x acc' acc1 := foldlM_inv (fun a => Same3 x a acc1) (fun c hc u u' hu hw =>
          same3_trans (foldlM_inv (fun a => Same3 x a u) (fun j hj w w' hw0 hp => by
              have hj' := List.mem_range.1 hj
              have hm : j * c.stride ≤ (c.count - 1) * c.stride := Nat.mul_le_mul (by omega) (Nat.le_refl _)
              have hc' := hcalls c hc
              exact same3_trans (ih hp x (by omega)) hw0) ⟨rfl, rfl, rfl⟩ hw) hu) ⟨rfl, rfl, rfl⟩ h2
      exact same3_trans s2 s1

/-- A fold, split at one of its elements: the steps before it, its own, and the steps after it. -/
theorem foldlM_split {α σ : Type} {f : σ → α → Except String σ} :
    ∀ {l : List α} {s s' : σ}, l.foldlM f s = .ok s' → ∀ {a : α}, a ∈ l →
      ∃ l1 l2 u u', l = l1 ++ a :: l2 ∧ l1.foldlM f s = .ok u ∧ f u a = .ok u' ∧ l2.foldlM f u' = .ok s'
  | [], _, _, _, _, ha => by cases ha
  | b :: l, s, s', h, a, ha => by
    rw [List.foldlM_cons] at h
    obtain ⟨s1, h1, h2⟩ := bind_ok h
    by_cases e : a = b
    · subst e; exact ⟨[], l, s, s1, rfl, rfl, h1, h2⟩
    · obtain ⟨l1, l2, u, u', el, e1, e2, e3⟩ := foldlM_split h2 ((List.mem_cons.1 ha).resolve_left e)
      exact ⟨b :: l1, l2, u, u', by rw [el]; rfl, by rw [List.foldlM_cons, h1]; exact e1, e2, e3⟩

/-- Copy `j`'s columns are inside its call's range. -/
theorem copy_in_span {c : Call} {W j y : Nat} (hj : j < c.count) (h1 : c.base + j * c.stride ≤ y)
    (h2 : y < c.base + j * c.stride + W) : c.base ≤ y ∧ y < c.base + (c.count - 1) * c.stride + W := by
  have hm : j * c.stride ≤ (c.count - 1) * c.stride := Nat.mul_le_mul (by omega) (Nat.le_refl _)
  omega

/-! ## Copies satisfy their Definitions -/

/-- The rows at column `x` hold at `vals`. -/
abbrev RowHolds (vals : Nat → Bool) (acc : Acc) (x : Nat) : Prop :=
  vals x = (xorAt (acc.ra.getD x #[]).toList vals && xorAt (acc.rb.getD x #[]).toList vals)

/-- **A Definition's calls, placed, satisfy their callees**: given that a copy whose columns were unwritten, and whose rows
hold after it, satisfies its Definition (`ih`), every copy of every call does, when its columns are unwritten before the
calls and its rows hold after them. -/
theorem calls_holds {defs : Array Def} {k : Nat} {vals : Nat → Bool} {fuel : Nat} {Q : Call → Prop}
    (hfits : ∀ (d : Nat) (df : Def), defs[d]? = some df → df.fits defs = true)
    (ih : ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → Sized acc → Clean acc →
      (∀ x, o ≤ x → x < o + (defs.getD d default).cols → Free x acc) →
      (∀ x, o ≤ x → x < o + (defs.getD d default).cols → RowHolds vals acc' x) →
      Holds defs (vals k) fuel d (fun y => vals (o + y)) (inp.map vals))
    {df : Def} {o : Nat} {inp : List Nat} {acc1 acc' : Acc} (hap : df.apart defs = true)
    (h2 : df.calls.toList.foldlM (fun acc c => (List.range c.count).foldlM (fun acc j =>
      place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc) acc1 = .ok acc')
    (hsized : Sized acc1) (hclean : Clean acc1)
    (hfree : ∀ c ∈ df.calls.toList, ∀ j < c.count, ∀ y, c.base + j * c.stride ≤ y →
      y < c.base + j * c.stride + (defs.getD c.def_ default).cols → Free (o + y) acc1)
    (hsat : ∀ c ∈ df.calls.toList, Q c → ∀ j < c.count, ∀ y, c.base + j * c.stride ≤ y →
      y < c.base + j * c.stride + (defs.getD c.def_ default).cols → RowHolds vals acc' (o + y)) :
    ∀ c ∈ df.calls.toList, Q c → ∀ j < c.count,
      Holds defs (vals k) fuel c.def_ (fun y => vals (o + c.base + j * c.stride + y))
        ((c.bound (df.ref o inp k) j).map vals) := by
  intro c hc hQ j hj
  have hap' := hap
  unfold Def.apart at hap'
  simp only [Bool.and_eq_true, List.all_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hap'
  have hcopies := hap'.2
  have hp := spans_pairwise hap
  unfold Def.spans at hp
  rw [List.pairwise_filter, List.pairwise_append, List.pairwise_map, List.pairwise_map] at hp
  obtain ⟨-, hpc, -⟩ := hp
  let fc := fun (acc : Acc) (c : Call) => (List.range c.count).foldlM (fun acc j =>
    place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc
  obtain ⟨l1, l2, u, u', el, e1, e2, e3⟩ := foldlM_split (f := fc) h2 hc
  obtain ⟨m1, m2, w, w', em, f1, f2, f3⟩ := foldlM_split e2 (List.mem_range.2 hj)
  -- each copy of a call leaves the columns outside it alone
  have callFrame : ∀ (c' : Call) (x : Nat), (∀ j' < c'.count, x < o + c'.base + j' * c'.stride ∨
      o + c'.base + j' * c'.stride + (defs.getD c'.def_ default).cols ≤ x) → ∀ s s', fc s c' = .ok s' →
      Same3 x s' s := fun c' x hx s s' hf =>
    foldlM_inv (fun a => Same3 x a s) (fun j' hj' w w' hw0 hp =>
      same3_trans (place_frame hfits _ hp x (hx j' (List.mem_range.1 hj'))) hw0) ⟨rfl, rfl, rfl⟩ hf
  -- a call other than `c` misses copy `j`'s columns
  have otherCall : ∀ c' : Call, (decide ((c'.base, c'.base + (c'.count - 1) * c'.stride + (defs.getD c'.def_ default).cols).1 <
        (c'.base, c'.base + (c'.count - 1) * c'.stride + (defs.getD c'.def_ default).cols).2) = true →
      decide ((c.base, c.base + (c.count - 1) * c.stride + (defs.getD c.def_ default).cols).1 <
        (c.base, c.base + (c.count - 1) * c.stride + (defs.getD c.def_ default).cols).2) = true →
      Apart (c'.base, c'.base + (c'.count - 1) * c'.stride + (defs.getD c'.def_ default).cols)
        (c.base, c.base + (c.count - 1) * c.stride + (defs.getD c.def_ default).cols)) →
      ∀ y, c.base + j * c.stride ≤ y → y < c.base + j * c.stride + (defs.getD c.def_ default).cols →
      ∀ j' < c'.count, o + y < o + c'.base + j' * c'.stride ∨
        o + c'.base + j' * c'.stride + (defs.getD c'.def_ default).cols ≤ o + y := by
    intro c' hR y hy1 hy2 j' hj'
    obtain ⟨s1, s2⟩ := copy_in_span hj hy1 hy2
    by_cases hin : c'.base + j' * c'.stride ≤ y ∧ y < c'.base + j' * c'.stride + (defs.getD c'.def_ default).cols
    · obtain ⟨t1, t2⟩ := copy_in_span hj' hin.1 hin.2
      have := hR (by simp only [decide_eq_true_eq]; omega) (by simp only [decide_eq_true_eq]; omega)
      simp only [Apart] at this
      omega
    · omega
  rw [el] at hpc
  obtain ⟨-, hpc2, hcross⟩ := List.pairwise_append.1 hpc
  have hafter := (List.pairwise_cons.1 hpc2).1
  have hm := List.pairwise_lt_range (n := c.count)
  rw [em] at hm
  obtain ⟨-, hm2, hmcross⟩ := List.pairwise_append.1 hm
  have hmafter := (List.pairwise_cons.1 hm2).1
  -- another copy of `c` misses copy `j`'s columns
  have otherCopy : ∀ j', j' < c.count → j' ≠ j → ∀ y, c.base + j * c.stride ≤ y →
      y < c.base + j * c.stride + (defs.getD c.def_ default).cols →
      o + y < o + c.base + j' * c.stride ∨ o + c.base + j' * c.stride + (defs.getD c.def_ default).cols ≤ o + y := by
    intro j' hj' hne y hy1 hy2
    rcases hcopies c hc with h1 | h1
    · omega
    · rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
      · have : j' * c.stride + c.stride ≤ j * c.stride := by
          rw [← Nat.succ_mul]; exact Nat.mul_le_mul (by omega) (Nat.le_refl _)
        omega
      · have : j * c.stride + c.stride ≤ j' * c.stride := by
          rw [← Nat.succ_mul]; exact Nat.mul_le_mul (by omega) (Nat.le_refl _)
        omega
  have hmem1 : ∀ c' ∈ l1, c' ∈ df.calls.toList := fun c' h => by rw [el]; exact List.mem_append_left _ h
  have hmem2 : ∀ c' ∈ l2, c' ∈ df.calls.toList := fun c' h => by
    rw [el]; exact List.mem_append_right _ (List.mem_cons_of_mem _ h)
  -- before copy `j`: what the earlier calls and copies keep
  have before : ∀ y, c.base + j * c.stride ≤ y → y < c.base + j * c.stride + (defs.getD c.def_ default).cols →
      Same3 (o + y) w acc1 := by
    intro y hy1 hy2
    have s1 : Same3 (o + y) u acc1 := foldlM_inv (fun a => Same3 (o + y) a acc1) (fun c' hc' s s' hs hf =>
        same3_trans (callFrame c' (o + y) (otherCall c' (hcross c' hc' c (List.mem_cons_self ..)) y hy1 hy2) s s' hf) hs)
      ⟨rfl, rfl, rfl⟩ e1
    have s2 : Same3 (o + y) w u := foldlM_inv (fun a => Same3 (o + y) a u) (fun j' hj' s s' hs hp =>
        same3_trans (place_frame hfits _ hp (o + y) (otherCopy j' (by
          have := hmcross j' hj' j (List.mem_cons_self ..); omega) (by
          have := hmcross j' hj' j (List.mem_cons_self ..); omega) y hy1 hy2)) hs) ⟨rfl, rfl, rfl⟩ f1
    exact same3_trans s2 s1
  -- after copy `j`: what the later copies and calls keep
  have after : ∀ y, c.base + j * c.stride ≤ y → y < c.base + j * c.stride + (defs.getD c.def_ default).cols →
      Same3 (o + y) acc' w' := by
    intro y hy1 hy2
    have s1 : Same3 (o + y) u' w' := foldlM_inv (fun a => Same3 (o + y) a w') (fun j' hj' s s' hs hp =>
        same3_trans (place_frame hfits _ hp (o + y) (otherCopy j' (by
          have := hmafter j' hj'
          have hr : j' ∈ List.range c.count := by rw [em]; exact List.mem_append_right _ (List.mem_cons_of_mem _ hj')
          exact List.mem_range.1 hr) (by have := hmafter j' hj'; omega) y hy1 hy2)) hs) ⟨rfl, rfl, rfl⟩ f3
    have s2 : Same3 (o + y) acc' u' := foldlM_inv (fun a => Same3 (o + y) a u') (fun c' hc' s s' hs hf =>
        same3_trans (callFrame c' (o + y) (otherCall c' (fun h1 h2 => apart_symm (hafter c' hc' h2 h1)) y hy1 hy2)
          s s' hf) hs) ⟨rfl, rfl, rfl⟩ e3
    exact same3_trans s2 s1
  have hsw : Sized w := foldlM_inv Sized (fun j' _ s s' hs hp => place_sized _ hp hs)
    (foldlM_inv Sized (fun c' _ s s' hs hf => foldlM_inv Sized (fun j' _ s s' hs hp => place_sized _ hp hs) hs hf)
      hsized e1) f1
  have hcw : Clean w := foldlM_inv Clean (fun j' _ s s' hs hp => place_inv Clean write_clean _ hp hs)
    (foldlM_inv Clean (fun c' _ s s' hs hf => foldlM_inv Clean (fun j' _ s s' hs hp =>
      place_inv Clean write_clean _ hp hs) hs hf) hclean e1) f1
  refine ih f2 hsw hcw (fun x hx1 hx2 => ?_) (fun x hx1 hx2 => ?_)
  · obtain ⟨y, rfl⟩ : ∃ y, x = o + y := ⟨x - o, by omega⟩
    exact free_of_same3 (before y (by omega) (by omega)) (hfree c hc j hj y (by omega) (by omega))
  · obtain ⟨y, rfl⟩ : ∃ y, x = o + y := ⟨x - o, by omega⟩
    have s := after y (by omega) (by omega)
    have := hsat c hc hQ j hj y (by omega) (by omega)
    unfold RowHolds at this ⊢
    rw [this, s.1, s.2.1]

/-- A copy's columns and bound inputs, as `place` writes them and as `Holds` reads them. -/
theorem calls_form {df : Def} {c : Call} {o j k : Nat} {inp : List Nat} {vals : Nat → Bool} :
    (fun y => vals (o + c.base + j * c.stride + y)) = (fun y => vals (o + (c.base + j * c.stride + y))) ∧
      (c.bound (df.ref o inp k) j).map vals = c.boundV (df.refV (fun y => vals (o + y)) (inp.map vals) (vals k)) j := by
  refine ⟨funext fun y => by congr 1; omega, ?_⟩
  rw [bound_val]
  exact congrArg (c.boundV · j) (funext fun x => ref_val df o inp k vals x)

theorem holds_calls_form {defs : Array Def} {kv : Bool} {fuel : Nat} {df : Def} {c : Call} {o j k : Nat}
    {inp : List Nat} {vals : Nat → Bool} (hk : kv = vals k)
    (h : Holds defs kv fuel c.def_ (fun y => vals (o + c.base + j * c.stride + y)) ((c.bound (df.ref o inp k) j).map vals)) :
    Holds defs kv fuel c.def_ (fun y => vals (o + (c.base + j * c.stride + y)))
      (c.boundV (df.refV (fun y => vals (o + y)) (inp.map vals) kv) j) := by
  obtain ⟨e1, e2⟩ := calls_form (df := df) (c := c) (o := o) (j := j) (k := k) (inp := inp) (vals := vals)
  subst hk
  rw [← e1, ← e2]
  exact h

/-- **A copy satisfies its Definition**: a copy `place` writes, all of whose columns were unwritten before it and whose
rows hold after it, satisfies `Holds` at the values of its columns, its bound inputs and the constant. -/
theorem place_holds {defs : Array Def} {k : Nat} (vals : Nat → Bool)
    (hfits : ∀ (d : Nat) (df : Def), defs[d]? = some df → df.fits defs = true)
    (hapart : ∀ (d : Nat) (df : Def), defs[d]? = some df → df.apart defs = true) (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → Sized acc → Clean acc →
      (∀ x, o ≤ x → x < o + (defs.getD d default).cols → Free x acc) →
      (∀ x, o ≤ x → x < o + (defs.getD d default).cols → RowHolds vals acc' x) →
      Holds defs (vals k) fuel d (fun y => vals (o + y)) (inp.map vals) := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h hsized hclean hfree hsat
    have h0 := h
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨-, hdd⟩ := getD_of_some hdf
      rw [hdd] at hfree hsat
      have hf := hfits d df hdf
      have hap := hapart d df hdf
      unfold Def.fits at hf
      simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hf
      obtain ⟨hrowsfit, hcallsfit⟩ := hf
      have hp := spans_pairwise hap
      unfold Def.spans at hp
      rw [List.pairwise_filter, List.pairwise_append, List.pairwise_map, List.pairwise_map] at hp
      obtain ⟨-, -, hprc⟩ := hp
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      let fr := fun (acc : Acc) (r : Nat × Array Nat × Array Nat) =>
        write acc (o + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref o inp k))) (Flock.Tmpl.canon (r.2.2.map (df.ref o inp k)))
      have hs1 : Sized acc1 := foldlM_inv Sized (fun _ _ u u' hu hw => write_sized hw hu) hsized h1
      have hc1 : Clean acc1 := foldlM_inv Clean (fun _ _ u u' hu hw => write_clean hw hu) hclean h1
      have hc' : Clean acc' := place_inv Clean write_clean _ h0 hclean
      -- the rows' writes miss every column but their own
      have rowsFrame : ∀ x, (∀ r ∈ df.rows.toList, x ≠ o + r.1) → Same3 x acc1 acc := fun x hx =>
        foldlM_inv (f := fr) (fun a => Same3 x a acc) (fun r hr u u' hu hw => same3_trans (write_same3 hw (hx r hr)) hu)
          ⟨rfl, rfl, rfl⟩ h1
      simp only [Holds, hdd]
      refine ⟨fun r hr => ?_, fun c hc j hj => ?_, fun y hy hrow hcopy => ?_⟩
      · -- a row: written once, kept to the end
        have hk1 := (foldlM_mem (f := fr) Sized
          (fun r acc => Kept (o + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref o inp k)))
            (Flock.Tmpl.canon (r.2.2.map (df.ref o inp k))) acc)
          (fun _ _ _ hu hw => write_sized hw hu) (fun _ _ _ _ _ hw hq => write_kept hw hq)
          (fun _ _ _ hu hw => write_made hu hw) hsized h1).2 r hr
        have hk := foldlM_inv (fun a => Kept (o + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref o inp k)))
            (Flock.Tmpl.canon (r.2.2.map (df.ref o inp k))) a)
          (fun c _ u u' hu hw => foldlM_inv (fun a => Kept (o + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref o inp k)))
            (Flock.Tmpl.canon (r.2.2.map (df.ref o inp k))) a)
            (fun _ _ w w' hw0 hp => place_inv _ (fun hw kk => write_kept hw kk) _ hp hw0) hu hw) hk1 h2
        have := hsat (o + r.1) (by omega) (by have := hrowsfit r hr; omega)
        unfold RowHolds at this
        rw [this, hk.2.1, hk.2.2, xorAt_canon, xorAt_canon, Array.toList_map, Array.toList_map, xorAt_map, xorAt_map]
        unfold rowVal
        simp only [ref_val]
      · -- a copy: its columns unwritten before the calls, and its rows holding after them
        refine holds_calls_form rfl (calls_holds (Q := fun _ => True) hfits
          (fun hp hs hc hf hsat' => ih hp hs hc hf hsat') hap h2 hs1 hc1
          (fun c hc j hj y hy1 hy2 => ?_) (fun c hc _ j hj y hy1 hy2 => ?_) c hc trivial j hj)
        · obtain ⟨s1, s2⟩ := copy_in_span hj hy1 hy2
          have hcf := hcallsfit c hc
          refine free_of_same3 (rowsFrame (o + y) fun r hr e => ?_) (hfree (o + y) (by omega) (by omega))
          have := hprc _ (List.mem_map.2 ⟨r, hr, rfl⟩) _ (List.mem_map.2 ⟨c, hc, rfl⟩) (by simp)
            (by simp only [decide_eq_true_eq]; omega)
          simp only [Apart] at this
          omega
        · obtain ⟨s1, s2⟩ := copy_in_span hj hy1 hy2
          have hcf := hcallsfit c hc
          exact hsat (o + y) (by omega) (by omega)
      · -- a gap: nothing writes it, so its row is empty
        have s1 := rowsFrame (o + y) fun r hr e => hrow r hr (by omega)
        have s2 : Same3 (o + y) acc' acc1 := foldlM_inv (fun a => Same3 (o + y) a acc1) (fun c hc u u' hu hw =>
            same3_trans (foldlM_inv (fun a => Same3 (o + y) a u) (fun j hj w w' hw0 hp => by
                have hn := copyAt_none hcopy c hc j (List.mem_range.1 hj)
                exact same3_trans (place_frame hfits _ hp (o + y) (by omega)) hw0) ⟨rfl, rfl, rfl⟩ hw) hu)
          ⟨rfl, rfl, rfl⟩ h2
        have hfr := free_of_same3 (same3_trans s2 s1) (hfree (o + y) (by omega) (by omega))
        have he := hc' (o + y) hfr
        have := hsat (o + y) (by omega) (by omega)
        unfold RowHolds at this
        rw [this, he.1, he.2]
        rfl

/-! ## Down the copies -/

/-- **`Holds` passes to every copy inside**: a copy that satisfies its Definition has each copy `place` writes inside it
satisfy its own, at the values of its columns and bound inputs. -/
theorem holds_within {defs : Array Def} {k : Nat} {vals : Nat → Bool} {f d o : Nat} {inp : List Nat} {f' e o' : Nat}
    {inp' : List Nat} (hw : Within defs k f d o inp f' e o' inp')
    (h : Holds defs (vals k) f d (fun y => vals (o + y)) (inp.map vals)) :
    Holds defs (vals k) f' e (fun y => vals (o' + y)) (inp'.map vals) := by
  induction hw with
  | refl => exact h
  | call c j hc hj _ ih =>
    simp only [Holds] at h
    apply ih
    obtain ⟨e1, e2⟩ := calls_form (df := _) (c := c) (o := _) (j := j) (k := k) (inp := _) (vals := vals)
    rw [e1, e2]
    exact h.2.1 c hc j hj

/-- Down the copies, a Definition stays one the flags mark ordered, or one with no columns. -/
theorem within_ordered {defs : Array Def} {k : Nat} {F : Array Bool}
    (hF : ∀ d, F.getD d false = true → (defs.getD d default).ordered F = true)
    (hfits : ∀ d, (defs.getD d default).fits defs = true) {f d o : Nat} {inp : List Nat} {f' e o' : Nat}
    {inp' : List Nat} (hw : Within defs k f d o inp f' e o' inp') :
    (F.getD d false = true ∨ (defs.getD d default).cols = 0) → (F.getD e false = true ∨ (defs.getD e default).cols = 0) := by
  induction hw with
  | refl => exact id
  | @call f d o inp f' e o' inp' c j hc hj _ ih =>
    intro h
    apply ih
    rcases h with h | h
    · have ho := hF d h
      unfold Def.ordered at ho
      simp only [Bool.and_eq_true, List.all_eq_true] at ho
      exact Or.inl (ho.2 c hc).1
    · have hf := hfits d
      unfold Def.fits at hf
      simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hf
      have := hf.2 c hc
      exact Or.inr (by omega)

/-- Down the copies, each Definition is below the fuel left and the template's size. -/
theorem within_lt {defs : Array Def} {k : Nat}
    (hwf : ∀ d, d < defs.size → (defs.getD d default).wellFormed defs d = true) {f d o : Nat} {inp : List Nat}
    {f' e o' : Nat} {inp' : List Nat} (hw : Within defs k f d o inp f' e o' inp') :
    d < f → d < defs.size → e < f' ∧ e < defs.size := by
  induction hw with
  | refl => exact fun h1 h2 => ⟨h1, h2⟩
  | @call f d o inp f' e o' inp' c j hc hj _ ih =>
    intro h1 h2
    have hw := hwf d h2
    unfold Def.wellFormed at hw
    simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hw
    have := (hw.2 c hc).1.1
    exact ih (by omega) (by omega)

/-! ## The unit -/

/-- **Every copy of the root's calls satisfies its Definition**, at column values that satisfy the unrolled rows. -/
theorem rows_holds {t : Tmpl} {acc : Acc} (hwf : t.wellFormed = true) (ha : t.apart = true) (hr : t.rows = .ok acc)
    (vals : Nat → Bool) (hsat : ∀ x, t.inCols ≤ x → x < t.constPos → RowHolds vals acc x) :
    ∀ c ∈ (t.defs.getD t.root default).calls.toList, t.inCols ≤ c.base → ∀ j < c.count,
      Holds t.defs (vals t.constPos) t.defs.size c.def_ (fun y => vals (c.base + j * c.stride + y))
        ((c.bound ((t.defs.getD t.root default).ref 0 [] t.constPos) j).map vals) := by
  unfold Tmpl.wellFormed at hwf
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq, beq_iff_eq] at hwf
  obtain ⟨⟨⟨hroot, -⟩, -⟩, -⟩ := hwf
  have ha' := ha
  unfold Tmpl.apart at ha'
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq] at ha'
  obtain ⟨hd, hk⟩ := ha'
  have hfits : ∀ (d : Nat) (df : Def), t.defs[d]? = some df → df.fits t.defs = true := fun d df hdf => by
    obtain ⟨h1, e⟩ := getD_of_some hdf; rw [← e]; exact (hd d h1).1
  have hapart : ∀ (d : Nat) (df : Def), t.defs[d]? = some df → df.apart t.defs = true := fun d df hdf => by
    obtain ⟨h1, e⟩ := getD_of_some hdf; rw [← e]; exact (hd d h1).2
  unfold Tmpl.rows at hr
  obtain ⟨acc1, h1, h2⟩ := bind_ok hr
  have hdf : t.defs[t.root]? = some (t.defs.getD t.root default) := by
    simp [Array.getD_eq_getD_getElem?, hroot]
  have hfit := hfits _ _ hdf
  have hap := hapart _ _ hdf
  generalize hdfe : t.defs.getD t.root default = df at hdf hsat hk h1 hfit hap ⊢
  simp only [place, hdf] at h1
  obtain ⟨acc2, h3, h4⟩ := bind_ok h1
  unfold Def.fits at hfit
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hfit
  have hp := spans_pairwise hap
  unfold Def.spans at hp
  rw [List.pairwise_filter, List.pairwise_append, List.pairwise_map, List.pairwise_map] at hp
  obtain ⟨-, -, hprc⟩ := hp
  let fr := fun (acc : Acc) (r : Nat × Array Nat × Array Nat) =>
    write acc (0 + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref 0 [] t.constPos)))
      (Flock.Tmpl.canon (r.2.2.map (df.ref 0 [] t.constPos)))
  have hs2 : Sized acc2 := foldlM_inv (f := fr) Sized (fun _ _ u u' hu hw => write_sized hw hu) (sized_empty _) h3
  have hc2 : Clean acc2 := foldlM_inv (f := fr) Clean (fun _ _ u u' hu hw => write_clean hw hu) (clean_empty _) h3
  have this := calls_holds (vals := vals) (Q := fun c => t.inCols ≤ c.base) hfits
    (fun hp hs hc hf hsat' => place_holds vals hfits hapart _ hp hs hc hf hsat')
    hap h4 hs2 hc2 (fun c hc j hj y hy1 hy2 => ?_) (fun c hc hQ j hj y hy1 hy2 => ?_)
  · intro c hc hQ j hj
    have h := this c hc hQ j hj
    simp only [Nat.zero_add] at h
    exact h
  · obtain ⟨s1, s2⟩ := copy_in_span hj hy1 hy2
    have hcf := hfit.2 c hc
    have hfree : Free (0 + y) (Acc.empty df.cols) := by
      simp only [Free, Acc.empty, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
      rw [ite_eq_left (by omega)]; rfl
    refine free_of_same3 (foldlM_inv (f := fr) (fun a => Same3 (0 + y) a (Acc.empty df.cols))
      (fun r hr u u' hu hw => same3_trans (write_same3 hw fun e => ?_) hu) ⟨rfl, rfl, rfl⟩ h3) hfree
    have := hprc _ (List.mem_map.2 ⟨r, hr, rfl⟩) _ (List.mem_map.2 ⟨c, hc, rfl⟩) (by simp)
      (by simp only [decide_eq_true_eq]; omega)
    simp only [Apart] at this
    omega
  · obtain ⟨s1, s2⟩ := copy_in_span hj hy1 hy2
    have hcf := hfit.2 c hc
    have hlt := hk _ (List.mem_filter.2 ⟨List.mem_append.2 (Or.inr (List.mem_map.2 ⟨c, hc, rfl⟩)),
      by simp only [decide_eq_true_eq]; omega⟩)
    simp only at hlt
    have s := write_same3 h2 (x := 0 + y) (by omega)
    have := hsat (0 + y) (by omega) (by omega)
    unfold RowHolds at this ⊢
    rw [← s.1, ← s.2.1]
    exact this

/-- **The eval composition**: column values that satisfy a templated unit's unrolled rows take, at every copy placed
under one of the root's calls past the input rows, at any depth, the value of that copy's Definition on the values of
its bound inputs, the constant at its column's value. -/
theorem rows_eval {t : Tmpl} {acc : Acc} (hwf : t.wellFormed = true) (ha : t.apart = true) (ho : t.ordered = true)
    (hr : t.rows = .ok acc) (vals : Nat → Bool) (hsat : ∀ x, t.inCols ≤ x → x < t.constPos → RowHolds vals acc x)
    {c : Call} {j f e o : Nat} {inp : List Nat} (hc : c ∈ (t.defs.getD t.root default).calls.toList)
    (hb : t.inCols ≤ c.base) (hj : j < c.count)
    (hw : Within t.defs t.constPos t.defs.size c.def_ (c.base + j * c.stride)
      (c.bound ((t.defs.getD t.root default).ref 0 [] t.constPos) j) f e o inp) :
    ∀ y < (t.defs.getD e default).cols, vals (o + y) = t.evalDef e (inp.map vals) (vals t.constPos) y := by
  have hH := holds_within hw (rows_holds hwf ha hr vals hsat c hc hb j hj)
  have hwf' := hwf
  unfold Tmpl.wellFormed at hwf'
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq, beq_iff_eq] at hwf'
  obtain ⟨⟨⟨hroot, -⟩, -⟩, hdefs⟩ := hwf'
  have hcroot : c.def_ < t.root := by
    have hw0 := hdefs t.root hroot
    unfold Def.wellFormed at hw0
    simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hw0
    exact (hw0.2 c hc).1.1
  obtain ⟨hef, hes⟩ := within_lt hdefs hw (by omega) (by omega)
  unfold Tmpl.ordered at ho
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at ho
  obtain ⟨hord, hroot'⟩ := ho
  have hF : ∀ d, (flags t.defs).getD d false = true → (t.defs.getD d default).ordered (flags t.defs) = true := by
    intro d hd
    by_cases h : d < t.defs.size
    · exact imp_of_or (hord d h).2 hd
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl
  have hfitsAll : ∀ d, (t.defs.getD d default).fits t.defs = true := by
    intro d
    by_cases h : d < t.defs.size
    · exact (hord d h).1
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl
  unfold Tmpl.rootOrdered at hroot'
  simp only [Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq,
    List.mem_append] at hroot'
  have h0 : (flags t.defs).getD c.def_ false = true ∨ (t.defs.getD c.def_ default).cols = 0 := by
    rcases hroot'.2 c hc with hin | ⟨⟨-, hFc⟩, -⟩
    · exact Or.inr (by omega)
    · exact Or.inl hFc
  intro y hy
  rcases within_ordered hF hfitsAll hw h0 with hFe | hz
  · rw [holds_eval hF f hFe hH y hy]
    unfold Tmpl.evalDef
    rw [eval_fuel hdefs f t.defs.size e hef hes hes]
  · omega

end Flock.TmplValue
