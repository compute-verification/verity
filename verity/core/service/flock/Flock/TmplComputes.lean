/-
What a Definition of a template computes, as a fact a leaf's proof supplies (PRF-133). `Tmpl.DefComputes t d outs spec`:
on any inputs, with the constant 1, Definition `d`'s columns `outs` are `spec`'s bits (`Tmpl.evalDef`). A leaf proved
relationally plugs in through `defComputes_of_rows`: `eval` satisfies the Definition's own rows (`eval_rowsAt`), so a
proof that any column values satisfying those rows, with the constant at 1, carry `spec` on `outs` proves
`DefComputes`. The rows are read as a partial map from the column each writes to its `A` and `B` references
(`Def.rowsAt`), in the Definition's own references: its columns, then its inputs, then the constant, the form
`FlockSoundness.Types.Dag.SatAt` reads, so the typed path's `layout_sound` gives `DefComputes` with `spec` its
`evalT` (`FlockSoundness.TmplValue.defComputes_of_layout`). Nothing on a verifier path reads this file.
-/
import Flock.TmplValue

namespace Flock.Tmpl

/-- Definition `df`'s rows as a partial map: the column `c` the first row that writes it, to that row's `A` and `B`
references. -/
def Def.rowsAt (df : Def) (c : Nat) : Option (List Nat × List Nat) :=
  (df.rows.toList.find? (·.1 == c)).map fun r => (r.2.1.toList, r.2.2.toList)

/-- **Definition `d` of `t` computes `spec` on its columns `outs`**: on inputs `x` (its first `ins`), with the constant
1, its column `outs[o]` is bit `o` of `spec x`. -/
def Tmpl.DefComputes (t : Tmpl) (d : Nat) (outs : List Nat) (spec : (Nat → Bool) → List Bool) : Prop :=
  ∀ (x : Nat → Bool), ∀ o < outs.length,
    t.evalDef d ((List.range (t.defs.getD d default).ins).map x) true (outs.getD o 0) = (spec x).getD o false

end Flock.Tmpl

namespace Flock.TmplValue

open Flock.Tmpl

/-- **`eval` satisfies a Definition's own rows**: read through its references' values (`refV`), the row `rowsAt` gives
for column `c` holds at `eval`, when each row reads only earlier columns, the inputs and the constant, and writes inside
the Definition's columns. -/
theorem eval_rowsAt {defs : Array Def} {kv : Bool} {fuel d : Nat} {inp : List Bool}
    (hord : ∀ r ∈ (defs.getD d default).rows.toList, ∀ w ∈ r.2.1.toList ++ r.2.2.toList,
      (defs.getD d default).cols ≤ w ∨ w < r.1)
    (hcols : ∀ r ∈ (defs.getD d default).rows.toList, r.1 < (defs.getD d default).cols)
    {c : Nat} {r : List Nat × List Nat} (hr : (defs.getD d default).rowsAt c = some r) :
    (defs.getD d default).refV (eval defs kv (fuel + 1) d inp) inp kv c =
      (xorAt r.1 ((defs.getD d default).refV (eval defs kv (fuel + 1) d inp) inp kv) &&
        xorAt r.2 ((defs.getD d default).refV (eval defs kv (fuel + 1) d inp) inp kv)) := by
  have hev : ∀ y, eval defs kv (fuel + 1) d inp y =
      (defs.getD d default).upTo defs kv (eval defs kv fuel) inp (y + 1) y := fun _ => rfl
  generalize hdf : defs.getD d default = df at hord hcols hr hev ⊢
  unfold Def.rowsAt at hr
  cases hfind : df.rows.toList.find? (·.1 == c) with
  | none => rw [hfind] at hr; cases hr
  | some r' =>
    rw [hfind] at hr
    simp only [Option.map_some, Option.some.injEq] at hr
    subst hr
    have hr' := List.mem_of_find?_eq_some hfind
    have hrc : r'.1 = c := by simpa using List.find?_some hfind
    have hc : c < df.cols := hrc ▸ hcols r' hr'
    have below : ∀ w, df.cols ≤ w ∨ w < c → df.refV (df.upTo defs kv (eval defs kv fuel) inp c) inp kv w =
        df.refV (eval defs kv (fuel + 1) d inp) inp kv w := by
      intro w hw
      unfold Def.refV
      by_cases h1 : w < df.cols
      · rw [ite_eq_left h1, ite_eq_left h1, hev]
        exact upTo_lt (by omega)
      · rw [ite_eq_right h1, ite_eq_right h1]
    have hstep : df.upTo defs kv (eval defs kv fuel) inp (c + 1) c =
        df.colAt defs kv (eval defs kv fuel) inp (df.upTo defs kv (eval defs kv fuel) inp c) c := ite_eq_left rfl
    have hcol : df.refV (eval defs kv (fuel + 1) d inp) inp kv c =
        rowVal (df.refV (df.upTo defs kv (eval defs kv fuel) inp c) inp kv) r' := by
      rw [show df.refV (eval defs kv (fuel + 1) d inp) inp kv c = eval defs kv (fuel + 1) d inp c by
        unfold Def.refV; rw [ite_eq_left hc], hev, hstep]
      unfold Def.colAt
      rw [hfind]
    rw [hcol]
    unfold rowVal
    rw [xorAt_congr fun w hw => below w (by rw [← hrc]; exact hord r' hr' w (List.mem_append_left _ hw)),
      xorAt_congr fun w hw => below w (by rw [← hrc]; exact hord r' hr' w (List.mem_append_right _ hw))]

/-- **A leaf proved on its rows computes**: if any column values that satisfy Definition `d`'s rows, with the constant
at 1, carry `spec` of their inputs on `outs`, then `d` computes `spec` on `outs`. -/
theorem defComputes_of_rows {t : Tmpl} {d : Nat} {outs : List Nat} {spec : (Nat → Bool) → List Bool}
    (hd : d < t.defs.size)
    (hord : ∀ r ∈ (t.defs.getD d default).rows.toList, ∀ w ∈ r.2.1.toList ++ r.2.2.toList,
      (t.defs.getD d default).cols ≤ w ∨ w < r.1)
    (hcols : ∀ r ∈ (t.defs.getD d default).rows.toList, r.1 < (t.defs.getD d default).cols)
    (hout : ∀ o < outs.length, outs.getD o 0 < (t.defs.getD d default).cols)
    (hspec : ∀ x y : Nat → Bool, (∀ i < (t.defs.getD d default).ins, x i = y i) → spec x = spec y)
    (hsound : ∀ z : Nat → Bool,
      (∀ c r, (t.defs.getD d default).rowsAt c = some r → z c = (xorAt r.1 z && xorAt r.2 z)) →
      z ((t.defs.getD d default).cols + (t.defs.getD d default).ins) = true →
      ∀ o < outs.length,
        z (outs.getD o 0) = (spec fun i => z ((t.defs.getD d default).cols + i)).getD o false) :
    t.DefComputes d outs spec := by
  intro x o ho
  obtain ⟨fuel, hf⟩ : ∃ fuel, t.defs.size = fuel + 1 := ⟨t.defs.size - 1, by omega⟩
  let inp := (List.range (t.defs.getD d default).ins).map x
  let z := (t.defs.getD d default).refV (eval t.defs true (fuel + 1) d inp) inp true
  have hz := hsound z (fun c r hr => eval_rowsAt hord hcols hr)
    (by simp only [z, Def.refV]; rw [ite_eq_right (by omega), ite_eq_right (by omega)]) o ho
  have hin : (fun i => z ((t.defs.getD d default).cols + i)) = fun i =>
      if i < (t.defs.getD d default).ins then x i else true := by
    funext i
    simp only [z, Def.refV, inp]
    rw [ite_eq_right (by omega)]
    by_cases hi : i < (t.defs.getD d default).ins
    · rw [ite_eq_left (by omega), ite_eq_left hi, Nat.add_sub_cancel_left, List.getD_eq_getElem?_getD,
        List.getElem?_map, List.getElem?_range hi]
      rfl
    · rw [ite_eq_right (by omega), ite_eq_right hi]
  rw [hin, hspec _ x fun i hi => by rw [ite_eq_left hi]] at hz
  have hzo : z (outs.getD o 0) = eval t.defs true (fuel + 1) d inp (outs.getD o 0) := by
    simp only [z, Def.refV]; rw [ite_eq_left (hout o ho)]
  rw [hzo] at hz
  unfold Tmpl.evalDef
  rw [hf]
  exact hz

/-! ## A copy's inputs number its Definition's -/

theorem length_flatMap_const {α β : Type} {f : α → List β} {k : Nat} :
    ∀ {l : List α}, (∀ a ∈ l, (f a).length = k) → (l.flatMap f).length = l.length * k
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.length_append, h a (List.mem_cons_self ..),
      length_flatMap_const fun b hb => h b (List.mem_cons_of_mem a hb), List.length_cons, Nat.succ_mul, Nat.add_comm]

/-- A copy's bound inputs number its segments' bits. -/
theorem bound_length (c : Call) (g : Nat → Nat) (j : Nat) :
    (c.bound g j).length = c.segs.foldl (fun n s => n + s.reps * s.len) 0 := by
  have inner : ∀ s : Seg, ((List.range s.reps).flatMap fun r =>
      (List.range s.len).map fun t => g (s.at_ + r * s.rstride + j * s.step + t)).length = s.reps * s.len := fun s => by
    rw [length_flatMap_const (k := s.len) (fun r _ => by simp), List.length_range]
  unfold Call.bound
  rw [← Array.foldl_toList]
  suffices h : ∀ (l : List Seg) (n : Nat), n + (l.flatMap fun s => (List.range s.reps).flatMap fun r =>
      (List.range s.len).map fun t => g (s.at_ + r * s.rstride + j * s.step + t)).length =
      l.foldl (fun n s => n + s.reps * s.len) n by simpa using h c.segs.toList 0
  intro l
  induction l with
  | nil => intro n; simp
  | cons s l ih => intro n; rw [List.flatMap_cons, List.length_append, inner, List.foldl_cons, ← ih, Nat.add_assoc]

/-- Every Definition of a well-formed template is well formed, and so is the default past its last. -/
theorem wf_getD {t : Tmpl} (hwf : t.wellFormed = true) : ∀ d, (t.defs.getD d default).wellFormed t.defs d = true := by
  unfold Tmpl.wellFormed at hwf
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at hwf
  intro d
  by_cases hd : d < t.defs.size
  · exact hwf.2 d hd
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]
    rfl

/-- A call's copy, in a well-formed Definition, is bound to as many inputs as its callee has. -/
theorem call_bound_length {defs : Array Def} {d : Nat} (hwf : (defs.getD d default).wellFormed defs d = true) {c : Call}
    (hc : c ∈ (defs.getD d default).calls.toList) (g : Nat → Nat) (j : Nat) :
    (c.bound g j).length = (defs.getD c.def_ default).ins := by
  unfold Def.wellFormed at hwf
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hwf
  rw [bound_length]
  exact (hwf.2 c hc).2

/-- **A copy placed inside a copy is bound to as many inputs as its Definition has**, when the outer one is. -/
theorem within_ins {defs : Array Def} {k : Nat} (hwf : ∀ d, (defs.getD d default).wellFormed defs d = true)
    {f d o : Nat} {inp : List Nat} {f' e o' : Nat} {inp' : List Nat} (hw : Within defs k f d o inp f' e o' inp') :
    inp.length = (defs.getD d default).ins → inp'.length = (defs.getD e default).ins := by
  induction hw with
  | refl => exact id
  | call c j hc hj h ih => exact fun _ => ih (call_bound_length (hwf _) hc _ j)

end Flock.TmplValue
