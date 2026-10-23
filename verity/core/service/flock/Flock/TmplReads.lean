/-
A templated unit's input columns that a computed row surely reads, from its description (PRF-133). `RecResiduals.readIns`
marks the input columns the unit's computed rows read by walking every row of the unrolled net; `Tmpl.readsSure` marks a
subset of them per Definition, so `KnownRead.readsOk` can check a templated unit's known bits without building its rows.

A row's side is a sum over GF(2), so a reference can cancel against another bound to the same column. A side that names
exactly one of its Definition's inputs, once (`soleIn`), keeps that input's column wherever the copy's inputs are bound:
its other references are the copy's own columns, which an ordered template binds no input to, and the constant. An input
of a Definition is then surely read where one of its rows has it as its side's sole input, or where one of its calls binds
it to a callee's input that is (`Def.sure`, Definitions in order, each callee before its callers). The root marks its
input rows' columns the same way, from its rows and calls past the input rows (`Tmpl.readsSure`): those are the computed
rows. Its cost is each Definition's rows and each call's bound inputs, never the unit's gates.
-/
import Flock.Tmpl

namespace Flock.Tmpl

/-- The one reference `v - lo` among `[lo, hi)` that side `xs` names, where it names exactly one such reference, once. -/
def soleIn (lo hi : Nat) (xs : Array Nat) : Option Nat :=
  match xs.toList.filter (fun v => decide (lo ≤ v ∧ v < hi)) with
  | [v] => some (v - lo)
  | _ => none

/-- Mark, in `m`, each reference `v - lo` in `[lo, hi)` that some copy of call `c` binds to an input its callee surely
reads (`se`). A segment of step 0 binds every copy alike, so only its first copy is read. -/
def Call.marks (c : Call) (se : Array Bool) (lo hi : Nat) (m : Array Bool) : Array Bool :=
  (c.segs.foldl (fun (acc : Array Bool × Nat) s =>
    let copies := if s.step == 0 then min 1 c.count else c.count
    let m := (List.range s.reps).foldl (fun m r => (List.range s.len).foldl (fun m t =>
      if se.getD (acc.2 + r * s.len + t) false then
        (List.range copies).foldl (fun m j =>
          let v := s.at_ + r * s.rstride + j * s.step + t
          if lo ≤ v ∧ v < hi then m.setIfInBounds (v - lo) true else m) m
      else m) m) acc.1
    (m, acc.2 + s.reps * s.len)) (m, 0)).1

/-- Mark, in `m`, each reference `v - lo` in `[lo, hi)` that a row of `rows` names as its side's sole one. -/
def rowMarks (rows : Array (Nat × Array Nat × Array Nat)) (lo hi : Nat) (m : Array Bool) : Array Bool :=
  rows.foldl (fun m r =>
    let m := match soleIn lo hi r.2.1 with | some i => m.setIfInBounds i true | none => m
    match soleIn lo hi r.2.2 with | some i => m.setIfInBounds i true | none => m) m

/-- Definition `df`'s inputs that each of its copies surely reads (`S`: the earlier Definitions'). -/
def Def.sure (S : Array (Array Bool)) (df : Def) : Array Bool :=
  df.calls.foldl (fun m c => c.marks (S.getD c.def_ #[]) df.cols (df.cols + df.ins) m)
    (rowMarks df.rows df.cols (df.cols + df.ins) (Array.replicate df.ins false))

/-- Each Definition's surely read inputs, in order. -/
def sures (defs : Array Def) : Array (Array Bool) := defs.foldl (fun S df => S.push (df.sure S)) #[]

/-- **The input rows' columns some computed row surely reads**: the root's rows past the input rows, and its calls past
them, each read as `Def.sure` reads a Definition's, over the root's columns below `inCols`. -/
def Tmpl.readsSure (t : Tmpl) : Array Bool :=
  let S := sures t.defs
  let df := t.defs.getD t.root default
  (df.calls.filter fun c => decide (t.inCols ≤ c.base)).foldl (fun m c => c.marks (S.getD c.def_ #[]) 0 t.inCols m)
    (rowMarks (df.rows.filter fun r => decide (t.inCols ≤ r.1)) 0 t.inCols (Array.replicate t.inCols false))

end Flock.Tmpl
