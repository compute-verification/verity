/-
The check the rows `deriveAll` builds pass before the verifier uses them (Project doc `verified-lowering-design.md` §1.2;
S3c). `deriveChecked` walks each layout's type item by item against its rows:

* an AND item's row is `form_a · form_b`, each form over the columns of the wires it reads;
* an inlined call walks its callee's items on the call's input forms, in the caller's rows;
* a placed call is its entry's segment: every row of the callee sits at its base, shifted, except its constant, which
  copies the caller's, and its bound inputs, each `form · [callee constant]` (the unit's are Δ's `[one] · [one]` and
  `form · form`); an exported input bit's form is one caller wire, held on the callee's input column;
* the input rows are self rows, the output rows copy the output forms, and the constant row copies the constant.

`compose_sound` (soundness package) is proved from the walk alone, so it holds of whatever `deriveAll` builds. Generated
read layouts and read items are refused for now; S3c-2 checks them.
-/
import Flock.DeriveAll

namespace Flock.DeriveAll

open Lean (Json)

/-- Row `r` shifted by `b`, as a placed callee's rows sit in its caller. -/
def shiftRow (b : Nat) (r : List Nat × List Nat) : List Nat × List Nat := (r.1.map (· + b), r.2.map (· + b))

/-- The unit's row `c` in the block: its Δ entry (`[one] · [one]`, or `src · src`), else its derived row. `one` is the
block's constant. -/
def blockRow (d : D) (one c : Nat) : Option (List Nat × List Nat) :=
  match d.delta.find? (·.1 == c) with
  | some (_, none) => some ([one], [one])
  | some (_, some s) => some (s, s)
  | none => d.rows[c]?

/-- A layout's rows, and whether they are the unit's, whose binding and constant copies are Δ entries. -/
structure Chk where
  d : D
  unit : Bool

namespace Chk

/-- Row `c` is `r`: an inner layout's row, or the unit's where Δ doesn't rewrite it. -/
def rowIs (k : Chk) (c : Nat) (r : List Nat × List Nat) : Bool :=
  k.d.rows[c]? == some r && (!k.unit || (k.d.delta.find? (·.1 == c)).isNone)

/-- Row `c` copies the constant: an inner layout's `[const] · [const]`, the unit's Δ entry `one`. -/
def oneAt (k : Chk) (c : Nat) : Bool :=
  if k.unit then k.d.delta.find? (·.1 == c) == some (c, none) else k.d.rows[c]? == some ([k.d.const], [k.d.const])

/-- Row `c` binds to the columns `src`: an inner layout's `src · [cb]`, the unit's Δ entry `src`, read as `src · src`. -/
def bindAt (k : Chk) (c : Nat) (src : List Nat) (cb : Nat) : Bool :=
  if k.unit then k.d.delta.find? (·.1 == c) == some (c, some src) else k.d.rows[c]? == some (src, [cb])

end Chk

/-- A placed call's rows: the callee `sub` at `base`, `ex` its exported bits, on the caller's wires and the call's input
forms. -/
def placedOk (k : Chk) (sub : D) (base : Nat) (ex : List Nat) (wire : Array CF) (args : Array Form) : Bool :=
  k.oneAt (base + sub.const) &&
  (List.range args.size).all (fun i =>
    let col := base + sub.inCols.getD i 0
    let f := args.getD i default
    if ex.contains i then
      match f.ws.toList with
      | [w] => f.c == 0 && wire[w]? == some ([col], 0)
      | _ => false
    else k.bindAt col (resolve k.d.const (ent (formOf wire f))) (base + sub.const)) &&
  (List.range sub.rows.size).all (fun c =>
    c == sub.const || (List.range args.size).any (fun i => !ex.contains i && sub.inCols.getD i 0 == c) ||
      k.rowIs (base + c) (shiftRow base (sub.rows.getD c ([], []))))

/-- Walk the items of type `t` under layout `l`, its call entries from `c` on, against `refs`, from the wires `wire`.
`jl` is the layout whose items these are; an inlined callee's layout is listed before it, which bounds the walk. `top`
marks the layout's own items, the only place a placed call sits. Returns the wires and the refs left. -/
def walk (types : String → Option CircuitType) (ls : Array Layout) (done : Array D) (unit j : Nat) (k : Chk)
    (jl : Nat) (top : Bool) (t : CircuitType) (l : Layout) (items : List Item) (c : Nat) (wire : Array CF)
    (refs : List Ref) : Option (Array CF × List Ref) :=
  match items with
  | [] => some (wire, refs)
  | .and fa fb :: rest =>
    match refs with
    | .row col :: refs' =>
      if fa.c ≤ 1 && fb.c ≤ 1 &&
          k.rowIs col (resolve k.d.const (ent (formOf wire fa)), resolve k.d.const (ent (formOf wire fb))) then
        walk types ls done unit j k jl top t l rest c (wire.push ([col], 0)) refs'
      else none
    | _ => none
  | .call kk args :: rest =>
    match (t.calls[kk]?).bind types with
    | none => none
    | some u =>
      let e := l.calls.getD c default
      match e.offset with
      | none =>
        if _h : e.layout < jl then
          if args.size == u.inBits && args.all (·.c ≤ 1) && u.outputs.size == u.outBits && u.outputs.all (·.c ≤ 1) then
            match walk types ls done unit j k e.layout false u (ls.getD e.layout default) u.items.toList 0
                (args.map (formOf wire)) refs with
            | some (w', refs') =>
              walk types ls done unit j k jl top t l rest (c + 1) (wire ++ u.outputs.map (formOf w')) refs'
            | none => none
          else none
        else none
      | some _ =>
        match refs with
        | .seg jj base ex :: refs' =>
          let sub := done.getD jj default
          if top && jj == e.layout && jj < j && jj != unit && (ls.getD jj default).type == t.calls[kk]? &&
              args.size == u.inBits && args.all (·.c ≤ 1) && sub.inCols.size == u.inBits &&
              sub.outCols.size == u.outBits && placedOk k sub base ex wire args then
            walk types ls done unit j k jl top t l rest (c + 1) (wire ++ sub.outCols.map fun col => ([base + col], 0))
              refs'
          else none
        | _ => none
  | _ :: _ => none
termination_by (jl, items.length)
decreasing_by
  all_goals first
    | exact Prod.Lex.right _ (by simp)
    | exact Prod.Lex.left _ _ _h

/-- **Layout `j`'s rows pass the check.** -/
def checkLayout (types : String → Option CircuitType) (ls : Array Layout) (done : Array D) (unit j : Nat) : Bool :=
  let l := ls.getD j default
  let d := done.getD j default
  let k : Chk := ⟨d, j == unit⟩
  match l.own, l.type.bind types with
  | .derive, some t =>
    t.outputs.size == t.outBits && t.outputs.all (·.c ≤ 1) && d.inCols.size == t.inBits &&
      d.outCols.size == t.outputs.size && k.oneAt d.const &&
      (List.range t.inBits).all (fun w => k.rowIs (d.inCols.getD w 0) ([d.inCols.getD w 0], [d.inCols.getD w 0])) &&
      match walk types ls done unit j k j true t l t.items.toList 0
          ((Array.range t.inBits).map fun w => ([d.inCols.getD w 0], 0)) d.items.toList with
      | some (wire, []) => (List.range t.outputs.size).all fun o =>
          k.rowIs (d.outCols.getD o 0) (resolve d.const (ent (formOf wire (t.outputs.getD o default))), [d.const])
      | _ => false
  | _, _ => false

/-- **The rows the verifier uses**: `deriveAll`'s, once every layout's pass the check. -/
def deriveChecked (types : String → Option CircuitType) (ls : Array Layout) (info : String → Option TableInfo)
    (words : String → Option (Array Nat)) (unit : Nat) : Except String (Array D) := do
  let done ← deriveAll types ls info words unit
  match (List.range ls.size).find? fun j => !checkLayout types ls done unit j with
  | some j => throw s!"derive: layout {j}'s rows fail the check"
  | none => return done

/-- `flock-rows --archive FILE --part check`: `ok` when every layout's rows pass the check. -/
def archiveCheck (j : Json) : V String := do
  let (types, ls, info, words, unit, _) ← load j
  let _ ← deriveChecked types ls info words unit
  return "ok\n"

end Flock.DeriveAll
