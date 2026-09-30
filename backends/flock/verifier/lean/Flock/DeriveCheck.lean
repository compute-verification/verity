/-
The check the rows `deriveAll` builds pass before the verifier uses them (Project doc `verified-lowering-design.md` §1.2;
S3c). `deriveChecked` walks each layout's type item by item against its rows:

* an AND item's row is `form_a · form_b`, each form over the columns of the wires it reads;
* an inlined call walks its callee's items on the call's input forms, in the caller's rows;
* a placed call is its entry's segment: every row of the callee sits at its base, shifted, except its constant, which
  copies the caller's, and its bound inputs, each `form · [callee constant]` (the unit's are Δ's `[one] · [one]` and
  `form · form`); an exported input bit's form is one caller wire, held on the callee's input column;
* the input rows are self rows, the output rows copy the output forms, and the constant row copies the constant.

It also checks the unit's logical order (`order`, the list `Rows.compose` reads): each of the unit's rows is listed once,
the inputs first; each row after them reads only earlier columns; every row outside it is a zero row. And it checks the
unit's parts as its template statement reads them (`partsChecked`): each part's rows are its callee's, shifted, past the
unit's own rows and apart from the other parts'; Δ is exactly the unit's constant and the parts' constants and bound
inputs; the own rows read only own and part columns; the unit's constant row is `[const]·[const]`; and its order lists
only own and part columns.

An inline read is checked as `table/v2` lays it out: its decoders' rows (`chkDecode`), its product rows, each a high minterm
times the table-direct side its record gives (`readSide`, the side `Lookup.foldB` folds), and its output rows; a value bit
past the product bits must be the same in every word of the table. A placed read is its entry's segment, embedded as a
placed call's is (every index bit bound); its callee is a generated layout, checked the same way in its own coordinates
(`genOk`: inputs `0 … n − 1`, the decoders over them, the product rows, the output rows at its output columns). The rows a
layout's soundness is stated over are `fullRow`, the derived rows with each product row's table-direct side, and for the
unit `blockRow`, those with Δ applied.

`compose_sound` (soundness package) is proved from the walk alone, so it holds of whatever `deriveAll` builds.
-/
import Std.Data.HashSet
import Flock.DeriveAll

namespace Flock.DeriveAll

open Lean (Json)

/-- Row `r` shifted by `b`, as a placed callee's rows sit in its caller. -/
def shiftRow (b : Nat) (r : List Nat × List Nat) : List Nat × List Nat := (r.1.map (· + b), r.2.map (· + b))

/-- A read record covers product row `c`: `prod0 ≤ c < prod0 + k · 2^(n − lo)`. -/
def ReadRec.covers (r : ReadRec) (c : Nat) : Bool := r.prod0 ≤ c && c < r.prod0 + r.k * 2 ^ (r.n - r.lo)

/-- A read's table-direct side at product row `c = prod0 + j·nh + h` (`nh = 2^(n − lo)`): the low minterm columns
`loTop + l` for which bit `j` of word `h·2^lo + l` is set, the side `Lookup.foldB` folds. -/
def readSide (ws : Array Nat) (r : ReadRec) (c : Nat) : List Nat :=
  ((List.range (2 ^ r.lo)).filter fun l =>
    (ws.getD ((c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + l) 0).testBit ((c - r.prod0) / 2 ^ (r.n - r.lo))).map
    (r.loTop + ·)

/-- Row `c` of a layout as the verifier folds it: a read's product row takes its table-direct side from its record,
the table's words by SHA-512 in `words`. -/
def fullRow (words : String → Option (Array Nat)) (d : D) (c : Nat) : Option (List Nat × List Nat) :=
  match d.reads.find? (·.covers c) with
  | some r => d.rows[c]?.map fun p => (p.1, readSide ((words r.table).getD #[]) r c)
  | none => d.rows[c]?

/-- The unit's row `c` in the block: its Δ entry (`[one] · [one]`, or `src · src`), else its row as the verifier folds
it. `one` is the block's constant. -/
def blockRow (words : String → Option (Array Nat)) (d : D) (one c : Nat) : Option (List Nat × List Nat) :=
  match d.delta.find? (·.1 == c) with
  | some (_, none) => some ([one], [one])
  | some (_, some s) => some (s, s)
  | none => fullRow words d c

/-- A layout's rows, and whether they are the unit's, whose binding and constant copies are Δ entries. -/
structure Chk where
  d : D
  unit : Bool

namespace Chk

/-- Row `c` is `r`: no read's product row, and (the unit's) not rewritten by Δ. -/
def rowIs (k : Chk) (c : Nat) (r : List Nat × List Nat) : Bool :=
  k.d.rows[c]? == some r && (k.d.reads.find? (·.covers c)).isNone && (!k.unit || (k.d.delta.find? (·.1 == c)).isNone)

/-- Row `c` is a read's product row `a · (side)`: derived as `a · []`, its side from record `r`. -/
def prodIs (k : Chk) (c : Nat) (a : List Nat) (r : ReadRec) : Bool :=
  k.d.rows[c]? == some (a, []) && decide (k.d.reads.find? (·.covers c) = some r) &&
    (!k.unit || (k.d.delta.find? (·.1 == c)).isNone)

/-- Row `c` copies the constant: an inner layout's `[const] · [const]`, the unit's Δ entry `one`. -/
def oneAt (k : Chk) (c : Nat) : Bool :=
  if k.unit then k.d.delta.find? (·.1 == c) == some (c, none)
  else k.d.rows[c]? == some ([k.d.const], [k.d.const]) && (k.d.reads.find? (·.covers c)).isNone

/-- Row `c` binds to the columns `src`: an inner layout's `src · [cb]`, the unit's Δ entry `src`, read as `src · src`. -/
def bindAt (k : Chk) (c : Nat) (src : List Nat) (cb : Nat) : Bool :=
  if k.unit then k.d.delta.find? (·.1 == c) == some (c, some src)
  else k.d.rows[c]? == some (src, [cb]) && (k.d.reads.find? (·.covers c)).isNone

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
      match sub.reads.find? (·.covers c) with
      | some r => k.prodIs (base + c) ((sub.rows.getD c ([], [])).1.map (· + base)) (r.shift base)
      | none => k.rowIs (base + c) (shiftRow base (sub.rows.getD c ([], []))))

/-- Rows `i, i + 1, …` of a decoder block: row `i` is `lo[i mod 2^h] · hi[i / 2^h]`, and minterm `i` is its column. -/
def blockRows (k : Chk) (lo hi : Array CF) (h : Nat) : Nat → Nat → List Ref → Array CF → Option (Array CF × List Ref)
  | 0, _, refs, acc => some (acc, refs)
  | cnt + 1, i, .row col :: refs, acc =>
    if k.rowIs col (resolve k.d.const (ent (lo.getD (i % 2 ^ h) ([], 0))),
        resolve k.d.const (ent (hi.getD (i / 2 ^ h) ([], 0)))) then
      blockRows k lo hi h cnt (i + 1) refs (acc.push ([col], 0))
    else none
  | _ + 1, _, _, _ => none

/-- `decode`'s rows for `leaves`, checked against `refs`: a block of two or more leaves commits its minterms, from the
minterms of its low and high halves. Returns the minterms and the refs left. -/
def chkDecode (k : Chk) (leaves : Array CF) (refs : List Ref) : Option (Array CF × List Ref) :=
  if _hs : leaves.size ≤ 1 then
    let f := leaves.getD 0 ([], 0)
    some (#[(f.1, (f.2 + 1) % 2), f], refs)
  else
    let h := leaves.size / 2
    match chkDecode k (leaves.extract 0 h) refs with
    | none => none
    | some (lo, refs1) =>
      match chkDecode k (leaves.extract h leaves.size) refs1 with
      | none => none
      | some (hi, refs2) => blockRows k lo hi h (2 ^ leaves.size) 0 refs2 #[]
termination_by leaves.size
decreasing_by all_goals simp only [Array.size_extract]; omega

/-- A read's product rows `x = j·nh + h`, from `prod0` on: row `prod0 + x` is `hi[h] · (side)`, its side from `rec`. -/
def prodRows (k : Chk) (rec : ReadRec) (hiM : Array CF) (nh : Nat) : Nat → Nat → List Ref → Option (List Ref)
  | 0, _, refs => some refs
  | cnt + 1, x, .row col :: refs =>
    if col == rec.prod0 + x && k.prodIs col (resolve k.d.const (ent (hiM.getD (x % nh) ([], 0)))) rec then
      prodRows k rec hiM nh cnt (x + 1) refs
    else none
  | _ + 1, _, _ => none

/-- A read's output bits `j, j + 1, …`: bit `j < kk` is its row `Σ_h product(j, h) · [const]`; any other bit is the same
in every word of the table, and wired as its value. -/
def outRows (k : Chk) (rec : ReadRec) (ws : Array Nat) (kk nh : Nat) : Nat → Nat → List Ref → Array CF →
    Option (Array CF × List Ref)
  | 0, _, refs, acc => some (acc, refs)
  | cnt + 1, j, refs, acc =>
    if j < kk then
      match refs with
      | .row col :: refs' =>
        if k.rowIs col ((List.range nh).map (rec.prod0 + j * nh + ·), [k.d.const]) then
          outRows k rec ws kk nh cnt (j + 1) refs' (acc.push ([col], 0))
        else none
      | _ => none
    else
      if (List.range ws.size).all fun i => (ws.getD i 0).testBit j == (ws.getD 0 0).testBit j then
        outRows k rec ws kk nh cnt (j + 1) refs (acc.push ([], if (ws.getD 0 0).testBit j then 1 else 0))
      else none

/-- An inline read's rows after its decoders (`table/v2`): its record, whose low minterms are the columns
`loTop + l`; its `kk · nh` product rows; its output bits. `kk` is the product bit count the table gives
(`Layout.kOf`). Returns the output wires. -/
def readOk (k : Chk) (ws : Array Nat) (sha : String) (n lo vb : Nat) (loM hiM : Array CF) (refs : List Ref) :
    Option (Array CF × List Ref) :=
  let kk := Layout.kOf sha vb (Layout.tableInfo ws)
  let nh := 2 ^ (n - lo)
  if kk = 0 then outRows k default ws 0 nh vb 0 refs #[]
  else
    match refs with
    | .row prod0 :: _ =>
      match k.d.reads.find? (·.covers prod0) with
      | none => none
      | some rec =>
        if rec.table == sha && rec.n == n && rec.lo == lo && rec.k == kk && rec.prod0 == prod0 && kk ≤ vb &&
            (List.range (2 ^ lo)).all (fun l => loM.getD l ([], 0) == ([rec.loTop + l], 0)) then
          match prodRows k rec hiM nh (kk * nh) 0 refs with
          | none => none
          | some refs' => outRows k rec ws kk nh vb 0 refs' #[]
        else none
    | _ => none

/-- A generated read layout's output rows, at its output columns: bit `j < kk` is `Σ_h product(j, h) · [const]`; any other
bit is the same in every word of the table, its row `[const] · [const]` when set and a zero row when not. -/
def genOuts (k : Chk) (ws : Array Nat) (rec : ReadRec) (vb kk nh : Nat) : Bool :=
  (List.range vb).all fun j =>
    let col := k.d.outCols.getD j 0
    if j < kk then k.rowIs col ((List.range nh).map (rec.prod0 + j * nh + ·), [k.d.const])
    else (List.range ws.size).all (fun i => (ws.getD i 0).testBit j == (ws.getD 0 0).testBit j) &&
      k.rowIs col (if (ws.getD 0 0).testBit j then ([k.d.const], [k.d.const]) else ([], []))

/-- A generated read layout's rows (`table/v2`, the callee of a placed read): its inputs are columns `0 … n − 1`, self rows;
its refs are its decoders over them and its product rows, the side from its record; its outputs are `genOuts`; its
constant copies itself. -/
def genOk (words : String → Option (Array Nat)) (k : Chk) (sha : String) (n vb lo : Nat) : Bool :=
  let d := k.d
  let ws := (words sha).getD #[]
  let kk := Layout.kOf sha vb (Layout.tableInfo ws)
  let nh := 2 ^ (n - lo)
  d.inCols == Array.range n && d.outCols.size == vb && k.oneAt d.const && ws.size == 2 ^ n && 2 ≤ lo && lo < n &&
    (List.range n).all (fun i => k.rowIs i ([i], [i])) &&
    match chkDecode k ((Array.range lo).map fun i => ([i], 0)) d.items.toList with
    | none => false
    | some (loM, refs1) =>
      match chkDecode k ((Array.range (n - lo)).map fun i => ([lo + i], 0)) refs1 with
      | none => false
      | some (hiM, refs2) =>
        if kk = 0 then refs2.isEmpty && genOuts k ws default vb 0 nh
        else
          match refs2 with
          | .row prod0 :: _ =>
            match d.reads.find? (·.covers prod0) with
            | none => false
            | some rec =>
              rec.table == sha && rec.n == n && rec.lo == lo && rec.k == kk && rec.prod0 == prod0 && kk ≤ vb &&
                (List.range (2 ^ lo)).all (fun l => loM.getD l ([], 0) == ([rec.loTop + l], 0)) &&
                (match prodRows k rec hiM nh (kk * nh) 0 refs2 with
                 | some [] => true
                 | _ => false) && genOuts k ws rec vb kk nh
          | _ => false

set_option maxRecDepth 4096 in
/-- Walk the items of type `t` under layout `l`, its call entries from `c` on and its read entries from `r` on, against
`refs`, from the wires `wire`. `jl` is the layout whose items these are; an inlined callee's layout is listed before it,
which bounds the walk. `top` marks the layout's own items, the only place a placed call sits. `words` holds the tables'
words by SHA-512. Returns the wires and the refs left. -/
def walk (types : String → Option CircuitType) (words : String → Option (Array Nat)) (ls : Array Layout)
    (done : Array D) (unit j : Nat) (k : Chk) (jl : Nat) (top : Bool) (t : CircuitType) (l : Layout)
    (items : List Item) (c r : Nat) (wire : Array CF) (refs : List Ref) : Option (Array CF × List Ref) :=
  match items with
  | [] => some (wire, refs)
  | .and fa fb :: rest =>
    match refs with
    | .row col :: refs' =>
      if fa.c ≤ 1 && fb.c ≤ 1 &&
          k.rowIs col (resolve k.d.const (ent (formOf wire fa)), resolve k.d.const (ent (formOf wire fb))) then
        walk types words ls done unit j k jl top t l rest c r (wire.push ([col], 0)) refs'
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
            match walk types words ls done unit j k e.layout false u (ls.getD e.layout default) u.items.toList 0 0
                (args.map (formOf wire)) refs with
            | some (w', refs') =>
              walk types words ls done unit j k jl top t l rest (c + 1) r (wire ++ u.outputs.map (formOf w')) refs'
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
            walk types words ls done unit j k jl top t l rest (c + 1) r
              (wire ++ sub.outCols.map fun col => ([base + col], 0)) refs'
          else none
        | _ => none
  | .read kk idx :: rest =>
    match t.tables[kk]? with
    | none => none
    | some (sha, n, vb) =>
      let e := l.reads.getD r default
      match e.placed with
      | some (jj, _) =>
        match refs with
        | .seg jj' base ex :: refs' =>
          let sub := done.getD jj default
          match (ls.getD jj default).own with
          | .gen sha' n' vb' _ =>
            if top && jj' == jj && jj < j && jj != unit && sha' == sha && n' == n && vb' == vb && ex.isEmpty &&
                idx.size == n && idx.all (·.c ≤ 1) && sub.outCols.size == vb && placedOk k sub base [] wire idx then
              walk types words ls done unit j k jl top t l rest c (r + 1)
                (wire ++ sub.outCols.map fun col => ([base + col], 0)) refs'
            else none
          | _ => none
        | _ => none
      | none =>
        let ws := (words sha).getD #[]
        if idx.size == n && idx.all (·.c ≤ 1) && 2 ≤ e.lo && e.lo < n && ws.size == 2 ^ n then
          let fs := idx.map (formOf wire)
          match chkDecode k (fs.extract 0 e.lo) refs with
          | none => none
          | some (loM, refs1) =>
            match chkDecode k (fs.extract e.lo n) refs1 with
            | none => none
            | some (hiM, refs2) =>
              match readOk k ws sha n e.lo vb loM hiM refs2 with
              | some (outs, refs3) => walk types words ls done unit j k jl top t l rest c (r + 1) (wire ++ outs) refs3
              | none => none
        else none
  | _ :: _ => none
termination_by (jl, items.length)
decreasing_by
  all_goals first
    | exact Prod.Lex.right _ (by simp)
    | exact Prod.Lex.left _ _ _h

/-- **Layout `j`'s rows pass the check.** -/
def checkLayout (types : String → Option CircuitType) (words : String → Option (Array Nat)) (ls : Array Layout)
    (done : Array D) (unit j : Nat) : Bool :=
  let l := ls.getD j default
  let d := done.getD j default
  let k : Chk := ⟨d, j == unit⟩
  match l.own, l.type.bind types with
  | .derive, some t =>
    t.outputs.size == t.outBits && t.outputs.all (·.c ≤ 1) && d.inCols.size == t.inBits &&
      d.outCols.size == t.outputs.size && k.oneAt d.const &&
      (List.range t.inBits).all (fun w => k.rowIs (d.inCols.getD w 0) ([d.inCols.getD w 0], [d.inCols.getD w 0])) &&
      match walk types words ls done unit j k j true t l t.items.toList 0 0
          ((Array.range t.inBits).map fun w => ([d.inCols.getD w 0], 0)) d.items.toList with
      | some (wire, []) => (List.range t.outputs.size).all fun o =>
          k.rowIs (d.outCols.getD o 0) (resolve d.const (ent (formOf wire (t.outputs.getD o default))), [d.const])
      | _ => false
  | .gen sha n vb lo, none => j != unit && genOk words k sha n vb lo
  | _, _ => false

/-! ## The unit's logical order -/

/-- The columns the unit's row `c` reads as the block holds it: none for a Δ constant copy (it reads only `one`), a Δ
binding's source, else the row's columns as the verifier folds it (a product row's with its table-direct side). -/
def readsOf (words : String → Option (Array Nat)) (u : D) (c : Nat) : List Nat :=
  match u.delta.find? (·.1 == c) with
  | some (_, none) => []
  | some (_, some s) => s
  | none =>
    match fullRow words u c with
    | some (a, b) => a ++ b
    | none => []

/-- One pass over the order: each column is one of the unit's rows, none repeats, and each row after the `nIn` inputs
reads only columns before it. -/
def orderOk (words : String → Option (Array Nat)) (u : D) (nIn : Nat) : List Nat → Nat → Std.HashSet Nat → Bool
  | [], _, _ => true
  | c :: rest, ℓ, seen =>
    c < u.rows.size && !seen.contains c && (decide (ℓ < nIn) || (readsOf words u c).all seen.contains) &&
      orderOk words u nIn rest (ℓ + 1) (seen.insert c)

/-- The columns of a list, as a set. -/
def colSet (o : List Nat) : Std.HashSet Nat := o.foldl (fun s c => s.insert c) {}

/-- Every row outside the order is a zero row, and every Δ entry and every read's product row sits in the order. -/
def outsideOk (u : D) (o : List Nat) : Bool :=
  let s := colSet o
  (List.range u.rows.size).all (fun c => s.contains c || u.rows[c]? == some ([], [])) &&
    u.delta.all (fun e => s.contains e.1) &&
    u.reads.all fun r => (List.range (r.k * 2 ^ (r.n - r.lo))).all fun x => s.contains (r.prod0 + x)

/-- **The unit's rows are in logical order**: `order` lists each of its rows once, the inputs first; each row after them
reads only earlier columns (and `one`); every other row is a zero row. -/
def orderChecked (words : String → Option (Array Nat)) (done : Array D) (unit : Nat) : Bool :=
  let u := done.getD unit default
  orderOk words u u.inCols.size (order done u) 0 {} && outsideOk u (order done u)

/-! ## The unit's parts -/

/-- Every column row `r` reads is below `n`. -/
def rowWithin (r : List Nat × List Nat) (n : Nat) : Bool := r.1.all (· < n) && r.2.all (· < n)

/-- No column repeats in `l`, nor is one of `seen`. -/
def distinctFrom : List Nat → Std.HashSet Nat → Bool
  | [], _ => true
  | c :: rest, seen => !seen.contains c && distinctFrom rest (seen.insert c)

/-- Column `x` is one of a part's rows. -/
def inPart (done : Array D) (u : D) (x : Nat) : Bool :=
  u.parts.any fun p => p.2 ≤ x && x < p.2 + (done.getD p.1 default).size

/-- **Part `(j, b)` of the unit `u`, placed**: its callee `done[j]`, an earlier layout whose rows are its size, sits at `b`,
past the unit's own rows and inside its rows. Each callee row reads only the callee's rows, and sits at `b + c`, shifted,
but for the callee's constant, which is Δ's copy of `one`, and its bound inputs, which are Δ's bindings. -/
def partOk (done : Array D) (unit : Nat) (u : D) (j b : Nat) : Bool :=
  let sub := done.getD j default
  j < unit && sub.rows.size == sub.size && u.size ≤ b && b + sub.size ≤ u.rows.size &&
    (List.range sub.size).all fun c =>
      rowWithin (sub.rows.getD c ([], [])) sub.size &&
        if c == sub.const then u.delta.find? (·.1 == b + c) == some (b + c, none)
        else match u.delta.find? (·.1 == b + c) with
          | some (_, some _) => sub.inCols.contains c
          | some (_, none) => false
          | none => u.rows[b + c]? == some (shiftRow b (sub.rows.getD c ([], [])))

/-- A Δ entry is the unit's constant, a part's constant, or a part's bound input. -/
def deltaEntryOk (done : Array D) (u : D) (e : Nat × Option (List Nat)) : Bool :=
  (e.1 == u.const && e.2.isNone) || u.parts.any fun p =>
    (e.1 == p.2 + (done.getD p.1 default).const && e.2.isNone) ||
      (e.2.isSome && (done.getD p.1 default).inCols.any (p.2 + · == e.1))

/-- **Part `(j, b)`'s reads**: at each callee row `c`, the unit's read covering `b + c` is the callee's read covering `c`,
shifted, or none when the callee has none. A covered callee row is a product row `a · []`, whose table-direct side is the
read's. Each callee read's low minterms are callee rows. -/
def partReadsOk (done : Array D) (u : D) (j b : Nat) : Bool :=
  let sub := done.getD j default
  sub.reads.all (fun r => decide (r.loTop + 2 ^ r.lo ≤ sub.size)) &&
    (List.range sub.size).all fun c =>
      u.reads.find? (·.covers (b + c)) == (sub.reads.find? (·.covers c)).map (·.shift b) &&
        match sub.reads.find? (·.covers c) with
        | some _ => (sub.rows.getD c ([], [])).2.isEmpty
        | none => true

/-- **The unit's parts, as its template statement reads them** (`Typed.templateOf`): each part is placed (`partOk`), and
the parts follow one another by base; Δ has one entry per row, each the unit's constant, a part's constant or a part's
bound input; each of the unit's own rows reads only its own and its parts' rows. The unit's constant row is
`[const]·[const]`, so that its root net's constant reads nothing but itself, and its order lists only own and part
columns. Each part's rows are read as its callee's reads read them (`partReadsOk`). -/
def partsChecked (done : Array D) (unit : Nat) : Bool :=
  let u := done.getD unit default
  u.parts.all (fun p => partOk done unit u p.1 p.2) &&
    (u.parts.toList.zip u.parts.toList.tail).all (fun pq => decide (pq.1.2 + (done.getD pq.1.1 default).size ≤ pq.2.2)) &&
    distinctFrom (u.delta.toList.map (·.1)) {} && u.delta.all (deltaEntryOk done u) &&
    ((List.range u.size).all fun c =>
      ((u.rows.getD c ([], [])).1 ++ (u.rows.getD c ([], [])).2).all fun x => decide (x < u.size) || inPart done u x) &&
    u.rows.getD u.const ([], []) == ([u.const], [u.const]) &&
    ((order done u).all fun c => decide (c < u.size) || inPart done u c) &&
    u.parts.all fun p => partReadsOk done u p.1 p.2

/-- **The rows the verifier uses**: `deriveAll`'s, once every layout's pass the check, the unit's are in logical order, and
its parts are as its template reads them. -/
def deriveChecked (types : String → Option CircuitType) (ls : Array Layout) (info : String → Option TableInfo)
    (words : String → Option (Array Nat)) (unit : Nat) : Except String (Array D) := do
  let done ← deriveAll types ls info words unit
  match (List.range ls.size).find? fun j => !checkLayout types words ls done unit j with
  | some j => throw s!"derive: layout {j}'s rows fail the check"
  | none =>
    if !orderChecked words done unit then throw "derive: the unit's rows are not in logical order"
    else if !partsChecked done unit then throw "derive: the unit's parts are not placed as its template reads them"
    else return done

/-- `flock-rows --archive FILE --part check`: `ok` when every layout's rows pass the check. -/
def archiveCheck (j : Json) : V String := do
  let (types, ls, info, words, unit, _) ← load j
  let _ ← deriveChecked types ls info words unit
  return "ok\n"

/-- The part of an archive's derivation that `part` names: `unit`, `delta`, `logical` or a layout's digest, from the rows
`deriveChecked` accepts, the ones the pins cover. -/
def archivePart (j : Json) (part : String) : V String := do
  let (types, ls, info, words, unit, ds) ← load j
  let done ← deriveChecked types ls info words unit
  let u := done.getD unit default
  if part == "unit" then return netlist u ((ds.getD unit "").take 16).toString
  if part == "delta" then return deltaText ls ds u
  if part == "logical" then return logicalText done u
  match ds.findIdx? (· == part) with
  | some i => return netlist (done.getD i default) (part.take 16).toString
  | none => throw s!"archive: no layout {part.take 16}"

end Flock.DeriveAll
