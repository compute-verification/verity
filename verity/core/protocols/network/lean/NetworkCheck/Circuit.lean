import Flock.Program
import NciCheck.Desc
import SampledProofs.Wires
import SampledProofs.PyJson

/-!
# A compute audit's circuit record, checked against its Program

A tied audit's period parameters carry its window's circuit record (`verity/window-circuit/v1`, the stated form:
`interface.circuit_record`'s templates, cap, wires, cuts and registered reads), a certificate that its wires are
acyclic (`verity/wire-order/v1`) and the verifier's pins. `check` reads them against the audit's Program, at the size
of their description (AGENTS.md "Design rules"), never per unit:

* **The population** (`populationOk`): the Program's root is the population program the record names, its template
  nodes the record's templates in order, each a batch of the record's units, each argument one whole root input node of
  its own (a batch of `Input<w>` gates) mapped over axis 0 at its parameter's leaves, no input node read twice, every
  input gate a template's parameter, and the root returning every template node's outputs, in order. The Program is
  laid out as C-Flock's decoder lays one out, within its resolution's walk (`NciCheck.Desc`), and topological.
* **The wires** (`SampledProofs.Wires.check`), with the templates' rows read off their Definitions (`templateOf`,
  `interface.layout`); each wire's reader row is its writer's output field bit for bit, each side in leaves of one
  width, the widths equal or one of them a Boolean-profile port's 1 (`leavesOk`). A writer's leaf is a gate its
  template computes or an input it passes through: a return names a parameter or a node, never a literal.
* **The declarations** (`declRects`): the registered parameters, tables, reads, table reads and openings well formed
  for the templates (`interface.check_inputs`), and no row filled twice by a wire or a declaration (`apart`). A record
  that lists its members, declares ingress, or registers values or formats beside its reads is refused.
* **The order** (`orderWhy`, `ordered`): a rank per unit, affine on each segment of a template's units, below along
  every wire: each wire's writer and reader sides lie in their segments without carries, so each side's rank is affine
  in the wire's digits, and the box maximum of rank(writer) − rank(reader) is below zero.

On acceptance it gives the record's `Counts`, in bits: every unit's input rows (`inputs`, the Program's input gates),
the rows its wires fill (`wired`), the rows its registered reads, table reads and openings of a held parameter fill
(`registered`) and the rest (`free`), as `interface.input_count` counts them.
-/

namespace NetworkCheck.Period

/-- Whether `j` is the string `s`: `Json`'s `==` on strings, by a match a proof unfolds. -/
def isStr (j : Lean.Json) (s : String) : Bool := match j with
  | .str x => x == s
  | _ => false

/-- A type's bits: its leaves' widths summed. -/
def bitsOf (t : Flock.Program.Ty) : Nat := t.widths.foldl (· + ·) 0

/-- The root's template nodes, in order: those calling a composite. -/
def templateNodes (P : Flock.Program.Prog) : List Nat :=
  let nodes := (P.defs[P.root]!).body.nodes
  (List.range nodes.size).filter fun k => match P.defs[nodes[k]!.fn]? with
    | some d => !d.prim
    | none => false

/-- The root node a template's argument reads when the argument is that input node's leaves in order from its first
(one affine run), as the generator form binds each parameter to an input array of its own. -/
def inputArg (P : Flock.Program.Prog) (a : Flock.Program.Arg) : Option Nat :=
  match a.seq.parts.toList with
  | [.affine false k 0 _] => match (P.defs[P.root]!).body.nodes[k]? with
    | some n => if NciCheck.Desc.isInputNode P n then some k else none
    | none => none
  | _ => none

/-- **A Program's input gates**, in bits: each of its root's input nodes (`Flock.Program.inputId?`, of either binding
time) gives its members times its width, so a `Value<w>` leaf counts `w`. The decoder refuses an input node outside the
root's body, so these are all of them. For a Boolean Program this is `#freeInputs` at no ingress; the sampled circuit
lays a wider leaf out as one gate, so then it is at least that. Either way it is at least `#(freeInputs \ reg)` at every
`reg` and horizon. It reads the root's nodes, not the unrolled circuit. -/
def inputGates (prog : Flock.Program.Prog) : Nat :=
  (((prog.defs[prog.root]?).map (·.body.nodes)).getD #[]).foldl (fun n x => n + match prog.defs[x.fn]? with
    | some d => if d.prim then ((Flock.Program.inputId? d.key).map (x.members * ·.2)).getD 0 else 0
    | none => 0) 0

end NetworkCheck.Period

namespace NetworkCheck.Circuit

open Lean (Json toJson)
open Flock.Program (Prog Defn Ty)
open NetworkCheck.Period (isStr bitsOf templateNodes inputArg inputGates)

abbrev Wire := SampledProofs.Wires.Wire
abbrev Wiring := SampledProofs.Wires.Wiring
abbrev Template := SampledProofs.Wires.Template

def FORMAT : String := "verity/window-circuit/v1"
def ORDER : String := "verity/wire-order/v1"

/-! ## The templates' rows, off their Definitions -/

/-- A return's fields: a tuple's or a record's, else the return itself (`interface._fields`). -/
def fieldsOf : Ty → List Ty
  | .tuple fs => fs.toList
  | .record fs => fs.toList.map (·.2)
  | t => [t]

/-- A parameter's outer length: its outermost array's, 1 when it is no array. -/
def outerOf : Ty → Nat
  | .array n _ => n
  | _ => 1

/-- The one width of a type's leaves, where they have one. -/
def widthOf (t : Ty) : Option Nat := Flock.Program.Widths.homo t

/-- Output field `j`'s row name. -/
def outName (j : Nat) : String := s!"o{j}"

/-- An input row: whole u16 words of leaves of one width. -/
def rowOk (t : Ty) : Bool := 0 < bitsOf t && bitsOf t % 16 == 0 && (widthOf t).isSome

/-- An output row: leaves of one width, in whole u16 words, a field of fewer than 16 bits one word. -/
def outOk (t : Ty) : Bool := 0 < bitsOf t && (widthOf t).isSome

/-- Template `d`'s rows with `n` units (`interface.layout`): a row a parameter, its words and outer length, and an
output row `o<j>` a return field; none unless each is a row, `n` is at least 1 and no parameter is named like an output
row. -/
def templateOf (d : Defn) (n : Nat) : Option Template :=
  let outs := (fieldsOf d.ret).zipIdx.map fun (t, j) => (outName j, (bitsOf t + 15) / 16)
  if 1 ≤ n && d.params.all (rowOk ·.2) && (fieldsOf d.ret).all outOk &&
      !d.params.any (fun p => outs.any (·.1 == p.1)) then
    some { units := n, rows := d.params.toList.map fun (p, t) => (p, ⟨bitsOf t / 16, outerOf t⟩), outs }
  else none

/-- Root template node `k`'s Definition. -/
def templateDef (P : Prog) (k : Nat) : Defn := P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!

/-! ## The record -/

/-- A registered table (`interface.Table`): `rows` rows of `words` leaves of parameter `param` from its leaf `offset`. -/
structure Table where
  name : String
  param : String
  offset : Nat
  words : Nat
  rows : Nat
  deriving Repr, BEq

/-- A registered read (`interface.Read`): the rows of reader units it covers, as a wire's pattern whose `writer` is the
parameter `param` and whose `out` is empty, each reader row reading the parameter's leaves from the pattern's source on. -/
structure Read where
  param : String
  pat : Wire
  deriving Repr, BEq

/-- A table read (`interface.TableRead`): units `[at, at + units)` of `reader` read `port` as table `table`'s root or
domain id (`of`). -/
structure TableRead where
  reader : String
  port : String
  table : String
  of : String
  start : Nat
  units : Nat
  deriving Repr, BEq

/-- A table opening (`interface.TableOpening`): units `[at, at + units)` of `reader` open table `table`'s row at their
`key` port, `ports` the opening and `verdict` the output row whose bit says it opens. -/
structure TableOpening where
  reader : String
  table : String
  key : String
  ports : List String
  verdict : String
  start : Nat
  units : Nat
  deriving Repr, BEq

/-- What the registration fixes that units read (`interface.Registered`), without `values` or `formats`. -/
structure Registered where
  params : List (String × Nat) := []
  tables : List Table := []
  reads : List Read := []
  roots : List TableRead := []
  openings : List TableOpening := []
  deriving Repr

/-- A stated circuit record, its templates' rows read off the Program: the templates by descriptor id with their units,
in the Program's order, the cap, the wiring (templates, cuts and wires, `SampledProofs.Wires.Wiring`) and the
registered reads. -/
structure Stated where
  templates : List (String × Nat)
  cap : Nat
  wiring : Wiring
  registered : Registered
  deriving Repr

open SampledProofs.Wires (field fields arrOf strOf natOf intOf natAt sameKeys keysOf)

/-- A list of integers each at least 1, `what` naming it. -/
def radixOf (what : String) (j : Json) : Except String (List Nat) := do
  let xs ← (← arrOf what j).mapM (intOf what)
  if xs.any (· < 1) then throw s!"{what}: each at least 1"
  pure (xs.map Int.toNat)

def readOf (i : Nat) (j : Json) : Except String Read := do
  let what := s!"registered read {i}"
  let base := ["reader", "port", "param", "at", "radix", "base", "step"]
  if ((keysOf j).getD []).contains "cut" then sameKeys j ("cut" :: base) what else sameKeys j base what
  let s := fun (k : String) => do strOf s!"{what}: {k}" (← field j k)
  let radix ← (← arrOf s!"{what}: radix" (← field j "radix")).mapM (natOf s!"{what}: a radix")
  let step ← (← arrOf s!"{what}: step" (← field j "step")).mapM (intOf s!"{what}: a step")
  if radix.isEmpty || radix.length != step.length || radix.any (· < 1) then
    throw s!"{what}: at least one digit of radix 1 or more, and one step a digit"
  let (cut0, cuts) ← match j.getObjVal? "cut" with
    | .ok cj => do
      match ← (← arrOf s!"{what}: cut" cj).mapM (natOf s!"{what}: a cut") with
      | [a, b] => if 1 ≤ b then pure (a, b) else throw s!"{what}: its cut is [c0, C], C at least 1"
      | _ => throw s!"{what}: its cut is [c0, C]"
    | .error _ => pure (0, 1)
  let param ← s "param"
  pure { param, pat := { reader := ← s "reader", port := ← s "port", writer := param, out := "",
                         start := ← natAt j "at", radix, base := ← intOf s!"{what}: base" (← field j "base"), step,
                         cut0, cuts } }

def tableOf (i : Nat) (j : Json) : Except String Table := do
  let what := s!"registered table {i}"
  sameKeys j ["name", "param", "offset", "words", "rows"] what
  pure { name := ← strOf s!"{what}: name" (← field j "name"), param := ← strOf s!"{what}: param" (← field j "param"),
         offset := ← natAt j "offset", words := ← natAt j "words", rows := ← natAt j "rows" }

def tableReadOf (i : Nat) (j : Json) : Except String TableRead := do
  let what := s!"registered table read {i}"
  sameKeys j ["reader", "port", "table", "of", "at", "units"] what
  let s := fun (k : String) => do strOf s!"{what}: {k}" (← field j k)
  pure { reader := ← s "reader", port := ← s "port", table := ← s "table", of := ← s "of", start := ← natAt j "at",
         units := ← natAt j "units" }

def openingOf (i : Nat) (j : Json) : Except String TableOpening := do
  let what := s!"registered opening {i}"
  sameKeys j ["reader", "table", "key", "ports", "verdict", "at", "units"] what
  let s := fun (k : String) => do strOf s!"{what}: {k}" (← field j k)
  pure { reader := ← s "reader", table := ← s "table", key := ← s "key",
         ports := ← (← arrOf s!"{what}: ports" (← field j "ports")).mapM (strOf s!"{what}: a port"),
         verdict := ← s "verdict", start := ← natAt j "at", units := ← natAt j "units" }

/-- The record's registered reads (`interface.Registered.of`). A record that registers values or formats is refused:
the period's count takes every registered read as 16-bit leaves of its parameter, and no value as the Program's. -/
def registeredOf (j : Json) : Except String Registered := do
  let ks := (keysOf j).getD []
  if ks.contains "values" || ks.contains "formats" then
    throw "registered: values or formats, which the period refuses: it reads registered parameters as 16-bit leaves \
      that no Program parameter takes"
  sameKeys j ["params", "tables", "reads", "roots", "openings"] "registered"
  let list := fun (k : String) => do arrOf s!"registered: {k}" (← field j k)
  pure { params := ← (← fields (← field j "params")).mapM fun (p, v) => do pure (p, ← natOf s!"param {p}'s leaves" v),
         tables := ← (← list "tables").zipIdx.mapM fun (t, i) => tableOf i t,
         reads := ← (← list "reads").zipIdx.mapM fun (r, i) => readOf i r,
         roots := ← (← list "roots").zipIdx.mapM fun (r, i) => tableReadOf i r,
         openings := ← (← list "openings").zipIdx.mapM fun (o, i) => openingOf i o }

/-- **The record**, read against Program `P`: a stated record (`format`, its `program` `P`'s SHA-512) whose templates
are `P`'s root template nodes' Definitions in order (`templateNodes`), each a template of rows (`templateOf`), named
once, with its cap, wires, cuts and registered reads. A record that lists its members (`members`, `window_ingress`) or
declares ingress is refused: the period counts a stated record's rows, and no wired gate is an ingress gate. -/
def recordOf (P : Prog) (j : Json) : Except String Stated := do
  let ks := (keysOf j).getD []
  if ks.contains "members" || ks.contains "window_ingress" then
    throw "the circuit lists its members: the period reads a stated record (its templates, cap and wires)"
  match j.getObjVal? "ingress" with
  | .ok (.arr xs) => if !xs.isEmpty then throw "the circuit declares ingress, which the period refuses: a wired gate \
      is no ingress gate"
  | .ok _ => throw "the circuit's ingress is not a list"
  | .error _ => pure ()
  unless isStr (j.getObjValD "format") FORMAT do throw s!"the circuit's format is not {FORMAT}"
  unless isStr (j.getObjValD "program") (Flock.hexOf P.sha512) do
    throw "the circuit's program is not the audit's Program's SHA-512"
  let cap ← natAt j "cap"
  let named ← (← arrOf "templates" (← field j "templates")).mapM fun t => do
    match ← arrOf "a template" t with
    | [a, b] => pure (← strOf "a template's descriptor id" a, ← natOf "a template's units" b)
    | _ => throw "a template is [descriptor id, units]"
  unless decide (named.map (·.1)).Nodup do throw "the circuit names a template twice"
  let tks := templateNodes P
  unless tks.length == named.length do
    throw s!"the circuit names {named.length} templates, the Program's root {tks.length}"
  let templates ← (tks.zip named).mapM fun (k, tid, n) => do
    let d := templateDef P k
    unless d.key == tid do throw s!"the circuit's template {tid} is the Program's {d.key}"
    match templateOf d n with
    | some t => pure (tid, t)
    | none => throw s!"template {tid}: not {n} ≥ 1 units of whole u16 words of leaves of one width a parameter and \
        leaves of one width a return field, no parameter named like an output row"
  let wires ← (← arrOf "wires" (← field j "wires")).mapM SampledProofs.Wires.Wire.ofJson
  let cuts ← match j.getObjVal? "cuts" with
    | .ok cj => do
      (← fields cj).mapM fun (tid, v) => do
        pure (tid, ← (← fields v).mapM fun (p, k) => do pure (p, ← natOf "a cut" k))
    | .error _ => pure []
  let registered ← match j.getObjVal? "registered" with
    | .ok rj => registeredOf rj
    | .error _ => pure {}
  pure { templates := named, cap, wiring := { templates, cuts, wires }, registered }

/-! ## The population -/

/-- Root template node `k`'s arguments: each one whole root input node of its own, a batch of `Input<w>` gates of its
parameter's width, mapped over axis 0 at its parameter's leaves, so member `m` reads leaves `[m·L, (m+1)·L)` of it. -/
def argsOk (P : Prog) (k : Nat) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let x := nodes[k]!
  let d := templateDef P k
  x.form == .batch && x.args.size == d.params.size && x.axes.size == x.args.size &&
    (List.range x.args.size).all fun i =>
      let a := x.args[i]!
      let t := (d.params[i]!).2
      x.axes[i]! == some 0 && a.dims == some (x.n, t.leaves) && a.seq.count == x.n * t.leaves &&
        match inputArg P a with
        | some j => let y := nodes[j]!
          y.form == .batch && y.n == x.n * t.leaves &&
            (Flock.Program.inputWidth? (P.defs[y.fn]!).key) == widthOf t
        | none => false

/-- The root returns every template node's outputs, whole and in order: one affine run a template node. -/
def returnsAll (P : Prog) : Bool :=
  let root := P.defs[P.root]!
  let ks := templateNodes P
  root.body.ret.parts.size == ks.length && (root.body.ret.parts.toList.zip ks).all fun (q, k) => match q with
    | .affine false k' 0 c => k' == k && c == (root.body.nodes[k]!).n * (templateDef P k).outLeaves
    | _ => false

/-- **The population**: `P`'s root is the population program of record `r`'s templates. Its template nodes are the
record's templates with their units, each a batch whose arguments are whole root input nodes of their own (`argsOk`),
no input node an argument twice; every root node is an input node or a template node, and its input gates are the
templates' parameters' bits (`inputGates`), so every input gate is a template's parameter's leaf; the root returns every
template node's outputs (`returnsAll`), so every unit's output is a Program output. The Program is laid out as C-Flock's
decoder lays one out and within its resolution's walk (`NciCheck.Desc`'s `rootLaidOut`, `inputsAtRoot` and
`walkable`), and topological (`Prog.orderViolation`). -/
def populationOk (P : Prog) (r : Stated) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let ks := templateNodes P
  let reads := ks.flatMap fun k => (nodes[k]!).args.toList.map (inputArg P ·)
  ks.length == r.templates.length &&
    (ks.zip r.templates).all (fun (k, tid, n) => (templateDef P k).key == tid && (nodes[k]!).n == n) &&
    ks.all (argsOk P) && decide reads.Nodup &&
    (List.range nodes.size).all (fun k => NciCheck.Desc.isInputNode P nodes[k]! || ks.contains k) &&
    inputGates P == (ks.map fun k => (nodes[k]!).n * ((templateDef P k).params.toList.map (bitsOf ·.2)).sum).sum &&
    returnsAll P &&
    NciCheck.Desc.rootLaidOut P && NciCheck.Desc.inputsAtRoot P && NciCheck.Desc.walkable P &&
    P.orderViolation.isNone

/-! ## Wired rows, leaf by leaf -/

/-- The root template node of record `r`'s template `tid`. -/
def nodeOf (P : Prog) (r : Stated) (tid : String) : Option Nat :=
  ((templateNodes P).zip r.templates).findSome? fun (k, t, _) => if t == tid then some k else none

/-- Output row `o`'s field of a return: its type, its first leaf and its index. -/
def fieldOf (ret : Ty) (o : String) : Option (Ty × Nat × Nat) :=
  let fs := fieldsOf ret
  (fs.zipIdx.find? (outName ·.2 == o)).map fun (t, j) => (t, ((fs.take j).map (·.leaves)).sum, j)

/-- Each wire's reader row is its writer's output field bit for bit: the reader's parameter, cut `k` ways, has `k`
times the field's bits, each side's leaves of one width, the two widths equal (leaf for leaf) or one of them 1, a
Boolean-profile port's, whose row is the same bits in the same order (`interface._width`). -/
def leavesOk (P : Prog) (r : Stated) : Bool :=
  r.wiring.wires.all fun w => match nodeOf P r w.reader, nodeOf P r w.writer with
    | some kr, some kw => match (templateDef P kr).params.find? (·.1 == w.port), fieldOf (templateDef P kw).ret w.out with
      | some (_, tp), some (to, _, _) => match widthOf tp, widthOf to with
        | some a, some b => (a == b || a == 1 || b == 1) && bitsOf tp == r.wiring.cutsOf w.reader w.port * bitsOf to
        | _, _ => false
      | _, _ => false
    | _, _ => false

/-! ## The declarations -/

/-- The rows a wire or declaration fills: reader template `reader`'s port `port`, units `[lo, hi)`, rows `[r0, r1)`. -/
structure Rect where
  reader : String
  port : String
  lo : Nat
  hi : Nat
  r0 : Nat
  r1 : Nat
  deriving Repr, BEq

def Rect.disjoint (a b : Rect) : Bool :=
  a.hi ≤ b.lo || b.hi ≤ a.lo || a.r1 ≤ b.r0 || b.r1 ≤ a.r0 || a.reader != b.reader || a.port != b.port

/-- A wire's rows (or a registered read's: its pattern's). -/
def wireRect (w : Wire) : Rect := ⟨w.reader, w.port, w.start, w.start + w.readerUnits, w.cut0, w.cut0 + w.cuts⟩

/-- Each of `rs` disjoint from those after it and from `earlier`. -/
def apart : List Rect → List Rect → Bool
  | [], _ => true
  | r :: rs, earlier => earlier.all r.disjoint && apart rs (r :: earlier)

/-- Template `tid`'s port `p`'s words and cut: an input row's, or, where `outs`, an output row's with cut 1. -/
def portOf (c : Wiring) (tid p : String) (outs : Bool) : Option (Nat × Nat) :=
  match c.templates.lookup tid with
  | some t => match t.rows.lookup p with
    | some q => some (q.words, c.cutsOf tid p)
    | none => if outs then (t.outs.lookup p).map (·, 1) else none
  | none => none

/-- Units `[at, at + n)`, `n ≥ 1`, among template `tid`'s. -/
def unitsOk (c : Wiring) (tid : String) (start n : Nat) : Bool :=
  1 ≤ n && match c.templates.lookup tid with
    | some t => start + n ≤ t.units
    | none => false

/-- **The declarations' rows** (`interface.check_inputs`): the registered parameters of at least one leaf each; each
table a new name over rows of a declared parameter's leaves; each read of a template's input or output port, its rows
among the port's (its last digit the cut, which divides its reader rows), its units the template's and its leaves the
parameter's; each table read a declared table's root or domain id read whole by a 64-byte port; each opening a
declared table's at a key port, its verdict an output and its ports input ports. The rows each fills, in order. -/
def declRects (c : Wiring) (reg : Registered) : Except String (List Rect) := do
  unless reg.params.all (0 < ·.2) do throw "registered: each parameter at least one leaf"
  let tables ← reg.tables.zipIdx.foldlM (init := ([] : List Table)) fun seen (t, i) => do
    let ok := match reg.params.lookup t.param with
      | some n => 1 ≤ t.words && 1 ≤ t.rows && t.offset + t.words * t.rows ≤ n
      | none => false
    unless ok && !seen.any (·.name == t.name) do
      throw s!"registered table {i}: a new name, rows of a registered parameter's leaves"
    pure (t :: seen)
  let reads ← reg.reads.zipIdx.mapM fun (r, i) => do
    let w := r.pat
    let some (words, k) := portOf c w.reader w.port true
      | throw s!"registered read {i}: names no port of a template of the circuit"
    unless w.cut0 + w.cuts ≤ k && (w.cuts == 1 || w.radix.getLast? == some w.cuts) && w.units % w.cuts == 0 do
      throw s!"registered read {i}: rows [{w.cut0}, {w.cut0 + w.cuts}) of {w.port}'s {k}, its last digit the cut, \
        which divides its reader rows"
    unless unitsOk c w.reader w.start w.readerUnits do throw s!"registered read {i}: units outside its template's"
    let ok := match reg.params.lookup r.param with
      | some n => 0 ≤ w.span.1 && w.span.2 + ((words / k : Nat) : Int) ≤ (n : Int)
      | none => false
    unless ok do throw s!"registered read {i}: leaves outside a registered parameter's"
    pure (wireRect w)
  let roots ← reg.roots.zipIdx.mapM fun (r, i) => do
    let some (words, k) := portOf c r.reader r.port false
      | throw s!"registered table read {i}: names no input port of a template of the circuit"
    unless unitsOk c r.reader r.start r.units && tables.any (·.name == r.table) && (r.of == "root" || r.of == "domain") &&
        words == 32 do
      throw s!"registered table read {i}: a declared table's root or domain id, a 64-byte port, of its template's units"
    pure (⟨r.reader, r.port, r.start, r.start + r.units, 0, k⟩ : Rect)
  let opens ← reg.openings.zipIdx.mapM fun (o, i) => do
    let verdict := match c.templates.lookup o.reader with
      | some t => (t.outs.lookup o.verdict).isSome
      | none => false
    unless unitsOk c o.reader o.start o.units && (portOf c o.reader o.key false).isSome &&
        tables.any (·.name == o.table) && verdict && !o.ports.isEmpty do
      throw s!"registered opening {i}: a declared table's row at its key, with a verdict output, of its template's units"
    o.ports.mapM fun p => match portOf c o.reader p false with
      | some (_, k) => pure (⟨o.reader, p, o.start, o.start + o.units, 0, k⟩ : Rect)
      | none => throw s!"registered opening {i}: port {p} is no input port of {o.reader}"
  pure (reads ++ roots ++ opens.flatten)

/-! ## The order -/

/-- Segment `s` of a template's units (`verity/wire-order/v1`): units `[at, at + Π radix)`, unit `at + i` with the
digits of `i` in mixed radix `radix`, slowest first, and rank `c + Σ_q r[q] · digit_q`. -/
structure Segment where
  template : String
  start : Nat
  radix : List Nat
  c : Int
  r : List Int
  deriving Repr, BEq

/-- The certificate: the segments, and per record wire its writer's and its reader's segment. -/
structure Order where
  segments : Array Segment
  wires : Array (Nat × Nat)
  deriving Repr

/-- `Π rs`. -/
def prod (rs : List Nat) : Nat := rs.foldr (· * ·) 1

/-- Each position's stride in mixed radix `rs`, slowest first: the product of the radices after it. -/
def strides : List Nat → List Nat
  | [] => []
  | _ :: rs => prod rs :: strides rs

/-- Position `q`'s digit of `b` in mixed radix `rs`. -/
def digitAt (rs : List Nat) (b q : Nat) : Nat := b / (strides rs).getD q 1 % rs.getD q 1

def Segment.size (s : Segment) : Nat := prod s.radix

def Segment.holds (s : Segment) (tid : String) (u : Nat) : Bool :=
  s.template == tid && s.start ≤ u && u < s.start + s.size

/-- Unit `u`'s rank in segment `s`: `c` plus `accum` of its digits (fastest first, as `Wire.pairs`) and `r`. -/
def Segment.rankAt (s : Segment) (u : Nat) : Int := s.c + SampledProofs.Wires.accum (s.radix.zip s.r).reverse (u - s.start)

/-- **A unit's rank**: its rank in the first segment of its template that holds it; 0 in none. -/
def rank (o : Order) (tid : String) (u : Nat) : Int :=
  match o.segments.toList.find? (·.holds tid u) with
  | some s => s.rankAt u
  | none => 0

/-- A wire digit counts when its radix is above 1 and its stride nonzero. -/
def counted (d : Nat × Int) : Bool := 1 < d.1 && d.2 != 0

/-- The segment position a stride `t` moves and its multiplier there: the position of radix above 1 whose stride is the
largest dividing `|t|`, and `t` over that stride. -/
def place (rs : List Nat) (t : Int) : Option (Nat × Int) :=
  ((List.range rs.length).find? fun q => 1 < rs.getD q 0 && t.natAbs % (strides rs).getD q 1 == 0).map
    fun q => (q, t / ((strides rs).getD q 1 : Int))

/-- The sum of `f m (radix − 1)` over the counted digits of `ds` that move position `q`, `m` each one's multiplier. -/
def reach (rs : List Nat) (ds : List (Nat × Int)) (q : Nat) (f : Int → Int → Int) : Int :=
  (ds.map fun d => if counted d then
    (match place rs d.2 with
     | some (p, m) => if p == q then f m ((d.1 : Int) - 1) else 0
     | none => 0) else 0).sum

/-- Position `q`'s least digit over the digits' box, from offset `b`. -/
def lo (rs : List Nat) (b : Nat) (ds : List (Nat × Int)) (q : Nat) : Int :=
  (digitAt rs b q : Int) + reach rs ds q fun m n => min 0 (m * n)

/-- Position `q`'s greatest digit over the digits' box, from offset `b`. -/
def hi (rs : List Nat) (b : Nat) (ds : List (Nat × Int)) (q : Nat) : Int :=
  (digitAt rs b q : Int) + reach rs ds q fun m n => max 0 (m * n)

/-- Why side `side` of wire `i`, at offset `o` with digits `ds` (radix, stride), doesn't lie in segment `si` without
carries: its offset outside the segment, a counted digit on no position, or a position whose digit leaves its radix
over the box; none when it lies there. -/
def sideWhy (i : Nat) (side : String) (si : Nat) (s : Segment) (o : Int) (ds : List (Nat × Int)) : Option String :=
  let b := o - (s.start : Int)
  if b < 0 ∨ (s.size : Int) ≤ b then some s!"wire {i}: {side} offset {b} outside segment {si}"
  else match ds.zipIdx.find? (fun (d, _) => counted d && (place s.radix d.2).isNone) with
    | some ((_, t), k) => some s!"wire {i}: {side} digit {k} (stride {t}) on no position of segment {si}"
    | none =>
      match (List.range s.radix.length).find? (fun q =>
          decide (lo s.radix b.toNat ds q < 0) || decide ((s.radix.getD q 1 : Int) - 1 < hi s.radix b.toNat ds q)) with
      | some q => some s!"wire {i}: {side} digit position {q} of segment {si} carries"
      | none => none

/-- A side's digit coefficients in its rank: `r` at the digit's position times its multiplier; 0 for a digit that
doesn't count. -/
def coeffs (s : Segment) (ds : List (Nat × Int)) : List Int :=
  ds.map fun d => if counted d then
    (match place s.radix d.2 with
     | some (q, m) => s.r.getD q 0 * m
     | none => 0) else 0

/-- The reader side's strides: each digit's reader-unit stride, the product of the faster digits' radices but the cut
digit's, which, last when the wire covers several rows, has stride 0. -/
def readerStrides (w : Wire) : List Int :=
  if w.cuts == 1 then (strides w.radix).map Int.ofNat
  else (strides w.radix.dropLast).map Int.ofNat ++ [0]

def writerDigits (w : Wire) : List (Nat × Int) := w.radix.zip w.step

def readerDigits (w : Wire) : List (Nat × Int) := w.radix.zip (readerStrides w)

/-- The box maximum of rank(writer) − rank(reader) over the wire's digits, from each side's constant and coefficients. -/
def boxMax (kw kr : Int) (aw ar : List Int) (radix : List Nat) : Int :=
  kw - kr + ((radix.zip (aw.zip ar)).map fun (n, x, y) => max 0 ((x - y) * ((n : Int) - 1))).sum

/-- Why wire `i` doesn't fall in the order: a segment index that names none, a segment of another template, a side that
doesn't lie in its segment (`sideWhy`), or a writer whose rank reaches its reader's. -/
def wireWhy (o : Order) (i : Nat) (w : Wire) : Option String :=
  match o.wires[i]? with
  | none => some s!"wire {i}: no wire pair"
  | some (sw, sr) => match o.segments[sw]?, o.segments[sr]? with
    | none, _ => some s!"wire {i}: segment {sw} is no segment"
    | _, none => some s!"wire {i}: segment {sr} is no segment"
    | some ws, some rs =>
      if ws.template != w.writer then some s!"wire {i}: writer segment {sw} is template {ws.template}, not {w.writer}"
      else if rs.template != w.reader then
        some s!"wire {i}: reader segment {sr} is template {rs.template}, not {w.reader}"
      else match sideWhy i "writer" sw ws w.base (writerDigits w) with
        | some e => some e
        | none => match sideWhy i "reader" sr rs w.start (readerDigits w) with
          | some e => some e
          | none =>
            let bw := (w.base - (ws.start : Int)).toNat
            let br := w.start - rs.start
            let d := boxMax (ws.rankAt (ws.start + bw)) (rs.rankAt (rs.start + br)) (coeffs ws (writerDigits w))
              (coeffs rs (readerDigits w)) w.radix
            if d ≤ -1 then none else some s!"wire {i}: rank(writer) - rank(reader) reaches {d} >= 0"

/-- Why segment `si` is no segment of the record's templates: its template none of them, or its units outside it. -/
def segmentWhy (ts : List (String × Template)) (si : Nat) (s : Segment) : Option String :=
  match ts.lookup s.template with
  | none => some s!"segment {si} names no template of the record"
  | some t => if s.start + s.size ≤ t.units then none
    else some s!"segment {si}: units [{s.start}, {s.start + s.size}) outside {s.template}'s {t.units}"

/-- The first two segments of one template that overlap. -/
def overlapWhy : List (Segment × Nat) → Option String
  | [] => none
  | (a, i) :: rest =>
    match rest.find? (fun (b, _) => a.template == b.template && a.start < b.start + b.size && b.start < a.start + a.size) with
    | some (_, j) => some s!"segments {i} and {j} of {a.template} overlap"
    | none => overlapWhy rest

/-- **Why the order fails**, in order: a segment of no template or outside its units, two segments of a template that
overlap, a wire pair count other than the record's wires', then wire by wire (`wireWhy`). -/
def orderWhy (c : Wiring) (o : Order) : Option String :=
  (o.segments.toList.zipIdx.findSome? fun (s, i) => segmentWhy c.templates i s) <|>
  overlapWhy o.segments.toList.zipIdx <|>
  (if o.wires.size != c.wires.length then some s!"{o.wires.size} wire pairs for {c.wires.length} record wires"
   else none) <|>
  c.wires.zipIdx.findSome? fun (w, i) => wireWhy o i w

/-- **The order holds**: every wire's writers rank below its readers (`orderWhy`). -/
def ordered (c : Wiring) (o : Order) : Bool := (orderWhy c o).isNone

/-- Segment `i`'s object: its keys exactly `{template, at, radix, c, r}`, a string template, a natural `at`, an integer
`c`. -/
def segmentHead (j : Json) : Except String (String × Nat × Int) := do
  sameKeys j ["template", "at", "radix", "c", "r"] ""
  pure (← strOf "" (← field j "template"), ← natAt j "at", ← intOf "" (← field j "c"))

def intsOf (j : Json) : Except String (List Int) := do (← arrOf "" j).mapM (intOf "")

def segmentOf (i : Nat) (j : Json) : Except String Segment := do
  let (template, start, c) ← (segmentHead j).mapError fun _ => s!"segment {i}: not a segment of integers at and c"
  let bad := s!"segment {i}: radix and r are one integer each a digit, each radix at least 1"
  let digits : Except String (List Int × List Int) := do pure (← intsOf (← field j "radix"), ← intsOf (← field j "r"))
  let (radix, r) ← digits.mapError fun _ => bad
  if radix.length != r.length || radix.any (· < 1) then throw bad
  pure { template, start, radix := radix.map Int.toNat, c, r }

/-- The certificate's segments and wires: its keys exactly `{format, segments, wires}`, its format `ORDER`. -/
def orderLists (j : Json) : Except String (List Json × List Json) := do
  sameKeys j ["format", "segments", "wires"] ""
  unless isStr (j.getObjValD "format") ORDER do throw ""
  pure (← arrOf "" (← field j "segments"), ← arrOf "" (← field j "wires"))

def pairOf (p : Json) : Except String (Nat × Nat) := do
  match ← arrOf "" p with
  | [a, b] => pure (← natOf "" a, ← natOf "" b)
  | _ => throw ""

/-- The certificate, `{"format", "segments", "wires"}` exactly, its format `verity/wire-order/v1`. -/
def orderOf (j : Json) : Except String Order := do
  let (segs, ws) ← (orderLists j).mapError fun _ => s!"not a {ORDER} certificate"
  let segments ← segs.zipIdx.mapM fun (s, i) => segmentOf i s
  let wires ← ws.zipIdx.mapM fun (p, i) =>
    (pairOf p).mapError fun _ => s!"wire {i}: not a [writer segment, reader segment] pair"
  pure { segments := segments.toArray, wires := wires.toArray }

/-! ## The count -/

/-- What `check` counts, in bits (`interface.input_count`): every unit's input rows, the rows wires fill, the rows
registered reads, table reads and openings of a held parameter fill, and the rest. -/
structure Counts where
  inputs : Nat
  wired : Nat
  registered : Nat
  free : Nat
  deriving Repr, BEq

/-- The count of record `r` with wire summary `s`, `held` the registered parameters held to the verifier's pins: a
read of a held parameter's fills its reader rows (none for an output port's, which fills no input); a table read or
opening of a table over one, its port's rows. -/
def countsOf (r : Stated) (s : SampledProofs.Wires.Summary) (held : List String) : Counts :=
  let c := r.wiring
  let words := fun (tid p : String) => (((c.templates.lookup tid).bind (·.rows.lookup p)).map (·.words)).getD 0
  let inputs := (c.templates.map fun (_, t) => 16 * (t.rows.map (·.2.words)).sum * t.units).sum
  let reg := r.registered
  let pinned := (reg.tables.filter (held.contains ·.param)).map (·.name)
  let registered :=
    ((reg.reads.filter (held.contains ·.param)).map fun x =>
      x.pat.units * (16 * (words x.pat.reader x.pat.port / c.cutsOf x.pat.reader x.pat.port))).sum +
    ((reg.roots.filter (pinned.contains ·.table)).map fun x => x.units * 16 * words x.reader x.port).sum +
    ((reg.openings.filter (pinned.contains ·.table)).map fun o => (o.ports.map fun p => o.units * 16 * words o.reader p).sum).sum
  { inputs, wired := s.wiredBits, registered, free := inputs - s.wiredBits - registered }

/-- The pins, `{param: root}` with each root a string, as (param, root) pairs. -/
def pinsOf (pins : Json) : Except String (List (String × String)) := do
  (← fields pins).mapM fun (p, v) => match v with
    | .str r => pure (p, r)
    | _ => throw "pins are {param: root}"

/-- The parameters `pins` names (`pinsOf`). -/
def heldOf (pins : Json) : Except String (List String) := do
  pure ((← pinsOf pins).map (·.1))

def refusalWhy (e : SampledProofs.Wires.Refusal) : String :=
  s!"wires: {e.code.name}" ++ match e.wire with
    | some i => s!" (wire {i})"
    | none => ""

/-- **The circuit record's checks**: record `record` read against Program `P` (`recordOf`) is its population program
(`populationOk`); its wires pass `SampledProofs.Wires.check`, each reader row its writer's output field bit for bit
(`leavesOk`); its declarations are well formed and fill no row a
wire or another declaration fills (`declRects`, `apart`); and `order` ranks every wire's writers below its readers
(`ordered`). A writer leaf is a gate its template computes or an input it passes through: the order ranks the chain's
writers below its readers, so it ends at a computed gate or a free or registered input. On acceptance, the record as
read against `P` and its wires' summary, which `countsOf` counts at any held parameters (`check`). -/
def checked (P : Prog) (record order : Json) : Except String (Stated × SampledProofs.Wires.Summary) := do
  let r ← recordOf P record
  unless populationOk P r do
    throw "the Program is not the record's population program: its root template nodes the record's templates, each a \
      batch of its units over whole input nodes of its own on axis 0, every input gate a template's parameter, the \
      root returning every template node's outputs, laid out and walkable as C-Flock's decoder lays a Program out, and \
      topological (NetworkCheck.Circuit.populationOk)"
  let s ← match SampledProofs.Wires.check r.wiring with
    | .ok s => pure s
    | .error e => throw (refusalWhy e)
  unless leavesOk P r do
    throw "wires: a reader row is not its writer's output field bit for bit, each in leaves of one width, the widths \
      equal or one of them a Boolean-profile port's"
  let rects ← declRects r.wiring r.registered
  unless apart rects (r.wiring.wires.map wireRect) do
    throw "the circuit's inputs: a row filled by two of its wires, registered reads, table reads or openings"
  match orderWhy r.wiring (← (orderOf order).mapError ("order: " ++ ·)) with
  | some e => throw s!"order: {e}"
  | none => pure ()
  pure (r, s)

/-- **The circuit check**: the record's checks (`checked`), its count holding the parameters `pins` names, the
registered parameters the verifier pinned whose window root the audit's registration records as the pin,
`{param: root}`. On acceptance, the record's `Counts`. -/
def check (P : Prog) (record order pins : Json) : Except String Counts := do
  let (r, s) ← checked P record order
  pure (countsOf r s (← heldOf pins))

/-- The circuit record's digest, `interface.circuit_bytes`: SHA-512 of its canonical JSON and a newline, in hex. -/
def digest (record : Json) : Except String String := do
  pure (Flock.hexOf (Flock.Sha512.hash ((← SampledProofs.PyJson.canonical record) ++ "\n").toUTF8))

/-- `network-records verdicts`' `circuit`: on `{"circuit", "order", "pins", "program"}`, the Program a descriptor
(`verity-ir/descriptor/v1`), `check`'s `Counts` and the record's `digest`, as a period's audit in the record form
takes them (`NetworkCheck.Period.auditOf`). -/
def circuitFn (j : Json) : Except String Json := do
  let P ← Flock.Program.decode (← Flock.canon (← field j "program")).toUTF8
  let record ← field j "circuit"
  let c ← check P record (← field j "order") (← field j "pins")
  pure (Json.mkObj [("counts", Json.mkObj [("free", toJson c.free), ("inputs", toJson c.inputs),
      ("registered", toJson c.registered), ("wired", toJson c.wired)]),
    ("digest", .str (← digest record))])

/-! ## The order on a toy wiring -/

private def toyWiring (ws : List Wire) : Wiring :=
  { templates := [("A", { units := 4, rows := [("x", ⟨1, 1⟩)], outs := [("o0", 1)] }),
                  ("B", { units := 4, rows := [("y", ⟨1, 1⟩)], outs := [("o0", 1)] })],
    cuts := [], wires := ws }

private def toyWire (base : Int) (radix : List Nat) (step : List Int) : Wire :=
  { reader := "B", port := "y", writer := "A", out := "o0", start := 0, radix, base, step, cut0 := 0, cuts := 1 }

private def toyOrder (ra rb : List Int) (cb : Int) (radix : List Nat := [4]) : Order :=
  { segments := #[{ template := "A", start := 0, radix, c := 0, r := ra },
                  { template := "B", start := 0, radix, c := cb, r := rb }], wires := #[(0, 1)] }

#guard ordered (toyWiring [toyWire 0 [4] [1]]) (toyOrder [0] [0] 1)
#guard orderWhy (toyWiring [toyWire 0 [4] [1]]) (toyOrder [0] [0] 0) ==
  some "wire 0: rank(writer) - rank(reader) reaches 0 >= 0"
#guard ordered (toyWiring [toyWire 0 [4] [1]]) (toyOrder [1] [1] 1)
#guard orderWhy (toyWiring [toyWire 1 [3] [1]]) (toyOrder [1] [1] 1) ==
  some "wire 0: rank(writer) - rank(reader) reaches 0 >= 0"
#guard orderWhy (toyWiring [toyWire 2 [3] [1]]) (toyOrder [1] [1] 1) ==
  some "wire 0: writer digit position 0 of segment 0 carries"
#guard orderWhy (toyWiring [toyWire 4 [1] [0]]) (toyOrder [1] [1] 1) ==
  some "wire 0: writer offset 4 outside segment 0"
#guard ordered (toyWiring [toyWire 0 [2, 2] [2, 1]]) (toyOrder [0, 0] [0, 0] 1 [1, 4])
#guard orderWhy (toyWiring [toyWire 0 [2, 2] [1, 1]]) (toyOrder [0, 0] [0, 0] 1 [2, 2]) ==
  some "wire 0: writer digit position 1 of segment 0 carries"
#guard ordered (toyWiring [toyWire 0 [2, 2] [2, 1]]) (toyOrder [5, 1] [5, 1] 1 [2, 2])
#guard orderWhy (toyWiring [toyWire 0 [2, 2] [2, 1]]) (toyOrder [5, 1] [1, 5] 1 [2, 2]) ==
  some "wire 0: rank(writer) - rank(reader) reaches 3 >= 0"
#guard orderWhy (toyWiring [toyWire 0 [2] [1]])
    { segments := #[{ template := "A", start := 0, radix := [1], c := 0, r := [0] },
                    { template := "B", start := 0, radix := [4], c := 1, r := [0] }], wires := #[(0, 1)] } ==
  some "wire 0: writer digit 0 (stride 1) on no position of segment 0"
#guard orderWhy (toyWiring [toyWire 0 [4] [1]]) { (toyOrder [0] [0] 1) with wires := #[(1, 0)] } ==
  some "wire 0: writer segment 1 is template B, not A"
#guard orderWhy (toyWiring [toyWire 0 [4] [1]]) { (toyOrder [0] [0] 1) with wires := #[] } ==
  some "0 wire pairs for 1 record wires"
private def toySegments (ss : List (String × Nat × Nat)) : Order :=
  { segments := (ss.map fun (t, s, n) => ({ template := t, start := s, radix := [n], c := 0, r := [0] } : Segment)).toArray,
    wires := #[] }

#guard orderWhy (toyWiring []) (toySegments [("A", 2, 3)]) == some "segment 0: units [2, 5) outside A's 4"
#guard orderWhy (toyWiring []) (toySegments [("C", 0, 1)]) == some "segment 0 names no template of the record"
#guard orderWhy (toyWiring []) (toySegments [("A", 0, 2), ("A", 1, 2)]) == some "segments 0 and 1 of A overlap"
private def orderError (format : String) (segments wires : Array Json) : String :=
  match orderOf (Json.mkObj [("format", .str format), ("segments", .arr segments), ("wires", .arr wires)]) with
  | .error e => e
  | .ok _ => ""
private def toySegment (at_ : Int) (radix : Array Json) : Json :=
  Json.mkObj [("template", .str "A"), ("at", toJson at_), ("radix", .arr radix), ("c", toJson (0 : Int)),
    ("r", .arr #[toJson (0 : Int)])]
#guard orderError ORDER #[toySegment 0 #[toJson 2]] #[.arr #[toJson 0, toJson 0]] == ""
#guard orderError "verity/wire-order/v2" #[] #[] == "not a verity/wire-order/v1 certificate"
#guard orderError ORDER #[toySegment (-1) #[toJson 2]] #[] == "segment 0: not a segment of integers at and c"
#guard orderError ORDER #[toySegment 0 #[toJson 0]] #[] ==
  "segment 0: radix and r are one integer each a digit, each radix at least 1"
#guard orderError ORDER #[] #[.arr #[toJson 0, toJson 1], .arr #[toJson 0]] ==
  "wire 1: not a [writer segment, reader segment] pair"
#guard rank (toyOrder [5, 1] [1, 5] 1 [2, 2]) "B" 2 == 2 && rank (toyOrder [5, 1] [1, 5] 1 [2, 2]) "A" 3 == 6

end NetworkCheck.Circuit
