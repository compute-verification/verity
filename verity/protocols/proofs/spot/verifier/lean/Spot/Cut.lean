module

public import Spot.Edges
public import Spot.Partition
public import Std.Data.HashMap
public import Std.Data.HashSet

/-!
# The cut Program is the wired one's cut

`check` holds a Program the verifier is handed as cut (`cut`) to the cut of a wired one (`wired`) under the circuit
record (`record`), structurally, on their decoded descriptors. The cut is `population.cut_wires`'s: the wired builder
over the record with no wires, so the root keeps the wired root's batches in order and reads input rows where the
wired root read a writer's outputs.

* The wired root is the record's wired population: its key is `Root[WiredPopulations_v1{templates, record, counts}]_v1`
  at the record's templates and units and the record's digest (`recordDigest`); `cut-input` for a root that is no wired
  population, `cut-record` for one of another record. The cut root's key is the same at the digest of the record with
  no wires (`cut-record`).
* The wires are whole writer fields at whole rows of the record's cuts (`Edges.edgesOf`, refused by its `edge-*`
  name: circuits' `cut-offset` and `cut-units` are its `edge-rows`, `cut-fields` its `edge-fields` or `edge-width`).
* Node for node, the batches are the wired root's: the same template (`cut-order` for one of the templates out of
  place, `cut-template` for another), the same members (`cut-count`), and each template's Definition the wired
  descriptor's, through every Definition it calls, compared by key (`cut-template`).
* Argument for argument, a part that reads input nodes reads them in the cut as in the wired root, each wired input
  node one cut input node of the same leaves (`cut-arg`); the parts that read a writer read, in the cut, as many leaves
  of input nodes in affine parts, which no other part reads (`cut-arg`: a wire left in place, or rows read twice).
* Nothing else: no batch more or fewer (`cut-extra`, `cut-order`), no input node after a batch or read by no part, and
  the return the wired one's at the nodes' places (`cut-extra`).

The work is the two roots' nodes and their arguments' parts, and each Definition the templates reach, once: nothing per
unit, member or gate.
-/

@[expose] public section

namespace Spot.Cut

open Lean (Json)
open Flock (V)
open Flock.Program (Prog Node Arg Part Seq Defn Body Ref Lane isInputId)
open Flock.TemplateQuery (TRange)

/-- The record's fields its digest reads (`population.WIRED_FIELDS`). -/
def WIRED_FIELDS : Array String :=
  #["wires", "cuts", "registered", "order", "ingress", "boundary_reads", "free", "external"]

/-- `population._record_digest`: SHA-512 of the canonical JSON of the record's `WIRED_FIELDS`, lowercase hex. -/
def recordDigest (record : Json) : V String := do
  let kvs := WIRED_FIELDS.toList.filterMap fun k => (field? record k).map (k, ·)
  return Flock.hexOf (Flock.Sha512.hash (← Flock.canon (Json.mkObj kvs)).toUTF8)

/-- The record with no wires, as `cut_wires` builds the cut over. -/
def noWires (record : Json) : Json := record.setObjVal! "wires" (Json.arr #[])

/-- The record's templates and units, `[[tid, n], ...]`. -/
def templatesOf (record : Json) : Except String (Array (String × Nat)) := do
  let .arr ts := getD record "templates" | throw "cut-record: the record's templates are not a list"
  ts.mapM fun t => match t with
    | .arr #[.str tid, n] => match nat? n with
      | some n => if tid.contains '\'' then throw s!"cut-record: template {tid} has a quote" else pure (tid, n)
      | none => throw "cut-record: a template's units are not a natural number"
    | _ => throw "cut-record: a template is not [id, units]"

/-- The root key `WiredPopulations_v1` over templates `ts` and record digest `d` has (`codec.descriptor_id`). -/
def rootKey (ts : Array (String × Nat)) (d : String) : String :=
  "Root[WiredPopulations_v1{templates=(" ++ ",".intercalate (ts.toList.map fun (t, _) => "'" ++ t ++ "'") ++
    "),record='" ++ d ++ "',counts=(" ++ ",".intercalate (ts.toList.map fun (_, n) => toString n) ++ ")}]_v1"

/-- Whether root node `x` of `P` calls an input. -/
def isInput (P : Prog) (x : Node) : Bool :=
  let f := P.defs[x.fn]!
  f.prim && isInputId f.key

def partEq : Part → Part → Bool
  | .affine p i b c, .affine p' i' b' c' => p == p' && i == i' && b == b' && c == c'
  | .strided p i b n os is c, .strided p' i' b' n' os' is' c' =>
    p == p' && i == i' && b == b' && n == n' && os == os' && is == is' && c == c'
  | .explicit a, .explicit a' => a == a'
  | .grid r ls, .grid r' ls' => r == r' && ls == ls'
  | _, _ => false

def seqEq (a b : Seq) : Bool :=
  a.count == b.count && a.starts == b.starts && a.parts.size == b.parts.size &&
    (a.parts.zip b.parts).all fun (x, y) => partEq x y

def argEq (a b : Arg) : Bool := a.ty == b.ty && seqEq a.seq b.seq && a.dims == b.dims

/-- Two nodes are one, a node of `W` and one of `U`, their callees named by key. -/
def nodeEq (W U : Prog) (a b : Node) : Bool :=
  a.form == b.form && (W.defs[a.fn]!).key == (U.defs[b.fn]!).key && a.args.size == b.args.size &&
    (a.args.zip b.args).all (fun (x, y) => argEq x y) && a.out == b.out && a.axes == b.axes && a.n == b.n &&
    a.nxs == b.nxs

def bodyEq (W U : Prog) (a b : Body) : Bool :=
  a.params == b.params && a.retType == b.retType && a.nodes.size == b.nodes.size &&
    (a.nodes.zip b.nodes).all (fun (x, y) => nodeEq W U x y) && seqEq a.ret b.ret && a.offsets == b.offsets

/-- Two Definitions are one, but for the Definitions they call, which `sameDefs` compares by key. -/
def defnEq (W U : Prog) (a b : Defn) : Bool :=
  a.key == b.key && a.id == b.id && a.prim == b.prim && a.params == b.params && a.ret == b.ret &&
    a.binding == b.binding && a.gates == b.gates && a.paramLeaves == b.paramLeaves && a.outLeaves == b.outLeaves &&
    bodyEq W U a.body b.body

/-- Whether Definition `key` is the same in `W` and `U`, with every Definition it calls in `U`, each compared once. -/
def sameDefs (W U : Prog) (key : String) : Bool := Id.run do
  let mut todo := [key]
  let mut seen : Std.HashSet String := {}
  for _ in [0:U.defs.size + 1] do
    match todo with
    | [] => return true
    | k :: rest =>
      todo := rest
      if seen.contains k then continue
      seen := seen.insert k
      match W.index[k]?, U.index[k]? with
      | some i, some j =>
        let d := U.defs[j]!
        if !defnEq W U W.defs[i]! d then return false
        todo := d.body.nodes.toList.map (fun x => (U.defs[x.fn]!).key) ++ todo
      | _, _ => return false
  return todo.isEmpty

/-- Whether input node `x` of `W` and input node `y` of `U` are one. -/
def inputEq (W U : Prog) (x y : Node) : Bool :=
  (W.defs[x.fn]!).key == (U.defs[y.fn]!).key && x.form == y.form && x.n == y.n && x.out == y.out

/-- The nodes two parts of one shape read, paired in order: none where their shapes differ but for the nodes, or
either reads a parameter. -/
def pairs : Part → Part → Option (Array (Nat × Nat))
  | .affine false i b c, .affine false i' b' c' => if b == b' && c == c' then some #[(i, i')] else none
  | .strided false i b n os is c, .strided false i' b' n' os' is' c' =>
    if b == b' && n == n' && os == os' && is == is' && c == c' then some #[(i, i')] else none
  | .explicit a, .explicit a' =>
    if a.size == a'.size && (a.zip a').all (fun (r, r') => !r.p && !r'.p && r.j == r'.j) then
      some ((a.zip a').map fun (r, r') => (r.i, r'.i)) else none
  | .grid r ls, .grid r' ls' =>
    if r == r' && ls.size == ls'.size &&
        (ls.zip ls').all (fun (l, l') => !l.p && !l'.p && l.b == l'.b && l.e == l'.e && l.lv == l'.lv) then
      some ((ls.zip ls').map fun (l, l') => (l.i, l'.i)) else none
  | _, _ => none

/-- The nodes a part reads, or none where it reads a parameter. -/
def nodesOf : Part → Option (Array Nat)
  | .affine false i _ _ => some #[i]
  | .strided false i _ _ _ _ _ => some #[i]
  | .explicit a => if a.any (·.p) then none else some (a.map (·.i))
  | .grid _ ls => if ls.any (·.p) then none else some (ls.map (·.i))
  | _ => none

/-- The leaves `[lo, lo + n)` of node `j` a part reads within: its run where it is affine, the node whole otherwise. -/
def hull (BU : Body) (j : Nat) : Part → Nat × Nat
  | .affine _ _ b c => (b, c)
  | _ => (0, (BU.nodes[j]!).out.leaves)

/-- The walk's state: the wired input nodes' cut nodes and back, and the leaves the kept parts and the cut rows read,
`(node, lo, n)` each. -/
structure Walk where
  φ : Std.HashMap Nat Nat := {}
  back : Std.HashMap Nat Nat := {}
  kept : Array (Nat × Nat × Nat) := #[]
  cuts : Array (Nat × Nat × Nat) := #[]

/-- Whether part `p` of the wired root reads writers alone: some when it reads no parameter, `true` for writers. -/
def wireOf (W : Prog) (BW : Body) (p : Part) : Option Bool :=
  (nodesOf p).map fun ns => ns.all fun i => !(BW.nodes[i]?.any (isInput W ·))

/-- The wired root's argument `a` against the cut's `b`, at the walk so far. -/
def walkArg (W U : Prog) (BW BU : Body) (at_ : String) (a b : Arg) (st0 : Walk) : Except String Walk := do
  let mut st := st0
  let ws := a.seq.parts
  let us := b.seq.parts
  let mut q := 0
  let mut k := 0
  for _ in [0:ws.size + 1] do
    if k ≥ ws.size then break
    let p := ws[k]!
    let some ns := nodesOf p | throw s!"cut-arg: {at_} reads a parameter"
    let wire := ns.all fun i => !(BW.nodes[i]?.any (isInput W ·))
    if !wire then
      if ns.any (fun i => !(BW.nodes[i]?.any (isInput W ·))) then
        throw s!"cut-arg: {at_}: a part reads a writer and an input"
      let some u := us[q]? | throw s!"cut-arg: {at_} has fewer parts in the cut"
      let some ps := pairs p u | throw s!"cut-arg: {at_}: a part reads otherwise in the cut"
      for (i, j) in ps do
        if !(BU.nodes[j]?.any (isInput U ·)) || !inputEq W U BW.nodes[i]! BU.nodes[j]! then
          throw s!"cut-arg: {at_}: input node {i} is read as root node {j}"
        match st.φ[i]?, st.back[j]? with
        | some j', _ => if j' != j then throw s!"cut-arg: {at_}: input node {i} is read as two"
        | none, some _ => throw s!"cut-arg: {at_}: two input nodes are read as root node {j}"
        | none, none => st := { st with φ := st.φ.insert i j, back := st.back.insert j i }
        let (lo, n) := hull BU j u
        st := { st with kept := st.kept.push (j, lo, n) }
      q := q + 1
      k := k + 1
    else
      let mut want := 0
      for _ in [0:ws.size] do
        let some p' := ws[k]? | break
        if wireOf W BW p' != some true then break
        want := want + p'.count
        k := k + 1
      let mut got := 0
      for _ in [0:us.size] do
        if got ≥ want then break
        match us[q]? with
        | some (.affine false j lo c) =>
          if !(BU.nodes[j]?.any (isInput U ·)) then
            throw s!"cut-arg: {at_} reads root node {j}, which is no input node: a wire left in place"
          st := { st with cuts := st.cuts.push (j, lo, c) }
          got := got + c
          q := q + 1
        | _ => throw s!"cut-arg: {at_}: a wire's rows are not affine reads of input nodes in the cut"
      if got != want then throw s!"cut-arg: {at_}: a wire's {want} leaves are {got} in the cut"
  if q != us.size then throw s!"cut-arg: {at_} has more parts in the cut"
  return st

/-- Whether `[lo, lo + n)` and `[lo', lo' + n')` meet. -/
def meets (lo n lo' n' : Nat) : Bool := lo < lo' + n' && lo' < lo + n

/-- The record's templates, where the wired root is the record's wired population and the cut root that of the record
with no wires; `cut-input` or `cut-record` otherwise. -/
def roots (record : Json) (wired cut : Prog) : Except String (Array (String × Nat)) := do
  let ts ← templatesOf record
  let dW ← recordDigest record
  let dU ← recordDigest (noWires record)
  let kW := (wired.defs[wired.root]!).key
  if kW != rootKey ts dW then
    throw (if kW.startsWith "Root[WiredPopulations_v1{" then
      s!"cut-record: the wired root {kW} names another record or templates than {dW} and {ts}"
      else s!"cut-input: the wired root {kW} is no wired population")
  if (cut.defs[cut.root]!).key != rootKey ts dU then
    throw s!"cut-record: the cut's root is not the wired population of the record with no wires, {dU}"
  return ts

/-- **The cut Program is the wired one's cut** under `record`, at the wired root's ranges `rs`: the module docstring's
rule, or the first refusal, by name. -/
def check (record : Json) (wired cut : Prog) (rs : Array TRange) : Except String Unit := do
  let ts ← roots record wired cut
  let templates := ts.map (·.1)
  let wiring : Spot.Wires.Wiring :=
    { templates := ts.toList.map fun (t, n) => (t, { units := n, rows := [], outs := [] })
      cuts := (entries (getD record "cuts")).toList.map fun (t, ps) =>
        (t, (entries ps).toList.filterMap fun (p, k) => (nat? k).map (p, ·))
      wires := [] }
  let _ ← Edges.edgesOf wired rs wiring
  let BW := (wired.defs[wired.root]!).body
  let BU := (cut.defs[cut.root]!).body
  let batW := (List.range BW.nodes.size).toArray.filter fun k => !isInput wired BW.nodes[k]!
  let ks := (List.range BU.nodes.size).toArray
  let insU := ks.filter fun k => isInput cut BU.nodes[k]!
  let batU := ks.filter fun k => !isInput cut BU.nodes[k]!
  for (k, k') in batW.zip batU do
    let x := BW.nodes[k]!
    let y := BU.nodes[k']!
    let t := (wired.defs[x.fn]!).key
    let key := (cut.defs[y.fn]!).key
    if key != t then
      throw (if templates.contains key then s!"cut-order: root node {k'} is a batch of {key}, the wired one's of {t}"
        else s!"cut-template: root node {k'} calls {key}, which is none of the templates {templates}")
    if y.form != x.form || y.form != .batch then throw s!"cut-template: root node {k'} is no batch of {t}"
    if y.n != x.n then throw s!"cut-count: root node {k'} has {y.n} members, the wired one's {x.n}"
    if y.out != x.out || y.axes != x.axes || y.nxs != x.nxs || y.args.size != x.args.size ||
        !(x.args.zip y.args).all (fun (a, b) => a.ty == b.ty && a.dims == b.dims && a.seq.count == b.seq.count) then
      throw s!"cut-arg: root node {k'}'s arguments are not the wired node's"
  if batU.size > batW.size then throw s!"cut-extra: root node {batU[batW.size]!} is past the wired root's batches"
  if batU.size < batW.size then throw s!"cut-order: the wired root's node {batW[batU.size]!} has no batch in the cut"
  if insU.any (fun i => batU.any (· < i)) then throw "cut-extra: an input node follows a batch"
  for (t, n) in ts do
    if rs.foldl (fun a r => if r.template == t then a + r.count else a) 0 != n then
      throw s!"cut-count: the wired root has other than the record's {n} units of {t}"
    if !sameDefs wired cut t then
      throw s!"cut-template: {t}'s Definition, or one it calls, is not the wired descriptor's"
  let mut st : Walk := {}
  for (k, k') in batW.zip batU do
    for ((a, b), i) in (BW.nodes[k]!.args.zip BU.nodes[k']!.args).zipIdx do
      st ← walkArg wired cut BW BU s!"root node {k'} argument {i}" a b st
  for x in [0:st.cuts.size] do
    let (j, lo, n) := st.cuts[x]!
    if lo + n > (BU.nodes[j]!).out.leaves then throw s!"cut-arg: a wire's rows run past input node {j}"
    if (st.cuts.extract (x + 1)).any (fun (j', lo', n') => j' == j && meets lo n lo' n') ||
        st.kept.any (fun (j', lo', n') => j' == j && meets lo n lo' n') then
      throw s!"cut-arg: input node {j}'s leaves [{lo}, {lo + n}) are read twice"
  for i in insU do
    if (st.back[i]?).isNone && !st.cuts.any (·.1 == i) then throw s!"cut-extra: input node {i} is read by no part"
  let place := fun (k : Nat) => if BW.nodes[k]?.any (isInput wired ·) then st.φ[k]? else
    (batW.findIdx? (· == k)).bind (batU[·]?)
  let retOk := BW.retType == BU.retType && BW.ret.count == BU.ret.count && BW.ret.starts == BU.ret.starts &&
    BW.ret.parts.size == BU.ret.parts.size &&
    (BW.ret.parts.zip BU.ret.parts).all fun (p, q) => match pairs p q with
      | some ps => ps.all fun (i, j) => place i == some j
      | none => false
  if !retOk then throw "cut-extra: the cut's return is not the wired root's at its nodes' places"

/-- `roots`, then the template query's ranges over the wired Program and `check`. -/
def checkWith (record : Json) (wired cut : Program) (query : Json) : V Unit := do
  let _ ← roots record wired.prog cut.prog
  check record wired.prog cut.prog (← Partition.evaluate wired query)

/-- `checkWith` on `{"wired", "cut", "record", "query"}`: two descriptors, the circuit record and the query. -/
def ofJson (j : Json) : V (Except String Unit) := do
  let some w@(.obj _) := field? j "wired" | throw "wired is not an object"
  let some u@(.obj _) := field? j "cut" | throw "cut is not an object"
  let wired ← Program.parse (← Flock.canon w).toUTF8
  let cut ← Program.parse (← Flock.canon u).toUTF8
  return checkWith (getD j "record") wired cut (getD j "query")

/-- The difftest's `cut` piece: `{"accept": BOOL}`. -/
def piece (j : Json) : V Json := do
  return Json.mkObj [("accept", (← ofJson j).isOk)]

/-- The difftest's `cut_why` piece: `{"refusal": NAME}`, the refusal's name, or `null` on an accept. -/
def pieceWhy (j : Json) : V Json := do
  return Json.mkObj [("refusal", match ← ofJson j with
    | .ok () => Json.null
    | .error e => Json.str ((e.splitOn ":").headD e))]

end Spot.Cut
