module

public import CoreProtocol.Blobs
public import Spot.OneStage
public import Spot.ChallengerTrees
public import Spot.Slots
public import Spot.PopOrder
public import Spot.FilePart
public import Spot.CircuitKept
public import Spot.Wires
public import Spot.Edges
public import Spot.RecordText
public import Spot.Restrict
public import Spot.Partition
public import Circuits.DefinitionDigest
public import Flock
public import Flock.JsonParse
public import Std.Data.HashMap
public import Std.Data.HashSet
import all Init.Util
meta import Flock
meta import Flock.JsonParse
meta import Spot.Restrict
meta import Spot.Edges
meta import Circuits.DefinitionDigest

/-!
# The one form the door lowers

`admitted`, C_S's form, and `admittedBlock`, the block's form in C, with `lowerBlock`, the block circuit the door lowers
once both hold, and their guards. `Spot/OneStageZkStatement.lean` imports it.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-- A block circuit's input rows: each parameter `(name, bits)` a row of its own, or, cut `k` ways (`cuts`, `ioCuts`'
for its template), `k` rows `name#j` of `bits / k` each, in order (`interface.RowLayout.cut`, `row_name`). -/
def cutRows (cuts : List (String × Nat)) (ins : Array (String × Nat)) : Array (String × Nat) :=
  ins.flatMap fun (n, b) => match cuts.lookup n with
    | some k => if k ≤ 1 then #[(n, b)] else (Array.range k).map fun j => (s!"{n}#{j}", b / k)
    | none => #[(n, b)]

#guard cutRows [("x", 4)] #[("x", 1024), ("w", 256)] ==
  #[("x#0", 256), ("x#1", 256), ("x#2", 256), ("x#3", 256), ("w", 256)]
#guard cutRows [] #[("x", 1024)] == #[("x", 1024)]

/-- The layout C_S's template is lowered at (`one_circuit.LAYOUT`). -/
def LAYOUT : String := "iv/hybrid/4"

/-- The templates a template query lists (`one_circuit._listed`); none for another query. -/
def listedOf (partition : Json) : Option (Array String) :=
  let q := partition.getObjValD "query"
  let ps := q.getObjValD "params"
  match q.getObjValAs? String "name" with
  | .ok "Q_template_instance" => some #[(ps.getObjValAs? String "template").toOption.getD ""]
  | .ok "Q_template_instances" => match ps.getObjValD "templates" with
    | .arr ts => some (ts.map fun t => (t.getStr?).toOption.getD "")
    | _ => none
  | _ => none

/-- The prefix of every refusal of `admitted`, which names its clause after it. -/
def NOT_ADMITTED : String := "lower: not the one form the door lowers (Spot.OneStageZk.admitted)"

/-- A refusal by `admitted`'s clause `clause`. -/
def notAdmitted {α : Type} (clause detail : String) : V α := throw s!"{NOT_ADMITTED}, {clause}: {detail}"

/-- What `admitted` accepts: the block's template `key`, its index `d` in C_S, and its block circuit's input rows and
return fields `(name, bits)`, in the order `Flock.Lower.lower` lays their leaves. -/
structure Admit where
  key : String
  d : Nat
  ins : Array (String × Nat)
  fields : Array (String × Nat)

/-- The composite Definitions Definition `d` of `P` reaches, itself included, whose parameter or return leaves are not
all one bit (`Flock.Lower.onBits`), by key, ascending. Its work is the Definitions reached, each once, with their nodes.
It reads no primitive: in a lift every word primitive is replaced by its Boolean Definition, a gate or a composite on bits
(`Flock.Lift.closure` refuses any other), and an input node appears only in the root's body (the decoder's
`inputOutsideRoot`), so every primitive a lifted template reaches is on bits. -/
def wideFrom (P : Flock.Program.Prog) (d : Nat) : Array String := Id.run do
  let bound := P.defs.foldl (fun n x => n + x.body.nodes.size + 1) 2
  let mut seen : Std.HashSet Nat := {}
  let mut todo := #[d]
  let mut at_ := 0
  let mut wide : Array String := #[]
  for _ in [0:bound] do
    if h : at_ < todo.size then
      let i := todo[at_]
      at_ := at_ + 1
      if seen.contains i then continue
      seen := seen.insert i
      let some x := P.defs[i]? | continue
      if x.prim then continue
      if !Flock.Lower.onBits x then wide := wide.push x.key
      todo := todo ++ x.body.nodes.map (·.fn)
    else break
  return wide.qsort (· < ·)

/-- Where `l`, Definition `fn` lifted to bits, has other leaves than `fn` has bits: the first parameter, then return
field (`fieldsOf`), whose bits are not its counterpart's leaves, or the totals `Flock.Lower.lower` reads
(`paramLeaves`, `outLeaves`); none where each agrees. -/
def leavesOff (fn l : Flock.Program.Defn) : Option String :=
  let bits := fun (t : Flock.Program.Ty) => t.widths.foldl (· + ·) 0
  let fs := Flock.Program.fieldsOf fn.ret
  let gs := Flock.Program.fieldsOf l.ret
  let ports := (fn.params.zip l.params).toList.map (fun ((n, t), (_, u)) => ("parameter", n, bits t, u.leaves)) ++
    (fs.zip gs).mapIdx fun j (t, u) => ("return field", s!"o{j}", bits t, u.leaves)
  let sIn := fn.params.foldl (· + bits ·.2) 0
  if fn.params.size != l.params.size || fs.length != gs.length then
    some s!"{fn.params.size} parameters and {fs.length} return fields, its lift's {l.params.size} and {gs.length}"
  else match ports.find? fun (_, _, b, k) => b != k with
    | some (what, n, b, k) => some s!"{what} {n} of {b} bits has {k} leaves after the lift to bits"
    | none =>
      if sIn != l.paramLeaves || bits fn.ret != l.outLeaves then
        some s!"parameters of {sIn} bits and a return of {bits fn.ret}, its lift's {l.paramLeaves} and {l.outLeaves} leaves"
      else none

/-- A port name the block circuit's file and `flock-lower-check compose`'s `name:bits,…` carry as written. -/
def plainName (n : String) : Bool := !n.isEmpty && n.all fun c => ![',', ':', '#', ' ', '\t', '\r', '\n'].contains c

/-- **The one form the door lowers**: C_S's block, its root calling one template, admitted only where every clause holds,
else refused under `NOT_ADMITTED` by the first that fails. Each clause makes the block circuit's rows, laid in whole u16
words a row (`Flock.Compose.compose`), agree with the leaves `Flock.Lower.lower` lays one after another from column 0:
* `one-template`: the root's non-input nodes call one Definition, the template;
* `listed`: the partition's template query lists it, and it is a Definition of C_S;
* `port-names`: its parameters' names are plain (`plainName`), and its rows (`cutRows`) and fields `o{j}` are named
  once each;
* `on-bits`: the template lifted to bits (`Flock.Lift.lift`) reaches no composite with a leaf wider than one bit
  (`wideFrom`), so no word is read as one column (`Flock.Lower.WIDE` never refuses what this admits);
* `cut-rows`: a cut of in(S)'s (`cuts`) names a parameter, at least two ways, whose leaves are of one width and divide
  by the cut, so its rows are its leaves in equal runs of equal bits;
* `whole-words`: there is a return field (`fieldsOf`), and every input row and return field is a positive whole number of
  u16 words, so each starts at the column of its first leaf and no column follows its last;
* `leaves-are-bits`: the rows' bits are the lifted template's parameter leaves and the fields' its return leaves, so no
  input or output column is past a leaf.
A function of the partition, `cuts` and C_S, in the template's description; `lift` is `Flock.Lift.lift` but in a guard
that feeds `leaves-are-bits` a lift it cannot otherwise meet. -/
def admitted (partition : Json) (cuts : List (String × List (String × Nat))) (S : Flock.Program.Prog)
    (lift : Flock.Program.Prog → Nat → V (Flock.Program.Prog × Nat) := Flock.Lift.lift) : V Admit := do
  let some listed := listedOf partition
    | throw (Flock.OneCircuit.notDerived
        "lower : C_S of a partition that is no template query (Q_template_instance, Q_template_instances) -> its Boolean block circuit")
  -- unreachable from a descriptor: `Spot.Restrict.restrict`'s C_S has its root among its Definitions
  let some root := S.defs[S.root]? | notAdmitted "one-template" "C_S's root is not one of its Definitions"
  let mut called : Array String := #[]
  for k in [0:root.body.nodes.size] do
    if Spot.Edges.isInputNode S k then continue
    let key := ((root.body.nodes[k]?).bind (S.defs[·.fn]?) |>.map (·.key)).getD ""
    if !called.contains key then called := called.push key
  let keys := called.qsort (· < ·)
  let #[key] := keys
    | notAdmitted "one-template" s!"C_S's root calls {keys.toList}, and a block's units are instances of one template"
  -- unreachable through the door: the partition's query (`Partition.evaluate`) refuses a unit of a template it lists not
  if !listed.contains key then notAdmitted "listed" s!"C_S's template {key} is not one the partition's query lists"
  -- this and the next unreachable from a descriptor: C_S indexes each of its Definitions
  let some d := S.index[key]? | notAdmitted "listed" s!"C_S's template {key} has no entry in C_S's index"
  let some fn := S.defs[d]? | notAdmitted "listed" s!"C_S's template {key} is no Definition of C_S"
  let cs := (cuts.lookup key).getD []
  let bits := fun (t : Flock.Program.Ty) => t.widths.foldl (· + ·) 0
  let ins := cutRows cs (fn.params.map fun (n, t) => (n, bits t))
  let fields := (Flock.Program.fieldsOf fn.ret).toArray.mapIdx fun j f => (s!"o{j}", bits f)
  if let some (n, _) := fn.params.find? (!plainName ·.1) then
    notAdmitted "port-names" s!"C_S's template {key}: parameter '{n}' is empty or holds ',', ':', '#' or a space"
  let names := ins.map (·.1) ++ fields.map (·.1)
  if let some n := names.find? fun n => (names.filter (· == n)).size > 1 then
    notAdmitted "port-names" s!"C_S's template {key}: port {n} is named twice"
  let (L, dl) ← match lift S d with
    | .ok r => pure r
    | .error _ => notAdmitted "on-bits" s!"the lift to bits refuses C_S's template {key}"
  if let some w := (wideFrom L dl)[0]? then
    notAdmitted "on-bits" s!"C_S's template {key} reaches {w}, whose leaves are not all one bit after the lift to bits"
  for (p, k) in cs do
    -- through the door only the width test refuses: `ioCuts` cuts a parameter of the template `leaves / fl` ways, `fl`
    -- dividing its leaves, and keeps only the cuts of at least two
    let some (_, t) := fn.params.find? (·.1 == p)
      | notAdmitted "cut-rows" s!"C_S's template {key}: a cut names {p}, which is no parameter"
    if k < 2 || (Flock.Program.Widths.homo t).isNone || t.leaves % k != 0 then
      notAdmitted "cut-rows" s!"C_S's template {key}: parameter {p} of {t.leaves} leaves cut {k} ways, not into equal runs of leaves of one width"
  if fields.isEmpty then notAdmitted "whole-words" s!"C_S's template {key} returns no field"
  for (what, rs) in [("input row", ins), ("return field", fields)] do
    if let some (n, b) := rs.find? fun (_, b) => b == 0 || b % 16 != 0 then
      notAdmitted "whole-words" s!"C_S's template {key}: {what} {n} of {b} bits is not a positive whole number of u16 words"
  -- unreachable from a descriptor (`Flock.Lift.lift` gives its template's lift, its leaves its bits): guarded on a lift
  -- changed by hand
  let some l := L.defs[dl]? | notAdmitted "leaves-are-bits" s!"C_S's template {key} has no lift to bits"
  if let some why := leavesOff fn l then notAdmitted "leaves-are-bits" s!"C_S's template {key}: {why}"
  return { key, d, ins, fields }

/-- **The door admits C_S**: `admitted` accepts it. The hypothesis of every statement about the block circuit the door
lowers. -/
def Admitted (partition : Json) (cuts : List (String × List (String × Nat))) (S : Flock.Program.Prog) : Prop :=
  (admitted partition cuts S).toOption.isSome = true

instance instDecidableAdmitted (partition : Json) (cuts : List (String × List (String × Nat))) (S : Flock.Program.Prog) :
    Decidable (Admitted partition cuts S) := inferInstanceAs (Decidable (_ = true))

/-- **The door's second predicate, the block's form in C**, beside `admitted`'s on C_S: the units `units` of C, whose C_S `admitted` gave the template
`key`, refused under `NOT_ADMITTED` by the first clause that fails:
* `no-recut`: C's root reads no input node recut (`body.recut` is empty), which C_S's agreement with C assumes and
  `Flock.Program.decode`, restrict's path, keeps;
* `disjoint-ranges`: no unit is named twice, and no two units hold a gate in common;
* `one-callee`: every unit's callee is a Definition of C, and is `key`;
* `operands-outside`: every operand gate of every unit is outside the block, so each is a port of the block's front at
  its template's parameter-leaf width and no row of the block circuit stands for a gate another unit computes.
A function of C, the units and `key`; its work is the units' operands. -/
def admittedBlock (P : Flock.Program.Prog) (units : Array Nat) (key : String) : V Unit := do
  -- unreachable from a descriptor: the decoder's root is one of its Definitions
  let some root := P.defs[P.root]? | notAdmitted "no-recut" "C's root is not one of its Definitions"
  -- unreachable through the door: a root read recut is refused before, by the gate layout restrict reads (`recut-gates`,
  -- `RC16`)
  if !root.body.recut.isEmpty then
    notAdmitted "no-recut" s!"C's root reads input nodes {(root.body.recut.map (·.1)).toList} recut, in leaves of part of their width"
  let named := units.qsort (· < ·)
  -- unreachable through the door, whose units are a block's range of the partition
  for i in [1:named.size] do
    if named[i - 1]! == named[i]! then notAdmitted "disjoint-ranges" s!"unit {named[i]!} is named twice in the block"
  -- `unitAt`'s node is one of the root's (`Prog.calls`), so "no node of C's root" is unreachable by construction; the
  -- member's scope is its node's callee (`Prog.child`)
  let us ← units.mapM fun u => do
    let (k, m) ← Spot.BlockFront.unitAt P u
    let some x := root.body.nodes[k]? | notAdmitted "one-callee" s!"unit {u} is no node of C's root"
    -- unreachable from a descriptor: the decoder resolves every node's callee to a Definition
    let some f := P.defs[x.fn]? | notAdmitted "one-callee" s!"unit {u}'s callee is no Definition of C"
    let off := (P.child P.rootScope k m).off
    pure (u, m, x, f, off, off + f.gates)
  let byLo := us.qsort fun a b => a.2.2.2.2.1 < b.2.2.2.2.1
  for i in [1:byLo.size] do
    let (u, _, _, _, _, hi) := byLo[i - 1]!
    let (v, _, _, _, lo, _) := byLo[i]!
    -- unreachable from a descriptor, whose decoder lays the root's nodes at disjoint offsets
    if lo < hi then notAdmitted "disjoint-ranges" s!"units {u} and {v} of the block hold gates in common"
  for (u, m, x, f, _, _) in us do
    -- unreachable through the door: C_S's root calls each unit's callee, and `admitted`'s `one-template` holds
    if f.key != key then notAdmitted "one-callee" s!"unit {u} calls {f.key}, not the block's template {key}"
    for (g, _) in ← Spot.BlockFront.operandsW P x m do
      if let some (v, _, _, _, _, _) := us.find? fun (_, _, _, _, lo, hi) => lo ≤ g && g < hi then
        notAdmitted "operands-outside" s!"unit {u} reads gate {g}, which unit {v} of the block computes: a unit reads only gates outside its block"

/-- **The door admits the block in C**: `admittedBlock` accepts it. -/
def AdmittedBlock (P : Flock.Program.Prog) (units : Array Nat) (key : String) : Prop :=
  (admittedBlock P units key).toOption.isSome = true

instance instDecidableAdmittedBlock (P : Flock.Program.Prog) (units : Array Nat) (key : String) : Decidable (AdmittedBlock P units key) :=
  inferInstanceAs (Decidable (_ = true))

/-- The template `lowerBlock` lowers C_S's block to at what `admitted` gives `a`: its template on bits, every input row
and return field in u16 words. -/
def blockTmpl (S : Flock.Program.Prog) (a : Admit) : V Flock.Tmpl.Tmpl :=
  Flock.Lower.lower S a.d (Array.replicate (a.ins.foldl (fun n p => n + p.2 / 16) 0) 16)
    (Array.replicate (a.fields.foldl (fun n p => n + p.2 / 16) 0) 16)

/-- **C_S's block circuit** (`one_circuit.lower`): `S`, the block's Program, refused unless `admitted` and `formed` at its
template (the door's `admittedBlock`, C's units' form), then lowered at its
one template by `Flock.Lower.lower` and written by `Flock.Compose.compose` at the rows and fields `admitted` gives:
output ports its return fields `o{j}` in u16 words, META's `program_digests` the template's `definitionDigest` read off
`CJ`, C's descriptor. -/
def lowerBlock (CJ : Json) (partition : Json) (cuts : List (String × List (String × Nat))) (S : Flock.Program.Prog)
    (formed : String → V Unit) : V ByteArray := do
  let a ← admitted partition cuts S
  formed a.key
  let outs := a.fields.map fun (n, b) => (n, b / 16)
  let tab := fun (k : String) => match CJ.getObjValD k with | .arr xs => xs | _ => #[]
  let digest ← Flock.DefinitionDigest.definitionDigest (tab "callees") (tab "types")
    ((CJ.getObjValD "definitions").getObjValD a.key)
  let t ← blockTmpl S a
  Flock.Compose.compose
    { name := s!"unit/{a.key}", body := .tmpl t LAYOUT, ins := a.ins, outs, programDigests := [(a.key, digest)] }
    ((partition.getObjValAs? String "program").toOption.getD "") (Flock.hexOf (← Flock.HmRow.partitionDigest partition))

/-- E(x : Array n (Value 1), y : Value 1) = (x, Not y), at n = 3 and 2047: the one-stage path would read `y` from x's
padding and leave the second output's word without a row. -/
def E3 : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",3,["Value",1]]],["y",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,3]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",3,["Value",1]],["Value",1]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":3,"out":1},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,3]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,4]},"ret_type":["Tuple",[["Array",3,["Value",1]],["Value",1]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",3,["Value",1]],["Tuple",[["Array",3,["Value",1]],["Value",1]]]]}"#
def E2047 : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",2047,["Value",1]]],["y",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,2047]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",2047,["Value",1]],["Value",1]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2047,"out":1},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,2047]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,2048]},"ret_type":["Tuple",[["Array",2047,["Value",1]],["Value",1]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2047,["Value",1]],["Tuple",[["Array",2047,["Value",1]],["Value",1]]]]}"#

/-- C(a : Value 32, b : Value 1) = (a, Not b), which reaches no word primitive, so `Flock.Lift.lift` leaves `a` a word
(`Flock.Test.LowerBad.PASS`). -/
def C32 : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,1]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,2]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#

/-- `admitted` on unit 0 of the descriptor `d`, the query listing template `t`, at the template's cuts `cs`. -/
def admitE (d : String) (t : String := "E_v1") (cs : List (String × Nat) := []) : V Admit := do
  let P ← Flock.Program.decode d.toUTF8
  let q := Json.mkObj [("name", "Q_template_instance"), ("params", Json.mkObj [("template", t)])]
  admitted (Json.mkObj [("query", q)]) [(t, cs)] (← Spot.Restrict.restrict P #[0])

/-- `lowerBlock` on unit 0 of the descriptor `d`, its template `E_v1`. -/
def lowerE (d : String) : V ByteArray := do
  let CJ ← Flock.JsonParse.parse d
  let P ← Flock.Program.decode d.toUTF8
  let q := Json.mkObj [("name", "Q_template_instance"), ("params", Json.mkObj [("template", "E_v1")])]
  lowerBlock CJ (Json.mkObj [("query", q)]) [] (← Spot.Restrict.restrict P #[0]) (admittedBlock P #[0])

/-- Whether `r` is refused by `admitted`'s clause `clause` with detail `detail`. -/
def refusedBy {α : Type} (r : V α) (clause detail : String) : Bool :=
  match r with | .error e => e == s!"{NOT_ADMITTED}, {clause}: {detail}" | .ok _ => false

#guard refusedBy (lowerE E3) "whole-words" "C_S's template E_v1: input row x of 3 bits is not a positive whole number of u16 words"
#guard refusedBy (lowerE E2047) "whole-words" "C_S's template E_v1: input row x of 2047 bits is not a positive whole number of u16 words"
#guard refusedBy (admitE C32 "C_v1") "on-bits" "C_S's template C_v1 reaches C_v1, whose leaves are not all one bit after the lift to bits"
#guard refusedBy (admitE E3 "F_v1") "listed" "C_S's template E_v1 is not one the partition's query lists"
#guard refusedBy (admitE (E3.replace "[[\"x\"," "[[\"o0\",")) "port-names" "C_S's template E_v1: port o0 is named twice"
#guard refusedBy (admitE (E3.replace "[[\"x\"," "[[\"x:1\",")) "port-names" "C_S's template E_v1: parameter 'x:1' is empty or holds ',', ':', '#' or a space"
#guard refusedBy (admitE E3 "E_v1" [("z", 3)]) "cut-rows" "C_S's template E_v1: a cut names z, which is no parameter"
#guard refusedBy (admitE E3 "E_v1" [("x", 2)]) "cut-rows" "C_S's template E_v1: parameter x of 3 leaves cut 2 ways, not into equal runs of leaves of one width"
#guard refusedBy (admitE E3 "E_v1" [("x", 3)]) "whole-words" "C_S's template E_v1: input row x#0 of 1 bits is not a positive whole number of u16 words"

/-- E(x y : Array 16 (Value 1)) = (x, map Not y): two rows and two fields of one word each. -/
def E16 : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,16]},"type":1}],"axes":[0],"fn":1,"form":"batch","n":16,"out":1}],"params":[["x",["Array",16,["Value",1]]],["y",["Array",16,["Value",1]]]],"ret":{"concat":[{"affine":["p",0,0,16]},{"affine":["n",0,0,16]}]},"ret_type":["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[{"refs":{"affine":["n",0,0,16]},"type":1},{"refs":{"affine":["n",1,0,16]},"type":1}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,32]},"ret_type":["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",16,["Value",1]],["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]]]}"#

#guard match admitE E16 with
  | .ok a => a.key == "E_v1" && a.ins == #[("x", 16), ("y", 16)] && a.fields == #[("o0", 16), ("o1", 16)]
  | .error _ => false
#guard refusedBy (admitE E16 "E_v1" [("x", 2)]) "whole-words" "C_S's template E_v1: input row x#0 of 8 bits is not a positive whole number of u16 words"
-- `leaves-are-bits` on E16's `E_v1` and its lift, then on a lift with one more leaf in `x` and one in the return
#guard match Flock.Program.decode E16.toUTF8 with
  | .ok P => match P.index["E_v1"]? with
    | some d => match Flock.Lift.lift P d with
      | .ok (L, dl) => match P.defs[d]?, L.defs[dl]? with
        | some fn, some l =>
          leavesOff fn l == none &&
          leavesOff fn { l with params := l.params.set! 0 ("x", .array 17 (.value 1)) } ==
            some "parameter x of 16 bits has 17 leaves after the lift to bits" &&
          leavesOff fn { l with ret := .tuple #[.array 16 (.value 1), .array 17 (.value 1)] } ==
            some "return field o1 of 16 bits has 17 leaves after the lift to bits" &&
          leavesOff fn { l with outLeaves := 33 } ==
            some "parameters of 32 bits and a return of 32, its lift's 32 and 33 leaves"
        | _, _ => false
      | .error _ => false
    | none => false
  | .error _ => false

/-- E(x y : Array 16 (Value 1)) = (Not y[0], x): a return field of one bit, then another. -/
def F16 : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",16,["Value",1]]],["y",["Array",16,["Value",1]]]],"ret":{"concat":[{"affine":["n",0,0,1]},{"affine":["p",0,0,16]}]},"ret_type":["Tuple",[["Value",1],["Array",16,["Value",1]]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[{"refs":{"affine":["n",0,0,16]},"type":1},{"refs":{"affine":["n",1,0,16]},"type":1}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,17]},"ret_type":["Tuple",[["Value",1],["Array",16,["Value",1]]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",16,["Value",1]],["Tuple",[["Value",1],["Array",16,["Value",1]]]]]}"#

#guard refusedBy (lowerE F16) "whole-words" "C_S's template E_v1: return field o0 of 1 bits is not a positive whole number of u16 words"

/-- E16 with its input nodes registered values (`RegisteredInput1_v1`): its block's root opens with registered input
nodes, as a block reading theta does under the marking. -/
def E16R : String := E16.replace "Input1_v1" "RegisteredInput1_v1"

/-- `restrict`'s C_S for unit 0 of the descriptor `d`. -/
def restrictE (d : String) : V Flock.Program.Prog := do
  Spot.Restrict.restrict (← Flock.Program.decode d.toUTF8) #[0]

-- the door's steps (`restrictOn`) on E16R: `restrict`, whose C_S's root opens with a registered input node, `admitted`
-- with its lift, the lift and `blockTmpl` pass; `lowerBlock` (`lowerE`) refuses it as it refuses E16, whose inputs are
-- run inputs, at compose's row floor, a block this small; and `ported` refuses the same block by name
#guard match restrictE E16R with
  | .ok S => ((S.defs[S.root]?).bind fun r => r.body.nodes[0]? >>= fun x => S.defs[x.fn]?).map (fun f => (f.id, f.binding))
      == some ("RegisteredInput1_v1", some Flock.Program.REGISTRATION)
  | .error _ => false
#guard (admitE E16R).toBool
#guard (restrictE E16R >>= fun S => admitE E16R >>= fun a => Flock.Lift.lift S a.d).toBool
#guard (restrictE E16R >>= fun S => admitE E16R >>= fun a => blockTmpl S a).toBool
#guard match lowerE E16R, lowerE E16 with
  | .error e, .error e' => e == e' && e == "unit/E_v1: no row has a compression before its last block"
  | _, _ => false
#guard match Flock.Program.decode E16R.toUTF8 >>= (Spot.Restrict.portedOf · #[0]) with
  | .error e => e == "ported: the block reads a registered value (RegisteredInput1_v1), and a parameter does not carry \
      that it is registered: its rows would be committed afresh"
  | .ok _ => false

/-- `admitted` on unit 0 of the descriptor `d`, its template `E_v1` changed by `f` in C_S, at the lift `lift`. -/
def admitWith (d : String) (f : Flock.Program.Defn → Flock.Program.Defn := id)
    (lift : Flock.Program.Prog → Nat → V (Flock.Program.Prog × Nat) := Flock.Lift.lift) : V Admit := do
  let S ← Spot.Restrict.restrict (← Flock.Program.decode d.toUTF8) #[0]
  let some i := S.index["E_v1"]? | throw "E_v1 is no Definition of C_S"
  let q := Json.mkObj [("name", "Q_template_instance"), ("params", Json.mkObj [("template", "E_v1")])]
  admitted (Json.mkObj [("query", q)]) [] { S with defs := S.defs.modify i f } lift

/-- `Flock.Lift.lift` with one more leaf in the lifted template's first parameter. -/
def liftPlusLeaf (S : Flock.Program.Prog) (d : Nat) : V (Flock.Program.Prog × Nat) := do
  let (L, dl) ← Flock.Lift.lift S d
  pure ({ L with defs := L.defs.modify dl fun l =>
    { l with params := l.params.modify 0 fun (n, _) => (n, .array 17 (.value 1)) } }, dl)

#guard match admitWith E16 (lift := liftPlusLeaf) with
  | .error e => e == s!"{NOT_ADMITTED}, leaves-are-bits: C_S's template E_v1: parameter x of 16 bits has 17 leaves after the lift to bits"
  | .ok _ => false
#guard (admitWith E16).toOption.isSome

/-- C of one unit each of two templates, A(x : Array 16 (Value 1)) = map Not x and B the same. -/
def AB16 : String := r#"{"boolean":{},"callees":["Not_v1","Input1_v1","A_v0","B_v0"],"definitions":{"A_v0":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,16]},"type":0}],"axes":[0],"fn":0,"form":"batch","n":16,"out":0}],"params":[["x",["Array",16,["Value",1]]]],"ret":{"affine":["n",0,0,16]},"ret_type":["Array",16,["Value",1]]},"form":"composite","id":"A_v0"},"B_v0":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,16]},"type":0}],"axes":[0],"fn":0,"form":"batch","n":16,"out":0}],"params":[["x",["Array",16,["Value",1]]]],"ret":{"affine":["n",0,0,16]},"ret_type":["Array",16,["Value",1]]},"form":"composite","id":"B_v0"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[WiredPopulations_v1{templates=('A_v0','B_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1,1)}]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":1,"form":"batch","n":16,"out":0},{"args":[],"axes":[],"fn":1,"form":"batch","n":16,"out":0},{"args":[{"refs":{"affine":["n",0,0,16]},"type":1}],"axes":[0],"fn":2,"form":"batch","n":1,"out":1},{"args":[{"refs":{"affine":["n",1,0,16]},"type":1}],"axes":[0],"fn":3,"form":"batch","n":1,"out":1}],"params":[],"ret":{"concat":[{"affine":["n",2,0,16]},{"affine":["n",3,0,16]}]},"ret_type":["Tuple",[["Array",1,["Array",16,["Value",1]]],["Array",1,["Array",16,["Value",1]]]]]},"form":"composite","id":"Root[WiredPopulations_v1{templates=('A_v0','B_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1,1)}]_v1"}},"program_fn":"WiredPopulations_v1{templates=[\"A_v0\",\"B_v0\"],record=\"27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd\",counts=[1,1]}","root":"Root[WiredPopulations_v1{templates=('A_v0','B_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1,1)}]_v1","schema":"verity-ir/descriptor/v2","types":[["Array",16,["Value",1]],["Array",1,["Array",16,["Value",1]]]]}"#

/-- `admitted` on the block of units `us` of the descriptor `d`, under a `Q_template_instances` query listing `ts`, at
the cuts `cs`. -/
def admitIn (d : String) (ts : List String) (us : Array Nat) (cs : List (String × List (String × Nat)) := []) :
    V Admit := do
  let P ← Flock.Program.decode d.toUTF8
  let q := Json.mkObj [("name", "Q_template_instances"), ("params", Json.mkObj [("templates", Lean.toJson ts)])]
  admitted (Json.mkObj [("query", q)]) cs (← Spot.Restrict.restrict P us)

#guard refusedBy (admitIn AB16 ["A_v0", "B_v0"] #[0, 1]) "one-template" "C_S's root calls [A_v0, B_v0], and a block's units are instances of one template"
#guard (admitIn AB16 ["A_v0", "B_v0"] #[1]).toOption.map (·.key) == some "B_v0"

/-- AB16 with its second unit an `A_v0` reading the first's return: units 0 and 1 of one template, 1 reading 0. -/
def AA16 : String := r#"{"boolean":{},"callees":["Not_v1","Input1_v1","A_v0","B_v0"],"definitions":{"A_v0":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,16]},"type":0}],"axes":[0],"fn":0,"form":"batch","n":16,"out":0}],"params":[["x",["Array",16,["Value",1]]]],"ret":{"affine":["n",0,0,16]},"ret_type":["Array",16,["Value",1]]},"form":"composite","id":"A_v0"},"B_v0":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,16]},"type":0}],"axes":[0],"fn":0,"form":"batch","n":16,"out":0}],"params":[["x",["Array",16,["Value",1]]]],"ret":{"affine":["n",0,0,16]},"ret_type":["Array",16,["Value",1]]},"form":"composite","id":"B_v0"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[WiredPopulations_v1{templates=('A_v0','B_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1,1)}]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":1,"form":"batch","n":16,"out":0},{"args":[],"axes":[],"fn":1,"form":"batch","n":16,"out":0},{"args":[{"refs":{"affine":["n",0,0,16]},"type":1}],"axes":[0],"fn":2,"form":"batch","n":1,"out":1},{"args":[{"refs":{"affine":["n",2,0,16]},"type":1}],"axes":[0],"fn":2,"form":"batch","n":1,"out":1}],"params":[],"ret":{"concat":[{"affine":["n",2,0,16]},{"affine":["n",3,0,16]}]},"ret_type":["Tuple",[["Array",1,["Array",16,["Value",1]]],["Array",1,["Array",16,["Value",1]]]]]},"form":"composite","id":"Root[WiredPopulations_v1{templates=('A_v0','B_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1,1)}]_v1"}},"program_fn":"WiredPopulations_v1{templates=[\"A_v0\",\"B_v0\"],record=\"27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd\",counts=[1,1]}","root":"Root[WiredPopulations_v1{templates=('A_v0','B_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1,1)}]_v1","schema":"verity-ir/descriptor/v2","types":[["Array",16,["Value",1]],["Array",1,["Array",16,["Value",1]]]]}"#

/-- `lowerBlock` on the block of units `us` of the descriptor `d`, its template `t`, with the door's `admittedBlock`. -/
def lowerIn (d t : String) (us : Array Nat) : V ByteArray := do
  let P ← Flock.Program.decode d.toUTF8
  let q := Json.mkObj [("name", "Q_template_instance"), ("params", Json.mkObj [("template", t)])]
  lowerBlock (← Flock.JsonParse.parse d) (Json.mkObj [("query", q)]) [] (← Spot.Restrict.restrict P us) (admittedBlock P us)

/-- C's root `d` reading its input node 0 recut by 16. -/
def recut0 (d : Flock.Program.Defn) : Flock.Program.Defn := { d with body := { d.body with recut := #[(0, 16)] } }

/-- `admittedBlock` on the descriptor `d`'s units `us` at the template `t`, C's root first changed by `f`. -/
def formedIn (d t : String) (us : Array Nat) (f : Flock.Program.Defn → Flock.Program.Defn := id) : V Unit := do
  let P ← Flock.Program.decode d.toUTF8
  admittedBlock { P with defs := P.defs.modify P.root f } us t

#guard refusedBy (lowerIn AA16 "A_v0" #[0, 1]) "operands-outside" "unit 1 reads gate 32, which unit 0 of the block computes: a unit reads only gates outside its block"
#guard refusedBy (formedIn AB16 "A_v0" #[0, 1]) "one-callee" "unit 1 calls B_v0, not the block's template A_v0"
#guard refusedBy (formedIn AB16 "A_v0" #[0, 0]) "disjoint-ranges" "unit 0 is named twice in the block"
#guard refusedBy (formedIn AB16 "A_v0" #[0] recut0) "no-recut" "C's root reads input nodes [0] recut, in leaves of part of their width"
-- a unit whose node's callee is no Definition of C, which no decoded descriptor has
#guard match formedIn AB16 "A_v0" #[0] fun d => { d with body := { d.body with nodes := d.body.nodes.modify 0 fun x => { x with fn := 999 } } } with
  | .error e => e == s!"{NOT_ADMITTED}, one-callee: unit 0's callee is no Definition of C"
  | .ok _ => false
#guard (formedIn AB16 "A_v0" #[0]).toOption.isSome && (formedIn AA16 "A_v0" #[1]).toOption.isSome

/-- C of two batches of two units of one template, the second's stepped argument the first's return: units 2 and 3
read units 0 and 1. -/
def BB16 : String := r#"{"boolean":{},"callees":["Not_v1","Input1_v1","WallsT0_v1"],"definitions":{"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[WallsRoot_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":1,"form":"batch","n":16,"out":1},{"args":[],"axes":[],"fn":1,"form":"batch","n":16,"out":1},{"args":[{"refs":{"concat":[{"affine":["n",0,0,16]},{"affine":["n",1,0,16]}]},"type":2}],"axes":[0],"fn":2,"form":"batch","n":2,"out":2},{"args":[{"refs":{"affine":["n",2,0,32]},"type":2}],"axes":[0],"fn":2,"form":"batch","n":2,"out":2}],"params":[],"ret":{"affine":["n",3,0,32]},"ret_type":["Array",2,["Array",16,["Value",1]]]},"form":"composite","id":"Root[WallsRoot_v1]_v1"},"WallsT0_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0}],"fn":0,"form":"primitive","out":0}],"params":[["x0",["Array",16,["Value",1]]]],"ret":{"strided":["n",0,0,1,0,0,16]},"ret_type":["Array",16,["Value",1]]},"form":"composite","id":"WallsT0_v1"}},"program_fn":"WallsRoot_v1","root":"Root[WallsRoot_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",16,["Value",1]],["Array",2,["Array",16,["Value",1]]]]}"#

#guard refusedBy (lowerIn BB16 "WallsT0_v1" #[0, 1, 2, 3]) "operands-outside" "unit 2 reads gate 32, which unit 0 of the block computes: a unit reads only gates outside its block"
#guard (formedIn BB16 "WallsT0_v1" #[2, 3]).toOption.isSome

/-- `E16` returning `Value 16` named by one leaf of `map Not y`: a template wide on its return alone. -/
def R16 : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,16]},"type":1}],"axes":[0],"fn":1,"form":"batch","n":16,"out":1}],"params":[["x",["Array",16,["Value",1]]],["y",["Array",16,["Value",1]]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",16]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[{"refs":{"affine":["n",0,0,16]},"type":1},{"refs":{"affine":["n",1,0,16]},"type":1}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,1]},"ret_type":["Value",16]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",16,["Value",1]],["Value",16]]}"#

/-- N(a : Value 16) = Inc16 a, a word primitive with no Boolean body: a template whose lift to bits would refuse. -/
def N16 : String := r#"{"annotations":{},"boolean":{},"callees":["Input16_v1","Inc16_v1","X_v1"],"definitions":{"Inc16_v1":{"form":"primitive","id":"Inc16_v1","params":[["a",["Value",16]]],"ret":["Value",16]},"Input16_v1":{"form":"primitive","id":"Input16_v1","params":[],"ret":["Value",16]},"Root[XProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0}],"fn":2,"form":"call","out":0}],"params":[],"ret":{"affine":["n",1,0,1]},"ret_type":["Value",16]},"form":"composite","id":"Root[XProg_v1]_v1"},"X_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["a",["Value",16]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",16]},"form":"composite","id":"X_v1"}},"program_fn":"XProg_v1","root":"Root[XProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",16]]}"#

/-- Whether the decoder refuses the descriptor `d` with `why`: what no template `admitted` reads reaches. -/
def undecoded (d why : String) : Bool :=
  match Flock.Program.decode d.toUTF8 with | .error e => e == why | .ok _ => false

#guard undecoded R16 "definition 'E_v1': E_REINTERPRET: the body's return: leaf 0 is typed Value<16> but names leaf 0 of node 0, a Value<1>: a silent reinterpretation. A word is read only at the width it was committed with; a runtime word is split into limbs only by a registered LimbSplit Definition, which computes the limbs and their carries"
#guard undecoded N16 "E_BODY_MISSING: the word primitives [Inc16_v1] have no Boolean body in `boolean`"

-- `C32`'s `C_v1` kept to its bit parameter, its return still a word: `on-bits` refuses it on its return alone, though
-- no descriptor decodes to it (`R16`)
#guard match Flock.Program.decode C32.toUTF8 with
  | .ok P => match P.index["C_v1"]? with
    | some i => match P.defs[i]? with
      | some d =>
        let d' := { d with params := d.params.filter (·.2.bits) }
        d'.params.size == 1 && wideFrom { P with defs := P.defs.set! i d' } i == #["C_v1"]
      | none => false
    | none => false
  | .error _ => false

/-- T(x : Array 2 (Value 16)) = (W x[0] x[1], W{IR="boolean"} x[1] x[0]), W a word composite: its lift's key is a
Definition of C already, so the lift to bits refuses T. -/
def COL : String := r#"{"boolean":{"CollideXor16_v1":"StandIn[CollideXor16]_v1"},"callees":["CollideXor16_v1","W_v0","W_v0{IR=\"boolean\"}","Input16_v1","T_v0","Xor2_v1"],"definitions":{"CollideXor16_v1":{"form":"primitive","id":"CollideXor16_v1","params":[["a",["Value",16]],["b",["Value",16]]],"ret":["Value",16]},"Input16_v1":{"form":"primitive","id":"Input16_v1","params":[],"ret":["Value",16]},"Root[WiredPopulations_v1{templates=('T_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1)}]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":3,"form":"batch","n":2,"out":1},{"args":[{"refs":{"affine":["n",0,0,2]},"type":2}],"axes":[0],"fn":4,"form":"batch","n":1,"out":3}],"params":[],"ret":{"affine":["n",1,0,2]},"ret_type":["Tuple",[["Array",1,["Tuple",[["Value",16],["Value",16]]]]]]},"form":"composite","id":"Root[WiredPopulations_v1{templates=('T_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1)}]_v1"},"StandIn[CollideXor16]_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"explicit":[["p",0,0],["p",1,0]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,1],["p",1,1]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,2],["p",1,2]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,3],["p",1,3]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,4],["p",1,4]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,5],["p",1,5]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,6],["p",1,6]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,7],["p",1,7]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,8],["p",1,8]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,9],["p",1,9]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,10],["p",1,10]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,11],["p",1,11]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,12],["p",1,12]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,13],["p",1,13]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,14],["p",1,14]]},"type":4}],"fn":5,"form":"primitive","out":5},{"args":[{"refs":{"explicit":[["p",0,15],["p",1,15]]},"type":4}],"fn":5,"form":"primitive","out":5}],"params":[["a",["Array",16,["Value",1]]],["b",["Array",16,["Value",1]]]],"ret":{"concat":[{"affine":["n",0,0,1]},{"affine":["n",1,0,1]},{"affine":["n",2,0,1]},{"affine":["n",3,0,1]},{"affine":["n",4,0,1]},{"affine":["n",5,0,1]},{"affine":["n",6,0,1]},{"affine":["n",7,0,1]},{"affine":["n",8,0,1]},{"affine":["n",9,0,1]},{"affine":["n",10,0,1]},{"affine":["n",11,0,1]},{"affine":["n",12,0,1]},{"affine":["n",13,0,1]},{"affine":["n",14,0,1]},{"affine":["n",15,0,1]}]},"ret_type":["Array",16,["Value",1]]},"form":"composite","id":"StandIn[CollideXor16]_v1"},"T_v0":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0},{"refs":{"affine":["p",0,1,1]},"type":0}],"fn":1,"form":"call","out":0},{"args":[{"refs":{"affine":["p",0,0,1]},"type":0},{"refs":{"affine":["p",0,1,1]},"type":0}],"fn":2,"form":"call","out":0}],"params":[["x",["Array",2,["Value",16]]]],"ret":{"concat":[{"affine":["n",0,0,1]},{"affine":["n",1,0,1]}]},"ret_type":["Tuple",[["Value",16],["Value",16]]]},"form":"composite","id":"T_v0"},"W_v0":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0},{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":0,"form":"primitive","out":0}],"params":[["a",["Value",16]],["b",["Value",16]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",16]},"form":"composite","id":"W_v0"},"W_v0{IR=\"boolean\"}":{"bindings":{"IR":"boolean"},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0},{"refs":{"affine":["p",0,0,1]},"type":0}],"fn":0,"form":"primitive","out":0}],"params":[["a",["Value",16]],["b",["Value",16]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",16]},"form":"composite","id":"W_v0"},"Xor2_v1":{"form":"primitive","id":"Xor2_v1","params":[["x",["Array",2,["Value",1]]]],"ret":["Value",1]}},"program_fn":"WiredPopulations_v1{templates=[\"T_v0\"],record=\"27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd\",counts=[1]}","root":"Root[WiredPopulations_v1{templates=('T_v0'),record='27c74670adb75075fad058d5ceaf7b20c4e7786c83bae8a32f626f9782af34c9a33c2046ef60fd2a7878d378e29fec851806bbd9a67878f3a9f1cda4830763fd',counts=(1)}]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",16],["Array",2,["Value",16]],["Array",1,["Array",2,["Value",16]]],["Array",1,["Tuple",[["Value",16],["Value",16]]]],["Array",2,["Value",1]],["Value",1]]}"#

#guard refusedBy (admitE COL "T_v0") "on-bits" "the lift to bits refuses C_S's template T_v0"

/-- E(x y : Array 16 (Value 1)) = (), a template that returns no field. -/
def NF16 : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,16]},"type":1}],"axes":[0],"fn":1,"form":"batch","n":16,"out":1}],"params":[["x",["Array",16,["Value",1]]],["y",["Array",16,["Value",1]]]],"ret":{"concat":[]},"ret_type":["Tuple",[]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[{"refs":{"affine":["n",0,0,16]},"type":1},{"refs":{"affine":["n",1,0,16]},"type":1}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"concat":[]},"ret_type":["Tuple",[]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",16,["Value",1]],["Tuple",[]]]}"#

#guard refusedBy (lowerE NF16) "whole-words" "C_S's template E_v1 returns no field"

/-- X(p : (Value 1, Value 16)) = (Not p.0, W16 p.1) reading p.0 from A(a : Value 1) = Not a, W16 a word primitive with
a Boolean body on bits: in(S)'s cuts give X's p two rows of one leaf each, of two widths. -/
def MIX : String := r#"{"annotations":{},"boolean":{"W16_v1":"W16B_v1"},"callees":["Input1_v1","Input16_v1","Not_v1","W16_v1","W16B_v1","A_v1","X_v1"],"definitions":{"A_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0}],"fn":2,"form":"primitive","out":0}],"params":[["a",["Value",1]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",1]},"form":"composite","id":"A_v1"},"Input16_v1":{"form":"primitive","id":"Input16_v1","params":[],"ret":["Value",16]},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[XProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0}],"fn":5,"form":"call","out":0},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"concat":[{"affine":["n",1,0,1]},{"affine":["n",2,0,1]}]},"type":3}],"fn":6,"form":"call","out":3}],"params":[],"ret":{"affine":["n",3,0,2]},"ret_type":["Tuple",[["Value",1],["Value",16]]]},"form":"composite","id":"Root[XProg_v1]_v1"},"W16B_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,16]},"type":2}],"axes":[0],"fn":2,"form":"batch","n":16,"out":2}],"params":[["a",["Array",16,["Value",1]]]],"ret":{"affine":["n",0,0,16]},"ret_type":["Array",16,["Value",1]]},"form":"composite","id":"W16B_v1"},"W16_v1":{"form":"primitive","id":"W16_v1","params":[["a",["Value",16]]],"ret":["Value",16]},"X_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0}],"fn":2,"form":"primitive","out":0},{"args":[{"refs":{"affine":["p",0,1,1]},"type":1}],"fn":3,"form":"primitive","out":1}],"params":[["p",["Tuple",[["Value",1],["Value",16]]]]],"ret":{"concat":[{"affine":["n",0,0,1]},{"affine":["n",1,0,1]}]},"ret_type":["Tuple",[["Value",1],["Value",16]]]},"form":"composite","id":"X_v1"}},"program_fn":"XProg_v1","root":"Root[XProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Value",16],["Array",16,["Value",1]],["Tuple",[["Value",1],["Value",16]]]]}"#

#guard refusedBy (admitIn MIX ["A_v1", "X_v1"] #[1] [("X_v1", [("p", 2)])]) "cut-rows" "C_S's template X_v1: parameter p of 2 leaves cut 2 ways, not into equal runs of leaves of one width"
#guard refusedBy (admitIn MIX ["A_v1", "X_v1"] #[1]) "whole-words" "C_S's template X_v1: input row p of 17 bits is not a positive whole number of u16 words"

/-- E(x y : Array 16 (Value 1)) = (x, sixteen calls of Zero1), Zero1 a word primitive whose Boolean body is the input
primitive `Input1_v1`, to which the evaluator gives no value. -/
def IB16 : String := r#"{"annotations":{},"boolean":{"Zero1_v1":"Input1_v1"},"callees":["Input1_v1","Zero1_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":1,"form":"batch","n":16,"out":1}],"params":[["x",["Array",16,["Value",1]]],["y",["Array",16,["Value",1]]]],"ret":{"concat":[{"affine":["p",0,0,16]},{"affine":["n",0,0,16]}]},"ret_type":["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[],"axes":[],"fn":0,"form":"batch","n":16,"out":1},{"args":[{"refs":{"affine":["n",0,0,16]},"type":1},{"refs":{"affine":["n",1,0,16]},"type":1}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,32]},"ret_type":["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]]},"form":"composite","id":"Root[EProg_v1]_v1"},"Zero1_v1":{"form":"primitive","id":"Zero1_v1","params":[],"ret":["Value",1]}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",16,["Value",1]],["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]]]}"#

/-- What refuses `IB16`: the decoder (`Descriptor.inputBody`), before `admitted` and the lowering. -/
def INPUT_BODY : String :=
  "definition 'E_v1' node 0: an input node (Input1_v1, the Boolean body of Zero1_v1) appears only in the root's body"

#guard match lowerE IB16 with | .error e => e == INPUT_BODY | .ok _ => false

/-- `E16` with its two input batches one batch of two 16-bit leaves, which C's root reads in 1-bit leaves (recut). -/
def RC16 : String := r#"{"annotations":{},"boolean":{},"callees":["Input16_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,16]},"type":1}],"axes":[0],"fn":1,"form":"batch","n":16,"out":1}],"params":[["x",["Array",16,["Value",1]]],["y",["Array",16,["Value",1]]]],"ret":{"concat":[{"affine":["p",0,0,16]},{"affine":["n",0,0,16]}]},"ret_type":["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]]},"form":"composite","id":"E_v1"},"Input16_v1":{"form":"primitive","id":"Input16_v1","params":[],"ret":["Value",16]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2,"out":4},{"args":[{"refs":{"affine":["n",0,0,16]},"type":1},{"refs":{"affine":["n",0,16,16]},"type":1}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",1,0,32]},"ret_type":["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",16,["Value",1]],["Tuple",[["Array",16,["Value",1]],["Array",16,["Value",1]]]],["Value",16],["Array",2,["Value",16]]]}"#

-- refused before `admittedBlock`'s `no-recut`, by the gate layout `Spot.Restrict.restrict` reads (`Circuits.Program.recutGates`)
#guard match lowerE RC16 with
  | .error e => e == "recut-gates: input nodes [0] are read in leaves of part of their width, which the gate layout doesn't lay out"
  | .ok _ => false

end Spot.OneStageZk
