/-
V*'s statements over one inner session, checked by the verifier against V*'s own executable Lean (`VStar`, in this
package so that the verifier can read it): `VStar.statements` at the inner session's shape (`VStar/Stage.lean:48-58`: one
statement per Ligerito level of `fast100(m)`, `rec-open/l<2^lanes>-h<d − cap>` over that level's `2 q` openings, then one
per part of the rep's algebra, `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` over the two reps). A circuit whose unit is one
of V*'s is refused unless the verifier names which of these statements the session is (`--rec-part J`) and its unit is that
statement's, its population is that statement's instance count, the verifier's own bindings (`--rec-bindings F`) are its
partition object's and program's, META binds exactly those, and the circuit file is byte for byte the one `vstar stage`
writes for that statement under the verifier's bindings (`circuitFile`).

The inner shape is the verifier's own (`rec-inner`): `Stmt.setupTables` on the inner statement, as `InnerFold.lean` sets it
up for V*'s native side, read once, since the inner circuit can be tens of GB; `verify --rec-inner F` reads the line it
printed.
-/
import Flock.HmRow
import Flock.Registered
import VStar.Cli
import VStar.Compose
import VStar.Stage

namespace Flock
namespace RecVStar

open Lean

/-- The inner session's shape: its table's `m` and `k_log` (`Setup.ofCircuit`), its claims, two and one per region and
link point (`Stmt.extraClaims`, `rec_vstage.verifier`'s `2 + len(extras)`), and its table's regions (`Stmt.regions`). -/
structure Inner where
  m : Nat
  kLog : Nat
  claims : Nat
  regions : Nat
  deriving Repr, BEq, Inhabited

/-- A recursive session's inner statement (§16.16) on the verifier's live coins, as `87-rec-private.sh` runs `inner-fold`. -/
def innerTags : Tags := Tags.circuitPlainLeaves.withCoins Coins.os

/-- The inner statement's shape, from its circuit, public file and the verifier's own partition object and program. -/
def innerOf (circuitB pubB : ByteArray) (partition : Option Json) (program : Option ByteArray) : V Inner := do
  let (spec, sts) ← Stmt.setupTables innerTags circuitB pubB Coins.os 1 (partition := partition) (program := program)
  let some st := sts[0]? | throw "rec-inner: the inner statement has no table"
  pure ⟨st.m, st.c.kLog, 2 + spec.points * st.regions.size, st.regions.size⟩

def FORMAT : String := "verity/rec-inner/v1"

def Inner.toJson (i : Inner) : Json :=
  Json.mkObj [("format", .str FORMAT), ("m", Lean.toJson i.m), ("k_log", Lean.toJson i.kLog), ("claims", Lean.toJson i.claims),
    ("regions", Lean.toJson i.regions)]

def Inner.ofJson (j : Json) : Except String Inner := do
  if (j.getObjValAs? String "format").toOption != some FORMAT then throw s!"rec-inner: not a {FORMAT} object"
  pure ⟨← j.getObjValAs? Nat "m", ← j.getObjValAs? Nat "k_log", ← j.getObjValAs? Nat "claims",
    ← j.getObjValAs? Nat "regions"⟩

/-- V*'s statements at the inner shape (`VStar.statements`), at a shape `RecResiduals.check` accepts a part's name at,
whose claims are two and two per region of the inner statement (an extra claim per region at each of two link points), and
where V*'s parts (`VStar.parts (spec vsh)`, `VStar/Parts.lean`) are as many as the verifier's own packing of its algebra
(`RecResiduals.groups (structureOf sh) (portSizes sh)`, `RecResiduals.lean:119-139`). -/
def statements (i : Inner) : Except String (Array VStar.Stmt) := do
  let some sh := RecResiduals.shape? i.m i.kLog i.claims
    | throw (s!"the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}) is not V*'s: a fast100 schedule, " ++
        "k_log at least 6, claims at least 2")
  if i.claims != 2 + 2 * i.regions then
    throw s!"the inner session's claims {i.claims}, not two and two per region of its {i.regions} ({2 + 2 * i.regions})"
  let vsh : VStar.Algebra.Shape := { m := i.m, kLog := i.kLog, claims := i.claims }
  let ps ← (do VStar.parts (← VStar.Algebra.spec vsh)).mapError (s!"V*'s parts: " ++ ·)
  let gs ← (do RecResiduals.groups (← RecAlgebra.structureOf sh) (RecResiduals.portSizes sh)).mapError
    (s!"the verifier's groups: " ++ ·)
  if ps.size != gs.size then
    throw s!"at the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}): V* has {ps.size} parts, the verifier's algebra {gs.size} groups"
  VStar.statements vsh

def isVStar (name : String) : Bool := name.startsWith "rec-open/" || name.startsWith "rec-residuals/"

/-- A unit named `name` over a population of `n` instances is one of V*'s statements at the inner shape `i`: a part's name
carries `i`'s `m`, `k_log` and claims, exactly one statement has the name, and `n` is its instance count. With `part = some
j`, it is statement `j` of `statements i` (`--rec-part J`: levels first, in level order, then the algebra's parts, in part
order, as `rec_vstage.stage` stages `L<level>/` and `alg/p<j>/`), so a session the verifier holds for statement `j` is
refused another statement's file. -/
def check (i : Inner) (name : String) (n : Nat) (part : Option Nat := none) : V VStar.Stmt := do
  if let some (m, k, c, _) := RecResiduals.params name then
    if m != i.m || k != i.kLog || c != i.claims then
      throw s!"{name}: shape (m {m}, k_log {k}, claims {c}), the inner session's (m {i.m}, k_log {i.kLog}, claims {i.claims})"
  let ss ← statements i
  match ss.toList.filter (·.name == name) with
  | [] =>
    throw s!"{name}: not one of V*'s statements at the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}): {ss.map (·.name)}"
  | [s] =>
    if n != s.instances then throw s!"{name}: instances {n}, statement says {s.instances}"
    if let some j := part then
      match ss[j]? with
      | none => throw s!"{name}: the session is V*'s statement {j} (--rec-part), but V* has {ss.size} statements at the inner session's shape"
      | some t => if t.name != name then throw s!"{name}: the session is V*'s statement {j} (--rec-part), {t.name}"
    pure s
  | ss => throw s!"{name}: {ss.length} of V*'s statements have this name"

/-- **V*'s circuit file of a statement**: the bytes `vstar stage` writes to `<dir>/circuit.txt` (`VStar.Cli.write`,
`VStar/Cli.lean:38-47`): the statement's unit, laid out, emitted with its subcircuit and the statement's bindings
(`VStar.emitCircuit`). -/
def circuitFile (s : VStar.Stmt) (b : VStar.Bindings) : Except String ByteArray := do
  let u ← s.unit
  let l ← u.laid
  let e ← VStar.emitCircuit (m := StateM String) (fun x => modify (· ++ x)) u l s.sub b
  pure (Id.run (e.run "")).2.toUTF8

/-- The first position at which two byte strings differ, if they do: the byte and its line (1-based). -/
def firstDiff? (a b : ByteArray) : Option (Nat × Nat) := Id.run do
  let mut line := 1
  for k in [0:min a.size b.size] do
    if a[k]! != b[k]! then return some (k, line)
    if a[k]! == 10 then line := line + 1
  if a.size == b.size then none else some (min a.size b.size, line)

/-- The circuit file is V*'s for `s` under the bindings `b`, the verifier's own (`circuitFile`). -/
def checkFile (s : VStar.Stmt) (b : VStar.Bindings) (circuitB : ByteArray) : V Unit := do
  let f ← match circuitFile s b with
    | .ok f => pure f
    | .error e => throw s!"{s.name}: V*'s circuit file: {e}"
  if let some (k, line) := firstDiff? circuitB f then
    throw s!"{s.name}: the circuit file ({circuitB.size} bytes) is not V*'s ({f.size} bytes): they differ at byte {k}, line {line}"

/-- A statement's population: the registered population's instances for a drawn session, else its own. -/
def population (st : Stmt) : Nat := (st.pub.population.map (·.1)).getD st.n

/-- META's bindings (`VStar.Bindings`, `VStar/Compose.lean:52-55`, written at `Compose.lean:228-229`): the
`verity/partition/v1` digest, `program_sha512` and `program_digests`. -/
def bindingsOf (c : Circuit) : V VStar.Bindings := do
  let name := c.unit.name
  let part := (c.metaJ.getObjVal? "partition").toOption.getD Json.null
  let (some "verity/partition/v1", some digest) := ((part.getObjValAs? String "rule").toOption,
      (part.getObjValAs? String "digest").toOption)
    | throw s!"{name}: META partition is not a verity/partition/v1 digest"
  let some program := (c.metaJ.getObjValAs? String "program_sha512").toOption
    | throw s!"{name}: META names no program_sha512"
  let some pd := (c.metaJ.getObjVal? "program_digests").toOption | throw s!"{name}: META names no program_digests"
  pure ⟨digest, program, pd⟩

/-- The verifier's own bindings of a statement of V*'s (`--rec-bindings F`): `vstar`'s `BINDINGS` object, `{"partition",
"program", "program_digests"}` (`VStar.Cli.bindingsOf`), which the auditor computes from the verifier's partition object and
program (`verity_flock.rec_bindings`), never from the prover's circuit file. -/
def ownBindingsOfJson (j : Json) : Except String VStar.Bindings :=
  (VStar.Cli.bindingsOf j).mapError (s!"rec-bindings: " ++ ·)

/-- The verifier's own bindings `f` are its own partition object's and program's: `f`'s partition digest is the object's
(`HmRow.partitionDigest`), `f`'s program and the object's `program` are the program's digest (the SHA-512 of its canonical
descriptor without `annotations`, `TemplateQuery.Desc.parse`), and `f`'s `program_digests` names one Definition, the
template the object's query names. Its digest is not recomputed: that is core's codec's digest of the one-Definition
program, which this verifier does not encode; the auditor's `verity_flock.rec_bindings` computes it from the same files. -/
def checkOwnBindings (name : String) (f : VStar.Bindings) (obj : Json) (prog : ByteArray) : V Unit := do
  let own := hexOf (← HmRow.partitionDigest obj)
  if f.partition != own then throw s!"{name}: the verifier's bindings' partition digest {f.partition}, its partition object's {own}"
  let ps := hexOf (← TemplateQuery.Desc.parse prog).sha512
  if f.program != ps then throw s!"{name}: the verifier's bindings' program {f.program}, its program's {ps}"
  let op := (obj.getObjValAs? String "program").toOption.getD ""
  if op != ps then throw s!"{name}: the partition object's program {op}, the verifier's program's {ps}"
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let t := ((q.getObjVal? "params").toOption.getD Json.null |>.getObjValAs? String "template").toOption.getD ""
  match f.programDigests with
  | .obj kvs =>
    match kvs.toArray.map (·.1) with
    | #[k] => if k != t then throw s!"{name}: the verifier's bindings' program_digests names {k}, the partition's template {t}"
    | ks => throw s!"{name}: the verifier's bindings' program_digests names {ks.size} Definitions, not the partition's template {t} alone"
  | _ => throw s!"{name}: the verifier's bindings' program_digests is not an object"

/-- **META binds the verifier's own bindings**: META's partition digest and `program_sha512` are `f`'s, and META's
`program_digests` is `f`'s as a value, compared as RFC 8785 canonical JSON (`canon`), byte for byte. -/
def checkMeta (c : Circuit) (f : VStar.Bindings) : V Unit := do
  let name := c.unit.name
  let b ← bindingsOf c
  if b.partition != f.partition then throw s!"{name}: META partition digest {b.partition}, the verifier's bindings' {f.partition}"
  if b.program != f.program then throw s!"{name}: META program_sha512 {b.program}, the verifier's bindings' {f.program}"
  let (mc, fc) := (← canon b.programDigests, ← canon f.programDigests)
  if mc != fc then throw s!"{name}: META program_digests {mc}, the verifier's bindings' {fc}"

/-- `checkMeta` of the circuit `c`, parsed from the circuit file `circuitB`, against the verifier's own bindings `f`, then
`check` on its unit over `n` instances at the inner shape `i` as V*'s statement `part`, and `checkFile` on `circuitB` under
`f`. -/
def checkCircuit (i : Inner) (part : Nat) (c : Circuit) (n : Nat) (circuitB : ByteArray) (f : VStar.Bindings) :
    V VStar.Stmt := do
  checkMeta c f
  let s ← check i c.unit.name n (some part)
  checkFile s f circuitB
  pure s

/-- `m<k>`'s `k`. -/
def mIndex (name : String) : Option Nat := if name.startsWith "m" then (name.drop 1).toString.toNat? else none

/-- `rec-c<k>`'s `k`. -/
def cIndex (value : String) : Option Nat := if value.startsWith "rec-c" then (value.drop 5).toString.toNat? else none

/-- A reads file's message-round ports, those named `m<k>` or registering a `rec-c<k>`: each one's name, value and reads. -/
def msgPorts (ports : Array Registered.Port) : List (String × String × Array (Option Nat)) :=
  (ports.toList.filter fun p => (mIndex p.name).isSome || (cIndex p.value).isSome).map fun p => (p.name, p.value, p.reads)

/-- A statement's message-round ports: a part's ports `m<k>` (`VStar/Algebra.lean:385-387`), each under `rec-c<k>` read
at rows `0` and `1`, the two reps' (`rec_residuals.write`, `rec_residuals.py:452-454`); a level has none. -/
def ownPorts : VStar.Stmt → List (String × String × Array (Option Nat))
  | .inner _ _ p => p.S.ports.filterMap fun (nm, _) => (mIndex nm).map fun k => (nm, s!"rec-c{k}", #[some 0, some 1])
  | .recOpen .. => []

def showPort (q : String × String × Array (Option Nat)) : String :=
  s!"{q.1} under {q.2.1} reading {q.2.2.toList.map fun r => (r.map toString).getD "null"}"

/-- **The verifier's reads file has the statement's message-round ports**: `msgPorts` of the `--registered` file and
`ownPorts` of the statement hold the same ports, compared by membership (`Registered.ofJson` lists a file's ports in its
object's key order). -/
def checkOwn (s : VStar.Stmt) (registered : Array Registered.Port) : V Unit := do
  let file := msgPorts registered
  let own := ownPorts s
  if let some q := file.find? (!own.contains ·) then
    let theirs := match own.find? (·.1 == q.1) with
      | some o => s!"the statement's is {showPort o}"
      | none => s!"the statement has no port {q.1} ({own.length} message-round ports)"
    throw s!"{s.name}: the registered reads' port {showPort q} is not the statement's: {theirs}"
  if let some q := own.find? (!file.contains ·) then
    throw s!"{s.name}: the statement's message-round port {showPort q} is not in the registered reads ({file.length} message-round ports)"

/-- With an inner shape, the statement must be V*'s: `tags`, the statement the verifier set it up as (`--statement`), is
V*'s own `Tags.circuit`, compared by name (`Tags.find?` gives each name one statement); it needs the verifier's own
partition object and program, which of V*'s statements the session is (`part`, `--rec-part J`) and the verifier's own
bindings (`bindings`, `--rec-bindings F`); then `checkOwnBindings` of those bindings against the partition object and
program, `checkCircuit` on the statement's circuit and population as statement `part` under those bindings, and
`checkOwn` on the verifier's own registered reads. -/
def checkStmt (inner : Option Inner) (tags : Tags) (st : Stmt) (circuitB : ByteArray) (partition : Option Json := none)
    (program : Option ByteArray := none) (registered : Option (Array Registered.Port) := none) (part : Option Nat := none)
    (bindings : Option VStar.Bindings := none) : V Unit :=
  let name := st.c.unit.name
  match inner with
  | some i => do
    if tags.name != Tags.circuit.name then
      throw s!"{name}: one of V*'s statements, which are {Tags.circuit.name} statements (--statement {Tags.circuit.name}), not {tags.name}"
    let (some obj, some prog) := (partition, program)
      | throw s!"{name}: one of V*'s statements; pass the verifier's own partition object and program (--partition F, --program F)"
    let some j := part
      | throw s!"{name}: one of V*'s statements; pass which of V*'s statements this session is (--rec-part J, its index in rec-statements' list)"
    let some f := bindings
      | throw s!"{name}: one of V*'s statements; pass the verifier's own bindings (--rec-bindings F, from verity_flock.rec_bindings)"
    checkOwnBindings name f obj prog
    let s ← checkCircuit i j st.c (population st) circuitB f
    if let some r := registered then checkOwn s r
  | none =>
    if isVStar name then
      throw s!"{name}: one of V*'s statements; pass the verifier's own inner shape (--rec-inner F, from rec-inner)"
    else pure ()

end RecVStar
end Flock
