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

/-- A statement's row ports: a part's ports (`VStar/Algebra.lean:385-387`: its `m<k>`, `acc0`, `acc1`), not its `v` and
`out`, which are its own (`rec_residuals.write`, rec_residuals.py:451-461); a level has none. -/
def rowPorts : VStar.Stmt → List String
  | .inner _ _ p => p.S.ports.map (·.1)
  | .recOpen .. => []

def showReads (r : Array (Option Nat)) : String := toString (r.toList.map fun x => (x.map toString).getD "null")

/-- The first field in which two registered ports differ, with both values: the registration (value, root, leaf count,
row width, domain) and, with `reads`, the read positions; none if they agree. -/
def portDiff? (a b : Registered.Port) (reads : Bool := true) : Option String :=
  let w := fun (x : Option Nat) => (x.map toString).getD "none"
  if a.value != b.value then some s!"value {a.value}, against {b.value}"
  else if a.root != b.root then some s!"root {hexOf a.root}, against {hexOf b.root}"
  else if a.leaves != b.leaves then some s!"leaves {a.leaves}, against {b.leaves}"
  else if a.words != b.words then some s!"words {w a.words}, against {w b.words}"
  else if a.wordBits != b.wordBits then some s!"word_bits {a.wordBits}, against {b.wordBits}"
  else if a.domain != b.domain then some s!"domain {hexOf a.domain}, against {hexOf b.domain}"
  else if reads && a.reads != b.reads then some s!"reads {showReads a.reads}, against {showReads b.reads}"
  else none

/-- **The verifier's reads file names a part's row ports, each the family's one entry of that name** (`OwnNamesRows`,
`PartPortsOne`): every row port of the statement (`rowPorts`) is a port of `--registered`, and agrees in every field
(`portDiff?`: value, root, leaf count, row width, domain and reads) with the port of that name in the verifier's record of
V*'s shared rows (`--rec-rows`), one file for every session of the family. -/
def checkRows (s : VStar.Stmt) (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) : V Unit := do
  let names := rowPorts s
  if let some nm := names.find? fun nm => !registered.any (·.name == nm) then
    throw s!"{s.name}: its row port {nm} is not a port of the registered reads ({registered.toList.map (·.name)})"
  if names.isEmpty then return
  let some rows := rows
    | throw s!"{s.name}: one of V*'s parts; pass the verifier's record of V*'s shared rows (--rec-rows F), which each of its row ports must be"
  for nm in names do
    let some reg := registered.find? (·.name == nm) | throw s!"{s.name}: its row port {nm} is not a port of the registered reads"
    let some want := rows.find? (·.name == nm)
      | throw s!"{s.name}: its row port {nm} is not a port of the verifier's shared rows (--rec-rows: {rows.toList.map (·.name)})"
    if let some d := portDiff? reg want then
      throw s!"{s.name}: its row port {nm} is not the verifier's shared row {nm} (--rec-rows): {d}"

def ACC_VALUE : String := "rec-acc"

/-- A statement's ports that read `rec-acc`: a level's `acc` and `acc_out` (`rec_outer.stage`, rec_outer.py:236), a
part's `acc0` and `acc1` among its ports (rec_vstage.py:357). -/
def accPorts : VStar.Stmt → List String
  | .inner _ _ p => (p.S.ports.map (·.1)).filter fun nm => nm == "acc0" || nm == "acc1"
  | .recOpen .. => ["acc", "acc_out"]

/-- **A session's `rec-acc` reads are one registration, the verifier's** (the `site` of `Family/AccRows.lean`: value,
root, leaf count, row width and domain; not name or reads): each of the statement's `accPorts` is a port of `--registered`
reading `rec-acc`, every port there reading `rec-acc` has the registration of the first, and that is the registration of
`acc0` in the verifier's shared rows (`--rec-rows`), one file for every session of the family, whose `rec-acc` ports all
have it. -/
def checkAcc (s : VStar.Stmt) (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) : V Unit := do
  for nm in accPorts s do
    match registered.find? (·.name == nm) with
    | some p => if p.value != ACC_VALUE then throw s!"{s.name}: its port {nm} reads {p.value}, not {ACC_VALUE}"
    | none => throw s!"{s.name}: its port {nm} is not a port of the registered reads ({registered.toList.map (·.name)})"
  let accs := registered.filter (·.value == ACC_VALUE)
  let some a := accs[0]? | return
  for b in accs do
    if let some d := portDiff? a b false then
      throw s!"{s.name}: its {ACC_VALUE} ports {a.name} and {b.name} are two registrations: {d}"
  let some rows := rows
    | throw s!"{s.name}: it reads {ACC_VALUE}; pass the verifier's record of V*'s shared rows (--rec-rows F), whose {ACC_VALUE} registration it must be"
  let some w := rows.find? (·.name == "acc0")
    | throw s!"{s.name}: the verifier's shared rows (--rec-rows) have no acc0, whose {ACC_VALUE} registration it must be"
  if w.value != ACC_VALUE then throw s!"the verifier's shared rows (--rec-rows): acc0 reads {w.value}, not {ACC_VALUE}"
  for b in rows.filter (·.value == ACC_VALUE) do
    if let some d := portDiff? w b false then
      throw s!"the verifier's shared rows (--rec-rows): its {ACC_VALUE} ports acc0 and {b.name} are two registrations: {d}"
  if let some d := portDiff? a w false then
    throw s!"{s.name}: its {ACC_VALUE} port {a.name} is not the verifier's {ACC_VALUE} registration (--rec-rows acc0): {d}"

def TRANSCRIPT_FORMAT : String := "rec/inner-transcript/v1"

/-- A round's coins, as hex strings. -/
def coinsOf (r : Json) : Array String := (r.getObjValAs? (Array String) "coins").toOption.getD #[]

/-- Candidate 3's streams beside the circuit table's reps (`rec_live.SPARSE`, `rec_live.STACKED`): each rep's sparse phase,
`sparse/rep<x>`, and the stacked table R's one stream, `stacked/rep0`. -/
def SPARSE : String := "sparse"
def STACKED : String := "stacked"

/-- One stream of the inner session as the coin server recorded it: its message rounds' `c_k` (`b ‖ c`), each with its
round number, in round order; and its table's committed tops by Ligerito level (`rec_vstar.tops`), each node's `b ‖ c`:
level 0 the table's `root_B` from `Commit` (the record's `link.roots`; R, which has none there, its binding round's salted
top), then each later salted top in the order the stream's rounds absorbed them. A sparse stream belongs to no table and has
none. -/
structure InnerStream where
  name : String
  commits : Array (Nat × ByteArray)
  tops : Array (Array ByteArray)

/-- What the verifier holds of the inner session's transcript: the circuit table's reps' `c_k` (`reps`, rep `x` at `x`), and
with candidate 3's streams every stream of the session (`streams`; empty for a session of one table's two reps). -/
structure InnerCommits where
  reps : Array (Array (Nat × ByteArray))
  streams : Array InnerStream := #[]

def InnerCommits.stream? (ic : InnerCommits) (name : String) : Option InnerStream := ic.streams.find? (·.name == name)

private def hexList (ctx : String) (xs : Array String) : V (Array ByteArray) :=
  xs.mapM fun x => do
    let b ← (unhex x).mapError (s!"rec inner transcript: {ctx}: " ++ ·)
    if b.size != 128 then throw s!"rec inner transcript: {ctx}: a b ‖ c of {b.size} bytes, not 128"
    pure b

/-- **A rep's coins, by the transcript's rounds**: what the inner session's record (the prover's, which check 5 reads)
holds of rep stream `s` at each of the transcript's rounds `0 … n-1` of it (`rs`, in round order): that round's coins, or
`none` at the round the record has no counterpart of. A holo session's firewall splits z_partial's round in two (`rec_session`:
a round of no coins, then h's round), and the transcript records both where the prover's record has the one round. So with
one round of no coins at `j` in the transcript's stream, the record has `n - 1` rounds, its round `k` the transcript's `k`
below `j` and `k + 1` from `j` on (h's round taking z_partial's coins); with none, the two have the same rounds. -/
def repCoins (s : String) (rs : Array Json) (srec : Json) : V (Array (Option (Array String))) := do
  let n := rs.size
  if rs.toList.map (fun r => (r.getObjValAs? Nat "k").toOption) != (List.range n).map some then
    throw s!"rec inner transcript: {s}'s rounds are not its rounds 0 to {n - 1}"
  let rr := (srec.getObjValAs? (Array Json) "rounds").toOption.getD #[]
  match (rs.zipIdx.filter fun (r, _) => (coinsOf r).isEmpty).map (·.2) with
  | #[] =>
    if rr.size != n then throw s!"rec inner transcript: {s} records {n} rounds and the inner session's record {rr.size}"
    return rr.map (some ∘ coinsOf)
  | #[j] =>
    if j + 1 == n then throw s!"rec inner transcript: {s}'s round {j} of no coins is its last; h's round follows z_partial's"
    if rr.size + 1 != n then
      throw s!"rec inner transcript: {s} splits z_partial's round {j} in two and records {n} rounds; the inner session's record has {rr.size}, not {n - 1} (it is the prover's record, check 5's)"
    return (Array.range n).map fun k => if k == j then none else rr[if k < j then k else k - 1]?.map coinsOf
  | js => throw s!"rec inner transcript: {s} has rounds of no coins at {js}, and only z_partial's round is split"

/-- **Candidate 3's session** (`rec_live.schedule_for` with `sparse`, and `stacked`): the circuit table `t`'s two reps, both
reps' sparse phases and, if it committed R, R's stream. The transcript's streams (`streams`) and the streams its message
rounds name are those, each declared where `schedule_for` puts it (a rep's sparse phase at the end of the rep's stream, the
reps and R at round 0 of no parent), each with every round from 0 to its count; the sparse streams and R's are the coin
server's alone, since the prover's record (`innerCommits`) holds only the reps; Hello's tables and the committed roots are
`t`'s alone (the inner statement's Hello), R's root being the top its stream's binding round commits; each round's tops are
lists of `b ‖ c`, none in a sparse stream; R's binding round's first top is R's registered top `regR`, `b ‖ c` for
`b ‖ c` (`registration`); and every `b ‖ c` the session holds (roots, tops, `c_k`) is a distinct one, since the firewall
commits each under a fresh salt. -/
def candidate3 (tr : Json) (rounds : Array Json) (t : String) (names : List String)
    (msgs : Array (String × Nat × ByteArray)) (regR : Array ByteArray := #[]) : V (Array InnerStream) := do
  let stacked := names.contains s!"{STACKED}/rep0"
  let want := [s!"{t}rep0", s!"{t}rep1", s!"{SPARSE}/rep0", s!"{SPARSE}/rep1"] ++ (if stacked then [s!"{STACKED}/rep0"] else [])
  let tbl := (t.dropEnd 1).toString
  let same := fun (a b : List String) => a.all b.contains && b.all a.contains && a.length == b.length
  if !same names want then
    throw s!"rec inner transcript: its message rounds' streams {names} are not one table's rep0 and rep1, both reps' sparse phases and at most R's stream ({want})"
  let declared ← match tr.getObjVal? "streams" with
    | .ok (.obj kvs) => pure kvs.toArray
    | _ => throw "rec inner transcript: streams is not an object"
  if !same (declared.toList.map (·.1)) want then
    throw s!"rec inner transcript: it declares the streams {declared.toList.map (·.1)}, not {want}"
  let decl := fun (s : String) => (declared.find? (·.1 == s)).map (·.2) |>.getD Json.null
  let count := fun (s : String) => ((decl s).getObjValAs? Nat "rounds").toOption.getD 0
  for s in want do
    let parent := if s.startsWith s!"{SPARSE}/" then s!"{t}{(s.drop (SPARSE.length + 1)).toString}" else ""
    let at_ := if parent.isEmpty then 0 else count parent
    if ((decl s).getObjValAs? String "parent").toOption != some parent || ((decl s).getObjValAs? Nat "at").toOption != some at_ then
      throw s!"rec inner transcript: {s} is not declared at {parent}'s round {at_}"
    let mine := (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).map fun r =>
      ((r.getObjValAs? Nat "k").toOption, r)
    if mine.size != count s then
      throw s!"rec inner transcript: {s} declares {count s} rounds and records {mine.size}"
    for k in [0:count s] do
      if !(mine.any (·.1 == some k)) then throw s!"rec inner transcript: {s} has no round {k}"
  let tables := (tr.getObjVal? "hello" |>.toOption.getD Json.null).getObjValAs? (Array String) "tables" |>.toOption.getD #[]
  let wantTables := #[tbl]
  if tables != wantTables then throw s!"rec inner transcript: Hello's tables {tables}, not {wantTables}"
  let roots ← match (tr.getObjVal? "link" |>.toOption.getD Json.null).getObjVal? "roots" with
    | .ok (.obj kvs) => pure kvs.toArray
    | _ => throw "rec inner transcript: link.roots is not an object"
  if !same (roots.toList.map (·.1)) wantTables.toList then
    throw s!"rec inner transcript: the committed roots are {roots.toList.map (·.1)}'s, not {wantTables}'s"
  let mut rootOf : Array (String × Array ByteArray) := #[]
  for (name, j) in roots do
    let .ok xs := Json.getArr? j | throw s!"rec inner transcript: {name}'s root is not a list"
    let bcs ← hexList s!"{name}'s root" (xs.map fun x => (x.getStr?).toOption.getD "")
    if bcs.isEmpty then throw s!"rec inner transcript: {name}'s root has no nodes"
    rootOf := rootOf.push (name, bcs)
  let mut out : Array InnerStream := #[]
  let mut all : Array ByteArray := rootOf.foldl (· ++ ·.2) #[]
  for s in want do
    let table := if s.startsWith s!"{SPARSE}/" then none else some (if s.startsWith s!"{STACKED}/" then STACKED else tbl)
    let committed := table == some tbl
    let mut tops : Array (Array ByteArray) := if committed then #[((rootOf.find? (·.1 == tbl)).map (·.2)).getD #[]] else #[]
    let rs := (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).qsort fun a b =>
      (a.getObjValAs? Nat "k").toOption.getD 0 < (b.getObjValAs? Nat "k").toOption.getD 0
    for r in rs do
      let k := (r.getObjValAs? Nat "k").toOption.getD 0
      let .ok ts := r.getObjValAs? (Array (Array String)) "tops" | throw s!"rec inner transcript: {s} round {k}'s tops are not lists of b ‖ c"
      if table.isNone && !ts.isEmpty then throw s!"rec inner transcript: {s} round {k} commits a top, and a sparse stream has no table"
      if table == some STACKED && k == 0 && ts.isEmpty then
        throw s!"rec inner transcript: {s}'s binding round commits no top: R's root is committed nowhere"
      for (top, i) in ts.zipIdx do
        if top.isEmpty then throw s!"rec inner transcript: {s} round {k}'s top {i} has no nodes"
        let bcs ← hexList s!"{s} round {k}'s top {i}" top
        if table == some STACKED && k == 0 && i == 0 && bcs.map hexOf != regR.map hexOf then
          throw s!"rec inner transcript: {s}'s binding round commits a top other than R's registered top (--rec-inner-registration)"
        tops := tops.push bcs
    let cs := ((msgs.filter (·.1 == s)).qsort (·.2.1 < ·.2.1)).map (·.2)
    all := all ++ (tops.extract (if committed then 1 else 0)).foldl (· ++ ·) #[] ++ cs.map (·.2)
    out := out.push { name := s, commits := cs, tops := if table.isSome then tops else #[] }
  let mut seen : Std.HashSet String := {}
  for b in all do
    let h := hexOf b
    if seen.contains h then
      throw s!"rec inner transcript: the b ‖ c {(h.take 16).toString}… is committed twice; the firewall commits each under a fresh salt"
    seen := seen.insert h
  return out

def REGISTRATION_FORMAT : String := "rec/inner-registration/v0"
/-- How a registered top's nodes are salted (`rec_live.TOPS`): each 64-byte node a registered row under its own salt. -/
def TOPS_FORMAT : String := "hm96-sha512/v1/node-row"

/-- **R's registered top** (`--rec-inner-registration`, `rec_live.registration_record`'s `registration.json`): the tops of
the tables with no root at `Commit`, salted once at registration, before the session and any coin (`rec_live.register_top`),
public: `{"kind": "rec/inner-registration/v0", "tops": "hm96-sha512/v1/node-row", "registered": {"stacked": {"top": [b ‖ c,
…]}}}`, each `b ‖ c` a node's 128 bytes in hex. R (`stacked`) is the only such table. -/
def registration (j : Json) : V (Array ByteArray) := do
  if (j.getObjValAs? String "kind").toOption != some REGISTRATION_FORMAT then
    throw s!"rec inner registration: not a {REGISTRATION_FORMAT} record"
  if (j.getObjValAs? String "tops").toOption != some TOPS_FORMAT then
    throw s!"rec inner registration: its tops are not salted as {TOPS_FORMAT}"
  let .ok (.obj kvs) := j.getObjVal? "registered" | throw "rec inner registration: registered is not an object"
  let names := kvs.toArray.map (·.1)
  if names != #[STACKED] then throw s!"rec inner registration: it registers the tops of {names}, not R's ({STACKED}) alone"
  let .ok xs := (kvs.toArray[0]!.2).getObjValAs? (Array String) "top" | throw "rec inner registration: R's top is not a list of b ‖ c"
  if xs.isEmpty then throw "rec inner registration: R's top has no nodes"
  (hexList "R's registered top" xs).mapError fun e => e.replace "rec inner transcript: " "rec inner registration: "

/-- **Each rep's `c_k`**: the message rounds of the coin server's record of V*'s inner session (`--rec-inner-transcript`,
`rec/inner-transcript/v1`), those holding commitments, of stream `<table>/rep<x>` for `x` 0 and 1, in round order, each
its round number and its one `c_k` (`b ‖ c`), as `rec_vstage.message_rounds` reads them (rec_vstage.py:140-154). The inner
session's record (`--rec-inner-session`) is the prover's, the one check 5 reads, and holds the two reps alone: each rep's
rounds' coins must be the record's at the same round, z_partial's split aside (`repCoins`). With candidate 3's streams beside
the reps (a stream named `sparse/…` or `stacked/…`), every stream of the session, as `candidate3` reads it from the coin
server's record alone. R has no root at `Commit`, so its registration (`reg`, `--rec-inner-registration`) binds it before
any coin: a session with R's stream needs R's registered top, which its binding round commits; and a session whose
registration has R's top needs R's stream, whose opening the statement then reads (a session without it is a downgrade). -/
def innerCommits (tr : Json) (record : String) (reg : Option Json := none) : V InnerCommits := do
  if (tr.getObjValAs? String "kind").toOption != some TRANSCRIPT_FORMAT then throw s!"rec inner transcript: not a {TRANSCRIPT_FORMAT} record"
  let regR ← match reg with
    | some j => some <$> registration j
    | none => pure none
  let .ok rounds := tr.getObjValAs? (Array Json) "rounds" | throw "rec inner transcript: rounds is not a list"
  let rec_ ← match Json.parse record with
    | .ok j => pure j
    | .error e => throw s!"rec inner transcript: the inner session's record is not JSON: {e}"
  let .ok streams := rec_.getObjValAs? (Array Json) "streams" | throw "rec inner transcript: the inner session's record has no streams"
  let mut msgs : Array (String × Nat × ByteArray) := #[]
  for r in rounds do
    let cs := (r.getObjValAs? (Array String) "commits").toOption.getD #[]
    if cs.isEmpty then continue
    let (.ok s, .ok k) := (r.getObjValAs? String "stream", r.getObjValAs? Nat "k")
      | throw "rec inner transcript: a message round names no stream or round"
    if cs.size != 1 then throw s!"rec inner transcript: {s} round {k} holds {cs.size} commitments, not one"
    msgs := msgs.push (s, k, ← unhex cs[0]!)
  let names := (msgs.map (·.1)).toList.eraseDups
  let some t := names.head?.map fun n => (n.dropEndWhile (· != '/')).toString | throw "rec inner transcript: no message round holds a commitment"
  let extra := fun (n : String) => n.startsWith s!"{SPARSE}/" || n.startsWith s!"{STACKED}/"
  let c3 := names.any extra
  let t ← if c3 then
      match (names.find? (!extra ·)).map fun n => (n.dropEndWhile (· != '/')).toString with
      | some t => pure t
      | none => throw s!"rec inner transcript: its message rounds' streams {names} hold no table's reps"
    else if names.length != 2 || !names.contains s!"{t}rep0" || !names.contains s!"{t}rep1" then
      throw s!"rec inner transcript: its message rounds' streams {names} are not one table's rep0 and rep1"
    else pure t
  let repNames := [s!"{t}rep0", s!"{t}rep1"]
  let recNames := streams.toList.filterMap fun x => (x.getObjValAs? String "stream").toOption
  if recNames != repNames then
    throw s!"rec inner transcript: the inner session's record has the streams {recNames}, not {repNames} (the prover's record, which check 5 reads)"
  for s in repNames do
    let rs := (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).qsort fun a b =>
      (a.getObjValAs? Nat "k").toOption.getD 0 < (b.getObjValAs? Nat "k").toOption.getD 0
    let srec := (streams.find? fun x => (x.getObjValAs? String "stream").toOption == some s).getD Json.null
    for (r, c) in rs.zip (← repCoins s rs srec), k in [0:rs.size] do
      if c.any (· != coinsOf r) then throw s!"rec inner transcript: {s} round {k}'s coins are not the inner session record's"
  let stackedS := s!"{STACKED}/rep0"
  let declaredR := (tr.getObjVal? "streams" >>= (·.getObjVal? stackedS)).toOption.isSome
  match names.contains stackedS, regR with
  | true, none =>
    throw s!"rec inner transcript: {stackedS} binds R, and the verifier holds no registered top of R (--rec-inner-registration)"
  | false, some _ =>
    -- declared with no rounds in candidate 3's session: `candidate3`'s layout check names it
    if !(declaredR && c3) then
      throw s!"rec inner transcript: the registration has R's top, and the session has no {stackedS}: R's stream is missing (a downgrade)"
  | _, _ => pure ()
  let reps := #[0, 1].map fun x => ((msgs.filter (·.1 == s!"{t}rep{x}")).qsort (·.2.1 < ·.2.1)).map (·.2)
  if c3 then return { reps, streams := ← candidate3 tr rounds t names msgs (regR.getD #[]) }
  return { reps }

/-- `rec-opening-c<i>`'s `i`. -/
def openingIndex (value : String) : Option Nat :=
  if value.startsWith "rec-opening-c" then (value.drop 13).toString.toNat? else none

/-- `rec-top-stacked-L<l>`'s `l`. -/
def stackedTopLevel (value : String) : Option Nat :=
  if value.startsWith "rec-top-stacked-L" then (value.drop 17).toString.toNat? else none

/-- `rec-sparse-<p>`'s `p` when `p` is one of RecSparse's message ports at some shape (`pre<j>`, `v2`, `sc<j>`, `u`, `vt`):
not `h` (registered as its `rec-c<i>`), `claim` (`rec-claim`), nor `v`, `zero` or `tau`, which no round commits. -/
def sparsePort (value : String) : Option String :=
  if !value.startsWith "rec-sparse-" then none else
  let p := (value.drop 11).toString
  let idx := fun (pre : String) => p.startsWith pre && ((p.drop pre.length).toString.toNat?).isSome
  if p == "v2" || p == "u" || p == "vt" || idx "pre" || idx "sc" then some p else none

/-- RecSparse's message ports at `sh` in the order its sparse phase commits them (`message_ports` without `h` and
`claim`): `pre<j>`, `v2`, `sc<j>`, `u`, and `vt` at T > 1. -/
def sparseMessages (sh : RecSparse.Shape) : List String :=
  ((RecSparse.messagePorts sh).map (·.1)).filter fun p => p != "h" && p != "claim"

/-- A registered value the inner transcript fixes, and what its row `x` is there: `rec-c<i>`, rep `x`'s `i`-th message
round's `c_i`; `rec-opening-c<i>` (R's registered opening, `rec_vstage.opening_messages`), R's one stream's `i`-th message
round's `c_i`, at row 0; `rec-top-stacked-L<l>` (R's levels, `rec_vstage.stage_stacked`), node `x` of R's level-`l` top;
`rec-sparse-<p>` (RecSparse's message ports, `rec_vstage.sparse_messages`), `sparse/rep<x>`'s `i`-th message round's `c_i`,
`i` `p`'s place in `sparseMessages sparse`, `sparse` the statement's RecSparse shape.
`none` for any other value; else the row's description and its `b ‖ c`, or why the transcript has no such row. -/
def fixedRow (cs : InnerCommits) (value : String) (x : Nat) (sparse : Option RecSparse.Shape := none) :
    Option (Except String (String × ByteArray)) :=
  match cIndex value, openingIndex value, stackedTopLevel value, sparsePort value with
  | some i, _, _, _ => some <| match (cs.reps[x]?).bind (·[i]?) with
    | some (k, c) => .ok (s!"rep {x}'s c_{i} (inner round {k})", c)
    | none => .error s!"the inner transcript has no rep {x} message round {i}"
  | _, some i, _, _ => some <| match (if x == 0 then (cs.stream? s!"{STACKED}/rep0").bind (·.commits[i]?) else none) with
    | some (k, c) => .ok (s!"R's c_{i} ({STACKED}/rep0 round {k})", c)
    | none => .error s!"the inner transcript has no row {x} of R's message round {i}, which is R's stream's at row 0"
  | _, _, some l, _ => some <| match ((cs.stream? s!"{STACKED}/rep0").bind (·.tops[l]?)).bind (·[x]?) with
    | some c => .ok (s!"node {x} of R's level {l} top", c)
    | none => .error s!"the inner transcript has no node {x} of R's level {l} top"
  | _, _, _, some p => some <| do
    let some sh := sparse | .error s!"{value} is a RecSparse message port and the statement is no rec-sparse unit"
    let names := sparseMessages sh
    let some i := names.idxOf? p | .error s!"{p} is not one of RecSparse's message ports {names} at {RecSparse.unitName sh}"
    let s := s!"{SPARSE}/rep{x}"
    let some st := cs.stream? s | .error s!"the inner transcript has no stream {s}"
    if st.commits.size != names.length then
      .error s!"{s} holds {st.commits.size} message rounds, not {RecSparse.unitName sh}'s {names.length} ({names})"
    let (k, c) := st.commits[i]!
    .ok (s!"{s}'s c_{i} ({p}, inner round {k})", c)
  | none, none, none, none => none

/-- **`pub.bin`'s rows at its reads of what the inner transcript fixes are the `b ‖ c` the verifier holds**: for each port of
the verifier's reads file registering a value `fixedRow` names (`rec-c<i>`, and with R `rec-opening-c<i>` and
`rec-top-stacked-L<l>`, and with a rec-sparse unit `rec-sparse-<p>`), every row of the port's table, at the registered
position `x` it opens, is that value's row `x` in `innerCommits`: for `rec-c<i>`, rep `x`'s `i`-th message round's `c_i`.
`Registered.check` puts each instance's row at the position its unit reads. -/
def checkPubRows (st : Stmt) (registered : Option (Array Registered.Port))
    (commits : Option InnerCommits) : V Unit := do
  let unit := st.c.unit.name
  let regs := (registered.getD #[]).filter fun reg => (fixedRow {reps := #[]} reg.value 0).isSome
  let some reg0 := regs[0]? | return
  let some cs := commits
    | throw s!"rec-pub: {unit}: the verifier holds no inner transcript (--rec-inner-transcript with --rec-inner-session), so no c_k to check port {reg0.name}'s rows of {reg0.value} against"
  let some t := st.pub.tables | throw s!"rec-pub: {unit}: {reg0.name} is not a table of a shared-row file"
  for reg in regs do
    let some p := (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name)
      | throw s!"rec-pub: {unit}: {reg.name} is not a row of the circuit"
    let (positions, _) ← Registered.entry st.pub reg
    for r in [0:positions.size] do
      let x := positions[r]!
      match fixedRow cs reg.value x (RecSparse.params unit) with
      | some (.ok (what, c)) =>
        if Registered.tableRow st.pub t p r != c then
          throw s!"rec-pub: {unit}: port {reg.name}'s table row {r} is not {what} of the inner transcript"
      | some (.error why) =>
        throw s!"rec-pub: {unit}: port {reg.name}'s table row {r} reads row {x} of {reg.value}, and {why}"
      | none => throw s!"rec-pub: {unit}: port {reg.name}: the inner transcript fixes no {reg.value}"

/-- With an inner shape, the statement must be V*'s: `tags`, the statement the verifier set it up as (`--statement`), is
V*'s own `Tags.circuit`, compared by name (`Tags.find?` gives each name one statement); it holds no lookup table
(`tables`, the verifier's `--tables`/`--archive` MUFU tables, is empty: V*'s circuits read none); it needs the verifier's own
partition object and program, which of V*'s statements the session is (`part`, `--rec-part J`) and the verifier's own
bindings (`bindings`, `--rec-bindings F`); then `checkOwnBindings` of those bindings against the partition object and
program, `checkCircuit` on the statement's circuit and population as statement `part` under those bindings, and
`checkOwn`, `checkRows` and `checkAcc` (against the verifier's shared rows `rows`, `--rec-rows`) on the verifier's own
registered reads. -/
def checkStmt (inner : Option Inner) (tags : Tags) (tables : List (String × ByteArray)) (st : Stmt) (circuitB : ByteArray)
    (partition : Option Json := none)
    (program : Option ByteArray := none) (registered : Option (Array Registered.Port) := none) (part : Option Nat := none)
    (bindings : Option VStar.Bindings := none) (rows : Option (Array Registered.Port) := none) : V Unit :=
  let name := st.c.unit.name
  match inner with
  | some i => do
    if tags.name != Tags.circuit.name then
      throw s!"{name}: one of V*'s statements, which are {Tags.circuit.name} statements (--statement {Tags.circuit.name}), not {tags.name}"
    if !tables.isEmpty then
      throw s!"{name}: one of V*'s statements, which read no lookup table; the verifier holds {tables.map (·.1)} (--tables, --archive)"
    let (some obj, some prog) := (partition, program)
      | throw s!"{name}: one of V*'s statements; pass the verifier's own partition object and program (--partition F, --program F)"
    let some j := part
      | throw s!"{name}: one of V*'s statements; pass which of V*'s statements this session is (--rec-part J, its index in rec-statements' list)"
    let some f := bindings
      | throw s!"{name}: one of V*'s statements; pass the verifier's own bindings (--rec-bindings F, from verity_flock.rec_bindings)"
    checkOwnBindings name f obj prog
    let s ← checkCircuit i j st.c (population st) circuitB f
    if let some r := registered then
      checkOwn s r
      checkRows s r rows
      checkAcc s r rows
  | none =>
    if isVStar name then
      throw s!"{name}: one of V*'s statements; pass the verifier's own inner shape (--rec-inner F, from rec-inner)"
    else pure ()

end RecVStar
end Flock
