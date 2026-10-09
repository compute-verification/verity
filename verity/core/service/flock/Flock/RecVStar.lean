/-
V*'s statements over one inner session, checked by the verifier against V*'s own executable Lean (`VStar`, in this
package so that the verifier can read it): `VStar.statements` at the inner session's shape (`VStar/Stage.lean:48-58`: one
statement per Ligerito level of `fast100(m)`, `rec-open/l<2^lanes>-h<d − cap>` over that level's `2 q` openings, then one
per part of the rep's algebra, `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` over the two reps). A circuit whose unit is one
of V*'s is refused unless its META binds the verifier's own partition object and program, its unit's name is one of these
statements' at the inner session's shape, its population is that statement's instance count, and the circuit file is
byte for byte the one `vstar stage` writes for that statement under the file's bindings (`circuitFile`).

The inner shape is the verifier's own (`rec-inner`): `Stmt.setupTables` on the inner statement, as `InnerFold.lean` sets it
up for V*'s native side, read once, since the inner circuit can be tens of GB; `verify --rec-inner F` reads the line it
printed.
-/
import Flock.HmRow
import VStar.Compose
import VStar.Stage

namespace Flock
namespace RecVStar

open Lean

/-- The inner session's shape: its table's `m` and `k_log` (`Setup.ofCircuit`), and its claims, two and one per region and
link point (`Stmt.extraClaims`, `rec_vstage.verifier`'s `2 + len(extras)`). -/
structure Inner where
  m : Nat
  kLog : Nat
  claims : Nat
  deriving Repr, BEq, Inhabited

/-- A recursive session's inner statement (§16.16) on the verifier's live coins, as `87-rec-private.sh` runs `inner-fold`. -/
def innerTags : Tags := Tags.circuitPlainLeaves.withCoins Coins.os

/-- The inner statement's shape, from its circuit, public file and the verifier's own partition object and program. -/
def innerOf (circuitB pubB : ByteArray) (partition : Option Json) (program : Option ByteArray) : V Inner := do
  let (spec, sts) ← Stmt.setupTables innerTags circuitB pubB Coins.os 1 (partition := partition) (program := program)
  let some st := sts[0]? | throw "rec-inner: the inner statement has no table"
  pure ⟨st.m, st.c.kLog, 2 + spec.points * st.regions.size⟩

def FORMAT : String := "verity/rec-inner/v0"

def Inner.toJson (i : Inner) : Json :=
  Json.mkObj [("format", .str FORMAT), ("m", Lean.toJson i.m), ("k_log", Lean.toJson i.kLog), ("claims", Lean.toJson i.claims)]

def Inner.ofJson (j : Json) : Except String Inner := do
  if (j.getObjValAs? String "format").toOption != some FORMAT then throw s!"rec-inner: not a {FORMAT} object"
  pure ⟨← j.getObjValAs? Nat "m", ← j.getObjValAs? Nat "k_log", ← j.getObjValAs? Nat "claims"⟩

/-- V*'s statements at the inner shape (`VStar.statements`), at a shape `RecResiduals.check` accepts a part's name at. -/
def statements (i : Inner) : Except String (Array VStar.Stmt) := do
  if (RecResiduals.shape? i.m i.kLog i.claims).isNone then
    throw (s!"the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}) is not V*'s: a fast100 schedule, " ++
      "k_log at least 6, claims at least 2")
  VStar.statements { m := i.m, kLog := i.kLog, claims := i.claims }

def isVStar (name : String) : Bool := name.startsWith "rec-open/" || name.startsWith "rec-residuals/"

/-- A unit named `name` over a population of `n` instances is one of V*'s statements at the inner shape `i`: a part's name
carries `i`'s `m`, `k_log` and claims, exactly one statement has the name, and `n` is its instance count. -/
def check (i : Inner) (name : String) (n : Nat) : V VStar.Stmt := do
  if let some (m, k, c, _) := RecResiduals.params name then
    if m != i.m || k != i.kLog || c != i.claims then
      throw s!"{name}: shape (m {m}, k_log {k}, claims {c}), the inner session's (m {i.m}, k_log {i.kLog}, claims {i.claims})"
  let ss ← statements i
  match ss.toList.filter (·.name == name) with
  | [] =>
    throw s!"{name}: not one of V*'s statements at the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}): {ss.map (·.name)}"
  | [s] =>
    if n != s.instances then throw s!"{name}: instances {n}, statement says {s.instances}"
    pure s
  | ss => throw s!"{name}: {ss.length} of V*'s statements have this name"

/-- **V*'s circuit file of a statement**: the bytes `vstar stage` writes to `<dir>/circuit.txt` (`VStar.Cli.write`,
`VStar/Cli.lean:43-53`): the statement's unit, laid out, emitted with its subcircuit and the statement's bindings
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

/-- The circuit file is V*'s for `s` under its own META bindings (`circuitFile`). -/
def checkFile (s : VStar.Stmt) (b : VStar.Bindings) (circuitB : ByteArray) : V Unit := do
  let f ← match circuitFile s b with
    | .ok f => pure f
    | .error e => throw s!"{s.name}: V*'s circuit file: {e}"
  if let some (k, line) := firstDiff? circuitB f then
    throw s!"{s.name}: the circuit file ({circuitB.size} bytes) is not V*'s ({f.size} bytes): they differ at byte {k}, line {line}"

/-- A statement's population: the registered population's instances for a drawn session, else its own. -/
def population (st : Stmt) : Nat := (st.pub.population.map (·.1)).getD st.n

/-- META's bindings (`VStar.Bindings`, `VStar/Compose.lean:50-55`, written at `Compose.lean:228-229`): the
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

/-- A statement of V*'s binds the verifier's own partition object and program: META's partition digest is the object's
(`HmRow.partitionDigest`), META's `program_sha512` and the object's `program` are the program's SHA-512, and
`program_digests` names one Definition, the template the object's query names. Its digest is not recomputed: that is
core's codec's digest of the one-Definition program, which this verifier does not encode. -/
def checkBindings (c : Circuit) (partition : Option Json) (program : Option ByteArray) : V VStar.Bindings := do
  let name := c.unit.name
  let (some obj, some prog) := (partition, program)
    | throw s!"{name}: one of V*'s statements; pass the verifier's own partition object and program (--partition F, --program F)"
  let b ← bindingsOf c
  let (digest, sha, pd) := (b.partition, b.program, b.programDigests)
  let own := hexOf (← HmRow.partitionDigest obj)
  if digest != own then throw s!"{name}: META partition digest {digest}, the verifier's partition object's {own}"
  let ps := hexOf (← TemplateQuery.Desc.parse prog).sha512
  if sha != ps then throw s!"{name}: META program_sha512 {sha}, the verifier's program's {ps}"
  let op := (obj.getObjValAs? String "program").toOption.getD ""
  if op != ps then throw s!"{name}: the partition object's program {op}, the verifier's program's {ps}"
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let t := ((q.getObjVal? "params").toOption.getD Json.null |>.getObjValAs? String "template").toOption.getD ""
  match pd with
  | .obj kvs =>
    match kvs.toArray.map (·.1) with
    | #[k] => if k != t then throw s!"{name}: META program_digests names {k}, the partition's template {t}" else pure b
    | ks => throw s!"{name}: META program_digests names {ks.size} Definitions, not the partition's template {t} alone"
  | _ => throw s!"{name}: META program_digests is not an object"

/-- `check` on the unit of `c`, parsed from the circuit file `circuitB`, over `n` instances at the inner shape `i`, then
`checkFile` on `circuitB` under its META bindings. -/
def checkCircuit (i : Inner) (c : Circuit) (n : Nat) (circuitB : ByteArray) : V Unit := do
  let s ← check i c.unit.name n
  checkFile s (← bindingsOf c) circuitB

/-- `checkBindings` against the verifier's own partition object and program, then `checkCircuit` on a statement's
circuit and population with its own inner shape `inner`; a statement of V*'s needs all three files, and with an inner
shape, the statement must be V*'s. -/
def checkStmt (inner : Option Inner) (st : Stmt) (circuitB : ByteArray) (partition : Option Json := none)
    (program : Option ByteArray := none) : V Unit :=
  match inner with
  | some i => do
    discard <| checkBindings st.c partition program
    checkCircuit i st.c (population st) circuitB
  | none =>
    if isVStar st.c.unit.name then
      throw s!"{st.c.unit.name}: one of V*'s statements; pass the verifier's own inner shape (--rec-inner F, from rec-inner)"
    else pure ()

end RecVStar
end Flock
