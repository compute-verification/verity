/-
V*'s statements over one inner session (`VStar.statements`, `verity/core/service/vstar/VStar/Stage.lean`), restated by
the verifier from the inner statement alone: one statement per Ligerito level of `fast100(m)`, `rec-open/l<2^lanes>-h<d − cap>`
over that level's `2 q` openings, then one per part of the rep's algebra, `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` over
the two reps. A circuit whose unit is one of V*'s is refused unless its unit's name is one of these at the inner session's
shape and its population is that statement's instance count.

The inner shape is the verifier's own (`rec-inner`): `Stmt.setupTables` on the inner statement, as `InnerFold.lean` sets it
up for V*'s native side, read once, since the inner circuit can be tens of GB; `verify --rec-inner F` reads the line it
printed.
-/
import Flock.HmRow

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

/-- V*'s statements at the inner shape, in `VStar.statements`' order: each one's unit name and instance count. The parts
are `RecResiduals.groups`', the grouping `RecResiduals.check` holds each part's unit to. -/
def statements (i : Inner) : Except String (Array (String × Nat)) := do
  let some sh := RecResiduals.shape? i.m i.kLog i.claims
    | throw (s!"the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}) is not V*'s: a fast100 schedule, " ++
        "k_log at least 6, claims at least 2")
  let opens := sh.sched.levels.map fun lv => (RecOpen.unitName (2 ^ lv.lanes) (lv.d - lv.cap), 2 * lv.q)
  let gs ← RecResiduals.groups (← RecAlgebra.structureOf sh) (RecResiduals.portSizes sh)
  pure (opens ++ (Array.range gs.size).map fun j => (RecResiduals.unitName i.m i.kLog i.claims j, 2))

def isVStar (name : String) : Bool := name.startsWith "rec-open/" || name.startsWith "rec-residuals/"

/-- A unit named `name` over a population of `n` instances is one of V*'s statements at the inner shape `i`: a part's name
carries `i`'s `m`, `k_log` and claims, and `n` is the statement's instance count. -/
def check (i : Inner) (name : String) (n : Nat) : V Unit := do
  if let some (m, k, c, _) := RecResiduals.params name then
    if m != i.m || k != i.kLog || c != i.claims then
      throw s!"{name}: shape (m {m}, k_log {k}, claims {c}), the inner session's (m {i.m}, k_log {i.kLog}, claims {i.claims})"
  let ss ← statements i
  let qs := (ss.filter (·.1 == name)).map (·.2)
  if qs.isEmpty then
    throw s!"{name}: not one of V*'s statements at the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}): {ss.map (·.1)}"
  if !qs.contains n then
    throw s!"{name}: instances {n}, statement says {if qs.size == 1 then toString qs[0]! else toString qs}"

/-- A statement's population: the registered population's instances for a drawn session, else its own. -/
def population (st : Stmt) : Nat := (st.pub.population.map (·.1)).getD st.n

/-- META's bindings (`VStar.Bindings`, `vstar/VStar/Compose.lean:50-55`, written at `Compose.lean:228-229`): the
`verity/partition/v1` digest, `program_sha512` and `program_digests`. -/
def bindingsOf (c : Circuit) : V (String × String × Json) := do
  let name := c.unit.name
  let part := (c.metaJ.getObjVal? "partition").toOption.getD Json.null
  let (some "verity/partition/v1", some digest) := ((part.getObjValAs? String "rule").toOption,
      (part.getObjValAs? String "digest").toOption)
    | throw s!"{name}: META partition is not a verity/partition/v1 digest"
  let some program := (c.metaJ.getObjValAs? String "program_sha512").toOption
    | throw s!"{name}: META names no program_sha512"
  let some pd := (c.metaJ.getObjVal? "program_digests").toOption | throw s!"{name}: META names no program_digests"
  pure (digest, program, pd)

/-- A statement of V*'s binds the verifier's own partition object and program: META's partition digest is the object's
(`HmRow.partitionDigest`), META's `program_sha512` and the object's `program` are the program's SHA-512, and
`program_digests` names one Definition, the template the object's query names. Its digest is not recomputed: that is
core's codec's digest of the one-Definition program, which this verifier does not encode. -/
def checkBindings (c : Circuit) (partition : Option Json) (program : Option ByteArray) : V Unit := do
  let name := c.unit.name
  let (some obj, some prog) := (partition, program)
    | throw s!"{name}: one of V*'s statements; pass the verifier's own partition object and program (--partition F, --program F)"
  let (digest, sha, pd) ← bindingsOf c
  let own := hexOf (← HmRow.partitionDigest obj)
  if digest != own then throw s!"{name}: META partition digest {digest}, the verifier's partition object's {own}"
  let ps := hexOf (Sha512.hash prog)
  if sha != ps then throw s!"{name}: META program_sha512 {sha}, the verifier's program's {ps}"
  let op := (obj.getObjValAs? String "program").toOption.getD ""
  if op != ps then throw s!"{name}: the partition object's program {op}, the verifier's program's {ps}"
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let t := ((q.getObjVal? "params").toOption.getD Json.null |>.getObjValAs? String "template").toOption.getD ""
  match pd with
  | .obj kvs =>
    match kvs.toArray.map (·.1) with
    | #[k] => if k != t then throw s!"{name}: META program_digests names {k}, the partition's template {t}"
    | ks => throw s!"{name}: META program_digests names {ks.size} Definitions, not the partition's template {t} alone"
  | _ => throw s!"{name}: META program_digests is not an object"

/-- `check` on a statement's unit and population, with the verifier's own inner shape `inner`, then `checkBindings` against
its own partition object and program; a statement of V*'s needs all three, and with an inner shape, the statement must be
V*'s. -/
def checkStmt (inner : Option Inner) (st : Stmt) (partition : Option Json := none) (program : Option ByteArray := none) :
    V Unit :=
  match inner with
  | some i => do
    check i st.c.unit.name (population st)
    checkBindings st.c partition program
  | none =>
    if isVStar st.c.unit.name then
      throw s!"{st.c.unit.name}: one of V*'s statements; pass the verifier's own inner shape (--rec-inner F, from rec-inner)"
    else pure ()

end RecVStar
end Flock
