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
  if j.getObjValAs? String "format" != .ok FORMAT then throw s!"rec-inner: not a {FORMAT} object"
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

/-- `check` on a statement's unit and population, with the verifier's own inner shape `inner`; a statement of V*'s needs
one, and with one, the statement must be V*'s. -/
def checkStmt (inner : Option Inner) (st : Stmt) : V Unit :=
  match inner with
  | some i => check i st.c.unit.name (population st)
  | none =>
    if isVStar st.c.unit.name then
      throw s!"{st.c.unit.name}: one of V*'s statements; pass the verifier's own inner shape (--rec-inner F, from rec-inner)"
    else pure ()

end RecVStar
end Flock
