import Lean
import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.HoloExec

/-! A scratch probe, on no branch that lands. `#classical_steps c` walks what `c` computes: the constants its body
names outside proofs, and theirs in turn. It prints every definition of the project on that walk whose own body names,
outside a proof, a library constant that reaches `Classical.choice` the same way, with the constant, one path from it
to `Classical.choice`, and each classical test (`Classical.propDecidable p` and its kin) with the proposition tested.
A proof is never entered, so classical reasoning inside proofs is not counted. -/

open Lean Meta Elab Command

namespace ClassicalSteps

/-- The module roots that count as library. -/
def libRoots : List Name := [`Init, `Lean, `Std, `Mathlib, `Batteries, `Aesop, `Qq, `ProofWidgets, `Plausible,
  `ImportGraph, `LeanSearchClient, `Cli]

def isLib (env : Environment) (c : Name) : Bool :=
  match env.getModuleIdxFor? c with
  | some i => libRoots.contains (env.header.moduleNames[i.toNat]!.getRoot)
  | none => false

/-- The heads that are classical tests. -/
def testHeads : List Name := [`Classical.propDecidable, `Classical.dec, `Classical.decEq, `Classical.decPred,
  `Classical.decRel]

structure Direct where
  consts : Array Name := #[]
  tests : Array String := #[]
  deriving Inhabited

/-- The constants `e` names outside proofs, and its classical tests. -/
def direct (e : Expr) : MetaM Direct := do
  let r ← IO.mkRef ({} : Direct)
  Meta.forEachExpr' e fun sub => do
    if sub.isSort || sub.isLit || sub.isBVar then return false
    let prf ← try isProof sub catch _ => pure false
    if prf then return false
    if let .const c _ := sub then
      r.modify fun d => if d.consts.contains c then d else { d with consts := d.consts.push c }
    if sub.isApp then
      if let .const h _ := sub.getAppFn then
        if testHeads.contains h then
          let s := toString (← ppExpr sub)
          r.modify fun d => if d.tests.contains s then d else { d with tests := d.tests.push s }
    return true
  r.get

structure St where
  deps : Std.HashMap Name Direct := {}
  /-- `some next`: reaches `Classical.choice` through `next`. -/
  reach : Std.HashMap Name (Option Name) := {}
  order : Array Name := #[]

abbrev M := StateRefT St MetaM

def bodyOf (c : Name) : MetaM (Option Expr) := do
  match (← getEnv).find? c with
  | some (.defnInfo d) => return some d.value
  | some (.opaqueInfo d) => return some d.value
  | _ => return none

/-- Whether `c` reaches `Classical.choice` outside proofs; records each constant's direct names on the way. -/
partial def visit (c : Name) : M Bool := do
  if c == ``Classical.choice then return true
  if let some r := (← get).reach[c]? then return r.isSome
  modify fun s => { s with reach := s.reach.insert c none }
  let some v ← bodyOf c | return false
  let d ← direct v
  modify fun s => { s with deps := s.deps.insert c d, order := s.order.push c }
  let mut via : Option Name := none
  for x in d.consts do
    if (← visit x) && via.isNone then via := some x
  modify fun s => { s with reach := s.reach.insert c via }
  return via.isSome

partial def pathOf (s : St) (c : Name) (fuel : Nat := 12) : List Name :=
  if c == ``Classical.choice || fuel == 0 then [c] else
  match s.reach[c]? with
  | some (some n) => c :: pathOf s n (fuel - 1)
  | _ => [c]

elab "#classical_steps " id:ident : command => do
  let c ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
  let (hit, s) ← liftTermElabM <| (visit c).run {}
  let env ← getEnv
  let mut out : Array String := #[]
  let mut entries : Array Name := #[]
  let mut nProj : Nat := 0
  for x in s.order do
    if isLib env x then continue
    nProj := nProj + 1
    let d := s.deps[x]!
    let cls := d.consts.filter fun y => (y == ``Classical.choice || isLib env y) &&
      (y == ``Classical.choice || (s.reach[y]?.join).isSome)
    if cls.isEmpty && d.tests.isEmpty then continue
    out := out.push s!"  {x}"
    for y in cls do
      if !entries.contains y then entries := entries.push y
      out := out.push s!"    names {y}  [{" → ".intercalate ((pathOf s y).map toString)}]"
    for t in d.tests do
      out := out.push s!"    test  {t}"
  logInfo m!"classical steps under {c}: reaches Classical.choice outside proofs: {hit}\n\
    definitions of the project walked: {nProj}; of all: {s.order.size}\n\
    library constants named by the project's definitions that reach Classical.choice: {entries}\n\
    {"\n".intercalate out.toList}"

end ClassicalSteps

#classical_steps FlockSoundness.Discharge.Recursion.schedSteps
#print axioms FlockSoundness.Discharge.Recursion.schedSteps
#classical_steps FlockSoundness.Game.playT
#print axioms FlockSoundness.Game.playT
#classical_steps FlockSoundness.Model.tableCL
#print axioms FlockSoundness.Model.tableCL
#classical_steps FlockSoundness.Model.holoTableCL
#print axioms FlockSoundness.Model.holoTableCL
#classical_steps FlockSoundness.Discharge.Recursion.outCL
#classical_steps FlockSoundness.Discharge.Recursion.outHL
