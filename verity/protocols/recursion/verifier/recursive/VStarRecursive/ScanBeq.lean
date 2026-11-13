import Lean
import Spot
import Flock
import FlockZkServer
import FlockRec
import VStar
import VStarRecursive
import CoreProtocol

/-! A count by type, not by text: every declaration of the verifiers this package requires that compares `Lean.Json` with
its `BEq` instance. A scratch file on a branch of its own; it is not for `main`. -/

open Lean

namespace ScanBeq

/-- The constants a declaration's type and body name. -/
def uses : ConstantInfo → Array Name
  | .thmInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .defnInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | c => c.type.getUsedConstants

def kind : ConstantInfo → String
  | .thmInfo _ => "theorem" | .defnInfo _ => "def" | .opaqueInfo _ => "opaque" | .inductInfo _ => "inductive"
  | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .axiomInfo _ => "axiom" | .quotInfo _ => "quot"

/-- A name that reads as a comparison's instance or its function: a derived `BEq`'s. -/
def carries (n : Name) : Bool :=
  let s := n.toString.toLower
  (s.splitOn "beq").length > 1

/-- Every declaration of the modules `ours` keeps that names `Lean.Json.instBEq`, and every one that names a
comparison of ours that does (a derived `BEq` over a type with a `Json` field), to a fixed point. -/
def scan (ours : Name → Bool) : CoreM Unit := do
  let env ← getEnv
  let mods := env.header.moduleNames
  let mut decls : Array (Name × Name × ConstantInfo × Array Name) := #[]
  let mut seen : NameSet := {}
  for (n, ci) in env.constants.map₁.toList do
    let some idx := env.getModuleIdxFor? n | continue
    let m := mods[idx.toNat]!
    unless ours m do continue
    seen := seen.insert m
    decls := decls.push (m, n, ci, uses ci)
  IO.println s!"SCANNED {decls.size} declarations of {seen.size} modules; with a body: {(decls.filter fun (_, _, ci, _) => ci.hasValue).size}"
  for m in (seen.toList.map (·.toString)).toArray.qsort (· < ·) do
    IO.println s!"MODULE {m}"
  let mut carriers : NameSet := NameSet.empty.insert ``Lean.Json.instBEq
  let mut found : NameSet := {}
  let mut level := 1
  let mut go := true
  while go do
    go := false
    let mut next := carriers
    for (m, n, ci, us) in decls do
      if found.contains n then continue
      let via := us.filter carriers.contains
      if via.isEmpty then continue
      found := found.insert n
      let line := match ← findDeclarationRanges? n with
        | some r => toString r.range.pos.line
        | none => "-"
      IO.println s!"HIT {level} {kind ci} {m} {line} {n} via {via.toList}"
      if carries n then
        next := next.insert n
        go := true
    carriers := next
    level := level + 1
  IO.println s!"FOUND {found.size}"

end ScanBeq

#eval ScanBeq.scan fun m => !([`Init, `Std, `Lean, `Lake].contains m.getRoot)
