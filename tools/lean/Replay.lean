import Lean

/-!
# Kernel replay of a package's own declarations

`lake env lean --run Replay.lean OUT M₁ M₂ …` reads every constant the modules `Mᵢ` declare from their `.olean` files,
imports once the modules they import from outside that set, and sends each constant to the kernel
(`Kernel.Environment.replay`). A declaration that never passed the kernel (`debug.skipKernelTC`, `addDeclWithoutChecking`,
a tampered `.olean`) is rejected here, which `#print axioms` cannot see. This is `leanchecker`'s check with one environment
in memory rather than one per module (`leanchecker` replays all the modules under a prefix at once, and runs out of memory
on Mathlib-sized imports).

It then writes to `OUT` the axioms those constants use, found by walking the replayed environment itself (the bodies it
holds, not the axiom sets each `.olean` recorded when it was built), so `audit.py` can compare the two. Exit 0 when every
constant is accepted, 1 when the kernel rejects one.
-/

open Lean

/-- The axioms reachable from `roots` through the bodies in `env`, then in `extra` (constants the kernel does not check). -/
def axiomsIn (env : Kernel.Environment) (extra : Std.HashMap Name ConstantInfo) (roots : Array Name) : Array Name := Id.run do
  let mut seen : NameSet := {}
  let mut stack := roots
  let mut axs : NameSet := {}
  repeat
    match stack.back? with
    | none => break
    | some c =>
      stack := stack.pop
      if seen.contains c then continue
      seen := seen.insert c
      match env.find? c <|> extra[c]? with
      | some (.axiomInfo v) => axs := axs.insert c; stack := stack ++ v.type.getUsedConstants
      | some (.defnInfo v) => stack := stack ++ v.type.getUsedConstants ++ v.value.getUsedConstants
      | some (.thmInfo v) => stack := stack ++ v.type.getUsedConstants ++ v.value.getUsedConstants
      | some (.opaqueInfo v) => stack := stack ++ v.type.getUsedConstants ++ v.value.getUsedConstants
      | some (.inductInfo v) => stack := stack ++ v.type.getUsedConstants ++ v.ctors.toArray
      | some (.ctorInfo v) => stack := stack ++ v.type.getUsedConstants
      | some (.recInfo v) => stack := stack ++ v.type.getUsedConstants
      | some (.quotInfo _) | none => pure ()
  return axs.toArray.qsort Name.lt

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let some out := args.head? | throw <| IO.userError "usage: Replay.lean OUT MODULE..."
  let mods := args.tail.toArray.map String.toName
  let mut constants : Std.HashMap Name ConstantInfo := {}
  let mut imports : Array Import := #[]
  for m in mods do
    let olean ← findOLean m
    let mut files := #[olean]
    for level in [OLeanLevel.server, OLeanLevel.private] do
      if ← (level.adjustFileName olean).pathExists then
        files := files.push (level.adjustFileName olean)
    let parts ← readModuleDataParts files
    let (some (data, _), some (header, _)) := (parts.back?, parts[0]?) | throw <| IO.userError s!"no module data for {m}"
    for c in data.constants do
      constants := constants.insert c.name c
    for i in header.imports do
      if !mods.contains i.module && !imports.any (·.module == i.module) then
        imports := imports.push i
  let env ← importModules imports {}
  if let some n := constants.keys.find? (env.contains ·) then
    throw <| IO.userError s!"{n} is declared by a replayed module and by one it imports from outside the set"
  let kenv ← try env.toKernelEnv.replay constants
    catch e =>
      IO.eprintln s!"replay: the kernel rejects a declaration of {", ".intercalate args.tail}:\n{e}"
      return 1
  let sent := constants.fold (fun n _ c => if c.isUnsafe || c.isPartial then n else n + 1) 0
  let axioms := axiomsIn kenv constants constants.keys.toArray
  IO.FS.writeFile out <| (Json.mkObj [("replayed", toJson sent), ("skipped", toJson (constants.size - sent)),
    ("axioms", toJson (axioms.map toString))]).compress
  IO.println s!"replay: {sent} constants of {mods.size} modules accepted by the kernel ({constants.size - sent} more are \
    unsafe or partial: compiled code, which the kernel does not check); axioms from the replayed environment: \
    {", ".intercalate (axioms.map toString).toList}"
  return 0
