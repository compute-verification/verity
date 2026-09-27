import Lean

/-!
# Kernel replay of a package's own declarations

`lake env lean --run Replay.lean M₁ M₂ …` reads every constant the modules `Mᵢ` declare from their `.olean` files, imports
once the modules they import from outside that set, and sends each constant to the kernel (`Kernel.Environment.replay`).
A declaration that never passed the kernel (`debug.skipKernelTC`, `addDeclWithoutChecking`, a tampered `.olean`) is
rejected here, which `#print axioms` cannot see. This is `leanchecker`'s check with one environment in memory rather than
one per module (`leanchecker` replays all the modules under a prefix at once, and runs out of memory on Mathlib-sized
imports). Exit 0 when every constant is accepted, 1 when the kernel rejects one.
-/

open Lean

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let mods := args.toArray.map String.toName
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
  try
    discard <| env.toKernelEnv.replay constants
  catch e =>
    IO.eprintln s!"replay: the kernel rejects a declaration of {", ".intercalate args}:\n{e}"
    return 1
  let sent := constants.fold (fun n _ c => if c.isUnsafe || c.isPartial then n else n + 1) 0
  IO.println s!"replay: {sent} constants of {mods.size} modules accepted by the kernel ({constants.size - sent} more are \
    unsafe or partial: compiled code, which the kernel does not check)"
  return 0
