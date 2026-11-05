import Lean
open Lean

/-- `lean --run Extract.lean ROOT MODS OUT`: every constant of the modules listed in MODS (one per line) in ROOT's import
closure, with the constants of those modules its type and value use. OUT holds `M idx module` lines, then
`C id moduleIdx name dep,dep,...` lines (deps as constant ids). -/
def main (args : List String) : IO UInt32 := do
  let [root, modsFile, outFile] := args | throw <| IO.userError "usage: Extract ROOT MODS OUT"
  let wanted : NameSet := (← IO.FS.lines modsFile).foldl
    (fun s l => if l.isEmpty then s else s.insert l.toName) {}
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := root.toName }] {} (trustLevel := 1024)
  let names := env.header.moduleNames
  let mut ids : Std.HashMap Name Nat := {}
  let mut consts : Array (Name × Nat) := #[]
  for (n, _) in env.constants.map₁ do
    if let some idx := env.getModuleIdxFor? n then
      let i : Nat := idx.toNat
      if wanted.contains names[i]! then
        ids := ids.insert n consts.size
        consts := consts.push (n, i)
  let h ← IO.FS.Handle.mk outFile .write
  for i in [0:names.size] do
    if wanted.contains names[i]! then h.putStrLn s!"M\t{i}\t{names[i]!}"
  for i in [0:consts.size] do
    let (n, m) := consts[i]!
    let some ci := env.find? n | continue
    let deps := ci.getUsedConstantsAsSet.toList.filterMap (ids[·]?)
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "ax" | .opaqueInfo _ => "opq"
      | .inductInfo _ => "ind" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"
    h.putStrLn s!"C\t{i}\t{m}\t{kind}\t{n}\t{",".intercalate (deps.map toString)}"
  IO.println s!"{consts.size} constants in {wanted.size} modules"
  return 0
