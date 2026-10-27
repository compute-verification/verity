import Lean

/-!
# What the listed guarantees use, from the `.olean` files as data

`lake env USED CONFIG`, where `USED` is this file compiled (`lean -c`, then `leanc`; `lake env lean --run Used.lean CONFIG`
is the same program interpreted), imports the modules `roots` names in `CONFIG` (a JSON file `unused.py` writes) as
`Facts.lean` does, with initializers disabled and no environment extension loaded, so none of the package's code runs. It
writes one JSON document to the path `out` names. It reports facts and makes no judgement; `unused.py` holds the policy.

The modules it reports on are the repository's own: those whose `.olean` is in a package's own `.lake/build`, not under
`.lake/packages` or the toolchain's (`check.py`'s `own_modules`). A constant of any other module cannot use one of theirs,
since it imports none of them, so every walk below stops at the first constant outside them.

- `pins`: for each name under `pins` (the guarantees), its module, or null when no imported module declares it; the
  names under `extra` (whatever else a lock names) are walked when they are constants and not reported.
- `modules`: for each own module, its `.olean`, how many constants it declares (`constants`), how many of them the
  closure of the pins holds (`used`), the first pin whose closure reaches one of them (`by`), and the other own modules
  whose constants its own constants use (`refs`: what a module needs from the rest to be rebuilt as it is).
- `imports`: what every imported module imports.

The closure of a pin is every constant its kernel check reads: its type and value (a proof included), an inductive type's
constructors and the other types of its mutual block, a constructor's type, a recursor's type and rules, and the same for
each constant those use, transitively. Notation, macros, tactics and attributes act at elaboration and leave no constant in
a term, so the closure never holds them: `unused.py` keeps their modules by what the sources declare.
-/

open Lean

namespace LeanUsed

/-- The module with index `i` (`EnvironmentHeader.moduleNames` maps over every imported module on each call). -/
def moduleName (env : Environment) (i : Nat) : Name := env.header.modules[i]!.module

/-- The constants a constant's kernel check reads. -/
def uses : ConstantInfo → Array Name
  | .axiomInfo v => v.type.getUsedConstants
  | .defnInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .thmInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray ++ v.all.toArray
  | .ctorInfo v => v.type.getUsedConstants.push v.induct
  | .recInfo v => v.rules.foldl (init := v.type.getUsedConstants ++ v.all.toArray) fun acc r => acc ++ r.rhs.getUsedConstants
  | .quotInfo _ => #[]

def names (j : Json) (k : String) : Array Name :=
  ((j.getObjValD k).getArr?.toOption.getD #[]).filterMap fun v => v.getStr?.toOption.map String.toName

end LeanUsed

open LeanUsed in
def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let some cfgPath := args.head? | throw <| IO.userError "usage: Used.lean CONFIG"
  let cfg ← IO.ofExcept (Json.parse (← IO.FS.readFile cfgPath))
  let roots := names cfg "roots"
  let pins := names cfg "pins"
  let extra := names cfg "extra"
  let out := (cfg.getObjValD "out").getStr?.toOption.getD "used.json"
  let t0 ← IO.monoMsNow
  let lap (what : String) (n : Nat) : IO Unit := do
    IO.eprintln s!"used: {what} ({n}) at {(← IO.monoMsNow) - t0} ms"
  let env ← importModules (roots.map fun r => { module := r }) {} (loadExts := false)
  let n := env.header.moduleData.size
  lap "import" n
  let found ← (List.range ((n + 255) / 256)).toArray.mapM fun k => IO.asTask do
    ((List.range n).toArray.extract (256 * k) (256 * k + 256)).mapM fun i => return toString (← findOLean (moduleName env i))
  let mut oleans := #[]
  for t in found do
    oleans := oleans ++ (← IO.ofExcept t.get)
  let own : Array Bool := oleans.map fun o =>
    (o.splitOn "/.lake/build/").length > 1 && (o.splitOn "/.lake/packages/").length == 1
  let ownOf (c : Name) : Option Nat := do
    let i := (← env.getModuleIdxFor? c).toNat
    if own[i]! then some i else none
  let ownIdx := (List.range n).toArray.filter (own[·]!)
  lap "own modules" ownIdx.size
  -- each own constant's uses among the own constants, and each own module's `refs`, a run of modules per task
  let tasks := (List.range ((ownIdx.size + 7) / 8)).toArray.map fun k => Task.spawn fun _ =>
    (ownIdx.extract (8 * k) (8 * k + 8)).map fun i => Id.run do
      let mut cs : Array (Name × Array Name) := #[]
      let mut refs : Std.HashSet Nat := {}
      for ci in env.header.moduleData[i]!.constants do
        let mut ds : Array Name := #[]
        let mut seen : NameSet := {}
        for d in uses ci do
          if seen.contains d then continue
          seen := seen.insert d
          if let some j := ownOf d then
            ds := ds.push d
            if j != i then refs := refs.insert j
        cs := cs.push (ci.name, ds)
      return (i, cs, refs)
  let mut table : Std.HashMap Name (Array Name) := {}
  let mut refsOf : Std.HashMap Nat (Array Nat) := {}
  for t in tasks do
    for (i, cs, refs) in t.get do
      refsOf := refsOf.insert i (refs.toArray.qsort (· < ·))
      for (c, ds) in cs do
        table := table.insert c ds
  lap "uses" table.size
  let mut seen : NameSet := {}
  let mut usedCount : Std.HashMap Nat Nat := {}
  let mut firstBy : Std.HashMap Nat Name := {}
  let mut pinned := #[]
  for p in pins ++ extra do
    if seen.contains p then continue
    let some _ := env.find? p | continue
    let mut stack := #[p]
    while h : stack.size > 0 do
      let c := stack[stack.size - 1]
      stack := stack.pop
      if seen.contains c then continue
      seen := seen.insert c
      let some i := ownOf c | continue
      usedCount := usedCount.insert i (usedCount.getD i 0 + 1)
      unless firstBy.contains i do firstBy := firstBy.insert i p
      stack := stack ++ (table.getD c ((env.find? c).map (fun ci => (uses ci).filter (ownOf · |>.isSome)) |>.getD #[]))
  for p in pins do
    pinned := pinned.push (p.toString, toJson ((env.getModuleIdxFor? p).map fun i => (moduleName env i.toNat).toString))
  lap "closure" seen.size
  let modules := ownIdx.map fun i =>
    (toString (moduleName env i), Json.mkObj [("olean", toJson oleans[i]!),
      ("constants", toJson env.header.moduleData[i]!.constants.size), ("used", toJson (usedCount.getD i 0)),
      ("by", toJson ((firstBy.get? i).map toString)),
      ("refs", toJson ((refsOf.getD i #[]).map fun j => toString (moduleName env j)))])
  let imports := (List.range n).toArray.map fun i =>
    (toString (moduleName env i), toJson (env.header.moduleData[i]!.imports.map (·.module.toString)))
  let doc := Json.mkObj [("lean", toJson Lean.versionString), ("pins", Json.mkObj pinned.toList),
    ("modules", Json.mkObj modules.toList), ("imports", Json.mkObj imports.toList)]
  IO.FS.writeFile out doc.compress
  lap "write" 0
  return 0
