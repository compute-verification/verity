import Lean

/-!
# Kernel replay of a package's declarations, and of its import closure

`lake env lean --run Replay.lean OUT M₁ M₂ … [--reused R₁ …] [--closure D₁ …] [--closure-reused E₁ …]` reads every
constant the named modules declare from their `.olean` files, imports once the modules they import from outside that set,
and sends each constant of the `Mᵢ` and the `Dᵢ` to the kernel (`Kernel.Environment.replay`). The `Mᵢ` and `Rᵢ` are the
package's own modules, the `Dᵢ` and `Eᵢ` modules of its import closure (Mathlib, ArkLib, the toolchain's own). A module
`Rᵢ` or `Eᵢ` passed this replay before with the same `.olean` files, its imports' included, on the same toolchain
(`check.py`'s replay cache): its constants go into the environment as an import adds them, unchecked. Every module a
module of the set imports is in the set or imported, and no imported module imports one of the set (`check.py` sends the
set closed upward), so with an empty cache and the whole closure in the set nothing is imported: each declaration of the
closure goes to the kernel once, `leanchecker --fresh`'s check, and a warm cache sends none. A declaration that never
passed the kernel (`debug.skipKernelTC`, `addDeclWithoutChecking`, a tampered `.olean`) is rejected here, which
`#print axioms` cannot see.

The constants go to the kernel side by side, on Lean's threads (`LEAN_NUM_THREADS`), one declaration at a time (an
inductive type with its constructors and recursors, the quotient's four constants, or one other constant). They are put
in an order in which each declaration comes after the ones it mentions, and each is checked in an environment holding
the import, the reused constants and every declaration before it, added as an import adds a module's constants. Each of
those is checked by its own task, so every constant is checked once, against constants that are checked: `leanchecker`'s
check per module, on one import rather than one per module (which runs out of memory on Mathlib-sized imports), and split
finer than its modules, so a module of slow `decide` proofs is spread over the threads too. At most `WINDOW` of those
tasks wait at once, each holding its environment. Unsafe and partial constants, which the kernel does not check, are in
every one of those environments, as an import puts them, and the kernel refuses a checked declaration that mentions one.

A replayed module's verdict is cached under a key of its own `.olean` files and its imports', so it must depend on
nothing else: each constant a checked declaration mentions must be declared by its module or by a module it imports,
transitively, and a theorem two modules declare with one statement (an equation lemma each realized) is checked in both,
the second under a fresh name. The quotient goes to the kernel in this thread, and its constants must be the ones the
kernel makes.

It then writes to `OUT` the axioms the package's own constants (the `Mᵢ` and `Rᵢ`) use, found by walking their bodies
(the ones the kernel accepted, not the axiom sets each `.olean` recorded when it was built), so `check.py` can compare the
two. Exit 0 when every constant is accepted, 1 when the kernel rejects one, after writing to `OUT` the modules of the
rejected declarations.
-/

open Lean Elab Term

/-- `Lean.Kernel.Environment.add`, which is private to `Lean.Environment`: the kernel environment's insertion of a constant
it does not check, as `importModules` inserts each imported constant. -/
elab "kernelEnvironmentAdd%" : term => do
  let n := Name.mkNum (`_private ++ `Lean.Environment) 0 ++ `Lean.Kernel.Environment.add
  unless (← getEnv).contains n do
    throwError "{n} is gone from this toolchain: find the insertion `importModules` uses now"
  return mkConst n

/-- `env` with `c` added unchecked, as an import adds it. -/
def addImported : Kernel.Environment → ConstantInfo → Kernel.Environment := kernelEnvironmentAdd%

/-- How many declarations may wait for a thread at once, each task holding the environment it is checked in. -/
def WINDOW : Nat := 8192

/-- The axioms reachable from `roots` through the bodies in `env`. -/
def axiomsIn (env : Kernel.Environment) (roots : Array Name) : Array Name := Id.run do
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
      match env.find? c with
      | some (.axiomInfo v) => axs := axs.insert c; stack := stack ++ v.type.getUsedConstants
      | some (.defnInfo v) => stack := stack ++ v.type.getUsedConstants ++ v.value.getUsedConstants
      | some (.thmInfo v) => stack := stack ++ v.type.getUsedConstants ++ v.value.getUsedConstants
      | some (.opaqueInfo v) => stack := stack ++ v.type.getUsedConstants ++ v.value.getUsedConstants
      | some (.inductInfo v) => stack := stack ++ v.type.getUsedConstants ++ v.ctors.toArray
      | some (.ctorInfo v) => stack := stack ++ v.type.getUsedConstants
      | some (.recInfo v) => stack := stack ++ v.type.getUsedConstants
      | some (.quotInfo _) | none => pure ()
  return axs.toArray.qsort Name.lt

/-- The declaration `c` goes to the kernel in, named by its first constant: an inductive type's block, for the block's
types, constructors and recursors (`Kernel.Environment.replay` checks them together), `Quot` for the quotient's
constants (one `quotDecl` adds the four), else `c` alone. -/
def declOf (constants : Std.HashMap Name ConstantInfo) (c : ConstantInfo) : Name :=
  match c with
  | .inductInfo v => v.all.head?.getD c.name
  | .recInfo v => v.all.head?.getD c.name
  | .ctorInfo v => match constants[v.induct]? with
    | some (.inductInfo i) => i.all.head?.getD v.induct
    | _ => v.induct
  | .quotInfo _ => `Quot
  | _ => c.name

/-- `decls` with each after the declarations it mentions (`deps`), or the first declaration found on a cycle among
them, which no environment the kernel built holds. -/
def declOrder (decls : Array Name) (deps : Std.HashMap Name (Array Name)) : Except Name (Array Name) := Id.run do
  let mut order := #[]
  let mut done : Std.HashMap Name Bool := {}     -- false while its dependencies are being ordered
  for root in decls do
    let mut stack : Array (Name × Bool) := #[(root, false)]
    repeat
      match stack.back? with
      | none => break
      | some (d, expanded) =>
        stack := stack.pop
        if expanded then
          done := done.insert d true
          order := order.push d
        else match done[d]? with
          | some true => continue
          | some false => return .error d
          | none =>
            done := done.insert d false
            stack := stack.push (d, true)
            for e in deps.getD d #[] do
              if done[e]? != some true then
                stack := stack.push (e, false)
  return .ok order

def sameKind : QuotKind → QuotKind → Bool
  | .type, .type | .ctor, .ctor | .lift, .lift | .ind, .ind => true
  | _, _ => false

/-- The modules `m` imports, transitively, and `m` itself. -/
def closureOf (importsOf : Std.HashMap Name (Array Name)) (m : Name) : Std.HashSet Name := Id.run do
  let mut seen : Std.HashSet Name := ({} : Std.HashSet Name).insert m
  let mut stack := #[m]
  repeat
    match stack.back? with
    | none => break
    | some n =>
      stack := stack.pop
      for i in importsOf.getD n #[] do
        unless seen.contains i do
          seen := seen.insert i
          stack := stack.push i
  return seen

/-- The arguments after `OUT`, by section: the own modules to check, then each flag's list. -/
def sections (args : List String) : Std.HashMap String (Array Name) := Id.run do
  let mut out : Std.HashMap String (Array Name) := {}
  let mut cur := ""
  for a in args do
    if a.startsWith "--" then cur := a
    else out := out.alter cur fun ms => some ((ms.getD #[]).push a.toName)
  return out

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let t0 ← IO.monoMsNow
  let some out := args.head? |
    throw <| IO.userError "usage: Replay.lean OUT MODULE... [--reused MODULE...] [--closure MODULE...] [--closure-reused MODULE...]"
  let parts := sections args.tail
  for k in parts.keys do
    unless ["", "--reused", "--closure", "--closure-reused"].contains k do
      throw <| IO.userError s!"Replay.lean: unknown flag {k}"
  let get k := parts.getD k #[]
  let own := get "" ++ get "--reused"
  let reusedSet := (get "--reused" ++ get "--closure-reused").foldl NameSet.insert {}
  let ownSet := own.foldl NameSet.insert {}
  -- the closure's modules first, so that a theorem a dependency and an own module both declare is the dependency's
  let mods := get "--closure" ++ get "--closure-reused" ++ own
  let modSet := mods.foldl NameSet.insert {}
  let mut constants : Std.HashMap Name ConstantInfo := {}
  let mut moduleOf : Std.HashMap Name Name := {}
  let mut alsoIn : Std.HashMap Name (Array Name) := {}     -- the other modules that declare a duplicated theorem
  let mut dups : Array (Name × ConstantInfo) := #[]      -- each second declaration of one, with its module
  let mut importsOf : Std.HashMap Name (Array Name) := {}
  let mut imports : Array Import := #[]
  for m in mods do
    let olean ← findOLean m
    let mut files := #[olean]
    for level in [OLeanLevel.server, OLeanLevel.private] do
      if ← (level.adjustFileName olean).pathExists then
        files := files.push (level.adjustFileName olean)
    let parts ← readModuleDataParts files
    let (some (data, _), some (header, _)) := (parts.back?, parts[0]?) | throw <| IO.userError s!"no module data for {m}"
    importsOf := importsOf.insert m (header.imports.map (·.module))
    for c in data.constants do
      match constants[c.name]?, c with
      | none, _ => constants := constants.insert c.name c; moduleOf := moduleOf.insert c.name m
      | some (.thmInfo a), .thmInfo b =>
        unless a.type == b.type && a.levelParams == b.levelParams do
          throw <| IO.userError s!"{c.name} is declared by {moduleOf[c.name]!} and by {m} with different statements"
        alsoIn := alsoIn.alter c.name fun ms => some ((ms.getD #[]).push m)
        unless reusedSet.contains m do
          dups := dups.push (m, c)
      | some _, _ => throw <| IO.userError s!"{c.name} is declared by {moduleOf[c.name]!} and by {m}"
    for i in header.imports do
      if !modSet.contains i.module && !imports.any (·.module == i.module) then
        imports := imports.push i
  let env ← if imports.isEmpty then mkEmptyEnvironment else importModules imports {}
  let tLoaded ← IO.monoMsNow
  for h : i in [0:env.header.moduleNames.size] do
    importsOf := importsOf.insert env.header.moduleNames[i] (env.header.moduleData[i]!.imports.map (·.module))
  -- An equation lemma (`f.eq_1`) is realized in each module that first asks for it, so an import from outside the set
  -- can hold the same theorem: the import's stays in the environment, and the set's is checked under a fresh name.
  for (n, c) in constants do
    match env.find? n, c with
    | none, _ => pure ()
    | some (.thmInfo a), .thmInfo b =>
      unless a.type == b.type && a.levelParams == b.levelParams do
        throw <| IO.userError s!"{n} is declared by {moduleOf[n]!} and by an import from outside the set with different statements"
      let m := moduleOf[n]!
      constants := constants.erase n
      moduleOf := moduleOf.erase n
      if let some i := env.getModuleIdxFor? n then
        if let some im := env.header.moduleNames[i.toNat]? then
          moduleOf := moduleOf.insert n im
      alsoIn := alsoIn.alter n fun ms => some ((ms.getD #[]).push m)
      unless reusedSet.contains m do
        dups := dups.push (m, c)
    | some _, _ =>
      throw <| IO.userError s!"{n} is declared by a replayed module and by one it imports from outside the set"
  -- after the loop above, which moves a theorem an import from outside the set also declares: a closure takes the values
  -- `moduleOf` and `alsoIn` have where it is defined
  let declaredIn (n : Name) : Array Name :=
    let here := match moduleOf[n]? with
      | some m => #[m]
      | none => ((env.getModuleIdxFor? n).bind fun i => env.header.moduleNames[i.toNat]?).toArray
    here ++ alsoIn.getD n #[]
  -- A second declaration of a theorem gets a name no module declares, so the kernel checks its proof too.
  let mut renamed : Std.HashMap Name ConstantInfo := {}
  let mut renamedFrom : Std.HashMap Name (Name × Name) := {}   -- fresh name ↦ (the theorem, its module)
  for h : i in [0:dups.size] do
    let (m, c) := dups[i]
    let .thmInfo v := c | unreachable!
    let fresh := Name.mkNum (c.name ++ `_replay_second) i
    renamed := renamed.insert fresh (.thmInfo { v with name := fresh, all := [fresh] })
    renamedFrom := renamedFrom.insert fresh (c.name, m)
  -- The kernel checks no unsafe or partial constant (compiled code, which may recurse through others), and refuses a
  -- checked one that mentions them, so they go in first, unchecked, and are not ordered; so do the reused modules'.
  let compiled (c : ConstantInfo) := c.isUnsafe || c.isPartial
  let mut kenv := env.toKernelEnv
  let mut skipped := 0
  let mut kept := 0
  let mut members : Std.HashMap Name (Array ConstantInfo) := {}
  for (n, c) in constants do
    if compiled c then
      kenv := addImported kenv c
      skipped := skipped + 1
    else if reusedSet.contains moduleOf[n]! then
      kenv := addImported kenv c
      kept := kept + 1
    else
      members := members.alter (declOf constants c) fun cs => some ((cs.getD #[]).push c)
  for (n, c) in renamed do
    members := members.insert n #[c]
  let moduleOfDecl (d : Name) : Name := match renamedFrom[d]? with
    | some (_, m) => m
    | none => moduleOf[d]!
  -- each constant a checked declaration mentions is declared by its module or by one that module imports
  let mut byModule : Std.HashMap Name (Array Name) := {}
  for (d, _) in members do
    byModule := byModule.alter (moduleOfDecl d) fun ds => some ((ds.getD #[]).push d)
  let mut deps : Std.HashMap Name (Array Name) := {}
  for (m, ds) in byModule do
    let reach := closureOf importsOf m
    for d in ds do
      let cs := members.getD d #[]
      let used := cs.foldl (fun s c => s ++ c.getUsedConstantsAsSet) ({} : NameSet)
      for n in used do
        let from_ := declaredIn n
        unless from_.isEmpty || from_.any reach.contains do
          IO.eprintln s!"replay: {d} (in {m}) mentions {n}, declared by {", ".intercalate (from_.toList.map toString)}, \
            which {m} does not import"
          return 1
      deps := deps.insert d <| used.toArray.filterMap fun n => do
        let c ← constants[n]?
        guard !(compiled c)
        let e := declOf constants c
        guard (e != d && members.contains e)
        return e
  let tPlanned ← IO.monoMsNow
  let order ← match declOrder (members.keys.toArray.qsort Name.lt) deps with
    | .ok order => pure order
    | .error d =>
      IO.eprintln s!"replay: {d} (in {moduleOfDecl d}) is on a cycle of declarations that mention each other"
      return 1
  let mut tasks : Array (Name × Task (Except IO.Error Nat)) := #[]
  let mut took : Array (Nat × Name) := #[]
  let mut rejected : NameSet := {}
  let mut waited := 0
  let mut checkedModules : NameSet := {}
  for d in order do
    let cs := members.getD d #[]
    checkedModules := checkedModules.insert (moduleOfDecl d)
    if cs.any (· matches .quotInfo _) then
      -- `quotDecl` makes the four constants itself; they must be the ones the `.olean` holds
      match ← (kenv.replay (cs.foldl (fun h c => h.insert c.name c) {})).toBaseIO with
      | .error e =>
        IO.eprintln s!"replay: the kernel rejects the quotient (in {moduleOfDecl d}):\n{e}"
        return 1
      | .ok k =>
        for c in cs do
          match k.find? c.name, c with
          | some (.quotInfo a), .quotInfo b =>
            unless a.type == b.type && a.levelParams == b.levelParams && sameKind a.kind b.kind do
              IO.eprintln s!"replay: {c.name} (in {moduleOfDecl d}) is not the quotient constant the kernel makes"
              return 1
          | _, _ =>
            IO.eprintln s!"replay: {c.name} (in {moduleOfDecl d}) is not a quotient constant the kernel makes"
            return 1
        kenv := k
      continue
    let before := kenv
    tasks := tasks.push (d, ← IO.asTask do
      let t ← IO.monoMsNow
      discard <| before.replay (cs.foldl (fun h c => h.insert c.name c) {})
      return (← IO.monoMsNow) - t)
    for c in cs do
      unless renamedFrom.contains c.name do
        kenv := addImported kenv c
    while tasks.size - waited > WINDOW do
      let (d', t) := tasks[waited]!
      match ← IO.wait t with
      | .ok ms => took := took.push (ms, d')
      | .error e =>
        IO.eprintln s!"replay: the kernel rejects {d'} (in {moduleOfDecl d'}):\n{e}"
        rejected := rejected.insert (moduleOfDecl d')
      waited := waited + 1
  for (d, t) in tasks.extract waited tasks.size do
    match ← IO.wait t with
    | .ok ms => took := took.push (ms, d)
    | .error e =>
      IO.eprintln s!"replay: the kernel rejects {d} (in {moduleOfDecl d}):\n{e}"
      rejected := rejected.insert (moduleOfDecl d)
  if !rejected.isEmpty then
    -- every other module's declarations passed with its imports present, which is what a module's cache entry says
    IO.FS.writeFile out <| (Json.mkObj [("rejected", toJson (rejected.toArray.qsort Name.lt |>.map toString))]).compress
    return 1
  let tChecked ← IO.monoMsNow
  let ownConstants := (constants.toArray.filter fun (n, _) => ownSet.contains moduleOf[n]!).map (·.1)
  let names := axiomsIn kenv ownConstants |>.map toString
  let sent := members.fold (fun a _ cs => a + cs.size) 0
  -- three passes, not `qsort`, which is quadratic on the many declarations that take 0 ms
  let top := (List.range 3).foldl (init := (#[] : Array (Nat × Name))) fun top _ =>
    match took.foldl (init := none) fun (best : Option (Nat × Name)) x =>
        if top.any (·.2 == x.2) then best else match best with
          | some b => if x.1 > b.1 then some x else best
          | none => some x with
    | some b => top.push b
    | none => top
  let slowest := top.map fun (ms, d) => s!"{d} {ms / 1000}.{ms % 1000 / 100} s"
  IO.println s!"replay: {sent} constants of {checkedModules.size} modules accepted by the kernel ({renamed.size} of them \
    a theorem another module declares too, checked under a fresh name; {skipped} more are unsafe or partial: compiled code, \
    which the kernel does not check; {kept} more, of {reusedSet.size} modules that passed before, added unchecked; \
    {env.header.moduleNames.size} modules imported unchecked), as {tasks.size} declarations side by side \
    (slowest: {", ".intercalate slowest.toList}); axioms of the own modules' bodies: {", ".intercalate names.toList}"
  let tEnd ← IO.monoMsNow
  let seconds := [("load", tLoaded - t0), ("plan", tPlanned - tLoaded), ("check", tChecked - tPlanned),
    ("axioms", tEnd - tChecked)].map fun (k, ms) => (k, toJson (ms / 1000))
  IO.println s!"replay: seconds {Json.mkObj seconds |>.compress}"
  IO.FS.writeFile out <| (Json.mkObj [("replayed", toJson sent), ("skipped", toJson skipped), ("reused", toJson kept),
    ("second_copies", toJson renamed.size), ("closure_checked", toJson (get "--closure").size),
    ("closure_reused", toJson (get "--closure-reused").size), ("imported", toJson env.header.moduleNames.size),
    ("axioms", toJson names), ("seconds", Json.mkObj seconds)]).compress
  return 0
