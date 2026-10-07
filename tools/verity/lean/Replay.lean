import Lean

/-!
# Kernel replay of a package's own declarations

`lake env lean --run Replay.lean OUT M₁ M₂ … [--reused R₁ R₂ …]` reads every constant the modules `Mᵢ` and `Rⱼ` declare
from their `.olean` files, imports once the modules they import from outside that set, and sends each constant of the
`Mᵢ` to the kernel (`Kernel.Environment.replay`). A module `Rⱼ` passed this replay before with the same `.olean` files,
its imports' included, on the same toolchain (`audit.py`'s replay cache): its constants go into the environment as an
import adds them, unchecked, and are walked for axioms with the rest. A declaration that never passed the kernel (`debug.skipKernelTC`, `addDeclWithoutChecking`,
a tampered `.olean`) is rejected here, which `#print axioms` cannot see.

The constants go to the kernel side by side, on Lean's threads (`LEAN_NUM_THREADS`), one declaration at a time (an
inductive type with its constructors and recursors, or one other constant). They are put in an order in which each
declaration comes after the ones it mentions, and each is checked in an environment holding that import and every
declaration before it, added as an import adds a module's constants. Each of those is checked by its own task, so every
constant is checked once, against constants that are checked: `leanchecker`'s check per module, on one import rather
than one per module (which runs out of memory on Mathlib-sized imports), and split finer than its modules, so a module
of slow `decide` proofs is spread over the threads too. Unsafe and partial constants, which the kernel does not check,
are in every one of those environments, as an import puts them, and the kernel refuses a checked declaration that
mentions one.

It then writes to `OUT` the axioms those constants use, found by walking their bodies (the ones the kernel accepted, not
the axiom sets each `.olean` recorded when it was built), so `audit.py` can compare the two. Exit 0 when every constant is
accepted, 1 when the kernel rejects one.
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
types, constructors and recursors (`Kernel.Environment.replay` checks them together), else `c` alone. -/
def declOf (constants : Std.HashMap Name ConstantInfo) (c : ConstantInfo) : Name :=
  match c with
  | .inductInfo v => v.all.head?.getD c.name
  | .recInfo v => v.all.head?.getD c.name
  | .ctorInfo v => match constants[v.induct]? with
    | some (.inductInfo i) => i.all.head?.getD v.induct
    | _ => v.induct
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

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let some out := args.head? | throw <| IO.userError "usage: Replay.lean OUT MODULE... [--reused MODULE...]"
  let (checked, reused) := args.tail.span (· != "--reused")
  let reusedMods := (reused.drop 1).toArray.map String.toName
  let reusedSet := reusedMods.foldl NameSet.insert {}
  let mods := checked.toArray.map String.toName ++ reusedMods
  let mut constants : Std.HashMap Name ConstantInfo := {}
  let mut moduleOf : Std.HashMap Name Name := {}
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
      match constants[c.name]?, c with
      | none, _ => constants := constants.insert c.name c; moduleOf := moduleOf.insert c.name m
      | some (.thmInfo a), .thmInfo b =>
        unless a.type == b.type && a.levelParams == b.levelParams do
          throw <| IO.userError s!"{c.name} is declared by {moduleOf[c.name]!} and by {m} with different statements"
      | some _, _ => throw <| IO.userError s!"{c.name} is declared by {moduleOf[c.name]!} and by {m}"
    for i in header.imports do
      if !mods.contains i.module && !imports.any (·.module == i.module) then
        imports := imports.push i
  let env ← importModules imports {}
  -- An equation lemma (`f.eq_1`) is realized in each module that first asks for it, so an import from outside the set
  -- can hold the same theorem: that one, already checked as imported, stands for it.
  for (n, c) in constants do
    match env.find? n, c with
    | none, _ => pure ()
    | some (.thmInfo a), .thmInfo b =>
      unless a.type == b.type && a.levelParams == b.levelParams do
        throw <| IO.userError s!"{n} is declared by {moduleOf[n]!} and by an import from outside the set with different statements"
      constants := constants.erase n
    | some _, _ =>
      throw <| IO.userError s!"{n} is declared by a replayed module and by one it imports from outside the set"
  -- The kernel checks no unsafe or partial constant (compiled code, which may recurse through others), and refuses a
  -- checked one that mentions them, so they go in first, unchecked, and are not ordered.
  let compiled (c : ConstantInfo) := c.isUnsafe || c.isPartial
  let skipped := constants.fold (fun a _ c => if compiled c then a.push c else a) #[]
  let mut kenv := skipped.foldl addImported env.toKernelEnv
  let mut members : Std.HashMap Name (Array ConstantInfo) := {}
  for (_, c) in constants do
    unless compiled c do
      members := members.alter (declOf constants c) fun cs => some ((cs.getD #[]).push c)
  let mut deps : Std.HashMap Name (Array Name) := {}
  for (d, cs) in members do
    let used := cs.foldl (fun s c => s ++ c.getUsedConstantsAsSet) ({} : NameSet)
    deps := deps.insert d <| used.toArray.filterMap fun n =>
      constants[n]?.filter (!compiled ·) |>.map (declOf constants) |>.filter (· != d)
  let order ← match declOrder (members.keys.toArray.qsort Name.lt) deps with
    | .ok order => pure order
    | .error d =>
      IO.eprintln s!"replay: {d} (in {moduleOf[d]!}) is on a cycle of declarations that mention each other"
      return 1
  let mut tasks := #[]
  let mut kept := 0
  for d in order do
    let cs := members.getD d #[]
    if reusedSet.contains moduleOf[d]! then
      kept := kept + cs.size
      for c in cs do
        kenv := addImported kenv c
      continue
    let before := kenv
    tasks := tasks.push (d, ← IO.asTask do
      let t ← IO.monoMsNow
      discard <| before.replay (cs.foldl (fun h c => h.insert c.name c) {})
      return (← IO.monoMsNow) - t)
    for c in cs do
      kenv := addImported kenv c
  let mut took : Array (Nat × Name) := #[]
  let mut rejected := false
  for (d, t) in tasks do
    match ← IO.wait t with
    | .ok ms => took := took.push (ms, d)
    | .error e =>
      IO.eprintln s!"replay: the kernel rejects {d} (in {moduleOf[d]!}):\n{e}"
      rejected := true
  if rejected then
    return 1
  let names := axiomsIn kenv constants.keys.toArray |>.map toString
  let sent := constants.size - skipped.size - kept
  IO.FS.writeFile out <| (Json.mkObj [("replayed", toJson sent), ("skipped", toJson skipped.size), ("reused", toJson kept),
    ("axioms", toJson names)]).compress
  let slowest := took.qsort (fun a b => a.1 > b.1) |>.extract 0 3 |>.map fun (ms, d) => s!"{d} {ms / 1000}.{ms % 1000 / 100} s"
  IO.println s!"replay: {sent} constants of {mods.size - reusedMods.size} modules accepted by the kernel ({skipped.size} \
    more are unsafe or partial: compiled code, which the kernel does not check; {kept} more, of {reusedMods.size} modules \
    that passed before, added unchecked), as {tasks.size} declarations side by side \
    (slowest: {", ".intercalate slowest.toList}); axioms of their bodies: {", ".intercalate names.toList}"
  return 0
