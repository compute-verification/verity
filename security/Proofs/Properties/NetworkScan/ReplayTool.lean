import Lean

/-!
# Kernel replay of a package's declarations, and of its import closure

`lake env lean --run Replay.lean OUT M₁ M₂ … [--reused R₁ …] [--closure D₁ …] [--closure-reused E₁ …]` reads every
constant the named modules declare from their `.olean` files (a `module`'s from its private part, `.olean.private`, the
only part that holds its proofs and unexposed bodies, so a `module` missing it or its server part is refused; a file
outside the module system from its `.olean` alone, whatever lies beside it), imports once the modules they import from
outside that set, and sends each constant of the `Mᵢ` and the `Dᵢ` to the kernel (`Kernel.Environment.replay`). The `Mᵢ`
and `Rᵢ` are the package's own modules, the `Dᵢ` and `Eᵢ` modules of its import closure (Mathlib, ArkLib, the
toolchain's own). A module `Rᵢ` or `Eᵢ` passed this replay before with the same `.olean` files, its imports' included,
on the same toolchain (`check.py`'s replay cache): its constants go into the environment as an import adds them,
unchecked. Every module a module of the set imports is in the set or imported, and no imported module imports one of the
set (`check.py` sends the set closed upward), so with an empty cache and the whole closure in the set nothing is
imported: each declaration of the closure goes to the kernel once, `leanchecker --fresh`'s check, and a warm cache sends
none. A declaration that never passed the kernel (`debug.skipKernelTC`, `addDeclWithoutChecking`, a tampered `.olean`)
is rejected here, which `#print axioms` cannot see.

The constants go to the kernel side by side, on Lean's threads (`LEAN_NUM_THREADS`), one declaration at a time (an
inductive type with its constructors and recursors, the quotient's four constants, or one other constant). They are put
in an order in which each declaration comes after the ones it mentions, and each is checked in an environment holding
the import, the reused constants and every declaration before it, added as an import adds a module's constants. Each of
those is checked by its own task, so every constant is checked once, against constants that are checked: `leanchecker`'s
check per module, on one import rather than one per module (which runs out of memory on Mathlib-sized imports), and split
finer than its modules, so a module of slow `decide` proofs is spread over the threads too. At most `WINDOW` of those
tasks wait at once, each holding its environment. Unsafe and partial constants, which the kernel does not check, are in
every one of those environments, as an import puts them, and the kernel refuses a checked declaration that mentions one.

`lean` checks a module under the module system on its imports' public parts, where a theorem or a definition they don't
expose is an axiom; with those bodies at hand the kernel can unfold its way into a computation it doesn't finish, and it
stops on its recursion. Such a declaration is checked again as `lean` checked it: on its module's other constants and the
constants its imports' public parts hold, each as this replay holds it (checked or imported), with the bodies those parts
hide taken out. `OUT` names the declarations accepted that way (`on_public_imports`).

A replayed module's verdict is cached under a key of its own `.olean` files and its imports', so it must depend on
nothing else: each constant a checked declaration mentions must be declared by its module or by a module it imports,
transitively, and a theorem two modules declare with one statement (an equation lemma each realized) is checked in both,
the second under a fresh name. A module `--closure-reused` names, whose own replay passed at its key (its graph task, or
`check.py`'s cache), may hold its copy as the axiom its public part makes of it, which that replay recorded the axioms
of; an import from outside the set, an own module and any module under `--env-out` may not. Copies that are all such
axioms must state a proposition, whatever bodies they hide. Lean's import takes two declarations of a name on those terms
(`subsumesInfo`), and this refuses any other. The quotient goes to the kernel in this thread, and its constants must be
the ones the kernel makes.

It then writes to `OUT` the axioms the package's own constants (the `Mᵢ` and `Rᵢ`) use, found by walking their bodies
(the ones the kernel accepted, not the axiom sets each `.olean` recorded when it was built), so `check.py` can compare the
two. Exit 0 when every constant is accepted, 1 when the kernel rejects one, after writing to `OUT` the modules of the
rejected declarations.

`Replay.lean OUT --records DIR M₁ … [--closure-reused E₁ …]` (the graph's lean-module runner) also writes the axioms of
each own constant (`declarations`, those with any), and walks no `Eᵢ` that has a record: an `Eᵢ`'s constants take the sets
its record `DIR/<its path>.json` holds (`{"module": E, "axioms": {name: [axiom, …]}}`, what this writes for `E`, a constant
with none left out). An `Eᵢ` is read from the parts its files have, so under the module system its public part alone
serves, where its theorems are axioms; one without a record is walked as those parts have it. A module read without its
private part brings in only what its `public import`s reach, as `lean` loads its importers: `importModulesCore` loads a
module's other imports only below an `import all`, so the runner reads no more.

`--env FILE` (before the modules) takes the sets of the imported modules' constants from FILE, the lean env's record
(`verity_graph/runners/lean_env.py` writes it: a line `"sets":[…],` naming each set once, then after a line `"axioms":{`
one line `"<module>":{"<constant>":<index into sets>,…}` per module, a constant with none left out), and walks no
constant of a module it holds; `OUT` then also names every module imported (`env_modules`), so the runner can hold them
to the record. `--env-out FILE` (the graph's lean-env runner) writes to FILE, a line each,
`{"module":M,"axioms":{…},"unsafe":[…]}` for every module of the set, own and closure alike (`unsafe`: its unsafe
constants with a set, which the kernel never checks and no checked declaration mentions), and last
`{"deferred":[…],"joined":{M:{…}}}`; a theorem two of them declare takes the union of the copies' sets under each, and
`joined` gives, for a module imported from an earlier replay, the sets of the theorems it declares that one of the set
declares too, for its record to take. Module names are written unescaped, as their files' paths have them. It checks a
set that can't all be imported together, as an env's modules can't (each executable's root module declares `main`): in an
order with each module after those it imports, a module that declares a constant a module before it declares too (not one
theorem with one statement), or that imports such a module, is deferred, left out of this replay and named in `deferred`
for the next, which imports what it imports from this one.
A module under the module system whose private part is missing is refused there too, since its public part makes its
theorems axioms; only `--records` and `--env` without `--env-out` read a module from the parts it has.
`--closure-of R₁ …` adds to `--closure` the `Rᵢ` and every module they import, transitively; an argument `@F` stands for
F's lines.
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

/-- What `axiomsIn` walks through from `c`, and `c` itself when it is an axiom. -/
def mentions (c : Name) : Option ConstantInfo → Array Name × Std.HashSet String
  | some (.axiomInfo v) => (v.type.getUsedConstants, ({} : Std.HashSet String).insert (toString c))
  | some (.defnInfo v) => (v.type.getUsedConstants ++ v.value.getUsedConstants, {})
  | some (.thmInfo v) => (v.type.getUsedConstants ++ v.value.getUsedConstants, {})
  | some (.opaqueInfo v) => (v.type.getUsedConstants ++ v.value.getUsedConstants, {})
  | some (.inductInfo v) => (v.type.getUsedConstants ++ v.ctors.toArray, {})
  | some (.ctorInfo v) => (v.type.getUsedConstants, {})
  | some (.recInfo v) => (v.type.getUsedConstants, {})
  | some (.quotInfo _) | none => (#[], {})

/-- The axioms each constant reached from `roots` uses (`axiomsIn`'s walk, one set per constant), where `recorded` gives
the set of a constant whose module recorded one, and the walk stops there. The constants of a strongly connected
component (an inductive type and its constructors mention each other) share one set: Tarjan's algorithm, iterative, so a
long chain of declarations takes no stack. -/
def axiomSets (find? : Name → Option ConstantInfo) (recorded : Name → IO (Option (Std.HashSet String)))
    (roots : Array Name) : IO (Std.HashMap Name (Std.HashSet String)) := do
  let mut done : Std.HashMap Name (Std.HashSet String) := {}   -- each constant of a finished component, with its set
  let mut index : Std.HashMap Name Nat := {}
  let mut low : Std.HashMap Name Nat := {}
  let mut reach : Std.HashMap Name (Std.HashSet String) := {}  -- what each open constant reaches so far
  let mut opened : Array Name := #[]
  let mut isOpen : Std.HashSet Name := {}
  for root in roots do
    if done.contains root || index.contains root then continue
    let mut frames : Array (Name × Array Name × Nat) := #[]
    let mut visit : Option Name := some root
    repeat
      if let some v := visit then
        visit := none
        let (ms, own) := mentions v (find? v)
        low := low.insert v index.size
        index := index.insert v index.size
        reach := reach.insert v own
        opened := opened.push v
        isOpen := isOpen.insert v
        frames := frames.push (v, ms, 0)
      let some (v, ms, i) := frames.back? | break
      if h : i < ms.size then
        frames := frames.modify (frames.size - 1) fun (v, ms, i) => (v, ms, i + 1)
        let w := ms[i]
        if let some s := done[w]? then
          reach := reach.modify v (·.insertMany s)
        else match ← recorded w with
          | some s =>
            done := done.insert w s
            reach := reach.modify v (·.insertMany s)
          | none =>
            match index[w]? with
            | some iw => if isOpen.contains w then low := low.modify v (min · iw)
            | none => visit := some w
      else
        frames := frames.pop
        if low[v]! == index[v]! then
          let mut members : Array Name := #[]
          let mut s : Std.HashSet String := {}
          repeat
            let some m := opened.back? | break
            opened := opened.pop
            isOpen := isOpen.erase m
            members := members.push m
            s := s.insertMany (reach.getD m {})
            if m == v then break
          for m in members do
            done := done.insert m s
            reach := reach.erase m
        if let some (u, _, _) := frames.back? then
          match done[v]? with
          | some s => reach := reach.modify u (·.insertMany s)
          | none => low := low.modify u (min · low[v]!)
  return done

/-- The sets `DIR/<m's path>.json` records for `m`'s constants, or none when `DIR` holds no record of `m`. -/
def readRecord (dir : System.FilePath) (m : Name) : IO (Option (Std.HashMap String (Std.HashSet String))) := do
  let f := modToFilePath dir m "json"
  unless ← f.pathExists do return none
  let axs ← IO.ofExcept <| (Json.parse (← IO.FS.readFile f)).bind (·.getObjVal? "axioms") |>.bind Json.getObj?
  let mut out : Std.HashMap String (Std.HashSet String) := {}
  for (n, v) in axs do
    let mut s : Std.HashSet String := {}
    for a in ← IO.ofExcept v.getArr? do
      s := s.insert (← IO.ofExcept a.getStr?)
    out := out.insert n s
  return some out

/-- The lean env's record (`--env FILE`): the sets it names, and each module's line by the module's name, which
`EnvRecord.module` parses once a constant of that module is looked up. -/
structure EnvRecord where
  sets : Array (Std.HashSet String)
  lines : Std.HashMap String String

def readEnvRecord (f : System.FilePath) : IO EnvRecord := do
  let mut sets : Array (Std.HashSet String) := #[]
  let mut lines : Std.HashMap String String := {}
  let mut inAxioms := false
  for line in ← IO.FS.lines f do
    if inAxioms then
      if line.startsWith "\"" then
        lines := lines.insert ((line.drop 1).takeWhile (· != '"')).toString line
    else if line.startsWith "\"axioms\":{" then
      inAxioms := true
    else if line.startsWith "\"sets\":" then
      let obj := "{" ++ (if line.endsWith "," then (line.dropEnd 1).toString else line) ++ "}"
      for s in ← IO.ofExcept ((Json.parse obj).bind (·.getObjVal? "sets") |>.bind Json.getArr?) do
        let mut h : Std.HashSet String := {}
        for a in ← IO.ofExcept s.getArr? do
          h := h.insert (← IO.ofExcept a.getStr?)
        sets := sets.push h
  unless inAxioms do
    throw <| IO.userError s!"{f} isn't a lean env record: no line `\"axioms\":\{`"
  return { sets, lines }

/-- The sets the record holds for module `m`'s constants, or none when it doesn't hold `m`. -/
def EnvRecord.module (r : EnvRecord) (m : String) : IO (Option (Std.HashMap String (Std.HashSet String))) := do
  let some line := r.lines[m]? | return none
  let obj := "{" ++ (if line.endsWith "," then (line.dropEnd 1).toString else line) ++ "}"
  let axs ← IO.ofExcept <| (Json.parse obj).bind (·.getObjVal? m) |>.bind Json.getObj?
  let mut out : Std.HashMap String (Std.HashSet String) := {}
  for (n, v) in axs do
    let i ← IO.ofExcept v.getNat?
    let some s := r.sets[i]? | throw <| IO.userError s!"the lean env record's {m} names set {i}, which it lacks"
    out := out.insert n s
  return some out

/-- The modules of `loaded` in an order with each after the ones of `loaded` it imports, ties in `loaded`'s order. -/
def importOrder (loaded : Array (Name × Array Import × ModuleData)) : Array Nat := Id.run do
  let idx : Std.HashMap Name Nat := (List.range loaded.size).foldl (fun h i => h.insert loaded[i]!.1 i) {}
  let mut order := #[]
  let mut done : Std.HashSet Nat := {}
  for root in [0:loaded.size] do
    let mut stack : Array (Nat × Bool) := #[(root, false)]
    repeat
      let some (i, expanded) := stack.back? | break
      stack := stack.pop
      if done.contains i then continue
      if expanded then
        done := done.insert i
        order := order.push i
      else
        stack := stack.push (i, true)
        for m in loaded[i]!.2.1.reverse do
          if let some j := idx[m.module]? then
            unless done.contains j do stack := stack.push (j, false)
  return order

/-- The modules of `loaded` that can't be imported with the ones before them (`importOrder`): one that declares a
constant a module kept before it declares too, but for a theorem both declare with one statement, and one that imports
a module deferred. -/
def deferredOf (loaded : Array (Name × Array Import × ModuleData)) : NameSet := Id.run do
  let mut declared : Std.HashMap Name ConstantInfo := {}
  let mut deferred : NameSet := {}
  for i in importOrder loaded do
    let (m, imports, data) := loaded[i]!
    let clash := imports.any (deferred.contains ·.module) || data.constants.any fun c =>
      match declared[c.name]?, c with
      | none, _ => false
      | some (.thmInfo a), .thmInfo b => !(a.type == b.type && a.levelParams == b.levelParams)
      | some _, _ => true
    if clash then
      deferred := deferred.insert m
    else
      for c in data.constants do
        unless declared.contains c.name do declared := declared.insert c.name c
  return deferred

/-- `args` with each `@F` replaced by the lines of F that aren't empty. -/
def expandArgs (args : List String) : IO (List String) := do
  let mut out : Array String := #[]
  for a in args do
    if a.startsWith "@" then
      for l in ← IO.FS.lines (a.drop 1).toString do
        unless l.isEmpty do out := out.push l
    else
      out := out.push a
  return out.toList

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

/-- Why `b`, of module `mb`, can't be one constant with `a`, of module `ma`, declared before it under its name, or none
when it can: as an import takes two declarations of a name (`subsumesInfo`), theorems, or a theorem and the axiom a
`module`'s public part makes of it, or two such axioms, with one statement. An axiom copy must come from a module
`readPublic` names, one `--closure-reused` names: its own replay passed at its key, putting a theorem's proof through the
kernel and recording a written axiom as one, so this replay may read it at the parts it has. A module this replay checks
is read at its private part, where an axiom is one. Two axioms must also state a proposition (`isPropCheap`), which proof
irrelevance makes them one constant, the trust an import already gets. -/
def notOne (readPublic : Name → Bool) (ma mb : Name) (a b : ConstantInfo) : Option String :=
  let axiomFrom (m : Name) : ConstantInfo → Option String
    | .axiomInfo v =>
      if v.isUnsafe then some s!"as an unsafe axiom in {m}"
      else if readPublic m then none
      else some s!"as an axiom in {m}, which `--closure-reused` doesn't name: only a module whose own replay passed \
        may hold a theorem as the axiom its public part makes of it"
    | _ => none
  let single : ConstantInfo → Bool
    | .thmInfo t => t.all == [t.name]
    | _ => true
  if !(a.type == b.type && a.levelParams == b.levelParams) then some "with different statements"
  else match a, b with
    | .thmInfo _, .thmInfo _ => none
    | .thmInfo _, .axiomInfo _ | .axiomInfo _, .thmInfo _ | .axiomInfo _, .axiomInfo _ =>
      if single a && single b then axiomFrom ma a <|> axiomFrom mb b else some "as a mutual theorem and an axiom"
    | _, _ => some "and not as theorems or axioms"

/-- `subsumesInfo`'s test that a type is a proposition, `∀ …, p xs` with `p : ∀ args, Prop` and as many `xs` as `args`,
which makes two axioms of one name one constant whatever each was before its module's public part made it one (two
definitions, perhaps, with different bodies). -/
def isPropCheap (find? : Name → Option ConstantInfo) (ty : Expr) : Bool := Id.run do
  let mut ty := ty
  while ty.isForall do
    let .forallE (body := body) .. := ty | return false
    ty := body
  let .const n .. := ty.getAppFn | return false
  let some decl := find? n | return false
  let mut p := decl.type
  for _ in [0:ty.getAppNumArgs] do
    let .forallE (body := body) .. := p | return false
    p := body
  return p.isProp

/-- The checked declarations (``checked n``) an unchecked constant of the set reaches through what it mentions, through
other unchecked ones of the set: the kernel unfolds an unchecked definition when it checks a declaration that mentions
it, so whatever that definition's body needs has to be in the kernel first. -/
partial def reachChecked (constants : Std.HashMap Name ConstantInfo) (checked : Name → Option Name)
    (memo : IO.Ref (Std.HashMap Name NameSet)) (onPath : NameSet) (n : Name) : IO NameSet := do
  if let some r := (← memo.get)[n]? then return r
  let some c := constants[n]? | return {}
  let mut out : NameSet := {}
  for m in c.getUsedConstantsAsSet do
    if let some d := checked m then
      out := out.insert d
    else if constants.contains m && !onPath.contains m then
      for d in ← reachChecked constants checked memo (onPath.insert n) m do
        out := out.insert d
  memo.modify (·.insert n out)
  return out

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
  let args ← expandArgs args
  let some out := args.head? |
    throw <| IO.userError "usage: Replay.lean OUT [--records DIR] [--env FILE] [--env-out FILE] MODULE... \
      [--reused MODULE...] [--closure MODULE...] [--closure-reused MODULE...] [--closure-of MODULE...]"
  let mut opts : Std.HashMap String System.FilePath := {}
  let mut rest := args.tail
  repeat
    match rest with
    | k :: v :: r =>
      unless ["--records", "--env", "--env-out"].contains k do break
      opts := opts.insert k v
      rest := r
    | _ => break
  let records := opts["--records"]?
  let envIn := opts["--env"]?
  let envOut := opts["--env-out"]?
  let parts := sections rest
  for k in parts.keys do
    unless ["", "--reused", "--closure", "--closure-reused", "--closure-of"].contains k do
      throw <| IO.userError s!"Replay.lean: unknown flag {k}"
  let get k := parts.getD k #[]
  let own := get "" ++ get "--reused"
  let reusedSet := (get "--reused" ++ get "--closure-reused").foldl NameSet.insert {}
  let ownSet := own.foldl NameSet.insert {}
  -- each module's imports and the data of its last part (its private one, where it has one), read once
  let mut loaded : Array (Name × Array Import × ModuleData) := #[]
  let mut publicOnly : NameSet := {}                     -- each `module` read without its private part
  let mut seen : NameSet := {}
  -- the closure's modules first, so that a theorem a dependency and an own module both declare is the dependency's
  let mut todo := (get "--closure" ++ get "--closure-of" ++ get "--closure-reused" ++ own).toList
  let mut expand := (get "--closure-of").foldl NameSet.insert {}
  while true do
    let m :: next := todo | break
    todo := next
    if seen.contains m then continue
    seen := seen.insert m
    let olean ← findOLean m
    -- as `importModules` reads them: a `module`'s proofs and unexposed bodies are only in its private part, read on top
    -- of its `.olean` and server part; a file outside the module system is its `.olean` alone, whatever lies beside it.
    -- A `module` missing a part is refused, but by `--records` or `--env` without `--env-out`, which read it from the
    -- parts it has (the lean-module runner's imports, whose public part alone serves a `module` importer)
    let (header, region) ← CompactedRegion.read (α := ModuleData) olean #[]
    let mut data := header
    if header.isModule then
      let mut regions := #[region]
      for level in [OLeanLevel.server, OLeanLevel.private] do
        let part := level.adjustFileName olean
        unless ← part.pathExists do
          if envOut.isNone && (records.isSome || envIn.isSome) then break
          throw <| IO.userError s!"{m} is a `module` and {part} is missing: its private part alone holds its proofs, \
            and its public part makes its theorems axioms"
        let (d, r) ← CompactedRegion.read (α := ModuleData) part regions
        data := d
        regions := regions.push r
    loaded := loaded.push (m, header.imports, data)
    if data.isModule && !(← (OLeanLevel.private.adjustFileName olean).pathExists) then
      publicOnly := publicOnly.insert m
    if expand.contains m then
      for i in header.imports do
        unless seen.contains i.module do
          expand := expand.insert i.module
          todo := todo ++ [i.module]
  let deferred := if envOut.isSome then deferredOf loaded else {}
  loaded := loaded.filter fun (m, _, _) => !deferred.contains m
  let mods := loaded.map (·.1)
  let modSet := mods.foldl NameSet.insert {}
  let mut constants : Std.HashMap Name ConstantInfo := {}
  let mut moduleOf : Std.HashMap Name Name := {}
  let mut alsoIn : Std.HashMap Name (Array Name) := {}     -- the other modules that declare a duplicated theorem
  let mut dups : Array (Name × ConstantInfo) := #[]      -- each second declaration of one, with its module
  let mut proved : NameSet := {}                         -- each duplicated name a theorem declares
  let mut importsOf : Std.HashMap Name (Array Name) := {}
  let mut imports : Array Import := #[]
  let mut importSet : NameSet := {}
  -- a module whose copy of a theorem may be an axiom: one `--closure-reused` names, whose own replay passed at its key
  -- (its graph task in this run, or `check.py`'s replay cache) and recorded what that copy rests on, read here at the
  -- parts it has; not an import from outside the set, nor an own module, nor any by `--env-out`, which writes every
  -- module's axioms
  let readPublic (m : Name) := envOut.isNone && !ownSet.contains m && reusedSet.contains m
  for (m, imported, data) in loaded do
    importsOf := importsOf.insert m (imported.map (·.module))
    for c in data.constants do
      match constants[c.name]? with
      | none => constants := constants.insert c.name c; moduleOf := moduleOf.insert c.name m
      | some a =>
        if let some why := notOne readPublic moduleOf[c.name]! m a c then
          throw <| IO.userError s!"{c.name} is declared by {moduleOf[c.name]!} and by {m} {why}"
        alsoIn := alsoIn.alter c.name fun ms => some ((ms.getD #[]).push m)
        if a matches .thmInfo _ || c matches .thmInfo _ then proved := proved.insert c.name
        if c matches .thmInfo _ && !reusedSet.contains m then
          dups := dups.push (m, c)
    for i in imported do
      -- a public part mentions nothing a private import declares, and `lean` never loaded it for the importer
      if publicOnly.contains m && !i.isExported then continue
      if !modSet.contains i.module && !importSet.contains i.module then
        imports := imports.push i
        importSet := importSet.insert i.module
  let env ← if imports.isEmpty then mkEmptyEnvironment else importModules imports {}
  let tLoaded ← IO.monoMsNow
  for h : i in [0:env.header.moduleNames.size] do
    importsOf := importsOf.insert env.header.moduleNames[i] (env.header.moduleData[i]!.imports.map (·.module))
  -- An equation lemma (`f.eq_1`) is realized in each module that first asks for it, so an import from outside the set
  -- can hold the same theorem (or the axiom a public part makes of it): the import's stays in the environment, and the
  -- set's theorem is checked under a fresh name.
  for (n, c) in constants do
    match env.find? n with
    | none => pure ()
    | some a =>
      let m := moduleOf[n]!
      let im := (env.getModuleIdxFor? n).bind fun i => env.header.moduleNames[i.toNat]?
      if let some why := notOne readPublic (im.getD .anonymous) m a c then
        throw <| IO.userError s!"{n} is declared by {m} and by {im.getD `«an import»} from outside the set {why}"
      constants := constants.erase n
      moduleOf := moduleOf.erase n
      if let some im := im then
        moduleOf := moduleOf.insert n im
      alsoIn := alsoIn.alter n fun ms => some ((ms.getD #[]).push m)
      if a matches .thmInfo _ || c matches .thmInfo _ then proved := proved.insert n
      if c matches .thmInfo _ && !reusedSet.contains m then
        dups := dups.push (m, c)
  -- after the loop above, which moves a theorem an import from outside the set also declares: a closure takes the values
  -- `moduleOf` and `alsoIn` have where it is defined
  let declaredIn (n : Name) : Array Name :=
    let here := match moduleOf[n]? with
      | some m => #[m]
      | none => ((env.getModuleIdxFor? n).bind fun i => env.header.moduleNames[i.toNat]?).toArray
    here ++ alsoIn.getD n #[]
  -- a name only axioms declare is one constant if they state a proposition, whatever bodies their public parts hide
  let lookup (n : Name) := constants[n]? <|> env.find? n
  for (n, _) in alsoIn do
    unless proved.contains n do
      let some c := lookup n | unreachable!
      unless isPropCheap lookup c.type do
        throw <| IO.userError s!"{n} is declared by {", ".intercalate ((declaredIn n).toList.map toString)}, axioms \
          each, of a type that isn't a proposition: the definitions their public parts hide may differ"
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
  -- the checked declaration a constant belongs to, if it is one; any other constant of the set went in unchecked
  let checkedDecl (n : Name) : Option Name := do
    let c ← constants[n]?
    guard !(compiled c)
    let e := declOf constants c
    guard (members.contains e)
    return e
  let behind ← IO.mkRef ({} : Std.HashMap Name NameSet)
  let moduleOfDecl (d : Name) : Name := match renamedFrom[d]? with
    | some (_, m) => m
    | none => moduleOf[d]!
  -- each constant a checked declaration mentions is declared by its module or by one that module imports. Two imported
  -- modules that each realize one theorem leave the environment naming one of them, so a module that imports only the
  -- other is looked up in what each imported module of its closure declares, with the environment's statement.
  let importedIdx : Std.HashMap Name Nat :=
    (List.range env.header.moduleNames.size).foldl (fun h i => h.insert env.header.moduleNames[i]! i) {}
  let alsoImported (reach : Std.HashSet Name) (n : Name) : Bool := Id.run do
    let some e := env.find? n | return false
    for r in reach do
      if let some i := importedIdx[r]? then
        for c in env.header.moduleData[i]!.constants do
          if c.name == n && (notOne (fun _ => true) r r e c).isNone then return true
    return false
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
        unless from_.isEmpty || from_.any reach.contains || alsoImported reach n do
          IO.eprintln s!"replay: {d} (in {m}) mentions {n}, declared by {", ".intercalate (from_.toList.map toString)}, \
            which {m} does not import"
          return 1
      let mut ds : NameSet := {}
      for n in used do
        if let some e := checkedDecl n then
          ds := ds.insert e
        else if constants.contains n then
          for e in ← reachChecked constants checkedDecl behind {} n do
            ds := ds.insert e
      deps := deps.insert d <| (ds.erase d).toArray
  let tPlanned ← IO.monoMsNow
  let order ← match declOrder (members.keys.toArray.qsort Name.lt) deps with
    | .ok order => pure order
    | .error d =>
      IO.eprintln s!"replay: {d} (in {moduleOfDecl d}) is on a cycle of declarations that mention each other"
      return 1
  let mut tasks : Array (Name × Task (Except IO.Error Nat)) := #[]
  let mut settled : Array (Name × Except IO.Error Nat) := #[]
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
      settled := settled.push (d', ← IO.wait t)
      waited := waited + 1
  for (d, t) in tasks.extract waited tasks.size do
    settled := settled.push (d, ← IO.wait t)
  let deepRecursion ← (Kernel.Exception.deepRecursion.toMessageData {}).toString
  let underModuleSystem : NameSet := loaded.foldl (fun s (m, _, data) => if data.isModule then s.insert m else s) {}
  let mut deep : Std.HashMap Name (Array Name) := {}
  for (d, r) in settled do
    match r with
    | .ok ms => took := took.push (ms, d)
    | .error e =>
      if underModuleSystem.contains (moduleOfDecl d) && (toString e).endsWith s!"\n{deepRecursion}" then
        deep := deep.alter (moduleOfDecl d) fun ds => some ((ds.getD #[]).push d)
      else
        IO.eprintln s!"replay: the kernel rejects {d} (in {moduleOfDecl d}):\n{e}"
        rejected := rejected.insert (moduleOfDecl d)
  -- each declaration the kernel stopped on, again on the constants `lean` checked it on (`Lean.Elab.processHeaderCore`
  -- imports a module's imports at `.exported`)
  let mut onPublic : Array Name := #[]
  for (m, ds) in deep do
    let some (_, mImports, data) := loaded.find? (·.1 == m) | unreachable!
    let view ← importModules mImports {} (level := .exported)
    let mut venv := (← mkEmptyEnvironment).toKernelEnv
    let mut quot : Std.HashMap Name ConstantInfo := {}
    let mut missing : Array String := #[]
    -- nothing taken from `view` outlives its regions: each constant is this replay's, under its own name
    for (n, c) in view.toKernelEnv.constants.map₁ do
      let some f := kenv.find? n | missing := missing.push (toString n); continue
      let f : ConstantInfo := match c, f with
        | .axiomInfo _, .thmInfo v => .axiomInfo { toConstantVal := v.toConstantVal, isUnsafe := false }
        | .axiomInfo _, .defnInfo v => .axiomInfo { toConstantVal := v.toConstantVal, isUnsafe := v.safety == .unsafe }
        | .axiomInfo _, .opaqueInfo v => .axiomInfo { toConstantVal := v.toConstantVal, isUnsafe := v.isUnsafe }
        | _, f => f
      if f matches .quotInfo _ then quot := quot.insert f.name f else venv := addImported venv f
    view.freeRegions
    let failed : Std.HashMap Name ConstantInfo :=
      ds.foldl (fun h d => (members.getD d #[]).foldl (fun h c => h.insert c.name c) h) {}
    for c in data.constants do
      unless failed.contains c.name do venv := addImported venv ((kenv.find? c.name).getD c)
    let r : Except IO.Error Kernel.Environment ← if !missing.isEmpty then
        pure (.error (IO.userError s!"its imports' public parts hold {missing.size} constants this replay lacks, \
          {missing[0]!} among them"))
      else IO.wait (← IO.asTask do (← venv.replay quot).replay failed)
    match r with
    | .ok _ => onPublic := onPublic ++ ds
    | .error e =>
      IO.eprintln s!"replay: the kernel stops on {", ".intercalate (ds.toList.map toString)} (in {m}), and rejects them \
        on the parts of its imports `lean` checked them on:\n{e}"
      rejected := rejected.insert m
  if !rejected.isEmpty then
    -- every other module's declarations passed with its imports present, which is what a module's cache entry says
    IO.FS.writeFile out <| (Json.mkObj [("rejected", toJson (rejected.toArray.qsort Name.lt |>.map toString))]).compress
    return 1
  let tChecked ← IO.monoMsNow
  let ownConstants := (constants.toArray.filter fun (n, _) => ownSet.contains moduleOf[n]!).map (·.1)
  let mut names : Array String := #[]
  let mut declarations : Array (String × Json) := #[]
  let envRecord ← envIn.mapM readEnvRecord
  let envRead ← IO.mkRef ({} : Std.HashMap String (Option (Std.HashMap String (Std.HashSet String))))
  -- the set the env's record holds for constant `w` of module `m`, or none when it doesn't hold `m`
  let fromEnv (m w : Name) : IO (Option (Std.HashSet String)) := do
    let some r := envRecord | return none
    let k := m.toString (escape := false)
    let got ← match (← envRead.get)[k]? with
      | some got => pure got
      | none => do
        let got ← r.module k
        envRead.modify (·.insert k got)
        pure got
    return got.map (·.getD (toString w) {})
  if records.isSome || envOut.isSome then
    let recordedModules := (get "--closure-reused").foldl NameSet.insert {}
    let read ← IO.mkRef ({} : Std.HashMap Name (Option (Std.HashMap String (Std.HashSet String))))
    let recorded (w : Name) : IO (Option (Std.HashSet String)) := do
      let some m := moduleOf[w]? | do
        -- a constant of an imported module: the env's
        let some i := env.getModuleIdxFor? w | return none
        let some m := env.header.moduleNames[i.toNat]? | return none
        fromEnv m w
      unless recordedModules.contains m do
        -- a module of the set is walked; another is the import a theorem the set declares too was moved to
        if modSet.contains m then return none
        return ← fromEnv m w
      let some dir := records | return none
      let r ← match (← read.get)[m]? with
        | some r => pure r
        | none => do
          let r ← readRecord dir m
          read.modify (·.insert m r)
          pure r
      return r.map (·.getD (toString w) {})
    if let some f := envOut then
      let sets ← axiomSets (fun n => renamed[n]? <|> kenv.find? n) recorded
        (constants.toArray.map (·.1) ++ renamed.toArray.map (·.1))
      -- the copies of a theorem two modules declare take their sets' union, so an importer finds it whichever module
      -- its environment names for the theorem
      let mut joined : Std.HashMap Name (Std.HashSet String) := {}
      for (n, _) in constants do
        joined := joined.insert n (sets.getD n {})
      for (fresh, (n, _)) in renamedFrom do
        joined := joined.alter n fun s => some ((s.getD {}).insertMany (sets.getD fresh {}))
      let mut perModule : Std.HashMap Name (Array (String × Json)) := {}
      let mut unsafeOf : Std.HashMap Name (Array String) := {}
      for (n, m) in (constants.toArray.map fun (n, _) => (n, moduleOf[n]!)) ++ renamedFrom.toArray.map (·.2) do
        let s := joined.getD n {}
        unless s.isEmpty do
          perModule := perModule.alter m fun ds => some ((ds.getD #[]).push (toString n, toJson (s.toArray.qsort (· < ·))))
          if constants[n]?.any (·.isUnsafe) then
            unsafeOf := unsafeOf.alter m fun ns => some ((ns.getD #[]).push (toString n))
      -- a theorem a module imported from an earlier replay declares too: its copy's set joins that module's record
      let mut earlier : Std.HashMap Name (Array (String × Json)) := {}
      for (fresh, (n, _)) in renamedFrom do
        if let some im := moduleOf[n]? then
          let s := sets.getD fresh {}
          unless modSet.contains im || s.isEmpty do
            earlier := earlier.alter im fun ds => some ((ds.getD #[]).push (toString n, toJson (s.toArray.qsort (· < ·))))
      let h ← IO.FS.Handle.mk f .write
      for m in mods do
        h.putStrLn (Json.mkObj [("module", m.toString (escape := false)),
          ("axioms", Json.mkObj (perModule.getD m #[]).toList),
          ("unsafe", toJson ((unsafeOf.getD m #[]).qsort (· < ·)))]).compress
      h.putStrLn (Json.mkObj [("deferred", toJson (deferred.toArray.qsort Name.lt |>.map (·.toString (escape := false)))),
        ("joined", Json.mkObj (earlier.toList.map fun (m, ds) => (m.toString (escape := false), Json.mkObj ds.toList)))]).compress
    if records.isSome then
      -- an own module's copy of a theorem a module before it declares too is recorded under the theorem's name
      let copies := renamedFrom.toArray.filter fun (_, (_, m)) => ownSet.contains m
      let sets ← axiomSets (fun n => renamed[n]? <|> kenv.find? n) recorded (ownConstants ++ copies.map (·.1))
      let mut all : Std.HashSet String := {}
      let named := ownConstants.map (fun n => (n, n)) ++ copies.map fun (fresh, (n, _)) => (fresh, n)
      for (k, n) in named do
        let s := sets.getD k {}
        unless s.isEmpty do
          all := all.insertMany s
          declarations := declarations.push (toString n, toJson (s.toArray.qsort (· < ·)))
      names := all.toArray.qsort (· < ·)
  else
    names := axiomsIn kenv ownConstants |>.map toString
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
    (slowest: {", ".intercalate slowest.toList}); {onPublic.size} of them, which the kernel stopped on, accepted on their \
    imports' public parts (`lean`'s check){if onPublic.isEmpty then "" else s!": {", ".intercalate (onPublic.toList.map toString)}"}; \
    axioms of the own modules' bodies: {", ".intercalate names.toList}"
  let tEnd ← IO.monoMsNow
  let seconds := [("load", tLoaded - t0), ("plan", tPlanned - tLoaded), ("check", tChecked - tPlanned),
    ("axioms", tEnd - tChecked)].map fun (k, ms) => (k, toJson (ms / 1000))
  IO.println s!"replay: seconds {Json.mkObj seconds |>.compress}"
  IO.FS.writeFile out <| (Json.mkObj <| [("replayed", toJson sent), ("skipped", toJson skipped), ("reused", toJson kept),
    ("second_copies", toJson renamed.size), ("closure_checked", toJson (get "--closure").size),
    ("closure_reused", toJson (get "--closure-reused").size), ("imported", toJson env.header.moduleNames.size),
    ("axioms", toJson names), ("seconds", Json.mkObj seconds),
    ("on_public_imports", toJson (onPublic.qsort Name.lt |>.map toString))] ++
    (if records.isSome then [("declarations", Json.mkObj declarations.toList)] else []) ++
    (if envIn.isSome then [("env_modules", toJson (env.header.moduleNames.map (·.toString (escape := false))))] else []) ++
    (if envOut.isSome then [("modules", toJson mods.size),
      ("deferred", toJson (deferred.toArray.qsort Name.lt |>.map (·.toString (escape := false))))] else [])).compress
  return 0
