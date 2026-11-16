import Lean

/-!
# The exact pass of the upstream watch

`lake env lean --run Upstream.lean CONFIG`, where `CONFIG` is JSON with `imports` (module names), `watch` (a list of
`{"entry", "mentions"}`, constants) and `out` (a path). It imports the modules and writes to `out`, for each entry, every
theorem of the environment whose elaborated type mentions all of the entry's constants, with the axioms outside `propext`,
`Classical.choice` and `Quot.sound` that its proof reaches. Types are matched on constants, so namespaces, notation and
implicit arguments don't matter. Lean's generated theorems (equation lemmas, `_proof_N`, `congr_simp`, a structure's
`inj` and `sizeOf_spec`) are left out.

The modules are read as data: no initializer runs and no environment extension is loaded, and the axioms are found by
walking the bodies the `.olean` files hold.
-/

open Lean

def standard : NameSet := NameSet.empty |>.insert ``propext |>.insert ``Classical.choice |>.insert ``Quot.sound

def generated (n : Name) : Bool :=
  n.isInternalDetail || match n with
    | .str _ s => ["congr_simp", "eq_def", "eq_unfold", "inj", "injEq", "sizeOf_spec"].contains s
    | _ => false

def uses (env : Environment) (c : Name) : Array Name :=
  match env.find? c with
  | some (.axiomInfo v) => v.type.getUsedConstants
  | some (.defnInfo v) => v.type.getUsedConstants ++ v.value.getUsedConstants
  | some (.thmInfo v) => v.type.getUsedConstants ++ v.value.getUsedConstants
  | some (.opaqueInfo v) => v.type.getUsedConstants ++ v.value.getUsedConstants
  | some (.inductInfo v) => v.type.getUsedConstants ++ v.ctors.toArray
  | some (.ctorInfo v) => v.type.getUsedConstants
  | some (.recInfo v) => v.type.getUsedConstants
  | _ => #[]

/-- The nonstandard axioms each constant reached from `roots` uses: one walk shared by all the roots, iterative because
Mathlib's dependency chains are too deep for recursion. -/
def nonstandard (env : Environment) (roots : Array Name) : Std.HashMap Name NameSet := Id.run do
  let mut memo : Std.HashMap Name NameSet := {}
  let mut open_ : NameSet := {}
  let mut stack : Array (Name × Option (Array Name)) := roots.map (·, none)
  while h : stack.size > 0 do
    let (c, deps) := stack[stack.size - 1]
    stack := stack.pop
    if memo.contains c then continue
    match deps with
    | some ds =>
      let mut s : NameSet := if (env.find? c matches some (.axiomInfo _)) && !standard.contains c then
        NameSet.empty.insert c else {}
      for d in ds do
        for a in memo.getD d {} do
          s := s.insert a
      memo := memo.insert c s
      open_ := open_.erase c
    | none =>
      if open_.contains c then continue
      let ds := uses env c
      open_ := open_.insert c
      stack := stack.push (c, some ds)
      for d in ds do
        unless memo.contains d || open_.contains d do
          stack := stack.push (d, none)
  return memo

unsafe def main (args : List String) : IO UInt32 := do
  let some path := args.head? | throw <| IO.userError "usage: Upstream.lean CONFIG"
  let cfg ← IO.ofExcept <| Json.parse (← IO.FS.readFile path)
  let imports ← IO.ofExcept <| cfg.getObjValAs? (Array String) "imports"
  let out ← IO.ofExcept <| cfg.getObjValAs? String "out"
  let watch ← IO.ofExcept <| cfg.getObjValAs? (Array Json) "watch"
  let mut entries : Array (String × Array Name) := #[]
  for w in watch do
    let entry ← IO.ofExcept <| w.getObjValAs? String "entry"
    let mentions ← IO.ofExcept <| w.getObjValAs? (Array String) "mentions"
    entries := entries.push (entry, mentions.map String.toName)
  initSearchPath (← findSysroot)
  let env ← importModules (imports.map fun m => { module := m.toName }) {}
  let hits := env.constants.fold (init := entries.map fun _ => (#[] : Array Name)) fun acc n ci => Id.run do
    let .thmInfo _ := ci | return acc
    if generated n then return acc
    let used := ci.type.getUsedConstants
    let mut acc := acc
    for i in [:entries.size] do
      if entries[i]!.2.all used.contains then
        acc := acc.set! i (acc[i]!.push n)
    return acc
  let axioms := nonstandard env (hits.foldl (· ++ ·) #[])
  let module (n : Name) : String := match env.getModuleIdxFor? n with
    | some i => env.header.modules[i.toNat]!.module.toString
    | none => "?"
  let mut found : List (String × Json) := []
  let mut missing : List (String × Json) := []
  for i in [:entries.size] do
    let (entry, cs) := entries[i]!
    missing := missing ++ [(entry, toJson ((cs.filter (!env.contains ·)).map toString))]
    let hs := hits[i]!.qsort (·.toString < ·.toString) |>.map fun n => Json.mkObj [("name", toJson n.toString),
      ("module", toJson (module n)), ("axioms", toJson ((axioms.getD n {}).toArray.map toString))]
    found := found ++ [(entry, toJson hs)]
  IO.FS.writeFile out (Json.mkObj [("watch", Json.mkObj found), ("missing", Json.mkObj missing)]).compress
  return 0
