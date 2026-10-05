import Lean

/-!
# Every declaration of a package, with its source range and what it uses

`lake env lean --run Decls.lean CONFIG` imports the roots `CONFIG` names (a JSON file: `roots`, `under`, `out`) with no
environment extension loaded, as `Facts.lean` does, and writes one JSON document to `out`: for each module under `under`,
its imports and each constant it declares, with its kind, its source range (`declRangeExt`: first line and column, last
line and column, columns in code points) when it has one, and the constants under `under` its type and value use. It
also lists every imported module's imports (`imports`), so a reader can follow a module's closure into its dependencies,
and the namespaces each module registers (`namespaces`), which an `open` needs.
`model_split.py` reads it; it reports facts and decides nothing.
-/

open Lean

namespace LeanDecls

def under (ps : Array Name) (m : Name) : Bool := ps.any (·.isPrefixOf m)

def kind : ConstantInfo → String
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "def"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quot"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"

def uses : ConstantInfo → Array Name
  | .axiomInfo v => v.type.getUsedConstants
  | .defnInfo v | .thmInfo v | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray
  | .ctorInfo v => v.type.getUsedConstants
  | .recInfo v => v.type.getUsedConstants
  | .quotInfo _ => #[]

unsafe def entriesOf (α : Type) (data : ModuleData) (ext : Name) : Array (Name × α) :=
  match data.entries.find? (·.1 == ext) with
  | some (_, es) => es.map fun e => unsafeCast e
  | none => #[]

/-- The namespaces a module registers (`namespace` commands and its declarations' prefixes), from the entries of core's
private `namespacesExt`. -/
unsafe def namespacesOf (data : ModuleData) : Array Name :=
  match data.entries.find? (fun e => match e.1 with | .str _ s => s == "namespacesExt" | _ => false) with
  | some (_, es) => es.map fun e => unsafeCast e
  | none => #[]

def names (j : Json) (k : String) : Array Name :=
  ((j.getObjValD k).getArr?.toOption.getD #[]).filterMap fun v => v.getStr?.toOption.map String.toName

end LeanDecls

open LeanDecls in
unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let some cfgPath := args.head? | throw <| IO.userError "usage: Decls.lean CONFIG"
  let cfg ← IO.ofExcept (Json.parse (← IO.FS.readFile cfgPath))
  let roots := names cfg "roots"
  let pre := names cfg "under"
  let out := (cfg.getObjValD "out").getStr?.toOption.getD "decls.json"
  let env ← importModules (roots.map fun r => { module := r }) {} (loadExts := false)
  let mnames := env.header.moduleNames
  let inPkg (c : Name) : Bool :=
    match env.getModuleIdxFor? c with
    | some i => under pre mnames[i.toNat]!
    | none => false
  let mut mods := #[]
  let mut imports := #[]
  let mut nss := #[]
  for h : i in [0:mnames.size] do
    let m := mnames[i]
    let data := env.header.moduleData[i]!
    imports := imports.push (m.toString, toJson (data.imports.map (·.module.toString)))
    let ns := namespacesOf data
    if !ns.isEmpty then
      nss := nss.push (m.toString, toJson (ns.map toString))
    if !under pre m then continue
    let mut ranges : Std.HashMap Name DeclarationRanges := {}
    for (n, r) in entriesOf DeclarationRanges data declRangeExt.name do
      ranges := ranges.insert n r
    let decls := data.constants.map fun ci =>
      let r : Option (Array Nat) := ranges[ci.name]?.map fun d =>
        #[d.range.pos.line, d.range.pos.column, d.range.endPos.line, d.range.endPos.column]
      let us := (uses ci).filter inPkg |>.qsort Name.lt
      Json.mkObj [("n", toJson ci.name.toString), ("k", toJson (kind ci)), ("r", toJson r),
        ("u", toJson (us.toList.eraseDups.toArray.map toString))]
    mods := mods.push (m.toString, Json.mkObj [("imports", toJson (data.imports.map (·.module.toString))),
      ("decls", toJson decls)])
  let doc := Json.mkObj [("lean", toJson Lean.versionString), ("modules", Json.mkObj mods.toList),
    ("imports", Json.mkObj imports.toList), ("namespaces", Json.mkObj nss.toList)]
  IO.FS.writeFile out doc.compress
  return 0
