import Lean

/-!
# The facts of what each declaration reads, from the `.olean` files as data

Compiled and run under `lake env` with a JSON file `CONFIG` (`unread.py` and `check.py`'s audit do both), it reads the
modules `CONFIG` names (`modules`), and each module of `follow` they import, transitively, from their `.olean` files as
`Replay.lean` reads them (each part, the private one where there is one, with no import and no initializer, so Mathlib's
and ArkLib's modules are never loaded) and writes to the path it names (`out`)
one JSON line per module and then one per constant those modules declare, for `unread.py`, which holds the policy: it
makes no judgement.

- a module's line: `module` (its name), `found` (an `.olean` on the search path), `imports`, and `foreign`: the constants
  it gives an instance or `simp` attribute that it does not declare (`attribute [instance] c` in it changes its importers);
- a constant's line, in the order the modules and their constants come: `n` its name, `m` its module's index, `k` its kind
  (`theorem`, `def`, `abbrev`, `opaque`, `axiom`, `inductive`, `ctor`, `rec`, `quot`), `r` the first and last line of its
  declaration range when it has one (what Lean generates has none), `p` the type it belongs to (a constructor's, a
  recursor's, a projection's), `c` an instance's class, `f` its flags, and `u` the indices of the constants of these
  modules that its type and value mention (an inductive type's constructors, a constructor's type, an
  `@[implemented_by]` target, the `@[csimp]` theorem that replaces it in compiled code), itself left out.

The flags: `unsafe`, `partial`, `instance`, `class`, `structure` (a type with projections), `defeq` (`@[defeq]` or
`@[backward_defeq]`, which `dsimp` uses with no trace in the proof term), `meta` (its type mentions syntax, a macro, an
elaborator, a tactic, a delaborator, a simproc, an attribute, an option or `IO`: a declaration used by name, not through
terms). A name two modules declare (an equation lemma each realized) is written once, from the first.
-/

open Lean

namespace LeanUnread

unsafe def entriesOf (α : Type) (data : ModuleData) (ext : Name) : Array α :=
  match data.entries.find? (·.1 == ext) with
  | some (_, es) => es.map fun e => unsafeCast e
  | none => #[]

def scopedValue : ScopedEnvExtension.Entry α → α
  | .global a | .scoped _ a => a

/-- The types whose mention in a constant's type makes it a declaration Lean runs or looks up by name. -/
def metaTypes : NameSet := [``Lean.ParserDescr, ``Lean.TrailingParserDescr, ``Lean.Parser.Parser,
  ``Lean.Parser.TrailingParser, ``Lean.Parser.Category, ``Lean.Macro, ``Lean.MacroM, ``Lean.Syntax, ``Lean.TSyntax,
  ``Lean.Elab.Term.TermElab, ``Lean.Elab.Term.TermElabM, ``Lean.Elab.Command.CommandElab,
  ``Lean.Elab.Command.CommandElabM, ``Lean.Elab.Tactic.Tactic, ``Lean.Elab.Tactic.TacticM, ``Lean.Meta.MetaM,
  ``Lean.Core.CoreM, ``Lean.PrettyPrinter.Unexpander, ``Lean.PrettyPrinter.UnexpandM,
  ``Lean.PrettyPrinter.Delaborator.Delab, ``Lean.PrettyPrinter.Delaborator.DelabM, ``Lean.Meta.Simp.Simproc,
  ``Lean.Meta.Simp.DSimproc, ``Lean.AttributeImpl, ``Lean.EnvExtension, ``Lean.PersistentEnvExtension,
  ``Lean.Meta.SimpExtension, ``Lean.Option, ``IO, ``EIO, ``BaseIO].foldl NameSet.insert {}

def kindOf : ConstantInfo → String
  | .thmInfo _ => "theorem"
  | .defnInfo v => if v.hints matches .abbrev then "abbrev" else "def"
  | .opaqueInfo _ => "opaque"
  | .axiomInfo _ => "axiom"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "ctor"
  | .recInfo _ => "rec"
  | .quotInfo _ => "quot"

def mentions : ConstantInfo → Array Name
  | .thmInfo v | .defnInfo v | .opaqueInfo v =>
    v.type.getUsedConstants ++ v.value.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray ++ v.all.toArray
  | .ctorInfo v => v.type.getUsedConstants.push v.induct
  | .recInfo v => v.type.getUsedConstants ++ v.all.toArray
  | .axiomInfo v => v.type.getUsedConstants
  | .quotInfo v => v.type.getUsedConstants

/-- A module's data: its imports and its most complete part (the private one, under the module system), or none when
no `.olean` of it is on the search path. -/
def load (m : Name) : IO (Option (Array Import × ModuleData)) := do
  let olean ← try findOLean m catch _ => return none
  unless ← olean.pathExists do return none
  let mut files := #[olean]
  for level in [OLeanLevel.server, OLeanLevel.private] do
    if ← (level.adjustFileName olean).pathExists then
      files := files.push (level.adjustFileName olean)
  let parts ← readModuleDataParts files
  let (some (data, _), some (header, _)) := (parts.back?, parts[0]?) | throw <| IO.userError s!"no module data for {m}"
  return some (header.imports, data)

structure Attrs where
  instances : NameSet := {}
  classes : NameSet := {}
  defeq : NameSet := {}
  structOf : Std.HashMap Name Name := {}
  implementedBy : Std.HashMap Name Name := {}
  csimp : Std.HashMap Name (Array Name) := {}
  ranges : Std.HashMap Name (Nat × Nat) := {}

unsafe def attrsOf (data : ModuleData) (a : Attrs) : Attrs := Id.run do
  let mut a := a
  for e in entriesOf (ScopedEnvExtension.Entry Meta.InstanceEntry) data Meta.instanceExtension.ext.name do
    if let some n := (scopedValue e).globalName? then a := { a with instances := a.instances.insert n }
  for e in entriesOf ClassEntry data classExtension.name do
    a := { a with classes := a.classes.insert e.name }
  for n in entriesOf Name data defeqAttr.ext.name ++ entriesOf Name data backwardDefeqAttr.ext.name do
    a := { a with defeq := a.defeq.insert n }
  for (n, i) in entriesOf (Name × ProjectionFunctionInfo) data projectionFnInfoExt.name do
    a := { a with structOf := a.structOf.insert n i.ctorName.getPrefix }
  for (n, t) in entriesOf (Name × Name) data Compiler.implementedByAttr.ext.name do
    a := { a with implementedBy := a.implementedBy.insert n t }
  for e in entriesOf (ScopedEnvExtension.Entry Compiler.CSimp.Entry) data Compiler.CSimp.ext.ext.name do
    let c := scopedValue e
    a := { a with csimp := a.csimp.alter c.fromDeclName fun ts => some ((ts.getD #[]).push c.thmName) }
  for (n, r) in entriesOf (Name × DeclarationRanges) data declRangeExt.name do
    a := { a with ranges := a.ranges.insert n (r.range.pos.line, r.range.endPos.line) }
  return a

/-- The constants `data` gives an instance or `simp` attribute without declaring them. -/
unsafe def foreignOf (data : ModuleData) : Array Name := Id.run do
  let own := data.constNames.foldl NameSet.insert {}
  let mut out : NameSet := {}
  for e in entriesOf (ScopedEnvExtension.Entry Meta.InstanceEntry) data Meta.instanceExtension.ext.name do
    if let some n := (scopedValue e).globalName? then
      unless own.contains n do out := out.insert n
  for e in entriesOf (ScopedEnvExtension.Entry Meta.SimpEntry) data Meta.simpExtension.ext.name do
    if let .thm t := scopedValue e then
      if let .decl n .. := t.origin then
        unless own.contains n do out := out.insert n
  return out.toArray.qsort Name.lt

def str (n : Name) : Json := toJson n.toString

end LeanUnread

open LeanUnread in
unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let some cfgPath := args.head? | throw <| IO.userError "usage: Unread.lean CONFIG"
  let cfg ← IO.ofExcept (Json.parse (← IO.FS.readFile cfgPath))
  let mods ← IO.ofExcept (cfg.getObjValAs? (Array String) "modules")
  let follow : NameSet := ((cfg.getObjValAs? (Array String) "follow").toOption.getD #[]).foldl
    (fun s m => s.insert m.toName) {}
  let out ← IO.ofExcept (cfg.getObjValAs? String "out")
  let t0 ← IO.monoMsNow
  let mut loaded : Array (Name × Option (Array Import × ModuleData)) := #[]
  let mut queue : Array Name := mods.map (·.toName)
  let mut queued : NameSet := queue.foldl NameSet.insert {}
  let mut k := 0
  while k < queue.size do
    let m := queue[k]!
    k := k + 1
    let l ← load m
    loaded := loaded.push (m, l)
    if let some (imports, _) := l then
      for i in imports do
        if follow.contains i.module && !queued.contains i.module then
          queued := queued.insert i.module
          queue := queue.push i.module
  let mut attrs : Attrs := {}
  let mut index : Std.HashMap Name Nat := {}
  let mut order : Array (Nat × ConstantInfo) := #[]
  for h : i in [0:loaded.size] do
    let some (_, data) := loaded[i].2 | continue
    attrs := attrsOf data attrs
    for c in data.constants do
      unless index.contains c.name do
        index := index.insert c.name order.size
        order := order.push (i, c)
  let h ← IO.FS.Handle.mk out .write
  for (m, l) in loaded do
    let fields := match l with
      | some (imports, data) => [("found", toJson true), ("imports", toJson (imports.map (str ·.module))),
          ("foreign", toJson ((foreignOf data).map str))]
      | none => [("found", toJson false)]
    h.putStrLn (Json.mkObj (("module", str m) :: fields)).compress
  let structures : NameSet := attrs.structOf.fold (fun s _ t => s.insert t) {}
  let line (i : Nat) (c : ConstantInfo) : String := Id.run do
    let mut seen : Std.HashSet Nat := {}
    for n in mentions c ++ (attrs.implementedBy[c.name]?).toArray ++ attrs.csimp.getD c.name #[] do
      if let some j := index[n]? then seen := seen.insert j
    let used := (seen.erase index[c.name]!).toArray.qsort (· < ·)
    let typeConsts := c.type.getUsedConstants
    let flags := #[("unsafe", c.isUnsafe), ("partial", c.isPartial), ("instance", attrs.instances.contains c.name),
      ("class", attrs.classes.contains c.name), ("structure", structures.contains c.name),
      ("defeq", attrs.defeq.contains c.name),
      ("meta", typeConsts.any metaTypes.contains)].filterMap fun (k, b) => if b then some (toJson k) else none
    let parent : Option Name := match c with
      | .ctorInfo v => some v.induct
      | .recInfo _ => some c.name.getPrefix
      | _ => attrs.structOf[c.name]?
    let cls : Option Name :=
      if attrs.instances.contains c.name then c.type.getForallBody.getAppFn.constName? else none
    let fields := [("n", str c.name), ("m", toJson i), ("k", toJson (kindOf c)), ("f", toJson flags),
      ("u", toJson used)] ++
      (match attrs.ranges[c.name]? with | some (a, b) => [("r", toJson #[a, b])] | none => []) ++
      (match parent with | some p => [("p", str p)] | none => []) ++
      (match cls with | some k => [("c", str k)] | none => [])
    return (Json.mkObj fields).compress
  let chunk := 4096
  let mut tasks := #[]
  for start in List.range ((order.size + chunk - 1) / chunk) do
    let part := order.extract (start * chunk) ((start + 1) * chunk)
    tasks := tasks.push <| Task.spawn fun _ => part.foldl (fun s (i, c) => s ++ line i c ++ "\n") ""
  for t in tasks do
    h.putStr t.get
  h.flush
  IO.eprintln s!"unread: {(loaded.filter (·.2.isSome)).size} of {loaded.size} modules read, {order.size} constants, \
    in {((← IO.monoMsNow) - t0) / 1000} s"
  return 0
