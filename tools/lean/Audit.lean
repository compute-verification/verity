/-!
# The Lean side of `tools/lean/audit.py`

`audit.py` writes a file that imports `Lean` and the package's roots, defines `LeanAudit.Config`, and appends this file; it
runs it with `lake env lean` in the package. The `#eval` at the end writes one JSON document to `LeanAudit.Config.out`. It
reports facts and makes no judgement; `audit.py` holds the policy.

- `modules`: the imported modules under `Config.prefixes` (the package's audited modules); `imports`: what each imports.
- `declarations`: how many constants of each kind those modules declare.
- `axioms`: every axiom any of them uses; `offenders`: each one that uses an axiom outside `Config.allowed`.
- `declared_axioms`: the `axiom` declarations among them.
- `escapes`: the `opaque` ones (a `partial def` is one), and the `unsafe`, `@[implemented_by]` and `@[extern]` ones: compiled
  code with no definition a proof can unfold. (The `_unsafe_rec` copies the compiler makes of every recursive definition
  are not listed: the kernel checks the definition itself.)
- `pins`: for each name in `Config.pins`, its kind, signature, axioms, `Prop` hypotheses, the hash of its type, and the
  definitions its statement reads, followed through every definition, inductive type and constructor declared under
  `Config.meaning` (proofs are skipped: they do not change what a statement means).

Axiom sets come from `Lean.collectAxioms`, which reads the sets each module's `.olean` recorded when it was built.
-/

open Lean Meta Elab Command

namespace LeanAudit

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

def moduleOf (env : Environment) (n : Name) : Option Name :=
  (env.getModuleIdxFor? n).map fun i => env.header.moduleNames[i.toNat]!

/-- What a statement means through this constant: its type, a definition's value, an inductive type's constructors.
Theorems contribute nothing (their proofs are irrelevant to what a statement says). -/
def meaningDeps : ConstantInfo → Array Name
  | .defnInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray
  | .ctorInfo v => v.type.getUsedConstants
  | .recInfo v => v.type.getUsedConstants
  | .axiomInfo v => v.type.getUsedConstants
  | .thmInfo _ | .quotInfo _ => #[]

def valueHash : ConstantInfo → UInt64
  | .defnInfo v => v.value.hash
  | .opaqueInfo v => v.value.hash
  | _ => 0

def hex (x : UInt64) : String :=
  let s := (Nat.toDigits 16 x.toNat).foldl String.push ""
  "".pushn '0' (16 - s.length) ++ s

/-- The definitions under `meaning` that the constants `roots` read, transitively. -/
def reads (env : Environment) (meaning : Array Name) (roots : Array Name) : Array (Name × Name × ConstantInfo) := Id.run do
  let mut seen : NameSet := {}
  let mut stack := roots
  let mut out := #[]
  repeat
    match stack.back? with
    | none => break
    | some c =>
      stack := stack.pop
      if seen.contains c then continue
      seen := seen.insert c
      let some ci := env.find? c | continue
      let some m := moduleOf env c | continue
      if !under meaning m then continue
      if ci matches .thmInfo _ then continue
      out := out.push (c, m, ci)
      stack := stack ++ meaningDeps ci
  return out

/-- One definition a statement reads: its name, module, kind, hash, and source lines when Lean recorded them. -/
def readJson : Name × Name × ConstantInfo → CommandElabM Json
  | (c, m, ci) => do
    let lines := (← findDeclarationRanges? c).map fun r => #[r.range.pos.line, r.range.endPos.line]
    return Json.mkObj [("name", toJson c.toString), ("module", toJson m.toString), ("kind", toJson (kind ci)),
      ("hash", toJson s!"{hex ci.type.hash}{hex (valueHash ci)}"), ("lines", toJson lines)]

def pinInfo (n : Name) : CommandElabM Json := do
  let env ← getEnv
  let some ci := env.find? n | return Json.mkObj [("kind", "missing")]
  let signature ← liftTermElabM do return toString (← PrettyPrinter.ppSignature n).fmt
  let axioms ← collectAxioms n
  let hypotheses ← liftTermElabM <| forallTelescope ci.type fun xs _ => do
    let mut out := #[]
    for x in xs do
      let t ← inferType x
      unless ← isProp t do continue
      let named : Option Name := match t with | .const c _ => some c | _ => none
      out := out.push <| Json.mkObj [("binder", toJson (← x.fvarId!.getDecl).userName.toString),
        ("type", toJson (toString (← ppExpr t))), ("closed", toJson (!t.hasFVar && !t.hasMVar)),
        ("named", toJson (named.map toString)), ("module", toJson ((named.bind (moduleOf env)).map toString))]
    return out
  return Json.mkObj [("kind", toJson (kind ci)), ("module", toJson ((moduleOf env n).map toString)),
    ("signature", toJson signature), ("axioms", toJson (axioms.map toString)), ("hypotheses", toJson hypotheses),
    ("type_hash", toJson (hex ci.type.hash)),
    ("reads", toJson (← (reads env Config.meaning ci.type.getUsedConstants).mapM readJson))]

def run : CommandElabM Unit := do
  let env ← getEnv
  let names := env.header.moduleNames
  let mut modules := #[]
  let mut imports := #[]
  let mut counts : Std.HashMap String Nat := {}
  let mut used : NameSet := {}
  let mut offenders := #[]
  let mut declared := #[]
  let mut escapes := #[]
  for h : i in [0:names.size] do
    let m := names[i]
    if !under Config.prefixes m then continue
    modules := modules.push (toJson m.toString)
    imports := imports.push (m.toString, toJson (env.header.moduleData[i]!.imports.map (·.module.toString)))
    for ci in env.header.moduleData[i]!.constants do
      let n := ci.name
      counts := counts.insert (kind ci) (counts.getD (kind ci) 0 + 1)
      if ci matches .axiomInfo _ then
        declared := declared.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString)]
      let axs ← collectAxioms n
      used := axs.foldl (·.insert ·) used
      let bad := axs.filter (!Config.allowed.contains ·)
      if !bad.isEmpty then
        offenders := offenders.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString),
          ("axioms", toJson (bad.map toString))]
      let why := (if ci matches .opaqueInfo _ then #["opaque"] else #[]) ++ (if ci.isUnsafe then #["unsafe"] else #[]) ++
        ((Compiler.getImplementedBy? env n).map (s!"implemented_by {·}") |>.toArray) ++ (if isExtern env n then #["extern"] else #[])
      if !why.isEmpty then
        escapes := escapes.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString), ("why", toJson why)]
  let mut pins := #[]
  for p in Config.pins do
    pins := pins.push (p.toString, ← pinInfo p)
  let doc := Json.mkObj [("lean", toJson Lean.versionString), ("modules", toJson modules), ("imports", Json.mkObj imports.toList),
    ("declarations", Json.mkObj (counts.toList.map fun (k, v) => (k, toJson v))),
    ("axioms", toJson (used.toArray.qsort Name.lt |>.map toString)), ("offenders", toJson offenders),
    ("declared_axioms", toJson declared), ("escapes", toJson escapes), ("pins", Json.mkObj pins.toList)]
  IO.FS.writeFile Config.out doc.compress

end LeanAudit

#eval LeanAudit.run
