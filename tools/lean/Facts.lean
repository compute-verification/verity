import Lean

/-!
# The audit's facts, from the package's `.olean` files as data

`lake env lean --run Facts.lean CONFIG` imports the roots named in `CONFIG` (a JSON file `audit.py` writes) with
initializers disabled and no environment extension loaded, so none of the package's code runs: no `initialize`, no
attribute handler, no delaborator. It writes one JSON document to the path `CONFIG` names. It reports facts and makes no
judgement; `audit.py` holds the policy.

- `modules`: the imported modules under the roots; `imports`: what each imports.
- `declarations`: how many constants of each kind those modules declare.
- `axioms`: every axiom any of them uses, found by walking the bodies; `offenders`: each one that uses an axiom outside
  `allowed`.
- `declared_axioms`: the `axiom` declarations among them.
- `escapes`: the `opaque` ones (a `partial def` is one), and the `unsafe`, `@[implemented_by]` and `@[extern]` ones:
  compiled code with no definition a proof can unfold. The attributes are read from the modules' data.
- `pins`: for each name in `pins`, its kind, signature, axioms, `Prop` hypotheses, the hash of its type, and the
  definitions its statement reads, followed through every definition, inductive type and constructor declared under
  `meaning`, with their source lines. A pin whose type is a `Prop` definition under `meaning` (a restated guarantee) also
  gets that definition (`head`) and its value (`statement`), printed as the signature is. Signatures print with no
  notation, since no notation is loaded: nothing a package declares can change how its statements read.

The hashes in the document are Lean's `Expr.hash`, which has 32 bits; `audit.py` keeps them only to check records written
before SHA-256. When `CONFIG` names a `canon` path, the pins' statements and the definitions they read are also written
there, one JSON line each, as `canon` texts, and `audit.py` hashes those with SHA-256.

With `moved` (pairs of a new name, a declaration or a namespace, and the old name it moved from), each of those texts is
written once more with the names as they were before the move (the longest moved prefix renamed back), and a pinned
statement that is a `Prop` definition under `meaning` also with that definition unfolded and the proofs elaboration took
out of its value put back inline (`inlineProofs`), as the theorem it restates kept them, so that `audit.py` can compare
them with records made before the move. `moved_conflicts` lists each constant of those texts (or pin) whose old name
would be another one's too: there the comparison would not tell the two apart. A constant that keeps its name stands for
itself, and an old name that only a declaration outside the texts still has (an old theorem kept as a lemma) is no
conflict.
-/

open Lean Meta

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

def hex (x : UInt64) : String :=
  let s := (Nat.toDigits 16 x.toNat).foldl String.push ""
  "".pushn '0' (16 - s.length) ++ s

def valueHash : ConstantInfo → UInt64
  | .defnInfo v => v.value.hash
  | .opaqueInfo v => v.value.hash
  | _ => 0

partial def levelKey : Level → String
  | .zero => "0"
  | .succ l => s!"s({levelKey l})"
  | .max a b => s!"m({levelKey a},{levelKey b})"
  | .imax a b => s!"i({levelKey a},{levelKey b})"
  | .param n => s!"p{n}"
  | .mvar m => s!"?{m.name}"

def kids : Expr → Array Expr
  | .app f a => #[f, a]
  | .lam _ t b _ => #[t, b]
  | .forallE _ t b _ => #[t, b]
  | .letE _ t v b _ => #[t, v, b]
  | .mdata _ e => #[e]
  | .proj _ _ e => #[e]
  | _ => #[]

/-- `n` under the name it had before a move: its longest prefix that `back` maps (a declaration or a namespace that
moved) replaced by the name it moved from. -/
def renameName (back : Std.HashMap Name Name) : Name → Name
  | .anonymous => .anonymous
  | n@(.str p s) => match back[n]? with
    | some o => o
    | none => let p' := renameName back p; if p' == p then n else .str p' s
  | n@(.num p i) => match back[n]? with
    | some o => o
    | none => let p' := renameName back p; if p' == p then n else .num p' i

/-- One line of `canonText`, for a term whose children are the lines `ix`; `none` for metadata, which is its child.
Constant and structure names are written as `rn` maps them. -/
def nodeKey (rn : Name → Name) (e : Expr) (ix : Array Nat) : Option String :=
  match e with
  | .bvar i => some s!"V{i}"
  | .sort l => some s!"S{levelKey l}"
  | .const n ls => some s!"C{rn n} {",".intercalate (ls.map levelKey)}"
  | .app .. => some s!"A{ix[0]!} {ix[1]!}"
  | .lam .. => some s!"L{ix[0]!} {ix[1]!}"
  | .forallE .. => some s!"P{ix[0]!} {ix[1]!}"
  | .letE .. => some s!"E{ix[0]!} {ix[1]!} {ix[2]!}"
  | .lit (.natVal n) => some s!"N{n}"
  | .lit (.strVal s) => some s!"T{(toJson s).compress}"
  | .mdata .. => none
  | .proj s i _ => some s!"J{rn s} {i} {ix[0]!}"
  | .fvar id => some s!"F{id.name}"
  | .mvar id => some s!"M{id.name}"

/-- A canonical text of `e`, which `audit.py` hashes with SHA-256: one line per distinct node, its children given as the
numbers of earlier lines, then the root. It keeps what `Expr.hash` keeps (constant names, universe levels, bound variable
indices, projections, literals) and drops what it drops (binder names and binder info, `let` names, metadata), so it tells
two terms apart exactly when `Expr.hash` would, collisions aside. The walk is iterative: Mathlib's terms are too deep for
recursion. With `rn`, the text of `e` with its constants renamed by `rn`. -/
def canonText (e : Expr) (rn : Name → Name := id) : String := Id.run do
  let mut lines : Array String := #[]
  let mut number : Std.HashMap String Nat := {}
  let mut seen : Std.HashMap Expr Nat := {}
  let mut stack : Array (Expr × Bool) := #[(e, false)]
  while h : stack.size > 0 do
    let (x, ready) := stack[stack.size - 1]
    stack := stack.pop
    if seen.contains x then continue
    let ks := kids x
    if !ready then
      stack := stack.push (x, true)
      for k in ks.reverse do
        unless seen.contains k do
          stack := stack.push (k, false)
    else
      let ix := ks.map (seen.getD · 0)
      match nodeKey rn x ix with
      | none => seen := seen.insert x ix[0]!
      | some key =>
        match number[key]? with
        | some i => seen := seen.insert x i
        | none =>
          number := number.insert key lines.size
          seen := seen.insert x lines.size
          lines := lines.push key
  return "\n".intercalate lines.toList ++ s!"\nR{seen.getD e 0}"

/-- What a read definition contributes to what a statement means: its type, and a definition's value. -/
def readCanon (ci : ConstantInfo) (rn : Name → Name := id) : String :=
  match ci with
  | .defnInfo v => canonText v.type rn ++ "\n=\n" ++ canonText v.value rn
  | .opaqueInfo v => canonText v.type rn ++ "\n=\n" ++ canonText v.value rn
  | ci => canonText ci.type rn

/-- When a statement is a `Prop` definition declared under `meaning` (a theorem `t : G` that proves a named statement),
that definition and its value. -/
def statementOf (env : Environment) (meaning : Array Name) (t : Expr) : Option (Name × Expr) := do
  let .const g us := t.consumeMData | none
  let .defnInfo v ← env.find? g | none
  let i ← env.getModuleIdxFor? g
  guard (v.type.isProp && under meaning env.header.moduleNames[i.toNat]!)
  return (g, v.value.instantiateLevelParams v.levelParams us)

/-- Whether `n` is a theorem that elaboration made of a proof nested in a definition's value (`abstractNestedProofs`, or a
tactic's `mkAuxTheorem`): named `<declaration>._proof_<i>` (`_<j>`… in a nested branch), with no declaration range, which
every declaration written in a source file has. The cache that lets a later definition reuse one is not in the `.olean`. -/
def auxProof (env : Environment) (ranges : Std.HashMap Name DeclarationRanges) (n : Name) : Bool :=
  match n, env.find? n with
  | .str _ s, some (.thmInfo _) =>
    s.startsWith "_proof_" && s.length > 7 && (s.toList.drop 7).all (fun c => c.isDigit || c == '_') && !ranges.contains n
  | _, _ => false

/-- `e` with each application of an auxiliary proof (`auxProof`) replaced by that theorem's value applied to its arguments,
beta-reduced: the term as written, its nested proofs inline, as a theorem's type keeps them. An auxiliary proof inside an
inlined value stays (a tactic's own, which the theorem's type would name too). Unfolding a theorem is definitional. -/
partial def inlineProofs (env : Environment) (aux : Name → Bool) (e : Expr) : Expr :=
  e.replace fun x => match x.getAppFn with
    | .const n us => match aux n, env.find? n with
      | true, some (.thmInfo v) =>
        some ((v.value.instantiateLevelParams v.levelParams us).beta (x.getAppArgs.map (inlineProofs env aux)))
      | _, _ => none
    | _ => none

/-- What a constant's axioms depend on (as `Lean.collectAxioms` walks it). -/
def uses : ConstantInfo → Array Name
  | .axiomInfo v => v.type.getUsedConstants
  | .defnInfo v | .thmInfo v | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray
  | .ctorInfo v => v.type.getUsedConstants
  | .recInfo v => v.type.getUsedConstants
  | .quotInfo _ => #[]

/-- What a statement means through a constant: its type, a definition's value, an inductive type's constructors.
Theorems contribute nothing (a proof is irrelevant to what a statement says). -/
def meaningDeps : ConstantInfo → Array Name
  | .thmInfo _ | .quotInfo _ => #[]
  | ci => uses ci

/-- The axioms each of `names` uses, by walking bodies, memoized. A constant already on the path contributes nothing, as
in `Lean.collectAxioms` (an inductive type and its constructors). -/
def axiomsOf (env : Environment) (names : Array Name) : Std.HashMap Name NameSet := Id.run do
  let own (c : Name) : NameSet := if env.find? c matches some (.axiomInfo _) then NameSet.empty.insert c else {}
  let deps (c : Name) : Array Name := (env.find? c).map uses |>.getD #[]
  let mut memo : Std.HashMap Name NameSet := {}
  for root in names do
    if memo.contains root then continue
    let mut stack : Array (Name × Array Name × Nat × NameSet) := #[(root, deps root, 0, own root)]
    let mut path : NameSet := NameSet.empty.insert root
    repeat
      match stack.back? with
      | none => break
      | some (c, ds, i, acc) =>
        if h : i < ds.size then
          let d := ds[i]
          stack := stack.set! (stack.size - 1) (c, ds, i + 1, acc)
          if let some s := memo[d]? then
            stack := stack.set! (stack.size - 1) (c, ds, i + 1, acc ∪ s)
          else if !path.contains d then
            path := path.insert d
            stack := stack.push (d, deps d, 0, own d)
        else
          memo := memo.insert c acc
          path := path.erase c
          stack := stack.pop
          if let some (c', ds', i', acc') := stack.back? then
            stack := stack.set! (stack.size - 1) (c', ds', i', acc' ∪ acc)
  return memo

/-- The declarations an extension's entries in one module name, and those entries themselves (`(Name × α)` pairs). -/
unsafe def entriesOf (α : Type) (data : ModuleData) (ext : Name) : Array (Name × α) :=
  match data.entries.find? (·.1 == ext) with
  | some (_, es) => es.map fun e => unsafeCast e
  | none => #[]

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
      let some i := env.getModuleIdxFor? c | continue
      let m := env.header.moduleNames[i.toNat]!
      if !under meaning m then continue
      if ci matches .thmInfo _ then continue
      out := out.push (c, m, ci)
      stack := stack ++ meaningDeps ci
  return out

def pinInfo (env : Environment) (meaning : Array Name) (axioms : Std.HashMap Name NameSet)
    (ranges : Std.HashMap Name DeclarationRanges) (n : Name) : MetaM Json := do
  let some ci := env.find? n | return Json.mkObj [("kind", "missing")]
  let moduleOf (c : Name) : Option Name := (env.getModuleIdxFor? c).map fun i => env.header.moduleNames[i.toNat]!
  let signature := toString (← PrettyPrinter.ppSignature n).fmt
  let hypotheses ← forallTelescope ci.type fun xs _ => do
    let mut out := #[]
    for x in xs do
      let t ← inferType x
      unless ← isProp t do continue
      let named : Option Name := match t with | .const c _ => some c | _ => none
      out := out.push <| Json.mkObj [("binder", toJson (← x.fvarId!.getDecl).userName.toString),
        ("type", toJson (toString (← ppExpr t))), ("closed", toJson (!t.hasFVar && !t.hasMVar)),
        ("named", toJson (named.map toString)), ("module", toJson ((named.bind moduleOf).map toString))]
    return out
  let readJson := fun (r : Name × Name × ConstantInfo) =>
    let (c, m, ci) := r
    Json.mkObj [("name", toJson c.toString), ("module", toJson m.toString), ("kind", toJson (kind ci)),
      ("hash", toJson s!"{hex ci.type.hash}{hex (valueHash ci)}"),
      ("lines", toJson (ranges[c]?.map fun (d : DeclarationRanges) => #[d.range.pos.line, d.range.endPos.line]))]
  let stated ← match statementOf env meaning ci.type with
    | some (g, s) => do pure [("head", toJson g.toString), ("statement", toJson (toString (← ppExpr s)))]
    | none => pure []
  return Json.mkObj ([("kind", toJson (kind ci)), ("module", toJson ((moduleOf n).map toString)),
    ("signature", toJson signature), ("axioms", toJson ((axioms.getD n {}).toArray.qsort Name.lt |>.map toString)),
    ("hypotheses", toJson hypotheses), ("type_hash", toJson (hex ci.type.hash)),
    ("reads", toJson ((reads env meaning ci.type.getUsedConstants).map readJson))] ++ stated)

def names (j : Json) (k : String) : Array Name :=
  ((j.getObjValD k).getArr?.toOption.getD #[]).filterMap fun v => v.getStr?.toOption.map String.toName

end LeanAudit

open LeanAudit in
unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let some cfgPath := args.head? | throw <| IO.userError "usage: Facts.lean CONFIG"
  let cfg ← IO.ofExcept (Json.parse (← IO.FS.readFile cfgPath))
  let roots := names cfg "roots"
  let meaning := names cfg "meaning"
  let allowed := names cfg "allowed"
  let pins := names cfg "pins"
  let out := (cfg.getObjValD "out").getStr?.toOption.getD "facts.json"
  let str (j : Json) (i : Nat) : Option Name := ((j.getArrVal? i).toOption.bind (·.getStr?.toOption)).map String.toName
  let back : Std.HashMap Name Name := ((cfg.getObjValD "moved").getArr?.toOption.getD #[]).foldl (init := {}) fun m j =>
    match str j 0, str j 1 with
    | some new, some old => m.insert new old
    | _, _ => m
  let rn := renameName back
  let env ← importModules (roots.map fun r => { module := r }) {} (loadExts := false)
  let mnames := env.header.moduleNames
  let mut modules := #[]
  let mut imports := #[]
  let mut audited := #[]
  let mut ranges : Std.HashMap Name DeclarationRanges := {}
  for h : i in [0:mnames.size] do
    let m := mnames[i]
    if !under roots m && !under meaning m then continue
    let data := env.header.moduleData[i]!
    for (n, r) in entriesOf DeclarationRanges data declRangeExt.name do
      ranges := ranges.insert n r
    if !under roots m then continue
    modules := modules.push (toJson m.toString)
    imports := imports.push (m.toString, toJson (data.imports.map (·.module.toString)))
    audited := audited.push (m, data)
  let decls := audited.foldl (fun acc (_, data) => acc ++ data.constNames) #[]
  let axioms := axiomsOf env (decls ++ pins)
  let mut counts : Std.HashMap String Nat := {}
  let mut used : NameSet := {}
  let mut offenders := #[]
  let mut declared := #[]
  let mut escapes := #[]
  for (m, data) in audited do
    let impl := (entriesOf Name data Compiler.implementedByAttr.ext.name).foldl (fun s (n, _) => s.insert n) NameSet.empty
    let ext := (entriesOf Unit data externAttr.ext.name).foldl (fun s (n, _) => s.insert n) NameSet.empty
    for ci in data.constants do
      let n := ci.name
      counts := counts.insert (kind ci) (counts.getD (kind ci) 0 + 1)
      if ci matches .axiomInfo _ then
        declared := declared.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString)]
      let axs := (axioms.getD n {}).toArray
      used := axs.foldl (·.insert ·) used
      let bad := axs.filter (!allowed.contains ·)
      if !bad.isEmpty then
        offenders := offenders.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString),
          ("axioms", toJson (bad.qsort Name.lt |>.map toString))]
      let why := (if ci matches .opaqueInfo _ then #["opaque"] else #[]) ++ (if ci.isUnsafe then #["unsafe"] else #[]) ++
        (if impl.contains n then #["implemented_by"] else #[]) ++ (if ext.contains n then #["extern"] else #[])
      if !why.isEmpty then
        escapes := escapes.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString), ("why", toJson why)]
  let opts : Options := (({} : Options).set `pp.proofs false).set `format.width (100 : Nat)
  let ctx : Core.Context := { fileName := "<audit>", fileMap := default, maxHeartbeats := 0, options := opts }
  let (pinned, _) ← (MetaM.run' (pins.mapM fun p => do return (p.toString, ← pinInfo env meaning axioms ranges p))).toIO
    ctx { env }
  let mut conflicts := #[]
  if let some path := (cfg.getObjValD "canon").getStr?.toOption then
    let h ← IO.FS.Handle.mk path .write
    let mut done : NameSet := {}
    let mut named : NameSet := {}
    for p in pins do
      let some ci := env.find? p | continue
      h.putStrLn (Json.mkObj [("id", toJson s!"pin:{p}"), ("canon", toJson (canonText ci.type))]).compress
      if !back.isEmpty then
        let head := (statementOf env meaning ci.type).map fun (g, s) => (g, inlineProofs env (auxProof env ranges) s)
        named := (ci.type.getUsedConstants ++ ((head.map (·.2.getUsedConstants)).getD #[])).foldl (·.insert ·) (named.insert p)
        let unfolded := match head with
          | some (g, s) => [("head", toJson g.toString), ("unfolded", toJson (canonText s rn))]
          | none => []
        h.putStrLn (Json.mkObj ([("id", toJson s!"moved-pin:{p}"), ("as", toJson (rn p).toString),
          ("canon", toJson (canonText ci.type rn))] ++ unfolded)).compress
      for (c, _, rci) in reads env meaning ci.type.getUsedConstants do
        if done.contains c then continue
        done := done.insert c
        h.putStrLn (Json.mkObj [("id", toJson s!"read:{c}"), ("canon", toJson (readCanon rci))]).compress
        if !back.isEmpty then
          named := (uses rci).foldl (·.insert ·) (named.insert c)
          h.putStrLn (Json.mkObj [("id", toJson s!"moved-read:{c}"), ("as", toJson (rn c).toString),
            ("canon", toJson (readCanon rci rn))]).compress
    let (kept, renamed) := (named.toArray.qsort Name.lt).partition fun c => rn c == c
    let mut took : Std.HashMap Name Name := kept.foldl (fun m c => m.insert c c) {}
    for c in renamed do
      if let some d := took[rn c]? then
        conflicts := conflicts.push <| Json.mkObj [("name", toJson c.toString), ("as", toJson (rn c).toString),
          ("with", toJson d.toString)]
      else
        took := took.insert (rn c) c
  let mut oleans := #[]
  for m in mnames do
    oleans := oleans.push (toString (← findOLean m))
  let doc := Json.mkObj [("lean", toJson Lean.versionString), ("modules", toJson modules), ("imports", Json.mkObj imports.toList),
    ("oleans", toJson oleans),
    ("declarations", Json.mkObj (counts.toList.map fun (k, v) => (k, toJson v))),
    ("axioms", toJson (used.toArray.qsort Name.lt |>.map toString)), ("offenders", toJson offenders),
    ("declared_axioms", toJson declared), ("escapes", toJson escapes), ("pins", Json.mkObj pinned.toList),
    ("moved_conflicts", toJson conflicts)]
  IO.FS.writeFile out doc.compress
  return 0
