import Lean

/-!
# The audit's facts, from the package's `.olean` files as data

`lake env lean --run Facts.lean CONFIG` imports the roots named in `CONFIG` (a JSON file `audit.py` writes) with
initializers disabled and no environment extension loaded, so none of the package's code runs: no `initialize`, no
attribute handler, no delaborator. It writes one JSON document to the path `CONFIG` names. It reports facts and makes no
judgement; `audit.py` holds the policy.

- `modules`: the imported modules under the roots; `imports`: what each imports. `oleans`: the `.olean` file of every
  imported module, the roots' dependencies' included; `graph`: each one's `.olean` file and what it imports.
- `declarations`: how many constants of each kind those modules declare.
- `axioms`: every axiom any of them uses, found by walking the bodies; `offenders`: each one that uses an axiom outside
  `allowed`; `unreported_errors`: each one holding a `sorry` the elaborator made for an error (`sorryAx α true`) where
  a module built, so the error was never reported.
- `declared_axioms`: the `axiom` declarations among them.
- `escapes`: the `opaque` ones (a `partial def` is one), and the `unsafe`, `@[implemented_by]` and `@[extern]` ones:
  compiled code with no definition a proof can unfold. The attributes are read from the modules' data.
- `pins`: for each name in `pins`, its kind, where its declaration starts (`at`: line, codepoint column), signature,
  axioms, `Prop` hypotheses, the hash of its type, and the
  definitions its statement reads, followed through every definition, inductive type and constructor declared under
  `meaning`, with their source lines and the constants under each one's declaration's name (`family`) that its own type,
  value or constructors use (`reads`, `own`: what its record hashes with it), and the constants other than theorems outside
  `meaning` where that walk
  stops, with their modules (`outside`). Signatures print with no notation, since no notation is loaded: nothing a
  package declares can change how its statements read.
- `spec_definitions`: the definitions and inductive types the audited modules under `spec` declare in their source (with a
  declaration range: what Lean generates for them has none), projections and internal names aside, each with whether it
  is a statement (`def G : Prop`).
- `instances`: each constant with a declaration range whose name (a private one's as written) ends in a component
  starting with `inst`, with that component as an identifier (`id`) and where its declaration's name is (`at`: line,
  codepoint column). An anonymous `instance` has its name there as the keyword `instance` itself.
- `spans`: for each module under the roots, the first and last source line of each declaration range it holds, so that
  `audit.py` can find a declaration in the source that the module does not hold.

The hashes in the document are Lean's `Expr.hash`, which has 32 bits; `audit.py` keeps them only to check records written
before SHA-256. When `CONFIG` names a `canon` path, the pins' statements and the definitions they read are also written
there, one JSON line each, as `canon` texts, and `audit.py` hashes those with SHA-256. A pinned statement that is a
definition under `meaning` applied under binders (`∀ xs, G as`) is also written with that definition unfolded once
(`liftOf`), with the definition and its module, so that `audit.py` can certify a statement lifted into a `Guarantees`
definition as the statement its record holds.

With `moved` (pairs of a new name, a declaration or a namespace, and the old name it moved from), each of those texts is
written once more with the names as they were before the move (the longest moved prefix renamed back), and a pinned
statement that is such a definition also with that definition unfolded (`liftOf`), so that
`audit.py` can compare them with records made before the move. `moved_conflicts` lists each constant of those texts (or
pin) whose old name would be another one's too: there the comparison would not tell the two apart. A constant that keeps
its name stands for itself, and an old name that only a declaration outside the texts still has (an old theorem kept as a
lemma) is no conflict.
-/

open Lean Meta

namespace LeanAudit

def under (ps : Array Name) (m : Name) : Bool := ps.any (·.isPrefixOf m)

/-- The module with index `i`. `EnvironmentHeader.moduleNames` maps over every imported module on each call, which made
the walks over `Security`'s guarantees' reads take 25 minutes. -/
def moduleName (env : Environment) (i : ModuleIdx) : Name := env.header.modules[i.toNat]!.module

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

/-- The value of a constant that is a theorem the elaborator made from a proof nested in a definition
(`<decl>._proof_<n>`, `abstractNestedProofs`), a proof a theorem's statement keeps in place. -/
def auxProof? (env : Environment) : Expr → Option Expr
  | .const n@(.str _ s) us =>
    match env.find? n with
    | some (.thmInfo v) =>
      if s.startsWith "_proof_" then some (v.value.instantiateLevelParams v.levelParams us) else none
    | _ => none
  | _ => none

/-- `e` with each auxiliary proof (`auxProof?`) replaced by its value applied to its arguments. -/
partial def inlineAuxProofs (env : Environment) (e : Expr) : Expr :=
  let e' := e.replace fun x => (auxProof? env x.getAppFn).map (·.beta x.getAppArgs)
  if e' == e then e else inlineAuxProofs env e'

/-- When a statement is a definition declared under `meaning` applied to arguments under binders, `∀ xs, G.{us} as` (a
theorem `t (xs) : G as` that proves a named statement), that definition and the statement with it unfolded once: the
binders `xs` as they are, and in place of `G.{us} as` the value of `G` at the levels `us` with its leading `fun`s
instantiated by `as` (`Expr.beta`, as `Meta.unfoldDefinition?` unfolds an applied constant), with the proofs the
elaborator abstracted out of that value put back, so that it reads as the same statement written as a theorem's type
would. Nothing else is reduced. -/
partial def liftOf (env : Environment) (meaning : Array Name) : Expr → Option (Name × Expr)
  | .forallE n d b bi => (liftOf env meaning b).map fun (g, b') => (g, .forallE n d b' bi)
  | .mdata _ e => liftOf env meaning e
  | e => do
    let .const g us := e.getAppFn | none
    let .defnInfo v ← env.find? g | none
    let i ← env.getModuleIdxFor? g
    guard (us.length == v.levelParams.length && under meaning (moduleName env i))
    return (g, inlineAuxProofs env ((v.value.instantiateLevelParams v.levelParams us).beta e.getAppArgs))

/-- What a constant's axioms depend on (as `Lean.collectAxioms` walks it). -/
def uses : ConstantInfo → Array Name
  | .axiomInfo v => v.type.getUsedConstants
  | .defnInfo v | .thmInfo v | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray
  | .ctorInfo v => v.type.getUsedConstants
  | .recInfo v => v.type.getUsedConstants
  | .quotInfo _ => #[]

/-- A `sorry` the elaborator made for an error (`sorryAx α true`), as opposed to a written one (`sorryAx α false`). -/
def syntheticSorry (e : Expr) : Bool :=
  (e.find? fun s => s.isAppOf ``sorryAx && s.getAppNumArgs ≥ 2 && s.getArg! 1 == mkConst ``Bool.true).isSome

/-- Whether `ci`'s type or value holds a `syntheticSorry`. -/
def unreported : ConstantInfo → Bool
  | .defnInfo v | .thmInfo v | .opaqueInfo v => syntheticSorry v.type || syntheticSorry v.value
  | ci => syntheticSorry ci.type

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

/-- The definitions under `meaning` that the constants `roots` read, transitively, and the constants other than theorems
outside `meaning` that the walk reaches and does not follow (with their modules). The state holds each definition's
`meaningDeps`, shared across calls: a read definition's value is walked once per run, not once per guarantee. -/
def reads (env : Environment) (meaning : Array Name) (roots : Array Name) :
    StateM (Std.HashMap Name (Array Name)) (Array (Name × Name × ConstantInfo) × Array (Name × Name)) := do
  let mut seen : NameSet := {}
  let mut stack := roots
  let mut out := #[]
  let mut outside := #[]
  repeat
    match stack.back? with
    | none => break
    | some c =>
      stack := stack.pop
      if seen.contains c then continue
      seen := seen.insert c
      let some ci := env.find? c | continue
      let some i := env.getModuleIdxFor? c | continue
      let m := moduleName env i
      if ci matches .thmInfo _ then continue
      if !under meaning m then
        outside := outside.push (c, m)
        continue
      out := out.push (c, m, ci)
      let ds ← match (← get)[c]? with
        | some ds => pure ds
        | none => let ds := meaningDeps ci; modifyGet fun memo => (ds, memo.insert c ds)
      stack := stack ++ ds
  return (out, outside)

/-- `c` less the trailing components with a leading `_` that Lean's compilers add (`batch._f`, `go._unary`): the declaration
under whose name `c`'s `own` constants are, so that a helper's `own` names its declaration's matchers. -/
def family : Name → Name
  | .str p s => if s.startsWith "_" && !p.isAnonymous then family p else .str p s
  | n => n

/-- The constants an expression uses and those the definitions under `meaning` it uses read, transitively (`reads`). -/
def reaches (env : Environment) (meaning : Array Name) (e : Expr) : Array String :=
  let cs := e.getUsedConstants
  let (rs, outside) := ((reads env meaning cs).run {}).1
  let all := (cs ++ rs.map (·.1) ++ outside.map (·.1)).foldl (fun (s : NameSet) c => s.insert c) {}
  all.toArray.qsort Name.lt |>.map toString

/-- The constants an expression uses and, transitively, those the definitions under `meaning` it uses read, without
entering an inductive type's constructors: mentioning a structure is not applying the functions its fields' types name
(an outer session's `hown` names `Zk.verify`), while a definition that applies them still reaches them. -/
def applies (env : Environment) (meaning : Array Name) (e : Expr) : Array String := Id.run do
  let mut seen : NameSet := {}
  let mut stack := e.getUsedConstants
  repeat
    match stack.back? with
    | none => break
    | some c =>
      stack := stack.pop
      if seen.contains c then continue
      seen := seen.insert c
      let some ci := env.find? c | continue
      let some i := env.getModuleIdxFor? c | continue
      if ci matches .thmInfo _ | .inductInfo _ | .ctorInfo _ | .recInfo _ then continue
      if !under meaning (moduleName env i) then continue
      stack := stack ++ meaningDeps ci
  return seen.toArray.qsort Name.lt |>.map toString

/-- One hypothesis of a statement's shape: where it came from (`binder`, a binder of the statement, or `field`, a `Prop`
field of a structure the package declares that a binder ranges over), its name, the binder's position (`at`), its type,
its head constant and that constant's module, and what it reaches (`reaches`). The head is its body's, under its leading
binders (`∀ j, P j` is headed by `P`); `guarded` says those binders include a hypothesis (`A → B`), and `applies` is what
its body applies (`applies`). A conjunction counts as each of its conjuncts. -/
partial def hypothesisJson (env : Environment) (meaning : Array Name) (moduleOf : Name → Option Name) (source name : String)
    (pos : Nat) (t : Expr) : MetaM (Array Json) := do
  let t ← instantiateMVars t
  if t.isAppOfArity ``And 2 then
    return (← hypothesisJson env meaning moduleOf source s!"{name}.1" pos t.appFn!.appArg!) ++
      (← hypothesisJson env meaning moduleOf source s!"{name}.2" pos t.appArg!)
  let (head, guarded, body) ← forallTelescope t fun ys b => do
    let guarded ← ys.anyM fun y => do isProp (← y.fvarId!.getType)
    let head : Option Name := match b.getAppFn with | .const c _ => some c | _ => none
    return (head, guarded, applies env meaning b)
  return #[Json.mkObj [("source", toJson source), ("name", toJson name), ("at", toJson pos),
    ("type", toJson (toString (← ppExpr t))),
    ("head", toJson (head.map toString)), ("module", toJson ((head.bind moduleOf).map toString)),
    ("reaches", toJson (reaches env meaning t)), ("guarded", toJson guarded), ("applies", toJson body)]]

partial def conjuncts (e : Expr) : List Expr :=
  if e.isAppOfArity ``And 2 then conjuncts e.appFn!.appArg! ++ conjuncts e.appArg! else [e]

/-- The events of a conclusion: each argument `fun o => A₁ ∧ … ∧ Aₙ` of a constant `c` (`prob (fun o => …) g σ`) whose
body is a proposition, under any binders of the conclusion (`∃ h, … ∨ prob …`), with `c` (`head`) and, for each
conjunct `Aᵢ` of the body (`conjuncts`, its `∧`s flattened), what it applies (`applies`) and whether it holds under a
hypothesis of its own (`guarded`). -/
partial def eventsOf (env : Environment) (meaning : Array Name) (e : Expr) : MetaM (Array Json) := do
  match e with
  | .mdata _ b => eventsOf env meaning b
  | .proj _ _ b => eventsOf env meaning b
  | .lam n t b bi | .forallE n t b bi =>
    withLocalDecl n bi t fun x => eventsOf env meaning (b.instantiate1 x)
  | .app .. =>
    let args := e.getAppArgs
    let mut out := #[]
    if let .const c _ := e.getAppFn then
      for a in args do
        if !a.isLambda then continue
        out := out ++ (← lambdaTelescope a fun _ b => do
          unless ← isProp b do return #[]
          let cs ← (conjuncts b).mapM fun e => forallTelescope e fun ys body => do
            let guarded ← ys.anyM fun y => do isProp (← y.fvarId!.getType)
            return Json.mkObj [("applies", toJson (applies env meaning body)), ("guarded", toJson guarded)]
          return #[Json.mkObj [("head", toJson c.toString), ("conjuncts", toJson cs)]])
    for a in args do
      out := out ++ (← eventsOf env meaning a)
    return out
  | _ => return #[]

/-- One part of a statement's shape, for the audit's shape rule: its binders (`inputs`: those that aren't propositions,
each with its position and its type's head), its hypotheses (`hypothesisJson`: each `Prop` binder, and each `Prop` field
of a structure an input ranges over), what its conclusion reaches, and its events (`eventsOf`). -/
def shapeOf (env : Environment) (meaning : Array Name) (moduleOf : Name → Option Name) (stmt : Expr) : MetaM Json :=
  forallTelescope stmt fun xs concl => do
    let mut inputs := #[]
    let mut hyps := #[]
    for h : i in [0:xs.size] do
      let d ← xs[i].fvarId!.getDecl
      let t ← instantiateMVars d.type
      if ← isProp t then
        hyps := hyps ++ (← hypothesisJson env meaning moduleOf "binder" d.userName.toString i t)
        continue
      let head : Option Name := match t.getAppFn with | .const c _ => some c | _ => none
      inputs := inputs.push <| Json.mkObj [("name", toJson d.userName.toString), ("at", toJson i),
        ("head", toJson (head.map toString))]
      -- a family of structures (`Vs : Fin S → VStmt`) holds their fields at every index
      hyps := hyps ++ (← forallTelescope t fun _ cod => do
        let cod ← whnfR cod
        let some s := (match cod.getAppFn with | .const c _ => some c | _ => none) | return #[]
        if !((moduleOf s).map (under meaning ·)).getD false then return #[]
        let some (.inductInfo iv) := env.find? s | return #[]
        let [ctor] := iv.ctors | return #[]
        if iv.numIndices != 0 || iv.isRec then return #[]
        let some (.ctorInfo cv) := env.find? ctor | return #[]
        let ctorType ← instantiateForall (cv.type.instantiateLevelParams cv.levelParams cod.getAppFn.constLevels!)
          (cod.getAppArgs.extract 0 cv.numParams)
        forallTelescope ctorType fun fs _ => do
          let mut out := #[]
          for f in fs do
            let fd ← f.fvarId!.getDecl
            if ← isProp fd.type then
              out := out ++ (← hypothesisJson env meaning moduleOf "field" s!"{d.userName}.{fd.userName}" i fd.type)
          return out)
    return Json.mkObj [("inputs", toJson inputs), ("hypotheses", toJson hyps),
      ("conclusion", toJson (reaches env meaning concl)), ("events", toJson (← eventsOf env meaning concl))]

/-- The statement's shape, one part per conjunct of a statement that is a conjunction (`shapeOf` each), after a lifted
definition is unfolded once (`liftOf`) and its `let`s are inlined. -/
def shapeInfo (env : Environment) (meaning : Array Name) (moduleOf : Name → Option Name) (stmt : Expr) : MetaM Json := do
  try
    let stmt ← zetaReduce (((liftOf env meaning stmt).map (·.2)).getD stmt)
    return toJson (← (conjuncts stmt).mapM (shapeOf env meaning moduleOf))
  catch e => return Json.mkObj [("error", toJson (← e.toMessageData.toString))]

/-- A premise's head constant: past its binders, after reducible unfolding, again past the binders an abbrev unfolds to
(`Premises.P x := ∀ y, Assumptions.P x y`), at most `n` times; an assumptions module's constant is the head as it
stands, never unfolded. -/
partial def premiseHead (moduleOf : Name → Option Name) (n : Nat) (t : Expr) : MetaM (Option Name) :=
  forallTelescope t fun _ b => do
    let head (e : Expr) : Option Name := match e.getAppFn with | .const h _ => some h | _ => none
    if let some h := head b then
      if ((moduleOf h).map (·.components.contains `Assumptions)).getD false then return some h
    let b ← whnfR b
    if b.isForall && n > 0 then premiseHead moduleOf (n - 1) b else return head b

/-- The premises a statement takes: the `Prop` hypotheses of each conjunct, also under the statement definitions (a
`Guarantees` namespace's) its conjuncts apply, unfolded at most `depth` deep, each named by its head after reducible
unfolding, so a `Premises` abbrev names the assumption it stands for. A conclusion is never unfolded. -/
partial def premisesOf (env : Environment) (moduleOf : Name → Option Name) (depth : Nat) (e : Expr) : MetaM (Array Json) := do
  let mut out := #[]
  for c in conjuncts e do
    out := out ++ (← forallTelescope c fun xs body => do
      let mut ps := #[]
      for x in xs do
        let t ← inferType x
        unless ← isProp t do continue
        let head ← premiseHead moduleOf 4 t
        if let some h := head then
          ps := ps.push <| Json.mkObj [("binder", toJson (← x.fvarId!.getDecl).userName.toString),
            ("head", toJson h.toString), ("module", toJson ((moduleOf h).map toString))]
      if depth > 0 then
        if let .const d us := body.getAppFn then
          if d.components.contains `Guarantees then
            if let some (.defnInfo dv) := env.find? d then
              ps := ps ++ (← premisesOf env moduleOf (depth - 1)
                ((dv.value.instantiateLevelParams dv.levelParams us).beta body.getAppArgs))
      return ps)
  return out

def pinInfo (env : Environment) (meaning : Array Name) (axioms : Std.HashMap Name NameSet)
    (ranges : Std.HashMap Name DeclarationRanges) (rs : Array (Name × Name × ConstantInfo) × Array (Name × Name)) (n : Name) :
    MetaM Json := do
  let some ci := env.find? n | return Json.mkObj [("kind", "missing")]
  let moduleOf (c : Name) : Option Name := (env.getModuleIdxFor? c).map fun i => moduleName env i
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
    let fam := family c
    let own := (meaningDeps ci).foldl (fun (s : NameSet) d => if d != c && fam.isPrefixOf d then s.insert d else s) {}
    Json.mkObj [("name", toJson c.toString), ("module", toJson m.toString), ("kind", toJson (kind ci)),
      ("hash", toJson s!"{hex ci.type.hash}{hex (valueHash ci)}"),
      ("own", toJson (own.toArray.qsort Name.lt |>.map toString)),
      ("lines", toJson (ranges[c]?.map fun (d : DeclarationRanges) => #[d.range.pos.line, d.range.endPos.line]))]
  return Json.mkObj [("kind", toJson (kind ci)), ("module", toJson ((moduleOf n).map toString)),
    ("at", toJson (ranges[n]?.map fun (d : DeclarationRanges) => #[d.range.pos.line, d.range.pos.column])),
    ("signature", toJson signature), ("axioms", toJson ((axioms.getD n {}).toArray.qsort Name.lt |>.map toString)),
    ("hypotheses", toJson hypotheses), ("shape", ← shapeInfo env meaning moduleOf ci.type),
    ("premises", ← (do return toJson (← premisesOf env moduleOf 3 ci.type)) <|> pure Json.null),
    ("type_hash", toJson (hex ci.type.hash)),
    ("reads", toJson (rs.1.map readJson)),
    ("outside", toJson (rs.2.map fun (c, m) => #[c.toString, m.toString]))]

/-- Whether a definition is a structure's projection, whose value is the field (the audit loads no environment
extension, so `isProjectionFn` would not know). -/
def projection : ConstantInfo → Bool
  | .defnInfo v => go v.value
  | _ => false
where go : Expr → Bool
  | .lam _ _ b _ => go b
  | .proj .. => true
  | _ => false

/-- Whether a constant is one of the definitions a spec module declares that `spec_definitions` lists: not a theorem, a
constructor or recursor, an internal name, or a projection. -/
def specDefinition (ci : ConstantInfo) : Bool :=
  (ci matches .defnInfo _ | .opaqueInfo _ | .inductInfo _) && !ci.name.isInternal && !projection ci

def names (j : Json) (k : String) : Array Name :=
  ((j.getObjValD k).getArr?.toOption.getD #[]).filterMap fun v => v.getStr?.toOption.map String.toName

end LeanAudit

open LeanAudit in
/-- The verifier's path: the constants its `entries` reach through definitions' values and `implemented_by` targets, in
every imported module, expanding nothing in `stop`. Reported: each reached `@[implemented_by]` and `@[extern]` constant,
and each constant of the `hash` namespace the walk enters from outside it, with the path from an entry. -/
unsafe def verifierInfo (env : Environment) (entries stop : Array Name) (hash : Name) : Json := Id.run do
  let moduleOf (c : Name) : Option Name := (env.getModuleIdxFor? c).map fun i => moduleName env i
  let mut impl : Std.HashMap Name Name := {}
  let mut ext : NameSet := {}
  for data in env.header.moduleData do
    for (n, t) in entriesOf Name data Compiler.implementedByAttr.ext.name do
      impl := impl.insert n t
    for (n, _) in entriesOf Unit data externAttr.ext.name do
      ext := ext.insert n
  let stops := stop.foldl NameSet.insert {}
  let mut seen : NameSet := {}
  let mut parent : Std.HashMap Name Name := {}
  let mut stack := #[]
  for e in entries do
    if env.contains e then
      seen := seen.insert e
      stack := stack.push e
  while !stack.isEmpty do
    let n := stack.back!
    stack := stack.pop
    if stops.contains n then continue
    let next := (match env.find? n with
      | some (.defnInfo d) => d.value.getUsedConstants
      | some (.opaqueInfo d) => d.value.getUsedConstants
      | _ => #[]) ++ (impl[n]?.toArray)
    for c in next do
      if !seen.contains c then
        seen := seen.insert c
        parent := parent.insert c n
        stack := stack.push c
  let pathTo (c : Name) : Array String := Id.run do
    let mut p := #[c.toString]
    let mut x := c
    for _ in [0:16] do
      match parent[x]? with
      | some y => p := p.push y.toString; x := y
      | none => break
    return p.reverse
  let reached := seen.toArray.qsort Name.lt
  let locOf (c : Name) : List (String × Json) := [("name", toJson c.toString), ("module", toJson ((moduleOf c).map toString))]
  return Json.mkObj [
    ("missing", toJson ((entries.filter (!env.contains ·)).map toString)), ("reached", toJson reached.size),
    ("implemented_by", toJson (reached.filterMap fun c => impl[c]?.map fun t => Json.mkObj (locOf c ++ [("by", toJson t.toString)]))),
    ("extern", toJson (reached.filter ext.contains |>.map fun c => Json.mkObj (locOf c))),
    ("hash", toJson (reached.filter (fun c => hash.isPrefixOf c && !(parent[c]?.map hash.isPrefixOf).getD true)
      |>.map fun c => Json.mkObj (locOf c ++ [("path", toJson (pathTo c))])))]

open LeanAudit in
unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let some cfgPath := args.head? | throw <| IO.userError "usage: Facts.lean CONFIG"
  let cfg ← IO.ofExcept (Json.parse (← IO.FS.readFile cfgPath))
  let roots := names cfg "roots"
  let meaning := names cfg "meaning"
  let allowed := names cfg "allowed"
  let pins := names cfg "pins"
  let spec := names cfg "spec"
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
  let mut spans := #[]
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
    spans := spans.push (m.toString, toJson ((entriesOf DeclarationRanges data declRangeExt.name).map fun (_, r) =>
      #[r.range.pos.line, r.range.endPos.line]))
    audited := audited.push (m, data)
  let decls := audited.foldl (fun acc (_, data) => acc ++ data.constNames) #[]
  let axioms := axiomsOf env (decls ++ pins)
  let mut counts : Std.HashMap String Nat := {}
  let mut used : NameSet := {}
  let mut offenders := #[]
  let mut unreportedErrors := #[]
  let mut declared := #[]
  let mut escapes := #[]
  let mut specDefs := #[]
  let mut instances := #[]
  for (m, data) in audited do
    let impl := (entriesOf Name data Compiler.implementedByAttr.ext.name).foldl (fun s (n, _) => s.insert n) NameSet.empty
    let ext := (entriesOf Unit data externAttr.ext.name).foldl (fun s (n, _) => s.insert n) NameSet.empty
    for ci in data.constants do
      let n := ci.name
      counts := counts.insert (kind ci) (counts.getD (kind ci) 0 + 1)
      if under spec m && specDefinition ci && ranges.contains n then
        specDefs := specDefs.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString),
          ("kind", toJson (kind ci)), ("statement", toJson ci.type.isProp)]
      if let (some r, .str _ s) := (ranges[n]?, privateToUserName n) then
        if s.startsWith "inst" then
          instances := instances.push <| Json.mkObj [("name", toJson (privateToUserName n).toString),
            ("id", toJson (Name.mkSimple s).toString), ("module", toJson m.toString),
            ("at", toJson #[r.selectionRange.pos.line, r.selectionRange.pos.column])]
      if ci matches .axiomInfo _ then
        declared := declared.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString)]
      let axs := (axioms.getD n {}).toArray
      used := axs.foldl (·.insert ·) used
      let bad := axs.filter (!allowed.contains ·)
      if !bad.isEmpty then
        offenders := offenders.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString),
          ("axioms", toJson (bad.qsort Name.lt |>.map toString))]
      if (uses ci).contains ``sorryAx && unreported ci then
        unreportedErrors := unreportedErrors.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString)]
      let why := (if ci matches .opaqueInfo _ then #["opaque"] else #[]) ++ (if ci.isUnsafe then #["unsafe"] else #[]) ++
        (if impl.contains n then #["implemented_by"] else #[]) ++ (if ext.contains n then #["extern"] else #[])
      if !why.isEmpty then
        escapes := escapes.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString), ("why", toJson why)]
  let opts : Options := (({} : Options).set `pp.proofs false).set `format.width (100 : Nat)
  let ctx : Core.Context := { fileName := "<audit>", fileMap := default, maxHeartbeats := 0, options := opts }
  let readsOf : Std.HashMap Name (Array (Name × Name × ConstantInfo) × Array (Name × Name)) := Id.run do
    let mut memo := {}
    let mut acc := {}
    for p in pins do
      if let some ci := env.find? p then
        let (rs, memo') := (reads env meaning ci.type.getUsedConstants).run memo
        memo := memo'
        acc := acc.insert p rs
    return acc
  let (pinned, _) ← (MetaM.run' (pins.mapM fun p => do
    return (p.toString, ← pinInfo env meaning axioms ranges (readsOf.getD p (#[], #[])) p))).toIO ctx { env }
  let mut conflicts := #[]
  if let some path := (cfg.getObjValD "canon").getStr?.toOption then
    let h ← IO.FS.Handle.mk path .write
    let mut done : NameSet := {}
    let mut named : NameSet := {}
    for p in pins do
      let some ci := env.find? p | continue
      h.putStrLn (Json.mkObj [("id", toJson s!"pin:{p}"), ("canon", toJson (canonText ci.type))]).compress
      let head := liftOf env meaning ci.type
      if let some (g, s) := head then
        let m := (env.getModuleIdxFor? g).map fun i => (moduleName env i).toString
        h.putStrLn (Json.mkObj [("id", toJson s!"lift:{p}"), ("head", toJson g.toString), ("module", toJson m),
          ("canon", toJson (canonText s))]).compress
      if !back.isEmpty then
        named := (ci.type.getUsedConstants ++ ((head.map (·.2.getUsedConstants)).getD #[])).foldl (·.insert ·) (named.insert p)
        let unfolded := match head with
          | some (g, s) => [("head", toJson g.toString), ("unfolded", toJson (canonText s rn))]
          | none => []
        h.putStrLn (Json.mkObj ([("id", toJson s!"moved-pin:{p}"), ("as", toJson (rn p).toString),
          ("canon", toJson (canonText ci.type rn))] ++ unfolded)).compress
      for (c, _, rci) in (readsOf.getD p (#[], #[])).1 do
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
  let mut graph := #[]
  for h : i in [0:mnames.size] do
    let olean := toString (← findOLean mnames[i])
    oleans := oleans.push olean
    graph := graph.push (mnames[i].toString, Json.mkObj [("olean", toJson olean),
      ("imports", toJson (env.header.moduleData[i]!.imports.map (·.module.toString)))])
  let doc := Json.mkObj [("lean", toJson Lean.versionString), ("modules", toJson modules), ("imports", Json.mkObj imports.toList),
    ("spans", Json.mkObj spans.toList), ("oleans", toJson oleans), ("graph", Json.mkObj graph.toList),
    ("declarations", Json.mkObj (counts.toList.map fun (k, v) => (k, toJson v))),
    ("axioms", toJson (used.toArray.qsort Name.lt |>.map toString)), ("offenders", toJson offenders),
    ("unreported_errors", toJson unreportedErrors), ("declared_axioms", toJson declared), ("escapes", toJson escapes), ("spec_definitions", toJson specDefs),
    ("instances", toJson instances), ("pins", Json.mkObj pinned.toList),
    ("moved_conflicts", toJson conflicts)]
  let vcfg := cfg.getObjValD "verifier"
  let doc := if (names vcfg "entries").isEmpty then doc else
    doc.setObjVal! "verifier" (verifierInfo env (names vcfg "entries") (names vcfg "stop")
      (((vcfg.getObjValD "hash").getStr?.toOption.getD "").toName))
  IO.FS.writeFile out doc.compress
  return 0
