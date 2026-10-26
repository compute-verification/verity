import Lean

/-!
# The audit's facts, from the package's `.olean` files as data

`lake env FACTS CONFIG`, where `FACTS` is this file compiled (`lean -c`, then `leanc`; `lake env lean --run Facts.lean CONFIG`
is the same program interpreted), imports the roots named in `CONFIG` (a JSON file `check.py` writes) with
initializers disabled and no environment extension loaded, so none of the package's code runs: no `initialize`, no
attribute handler, no delaborator. It writes one JSON document to the path `CONFIG` names. It reports facts and makes no
judgement; `check.py` holds the policy.

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
- `assumption_reads`: for each name under `assumptions`, and each premise head and named hypothesis a pin reports, that
  constant and every definition, inductive type and constructor it reads, transitively, in the modules under `meaning`
  and those a package of this repository builds (an `.olean` in its own `.lake/build`), wherever its module lies; its
  canon text and theirs are `read:` lines. A name the walk doesn't hold (a theorem, a fetched dependency's) is left out.
- `spec_definitions`: the definitions and inductive types the audited modules under `spec` declare in their source (with a
  declaration range: what Lean generates for them has none), projections and internal names aside, each with whether it
  is a statement (`def G : Prop`) and whether it is a predicate (its type, past its binders, is `Prop`).
- each pin's `body`: the constant its statement applies past its `∀` and `∃` binders (`head`), and whether a `¬` stands
  before that application (`negated`), read as written: nothing is unfolded; and its `conclusion`, split at `∧`, with
  what each conjunct applies (`conclusionOf`).
- `instances`: each constant with a declaration range whose name (a private one's as written) ends in a component
  starting with `inst`, with that component as an identifier (`id`) and where its declaration's name is (`at`: line,
  codepoint column). An anonymous `instance` has its name there as the keyword `instance` itself.
- `spans`: for each module under the roots, the first and last source line of each declaration range it holds, so that
  `check.py` can find a declaration in the source that the module does not hold.

The hashes in the document are Lean's `Expr.hash`, which has 32 bits; `check.py` keeps them only to check records written
before SHA-256. When `CONFIG` names a `canon` path, the pins' statements and the definitions they read are also written
there, one JSON line each, as `canon` texts (and those `assumption_reads` holds), and `check.py` hashes those with SHA-256. A pinned statement that is a
definition under `meaning` applied under binders (`∀ xs, G as`) is also written with that definition unfolded once
(`liftOf`), with the definition and its module, so that `check.py` can certify a statement lifted into a `Guarantees`
definition as the statement its record holds.

With `moved` (pairs of a new name, a declaration or a namespace, and the old name it moved from), each of those texts is
written once more with the names as they were before the move (the longest moved prefix renamed back), and a pinned
statement that is such a definition also with that definition unfolded (`liftOf`), so that
`check.py` can compare them with records made before the move. `moved_conflicts` lists each constant of those texts (or
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

/-- A canonical text of `e`, which `check.py` hashes with SHA-256: one line per distinct node, its children given as the
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

/-- Each imported constant's `uses`, so that the walks below read a body's constants once per run. The modules are
walked in tasks, a run of them each. -/
def usesTable (env : Environment) : Std.HashMap Name (Array Name) :=
  let data := env.header.moduleData
  let tasks := (List.range ((data.size + 15) / 16)).toArray.map fun k => Task.spawn fun _ =>
    (data.extract (16 * k) (16 * k + 16)).foldl (init := #[]) fun acc d =>
      d.constants.foldl (init := acc) fun acc ci => acc.push (ci.name, uses ci)
  tasks.foldl (init := {}) fun t task => task.get.foldl (init := t) fun t (n, ds) => t.insert n ds

/-- `uses` of the constant `c`, from `table` (`usesTable`) when it holds it. -/
def usesIn (env : Environment) (table : Std.HashMap Name (Array Name)) (c : Name) : Array Name :=
  match table[c]? with
  | some ds => ds
  | none => (env.find? c).map uses |>.getD #[]

/-- A `sorry` the elaborator made for an error (`sorryAx α true`), as opposed to a written one (`sorryAx α false`). -/
def syntheticSorry (e : Expr) : Bool :=
  (e.find? fun s => s.isAppOf ``sorryAx && s.getAppNumArgs ≥ 2 && s.getArg! 1 == mkConst ``Bool.true).isSome

/-- Whether `ci`'s type or value holds a `syntheticSorry`. -/
def unreported : ConstantInfo → Bool
  | .defnInfo v | .thmInfo v | .opaqueInfo v => syntheticSorry v.type || syntheticSorry v.value
  | ci => syntheticSorry ci.type

/-- What a statement means through a constant: its type, a definition's value, an inductive type's constructors.
Theorems contribute nothing (a proof is irrelevant to what a statement says). Its `uses` are read from `table`
(`usesTable`). -/
def meaningDeps (env : Environment) (table : Std.HashMap Name (Array Name)) : ConstantInfo → Array Name
  | .thmInfo _ | .quotInfo _ => #[]
  | ci => usesIn env table ci.name

/-- The axioms each of `names` uses, by walking bodies (their `uses` read from `table`), memoized. A constant already on
the path contributes nothing, as in `Lean.collectAxioms` (an inductive type and its constructors). A constant's
dependencies are taken in order, up to the first one neither memoized nor on the path, which is walked next. -/
def axiomsOf (env : Environment) (table : Std.HashMap Name (Array Name)) (names : Array Name) :
    Std.HashMap Name NameSet := Id.run do
  let own (c : Name) : NameSet := if env.find? c matches some (.axiomInfo _) then NameSet.empty.insert c else {}
  let deps (c : Name) : Array Name := usesIn env table c
  let join (acc s : NameSet) : NameSet := if s.all acc.contains then acc else acc ∪ s
  let mut memo : Std.HashMap Name NameSet := {}
  for root in names do
    if memo.contains root then continue
    let mut stack : Array (Name × Array Name × Nat × NameSet) := #[(root, deps root, 0, own root)]
    let mut path : NameSet := NameSet.empty.insert root
    repeat
      let some (c, ds, i, acc) := stack.back? | break
      stack := stack.pop
      let mut i := i
      let mut acc := acc
      let mut next := none
      while h : i < ds.size do
        let d := ds[i]
        i := i + 1
        if let some s := memo[d]? then
          acc := join acc s
        else if !path.contains d then
          next := some d
          break
      if let some d := next then
        stack := stack.push (c, ds, i, acc)
        path := path.insert d
        stack := stack.push (d, deps d, 0, own d)
      else
        memo := memo.insert c acc
        path := path.erase c
        if let some (c', ds', i', acc') := stack.back? then
          stack := stack.set! (stack.size - 1) (c', ds', i', join acc' acc)
  return memo

/-- The declarations an extension's entries in one module name, and those entries themselves (`(Name × α)` pairs). -/
unsafe def entriesOf (α : Type) (data : ModuleData) (ext : Name) : Array (Name × α) :=
  match data.entries.find? (·.1 == ext) with
  | some (_, es) => es.map fun e => unsafeCast e
  | none => #[]

/-- The definitions of the modules `inScope` admits that the constants `roots` read, transitively, and the constants other
than theorems outside them that the walk reaches and does not follow (with their modules). Each definition's `meaningDeps`
is read from `table` (`usesTable`): a read definition's value is walked once per run, not once per guarantee. -/
def readsWhere (env : Environment) (table : Std.HashMap Name (Array Name)) (inScope : Name → Bool) (roots : Array Name) :
    Array (Name × Name × ConstantInfo) × Array (Name × Name) := Id.run do
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
      if !inScope m then
        outside := outside.push (c, m)
        continue
      out := out.push (c, m, ci)
      stack := stack ++ meaningDeps env table ci
  return (out, outside)

/-- `readsWhere` over the modules under `meaning`: what a guarantee's statement reads. -/
def reads (env : Environment) (table : Std.HashMap Name (Array Name)) (meaning : Array Name) (roots : Array Name) :
    Array (Name × Name × ConstantInfo) × Array (Name × Name) :=
  readsWhere env table (under meaning) roots

/-- `c` less the trailing components with a leading `_` that Lean's compilers add (`batch._f`, `go._unary`): the declaration
under whose name `c`'s `own` constants are, so that a helper's `own` names its declaration's matchers. -/
def family : Name → Name
  | .str p s => if s.startsWith "_" && !p.isAnonymous then family p else .str p s
  | n => n

/-- The constants an expression uses and those the definitions under `meaning` it uses read, transitively (`reads`). -/
def reaches (env : Environment) (table : Std.HashMap Name (Array Name)) (meaning : Array Name) (e : Expr) :
    Array String :=
  let cs := e.getUsedConstants
  let (rs, outside) := reads env table meaning cs
  let all := (cs ++ rs.map (·.1) ++ outside.map (·.1)).foldl (fun (s : NameSet) c => s.insert c) {}
  all.toArray.qsort Name.lt |>.map toString

/-- The constants an expression uses and, transitively, those the definitions under `meaning` it uses read, without
entering an inductive type's constructors: mentioning a structure is not applying the functions its fields' types name
(an outer session's `hown` names `Zk.verify`), while a definition that applies them still reaches them. -/
def applies (env : Environment) (table : Std.HashMap Name (Array Name)) (meaning : Array Name) (e : Expr) :
    Array String := Id.run do
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
      stack := stack ++ meaningDeps env table ci
  return seen.toArray.qsort Name.lt |>.map toString

/-- One hypothesis of a statement's shape: where it came from (`binder`, a binder of the statement, or `field`, a `Prop`
field of a structure the package declares that a binder ranges over), its name, the binder's position (`at`), its type,
its head constant and that constant's module, and what it reaches (`reaches`). The head is its body's, under its leading
binders (`∀ j, P j` is headed by `P`); `guarded` says those binders include a hypothesis (`A → B`), and `applies` is what
its body applies (`applies`). A conjunction counts as each of its conjuncts. -/
partial def hypothesisJson (env : Environment) (table : Std.HashMap Name (Array Name)) (meaning : Array Name) (moduleOf : Name → Option Name) (source name : String)
    (pos : Nat) (t : Expr) (extra : Expr → List (String × Json) := fun _ => []) : MetaM (Array Json) := do
  let t ← instantiateMVars t
  if t.isAppOfArity ``And 2 then
    return (← hypothesisJson env table meaning moduleOf source s!"{name}.1" pos t.appFn!.appArg! extra) ++
      (← hypothesisJson env table meaning moduleOf source s!"{name}.2" pos t.appArg! extra)
  let (head, guarded, body) ← forallTelescope t fun ys b => do
    let guarded ← ys.anyM fun y => do isProp (← y.fvarId!.getType)
    let head : Option Name := match b.getAppFn with | .const c _ => some c | _ => none
    return (head, guarded, applies env table meaning b)
  return #[Json.mkObj ([("source", toJson source), ("name", toJson name), ("at", toJson pos),
    ("type", toJson (toString (← ppExpr t))),
    ("head", toJson (head.map toString)), ("module", toJson ((head.bind moduleOf).map toString)),
    ("reaches", toJson (reaches env table meaning t)), ("guarded", toJson guarded), ("applies", toJson body)] ++ extra t)]

/-- The positions of the binders `xs` that `e` mentions. -/
def usesOf (xs : Array Expr) (e : Expr) : Array Nat :=
  (Array.range xs.size).filter fun j => e.containsFVar xs[j]!.fvarId!

/-- `e` as a field path of one of the binders `xs`: `x.a.b` (`T.b (S.a x)`, structure projections applied to `x`) is
`x`'s position and the projections, innermost first; a bare binder has none. Anything else is none. -/
partial def fieldPath (env : Environment) (xs : Array Expr) (e : Expr) : Option (Nat × List String) :=
  match e.consumeMData with
  | .fvar f => (xs.findIdx? (·.fvarId! == f)).map (·, [])
  | .proj s i b => (fieldPath env xs b).map fun (j, p) => (j, p ++ [s!"{s}.{i}"])
  | e => match e.getAppFn with
    | .const c _ => match env.getProjectionFnInfo? c with
      | some info => if e.getAppNumArgs == info.numParams + 1 then
          (fieldPath env xs e.appArg!).map fun (j, p) => (j, p ++ [c.toString]) else none
      | none => none
    | _ => none

/-- `shares` for a hypothesis `x.f = y.f`: one field path (`fieldPath`, not empty) of two different binders, by their
positions (`inputs`) and the path; nothing for any other hypothesis. -/
def sharesOf (env : Environment) (xs : Array Expr) (t : Expr) : List (String × Json) :=
  if !t.isAppOfArity ``Eq 3 then [] else
  match fieldPath env xs t.appFn!.appArg!, fieldPath env xs t.appArg! with
  | some (j, p), some (k, q) =>
    if j != k && p == q && !p.isEmpty then
      [("shares", Json.mkObj [("inputs", toJson #[j, k]), ("path", toJson p)])] else []
  | _, _ => []

partial def conjuncts (e : Expr) : List Expr :=
  if e.isAppOfArity ``And 2 then conjuncts e.appFn!.appArg! ++ conjuncts e.appArg! else [e]

/-- The events of a conclusion: each argument `fun o => A₁ ∧ … ∧ Aₙ` of a constant `c` (`prob (fun o => …) g σ`) whose
body is a proposition, under any binders of the conclusion (`∃ h, … ∨ prob …`), with `c` (`head`) and, for each
conjunct `Aᵢ` of the body (`conjuncts`, its `∧`s flattened), what it applies (`applies`) and whether it holds under a
hypothesis of its own (`guarded`). -/
partial def eventsOf (env : Environment) (table : Std.HashMap Name (Array Name)) (meaning : Array Name) (e : Expr) : MetaM (Array Json) := do
  match e with
  | .mdata _ b => eventsOf env table meaning b
  | .proj _ _ b => eventsOf env table meaning b
  | .lam n t b bi | .forallE n t b bi =>
    withLocalDecl n bi t fun x => eventsOf env table meaning (b.instantiate1 x)
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
            return Json.mkObj [("applies", toJson (applies env table meaning body)), ("guarded", toJson guarded)]
          return #[Json.mkObj [("head", toJson c.toString), ("conjuncts", toJson cs)]])
    for a in args do
      out := out ++ (← eventsOf env table meaning a)
    return out
  | _ => return #[]

/-- One part of a statement's shape, for the audit's shape rule: its binders (`inputs`: those that aren't propositions,
each with its position and its type's head), its hypotheses (`hypothesisJson`: each `Prop` binder, and each `Prop` field
of a structure an input ranges over), what its conclusion reaches, and its events (`eventsOf`). Each binder, and the
conclusion, gives the positions of the binders it mentions (`uses`, `usesOf`), so that `check.py` can reorder them, and
a hypothesis equating one field of two binders says so (`shares`, `sharesOf`). -/
def shapeOf (env : Environment) (table : Std.HashMap Name (Array Name)) (meaning : Array Name) (moduleOf : Name → Option Name) (stmt : Expr) : MetaM Json :=
  forallTelescope stmt fun xs concl => do
    let mut inputs := #[]
    let mut hyps := #[]
    for h : i in [0:xs.size] do
      let d ← xs[i].fvarId!.getDecl
      let t ← instantiateMVars d.type
      let uses := toJson (usesOf xs t)
      if ← isProp t then
        hyps := hyps ++ ((← hypothesisJson env table meaning moduleOf "binder" d.userName.toString i t (sharesOf env xs)).map
          (·.setObjVal! "uses" uses))
        continue
      let head : Option Name := match t.getAppFn with | .const c _ => some c | _ => none
      inputs := inputs.push <| Json.mkObj [("name", toJson d.userName.toString), ("at", toJson i),
        ("head", toJson (head.map toString)), ("uses", uses)]
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
              out := out ++ (← hypothesisJson env table meaning moduleOf "field" s!"{d.userName}.{fd.userName}" i fd.type)
          return out)
    return Json.mkObj [("inputs", toJson inputs), ("hypotheses", toJson hyps),
      ("conclusion", toJson (reaches env table meaning concl)), ("events", toJson (← eventsOf env table meaning concl)),
      ("uses", toJson (usesOf xs concl))]

/-- The statement's shape, one part per conjunct of a statement that is a conjunction (`shapeOf` each), after a lifted
definition is unfolded once (`liftOf`) and its `let`s are inlined. -/
def shapeInfo (env : Environment) (table : Std.HashMap Name (Array Name)) (meaning : Array Name) (moduleOf : Name → Option Name) (stmt : Expr) : MetaM Json := do
  try
    let stmt ← zetaReduce (((liftOf env meaning stmt).map (·.2)).getD stmt)
    return toJson (← (conjuncts stmt).mapM (shapeOf env table meaning moduleOf))
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

/-- Whether a constant's result, past its binders, is one of them, as a `match`'s, a recursor's or `ite`'s is (its motive,
`α`): what it returns is one of its arguments, its branches. -/
def branching (c : Name) : MetaM Bool := do
  let some ci := (← getEnv).find? c | return false
  forallTelescope ci.type fun xs r => return xs.contains r.getAppFn

/-- Whether `c` is a constant of an assumptions module. -/
def assumption (moduleOf : Name → Option Name) (c : Name) : Bool :=
  ((moduleOf c).map (·.components.contains `Assumptions)).getD false

/-- The heads (`premiseHead`) of the parts a hypothesis packs in `∧` and `∃`, past its binders: `(∃ x, A x) ∧ B` has
`A` and `B`, and so does an abbrev outside an assumptions module that unfolds to it (`abbrev P x := A x ∧ B x`), at most
`n` times; one that applies a `branching` constant (a `match`) has those of its branches. -/
partial def packedHeads (moduleOf : Name → Option Name) (t : Expr) (n : Nat := 4) : MetaM (Array Name) :=
  forallTelescope t fun _ b => do
    if b.isAppOfArity ``And 2 then
      return (← packedHeads moduleOf b.appFn!.appArg! n) ++ (← packedHeads moduleOf b.appArg! n)
    if b.isAppOfArity ``Exists 2 then
      return ← withLocalDeclD `x b.appFn!.appArg! fun x => packedHeads moduleOf (b.appArg!.beta #[x]) n
    if let .const c _ := b.getAppFn then
      if ← branching c then
        return ← b.getAppArgs.foldlM (init := #[]) fun out a => do
          unless a.isLambda do return out
          lambdaTelescope a fun _ body => do
            return out ++ (← if ← isProp body then packedHeads moduleOf body n else pure #[])
      if !assumption moduleOf c && n > 0 then
        let u ← whnfR b
        if u != b && (u.isAppOfArity ``And 2 || u.isAppOfArity ``Exists 2 || u.isForall) then
          return ← packedHeads moduleOf u (n - 1)
    return (← premiseHead moduleOf 4 b).toArray

/-- `e` with its `let`s inlined, its metadata dropped and its β-redexes reduced: `forallTelescope` stops at each of them,
and a premise past one would go unrecorded. -/
def plain (e : Expr) : MetaM Expr := do
  transform (← zetaReduce e) (pre := fun e => return if let .mdata _ b := e then .visit b else .continue)
    (post := fun e => return if e.isHeadBetaTarget then .visit e.headBeta else .done e)

/-- The assumptions a premise's type mentions: the constants of an assumptions module, or of a module under `premises`
(the lock's `shape.premises`), that are `Prop`s or predicates, wherever they stand in it (`A ∧ B`, `A ↔ B`, `¬ A`,
`∃ x, A x`, `id A`), their types unfolded (`A : Pred`). A premises module's abbrev stands for the ones its value
mentions, as `premiseHead` sees through it, and for itself when its value mentions none. -/
partial def mentioned (env : Environment) (premises : Array Name) (moduleOf : Name → Option Name) (t : Expr) :
    MetaM (Array Name) :=
  t.getUsedConstants.foldlM (init := #[]) fun out c => do
    let some m := moduleOf c | return out
    let assumed := m.components.contains `Assumptions
    unless assumed || under premises m do return out
    let some ci := env.find? c | return out
    let predicate ← forallTelescopeReducing ci.type fun _ b => return b.isProp
    unless predicate do return out
    if let .defnInfo dv := ci then
      if !assumed && (dv.hints matches .abbrev) then
        let inner ← mentioned env premises moduleOf dv.value
        return out ++ (if inner.isEmpty then #[c] else inner)
    return out.push c

/-- The assumptions the fields of the structure a binder ranges over mention (`s : {k // A k}`, `g : G` with a field
`h : A g.k`), at every index of a family (`Fin n → G`), and those of the structures under `meaning` its fields range
over, transitively. -/
partial def fieldPremises (env : Environment) (meaning premises : Array Name) (moduleOf : Name → Option Name)
    (top : Bool) (t : Expr) : MetaM (Array Name) :=
  forallTelescope t fun _ cod => do
    let cod ← whnfR cod
    let .const s us := cod.getAppFn | return #[]
    unless top || ((moduleOf s).map (under meaning ·)).getD false do return #[]
    let some (.inductInfo iv) := env.find? s | return #[]
    let [ctor] := iv.ctors | return #[]
    if iv.numIndices != 0 || iv.isRec then return #[]
    let some (.ctorInfo cv) := env.find? ctor | return #[]
    let ctorType ← instantiateForall (cv.type.instantiateLevelParams cv.levelParams us)
      (cod.getAppArgs.extract 0 cv.numParams)
    forallTelescope ctorType fun fs _ => fs.foldlM (init := #[]) fun out f => do
      let ft ← inferType f
      let inner ← if ← isProp ft then pure #[] else fieldPremises env meaning premises moduleOf false ft
      return out ++ (← mentioned env premises moduleOf ft) ++ inner

/-- What the proposition `e` asserts: its parts through `∧`, the bodies of `∀` and `→`, `let`, and reducible unfolding
outside an assumptions module (as `premiseHead` sees through an abbrev, at most `n` times); `True` asserts nothing.
Each part is its head constant, with whether it applies that constant to a proposition or a predicate, or no head for
a part that applies a bound variable. What `e` takes rather than asserts (an antecedent, the type a `∀` ranges over)
and a part's arguments add, as parts, the assumptions they mention (`mentioned`, `fieldPremises`). -/
partial def parts (env : Environment) (meaning premises : Array Name) (moduleOf : Name → Option Name) (n : Nat)
    (e : Expr) : MetaM (Array (Option Name × Bool)) := do
  let e := e.consumeMData.headBeta
  let named (cs : Array Name) := cs.map fun c => (some c, false)
  match e with
  | .forallE x t b bi =>
    let taken ← if ← isProp t then mentioned env premises moduleOf t
      else fieldPremises env meaning premises moduleOf true t
    withLocalDecl x bi t fun y => return named taken ++ (← parts env meaning premises moduleOf n (b.instantiate1 y))
  | .letE _ _ v b _ => parts env meaning premises moduleOf n (b.instantiate1 v)
  | _ =>
    if e.isAppOfArity ``And 2 then
      return (← parts env meaning premises moduleOf n e.appFn!.appArg!) ++ (← parts env meaning premises moduleOf n e.appArg!)
    if e.isConstOf ``True then return #[]
    let .const c _ := e.getAppFn | return #[(none, false)]
    if !assumption moduleOf c && n > 0 then
      let u ← whnfR e
      if u != e then return ← parts env meaning premises moduleOf (n - 1) u
    let args := e.getAppArgs
    let props ← args.anyM fun a => do forallTelescopeReducing (← inferType a) fun _ r => return r.isProp
    let taken ← args.foldlM (init := #[]) fun out a => return out ++ (← mentioned env premises moduleOf a)
    return #[(some c, props)] ++ named taken

/-- The parts (`parts`) that the predicate `c`'s value asserts past its parameters, or none when `c` is a constant of
an assumptions module, no definition, a `branching` one, or not a predicate. -/
def assertedParts (env : Environment) (meaning premises : Array Name) (moduleOf : Name → Option Name) (c : Name) :
    MetaM (Option (Array (Option Name × Bool))) := do
  if assumption moduleOf c then return none
  let some (.defnInfo v) := env.find? c | return none
  if ← branching c then return none
  unless ← forallTelescopeReducing v.type fun _ r => return r.isProp do return none
  lambdaTelescope (← plain v.value) fun _ b => return some (← parts env meaning premises moduleOf 4 b)

/-- For a premise headed by `h`, a definition of a module this repository builds (`built`), outside an assumptions module,
that the lock doesn't list (`listed`): the parts it asserts (`assertedParts`), and those of each such definition among
them, breadth first, at most 64 definitions; `{definition: [{head, module, props}]}`. Null for any other head. -/
def unfoldsOf (env : Environment) (meaning premises : Array Name) (listed built : NameSet)
    (moduleOf : Name → Option Name) (h : Name) : MetaM Json := do
  let mine (c : Name) : Bool := !listed.contains c && ((moduleOf c).map built.contains).getD false
  if !mine h then return Json.null
  let some ps ← assertedParts env meaning premises moduleOf h | return Json.null
  let part (p : Option Name × Bool) : Json := Json.mkObj [("head", toJson (p.1.map toString)),
    ("module", toJson ((p.1.bind moduleOf).map toString)), ("props", toJson p.2)]
  let mut out := #[(h.toString, toJson (ps.map part))]
  let mut seen : NameSet := ({} : NameSet).insert h
  let mut queue := ps.filterMap (·.1)
  let mut i := 0
  while h : i < queue.size do
    let c := queue[i]
    i := i + 1
    if seen.contains c || !mine c then continue
    seen := seen.insert c
    if out.size ≥ 64 then break
    if let some qs ← assertedParts env meaning premises moduleOf c then
      out := out.push (c.toString, toJson (qs.map part))
      queue := queue ++ qs.filterMap (·.1)
  return Json.mkObj out.toList

/-- For `e` applying a `Guarantees` definition or another definition under `meaning` outside an assumptions module,
whether it is a `Guarantees` one and its value applied to `e`'s arguments. -/
def unfoldSpec? (env : Environment) (meaning : Array Name) (moduleOf : Name → Option Name) (e : Expr) :
    Option (Bool × Expr) := do
  let .const d us := e.getAppFn | none
  let .defnInfo dv ← env.find? d | none
  let m ← moduleOf d
  let statement := d.components.contains `Guarantees
  guard (statement || (under meaning m && !m.components.contains `Assumptions))
  return (statement, (dv.value.instantiateLevelParams dv.levelParams us).beta e.getAppArgs)

mutual
/-- Whether the constant `c` states its argument `i`, a predicate: whether that argument stands as a proposition
(`states`) in `c`'s value or in a `Prop` field of one of its constructors (`List.Pairwise R` takes `∀ a' ∈ l, R a a'`).
A constant that takes it as data does not: a measure takes its event (`prob p`, `if p x then 1 else 0`, `{x | p x} ∈ f`),
and so does a constant whose value isn't a proposition. A `branching` constant does, and so does one the walk can't see
into (no value, the argument an index, `fuel` spent). -/
partial def asserts (fuel : Nat) (c : Name) (i : Nat) : MetaM Bool := do
  let some ci := (← getEnv).find? c | return true
  if fuel == 0 || (← branching c) then return true
  unless ← withReducible (forallTelescopeReducing ci.type fun _ r => return r.isProp) do return false
  match ci with
  | .defnInfo v => lambdaTelescope v.value fun xs b => do
    let some p := xs[i]? | return true
    if ← isProp b then states (fuel - 1) c p b else return true
  | .inductInfo v =>
    if i ≥ v.numParams then return true
    v.ctors.anyM fun k => do
      let some kc := (← getEnv).find? k | return true
      forallTelescope kc.type fun xs _ => do
        let some p := xs[i]? | return true
        (xs.extract v.numParams xs.size).anyM fun f => do
          let t ← inferType f
          return (← isProp t) && (← states (fuel - 1) c p t)
  | _ => return true

/-- Whether the predicate `p` stands as a proposition in the proposition `e` (`asserts`, for `c`'s value): applied, under
`∀`, `→` and `let`, or passed (itself, in a `fun`, or in a proposition) to a constant that states that argument. `c`
applied again (a recursive definition, an inductive type's constructor) decides nothing. -/
partial def states (fuel : Nat) (c : Name) (p e : Expr) : MetaM Bool := do
  match e with
  | .mdata _ b => states fuel c p b
  | .forallE n t b bi =>
    if (← isProp t) && (← states fuel c p t) then return true
    withLocalDecl n bi t fun x => states fuel c p (b.instantiate1 x)
  | .letE _ _ v b _ => states fuel c p (b.instantiate1 v)
  | .app .. =>
    let f := e.getAppFn
    if f == p then return true
    let args := e.getAppArgs
    for h : j in [0:args.size] do
      let a := args[j]
      unless a.containsFVar p.fvarId! do continue
      if let .const g _ := f then
        if g == c then continue
        unless ← asserts fuel g j do continue
      if a == p then return true
      let stated ← if a.isLambda then lambdaTelescope a fun _ b => do return (← isProp b) && (← states fuel c p b)
        else if ← isProp a then states fuel c p a
        else withReducible <| forallTelescopeReducing (← inferType a) fun ys t => do
          return t.isProp && !ys.isEmpty && (← states fuel c p (mkAppN a ys))
      if stated then return true
    return false
  | _ => return e == p
end

/-- The premises a statement takes (`plain`): the `Prop` hypotheses of each conjunct, each named by its head after
reducible unfolding (so a `Premises` abbrev names the assumption it stands for), by the heads of the parts it packs in `∧`
and `∃` (`packedHeads`) and by the assumptions it mentions, and the assumptions the fields of a structure a binder ranges
over mention (`fieldPremises`); also those under the statement definitions (a `Guarantees` namespace's) a conclusion
applies. In each conjunct, disjunct, `if` branch and `∃` body of a conclusion, under any other definition under `meaning`
a conclusion applies (outside an assumptions module, `unfoldSpec?`), and in each predicate a conclusion passes to a
constant it doesn't unfold that states it (`asserts`: a `match`'s branches, `List.Pairwise (fun a b => …)`, a recursive
definition's step), only the assumptions and the constants of a module under `premises` (`shape.premises`) its
hypotheses mention (`only`): its other hypotheses (an acceptance event, a target) are what the conclusion means, not
premises. A predicate that constant takes as data, an event it measures (`Likely (fun o => A o → bad o)` with `Likely E`
holding when `prob E ≤ ε`), is what the conclusion means too: `A` there is part of the event, not something the
statement rests on. -/
partial def premisesOf (env : Environment) (meaning premises : Array Name) (moduleOf : Name → Option Name) (only : Bool)
    (e : Expr) : MetaM (Array Json) := do
  let mut out := #[]
  for c in conjuncts e do
    out := out ++ (← forallTelescope c fun xs body => do
      let mut ps := #[]
      for x in xs do
        let t ← inferType x
        let hs ← if !(← isProp t) then fieldPremises env meaning premises moduleOf true t
          else if only then mentioned env premises moduleOf t
          else do pure ((← packedHeads moduleOf t) ++ (← mentioned env premises moduleOf t))
        for h in hs do
          ps := ps.push <| Json.mkObj [("binder", toJson (← x.fvarId!.getDecl).userName.toString),
            ("head", toJson h.toString), ("module", toJson ((moduleOf h).map toString))]
      if body.isAppOfArity ``And 2 || body.isAppOfArity ``Or 2 || body.isAppOfArity ``ite 5 then
        for b in body.getAppArgs.extract (body.getAppNumArgs - 2) body.getAppNumArgs do
          ps := ps ++ (← premisesOf env meaning premises moduleOf true b)
      else if body.isAppOfArity ``Exists 2 then
        if let .lam n ty b bi := body.appArg! then
          ps := ps ++ (← withLocalDecl n bi ty fun y => premisesOf env meaning premises moduleOf true (b.instantiate1 y))
      else if let some (statement, v) := unfoldSpec? env meaning moduleOf body then
        ps := ps ++ (← premisesOf env meaning premises moduleOf (only || !statement) (← plain v))
      else
        let args := body.getAppArgs
        for h : j in [0:args.size] do
          let a := args[j]
          unless a.isLambda || (unfoldSpec? env meaning moduleOf a).isSome do continue
          if let .const c _ := body.getAppFn then
            unless ← (asserts 6 c j <|> pure true) do continue
          if a.isLambda then
            ps := ps ++ (← lambdaTelescope a fun _ b => do
              if ← isProp b then premisesOf env meaning premises moduleOf true b else return #[])
          else
            ps := ps ++ (← forallTelescopeReducing (← inferType a) fun ys t => do
              if t.isProp then premisesOf env meaning premises moduleOf true (mkAppN a ys) else return #[])
      return ps)
  return out

/-- The constant a statement applies past its `∀` and `∃` binders, and whether `¬` stands before it. -/
partial def bodyHead (t : Expr) : MetaM (Option Name × Bool) :=
  forallTelescope t fun _ b => do
    let b := b.consumeMData
    if b.isAppOfArity ``Exists 2 then
      match b.appArg!.consumeMData with
      | .lam n ty body bi => withLocalDecl n bi ty fun x => bodyHead (body.instantiate1 x)
      | _ => return (none, false)
    else if b.isAppOfArity ``Not 1 then
      return (b.appArg!.consumeMData.getAppFn.constName?, true)
    else
      return (b.getAppFn.constName?, false)

/-- A statement's conclusion past its `∀`, `→` and `∃` binders, split at its `∧`s, each conjunct again past its own: for
each conjunct its head constant (`head`), the definitions declared under `meaning` it applies itself (`defs`, followed no
further), whether it is a reflexivity (`a = a`, `a ↔ a`, `HEq a a`: `refl`) and whether it is a hypothesis in scope, the
hypotheses split at `∧` (`assumed`); and whether a hypothesis is `False` (`absurd`). Read as written: nothing is
unfolded. -/
partial def conclusionOf (env : Environment) (meaning : Array Name) (t : Expr) (hyps : Array Expr := #[]) :
    MetaM (Array Json × Bool) :=
  forallTelescope t fun xs b => do
    let mut hs := hyps
    for x in xs do
      let ty ← inferType x
      if ← isProp ty then hs := hs ++ (conjuncts ty.consumeMData).toArray.map (·.consumeMData)
    let b := b.consumeMData
    if b.isAppOfArity ``And 2 then
      let (l, a) ← conclusionOf env meaning b.appFn!.appArg! hs
      let (r, a') ← conclusionOf env meaning b.appArg! hs
      return (l ++ r, a || a')
    if b.isAppOfArity ``Exists 2 then
      if let .lam n ty body bi := b.appArg!.consumeMData then
        return ← withLocalDecl n bi ty fun x => conclusionOf env meaning (body.instantiate1 x) hs
    let defs := b.getUsedConstants.filter fun c => match env.find? c, env.getModuleIdxFor? c with
      | some (.defnInfo _), some i | some (.opaqueInfo _), some i => under meaning (moduleName env i)
      | _, _ => false
    let refl := ((b.isAppOfArity ``Eq 3 || b.isAppOfArity ``Iff 2) && b.appFn!.appArg! == b.appArg!) ||
      (b.isAppOfArity ``HEq 4 && b.getArg! 1 == b.getArg! 3)
    return (#[Json.mkObj [("head", toJson (b.getAppFn.constName?.map toString)),
      ("defs", toJson ((defs.qsort Name.lt).map toString)), ("refl", toJson refl), ("assumed", toJson (hs.contains b))]],
      hs.any (·.isConstOf ``False))

def pinInfo (env : Environment) (table : Std.HashMap Name (Array Name)) (meaning premises : Array Name)
    (listed built : NameSet) (axioms : Std.HashMap Name NameSet)
    (ranges : Std.HashMap Name DeclarationRanges) (rs : Array (Name × Name × ConstantInfo) × Array (Name × Name)) (n : Name) :
    MetaM Json := do
  let some ci := env.find? n | return Json.mkObj [("kind", "missing")]
  let moduleOf (c : Name) : Option Name := (env.getModuleIdxFor? c).map fun i => moduleName env i
  let signature := toString (← PrettyPrinter.ppSignature n).fmt
  let hypotheses ← forallTelescope (← plain ci.type) fun xs _ => do
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
    let own := (meaningDeps env table ci).foldl (fun (s : NameSet) d => if d != c && fam.isPrefixOf d then s.insert d else s) {}
    Json.mkObj [("name", toJson c.toString), ("module", toJson m.toString), ("kind", toJson (kind ci)),
      ("hash", toJson s!"{hex ci.type.hash}{hex (valueHash ci)}"),
      ("own", toJson (own.toArray.qsort Name.lt |>.map toString)),
      ("lines", toJson (ranges[c]?.map fun (d : DeclarationRanges) => #[d.range.pos.line, d.range.endPos.line]))]
  return Json.mkObj [("kind", toJson (kind ci)), ("module", toJson ((moduleOf n).map toString)),
    ("at", toJson (ranges[n]?.map fun (d : DeclarationRanges) => #[d.range.pos.line, d.range.pos.column])),
    ("signature", toJson signature), ("axioms", toJson ((axioms.getD n {}).toArray.qsort Name.lt |>.map toString)),
    ("hypotheses", toJson hypotheses), ("shape", ← shapeInfo env table meaning moduleOf ci.type),
    ("premises", ← (do
      let ps ← premisesOf env meaning premises moduleOf false (← plain ci.type)
      return toJson (← ps.mapM fun p => do
        let some h := (p.getObjValAs? String "head").toOption | return p
        return p.setObjVal! "unfolds" (← unfoldsOf env meaning premises listed built moduleOf h.toName))) <|> pure Json.null),
    ("body", ← (do
      let (h, neg) ← bodyHead ci.type
      let (cs, absurd) ← conclusionOf env meaning ci.type
      return Json.mkObj [("head", toJson (h.map toString)), ("negated", toJson neg), ("conclusion", toJson cs),
        ("absurd", toJson absurd)]) <|> pure Json.null),
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
/-- The verifier's path: the constants its `entries` reach through definitions' values, `implemented_by` targets and
`@[csimp]` replacements (the constant compiled code runs in place of one, by a theorem `@f = @g`), in every imported
module, expanding nothing in `stop`. Reported: each reached `@[implemented_by]` and `@[extern]` constant, each reached
`@[csimp]` rewrite with its replacement and theorem, and each constant of the `hash` namespace the walk enters from
outside it, with the path from an entry. -/
unsafe def verifierInfo (env : Environment) (entries stop : Array Name) (hash : Name) : Json := Id.run do
  let moduleOf (c : Name) : Option Name := (env.getModuleIdxFor? c).map fun i => moduleName env i
  let mut impl : Std.HashMap Name Name := {}
  let mut ext : NameSet := {}
  let mut csimp : Std.HashMap Name Compiler.CSimp.Entry := {}
  for data in env.header.moduleData do
    for (n, t) in entriesOf Name data Compiler.implementedByAttr.ext.name do
      impl := impl.insert n t
    for (n, _) in entriesOf Unit data externAttr.ext.name do
      ext := ext.insert n
    if let some (_, es) := data.entries.find? (·.1 == Compiler.CSimp.ext.ext.name) then
      for e in es do
        match (unsafeCast e : ScopedEnvExtension.Entry Compiler.CSimp.Entry) with
        | .global c | .scoped _ c => csimp := csimp.insert c.fromDeclName c
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
      | _ => #[]) ++ (impl[n]?.toArray) ++ ((csimp[n]?.map (·.toDeclName)).toArray)
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
    ("csimp", toJson (reached.filterMap fun c => csimp[c]?.map fun e =>
      Json.mkObj (locOf c ++ [("to", toJson e.toDeclName.toString), ("by", toJson e.thmName.toString)]))),
    ("hash", toJson (reached.filter (fun c => hash.isPrefixOf c && !(parent[c]?.map hash.isPrefixOf).getD true)
      |>.map fun c => Json.mkObj (locOf c ++ [("path", toJson (pathTo c))])))]

open LeanAudit in
unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let some cfgPath := args.head? | throw <| IO.userError "usage: Facts.lean CONFIG"
  let cfg ← IO.ofExcept (Json.parse (← IO.FS.readFile cfgPath))
  let roots := names cfg "roots"
  let meaning := names cfg "meaning"
  let premises := names cfg "premises"
  let listed := (names cfg "assumptions").foldl NameSet.insert {}
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
  let t0 ← IO.monoMsNow
  let lap (what : String) (n : Nat) : IO Unit := do
    IO.eprintln s!"facts: {what} ({n}) at {(← IO.monoMsNow) - t0} ms"
  let env ← importModules (roots.map fun r => { module := r }) {} (loadExts := false)
  let mnames := env.header.moduleNames
  lap "import" mnames.size
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
  lap "modules" decls.size
  let table := usesTable env
  lap "uses" table.size
  let axioms := axiomsOf env table (decls ++ pins)
  lap "axioms" axioms.size
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
          ("kind", toJson (kind ci)), ("statement", toJson ci.type.isProp),
          ("predicate", toJson (ci.type.getForallBody matches .sort .zero))]
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
      if (usesIn env table n).contains ``sorryAx && unreported ci then
        unreportedErrors := unreportedErrors.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString)]
      let why := (if ci matches .opaqueInfo _ then #["opaque"] else #[]) ++ (if ci.isUnsafe then #["unsafe"] else #[]) ++
        (if impl.contains n then #["implemented_by"] else #[]) ++ (if ext.contains n then #["extern"] else #[])
      if !why.isEmpty then
        escapes := escapes.push <| Json.mkObj [("name", toJson n.toString), ("module", toJson m.toString), ("why", toJson why)]
  lap "declarations" offenders.size
  let opts : Options := (({} : Options).set `pp.proofs false).set `format.width (100 : Nat)
  let ctx : Core.Context := { fileName := "<audit>", fileMap := default, maxHeartbeats := 0, options := opts }
  let readsOf : Std.HashMap Name (Array (Name × Name × ConstantInfo) × Array (Name × Name)) := Id.run do
    let tasks := pins.map fun p => Task.spawn fun _ => (env.find? p).map fun ci => reads env table meaning ci.type.getUsedConstants
    let mut acc := {}
    for h : i in [0:pins.size] do
      if let some rs := tasks[i]!.get then
        acc := acc.insert pins[i] rs
    return acc
  lap "reads" readsOf.size
  let found ← (List.range ((mnames.size + 255) / 256)).toArray.mapM fun k => IO.asTask do
    (mnames.extract (256 * k) (256 * k + 256)).mapM fun m => return toString (← findOLean m)
  let mut oleans := #[]
  for t in found do
    oleans := oleans ++ (← IO.ofExcept t.get)
  -- the modules a package of this repository builds (an `.olean` in its own `.lake/build`), as `check.py`'s `own_modules`
  let built : NameSet := (List.range mnames.size).foldl (init := {}) fun s i =>
    let o := oleans[i]!
    if (o.splitOn "/.lake/build/").length > 1 && (o.splitOn "/.lake/packages/").length == 1 then s.insert mnames[i]! else s
  -- each pin's facts are a function of the environment and the maps above, so each pin is a task, joined in order
  let tasks ← pins.mapM fun p => IO.asTask do
    let (out, _) ← (MetaM.run' do
      return (p.toString, ← pinInfo env table meaning premises listed built axioms ranges (readsOf.getD p (#[], #[])) p)).toIO
        ctx { env }
    return out
  let pinned ← tasks.mapM fun t => IO.ofExcept t.get
  lap "pins" pinned.size
  let field (j : Json) (k : String) : Array Json := (j.getObjValD k).getArr?.toOption.getD #[]
  let nameAt (j : Json) (k : String) : Option Name := (j.getObjValD k).getStr?.toOption.map String.toName
  let candidates := names cfg "assumptions" ++ pinned.foldl (init := #[]) fun acc (_, j) =>
    acc ++ (field j "premises").filterMap (nameAt · "head") ++ (field j "hypotheses").filterMap (nameAt · "named")
  let assumed := (candidates.foldl (init := (({} : NameSet), #[])) fun (s, out) a =>
    if s.contains a then (s, out) else (s.insert a, out.push a)).2
  let inScope (m : Name) : Bool := under meaning m || built.contains m
  let walks : Array (Name × Array (Name × Name × ConstantInfo)) := Id.run do
    let tasks := assumed.map fun a => Task.spawn fun _ => (readsWhere env table inScope #[a]).1
    let mut acc := #[]
    for h : i in [0:assumed.size] do
      let rs := tasks[i]!.get
      if rs.any (·.1 == assumed[i]) then
        acc := acc.push (assumed[i], rs)
    return acc
  lap "assumptions" walks.size
  let mut conflicts := #[]
  if let some path := (cfg.getObjValD "canon").getStr?.toOption then
    -- each line's text is computed in a task (`Task.spawn` takes the closure, so the text is not built here) and the
    -- lines are written in this order
    let mut lines : Array (Task String) := #[]
    let mut done : NameSet := {}
    let mut named : NameSet := {}
    for p in pins do
      let some ci := env.find? p | continue
      lines := lines.push <| Task.spawn fun _ =>
        (Json.mkObj [("id", toJson s!"pin:{p}"), ("canon", toJson (canonText ci.type))]).compress
      let head := liftOf env meaning ci.type
      if let some (g, s) := head then
        let m := (env.getModuleIdxFor? g).map fun i => (moduleName env i).toString
        lines := lines.push <| Task.spawn fun _ =>
          (Json.mkObj [("id", toJson s!"lift:{p}"), ("head", toJson g.toString), ("module", toJson m),
            ("canon", toJson (canonText s))]).compress
      if !back.isEmpty then
        named := (ci.type.getUsedConstants ++ ((head.map (·.2.getUsedConstants)).getD #[])).foldl (·.insert ·) (named.insert p)
        lines := lines.push <| Task.spawn fun _ =>
          let unfolded := match head with
            | some (g, s) => [("head", toJson g.toString), ("unfolded", toJson (canonText s rn))]
            | none => []
          (Json.mkObj ([("id", toJson s!"moved-pin:{p}"), ("as", toJson (rn p).toString),
            ("canon", toJson (canonText ci.type rn))] ++ unfolded)).compress
      for (c, _, rci) in (readsOf.getD p (#[], #[])).1 do
        if done.contains c then continue
        done := done.insert c
        lines := lines.push <| Task.spawn fun _ =>
          (Json.mkObj [("id", toJson s!"read:{c}"), ("canon", toJson (readCanon rci))]).compress
        if !back.isEmpty then
          named := (uses rci).foldl (·.insert ·) (named.insert c)
          lines := lines.push <| Task.spawn fun _ =>
            (Json.mkObj [("id", toJson s!"moved-read:{c}"), ("as", toJson (rn c).toString),
              ("canon", toJson (readCanon rci rn))]).compress
    for (_, rs) in walks do
      for (c, _, rci) in rs do
        if done.contains c then continue
        done := done.insert c
        lines := lines.push <| Task.spawn fun _ =>
          (Json.mkObj [("id", toJson s!"read:{c}"), ("canon", toJson (readCanon rci))]).compress
    let h ← IO.FS.Handle.mk path .write
    for l in lines do
      h.putStrLn l.get
    let (kept, renamed) := (named.toArray.qsort Name.lt).partition fun c => rn c == c
    let mut took : Std.HashMap Name Name := kept.foldl (fun m c => m.insert c c) {}
    for c in renamed do
      if let some d := took[rn c]? then
        conflicts := conflicts.push <| Json.mkObj [("name", toJson c.toString), ("as", toJson (rn c).toString),
          ("with", toJson d.toString)]
      else
        took := took.insert (rn c) c
  lap "canon" conflicts.size
  let mut graph := #[]
  for h : i in [0:mnames.size] do
    let olean := oleans[i]!
    graph := graph.push (mnames[i].toString, Json.mkObj [("olean", toJson olean),
      ("imports", toJson (env.header.moduleData[i]!.imports.map (·.module.toString)))])
  let doc := Json.mkObj [("lean", toJson Lean.versionString), ("modules", toJson modules), ("imports", Json.mkObj imports.toList),
    ("spans", Json.mkObj spans.toList), ("oleans", toJson oleans), ("graph", Json.mkObj graph.toList),
    ("declarations", Json.mkObj (counts.toList.map fun (k, v) => (k, toJson v))),
    ("axioms", toJson (used.toArray.qsort Name.lt |>.map toString)), ("offenders", toJson offenders),
    ("unreported_errors", toJson unreportedErrors), ("declared_axioms", toJson declared), ("escapes", toJson escapes), ("spec_definitions", toJson specDefs),
    ("instances", toJson instances), ("pins", Json.mkObj pinned.toList),
    ("assumption_reads", Json.mkObj (walks.toList.map fun (a, rs) => (a.toString, toJson (rs.map fun (c, m, ci) =>
      Json.mkObj [("name", toJson c.toString), ("module", toJson m.toString), ("kind", toJson (kind ci))])))),
    ("moved_conflicts", toJson conflicts)]
  let vcfg := cfg.getObjValD "verifier"
  let doc := if (names vcfg "entries").isEmpty then doc else
    doc.setObjVal! "verifier" (verifierInfo env (names vcfg "entries") (names vcfg "stop")
      (((vcfg.getObjValD "hash").getStr?.toOption.getD "").toName))
  lap "graph" oleans.size
  let text := doc.compress
  lap "render" text.length
  IO.FS.writeFile out text
  lap "write" 0
  return 0
