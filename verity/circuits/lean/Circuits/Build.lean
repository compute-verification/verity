/-
`Flock.Program.Build`: a builder that writes no ports, the writer of record for no description that has ports (§12), as
`Prog.encode` writes none. It builds a Program's Definitions in Lean and `finish` writes them with `Prog.encode`, then
decodes those bytes with `decode`: the Program a builder registers is the decode of the bytes it wrote, so no round trip is
stated or owed. A reference is a value an earlier push returned in the same body, so no forward reference can be written,
and parts are made only by `normal`, `refs.concat`'s normal form.
-/
module

public import Circuits.Encode
meta import Circuits.Encode

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock.Program.Build

open Lean (Json)

/-- A body under construction: its parameters and the nodes pushed so far. -/
structure Scope where
  id : Nat
  params : Array (String × Ty) := #[]
  nodes : Array Node := #[]

structure Builder where
  defs : Array Defn := #[]
  index : Std.HashMap String Nat := {}
  boolean : Std.HashMap String String := {}
  scope : Scope := { id := 0 }
  serial : Nat := 0

abbrev M := StateT Builder (Except String)

/-- A collection of `ty` in the body `scope`: its references, as parts. -/
structure Val where
  scope : Nat
  ty : Ty
  parts : Array Part

def Val.count (v : Val) : Nat := v.parts.foldl (· + ·.count) 0

def defn (f : Nat) : M Defn := do
  match (← get).defs[f]? with
  | some d => pure d
  | none => throw s!"build: no Definition {f} (the builder holds {(← get).defs.size})"

/-- Refuses a value of another body than the one open. -/
def own (v : Val) : M Unit := do
  let s := (← get).scope.id
  if v.scope != s then throw s!"build: a value of body {v.scope} read in body {s}: a reference is an earlier push's value"

def bcast? : Part → Option (Bool × Nat × Int)
  | .affine p i b 1 => some (p, i, b)
  | .strided p i b _ 0 0 _ => some (p, i, b)
  | _ => none

/-- `refs.broadcast`: `c` references to one leaf. -/
def broadcast (p : Bool) (i : Nat) (b : Int) (c : Nat) : Part :=
  if c == 1 then .affine p i b.toNat 1 else .strided p i b 1 0 0 c

/-- `refs._merge_broadcasts`: adjacent repetitions of one leaf as one zero-stride part. -/
def mergeBroadcasts (ps : Array Part) : Array Part :=
  ps.foldl (fun (out : Array Part) p => match out.back?, bcast? p with
    | some q, some sp => if bcast? q == some sp then out.pop.push (broadcast sp.1 sp.2.1 sp.2.2 (q.count + p.count))
        else out.push p
    | _, _ => out.push p) #[]

/-- The parts of a run of contiguous affine parts of one collection as one affine part, else none. -/
def contiguous (ps : Array Part) : Option Part :=
  match (ps[0]? : Option Part) with
  | some (.affine p i b c) =>
    (ps.extract 1 ps.size).foldlM (fun (q : Part) x => match q, x with
      | .affine _ _ b0 n, .affine p' i' b' c' => if p' == p && i' == i && b' == b0 + n then some (.affine p i b0 (n + c'))
          else none
      | _, _ => none) (.affine p i b c)
  | _ => none

/-- The references of a part, one by one (`Part.get` over its count); a grid part has no such list here. -/
def refsOf : Part → Except String (Array Ref)
  | .affine p i b c => pure ((Array.range c).map fun k => ⟨p, i, b + k⟩)
  | .strided p i b inner os is c => pure ((Array.range c).map fun k =>
      ⟨p, i, (b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is).toNat⟩)
  | .explicit it => pure it
  | .grid _ _ => throw "build: a grid part is not gathered"

/-- `refs.concat`: no empty part, repetitions of one leaf merged, one part as itself, contiguous affine parts as one, at most
64 references all explicit as one explicit part, else the parts. -/
def normal (ps : Array Part) : Array Part :=
  let ps := mergeBroadcasts (ps.filter (·.count > 0))
  if ps.isEmpty then #[.explicit #[]] else if ps.size == 1 then ps else
  match contiguous ps with
  | some q => #[q]
  | none =>
    let total := ps.foldl (· + ·.count) 0
    let items := ps.foldl (fun acc q => match acc, q with
      | some a, .explicit it => some (a ++ it)
      | _, _ => none) (some #[])
    match items with
    | some it => if total ≤ 64 then #[.explicit it] else ps
    | none => ps

/-- `vs` as one collection of `ty` (`concat` of their references). -/
def concat (ty : Ty) (vs : Array Val) : M Val := do
  for v in vs do own v
  let n := vs.foldl (· + ·.count) 0
  if n != ty.leaves then throw s!"build: {n} references for {ty.render}, which has {ty.leaves} leaves"
  return ⟨(← get).scope.id, ty, normal (vs.foldl (· ++ ·.parts) #[])⟩

/-- `vs`' references one by one, as one explicit part of `ty` (a gather, as a traced Boolean body writes it). -/
def gather (ty : Ty) (vs : Array Val) : M Val := do
  for v in vs do own v
  let n := vs.foldl (· + ·.count) 0
  if n != ty.leaves then throw s!"build: {n} references for {ty.render}, which has {ty.leaves} leaves"
  let mut items := #[]
  for v in vs do
    for q in v.parts do items := items ++ (← StateT.lift (refsOf q))
  return ⟨(← get).scope.id, ty, #[.explicit items]⟩

/-- `v`'s references `lo … hi - 1` as `ty` (`Refs.slice`). -/
def Val.span (v : Val) (ty : Ty) (lo hi : Nat) : M Val := do
  own v
  if !(lo ≤ hi && hi ≤ v.count && hi - lo == ty.leaves) then
    throw s!"build: references {lo} … {hi} of {v.count} as {ty.render} ({ty.leaves} leaves)"
  let (_, out) := v.parts.foldl (fun (s, out) q =>
    let a := max lo s
    let b := min hi (s + q.count)
    (s + q.count, if a < b then out.push (Widths.sliceOf q (a - s) (b - s)) else out)) (0, (#[] : Array Part))
  return { v with ty, parts := normal out }

/-- Element `i` of an Array value. -/
def Val.elem (v : Val) (i : Nat) : M Val := do
  let .array n e := v.ty | throw s!"build: element {i} of {v.ty.render}, not an Array"
  if i ≥ n then throw s!"build: element {i} of {v.ty.render}"
  v.span e (i * e.leaves) ((i + 1) * e.leaves)

/-- `v`'s references read as `ty`, of as many leaves. -/
def Val.as (v : Val) (ty : Ty) : M Val := do
  if v.count != ty.leaves then throw s!"build: {v.ty.render} read as {ty.render}: {v.count} references, {ty.leaves} leaves"
  return { v with ty }

/-- `t` with an axis of length `n` inserted at `ax` (`Ty.removeAxis`'s inverse): a batch argument's type. -/
def insertAxis : Ty → Nat → Nat → Option Ty
  | t, 0, n => some (.array n t)
  | .array m e, ax + 1, n => (insertAxis e ax n).map (.array m ·)
  | _, _, _ => none

/-- Pushes node `x` with its arguments' declared types, and returns its output. -/
def push (form : Form) (f : Nat) (args : Array (Ty × Val)) (out : Ty) (axes : Array (Option Nat) := #[]) (n : Nat := 1)
    (nxs : Nat := 0) : M Val := do
  let d ← defn f
  for h : j in [0:args.size] do
    let (t, v) := args[j]
    own v
    if v.count != t.leaves then
      throw s!"build: argument {j} of {d.key}: {v.count} references, its type {t.render} has {t.leaves} leaves"
  let s := (← get).scope
  let node : Node := { form, fn := f, args := args.map fun (t, v) => { ty := t, seq := Seq.ofParts v.parts }, out, axes, n, nxs }
  modify fun B => { B with scope := { s with nodes := s.nodes.push node } }
  return ⟨s.id, out, #[.affine false s.nodes.size 0 out.leaves]⟩

/-- A node calling `f` once: a primitive node for a primitive, else a call. -/
def call (f : Nat) (args : Array Val) : M Val := do
  let d ← defn f
  if args.size != d.params.size then throw s!"build: {d.key} takes {d.params.size} arguments, not {args.size}"
  push (if d.prim then .prim else .call) f (d.params.zip args |>.map fun ((_, t), v) => (t, v)) d.ret

/-- A batch of `n` calls of `f`, each argument mapped over an axis or passed whole (`none`). -/
def batch (f : Nat) (n : Nat) (args : Array (Val × Option Nat)) : M Val := do
  let d ← defn f
  if args.size != d.params.size then throw s!"build: batch {d.key} takes {d.params.size} arguments, not {args.size}"
  let mut typed := #[]
  for ((pn, pt), (v, ax)) in d.params.zip args do
    match ax with
    | none => typed := typed.push (pt, v)
    | some a => match insertAxis pt a n with
      | some t => typed := typed.push (t, v)
      | none => throw s!"build: batch {d.key}: no axis {a} in parameter '{pn}' of {pt.render}"
  push .batch f typed (.array n d.ret) (args.map (·.2)) n

/-- A scan of `n` steps of `f` from `init`, reading `xs` one element a step and `shared` whole. -/
def scan (f : Nat) (n : Nat) (init : Val) (xs shared : Array Val) : M Val := do
  let d ← defn f
  let .tuple #[carry, y] := d.ret | throw s!"build: scan {d.key}: a step returns (carry, y), not {d.ret.render}"
  if d.params.size != 1 + xs.size + shared.size then
    throw s!"build: scan {d.key} takes {d.params.size} arguments, not {1 + xs.size + shared.size}"
  let ts := d.params.map (·.2)
  let typed := #[(carry, init)] ++ ((ts.extract 1 (1 + xs.size)).zip xs).map (fun (t, v) => (Ty.array n t, v))
    ++ (ts.extract (1 + xs.size) ts.size).zip shared
  push .scan f typed (.tuple #[carry, .array n y]) #[] n xs.size

private def add (d : Defn) : M Nat := do
  let B ← get
  set { B with defs := B.defs.push d, index := B.index.insert d.key B.defs.size }
  return B.defs.size

/-- Primitive `id`, added once by key. -/
def prim (id : String) (params : Array (String × Ty)) (ret : Ty) : M Nat := do
  if let some i := (← get).index[id]? then
    let d ← defn i
    if !(d.prim && d.params == params && d.ret == ret) then throw s!"build: '{id}' is already a Definition of another signature"
    return i
  add { key := id, id, prim := true, params, ret, gates := 1, paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := 1 }

/-- Composite `id` with its binding record (§4.3, in its statics' order), its body built by `body` from its parameters' values,
added once by key; the key is the id and the record as the format writes it, so the decoder's key check holds. -/
def composite (id : String) (bindings : Array (String × Json)) (params : Array (String × Ty)) (body : Array Val → M Val) :
    M Nat := do
  let rec_ ← bindings.mapM fun (k, v) => do return s!"{k}={← StateT.lift (canon v)}"
  let key := if bindings.isEmpty then id else id ++ "{" ++ ",".intercalate rec_.toList ++ "}"
  if let some i := (← get).index[key]? then return i
  for f in recordFunctionIds key do
    if !(← get).index.contains f then throw s!"build: {key}'s record names '{f}', no Definition of the builder"
  let outer := (← get).scope
  let sid := (← get).serial + 1
  modify fun B => { B with serial := sid, scope := { id := sid, params } }
  let vals := (Array.range params.size).zip params |>.map fun (i, _, t) => (⟨sid, t, #[.affine true i 0 t.leaves]⟩ : Val)
  let ret ← body vals
  own ret
  let nodes := (← get).scope.nodes
  modify fun B => { B with scope := outer }
  add { key, id, prim := false, params, ret := ret.ty, gates := 0, paramLeaves := 0, outLeaves := 0,
        body := { params, retType := ret.ty, nodes, ret := Seq.ofParts ret.parts, offsets := #[] } }

/-- Definition `key` of `src` and every Definition it reaches, through at most `fuel` nested ones, as `src` holds them. -/
def adoptIn (src : Prog) : Nat → String → M Nat
  | 0, key => throw s!"build: '{key}' nests deeper than adopt follows"
  | fuel + 1, key => do
    if let some i := (← get).index[key]? then return i
    let some d := src.index[key]? >>= (src.defs[·]?) | throw s!"build: the source Program has no Definition '{key}'"
    let mut nodes := #[]
    for x in d.body.nodes do
      let some c := src.defs[x.fn]? | throw s!"build: '{key}' calls Definition {x.fn}, which the source Program lacks"
      nodes := nodes.push { x with fn := ← adoptIn src fuel c.key }
    for f in recordFunctionIds d.key do discard (adoptIn src fuel f)
    add { d with body := { d.body with nodes } }

/-- Definition `key` of `src`, as a registry's: a primitive or a Definition the builder does not write itself. -/
def adopt (src : Prog) (key : String) : M Nat := adoptIn src (src.defs.size + 1) key

/-- §4.9: each word primitive of the builder that `src` maps to a Boolean Definition, mapped to it, adopted. -/
def adoptBooleans (src : Prog) : M Unit := do
  for (w, b) in src.boolean.toArray.qsort (·.1 < ·.1) do
    if (← get).index.contains w then
      discard (adopt src b)
      modify fun B => { B with boolean := B.boolean.insert w b }

/-- The Program whose root is `rootId`, its body built by `body`: `Prog.encode`'s bytes and their decode by `decode`, or the
decoder's refusal. -/
def finish (rootId programFn : String) (body : M Val) : Except String (ByteArray × Prog) := do
  let (root, B) ← (composite rootId #[] #[] fun _ => body).run {}
  let P : Prog := { defs := B.defs, index := B.index, root, programFn, sha512 := .empty, boolean := B.boolean }
  let bytes ← P.encode
  match decode bytes with
  | .ok Q => pure (bytes, Q)
  | .error e => throw s!"build: the decoder refuses the bytes written: {e}"

/-- Equality of two `2^k`-bit words on bits (`I32Eq_v2`'s form for `k = 5`): each bit's Xor, its Not, and an And tree that
pairs the even and the odd places at each level. -/
def eqBits (k : Nat) : M Val := do
  let bit := Ty.value 1
  let input ← prim "Input1_v1" #[] bit
  let xor ← prim "Xor2_v1" #[("x", .array 2 bit)] bit
  let not ← prim "Not_v1" #[("a", bit)] bit
  let and ← prim "And2_v1" #[("x", .array 2 bit)] bit
  let w := 2 ^ k
  let a ← batch input w #[]
  let b ← batch input w #[]
  let x ← batch xor w #[(← concat (.array 2 (.array w bit)) #[a, b], some 1)]
  let mut v ← batch not w #[(x, some 0)]
  for j in [0:k] do
    let m := 2 ^ (k - j - 1)
    let evens ← (Array.range m).mapM fun i => v.elem (2 * i)
    let odds ← (Array.range m).mapM fun i => v.elem (2 * i + 1)
    v ← batch and m #[(← gather (.array 2 (.array m bit)) (evens ++ odds), some 1)]
  gather bit #[← v.elem 0]

private def built (r : Except String (ByteArray × Prog)) : Bool :=
  match r with
  | .ok (b, Q) => (match Q.encode with | .ok b' => b' == b | .error _ => false) &&
      (match Q.digest with | .ok h => h == hexOf (Sha512.hash b) | .error _ => false)
  | .error _ => false

private def refused (r : Except String (ByteArray × Prog)) (what : String) : Bool :=
  match r with
  | .ok _ => false
  | .error e => (e.splitOn what).length > 1

private def EQ4 := finish "Root[Eq4_v1]_v1" "Eq4_v1" (eqBits 2)

/-- A composite with a binding record naming a function, a batch over an axis and a call. -/
private def PAIRS := finish "Root[Pairs_v1]_v1" "Pairs_v1" do
  let bit := Ty.value 1
  let input ← prim "Input1_v1" #[] bit
  let xor ← prim "Xor2_v1" #[("x", .array 2 bit)] bit
  let g ← composite "Pair_v1" #[("F", Json.mkObj [("fn", .str "Xor2_v1")]), ("N", Lean.toJson (2 : Nat))] #[("x", .array 2 bit)] fun ps => do
    let some x := ps[0]? | throw "no parameter"
    call xor #[x]
  let a ← batch input 8 #[]
  let y ← batch g 4 #[(← a.as (.array 4 (.array 2 bit)), some 0)]
  concat (.array 5 bit) #[y, ← call g #[← a.span (.array 2 bit) 2 4]]

/-- A scan: a carry and a bit each step, the carry passed on. -/
private def CARRY := finish "Root[Carry_v1]_v1" "Carry_v1" do
  let bit := Ty.value 1
  let input ← prim "Input1_v1" #[] bit
  let and ← prim "And2_v1" #[("x", .array 2 bit)] bit
  let xor ← prim "Xor2_v1" #[("x", .array 2 bit)] bit
  let step ← composite "Step_v1" #[] #[("c", bit), ("x", bit)] fun ps => do
    let (some c, some x) := (ps[0]?, ps[1]?) | throw "no parameter"
    let cx ← concat (.array 2 bit) #[c, x]
    concat (.tuple #[bit, bit]) #[← call and #[cx], ← call xor #[cx]]
  let c ← batch input 1 #[]
  let xs ← batch input 4 #[]
  scan step 4 (← c.as bit) #[xs] #[]

#guard built EQ4 && built PAIRS && built CARRY
#guard refused (finish "Root[Bad_v1]_v1" "Bad_v1" (call 7 #[])) "no Definition 7"
#guard refused (finish "Root[Bad_v1]_v1" "Bad_v1" do
    let x ← batch (← prim "Input16_v1" #[] (.value 16)) 2 #[]
    call (← prim "And2_v1" #[("x", .array 2 (.value 1))] (.value 1)) #[x]) "the decoder refuses"
#guard refused (finish "Root[Bad_v1]_v1" "Bad_v1" do
    let a ← batch (← prim "Input1_v1" #[] (.value 1)) 1 #[]
    let leak ← composite "Leak_v1" #[] #[] fun _ => pure a
    call leak #[]) "a reference is an earlier push's value"

end Flock.Program.Build
