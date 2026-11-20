/-
`Flock.Partition`: the program format `verity-ir/descriptor/v2` (`verity/circuits/PROTOCOL.md`), as the verifier
reads its own copy of a Program: the canonical JSON domain and the program's SHA-512 (§1), types and leaves (§2), reference
sequences (§3), the descriptor and what the decoder refuses (§4) and input nodes (§5), with each word primitive's
one Boolean Definition (§4.9, `Prog.boolean`).

The verifier evaluates no primitive, so it holds no registry: a primitive is taken as the descriptor declares it (its parameters
and return type are under the program's SHA-512), and an input (`Input<w>_v1`, bound per run, or `RegisteredInput<w>_v1`,
bound at registration; §5.2) must declare no parameters and `Value<w>`. Definitions
are decoded on demand from the root and from the functions `program_fn`'s record names, as core decodes them, and each callee
gets a smaller index than its callers.
-/
module

public import Base.Canon
public import Base.Hash
public import Base.JsonParse
public import Base.Sizes
public import Circuits.Steps
public import Lean.Data.Json.FromToJson.Basic
public import Std.Data.HashMap

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock.Program

open Lean (Json JsonNumber)

def SCHEMA : String := "verity-ir/descriptor/v2"
@[noinline] def MAX_SAFE_INT : Nat := 2 ^ 53 - 1

private def natJ? : Json → Option Nat
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

private def intJ? : Json → Option Int
  | .num n => if n.exponent == 0 then some n.mantissa else none
  | _ => none

/-! ### §1 The canonical JSON domain -/

/-- Where `j` leaves the domain on which canonical JSON is RFC 8785 (strings ASCII without DEL, integers of magnitude at most
2^53 − 1, no other numbers), or none (`codec.jcs_domain_error`). -/
def domainError (j : Json) (path : String := "$") : Option String :=
  match j with
  | .null | .bool _ => none
  | .num n =>
    if n.exponent != 0 then some s!"{path}: a float is not a JSON integer, string, boolean, null, array or object"
    else if n.mantissa.natAbs > MAX_SAFE_INT then some s!"{path}: integer {n.mantissa} exceeds 2^53 - 1 in magnitude"
    else none
  | .str s => if s.toList.all (fun c : Char => c.toNat < 0x7f) then none else some s!"{path}: string is not ASCII without DEL"
  | .arr a => (Array.range a.size).attach.findSome? fun i =>
    domainError (a[i.1]'(Array.mem_range.1 i.2)) s!"{path}[{i.1}]"
  | .obj kvs => kvs.toArray.attach.findSome? fun x =>
    if x.1.1.toList.all (fun c : Char => c.toNat < 0x7f) then domainError x.1.2 s!"{path}.{x.1.1}"
    else some s!"{path}.<key>: string is not ASCII without DEL"
termination_by j
decreasing_by
  · exact sizeOf_lt_of_mem_arr (Array.getElem_mem _)
  · exact sizeOf_lt_of_mem_obj x.2

/-! ### §2 Types and leaves -/

inductive Ty where
  | value (w : Nat)
  | array (n : Nat) (e : Ty)
  | tuple (fs : Array Ty)
  | record (fs : Array (String × Ty))
  deriving Inhabited

namespace Ty

private theorem sizeOf_lt_of_mem {fs : Array Ty} {t : Ty} (h : t ∈ fs) : sizeOf t < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  omega

private theorem sizeOf_snd_lt_of_mem {fs : Array (String × Ty)} {f : String × Ty} (h : f ∈ fs) :
    sizeOf f.2 < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  obtain ⟨k, t⟩ := f
  simp only [Prod.mk.sizeOf_spec] at this ⊢
  omega

def beq : Ty → Ty → Bool
  | .value a, .value b => a == b
  | .array n a, .array m b => n == m && beq a b
  | .tuple a, .tuple b => a.size == b.size && (a.attach.zip b).all fun (x, y) => beq x.1 y
  | .record a, .record b => a.size == b.size && (a.attach.zip b).all fun (x, y) => x.1.1 == y.1 && beq x.1.2 y.2
  | _, _ => false
termination_by t _ => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem x.2
  · exact sizeOf_snd_lt_of_mem x.2

instance instBEq : BEq Ty := ⟨beq⟩

def leaves : Ty → Nat
  | .value _ => 1
  | .array n e => n * leaves e
  | .tuple fs => fs.foldl (fun a t => a + leaves t) 0
  | .record fs => fs.foldl (fun a t => a + leaves t.2) 0
decreasing_by all_goals first | decreasing_tactic | (simp only [Ty.record.sizeOf_spec]; exact sizeOf_snd_lt_of_mem ‹_›)

/-- The leaf widths in leaf order (row-major: an Array's elements in index order, fields in declared order). -/
def widths : Ty → Array Nat
  | .value w => #[w]
  | .array n e => let w := widths e; (Array.range n).foldl (fun a _ => a ++ w) #[]
  | .tuple fs => fs.foldl (fun a t => a ++ widths t) #[]
  | .record fs => fs.foldl (fun a t => a ++ widths t.2) #[]
decreasing_by all_goals first | decreasing_tactic | (simp only [Ty.record.sizeOf_spec]; exact sizeOf_snd_lt_of_mem ‹_›)

def toJson : Ty → Json
  | .value w => Json.arr #[Json.str "Value", Lean.toJson w]
  | .array n e => Json.arr #[Json.str "Array", Lean.toJson n, toJson e]
  | .tuple fs => Json.arr #[Json.str "Tuple", Json.arr (fs.map toJson)]
  | .record fs => Json.arr #[Json.str "Record", Json.arr (fs.map fun f => Json.arr #[Json.str f.1, toJson f.2])]
decreasing_by all_goals first | decreasing_tactic | (simp only [Ty.record.sizeOf_spec]; exact sizeOf_snd_lt_of_mem ‹_›)

def render (t : Ty) : String := (toJson t).compress

/-- `type_from_json` with its constructors' checks: a width a positive integer, a length a natural number (neither a boolean),
a record's field names distinct strings. -/
def ofJson (j : Json) : V Ty := do
  let bad := fun (why : String) => (throw s!"type {j.compress}: {why}" : V Ty)
  match j with
  | .arr a =>
    match (a[0]? : Option Json) with
    | some (.str "Value") =>
      match (a[1]? : Option Json).bind natJ?, a.size with
      | some w, 2 => if w > 0 then pure (.value w) else bad "Value<w> needs a positive int width"
      | _, _ => bad "Value<w> needs a positive int width"
    | some (.str "Array") =>
      match (a[1]? : Option Json).bind natJ?, h : (a[2]? : Option Json), a.size with
      | some n, some e, 3 => pure (.array n (← ofJson e))
      | _, _, _ => bad "Array<n,T> needs a non-negative int length and an element type"
    | some (.str "Tuple") =>
      match h : (a[1]? : Option Json), a.size with
      | some (.arr fs), 2 => pure (.tuple (← fs.attach.mapM fun f => ofJson f.1))
      | _, _ => bad "a Tuple's fields are a list"
    | some (.str "Record") =>
      match h : (a[1]? : Option Json), a.size with
      | some (.arr fs), 2 => do
        let fs ← fs.attach.mapM fun f => match hf : f.1 with
          | .arr ys =>
            if hy : ys.size = 2 then
              match ys[0] with
              | .str k => do pure (k, ← ofJson ys[1])
              | _ => throw s!"type {j.compress}: a Record field is [name, type]"
            else throw s!"type {j.compress}: a Record field is [name, type]"
          | _ => throw s!"type {j.compress}: a Record field is [name, type]"
        if (List.range fs.size).any (fun i => (fs.extract 0 i).any (·.1 == fs[i]!.1)) then
          throw s!"type {j.compress}: duplicate record field names"
        pure (.record fs)
      | _, _ => bad "a Record's fields are a list"
    | _ => bad "unknown type tag"
  | _ => bad "not a list"
termination_by j
decreasing_by
  · exact sizeOf_lt_of_getElem?_arr h
  · exact Nat.lt_trans (sizeOf_lt_of_mem_arr f.2) (sizeOf_lt_of_getElem?_arr h)
  · have h1 := sizeOf_lt_of_mem_arr (Array.getElem_mem (xs := ys) (i := 1) (by omega))
    rw [← hf] at h1
    exact Nat.lt_trans h1 (Nat.lt_trans (sizeOf_lt_of_mem_arr f.2) (sizeOf_lt_of_getElem?_arr h))

/-- `remove_axis`: array axis `ax` stripped, with its length. -/
def removeAxis : Ty → Nat → Option (Nat × Ty)
  | .array n e, 0 => some (n, e)
  | .array n e, ax + 1 => (removeAxis e ax).map fun (m, inner) => (m, .array n inner)
  | _, _ => none

/-- For member `m` along axis `ax`: the axis' length and the leaves below it. Leaf `ℓ` of the member is leaf
`((ℓ / L) · d + m) · L + ℓ % L` of the whole (§3, members of a batch argument). -/
def axisDims : Ty → Nat → Option (Nat × Nat)
  | .array n e, 0 => some (n, leaves e)
  | .array _ e, ax + 1 => axisDims e ax
  | _, _ => none

/-- `boolean.bit_type` (§9): each `Value<w>` an `Array<w, Value<1>>`, a `Value<1>` kept a bit. -/
def bitType : Ty → Ty
  | .value w => if w == 1 then .value 1 else .array w (.value 1)
  | .array n e => .array n (bitType e)
  | .tuple fs => .tuple (fs.attach.map fun x => bitType x.1)
  | .record fs => .record (fs.attach.map fun x => (x.1.1, bitType x.1.2))
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem x.2
  · exact sizeOf_snd_lt_of_mem x.2

/-- Every leaf one bit wide (`width_set() <= {1}`); a type without leaves too. -/
def bits : Ty → Bool
  | .value w => w == 1
  | .array _ e => bits e
  | .tuple fs => fs.attach.all fun x => bits x.1
  | .record fs => fs.attach.all fun x => bits x.1.2
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem x.2
  · exact sizeOf_snd_lt_of_mem x.2

end Ty

/-- **A return's fields**: a Tuple's or a Record's, else the return itself. A Tuple or a Record of no field has no
field. -/
def fieldsOf : Ty → List Ty
  | .tuple fs => fs.toList
  | .record fs => fs.toList.map (·.2)
  | t => [t]

/-! ### §3 Reference sequences -/

/-- A body-scoped reference: space `p` (a parameter of the body) when `p`, else `n` (a node's output); the collection; the leaf. -/
structure Ref where
  p : Bool
  i : Nat
  j : Nat
  deriving Inhabited, BEq

def Ref.toJson (r : Ref) : Json := Json.arr #[Json.str (if r.p then "p" else "n"), Lean.toJson r.i, Lean.toJson r.j]

/-- A lane of a grid (PROTOCOL.md §3): its collection `(p, i)`, base `b`, inner stride `e` and levels `(radix, step)`,
innermost first, the last one unbounded. -/
structure Lane where
  p : Bool
  i : Nat
  b : Int
  e : Int
  lv : List (Nat × Int)
  deriving Inhabited, BEq

/-- The periods a lane's levels count: their radices' product. -/
def Lane.periods (l : Lane) : Nat := l.lv.foldl (fun a (r, _) => a * r) 1

/-- The leaf of lane `l` at period `t`, offset `o` in its run (`refs.grid_ref`'s arithmetic). -/
def Lane.leafAt (l : Lane) (t o : Nat) : Int :=
  let rec go : List (Nat × Int) → Nat → Int
    | [], _ => 0
    | [(_, d)], t => (t : Int) * d
    | (r, d) :: rest, t => ((t % r : Nat) : Int) * d + go rest (t / r)
  l.b + (o : Int) * l.e + go l.lv t

/-- The least and greatest leaf lane `l` of a whole grid of runs of `run` reaches (`refs.lane_bounds`). -/
def Lane.bounds (run : Nat) (l : Lane) : Int × Int :=
  ((run, l.e) :: l.lv).foldl (fun (lo, hi) (r, d) => (lo + ((r : Int) - 1) * min d 0, hi + ((r : Int) - 1) * max d 0))
    (l.b, l.b)

inductive Part where
  | affine (p : Bool) (i base count : Nat)
  | strided (p : Bool) (i : Nat) (base : Int) (inner : Nat) (os is : Int) (count : Nat)
  | explicit (items : Array Ref)
  | grid (run : Nat) (lanes : Array Lane)
  deriving Inhabited

def Part.count : Part → Nat
  | .affine _ _ _ c => c
  | .strided _ _ _ _ _ _ c => c
  | .explicit it => it.size
  | .grid run lanes => lanes.size * run * ((lanes[0]?.map Lane.periods).getD 0)

/-- The first node at or after node `k` that a part of `ps` holding a reference names (`codec._forward`). -/
def Part.forwardOf (ps : Array Part) (k : Nat) : Option Nat :=
  ps.findSome? fun q => if q.count == 0 then none else match q with
    | .affine p i _ _ | .strided p i _ _ _ _ _ => if !p && k ≤ i then some i else none
    | .explicit it => it.findSome? fun r => if !r.p && k ≤ r.i then some r.i else none
    | .grid _ lanes => lanes.findSome? fun l => if !l.p && k ≤ l.i then some l.i else none

/-- The refusal of node `k`'s argument naming node `i`, `i ≥ k` (`codec.forward_refusal`). -/
def forwardRefusal (k i : Nat) : String :=
  s!"forward-reference: node {k} reads node {i}, which is not before it; a node reads the body's parameters and earlier nodes only"

@[inline] def Part.get : Part → Nat → Ref
  | .affine p i b _, k => ⟨p, i, b + k⟩
  | .strided p i b inner os is _, k => ⟨p, i, (b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is).toNat⟩
  | .explicit it, k => it[k]!
  | .grid run lanes, k =>
    let l := lanes[k % (lanes.size * run) / run]!
    ⟨l.p, l.i, (l.leafAt (k / (lanes.size * run)) (k % run)).toNat⟩

/-- Adjacent levels where one continues the other, merged (`refs._merged_levels`). -/
def mergedLevels (lv : List (Nat × Int)) : List (Nat × Int) :=
  (lv.foldl (fun out (r, d) => match out with
    | (r', d') :: rest => if d == (r' : Int) * d' then (r' * r, d') :: rest else (r, d) :: out
    | [] => [(r, d)]) []).reverse

/-- Whether the grid `(run, lanes)` needs the grid form: no affine, strided or explicit part spells it (`refs.spell`,
which tries those first, after merging). -/
def gridNeeded (run : Nat) (lanes : Array Lane) : Bool := Id.run do
  let mut lanes := lanes.map fun l => { l with lv := mergedLevels l.lv }
  let mut run := run
  let g := lanes.size
  if g ≥ 2 then
    let l0 := lanes[0]!
    if lanes.all (fun l => l.p == l0.p && l.i == l0.i && l.e == l0.e && l.lv == l0.lv) then
      let c := lanes[1]!.b - l0.b
      if (List.range g).all (fun j => lanes[j]!.b == l0.b + (j : Int) * c) && (run == 1 || c == (run : Int) * l0.e) then
        lanes := #[{ l0 with e := if run == 1 then c else l0.e }]
        run := run * g
  if lanes.size == 1 then
    let l := lanes[0]!
    let mut e := l.e
    let mut lv := l.lv
    for _ in [0:l.lv.length] do
      match lv with
      | (r, d) :: rest =>
        if run == 1 || d == (run : Int) * e then
          e := if run == 1 then d else e
          run := run * r
          lv := mergedLevels rest
        else break
      | [] => break
    if lv.length ≤ 1 then return false
  if run == 1 && lanes.all (·.lv.isEmpty) then return false
  return true

/-- A flat sequence of parts (a concatenation spliced in), with each part's first index. -/
structure Seq where
  parts : Array Part
  starts : Array Nat
  count : Nat
  deriving Inhabited

def Seq.ofParts (ps : Array Part) : Seq :=
  let (starts, count) := ps.foldl (fun (acc : Array Nat × Nat) q => (acc.1.push acc.2, acc.2 + q.count)) (#[], 0)
  { parts := ps, starts, count }

/-- Reference `k` (`k < count`): the part holding it, by bisection over the starts (each round narrows `[lo, hi)`, so as many
rounds as parts finish it). -/
def Seq.get (s : Seq) (k : Nat) : Ref := Id.run do
  if s.parts.size == 1 then return s.parts[0]!.get k
  let mut lo := 0
  let mut hi := s.parts.size
  for _ in [0:s.parts.size] do
    if hi - lo ≤ 1 then break
    let mid := (lo + hi) / 2
    if s.starts[mid]! ≤ k then lo := mid else hi := mid
  return s.parts[lo]!.get (k - s.starts[lo]!)

/-- The references in order, repeats kept. -/
def Seq.toArray (s : Seq) : Array Ref :=
  s.parts.foldl (fun acc q => (Array.range q.count).foldl (fun acc k => acc.push (q.get k)) acc) #[]

/-- The least leaf position or step a grid states. -/
def gridLow : Int := -(2 ^ 62 : Nat)

/-- A lane's level `[radix, step]`; `shape` is the refusal. -/
def gridLevel (shape : String) (x : Json) : V (Nat × Int) := match x with
  | .arr #[r, d] => match natJ? r, intJ? d with
    | some r, some d => if r < 2 || d < gridLow then throw shape else pure (r, d)
    | _, _ => throw shape
  | _ => throw shape

/-- A lane's space, `"p"` or `"n"`; `shape` is the refusal. -/
def gridSpace (shape : String) (s : Json) : V Bool := match s with
  | .str "p" => pure true
  | .str "n" => pure false
  | _ => throw shape

/-- A lane `[space, index, base, inner_stride, [[radix, step], ...]]`; `shape` is the refusal. -/
def gridLane (shape : String) (l : Json) : V Lane := match l with
  | .arr #[sp, i, b, e, .arr lv] => do
    let p ← gridSpace shape sp
    match natJ? i, intJ? b, intJ? e with
    | some i, some b, some e =>
      if b < gridLow || e < gridLow then throw shape
      else do
        let levels ← lv.mapM (gridLevel shape)
        pure { p, i, b, e, lv := levels.toList }
    | _, _, _ => throw shape
  | _ => throw shape

/-- Whether lane `l`'s extremes over runs of `run` lie inside its parameter or node. -/
def Lane.fits (leaves : Bool → Nat → Option Nat) (run : Nat) (l : Lane) : Bool :=
  match leaves l.p l.i with
  | some n => 0 ≤ (l.bounds run).1 && (l.bounds run).2 < n
  | none => false

/-- The checks on a grid's parsed lanes; `j` and `g` only name the refusals. -/
def gridChecks (j g : Json) (run : Nat) (lanes : Array Lane) (leaves : Bool → Nat → Option Nat) : V (Array Part) :=
  if lanes.any (·.periods != lanes[0]!.periods) then
    throw s!"grid-shape: grid {g.compress}: its lanes' radices multiply to different period counts"
  else if !gridNeeded run lanes then
    throw s!"grid-spelled: grid {g.compress}: an affine, strided or explicit part spells it"
  else if !lanes.all (·.fits leaves run) then
    throw s!"grid-overrun: references {j.compress} reach outside a lane's parameter or node"
  else pure #[.grid run lanes]

/-- A `grid` part `g` of references `j` (PROTOCOL.md §3), checked as `refs.Grid` checks it: its shape (`grid-shape`),
that no affine, strided or explicit part spells it (`grid-spelled`), and every lane's extremes inside its parameter or
node (`grid-overrun`), by arithmetic a lane. -/
def parseGrid (j g : Json) (leaves : Bool → Nat → Option Nat) : V (Array Part) := do
  let shape := s!"grid-shape: grid {g.compress} is not [run, [[space, index, base, inner_stride, [[radix, step], ...]], ...]]"
  let .arr #[run, .arr ls] := g | throw shape
  let some run := natJ? run | throw shape
  if run == 0 || ls.isEmpty then throw shape
  let lanes ← ls.mapM (gridLane shape)
  gridChecks j g run lanes leaves

/-- A sequence's JSON parts, checked as `codec._checked_refs` checks them: every reference names a leaf of a parameter of the
body (`p`) or of a node's declared output (`n`); a strided sequence by the extremes of its pattern. `leaves p i` is that
collection's leaf count. An `alias` stands for an earlier node argument's sequence (`args k a`). Concatenations nest at most
`fuel` deep. -/
def parsePartsIn (fuel : Nat) (j : Json) (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq) :
    V (Array Part) := do
  let n + 1 := fuel | throw "references nest deeper than this verifier follows"
  let noLeaf := s!"references {j.compress} name no leaf of a parameter or a node"
  let space := fun (s : Json) => match s with
    | .str "p" => some true
    | .str "n" => some false
    | _ => none
  let within := fun (s i : Json) (lo hi : Int) => match space s, natJ? i with
    | some p, some i => match leaves p i with
      | some n => if 0 ≤ lo && hi < n then some (p, i) else none
      | none => none
    | _, _ => none
  let .obj _ := j | throw s!"references {j.compress}: not an object"
  if let .ok a := j.getObjVal? "alias" then
    match a with
    | .arr #[k, x] => match natJ? k, natJ? x with
      | some k, some x => match args k x with
        | some s => return s.parts
        | none => throw s!"alias to argument {x} of node {k}: aliases name an argument of an earlier node"
      | _, _ => throw s!"alias {a.compress}: aliases name an argument of an earlier node"
    | _ => throw s!"alias {a.compress}: aliases name an argument of an earlier node"
  if let .ok c := j.getObjVal? "concat" then
    let .arr ps := c | throw "a concat is a list"
    return (← ps.mapM (parsePartsIn n · leaves args)).foldl (· ++ ·) #[]
  if let .ok a := j.getObjVal? "affine" then
    let .arr #[s, i, b, c] := a | throw s!"affine {a.compress}: not [space, index, base, count]"
    let (some b, some c) := (intJ? b, natJ? c) | throw noLeaf
    if c == 0 then return #[.affine (space s == some true) 0 0 0]
    let some (p, i) := within s i b (b + c - 1) | throw noLeaf
    return #[.affine p i b.toNat c]
  if let .ok a := j.getObjVal? "strided" then
    let .arr #[s, i, b, inner, os, is, c] := a | throw s!"strided {a.compress}: not [space, index, base, inner, outer_stride, inner_stride, count]"
    let (some b, some inner, some os, some is, some c) := (intJ? b, natJ? inner, intJ? os, intJ? is, natJ? c) | throw noLeaf
    if inner == 0 then throw noLeaf
    let p0 := space s == some true
    if c == 0 then return #[.strided p0 0 0 inner os is 0]
    let q := (c - 1) / inner
    let last := (c - 1) % inner
    let corners : List (Nat × Nat) := [(0, 0), (0, min inner c - 1), (q, 0), (q, last)] ++ (if q > 0 then [(q - 1, inner - 1)] else [])
    let vals := corners.map fun (x, y) => b + (x : Int) * os + (y : Int) * is
    let some (p, i) := within s i (vals.foldl min vals.head!) (vals.foldl max vals.head!) | throw noLeaf
    return #[.strided p i b inner os is c]
  if let .ok e := j.getObjVal? "explicit" then
    let .arr items := e | throw "explicit references are a list"
    let rs ← items.mapM fun r => match r with
      | .arr #[s, i, l] => match intJ? l with
        | some l => match within s i l l with
          | some (p, i) => pure (⟨p, i, l.toNat⟩ : Ref)
          | none => throw s!"references {j.compress} name no leaf of a parameter or a node"
        | none => throw s!"references {j.compress} name no leaf of a parameter or a node"
      | _ => throw s!"references {j.compress} name no leaf of a parameter or a node"
    return #[.explicit rs]
  match j.getObjVal? "grid" with
  | .ok g => parseGrid j g leaves
  | .error _ => throw s!"references {j.compress}: not affine, strided, explicit, grid, concat or alias"

private def lane (s : String) (i b l : Nat) (axes : List (Nat × Nat)) : Json :=
  .arr #[.str s, i, b, l, .arr (axes.map fun (c, d) => Json.arr #[c, d]).toArray]

/-- The name the grid `[len, lanes]` is refused by over collections of `n` leaves, or `ok` (`test_grid.py` holds the
same table). -/
private def gridRefusal (len : Nat) (lanes : List Json) (n : Nat) : String :=
  match parsePartsIn 4 (Json.mkObj [("grid", .arr #[len, .arr lanes.toArray])]) (fun _ _ => some n) (fun _ _ => none) with
  | .ok _ => "ok"
  | .error e => (e.splitOn ":")[0]!

#guard gridRefusal 64 [lane "n" 2091 0 1 [(4, 0), (32, 64)]] 2048 == "ok"
#guard gridRefusal 64 [lane "n" 2091 0 1 [(4, 0), (32, 64)]] 2047 == "grid-overrun"
#guard gridRefusal 0 [lane "n" 1 0 1 []] 4096 == "grid-shape"
#guard gridRefusal 2 [] 4096 == "grid-shape"
#guard gridRefusal 2 [lane "n" 1 0 1 [(1, 4)]] 4096 == "grid-shape"
#guard gridRefusal 2 [lane "n" 1 0 1 [(2, 4)], lane "n" 2 0 1 [(3, 4)]] 4096 == "grid-shape"
#guard gridRefusal 2 [lane "q" 1 0 1 [], lane "n" 2 0 1 []] 4096 == "grid-shape"
#guard gridRefusal 2 [lane "n" 1 0 1 [(3, 10)]] 4096 == "grid-spelled"
#guard gridRefusal 1 [lane "n" 1 0 1 [], lane "n" 2 0 1 []] 4096 == "grid-spelled"
#guard gridRefusal 2 [lane "n" 1 0 1 [(3, 10)], lane "n" 1 2 1 [(3, 10)]] 4096 == "grid-spelled"
#guard gridRefusal 2048 [lane "n" 108 0 1 [(192, 0), (4, 2048)]] 8192 == "ok"
#guard gridRefusal 2048 [lane "n" 108 0 1 [(192, 0), (4, 2048)]] 8191 == "grid-overrun"

def parseParts (j : Json) (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq) : V (Array Part) :=
  parsePartsIn Steps.FUEL j leaves args

/-! ### §4 The descriptor -/

inductive Form where
  | prim | call | batch | scan
  deriving Inhabited, BEq

def Form.name : Form → String
  | .prim => "primitive" | .call => "call" | .batch => "batch" | .scan => "scan"

structure Arg where
  ty : Ty
  seq : Seq
  /-- a batch argument mapped over an axis, or a scan's `xs`: `(d, L)` of `Ty.axisDims` -/
  dims : Option (Nat × Nat) := none
  deriving Inhabited

structure Node where
  form : Form
  fn : Nat
  args : Array Arg
  out : Ty
  axes : Array (Option Nat) := #[]
  n : Nat := 1
  nxs : Nat := 0
  deriving Inhabited

/-- Members of a node (`parts.n_members`): a batch's members or a scan's iterations, else one. -/
def Node.members (x : Node) : Nat := if x.form == .batch || x.form == .scan then x.n else 1

structure Body where
  params : Array (String × Ty)
  retType : Ty
  nodes : Array Node
  ret : Seq
  offsets : Array Nat
  /-- the input nodes the body reads at another width, each with its `q` (`defs.Body.recut`) -/
  recut : Array (Nat × Nat) := #[]
  deriving Inhabited

structure Defn where
  /-- the descriptor id (the `definitions` key) -/
  key : String
  /-- `Name_vN` -/
  id : String
  prim : Bool
  params : Array (String × Ty)
  ret : Ty
  /-- an input primitive's binding time (`inputId?`), else none -/
  binding : Option String := none
  body : Body := default
  gates : Nat
  paramLeaves : Nat
  outLeaves : Nat
  deriving Inhabited

structure Prog where
  defs : Array Defn
  index : Std.HashMap String Nat
  root : Nat
  programFn : String
  /-- SHA-512 of the canonical descriptor without `annotations` (`partition_object.program_sha512`) -/
  sha512 : ByteArray
  /-- §4.9: each word primitive's key to its one Boolean Definition's key, both in `defs` -/
  boolean : Std.HashMap String String := {}
  deriving Inhabited

/-- `sha512` in hex, refused unless it is a SHA-512 digest: a Program built from another (a block's, by
`Spot.Restrict.restrict`) carries none, so no reader compares it with a record's. -/
def Prog.digest (P : Prog) : Except String String :=
  if P.sha512.size == 64 then .ok (hexOf P.sha512) else .error "program: no SHA-512 digest, so not a Program decoded from bytes"

#guard !(default : Prog).digest.toBool

def RUN : String := "run"
def REGISTRATION : String := "registration"
def SUPPLIED : String := "supplied"

private def widthAfter? (pre id : String) : Option Nat :=
  if !(id.startsWith pre && id.endsWith "_v1") then none else
  let w := ((id.drop pre.length).dropEnd 3).toString
  if w.length > 0 && w.all Char.isDigit && w.front != '0' then w.toNat? else none

/-- `program.input_id`: an input id's binding time and width: `run` for `^Input([1-9][0-9]*)_v1$` (`program.INPUT_ID`),
`registration` for `^RegisteredInput([1-9][0-9]*)_v1$` (`program.REGISTERED_ID`), `supplied` for
`^SuppliedInput([1-9][0-9]*)_v1$` (`program.SUPPLIED_ID`, §5.4); none for any other id. -/
def inputId? (id : String) : Option (String × Nat) :=
  match widthAfter? "Input" id with
  | some w => some (RUN, w)
  | none => match widthAfter? "RegisteredInput" id with
    | some w => some (REGISTRATION, w)
    | none => (widthAfter? "SuppliedInput" id).map (SUPPLIED, ·)

def inputWidth? (id : String) : Option Nat := (inputId? id).map (·.2)

def isInputId (id : String) : Bool := (inputId? id).isSome

/-- A gate or a constant bit of the Boolean profile (§9). -/
inductive GateKind where
  | and (n : Nat) | xor (n : Nat) | not | const (b : Bool)
  deriving Inhabited, BEq, Repr

/-- §9, by id with no registry: `And<n>_v1` and `Xor<n>_v1` for n ≥ 2 in decimal without leading zeros, `Not_v1`, and
`Const1[0x0]_v1` and `Const1[0x1]_v1`; none for any other id. -/
def gateKind? (id : String) : Option GateKind :=
  if id == "Not_v1" then some .not
  else if id == "Const1[0x0]_v1" then some (.const false)
  else if id == "Const1[0x1]_v1" then some (.const true)
  else match widthAfter? "And" id, widthAfter? "Xor" id with
    | some n, _ => if n ≥ 2 then some (.and n) else none
    | _, some n => if n ≥ 2 then some (.xor n) else none
    | _, _ => none

/-- §9's signature of a gate or constant bit: its parameters and its return. -/
def GateKind.signature : GateKind → Array (String × Ty) × Ty
  | .and n | .xor n => (#[("x", .array n (.value 1))], .value 1)
  | .not => (#[("a", .value 1)], .value 1)
  | .const _ => (#[], .value 1)

/-- A primitive a Boolean Definition may reach (§9): a gate, a constant bit, or a 1-bit input. -/
def isBooleanPrim (id : String) : Bool :=
  (gateKind? id).isSome || id == "Input1_v1" || id == "RegisteredInput1_v1" || id == "SuppliedInput1_v1"

/-- §4.9's word primitive: a primitive that is no gate, constant bit or input. -/
def isWordPrim (prim : Bool) (id : String) : Bool := prim && !isInputId id && !isBooleanPrim id

/-- The place in `P.defs` of word primitive `w`'s Boolean Definition (§4.9). -/
def Prog.bodyOf? (P : Prog) (w : String) : Option Nat := P.boolean[w]? >>= (P.index[·]?)

/-- The `k=<canonical json>` pairs' keys of a specialization id `Name_vN{k1=..,k2=..}`, in id order (`codec._split_record`). -/
def recordPairs (fid : String) : Array (String × String) := Id.run do
  if !(fid.contains '{' && fid.endsWith "}") then return #[]
  let cs := fid.toList
  let some open_ := cs.findIdx? (· == '{') | return #[]
  let rec_ := (cs.drop (open_ + 1)).dropLast ++ [',']
  let mut out := #[]
  let mut cur : List Char := []
  let mut depth : Int := 0
  let mut inStr := false
  let mut esc := false
  for ch in rec_ do
    if inStr then
      if esc then esc := false
      else if ch == '\\' then esc := true
      else if ch == '"' then inStr := false
      cur := ch :: cur
      continue
    if ch == '"' then inStr := true
    else if ch == '{' || ch == '[' then depth := depth + 1
    else if ch == '}' || ch == ']' then depth := depth - 1
    else if ch == ',' && depth == 0 then
      let piece := String.ofList cur.reverse
      let k := (piece.splitOn "=").head!
      out := out.push (k, (piece.drop (k.length + 1)).toString)
      cur := []
      continue
    cur := ch :: cur
  return out

private def fnRefs (acc : Array String) : Json → Array String
  | .obj kvs =>
    let items := kvs.toArray
    if items.size == 1 && items[0]!.1 == "fn" then
      match items[0]!.2 with
      | .str s => acc.push s
      | _ => acc
    else (items.attach.qsort (fun a b => a.1.1 < b.1.1)).foldl (fun acc x => fnRefs acc x.1.2) acc
  | .arr a => a.foldl (fun acc x => fnRefs acc x) acc
  | _ => acc
termination_by j => j
decreasing_by
  · exact sizeOf_lt_of_mem_obj x.2
  · exact sizeOf_lt_of_mem_arr ‹_›

/-- Every `{"fn": id}` in a specialization id's binding record, in encoding order (`codec._record_function_ids`). -/
def recordFunctionIds (fid : String) : Array String :=
  (recordPairs fid).foldl (fun acc (_, v) => match JsonParse.parse v with
    | .ok j => fnRefs acc j
    | .error _ => acc) #[]

/-! ### §4.4 A leaf read at the width it was committed with

`E_REINTERPRET` (`defs.Builder.finish`): a node's argument or a body's return that names a leaf of a parameter or of an
earlier node (any node, for the return) of another width than its own type gives that position is a silent
reinterpretation, and the decoder refuses the Definition as it finishes decoding it, with the codec's message. A node's
argument naming the node itself or a later one never reaches this check: the decoder refuses it as it reads the argument
(`forwardRefusal`). The check reads the description: each reference part is compared
by the structure of the two types, against the widths its own slice of the type it is read as gives it. A strided or
grid part is decided lane by lane, by the source's widths between the lane's extremes or by the residues the lane
reaches in the period of the array holding them (`laneWidth`); one neither decides is refused by name as not decided
by structure (`E_REINTERPRET unchecked`), never walked. Only an affine part's elements are walked where its two types'
shapes differ, and the first wrong leaf of a refused part is found leaf by leaf, at most `WALK` of them in a Program. -/

namespace Widths

/-- The widths of `t`'s leaves `lo … hi - 1`, each once (`types.range_widths`), from the type's structure. -/
def rangeWidths (t : Ty) (lo hi : Nat) : List Nat :=
  match t with
  | .value w => if lo == 0 && 0 < hi then [w] else []
  | .array n e =>
    let L := e.leaves
    let hi := min hi (n * L)
    if L == 0 || hi ≤ lo then [] else
      let a := lo / L
      let b := (hi - 1) / L
      if a == b then rangeWidths e (lo - a * L) (hi - a * L)
      else (rangeWidths e (lo - a * L) L ++ (if a + 1 < b then rangeWidths e 0 L else []) ++
        rangeWidths e 0 (hi - b * L)).eraseDups
  | .tuple fs => (fs.foldl (fun (acc : List Nat × Nat) f =>
      let o := acc.2
      let m := f.leaves
      (if o + m ≤ lo || hi ≤ o then acc.1 else acc.1 ++ rangeWidths f (lo - o) (min hi (o + m) - o), o + m)) ([], 0)).1.eraseDups
  | .record fs => (fs.foldl (fun (acc : List Nat × Nat) f =>
      let o := acc.2
      let m := f.2.leaves
      (if o + m ≤ lo || hi ≤ o then acc.1 else acc.1 ++ rangeWidths f.2 (lo - o) (min hi (o + m) - o), o + m)) ([], 0)).1.eraseDups
termination_by t
decreasing_by
  all_goals first
    | decreasing_tactic
    | (simp only [Ty.tuple.sizeOf_spec]; exact Ty.sizeOf_lt_of_mem ‹_›)
    | (simp only [Ty.record.sizeOf_spec]; exact Ty.sizeOf_snd_lt_of_mem ‹_›)

/-- The width of leaf `j` of `t`. -/
def wAt (t : Ty) (j : Nat) : Nat :=
  match t with
  | .value w => w
  | .array _ e => let L := e.leaves; if L == 0 then 0 else wAt e (j % L)
  | .tuple fs => (fs.foldl (fun (acc : Option Nat × Nat) f => match acc.1 with
      | some _ => acc
      | none => let m := f.leaves; (if j < acc.2 + m then some (wAt f (j - acc.2)) else none, acc.2 + m)) (none, 0)).1.getD 0
  | .record fs => (fs.foldl (fun (acc : Option Nat × Nat) f => match acc.1 with
      | some _ => acc
      | none => let m := f.2.leaves; (if j < acc.2 + m then some (wAt f.2 (j - acc.2)) else none, acc.2 + m)) (none, 0)).1.getD 0
termination_by t
decreasing_by
  all_goals first
    | decreasing_tactic
    | (simp only [Ty.tuple.sizeOf_spec]; exact Ty.sizeOf_lt_of_mem ‹_›)
    | (simp only [Ty.record.sizeOf_spec]; exact Ty.sizeOf_snd_lt_of_mem ‹_›)

/-- The smallest part of `t` holding its leaves `lo … hi - 1` (`hi > lo`), and the leaf of `t` it starts at. -/
def cover (t : Ty) (lo hi : Nat) : Ty × Nat :=
  match t with
  | .value _ => (t, 0)
  | .array _ e =>
    let L := e.leaves
    let a := lo / L
    if L == 0 || a != (hi - 1) / L then (t, 0) else
      let (c, o) := cover e (lo - a * L) (hi - a * L)
      (c, o + a * L)
  | .tuple fs => (fs.foldl (fun (acc : Option (Ty × Nat) × Nat) f => match acc.1 with
      | some _ => acc
      | none =>
        let m := f.leaves
        (if acc.2 ≤ lo && hi ≤ acc.2 + m then
          let (c, o) := cover f (lo - acc.2) (hi - acc.2)
          some (c, o + acc.2) else none, acc.2 + m)) (none, 0)).1.getD (t, 0)
  | .record fs => (fs.foldl (fun (acc : Option (Ty × Nat) × Nat) f => match acc.1 with
      | some _ => acc
      | none =>
        let m := f.2.leaves
        (if acc.2 ≤ lo && hi ≤ acc.2 + m then
          let (c, o) := cover f.2 (lo - acc.2) (hi - acc.2)
          some (c, o + acc.2) else none, acc.2 + m)) (none, 0)).1.getD (t, 0)
termination_by t
decreasing_by
  all_goals first
    | decreasing_tactic
    | (simp only [Ty.tuple.sizeOf_spec]; exact Ty.sizeOf_lt_of_mem ‹_›)
    | (simp only [Ty.record.sizeOf_spec]; exact Ty.sizeOf_snd_lt_of_mem ‹_›)

/-- The leaf counts of `t`'s children where it has more than one (an Array's elements, a Tuple's or Record's fields). -/
def childLeaves : Ty → Array Nat
  | .value _ => #[]
  | .array n e => Array.replicate (min n 2) e.leaves
  | .tuple fs => fs.map (·.leaves)
  | .record fs => fs.map (·.2.leaves)

/-- `t`'s one width, where all its leaves have one. -/
def homo (t : Ty) : Option Nat :=
  match rangeWidths t 0 t.leaves with
  | [w] => some w
  | _ => none

/-- Leaves walked one by one, past which a Program is refused: references this irregular are not read in closed form. -/
def WALK : Nat := 2 ^ 20

/-- How deep two types are taken apart. -/
def DEPTH : Nat := 256

abbrev M := EStateM String Nat

def UNCHECKED : String :=
  s!"its references are read leaf by leaf past {WALK} leaves to compare their widths, past which this verifier refuses them"

def step : M Unit := do
  let s ← get
  if s ≥ WALK then throw UNCHECKED
  set (s + 1)

/-- `{k · stride mod L : k < count}`, where it has at most `cap` members. -/
def residues (stride : Int) (count L cap : Nat) : Option (List Nat) :=
  let s := (stride % (L : Int)).toNat
  let g := Nat.gcd s L
  if count ≥ L / g then (if L / g ≤ cap then some ((List.range (L / g)).map (· * g)) else none)
  else if count ≤ cap then some ((List.range count).map fun k => k * s % L) else none

/-- `{(x + y) mod L : x ∈ A, y ∈ B}`. -/
def sumMod (L : Nat) (A B : List Nat) : List Nat := A.flatMap fun x => B.map fun y => (x + y) % L

/-- The residues mod `L` that `Lane.leafAt`'s levels `lv` add at the periods they count, where each level's and each
sum's have at most `cap` members. -/
def levelRes (L cap : Nat) : List (Nat × Int) → Option (List Nat)
  | [] => some [0]
  | [(r, d)] => residues d r L cap
  | (r, d) :: rest => do
    let A ← residues d r L cap
    let B ← levelRes L cap rest
    if A.length * B.length ≤ cap then some (sumMod L A B) else none

/-- The residues mod `L`, from leaf `off` on, of the leaves `b + o · e + ⋯` a lane of runs of `run` reaches over its
levels `lv` (a strided part is the lane of runs of `inner` with one level), where each set has at most `cap` members. -/
def laneRes (b : Int) (off run : Nat) (e : Int) (lv : List (Nat × Int)) (L cap : Nat) : Option (List Nat) := do
  let O ← residues e run L cap
  let G ← levelRes L cap lv
  if O.length * G.length ≤ cap then
    some ((sumMod L O G).map fun x => (((b - (off : Int)) % (L : Int)).toNat + x) % L)
  else none

/-- The most residues a lane is decided by. -/
def RCAP : Nat := 2 ^ 16

/-- Whether every leaf of `S` that a lane reaches between its extremes `lo … hi` has width `w`, by structure: `S`'s
widths between the extremes, or the residues the lane reaches in the period of the array holding them all, each of
which it reaches where `exact`. `none` where neither decides; `some false` only where a leaf the lane reaches has
another width. -/
def laneWidth (S : Ty) (w lo hi : Nat) (b : Int) (run : Nat) (e : Int) (lv : List (Nat × Int)) (exact : Bool) :
    Option Bool :=
  match rangeWidths S lo (hi + 1) with
  | [x] => some (x == w)
  | _ =>
    match cover S lo (hi + 1) with
    | (.array _ f, off) =>
      if f.leaves == 0 then none else
      match laneRes b off run e lv f.leaves RCAP with
      | some R => if R.all fun r => wAt f r == w then some true else if exact then some false else none
      | none => none
    | _ => none

def formName : Part → String
  | .affine .. => "affine"
  | .strided .. => "strided"
  | .explicit .. => "explicit"
  | .grid .. => "grid"

/-- The refusal of a part whose widths the structure of its types does not decide, which the check never walks. -/
def undecided (q : Part) (why : String) : String :=
  s!"a {formName q} part of {q.count} references, not decided by structure: {why}"

/-- Whether `t`'s leaves `a … a + n - 1` have the widths of `S`'s leaves `b … b + n - 1`. -/
def eqRange : Nat → Ty → Nat → Ty → Nat → Nat → M Bool
  | 0, _, _, _, _, _ => throw UNCHECKED
  | fuel + 1, t, a, S, b, n => do
    if n == 0 then return true
    match rangeWidths t a (a + n), rangeWidths S b (b + n) with
    | [x], [y] => return x == y
    | [_], _ | _, [_] => return false
    | _, _ =>
      let (ct, ot) := cover t a (a + n)
      let (cs, os) := cover S b (b + n)
      if a - ot == b - os && ct == cs then return true
      if let (.array _ e, .array _ f) := (ct, cs) then
        let L := e.leaves
        let L' := f.leaves
        if L > 0 && L' > 0 then
          if L == L' && n ≥ L then
            -- every residue mod `L` is met: `e`'s leaf `r` is read as `f`'s leaf `(r + d) % L`
            let d := ((b - os) + L - (a - ot) % L) % L
            return (← eqRange fuel e 0 f d (L - d)) && (← eqRange fuel e (L - d) f 0 d)
          let P := Nat.lcm L L'
          -- both sides repeat every `P` leaves: one window of `P` decides the rest
          if n > P then return ← eqRange fuel t a S b P
      -- a Tuple's or Record's fields are the description's, an Array's elements are walked: split by the side with fields
      let byS := match cs with | .array _ _ | .value _ => false | _ => true
      let (c, oc, base) := if byS then (cs, os, b) else (ct, ot, a)
      let pieces : Array (Nat × Nat) := match c with
        | .array _ e =>
          let L := e.leaves
          if L == 0 then #[] else
            let first := (base - oc) / L
            let last := (base + n - 1 - oc) / L
            (Array.range (min (last + 1 - first) (WALK + 1))).map fun k => (oc + (first + k) * L, L)
        | _ => (childLeaves c).foldl (fun (acc : Array (Nat × Nat) × Nat) m => (acc.1.push (acc.2, m), acc.2 + m)) (#[], oc) |>.1
      if pieces.isEmpty then return false
      for (o, m) in pieces do
        let lo := max base o
        let hi := min (base + n) (o + m)
        if lo < hi then
          step
          let k := lo - base
          if !(← eqRange fuel t (a + k) S (b + k) (hi - lo)) then return false
      if let some (o, m) := pieces.back? then
        if o + m < base + n then throw UNCHECKED
      return true

/-- A body's leaf sources: its parameters' types, and its nodes' outputs below `below`. -/
structure Src where
  params : Array Ty
  nodes : Array Ty
  hp : Array (Option Nat)
  hn : Array (Option Nat)
  below : Nat
  /-- each node's leaf width where it is an input node (PROTOCOL.md §5.2), else none -/
  inp : Array (Option Nat) := #[]
  /-- each input node's first reader width in `defs.Builder.finish`'s order, where one reads it -/
  first : Array (Option Nat) := #[]

def Src.ty? (s : Src) (p : Bool) (i : Nat) : Option Ty :=
  if p then s.params[i]? else if i < s.below then s.nodes[i]? else none

def Src.homo? (s : Src) (p : Bool) (i : Nat) : Option Nat :=
  if p then (s.hp[i]?).join else (s.hn[i]?).join

/-! #### §4.4's read at another width

An input node of `N` leaves of width `W` read at a width `w` that divides `W`, `q = W / w`: its references index its `N·q`
leaves of width `w`, leaf `j` the bits `[(j mod q)·w, (j mod q + 1)·w)` of its leaf `j div q`. `recutOk` is `defs`'s width
check's input-node path (`_Widths.other`, through `misread`'s cut of a part into runs of one width, and a grid's pieces),
and `recut` is `refs.recut` on one such run, refused by the codec's names and words: `recut-width`, `recut-boundary`,
`recut-overrun`. Decided by structure, but where a strided run falls apart into single references, a grid lane overruns
its node, or a grid is read as leaves of more than one width. -/

/-- The leaf width of source `(p, i)` where it is an input node below `s.below`. -/
def Src.inW? (s : Src) (p : Bool) (i : Nat) : Option Nat :=
  if p || s.below ≤ i then none else (s.inp[i]?).join

private def pySpace (p : Bool) : String := if p then "'p'" else "'n'"

/-- `part.to_json()` as Python prints it in a message. -/
def pyPart : Part → String
  | .affine p i b c => s!"\{'affine': [{pySpace p}, {i}, {b}, {c}]}"
  | .strided p i b inner os is c => s!"\{'strided': [{pySpace p}, {i}, {b}, {inner}, {os}, {is}, {c}]}"
  | .explicit it => "{'explicit': [" ++ ", ".intercalate (it.toList.map fun r => s!"[{pySpace r.p}, {r.i}, {r.j}]") ++ "]}"
  | .grid run lanes => s!"\{'grid': [{run}, [" ++ ", ".intercalate (lanes.toList.map fun l =>
      s!"[{pySpace l.p}, {l.i}, {l.b}, {l.e}, [" ++ ", ".intercalate (l.lv.map fun (r, d) => s!"[{r}, {d}]") ++ "]]") ++
      "]]}"

/-- `refs.recut(W, w, sl, N)`, then `_recut(i, q)` against the node's first reader width `w0`: input node `i` of `N` leaves of
width `W` read by `sl` (an affine or strided part, or an explicit one of one reference, not empty) as leaves of width `w`. -/
def recut (i N W w w0 : Nat) (sl : Part) : M Unit := do
  if w == 0 || W % w != 0 then
    throw s!"recut-width: an input node of {W}-bit leaves read in {w}-bit leaves: the reader's width divides the registered width"
  let q := W / w
  let qi : Int := q
  -- `refs.as_grid`'s one lane: its run, base, inner stride and levels; and `lane_bounds`, or a strided part's corners
  let (run, b, e, lv, lo, hi) : Nat × Int × Int × List (Nat × Int) × Int × Int := match sl with
    | .affine _ _ b c => (c, b, 1, [], b, (b : Int) + c - 1)
    | .strided _ _ b inner os is c =>
      let k := (c - 1) / inner
      let corners : List (Nat × Nat) :=
        [(0, 0), (0, min inner c - 1), (k, 0), (k, (c - 1) % inner)] ++ (if k > 0 then [(k - 1, inner - 1)] else [])
      let vals := corners.map fun (x, y) => b + (x : Int) * os + (y : Int) * is
      (inner, b, is, if (c + inner - 1) / inner > 1 then [((c + inner - 1) / inner, os)] else [],
        vals.foldl min vals.head!, vals.foldl max vals.head!)
    | .explicit it => let j : Int := (it[0]!).j; (1, j, 0, [], j, j)
    | .grid .. => (0, 0, 0, [], 0, 0)
  if q > 1 && !(e == 1 && b % qi == 0 && run % q == 0 && sl.count % q == 0 && lv.all fun (_, d) => d % qi == 0) then
    throw (s!"recut-boundary: {pyPart sl} reads an input node of {W}-bit leaves in {w}-bit leaves other than whole " ++
      "registered leaves")
  if !(0 ≤ lo && hi < ((N * q : Nat) : Int)) then
    throw s!"recut-overrun: {pyPart sl} reads past the {N} {W}-bit leaves of its input node"
  if w != w0 then
    throw s!"recut-width: input node {i} read in leaves of two widths; an input node is read at one"

/-- References `lo … hi - 1` of part `q` as a part (`Refs.slice`): an affine or strided part's run, else its references. -/
def sliceOf (q : Part) (lo hi : Nat) : Part :=
  match q with
  | .affine p i b c => .affine p i (b + min lo c) (min hi c - min lo c)
  | .strided p i b inner os is c =>
    let hi := min hi c
    if hi ≤ lo then .explicit #[]
    else if lo % inner == 0 && (hi % inner == 0 || hi == c) then
      .strided p i (b + ((lo / inner : Nat) : Int) * os) inner os is (hi - lo)
    else if hi - lo ≤ inner - lo % inner then
      if is == 1 then .affine p i (q.get lo).j (hi - lo) else .strided p i (q.get lo).j 1 is 0 (hi - lo)
    else .explicit ((Array.range (hi - lo)).map fun k => q.get (lo + k))
  | .explicit it => .explicit (it.extract lo hi)
  | g => g

/-- `misread`'s cut of `t`'s leaves `pos … pos + n - 1` into runs of one width, in order, each given to `f` as
`(pos, n, w)`: a run of one width whole, else the smallest part of `t` holding them cut at its elements or fields. -/
def bySlices (f : Nat → Nat → Nat → M Unit) : Nat → Ty → Nat → Nat → M Unit
  | 0, _, _, _ => throw UNCHECKED
  | fuel + 1, t, pos, n => do
    if n == 0 then return
    match rangeWidths t pos (pos + n) with
    | [w] => f pos n w
    | _ =>
      let (c, off) := cover t pos (pos + n)
      let pieces : Array (Nat × Nat) := match c with
        | .array _ e =>
          let L := e.leaves
          if L == 0 then #[] else
            let first := (pos - off) / L
            (Array.range ((pos + n - 1 - off) / L + 1 - first)).map fun k => (off + (first + k) * L, L)
        | _ => (childLeaves c).foldl (fun (acc : Array (Nat × Nat) × Nat) m => (acc.1.push (acc.2, m), acc.2 + m)) (#[], off) |>.1
      for (o, m) in pieces do
        let a := max pos o
        let b := min (pos + n) (o + m)
        if a < b then bySlices f fuel t a (b - a)

/-- A grid's piece: lane `l`'s run of period `t` (`Grid.pieces`). -/
def gridPiece (run : Nat) (l : Lane) (t : Nat) : Part :=
  let leaf := (l.leafAt t 0).toNat
  if l.e == 1 || run == 1 then .affine l.p l.i leaf run else .strided l.p l.i leaf run 0 l.e run

/-- The first period of a lane of runs of `run` reading an input node at `q` that `refs.recut` refuses as off whole leaves,
where one is: `0` if its first run is, else the first period that steps a level by a whole number of leaves short. -/
def offWhole (run q : Nat) (l : Lane) : Option Nat :=
  let qi : Int := q
  if q ≤ 1 then none
  else if !((l.e == 1 || run == 1) && run % q == 0 && l.b % qi == 0) then some 0
  else (l.lv.foldl (fun (acc : Option Nat × Nat) (r, d) => match acc.1 with
      | some t => (some t, acc.2)
      | none => (if d % qi != 0 then some acc.2 else none, acc.2 * r)) (none, 1)).1

/-- The read at another width of part `q`, read as `t`'s leaves `a …` (`t`'s one width `ht`, if it has one): `none` where
`q` reads no input node, and the rest of `partOk` decides it; else refused by name, `some false` where it names a leaf of
another source at another width, or `some true`. An explicit part's input nodes are checked here and its other leaves by
`partOk`. -/
def recutOk (s : Src) (t : Ty) (ht : Option Nat) (a : Nat) (q : Part) : M (Option Bool) := do
  let N := fun (i : Nat) => ((s.nodes[i]?).map Ty.leaves).getD 0
  let w0 := fun (i w : Nat) => ((s.first[i]?).join).getD w
  match q with
  | .affine p i _ n | .strided p i _ _ _ _ n =>
    let some W := s.inW? p i | return none
    bySlices (fun x m w => recut i (N i) W w (w0 i w) (sliceOf q (x - a) (x - a + m))) DEPTH t a n
    return some true
  | .explicit items =>
    if !items.any (fun r => (s.inW? r.p r.i).isSome) then return none
    for k in [0:items.size] do
      let r := items[k]!
      if let some W := s.inW? r.p r.i then
        recut r.i (N r.i) W (wAt t (a + k)) (w0 r.i (wAt t (a + k))) (.explicit #[r])
    return none
  | .grid run lanes =>
    if !lanes.any (fun l => (s.inW? l.p l.i).isSome) then return none
    let g := lanes.size
    let P := (lanes[0]?.map Lane.periods).getD 0
    let some w := ht | do
      -- read as leaves of more than one width: piece by piece, in `Grid.pieces`' order
      for t' in [0:P] do
        for j in [0:g] do
          step
          let l := lanes[j]!
          let pos := a + (t' * g + j) * run
          match s.inW? l.p l.i with
          | some W =>
            let f := fun x m w => recut l.i (N l.i) W w (w0 l.i w) (sliceOf (gridPiece run l t') (x - pos) (x - pos + m))
            bySlices f DEPTH t pos run
          | none => for o in [0:run] do
              if let some S := s.ty? l.p l.i then
                if wAt S (l.leafAt t' o).toNat != wAt t (pos + o) then return some false
      return some true
    -- one width: each input lane's first piece refused, at its place in `Grid.pieces`' order; the earliest is the refusal
    let mut first : Option (Nat × Nat × Nat) := none
    for j in [0:g] do
      let l := lanes[j]!
      let some W := s.inW? l.p l.i | continue
      let q := if w == 0 then 0 else W / w
      let fits := fun (lo hi : Int) => 0 ≤ lo && hi < ((N l.i * q : Nat) : Int)
      let over ← if w == 0 || W % w != 0 || fits (l.bounds run).1 (l.bounds run).2 then pure none else do
        let t' := (List.range (min P (WALK + 1))).find? fun t' =>
          let (lo, hi) := Lane.bounds run { l with b := l.leafAt t' 0, lv := [] }
          !fits lo hi
        if t'.isNone then throw UNCHECKED
        pure t'
      let bad := if w == 0 || W % w != 0 then [0] else
        [offWhole run q l, over, if w != w0 l.i w then some 0 else none].filterMap id
      if let some t' := bad.min? then
        let at_ := t' * g + j
        if first.all (at_ < ·.1) then first := some (at_, j, t')
    if let some (_, j, t') := first then
      let l := lanes[j]!
      recut l.i (N l.i) ((s.inW? l.p l.i).getD 0) w (w0 l.i w) (gridPiece run l t')
    -- every other lane: its source's one width, or one width over the leaves between its extremes
    return some (lanes.all fun l => (s.inW? l.p l.i).isSome || match s.ty? l.p l.i with
      | none => true
      | some S => s.homo? l.p l.i == some w ||
        let (lo, hi) := l.bounds run
        rangeWidths S lo.toNat (hi.toNat + 1) == [w])

/-- `recutOk`'s verdict on part `q` of a 16-bit input node of 4 leaves, read as a `t`, its first reader width `w0`. -/
private def recutAt (w0 : Nat) (t : Ty) (q : Part) : String :=
  let s : Src := { params := #[], nodes := #[.array 4 (.value 16)], hp := #[], hn := #[some 16], below := 1,
                   inp := #[some 16], first := #[some w0] }
  match (recutOk s t (homo t) 0 q).run 0 with
  | .ok (some b) _ => toString b
  | .ok none _ => "none"
  | .error e _ => e

-- the codec's verdicts (`refs.recut`) on the same reads
#guard recutAt 8 (.array 8 (.value 8)) (.affine false 0 0 8) == "true"
#guard recutAt 8 (.array 4 (.value 8)) (.strided false 0 2 2 4 1 4) == "true"
#guard recutAt 16 (.array 3 (.value 16)) (.affine false 0 1 3) == "true"
#guard recutAt 32 (.array 1 (.value 32)) (.affine false 0 0 1) ==
  "recut-width: an input node of 16-bit leaves read in 32-bit leaves: the reader's width divides the registered width"
#guard recutAt 8 (.array 4 (.value 8)) (.affine false 0 1 4) == "recut-boundary: {'affine': ['n', 0, 1, 4]} reads an " ++
  "input node of 16-bit leaves in 8-bit leaves other than whole registered leaves"
#guard recutAt 8 (.array 1 (.value 8)) (.explicit #[⟨false, 0, 2⟩]) == "recut-boundary: {'explicit': [['n', 0, 2]]} " ++
  "reads an input node of 16-bit leaves in 8-bit leaves other than whole registered leaves"
#guard recutAt 8 (.array 6 (.value 8)) (.affine false 0 4 6) ==
  "recut-overrun: {'affine': ['n', 0, 4, 6]} reads past the 4 16-bit leaves of its input node"
#guard recutAt 8 (.array 4 (.value 4)) (.affine false 0 0 4) ==
  "recut-width: input node 0 read in leaves of two widths; an input node is read at one"
#guard recutAt 16 (.array 2 (.value 16)) (.grid 1 #[⟨false, 0, 0, 0, [(2, 1)]⟩]) == "true"

/-- Each input node's first reader width in `defs.Builder.finish`'s order (each node's arguments, then the return), over
the sources `s` gives each: of a part's first reference into it. -/
def firstReads (inp : Array (Option Nat)) (B : Body) : Array (Option Nat) := Id.run do
  let mut first : Array (Option Nat) := Array.replicate B.nodes.size none
  let seqs : Array (Nat × Ty × Seq) :=
    (B.nodes.mapIdx fun k nd => nd.args.map fun a => (k, a.ty, a.seq)).flatten.push (B.nodes.size, B.retType, B.ret)
  for (below, t, seq) in seqs do
    for q in seq.parts, a in seq.starts do
      let reads : List (Bool × Nat × Nat) := match q with
        | .affine p i _ n | .strided p i _ _ _ _ n => if n == 0 then [] else [(p, i, a)]
        | .explicit it => (List.range it.size).map fun k => (it[k]!.p, it[k]!.i, a + k)
        | .grid run lanes => (List.range lanes.size).map fun j => (lanes[j]!.p, lanes[j]!.i, a + j * run)
      for (p, i, pos) in reads do
        if !p && i < below && ((inp[i]?).join).isSome && ((first[i]?).join).isNone then
          first := first.set! i (some (wAt t pos))
  return first

/-- Each node's leaf width where it is an input node (`defs._Widths.input_width`): a call of an input primitive, its
output of one width. -/
def inputWidths (defs : Array Defn) (B : Body) : Array (Option Nat) :=
  B.nodes.map fun nd => match defs[nd.fn]? with
    | some c => if c.prim && c.binding.isSome then homo nd.out else none
    | none => none

/-- The input nodes body `B` reads at another width, each with its `q` above 1 (`Body.recut`), in node order. -/
def recutOf (defs : Array Defn) (B : Body) : Array (Nat × Nat) :=
  let inp := inputWidths defs B
  ((firstReads inp B).zip inp).zipIdx.filterMap fun ((f, W), i) => match f, W with
    | some w, some W => if 0 < w && W % w == 0 && W / w > 1 then some (i, W / w) else none
    | _, _ => none

/-- Whether the references of part `q`, read as `t`'s leaves `a …` (`t`'s one width `ht`, if it has one), name leaves of
the widths `t` gives them. -/
def partOk (s : Src) (t : Ty) (ht : Option Nat) (a : Nat) (q : Part) : M Bool := do
  if let some r ← recutOk s t ht a q then return r
  match q with
  | .explicit items =>
    return (List.range items.size).all fun k =>
      let r := items[k]!
      match s.ty? r.p r.i with
      | some S => wAt S r.j == wAt t (a + k)
      | none => true
  | .affine p i base n =>
    let some S := s.ty? p i | return true
    if ht.isSome && ht == s.homo? p i then return true
    eqRange DEPTH t a S base n
  | .strided p i base inner os is n =>
    let some S := s.ty? p i | return true
    if n == 0 || (ht.isSome && ht == s.homo? p i) then return true
    let some w := ht | throw (undecided q "the type it is read as has leaves of several widths along it")
    let q' := (n - 1) / inner
    let corners : List (Nat × Nat) := [(0, 0), (0, min inner n - 1), (q', 0), (q', (n - 1) % inner)] ++
      (if q' > 0 then [(q' - 1, inner - 1)] else [])
    let vals := corners.map fun (x, y) => (base + (x : Int) * os + (y : Int) * is).toNat
    let lo := vals.foldl min vals.head!
    let hi := vals.foldl max vals.head!
    match laneWidth S w lo hi base inner is [((n + inner - 1) / inner, os)] (n % inner == 0) with
    | some true => return true
    | some false => return false
    | none => throw (undecided q s!"the leaves of {if p then "parameter" else "node"} {i} it reads are of several widths")
  | .grid run lanes =>
    let some w := ht | throw (undecided q "the type it is read as has leaves of several widths along it")
    let verdict := fun (l : Lane) => match s.ty? l.p l.i with
      | none => some true
      | some S =>
        if s.homo? l.p l.i == some w then some true else
        let (lo, hi) := l.bounds run
        laneWidth S w lo.toNat hi.toNat l.b run l.e l.lv true
    let vs := lanes.map verdict
    if vs.all (· == some true) then return true
    if vs.any (· == some false) then return false
    throw (undecided q "a lane reads leaves of several widths")

/-- The first leaf of part `q`, read as `t`'s leaves `a …`, that names a leaf of another width than `t` gives it, as an
offset into `q`, where `partOk` found one: an affine part's by bisection over its prefixes (`eqRange` on each), any
other's leaf by leaf. -/
def firstBad (s : Src) (t : Ty) (a : Nat) (q : Part) : M (Option Nat) := do
  let bad := fun (k : Nat) => let r := q.get k
    match s.ty? r.p r.i with
    | some S => (s.inW? r.p r.i).isNone && wAt S r.j != wAt t (a + k)
    | none => false
  match q with
  | .affine p i base n =>
    let some S := s.ty? p i | return none
    -- leaves `0 … lo - 1` are read at their widths, and some leaf below `hi` is not
    let mut lo := 0
    let mut hi := n
    for _ in [0:n.log2 + 2] do
      if hi ≤ lo + 1 then break
      let mid := (lo + hi) / 2
      if ← eqRange DEPTH t a S base mid then lo := mid else hi := mid
    return if hi == lo + 1 && bad lo then some lo else none
  | _ =>
    for k in [0:q.count] do
      step
      if bad k then return some k
    return none

/-- The one width of `t`'s leaves `a … a + n`, if they have one. -/
def sliceWidth (t : Ty) (a n : Nat) : Option Nat :=
  match rangeWidths t a (a + n) with
  | [w] => some w
  | _ => none

/-- The first part of the references `seq`, read as a `t`, that names leaves of another width, with the leaf it starts at. -/
def seqBad (s : Src) (t : Ty) (seq : Seq) : M (Option (Nat × Part)) := do
  for q in seq.parts, a in seq.starts do
    if !(← partOk s t (sliceWidth t a q.count) a q) then return some (a, q)
  return none

def TAIL : String :=
  "a silent reinterpretation. A word is read only at the width it was committed with; a runtime word is split into " ++
  "limbs only by a registered LimbSplit Definition, which computes the limbs and their carries"

/-- The refusal of composite `key`, whose body `B`'s references `what` read as a `t` name leaves of other widths in part
`q` from leaf `a` on: the codec's message, at the first such leaf, where it is found within `WALK` steps. -/
def refusal (B : Body) (key what : String) (s : Src) (t : Ty) (a : Nat) (q : Part) : String :=
  match (firstBad s t a q).run 0 with
  | .ok (some k) _ =>
    let r := q.get k
    let src := if r.p then s!"parameter '{(B.params[r.i]?.map (·.1)).getD ""}'" else s!"node {r.i}"
    s!"definition '{key}': E_REINTERPRET: {what}: leaf {a + k} is typed Value<{wAt t (a + k)}> but names leaf {r.j} of " ++
      s!"{src}, a Value<{((s.ty? r.p r.i).map (wAt · r.j)).getD 0}>: {TAIL}"
  | _ => s!"definition '{key}': E_REINTERPRET: {what}: the references from its leaf {a} name leaves of other widths " ++
      s!"than its type {t.render} gives them: {TAIL}"

/-- Where the body `B` of composite `key` reads a leaf at another width (`defs.Builder.finish`'s order: each node's
arguments, then the return), as its refusal; `defs` holds its callees. With `recut`, an input node may be read at a width
dividing its own (§4.4, `recutOk`); without, every leaf is read at its own width. -/
def bodyBad (defs : Array Defn) (key : String) (B : Body) (recut : Bool := false) : M (Option String) := do
  let outs := B.nodes.map (·.out)
  let inp := if recut then inputWidths defs B else #[]
  let s : Src := { params := B.params.map (·.2), nodes := outs, hp := B.params.map (homo ·.2), hn := outs.map homo,
                   below := 0, inp, first := if recut then firstReads inp B else #[] }
  for k in [0:B.nodes.size] do
    let node := B.nodes[k]!
    let callee := defs[node.fn]!
    for i in [0:node.args.size] do
      let a := node.args[i]!
      let sk := { s with below := k }
      if let some (pos, q) ← seqBad sk a.ty a.seq then
        let pn := (callee.params[i]?.map (·.1)).getD s!"{i}"
        return some (refusal B key s!"node {k} ({node.form.name} {callee.id}) argument '{pn}'" sk a.ty pos q)
  let sr := { s with below := B.nodes.size }
  if let some (pos, q) ← seqBad sr B.retType B.ret then
    return some (refusal B key "the body's return" sr B.retType pos q)
  return none

end Widths

private structure DecState where
  defs : Array Defn := #[]
  index : Std.HashMap String Nat := {}
  busy : Std.HashSet String := {}
  /-- leaves the width check has walked one by one so far (`Widths.WALK`) -/
  walked : Nat := 0

private abbrev D := StateT DecState (Except String)

private structure Desc where
  ddefs : Std.HashMap String Json
  callees : Array Json
  types : Array Ty

private def at_ {α} [Inhabited α] (table : Array α) (j : Json) (what : String) : V α :=
  match intJ? j with
  | some i => if 0 ≤ i && i < table.size then pure table[i.toNat]! else throw s!"{what} index {i} outside 0..{(table.size : Int) - 1}"
  | none => throw s!"{what} index {j.compress} outside 0..{(table.size : Int) - 1}"

private def liftV {α} (x : V α) : D α := fun s => match x with
  | .ok a => .ok (a, s)
  | .error e => .error e

private def paramsOf (j : Json) (what : String) : V (Array (String × Ty)) := do
  let .arr ps := j | throw s!"{what}: params are a list"
  let ps ← ps.mapM fun p => match p with
    | .arr #[.str k, t] => do pure (k, ← Ty.ofJson t)
    | _ => throw s!"{what}: a parameter is [name, type]"
  if (ps.foldl (fun s p => s.insert p.1) ({} : Std.HashSet String)).size != ps.size then
    throw s!"{what}: parameters: a name twice"
  return ps

/-- `_build_node`: the node rebuilt as `defs.Builder` builds it, its arguments checked against the callee, and its computed
output type, which must be the declared one. -/
private def buildNode (fid : String) (k : Nat) (form : Form) (callee : Defn) (args : Array Arg) (axes : Array (Option Nat))
    (n nxs : Nat) : V (Ty × Array Arg) := do
  let pre := s!"definition '{fid}' node {k}"
  let params := callee.params
  match form with
  | .prim | .call =>
    if args.size != params.size then throw s!"{pre}: call {callee.key}: expected {params.size} arguments, got {args.size}"
    for (pn, pt) in params, a in args do
      if !(a.ty == pt) then throw s!"{pre}: call {callee.key}: argument '{pn}' has type {a.ty.render}, expected {pt.render}"
    return (callee.ret, args)
  | .batch =>
    if axes.size != args.size then throw s!"{pre}: batch: one axis entry per argument"
    if params.size != args.size then throw s!"{pre}: batch {callee.key}: expected {params.size} arguments, got {args.size}"
    let mut lengths : Array Nat := #[]
    let mut out := #[]
    for (pn, pt) in params, a in args, ax in axes do
      match ax with
      | none =>
        if !(a.ty == pt) then throw s!"{pre}: batch {callee.key}: member type of '{pn}' is {a.ty.render}, expected {pt.render}"
        out := out.push a
      | some ax =>
        let some (ln, mt) := a.ty.removeAxis ax | throw s!"{pre}: axis {ax} requires nested Arrays, got {a.ty.render}"
        if !(mt == pt) then throw s!"{pre}: batch {callee.key}: member type of '{pn}' is {mt.render}, expected {pt.render}"
        if !lengths.contains ln then lengths := lengths.push ln
        out := out.push { a with dims := a.ty.axisDims ax }
    if lengths.size > 1 then throw s!"{pre}: batch {callee.key}: mapped axes have different lengths {lengths.qsort (· < ·)}"
    if lengths.size == 1 && lengths[0]! != n then throw s!"{pre}: batch: explicit n disagrees with mapped axis length"
    return (.array n callee.ret, out)
  | .scan =>
    let .tuple #[carry, y] := callee.ret | throw s!"{pre}: scan {callee.key}: step must return (carry, y), got {callee.ret.render}"
    if params.size != args.size then throw s!"{pre}: scan {callee.key}: step arity {params.size} != 1 + {nxs} xs + {args.size - 1 - nxs} shared"
    if !(params[0]!.2 == carry) then throw s!"{pre}: scan {callee.key}: carry parameter {params[0]!.2.render} != carried type {carry.render}"
    if !(args[0]!.ty == carry) then throw s!"{pre}: scan {callee.key}: init has type {args[0]!.ty.render}, expected {carry.render}"
    let mut lengths : Array Nat := #[]
    let mut out := #[args[0]!]
    for i in [1:args.size] do
      let (pn, pt) := params[i]!
      let a := args[i]!
      if i ≤ nxs then
        let .array ln e := a.ty | throw s!"{pre}: scan {callee.key}: xs argument '{pn}' must be an Array"
        if !lengths.contains ln then lengths := lengths.push ln
        if !(e == pt) then throw s!"{pre}: scan {callee.key}: xs element type {e.render} != {pt.render} for '{pn}'"
        out := out.push { a with dims := some (ln, e.leaves) }
      else
        if !(a.ty == pt) then throw s!"{pre}: scan {callee.key}: shared argument '{pn}' has type {a.ty.render}, expected {pt.render}"
        out := out.push a
    if lengths.size > 1 then throw s!"{pre}: scan {callee.key}: xs lengths differ {lengths.qsort (· < ·)}"
    if lengths.size == 1 && lengths[0]! != n then throw s!"{pre}: scan: explicit n disagrees with xs length"
    return (.tuple #[carry, .array n y], out)

/-- The leaf counts references are parsed against (`codec`'s `leaves`). With `recut`, an input node among `nodes` as leaves
of one bit, `N·W` (§4.4: read at a width that divides `W`, `refs.recut` then holds its references under `N·q`); every other
collection, and every one without `recut`, as `leaves` gives it. -/
def leavesFor (recut : Bool) (defs : Array Defn) (nodes : Array Node) (leaves : Bool → Nat → Option Nat) :
    Bool → Nat → Option Nat := if !recut then leaves else fun p i =>
  match p, nodes[i]? with
  | false, some nd => match defs[nd.fn]? with
    | some c => if c.prim && c.binding.isSome then some (nd.out.leaves * (Widths.homo nd.out).getD 1) else leaves p i
    | none => leaves p i
  | _, _ => leaves p i

theorem leavesFor_false (defs : Array Defn) (nodes : Array Node) (leaves : Bool → Nat → Option Nat) :
    leavesFor false defs nodes leaves = leaves := rfl

/-- The input nodes `B` reads at another width, with `recut`; none without. -/
def recutFor (recut : Bool) (defs : Array Defn) (B : Body) : Array (Nat × Nat) :=
  if recut then Widths.recutOf defs B else #[]

private def nodeKeys : Form → List String
  | .batch => ["args", "axes", "fn", "form", "n", "out"]
  | .scan => ["args", "fn", "form", "n", "n_xs", "out"]
  | _ => ["args", "fn", "form", "out"]

mutual

/-- Decode definition `fid` (core's `decode_program.get`), and first every definition it depends on, through at most `fuel`
nested definitions and static values; with `recut`, a body may read an input node at a width dividing its own (§4.4). -/
private def getDef (fuel : Nat) (desc : Desc) (fid : String) (recut : Bool := false) : D Nat := do
  let n + 1 := fuel | throw "definitions nest deeper than this verifier follows"
  if let some i := (← get).index[fid]? then return i
  if (← get).busy.contains fid then throw s!"definition '{fid}' depends on itself"
  let some d := desc.ddefs[fid]? | throw s!"KeyError: no definition {fid}"
  let form := (d.getObjValAs? String "form").toOption.getD ""
  let id := (d.getObjValAs? String "id").toOption.getD ""
  let defn ← if form == "primitive" then do
      if id != fid then throw s!"definition '{fid}': a primitive is keyed by its id, not '{id}'"
      let params ← liftV (paramsOf ((d.getObjVal? "params").toOption.getD Json.null) s!"primitive {fid}")
      let ret ← liftV (Ty.ofJson ((d.getObjVal? "ret").toOption.getD Json.null))
      let .value w := ret | throw s!"a primitive returns exactly one Value<w>; {fid} declares {ret.render}"
      let input := inputId? fid
      if let some (_, iw) := input then
        if !(params.isEmpty && w == iw) then throw s!"primitive {fid} signature differs from the registry"
      pure ({ key := fid, id, prim := true, params, ret, binding := input.map (·.1), gates := 1,
              paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := 1 } : Defn)
    else composite n desc fid d form id recut
  let s ← get
  set { s with defs := s.defs.push defn, index := s.index.insert fid s.defs.size, busy := s.busy.erase fid }
  return s.defs.size

private def composite (fuel : Nat) (desc : Desc) (fid : String) (d : Json) (form id : String) (recut : Bool := false) :
    D Defn := do
  let n + 1 := fuel | throw "definitions nest deeper than this verifier follows"
  let parts := id.splitOn "_v"
  let name := "_v".intercalate parts.dropLast
  let ver := parts.getLast!
  if !(form == "composite" && parts.length ≥ 2 && name.length > 0 && ver.length > 0 && ver.all Char.isDigit
      && s!"{name}_v{ver.toNat!}" == id) then
    throw s!"definition '{fid}': form '{form}', id '{id}' (a composite's id is Name_vN)"
  let bindings := (d.getObjVal? "bindings").toOption.getD Json.null
  let .obj bkv := bindings | throw s!"definition '{fid}': bindings"
  let names := bkv.toArray.map (·.1)
  let statics ← if names.size < 2 || !fid.contains '{' then pure names else do
    let order := (recordPairs fid).map (·.1)
    if order.qsort (· < ·) != names.qsort (· < ·) then
      throw s!"definition '{fid}': binding record keys {names.qsort (· < ·)} do not match its id"
    pure order
  let mut rec_ := #[]
  for k in statics do rec_ := rec_.push s!"{k}={← liftV (canon ((bindings.getObjVal? k).toOption.getD Json.null))}"
  let key := id ++ (if statics.isEmpty then "" else "{" ++ ",".intercalate rec_.toList ++ "}")
  modify fun s => { s with busy := s.busy.insert fid }
  if key != fid then throw s!"definition '{fid}': a composite is keyed by its id and binding record, '{key}'"
  for (_, v) in bkv.toArray do staticFns n desc v recut
  let b := (d.getObjVal? "body").toOption.getD Json.null
  let params ← liftV (paramsOf ((b.getObjVal? "params").toOption.getD Json.null) s!"definition '{fid}'")
  let retType ← liftV (Ty.ofJson ((b.getObjVal? "ret_type").toOption.getD Json.null))
  let .ok (.arr nds) := b.getObjVal? "nodes" | throw s!"definition '{fid}': body nodes"
  let outs ← liftV (nds.mapM fun nd => at_ desc.types ((nd.getObjVal? "out").toOption.getD Json.null) "types")
  let leaves := fun (p : Bool) (i : Nat) => if p then params[i]?.map (·.2.leaves) else outs[i]?.map (·.leaves)
  let mut nodes : Array Node := #[]
  let mut offsets := #[0]
  for k in [0:nds.size] do
    let nd := nds[k]!
    let pre := s!"definition '{fid}' node {k}"
    let fname ← liftV (at_ desc.callees ((nd.getObjVal? "fn").toOption.getD Json.null) "callees")
    let .str fname := fname | throw s!"KeyError: no definition {fname.compress}"
    let ci ← getDef n desc fname recut
    let st ← get
    let callee := st.defs[ci]!
    let fstr := (nd.getObjValAs? String "form").toOption.getD ""
    let keys : Array String := match nd with
      | .obj kvs => (kvs.toArray.map (fun (kv : String × Json) => kv.1)).qsort (· < ·)
      | _ => #[]
    let form ← match fstr with
      | "primitive" => pure Form.prim
      | "call" => pure Form.call
      | "batch" => pure Form.batch
      | "scan" => pure Form.scan
      | _ => throw s!"{pre}: form '{fstr}' with keys {keys}"
    if keys.toList != nodeKeys form then throw s!"{pre}: form '{fstr}' with keys {keys}"
    if (form == .prim && !callee.prim) then throw s!"{pre}: a primitive node does not call a composite ({callee.key})"
    if ((form == .call || form == .scan) && callee.prim) then throw s!"{pre}: a {fstr} node does not call a primitive ({callee.key})"
    let .ok (.arr ajs) := nd.getObjVal? "args" | throw s!"{pre}: args"
    let axes ← if form == .batch then
        match nd.getObjVal? "axes" with
        | .ok (.arr xs) => xs.mapM fun x => match x with
          | .null => pure none
          | x => match natJ? x with
            | some a => pure (some a)
            | none => throw s!"{pre}: batch counts"
        | _ => throw s!"{pre}: batch counts"
      else pure #[]
    let n ← if form == .batch || form == .scan then
        match (nd.getObjVal? "n").toOption.bind natJ? with
        | some n => pure n
        | none => throw s!"{pre}: {fstr} counts"
      else pure 1
    let nxs ← if form == .scan then
        match (nd.getObjVal? "n_xs").toOption.bind natJ? with
        | some x => if x < ajs.size then pure x else throw s!"{pre}: scan counts"
        | none => throw s!"{pre}: scan counts"
      else pure 0
    let earlier := fun (k2 a : Nat) => if k2 < nodes.size then nodes[k2]!.args[a]?.map (·.seq) else none
    let args ← liftV (ajs.mapM fun aj => do
      let ty ← at_ desc.types ((aj.getObjVal? "type").toOption.getD Json.null) "types"
      let ps ← match parseParts ((aj.getObjVal? "refs").toOption.getD Json.null) (leavesFor recut st.defs nodes leaves)
          earlier with
        | .ok ps => pure ps
        | .error e => throw s!"{pre}: {e}"
      let seq := Seq.ofParts ps
      if seq.count != ty.leaves then throw s!"{pre}: reference count {seq.count} does not match type {ty.render} ({ty.leaves} leaves)"
      if (Part.forwardOf ps k).isSome then throw s!"{pre}: {forwardRefusal k ((Part.forwardOf ps k).getD 0)}"
      pure ({ ty, seq } : Arg))
    let (out, args) ← liftV (buildNode fid k form callee args axes n nxs)
    if !(out == outs[k]!) then throw s!"{pre}: declares out {outs[k]!.render}, computes {out.render}"
    let node : Node := { form, fn := ci, args, out, axes, n, nxs }
    nodes := nodes.push node
    offsets := offsets.push (offsets.back! + (if form == .prim then 1 else node.members * callee.gates))
  let earlier := fun (k2 a : Nat) => nodes[k2]?.bind fun x => x.args[a]?.map (·.seq)
  let st ← get
  let ret ← match parseParts ((b.getObjVal? "ret").toOption.getD Json.null) (leavesFor recut st.defs nodes leaves) earlier with
    | .ok ps => pure (Seq.ofParts ps)
    | .error e => throw s!"definition '{fid}' ret: {e}"
  if ret.count != retType.leaves then
    throw s!"definition '{fid}' ret: reference count {ret.count} does not match type {retType.render} ({retType.leaves} leaves)"
  let body : Body := { params, retType, nodes, ret, offsets }
  match (Widths.bodyBad st.defs fid body recut).run st.walked with
  | .ok (some m) _ => throw m
  | .ok none w => set { st with walked := w }
  | .error e _ => throw (if e.startsWith "recut-" then e else s!"definition '{fid}': E_REINTERPRET unchecked: {e}")
  return { key := fid, id, prim := false, params, ret := retType, body := { body with recut := recutFor recut st.defs body },
           gates := offsets.back!, paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := retType.leaves }

/-- The functions a static value names (`static_of`): `{"fn": id}` decodes `id`. -/
private def staticFns (fuel : Nat) (desc : Desc) (v : Json) (recut : Bool := false) : D Unit := do
  let n + 1 := fuel | throw "definitions nest deeper than this verifier follows"
  match v with
  | .obj _ =>
    if let .ok f := v.getObjVal? "fn" then
      let .str f := f | throw s!"KeyError: no definition {f.compress}"
      discard (getDef n desc f recut)
    else if let .ok f := v.getObjVal? "f64" then
      let .str _ := f | throw s!"a float static is written as its hex string, not {f.compress}"
    else if let .ok d := v.getObjVal? "dict" then
      let .obj kvs := d | throw s!"a dict static is an object, not {d.compress}"
      for (_, x) in kvs.toArray do staticFns n desc x recut
  | .arr a => for x in a do staticFns n desc x recut
  | _ => pure ()

end

/-- §5.2 on the Definitions from place `lo`: an input node appears only in the root's body. -/
private def inputOutsideRoot (defs : Array Defn) (rootKey : String) (lo : Nat) : Option String := Id.run do
  for i in [lo:defs.size] do
    let d := defs[i]!
    if d.prim || d.key == rootKey then continue
    for k in [0:d.body.nodes.size] do
      let c := defs[d.body.nodes[k]!.fn]!
      if c.prim && isInputId c.key then
        return some s!"definition '{d.key}' node {k}: an input node ({c.key}) appears only in the root's body"
  return none

/-- §5.2 read through word `w`'s body `b` at place `bi`: a body that is an input primitive is an input node wherever `w` is
called, refused at the node of the least non-root composite key that calls `w`, its first, else as the body itself. -/
private def inputBody (defs : Array Defn) (rootKey : String) (w b : String) (bi : Nat) : Option String := Id.run do
  let bd := defs[bi]!
  if !(bd.prim && isInputId bd.key) then return none
  let mut at_ : Option (String × Nat) := none
  for d in defs do
    if d.prim || d.key == rootKey then continue
    match (List.range d.body.nodes.size).find? fun k => let c := defs[d.body.nodes[k]!.fn]!; c.prim && c.key == w with
    | some k => if at_.all (fun p => d.key < p.1) then at_ := some (d.key, k)
    | none => pure ()
  return some <| match at_ with
    | some (key, k) => s!"definition '{key}' node {k}: an input node ({b}, the Boolean body of {w}) appears only in the root's body"
    | none => s!"E_BODY_NOT_BOOLEAN: {w}'s body {b} is an input primitive, a value no claim registers"

/-- §5.2 read through word `w`'s body `b` at place `bi`: no Definition calls the root, so a body is not the root and reaches
no call of it, else it would read the root's input nodes. A body that reaches the root is Boolean only where the root is
(`reach`), so the walk runs only then. Each place below `defs.size` enters `todo` at most once, so `defs.size + 1`
iterations of one pop each empty it. -/
private def rootInBody (defs : Array Defn) (index : Std.HashMap String Nat) (rootKey : String) (reach : Array Bool)
    (w b : String) (bi : Nat) : Option String := Id.run do
  let some ri := index[rootKey]? | return none
  if !(reach[ri]?.getD false) then return none
  if bi == ri then return some s!"E_BODY_NOT_BOOLEAN: {w}'s body {b} is the root, which no Definition calls"
  let mut seen : Std.HashSet Nat := ({} : Std.HashSet Nat).insert bi
  let mut todo := #[bi]
  for _ in [0:defs.size + 1] do
    let some i := todo.back? | return none
    todo := todo.pop
    let some d := defs[i]? | continue
    if d.prim then continue
    for x in d.body.nodes do
      if x.fn == ri then
        return some s!"E_BODY_NOT_BOOLEAN: {w}'s body {b} calls the root {rootKey}, which no Definition calls"
      if x.fn < defs.size && !seen.contains x.fn then
        seen := seen.insert x.fn
        todo := todo.push x.fn
  return none

/-- `reach` extended to every Definition of `defs`: whether each primitive it reaches through node callees is a gate, a
constant bit or a 1-bit input (§9). A callee has a smaller place than its callers, so one pass in order suffices. -/
private def extendReach (defs : Array Defn) (reach : Array Bool) : Array Bool := Id.run do
  let mut r := reach
  for i in [reach.size:defs.size] do
    let d := defs[i]!
    let here := r
    r := r.push (if d.prim then isBooleanPrim d.key else d.body.nodes.all fun x =>
      let c := defs[x.fn]!
      if c.prim then isBooleanPrim c.key else here[x.fn]?.getD false)
  return r

/-- Each of `fs` that `definitions` has, decoded by `getDef`. -/
private def decodeAll (desc : Desc) (fs : Array String) : D Unit := do
  for f in fs do
    if desc.ddefs.contains f then discard (getDef Steps.FUEL desc f)

/-- An id spelt as a member of the input families or the gates' (`Input`, `RegisteredInput`, `SuppliedInput`, `And` or `Xor`, then a
digit) that is no member's id: a family member spelt otherwise than its id (§4.6), which no registry holds. -/
def misspelt (id : String) : Bool :=
  ["Input", "RegisteredInput", "SuppliedInput", "And", "Xor"].any (fun pre =>
      id.startsWith pre && (id.drop pre.length).toString.front.isDigit) &&
    !isInputId id && (gateKind? id).isNone

/-- §4.9's keys (`codec._decode_boolean`), once every primitive entry is decoded: each gate's signature §9's and no
primitive a family member spelt otherwise, then `E_BODY_KEY`, `E_BODY_MISSING` and `E_BODY_KEY` in the format's order;
the map's entries in key order. -/
def booleanKeys (defs : Array Defn) (index : Std.HashMap String Nat) (prims : Array String) (bm : Json) :
    V (Array (String × String)) := do
  for d in defs do
    if d.prim && misspelt d.key then
      throw s!"primitive {d.key} spells a family member otherwise than its id (§4.6)"
    if let some g := (if d.prim then gateKind? d.key else none) then
      let (ps, r) := g.signature
      if !(d.params == ps && d.ret == r) then throw s!"primitive {d.key} signature differs from the registry"
  let .obj kvs := bm | throw "E_BODY_KEY: `boolean` is not an object of descriptor ids"
  let some pairs := kvs.toArray.mapM (fun (k, v) => match v with | .str s => some (k, s) | _ => none)
    | throw "E_BODY_KEY: `boolean` is not an object of descriptor ids"
  let word := prims.filter fun f => (index[f]?.map fun i => isWordPrim defs[i]!.prim f).getD false
  let keys : Std.HashSet String := pairs.foldl (fun s p => s.insert p.1) {}
  let missing := word.filter (!keys.contains ·)
  if !missing.isEmpty then
    throw s!"E_BODY_MISSING: the word primitives {missing.toList.take 4} have no Boolean body in `boolean`"
  let words : Std.HashSet String := word.foldl (·.insert ·) {}
  let extra := (pairs.filter (!words.contains ·.1)).map (·.1)
  if !extra.isEmpty then
    throw s!"E_BODY_KEY: `boolean` names {extra.toList.take 4}, which are no word primitive among `definitions`"
  return pairs

/-- `E_BODY_NOT_BOOLEAN` for the body of word `w` at place `bi`, just decoded: no input node from place `lo` on, and no
input primitive as the body (§5.2, `inputBody`), not the root and no call of it (`rootInBody`), Boolean (`reach`, its
primitives gates, constant bits or 1-bit inputs) and of its word's bit signature. -/
def checkBody (defs : Array Defn) (index : Std.HashMap String Nat) (rootKey : String) (lo : Nat) (reach : Array Bool)
    (w b : String) (bi : Nat) : V Unit := do
  if let some e := inputOutsideRoot defs rootKey lo <|> inputBody defs rootKey w b bi <|>
      rootInBody defs index rootKey reach w b bi then throw e
  let bd := defs[bi]!
  let wd := defs[index[w]?.getD 0]!
  let boolean := if bd.prim then isBooleanPrim bd.key
    else bd.params.all (·.2.bits) && bd.ret.bits && reach[bi]!
  if !(boolean && bd.params.map (·.2) == wd.params.map (·.2.bitType) && bd.ret == wd.ret.bitType) then
    throw s!"E_BODY_NOT_BOOLEAN: {w}'s body {b} is not a Boolean Definition of its bit signature"

/-- §4.9's bodies, in key order: each refused if `definitions` lacks it, else decoded by `getDef` and checked
(`checkBody`) before the next. `reach` grows with the Definitions as they are pushed, once each. -/
private def decodeBodies (desc : Desc) (rootKey : String) (pairs : Array (String × String)) : D Unit := do
  let mut reach : Array Bool := #[]
  for (w, b) in pairs do
    if !desc.ddefs.contains b then throw s!"E_BODY_NOT_BOOLEAN: {w}'s body '{b}' is no definition"
    let lo := (← get).defs.size
    let bi ← getDef Steps.FUEL desc b
    let s ← get
    reach := extendReach s.defs reach
    liftV (checkBody s.defs s.index rootKey lo reach w b bi)

/-- The primitive entries of `definitions`, in key order. -/
private def primEntries (dkv : Array (String × Json)) : Array String :=
  dkv.filterMap fun (k, d) => if (d.getObjValAs? String "form").toOption == some "primitive" then some k else none

/-- The Definitions the root and the program record's functions need, decoded (`decode_program`'s walk), each body read as
`getDef` reads it with `recut`. -/
private def goOf (desc : Desc) (rootId programFn : String) (recut : Bool) : D Nat := do
  let r ← getDef Steps.FUEL desc rootId recut
  for f in recordFunctionIds programFn do
    if desc.ddefs.contains f then discard (getDef Steps.FUEL desc f recut)
  return r

/-- The codec's refusal of a Program whose Definitions the plain walk refuses with `e`, from the walk `r` that reads an input
node at a width dividing its own (§4.4) and reads none in its root: its refusal; else the root's and the input nodes'
(`decode`'s, in its order); else `e`. -/
private def recutRefusal (e rootId : String) (r : Except String (Nat × DecState)) : String :=
  match r with
  | .error e' => e'
  | .ok (root, st) =>
    let rd := st.defs[root]!
    if rd.prim || !rd.params.isEmpty then s!"root '{rootId}': the root is a composite without parameters" else
    let input := st.defs.toList.findSome? fun d => if d.prim || d.key == rootId then none else
      (List.range d.body.nodes.size).findSome? fun k =>
        let c := st.defs[d.body.nodes[k]!.fn]!
        if c.prim && (isInputId c.key || (c.key.startsWith "Registered" && isInputId (c.key.drop 10).toString)) then
          some s!"definition '{d.key}' node {k}: an input node ({c.key}) appears only in the root's body"
        else none
    input.getD e

/-- One of the codec's port checks, the keys of a port (`ports.py` `_port_from_json`): the first port of definition `fid`'s
`ports` rows (§12) that is not a non-empty object of a parameter's or an input node's keys (`bound`, `kind`, `receipt`,
`word`, `certified`) or the return's (`interval`, `sink`, `word`, `certified`). Lean decodes no other part of a port row. -/
def portKeysError (fid : String) (d : Json) : Option String := Id.run do
  let ports := (d.getObjVal? "ports").toOption.getD .null
  for (arr, keys, part) in [("params", ["bound", "kind", "receipt", "word", "certified"], "parameter"),
      ("ret", ["interval", "sink", "word", "certified"], "return"),
      ("inputs", ["bound", "kind", "receipt", "word", "certified"], "input node")] do
    let .ok (.arr rows) := ports.getObjVal? arr | continue
    for r in rows do
      let .arr xs := r | continue
      let some p := xs.back? | continue
      let ks := match p with
        | .obj kvs => kvs.keysArray
        | _ => #[]
      if ks.isEmpty || ks.any (!keys.contains ·) then
        return some s!"E_PORT_DECL: {fid} {part}: a port is a non-empty object with keys from {keys}, not {p.compress}"
  return none

/-- The decoded Definitions and the root's place: the plain walk's, every body read at its own widths, where it accepts; where
it refuses, the walk that reads an input node at a width dividing its own (§4.4) where that one accepts with its root reading
one (`Body.recut`), else the codec's refusal (`recutRefusal`). So a decoded Program is the plain walk's or has a recut root. -/
private def decodeDefs (desc : Desc) (rootId programFn : String) : V (Nat × DecState) :=
  match (goOf desc rootId programFn false).run {} with
  | .ok r => .ok r
  | .error e => match (goOf desc rootId programFn true).run {} with
    | .ok (root, st) =>
      if (st.defs[root]!).body.recut.isEmpty then .error (recutRefusal e rootId (.ok (root, st))) else .ok (root, st)
    | r => .error (recutRefusal e rootId r)

/-- Decoded Definitions are one walk's. -/
private theorem decodeDefs_walk {desc : Desc} {rootId programFn : String} {r : Nat × DecState}
    (h : decodeDefs desc rootId programFn = .ok r) : ∃ recut, (goOf desc rootId programFn recut).run {} = .ok r := by
  unfold decodeDefs at h
  split at h
  next r' hp => exact ⟨false, hp.trans h⟩
  next =>
    split at h
    next root' st' hp =>
      split at h
      · cases h
      · exact ⟨true, hp.trans h⟩
    next => cases h

/-- Decoded Definitions whose root reads no input node at another width are the plain walk's. -/
private theorem decodeDefs_plain {desc : Desc} {rootId programFn : String} {root : Nat} {st : DecState}
    (h : decodeDefs desc rootId programFn = .ok (root, st)) (hr : (st.defs[root]!).body.recut = #[]) :
    (goOf desc rootId programFn false).run {} = .ok (root, st) := by
  unfold decodeDefs at h
  split at h
  next r hp => rw [hp]; exact h
  next =>
    split at h
    next root' st' _ =>
      split at h
      · cases h
      · rename_i hne
        have h0 := Except.ok.inj h
        have h1 : root' = root := congrArg Prod.fst h0
        have h2 : st' = st := congrArg Prod.snd h0
        rw [h1, h2, hr] at hne
        exact absurd rfl hne
    next => cases h

/-- `decode_program` on a descriptor's bytes: the refusals of PROTOCOL.md §4.6 (a primitive's signature is the descriptor's own,
see the module docstring), then the program's SHA-512. -/
def decode (bytes : ByteArray) : V Prog := do
  let some text := String.fromUTF8? bytes | throw "program: not UTF-8"
  let j ← match JsonParse.parse text with
    | .ok j => pure j
    | .error e => throw s!"program: not JSON: {e}"
  let .obj top := j | throw "program: not an object"
  match (j.getObjValAs? String "schema").toOption with
  | some s => if s != SCHEMA then throw s!"E_SCHEMA: unexpected descriptor schema '{s}' (this verifier reads {SCHEMA})"
  | none => throw s!"E_SCHEMA: unexpected descriptor schema (this verifier reads {SCHEMA})"
  let .ok (.obj dkv) := j.getObjVal? "definitions" | throw "program: definitions"
  for (fid, d) in dkv.toArray do
    let .obj _ := d | throw s!"definition '{fid}': not an object"
    if let some e := portKeysError fid d then throw e
  -- the free-form parts (`codec._free_form`): every top-level field but definitions and annotations, and each definition's key,
  -- id, bindings, params, ret, and its body's params and ret_type
  let freeDefs := dkv.toArray.map fun (fid, d) => (fid, match d with
    | .obj kvs => Json.mkObj ((kvs.toArray.filter (·.1 != "body")).toList ++
        ((d.getObjVal? "body").toOption.map (fun b => ["params", "ret_type"].filterMap fun k =>
          (b.getObjVal? k).toOption.map (k, ·))).getD [])
    | x => x)
  let free := Json.mkObj ((top.toArray.filter (fun (k, _) => k != "definitions" && k != "annotations")).toList ++
    [("definitions", Json.mkObj freeDefs.toList)])
  if let some e := domainError free then throw s!"descriptor outside the canonical JSON domain (PROTOCOL.md §1): {e}"
  let .ok (.arr callees) := j.getObjVal? "callees" | throw "program: callees"
  let .ok (.arr tys) := j.getObjVal? "types" | throw "program: types"
  let types ← tys.mapM Ty.ofJson
  let desc : Desc := { ddefs := dkv.toArray.foldl (fun m (k, v) => m.insert k v) {}, callees, types }
  let some rootId := (j.getObjValAs? String "root").toOption | throw "program: root"
  let some programFn := (j.getObjValAs? String "program_fn").toOption | throw "program: program_fn"
  let (root, st) ← decodeDefs desc rootId programFn
  let rd := st.defs[root]!
  if rd.prim || !rd.params.isEmpty then throw s!"root '{rootId}': the root is a composite without parameters"
  -- an input node (`Input<w>_v1` or `RegisteredInput<w>_v1`) appears only in the root's body (PROTOCOL.md §5.2)
  for d in st.defs do
    if d.prim || d.key == rootId then continue
    for k in [0:d.body.nodes.size] do
      let c := st.defs[d.body.nodes[k]!.fn]!
      if c.prim && (isInputId c.key || (c.key.startsWith "Registered" && isInputId (c.key.drop 10).toString)) then
        throw s!"definition '{d.key}' node {k}: an input node ({c.key}) appears only in the root's body"
  let prims := primEntries dkv.toArray
  let bm := (j.getObjVal? "boolean").toOption.getD .null
  let ((), st) ← (decodeAll desc prims).run st
  let pairs ← booleanKeys st.defs st.index prims bm
  let ((), st) ← (decodeBodies desc rootId pairs).run st
  let boolean := pairs.foldl (fun m p => m.insert p.1 p.2) {}
  let bare := Json.mkObj (top.toArray.filter (·.1 != "annotations")).toList
  return { defs := st.defs, index := st.index, root, programFn, sha512 := Sha512.hash (← canon bare).toUTF8, boolean }


/-- A descriptor's bytes read as JSON: none for bytes that are not UTF-8, else `JsonParse.parse`'s verdict on their text. Every
reader of a descriptor starts with it (`decode_eq_of`). -/
def readJson (bytes : ByteArray) : Option (Except String Json) := (String.fromUTF8? bytes).map JsonParse.parse

/-- `decode` from the descriptor's bytes as `readJson` read them (`decode_eq_of`). -/
def decodeOf (read : Option (Except String Json)) : V Prog := do
  let some parsed := read | throw "program: not UTF-8"
  let j ← match parsed with
    | .ok j => pure j
    | .error e => throw s!"program: not JSON: {e}"
  let .obj top := j | throw "program: not an object"
  match (j.getObjValAs? String "schema").toOption with
  | some s => if s != SCHEMA then throw s!"E_SCHEMA: unexpected descriptor schema '{s}' (this verifier reads {SCHEMA})"
  | none => throw s!"E_SCHEMA: unexpected descriptor schema (this verifier reads {SCHEMA})"
  let .ok (.obj dkv) := j.getObjVal? "definitions" | throw "program: definitions"
  for (fid, d) in dkv.toArray do
    let .obj _ := d | throw s!"definition '{fid}': not an object"
    if let some e := portKeysError fid d then throw e
  -- the free-form parts (`codec._free_form`): every top-level field but definitions and annotations, and each definition's key,
  -- id, bindings, params, ret, and its body's params and ret_type
  let freeDefs := dkv.toArray.map fun (fid, d) => (fid, match d with
    | .obj kvs => Json.mkObj ((kvs.toArray.filter (·.1 != "body")).toList ++
        ((d.getObjVal? "body").toOption.map (fun b => ["params", "ret_type"].filterMap fun k =>
          (b.getObjVal? k).toOption.map (k, ·))).getD [])
    | x => x)
  let free := Json.mkObj ((top.toArray.filter (fun (k, _) => k != "definitions" && k != "annotations")).toList ++
    [("definitions", Json.mkObj freeDefs.toList)])
  if let some e := domainError free then throw s!"descriptor outside the canonical JSON domain (PROTOCOL.md §1): {e}"
  let .ok (.arr callees) := j.getObjVal? "callees" | throw "program: callees"
  let .ok (.arr tys) := j.getObjVal? "types" | throw "program: types"
  let types ← tys.mapM Ty.ofJson
  let desc : Desc := { ddefs := dkv.toArray.foldl (fun m (k, v) => m.insert k v) {}, callees, types }
  let some rootId := (j.getObjValAs? String "root").toOption | throw "program: root"
  let some programFn := (j.getObjValAs? String "program_fn").toOption | throw "program: program_fn"
  let (root, st) ← decodeDefs desc rootId programFn
  let rd := st.defs[root]!
  if rd.prim || !rd.params.isEmpty then throw s!"root '{rootId}': the root is a composite without parameters"
  -- an input node (`Input<w>_v1` or `RegisteredInput<w>_v1`) appears only in the root's body (PROTOCOL.md §5.2)
  for d in st.defs do
    if d.prim || d.key == rootId then continue
    for k in [0:d.body.nodes.size] do
      let c := st.defs[d.body.nodes[k]!.fn]!
      if c.prim && (isInputId c.key || (c.key.startsWith "Registered" && isInputId (c.key.drop 10).toString)) then
        throw s!"definition '{d.key}' node {k}: an input node ({c.key}) appears only in the root's body"
  let prims := primEntries dkv.toArray
  let bm := (j.getObjVal? "boolean").toOption.getD .null
  let ((), st) ← (decodeAll desc prims).run st
  let pairs ← booleanKeys st.defs st.index prims bm
  let ((), st) ← (decodeBodies desc rootId pairs).run st
  let boolean := pairs.foldl (fun m p => m.insert p.1 p.2) {}
  let bare := Json.mkObj (top.toArray.filter (·.1 != "annotations")).toList
  return { defs := st.defs, index := st.index, root, programFn, sha512 := Sha512.hash (← canon bare).toUTF8, boolean }

set_option maxRecDepth 20000 in
theorem decode_eq_of (bytes : ByteArray) : decode bytes = decodeOf (readJson bytes) := by
  unfold decode decodeOf readJson
  generalize String.fromUTF8? bytes = o
  cases o <;> rfl

/-- `decode` in a verdict monad `M`, the program's SHA-512 by `sha512` (`Flock.With.Setup`). It is here because `decode`'s
decoder state is this module's own. -/
def decodeWith {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (sha512 : ByteArray → M ByteArray)
    (bytes : ByteArray) : M Prog := do
  let some text := String.fromUTF8? bytes | throw "program: not UTF-8"
  let j ← match JsonParse.parse text with
    | .ok j => pure j
    | .error e => throw s!"program: not JSON: {e}"
  let .obj top := j | throw "program: not an object"
  match (j.getObjValAs? String "schema").toOption with
  | some s => if s != SCHEMA then throw s!"E_SCHEMA: unexpected descriptor schema '{s}' (this verifier reads {SCHEMA})"
  | none => throw s!"E_SCHEMA: unexpected descriptor schema (this verifier reads {SCHEMA})"
  let .ok (.obj dkv) := j.getObjVal? "definitions" | throw "program: definitions"
  for (fid, d) in dkv.toArray do
    let .obj _ := d | throw s!"definition '{fid}': not an object"
    if let some e := portKeysError fid d then throw e
  let freeDefs := dkv.toArray.map fun (fid, d) => (fid, match d with
    | .obj kvs => Json.mkObj ((kvs.toArray.filter (·.1 != "body")).toList ++
        ((d.getObjVal? "body").toOption.map (fun b => ["params", "ret_type"].filterMap fun k =>
          (b.getObjVal? k).toOption.map (k, ·))).getD [])
    | x => x)
  let free := Json.mkObj ((top.toArray.filter (fun (k, _) => k != "definitions" && k != "annotations")).toList ++
    [("definitions", Json.mkObj freeDefs.toList)])
  if let some e := domainError free then throw s!"descriptor outside the canonical JSON domain (PROTOCOL.md §1): {e}"
  let .ok (.arr callees) := j.getObjVal? "callees" | throw "program: callees"
  let .ok (.arr tys) := j.getObjVal? "types" | throw "program: types"
  let types ← (tys.mapM Ty.ofJson : V _)
  let desc : Desc := { ddefs := dkv.toArray.foldl (fun m (k, v) => m.insert k v) {}, callees, types }
  let some rootId := (j.getObjValAs? String "root").toOption | throw "program: root"
  let some programFn := (j.getObjValAs? String "program_fn").toOption | throw "program: program_fn"
  let (root, st) ← (decodeDefs desc rootId programFn : V _)
  let rd := st.defs[root]!
  if rd.prim || !rd.params.isEmpty then throw s!"root '{rootId}': the root is a composite without parameters"
  for d in st.defs do
    if d.prim || d.key == rootId then continue
    for k in [0:d.body.nodes.size] do
      let c := st.defs[d.body.nodes[k]!.fn]!
      if c.prim && (isInputId c.key || (c.key.startsWith "Registered" && isInputId (c.key.drop 10).toString)) then
        throw s!"definition '{d.key}' node {k}: an input node ({c.key}) appears only in the root's body"
  let prims := primEntries dkv.toArray
  let bm := (j.getObjVal? "boolean").toOption.getD .null
  let ((), st) ← ((decodeAll desc prims).run st : V _)
  let pairs ← (booleanKeys st.defs st.index prims bm : V _)
  let ((), st) ← ((decodeBodies desc rootId pairs).run st : V _)
  let boolean := pairs.foldl (fun m p => m.insert p.1 p.2) {}
  let bare := Json.mkObj (top.toArray.filter (·.1 != "annotations")).toList
  return { defs := st.defs, index := st.index, root, programFn, sha512 := ← sha512 (← canon bare).toUTF8, boolean }

end Flock.Program
