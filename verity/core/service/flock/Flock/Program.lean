/-
PROTOCOL.md §16.7: the program format `verity-ir/descriptor/v1` (core's `verity/ir/PROTOCOL.md`, PR #120), as the verifier
reads its own copy of a Program: the canonical JSON domain and the program's SHA-512 (§1), types and leaves (§2), reference
sequences (§3), the descriptor and what the decoder refuses (§4), input nodes (§5), the canonical layout (§6) and scope
resolution (§7).

The verifier evaluates no primitive, so it holds no registry: a primitive is taken as the descriptor declares it (its parameters
and return type are under the program's SHA-512), and an input (`Input<w>_v1`, bound per run, or `RegisteredInput<w>_v1`,
bound at registration; §5.2) must declare no parameters and `Value<w>`. Definitions
are decoded on demand from the root and from the functions `program_fn`'s record names, as core decodes them, and each callee
gets a smaller index than its callers.
-/
import Lean.Data.Json
import Std.Data.HashMap
import Flock.Canon
import Flock.Hash
import Flock.Sizes
import Flock.Steps

namespace Flock.Program

open Lean (Json JsonNumber)

def SCHEMA : String := "verity-ir/descriptor/v1"
def MAX_SAFE_INT : Nat := 2 ^ 53 - 1

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

end Ty

/-! ### §3 Reference sequences -/

/-- A body-scoped reference: space `p` (a parameter of the body) when `p`, else `n` (a node's output); the collection; the leaf. -/
structure Ref where
  p : Bool
  i : Nat
  j : Nat
  deriving Inhabited, BEq

def Ref.toJson (r : Ref) : Json := Json.arr #[Json.str (if r.p then "p" else "n"), Lean.toJson r.i, Lean.toJson r.j]

inductive Part where
  | affine (p : Bool) (i base count : Nat)
  | strided (p : Bool) (i : Nat) (base : Int) (inner : Nat) (os is : Int) (count : Nat)
  | explicit (items : Array Ref)
  deriving Inhabited

def Part.count : Part → Nat
  | .affine _ _ _ c => c
  | .strided _ _ _ _ _ _ c => c
  | .explicit it => it.size

@[inline] def Part.get : Part → Nat → Ref
  | .affine p i b _, k => ⟨p, i, b + k⟩
  | .strided p i b inner os is _, k => ⟨p, i, (b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is).toNat⟩
  | .explicit it, k => it[k]!

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
  throw s!"references {j.compress}: not affine, strided, explicit, concat or alias"

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
  deriving Inhabited

def RUN : String := "run"
def REGISTRATION : String := "registration"

private def widthAfter? (pre id : String) : Option Nat :=
  if !(id.startsWith pre && id.endsWith "_v1") then none else
  let w := ((id.drop pre.length).dropEnd 3).toString
  if w.length > 0 && w.all Char.isDigit && w.front != '0' then w.toNat? else none

/-- `program.input_id`: an input id's binding time and width: `run` for `^Input([1-9][0-9]*)_v1$` (`program.INPUT_ID`),
`registration` for `^RegisteredInput([1-9][0-9]*)_v1$` (`program.REGISTERED_ID`); none for any other id. -/
def inputId? (id : String) : Option (String × Nat) :=
  match widthAfter? "Input" id with
  | some w => some (RUN, w)
  | none => (widthAfter? "RegisteredInput" id).map (REGISTRATION, ·)

def inputWidth? (id : String) : Option Nat := (inputId? id).map (·.2)

def isInputId (id : String) : Bool := (inputId? id).isSome

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
  (recordPairs fid).foldl (fun acc (_, v) => match Json.parse v with
    | .ok j => fnRefs acc j
    | .error _ => acc) #[]

/-! ### §4.4 A leaf read at the width it was committed with

`E_REINTERPRET` (`defs.Builder.finish`): a node's argument or a body's return that names a leaf of a parameter or of an
earlier node (any node, for the return) of another width than its own type gives that position is a silent
reinterpretation, and the decoder refuses the Definition as it finishes decoding it, with the codec's message. A reference
to the node itself or a later one is not this check's. The check reads the description: each reference part is compared
by the structure of the two types, and leaves are walked one by one only where the structure doesn't decide, at most
`WALK` of them in a Program, past which it is refused (`E_REINTERPRET unchecked`). -/

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
  else if count ≤ cap then some ((List.range count).map fun k => k * s % L).eraseDups else none

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

def Src.ty? (s : Src) (p : Bool) (i : Nat) : Option Ty :=
  if p then s.params[i]? else if i < s.below then s.nodes[i]? else none

def Src.homo? (s : Src) (p : Bool) (i : Nat) : Option Nat :=
  if p then (s.hp[i]?).join else (s.hn[i]?).join

/-- Whether the references of part `q`, read as `t`'s leaves `a …` (`t`'s one width `ht`, if it has one), name leaves of
the widths `t` gives them. -/
def partOk (s : Src) (t : Ty) (ht : Option Nat) (a : Nat) (q : Part) : M Bool := do
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
    let at_ := fun (k : Nat) => (base + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is).toNat
    if let some w := ht then
      let q' := (n - 1) / inner
      let corners : List (Nat × Nat) := [(0, 0), (0, min inner n - 1), (q', 0), (q', (n - 1) % inner)] ++
        (if q' > 0 then [(q' - 1, inner - 1)] else [])
      let vals := corners.map fun (x, y) => (base + (x : Int) * os + (y : Int) * is).toNat
      let lo := vals.foldl min vals.head!
      let hi := vals.foldl max vals.head!
      if rangeWidths S lo (hi + 1) == [w] then return true
      let (cs, off) := cover S lo (hi + 1)
      if let .array _ f := cs then
        let L := f.leaves
        if L > 0 then
          if let (some A, some B) := (residues os ((n + inner - 1) / inner) L 4096, residues is inner L 4096) then
            if A.length * B.length ≤ 4096 then
              let r0 := (base - (off : Int)) % (L : Int)
              if A.all fun x => B.all fun y => wAt f ((r0 + x + y) % (L : Int)).toNat == w then return true
    for k in [0:n] do
      step
      if wAt S (at_ k) != wAt t (a + k) then return false
    return true

/-- The first leaf of part `q`, read as `t`'s leaves `a …`, that names a leaf of another width than `t` gives it, as an
offset into `q`, where `partOk` found one: an affine part's by bisection over its prefixes (`eqRange` on each), any
other's leaf by leaf. -/
def firstBad (s : Src) (t : Ty) (a : Nat) (q : Part) : M (Option Nat) := do
  let bad := fun (k : Nat) => let r := q.get k
    match s.ty? r.p r.i with
    | some S => wAt S r.j != wAt t (a + k)
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

/-- The first part of the references `seq`, read as a `t`, that names leaves of another width, with the leaf it starts at. -/
def seqBad (s : Src) (t : Ty) (seq : Seq) : M (Option (Nat × Part)) := do
  let ht := homo t
  for q in seq.parts, a in seq.starts do
    if !(← partOk s t ht a q) then return some (a, q)
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
arguments, then the return), as its refusal; `defs` holds its callees. -/
def bodyBad (defs : Array Defn) (key : String) (B : Body) : M (Option String) := do
  let outs := B.nodes.map (·.out)
  let s : Src := { params := B.params.map (·.2), nodes := outs, hp := B.params.map (homo ·.2), hn := outs.map homo,
                   below := 0 }
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
  ps.mapM fun p => match p with
    | .arr #[.str k, t] => do pure (k, ← Ty.ofJson t)
    | _ => throw s!"{what}: a parameter is [name, type]"

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

private def nodeKeys : Form → List String
  | .batch => ["args", "axes", "fn", "form", "n", "out"]
  | .scan => ["args", "fn", "form", "n", "n_xs", "out"]
  | _ => ["args", "fn", "form", "out"]

mutual

/-- Decode definition `fid` (core's `decode_program.get`), and first every definition it depends on, through at most `fuel`
nested definitions and static values. -/
private def getDef (fuel : Nat) (desc : Desc) (fid : String) : D Nat := do
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
    else composite n desc fid d form id
  let s ← get
  set { s with defs := s.defs.push defn, index := s.index.insert fid s.defs.size, busy := s.busy.erase fid }
  return s.defs.size

private def composite (fuel : Nat) (desc : Desc) (fid : String) (d : Json) (form id : String) : D Defn := do
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
  for (_, v) in bkv.toArray do staticFns n desc v
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
    let ci ← getDef n desc fname
    let callee := (← get).defs[ci]!
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
      let ps ← match parseParts ((aj.getObjVal? "refs").toOption.getD Json.null) leaves earlier with
        | .ok ps => pure ps
        | .error e => throw s!"{pre}: {e}"
      let seq := Seq.ofParts ps
      if seq.count != ty.leaves then throw s!"{pre}: reference count {seq.count} does not match type {ty.render} ({ty.leaves} leaves)"
      pure ({ ty, seq } : Arg))
    let (out, args) ← liftV (buildNode fid k form callee args axes n nxs)
    if !(out == outs[k]!) then throw s!"{pre}: declares out {outs[k]!.render}, computes {out.render}"
    let node : Node := { form, fn := ci, args, out, axes, n, nxs }
    nodes := nodes.push node
    offsets := offsets.push (offsets.back! + (if form == .prim then 1 else node.members * callee.gates))
  let earlier := fun (k2 a : Nat) => nodes[k2]?.bind fun x => x.args[a]?.map (·.seq)
  let ret ← match parseParts ((b.getObjVal? "ret").toOption.getD Json.null) leaves earlier with
    | .ok ps => pure (Seq.ofParts ps)
    | .error e => throw s!"definition '{fid}' ret: {e}"
  if ret.count != retType.leaves then
    throw s!"definition '{fid}' ret: reference count {ret.count} does not match type {retType.render} ({retType.leaves} leaves)"
  let body : Body := { params, retType, nodes, ret, offsets }
  let st ← get
  match (Widths.bodyBad st.defs fid body).run st.walked with
  | .ok (some m) _ => throw m
  | .ok none w => set { st with walked := w }
  | .error e _ => throw s!"definition '{fid}': E_REINTERPRET unchecked: {e}"
  return { key := fid, id, prim := false, params, ret := retType, body,
           gates := offsets.back!, paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := retType.leaves }

/-- The functions a static value names (`static_of`): `{"fn": id}` decodes `id`. -/
private def staticFns (fuel : Nat) (desc : Desc) (v : Json) : D Unit := do
  let n + 1 := fuel | throw "definitions nest deeper than this verifier follows"
  match v with
  | .obj _ =>
    if let .ok f := v.getObjVal? "fn" then
      let .str f := f | throw s!"KeyError: no definition {f.compress}"
      discard (getDef n desc f)
    else if let .ok f := v.getObjVal? "f64" then
      let .str _ := f | throw s!"a float static is written as its hex string, not {f.compress}"
    else if let .ok d := v.getObjVal? "dict" then
      let .obj kvs := d | throw s!"a dict static is an object, not {d.compress}"
      for (_, x) in kvs.toArray do staticFns n desc x
  | .arr a => for x in a do staticFns n desc x
  | _ => pure ()

end

/-- `decode_program` on a descriptor's bytes: the refusals of PROTOCOL.md §4.6 (a primitive's signature is the descriptor's own,
see the module docstring), then the program's SHA-512. -/
def decode (bytes : ByteArray) : V Prog := do
  let some text := String.fromUTF8? bytes | throw "program: not UTF-8"
  let j ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"program: not JSON: {e}"
  let .obj top := j | throw "program: not an object"
  match (j.getObjValAs? String "schema").toOption with
  | some s => if s != SCHEMA then throw s!"unexpected descriptor schema '{s}' (this verifier reads {SCHEMA})"
  | none => throw s!"unexpected descriptor schema (this verifier reads {SCHEMA})"
  let .ok (.obj dkv) := j.getObjVal? "definitions" | throw "program: definitions"
  for (fid, d) in dkv.toArray do
    let .obj _ := d | throw s!"definition '{fid}': not an object"
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
  let go : D Nat := do
    let r ← getDef Steps.FUEL desc rootId
    for f in recordFunctionIds programFn do
      if desc.ddefs.contains f then discard (getDef Steps.FUEL desc f)
    return r
  let (root, st) ← go.run {}
  let rd := st.defs[root]!
  if rd.prim || !rd.params.isEmpty then throw s!"root '{rootId}': the root is a composite without parameters"
  -- an input node (`Input<w>_v1` or `RegisteredInput<w>_v1`) appears only in the root's body (PROTOCOL.md §5.2)
  for d in st.defs do
    if d.prim || d.key == rootId then continue
    for k in [0:d.body.nodes.size] do
      let c := st.defs[d.body.nodes[k]!.fn]!
      if c.prim && (isInputId c.key || (c.key.startsWith "Registered" && isInputId (c.key.drop 10).toString)) then
        throw s!"definition '{d.key}' node {k}: an input node ({c.key}) appears only in the root's body"
  let bare := Json.mkObj (top.toArray.filter (·.1 != "annotations")).toList
  return { defs := st.defs, index := st.index, root, programFn, sha512 := Sha512.hash (← canon bare).toUTF8 }

/-- `decode` in a verdict monad `M`, the program's SHA-512 by `sha512` (`Flock.With.Setup`). It is here because `decode`'s
decoder state is this module's own. -/
def decodeWith {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (sha512 : ByteArray → M ByteArray)
    (bytes : ByteArray) : M Prog := do
  let some text := String.fromUTF8? bytes | throw "program: not UTF-8"
  let j ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"program: not JSON: {e}"
  let .obj top := j | throw "program: not an object"
  match (j.getObjValAs? String "schema").toOption with
  | some s => if s != SCHEMA then throw s!"unexpected descriptor schema '{s}' (this verifier reads {SCHEMA})"
  | none => throw s!"unexpected descriptor schema (this verifier reads {SCHEMA})"
  let .ok (.obj dkv) := j.getObjVal? "definitions" | throw "program: definitions"
  for (fid, d) in dkv.toArray do
    let .obj _ := d | throw s!"definition '{fid}': not an object"
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
  let go : D Nat := do
    let r ← getDef Steps.FUEL desc rootId
    for f in recordFunctionIds programFn do
      if desc.ddefs.contains f then discard (getDef Steps.FUEL desc f)
    return r
  let (root, st) ← (go.run {} : V _)
  let rd := st.defs[root]!
  if rd.prim || !rd.params.isEmpty then throw s!"root '{rootId}': the root is a composite without parameters"
  for d in st.defs do
    if d.prim || d.key == rootId then continue
    for k in [0:d.body.nodes.size] do
      let c := st.defs[d.body.nodes[k]!.fn]!
      if c.prim && (isInputId c.key || (c.key.startsWith "Registered" && isInputId (c.key.drop 10).toString)) then
        throw s!"definition '{d.key}' node {k}: an input node ({c.key}) appears only in the root's body"
  let bare := Json.mkObj (top.toArray.filter (·.1 != "annotations")).toList
  return { defs := st.defs, index := st.index, root, programFn, sha512 := ← sha512 (← canon bare).toUTF8 }

/-! ### §6 The canonical layout and §7 scope resolution -/

/-- An activation: the root (no parameters), a Call cut as its Definition's activation in a one-Call Program (its parameter `i`'s
leaf `j` is Input gate `starts[i] + j`), or child `(node, member)` of an activation. -/
inductive Scope where
  | top (d : Nat) (off : Nat) (starts : Option (Array Nat))
  | child (parent : Scope) (d : Nat) (off : Nat) (node : Nat) (member : Nat)
  deriving Inhabited

def Scope.d : Scope → Nat
  | .top d _ _ => d
  | .child _ d _ _ _ => d

def Scope.off : Scope → Nat
  | .top _ o _ => o
  | .child _ _ o _ _ => o

/-- Reference `k` of member `m` of a node's argument (§3): the whole argument when shared or unmapped. -/
@[inline] def Arg.memberRef (a : Arg) (m k : Nat) : Ref :=
  match a.dims with
  | none => a.seq.get k
  | some (d, L) => a.seq.get ((k / L * d + m) * L + k % L)

/-- One step of `layout.resolve`: the gate a body-scoped reference denotes in an activation when it is a prescribed input's or
a primitive's, else the reference it denotes one activation up (through a parameter) or down (through a node's output). -/
def resolveStep (P : Prog) : Scope × Ref → V (Nat ⊕ (Scope × Ref))
  | (.top _ _ none, ⟨true, _, _⟩) => throw "the root has no parameters (prescribed inputs are Input gates)"
  | (.top _ _ (some starts), ⟨true, i, j⟩) => match starts[i]? with
    | some s => pure (.inl (s + j))
    | none => throw s!"parameter {i} of the Call does not exist"
  | (.child parent d off k m, ⟨true, i, j⟩) => do
    let node := (P.defs[parent.d]!).body.nodes[k]!
    let a := node.args[i]!
    match node.form with
    | .scan =>
      if i == 0 then
        if m == 0 then pure (.inr (parent, a.seq.get j))
        else pure (.inr (.child parent d (off - (P.defs[d]!).gates) k (m - 1), (P.defs[d]!).body.ret.get j))
      else pure (.inr (parent, if i ≤ node.nxs then a.seq.get (m * (a.dims.map (·.2)).getD 0 + j) else a.seq.get j))
    | .batch => pure (.inr (parent, a.memberRef m j))
    | _ => pure (.inr (parent, a.seq.get j))
  | (sc, ⟨false, k, j⟩) => do
    let body := (P.defs[sc.d]!).body
    let node := body.nodes[k]!
    let o := sc.off + body.offsets[k]!
    let callee := P.defs[node.fn]!
    match node.form with
    | .prim => pure (.inl o)
    | .call => pure (.inr (.child sc node.fn o k 0, callee.body.ret.get j))
    | .batch =>
      let L := callee.outLeaves
      if callee.prim then pure (.inl (o + j / L))
      else pure (.inr (.child sc node.fn (o + j / L * callee.gates) k (j / L), callee.body.ret.get (j % L)))
    | .scan =>
      if node.n == 0 then throw s!"a scan with no iterations has no output to read"
      let .tuple #[carry, y] := callee.ret | throw "a scan step returns (carry, y)"
      let c := carry.leaves
      let (m, jj) := if j < c then (node.n - 1, j) else ((j - c) / y.leaves, c + (j - c) % y.leaves)
      pure (.inr (.child sc node.fn (o + m * callee.gates) k m, callee.body.ret.get jj))

/-- `layout.resolve`: the gate a body-scoped reference denotes in an activation, up through parameters and down through node
outputs until a primitive. -/
def resolve (P : Prog) (sc : Scope) (r : Ref) : V Nat := Steps.run "scope resolution" (resolveStep P) Steps.FUEL (sc, r)

/-- `partition_object.order_violation`: the first reference naming neither a parameter of its body nor an earlier node of it
(a body's `ret` may name any node), over every body the root reaches, root first and callees in node order, each
Definition once; its detail `{definition, node, arg, reference, why}`, or none when the program is topological. -/
def Prog.orderViolation (P : Prog) : Option Json := Id.run do
  let mut seen : Std.HashSet Nat := {}
  let mut todo : Array Nat := #[P.root]
  let mut at_ := 0
  for _ in [0:Steps.FUEL] do
    if at_ ≥ todo.size then break
    let d := todo[at_]!
    at_ := at_ + 1
    if seen.contains d then continue
    seen := seen.insert d
    let defn := P.defs[d]!
    if defn.prim then continue
    let body := defn.body
    for k in [0:body.nodes.size] do
      let node := body.nodes[k]!
      for i in [0:node.args.size] do
        for part in node.args[i]!.seq.parts do
          let names : Array (Bool × Nat) := match part with
            | .affine p idx _ c => if c > 0 then #[(p, idx)] else #[]
            | .strided p idx _ _ _ _ c => if c > 0 then #[(p, idx)] else #[]
            | .explicit it => it.map fun r => (r.p, r.i)
          for (p, idx) in names do
            if !(if p then idx < body.params.size else idx < k) then
              return some (Json.mkObj [("definition", Json.str defn.key), ("node", Lean.toJson k), ("arg", Lean.toJson i),
                ("reference", Json.arr #[Json.str (if p then "p" else "n"), Lean.toJson idx]), ("why", Json.str "not topological")])
      todo := todo.push node.fn
  if at_ < todo.size then return some (Json.mkObj [("why", Json.str "a walk of more steps than this verifier follows")])
  return none

/-- The root body's Calls (`partition_object.calls`): every node that is not an input node, with its root-body index. -/
def Prog.calls (P : Prog) : Array Nat :=
  let nodes := (P.defs[P.root]!).body.nodes
  (Array.range nodes.size).filter fun k => let f := P.defs[nodes[k]!.fn]!; !(f.prim && isInputId f.key)

def Prog.rootScope (P : Prog) : Scope := .top P.root 0 none

/-- The activation of root node `k`'s member `m`. -/
def Prog.child (P : Prog) (sc : Scope) (k m : Nat) : Scope :=
  let body := (P.defs[sc.d]!).body
  let node := body.nodes[k]!
  .child sc node.fn (sc.off + body.offsets[k]! + m * (P.defs[node.fn]!).gates) k m

/-- Member `m` of a primitive node, or the node itself: its operands, per argument in order, repeats kept (§6). -/
def operands (P : Prog) (sc : Scope) (node : Node) (m : Nat) : V (Array Nat) := do
  let callee := P.defs[node.fn]!
  let mut ops := #[]
  for i in [0:node.args.size] do
    let a := node.args[i]!
    let cnt := if node.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
    for t in [0:cnt] do ops := ops.push (← resolve P sc (if node.form == .batch then a.memberRef m t else a.seq.get t))
  return ops

/-- One step of locating gate `g` from an activation: its primitive, operands and activation when the node holding it is a
primitive's, else the activation of the node's member that holds it. -/
def gateStep (P : Prog) (g : Nat) (sc : Scope) : V ((Nat × Array Nat × Scope) ⊕ Scope) := do
  let body := (P.defs[sc.d]!).body
  let rel := g - sc.off
  let some k := (Array.range body.nodes.size).find? fun k => body.offsets[k]! ≤ rel && rel < body.offsets[k + 1]!
    | throw s!"gate {g} outside the circuit"
  let node := body.nodes[k]!
  let callee := P.defs[node.fn]!
  let m := if node.form == .prim || node.form == .call then 0 else (rel - body.offsets[k]!) / callee.gates
  if callee.prim then return .inl (node.fn, ← operands P sc node m, sc)
  return .inr (P.child sc k m)

/-- A gate's primitive and operands (`layout.Circuit.gate`): located through the offsets along the call hierarchy. -/
def Prog.gate (P : Prog) (g : Nat) : V (Nat × Array Nat × Scope) := do
  if g ≥ (P.defs[P.root]!).gates then throw s!"gate {g} outside the circuit"
  Steps.run "gate location" (gateStep P g) Steps.FUEL P.rootScope

end Flock.Program
