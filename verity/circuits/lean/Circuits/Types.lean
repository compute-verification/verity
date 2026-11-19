/-
`Flock.Program`'s format vocabulary below the descriptor (`verity/circuits/PROTOCOL.md`): the canonical JSON domain (§1),
types and leaves (§2) and reference sequences (§3), what `Circuits.Descriptor` reads them with. `Ty.ofJson`, which reads
a type from JSON, is the descriptor's.
-/
module

public import Base.Canon
public import Base.Hash
public import Base.JsonParse
public import Base.Sizes
public import Lean.Data.Json.FromToJson.Basic
meta import Base.Canon
meta import Base.Hash
meta import Base.JsonParse
meta import Base.Sizes

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock.Program

open Lean (Json JsonNumber)

@[noinline] def MAX_SAFE_INT : Nat := 2 ^ 53 - 1

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
end Ty

namespace Ty

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

end Flock.Program
