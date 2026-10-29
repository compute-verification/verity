import Std.Data.HashSet
import Flock.Program

/-!
# The NCI policy's check of a declared division

`Nci.Complies Γ α` (`Security/Definitions/Nci.lean`) holds of a declaration `(D, θ)` when `D`'s computed gates divide
into isolation units (IUs) of at most `Γ` gates, each reading at most `α` bits of novel information per gate:
`novelReads R = width(into(R) \ θ) ≤ α · size R`. The verifier's side is the check of one declared division: `evaluate`
decides that it is a partition of the computed gates into IUs (`Verity.Partition`) and that every IU meets both bounds,
and reports per IU its size and novel reads, and the first IU that fails, with the rule it fails.

## The input, and how descriptor v1 is the definition's circuit

An instance is one JSON object:
* `program`: the declaration's circuit `D`, a `verity-ir/descriptor/v1` Program as text (`verity/ml/circuits/PROTOCOL.md`),
  read with C-Flock's decoder (`Flock.Program.decode`). Its canonical layout (§6) is `D`: gate `g` of the layout is
  `Fin D.N` gate `g`, one primitive call returning one `Value<w>`, and `w g` is that `w`, in bits (`Circuit.w`, which
  the definition calls `verity.ir`'s `Value<w>`). The input gates are the input nodes, `Input<w>_v1` and
  `RegisteredInput<w>_v1` (§5). `reads g` is the set of gates `g`'s operands name (`Flock.Program.Prog.gate`: the
  layout's operands; a repeat names the same gate). The definition's gates read only earlier gates (`reads_lt`), so a
  Program with a gate reading itself or a later one is refused (`not-topological`), and its input gates read nothing
  (`input_reads`), so an input node with operands is refused (`program`; descriptor v1's input primitives take no
  parameters, so the decoder admits none, but the check does not rely on it). Nothing here reads a gate's operation or
  the outputs, and neither does `novelReads`.
* `theta`: θ, a list of gate ids (canonical root-scoped gate indices), each an input gate bound at registration
  (`RegisteredInput<w>_v1`, binding `registration` in `Flock.Program.inputId?`), none twice. The baseline is what the
  developer committed before the run, so an input bound per run (`Input<w>_v1`) is refused (`theta-not-registered`):
  in θ it would make any value the run brings, recommitted, free to read.
* `owner`: the division, one entry per gate as Flock's partition check takes it (`Flock.Partition.Owner`): `"input"` for
  an input gate, which is in no IU, and `u` for a computed gate of IU `u`. This is the definition's `Partition.unit`,
  with `none` written `"input"`, so a gate in two IUs has no encoding. The IUs are `0 … n - 1`, `n` one more than the
  largest `u`, and each holds a gate (`empty-iu`): an empty unit has size and reads 0 and meets both bounds, so a
  division with empty units complies exactly when the one with them dropped and the rest renumbered does.
* `gamma`: `Γ`, a natural number; `alpha`: `α` as `[num, den]`, naturals with `den > 0`. `novelReads R ≤ α · size R`
  is checked as `novelReads R · den ≤ num · size R` in ℕ.

`evaluate` refuses a malformed instance with the first failing code, in the order checked: `instance` (a field missing
or of the wrong JSON kind), `params` (`Γ` or `α`), `program` (the decoder refuses it, or a reference reads a leaf at
another width than it was committed with, `E_REINTERPRET`: `decodeProgram`, with a `message`), then gate by gate `program` or
`not-topological`, then θ entry by
entry (`theta-unknown`: not a gate id, `theta-not-input`, `theta-not-registered`: an input bound per run,
`theta-duplicate`), then the owners (`owner-length`: not one per gate, so an id outside the circuit or a gate left out;
then gate by gate `owner-entry`: neither `"input"` nor a natural, `input-in-iu`, `gate-in-no-iu`), then `empty-iu`.
`check` writes a refusal as `{"error": code}`, with a `message` giving the fix where one is the declaration's to make
(`fixOf`), or for a `program` refusal of a reinterpretation, where (`refusalOf`).
-/

namespace NciCheck

open Lean (Json toJson)
open Flock.Program (GateTable)

/-- A natural number in JSON: an integer with no exponent, at least 0. -/
def natOf? : Json → Option Nat
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

/-- Whether a primitive id is an input bound at registration, `RegisteredInput<w>_v1` (`Flock.Program.inputId?`). -/
def isRegisteredId (id : String) : Bool :=
  (Flock.Program.inputId? id).map (·.1) == some Flock.Program.REGISTRATION

/-- Gate `g` of the Program's canonical circuit: its width, whether it is an input gate, whether it is one bound at
registration, and its operands. Refused (`program`) where its primitive can't be located, returns more than one value, or
is an input node with operands, and (`not-topological`) where it reads a gate not before it. -/
def gateOf (P : Flock.Program.Prog) (g : Nat) : Except String (Nat × Bool × Bool × Array Nat) := do
  let (p, ops, _) ← match P.gate g with
    | .ok x => pure x
    | .error _ => throw "program"
  let defn := P.defs[p]!
  let .value w := defn.ret | throw "program"
  let inp := Flock.Program.isInputId defn.key
  if inp && !ops.isEmpty then throw "program"
  if ops.any (· ≥ g) then throw "not-topological"
  return (w, inp, isRegisteredId defn.key, ops)

/-- `D` as the definition reads it, gate by gate (`Flock.Program.GateTable`): the width of its value, whether it is an input
gate, and the gates its operands name (repeats kept: only the set is read); and, which the definition doesn't read but θ
does, whether it is an input bound at registration. Refused at the first gate `gateOf` refuses. -/
def circuitOf (P : Flock.Program.Prog) : Except String GateTable :=
  (List.range (P.defs[P.root]!).gates).foldlM (fun c g => c.push <$> gateOf P g) ⟨#[], #[], #[], #[]⟩

/-- θ and the division, checked: θ as a set of gates, each gate's IU (`none` for an input gate), and the number of IUs. -/
structure Division where
  theta : Std.HashSet Nat
  owner : Array (Option Nat)
  n : Nat

/-- One entry of θ, added to the entries before it. -/
def thetaStep (c : GateTable) (s : Std.HashSet Nat) (t : Json) : Except String (Std.HashSet Nat) := do
  let some g := natOf? t | throw "theta-unknown"
  if g ≥ c.N then throw "theta-unknown"
  if !c.isInput[g]! then throw "theta-not-input"
  if !c.registered[g]! then throw "theta-not-registered"
  if s.contains g then throw "theta-duplicate"
  return s.insert g

def isInputTag : Json → Bool
  | .str s => s == "input"
  | _ => false

/-- Gate `g`'s owner entry `j`: `none` for `"input"`, `some u` for IU `u`. -/
def ownerOf (c : GateTable) (g : Nat) (j : Json) : Except String (Option Nat) := do
  let o ← if isInputTag j then pure none else match natOf? j with
    | some u => pure (some u)
    | none => throw "owner-entry"
  if o.isSome && c.isInput[g]! then throw "input-in-iu"
  if o.isNone && !c.isInput[g]! then throw "gate-in-no-iu"
  return o

/-- The number of IUs: one more than the largest IU an owner names. -/
def iuCount (own : Array (Option Nat)) : Nat :=
  own.foldl (fun m o => match o with | some u => max m (u + 1) | none => m) 0

/-- The IUs an owner names. -/
def usedOf (own : Array (Option Nat)) : Std.HashSet Nat :=
  own.foldl (fun s o => match o with | some u => s.insert u | none => s) {}

def divisionOf (c : GateTable) (theta owner : Array Json) : Except String Division := do
  let mask ← (List.range theta.size).foldlM (fun s t => thetaStep c s theta[t]!) {}
  if owner.size != c.N then throw "owner-length"
  let own ← (List.range c.N).foldlM (fun own g => own.push <$> ownerOf c g owner[g]!) #[]
  let n := iuCount own
  -- more IUs than gates leaves one empty, and the check below stays as small as the circuit
  if n > c.N then throw "empty-iu"
  let used := usedOf own
  if !(List.range n).all used.contains then throw "empty-iu"
  return { theta := mask, owner := own, n }

/-- Per IU `u`, in one pass over the gates: its number of gates, and the gates outside it that a gate of it reads,
`into(R)`. -/
def tally (c : GateTable) (d : Division) : Array (Nat × Std.HashSet Nat) :=
  (List.range c.N).foldl (fun acc g => match d.owner[g]! with
    | some u => acc.modify u fun (s, into) =>
        (s + 1, c.reads[g]!.foldl (fun into a => if d.owner[a]! = some u then into else into.insert a) into)
    | none => acc) (Array.replicate d.n (0, {}))

/-- Per IU `u`, with gates `R`: `(size R, novelReads R)`, `novelReads R` the width of the gates of `into(R)` not in θ. -/
def ius (c : GateTable) (d : Division) : Array (Nat × Nat) :=
  (tally c d).map fun (s, into) => (s, into.fold (fun x a => if d.theta.contains a then x else x + c.w[a]!) 0)

/-- The first IU that breaks the policy, with the rule: `size` when it has more than `Γ` gates, else `nci` when its
novel reads exceed `num/den` times its size. -/
def failure (gamma num den : Nat) (r : Array (Nat × Nat)) : Option (Nat × String) :=
  (List.range r.size).findSome? fun u =>
    let (size, novel) := r[u]!
    if size > gamma then some (u, "size") else if novel * den > num * size then some (u, "nci") else none

/-- An instance's fields: the program's text, θ, the owners, `Γ`, and `α = num/den`. -/
structure Instance where
  program : String
  theta : Array Json
  owner : Array Json
  gamma : Nat
  num : Nat
  den : Nat

def field (j : Json) (k : String) : Except String Json :=
  match j.getObjVal? k with
  | .ok v => pure v
  | .error _ => throw "instance"

/-- A JSON array of two entries. -/
def pairOf? : Json → Option (Json × Json)
  | .arr a => if a.size = 2 then some (a[0]!, a[1]!) else none
  | _ => none

/-- The instance's fields, refused (`instance`) where one is missing or of the wrong JSON kind, and (`params`) where `Γ`
or `α` is not as stated. -/
def parse (j : Json) : Except String Instance := do
  let .str program ← field j "program" | throw "instance"
  let .arr theta ← field j "theta" | throw "instance"
  let .arr owner ← field j "owner" | throw "instance"
  let gammaJ ← field j "gamma"
  let alphaJ ← field j "alpha"
  let some gamma := natOf? gammaJ | throw "params"
  let some (a, b) := pairOf? alphaJ | throw "params"
  let some num := natOf? a | throw "params"
  let some den := natOf? b | throw "params"
  if den = 0 then throw "params"
  return { program, theta, owner, gamma, num, den }

/-- The verdict as JSON: `{gates, ius: [[size, novelReads], …], complies, failure: null | [iu, rule]}`. -/
def verdict (gates : Nat) (r : Array (Nat × Nat)) (f : Option (Nat × String)) : Json :=
  Json.mkObj [("gates", toJson gates),
    ("ius", Json.arr (r.map fun (s, x) => Json.arr #[toJson s, toJson x])),
    ("complies", Json.bool f.isNone),
    ("failure", match f with
      | none => Json.null
      | some (u, rule) => Json.arr #[toJson u, Json.str rule])]

/-! ## A leaf read at the width it was committed with

`E_REINTERPRET` (`verity/ml/circuits/PROTOCOL.md` §4.4, `defs.Builder.finish`): a node's argument or a body's return that
names a leaf of a parameter or of an earlier node (any node, for the return) of another width than its own type gives
that position is a silent reinterpretation, which the Python codec refuses as it decodes and C-Flock's decoder doesn't
look for. A reference to the node itself or a later one is not this check's: `not-topological` names it. -/

namespace Widths

open Flock.Program

private theorem lt_of_mem {fs : Array Ty} {t : Ty} (h : t ∈ fs) : sizeOf t < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  omega

private theorem snd_lt_of_mem {fs : Array (String × Ty)} {f : String × Ty} (h : f ∈ fs) : sizeOf f.2 < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  obtain ⟨k, t⟩ := f
  simp only [Prod.mk.sizeOf_spec] at this ⊢
  omega

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
    | (simp only [Ty.tuple.sizeOf_spec]; exact lt_of_mem ‹_›)
    | (simp only [Ty.record.sizeOf_spec]; exact snd_lt_of_mem ‹_›)

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
    | (simp only [Ty.tuple.sizeOf_spec]; exact lt_of_mem ‹_›)
    | (simp only [Ty.record.sizeOf_spec]; exact snd_lt_of_mem ‹_›)

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
    | (simp only [Ty.tuple.sizeOf_spec]; exact lt_of_mem ‹_›)
    | (simp only [Ty.record.sizeOf_spec]; exact snd_lt_of_mem ‹_›)

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

/-- The first leaf of the references `seq`, read as a `t`, from which a part names leaves of another width. -/
def seqBad (s : Src) (t : Ty) (seq : Seq) : M (Option Nat) := do
  let ht := homo t
  for q in seq.parts, a in seq.starts do
    if !(← partOk s t ht a q) then return some a
  return none

/-- Where the body of composite `d` reads a leaf at another width, as a message. -/
def bodyBad (P : Prog) (d : Defn) : M (Option String) := do
  let B := d.body
  let outs := B.nodes.map (·.out)
  let s : Src := { params := B.params.map (·.2), nodes := outs, hp := B.params.map (homo ·.2), hn := outs.map homo,
                   below := 0 }
  let what := fun (pos : Nat) (t : Ty) (w : String) =>
    s!"E_REINTERPRET: definition '{d.key}' {w}: the references from its leaf {pos} name leaves of other widths than " ++
    s!"its type {t.render} gives them: a silent reinterpretation. A word is read only at the width it was committed with"
  for k in [0:B.nodes.size] do
    let node := B.nodes[k]!
    let callee := P.defs[node.fn]!
    for i in [0:node.args.size] do
      let a := node.args[i]!
      if let some pos ← seqBad { s with below := k } a.ty a.seq then
        let pn := (callee.params[i]?.map (·.1)).getD s!"{i}"
        return some (what pos a.ty s!"node {k} ({node.form.name} {callee.key}) argument '{pn}'")
  if let some pos ← seqBad { s with below := B.nodes.size } B.retType B.ret then
    return some (what pos B.retType "the body's return")
  return none

/-- Where some composite of `P` reads a leaf at another width than it was committed with, as a message. -/
def reinterpretation (P : Prog) : Option String :=
  let go : M (Option String) := do
    for d in P.defs do
      if d.prim then continue
      if let some m ← bodyBad P d then return some m
    return none
  match go.run 0 with
  | .ok r _ => r
  | .error e _ => some s!"E_REINTERPRET unchecked: {e}"

end Widths

/-- The instance's Program as C-Flock's decoder reads it, refused (`program`) where the decoder refuses it, and where a
reference reads a leaf at another width than it was committed with (`Widths.reinterpretation`; `refusalOf` gives its
message). -/
def decodeProgram (text : String) : Except String Flock.Program.Prog :=
  match Flock.Program.decode text.toUTF8 with
  | .ok P => match Widths.reinterpretation P with
    | some _ => .error "program"
    | none => .ok P
  | .error _ => .error "program"

/-- The check on one instance: its verdict, or the first failing code of a malformed instance. -/
def evaluate (j : Json) : Except String Json := do
  let i ← parse j
  let P ← decodeProgram i.program
  let c ← circuitOf P
  let d ← divisionOf c i.theta i.owner
  let r := ius c d
  return verdict c.N r (failure i.gamma i.num i.den r)

/-- The fix a refusal's `message` gives, for a code whose fix is the declaration's to make. -/
def fixOf : String → Option String
  | "theta-not-registered" => some ("theta names an input bound per run (Input<w>_v1), and the baseline holds only " ++
      "what was committed at registration: declare that parameter registered (RegisteredInput<w>_v1, the Program's " ++
      "`registered`) or drop it from theta")
  | _ => none

/-- A refusal as JSON: `{"error": code}`, with `message` where there is one. -/
def refusalWith (code : String) (message : Option String) : Json :=
  Json.mkObj (("error", Json.str code) :: match message with
    | some m => [("message", Json.str m)]
    | none => [])

/-- A refusal as JSON, with its `message` where `fixOf` has one. -/
def refusal (code : String) : Json :=
  refusalWith code (fixOf code)

/-- Where `j`'s Program decodes and a reference of it reads a leaf at another width, `Widths.reinterpretation`'s
message. -/
def reinterpretationOf (j : Json) : Option String :=
  match parse j with
  | .ok i => match Flock.Program.decode i.program.toUTF8 with
    | .ok P => Widths.reinterpretation P
    | .error _ => none
  | .error _ => none

/-- `refusal`, a `program` refusal of `j` with the reinterpretation that `decodeProgram` refused as its `message`. -/
def refusalOf (j : Json) (code : String) : Json :=
  match (if code == "program" then reinterpretationOf j else none) with
  | some m => refusalWith code (some m)
  | none => refusal code

/-- `evaluate`, with a refusal as `refusalOf` writes it. -/
def check (j : Json) : Json :=
  match evaluate j with
  | .ok r => r
  | .error code => refusalOf j code

end NciCheck
