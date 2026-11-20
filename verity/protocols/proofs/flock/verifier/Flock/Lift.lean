module

public import Circuits.Program
public import Std.Data.HashMap
public import Std.Data.HashSet

/-!
# A word Definition lifted to bits

`lift P d` is Definition `d` of the decoded Program `P` on bits, as circuits' `versions.lift` lifts it: each word
composite `d` reaches gets its lift (`liftDef`: every type `Ty.bitType`, every reference `liftSeq`), and each node calling a
word primitive calls the Boolean Definition `P.boolean` names for it (PROTOCOL.md §4.9), which is a gate or a composite on
bits that reaches no word primitive, or is refused by name. The lifts are appended after `P`'s Definitions, callees first,
so every callee stays before its caller: `decode` puts the Boolean Definitions after the word composites that call their
primitives. A Definition that
reaches no word primitive stays as it is, so the lift of a Boolean Program is that Program.

Its cost is the Definitions `d` reaches, each once, with their nodes and parts: the description, never a batch member.
-/

@[expose] public section

namespace Flock.Lift

open Flock.Program

/-- `bit_offsets`: word leaf `j` of `t` as its first bit in `Ty.bitType t` and its width, each leaf `w` consecutive bits in
leaf order. -/
def bitAt (t : Ty) : Nat → Nat × Nat :=
  match Widths.homo t with
  | some w => fun j => (j * w, w)
  | none =>
    let ws := t.widths
    let pre := ws.foldl (fun a w => a.push (a.back! + w)) #[0]
    fun j => (pre[j]!, ws[j]!)

/-- A body's references on bits: each word reference as the bits it occupies, an affine run over a collection of one width
an affine run of bits, every other part its bits named one by one. `at_ p i` is collection `(p, i)`'s `bitAt`. -/
def liftSeq (at_ : Bool → Nat → Nat → Nat × Nat) (homo : Bool → Nat → Option Nat) (s : Seq) : Seq :=
  let expand := fun (q : Part) => (Array.range q.count).foldl (fun a k =>
    let r := q.get k
    let (o, w) := at_ r.p r.i r.j
    a ++ (Array.range w).map fun t => ({ p := r.p, i := r.i, j := o + t } : Ref)) #[]
  Seq.ofParts <| s.parts.map fun q => match q with
    | .affine p i b c => match homo p i with
      | some w => .affine p i (b * w) (c * w)
      | none => .explicit (expand q)
    | q => .explicit (expand q)

/-- A lifted Definition's descriptor key: its word's, with the static `IR="boolean"` after its own (`versions.lift`). -/
def liftedKey (key : String) : String :=
  if key.endsWith "}" then String.mk key.toList.dropLast ++ ",IR=\"boolean\"}" else key ++ "{IR=\"boolean\"}"

/-- Word composite `d` on bits (`versions.lift_body`): every type `Ty.bitType`, every reference `liftSeq`, each node calling
`fnOf` of its Definition, a word primitive's node a Call of its Boolean Definition, and the offsets and gate count from
the callees' `gatesOf`. -/
def liftDef (d : Defn) (fnOf : Nat → Nat) (primOf : Nat → Bool) (gatesOf : Nat → Nat) : Defn :=
  let b := d.body
  let tyOf := fun (p : Bool) (i : Nat) =>
    if p then (d.params[i]?.map (·.2)).getD (.value 1) else (b.nodes[i]?.map (·.out)).getD (.value 1)
  let pAt := d.params.map fun (_, t) => bitAt t
  let nAt := b.nodes.map fun x => bitAt x.out
  let at_ := fun (p : Bool) (i : Nat) => ((if p then pAt[i]? else nAt[i]?).getD fun _ => (0, 0))
  let homo := fun p i => Widths.homo (tyOf p i)
  let lseq := liftSeq at_ homo
  let nodes := b.nodes.map fun x =>
    let f := fnOf x.fn
    let args := x.args.mapIdx fun i a =>
      let ty := Ty.bitType a.ty
      let dims := match x.form with
        | .batch => match x.axes[i]? with
          | some (some ax) => ty.axisDims ax
          | _ => none
        | .scan => if 1 ≤ i && i ≤ x.nxs then (match ty with | .array ln e => some (ln, e.leaves) | _ => none) else none
        | _ => none
      ({ ty, seq := lseq a.seq, dims } : Arg)
    { x with fn := f, args, out := Ty.bitType x.out, form := if x.form == .prim && !primOf f then .call else x.form }
  let offsets := nodes.foldl (fun o x => o.push (o.back! + (if x.form == .prim then 1 else x.members * gatesOf x.fn))) #[0]
  let params := d.params.map fun (n, t) => (n, Ty.bitType t)
  let ret := Ty.bitType d.ret
  { d with key := liftedKey d.key, params, ret, gates := offsets.back!,
           paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := ret.leaves,
           body := { params, retType := ret, nodes, ret := lseq b.ret, offsets } }

/-- **The lift** (the module docstring): `P` with the lift of every word composite the Definitions `start` reach appended,
and each Definition's index on bits in the result: a word composite's its lift's, a word primitive's its Boolean
Definition's, any other its own. -/
def closure (P : Prog) (start : Array Nat) : V (Prog × (Nat → Nat)) := do
  let word := fun (d : Defn) => isWordPrim d.prim d.key
  let bound := P.defs.foldl (fun n d => n + d.body.nodes.size + 1) (start.size + 1)
  let mut seen : Std.HashSet Nat := {}
  let mut todo := start
  let mut at_ := 0
  let mut toB : Std.HashMap Nat Nat := {}
  for _ in [0:bound] do
    if h : at_ < todo.size then
      let d := todo[at_]
      at_ := at_ + 1
      if seen.contains d then continue
      seen := seen.insert d
      let some defn := P.defs[d]? | throw s!"the lift to bits: no Definition {d}"
      if word defn then
        let some bk := P.boolean[defn.key]?
          | throw s!"the lift to bits: word primitive '{defn.key}' has no Boolean Definition in C's Boolean map"
        let some bi := P.index[bk]? | throw s!"the lift to bits: '{defn.key}''s Boolean Definition '{bk}' is not in C"
        let bd := P.defs[bi]!
        if !(bd.ret.bits && bd.params.all fun (_, t) => t.bits) || (bd.prim && !isBooleanPrim bd.key) then
          throw s!"the lift to bits: '{defn.key}''s Boolean Definition '{bk}' is neither a gate nor a composite on bits"
        toB := toB.insert d bi
        todo := todo.push bi
      else if !defn.prim then
        todo := todo ++ defn.body.nodes.map (·.fn)
    else break
  if at_ < todo.size then throw "the lift to bits: a walk of more steps than this verifier follows"
  let order := seen.toArray.qsort (· < ·)
  let mut wordy : Std.HashMap Nat Bool := {}
  for d in order do
    let defn := P.defs[d]!
    if defn.prim then wordy := wordy.insert d (word defn)
    else
      let mut w := false
      for x in defn.body.nodes do
        match wordy[x.fn]? with
        | some b => w := w || b
        | none => throw s!"the lift to bits: '{defn.key}' calls a Definition decoded after it"
      wordy := wordy.insert d w
  for (_, bi) in toB do
    if wordy[bi]?.getD true then throw s!"the lift to bits: Boolean Definition '{(P.defs[bi]!).key}' reaches a word primitive"
  let mut defs := P.defs
  let mut index := P.index
  let mut toL := toB
  for d in order do
    if wordy[d]?.getD false && !(P.defs[d]!).prim then
      let ds := defs
      let m := toL
      let l := liftDef (P.defs[d]!) (fun f => m[f]?.getD f) (fun f => (ds[f]!).prim) (fun f => (ds[f]!).gates)
      if index.contains l.key then throw s!"the lift to bits: '{l.key}' is already a Definition of C"
      toL := toL.insert d defs.size
      index := index.insert l.key defs.size
      defs := defs.push l
  let m := toL
  return ({ P with defs, index }, fun d => m[d]?.getD d)

/-- Definition `d` of `P` on bits (`closure` from `d`) and its index there. -/
def lift (P : Prog) (d : Nat) : V (Prog × Nat) := do
  let (Q, fnOf) ← closure P #[d]
  return (Q, fnOf d)

end Flock.Lift
