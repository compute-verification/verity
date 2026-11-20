import Spot.Admitted
import Spot.Restrict
import Spot.BlockFront
import Flock.Lift
import Flock.Lower

/-! Scratch, not for merge: both sides of `LiftWordEval` and `UnitWordComputes` (security's `Flock.Chain`) on one unit of
a C given as a file, compiled. `Prog.evalBool`'s walk and `ofProgram`'s gates are copied from security's
`Properties.Flock.EvalBool` and `Properties.Flock.Circuit.OfProgram` verbatim but for a gate's value as a bit list,
since those packages read Mathlib. Usage: `scratch-gemm C.json UNIT SEEDS`. -/

namespace ScratchGemm

open Flock Flock.Program Lean

def gateVal : Flock.Lower.Gate → List Bool → Bool
  | .and_, ops => ops.all id
  | .xor_, ops => ops.foldl xor false
  | .not_, ops => !(ops.headD false)
  | .const_ b, _ => b

def paramStarts (params : Array (String × Ty)) : Array Nat :=
  (params.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.2.leaves)) (#[], 0)).1

def refVal {W : Type} [Inhabited W] (x : List W) (st : Array Nat) (outs : Array (List W)) (r : Ref) : W :=
  if r.p then x.getD (st.getD r.i 0 + r.j) default else (outs.getD r.i []).getD r.j default

def argVals {W : Type} (v : Ref → W) (a : Arg) (cnt : Nat) (k : Nat → Nat) : List W :=
  let rs := a.seq.toArray
  (List.range cnt).map fun t => v (rs.getD (k t) default)

structure Wires (m : Type → Type) (W : Type) where
  gate : Flock.Lower.Gate → List W → m W
  lift : {α : Type} → Option α → m α

def Wires.bool : Wires Option Bool := ⟨fun g x => some (gateVal g x), fun o => o⟩

def walkStep {m : Type → Type} [Monad m] [Alternative m] {W : Type} [Inhabited W] (I : Wires m W) (P : Prog)
    (ev : Nat → List W → m (List W)) (d : Nat) (x : List W) : m (List W) := do
  let defn ← I.lift P.defs[d]?
  if defn.prim then
    match Flock.Lower.gateOf? defn.id with
    | some g => return [← I.gate g x]
    | none => return (← ev (← I.lift (P.bodyOf? defn.key)) x)
  let callOf (e : Nat) (ins : List W) : m (List W) := ev e ins
  let st := paramStarts defn.params
  let outs ← defn.body.nodes.toList.foldlM (init := (#[] : Array (List W))) fun outs nd => do
    let v := refVal x st outs
    if d ≤ nd.fn then failure
    let callee ← I.lift P.defs[nd.fn]?
    let leaves (i : Nat) : Nat := ((callee.params.getD i default).2).leaves
    match nd.form with
    | .prim | .call => outs.push <$> callOf nd.fn (nd.args.toList.flatMap fun a => argVals v a a.seq.count id)
    | .batch =>
      let ms ← (List.range nd.n).mapM fun m => callOf nd.fn ((List.range nd.args.size).flatMap fun i =>
        let a := nd.args.getD i default
        match a.dims with
        | none => argVals v a a.seq.count id
        | some (dd, L) => argVals v a (leaves i) fun t => (t / L * dd) * L + t % L + m * L)
      return outs.push ms.flatten
    | .scan =>
      let c := leaves 0
      let init := argVals v (nd.args.getD 0 default) c id
      let (carry, ys) ← (List.range nd.n).foldlM (init := (init, ([] : List W))) fun (carry, ys) m => do
        let ins := carry ++ ((List.range nd.args.size).drop 1).flatMap fun i =>
          let a := nd.args.getD i default
          if i ≤ nd.nxs then argVals v a (leaves i) fun t => t + m * leaves i else argVals v a (leaves i) id
        let o ← callOf nd.fn ins
        return (o.take c, ys ++ o.drop c)
      return outs.push (carry ++ ys)
  let rs := defn.body.ret.toArray
  return (List.range defn.body.ret.count).map fun t => refVal x st outs (rs.getD t default)

def evalDef? (P : Prog) : Nat → Nat → List Bool → Option (List Bool)
  | 0, _, _ => none
  | fuel + 1, d, x => walkStep Wires.bool P (evalDef? P fuel) d x

def evalBool (P : Prog) (d : Nat) (x : List Bool) : List Bool :=
  (evalDef? P (2 * P.defs.size + 2) d x).getD []

/-! `LowerEval`'s word walk. -/

def wordWires : Wires Option (List Bool) :=
  ⟨fun g xs => some [gateVal g (xs.map (·.headD false))], fun o => o⟩

def wordDef? (P : Prog) : Nat → Nat → List (List Bool) → Option (List (List Bool))
  | 0, _, _ => none
  | fuel + 1, d, x => do
    let defn ← P.defs[d]?
    if defn.prim then some [evalBool P d x.flatten]
    else walkStep wordWires P (wordDef? P fuel) d x

def wordEval (P : Prog) (d : Nat) (x : List (List Bool)) : List (List Bool) :=
  (wordDef? P (2 * P.defs.size + 2) d x).getD []

def leafWidths (d : Defn) : List Nat := d.params.toList.flatMap fun p => p.2.widths.toList

/-! `ofProgram`'s gates, a value a bit list of the gate's width. -/

def gateAt (P : Prog) (g : Nat) : Option (Nat × Array Nat) :=
  match P.gate g with
  | .ok (p, ops, _) => some (p, ops)
  | .error _ => none

def nGates (P : Prog) : Nat := (P.defs[P.root]!).gates

def widthAt (P : Prog) (g : Nat) : Nat :=
  match gateAt P g with
  | some (p, _) => match (P.defs[p]!).ret with
    | .value w => w
    | _ => 0
  | none => 0

def inputAt (P : Prog) (g : Nat) : Bool :=
  match gateAt P g with
  | some (p, _) => isInputId (P.defs[p]!).key
  | none => true

def opsBelow (P : Prog) (g : Nat) : List Nat :=
  if inputAt P g then [] else
  match gateAt P g with
  | some (_, ops) => ops.toList.filter (· < g)
  | none => []

/-- `ofProgram`'s `op g` on the trace `v`, as bits. -/
def opAt (P : Prog) (g : Nat) (v : Nat → List Bool) : List Bool :=
  match gateAt P g with
  | some (p, _) =>
    let r := evalBool P p ((opsBelow P g).flatMap v)
    (List.range (widthAt P g)).map fun i => r.getD i false
  | none => List.replicate (widthAt P g) false

/-! `Chain`'s unit ports. -/

def unitArgs (P : Prog) (g : Nat) : Array Spot.BlockFront.Port :=
  match (do
    let (k, m) ← Spot.BlockFront.unitAt P g
    let some x := (P.defs.getD P.root default).body.nodes[k]? | throw "no root node"
    Spot.BlockFront.operandsW P x m : V (Array (Nat × Nat))) with
  | .ok ps => ps.map fun (q, w) => { gate := q, width := w }
  | .error _ => #[]

def unitOuts (P : Prog) (g : Nat) : Array Spot.BlockFront.Port :=
  match (do
    let (k, m) ← Spot.BlockFront.unitAt P g
    let some x := (P.defs.getD P.root default).body.nodes[k]? | throw "no root node"
    let c := P.defs.getD x.fn default
    (Array.range c.ret.leaves).mapM fun l => do
      let q ← resolve P P.rootScope { p := false, i := k, j := m * c.ret.leaves + l }
      pure ({ gate := q, width := c.ret.widths.getD l 0 } : Spot.BlockFront.Port) : V (Array Spot.BlockFront.Port)) with
  | .ok ps => ps
  | .error _ => #[]

/-- `portBits` of one port: the gate's value where it is a gate of C, else `width` zeros. -/
def portWord (P : Prog) (v : Nat → List Bool) (q : Spot.BlockFront.Port) : List Bool :=
  if q.gate < nGates P then v q.gate else List.replicate q.width false

/-! The check. -/

def rnd (s i : Nat) : Bool := ((i * 2654435761 + s * 40503 + 12345) / 65536) % 2 == 1

def words (s : Nat) (ws : List Nat) : List (List Bool) :=
  (ws.foldl (fun (acc : Array (List Bool) × Nat) w =>
    (acc.1.push ((List.range w).map fun t => rnd s (acc.2 + t)), acc.2 + w)) (#[], 0)).1.toList

def randGate (P : Prog) (s g : Nat) : List Bool := (List.range (widthAt P g)).map fun t => rnd s (g * 97 + t)

/-- A trace of C: each non-input gate of `[lo, hi)` its operation on the trace, every other gate sampled. -/
def traceFor (P : Prog) (s lo hi : Nat) : Nat → List Bool :=
  let m : Std.HashMap Nat (List Bool) := (List.range (hi - lo)).foldl (fun m t =>
    let g := lo + t
    if inputAt P g then m
    else m.insert g (opAt P g fun a => m.getD a (randGate P s a))) {}
  fun g => m.getD g (randGate P s g)

def qOf (key : String) : Json := Json.mkObj [("query", Json.mkObj [("name", "Q_template_instance"),
  ("version", (0 : Nat)), ("params", Json.mkObj [("template", Json.str key)])])]

def ms (t0 : Nat) : IO Nat := return (← IO.monoMsNow) - t0

def ok {α : Type} (what : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => throw (IO.userError s!"{what} refuses: {e}")

def main (args : List String) : IO UInt32 := do
  let path := args.getD 0 "C.json"
  let u := (args.getD 1 "0").toNat!
  let seeds := ((args.getD 2 "1,2").splitOn ",").map String.toNat!
  let t0 ← IO.monoMsNow
  let bytes ← IO.FS.readBinFile path
  let P ← ok "decode" (decode bytes)
  IO.println s!"decode: {P.defs.size} Definitions, {nGates P} gates, {← ms t0} ms"
  let (k, m) ← ok "unitAt" (Spot.BlockFront.unitAt P u)
  let sc := P.child P.rootScope k m
  let d := sc.d
  let key := (P.defs[d]!).key
  let hi := sc.off + (((P.defs[d]?).map (·.gates)).getD 0)
  IO.println s!"unit {u}: root node {k}, member {m}, template {key}, gates [{sc.off}, {hi})"
  let S ← ok "restrict" (Spot.Restrict.restrict P #[u])
  let a ← ok "admitted" (Spot.OneStageZk.admitted (qOf key) [] S)
  let (L, e) ← ok "lift" (Flock.Lift.lift S a.d)
  let ws := leafWidths (P.defs[a.d]!)
  IO.println s!"admitted: a.d {a.d}, d {d}, ins {a.ins.toList}; {ws.length} leaves, {ws.foldl (· + ·) 0} bits; restrict, admitted, lift {← ms t0} ms"
  for s in seeds do
    let x := words s ws
    let t1 ← IO.monoMsNow
    let lhs := evalBool L e x.flatten
    IO.println s!"  seed {s}: lift on bits {lhs.length} bits, {← ms t1} ms"
    let t2 ← IO.monoMsNow
    let rhs := (wordEval P a.d x).flatten
    IO.println s!"  seed {s}: word walk {rhs.length} bits, {← ms t2} ms"
    IO.println s!"  seed {s} LiftWordEval: {if lhs == rhs && !lhs.isEmpty then "AGREE" else "DISAGREE"}"
    let t3 ← IO.monoMsNow
    let Y := traceFor P s sc.off hi
    let aw := (unitArgs P u).toList.map (portWord P Y)
    let ow := (unitOuts P u).toList.map (portWord P Y)
    IO.println s!"  seed {s}: trace of the unit's gates, {ow.length} return leaves, {← ms t3} ms"
    let t4 ← IO.monoMsNow
    let wv := wordEval P d aw
    IO.println s!"  seed {s}: word walk on the trace's arguments, {← ms t4} ms"
    IO.println s!"  seed {s} UnitWordComputes widths: {if aw.map List.length == leafWidths (P.defs[d]!) then "AGREE" else "DISAGREE"} ({aw.length} args); outs: {if ow == wv && !ow.isEmpty then "AGREE" else "DISAGREE"} (C {ow.length} leaves, walk {wv.length})"
  IO.println s!"total {← ms t0} ms"
  return 0

end ScratchGemm

def main (args : List String) : IO UInt32 := ScratchGemm.main args
