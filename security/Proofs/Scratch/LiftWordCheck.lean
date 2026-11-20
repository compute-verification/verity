module

public import Proofs.Properties.Flock.Circuit.LowerEval
public import Proofs.Properties.Flock.EvalBool
meta import Proofs.Properties.Flock.Circuit.LowerEval
meta import Proofs.Properties.Flock.EvalBool

/-! Scratch, not for merge: both sides of `Flock.Chain.LiftWordEval` and `Flock.Chain.UnitWordComputes` computed on
C's units, for the units `restrict` and `admitted` accept, on sampled words and sampled traces. Run it with the
lakefile's roots holding `Proofs.Scratch.LiftWordCheck` (`lakefile.patch`), by
`verity lean try security/Proofs/Scratch/LiftWordCheck.lean`. -/

@[expose] public section

namespace Scratch.LiftWordCheck

open Flock Flock.Program Flock.Chain Lean

def rnd (s i : Nat) : Bool := ((i * 2654435761 + s * 40503 + 12345) / 65536) % 2 == 1

def words (s : Nat) (ws : List Nat) : List (List Bool) :=
  (ws.foldl (fun (acc : List (List Bool) × Nat) w =>
    (acc.1 ++ [(List.range w).map fun t => rnd s (acc.2 + t)], acc.2 + w)) ([], 0)).1

def randNat (s g w : Nat) : Nat := (List.range w).foldl (fun n t => 2 * n + if rnd s (g * 97 + t) then 1 else 0) 0

/-- A trace of C: gates in `[lo, hi)` that are not input gates are their operation on the trace, every other gate a
sampled value. -/
def traceFor (P : Prog) (s lo hi : Nat) : (Flock.OfProgram.ofProgram P).Trace :=
  let C := Flock.OfProgram.ofProgram P
  let arr : Array Nat := (List.range C.N).foldl (fun arr g =>
    if h : g < C.N then
      let gi : Fin C.N := ⟨g, h⟩
      if lo ≤ g && g < hi && !(Flock.OfProgram.inputAt P g) then
        let Y : C.Trace := fun a => BitVec.ofNat (C.w a) (arr.getD a.val 0)
        arr.push (C.op gi Y).toNat
      else arr.push (randNat s g (C.w gi))
    else arr) #[]
  fun a => BitVec.ofNat (C.w a) (arr.getD a.val 0)

def qOf (key : String) : Json := Json.mkObj [("query", Json.mkObj [("name", "Q_template_instance"),
  ("version", (0 : Nat)), ("params", Json.mkObj [("template", Json.str key)])])]

def lens (x : List (List Bool)) : List Nat := x.map List.length

/-- For unit `u` of C (`d`, its descriptor): the door's acceptance and both statements' sides at each seed. -/
def checkUnit (P : Prog) (u : Nat) (cuts : List (String × List (String × Nat))) (seeds : List Nat) : List String :=
  match Spot.BlockFront.unitAt P u with
  | .error e => [s!"unit {u}: unitAt refuses: {e}"]
  | .ok (k, m) =>
    let sc := P.child P.rootScope k m
    let d := sc.d
    let key := (P.defs[d]!).key
    let hd := s!"unit {u} (root node {k}, member {m}, {(P.defs[P.root]!).body.nodes[k]!.form.name}) template {key}"
    match Spot.Restrict.restrict P #[u] with
    | .error e => [hd ++ s!": restrict refuses: {e}"]
    | .ok S =>
      match Spot.OneStageZk.admitted (qOf key) cuts S with
      | .error e => [hd ++ s!": admitted refuses: {e}"]
      | .ok a =>
        match Flock.Lift.lift S a.d with
        | .error e => [hd ++ s!": lift refuses: {e}"]
        | .ok (L, e) =>
          let ws := leafWidths (P.defs[a.d]!)
          let hi := sc.off + (((P.defs[d]?).map (·.gates)).getD 0)
          [hd ++ s!": admitted (a.d {a.d}, d {d}, ins {a.ins.toList}); leaf widths {ws}; unit gates [{sc.off}, {hi})"] ++
          seeds.flatMap fun s =>
            let x := words s ws
            let lhs := Prog.evalBool L e x.flatten
            let rhs := (wordEval P a.d x).flatten
            let Y := traceFor P s sc.off hi
            let aw := argWords Y u
            let ow := outWords Y u
            let wv := wordEval P d aw
            [s!"  seed {s} LiftWordEval: {if lhs == rhs then "AGREE" else "DISAGREE"} (lift {lhs.length} bits, word walk {rhs.length} bits)" ++
              (if lhs == rhs then "" else s!" lift {lhs.take 64} walk {rhs.take 64}"),
             s!"  seed {s} UnitWordComputes widths: {if lens aw == ws then "AGREE" else "DISAGREE"} (args {lens aw}); outs: {if ow == wv then "AGREE" else "DISAGREE"} (C {lens ow}, walk {lens wv})" ++
              (if ow == wv then "" else s!" C {(ow.map fun w => w.take 40).take 3} walk {(wv.map fun w => w.take 40).take 3}")]

def checkAll (desc : String) (units : List Nat) (cuts : List (String × List (String × Nat)) := [])
    (seeds : List Nat := [1, 2, 3]) : List String :=
  match decode desc.toUTF8 with
  | .error e => ["decode refuses: " ++ e]
  | .ok P => units.flatMap fun u => checkUnit P u cuts seeds

#eval checkAll FlockSoundness.EvalBool.WORD_DESC (List.range 16)

-- `TRow`'s `row` parameter (4 leaves of 32 bits) cut 2 ways, and `TCol`'s `c` (2 leaves) cut 2 ways.
#eval checkAll FlockSoundness.EvalBool.WORD_DESC (List.range 16)
  [("TRow_v1{N=4,OP={\"fn\":\"TAdd32_v1\"},C={\"f64\":\"0x1.8000000000000p+0\"},D={\"dict\":{\"a\":[1,\"x\"],\"k\":2}}}",
    [("row", 2)]), ("TCol_v1", [("c", 2)])] [4]

end Scratch.LiftWordCheck
