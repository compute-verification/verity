module

public import Properties.Flock.Circuit.Inline
public import Proofs.Properties.Flock.EvalBool
public import Proofs.Properties.Flock.DecodeBoolean
import Batteries.Tactic.OpenPrivate
meta import Properties.Flock.Circuit.Inline
meta import Proofs.Properties.Flock.EvalBool
meta import Flock.Test.Lower

/-!
# `inline` against the evaluator

On `Flock.Test.Lower`'s Boolean program and on the format's descriptor vector (`WORD_DESC`), on four input seeds each,
every bit of every gate of C, read from the inlined gates, is the evaluator's (`Prog.evalBool`), and the inlining's AND
gates are at most the bodies' ANDs (`bodyAnds`). On `WORD_DESC` the block of its `TWiden16_v1` gate and that gate's 16-bit
operand gives the same bits gate by gate, each gate's emitted span within its `gateAnds`; and a Definition handed fewer
wires than its parameters have reads past them, where the inlining's gate 0 gives the evaluator's `false`.

`decode_gates`: in a decoded Program every primitive is spelt as its family's id and every gate has §9's signature.
-/

@[expose] public section

namespace FlockSoundness.Inline

open Flock.Program Flock.OfProgram

/-- The raw gates' values on inputs `a`, by index. -/
def rawEval (ops : Array RawOp) (a : ℕ → Bool) : Array Bool :=
  (List.range ops.size).foldl (fun v g => v.push (match ops[g]! with
    | .input => a g | .and x y => v[x]! && v[y]! | .xor x y => xor v[x]! v[y]! | .not x => !v[x]! | .zero => false)) #[]

/-- C's gates' bits by the evaluator, input gate `g`'s bit `i` given by `inp g i`. -/
def cEval (P : Flock.Program.Prog) (inp : ℕ → ℕ → Bool) : Array (List Bool) :=
  (List.range (size P)).foldl (fun vals g =>
    let w := widthAt P g
    if inputAt P g then vals.push ((List.range w).map (inp g)) else
    match gateAt P g with
    | none => vals.push []
    | some (p, _) =>
      let xs := (opsBelowN P g).flatMap fun a => vals[a]!
      vals.push ((List.range w).map fun i => (P.evalBool p xs).getD i false)) #[]

def seedBit (s x : ℕ) : Bool := ((x * 2654435761 + s * 40503 + 12345) / 65536) % 3 == 0

def andsIn (ops : Array RawOp) (lo hi : ℕ) : ℕ :=
  ((List.range (hi - lo)).filter fun k => match ops[lo + k]! with | .and _ _ => true | _ => false).length

/-- On seed `s`, each gate `g` of `gs`: its bits, inlined and evaluated, are the evaluator's, and its emitted span has at
most its `gateAnds` AND gates. The walk is run once. -/
def agrees (P : Flock.Program.Prog) (gs : List ℕ) (s : ℕ) : Bool :=
  let (done, ops) := walk P
  let bit (g i : ℕ) : ℕ := (done.getD g default).bits.getD i (ops.size - 1)
  let v := rawEval ops (seedBit s)
  let c := cEval P fun g i => seedBit s (bit g i)
  gs.all fun g =>
    let x := done.getD g default
    ((List.range (widthAt P g)).all fun i => v.getD (bit g i) false == (c.getD g []).getD i false) &&
      andsIn ops x.lo x.hi ≤ (if inputAt P g then 0 else match gateAt P g with | some (p, _) => bodyAnds P p | none => 0)

def wholeOk (P : Flock.Program.Prog) : Bool := (List.range 4).all (agrees P (List.range (size P)))

def wordP : Flock.Program.Prog := match decode FlockSoundness.EvalBool.WORD_DESC.toUTF8 with | .ok P => P | .error _ => default
def toyP : Flock.Program.Prog := match Flock.Test.Lower.prog with | .ok P => P | .error _ => default

#guard size toyP > 0 && wholeOk toyP
#guard size wordP > 0 && wholeOk wordP
#guard let ops := (walk toyP).2; andsIn ops 0 ops.size > 0

/-- The block of `TWiden16_v1`'s gate and its operands. -/
def widenBlock (P : Flock.Program.Prog) : List ℕ :=
  ((List.range (size P)).filter fun g => keyAt P g == "TWiden16_v1").flatMap fun g => g :: opsBelowN P g

#guard (widenBlock wordP).any (fun g => widthAt wordP g == 16) &&
  (widenBlock wordP).any (fun g => keyAt wordP g == "TWiden16_v1") &&
  (List.range 4).all (agrees wordP (widenBlock wordP))

/-- Definition `d` on `k` input wires, inlined and evaluated against `evalDef?` on the same bits, on seed `s`. -/
def shortAgrees (P : Flock.Program.Prog) (d k s : ℕ) : Bool :=
  let fuel := 2 * P.defs.size + 2
  let ins := (List.range k).map (· + 1)
  match (inlDef? P fuel d ins).run (#[RawOp.zero] ++ Array.replicate k RawOp.input) with
  | (some ws, ops) =>
    let v := rawEval ops (seedBit s)
    match evalDef? P fuel d (ins.map (seedBit s)) with
    | some bs => ws.length == bs.length && (ws.zip bs).all fun (w, b) => v.getD w false == b
    | none => false
  | (none, _) => false

#guard match wordP.bodyOf? "TAdd32_v1" with
  | some b => (wordP.defs[b]!).params.foldl (fun n p => n + p.2.leaves) 0 > 10 &&
      (List.range 4).all fun s => shortAgrees wordP b 10 s
  | none => false

/-! ## The gates' signatures in a decoded Program -/

/-- **Every primitive of a decoded Program is spelt as its family's id, and each gate has §9's signature**
(`DecodeBoolean.decode_facts`). -/
theorem decode_gates (bytes : ByteArray) (P : Flock.Program.Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    ∀ w < P.defs.size, (P.defs[w]!).prim = true → misspelt (P.defs[w]!).key = false ∧
      ∀ g, gateKind? (P.defs[w]!).key = some g →
        (P.defs[w]!).params = g.signature.1 ∧ (P.defs[w]!).ret = g.signature.2 :=
  fun w hw hp => ((DecodeBoolean.decode_facts bytes P h hr).2 w hw hp).2

end FlockSoundness.Inline
