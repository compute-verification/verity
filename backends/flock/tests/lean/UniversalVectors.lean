import Lean.Data.Json
import FlockSoundness.Discharge.PrivateCircuit.Interp

/-!
Vectors for `verity.ml.boolean.universal.UniversalUnit_v1` from the soundness package's interpreter
(`FlockSoundness/Discharge/PrivateCircuit/Interp.lean`), for `tests/fixtures/universal_lean.json`.

* `random`: `interp` on random programs (every `K`-bit index, so forward reads and, at `(20, 13, 5)`, reads past `W`)
  and random data, at `(16, 16, 8)` and `(20, 13, 5)`.
* `pad`: random in-range netlists of at most `(16, 16)` and `nOut = 8`, `padUnit`'s program for each at `(16, 16, 8)`,
  and `interp` of it on random data.
* `dummy`: `dummyUnit`'s program at each class and `interp` of it on random data.

Every program is given as the Definition's arguments (`op`, `src`, `out`, `x`), each a string of bits in leaf order.
Regenerate with `tests/lean/universal.sh`.
-/

open Lean FlockSoundness.Discharge.PrivateCircuit.Universal

/-- splitmix64 -/
def next (s : UInt64) : UInt64 × UInt64 :=
  let s := s + 0x9e3779b97f4a7c15
  let z := (s ^^^ (s >>> 30)) * 0xbf58476d1ce4e5b9
  let z := (z ^^^ (z >>> 27)) * 0x94d049bb133111eb
  (z ^^^ (z >>> 31), s)

/-- `k` draws below `m` each. -/
def draws (s : UInt64) (k m : Nat) : Array Nat × UInt64 := Id.run do
  let mut s := s
  let mut out := #[]
  for _ in [0:k] do
    let (z, s') := next s
    s := s'
    out := out.push (z.toNat % m)
  return (out, s)

def bitStr (bs : List Bool) : String := String.ofList (bs.map fun b => if b then '1' else '0')

def bvBits {K : Nat} (v : BitVec K) : List Bool := (List.range K).map v.getLsbD

def progJson {G nIn nOut : Nat} (p : Prog G nIn nOut) (x : Fin nIn → Bool) : List (String × Json) :=
  let gs := List.ofFn p.gates
  [("cls", toJson [G, nIn, nOut]),
   ("op", .str (bitStr (gs.flatMap fun g => bvBits g.op))),
   ("src", .str (bitStr (gs.flatMap fun g => bvBits g.a ++ bvBits g.b))),
   ("out", .str (bitStr ((List.ofFn p.out).flatMap bvBits))),
   ("x", .str (bitStr (List.ofFn x))),
   ("y", .str (bitStr (List.ofFn (interp p x))))]

def randomCase (G nIn nOut : Nat) (s : UInt64) : Json × UInt64 :=
  let K := idxBits G nIn
  let (ops, s) := draws s G 4
  let (src, s) := draws s (2 * G) (2 ^ K)
  let (out, s) := draws s nOut (2 ^ K)
  let (x, s) := draws s nIn 2
  let p : Prog G nIn nOut :=
    ⟨fun g => ⟨BitVec.ofNat 2 ops[g]!, BitVec.ofNat K src[2 * g.val]!, BitVec.ofNat K src[2 * g.val + 1]!⟩,
     fun i => BitVec.ofNat K out[i]!⟩
  (Json.mkObj (progJson p fun j => x[j]! == 1), s)

def padCase (c : Cls) (s0 : UInt64) : Json × UInt64 := Id.run do
  let (gn, s1) := draws s0 2 (max c.G c.nIn)
  let g := gn[0]! % c.G + 1
  let n := gn[1]! % c.nIn + 1
  let mut s := s1
  let mut gates : List NGate := []
  for j in [0:g] do
    let (r, s') := draws s 3 (4 * (n + j))
    s := s'
    gates := gates ++ [⟨⟨r[0]! % 4, by omega⟩, r[1]! % (n + j), r[2]! % (n + j)⟩]
  let (outs, s2) := draws s c.nOut (n + g)
  let (x, s3) := draws s2 c.nIn 2
  let u : Netlist := ⟨n, gates, outs.toList⟩
  let xf : Nat → Bool := fun j => x[j]! == 1
  let net := Json.mkObj [("n_in", toJson n), ("gates", toJson (gates.map fun g => [g.op.val, g.a, g.b])),
    ("outputs", toJson outs.toList), ("interpret", .str (bitStr (interpret u xf)))]
  return (Json.mkObj (("netlist", net) :: progJson (padUnit c u) fun j => xf j.val), s3)

def dummyCase (c : Cls) (s : UInt64) : Json × UInt64 :=
  let (x, s) := draws s c.nIn 2
  (Json.mkObj (progJson (dummyUnit c) fun j => x[j]! == 1), s)

def many (k : Nat) (f : UInt64 → Json × UInt64) (s : UInt64) : List Json × UInt64 := Id.run do
  let mut s := s
  let mut out := []
  for _ in [0:k] do
    let (j, s') := f s
    s := s'
    out := out ++ [j]
  return (out, s)

def main : IO Unit := do
  let s : UInt64 := 0x756e697665727361
  let (r1, s) := many 64 (randomCase 16 16 8) s
  let (r2, s) := many 32 (randomCase 20 13 5) s
  let (p, s) := many 64 (padCase ⟨1, 16, 16, 8⟩) s
  let (d1, s) := many 8 (dummyCase ⟨1, 16, 16, 8⟩) s
  let (d2, _) := many 8 (dummyCase ⟨1, 20, 13, 5⟩) s
  IO.println (Json.mkObj [("random", toJson (r1 ++ r2)), ("pad", toJson p), ("dummy", toJson (d1 ++ d2))]).compress
