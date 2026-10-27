import Flock.Canon
import Flock.Hash
import PousCheck.Verdict

/-!
# The verdict unit's difftest: `PousCheck.Verdict`'s functions on JSON

`pous-verdict-difftest` serves them to the core difftest harness, whose Python half (`tests/test_verdict_difftest.py`)
holds each one's output to the Definition `verdict_circuit.PousVerdict`, compiled:

* `gates`: `Verdict.prog K M LIMIT`'s gate count, output wires and the SHA-512 of its gates as `text`, against the
  Definition's canonical layout (`standalone_call`'s circuit), gate for gate and operand for operand;
* `eval`: `Prog.eval` of `prog` on a session's openings laid out by `encode`, the code's `verdicts` and `accepts`,
  against the compiled circuit's output (`evaluation.bits.evaluate_bits`), `verdict_circuit.round_verdict` and
  `verdict_circuit.verdict_value`.

Both refuse the statics `verdict_circuit` refuses (it admits `K ≥ 1`, `1 ≤ LIMIT < 2⁶⁴`); `eval` refuses a session
that is not `K` records of 92 bytes and `K` digests of 64.
-/

namespace PousCheck.VerdictDifftest

open Lean Flock PousCheck.Verdict

def num (n : Nat) : Json := .num (JsonNumber.fromNat n)

/-- The statics `verdict_circuit` admits. -/
def statics (i : Json) : V (Nat × Nat × Nat) := do
  let (K, M, LIMIT) := (← J.nat i "K" "instance", ← J.nat i "M" "instance", ← J.nat i "LIMIT" "instance")
  if K = 0 then throw "K is at least 1"
  if LIMIT = 0 ∨ 2 ^ 64 ≤ LIMIT then throw "LIMIT is in [1, 2^64)"
  return (K, M, LIMIT)

/-- A gate as one line: `input`, `const 0` or `const 1`, or `not`, `and` or `xor` and its operand wires. -/
def gateText : Gate → String
  | .input => "input"
  | .const b => if b then "const 1" else "const 0"
  | .not i => s!"not {i}"
  | .and ws => " ".intercalate ("and" :: ws.map toString)
  | .xor ws => " ".intercalate ("xor" :: ws.map toString)

/-- A program's gates, one line each. -/
def text (C : Prog) : String := C.gates.foldl (fun s g => s ++ gateText g ++ "\n") ""

def gates (i : Json) : V Json := do
  let (K, M, LIMIT) ← statics i
  let C := prog K M LIMIT
  return Json.mkObj [("count", num C.gates.size), ("outs", .arr (C.outs.toArray.map num)),
    ("sha512", .str (hexOf (Sha512.hash (text C).toUTF8)))]

/-- A byte string's bits, least significant first in each byte, as `verdict_circuit` lays out its inputs. -/
def bitsOf (b : ByteArray) : List Bool := b.toList.flatMap fun x => (List.range 8).map x.toNat.testBit

def verdictName : RoundVerdict → String
  | .hit => "hit"
  | .wrong => "wrong"
  | .late => "late"
  | .reject => "reject"

/-- `K` byte strings of `n` bytes each, in hex, at `i[k]`. -/
def byteStrings (i : Json) (k : String) (K n : Nat) : V (List ByteArray) := do
  let xs ← (← J.arr i k "instance").mapM fun x => do unhex (← J.asStr x s!"instance.{k}")
  if xs.size ≠ K then throw s!"{k}: {xs.size} for {K} rounds"
  if xs.any (·.size ≠ n) then throw s!"{k}: each is {n} bytes"
  return xs.toList

def eval (i : Json) : V Json := do
  let (K, M, LIMIT) ← statics i
  let os := List.zipWith (fun r e => Opening.mk (bitsOf r) (bitsOf e)) (← byteStrings i "records" K 92)
    (← byteStrings i "expected" K 64)
  let vs := verdicts LIMIT os
  return Json.mkObj [("out", .arr (((prog K M LIMIT).eval (encode os)).toArray.map (num ·.toNat))),
    ("verdicts", .arr (vs.toArray.map (.str ∘ verdictName))), ("accepts", num (accepts M vs).toNat)]

end PousCheck.VerdictDifftest
