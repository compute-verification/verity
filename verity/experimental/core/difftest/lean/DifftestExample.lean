import Difftest

/-! The harness's example port, which its own tests drive: `add` and `div` on integers `a` and `b` (`div` is Lean's `Int`
division, which rounds so that the remainder is non-negative, and rejects `b = 0`), and `echo`, which returns its input and
so holds the wire format (parse, then `Flock.canon`) to Python's canonical JSON. -/

namespace DifftestExample

open Lean (Json ToJson toJson)

def int (j : Json) (k : String) : Except String Int :=
  match j.getObjValAs? Int k with
  | .ok v => .ok v
  | .error _ => .error s!"{k}: not an integer"

def add : Difftest.Fn := fun j => do
  return toJson ((← int j "a") + (← int j "b"))

def div : Difftest.Fn := fun j => do
  let a ← int j "a"
  let b ← int j "b"
  if b == 0 then throw "division by zero"
  return toJson (a / b)

def echo : Difftest.Fn := pure

end DifftestExample

def main : IO Unit :=
  Difftest.serve (Difftest.table [("add", DifftestExample.add), ("div", DifftestExample.div), ("echo", DifftestExample.echo)])
