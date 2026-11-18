module

public import Properties.Flock.Circuit.OfProgram
public import Flock.Test.Lower
meta import Properties.Flock.Circuit.OfProgram
meta import Flock.Test.Lower
meta import Definitions.Core.Circuit

/-!
`ofProgram` on PROTOCOL.md §9's Boolean program (`Flock.Test.Lower.DESC`, `ExampleAdd2_v1`): core's circuit of its
description has the root's 19 gates, the 4 input gates `Input1_v1` first, no registered gate, and the 4 gates the root
returns as its outputs; on each of the 16 inputs its evaluation, each gate by `Prog.evalBool`, gives the vectors' word,
`x + y + 8`.
-/

@[expose] public section

namespace Flock.OfProgram.Toy

open Flock.Program

def P : Prog := match Flock.Test.Lower.prog with
  | .ok P => P
  | .error _ => default

def C : Verity.Circuit := ofProgram P

/-- Gate `g`'s value's bit 0 on the inputs `x`, `y` (gates 0-1 and 2-3, low bit first). -/
def evalBit (x y g : ℕ) : Bool :=
  if h : g < C.N then
    (C.eval (fun a => BitVec.ofNat (C.w a) (if a.val < 2 then (x >>> a.val) % 2 else (y >>> (a.val - 2)) % 2)) ⟨g, h⟩).getLsbD 0
  else false

/-- The returned word: the outputs in the root's return order, low bit first. -/
def word (x y : ℕ) : ℕ :=
  (outputGates P).foldr (fun g acc => 2 * acc + if evalBit x y g then 1 else 0) 0

#guard C.N == 19
#guard (List.range 19).map (inputAt P) == [true, true, true, true] ++ List.replicate 15 false
#guard !(List.range 19).any (registeredAt P)
#guard (outputGates P).length == 4
#guard Flock.Test.Lower.WORDS.all fun (x, y, z) => word x y == z

end Flock.OfProgram.Toy
