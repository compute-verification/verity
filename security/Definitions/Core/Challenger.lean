module

/-! The challenger's log, the verifier's one source of coins, and the two cases where randomness is derived from a
coin by hashing. -/

@[expose] public section

namespace Verity

namespace Challenger

/-- **An entry of a challenger's log**: a commitment it recorded, or a coin it released. -/
inductive Entry (R C : Type) where
  | recorded (r : R)
  | released (c : C)

/-- **A challenger's log**: its entries, oldest first. -/
abbrev Log (R C : Type) : Type := List (Entry R C)

namespace Log

variable {R C : Type}

/-- The coins the log released, in order. -/
def coins : Log R C → List C
  | [] => []
  | .recorded _ :: l => coins l
  | .released c :: l => c :: coins l

/-- **The log before its coin `i`**: every entry before the release of coin `i` (counting from 0), the commitments it
recorded and the coins it released before it; the whole log when it released fewer than `i + 1` coins. -/
def before : Log R C → Nat → Log R C
  | [], _ => []
  | .recorded r :: l, i => .recorded r :: before l i
  | .released _ :: _, 0 => []
  | .released c :: l, i + 1 => .released c :: before l i

end Log

end Challenger

end Verity
