/-!
# The challenger and its log

Every coin a protocol uses comes from a **challenger**: an auditor-controlled device that flips coins only after the
commitments they sample are recorded, and logs every coin it releases. A challenger may run on the same device as
another auditor device: onsite, the network certifier's machine runs one; for remote, it is an auditor host. The
verifier reads coins only from a challenger's log, and uses no randomness of its own.

* `Challenger.Entry R C`: one entry of a challenger's log, a commitment `r : R` it recorded or a coin `c : C` it
  released;
* `Challenger.Log R C`: its log, oldest entry first;
* `Log.coins`: the coins the log released, in order;
* `Log.before l i`: the log before its coin `i`, everything it recorded and released before releasing coin `i` (the
  whole log, when it released fewer than `i + 1` coins).

A run of a challenger that flips `n` coins of a finite type `C` is weighed by its flips `ω : Fin n → C`, each sequence
of flips equally likely (`Game.prCoin`), as `Game.coin` weighs a live coin. What the challenger is taken to do with its
flips and its log, and what the verifier reads, are the named premises of `Definitions.Core.Assumptions.Challenger`.

**Derived by hashing.** Fiat–Shamir is out, with two named exceptions, each at a rate where a round trip to the
challenger is impractical; there, randomness is derived from a challenger's coin by SHA-512:

* PoUW's per-matmul noise seeds, each hashed from the verification phase's salt (one challenger coin per phase), the
  call's position, and the committed roots of its operands;
* the remote bandwidth audit's pass seeds within a phase, each hashed from the phase's challenger coin and the previous
  pass's root.

Neither is a coin of this log: a statement about one reads it through the coin it is derived from.
-/

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
