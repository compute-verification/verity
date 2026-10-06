import Definitions.Core.Partition
import Definitions.Core.Game

/-!
# Draft of sampled proofs' registered soundness (provisional; trusted)

**Provisional.** This file is the draft of sampled proofs' trusted text, held here until lean's sampled-proofs package
lands; PoUW then imports that package instead, and only these constants' names change. It holds a proof system's data
(`System`), its soundness against registered values (`RegisteredSound`) and the honest prover's membership in its class
(`HonestIn`), and nothing else.

A `System` proves the circuit `C` under the partition `P` into proof units. The registered values `a` are committed
under an opening `op` to the root `commit a op`, and the verifier plays `game (commit a op)` with the prover, ending in
an output that says whether it accepts and which units were drawn. The commitment binds `a`'s values at the
`registered` gates: the inputs, and the outputs of units it registers (for C-Flock, the gates its registered reads
read, each in some unit's `P.io`). The proof system's class `Cls a op σ q` is the strategies its soundness speaks of at
budget `q`, on the registered values `a` under `op` (for C-Flock, every collision finder its theorems build from `σ`,
the registered-path finder from `a`'s leaves under `op`, meets `cr/sha-512` or `ecr/sha-512` at `q`). The class is the
proof system's own: it must not read whether `a` passes the check, which its type cannot rule out.

`RegisteredSound sys η`: every strategy in the class has a transcript `X` of `C` such that the verifier accepts while a
drawn unit is wrong in `X` or `X` is off `a` at a registered gate with probability at most `η q` at the strategy's
registration. With only the inputs registered the statement would be empty (`X := C.eval a`); it binds what the
circuit computes through the registered outputs.
-/

namespace Pouw.Draft.SampledProofs

open scoped ENNReal

/-- A proof system for the circuit `C` under the proof units `P`, on values committed under openings of type `Opening`
to roots of type `Root`. -/
structure System (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) (Opening Root : Type) where
  /-- The gates whose values the commitment binds. -/
  registered : Finset (Fin C.N)
  /-- The commitment to the registered values (only a transcript's values at `registered` gates are bound). -/
  commit : C.Transcript → Opening → Root
  Out : Type
  /-- The verifier's game on a root. -/
  game : Root → Verity.Game Out
  accept : Out → Bool
  /-- The units drawn on a run. -/
  drawn : Out → Finset (Fin n)
  /-- What the error depends on: for C-Flock, the prover's registration. -/
  Reg : Type
  reg : (r : Root) → Verity.Game.Strategy (game r) → Reg
  /-- The proof system's class at a budget, on the registered values and their opening. -/
  Cls : (a : C.Transcript) → (op : Opening) → Verity.Game.Strategy (game (commit a op)) → ℕ → Prop
  /-- The honest prover. -/
  honest : (a : C.Transcript) → (op : Opening) → Verity.Game.Strategy (game (commit a op))

/-- **Soundness against registered values, up to `η`.** For every budget `q`, registered values `a`, opening `op` and
strategy `σ` in the class: for some transcript `X`, the probability that the verifier accepts while a drawn unit is
wrong in `X` or `X` is off `a` at a registered gate is at most `η q` at `σ`'s registration. -/
def RegisteredSound {C : Verity.Circuit} {n : ℕ} {P : Verity.Partition C n} {Opening Root : Type}
    (sys : System C n P Opening Root) (η : ℕ → sys.Reg → ℝ≥0∞) : Prop :=
  ∀ (q : ℕ) (a : C.Transcript) (op : Opening) (σ : Verity.Game.Strategy (sys.game (sys.commit a op))),
    sys.Cls a op σ q → ∃ X : C.Transcript,
      Verity.Game.prob (fun o => sys.accept o = true ∧
          (¬ Disjoint (P.wrong X) (sys.drawn o) ∨ ∃ g ∈ sys.registered, X g ≠ a g))
        (sys.game (sys.commit a op)) σ ≤ η q (sys.reg _ σ)

/-- **The honest prover is in the class**, at every budget, on every registered values and opening (whether or not
they pass the check). -/
def HonestIn {C : Verity.Circuit} {n : ℕ} {P : Verity.Partition C n} {Opening Root : Type}
    (sys : System C n P Opening Root) : Prop :=
  ∀ (q : ℕ) (a : C.Transcript) (op : Opening), sys.Cls a op (sys.honest a op) q

end Pouw.Draft.SampledProofs
