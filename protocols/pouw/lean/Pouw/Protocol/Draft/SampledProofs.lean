import Verity.Protocol.Partition
import Verity.Protocol.Game

/-!
# Draft of sampled proofs' registered soundness (provisional; trusted)

**Provisional.** This file is the draft of sampled proofs' trusted text, held here until lean's sampled-proofs package
lands; PoUW then imports that package instead, and only these constants' names change. It holds a proof system's data
(`System`), its soundness against registered inputs (`RegisteredSound`) and the honest prover's membership in its class
(`HonestIn`), and nothing else.

A `System` proves one circuit `C` under the partition `P` into proof units. The registered inputs `a` are committed
under an opening `op` to the root `commit a op`, and the verifier plays `game (commit a op)` with the prover, ending in
an output that says whether it accepts and which units were drawn. Some gates' values are public: the verifier fixes
them itself (`fixed`; for C-Flock, the constant columns `ones` and `zeros`, the tile check's output among them). The
proof system's class `Cls a op σ q` is the strategies its soundness speaks of at budget `q`, on the registered inputs
`a` under `op` (for C-Flock, every collision finder its theorems build from `σ`, the registered-path finder from `a`'s
leaves under `op`, meets `cr/sha-512` or `ecr/sha-512` at `q`). The class is the proof system's own: it must not read
whether `a` passes the check, which its type cannot rule out.

`RegisteredSound sys η`: every strategy in the class has a transcript `X` of `C`, agreeing with the public values,
such that the verifier accepts while a drawn unit is wrong in `X` or `X`'s inputs are not `a` with probability at most
`η q` at the strategy's registration. Without the public values the statement would be empty (`X := C.eval a`).
-/

namespace Pouw.Draft.SampledProofs

open scoped ENNReal

/-- A proof system for one circuit, on inputs committed under openings of type `Opening` to roots of type `Root`. -/
structure System (Opening Root : Type) where
  C : Verity.Circuit
  n : ℕ
  /-- The proof units. -/
  P : Verity.Partition C n
  /-- The commitment to the registered inputs (only the input gates of a transcript are registered). -/
  commit : C.Transcript → Opening → Root
  /-- The values the verifier fixes itself. -/
  fixed : (g : Fin C.N) → Option (BitVec (C.w g))
  Out : Type
  /-- The verifier's game on a root. -/
  game : Root → Verity.Game Out
  accept : Out → Bool
  /-- The units drawn on a run. -/
  drawn : Out → Finset (Fin n)
  /-- What the error depends on: for C-Flock, the prover's registration. -/
  Reg : Type
  reg : (r : Root) → Verity.Game.Strategy (game r) → Reg
  /-- The proof system's class at a budget, on the registered inputs and their opening. -/
  Cls : (a : C.Transcript) → (op : Opening) → Verity.Game.Strategy (game (commit a op)) → ℕ → Prop
  /-- The honest prover. -/
  honest : (a : C.Transcript) → (op : Opening) → Verity.Game.Strategy (game (commit a op))

/-- **Soundness against registered inputs, up to `η`.** For every budget `q`, registered inputs `a`, opening `op` and
strategy `σ` in the class: some transcript `X` holds the public values, and the probability that the verifier accepts
while a drawn unit is wrong in `X` or `X`'s inputs are not `a` is at most `η q` at `σ`'s registration. -/
def RegisteredSound {Opening Root : Type} (sys : System Opening Root) (η : ℕ → sys.Reg → ℝ≥0∞) : Prop :=
  ∀ (q : ℕ) (a : sys.C.Transcript) (op : Opening) (σ : Verity.Game.Strategy (sys.game (sys.commit a op))),
    sys.Cls a op σ q → ∃ X : sys.C.Transcript, (∀ g b, sys.fixed g = some b → X g = b) ∧
      Verity.Game.prob (fun o => sys.accept o = true ∧
          (¬ Disjoint (sys.P.wrong X) (sys.drawn o) ∨ ¬ sys.C.InputsAgree X a))
        (sys.game (sys.commit a op)) σ ≤ η q (sys.reg _ σ)

/-- **The honest prover is in the class**, at every budget, on every registered inputs and opening (whether or not the
inputs pass the check). -/
def HonestIn {Opening Root : Type} (sys : System Opening Root) : Prop :=
  ∀ (q : ℕ) (a : sys.C.Transcript) (op : Opening), sys.Cls a op (sys.honest a op) q

end Pouw.Draft.SampledProofs
