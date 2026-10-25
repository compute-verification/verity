import Definitions.Flock.Compiled
import Proofs.Flock.Soundness.Merkle

/-!
# The compiled protocol: Merkle caps, openings after the last coin (PROTOCOL.md §13–§14)

The protocol as run commits each table by a Merkle cap (§14) and opens the queried rows with
their sibling paths in the proof file, which the server receives after every round (S17). The
openings enter no round, so the prover chooses them after all coins of both reps: the table's
last message here (`Opens`, per rep, level and query). The verifier then checks every path
against its level's cap and computes the consistency values from the opened rows.

The model follows `Model.Ligerito` step for step. The running claim and the round polynomial are
affine in the consistency values, so the compiled verifier carries them as functions of the
openings (`LigStateC.s`, `LigStateC.q`) and evaluates them once the openings arrive; every query
records its request (`Req`) for the path checks.

**Salted leaves.** `tableCL` is `tableC` under a leaf scheme (`Merkle.Leaf`): the last message carries the openings
and their salts (`Salts`), and each path verifies from its row's leaf under its salt (`OpensOKL`). At `Leaf.plain`,
`OpensOKL` is `OpensOK` (`opensOKL_plain`).
-/

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting
variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-! ## Salted leaves (hm96) -/

omit [Field F] [Fintype F] [DecidableEq F] in
theorem opensOKL_plain (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap₀ : Cap D) (r : ℕ) (p : PendL F D)
    (o : Opens F D) : OpensOKL H E (Merkle.Leaf.plain H E) cap₀ r p o (fun _ _ _ => ()) ↔ OpensOK H E cap₀ r p o :=
  Iff.rfl

end FlockSoundness.Model
