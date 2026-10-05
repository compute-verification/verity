import FlockSoundness.Model.Session
import FlockSoundness.Merkle
import FlockSoundness.Protocol.Model.Compiled
import FlockSoundness.Accounting.Bound
import FlockSoundness.Game.Interleave
import FlockSoundness.Model.Statement

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

open Classical in
/-- Every requested opening of a rep is a row of the level's lane count whose path verifies
against the level's cap (§14). -/
def OpensOK (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap₀ : Cap D) (r : ℕ)
    (p : PendL F D) (o : Opens F D) : Prop :=
  ∀ req ∈ p.reqs, ∃ cap, capOf cap₀ p.caps req.level = some cap ∧
    ∀ j < req.positions.length, (o r req.level j).1.length = 2 ^ req.lanes ∧
      Merkle.Verifies H E cap req.depth (req.positions.getD j 0) (o r req.level j).1
        (o r req.level j).2

/-- The compiled table's output: the verdict, the level-0 cap, both reps' results, and the
openings. -/
structure OutC (F D : Type) where
  accepted : Bool
  cap₀ : Cap D
  pend : PendL F D × PendL F D
  opens : Opens F D

open Classical in
/-- **One table's compiled session** (§5, §13–§14): the level-0 cap, the link points, both
reps, then the openings; accepted iff both reps accept on the openings and every path verifies. -/
noncomputable def tableC (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) : Game (OutC F D) := do
  let cap₀ ← recv (Cap D)
  let pts ← draw (Fin 2 → Fin mPts → F)
  let extra := extraClaims S ptLocal mPts pts
  let pend ← interleave (repC A 0 S sch hm extra) (repC A 1 S sch hm extra)
  let o ← recv (Opens F D)
  pure ⟨pend.1.verdict o && pend.2.verdict o &&
      decide (OpensOK H E cap₀ 0 pend.1 o ∧ OpensOK H E cap₀ 1 pend.2 o), cap₀, pend, o⟩

/-! ## Salted leaves (hm96) -/

omit [Field F] [Fintype F] [DecidableEq F] in
theorem opensOKL_plain (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap₀ : Cap D) (r : ℕ) (p : PendL F D)
    (o : Opens F D) : OpensOKL H E (Merkle.Leaf.plain H E) cap₀ r p o (fun _ _ _ => ()) ↔ OpensOK H E cap₀ r p o :=
  Iff.rfl

open Classical in
/-- **One table's compiled session under a leaf scheme `Lf`**: `tableC`, with the salts in the openings' message (one
final message, after the last coin), and every path verified from its row's leaf under its salt. -/
noncomputable def tableCL {Sl : Type} (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) :
    Game (OutCL F D Sl) := do
  let cap₀ ← recv (Cap D)
  let pts ← draw (Fin 2 → Fin mPts → F)
  let extra := extraClaims S ptLocal mPts pts
  let pend ← interleave (repC A 0 S sch hm extra) (repC A 1 S sch hm extra)
  let os ← recv (Opens F D × Salts Sl)
  pure ⟨pend.1.verdict os.1 && pend.2.verdict os.1 &&
      decide (OpensOKL H E Lf cap₀ 0 pend.1 os.1 os.2 ∧ OpensOKL H E Lf cap₀ 1 pend.2 os.1 os.2), cap₀, pend, os.1,
    os.2⟩

end FlockSoundness.Model
