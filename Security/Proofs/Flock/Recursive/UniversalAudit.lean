import Proofs.Flock.Recursive.Property
import Proofs.Flock.Recursive.UniversalStmt

/-!
# Recursion: what `RecursiveAudit`'s (i) reads of the registered private circuit at the executable's class

Both branches of `execClassAt st` set no hidden entries (`execClassAt_nnz`), so `Hidden (execClassAt st).cls` has one
element (`hidden_execClassAt_eq`). Therefore:

* every `stmt : (ℕ → ByteArray) → Ω → Hidden (execClassAt st).cls` is constant (`stmt_execClassAt_eq`), and (i)'s
  last event `¬ (innerAt T CT st).holds (stmt p ω)` is one event for every registered `p` (`holds_stmt_iff`): it reads
  no row of the registered private circuit;
* `_hx`'s first disjunct holds at every outcome, hidden statement `x`, `stmt` and strings `q` (`hx_execClassAt`).

At this class the registered rows enter (i) only through an event that reads them: the session's claimed outputs
differing from the decoded description's on its inputs (`UniversalStmt.SessionFaulty`), bounded by (i)'s event through
`UniversalStmt.prob_faulty_le` at the outcomes where `holds` rules it out.
-/

namespace FlockSoundness.Discharge.Recursion.UniversalAudit

open FlockSoundness.Discharge.Recursion

/-- **The executable's class has no hidden entries**, at every statement: `ExecClass.ofStmt` and `ExecClass.fixed`
both set `nnz := 0`. -/
theorem execClassAt_nnz (st : Flock.Stmt) : (execClassAt st).cls.nnz = 0 := by
  unfold execClassAt
  split
  · rfl
  · rfl

/-- **The executable's class has one hidden statement.** -/
theorem hidden_execClassAt_eq (st : Flock.Stmt) (M M' : Hidden (execClassAt st).cls) : M = M' :=
  UniversalStmt.hidden_eq (execClassAt_nnz st) M M'

/-- **A statement by commit strings at the executable's class is constant**: it reads neither the strings nor the
draw. -/
theorem stmt_execClassAt_eq {st : Flock.Stmt} {Ω : Type} (stmt : (ℕ → ByteArray) → Ω → Hidden (execClassAt st).cls)
    (q q' : ℕ → ByteArray) (ω ω' : Ω) : stmt q ω = stmt q' ω' :=
  hidden_execClassAt_eq st _ _

/-- **(i)'s last event reads no registered row**: whether the compiled table's statement holds at `stmt q ω` is the
same for every `q` and `ω`. -/
theorem holds_stmt_iff {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} {Ω : Type} (stmt : (ℕ → ByteArray) → Ω → Hidden (execClassAt st).cls)
    (q q' : ℕ → ByteArray) (ω ω' : Ω) :
    (Flock.Guarantees.innerAt T CT st).holds (stmt q ω) ↔
      (Flock.Guarantees.innerAt T CT st).holds (stmt q' ω') := by
  rw [stmt_execClassAt_eq stmt q q' ω ω']

/-- **`_hx`'s first disjunct at the executable's class**: the hidden statement is every commit strings' statement. -/
theorem hx_execClassAt {st : Flock.Stmt} {Reg Ω : Type} (x : Reg → Ω → Hidden (execClassAt st).cls)
    (stmt : (ℕ → ByteArray) → Ω → Hidden (execClassAt st).cls) (R : Reg) (q : ℕ → ByteArray) (ω : Ω) :
    x R ω = stmt q ω :=
  hidden_execClassAt_eq st _ _

end FlockSoundness.Discharge.Recursion.UniversalAudit
