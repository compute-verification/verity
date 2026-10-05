import FlockSoundness.Protocol.Audit.FlockBatched
import FlockSoundness.Protocol.Refine.Salted

/-!
# The batched session under a leaf scheme (PROTOCOL.md §5, §14)

`batchedSession` (`Audit/FlockBatched.lean`) checks every opened path from the unsalted leaf `H(encode row)`
(`OpensOK`). The statements the headline is about commit their proof rows with `hm96-sha512/v1` leaves, whose salts the
prover sends with the openings, and flock-verify checks those paths from the salted leaf (`verify_refines_hm96`:
`OpensOKL` at `leaf512`). An honest hm96 transcript fails `OpensOK`, so flock-verify's acceptance refines the session
below, not `batchedSession`.

The session is `sessionB` with each table's run after the link points `tableAfterL`: `tableAfter`, whose openings'
message also carries the salts (`Opens F D × Salts Sl`), and whose paths verify from their rows' leaves under those
salts (`OpensOKL`). Its one table is `Model.tableCL` (`tableCL_eq`), and at `Leaf.plain` it is the unsalted session
(`opensOKL_plain_pair`, up to the salts' `Unit` message). `batchedSessionL` is `batchedSession` over it.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type} {mPts : ℕ}

open Classical in
/-- **A table's run after the link points under the leaf scheme `Lf`**: both reps, then its openings with their
salts; accepted iff both reps accept on the openings and every path verifies from its row's leaf under its salt. -/
noncomputable def tableAfterL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) : Game (OutCL F D Sl) :=
  (interleave (repC A 0 S sch hm (extraClaims S ptLocal mPts pts))
    (repC A 1 S sch hm (extraClaims S ptLocal mPts pts))).bind
    fun pend => Game.send (Opens F D × Salts Sl) fun os => Game.ret
      ⟨pend.1.verdict os.1 && pend.2.verdict os.1 &&
        decide (OpensOKL H E Lf cap₀ 0 pend.1 os.1 os.2 ∧ OpensOKL H E Lf cap₀ 1 pend.2 os.1 os.2), cap₀, pend,
        os.1, os.2⟩

/-- A default output, for positions outside a batch. -/
def dfltOutL [Inhabited D] [Inhabited Sl] : OutCL F D Sl :=
  ⟨false, fun _ => default, (PendL.reject, PendL.reject), fun _ _ _ => ([], []), fun _ _ _ => default⟩

/-- The table's run after the link points, with level-0 cap `cap₀`, under `Lf`. -/
noncomputable def tabAfterL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (T : TabSpec mPts) (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Game (OutCL F D Sl) :=
  tableAfterL A H E Lf T.S T.sch T.hm T.ptLocal mPts cap₀ pts

/-- The tables' runs after `Commit`, with the caps it sent, under `Lf`. -/
noncomputable def gamesBL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (caps : Fin ts.length → Cap D) :
    List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl)) :=
  List.ofFn fun i => tabAfterL A H E Lf (ts.get i) (caps i)

/-- After `Commit`: one link-point draw, then the games concurrently. -/
noncomputable def sessAfterBL (fs : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) :
    Game (List (OutCL F D Sl)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts => batch (fs.map (· pts))

/-- **A batched session under `Lf`** (PROTOCOL.md §5): `Commit` sends every table's level-0 cap, then one link-point
draw, then every table's reps and salted openings concurrently. -/
noncomputable def sessionBL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) : Game (List (OutCL F D Sl)) :=
  Game.send (Fin ts.length → Cap D) fun caps => sessAfterBL (gamesBL A H E Lf ts caps)

/-- The salted session accepts when every table accepts. -/
def acceptedBL [Inhabited D] [Inhabited Sl] (ts : List (TabSpec mPts)) (outs : List (OutCL F D Sl)) : Prop :=
  ∀ i : Fin ts.length, (outs.getD i dfltOutL).accepted = true

local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **The batched session on a draw, under `Lf`**: `Audit.Partition.batchedSession` with salted openings. At
`Refine.leaf512` it is the session flock-verify's hm96 statements run. -/
noncomputable def batchedSessionL {n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) : Game Bool :=
  (sessionBL execArith H E Lf (plan S R)).map fun outs => decide (acceptedBL (plan S R) outs)

end FlockSoundness.Discharge.Exec
