import FlockSoundness.Audit.FlockBatched
import FlockSoundness.Refine.Salted
import FlockSoundness.Protocol.Discharge.Exec.SessionL

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

omit [Fintype K] in
/-- One table's salted session is its level-0 cap, the link points, then `tableAfterL`. -/
theorem tableCL_eq (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) :
    tableCL A H E Lf S sch hm ptLocal mPts =
      Game.send (Cap D) fun cap₀ => Game.coin (Fin 2 → Fin mPts → F) fun pts =>
        tableAfterL A H E Lf S sch hm ptLocal mPts cap₀ pts :=
  rfl

/-- **A batched session over any table run** (PROTOCOL.md §5): `Commit` sends every table's commitment `Cm`, then one
link-point draw, then every table's run `tab` after them, concurrently. `sessionBL` is its instance at `tabAfterL`
(`sessionBL_eq`); a session whose tables commit or open more (`--zk`'s pads) is another. -/
def sessionG {Cm Out : Type} (tab : TabSpec mPts → Cm → (Fin 2 → Fin mPts → F) → Game Out)
    (ts : List (TabSpec mPts)) : Game (List Out) :=
  Game.send (Fin ts.length → Cm) fun cms =>
    Game.coin (Fin 2 → Fin mPts → F) fun pts => batch ((List.ofFn fun i => tab (ts.get i) (cms i)).map (· pts))

/-- `sessionG` accepts when every table's output is accepted by `acc` (`dflt` past the batch). -/
def acceptedG {Out : Type} (acc : Out → Bool) (dflt : Out) (ts : List (TabSpec mPts)) (outs : List Out) : Prop :=
  ∀ i : Fin ts.length, acc (outs.getD i dflt) = true

omit [Fintype K] in
theorem sessionBL_eq (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) :
    sessionBL A H E Lf ts = sessionG (tabAfterL A H E Lf) ts :=
  rfl

omit [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] in
theorem acceptedBL_eq [Inhabited D] [Inhabited Sl] (ts : List (TabSpec mPts)) (outs : List (OutCL F D Sl)) :
    acceptedBL ts outs ↔ acceptedG (·.accepted) dfltOutL ts outs :=
  Iff.rfl

omit [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] in
/-- The salts' leaf scheme `Leaf.plain` checks what `OpensOK` checks: at it, a table's salted verdict is the
unsalted one, whatever the (unit) salts. -/
theorem opensOKL_plain_pair (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap₀ : Cap D)
    (pend : PendL F D × PendL F D) (os : Opens F D × Salts Unit) :
    (OpensOKL H E (Merkle.Leaf.plain H E) cap₀ 0 pend.1 os.1 os.2 ∧
        OpensOKL H E (Merkle.Leaf.plain H E) cap₀ 1 pend.2 os.1 os.2) ↔
      (OpensOK H E cap₀ 0 pend.1 os.1 ∧ OpensOK H E cap₀ 1 pend.2 os.1) :=
  Iff.rfl

local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **The batched session on a draw, over any table run**: `sessionG tab` accepted by `acc`. `batchedSessionL` is its
instance at `tabAfterL` (`batchedSessionL_eq`). -/
noncomputable def batchedSessionG {n : ℕ} {Reg Cm Out : Type}
    (tab : TabSpec mPts → Cm → (Fin 2 → Fin mPts → 𝔽) → Game Out) (acc : Out → Bool) (dflt : Out)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (S : Finset (Fin n)) (R : Reg) : Game Bool :=
  (sessionG tab (plan S R)).map fun outs => decide (acceptedG acc dflt (plan S R) outs)

theorem batchedSessionL_eq {n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) :
    batchedSessionL H E Lf plan = batchedSessionG (tabAfterL execArith H E Lf) (·.accepted) dfltOutL plan :=
  rfl

end FlockSoundness.Discharge.Exec
