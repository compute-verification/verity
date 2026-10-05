import FlockSoundness.Discharge.Integrate.Sites
import FlockSoundness.Protocol.Discharge.Integrate.Bridge

/-!
# The accepted statement `Placed.Accepted` reads is the one the refinement plays

`Placed.Accepted` reads an accepted statement as `Layout.model` (the matrices `placedA`, `placedB` of flock-verifier's
`FlockLevel3.Placed`); the executable's refinement (`stmtOf_fold`, `accepts_tables_hm96`) plays at `Refine.stmtOf` (`matA`,
`matB`). They read the same layout two ways:
* the covering range: `FlockLevel3.lastCover` is a left fold, `Refine.lastCover` a right recursion (`lastCover_eq`);
* Δ: a filter's length against `List.count` (`deltaEntry_eq`);
* a net's entries: `sparseEntry` counts a column at any index, `spCnt` only below the matrix's width (`sparse_eq`). At a
  net range's local columns (below `2^slotLog`) the width check holds: every net `HmRow.parse` builds is `2^unitLog`
  wide (`net_width`), and a net range's slots are its net's rows;
* a lookup slot's table rows: `lookupEntry` against `lkCnt`, which reads bit indices below 32 (`lookup_eq`, as
  `Lookup.build` refuses more than 32 output bits);
* a compression-leaf range: `setupH`'s leaf is the default one, whose matrices are empty (`setupH_leaf`).

`model_eq_stmtOf`: for a statement `Stmt.setupH` accepted on the untyped path, `Layout.model regs = stmtOf st hwf regs` at
any regions. `table_stmtOf`, `tableAt_stmtOf`: a drawn table of `sitesAt` is `stmtOf` of its statement at `regionsOf`.
-/

namespace FlockSoundness.Discharge.Integrate

open Flock FlockLevel3 Refine

/-! ### The covering range and Δ -/

/-! ### A slot's entries -/

/-! ### The nets `HmRow.parse` builds -/

/-! ### The statement -/

section Model

variable {st : Stmt} {tags : Tags} (P : ParseFacts tags st.c) (L : Layout st) (hleaf : st.leaf = default)
include P L hleaf

end Model

/-- **`tableAt`'s table at a draw that sets up is `stmtOf` of a statement `Stmt.setupH` accepted** on the session's inputs
and the draw's JSON, with the same statement as any other setup of that draw. -/
theorem tableAt_stmtOf {I : Exec.Inputs} {mPts n : ℕ} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) {S : Finset (Fin n)} (x : DrawSetup I mPts S (dj S)) :
    ∃ y : DrawSetup I mPts S (dj S), y.st = x.st ∧
      (tableAt ht dj x₀ S).T.S = stmtOf y.st y.wf.1 (regionsOf y.st) := by
  obtain ⟨y, hy, e⟩ := tableAt_stmt ht dj x₀ x
  exact ⟨y, hy, by rw [e]; exact table_stmtOf ht y⟩

end FlockSoundness.Discharge.Integrate
