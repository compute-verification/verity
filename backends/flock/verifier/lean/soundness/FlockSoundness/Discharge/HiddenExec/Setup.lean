import FlockSoundness.Discharge.HiddenExec.Statements

/-!
# What `Stmt.setupHidden` returns, as this area reads it

`setupOne_hidden`, `setupOne_public`: `setupOne` is `setupHidden` under hidden outputs and `setupH` otherwise.

The rest are what this area reads of `Stmt.setupHidden`'s statements. [wave2-hidden] owns them (`Discharge/Hidden/`,
`setupHidden_spec` and its `Spec` in `Spec.lean`, `parse_untyped` in `Compare.lean`), under these names and statements;
until its branch lands they are stubs here, and on merge each is deleted for its own (`setupHidden_parseC` and
`setupHidden_n` become projections of `Spec`: its `parse`, and its `n`, `blocksReal` and `nbl`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}

theorem setupOne_hidden (hh : tags.hiddenOutputs = true) :
    setupOne tags cf pf coins tables draw partition program =
      Stmt.setupHidden tags cf pf coins tables draw partition program := by
  simp [setupOne, hh]

theorem setupOne_public (hh : tags.hiddenOutputs = false) :
    setupOne tags cf pf coins tables draw partition program =
      Stmt.setupH tags cf pf coins tables draw partition program := by
  simp [setupOne, hh]

/-- [wave2-hidden]'s `setupHidden_tags` (`Spec.tags`). -/
theorem setupHidden_tags (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.tags = tags := by
  sorry

/-- [wave2-hidden]'s `setupHidden_hm96`. -/
theorem setupHidden_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 := by
  sorry

/-- [wave2-hidden]'s `setupHidden_wf`. -/
theorem setupHidden_wf (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    StmtWF st ∧ RegionsWF st := by
  sorry

/-- [wave2-hidden]'s `setupHidden_drawn` (`Spec.pub` at a draw). -/
theorem setupHidden_drawn {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ HmOut.drawn st.c pop d = .ok st.pub := by
  sorry

/-- `Spec.parse`: the statement's circuit is the one `HmOut.parse` accepts. -/
theorem setupHidden_parseC (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parse tags cf tables = .ok st.c := by
  sorry

/-- `Spec.n`, `Spec.blocksReal`, `Spec.nbl`: `n` is the (drawn) public file's, and `nbl = max 3 (log2ceil ⌈n / c.g⌉)`,
as setupH's (`Integrate.setupH_n`). -/
theorem setupHidden_n (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.n = st.pub.n ∧ st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) := by
  sorry

/-- [wave2-hidden]'s `parse_untyped`: an untyped `HmOut.parse` is `HmRow.parse`'s circuit, then `HmOut.checkRows`. -/
theorem parse_untyped {c : Circuit} (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) :
    HmRow.parse tags cf tables = .ok c ∧ HmOut.checkRows c = .ok () := by
  sorry

end FlockSoundness.Discharge.HiddenExec
