import Proofs.Flock.Soundness.Discharge.HiddenExec.Statements
import Proofs.Flock.Soundness.Discharge.Hidden.Theorems

/-!
# What `Stmt.setupHidden` returns, as this area reads it

`setupOne_hidden`, `setupOne_public`: `setupOne` is `setupHidden` under hidden outputs and `setupH` otherwise.

The rest are what this area reads of `Stmt.setupHidden`, all from [wave2-hidden]'s `Hidden.setupHidden_spec`
(`Discharge/Hidden/`). `setupHidden_tags`, `setupHidden_drawn`, `setupHidden_parseC` and `setupHidden_n` are projections
of `Spec` (`tagsEq`, `pub`, `parse`, and `n`, `blocksReal`, `nbl`); `setupHidden_hm96`, `setupHidden_wf` and
`parse_untyped` are [wave2-hidden]'s (`Hidden/Facts.lean`).
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

/-- `Spec.tagsEq`. -/
theorem setupHidden_tags (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.tags = tags :=
  (Hidden.setupHidden_spec h).tagsEq

/-- `Spec.merkle` at `Spec.parse`'s leaf scheme, which is the tags' ([wave2-hidden]'s `Hidden.setupHidden_hm96`). -/
theorem setupHidden_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 :=
  Hidden.setupHidden_hm96 hl h

/-- `StmtWF` and `RegionsWF` ([wave2-hidden]'s `Hidden.setupHidden_wf`). -/
theorem setupHidden_wf (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    StmtWF st ∧ RegionsWF st :=
  Hidden.setupHidden_wf h

/-- `Spec.pub`, at a draw. -/
theorem setupHidden_drawn {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ HmOut.drawn st.c pop d = .ok st.pub := by
  obtain ⟨pop, hl, -, hd⟩ := (Hidden.setupHidden_spec h).pub
  exact ⟨pop, hl, hd d rfl⟩

/-- `Spec.parse`: the statement's circuit is the one `HmOut.parse` accepts. -/
theorem setupHidden_parseC (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parse tags cf tables = .ok st.c :=
  (Hidden.setupHidden_spec h).parse

/-- `Spec.n`, `Spec.blocksReal`, `Spec.nbl`: `n` is the (drawn) public file's, and `nbl = max 3 (log2ceil ⌈n / c.g⌉)`,
as setupH's (`Integrate.setupH_n`). -/
theorem setupHidden_n (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.n = st.pub.n ∧ st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) := by
  have s := Hidden.setupHidden_spec h
  exact ⟨s.n, by rw [s.nbl, s.blocksReal]⟩

/-- An untyped `HmOut.parse` is `HmRow.parse`'s circuit, then `HmOut.checkRows` ([wave2-hidden]'s
`Hidden.parse_untyped`). -/
theorem parse_untyped {c : Circuit} (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) :
    HmRow.parse tags cf tables = .ok c ∧ HmOut.checkRows c = .ok () :=
  Hidden.parse_untyped ht h

end FlockSoundness.Discharge.HiddenExec
