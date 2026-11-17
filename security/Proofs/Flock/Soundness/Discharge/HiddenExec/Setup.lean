module

public import Proofs.Flock.Soundness.Discharge.Hidden.Theorems
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Defs
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Statements
public import Proofs.Flock.Soundness.Refine.Regions
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.Hidden.Commit
import Proofs.Flock.Soundness.Discharge.Hidden.Facts

/-! What this area reads of `Stmt.setupHidden`, from `Hidden.setupHidden_spec`. -/

@[expose] public section

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

theorem setupHidden_tags (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.tags = tags :=
  (Hidden.setupHidden_spec h).tagsEq

theorem setupHidden_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 :=
  Hidden.setupHidden_hm96 hl h

theorem setupHidden_plain (hl : tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" :=
  Hidden.setupHidden_plain hl h

theorem setupHidden_wf (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    StmtWF st ∧ RegionsWF st :=
  Hidden.setupHidden_wf h

theorem setupHidden_sampled {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ HmOut.drawn st.c pop d = .ok st.pub := by
  obtain ⟨pop, hl, -, hd⟩ := (Hidden.setupHidden_spec h).pub
  exact ⟨pop, hl, hd d rfl⟩

theorem setupHidden_parseC (hi : tags.isTmpl = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parse tags cf tables = .ok st.c :=
  (Hidden.setupHidden_spec h).parseF hi

theorem setupHidden_c_of_parse {c : Circuit} (hc : HmOut.parse tags cf tables = .ok c)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : st.c = c :=
  Except.ok.inj ((setupHidden_parseC (HmOut.parse_isTmpl hc) h).symm.trans hc)

theorem setupHidden_n (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.n = st.pub.n ∧ st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) := by
  have s := Hidden.setupHidden_spec h
  exact ⟨s.n, by rw [s.nbl, s.blocksReal]⟩

theorem parse_untyped {c : Circuit} (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) :
    HmRow.parse tags cf tables = .ok c ∧ HmOut.checkRows c = .ok () :=
  Hidden.parse_untyped ht h

end FlockSoundness.Discharge.HiddenExec
