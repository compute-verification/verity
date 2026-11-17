module

public import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.Defs
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
import Proofs.Flock.Soundness.Discharge.Composed.TagsJ
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkRegistered
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-! A statement `verify --zk --registered F` accepts has a shared-row public file, so it fails `ScopeZ` and, where it passes `ScopeZR`, has hidden outputs and the hidden-output chain's tags. -/

@[expose] public section

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec
open FlockSoundness.Discharge.ZkReg (AcceptsZKR checkRegistered_ok check_tables)
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ)

variable {I : Inputs} {own : Array Registered.Port} {t : Exec.Transcript} {d : Lean.Json}

theorem tables_of_acceptsZKR {reg : Bool} (hck : Checks (zkTags I) true I reg) (hd : recordDraw t.record = some d)
    (h : AcceptsZKR I own t) : ∃ tb, (pubZ I).tables = some tb := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := h
  have hty : (zkTags I).typed = false := hck.typed
  have key : ∃ (c' : Circuit) (d' : Lean.Json) (i : ℕ) (hi : i < sts.size),
      HmRow.drawn c' (pubZ I) d' = .ok sts[i].pub := by
    cases hh : I.tags.hiddenOutputs
    · have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
      obtain ⟨hsz, hst⟩ := Composed.setupOf_tableJ hh hd hs
      have h0 : 0 < sts.size := by rw [hsz]; exact Composed.nTab_pos I
      have hc0 := (setupH_spec ((tagsAt_typed _ _ _).trans hty) (hst 0 h0)).1
      rw [Composed.parse_tagsAt, hck.parse hh'] at hc0
      obtain ⟨pop, hpop, hdr⟩ := Layout.setupH_sampled (hst 0 h0)
      rw [Composed.loadPublic_tagsAt, ← Except.ok.inj hc0, hck.load hh'] at hpop
      obtain rfl := Except.ok.inj hpop
      exact ⟨_, _, 0, h0, hdr⟩
    · have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
      obtain ⟨hsz, hst⟩ := ZkHidden.setupOf_tableJH hh hd hs
      have h0 : 0 < sts.size := by rw [hsz]; exact Composed.nTab_pos I
      have s := Hidden.setupHidden_spec' (hst 0 h0)
      have hc0 : HmOut.loadPublic (zkTags I) sts[0].c I.publicFile = .ok (pubZ I) := by
        cases hf : (circC (zkTags I) I).tmpl with
        | none =>
          have hp := hck.parseH hh' hf
          have h1 := s.parseF ((tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl hp))
          rw [ZkHidden.parseH_tagsAt hty, hp] at h1
          rw [← Except.ok.inj h1]
          exact hck.loadH hh'
        | some tu =>
          obtain ⟨-, -, hn, -⟩ := hck.templated hf
          have h1 := s.parse
          rw [show tagsAt (zkTags I) I.count 0 = zkTags I by simp [tagsAt, hn], hck.parseC hh'] at h1
          rw [← Except.ok.inj h1]
          exact hck.loadHC hh'
      obtain ⟨pop, hpop, -, hdr⟩ := s.pub
      rw [ZkHidden.loadPublicH_tagsAt, hc0] at hpop
      obtain rfl := Except.ok.inj hpop
      exact ⟨{ sts[0].c with outPorts := #[] }, _, 0, h0, hdr _ rfl⟩
  obtain ⟨c', d', i, hi, hdr⟩ := key
  obtain ⟨tb, -, htb, -⟩ := check_tables (checkRegistered_ok (array_forM_ok hreg i hi))
  obtain ⟨D, -, -, -, hpt, -⟩ := Layout.sampled_spec hdr
  rw [htb] at hpt
  cases hp : (pubZ I).tables with
  | none => rw [hp] at hpt; simp at hpt
  | some a => exact ⟨a, rfl⟩

theorem not_scopeZ_of_acceptsZKR (hd : recordDraw t.record = some d) (h : AcceptsZKR I own t) : ¬ ScopeZ I := by
  intro hs
  obtain ⟨tb, htb⟩ := tables_of_acceptsZKR hs hd h
  exact absurd ((ScopeZ.tables hs).symm.trans htb) (by simp)

theorem hidden_of_acceptsZKR (hs : ScopeZR I) (hd : recordDraw t.record = some d) (h : AcceptsZKR I own t) :
    I.tags.hiddenOutputs = true := by
  obtain ⟨tb, htb⟩ := tables_of_acceptsZKR hs hd h
  rcases Checks.tables hs with hn | ⟨hh, -, -⟩
  · exact absurd (hn.symm.trans htb) (by simp)
  · exact (zkTags_hiddenOutputs I).symm.trans hh

theorem tagsOkZHJ_of_acceptsZKR (hs : ScopeZR I) (hd : recordDraw t.record = some d) (h : AcceptsZKR I own t) :
    TagsOkZHJ I :=
  hs.tagsOkZHJ (hidden_of_acceptsZKR hs hd h)

end FlockSoundness.Discharge.FailClosed
