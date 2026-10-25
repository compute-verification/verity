import Proofs.Flock.Soundness.Discharge.FailClosed.ZkRegistered
import Proofs.Flock.Soundness.Discharge.Composed.TagsJ

/-!
# The tags of a statement `verify --zk --registered F` accepts

`ZkReg.AcceptsZKR` runs `checkRegistered` on every table's statement, and `Registered.check` with the verifier's own
reads accepts only a statement over a shared-row public file (`ZkReg.check_tables`). At a record that carries a draw,
each table's statement is drawn from the population the statement check loads (`pubZ I`), and drawing keeps its tables
(`Layout.drawn_spec`), so `pubZ I` is a shared-row file (`tables_of_acceptsZKR`). The statement check passes a shared-row
file only at `hiddenOutputs && registered` and `hm96-sha512/v1` leaves (`Checks.tables`, `ProvedScope.sharedRowsOk`):

* `not_scopeZ_of_acceptsZKR`: no such outcome passes `ScopeZ`, the check without `--registered`;
* `hidden_of_acceptsZKR`, `tagsOkZHJ_of_acceptsZKR`: at one that passes `ScopeZR`, the statement has hidden outputs, and
  the hidden-output chain's tags `TagsOkZHJ` hold.
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec
open FlockSoundness.Discharge.ZkReg (AcceptsZKR checkRegistered_ok check_tables)
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ)

variable {I : Inputs} {own : Array Registered.Port} {t : Exec.Transcript} {d : Lean.Json}

/-- **An accepted `--zk --registered` statement's population is a shared-row file**: at a transcript whose record
carries a draw and that `AcceptsZKR` accepts, under a statement check that passes (with or without `--registered`), the
public file the check loads has tables. -/
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
      obtain ⟨pop, hpop, hdr⟩ := Layout.setupH_drawn (hst 0 h0)
      rw [Composed.loadPublic_tagsAt, ← Except.ok.inj hc0, hck.load hh'] at hpop
      obtain rfl := Except.ok.inj hpop
      exact ⟨_, _, 0, h0, hdr⟩
    · have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
      obtain ⟨hsz, hst⟩ := ZkHidden.setupOf_tableJH hh hd hs
      have h0 : 0 < sts.size := by rw [hsz]; exact Composed.nTab_pos I
      have s := Hidden.setupHidden_spec' (hst 0 h0)
      have hc0 := s.parse
      rw [ZkHidden.parseH_tagsAt hty, hck.parseH hh'] at hc0
      obtain ⟨pop, hpop, -, hdr⟩ := s.pub
      rw [ZkHidden.loadPublicH_tagsAt, ← Except.ok.inj hc0, hck.loadH hh'] at hpop
      obtain rfl := Except.ok.inj hpop
      exact ⟨{ sts[0].c with outPorts := #[] }, _, 0, h0, hdr _ rfl⟩
  obtain ⟨c', d', i, hi, hdr⟩ := key
  obtain ⟨tb, -, htb, -⟩ := check_tables (checkRegistered_ok (array_forM_ok hreg i hi))
  obtain ⟨D, -, -, -, hpt, -⟩ := Layout.drawn_spec hdr
  rw [htb] at hpt
  cases hp : (pubZ I).tables with
  | none => rw [hp] at hpt; simp at hpt
  | some a => exact ⟨a, rfl⟩

/-- **No accepted `--zk --registered` outcome passes `ScopeZ`**: `ScopeZ` is the statement check without the verifier's
registered reads, which passes no shared-row public file. -/
theorem not_scopeZ_of_acceptsZKR (hd : recordDraw t.record = some d) (h : AcceptsZKR I own t) : ¬ ScopeZ I := by
  intro hs
  obtain ⟨tb, htb⟩ := tables_of_acceptsZKR hs hd h
  exact absurd ((ScopeZ.tables hs).symm.trans htb) (by simp)

/-- **An accepted `--zk --registered` statement that passes `ScopeZR` has hidden outputs.** -/
theorem hidden_of_acceptsZKR (hs : ScopeZR I) (hd : recordDraw t.record = some d) (h : AcceptsZKR I own t) :
    I.tags.hiddenOutputs = true := by
  obtain ⟨tb, htb⟩ := tables_of_acceptsZKR hs hd h
  rcases Checks.tables hs with hn | ⟨hh, -, -⟩
  · exact absurd (hn.symm.trans htb) (by simp)
  · exact (zkTags_hiddenOutputs I).symm.trans hh

/-- **The hidden-output chain's tags at an accepted `--zk --registered` statement that passes `ScopeZR`.** -/
theorem tagsOkZHJ_of_acceptsZKR (hs : ScopeZR I) (hd : recordDraw t.record = some d) (h : AcceptsZKR I own t) :
    TagsOkZHJ I :=
  hs.tagsOkZHJ (hidden_of_acceptsZKR hs hd h)

end FlockSoundness.Discharge.FailClosed
