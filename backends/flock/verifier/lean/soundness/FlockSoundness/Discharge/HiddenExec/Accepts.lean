import FlockSoundness.Discharge.HiddenExec.Setup
import FlockSoundness.Protocol.Discharge.HiddenExec.Accepts
import FlockSoundness.Discharge.Hidden.Theorems
import FlockSoundness.Discharge.HiddenExec.Statements

/-!
# flock-verify's acceptance under either tag

`setupTables_okH` and `setupTables_retainH` are e2e-exec's `setupTables_ok` and `setupTables_retain` under hidden outputs
(`hh : tags.hiddenOutputs = true`), by the same walk with `Stmt.setupHidden` for `Stmt.setupH`; `setupTables_okAny` and
`setupTables_retainAny` put the two tags together through `setupOne`. The acceptance lemmas follow from them as
e2e-exec's do from `setupTables_ok`, with no `hh`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine Game Transcript Model Accounting
open FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}

theorem accepts_setupOne : AcceptsSetupOne := by
  intro I t h
  obtain ⟨spec, sts, hs, hv⟩ := h
  exact ⟨spec, sts, hs, hv, setupTables_setupOne hs⟩

theorem accepts_retainedAny : AcceptsRetainedAny := by
  intro I t hr h
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, -, hd, -⟩ := verify_ok_tables hv
  have hsr : spec.retainRounds = true := (setupTables_retainAny hs).trans hr
  exact ⟨spec, sts, sess, hs, hd, hsr, decode_retained hsr hd⟩

theorem accepts_allRetainedAny : AcceptsAllRetainedAny := by
  intro I t hr h
  obtain ⟨spec, sts, sess, hs, hd, hsr, -⟩ := accepts_retainedAny hr h
  exact ⟨spec, sts, sess, hs, hd, decode_allRetained hsr hd⟩

theorem accepts_tables_hm96Any : AcceptsTablesHm96Any := by
  intro I t h
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, hpos, hschsz, hall⟩ := verify_ok_tables hv
  obtain ⟨-, hsetup⟩ := setupTables_setupOne hs
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hs
  have hpts := decode_points hsess
  rw [Array.size_map] at hsz hpos hschsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, hpos, fun j hj => ⟨hsetup j hj, ?_⟩⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  refine ⟨schs[j], hf, exec_fast100_ge hf, fun hwf hr hm hms => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun p hp => (hpts.2 p hp).trans (hmp j hj).symm)
  exact table_tableAfterL spec (Setup.ofCircuit sts[j]) (stmtOf sts[j] hwf (regionsOf sts[j])) hm rfl rfl
    (ofCircuit_fold sts[j] hwf _) hms PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points) sess hext
    (decode_roundsWF hsess) schs[j] hf j _ _ c h0 h1

theorem accepts_tables_hm96_of_leafAny : AcceptsTablesHm96OfLeafAny := by
  intro I t hl h
  obtain ⟨spec, sts, sess, hs, hd, hsz, hpos, hall⟩ := accepts_tables_hm96Any h
  refine ⟨spec, sts, sess, hs, hd, hsz, hpos, fun j hj => ?_⟩
  obtain ⟨⟨d, hset, hdr⟩, sch, hf, hm22, hrest⟩ := hall j hj
  have hms := setupOne_tagsAt_hm96 hl hset
  exact ⟨⟨d, hset, hdr⟩, hms, sch, hf, hm22, fun hwf hr hm => hrest hwf hr hm hms⟩

end FlockSoundness.Discharge.HiddenExec
