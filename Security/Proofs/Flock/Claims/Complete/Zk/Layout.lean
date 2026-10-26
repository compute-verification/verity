import Proofs.Flock.Claims.Complete.Zk.MaskRank
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Discharge.Hidden.Spec

/-!
# The mask slot's layout at every in-scope table

`MaskRank.LayoutOk`, what the rank bound reads of a table, holds at every table of a masked session in scope
(`layoutOk_of_scope`): `Zk.shapeOk` from the `--zk` setup (`Custody.setup_shapeOk`), and the slot's place in the
block and `m = k_log + nbl` from the table's setup, public (`setupH_mask`) or with hidden outputs (`Hidden.Spec`, whose
circuit is `HmOut.parse`'s: `hmOut_parse_mask`).
-/

namespace FlockComplete.MaskRank

open Flock FlockDeploy FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge

/-- **`HmOut.parse`'s mask slot is a slot of at least a word inside the block**: every branch ends in `HmRow.check`
(`hm_check_mask`), and the record updates after it keep the ranges and `k_log`. -/
theorem hmOut_parse_mask {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmOut.parse tags bytes tables = .ok c) :
    ∀ r ∈ c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ c.kLog := by
  have hjp : ∀ c' : Circuit, (do HmOut.checkRows c'; pure c' : V Circuit) = .ok c → c = c' := by
    intro c' h'
    obtain ⟨_, -, h'⟩ := ebind_ok h'
    simp only [pure, Except.pure, Except.ok.injEq] at h'
    exact h'.symm
  have hi := HmOut.parse_isTmpl h
  unfold HmOut.parse at h
  rw [if_neg (by simp [hi])] at h
  dsimp only at h
  split at h
  · obtain ⟨c', hc', h⟩ := ebind_ok h
    obtain rfl := hjp c' h
    obtain ⟨ls, hls⟩ := hm_parse_check_any hc'
    exact hm_check_mask hls
  · obtain ⟨⟨k, r⟩, -, h⟩ := ebind_ok h
    dsimp only at h
    cases r with
    | inr t =>
      obtain ⟨c', hc', h⟩ := ebind_ok h
      obtain ⟨c'', hc'', h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at hc''
      subst hc''
      obtain rfl := hjp _ h
      obtain ⟨ls, hls⟩ := hm_parse_check_any hc'
      dsimp only
      exact hm_check_mask hls
    | inl u =>
      obtain ⟨c', hc', h⟩ := ebind_ok h
      obtain ⟨c'', hc'', h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at hc''
      subst hc''
      obtain rfl := hjp _ h
      obtain ⟨ls, hls⟩ := hm_parse_check_any hc'
      dsimp only
      exact hm_check_mask hls

/-- **`hmOut_parse_mask` at the committed circuit** (`HmOut.parseCommit`): a templated file's is `HmRow.parse`'s with the
template's net first, which passed `HmRow.check` (`hm_parse_checkPre`). -/
theorem hmOut_parseCommit_mask {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmOut.parseCommit tags bytes tables = .ok c) :
    ∀ r ∈ c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ c.kLog := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    exact hmOut_parse_mask h
  · rw [parseCommit_tmpl hi] at h
    obtain ⟨tu, c₀, hp, -, -, rfl⟩ := TmplValue.parseTmpl_cases h
    obtain ⟨_, hls⟩ := hm_parse_checkPre hp
    exact hm_check_mask hls

/-- **A table's setup places its mask slot in the block**, and its `m` is `k_log + nbl`, with or without hidden
outputs. -/
theorem setupOne_layout {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) :
    st.m = st.c.kLog + st.nbl ∧
      ∀ r ∈ st.c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ st.c.kLog := by
  cases hh : tags.hiddenOutputs
  · rw [HiddenExec.setupOne_public hh] at h
    exact ⟨(setupH_mask h).2, (setupH_mask h).1⟩
  · rw [HiddenExec.setupOne_hidden hh] at h
    have S := Hidden.setupHidden_spec' h
    exact ⟨S.m, hmOut_parseCommit_mask S.parse⟩

/-- **Every table of a masked session in scope has the layout the rank bound reads** (`LayoutOk`). -/
theorem layoutOk_of_scope {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hsc : Claims.Flock.Scope (deployZk rel) x)
    (hs : setupOf true x = some (s, spec, sts)) : ∀ st ∈ sts, LayoutOk st := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) x).1 hsc
  have hi := (setupOf_some hs).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := s.read) hp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset2 := Ins.setup_read (setupOf_some hs).2
  rw [← ht] at hset2
  unfold Readable.Read.setup at hset2
  rw [hpi] at hset2
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset2
  have hr : readOf true x = some s.read := by unfold readOf; rw [hi]; rfl
  have hz := Claims.Bridge.Custody.setup_shapeOk (setupOf_some hs).2 (readOf_zk hr)
  intro st hst
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hst
  obtain ⟨d, hd, -⟩ := hall j hj
  obtain ⟨hm, hmask⟩ := setupOne_layout hd
  exact ⟨Readable.array_forM_ok hz j hj, hm, hmask⟩

end FlockComplete.MaskRank
