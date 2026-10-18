import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.Zk.Link

/-!
# The first masked send is the link message, at several tables

`Zk/Link.lean` at the masked session on each table's share:

* `sessionS_link`: a pure masked session that returns sent the link message of its level-0 caps, with no public words,
  first (`sessionSZk_link` at `traces`).
* `linkFirst_goal` (`LinkFirstZk`), for a SHA-512 of 64-byte outputs: at most `Tags.MAX_TABLES` caps, each table's at
  most `2^56` digests of 64 bytes (every table commits with `hm96-sha512/v1`, at its `--zk` schedule).
-/

open Flock FlockDeploy

namespace FlockComplete.ZkTables

open With

/-- **The first masked send is the link message**: a pure masked session that returns ran its schedules and level-0
commitments (their verdict-monad runs return), then sent the link message of the caps, with no public words, first. -/
theorem sessionS_link {H : Claims.Msg → Claims.Msg} {resp : ℕ → Claims.Msg} {spec : SessionSpec}
    {sts : Array Flock.Stmt} {w : Claims.Flock.Wit} {ω : TapeZk} {rec : String} {pfs : Array ByteArray}
    {bodies : Array Claims.Msg} (h : sessionS H resp spec sts w ω = (.ok (rec, pfs), bodies)) :
    ∃ schs cms, (schedsPZk (tabsOf sts) : V _) = .ok schs ∧
      (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) (tabsOf sts) schs (tablesP (tabsOf sts) (traces sts w)) ω :
        V _) = .ok cms ∧
      bodies.toList.head? = some (linkBody (cms.map fun c => flat c.cap).toList
        ((tabsOf sts).map fun _ => ByteArray.empty).toList) := by
  unfold sessionS at h
  rw [sessionPZk_eq, go_bind, schedsPZk_quiet] at h
  cases hs : (schedsPZk (tabsOf sts) : V _) with
  | error e => rw [hs] at h; cases h
  | ok schs =>
    rw [hs] at h
    dsimp only at h
    rw [go_bind, commitsPZk_quiet] at h
    cases hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) (tabsOf sts) schs
        (tablesP (tabsOf sts) (traces sts w)) ω : V _) with
    | error e => rw [hc] at h; cases h
    | ok cms =>
      rw [hc] at h
      dsimp only at h
      rw [go_bind, asksS_link_go] at h
      dsimp only at h
      rw [Array.size_empty] at h
      refine ⟨schs, cms, rfl, hc, ?_⟩
      obtain ⟨bs, -, hl⟩ := afterLinkPZk_sim H resp spec (tabsOf sts) (MaskRank.masksOf sts) (traces sts w) ω
        schs
        (tablesP (tabsOf sts) (traces sts w)) cms (coinOf (resp 0)) (#[].push (linkBody
          (cms.map fun c => flat c.cap).toList ((tabsOf sts).map fun _ => ByteArray.empty).toList))
      rw [h] at hl
      dsimp only at hl
      rw [hl]
      simp

/-- **The first masked send is the link message**, each length in its word: the goal `LinkFirstZk`, for a SHA-512 of
64-byte outputs. -/
theorem linkFirst_goal (H : Claims.Msg → Claims.Msg) (hH : ∀ m, (H m).length = 64)
    (rel : Readable.Parsed → Readable.Relation) : LinkFirstZk H rel := by
  intro o
  have h64 : ∀ b, (Claims.hOf H b).size = 64 := fun b => by
    simp [Claims.hOf, Claims.ofMsg, ByteArray.size, hH]
  obtain ⟨schs, cms, hs, hc, hh⟩ := sessionS_link o.ran
  obtain ⟨-, hmax, -, hall⟩ := o.tables
  unfold schedsPZk at hs
  obtain ⟨hs1, hs2⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hs
  unfold commitsPZk at hc
  obtain ⟨hcs, hcall⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok hc
  have htabs : (tabsOf o.sts).size = o.sts.size := by simp [tabsOf]
  have hmax' : Tags.MAX_TABLES = 64 := rfl
  refine ⟨_, _, hh, ?_, ?_, ?_, ?_⟩
  · simp only [Array.toList_map, List.length_map, Array.length_toList, hcs, htabs]
    omega
  · simp only [Array.toList_map, List.length_map, Array.length_toList, htabs]
    omega
  · intro b hb
    obtain ⟨c, hcm, rfl⟩ : ∃ c ∈ cms, flat c.cap = b := by simpa using hb
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hcm
    have hjt : j < o.sts.size := by omega
    obtain ⟨-, -, hms⟩ := hall j hjt
    have hv := hs2 j (by omega) (by omega)
    have hcj := hcall j hj
    simp only [tabsOf, Array.getElem_map, Setup.ofCircuit, monadLift_self] at hv
    rw [getElem!_pos (tabsOf o.sts) j (by omega), getElem!_pos schs j (by omega)] at hcj
    simp only [tabsOf, Array.getElem_map, Setup.ofCircuit, hms, commitPZk] at hcj
    obtain ⟨hcap, hcap64⟩ := commitRowsP_cap h64 _ _ _ hcj
    rw [flat_size hcap64]
    have hd := zkSchedule_d0 hv
    calc 64 * cms[j].cap.size ≤ 64 * 2 ^ 56 :=
          Nat.mul_le_mul_left _ (hcap.trans (Nat.pow_le_pow_right (by norm_num) hd))
      _ < 2 ^ 64 := by norm_num
  · simp

end FlockComplete.ZkTables
