import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup

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

set_option maxHeartbeats 4000000 in
/-- **`setupTables` sets up each table with `setupHidden`** under a tag with hidden outputs (`Exec.setupTables_ok`'s
counterpart). -/
theorem setupTables_okH (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    (count ≤ 1 ∧ ∃ st, Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st ∧ spec = st.spec ∧
        sts = #[st]) ∨
      (1 < count ∧ sts.size = count ∧ (∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        Stmt.setupHidden { tags with identity := tags.tableIdentity count j } cf pf coins tables (some d) partition
          program = .ok sts[j]) ∧ (∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts) ∧ spec.points = 2) := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind, ite_eq_left hh] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  · exact Or.inl ⟨by assumption, _, ok_of_refuse (by assumption), rfl, rfl⟩
  all_goals
    obtain ⟨hsz, hall⟩ := range_mapM_ok (by assumption)
    exact Or.inr ⟨by omega, hsz, fun j hj => ⟨_, ok_of_refuse (hall j hj)⟩, mPts_of_any (by assumption), rfl⟩

set_option maxHeartbeats 4000000 in
/-- **`setupTables` refuses pinned outputs** under a tag with hidden outputs: no table it returns has any
(`HmRow.pinnedCount`). -/
theorem setupTables_pinnedH (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    ∀ (j : ℕ) (hj : j < sts.size), HmRow.pinnedCount sts[j].c.metaJ = 0 := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind, ite_eq_left hh] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  · intro j hj
    have hj0 : j = 0 := by simpa using hj
    subst hj0
    have hr := refuse_ok (by assumption)
    first | exact bne_eq_false_iff_eq.1 hr.2 | simpa using hr.2
  all_goals
    obtain ⟨-, hall⟩ := range_mapM_ok (by assumption)
    intro j hj
    have hr := refuse_ok (hall j hj)
    first | exact bne_eq_false_iff_eq.1 hr.2 | simpa using hr.2

theorem setupTables_okAny : SetupTablesOkAny := by
  intro tags cf pf coins count tables draw partition program work stratified closure spec sts h
  rcases Bool.eq_false_or_eq_true tags.hiddenOutputs with hh | hh
  · rcases setupTables_okH hh h with ⟨hc, st, hst, hs, hsts⟩ | ⟨hc, hsz, hall, hm, hp⟩
    · exact Or.inl ⟨hc, st, (setupOne_hidden hh).trans hst, hs, hsts⟩
    · refine Or.inr ⟨hc, hsz, fun j hj => ?_, hm, hp⟩
      obtain ⟨d, hd⟩ := hall j hj
      exact ⟨d, (setupOne_hidden (tags := { tags with identity := tags.tableIdentity count j }) hh).trans hd⟩
  · rcases setupTables_ok hh h with ⟨hc, st, hst, hs, hsts⟩ | ⟨hc, hsz, hall, hm, hp⟩
    · exact Or.inl ⟨hc, st, (setupOne_public hh).trans hst, hs, hsts⟩
    · refine Or.inr ⟨hc, hsz, fun j hj => ?_, hm, hp⟩
      obtain ⟨d, hd⟩ := hall j hj
      exact ⟨d, (setupOne_public (tags := { tags with identity := tags.tableIdentity count j }) hh).trans hd⟩

/-- Every table `setupTables` returns is `setupOne`'s statement, under the table's tags (`tagsAt`) and at some draw
(the session's, for one table): `Exec.setupTables_setupH` without `hh`. -/
theorem setupTables_setupOne
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    0 < sts.size ∧ ∀ (j : ℕ) (hj : j < sts.size), ∃ d,
      setupOne (tagsAt tags count j) cf pf coins tables d partition program = .ok sts[j] ∧
        (count ≤ 1 → d = draw) := by
  rcases setupTables_okAny h with ⟨hc, st, hst, -, rfl⟩ | ⟨hc, hsz, hall, -⟩
  · refine ⟨by simp, fun j hj => ⟨draw, ?_, fun _ => rfl⟩⟩
    have hj0 : j = 0 := by simpa using hj
    subst hj0
    have ht : tagsAt tags count 0 = tags := ite_eq_left hc
    rw [ht]
    exact hst
  · refine ⟨by omega, fun j hj => ?_⟩
    obtain ⟨d, hd⟩ := hall j hj
    refine ⟨some d, ?_, fun h1 => absurd h1 (by omega)⟩
    have ht : tagsAt tags count j = { tags with identity := tags.tableIdentity count j } :=
      ite_eq_right (by omega)
    rw [ht]
    exact hd

/-- The session's two link points have every table's length: `Exec.setupTables_points` without `hh`. -/
theorem setupTables_pointsAny
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    spec.points = 2 ∧ ∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts := by
  rcases setupTables_okAny h with ⟨-, st, -, rfl, rfl⟩ | ⟨-, -, -, hm, hp⟩
  · exact ⟨rfl, fun j hj => by
      have : j = 0 := by simpa using hj
      subst this
      rfl⟩
  · exact ⟨hp, hm⟩

theorem accepts_setupOne : AcceptsSetupOne := by
  intro I t h
  obtain ⟨spec, sts, hs, hv⟩ := h
  exact ⟨spec, sts, hs, hv, setupTables_setupOne hs⟩

theorem accepts_oneAny : AcceptsOneAny := by
  intro I t hc h
  obtain ⟨spec, sts, hs, hv⟩ := h
  rcases setupTables_okAny hs with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  · exact ⟨st, hst, by simpa using hv⟩
  · omega

set_option maxHeartbeats 4000000 in
/-- The session's spec retains rounds exactly when its tags do, under hidden outputs (`Exec.setupTables_retain`'s
counterpart). -/
theorem setupTables_retainH (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    spec.retainRounds = tags.retainRounds := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind, ite_eq_left hh] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  · exact congrArg Tags.retainRounds (setupHidden_tags (ok_of_refuse (by assumption)))
  all_goals rfl

/-- The session's spec retains rounds exactly when its tags do, for either tag. -/
theorem setupTables_retainAny
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    spec.retainRounds = tags.retainRounds := by
  cases hh : tags.hiddenOutputs
  · exact setupTables_retain hh h
  · exact setupTables_retainH hh h

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

theorem accepts_rep_roundsAny : AcceptsRepRoundsAny := by
  intro I t hr h
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verify_ok_tables hv
  have hret := decode_allRetained ((setupTables_retainAny hs).trans hr) hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨hdim, -⟩ := fast100_dims _ _ hf
  have hlv : ∀ i < schs[j].levels.size, LevelSeam (schs[j].levels.getD i default) := fun i hi =>
    fast100_level _ _ hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hm : 14 ≤ (Setup.ofCircuit sts[j]).m := by have := exec_fast100_ge hf; omega
  exact ⟨schs[j], hf, rep_rounds_retained hm hdim hlv hret h0, rep_rounds_retained hm hdim hlv hret h1⟩

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

/-- `setupH_tagsAt_hm96` for either tag: a table's statement under tags that implement `hm96-sha512/v1` commits with
`hm96Sha512`. -/
theorem setupOne_tagsAt_hm96 {j : ℕ} {st : Flock.Stmt} (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : setupOne (tagsAt tags count j) cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 := by
  cases hh : (tagsAt tags count j).hiddenOutputs
  · exact setupH_tagsAt_hm96 hl ((setupOne_public hh).symm.trans h)
  · exact setupHidden_hm96 ((tagsAt_merkleLeaf tags count j).trans hl) ((setupOne_hidden hh).symm.trans h)

theorem accepts_tables_hm96_of_leafAny : AcceptsTablesHm96OfLeafAny := by
  intro I t hl h
  obtain ⟨spec, sts, sess, hs, hd, hsz, hpos, hall⟩ := accepts_tables_hm96Any h
  refine ⟨spec, sts, sess, hs, hd, hsz, hpos, fun j hj => ?_⟩
  obtain ⟨⟨d, hset, hdr⟩, sch, hf, hm22, hrest⟩ := hall j hj
  have hms := setupOne_tagsAt_hm96 hl hset
  exact ⟨⟨d, hset, hdr⟩, hms, sch, hf, hm22, fun hwf hr hm => hrest hwf hr hm hms⟩

theorem accepts_verdict_hm96Any : AcceptsVerdictHm96Any := by
  intro I t hl h
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verify_ok_tables hv
  obtain ⟨-, hsetup⟩ := setupTables_setupOne hs
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hs
  have hpts := decode_points hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨d, hset, -⟩ := hsetup j hj
  refine ⟨schs[j], hf, fun hwf hr hm => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun p hp => (hpts.2 p hp).trans (hmp j hj).symm)
  exact table_verdict_hm96 spec (Setup.ofCircuit sts[j]) (stmtOf sts[j] hwf (regionsOf sts[j])) hm rfl rfl
    (ofCircuit_fold sts[j] hwf _) (setupOne_tagsAt_hm96 hl hset) _ sess hext (decode_roundsWF hsess) schs[j] hf j _ _
    c h0 h1

/-- `setupOne_tagsAt_hm96` for every leaf the non-ZK headlines cover: under `hm96-sha512/v1` leaves a table's statement
commits with `hm96Sha512`, and under hidden outputs with plain SHA-512 leaves with `MerkleScheme.sha512`. -/
theorem setupOne_tagsAt_leaf {j : ℕ} {st : Flock.Stmt}
    (hl : tags.merkleLeaf = "hm96-sha512/v1" ∨
      (tags.hiddenOutputs = true ∧ tags.merkleLeaf = "flock-leaf/sha512-unsalted"))
    (h : setupOne (tagsAt tags count j) cf pf coins tables draw partition program = .ok st) :
    (tags.merkleLeaf = "hm96-sha512/v1" ∧ st.merkle = MerkleScheme.hm96Sha512) ∨
      (tags.merkleLeaf = "flock-leaf/sha512-unsalted" ∧
        st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") := by
  rcases hl with hl | ⟨hh, hl⟩
  · exact Or.inl ⟨hl, setupOne_tagsAt_hm96 hl h⟩
  · have hh' : (tagsAt tags count j).hiddenOutputs = true := by unfold tagsAt; split <;> exact hh
    exact Or.inr ⟨hl, setupHidden_plain ((tagsAt_merkleLeaf tags count j).trans hl)
      ((setupOne_hidden hh').symm.trans h)⟩

/-- **flock-verify's acceptance makes each table's verdict accept, at the model's leaf of the tags' leaf scheme**
(`leafOfId`): `accepts_verdict_hm96Any` under `hm96-sha512/v1` leaves, and under hidden outputs with plain SHA-512
leaves the verdict at `leafPlain` (`table_verdict_leaf`). -/
theorem accepts_verdict_leafAny {I : Exec.Inputs} {t : Exec.Transcript}
    (hl : I.tags.merkleLeaf = "hm96-sha512/v1" ∨
      (I.tags.hiddenOutputs = true ∧ I.tags.merkleLeaf = "flock-leaf/sha512-unsalted"))
    (h : Accepts I t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m),
        ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ ∧
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ ∧
          (tableOutL H512 enc512 (leafOfId I.tags.merkleLeaf) (capOfRoot sess.rootB[j]!) (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!])).accepted =
            true := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verify_ok_tables hv
  obtain ⟨-, hsetup⟩ := setupTables_setupOne hs
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hs
  have hpts := decode_points hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨d, hset, -⟩ := hsetup j hj
  refine ⟨schs[j], hf, fun hwf hr hm => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun p hp => (hpts.2 p hp).trans (hmp j hj).symm)
  exact table_verdict_leaf spec (Setup.ofCircuit sts[j]) (stmtOf sts[j] hwf (regionsOf sts[j])) hm rfl rfl
    (ofCircuit_fold sts[j] hwf _) (setupOne_tagsAt_leaf hl hset) _ sess hext (decode_roundsWF hsess) schs[j] hf j _ _
    c h0 h1

end FlockSoundness.Discharge.HiddenExec
