import Proofs.Flock.Claims.Complete.Link
import Proofs.Flock.Claims.Complete.Zk.Discharge

/-!
# The first masked send is the link message

`Link.lean` at the masked session:

* **The masked session in three parts** (`sessionPZk_eq`): the `--zk` schedules (`schedsPZk`), the masked level-0
  commitments (`commitsPZk`), then the link message and what follows its coin (`afterLinkPZk`), whose sends only append
  to the bodies (`afterLinkPZk_sim`). The schedules and the commitments are quiet (`schedsPZk_quiet`,
  `commitsPZk_quiet`).
* `sessionSZk_link`: a pure masked session that returns sent the link message of its level-0 caps, with no public words,
  first.
* `commitRowsP_cap`: a level's cap under `hm96-sha512/v1` is at most `2^d` digests of 64 bytes, whatever the rows;
  the `--zk` schedule keeps fast100's level-0 `d` (`zkSchedule_d0`).
* `linkFirstZk_goal` (`LinkFirstZk`), for a SHA-512 of 64-byte outputs, as `linkFirst_goal`.
-/

open Flock FlockDeploy

namespace FlockComplete

open With

/-! ## The masked session in three parts -/

/-- The masked session's schedules: each table's `--zk` schedule and `t_pad`. -/
def schedsPZk {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (tabs : Array Setup) :
    M (Array (Schedule × ℕ)) :=
  tabs.mapM fun st => (monadLift (Zk.schedule st.m) : M (Schedule × Nat))

/-- Each table's masked level-0 commitment, salted from the tape's level `0`. -/
def commitsPZk {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (hs : Hashes M)
    (tabs : Array Setup) (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128)) (tape : TapeZk) :
    M (Array Commit) :=
  (Array.range tabs.size).mapM fun i =>
    commitPZk hs tabs[i]!.merkle (schs[i]!.1.levels.getD 0 default) schs[i]!.2 tbls[i]! (tape.extra i)
      (tape.lanePad i) (tape.salt i 0 0)

/-- The masked session from the link message's coin `c1` on: the rep streams, the record and the proof files. -/
def afterLinkPZk {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (io : Asks M)
    (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk)
    (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128) :
    M (String × Array ByteArray) := do
  let rootB := cms.map fun c => flat c.cap
  let points := (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)
  let mut streams : Array (Array Round) := #[]
  let mut proofs : Array ByteArray := #[]
  for (i, r, _) in spec.repStreams do
    let st := tabs[i]!
    let sch := schs[i]!.1
    let extras ← (monadLift (st.extra points) : M _)
    let pads := padsOf tape i r (Zk.PadLayout.new st.m st.kLog sch.k0)
    let ic ← innerCommitP hs pads (tape.innerSalt i r)
    let dom := (spec.domain spec.tables[i]! r).toUTF8
    let (p, t) ← (repPZk io hs st sch schs[i]!.2 r dom trs[i]! tbls[i]! cms[i]! extras pads ic (tape.extra i)
      (tape.lanePad i) (tape.salt i r) (tape.nonce i r) (priorOf st streams r) words[i]!).run
      (Transcript.init #[] dom)
    streams := streams.push t.rounds
    proofs := proofs.push (FlockSoundness.Discharge.ZkView.encZk (shapeZk st sch extras.size) p)
  let record ← recordJson hs spec rootB points streams proofs
  return (record, proofs)

/-- **The masked session in three parts**: the schedules, the level-0 commitments, the link message, and the rest. -/
theorem sessionPZk_eq {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (io : Asks M)
    (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) :
    sessionPZk io hs spec tabs words trs tape = (do
      let schs ← schedsPZk tabs
      let cms ← commitsPZk hs tabs schs (tablesP tabs trs) tape
      let c1 ← io.link (cms.map fun c => flat c.cap) (tabs.map fun _ => ByteArray.empty)
      afterLinkPZk io hs spec tabs words trs tape schs (tablesP tabs trs) cms c1) := rfl

theorem schedsPZk_quiet (tabs : Array Setup) : quietS.R (schedsPZk tabs : V _) (schedsPZk tabs : PS _) := by
  unfold schedsPZk
  sim_walk

theorem commitRowsP_quiet (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (rows : Array (Array F128))
    (salt : ℕ → ByteArray) :
    quietS.R (commitRowsP (Hashes.ofSha512 h) ms lv rows salt : V _)
      (commitRowsP (Hashes.ofSha512 h) ms lv rows salt : PS _) := by
  have ho := opsWith_sim quietS (quietS_hashes h) ms
  unfold commitRowsP
  sim_walk [ho.leaf, ho.node]

theorem commitsPZk_quiet (h : ByteArray → ByteArray) (tabs : Array Setup) (schs : Array (Schedule × ℕ))
    (tbls : Array (Array F128)) (tape : TapeZk) :
    quietS.R (commitsPZk (Hashes.ofSha512 h) tabs schs tbls tape : V _)
      (commitsPZk (Hashes.ofSha512 h) tabs schs tbls tape : PS _) := by
  unfold commitsPZk commitPZk
  sim_walk [commitRowsP_quiet h]

theorem afterLinkPZk_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk) (schs : Array (Schedule × ℕ))
    (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128) :
    (simS H resp).R (afterLinkPZk asks Hashes.pi spec tabs words trs tape schs tbls cms c1)
      (afterLinkPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape schs tbls cms c1) := by
  unfold afterLinkPZk
  sim_walk [innerCommitP_sim H resp, repPZk_sim H resp, recordJson_sim H resp]

/-- **The first masked send is the link message**: a pure masked session that returns ran its schedules and level-0
commitments (their verdict-monad runs return), then sent the link message of the caps, with no public words, first. -/
theorem sessionSZk_link {H : Claims.Msg → Claims.Msg} {resp : ℕ → Claims.Msg} {spec : SessionSpec}
    {sts : Array Flock.Stmt} {w : Claims.Flock.Wit} {ω : TapeZk} {rec : String} {pfs : Array ByteArray}
    {bodies : Array Claims.Msg} (h : sessionSZk H resp spec sts w ω = (.ok (rec, pfs), bodies)) :
    ∃ schs cms, (schedsPZk (tabsOf sts) : V _) = .ok schs ∧
      (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) (tabsOf sts) schs (tablesP (tabsOf sts) (tracesOf sts w)) ω :
        V _) = .ok cms ∧
      bodies.toList.head? = some (linkBody (cms.map fun c => flat c.cap).toList
        ((tabsOf sts).map fun _ => ByteArray.empty).toList) := by
  unfold sessionSZk at h
  rw [sessionPZk_eq, go_bind, schedsPZk_quiet] at h
  cases hs : (schedsPZk (tabsOf sts) : V _) with
  | error e => rw [hs] at h; cases h
  | ok schs =>
    rw [hs] at h
    dsimp only at h
    rw [go_bind, commitsPZk_quiet] at h
    cases hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) (tabsOf sts) schs
        (tablesP (tabsOf sts) (tracesOf sts w)) ω : V _) with
    | error e => rw [hc] at h; cases h
    | ok cms =>
      rw [hc] at h
      dsimp only at h
      rw [go_bind, asksS_link_go] at h
      dsimp only at h
      rw [Array.size_empty] at h
      refine ⟨schs, cms, rfl, hc, ?_⟩
      obtain ⟨bs, -, hl⟩ := afterLinkPZk_sim H resp spec (tabsOf sts) (MaskRank.masksOf sts) (tracesOf sts w) ω
        schs (tablesP (tabsOf sts) (tracesOf sts w)) cms (coinOf (resp 0)) (#[].push (linkBody
          (cms.map fun c => flat c.cap).toList ((tabsOf sts).map fun _ => ByteArray.empty).toList))
      rw [h] at hl
      dsimp only at hl
      rw [hl]
      simp

/-! ## The caps' sizes -/

/-- **A level's cap under `hm96-sha512/v1`**, for a SHA-512 of 64-byte outputs: at most `2^d` digests of 64 bytes,
whatever the rows. -/
theorem commitRowsP_cap {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (lv : Level)
    (rows : Array (Array F128)) (salt : ℕ → ByteArray) {c : Commit}
    (hc : (commitRowsP (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 lv rows salt : V Commit) = .ok c) :
    c.cap.size ≤ 2 ^ lv.d ∧ ∀ b ∈ c.cap, b.size = 64 := by
  unfold commitRowsP at hc
  rw [hm96_ops] at hc
  obtain ⟨leaves, hl, hc⟩ := except_ok_bind hc
  obtain ⟨s, hs, hc⟩ := except_ok_bind hc
  cases hc
  have hL := mapM_sized (fun _ _ hr => hm96_leaf h64 _ _ hr) hl
  rw [Array.size_range] at hL
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hs
  refine except_forIn_inv (fun s : Array (Array ByteArray) × Array ByteArray => s.2.size ≤ 2 ^ lv.d ∧
    ∀ b ∈ s.2, b.size = 64) ?_ _ _ _ ⟨hL.1.le, hL.2⟩ hs
  rintro - ⟨layers, cur⟩ r ⟨h1, -⟩ hr
  obtain ⟨cur', hc', hr⟩ := except_ok_bind hr
  cases hr
  have hN := mapM_sized (fun _ _ hr => hm96_node h64 _ _ hr) hc'
  rw [Array.size_range] at hN
  exact ⟨hN.1.trans_le ((Nat.div_le_self _ _).trans h1), hN.2⟩

theorem getD_set_zero_d (a : Array Level) (l : Level) (hl : l.d = a[0]!.d) :
    ((a.set! 0 l).getD 0 default).d = (a.getD 0 default).d := by
  by_cases h0 : 0 < a.size
  · simp [Array.getD, Array.set!, Array.setIfInBounds, h0, hl]
  · simp [Array.getD, Array.set!, Array.setIfInBounds, h0]

/-- The `--zk` schedule keeps fast100's level-0 `d`, so it is at most `56`. -/
theorem zkSchedule_d0 {m : ℕ} {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) : (v.1.levels.getD 0 default).d ≤ 56 := by
  unfold Zk.schedule at h
  obtain ⟨q0, -, h⟩ := except_ok_bind h
  cases hf : fast100 m with
  | none => rw [hf] at h; cases h
  | some sch =>
    rw [hf] at h
    cases h
    show ((sch.levels.set! 0 _).getD 0 default).d ≤ 56
    simp only [getD_set_zero_d]
    exact fast100_d0 hf

/-- **The masked case's statement commits with `hm96-sha512/v1`**: the scope check's leaf scheme. -/
theorem CaseZk.hm96 {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : CaseZk H rel) :
    ∃ st, c.sts = #[st] ∧ st.merkle = MerkleScheme.hm96Sha512 := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  have hzk : p.zk = true := readOf_zk (x := c.x) (by
    unfold Readable.parseBytes at hp
    obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
    rw [hr, ← Readable.toRead_of_parse hrp])
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hc' : ProvedScope.check p.tags p.zk p.I.circuitFile p.I.publicFile p.I.tables p.I.count false p.registered =
      .ok () := by
    have := hc
    unfold Readable.Read.check at this
    rw [hpi] at this
    exact this
  have hl : p.tags.merkleLeaf = "hm96-sha512/v1" := by
    rcases (FlockSoundness.Discharge.FailClosed.check_ok hc').1.2.2.2.1 with hl | ⟨hz, -⟩
    · exact hl
    · rw [hzk] at hz
      cases hz
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset with ⟨-, st, hst, -, hsts⟩ | ⟨hc1, hn, -⟩
  · refine ⟨st, hsts, ?_⟩
    cases hh : p.tags.hiddenOutputs
    · exact FlockSoundness.Discharge.Exec.setupH_hm96 hl
        ((FlockSoundness.Discharge.HiddenExec.setupOne_public hh).symm.trans hst)
    · exact FlockSoundness.Discharge.HiddenExec.setupHidden_hm96 hl
        ((FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh).symm.trans hst)
  · have := c.ready_parts.1
    omega

/-! ## The goal -/

/-- **The first masked send is the link message**, each length in its word: the goal `LinkFirstZk`, for a SHA-512 of
64-byte outputs. -/
theorem linkFirstZk_goal (H : Claims.Msg → Claims.Msg) (hH : ∀ m, (H m).length = 64)
    (rel : Readable.Parsed → Readable.Relation) : LinkFirstZk H rel := by
  intro o
  have h64 : ∀ b, (Claims.hOf H b).size = 64 := fun b => by
    simp [Claims.hOf, Claims.ofMsg, ByteArray.size, hH]
  obtain ⟨schs, cms, hs, hc, hh⟩ := sessionSZk_link o.ran
  obtain ⟨st, hsts, hms⟩ := o.hm96
  have htabs : tabsOf o.sts = #[Setup.ofCircuit st] := by rw [hsts]; simp [tabsOf]
  have hsch : ∃ v, Zk.schedule st.m = .ok v ∧ schs = #[v] := by
    unfold schedsPZk at hs
    rw [htabs] at hs
    obtain ⟨h1, h2⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hs
    have h0 := h2 0 (by simp) (by rw [h1]; simp)
    simp only [List.getElem_toArray, List.getElem_cons_zero, Setup.ofCircuit, monadLift_self] at h0
    refine ⟨_, h0, ?_⟩
    apply Array.ext (by simpa using h1)
    intro j hj _
    have : j = 0 := by simp at h1; omega
    subst this
    rfl
  obtain ⟨v, hv, rfl⟩ := hsch
  unfold commitsPZk at hc
  obtain ⟨hcs, hcall⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok hc
  rw [htabs] at hcs hcall
  simp only [List.size_toArray, List.length_cons, List.length_nil, zero_add] at hcs
  refine ⟨_, _, hh, ?_, ?_, ?_, ?_⟩
  · simp [hcs]
  · simp [htabs]
  · intro b hb
    obtain ⟨c, hcm, rfl⟩ : ∃ c ∈ cms, flat c.cap = b := by simpa using hb
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hcm
    have hj0 : j = 0 := by omega
    have hcj := hcall j hj
    subst hj0
    simp only [List.getElem!_toArray, List.getElem!_cons_zero, Setup.ofCircuit, hms, commitPZk] at hcj
    obtain ⟨hcap, hcap64⟩ := commitRowsP_cap h64 _ _ _ hcj
    rw [flat_size hcap64]
    have hd := zkSchedule_d0 hv
    calc 64 * cms[0].cap.size ≤ 64 * 2 ^ 56 :=
          Nat.mul_le_mul_left _ (hcap.trans (Nat.pow_le_pow_right (by norm_num) hd))
      _ < 2 ^ 64 := by norm_num
  · simp

end FlockComplete
