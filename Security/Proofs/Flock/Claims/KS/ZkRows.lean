import Proofs.Flock.Claims.KS.ZkEvent
import Proofs.Flock.Claims.KS.ZkLevels
import Proofs.Flock.Claims.KS.ZkLevelsAll

/-!
# The masked reader's opened rows at a rep's run

`ZkEntries` and `ZkDrawJ` with each opened row whole: at level 0 its witness lanes and both reps' extra lanes, and past
level 0 every level's. A masked self-clash (`ClashW`, `ClashL`) compares whole rows, so where `OpensEntries` reads an
output's level-0 openings as the reader's entries on the witness lanes, `OpensRows` reads each level's openings as the
rep's opened rows, row by row:

* `Claims.Flock.zkRepRowsOf`, `Claims.Flock.zkRepRowsOf_ok`, `Claims.Flock.slotOpensZ_rows`: `zkRepColsOf`,
  `zkRepColsOf_ok` and `slotOpensZ_entries` with no row cut; `Claims.Flock.slotOpensZ_rowsAt` at any level.
* `FlockSoundness.Discharge.ZkExec.zk_rep_rows`, `FlockSoundness.Discharge.ZkExec.zk_table_rows₀Z`: a rep's, and an
  accepted table's, level-0 openings are its reps' `zkRepRowsOf` rows (`zk_rep_entries`, `zk_table_opens₀Z`);
  `FlockSoundness.Discharge.ZkExec.zk_rep_rowsL`, `FlockSoundness.Discharge.ZkExec.zk_table_rowsLZ`: their level-`ℓ`
  openings past level 0 are their reps' `zkLevelRowsOf` rows (`zk_rep_reqsL`).
* `FlockSoundness.Discharge.ZkExec.zk_table_capsZ`: an accepted table's reps record their proof files' recursive caps
  (`zk_rep_reqs`).
* `OpensRows`, `runsAtZCJ_live_rows`, `runsAtZCJH_live_rows`: at public and hidden outputs, each accepted output's
  openings at every level are the reps' rows, and its recorded caps their proofs' (`runsAtZCJ_live_entries`,
  `runsAtZCJH_live_entries` at `OpensRows`).
* `decodesTableR_zkTabs`, `decodesTableHR_zkTabs`, `decodesTableR_deploy`: the deployed `opnD` and `decD` at `zkTabs`
  are right at `OpensRows` at one live outcome `o₀`, on `o₀` where `verify --zk` accepts what `wrS` writes for it, at
  public and hidden outputs and at a masked statement in scope (`decodesTableW_zkTabs`, `decodesTableHW_zkTabs`,
  `decodesTableW_deploy`). The verdict reads `o₀`'s transcript, so the outcome is pinned.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

/-- **A rep's opened level-0 rows** from its recorded rounds and proof bytes: `zkRepColsOf` with each row whole, none
if the proof does not decode. -/
def zkRepRowsOf (sh : _root_.Flock.Shape) (rounds : Array _root_.Flock.Round) (bytes : ByteArray) :
    List (ℕ × List 𝔽) :=
  match _root_.Flock.Zk.decode sh bytes with
  | .ok zp =>
    zkRepCols (sh.sched.levels.getD 0 default)
      ((rounds[zkQueryRound sh.m sh.kLog sh.sched.k0 sh.claims]?.map (·.coins)).getD #[])
      zp.proof.ligerito.initialProof.rows
  | .error _ => []

/-- The rows at a proof that decodes: `zkRepCols` at level 0, the coins of round `zkQueryRound` and the proof's
level-0 rows. -/
theorem zkRepRowsOf_ok {sh : _root_.Flock.Shape} {rounds : Array _root_.Flock.Round} {bytes : ByteArray}
    {zp : _root_.Flock.Zk.ZkProof} (h : _root_.Flock.Zk.decode sh bytes = .ok zp) :
    zkRepRowsOf sh rounds bytes = zkRepCols (sh.sched.levels.getD 0 default)
      (rounds[zkQueryRound sh.m sh.kLog sh.sched.k0 sh.claims]!).coins zp.proof.ligerito.initialProof.rows := by
  unfold zkRepRowsOf
  rw [h, coins_getElem!]

/-- **A masked rep's level-0 openings are its rows**: `slotOpensZ_entries` with each row whole. -/
theorem slotOpensZ_rows {D Sl : Type} (out : OutZC 𝔽 D Sl) (r : ℕ) (lv : _root_.Flock.Level)
    (coins : Array _root_.Flock.F128) (rows : Array (Array _root_.Flock.F128)) (R : Model.Req)
    (rest : List Model.Req) (hreqs : (if r = 0 then out.pend.1 else out.pend.2).lig.reqs = R :: rest)
    (hl : R.level = 0) (hpos : R.positions = (_root_.Flock.positions lv coins).toList)
    (hq : R.positions.length = lv.q) (hrows : rows.size = R.positions.length)
    (hop : ∀ j, (out.opens.rows r 0 j).1 = rows[j]!.toList.map FlockLevel3.GF128.of) :
    (slotOpensZ out r 0 lv.q).map (fun x => (x.1, x.2.1)) = zkRepCols lv coins rows := by
  unfold slotOpensZ zkRepCols
  rw [hreqs, List.find?_cons_of_pos (by simp [hl])]
  simp only [List.map_map]
  rw [← hq, List.take_length, ← hpos]
  apply List.ext_getElem
  · simp [hrows]
  · intro i h1 h2
    simp only [List.length_map, List.length_zipIdx] at h1
    simp [hop, hrows, h1]

/-- **A masked rep's openings at a level are its rows there**: `slotOpensZ_rows` at the rep's first request of any
level `ℓ`. -/
theorem slotOpensZ_rowsAt {D Sl : Type} (out : OutZC 𝔽 D Sl) (r ℓ : ℕ) (lv : _root_.Flock.Level)
    (coins : Array _root_.Flock.F128) (rows : Array (Array _root_.Flock.F128)) (R : Model.Req)
    (hfind : ((if r = 0 then out.pend.1 else out.pend.2).lig.reqs.find? fun req => req.level == ℓ) = some R)
    (hpos : R.positions = (_root_.Flock.positions lv coins).toList)
    (hq : R.positions.length = lv.q) (hrows : rows.size = R.positions.length)
    (hop : ∀ j, (out.opens.rows r ℓ j).1 = rows[j]!.toList.map FlockLevel3.GF128.of) :
    (slotOpensZ out r ℓ lv.q).map (fun x => (x.1, x.2.1)) = zkRepCols lv coins rows := by
  unfold slotOpensZ zkRepCols
  rw [hfind]
  simp only [List.map_map]
  rw [← hq, List.take_length, ← hpos]
  apply List.ext_getElem
  · simp [hrows]
  · intro i h1 h2
    simp only [List.length_map, List.length_zipIdx] at h1
    simp [hop, hrows, h1]

end Claims.Flock

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model

local notation "𝔽" => FlockLevel3.GF128

/-- **A masked rep's level-0 openings are its deployed rows**: `zk_rep_entries` with each row whole. -/
theorem zk_rep_rows {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (extras : Array Flock.Claim) (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (bytes : ByteArray) (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam T st sch tPad)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) (hdec : Zk.decode (shapeZK st sch extras) bytes = .ok zp)
    (out : ZkLink.OutZC 𝔽 Dig Sl)
    (hp : Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
      (rounds.toList.map (·.coins)) (if r.val = 0 then out.pend.1 else out.pend.2))
    (ho : ∀ j, (out.opens.rows r.val 0 j).1 = zp.proof.ligerito.initialProof.rows[j]!.toList.map fOf) :
    (ZkLink.slotOpensZ out r.val 0 (lvOf T.sch 0).queries).map (fun x => (x.1, x.2.1)) =
      Claims.Flock.zkRepRowsOf (shapeZK st sch extras) rounds bytes := by
  obtain ⟨-, -, -, -, -, hdim, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨-, h2, -, -⟩ := hdim
  obtain ⟨pend', hp', -, hreqs, R, rest, hR, hl, hpos⟩ :=
    zk_rep_reqs Hs T pts r st sch tPad extras zp rounds dom tr hsch hseam hlen hext hwf hrun hk
  have he := plays_det hp hp'
  have hseam0 : LevelSeam (sch.levels.getD 0 default) := hlv 0 (by omega)
  have hQ : (lvOf T.sch 0).queries = (sch.levels.getD 0 default).q := by
    rw [hseam.hsch, lvOf_schedOf (by omega)]
    rfl
  have hmc := ((hreqs R (by rw [hR]; exact List.mem_cons_self)).1 hl).2
  have hrows : zp.proof.ligerito.initialProof.rows.size = R.positions.length := by
    simpa using (merkleCheck_rows hmc).1
  rw [hQ, Claims.Flock.zkRepRowsOf_ok hdec]
  exact Claims.Flock.slotOpensZ_rows out r.val _ _ _ R rest (by rw [he]; exact hR) hl hpos
    (by rw [hpos, Array.length_toList, positions_size _ hseam0]) hrows ho

/-- **A masked rep's openings past level 0 are its deployed rows**: at a level `1 ≤ ℓ` of the schedule, rep `r`'s
first level-`ℓ` request is the `ℓ`-th after level 0's, at the positions of round `zkLevelRound`'s coins (`zk_rep_reqsL`,
`find?_later`), so its openings there are the rep's `zkLevelRowsOf` rows; past the last level it has none. -/
theorem zk_rep_rowsL {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (extras : Array Flock.Claim) (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (bytes : ByteArray) (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam T st sch tPad)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) (hdec : Zk.decode (shapeZK st sch extras) bytes = .ok zp)
    (out : ZkLink.OutZC 𝔽 Dig Sl)
    (hp : Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
      (rounds.toList.map (·.coins)) (if r.val = 0 then out.pend.1 else out.pend.2))
    (ho : ∀ ℓ j, (out.opens.rows r.val ℓ j).1 =
      ((openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    (ℓ : ℕ) (h1 : 1 ≤ ℓ) :
    (ZkLink.slotOpensZ out r.val ℓ (lvOf T.sch ℓ).queries).map (fun x => (x.1, x.2.1)) =
      FlockDeploy.KS.zkLevelRowsOf (shapeZK st sch extras) rounds bytes ℓ := by
  obtain ⟨-, -, -, -, -, -, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨pend', hp', -, hreqs, R, rest, hR, hl, -, hlenR, hlatR⟩ :=
    zk_rep_reqsL Hs T pts r st sch tPad extras zp rounds dom tr hsch hseam hlen hext hwf hrun hk
  have he := plays_det hp hp'
  rw [← he] at hR hreqs
  by_cases hℓ : ℓ < sch.levels.size
  · have hj : ℓ - 1 < rest.length := by omega
    have hfind := find?_later hl hlatR h1 (show ℓ ≤ rest.length by omega)
    have hat := hlatR (ℓ - 1) hj
    have hmem : rest[ℓ - 1] ∈ (if r.val = 0 then out.pend.1 else out.pend.2).lig.reqs := by
      rw [hR]
      exact List.mem_cons_of_mem _ (List.getElem_mem hj)
    generalize rest[ℓ - 1] = Q at hfind hat hmem
    rw [Nat.sub_add_cancel h1] at hat
    obtain ⟨hlQ, hposQ⟩ := hat
    obtain ⟨-, -, -, hmc⟩ := (hreqs Q hmem).2 (by omega)
    rw [hlQ] at hmc
    have hseamℓ : LevelSeam (sch.levels.getD ℓ default) := hlv ℓ hℓ
    have hQ : (lvOf T.sch ℓ).queries = (sch.levels.getD ℓ default).q := by
      rw [hseam.hsch, lvOf_schedOf hℓ]
      rfl
    have hrows : (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1.size = Q.positions.length := by
      simpa using (merkleCheck_rows hmc).1
    have hc : 1 ≤ ℓ ∧ ℓ < (shapeZK st sch extras).sched.levels.size := ⟨h1, hℓ⟩
    rw [hQ, FlockDeploy.KS.zkLevelRowsOf, ite_eq_left hc, hdec]
    simp only [shapeZK]
    refine (Claims.Flock.slotOpensZ_rowsAt out r.val ℓ _
      (rounds[FlockDeploy.KS.zkLevelRound st.m st.kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins
      _ Q (by rw [hR]; exact hfind) hposQ (by rw [hposQ, Array.length_toList, positions_size _ hseamℓ]) hrows
      (ho ℓ)).trans ?_
    rw [Claims.Flock.coins_getElem!]
  · have hc : ¬ (1 ≤ ℓ ∧ ℓ < (shapeZK st sch extras).sched.levels.size) := fun h => hℓ h.2
    rw [FlockDeploy.KS.zkLevelRowsOf, ite_eq_right hc]
    have hnone : ((if r.val = 0 then out.pend.1 else out.pend.2).lig.reqs.find?
        fun req => req.level == ℓ) = none := by
      rw [hR, List.find?_eq_none]
      intro q hq
      simp only [beq_iff_eq]
      rcases List.mem_cons.1 hq with rfl | hq
      · omega
      · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hq
        rw [(hlatR i hi).1]
        omega
    unfold ZkLink.slotOpensZ
    rw [hnone]
    rfl

/-- **At `verify --zk`'s acceptance of a table, its level-0 openings are its reps' rows**: `zk_table_opens₀Z`'s
entries with each row whole (`zk_rep_rows`), rep 0's then rep 1's. -/
theorem zk_table_rows₀Z {n mPts : ℕ} {Reg : Type} (T : Finset (Fin n) → Reg → ZkSession.TabZK 𝔽 mPts)
    (S : Finset (Fin n)) (R : Reg) (pts : Fin 2 → Fin mPts → 𝔽) (spec : SessionSpec) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (sess : Session) (i : ℕ) (b₀ b₁ c : ByteArray) (N : ℕ)
    (hN : (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length = N)
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam (T S R) st sch tPad)
    (hext : ExtraRealizes st (T S R).S sess.points (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts))
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (h0 : Zk.verifyRep spec st sch tPad sess i 0 b₀ = .ok c) (h1 : Zk.verifyRep spec st sch tPad sess i 1 b₁ = .ok c)
    (s₀ s₁ : StreamRec) (zp₀ zp₁ : Zk.ZkProof)
    (hs₀ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) =
      some s₀)
    (hs₁ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) =
      some s₁)
    (hd₀ : Zk.decode (shapeZKn st sch N) b₀ = .ok zp₀) (hd₁ : Zk.decode (shapeZKn st sch N) b₁ = .ok zp₁)
    (pend : ZkLink.PendZ 𝔽 Dig (List UInt8) × ZkLink.PendZ 𝔽 Dig (List UInt8)) (cap : Cap Dig)
    (hp₀ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 0) (repMsgsZ st sch (2 + N) tPad zp₀)
      (s₀.rounds.toList.map (·.coins)) pend.1)
    (hp₁ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 1) (repMsgsZ st sch (2 + N) tPad zp₁)
      (s₁.rounds.toList.map (·.coins)) pend.2) :
    (ZkLink.opens₀Z (T S R).sch (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁]))).map
        (fun x => (x.1, x.2.1)) =
      Claims.Flock.zkRepRowsOf (shapeZKn st sch N) s₀.rounds b₀ ++
        Claims.Flock.zkRepRowsOf (shapeZKn st sch N) s₁.rounds b₁ := by
  obtain ⟨s₀', ex, zp₀', tr₀, hs₀', hx₀, hd₀', -, -, hrun₀, hk₀, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 0 b₀ c h0
  obtain ⟨s₁', ex₁, zp₁', tr₁, hs₁', hx₁, hd₁', -, -, hrun₁, hk₁, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 1 b₁ c h1
  obtain rfl : ex = ex₁ := Except.ok.inj (hx₀.symm.trans hx₁)
  obtain ⟨hsz, hrel⟩ := hext ex hx₀
  have hsh : shapeZK st sch ex = shapeZKn st sch N := by
    unfold shapeZK shapeZKn; rw [hsz, hN]
  obtain rfl : s₀' = s₀ := Option.some.inj (hs₀'.symm.trans hs₀)
  obtain rfl : s₁' = s₁ := Option.some.inj (hs₁'.symm.trans hs₁)
  have hz₀ := hd₀'
  have hz₁ := hd₁'
  rw [hsh] at hz₀ hz₁
  obtain rfl : zp₀' = zp₀ := Except.ok.inj (hz₀.symm.trans hd₀)
  obtain rfl : zp₁' = zp₁ := Except.ok.inj (hz₁.symm.trans hd₁)
  have hn : ex.size = N := hsz.trans hN
  rw [← hn] at hp₀ hp₁
  obtain ⟨hro, -, -, -⟩ := opnZ_pair hd₀ hd₁
  have he₀ := zk_rep_rows hs512 (T S R) pts 0 st sch tPad ex zp₀' s₀'.rounds _ tr₀ b₀ hsch hseam hsz
    hrel (hwf s₀' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₀))) hrun₀ hk₀ hd₀'
    (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])) hp₀
    (fun j => by show ((opnZ st sch N #[b₀, b₁]).rows 0 0 j).1 = _; rw [hro]; simp [opensOf, openRow, openAt])
  have he₁ := zk_rep_rows hs512 (T S R) pts 1 st sch tPad ex zp₁' s₁'.rounds _ tr₁ b₁ hsch hseam hsz
    hrel (hwf s₁' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₁))) hrun₁ hk₁ hd₁'
    (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])) hp₁
    (fun j => by show ((opnZ st sch N #[b₀, b₁]).rows 1 0 j).1 = _; rw [hro]; simp [opensOf, openRow, openAt])
  rw [hsh] at he₀ he₁
  unfold ZkLink.opens₀Z
  rw [List.map_append]
  exact congrArg₂ (· ++ ·) he₀ he₁

/-- **At `verify --zk`'s acceptance of a table, its openings past level 0 are its reps' rows**: at every level `ℓ ≥ 1`,
rep 0's and rep 1's openings are their `zkLevelRowsOf` rows (`zk_rep_rowsL`). -/
theorem zk_table_rowsLZ {n mPts : ℕ} {Reg : Type} (T : Finset (Fin n) → Reg → ZkSession.TabZK 𝔽 mPts)
    (S : Finset (Fin n)) (R : Reg) (pts : Fin 2 → Fin mPts → 𝔽) (spec : SessionSpec) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (sess : Session) (i : ℕ) (b₀ b₁ c : ByteArray) (N : ℕ)
    (hN : (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length = N)
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam (T S R) st sch tPad)
    (hext : ExtraRealizes st (T S R).S sess.points (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts))
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (h0 : Zk.verifyRep spec st sch tPad sess i 0 b₀ = .ok c) (h1 : Zk.verifyRep spec st sch tPad sess i 1 b₁ = .ok c)
    (s₀ s₁ : StreamRec) (zp₀ zp₁ : Zk.ZkProof)
    (hs₀ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) =
      some s₀)
    (hs₁ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) =
      some s₁)
    (hd₀ : Zk.decode (shapeZKn st sch N) b₀ = .ok zp₀) (hd₁ : Zk.decode (shapeZKn st sch N) b₁ = .ok zp₁)
    (pend : ZkLink.PendZ 𝔽 Dig (List UInt8) × ZkLink.PendZ 𝔽 Dig (List UInt8)) (cap : Cap Dig)
    (hp₀ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 0) (repMsgsZ st sch (2 + N) tPad zp₀)
      (s₀.rounds.toList.map (·.coins)) pend.1)
    (hp₁ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 1) (repMsgsZ st sch (2 + N) tPad zp₁)
      (s₁.rounds.toList.map (·.coins)) pend.2) (ℓ : ℕ) (hℓ1 : 1 ≤ ℓ) :
    (ZkLink.slotOpensZ (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])) 0 ℓ
        (lvOf (T S R).sch ℓ).queries).map (fun x => (x.1, x.2.1)) =
      FlockDeploy.KS.zkLevelRowsOf (shapeZKn st sch N) s₀.rounds b₀ ℓ ∧
    (ZkLink.slotOpensZ (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])) 1 ℓ
        (lvOf (T S R).sch ℓ).queries).map (fun x => (x.1, x.2.1)) =
      FlockDeploy.KS.zkLevelRowsOf (shapeZKn st sch N) s₁.rounds b₁ ℓ := by
  obtain ⟨s₀', ex, zp₀', tr₀, hs₀', hx₀, hd₀', -, -, hrun₀, hk₀, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 0 b₀ c h0
  obtain ⟨s₁', ex₁, zp₁', tr₁, hs₁', hx₁, hd₁', -, -, hrun₁, hk₁, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 1 b₁ c h1
  obtain rfl : ex = ex₁ := Except.ok.inj (hx₀.symm.trans hx₁)
  obtain ⟨hsz, hrel⟩ := hext ex hx₀
  have hsh : shapeZK st sch ex = shapeZKn st sch N := by
    unfold shapeZK shapeZKn; rw [hsz, hN]
  obtain rfl : s₀' = s₀ := Option.some.inj (hs₀'.symm.trans hs₀)
  obtain rfl : s₁' = s₁ := Option.some.inj (hs₁'.symm.trans hs₁)
  have hz₀ := hd₀'
  have hz₁ := hd₁'
  rw [hsh] at hz₀ hz₁
  obtain rfl : zp₀' = zp₀ := Except.ok.inj (hz₀.symm.trans hd₀)
  obtain rfl : zp₁' = zp₁ := Except.ok.inj (hz₁.symm.trans hd₁)
  have hn : ex.size = N := hsz.trans hN
  rw [← hn] at hp₀ hp₁
  obtain ⟨hro, -, -, -⟩ := opnZ_pair hd₀ hd₁
  have he₀ := zk_rep_rowsL hs512 (T S R) pts 0 st sch tPad ex zp₀' s₀'.rounds _ tr₀ b₀ hsch hseam hsz
    hrel (hwf s₀' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₀))) hrun₀ hk₀ hd₀'
    (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])) hp₀
    (fun ℓ j => by show ((opnZ st sch N #[b₀, b₁]).rows 0 ℓ j).1 = _; rw [hro]; simp [opensOf, openRow]) ℓ hℓ1
  have he₁ := zk_rep_rowsL hs512 (T S R) pts 1 st sch tPad ex zp₁' s₁'.rounds _ tr₁ b₁ hsch hseam hsz
    hrel (hwf s₁' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₁))) hrun₁ hk₁ hd₁'
    (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])) hp₁
    (fun ℓ j => by show ((opnZ st sch N #[b₀, b₁]).rows 1 ℓ j).1 = _; rw [hro]; simp [opensOf, openRow]) ℓ hℓ1
  rw [hsh] at he₀ he₁
  exact ⟨he₀, he₁⟩

/-- **A rep's pads at an output**: its inner openings are those of the inner proof its proof file decodes to, and it
recorded at inner query `jj` the position `Zk.verifyInner` reads there: its stream's last round's coin `jj`, `lo` mod
`innerN`. -/
def PadsRep (st : Setup) (sch : Flock.Schedule) (n : ℕ) (s : StreamRec) (bytes : ByteArray)
    (io : ZkLink.InnerOpens 𝔽 Dig (List UInt8)) (qpos : ℕ → ℕ) : Prop :=
  ∃ zp : Zk.ZkProof, Zk.decode (shapeZKn st sch n) bytes = .ok zp ∧ io = innerOpensOf zp.inner ∧
    ∀ jj < Zk.Q, qpos jj = ((s.rounds[s.rounds.size - 1]!).coins[jj]!).lo.toNat % innerN (layZ st sch)

/-- **At `verify --zk`'s acceptance of a table, its reps' pads are their proof files' at their streams' query words**:
each rep's inner openings are its decoded inner proof's (`opnZ_pair`), and the positions it recorded are its last
round's coins, through the table's `pos` (`zk_qpos_of_rep`, `ZkTabLay.hpos`). -/
theorem zk_table_padsZ {n mPts : ℕ} {Reg : Type} (T : Finset (Fin n) → Reg → ZkSession.TabZK 𝔽 mPts)
    (S : Finset (Fin n)) (R : Reg) (pts : Fin 2 → Fin mPts → 𝔽) (spec : SessionSpec) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (sess : Session) (i : ℕ) (b₀ b₁ c : ByteArray) (N : ℕ)
    (hN : (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length = N)
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam (T S R) st sch tPad)
    (htab : ZkTabLay (T S R) st sch) (hfold : FoldRealizes st.fold (T S R).S)
    (hext : ExtraRealizes st (T S R).S sess.points (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts))
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (h0 : Zk.verifyRep spec st sch tPad sess i 0 b₀ = .ok c) (h1 : Zk.verifyRep spec st sch tPad sess i 1 b₁ = .ok c)
    (s₀ s₁ : StreamRec) (zp₀ zp₁ : Zk.ZkProof)
    (hs₀ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) =
      some s₀)
    (hs₁ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) =
      some s₁)
    (hd₀ : Zk.decode (shapeZKn st sch N) b₀ = .ok zp₀) (hd₁ : Zk.decode (shapeZKn st sch N) b₁ = .ok zp₁)
    (pend : ZkLink.PendZ 𝔽 Dig (List UInt8) × ZkLink.PendZ 𝔽 Dig (List UInt8)) (cap : Cap Dig)
    (hp₀ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 0) (repMsgsZ st sch (2 + N) tPad zp₀)
      (s₀.rounds.toList.map (·.coins)) pend.1)
    (hp₁ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 1) (repMsgsZ st sch (2 + N) tPad zp₁)
      (s₁.rounds.toList.map (·.coins)) pend.2) :
    PadsRep st sch N s₀ b₀ ((ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])).opens.inner 0)
        (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])).pend.1.qpos ∧
      PadsRep st sch N s₁ b₁ ((ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])).opens.inner 1)
        (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])).pend.2.qpos := by
  obtain ⟨s₀', ex, zp₀', tr₀, hs₀', hx₀, hd₀', -, -, hrun₀, hk₀, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 0 b₀ c h0
  obtain ⟨s₁', ex₁, zp₁', tr₁, hs₁', hx₁, hd₁', -, -, hrun₁, hk₁, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 1 b₁ c h1
  obtain rfl : ex = ex₁ := Except.ok.inj (hx₀.symm.trans hx₁)
  obtain ⟨hsz, hrel⟩ := hext ex hx₀
  have hsh : shapeZK st sch ex = shapeZKn st sch N := by
    unfold shapeZK shapeZKn; rw [hsz, hN]
  obtain rfl : s₀' = s₀ := Option.some.inj (hs₀'.symm.trans hs₀)
  obtain rfl : s₁' = s₁ := Option.some.inj (hs₁'.symm.trans hs₁)
  have hz₀ := hd₀'
  have hz₁ := hd₁'
  rw [hsh] at hz₀ hz₁
  obtain rfl : zp₀' = zp₀ := Except.ok.inj (hz₀.symm.trans hd₀)
  obtain rfl : zp₁' = zp₁ := Except.ok.inj (hz₁.symm.trans hd₁)
  have hn : ex.size = N := hsz.trans hN
  rw [← hn] at hp₀ hp₁
  obtain ⟨-, -, hi₀, hi₁⟩ := opnZ_pair hd₀ hd₁
  have hq₀ := zk_qpos_of_rep (T S R) pts 0 st sch tPad ex b₀ zp₀' s₀'.rounds _ tr₀ hsch hseam htab hfold hsz hrel
    hd₀' (hwf s₀' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₀))) hrun₀ hk₀ pend.1 hp₀
  have hq₁ := zk_qpos_of_rep (T S R) pts 1 st sch tPad ex b₁ zp₁' s₁'.rounds _ tr₁ hsch hseam htab hfold hsz hrel
    hd₁' (hwf s₁' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₁))) hrun₁ hk₁ pend.2 hp₁
  refine ⟨⟨zp₀', hd₀, hi₀, fun jj hjj => ?_⟩, ⟨zp₁', hd₁, hi₁, fun jj hjj => ?_⟩⟩
  · show pend.1.qpos jj = _
    rw [hq₀ jj hjj, htab.hpos, htab.hN]
    rfl
  · show pend.2.qpos jj = _
    rw [hq₁ jj hjj, htab.hpos, htab.hN]
    rfl

/-- **At `verify --zk`'s acceptance of a table, its reps record their proof files' recursive caps**: each rep's
pending's caps are its decoded proof's recursive caps, one per level past level 0, through `capDig`
(`zk_rep_reqs`). -/
theorem zk_table_capsZ {n mPts : ℕ} {Reg : Type} (T : Finset (Fin n) → Reg → ZkSession.TabZK 𝔽 mPts)
    (S : Finset (Fin n)) (R : Reg) (pts : Fin 2 → Fin mPts → 𝔽) (spec : SessionSpec) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (sess : Session) (i : ℕ) (b₀ b₁ c : ByteArray) (N : ℕ)
    (hN : (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length = N)
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam (T S R) st sch tPad)
    (hext : ExtraRealizes st (T S R).S sess.points (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts))
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (h0 : Zk.verifyRep spec st sch tPad sess i 0 b₀ = .ok c) (h1 : Zk.verifyRep spec st sch tPad sess i 1 b₁ = .ok c)
    (s₀ s₁ : StreamRec) (zp₀ zp₁ : Zk.ZkProof)
    (hs₀ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) =
      some s₀)
    (hs₁ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) =
      some s₁)
    (hd₀ : Zk.decode (shapeZKn st sch N) b₀ = .ok zp₀) (hd₁ : Zk.decode (shapeZKn st sch N) b₁ = .ok zp₁)
    (pend : ZkLink.PendZ 𝔽 Dig (List UInt8) × ZkLink.PendZ 𝔽 Dig (List UInt8))
    (hp₀ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 0) (repMsgsZ st sch (2 + N) tPad zp₀)
      (s₀.rounds.toList.map (·.coins)) pend.1)
    (hp₁ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 1) (repMsgsZ st sch (2 + N) tPad zp₁)
      (s₁.rounds.toList.map (·.coins)) pend.2) :
    pend.1.lig.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig zp₀.proof.ligerito.recursiveCaps[j]!) ∧
      pend.2.lig.caps =
        (List.range (sch.levels.size - 1)).map (fun j => capDig zp₁.proof.ligerito.recursiveCaps[j]!) := by
  obtain ⟨s₀', ex, zp₀', tr₀, hs₀', hx₀, hd₀', -, -, hrun₀, hk₀, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 0 b₀ c h0
  obtain ⟨s₁', ex₁, zp₁', tr₁, hs₁', hx₁, hd₁', -, -, hrun₁, hk₁, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 1 b₁ c h1
  obtain rfl : ex = ex₁ := Except.ok.inj (hx₀.symm.trans hx₁)
  obtain ⟨hsz, hrel⟩ := hext ex hx₀
  have hsh : shapeZK st sch ex = shapeZKn st sch N := by
    unfold shapeZK shapeZKn; rw [hsz, hN]
  obtain rfl : s₀' = s₀ := Option.some.inj (hs₀'.symm.trans hs₀)
  obtain rfl : s₁' = s₁ := Option.some.inj (hs₁'.symm.trans hs₁)
  rw [hsh] at hd₀' hd₁'
  obtain rfl : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hd₀)
  obtain rfl : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hd₁)
  have hn : ex.size = N := hsz.trans hN
  rw [← hn] at hp₀ hp₁
  obtain ⟨q₀, hq₀, hc₀, -⟩ := zk_rep_reqs hs512 (T S R) pts 0 st sch tPad ex zp₀' s₀'.rounds _ tr₀ hsch hseam hsz
    hrel (hwf s₀' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₀))) hrun₀ hk₀
  obtain ⟨q₁, hq₁, hc₁, -⟩ := zk_rep_reqs hs512 (T S R) pts 1 st sch tPad ex zp₁' s₁'.rounds _ tr₁ hsch hseam hsz
    hrel (hwf s₁' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₁))) hrun₁ hk₁
  rw [plays_det hp₀ hq₀, plays_det hp₁ hq₁]
  exact ⟨hc₀, hc₁⟩

end FlockSoundness.Discharge.ZkExec

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable repTapeJ)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **An output's openings are its reps' rows at table `j`**: `OpensEntries` with each row whole, at the setup at
the draw `d`, table `j`'s `Zk.schedule` schedule and the session's two rep streams of table `j`, at `2·|regions|` extra
claims on the streams' rounds and table `j`'s proof files; each rep's pads are its proof file's, recorded at its
stream's query words (`PadsRep`); and each rep's recorded caps are its decoded proof file's recursive caps. -/
def OpensRows (I : Inputs) (d : Lean.Json) (t : Exec.Transcript) (j : ℕ) (T : TabZK 𝔽 mPts)
    (out : OutZC 𝔽 Dig (List UInt8)) : Prop :=
  ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session) (sch : Flock.Schedule) (tPad : ℕ)
    (s₀ s₁ : StreamRec) (h : j < sts.size),
    FlockDeploy.KS.setupAt (zkInputs I) (some d) = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
    Zk.schedule sts[j].m = .ok (sch, tPad) ∧
    sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
    sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
    (opens₀Z T.sch out).map (fun x => (x.1, x.2.1)) =
      Claims.Flock.zkRepRowsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₀.rounds
          t.proofs[2 * j]! ++
        Claims.Flock.zkRepRowsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₁.rounds
          t.proofs[2 * j + 1]! ∧
    (∀ ℓ, 1 ≤ ℓ →
      (slotOpensZ out 0 ℓ (lvOf T.sch ℓ).queries).map (fun x => (x.1, x.2.1)) =
          FlockDeploy.KS.zkLevelRowsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₀.rounds
            t.proofs[2 * j]! ℓ ∧
        (slotOpensZ out 1 ℓ (lvOf T.sch ℓ).queries).map (fun x => (x.1, x.2.1)) =
          FlockDeploy.KS.zkLevelRowsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₁.rounds
            t.proofs[2 * j + 1]! ℓ) ∧
    T.q = Zk.Q ∧
    PadsRep (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) s₀ t.proofs[2 * j]! (out.opens.inner 0)
      out.pend.1.qpos ∧
    PadsRep (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) s₁ t.proofs[2 * j + 1]! (out.opens.inner 1)
      out.pend.2.qpos ∧
    ∀ r < 2, ∃ zp : Zk.ZkProof,
      Zk.decode (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) t.proofs[2 * j + r]! = .ok zp ∧
      (if r = 0 then out.pend.1 else out.pend.2).lig.caps =
        (List.range (sch.levels.size - 1)).map (fun i => capDig zp.proof.ligerito.recursiveCaps[i]!)

/-- **`runsZCJ_live` at the setup at each draw, with the rows**: `runsAtZCJ_live_entries` at `OpensRows`
(`zk_table_rows₀Z`). -/
theorem runsAtZCJ_live_rows (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts (L.draw ω)) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZJ dj y₀) (opnAtZJ (planZJ dj y₀) I dj) (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o))
      (fun S R o j out => out.accepted = true ∧
        OpensRows I (dj S) (wr S R o) j.val ((planZJ dj y₀ S R).get j) out) := by
  intro ω R o hacc j
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  have hpts2 := setupTables_points hh' hs
  have hsr : spec.retainRounds = true := ((setupTables_retain hh' hs).trans (zkTags_retainRounds I)).trans hrr
  obtain ⟨hsz, hstJ⟩ := FlockDeploy.KS.setupAt_tableJ hh hs
  have hjn : j.val < nTab I := planZJ_length dj y₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, c, hz0, hv0, hv1⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨-, -, sch₀, hf⟩ := schedule_spec hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ _ht j.val
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetup_of_setupH_g (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hst hf hD
    ((hDraw ω ⟨j.val, hjn⟩).1 D hD) ((hDraw ω ⟨j.val, hjn⟩).2 _ hst)
  obtain ⟨y, hyx, hysch, hytp⟩ := drawSetupZJ_of ht' x (sch := schs[j.val].1) (tPad := schs[j.val].2)
    (by rw [hx]; exact hz')
  have hyst : y.x.st = sts[j.val] := by rw [hyx]; exact hx
  have hTy : (planZJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = y.table := by
    have hget : (planZJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = tableZJ dj y₀ (L.draw ω) ⟨j.val, hjn⟩ := by
      exact List.get_ofFn _ j
    rw [hget]
    unfold tableZJ
    rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩)]
    exact DrawSetupZJ.table_eq _ _
  rw [← hysch, ← hytp, ← hyst] at hv0 hv1
  rw [← hyst] at hst
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hs hdecS
  obtain ⟨hroot, hstreams⟩ := hrj j
  obtain ⟨hp2, hmp⟩ := hpts2
  have hpd := decode_points hdecS
  have hmP : sts[j.val].mPts = mPts := by rw [← hyst]; exact y.x.hmPts
  have hsz' : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp j.val hj).symm.trans hmP)
  have hwf := setupH_wf y.x.h
  have hExt : ExtraRealizes (Setup.ofCircuit y.x.st) y.table.S sess.points
      (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts) := by
    have := ofCircuit_extra y.x.st hwf.1 hwf.2 sess.points (hpd.1.trans hp2) (by rw [y.x.hmPts]; exact hsz')
    rw [y.x.hmPts, hpts] at this
    exact this
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupH_tagsAt_hm96 ((zkTags_merkleLeaf I).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam := y.table_seam
  have hlay := y.table_lay
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold y.table.S :=
    ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hwfA := decode_roundsWF hdecS
  have hretA := decode_allRetained hsr hdecS
  obtain ⟨s₀, ex₀, zp₀, tr₀, hf₀, hex₀, hdz₀, -, hrt₀, hrun₀, hk₀, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hf₁, hex₁, hdz₁, -, hrt₁, hrun₁, hk₁, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv1
  have e₁ : ex₁ = ex₀ := Except.ok.inj (hex₁.symm.trans hex₀)
  subst ex₁
  obtain ⟨hlen₀, hcr₀⟩ := hExt ex₀ hex₀
  have hm₀ : s₀ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₀)
  have hm₁ : s₁ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₁)
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀
    s₀.rounds _ tr₀ y.hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀ (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁
    s₁.rounds _ tr₁ y.hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁ (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess j.val _ _ c y.hz hms hseam hlay hfold hExt hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZJ y.table = ex₀.size := by
    unfold nXZJ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZJ (planZJ (Reg := Reg) dj y₀) (L.draw ω) R j =
      decZ (wRestZ (Setup.ofCircuit y.x.st).m (Setup.ofCircuit y.x.st).kLog (2 + ex₀.size) (schedOf y.sch) y.tPad
        (layZ (Setup.ofCircuit y.x.st) y.sch).len) := by
    unfold decAtZJ
    rw [hTy, hseam.hm, hseam.hk, hseam.hsch, hseam.ht, hseam.hnp, hnX]
  obtain ⟨e₀, he₀, hr₀⟩ := hstreams 0
  obtain ⟨e₁, he₁, hr₁⟩ := hstreams 1
  have e₆ : e₀ = s₀ := Option.some.inj (he₀.symm.trans hf₀)
  subst e₀
  have e₇ : e₁ = s₁ := Option.some.inj (he₁.symm.trans hf₁)
  subst e₁
  have hopn : opnAtZJ (planZJ (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs j =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnAtZJ
    rw [hst]
    dsimp only
    rw [y.hz]
    dsimp only
    rw [hTy, hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · rw [hTy, hdecEq, ← hr₀]
    exact hrr₀
  · rw [hTy, hdecEq, ← hr₁]
    exact hrr₁
  · dsimp only
    rw [hTy, hopn, ← hroot]
    have hok := outZC_of_accZ hacc'
    have hN : (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length = 2 * sts[j.val].regions.size := by
      rw [DecodeZL.extraClaims_length, ← hyst]
      show 2 * (regionsOf y.x.st).length = _
      simp [regionsOf]
    have hpads := fun cp : Cap Dig => zk_table_padsZ (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
      (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ c _ rfl y.hz hseam hlay hfold hExt hwfA hv0 hv1 s₀ s₁
      zp₀ zp₁ hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) cp hpl₀ hpl₁
    rw [hN, hyst] at hpads
    refine ⟨hok, spec, sts, sess, y.sch, y.tPad, s₀, s₁, hj, hs, hdecS, by rw [← hyst]; exact y.hz, hf₀, hf₁,
      (zk_table_rows₀Z (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
        (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ c _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
        hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) _ hpl₀ hpl₁).trans ?_, fun ℓ h1 =>
      ⟨(zk_table_rowsLZ (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
        (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ c _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
        hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) _ hpl₀ hpl₁ ℓ h1).1.trans ?_,
      (zk_table_rowsLZ (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
        (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ c _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
        hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) _ hpl₀ hpl₁ ℓ h1).2.trans ?_⟩, hseam.hq, ?_, ?_, ?_⟩
    rotate_right
    · have hcaps := zk_table_capsZ (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
        (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ c _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
        hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) hpl₀ hpl₁
      rw [hN, hyst] at hdz₀ hdz₁
      intro r hr
      rcases r with _ | _ | r
      · exact ⟨zp₀, hdz₀, hcaps.1⟩
      · exact ⟨zp₁, hdz₁, hcaps.2⟩
      · omega
    all_goals rw [hN, hyst]
    exacts [(hpads _).1, (hpads _).2]

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`runsZCJH_liveI` at the setup at each draw, with the rows**: `runsAtZCJH_live_entries` at `OpensRows`
(`zk_table_rows₀Z`). -/
theorem runsAtZCJH_live_rows (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZHJ dj y₀) (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o))
      (fun S R o j out => out.accepted = true ∧
        OpensRows I (dj S) (wr S R o) j.val ((planZHJ dj y₀ S R).get j) out) := by
  intro ω R o hacc j
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, hshape, hv⟩ := hacc
  have hpts2 := setupTables_pointsAny hs
  have hsr : spec.retainRounds = true := ((setupTables_retainAny hs).trans (zkTags_retainRounds I)).trans hrr
  obtain ⟨hsz, hstJ⟩ := FlockDeploy.KS.setupAt_tableJH hh hs
  have hjn : j.val < nTab I := planZHJ_length dj y₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, cap, hz0, hv0, hv1⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨sch₀, hf⟩ := fast100_of_schedule hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ ht j.val
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := HiddenExec.drawSetupH_of_setupHiddenU (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hst hf hD
    ((hDraw ω ⟨j.val, hjn⟩).1 D hD) ((hDraw ω ⟨j.val, hjn⟩).2 _ hst)
  obtain ⟨y, hyx, hysch, hytp⟩ := drawSetupZHJ_of x (by rw [hx]; exact array_forM_ok hshape j.val hj)
    (sch := schs[j.val].1) (tPad := schs[j.val].2) (by rw [hx]; exact hz')
  have hyst : y.x.st = sts[j.val] := by rw [hyx]; exact hx
  have hTy : (planZHJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = y.table := by
    have hget : (planZHJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = tableZHJ dj y₀ (L.draw ω) ⟨j.val, hjn⟩ := by
      exact List.get_ofFn _ j
    rw [hget]
    unfold tableZHJ
    rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩)]
    exact DrawSetupZHJ.table_eq _ _
  rw [← hysch, ← hytp, ← hyst] at hv0 hv1
  rw [← hyst] at hst
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hs hdecS
  obtain ⟨hroot, hstreams⟩ := hrj j
  obtain ⟨hp2, hmp⟩ := hpts2
  have hpd := decode_points hdecS
  have hmP : sts[j.val].mPts = mPts := by rw [← hyst]; exact y.x.hmPts
  have hsz' : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp j.val hj).symm.trans hmP)
  have hExt : ExtraRealizes (Setup.ofCircuit y.x.st) y.table.S sess.points
      (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts) := by
    have := ofCircuit_extra y.x.st y.x.wf.1 y.x.wf.2 sess.points (hpd.1.trans hp2) (by rw [y.x.hmPts]; exact hsz')
    rw [y.x.hmPts, hpts] at this
    exact this
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupHidden_hm96 (((tagsAt_merkleLeaf _ _ _).trans (zkTags_merkleLeaf I)).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam := y.table_seam
  have hlay := y.table_lay
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold y.table.S :=
    ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hwfA := decode_roundsWF hdecS
  have hretA := decode_allRetained hsr hdecS
  obtain ⟨s₀, ex₀, zp₀, tr₀, hf₀, hex₀, hdz₀, -, hrt₀, hrun₀, hk₀, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hf₁, hex₁, hdz₁, -, hrt₁, hrun₁, hk₁, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv1
  have e₁ : ex₁ = ex₀ := Except.ok.inj (hex₁.symm.trans hex₀)
  subst ex₁
  obtain ⟨hlen₀, hcr₀⟩ := hExt ex₀ hex₀
  have hm₀ : s₀ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₀)
  have hm₁ : s₁ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₁)
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀
    s₀.rounds _ tr₀ y.hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀ (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁
    s₁.rounds _ tr₁ y.hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁ (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess j.val _ _ cap y.hz hms hseam hlay hfold hExt hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZJ y.table = ex₀.size := by
    unfold nXZJ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZJ (planZHJ (Reg := Reg) dj y₀) (L.draw ω) R j =
      decZ (wRestZ (Setup.ofCircuit y.x.st).m (Setup.ofCircuit y.x.st).kLog (2 + ex₀.size) (schedOf y.sch) y.tPad
        (layZ (Setup.ofCircuit y.x.st) y.sch).len) := by
    unfold decAtZJ
    rw [hTy, hseam.hm, hseam.hk, hseam.hsch, hseam.ht, hseam.hnp, hnX]
  obtain ⟨e₀, he₀, hr₀⟩ := hstreams 0
  obtain ⟨e₁, he₁, hr₁⟩ := hstreams 1
  have e₆ : e₀ = s₀ := Option.some.inj (he₀.symm.trans hf₀)
  subst e₀
  have e₇ : e₁ = s₁ := Option.some.inj (he₁.symm.trans hf₁)
  subst e₁
  have hopn : opnAtZHJ (planZHJ (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs j =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnAtZHJ
    rw [hst]
    dsimp only
    rw [y.hz]
    dsimp only
    rw [hTy, hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · rw [hTy, hdecEq, ← hr₀]
    exact hrr₀
  · rw [hTy, hdecEq, ← hr₁]
    exact hrr₁
  · dsimp only
    rw [hTy, hopn, ← hroot]
    have hok := outZC_of_accZ hacc'
    have hN : (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length = 2 * sts[j.val].regions.size := by
      rw [DecodeZL.extraClaims_length, ← hyst]
      show 2 * (regionsOf y.x.st).length = _
      simp [regionsOf]
    have hpads := fun cp : Cap Dig => zk_table_padsZ (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
      (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ cap _ rfl y.hz hseam hlay hfold hExt hwfA hv0 hv1 s₀ s₁
      zp₀ zp₁ hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) cp hpl₀ hpl₁
    rw [hN, hyst] at hpads
    refine ⟨hok, spec, sts, sess, y.sch, y.tPad, s₀, s₁, hj, hs, hdecS, by rw [← hyst]; exact y.hz, hf₀, hf₁,
      (zk_table_rows₀Z (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
        (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ cap _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
        hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) _ hpl₀ hpl₁).trans ?_, fun ℓ h1 =>
      ⟨(zk_table_rowsLZ (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
        (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ cap _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
        hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) _ hpl₀ hpl₁ ℓ h1).1.trans ?_,
      (zk_table_rowsLZ (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
        (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ cap _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
        hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) _ hpl₀ hpl₁ ℓ h1).2.trans ?_⟩, hseam.hq, ?_, ?_, ?_⟩
    rotate_right
    · have hcaps := zk_table_capsZ (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec
        (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ cap _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
        hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) hpl₀ hpl₁
      rw [hN, hyst] at hdz₀ hdz₁
      intro r hr
      rcases r with _ | _ | r
      · exact ⟨zp₀, hdz₀, hcaps.1⟩
      · exact ⟨zp₁, hdz₁, hcaps.2⟩
      · omega
    all_goals rw [hN, hyst]
    exacts [(hpads _).1, (hpads _).2]

end FlockSoundness.Discharge.ZkHidden

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (planZJ outJ repJ)
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ planZHJ)
open FlockSoundness.Discharge.Tables (DecodesTable LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128

/-- **The deployed `--zk` decoders are right at `zkTabs` at the rows, at one outcome**: where `verify --zk` sets its
session up at the empty draw `dj ∅`, on untyped inputs under `TagsOkZJ`, the openings `opnD` and decoder `decD` at
`tabsJ` of `zkTabs` are right at `OpensRows` of `o₀`'s transcript at `wrS`, on `o₀` where `verify --zk` accepts that
transcript (`decodesTableW_zkTabs` with `runsAtZCJ_live_rows`). -/
theorem decodesTableR_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : Composed.TagsOkZJ I) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (o₀ : LiveOutJ (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length spec.mPts)
    (i : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun j out => out.accepted = true ∧ Composed.OpensRows I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o₀) j.val
        ((tabsJ sts.size (zkTabs h0 L hz hk hlay)).get j) out)
      (fun o => Composed.AcceptsAt I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o) ∧ o = o₀)
      (opnD sts (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (decD (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKJ_empty hT.1 dj hs
  obtain ⟨y₀, -, -, -⟩ := setupZJ_of_setupAt hT.1 ht dj hs hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Composed.decodes_of_runsOnZCJW
    (Composed.runsAtZCJ_live_rows (Reg := Unit) I ht dj y₀ (fun S _ o => Claims.Bridge.Custody.wrS I (dj S) o)
      { Ω := Unit, draw := fun _ => ∅ } hT (fun S _ o => Claims.Bridge.Custody.recordsLiveAtS_wrS I (dj S) o)
      (fun _ => hok)) () ()
  have hp := planZJ_eq_tabsJ hT.1 ht dj y₀ hs hok h0 L hz hk hlay ()
  revert i o₀
  rw [← hp, ← opnAtZJ_eq_opnD (planZJ dj y₀) dj hT.1 hs (by rw [hp, tabsJ_length])]
  intro o₀ i
  exact h _ _ (fun o ho => by obtain ⟨ha, rfl⟩ := ho; exact ⟨ha, fun _ _ hw => ⟨hw.1, hw⟩⟩) i

/-- **The deployed hidden-output `--zk` decoders are right at `zkTabs` at the rows, at one outcome**
(`decodesTableHW_zkTabs` with `runsAtZCJH_live_rows`). -/
theorem decodesTableHR_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : TagsOkZHJ I) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts)) (hshape : sts.forM Flock.Zk.shapeOk = .ok ())
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (o₀ : LiveOutJ (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length spec.mPts)
    (i : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun j out => out.accepted = true ∧ Composed.OpensRows I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o₀) j.val
        ((tabsJ sts.size (zkTabs h0 L hz hk hlay)).get j) out)
      (fun o => Composed.AcceptsAt I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o) ∧ o = o₀)
      (opnD sts (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (decD (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKHJ_empty hT.1 dj hs
  obtain ⟨y₀, -, -, -⟩ :=
    setupZHJ_of_setupAt hT.1 ht dj hs hshape hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Composed.decodes_of_runsOnZCJW
    (ZkHidden.runsAtZCJH_live_rows (Reg := Unit) I ht dj y₀ (fun S _ o => Claims.Bridge.Custody.wrS I (dj S) o)
      { Ω := Unit, draw := fun _ => ∅ } hT (fun S _ o => Claims.Bridge.Custody.recordsLiveAtS_wrS I (dj S) o)
      (fun _ => hok)) () ()
  have hp := planZHJ_eq_tabsJ hT.1 ht dj y₀ hs hshape hok h0 L hz hk hlay ()
  revert i o₀
  rw [← hp, ← opnAtZHJ_eq_opnD (planZHJ dj y₀) dj hT.1 hs (by rw [hp, tabsJ_length])]
  intro o₀ i
  exact h _ _ (fun o ho => by obtain ⟨ha, rfl⟩ := ho; exact ⟨ha, fun _ _ hw => ⟨hw.1, hw⟩⟩) i

namespace Gen

variable [S : ZkSetup]

/-- **The deployed `--zk` decoders are right at `zkTabs` at the rows at a masked statement in scope, at one outcome**:
`decodesTableR_zkTabs` at public outputs, `decodesTableHR_zkTabs` at hidden outputs (`decodesTableW_deploy`). -/
theorem decodesTableR_deploy {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d)
    (o₀ : LiveOutJ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts)
    (i : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (fun j out => out.accepted = true ∧ Composed.OpensRows ins.read.I d (Claims.Bridge.Custody.wrS ins.read.I d o₀)
        j.val ((tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).get j) out)
      (fun o => Composed.AcceptsAt ins.read.I d (Claims.Bridge.Custody.wrS ins.read.I d o) ∧ o = o₀)
      (opnD sts (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (decD (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) i := by
  have hlen : (tables true x).length = sts.size := by
    unfold tables
    rw [hset]
    simp
  revert i o₀
  rw [hlen]
  intro o₀ i
  obtain ⟨hzk, hsA⟩ := setupAt_of_setupOf hs hin hset
  rw [hd] at hsA
  obtain ⟨ht, hrr, hl⟩ := tags_of_inScope hs hin hset
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact decodesTableR_zkTabs ht ⟨hh, hrr, hl⟩ (fun _ : Finset (Fin 0) => d) hsA h0 L hz hk hlay o₀ i
  · have hshape := Claims.Bridge.Custody.setup_shapeOk (setupOf_some hset).2 hzk
    exact decodesTableHR_zkTabs ht ⟨hh, hrr, hl⟩ (fun _ : Finset (Fin 0) => d) hsA hshape h0 L hz hk hlay o₀ i

end Gen

/-- **The deployed `--zk` decoders are right at `zkTabs` at the rows at a masked statement in scope, at one outcome**:
`decodesTableR_zkTabs` at public outputs, `decodesTableHR_zkTabs` at hidden outputs (`decodesTableW_deploy`). -/
theorem decodesTableR_deploy {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d)
    (o₀ : LiveOutJ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts)
    (i : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (fun j out => out.accepted = true ∧ Composed.OpensRows ins.read.I d (Claims.Bridge.Custody.wrS ins.read.I d o₀)
        j.val ((tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).get j) out)
      (fun o => Composed.AcceptsAt ins.read.I d (Claims.Bridge.Custody.wrS ins.read.I d o) ∧ o = o₀)
      (opnD sts (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (decD (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) i :=
  Gen.decodesTableR_deploy (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (o₀ := o₀) (i := i)

end FlockDeploy.KS
