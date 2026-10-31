import Proofs.Flock.Claims.KS.PlainAccPrR
import Proofs.Flock.Claims.KS.PlainLeafClashJK

/-!
# The run-level half of `PlainRdInR`, at `VR rd`

`PlainRdRunK` and `PlainClashRunK` read an accepted run of `VK keep` through its record; here the same at the deployed
reader `VR rd`, whose accepted run decides (`accR_decide`) at the parsed value `jAt sha512` and the read proofs' files,
and whose simulator's live outcome is that decision's view (`PlainSimRunR`).

* `runOfRV_eq`, `runOfRV_runD`, `plainRawLevelColsRV_plain`: the run `plainRawLevelColsRV rd` reads on an accepted run's
  view is `runOfR`'s, so its level-`ℓ` rows are `plainLevelColsOf` at `shapeOf` on `liveOfJ`'s rep streams and the
  files (`colsRV_eq` and `colsR_plain` at every level).
* `wrPDJK_viewR`: the session `wrPDJK id` writes for the decision's view has its files and, at each rep stream, its
  rounds' coins.
* `plainR_opening₀`, `plainR_levelOpeningD`: each row the decision's rep `k` opens, at level 0 and at a later level, is
  an `Opening` in the run's log, against the link's `root_B` and against the rep's `ℓ − 1`-th recursive cap.
* `plainClashRunR_of`: a self-clash at the simulator's decoded outcome is a collision of an accepted run of `VR rd`
  (`PlainClashRunR`), from the J decoders' entries clash (`EntriesClashJK`, `LaterEntriesClashJK`).
* `plainRel_ofR`: an opening at the simulator's decoded outcome is a row the accepted run opens
  (`plainRawLevelColsRV`), from the J decoders' later entries (`LaterEntriesAtJK`).
* `plainRdInR_of_sim`: `PlainRdInR rd x` from a simulator at every prover (`PlainSimAtR`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open RecDeploy (outCoin)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The run the level reader reads -/

/-- **The run `plainRawLevelColsRV rd` reads on a view is `runOfR`'s** (`colsRV_eq` at `runOfRV`). -/
theorem runOfRV_eq {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
    {v : Claims.Seen} (hm : v.msgs.map Prod.fst = ms)
    (hcs : (v.msgs.map Prod.snd).map outCoin = cs.dropLast ++ [default]) (hk : cs.length = (FlockDeploy.V false).k x)
    (hacc : ((nextR rd x ms cs).eval sha512).1 = .acc) : runOfRV rd v = runOfR rd x ms cs := by
  cases hact : actR rd x ms cs with
  | rej => revert hacc; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert hacc; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    have hk' : (FlockDeploy.V false).k x = 2 + FlockDeploy.rounds false spec sts := by
      simp only [FlockDeploy.V, actR_setupOf hact]
    have h2 : 2 ≤ cs.length := by omega
    obtain ⟨fl, hfl, hfs⟩ := decR_rd (actR_decide hact)
    have hx : fl.stmt = x := hfs.symm.trans (checksAtJ_ok (decideR_sha512 hact hacc).2.1).1
    funext j
    unfold runOfRV
    rw [runOfR_of_actR hact, hm, hcs, hfl]
    simp only [hx, decR_dropLast rd x ms h2, actR_decide hact]

/-- **On an accepted run of `VR rd`, `runOfRV` is `runOfR` at its messages and coins** (`colsRV_run` at `runOfRV`). -/
theorem runOfRV_runD {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) :
    runOfRV rd (Claims.run sha512 P (VR rd) x ω r).seen =
      runOfR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) := by
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  refine runOfRV_eq rfl ?_ ?_ hacc
  · rw [List.length_map]
    exact run_outCoin_VR rd P x ω r h
  · have hk := accLast_VR rd x _ _ (by simp) hacc
    rw [List.length_map] at hk
    simp only [List.length_map, List.length_range]
    exact hk

/-- **The level reader at a `nextR` decision reads rep `k`'s level-`ℓ` rows at `TxNoOffAtJK`'s shape**
(`colsR_plain` at every level): where the view's run is `runOfR`'s at a step that decides at `(spec, sts)` with first
table `st` and two link points, `liveOfJ` reads `jAt sha512` and the verdict accepts it with the read proofs' files,
rep `k`'s level-`ℓ` rows are `plainLevelColsOf` at `shapeOf` with `2 · |regions|` extra claims, on `liveOfJ`'s rep
stream `k` and file `k`. -/
theorem plainRawLevelColsRV_plain {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {v : Claims.Seen} (hrv : runOfRV rd v 0 = runOfR rd x as cs 0)
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {lv : Live} {st : Stmt} {sch : Schedule}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hlv : liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv)
    (hv : verifyJWithMK id With.std spec (tabsOf sts) (jAt sha512 spec sts c₁ mids f)
      (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps) = .ok ())
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hp2 : spec.points = 2) {k : ℕ} (hk : k < 2)
    (ℓ : ℕ) :
    plainRawLevelColsRV rd v k ℓ =
      plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD k #[])
        ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty) ℓ := by
  obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hst
  obtain ⟨sess, schs, hdec, -, -, -, hall⟩ := verifyJWithMK_tables_cap id With.std hv
  have h0' : 0 < (tabsOf sts).size := by simpa [tabsOf] using h0
  obtain ⟨_, _, hf, hrep, -⟩ := hall 0 h0'
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit sts[0] := tabsOf_getElem h0'
  simp only [ht0] at hf hrep
  obtain ⟨-, extras, -, -, hex, -⟩ := verifyRepWith_rounds With.std rfl hf hrep
  obtain ⟨-, hpts, -, -⟩ := decodeJWithK_live id With.std hdec hlv
  rw [hpts] at hex
  have hx : extras.size = 2 * sts[0].regions.size := by
    rw [extraClaims_size (st := sts[0]) hex, liveOfJ_points spec c₁ mids _ hlv,
      show (pointsOf spec c₁).size = spec.points by simp [pointsOf], hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit sts[0]).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit sts[0]).m = sts[0].m from rfl, hsch]
    rfl
  have hrun : runOfR rd x as cs 0 = some
      { m := (Setup.ofCircuit sts[0]).m, kLog := (Setup.ofCircuit sts[0]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[0]).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit sts[0]),
        reps := [(lv.streams.getD 0 #[], (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 0
                    ByteArray.empty),
                 (lv.streams.getD 1 #[], (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 1
                    ByteArray.empty)] } := by
    unfold runOfR
    simp [ha, hlv, hst, hex, Except.toOption]
  unfold plainRawLevelColsRV
  rw [hrv, hrun]
  rcases k with _ | _ | k
  · simp only [Claims.Flock.ZkRun.plainShape, FlockDeploy.plainProofShape_eq hsched, hx, List.getElem?_cons_zero]
  · simp only [Claims.Flock.ZkRun.plainShape, FlockDeploy.plainProofShape_eq hsched, hx, List.getElem?_cons_succ,
      List.getElem?_cons_zero]
  · omega

/-! ## The session `wrPDJK` writes for a decision's view -/

/-- **The session `wrPDJK id` writes for a `nextR` decision's view**: wherever the verdict accepts it at the setup
at `d`, its proof files are the decision's files, and each rep stream of table 0 the session decodes to has, round by
round, the coins of `liveOfJ`'s rep stream (`RecordsLiveAtDJK`, through the rounds' digests). -/
theorem wrPDJK_viewR {I : Exec.Inputs} {d : Json} {mPts : ℕ} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP} {lv : Live}
    (hlv : liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv)
    (hA : AcceptsAtDJK id I (some d) (wrPDJK id I d (viewOfJ 1 mPts spec link c₁ (jAt sha512 spec sts c₁ mids f)
      (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)))) :
    (wrPDJK id I d (viewOfJ 1 mPts spec link c₁ (jAt sha512 spec sts c₁ mids f)
        (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps))).proofs =
      FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps ∧
    ∀ (spec' : SessionSpec) (sts' : Array Stmt) (sess : RunRecord), setupAtD I (some d) = .ok (spec', sts') →
      Record.decodeJWithK id With.std spec' (wrPDJK id I d (viewOfJ 1 mPts spec link c₁
        (jAt sha512 spec sts c₁ mids f) (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps))).j = .ok sess →
      ∀ k < 2, ∀ y : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec'.tables[0]! ++ toString "/rep" ++ toString k) =
          some y →
        ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
          (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
  obtain ⟨hpr, hall⟩ := recordsLiveAtDJK_wrPDJK id I d _ hA
  refine ⟨hpr, fun spec' sts' sess hsu hdec k hk y hy j => ?_⟩
  obtain ⟨-, hrs⟩ := hall spec' sts' sess hsu hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨k, hk⟩
  obtain rfl : y₀ = y := Option.some.inj (hy₀.symm.trans hy)
  have hD : liveDJ spec (jAt sha512 spec sts c₁ mids f) = lv := by simp [liveDJ, hlv]
  rw [← Array.getElem?_toList, coins_getElem?_of_erase hr₀ j]
  simp only [viewOfJ, hD]
  rw [show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
  simp [withSha, Option.map_map, Function.comp_def]

/-! ## An accepted run's openings at every level -/

/-- **Each level-0 row a `nextR` decision's rep opens is an `Opening` in the run's log** (`plainR_opening`'s proof,
from the decision's rep columns): on an accepted run of `VR rd` deciding at `(spec, sts)`, with first table `st` and two
link points, a column rep `k` opens at `q` (`plainRepColsOf` at `shapeOf` on `liveOfJ`'s rep stream `k` and file `k`)
is an opened row whose opening under table 0's scheme against `root_B[0]` of the decision's link, every SHA-512 input
of it in the run's verifier log. -/
theorem plainR_opening₀ {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP}
    (hact : actR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
      .decide s spec sts link c₁ mids f)
    (hrun : Runs sha512 (decisionSF s x link c₁ mids f) () ((Claims.run sha512 P (VR rd) x ω r).vLog.map Prod.fst))
    {lv : Live} (hlv : liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv) {st : Stmt} {sch : Schedule}
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hp2 : spec.points = 2) {k : ℕ} (hk : k < 2)
    {q : ℕ} {u : List 𝔽}
    (hu : (q, u) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size))
      (lv.streams.getD k #[]) ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty)) :
    ∃ row : Array F128, Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default)
        (2 ^ (sch.levels.getD 0 default).lanes) link.1.toArray[0]! (Claims.run sha512 P (VR rd) x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  obtain ⟨-, hc, hs, -, -⟩ := decideR_sha512 hact hacc
  have hv := verdictJ_within (decisionSF_runs hs hrun)
  generalize hL : (Claims.run sha512 P (VR rd) x ω r).vLog = L at hv ⊢
  obtain ⟨-, lv', hlv', hcu⟩ := checksAtJ_ok hc
  rw [hlv] at hlv'
  obtain rfl : lv = lv' := Except.ok.inj hlv'
  obtain ⟨hroot, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  have hsts0 : 0 < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hst0 : sts[0] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, hdec, hsz, hreps⟩ := plainQueryRound_verifyJWithMK id (hashesIn (L.map Prod.fst)) spec
    (tabsOf sts) _ _ (tabsOf_fold sts) hv
  obtain ⟨sch', hsch', hrep⟩ := hreps 0 h0
  obtain ⟨stream, extras, p, hstream, hext, hp', hlt, hcols, hmc, hcap⟩ := hrep k hk
  simp only [ht0] at hsch' hext hp' hlt hcols hmc
  have e : sch' = sch := Option.some.inj (hsch'.symm.trans hsch)
  subst e
  obtain ⟨hrB, hpts, hlsz, hstreams⟩ := decodeJWithK_live id (hashesIn (L.map Prod.fst)) hdec hlv
  rw [hpts] at hext
  have hx : extras.size = 2 * st.regions.size := by
    rw [extraClaims_size (st := st) hext, liveOfJ_points spec c₁ mids _ hlv,
      show (pointsOf spec c₁).size = spec.points by simp [pointsOf], hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit st).m = sch' := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit st).m = st.m from rfl, hsch]
    rfl
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hcolsEq : Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch' (2 * st.regions.size))
      (lv.streams.getD k #[]) ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty) =
      Claims.Flock.zkRepCols (sch'.levels.getD 0 default)
        (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch'.k0
          (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcols, ← hx, ← FlockDeploy.plainProofShape_eq hsched]
    have e1 : lv.streams.getD k #[] = lv.streams[2 * 0 + k]! := by
      rw [Nat.mul_zero, Nat.zero_add, show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    have e2 : (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty =
        (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[2 * 0 + k]! := by
      rw [Nat.mul_zero, Nat.zero_add, show ByteArray.empty = default from rfl, getD_eq_getElem!]
    rw [e1, e2]
    refine plainRepColsOf_congr _ _ fun i => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins i).symm
  rw [hcolsEq] at hu
  obtain ⟨j, hj, -, hjq, hju⟩ := mem_zkRepCols hu
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
      (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases checksAtJ_leaf hc hs 0 hsts0 with e | ⟨-, e⟩
    · rw [hst0] at e
      exact Or.inl e
    · rw [hst0] at e
      exact Or.inr e
  have hdl : (Setup.ofCircuit st).merkle.digestLen = 64 := by
    rcases hms with e | e <;> rw [e] <;> rfl
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) (Setup.ofCircuit st).merkle)
    (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
  obtain ⟨hpure, hlog⟩ := merkleCheck_logsOf hms hruns
  obtain ⟨hcs, hcw⟩ := FlockSoundness.Refine.merkleCheck_cap hpure
  refine ⟨p.ligerito.initialProof.rows[j]!,
    ⟨hms, p.cap, p.ligerito.initialProof.salts.getD j ByteArray.empty,
      sibsOf p.ligerito.initialProof.siblings ((sch'.levels.getD 0 default).d - (sch'.levels.getD 0 default).cap) j,
      hcs, fun b hb => (hcw b hb).trans hdl, ?_, ?_,
      fun b hb => (merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure j b hb).trans hdl, ?_⟩, hju⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hjq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure j hj
  · rw [← hjq]
    exact hlog j hj

/-- **Each later-level row a `nextR` decision's rep opens is an `Opening` in the run's log** (`plain_levelOpeningK`'s
proof, at the decision): on an accepted run of `VR rd` deciding at `(spec, sts)`, with first table `st` and two link
points, a level-`ℓ` row rep `k` opens at `q` (`plainLevelColsOf` at `shapeOf` on `liveOfJ`'s rep stream `k` and file
`k`) is an `Opening` of `q` against the flattened `ℓ − 1`-th recursive cap of rep `k`'s proof, file `k` decoded at
`shapeOf`, every SHA-512 input of it in the run's verifier log. -/
theorem plainR_levelOpeningD {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP}
    (hact : actR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
      .decide s spec sts link c₁ mids f)
    (hrun : Runs sha512 (decisionSF s x link c₁ mids f) () ((Claims.run sha512 P (VR rd) x ω r).vLog.map Prod.fst))
    {lv : Live} (hlv : liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv) {st : Stmt} {sch : Schedule}
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hp2 : spec.points = 2) {k : ℕ} (hk : k < 2)
    {ℓ q : ℕ} {a : List 𝔽}
    (ha : (q, a) ∈ plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size))
      (lv.streams.getD k #[]) ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty)
      ℓ) :
    ∃ (p : Proof) (row : Array F128),
      decodeProof (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size))
        ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty) = .ok p ∧
      a = row.toList.map FlockLevel3.GF128.of ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) (Claims.run sha512 P (VR rd) x ω r).vLog
        q row := by
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  obtain ⟨-, hc, hs, -, -⟩ := decideR_sha512 hact hacc
  have hv := verdictJ_within (decisionSF_runs hs hrun)
  generalize hL : (Claims.run sha512 P (VR rd) x ω r).vLog = L at hv ⊢
  have hsts0 : 0 < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hst0 : sts[0] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, schs, hdec, hsz, -, -, hall⟩ := verifyJWithMK_tables_cap id (hashesIn (L.map Prod.fst)) hv
  obtain ⟨hi', c, hschs, h0r, h1r⟩ := hall 0 h0
  have hrep : verifyRepWith (hashesIn (L.map Prod.fst)) spec (tabsOf sts)[0] schs[0] sess 0 k
      (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[2 * 0 + k]! = .ok c := by
    rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
    · simpa using h0r
    · exact h1r
  rw [ht0] at hschs hrep
  have e : schs[0] = sch := Option.some.inj (hschs.symm.trans hsch)
  rw [e] at hrep
  obtain ⟨hfi, hkl⟩ := tabsOf_fold sts (Setup.ofCircuit st) (ht0 ▸ Array.getElem_mem h0)
  obtain ⟨stream, extras, p, hstream, hext, hp', hlev⟩ :=
    plainLevels_verifyRepWith _ spec (Setup.ofCircuit st) sch sess 0 k _ c hsch hfi hkl hrep
  obtain ⟨-, hpts, -, hstreams⟩ := decodeJWithK_live id (hashesIn (L.map Prod.fst)) hdec hlv
  rw [hpts] at hext
  have hx : extras.size = 2 * st.regions.size := by
    rw [extraClaims_size (st := st) hext, liveOfJ_points spec c₁ mids _ hlv,
      show (pointsOf spec c₁).size = spec.points by simp [pointsOf], hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit st).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit st).m = st.m from rfl, hsch]
    rfl
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have e2 : (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty =
      (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[2 * 0 + k]! := by
    rw [Nat.mul_zero, Nat.zero_add, show ByteArray.empty = default from rfl, getD_eq_getElem!]
  have hconv : plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size))
      (lv.streams.getD k #[]) ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty) ℓ =
      plainLevelColsOf (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
        (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
        y.rounds (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[2 * 0 + k]! ℓ := by
    rw [← hx, ← FlockDeploy.plainProofShape_eq hsched, e2]
    have e1 : lv.streams.getD k #[] = lv.streams[2 * 0 + k]! := by
      rw [Nat.mul_zero, Nat.zero_add, show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    rw [e1]
    refine plainLevelColsOf_congr _ _ _ fun i => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins i).symm
  rw [hconv] at ha
  have hshs : (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
      (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st))).sched = sch := by
    have hsch' : fast100 (Setup.ofCircuit st).m = some sch := hsch
    simp only [Claims.Flock.plainProofShape, Claims.Flock.plainSched, hsch']
    rfl
  have hℓ : 1 ≤ ℓ ∧ ℓ < sch.levels.size := by
    by_contra hn
    rw [plainLevelColsOf, ite_eq_right (by rw [hshs]; exact hn)] at ha
    simp at ha
  obtain ⟨hent, hmc⟩ := hlev ℓ hℓ.1 hℓ.2
  have hent' : plainLevelColsOf (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
      (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
      y.rounds (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[2 * 0 + k]! ℓ =
      Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
        (y.rounds[plainLevelRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size)
          (sch.levels.size - 1) ℓ]!).coins (openAt p.ligerito (sch.levels.size - 1) ℓ).1 := hent
  rw [hent'] at ha
  obtain ⟨j, hj, -, hjq, hja⟩ := mem_zkRepCols ha
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
      (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases checksAtJ_leaf hc hs 0 hsts0 with e | ⟨-, e⟩
    · rw [hst0] at e
      exact Or.inl e
    · rw [hst0] at e
      exact Or.inr e
  have hdl : (Setup.ofCircuit st).merkle.digestLen = 64 := by
    rcases hms with e | e <;> rw [e] <;> rfl
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) (Setup.ofCircuit st).merkle)
    (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
  obtain ⟨hpure, hlog⟩ := merkleCheck_logsOf hms hruns
  obtain ⟨hcs, hcw⟩ := FlockSoundness.Refine.merkleCheck_cap hpure
  rw [hx, ← e2] at hp'
  refine ⟨p, _, hp', hja, ?_⟩
  rw [← hjq]
  exact ⟨hms, p.ligerito.recursiveCaps[ℓ - 1]!,
    (openAt p.ligerito (sch.levels.size - 1) ℓ).2.2.getD j ByteArray.empty,
    sibsOf (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
      ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap) j,
    hcs, fun b hb => (hcw b hb).trans hdl, rfl, merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure j hj,
    fun b hb => (merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure j b hb).trans hdl, hlog j hj⟩

/-! ## The abort -/

/-- **The verdict at the draw, at the parsed value, refuses an outcome without proof files**
(`not_plainVerdictAtK_nil` at `plainVerdictAtJK`). -/
theorem not_plainVerdictAtJK_nil (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {N mPts : ℕ}
    (C : Fin N → List 𝔽) (rootB₀ : Array ByteArray) {o : LiveOutJ 1 mPts} (h : o.proofs = #[]) :
    ¬ plainVerdictAtJK keep I d C rootB₀ o := by
  rintro ⟨hA, -, -⟩
  have hp := (recordsLiveAtDJK_wrPDJK keep I d o hA).1
  rw [h] at hp
  generalize wrPDJK keep I d o = t at hA hp
  rcases t with ⟨j, proofs⟩
  subst hp
  exact not_acceptsAtDJK_nil keep I (some d) j hA

/-- **The live verdict at the draw at the simulator's outcome gives an accepting run of `VR rd`**, where the
simulator sends no proof files on a rejected run (`PlainSimR`'s fourth part). -/
theorem plain_acc_of_nilR (rd : Claims.Msg → Option FinalP) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {mPts : ℕ} {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts}
    {τ : Type} {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    (hnil : ∀ u r', (Claims.run sha512 P (VR rd) x ω (Φ (u, r'))).out ≠ .acc →
      (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').proofs = #[])
    {I : Exec.Inputs} {d : Json} {M : ℕ} {C : Fin M → List 𝔽} {rootB₀ : Array ByteArray} {u : τ}
    {r' : Fin (N x + 2) → LCoin 65536}
    (hV : plainVerdictAtJK id I d C rootB₀ (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r')) :
    (Claims.run sha512 P (VR rd) x ω (Φ (u, r'))).out = .acc := by
  by_contra hn
  exact not_plainVerdictAtJK_nil id I d C rootB₀ (hnil u r' hn) hV

/-! ## The clash -/

open Classical in
/-- **A self-clash at the simulator's decoded outcome is a collision of an accepted run of `VR rd`**
(`plainClashQ₀K` and `plainClashQLK` at `VR rd`): at a simulator of `P` that sends no proof files on a rejected run, a
live self-clash at the verdict at the draw holds only on an accepted run of `VR rd`, whose decision's view is the
outcome (`PlainSimRunR`). The J decoders' entries clash (`hE`) puts two different rows of one position among the
decision's rep rows, at level 0 or at a later level, and both are `Opening`s in the run's log against one root
(`plainR_opening₀`, `plainR_levelOpeningD`). -/
theorem plainClashRunR_of (rd : Claims.Msg → Option FinalP) {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt}
    {d : Json} (hd : ins.read.draw = some d) (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512)
    (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDJK id ins.read.I (some d) (wrPDJK id ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        SelfClashL H512 enc512 Lf sch (tableOutL H512 enc512 Lf (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        EntriesClashJK id ins.read.I d (wrPDJK id ins.read.I d o) ∨
          LaterEntriesClashJK id ins.read.I d (wrPDJK id ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (P' : Claims.Prover (Ω × plainTape x))
    (hsim : PlainSimRunR rd P x ω spec.mPts plainN (plainDg spec) (plainWR spec (plain_mPts hs hin hset).1) P'
      (plainΦ x spec st))
    (hnil : ∀ u r', (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).out ≠ .acc →
      (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec (plain_mPts hs hin hset).1)
        sha512 P' x (ω, u) r').proofs = #[])
    (C : Fin (wordLen false x) → List 𝔽) :
    PlainClashRunR rd P x ω spec.mPts plainN (plainDg spec) (plainWR spec (plain_mPts hs hin hset).1) P'
      (plainΦ x spec st)
      (LiveSelfClash H512 enc512 Lf sch (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn (plainDec T)
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        fun j => repsOf (D := Dig) execArith ([T].get j)) := by
  intro u r' hq
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨a, ⟨hV, pend, hpl, rfl⟩, hcl⟩ := hq
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hcl
  have hacc := plain_acc_of_nilR rd hnil hV
  refine ⟨hacc, ?_⟩
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := accR_decide hacc
  have e := (actR_setupOf hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := hsim u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hV hpl hcl
  have hacc' := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω _ hacc
  obtain ⟨-, hc, hsu₀, hzk, hpi⟩ := decideR_sha512 hact hacc'
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨-, lv, hlv, -⟩ := checksAtJ_ok hc
  obtain ⟨hpr, hco⟩ := wrPDJK_viewR (mPts := spec.mPts) hlv hV.1
  rcases hE _ hV.1 (pend 0) (hpl 0) hcl with hc₀ | hcL
  · obtain ⟨spec'', st', sess, sch', s₀, s₁, hsu, hdec, hf', hs₀, hs₁, q, a₁, b₁, ha, hb, hab⟩ := hc₀
    obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
    subst spec'' e₅
    have hmem : ∀ {v : List 𝔽}, (q, v) ∈
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st') sch' (2 * st'.regions.size)) s₀.rounds
          (wrPDJK id ins.read.I d (viewOfJ 1 spec.mPts spec link c₁ (jAt sha512 spec #[st'] c₁ mids f)
            (FlockDeploy.filesAt (tabsOf #[st']) (pointsOf spec c₁) f.ps))).proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st') sch' (2 * st'.regions.size)) s₁.rounds
          (wrPDJK id ins.read.I d (viewOfJ 1 spec.mPts spec link c₁ (jAt sha512 spec #[st'] c₁ mids f)
            (FlockDeploy.filesAt (tabsOf #[st']) (pointsOf spec c₁) f.ps))).proofs[1]! →
        ∃ row : Array F128, Opening (Setup.ofCircuit st').merkle (sch'.levels.getD 0 default)
          (2 ^ (sch'.levels.getD 0 default).lanes) link.1.toArray[0]!
          (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).vLog q row ∧
          v = row.toList.map FlockLevel3.GF128.of := by
      intro v hv
      rw [hpr, plainRepColsOf_congr _ _ (hco spec #[st'] sess hsu hdec 0 (by omega) s₀ hs₀),
        plainRepColsOf_congr _ _ (hco spec #[st'] sess hsu hdec 1 (by omega) s₁ hs₁), ← getD_empty,
        ← getD_empty] at hv
      rcases List.mem_append.1 hv with hv | hv
      · exact plainR_opening₀ hacc hact hrun hlv (by simp) hf' hp2 (by omega) hv
      · exact plainR_opening₀ hacc hact hrun hlv (by simp) hf' hp2 (by omega) hv
    obtain ⟨row, ho, hra⟩ := hmem ha
    obtain ⟨row', ho', hrb⟩ := hmem hb
    exact tiedColl_of_append_self (tiedColl_of_openings ho ho' fun e => hab (by rw [hra, hrb, e]))
  · obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', k, hk, y, hy, ℓ, q, a₁, b₁, ha, hb, hab⟩ := hcL
    obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
    subst spec'' e₅
    have hmem : ∀ {v : List 𝔽}, (q, v) ∈
        plainLevelColsOf (shapeOf (Setup.ofCircuit st') sch' (2 * st'.regions.size)) y.rounds
          (wrPDJK id ins.read.I d (viewOfJ 1 spec.mPts spec link c₁ (jAt sha512 spec #[st'] c₁ mids f)
            (FlockDeploy.filesAt (tabsOf #[st']) (pointsOf spec c₁) f.ps))).proofs[k]! ℓ →
        ∃ (p : Proof) (row : Array F128),
          decodeProof (shapeOf (Setup.ofCircuit st') sch' (2 * st'.regions.size))
            ((FlockDeploy.filesAt (tabsOf #[st']) (pointsOf spec c₁) f.ps).getD k ByteArray.empty) = .ok p ∧
          v = row.toList.map FlockLevel3.GF128.of ∧
          Opening (Setup.ofCircuit st').merkle (sch'.levels.getD ℓ default) (2 ^ (sch'.levels.getD ℓ default).lanes)
            (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
            (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).vLog q row := by
      intro v hv
      rw [hpr, plainLevelColsOf_congr _ _ _ (hco spec #[st'] sess hsu hdec k hk y hy), ← getD_empty] at hv
      exact plainR_levelOpeningD hacc hact hrun hlv (by simp) hf' hp2 hk hv
    obtain ⟨p, row, hp, hra, ho⟩ := hmem ha
    obtain ⟨p', row', hp', hrb, ho'⟩ := hmem hb
    obtain rfl := Except.ok.inj (hp.symm.trans hp')
    exact plainLevelClash_tied ⟨_, q, row, row', fun e => hab (by rw [hra, hrb, e]), ho, ho'⟩

/-! ## The rows -/

/-- **An opening at the simulator's decoded outcome is a row the accepted run opens** (`plainRel_ofK` at `VR rd`): at
a simulator of `P` that sends no proof files on a rejected run, a decoded live outcome at the verdict at the draw is
an accepted run of `VR rd`, and an opening of its rep `r`'s level-`ℓ + 1` slot is a row of that level the run's level
reader holds (`plainRawLevelColsRV`), through the J decoders' later entries (`hE`). -/
theorem plainRel_ofR (rd : Claims.Msg → Option FinalP) {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt}
    {d : Json} (hd : ins.read.draw = some d) (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512)
    (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDJK id ins.read.I (some d) (wrPDJK id ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAtJK id sch ins.read.I d (wrPDJK id ins.read.I d o)
          (tableOutL H512 enc512 Lf (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (P' : Claims.Prover (Ω × plainTape x))
    (hsim : PlainSimRunR rd P x ω spec.mPts plainN (plainDg spec) (plainWR spec (plain_mPts hs hin hset).1) P'
      (plainΦ x spec st))
    (hnil : ∀ u r', (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).out ≠ .acc →
      (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec (plain_mPts hs hin hset).1)
        sha512 P' x (ω, u) r').proofs = #[])
    (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536)
    (a : List (Model.OutCL 𝔽 Dig (List UInt8)))
    (ha : Claims.Bridge.RetOK (fun _ cap pend os => tableOutL H512 enc512 Lf cap pend os)
      (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Dig) execArith ([T].get j))
      (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec (plain_mPts hs hin hset).1)
        sha512 P' x (ω, u) rr) a)
    (r : Fin 2) (ℓ : ℕ) (q : Fin (2 ^ (lvOf sch (ℓ + 1)).logLen)) (v : Fin (2 ^ (lvOf sch (ℓ + 1)).lanes) → 𝔽)
    (hv : opCapL H512 enc512 Lf sch r.val ℓ a q = some v) :
    (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, rr))).out = .acc ∧
      ∃ e, (q.val, e) ∈ plainRawLevelColsRV rd (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, rr))).seen
        r.val (ℓ + 1) ∧ rowFin (lvOf sch (ℓ + 1)).lanes e = v := by
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨hV, pend, hpl, rfl⟩ := ha
  have hacc := plain_acc_of_nilR rd hnil hV
  refine ⟨hacc, ?_⟩
  unfold opCapL at hv
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hv
  obtain ⟨dc, -, hdc⟩ := Option.bind_eq_some_iff.1 hv
  obtain ⟨y, hy, hyq, hyv⟩ := opRL_eq_some (Option.isSome_iff_exists.2 ⟨v, hdc⟩)
  refine ⟨y.2.1, ?_, Option.some.inj (hyv.symm.trans hdc)⟩
  rw [← hyq]
  have hL := hE _ hV.1 (pend 0) (hpl 0) r.val r.isLt
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := accR_decide hacc
  have e := (actR_setupOf hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := hsim u rr hacc ins spec sts link c₁ mids f hact
  rw [hview] at hV hL
  have hacc' := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω _ hacc
  obtain ⟨hvd, hc, hsu₀, hzk, hpi⟩ := decideR_sha512 hact hacc'
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨-, lv, hlv, -⟩ := checksAtJ_ok hc
  obtain ⟨hpr, hco⟩ := wrPDJK_viewR (mPts := spec.mPts) hlv hV.1
  obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', -, y₀, p, hy₀, -, -, hent⟩ := hL
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  have hcols := plainRawLevelColsRV_plain (congrFun (runOfRV_runD hacc) 0) hact hlv hvd (by simp) hf' hp2 r.isLt (ℓ + 1)
  rw [hcols, ← hpr]
  have hmem := List.mem_map_of_mem (f := fun x => (x.1, x.2.1)) hy
  rw [hview, hent (ℓ + 1) (by omega), plainLevelColsOf_congr _ _ _ (hco spec #[st'] sess hsu hdec r.val r.isLt y₀ hy₀),
    ← getD_empty] at hmem
  exact hmem

/-! ## `PlainRdInR` from the simulator -/

/-- **A simulator at every prover** at the plain statement's one table: at every in-scope plain statement that sets up
with one table, each prover `P` and tape `ω` have a simulator `P'` (`PlainSimR`). -/
def PlainSimAtR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω),
    ∃ P' : Claims.Prover (Ω × plainTape x), PlainSimR rd P x ω spec st (plain_mPts hs hin hset).1 P'

/-- **`PlainRdInR` from a simulator**: at each leaf form of the statement's tags, the J decoders (`plain_hdecAtJK`,
`plain_hdecHAtJK`, `plain_hdecHUAtJK`), their self-clash entries (`plain_clash₀*AtJK`, `plain_laterClash*AtJK`) and
later entries (`plain_laterEntries*AtJK`) at `keep := id`, at the simulator `PlainSimAtR` gives, through
`plainClashRunR_of` and `plainRel_ofR`. -/
theorem plainRdInR_of_sim (rd : Claims.Msg → Option FinalP) {x : Claims.Msg} (hsim : PlainSimAtR rd x) :
    PlainRdInR rd x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay
  cases hh : ins.read.I.tags.hiddenOutputs
  · refine ⟨leaf512, _, plain_hdecAtJK id hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay,
      fun P ω => ?_⟩
    obtain ⟨P', hS⟩ := hsim s hs hin ins spec sts hset st h P ω
    refine ⟨P', hS, fun C _ => ⟨?_, fun u rr a ha r ℓ q v hv => ?_⟩⟩
    · refine plainClashRunR_of rd hs hin hset hd leaf512 _ _ _ (fun o hA pend' hp hcl => ?_) P ω P' hS.2.2.1
        hS.2.2.2 C
      rcases hcl with hcl | hcl
      · exact .inl (plain_clash₀AtJK id hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay o hA pend' hp hcl)
      · exact .inr (plain_laterClashAtJK id hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay o hA pend' hp hcl)
    · exact plainRel_ofR rd hs hin hset hd leaf512 _ _ _
        (plain_laterEntriesAtJK id hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω P' hS.2.2.1 hS.2.2.2
        C u rr a ha r ℓ q v hv
  · by_cases hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"
    · refine ⟨leaf512, _, plain_hdecHAtJK id hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay,
        fun P ω => ?_⟩
      obtain ⟨P', hS⟩ := hsim s hs hin ins spec sts hset st h P ω
      refine ⟨P', hS, fun C _ => ⟨?_, fun u rr a ha r ℓ q v hv => ?_⟩⟩
      · refine plainClashRunR_of rd hs hin hset hd leaf512 _ _ _ (fun o hA pend' hp hcl => ?_) P ω P' hS.2.2.1
          hS.2.2.2 C
        rcases hcl with hcl | hcl
        · exact .inl (plain_clash₀HAtJK id hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hA pend' hp
            hcl)
        · exact .inr (plain_laterClashHAtJK id hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hA pend'
            hp hcl)
      · exact plainRel_ofR rd hs hin hset hd leaf512 _ _ _
          (plain_laterEntriesHAtJK id hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω P' hS.2.2.1
          hS.2.2.2 C u rr a ha r ℓ q v hv
    · have hl' := plain_leafU hs hin hset hl
      refine ⟨leafPlain, _, plain_hdecHUAtJK id hs hin hset h hh hl' hd L m k0 l₀ l₁ rest ext hsch hm hlay,
        fun P ω => ?_⟩
      obtain ⟨P', hS⟩ := hsim s hs hin ins spec sts hset st h P ω
      refine ⟨P', hS, fun C _ => ⟨?_, fun u rr a ha r ℓ q v hv => ?_⟩⟩
      · refine plainClashRunR_of rd hs hin hset hd leafPlain _ _ _ (fun o hA pend' hp hcl => ?_) P ω P' hS.2.2.1
          hS.2.2.2 C
        rcases hcl with hcl | hcl
        · exact .inl (plain_clash₀HUAtJK id hs hin hset h hh hl' hd L m k0 l₀ l₁ rest ext hsch hm hlay o hA pend' hp
            hcl)
        · exact .inr (plain_laterClashHUAtJK id hs hin hset h hh hl' hd L m k0 l₀ l₁ rest ext hsch hm hlay o hA
            pend' hp hcl)
      · exact plainRel_ofR rd hs hin hset hd leafPlain _ _ _
          (plain_laterEntriesHUAtJK id hs hin hset h hh hl' hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω P'
          hS.2.2.1 hS.2.2.2 C u rr a ha r ℓ q v hv

/-- **`RdAllAt` at the plain record at the deployed reader, from a simulator** (`rdAllAt_R` at `plainRdInR_of_sim`). -/
theorem rdAllAt_R_of_sim (rd : Claims.Msg → Option FinalP)
    (hsim : ∀ x, ¬ Claims.Flock.NoAcc sha512 (VR rd) x → PlainSimAtR rd x) :
    RdAllAt (deployPlainR Readable.relation rd) (fun _ => plainRawLevelColsRV rd) :=
  rdAllAt_R rd fun x hno => plainRdInR_of_sim rd (hsim x hno)

end FlockDeploy.KS
