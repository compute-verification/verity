import Proofs.Flock.Claims.KS.PlainClash0
import Proofs.Flock.Claims.KS.LevelsAll

/-!
# The live self-clash past level 0

A live self-clash past level 0 is two openings of one position to different columns among one rep's level-`ℓ`
openings, at the pendings the reps' decoded tapes play to. Where `verify` accepts a transcript recording the outcome at
the setup at the draw, those tapes play to the deployed reps' pendings, whose later requests are levels `1, …, R` in
order at the positions of their query rounds' coins (`rep_refines_reqs`), so their output's level-`ℓ` openings are the
rep's `plainLevelColsOf` entries at the transcript's proof files (`accepts_opensLDK`), and the clash is two of those
entries at one position with different columns. At the concrete simulator, where `P`'s run accepts, the outcome is the
run's and those entries are the run's level-`ℓ` rows (`rawLevelCols_plain`), which clash in its log
(`plain_levelClash`). So
`PlainSimLaterClashAt` holds at every statement, and KS at the deployed plain verifier's trimmed record holds from
`PlainSimForkLe` alone.

The draw-layer lemmas are proved once, at `VK keep` over any transcript recording the outcome (`RecordsLiveAtDK`), and
their `wrPD` forms are them at `keep := id`; `PlainClashLaterK` has them at `wrPDK keep`. The hidden-output lemmas are
proved once more, at a model leaf or a leaf form (`PlainClash0`): hm96 here and plain SHA-512 leaves in
`PlainLeafClash` are instances. The entries are `plainLevelColsOf` at the decoded rep stream's rounds, which reads their
coins only.

* `plainLevelColsOf_ok`, `plainLevelColsOf_ge`: the reader's level-`ℓ` rows at a proof that decodes, and none past the
  schedule.
* `slotOpensL_entriesAt`, `slotOpensL_level`: a rep's level-`ℓ` openings are the reader's level-`ℓ` entries.
* `accepts_reqsDK`, `accepts_opensLDLfK`: at flock-verify's acceptance through `keep` at a draw value, each rep's
  recorded caps and request at level `ℓ ≥ 1`, and each table's recorded caps and level-`ℓ` openings at a model leaf as
  its reps' proofs' recursive caps and `plainLevelColsOf` entries; `accepts_opensLDK` at `leaf512`; `accepts_reqsD`,
  `accepts_opensLD` at `keep := id`.
* `LaterEntriesAt`, `LaterEntriesAtK`, `laterEntries_wrK_of`, `laterEntries_wrKHLf_of`: at the deployed plain
  decoders, each rep's recorded caps and later openings are the deployed rep's, at public outputs and at hidden outputs
  under a leaf form; `laterEntries_wrKH_of` at hm96; `laterEntries_wrPD`, `laterEntries_wrPDH` at `wrPD`.
* `LaterEntriesClash`, `LaterEntriesClashK`, `laterEntriesClashKLf_of`, `laterEntriesClash_of`, `laterClash_wrPD`,
  `laterClash_wrPDH`: so the self-clash past level 0 there is two of one rep's level-`ℓ` entries at one position with
  different columns; `laterEntriesClashK_of` at `leaf512`. `laterEntriesAt_iff_K_id`, `laterEntriesClash_iff_K_id`:
  the `K` forms at `keep := id`.
* `plain_laterEntriesAtK_of`, `plain_laterEntriesHLfAtK_of`: those at the statement's draw, at `plainTab`;
  `plain_laterEntriesHAtK_of` at hm96; `plain_laterEntriesAt`, `plain_laterEntriesHAt`, `plain_laterClashAt`,
  `plain_laterClashHAt` at `wrPD`.
* `rawLevelCols_plain`: an accepted plain run's level-`ℓ` rows are its reps' `plainLevelColsOf` entries.
* `plainClashQL`, `plainSimLaterClashAt`: so a self-clash past level 0 at the concrete simulator's outcome gives
  `P`'s accepting run a raw clash, at every statement.
-/

namespace FlockSoundness.Discharge.AtDraw

open Claims.Bridge.PlainDraw (setupAtD AcceptsAtD AcceptsAtDK named_of_setupAtD acceptsAtD_iff_acceptsAtDK_id)
open Flock Refine Model Game FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- The deployed plain reader's level-`ℓ` rows at a proof that decodes, for a level `1 ≤ ℓ` of its schedule:
`zkRepCols` at level `ℓ`, the coins of round `plainLevelRound` and the proof's level-`ℓ` rows. -/
theorem plainLevelColsOf_ok {sh : Shape} {rounds : Array Flock.Round} {bytes : ByteArray} {p : Proof} {ℓ : ℕ}
    (h : decodeProof sh bytes = .ok p) (h1 : 1 ≤ ℓ) (hℓ : ℓ < sh.sched.levels.size) :
    FlockDeploy.KS.plainLevelColsOf sh rounds bytes ℓ = Claims.Flock.zkRepCols (sh.sched.levels.getD ℓ default)
      (rounds[FlockDeploy.KS.plainLevelRound sh.m sh.kLog sh.sched.k0 sh.claims (sh.sched.levels.size - 1) ℓ]!).coins
      (openAt p.ligerito (sh.sched.levels.size - 1) ℓ).1 := by
  rw [FlockDeploy.KS.plainLevelColsOf, ite_eq_left ⟨h1, hℓ⟩, h, Claims.Flock.coins_getElem!]

theorem plainLevelColsOf_ge {sh : Shape} {rounds : Array Flock.Round} {bytes : ByteArray} {ℓ : ℕ}
    (h : ¬ (1 ≤ ℓ ∧ ℓ < sh.sched.levels.size)) : FlockDeploy.KS.plainLevelColsOf sh rounds bytes ℓ = [] := by
  rw [FlockDeploy.KS.plainLevelColsOf, ite_eq_right h]

/-- **A rep's level-`ℓ` openings are the reader's entries**: `Claims.Flock.slotOpensL_entries` at the rep's first
request of level `ℓ`. -/
theorem slotOpensL_entriesAt {D Sl : Type} (out : Model.OutCL 𝔽 D Sl) (r ℓ : ℕ) (lv : Flock.Level)
    (coins : Array F128) (rows : Array (Array F128)) (R : Req)
    (hfind : ((if r = 0 then out.pend.1 else out.pend.2).reqs.find? fun req => req.level == ℓ) = some R)
    (hpos : R.positions = (Flock.positions lv coins).toList) (hq : R.positions.length = lv.q)
    (hrows : rows.size = R.positions.length)
    (hop : ∀ j, (out.opens r ℓ j).1 = rows[j]!.toList.map FlockLevel3.GF128.of) :
    (slotOpensL out r ℓ lv.q).map (fun x => (x.1, x.2.1)) = Claims.Flock.zkRepCols lv coins rows := by
  unfold slotOpensL Claims.Flock.zkRepCols
  rw [hfind]
  simp only [List.map_map]
  rw [← hq, List.take_length, ← hpos]
  apply List.ext_getElem
  · simp [hrows]
  · intro i h1 h2
    simp only [List.length_map, List.length_zipIdx] at h1
    simp [hop, hrows, h1]

/-- **A rep's level-`ℓ` openings, at any level `ℓ ≥ 1`, are the reader's level-`ℓ` entries**: where the rep's
requests are below the schedule's levels, its first request of a level `ℓ` of the schedule is at the positions of the
coins of round `plainLevelRound`, one opened row per position, and its openings are the proof's rows. -/
theorem slotOpensL_level {Sl : Type} (out : Model.OutCL 𝔽 Dig Sl) (r ℓ Q : ℕ) (sh : Shape)
    (rounds : Array Flock.Round) (bytes : ByteArray) (p : Proof) (pend : PendL 𝔽 Dig)
    (hp : decodeProof sh bytes = .ok p) (hpend : (if r = 0 then out.pend.1 else out.pend.2) = pend)
    (hop : ∀ k, (out.opens r ℓ k).1 = ((openAt p.ligerito (sh.sched.levels.size - 1) ℓ).1)[k]!.toList.map fOf)
    (hlv : ∀ i < sh.sched.levels.size, LevelSeam (sh.sched.levels.getD i default))
    (hQ : ℓ < sh.sched.levels.size → Q = (sh.sched.levels.getD ℓ default).q)
    (hlt : ∀ R ∈ pend.reqs, R.level < sh.sched.levels.size)
    (hfind : ℓ < sh.sched.levels.size → ∃ R : Req, pend.reqs.find? (fun q => q.level == ℓ) = some R ∧
      R.positions = (Flock.positions (sh.sched.levels.getD ℓ default)
        (rounds[FlockDeploy.KS.plainLevelRound sh.m sh.kLog sh.sched.k0 sh.claims (sh.sched.levels.size - 1)
          ℓ]!).coins).toList ∧
      (openAt p.ligerito (sh.sched.levels.size - 1) ℓ).1.size = R.positions.length)
    (h1 : 1 ≤ ℓ) :
    (slotOpensL out r ℓ Q).map (fun x => (x.1, x.2.1)) = FlockDeploy.KS.plainLevelColsOf sh rounds bytes ℓ := by
  by_cases hℓ : ℓ < sh.sched.levels.size
  · obtain ⟨R, hR, hpos, hrows⟩ := hfind hℓ
    rw [hQ hℓ, plainLevelColsOf_ok hp h1 hℓ]
    exact slotOpensL_entriesAt out r ℓ _ _ _ R (by rw [hpend]; exact hR) hpos
      (by rw [hpos, Array.length_toList, positions_size _ (hlv ℓ hℓ)]) hrows hop
  · rw [plainLevelColsOf_ge (fun h => hℓ h.2)]
    unfold slotOpensL
    rw [hpend, List.find?_eq_none.2 fun R hR => by have := hlt R hR; simp; omega]
    rfl

/-- **An accepted rep at `dr` through `keep` requests every later level**: `accepts_req0DK` past level 0, where the rep
records its proof's recursive caps, each request is below the schedule's levels and the first of level `ℓ ≥ 1` is at
the positions of the coins of round `plainLevelRound`, one opened level-`ℓ` row per position. -/
theorem accepts_reqsDK (keep : Bool → Bool)
    {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtDK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeWithK keep With.std spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (r : ℕ), r < 2 →
        ∀ (s : StreamRec) (p : Proof) (pend : PendL 𝔽 Dig),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r) =
            some s →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + r]! = .ok p →
          Plays fOf (Model.repC execArith r (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) pend →
          pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.ligerito.recursiveCaps[j]!) ∧
          (∀ ℓ, 1 ≤ ℓ → ℓ < sch.levels.size → ∃ R : Req, pend.reqs.find? (fun q => q.level == ℓ) = some R ∧
            R.positions = (Flock.positions (sch.levels.getD ℓ default)
              (s.rounds[FlockDeploy.KS.plainLevelRound sts[j].m sts[j].c.kLog sch.k0
                (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) (sch.levels.size - 1) ℓ]!).coins).toList ∧
            (openAt p.ligerito (sch.levels.size - 1) ℓ).1.size = R.positions.length) ∧
          ∀ R ∈ pend.reqs, R.level < sch.levels.size := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, -, -, -, hall⟩ := verifyK_ok_tables keep (named_of_setupAtD hs) hv
  obtain ⟨hp2, hmp⟩ := HiddenExec.setupTables_pointsAny hs
  have hpts := decodeWithK_points keep With.std hsess
  refine ⟨spec, sts, sess, hs, hsess, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  refine ⟨schs[j], hf, fun hwf hr hm r hr2 s p pend hsr hp hpl => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun q hq => (hpts.2 q hq).trans (hmp j hj).symm)
  have h15 : 15 ≤ sts[j].m := by have : 22 ≤ sts[j].m := exec_fast100_ge hf; omega
  obtain ⟨hdim, hsm⟩ := fast100_dims _ schs[j] hf
  have hlv : ∀ i < schs[j].levels.size, LevelSeam (schs[j].levels.getD i default) := fun i hi =>
    fast100_level _ schs[j] hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < schs[j].levels.size, Accounting.ceilLog2 (schs[j].levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha _ schs[j] hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hrep : verifyRep spec (Setup.ofCircuit sts[j]) schs[j] sess j r t.proofs[2 * j + r]! = .ok c := by
    rcases r with _ | _ | r
    · exact h0
    · exact h1
    · omega
  obtain ⟨s', p', hs'', hp', -, -, -, -, hk⟩ := rep_refines_reqs (D := Dig) spec (Setup.ofCircuit sts[j])
    (stmtOf sts[j] hwf (regionsOf sts[j])) hm h15 schs[j] rfl rfl (ofCircuit_fold sts[j] hwf _) _ sess hext hdim hlv
    hα1 hsm (decodeWithK_roundsWF keep With.std hsess) j r _ _ hrep
  obtain rfl : s' = s := Option.some.inj (hs''.symm.trans hsr)
  obtain rfl : p' = p := Except.ok.inj (hp'.symm.trans hp)
  obtain ⟨pend', hpl', hcaps, hok, -, R, rest, hRe, hRl, -, hlen, hlat⟩ :=
    hk capDig (fun _ ℓ k => (((openAt p'.ligerito (schs[j].levels.size - 1) ℓ).1)[k]!.toList.map fOf, []))
      (fun _ _ _ _ => rfl)
  obtain rfl := ZkExec.plays_det hpl hpl'
  refine ⟨hcaps, fun ℓ h1 hℓ => ?_, fun R' hR' => (hok R' hR').1⟩
  have hlt : ℓ - 1 < rest.length := by omega
  obtain ⟨hlv', hpos'⟩ := hlat (ℓ - 1) hlt
  rw [Nat.sub_add_cancel h1] at hlv' hpos'
  have hmc := (hok rest[ℓ - 1] (by rw [hRe]; exact List.mem_cons_of_mem _ (List.getElem_mem hlt))).2.2.2
  rw [hlv'] at hmc
  refine ⟨rest[ℓ - 1], by rw [hRe]; exact find?_later hRl hlat h1 (by omega), hpos', ?_⟩
  simpa using (merkleCheck_rows hmc).1

/-- **An accepted rep at `dr` requests every later level**: `accepts_reqsDK` at `keep := id`. -/
theorem accepts_reqsD {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (r : ℕ), r < 2 →
        ∀ (s : StreamRec) (p : Proof) (pend : PendL 𝔽 Dig),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r) =
            some s →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + r]! = .ok p →
          Plays fOf (Model.repC execArith r (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) pend →
          pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.ligerito.recursiveCaps[j]!) ∧
          (∀ ℓ, 1 ≤ ℓ → ℓ < sch.levels.size → ∃ R : Req, pend.reqs.find? (fun q => q.level == ℓ) = some R ∧
            R.positions = (Flock.positions (sch.levels.getD ℓ default)
              (s.rounds[FlockDeploy.KS.plainLevelRound sts[j].m sts[j].c.kLog sch.k0
                (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) (sch.levels.size - 1) ℓ]!).coins).toList ∧
            (openAt p.ligerito (sch.levels.size - 1) ℓ).1.size = R.positions.length) ∧
          ∀ R ∈ pend.reqs, R.level < sch.levels.size := by
  simpa only [decodeWithK_id_std] using accepts_reqsDK id ((acceptsAtD_iff_acceptsAtDK_id I dr t).1 h)

/-- **At flock-verify's acceptance at `dr` through `keep`, each table's recorded caps and level-`ℓ` openings at the
model leaf `lf` are its reps' deployed ones**, at every level `ℓ ≥ 1`: each rep's caps its proof's recursive caps and
its level-`ℓ` openings its `plainLevelColsOf` entries (`accepts_reqsDK`). -/
theorem accepts_opensLDLfK (keep : Bool → Bool) (lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512)
    {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtDK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeWithK keep With.std spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
          (pend₀ pend₁ : PendL 𝔽 Dig) (cap : Cap Dig) (out : OutCL 𝔽 Dig (List UInt8)),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) =
            some s₀ →
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) =
            some s₁ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ →
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ →
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ →
          out = tableOutL H512 enc512 lf cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          pend₀.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₀.ligerito.recursiveCaps[j]!) ∧
          pend₁.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₁.ligerito.recursiveCaps[j]!) ∧
          ∀ ℓ, 1 ≤ ℓ →
            (slotOpensL out 0 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ℓ ∧
            (slotOpensL out 1 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! ℓ := by
  obtain ⟨spec, sts, sess, hs, hsess, hall⟩ := accepts_reqsDK keep h
  refine ⟨spec, sts, sess, hs, hsess, fun j hj => ?_⟩
  obtain ⟨sch, hf, hk⟩ := hall j hj
  refine ⟨sch, hf, fun hwf hr hm p₀ p₁ s₀ s₁ pend₀ pend₁ cap out hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ hout => ?_⟩
  subst hout
  obtain ⟨hc₀, hf₀, hlt₀⟩ := hk hwf hr hm 0 (by norm_num) s₀ p₀ pend₀ hs₀ hp₀ hpl₀
  obtain ⟨hc₁, hf₁, hlt₁⟩ := hk hwf hr hm 1 (by norm_num) s₁ p₁ pend₁ hs₁ hp₁ hpl₁
  refine ⟨hc₀, hc₁, fun ℓ h1 => ?_⟩
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level _ sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hQ : ℓ < sch.levels.size → (lvOf (schedOf sch) ℓ).queries = (sch.levels.getD ℓ default).q := fun hℓ => by
    rw [lvOf_schedOf hℓ]
    rfl
  rw [opnOf_pair hp₀ hp₁]
  exact ⟨slotOpensL_level _ 0 ℓ _ _ _ _ p₀ pend₀ hp₀ rfl (fun _ => rfl) hlv hQ hlt₀ (fun hℓ => hf₀ ℓ h1 hℓ) h1,
    slotOpensL_level _ 1 ℓ _ _ _ _ p₁ pend₁ hp₁ rfl (fun _ => rfl) hlv hQ hlt₁ (fun hℓ => hf₁ ℓ h1 hℓ) h1⟩

/-- **At flock-verify's acceptance at `dr` through `keep`, each table's recorded caps and level-`ℓ` openings are its
reps' deployed ones**, at every level `ℓ ≥ 1`: each rep's caps its proof's recursive caps and its level-`ℓ` openings its
`plainLevelColsOf` entries (`accepts_reqsDK`); `accepts_opensLDLfK` at `leaf512`. -/
theorem accepts_opensLDK (keep : Bool → Bool)
    {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtDK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeWithK keep With.std spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
          (pend₀ pend₁ : PendL 𝔽 Dig) (cap : Cap Dig) (out : OutCL 𝔽 Dig (List UInt8)),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) =
            some s₀ →
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) =
            some s₁ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ →
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ →
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ →
          out = tableOutL H512 enc512 leaf512 cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          pend₀.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₀.ligerito.recursiveCaps[j]!) ∧
          pend₁.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₁.ligerito.recursiveCaps[j]!) ∧
          ∀ ℓ, 1 ≤ ℓ →
            (slotOpensL out 0 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ℓ ∧
            (slotOpensL out 1 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! ℓ :=
  accepts_opensLDLfK keep leaf512 h

/-- **At flock-verify's acceptance at `dr`, each table's recorded caps and level-`ℓ` openings are its reps' deployed
ones**, at every level `ℓ ≥ 1`: `accepts_opensLDK` at `keep := id`. -/
theorem accepts_opensLD {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
          (pend₀ pend₁ : PendL 𝔽 Dig) (cap : Cap Dig) (out : OutCL 𝔽 Dig (List UInt8)),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) =
            some s₀ →
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) =
            some s₁ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ →
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ →
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ →
          out = tableOutL H512 enc512 leaf512 cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          pend₀.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₀.ligerito.recursiveCaps[j]!) ∧
          pend₁.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₁.ligerito.recursiveCaps[j]!) ∧
          ∀ ℓ, 1 ≤ ℓ →
            (slotOpensL out 0 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ℓ ∧
            (slotOpensL out 1 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! ℓ := by
  simpa only [decodeWithK_id_std] using accepts_opensLDK id ((acceptsAtD_iff_acceptsAtDK_id I dr t).1 h)

end FlockSoundness.Discharge.AtDraw

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **Two of one rep's level-`ℓ` entries at one position with different columns**, in the transcript `t` at the setup
at the draw `d`: at its one table, that table's `fast100` schedule and one of its record's rep streams there, two of
the rep's `plainLevelColsOf` entries at `t`'s proof file, at the shape `TxNoOffAt` reads, open one position to
different columns. -/
def LaterEntriesClash (I : Exec.Inputs) (d : Json) (t : Exec.Transcript) : Prop :=
  ∃ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : Session) (sch : _root_.Flock.Schedule),
    setupAtD I (some d) = .ok (spec, #[st]) ∧ Record.decode spec t.record = .ok sess ∧
    _root_.Flock.fast100 st.m = some sch ∧ ∃ r < 2, ∃ s : StreamRec,
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r) = some s ∧
    ∃ (ℓ q : ℕ) (a b : List 𝔽),
      (q, a) ∈ FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ ∧
      (q, b) ∈ FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ ∧ a ≠ b

/-- **Rep `r`'s recorded caps and later openings in `out` are the deployed rep's**, in the transcript `t` at the setup
at the draw `d`: at its one table, that table's `fast100` schedule (`sch` as the model's), rep `r`'s record stream and
`t`'s proof file decoded at the shape `TxNoOffAt` reads, `out`'s caps for rep `r` are the proof's recursive caps, and
its level-`ℓ` openings, at every level `ℓ ≥ 1`, are the rep's `plainLevelColsOf` entries. -/
def LaterEntriesAt (sch : Accounting.Schedule) (I : Exec.Inputs) (d : Json) (t : Exec.Transcript)
    (out : Model.OutCL 𝔽 Dig (List UInt8)) (r : ℕ) : Prop :=
  ∃ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : Session) (sch' : _root_.Flock.Schedule),
    setupAtD I (some d) = .ok (spec, #[st]) ∧ Record.decode spec t.record = .ok sess ∧
    _root_.Flock.fast100 st.m = some sch' ∧ schedOf sch' = sch ∧ ∃ (s : StreamRec) (p : Proof),
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r) = some s ∧
    decodeProof (shapeOf (Setup.ofCircuit st) sch' (2 * st.regions.size)) t.proofs[r]! = .ok p ∧
    (if r = 0 then out.pend.1 else out.pend.2).caps =
      (List.range (sch'.levels.size - 1)).map (fun j => capDig p.ligerito.recursiveCaps[j]!) ∧
    ∀ ℓ, 1 ≤ ℓ → (slotOpensL out r ℓ (lvOf sch ℓ).queries).map (fun x => (x.1, x.2.1)) =
      FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch' (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ

/-- **Two of one rep's level-`ℓ` entries at one position with different columns**, in `t` at the setup at `d`, the
session the record decodes to through `keep` (`LaterEntriesClash`). -/
def LaterEntriesClashK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) (t : Exec.Transcript) : Prop :=
  ∃ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : Session) (sch : _root_.Flock.Schedule),
    setupAtD I (some d) = .ok (spec, #[st]) ∧ Record.decodeWithK keep With.std spec t.record = .ok sess ∧
    _root_.Flock.fast100 st.m = some sch ∧ ∃ r < 2, ∃ s : StreamRec,
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r) = some s ∧
    ∃ (ℓ q : ℕ) (a b : List 𝔽),
      (q, a) ∈ FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ ∧
      (q, b) ∈ FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ ∧ a ≠ b

/-- **Rep `r`'s recorded caps and later openings in `out` are the deployed rep's**, in `t` at the setup at `d`, the
session the record decodes to through `keep` (`LaterEntriesAt`). -/
def LaterEntriesAtK (keep : Bool → Bool) (sch : Accounting.Schedule) (I : Exec.Inputs) (d : Json)
    (t : Exec.Transcript) (out : Model.OutCL 𝔽 Dig (List UInt8)) (r : ℕ) : Prop :=
  ∃ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : Session) (sch' : _root_.Flock.Schedule),
    setupAtD I (some d) = .ok (spec, #[st]) ∧ Record.decodeWithK keep With.std spec t.record = .ok sess ∧
    _root_.Flock.fast100 st.m = some sch' ∧ schedOf sch' = sch ∧ ∃ (s : StreamRec) (p : Proof),
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r) = some s ∧
    decodeProof (shapeOf (Setup.ofCircuit st) sch' (2 * st.regions.size)) t.proofs[r]! = .ok p ∧
    (if r = 0 then out.pend.1 else out.pend.2).caps =
      (List.range (sch'.levels.size - 1)).map (fun j => capDig p.ligerito.recursiveCaps[j]!) ∧
    ∀ ℓ, 1 ≤ ℓ → (slotOpensL out r ℓ (lvOf sch ℓ).queries).map (fun x => (x.1, x.2.1)) =
      FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch' (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ

theorem laterEntriesClash_iff_K_id (I : Exec.Inputs) (d : Json) (t : Exec.Transcript) :
    LaterEntriesClash I d t ↔ LaterEntriesClashK id I d t := by
  unfold LaterEntriesClash LaterEntriesClashK
  simp only [decodeWithK_id_std]

theorem laterEntriesAt_iff_K_id (sch : Accounting.Schedule) (I : Exec.Inputs) (d : Json) (t : Exec.Transcript)
    (out : Model.OutCL 𝔽 Dig (List UInt8)) (r : ℕ) :
    LaterEntriesAt sch I d t out r ↔ LaterEntriesAtK id sch I d t out r := by
  unfold LaterEntriesAt LaterEntriesAtK
  simp only [decodeWithK_id_std]

/-- **A self-clash past level 0 at the model leaf `lf` in an output whose reps' later openings are the deployed reps'
is a clash of one rep's level-`ℓ` entries**, through `keep` (`laterEntriesClash_of`); the proof reads no leaf. -/
theorem laterEntriesClashKLf_of (keep : Bool → Bool) (lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512)
    {sch : Accounting.Schedule} {I : Exec.Inputs} {d : Json}
    {t : Exec.Transcript} {out : Model.OutCL 𝔽 Dig (List UInt8)} (he : ∀ r < 2, LaterEntriesAtK keep sch I d t out r)
    (hcl : FlockDeploy.KS.LaterClashLLf lf sch out) : LaterEntriesClashK keep I d t := by
  obtain ⟨r, hr2, ℓ, h1, -, -, u, hu, v, hv, huv, -, -, hne⟩ := hcl
  obtain ⟨spec, st, sess, sch', hsu, hdec, hf, -, s, -, hs, -, -, hent⟩ := he r hr2
  have hmem : ∀ z ∈ slotOpensL out r ℓ (lvOf sch ℓ).queries, (z.1, z.2.1) ∈
      FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch' (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ :=
    fun z hz => hent ℓ h1 ▸ List.mem_map_of_mem hz
  exact ⟨spec, st, sess, sch', hsu, hdec, hf, r, hr2, s, hs, ℓ, u.1, u.2.1, v.2.1, hmem u hu, huv ▸ hmem v hv, hne⟩

/-- **A self-clash past level 0 in an output whose reps' later openings are the deployed reps' is a clash of one rep's
level-`ℓ` entries**, through `keep` (`laterEntriesClash_of`); `laterEntriesClashKLf_of` at `leaf512`. -/
theorem laterEntriesClashK_of (keep : Bool → Bool) {sch : Accounting.Schedule} {I : Exec.Inputs} {d : Json}
    {t : Exec.Transcript} {out : Model.OutCL 𝔽 Dig (List UInt8)} (he : ∀ r < 2, LaterEntriesAtK keep sch I d t out r)
    (hcl : FlockDeploy.KS.LaterClashL sch out) : LaterEntriesClashK keep I d t :=
  laterEntriesClashKLf_of keep leaf512 he hcl

/-- **A self-clash past level 0 in an output whose reps' later openings are the deployed reps' is a clash of one rep's
level-`ℓ` entries**: `laterEntriesClashK_of` at `keep := id`. -/
theorem laterEntriesClash_of {sch : Accounting.Schedule} {I : Exec.Inputs} {d : Json} {t : Exec.Transcript}
    {out : Model.OutCL 𝔽 Dig (List UInt8)} (he : ∀ r < 2, LaterEntriesAt sch I d t out r)
    (hcl : FlockDeploy.KS.LaterClashL sch out) : LaterEntriesClash I d t :=
  (laterEntriesClash_iff_K_id _ _ _).2 <|
    laterEntriesClashK_of id (fun r hr => (laterEntriesAt_iff_K_id _ _ _ _ _ _).1 (he r hr)) hcl

open DecodeZL Integrate in
/-- **At the deployed plain decoders, the reps' recorded caps and later openings are the deployed reps'**, at one
table: where `verify` at `keep` accepts a transcript `t` recording `o` at the setup at the draw `dj S` (`hrl`), two
pendings the reps' decoded tapes play to (`hp`) record, in the output at `o`'s cap and openings, each rep's proof's
recursive caps, and open at each level `ℓ ≥ 1` the rep's `plainLevelColsOf` entries at the setup's table, its
`fast100` schedule and `t`'s proof file (`accepts_opensLDK`). -/
theorem laterEntries_wrK_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I (dj S) o t)
    (hacc : AcceptsAtDK keep I (some (dj S)) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch I (dj S) t
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs)) r := by
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hprf, hrec⟩ := recordsLive_wrK1_of keep I (dj S) o hrl hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAt_of_acceptsDK keep ht hc ho hh hc1 dj x₀ hacc hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_repsOneDK keep hrr hl hc1 hacc
  rw [HiddenExec.setupOne_public hh] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨-, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupH_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle = MerkleScheme.hm96Sha512 := setupH_hm96 hl hst
  obtain ⟨pend, hreps, -⟩ := hk hwf hr hm
  have hn : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL q pts).length = 2 * st.regions.size := by
    intro q pts
    rw [extraClaims_length]
    simp [stmtOf, regionsOf]
  obtain ⟨s₀, p₀, -, hs₀, hp₀, hpl₀, -⟩ := hreps 0
  obtain ⟨s₁, p₁, -, hs₁, hp₁, hpl₁, -⟩ := hreps 1
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opensLDK keep hacc
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨sch', hf', hk'⟩ := hall 0 (by simp)
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  have hn' : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf #[st][0] hwf (regionsOf #[st][0])) PT_LOCAL q pts).length = 2 * st.regions.size :=
    hn
  have hent := hk' hwf hr hm p₀ p₁ s₀ s₁ (pend 0) (pend 1) (capOfRoot o.rootB[0]!)
    (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[t.proofs[0]!, t.proofs[1]!]))
    hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ (by rw [hn']; rfl)
  rw [hn'] at hent
  rw [hmP, hpts] at hreps
  have hS' : (tablesAt (Reg := Reg) ht dj x₀ S R).S = stmtOf st hwf (regionsOf st) := hTS
  have hpl : (tablesAt (Reg := Reg) ht dj x₀ S R).ptLocal = PT_LOCAL := tableAt_ptLocal ht dj x₀ _
  have hsch : (tablesAt (Reg := Reg) ht dj x₀ S R).sch = schedOf sch := by
    have h1 : Accounting.fast100 st.m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := (tablesAt (Reg := Reg) ht dj x₀ S R).hsch
    rw [hS'] at h2
    exact Option.some.inj (h2.symm.trans h1)
  have hrep : ∀ r : Fin 2, PlaysCoin.RepRun (repsOf (D := Dig) execArith (tablesAt (Reg := Reg) ht dj x₀ S R)
      (oneOf o).pts r) (PlaysCoin.repSpine Dig (tablesAt (Reg := Reg) ht dj x₀ S R) (oneOf o).pts)
      (decAt (tablesAt ht dj x₀) wOP wLIG S R) (pick ((oneOf o).rep0, (oneOf o).rep1) r) [] (pend r) := by
    intro r
    obtain ⟨s, p, Tv, hs, hp, hplay, hlen, hret⟩ := hreps r
    obtain ⟨s', hs', hrs⟩ := hstreams r
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    exact PlaysCoin.repRun_of_erase hrs <| repRun_at op_carries lig_carries decodeProof_sizes _ st hwf sch hf
      (by rw [hms]; rfl) hS' hsch hpl
      (oneOf o).pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decodeWithK_roundsWF keep With.std hdec s'
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  have hpe : ∀ r : Fin 2, Tables.sel pend' r = pend r := fun r =>
    playsT_det (hp r) (by
      have := PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig (tablesAt (Reg := Reg) ht dj x₀ S R)
        (oneOf o).pts r) (hrep r)
      fin_cases r <;> exact this)
  have hpe' : pend' = (pend 0, pend 1) := Prod.ext (hpe 0) (hpe 1)
  subst hpe'
  have hprf' : t.proofs = o.proofs := hprf
  have hopn : opnAt (tablesAt (Reg := Reg) ht dj x₀) I dj S R o.proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[t.proofs[0]!, t.proofs[1]!] := by
    rw [hprf']
    unfold opnAt
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nX
    rw [extraClaims_length, hS']
    simp [stmtOf, regionsOf]
  rw [hopn, hsch]
  rw [hn] at hp₀ hp₁
  intro r hr2
  refine ⟨spec, st, sess, sch, hsu, hdec, hf, rfl, ?_⟩
  rcases r with _ | _ | r
  · exact ⟨s₀, p₀, hs₀, hp₀, hent.1, fun ℓ h1 => (hent.2.2 ℓ h1).1⟩
  · exact ⟨s₁, p₁, hs₁, hp₁, hent.2.1, fun ℓ h1 => (hent.2.2 ℓ h1).2⟩
  · omega

open DecodeZL Integrate in
/-- **At the deployed plain decoders, the reps' recorded caps and later openings are the deployed reps'**:
`laterEntries_wrK_of` at `keep := id` and what `wrPD` writes for `o`. -/
theorem laterEntries_wrPD {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAt (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch I (dj S) (wrPD I (dj S) o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs)) r :=
  fun r hr => (laterEntriesAt_iff_K_id _ _ _ _ _ _).2 <| laterEntries_wrK_of id ht hc ho dj x₀ hT hS R o
    (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD I (dj S) o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp
    r hr

open DecodeZL Integrate in
/-- **A self-clash past level 0 at the deployed plain decoders is a clash of one rep's level-`ℓ` entries**, at one
table: where `verify` accepts what `wrPD` writes for `o` at the setup at the draw `dj S`, two pendings the reps' decoded
tapes play to (`hp`), whose output at `o`'s cap and openings has one rep's two level-`ℓ` openings of one position to
different columns (`hcl`), give two of that rep's `plainLevelColsOf` entries at the setup's table, its `fast100`
schedule and `wrPD`'s proof file at one position with different columns (`laterEntries_wrPD`). -/
theorem laterClash_wrPD {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : FlockDeploy.KS.LaterClashL (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch (tableOutL H512 enc512 leaf512
      (capOfRoot o.rootB[0]!) pend' (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs))) :
    LaterEntriesClash I (dj S) (wrPD I (dj S) o) :=
  laterEntriesClash_of (laterEntries_wrPD ht hc ho dj x₀ hT hS R o hacc pend' hp) hcl

open DecodeZL HiddenExec HiddenDec in
/-- **At the deployed plain decoders at hidden outputs, at a leaf form `F`, the reps' recorded caps and later openings
are the deployed reps'**: `laterEntries_wrK_of` at the hidden tables and their openings `opnAtH`. -/
theorem laterEntries_wrKHLf_of (keep : Bool → Bool) (F : LeafForm)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : F.TagsOk I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I (dj S) o t)
    (hacc : AcceptsAtDK keep I (some (dj S)) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep (tablesAtH (Reg := Reg) dj x₀ S R).sch I (dj S) t
      (tableOutL H512 enc512 F.lf (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)) r := by
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hprf, hrec⟩ := recordsLive_wrK1_of keep I (dj S) o hrl hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAtH_of_acceptsDK keep ht hc hh hc1 dj x₀ hacc hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ :=
    accepts_repsOneDLfK keep F.lf hrr hc1 (F.verdicts keep hh hl hacc) hacc
  rw [setupOne_hidden hh] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨-, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupHidden_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle.digestLen = 64 := F.digestLen hst hl
  obtain ⟨pend, hreps, -⟩ := hk hwf hr hm
  have hn : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL q pts).length = 2 * st.regions.size := by
    intro q pts
    rw [extraClaims_length]
    simp [stmtOf, regionsOf]
  obtain ⟨s₀, p₀, -, hs₀, hp₀, hpl₀, -⟩ := hreps 0
  obtain ⟨s₁, p₁, -, hs₁, hp₁, hpl₁, -⟩ := hreps 1
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opensLDLfK keep F.lf hacc
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨sch', hf', hk'⟩ := hall 0 (by simp)
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  have hn' : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf #[st][0] hwf (regionsOf #[st][0])) PT_LOCAL q pts).length = 2 * st.regions.size :=
    hn
  have hent := hk' hwf hr hm p₀ p₁ s₀ s₁ (pend 0) (pend 1) (capOfRoot o.rootB[0]!)
    (tableOutL H512 enc512 F.lf (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[t.proofs[0]!, t.proofs[1]!]))
    hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ (by rw [hn']; rfl)
  rw [hn'] at hent
  rw [hmP, hpts] at hreps
  have hS' : (tablesAtH (Reg := Reg) dj x₀ S R).S = stmtOf st hwf (regionsOf st) := hTS
  have hpl : (tablesAtH (Reg := Reg) dj x₀ S R).ptLocal = PT_LOCAL := tableAtH_ptLocal dj x₀ _
  have hsch : (tablesAtH (Reg := Reg) dj x₀ S R).sch = schedOf sch := by
    have h1 : Accounting.fast100 st.m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := (tablesAtH (Reg := Reg) dj x₀ S R).hsch
    rw [hS'] at h2
    exact Option.some.inj (h2.symm.trans h1)
  have hrep : ∀ r : Fin 2, PlaysCoin.RepRun (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R)
      (oneOf o).pts r) (PlaysCoin.repSpine Dig (tablesAtH (Reg := Reg) dj x₀ S R) (oneOf o).pts)
      (decAt (tablesAtH dj x₀) wOP wLIG S R) (pick ((oneOf o).rep0, (oneOf o).rep1) r) [] (pend r) := by
    intro r
    obtain ⟨s, p, Tv, hs, hp, hplay, hlen, hret⟩ := hreps r
    obtain ⟨s', hs', hrs⟩ := hstreams r
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    exact PlaysCoin.repRun_of_erase hrs <| repRun_at op_carries lig_carries decodeProof_sizes _ st hwf sch hf
      hms hS' hsch hpl
      (oneOf o).pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decodeWithK_roundsWF keep With.std hdec s'
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  have hpe : ∀ r : Fin 2, Tables.sel pend' r = pend r := fun r =>
    playsT_det (hp r) (by
      have := PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig (tablesAtH (Reg := Reg) dj x₀ S R)
        (oneOf o).pts r) (hrep r)
      fin_cases r <;> exact this)
  have hpe' : pend' = (pend 0, pend 1) := Prod.ext (hpe 0) (hpe 1)
  subst hpe'
  have hprf' : t.proofs = o.proofs := hprf
  have hopn : opnAtH (tablesAtH (Reg := Reg) dj x₀) I dj S R o.proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[t.proofs[0]!, t.proofs[1]!] := by
    rw [hprf']
    unfold opnAtH
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nX
    rw [extraClaims_length, hS']
    simp [stmtOf, regionsOf]
  rw [hopn, hsch]
  rw [hn] at hp₀ hp₁
  intro r hr2
  refine ⟨spec, st, sess, sch, hsu, hdec, hf, rfl, ?_⟩
  rcases r with _ | _ | r
  · exact ⟨s₀, p₀, hs₀, hp₀, hent.1, fun ℓ h1 => (hent.2.2 ℓ h1).1⟩
  · exact ⟨s₁, p₁, hs₁, hp₁, hent.2.1, fun ℓ h1 => (hent.2.2 ℓ h1).2⟩
  · omega

open DecodeZL HiddenExec HiddenDec in
/-- **At the deployed plain decoders at hidden outputs, the reps' recorded caps and later openings are the deployed
reps'**: `laterEntries_wrK_of` at the hidden tables and their openings `opnAtH`; `laterEntries_wrKHLf_of` at `hm96`. -/
theorem laterEntries_wrKH_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I (dj S) o t)
    (hacc : AcceptsAtDK keep I (some (dj S)) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep (tablesAtH (Reg := Reg) dj x₀ S R).sch I (dj S) t
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)) r :=
  laterEntries_wrKHLf_of keep LeafForm.hm96 ht hc dj x₀ hT hS R o hrl hacc pend' hp

open DecodeZL HiddenExec HiddenDec in
/-- **At the deployed plain decoders at hidden outputs, the reps' recorded caps and later openings are the deployed
reps'**: `laterEntries_wrKH_of` at `keep := id` and what `wrPD` writes for `o`. -/
theorem laterEntries_wrPDH {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAt (tablesAtH (Reg := Reg) dj x₀ S R).sch I (dj S) (wrPD I (dj S) o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)) r :=
  fun r hr => (laterEntriesAt_iff_K_id _ _ _ _ _ _).2 <| laterEntries_wrKH_of id ht hc dj x₀ hT hS R o
    (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD I (dj S) o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp
    r hr

open DecodeZL HiddenExec HiddenDec in
/-- **A self-clash past level 0 at the deployed plain decoders at hidden outputs is a clash of one rep's level-`ℓ`
entries**: `laterClash_wrPD` at the hidden tables and their openings `opnAtH` (`laterEntries_wrPDH`). -/
theorem laterClash_wrPDH {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : FlockDeploy.KS.LaterClashL (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leaf512
      (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs))) :
    LaterEntriesClash I (dj S) (wrPD I (dj S) o) :=
  laterEntriesClash_of (laterEntries_wrPDH ht hc dj x₀ hT hS R o hacc pend' hp) hcl

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw**: `laterEntries_wrK_of` at the
statement's draw `d` and no drawn unit, as `plain_clash₀AtK_of` is `clash₀_wrK_of` there. -/
theorem plain_laterEntriesAtK_of (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep ins.read.I d o t) (hacc : AcceptsAtDK keep ins.read.I (some d) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d t
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r := by
  obtain ⟨ht, hc, ho, hT, hst, rfl, -, -⟩ := plain_setupH hs hin hset h hh hd
  obtain ⟨sch, hf⟩ : ∃ sch, Flock.fast100 st.m = some sch := by
    have e := fast100_schedOf st.m
    rw [hsch] at e
    cases hf : Flock.fast100 st.m with
    | none => rw [hf] at e; cases e
    | some sch => exact ⟨sch, rfl⟩
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x₀, hx₀⟩ := Integrate.drawSetup_of_setupH (S := (∅ : Finset (Fin 0))) (mPts := st.spec.mPts) ht hc ho hst
    hf hD (fun u hu => absurd hu (Finset.notMem_empty u)) rfl
  have e : Integrate.tablesAt (Reg := Unit) ht (fun _ : Finset (Fin 0) => d) x₀ =
      fun _ _ => plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay := by
    funext S R
    obtain rfl : S = ∅ := Finset.eq_empty_of_isEmpty S
    exact tableAt_plainTab ht (fun _ : Finset (Fin 0) => d) (spec := st.spec) (S := ∅) x₀ x₀ hx₀ L m k0 l₀ l₁ rest ext
      hsch hm hlay
  have hS : DecodeZL.DrawOk ins.read.I (fun _ : Finset (Fin 0) => d) st.spec.mPts ∅ :=
    ⟨fun _ _ u hu => absurd hu (Finset.notMem_empty u), fun st' hst' => by
      obtain rfl := Except.ok.inj (hst.symm.trans hst')
      rfl⟩
  have he := laterEntries_wrK_of keep ht hc ho (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () o hrl hacc pend'
    (by rw [e]; exact hp)
  rw [e] at he
  exact he

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw, at hidden outputs under a leaf form
`F`**: `laterEntries_wrKHLf_of` at the statement's draw `d` and no drawn unit. -/
theorem plain_laterEntriesHLfAtK_of (keep : Bool → Bool) (F : AtDraw.LeafForm)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = F.tag) {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep ins.read.I d o t) (hacc : AcceptsAtDK keep ins.read.I (some d) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d t
      (tableOutL H512 enc512 F.lf (capOfRoot o.rootB[0]!) pend'
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r := by
  obtain ⟨ht, hc, hT, hst, rfl, -, -⟩ := plain_setupHiddenT hs hin hset h hh hl hd
  obtain ⟨sch, hf⟩ : ∃ sch, Flock.fast100 st.m = some sch := by
    have e := fast100_schedOf st.m
    rw [hsch] at e
    cases hf : Flock.fast100 st.m with
    | none => rw [hf] at e; cases e
    | some sch => exact ⟨sch, rfl⟩
  obtain ⟨pop, -, hdr⟩ := HiddenExec.setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
  obtain ⟨x₀, hx₀⟩ := HiddenExec.drawSetupH_of_setupHiddenI (S := (∅ : Finset (Fin 0))) (mPts := st.spec.mPts) ht hc
    hst hf hD (fun u hu => absurd hu (Finset.notMem_empty u)) rfl
  have e : HiddenExec.tablesAtH (Reg := Unit) (fun _ : Finset (Fin 0) => d) x₀ =
      fun _ _ => plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay := by
    funext S R
    obtain rfl : S = ∅ := Finset.eq_empty_of_isEmpty S
    exact tableAtH_plainTab ht (fun _ : Finset (Fin 0) => d) (spec := st.spec) (S := ∅) x₀ x₀ hx₀ L m k0 l₀ l₁
      rest ext hsch hm hlay
  have hS : HiddenDec.DrawOkH ins.read.I (fun _ : Finset (Fin 0) => d) st.spec.mPts ∅ :=
    ⟨fun _ _ u hu => absurd hu (Finset.notMem_empty u), fun st' hst' => by
      obtain rfl := Except.ok.inj (hst.symm.trans hst')
      rfl⟩
  have he := laterEntries_wrKHLf_of keep F ht hc (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () o hrl hacc
    pend' (by rw [e]; exact hp)
  rw [e] at he
  exact he

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw, at hidden outputs under hm96 leaves**:
`laterEntries_wrKH_of` at the statement's draw `d` and no drawn unit; `plain_laterEntriesHLfAtK_of` at `hm96`. -/
theorem plain_laterEntriesHAtK_of (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep ins.read.I d o t) (hacc : AcceptsAtDK keep ins.read.I (some d) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d t
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r :=
  plain_laterEntriesHLfAtK_of keep AtDraw.LeafForm.hm96 hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hrl
    hacc pend' hp

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw**: `plain_laterEntriesAtK_of` at
`keep := id` and what `wrPD` writes for `o`. -/
theorem plain_laterEntriesAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAt ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPD ins.read.I d o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r :=
  fun r hr => (laterEntriesAt_iff_K_id _ _ _ _ _ _).2 <| plain_laterEntriesAtK_of id hs hin hset h hh hd L m k0 l₀ l₁
    rest ext hsch hm hlay o (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD _ d o))
    ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp r hr

/-- **The deployed plain decoders' self-clash past level 0 at `plainTab`, at the draw**: `laterEntriesClash_of` at
`plain_laterEntriesAt`. -/
theorem plain_laterClashAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    LaterEntriesClash ins.read.I d (wrPD ins.read.I d o) :=
  laterEntriesClash_of (plain_laterEntriesAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay o hacc pend' hp)
    hcl

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw, at hidden outputs under hm96 leaves**:
`plain_laterEntriesHAtK_of` at `keep := id` and what `wrPD` writes for `o`. -/
theorem plain_laterEntriesHAt {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAt ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPD ins.read.I d o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r :=
  fun r hr => (laterEntriesAt_iff_K_id _ _ _ _ _ _).2 <| plain_laterEntriesHAtK_of id hs hin hset h hh hl hd L m k0 l₀
    l₁ rest ext hsch hm hlay o (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD _ d o))
    ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp r hr

/-- **The deployed plain decoders' self-clash past level 0 at `plainTab`, at the draw, at hidden outputs under hm96
leaves**:
`laterEntriesClash_of` at `plain_laterEntriesHAt`. -/
theorem plain_laterClashHAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    LaterEntriesClash ins.read.I d (wrPD ins.read.I d o) :=
  laterEntriesClash_of
    (plain_laterEntriesHAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hacc pend' hp) hcl

/-- **An accepted plain run's reps' level-`ℓ` rows**: `rawCols_plain` past level 0, so `plainRawLevelCols` at rep
`k < 2` is `plainLevelColsOf` at `shapeOf` with `2 · |regions|` extra claims, on the live session's `k`-th rep stream
and proof file. -/
theorem rawLevelCols_plain (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {st : Stmt} {sch : Flock.Schedule} (hf : lastFinal seen = some f)
    (hset : setupOf false f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hst : sts[0]? = some st)
    (hsch : Flock.fast100 st.m = some sch) (hp2 : spec.points = 2) {k : ℕ} (hk : k < 2) (ℓ : ℕ) :
    plainRawLevelCols seen k ℓ =
      plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD k #[])
        (f.proofs.getD k ByteArray.empty) ℓ := by
  obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hst
  have h0' : sts[0]! = sts[0] := getElem!_pos sts 0 h0
  obtain ⟨hsz, hall⟩ := rep_plain hs hv hlv
  have hk : 0 < lv.streams.size := by omega
  obtain ⟨sess, -, -, -, hpts, hpn, -⟩ := hall 0 hk
  obtain ⟨extras, -, -, hex, -⟩ := queryRound_plain hs hv hlv hk (tab := Setup.ofCircuit sts[0])
    (by rw [Nat.zero_div, h0'])
  have hx : extras.size = 2 * sts[0].regions.size := by
    rw [extraClaims_size (st := sts[0]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit sts[0]).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit sts[0]).m = sts[0].m from rfl, hsch]
    rfl
  have hrun : runOf false seen 0 = some
      { m := (Setup.ofCircuit sts[0]).m, kLog := (Setup.ofCircuit sts[0]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[0]).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit sts[0]),
        reps := [(lv.streams.getD 0 #[], f.proofs.getD 0 ByteArray.empty),
                 (lv.streams.getD 1 #[], f.proofs.getD 1 ByteArray.empty)] } := by
    unfold runOf
    simp [hf, hset, hst, hlv, hex, Except.toOption]
  unfold plainRawLevelCols
  rw [hrun]
  rcases k with _ | _ | k
  · simp only [Claims.Flock.ZkRun.plainShape, FlockDeploy.plainProofShape_eq hsched, hx, List.getElem?_cons_zero]
  · simp only [Claims.Flock.ZkRun.plainShape, FlockDeploy.plainProofShape_eq hsched, hx, List.getElem?_cons_succ,
      List.getElem?_cons_zero]
  · omega

/-- **A self-clash past level 0 at the concrete simulator's outcome is a raw clash of `P`'s accepting run**: at
openings `opn` whose self-clash past level 0 at an outcome `wrPD` writes an accepted transcript for is a
`LaterEntriesClash` there (`hE`, as `plain_laterClashAt` and `plain_laterClashHAt` give), where `P`'s run accepts at
`plainΦ x`, the simulator's outcome is the run's (`plainSimRun`), `wrPD` records the run's rep streams and proof files,
those entries are the run's level-`ℓ` rows (`rawLevelCols_plain`), and two of them at one position that differ clash in
the run's log (`plain_levelClash`), so the run has a raw clash (`PlainClashQ` at `LaterClashL`). -/
theorem plainClashQL {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        LaterClashL sch (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        LaterEntriesClash ins.read.I d (wrPD ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQ P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (LaterClashL sch) := by
  intro C u r' hq hacc
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨a, ⟨hV, pend, hpl, rfl⟩, hcl⟩ := hq
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hcl
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := acc_decide hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRun P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hV hpl hcl
  obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', k, hk, s₀, hs₀, ℓ, q, a₁, b₁, ha, hb, hab⟩ :=
    hE _ hV.1 (pend 0) (hpl 0) hcl
  have hacc' := run_acc_next sha512 P (V false) x ω _ hacc
  replace hacc' : ((next false x ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
        (Claims.coinSeq (plainΦ x spec st (u, r'))))).eval sha512).1 = .acc := hacc'
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc'
  have hpi := publicInputs_none_of_checks hc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hsu₀⟩ := setupOf_some (act_decide hact)
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  obtain ⟨hpr, hall⟩ := recordsLiveAtD_wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, hrs⟩ := hall spec #[st'] sess hsu hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨k, hk⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decision_runs hsu₀ hrun
  have hv' : verifyWithM (hashesIn ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).vLog.map
      Prod.fst)) spec (tabsOf #[st']) f.record f.proofs = .ok () := verdict_within hvr
  have hcols := rawLevelCols_plain (st := st') _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hf'
    (HiddenExec.setupTables_pointsAny hsetA).1 hk ℓ
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (k : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec) →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro k y hy j
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hcl' : ∀ {q : ℕ} {v : List 𝔽}, (q, v) ∈ plainLevelColsOf (shapeOf (Setup.ofCircuit st') sch'
        (2 * st'.regions.size)) y₀.rounds (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[k]! ℓ →
      (q, v) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen k ℓ := by
    intro q v hqv
    rw [hpr', plainLevelColsOf_congr _ _ _ (hco k y₀ hr₀), ← getD_empty, ← hcols] at hqv
    exact hqv
  exact Or.inr (Or.inl (plain_levelClash hacc (hcl' ha) (hcl' hb) hab))

/-- **The live self-clash's clash past level 0 holds at every statement**: `plainClashQL` at `plain_laterClashAt` and
`plain_laterClashHAt`. -/
theorem plainSimLaterClashAt (x : Claims.Msg) : PlainSimLaterClashAt x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω _ _ P ω
  exact ⟨fun hh => plainClashQL hs hin hset h hd _ _ _
      (plain_laterClashAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω,
    fun hh hl => plainClashQL hs hin hset h hd _ _ _
      (plain_laterClashHAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω⟩

end FlockDeploy.KS
