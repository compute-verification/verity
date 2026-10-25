import Proofs.Flock.Claims.KS.PlainDrawDecK
import Proofs.Flock.Claims.KS.PlainClashLater

/-!
# The live self-clash past level 0, at `VK keep`

`PlainClashLater`'s draw-layer lemmas at `AcceptsAtDK keep` and `wrPDK keep`, by its proofs on `PlainDrawAccK`'s
acceptance lemmas, each session the one the record decodes to through `keep`. The entries are `plainLevelColsOf` at the
decoded rep stream's rounds, which reads their coins only, so the rounds' digests, unconstrained at `keep`, never enter:

* `accepts_reqsDK`, `accepts_opensLDK`: `accepts_reqsD` and `accepts_opensLD`.
* `LaterEntriesAtK`, `LaterEntriesClashK`, with `laterEntriesAt_iff_K_id` and `laterEntriesClash_iff_K_id` at
  `keep := id`, and `laterEntriesClashK_of`: `LaterEntriesAt`, `LaterEntriesClash` and `laterEntriesClash_of` over
  `Record.decodeWithK keep`.
* `laterEntries_wrPDK`, `laterClash_wrPDK`, `laterEntries_wrPDHK`, `laterClash_wrPDHK`: `laterEntries_wrPD`,
  `laterClash_wrPD`, `laterEntries_wrPDH`, `laterClash_wrPDH` at `wrPDK keep`.
* `plain_laterEntriesAtK`, `plain_laterClashAtK`, `plain_laterEntriesHAtK`, `plain_laterClashHAtK`: those at the
  statement's draw, at `plainTab`.

The run-level half (`rawLevelCols_plain`, `plainClashQL`, `plainSimLaterClashAt`) reads `P`'s run at the verifier, and
is not restated here.
-/

namespace FlockSoundness.Discharge.AtDraw

open Claims.Bridge.PlainDraw (setupAtD AcceptsAtDK named_of_setupAtD)
open Flock Refine Model Game FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **An accepted rep at `dr` through `keep` requests every later level** (`accepts_reqsD`). -/
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

/-- **At flock-verify's acceptance at `dr` through `keep`, each table's recorded caps and level-`ℓ` openings are its
reps' deployed ones**, at every level `ℓ ≥ 1` (`accepts_opensLD`). -/
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

end FlockSoundness.Discharge.AtDraw

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

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

/-- **A self-clash past level 0 in an output whose reps' later openings are the deployed reps' is a clash of one rep's
level-`ℓ` entries**, through `keep` (`laterEntriesClash_of`). -/
theorem laterEntriesClashK_of (keep : Bool → Bool) {sch : Accounting.Schedule} {I : Exec.Inputs} {d : Json}
    {t : Exec.Transcript} {out : Model.OutCL 𝔽 Dig (List UInt8)} (he : ∀ r < 2, LaterEntriesAtK keep sch I d t out r)
    (hcl : FlockDeploy.KS.LaterClashL sch out) : LaterEntriesClashK keep I d t := by
  obtain ⟨r, hr2, ℓ, h1, -, -, u, hu, v, hv, huv, -, -, hne⟩ := hcl
  obtain ⟨spec, st, sess, sch', hsu, hdec, hf, -, s, -, hs, -, -, hent⟩ := he r hr2
  have hmem : ∀ z ∈ slotOpensL out r ℓ (lvOf sch ℓ).queries, (z.1, z.2.1) ∈
      FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch' (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ :=
    fun z hz => hent ℓ h1 ▸ List.mem_map_of_mem hz
  exact ⟨spec, st, sess, sch', hsu, hdec, hf, r, hr2, s, hs, ℓ, u.1, u.2.1, v.2.1, hmem u hu, huv ▸ hmem v hv, hne⟩

open DecodeZL Integrate in
/-- **At the deployed plain decoders, the reps' recorded caps and later openings are the deployed reps'**, at one
table, where `verifyWithMK keep` accepts what `wrPDK keep` writes for `o` at the setup at the draw `dj S`
(`laterEntries_wrPD`). -/
theorem laterEntries_wrPDK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch I (dj S) (wrPDK keep I (dj S) o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs)) r := by
  change AcceptsAtDK keep I (some (dj S)) (wrPDK keep (J := 1) I (dj S) o) at hacc
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hprf, hrec⟩ := recordsLive_wrPDK1 keep I (dj S) o hacc
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
        #[(wrPDK keep (J := 1) I (dj S) o).proofs[0]!, (wrPDK keep (J := 1) I (dj S) o).proofs[1]!]))
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
  have hprf' : (wrPDK keep (J := 1) I (dj S) o).proofs = o.proofs := hprf
  have hopn : opnAt (tablesAt (Reg := Reg) ht dj x₀) I dj S R o.proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPDK keep (J := 1) I (dj S) o).proofs[0]!, (wrPDK keep (J := 1) I (dj S) o).proofs[1]!] := by
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
/-- **A self-clash past level 0 at the deployed plain decoders is a clash of one rep's level-`ℓ` entries**, at
`wrPDK keep` (`laterClash_wrPD`). -/
theorem laterClash_wrPDK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : FlockDeploy.KS.LaterClashL (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch (tableOutL H512 enc512 leaf512
      (capOfRoot o.rootB[0]!) pend' (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs))) :
    LaterEntriesClashK keep I (dj S) (wrPDK keep I (dj S) o) :=
  laterEntriesClashK_of keep (laterEntries_wrPDK keep ht hc ho dj x₀ hT hS R o hacc pend' hp) hcl

open DecodeZL HiddenExec HiddenDec in
/-- **At the deployed plain decoders at hidden outputs, the reps' recorded caps and later openings are the deployed
reps'**, at `wrPDK keep` (`laterEntries_wrPDH`). -/
theorem laterEntries_wrPDHK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep (tablesAtH (Reg := Reg) dj x₀ S R).sch I (dj S) (wrPDK keep I (dj S) o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)) r := by
  change AcceptsAtDK keep I (some (dj S)) (wrPDK keep (J := 1) I (dj S) o) at hacc
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hprf, hrec⟩ := recordsLive_wrPDK1 keep I (dj S) o hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAtH_of_acceptsDK keep ht hc hh hc1 dj x₀ hacc hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_repsOneDK keep hrr hl hc1 hacc
  rw [setupOne_hidden hh] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨-, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupHidden_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle = MerkleScheme.hm96Sha512 := setupHidden_hm96 hl hst
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
        #[(wrPDK keep (J := 1) I (dj S) o).proofs[0]!, (wrPDK keep (J := 1) I (dj S) o).proofs[1]!]))
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
      (by rw [hms]; rfl) hS' hsch hpl
      (oneOf o).pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decodeWithK_roundsWF keep With.std hdec s'
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  have hpe : ∀ r : Fin 2, Tables.sel pend' r = pend r := fun r =>
    playsT_det (hp r) (by
      have := PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig (tablesAtH (Reg := Reg) dj x₀ S R)
        (oneOf o).pts r) (hrep r)
      fin_cases r <;> exact this)
  have hpe' : pend' = (pend 0, pend 1) := Prod.ext (hpe 0) (hpe 1)
  subst hpe'
  have hprf' : (wrPDK keep (J := 1) I (dj S) o).proofs = o.proofs := hprf
  have hopn : opnAtH (tablesAtH (Reg := Reg) dj x₀) I dj S R o.proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPDK keep (J := 1) I (dj S) o).proofs[0]!, (wrPDK keep (J := 1) I (dj S) o).proofs[1]!] := by
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
/-- **A self-clash past level 0 at the deployed plain decoders at hidden outputs is a clash of one rep's level-`ℓ`
entries**, at `wrPDK keep` (`laterClash_wrPDH`). -/
theorem laterClash_wrPDHK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : FlockDeploy.KS.LaterClashL (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leaf512
      (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs))) :
    LaterEntriesClashK keep I (dj S) (wrPDK keep I (dj S) o) :=
  laterEntriesClashK_of keep (laterEntries_wrPDHK keep ht hc dj x₀ hT hS R o hacc pend' hp) hcl

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw, at `wrPDK keep`**: `plain_laterEntriesAt`
through `keep`, from `laterEntries_wrPDK`. -/
theorem plain_laterEntriesAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPDK keep ins.read.I d o)
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
  have he := laterEntries_wrPDK keep ht hc ho (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () o hacc pend'
    (by rw [e]; exact hp)
  rw [e] at he
  exact he

/-- **The deployed plain decoders' self-clash past level 0 at `plainTab`, at the draw, at `wrPDK keep`**:
`laterEntriesClashK_of` at `plain_laterEntriesAtK`. -/
theorem plain_laterClashAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    LaterEntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o) :=
  laterEntriesClashK_of keep
    (plain_laterEntriesAtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay o hacc pend' hp) hcl

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw, at hidden outputs under hm96 leaves, at
`wrPDK keep`**: `plain_laterEntriesHAt` through `keep`, from `laterEntries_wrPDHK`. -/
theorem plain_laterEntriesHAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPDK keep ins.read.I d o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r := by
  obtain ⟨ht, hc, hT, hst, rfl, -, -⟩ := plain_setupHidden hs hin hset h hh hl hd
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
  have he := laterEntries_wrPDHK keep ht hc (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () o hacc pend'
    (by rw [e]; exact hp)
  rw [e] at he
  exact he

/-- **The deployed plain decoders' self-clash past level 0 at `plainTab`, at the draw, at hidden outputs under hm96
leaves, at `wrPDK keep`**: `laterEntriesClashK_of` at `plain_laterEntriesHAtK`. -/
theorem plain_laterClashHAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    LaterEntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o) :=
  laterEntriesClashK_of keep
    (plain_laterEntriesHAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hacc pend' hp) hcl

end FlockDeploy.KS
