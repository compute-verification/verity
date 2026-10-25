import Proofs.Flock.Soundness.Discharge.PlainJ.Defs

/-!
# The decoder is right at a plain session of `J` tables

`decodesOnPJ_exec`: `DecodesOnJ` at `planPJ`, table by table, at every number of tables (`DecodeZLJ.decodesOnJ_exec`
reads `1 < count` and `planJ`'s fallback tables; here one table is the session's statement, `Composed.drawJ`). At an
outcome flock-verify accepts, at table `j`:

* `RecordsLiveJ` gives the record's draw `dj S`, the proof files, the link points, and table `j`'s root and streams.
* `setupOf_tablePJ` sets table `j` up with setupH under `tagsAt` at `drawJ` (`setupTables_ok` at one table,
  `Tables.setupTables_parts` at several); `DrawOkPJ` gives its proved units and link points, so a `DrawSetup` exists, and
  `planPJ`'s table `j` is its table (`tablePJ_of`), `stmtOf` of setupH's statement (`table_stmtOf`).
* `accepts_reps_at` gives each rep's run and rounds and the salted verdict; `repRun_at` makes each run a `RepRun` for
  `decJ`, and `playsT_of_repRun` a run on its tape.
-/

namespace FlockSoundness.Discharge.PlainJ

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Tables FlockSoundness.Discharge.DecodeZL FlockSoundness.Discharge.Integrate
open FlockSoundness.Discharge.Composed (nTab nTab_pos drawJ partS TagsOkZJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson accepts_reps_at decJ nXT RecordsLiveJ)

local notation "𝔽" => FlockLevel3.GF128

variable {n mPts : ℕ} {Reg : Type}

/-- **flock-verify sets table `j` up under its tags at its draw**, without `--zk`: at the record's draw `d`, `setupTables`
at `I` has `nTab I` tables, table `j` setupH's statement under `tagsAt` at `drawJ` (`Composed.setupOf_tableJ` at the
plain inputs). -/
theorem setupOf_tablePJ {I : Inputs} {record : String} {d : Lean.Json} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hh : I.tags.hiddenOutputs = false) (hd : recordDraw record = some d) (hs : setupOf I record = .ok (spec, sts)) :
    sts.size = nTab I ∧ ∀ (j : ℕ) (hj : j < sts.size),
      Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables
        (some (drawJ I d j)) I.partition I.program = .ok sts[j] := by
  by_cases hc : I.count ≤ 1
  · rcases setupTables_ok hh hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
    · refine ⟨by simp [nTab, hc], fun j hj => ?_⟩
      have : j = 0 := by simpa using hj
      subst this
      have e1 : tagsAt I.tags I.count 0 = I.tags := by simp [tagsAt, hc]
      have e2 : drawJ I d 0 = d := by simp [drawJ, hc]
      rw [e1, e2, ← hd]
      exact hst
    · exact absurd hc' (by omega)
  · have hc1 : 1 < I.count := by omega
    obtain ⟨-, -, population, units, hD1, -, -, -, hsz, hall⟩ := Tables.setupTables_parts hh hc1 hs
    obtain ⟨u, hu, -, hpop, hun⟩ := hD1 _ hd
    refine ⟨by simp only [nTab, hc, ite_false]; exact hsz, fun j hj => ?_⟩
    have hpart : drawJ I d j = Tables.partDraw population units I.count j := by
      simp only [drawJ, hc, ite_false]
      unfold partJson
      rw [hu]
      dsimp only
      rw [hpop, hun]
    have e1 : tagsAt I.tags I.count j = { I.tags with identity := I.tags.tableIdentity I.count j } := by
      simp [tagsAt, hc]
    rw [e1, hpart]
    exact hall j hj

private theorem stmtOf_congr {a b : Flock.Stmt} (h : a = b) (ha : StmtWF a) (hb : StmtWF b) :
    stmtOf a ha (regionsOf a) = stmtOf b hb (regionsOf b) := by
  subst h
  rfl

variable {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (x₀ : DrawSetup (inputsPJ I 0) mPts S₀ d₀)

set_option maxHeartbeats 4000000 in
/-- **The decoder is right at a plain session of `J` tables** (R11b at `--plain-tables --session-tables J`): given the
tags, `wr` the challenger's record and the challenger's draws setting each table up at `mPts` link points, `DecodesOnJ`
holds at `planPJ` with the decoder `decJ` on each table's two streams, single coins as field elements, no final
messages, table `j`'s cap from its root `root_B[j]` (`capOfRoot`) and its openings `opnPJ`. -/
theorem decodesOnPJ_exec (L : Law n)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planPJ (Reg := Reg) ht dj x₀ S R).length mPts → Exec.Transcript)
    (hT : TagsOkZJ I) (hwr : RecordsLiveJ I dj (fun S R => (planPJ (Reg := Reg) ht dj x₀ S R).length) wr)
    (hdraw : ∀ ω, DrawOkPJ I dj mPts (L.draw ω)) :
    DecodesOnJ H512 enc512 leaf512 (planPJ ht dj x₀) I wr (opnPJ I (planPJ ht dj x₀) dj) (decJ (planPJ ht dj x₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  intro ω R j o hacc
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hd, hprf, hrec⟩ := hwr _ R o hacc
  obtain ⟨spec, sts, sess, hsu, hdec, hk⟩ := accepts_reps_at hh hrr hl hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tablePJ hh hd hsu
  have hjn : j.val < nTab I := planPJ_length ht dj x₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hsu hdec
  obtain ⟨sch, hf, hk⟩ := hk j.val hj
  have hmP : sts[j.val].mPts = mPts := (hdraw ω ⟨j.val, hjn⟩).2 _ hst
  obtain ⟨hwf, hr⟩ := setupH_wf hst
  have hm : 13 ≤ sts[j.val].m := by have := exec_fast100_ge hf; omega
  have hms : sts[j.val].merkle = MerkleScheme.hm96Sha512 := setupH_tagsAt_hm96 hl hst
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := ZkLink.drawSetup_of_setupH_g (I := inputsPJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) (typedPJ ht j.val) hst hf hD
    ((hdraw ω ⟨j.val, hjn⟩).1 D hD) hmP
  obtain ⟨y, hyx, e⟩ := tablePJ_of ht dj x₀ (S := L.draw ω) (j := ⟨j.val, hjn⟩) x
  have hget : (planPJ (Reg := Reg) ht dj x₀ (L.draw ω) R).get j =
      (tablePJ ht dj x₀ (L.draw ω) ⟨j.val, hjn⟩).T :=
    List.get_ofFn _ j
  have hS : ((planPJ (Reg := Reg) ht dj x₀ (L.draw ω) R).get j).S = stmtOf sts[j.val] hwf (regionsOf sts[j.val]) := by
    rw [hget, e, table_stmtOf]
    exact stmtOf_congr (hyx.trans hx) _ _
  have hpl : ((planPJ (Reg := Reg) ht dj x₀ (L.draw ω) R).get j).ptLocal = PT_LOCAL := by rw [hget, e]; rfl
  have hsch : ((planPJ (Reg := Reg) ht dj x₀ (L.draw ω) R).get j).sch = schedOf sch := by
    have h1 : Accounting.fast100 sts[j.val].m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := ((planPJ (Reg := Reg) ht dj x₀ (L.draw ω) R).get j).hsch
    rw [hS] at h2
    exact Option.some.inj (h2.symm.trans h1)
  rw [hmP, hpts] at hk
  obtain ⟨pend, hreps, hv⟩ := hk hwf hr hm
  have hrep : ∀ r : Fin 2,
      PlaysCoin.RepRun (repsOf (D := Dig) execArith ((planPJ (Reg := Reg) ht dj x₀ (L.draw ω) R).get j) o.pts r)
        (PlaysCoin.repSpine Dig ((planPJ (Reg := Reg) ht dj x₀ (L.draw ω) R).get j) o.pts)
        (decJ (planPJ ht dj x₀) (L.draw ω) R j) (o.reps j r) [] (pend r) := by
    intro r
    obtain ⟨s, p, Tv, hs, hp, hplay, hlen, hret⟩ := hreps r
    obtain ⟨s', hs', hrs⟩ := (hrj j).2 r
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    rw [← hrs]
    exact repRun_at op_carries lig_carries decodeProof_sizes _ sts[j.val] hwf sch hf (by rw [hms]; rfl) hS hsch hpl
      o.pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s' (Array.mem_toList_iff.2
        (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  refine ⟨(pend 0, pend 1), PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig _ o.pts 0) (hrep 0),
    PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig _ o.pts 1) (hrep 1), ?_⟩
  have hopn : opnPJ I (planPJ ht dj x₀) dj (L.draw ω) R o.proofs j =
      opnOf (Setup.ofCircuit sts[j.val]) sch
        (Model.extraClaims (stmtOf sts[j.val] hwf (regionsOf sts[j.val])) PT_LOCAL mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnPJ
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nXT
    rw [extraClaims_length, extraClaims_length, hS]
  show (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[j.val]!) (pend 0, pend 1) _).accepted = true
  rw [hopn, ← (hrj j).1]
  exact hv

end FlockSoundness.Discharge.PlainJ
