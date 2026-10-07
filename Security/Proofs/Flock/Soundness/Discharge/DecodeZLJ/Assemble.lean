import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Reps

/-!
# The decoder is right at `J` tables

`decodesOnJ_exec`: `DecodesOnJ` at `planJ`, table by table. At an outcome flock-verify accepts, at table `j`:

* `RecordsLiveJ` gives the record's draw `dj S`, the proof files, the link points, and table `j`'s root and streams.
* `accepts_tables` sets table `j` up with setupH under `tagsAt` at `partDraw` of part `j`, which is `partJson (dj S)`;
  `DrawOkJ` gives its link points. So a `SetupJ` exists, and `planJ`'s table `j` is `stmtOf` of setupH's statement
  (`tableJ_of`, `model_eq_stmtOf`), with its `fast100` schedule and `PT_LOCAL`.
* `accepts_reps_at` gives each rep's run and rounds and the salted verdict; `repRun_at` (at `op_carries`,
  `lig_carries`) makes each run a `RepRun` for `decJ`, and `playsT_of_repRun` a run on its tape `repTapeJ`.
-/

namespace FlockSoundness.Discharge.DecodeZLJ

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Tables FlockSoundness.Discharge.DecodeZL FlockSoundness.Discharge.Integrate
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

theorem decodesOnJ_exec : DecodesOnJExec := by
  intro n I ht mPts T₀ dj Reg L wr hT hwr hdraw ω R j o hacc
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hd, hprf, hrec⟩ := hwr _ R o hacc
  obtain ⟨spec, sts, hsu, -, -, -, population, units, hD1, -, -, -, hsz, hall⟩ := accepts_tables hh hc1 hacc
  obtain ⟨u, hu, -, hpop, hun⟩ := hD1 _ hd
  have hpart : partJson (dj (L.draw ω)) I.count j.val = partDraw population units I.count j.val := by
    unfold partJson
    rw [hu]
    dsimp only
    rw [hpop, hun]
  have hjc : j.val < I.count := planJ_length ht T₀ dj (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjc
  have hst := hall j.val hj
  rw [← hpart] at hst
  obtain ⟨spec', sts', sess, hsu', hdec, hk⟩ := accepts_reps_at hh hrr hl hacc
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hsu hdec
  obtain ⟨sch, hf, hk⟩ := hk j.val hj
  have hmP : sts[j.val].mPts = mPts := hdraw ω j.val hjc _ hst
  obtain ⟨hwf, hr⟩ := setupH_wf hst
  have hm : 13 ≤ sts[j.val].m := by have := exec_fast100_ge hf; omega
  have hms : sts[j.val].merkle = MerkleScheme.hm96Sha512 := setupH_tagsAt_hm96 hl hst
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := model_sched hf
  obtain ⟨⟨st', h', m', k0', l₀', l₁', rest', ext', hsch', hm', hmP'⟩, hy, e⟩ :=
    tableJ_of ht (T₀ := T₀) (⟨sts[j.val], hst, _, k0, l₀, l₁, rest, ext, hsch, hm, hmP⟩ :
      SetupJ I mPts j.val (partJson (dj (L.draw ω)) I.count j.val))
  dsimp only at hy
  subst hy
  have hget := (planJ_get ht T₀ dj (L.draw ω) R j).trans e
  have hS : ((planJ ht T₀ dj (L.draw ω) R).get j).S = stmtOf sts[j.val] hwf (regionsOf sts[j.val]) := by
    rw [hget]
    exact model_eq_stmtOf ((tagsAt_typed I.tags I.count j.val).trans ht) h' hwf _
  have hpl : ((planJ ht T₀ dj (L.draw ω) R).get j).ptLocal = PT_LOCAL := by rw [hget]; rfl
  have hsch : ((planJ ht T₀ dj (L.draw ω) R).get j).sch = schedOf sch := by
    have h1 : Accounting.fast100 sts[j.val].m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := ((planJ ht T₀ dj (L.draw ω) R).get j).hsch
    rw [hS] at h2
    exact Option.some.inj (h2.symm.trans h1)
  rw [hmP, hpts] at hk
  obtain ⟨pend, hreps, hv⟩ := hk hwf hr hm
  have hrep : ∀ r : Fin 2, PlaysCoin.RepRun (repsOf (D := Dig) execArith ((planJ ht T₀ dj (L.draw ω) R).get j) o.pts r)
      (PlaysCoin.repSpine Dig ((planJ ht T₀ dj (L.draw ω) R).get j) o.pts)
      (decJ (planJ ht T₀ dj) (L.draw ω) R j) (o.reps j r) [] (pend r) := by
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
  have hopn : opnJ I (planJ ht T₀ dj) dj (L.draw ω) R o.proofs j =
      opnOf (Setup.ofCircuit sts[j.val]) sch
        (Model.extraClaims (stmtOf sts[j.val] hwf (regionsOf sts[j.val])) PT_LOCAL mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnJ
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

end FlockSoundness.Discharge.DecodeZLJ
