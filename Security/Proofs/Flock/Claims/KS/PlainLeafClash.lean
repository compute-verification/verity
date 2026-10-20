import Proofs.Flock.Claims.KS.PlainLeafDec

/-!
# The live self-clash at plain SHA-512 leaves

`PlainClash0` and `PlainClashLater` at hidden outputs under plain SHA-512 leaves, at the model leaf
`Refine.leafPlain`. A self-clash at `leafPlain` is two verified openings of one position to different columns, which
the deployed acceptance makes two of the reps' entries (`accepts_opens₀LDU`, `accepts_opensLDU`), so a raw clash of
the accepting run (`EntriesClash`, `LaterEntriesClash`); the raw clash's collision (`plain_selfClash_tied`,
`plain_levelClash`) reads the deployed Merkle scheme, whichever it is.

* `accepts_opensLDU`, `clash₀_wrPDHU`, `laterEntries_wrPDHU`, `laterClash_wrPDHU`: `accepts_opensLD`,
  `clash₀_wrPDH`, `laterEntries_wrPDH`, `laterClash_wrPDH` at `leafPlain`.
* `Clash₀LU`, `LaterClashLU`, `plainDecQU`, `PlainClashQU`: `Clash₀L`, `LaterClashL`, `plainDecQ`, `PlainClashQ` at
  `leafPlain`.
* `plain_clash₀HUAt`, `plain_laterEntriesHUAt`, `plain_laterClashHUAt`, `plainClashQ₀U`, `plainClashQ_orU`,
  `plainClashQLU`: their hm96 counterparts at the statement's draw.
* `plainSimClashHU`: so a live self-clash at `leafPlain` gives `P`'s accepting runs a raw clash.
-/

namespace FlockSoundness.Discharge.AtDraw

open Claims.Bridge.PlainDraw (setupAtD AcceptsAtD)
open Flock Refine Model Game FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **At flock-verify's acceptance at `dr`, each table's recorded caps and level-`ℓ` openings are its reps' deployed
ones**, at every level `ℓ ≥ 1`: `accepts_opensLD` at `leafPlain`, `accepts_opens₀LD` past level 0, each rep's caps its proof's recursive caps and its
level-`ℓ` openings its `plainLevelColsOf` entries. -/
theorem accepts_opensLDU {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtD I dr t) :
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
          out = tableOutL H512 enc512 leafPlain cap (pend₀, pend₁)
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
  obtain ⟨spec, sts, sess, hs, hsess, hall⟩ := accepts_reqsD h
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

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
  Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A self-clash at level 0** at a table's output: `SelfClashL`'s first case at `leafPlain`. -/
def Clash₀LU (sch : Accounting.Schedule) (out : Model.OutCL 𝔽 Dig (List UInt8)) : Prop :=
  ClashL H512 enc512 leafPlain out.cap₀ (lvOf sch 0).lanes (Model.pathLen (lvOf sch 0)) (opens₀L sch out)

/-- **A self-clash past level 0** at a table's output: `SelfClashL`'s second case at `leafPlain`. -/
def LaterClashLU (sch : Accounting.Schedule) (out : Model.OutCL 𝔽 Dig (List UInt8)) : Prop :=
  ∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some cap ∧
    ClashL H512 enc512 leafPlain cap (lvOf sch ℓ).lanes (Model.pathLen (lvOf sch ℓ))
      (slotOpensL out r ℓ (lvOf sch ℓ).queries)

end FlockDeploy.KS

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A self-clash past level 0 at `leafPlain` in an output whose reps' later openings are the deployed reps' is a
clash of one rep's level-`ℓ` entries**: `laterEntriesClash_of`'s proof, which reads no leaf. -/
theorem laterEntriesClash_ofU {sch : Accounting.Schedule} {I : Exec.Inputs} {d : Json} {t : Exec.Transcript}
    {out : Model.OutCL 𝔽 Dig (List UInt8)} (he : ∀ r < 2, LaterEntriesAt sch I d t out r)
    (hcl : FlockDeploy.KS.LaterClashLU sch out) : LaterEntriesClash I d t := by
  obtain ⟨r, hr2, ℓ, h1, -, -, u, hu, v, hv, huv, -, -, hne⟩ := hcl
  obtain ⟨spec, st, sess, sch', hsu, hdec, hf, -, s, -, hs, -, -, hent⟩ := he r hr2
  have hmem : ∀ z ∈ slotOpensL out r ℓ (lvOf sch ℓ).queries, (z.1, z.2.1) ∈
      FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch' (2 * st.regions.size)) s.rounds
        t.proofs[r]! ℓ :=
    fun z hz => hent ℓ h1 ▸ List.mem_map_of_mem hz
  exact ⟨spec, st, sess, sch', hsu, hdec, hf, r, hr2, s, hs, ℓ, u.1, u.2.1, v.2.1, hmem u hu, huv ▸ hmem v hv, hne⟩

open DecodeZL HiddenExec HiddenDec in
/-- **A level-0 self-clash at the deployed plain decoders at hidden outputs is a clash of the reps' entries**:
`clash₀_wrPD` at the hidden tables and their openings `opnAtH`. -/
theorem clash₀_wrPDHU {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leafPlain (capOfRoot o.rootB[0]!)
      (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0))
      (opens₀L (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leafPlain
        (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)))) :
    EntriesClash I (dj S) (wrPD I (dj S) o) := by
  change AcceptsAtD I (some (dj S)) (wrPD (J := 1) I (dj S) o) at hacc
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hprf, hrec⟩ := recordsLive_wrPD1 I (dj S) o hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAtH_of_acceptsD ht hc hh hc1 dj x₀ hacc hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_repsOneDU hrr hh hl hc1 hacc
  rw [setupOne_hidden hh] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨hroot, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupHidden_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := setupHidden_plain hl hst
  obtain ⟨pend, hreps, hv⟩ := hk hwf hr hm
  have hn : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL q pts).length = 2 * st.regions.size := by
    intro q pts
    rw [extraClaims_length]
    simp [stmtOf, regionsOf]
  obtain ⟨s₀, p₀, -, hs₀, hp₀, hpl₀, -⟩ := hreps 0
  obtain ⟨s₁, p₁, -, hs₁, hp₁, hpl₁, -⟩ := hreps 1
  have hcap : sess.rootB[0]! = o.rootB[0]! := hroot
  have hv' := hv
  rw [hn, hcap] at hv'
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opens₀LDU hacc
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨sch', hf', hk'⟩ := hall 0 (by simp)
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  have hn' : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf #[st][0] hwf (regionsOf #[st][0])) PT_LOCAL q pts).length = 2 * st.regions.size :=
    hn
  obtain ⟨-, hent⟩ := hk' hwf hr hm p₀ p₁ s₀ s₁ (pend 0) (pend 1) (capOfRoot o.rootB[0]!)
    (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPD (J := 1) I (dj S) o).proofs[0]!, (wrPD (J := 1) I (dj S) o).proofs[1]!]))
    hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ (by rw [hn']; rfl) hv'
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
    rw [← hrs]
    exact repRun_at op_carries lig_carries decodeProof_sizes _ st hwf sch hf (by rw [hms]; rfl) hS' hsch hpl
      (oneOf o).pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s'
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  have hpe : ∀ r : Fin 2, Tables.sel pend' r = pend r := fun r =>
    playsT_det (hp r) (by
      have := PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig (tablesAtH (Reg := Reg) dj x₀ S R)
        (oneOf o).pts r) (hrep r)
      fin_cases r <;> exact this)
  have hpe' : pend' = (pend 0, pend 1) := Prod.ext (hpe 0) (hpe 1)
  subst hpe'
  have hprf' : (wrPD (J := 1) I (dj S) o).proofs = o.proofs := hprf
  have hopn : opnAtH (tablesAtH (Reg := Reg) dj x₀) I dj S R o.proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPD (J := 1) I (dj S) o).proofs[0]!, (wrPD (J := 1) I (dj S) o).proofs[1]!] := by
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
  rw [hopn, hsch] at hcl
  obtain ⟨u, hu, v, hv, huv, -, -, hne⟩ := hcl
  have hmem : ∀ z ∈ opens₀L (schedOf sch) (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPD (J := 1) I (dj S) o).proofs[0]!, (wrPD (J := 1) I (dj S) o).proofs[1]!])),
      (z.1, z.2.1) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds
          (wrPD (J := 1) I (dj S) o).proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds
          (wrPD (J := 1) I (dj S) o).proofs[1]! := fun z hz => by
    have h' := List.mem_map_of_mem (f := fun x : ℕ × List 𝔽 × List UInt8 × List Dig => (x.1, x.2.1)) hz
    rw [hent] at h'
    exact h'
  refine ⟨spec, st, sess, sch, s₀, s₁, hsu, hdec, hf, hs₀, hs₁, u.1, u.2.1, v.2.1, hmem u hu, ?_, hne⟩
  rw [huv]
  exact hmem v hv

open DecodeZL HiddenExec HiddenDec in
/-- **At the deployed plain decoders at hidden outputs, the reps' recorded caps and later openings are the deployed
reps'**: `laterEntries_wrPD` at the hidden tables and their openings `opnAtH`. -/
theorem laterEntries_wrPDHU {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAt (tablesAtH (Reg := Reg) dj x₀ S R).sch I (dj S) (wrPD I (dj S) o)
      (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)) r := by
  change AcceptsAtD I (some (dj S)) (wrPD (J := 1) I (dj S) o) at hacc
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hprf, hrec⟩ := recordsLive_wrPD1 I (dj S) o hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAtH_of_acceptsD ht hc hh hc1 dj x₀ hacc hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_repsOneDU hrr hh hl hc1 hacc
  rw [setupOne_hidden hh] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨-, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupHidden_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := setupHidden_plain hl hst
  obtain ⟨pend, hreps, -⟩ := hk hwf hr hm
  have hn : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL q pts).length = 2 * st.regions.size := by
    intro q pts
    rw [extraClaims_length]
    simp [stmtOf, regionsOf]
  obtain ⟨s₀, p₀, -, hs₀, hp₀, hpl₀, -⟩ := hreps 0
  obtain ⟨s₁, p₁, -, hs₁, hp₁, hpl₁, -⟩ := hreps 1
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opensLDU hacc
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
    (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPD (J := 1) I (dj S) o).proofs[0]!, (wrPD (J := 1) I (dj S) o).proofs[1]!]))
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
    rw [← hrs]
    exact repRun_at op_carries lig_carries decodeProof_sizes _ st hwf sch hf (by rw [hms]; rfl) hS' hsch hpl
      (oneOf o).pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s'
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  have hpe : ∀ r : Fin 2, Tables.sel pend' r = pend r := fun r =>
    playsT_det (hp r) (by
      have := PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig (tablesAtH (Reg := Reg) dj x₀ S R)
        (oneOf o).pts r) (hrep r)
      fin_cases r <;> exact this)
  have hpe' : pend' = (pend 0, pend 1) := Prod.ext (hpe 0) (hpe 1)
  subst hpe'
  have hprf' : (wrPD (J := 1) I (dj S) o).proofs = o.proofs := hprf
  have hopn : opnAtH (tablesAtH (Reg := Reg) dj x₀) I dj S R o.proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPD (J := 1) I (dj S) o).proofs[0]!, (wrPD (J := 1) I (dj S) o).proofs[1]!] := by
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
entries**: `laterClash_wrPD` at the hidden tables and their openings `opnAtH` (`laterEntries_wrPDHU`). -/
theorem laterClash_wrPDHU {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : FlockDeploy.KS.LaterClashLU (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leafPlain
      (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs))) :
    LaterEntriesClash I (dj S) (wrPD I (dj S) o) :=
  laterEntriesClash_ofU (laterEntries_wrPDHU ht hc dj x₀ hT hS R o hacc pend' hp) hcl

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
  Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The deployed plain decoders' level-0 self-clash at `plainTab`, at the draw, at hidden outputs under plain SHA-512
leaves**: `clash₀_wrPDHU` at the statement's draw `d` and no drawn unit, as `plain_hdecHUAt` is
`decodesTable_wrPD_scHU` there. -/
theorem plain_clash₀HUAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true)
    (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀LU ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClash ins.read.I d (wrPD ins.read.I d o) := by
  obtain ⟨ht, hc, hT, hst, rfl, -, -⟩ := plain_setupHiddenU hs hin hset h hh hl hd
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
  refine clash₀_wrPDHU ht hc (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () o hacc pend' ?_ ?_ <;>
    rw [e]
  · exact hp
  · exact hcl

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw, at hidden outputs under plain SHA-512
leaves**: `laterEntries_wrPDHU` at the statement's draw `d` and no drawn unit. -/
theorem plain_laterEntriesHUAt {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAt ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPD ins.read.I d o)
      (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend'
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r := by
  obtain ⟨ht, hc, hT, hst, rfl, -, -⟩ := plain_setupHiddenU hs hin hset h hh hl hd
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
  have he := laterEntries_wrPDHU ht hc (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () o hacc pend'
    (by rw [e]; exact hp)
  rw [e] at he
  exact he

/-- **The deployed plain decoders' self-clash past level 0 at `plainTab`, at the draw, at hidden outputs under plain
SHA-512 leaves**: `laterEntriesClash_ofU` at `plain_laterEntriesHUAt`. -/
theorem plain_laterClashHUAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : LaterClashLU ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    LaterEntriesClash ins.read.I d (wrPD ins.read.I d o) :=
  laterEntriesClash_ofU
    (plain_laterEntriesHUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hacc pend' hp) hcl

/-- **A live outcome decoded at the verdict at the draw, with a property `Q` of table 0's output**: the verdict
`plainVerdictAt I d C rootB₀` holds, the reps' decoded tapes play to pendings, and the output at them and the
openings `opn` has `Q`. At `Q := SelfClashL` it is `LiveSelfClash` at the plain decoders. -/
def plainDecQU {mPts : ℕ} (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) (T : TabSpec mPts) (I : Exec.Inputs)
    (d : Json) {M : ℕ} (C : Fin M → List 𝔽) (rootB₀ : Array ByteArray)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) (o : LiveOutJ 1 mPts) :
    Prop :=
  ∃ a, Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
    (plainVerdictAt I d C rootB₀) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)) o a ∧
    Q (a.getD 0 dfltOutL)

/-- **`Q` at the decoded live outcome gives `P`'s accepting runs a raw clash**: at the concrete simulator, wherever
`plainDecQU Q` holds at `plainSim P`'s live outcome and `V false` accepts `P` at `plainΦ x`, that run has a raw clash
(`PlainRawClashL`). -/
def PlainClashQU {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536),
    plainDecQU Q T I d C (plainRoot P x ω) opn (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec)
      (plainW spec hw) sha512 (plainSim P x spec st) x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).out = .acc →
    PlainRawClashL P x ω (plainΦ x spec st (u, r'))

/-- **A level-0 self-clash at the concrete simulator's outcome is a raw clash of `P`'s accepting run**: at openings
`opn` whose level-0 self-clash at an outcome `wrPD` writes an accepted transcript for is an `EntriesClash` there
(`hE`, as `plain_clash₀At` and `plain_clash₀HAt` give), where `P`'s run accepts at `plainΦ x`, the simulator's outcome
is the run's (`plainSimRun`), `wrPD` records the run's rep streams and proof files, and those entries are the run's
raw columns (`rawCols_plain`), so the run has a raw clash at level 0 (`PlainClashQU` at `Clash₀LU`). -/
theorem plainClashQ₀U {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        Clash₀LU sch (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        EntriesClash ins.read.I d (wrPD ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQU P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (Clash₀LU sch) := by
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
  obtain ⟨spec'', st', sess, sch', s₀, s₁, hsu, hdec, hf', hs₀, hs₁, q, a₁, b₁, ha, hb, hab⟩ :=
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
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decision_runs hsu₀ hrun
  have hv' : verifyWithM (hashesIn ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).vLog.map
      Prod.fst)) spec (tabsOf #[st']) f.record f.proofs = .ok () := verdict_within hvr
  have hcols := rawCols_plain (st := st') _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hf'
    (HiddenExec.setupTables_pointsAny hsetA).1
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (k : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec) →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro k y hy j
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hcl' : ∀ {q : ℕ} {v : List 𝔽}, (q, v) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st') sch'
        (2 * st'.regions.size)) y₀.rounds (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[0]! ++
      Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st') sch' (2 * st'.regions.size)) y₁.rounds
        (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[1]! →
      (q, v) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen := by
    intro q v hqv
    rw [hpr', plainRepColsOf_congr _ _ (hco 0 y₀ hr₀), plainRepColsOf_congr _ _ (hco 1 y₁ hr₁), ← getD_empty,
      ← getD_empty, ← hcols] at hqv
    exact hqv
  exact Or.inl ⟨q, a₁, b₁, hcl' ha, hcl' hb, hab⟩

/-- **`PlainClashQU` at a property two others cover**: where `Q` implies `Q₁ ∨ Q₂` and each of those gives `P`'s
accepting runs a raw clash, so does `Q`. -/
theorem plainClashQ_orU {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec} {st : Stmt}
    {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8)}
    {Q₁ Q₂ : Model.OutCL 𝔽 Dig (List UInt8) → Prop} (Q : Model.OutCL 𝔽 Dig (List UInt8) → Prop)
    (h₁ : PlainClashQU P x ω st T hw I d opn Q₁) (h₂ : PlainClashQU P x ω st T hw I d opn Q₂)
    (hQ : ∀ out, Q out → Q₁ out ∨ Q₂ out) : PlainClashQU P x ω st T hw I d opn Q := by
  rintro C u r' ⟨a, ha, hq⟩ hacc
  rcases hQ _ hq with hq | hq
  · exact h₁ C u r' ⟨a, ha, hq⟩ hacc
  · exact h₂ C u r' ⟨a, ha, hq⟩ hacc

/-- **A self-clash past level 0 at the concrete simulator's outcome is a raw clash of `P`'s accepting run**: at
openings `opn` whose self-clash past level 0 at an outcome `wrPD` writes an accepted transcript for is a
`LaterEntriesClash` there (`hE`, as `plain_laterClashAt` and `plain_laterClashHAt` give), where `P`'s run accepts at
`plainΦ x`, the simulator's outcome is the run's (`plainSimRun`), `wrPD` records the run's rep streams and proof files,
those entries are the run's level-`ℓ` rows (`rawLevelCols_plain`), and two of them at one position that differ clash in
the run's log (`plain_levelClash`), so the run has a raw clash (`PlainClashQU` at `LaterClashLU`). -/
theorem plainClashQLU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        LaterClashLU sch (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        LaterEntriesClash ins.read.I d (wrPD ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQU P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (LaterClashLU sch) := by
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

/-- **The live self-clash's clash at hidden outputs under plain SHA-512 leaves**: at a plain statement in scope there
and its one table's `fast100` schedule, at the hidden openings `plainOpnH`, a live self-clash at `leafPlain` at the
concrete simulator's outcome where `P`'s run accepts gives that run a raw clash: `SelfClashL` is `Clash₀LU` or
`LaterClashLU`, and `plainClashQ₀U` and `plainClashQLU` give each. -/
theorem plainSimClashHU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (regionsOf st)).m) (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQU P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
      ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (SelfClashL H512 enc512 leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) :=
  plainClashQ_orU _ (plainClashQ₀U hs hin hset h hd _ _ _
      (plain_clash₀HUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω)
    (plainClashQLU hs hin hset h hd _ _ _
      (plain_laterClashHUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω) fun _ hq => hq

end FlockDeploy.KS
