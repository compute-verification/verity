import Proofs.Flock.Claims.KS.PlainParts
import Proofs.Flock.Claims.KS.PlainDrawDecK

/-!
# The live self-clash at level 0

A live self-clash at level 0 is two openings of one position to different columns among table 0's level-0 openings,
each verifying against the outcome's cap, at the pendings the reps' decoded tapes play to. Where `verify` accepts a
transcript recording an outcome at the setup at the draw, those tapes play to the deployed reps' pendings, whose
output's level-0 openings are the reps' `plainRepColsOf` entries at the transcript's proof files (`accepts_opens₀LDK`),
so the clash is two of those entries at one position with different columns. At the concrete simulator, where `P`'s
run accepts, the outcome is the run's and those entries are the run's raw columns (`rawCols_plain`), so a live
self-clash at level 0 gives the run a raw clash at level 0.

The draw-layer lemmas are proved once, at `VK keep` over any transcript recording the outcome (`RecordsLiveAtDK`),
and `clash₀_wrPD`, `clash₀_wrPDH`, `plain_clash₀At` and `plain_clash₀HAt` are them at `keep := id` and `wrPD`;
`PlainClash0K` has them at `wrPDK keep`. The hidden-output lemmas are proved once more, at a leaf form `F` (its model
leaf, tag premise, verdicts and digest length): hm96 here and plain SHA-512 leaves in `PlainLeafClash` are instances.
The entries are `plainRepColsOf` at the decoded rep streams' rounds, which reads their coins only.

* `playsT_det`: a run on one tape ends at one outcome (`Game.PlaysT.det`).
* `EntriesClash`: two of the reps' entries at one position with different columns, at the setup at the draw;
  `EntriesClashK` over `Record.decodeWithK keep`, with `entriesClash_iff_K_id` at `keep := id`.
* `clash₀_wrK_of`, `clash₀_wrKHLf_of`: the level-0 self-clash at the deployed plain decoders is an `EntriesClashK`, at
  public outputs and at hidden outputs under a leaf form; `clash₀_wrKH_of` at hm96; `clash₀_wrPD`, `clash₀_wrPDH` at
  `wrPD`.
* `Clash₀LLf`, `LaterClashLLf`: `SelfClashL`'s two cases, at level 0 and past it, at a model leaf; `Clash₀L`,
  `LaterClashL` at `leaf512`.
* `plain_clash₀AtK_of`, `plain_clash₀HLfAtK_of`: those at the statement's draw, at `plainTab`; `plain_clash₀HAtK_of` at
  hm96; `plain_clash₀At`, `plain_clash₀HAt` at `wrPD`.
* `rawCols_plain`: an accepted plain run's raw columns are its reps' `plainRepColsOf` entries.
* `plainClashQ₀`: so a level-0 self-clash at the concrete simulator's outcome gives `P`'s accepting run a raw clash.
* `plainClashQ_or`, `PlainSimLaterClashAt`, `plainSimClashAt_of_later`: so `PlainSimClashAt` needs only its case
  past level 0.
-/

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A run on one tape ends at one outcome.** -/
theorem playsT_det {α : Type} {g : Game α} {ts : List Step} {a b : α} (ha : PlaysT g ts a) (hb : PlaysT g ts b) :
    a = b := by
  induction ha generalizing b with
  | ret _ => exact hb.ret_inv.2.symm
  | send _ _ ih => exact ih hb.send_msg
  | coin _ _ ih => exact ih hb.coin_val

/-- **Two of the reps' entries at one position with different columns**, in the transcript `t` at the setup at the
draw `d`: at its one table, that table's `fast100` schedule and its record's rep streams there, two of the reps'
`plainRepColsOf` entries at `t`'s proof files, at the shape `TxNoOffAt` reads, open one position to different
columns. -/
def EntriesClash (I : Exec.Inputs) (d : Json) (t : Exec.Transcript) : Prop :=
  ∃ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : Session) (sch : _root_.Flock.Schedule) (s₀ s₁ : StreamRec),
    setupAtD I (some d) = .ok (spec, #[st]) ∧ Record.decode spec t.record = .ok sess ∧
    _root_.Flock.fast100 st.m = some sch ∧
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
    ∃ (q : ℕ) (a b : List 𝔽),
      (q, a) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds
          t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds t.proofs[1]! ∧
      (q, b) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds
          t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds t.proofs[1]! ∧
      a ≠ b

/-- **Two of the reps' entries at one position with different columns**, in `t` at the setup at `d`, the session the
record decodes to through `keep` (`EntriesClash`). -/
def EntriesClashK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) (t : Exec.Transcript) : Prop :=
  ∃ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : Session) (sch : _root_.Flock.Schedule) (s₀ s₁ : StreamRec),
    setupAtD I (some d) = .ok (spec, #[st]) ∧ Record.decodeWithK keep With.std spec t.record = .ok sess ∧
    _root_.Flock.fast100 st.m = some sch ∧
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
    ∃ (q : ℕ) (a b : List 𝔽),
      (q, a) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds
          t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds t.proofs[1]! ∧
      (q, b) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds
          t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds t.proofs[1]! ∧
      a ≠ b

/-- **`EntriesClash` is `EntriesClashK` at `keep := id`.** -/
theorem entriesClash_iff_K_id (I : Exec.Inputs) (d : Json) (t : Exec.Transcript) :
    EntriesClash I d t ↔ EntriesClashK id I d t := by
  unfold EntriesClash EntriesClashK
  simp only [decodeWithK_id_std]

open DecodeZL Integrate in
/-- **A level-0 self-clash at the deployed plain decoders is a clash of the reps' entries**, at one table: where
`verify` at `keep` accepts a transcript `t` recording `o` at the setup at the draw `dj S` (`hrl`), two pendings the
reps' decoded tapes play to (`hp`), whose output at `o`'s cap and openings has two openings of one position to different
columns verifying at level 0 (`hcl`), give two of the reps' `plainRepColsOf` entries at the setup's table, its `fast100`
schedule and `t`'s proof files at one position with different columns. -/
theorem clash₀_wrK_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I (dj S) o t)
    (hacc : AcceptsAtDK keep I (some (dj S)) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leaf512 (capOfRoot o.rootB[0]!)
      (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0))
      (opens₀L (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch (tableOutL H512 enc512 leaf512
        (capOfRoot o.rootB[0]!) pend' (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs)))) :
    EntriesClashK keep I (dj S) t := by
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hprf, hrec⟩ := recordsLive_wrK1_of keep I (dj S) o hrl hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAt_of_acceptsDK keep ht hc ho hh hc1 dj x₀ hacc hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_repsOneDK keep hrr hl hc1 hacc
  rw [HiddenExec.setupOne_public hh] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨hroot, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupH_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle = MerkleScheme.hm96Sha512 := setupH_hm96 hl hst
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
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opens₀LDK keep hacc
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
    (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[t.proofs[0]!, t.proofs[1]!]))
    hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ (by rw [hn']; rfl) hv'
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
  rw [hopn, hsch] at hcl
  obtain ⟨u, hu, v, hv, huv, -, -, hne⟩ := hcl
  have hmem : ∀ z ∈ opens₀L (schedOf sch) (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[t.proofs[0]!, t.proofs[1]!])),
      (z.1, z.2.1) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds
          t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds
          t.proofs[1]! := fun z hz => by
    have h' := List.mem_map_of_mem (f := fun x : ℕ × List 𝔽 × List UInt8 × List Dig => (x.1, x.2.1)) hz
    rw [hent] at h'
    exact h'
  refine ⟨spec, st, sess, sch, s₀, s₁, hsu, hdec, hf, hs₀, hs₁, u.1, u.2.1, v.2.1, hmem u hu, ?_, hne⟩
  rw [huv]
  exact hmem v hv

open DecodeZL HiddenExec HiddenDec in
/-- **A level-0 self-clash at the deployed plain decoders at hidden outputs, at a leaf form `F`, is a clash of the reps'
entries**, at `keep` over any transcript recording `o`: `clash₀_wrK_of` at the hidden tables and their openings
`opnAtH`. -/
theorem clash₀_wrKHLf_of (keep : Bool → Bool) (F : LeafForm)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : F.TagsOk I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I (dj S) o t)
    (hacc : AcceptsAtDK keep I (some (dj S)) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 F.lf (capOfRoot o.rootB[0]!)
      (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0))
      (opens₀L (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 F.lf
        (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)))) :
    EntriesClashK keep I (dj S) t := by
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨hprf, hrec⟩ := recordsLive_wrK1_of keep I (dj S) o hrl hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAtH_of_acceptsDK keep ht hc hh hc1 dj x₀ hacc hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ :=
    accepts_repsOneDLfK keep F.lf hrr hc1 (F.verdicts keep hh hl hacc) hacc
  rw [setupOne_hidden hh] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨hroot, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupHidden_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle.digestLen = 64 := F.digestLen hst hl
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
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opens₀LDLfK keep F.lf hacc
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
    (tableOutL H512 enc512 F.lf (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[t.proofs[0]!, t.proofs[1]!]))
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
  rw [hopn, hsch] at hcl
  obtain ⟨u, hu, v, hv, huv, -, -, hne⟩ := hcl
  have hmem : ∀ z ∈ opens₀L (schedOf sch) (tableOutL H512 enc512 F.lf (capOfRoot o.rootB[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[t.proofs[0]!, t.proofs[1]!])),
      (z.1, z.2.1) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds
          t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds
          t.proofs[1]! := fun z hz => by
    have h' := List.mem_map_of_mem (f := fun x : ℕ × List 𝔽 × List UInt8 × List Dig => (x.1, x.2.1)) hz
    rw [hent] at h'
    exact h'
  refine ⟨spec, st, sess, sch, s₀, s₁, hsu, hdec, hf, hs₀, hs₁, u.1, u.2.1, v.2.1, hmem u hu, ?_, hne⟩
  rw [huv]
  exact hmem v hv

open DecodeZL HiddenExec HiddenDec in
/-- **A level-0 self-clash at the deployed plain decoders at hidden outputs is a clash of the reps' entries**:
`clash₀_wrK_of` at the hidden tables and their openings `opnAtH`; `clash₀_wrKHLf_of` at `hm96`. -/
theorem clash₀_wrKH_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I (dj S) o t)
    (hacc : AcceptsAtDK keep I (some (dj S)) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leaf512 (capOfRoot o.rootB[0]!)
      (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0))
      (opens₀L (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leaf512
        (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)))) :
    EntriesClashK keep I (dj S) t :=
  clash₀_wrKHLf_of keep LeafForm.hm96 ht hc dj x₀ hT hS R o hrl hacc pend' hp hcl

open DecodeZL Integrate in
/-- **A level-0 self-clash at the deployed plain decoders is a clash of the reps' entries**: `clash₀_wrK_of` at
`keep := id` and what `wrPD` writes for `o`. -/
theorem clash₀_wrPD {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leaf512 (capOfRoot o.rootB[0]!)
      (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0))
      (opens₀L (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch (tableOutL H512 enc512 leaf512
        (capOfRoot o.rootB[0]!) pend' (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs)))) :
    EntriesClash I (dj S) (wrPD I (dj S) o) :=
  (entriesClash_iff_K_id _ _ _).2 <| clash₀_wrK_of id ht hc ho dj x₀ hT hS R o
    (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD I (dj S) o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp hcl

open DecodeZL HiddenExec HiddenDec in
/-- **A level-0 self-clash at the deployed plain decoders at hidden outputs is a clash of the reps' entries**:
`clash₀_wrKH_of` at `keep := id` and what `wrPD` writes for `o`. -/
theorem clash₀_wrPDH {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leaf512 (capOfRoot o.rootB[0]!)
      (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0))
      (opens₀L (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leaf512
        (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)))) :
    EntriesClash I (dj S) (wrPD I (dj S) o) :=
  (entriesClash_iff_K_id _ _ _).2 <| clash₀_wrKH_of id ht hc dj x₀ hT hS R o
    (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD I (dj S) o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp hcl

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A self-clash at level 0** at a table's output, at the model leaf `lf`: `SelfClashL`'s first case. -/
def Clash₀LLf (lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512) (sch : Accounting.Schedule)
    (out : Model.OutCL 𝔽 Dig (List UInt8)) : Prop :=
  ClashL H512 enc512 lf out.cap₀ (lvOf sch 0).lanes (Model.pathLen (lvOf sch 0)) (opens₀L sch out)

/-- **A self-clash past level 0** at a table's output, at the model leaf `lf`: `SelfClashL`'s second case. -/
def LaterClashLLf (lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512) (sch : Accounting.Schedule)
    (out : Model.OutCL 𝔽 Dig (List UInt8)) : Prop :=
  ∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some cap ∧
    ClashL H512 enc512 lf cap (lvOf sch ℓ).lanes (Model.pathLen (lvOf sch ℓ))
      (slotOpensL out r ℓ (lvOf sch ℓ).queries)

/-- **A self-clash at level 0** at a table's output: `SelfClashL`'s first case. -/
def Clash₀L (sch : Accounting.Schedule) (out : Model.OutCL 𝔽 Dig (List UInt8)) : Prop :=
  ClashL H512 enc512 leaf512 out.cap₀ (lvOf sch 0).lanes (Model.pathLen (lvOf sch 0)) (opens₀L sch out)

/-- **A self-clash past level 0** at a table's output: `SelfClashL`'s second case. -/
def LaterClashL (sch : Accounting.Schedule) (out : Model.OutCL 𝔽 Dig (List UInt8)) : Prop :=
  ∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some cap ∧
    ClashL H512 enc512 leaf512 cap (lvOf sch ℓ).lanes (Model.pathLen (lvOf sch ℓ))
      (slotOpensL out r ℓ (lvOf sch ℓ).queries)

/-- **The deployed plain decoders' level-0 self-clash at `plainTab`, at the draw**: `clash₀_wrK_of` at the statement's
draw `d` and no drawn unit, as `plain_hdecAtK_of` is `decodesTable_wrK_sc_of` there. -/
theorem plain_clash₀AtK_of (keep : Bool → Bool)
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
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀L ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClashK keep ins.read.I d t := by
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
  refine clash₀_wrK_of keep ht hc ho (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () o hrl hacc pend' ?_ ?_ <;>
    rw [e]
  · exact hp
  · exact hcl

/-- **The deployed plain decoders' level-0 self-clash at `plainTab`, at the draw, at hidden outputs under a leaf form
`F`**: `clash₀_wrKHLf_of` at the statement's draw `d` and no drawn unit. -/
theorem plain_clash₀HLfAtK_of (keep : Bool → Bool) (F : AtDraw.LeafForm)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true)
    (hl : ins.read.I.tags.merkleLeaf = F.tag) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep ins.read.I d o t) (hacc : AcceptsAtDK keep ins.read.I (some d) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀LLf F.lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 F.lf (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClashK keep ins.read.I d t := by
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
    exact tableAtH_plainTab ht (HmOut.parse_isTmpl hc) (fun _ : Finset (Fin 0) => d) (spec := st.spec) (S := ∅) x₀ x₀
      hx₀ L m k0 l₀ l₁
      rest ext hsch hm hlay
  have hS : HiddenDec.DrawOkH ins.read.I (fun _ : Finset (Fin 0) => d) st.spec.mPts ∅ :=
    ⟨fun _ _ u hu => absurd hu (Finset.notMem_empty u), fun st' hst' => by
      obtain rfl := Except.ok.inj (hst.symm.trans hst')
      rfl⟩
  refine clash₀_wrKHLf_of keep F ht hc (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () o hrl hacc pend'
    ?_ ?_ <;> rw [e]
  · exact hp
  · exact hcl

/-- **The deployed plain decoders' level-0 self-clash at `plainTab`, at the draw, at hidden outputs under hm96 leaves**:
`clash₀_wrKH_of` at the statement's draw `d` and no drawn unit, as `plain_hdecHAtK_of` is `decodesTable_wrK_scH_of`
there; `plain_clash₀HLfAtK_of` at `hm96`. -/
theorem plain_clash₀HAtK_of (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true)
    (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep ins.read.I d o t) (hacc : AcceptsAtDK keep ins.read.I (some d) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀L ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClashK keep ins.read.I d t :=
  plain_clash₀HLfAtK_of keep AtDraw.LeafForm.hm96 hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hrl hacc
    pend' hp hcl

/-- **The deployed plain decoders' level-0 self-clash at `plainTab`, at the draw**: `plain_clash₀AtK_of` at
`keep := id` and what `wrPD` writes for `o`. -/
theorem plain_clash₀At {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
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
    (hcl : Clash₀L ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClash ins.read.I d (wrPD ins.read.I d o) :=
  (entriesClash_iff_K_id _ _ _).2 <| plain_clash₀AtK_of id hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay o
    (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD _ d o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp hcl

/-- **The deployed plain decoders' level-0 self-clash at `plainTab`, at the draw, at hidden outputs under hm96 leaves**:
`plain_clash₀HAtK_of` at `keep := id` and what `wrPD` writes for `o`. -/
theorem plain_clash₀HAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true)
    (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀L ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClash ins.read.I d (wrPD ins.read.I d o) :=
  (entriesClash_iff_K_id _ _ _).2 <| plain_clash₀HAtK_of id hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o
    (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD _ d o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp hcl

/-- **An accepted plain run's reps' columns, every one**: `cols_plain` before `distinctBelow`, so `plainRawCols` is
`plainRepColsOf` at `shapeOf` with `2 · |regions|` extra claims, on the live session's first two rep streams and proof
files. -/
theorem rawCols_plain (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {st : Stmt} {sch : Flock.Schedule} (hf : lastFinal seen = some f)
    (hset : setupOf false f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hst : sts[0]? = some st)
    (hsch : Flock.fast100 st.m = some sch) (hp2 : spec.points = 2) :
    plainRawCols seen =
      Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 0 #[])
          (f.proofs.getD 0 ByteArray.empty) ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 1 #[])
          (f.proofs.getD 1 ByteArray.empty) := by
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
  unfold plainRawCols
  rw [hrun]
  simp only [Claims.Flock.ZkRun.plainShape, FlockDeploy.plainProofShape_eq hsched, hx, List.flatMap_cons,
    List.flatMap_nil, List.append_nil]

/-- **A level-0 self-clash at the concrete simulator's outcome is a raw clash of `P`'s accepting run**: at openings
`opn` whose level-0 self-clash at an outcome `wrPD` writes an accepted transcript for is an `EntriesClash` there
(`hE`, as `plain_clash₀At` and `plain_clash₀HAt` give), where `P`'s run accepts at `plainΦ x`, the simulator's outcome
is the run's (`plainSimRun`), `wrPD` records the run's rep streams and proof files, and those entries are the run's
raw columns (`rawCols_plain`), so the run has a raw clash at level 0 (`PlainClashQ` at `Clash₀L`). -/
theorem plainClashQ₀ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        Clash₀L sch (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        EntriesClash ins.read.I d (wrPD ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQ P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (Clash₀L sch) := by
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

/-- **`PlainClashQ` at a property two others cover**: where `Q` implies `Q₁ ∨ Q₂` and each of those gives `P`'s
accepting runs a raw clash, so does `Q`. -/
theorem plainClashQ_or {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec} {st : Stmt}
    {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8)}
    {Q₁ Q₂ : Model.OutCL 𝔽 Dig (List UInt8) → Prop} (Q : Model.OutCL 𝔽 Dig (List UInt8) → Prop)
    (h₁ : PlainClashQ P x ω st T hw I d opn Q₁) (h₂ : PlainClashQ P x ω st T hw I d opn Q₂)
    (hQ : ∀ out, Q out → Q₁ out ∨ Q₂ out) : PlainClashQ P x ω st T hw I d opn Q := by
  rintro C u r' ⟨a, ha, hq⟩ hacc
  rcases hQ _ hq with hq | hq
  · exact h₁ C u r' ⟨a, ha, hq⟩ hacc
  · exact h₂ C u r' ⟨a, ha, hq⟩ hacc

/-- **The live self-clash's clash past level 0**: `PlainSimClashAt` at `LaterClashL` in place of `SelfClashL`, at
every plain statement in scope, its one table's `fast100` schedule and the plain decoders' openings, public and
hidden under hm96 leaves. -/
def PlainSimLaterClashAt (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (d : Json) (_ : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Claims.Prover Ω) (ω : Ω),
    (ins.read.I.tags.hiddenOutputs = false →
      PlainClashQ P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)) ∧
    (ins.read.I.tags.hiddenOutputs = true → ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" →
      PlainClashQ P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))

/-- **The live self-clash's clash from its clash past level 0**: `SelfClashL` is `Clash₀L` or `LaterClashL`, and
`plainClashQ₀` at `plain_clash₀At` and `plain_clash₀HAt` gives the first. -/
theorem plainSimClashAt_of_later {x : Claims.Msg} (hl : PlainSimLaterClashAt x) : PlainSimClashAt x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω _ _ P ω
  refine ⟨fun hh => ?_, fun hh hleaf => ?_⟩
  · exact plainClashQ_or _ (plainClashQ₀ hs hin hset h hd _ _ _
      (plain_clash₀At hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω)
      ((hl s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).1 hh) fun _ hq => hq
  · exact plainClashQ_or _ (plainClashQ₀ hs hin hset h hd _ _ _
      (plain_clash₀HAt hs hin hset h hh hleaf hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω)
      ((hl s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).2 hh hleaf) fun _ hq => hq

end FlockDeploy.KS
