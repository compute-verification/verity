import Proofs.Flock.Claims.KS.PlainDrawAccK

/-!
# The deployed plain decoders at the statement's draw, at `VK keep`

The deployed plain decoders for the verdict at the statement's draw read through `keep`, on `PlainDrawAccK`'s
acceptance lemmas, at any record writer `wr` whose transcripts record their outcome at the setup at the draw
(`RecordsLiveAtDK`). The one place they meet the outcome's rounds is the rep's run (`PlaysCoin.RepRun`), which reads a
round's coin count, coins and bytes and never its digest, so the decoded rounds carry it to the outcome's up to their
digests (`PlaysCoin.repRun_of_erase`). At `wrPDK` they are the decoders for `plainVerdictAtK keep`, and at `keep := id`
and `wrPD` `PlainDrawDec`'s for `plainVerdictAt`:

* `recordsLive_wrK1_of`, `recordsLive_wrPDK1`: a recording transcript, and `wrPDK`, at a one-table outcome record it as
  one table's, the rounds up to their digests.
* `decodesTable_wrK_sc_of`, `decodesTable_wrK_scH_of`: the decoders right at public and at hidden outputs, on the
  outcomes whose transcript `verifyWithMK keep` accepts at the draw `dj S` with no column off `C` (`AcceptsAtDK`,
  `TxNoOffAtK`); `decodesTable_wrPDK_sc`, `decodesTable_wrPDK_scH` at `wrPDK`.
* `plain_hdecAtK_of`, `plain_hdecHAtK_of`: at `plainTab`; `plain_hdecAtK`, `plain_hdecHAtK` at `wrPDK`, with
  `plainVerdictAtK keep`.
-/

namespace FlockSoundness.Discharge.PlaysCoin

open Flock FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Exec Model Accounting Audit

theorem roundGroups_erase (dec : ℕ → ByteArray → List Msg) :
    ∀ (j : ℕ) (rs : List Round), roundGroups dec j (rs.map Round.erase) = roundGroups dec j rs
  | _, [] => rfl
  | j, rd :: rs => by
    simp only [List.map_cons, roundGroups, Round.erase_bytes, roundGroups_erase dec (j + 1) rs]

/-- **A rep's run reads its rounds as replay does**: `RepRun` reads a round's coin count, coins and bytes and never
its digest, so rounds that agree after `Round.erase` run alike. -/
theorem repRun_of_erase {P : Type} {g : Game P} {σ : List Exec.Kind} {dec : ℕ → ByteArray → List Msg}
    {rs rs' : List Round} {tail : List Msg} {p : P} (h : rs.map Round.erase = rs'.map Round.erase)
    (hr : RepRun g σ dec rs tail p) : RepRun g σ dec rs' tail p := by
  have key : ∀ l : List Round, RepRun g σ dec l tail p ↔ RepRun g σ dec (l.map Round.erase) tail p := fun l => by
    simp only [RepRun, List.forall_mem_map, Round.erase_coins, Round.erase_n, List.map_map, Function.comp_def,
      List.length_map, roundGroups_erase]
  exact (key rs').2 (by rw [← h]; exact (key rs).1 hr)

end FlockSoundness.Discharge.PlaysCoin

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A transcript that records one table's live session at the setup at `d`, through `keep`**: its proof files are
the outcome's, and the session its record decodes to through `keep` there has the outcome's root, link points and rep
streams' rounds up to their digests, read as one table's (`oneOf`). -/
theorem recordsLive_wrK1_of (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {mPts : ℕ} (o : Tables.LiveOutJ 1 mPts)
    {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I d o t) (hacc : AcceptsAtDK keep I (some d) t) :
    t.proofs = (oneOf o).proofs ∧
    ∀ spec sts sess, setupAtD I (some d) = .ok (spec, sts) →
      Record.decodeWithK keep With.std spec t.record = .ok sess →
      sess.rootB[0]! = (oneOf o).rootB ∧ ptsOf mPts sess.points = (oneOf o).pts ∧
      ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList.map Round.erase =
          (pick ((oneOf o).rep0, (oneOf o).rep1) r).map Round.erase := by
  obtain ⟨hpr, hrec⟩ := hrl hacc
  refine ⟨hpr, fun spec sts sess hset hdec => ?_⟩
  obtain ⟨hpts, hj⟩ := hrec spec sts sess hset hdec
  obtain ⟨hroot, hstr⟩ := hj 0
  refine ⟨hroot, hpts, fun r => ?_⟩
  obtain ⟨s, hs, hr⟩ := hstr r
  refine ⟨s, hs, ?_⟩
  rw [hr]
  fin_cases r <;> rfl

/-- **`wrPDK` records one table's live session at the setup at `d`** (`recordsLive_wrK1_of`). -/
theorem recordsLive_wrPDK1 (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {mPts : ℕ} (o : Tables.LiveOutJ 1 mPts)
    (hacc : AcceptsAtDK keep I (some d) (wrPDK keep I d o)) :
    (wrPDK keep I d o).proofs = (oneOf o).proofs ∧
    ∀ spec sts sess, setupAtD I (some d) = .ok (spec, sts) →
      Record.decodeWithK keep With.std spec (wrPDK keep I d o).record = .ok sess →
      sess.rootB[0]! = (oneOf o).rootB ∧ ptsOf mPts sess.points = (oneOf o).pts ∧
      ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList.map Round.erase =
          (pick ((oneOf o).rep0, (oneOf o).rep1) r).map Round.erase :=
  recordsLive_wrK1_of keep I d o (recordsLiveAtDK_wrPDK keep I d o) hacc

open DecodeZL Integrate in
/-- **The deployed plain decoders are right at a record writer for accepting with no opening off `C` without a
self-clash**, at one table: `decodesTable_wrPD_sc` through `keep`, at any writer `wr` whose transcripts record their
outcome at the setup at the draw `dj S` (`RecordsLiveAtDK`), on the outcomes whose transcript `verifyWithMK keep`
accepts there, whose root is `rootB₀`'s and on which the deployed plain reader looks up no column off `C` there
(`TxNoOffAtK`). -/
theorem decodesTable_wrK_sc_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (wr : Tables.LiveOutJ 1 mPts → Exec.Transcript)
    (hwr : ∀ o, RecordsLiveAtDK keep I (dj S) o (wr o))
    (j : Fin [Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leaf512 (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch o))
      (fun o => AcceptsAtDK keep I (some (dj S)) (wr o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep I (dj S) C (wr o))
      (fun ps _ => DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R ps)
      (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].get j)) j := by
  have hj : j.val = 0 := by
    have h := j.isLt
    simp only [List.length_cons, List.length_nil] at h
    omega
  obtain rfl : j = ⟨0, Nat.one_pos⟩ := Fin.ext hj
  rintro o ⟨hacc, hr₀, hno⟩
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  have h := PlaysCoin.decodesOfRuns (fun cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
    (fun o' => o' = oneOf o)
    (fun out => out.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (tablesAt (Reg := Reg) ht dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) out ∧
        ¬ SelfClashL H512 enc512 leaf512 (tablesAt (Reg := Reg) ht dj x₀ S R).sch out))
    (opnAt (tablesAt ht dj x₀) I dj S R) (decAt (tablesAt ht dj x₀) wOP wLIG S R) (fun _ _ => []) capOfRoot
    (fun pts r => repsOf (D := Dig) execArith (tablesAt (Reg := Reg) ht dj x₀ S R) pts r)
    (fun pts _ => PlaysCoin.repSpine Dig (tablesAt (Reg := Reg) ht dj x₀ S R) pts)
    (PlaysCoin.repsOf_spine Dig (tablesAt (Reg := Reg) ht dj x₀ S R)) ?_ (oneOf o) rfl
  · obtain ⟨pend, h0, h1, hok⟩ := h
    exact ⟨pend, h0, h1, hok⟩
  rintro o' rfl
  obtain ⟨hprf, hrec⟩ := recordsLive_wrK1_of keep I (dj S) o (hwr o) hacc
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
  have hcap : sess.rootB[0]! = rootB₀[0]! := hroot.trans hr₀
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
  obtain ⟨hver, hent⟩ := hk' hwf hr hm p₀ p₁ s₀ s₁ (pend 0) (pend 1) (capOfRoot rootB₀[0]!)
    (tableOutL H512 enc512 leaf512 (capOfRoot rootB₀[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wr o).proofs[0]!, (wr o).proofs[1]!]))
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
  refine ⟨(pend 0, pend 1), hrep 0, hrep 1, ?_⟩
  have hopn : opnAt (tablesAt (Reg := Reg) ht dj x₀) I dj S R (oneOf o).proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wr o).proofs[0]!, (wr o).proofs[1]!] := by
    rw [← hprf]
    unfold opnAt
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nX
    rw [extraClaims_length, hS']
    simp [stmtOf, regionsOf]
  have e₀ : (oneOf o).rootB = rootB₀[0]! := hr₀
  rw [hopn, e₀, hsch]
  refine ⟨hv', fun ⟨hoff, hsc⟩ => hno spec st sess sch s₀ s₁ hsu hdec hf hs₀ hs₁ ?_⟩
  rw [hsch] at hN
  exact Claims.Flock.exists_ne_of_entries H512 enc512 leaf512 (schedOf sch) (capOfRoot rootB₀[0]!) _ hN C _ _ rfl
    hver hent hoff hsc

open DecodeZL Integrate in
/-- **The deployed plain decoders are right at `wrPDK` for accepting with no opening off `C` without a self-clash**
(`decodesTable_wrK_sc_of` at `wrPDK`). -/
theorem decodesTable_wrPDK_sc (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (j : Fin [Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leaf512 (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch o))
      (fun o => AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep I (dj S) C (wrPDK keep I (dj S) o))
      (fun ps _ => DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R ps)
      (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].get j)) j :=
  decodesTable_wrK_sc_of keep ht hc ho dj x₀ hT hS R hN C rootB₀ (wrPDK keep I (dj S))
    (recordsLiveAtDK_wrPDK keep I (dj S)) j

open DecodeZL HiddenExec HiddenDec in
/-- **The deployed plain decoders are right at a record writer at hidden outputs, at a leaf form `F`**:
`decodesTable_wrK_sc_of` at the hidden tables and their openings `opnAtH` (`decodesTable_wrPD_scH` through `keep`). -/
theorem decodesTable_wrK_scHLf_of (keep : Bool → Bool) (F : LeafForm)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : F.TagsOk I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (wr : Tables.LiveOutJ 1 mPts → Exec.Transcript)
    (hwr : ∀ o, RecordsLiveAtDK keep I (dj S) o (wr o)) (j : Fin [tablesAtH (Reg := Reg) dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 F.lf cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 F.lf (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 F.lf (tablesAtH (Reg := Reg) dj x₀ S R).sch o))
      (fun o => AcceptsAtDK keep I (some (dj S)) (wr o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep I (dj S) C (wr o))
      (fun ps _ => opnAtH (tablesAtH dj x₀) I dj S R ps)
      (fun _ => decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([tablesAtH (Reg := Reg) dj x₀ S R].get j)) j := by
  have hj : j.val = 0 := by
    have h := j.isLt
    simp only [List.length_cons, List.length_nil] at h
    omega
  obtain rfl : j = ⟨0, Nat.one_pos⟩ := Fin.ext hj
  rintro o ⟨hacc, hr₀, hno⟩
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  have h := PlaysCoin.decodesOfRuns (fun cap pend os => tableOutL H512 enc512 F.lf cap pend os)
    (fun o' => o' = oneOf o)
    (fun out => out.accepted = true ∧ ¬ (Off₀L H512 enc512 F.lf (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) out ∧
        ¬ SelfClashL H512 enc512 F.lf (tablesAtH (Reg := Reg) dj x₀ S R).sch out))
    (opnAtH (tablesAtH dj x₀) I dj S R) (decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => []) capOfRoot
    (fun pts r => repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) pts r)
    (fun pts _ => PlaysCoin.repSpine Dig (tablesAtH (Reg := Reg) dj x₀ S R) pts)
    (PlaysCoin.repsOf_spine Dig (tablesAtH (Reg := Reg) dj x₀ S R)) ?_ (oneOf o) rfl
  · obtain ⟨pend, h0, h1, hok⟩ := h
    exact ⟨pend, h0, h1, hok⟩
  rintro o' rfl
  obtain ⟨hprf, hrec⟩ := recordsLive_wrK1_of keep I (dj S) o (hwr o) hacc
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
  have hcap : sess.rootB[0]! = rootB₀[0]! := hroot.trans hr₀
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
  obtain ⟨hver, hent⟩ := hk' hwf hr hm p₀ p₁ s₀ s₁ (pend 0) (pend 1) (capOfRoot rootB₀[0]!)
    (tableOutL H512 enc512 F.lf (capOfRoot rootB₀[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wr o).proofs[0]!, (wr o).proofs[1]!]))
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
  refine ⟨(pend 0, pend 1), hrep 0, hrep 1, ?_⟩
  have hopn : opnAtH (tablesAtH (Reg := Reg) dj x₀) I dj S R (oneOf o).proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wr o).proofs[0]!, (wr o).proofs[1]!] := by
    rw [← hprf]
    unfold opnAtH
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nX
    rw [extraClaims_length, hS']
    simp [stmtOf, regionsOf]
  have e₀ : (oneOf o).rootB = rootB₀[0]! := hr₀
  rw [hopn, e₀, hsch]
  refine ⟨hv', fun ⟨hoff, hsc⟩ => hno spec st sess sch s₀ s₁ hsu hdec hf hs₀ hs₁ ?_⟩
  rw [hsch] at hN
  exact Claims.Flock.exists_ne_of_entries H512 enc512 F.lf (schedOf sch) (capOfRoot rootB₀[0]!) _ hN C _ _ rfl
    hver hent hoff hsc

open DecodeZL HiddenExec HiddenDec in
/-- **The deployed plain decoders are right at a record writer at hidden outputs**: `decodesTable_wrK_sc_of` at the
hidden tables and their openings `opnAtH` (`decodesTable_wrPD_scH` through `keep`):
`decodesTable_wrK_scHLf_of` at `LeafForm.hm96`. -/
theorem decodesTable_wrK_scH_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (wr : Tables.LiveOutJ 1 mPts → Exec.Transcript)
    (hwr : ∀ o, RecordsLiveAtDK keep I (dj S) o (wr o)) (j : Fin [tablesAtH (Reg := Reg) dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch o))
      (fun o => AcceptsAtDK keep I (some (dj S)) (wr o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep I (dj S) C (wr o))
      (fun ps _ => opnAtH (tablesAtH dj x₀) I dj S R ps)
      (fun _ => decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([tablesAtH (Reg := Reg) dj x₀ S R].get j)) j :=
  decodesTable_wrK_scHLf_of keep LeafForm.hm96 ht hc dj x₀ hT hS R hN C rootB₀ wr hwr j

open DecodeZL HiddenExec HiddenDec in
/-- **The deployed plain decoders are right at `wrPDK` at hidden outputs** (`decodesTable_wrK_scH_of` at `wrPDK`). -/
theorem decodesTable_wrPDK_scH (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (j : Fin [tablesAtH (Reg := Reg) dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch o))
      (fun o => AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep I (dj S) C (wrPDK keep I (dj S) o))
      (fun ps _ => opnAtH (tablesAtH dj x₀) I dj S R ps)
      (fun _ => decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([tablesAtH (Reg := Reg) dj x₀ S R].get j)) j :=
  decodesTable_wrK_scH_of keep ht hc dj x₀ hT hS R hN C rootB₀ (wrPDK keep I (dj S))
    (recordsLiveAtDK_wrPDK keep I (dj S)) j

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw, at a record writer**:
`plain_hdecAt` through `keep`, at any writer `wr` whose transcripts record their outcome at the setup at `d`, from
`decodesTable_wrK_sc_of` at `d` and no drawn unit. -/
theorem plain_hdecAtK_of (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray)
    (wr : Tables.LiveOutJ 1 spec.mPts → Exec.Transcript) (hwr : ∀ o, RecordsLiveAtDK keep ins.read.I d o (wr o)) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (fun o => AcceptsAtDK keep ins.read.I (some d) (wr o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep ins.read.I d C (wr o))
      (fun ps _ => DecodeZL.opnAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j := by
  obtain ⟨ht, hc, ho, hT, hst, rfl, h22, h35⟩ := plain_setupH hs hin hset h hh hd
  obtain ⟨sch, hf⟩ : ∃ sch, Flock.fast100 st.m = some sch := by
    have e := Refine.fast100_schedOf st.m
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
  have hN : wordLen false x =
      2 ^ (lvOf (Integrate.tablesAt (Reg := Unit) ht (fun _ : Finset (Fin 0) => d) x₀ ∅ ()).sch 0).logLen := by
    rw [e]
    exact plain_wordLen h h22 h35 hsch
  have hdec := decodesTable_wrK_sc_of keep ht hc ho (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () hN C rootB₀
    wr hwr
  rw [e] at hdec
  exact hdec

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw, at hidden outputs and a leaf
form `F`, at a record writer**: `plain_hdecHAt` through `keep`, at any writer `wr` whose transcripts record their
outcome at the setup at `d`, from `decodesTable_wrK_scHLf_of`. -/
theorem plain_hdecHLfAtK_of (keep : Bool → Bool) (F : AtDraw.LeafForm)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = F.tag) {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray)
    (wr : Tables.LiveOutJ 1 spec.mPts → Exec.Transcript) (hwr : ∀ o, RecordsLiveAtDK keep ins.read.I d o (wr o)) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 F.lf cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 F.lf
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 F.lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (fun o => AcceptsAtDK keep ins.read.I (some d) (wr o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep ins.read.I d C (wr o))
      (fun ps _ => HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j := by
  obtain ⟨ht, hc, hT, hst, rfl, h22, h35⟩ := plain_setupHiddenT hs hin hset h hh hl hd
  obtain ⟨sch, hf⟩ : ∃ sch, Flock.fast100 st.m = some sch := by
    have e := Refine.fast100_schedOf st.m
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
  have hN : wordLen false x =
      2 ^ (lvOf (HiddenExec.tablesAtH (Reg := Unit) (fun _ : Finset (Fin 0) => d) x₀ ∅ ()).sch 0).logLen := by
    rw [e]
    exact plain_wordLen h h22 h35 hsch
  have hdec := decodesTable_wrK_scHLf_of keep F ht hc (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () hN C
    rootB₀ wr hwr
  rw [e] at hdec
  exact hdec

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw, at hidden outputs and
`hm96-sha512/v1` leaves, at a record writer**: `plain_hdecHAt` through `keep`, at any writer `wr` whose transcripts
record their outcome at the setup at `d`: `plain_hdecHLfAtK_of` at `LeafForm.hm96`. -/
theorem plain_hdecHAtK_of (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray)
    (wr : Tables.LiveOutJ 1 spec.mPts → Exec.Transcript) (hwr : ∀ o, RecordsLiveAtDK keep ins.read.I d o (wr o)) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (fun o => AcceptsAtDK keep ins.read.I (some d) (wr o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep ins.read.I d C (wr o))
      (fun ps _ => HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j :=
  plain_hdecHLfAtK_of keep AtDraw.LeafForm.hm96 hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay C rootB₀ wr
    hwr

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw**: `plain_hdecAt` with
`plainVerdictAtK keep` for `plainVerdictAt` (`plain_hdecAtK_of` at `wrPDK`). -/
theorem plain_hdecAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAtK keep ins.read.I d C rootB₀)
      (fun ps _ => DecodeZL.opnAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j :=
  plain_hdecAtK_of keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay C rootB₀ (wrPDK keep ins.read.I d)
    (recordsLiveAtDK_wrPDK keep ins.read.I d)

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw, at hidden outputs and
`hm96-sha512/v1` leaves**: `plain_hdecHAt` with `plainVerdictAtK keep` for `plainVerdictAt` (`plain_hdecHAtK_of` at
`wrPDK`). -/
theorem plain_hdecHAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAtK keep ins.read.I d C rootB₀)
      (fun ps _ => HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j :=
  plain_hdecHAtK_of keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay C rootB₀ (wrPDK keep ins.read.I d)
    (recordsLiveAtDK_wrPDK keep ins.read.I d)

end FlockDeploy.KS
