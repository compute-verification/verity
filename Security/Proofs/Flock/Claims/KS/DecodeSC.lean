import Proofs.Flock.Claims.Bridge.DecodePlain
import Proofs.Flock.Claims.KS.Entries

/-!
# The deployed plain decoder, with no opening off the table

`PlainLiveAccAt` asks the decoder to be right for the live verdict accepting with no level-0 opening off the table `C`
unless it self-clashes. At `wrP`, at one table, that holds on the outcomes whose transcript flock-verify accepts, whose
root is the live strategies' and on which the deployed plain reader looks up no column off `C` (`TxNoOff`):

* `wrP1_oneOf`: `wrP1` at a one-table session's outcome read as one table's is `wrP` at the session's outcome.
* `decodesTable_wrP_sc`: `decodesTable_wrP` with that verdict. The runs `decodesOn_exec` builds are the ones
  `accepts_opens₀L` reads, so an opening off `C` without a self-clash is a column off `C` the reader looks up
  (`Claims.Flock.exists_ne_of_entries`), which `TxNoOff` excludes.
-/

namespace Claims.Bridge.DecodePlain

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **`wrP1` at a one-table session's outcome is `wrP` at the session's**: the two choose among the same transcripts
(`recordsLiveAtP_oneOf`). -/
theorem wrP1_oneOf {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json) (S : Finset (Fin n))
    (R : Reg) (o : Tables.LiveOutJ 1 mPts) : wrP1 I dj S R (oneOf o) = wrP I dj (fun _ _ => 1) S R o := by
  have e : (fun t => Exec.Accepts I t ∧ RecordsLiveAtP I (dj S) (jOf (oneOf o)) t) =
      fun t => Exec.Accepts I t ∧ RecordsLiveAtP I (dj S) o t :=
    funext fun t => propext (and_congr_right fun _ => recordsLiveAtP_oneOf I (dj S) o t)
  unfold wrP1 wrP
  rw [e]

/-- **No column off `C` in a transcript**: wherever the record sets up the one table `st`, with its `fast100` schedule,
and decodes, the deployed plain reader looks up no column off `C` in the entries of the table's reps' stream rounds and
the first two proof files, at the table's shape, below level 0's `2^d`. -/
def TxNoOff (I : Exec.Inputs) {N : ℕ} (C : Fin N → List 𝔽) (t : Exec.Transcript) : Prop :=
  ∀ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : RunRecord) (sch : _root_.Flock.Schedule) (s₀ s₁ : StreamRec),
    Exec.setupOf I t.record = .ok (spec, #[st]) → Record.decode spec t.record = .ok sess →
    _root_.Flock.fast100 st.m = some sch →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ →
    ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds
          t.proofs[1]!)) p.val = some u ∧ u ≠ C p

open DecodeZL Integrate in
/-- **The deployed plain decoders are right at `wrP` for accepting with no opening off `C` without a self-clash**, at
one table: `decodesTable_wrP` on the outcomes whose transcript flock-verify accepts, whose root is `rootB₀`'s and on
which the deployed plain reader looks up no column off `C` (`TxNoOff`), with the verdict that the table accepts and does
not open a level-0 position off `C` against `rootB₀`'s cap unless it self-clashes. -/
theorem decodesTable_wrP_sc {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (j : Fin [Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leaf512 (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch o))
      (fun o => Exec.Accepts I (wrP I dj (fun _ _ => 1) S R o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOff I C (wrP I dj (fun _ _ => 1) S R o))
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
  rw [← wrP1_oneOf] at hacc hno
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
  obtain ⟨hd, hprf, hrec⟩ := recordsLive_wrP1 (Reg := Reg) (mPts := mPts) I dj S R (oneOf o) hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAt_of_accepts ht hc ho hh hc1 dj x₀ hacc hd hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_reps hh hrr hl hc1 hacc
  rw [hd] at hst
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
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opens₀L hacc
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
        #[(wrP1 I dj S R (oneOf o)).proofs[0]!, (wrP1 I dj S R (oneOf o)).proofs[1]!]))
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
    rw [← hrs]
    exact repRun_at op_carries lig_carries decodeProof_sizes _ st hwf sch hf (by rw [hms]; rfl) hS' hsch hpl
      (oneOf o).pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s'
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  refine ⟨(pend 0, pend 1), hrep 0, hrep 1, ?_⟩
  have hopn : opnAt (tablesAt (Reg := Reg) ht dj x₀) I dj S R (oneOf o).proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrP1 I dj S R (oneOf o)).proofs[0]!, (wrP1 I dj S R (oneOf o)).proofs[1]!] := by
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

end Claims.Bridge.DecodePlain
