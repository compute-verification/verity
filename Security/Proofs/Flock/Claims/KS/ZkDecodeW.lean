import Proofs.Flock.Claims.KS.ZkDrawJ
import Proofs.Flock.Claims.KS.ZkOffSC

/-!
# The deployed `--zk` decoders, with no witness-lane opening off the table

`DecodeSC` at `--zk`, at the setup at the draw, with the level-0 event on the witness lanes (`OffW₀Z`, which
`session_sound_of_table_accZ_w` reads). Where the deployed masked reader looks up no column of table `j` off `C`, every
accepted output table `j`'s reps play to on the decoded tapes opens no level-0 position off `C` on a witness lane unless
it self-clashes:

* `ZkTxNoOff`: wherever the setup at the draw has table `j` with `Zk.schedule`'s schedule and the record decodes to a
  session with table `j`'s two rep streams, the reader looks up no column off `C` in the reps' `zkRepColsOf` entries
  below level 0's `2^d`.
* `not_offW_of_entries`: an output whose level-0 openings are those entries at its own cap (`OpensEntries`) opens no
  position off `C` on a witness lane without a self-clash, in a transcript with `ZkTxNoOff`
  (`Claims.Flock.exists_ne_of_zk_entries`).
* `runsAtZCJ_live_offW`, `runsAtZCJH_live_offW`: `runsAtZCJ_live_entries` and `runsAtZCJH_live_entries` at that
  verdict at table `j₀`, on the outcomes `verify --zk` accepts at the setup at the draw whose transcript has
  `ZkTxNoOff` at table `j₀`, at public and hidden outputs.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **No column of table `j` off `C` in a transcript**: wherever the setup at the draw `d` has a table `j`, whose
schedule is `Zk.schedule`'s, and the record decodes to a session with table `j`'s two rep streams, the deployed masked
reader looks up no column off `C` in the reps' `zkRepColsOf` entries at `2·|regions|` extra claims on the streams'
rounds and table `j`'s proof files, below level 0's `2^d`. -/
def ZkTxNoOff (I : Inputs) (d : Lean.Json) (j : ℕ) {N : ℕ} (C : Fin N → List 𝔽) (t : Exec.Transcript) : Prop :=
  ∀ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session) (sch : Flock.Schedule) (tPad : ℕ)
    (s₀ s₁ : StreamRec) (h : j < sts.size),
    FlockDeploy.KS.setupAt (zkInputs I) (some d) = .ok (spec, sts) → Record.decode spec t.record = .ok sess →
    Zk.schedule sts[j].m = .ok (sch, tPad) →
    sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ →
    sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ →
    ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.zkRepColsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₀.rounds
          t.proofs[2 * j]! ++
        Claims.Flock.zkRepColsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₁.rounds
          t.proofs[2 * j + 1]!)) p.val = some u ∧ u ≠ C p

/-- **No witness-lane opening off `C` without a self-clash where the reader looks up no column off `C`**: an output
whose level-0 openings are the reader's entries at table `j` at its own cap (`OpensEntries`), in a transcript with
`ZkTxNoOff` at table `j`, opens no level-0 position off the oracle of `C` on a witness lane unless it self-clashes. -/
theorem not_offW_of_entries {I : Inputs} {d : Lean.Json} {t : Exec.Transcript} {j : ℕ} {T : TabZK 𝔽 mPts}
    {cap : Cap Dig} {out : OutZC 𝔽 Dig (List UInt8)} {N : ℕ} (hN : N = 2 ^ (lvOf T.sch 0).logLen)
    (C : Fin N → List 𝔽) (hcap : out.cap₀ = cap) (he : OpensEntries I d t j T cap out) (hno : ZkTxNoOff I d j C t) :
    ¬ (OffW₀Z hs512 T cap (Claims.Flock.oracleOf C) out ∧ ¬ SelfClashZ hs512 T out) := by
  rintro ⟨hoff, hsc⟩
  obtain ⟨spec, sts, sess, sch, tPad, s₀, s₁, h, hs, hd, hz, hf₀, hf₁, hver, hent⟩ := he
  exact hno spec sts sess sch tPad s₀ s₁ h hs hd hz hf₀ hf₁
    (Claims.Flock.exists_ne_of_zk_entries hs512 T cap out hN C _ _ hcap hver hent hoff hsc)

/-- **`runsAtZCJ_live` with no witness-lane opening off `C` at table `j₀` unless it self-clashes**: on the outcomes
`verify --zk` accepts at the setup at the law's draw whose transcript has `ZkTxNoOff` at table `j₀`, each table's
output is accepted and, at table `j₀` at its own cap, opens no level-0 position off `C` on a witness lane without a
self-clash (`runsAtZCJ_live_entries`, `not_offW_of_entries`). -/
theorem runsAtZCJ_live_offW (I : Inputs) (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts (L.draw ω)) (j₀ : ℕ) {N : ℕ} (C : Fin N → List 𝔽) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZJ dj y₀) (opnAtZJ (planZJ dj y₀) I dj) (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o) ∧ ZkTxNoOff I (dj S) j₀ C (wr S R o))
      (fun S R o j out => out.accepted = true ∧ (j.val = j₀ → N = 2 ^ (lvOf ((planZJ dj y₀ S R).get j).sch 0).logLen →
        ¬ (OffW₀Z hs512 ((planZJ dj y₀ S R).get j) (capOfRoot o.rootB[j.val]!) (Claims.Flock.oracleOf C) out ∧
          ¬ SelfClashZ hs512 ((planZJ dj y₀ S R).get j) out))) := by
  intro ω R o ho j
  obtain ⟨hacc, hno⟩ := ho
  obtain ⟨pend, h₀, h₁, hok, he⟩ := runsAtZCJ_live_entries I ht dj y₀ wr L hT hRec hDraw ω R o hacc j
  refine ⟨pend, h₀, h₁, hok, fun hj hN => ?_⟩
  subst hj
  exact not_offW_of_entries hN C rfl he hno

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`runsAtZCJH_live` with no witness-lane opening off `C` at table `j₀` unless it self-clashes**, under hidden
outputs: `runsAtZCJ_live_offW` from `runsAtZCJH_live_entries`. -/
theorem runsAtZCJH_live_offW (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) (j₀ : ℕ) {N : ℕ} (C : Fin N → List 𝔽) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZHJ dj y₀) (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o) ∧ ZkTxNoOff I (dj S) j₀ C (wr S R o))
      (fun S R o j out => out.accepted = true ∧ (j.val = j₀ → N = 2 ^ (lvOf ((planZHJ dj y₀ S R).get j).sch 0).logLen →
        ¬ (OffW₀Z hs512 ((planZHJ dj y₀ S R).get j) (capOfRoot o.rootB[j.val]!) (Claims.Flock.oracleOf C) out ∧
          ¬ SelfClashZ hs512 ((planZHJ dj y₀ S R).get j) out))) := by
  intro ω R o ho j
  obtain ⟨hacc, hno⟩ := ho
  obtain ⟨pend, h₀, h₁, hok, he⟩ := runsAtZCJH_live_entries I ht hc dj y₀ wr L hT hRec hDraw ω R o hacc j
  refine ⟨pend, h₀, h₁, hok, fun hj hN => ?_⟩
  subst hj
  exact not_offW_of_entries hN C rfl he hno

end FlockSoundness.Discharge.ZkHidden
