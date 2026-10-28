import Proofs.Flock.Claims.KS.PlainDrawKDefs
import Proofs.Flock.Soundness.Discharge.Exec.VerifyKJ

/-!
# The plain verdict at the statement's draw, at the record's parsed value

`PlainDrawKDefs`' predicates read a transcript's record as text (`Exec.Transcript.record`), and flock-verify parses it
first. The recursion's last step at the parsed value (`FlockDeploy.nextJ`) runs the verdict on a parsed value, which
need not be any text's parse, so the KS and Bits route is stated here at a parsed value `j` (`TranscriptJ`), through
`verifyJWithMK keep` and `Record.decodeJWithK keep` at SHA-512 (`With.std`):

* `AcceptsAtDJK`, `RecordsLiveAtDJK`, `TxNoOffAtJK`, `wrPDJK`, `plainVerdictAtJK`: `AcceptsAtDK`, `RecordsLiveAtDK`,
  `TxNoOffAtK`, `wrPDK` and `plainVerdictAtK` at a parsed value.
* `verifyJWithMK_nil`, `not_acceptsAtDJK_nil`, `recordsLiveAtDJK_wrPDJK`, `acceptsAtDJK_wrPDJK`: the writer's facts.
* **The transfer** (`acceptsAtDK_iff_J`, `recordsLiveAtDK_iff_J`, `txNoOffAtK_iff_J`): at a record that parses to `j`,
  each text form is the form here at `j`; `acceptsAtDK_iff_exists_J` adds that a text form accepts only where its
  record parses. So `acceptsAtDJK_wrPDJK_of_K` moves an accepting record of the outcome at the text forms to `wrPDJK`.
-/

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open FlockSoundness.Discharge.Exec (verifyWithMK_ok_iff)
open FlockSoundness.Refine (ptsOf shapeOf ebind_ok)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A session at the record's parsed value**: the parsed record `j` and the proof files, in `spec.repStreams` order
(`Exec.Transcript` after the parse). -/
structure TranscriptJ where
  j : Json
  proofs : Array ByteArray

/-- **flock-verify accepts at the setup at the draw `dr`, at the parsed value, through `keep`**: `AcceptsAtDK` with
`verifyJWithMK keep` on the parsed record for `verifyWithMK keep` on its text. -/
def AcceptsAtDJK (keep : Bool → Bool) (I : Exec.Inputs) (dr : Option Json) (t : TranscriptJ) : Prop :=
  ∃ spec sts, setupAtD I dr = .ok (spec, sts) ∧
    verifyJWithMK keep With.std spec (sts.map _root_.Flock.Setup.ofCircuit) t.j t.proofs = .ok ()

/-- **`RecordsLiveAtDK` at the parsed value**: where `verifyJWithMK keep` accepts `t` at the session it sets up at
`d`, its proof files are the outcome `o`'s, and the session `t.j` decodes to through `keep` there has `o`'s link points
and, at each of `o`'s tables, `o`'s `root_B` and rep streams' rounds up to their digests (`Round.erase`). -/
def RecordsLiveAtDJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts)
    (t : TranscriptJ) : Prop :=
  AcceptsAtDJK keep I (some d) t →
    t.proofs = o.proofs ∧
    ∀ spec sts sess, setupAtD I (some d) = .ok (spec, sts) →
      Record.decodeJWithK keep With.std spec t.j = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList.map Round.erase = (o.reps j r).map Round.erase

/-- **`TxNoOffAtK` at the parsed value**: no column off `C` at the session `t.j` decodes to through `keep`. -/
def TxNoOffAtJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {N : ℕ} (C : Fin N → List 𝔽) (t : TranscriptJ) :
    Prop :=
  ∀ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : RunRecord) (sch : _root_.Flock.Schedule) (s₀ s₁ : StreamRec),
    setupAtD I (some d) = .ok (spec, #[st]) → Record.decodeJWithK keep With.std spec t.j = .ok sess →
    _root_.Flock.fast100 st.m = some sch →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ →
    ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds
          t.proofs[1]!)) p.val = some u ∧ u ≠ C p

open Classical in
/-- **The record writer at the draw `d`, at the parsed value, through `keep`**: at the live outcome `o`, a parsed
session `verifyJWithMK keep` accepts at the setup at `d` that records `o` there up to the rounds' digests, where one
exists, else a session without proof files, which the verdict refuses at every `keep` (`verifyJWithMK_nil`). -/
noncomputable def wrPDJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) :
    TranscriptJ :=
  if h : ∃ t, AcceptsAtDJK keep I (some d) t ∧ RecordsLiveAtDJK keep I d o t then h.choose else ⟨Json.null, #[]⟩

/-- **The verdict at a parsed value refuses a session without proof files, at every `keep`** (`verifyWithMK_nil`'s
proof). -/
theorem verifyJWithMK_nil (keep : Bool → Bool) (hs : Hashes _root_.Flock.V) (spec : SessionSpec) (tabs : Array Setup)
    (j : Json) : verifyJWithMK keep hs spec tabs j #[] ≠ .ok () := by
  intro h
  unfold verifyJWithMK at h
  simp only [] at h
  by_cases c1 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c1] at h
    obtain ⟨_, h', _⟩ := ebind_ok h
    cases h'
  rw [ite_eq_right c1] at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  have c3 : ((#[] : Array ByteArray).size != spec.repStreams.size) = true := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at c1
    simp only [List.size_toArray, List.length_nil, Custody.repStreams_size, bne_iff_ne, ne_eq]
    omega
  rw [ite_eq_left c3] at h
  obtain ⟨_, h', _⟩ := ebind_ok h
  cases h'

theorem not_acceptsAtDJK_nil (keep : Bool → Bool) (I : Exec.Inputs) (dr : Option Json) (j : Json) :
    ¬ AcceptsAtDJK keep I dr ⟨j, #[]⟩ := by
  rintro ⟨spec, sts, -, h⟩
  exact verifyJWithMK_nil keep _ _ _ _ h

/-- **`wrPDJK` records the live session at the setup at `d`, up to the rounds' digests.** -/
theorem recordsLiveAtDJK_wrPDJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ}
    (o : Tables.LiveOutJ J mPts) : RecordsLiveAtDJK keep I d o (wrPDJK keep I d o) := by
  unfold wrPDJK
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h (not_acceptsAtDJK_nil _ _ _ _)

/-- **`verifyJWithMK keep` accepts what `wrPDJK` writes** wherever it accepts some parsed session at the setup at `d`
that records the outcome there. -/
theorem acceptsAtDJK_wrPDJK {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {J mPts : ℕ}
    {o : Tables.LiveOutJ J mPts} {t : TranscriptJ} (hA : AcceptsAtDJK keep I (some d) t)
    (hR : RecordsLiveAtDJK keep I d o t) : AcceptsAtDJK keep I (some d) (wrPDJK keep I d o) := by
  have hex : ∃ t, AcceptsAtDJK keep I (some d) t ∧ RecordsLiveAtDJK keep I d o t := ⟨t, hA, hR⟩
  unfold wrPDJK
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

/-! ## The transfer from the text forms -/

/-- **At a record that parses to `j`, `AcceptsAtDK` is `AcceptsAtDJK` at `j`** (`verifyWithMK_of_parse`). -/
theorem acceptsAtDK_iff_J {keep : Bool → Bool} {I : Exec.Inputs} {dr : Option Json} {text : String} {j : Json}
    (proofs : Array ByteArray) (h : Json.parse text = .ok j) :
    AcceptsAtDK keep I dr ⟨text, proofs⟩ ↔ AcceptsAtDJK keep I dr ⟨j, proofs⟩ := by
  have hv : ∀ spec tabs, verifyWithMK keep With.std spec tabs text proofs =
      verifyJWithMK keep With.std spec tabs j proofs := fun spec tabs =>
    verifyWithMK_of_parse keep With.std spec tabs proofs h
  simp only [AcceptsAtDK, AcceptsAtDJK, hv]

/-- **`AcceptsAtDK` holds exactly where the record parses and `AcceptsAtDJK` holds at its parsed value.** -/
theorem acceptsAtDK_iff_exists_J {keep : Bool → Bool} {I : Exec.Inputs} {dr : Option Json} (t : Exec.Transcript) :
    AcceptsAtDK keep I dr t ↔ ∃ j, Json.parse t.record = .ok j ∧ AcceptsAtDJK keep I dr ⟨j, t.proofs⟩ := by
  refine ⟨fun ⟨spec, sts, hs, hv⟩ => ?_, fun ⟨j, hj, hA⟩ => (acceptsAtDK_iff_J t.proofs hj).2 hA⟩
  obtain ⟨j, hj, hvj⟩ := (verifyWithMK_ok_iff keep With.std spec _ t.record t.proofs).1 hv
  exact ⟨j, hj, spec, sts, hs, hvj⟩

/-- **At a record that parses to `j`, `RecordsLiveAtDK` is `RecordsLiveAtDJK` at `j`** (`decodeWithK_of_parse`). -/
theorem recordsLiveAtDK_iff_J {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {J mPts : ℕ}
    {o : Tables.LiveOutJ J mPts} {text : String} {j : Json} (proofs : Array ByteArray) (h : Json.parse text = .ok j) :
    RecordsLiveAtDK keep I d o ⟨text, proofs⟩ ↔ RecordsLiveAtDJK keep I d o ⟨j, proofs⟩ := by
  have hdec : ∀ spec, Record.decodeWithK keep With.std spec text = Record.decodeJWithK keep With.std spec j :=
    fun spec => Record.decodeWithK_of_parse keep With.std spec h
  simp only [RecordsLiveAtDK, RecordsLiveAtDJK, acceptsAtDK_iff_J proofs h, hdec]

/-- **At a record that parses to `j`, `TxNoOffAtK` is `TxNoOffAtJK` at `j`.** -/
theorem txNoOffAtK_iff_J {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N : ℕ} {C : Fin N → List 𝔽}
    {text : String} {j : Json} (proofs : Array ByteArray) (h : Json.parse text = .ok j) :
    TxNoOffAtK keep I d C ⟨text, proofs⟩ ↔ TxNoOffAtJK keep I d C ⟨j, proofs⟩ := by
  have hdec : ∀ spec, Record.decodeWithK keep With.std spec text = Record.decodeJWithK keep With.std spec j :=
    fun spec => Record.decodeWithK_of_parse keep With.std spec h
  simp only [TxNoOffAtK, TxNoOffAtJK, hdec]

/-- **An accepting record of the outcome at the text forms makes `wrPDJK` accept**: its record parses, and at the
parsed value it accepts and records the outcome (`acceptsAtDK_iff_exists_J`, `recordsLiveAtDK_iff_J`). -/
theorem acceptsAtDJK_wrPDJK_of_K {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {J mPts : ℕ}
    {o : Tables.LiveOutJ J mPts} {t : Exec.Transcript} (hA : AcceptsAtDK keep I (some d) t)
    (hR : RecordsLiveAtDK keep I d o t) : AcceptsAtDJK keep I (some d) (wrPDJK keep I d o) := by
  obtain ⟨j, hj, hAj⟩ := (acceptsAtDK_iff_exists_J t).1 hA
  exact acceptsAtDJK_wrPDJK hAj ((recordsLiveAtDK_iff_J t.proofs hj).1 hR)

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The plain verdict at the draw `d`, at the parsed value, through `keep`**: `verifyJWithMK keep` accepts what
`wrPDJK` writes for the outcome at the setup at `d`, the outcome's root is `rootB₀`'s, and the deployed plain reader
looks up no column off `C` in that parsed session there (`TxNoOffAtJK`). `plainVerdictAtK` at the parsed value. -/
def plainVerdictAtJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {N mPts : ℕ} (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (o : Tables.LiveOutJ 1 mPts) : Prop :=
  AcceptsAtDJK keep I (some d) (wrPDJK keep I d o) ∧ o.rootB[0]! = rootB₀[0]! ∧
    TxNoOffAtJK keep I d C (wrPDJK keep I d o)

end FlockDeploy.KS
