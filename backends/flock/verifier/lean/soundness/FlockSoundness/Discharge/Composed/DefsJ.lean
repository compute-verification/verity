import FlockSoundness.Discharge.Composed.Defs
import FlockSoundness.Discharge.DecodeZLJ
import FlockSoundness.Discharge.Tables
import FlockSoundness.Discharge.Composed.TagsJ
import FlockSoundness.Protocol.Discharge.Composed.DefsJ

/-!
# One `--zk` sampled session of `J` tables: the pieces `zk_session_soundJ` reads

`zk_session_sound` (`Composed/Sound.lean`) reads one table. `flock-circuit serve|prove --zk --session-tables J` on the
CPU prover (`--gpu` refuses `J > 1`), and `flock-verify verify --zk --session-tables J`, run `J` tables in one session.
`Stmt.setupTables` then sets table `j` up under `tagsAt` (`Tags.tableIdentity count j`) at part `j` of the session's
subset draw (`partJson`, `accepts_tables`); at one table it is `zk_session_sound`'s table. Here are the pieces at `J`
tables, each the one-table piece's counterpart:

* `nTab`, `inputsJ`, `drawJ`, `partOf`, `partS`: the number of tables, table `j`'s inputs and draw, the table unit `u`
  is drawn into (its position among the draw's units over the parts' size), and the drawn units of table `j`.
* `DrawSetupZJ`, `tableZJ`, `planZJ`, `tabZJ`: `DrawSetupZK`, `tableAtZK`, `planZAt` and `tabZAt` at table `j`'s inputs
  and draw (where `verify --zk` sets none up, the refused table of `y₀`, a `DrawSetupZJ` at table 0's inputs).
* `tableShJ`, `planShJ`, `sitesShJ`, `sitesZJ`, `layZJ`, `rsZJ`, `rsZJ_computes`: `ZkBind`'s public shadow, sites,
  layout and rows (`At.lean`) at `J` tables.
* `liveOfZJ`, `DecodesOnZKJ`, `simZJ`, `zk_headline_exec_tables`: `Tables.liveOfJ`, `DecodesOnJ`, `simAuditJ` and
  `execRefines_tables` at the compiled `--zk` tables.
* `RunsOnZCJ`, `decodesOnZKJ_of_runsOnZCJ`: `RunsOnZC` and `decodesOnZK_of_runsOnZC` table by table.
* `TagsOkZJ`, `RecordsLiveZKJ`, `DrawOkZKJ`: `TagsOk` without `count ≤ 1`, and `RecordsLiveZK`, `DrawOkZK` at `J` tables.
* `runsZCJ_live`: `runsZC_live` at `J` tables.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession modelSession posOf DecodesTable simSession live_le_session
  AllOk repTapeJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-! ## The session's tables -/

theorem nTab_pos (I : Inputs) : 0 < nTab I := by
  unfold nTab
  split <;> omega

namespace DrawSetupZJ

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZJ I' mPts S d)

end DrawSetupZJ

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)

theorem planZJ_length (S : Finset (Fin n)) (R : Reg) : (planZJ (Reg := Reg) dj y₀ S R).length = nTab I :=
  List.length_ofFn

/-! ## The public shadow, the sites, the layout and the rows -/

variable (ht : I.tags.typed = false)

variable {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)

variable (hU : Built c.unit) {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none) (hscope : Layout.scopeOk c = true)

/-- **`hHm` at `J` tables' rows** (`rsZAt_computes`). -/
theorem rsZJ_computes : (rsZJ (Reg := Reg) dj y₀ ht hc hU hpub hpt hscope).Computes :=
  HmRowsZ.ofShadow_computes (planZJ_S dj y₀ ht) (decAgreeJ dj y₀ ht hc hU) _
    (Hm.keyedRowsOuts_computes (sitesShJ_parse dj y₀ ht hc) hc (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope)
      (sitesShJ_bcSite dj y₀ ht hc hU hpub hpt hscope))

end Shadow

/-! ## The compiled session of `J` tables, and the live one simulated in it -/

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K]

variable [Inhabited D] [Inhabited Sl]

/-- **The transfer at `J` compiled `--zk` tables** (`live_le_sessionBLJ` at `--zk`). -/
theorem live_le_sessionZJ (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (pL : LiveOutJ ts.length mPts → Prop) (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => o.accepted = true) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob pL (liveSession dg ts.length mPts N) Pr ≤
      prob (acceptedZC ts) (ZkLink.sessionZC A Hs ts)
        (simSessZJ A Hs ts pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessZJ
  rw [Exec.prob_cast]
  exact live_le_session _ _ _ pL opn dg dec one fin capOf N _ _ hdec Pr

variable (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dg : ByteArray → ByteArray)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (one : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → Bool)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (N : ℕ) (L : Law n)

variable {L}

/-- **R11 at `--zk` for sessions of `J` tables** (`execRefines_tables` at the compiled `--zk` tables). -/
theorem execRefinesZK_tables (hdec : DecodesOnZKJ A Hs planZ I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) :
    ExecRefines (auditLive L Reg (liveOfZJ dg planZ N wr)) τ (LiveAcceptsZK I) (fun _ => False)
      (audit L Reg (batchedSessionZC A Hs planZ)) (simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAcceptsZK I) ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((batchedSessionZC A Hs planZ (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simZJ batchedSessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOfZJ
  rw [prob_map]
  refine (live_le_sessionZJ A Hs (planZ (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
    (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1) _).trans_eq ?_
  congr 1
  funext outs
  classical
  exact propext decide_eq_true_iff.symm

/-- **The `--zk` headline at flock-verify's acceptance, for sessions of `J` tables**: a bound on the compiled batched
audit's acceptance with any event `Q` at `simZJ … τ` bounds flock-verify `--zk`'s acceptance with `Q`. -/
theorem zk_headline_exec_tables (hdec : DecodesOnZKJ A Hs planZ I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) (Q : Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q) (audit L Reg (batchedSessionZC A Hs planZ))
      (simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => LiveAcceptsZK I o ∧ Q) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ ≤ B := by
  have h := exec_le_of_refines (execRefinesZK_tables A Hs planZ I wr opn dg dec one fin capOf N hdec τ) Q hB
  rwa [prob_eq_zero_of_out (fun _ _ h => h) τ, add_zero] at h

end Session

/-! ## The message half, table by table -/

section Runs

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (L : Law n)

/-- **`RunsOnZC` at `J` tables**: at the law's draws, on every live outcome flock-verify `--zk` accepts, each table's
reps have a `RepRun` on their streams' rounds along `repShapeZ`, with `fin`'s messages as the tail, and the table's
compiled verdict accepts at the two runs' ends, with its cap and openings. -/
def RunsOnZCJ : Prop :=
  ∀ ω R (o : LiveOutJ (planZ (L.draw ω) R).length mPts), AcceptsZK I (wr (L.draw ω) R o) →
    ∀ j, ∃ pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl,
      PlaysCoin.RepRun (repZCAt A Hs ((planZ (L.draw ω) R).get j) o.pts 0)
        (repShapeZ 𝔽 K D ((planZ (L.draw ω) R).get j)
          (extraClaims ((planZ (L.draw ω) R).get j).S ((planZ (L.draw ω) R).get j).ptLocal mPts o.pts).length)
        (dec (L.draw ω) R j) (o.reps j 0) (fin (L.draw ω) R j 0 o.proofs) pend.1 ∧
      PlaysCoin.RepRun (repZCAt A Hs ((planZ (L.draw ω) R).get j) o.pts 1)
        (repShapeZ 𝔽 K D ((planZ (L.draw ω) R).get j)
          (extraClaims ((planZ (L.draw ω) R).get j).S ((planZ (L.draw ω) R).get j).ptLocal mPts o.pts).length)
        (dec (L.draw ω) R j) (o.reps j 1) (fin (L.draw ω) R j 1 o.proofs) pend.2 ∧
      (outZC Hs ((planZ (L.draw ω) R).get j) (capOf (L.draw ω) R o.rootB j) pend
        (opn (L.draw ω) R o.proofs j)).accepted = true

omit [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] in
open Classical in
/-- A compiled table that accepts has no rejected rep (`accZ_lig`). -/
theorem outZC_lig {T : TabZK 𝔽 mPts} {cap : Cap D} {pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl} {os : OpensZ 𝔽 D Sl}
    (h : (outZC Hs T cap pend os).accepted = true) :
    pend.1.lig ≠ PendL.reject ∧ pend.2.lig ≠ PendL.reject := by
  simp only [ZkLink.outZC, Bool.and_eq_true] at h
  exact ⟨ne_reject_of_verdict h.1.1, ne_reject_of_verdict h.1.2⟩

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
/-- **The decoder at the law's draws, table by table, from `RunsOnZCJ`** (`decodesOnZK_of_runsOnZC`), with the coin
step `true`. -/
theorem decodesOnZKJ_of_runsOnZCJ (h : RunsOnZCJ A Hs planZ I wr opn dec fin capOf L) :
    DecodesOnZKJ A Hs planZ I wr opn dec (fun _ _ _ _ => true) fin capOf L := by
  intro ω R j o ho
  obtain ⟨pend, h₀, h₁, hacc⟩ := h ω R o ho j
  obtain ⟨hl₀, hl₁⟩ := outZC_lig Hs hacc
  exact ⟨pend, PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 0 hl₀) h₀,
    PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 1 hl₁) h₁, hacc⟩

end Runs

/-! ## The live session's record, the coin server's draws, and the tags -/

/-- **The verify command's tags at any number of tables** (`TagsOk` without `count ≤ 1`): public outputs, retained
rounds, `hm96-sha512/v1` leaves. -/
def TagsOkZJ (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = false ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1"

/-! ## The decoder and openings at table `j` -/

/-! ## Table `j` where `verify --zk` sets it up -/

section Live

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json}

/-- A `DrawSetup` with `Zk.schedule`'s schedule at its `m` is a `DrawSetupZJ` (`drawSetupZK_of`). -/
theorem drawSetupZJ_of (ht' : I'.tags.typed = false) (x : DrawSetup I' mPts S d) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule x.st.m = .ok (sch, tPad)) :
    ∃ y : DrawSetupZJ I' mPts S d, y.x = x ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨hshape, hmsch, hpad⟩ := schedule_zk hz
  obtain ⟨-, hm35, -⟩ := schedule_spec hz
  have hk := (setupH_spec ht' x.h).2.2.2.2.2
  have h0 : sch.k0 ≤ 35 := by
    have := hshape.2.2.1
    simp only [schedOf] at this hmsch
    omega
  have hlen := layLen_le hm35 hk h0
  obtain ⟨k, hk', hk13⟩ := innerN_pow _ hlen
  have hN : ZkExec.innerN (layOf x.st sch) < 2 ^ 64 := by
    rw [layOf, hk']
    exact Nat.pow_lt_pow_right (by norm_num) (by omega)
  have hd : ZkExec.innerN (layOf x.st sch) ∣ 2 ^ 64 := by
    rw [layOf, hk']
    exact Nat.pow_dvd_pow 2 (by omega)
  exact ⟨⟨x, sch, tPad, hz, hshape, hmsch, hpad, hN, innerN_ge _,
    fun D => prCoin_innerPos_le _ _ hd D⟩, rfl, rfl, rfl⟩

/-- A table's `--zk` table is the same at every `DrawSetupZJ` of it (`tableZK_eq`). -/
theorem DrawSetupZJ.table_eq (y₁ y₂ : DrawSetupZJ I' mPts S d) : y₁.table = y₂.table := by
  obtain ⟨⟨st₁, h₁, D₁, hD₁, hS₁, hg₁, hb₁, m₁, k₁, a₁, b₁, r₁, e₁, hs₁, hm₁, hp₁⟩, sch₁, t₁, hz₁, _, _, _, _, _, _⟩ := y₁
  obtain ⟨⟨st₂, h₂, D₂, hD₂, hS₂, hg₂, hb₂, m₂, k₂, a₂, b₂, r₂, e₂, hs₂, hm₂, hp₂⟩, sch₂, t₂, hz₂, _, _, _, _, _, _⟩ := y₂
  obtain rfl : st₁ = st₂ := Except.ok.inj (h₁.symm.trans h₂)
  obtain ⟨rfl, rfl⟩ := Prod.ext_iff.1 (Except.ok.inj (hz₁.symm.trans hz₂))
  rfl

theorem DrawSetupZJ.table_seam (y : DrawSetupZJ I' mPts S d) :
    ZkTabSeam y.table (Setup.ofCircuit y.x.st) y.sch y.tPad :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem DrawSetupZJ.table_lay (y : DrawSetupZJ I' mPts S d) : ZkTabLay y.table (Setup.ofCircuit y.x.st) y.sch := by
  constructor <;> intros <;> rfl

end Live

/-- **`verify --zk` sets table `j` up under its tags at its draw**: at the record's draw `d`, `setupTables` at
`zkInputs I` has `nTab I` tables, table `j` setupH's statement under `tagsAt` at `drawJ` (`setupTables_ok` at one table,
`setupTables_parts` at several). -/
theorem setupOf_tableJ {I : Inputs} {record : String} {d : Lean.Json} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hh : I.tags.hiddenOutputs = false) (hd : recordDraw record = some d)
    (hs : setupOf (zkInputs I) record = .ok (spec, sts)) :
    sts.size = nTab I ∧ ∀ (j : ℕ) (hj : j < sts.size),
      Stmt.setupH (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
        (some (drawJ I d j)) I.partition I.program = .ok sts[j] := by
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  by_cases hc : I.count ≤ 1
  · rcases setupTables_ok hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
    · refine ⟨by simp [nTab, hc], fun j hj => ?_⟩
      have : j = 0 := by simpa using hj
      subst this
      have e1 : tagsAt (zkTags I) I.count 0 = zkTags I := by simp [tagsAt, hc]
      have e2 : drawJ I d 0 = d := by simp [drawJ, hc]
      rw [e1, e2, ← hd]
      exact hst
    · exact absurd hc' (by change ¬ 1 < I.count; omega)
  · have hc1 : 1 < I.count := by omega
    obtain ⟨-, -, population, units, hD1, -, -, -, hsz, hall⟩ := Tables.setupTables_parts hh' hc1 hs
    obtain ⟨u, hu, -, hpop, hun⟩ := hD1 _ hd
    refine ⟨by simp only [nTab, hc, ite_false]; exact hsz, fun j hj => ?_⟩
    have hpart : drawJ I d j = Tables.partDraw population units I.count j := by
      simp only [drawJ, hc, ite_false]
      unfold partJson
      rw [hu]
      dsimp only
      rw [hpop, hun]
    have e1 : tagsAt (zkTags I) I.count j = { zkTags I with identity := (zkTags I).tableIdentity I.count j } := by
      simp [tagsAt, hc]
    rw [e1, hpart]
    exact hall j hj

open Classical in
theorem outZC_of_accZ {D Sl : Type} [Inhabited D] [Inhabited Sl]
    {Hs : Hashes 𝔽 D Sl} {T : Finset (Fin n) → Reg → TabZK 𝔽 mPts} {S : Finset (Fin n)} {R : Reg}
    {caps : Fin 1 → Cap D} {pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl} {os : OpensZ 𝔽 D Sl}
    (h : accZ T S R (ZkExec.outZ Hs T S R caps pend os) = true) : (outZC Hs (T S R) (caps 0) pend os).accepted = true := by
  unfold accZ ZkExec.outZ at h
  exact (of_decide_eq_true h) ⟨0, by simp⟩

/-- **`RunsZC` at the live `--zk` tables of a session of `J` tables, at the law's draws** (`runsZC_live` at `J`
tables): on untyped inputs under the tags `TagsOkZJ` fixes, with `wr` the coin server's record of the live session and
each of the law's draws setting each table up at `mPts` link points, `RunsOnZCJ` holds at the tables `planZJ` under
`hm96-sha512/v1`, with the decoder `decAtZJ`, the openings `opnAtZJ`, no tail and table `j`'s cap from `root_B[j]`. -/
theorem runsZCJ_live (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  intro ω R o hacc j
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hd, hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  have hpts2 := setupTables_points hh' hs
  have hsr : spec.retainRounds = true := ((setupTables_retain hh' hs).trans (zkTags_retainRounds I)).trans hrr
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJ hh hd hs
  have hjn : j.val < nTab I := planZJ_length dj y₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, c, hz0, hv0, hv1⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨-, -, sch₀, hf⟩ := schedule_spec hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ _ht j.val
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetup_of_setupH_g (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hst hf hD
    ((hDraw ω ⟨j.val, hjn⟩).1 D hD) ((hDraw ω ⟨j.val, hjn⟩).2 _ hst)
  obtain ⟨y, hyx, hysch, hytp⟩ := drawSetupZJ_of ht' x (sch := schs[j.val].1) (tPad := schs[j.val].2)
    (by rw [hx]; exact hz')
  have hyst : y.x.st = sts[j.val] := by rw [hyx]; exact hx
  have hTy : (planZJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = y.table := by
    have hget : (planZJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = tableZJ dj y₀ (L.draw ω) ⟨j.val, hjn⟩ := by
      exact List.get_ofFn _ j
    rw [hget]
    unfold tableZJ
    rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩)]
    exact DrawSetupZJ.table_eq _ _
  rw [← hysch, ← hytp, ← hyst] at hv0 hv1
  rw [← hyst] at hst
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hs hdecS
  obtain ⟨hroot, hstreams⟩ := hrj j
  obtain ⟨hp2, hmp⟩ := hpts2
  have hpd := decode_points hdecS
  have hmP : sts[j.val].mPts = mPts := by rw [← hyst]; exact y.x.hmPts
  have hsz' : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp j.val hj).symm.trans hmP)
  have hwf := setupH_wf y.x.h
  have hExt : ExtraRealizes (Setup.ofCircuit y.x.st) y.table.S sess.points
      (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts) := by
    have := ofCircuit_extra y.x.st hwf.1 hwf.2 sess.points (hpd.1.trans hp2) (by rw [y.x.hmPts]; exact hsz')
    rw [y.x.hmPts, hpts] at this
    exact this
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupH_tagsAt_hm96 ((zkTags_merkleLeaf I).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam := y.table_seam
  have hlay := y.table_lay
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold y.table.S :=
    ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hwfA := decode_roundsWF hdecS
  have hretA := decode_allRetained hsr hdecS
  obtain ⟨s₀, ex₀, zp₀, tr₀, hf₀, hex₀, hdz₀, -, hrt₀, hrun₀, hk₀, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hf₁, hex₁, hdz₁, -, hrt₁, hrun₁, hk₁, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv1
  have e₁ : ex₁ = ex₀ := Except.ok.inj (hex₁.symm.trans hex₀)
  subst ex₁
  obtain ⟨hlen₀, hcr₀⟩ := hExt ex₀ hex₀
  have hm₀ : s₀ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₀)
  have hm₁ : s₁ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₁)
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀
    s₀.rounds _ tr₀ y.hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀ (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁
    s₁.rounds _ tr₁ y.hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁ (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess j.val _ _ c y.hz hms hseam hlay hfold hExt hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZJ y.table = ex₀.size := by
    unfold nXZJ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZJ (planZJ (Reg := Reg) dj y₀) (L.draw ω) R j =
      decZ (wRestZ (Setup.ofCircuit y.x.st).m (Setup.ofCircuit y.x.st).kLog (2 + ex₀.size) (schedOf y.sch) y.tPad
        (layZ (Setup.ofCircuit y.x.st) y.sch).len) := by
    unfold decAtZJ
    rw [hTy, hseam.hm, hseam.hk, hseam.hsch, hseam.ht, hseam.hnp, hnX]
  obtain ⟨e₀, he₀, hr₀⟩ := hstreams 0
  obtain ⟨e₁, he₁, hr₁⟩ := hstreams 1
  have e₆ : e₀ = s₀ := Option.some.inj (he₀.symm.trans hf₀)
  subst e₀
  have e₇ : e₁ = s₁ := Option.some.inj (he₁.symm.trans hf₁)
  subst e₁
  have hopn : opnAtZJ (planZJ (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs j =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnAtZJ
    rw [hst]
    dsimp only
    rw [y.hz]
    dsimp only
    rw [hTy, hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · rw [hTy, hdecEq, ← hr₀]
    exact hrr₀
  · rw [hTy, hdecEq, ← hr₁]
    exact hrr₁
  · dsimp only
    rw [hTy, hopn, ← hroot]
    exact outZC_of_accZ hacc'

end FlockSoundness.Discharge.Composed
