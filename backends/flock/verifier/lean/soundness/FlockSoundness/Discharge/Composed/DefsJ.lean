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

namespace DrawSetupZJ

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZJ I' mPts S d)

end DrawSetupZJ

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)

/-! ## The public shadow, the sites, the layout and the rows -/

variable (ht : I.tags.typed = false)

variable {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)

variable (hU : Built c.unit) {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none) (hscope : Layout.scopeOk c = true)

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

end Runs

/-! ## The live session's record, the coin server's draws, and the tags -/

/-! ## The decoder and openings at table `j` -/

/-! ## Table `j` where `verify --zk` sets it up -/

section Live

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json}

end Live

end FlockSoundness.Discharge.Composed
