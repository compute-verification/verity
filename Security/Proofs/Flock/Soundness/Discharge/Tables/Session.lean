import Proofs.Flock.Soundness.Discharge.Tables.Sim
import Proofs.Flock.Soundness.Discharge.Exec.LiveOn

/-!
# R11 for sessions of `J` tables: the live audit simulated in the salted audit

`execRefines_on` (`Exec/LiveOn.lean`) is R11 for one-table plans. This file is R11 for any plan: at each draw the live
session is `liveSession` over the plan's `J` tables, written out as a session directory by `wr`, and the audit's
session is `batchedSessionL` on the plan.

* `sessionBL_eq_modelSession`: the salted session on the tables `ts` is `modelSession`'s, with table `j`'s reps
  `repsOf A (ts.get j)` and output `tableOutL`.
* `live_le_sessionBLJ`: against any live prover of `liveSession`, the live verdict is no likelier than the salted
  session's acceptance (every table's) against the simulated prover (`live_le_session` at `sessionBL_eq_modelSession`).
* `DecodesOnJ`: the decoder is right on the sessions flock-verify accepts, at the law's draws, table by table: on every
  live outcome flock-verify accepts, table `j`'s reps run on the tapes decoded from its streams `(j, 0)`, `(j, 1)` and
  its proof files, and the salted table's verdict on those runs holds. One `DecodesTable` per draw, registration and
  table; at one table it is `DecodesOn`'s.
* `simAuditJ`, `execRefines_tables`: the audit strategy built from a live one, and R11 at it with `bad = ⊥`.
-/

namespace FlockSoundness.Discharge.Tables

open Game Model Accounting Refine Audit Audit.Partition Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

omit [Fintype K] in
/-- **The salted session on the tables `ts` is `modelSession`'s**: `Commit`'s caps, the link points, every table's
reps concurrently and its salted openings, the tables concurrently. -/
theorem sessionBL_eq_modelSession (A : Arith 𝔽 K) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (ts : List (TabSpec mPts)) :
    modelSession (J := ts.length) (fun _ cap pend os => tableOutL H E Lf cap pend os)
      (fun j => repsOf (D := D) A (ts.get j)) = sessionBL A H E Lf ts := by
  unfold modelSession sessionBL sessAfterBL posOf
  congr 1
  funext caps
  congr 1
  funext pts
  simp only [gamesBL, List.map_ofFn]
  rfl

variable [Nonempty D] [Inhabited D] [Inhabited Sl]

/-- The simulated session strategy against a live prover of `liveSession` over the tables `ts`. -/
noncomputable def simSessJ (A : Arith 𝔽 K) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (ts : List (TabSpec mPts)) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL H E Lf cap pend os) (fun _ o => o.accepted = true) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) : Strategy (sessionBL A H E Lf ts) :=
  sessionBL_eq_modelSession A H E Lf ts ▸
    (simSession (fun _ cap pend os => tableOutL H E Lf cap pend os) (fun _ o => o.accepted = true) dfltOutL pL opn dg
      dec one fin capOf N (fun j => repsOf (D := D) A (ts.get j))
      (fun j _ _ => Model.allInh_repC A _ (ts.get j).S (ts.get j).sch (ts.get j).hm _) hdec).strategy Pr

/-- **The transfer at `J` tables, for the salted session.** Against any live prover of `liveSession` over the tables
`ts`, the live verdict `pL` holds with probability at most the salted session's acceptance of every table against the
simulated prover, whenever every table's decoder is right on accepted sessions. -/
theorem live_le_sessionBLJ (A : Arith 𝔽 K) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (ts : List (TabSpec mPts)) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL H E Lf cap pend os) (fun _ o => o.accepted = true) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob pL (liveSession dg ts.length mPts N) Pr ≤
      prob (acceptedBL ts) (sessionBL A H E Lf ts) (simSessJ A H E Lf ts pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessJ
  rw [prob_cast]
  exact live_le_session _ _ _ pL opn dg dec one fin capOf N _ _ hdec Pr

/-! ## The audit -/

variable {n : ℕ} {Reg : Type}

/-- **The live audit's session at a draw**: `liveSession` over the plan's tables, its outcome written out as the
session directory by `wr` (the record the challenger writes, and the proof files). -/
def liveOfJ (dg : ByteArray → ByteArray) (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (N : ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (S : Finset (Fin n)) (R : Reg) : Game Transcript :=
  (liveSession dg (plan S R).length mPts N).map (wr S R)

variable (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (I : Inputs)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 D × Salts Sl)
  (dg : ByteArray → ByteArray)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
  (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap D) (N : ℕ)

/-- **The decoder is right on the sessions flock-verify accepts, at the law's draws, table by table** (R11b, named):
at every draw `L.draw ω`, registration `R` and table `j` of the plan, on every live outcome whose session directory
flock-verify accepts, table `j`'s reps run on the tapes decoded from its streams and its proof files, and the salted
table's verdict on those runs, with the cap `root_B` decodes to and the openings and salts the proof files decode to,
holds. -/
def DecodesOnJ (L : Law n) : Prop :=
  ∀ ω R (j : Fin (plan (L.draw ω) R).length),
    DecodesTable (fun _ cap pend os => tableOutL H E Lf cap pend os) (fun _ o => o.accepted = true)
      (fun o => Accepts I (wr (L.draw ω) R o)) (opn (L.draw ω) R) (dec (L.draw ω) R) (one (L.draw ω) R)
      (fin (L.draw ω) R) (capOf (L.draw ω) R) (fun j => repsOf (D := D) execArith ((plan (L.draw ω) R).get j)) j

variable {L : Law n}

/-- **The simulated audit strategy**, from the decoder at the law's draws: the live prover's registration, and at each
draw the simulated session strategy against its live session. -/
noncomputable def simAuditJ (hdec : DecodesOnJ H E Lf plan I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfJ dg plan N wr))) :
    Strategy (audit L Reg (batchedSessionL H E Lf plan)) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    (simSessJ execArith H E Lf (plan (L.draw ω) τ.1) (fun o => Accepts I (wr (L.draw ω) τ.1 o)) (opn (L.draw ω) τ.1)
      dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1)
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **R11 for sessions of `J` tables.** Against any live prover `τ` of the live audit whose session at each draw is
`liveSession` over the plan's tables, written out by `wr`, flock-verify's acceptance is no likelier than the salted
audit's acceptance against the simulated prover `simAuditJ … τ`, with no live event added (`bad = ⊥`), whenever the
decoder is right on accepted sessions at the law's draws, table by table (`DecodesOnJ`). -/
theorem execRefines_tables (hdec : DecodesOnJ H E Lf plan I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfJ dg plan N wr))) :
    ExecRefines (auditLive L Reg (liveOfJ dg plan N wr)) τ (LiveAccepts I) (fun _ => False)
      (audit L Reg (batchedSessionL H E Lf plan)) (simAuditJ H E Lf plan I wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOfJ dg plan N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAccepts I) ((liveOfJ dg plan N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((batchedSessionL H E Lf plan (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simAuditJ H E Lf plan I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAuditJ batchedSessionL
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOfJ
  rw [prob_map]
  refine (live_le_sessionBLJ execArith H E Lf (plan (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
    (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1) _).trans_eq ?_
  congr 1
  funext outs
  classical
  exact propext decide_eq_true_iff.symm

end FlockSoundness.Discharge.Tables
