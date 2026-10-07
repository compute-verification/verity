import Proofs.Flock.Soundness.Discharge.Composed.DefsJ

set_option Elab.async false

/-!
# One `--zk` sampled session of `J` tables at any acceptance

Composed's `DecodesOnZKJ`, `simZJ`, `execRefinesZK_tables`, `zk_headline_exec_tables`, `RunsOnZCJ` and
`decodesOnZKJ_of_runsOnZCJ` (`Composed/DefsJ.lean`) read flock-verify's acceptance `AcceptsZK I` only as the live
outcomes' event. With `--public-inputs F` the event is another (`AcceptsZKP I own`). Each definition here is the
original at an event `acc` on transcripts, with the same proof, as `ZkPubIn/Composed.lean` is at one table; at
`acc := AcceptsZK I` it is the original.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL Composed
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl]

section Session

variable (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts))
  (acc : Exec.Transcript → Prop)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dg : ByteArray → ByteArray)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (one : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → Bool)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (N : ℕ) (L : Law n)

/-- **The decoder is right at the law's draws, table by table, at the event `acc`** (`DecodesOnZKJ` with `acc` for
`AcceptsZK I`). -/
def DecodesOnZKJA : Prop :=
  ∀ ω R (j : Fin (planZ (L.draw ω) R).length),
    DecodesTable (outJ Hs (planZ (L.draw ω) R)) (fun _ o => o.accepted = true)
      (fun o => acc (wr (L.draw ω) R o)) (opn (L.draw ω) R) (dec (L.draw ω) R) (one (L.draw ω) R)
      (fin (L.draw ω) R) (capOf (L.draw ω) R) (repJ A Hs (planZ (L.draw ω) R)) j

variable {L}

/-- **The simulated audit strategy in the compiled batched session of `J` tables, at the event `acc`** (`simZJ`). -/
noncomputable def simZJA (hdec : DecodesOnZKJA A Hs planZ acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) :
    Strategy (audit L Reg (batchedSessionZC A Hs planZ)) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    (simSessZJ A Hs (planZ (L.draw ω) τ.1) (fun o => acc (wr (L.draw ω) τ.1 o)) (opn (L.draw ω) τ.1)
      dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1)
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **R11 at `--zk` for sessions of `J` tables, at the event `acc`** (`execRefinesZK_tables`). -/
theorem execRefinesZK_tablesA (hdec : DecodesOnZKJA A Hs planZ acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) :
    ExecRefines (auditLive L Reg (liveOfZJ dg planZ N wr)) τ (fun o => acc o.2.2) (fun _ => False)
      (audit L Reg (batchedSessionZC A Hs planZ)) (simZJA A Hs planZ acc wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => acc o.2.2) ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((batchedSessionZC A Hs planZ (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simZJA A Hs planZ acc wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simZJA batchedSessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOfZJ
  rw [prob_map]
  refine (live_le_sessionZJ A Hs (planZ (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
    (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1) _).trans_eq ?_
  congr 1
  funext outs
  classical
  exact propext decide_eq_true_iff.symm

/-- **The `--zk` headline at flock-verify's acceptance `acc`, for sessions of `J` tables**
(`zk_headline_exec_tables`): a bound on the compiled batched audit's acceptance with any event `Q` at `simZJA … τ`
bounds `acc` with `Q`. -/
theorem zk_headline_exec_tablesA (hdec : DecodesOnZKJA A Hs planZ acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) (Q : Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q) (audit L Reg (batchedSessionZC A Hs planZ))
      (simZJA A Hs planZ acc wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => acc o.2.2 ∧ Q) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ ≤ B := by
  have h := exec_le_of_refines (execRefinesZK_tablesA A Hs planZ acc wr opn dg dec one fin capOf N hdec τ) Q hB
  rwa [prob_eq_zero_of_out (fun _ _ h => h) τ, add_zero] at h

end Session

section Runs

variable (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts))
  (acc : Exec.Transcript → Prop)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (L : Law n)

/-- **`RunsOnZCJ` at the event `acc`**: at the law's draws, on every live outcome in `acc`, each table's reps have a
`RepRun` on their streams' rounds along `repShapeZ`, with `fin`'s messages as the tail, and the table's compiled
verdict accepts at the two runs' ends, with its cap and openings. -/
def RunsOnZCJA : Prop :=
  ∀ ω R (o : LiveOutJ (planZ (L.draw ω) R).length mPts), acc (wr (L.draw ω) R o) →
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

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
/-- **The decoder at the law's draws, table by table, from `RunsOnZCJA`** (`decodesOnZKJ_of_runsOnZCJ`), with the
coin step `true`. -/
theorem decodesOnZKJA_of_runsOnZCJA (h : RunsOnZCJA A Hs planZ acc wr opn dec fin capOf L) :
    DecodesOnZKJA A Hs planZ acc wr opn dec (fun _ _ _ _ => true) fin capOf L := by
  intro ω R j o ho
  obtain ⟨pend, h₀, h₁, hacc⟩ := h ω R o ho j
  obtain ⟨hl₀, hl₁⟩ := outZC_lig Hs hacc
  exact ⟨pend, PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 0 hl₀) h₀,
    PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 1 hl₁) h₁, hacc⟩

end Runs

end FlockSoundness.Discharge.ZkPubIn
