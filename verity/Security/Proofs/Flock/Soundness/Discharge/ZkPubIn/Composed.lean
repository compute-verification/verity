import Proofs.Flock.Soundness.Discharge.Composed.Defs

/-!
# One `--zk` sampled session at any acceptance

Composed's `DecodesOnZK`, `simAuditZKOn`, `simZ`, `execRefinesZK_on`, `zk_headline_exec_on` and ZkExec's `RunsOnZC`,
`decodes_of_runsOnZC` read flock-verify's acceptance `AcceptsZK I` only as the live outcomes' event. With
`--public-inputs F` the event is another (`AcceptsZKP I own`, `AcceptsZKRP I reg own`: `Stmt.setupHidden` refuses a
circuit with public input ports when no `F` is given, so `AcceptsZK I` is empty there). Each definition here is the
original at an event `acc` on transcripts, with the same proof; at `acc := AcceptsZK I` it is the original.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL Composed
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl]

section On

variable (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts)
  (acc : Exec.Transcript → Prop)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) (L : Law n)

/-- **The decoder is right at the law's draws, at the event `acc`** (`DecodesOnZK` with `acc` for `AcceptsZK I`). -/
def DecodesOnZKA : Prop :=
  ∀ ω R, Decodes (outz Hs T (L.draw ω) R) (fun o => acc (wr (L.draw ω) R o))
    (fun x => accz T (L.draw ω) R x = true) (opn (L.draw ω) R) (dec (L.draw ω) R) (one (L.draw ω) R)
    (fin (L.draw ω) R) (capOf (L.draw ω) R) (Gz A Hs T (L.draw ω) R)

variable {L}

/-- `simAuditZKOn` at the event `acc`. -/
noncomputable def simAuditZKOnA (hdec : DecodesOnZKA A Hs T acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T))) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    ((simTable (outz Hs T (L.draw ω) τ.1) (fun o => acc (wr (L.draw ω) τ.1 o))
      (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1)
      (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (Gz A Hs T (L.draw ω) τ.1)
      (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1)).strategy
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **The simulated audit strategy in the compiled batched session, at the event `acc`** (`simZ`). -/
noncomputable def simZA (hdec : DecodesOnZKA A Hs T acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) : Strategy (audit L Reg (batchedSessionZC A Hs (planOf T))) :=
  cast (congrArg (fun g => Strategy (audit L Reg g)) (sessionZC_planOf A Hs T))
    (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ)

/-- **R11 at `--zk` from the decoder at the law's draws, at the event `acc`** (`execRefinesZK_on`). -/
theorem execRefinesZK_onA (hdec : DecodesOnZKA A Hs T acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (fun o => acc o.2.2) (fun _ => False)
      (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
      (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOf dg mPts N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => acc o.2.2) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true)
        ((ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) (L.draw ω) τ.1).map fun ok => (ok, L.draw ω))
        ((simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAuditZKOnA ZkExec.sessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOf
  rw [prob_map]
  exact live_le (outz Hs T (L.draw ω) τ.1) _ (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg
    (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (Gz A Hs T (L.draw ω) τ.1)
    (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1) _

/-- **The `--zk` headline at flock-verify's acceptance `acc`, from the decoder at the law's draws**
(`zk_headline_exec_on`): a bound on the compiled batched audit's acceptance with any event `Q` at `simZA … τ` bounds
`acc` with `Q`. -/
theorem zk_headline_exec_onA (hdec : DecodesOnZKA A Hs T acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) (Q : Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q) (audit L Reg (batchedSessionZC A Hs (planOf T)))
      (simZA A Hs T acc wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => acc o.2.2 ∧ Q) (auditLive L Reg (liveOf dg mPts N wr)) τ ≤ B := by
  have hc : ∀ (g : Finset (Fin n) → Reg → Game Bool)
      (h : ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = g),
      prob (fun o => o.1 = true ∧ Q) (audit L Reg g)
          (cast (congrArg (fun g => Strategy (audit L Reg g)) h)
            (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ)) =
        prob (fun o => o.1 = true ∧ Q) (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
          (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ) := by
    intro g h
    subst h
    rfl
  have hB' := (hc _ (sessionZC_planOf A Hs T)).symm.trans_le hB
  have h := exec_le_of_refines (execRefinesZK_onA A Hs T acc wr opn dg dec one fin capOf N hdec τ) Q hB'
  rwa [prob_eq_zero_of_out (fun _ _ h => h) τ, add_zero] at h

/-- **`RunsZC` at the law's draws, at the event `acc`** (`ZkExec.RunsOnZC` with `acc` for `AcceptsZK I`). -/
def RunsOnZCA (L : Law n) : Prop :=
  ∀ ω R (o : LiveOut mPts), acc (wr (L.draw ω) R o) →
    ∃ pend : ZkLink.PendZ 𝔽 D Sl × ZkLink.PendZ 𝔽 D Sl,
      PlaysCoin.RepRun (repZ A Hs T (L.draw ω) R o.pts 0) (shapeZ (K := K) (D := D) T (L.draw ω) R o.pts)
        (dec (L.draw ω) R) o.rep0 (fin (L.draw ω) R 0 o.proofs) pend.1 ∧
      PlaysCoin.RepRun (repZ A Hs T (L.draw ω) R o.pts 1) (shapeZ (K := K) (D := D) T (L.draw ω) R o.pts)
        (dec (L.draw ω) R) o.rep1 (fin (L.draw ω) R 1 o.proofs) pend.2 ∧
      accZ T (L.draw ω) R
        (outZ Hs T (L.draw ω) R (capOf (L.draw ω) R o.rootB) pend (opn (L.draw ω) R o.proofs)) = true

omit [Fintype K] in
/-- **The decoder at the law's draws from `RunsOnZCA`** (`decodesOnZK_of_runsOnZC`, the coin step `true`). -/
theorem decodesOnZKA_of_runsOnZCA (h : RunsOnZCA A Hs T acc wr opn dec fin capOf L) :
    DecodesOnZKA A Hs T acc wr opn dec (fun _ _ _ => true) fin capOf L := by
  intro ω R
  refine PlaysCoin.decodesOfRunsAt (outZ Hs T (L.draw ω) R) _ _ (opn (L.draw ω) R) (dec (L.draw ω) R)
    (fin (L.draw ω) R) (capOf (L.draw ω) R) (repZ A Hs T (L.draw ω) R) fun o ho => ?_
  obtain ⟨pend, h₀, h₁, hacc⟩ := h ω R o ho
  obtain ⟨hl₀, hl₁⟩ := accZ_lig Hs T hacc
  exact ⟨pend, _, _, repZCAt_spineAt A Hs (T (L.draw ω) R) o.pts 0 hl₀, h₀,
    repZCAt_spineAt A Hs (T (L.draw ω) R) o.pts 1 hl₁, h₁, hacc⟩

end On

end FlockSoundness.Discharge.ZkPubIn
