import Proofs.Flock.Soundness.Discharge.ZkExec.Statements
import Proofs.Flock.Soundness.Discharge.Exec.Headline

/-!
# R11 at `--zk`, one table, and the `--zk` headline at flock-verify's acceptance

* `execRefinesZK_one : ExecRefinesZKOne …`: the live `--zk` audit simulated in the compiled `--zk` audit, by the live
  challenger's simulation (`live_le`, `Refine/LiveSim.lean`) at each draw, with `bad = ⊥`, given `DecodesZK`.
* `zk_headline_exec_one : ZkHeadlineExecOne …`: from it, any bound on the compiled audit's acceptance with an event of
  the simulated prover bounds flock-verify `--zk`'s acceptance with that event (`exec_le_of_refines`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Game Audit Exec Refine Flock
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {P Cp Op Out : Finset (Fin n) → Reg → Type}
  (G : ∀ S R, (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (P S R))
  (out : ∀ S R, Cp S R → P S R × P S R → Op S R → Out S R) (acc : ∀ S R, Out S R → Bool)
  [∀ S R, Nonempty (Op S R)] (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ S R, Array ByteArray → Op S R) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ S R, ByteArray → Cp S R) (N : ℕ) (L : Law n)

theorem execRefinesZK_one : ExecRefinesZKOne G out acc I wr opn dg dec one fin capOf N L := by
  intro hinh hdec τ
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOf dg mPts N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAcceptsZK I) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((sessionZC G out acc (L.draw ω) τ.1).map fun ok => (ok, L.draw ω))
        ((simAuditZK G out acc I wr opn dg dec one fin capOf N hinh hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAuditZK sessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOf
  rw [prob_map]
  exact live_le (out (L.draw ω) τ.1) _ (fun x => acc (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg
    (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (G (L.draw ω) τ.1)
    (hinh (L.draw ω) τ.1) (hdec (L.draw ω) τ.1) _

theorem zk_headline_exec_one : ZkHeadlineExecOne G out acc I wr opn dg dec one fin capOf N L := by
  intro hinh hdec τ Q B hB
  have h := exec_le_of_refines (execRefinesZK_one G out acc I wr opn dg dec one fin capOf N L hinh hdec τ) Q hB
  rwa [prob_eq_zero_of_out (fun _ _ h => h) τ, add_zero] at h

end FlockSoundness.Discharge.ZkExec
