import FlockSoundness.Discharge.ZkExec.Statements
import FlockSoundness.Discharge.Exec.Headline

/-!
# R11 at `--zk`, one table, and the `--zk` headline at flock-verify's acceptance

* `execRefinesZK_one : ExecRefinesZKOne …`: the live `--zk` audit simulated in the compiled `--zk` audit, by the live
  coin server's simulation (`live_le`, `Refine/LiveSim.lean`) at each draw, with `bad = ⊥`, given `DecodesZK`.
* `zk_headline_exec_one : ZkHeadlineExecOne …`: from it, any bound on the compiled audit's acceptance with an event of
  the simulated prover bounds flock-verify `--zk`'s acceptance with that event (`exec_le_of_refines`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Game Audit Exec Refine Flock

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
  sorry

theorem zk_headline_exec_one : ZkHeadlineExecOne G out acc I wr opn dg dec one fin capOf N L := by
  sorry

end FlockSoundness.Discharge.ZkExec
