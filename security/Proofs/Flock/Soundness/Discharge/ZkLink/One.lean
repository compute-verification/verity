module

public import Proofs.Flock.Soundness.Discharge.ZkLink.Compiled
public import Proofs.Flock.Soundness.Refine.LiveSim
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Discharge.Exec.Live
import Proofs.Flock.Soundness.Refine.Plays

/-! One compiled `--zk` table as `Refine.modelTable`. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession Refine

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

omit [Fintype K] in
theorem sessionZC_one (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) :
    modelTable (repZCAt A Hs T) (fun (caps : Fin 1 → Cap D) pend os => [outZC Hs T (caps 0) pend os]) =
      sessionZC A Hs [T] := by
  show Game.send (Fin 1 → Cap D) (fun caps => Game.coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
      (interleave (repZCAt A Hs T pts 0) (repZCAt A Hs T pts 1)).bind fun pend =>
        Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret [outZC Hs T (caps 0) pend os]) =
    Game.send (Fin 1 → Cap D) fun caps => Game.coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
      batch ((gamesZC A Hs [T] caps).map (· pts))
  congr 1
  funext caps
  congr 1
  funext pts
  simp only [gamesZC, List.ofFn_succ, List.ofFn_zero, List.map_cons, List.map_nil, batch, Plays.interleave_ret,
    Exec.map_map']
  rw [show (fun a : OutZC 𝔽 D Sl => (a, ([] : List (OutZC 𝔽 D Sl))).1 :: (a, ([] : List (OutZC 𝔽 D Sl))).2) =
    fun a => [a] from rfl]
  unfold tabAfterZC
  rw [Exec.bind_map]
  rfl

variable [Inhabited D] [Inhabited Sl]

omit [Fintype K] in
open Classical in
theorem batchedSessionZC_one (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) {n : ℕ} {Reg : Type}
    (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (S : Finset (Fin n)) (R : Reg) :
    (modelTable (repZCAt A Hs (T S R)) (fun (caps : Fin 1 → Cap D) pend os => [outZC Hs (T S R) (caps 0) pend os])).map
        (fun outs => decide (acceptedZC [T S R] outs)) =
      batchedSessionZC A Hs (fun S R => [T S R]) S R := by
  rw [sessionZC_one]
  rfl

end FlockSoundness.Discharge.ZkLink
