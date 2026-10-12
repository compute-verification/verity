import Proofs.Flock.C3.RegLevels.SrcTie
import Proofs.Flock.C3.Compose.LastRead
import Proofs.Flock.C3.RegLevels.Read
import Proofs.Flock.Soundness.Refine.LigeritoRun
import Proofs.Flock.Soundness.Refine.Final

/-!
# n9: R's Ligerito ends in a coin (H's `LigWhole`) from the source's tie

At an accepting `LevelTie` the executable's Ligerito refines to the model's compiled Ligerito played on its messages
and the rounds it consumes (`ligerito_refines`, whose final program leaves the transcript as it is, `finalProg_ok`),
and that play is a play on `u.lg` (`Read.playsT_lig_of_tape`). A whole play of compiled Ligerito at a `fast100`
schedule ends in a coin (`LigEndsInCoin`, named: the last level's queries, `qbShape lv false`, close with their coin),
so `u.lg` does.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

/-- **A whole play of compiled Ligerito at a `fast100` schedule ends in a coin** (named): such a schedule has more than
one level, and the last level's queries close with their coin. -/
def LigEndsInCoin : Prop :=
  ∀ (m : ℕ) (sch : Flock.Schedule), Flock.fast100 m = some sch → ∀ (T₀ : GF128) (b : ℕ → GF128) (ts : List Step)
    (pend : PendL GF128 Dig), PlaysT (ligeritoC (D := Dig) execArith 2 (schedOf sch) T₀ b) ts pend →
      ∃ (l : List Step) (c : Game.Msg), ts = l ++ [.inr c]

/-- **An accepting tie's Ligerito ends in a coin.** -/
theorem ligWhole_of_levelTie (hend : LigEndsInCoin) {cap₀ : Cap Dig} {e : ExecHolo} {u : C3Run e}
    (h : LevelTie cap₀ e u) : ∃ (l : List Step) (c : Game.Msg), u.lg = l ++ [.inr c] := by
  obtain ⟨msId, m, sch, Tw, sws, _nc, cap, p, _rt, _rt', hf, hsch, -, -, hwf, hlig, hk, hl, hr, -⟩ := h
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨pend, _initCh, _levels, _stF, _stF', tF, -, -, hpl, -, -, -, -, -, -, -, hfin, -⟩ :=
    ligerito_refines (Flock.MerkleScheme.sha512 msId) sch Tw cap p (Flock.basis sws) 2 u.opened.b capDig
      (fun _ ℓ j => (((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf, []))
      (fast100_dims m sch hf).1 hlv hα1 (fun _ _ _ _ => rfl) hwf hlig
  rw [← (finalProg_ok hfin).2.1] at hpl
  rw [← hsch] at hk
  exact hend m sch hf _ _ _ _ (FlockC3.RegLevels.Read.playsT_lig_of_tape hpl hk hl hr)

noncomputable section

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **H's `LigWhole` from the source's tie**, at any registered cap. -/
theorem ligWhole_of_srcLevelTie (hend : LigEndsInCoin)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {cap₀ : Cap Dig} (h : SrcLevelTie zs dg N k Rw src cap₀) :
    FlockC3.Compose.LigWhole zs dg N k Rw src :=
  fun R ω t R₂ τ o hA => ligWhole_of_levelTie hend (h R ω t R₂ τ o hA)

end

end FlockC3.RegLevels
