import Proofs.Flock.Soundness.Discharge.ZkCoins.Hybrid
import Proofs.Flock.Soundness.Discharge.ZkCoins.Erase
import Proofs.Flock.Soundness.Discharge.ZkCoins.Reads

/-!
# The batched `--zk` session on C-Flock ZK's coin tree

`session_coins_sound` (`SessionCoinsSound`) and `session_coins_hm96` (`SessionCoinsHm96`): `committed_le` (resp.
`committed_le_hm96`) on `sessionZKC`, which opens each committed round at most once (`distinct_sessionZKC`) and, drawn
live, has `sessionZK`'s value (`er_sessionZKC`, `value_erase`), which table `j`'s bound bounds (`value_sessionZK_le`,
`sessionZK_sound`'s argument). The session has `1 + 2·J·R` committed rounds (`card_sessIdx`).
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Model ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

/-- **The live session's value** for table `j`'s failure is at most table `j`'s bound (`sessionZK_sound`, for the
optimal prover). -/
theorem value_sessionZK_le (A : Model.Arith F K) (hA : A.Correct) (ts : List (TabZK F mPts)) (j : Fin ts.length) :
    value (fun outs => acceptedZK ts outs ∧ ¬ (ts.get j).Committed A (outs.getD j dfltZK).2) (sessionZK A ts) ≤
      (ts.get j).bound := by
  set Q : Bool × Oracle F → Prop := fun out => out.1 = true ∧ ¬ (ts.get j).Committed A out.2 with hQ
  refine (value_mono (q := fun outs => Q (outs.getD j dfltZK)) (fun outs h => ⟨h.1 j, h.2⟩) _).trans ?_
  simp only [sessionZK, value_send]
  refine iSup_le fun Cs => le_trans ?_ ((ts.get j).after_sound A hA (Cs j))
  simp only [value_coin]
  refine Finset.sum_le_sum fun pts _ => mul_le_mul_right ?_ _
  have hs := ofFn_split (fun i => (ts.get i).after A (Cs i) pts) j
  have hl : ((List.ofFn fun i => (ts.get i).after A (Cs i) pts).take j).length = j := by simp [j.isLt.le]
  have h := value_batch_getD_le Q dfltZK ((ts.get j).after A (Cs j) pts)
    ((List.ofFn fun i => (ts.get i).after A (Cs i) pts).drop (j + 1))
    ((List.ofFn fun i => (ts.get i).after A (Cs i) pts).take j)
  rw [hl, ← hs] at h
  exact h

theorem session_coins_sound : SessionCoinsSound := by
  intro F K _ _ _ _ _ _ mPts A hA ts j Blk Salt B Cs D _ _ _ _ _ _ dc R dig My cs H δ₁ hT τ
  refine (committed_le _ (sessionZKC dc R A ts) (distinct_sessionZKC dc R A ts) dig My cs H δ₁ hT τ).trans
    (add_le_add ?_ (le_of_eq ?_))
  · rw [value_erase (er_sessionZKC dc R A ts)]
    exact value_sessionZK_le A hA ts j
  · rw [card_sessIdx]

theorem session_coins_hm96 : SessionCoinsHm96 := by
  intro F K E _ _ _ _ _ _ _ _ _ mPts A hA hE ts j Blk Cs D _ _ _ dc R dig cs hcs H τ
  refine (committed_le_hm96 hE
    (fun _ (outs : List (Bool × Oracle F)) => acceptedZK ts outs ∧ ¬ (ts.get j).Committed A (outs.getD j dfltZK).2)
    (fun _ => sessionZKC dc R A ts) (fun _ => distinct_sessionZKC dc R A ts) dig cs hcs H τ).trans
    (add_le_add ?_ (le_of_eq ?_))
  · simp only [avg_const]
    rw [value_erase (er_sessionZKC dc R A ts)]
    exact value_sessionZK_le A hA ts j
  · rw [card_sessIdx]

end FlockSoundness.Discharge.ZkCoins
