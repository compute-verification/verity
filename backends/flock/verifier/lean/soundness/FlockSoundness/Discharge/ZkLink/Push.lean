import FlockSoundness.Discharge.ZkLink.Compiled
import FlockSoundness.Game.Batch
import FlockSoundness.Protocol.Discharge.ZkLink.Push

/-!
# The compiled `--zk` table with its digests read as caps

A Lock's table nodes have one type of table and one type of commitment. The compiled `--zk` table commits three kinds
of object: every level by a cap (`Cap D`), and per rep the pads by a root (`D`) and `τ` by a leaf (`D`). The primed
table (`tabAfterZCP`) receives the pads root and `τ`'s commitment as caps and reads them at entry `0`
(`repZCP`, `innerGameCP`); otherwise it is the compiled table.

`expect_sessAtZ_push`: inside any session, the compiled table and the primed one have the same expectation of every
function of the session's outputs, the primed prover sending the constant cap at every digest. The proof is a Lock
(`Lock (Cap D) D`) between the two (`pushSess`) whose leaves relate equal outputs while every cap sent is constant at
its digest (`Pushed`), which the pushing chooser keeps (`expect_push`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Pushing digests to caps -/

section Push

variable {α β I : Type}

/-- **A pushed prover has the same expectations**: along a Lock whose leaves relate equal outputs while every cap is
pushed, the prover that sends the constant cap at every digest has the compiled prover's expectation of every `f`. -/
theorem expect_push {get : I → ℕ → Option (Cap D × D)} (f : α → ℝ) :
    ∀ {i : I} {g g' : Game α} (L : Lock (Cap D) D get (fun i a b => Pushed get i → a = b) i g g')
      (σ' : Strategy g'), Pushed get i → expect f g' σ' = expect f g (Lock.oracleStrat pushC L σ')
  | _, _, _, .ret h, _, hi => by
      simp only [expect]
      rw [h hi]
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [expect, Lock.oracleStrat]
      congr 1
      exact Finset.sum_congr rfl fun c _ => expect_push f (L c) (σ' c) hi
  | _, _, _, .send _ L, σ', hi => expect_push f (L σ'.1) σ'.2 hi
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s upd _ k' _ hset hkeep L, σ', hi => by
      simp only [expect, Lock.oracleStrat]
      refine expect_push f _ σ'.2 ?_
      intro s' c d h
      by_cases hs : s' = s
      · subst hs
        rw [hset] at h
        cases h
        rfl
      · rw [hkeep _ _ _ hs] at h
        exact hi s' c d h

end Push

/-! ## The primed table -/

omit [Fintype K] in
theorem take_gamesZC_length (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts))
    (caps : Fin ts.length → Cap D) (j : Fin ts.length) : ((gamesZC A Hs ts caps).take j).length = j := by
  simp [gamesZC, j.isLt.le]

/-! ## The Lock -/

omit [Fintype K] in
/-- **The compiled table and the primed one have the same expectations** inside any session, against the primed
prover. -/
theorem expect_sessAtZ_push [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (f : List (OutZC F D Sl) → ℝ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    expect f (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ =
      expect f (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) :=
  expect_push f (pushSess A Hs T cap₀ pre post) σ fun s c d h => by
    simp [Lock.getPair, getP] at h

end FlockSoundness.Discharge.ZkLink
