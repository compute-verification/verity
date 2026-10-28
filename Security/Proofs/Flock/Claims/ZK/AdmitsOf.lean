import Proofs.Flock.Claims.ZK.FinishRank

/-!
# Every table admits the coins where no refusal shows

`not_refused_of_admits`'s converse, at the executable's coin plan (`czExe`) of any prover: where every refusal's round is
below `n` (`SlotsBelow`) and no refusal shows by round `n` (`Refused`), every table admits the opened coins. With it
`LeanRelease.FrameAdmits` at the session loop comes down to the frames refusing each fault `Refused` names, at its
round.

* `SlotsBelow x n`: every table's reps' outer coins, pair `ρ₀, ρ₁`, `β` and rank check rounds are below `n` (the rank
  check's at most `n`), so `Refused x c n` sees each.
* **`admits_of_not_refused`**: `CoinedSetup.Admits`' four conjuncts from `¬ Refused x c n`. `β ≠ 0` and `ρ₀, ρ₁`
  independent read the coins at their rounds (`coinReads_czExe`); no outer coin `1` reads them at the rep's outer round
  (`pointReads_czExe`); full rank of the table's four claim points is rep `1`'s rank check, after rep `0`'s points
  (`RankBad`), read as the plan's points (`pointsOf_reads`, `maskRankOk_compose`).
* `slotsBelow_refK`: at a `Live` statement every refusal's round is a round of the session, each within its rep's
  `repRounds`, so **`admits_of_not_refused_refK`** needs no refusal by `refK x` only.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **Every refusal's round is below `n`** at `x`: each table's reps' outer coins, `ρ₁`, `β` and rank check. -/
def SlotsBelow (x : Msg) (n : ℕ) : Prop :=
  ∀ j < (stsOf x).size, ∀ r < 2,
    outerSlot x j r < n ∧ rhoSlot x j r + 1 < n ∧ betaSlot x j r < n ∧ rankSlot x j r ≤ n

/-- **Where no refusal shows, every table admits the opened coins** (`not_refused_of_admits`' converse): at the
executable's coin plan, `β ≠ 0`, `ρ₀, ρ₁` independent and no outer coin `1` in each rep, and full rank of the table's
claim points, wherever the coins below a round `n` past every refusal's show none. -/
theorem admits_of_not_refused (P : Composed.LigProver) {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin}
    {n : ℕ} (hs : SlotsBelow x n) (hn : ¬ Refused x c n) (j : Fin (jOf x)) :
    (tabC P x j).Admits (czExe P x c j) := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨hm, hk6⟩ := tab_bounds hx hj
  obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  refine ⟨fun r => ?_, ?_, fun r => ?_, fun r y hy => ?_⟩
  · rw [(coinReads_czExe P x c j r).1]
    cases hb : (betaAt c j.val r.val).isZero
    · rfl
    · exact (hn (.inr ⟨j.val, hj, r.val, r.isLt, .inr (.inr ⟨(hs j.val hj r.val r.isLt).2.2.1, hb⟩)⟩)).elim
  · have hb : FlockComplete.MaskRank.maskRankOk (ptsOpen c (stsOf x)[j.val] (2 * j.val) ++
        ptsOpen c (stsOf x)[j.val] (2 * j.val + 1)) (Composed.maskWordsOf (stsOf x)[j.val]) = true := by
      cases hb : FlockComplete.MaskRank.maskRankOk (ptsOpen c (stsOf x)[j.val] (2 * j.val) ++
          ptsOpen c (stsOf x)[j.val] (2 * j.val + 1)) (Composed.maskWordsOf (stsOf x)[j.val])
      · refine (hn (.inr ⟨j.val, hj, 1, by norm_num, .inl ⟨hm, hk6, (hs j.val hj 0 (by norm_num)).2.2.2,
          (hs j.val hj 1 (by norm_num)).2.2.2, ?_⟩⟩)).elim
        unfold RankBad
        rw [getElem!_pos (stsOf x) j.val hj, if_pos rfl]
        exact hb
      · rfl
    refine maskRankOk_compose (k := _root_.Flock.Zk.RANK_POINTS) (by rfl) (fun k => ?_) hb
    refine pointsOf_reads ((czExe P x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j))
      (fun r n e => slotAt c (repBase x (2 * j.val + r) + n) e)
      (fun r => (pointReads_czExe P x c j r).1) (fun r => (pointReads_czExe P x c j r).2.1)
      (fun r => ?_) k
    rw [(pointReads_czExe P x c j r).2.2]
    funext i
    unfold outerAt slotAt
    rw [outerSlot_eq hsx]
  · rw [(coinReads_czExe P x c j r).2]
    cases hb : _root_.Flock.Zk.rhoIndependent (rhoAt c j.val r.val)
    · exact (hn (.inr ⟨j.val, hj, r.val, r.isLt, .inr (.inl ⟨(hs j.val hj r.val r.isLt).2.1, hb⟩)⟩)).elim
    · rfl
  · intro h1
    obtain ⟨⟨e, he⟩, hey⟩ := List.mem_ofFn.1 hy
    have ho := congrFun (pointReads_czExe P x c j r).2.2 ⟨e, he⟩
    refine hn (.inl ⟨j.val, hj, r.val, r.isLt, (hs j.val hj r.val r.isLt).1, e, ?_, ?_⟩)
    · rw [getElem!_pos (stsOf x) j.val hj]
      exact he
    · exact ho.symm.trans (hey.trans h1)

/-- **Every refusal's round is a round of the session**: each rep's outer coins, pair, `β` and rank check sit within
its `repRounds`, and the reps' rounds end by `refK x`. -/
theorem slotsBelow_refK {x : Msg} (hx : Live x) : SlotsBelow x (refK x) := by
  intro j hj r hr
  obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  have hK := refK_eq hsx
  obtain ⟨y⟩ := hx.2 j hj
  have hst : (stsOf x)[(2 * j + r) / 2]! = (stsOf x)[j]! := by
    rw [show (2 * j + r) / 2 = j by omega]
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst, getElem!_pos (stsOf x) j hj]
    exact y.hz
  have hk6 : 6 ≤ (stsOf x)[(2 * j + r) / 2]!.c.kLog := by
    rw [hst, getElem!_pos (stsOf x) j hj]
    exact y.wf.1.hk6
  have hend : FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
      FlockDeploy.repRounds true spec (stsOf x)[(2 * j + r) / 2]! ≤ refK x := by
    rw [← FlockComplete.ZkTables.repStart_succ, hK, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono spec (stsOf x) (by omega)
  have hrr : FlockDeploy.repRounds true spec (stsOf x)[(2 * j + r) / 2]! =
      FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! +
        (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 4 := by
    unfold FlockDeploy.repRounds FlockComplete.piopRoundsZk
    simp only [↓reduceIte, hz]
    omega
  have hp : FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! =
      ((stsOf x)[j]!.m + (stsOf x)[j]!.c.kLog - 6) + (2 + spec.points * (stsOf x)[j]!.regions.size + 1) := by
    rw [hst]
    rfl
  rw [hrr, hp] at hend
  have hb0 : repBase x (2 * j + r) = FlockComplete.repStartZk spec (stsOf x) (2 * j + r) := repBase_eq hsx _
  have hρ : rhoSlot x j r = FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
      FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! + (y.sch.k0 + 2) := by
    unfold rhoSlot
    rw [repSlot_eq hsx hz]
  have hβ : betaSlot x j r = FlockComplete.repStartZk spec (stsOf x) (2 * j + r) +
      FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j + r) / 2]! +
        (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1) + 2) := by
    unfold betaSlot
    rw [repSlot_eq hsx hz]
  rw [hp] at hρ hβ
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [outerSlot_eq hsx, hb0]
    omega
  · rw [hρ]
    omega
  · rw [hβ]
    omega
  · unfold rankSlot
    rw [hb0]
    omega

/-- **Where no refusal shows by the session's last coin, every table admits the opened coins**
(`admits_of_not_refused` at `refK x`, `slotsBelow_refK`). -/
theorem admits_of_not_refused_refK (P : Composed.LigProver) {x : Msg} (hx : Live x)
    {c : Fin (refK x) → FlockDeploy.Coin} (hn : ¬ Refused x c (refK x)) (j : Fin (jOf x)) :
    (tabC P x j).Admits (czExe P x c j) :=
  admits_of_not_refused P hx (slotsBelow_refK hx) hn j

end Masked

end Claims.ZKMal
