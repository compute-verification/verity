import Proofs.Flock.Soundness.Discharge.PlaysCoin.Statements

/-!
# A run's coin steps

`toT_exact` (`ToTExact`): with `Plays.coinVec` at `n ≠ 1`, a run on two lists is a run on one tape whose coin steps are
exactly its coin arrays as `coinArr` reads them, by the induction of `Plays.toT` with no inversion. `coinMsg_true`: on a
round whose coin count is its array's size, `coinMsg true` is `coinArr`. So `coinsOfRounds` (`CoinsOfRounds`): a run on
well-formed rounds' coins has the coin steps `coinMsg true` of the rounds.
-/

namespace FlockSoundness.Discharge.PlaysCoin

open Flock FlockSoundness.Game FlockSoundness.Refine

theorem toT_exact : ToTExact := by
  intro W F _ ι α g ms cs a h
  induction h with
  | ret a => exact ⟨[], .ret a, rfl, rfl⟩
  | send m _ ih =>
      obtain ⟨ts, hp, h1, h2⟩ := ih
      exact ⟨.inl ⟨_, m⟩ :: ts, .send m hp, by rw [List.filterMap_cons, Sum.getLeft?_inl, h1],
        by rw [List.filterMap_cons, Sum.getRight?_inl]; exact h2⟩
  | coinVec c hc hn _ ih =>
      obtain ⟨ts, hp, h1, h2⟩ := ih
      subst hc
      refine ⟨.inr ⟨_, fun i => ι c[i.val]!⟩ :: ts, .coin _ hp, by rw [List.filterMap_cons, Sum.getLeft?_inr, h1], ?_⟩
      simp only [List.filterMap_cons, Sum.getRight?_inr, List.map_cons, h2, coinArr, hn, ite_false]
  | coinOne c hc _ ih =>
      obtain ⟨ts, hp, h1, h2⟩ := ih
      refine ⟨.inr ⟨F, ι c[0]!⟩ :: ts, .coin _ hp, by rw [List.filterMap_cons, Sum.getLeft?_inr, h1], ?_⟩
      simp only [List.filterMap_cons, Sum.getRight?_inr, List.map_cons, h2, coinArr, hc, ite_true]

/-- On a round whose coin count is its coin array's size, `coinMsg true` is the array's coin step. -/
theorem coinMsg_true (rd : Round) (h : rd.coins.size = rd.n) : coinMsg true rd = coinArr fOf rd.coins := by
  unfold coinMsg coinArr
  rcases rd with ⟨n, sha, coins, bytes⟩
  dsimp only at h ⊢
  subst h
  simp

theorem coinsOfRounds : CoinsOfRounds := by
  intro α g ms rs a hwf h
  obtain ⟨ts, hp, h1, h2⟩ := toT_exact fOf h
  refine ⟨ts, hp, h1, ?_⟩
  rw [h2, List.map_map]
  exact List.map_congr_left fun rd hrd => (coinMsg_true rd (hwf rd hrd)).symm

/-- A record's well-formed rounds (`RoundsWF`, S12), as a list. -/
theorem roundsWF_mem {rs : Array Round} (h : RoundsWF rs) : ∀ rd ∈ rs.toList, rd.coins.size = rd.n := by
  intro rd hrd
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hrd
  rw [Array.length_toList] at hi
  have := h i hi
  rw [getElem!_pos rs i hi] at this
  simpa using this

end FlockSoundness.Discharge.PlaysCoin
