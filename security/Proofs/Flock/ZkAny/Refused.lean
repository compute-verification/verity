module

public import Proofs.Flock.ZkAny.Tables
public import Proofs.Flock.Soundness.ZK.Refusal

/-!
# A release at coins it refuses, at any values

The ZK server refuses three kinds of coins, each checked just before the messages it guards and each stopping the whole
release before any proof file leaves (`Definitions/Flock/ZK/Cut.lean`): the opening's rank check, the restart's `ρ`
check and the inner proof's `β` check. Each reads the coins alone, so where the release stops is public, and what it
released is its cut (`Table.cut` at `Session.sent`). Lemma A at every coin vector (`session_prefinal_indep_cut`) reads
no constraint, so the cut is the dummy's at any witnesses carrying the public region data, whatever their values.

* `Table.rank_of_rankOn`: the rank check over every claim is `Rank`.
* `Release.simulable_of_passes`: where every check of table `j` passes, its coins are non-degenerate for it
  (`Table.Simulable`, with the table's own `PadOnto` and `PadsOnto`), which `Release.release_any` takes.
* **`Release.cut_within`**: at every coin vector, the release's cut is simulated exactly by the dummy's.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] (T : Table F K W Salt Seed D Hi VL)

/-- **The rank check over every claim is `Rank`.** -/
theorem rank_of_rankOn {P : Fin T.nC → Prop} (hP : ∀ k, P k) (h : T.RankOn P) : T.Rank := by
  intro t
  obtain ⟨c, hc⟩ := h fun k => t k.1
  exact ⟨c, funext fun k => congrFun hc ⟨k, hP k⟩⟩

end Table

namespace Release

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] {J : ℕ}
  (T : Fin J → Table F K W Salt Seed D Hi VL)

/-- **Where every check of table `j` passes, its coins are non-degenerate for it**: rep 1's opening check covers every
claim (`Rank`), each rep's `W^r` is a bijection and each `β^r` is nonzero; `PadOnto` and `PadsOnto` are the table's. -/
theorem simulable_of_passes (rep : ∀ j, Fin (T j).nC → Fin 2) (j : Fin J) (hp : ∀ r p, Session.passes T rep j r p)
    (hpad : (T j).PadOnto) (hpads : (T j).PadsOnto) : (T j).Simulable := by
  refine ⟨(T j).rank_of_rankOn (P := fun k => rep j k ≤ 1) (fun k => ?_) (Session.passes_guard T (hp 1 .opening)),
    hpad, fun r => Session.passes_guard T (hp r .restart), hpads, fun r => Session.passes_guard T (hp r .inner)⟩
  have := (rep j k).isLt
  rw [Fin.le_iff_val_le_val]
  simp only [Fin.isValue, Fin.val_one]
  omega

variable [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
  [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi] [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

/-- **At every coin vector, the release's cut is the dummy's**: what each table sent before the release's first failing
check, on witnesses carrying the public region data, is distributed as on the dummy, whatever the witnesses' values
(`session_prefinal_indep_cut`, Lemma A at every coin vector). -/
theorem cut_within (rep : ∀ j, Fin (T j).nC → Fin 2) (w w₀ : Fin J → W) (hreg : ∀ j, (T j).RegPublic (w j))
    (hreg₀ : ∀ j, (T j).RegPublic (w₀ j)) :
    Within (fun ω : ∀ j, (T j).Rand => fun j => (T j).cut (Session.sent T rep j) ((T j).preView (w j) (ω j)))
      (fun ω : ∀ j, (T j).Rand => fun j => (T j).cut (Session.sent T rep j) ((T j).preView (w₀ j) (ω j))) 0 := by
  have : ∀ j, Nonempty (T j).Rand := fun _ => ⟨0⟩
  intro E
  rw [add_zero]
  exact ((Session.session_prefinal_indep_cut T rep w w₀ hreg hreg₀).prCoin_eq E).le

end Release

end FlockSoundness.ZK
