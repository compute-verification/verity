module

public import Proofs.Flock.ZkAny.Table
public import Proofs.Flock.Soundness.ZK.Session

/-!
# A release of `J` tables behind the ZK server's checks, at any values

A release of `J` tables stops at some outcome `s` (where it stopped), and the tables whose files left by then
(`full s j`) show their views, the others their pre-final views (`Release.bytesAt`). A file leaves only behind its
table's check (`Release.Gated`), so where it left the constraint holds at that table's sampled pads, and the view is the
derived view there (`psi_realCore_at`). At each outcome the released bytes are then a tuple of independent per-table
branches (`branch`): the derived view, distributed as `S_shvzk(x, e)` at any witness (`derived_sim`), or the pre-final
view, distributed as the dummy's (Lemma A). `prCoin_pi_close` bounds the tuple by the simulator's (`Release.simAt`), and
`perOutcome_of_split` makes the bytes simulated per outcome.

* `Release.branch`, `Release.branch_le`: one table's branch at an outcome, bounded by its simulator.
* `Release.release_any`: the released bytes simulated per outcome, at any witnesses carrying the public region data.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Release

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
  [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] {J : ℕ}

/-- **Table `j`'s branch at the outcome `s`**: the derived view where its file left, the pre-final view elsewhere. -/
noncomputable def branch (T : Fin J → Table F K W Salt Seed D Hi VL) {St : Type} (full : St → Fin J → Bool)
    (w : Fin J → W) (s : St) (j : Fin J) (x : (T j).Rand) : (T j).View ⊕ (T j).PreView :=
  if full s j then .inl ((T j).psi ((T j).realCore (w j) x)) else .inr ((T j).preView (w j) x)

/-- **Each table's branch is bounded by its simulator**: `derived_sim` where its file left, Lemma A elsewhere. -/
theorem branch_le (T : Fin J → Table F K W Salt Seed D Hi VL) (hrank : ∀ j, (T j).Rank)
    (hpad : ∀ j, (T j).PadOnto) (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2)
    (hpads : ∀ j, (T j).PadsOnto) (hβ : ∀ j r, (T j).β r ≠ 0) {St : Type} (full : St → Fin J → Bool)
    (w w₀ : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg) (s : St) (j : Fin J) (E : (T j).View ⊕ (T j).PreView → Prop) :
    prCoin (fun x => E (branch T full w s j x)) ≤
      prCoin (fun p : (T j).SimRand × (T j).Rand =>
        E (if full s j then .inl ((T j).sim (w₀ j) p.1) else .inr ((T j).preView (w₀ j) p.2))) + 0 := by
  have : Nonempty (T j).Rand := ⟨0⟩
  have : Nonempty (T j).SimRand := ⟨0⟩
  rw [add_zero]
  unfold branch
  cases hf : full s j
  · show prCoin (fun x => E (.inr ((T j).preView (w j) x))) ≤
      prCoin (fun p : (T j).SimRand × (T j).Rand => E (.inr ((T j).preView (w₀ j) p.2)))
    rw [prCoin_snd (A := (T j).SimRand) (fun ω => E (.inr ((T j).preView (w₀ j) ω)))]
    exact ((T j).table_prefinal_indep (hrank j) (hW j) (hβ j) (w j) (w₀ j) (hreg j) (hreg₀ j)).prCoin_eq
      (fun v => E (.inr v)) |>.le
  · show prCoin (fun x => E (.inl ((T j).psi ((T j).realCore (w j) x)))) ≤
      prCoin (fun p : (T j).SimRand × (T j).Rand => E (.inl ((T j).sim (w₀ j) p.1)))
    rw [prCoin_fst (E := (T j).Rand) (fun σ => E (.inl ((T j).sim (w₀ j) σ)))]
    exact ((T j).derived_sim (hrank j) (hpad j) (hW j) (hpads j) (hβ j) (w j) (w₀ j)).prCoin_eq
      (fun v => E (.inl v)) |>.le

/-- **A release of `J` tables behind the ZK server's checks, at any values**: the released bytes are simulated per
outcome, exactly, by the simulator at that outcome. -/
theorem release_any (T : Fin J → Table F K W Salt Seed D Hi VL) (hrank : ∀ j, (T j).Rank)
    (hpad : ∀ j, (T j).PadOnto) (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2)
    (hpads : ∀ j, (T j).PadsOnto) (hβ : ∀ j r, (T j).β r ≠ 0) {St : Type}
    (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool) (w w₀ : Fin J → W)
    (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg) (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg)
    (hgo : Gated T out full w) :
    PerOutcome out (fun ω => bytesAt T full w (out ω) ω) (simAt T full w₀) fun _ => 0 := by
  have : ∀ j, Nonempty (T j).Rand := fun _ => ⟨0⟩
  have : ∀ j, Nonempty ((T j).SimRand × (T j).Rand) := fun _ => ⟨(0, 0)⟩
  have hrel : Splits (fun ω => bytesAt T full w (out ω) ω) out
      (fun s (ω : ∀ j, (T j).Rand) => fun j => branch T full w s j (ω j)) := fun ω =>
    funext fun j => by
      simp only [bytesAt, branch]
      by_cases hf : full (out ω) j = true
      · rw [ite_eq_left hf, ite_eq_left hf,
          (T j).psi_realCore_at (hW j) (hβ j) (w j) (hreg j) (ω j) (hgo ω j hf)]
      · rw [ite_eq_right hf, ite_eq_right hf]
  refine perOutcome_of_split hrel fun s E' => ?_
  have h := prCoin_pi_close (Ω := fun j => (T j).Rand) (Ω' := fun j => (T j).SimRand × (T j).Rand)
    (V := fun j => (T j).View ⊕ (T j).PreView) (fun j => branch T full w s j)
    (fun j p => if full s j then .inl ((T j).sim (w₀ j) p.1) else .inr ((T j).preView (w₀ j) p.2))
    (fun _ => 0) (fun j E'' => branch_le T hrank hpad hW hpads hβ full w w₀ hreg hreg₀ s j E'') E'
  simp only [Finset.sum_const_zero] at h
  exact h

end Release

end FlockSoundness.ZK
