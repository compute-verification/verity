module

public import Proofs.Flock.ZkAny.Table
public import Proofs.Flock.Soundness.ZK.Session

/-!
# A session's release behind the ZK server's checks, at any values

A session of `J` tables stops at some outcome `s` (where it stopped), and the tables whose files left by then
(`full s j`) show their views, the others their pre-final views (`Session.releaseAt`). A file leaves only behind its
table's check (`Session.Gated`), so where it left the constraint holds at that table's drawn pads, and the view is the
derived view there (`psi_realCore_at`). At each outcome the release is then a tuple of independent per-table branches
(`branch`): the derived view, distributed as `S_shvzk(x, e)` at any witness (`derived_sim`), or the pre-final view,
distributed as the dummy's (Lemma A). `prCoin_pi_close` bounds the tuple by the simulator's (`Session.simAt`), and
`ZkAnyOutcome` sums over outcomes.

* `prCoin_snd`: an event of the second factor of a product space has its probability on that factor (`prCoin_fst`'s
  twin).
* `Session.branch`, `Session.branch_le`: one table's branch at an outcome, bounded by its simulator.
* `Session.release_any`, `Flock.SecurityProofs.SessionReleaseAny`: the statement `Flock.Guarantees.SessionReleaseAny`.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- An event of the second factor of a product randomness space has its probability on that factor. -/
theorem prCoin_snd {A B : Type} [Fintype A] [Fintype B] [Nonempty A] (P : B → Prop) :
    prCoin (fun p : A × B => P p.2) = prCoin P := by
  rw [prCoin_prod_split (fun (_ : A) b => P b), avg_const]

namespace Session

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

/-- **A session's release behind the ZK server's checks, at any values**: at each outcome, jointly with it, bounded by
the simulator at that outcome; overall by the sum over outcomes, and by `|St|` times the uniform-outcome simulator. -/
theorem release_any (T : Fin J → Table F K W Salt Seed D Hi VL) (hrank : ∀ j, (T j).Rank)
    (hpad : ∀ j, (T j).PadOnto) (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2)
    (hpads : ∀ j, (T j).PadsOnto) (hβ : ∀ j r, (T j).β r ≠ 0) {St : Type} [Fintype St]
    (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool) (w w₀ : Fin J → W)
    (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg) (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg)
    (hgo : ∀ ω j, full (out ω) j = true → (T j).HoldsAt (w j) (ω j))
    (E : (∀ j, (T j).View ⊕ (T j).PreView) → Prop) :
    (∀ s, prCoin (fun ω => out ω = s ∧ E (releaseAt T full w (out ω) ω)) ≤
      prCoin (fun σ => E (simAt T full w₀ s σ))) ∧
    prCoin (fun ω => E (releaseAt T full w (out ω) ω)) ≤ ∑ s, prCoin (fun σ => E (simAt T full w₀ s σ)) ∧
    prCoin (fun ω => E (releaseAt T full w (out ω) ω)) ≤
      (Fintype.card St : ℝ≥0∞) *
        prCoin (fun p : St × (∀ j, (T j).SimRand × (T j).Rand) => E (simAt T full w₀ p.1 p.2)) := by
  have : ∀ j, Nonempty (T j).Rand := fun _ => ⟨0⟩
  have : ∀ j, Nonempty ((T j).SimRand × (T j).Rand) := fun _ => ⟨(0, 0)⟩
  have hrel : ∀ ω, releaseAt T full w (out ω) ω = fun j => branch T full w (out ω) j (ω j) := fun ω =>
    funext fun j => by
      simp only [releaseAt, branch]
      by_cases hf : full (out ω) j = true
      · rw [ite_eq_left hf, ite_eq_left hf,
          (T j).psi_realCore_at (hW j) (hβ j) (w j) (hreg j) (ω j) (hgo ω j hf)]
      · rw [ite_eq_right hf, ite_eq_right hf]
  have hB : ∀ s (E' : (∀ j, (T j).View ⊕ (T j).PreView) → Prop),
      prCoin (fun ω : ∀ j, (T j).Rand => E' fun j => branch T full w s j (ω j)) ≤
        prCoin (fun σ => E' (simAt T full w₀ s σ)) + 0 := by
    intro s E'
    have h := prCoin_pi_close (Ω := fun j => (T j).Rand) (Ω' := fun j => (T j).SimRand × (T j).Rand)
      (V := fun j => (T j).View ⊕ (T j).PreView) (fun j => branch T full w s j)
      (fun j p => if full s j then .inl ((T j).sim (w₀ j) p.1) else .inr ((T j).preView (w₀ j) p.2))
      (fun _ => 0) (fun j E'' => branch_le T hrank hpad hW hpads hβ full w w₀ hreg hreg₀ s j E'') E'
    simp only [Finset.sum_const_zero] at h
    exact h
  obtain ⟨hper, hsum, hcard⟩ := Flock.SecurityProofs.ZkAnyOutcome (fun ω => releaseAt T full w (out ω) ω) out
    (fun s (ω : ∀ j, (T j).Rand) => fun j => branch T full w s j (ω j)) (fun s σ => simAt T full w₀ s σ) (fun _ => 0)
    hrel hB E
  exact ⟨fun s => by simpa only [add_zero] using hper s, by simpa only [add_zero] using hsum,
    by simpa only [Finset.sum_const_zero, add_zero] using hcard⟩

end Session

end FlockSoundness.ZK

namespace Flock.SecurityProofs

/-- **A session's release behind the ZK server's checks, at any values** (`Flock.Guarantees.SessionReleaseAny`). -/
theorem SessionReleaseAny : Flock.Guarantees.SessionReleaseAny := by
  intro F K W Salt Seed D Hi VL _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ J T St _ out full w w₀ hsim hreg hreg₀ hgo E
  exact FlockSoundness.ZK.Session.release_any T (fun j => (hsim j).1) (fun j => (hsim j).2.1)
    (fun j => (hsim j).2.2.1) (fun j => (hsim j).2.2.2.1) (fun j => (hsim j).2.2.2.2) out full w w₀ hreg hreg₀ hgo E

end Flock.SecurityProofs
