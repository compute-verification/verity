module

public import Proofs.Flock.Claims.Sampling.Select.SecondUnits
import Spot.Draw
import Proofs.Core.QSort
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Soundness.ExecDraw

/-!
# A uniform subset sample is a replay sample the second stage's first check passes

`Select.secondProblem` reads the replay sample only in its first test: that it is not empty and strictly ascending. Its
other tests read the cells alone. A first stage whose sampling design is a uniform subset hands the second stage the
units `Flock.Draw.subset` returns, sorted and each once, so a first stage that samples at least one unit hands it a
replay sample that test passes.

* `subset_ascending`, `drawUnits_subset_ascending`: a `subset` sample that succeeds, and a subset sampling design's
  sample that ends, are strictly ascending.
* `secondProblem_of_ascending`: cells `secondProblem` passes on one replay sample pass on every nonempty, strictly
  ascending one.
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- A `subset` sample that succeeds is strictly ascending. -/
theorem subset_ascending {k n : ℕ} {src : ByteArray} {D : Array ℕ} (h : Flock.Draw.subset k n src = some D) :
    D.toList.Pairwise (· < ·) := by
  have hnd : D.toList.Pairwise (fun x y => x ≠ y) := (subset_nodup h).1
  rw [DrawExec.subset_eq] at h
  obtain ⟨sw, -, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ((Verity.QSort.ByKey.qsort_sorted (fun x : ℕ => x) _).and hnd).imp fun hab => lt_of_le_of_ne hab.1 hab.2

/-- A subset sampling design's sample that ends is strictly ascending. -/
theorem drawUnits_subset_ascending {k n read : ℕ} {src : ByteArray} {units : Array ℕ}
    (h : Draw.drawUnits (.subset k) n src = .ok (units, read)) : units.toList.Pairwise (· < ·) := by
  unfold Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Draw.take src need) (.subset k) n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok x =>
    obtain ⟨r, read'⟩ := x
    rw [hw, bind_ok] at h
    obtain ⟨d, rfl⟩ : ∃ d, r = some d := by
      cases r with
      | none => exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      | some d => exact ⟨d, rfl⟩
    dsimp only at h
    obtain ⟨hu, hr⟩ := Prod.mk.inj (Except.ok.inj h)
    subst hu
    subst hr
    unfold Flock.Draw.drawWith at hw
    dsimp only at hw
    rw [StateT.run_bind] at hw
    cases ho : (Flock.Draw.onWith (fun need => Draw.take src need) (2 * k * Flock.Draw.width n + 64)
        (Flock.Draw.subset k n)).run 0 with
    | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨v, p⟩ := y
      rw [ho, bind_ok, StateT.run_pure] at hw
      obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
      obtain ⟨j, D, rfl, -, hst⟩ := onWith_take src _ (Nat.lt_of_lt_of_le (by decide : 0 < 64) (Nat.le_add_left _ _))
        _ 0 _ _ ho
      obtain rfl := (Option.some.inj h1).symm
      exact subset_ascending hst.2.2.1

/-- A strictly ascending list passes `secondProblem`'s test of the replay sample's order. -/
theorem zip_tail_all_lt : ∀ {l : List ℕ}, l.Pairwise (· < ·) →
    (l.zip l.tail).all (fun (a, b) => decide (a < b)) = true
  | [], _ => rfl
  | [_], _ => rfl
  | a :: b :: l, h => by
    have hab : a < b := List.rel_of_pairwise_cons h List.mem_cons_self
    have ih := zip_tail_all_lt (List.Pairwise.of_cons h)
    simp only [List.tail_cons, List.zip_cons_cons, List.all_cons, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨hab, by simpa using ih⟩

/-- **`secondProblem` reads the replay sample only in its first test**: cells it passes on one replay sample pass on
every nonempty, strictly ascending one. -/
theorem secondProblem_of_ascending {s s' ns ks : Array ℕ} (hne : s.size ≠ 0) (hasc : s.toList.Pairwise (· < ·))
    (h : Select.secondProblem s' ns ks = none) : Select.secondProblem s ns ks = none := by
  have h1 : (s.isEmpty || !(s.toList.zip s.toList.tail).all (fun (a, b) => decide (a < b))) = false := by
    have he : s.isEmpty = false := by
      rw [Array.isEmpty_eq_false_iff]
      exact fun h0 => hne (by rw [h0]; rfl)
    rw [he, zip_tail_all_lt hasc]
    rfl
  unfold Select.secondProblem at h ⊢
  rw [h1]
  split at h
  · cases h
  · exact h

end Properties.Spot.SecurityProofs
