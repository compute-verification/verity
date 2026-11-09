module

public import Proofs.Flock.Recursive.StopCountProof

/-!
# A stream's stops, counted together

A stream is recursive sessions `ss i`, `i < m`, session `ss i` with `N i` statements, run through one ZK server. Its
joint stop is each session's stop at its statements (`recStopAt`), and what the stream's released views leak is at most
that tuple, by `prCoin_cut_le` at the tuples it can take. By `RecStopsOfRun` a session that stops has a statement not
released, so where at most `B` of the stream's sessions are left with a statement not released, at most `B` of the
tuple's entries are stops.

* `Pattern α B`, `card_pattern`: the tuples of `Option (α i)` with at most `B` entries `some`, which are
  `Σ_{|S| ≤ B} Π_{i ∈ S} |α i|`.
* `streamStop`, `card_streamStop_le`: the joint stop at a run, and that no more of its entries are stops than the
  stream has sessions not released.
* `streamStopIn`, `card_streamStops`: the joint stop as a `Pattern` at `B` where at most `B` sessions are not released,
  and how many there are, `Σ_{|S| ≤ B} Π_{i ∈ S} Σ_p (M_p + 1)`, each factor session `ss i`'s `RecStops` less its none.

Nothing here bounds `B`; `StopCountBudget` bounds it by the stream's failure budget, where the stream has one.
-/

@[expose] public section

namespace Flock.StopCount

open Finset

/-- **Tuples with at most `B` entries `some`**: the joint stops of `m` sessions of which at most `B` stop. -/
abbrev Pattern {m : ℕ} (α : Fin m → Type) (B : ℕ) : Type :=
  {J : (i : Fin m) → Option (α i) // (univ.filter fun i => (J i).isSome).card ≤ B}

instance instNonemptyPattern {m : ℕ} (α : Fin m → Type) (B : ℕ) : Nonempty (Pattern α B) :=
  ⟨⟨fun _ => none, by simp⟩⟩

/-- The tuples whose entries `some` are exactly at `S`, as their entries there. -/
def supportEquiv {m : ℕ} (α : Fin m → Type) (S : Finset (Fin m)) :
    {J : (i : Fin m) → Option (α i) // univ.filter (fun i => (J i).isSome) = S} ≃ ((i : S) → α i) where
  toFun J i := (J.1 i).get (by
    have h : (i : Fin m) ∈ univ.filter (fun j => (J.1 j).isSome) := by
      rw [J.2]
      exact i.2
    exact (mem_filter.mp h).2)
  invFun f := ⟨fun i => if h : i ∈ S then some (f ⟨i, h⟩) else none, by
    ext i
    by_cases h : i ∈ S <;> simp [h]⟩
  left_inv J := by
    obtain ⟨J, hJ⟩ := J
    apply Subtype.ext
    funext i
    by_cases h : i ∈ S
    · simp [h]
    · have : (J i).isSome = false := by
        rw [← hJ] at h
        simpa using h
      simp only [h, dite_false]
      exact (Option.not_isSome_iff_eq_none.mp (by simp [this])).symm
  right_inv f := by
    funext i
    simp [i.2]

/-- **How many tuples with at most `B` entries `some`**: `Σ_{|S| ≤ B} Π_{i ∈ S} |α i|`. -/
theorem card_pattern {m : ℕ} (α : Fin m → Type) [∀ i, Fintype (α i)] (B : ℕ) :
    Fintype.card (Pattern α B) =
      ∑ S ∈ univ.filter (fun S : Finset (Fin m) => S.card ≤ B), ∏ i ∈ S, Fintype.card (α i) := by
  classical
  let f : Pattern α B → {S : Finset (Fin m) // S.card ≤ B} := fun J => ⟨_, J.2⟩
  rw [Fintype.card_congr (Equiv.sigmaFiberEquiv f).symm, Fintype.card_sigma]
  rw [Finset.sum_subtype (univ.filter fun S : Finset (Fin m) => S.card ≤ B) (p := fun S => S.card ≤ B) (by simp)]
  refine Finset.sum_congr rfl fun y _ => ?_
  have e : {x : Pattern α B // f x = y} ≃
      {J : (i : Fin m) → Option (α i) // univ.filter (fun i => (J i).isSome) = y.1} :=
    { toFun := fun x => ⟨x.1.1, congrArg Subtype.val x.2⟩
      invFun := fun J => ⟨⟨J.1, by rw [J.2]; exact y.2⟩, Subtype.ext J.2⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [Fintype.card_congr (e.trans (supportEquiv α y.1)), Fintype.card_pi,
    Finset.prod_coe_sort y.1 (fun i => Fintype.card (α i))]

/-- Where `s` runs through, so does its stop at every number of statements. -/
theorem recStopAt_none {n : ℕ} {M : String → ℕ → ℕ} {σ : St} {s : String} (h : recStop M σ s = none) :
    recStopAt n M σ s = none := by
  unfold recStopAt
  split
  · rename_i hn
    subst hn
    exact h
  · rfl

/-- **A stream's joint stop**: session `ss i`'s stop at its `N i` statements, for each `i`. -/
def streamStop {m : ℕ} (N : Fin m → ℕ) (M : String → ℕ → ℕ) (σ : St) (ss : Fin m → String) :
    (i : Fin m) → Flock.SecurityProofs.RecStops (fun p : Fin (N i) => M (ss i) p) :=
  fun i => recStopAt (N i) M σ (ss i)

/-- **No more of a stream's sessions stop than are not released**, at every run. -/
theorem card_streamStop_le {m : ℕ} (N : Fin m → ℕ) (M : String → ℕ → ℕ) (steps : List Step)
    (ss : Fin m → String) :
    (univ.filter fun i => (streamStop N M (run M steps) ss i).isSome).card ≤
      (univ.filter fun i => unreleasedOf (run M steps) (ss i)).card := by
  refine card_le_card fun i hi => ?_
  simp only [mem_filter, mem_univ, true_and] at hi ⊢
  rcases (Flock.SecurityProofs.RecStopsOfRun M steps (ss i)).2.2.1 with h | h
  · simp [streamStop, recStopAt_none h] at hi
  · exact h

/-- **The joint stop where at most `B` of the stream's sessions are not released**, a `Pattern` at `B`. -/
def streamStopIn {m : ℕ} (N : Fin m → ℕ) (M : String → ℕ → ℕ) (steps : List Step) (ss : Fin m → String) (B : ℕ)
    (hB : (univ.filter fun i => unreleasedOf (run M steps) (ss i)).card ≤ B) :
    Pattern (fun i => Σ p : Fin (N i), Fin (M (ss i) p + 1)) B :=
  ⟨streamStop N M (run M steps) ss, (card_streamStop_le N M steps ss).trans hB⟩

/-- **How many joint stops at `B`**: `Σ_{|S| ≤ B} Π_{i ∈ S} Σ_p (M_p + 1)`, each factor session `ss i`'s
`card_recStops` less its none. -/
theorem card_streamStops {m : ℕ} (N : Fin m → ℕ) (M : String → ℕ → ℕ) (ss : Fin m → String) (B : ℕ) :
    Fintype.card (Pattern (fun i => Σ p : Fin (N i), Fin (M (ss i) p + 1)) B) =
      ∑ S ∈ univ.filter (fun S : Finset (Fin m) => S.card ≤ B), ∏ i ∈ S, ∑ p : Fin (N i), (M (ss i) p + 1) := by
  rw [card_pattern]
  simp only [Fintype.card_sigma, Fintype.card_fin]

end Flock.StopCount
