module

public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Bridge.SampledDesign
public import Spot.PopOrder

/-!
# The registration's strata cover the population's units (`Bridge.PopCovers`)

The uniform-subset form of the population: the population is the units `0, …, n − 1` in unit order, and the strata's
ranges need only be a disjoint cover of them. `Bridge.PopCovers desc q n`, a decidable check on the files, as
`Bridge.PopMatches` is: `desc` parses, its layout under the query `q` (`Spot.Slots.layout`) is defined, and its
population's units are `0, …, n − 1`, each once, in any order.

* `popCovers_iff`: `PopCovers` is that the parse and the layout succeed and the population's units are a permutation
  of `0, …, n − 1`.
* `popCovers_iff_ranges`: `PopCovers` is the check the verifier runs, the ranges form (`Spot.Slots.popCoverOk`,
  through `Spot.Slots.popCovers_eq_popCoverOk`).
* `popCovers_of_popMatches`: the population in order (`PopMatches`) is a cover.
* `nsOf_sum_of_popCovers`: at `PopCovers`, the strata sum to `n`.
* `IsSubsetDesign`: the sampling design is `subset k`, the uniform subset (`All` is `subset n`).
* `PopHeld`: the population in order, or covered under a uniform subset: the one population hypothesis.
* `samplingDesignOn_samplesUnder_cover`: at `PopCovers`, a subset sampling design the verifier samples under is on the
  registration's strata (`SamplingDesignOn`), which reads only their sum.
-/

@[expose] public section

namespace Flock.Bridge

/-- `PopCovers`'s check: the verifier's (`Spot.Slots.popCovers`) at the program `desc` parses to. -/
def popCoverCheck (desc : Hash.Msg) (q : Lean.Json) (n : ℕ) : Bool :=
  match Spot.Program.parse ⟨desc.toArray⟩ with
  | .error _ => false
  | .ok p => Spot.Slots.popCovers p q n

/-- **The registration's strata cover the population's `n` units**: `desc` parses, its layout under the query `q` is
defined, and the layout's population's units are `0, …, n − 1`, each once, in any order. -/
@[reducible] def PopCovers (desc : Hash.Msg) (q : Lean.Json) (n : ℕ) : Prop := popCoverCheck desc q n = true

instance instDecidablePopCovers (desc : Hash.Msg) (q : Lean.Json) (n : ℕ) : Decidable (PopCovers desc q n) :=
  inferInstanceAs (Decidable (_ = true))

/-- **The uniform-subset sampling design**: `subset k`; `All` is `subset n`. -/
def IsSubsetDesign (samplingDesign : Flock.Draw.SamplingDesign) : Prop := ∃ k, samplingDesign = .subset k

/-- **The registration's population is held**: its units are the `n` proof blocks in order (`PopMatches`), or the strata
cover them (`PopCovers`) and the sampling design the verifier samples under (`designAt`) is a uniform subset. A check
the verifier runs at registration (`Spot.Slots.popOk`, or `Spot.Slots.popCoverOk` under a subset design), which the
bridge concludes from an accept; not an assumption. -/
@[reducible] def PopHeld (desc : Hash.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Prop :=
  PopMatches desc q n ∨ (PopCovers desc q n ∧ IsSubsetDesign (designAt desc q lj tb n))

theorem popCovers_iff {desc : Hash.Msg} {q : Lean.Json} {n : ℕ} :
    PopCovers desc q n ↔ ∃ p l, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧
      Spot.Slots.layout p q = .ok l ∧ ((popOf l).map (·.unit)).Perm ((List.range n).map some) := by
  unfold PopCovers popCoverCheck Spot.Slots.popCovers
  constructor
  · intro h
    split at h
    · exact absurd h Bool.false_ne_true
    · rename_i p hp
      split at h
      · exact absurd h Bool.false_ne_true
      · rename_i l hl
        exact ⟨p, l, hp, hl, of_decide_eq_true h⟩
  · rintro ⟨p, l, hp, hl, h⟩
    rw [hp]
    show (match Spot.Slots.layout p q with
      | .error _ => false
      | .ok l => decide (((popOf l).map (·.unit)).Perm ((List.range n).map some))) = true
    rw [hl]
    exact decide_eq_true h

/-- **`PopCovers` is the ranges form the verifier runs**: `desc` parses to a program whose query's ranges are a
disjoint cover of `0, …, n − 1` with every unit's gates (`Spot.Slots.popCoverOk`), which `Spot.Slots.popCovers` runs
as (`Spot.Slots.popCovers_eq_popCoverOk`). -/
theorem popCovers_iff_ranges {desc : Hash.Msg} {q : Lean.Json} {n : ℕ} :
    PopCovers desc q n ↔ ∃ p, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧ Spot.Slots.popCoverOk p q n = true := by
  unfold PopCovers popCoverCheck
  rw [Spot.Slots.popCovers_eq_popCoverOk]
  split
  · simp_all
  · rename_i p hp
    exact ⟨fun h => ⟨p, hp, h⟩, fun ⟨_, hp', h⟩ => by rw [hp] at hp'; cases hp'; exact h⟩

/-- **The population in order is a cover**: `PopMatches` gives `PopCovers`. -/
theorem popCovers_of_popMatches {desc : Hash.Msg} {q : Lean.Json} {n : ℕ} (h : PopMatches desc q n) :
    PopCovers desc q n := by
  obtain ⟨p, l, hp, hl, hu⟩ := popMatches_iff.1 h
  exact popCovers_iff.2 ⟨p, l, hp, hl, List.Perm.of_eq hu⟩

/-- **The verifier's population in order is its cover**: `Slots.popOk` gives `Slots.popCovers`. -/
theorem slots_popCovers_of_popOk {p : Spot.Program} {q : Lean.Json} {n : ℕ} (h : Spot.Slots.popOk p q n = true) :
    Spot.Slots.popCovers p q n = true := by
  unfold Spot.Slots.popOk at h
  unfold Spot.Slots.popCovers
  split at h
  · cases h
  · exact decide_eq_true (List.Perm.of_eq (of_decide_eq_true h))

/-- **The verifier's cover check is `PopCovers`.** -/
theorem popCovers_of_slots {desc : Hash.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (h : Spot.Slots.popCovers p q n = true) :
    PopCovers desc q n := by
  show popCoverCheck desc q n = true
  unfold popCoverCheck
  rw [hp]
  exact h

/-- **The strata sum to `n`**: at `PopCovers`, the registration's strata cover the population's `n` slots. -/
theorem nsOf_sum_of_popCovers {desc : Hash.Msg} {q : Lean.Json} {n : ℕ} (h : PopCovers desc q n) :
    (nsOf desc q).sum = n := by
  obtain ⟨p, l, hp, hl, hu⟩ := popCovers_iff.1 h
  rw [nsOf_eq hp hl, sum_strata_popOf]
  simpa only [List.length_map, List.length_range] using hu.length_eq

/-- **A subset sampling design the verifier samples under is on the strata at `PopCovers`**: `SamplingDesignOn` at
`nsOf` and the `kd` units it samples, which for `subset k` reads only that `k ≤` the strata's sum, `n`
(`sampledDesign_cases`: a derived sampling design is stratified or work, never a subset). -/
theorem samplingDesignOn_samplesUnder_cover {desc : Hash.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} {kd : ℕ} (hpop : PopCovers desc q n)
    (h : SamplesUnder desc q lj tb n samplingDesign kd) (hs : IsSubsetDesign samplingDesign) :
    SamplingDesignOn samplingDesign (nsOf desc q) kd := by
  obtain ⟨k, rfl⟩ := hs
  obtain ⟨h, rfl⟩ := h
  have hsum := nsOf_sum_of_popCovers hpop
  rcases sampledDesign_cases h with ⟨k', hk', hk⟩ | ⟨a, b, h'⟩ | ⟨p, S, k', -, -, hder | ⟨tab, hwork⟩⟩
  · cases hk'
    exact ⟨rfl, by rw [hsum]; exact hk⟩
  · cases h'
  · cases Properties.Spot.SecurityProofs.derive_ok k' S _ hder
  · obtain ⟨ws, -, h'⟩ := Properties.Spot.SecurityProofs.deriveWork_ok k' S tab _ hwork
    cases h'

end Flock.Bridge
