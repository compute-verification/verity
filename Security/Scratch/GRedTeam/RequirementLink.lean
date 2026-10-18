import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Definitions.Accounting.Requirement

/-!
# Red team of G: what the requirement side of G does and doesn't say

Scratch, not built by any `lean_lib`, over A1's definitions as they are on `cursor/stacking-composed-6cd3`.

* `creditedReads_subset_credited`: when each gate uses only its operands (`Execution.Follows`'s second conjunct), the
  values a step reads from complete blocks are among the live values the audit credits. So G's "moved" disjunct is
  implied by its "held" one: it adds nothing that the store count doesn't already give.
* `filler_breaks_StoreMatches`: an honest store under memory-map/v1, whose KV pool pages are registered as filler,
  has a resident block that holds no live value, so `StoreMatches` (an equality) fails, and G's row-12 hypothesis
  can't hold for that node. `filler_subset`: the `⊆` form, the only direction `RequirementHeld` uses, holds there.
-/

namespace GRedTeam

open Accounting Accounting.Requirement

variable {V : Type} [DecidableEq V]

theorem creditedReads_subset_credited (W : Wiring V) (e : Execution V) (ℓ : Layout V)
    (huses : ∀ g ∈ e.ran, W.uses e.value g ⊆ W.ops g) (s : ℕ) :
    creditedReads W e ℓ s ⊆ credited W e ℓ s := by
  intro v hv
  simp only [creditedReads, credited, reads, live, Finset.mem_filter] at hv ⊢
  obtain ⟨⟨hran, hreg, g, hg, hstep, hu⟩, hinc⟩ := hv
  exact ⟨⟨hran, hreg, g, hg, hstep.ge, huses g hg hu⟩, hinc⟩

theorem creditedReadBits_le (W : Wiring V) (e : Execution V) (ℓ : Layout V)
    (huses : ∀ g ∈ e.ran, W.uses e.value g ⊆ W.ops g) (s : ℕ) :
    creditedReadBits W e ℓ s ≤ creditedBits W e ℓ s :=
  Finset.sum_le_sum_of_subset (creditedReads_subset_credited W e ℓ huses s)

/-- Two values: `0`, a registered weight in block 0, and `1`, a KV value produced at step 0 from it, in block 5. -/
def W₀ : Wiring (Fin 2) where
  width := fun _ => 8
  registered := {0}
  ops := fun g => if g = 1 then {0} else ∅
  uses := fun _ g => if g = 1 then {0} else ∅

def e₀ : Execution (Fin 2) where
  ran := Finset.univ
  step := fun _ => 0
  value := fun _ => 0

def ℓ₀ : Layout (Fin 2) where
  block := fun v => if v = 0 then 0 else 5
  offset := fun _ => 0

/-- The registered store at registration: block 0 holds the weight, block 1 is a KV pool page registered as filler. -/
def σ₀ : Store where
  registered := 2
  written := fun _ => ∅
  freed := fun _ => ∅
  payload := fun _ => 0

theorem filler_breaks_StoreMatches : ¬ StoreMatches W₀ e₀ ℓ₀ σ₀ 32768 0 := by
  intro h
  have h1 := h.1
  have hres : σ₀.resident 0 = {0, 1} := by decide
  have hcred : blocks ℓ₀ (credited W₀ e₀ ℓ₀ 0) = {0} := by decide
  rw [hres, hcred] at h1
  exact absurd (congrArg (1 ∈ ·) h1) (by decide)

theorem filler_subset : blocks ℓ₀ (credited W₀ e₀ ℓ₀ 0) ⊆ σ₀.resident 0 := by decide

end GRedTeam

#print axioms GRedTeam.creditedReads_subset_credited
#print axioms GRedTeam.creditedReadBits_le
#print axioms GRedTeam.filler_breaks_StoreMatches
#print axioms GRedTeam.filler_subset
