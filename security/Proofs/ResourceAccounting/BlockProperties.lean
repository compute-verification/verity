module

public import Definitions.ResourceAccounting.BlockProperties.Policy
import Mathlib.Tactic

/-!
# The conversion and the chain

`freeBlockAtMost` proves `Period.FreeBlockAtMost`: outside the use's failure event, capacity minus use bounds each
free block. `chain` proves `Chain`: outside the union of the accounting statements' failure events every claim holds,
so the run's actual circuit is in the feasible set, and the policy's theorem makes it policy-compliant.
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.BlockProperties

open Verity Finset ResourceAccounting
open scoped ENNReal

theorem freeBlockAtMost {L : Type*} {St : Statement} (P : Period L St) : P.FreeBlockAtMost := by
  intro K F encl free use enclosing cap b ε hF huse hcap hex
  refine le_trans (P.mono _ _ ?_) huse
  rintro r ⟨ha, hn⟩
  refine ⟨ha, fun hb => hn fun j hj => ?_⟩
  have h1 := hb (encl j) (hF j hj)
  have h2 := hcap r (encl j) (hF j hj)
  have h3 := hex r ha j hj
  omega

theorem chain {L : Type*} {St : Statement} (P : Period L St) (pol : Policy L) {κ : Type} (s : Finset κ)
    (K : κ → Finset St.ι) (p : κ → St.ι → Property) (ε : κ → ℝ≥0∞) : Chain P pol s K p ε := by
  intro hcheck hthm hcover hest
  have hsub : ∀ r, P.accepts r ∧ ¬ pol.compliant (P.actual r) →
      ∃ x ∈ s, P.accepts r ∧ ¬ ∀ i ∈ K x, P.Holds r i (p x i) := by
    rintro r ⟨ha, hn⟩
    by_contra hall
    push_neg at hall
    refine hn (hthm St (P.actual r) (P.π r) (P.θ r) hcheck ⟨P.blocks r, fun i q hq => ?_⟩)
    obtain ⟨x, hx, hi, rfl⟩ := hcover i q hq
    exact hall x hx ha i hi
  calc P.pr (fun r => P.accepts r ∧ ¬ pol.compliant (P.actual r))
      ≤ P.pr (fun r => ∃ x ∈ s, P.accepts r ∧ ¬ ∀ i ∈ K x, P.Holds r i (p x i)) := P.mono _ _ hsub
    _ ≤ ∑ x ∈ s, P.pr (fun r => P.accepts r ∧ ¬ ∀ i ∈ K x, P.Holds r i (p x i)) := P.union _ _
    _ ≤ ∑ x ∈ s, ε x := sum_le_sum fun x hx => hest x hx

end ResourceAccounting.SecurityProofs.BlockProperties
