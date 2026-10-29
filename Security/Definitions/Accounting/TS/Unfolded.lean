import Definitions.Accounting.TS.Receipt

/-!
# A round's unfolded share and the whole-unit luck (trusted)

HMU's B′ at a whole unit (`p₀ = ½`): a run that skips a unit's fold commits a value for the unit's receipt
that no word of the unit determines, so the round's check passes only if it guessed the unit's committed value.

* `Unfolded`: per round, the share `e` of the run's words whose fold it skips, fixed before the round's coins, then the
  rest of the play, which may depend on the round's coins.
* `Unfolded.share`: the unfolded share `Σ_w e_w` along the realized coins.
* `ReceiptGame.WholeLuck p₀`: every round's unfolded share lies in `[0, 1]`, and at every round, whatever the past, a
  round with a positive unfolded share keeps the honest committed value with probability at most `p₀`. This is HMU's
  B′ at a unit whose fold the run skips.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **The unfolded share** over `n` rounds of coins on `Ω`: per round, the share `e` of the run's words whose fold it
skips, fixed before the round's coins, then the rest of the play, which may depend on the round's coins. -/
inductive Unfolded (Ω : Type) : ℕ → Type
  | done : Unfolded Ω 0
  | round {n : ℕ} (e : ℝ) (next : Ω → Unfolded Ω n) : Unfolded Ω (n + 1)

namespace Unfolded

variable {Ω : Type}

/-- The unfolded share `Σ_w e_w` along the realized coins. -/
def share : {n : ℕ} → Unfolded Ω n → Paths Ω n → ℝ
  | _, .done, _ => 0
  | _, .round e next, x => e + (next x.1).share x.2

end Unfolded

namespace ReceiptGame

variable {Ω : Type}

/-- **The whole-unit luck at `p₀`** (HMU's B′ at a unit whose fold the run skips): every round's unfolded share lies in
`[0, 1]`, and at every round, whatever the past, a round with a positive unfolded share keeps the honest committed
value with probability at most `p₀` over the round's coins. -/
def WholeLuck [Fintype Ω] (p0 : ℝ) : {n : ℕ} → ReceiptGame Ω n → Unfolded Ω n → Prop
  | _, .done, .done => True
  | _, .round _ _ _ same next, .round e vnext =>
      (0 ≤ e ∧ e ≤ 1) ∧ (0 < e → prob same ≤ p0) ∧ ∀ ω, WholeLuck p0 (next ω) (vnext ω)

end ReceiptGame

end Accounting.TS
