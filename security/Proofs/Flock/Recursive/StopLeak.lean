module

public import Definitions.Core.Game
public import Mathlib.Data.Fintype.Option
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Data.Fintype.Sigma
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# The recursive session's stop leak: the counting lemma and the stops

A recursive session releases the auditor's whole view cut at a stop: where its first statement to stop stopped, or
nowhere. Which stop comes may depend on anything, the witness included, so the stop is a leak. These two pieces bound it
(`verity/ml/flock/live/PROTOCOL.md` §10.5, its first row).

* `prCoin_cut_le`: where the whole view is within `ε` of an ideal one at every event, the whole view cut at a stop of
  `St`, which may depend on everything, is within `|St|` times the ideal view cut at a uniform stop, plus `ε`. The
  zero-knowledge bound is applied once, at the event "some stop's cut of the view is in `E`", so `ε` is not multiplied;
  the union bound over the stops gives the factor. The stop leaks at most `log₂ |St|` bits.
* `RecStops M`, `card_recStops`: a recursive session's stops, none or the first statement `p` to stop after its `k`-th
  message, `k ≤ M p`, which are `1 + Σ_p (M p + 1)`.
-/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.ZK
open Game
open scoped ENNReal

/-- **A view released at a stop leaks at most the stop.** The whole view is `τ ω` on the coins `ω`, within `ε` of the
ideal view `ideal c` at every event (`hzk`). The released view is the whole view cut (`cut s`) at the stop
`s = stop (ω, κ)`, which may depend on `ω` and on further coins `κ` in any way. Then every event of the released view is
at most `|St|` times as likely as at the ideal view cut at a uniform stop, plus `ε`, added once. -/
theorem prCoin_cut_le {Ω Ωk C St T W : Type} [Fintype Ω] [Fintype Ωk] [Nonempty Ωk] [Fintype C] [Fintype St]
    [Nonempty St] (τ : Ω → T) (ideal : C → T) (cut : St → T → W) (stop : Ω × Ωk → St) {ε : ℝ≥0∞}
    (hzk : ∀ E : T → Prop, prCoin (fun ω => E (τ ω)) ≤ prCoin (fun c => E (ideal c)) + ε) (E : W → Prop) :
    prCoin (fun u => E (cut (stop u) (τ u.1))) ≤
      (Fintype.card St : ℝ≥0∞) * prCoin (fun p : St × C => E (cut p.1 (ideal p.2))) + ε := by
  have hsum : (∑ s, prCoin fun c => E (cut s (ideal c))) =
      (Fintype.card St : ℝ≥0∞) * prCoin (fun p : St × C => E (cut p.1 (ideal p.2))) := by
    rw [prCoin_prod_split fun s c => E (cut s (ideal c))]
    unfold avg
    rw [← Finset.mul_sum, ← mul_assoc,
      ENNReal.mul_inv_cancel (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _), one_mul]
  calc prCoin (fun u => E (cut (stop u) (τ u.1)))
      ≤ prCoin (fun u : Ω × Ωk => ∃ s, E (cut s (τ u.1))) := prCoin_mono fun u hu => ⟨stop u, hu⟩
    _ = prCoin (fun ω => ∃ s, E (cut s (τ ω))) := prCoin_fst fun ω => ∃ s, E (cut s (τ ω))
    _ ≤ prCoin (fun c => ∃ s, E (cut s (ideal c))) + ε := hzk fun t => ∃ s, E (cut s t)
    _ ≤ (∑ s, prCoin fun c => E (cut s (ideal c))) + ε :=
        add_le_add ((prCoin_mono fun c (h : ∃ s, E (cut s (ideal c))) =>
            h.elim fun s hs => ⟨s, Finset.mem_univ s, hs⟩).trans
          (prCoin_exists_le Finset.univ fun s c => E (cut s (ideal c)))) le_rfl
    _ = _ := by rw [hsum]

/-- **A recursive session's stops**: none, every statement released; or the first statement to stop, `p`, after its
`k`-th message, `k ≤ M p` (`0`: never opened; `M p`: every message sent, then refused at its release or rejected).
Statement `0` is the inner session and the rest are V*'s, in the order the record's `open` fixed; `M p` is statement
`p`'s message count. -/
abbrev RecStops {n : ℕ} (M : Fin n → ℕ) : Type := Option (Σ p : Fin n, Fin (M p + 1))

/-- **How many stops**: `1 + Σ_p (M p + 1)`. Its `log₂` is what a recursive session's stop leaks, in bits. -/
theorem card_recStops {n : ℕ} (M : Fin n → ℕ) : Fintype.card (RecStops M) = 1 + ∑ p, (M p + 1) := by
  rw [Fintype.card_option, Fintype.card_sigma, Nat.add_comm]
  simp only [Fintype.card_fin]

end Flock.SecurityProofs
