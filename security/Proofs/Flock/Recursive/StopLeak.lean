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

/-! The recursive session's stop leak: a view released at a stop of `St` leaks at most `log₂ |St|` bits (`prCoin_cut_le`), and a session has `1 + Σ_p (M p + 1)` stops (`card_recStops`). -/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.ZK
open Game
open scoped ENNReal

/-- The stop may depend on the coins in any way; `ε` is added once, not once per stop. -/
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

/-- Its `log₂` is what a recursive session's stop leaks, in bits. -/
theorem card_recStops {n : ℕ} (M : Fin n → ℕ) : Fintype.card (RecStops M) = 1 + ∑ p, (M p + 1) := by
  rw [Fintype.card_option, Fintype.card_sigma, Nat.add_comm]
  simp only [Fintype.card_fin]

end Flock.SecurityProofs
