module

public import Proofs.Accounting.Bandwidth.Honest
public import Proofs.Accounting.RateAudit.Witnesses

/-!
# PoUB's per-pass lemma: its honest and nontrivial lemmas

For `Remote.AuditRO`, at a store of one block of one bit, one-bit tags, no state, side sources or regeneration, one
sampled block, `k = 1` and `λ = 0`:

* **The honest lemma** (`AuditRO_honest`): the honest device (`Bandwidth.Completeness.honest`) is accepted and reads
  the whole store, with probability 1.
* **The nontrivial lemma** (`AuditRO_nontrivial`): with the check replaced by `True`, a device that reads nothing
  (`Bandwidth.blank`) falls short of the store with probability 1, above the error `2^-1`.
-/

@[expose] public section

namespace Bandwidth

open Pous (Bits)

/-- The device that reads nothing: no state, nothing in device memory or the side sources, and a chip that commits to
all-`false` tags at once. -/
def blank (N b h S c G q : ℕ) : Device N b h S c G q where
  state _ := fun _ => false
  mem _ _ := false
  side _ _ := false
  chip _ := .done fun _ _ => false
  bounded _ _ := ⟨Nat.zero_le _, Nat.zero_le _, Nat.zero_le _⟩

theorem blank_ingress {N b h S c G q : ℕ} (E : Store N b) (O : Oracle N b h) :
    (blank N b h S c G q).ingress E O = 0 := by
  simp [Device.ingress, blank, Chip.ingressBits]

end Bandwidth

namespace Accounting.SecurityProofs.Remote

open Pous (Bits pr)
open scoped ENNReal

/-- **`AuditRO`'s honest lemma**: the honest device, at one block of one bit, is accepted and reads the store, with
probability 1. -/
theorem AuditRO_honest :
    pr (fun x : Bandwidth.Store 1 1 × Bandwidth.Oracle 1 1 1 × (Fin 1 → Fin 1) =>
        Bandwidth.Accepts (Bandwidth.Completeness.honest 1 1 1) x.1 x.2.1 x.2.2 ∧
          ¬ (Bandwidth.Completeness.honest 1 1 1).ingress x.1 x.2.1 + 0 + 0 + 0 * 1 + 0 < 1 * 1) = 1 :=
  Accounting.SecurityProofs.pr_one fun x => by
    obtain ⟨ha, hi⟩ := Bandwidth.Completeness.honest_accepts x.1 x.2.1 x.2.2
    refine ⟨ha, ?_⟩
    rw [hi]
    omega

/-- **`AuditRO`'s nontrivial lemma**: with the check accepting everything, the device that reads nothing falls short of
the store on every coin, above the error `2^-1`. -/
theorem AuditRO_nontrivial :
    ¬ pr (fun x : Bandwidth.Store 1 1 × Bandwidth.Oracle 1 1 1 × (Fin 1 → Fin 1) =>
        True ∧ (Bandwidth.blank 1 1 1 0 0 0 0).ingress x.1 x.2.1 + 0 + 0 + 0 * 1 + 0 < 1 * 1)
      ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
        ((Nat.choose 0 1 : ℕ) : ℝ≥0∞) / 2 ^ 0 := by
  have hev : ∀ x : Bandwidth.Store 1 1 × Bandwidth.Oracle 1 1 1 × (Fin 1 → Fin 1),
      True ∧ (Bandwidth.blank 1 1 1 0 0 0 0).ingress x.1 x.2.1 + 0 + 0 + 0 * 1 + 0 < 1 * 1 := fun x =>
    ⟨trivial, by rw [Bandwidth.blank_ingress]; omega⟩
  rw [Accounting.SecurityProofs.pr_one hev, not_le]
  exact Accounting.SecurityProofs.error_one_lt

end Accounting.SecurityProofs.Remote
