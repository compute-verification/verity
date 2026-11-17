module

public import Proofs.Poub.Bandwidth.Honest
public import Proofs.Poub.RateAudit.Witnesses

/-! The honest and nontrivial lemmas of PoUB's per-pass lemma `Remote.AuditRO`, at a store of one block of one bit. -/

@[expose] public section

namespace Bandwidth

open Pous (Bits)

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

namespace ResourceAccounting.SecurityProofs.Remote

open Pous (Bits pr)
open scoped ENNReal

/-- **`AuditRO`'s honest lemma**: some device (the honest one), at one block of one bit, is accepted and reads the
store, with probability 1. -/
theorem AuditRO_honest :
    ∃ A : Bandwidth.Device 1 1 1 0 0 0 1,
      pr (fun x : Bandwidth.Store 1 1 × Bandwidth.Oracle 1 1 1 × (Fin 1 → Fin 1) =>
          Bandwidth.Accepts A x.1 x.2.1 x.2.2 ∧ ¬ A.ingress x.1 x.2.1 + 0 + 0 + 0 * 1 + 0 < 1 * 1) = 1 :=
  ⟨Bandwidth.Completeness.honest 1 1 1, ResourceAccounting.SecurityProofs.pr_one fun x => by
    obtain ⟨ha, hi⟩ := Bandwidth.Completeness.honest_accepts x.1 x.2.1 x.2.2
    refine ⟨ha, ?_⟩
    rw [hi]
    omega⟩

/-- **`AuditRO`'s nontrivial lemma**: with the check accepting everything, some device (the one that reads nothing)
falls short of the store on every coin, above the error `2^-1`. -/
theorem AuditRO_nontrivial :
    ∃ A : Bandwidth.Device 1 1 1 0 0 0 0,
      ¬ pr (fun x : Bandwidth.Store 1 1 × Bandwidth.Oracle 1 1 1 × (Fin 1 → Fin 1) =>
          True ∧ A.ingress x.1 x.2.1 + 0 + 0 + 0 * 1 + 0 < 1 * 1)
        ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
          ((Nat.choose 0 1 : ℕ) : ℝ≥0∞) / 2 ^ 0 := by
  refine ⟨Bandwidth.blank 1 1 1 0 0 0 0, ?_⟩
  have hev : ∀ x : Bandwidth.Store 1 1 × Bandwidth.Oracle 1 1 1 × (Fin 1 → Fin 1),
      True ∧ (Bandwidth.blank 1 1 1 0 0 0 0).ingress x.1 x.2.1 + 0 + 0 + 0 * 1 + 0 < 1 * 1 := fun x =>
    ⟨trivial, by rw [Bandwidth.blank_ingress]; omega⟩
  rw [ResourceAccounting.SecurityProofs.pr_one hev, not_le]
  exact ResourceAccounting.SecurityProofs.error_one_lt

end ResourceAccounting.SecurityProofs.Remote
