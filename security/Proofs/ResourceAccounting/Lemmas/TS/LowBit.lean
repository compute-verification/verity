module

public import Definitions.Pouw.Guarantees.TS.LowBit

/-! Steps of `Definitions.Pouw.Guarantees.TS.LowBit`'s proofs, each a `def G : Prop` proved by the `SecurityProofs`
theorem of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS

def OneRegisterLowBit : Prop :=
  ∀ (w N : ℕ) (g : Fin N → BindingStep 1 (BitVec (w + 1))),
    (∀ i, ∃ h : Bool → Bool → Bool, ∀ s d, ((g i).1 s d 0).getLsbD 0 = h ((s 0).getLsbD 0) (d.getLsbD 0)) →
    ∃ c : Bool, ∀ s d,
      (foldRun (fun i => (g i).1) s d 0).getLsbD 0 =
        (Fin.foldl N (fun x i => x ^^ (d i).getLsbD 0) ((s 0).getLsbD 0) ^^ c)

end ResourceAccounting.Guarantees.TS
