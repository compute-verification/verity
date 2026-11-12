module

public import Definitions.Accounting.Guarantees.TS.LowBit

/-!
# A one-register fold's low bit: lemmas

The statements of `Definitions.Accounting.Guarantees.TS.LowBit` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.

* `OneRegisterLowBit`: a fold of binding steps on one register, each step's bit 0 a function of the state's and the
  word's bits 0 (as every T-function's is), has bit 0 the XOR of the start's and the words' bits 0 and a constant.
-/

@[expose] public section

namespace Accounting.Guarantees.TS

open Accounting.TS

/-- **A one-register binding fold shows its low bit as a parity.** For registers of `w + 1` bits and `N` steps on one
register, each binding its word and its state and with its result's bit 0 a function of the state's and the word's
bits 0: there is a bit `c` such that every run's bit 0 is the XOR of the start's bit 0, every word's bit 0 and `c`. -/
def OneRegisterLowBit : Prop :=
  ∀ (w N : ℕ) (g : Fin N → BindingStep 1 (BitVec (w + 1))),
    (∀ i, ∃ h : Bool → Bool → Bool, ∀ s d, ((g i).1 s d 0).getLsbD 0 = h ((s 0).getLsbD 0) (d.getLsbD 0)) →
    ∃ c : Bool, ∀ s d,
      (foldRun (fun i => (g i).1) s d 0).getLsbD 0 =
        (Fin.foldl N (fun x i => x ^^ (d i).getLsbD 0) ((s 0).getLsbD 0) ^^ c)

end Accounting.Guarantees.TS
