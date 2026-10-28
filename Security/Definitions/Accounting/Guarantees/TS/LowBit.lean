import Definitions.Accounting.TS.Fold
import Definitions.Pouw.Receipt.Li

/-!
# A one-register fold's low bit (trusted)

What a lower bound that reads a fold through its low bit can see, each `def G : Prop` proved as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS.LowBit`:

* `OneRegisterLowBit`: a fold of binding steps on one register, each step's bit 0 a function of the state's and the
  word's bits 0 (as every T-function's is), has bit 0 the XOR of the start's and the words' bits 0 and a constant.
* `LiLowBit`: li's chain with odd multipliers has bit 0 the XOR of its start's, its keys' and its words' bits 0.

A step that binds its word and its state on one register has no other choice at bit 0: a function of two bits that is a
bijection in each is their XOR or its negation. So a fold of such steps, reduced modulo 2, is an XOR of its words' low
bits, and any lower bound read off that reduction (its multiplicative complexity, its degree, or a Boolean circuit bound
on bit 0) is a bound on XOR. A tree of `f`-input nodes computes that XOR with `⌈(N − 1)/(f − 1)⌉` of them, which is no
more than `FoldFanIn` gives. A residue argument beats the fan-in floor only on a fold whose state has more than one
register.
-/

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

/-- **li's low bit is a parity.** For words of `n + 1` bits, any half steps with odd multipliers and any start `y`:
bit 0 of li's chain is the XOR of `y`'s bit 0 and, half step by half step, the bits 0 of `d₀`, the key `k` and
`d₁`. -/
def LiLowBit : Prop :=
  ∀ (n : ℕ) (hs : List (Pouw.Receipt.LiOddHalfStep (n + 1))) (y : BitVec (n + 1)),
    (Pouw.Receipt.liChain (hs.map Pouw.Receipt.LiOddHalfStep.val) y).getLsbD 0 =
      hs.foldl (fun x h => x ^^ h.2.2.1.getLsbD 0 ^^ h.1.getLsbD 0 ^^ h.2.2.2.getLsbD 0) (y.getLsbD 0)

end Accounting.Guarantees.TS
