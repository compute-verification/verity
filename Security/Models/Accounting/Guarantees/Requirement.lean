import Definitions.Accounting.Requirement

/-!
# A declared circuit's requirement, from what the audit credits (trusted)

The remote bandwidth audit credits blocks of its store: the registered store and the memory map beside it (I3), and
the blocks each pass's receipts cover. The program's theorem is about a declared circuit `W`'s requirement instead: the
bits `W`'s execution keeps live and the bits it reads (`Definitions/Accounting/Requirement.lean`). These guarantees turn
one into the other, at each step, once the store and the plan are `W`'s block for block (`StoreMatches`,
`PlanMatches`). Their proofs are in `Security/Proofs/Accounting/Requirement.lean`.

* `RequirementHeld`: a device holding resident blocks `H` holds all of `W`'s live bits but at most a block's `B` bits for
  each resident block outside `H`.
* `RequirementHeldLive`: the same, a missing resident block costing the live bits it holds rather than `B`: `W`'s live
  bits are those in `H` plus those of the resident blocks outside `H`.
* `RequirementDense`: when each resident block holds at least `D` of `W`'s live bits, the resident blocks number at most
  `liveBits / D`, so missing a fraction `ε` of them misses at most a fraction `ε·B/D` of `W`'s live bits.
* `RequirementMoved`: a pass that moved the planned blocks `M` moved all of the bits `W` reads at that step but at most a
  block's `B` bits for each planned block outside `M`.

Frozen (no-function): counting bounds over a wiring, its execution and a store, about no function of the code.
-/

namespace Accounting.Guarantees

open Accounting.Requirement

/-- **Held blocks hold `W`'s live bits.** If the store is `W`'s when step `t` begins, a device holding the resident blocks
`H` holds every bit `W` keeps live but at most `B` for each resident block it doesn't hold:

  `liveBits t ≤ bitsAt (live t) H + |resident t \ H|·B`.

**The reading.** With the audit's verdict that the device held all but `k` of the resident blocks, it held all but
`k·B` of `W`'s live bits. The store must hold every live block for this (`liveBlocks t ⊆ resident t`): a live value in no
resident block is never audited.

Proof: `Accounting.SecurityProofs.RequirementHeld`: each live value outside `H` is in a resident block outside `H`, and
the values packed in one block total at most `B` bits.

Frozen (no-function): a counting bound over a wiring, its execution and a store, about no function of the code. -/
def RequirementHeld : Prop :=
  ∀ {G : Type} [DecidableEq G] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (s : Store) (B t : ℕ)
    (H : Finset ℕ), StoreMatches W e ℓ s B t →
    liveBits W e t ≤ bitsAt W ℓ (live W e t) H + (s.resident t \ H).card * B

/-- **Held blocks hold `W`'s live bits, a missing block costing the live bits it holds.** If the store is `W`'s when step
`t` begins, a device holding the resident blocks `H` holds every bit `W` keeps live but those of the resident blocks it
doesn't hold:

  `liveBits t = bitsAt (live t) H + Σ_{j ∈ resident t \ H} liveIn t j`.

**The reading.** An audit that credits each block it verified by `W`'s live bits in it (`liveIn`) credits exactly what
the device held of `W`'s live bits, and a missing block costs only what it holds: a partly filled block (a request's
last KV block) costs its live bits, not a block's `B`, so no density `D` (`RequirementDense`) is needed. As
`RequirementHeld`, it needs every live block resident.

Proof: `Accounting.SecurityProofs.RequirementHeldLive`: the live values split by whether their block is in `H`, and those
outside `H` by their block, each a resident block outside `H`.

Frozen (no-function): a counting identity over a wiring, its execution and a store, about no function of the code. -/
def RequirementHeldLive : Prop :=
  ∀ {G : Type} [DecidableEq G] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (s : Store) (B t : ℕ)
    (H : Finset ℕ), StoreMatches W e ℓ s B t →
    liveBits W e t = bitsAt W ℓ (live W e t) H + ∑ j ∈ s.resident t \ H, liveIn W e ℓ t j

/-- **A dense store's blocks are `W`'s bits.** If the store is `W`'s when step `t` begins and each resident block holds at
least `D` of `W`'s live bits, the resident blocks hold `D` live bits apiece:

  `|resident t|·D ≤ liveBits t`.

**The reading.** With `RequirementHeld`, a device missing a fraction `ε` of the resident blocks misses at most
`ε·|resident t|·B ≤ ε·(B/D)·liveBits t` of `W`'s live bits. Every resident block must hold live bits for this
(`resident t ⊆ liveBlocks t`): a block `W` doesn't keep live, registered or appended, credits bits `W` doesn't need.

Proof: `Accounting.SecurityProofs.RequirementDense`: the live bits are the sum of each resident block's.

Frozen (no-function): a counting bound over a wiring, its execution and a store, about no function of the code. -/
def RequirementDense : Prop :=
  ∀ {G : Type} [DecidableEq G] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (s : Store) (B t D : ℕ),
    StoreMatches W e ℓ s B t → (∀ j ∈ s.resident t, D ≤ liveIn W e ℓ t j) →
    (s.resident t).card * D ≤ liveBits W e t

/-- **Moved blocks carry `W`'s reads.** If the plan of the pass bound to step `t` is the blocks `W` reads at step `t`, a
pass that moved the planned blocks `M` moved every bit step `t` reads but at most `B` for each planned block it didn't:

  `readBits t ≤ bitsAt (reads t) M + |plan \ M|·B`.

**The reading.** Summed over a window's steps, the receipts' moved blocks bound the bits `W` reads in the window
(`Requirement.read`). The plan must hold every block `W` reads then for this, and hold no other (`PlanMatches`): a
planned block `W` doesn't read then credits a read `W` doesn't make.

Proof: `Accounting.SecurityProofs.RequirementMoved`, as `RequirementHeld` with the values read in place of the live ones.

Frozen (no-function): a counting bound over a wiring, its execution and a plan, about no function of the code. -/
def RequirementMoved : Prop :=
  ∀ {G : Type} [DecidableEq G] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (plan : Finset ℕ) (B t : ℕ)
    (M : Finset ℕ), PlanMatches W e ℓ plan B t →
    readBits W e t ≤ bitsAt W ℓ (reads W e t) M + (plan \ M).card * B

end Accounting.Guarantees
