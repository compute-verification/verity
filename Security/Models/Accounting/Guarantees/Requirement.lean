import Definitions.Accounting.Requirement

/-!
# A declared circuit's requirement, from what the audit credits (trusted)

The remote bandwidth audit credits blocks of its store: the registered store and the memory map beside it (I3), and
the blocks each pass's receipts cover. The program's theorem is about a declared circuit `W`'s requirement instead: the
bits `W`'s execution keeps live and the bits it reads (`Definitions/Accounting/Requirement.lean`). These guarantees turn
one into the other, at each engine step, once the store and the plan are `W`'s block for block (`StoreMatches`,
`PlanMatches`). The audit credits the live values in complete blocks (`credited`): a KV block later steps fill is in no
record yet. The first four bound what it credits, and the last two bound what it leaves out. Their proofs are in
`Security/Proofs/Accounting/Requirement.lean`.

* `RequirementHeld`: a device holding resident blocks `H` holds all of `W`'s credited live bits but at most a block's
  `B` bits for each resident block outside `H`.
* `RequirementHeldLive`: the same, a missing resident block costing the live bits it holds rather than `B`: `W`'s
  credited live bits are those in `H` plus those of the resident blocks outside `H`.
* `RequirementDense`: when each resident block holds at least `D` of `W`'s live bits, the resident blocks number at most
  `creditedBits / D`, so missing a fraction `ε` of them misses at most a fraction `ε·B/D` of `W`'s credited live bits.
* `RequirementMoved`: a pass that moved the planned blocks `M` moved all of the bits `W` reads from complete blocks at
  that step but at most a block's `B` bits for each planned block outside `M`.
* `RequirementUncredited`: when the live values in incomplete blocks lie in one block per group (a sequence, layer and
  plane), `W`'s live bits exceed the credited ones by at most `B` per group.
* `RequirementUnplanned`: the same for the bits a step reads from incomplete blocks.

Frozen (no-function): counting bounds over a wiring, its execution and a store, about no function of the code.
-/

namespace Accounting.Guarantees

open Accounting.Requirement

/-- **Held blocks hold `W`'s credited bits.** If the store is `W`'s when step `s` begins, a device holding the resident
blocks `H` holds every live bit the audit credits but at most `B` for each resident block it doesn't hold:

  `creditedBits s ≤ bitsAt (credited s) H + |resident s \ H|·B`.

**The reading.** With the audit's verdict that the device held all but `k` of the resident blocks, it held all but
`k·B` of `W`'s credited live bits, and `RequirementUncredited` bounds the rest of `W`'s live bits. The store must hold
every complete block that holds a live value for this (`blocks (credited s) ⊆ resident s`): a credited value in no
resident block is never audited.

Proof: `Accounting.SecurityProofs.RequirementHeld`: each credited value outside `H` is in a resident block outside `H`,
and the values packed in one block total at most `B` bits.

Frozen (no-function): a counting bound over a wiring, its execution and a store, about no function of the code. -/
def RequirementHeld : Prop :=
  ∀ {G : Type} [DecidableEq G] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (σ : Store) (B s : ℕ)
    (H : Finset ℕ), StoreMatches W e ℓ σ B s →
    creditedBits W e ℓ s ≤ bitsAt W ℓ (credited W e ℓ s) H + (σ.resident s \ H).card * B

/-- **Held blocks hold `W`'s credited bits, a missing block costing the live bits it holds.** If the store is `W`'s when
step `s` begins, a device holding the resident blocks `H` holds every live bit the audit credits but those of the
resident blocks it doesn't hold:

  `creditedBits s = bitsAt (credited s) H + Σ_{j ∈ resident s \ H} liveIn s j`.

**The reading.** An audit that credits each block it verified by `W`'s live bits in it (`liveIn`) credits exactly what
the device held of `W`'s credited live bits, and a missing block costs only what it holds: a block holding fewer live
bits than `B` (some of its values dead, or the registered store's last block) costs its live bits, not a block's `B`,
so no density `D` (`RequirementDense`) is needed. A resident block is complete, so all its live bits are credited. As
`RequirementHeld`, it needs every complete block that holds a live value resident.

Proof: `Accounting.SecurityProofs.RequirementHeldLive`: the credited values split by whether their block is in `H`, and
those outside `H` by their block, each a resident block outside `H`, whose live values are all credited.

Frozen (no-function): a counting identity over a wiring, its execution and a store, about no function of the code. -/
def RequirementHeldLive : Prop :=
  ∀ {G : Type} [DecidableEq G] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (σ : Store) (B s : ℕ)
    (H : Finset ℕ), StoreMatches W e ℓ σ B s →
    creditedBits W e ℓ s = bitsAt W ℓ (credited W e ℓ s) H + ∑ j ∈ σ.resident s \ H, liveIn W e ℓ s j

/-- **A dense store's blocks are `W`'s bits.** If the store is `W`'s when step `s` begins and each resident block holds at
least `D` of `W`'s live bits, the resident blocks hold `D` credited live bits apiece:

  `|resident s|·D ≤ creditedBits s`.

**The reading.** With `RequirementHeld`, a device missing a fraction `ε` of the resident blocks misses at most
`ε·|resident s|·B ≤ ε·(B/D)·creditedBits s` of `W`'s credited live bits: it held at least `(1 - ε·B/D)` of them. Every
resident block must hold live bits for this (`resident s ⊆ blocks (credited s)`): a block `W` doesn't keep live,
registered or appended, credits bits `W` doesn't need.

Proof: `Accounting.SecurityProofs.RequirementDense`: the credited bits are the sum of each resident block's live bits.

Frozen (no-function): a counting bound over a wiring, its execution and a store, about no function of the code. -/
def RequirementDense : Prop :=
  ∀ {G : Type} [DecidableEq G] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (σ : Store) (B s D : ℕ),
    StoreMatches W e ℓ σ B s → (∀ j ∈ σ.resident s, D ≤ liveIn W e ℓ s j) →
    (σ.resident s).card * D ≤ creditedBits W e ℓ s

/-- **Moved blocks carry `W`'s reads.** If the plan of the pass bound to step `s` is the complete blocks `W` reads at step
`s`, a pass that moved the planned blocks `M` moved every bit step `s` reads from complete blocks but at most `B` for
each planned block it didn't:

  `creditedReadBits s ≤ bitsAt (creditedReads s) M + |plan \ M|·B`.

**The reading.** Summed over a window's steps, the receipts' moved blocks bound the bits `W` reads from complete blocks
in the window, and `RequirementUnplanned` bounds the rest. The plan must hold every complete block `W` reads then for
this, and hold no other (`PlanMatches`): a planned block `W` doesn't read then credits a read `W` doesn't make.

Proof: `Accounting.SecurityProofs.RequirementMoved`, as `RequirementHeld` with the values read in place of the live ones.

Frozen (no-function): a counting bound over a wiring, its execution and a plan, about no function of the code. -/
def RequirementMoved : Prop :=
  ∀ {G : Type} [DecidableEq G] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (plan : Finset ℕ) (B s : ℕ)
    (M : Finset ℕ), PlanMatches W e ℓ plan B s →
    creditedReadBits W e ℓ s ≤ bitsAt W ℓ (creditedReads W e ℓ s) M + (plan \ M).card * B

/-- **What the audit doesn't credit is a block per group.** If the store is `W`'s when step `s` begins and the live
values in incomplete blocks lie in one block per group of `g` (`OneBlockEach`), `W`'s live bits exceed its credited
live bits by at most `B` per group that has such a value:

  `liveBits s ≤ creditedBits s + |g(live s \ credited s)|·B`.

**The reading.** With `g` a KV value's sequence, layer and plane, a KV pool fills each one's pages in turn, so the live
bits the audit can't credit, those in blocks still being written, are at most `B` per sequence, layer and plane being
served. With `RequirementHeld`, a device that held all but `k` resident blocks held all of `W`'s live bits but `k·B`
and that share. A block of registered values is complete from the start, so the weights add nothing to the share.

Proof: `Accounting.SecurityProofs.RequirementUncredited`: the live values outside `credited` split by group, and the
values of a group lie in one block, packed into its `B` bits.

Frozen (no-function): a counting bound over a wiring, its execution and a layout, about no function of the code. -/
def RequirementUncredited : Prop :=
  ∀ {G : Type} [DecidableEq G] {K : Type} [DecidableEq K] (W : Wiring G) (e : Execution G) (ℓ : Layout G) (σ : Store)
    (B s : ℕ) (g : G → K), StoreMatches W e ℓ σ B s → OneBlockEach ℓ g (live W e s \ credited W e ℓ s) →
    liveBits W e s ≤ creditedBits W e ℓ s + ((live W e s \ credited W e ℓ s).image g).card * B

/-- **What no plan holds is a block per group.** If the plan of the pass bound to step `s` is `W`'s and the values step
`s` reads from incomplete blocks lie in one block per group of `g` (`OneBlockEach`), the bits step `s` reads exceed
those it reads from complete blocks by at most `B` per group that has such a value:

  `readBits s ≤ creditedReadBits s + |g(reads s \ creditedReads s)|·B`.

**The reading.** As `RequirementUncredited`, for the reads: with `g` a KV value's sequence, layer and plane, the reads
no receipt can cover are at most `B` per sequence, layer and plane being served.

Proof: `Accounting.SecurityProofs.RequirementUnplanned`, as `RequirementUncredited` with the values read in place of
the live ones.

Frozen (no-function): a counting bound over a wiring, its execution and a layout, about no function of the code. -/
def RequirementUnplanned : Prop :=
  ∀ {G : Type} [DecidableEq G] {K : Type} [DecidableEq K] (W : Wiring G) (e : Execution G) (ℓ : Layout G)
    (plan : Finset ℕ) (B s : ℕ) (g : G → K), PlanMatches W e ℓ plan B s →
    OneBlockEach ℓ g (reads W e s \ creditedReads W e ℓ s) →
    readBits W e s ≤ creditedReadBits W e ℓ s + ((reads W e s \ creditedReads W e ℓ s).image g).card * B

end Accounting.Guarantees
