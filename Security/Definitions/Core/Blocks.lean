import Definitions.Core.Circuit
import Definitions.Accounting.Device
import Definitions.Core.Game

/-!
# Server-time blocks and the wipes between them (trusted)

A server's life is cut into server-time blocks, each opened by a certified memory wipe: the device model
(`Accounting.Device`) extended by one thing, wipes. This module states what a block is, what crosses the boundary into it,
and what the wipe there bounds. It states no guarantee; a joint guarantee over a schedule of blocks reads these.

* **A block** (`Block`) of the declared circuit `C` on the device: the window of ticks it runs, and the gates of `C` it
  computes there. **A schedule of `B` blocks** (`Blocks`): the blocks in order, no gate computed in two of them. There
  is one wipe per block, the one that opens it: boundary `b` is block `b`'s, so block 0 has one too, which bounds what
  survives from before the schedule.
* **The carried state** at boundary `b` (`Blocks.carried`): the gates computed before block `b` that a gate computed
  from block `b` on reads, the values crossing a time boundary (the KV cache, say). It is derived from the circuit and
  the blocks, never declared, as a partition's committed wires are. Its value is the committed trace's
  (`Blocks.carriedValue`): the device commits it before the wipe and re-reads it after. That the value it re-reads is
  the committed one is the commitment's binding, which this module doesn't state.
* **The device's state** (`Device.State`): its on-chip state, `chip` bits, and its store, `mem` bits.
* **The state a block holds** (`Blocks.State`): the carried state's value it re-reads, and the device's state, its store
  among it.
* **The wipe** at boundary `b` (`Wipe`), the streamed-label wipe: the memory challenger sends a fresh seed, the device
  fills `fill` bits of its store with labels derived from it, and once the fill ends the challenger reveals spot reads
  one at a time and reads the labels back under a deadline. The wipe's egress, every bit that leaves the device during
  it, is counted (`egress` bits, `E`, as the network certifier counts them). The state block `b` starts from
  (`Wipe.start`) is a function of the device's state at the boundary, the seed and the spot reads.
* **What block `b` may depend on** (`Wipe.Separates`): of the device's state at the boundary and the seed, only the
  state block `b` starts from; the spot reads are public. What the prover kept off the device (the egress, NVMe, the
  network) can come back after the wipe over the side channels: `Accounting.Device.cap` limits that per tick, and no
  wipe bounds it.
* **The wipe's survivor bound** (`Wipe.SurvivorBound foreignBits ε`), at the pair the wipe's protocol publishes with an
  accepted verdict: foreign content of at least `foreignBits` bits, plus the state the fill leaves out (`Wipe.unfilled`:
  the on-chip state and the store the fill doesn't reach), is read back from the device's state when the fill ends, the
  egress and the spot reads, with the challenger accepting, with probability at most `ε`. `ε` may depend on the
  adversary's moves, so it can carry the labels' PRF advantage.

**One GPU, one store**: the device is one GPU, and its one store is the GPU's HBM
(`Accounting.Device.mem`, M). The model has no host store.

**The blocks are private.** The blocks' gate sets split the circuit's gates over time, as a partition does, and like a
partition's they are private: a statement over a schedule reads `Blocks` as the prover's value, never as a public
parameter. Its properties (the number of blocks, the carried state's size) may be public.
-/

namespace Verity

open Finset
open scoped ENNReal

variable {ι : Type}

/-- **The device's state**: its on-chip state, `chip` bits (S_chip), and its store, `mem` bits (M), the GPU's HBM. The
model has no host store. It is written `Device.State d`: `d` is an `Accounting.Device`, so `d.State` doesn't resolve. -/
structure Device.State (d : Accounting.Device ι) where
  /-- the on-chip state: registers, shared memory, caches -/
  chip : Fin d.chip → Bool
  /-- the store: the device memory, the GPU's HBM -/
  store : Fin d.mem → Bool

/-- **A server-time block** of the circuit `C` on the device: the window of ticks it runs after the wipe that opens it,
and the gates of `C` it computes there. -/
structure Block (C : Circuit) where
  /-- the ticks the block runs -/
  win : Accounting.Window
  /-- the gates of `C` whose values the block computes -/
  gates : Finset (Fin C.N)

/-- **A schedule of `B` server-time blocks** of the circuit `C` on the device: a block ends before every later block
starts, the ticks between them being the wipes', and no gate is computed in two blocks. -/
structure Blocks (C : Circuit) (B : ℕ) where
  block : Fin B → Block C
  ordered : ∀ b k, b < k → (block b).win.start + (block b).win.len ≤ (block k).win.start
  disjoint : ∀ b k, b ≠ k → Disjoint (block b).gates (block k).gates

namespace Blocks

variable {C : Circuit} {B : ℕ} (s : Blocks C B)

/-- The gates computed before block `b`. -/
def before (b : Fin B) : Finset (Fin C.N) := (univ.filter (· < b)).biUnion fun k => (s.block k).gates

/-- The gates computed from block `b` on. -/
def since (b : Fin B) : Finset (Fin C.N) := (univ.filter (b ≤ ·)).biUnion fun k => (s.block k).gates

/-- **The carried state at boundary `b`**, the wipe that opens block `b`: the gates computed before block `b` that a
gate computed from block `b` on reads. A gate computed before block `b` and read only after block `b` crosses each
boundary in between. `C.width (s.carried b)` is its size in bits. -/
def carried (b : Fin B) : Finset (Fin C.N) := (s.before b).filter fun a => ∃ g ∈ s.since b, a ∈ C.reads g

/-- A value of the carried state at boundary `b`: one for each of its gates. -/
abbrev Carried (b : Fin B) : Type := (g : s.carried b) → BitVec (C.w g)

/-- **The carried state's committed value**: what the committed trace `X` gives the carried gates. -/
def carriedValue (X : C.Transcript) (b : Fin B) : s.Carried b := fun g => X g

end Blocks

/-- **The state block `b` holds** on the device `d`, from its start: the carried state's value, re-read after the wipe
that opens the block, and the device's state, its store among it. The re-read value is the committed one
(`s.carriedValue X b`) when the commitment made before the wipe binds it, which a joint guarantee states, not this
type. -/
structure Blocks.State {C : Circuit} {B : ℕ} (s : Blocks C B) (d : Accounting.Device ι) (b : Fin B) where
  /-- the carried state's value, re-read after the wipe -/
  carried : s.Carried b
  /-- the device's state: its on-chip state and its store -/
  device : Device.State d

/-- **The wipe at boundary `b`** of the schedule `s` on the device `d`, the streamed-label wipe, with the seed drawn
from `Ω` and the spot reads from `Ch`, each uniform once the device's state at the boundary is fixed: the device fills
`fill` bits of its store with labels derived from the seed, then the memory challenger reveals the spot reads one at a
time and reads the labels back under a deadline.

* `carry σ`: the carried state's value as the device in state `σ` holds it at the boundary. The device commits it before
  the wipe and re-reads it after. The check of what it re-reads against that commitment isn't in `accepts`.
* `filled σ ω`: the device's state when the fill ends, and the egress: the `egress` bits that left the device during the
  wipe, as the network certifier counts them. Nothing enters the device during the fill but the seed.
* `accepts σ ω c`: the memory challenger's verdict.
* `after σ' ξ c`: the device's state when block `b` starts, from its state `σ'` when the fill ends, the egress `ξ` and
  the spot reads `c`. During the spot reads nothing enters the device but the spot reads and what the outside sends
  back, a function of the egress and the spot reads. -/
structure Wipe (d : Accounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B) (Ω Ch : Type) where
  /-- the bits of the device's store the fill covers -/
  fill : ℕ
  fill_le : fill ≤ d.mem
  /-- the bits that leave the device during the wipe (`E`) -/
  egress : ℕ
  /-- the carried state's value, as the device holds it at the boundary -/
  carry : Device.State d → s.Carried b
  /-- the device's state when the fill ends, and the egress -/
  filled : Device.State d → Ω → Device.State d × (Fin egress → Bool)
  /-- the memory challenger's verdict -/
  accepts : Device.State d → Ω → Ch → Prop
  /-- the device's state when block `b` starts -/
  after : Device.State d → (Fin egress → Bool) → Ch → Device.State d

namespace Wipe

variable {d : Accounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {b : Fin B} {Ω Ch : Type}
  (w : Wipe d s b Ω Ch)

/-- **The state block `b` starts from**, when the device's state at the boundary is `σ`, the seed `ω` and the spot reads
`c`: the carried value it re-reads, and the device's state the wipe leaves. -/
def start (σ : Device.State d) (ω : Ω) (c : Ch) : s.State d b :=
  ⟨w.carry σ, w.after (w.filled σ ω).1 (w.filled σ ω).2 c⟩

/-- The bits of the device's state the fill leaves out: its on-chip state and the store the fill doesn't reach. -/
def unfilled : ℕ := d.chip + (d.mem - w.fill)

/-- **Only the carried state crosses the boundary, besides what the wipe leaves**: what the device does from block `b`
on, `next σ ω c y`, from its state `σ` at the boundary, the seed `ω`, the spot reads `c` and everything that reaches it
later `y` (its ingress over the side channels and the auditors' later coins), depends on `σ` and `ω` only through the
state block `b` starts from. The spot reads are public. The seed reaches block `b` only through the device's state, so a
statement in which the seed is published later and reaches the device in `y` has to account for it there. This is the
device's side: `y` is an argument, so what reaches the device after the wipe is not bounded here, including content the
prover kept off the device (the egress among it) and sends back over a side channel. -/
def Separates {Y Z : Type} (next : Device.State d → Ω → Ch → Y → Z) : Prop :=
  ∃ g : s.State d b → Ch → Y → Z, ∀ σ ω c y, next σ ω c y = g (w.start σ ω c) c y

/-- **The wipe's survivor bound** at `foreignBits` and `ε`, in the streamed-label wipe's game (`Pous.StreamedWipe`):
for every `S_φ` of at least `foreignBits` bits plus the state the fill leaves out (`unfilled`: the on-chip state and the
store the fill doesn't reach), every way `hold` that the device's state at the boundary holds foreign content `z` of
`S_φ` bits, and every way `keep` of reading it back from the device's state when the fill ends, the egress and the spot
reads, the probability over a uniform `z`, a uniform seed and uniform spot reads that the memory challenger accepts and
`keep` gives back `z` is at most `ε S_φ hold keep`. The probability is the value of a draw with no prover move
(`Game.value` on `Game.draw`). At `foreignBits = f · fill + 2λ` the threshold is `Accounting.BlockWipe.Survives`'s.

`z` is foreign content as in PoUS's games: uniform, and independent of the wipe's coins. `keep` reads all that the state
block `b` starts from is a function of, besides the carried value, which, once bound to the committed trace, is
fixed before `z` is drawn. The on-chip state is counted, not assumed wiped: chip bits may survive any wipe.

**The pair is the protocol's.** `foreignBits` and `ε` are what the wipe's protocol publishes with an accepted verdict
(the streamed wipe's verdict publishes `foreign_bits = ⌈fill/19⌉ + E`, `E` the egress), not constants of this
definition. `ε` is a function of the adversary's moves (`S_φ`, `hold`, `keep`), so it can cover the wipe's own error,
the labels' PRF advantage against the distinguisher those moves make (the standard-model term, as
`Pous.Guarantees.StreamedWipe` adds `Pous.StreamedWipe.prfAdvantage`), and `δ_net` once the network certifier's count of
the egress enters. A constant `ε` reads the labels as ideal. `Pous.Guarantees.StreamedWipe` is the streamed game's
proved bound. The bridge from an accepted verdict to this bound is memory-accounting's, in a package that sees both. -/
def SurvivorBound [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] (foreignBits : ℕ)
    (ε : (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool) → ℝ≥0∞) : Prop :=
  ∀ S_φ : ℕ, foreignBits + w.unfilled ≤ S_φ →
    ∀ (hold : (Fin S_φ → Bool) → Device.State d)
      (keep : Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool),
      Game.value
        (fun x : (Fin S_φ → Bool) × Ω × Ch =>
          w.accepts (hold x.1) x.2.1 x.2.2 ∧
            keep (w.filled (hold x.1) x.2.1).1 (w.filled (hold x.1) x.2.1).2 x.2.2 = x.1)
        (Game.draw _) ≤ ε S_φ hold keep

end Wipe

end Verity
