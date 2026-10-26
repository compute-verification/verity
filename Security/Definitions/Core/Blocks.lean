import Definitions.Core.Circuit
import Definitions.Core.Device
import Definitions.Core.Game

/-!
# Server-time blocks and the wipes between them (trusted)

A server's life is cut into server-time blocks, each opened by a certified memory wipe: the device model
(`Verity.Device`) extended by one thing, wipes. This module states what a block is, what crosses the boundary into it,
and what the wipe there bounds. It states no guarantee; a joint guarantee over a schedule of blocks reads these.

* **A block** (`Block`) of the declared circuit `C` on the device: the window of ticks it runs, and the gates of `C` it
  computes there. **A schedule of `B` blocks** (`Blocks`): the blocks in order, no gate computed in two of them. There
  is one wipe per block, the one that opens it: boundary `j` is block `j`'s, so block 0 has one too, which bounds what
  survives from before the schedule.
* **The carried state** at boundary `j` (`Blocks.carried`): the gates computed before block `j` that a gate computed from
  block `j` on reads, the values crossing a time boundary (the KV cache, say). It is derived from the circuit and the
  blocks, never declared, as a partition's committed wires are. Its value is the committed transcript's
  (`Blocks.carriedValue`): the device commits it before the wipe and re-reads it after. That the value it re-reads is
  the committed one is the commitment's binding, which this module doesn't state.
* **The device's state** (`Device.State`): its on-chip state, `chip` bits, and its store, `mem` bits.
* **The state a block holds** (`Blocks.State`): the carried state's value it re-reads, and the device's state, its store
  among it.
* **The wipe** at boundary `j` (`Wipe`), the streamed-label wipe: the memory challenger sends a fresh seed, the device
  fills `fill` bits of its store with labels derived from it, and once the fill ends the challenger reveals spot reads
  one at a time and reads the labels back under a deadline. The wipe's egress, every bit that leaves the device during
  it, is counted (`egress` bits, `E`, as the network certifier counts them). The state block `j` starts from
  (`Wipe.start`) is a function of the device's state at the boundary, the seed and the spot reads.
* **What block `j` may depend on** (`Wipe.Separates`): of the device's state at the boundary and the seed, only the
  state block `j` starts from; the spot reads are public. What the prover kept off the device (the egress, NVMe, the
  network) can come back after the wipe over the side channels: `Device.cap` limits that per tick, and no wipe bounds
  it.
* **The wipe's survivor bound** (`Wipe.SurvivorBound foreignBits ε`), the pair an accepted verdict of the deployed wipe
  publishes (`foreign_bits = ⌈fill/19⌉ + E` at `eps = 2^-128`): foreign content of at least `foreignBits` bits, plus the
  store the fill leaves out, plus the on-chip state, is read back from the device's state when the fill ends, the egress
  and the spot reads, with the challenger accepting, with probability at most `ε`.

**One GPU, one store** (Daniel's rulings of 25 Oct): the device is one GPU, and its one store is the GPU's HBM
(`Device.mem`, M). The model has no host store.

**The blocks are private.** The blocks' gate sets split the circuit's gates over time, as a partition does, and like a
partition's they are private: a statement over a schedule reads `Blocks` as the prover's value, never as a public
parameter. Its properties (the number of blocks, the carried state's size) may be public.
-/

namespace Verity

open Finset
open scoped ENNReal

variable {ι : Type}

/-- **The device's state**: its on-chip state, `chip` bits (S_chip), and its store, `mem` bits (M), the GPU's HBM. The
model has no host store. -/
structure Device.State (d : Device ι) where
  /-- the on-chip state: registers, shared memory, caches -/
  chip : Fin d.chip → Bool
  /-- the store: the device memory, the GPU's HBM -/
  store : Fin d.mem → Bool

/-- **A server-time block** of the circuit `C` on the device: the window of ticks it runs after the wipe that opens it,
and the gates of `C` it computes there. -/
structure Block (C : Circuit) where
  /-- the ticks the block runs -/
  win : Window
  /-- the gates of `C` whose values the block computes -/
  gates : Finset (Fin C.N)

/-- **A schedule of `B` server-time blocks** of the circuit `C` on the device: a block ends before every later block
starts, the ticks between them being the wipes', and no gate is computed in two blocks. -/
structure Blocks (C : Circuit) (B : ℕ) where
  block : Fin B → Block C
  ordered : ∀ j k, j < k → (block j).win.start + (block j).win.len ≤ (block k).win.start
  disjoint : ∀ j k, j ≠ k → Disjoint (block j).gates (block k).gates

namespace Blocks

variable {C : Circuit} {B : ℕ} (s : Blocks C B)

/-- The gates computed before block `j`. -/
def before (j : Fin B) : Finset (Fin C.N) := (univ.filter (· < j)).biUnion fun k => (s.block k).gates

/-- The gates computed from block `j` on. -/
def since (j : Fin B) : Finset (Fin C.N) := (univ.filter (j ≤ ·)).biUnion fun k => (s.block k).gates

/-- **The carried state at boundary `j`**, the wipe that opens block `j`: the gates computed before block `j` that a gate
computed from block `j` on reads. A gate computed before block `j` and read only after block `j` crosses each boundary
in between. `C.width (s.carried j)` is its size in bits. -/
def carried (j : Fin B) : Finset (Fin C.N) := (s.before j).filter fun a => ∃ g ∈ s.since j, a ∈ C.reads g

/-- A value of the carried state at boundary `j`: one for each of its gates. -/
abbrev Carried (j : Fin B) : Type := (g : s.carried j) → BitVec (C.w g)

/-- **The carried state's committed value**: what the committed transcript `X` gives the carried gates. -/
def carriedValue (X : C.Transcript) (j : Fin B) : s.Carried j := fun g => X g

end Blocks

/-- **The state block `j` holds** on the device `d`, from its start: the carried state's value, re-read after the wipe
that opens the block, and the device's state, its store among it. The re-read value is the committed one
(`s.carriedValue X j`) when the commitment made before the wipe binds it, which a joint guarantee states, not this
type. -/
structure Blocks.State {C : Circuit} {B : ℕ} (s : Blocks C B) (d : Device ι) (j : Fin B) where
  /-- the carried state's value, re-read after the wipe -/
  carried : s.Carried j
  /-- the device's state: its on-chip state and its store -/
  device : d.State

/-- **The wipe at boundary `j`** of the schedule `s` on the device `d`, the streamed-label wipe, with the seed drawn from
`Ω` and the spot reads from `Ch`, each uniform once the device's state at the boundary is fixed: the device fills `fill`
bits of its store with labels derived from the seed, then the memory challenger reveals the spot reads one at a time
and reads the labels back under a deadline.

* `carry σ`: the carried state's value as the device in state `σ` holds it at the boundary. The device commits it before
  the wipe and re-reads it after. The check of what it re-reads against that commitment isn't in `accepts`.
* `filled σ ω`: the device's state when the fill ends, and the egress: the `egress` bits that left the device during the
  wipe, as the network certifier counts them. Nothing enters the device during the fill but the seed.
* `accepts σ ω c`: the memory challenger's verdict.
* `after σ' ξ c`: the device's state when block `j` starts, from its state `σ'` when the fill ends, the egress `ξ` and
  the spot reads `c`. During the spot reads nothing enters the device but the spot reads and what the outside sends
  back, a function of the egress and the spot reads. -/
structure Wipe (d : Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (j : Fin B) (Ω Ch : Type) where
  /-- the bits of the device's store the fill covers -/
  fill : ℕ
  fill_le : fill ≤ d.mem
  /-- the bits that leave the device during the wipe (`E`) -/
  egress : ℕ
  /-- the carried state's value, as the device holds it at the boundary -/
  carry : d.State → s.Carried j
  /-- the device's state when the fill ends, and the egress -/
  filled : d.State → Ω → d.State × (Fin egress → Bool)
  /-- the memory challenger's verdict -/
  accepts : d.State → Ω → Ch → Prop
  /-- the device's state when block `j` starts -/
  after : d.State → (Fin egress → Bool) → Ch → d.State

namespace Wipe

variable {d : Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {j : Fin B} {Ω Ch : Type} (w : Wipe d s j Ω Ch)

/-- **The state block `j` starts from**, when the device's state at the boundary is `σ`, the seed `ω` and the spot reads
`c`: the carried value it re-reads, and the device's state the wipe leaves. -/
def start (σ : d.State) (ω : Ω) (c : Ch) : s.State d j :=
  ⟨w.carry σ, w.after (w.filled σ ω).1 (w.filled σ ω).2 c⟩

/-- The bits of the store the fill doesn't reach. -/
def unfilled : ℕ := d.mem - w.fill

/-- **Only the carried state crosses the boundary, besides what the wipe leaves**: what the device does from block `j`
on, `next σ ω c y`, from its state `σ` at the boundary, the seed `ω`, the spot reads `c` and everything that reaches it
later `y` (its ingress over the side channels and the auditors' later coins), depends on `σ` and `ω` only through the
state block `j` starts from. The spot reads are public. The seed reaches block `j` only through the device's state, so a
statement in which the seed is published later and reaches the device in `y` has to account for it there. This is the
device's side: `y` is an argument, so what reaches the device after the wipe is not bounded here, including content the
prover kept off the device (the egress among it) and sends back over a side channel. -/
def Separates {Y Z : Type} (next : d.State → Ω → Ch → Y → Z) : Prop :=
  ∃ g : s.State d j → Ch → Y → Z, ∀ σ ω c y, next σ ω c y = g (w.start σ ω c) c y

/-- **The wipe's survivor bound** at `foreignBits` and `ε`, the pair an accepted verdict of the deployed wipe publishes
(`foreign_bits = ⌈fill/19⌉ + E` at `eps = 2^-128`, `E` the egress), in the streamed-label wipe's game
(`Pous.StreamedWipe`): for every `S_φ` of at least `foreignBits` bits plus the unfilled store plus the on-chip state,
every way `hold` that the device's state at the boundary holds foreign content `z` of `S_φ` bits, and every way `keep`
of reading it back from the device's state when the fill ends, the egress and the spot reads, the probability over a
uniform `z`, a uniform seed and uniform spot reads that the memory challenger accepts and `keep` gives back `z` is at
most `ε`. The probability is the value of a draw with no prover move (`Game.value` on `Game.draw`).

`z` is foreign content as in PoUS's games: uniform, and independent of the wipe's coins. `keep` reads all that the state
block `j` starts from is a function of, besides the carried value, which, once bound to the committed transcript, is
fixed before `z` is drawn. The on-chip state is counted, not assumed wiped: chip bits may survive any wipe.

The bound is a parameter. `Pous.Guarantees.SecureErasureMeetsFamily` is the proved family bound underneath it, and
`Pous.Guarantees.StreamedWipe` the streamed game's proved bound. The bridge from an accepted verdict to this bound is
memory-accounting's, in a package that sees both, and it has to account for the labels' PRF advantage, which the
published `eps` leaves out. -/
def SurvivorBound [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] (foreignBits : ℕ) (ε : ℝ≥0∞) : Prop :=
  ∀ S_φ : ℕ, foreignBits + w.unfilled + d.chip ≤ S_φ →
    ∀ (hold : (Fin S_φ → Bool) → d.State) (keep : d.State → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool),
      Game.value
        (fun x : (Fin S_φ → Bool) × Ω × Ch =>
          w.accepts (hold x.1) x.2.1 x.2.2 ∧
            keep (w.filled (hold x.1) x.2.1).1 (w.filled (hold x.1) x.2.1).2 x.2.2 = x.1)
        (Game.draw _) ≤ ε

end Wipe

end Verity
