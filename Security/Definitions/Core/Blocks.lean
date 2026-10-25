import Definitions.Core.Circuit
import Definitions.Core.Device
import Definitions.Core.Game

/-!
# Server-time blocks and the wipes between them (trusted)

A server's life is cut into server-time blocks separated by certified memory wipes: the device model (`Verity.Device`)
extended by one thing, wipes. This module states what a block is, what crosses the boundary between two blocks, and what
the wipe at a boundary bounds. It states no guarantee; a joint guarantee over a schedule of blocks reads these.

* **A block** (`Block`) of the declared circuit `C` on one server: the window of ticks it runs between two wipes, and the
  gates of `C` it computes there, whose values are the state it holds. **A schedule of `B` blocks** (`Blocks`): the
  blocks in order, no gate computed in two of them. Boundary `j` is the wipe that opens block `j`, so block 0 has one
  too, which bounds what survives from before the schedule.
* **The carried state** at boundary `j` (`Blocks.carried`): the gates computed before block `j` that a gate computed from
  block `j` on reads, the values crossing a time boundary on one server (the KV cache, say). It is derived from the
  circuit and the blocks, never declared, as a partition's committed wires are. Its value is the committed transcript's
  (`Blocks.carriedValue`): the device commits it before the wipe and re-reads it after.
* **The device's volatile state** (`Device.State`): its on-chip state and its device memory, `chip + mem` bits.
* **The wipe** at boundary `j` (`Wipe`): the memory challenger sends a fresh seed, the device fills `fill` bits of its
  volatile state with labels derived from it, and the challenger runs timed spot reads. Its verdict, and the device's
  state when block `j` starts, are functions of the device's state at the boundary and the wipe's coins alone.
* **What block `j` may depend on** (`Wipe.Separates`): of the device's state at the boundary, only the carried value it
  re-reads and the state the wipe leaves. Everything else the device held there is wiped.
* **The wipe's survivor bound** (`Wipe.SurvivorBound f ε`), in the form of PoUS's secure-erasure requirement
  (`Pous.EraseMeets`, which `Pous.Guarantees.SecureErasureMeetsFamily` proves at `f = 1/19`, `ε = 2^-128`): foreign
  content of at least `f · fill + 2λ` bits (`λ = 128`), plus the volatile state the fill leaves out, is read back from
  what block `j` starts from with probability at most `ε`.

**Host DRAM is outside the model.** It is not part of the device's state, and a wipe bounds nothing kept there. During
the wipe nothing enters the device, as in PoUS's secure-erasure game (perfect isolation). After it, whatever host DRAM,
NVMe or the network hands the device is ingress over its side channels: `Device.cap` limits it per tick, and no wipe
bounds what it carries.
-/

namespace Verity

open Finset
open scoped ENNReal

variable {ι : Type}

/-- **The device's volatile state**: its on-chip state and its device memory, `chip + mem` bits. Host DRAM is not in
it. -/
abbrev Device.State (d : Device ι) : Type := Fin (d.chip + d.mem) → Bool

/-- **A server-time block** of the circuit `C` on one server: the window of ticks it runs between two wipes, and the
gates of `C` it computes there, whose values are the state it holds. -/
structure Block (C : Circuit) where
  /-- the ticks the block runs -/
  win : Window
  /-- the gates of `C` whose values the block computes -/
  gates : Finset (Fin C.N)

/-- **A schedule of `B` server-time blocks** of the circuit `C` on one server: a block ends before every later block
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

/-- **The wipe at boundary `j`** of the schedule `s` on the device `d`, with coins `Ω` (the seed and the spot reads'
indices, drawn once the device's state at the boundary is fixed): the device fills `fill` bits of its volatile state
with labels derived from the seed, and the memory challenger reads them back at random places under a deadline.

* `carry σ`: the carried state's value as the device in state `σ` holds it at the boundary. The device commits it before
  the wipe and re-reads it after, checked against that commitment.
* `accepts σ ω`: the memory challenger's verdict. `after σ ω`: the device's volatile state when block `j` starts. Nothing
  enters the device during the wipe, so both are functions of its state at the boundary and the coins alone. -/
structure Wipe (d : Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (j : Fin B) (Ω : Type) where
  /-- the bits of the device's volatile state the fill covers -/
  fill : ℕ
  fill_le : fill ≤ d.chip + d.mem
  /-- the carried state's value, as the device holds it at the boundary -/
  carry : d.State → s.Carried j
  /-- the memory challenger's verdict -/
  accepts : d.State → Ω → Prop
  /-- the device's volatile state when block `j` starts -/
  after : d.State → Ω → d.State

namespace Wipe

variable {d : Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {j : Fin B} {Ω : Type} (w : Wipe d s j Ω)

/-- The bits of the device's volatile state the fill leaves out: on-chip state, and device memory the fill doesn't
reach. -/
def unfilled : ℕ := d.chip + d.mem - w.fill

/-- **Only the carried state crosses the boundary, besides what the wipe leaves**: what the device does from block `j`
on, `next σ ω y`, from its volatile state `σ` at the boundary, the wipe's coins `ω` (public) and everything that reaches
it later `y` (its ingress over the side channels and the auditors' later coins), depends on `σ` only through the carried
value it re-reads and the state the wipe leaves. This is the device's side: `y` is an argument, and what reaches the
device from outside it (host DRAM among them) is not bounded here. -/
def Separates {Y Z : Type} (next : d.State → Ω → Y → Z) : Prop :=
  ∃ g : s.Carried j → d.State → Ω → Y → Z, ∀ σ ω y, next σ ω y = g (w.carry σ) (w.after σ ω) ω y

/-- **The wipe's survivor bound** at fraction `f` and error `ε`, in the form of `Pous.EraseMeets`' erasure clause, over
the device's whole volatile state: for every `S_φ` of at least `f · fill + 2λ` bits (`λ = 128`) plus the unfilled state,
every way `hold` that the device's state at the boundary holds foreign content `z` of `S_φ` bits, and every way `keep`
of reading it back from the state block `j` starts from and the wipe's coins, the probability over a uniform `z` and the
wipe's uniform coins that the memory challenger accepts and `keep` gives back `z` is at most `ε`. The probability is the
value of a draw of `(z, ω)` with no prover move (`Game.value` on `Game.draw`).

So `z` is foreign content as in PoUS's secure-erasure game: uniform, and independent of the wipe's coins. Two things
differ from `EraseMeets`, and a bridge from `Pous.Guarantees.SecureErasureMeetsFamily` to this bound has to cover both.
`keep` reads the state the next block starts from, after the last spot read, and the wipe's coins, which the device saw:
what block `j` may read (`Separates`). `EraseMeets`' keep reads the memory alone, at every spot read's arrival. And the
fill may cover less than the device's volatile state, the rest added to what may survive; `EraseMeets` takes a fill that
covers the prover's whole volatile memory. -/
def SurvivorBound [Fintype Ω] [Nonempty Ω] (f : ℚ) (ε : ℝ≥0∞) : Prop :=
  ∀ S_φ : ℕ, f * w.fill + 2 * 128 + w.unfilled ≤ (S_φ : ℚ) →
    ∀ (hold : (Fin S_φ → Bool) → d.State) (keep : d.State → Ω → Fin S_φ → Bool),
      Game.value (fun x : (Fin S_φ → Bool) × Ω => w.accepts (hold x.1) x.2 ∧ keep (w.after (hold x.1) x.2) x.2 = x.1)
        (Game.draw _) ≤ ε

end Wipe

end Verity
