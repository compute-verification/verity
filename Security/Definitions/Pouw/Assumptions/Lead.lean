import Definitions.Pouw.Lead.Gap
import Definitions.Pouw.Assumptions.Dimension.H100W
import Definitions.Core.Assumptions

/-!
# The lead's named premises (trusted statements)

The premises `Pouw.Guarantees.Lead.G` reads, beside `H32` (`Pouw.Dimension.H100.H32`, reused). Each is a named
`Prop` about `Real`, the strategies the node can run on a call, or about the protocol's devices. Each docstring opens
with its claim id; a `(proposed)` id is not yet in `verity.core.claims`.

The device and model premises:
* `W1`: the node's instructions are `MH100w`'s, priced by the W1 table, with no free value at a coin value.
* `TileGrain` (CopyGrain at the tile width, under model W): what the node holds, its run at tile grain holds.
* `LaneSemantics`: each word the run writes is a register value its trace computes.
* `OneInstructionMinimal`: each word written and each compression register is its own register of the trace.
* `CardWithinAccelW1`: every strategy the node's processors run, hosts counted, is in `Real`.
* `ShaWrites`: a SHA-512 compression writes about 3,500 registers, credited per form.

The protocol premises the code can't be checked against yet:
* `HonestHashRuled`: an accepted execution hashes, for each message of the honest commitment, a message with its digest.
* `TrustedNebiusHost`: the network certifier draws `G` and streams the coin lanes after it holds `A_j`.
* `CommitL1w`: the honest commitment is the lead's layout, at its credited compressions.
* `OutsourceWithinEgressLead`: work outsourced within the egress cap is at most the network term.
-/

namespace Pouw.Assumptions.Lead

open Pouw.Lead Pouw.Dimension.H100 Pouw.Dimension.Pointwise

variable {L : Pouw.Lead.Layout}

/-- `pouw-w1` (proposed). **The node's instructions are W1's**: every instruction a realizable strategy executes at a
coin value is a well-formed `MH100w` instruction with no free operand (`Op100.WFw pr ∅`), priced by the W1 table `pr`.
Values fixed before the coins enter through the strategy's pre-coin tables, never as free operands.

Rationale: W1 is the work model the card's measured prices define; it prices every register writer by its class, and
a strategy's priced work is the sum (`Strategy.work`). -/
def W1 (pr : Prices100) (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, ∀ o ∈ (σ.exec ω).trace, o.WFw pr ∅

/-- `pouw-tile-grain` (proposed; CopyGrain at the tile width, under model W). **What the node holds, its run at tile
grain holds**: at a coin value where a realizable strategy exchanges nothing, a tile its execution holds whole is held
by its run (`Exec.steps`, from the pre-coin tables and the coin stream) at some point. The run's only free step is a
whole MMA tile's copy (`tileWords` words); writing any word of a tile writes the tile (model W at the tile's width).

Rationale: on the measured kernels a reuse saves time only where a whole MMA tile (16 rows × 8 columns) repeats;
this is the copy model, one line to change. -/
def TileGrain (stream : CoinStream L) (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, (σ.exec ω).egress = 0 → ∀ y, (σ.exec ω).holds y →
    ∃ k a, run (σ.init stream ω) ((σ.exec ω).steps.take k) a 0 = y

/-- `pouw-lane-semantics` (proposed). **Each word the run writes is a register value its trace computes**: at every
coin value, every word of every tile a realizable strategy's run writes is the value of a register of its trace under
the measured lane functions `fpSem`.

Rationale: the lane forms' semantics are measured on the card. -/
def LaneSemantics (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, ∀ a p v, Step.write a p v ∈ (σ.exec ω).steps →
    ∀ q, (fun _ : Unit => (v q : ℚ)) ∈ regs100 fpSem (σ.exec ω).trace []

/-- `pouw-one-instruction-minimal` (proposed). **Each word written and each compression register is its own register
of the trace**: at every coin value, a realizable strategy's run's tile writes (`tileWords` registers each) and its
SHA-512 compressions' registers number at most the registers its trace writes.

Rationale: one instruction writes each register, and its price per register is its class's (`H32` floors it). -/
def OneInstructionMinimal (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, tileWords * writes (σ.exec ω).steps + (σ.exec ω).shaRegs ≤ regCount (σ.exec ω).trace

/-- `pouw-card-within-accel-w1` (proposed; `CardWithinAccel` and `PeriodInventory` at W1, hosts counted).
**Every strategy the node's processors run is realizable**: what the node's cards, host CPUs and their AMX units
(`can`) run on a call is a strategy in `Real`, at its priced W1 work.

Rationale: the card's instruction set is `MH100w`'s at W1 (the lane forms' census), and the period's inventory counts
the hosts. -/
def CardWithinAccelW1 (can Real : Strategy L → Prop) : Prop :=
  ∀ σ, can σ → Real σ

/-- The registers ShaWrites credits a message: `3,449.05` a compression for an A message (`isA`) and `3,515.38` for a
words message. -/
def shaCredit (isA : Hash.Msg → Bool) (x : Hash.Msg) : ℚ :=
  (if isA x then shaARegs else shaWordsRegs) * blocks x

/-- `pouw-sha-writes` (proposed; Daniel's ruling of 29 October, 16:00Z: a premise milder than SAK). **A SHA-512
compression writes about 3,500 registers**, credited per form: at every coin value, a realizable strategy's
compressions write at least `3,515.38` registers each for the words and `3,449.05` for A (`shaCredit`), over the
messages it hashes.

Rationale: the measured compression forms on the card; the ruling accepts the count as a premise. -/
def ShaWrites (isA : Hash.Msg → Bool) (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, ((σ.exec ω).hashed.map (shaCredit isA)).sum ≤ (σ.exec ω).shaRegs

/-- `pouw-honest-hash-ruled` (proposed; a ruling: the honest hashing is the minimum). **An accepted execution hashes
the honest commitment's messages**: for each message the honest commitment of the call's words hashes, a realizable
strategy's accepted execution hashes a message with its digest.

Rationale: the premise counts registers; a digest is reached only by hashing. -/
def HonestHashRuled (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, Accepted L H C κ σ ω →
    ∀ y ∈ C.msgs H (word L κ ω) ++ C.aMsgs, ∃ x ∈ (σ.exec ω).hashed, H x = H y

/-- `pouw-commit-l1w` (proposed; the commitments' layout `l1w`, until the Lean code is the served format).
**The honest commitment is the lead's layout**: its words chunks number `wordChunks L`, and at every coin value its
distinct messages' ShaWrites credit is at least `3,515.38` a words compression of `wordsBlocks` and `3,449.05` an A
compression of `aBlocks`.

Rationale: the served commitment hashes each 256 KiB words chunk under one leaf and A's rows under its own. -/
def CommitL1w (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L) (isA : Hash.Msg → Bool)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) : Prop :=
  C.chunks = wordChunks L ∧ ∀ ω : Coins L,
    shaWordsRegs * wordsBlocks L C + shaARegs * aBlocks L ≤
      (((C.msgs H (word L κ ω) ++ C.aMsgs).dedup).map (shaCredit isA)).sum

/-- The network certifier's code on a call: it holds `A_j`'s root, then draws the coins from its own entropy and
streams them, bound to that root. -/
def certCode : Hash.Digest × Coins L → Hash.Digest × Coins L := id

/-- `trusted-host`, at the lead's network certifier (`trusted-nebius-host`). **The certifier draws `G` and streams the
coin lanes after it holds `A_j`**: the certifier device runs `certCode` on its host, so the coins it streams are its own
draw, uniform and fixed after the call's `A_j` and the strategy's pre-coin tables.

Rationale: the auditor's hardware on a host the prover doesn't control. -/
def TrustedNebiusHost {Device : Type} (cert : Device)
    (runC : Device → Hash.Digest × Coins L → Hash.Digest × Coins L) : Prop :=
  Verity.Assumptions.TrustedHost cert runC certCode

/-- `pouw-outsource-egress` (proposed; the lead's instance at its cap). **Outsourcing within the egress cap costs at
most the network term**: every realizable strategy has a realizable twin that exchanges nothing, with the same
pre-coin tables, and at every coin value holding what it holds, with the same registered root and hashing, at priced W1
work at most its own plus `32 · networkGap L`.

Rationale: the network certifier caps the bits a call's node exchanges, and an exchanged bit replaces a bounded amount
of priced work. -/
def OutsourceWithinEgressLead (pr : Prices100) (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∃ σ', Real σ' ∧ σ'.pre = σ.pre ∧ ∀ ω, (σ'.exec ω).egress = 0 ∧
    (∀ y, (σ.exec ω).holds y → (σ'.exec ω).holds y) ∧ (σ'.exec ω).root = (σ.exec ω).root ∧
    (σ'.exec ω).hashed = (σ.exec ω).hashed ∧ (σ'.work pr ω : ℚ) ≤ σ.work pr ω + 32 * networkGap L

end Pouw.Assumptions.Lead
