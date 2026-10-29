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
* `TileGrain` (CopyGrain at the tile width, under model W): an execution that hashes the honest words writes each
  committed tile it doesn't start with as one whole-tile write, its words at any points of the execution (they need
  never be in memory together). Acceptance is only what the verifier sees (`Accepted`), so the copy model is carried
  here alone.
* `OneInstructionMinimal`: each word written and each compression register is its own register of the trace.
* `CardWithinAccelW1`: every strategy the node's processors run, hosts counted, is in `Real`.
* `ShaWrites`: a SHA-512 compression writes about 3,500 registers, credited per form.

The protocol premises the code can't be checked against yet:
* `HonestHashRuled`: an accepted execution that exchanges nothing hashes, for each message of the honest commitment, a
  message with its digest.
* `CommitL1w`: the honest commitment is the lead's layout, at its credited compressions: every words chunk and every
  1 KiB segment of A's bytes is a frame-v3 leaf at its own position.
* `OutsourceWithinEgressLead`: an exchanged bit replaces at most `β` of priced W1 work (class L). The bits are
  bounded by the protocol: `Accepted` refuses a call past its share of the cap (`Layout.callBits`).

The device premise:
* `CertifierCoins` (`trusted-host` at the network certifier): the certifier draws the coins from its own entropy after
  it holds the call's `A`. `G` reads it through the model rather than as a hypothesis: the certifier's coins are a
  `CoinStream`, a strategy fixes its tables before the coins (`Strategy`), `G` fixes the data before them, and every
  bound averages over the coins uniformly. `CertifierCoins` is what ties that model to the deployed certifier.
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

/-- `pouw-tile-grain` (proposed; CopyGrain at the tile width, under model W). **An execution that hashes the honest
words writes each committed tile it doesn't start with, as a whole tile**: at a coin value where a realizable strategy
exchanges nothing and hashes every message of the honest commitment of the call's words, each committed MMA tile
(`tileOf`, 16 rows × 8 columns) is the value of one of its run's write steps (`Exec.steps`), or a tile of the pre-coin
tables or the coin stream it starts from (`Strategy.init`).

* No tile need ever be in memory whole. A write step is the execution's writing of the tile's `tileWords` words, at
  any points of the execution and in any order (SHA-512 reads 128-byte blocks, so it can hash the words as it writes
  them), and `OneInstructionMinimal` charges it `tileWords` registers. That the run holds the tile in some slot at some
  point says exactly the same (`CoinBranch.holds_iff`): copies create no value.
* It says more than "each committed word is written at some point". A tile is free only as a copy of one whole tile
  the execution already wrote or started with; a tile with any other word is charged whole (model W at the tile's
  width), so patching an earlier tile costs the tile. With words written one at a time and tiles copied free, a run
  need write only the call's new (position, value) pairs (`Pouw.Dimension.Pointwise.run_pairs_le`, which patching
  attains): a tile that agrees with an earlier one in most positions would be mostly free, and the copy argument
  (`CoinBranch.distinct_le_writes`) would not follow.

Rationale: the messages carry every committed word, so with no link out the node computed them. A whole-tile copy is
the reuse the premise leaves free. Partial reuse is charged whole because on the lead's words it doesn't pay once the
choice of the words to keep is priced: that backs the charge in effect, not to the letter, since at W1 one IADD
patching one word writes one register. This is the copy model, one line to change. -/
def TileGrain (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (stream : CoinStream L) (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, (σ.exec ω).egress = 0 → (∀ y ∈ C.msgs H (word L κ ω), y ∈ (σ.exec ω).hashed) →
    ∀ rb < L.rowBlocks, ∀ cb < L.colBlocks, ∀ t : Fin L.S,
      (∃ a p, Step.write a p (tileOf L κ ω rb cb t) ∈ (σ.exec ω).steps) ∨
        ∃ a, σ.init stream ω a 0 = tileOf L κ ω rb cb t

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

/-- `pouw-sha-writes` (proposed; a standing ruling admits it, a premise milder than SAK). **A SHA-512
compression writes about 3,500 registers**, credited per form: at every coin value, a realizable strategy's
compressions write at least `3,515.38` registers each for the words and `3,449.05` for A (`shaCredit`), over the
messages it hashes.

Rationale: the measured compression forms on the card; the ruling accepts the count as a premise. -/
def ShaWrites (isA : Hash.Msg → Bool) (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, ((σ.exec ω).hashed.map (shaCredit isA)).sum ≤ (σ.exec ω).shaRegs

/-- `pouw-honest-hash-ruled` (proposed; a ruling: the honest hashing is the minimum). **An accepted execution that
exchanges nothing hashes the honest commitment's messages**: for each message the honest commitment of the call's
words hashes, a realizable strategy's accepted execution at a coin value where it exchanges nothing hashes a message
with its digest.

Rationale: the premise counts registers; with no link out, a digest is reached only by hashing on the node. -/
def HonestHashRuled (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∀ ω, (σ.exec ω).egress = 0 → Accepted L H C κ σ ω →
    ∀ y ∈ C.msgs H (word L κ ω) ++ C.aMsgs, ∃ x ∈ (σ.exec ω).hashed, H x = H y

/-- `pouw-commit-l1w` (proposed; the commitments' layout `l1w`, until the Lean code is the served format).
**The honest commitment is the lead's layout**: each 256 KiB words chunk and each 1 KiB segment of A's bytes is a
frame-v3 leaf at its own position, whose message holds the chunk's or the segment's bytes. So its words chunks number
`wordChunks L`, and at every coin value its distinct messages' ShaWrites credit is at least `3,515.38` a words
compression of `wordsBlocks` and `3,449.05` an A compression of `aBlocks`, whatever A is: equal chunks or equal
segments sit at different positions, so they are different messages.

The premise is stated at this layout and fails for a format that hashes a row's bytes alone. In an `hm96-sha512` row
tree (`Hash.RowLeaf`), only the 64-byte leaf is framed at its position, so two equal rows of A share their row
compressions and `dedup` drops the repeat below `aBlocks`.

Rationale: the lead's committer frames each words chunk and each 1 KiB segment of A as its own positioned leaf
(`l1wCommit`). -/
def CommitL1w (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L) (isA : Hash.Msg → Bool)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) : Prop :=
  C.chunks = wordChunks L ∧ ∀ ω : Coins L,
    shaWordsRegs * wordsBlocks L C + shaARegs * aBlocks L ≤
      (((C.msgs H (word L κ ω) ++ C.aMsgs).dedup).map (shaCredit isA)).sum

/-- `pouw-outsource-egress` (proposed; class L, the lead's instance at `β`). **An exchanged bit replaces at most `β`
of priced W1 work**: every realizable strategy has a realizable twin that exchanges nothing, with the same pre-coin
tables, and at every coin value registering the same root, hashing beyond what it hashes only the honest commitment's
messages, at priced W1 work at most its own plus `β` a bit it exchanges.

Rationale: what an outside machine computes for the node, the node can compute itself, the honest commitment's
hashing included; a coin bit carries at most `n_max` MACs of the coins' work (every step carries the three coin
lanes), each worth at most the call's honest W1 work a MAC (`betaLead`). The bits are the protocol's: `Accepted`
bounds them by the call's share of the cap. -/
def OutsourceWithinEgressLead (pr : Prices100) (β : ℚ) (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (Real : Strategy L → Prop) : Prop :=
  ∀ σ, Real σ → ∃ σ', Real σ' ∧ σ'.pre = σ.pre ∧ ∀ ω, (σ'.exec ω).egress = 0 ∧ (σ'.exec ω).root = (σ.exec ω).root ∧
    (∀ x ∈ (σ'.exec ω).hashed, x ∈ (σ.exec ω).hashed ∨ x ∈ C.msgs H (word L κ ω) ++ C.aMsgs) ∧
    (σ'.work pr ω : ℚ) ≤ σ.work pr ω + β * (σ.exec ω).egress

/-- The network certifier's code for the lead's coins: it reads `A`'s root and its own entropy, and returns the root
with `coinsOf` of the entropy, a fixed decoding of the entropy alone. -/
def certCode {Ent : Type} (coinsOf : Ent → Coins L) : Hash.Digest × Ent → Hash.Digest × Coins L :=
  fun x => (x.1, coinsOf x.2)

/-- `pouw-certifier-coins` (proposed; `trusted-host` at the network certifier). **The certifier draws the coins after
it holds `A`**: the auditor-trusted certifier `cert` runs `certCode coinsOf`, reading `A`'s root and the device's own
entropy and releasing `coinsOf` of the entropy, so the coins are uniform when the entropy is and independent of `A`
and of a strategy's pre-coin tables. Not a property of `NetworkCertifier`: that program draws and releases no coins.

Rationale: `trusted-host`; the certifier is the auditor's hardware on a machine the prover doesn't control. -/
def CertifierCoins {Device Ent : Type} (cert : Device) (coinsOf : Ent → Coins L)
    (runC : Device → Hash.Digest × Ent → Hash.Digest × Coins L) : Prop :=
  Verity.Assumptions.TrustedHost cert runC (certCode coinsOf)

end Pouw.Assumptions.Lead
