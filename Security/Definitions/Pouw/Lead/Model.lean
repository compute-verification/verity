import Definitions.Pouw.Lead.Layout
import Definitions.Pouw.Assumptions.Dimension.H100
import Definitions.Pouw.Dimension.Pointwise
import Definitions.Hash.Basic

/-!
# A prover's strategy on one call of the lead, in model W at tile grain (trusted definitions)

A strategy fixes its tables before the coins (`Strategy.pre`) and then, at each value `ω` of the call's coins, runs
an execution (`Exec`), so it may branch on the coins anywhere.

* **An execution** carries
  - `trace`: the instructions it executes, in `MH100w`'s classes (`Op100 Unit`: at one coin value a register is a
    value), each at its W1 price. Its **priced W1 work** is `cost100 pr trace` (`Strategy.work`);
  - `steps`: its effect on memory at the copy grain of an MMA tile (`tileWords` words, 16 rows × 8 columns), a run of
    the chunk model (`Pouw.Dimension.Pointwise`) from the pre-coin tables and the certifier's coin stream (`init`);
  - `shaRegs`: the registers its SHA-512 compressions write, and `hashed`, the messages it hashes;
  - `egress`: the bits it exchanges with machines outside the node, other than the certifier's coin stream;
  - `root`: the words' commitment it registers, and `holds`, what it holds at each tile position at some point.
* **The commitment format** (`Commit`): the messages the honest commitment of a call's words hashes (leaves and
  nodes), its root, and their SHA-512 compressions.
* **Accepted** (`Accepted`): the execution holds every committed word at its tile position at some point, and
  registers the honest commitment's root.
* **The break** (`Break`): a SHA-512 collision among what the execution hashed and what the honest commitment hashes.
-/

namespace Pouw.Lead

open Pouw.Dimension.H100 Pouw.Dimension.Pointwise

/-- An MMA tile's words, 16 rows × 8 columns: under tile grain a free copy moves one whole. -/
def tileWords : ℕ := 128

/-- A word's position in its MMA tile: row `i mod 16`, column `j mod 8`. -/
def tilePos (i j : ℕ) : Fin tileWords := ⟨(i % 16) * 8 + j % 8, by unfold tileWords; omega⟩

/-- Addresses: the pre-coin tables (`inl`) and the certifier's coin stream (`inr`). -/
abbrev Addr : Type := ℕ ⊕ ℕ

/-- **One coin value's execution** of a strategy on a call. -/
structure Exec where
  /-- The instructions it executes, in `MH100w`'s classes, one value a register. -/
  trace : List (Op100 Unit)
  /-- Its effect on memory at tile grain. -/
  steps : List (Step Addr ℤ tileWords)
  /-- Registers its SHA-512 compressions write. -/
  shaRegs : ℕ
  /-- Messages it hashes with SHA-512. -/
  hashed : List Hash.Msg
  /-- Bits it exchanges with machines outside the node, other than the certifier's coin stream. -/
  egress : ℕ
  /-- The words' commitment it registers. -/
  root : Hash.Digest
  /-- The values it holds at each tile position at some point, wherever they came from. -/
  holds : Fin tileWords → ℤ → Prop

/-- **A prover's strategy on a call**: tables fixed before the coins, and an execution at each coin value. -/
structure Strategy (L : Layout) where
  /-- The pre-coin tables (any, chosen after the data). -/
  pre : ℕ → Fin tileWords → ℤ
  /-- The execution at each value of the coins. -/
  exec : Coins L → Exec

/-- The memory an execution starts from: the pre-coin tables and the coin stream `stream ω` (the certifier's encoding
of the coins in tiles). -/
def Strategy.init {L : Layout} (σ : Strategy L) (stream : Coins L → ℕ → Fin tileWords → ℤ) (ω : Coins L) :
    Addr → Fin tileWords → ℤ :=
  Sum.elim σ.pre (stream ω)

/-- **The priced W1 work** of a strategy at a coin value: its trace at the W1 table `pr`. -/
def Strategy.work {L : Layout} (pr : Prices100) (σ : Strategy L) (ω : Coins L) : ℕ :=
  cost100 pr (σ.exec ω).trace

/-- The registers an instruction writes: a tensor-core instruction its accumulators, an FP32, INT32, FP64 or other
writer one, a move or an operand none. -/
def Op100.regs {G : Type} : Op100 G → ℕ
  | .wgmma _ outs => outs.length
  | .mma outs _ => outs.length
  | .fp32 .. => 1
  | .int _ => 1
  | .fp64 _ => 1
  | .other _ => 1
  | .move _ => 0
  | .operand _ => 0

/-- The registers a trace writes. -/
def regCount {G : Type} (P : List (Op100 G)) : ℕ := (P.map Op100.regs).sum

/-- SHA-512 compressions of a message: its padded length (the message, the `0x80` byte and the 16-byte length) in
128-byte blocks. -/
def blocks (x : Hash.Msg) : ℕ := (x.length + 17 + 127) / 128

/-- **The commitment format** of a call's words: the messages the honest commitment hashes (every leaf and node, in
order) and its root, at the hash `H`. -/
structure Commit (L : Layout) where
  /-- Every message the honest commitment of the words hashes. -/
  msgs : (Hash.Msg → Hash.Digest) → (Fin L.m → Fin L.S → Fin L.n → ℤ) → List Hash.Msg
  /-- The honest root. -/
  root : (Hash.Msg → Hash.Digest) → (Fin L.m → Fin L.S → Fin L.n → ℤ) → Hash.Digest
  /-- The messages A's commitment hashes, fixed before the coins. -/
  aMsgs : List Hash.Msg
  /-- The words chunks: leaves holding at most `chunkWords` words each. -/
  chunks : ℕ

/-- **The execution at `ω` is accepted**: it holds every committed word at its tile position at some point, and
registers the honest root. -/
def Accepted (L : Layout) (H : Hash.Msg → Hash.Digest) (C : Commit L) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (σ : Strategy L) (ω : Coins L) : Prop :=
  (∀ (i : Fin L.m) (t : Fin L.S) (j : Fin L.n), (σ.exec ω).holds (tilePos i j) (word L κ ω i t j)) ∧
    (σ.exec ω).root = C.root H (word L κ ω)

/-- The break a statement ends in: two different messages with one SHA-512 digest, both among what the execution at
`ω` hashed and what the honest commitment of its words hashes (the verifier's check hashes those). -/
def Break (L : Layout) (H : Hash.Msg → Hash.Digest) (C : Commit L) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (σ : Strategy L) (ω : Coins L) : Prop :=
  Hash.CollisionIn H ((σ.exec ω).hashed ++ (C.msgs H (word L κ ω) ++ C.aMsgs))

end Pouw.Lead
