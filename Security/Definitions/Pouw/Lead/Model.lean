import Definitions.Pouw.Lead.Layout
import Definitions.Pouw.Assumptions.Dimension.H100
import Definitions.Pouw.Dimension.Pointwise
import Definitions.Hash.Basic

/-!
# A prover's strategy on one call of the lead, in model W at tile grain (trusted definitions)

A strategy fixes its tables and registers `A`'s root before the coins (`Strategy.pre`, `Strategy.aRoot`) and then, at
each value `ω` of the call's coins, runs an execution (`Exec`), so it may branch on the coins anywhere. The coins are
what the network certifier releases after it reads `A`'s root (`Pouw.Assumptions.Lead.CertifierCoins`, which `G`
takes).

* **An execution** carries
  - `trace`: the instructions it executes, in `MH100w`'s classes (`Op100 Unit`: at one coin value a register is a
    value), each at its W1 price. Its **priced W1 work** is `cost100 pr trace` (`Strategy.work`);
  - `steps`: its account at the grain of an MMA tile (`tileWords` words, 16 rows × 8 columns), a run of the chunk
    model (`Pouw.Dimension.Pointwise`) whose slots hold one tile, from the pre-coin tables (within the node's memory)
    and the certifier's coin stream (`init`). A write step is the execution's writing of one tile's words, at any points
    of the execution and in any order, so the words need never be in memory together; a copy moves a tile whole, free;
  - `shaRegs`: the registers its SHA-512 compressions write, and `hashed`, the messages it hashes;
  - `egress`: the bits it exchanges with machines outside the node, other than the certifier's coin stream;
  - `root`: the words' commitment it registers.
* **The certifier's coin stream** (`CoinStream`): raw coins in a fixed arrangement, each coin once at most.
* **The commitment format** (`Commit`): the messages the honest commitment of a call's words hashes (leaves and
  nodes), its root, and their SHA-512 compressions.
* **Accepted** (`Accepted`): what the verifier sees. The call is one it admits (`Layout.Admits`: an inner dimension of
  at least 1 and at most `2^40` MMA tiles, by the call's public shape), and the execution registers the honest
  commitment's root and exchanges at most the call's bits (`Layout.callBits`), which the network certifier counts and
  refuses past. How it computed the words is not part of acceptance: that it writes each committed tile it doesn't
  start with is the copy model's premise (`Pouw.Assumptions.Lead.TileGrain`).
* **The break** (`Break`): a SHA-512 collision among what the execution hashed and what the honest commitment hashes.
-/

namespace Pouw.Lead

open Pouw.Dimension.H100 Pouw.Dimension.Pointwise

/-- An MMA tile's words, 16 rows × 8 columns: under tile grain a free copy moves one whole. -/
def tileWords : ℕ := 128

/-- Addresses: the pre-coin tables (`inl`) and the certifier's coin stream (`inr`). -/
abbrev Addr : Type := ℕ ⊕ ℕ

/-- **One coin value's execution** of a strategy on a call. -/
structure Exec where
  /-- The instructions it executes, in `MH100w`'s classes, one value a register. -/
  trace : List (Op100 Unit)
  /-- Its account at tile grain: a write step is its writing of one MMA tile's words, at any points of the execution;
  a copy moves a whole tile free. -/
  steps : List (Step Addr (Fin tileWords → ℤ) 1)
  /-- Registers its SHA-512 compressions write. -/
  shaRegs : ℕ
  /-- Messages it hashes with SHA-512. -/
  hashed : List Hash.Msg
  /-- Bits it exchanges with machines outside the node, other than the certifier's coin stream. -/
  egress : ℕ
  /-- The words' commitment it registers. -/
  root : Hash.Digest

/-- A coin's index: a tag bit, a step-1 bit, a tag weight or a coin lane. -/
abbrev CoinIdx (L : Layout) : Type :=
  (Fin L.m × Fin L.S) ⊕ (Fin L.m × Fin 4) ⊕ (Fin 4 × Fin L.n) ⊕ (Fin L.S × Fin L.tiles × Fin 3 × Fin L.n)

/-- A coin's value at `ω`. -/
def coinAt {L : Layout} (ω : Coins L) : CoinIdx L → ℤ
  | .inl (i, t) => ((ω.1 i t : ℕ) : ℤ)
  | .inr (.inl (i, q)) => ((ω.2.1 i q : ℕ) : ℤ)
  | .inr (.inr (.inl (q, j))) => ω.G q j
  | .inr (.inr (.inr (t, k, ℓ, j))) => ω.s t k ℓ j

/-- **The certifier's coin stream**: raw coins in an arrangement fixed before the coins (`ι`), each streamed word zero
or one coin, each coin at one position at most, within the node's memory of tiles. The certifier draws the coins after
it reads `A_j`'s root and the strategy's tables are fixed (`Pouw.Assumptions.Lead.CertifierCoins`): `Strategy` fixes
its tables and `A`'s root before the coins, `G` fixes the data before them, and every bound averages over the
certifier's entropy uniformly, which `CertifierCoins` decodes one to one into `Coins`. -/
structure CoinStream (L : Layout) where
  /-- The coin at each streamed tile's word, or none (zero). -/
  ι : ℕ → Fin tileWords → Option (CoinIdx L)
  /-- The stream fits the node's memory: `L.mem` tiles. -/
  supp : ∀ c p, L.mem ≤ c → ι c p = none
  /-- Each coin is streamed once at most. -/
  once : ∀ c p c' p' x, ι c p = some x → ι c' p' = some x → c = c' ∧ p = p'

/-- The stream's tile `c` at `ω`. -/
def CoinStream.val {L : Layout} (st : CoinStream L) (ω : Coins L) (c : ℕ) : Fin tileWords → ℤ :=
  fun p => (st.ι c p).elim 0 (coinAt ω)

/-- **A prover's strategy on a call**: tables fixed before the coins, within the node's memory, `A`'s root registered
before the coins, and an execution at each coin value. -/
structure Strategy (L : Layout) where
  /-- The pre-coin tables (any, chosen after the data), one tile a slot. -/
  pre : ℕ → Fin tileWords → ℤ
  /-- The tables fit the node's memory: `L.mem` tiles. -/
  pre_supp : ∀ a, L.mem ≤ a → pre a = 0
  /-- The `A` root it registers before the coins, which the certifier reads before it releases them. -/
  aRoot : Hash.Digest
  /-- The execution at each value of the coins. -/
  exec : Coins L → Exec

/-- The memory an execution starts from: the pre-coin tables and the certifier's coin stream at `ω`. -/
def Strategy.init {L : Layout} (σ : Strategy L) (stream : CoinStream L) (ω : Coins L) :
    Addr → Fin 1 → Fin tileWords → ℤ :=
  Sum.elim (fun a _ => σ.pre a) (fun c _ => stream.val ω c)

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

/-- Row blocks of 16 and column blocks of 8: the call's MMA tiles a step. -/
def Layout.rowBlocks (L : Layout) : ℕ := (L.m + 15) / 16

/-- Column blocks of 8. -/
def Layout.colBlocks (L : Layout) : ℕ := (L.n + 7) / 8

/-- **The committed words' MMA tile** at row block `rb`, column block `cb`, step `t`: the word of row `16 rb + r` and
column `8 cb + c` at position `8 r + c`, zero outside the call. -/
def tileOf (L : Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (rb cb : ℕ) (t : Fin L.S) :
    Fin tileWords → ℤ := fun p =>
  if h : 16 * rb + (p : ℕ) / 8 < L.m ∧ 8 * cb + (p : ℕ) % 8 < L.n then
    word L κ ω ⟨16 * rb + (p : ℕ) / 8, h.1⟩ t ⟨8 * cb + (p : ℕ) % 8, h.2⟩
  else 0

/-- The call's committed MMA tiles: row block, column block and step. -/
def tileIdx (L : Layout) : Finset (ℕ × ℕ × Fin L.S) :=
  Finset.range L.rowBlocks ×ˢ Finset.range L.colBlocks ×ˢ Finset.univ

/-- **Repeated tiles** at `ω`: the committed tiles less their distinct values, what copying whole tiles saves. -/
def repeats (L : Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) : ℕ :=
  (tileIdx L).card - ((tileIdx L).image fun x => tileOf L κ ω x.1 x.2.1 x.2.2).card

open Classical in
/-- **Table hits** at `ω`: the committed tiles whose value a strategy's memory starts with, a pre-coin table's tile or
a coin stream's. -/
noncomputable def tableHits (L : Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L)
    (stream : CoinStream L) (ω : Coins L) : ℕ :=
  ((tileIdx L).filter fun x =>
    (∃ a, σ.pre a = tileOf L κ ω x.1 x.2.1 x.2.2) ∨ ∃ c, stream.val ω c = tileOf L κ ω x.1 x.2.1 x.2.2).card

/-- **The calls the verifier admits**: an inner dimension of at least 1 and at most `2^40` MMA tiles of words
(`rowBlocks · colBlocks · S`), read off the call's public shape. `Accepted` refuses every other call. At `K = 0` stock
is zero while tiles still repeat, and past `2^40` tiles the call's pairs of tiles are too many for their chance of
agreeing (`2^−56` a pair) to stay within `coinBranchGap`. -/
def Layout.Admits (L : Layout) : Prop := 1 ≤ L.K ∧ L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 40

/-- **The execution at `ω` is accepted**: the verifier admits the call (`Layout.Admits`), and the execution registers
the honest root and exchanges at most the call's bits (the network certifier refuses a call past them). All three are
what the verifier sees. -/
def Accepted (L : Layout) (H : Hash.Msg → Hash.Digest) (C : Commit L) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (σ : Strategy L) (ω : Coins L) : Prop :=
  L.Admits ∧ (σ.exec ω).root = C.root H (word L κ ω) ∧ ((σ.exec ω).egress : ℚ) ≤ L.callBits

/-- The break a statement ends in: two different messages with one SHA-512 digest, both among what the execution at
`ω` hashed and what the honest commitment of its words hashes (the verifier's check hashes those). -/
def Break (L : Layout) (H : Hash.Msg → Hash.Digest) (C : Commit L) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (σ : Strategy L) (ω : Coins L) : Prop :=
  Hash.CollisionIn H ((σ.exec ω).hashed ++ (C.msgs H (word L κ ω) ++ C.aMsgs))

end Pouw.Lead
