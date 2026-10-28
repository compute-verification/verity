import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditJoint

/-!
# The rate audit with partial completion (trusted)

`RateAudit.lean`'s game, except that rebuilding is no longer all-or-nothing. There, `rebuild j` returns the whole word
`E_j` at `c_enc`, so a device that keeps part of a word and completes the rest (keep a word's top bits, complete the
bottom `x` from the known payload by a lattice) falls outside the game. Here the chip may instead `complete j x g`:
obtain `x ≤ b` bits of information about word `j`, any `x`-bit function `g` of it (its bottom `x` bits, or anything
else), at a compute cost `cost x` that depends only on `x`. The device keeps whatever else of the word it needs in its
on-chip state, device memory or side sources, which the audit counts as before. A whole rebuild is `complete j b _ id`,
at `cost b`, so every device of `RateAudit.lean`'s game is one of this game.

The cost is a parameter of the game, as `c_enc` is in `RateAudit.lean`. What it must be for a deployed codec is the
named assumption `Accounting.Assumptions.WorkHardness` (`Definitions/Accounting/Assumptions/RateAudit.lean`): completing `x`
of a word's `b` bits costs at least `x/b` of a word's price, `rounds·c_root` for a codec of `rounds` rounds. The game is
generous to the device: a completion returns its bits whether or not the device holds the rest of the word, and its cost does not depend on what it holds, so a device that
completes from a hint is a device of this game whose hint it reads as before. A completion takes one word, so the game
has no amortization across words (the claim `rate-codec-work`).

Everything else (the random oracle at a fresh seed, the certified stream, on-chip state laid out before the seed,
device memory read only through `ingress`, side sources answering any function of the store and the chip's address, the
side link's `L` bits with the stream's `ρ` among them, the `q` queries, the sample drawn after the output) is
`RateAudit.lean`'s, and a run's world is `Rate.World`, whose `store` answers completions.

`RealizesRatePartial` is `RealizesRate` (`RateAuditJoint.lean`) for a joint run whose bandwidth audit a partial device
plays, without `RealizesRate`'s compute field: that the game's compute and the compute account's work share no work is
the named assumption `Accounting.Assumptions.RateComputeAdditive` (the actual-circuit model's W), which the guarantees take
as a hypothesis.
-/

namespace Accounting.Rate

open Pous (Bits)

/-- A chip program in the window with partial completion: `Chip`'s steps, with `complete j x hx g k` in place of
`rebuild`, which obtains the `x ≤ b` bits `g (E_j)` of word `j` and continues with `k`. -/
inductive PChip (N b h ρ : ℕ) (α : Type) : Type where
  | done : α → PChip N b h ρ α
  | tag : Bits ρ × Fin N × Bits b → (Bits h → PChip N b h ρ α) → PChip N b h ρ α
  | ingress : ℕ → (Bool → PChip N b h ρ α) → PChip N b h ρ α
  | side : ℕ → (Bool → PChip N b h ρ α) → PChip N b h ρ α
  | stream : Fin ρ → (Bool → PChip N b h ρ α) → PChip N b h ρ α
  | complete : Fin N → (x : ℕ) → x ≤ b → (Bits b → Bits x) → (Bits x → PChip N b h ρ α) → PChip N b h ρ α

namespace PChip

variable {N b h ρ : ℕ} {α : Type}

/-- The program's value in a world. -/
def run (w : World N b h ρ) : PChip N b h ρ α → α
  | done a => a
  | tag x k => (k (w.oracle x)).run w
  | ingress a k => (k (w.mem a)).run w
  | side a k => (k (w.side a)).run w
  | stream i k => (k (w.stream i)).run w
  | complete j _ _ g k => (k (g (w.store j))).run w

/-- The bits read from device memory in a world: one per `ingress` step. -/
def ingressBits (w : World N b h ρ) : PChip N b h ρ α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).ingressBits w
  | ingress a k => (k (w.mem a)).ingressBits w + 1
  | side a k => (k (w.side a)).ingressBits w
  | stream i k => (k (w.stream i)).ingressBits w
  | complete j _ _ g k => (k (g (w.store j))).ingressBits w

/-- The device-memory addresses read in a world. -/
def addrs (w : World N b h ρ) : PChip N b h ρ α → Finset ℕ
  | done _ => ∅
  | tag x k => (k (w.oracle x)).addrs w
  | ingress a k => insert a ((k (w.mem a)).addrs w)
  | side a k => (k (w.side a)).addrs w
  | stream i k => (k (w.stream i)).addrs w
  | complete j _ _ g k => (k (g (w.store j))).addrs w

/-- The bits that came in over the side link in a world: one per `side` step and one per `stream` step. -/
def linkBits (w : World N b h ρ) : PChip N b h ρ α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).linkBits w
  | ingress a k => (k (w.mem a)).linkBits w
  | side a k => (k (w.side a)).linkBits w + 1
  | stream i k => (k (w.stream i)).linkBits w + 1
  | complete j _ _ g k => (k (g (w.store j))).linkBits w

/-- The bits completed in a world: `x` per `complete` step of `x` bits. -/
def completed (w : World N b h ρ) : PChip N b h ρ α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).completed w
  | ingress a k => (k (w.mem a)).completed w
  | side a k => (k (w.side a)).completed w
  | stream i k => (k (w.stream i)).completed w
  | complete j x _ g k => (k (g (w.store j))).completed w + x

/-- The compute the completions cost in a world, at `cost x` per `complete` step of `x` bits. -/
def work (cost : ℕ → ℕ) (w : World N b h ρ) : PChip N b h ρ α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).work cost w
  | ingress a k => (k (w.mem a)).work cost w
  | side a k => (k (w.side a)).work cost w
  | stream i k => (k (w.stream i)).work cost w
  | complete j x _ g k => (k (g (w.store j))).work cost w + cost x

/-- The oracle queries in a world, in order. -/
def queries (w : World N b h ρ) : PChip N b h ρ α → List (Bits ρ × Fin N × Bits b)
  | done _ => []
  | tag x k => x :: (k (w.oracle x)).queries w
  | ingress a k => (k (w.mem a)).queries w
  | side a k => (k (w.side a)).queries w
  | stream i k => (k (w.stream i)).queries w
  | complete j _ _ g k => (k (g (w.store j))).queries w

end PChip

/-- A device with partial completion: `Prover`'s on-chip state, device memory and free side sources, laid out from the
store before the seed and the stream, and a chip program with completions from that state. In every world the program
takes in at most `L` bits over the side link (free side sources and the certified stream together) and makes at most
`q` oracle queries. -/
structure PartialDevice (N b h ρ S L q : ℕ) where
  state : Store N b → Bits S
  mem : Store N b → ℕ → Bool
  side : Store N b → ℕ → Bool
  chip : Bits S → PChip N b h ρ (Fin N → Bits h)
  bounded : ∀ (st : Bits S) (w : World N b h ρ), (chip st).linkBits w ≤ L ∧ ((chip st).queries w).length ≤ q

namespace PartialDevice

variable {N b h ρ S L q : ℕ}

/-- The world a device's chip runs in, for store `E`, certified stream `R` and oracle `O`. -/
def world (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : World N b h ρ :=
  ⟨O, A.mem E, A.side E, R, E⟩

/-- The tags the device commits to. -/
def tags (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : Fin N → Bits h :=
  (A.chip (A.state E)).run (A.world E R O)

/-- The device's ingress in the window: every bit it read from device memory, in bits. -/
def ingress (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).ingressBits (A.world E R O)

/-- The device's footprint in the window: the distinct device-memory addresses it read, in bits. -/
def footprint (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  ((A.chip (A.state E)).addrs (A.world E R O)).card

/-- The bits the device completed in the window. -/
def completed (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).completed (A.world E R O)

/-- The compute the device's completions cost in the window, at `cost x` per completion of `x` bits. -/
def work (A : PartialDevice N b h ρ S L q) (cost : ℕ → ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).work cost (A.world E R O)

/-- The device's compute in the window, at `c_tag` per oracle query and `cost x` per completion of `x` bits. -/
def compute (A : PartialDevice N b h ρ S L q) (cTag : ℕ) (cost : ℕ → ℕ) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : ℕ :=
  ((A.chip (A.state E)).queries (A.world E R O)).length * cTag + A.work cost E R O

/-- The verifier accepts: every sampled word's committed tag is the oracle's tag of that word under the stream. -/
def Accepts {s : ℕ} (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ)
    (I : Fin s → Fin N) : Prop :=
  ∀ i, A.tags E R O (I i) = O (R, I i, E (I i))

end PartialDevice

end Accounting.Rate

namespace Accounting

/-- Run `r`'s bandwidth audit is the rate audit's game with partial completion, played at each value of the other
audits' coins by a partial device within the device model's limits over the bandwidth window, whose ingress the
bandwidth account's reads from device memory cover. Unlike `RealizesRate`, it says nothing of the game's compute: that
it shares no work with the compute account is `Accounting.Assumptions.RateComputeAdditive`. -/
structure RealizesRatePartial {N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι} {α ΩW ΩS : Type} [Fintype α]
    (r : Run d α ΩW (RateCoins N b h ρ s) ΩS) where
  /-- the device of the game at the other audits' coins -/
  dev : ΩW × ΩS → Rate.PartialDevice N b h ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
    (d.rate .oracle * (r.win .bandwidth).len)
  /-- the bandwidth audit's verdict is the game's -/
  accepts_iff : ∀ x y s', r.accepts .bandwidth (x, y, s') ↔ (dev (x, s')).Accepts y.1 y.2.1.1 y.2.1.2 y.2.2
  /-- the bandwidth account's reads from device memory cover the game's ingress -/
  ingress_le : ∀ x y s', (dev (x, s')).ingress y.1 y.2.1.1 y.2.1.2 ≤ r.read (x, y, s')

end Accounting
