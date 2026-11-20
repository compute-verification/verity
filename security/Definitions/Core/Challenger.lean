module

public import Definitions.Core.Hash

/-!
# The challenger and its log

Every coin a protocol uses comes from a **challenger**: an auditor-controlled device that flips coins only after the
commitments they sample are recorded, and logs every coin it releases. A challenger may run on the same device as
another auditor device: onsite, the network attester's machine runs one; for remote, it is an auditor device. The
verifier reads coins only from a challenger's log, and uses no randomness of its own.

* `Challenger.Attested k`: an attester's entry, one the key `k` of an auditor's device covers: the digests of what it
  was shown, and its data, a digest and an optional reading of its clock (`Challenger.Reading k`);
  `Reading.elapsed a b`, the difference of two readings of that one device, and `Reading.tick`, a reading counted in a
  registered period;
* `Challenger.Between reads digest e a b`: an entry `e` after attester's entry `a`, which it reads, and before `b`,
  which holds its digest; `Challenger.Bracketed`, the same with both readings, the form every deadline reads;
* `Challenger.Entry R C`: one entry of a challenger's log, a commitment `r : R` it recorded, a coin `c : C` it
  released, or an attester's entry;
* `Challenger.Log R C`: its log, oldest entry first;
* `Log.coins`: the coins the log released, in order;
* `Log.before l i`: the log before its coin `i`, everything it recorded and released before releasing coin `i` (the
  whole log, when it released fewer than `i + 1` coins);
* `Log.RecordedBefore l r i`, `Log.ReleasedBefore l c i`: the log records commitment `r`, or releases coin `c`, before
  its coin `i`. Coins are counted in the one order the log releases them, whatever their families, so the order between
  families is the log's;
* `Log.Cut l fixed coins`: the cut, every commitment `fixed` picks out recorded before every coin `coins` picks out,
  whichever scheme commits or draws it;
* `Log.KeyedBySlot l member fam root`: member `s` of family `fam` is released after slot `s`'s root is recorded, once;
* `Log.RecordsInOrder l root`, `Log.JoinsAt l root j`: the roots `root t` recorded in the order of `t`, and the coin
  `j t` each joins at, which then never decreases in `t` (`Proofs.Core.Challenger`);
* `Family`: a family of coins, a label and a count, its coins `(label, j)` for `j < count` (`Family.Holds`);
* `DerivedBefore log v i`: a value derived from the log's prefix, a function of the log before coin `i` on every run.

An entry is covered by its device's key when it is authenticated by a signature of that device over it, or over a chain
that holds it. What a verifier checks of a chain shows the cover: each entry holds its sequence number and the chain
value before it, a whole SHA-512 digest; the closing signature covers the head, the count and the session; and the chain
recomputes from its first entry to the signed head with no gap, no repeat and the signed count. A session with no
closing signature has no attested entries, and a consumer's check refuses it by name, never with a partial verdict. That
the device writes an entry into its chain before it releases what the entry is about (a coin) is assumed of the device
(`trusted-device`), since no signature at a close shows it.

Order is the model's: an attester's entry is after the entries whose digests it holds, and before the close that covers
it; an entry that reads another is after it. A reading is taken when its entry is made, not at the close. Time is a
protocol on that order, not a part of it: a reading is data on an attester's entry, a statement reads only the
difference of two readings of one device (`Reading.elapsed`), never a time of day and never two devices' readings
against each other, and a prover's entry has no time but a bracket (`Between`). How a device's clock runs against real
time is an assumption about the device, `Assumptions.ClockWithin`.

A run of a challenger that flips `n` coins, coin `i` of a finite type `C i`, is weighed by its flips
`ω : (i : Fin n) → C i`, each sequence of flips equally likely (`Game.prCoin`), as `Game.coin` weighs a live coin. Its log
shows the coins in one type of its own, each with its family (the code's `(label, index)`); the log's definitions read
only that type, so coins of several types need nothing of them. What the challenger is taken to do with its
flips and its log, and what the verifier reads, are the named premises of `Definitions.Core.Assumptions.Challenger`.

**Derived by hashing.** Fiat–Shamir is out, with two named exceptions, each at a rate where a round trip to the
challenger is impractical; there, randomness is derived from a challenger's coin by SHA-512:

* PoUW's per-matmul noise seeds, each hashed from the verification phase's salt (one challenger coin per phase), the
  call's position, and the committed roots of its operands;
* PoUB's pass seeds within a phase, each hashed from the phase's challenger coin and the previous
  pass's root.

Neither is a coin of this log: a statement about one reads it through the coin it is derived from. Each is a value
derived from the log's prefix (`DerivedBefore`): the salt coin and the operands' recorded roots, or the phase's coin
and the previous pass's recorded root, are in the log before the next coin it releases.
-/

@[expose] public section

namespace Verity

namespace Challenger

/-- **A reading of the auditor's device registered under key `k`** (`Site.Loc.key`): the count of its raw monotonic
clock in nanoseconds as it makes an entry. The key is in the type, so readings of two devices are never one type and
are never compared. -/
structure Reading (k : String) where
  ns : Nat
  deriving DecidableEq

namespace Reading

variable {k : String}

/-- **Elapsed**: reading `b` less reading `a`, in nanoseconds of the one device's clock. -/
def elapsed (a b : Reading k) : Int := (b.ns : Int) - a.ns

/-- **A tick**: reading `t` counted in a registered period of `period` nanoseconds. -/
def tick (period : Nat) (t : Reading k) : Nat := t.ns / period

end Reading

/-- **An attester's entry**: an entry the key of the auditor's device registered under `k` covers, signed by the
device itself or held by a chain whose close it signs. It holds the digests of what it was shown (`shown`), which are
therefore before it, and carries data: a digest of what it says (`data`) and, where the device reads its clock as it
makes the entry, that reading (`reading`). The network attester's signed records (`NetTiming.RecordFormat`), the memory
challenger's signatures (`Properties.Challenger.Signed`, the memory attester's `Signed`), the sampled audit's
challenger's signed messages (`Vocab.signer`) and the entry beside a coin's release that holds the coin's digest with
its reading are each such an entry; a clock reading is one field of it. A coin is
released only as a `.released` entry, so `Log.coins` reads every coin. -/
structure Attested (k : String) where
  shown : List Hash.Msg
  data : Hash.Msg
  reading : Option (Reading k)

/-- **Between two of one attester's entries**, by the model's two rules alone: the entry `e` reads `a` (`reads`), so it
is after `a`; and `b` holds `e`'s digest, so it is before `b`. What `e` reads is the protocol's (`reads`), and its
digest is the protocol's encoding of it (`digest`); nothing else orders them. -/
def Between {E : Type} {k : String} (reads : E → Attested k → Prop) (digest : E → Hash.Msg) (e : E)
    (a b : Attested k) : Prop :=
  reads e a ∧ digest e ∈ b.shown

/-- **A bracketed entry**: attester `k`'s entry `a` holds a released coin's digest with its reading `ra`, the prover's
entry `e` reads it, and `k`'s entry `b` holds `e`'s digest with its reading `rb`. It is the four steps (a reading, the
coin released, the commitment recorded, a reading) with the coin's reading on the entry beside its release. What it
bounds is the device's real time between its own two entries `a` and `b`; it says nothing of when `e` was made, which
needs the model's order tied to real time, a premise no module declares. A deadline and a salt's age read it and bound
`ra.elapsed rb`. -/
def Bracketed {E : Type} {k : String} (reads : E → Attested k → Prop) (digest : E → Hash.Msg) (e : E)
    (a b : Attested k) (ra rb : Reading k) : Prop :=
  Between reads digest e a b ∧ a.reading = some ra ∧ b.reading = some rb

/-- **An entry of a challenger's log**: a commitment it recorded, a coin it released, or an attester's entry `s`, one
the key of the auditor's device registered under `k` covers. -/
inductive Entry (R C : Type) where
  | recorded (r : R)
  | released (c : C)
  | attested (k : String) (s : Attested k)

/-- **A challenger's log**: its entries, oldest first. -/
abbrev Log (R C : Type) : Type := List (Entry R C)

namespace Log

variable {R C : Type}

/-- The coins the log released, in order. -/
def coins : Log R C → List C
  | [] => []
  | .recorded _ :: l => coins l
  | .released c :: l => c :: coins l
  | .attested _ _ :: l => coins l

/-- **The log before its coin `i`**: every entry before the release of coin `i` (counting from 0), the commitments it
recorded and the coins it released before it; the whole log when it released fewer than `i + 1` coins. -/
def before : Log R C → Nat → Log R C
  | [], _ => []
  | .recorded r :: l, i => .recorded r :: before l i
  | .released _ :: _, 0 => []
  | .released c :: l, i + 1 => .released c :: before l i
  | .attested k s :: l, i => .attested k s :: before l i

/-- **Recorded before coin `i`**: the log records the commitment `r` before it releases its coin `i`. -/
def RecordedBefore (l : Log R C) (r : R) (i : Nat) : Prop :=
  Entry.recorded r ∈ l.before i

/-- **Released before coin `i`**: the log releases coin `c`, of whichever family, before its coin `i`. -/
def ReleasedBefore (l : Log R C) (c : C) (i : Nat) : Prop :=
  Entry.released c ∈ l.before i

/-- **The cut**: every commitment `fixed` picks out is recorded before every coin `coins` picks out, whichever scheme
commits or draws it. Every credited payload committed before every credited key is drawn is the cut at the credited
payloads and the credited keys' coins. -/
def Cut (l : Log R C) (fixed : R → Prop) (coins : Nat → Prop) : Prop :=
  ∀ i, coins i → ∀ r, fixed r → l.RecordedBefore r i

/-- **Coins keyed by slot**: each coin `c` the log releases is member `(member c).2` of family `(member c).1`; every
member `s` of family `fam` is released after the log records slot `s`'s root `root s`, and the log releases member `s`
once. -/
def KeyedBySlot {Λ : Type} (l : Log R C) (member : C → Λ × Nat) (fam : Λ) (root : Nat → R) : Prop :=
  (∀ (i : Nat) c, l.coins[i]? = some c → (member c).1 = fam → l.RecordedBefore (root (member c).2) i) ∧
    ∀ (i j : Nat) c c', l.coins[i]? = some c → l.coins[j]? = some c' → (member c).1 = fam → member c = member c' → i = j

/-- **Recorded in order**: the log records `root t` before every coin it records `root t'` before, for `t ≤ t'`. -/
def RecordsInOrder (l : Log R C) (root : Nat → R) : Prop :=
  ∀ t t', t ≤ t' → ∀ i, l.RecordedBefore (root t') i → l.RecordedBefore (root t) i

/-- **The coin each root joins at**: `j t` is the first coin the log releases after recording `root t`, the number of
coins released before it is recorded. -/
def JoinsAt (l : Log R C) (root : Nat → R) (j : Nat → Nat) : Prop :=
  ∀ t i, l.RecordedBefore (root t) i ↔ j t ≤ i

end Log

/-- **A family of coins**: a label and a count; its coins are `(label, j)` for `j < count`. A challenger's coins are
in families, named by no protocol. -/
structure Family (Λ : Type) where
  label : Λ
  count : Nat

/-- The families `fs` hold the coin `(label, j)`: one of them has that label and more than `j` coins. -/
def Family.Holds {Λ : Type} (fs : List (Family Λ)) (c : Λ × Nat) : Prop :=
  ∃ f ∈ fs, f.label = c.1 ∧ c.2 < f.count

/-- **A value derived from the log's prefix**: on every run of a challenger that flips `n` coins, coin `i` of type
`C i`, `v` is a function of the log before coin `i`, the commitments recorded and the coins released before it. -/
def DerivedBefore {R T V : Type} {n : Nat} {C : Fin n → Type} (log : ((i : Fin n) → C i) → Log R T)
    (v : ((i : Fin n) → C i) → V) (i : Nat) : Prop :=
  ∃ f : Log R T → V, ∀ ω, v ω = f ((log ω).before i)

end Challenger

end Verity
