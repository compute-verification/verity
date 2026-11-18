module

/-!
# The challenger and its log

Every coin a protocol uses comes from a **challenger**: an auditor-controlled device that flips coins only after the
commitments they sample are recorded, and logs every coin it releases. A challenger may run on the same device as
another auditor device: onsite, the network attester's machine runs one; for remote, it is an auditor device. The
verifier reads coins only from a challenger's log, and uses no randomness of its own.

* `Challenger.Entry R C`: one entry of a challenger's log, a commitment `r : R` it recorded or a coin `c : C` it
  released;
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

A scheme's times are its own: the log orders entries and holds no time.

A run of a challenger that flips `n` coins of a finite type `C` is weighed by its flips `ω : Fin n → C`, each sequence
of flips equally likely (`Game.prCoin`), as `Game.coin` weighs a live coin. What the challenger is taken to do with its
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

/-- **An entry of a challenger's log**: a commitment it recorded, or a coin it released. -/
inductive Entry (R C : Type) where
  | recorded (r : R)
  | released (c : C)

/-- **A challenger's log**: its entries, oldest first. -/
abbrev Log (R C : Type) : Type := List (Entry R C)

namespace Log

variable {R C : Type}

/-- The coins the log released, in order. -/
def coins : Log R C → List C
  | [] => []
  | .recorded _ :: l => coins l
  | .released c :: l => c :: coins l

/-- **The log before its coin `i`**: every entry before the release of coin `i` (counting from 0), the commitments it
recorded and the coins it released before it; the whole log when it released fewer than `i + 1` coins. -/
def before : Log R C → Nat → Log R C
  | [], _ => []
  | .recorded r :: l, i => .recorded r :: before l i
  | .released _ :: _, 0 => []
  | .released c :: l, i + 1 => .released c :: before l i

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

/-- **A value derived from the log's prefix**: on every run of a challenger that flips `n` coins, `v` is a function of
the log before coin `i`, the commitments recorded and the coins released before it. -/
def DerivedBefore {R C V : Type} {n : Nat} (log : (Fin n → C) → Log R C) (v : (Fin n → C) → V) (i : Nat) : Prop :=
  ∃ f : Log R C → V, ∀ ω, v ω = f ((log ω).before i)

end Challenger

end Verity
