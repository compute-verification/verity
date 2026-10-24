import Definitions.NetworkCertifier.AdviceV1
import Definitions.NetworkCertifier.AdviceV1Counted
import Definitions.NetworkCertifier.AdviceString
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# Advice v1's charge counts: Kraft, and the leak bound with no assumption on the charge

The leak bounds of `Guarantees.AdviceString` take any charge `bits` that meets `BitsCount`: at most `2^c` advice values
are charged `c` bits or fewer in all. Advice v1's charge (`AdviceV1.window`, `ChargeCount_v1{N}`) meets it, under any
registered book: `EncardChargedLe`. So the leak bound on the constant-rate Program holds with advice v1's charge and no
assumption on it (`EncardAccSeqsWithinLeConstantRateV1`). The same holds for windows that send counts, charged the
decisions' charge plus any count code's length (`EncardChargedLeCounted`, `EncardAccSeqsWithinLeConstantRateV1Counted`).

**Why, without the codewords.** An arithmetic coder's charge counts like a prefix-free code's length. Weigh a state by
its interval's width over `2^bits`. A scaling doubles the width and counts a bit, so it keeps the weight. A symbol's
outcomes take disjoint sub-intervals, so their weights sum to at most the state's. The finish of a used window starts
from an interval wider than `2^30`, so `2^(−charge) · 2^32` is at most its weight. Summed over every decision stream, the
weights `2^(−charge)` come to at most 1, and at most `2^c` streams are charged `c` or less.

**Scope.** `EncardChargedLe` and its bound: decision-only windows, a window sending its four counts as 0
(`ChargeCount_v1`'s scope). `EncardChargedLeCounted` and its bound: windows that may send nonzero counts, each charged
its decisions' charge plus a count code's length for its counts, under any count code (`CountCode`: lengths meeting
Kraft's inequality, as every prefix-free code's do), the same code in every window and carrying nothing across
windows. Its proof multiplies the two sums: over a window's decision streams and counts, the weights `2^(−charge)` are
the decisions' times the counts', each summing to at most 1. Which slots ask (`en`) is not charged because it is not
advice: the compile slice, its charge included, is proved on every advised window and never sampled, so `en` is pinned
by that slice's scan. The carry across windows is chained from the book's priors (#3801, `charge_chain.check_chain`).

**What is left, the one hypothesis.** That `ChargeCount_v1`'s circuit computes `AdviceV1.window`. It is not proved in
Lean: circuit-check holds the circuit to the Python reference (`arith_encode.py`'s `reference_charge`), and the Lean
functions are that reference one for one, held to it by the charge difftest (the certifier program's
`NetworkCertifier.AdviceV1`, proved equal to these by `CodeChargesV1`, against `reference_charge` on 4,000 random
instances, accepted and refused). For windows with counts, the count code is a parameter as well. It is a separate
code, which is not how advice v1 sends counts: `code_v1` codes them inside the arithmetic coder (`ChargeCount_v2`'s
window).
-/

namespace NetTiming.Guarantees

open AdviceV1

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **Advice v1's charge counts (Kraft).** Under any registered book `bk`, with windows of `N` slots, at most `2^c`
decision streams of `W` windows are charged `c` bits or fewer in all, with no window refused. The windows are
decision-only (their four counts sent as 0), and the charge is `ChargeCount_v1`'s as its Python reference defines it:
that the circuit computes it is the one hypothesis left (module doc). -/
def EncardChargedLe : Prop :=
  ∀ (bk : Book) (N W c : ℕ),
    {a : Fin W → List Bool | ∃ ls, charges bk N a = some ls ∧ ls.sum ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞)

/-- **The leak bound with advice v1's charge.** `EncardAccSeqsWithinLeConstantRate` with the advice string advice v1's
decision streams (`Streams`) and its charge `AdviceV1.bits`, so with no assumption on the charge: the constant-rate
Program's observation sequences of length `n` (within the `W` windows with advice, and the first `m` windows), with
jitter below `D`, accepted with advice charged at most `b w` in every window `w`, number at most
`2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n`. Decision-only windows, and the circuit's agreement with the reference is the one
hypothesis left (module doc). -/
def EncardAccSeqsWithinLeConstantRateV1 : Prop :=
  ∀ {L K S W : ℕ} {In : Type} (bk : Book) (N : ℕ) (b : Fin W → ℕ) (T : ℕ) (ss : SyncSet)
    (frames : List In → Streams bk N W → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T),
    (AccSeqsWithin (AdviceV1.bits bk N) b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **Advice with counts: the charge counts (Kraft).** Under any registered book `bk` and any count code `cc` (lengths
meeting Kraft's inequality, as every prefix-free code's do), with windows of `N` slots, at most `2^c` advice values of
`W` windows, each window's decision stream and counts, are charged `c` bits or fewer in all, with no window refused. A
window's charge is its decisions' charge (`AdviceV1.window`) plus `cc`'s length for its counts (`windowCounted`), so
the windows may send nonzero counts. The code is the same in every window and carries nothing across windows. That
the circuit computes the decisions' charge is the hypothesis left, as for `EncardChargedLe` (module doc), and the
count code is a parameter, a separate code (not advice v1's coding of counts, which is inside the coder). -/
def EncardChargedLeCounted : Prop :=
  ∀ {C : Type} (bk : Book) (N W c : ℕ) (cc : CountCode C),
    {a : Fin W → List Bool × C | ∃ ls, chargesCounted bk N cc a = some ls ∧ ls.sum ≤ c}.encard ≤
      ((2 ^ c : ℕ) : ℕ∞)

/-- **The leak bound with the charge with counts.** `EncardAccSeqsWithinLeConstantRate` with the advice string each
window's decision stream and counts (`StreamsCounted`) and its charge `bitsCounted` under any count code `cc`, so with
no assumption on the charge: the constant-rate Program's observation sequences of length `n` (within the `W` windows
with advice, and the first `m` windows), with jitter below `D`, accepted with advice charged at most `b w` in every
window `w`, number at most `2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n`. The windows may send nonzero counts, and the
scope is `EncardChargedLeCounted`'s. -/
def EncardAccSeqsWithinLeConstantRateV1Counted : Prop :=
  ∀ {L K S W : ℕ} {In C : Type} (bk : Book) (N : ℕ) (cc : CountCode C) (b : Fin W → ℕ) (T : ℕ) (ss : SyncSet)
    (frames : List In → StreamsCounted bk N cc W → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T),
    (AccSeqsWithin (AdviceV1.bitsCounted bk N cc) b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

end NetTiming.Guarantees
