import Definitions.NetworkCertifier.AdviceV1
import Definitions.NetworkCertifier.AdviceString
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# Advice v1's charge counts: Kraft, and the leak bound with no assumption on the charge

The leak bounds of `Guarantees.AdviceString` take any charge `bits` that meets `BitsCount`: at most `2^c` advice values
are charged `c` bits or fewer in all. Advice v1's charge (`AdviceV1.window`, `ChargeCount_v1{N}`) meets it, under any
registered book: `EncardChargedLe`. So the leak bound on the constant-rate Program holds with advice v1's charge and no
assumption on it (`EncardAccSeqsWithinLeConstantRateV1`).

**Why, without the codewords.** An arithmetic coder's charge counts like a prefix-free code's length. Weigh a state by
its interval's width over `2^bits`. A scaling doubles the width and counts a bit, so it keeps the weight. A symbol's
outcomes take disjoint sub-intervals, so their weights sum to at most the state's. The finish of a used window starts
from an interval wider than `2^30`, so `2^(−charge) · 2^32` is at most its weight. Summed over every decision stream, the
weights `2^(−charge)` come to at most 1, and at most `2^c` streams are charged `c` or less.

**What this covers.** The Lean functions are the Python references (`arith_encode.py`'s `reference_charge`), one for one.
That `ChargeCount_v1`'s circuit computes them is circuit-check's (its Definition is held to the reference), not
`security`'s.
-/

namespace NetTiming.Guarantees

open AdviceV1

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **Advice v1's charge counts (Kraft).** Under any registered book `bk`, with windows of `N` slots, at most `2^c`
decision streams of `W` windows are charged `c` bits or fewer in all, with no window refused. -/
def EncardChargedLe : Prop :=
  ∀ (bk : Book) (N W c : ℕ),
    {a : Fin W → List Bool | ∃ ls, charges bk N a = some ls ∧ ls.sum ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞)

/-- **The leak bound with advice v1's charge.** `EncardAccSeqsWithinLeConstantRate` with the advice string advice v1's
decision streams (`Streams`) and its charge `AdviceV1.bits`, so with no assumption on the charge: the constant-rate
Program's observation sequences of length `n` (within the `W` windows with advice, and the first `m` windows), with
jitter below `D`, accepted with advice charged at most `b w` in every window `w`, number at most
`2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n`. -/
def EncardAccSeqsWithinLeConstantRateV1 : Prop :=
  ∀ {L K S W : ℕ} {In : Type} (bk : Book) (N : ℕ) (b : Fin W → ℕ) (T : ℕ) (ss : SyncSet)
    (frames : List In → Streams bk N W → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T),
    (AccSeqsWithin (AdviceV1.bits bk N) b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

end NetTiming.Guarantees
