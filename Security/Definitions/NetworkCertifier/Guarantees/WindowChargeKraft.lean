import Definitions.NetworkCertifier.WindowCharge
import Definitions.NetworkCertifier.AdviceString
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# Advice v1's whole window charged: Kraft, and the leak bound with no assumption on the charge

The leak bounds of `Guarantees.AdviceString` take any charge `bits` that meets `BitsCount`: at most `2^c` advice values
are charged `c` bits or fewer in all. Advice v1's whole-window charge (`WindowCharge.window`, `ChargeCount_v2{N, H, O}`:
its decisions, its hits and its order entries, with every count, as `code_v1` codes them in one segment) meets it,
under any registered book: `EncardChargedLeWhole`. So the leak bound on the constant-rate Program holds with that
charge and no assumption on it (`EncardAccSeqsWithinLeConstantRateV1Whole`).

**Why, without the codewords.** As for `EncardChargedLe`: a state weighs its interval's width over `2^bits`, a
symbol's outcomes split that weight, and the finish charges at most what the last state's weight pays for. A whole
window is a sequence of symbols, each of whose models (its total, and the intervals of its outcomes) the book, the
carry and the symbols before it fix. Two windows' sequences first differ at a symbol of one model, at disjoint
intervals: a count before the values it counts, a decision's two outcomes, an integer's bins, a bin's offsets
(`charge-chain.md`, "The whole window"). So the weights of the window's sequences sum to at most the first state's,
and the weights `2^(−charge)` of a run's windows to at most 1.

**Scope.** Every advice v1 window, from any carry chained from the book's first (`Book.first`): its decisions, hits
and order entries, with n_resumes and n_lag sent as 0 (advice v1 never sends resumes or lag bits). Which slots ask is
not charged, as for `EncardChargedLe`. A `Window` holds each field's values at its enabled slots, in order, so these
statements count advice values, not slot arrangements. The slice that wires `ChargeLink_v2` takes `hen` and `oen` from
the replay, as `en` is (enabled slots first, never the prover's), and reads no disabled slot's value (`charge-chain.md`,
"The slice").

**What is left, the one hypothesis.** That `ChargeCount_v2`'s circuit computes `WindowCharge.window`. It is not proved
in Lean: circuit-check holds the circuit to the Python reference (`window_charge.py`'s `reference_window`), and the
Lean functions are that reference one for one, held to it by the window charge's difftest (the certifier program's
`NetworkCertifier.WindowCharge`, proved equal to these by `CodeChargesWhole`, against `reference_window` on 4,000
random instances, accepted and refused).
-/

namespace NetTiming.Guarantees

open WindowCharge

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **Advice v1's whole-window charge counts (Kraft).** Under any registered book `bk`, with windows of `N` decision
slots, `H` hit slots and `O` order slots, at most `2^c` advice values of `W` windows (each window's decisions, hits and
order entries) are charged `c` bits or fewer in all, with no window refused. The charge is `ChargeCount_v2`'s as its
Python reference defines it: that the circuit computes it is the one hypothesis left (module doc). -/
def EncardChargedLeWhole : Prop :=
  ∀ (bk : Book) (N H O W c : ℕ),
    {a : Fin W → Window | ∃ ls, charges bk N H O a = some ls ∧ ls.sum ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞)

/-- **The leak bound with advice v1's whole-window charge.** `EncardAccSeqsWithinLeConstantRate` with the advice
string each window's decisions, hits and order entries (`WindowCharge.Streams`) and its charge `WindowCharge.bits`,
so with no assumption on the charge: the constant-rate Program's observation sequences of length `n` (within the `W`
windows with advice, and the first `m` windows), with jitter below `D`, accepted with advice charged at most `b w` in
every window `w`, number at most `2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n`. The circuit's agreement with the reference
is the one hypothesis left (module doc). -/
def EncardAccSeqsWithinLeConstantRateV1Whole : Prop :=
  ∀ {L K S W : ℕ} {In : Type} (bk : Book) (N H O : ℕ) (b : Fin W → ℕ) (T : ℕ) (ss : SyncSet)
    (frames : List In → Streams bk N H O W → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T),
    (AccSeqsWithin (WindowCharge.bits bk N H O) b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

end NetTiming.Guarantees
