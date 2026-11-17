module

public import Properties.Pous.PeriodDeployed
import Mathlib.Basic.ENNReal.Real

/-! The honest and nontrivial lemmas of PoUS over one period; neither is a property or a witness, and no record pins
them. -/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- The honest lemma of `PeriodHeld` and `PeriodHeldDeployed`; `P2RingAuditSecure` is an open conjecture, not a fact
of the instance. -/
def PeriodHeldHonest : Prop :=
  (∀ P : Audits Unit Unit, P.sid.size = 32 →
      DeviceModel honestRun P honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates P honestStrategy) ∧
    0 < (HONEST_S : ℝ) * (honestRun.win .capacity).len ∧
    (HONEST_S : ℝ) * (honestRun.win .capacity).len + honestDevice.chip +
        ((∑ i, honestDevice.cap i * 1 : ℕ) : ℝ) ≤ Pous.EndToEnd.S ∧
    DEADLINE_ROUNDS ≤ RING_D ∧
    (1995 / 1000 : ℚ) * LIMIT_NS ≤ (RING_D + 1) * ROUND_NS

/-- The nontrivial lemma of `PeriodHeld` and `PeriodHeldDeployed`. Its first conjunct takes the period's data's lack
of a break as a hypothesis, because a concrete audit with no break needs SHA-512 on each of the table's `2^20`
strings. -/
def PeriodHeldNontrivial : Prop :=
  (∀ (P : Audits Unit Unit) (o : Unit × Unit),
      (∀ x, ¬ P.Break (o.1, o.2, x)) →
        ¬ Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
            True ∧ ¬ P.Break (o.1, o.2, x) ∧
              (emptyRun.held (o.1, o.2, x) : ℝ) < HONEST_S * (emptyRun.win .capacity).len) ≤
          (Params.δ : ℝ) + Params.εMax.toReal) ∧
    ∀ (P : Audits Unit Unit) (o : Unit × Unit) (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams),
      ¬ (P.Accepts (o.1, o.2, x) ∧ ¬ P.Break (o.1, o.2, x) ∧
          (honestRun.held (o.1, o.2, x) : ℝ) < HONEST_S * (honestRun.win .capacity).len)

end Pous.Period
