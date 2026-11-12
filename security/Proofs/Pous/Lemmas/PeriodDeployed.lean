module

public import Properties.Pous.PeriodDeployed
import Mathlib.Basic.ENNReal.Real

/-!
# PoUS over one period: the honest and nontrivial lemmas

The two lemmas beside `Properties.Pous.PeriodDeployed`'s `PeriodHeldDeployed` and `Properties.Pous.Period`'s
`PeriodHeld`: their hypotheses hold at an honest period, and their conclusion fails under a verdict that accepts
everything. Neither is a property, nor a witness of one: a witness constructs its instance and takes no hypothesis
beyond named assumptions, and the honest lemma quantifies over every audit, the nontrivial one takes the period's data's
lack of a break as a hypothesis. Each is a `def G : Prop` under the properties' namespace, proved by the `SecurityProofs`
theorem of its name in lower camel case, and audited with the proofs: no record pins it.

* `PeriodHeldHonest`: at an honest period (`honestRun`) and its honest strategy (`honestStrategy`), the device model
  holds at the deployed budget and the instantiation holds, for every audit under a 32-byte session id; the certificate
  `HONEST_S` is positive and fits the state bound; and the budget is the deadline's at the floor.
* `PeriodHeldNontrivial`: at a period whose capacity account holds nothing (`emptyRun`), a verifier that accepts
  everything fails the conclusion wherever the period's data holds no break, and the bound's event is false at the
  honest period.
-/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **The honest lemma of `PeriodHeld` and `PeriodHeldDeployed`: their hypotheses hold at the honest period.** At
`honestRun` on `honestDevice`, with one tick a round, and the honest strategy (`honestStrategy`), for every audit `P`
whose designated session id has 32 bytes, as the memory challenger's ids do: the device model (`DeviceModel`) holds at
the deployed budget (each answer is a lookup, no query, and the state is `C`'s `FULL` bits, which the account holds at
every tick), and the instantiation (`Instantiates`) holds (the code's sample never fails, and a lookup returns the
challenged block of `C`); the certificate `HONEST_S` is positive and fits `S₀` beside the device's on-chip state and its
side channel's capacity over a round; and the budget is the deadline's at the measured floor:
`DEADLINE_ROUNDS ≤ RING_D`, with `RING_D + 1` rounds taking at least 1.995 × `LIMIT_NS` at `ROUND_NS`.
`P2RingAuditSecure` is an open conjecture, not a fact of the instance. -/
def PeriodHeldHonest : Prop :=
  (∀ P : Audits Unit Unit, P.sid.size = 32 →
      DeviceModel honestRun P honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates P honestStrategy) ∧
    0 < (HONEST_S : ℝ) * (honestRun.win .capacity).len ∧
    (HONEST_S : ℝ) * (honestRun.win .capacity).len + honestDevice.chip +
        ((∑ i, honestDevice.cap i * 1 : ℕ) : ℝ) ≤ Pous.EndToEnd.S ∧
    DEADLINE_ROUNDS ≤ RING_D ∧
    (1995 / 1000 : ℚ) * LIMIT_NS ≤ (RING_D + 1) * ROUND_NS

/-- **The nontrivial lemma of `PeriodHeld` and `PeriodHeldDeployed`.** At the empty period, a verifier that accepts
everything (`P.Accepts` replaced by `True`) fails their conclusion at the certificate `HONEST_S`, wherever the period's
data holds no break: the account held nothing, so the event has probability 1. And the conclusion's event is false at
the honest period, for every audit and coin, since its account held `105 · FULL` bit-ticks. A concrete audit with no
break needs SHA-512 on each of the table's `2^20` strings, so the first conjunct takes it as a hypothesis. -/
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
