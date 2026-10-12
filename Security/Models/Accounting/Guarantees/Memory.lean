import Definitions.Accounting.Memory
import Definitions.Pous.P2.Cost
import Models.Pous.Assumptions.P2

/-!
# Memory adds, by counting (trusted)

Lemma M of the actual-circuit model (note `console/20261012T0645Z-draft-actual-circuit-model`, section 5): a location
that passes a PoUS audit while it keeps `q` other bits, uniform and independent of PoUS's setup and challenges, held
more than PoUS's state bound `S` plus those `q` bits, up to a slack `k`. The model's `mem_X(t)` instantiates the
location's state once the actual circuit's definitions land.

* `Counting`: the counting core. Of the `2^q` values `z` that an `M`-bit state can carry, at most `2^(q − k)` leave more
  than `2^(M − q + k)` states that carry them: the fibres are disjoint, and Markov's inequality over `z`.
* `Adds`: the composition with PoUS's guarantee, taken as a hypothesis in its own form (`AuditSecure`, sequential
  reveal, `c` challenges, error `δ + ε`). If the state held before each challenge determines `z` (`Determines`), the
  audit accepts with `q + S ≥ M + k` with probability at most `δ + ε + c · 2^-k`. The `c` is the union over the `c`
  states a sequential responder holds, one before each challenge, each read by its own decoder: for all but
  `c · 2^-k` of the `z`, every one of them leaves at most `2^(M − q + k) ≤ 2^S` states, so fixing `z` leaves an
  `S`-bit PoUS adversary.
* `AddsOn`: the same in the ring model (`AuditSecureOn`, `(D, Q, L)`-bounded responders).
* `AddsP2Ring`: P2 at its deployed point, under its named assumption `P2RingAuditSecure`: `c = 105` challenges and
  PoUS's error `δ + 2^-129 + 2^-130`, so `δ + 2^-129 + 2^-130 + 105 · 2^-k`.

The probability is over `z`, uniform over `q` bits, and PoUS's outcome (the primitive, the encoder's coins and the
challenges), independently: `z` is fixed before setup, as `W` is, and `W` does not depend on it. The proofs are in
`Security/Proofs/Accounting/Memory/`.
-/

namespace Accounting.Guarantees.Memory

open Accounting.Memory
open scoped ENNReal

/-- **The counting core.** For every map `π` reading a `q`-bit value from an `M`-bit state, at most `2^(q − k)` of the
`2^q` values leave more than `2^(M − q + k)` states (`fibre π z`): a uniform `z` leaves at most that many except with
probability `2^-k`. The fibres partition the `2^M` states, so the values with fibres above `T` number at most
`2^M / T`. -/
def Counting : Prop :=
  ∀ (M q k : ℕ) (π : (Fin M → Bool) → Fin q → Bool),
    2 ^ k * (Finset.univ.filter fun z => 2 ^ (M + k - q) < (fibre π z).card).card ≤ 2 ^ q

/-- **Memory adds, against PoUS's guarantee.** Let PoUS's sequential audit with `c` challenges accept every `S`-bit
adversary with probability at most `δ + ε` (`AuditSecure`). Let a location run, for each `q`-bit `z`, a PoUS adversary
`(A₁ z, A₂ z)` with an `M`-bit state from which `z` can be read back before every challenge (`Determines`). With `z`
uniform and independent of the setup and the challenges, the audit accepts while `q + S ≥ M + k` with probability at
most `δ + ε + c · 2^-k`: an accept gives `q < M − S + k`, except with that probability. -/
def Adds : Prop :=
  ∀ {I : Pous.IdealModel} (sch : Pous.Scheme I) (S D Q c : ℕ) (δ ε : ℝ≥0∞),
    Pous.AuditSecure sch .sequential S D Q c δ ε →
    ∀ (M q k : ℕ) (W : Pous.Bits sch.n) (π : ℕ → Pous.Bits M → Pous.Bits q)
      (A₁ : Pous.Bits q → Pous.Preprocessor sch M) (A₂ : Pous.Bits q → Pous.SeqResponder sch M D Q),
      Determines sch c π A₁ (fun z => (A₂ z).step) →
      Pous.pr (fun x : Pous.Bits q × Pous.Outcome sch c =>
          Pous.seqAccepts sch c W (A₁ x.1) (A₂ x.1) x.2 ∧ ¬ q + S < M + k) ≤
        δ + ε + c * (2⁻¹ : ℝ≥0∞) ^ k

/-- **Memory adds, in the ring model.** `Adds` with `AuditSecureOn` and `(D, Q, L)`-bounded responders
(`SeqResponderOn`) in place of `AuditSecure` and `(D, Q)`-bounded ones. -/
def AddsOn : Prop :=
  ∀ {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (S D Q L c : ℕ) (δ ε : ℝ≥0∞),
    Pous.AuditSecureOn sch .sequential S D Q L c δ ε →
    ∀ (M q k : ℕ) (W : Pous.Bits sch.n) (π : ℕ → Pous.Bits M → Pous.Bits q)
      (A₁ : Pous.Bits q → Pous.Preprocessor sch M) (A₂ : Pous.Bits q → Pous.SeqResponderOn sch M D Q L),
      Determines sch c π A₁ (fun z => (A₂ z).step) →
      Pous.pr (fun x : Pous.Bits q × Pous.Outcome sch c =>
          Pous.seqAcceptsOn sch c W (A₁ x.1) (A₂ x.1) x.2 ∧ ¬ q + S < M + k) ≤
        δ + ε + c * (2⁻¹ : ℝ≥0∞) ^ k

/-- **Memory adds at P2's deployed point.** Under `P2RingAuditSecure D Q L` (P2 in the ring model, 2,054 payload bytes
per block, 105 challenges, state bound `⌊ρ·|C|⌋`), a location that keeps a uniform `q`-bit `z` through the audit,
readable before every challenge, is accepted while `q + ⌊ρ·|C|⌋ ≥ M + k` with probability at most
`δ + 2^-129 + 2^-130 + 105 · 2^-k`. -/
def AddsP2Ring : Prop :=
  ∀ D Q L, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (M q k : ℕ) (W : Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n)
      (π : ℕ → Pous.Bits M → Pous.Bits q)
      (A₁ : Pous.Bits q → Pous.Preprocessor (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) M)
      (A₂ : Pous.Bits q → Pous.SeqResponderOn (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) M D Q L),
      Determines (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 π A₁ (fun z => (A₂ z).step) →
      Pous.pr (fun x : Pous.Bits q × Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 =>
          Pous.seqAcceptsOn (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 W (A₁ x.1) (A₂ x.1) x.2 ∧
            ¬ q + (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ < M + k) ≤
        ENNReal.ofReal Pous.Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) + 105 * (2⁻¹ : ℝ≥0∞) ^ k

end Accounting.Guarantees.Memory
