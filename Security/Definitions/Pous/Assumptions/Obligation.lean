import Definitions.Pous.P2.Mutable

/-!
# P2's per-round floor for content committed before its key coin (conjectured)

The mutable store re-encodes a KV page under a fresh tag whose coin the memory challenger draws after the page's plaintext
commitment, and the prover chooses what it serves, so a page's content may depend on what it has seen of the ring: the
labeling, through earlier encodings, and the keys of pages written before it. `P2RingRoundSecure` fixes the payload
before the ring is drawn, and that is not a special case of it: a payload correlated with the labeling has no fixed
counterpart, and translating a responder's offset-key answers to a fixed payload needs the payload itself, which a
responder holding `⌊ρ·|C|⌋` bits doesn't have. This file states the conjecture with the payload read from the oracle.
-/

namespace Pous.P2.Assumptions

open scoped ENNReal

/-- **`pous-ring-instantiation` read for content committed before its key coin, at `sms-m1-instantiation`/`seq-root`'s
per-round floor (conjectured).** `P2RingRoundSecure` with a payload drawn from the oracle: for
`p2ring N 16448 (2^16448 − 21065) 16432` with `2^19 ≤ N ≤ 2^25`, every commit order `π` and every payload `W ω` whose block
`j` reads the oracle only through the labeling and the key pairs of the blocks before it in `π` (`CommittedBefore`), and
every responder holding `Sz j` bits at boundary `j` (`VarResponderOn`, each round within `(D, Q, L)`), there is an event
`bad` of probability at most `k·ε₀` outside which, at every boundary `j < k` held at most `S₀ = ⌊ρ·|C|⌋` bits, round `j`
is answered correctly with probability at most `1 − p` given that the first `j` were.

A fixed payload is committed before every key, so this implies `P2RingRoundSecure p ε₀ D Q L`. Why it should hold at the
same `p` and `ε₀`: block `j`'s code is `ρ(σ_K(ρ(m_j + t_j)))` with its key pair `(K_j, t_j)` uniform and independent of
`m_j`, so for each labeling the offset absorbs the content (`Pous.Guarantees.P2ContentAbsorbs`): the code's distribution
over the key pair is the same for every content, and what the prover learned before committing is about other blocks'
independent keys. What it excludes: content that reads its own block's keys. Chosen after both keys, every block can be
made one code `c₀` wherever `c₀`'s decoding is a payload, and a responder holding one block answers every challenge to such
blocks. -/
def P2RingRoundSecureCommitted (p ε₀ : ℝ≥0∞) (D Q L : ℕ) : Prop :=
  ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 25 →
    ∀ (π : Fin N → ℕ)
      (W : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω → Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n),
      Mutable.CommittedBefore (b := 16432) π W →
      ∀ (Sz : ℕ → ℕ) (k : ℕ)
        (A₁ : Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
        (A₂ : VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
        ∃ bad : Outcome (p2ring N 16448 (2 ^ 16448 - 21065) 16432) k → Prop, pr bad ≤ k * ε₀ ∧
          ∀ j < k, Sz j ≤ (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ →
            pr (fun x => varPassesOn _ k (j + 1) (W x.1) A₁ A₂ x ∧ ¬ bad x) ≤
              (1 - p) * pr (fun x => varPassesOn _ k j (W x.1) A₁ A₂ x ∧ ¬ bad x)

end Pous.P2.Assumptions
