import Definitions.Pous.P2.Cost

/-!
# P2's open steps in the ring model (named assumptions)

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`: the primality of P2's
modulus (`PrimeP16448`), and the two forms of the conjecture that P2 in the ring model is as hard as `m1p` at the
deployed point (`P2RingAuditSecure`, the weak form a statement takes, and `P2RingReducesToM1p`, the reduction the spec
states, which implies it through the proved `m1p` row: `Pous.SecurityProofs.P2RingAuditSecureOfReduces`).
-/

namespace Pous.P2.Assumptions

open scoped ENNReal

/-- **P2's registered modulus is prime (open in Lean)**: `p = 2^16448 − 21065`. Its primality certificate is ECPP, and
Lean checks none here. `p ≡ 3 mod 4` is proved (`PousP2Ring.p16448_mod_four`). -/
def PrimeP16448 : Prop := Nat.Prime (2 ^ 16448 - 21065)

/-- **`sms-m1-instantiation`, `seq-root` (conjectured): P2 in the ring model at the deployed point, the weak form.**
`p2ring (2^19) 16448 (2^16448 − 21065) 16432` (2,054 payload bytes per block) with sequential reveal and 105
challenges accepts every `⌊ρ·|C|⌋`-bit (`ρ = 18/19`), `(D, Q, L)`-bounded responder with probability at most
`1% + 2^-129 + 2^-130`: the ideal term `2^-129` of `P2ErrorSplitB19Uncond`'s `λ = 128` row and the instantiation's
`2^-130`.

It is the conjecture that square–mask–square over `Z/p` with a random odd labeling and random keys is no easier than
independent ideal permutations, once a responder may not finish a root within the deadline. If `p` is prime
(`PrimeP16448`; both refutations read only the `AuditSecureOn` conjunct of `MeetsOn` and `ε ≤ 2^-128`), it is false at
`D ≥ 32,894` with `Q ≥ 2` and `L ≥ 65,789` (`PousP2Ring.tight_p16448`: the responder recomputes the roots), at every
`k`. Its form at `k ≤ 88` is false at `L ≥ 2^30` (`PousP2Ring.not_meetsOn_b19_k88`: a responder that stores truncated
roots and recomputes the dropped bits); at `k = 105` the same attack needs about 167 dropped bits per block, which no
theorem here refutes or rules out. So it is a hypothesis only at the `(D, Q, L)` a deadline gives. Those two theorems
are its falsification tests. -/
def P2RingAuditSecure (D Q L : ℕ) : Prop :=
  AuditSecureOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) .sequential
    ((p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) D Q L 105
    (ENNReal.ofReal Params.δ) ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130)

/-- **`sms-m1-instantiation` (conjectured): P2–M1-SGI as the spec states it, at the deployed point.** Every
`⌊ρ·|C|⌋`-bit, `(D, Q, L)`-bounded responder against `p2ring (2^19) 16448 (2^16448 − 21065) 16432` with 105
challenges is matched, up to `2^-130`, by a responder against `m1p` on the same domain holding 128 more bits per block
(the `λ = 128` slack), with `2^20` forward queries per answer and some depth. With the proved `m1p` row
(`P2ErrorSplitB19Uncond`, `λ = 128`) it implies `P2RingAuditSecure D Q L`
(`Pous.SecurityProofs.P2RingAuditSecureOfReduces`). Its falsification tests are `P2RingAuditSecure`'s: a refutation of
that is one of this. -/
def P2RingReducesToM1p (D Q L : ℕ) : Prop :=
  ∀ (W : Bits (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n)
    (A₁ : Preprocessor (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432)
      ((p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ))
    (A₂ : SeqResponderOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432)
      ((p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) D Q L),
    ∃ (W' : Bits (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n) (D' : ℕ)
      (A₁' : Preprocessor (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432)
        ((m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ + 128 * 2 ^ 19))
      (A₂' : SeqResponder (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432)
        ((m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ + 128 * 2 ^ 19) D' (2 ^ 20)),
      pr (seqAcceptsOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 W A₁ A₂) ≤
        pr (seqAccepts (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 W' A₁' A₂') + (2⁻¹ : ℝ≥0∞) ^ 130

end Pous.P2.Assumptions
