import Definitions.Pous.P2.Cost
import Definitions.Pous.P2.Rounds

/-!
# P2's open steps in the ring model (named assumptions)

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`: the primality of P2's
modulus (`PrimeP16448`), and the forms of the conjecture that P2 in the ring model is as hard as `m1p`: per round and
at every block count from `2^19` to `2^25` (`P2RingRoundSecure`, which gives the deployed point's whole-audit form
through `Pous.SecurityProofs.P2RingAuditSecureOfRoundSecure` and the graded bound through
`Pous.SecurityProofs.P2RingGradedOfRoundSecure`), and at the deployed point (`P2RingAuditSecure`, the weak form a
statement takes, and `P2RingReducesToM1p`, the reduction the spec states, which implies it through the proved `m1p`
row: `Pous.SecurityProofs.P2RingAuditSecureOfReduces`).
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
roots and recomputes the dropped bits), and at `k = 105` it is false at `L ≥ 2^169` with `D ≥ 2` and `Q ≥ 2`
(`Pous.Period.SecurityProofs.not_p2RingAuditSecure_trunc`: the same attack with 168 dropped bits per block). So it is
a hypothesis only at the `(D, Q, L)` a deadline gives, as `Pous.Period.PeriodHeldDeployed` takes it at
`(16,446, 2^20, 2^64)`. Those three theorems are its falsification tests. It follows from the per-round form `P2RingRoundSecure p ε₀ D Q L` whenever
`(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130` (`Pous.SecurityProofs.P2RingAuditSecureOfRoundSecure`). -/
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

/-- **`sms-m1-instantiation`, `seq-root` (conjectured): P2 in the ring model, per round, at every block count from
`2^19` to `2^25`.** For `p2ring N 16448 (2^16448 − 21065) 16432` with `2^19 ≤ N ≤ 2^25`: take any `k`, any payload
`W`, and any responder whose state at boundary `j` is `Sz j` bits (`VarResponderOn`, each round within
`(D, Q, L)`). Then there is an event `bad` of probability at most `k·ε₀` outside which, at every boundary `j < k`
where the state is at most `S₀ = ⌊ρ·|C|⌋` bits (`ρ = 18/19`, `stateBound`), the uniform challenge `i_j` is answered
correctly with probability at most `1 − p` given that the first `j` answers were:
`Pr[first j + 1 correct, not bad] ≤ (1 − p) · Pr[first j correct, not bad]`.

`p` is the per-round floor and `ε₀` the bad-event mass a round may add, both parameters: the conjecture is that this
holds at the `p` and `ε₀` a certificate cites. The bad event's mass is pinned to `k·ε₀`, one `ε₀` per round and the
same at every `N`, so the one `ε₀` the deployed derivation fixes bounds it at every block count and audit length: a
free `η N k` could be `1` everywhere but at `(2^19, 105)` and leave the conjecture empty there. `k·ε₀` is the simplest
such shape that keeps today's error exactly. With `(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130` it gives
`P2RingAuditSecure D Q L` (`Pous.SecurityProofs.P2RingAuditSecureOfRoundSecure`), and `p = 0.043` with
`ε₀ = 2^-137` meets that (`PousP2Rounds.floor_043_eps`: `105·2^-137 < 2^-130`, and `0.957^105 ≈ 0.0099`); with
`105·ε₀ ≤ 2^-130` it needs `p ≥ 1 − 0.01^(1/105) ≈ 0.04291`. At `ε₀ = 2^-137`, `k·ε₀ < 1` for every `k < 2^137`. The
abstract model's bad event, Lemma A's `|Hist_k|·Γ`, is not linear in `k`, but in `m1p`'s `λ = 128` row (`2^19`
blocks, `k = 104`) it is below
`2^-1,959,876` (`PousMargin.log_b19_128`), far below `k·2^-137`. A responder at most `S₀` bits at `m` of the `k`
boundaries passes with probability at most `(1 − p)^m + k·ε₀` (`Pous.SecurityProofs.P2RingGradedOfRoundSecure`).
Nothing here derives a value of `p` for P2.

The modulus, the block width `w = 16448` and the payload width `b = 16432` are the same at every `N`: they set one
block's cost, `N` sets only the number of blocks, the challenge space and `S₀`. The range is the deployed sizes, from
the MVP's `2^19` blocks (1.1 GB of payload) to item 10's `2^25` (69 GB). The falsification tests below don't depend on
`N` beyond the state bound, but nobody has checked the `ε₀` a proof would give against `N`, so the range is not
widened to the abstract model's.

Allowing `bad` to depend on the responder and on `k` makes this weaker than a bound paid per round. Each round
is a fresh uniform index, independent of the history, so a round's pass rate is the share of indices the state held
at that boundary answers. The abstract model already argues per state: Lemma A for `m1p` (`PousP2p.lemmaA_m1p`)
bounds every state, outside an event of mass `Γ`, to fewer than `g` answerable indices, unioned over the `|Hist_k|`
histories. That is a per-round factor `(g − 1)/B` with bad-event mass `|Hist_k|·Γ`. At the reduction's `λ = 128`
slack (`PousMargin.log_b19_128`: `g = 501,579` at `2^19` blocks, `k = 104`) the factor is `501,578/2^19`, so
`p = 22,710/2^19 ≈ 0.0433`, above both `0.043` and the derivation's `0.04291` floor. It is evidence about `m1p` with
128 more bits per block, not a proof for P2.

Falsification tests, at every `N` in the range and if `p` is prime: recomputation (`RingTight`'s responder, which keeps
no state) forces `p = 0` unless `ε₀ ≥ 1`, at `D ≥ 32,894`, `Q ≥ 2`, `L ≥ 65,789` (`PousP2Rounds.round_tight`), and
keeping `m` blocks without their low `t` bits, `m·(16448 − t) ≤ S₀`, forces `m/N ≤ 1 − p + ε₀` at `D ≥ 2`, `Q ≥ 2`,
`L ≥ 2^(t+1)` (`PousP2Rounds.round_trunc`; at `2^19` and `p = 0.043` it refutes from `t = 166`, `m = 501,757`, so
`L ≥ 2^167`, and `m/N` hardly moves with `N`). -/
def P2RingRoundSecure (p ε₀ : ℝ≥0∞) (D Q L : ℕ) : Prop :=
  ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 25 →
    ∀ (Sz : ℕ → ℕ) (k : ℕ) (W : Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
      (A₁ : Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
      (A₂ : VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
      ∃ bad : Outcome (p2ring N 16448 (2 ^ 16448 - 21065) 16432) k → Prop, pr bad ≤ k * ε₀ ∧
        ∀ j < k, Sz j ≤ (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ →
          pr (fun x => varPassesOn _ k (j + 1) W A₁ A₂ x ∧ ¬ bad x) ≤
            (1 - p) * pr (fun x => varPassesOn _ k j W A₁ A₂ x ∧ ¬ bad x)

/-- **`P2RingRoundSecure` to `2^27` blocks (conjectured; unchecked in `N` above `2^25`).** The same statement as
`P2RingRoundSecure`, at every block count from `2^19` to `2^27`, so that it reaches the demo's store, `1038·2^16 ≈
2^26.02` blocks (139.7 GB of payload). Nobody has checked the `ε₀` a proof would give against `N` above `2^25`: that
analysis is pending. Its range contains `P2RingRoundSecure`'s, so it
implies that one (`Pous.SecurityProofs.roundSecure_of_wide`) and every refutation of that one refutes it: the
falsification tests there (`PousP2Rounds.round_tight`, `PousP2Rounds.round_trunc`) and its scaled-point test. -/
def P2RingRoundSecureWide (p ε₀ : ℝ≥0∞) (D Q L : ℕ) : Prop :=
  ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 27 →
    ∀ (Sz : ℕ → ℕ) (k : ℕ) (W : Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
      (A₁ : Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
      (A₂ : VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
      ∃ bad : Outcome (p2ring N 16448 (2 ^ 16448 - 21065) 16432) k → Prop, pr bad ≤ k * ε₀ ∧
        ∀ j < k, Sz j ≤ (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ →
          pr (fun x => varPassesOn _ k (j + 1) W A₁ A₂ x ∧ ¬ bad x) ≤
            (1 - p) * pr (fun x => varPassesOn _ k j W A₁ A₂ x ∧ ¬ bad x)

end Pous.P2.Assumptions
