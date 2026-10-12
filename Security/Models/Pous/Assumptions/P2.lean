import Definitions.Pous.P2.Cost

/-!
# P2's open steps in the ring model (named assumptions)

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`: the primality of P2's
modulus (`PrimeP16448`), the two forms of the conjecture that P2 in the ring model is as hard as `m1p` at the deployed
point (`P2RingRoundSecure`, per round, the form a statement takes, and `P2RingReducesToM1p`, the reduction the spec
states), and the hiding of the memory challenger's coin commitments (`CoinCommitHides`). `P2RingAuditSecure`, the
whole audit's form, is no longer assumed: it follows from either form (`Pous.SecurityProofs.P2RingAuditSecureOfRound`,
`Pous.SecurityProofs.P2RingAuditSecureOfReduces`, through the proved `m1p` row).
-/

namespace Pous.P2.Assumptions

open scoped ENNReal

/-- **P2's registered modulus is prime (open in Lean)**: `p = 2^16448 − 21065`. Its primality certificate is ECPP, and
Lean checks none here. `p ≡ 3 mod 4` is proved (`PousP2Ring.p16448_mod_four`). -/
def PrimeP16448 : Prop := Nat.Prime (2 ^ 16448 - 21065)

/-- **`sms-m1-instantiation`, `seq-root` (conjectured): P2 in the ring model at the deployed point, per round.**
Against `p2ring (2^19) 16448 (2^16448 − 21065) 16432` (2,054 payload bytes per block), a responder whose state at a
round's boundary is at most `⌊ρ·|C|⌋` bits (`ρ = 18/19`, `8,169,621,180` bits) answers that round's uniform index
within `(D, Q, L)` with probability at most `1 − p`, `p = 22,710/2^19 ≈ 4.33%`, off an event of probability at most
`2^-136` over the setup (`RoundSecureOn` at `c = 501,578/2^19`, `ε = 2^-136`). The state and the answering program are
any functions of the setup and of the history before the round, so the bound holds at every round, after any history,
whatever the state was at the other rounds.

`501,578/2^19` is the per-round bound of the `m1p` row that holds 128 more bits per block: `g − 1` of `2^19` blocks,
`g = 501,579`, in `P2ErrorSplitB19Uncond`'s `λ = 128` row. `2^-136` is the instantiation's error per round, and
`105 · 2^-136 ≤ 2^-129`. It implies the whole audit's form `P2RingAuditSecure` at 105 challenges, since
`(501,578/2^19)^105 ≤ 1%` (`Pous.SecurityProofs.P2RingAuditSecureOfRound`), and it bounds every round at which a
server's state is at most `⌊ρ·|C|⌋` bits, which the whole audit's form does not (`Pous.EndToEnd.EndToEndRingGraded`).

It is the conjecture that square–mask–square over `Z/p` with a random odd labeling and random keys is no easier, round
by round, than independent ideal permutations, once a responder may not finish a root within the deadline. If `p` is
prime (`PrimeP16448`), it is false at `D ≥ 32,894` with `Q ≥ 2` and `L ≥ 65,789` (`PousP2Ring.not_roundSecure_tight`:
the responder holds nothing and recomputes the roots), and at `D ≥ 2` and `Q ≥ 2` with `L ≥ 2^162`
(`PousP2Ring.not_roundSecure_trunc`: a responder that keeps `501,579` blocks without their low 161 bits and searches
the `2^161` completions). So it is a hypothesis only at the `(D, Q, L)` a deadline gives. Those two theorems are its
falsification tests. -/
def P2RingRoundSecure (D Q L : ℕ) : Prop :=
  RoundSecureOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432)
    ((p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) D Q L (501578 / 2 ^ 19)
    ((2⁻¹ : ℝ≥0∞) ^ 136)

/-- **P2 in the ring model at the deployed point, the whole audit's form (no longer assumed).**
`p2ring (2^19) 16448 (2^16448 − 21065) 16432` (2,054 payload bytes per block) with sequential reveal and 105
challenges accepts every `⌊ρ·|C|⌋`-bit (`ρ = 18/19`), `(D, Q, L)`-bounded responder with probability at most
`1% + 2^-129 + 2^-130`: the ideal term `2^-129` of `P2ErrorSplitB19Uncond`'s `λ = 128` row and the instantiation's
`2^-130`.

No statement takes it as a hypothesis. It follows from the per-round form, `P2RingRoundSecure D Q L`
(`Pous.SecurityProofs.P2RingAuditSecureOfRound`), and from the reduction the spec states, `P2RingReducesToM1p D Q L`
(`Pous.SecurityProofs.P2RingAuditSecureOfReduces`). If `p` is prime (`PrimeP16448`; both refutations read only the
`AuditSecureOn` conjunct of `MeetsOn` and `ε ≤ 2^-128`), it is false at `D ≥ 32,894` with `Q ≥ 2` and `L ≥ 65,789`
(`PousP2Ring.tight_p16448`: the responder recomputes the roots), at every `k`. Its form at `k ≤ 88` is false at
`L ≥ 2^30` (`PousP2Ring.not_meetsOn_b19_k88`: a responder that stores truncated roots and recomputes the dropped bits);
at `k = 105` the same attack needs about 167 dropped bits per block, which no theorem here refutes or rules out. -/
def P2RingAuditSecure (D Q L : ℕ) : Prop :=
  AuditSecureOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) .sequential
    ((p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) D Q L 105
    (ENNReal.ofReal Params.δ) ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130)

/-- **`sms-m1-instantiation` (conjectured): P2–M1-SGI as the spec states it, at the deployed point.** Every
`⌊ρ·|C|⌋`-bit, `(D, Q, L)`-bounded responder against `p2ring (2^19) 16448 (2^16448 − 21065) 16432` with 105
challenges is matched, up to `2^-130`, by a responder against `m1p` on the same domain holding 128 more bits per block
(the `λ = 128` slack), with `2^20` forward queries per answer and some depth. With the proved `m1p` row
(`P2ErrorSplitB19Uncond`, `λ = 128`) it implies `P2RingAuditSecure D Q L`
(`Pous.SecurityProofs.P2RingAuditSecureOfReduces`), so a refutation of that is one of this: its falsification tests are
`PousP2Ring.tight_p16448` and `PousP2Ring.not_meetsOn_b19_k88`. -/
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

/-- **`statistical-hiding` (open in Lean): the memory challenger's coin commitments hide its indices, up to `ε`.**
`com s c I` is what the challenger publishes about the indices `I` before a session `s`, from its fresh coins `c`
(for the deployed code, `Pous.EndToEnd.coinView`: the coin tree's root and its hm96 key). For every session, and every
event `E` of a quantity `a` drawn independently of the coins and indices (a server's whole view apart from them), the
event holds with the published commitment and the indices it commits to at most `ε` more often than with the
commitment to independent indices `I'`. Unbounded `E`: the hiding is statistical.

Why it should hold at `ε = 2^-131`, for every hash: each coin is an hm96 commitment `b ‖ c` with `b = x ⊕ M·y` and
`c = H(SALT_TAG ‖ y)`, the salt `y` uniform over 1,536 bits and `M` the 512 × 1,536 Toeplitz matrix of the fresh key,
uniform over its 2,047 free bits and published. Given `c` (512 bits), `y` keeps average min-entropy at least 1,024, and
Toeplitz matrices are a universal family, so by the generalized leftover hash lemma `(M, M·y, c)` is within
`½·2^((512 − 1024)/2) = 2^-257` of `(M, U, c)`, whatever `H` is; a hybrid over the 105 coins, from `I` to uniform masks
and back to `I'`, gives at most `210 · 2^-257 < 2^-249`, and the tree's root is a function of the key and the coins.
Lean has neither the leftover hash lemma nor a proof that the code's `Hm96.mask` is that Toeplitz product, so it is
a named assumption about the code's function, not an open conjecture of mathematics alone. -/
def CoinCommitHides {σ Coins Γ : Type} [Fintype Coins] {K B : ℕ} (com : σ → Coins → (Fin K → Fin B) → Γ)
    (ε : ℝ≥0∞) : Prop :=
  ∀ (s : σ) (α : Type) [Fintype α] (E : α → Γ → (Fin K → Fin B) → Prop),
    pr (fun x : α × Coins × (Fin K → Fin B) => E x.1 (com s x.2.1 x.2.2) x.2.2) ≤
      pr (fun x : α × Coins × (Fin K → Fin B) × (Fin K → Fin B) => E x.1 (com s x.2.1 x.2.2.2) x.2.2.1) + ε

end Pous.P2.Assumptions
