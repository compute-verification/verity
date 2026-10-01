import Pous.P2.Assumptions
import PousProofs.P2.Margin

/-!
# The deployed P2's acceptance, from the bridges and the ideal-model pins

`accept_le`: if the ideal-model scheme is `AuditSecure` with an `S'`-bit state at every depth, then under P2-EXP-IO,
SeqRoot and P2–M1-SGI (`Pous/P2/Assumptions.lean`) a deployed strategy that stores at most `S` bits passes `k`
challenges with probability at most `δ + ε + ε_SGI + ε_EXP`. Each bridge is used once: P2-EXP-IO swaps the deployed
keys for uniform ones, SeqRoot makes the strategy forward-only, and P2–M1-SGI turns it into an ideal-model adversary,
which the pin bounds.

At P2's parameters:

* `accept_le_b19`: `2^19` blocks and `k = 88`, from `Pous.Pinned.P2MeetsM1pB19Uncond`, with SGI onto `⌊ρ·|C|⌋` bits;
* `accept_le_family_105`: `2^8 ≤ B ≤ 2^23` and `k = 105` (the specification's `k`), from the `λ = 128` row of
  `Pous.Pinned.P2SlackFamilyUncond`, with SGI onto `⌊ρ·|C|⌋ + 128·B` bits.

The bounds are only as good as the bridges, which are claims about the deployment that Lean does not check.
-/

namespace Pous.P2

open scoped ENNReal

variable {M : IdealModel} {sch : Scheme M}

/-- **The composition.** The ideal model's sequential bound at state `S'`, P2-EXP-IO, SeqRoot and P2–M1-SGI onto
`S'` bound a deployed strategy's acceptance by `δ + ε + ε_SGI + ε_EXP`. -/
theorem accept_le (A : RealAudit sch) {S S' Q k : ℕ} {δ ε εexp εsgi : ℝ≥0∞}
    (hsec : ∀ D, AuditSecure sch .sequential S' D Q k δ ε)
    (hexp : ExpIO A S εexp) (hseq : SeqRoot A S Q) (hsgi : SGI A S S' Q εsgi)
    (W : Bits sch.n) (s : A.Strategy) (hs : A.Bounded S s) :
    A.accept k W s ≤ δ + ε + εsgi + εexp := by
  obtain ⟨D, A₁, A₂, h⟩ := hsgi k W s hs (hseq s hs)
  calc A.accept k W s ≤ A.acceptIdeal k W s + εexp := hexp k W s hs
    _ ≤ pr (seqAccepts sch k W A₁ A₂) + εsgi + εexp := by gcongr
    _ ≤ δ + ε + εsgi + εexp := by gcongr; exact hsec D W A₁ A₂

/-- **The MVP's block count, `k = 88`.** A deployed strategy on `m1p (2^19) 16448 (2^16448 − 21065) b`'s payloads,
storing at most `S` bits, passes 88 challenges with probability at most `1% + 2^-128 + ε_SGI + ε_EXP`, given the
bridges with `2^20` forward evaluations per answer and SGI onto `⌊ρ·|C|⌋` bits. -/
theorem accept_le_b19 (b : ℕ) (hsp : 20 * 16448 ≤ 21 * b) (hb : b ≤ 16447)
    (A : RealAudit (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b)) {S : ℕ} {εexp εsgi : ℝ≥0∞}
    (hexp : ExpIO A S εexp) (hseq : SeqRoot A S (2 ^ 20))
    (hsgi : SGI A S ((m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b).stateBound Params.ρ) (2 ^ 20) εsgi)
    (W : Bits (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b).n) (s : A.Strategy) (hs : A.Bounded S s) :
    A.accept 88 W s ≤ ENNReal.ofReal Params.δ + Params.εMax + εsgi + εexp :=
  accept_le A (fun D => (PousP2p.p2_meets_m1p_b19 b hsp hb D).2.2.2.2) hexp hseq hsgi W s hs

/-- **Every block count from `2^8` to `2^23`, `k = 105`.** A deployed strategy on `m1p B 16448 (2^16448 − 21065) b`'s
payloads, storing at most `S` bits, passes 105 challenges with probability at most `1% + 2^-128 + ε_SGI + ε_EXP`, given
the bridges with `2^20` forward evaluations per answer and SGI onto `⌊ρ·|C|⌋ + 128·B` bits: the instantiation may leak
128 bits per block beyond the ideal model. -/
theorem accept_le_family_105 (B : ℕ) [NeZero B] (hB₀ : 2 ^ 8 ≤ B) (hB₁ : B ≤ 2 ^ 23) (b : ℕ)
    (hsp : 20 * 16448 ≤ 21 * b) (hb : b ≤ 16447)
    (A : RealAudit (m1p B 16448 (2 ^ 16448 - 21065) b)) {S : ℕ} {εexp εsgi : ℝ≥0∞}
    (hexp : ExpIO A S εexp) (hseq : SeqRoot A S (2 ^ 20))
    (hsgi : SGI A S ((m1p B 16448 (2 ^ 16448 - 21065) b).stateBound Params.ρ + 128 * B) (2 ^ 20) εsgi)
    (W : Bits (m1p B 16448 (2 ^ 16448 - 21065) b).n) (s : A.Strategy) (hs : A.Bounded S s) :
    A.accept 105 W s ≤ ENNReal.ofReal Params.δ + Params.εMax + εsgi + εexp :=
  accept_le A (fun D => (PousMargin.fam_128_105 B hB₀ hB₁ b hsp hb D).2.2.2.2) hexp hseq hsgi W s hs

end Pous.P2
