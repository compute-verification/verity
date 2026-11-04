import Properties.Pous.Period

/-!
# PoUS over one period at any point and block count

`Pous.Period` states the per-period guarantee (`PeriodHeld`) at the deployed point: its audit is `Run.check`, at
`Run.DEPLOYED`, and its ring game is at `2^19` blocks. This module states it at every point `pt` of the composed check
(`Run.checkAt SCHEMES pt`, `Pous.EndToEnd.AcceptedAt`) and every block count `N` of the ring game
(`Pous.EndToEnd.PassesAt N`) that the widened per-round floor covers, `2^19 ≤ N ≤ 2^27`, and at Demo B's store,
`N = 1038·2^16`, for every prover strategy at `N` (`StrategyAt N`) under the device model and the instantiation at `pt`
and `N` (`DeviceModel pt`, `Instantiates pt`), the premises `PeriodHeld` reads at `Run.DEPLOYED` and `2^19`; the ring
game's server is the same fixed extractor (`extract`).

* `PeriodHeldAt`: at every such `pt` and `N`, from the coin commitments' hiding at `N` and `P2RingRoundSecureWide`, the
  verifier's whole decision at `pt` (`AuditsOf.Accepts pt`) accepts a period with no break (`AuditsOf.Break pt`) while
  the capacity account held fewer than `S · len` bit-ticks with probability at most `δ + ε_max`.
* `PeriodHeldDemo`: `PeriodHeldAt` at `N = 1038·2^16`, whose state bound is 1,060,008,348,186 bits.
-/

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **PoUS's per-period guarantee at any point and block count to `2^27`.** `PeriodHeld` with the composed check at
`pt`, the ring game at `N` and `P2RingRoundSecureWide` (with `(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130`) in place of
`P2RingAuditSecure`: given the coin commitments' hiding at `N`, for every joint run whose capacity coins are the ring
game's at `N`, every period's audit `P`, every prover strategy `τ` of its device under the device model (`DeviceModel`)
and the instantiation (`Instantiates`) at `pt` and `N`, every certificate `S` with
`S · len + S_chip + ∑ cap_i · round ≤ SAt N` and every value of the other audits' coins, the verifier's whole decision
at `pt` accepts the period with no break while the capacity account held fewer than `S · len` bit-ticks with
probability at most `δ + ε_max`. Rests on the open conjecture `P2RingRoundSecureWide`, unchecked in `N` above
`2^25`. -/
def PeriodHeldAt : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    (1 - p) ^ 105 + 105 * ε₀ ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
      ∀ (pt : Run.Point) (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 27 →
        Pous.P2.Assumptions.CoinCommitHides (coinViewAt N) ((2⁻¹ : ℝ≥0∞) ^ 131) →
        ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
          (r : Accounting.Run d α ΩW ΩQ ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
          (P : AuditsAt N ΩW ΩQ) (τ : StrategyAt N ΩW ΩQ) (round : ℕ) (S : ℝ),
          DeviceModel pt r P τ D Q L round → Instantiates pt P τ →
            S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ SAt N →
              ∀ o : ΩW × ΩQ,
                Pouw.pr (fun x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
                    P.Accepts pt (o.1, o.2, x) ∧ ¬ P.Break pt (o.1, o.2, x) ∧
                      (r.held (o.1, o.2, x) : ℝ) < S * (r.win .capacity).len) ≤
                  (Params.δ : ℝ) + Params.εMax.toReal

/-- **PoUS's per-period guarantee at Demo B's store**: `PeriodHeldAt` at `N = 1038·2^16` blocks, whose state bound
`⌊ρ·|C|⌋` is 1,060,008,348,186 bits, at every point `pt` (the demo's is `pt.params = ⟨16448, 21065, 2^16, 1038⟩`).
Rests on the open conjecture `P2RingRoundSecureWide`, unchecked in `N` above `2^25`. -/
def PeriodHeldDemo : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    (1 - p) ^ 105 + 105 * ε₀ ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
      ∀ (pt : Run.Point) (N : ℕ) [NeZero N], N = 1038 * 2 ^ 16 →
        Pous.P2.Assumptions.CoinCommitHides (coinViewAt N) ((2⁻¹ : ℝ≥0∞) ^ 131) →
        ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
          (r : Accounting.Run d α ΩW ΩQ ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
          (P : AuditsAt N ΩW ΩQ) (τ : StrategyAt N ΩW ΩQ) (round : ℕ) (S : ℝ),
          DeviceModel pt r P τ D Q L round → Instantiates pt P τ →
            S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ 1060008348186 →
              ∀ o : ΩW × ΩQ,
                Pouw.pr (fun x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
                    P.Accepts pt (o.1, o.2, x) ∧ ¬ P.Break pt (o.1, o.2, x) ∧
                      (r.held (o.1, o.2, x) : ℝ) < S * (r.win .capacity).len) ≤
                  (Params.δ : ℝ) + Params.εMax.toReal

end Pous.Period
