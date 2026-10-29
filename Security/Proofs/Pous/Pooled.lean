import Proofs.Pous.Period
import Proofs.Flock.Soundness.Audit.Periods
import Proofs.Flock.Claims.Sampling.Draws

/-!
# PoUS pooled over `U` windows: how many windows can be accepted short

`PeriodHeld` bounds one window: at every value of the other audits' coins, the verifier accepts the window's PoUS audit
with no break in its data while the capacity account held fewer than `S · len` bit-ticks (`ShortAccepted`) with
probability at most `p = δ + ε_max` (`1% + 2^-128`) over the window's capacity coins. Over `U` windows whose capacity
coins are drawn independently, window after window, the monitoring-period step (`FlockSoundness.Audit.Periods`) pools
it, given that each window is fixed before its own capacity coins (`FixedBefore`): window `u`'s joint run, its audit
and the other audits' coins are functions of the earlier windows' capacity coins, so `PeriodHeld` applies to window `u`,
at its strategy under the device model and the instantiation (`DeviceModel`, `Instantiates`), whatever the earlier
windows' coins. `FixedBefore` is a hypothesis of each statement, not an assumption: it says how the windows are run.

* `pooledShort`: at least `M` of the `U` windows are accepted, unbroken and short with a probability `P` with
  `P · z^M ≤ (1 + (z − 1) p)^U` for every `z ≥ 1`; `pooledShort_choose`: `P ≤ C(U, M) p^M`; `pooledShort_log`:
  `P ≤ δ'` once `M ≥ ((z − 1) U p + ln(1/δ')) / ln z`.
* `pooledSureShort`: every window accepted with no break while at least `M` of them are short on every capacity coin
  (`Accounting.Run.CapacityShort`, which reads only the window's other coins), with probability at most `p^M`.

The two differ because `ShortAccepted` reads the window's own capacity coins: a run short on a `p`-share of each
window's capacity coins, where it is accepted, and full on the rest meets `PeriodHeld` in every window, is accepted in
every window and is short in a binomial `(U, p)` number of them. So at least `M` short accepted windows out of `U` is
unlikely only for `M` above `p · U`, and `p^M` bounds only the windows short whatever their capacity coins.
-/

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties FlockSoundness.Audit FlockSoundness.Game
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- A per-window bound in `ℝ` (`Pouw.pr`) is one on `prCoin`. -/
theorem prCoin_le_ofReal {X : Type} [Fintype X] [Nonempty X] (E : X → Prop) {b : ℝ} (h : Pouw.pr E ≤ b) :
    prCoin E ≤ ENNReal.ofReal b := by
  have hE : prCoin E ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top
    ((prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans_eq (by simp [prCoin, avg_const]))
  rw [← ENNReal.ofReal_toReal hE]
  exact ENNReal.ofReal_le_ofReal ((Claims.Sampling.Pr_eq_toReal_prCoin E).symm.trans_le h)

variable {U : ℕ} {ι α ΩW ΩQ : Type} [Fintype ι] [Fintype α] {d : Fin U → Accounting.Device ι}

/-- **Each window is fixed before its capacity coins**: at the run's capacity coins `x`, window `u`'s joint run
`r u x`, its audit `P u x` and the other audits' coins `o u x` read only the earlier windows' coins `x i`, `i < u`. -/
def FixedBefore
    (r : (u : Fin U) → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
      Accounting.Run (d u) α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Audits ΩW ΩQ)
    (o : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → ΩW × ΩQ) : Prop :=
  Periods.Predictable (Ω := fun _ => (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) r ∧
    Periods.Predictable (Ω := fun _ => (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) P ∧
    Periods.Predictable (Ω := fun _ => (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) o

/-- Window `u` is accepted with no break in its data while its capacity account held fewer than `S · len` bit-ticks
(`PeriodHeld`'s event), at the run's capacity coins `x`. -/
def ShortAccepted
    (r : (u : Fin U) → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
      Accounting.Run (d u) α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Audits ΩW ΩQ)
    (o : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → ΩW × ΩQ) (S : ℝ)
    (u : Fin U) (x : Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) : Prop :=
  (P u x).Accepts ((o u x).1, (o u x).2, x u) ∧ ¬ (P u x).Break ((o u x).1, (o u x).2, x u) ∧
    ((r u x).held ((o u x).1, (o u x).2, x u) : ℝ) < S * ((r u x).win .capacity).len

variable {r : (u : Fin U) → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
      Accounting.Run (d u) α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    {P : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Audits ΩW ΩQ}
    {o : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → ΩW × ΩQ}

omit [Fintype ι] in
theorem FixedBefore.adapted (h : FixedBefore r P o) (S : ℝ) :
    Periods.Adapted (Ω := fun _ => (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
      (ShortAccepted r P o S) := fun u x x' hx => by
  simp only [ShortAccepted, h.1 u x x' fun i hi => hx i hi.le, h.2.1 u x x' fun i hi => hx i hi.le,
    h.2.2 u x x' fun i hi => hx i hi.le, hx u le_rfl]

/-- **One window at a time**: whatever the earlier windows' capacity coins, `PeriodHeld` bounds window `u`'s
`ShortAccepted` over its own capacity coins by `δ + ε_max`. -/
theorem shortAccepted_le (hH : Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) (D Q L : ℕ)
    (hP2 : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (hfix : FixedBefore r P o) (round : ℕ) (S : ℝ)
    (τ : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Strategy ΩW ΩQ)
    (hdev : ∀ u x, DeviceModel (r u x) (P u x) (τ u x) D Q L round) (hinst : ∀ u x, Instantiates (P u x) (τ u x))
    (hS : ∀ u x, S * ((r u x).win .capacity).len + (d u).chip + ((∑ i, (d u).cap i * round : ℕ) : ℝ) ≤
      Pous.EndToEnd.S)
    (u : Fin U) (x : Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :
    prCoin (fun a => ShortAccepted r P o S u (Function.update x u a)) ≤
      ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal) := by
  have e : ∀ a i, i < u → Function.update x u a i = x i := fun a i hi => Function.update_of_ne hi.ne _ _
  have hr : ∀ a, r u (Function.update x u a) = r u x := fun a => hfix.1 u _ _ (e a)
  have hP : ∀ a, P u (Function.update x u a) = P u x := fun a => hfix.2.1 u _ _ (e a)
  have ho : ∀ a, o u (Function.update x u a) = o u x := fun a => hfix.2.2 u _ _ (e a)
  simp only [ShortAccepted, Function.update_self, hr, hP, ho]
  exact prCoin_le_ofReal _
    (periodHeld hH D Q L hP2 ι α ΩW ΩQ (d u) (r u x) (P u x) (τ u x) round S (hdev u x) (hinst u x) (hS u x)
      (o u x))

open Classical in
/-- **PoUS pooled over `U` windows, Chernoff form.** Under `PeriodHeld`'s premises, with each window's strategy under the
device model and the instantiation (`DeviceModel`, `Instantiates`), each window fixed before its capacity coins
(`FixedBefore`) and every window's certificate fitting the ring game's state bound: at least `M` windows accepted with no break while short (`ShortAccepted`) has probability `P` with
`P · z^M ≤ (1 + (z − 1)(δ + ε_max))^U` for every `z ≥ 1`. -/
theorem pooledShort (hH : Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) (D Q L : ℕ)
    (hP2 : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (hfix : FixedBefore r P o) (round : ℕ) (S : ℝ)
    (τ : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Strategy ΩW ΩQ)
    (hdev : ∀ u x, DeviceModel (r u x) (P u x) (τ u x) D Q L round) (hinst : ∀ u x, Instantiates (P u x) (τ u x))
    (hS : ∀ u x, S * ((r u x).win .capacity).len + (d u).chip + ((∑ i, (d u).cap i * round : ℕ) : ℝ) ≤
      Pous.EndToEnd.S)
    (M : ℕ) {z : ℝ≥0∞} (hz : 1 ≤ z) :
    prCoin (fun x => M ≤ (Finset.univ.filter fun u => ShortAccepted r P o S u x).card) * z ^ M ≤
      (1 + (z - 1) * ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal)) ^ U :=
  Periods.prCoin_count_mul_pow_le _ (hfix.adapted S) hz
    (shortAccepted_le hH D Q L hP2 hfix round S τ hdev hinst hS) M

open Classical in
/-- **PoUS pooled over `U` windows, binomial form**: at least `M` windows accepted short with probability at most
`C(U, M) (δ + ε_max)^M`. -/
theorem pooledShort_choose (hH : Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) (D Q L : ℕ)
    (hP2 : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (hfix : FixedBefore r P o) (round : ℕ) (S : ℝ)
    (τ : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Strategy ΩW ΩQ)
    (hdev : ∀ u x, DeviceModel (r u x) (P u x) (τ u x) D Q L round) (hinst : ∀ u x, Instantiates (P u x) (τ u x))
    (hS : ∀ u x, S * ((r u x).win .capacity).len + (d u).chip + ((∑ i, (d u).cap i * round : ℕ) : ℝ) ≤
      Pous.EndToEnd.S)
    (M : ℕ) :
    prCoin (fun x => M ≤ (Finset.univ.filter fun u => ShortAccepted r P o S u x).card) ≤
      U.choose M * ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal) ^ M :=
  Periods.prCoin_count_le_choose _ (hfix.adapted S) (shortAccepted_le hH D Q L hP2 hfix round S τ hdev hinst hS) M

open Classical in
/-- **PoUS pooled over `U` windows, at `δ'`**: with `p = δ + ε_max`, at least `M` windows accepted short has
probability at most `δ'` once `M ≥ ((z − 1) U p + ln(1/δ')) / ln z` for some `z > 1`, so out of `U` windows at least
`U − M + 1` certify `S`. -/
theorem pooledShort_log (hH : Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) (D Q L : ℕ)
    (hP2 : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (hfix : FixedBefore r P o) (round : ℕ) (S : ℝ)
    (τ : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Strategy ΩW ΩQ)
    (hdev : ∀ u x, DeviceModel (r u x) (P u x) (τ u x) D Q L round) (hinst : ∀ u x, Instantiates (P u x) (τ u x))
    (hS : ∀ u x, S * ((r u x).win .capacity).len + (d u).chip + ((∑ i, (d u).cap i * round : ℕ) : ℝ) ≤
      Pous.EndToEnd.S)
    {z δ' : ℝ} (hz : 1 < z) (hδ' : 0 < δ') {M : ℕ}
    (hM : ((z - 1) * U * ((Params.δ : ℝ) + Params.εMax.toReal) + Real.log (1 / δ')) / Real.log z ≤ M) :
    prCoin (fun x => M ≤ (Finset.univ.filter fun u => ShortAccepted r P o S u x).card) ≤ ENNReal.ofReal δ' :=
  Periods.prCoin_count_le_of_log _ (hfix.adapted S) (add_nonneg δ_nonneg ENNReal.toReal_nonneg)
    (shortAccepted_le hH D Q L hP2 hfix round S τ hdev hinst hS) hz hδ' hM

open Classical in
/-- **PoUS pooled over `U` windows, sure shortness**: every window accepted with no break while at least `M` of them
are short on every capacity coin (`Accounting.Run.CapacityShort`, fixed before the window's capacity coins) has
probability at most `(δ + ε_max)^M`. -/
theorem pooledSureShort (hH : Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) (D Q L : ℕ)
    (hP2 : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (hfix : FixedBefore r P o) (round : ℕ) (S : ℝ)
    (τ : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Strategy ΩW ΩQ)
    (hdev : ∀ u x, DeviceModel (r u x) (P u x) (τ u x) D Q L round) (hinst : ∀ u x, Instantiates (P u x) (τ u x))
    (hS : ∀ u x, S * ((r u x).win .capacity).len + (d u).chip + ((∑ i, (d u).cap i * round : ℕ) : ℝ) ≤
      Pous.EndToEnd.S)
    (M : ℕ) :
    prCoin (fun x => (∀ u, (P u x).Accepts ((o u x).1, (o u x).2, x u) ∧
        ¬ (P u x).Break ((o u x).1, (o u x).2, x u)) ∧
      M ≤ (Finset.univ.filter fun u => (r u x).CapacityShort S ((o u x).1, (o u x).2, x u)).card) ≤
      ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal) ^ M := by
  refine Periods.prCoin_forall_count_le_pow
    (Ω := fun _ => (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (fun u x => (P u x).Accepts ((o u x).1, (o u x).2, x u) ∧ ¬ (P u x).Break ((o u x).1, (o u x).2, x u))
    (fun u x => (r u x).CapacityShort S ((o u x).1, (o u x).2, x u)) ?_ ?_
    (fun u x => (prCoin_mono fun a h => ⟨h.1.1, h.1.2, h.2 _⟩).trans
      (shortAccepted_le hH D Q L hP2 hfix round S τ hdev hinst hS u x)) M
  · intro u x x' hx
    simp only [hfix.2.1 u x x' fun i hi => hx i hi.le, hfix.2.2 u x x' fun i hi => hx i hi.le, hx u le_rfl]
  · intro u x x' hx
    simp only [Accounting.Run.CapacityShort, hfix.1 u x x' hx, hfix.2.2 u x x' hx]

end Pous.Period.SecurityProofs
