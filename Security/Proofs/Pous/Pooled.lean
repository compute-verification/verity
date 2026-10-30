import Properties.Pous.Pooled
import Proofs.Pous.PeriodDeployed
import Proofs.Flock.Soundness.Audit.Periods
import Proofs.Flock.Claims.Sampling.Draws
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# PoUS pooled over `U` windows: how many windows can be accepted short

`PeriodHeld` bounds one window: at every value of the other audits' coins, the verifier accepts the window's PoUS audit
with no break in its data while the capacity account held fewer than `S · len` bit-ticks (`ShortAccepted`) with
probability at most `p = δ + ε_max` (`1% + 2^-128`) over the window's capacity coins. Over `U` windows whose capacity
coins are drawn independently, window after window, the monitoring-period step (`FlockSoundness.Audit.Periods`) pools
it, given that each window is fixed before its own capacity coins (`FixedBefore`): window `u`'s joint run, its audit
and the other audits' coins are functions of the earlier windows' capacity coins, so `PeriodHeld` applies to window `u`,
at its strategy under the device model and the instantiation (`DeviceModel`, `Instantiates`), whatever the earlier
windows' coins. `FixedBefore` is a named assumption (`pous-ring-device`, `trusted-host`), which the memory challenger
enforces by releasing each window's capacity coins only after that window's commitment is in.

* `pooledShort`: at least `M` of the `U` windows are accepted, unbroken and short with a probability `P` with
  `P · z^M ≤ (1 + (z − 1) p)^U` for every `z ≥ 1`; `pooledShort_choose`: `P ≤ C(U, M) p^M`; `pooledShort_log`
  (`Pous.Period.PooledShortLog`): `P ≤ δ'` once `M ≥ ((z − 1) U p + ln(1/δ')) / ln z`.
* `pooledSureShort`: every window accepted with no break while at least `M` of them are short on every capacity coin
  (`Accounting.Run.CapacityShort`, which reads only the window's other coins), with probability at most `p^M`.
* `pooledShort_log_honest`: `PooledShortLog`'s premises at `10,080` honest periods (`periodHeldHonest`'s device
  premises and certificate, each window reading no capacity coin), and its count condition at `z = 2`, `δ' = 2^-64`,
  `M = 210`: `10,080 · (1% + 2^-128) ≤ 146 · ln 2`. `pooledShort_log_nontrivial`: at `10,080` empty periods the
  accept-all event holds on every coin, so it has probability `1 > 2^-64`.

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

/-- A per-window bound on `prCoin` is one in `ℝ` (`Pouw.pr`). -/
theorem pr_le_of_prCoin_le {X : Type} [Fintype X] (E : X → Prop) {b : ℝ} (hb : 0 ≤ b)
    (h : prCoin E ≤ ENNReal.ofReal b) : Pouw.pr E ≤ b :=
  (Claims.Sampling.Pr_eq_toReal_prCoin E).trans_le (ENNReal.toReal_le_of_le_ofReal hb h)

variable {U : ℕ} {ι α ΩW ΩQ : Type} [Fintype ι] [Fintype α] {d : Fin U → Accounting.Device ι}

variable {r : (u : Fin U) → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
      Accounting.Run (d u) α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    {P : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Audits ΩW ΩQ}
    {o : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → ΩW × ΩQ}

omit [Fintype ι] in
theorem fixedBefore_adapted (h : FixedBefore r P o) (S : ℝ) :
    Periods.Adapted (Ω := fun _ => (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
      (ShortAccepted r P o S) := fun u x x' hx => by
  simp only [ShortAccepted, h.1 u x x' fun i hi => hx i hi.le, h.2.1 u x x' fun i hi => hx i hi.le,
    h.2.2 u x x' fun i hi => hx i hi.le, hx u le_rfl]

/-- **One window at a time**: whatever the earlier windows' capacity coins, `PeriodHeld` bounds window `u`'s
`ShortAccepted` over its own capacity coins by `δ + ε_max`. -/
theorem shortAccepted_le (D Q L : ℕ)
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
    (periodHeld D Q L hP2 ι α ΩW ΩQ (d u) (r u x) (P u x) (τ u x) round S (hdev u x) (hinst u x) (hS u x)
      (o u x))

open Classical in
/-- **PoUS pooled over `U` windows, Chernoff form.** Under `PeriodHeld`'s premises, with each window's strategy under the
device model and the instantiation (`DeviceModel`, `Instantiates`), each window fixed before its capacity coins
(`FixedBefore`) and every window's certificate fitting the ring game's state bound: at least `M` windows accepted with no break while short (`ShortAccepted`) has probability `P` with
`P · z^M ≤ (1 + (z − 1)(δ + ε_max))^U` for every `z ≥ 1`. -/
theorem pooledShort (D Q L : ℕ)
    (hP2 : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (hfix : FixedBefore r P o) (round : ℕ) (S : ℝ)
    (τ : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Strategy ΩW ΩQ)
    (hdev : ∀ u x, DeviceModel (r u x) (P u x) (τ u x) D Q L round) (hinst : ∀ u x, Instantiates (P u x) (τ u x))
    (hS : ∀ u x, S * ((r u x).win .capacity).len + (d u).chip + ((∑ i, (d u).cap i * round : ℕ) : ℝ) ≤
      Pous.EndToEnd.S)
    (M : ℕ) {z : ℝ≥0∞} (hz : 1 ≤ z) :
    prCoin (fun x => M ≤ (Finset.univ.filter fun u => ShortAccepted r P o S u x).card) * z ^ M ≤
      (1 + (z - 1) * ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal)) ^ U :=
  Periods.prCoin_count_mul_pow_le _ (fixedBefore_adapted hfix S) hz
    (shortAccepted_le D Q L hP2 hfix round S τ hdev hinst hS) M

open Classical in
/-- **PoUS pooled over `U` windows, binomial form**: at least `M` windows accepted short with probability at most
`C(U, M) (δ + ε_max)^M`. -/
theorem pooledShort_choose (D Q L : ℕ)
    (hP2 : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (hfix : FixedBefore r P o) (round : ℕ) (S : ℝ)
    (τ : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Strategy ΩW ΩQ)
    (hdev : ∀ u x, DeviceModel (r u x) (P u x) (τ u x) D Q L round) (hinst : ∀ u x, Instantiates (P u x) (τ u x))
    (hS : ∀ u x, S * ((r u x).win .capacity).len + (d u).chip + ((∑ i, (d u).cap i * round : ℕ) : ℝ) ≤
      Pous.EndToEnd.S)
    (M : ℕ) :
    prCoin (fun x => M ≤ (Finset.univ.filter fun u => ShortAccepted r P o S u x).card) ≤
      U.choose M * ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal) ^ M :=
  Periods.prCoin_count_le_choose _ (fixedBefore_adapted hfix S)
    (shortAccepted_le D Q L hP2 hfix round S τ hdev hinst hS) M

/-- **PoUS pooled over `U` windows, at `δ'`** (`Pous.Period.PooledShortLog`): with `p = δ + ε_max`, at least `M`
windows accepted short has probability at most `δ'` once `M ≥ ((z − 1) U p + ln(1/δ')) / ln z` for some `z > 1`, so out
of `U` windows accepted with no break at least `U − M + 1` certify `S`. -/
theorem pooledShort_log : PooledShortLog := by
  intro D Q L hP2 U ι α ΩW ΩQ _ _ d r P o τ round S hfix hdev hinst hS δ' z M hz hδ' hM
  exact pr_le_of_prCoin_le _ hδ'.le
    (Periods.prCoin_count_le_of_log _ (fixedBefore_adapted hfix S) (add_nonneg δ_nonneg ENNReal.toReal_nonneg)
      (shortAccepted_le D Q L hP2 hfix round S τ hdev hinst hS) hz hδ' hM)

open Classical in
/-- **PoUS pooled over `U` windows, sure shortness**: every window accepted with no break while at least `M` of them
are short on every capacity coin (`Accounting.Run.CapacityShort`, fixed before the window's capacity coins) has
probability at most `(δ + ε_max)^M`. -/
theorem pooledSureShort (D Q L : ℕ)
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
      (shortAccepted_le D Q L hP2 hfix round S τ hdev hinst hS u x)) M
  · intro u x x' hx
    simp only [hfix.2.1 u x x' fun i hi => hx i hi.le, hfix.2.2 u x x' fun i hi => hx i hi.le, hx u le_rfl]
  · intro u x x' hx
    simp only [Accounting.Run.CapacityShort, hfix.1 u x x' hx, hfix.2.2 u x x' hx]

/-! ## The honest and nontrivial lemmas of `pooledShort_log` -/

theorem sidAudit_sid : sidAudit.sid.size = 32 := rfl

/-- **`pooledShort_log`'s premises hold at the honest pooled period** (`Pous.Period.PooledShortLog_honest`): no window
reads a capacity coin, `periodHeldHonest` gives the device premises at `sidAudit` and the certificate's fit, and
`10,080 · (1% + 2^-128) ≤ 146 · ln 2` gives the count condition at `z = 2`, `δ' = 2^-64`, `M = 210`. -/
theorem pooledShort_log_honest : PooledShortLog_honest := by
  obtain ⟨hdev, -, hS, -, -⟩ := periodHeldHonest
  refine ⟨⟨fun _ _ _ _ => rfl, fun _ _ _ _ => rfl, fun _ _ _ _ => rfl⟩, (hdev sidAudit sidAudit_sid).1,
    (hdev sidAudit sidAudit_sid).2, hS, ?_⟩
  have hlog : Real.log (1 / (2⁻¹ : ℝ) ^ 64) = 64 * Real.log 2 := by
    rw [one_div, inv_pow, inv_inv, Real.log_pow]
    norm_num
  have hl := Real.log_two_gt_d9
  rw [hlog, εMax_toReal, div_le_iff₀ (Real.log_pos one_lt_two)]
  norm_num [Params.δ]
  linarith

open Classical in
/-- **`pooledShort_log` is nontrivial** (`Pous.Period.PooledShortLog_nontrivial`): at `10,080` empty periods every
window is accepted short by `acceptAll` on every capacity coin, so the event has probability `1 > 2^-64`. -/
theorem pooledShort_log_nontrivial : PooledShortLog_nontrivial := by
  have hs : ∀ (x : Fin 10080 → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) (u : Fin 10080),
      acceptAll ((), (), x u) ∧ (emptyRun.held ((), (), x u) : ℝ) < HONEST_S * (emptyRun.win .capacity).len := by
    intro x u
    refine ⟨trivial, ?_⟩
    have he : emptyRun.held ((), (), x u) = K * 0 := held_periodRun 0 _ _
    rw [he]
    show ((K * 0 : ℕ) : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ)
    norm_num [HONEST_S, K]
  have h1 : Pouw.pr (fun x : Fin 10080 → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
      210 ≤ (Finset.univ.filter fun u => acceptAll ((), (), x u) ∧
        (emptyRun.held ((), (), x u) : ℝ) < HONEST_S * (emptyRun.win .capacity).len).card) = 1 :=
    Pouw.pr_eq_one fun x => by
      rw [Finset.filter_true_of_mem fun u _ => hs x u]
      simp
  intro h
  rw [h1] at h
  norm_num at h

end Pous.Period.SecurityProofs
