import Properties.Pous.PeriodDeployed
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# PoUS pooled over a monitoring period's windows: how many can be accepted short

`PeriodHeld` bounds one window: at every value of the other audits' coins, the verifier's whole decision accepts the
window's PoUS audit with no break in its data while its capacity account held fewer than `S · len` bit-ticks with
probability at most `p = δ + ε_max` (`1% + 2^-128`) over the window's capacity coins. A monitoring period runs `U`
windows whose capacity coins are drawn live, window after window: each window's joint run, its audit and the other
audits' coins may depend on the earlier windows' capacity coins, never on its own (`FixedBefore`), and its strategy is
any one under the device model, so `PeriodHeld` bounds each window whatever came before, and the windows pool.

* `ShortAccepted`: window `u` is accepted with no break while short, `PeriodHeld`'s event, at the period's capacity
  coins.
* `PooledShortLog`, **about the deployed check**: under `PeriodHeld`'s premises in every window, at least `M` of the `U`
  windows are accepted with no break while short with probability at most `δ'`, once
  `M ≥ ((z − 1) · U · p + ln(1/δ')) / ln z` for some `z > 1`. Outside that event at most `M − 1` windows are accepted
  short, so when every window is accepted with no break, at least `U − M + 1` of them held `S · len` bit-ticks.
* `PooledShortLog_honest`: its premises hold at an honest pooled period, and its count condition at one `δ'`.
* `PooledShortLog_nontrivial`: a verifier that accepts every window (`acceptAll`) fails its conclusion.

`FixedBefore` is a named assumption (`pous-ring-device`, `trusted-host`): the memory challenger releases each
window's capacity coins only after that window's commitment is in. `δ'` is `δ_PoUS`, the failure probability of PoUS
over a monitoring period, a public deployment parameter that no component publishes.
-/

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- `pous-ring-device` and `trusted-host`, per monitoring period: **each window is fixed before its capacity coins**. At
the period's capacity coins `x`, window `u`'s joint run `r u x`, its audit `P u x` and the other audits' coins `o u x`
read only the earlier windows' coins `x i`, `i < u`.

The memory challenger enforces it. Window `u`'s capacity coins are drawn on the auditor's device, which runs its stated
code on a host the prover doesn't control (`trusted-host`), and are released only after the window's commitment is in:
each index after the previous answer, the coin root only after the last answer and the signature (`pous-ring-device`:
the coins drawn online, after the server's state is fixed). So nothing fixed for window `u` reads its own coins, while
the earlier windows' coins are out by then and may be read. -/
def FixedBefore {U : ℕ} {ι α ΩW ΩQ : Type} [Fintype α] {d : Fin U → Accounting.Device ι}
    (r : (u : Fin U) → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
      Accounting.Run (d u) α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Audits ΩW ΩQ)
    (o : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → ΩW × ΩQ) : Prop :=
  (∀ u x x', (∀ i, i < u → x i = x' i) → r u x = r u x') ∧
    (∀ u x x', (∀ i, i < u → x i = x' i) → P u x = P u x') ∧
    ∀ u x x', (∀ i, i < u → x i = x' i) → o u x = o u x'

/-- Window `u` is accepted with no break in its data while its capacity account held fewer than `S · len` bit-ticks
(`PeriodHeld`'s event), at the period's capacity coins `x`. -/
def ShortAccepted {U : ℕ} {ι α ΩW ΩQ : Type} [Fintype α] {d : Fin U → Accounting.Device ι}
    (r : (u : Fin U) → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
      Accounting.Run (d u) α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Audits ΩW ΩQ)
    (o : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → ΩW × ΩQ) (S : ℝ)
    (u : Fin U) (x : Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) : Prop :=
  (P u x).Accepts ((o u x).1, (o u x).2, x u) ∧ ¬ (P u x).Break ((o u x).1, (o u x).2, x u) ∧
    ((r u x).held ((o u x).1, (o u x).2, x u) : ℝ) < S * ((r u x).win .capacity).len

open Classical in
/-- **PoUS pooled over a monitoring period, about the deployed check.** Given P2's security in the ring model at
`(D, Q, L)` (`P2RingAuditSecure`): for every monitoring period of `U` windows, each a joint run on a device whose
capacity coins are the ring game's, with its audit and the other audits' coins, each fixed before its window's capacity
coins (`FixedBefore`), and a prover strategy, every window's strategy under the device model (`DeviceModel`) and the
instantiation (`Instantiates`), and every certificate `S` with `S · len + S_chip + ∑ cap_i · round ≤ S₀` in every
window: at least `M` windows accepted by the verifier's whole decision with no break while short (`ShortAccepted`) has
probability at most `δ'` over the period's capacity coins, once `M ≥ ((z − 1) · U · p + ln(1/δ')) / ln z` for some
`z > 1`, `p = δ + ε_max` the per-window bound of `PeriodHeld`. So outside an event of probability `δ'`, when every
window is accepted with no break, at least `U − M + 1` of the `U` windows certify `S`.

`δ'` is `δ_PoUS`, the failure probability of PoUS over the monitoring period. It is a public deployment parameter and a
free parameter of this statement, which holds at every `δ' > 0` (a smaller `δ'` asks a larger `M`): no component
publishes it. -/
def PooledShortLog : Prop :=
  ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (U : ℕ) (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Fin U → Accounting.Device ι)
      (r : (u : Fin U) → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
        Accounting.Run (d u) α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
      (P : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Audits ΩW ΩQ)
      (o : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → ΩW × ΩQ)
      (τ : Fin U → (Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Strategy ΩW ΩQ)
      (round : ℕ) (S : ℝ),
      FixedBefore r P o →
        (∀ u x, DeviceModel (r u x) (P u x) (τ u x) D Q L round) → (∀ u x, Instantiates (P u x) (τ u x)) →
          (∀ u x, S * ((r u x).win .capacity).len + (d u).chip + ((∑ i, (d u).cap i * round : ℕ) : ℝ) ≤
            Pous.EndToEnd.S) →
            ∀ (δ' z : ℝ) (M : ℕ), 1 < z → 0 < δ' →
              ((z - 1) * U * ((Params.δ : ℝ) + Params.εMax.toReal) + Real.log (1 / δ')) / Real.log z ≤ M →
                Pouw.pr (fun x : Fin U → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
                    M ≤ (Finset.univ.filter fun u => ShortAccepted r P o S u x).card) ≤ δ'

/-! ## The honest and nontrivial lemmas -/

/-- **An audit under a 32-byte session id**, as the memory challenger's ids are, with nothing else in it: the audit the
honest lemma's windows run, since the device premises hold at every such audit. -/
def sidAudit : Audits Unit Unit where
  trusted := []
  sid := ⟨Array.replicate 32 0⟩
  commitment _ := ⟨ByteArray.empty, ByteArray.empty⟩
  store _ := #[]
  pub _ := ⟨0, ByteArray.empty, ByteArray.empty, ByteArray.empty⟩
  digests _ := ByteArray.empty
  W _ _ := false
  session _ := ⟨default, default, #[]⟩
  device _ := ⟨fun _ => []⟩

/-- **The verifier that accepts every window**, in place of the deployed decision and its no-break condition: the
nontrivial lemma's verdict. -/
def acceptAll (_ : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : Prop := True

/-- **The honest lemma of `PooledShortLog`: its premises hold at an honest pooled period.** `10,080` windows, each
the honest period (`honestRun` on `honestDevice`, one tick a round) run by the honest strategy (`honestStrategy`, which
keeps all of `C` and answers by lookup) against `sidAudit`, the other audits' coins fixed: no window reads any capacity
coin, so each is fixed before its own (`FixedBefore`); the device model holds at the deployed budget and the
instantiation holds (`DeviceModel`, `Instantiates`); the certificate `HONEST_S` fits the state bound beside the
device's on-chip state and side channel; and the count `M = 210` meets the condition at `z = 2` and `δ' = 2^-64`, one
value of the free parameter: `(10,080 · (1% + 2^-128) + 64 · ln 2) / ln 2 ≈ 209.4`. `P2RingAuditSecure` is an open
conjecture, not a fact of the instance. -/
def PooledShortLog_honest : Prop :=
  FixedBefore (U := 10080) (d := fun _ => honestDevice) (fun _ _ => honestRun) (fun _ _ => sidAudit)
      (fun _ _ => ((), ())) ∧
    DeviceModel honestRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy ∧
    (HONEST_S : ℝ) * (honestRun.win .capacity).len + honestDevice.chip +
        ((∑ i, honestDevice.cap i * 1 : ℕ) : ℝ) ≤ Pous.EndToEnd.S ∧
    ((2 - 1) * ((10080 : ℕ) : ℝ) * ((Params.δ : ℝ) + Params.εMax.toReal) + Real.log (1 / (2⁻¹ : ℝ) ^ 64)) /
        Real.log 2 ≤ ((210 : ℕ) : ℝ)

open Classical in
/-- **The nontrivial lemma of `PooledShortLog`.** At `10,080` empty periods (`emptyRun`: the capacity account holds
nothing), a verifier that accepts every window (`acceptAll`) fails the conclusion at the honest lemma's `M = 210`,
`δ' = 2^-64` and certificate `HONEST_S`: every window is accepted short on every capacity coin, so the event has
probability `1`. -/
def PooledShortLog_nontrivial : Prop :=
  ¬ Pouw.pr (fun x : Fin 10080 → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
      210 ≤ (Finset.univ.filter fun u => acceptAll ((), (), x u) ∧
        (emptyRun.held ((), (), x u) : ℝ) < HONEST_S * (emptyRun.win .capacity).len).card) ≤ (2⁻¹ : ℝ) ^ 64

end Pous.Period
