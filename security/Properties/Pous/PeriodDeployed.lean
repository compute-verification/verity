module

public import Properties.Pous.Period
import Mathlib.Basic.ENNReal.Real

/-! PoUS over one period at the deployed ring budget (`PeriodHeldDeployed`), outside the budgets at which
`P2RingAuditSecure` is false. -/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **The fastest modelled multiplication round, in ns**: one 16,448-bit squaring at the fastest modelled speed,
eight cores cooperating. -/
def ROUND_NS : ℚ := 36155 / 349

/-- The sequential multiplication rounds that fit the memory challenger's limit at the floor:
`⌊854,000 / ROUND_NS⌋ = 8,243`. -/
def DEADLINE_ROUNDS : ℕ := ⌊(LIMIT_NS : ℚ) / ROUND_NS⌋₊

/-- The deployed multiplication rounds an answer (`pous-ring-speed`): one root's depth less one, `16,446`. -/
def RING_D : ℕ := 16446

/-- The deployed key queries an answer (`pous-ring-speed`). -/
def RING_Q : ℕ := 2 ^ 20

/-- The deployed ring operations an answer (`pous-ring-speed`). -/
def RING_L : ℕ := 2 ^ 64

/-- **PoUS's per-period guarantee at the deployed point and ring budget**: one audit of 105 rounds a period, on
`2^19` blocks of 16,448 bits, with probability at most `δ + ε_max` (`Params.δ`, `Params.εMax`; `1% + 2^-128`) for that
one audit, not pooled over a monitoring period's audits; a break is excluded from the event, not bounded. Another
point or block count is `PeriodHeldAt`'s. It is `PeriodHeld` with `P2RingAuditSecure` and
`DeviceModel` at `(RING_D, RING_Q, RING_L) = (16,446, 2^20, 2^64)`, the budget `pous-ring-speed` derives from
`Δ + RTT` and the round floor (this module's header): for every joint run whose capacity coins are the ring game's,
every period's audit `P`, every prover strategy `τ` of its device under the device model at that budget and the
instantiation (`Instantiates`), every certificate `S` with `S · len + S_chip + ∑ cap_i · round ≤ S₀` and every value
of the other audits' coins, the verifier's whole decision accepts the period with no break while the capacity account
held fewer than `S · len` bit-ticks with probability at most `δ + ε_max`. -/
def PeriodHeldDeployed : Prop :=
  Pous.P2.Assumptions.P2RingAuditSecure RING_D RING_Q RING_L →
    ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : ResourceAccounting.Device ι)
      (r : ResourceAccounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
      (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ),
      DeviceModel r P τ RING_D RING_Q RING_L round → Instantiates P τ →
        S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
          ∀ o : ΩW × ΩQ,
            Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
                P.Accepts (o.1, o.2, x) ∧ ¬ P.Break (o.1, o.2, x) ∧
                  (r.held (o.1, o.2, x) : ℝ) < S * (r.win .capacity).len) ≤
              (Params.δ : ℝ) + Params.εMax.toReal

/-- **The honest device**, one H200 serving `C*`, with one tick a round of the session: `2^30` bits of on-chip state, at
least the H200's 992,061,184 bits of registers, shared memory with L1 and L2 at its data sheet
(`store` in `hardware/cards/h200-sxm-141gb.toml`); `C*`'s `2^19 · 16,448` bits of device memory; and one side
channel, its PCIe Gen 5 ×16 link at its 64 GB/s line rate, 437,248,000 bits over a round of `854 µs` (the measured
GPU-to-host copy is 57.2 GB/s). -/
def honestDevice : ResourceAccounting.Device Unit := ⟨2 ^ 30, FULL, fun _ => 0, 0, fun _ => 437248000⟩

/-- A period on the honest device whose capacity account holds `m ≤ FULL` bits at every tick, its window the session's
`105` rounds, one tick each. -/
def periodRun (m : ℕ) (hm : m ≤ FULL) :
    ResourceAccounting.Run honestDevice Unit Unit Unit ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) where
  acct _ := ()
  win _ := ⟨0, K⟩
  schedule _ _ := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => m⟩
  fits := by
    intro _ _
    simp [ResourceAccounting.Step.Fits, honestDevice, hm]
  accepts _ _ := True

/-- **The honest period**: the capacity account holds all of `C*`, `FULL` bits, at every tick. -/
def honestRun : ResourceAccounting.Run honestDevice Unit Unit Unit
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :=
  periodRun FULL le_rfl

/-- **The empty period**: the capacity account holds nothing. -/
def emptyRun : ResourceAccounting.Run honestDevice Unit Unit Unit
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :=
  periodRun 0 (Nat.zero_le _)

/-- **The honest strategy**: it keeps all of `C`, `FULL` bits, as what survives setup, and answers each index by
looking its block up, with no query, keeping its state. -/
noncomputable def honestStrategy : Strategy Unit Unit where
  pre _ _ _ C _ := List.ofFn (codeEquiv sch.B sch.ℓ C)
  step _ _ _ _ σ i := .ret ((codeEquiv sch.B sch.ℓ).symm (fun k => σ.getD k.val false) i, σ)

/-- The honest certificate, in bits on average over the session's `105` rounds: `63,000,000`, within
`(S₀ − S_chip − c) / 105` (`PeriodHeld`'s peak grading, `Pous.Period`'s header). -/
def HONEST_S : ℕ := 63000000

end Pous.Period
