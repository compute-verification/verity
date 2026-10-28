import Properties.Pous.Period

/-!
# PoUS over one period at the deployed ring budget

`PeriodHeld` (`Pous.Period`) takes P2's ring-model conjecture `P2RingAuditSecure D Q L` at a free budget `(D, Q, L)`,
and the conjecture is false at some budgets, if `p` is prime: from `D ≥ 32,894` with `Q ≥ 2` and `L ≥ 65,789`, a
responder that recomputes the block's two roots (`PousP2Ring.tight_p16448`), and from `L ≥ 2^169` with `D ≥ 2` and
`Q ≥ 2`, a responder that keeps `501,819` blocks without their low `168` bits and recomputes them
(`Pous.Period.SecurityProofs.not_p2RingAuditSecure_trunc`; the same family refutes `k ≤ 88` from `L ≥ 2^30`,
`PousP2Ring.not_meetsOn_b19_k88`). This module fixes the budget to the one the deadline and the device model's round
floor give, outside both regions.

* **The deadline and the floor.** `LIMIT_NS = Δ + RTT = 854,000 ns`, the memory challenger's limit. The fastest
  modelled multiplication round is `ROUND_NS = 361.55 / 3.490 ≈ 103.6 ns`: the timing red team's 361.55 ns per
  16,448-bit squaring on one Zen 5 core, cut by the measured 8-core cooperation speedup of 3.490×. At that floor
  `DEADLINE_ROUNDS = ⌊LIMIT_NS / ROUND_NS⌋ = 8,243` sequential rounds fit the deadline. The PoUS node itself roots a
  block in 1.0931 s on its H200 and
  0.4602 s on one host core, about 26 and 61 of a block's 32,894 rounds in 854 µs.
* **The budget** (`pous-ring-speed`). `RING_D = 16,446` multiplication rounds an answer: the most at which `rootProg`
  cannot finish one root (`PousP2Ring.rootProg_cost`, `PousP2Ring.p16448_root_bits`), at least `DEADLINE_ROUNDS`, and
  `RING_D + 1` rounds take more than 1.995 × `LIMIT_NS` at the floor, so a device up to 1.995 times as fast as the
  floor still stays within it. `RING_Q = 2^20` key queries and `RING_L = 2^64` ring operations an answer, far above what any device
  does in 854 µs and far below `2^169`.
* `PeriodHeldDeployed`: `PeriodHeld` at `(RING_D, RING_Q, RING_L)`, so `P2RingAuditSecure` is taken only there.
* `PeriodHeldHonest`, the honest lemma of `PeriodHeld` and `PeriodHeldDeployed`: at an honest period (`honestRun`, a
  device that holds all of `C*` in its capacity account at every tick of the session's `105` rounds), `RingDevice`
  holds at the deployed budget for every audit, the certificate `HONEST_S = 63,000,000` bits is positive and fits the
  state bound beside the device's on-chip state and side channel, and the budget is the deadline's at the floor.
* `PeriodHeldNontrivial`, their nontrivial lemma: at a period whose capacity account holds nothing (`emptyRun`), a
  verifier that accepts everything fails the conclusion wherever the period's data holds no break, and the bound's
  event is false at the honest period.

`RingDevice` is the step from the deployed device to the ring model, and no theorem here discharges it: its device-model
part (the state between rounds within `B + S_chip + ∑ cap_i · round`, and the speed `(D, Q, L)` within `Δ + RTT`, which
this module's budget fixes) is falsifiable by measurement; its modeling part (the server computes through the ring's
operations only, `pous-ring-device`, `pous-ring-instantiation`, `prf/sha-512`) restricts the prover's strategy, and its
server is a witness per value of the other audits' coins, not one read from the device.
-/

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **The fastest modelled multiplication round, in ns**: the timing red team's 361.55 ns per 16,448-bit squaring on one
Zen 5 core cut by the measured 8-core cooperation speedup of 3.490×, about 103.6 ns. -/
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

/-- **PoUS's per-period guarantee at the deployed ring budget.** `PeriodHeld` with `P2RingAuditSecure` and
`RingDevice` at `(RING_D, RING_Q, RING_L) = (16,446, 2^20, 2^64)`, the budget `pous-ring-speed` derives from
`Δ + RTT` and the round floor (this module's header): given the coin commitments' hiding, for every joint run whose
capacity coins are the ring game's, every period's audit `P` whose device is a ring-model server at that budget, every
certificate `S` with `S · len + S_chip + ∑ cap_i · round ≤ S₀` and every value of the other audits' coins, the
verifier's whole decision accepts the period with no break while the capacity account held fewer than `S · len`
bit-ticks with probability at most `δ + ε_max`. -/
def PeriodHeldDeployed : Prop :=
  Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131) →
    Pous.P2.Assumptions.P2RingAuditSecure RING_D RING_Q RING_L →
      ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
        (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
        (P : Audits ΩW ΩQ) (round : ℕ) (S : ℝ),
        RingDevice r P RING_D RING_Q RING_L round →
          S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
            ∀ o : ΩW × ΩQ,
              Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
                  P.Accepts (o.1, o.2, x) ∧ ¬ P.Break (o.1, o.2, x) ∧
                    (r.held (o.1, o.2, x) : ℝ) < S * (r.win .capacity).len) ≤
                (Params.δ : ℝ) + Params.εMax.toReal

/-! ## The honest period -/

/-- **The honest device**, one H200 serving `C*`, with one tick a round of the session: `2^30` bits of on-chip state, at
least the H200's 992,061,184 bits of registers, shared memory with L1 and L2 at its data sheet
(`Verity.Accel.H200.size`); `C*`'s `2^19 · 16,448` bits of device memory; and one side channel, its PCIe Gen 5 ×16
link at its 64 GB/s line rate, 437,248,000 bits over a round of `854 µs` (the measured GPU-to-host copy is
57.2 GB/s). -/
def honestDevice : Accounting.Device Unit := ⟨2 ^ 30, FULL, fun _ => 0, 0, fun _ => 437248000⟩

/-- A period on the honest device whose capacity account holds `m ≤ FULL` bits at every tick, its window the session's
`105` rounds, one tick each. -/
def periodRun (m : ℕ) (hm : m ≤ FULL) :
    Accounting.Run honestDevice Unit Unit Unit ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) where
  acct _ := ()
  win _ := ⟨0, K⟩
  schedule _ _ := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => m⟩
  fits := by
    intro _ _
    simp [Accounting.Step.Fits, honestDevice, hm]
  accepts _ _ := True

/-- **The honest period**: the capacity account holds all of `C*`, `FULL` bits, at every tick. -/
def honestRun : Accounting.Run honestDevice Unit Unit Unit
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :=
  periodRun FULL le_rfl

/-- **The empty period**: the capacity account holds nothing. -/
def emptyRun : Accounting.Run honestDevice Unit Unit Unit
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :=
  periodRun 0 (Nat.zero_le _)

/-- The honest certificate, in bits on average over the session's `105` rounds: `63,000,000`, within
`(S₀ − S_chip − c) / 105` (`PeriodHeld`'s peak grading, `Pous.Period`'s header). -/
def HONEST_S : ℕ := 63000000

/-- **The honest lemma of `PeriodHeld` and `PeriodHeldDeployed`: their hypotheses hold at the honest period.** At
`honestRun` on `honestDevice`, with one tick a round: `RingDevice` holds at the deployed budget for every audit `P` (the
account holds more than any `B` within the state bound, so no capacity coin meets its antecedent); the certificate
`HONEST_S` is positive and fits `S₀` beside the device's on-chip state and its side channel's capacity over a round;
and the budget is the deadline's at the measured floor: `DEADLINE_ROUNDS ≤ RING_D`, with `RING_D + 1` rounds taking
at least 1.995 × `LIMIT_NS` at `ROUND_NS`. `CoinCommitHides` and `P2RingAuditSecure` are a named assumption and an open
conjecture, not facts of the instance. -/
def PeriodHeldHonest : Prop :=
  (∀ P : Audits Unit Unit, RingDevice honestRun P RING_D RING_Q RING_L 1) ∧
    0 < (HONEST_S : ℝ) * (honestRun.win .capacity).len ∧
    (HONEST_S : ℝ) * (honestRun.win .capacity).len + honestDevice.chip +
        ((∑ i, honestDevice.cap i * 1 : ℕ) : ℝ) ≤ Pous.EndToEnd.S ∧
    DEADLINE_ROUNDS ≤ RING_D ∧
    (1995 / 1000 : ℚ) * LIMIT_NS ≤ (RING_D + 1) * ROUND_NS

/-- **The nontrivial lemma of `PeriodHeld` and `PeriodHeldDeployed`.** At the empty period, a verifier that accepts
everything (`P.Accepts` replaced by `True`) fails their conclusion at the certificate `HONEST_S`, wherever the period's
data holds no break: the account held nothing, so the event has probability 1. And the conclusion's event is false at
the honest period, for every audit and coin, since its account held `105 · FULL` bit-ticks. A concrete audit with no
break needs SHA-512 on each of the table's `2^20` strings, so the first conjunct takes it as a hypothesis. -/
def PeriodHeldNontrivial : Prop :=
  (∀ (P : Audits Unit Unit) (o : Unit × Unit),
      (∀ x, ¬ P.Break (o.1, o.2, x)) →
        ¬ Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
            True ∧ ¬ P.Break (o.1, o.2, x) ∧
              (emptyRun.held (o.1, o.2, x) : ℝ) < HONEST_S * (emptyRun.win .capacity).len) ≤
          (Params.δ : ℝ) + Params.εMax.toReal) ∧
    ∀ (P : Audits Unit Unit) (o : Unit × Unit) (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams),
      ¬ (P.Accepts (o.1, o.2, x) ∧ ¬ P.Break (o.1, o.2, x) ∧
          (honestRun.held (o.1, o.2, x) : ℝ) < HONEST_S * (honestRun.win .capacity).len)

end Pous.Period
