module

public import Properties.Pous.EndToEnd
public import Properties.Assumptions.Pous
public import Definitions.Pouw.PerRun
public import Definitions.Pous.P2.Clip

/-! PoUS over one period: the deployed check's acceptance certifies the capacity account's space (`PeriodHeld`), for
every prover strategy under the device model and the instantiation. -/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- `pous-ring-device` and `pous-ring-speed`, per run: **the device model**, falsifiable by measurement. At every value
`o` of the other audits' coins, every capacity coin `x` and every sample of the memory challenger (`Drawn`), the
strategy `τ` the device runs:

* computes each round's answer within `(D, Q, L)` (at most `D` rounds with a multiplication, `Q` key queries and `L`
  ring operations, on the period's primitive) whenever the memory challenger's signed record has every answer on time
  (`Audits.OnTime`): no allowed hardware does more within `Δ + RTT` (`pous-ring-speed`, whose numbers
  `Pous.Period.PeriodHeldDeployed` fixes by measurement);
* holds at most `B + S_chip + ∑ cap_i · round` bits of state at every round boundary, setup's included, whenever the
  capacity account holds at most `B` bits of device memory at every tick of the capacity window, the window holding the
  whole session: the device's memory, the whole chip and every side channel's capacity over a round
  (`pous-ring-device`).

It bounds `τ`'s state and time only, never what `τ` answers. -/
def DeviceModel {ι : Type} [Fintype ι] {d : ResourceAccounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : ResourceAccounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (D Q L round : ℕ) : Prop :=
  ∀ (o : ΩW × ΩQ) (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (I : Fin 105 → Fin (2 ^ 19)) (v : ByteArray × ByteArray), Drawn P.sid x I v →
    (P.OnTime (o.1, o.2, x) → ∀ e ∈ P.trace τ o x I,
      e.2.mulDepth ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ D ∧
        e.2.keyWork ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ Q ∧
        e.2.ringWork ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ L) ∧
    ∀ B : ℕ, (∀ t ∈ (r.win .capacity).ticks, (r.schedule (o.1, o.2, x) t).mem (r.acct .capacity) ≤ B) →
      ∀ e ∈ P.trace τ o x I, e.1.length ≤ B + d.chip + ∑ i, d.cap i * round

/-- **The fixed extractor**: the ring game's server read from `τ` at the other audits' coins `o`, an `S₀`-bit,
`(D, Q, L)`-bounded `Server` whatever `τ` is; on a trace within those bounds it answers as `τ` does. -/
noncomputable def extract {ΩW ΩQ : Type} (τ : Strategy ΩW ΩQ) (o : ΩW × ΩQ) (D Q L : ℕ) :
    Server Pous.EndToEnd.S D Q L where
  pre W pp C ω := StateBits.pack (τ.pre o W pp C ω)
  step W pp j σ i :=
    OracleAlgorithm.clip (fun out => (out.1, StateBits.pack out.2)) (fun _ => false, fun _ => false) D Q L
      (τ.step o W pp j (StateBits.unpack σ) i)
  bounded _ _ _ _ _ := OracleAlgorithm.clip_ringBounded _ _ _ _ _ _

/-- **PoUS's per-period guarantee, about the deployed check.** Given P2's security in the ring model at `(D, Q, L)`
(`P2RingAuditSecure`): for every joint run on every device whose capacity coins are the ring game's, every period's
audit `P`, every prover strategy `τ` of its device under the device
model (`DeviceModel`) and the instantiation (`Instantiates`), every certificate `S` with
`S · len + S_chip + ∑ cap_i · round ≤ S₀` and every value of the other audits' coins, the verifier's whole decision
accepts the period with no break in its own data while the capacity account held fewer than `S · len` bit-ticks over the
capacity window with probability at most `δ + ε_max` (`1% + 2^-128`) over the capacity coins. -/
def PeriodHeld : Prop :=
  ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : ResourceAccounting.Device ι)
      (r : ResourceAccounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
      (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ),
      DeviceModel r P τ D Q L round → Instantiates P τ →
        S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
          ∀ o : ΩW × ΩQ,
            Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
                P.Accepts (o.1, o.2, x) ∧ ¬ P.Break (o.1, o.2, x) ∧
                  (r.held (o.1, o.2, x) : ℝ) < S * (r.win .capacity).len) ≤
              (Params.δ : ℝ) + Params.εMax.toReal

/-- **PoUS's per-period guarantee is the joint's capacity guarantee.** Under `PeriodHeld`'s premises, in a joint run
whose capacity verdict is the verifier's whole decision outside a break (`r.accepts .capacity ω → P.Accepts ω ∧
¬ P.Break ω`, so the joint's acceptance reads "every audit accepts and the period's data holds no break"), the capacity
audit has its per-run guarantee (`CapacitySoundPerRun`, `JointPerRun`'s input) and its sure-bound form
(`CapacitySound`, `Joint`'s) at certificate `S` and error `δ + ε_max`. -/
def PeriodCapacity : Prop :=
  ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] [Fintype ΩW] [Fintype ΩQ] (d : ResourceAccounting.Device ι)
      (r : ResourceAccounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
      (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ),
      DeviceModel r P τ D Q L round → Instantiates P τ →
        S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
          (∀ ω, r.accepts .capacity ω → P.Accepts ω ∧ ¬ P.Break ω) →
            r.CapacitySoundPerRun S ((Params.δ : ℝ) + Params.εMax.toReal) ∧
              r.CapacitySound S ((Params.δ : ℝ) + Params.εMax.toReal)

end Pous.Period
