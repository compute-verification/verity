module

public import Properties.Pous.EndToEnd
public import Properties.Assumptions.Pous
public import Definitions.Accounting.PerRun
public import Definitions.Pous.P2.Clip

/-!
# PoUS over one period: the deployed check's acceptance certifies the capacity account's space

One period of a joint run (`Accounting.Run`, the compute, bandwidth and capacity audits on one device) runs one PoUS
audit: setup publishes `C*`'s root, and the memory challenger runs one session against it. Its capacity coins are the ring
game's (`Pous.EndToEnd.Passes`'s `x`: the primitive `ω`, the memory challenger's hm96 coins and its streams), and what
the verifier reads of the period (`Audits`) is setup's inputs and outputs, fixed before those coins, and the session's
record, a function of all the run's coins.

* `PeriodHeld`, **about the deployed code**: at every value of the other audits' coins, the verifier's whole decision
  (`Pous.EndToEnd.Accepted`: `SetupProtocol.protocol.check` and `PousCheck.Run.check`) accepts, with no break in the
  run's own data (`Audits.Break`: a SHA-512 collision among the strings the run hashes, or an LMS forgery under a key
  of the verifier's list), while the capacity account held fewer than `S · len` bit-ticks over the capacity window,
  with probability at most `δ + ε_max` (`1% + 2^-128`) over the capacity coins.
* `PeriodCapacity`, the joint's reading: in a run whose capacity verdict is that acceptance outside a break, the capacity
  audit has its per-run guarantee (`Accounting.Run.CapacitySoundPerRun`), the input `Accounting.Guarantees.JointPerRun`
  takes, and its sure-bound form (`Accounting.Run.CapacitySound`), the one `Accounting.Guarantees.Joint` takes.
* In the shared coin game (`Verity.Game.prCoin`), `PeriodHeldCoins` over the hm96 coins and uniform streams, and
  `PeriodHeldSha512` over the memory challenger's SHA-512 derivation of its streams, at `δ + ε_max + ε_prf` under
  `Hash.Assumptions.Sha512Prf` (`Properties.Pous.PeriodGame`); both are proved from `PeriodHeld`.

**Every prover strategy, one fixed extractor.** Both are stated for every prover strategy `τ` in the ring model
(`Strategy`): what survives setup is any function of the payload, `C` and the whole primitive, and each round's answer
is any program, with no bound on the strategy's state or its time. Its type reads the memory challenger's coins only as
the challenger releases them (each index as the previous answer returns; the coin commitment after the last answer, so
no round sees it), which is the challenger being `Faithful` and `Secret` (the per-party names), covered today by
`trusted-host`. The ring game's server is read from `τ` by one fixed extractor (`extract`): it packs `τ`'s state into
`S₀` bits and cuts each round's program to `(D, Q, L)`, so it is a `Server` by construction, whatever `τ` does. Two
named premises about the period's strategy take the place of the earlier per-run premise, which handed the proof a
server the prover picked:

* `DeviceModel`, the device model (`pous-ring-device`, `pous-ring-speed`), falsifiable by measurement: on the period's
  coins, `τ`'s state at each round boundary is within the capacity account's memory, the chip and every side channel's
  capacity over a round, and when the memory challenger's signed record has every answer on time (`Audits.OnTime`,
  `AnsweredOnTime`), each round's program stayed within `(D, Q, L)`. It says nothing of what `τ` answers.
* `Instantiates`, the instantiation (`pous-ring-instantiation`, a named assumption declared in
  `Properties.Assumptions.Pous`) in reduction form: on a certified coin
  (`Certified`, which reads the challenger-signed `AnsweredOnTime`: every recorded answer the challenged block of `C*`),
  the challenger's sample is the ring game's and `τ`'s answers in the ring model are the challenged blocks of `C`. It is
  the step from the deployed device to the ring model, so no Lean theorem discharges it.

Under the two, the extractor's server passes the ring game's session on every certified capacity coin where the account
holds at most `B` bits at every tick, `B + S_chip + c < S₀` (`Pous.Period.SecurityProofs.extractPasses`, a lemma of
`PeriodHeld`'s proof).

Both rest on the conjecture `P2RingAuditSecure` at a budget `(D, Q, L)`, which `Pous.EndToEnd.EndToEndRing` takes,
and which is false at some budgets; `Pous.Period.PeriodHeldDeployed` (`Properties.Pous.PeriodDeployed`) takes it only
at the budget the deadline and the round floor give. The setup check's `AcceptsDecodes` and `PublishesRoot`
(`Properties.Pous`) are proved against `SetupProtocol.protocol.check` (`Proofs.Properties.Pous.Setup`), so neither is a
premise.

The certificate is graded by the peak (`Accounting.Guarantees.CapacityOfStateBound`'s reading): the bit-ticks below
`S · len` bound the capacity account's memory at every tick of the window by `S · len`, so `S · len + S_chip + c` must
fit the ring game's state bound `S₀ = ⌊ρ·|C|⌋` (`Pous.EndToEnd.S`), with `c = ∑ cap_i · round` every side channel's
capacity over a round. At a window of one tick that is `S = S₀ − S_chip − c` bits; over a window of `T` ticks, about
`S₀/T` bits on average. A certificate of `S₀` on average over a longer window needs a game graded at each round
boundary (`Pous.Guarantees.ContinuousAudit`), whose verdict is the model's, not yet the deployed check's.
-/

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
def DeviceModel {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (D Q L round : ℕ) : Prop :=
  ∀ (o : ΩW × ΩQ) (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (I : Fin 105 → Fin (2 ^ 19)) (v : ByteArray × ByteArray), Drawn P.sid x I v →
    (P.OnTime (o.1, o.2, x) → ∀ e ∈ P.trace τ o x I,
      e.2.mulDepth ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ D ∧
        e.2.keyWork ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ Q ∧
        e.2.ringWork ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ L) ∧
    ∀ B : ℕ, (∀ t ∈ (r.win .capacity).ticks, (r.schedule (o.1, o.2, x) t).mem (r.acct .capacity) ≤ B) →
      ∀ e ∈ P.trace τ o x I, e.1.length ≤ B + d.chip + ∑ i, d.cap i * round

/-- **The fixed extractor**: the ring game's server read from the strategy `τ` at the other audits' coins `o`. It packs
`τ`'s state into `S₀` bits (`StateBits.pack`, read back by `StateBits.unpack`) and cuts each round's program to
`(D, Q, L)` (`OracleAlgorithm.clip`, which ends a round past the budget with the all-`false` block), so it is an `S₀`-bit,
`(D, Q, L)`-bounded `Server` whatever `τ` is; on a trace whose states have fewer than `S₀` bits and whose programs stay
within the budget, it answers as `τ` does. -/
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
    ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
      (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
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
    ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] [Fintype ΩW] [Fintype ΩQ] (d : Accounting.Device ι)
      (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
      (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ),
      DeviceModel r P τ D Q L round → Instantiates P τ →
        S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
          (∀ ω, r.accepts .capacity ω → P.Accepts ω ∧ ¬ P.Break ω) →
            r.CapacitySoundPerRun S ((Params.δ : ℝ) + Params.εMax.toReal) ∧
              r.CapacitySound S ((Params.δ : ℝ) + Params.εMax.toReal)

end Pous.Period
