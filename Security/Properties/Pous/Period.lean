import Properties.Pous.EndToEnd
import Definitions.Accounting.PerRun

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

Both rest on `CoinCommitHides` and the conjecture `P2RingAuditSecure` at a budget `(D, Q, L)`, which
`Pous.EndToEnd.EndToEndRing` takes, and which is false at some budgets; `Pous.Period.PeriodHeldDeployed`
(`Properties.Pous.PeriodDeployed`) takes it only at the budget the deadline and the round floor give. Both rest also on
`RingDevice`, the per-run reading of `pous-ring-device`, `pous-ring-instantiation` and `prf/sha-512` (`Pous.EndToEnd`'s
header): no Lean theorem discharges it, since it is the step from the deployed device to the ring model. The setup
check's `AcceptsDecodes` and `PublishesRoot` (`Properties.Pous`) are proved against `SetupProtocol.protocol.check`
(`Proofs.Properties.Pous.Setup`), so neither is a premise.

The certificate is graded by the peak (`Accounting.Guarantees.CapacityOfStateBound`'s reading): the bit-ticks below
`S · len` bound the capacity account's memory at every tick of the window by `S · len`, so `S · len + S_chip + c` must
fit the ring game's state bound `S₀ = ⌊ρ·|C|⌋` (`Pous.EndToEnd.S`), with `c = ∑ cap_i · round` every side channel's
capacity over a round. At a window of one tick that is `S = S₀ − S_chip − c` bits; over a window of `T` ticks, about
`S₀/T` bits on average. A certificate of `S₀` on average over a longer window needs a game graded at each round
boundary (`Pous.Guarantees.ContinuousAudit`), whose verdict is the model's, not yet the deployed check's.
-/

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **What the verifier reads of a period's PoUS audit**, at the joint run's coins: the verifier's list of
memory-challenger keys and the session id the auditor designated, fixed for the period; the server's commitment and
salt, its `C*` (`store`), setup's public outputs, the memory challenger's table of row digests and the payload as the
ring game's bits (`W`), each fixed before the capacity coins are drawn, so a function of the compute and bandwidth audits'
coins alone; and the memory challenger's signed session and its device record (`MemoryChallenger.Properties.Device`), a
function of all the run's coins. -/
structure Audits (ΩW ΩQ : Type) where
  /-- the verifier's list of memory-challenger keys (never the run's record) -/
  trusted : List ByteArray
  /-- the session id the auditor designated -/
  sid : ByteArray
  /-- the server's commitment and salt -/
  commitment : ΩW × ΩQ → SetupProtocol.Commitment
  /-- `C*`, the blocks setup checks -/
  store : ΩW × ΩQ → Array (Array ByteArray)
  /-- setup's public outputs -/
  pub : ΩW × ΩQ → SetupProtocol.Public
  /-- the memory challenger's table of `C*`'s row digests -/
  digests : ΩW × ΩQ → ByteArray
  /-- the payload `C*` decodes to, as the ring game's bits -/
  W : ΩW × ΩQ → Bits sch.n
  /-- the memory challenger's signed session -/
  session : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → PousCheck.Run.Session
  /-- the memory challenger's record of its sessions -/
  device : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
    MemoryChallenger.Properties.Device

namespace Audits

variable {ΩW ΩQ : Type} (P : Audits ΩW ΩQ)

/-- The verifier's whole decision on the period's audit at the coins `ω` (`Pous.EndToEnd.Accepted`): setup accepts
`C*` and publishes `pub`, and the composed check accepts the session. -/
def Accepts (ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : Prop :=
  Accepted P.trusted (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1))
    (P.session ω) P.sid

/-- The period's own data holds a break at the coins `ω`: a SHA-512 collision among the strings the session and `C*`
hash or an LMS forgery under a key of the verifier's list (`RunBreak`), or a SHA-512 collision in the table's and `C*`'s
trees (`TableBreak`). -/
def Break (ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : Prop :=
  RunBreak P.trusted (P.session ω).signed (P.device ω) (P.store (ω.1, ω.2.1)) ∨
    TableBreak (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1))

/-- What an accepted period certifies outside a break (`Pous.EndToEnd.RunCertifies`'s conclusion): `C*` decodes to a
payload with the server's commitment, setup published its root, the session audits that setup at the deployed point
under the designated id and is signed under a key of the verifier's list, and that key's holder ran a session with the
accepted commitment in which every recorded answer is the challenged block of `C*`, on time on its own clock. -/
def Certified (ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : Prop :=
  DecodesToCommitment q (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) ∧
    Published q (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) (P.pub (ω.1, ω.2.1)) ∧
    AuditsSetup (P.pub (ω.1, ω.2.1)) (P.session ω).commitment ∧ (P.session ω).commitment.session = P.sid ∧
    (P.session ω).signed.publicKey ∈ P.trusted ∧
    ∃ run ∈ (P.device ω).sessions (P.session ω).signed.publicKey,
      run.commitment = (P.session ω).commitment ∧ AnsweredOnTime run (P.store (ω.1, ω.2.1))

end Audits

/-- `pous-ring-device`, `pous-ring-instantiation` and `prf/sha-512`, per run: **the device is a ring-model server on
the period's coins.** At each value `o` of the other audits' coins and each bound `B` with
`B + S_chip + ∑ cap_i · round` within the ring game's state bound `S₀`, there is one `(D, Q, L)`-bounded server with
`S₀` bits of state, fixed before the capacity coins (online coins), that passes the ring game's session (`Passes`, at
the designated id, bound to the published root, on the payload `W`) on every capacity coin where the capacity account
holds at most `B` bits of device memory at every tick of the capacity window and the period is certified (`Certified`).
That server holds the capacity account's memory cut to `B` bits, the whole chip and every side channel's capacity over a
round, and agrees with the device wherever the account stays within `B`; the window holds the whole session, every round
boundary. It is the deployed device's modeling, not a property of a Lean function, so no Lean theorem discharges it. -/
def RingDevice {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (D Q L round : ℕ) : Prop :=
  ∀ (o : ΩW × ΩQ) (B : ℕ), B + d.chip + ∑ i, d.cap i * round ≤ Pous.EndToEnd.S →
    ∃ A : Server Pous.EndToEnd.S D Q L,
      ∀ x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams,
        (∀ t ∈ (r.win .capacity).ticks, (r.schedule (o.1, o.2, x) t).mem (r.acct .capacity) ≤ B) →
          P.Certified (o.1, o.2, x) → Passes P.sid (P.pub o).root (P.W o) A x

/-- **PoUS's per-period guarantee, about the deployed check.** Given the coin commitments' hiding (`CoinCommitHides`)
and P2's security in the ring model at `(D, Q, L)` (`P2RingAuditSecure`): for every joint run on every device whose
capacity coins are the ring game's, every period's
audit `P` whose device is a ring-model server (`RingDevice`), every certificate `S` with
`S · len + S_chip + ∑ cap_i · round ≤ S₀` and every value of the other audits' coins, the verifier's whole decision
accepts the period with no break in its own data while the capacity account held fewer than `S · len` bit-ticks over the
capacity window with probability at most `δ + ε_max` (`1% + 2^-128`) over the capacity coins. -/
def PeriodHeld : Prop :=
  Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131) →
    ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
      ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
        (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
        (P : Audits ΩW ΩQ) (round : ℕ) (S : ℝ),
        RingDevice r P D Q L round →
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
  Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131) →
    ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
      ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] [Fintype ΩW] [Fintype ΩQ] (d : Accounting.Device ι)
        (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
        (P : Audits ΩW ΩQ) (round : ℕ) (S : ℝ),
        RingDevice r P D Q L round →
          S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
            (∀ ω, r.accepts .capacity ω → P.Accepts ω ∧ ¬ P.Break ω) →
              r.CapacitySoundPerRun S ((Params.δ : ℝ) + Params.εMax.toReal) ∧
                r.CapacitySound S ((Params.δ : ℝ) + Params.εMax.toReal)

end Pous.Period
