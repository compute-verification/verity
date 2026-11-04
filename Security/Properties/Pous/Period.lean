import Properties.Pous.EndToEndAt
import Definitions.Accounting.PerRun
import Definitions.Pous.P2.Clip

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

**Every prover strategy, one fixed extractor.** Both are stated for every prover strategy `τ` in the ring model
(`Strategy`): what survives setup is any function of the payload, `C` and the whole primitive, and each round's answer
is any program, with no bound on the strategy's state or its time. Its type reads the memory challenger's coins only as
the challenger releases them (the session commitment before the session, each index as the previous answer returns),
which is the challenger being `Faithful` and `Secret` (the per-party names), covered today by `trusted-host`. The ring
game's server is read from `τ` by one fixed extractor (`extract`): it packs `τ`'s state into `S₀` bits and cuts each
round's program to `(D, Q, L)`, so it is a ring-model server (`RingServer`) by construction, whatever `τ` does, never a
server the prover picks. Two named premises are about the period's strategy:

* `DeviceModel`, the device model (`pous-ring-device`, `pous-ring-speed`), falsifiable by measurement: on the period's
  coins, `τ`'s state at each round boundary is within the capacity account's memory, the chip and every side channel's
  capacity over a round, and when the memory challenger's signed record has every answer on time (`AuditsOf.OnTime`,
  `AnsweredOnTime`), each round's program stayed within `(D, Q, L)`. It says nothing of what `τ` answers.
* `Instantiates`, the instantiation (`pous-ring-instantiation`, `prf/sha-512`) in reduction form: on a certified coin
  (`AuditsOf.Certified`, which reads the challenger-signed `AnsweredOnTime`: every recorded answer the challenged block
  of `C*`), the challenger's draw is the ring game's and `τ`'s answers in the ring model are the challenged blocks of
  `C`. It is the step from the deployed device to the ring model, so no Lean theorem discharges it.

Both are stated at a point `pt` of the composed check and a block count `N` of the ring game, over `AuditsOf` and
`StrategyOf` at `N`'s scheme; `PeriodHeld` reads them at `Run.DEPLOYED` and `2^19` blocks (`Audits`, `Strategy`), and
`Pous.Period.PeriodHeldAt` at every point and `N` it covers. Under the two, the extractor's server passes the ring
game's session at `N` on every certified capacity coin where the account holds at most `B` bits at every tick,
`B + S_chip + c` within its state (`Pous.Period.SecurityProofs.extractPasses`, a lemma of both proofs).

Both rest on `CoinCommitHides` and the conjecture `P2RingAuditSecure` at a budget `(D, Q, L)`, which
`Pous.EndToEnd.EndToEndRing` takes, and which is false at some budgets; `Pous.Period.PeriodHeldDeployed`
(`Properties.Pous.PeriodDeployed`) takes it only at the budget the deadline and the round floor give. The setup
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

/-- **What the verifier reads of a period's PoUS audit**, at the joint run's coins, for a scheme `sch` in the ring
model: the verifier's list of memory-challenger keys and the session id the auditor designated, fixed for the period;
the server's commitment and salt, its `C*` (`store`), setup's public outputs, the memory challenger's table of row
digests and the payload as `sch`'s bits (`W`), each fixed before the capacity coins are drawn, so a function of the
compute and bandwidth audits' coins alone; and the memory challenger's signed session and its device record
(`MemoryChallenger.Properties.Challengers`), a function of all the run's coins. It is stated over the scheme, not at a
block count, because Lean's structure command unfolds `N * 16432` with `N` free (`Pous.EndToEnd.RingServer`). -/
structure AuditsOf {B w p : ℕ} [NeZero p] (sch : Scheme (ringRO B w p)) (ΩW ΩQ : Type) where
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
  session : ΩW × ΩQ × ((ringRO B w p).Ω × HmCoins × Streams) → PousCheck.Run.Session
  /-- the memory challenger's record of its sessions -/
  device : ΩW × ΩQ × ((ringRO B w p).Ω × HmCoins × Streams) → MemoryChallenger.Properties.Challengers

/-- `AuditsOf` at the deployed ring game, `2^19` blocks (`sch`). -/
abbrev Audits (ΩW ΩQ : Type) : Type := AuditsOf sch ΩW ΩQ

/-- `AuditsOf` at `N` blocks: the payload as `schAt N`'s bits. -/
abbrev AuditsAt (N : ℕ) [NeZero N] (ΩW ΩQ : Type) : Type := AuditsOf (schAt N) ΩW ΩQ

/-- **A prover strategy in the ring model for a scheme `sch`**, at the other audits' coins, with no bound on its state
or its time: what survives setup is any list of bits computed from the payload `W`, the public parameters, `C` and the
whole primitive (perfect isolation: nothing else survives); round `j`'s answer to index `i` is any program over the
ring, from the state the previous round left, shown the session commitment, returning the block and the next state. The
memory challenger's coins reach it only as the challenger releases them: the commitment before the session, each index
as the previous answer returns. Stated over the scheme for `AuditsOf`'s reason. -/
structure StrategyOf {B w p : ℕ} [NeZero p] (sch : Scheme (ringRO B w p)) (ΩW ΩQ : Type) where
  /-- what survives setup -/
  pre : ΩW × ΩQ → Bits sch.n → Bits sch.t → sch.Code → (ringRO B w p).Ω → List Bool
  /-- round `j`'s answer to index `i` from the state `σ`, shown the session commitment -/
  step : ΩW × ΩQ → MemoryChallenger.SessionCommitment → Bits sch.n → Bits sch.t → ℕ → List Bool → Fin sch.B →
    Prog (ringRO B w p) (Bits sch.ℓ × List Bool)

/-- `StrategyOf` at the deployed ring game, `2^19` blocks (`sch`). -/
abbrev Strategy (ΩW ΩQ : Type) : Type := StrategyOf sch ΩW ΩQ

/-- `StrategyOf` at `N` blocks (`schAt N`). -/
abbrev StrategyAt (N : ℕ) [NeZero N] (ΩW ΩQ : Type) : Type := StrategyOf (schAt N) ΩW ΩQ

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

end Audits

namespace AuditsOf

variable {B w p : ℕ} [NeZero p] {sch : Scheme (ringRO B w p)} {ΩW ΩQ : Type} (P : AuditsOf sch ΩW ΩQ)

/-- The verifier's whole decision on the period's audit at the point `pt` (`Pous.EndToEnd.AcceptedAt`): setup accepts
`C*` at `pt`'s parameters and publishes `pub`, and the composed check at `pt` accepts the session. -/
def Accepts (pt : Run.Point) (ω : ΩW × ΩQ × ((ringRO B w p).Ω × HmCoins × Streams)) : Prop :=
  AcceptedAt pt P.trusted (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) (P.pub (ω.1, ω.2.1))
    (P.digests (ω.1, ω.2.1)) (P.session ω) P.sid

/-- The period's own data holds a break at `pt`: `RunBreak`, or a SHA-512 collision in the table's and `C*`'s trees at
`pt`'s parameters (`TableBreakAt`). -/
def Break (pt : Run.Point) (ω : ΩW × ΩQ × ((ringRO B w p).Ω × HmCoins × Streams)) : Prop :=
  RunBreak P.trusted (P.session ω).signed (P.device ω) (P.store (ω.1, ω.2.1)) ∨
    TableBreakAt pt.params (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1))

/-- What an accepted period certifies at `pt` outside a break (`Pous.EndToEnd.RunCertifiesAt`'s conclusion): `C*`
decodes to a payload with the server's commitment, setup published its root, the session audits that setup at `pt`
under the designated id and is signed under a key of the verifier's list, and that key's holder ran a session with the
accepted commitment in which every recorded answer is the challenged block of `C*`, on time on its own clock. -/
def Certified (pt : Run.Point) (ω : ΩW × ΩQ × ((ringRO B w p).Ω × HmCoins × Streams)) : Prop :=
  DecodesToCommitment pt.params (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) ∧
    Published pt.params (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) (P.pub (ω.1, ω.2.1)) ∧
    AuditsSetupAt pt (P.pub (ω.1, ω.2.1)) (P.session ω).commitment ∧ (P.session ω).commitment.session = P.sid ∧
    (P.session ω).signed.publicKey ∈ P.trusted ∧
    ∃ run ∈ (P.device ω).sessions (P.session ω).signed.publicKey,
      run.commitment = (P.session ω).commitment ∧ AnsweredOnTime run (P.store (ω.1, ω.2.1))

/-- **The memory challenger's signed record has every answer on time at `pt`**: the session key's holder ran a session
with the signed commitment, at a limit within `pt`'s (`Δ + RTT` at `Run.DEPLOYED`), in which every recorded answer is
the challenged block of `C*` and arrived within the limit on the challenger's own clock (`AnsweredOnTime`). It reads
the challenger's record only, never what the prover reports. -/
def OnTime (pt : Run.Point) (ω : ΩW × ΩQ × ((ringRO B w p).Ω × HmCoins × Streams)) : Prop :=
  ∃ run ∈ (P.device ω).sessions (P.session ω).signed.publicKey,
    run.commitment = (P.session ω).commitment ∧ run.limitNs ≤ pt.limitNs ∧ AnsweredOnTime run (P.store (ω.1, ω.2.1))

end AuditsOf

/-- **The device's trace** of the period's session in the ring model at `N` blocks, for the strategy `τ` at the other
audits' coins `o`, the capacity coins `x` and the memory challenger's draw `(I, v)`: for each of the 105 rounds, the
state `τ` starts it from and the program it runs on index `I j`, shown the session commitment (`commitmentOfAt N`, at
the designated id, bound to the published root), the first round starting from what survived setup. -/
noncomputable def AuditsOf.trace {N : ℕ} [NeZero N] {ΩW ΩQ : Type} (P : AuditsOf (schAt N) ΩW ΩQ)
    (τ : StrategyAt N ΩW ΩQ) (o : ΩW × ΩQ) (x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (I : Fin 105 → Fin N) (v : ByteArray × ByteArray) :
    List (List Bool × Prog (ringRO N 16448 (2 ^ 16448 - 21065)) (Bits (schAt N).ℓ × List Bool)) :=
  seqRounds ((ringRO N 16448 (2 ^ 16448 - 21065)).answer x.1)
    (τ.step o (commitmentOfAt N P.sid (P.pub o).root v) (P.W o) ((schAt N).enc (P.W o) x.1 ()).1) 0 (List.ofFn I)
    (τ.pre o (P.W o) ((schAt N).enc (P.W o) x.1 ()).1 ((schAt N).enc (P.W o) x.1 ()).2 x.1)

/-- The blocks `τ` returns in its trace's rounds, on the period's primitive. -/
noncomputable def AuditsOf.answers {N : ℕ} [NeZero N] {ΩW ΩQ : Type} (P : AuditsOf (schAt N) ΩW ΩQ)
    (τ : StrategyAt N ΩW ΩQ) (o : ΩW × ΩQ) (x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (I : Fin 105 → Fin N) (v : ByteArray × ByteArray) : List (Bits (schAt N).ℓ) :=
  (P.trace τ o x I v).map fun e => (e.2.run ((ringRO N 16448 (2 ^ 16448 - 21065)).answer x.1)).1

/-- **The memory challenger's draw at `N` blocks, as the ring game reads it** (`PassesAt N`'s): the code's sampler
reads index `I j` from stream `j` at `N` blocks, and the code builds the coin tree over those indices under the session
id `sid`, publishing `v` (`coinViewAt N`). -/
def Drawn {N : ℕ} (sid : ByteArray) (x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (I : Fin 105 → Fin N) (v : ByteArray × ByteArray) : Prop :=
  (∀ j, FlockSoundness.Randomness.uniformOn N (bytesOf (x.2.2 j)) = some (I j).val) ∧
    coinViewAt N sid x.2.1 I = some v

/-- `pous-ring-device` and `pous-ring-speed`, per run, at the point `pt` and `N` blocks: **the device model**,
falsifiable by measurement. At every value `o` of the other audits' coins, every capacity coin `x` and every draw of the
memory challenger (`Drawn`), the strategy `τ` the device runs:

* computes each round's answer within `(D, Q, L)` (at most `D` rounds with a multiplication, `Q` key queries and `L`
  ring operations, on the period's primitive) whenever the memory challenger's signed record has every answer on time
  at `pt` (`AuditsOf.OnTime`): no allowed hardware does more within `pt`'s limit (`pous-ring-speed`, whose numbers
  `Pous.Period.PeriodHeldDeployed` fixes by measurement at `Δ + RTT`);
* holds at most `B + S_chip + ∑ cap_i · round` bits of state at every round boundary, setup's included, whenever the
  capacity account holds at most `B` bits of device memory at every tick of the capacity window, the window holding the
  whole session: the device's memory, the whole chip and every side channel's capacity over a round
  (`pous-ring-device`).

It bounds `τ`'s state and time only, never what `τ` answers. It is meaningful at `N = pt.params.blocks`, the block count
`pt`'s session challenges. -/
def DeviceModel (pt : Run.Point) {N : ℕ} [NeZero N] {ι : Type} [Fintype ι] {d : Accounting.Device ι}
    {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : AuditsAt N ΩW ΩQ) (τ : StrategyAt N ΩW ΩQ) (D Q L round : ℕ) : Prop :=
  ∀ (o : ΩW × ΩQ) (x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (I : Fin 105 → Fin N) (v : ByteArray × ByteArray), Drawn P.sid x I v →
    (P.OnTime pt (o.1, o.2, x) → ∀ e ∈ P.trace τ o x I v,
      e.2.mulDepth ((ringRO N 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ D ∧
        e.2.keyWork ((ringRO N 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ Q ∧
        e.2.ringWork ((ringRO N 16448 (2 ^ 16448 - 21065)).answer x.1) ≤ L) ∧
    ∀ B : ℕ, (∀ t ∈ (r.win .capacity).ticks, (r.schedule (o.1, o.2, x) t).mem (r.acct .capacity) ≤ B) →
      ∀ e ∈ P.trace τ o x I v, e.1.length ≤ B + d.chip + ∑ i, d.cap i * round

/-- `pous-ring-instantiation` and `prf/sha-512`, per run, at the point `pt` and `N` blocks, in reduction form: **in the
ring model, the period's strategy answers what its certified session recorded.** On every capacity coin where the
period is certified at `pt` (`AuditsOf.Certified`, which reads the memory challenger's signed `AnsweredOnTime`: every
recorded answer is the challenged block of `C*`), the challenger's draw is the ring game's (`Drawn`, its streams as
good as uniform bytes) and `τ`'s answers in the ring model are the challenged blocks of `C`. It carries the
challenger's record into the ring model and bounds no strategy's state or time; it is the step from the deployed device
to the ring model, so no Lean theorem discharges it. It is meaningful at `N = pt.params.blocks`. -/
def Instantiates (pt : Run.Point) {N : ℕ} [NeZero N] {ΩW ΩQ : Type} (P : AuditsAt N ΩW ΩQ)
    (τ : StrategyAt N ΩW ΩQ) : Prop :=
  ∀ (o : ΩW × ΩQ) (x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams),
    P.Certified pt (o.1, o.2, x) →
      ∃ I v, Drawn P.sid x I v ∧ P.answers τ o x I v = List.ofFn fun j => ((schAt N).enc (P.W o) x.1 ()).2 (I j)

/-- **The fixed extractor**: the ring game's server read from the strategy `τ` at the other audits' coins `o`, for
any scheme in the ring model. It packs `τ`'s state into `S` bits (`StateBits.pack`, read back by `StateBits.unpack`)
and cuts each round's program to `(D, Q, L)` (`Prog.clip`, which ends a round past the budget with the all-`false`
block), so it is an `S`-bit, `(D, Q, L)`-bounded server whatever `τ` is; on a trace whose states have fewer than `S`
bits and whose programs stay within the budget, it answers as `τ` does. -/
noncomputable def extract {B w p : ℕ} [NeZero p] {sch : Scheme (ringRO B w p)} {ΩW ΩQ : Type}
    (τ : StrategyOf sch ΩW ΩQ) (o : ΩW × ΩQ) (S D Q L : ℕ) : RingServer sch S D Q L where
  pre W pp C ω := StateBits.pack (τ.pre o W pp C ω)
  step sc W pp j σ i :=
    Prog.clip (fun out => (out.1, StateBits.pack out.2)) (fun _ => false, fun _ => false) D Q L
      (τ.step o sc W pp j (StateBits.unpack σ) i)
  bounded _ _ _ _ _ _ := Prog.clip_ringBounded _ _ _ _ _ _

/-- **PoUS's per-period guarantee, about the deployed check.** Given the coin commitments' hiding (`CoinCommitHides`)
and P2's security in the ring model at `(D, Q, L)` (`P2RingAuditSecure`): for every joint run on every device whose
capacity coins are the ring game's, every period's audit `P`, every prover strategy `τ` of its device under the device
model (`DeviceModel`) and the instantiation (`Instantiates`) at the deployed point, every certificate `S` with
`S · len + S_chip + ∑ cap_i · round ≤ S₀` and every value of the other audits' coins, the verifier's whole decision
accepts the period with no break in its own data while the capacity account held fewer than `S · len` bit-ticks over the
capacity window with probability at most `δ + ε_max` (`1% + 2^-128`) over the capacity coins. -/
def PeriodHeld : Prop :=
  Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131) →
    ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
      ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
        (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
        (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ),
        DeviceModel Run.DEPLOYED r P τ D Q L round → Instantiates Run.DEPLOYED P τ →
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
        (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ),
        DeviceModel Run.DEPLOYED r P τ D Q L round → Instantiates Run.DEPLOYED P τ →
          S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
            (∀ ω, r.accepts .capacity ω → P.Accepts ω ∧ ¬ P.Break ω) →
              r.CapacitySoundPerRun S ((Params.δ : ℝ) + Params.εMax.toReal) ∧
                r.CapacitySound S ((Params.δ : ℝ) + Params.εMax.toReal)

end Pous.Period
