import Properties.Pous.Period
import Properties.Pous.EndToEndAt

/-!
# PoUS over one period at any point and block count

`Pous.Period` states the per-period guarantee (`PeriodHeld`) at the deployed point: its audit is `Run.check`, at
`Run.DEPLOYED`, and its ring game is at `2^19` blocks. This module states it at every point `pt` of the composed check
(`Run.checkAt SCHEMES pt`, `Pous.EndToEnd.AcceptedAt`) and every block count `N` of the ring game
(`Pous.EndToEnd.PassesAt N`) that the widened per-round floor covers, `2^19 ≤ N ≤ 2^27`, and at Demo B's store,
`N = 1038·2^16`.

* `AuditsOf sch`, `AuditsAt N`: `Audits` with the payload as `sch`'s bits and the capacity coins the ring model's at `N`.
  `Accepts pt`, `Break pt` and `Certified pt` read the composed check, the table's break and the certificate at `pt`.
* `RingDeviceAt N pt`: `RingDevice` with `pt`'s certificate and the ring game at `N` (`ServerAt N`, `PassesAt N`, state
  bound `SAt N`): `pous-ring-device`, `pous-ring-instantiation` and `prf/sha-512` per run, read at `N`. It is meaningful
  at `N = pt.params.blocks`, the block count `pt`'s session challenges.
* `PeriodHeldAt`: at every such `pt` and `N`, from the coin commitments' hiding at `N` and `P2RingRoundSecureWide`, the
  verifier's whole decision at `pt` accepts a period with no break while the capacity account held fewer than `S · len`
  bit-ticks with probability at most `δ + ε_max`.
* `PeriodHeldDemo`: `PeriodHeldAt` at `N = 1038·2^16`, whose state bound is 1,060,008,348,186 bits.
-/

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **What the verifier reads of a period's PoUS audit, for a scheme `sch` in the ring model**: `Audits` with the
payload as `sch`'s bits and the capacity coins the ring model's. It is stated over the scheme, not at `N`, because
Lean's structure command unfolds `N * 16432` with `N` free (`Pous.EndToEnd.RingServer`). -/
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

/-- `Audits` at `N` blocks: the payload as `schAt N`'s bits. -/
abbrev AuditsAt (N : ℕ) [NeZero N] (ΩW ΩQ : Type) : Type := AuditsOf (schAt N) ΩW ΩQ

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

/-- What an accepted period certifies at `pt` outside a break (`Pous.EndToEnd.RunCertifiesAt`'s conclusion). -/
def Certified (pt : Run.Point) (ω : ΩW × ΩQ × ((ringRO B w p).Ω × HmCoins × Streams)) : Prop :=
  DecodesToCommitment pt.params (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) ∧
    Published pt.params (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) (P.pub (ω.1, ω.2.1)) ∧
    AuditsSetupAt pt (P.pub (ω.1, ω.2.1)) (P.session ω).commitment ∧ (P.session ω).commitment.session = P.sid ∧
    (P.session ω).signed.publicKey ∈ P.trusted ∧
    ∃ run ∈ (P.device ω).sessions (P.session ω).signed.publicKey,
      run.commitment = (P.session ω).commitment ∧ AnsweredOnTime run (P.store (ω.1, ω.2.1))

end AuditsOf

/-- `pous-ring-device`, `pous-ring-instantiation` and `prf/sha-512`, per run, **at the point `pt` and `N` blocks**:
`RingDevice` with `pt`'s certificate and the ring game at `N`. At each value `o` of the other audits' coins and each
bound `B` with `B + S_chip + ∑ cap_i · round` within the ring game's state bound `SAt N`, there is one `(D, Q, L)`-bounded
server at `N` with `SAt N` bits of state, fixed before the capacity coins, that passes the ring game's session at `N`
(`PassesAt N`, at the designated id, bound to the published root, on the payload `W`) on every capacity coin where the
capacity account holds at most `B` bits at every tick of the capacity window and the period is certified at `pt`. It
is the deployed device's modeling, not a property of a Lean function; it is meaningful at `N = pt.params.blocks`. -/
def RingDeviceAt (N : ℕ) [NeZero N] (pt : Run.Point) {ι : Type} [Fintype ι] {d : Accounting.Device ι}
    {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : AuditsAt N ΩW ΩQ) (D Q L round : ℕ) : Prop :=
  ∀ (o : ΩW × ΩQ) (B : ℕ), B + d.chip + ∑ i, d.cap i * round ≤ SAt N →
    ∃ A : ServerAt N (SAt N) D Q L,
      ∀ x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams,
        (∀ t ∈ (r.win .capacity).ticks, (r.schedule (o.1, o.2, x) t).mem (r.acct .capacity) ≤ B) →
          P.Certified pt (o.1, o.2, x) → PassesAt N P.sid (P.pub o).root (P.W o) A x

/-- **PoUS's per-period guarantee at any point and block count to `2^27`.** `PeriodHeld` with the composed check at
`pt`, the ring game at `N` and `P2RingRoundSecureWide` (with `(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130`) in place of
`P2RingAuditSecure`: given the coin commitments' hiding at `N`, for every joint run whose capacity coins are the ring
game's at `N`, every period's audit `P` whose device is a ring-model server at `N` for `pt` (`RingDeviceAt`), every
certificate `S` with `S · len + S_chip + ∑ cap_i · round ≤ SAt N` and every value of the other audits' coins, the
verifier's whole decision at `pt` accepts the period with no break while the capacity account held fewer than `S · len`
bit-ticks with probability at most `δ + ε_max`. Rests on the open conjecture `P2RingRoundSecureWide`, unchecked in `N`
above `2^25`. -/
def PeriodHeldAt : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    (1 - p) ^ 105 + 105 * ε₀ ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
      ∀ (pt : Run.Point) (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 27 →
        Pous.P2.Assumptions.CoinCommitHides (coinViewAt N) ((2⁻¹ : ℝ≥0∞) ^ 131) →
        ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
          (r : Accounting.Run d α ΩW ΩQ ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
          (P : AuditsAt N ΩW ΩQ) (round : ℕ) (S : ℝ),
          RingDeviceAt N pt r P D Q L round →
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
          (P : AuditsAt N ΩW ΩQ) (round : ℕ) (S : ℝ),
          RingDeviceAt N pt r P D Q L round →
            S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ 1060008348186 →
              ∀ o : ΩW × ΩQ,
                Pouw.pr (fun x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
                    P.Accepts pt (o.1, o.2, x) ∧ ¬ P.Break pt (o.1, o.2, x) ∧
                      (r.held (o.1, o.2, x) : ℝ) < S * (r.win .capacity).len) ≤
                  (Params.δ : ℝ) + Params.εMax.toReal

end Pous.Period
