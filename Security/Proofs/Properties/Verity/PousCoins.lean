import Proofs.Properties.Verity.Subprotocols
import Proofs.Pous.PeriodGame

/-!
# PoUS's target from its period bound at the deployed challenger's SHA-512 streams (conditional)

`Accounting.RealizesSpaceOfPeriodHeld` ties a period run's PoUS coordinate to a joint run's capacity coins by an
equivalence. At the ring game's coins (`ringΩ × HmCoins × Streams`, `SpaceOfPeriodHeld`'s) no coordinate of the game's
coins can be one: `|ringΩ|` has the odd factor `p = 2^16448 − 21065`, and the game's coins number a power of two.
`Pous.Period.PeriodHeldSha512` removes the primitive: where the session, the device record and the schedule don't read
it, the period's event over the memory challenger's 64-byte secret source and its hm96 coins, its streams derived from
the source by SHA-512 (`challengerStreams`), has probability at most `δ + ε_max + ε(t, 1) + ε(t, 6720)` at every
primitive, under `Hash.Assumptions.Sha512Prf` at the derivation's two tests. So the realization here is over
`Bytes 64 × HmCoins`, bytes, which a round of the game's coins can be:

* `Accounting.Run.withRing`: a joint run over `HmCoins × Streams`, read at coins that also carry the primitive, which it
  ignores; `Accounting.Run.ofSource`: the same run read at the source and the hm96 coins, its streams the source's
  derivation; `PousRingFree`: the audit's session and device record don't read the primitive.
* `spaceOfPeriodHeldSha512`: `PeriodRun.Sound` for `.pous` at `δ + ε_max + ε(t, 1) + ε(t, 6720)` at a period realizing
  its PoUS audit over the source and the hm96 coins, from `Pous.Period.SecurityProofs.periodHeldSha512`, under its
  premises (`CoinCommitHides`, `P2RingAuditSecure`, `DeviceModel`, `Instantiates`, the certificate's fit and the two
  `Sha512Prf` tests, `PrfTests`), read at any one primitive `ω₀`.
* `pous_atCode_sha512`, `Vocab.pousSound_of_sha512`: through `Vocab.Coupling` (`pousSound_of_coupling`), `PousSound k`
  conditional on a coupling at each `θ F c σ` whose run realizes its PoUS audit over the source and the hm96 coins
  (`PousCoins`), and on its break being answered for by the finder and forger terms where the finder meets `FinderCR`
  (`Coupling.BreakBound`), with PoUS's published `δ` at least the bound.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Subprotocol

section Space

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period

set_option exponentiation.threshold 20000

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {ΩW ΩQ : Type}

/-- **A joint run over the memory challenger's coins, read in the ring model**: `run`, whose capacity coins are the hm96
coins and the streams, at coins that also carry the primitive, which it doesn't read. -/
def _root_.Accounting.Run.withRing (run : Run d α ΩW ΩQ (HmCoins × Streams)) :
    Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) where
  acct := run.acct
  win := run.win
  schedule ω := run.schedule (ω.1, ω.2.1, ω.2.2.2)
  fits ω := run.fits _
  accepts a ω := run.accepts a (ω.1, ω.2.1, ω.2.2.2)

/-- **A joint run over the memory challenger's coins, read at its source**: `run` at the challenger's 64-byte secret
source and its hm96 coins, its streams the source's SHA-512 derivation for the session `sid` at `2^19` blocks
(`challengerStreams`). -/
def _root_.Accounting.Run.ofSource (run : Run d α ΩW ΩQ (HmCoins × Streams)) (sid : ByteArray) :
    Run d α ΩW ΩQ (Bytes 64 × HmCoins) where
  acct := run.acct
  win := run.win
  schedule x := run.schedule (x.1, x.2.1, (x.2.2.2, challengerStreams (2 ^ 19) sid x.2.2.1))
  fits x := run.fits _
  accepts a x := run.accepts a (x.1, x.2.1, (x.2.2.2, challengerStreams (2 ^ 19) sid x.2.2.1))

/-- **The audit doesn't read the primitive**: at every value of the other audits' coins and the challenger's coins, the
memory challenger's signed session and its device record are the same at every primitive. -/
abbrev PousRingFree (P : Audits ΩW ΩQ) : Prop :=
  ∀ (o : ΩW × ΩQ) (ω ω' : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (y : HmCoins × Streams),
    P.session (o.1, o.2, (ω, y)) = P.session (o.1, o.2, (ω', y)) ∧
      P.device (o.1, o.2, (ω, y)) = P.device (o.1, o.2, (ω', y))

/-- **The verifier's whole decision outside a break, at the challenger's source and hm96 coins**, its streams the
source's derivation for the designated session, read at the primitive `ω₀`. -/
abbrev sourceVerdict (P : Audits ΩW ΩQ) (ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω)
    (x : ΩW × ΩQ × (Bytes 64 × HmCoins)) : Prop :=
  P.Accepts (x.1, x.2.1, (ω₀, x.2.2.2, challengerStreams (2 ^ 19) P.sid x.2.2.1)) ∧
    ¬ P.Break (x.1, x.2.1, (ω₀, x.2.2.2, challengerStreams (2 ^ 19) P.sid x.2.2.1))

/-- **`Sha512Prf` at the challenger's two tests**, at every value of the other audits' coins: the coin key (`derive`'s
one input after the source) and the streams' blocks (105 · 64 inputs after the key), each the period's event at the
primitive `ω₀`, at advantage `ε` and the tests' time `t` (`PeriodHeldSha512`'s premises). -/
abbrev PrfTests (run : Run d α ΩW ΩQ (HmCoins × Streams)) (P : Audits ΩW ΩQ) (S : ℝ)
    (ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (ε : ℕ → ℕ → ℝ≥0∞) (t : ℕ) : Prop :=
  ∀ o : ΩW × ΩQ,
    Hash.Assumptions.Sha512Prf sha512Digest ε derivePre 64 (fun _ : Unit => deriveTail (2 ^ 19) P.sid)
        (fun (c : HmCoins) k => AcceptedShort run.withRing P S (o.1, o.2, (ω₀, c, streamsOf (keyOf (k ()))))) t ∧
      Hash.Assumptions.Sha512Prf sha512Digest ε FlockSoundness.Randomness.STREAM.data.toList 32 streamInputs
        (fun (c : HmCoins) U => AcceptedShort run.withRing P S (o.1, o.2, (ω₀, c, streamsOfBlocks U))) t

theorem εMax_ne_top : Params.εMax ≠ ⊤ := by
  unfold Params.εMax
  exact ENNReal.pow_ne_top (ENNReal.inv_ne_top.2 two_ne_zero)

/-- The bound at the deployed challenger's streams is finite where `Sha512Prf`'s advantage is. -/
theorem sha512Bound_ne_top {ε : ℕ → ℕ → ℝ≥0∞} (hε : ∀ t n, ε t n ≠ ⊤) (t : ℕ) :
    ENNReal.ofReal Params.δ + Params.εMax + ε t 1 + ε t (105 * 64) ≠ ⊤ :=
  ENNReal.add_ne_top.2 ⟨ENNReal.add_ne_top.2 ⟨ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, εMax_ne_top⟩, hε _ _⟩,
    hε _ _⟩

/-- **PoUS's per-period guarantee at a period that realizes it over the challenger's source and hm96 coins**:
`PeriodRun.Sound` for `.pous` at `δ + ε_max + ε(t, 1) + ε(t, 6720)`, from `periodHeldSha512` at the other coins the
realization reads and the primitive `ω₀`. -/
theorem spaceOfPeriodHeldSha512 [Fintype ι]
    (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) {D Q L : ℕ}
    (hP : P2.Assumptions.P2RingAuditSecure D Q L) {C : Verity.Circuit} {I : Type} [DecidableEq I] {Ω : I → Type}
    [∀ i, Fintype (Ω i)] {r : PeriodRun d α C Ω} {run : Run d α ΩW ΩQ (HmCoins × Streams)} {P : Audits ΩW ΩQ}
    {τ : Pous.Period.Strategy ΩW ΩQ} {round : ℕ} {c : Certificates} {p : Price}
    (ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω)
    (R : RealizesSpaceOfPeriodHeld r (run.ofSource P.sid) (sourceVerdict P ω₀))
    (hfree : PousRingFree P) (hdev : DeviceModel run.withRing P τ D Q L round) (hinst : Instantiates P τ)
    (hS : c.S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S)
    {ε : ℕ → ℕ → ℝ≥0∞} {t : ℕ} (hprf : PrfTests run P c.S ω₀ ε t)
    (hB : ENNReal.ofReal Params.δ + Params.εMax + ε t 1 + ε t (105 * 64) ≠ ⊤) :
    r.Sound c p .pous (ENNReal.ofReal Params.δ + Params.εMax + ε t 1 + ε t (105 * 64)).toReal := by
  intro ξ
  have h := Pous.Period.SecurityProofs.periodHeldSha512 hH D Q L hP ι α ΩW ΩQ d run.withRing P τ round c.S hdev
    hinst hS (R.other ξ) (fun ω ω' y => ⟨(hfree _ ω ω' y).1, (hfree _ ω ω' y).2, rfl⟩) ω₀ ε t
    (hprf (R.other ξ)).1 (hprf (R.other ξ)).2
  rw [prCoin_eq_ofReal_pr] at h
  refine le_trans ?_ ((Pouw.pr_equiv R.coins _).trans_le ((ENNReal.ofReal_le_iff_le_toReal hB).1 h))
  refine Pouw.pr_mono fun x hx => ?_
  obtain ⟨ha, hn⟩ := hx
  have hA := R.accepts _ ha
  have hH' := R.held (Function.update ξ (r.own .pous) x)
  rw [R.other_update, Function.update_self] at hA hH'
  show P.Accepts _ ∧ ¬ P.Break _ ∧ _
  refine ⟨hA.1, hA.2, ?_⟩
  have hn' : ¬ c.S * (r.win.len : ℝ) ≤ (r.held (Function.update ξ (r.own .pous) x) : ℝ) := hn
  rw [hH', R.len] at hn'
  exact lt_of_not_ge hn'

/-- **PoUS at the code, over the challenger's source and hm96 coins**: `spaceOfPeriodHeldSha512` at `atCode r A`, from
a realization at `r` (its own `accepts` not read) and what `A .pous` gives at the joint run's coins (`PousAtCode`). -/
theorem pous_atCode_sha512 [Fintype ι]
    (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) {D Q L : ℕ}
    (hP : P2.Assumptions.P2RingAuditSecure D Q L) {C : Verity.Circuit} {I : Type} [DecidableEq I] {Ω : I → Type}
    [∀ i, Fintype (Ω i)] {r : PeriodRun d α C Ω} {run : Run d α ΩW ΩQ (HmCoins × Streams)} {P : Audits ΩW ΩQ}
    {τ : Pous.Period.Strategy ΩW ΩQ} {round : ℕ} {c : Certificates} {p : Price}
    (ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω)
    (R : RealizesSpaceOfPeriodHeld r (run.ofSource P.sid) (sourceVerdict P ω₀))
    (hfree : PousRingFree P) (hdev : DeviceModel run.withRing P τ D Q L round) (hinst : Instantiates P τ)
    (hS : c.S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S)
    {ε : ℕ → ℕ → ℝ≥0∞} {t : ℕ} (hprf : PrfTests run P c.S ω₀ ε t)
    (hB : ENNReal.ofReal Params.δ + Params.εMax + ε t 1 + ε t (105 * 64) ≠ ⊤)
    {A : Audit → ((i : I) → Ω i) → Prop} (hA : PousAtCode R A (sourceVerdict P ω₀)) :
    (atCode r A).Sound c p .pous (ENNReal.ofReal Params.δ + Params.εMax + ε t 1 + ε t (105 * 64)).toReal :=
  spaceOfPeriodHeldSha512 hH hP ω₀ (R.atCode A hA) hfree hdev hinst hS hprf hB

end Space

end Subprotocol

namespace Vocab

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period Subprotocol

set_option exponentiation.threshold 20000

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) {k : Verity.Registration.Composite}

/-- **A PoUS coupling's run realizes its audit over the memory challenger's source and hm96 coins**: a joint run over
`HmCoins × Streams`, read at the source (`Run.ofSource`, the streams the source's SHA-512 derivation), and the audit
`P` the period's data decodes to, which doesn't read the primitive, with a prover strategy of the device under the
device model and the instantiation at budget `(D, Q, L)`, the certificate's fit in the ring game's state bound, the
time `t` of the period's event as a test of `Sha512Prf`, and the coupling's verdict at the code meaning the verifier's
whole decision (`accepts`) outside a break (`brk`: a PoUS break is the coupling's, which its `BreakBound` answers
for). -/
structure PousCoins [Fintype ι] {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (q : V.Coupling k .pous θ F c σ) (D Q L : ℕ) where
  ΩW : Type
  ΩQ : Type
  run : Run d α ΩW ΩQ (HmCoins × Streams)
  P : Audits ΩW ΩQ
  τ : Pous.Period.Strategy ΩW ΩQ
  round : ℕ
  ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω
  t : ℕ
  real : RealizesSpaceOfPeriodHeld q.run (run.ofSource P.sid) (sourceVerdict P ω₀)
  free : PousRingFree P
  device : DeviceModel run.withRing P τ D Q L round
  inst : Instantiates P τ
  fit : (V.certOf θ).S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S
  accepts : ∀ ξ, ComponentAccepts k .pous (V.outcome θ F c σ q.coins ξ).sent →
    V.Logged (q.pos .pous) .pous (V.outcome θ F c σ q.coins ξ) →
      P.Accepts ((real.other ξ).1, (real.other ξ).2,
        (ω₀, (real.coins (ξ (q.run.own .pous))).2, challengerStreams (2 ^ 19) P.sid (real.coins (ξ (q.run.own .pous))).1))
  brk : ∀ ξ, P.Break ((real.other ξ).1, (real.other ξ).2,
      (ω₀, (real.coins (ξ (q.run.own .pous))).2, challengerStreams (2 ^ 19) P.sid (real.coins (ξ (q.run.own .pous))).1)) →
    q.brk ξ

/-- **The coupling's run at the verdict at the code has PoUS's per-period guarantee** at
`δ + ε_max + ε(t, 1) + ε(t, 6720)`, given `Sha512Prf` at the challenger's two tests. -/
theorem PousCoins.sound [Fintype ι] (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131))
    {D Q L : ℕ} (hP : P2.Assumptions.P2RingAuditSecure D Q L) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {q : V.Coupling k .pous θ F c σ} (w : V.PousCoins q D Q L) {ε : ℕ → ℕ → ℝ≥0∞}
    (hprf : PrfTests w.run w.P (V.certOf θ).S w.ω₀ ε w.t)
    (hB : ENNReal.ofReal Params.δ + Params.εMax + ε w.t 1 + ε w.t (105 * 64) ≠ ⊤) :
    q.atCode.Sound (V.certOf θ) q.price .pous
      (ENNReal.ofReal Params.δ + Params.εMax + ε w.t 1 + ε w.t (105 * 64)).toReal :=
  pous_atCode_sha512 hH hP w.ω₀ w.real w.free w.device w.inst w.fit hprf hB fun ξ
    (hx : ComponentAccepts k .pous (V.outcome θ F c σ q.coins ξ).sent ∧
      V.Logged (q.pos .pous) .pous (V.outcome θ F c σ q.coins ξ) ∧ ¬ q.brk ξ) =>
    And.intro (w.accepts ξ hx.1 hx.2.1) fun hb => hx.2.2 (w.brk ξ hb)

/-- **PoUS's target from its period bound at the deployed challenger's streams, conditional on a coupling**: at every
`θ F c σ`, a coupling whose run realizes PoUS's audit over the challenger's source and hm96 coins (`PousCoins`),
`Sha512Prf` at its two tests with a finite advantage `ε`, PoUS's published `δ` at least
`δ + ε_max + ε(t, 1) + ε(t, 6720)` (`1% + 2^-128` plus the PRF terms) at the strategy's calls, and its break and
out-of-place releases answered for by the forger and finder terms where the finder meets `FinderCR`
(`Coupling.BreakBound`), under `CoinCommitHides` and `P2RingAuditSecure` at `(D, Q, L)`. -/
theorem pousSound_of_sha512 [Fintype ι] (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131))
    {D Q L : ℕ} (hP : P2.Assumptions.P2RingAuditSecure D Q L) (q : ∀ θ F c σ, V.Coupling k .pous θ F c σ)
    (w : ∀ θ F c σ, V.PousCoins (q θ F c σ) D Q L) (ε : ℕ → ℕ → ℝ≥0∞) (hε : ∀ t n, ε t n ≠ ⊤)
    (hprf : ∀ θ F c σ, PrfTests (w θ F c σ).run (w θ F c σ).P (V.certOf θ).S (w θ F c σ).ω₀ ε (w θ F c σ).t)
    (hδ : ∀ θ F c σ, ENNReal.ofReal Params.δ + Params.εMax + ε (w θ F c σ).t 1 + ε (w θ F c σ).t (105 * 64) ≤
      V.delta θ .pous (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .pous θ F c σ → (q θ F c σ).BreakBound) : V.PousSound k :=
  V.pousSound_of_coupling q (fun θ F c σ =>
    ⟨_, PousCoins.sound V hH hP (w θ F c σ) (hprf θ F c σ) (sha512Bound_ne_top hε _), by
      rw [ENNReal.ofReal_toReal (sha512Bound_ne_top hε _)]; exact hδ θ F c σ⟩) hb

end Vocab

end Verity
