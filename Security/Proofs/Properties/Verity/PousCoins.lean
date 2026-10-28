import Proofs.Properties.Verity.Subprotocols
import Proofs.Pous.PeriodGame

/-!
# PoUS's target from its period bound in the coin game

`Accounting.RealizesSpaceOfPeriodHeld` ties a period run's PoUS coordinate to a joint run's capacity coins by an
equivalence. At the ring game's coins (`ringΩ × HmCoins × Streams`, `SpaceOfPeriodHeld`'s) no coordinate of the game's
coins can be one: `|ringΩ|` has the odd factor `p = 2^16448 − 21065`, and the game's coins number a power of two.
`Pous.Period.PeriodHeldCoins` removes the primitive: where the session, the device record and the schedule don't read
it, the period's event over the memory challenger's hm96 coins and its streams alone has probability at most
`δ + ε_max` at every primitive. So the realization here is over `HmCoins × Streams`, bytes, which a round of the game's
coins can be:

* `Accounting.Run.withRing`: a joint run over `HmCoins × Streams`, read at coins that also carry the primitive, which it
  ignores; `PousRingFree`: the audit's session and device record don't read the primitive.
* `spaceOfPeriodHeldCoins`: `PeriodRun.Sound` for `.pous` at `δ + ε_max` at a period realizing its PoUS audit over
  those coins, from `Pous.Period.SecurityProofs.periodHeldCoins`, under its premises (`CoinCommitHides`,
  `P2RingAuditSecure`, `DeviceModel`, `Instantiates`, the certificate's fit), read at any one primitive `ω₀`.
* `pous_atCode_coins`, `Vocab.pousSound_of_coins`: through `Vocab.Coupling` (`pousSound_of_coupling`), `PousSound k`
  from a coupling at each `θ F c σ` whose run realizes its PoUS audit over the challenger's coins (`PousCoins`), with
  PoUS's published `δ` at least `δ + ε_max` and its break answered for by the finder and forger terms
  (`Coupling.BreakBound`).
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

/-- **The audit doesn't read the primitive**: at every value of the other audits' coins and the challenger's coins, the
memory challenger's signed session and its device record are the same at every primitive. -/
abbrev PousRingFree (P : Audits ΩW ΩQ) : Prop :=
  ∀ (o : ΩW × ΩQ) (ω ω' : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (y : HmCoins × Streams),
    P.session (o.1, o.2, (ω, y)) = P.session (o.1, o.2, (ω', y)) ∧
      P.device (o.1, o.2, (ω, y)) = P.device (o.1, o.2, (ω', y))

/-- **The verifier's whole decision outside a break, at the challenger's coins**, read at the primitive `ω₀`. -/
abbrev coinsVerdict (P : Audits ΩW ΩQ) (ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω)
    (x : ΩW × ΩQ × (HmCoins × Streams)) : Prop :=
  P.Accepts (x.1, x.2.1, (ω₀, x.2.2)) ∧ ¬ P.Break (x.1, x.2.1, (ω₀, x.2.2))

theorem εMax_ne_top : Params.εMax ≠ ⊤ := by
  unfold Params.εMax
  exact ENNReal.pow_ne_top (ENNReal.inv_ne_top.2 two_ne_zero)

theorem δ_nonneg : (0 : ℝ) ≤ (Params.δ : ℝ) := by
  unfold Params.δ
  norm_num

theorem ofReal_δ_εMax :
    ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal) = ENNReal.ofReal Params.δ + Params.εMax := by
  rw [ENNReal.ofReal_add δ_nonneg ENNReal.toReal_nonneg, ENNReal.ofReal_toReal εMax_ne_top]

/-- **PoUS's per-period guarantee at a period that realizes it over the challenger's coins**: `PeriodRun.Sound` for
`.pous` at `δ + ε_max`, from `periodHeldCoins` at the other coins the realization reads and the primitive `ω₀`. -/
theorem spaceOfPeriodHeldCoins [Fintype ι]
    (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) {D Q L : ℕ}
    (hP : P2.Assumptions.P2RingAuditSecure D Q L) {C : Verity.Circuit} {I : Type} [DecidableEq I] {Ω : I → Type}
    [∀ i, Fintype (Ω i)] {r : PeriodRun d α C Ω} {run : Run d α ΩW ΩQ (HmCoins × Streams)} {P : Audits ΩW ΩQ}
    {τ : Pous.Period.Strategy ΩW ΩQ} {round : ℕ} {c : Certificates} {p : Price}
    (ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (R : RealizesSpaceOfPeriodHeld r run (coinsVerdict P ω₀))
    (hfree : PousRingFree P) (hdev : DeviceModel run.withRing P τ D Q L round) (hinst : Instantiates P τ)
    (hS : c.S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S) :
    r.Sound c p .pous ((Params.δ : ℝ) + Params.εMax.toReal) := by
  intro ξ
  have h := Pous.Period.SecurityProofs.periodHeldCoins hH D Q L hP ι α ΩW ΩQ d run.withRing P τ round c.S hdev hinst
    hS (R.other ξ) (fun ω ω' y => ⟨(hfree _ ω ω' y).1, (hfree _ ω ω' y).2, rfl⟩) ω₀
  rw [prCoin_eq_ofReal_pr, ← ofReal_δ_εMax,
    ENNReal.ofReal_le_ofReal_iff (add_nonneg δ_nonneg ENNReal.toReal_nonneg)] at h
  refine le_trans ?_ ((Pouw.pr_equiv R.coins _).trans_le h)
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

/-- **PoUS at the code, over the challenger's coins**: `spaceOfPeriodHeldCoins` at `atCode r A`, from a realization at
`r` (its own `accepts` not read) and what `A .pous` gives at the joint run's coins (`PousAtCode`). -/
theorem pous_atCode_coins [Fintype ι]
    (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) {D Q L : ℕ}
    (hP : P2.Assumptions.P2RingAuditSecure D Q L) {C : Verity.Circuit} {I : Type} [DecidableEq I] {Ω : I → Type}
    [∀ i, Fintype (Ω i)] {r : PeriodRun d α C Ω} {run : Run d α ΩW ΩQ (HmCoins × Streams)} {P : Audits ΩW ΩQ}
    {τ : Pous.Period.Strategy ΩW ΩQ} {round : ℕ} {c : Certificates} {p : Price}
    (ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (R : RealizesSpaceOfPeriodHeld r run (coinsVerdict P ω₀))
    (hfree : PousRingFree P) (hdev : DeviceModel run.withRing P τ D Q L round) (hinst : Instantiates P τ)
    (hS : c.S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S)
    {A : Audit → ((i : I) → Ω i) → Prop} (hA : PousAtCode R A (coinsVerdict P ω₀)) :
    (atCode r A).Sound c p .pous ((Params.δ : ℝ) + Params.εMax.toReal) :=
  spaceOfPeriodHeldCoins hH hP ω₀ (R.atCode A hA) hfree hdev hinst hS

end Space

end Subprotocol

namespace Vocab

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period Subprotocol

set_option exponentiation.threshold 20000

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) {k : Verity.Registration.Composite}

/-- **A PoUS coupling's run realizes its audit over the memory challenger's coins**: a joint run over `HmCoins ×
Streams` and the audit `P` the period's data decodes to, which doesn't read the primitive, with a prover strategy of the
device under the device model and the instantiation at budget `(D, Q, L)`, the certificate's fit in the ring game's
state bound, and the coupling's verdict at the code meaning the verifier's whole decision (`accepts`) outside a break
(`brk`: a PoUS break is the coupling's, which its `BreakBound` answers for). -/
structure PousCoins [Fintype ι] {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (q : V.Coupling k .pous θ F c σ) (D Q L : ℕ) where
  ΩW : Type
  ΩQ : Type
  run : Run d α ΩW ΩQ (HmCoins × Streams)
  P : Audits ΩW ΩQ
  τ : Pous.Period.Strategy ΩW ΩQ
  round : ℕ
  ω₀ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω
  real : RealizesSpaceOfPeriodHeld q.run run (coinsVerdict P ω₀)
  free : PousRingFree P
  device : DeviceModel run.withRing P τ D Q L round
  inst : Instantiates P τ
  fit : (V.certOf θ).S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S
  accepts : ∀ ξ, ComponentAccepts k .pous (V.outcome θ F c σ q.coins ξ).sent →
    V.Logged (q.pos .pous) .pous (V.outcome θ F c σ q.coins ξ) →
      P.Accepts ((real.other ξ).1, (real.other ξ).2, (ω₀, real.coins (ξ (q.run.own .pous))))
  brk : ∀ ξ, P.Break ((real.other ξ).1, (real.other ξ).2, (ω₀, real.coins (ξ (q.run.own .pous)))) → q.brk ξ

/-- **The coupling's run at the verdict at the code has PoUS's per-period guarantee** at `δ + ε_max`. -/
theorem PousCoins.sound [Fintype ι] (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131))
    {D Q L : ℕ} (hP : P2.Assumptions.P2RingAuditSecure D Q L) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {q : V.Coupling k .pous θ F c σ} (w : V.PousCoins q D Q L) :
    q.atCode.Sound (V.certOf θ) q.price .pous ((Params.δ : ℝ) + Params.εMax.toReal) :=
  pous_atCode_coins hH hP w.ω₀ w.real w.free w.device w.inst w.fit fun ξ
    (hx : ComponentAccepts k .pous (V.outcome θ F c σ q.coins ξ).sent ∧
      V.Logged (q.pos .pous) .pous (V.outcome θ F c σ q.coins ξ) ∧ ¬ q.brk ξ) =>
    And.intro (w.accepts ξ hx.1 hx.2.1) fun hb => hx.2.2 (w.brk ξ hb)

/-- **PoUS's target from its period bound in the coin game**: at every `θ F c σ`, a coupling whose run realizes PoUS's
audit over the memory challenger's coins (`PousCoins`), PoUS's published `δ` at least `δ + ε_max` (`1% + 2^-128`) at the
strategy's calls, and its break and out-of-place releases answered for by the forger and finder terms where the finder
meets `FinderCR` (`Coupling.BreakBound`), under `CoinCommitHides` and `P2RingAuditSecure` at `(D, Q, L)`. -/
theorem pousSound_of_coins [Fintype ι] (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131))
    {D Q L : ℕ} (hP : P2.Assumptions.P2RingAuditSecure D Q L) (q : ∀ θ F c σ, V.Coupling k .pous θ F c σ)
    (w : ∀ θ F c σ, V.PousCoins (q θ F c σ) D Q L)
    (hδ : ∀ θ F c σ, ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal) ≤ V.delta θ .pous (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .pous θ F c σ → (q θ F c σ).BreakBound) : V.PousSound k :=
  V.pousSound_of_coupling q (fun θ F c σ => ⟨_, PousCoins.sound V hH hP (w θ F c σ), hδ θ F c σ⟩) hb

end Vocab

end Verity
