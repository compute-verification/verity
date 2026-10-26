import Proofs.Flock.EndToEndHidden.Statement
import Proofs.Flock.EndToEnd.Period.Run

/-!
# The integrity audit over a period, and over a horizon (trusted)

A period's integrity audit is a sequence of windows, each one `--zk` session under hidden outputs with a fresh live
subset draw, as in `EndToEndHidden` (a `HiddenWindow`: every binder of `EndToEndHidden` but the prover). One prover
strategy plays the whole period (`Period.game`), adaptively: the strategy each window is played with may depend on
everything before it. `Period.run` records, for each window, whether flock-verify accepted it, its wrong units in the
committed transcript fixed at its registration (`wrongRegZ` at `XplurZC`, as in `EndToEndHidden`), and whether a drawn
unit is one of them.

* `IntegrityPeriod`: the probability that some window is accepted with at least its `K₀` units wrong is at most
  `Σ_w (miss_w + ε_ks,w + δ_link,w)`, each window's `EndToEndHidden` bound. So outside that probability, if flock-verify
  accepts every window, every window has fewer than its `K₀` units wrong.
* `IntegrityPeriodLog`: `IntegrityPeriod` with each window's threshold `K₀ ≥ ln(1/δ)/q`, where its draw covers a
  fraction `q` of its units, so each window's sampling term is at most `δ`.
* `IntegrityHorizon`: over `T` windows whose draws each cover at least a fraction `q` of their units, the probability
  that every window is accepted and the wrong units total at least `ln(1/δ)/q` is at most `δ + Σ_w (ε_ks,w + δ_link,w)`.

`ε_ks,w` is the window's knowledge term at the worst registration (`HiddenWindow.ks`), since the registration is the
prover's; `miss_w` and `δ_link,w` are `EndToEndHidden`'s. The named assumptions are `EndToEndHidden`'s, at every window:
A3 (`uniform/io-getrandombytes`) for the window's draw bytes, `live-verifier` (`HiddenWindow.hCust`), and `cr/sha-512`
and `ecr/sha-512` (`HiddenWindow.CR`) at the strategy the period's prover plays the window with, on every run
(`Period.CR`). The proofs are `Flock.SecurityProofs.IntegrityPeriod`, `IntegrityPeriodLog` and `IntegrityHorizon`.

C-Flock's definitions are still in its proofs package, so, like `EndToEndHidden`, this module imports them.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **One window of a period**: one `--zk` session under hidden outputs over `n` proof units, with every binder of
`EndToEndHidden` but the prover: its statement (circuit descriptor, public file), its law (a uniform `kd`-subset of the
`n` units, split evenly over the session's tables), its record (draw files, session writer, reference setup), the
verifier's byte source, the verify command's tags, `live-verifier` (`hCust`), the count curve's parameters, the
threshold `K₀` and the knowledge finders' budgets `qF`, `qS`. -/
structure HiddenWindow where
  n : ℕ
  Reg : Type
  I : Inputs
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : Built D.unroll.unit
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub
  hpt : pub.tables = none
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  kd : ℕ
  hkd : kd ≤ n
  hdiv : kd % nTab I = 0
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ
  t' : ℝ
  ht' : 0 ≤ t'
  K₀ : ℕ
  hK₀ : K₀ ≤ n
  qF : ℝ
  qS : ℝ

attribute [instance] HiddenWindow.fintypeΩ HiddenWindow.nonemptyΩ

namespace HiddenWindow

open Classical

variable (w : HiddenWindow)

/-- The window's draw: `Flock.Draw.drawOS`'s uniform `kd`-subset on the verifier's byte source. -/
def L : Law w.n := Law.execOS (n := w.n) w.src (.subset w.kd) w.n

/-- The window's audit as it runs: `EndToEndHidden`'s game. -/
noncomputable def game : Game (w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript) :=
  auditLive w.L w.Reg (liveOfZJ w.dg (planZHJ w.dj w.y₀) w.N w.wr)

/-- The key program's partition into proof units. -/
noncomputable def P := (keyProg w.D.unroll w.hU w.n).partition (Discharge.Layout.outsOf w.D.unroll w.hU w.n)

/-- The public file's commitments. -/
noncomputable def vb :=
  (rsZHJ w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg)).bindingZ
  (rsZHJ_computes w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg))

/-- The model audit's strategy that a live strategy `τ` refines to (`EndToEndHidden`'s `σ`). -/
noncomputable def sim (τ : Strategy w.game) :=
  simZJ execArith hs512 (planZHJ w.dj w.y₀) w.I w.wr (opnAtZHJ (planZHJ w.dj w.y₀) w.I w.dj) w.dg
    (decAtZJ (planZHJ w.dj w.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) w.N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ w.I w.wr _ _ _ _ _
      (runsZCJH_live w.I w.ht (Flock.Desc.parse_flat w.hD) (Discharge.Layout.scopeOkH_spec w.hU w.hscope).2.1 w.dj w.y₀
        w.wr _ w.hTags w.hCust.1
        fun _ => drawOkZKHJ_of_file w.I w.ht (Flock.Desc.parse_flat w.hD) w.kd w.n w.hdiv w.dj w.hCust.2 _)) τ

/-- **The window's wrong units** at the strategy `τ`: `wrongRegZ` in the committed transcript fixed at registration
(`XplurZC`), as in `EndToEndHidden`. -/
noncomputable def wrongSet (τ : Strategy w.game) : Finset (Fin w.n) :=
  wrongRegZ execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ))
    (XplurZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ)) (cont (w.sim τ)) w.k w.Rw)

/-- **`cr/sha-512` and `ecr/sha-512` at the strategy `τ`**: `EndToEndHidden`'s hypotheses `TableCRZ`, for the knowledge
finders of each table the prover's session runs, and `LinkCRZC`, for its link finders. -/
def CR (τ : Strategy w.game) : Prop :=
  (∀ ω j, TableCRZ execArith hs512 (planZHJ w.dj w.y₀ (w.L.draw ω) (reg (w.sim τ))) j
      (zcStrat execArith hs512 (planZHJ w.dj w.y₀) (cont (w.sim τ) ω)) w.qF w.qS) ∧
    LinkCRZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ)) (cont (w.sim τ)) w.k w.Rw
      w.M w.t'

/-- The sampling term, `C(n − K₀, kd)/C(n, kd)`. -/
noncomputable def miss : ℝ≥0∞ := ((w.n - w.K₀).choose w.kd : ℝ≥0∞) / (w.n.choose w.kd : ℝ≥0∞)

/-- The knowledge term `ε_ks`, `ksAvgStrictZ` at the worst registration. -/
noncomputable def ks : ℝ≥0∞ := ⨆ R : w.Reg, ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS R

/-- The link term `δ_link = Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`, as in `EndToEndHidden`. -/
noncomputable def link : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ p : w.vb.Pos,
      (Fintype.card (ΩcZ execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb w.L p) : ℝ) / Fintype.card w.L.Ω) *
    (2 * (w.t' * (1 + w.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * w.M) + w.k / (Real.exp 1 * w.Rw)) / (1 - w.ρ))

end HiddenWindow

/-- What a period's run records of window `idx`: whether flock-verify accepted it, how many of its units are wrong
(`HiddenWindow.wrongSet`), and whether a drawn unit is one of them. -/
structure WindowRec where
  idx : ℕ
  accepted : Prop
  wrong : ℕ
  drawnWrong : Prop

namespace HiddenWindow

open Classical

/-- Window `t`'s record, at the strategy `τ` that played it and its output `o`. -/
noncomputable def record (w : HiddenWindow) (t : ℕ) (τ : Strategy w.game)
    (o : w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript) : WindowRec :=
  ⟨t, LiveAcceptsZK w.I o, (w.wrongSet τ).card, ∃ u ∈ o.2.1, u ∈ w.wrongSet τ⟩

end HiddenWindow

namespace Period

/-- **A period of `T` windows**, window `t`'s audit after window `t − 1`'s, against one prover. -/
noncomputable def game (W : ℕ → HiddenWindow) (T : ℕ) : Game Unit := periodG (fun t => (W t).game) 0 T

/-- The period's run at the prover strategy `s`: each window's record, in order. -/
noncomputable def run (W : ℕ → HiddenWindow) (T : ℕ) (s : Strategy (game W T)) : Game (List WindowRec) :=
  periodRun (fun t => (W t).game) (fun t => (W t).record t) 0 T s

/-- `cr/sha-512` and `ecr/sha-512` (`HiddenWindow.CR`) at every window's strategy, on every run of the period. -/
def CR (W : ℕ → HiddenWindow) (T : ℕ) (s : Strategy (game W T)) : Prop :=
  PeriodGood (fun t => (W t).game) (fun t => (W t).CR) 0 T s

end Period

/-- **The integrity audit over a period.** In a period of `T` windows `W t`, each a `--zk` session under hidden outputs
with a fresh live uniform `kd`-subset draw, played by one adaptive prover `s`, the probability that some window is
accepted by flock-verify with at least its `K₀` units wrong is at most the sum of the windows' `EndToEndHidden` bounds,
$$\Pr[\exists t,\ \text{window } t \text{ accepted} \wedge K_{0,t} \le \text{wrong}_t] \le
  \sum_t \left(\frac{\binom{n_t-K_{0,t}}{kd_t}}{\binom{n_t}{kd_t}} + \varepsilon_{ks,t} + \delta_{link,t}\right).$$
Its named assumptions, at every window: A3 (`Assumptions.UniformRandomBytes`) for the window's draw bytes,
`live-verifier` (`HiddenWindow.hCust`), and `cr/sha-512` and `ecr/sha-512` at the strategy each window is played with
on every run (`Period.CR`). -/
def IntegrityPeriod : Prop :=
  ∀ (W : ℕ → HiddenWindow) (T : ℕ),
    (∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src) →
    ∀ s : Strategy (Period.game W T), Period.CR W T s →
    value (fun l => ∃ r ∈ l, r.accepted ∧ (W r.idx).K₀ ≤ r.wrong) (Period.run W T s) ≤
      ∑ t ∈ Finset.range T, ((W t).miss + (W t).ks + (W t).link)

/-- **The integrity audit over a period, at `δ`.** In a period as in `IntegrityPeriod` whose windows' draws each cover
at least a fraction `q > 0` of their units (`q ≤ kd_t/n_t`), with each window's threshold `K₀,t ≥ ln(1/δ)/q`, the
probability that some window is accepted with at least its `K₀` units wrong is at most
$$\sum_t (\delta + \varepsilon_{ks,t} + \delta_{link,t}),$$
each window's sampling term `C(n − K₀, kd)/C(n, kd) ≤ (1 − q)^{K₀}` being at most `δ`. Its named assumptions are
`IntegrityPeriod`'s. -/
def IntegrityPeriodLog : Prop :=
  ∀ (W : ℕ → HiddenWindow) (T : ℕ) (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t < T, q ≤ ((W t).kd : ℝ) / (W t).n) →
    (∀ t < T, Real.log (1 / δ) / q ≤ (W t).K₀) →
    (∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src) →
    ∀ s : Strategy (Period.game W T), Period.CR W T s →
    value (fun l => ∃ r ∈ l, r.accepted ∧ (W r.idx).K₀ ≤ r.wrong) (Period.run W T s) ≤
      ∑ t ∈ Finset.range T, (ENNReal.ofReal δ + (W t).ks + (W t).link)

/-- **The integrity audit over a horizon.** In `T` windows as in `IntegrityPeriod`, each window's draw covering at
least a fraction `q > 0` of its units (`q ≤ kd_t/n_t`), the probability that every window is accepted and the wrong
units total at least `ln(1/δ)/q` is at most
$$\delta + \sum_t (\varepsilon_{ks,t} + \delta_{link,t}).$$
Its named assumptions are `IntegrityPeriod`'s. -/
def IntegrityHorizon : Prop :=
  ∀ (W : ℕ → HiddenWindow) (T : ℕ) (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t < T, q ≤ ((W t).kd : ℝ) / (W t).n) →
    (∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src) →
    ∀ s : Strategy (Period.game W T), Period.CR W T s →
    value (fun l => (∀ r ∈ l, r.accepted) ∧ Real.log (1 / δ) / q ≤ ((l.map WindowRec.wrong).sum : ℝ))
        (Period.run W T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).ks + (W t).link)

end Flock.Guarantees
