import Definitions.Accounting.JointPeriod
import Proofs.Flock.EndToEnd.Period.Statement

/-!
# The integrity audit of a period is C-Flock's integrity audit over a monitoring period's sampling blocks (trusted)

A period's sampled-proofs audit (`Audit.sampled`) is C-Flock's integrity audit over the period's sampling blocks
(`Flock.Guarantees.IntegrityHorizon`): `T` sampling blocks `W t`, each a `--zk` session under hidden outputs with a
fresh live subset draw, against one adaptive prover. `RealizesIntegrityOfIntegrityPeriod` ties a `PeriodRun` to it: the
sampled audit's own coins drive the period's game at the prover's strategy, its verdict is flock-verify's on every
sampling block, and the claim's units and faulty proof blocks are the sampling blocks'. `IntegrityOfIntegrityPeriod` is
then `PeriodRun.Sound` for the sampled audit at error `δ + Σ_t (ε_ks,t + δ_link,t)`, the monitoring period's.

C-Flock's definitions are still in its proofs package, so, like `Flock.Guarantees.IntegrityHorizon`, this module imports
them.
-/

namespace Accounting

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- Period `r`'s sampled-proofs audit is C-Flock's integrity audit over the `T` sampling blocks `W t` of the period
(`Flock.Guarantees.Period.run`). Every field is a property of the deployed run:

* `strat`, `record`, `law`: the node is one prover strategy of the period's game at each value of the other audits'
  coins, and the sampled audit's own coins (`r.own .sampled`), drawn uniformly, drive that game: every event of the
  sampling blocks' records has the probability, over those coins, that the game's run gives it at that strategy. This is
  the live coin flow: each sampling block's subset draw comes from the verifier's byte source, which the period's
  sampled coordinate is (A3, `FlockSoundness.Assumptions.UniformRandomBytes`, stays a hypothesis of the guarantee).
* `accepts_iff`: the sampled audit accepts exactly when flock-verify accepts every sampling block.
* `wrong_eq`: the faulty proof blocks of the run's partition in its committed trace are, in number, the sampling blocks'
  faulty proof blocks (`SamplingBlock.wrongSet`: `wrongRegZ` in the trace fixed at registration, over the key program's
  partition). This is the program-from-circuit correspondence (M1: the key program and its partition are the circuit
  `C`'s lowering and `r.parts`), which the bridge PRs land as `Lowers`/`progOf`/`partOf`; until then it is this field.
* `units_eq`, `gates_nonempty`: the partition's public unit-size property. Its units are the sampling blocks' units, and
  every unit has at least one of the circuit's gates, so no empty unit, trivially correct, dilutes the share `ε` the
  claim bounds. -/
structure RealizesIntegrityOfIntegrityPeriod {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)] (r : PeriodRun d α C Ω)
    (W : ℕ → Flock.Guarantees.SamplingBlock) (T : ℕ) where
  /-- the prover strategy the node plays the period's game with, at the coins (it reads only the other audits') -/
  strat : ((i : I) → Ω i) → FlockSoundness.Game.Strategy (Flock.Guarantees.Period.game W T)
  /-- the sampling blocks' records at the coins: flock-verify's verdict and the faulty proof blocks of each sampling
  block -/
  record : ((i : I) → Ω i) → List Flock.Guarantees.WindowRec
  /-- the sampled audit's own coins drive the period's game at the strategy -/
  law : ∀ (ξ : (i : I) → Ω i) (E : List Flock.Guarantees.WindowRec → Prop),
    ENNReal.ofReal (Pouw.pr fun x : Ω (r.own .sampled) => E (record (Function.update ξ (r.own .sampled) x))) =
      FlockSoundness.Game.value E (Flock.Guarantees.Period.run W T (strat ξ))
  /-- the sampled audit accepts exactly when flock-verify accepts every sampling block -/
  accepts_iff : ∀ ξ, r.accepts .sampled ξ ↔ ∀ w ∈ record ξ, w.accepted
  /-- the run's faulty proof blocks are the sampling blocks' (M1, the program-from-circuit correspondence) -/
  wrong_eq : ∀ ξ, (r.parts.wrong (r.transcript ξ)).card = ((record ξ).map Flock.Guarantees.WindowRec.wrong).sum
  /-- the partition's units are the sampling blocks' units -/
  units_eq : r.units = ∑ t ∈ Finset.range T, (W t).n
  /-- every unit of the partition has at least one gate -/
  gates_nonempty : ∀ u, (r.parts.gates u).Nonempty

namespace Guarantees.Bridge

open FlockSoundness.Audit FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkHidden in
/-- **The integrity audit of a period, from C-Flock's integrity audit over a monitoring period.** For a period `r` whose
sampled-proofs audit realizes C-Flock's integrity audit over `T` sampling blocks `W t`
(`RealizesIntegrityOfIntegrityPeriod`), each sampling block's draw covering at least a fraction `q > 0` of its units,
under A3 for every sampling block's draw bytes and `cr/sha-512` (`TableCRZ`) and `ecr/sha-512` (`LinkCRZC`) at every
strategy the node plays a sampling block with on some run (`Period.Plays`), at certificates whose share `ε` of the
period's units is at least `ln(1/δ)/q`, the sampled audit has its per-period guarantee (`PeriodRun.Sound`) at error
$$\delta + \sum_t (\varepsilon_{ks,t} + \delta_{link,t}),$$
`Flock.Guarantees.IntegrityHorizon`'s, when that is finite. -/
def IntegrityOfIntegrityPeriod : Prop :=
  ∀ (ι : Type) (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) (I : Type) (Ω : I → Type)
    [DecidableEq I] [∀ i, Fintype (Ω i)] (r : PeriodRun d α C Ω) (W : ℕ → Flock.Guarantees.SamplingBlock) (T : ℕ)
    (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t < T, q ≤ ((W t).kd : ℝ) / (W t).n) →
    (∀ t < T, FlockSoundness.Assumptions.UniformRandomBytes (W t).Lb (W t).src) →
    ∀ R : RealizesIntegrityOfIntegrityPeriod r W T,
    (∀ ξ t (τ : Strategy (W t).game), Flock.Guarantees.Period.Plays W T (R.strat ξ) t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (W t).dj (W t).y₀ ((W t).L.draw ω) (reg ((W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (W t).dj (W t).y₀) (cont ((W t).sim τ) ω)) (W t).qF (W t).qS) →
    (∀ ξ t (τ : Strategy (W t).game), Flock.Guarantees.Period.Plays W T (R.strat ξ) t τ →
      LinkCRZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
        (reg ((W t).sim τ)) (cont ((W t).sim τ)) (W t).k (W t).Rw (W t).M (W t).t') →
    ∀ (c : Certificates) (p : Price), Real.log (1 / δ) / q ≤ c.ε * r.units →
    ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).ks + (W t).link) ≠ ⊤ →
    r.Sound c p .sampled (ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).ks + (W t).link)).toReal

end Guarantees.Bridge

end Accounting
