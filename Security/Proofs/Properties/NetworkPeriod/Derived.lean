import Proofs.Properties.NetworkPeriod
import Proofs.Properties.NetworkPeriod.Bernoulli
import Proofs.Flock.Claims.Sampling.Select.Strata
import Proofs.Flock.Claims.Sampling.Reference.DeployedCode

/-!
# `WrongBounds` at the parsed statement's design (`wrongBounds_derived`)

`ScanWindowsPlaced`'s `WrongBounds` holds at the bound the audit publishes. Where the registration is the statement's
(`SamplingDeploy.RegMatches`), the statement's design is the one the draw runs: it draws the strata the audit reads off
the law the rule names (`Properties.SampledProofs.SecurityProofs.strataDerived`), or the population at the Bernoulli rate
the audit reads (`Properties.SampledProofs.SecurityProofs.bernoulliDerived`). So `WrongBounds` holds at the bound the
audit publishes for that law, whichever draw it is (`wrongBounds_derived`, over `wrongBoundsDrawn` and
`wrongBounds_of_bernoulliBound`).

Nothing imports this module: `Reference.DeployedCode` reaches ArkLib, whose global `notation3 "X"` makes `X` a token in
every module that imports it, and `Proofs.Properties.NetworkPeriod`'s importers name arguments `X`.
-/

namespace Properties.NetworkPeriod

/-- **`WrongBounds` at the parsed statement's design**: where the registration a statement names is the statement's
(`SamplingDeploy.RegMatches`: its layout the statement's partition, its design the statement's), the bound the audit
publishes for the law the registration's rule names over its population, as `OneStage.check` reads it (`Draw.derive`,
`Record.profileDraw`, `Bound.boundOf`), holds for the statement on its partition at every `δ = num/den` with `0 < num`
and `0 < den`. Where the audit reads the draw as strata (`Bound.strataOf`), the design draws those strata, then the input
class whole (`strataDerived`, `wrongBoundsDrawn`); else the draw is a Bernoulli draw, and the design draws the
population at its rate and the input class whole (`bernoulliDerived`, `wrongBounds_of_bernoulliBound`). -/
theorem wrongBounds_derived {c : SamplingDeploy.Code} {reg : Claims.Msg → Option SamplingDeploy.Reg}
    (hreg : SamplingDeploy.RegMatches c reg) {x : Claims.Msg} {s : Claims.Sampling.AuditStatement}
    {Pt : Claims.Sampling.Partition} (hx : c.parse x = some s) (hP : c.partition s = some Pt) :
    ∃ g, reg x = some g ∧ ∀ (n : ℕ) (named law : Lean.Json) (draw : Core.SamplingDesign),
      SampledProofs.Partition.units g.program g.query = .ok n → g.rule.lawOver n = .ok named →
      SampledProofs.Draw.derive g.program g.query named g.table = .ok law →
      SampledProofs.Record.profileDraw law = some draw →
      ∀ δ : Core.Ratio, 0 < δ.num → 0 < δ.den →
        WrongBounds s Pt ((δ.num : ℝ) / δ.den) (SampledProofs.Bound.boundOf n draw δ) := by
  obtain ⟨g, l, hg, -, hl, rfl, hd⟩ := hreg x s Pt hx hP
  refine ⟨g, hg, fun n named law draw hn hnamed hlaw hdraw δ hδ hδd => ?_⟩
  cases hs : SampledProofs.Bound.strataOf n draw with
  | some strata =>
    exact wrongBoundsDrawn n draw strata hs s _ _ δ
      (Properties.SampledProofs.SecurityProofs.strataDerived g.rule g.program g.query g.table l s.design n named law
        draw strata hl hd hn hnamed hlaw hdraw hs) hδ hδd
  | none =>
    obtain ⟨p, rfl⟩ : ∃ p, draw = .bernoulli p := by
      cases draw with
      | bernoulli p => exact ⟨p, rfl⟩
      | _ => simp [SampledProofs.Bound.strataOf] at hs
    exact wrongBounds_of_bernoulliBound
      (Properties.SampledProofs.SecurityProofs.bernoulliDerived g.rule g.program g.query g.table l s.design n named law
        p hl hd hn hnamed hlaw hdraw) δ hδd

end Properties.NetworkPeriod
