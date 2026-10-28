import Definitions.Pouw.Assumptions.OnsiteFreed

/-!
# The onsite PoUW protocol's freed compute (trusted)

`OnsiteFreed`: an accepted onsite period frees at most `ε C` of the registered devices' capacity at the period's dual
price (`Pouw.Dither.FreedSound`), for any work check over a round protocol. It composes:

* the remote form's floor (`RemoteWorkFloor`): an accepted period of a prover whose whole inventory is the registered
  devices spent at least `F` of priced timed work on them, but with probability `κ`;
* the network certifier's count (`CertifiedImport`): the certifier's check accepts a period whose programs exchanged
  more than `K` bits with machines outside the registered devices but with probability `δ`;
* the price of an exchanged bit over the period's programs together (`OutsourceWithinEgressPeriod`):
  every prover has an exchange-free one with the same transcript that costs at most `β` a bit more;
* weak duality (`Pouw.Guarantees.Dither.FreedOfFloor`): a floor on the credited work at the dual price bounds what
  the period frees.

The remote and onsite forms differ in one premise: the remote form prices no outside machine, since the inventory is
everything the prover runs, and the onsite form prices each bit that crosses the registered devices' boundary at `β`,
the certifier bounding how many do.

At the served check (the J10-R period, `PouwCheck.Credit.window`) the same composition is
`Properties.Pouw.Served.ServedFreed`, over `ServedWork` in the play's vocabulary, with no remote floor taken as a
premise.
-/

namespace Pouw.Guarantees

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **An accepted onsite period frees at most `ε C`**. In every accelerator model `G` at its
device premises (G1, G2, P1, P2), for every round protocol `P` over a period of `k` sampling blocks, every work check
`vw` on the transcript (the proof service's, which reads no device), every network certifier's check `vn` on the
transcript and the devices' readings `dev`, every period `Pd` of the registered devices and every price `μ` that is
the period's dual price at one credit (`Period.DualOpt μ 1`), with `C = μ · hon > 0` the credited work's honest cost
there and `β ≥ 0`: if

* a prover that exchanges nothing, whose period the work check accepts, spent at least `F` of priced timed work, but
  with probability `κ` (`RemoteWorkFloor`, the remote form),
* every prover has an exchange-free one with the same transcript whose timed work costs at most `β` a bit more, over
  the period's programs together (`OutsourceWithinEgressPeriod`), and
* the certifier's check accepts a period that exchanged more than `K` bits but with probability `δ`
  (`CertifiedImport`),

then for every prover, the probability over the oracle and the verifier's coins that the onsite verdict accepts its
period (`onsiteVerdict vw vn`: both checks) and some other work `z` fits on the registered devices beside its sampling
blocks' timed work with `rv + W_other(z) > OPT_honest + ε C` is at most `κ + δ`, where `ε C = max(0, C − F + β K)`.

**Non-vacuity rests on the fixed model.** On 𝓜 as `Definitions.Core.Accel` writes it, SAK for matrix products is
false, and so is `RemoteWorkFloor` at any useful `F` for a check that credits matrix products. This statement says
something only on a model that charges the free register reductions that falsify them.

Open (stated so the red team can aim at them):
* `RemoteWorkFloor` is the remote form's conclusion taken as a premise. At the served check it isn't needed:
  `Properties.Pouw.Served.ServedFreed` composes `ServedWork` itself;
* `CertifiedImport` counts bits (`Prover.ext`), while the certifier's guarantee
  (`NetTiming.Guarantees.PeriodLeakAccepted`) bounds the information an accepted period carries out (at most `2^K`
  values) in Accounting's device vocabulary: the step from one to the other is open;
* `OutsourceWithinEgressPeriod` is claimed where the coins stay in the certifier's boundary (`CoinsConfidential`);
  for the MVP that is placement (`WitnessPathPrivate`), whose discharge is the secure-transport workstream's. -/
def OnsiteFreed : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k : ℕ) (Obs : Type) (vw : Verdict P k Unit) (vn : Verdict P k Obs)
    (dev : Readings G P k Obs) (Pd : Period G) (μ : G.Res → ℝ) (F β K κ δ : ℝ),
    Pd.DualOpt μ 1 → 0 < ∑ r, μ r * (1 * Pd.hon r) → 0 ≤ β →
    G.G1 → G.G2 → G.P1 → G.P2 →
    RemoteWorkFloor G P k vw μ F κ → OutsourceWithinEgressPeriod G P k μ β → CertifiedImport G P k vn dev K δ →
    FreedSound G P k (onsiteVerdict vw vn) dev Pd 1
      (fun _ => max 0 (∑ r, μ r * (1 * Pd.hon r) - F + β * K) / ∑ r, μ r * (1 * Pd.hon r))
      (∑ r, μ r * (1 * Pd.hon r)) (fun _ => κ + δ)

end Pouw.Guarantees
