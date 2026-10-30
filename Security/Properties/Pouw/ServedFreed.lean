import Definitions.Pouw.Dither
import Properties.Pouw.ServedWork

/-!
# The served PoUW protocol's freed compute, in the onsite form

`ServedWork` is the remote form: the prover's whole inventory is one machine of 𝓜 whose programs exchange nothing
outside it, and an accepted period's priced work on it is at least `p` a credited word, less the luck allowances.
`ServedFreed` is the onsite form at the same check (the J10-R period, `PouwCheck.Credit.window` on each window): the
inventory is the registered devices, a network certifier counts the bits their window programs exchange with machines
outside them, and an accepted period that the certifier also accepts frees at most `ε C` of the registered devices'
capacity at the period's dual price (`Pouw.Dither.Jobs.FreedWithin`). It composes:

* `ServedWork` at an exchange-free replacement of the window programs;
* the price of an exchanged bit over the period's window programs together (`OutsourceWithinEgressServed`): the
  window programs have exchange-free ones, each fixed before its window's salt, with the same outputs at the windows'
  inputs, whose priced work is at most `β` a bit the originals exchanged more;
* the certifier's count (`CertifiedExchange`): it accepts a period whose window programs exchanged more than `K` bits
  but with probability `δ`;
* weak duality (`Pouw.Guarantees.Dither.FreedOfFloor`): a floor on the credited work at the dual price bounds what
  the period frees.

`ε` is the period's own: an accepted period's credit (`creditPrefix`) depends on its prefix, so the bound reads
`ε(x) C = max(0, C − (p · creditPrefix(x) − Σ_w lam_w) + β K)`.

**Non-vacuity rests on the fixed model.** `ServedWork`'s `SAKServed` is false on 𝓜 as written, so this statement says
something only on a model that charges the free register reductions that falsify it.

**The coins stay in the boundary.** `OutsourceWithinEgressServed` is claimed where a window's salt
reaches a machine outside the registered devices only through a bit their programs exchange (`CoinsConfidential`).
For the MVP that holds by placement: the device that runs a server's network certifier produces that server's live
coins, and private messages travel only on private links (`WitnessPathPrivate`). Trust in that device and its host
provider is `trusted-host`.

The proof is `Proofs.Pouw.ServedFreed`; `Proofs.Pouw.ServedFreed.Honest` has its honest and nontrivial lemmas.
-/

namespace Properties.Pouw.Served

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Properties.Pouw.PeriodJ10R

/-- `pouw-outsource-egress-served` (proposed; class L; `pouw-outsource-egress` over a period's window programs
together). **An exchanged bit replaces at most `β` of priced work, over a period's window programs**: in the
model at its device premises, for the prover's instances `inst` and advice `state`, every family of window programs
`work` fixed before its windows' salts (`FixedBeforeSalt`) has an exchange-free family `work'`, also fixed before the
salts, that outputs what `work` does at each window's advice and salt, and whose priced work over the period is at
most `work`'s plus `β` a bit `work` exchanged with machines outside the registered devices.

Main's per-program `Pouw.Assumptions.OutsourceWithinEgress` implies it
(`Properties.Pouw.Served.SecurityProofs.outsourceServed_of_perProgram`): this is the weaker claim. At `β` it holds
only if a window's honest priced cost is at most `β` times the fewest bits that carry its salt out and its outputs
back. It is falsified by window programs, fixed before their salts, whose every exchange-free
replacement with the same outputs costs more than `β` a bit they exchanged more, or by a salt readable outside the
boundary but through a counted bit (module doc). -/
def OutsourceWithinEgressServed {C B I St : Type} (O : Type) {m n : ℕ} (hm : 0 < m) (G : Pouw.Dither.Accel)
    (μ : G.Res → ℝ) (β : ℝ) (inst : Paths (Paths C m × B) n → Fin n → I)
    (state : Paths (Paths C m × B) n → Fin n → St) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ work : Paths (Paths C m × B) n → Fin n → G.Mc.Alg (St × C) O,
    FixedBeforeSalt (fun x w => (inst x w, state x w, work x w)) →
    ∃ work' : Paths (Paths C m × B) n → Fin n → G.Mc.Alg (St × C) O,
      FixedBeforeSalt (fun x w => (inst x w, state x w, work' x w)) ∧ (∀ x w i, G.ext (work' x w) i = 0) ∧
      ∀ x, (∀ w, G.Mc.eval (work' x w) (state x w, saltCoin hm x w) =
          G.Mc.eval (work x w) (state x w, saltCoin hm x w)) ∧
        ∑ w, G.cost μ (work' x w) (state x w, saltCoin hm x w) ≤
          ∑ w, G.cost μ (work x w) (state x w, saltCoin hm x w) +
            β * ∑ w, G.ext (work x w) (state x w, saltCoin hm x w)

/-- `pouw-certified-exchange` (proposed; the network certifier's count at the served check). **The certifier
accepts a period that exchanged more than `K` bits but with probability `δ`**: the probability over the period's coins
that the certifier's check `cert` accepts while the period's window programs exchanged `ext > K` bits with machines
outside the registered devices is at most `δ`. The served form of `Pouw.Assumptions.CertifiedImport`; the certifier's
guarantee (`NetTiming.Guarantees.PeriodLeakAccepted`) bounds the information an accepted period carries out, and the
step from it to this count is open. It is falsified by a strategy whose periods the certifier accepts, with
probability more than `δ`, after its window programs exchanged more than `K` bits. -/
def CertifiedExchange {Ω : Type} [Fintype Ω] (cert : Ω → Prop) (ext : Ω → ℝ) (K δ : ℝ) : Prop :=
  prob (fun x => cert x ∧ K < ext x) ≤ δ

/-- **The served PoUW protocol's freed compute, onsite** (the J10-R period, `Credit.window`'s verdict).
Under `ServedWork`'s premises and setting, with the inventory's closure (`ext = 0`) dropped: let the window programs
`work` run on the registered devices, one machine `G` of 𝓜 at its device premises, and exchange bits with machines
outside them; let `Pd` be the registered devices' period with `μ` its dual price at one credit (`Jobs.DualOpt μ 1`),
`C = μ · hon > 0`, and `β ≥ 0`. If an exchanged bit replaces at most `β` of priced work over the period's window
programs (`OutsourceWithinEgressServed`) and the certifier's check `cert` accepts a period whose window programs
exchanged more than `K` bits but with probability `δ` (`CertifiedExchange`), then except with probability
`δ_s + n·(δ_bind + δ_ks) + n·κ_deg(B) + δ`, a period that the work check and the certifier both accept frees at most
`ε(x) C` of the registered devices' capacity beside its windows' timed work, where
`ε(x) C = max(0, C − (p · creditPrefix(x) − Σ_w lam(inst_w)) + β K)`: no other work `z` that fits beside the windows'
use is worth more than an honest strategy's plus `ε(x) C`. Non-vacuous only on the fixed model (module doc). -/
def ServedFreed : Prop :=
  ∀ {ι K T L U C Ξ V I W St : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty C] [Nonempty Ξ] [Nonempty V] {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ) (Sc : Scheme K) (Cd : Declared ι)
    {n : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ tileCount σ sT) (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1)
    (_hKd : 1 ≤ k sT) (ws : CreditWindows (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V) (enc : C → ByteArray)
    (place : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    (Wp : Period C (Coin C Ξ Cd.rounds σ sT) (Fin (tileCount σ sT)) Cd.rounds n)
    (Pl : Play (Coin C Ξ Cd.rounds σ sT) n K T L U)
    (folded : Paths (Coin C Ξ Cd.rounds σ sT) n → ℝ) (ρ δs δbind δks : ℝ)
    (G : Pouw.Dither.Accel) (μ : G.Res → ℝ) (Reg : Set I) (hon : I → C → Fin (tileCount σ sT) → W) (p : ℝ) (lam : I → ℝ)
    (κdeg : ℕ → ℝ) (B : ℕ) (inst : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → I)
    (state : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → St)
    (work : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → G.Mc.Alg (St × C) (Fin (tileCount σ sT) → W))
    (Pd : Pouw.Dither.Jobs G) (cert : Paths (Coin C Ξ Cd.rounds σ sT) n → Prop) (β Kc δ : ℝ),
    Scope Sc Cd δs →
    0 < ρ → ρ ≤ 1 →
    (∀ t : Fin (tileCount σ sT), ρ * (tileCount σ sT) * shareCone Sc Cd (tileCount σ sT) t ≤ 1) →
    (∀ j, stageShare (tileStage Cd (tileCount σ sT)) (shareCone Sc Cd (tileCount σ sT)) j ≤ 1) →
    (∀ t : Fin (tileCount σ sT), 0 < shareCone Sc Cd (tileCount σ sT) t →
      tileLuck Sc Cd (tileCount σ sT) t ≤ 7 / 8 ∧ tileLuck0 Sc Cd (tileCount σ sT) t ≤ 1 / 2) →
    Pl.game = periodGame (tileStage Cd (tileCount σ sT)) (liveCone Sc Cd (tileCount σ sT)) 1 1 (k sT) Wp →
    Wp.EveryWindow (fun live _ => DeadConeDead Sc Cd (tileCount σ sT) live) →
    PrefixOf (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
      (j10rAtCoin hr enc place tile)) Pl.game →
    (∀ w, prob (fun x => (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
      (j10rAtCoin hr enc place tile)).MovedIn x w) ≤ δbind) →
    RegisteredBeforeDraw ws → TileDrawIsCoin σ k hk sT unitOf ws →
    TileServiceSound σ sT unitOf ws opened (pinnedAt (saltOf ws) place tile) δks →
    Properties.Pouw.Period.SaltIsLiveCoin hr (saltOf ws) enc → SharedNoiseLucks Sc Cd Wp →
    UnitFixed Pl (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
        (j10rAtCoin hr enc place tile))
      (periodMode Cd Wp (tileRun σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
        (j10rAtCoin hr enc place tile)))
      (fun _ t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) folded →
    G.G1 → G.G2 → G.P1 → G.P2 →
    Pouw.Assumptions.SAKServed G μ Reg hon (fun t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) p lam κdeg →
    (∃ e : St → Fin (2 ^ B), Function.Injective e) →
    (∀ x w, inst x w ∈ Reg) → 0 ≤ p →
    FixedBeforeSalt (fun x w => (inst x w, state x w, work x w)) →
    FixedAreRight Pl (fun x w => Pouw.Assumptions.rightWords hon (fun t => (tileWords Sc Cd (tileCount σ sT) t : ℝ))
      (inst x w) (saltCoin hr x w) (G.Mc.eval (work x w) (state x w, saltCoin hr x w))) →
    Pd.DualOpt μ 1 → 0 < ∑ r, μ r * (1 * Pd.hon r) → 0 ≤ β →
    OutsourceWithinEgressServed (Fin (tileCount σ sT) → W) hr G μ β inst state →
    CertifiedExchange cert (fun x => ∑ w, G.ext (work x w) (state x w, saltCoin hr x w)) Kc δ →
    prob (fun x => Pl.accepts x ∧ cert x ∧
      ∃ z, Pd.Fits (fun r => ∑ w, G.use (work x w) (state x w, saltCoin hr x w) r) z ∧
        ¬ Pd.FreedWithin 1 z
          (max 0 (∑ r, μ r * (1 * Pd.hon r) -
              (p * creditPrefix Sc Cd (tileCount σ sT) (ws.prefixLen x) (creditLive Sc Cd Wp ws x) 1 1
                (j10rRate (k sT) ρ) δs - ∑ w, lam (inst x w)) + β * Kc) /
            ∑ r, μ r * (1 * Pd.hon r))
          (∑ r, μ r * (1 * Pd.hon r))) ≤
      δs + n * (δbind + δks) + n * κdeg B + δ

end Properties.Pouw.Served
