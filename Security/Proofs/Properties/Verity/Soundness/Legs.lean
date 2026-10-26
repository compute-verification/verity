import Properties.Verity.Soundness
import Properties.Verity.Legs
import Proofs.Core.Game.Prob
import Proofs.Core.UnionBound
import Proofs.Pouw.Prob
import Proofs.Accounting.JointBridge.WorkOfCreditedWords
import Proofs.Accounting.JointBridge.SpaceOfPeriodHeld
import Proofs.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
import Proofs.Accounting.JointBridge.CommOfPeriodLeakAccepted

/-!
# The work, space, bandwidth and communication targets from their component-level bounds (ONE-18)

`Verity.soundness_of_subprotocols` takes each subprotocol's target in the causal game (`Vocab.PouwSound`, `PousSound`,
`BandwidthAuditSound`, `NetworkCertifierSound`). Each follows from its owner's component-level bound
(`Vocab.ComponentBound`, `Properties.Verity.Legs`) by monotonicity: `Vocab.Verdict a o` is the file check and
`ComponentBound`'s first two conjuncts (`pouwSound_of`, `pousSound_of`, `poubSound_of`, `networkSound_of`).

`ComponentBound a` comes from the period guarantees (LEAN-52's bridges, `Accounting.SecurityProofs.Bridge`) in two
steps:

* **In the game** (`bound_of_sound`): at one `θ F c σ`, a period run `r` whose coins realize the play
  (`Vocab.Coins`: the game's probability of any event is the coins' probability of the outcome they lead to), whose
  verdict for `a` holds where the component accepts with its releases logged and no break, and whose claim for `a`
  holding means the game's does (`Vocab.WorldRuns`, the world the play realizes), gives `ComponentBound`'s event at
  most `δ` from `r.Sound … a δ`, averaged over the other coins (`Pouw.pr_le_of_update`).
* **At the code** (`work`, `space`, `bandwidth`, `network`): `r.Sound` at the run whose verdict is the compiled
  component's (`atCode r s`, verdict `ComponentAccepts a (s ξ)`), from the bridge and one refinement hypothesis: what the
  compiled component's acceptance of its case means for the bridge's verdict. The refinements are the component
  owners' pieces of note:proofs/20261026T0332Z-report-onethm (ONE-10, ONE-11, ONE-12, ONE-16); until they are proved
  they are hypotheses here, each a named `Prop` below in the shape its owner is to prove.

| leg | bridge | refinement (owner) |
| --- | --- | --- |
| work | `WorkOfCreditedWords` | `PouwRefines` (ONE-16, compute-accounting) |
| space | `SpaceOfPeriodHeld` | `PousAccepts` (ONE-10, memory-accounting) and `PousNoBreak` (open) |
| bandwidth | `BandwidthOfAuditPerRunPeriod` | `PoubRefines` (ONE-11, memory-accounting) |
| communication | `CommOfPeriodLeakAccepted` | `NetworkWindowLe` (ONE-12, network-accounting) |

Two shapes of bridge:

* Work and bandwidth read the verdict through an equivalence (`accepts_iff`), so they are taken at a model run whose
  verdict is the source game's, and `Sound` passes to the compiled verdict when that implies the model's
  (`sound_atCode_of_imp`). `atCode` ignores the run's own verdicts (`atCode_withAccepts`), so the model run is any
  `{r with accepts := …}`.
* Space and communication read the verdict in one field (`accepts`, `window_le`), so their realization is rebuilt at
  `atCode` (`RealizesSpaceOfPeriodHeld.atCode`, `RealizesCommOfPeriodLeakAccepted.atCode`) with that field replaced by
  the refinement; the realization's own verdict field is not read.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

theorem sound_of (a : Audit) (h : V.ComponentBound a) : V.Target a := fun θ F c σ =>
  le_trans (Game.prob_mono (fun _ ⟨⟨_, hc, hl⟩, hh, hb⟩ => ⟨hc, hl, hh, hb⟩) _ σ) (h θ F c σ)

/-- **PoUW's target from its component-level bound** (ONE-16's). -/
theorem pouwSound_of (h : V.ComponentBound .pouw) : V.PouwSound := V.sound_of .pouw h

/-- **PoUS's target from its component-level bound** (ONE-10's). -/
theorem pousSound_of (h : V.ComponentBound .pous) : V.PousSound := V.sound_of .pous h

/-- **The bandwidth audit's target from its component-level bound** (ONE-11's). -/
theorem poubSound_of (h : V.ComponentBound .poub) : V.BandwidthAuditSound := V.sound_of .poub h

/-- **The network certifier's target from its component-level bound** (ONE-12's). -/
theorem networkSound_of (h : V.ComponentBound .network) : V.NetworkCertifierSound := V.sound_of .network h

section Run

variable {C : Verity.Circuit} {I : Type} {Ω : I → Type} [∀ i, Fintype (Ω i)]

/-- **The coins realize the play of `σ`**: the game's probability of any event is the probability over the coins `ξ`
of the outcome `out ξ` they lead to. -/
def Coins (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (out : ((i : I) → Ω i) → Outcome) : Prop :=
  ∀ q : Outcome → Prop, Game.prob q (V.game θ F c) σ = ENNReal.ofReal (Pouw.pr fun ξ => q (out ξ))

/-- **The run's claim for `a` is the world's**: where the period run `r`'s claim for `a` holds at the coins `ξ`, at the
published certificates and price, the claim `a` certifies holds in the game at the outcome they lead to. -/
def WorldRuns (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : PeriodRun d α C Ω)
    (out : ((i : I) → Ω i) → Outcome) (a : Audit) : Prop :=
  ∀ ξ, r.Holds (V.certOf θ) (V.priceOf θ) a ξ → V.Holds θ F c σ a (out ξ)

variable [Fintype I] [DecidableEq I] [∀ i, Nonempty (Ω i)]

/-- **A period guarantee bounds the component's event in the game.** -/
theorem bound_of_sound {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {a : Audit} (r : PeriodRun d α C Ω)
    (out : ((i : I) → Ω i) → Outcome) (hc : V.Coins θ F c σ out)
    (hv : ∀ ξ, ComponentAccepts a (out ξ).sent → (∀ x ∈ V.coinsRead a (out ξ).sent, x ∈ (out ξ).log) →
      ¬ V.Break θ F c (out ξ) → r.accepts a ξ)
    (hw : V.WorldRuns θ F c σ r out a) {δ : ℝ} (hr : r.Sound (V.certOf θ) (V.priceOf θ) a δ)
    (hδ : ENNReal.ofReal δ ≤ V.delta θ a) :
    Game.prob (fun o => ComponentAccepts a o.sent ∧ (∀ x ∈ V.coinsRead a o.sent, x ∈ o.log) ∧
        ¬ V.Holds θ F c σ a o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ ≤ V.delta θ a := by
  rw [hc]
  refine le_trans (ENNReal.ofReal_le_ofReal ?_) hδ
  exact le_trans (Pouw.pr_mono fun ξ ⟨h1, h2, h3, h4⟩ => ⟨hv ξ h1 h2 h4, fun hh => h3 (hw ξ hh)⟩)
    (Pouw.pr_le_of_update (r.own a) _ hr)

end Run

end Vocab

namespace Legs

section

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}

/-- The period run whose verdicts are onsite's compiled components' on what the prover sent at the coins. -/
def atCode (r : PeriodRun d α C Ω) (s : ((i : I) → Ω i) → Sent) : PeriodRun d α C Ω :=
  { r with accepts := fun a ξ => ComponentAccepts a (s ξ) }

/-- `atCode` doesn't read the run's own verdicts. -/
theorem atCode_withAccepts (r : PeriodRun d α C Ω) (M : Audit → ((i : I) → Ω i) → Prop)
    (s : ((i : I) → Ω i) → Sent) : atCode { r with accepts := M } s = atCode r s := rfl

/-- **`atCode`'s verdict in the game**: where the coins lead to `out ξ` and the prover sent `(out ξ).sent`, the
component's acceptance is `atCode`'s verdict, so `bound_of_sound` at `atCode` takes no verdict hypothesis. -/
theorem bound_atCode {V : Vocab d α} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {a : Audit}
    [Fintype I] [DecidableEq I] [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (r : PeriodRun d α C Ω)
    (out : ((i : I) → Ω i) → Outcome) (hc : V.Coins θ F c σ out) (hw : V.WorldRuns θ F c σ r out a) {δ : ℝ}
    (hr : (atCode r fun ξ => (out ξ).sent).Sound (V.certOf θ) (V.priceOf θ) a δ)
    (hδ : ENNReal.ofReal δ ≤ V.delta θ a) :
    Game.prob (fun o => ComponentAccepts a o.sent ∧ (∀ x ∈ V.coinsRead a o.sent, x ∈ o.log) ∧
        ¬ V.Holds θ F c σ a o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ ≤ V.delta θ a :=
  V.bound_of_sound (atCode r fun ξ => (out ξ).sent) out hc (fun _ h _ _ => h) hw hr hδ

variable [DecidableEq I] [∀ i, Fintype (Ω i)]

/-- A per-period guarantee passes to `atCode`'s verdict when the compiled component's acceptance implies the run's
verdict. -/
theorem sound_atCode_of_imp {r : PeriodRun d α C Ω} {s : ((i : I) → Ω i) → Sent} {c : Certificates} {p : Price}
    {a : Audit} {δ : ℝ} (h : ∀ ξ, ComponentAccepts a (s ξ) → r.accepts a ξ) (hr : r.Sound c p a δ) :
    (atCode r s).Sound c p a δ := fun ξ =>
  le_trans (Pouw.pr_mono fun _ hx => ⟨h _ hx.1, hx.2⟩) (hr ξ)

/-! ## Work -/

/-- **ONE-16's refinement (CMP-145): PoUW's compiled component accepts only where the source game does.** At every
value `ξ` of the period's coins, if `PouwCheck.protocol` accepts its case in what the prover sent at `ξ`, the source
game the realization names (`acc`; for H1's checked credit, the play's acceptance) accepts at PoUW's coins there. -/
def PouwRefines {r : PeriodRun d α C Ω} {c : Certificates} {p : Price} {m u bUnit : ℕ} {X : Type}
    {acc : ((i : I) → Ω i) → X → Prop} {credit : ((i : I) → Ω i) → X → ℝ}
    (real : RealizesCreditedWords r c p m u bUnit X acc credit) (s : ((i : I) → Ω i) → Sent) : Prop :=
  ∀ ξ, ComponentAccepts .pouw (s ξ) → acc ξ (real.coin (ξ (r.own .pouw)))

/-- **The work leg at the code**: `WorkOfCreditedWords` at the model run `r`, whose PoUW verdict is the source game's
(`real`), and `PouwRefines`. -/
theorem work {r : PeriodRun d α C Ω} {c : Certificates} {p : Price} {m u bUnit : ℕ} {X : Type} [Fintype X]
    {acc : ((i : I) → Ω i) → X → Prop} {fixed credit help : ((i : I) → Ω i) → X → ℝ} {δ : ℝ}
    (real : RealizesCreditedWords r c p m u bUnit X acc credit)
    (hw : Assumptions.WordCostsMacs r p m u real.coin acc fixed help)
    (hh : Assumptions.HelpWithinCertifier r c bUnit real.coin acc help)
    (hsrc : ∀ ξ, Pouw.pr (fun x : X => acc ξ x ∧ fixed ξ x < credit ξ x) ≤ δ)
    {s : ((i : I) → Ω i) → Sent} (hR : PouwRefines real s) : (atCode r s).Sound c p .pouw δ := by
  refine sound_atCode_of_imp (fun ξ h => ?_)
    (Accounting.SecurityProofs.Bridge.WorkOfCreditedWords ι α I Ω d C r c p m u bUnit X acc fixed credit help δ real
      hw hh hsrc)
  have := (real.accepts_iff ξ (ξ (r.own .pouw))).2 (hR ξ h)
  rwa [Function.update_eq_self] at this

/-! ## Bandwidth -/

/-- **ONE-11's refinement (MEM-93): the bandwidth component accepts only where the bandwidth audit's game does.** At
every value `ξ` of the period's coins, if onsite's bandwidth component (`RemoteAudit.Protocol.protocol`) accepts its
case in what the prover sent at `ξ`, the game the realization names accepts its device at PoUB's coins there (the
store, the oracle at the fresh seed and the sample). -/
def PoubRefines {G N b h sb : ℕ} [Fintype ι] {r : PeriodRun d α C Ω}
    (real : RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := b) (h := h) (s := sb) r)
    (s : ((i : I) → Ω i) → Sent) : Prop :=
  ∀ ξ, ComponentAccepts .poub (s ξ) →
    Bandwidth.Accepts (real.dev ξ) (real.coins (ξ (r.own .poub))).1 (real.coins (ξ (r.own .poub))).2.1
      (real.coins (ξ (r.own .poub))).2.2

/-- **The bandwidth leg at the code**: `BandwidthOfAuditPerRunPeriod` at the model run `r`, whose PoUB verdict is the
game's (`real`), and `PoubRefines`. -/
theorem bandwidth {G N b h sb k lam : ℕ} [Fintype ι] {r : PeriodRun d α C Ω} {c : Certificates} {p : Price}
    (hN : 0 < N) (real : RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := b) (h := h) (s := sb) r)
    (hQ : c.Q ≤ ((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * r.win.len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam)
    {s : ((i : I) → Ω i) → Sent} (hR : PoubRefines real s) :
    (atCode r s).Sound c p .poub
      ((((k - 1 : ℕ) : ℝ) / N) ^ sb + (sb : ℝ) / 2 ^ h + (((d.rate .oracle * r.win.len).choose k : ℕ) : ℝ) / 2 ^ lam) :=
  sound_atCode_of_imp (fun ξ hx => (real.accepts_iff ξ).2 (hR ξ hx))
    (Accounting.SecurityProofs.Bridge.BandwidthOfAuditPerRunPeriod N b h sb G k lam ι α d C I Ω r c p hN real hQ)

/-! ## Space -/

section Space

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period

set_option exponentiation.threshold 20000

/-- What PoUS's compiled component accepting its case at `ξ` gives at the joint run's coins there, through the
realization: `V` at the other audits' coins and PoUS's coordinate carried to the ring game's. -/
def PousAtCode {r : PeriodRun d α C Ω} {ΩW ΩQ ΩS : Type} {run : Run d α ΩW ΩQ ΩS} {verdict : ΩW × ΩQ × ΩS → Prop}
    (R : RealizesSpaceOfPeriodHeld r run verdict) (s : ((i : I) → Ω i) → Sent) (V : ΩW × ΩQ × ΩS → Prop) : Prop :=
  ∀ ξ, ComponentAccepts .pous (s ξ) → V ((R.other ξ).1, (R.other ξ).2, R.coins (ξ (r.own .pous)))

/-- **ONE-10's refinement (MEM-93): PoUS's compiled component accepts only what the verifier's whole decision
accepts.** If `PousCheck.RunProtocol.protocol` accepts its case in what the prover sent at `ξ`, the audit `P` the
period's data decodes to is accepted by `Pous.EndToEnd.Accepted` (setup's check and the run's) at the joint run's coins. -/
def PousAccepts {r : PeriodRun d α C Ω} {ΩW ΩQ : Type}
    {run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    {verdict : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Prop}
    (R : RealizesSpaceOfPeriodHeld r run verdict) (s : ((i : I) → Ω i) → Sent) (P : Audits ΩW ΩQ) : Prop :=
  PousAtCode R s P.Accepts

/-- **What `SpaceOfPeriodHeld`'s verdict asks beyond acceptance: no break where the compiled component accepts.** If
PoUS's compiled component accepts at `ξ`, the period's own data holds no break (`Audits.Break`: a SHA-512 collision
among the strings the run hashes, or an LMS forgery under a key of the verifier's list). No check refuses a break, so
this is not a property of the code: it is the gap between `PeriodRun.Sound`, whose event has no break, and
`PeriodHeld`, whose event excludes one. -/
def PousNoBreak {r : PeriodRun d α C Ω} {ΩW ΩQ : Type}
    {run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    {verdict : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Prop}
    (R : RealizesSpaceOfPeriodHeld r run verdict) (s : ((i : I) → Ω i) → Sent) (P : Audits ΩW ΩQ) : Prop :=
  PousAtCode R s fun ω => ¬ P.Break ω

/-- A period's PoUS realization at the compiled run: the same coins, other coins, window and bit-ticks, with the
verdict the compiled component's. -/
def _root_.Accounting.RealizesSpaceOfPeriodHeld.atCode {r : PeriodRun d α C Ω} {ΩW ΩQ ΩS : Type}
    {run : Run d α ΩW ΩQ ΩS} {verdict : ΩW × ΩQ × ΩS → Prop} (R : RealizesSpaceOfPeriodHeld r run verdict)
    (s : ((i : I) → Ω i) → Sent) (h : PousAtCode R s verdict) :
    RealizesSpaceOfPeriodHeld (atCode r s) run verdict where
  coins := R.coins
  other := R.other
  other_update := R.other_update
  len := R.len
  held := R.held
  accepts := h

/-- **The space leg at the code**: `SpaceOfPeriodHeld` at the compiled run, from a realization at `r` (its own
`accepts` not read), `PousAccepts` and `PousNoBreak`. -/
theorem space (hd : AcceptsDecodes) (hp : PublishesRoot) (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ENNReal) ^ 131))
    {D Q L : ℕ} (hP : P2.Assumptions.P2RingAuditSecure D Q L) [Fintype ι] {r : PeriodRun d α C Ω} {ΩW ΩQ : Type}
    {run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)} {P : Audits ΩW ΩQ}
    {round : ℕ} {c : Certificates} {p : Price} (R : RealizesSpaceOfPeriodHeld r run fun ω => P.Accepts ω ∧ ¬ P.Break ω)
    (hdev : RingDevice run P D Q L round)
    (hS : c.S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S)
    {s : ((i : I) → Ω i) → Sent} (hA : PousAccepts R s P) (hB : PousNoBreak R s P) :
    (atCode r s).Sound c p .pous ((Params.δ : ℝ) + Params.εMax.toReal) :=
  Accounting.SecurityProofs.Bridge.SpaceOfPeriodHeld hd hp hH D Q L hP ι α d C I Ω (atCode r s) ΩW ΩQ run P round c p
    (R.atCode s fun ξ hx => ⟨hA ξ hx, hB ξ hx⟩) hdev hS

end Space

/-! ## Communication -/

section Comm

open Claims Claims.Sampling NetTiming Accounting.JointBridge

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
  {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ}
  {decode : List (ℕ × Col) → List T} {reg : Stmt → Finset ℕ} {Dj n : ℕ}

/-- **ONE-12's refinement (NET-82): the network certifier's per-window bound at its compiled component.** At every
window `i` and every value `ξ` of the other coins, the probability over the network audit's own coins that
`NetworkCheck.Period.protocol` accepts its case in what the prover sent while window `i` fails its good event
(`WindowGood`) is at most the sampled verifier's `PrBad` at the leak relation at the network certifier's rows, at the
window's statement and prover: `RealizesCommOfPeriodLeakAccepted.window_le` (and `PeriodClaim.realizes`'s `window_le`)
with the verdict the compiled component's. -/
def NetworkWindowLe {r : PeriodRun d α C Ω} {K : ℝ}
    (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n) (s : ((i : I) → Ω i) → Sent) :
    Prop :=
  ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      ComponentAccepts .network (s (Function.update ξ (r.own .network) x)) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt (Function.update ξ (r.own .network) x) i)
          (R.wit (Function.update ξ (r.own .network) x) i) (R.obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V
      (RLeakTap H D.commit D.parse D.partition L Kₒ S lay k δ (fun m => AsReg m.prog (lay m) (k m) (reg m)) Dj R.tops)
      (E D decode) (R.stmt ξ i) (R.prover ξ i)

/-- A period's network realization at the compiled run: the same windows, claim, scope, provers and network
certifier, with the per-window bound at the compiled component's verdict. -/
def _root_.Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted.atCode {r : PeriodRun d α C Ω} {K : ℝ}
    (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n) (s : ((i : I) → Ω i) → Sent)
    (h : NetworkWindowLe R s) : RealizesCommOfPeriodLeakAccepted (atCode r s) K H D L Kₒ S lay k δ decode reg Dj n where
  sw := R.sw
  ρ := R.ρ
  tops := R.tops
  stmt := R.stmt
  wit := R.wit
  obs := R.obs
  comm_iff := R.comm_iff
  scope := R.scope
  Tape := R.Tape
  tapeFintype := R.tapeFintype
  tapeNonempty := R.tapeNonempty
  prover := R.prover
  Conn := R.Conn
  connFintype := R.connFintype
  Cell := R.Cell
  Srv := R.Srv
  Dev := R.Dev
  arr := R.arr
  fmt := R.fmt
  device := R.device
  run := R.run
  trusted := R.trusted
  window_le := h

/-- **The communication leg at the code**: `CommOfPeriodLeakAccepted` at the compiled run, from a realization at `r`
(its own `window_le` not read) and `NetworkWindowLe`. -/
theorem network {r : PeriodRun d α C Ω} (c : Certificates) (p : Price)
    (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ decode reg Dj n) (hks : KS H D)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (R.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V (E D decode) D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      εtap)
    {s : ((i : I) → Ω i) → Sent} (hR : NetworkWindowLe R s) :
    (atCode r s).Sound c p .network (n * (δ + εks + εbind + εtap)) :=
  Accounting.SecurityProofs.Bridge.CommOfPeriodLeakAccepted c p (R.atCode s hR) hks hb hδ hdec εks εbind εtap herr hbind
    htap

end Comm

end

end Legs

end Verity
