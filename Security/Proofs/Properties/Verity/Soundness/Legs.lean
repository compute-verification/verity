import Properties.Verity.Soundness
import Proofs.Pouw.Prob
import Proofs.Accounting.JointBridge.WorkOfCreditedWords
import Proofs.Accounting.JointBridge.SpaceOfPeriodHeld
import Proofs.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
import Proofs.Accounting.JointBridge.CommOfPeriodLeakAccepted

/-!
# The work, space, bandwidth and communication legs at onsite's compiled verifier (ONE-18)

`Verity.soundness_of_legs` takes each audit's per-period guarantee at its compiled component,
`(Verity.compiled r s).Sound c p a δ`, whose verdict is `Verity.ComponentAccepts a (s ξ)`. Each lemma here gives one
leg from its protocol's joint-period bridge (`Accounting.SecurityProofs.Bridge`) and one refinement hypothesis: what
the compiled component's acceptance of its case means for the bridge's verdict. The refinements are the component
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
  (`sound_compiled_of_imp`). `compiled` ignores the run's own verdicts (`compiled_withAccepts`), so the model run is
  any `{r with accepts := …}`.
* Space and communication read the verdict in one field (`accepts`, `window_le`), so their realization is rebuilt at
  the compiled run (`RealizesSpaceOfPeriodHeld.atCode`, `RealizesCommOfPeriodLeakAccepted.atCode`) with that field
  replaced by the refinement; the realization's own verdict field is not read.

With these four, `soundness_of_legs` needs only the integrity leg (ONE-8): `legs_of` assembles the five.
-/

namespace Verity.Legs

open Accounting

section

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}

/-- The compiled run doesn't read the run's own verdicts. -/
theorem compiled_withAccepts (r : PeriodRun d α C Ω) (M : Audit → ((i : I) → Ω i) → Prop)
    (s : ((i : I) → Ω i) → Sent) : compiled { r with accepts := M } s = compiled r s := rfl

variable [DecidableEq I] [∀ i, Fintype (Ω i)]

/-- A per-period guarantee passes to the compiled verdict when the compiled component's acceptance implies the run's
verdict. -/
theorem sound_compiled_of_imp {r : PeriodRun d α C Ω} {s : ((i : I) → Ω i) → Sent} {c : Certificates} {p : Price}
    {a : Audit} {δ : ℝ} (h : ∀ ξ, ComponentAccepts a (s ξ) → r.accepts a ξ) (hr : r.Sound c p a δ) :
    (compiled r s).Sound c p a δ := fun ξ =>
  le_trans (Pouw.pr_mono fun _ hx => ⟨h _ hx.1, hx.2⟩) (hr ξ)

/-- **The five legs from each audit's.** -/
theorem legs_of {r : PeriodRun d α C Ω} {s : ((i : I) → Ω i) → Sent} {c : Certificates} {p : Price}
    {δ : Audit → ℝ} (hI : (compiled r s).Sound c p .sampled (δ .sampled))
    (hW : (compiled r s).Sound c p .pouw (δ .pouw)) (hS : (compiled r s).Sound c p .pous (δ .pous))
    (hB : (compiled r s).Sound c p .poub (δ .poub)) (hN : (compiled r s).Sound c p .network (δ .network)) :
    ∀ a, (compiled r s).Sound c p a (δ a)
  | .sampled => hI
  | .pouw => hW
  | .pous => hS
  | .poub => hB
  | .network => hN

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
    {s : ((i : I) → Ω i) → Sent} (hR : PouwRefines real s) : (compiled r s).Sound c p .pouw δ := by
  refine sound_compiled_of_imp (fun ξ h => ?_)
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
    (compiled r s).Sound c p .poub
      ((((k - 1 : ℕ) : ℝ) / N) ^ sb + (sb : ℝ) / 2 ^ h + (((d.rate .oracle * r.win.len).choose k : ℕ) : ℝ) / 2 ^ lam) :=
  sound_compiled_of_imp (fun ξ hx => (real.accepts_iff ξ).2 (hR ξ hx))
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
    RealizesSpaceOfPeriodHeld (compiled r s) run verdict where
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
    (compiled r s).Sound c p .pous ((Params.δ : ℝ) + Params.εMax.toReal) :=
  Accounting.SecurityProofs.Bridge.SpaceOfPeriodHeld hd hp hH D Q L hP ι α d C I Ω (compiled r s) ΩW ΩQ run P round c p
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
    (h : NetworkWindowLe R s) : RealizesCommOfPeriodLeakAccepted (compiled r s) K H D L Kₒ S lay k δ decode reg Dj n where
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
    (compiled r s).Sound c p .network (n * (δ + εks + εbind + εtap)) :=
  Accounting.SecurityProofs.Bridge.CommOfPeriodLeakAccepted c p (R.atCode s hR) hks hb hδ hdec εks εbind εtap herr hbind
    htap

end Comm

end

end Verity.Legs
