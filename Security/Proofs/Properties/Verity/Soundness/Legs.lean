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
  verdict for `a` holds where `Vocab.Logged` does (the component accepts, its releases are logged, no break), and whose claim for `a`
  holding means the game's does (`Vocab.WorldRuns`, the world the play realizes), gives `ComponentBound`'s event at
  most `δ` from `r.Sound … a δ`, averaged over the other coins (`Pouw.pr_le_of_update`). `Coins` holds of every play
  whose run's coins are in bijection with the device's rounds' (`coins_play`, at `Legs.playOut`); `WorldRuns` is the
  world's (`Vocab.world`).
* **At the code** (`work`, `space`, `bandwidth`, `network`): `Sound` at `atCode r A`, the run `r` with the verdicts
  `A`, from the bridge and one refinement hypothesis: what `A` means for the bridge's verdict. In the game `A` is
  `Vocab.Logged` at the outcome the coins lead to (the component accepts its case, every release its check read is
  logged, no break), and `bound_atCode` takes the leg to `ComponentBound`'s event. The refinements are the component
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
  verdict is the source game's, and `Sound` passes to the verdicts `A` when they imply the model's
  (`sound_atCode_of_imp`). `atCode` ignores the run's own verdicts (`atCode_withAccepts`), so the model run is any
  `{r with accepts := …}`.
* Space and communication read the verdict in one field (`accepts`, `window_le`), so their realization is rebuilt at
  `atCode` (`RealizesSpaceOfPeriodHeld.atCode`, `RealizesCommOfPeriodLeakAccepted.atCode`) with that field replaced by
  the refinement; the realization's own verdict field is not read.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Legs

/-! ## The play's coins -/

variable {α β γ : Type}

theorem map_bind (h : α → β) (K : β → Game γ) : ∀ g : Game α, (g.map h).bind K = g.bind (fun a => K (h a))
  | .ret _ => rfl
  | .send M k => congrArg (Game.send M) (funext fun m => map_bind h K (k m))
  | @Game.coin _ C _ _ k => congrArg (@Game.coin _ C _ _) (funext fun c => map_bind h K (k c))

theorem prob_cast (p : α → Prop) {g g' : Game α} (h : g = g') (s : Game.Strategy g) :
    Game.prob p g s = Game.prob p g' (cast (congrArg Game.Strategy h) s) := by
  subst h; rfl

theorem avg_prod {A B : Type} [Fintype A] [Fintype B] [Nonempty A] [Nonempty B] (f : A × B → ℝ≥0∞) :
    Game.avg f = Game.avg fun a => Game.avg fun b => f (a, b) := by
  simp only [Game.avg, Fintype.card_prod, Nat.cast_mul, Fintype.sum_prod_type, Finset.mul_sum, ← mul_assoc]
  rw [ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]

/-- The coins of the device's rounds `ns`, one per round. -/
def CoinsOf : List ℕ → Type
  | [] => Unit
  | n :: ns => (Fin n → Bool) × CoinsOf ns

instance instFintypeCoinsOf : ∀ ns : List ℕ, Fintype (CoinsOf ns)
  | [] => inferInstanceAs (Fintype Unit)
  | n :: ns => @instFintypeProd (Fin n → Bool) (CoinsOf ns) _ (instFintypeCoinsOf ns)

instance instNonemptyCoinsOf : ∀ ns : List ℕ, Nonempty (CoinsOf ns)
  | [] => inferInstanceAs (Nonempty Unit)
  | n :: ns => @instNonemptyProd (Fin n → Bool) (CoinsOf ns) _ (instNonemptyCoinsOf ns)

/-- The device's rounds `ns`, then `K` of the log. -/
def releasesK : List ℕ → (List Release → Game α) → Game α
  | [], K => K []
  | n :: ns, K => .send ByteArray fun b => .coin (Fin n → Bool) fun x =>
      releasesK ns fun log => K (⟨b, List.ofFn x⟩ :: log)

theorem releases_bind : ∀ (ns : List ℕ) (K : List Release → Game α), (releases ns).bind K = releasesK ns K
  | [], _ => rfl
  | n :: ns, K => congrArg (Game.send ByteArray) (funext fun _ => congrArg (@Game.coin _ (Fin n → Bool) _ _)
      (funext fun _ => (map_bind _ K (releases ns)).trans (releases_bind ns _)))

/-- Where the coins `xs` lead under the strategy `s`: the device's log, and the strategy left for `K` of it. -/
def outK : (ns : List ℕ) → (K : List Release → Game α) → Game.Strategy (releasesK ns K) → CoinsOf ns →
    (log : List Release) × Game.Strategy (K log)
  | [], _, s, _ => ⟨[], s⟩
  | _ :: ns, K, s, (x, xs) =>
      let o := outK ns (fun log => K (⟨s.1, List.ofFn x⟩ :: log)) (s.2 x) xs
      ⟨⟨s.1, List.ofFn x⟩ :: o.1, o.2⟩

theorem prob_releasesK (q : α → Prop) : ∀ (ns : List ℕ) (K : List Release → Game α)
    (s : Game.Strategy (releasesK ns K)),
    Game.prob q (releasesK ns K) s = Game.avg fun xs => Game.prob q (K (outK ns K s xs).1) (outK ns K s xs).2
  | [], _, _ => (Game.avg_const _).symm
  | n :: ns, K, s => by
      have ih := fun x => prob_releasesK q ns (fun log => K (⟨s.1, List.ofFn x⟩ :: log)) (s.2 x)
      show Game.avg (fun x => Game.prob q (releasesK ns fun log => K (⟨s.1, List.ofFn x⟩ :: log)) (s.2 x)) = _
      rw [funext ih]
      exact (avg_prod (A := Fin n → Bool) (B := CoinsOf ns)
        (fun xs => Game.prob q (K (outK (n :: ns) K s xs).1) (outK (n :: ns) K s xs).2)).symm

/-- **The outcome of the play of `σ` at the device's coins `xs`.** -/
def playOut (ns : List ℕ) (θ F c : ByteArray) (σ : Game.Strategy (play ns θ F c)) (xs : CoinsOf ns) : Outcome :=
  let o := outK ns _ (cast (congrArg Game.Strategy (releases_bind ns _)) σ) xs
  ⟨θ, F, c, o.1, o.2.1⟩

theorem prob_play (q : Outcome → Prop) (ns : List ℕ) (θ F c : ByteArray) (σ : Game.Strategy (play ns θ F c)) :
    Game.prob q (play ns θ F c) σ = Game.prCoin fun xs => q (playOut ns θ F c σ xs) :=
  (prob_cast q (releases_bind ns _) σ).trans (prob_releasesK q ns _ _)

theorem prCoin_eq_ofReal_pr {C : Type} [Fintype C] (p : C → Prop) :
    Game.prCoin p = ENNReal.ofReal (Pouw.pr p) := by
  classical
  rw [Game.prCoin_eq_card p]
  unfold Pouw.pr
  rcases (Fintype.card C).eq_zero_or_pos with h | h
  · have := Fintype.card_eq_zero_iff.mp h
    simp
  · rw [ENNReal.ofReal_div_of_pos (by exact_mod_cast h), ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]

end Legs

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

variable {C : Verity.Circuit} {I : Type} {Ω : I → Type} [Fintype I] [DecidableEq I] [∀ i, Fintype (Ω i)]

/-- **The coins realize the play of `σ`**: the game's probability of any event is the probability over the coins `ξ`
of the outcome `out ξ` they lead to. -/
def Coins (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (out : ((i : I) → Ω i) → Outcome) : Prop :=
  ∀ q : Outcome → Prop, Game.prob q (V.game θ F c) σ = ENNReal.ofReal (Pouw.pr fun ξ => q (out ξ))

/-- **`Coins` holds of every play**: a period run's coins in bijection `e` with the device's rounds' coins realize
the play of `σ`, each leading to the outcome its rounds' coins do. -/
theorem coins_play (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (e : ((i : I) → Ω i) ≃ Legs.CoinsOf (V.rounds θ)) :
    V.Coins θ F c σ fun ξ => Legs.playOut (V.rounds θ) θ F c σ (e ξ) := fun q =>
  (Legs.prob_play q (V.rounds θ) θ F c σ).trans (by
    rw [Legs.prCoin_eq_ofReal_pr, Pouw.pr_equiv e fun xs => q (Legs.playOut _ θ F c σ xs)])

/-- **The run's claim for `a` is the world's**: where the period run `r`'s claim for `a` holds at the coins `ξ`, at the
published certificates and price, the claim `a` certifies holds in the game at the outcome they lead to. -/
def WorldRuns (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : PeriodRun d α C Ω)
    (out : ((i : I) → Ω i) → Outcome) (a : Audit) : Prop :=
  ∀ ξ, r.Holds (V.certOf θ) (V.priceOf θ) a ξ → V.Holds θ F c σ a (out ξ)

/-- **`ComponentBound`'s event at the coins, its claim aside**: at the outcome the coins `ξ` lead to, `a`'s compiled
component accepts its case in what the prover sent, every release its check read is in the device's log, and there is
no break. -/
def Logged (θ F c : ByteArray) (out : ((i : I) → Ω i) → Outcome) (a : Audit) (ξ : (i : I) → Ω i) : Prop :=
  ComponentAccepts a (out ξ).sent ∧ (∀ x ∈ V.coinsRead a (out ξ).sent, x ∈ (out ξ).log) ∧ ¬ V.Break θ F c (out ξ)

variable [∀ i, Nonempty (Ω i)]

/-- **A period guarantee bounds the component's event in the game.** -/
theorem bound_of_sound {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {a : Audit} (r : PeriodRun d α C Ω)
    (out : ((i : I) → Ω i) → Outcome) (hc : V.Coins θ F c σ out) (hv : ∀ ξ, V.Logged θ F c out a ξ → r.accepts a ξ)
    (hw : V.WorldRuns θ F c σ r out a) {δ : ℝ} (hr : r.Sound (V.certOf θ) (V.priceOf θ) a δ)
    (hδ : ENNReal.ofReal δ ≤ V.delta θ a) :
    Game.prob (fun o => ComponentAccepts a o.sent ∧ (∀ x ∈ V.coinsRead a o.sent, x ∈ o.log) ∧
        ¬ V.Holds θ F c σ a o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ ≤ V.delta θ a := by
  rw [hc]
  refine le_trans (ENNReal.ofReal_le_ofReal ?_) hδ
  exact le_trans (Pouw.pr_mono fun ξ ⟨h1, h2, h3, h4⟩ => ⟨hv ξ ⟨h1, h2, h4⟩, fun hh => h3 (hw ξ hh)⟩)
    (Pouw.pr_le_of_update (r.own a) (fun ξ => r.accepts a ξ ∧ ¬ r.Holds (V.certOf θ) (V.priceOf θ) a ξ) hr)

end Run

end Vocab

namespace Legs

section

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}

/-- The period run `r` with the verdicts `A`: at the code, `A` is `Vocab.Logged` at the play the coins lead to. -/
def atCode (r : PeriodRun d α C Ω) (A : Audit → ((i : I) → Ω i) → Prop) : PeriodRun d α C Ω :=
  { r with accepts := A }

/-- `atCode` doesn't read the run's own verdicts. -/
theorem atCode_withAccepts (r : PeriodRun d α C Ω) (M : Audit → ((i : I) → Ω i) → Prop)
    (A : Audit → ((i : I) → Ω i) → Prop) : atCode { r with accepts := M } A = atCode r A := rfl

/-- **A leg at the code bounds its component's event in the game**: `bound_of_sound` at `atCode r (V.Logged …)`,
whose verdict is the event's, so it takes no verdict hypothesis. -/
theorem bound_atCode {V : Vocab d α} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {a : Audit}
    [Fintype I] [DecidableEq I] [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (r : PeriodRun d α C Ω)
    (out : ((i : I) → Ω i) → Outcome) (hc : V.Coins θ F c σ out) (hw : V.WorldRuns θ F c σ r out a) {δ : ℝ}
    (hr : (atCode r (V.Logged θ F c out)).Sound (V.certOf θ) (V.priceOf θ) a δ)
    (hδ : ENNReal.ofReal δ ≤ V.delta θ a) :
    Game.prob (fun o => ComponentAccepts a o.sent ∧ (∀ x ∈ V.coinsRead a o.sent, x ∈ o.log) ∧
        ¬ V.Holds θ F c σ a o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ ≤ V.delta θ a :=
  V.bound_of_sound (atCode r (V.Logged θ F c out)) out hc (fun _ h => h) hw hr hδ

variable [DecidableEq I] [∀ i, Fintype (Ω i)]

/-- A per-period guarantee passes to `atCode`'s verdicts `A` when `A` implies the run's verdict. -/
theorem sound_atCode_of_imp {r : PeriodRun d α C Ω} {A : Audit → ((i : I) → Ω i) → Prop} {c : Certificates} {p : Price}
    {a : Audit} {δ : ℝ} (h : ∀ ξ, A a ξ → r.accepts a ξ) (hr : r.Sound c p a δ) :
    (atCode r A).Sound c p a δ := fun ξ =>
  le_trans (Pouw.pr_mono fun _ hx => ⟨h _ hx.1, hx.2⟩) (hr ξ)

/-! ## Work -/

/-- **ONE-16's refinement (CMP-145): PoUW's compiled component accepts only where the source game does.** At every
value `ξ` of the period's coins where PoUW's verdict `A .pouw` holds (at the code, `Vocab.Logged`: `PouwCheck.protocol`
accepts its case in what the prover sent, every release it read is logged, and there is no break), the source game the
realization names (`acc`; for H1's checked credit, the play's acceptance) accepts at PoUW's coins there. -/
def PouwRefines {r : PeriodRun d α C Ω} {c : Certificates} {p : Price} {m u bUnit : ℕ} {X : Type}
    {acc : ((i : I) → Ω i) → X → Prop} {credit : ((i : I) → Ω i) → X → ℝ}
    (real : RealizesCreditedWords r c p m u bUnit X acc credit) (A : Audit → ((i : I) → Ω i) → Prop) : Prop :=
  ∀ ξ, A .pouw ξ → acc ξ (real.coin (ξ (r.own .pouw)))

/-- **The work leg at the code**: `WorkOfCreditedWords` at the model run `r`, whose PoUW verdict is the source game's
(`real`), and `PouwRefines`. -/
theorem work {r : PeriodRun d α C Ω} {c : Certificates} {p : Price} {m u bUnit : ℕ} {X : Type} [Fintype X]
    {acc : ((i : I) → Ω i) → X → Prop} {fixed credit help : ((i : I) → Ω i) → X → ℝ} {δ : ℝ}
    (real : RealizesCreditedWords r c p m u bUnit X acc credit)
    (hw : Assumptions.WordCostsMacs r p m u real.coin acc fixed help)
    (hh : Assumptions.HelpWithinCertifier r c bUnit real.coin acc help)
    (hsrc : ∀ ξ, Pouw.pr (fun x : X => acc ξ x ∧ fixed ξ x < credit ξ x) ≤ δ)
    {A : Audit → ((i : I) → Ω i) → Prop} (hR : PouwRefines real A) : (atCode r A).Sound c p .pouw δ := by
  refine sound_atCode_of_imp (fun ξ h => ?_)
    (Accounting.SecurityProofs.Bridge.WorkOfCreditedWords ι α I Ω d C r c p m u bUnit X acc fixed credit help δ real
      hw hh hsrc)
  have := (real.accepts_iff ξ (ξ (r.own .pouw))).2 (hR ξ h)
  rwa [Function.update_eq_self] at this

/-! ## Bandwidth -/

/-- **ONE-11's refinement (MEM-93): the bandwidth component accepts only where the bandwidth audit's game does.** At
every value `ξ` of the period's coins where the bandwidth verdict `A .poub` holds (at the code, `Vocab.Logged`: onsite's
bandwidth component, `RemoteAudit.Protocol.protocol`, accepts its case, its releases are logged, and there is no
break), the game the realization names accepts its device at PoUB's coins there (the store, the oracle at the fresh
seed and the sample). -/
def PoubRefines {G N b h sb : ℕ} [Fintype ι] {r : PeriodRun d α C Ω}
    (real : RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := b) (h := h) (s := sb) r)
    (A : Audit → ((i : I) → Ω i) → Prop) : Prop :=
  ∀ ξ, A .poub ξ →
    Bandwidth.Accepts (real.dev ξ) (real.coins (ξ (r.own .poub))).1 (real.coins (ξ (r.own .poub))).2.1
      (real.coins (ξ (r.own .poub))).2.2

/-- **The bandwidth leg at the code**: `BandwidthOfAuditPerRunPeriod` at the model run `r`, whose PoUB verdict is the
game's (`real`), and `PoubRefines`. -/
theorem bandwidth {G N b h sb k lam : ℕ} [Fintype ι] {r : PeriodRun d α C Ω} {c : Certificates} {p : Price}
    (hN : 0 < N) (real : RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := b) (h := h) (s := sb) r)
    (hQ : c.Q ≤ ((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * r.win.len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam)
    {A : Audit → ((i : I) → Ω i) → Prop} (hR : PoubRefines real A) :
    (atCode r A).Sound c p .poub
      ((((k - 1 : ℕ) : ℝ) / N) ^ sb + (sb : ℝ) / 2 ^ h + (((d.rate .oracle * r.win.len).choose k : ℕ) : ℝ) / 2 ^ lam) :=
  sound_atCode_of_imp (fun ξ hx => (real.accepts_iff ξ).2 (hR ξ hx))
    (Accounting.SecurityProofs.Bridge.BandwidthOfAuditPerRunPeriod N b h sb G k lam ι α d C I Ω r c p hN real hQ)

/-! ## Space -/

section Space

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period

set_option exponentiation.threshold 20000

/-- What PoUS's verdict `A .pous` at `ξ` gives at the joint run's coins there, through the realization: `V` at the
other audits' coins and PoUS's coordinate carried to the ring game's. -/
def PousAtCode {r : PeriodRun d α C Ω} {ΩW ΩQ ΩS : Type} {run : Run d α ΩW ΩQ ΩS} {verdict : ΩW × ΩQ × ΩS → Prop}
    (R : RealizesSpaceOfPeriodHeld r run verdict) (A : Audit → ((i : I) → Ω i) → Prop) (V : ΩW × ΩQ × ΩS → Prop) : Prop :=
  ∀ ξ, A .pous ξ → V ((R.other ξ).1, (R.other ξ).2, R.coins (ξ (r.own .pous)))

/-- **ONE-10's refinement (MEM-93): PoUS's compiled component accepts only what the verifier's whole decision
accepts.** Where PoUS's verdict `A .pous` holds at `ξ` (at the code, `Vocab.Logged`: `PousCheck.RunProtocol.protocol`
accepts its case in what the prover sent, every release it read is logged, and there is no break), the audit `P` the
period's data decodes to is accepted by `Pous.EndToEnd.Accepted` (setup's check and the run's) at the joint run's coins. -/
def PousAccepts {r : PeriodRun d α C Ω} {ΩW ΩQ : Type}
    {run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    {verdict : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Prop}
    (R : RealizesSpaceOfPeriodHeld r run verdict) (A : Audit → ((i : I) → Ω i) → Prop) (P : Audits ΩW ΩQ) : Prop :=
  PousAtCode R A P.Accepts

/-- **What `SpaceOfPeriodHeld`'s verdict asks beyond acceptance: no PoUS break where PoUS's verdict holds.** Where
`A .pous` holds at `ξ`, the period's own data holds no break (`Audits.Break`: a SHA-512 collision among the strings the
run hashes, or an LMS forgery under a key of the verifier's list). No check refuses a break, so this is not a property
of the code. At the code `A` is `Vocab.Logged`, which excludes the game's break (`Vocab.Break`: a release missing from
the log, two openings of one position, a second registered file), so this asks that PoUS's break be one of the game's;
`Vocab.Break` doesn't name a collision among the strings PoUS's run hashes, and that is the gap. -/
def PousNoBreak {r : PeriodRun d α C Ω} {ΩW ΩQ : Type}
    {run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    {verdict : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Prop}
    (R : RealizesSpaceOfPeriodHeld r run verdict) (A : Audit → ((i : I) → Ω i) → Prop) (P : Audits ΩW ΩQ) : Prop :=
  PousAtCode R A fun ω => ¬ P.Break ω

/-- A period's PoUS realization at `atCode r A`: the same coins, other coins, window and bit-ticks, with the verdict
`A`'s. -/
def _root_.Accounting.RealizesSpaceOfPeriodHeld.atCode {r : PeriodRun d α C Ω} {ΩW ΩQ ΩS : Type}
    {run : Run d α ΩW ΩQ ΩS} {verdict : ΩW × ΩQ × ΩS → Prop} (R : RealizesSpaceOfPeriodHeld r run verdict)
    (A : Audit → ((i : I) → Ω i) → Prop) (h : PousAtCode R A verdict) :
    RealizesSpaceOfPeriodHeld (atCode r A) run verdict where
  coins := R.coins
  other := R.other
  other_update := R.other_update
  len := R.len
  held := R.held
  accepts := h

/-- **The space leg at the code**: `SpaceOfPeriodHeld` at `atCode r A`, from a realization at `r` (its own
`accepts` not read), `PousAccepts` and `PousNoBreak`. -/
theorem space (hd : AcceptsDecodes) (hp : PublishesRoot) (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ENNReal) ^ 131))
    {D Q L : ℕ} (hP : P2.Assumptions.P2RingAuditSecure D Q L) [Fintype ι] {r : PeriodRun d α C Ω} {ΩW ΩQ : Type}
    {run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)} {P : Audits ΩW ΩQ}
    {round : ℕ} {c : Certificates} {p : Price} (R : RealizesSpaceOfPeriodHeld r run fun ω => P.Accepts ω ∧ ¬ P.Break ω)
    (hdev : RingDevice run P D Q L round)
    (hS : c.S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S)
    {A : Audit → ((i : I) → Ω i) → Prop} (hA : PousAccepts R A P) (hB : PousNoBreak R A P) :
    (atCode r A).Sound c p .pous ((Params.δ : ℝ) + Params.εMax.toReal) :=
  Accounting.SecurityProofs.Bridge.SpaceOfPeriodHeld hd hp hH D Q L hP ι α d C I Ω (atCode r A) ΩW ΩQ run P round c p
    (R.atCode A fun ξ hx => ⟨hA ξ hx, hB ξ hx⟩) hdev hS

end Space

/-! ## Communication -/

section Comm

open Claims Claims.Sampling NetTiming Accounting.JointBridge

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
  {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ}
  {decode : List (ℕ × Col) → List T} {reg : Stmt → Finset ℕ} {Dj n : ℕ}

/-- **ONE-12's refinement (NET-82): the network certifier's per-window bound at its compiled component.** At every
window `i` and every value `ξ` of the other coins, the probability over the network audit's own coins that
its verdict `A .network` holds (at the code, `Vocab.Logged`: `NetworkCheck.Period.protocol` accepts its case in what
the prover sent, its releases are logged, and there is no break) while window `i` fails its good event
(`WindowGood`) is at most the sampled verifier's `PrBad` at the leak relation at the network certifier's rows, at the
window's statement and prover: `RealizesCommOfPeriodLeakAccepted.window_le` (and `PeriodClaim.realizes`'s `window_le`)
at the verdict `A`. -/
def NetworkWindowLe {r : PeriodRun d α C Ω} {K : ℝ}
    (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n) (A : Audit → ((i : I) → Ω i) → Prop) :
    Prop :=
  ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      A .network (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt (Function.update ξ (r.own .network) x) i)
          (R.wit (Function.update ξ (r.own .network) x) i) (R.obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V
      (RLeakTap H D.commit D.parse D.partition L Kₒ S lay k δ (fun m => AsReg m.prog (lay m) (k m) (reg m)) Dj R.tops)
      (E D decode) (R.stmt ξ i) (R.prover ξ i)

/-- A period's network realization at `atCode r A`: the same windows, claim, scope, provers and network certifier,
with the per-window bound at the verdict `A`. -/
def _root_.Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted.atCode {r : PeriodRun d α C Ω} {K : ℝ}
    (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n) (A : Audit → ((i : I) → Ω i) → Prop)
    (h : NetworkWindowLe R A) : RealizesCommOfPeriodLeakAccepted (atCode r A) K H D L Kₒ S lay k δ decode reg Dj n where
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

/-- **The communication leg at the code**: `CommOfPeriodLeakAccepted` at `atCode r A`, from a realization at `r`
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
    {A : Audit → ((i : I) → Ω i) → Prop} (hR : NetworkWindowLe R A) :
    (atCode r A).Sound c p .network (n * (δ + εks + εbind + εtap)) :=
  Accounting.SecurityProofs.Bridge.CommOfPeriodLeakAccepted c p (R.atCode A hR) hks hb hδ hdec εks εbind εtap herr hbind
    htap

end Comm

end

end Legs

end Verity
