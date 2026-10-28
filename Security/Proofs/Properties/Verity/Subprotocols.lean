import Properties.Verity.Subprotocols
import Proofs.Properties.Verity.Soundness
import Proofs.Core.Game.Prob
import Proofs.Core.UnionBound
import Proofs.Pouw.Prob
import Proofs.Accounting.JointBridge.WorkOfCreditedWords
import Proofs.Accounting.JointBridge.SpaceOfPeriodHeld
import Proofs.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
import Proofs.Accounting.JointBridge.CommOfPeriodLeakAccepted

/-!
# PoUW's, PoUS's, the bandwidth audit's and the network certifier's targets from their period guarantees

`Verity.soundness_of_subprotocols` takes each subprotocol's target in the causal game (`Vocab.PouwSound`, `PousSound`,
`BandwidthAuditSound`, `NetworkCertifierSound`). Each follows from its component-level bound (`Vocab.ComponentBound`,
`Properties.Verity.Subprotocols`) by monotonicity, since `Vocab.Verdict a` includes `a`'s component's acceptance
(`pouwSound_of`, `pousSound_of`, `poubSound_of`, `networkSound_of`). Beside them, rule 3's checks:
`pouwSound_of_honest` and its siblings (at registered files, where the subprotocol's honest lemma `Vocab.HonestHolds`
holds, the deployment's honest prover meets `ComponentBound`'s event and the target's with probability `0`), and
`pouwSound_of_nontrivial` and its siblings (the target fails for a verifier that accepts everything, at a strategy on
whose every play the claim fails and whose error is below one, so its finder meets `FinderCR`).

`ComponentBound a` comes from a per-period guarantee of the coins model (`Accounting.PeriodRun.Sound`, the
bridges `Accounting.SecurityProofs.Bridge.*`) through a coupling (`Vocab.Coupling`) at each `θ F c σ`:

* **The coins.** A period run whose coins are in bijection with the device's rounds' coins
  (`Subprotocol.CoinsOf (V.rounds θ)`): every event of the game has the probability over the run's coins of the outcome
  they lead to (`Vocab.prob_eq_pr`, from `Subprotocol.prob_play`), so a bound over the coins is one in the game.
* **The world.** The run's claim for `a` at the coupling's price implies the game's at that outcome (`world`, the world
  each play realizes).
* **The verdict at the code** (`Coupling.verdict`): `a`'s component accepts, every release its check read is the
  device's at the round the coupling assigns it (`pos`, `Logged`), and the coupling's break (`brk`) is absent. The rest
  of the event, a release out of place or a break, is the finder's: where it meets `FinderCR`, its probability is at
  most `a`'s collision term (`Coupling.BreakBound`).

`Coupling.bound` gives `ComponentBound`'s event at most `a`'s error from the run at the verdict at the code
(`Coupling.atCode`) meeting a guarantee within `a`'s published `δ`, and `componentBound_of_coupling` collects it over
`θ F c σ`. The guarantee at `atCode` comes from the bridge and one refinement hypothesis, what the verdict at the code
means for the bridge's verdict (`Subprotocol.pouw_atCode`, `pous_atCode`, `poub_atCode`, `network_atCode`). Until the
refinements are proved of the code they are hypotheses, each a named `Prop`:

| subprotocol | bridge | refinement |
| --- | --- | --- |
| PoUW | `WorkOfCreditedWords` | `PouwRefines` |
| PoUS | `SpaceOfPeriodHeld` | `PousAccepts`; `PousNoBreak` by construction (`Coupling.pousNoBreak`) |
| bandwidth audit | `BandwidthOfAuditPerRunPeriod` | `PoubRefines` |
| network certifier | `CommOfPeriodLeakAccepted` | `NetworkWindowLe` |

Two shapes of bridge:

* PoUW's and the bandwidth audit's read the verdict through an equivalence (`accepts_iff`), so they are taken at a model
  run whose verdict is the source game's, and `Sound` passes to the verdicts `A` when they imply the model's
  (`sound_atCode_of_imp`). `atCode` ignores the run's own verdicts (`atCode_withAccepts`), so the model run is any
  `{r with accepts := …}`.
* PoUS's and the network certifier's read the verdict in one field (`accepts`, `window_le`), so their realization is
  rebuilt at `atCode` (`RealizesSpaceOfPeriodHeld.atCode`, `RealizesCommOfPeriodLeakAccepted.atCode`) with that field
  replaced by the refinement; the realization's own verdict field is not read.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Subprotocol

section Coins

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

/-- Each round's coins, in order, as the device releases them. -/
def CoinsOf.bits : (ns : List ℕ) → CoinsOf ns → List (List Bool)
  | [], _ => []
  | _ :: ns, (x, xs) => List.ofFn x :: CoinsOf.bits ns xs

/-- **The device's log holds each round's coins at its round**, whatever the strategy sent. -/
theorem outK_coins : ∀ (ns : List ℕ) (K : List Release → Game α) (s : Game.Strategy (releasesK ns K))
    (xs : CoinsOf ns), (outK ns K s xs).1.map Release.coins = CoinsOf.bits ns xs
  | [], _, _, _ => rfl
  | _ :: ns, K, s, (x, xs) => congrArg (List.ofFn x :: ·) (outK_coins ns _ (s.2 x) xs)

/-- **The play's log holds each round's coins at its round**, whatever the strategy. -/
theorem playOut_coins (ns : List ℕ) (θ F c : ByteArray) (σ : Game.Strategy (play ns θ F c)) (xs : CoinsOf ns) :
    (playOut ns θ F c σ xs).log.map Release.coins = CoinsOf.bits ns xs :=
  outK_coins ns _ _ xs

/-- The device's log at the coins `xs` when every round's release answers `b`. -/
def logOf (b : ByteArray) : (ns : List ℕ) → CoinsOf ns → List Release
  | [], _ => []
  | _ :: ns, (x, xs) => ⟨b, List.ofFn x⟩ :: logOf b ns xs

/-- **Forwarding through the rounds `ns`, then `K`**: send `b` before every round, then play `t` at the device's
log. -/
def forwardK (b : ByteArray) : (ns : List ℕ) → (K : List Release → Game α) → (∀ log, Game.Strategy (K log)) →
    Game.Strategy (releasesK ns K)
  | [], _, t => t []
  | _ :: ns, K, t => ⟨b, fun x => forwardK b ns (fun log => K (⟨b, List.ofFn x⟩ :: log))
      fun log => t (⟨b, List.ofFn x⟩ :: log)⟩

theorem outK_forwardK (b : ByteArray) : ∀ (ns : List ℕ) (K : List Release → Game α)
    (t : ∀ log, Game.Strategy (K log)) (xs : CoinsOf ns),
    outK ns K (forwardK b ns K t) xs = ⟨logOf b ns xs, t (logOf b ns xs)⟩
  | [], _, _, _ => rfl
  | _ :: ns, K, t, (x, xs) => by
      simp only [outK, forwardK, logOf]
      rw [outK_forwardK b ns]

/-- **The forwarding strategy**: it sends `b` before every round, and then, as its evidence, `enc` of the device's
log, as a prover that forwards the device's records does. -/
def forward (ns : List ℕ) (θ F c b : ByteArray) (enc : List Release → ByteArray) : Game.Strategy (play ns θ F c) :=
  cast (congrArg Game.Strategy (releases_bind ns _)).symm
    (forwardK b ns (fun log => Game.send ByteArray fun e => Game.ret (⟨θ, F, c, log, e⟩ : Outcome))
      fun log => ⟨enc log, PUnit.unit⟩)

/-- **The forwarding strategy's play**: the device's log, and `enc` of it as the evidence. -/
theorem playOut_forward (ns : List ℕ) (θ F c b : ByteArray) (enc : List Release → ByteArray) (xs : CoinsOf ns) :
    playOut ns θ F c (forward ns θ F c b enc) xs = ⟨θ, F, c, logOf b ns xs, enc (logOf b ns xs)⟩ := by
  simp only [playOut, forward, cast_cast, cast_eq]
  rw [outK_forwardK]

end Coins

section

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}

/-- The period run `r` with the verdicts `A`: at a coupling, `A` is its verdict at the code (`Vocab.Coupling.verdict`). -/
def atCode (r : PeriodRun d α C Ω) (A : Audit → ((i : I) → Ω i) → Prop) : PeriodRun d α C Ω :=
  { r with accepts := A }

/-- `atCode` doesn't read the run's own verdicts. -/
theorem atCode_withAccepts (r : PeriodRun d α C Ω) (M : Audit → ((i : I) → Ω i) → Prop)
    (A : Audit → ((i : I) → Ω i) → Prop) : atCode { r with accepts := M } A = atCode r A := rfl

/-- `atCode`'s claims are the run's. -/
theorem holds_atCode (r : PeriodRun d α C Ω) (A : Audit → ((i : I) → Ω i) → Prop) (c : Certificates) (p : Price)
    (a : Audit) (ξ : (i : I) → Ω i) : (atCode r A).Holds c p a ξ ↔ r.Holds c p a ξ := by
  cases a <;> exact Iff.rfl

variable [DecidableEq I] [∀ i, Fintype (Ω i)]

/-- A per-period guarantee passes to `atCode`'s verdicts `A` when `A` implies the run's verdict. -/
theorem sound_atCode_of_imp {r : PeriodRun d α C Ω} {A : Audit → ((i : I) → Ω i) → Prop} {c : Certificates} {p : Price}
    {a : Audit} {δ : ℝ} (h : ∀ ξ, A a ξ → r.accepts a ξ) (hr : r.Sound c p a δ) :
    (atCode r A).Sound c p a δ := fun ξ =>
  le_trans (Pouw.pr_mono fun _ hx => ⟨h _ hx.1, (holds_atCode r A c p a _).not.1 hx.2⟩) (hr ξ)

/-! ## Work -/

/-- **PoUW's refinement: its compiled component accepts only where the source game does.** At every
value `ξ` of the period's coins where PoUW's verdict `A .pouw` holds (at the code, `Vocab.Coupling.verdict`: `PouwCheck.Period.protocol`
accepts its case in what the prover sent, every release it read is logged, and the coupling's break is absent), the source game the
realization names (`acc`; for H1's checked credit, the play's acceptance) accepts at PoUW's coins there. -/
def PouwRefines {r : PeriodRun d α C Ω} {c : Certificates} {p : Price} {m u bUnit : ℕ} {X : Type}
    {acc : ((i : I) → Ω i) → X → Prop} {credit : ((i : I) → Ω i) → X → ℝ}
    (real : RealizesCreditedWords r c p m u bUnit X acc credit) (A : Audit → ((i : I) → Ω i) → Prop) : Prop :=
  ∀ ξ, A .pouw ξ → acc ξ (real.coin (ξ (r.own .pouw)))

/-- **PoUW at the code**: `WorkOfCreditedWords` at the model run `r`, whose PoUW verdict is the source game's
(`real`), and `PouwRefines`. -/
theorem pouw_atCode {r : PeriodRun d α C Ω} {c : Certificates} {p : Price} {m u bUnit : ℕ} {X : Type} [Fintype X]
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

/-- **The bandwidth audit's refinement: its component accepts only where the bandwidth audit's game does.** At
every value `ξ` of the period's coins where the bandwidth verdict `A .poub` holds (at the code, `Vocab.Coupling.verdict`:
onsite's bandwidth component, `RemoteAudit.Protocol.protocol`, accepts its case, its releases are logged, and the
coupling's break is absent), the game the realization names accepts its device at PoUB's coins there (the store, the oracle at the fresh
seed and the sample). -/
def PoubRefines {G N b h sb : ℕ} [Fintype ι] {r : PeriodRun d α C Ω}
    (real : RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := b) (h := h) (s := sb) r)
    (A : Audit → ((i : I) → Ω i) → Prop) : Prop :=
  ∀ ξ, A .poub ξ →
    Bandwidth.Accepts (real.dev ξ) (real.coins (ξ (r.own .poub))).1 (real.coins (ξ (r.own .poub))).2.1
      (real.coins (ξ (r.own .poub))).2.2

/-- **The bandwidth audit at the code**: `BandwidthOfAuditPerRunPeriod` at the model run `r`, whose PoUB verdict is the
game's (`real`), and `PoubRefines`. -/
theorem poub_atCode {G N b h sb k lam : ℕ} [Fintype ι] {r : PeriodRun d α C Ω} {c : Certificates} {p : Price}
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

/-- **PoUS's refinement: its compiled component accepts only what the verifier's whole decision
accepts.** Where PoUS's verdict `A .pous` holds at `ξ` (at the code, `Vocab.Coupling.verdict`:
`PousCheck.RunProtocol.protocol` accepts its case in what the prover sent, every release it read is logged, and the
coupling's break is absent), the audit `P` the
period's data decodes to is accepted by `Pous.EndToEnd.Accepted` (setup's check and the run's) at the joint run's coins. -/
def PousAccepts {r : PeriodRun d α C Ω} {ΩW ΩQ : Type}
    {run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    {verdict : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Prop}
    (R : RealizesSpaceOfPeriodHeld r run verdict) (A : Audit → ((i : I) → Ω i) → Prop) (P : Audits ΩW ΩQ) : Prop :=
  PousAtCode R A P.Accepts

/-- **What `SpaceOfPeriodHeld`'s verdict asks beyond acceptance: no PoUS break where PoUS's verdict holds.** Where
`A .pous` holds at `ξ`, the period's own data holds no break (`Audits.Break`: a SHA-512 collision among the strings the
run hashes, or an LMS forgery under a key of the verifier's list). No check refuses a break, so this is not a property
of the code; at a coupling whose break (`Vocab.Coupling.brk`) is PoUS's it holds by construction
(`Vocab.Coupling.pousNoBreak`), and the break goes to PoUS's finder (`Vocab.Coupling.BreakBound`). -/
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

/-- **PoUS at the code**: `SpaceOfPeriodHeld` at `atCode r A`, from a realization at `r` (its own
`accepts` not read), `PousAccepts` and `PousNoBreak`. -/
theorem pous_atCode (hd : AcceptsDecodes) (hp : PublishesRoot) (hH : P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ENNReal) ^ 131))
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

/-- **The network certifier's refinement: the network certifier's per-window bound at its compiled component.** At every
window `i` and every value `ξ` of the other coins, the probability over the network audit's own coins that
its verdict `A .network` holds (at the code, `Vocab.Coupling.verdict`: `NetworkCheck.Period.protocol` accepts its case
in what the prover sent, its releases are logged, and the coupling's break is absent) while window `i` fails its good event
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

/-- **The network certifier at the code**: `CommOfPeriodLeakAccepted` at `atCode r A`, from a realization at `r`
(its own `window_le` not read) and `NetworkWindowLe`. -/
theorem network_atCode {r : PeriodRun d α C Ω} (c : Certificates) (p : Price)
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

end Subprotocol

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-! ## The targets from the component-level bounds -/

theorem target_eq (a : Audit) : V.Target a = V.TargetFor (V.Verdict a) a := rfl

/-- **A subprotocol's target from its component-level bound.** -/
theorem sound_of (a : Audit) (h : V.ComponentBound a) : V.Target a := fun θ F c σ =>
  le_trans (Game.prob_mono (p := fun o => V.Verdict a o ∧ ¬ V.Holds θ F c σ a o)
    (q := fun o => ComponentAccepts a o.sent ∧ ¬ V.Holds θ F c σ a o)
    (fun _ ⟨⟨_, _, hc⟩, hh⟩ => ⟨hc, hh⟩) _ σ) (h θ F c σ)

/-- **PoUW's target from its component-level bound** -/
theorem pouwSound_of (h : V.ComponentBound .pouw) : V.PouwSound := V.sound_of .pouw h

/-- **PoUS's target from its component-level bound** -/
theorem pousSound_of (h : V.ComponentBound .pous) : V.PousSound := V.sound_of .pous h

/-- **The bandwidth audit's target from its component-level bound** -/
theorem poubSound_of (h : V.ComponentBound .poub) : V.BandwidthAuditSound := V.sound_of .poub h

/-- **The network certifier's target from its component-level bound** -/
theorem networkSound_of (h : V.ComponentBound .network) : V.NetworkCertifierSound := V.sound_of .network h

/-- **At the honest deployment**: at registered files, where subprotocol `a`'s honest lemma holds, the deployment's
honest prover meets `ComponentBound`'s event with probability `0`, within `a`'s error, and the target's event with
probability `0`. -/
theorem sound_of_honest (a : Audit) (h : V.HonestHolds a) (θ F c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts a o.sent ∧ ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c)
        (V.honest θ F c) ≤ V.error a θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict a o ∧ ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) = 0 := by
  have h0 := h θ F c hreg
  have hc : Game.prob (fun o => ComponentAccepts a o.sent ∧ ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c)
      (V.honest θ F c) = 0 :=
    le_antisymm (le_trans (Game.prob_mono
      (p := fun o => ComponentAccepts a o.sent ∧ ¬ V.Holds θ F c (V.honest θ F c) a o)
      (q := fun o => ¬ V.Holds θ F c (V.honest θ F c) a o) (fun _ ho => ho.2) _ _) h0.le) zero_le
  refine ⟨hc.le.trans zero_le, le_antisymm (le_trans (Game.prob_mono
    (p := fun o => V.Verdict a o ∧ ¬ V.Holds θ F c (V.honest θ F c) a o)
    (q := fun o => ¬ V.Holds θ F c (V.honest θ F c) a o) (fun _ ho => ho.2) _ _) h0.le) zero_le⟩

/-- **At the honest deployment, PoUW**: `sound_of_honest` at `.pouw`. -/
theorem pouwSound_of_honest (h : V.HonestHolds .pouw) (θ F c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts .pouw o.sent ∧ ¬ V.Holds θ F c (V.honest θ F c) .pouw o) (V.game θ F c)
        (V.honest θ F c) ≤ V.error .pouw θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict .pouw o ∧ ¬ V.Holds θ F c (V.honest θ F c) .pouw o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest .pouw h θ F c hreg

/-- **At the honest deployment, PoUS**: `sound_of_honest` at `.pous`. -/
theorem pousSound_of_honest (h : V.HonestHolds .pous) (θ F c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts .pous o.sent ∧ ¬ V.Holds θ F c (V.honest θ F c) .pous o) (V.game θ F c)
        (V.honest θ F c) ≤ V.error .pous θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict .pous o ∧ ¬ V.Holds θ F c (V.honest θ F c) .pous o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest .pous h θ F c hreg

/-- **At the honest deployment, the bandwidth audit**: `sound_of_honest` at `.poub`. -/
theorem poubSound_of_honest (h : V.HonestHolds .poub) (θ F c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts .poub o.sent ∧ ¬ V.Holds θ F c (V.honest θ F c) .poub o) (V.game θ F c)
        (V.honest θ F c) ≤ V.error .poub θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict .poub o ∧ ¬ V.Holds θ F c (V.honest θ F c) .poub o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest .poub h θ F c hreg

/-- **At the honest deployment, the network certifier**: `sound_of_honest` at `.network`. -/
theorem networkSound_of_honest (h : V.HonestHolds .network) (θ F c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts .network o.sent ∧ ¬ V.Holds θ F c (V.honest θ F c) .network o) (V.game θ F c)
        (V.honest θ F c) ≤ V.error .network θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict .network o ∧ ¬ V.Holds θ F c (V.honest θ F c) .network o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest .network h θ F c hreg

/-- **The target fails for a verifier that accepts everything**: at a strategy on whose every play subprotocol `a`'s
claim fails, with its error there below one (so its finder meets `FinderCR`, and its collision term is below one). -/
theorem sound_of_nontrivial (a : Audit) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds θ F c σ a o) (hlt : V.error a θ F c σ < 1) : ¬ V.TargetFor (fun _ => True) a := by
  intro h
  have h1 := prob_play_eq_one (V.rounds θ) θ F c (fun o => (fun _ : Outcome => True) o ∧ ¬ V.Holds θ F c σ a o)
    (fun _ _ => ⟨trivial, hfail _⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h θ F c σ)).trans_lt hlt)

/-- **PoUW's target fails for a verifier that accepts everything**: `sound_of_nontrivial` at `.pouw`. -/
theorem pouwSound_of_nontrivial (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds θ F c σ .pouw o) (hlt : V.error .pouw θ F c σ < 1) : ¬ V.TargetFor (fun _ => True) .pouw :=
  V.sound_of_nontrivial .pouw θ F c σ hfail hlt

/-- **PoUS's target fails for a verifier that accepts everything**: `sound_of_nontrivial` at `.pous`. -/
theorem pousSound_of_nontrivial (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds θ F c σ .pous o) (hlt : V.error .pous θ F c σ < 1) : ¬ V.TargetFor (fun _ => True) .pous :=
  V.sound_of_nontrivial .pous θ F c σ hfail hlt

/-- **The bandwidth audit's target fails for a verifier that accepts everything**: `sound_of_nontrivial` at `.poub`. -/
theorem poubSound_of_nontrivial (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds θ F c σ .poub o) (hlt : V.error .poub θ F c σ < 1) : ¬ V.TargetFor (fun _ => True) .poub :=
  V.sound_of_nontrivial .poub θ F c σ hfail hlt

/-- **The network certifier's target fails for a verifier that accepts everything**: `sound_of_nontrivial` at
`.network`. -/
theorem networkSound_of_nontrivial (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds θ F c σ .network o) (hlt : V.error .network θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) .network :=
  V.sound_of_nontrivial .network θ F c σ hfail hlt

/-! ## The component-level bounds from the period guarantees -/

/-- **The outcome the coins `ξ` lead to** in the play of `σ`, through the bijection `e` of a period run's coins with
the device's rounds' coins. -/
def outcome {I : Type} {Ω : I → Type} (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (e : ((i : I) → Ω i) ≃ Subprotocol.CoinsOf (V.rounds θ)) (ξ : (i : I) → Ω i) : Outcome :=
  Subprotocol.playOut (V.rounds θ) θ F c σ (e ξ)

/-- **The game's probability of an event is the coins'**: a period run's coins in bijection `e` with the device's
rounds' coins realize the play of `σ`, each leading to the outcome its rounds' coins do. -/
theorem prob_eq_pr {I : Type} {Ω : I → Type} [Fintype I] [DecidableEq I] [∀ i, Fintype (Ω i)]
    (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (e : ((i : I) → Ω i) ≃ Subprotocol.CoinsOf (V.rounds θ))
    (q : Outcome → Prop) :
    Game.prob q (V.game θ F c) σ = ENNReal.ofReal (Pouw.pr fun ξ => q (V.outcome θ F c σ e ξ)) :=
  (Subprotocol.prob_play q (V.rounds θ) θ F c σ).trans ((Subprotocol.prCoin_eq_ofReal_pr _).trans
    (congrArg ENNReal.ofReal (Pouw.pr_equiv e fun xs => q (Subprotocol.playOut (V.rounds θ) θ F c σ xs)).symm))

/-- **The honest play meets `Logged`**: where `a`'s check reads back, from the evidence `enc log`, releases of `log`
at the rounds `pos` assigns them, the strategy that forwards the device's log as its evidence (`Subprotocol.forward`)
meets `Logged pos a` at every coin. -/
theorem logged_forward (θ F c b : ByteArray) (enc : List Release → ByteArray) (pos : ℕ → ℕ) (a : Audit)
    (h : ∀ log k r, (V.coinsRead a ⟨θ, c, enc log, #[F]⟩)[k]? = some r → log[pos k]? = some r)
    (xs : Subprotocol.CoinsOf (V.rounds θ)) :
    V.Logged pos a (Subprotocol.playOut (V.rounds θ) θ F c (Subprotocol.forward (V.rounds θ) θ F c b enc) xs) := by
  rw [Subprotocol.playOut_forward]
  exact h _

/-- **A period run coupled to the play of `σ` for subprotocol `a`**: a run of the coins model whose coins are in
bijection with the device's rounds' coins (`coins`), at a price, with the round of the device's log each release a
subprotocol's check reads sits at (`pos`), a break on the coins (`brk`, the subprotocol's events its finder answers
for), and whose claim for `a` implies the game's at the outcome the coins lead to (`world`). -/
structure Coupling (a : Audit) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) where
  C : Verity.Circuit
  I : Type
  Ω : I → Type
  [fintypeI : Fintype I]
  [decEqI : DecidableEq I]
  [fintypeΩ : ∀ i, Fintype (Ω i)]
  [nonemptyΩ : ∀ i, Nonempty (Ω i)]
  coins : ((i : I) → Ω i) ≃ Subprotocol.CoinsOf (V.rounds θ)
  run : PeriodRun d α C Ω
  price : Price
  pos : Audit → ℕ → ℕ
  brk : ((i : I) → Ω i) → Prop
  world : ∀ ξ, run.Holds (V.certOf θ) price a ξ → V.Holds θ F c σ a (V.outcome θ F c σ coins ξ)

attribute [instance] Coupling.fintypeI Coupling.decEqI Coupling.fintypeΩ Coupling.nonemptyΩ

namespace Coupling

variable {V} {a : Audit} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} (k : V.Coupling a θ F c σ)

/-- **The verdict at the code**: subprotocol `b`'s component accepts its own case at the outcome the coins lead to,
every release its check read is the device's at its round (`pos b`), and the coupling's break is absent. -/
def verdict (b : Audit) (ξ : (i : k.I) → k.Ω i) : Prop :=
  ComponentAccepts b (V.outcome θ F c σ k.coins ξ).sent ∧ V.Logged (k.pos b) b (V.outcome θ F c σ k.coins ξ) ∧
    ¬ k.brk ξ

/-- The coupled run with the verdict at the code. -/
def atCode : PeriodRun d α k.C k.Ω := Subprotocol.atCode k.run k.verdict

/-- **The finder answers for the rest of the event**: the probability over the coins that `a`'s component accepts
while a release its check read is not the device's at its round, or the coupling's break holds, is at most `a`'s
collision term at the strategy (a release out of place is a forgery of the device's LMS signature, a break a collision,
each of which `a`'s finder outputs). -/
def BreakBound : Prop :=
  ENNReal.ofReal (Pouw.pr fun ξ => ComponentAccepts a (V.outcome θ F c σ k.coins ξ).sent ∧
      (¬ V.Logged (k.pos a) a (V.outcome θ F c σ k.coins ξ) ∨ k.brk ξ)) ≤
    V.loss a θ ((V.live θ F c).collides sha512Msg a σ)

/-- **A coupled period guarantee bounds the component's event in the game**: where the coupled run at the verdict at
the code meets a per-period guarantee within `a`'s published `δ` at the strategy's calls, and the finder answers for the
rest of the event where it meets `FinderCR`, `ComponentBound`'s event at `σ` is at most `a`'s error. -/
theorem bound {δ : ℝ} (hs : k.atCode.Sound (V.certOf θ) k.price a δ)
    (hδ : ENNReal.ofReal δ ≤ V.delta θ a (V.cost θ F c σ)) (hb : V.FinderCR a θ F c σ → k.BreakBound) :
    Game.prob (fun o => ComponentAccepts a o.sent ∧ ¬ V.Holds θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ := by
  unfold Vocab.error Accounting.condError
  split_ifs with hcr
  · classical
    show _ ≤ V.delta θ a (V.cost θ F c σ) + V.loss a θ ((V.live θ F c).collides sha512Msg a σ)
    refine (V.prob_eq_pr θ F c σ k.coins _).trans_le ?_
    refine le_trans (ENNReal.ofReal_le_ofReal (le_trans (Pouw.pr_mono
      (P' := fun ξ => (k.verdict a ξ ∧ ¬ V.Holds θ F c σ a (V.outcome θ F c σ k.coins ξ)) ∨
        (ComponentAccepts a (V.outcome θ F c σ k.coins ξ).sent ∧
          (¬ V.Logged (k.pos a) a (V.outcome θ F c σ k.coins ξ) ∨ k.brk ξ))) ?_) (Pouw.pr_or_le _ _))) ?_
    · intro ξ hξ
      by_cases hl : V.Logged (k.pos a) a (V.outcome θ F c σ k.coins ξ)
      · by_cases hk : k.brk ξ
        · exact Or.inr ⟨hξ.1, Or.inr hk⟩
        · exact Or.inl ⟨⟨hξ.1, hl, hk⟩, hξ.2⟩
      · exact Or.inr ⟨hξ.1, Or.inl hl⟩
    refine le_trans ENNReal.ofReal_add_le (add_le_add ?_ (hb hcr))
    refine le_trans (ENNReal.ofReal_le_ofReal ?_) hδ
    exact le_trans (Pouw.pr_mono fun ξ (h : k.verdict a ξ ∧ _) =>
        (⟨h.1, fun hh => h.2 (k.world ξ ((Subprotocol.holds_atCode k.run k.verdict _ _ a ξ).1 hh))⟩ :
          k.atCode.accepts a ξ ∧ ¬ k.atCode.Holds (V.certOf θ) k.price a ξ))
      (Pouw.pr_le_of_update (k.atCode.own a)
        (fun ξ => k.atCode.accepts a ξ ∧ ¬ k.atCode.Holds (V.certOf θ) k.price a ξ) hs)
  · exact le_top

section Pous

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period

set_option exponentiation.threshold 20000

/-- **`PousNoBreak` at a coupling whose break is PoUS's**: where the coupling's break holds at every `ξ` whose PoUS data
has a break at the realization's coins, the verdict at the code excludes it. -/
theorem pousNoBreak (kp : V.Coupling .pous θ F c σ) {ΩW ΩQ : Type}
    {run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    {verdict : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Prop}
    (R : RealizesSpaceOfPeriodHeld kp.run run verdict) (P : Audits ΩW ΩQ)
    (h : ∀ ξ, P.Break ((R.other ξ).1, (R.other ξ).2, R.coins (ξ (kp.run.own .pous))) → kp.brk ξ) :
    Subprotocol.PousNoBreak R kp.verdict P :=
  fun ξ hA hB => hA.2.2 (h ξ hB)

end Pous

end Coupling

/-- **`ComponentBound` from coupled period guarantees**: at every `θ F c σ`, a coupling for `a` whose run at the verdict
at the code meets a per-period guarantee within `a`'s published `δ` at the strategy's calls, and whose finder answers for
the rest of the event where it meets `FinderCR`. -/
theorem componentBound_of_coupling (a : Audit) (k : ∀ θ F c σ, V.Coupling a θ F c σ)
    (hs : ∀ θ F c σ, ∃ δ : ℝ, (k θ F c σ).atCode.Sound (V.certOf θ) (k θ F c σ).price a δ ∧
      ENNReal.ofReal δ ≤ V.delta θ a (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR a θ F c σ → (k θ F c σ).BreakBound) : V.ComponentBound a := by
  intro θ F c σ
  obtain ⟨δ, h1, h2⟩ := hs θ F c σ
  exact (k θ F c σ).bound h1 h2 (hb θ F c σ)

/-- **PoUW's target from coupled period guarantees** (`componentBound_of_coupling`, `pouwSound_of`). -/
theorem pouwSound_of_coupling (k : ∀ θ F c σ, V.Coupling .pouw θ F c σ)
    (hs : ∀ θ F c σ, ∃ δ : ℝ, (k θ F c σ).atCode.Sound (V.certOf θ) (k θ F c σ).price .pouw δ ∧
      ENNReal.ofReal δ ≤ V.delta θ .pouw (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .pouw θ F c σ → (k θ F c σ).BreakBound) : V.PouwSound :=
  V.pouwSound_of (V.componentBound_of_coupling .pouw k hs hb)

/-- **PoUS's target from coupled period guarantees** (`componentBound_of_coupling`, `pousSound_of`). -/
theorem pousSound_of_coupling (k : ∀ θ F c σ, V.Coupling .pous θ F c σ)
    (hs : ∀ θ F c σ, ∃ δ : ℝ, (k θ F c σ).atCode.Sound (V.certOf θ) (k θ F c σ).price .pous δ ∧
      ENNReal.ofReal δ ≤ V.delta θ .pous (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .pous θ F c σ → (k θ F c σ).BreakBound) : V.PousSound :=
  V.pousSound_of (V.componentBound_of_coupling .pous k hs hb)

/-- **The bandwidth audit's target from coupled period guarantees** (`componentBound_of_coupling`, `poubSound_of`). -/
theorem poubSound_of_coupling (k : ∀ θ F c σ, V.Coupling .poub θ F c σ)
    (hs : ∀ θ F c σ, ∃ δ : ℝ, (k θ F c σ).atCode.Sound (V.certOf θ) (k θ F c σ).price .poub δ ∧
      ENNReal.ofReal δ ≤ V.delta θ .poub (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .poub θ F c σ → (k θ F c σ).BreakBound) : V.BandwidthAuditSound :=
  V.poubSound_of (V.componentBound_of_coupling .poub k hs hb)

/-- **The network certifier's target from coupled period guarantees** (`componentBound_of_coupling`,
`networkSound_of`). -/
theorem networkSound_of_coupling (k : ∀ θ F c σ, V.Coupling .network θ F c σ)
    (hs : ∀ θ F c σ, ∃ δ : ℝ, (k θ F c σ).atCode.Sound (V.certOf θ) (k θ F c σ).price .network δ ∧
      ENNReal.ofReal δ ≤ V.delta θ .network (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .network θ F c σ → (k θ F c σ).BreakBound) : V.NetworkCertifierSound :=
  V.networkSound_of (V.componentBound_of_coupling .network k hs hb)

end Vocab

end Verity
