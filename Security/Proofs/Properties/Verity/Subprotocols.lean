import Properties.Verity.Subprotocols
import Proofs.Properties.Verity.Soundness
import Proofs.Core.Game.Prob
import Proofs.Core.UnionBound
import Proofs.Pouw.Prob
import Proofs.Accounting.JointBridge.WorkOfCreditedWords

/-!
# Each subprotocol's target from its bound at its component, and soundness from those bounds

`Verity.soundness_at_of_targets` takes each subprotocol's target at a composite `k` in the causal game (`Vocab.Target k
a`). Each follows from its bound at its compiled component (`Vocab.ComponentBound k a`, `Properties.Verity.Subprotocols`)
by monotonicity, since `Vocab.Verdict k a` includes `a`'s component's acceptance at `k`: one family over every composite
and subprotocol (`Vocab.sound_of`), at remote's three `sampledProofsSound_of`, `pouwSound_of` and
`inferenceOnlySound_of`. So the union bound over a composite's subprotocols holds from their component-level bounds
(`soundness_at_of_componentBounds`, `Verity.SoundnessOfComponents`), and a composite with one more subprotocol takes
one more `sound_of` instance in it.

Beside them, the two checks every guarantee carries: `sound_of_honest` (at registered files, where the subprotocol's
honest lemma `Vocab.HonestHolds k a` holds, the deployment's honest prover meets `ComponentBound`'s event and the
target's with probability `0`) and `sound_of_nontrivial` (the target fails for a verifier that accepts everything, at a
strategy on whose every play the claim fails and whose error is below one, so its finder meets `FinderCR` and its
forger `Unforgeable`), with their instances at remote's three. `soundness_at_of_componentBounds`' own witnesses are at the idle
deployment (`Properties.Verity.Idle`): its hypotheses hold at `Idle.unbounded` at remote's compiled composite
(`soundness_at_of_componentBounds_honest`), and its conclusion fails there at `Idle.exact` for a verifier that accepts
everything (`soundness_at_of_componentBounds_nontrivial`).

For a subprotocol that decides `Audit` `a` (`Subprotocol.ofAudit a`: sampled proofs and PoUW at remote),
`ComponentBound k (.ofAudit a)` comes from a per-period guarantee of the coins model (`Accounting.PeriodRun.Sound`, the
bridges `Accounting.SecurityProofs.Bridge.*`) through a coupling (`Vocab.Coupling k a`) at each `θ F c σ`. The
inference-only policy's claim is about the declaration and reads no coin, so its bound is its own
(`inferenceOnlySound_of`).

* **The coins.** A period run whose coins are in bijection with the device's rounds' coins
  (`Subprotocol.CoinsOf (V.rounds θ)`): every event of the game has the probability over the run's coins of the outcome
  they lead to (`Vocab.prob_eq_pr`, from `Subprotocol.prob_play`), so a bound over the coins is one in the game. The
  committed files come before any coin and the evidence's blobs with the evidence, so the outcome's `sent` is what
  the composite reads, the committed files then the evidence's blobs (`Exchange.sent`).
* **The world.** The run's claim for `a` at the coupling's price implies the game's at `k` at that outcome (`world`, the
  world each play realizes).
* **The verdict at the code** (`Coupling.verdict`): `a`'s component at `k` accepts, every release its check read is the
  device's at the round the coupling assigns it (`pos`, `Logged`), and the coupling's break (`brk`) is absent. The rest
  of the event, a release out of place or a break, is answered for by `a`'s error beyond `δ`: where its finder meets
  `FinderCR`, its probability is at most the probability that `a`'s forger forges plus its collision term
  (`Coupling.BreakBound`), and where the forger meets `Hash.Assumptions.LmsUnforgeable` (`Vocab.Unforgeable`), that is
  at most the published `ε_forge(t, Q)` (`Vocab.epsForge`).

`Coupling.bound` gives `ComponentBound`'s event at most `a`'s error from the run at the verdict at the code
(`Coupling.atCode`) meeting a guarantee within `a`'s published `δ`, and `componentBound_of_coupling` collects it over
`θ F c σ`. The guarantee at `atCode` comes from the bridge and one refinement hypothesis, what the verdict at the code
means for the bridge's verdict: for PoUW, `WorkOfCreditedWords` and `PouwRefines` (`Subprotocol.pouw_atCode`). Until the
refinement is proved of the code it is a hypothesis, a named `Prop`. The bridge reads the verdict through an
equivalence (`accepts_iff`), so it is taken at a model run whose verdict is the source game's, and `Sound` passes to the
verdicts `A` when they imply the model's (`sound_atCode_of_imp`). `atCode` ignores the run's own verdicts
(`atCode_withAccepts`), so the model run is any `{r with accepts := …}`.
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
def playOut (ns : List ℕ) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (play ns θ F c))
    (xs : CoinsOf ns) : Exchange :=
  let o := outK ns _ (cast (congrArg Game.Strategy (releases_bind ns _)) σ) xs
  ⟨θ, F, c, o.1, o.2.1.1, o.2.1.2⟩

theorem prob_play (q : Exchange → Prop) (ns : List ℕ) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (play ns θ F c)) :
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
  | _ :: ns, _, s, (x, xs) => congrArg (List.ofFn x :: ·) (outK_coins ns _ (s.2 x) xs)

/-- **The play's log holds each round's coins at its round**, whatever the strategy. -/
theorem playOut_coins (ns : List ℕ) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (play ns θ F c)) (xs : CoinsOf ns) :
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

/-- The game's last move after the log `log`: the evidence with its blobs, then the play's end. -/
abbrev evidenceK (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (log : List Release) : Game Exchange :=
  Game.send (ByteArray × Array ByteArray) fun e => Game.ret ⟨θ, F, c, log, e.1, e.2⟩

/-- **The forwarding strategy**: it sends `b` before every round, and then, as its evidence and the evidence's blobs,
`enc` of the device's log, as a prover that forwards the device's records does. -/
def forward (ns : List ℕ) (θ : ByteArray) (F : Array ByteArray) (c b : ByteArray)
    (enc : List Release → ByteArray × Array ByteArray) : Game.Strategy (play ns θ F c) :=
  cast (congrArg Game.Strategy (releases_bind ns _)).symm
    (forwardK b ns (evidenceK θ F c) fun log => ⟨enc log, PUnit.unit⟩)

/-- **The forwarding strategy's play**: the device's log, and `enc` of it as the evidence and its blobs. -/
theorem playOut_forward (ns : List ℕ) (θ : ByteArray) (F : Array ByteArray) (c b : ByteArray)
    (enc : List Release → ByteArray × Array ByteArray) (xs : CoinsOf ns) :
    playOut ns θ F c (forward ns θ F c b enc) xs =
      ⟨θ, F, c, logOf b ns xs, (enc (logOf b ns xs)).1, (enc (logOf b ns xs)).2⟩ := by
  have h : cast (congrArg Game.Strategy (releases_bind ns _)) (forward ns θ F c b enc) =
      forwardK b ns (evidenceK θ F c) fun log => ⟨enc log, PUnit.unit⟩ :=
    eq_of_heq ((cast_heq _ _).trans (cast_heq _ _))
  have hout := outK_forwardK b ns (evidenceK θ F c) (fun log => ⟨enc log, PUnit.unit⟩) xs
  unfold playOut
  rw [h]
  exact congrArg (fun o : (log : List Release) × Game.Strategy (evidenceK θ F c log) =>
    (⟨θ, F, c, o.1, o.2.1.1, o.2.1.2⟩ : Exchange)) hout

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
value `ξ` of the period's coins where PoUW's verdict `A .pouw` holds (at the code, `Vocab.Coupling.verdict`: the
composite's PoUW component, `Verity.component k .pouw`, accepts its case in
what the prover sent, every release it read is logged, and the coupling's break is absent), the source game the
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

end

end Subprotocol

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-! ## The targets from the component-level bounds -/

theorem target_eq (k : Verity.Registration.Composite) (a : Subprotocol) :
    V.Target k a = V.TargetFor (V.Verdict k a) k a := rfl

/-- **A subprotocol's target from its component-level bound.** -/
theorem sound_of (k : Verity.Registration.Composite) (a : Subprotocol) (h : V.ComponentBound k a) : V.Target k a :=
  fun θ F c σ =>
    le_trans (Game.prob_mono (p := fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o)
      (q := fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c σ a o)
      (fun _ ⟨⟨_, _, hc⟩, hh⟩ => ⟨hc, hh⟩) _ σ) (h θ F c σ)

/-- **At the honest deployment**: at registered files, where subprotocol `a`'s honest lemma at composite `k` holds, the
deployment's honest prover meets `ComponentBound`'s event with probability `0`, within `a`'s error, and the target's
event with probability `0`. -/
theorem sound_of_honest (k : Verity.Registration.Composite) (a : Subprotocol) (h : V.HonestHolds k a)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c)
        (V.honest θ F c) ≤ V.error a θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c)
        (V.honest θ F c) = 0 := by
  have h0 := h θ F c hreg
  have hc : Game.prob (fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
      (V.game θ F c) (V.honest θ F c) = 0 :=
    le_antisymm (le_trans (Game.prob_mono
      (p := fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
      (q := fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (fun _ ho => ho.2) _ _) h0.le) zero_le
  refine ⟨hc.le.trans zero_le, le_antisymm (le_trans (Game.prob_mono
    (p := fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
    (q := fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (fun _ ho => ho.2) _ _) h0.le) zero_le⟩

/-- **The target fails for a verifier that accepts everything**: at a strategy on whose every play subprotocol `a`'s
claim at composite `k` fails, with its error there below one (so its finder meets `FinderCR` and its forger
`Unforgeable`, and its `ε_forge` and collision terms are below one). -/
theorem sound_of_nontrivial (k : Verity.Registration.Composite) (a : Subprotocol) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ a o) (hlt : V.error a θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k a := by
  intro h
  have h1 := prob_play_eq_one (V.rounds θ) θ F c
    (fun o => (fun _ : Exchange => True) o ∧ ¬ V.Holds k θ F c σ a o) (fun _ _ => ⟨trivial, hfail _⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h θ F c σ)).trans_lt hlt)

section

variable {k : Verity.Registration.Composite}

/-- **Sampled proofs' target from its component-level bound** -/
theorem sampledProofsSound_of (h : V.ComponentBound k .sampled) : V.SampledProofsSound k := V.sound_of k .sampled h

/-- **PoUW's target from its component-level bound** -/
theorem pouwSound_of (h : V.ComponentBound k .pouw) : V.PouwSound k := V.sound_of k .pouw h

/-- **The inference-only policy's target from its component-level bound** -/
theorem inferenceOnlySound_of (h : V.ComponentBound k .inferenceOnly) : V.InferenceOnlySound k :=
  V.sound_of k .inferenceOnly h

/-- **At the honest deployment, sampled proofs**: `sound_of_honest` at `.sampled`. -/
theorem sampledProofsSound_of_honest (h : V.HonestHolds k .sampled) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k .sampled o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) .sampled o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error .sampled θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c (V.honest θ F c) .sampled o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest k .sampled h θ F c hreg

/-- **At the honest deployment, PoUW**: `sound_of_honest` at `.pouw`. -/
theorem pouwSound_of_honest (h : V.HonestHolds k .pouw) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k .pouw o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) .pouw o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error .pouw θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k .pouw o ∧ ¬ V.Holds k θ F c (V.honest θ F c) .pouw o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest k .pouw h θ F c hreg

/-- **At the honest deployment, the inference-only policy**: `sound_of_honest` at `.inferenceOnly`. -/
theorem inferenceOnlySound_of_honest (h : V.HonestHolds k .inferenceOnly) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k .inferenceOnly o.sent ∧
          ¬ V.Holds k θ F c (V.honest θ F c) .inferenceOnly o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error .inferenceOnly θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k .inferenceOnly o ∧ ¬ V.Holds k θ F c (V.honest θ F c) .inferenceOnly o)
        (V.game θ F c) (V.honest θ F c) = 0 :=
  V.sound_of_honest k .inferenceOnly h θ F c hreg

/-- **Sampled proofs' target fails for a verifier that accepts everything**: `sound_of_nontrivial` at `.sampled`. -/
theorem sampledProofsSound_of_nontrivial (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ .sampled o) (hlt : V.error .sampled θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k .sampled :=
  V.sound_of_nontrivial k .sampled θ F c σ hfail hlt

/-- **PoUW's target fails for a verifier that accepts everything**: `sound_of_nontrivial` at `.pouw`. -/
theorem pouwSound_of_nontrivial (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ .pouw o) (hlt : V.error .pouw θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k .pouw :=
  V.sound_of_nontrivial k .pouw θ F c σ hfail hlt

/-- **The inference-only policy's target fails for a verifier that accepts everything**: `sound_of_nontrivial` at
`.inferenceOnly`. -/
theorem inferenceOnlySound_of_nontrivial (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ .inferenceOnly o) (hlt : V.error .inferenceOnly θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k .inferenceOnly :=
  V.sound_of_nontrivial k .inferenceOnly θ F c σ hfail hlt

end

/-! ## The component-level bounds from the period guarantees -/

/-- **The outcome the coins `ξ` lead to** in the play of `σ`, through the bijection `e` of a period run's coins with
the device's rounds' coins. -/
def outcome {I : Type} {Ω : I → Type} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (e : ((i : I) → Ω i) ≃ Subprotocol.CoinsOf (V.rounds θ)) (ξ : (i : I) → Ω i) : Exchange :=
  Subprotocol.playOut (V.rounds θ) θ F c σ (e ξ)

/-- **The game's probability of an event is the coins'**: a period run's coins in bijection `e` with the device's
rounds' coins realize the play of `σ`, each leading to the outcome its rounds' coins do. -/
theorem prob_eq_pr {I : Type} {Ω : I → Type} [Fintype I] [DecidableEq I] [∀ i, Fintype (Ω i)]
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (e : ((i : I) → Ω i) ≃ Subprotocol.CoinsOf (V.rounds θ))
    (q : Exchange → Prop) :
    Game.prob q (V.game θ F c) σ = ENNReal.ofReal (Pouw.pr fun ξ => q (V.outcome θ F c σ e ξ)) :=
  (Subprotocol.prob_play q (V.rounds θ) θ F c σ).trans ((Subprotocol.prCoin_eq_ofReal_pr _).trans
    (congrArg ENNReal.ofReal (Pouw.pr_equiv e fun xs => q (Subprotocol.playOut (V.rounds θ) θ F c σ xs)).symm))

/-- **The honest play meets `Logged`**: where `a`'s check reads back, from the evidence and its blobs `enc log` after
the committed files, releases of `log` at the rounds `pos` assigns them, the strategy that forwards the device's log as
its evidence (`Subprotocol.forward`) meets `Logged pos a` at every coin. -/
theorem logged_forward (θ : ByteArray) (F : Array ByteArray) (c b : ByteArray)
    (enc : List Release → ByteArray × Array ByteArray) (pos : ℕ → ℕ) (a : Subprotocol)
    (h : ∀ log k r, (V.coinsRead a ⟨θ, c, (enc log).1, F ++ (enc log).2⟩)[k]? = some r → log[pos k]? = some r)
    (xs : Subprotocol.CoinsOf (V.rounds θ)) :
    V.Logged pos a (Subprotocol.playOut (V.rounds θ) θ F c (Subprotocol.forward (V.rounds θ) θ F c b enc) xs) := by
  rw [Subprotocol.playOut_forward]
  exact h _

/-- **A period run coupled to the play of `σ` for the subprotocol that decides `Audit` `a` at composite `k`**: a run of
the coins model whose coins are in bijection with the device's rounds' coins (`coins`), at a price, with the round of
the device's log each release a subprotocol's check reads sits at (`pos`), a break on the coins (`brk`, the
subprotocol's events its finder answers for), whose claim for `a` implies the game's at `k` at the outcome the coins
lead to (`world`). -/
structure Coupling (k : Verity.Registration.Composite) (a : Audit) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) where
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
  world : ∀ ξ, run.Holds (V.certOf θ) price a ξ → V.Holds k θ F c σ (.ofAudit a) (V.outcome θ F c σ coins ξ)

attribute [instance] Coupling.fintypeI Coupling.decEqI Coupling.fintypeΩ Coupling.nonemptyΩ

namespace Coupling

variable {V} {k : Verity.Registration.Composite} {a : Audit} {θ c : ByteArray} {F : Array ByteArray}
  {σ : Game.Strategy (V.game θ F c)}
  (q : V.Coupling k a θ F c σ)

/-- **The verdict at the code**: the component at composite `k` of the subprotocol that decides `b` accepts its own case
at the outcome the coins lead to, every release its check read is the device's at its round (`pos b`), and the
coupling's break is absent. -/
def verdict (b : Audit) (ξ : (i : q.I) → q.Ω i) : Prop :=
  ComponentAccepts k (.ofAudit b) (V.outcome θ F c σ q.coins ξ).sent ∧
    V.Logged (q.pos b) (.ofAudit b) (V.outcome θ F c σ q.coins ξ) ∧ ¬ q.brk ξ

/-- The coupled run with the verdict at the code. -/
def atCode : PeriodRun d α q.C q.Ω := Subprotocol.atCode q.run q.verdict

/-- **The error beyond `δ` answers for the rest of the event**: the probability over the coins that `a`'s component
accepts while a release its check read is not the device's at its round, or the coupling's break holds, is at most the
probability that `a`'s forger forges under the compiled LMS verifier (`Vocab.forger`, `lmsVerify`) plus `a`'s collision
term at the strategy (a release out of place is a forgery of the device's LMS signature over its ledger or a session's
close, which the forger outputs, and a break a collision, which `a`'s finder outputs). Where the forger meets
`Hash.Assumptions.LmsUnforgeable` (`Vocab.Unforgeable`), its probability is at most the published `ε_forge(t, Q)`
(`bound`). -/
def BreakBound : Prop :=
  ENNReal.ofReal (Pouw.pr fun ξ => ComponentAccepts k (.ofAudit a) (V.outcome θ F c σ q.coins ξ).sent ∧
      (¬ V.Logged (q.pos a) (.ofAudit a) (V.outcome θ F c σ q.coins ξ) ∨ q.brk ξ)) ≤
    ENNReal.ofReal ((V.forger (.ofAudit a) θ F c σ).Pr lmsVerify) +
      V.loss (.ofAudit a) θ (ENNReal.ofReal ((V.finder (.ofAudit a) θ F c σ).Pr sha512Msg))

/-- **A coupled period guarantee bounds the component's event in the game**: where the coupled run at the verdict at
the code meets a per-period guarantee within `a`'s published `δ` at the strategy's calls, and the error beyond `δ`
answers for the rest of the event where the finder meets `FinderCR`, `ComponentBound`'s event at `σ` is at most `a`'s
error (where the forger also meets `Unforgeable`, its forging probability is at most `epsForge` at its budget and the
device's signatures, `forger_calls` and `forger_signed`). -/
theorem bound {δ : ℝ} (hs : q.atCode.Sound (V.certOf θ) q.price a δ)
    (hδ : ENNReal.ofReal δ ≤ V.delta θ (.ofAudit a) (V.cost θ F c σ))
    (hb : V.FinderCR (.ofAudit a) θ F c σ → q.BreakBound) :
    Game.prob (fun o => ComponentAccepts k (.ofAudit a) o.sent ∧ ¬ V.Holds k θ F c σ (.ofAudit a) o)
      (V.game θ F c) σ ≤ V.error (.ofAudit a) θ F c σ := by
  unfold Vocab.error Accounting.condError
  split_ifs with hcr
  · classical
    obtain ⟨hcr, hun⟩ := hcr
    have hf : ENNReal.ofReal ((V.forger (.ofAudit a) θ F c σ).Pr lmsVerify) ≤
        ENNReal.ofReal (V.epsForge θ (V.callsMul θ (.ofAudit a) * V.cost θ F c σ + V.callsAdd θ (.ofAudit a))
          (V.signatures θ)) :=
      ENNReal.ofReal_le_ofReal (hun (V.forger_calls _ θ F c σ) (V.forger_signed _ θ F c σ))
    show _ ≤ V.delta θ (.ofAudit a) (V.cost θ F c σ) +
      ENNReal.ofReal (V.epsForge θ (V.callsMul θ (.ofAudit a) * V.cost θ F c σ + V.callsAdd θ (.ofAudit a))
        (V.signatures θ)) +
      V.loss (.ofAudit a) θ (ENNReal.ofReal ((V.finder (.ofAudit a) θ F c σ).Pr sha512Msg))
    rw [add_assoc]
    refine (V.prob_eq_pr θ F c σ q.coins _).trans_le ?_
    refine le_trans (ENNReal.ofReal_le_ofReal (le_trans (Pouw.pr_mono
      (P' := fun ξ => (q.verdict a ξ ∧ ¬ V.Holds k θ F c σ (.ofAudit a) (V.outcome θ F c σ q.coins ξ)) ∨
        (ComponentAccepts k (.ofAudit a) (V.outcome θ F c σ q.coins ξ).sent ∧
          (¬ V.Logged (q.pos a) (.ofAudit a) (V.outcome θ F c σ q.coins ξ) ∨ q.brk ξ))) ?_)
      (Pouw.pr_or_le _ _))) ?_
    · intro ξ hξ
      by_cases hl : V.Logged (q.pos a) (.ofAudit a) (V.outcome θ F c σ q.coins ξ)
      · by_cases hk : q.brk ξ
        · exact Or.inr ⟨hξ.1, Or.inr hk⟩
        · exact Or.inl ⟨⟨hξ.1, hl, hk⟩, hξ.2⟩
      · exact Or.inr ⟨hξ.1, Or.inl hl⟩
    refine le_trans ENNReal.ofReal_add_le (add_le_add ?_ ((hb hcr).trans (add_le_add hf le_rfl)))
    refine le_trans (ENNReal.ofReal_le_ofReal ?_) hδ
    exact le_trans (Pouw.pr_mono fun ξ (h : q.verdict a ξ ∧ _) =>
        (⟨h.1, fun hh => h.2 (q.world ξ ((Subprotocol.holds_atCode q.run q.verdict _ _ a ξ).1 hh))⟩ :
          q.atCode.accepts a ξ ∧ ¬ q.atCode.Holds (V.certOf θ) q.price a ξ))
      (Pouw.pr_le_of_update (q.atCode.own a)
        (fun ξ => q.atCode.accepts a ξ ∧ ¬ q.atCode.Holds (V.certOf θ) q.price a ξ) hs)
  · exact le_top

end Coupling

/-- **`ComponentBound` from coupled period guarantees**: at every `θ F c σ`, a coupling at composite `k` for `a` whose
run at the verdict at the code meets a per-period guarantee within `a`'s published `δ` at the strategy's calls, and
whose error beyond `δ` answers for the rest of the event where the finder meets `FinderCR`. -/
theorem componentBound_of_coupling (k : Verity.Registration.Composite) (a : Audit)
    (q : ∀ θ F c σ, V.Coupling k a θ F c σ)
    (hs : ∀ θ F c σ, ∃ δ : ℝ, (q θ F c σ).atCode.Sound (V.certOf θ) (q θ F c σ).price a δ ∧
      ENNReal.ofReal δ ≤ V.delta θ (.ofAudit a) (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR (.ofAudit a) θ F c σ → (q θ F c σ).BreakBound) :
    V.ComponentBound k (.ofAudit a) := by
  intro θ F c σ
  obtain ⟨δ, h1, h2⟩ := hs θ F c σ
  exact (q θ F c σ).bound h1 h2 (hb θ F c σ)

/-- **PoUW's target from coupled period guarantees** (`componentBound_of_coupling`, `pouwSound_of`). -/
theorem pouwSound_of_coupling {k : Verity.Registration.Composite} (q : ∀ θ F c σ, V.Coupling k .pouw θ F c σ)
    (hs : ∀ θ F c σ, ∃ δ : ℝ, (q θ F c σ).atCode.Sound (V.certOf θ) (q θ F c σ).price .pouw δ ∧
      ENNReal.ofReal δ ≤ V.delta θ .pouw (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .pouw θ F c σ → (q θ F c σ).BreakBound) : V.PouwSound k :=
  V.pouwSound_of (V.componentBound_of_coupling k .pouw q hs hb)

end Vocab

/-- **Soundness at every composite, from its subprotocols' bounds at their components**: each bound gives its
subprotocol's target (`Vocab.sound_of`), and the targets the union bound over the composite's subprotocols
(`soundness_at_of_targets`). -/
theorem soundness_at_of_componentBounds : SoundnessOfComponents := by
  intro ι α _ d V k h
  exact soundness_at_of_targets ι α d V k fun a ha => V.sound_of k a (h a ha)

namespace Idle

/-- At `unbounded`, every subprotocol meets its bound at its component at every composite. -/
theorem componentBound_unbounded (k : Verity.Registration.Composite) (a : Subprotocol) :
    unbounded.ComponentBound k a := by
  intro θ F c σ
  rw [error_unbounded]
  exact le_top

end Idle

/-- **`SoundnessOfComponents`' honest lemma**: its hypotheses hold at the idle deployment's compiled remote composite,
sampled proofs, PoUW and the inference-only policy each meeting its bound at its component. -/
theorem soundness_at_of_componentBounds_honest :
    Idle.unbounded.ComponentBound .remote .sampled ∧ Idle.unbounded.ComponentBound .remote .pouw ∧
      Idle.unbounded.ComponentBound .remote .inferenceOnly :=
  ⟨Idle.componentBound_unbounded _ _, Idle.componentBound_unbounded _ _, Idle.componentBound_unbounded _ _⟩

/-- **`SoundnessOfComponents`' nontrivial lemma**: its conclusion fails at remote for a verifier that accepts
everything, at the idle deployment whose errors are all `0`. -/
theorem soundness_at_of_componentBounds_nontrivial : ¬ Idle.exact.SoundnessFor .remote fun _ => True :=
  Idle.nontrivial_at .remote (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl⟩)

end Verity
