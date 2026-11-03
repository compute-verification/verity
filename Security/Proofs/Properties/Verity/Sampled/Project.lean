import Proofs.Properties.Verity.Sampled.Coupling
import Proofs.Core.Game.InterleaveBridge

/-!
# Sampled proofs' coupling: the projection

`Constructions.project`, the strategy of the live audit a strategy `σ` of the period's game is read as: it simulates
`σ` (`Verity.Game.Pos`, in `Proofs.Core.Game.InterleaveBridge`). It registers the decoding of the record's most likely
message; on the draw's widened sample it starts a tape of the period's game (the coin map's `start`); at each stop, the
first node of `σ`'s play on the tape where a round's cut (`roundCut`) fixes a message, it sends the decoding of the most
likely message there (`Commit`'s roots at the first stop, a round's submission at each later one) and writes the live
coin it gets back into the tape there (the coin map's `link` and `round`); after the last round it sends the proof files
decoded from the play's output on the tape.

* **The parameters** (`Decoders`, `CoinMap`): the decoders, and the coin map with the widening of the draw's sample
  it reads. They are the projection's two inputs: what each stop's message decodes to, and how the live coins become
  the period's.
* **The projection** (`projectOf`, `Constructions.withProject`), and **the joint law** it simulates (`jointG`, `jointS`):
  the projected play's outcome beside the period's tape it plays `σ` on, over the live audit's coins. The projected
  play's events have their probabilities under it (`prob_projectOf`).
* **The two premises** the logged coupling takes at it: the joint law's tape is, at every event, within the hiding
  term of the uniform tape (`Couples`, the coin map's), and on the joint law's support an accepted play `P` holds of
  that misses nowhere has an outcome flock-verify `--zk` accepts (`Decodes`, the decoders'). Off them the logged
  coupling holds at the projection (`loggedCoupling_withProject`), so `CouplingTarget` holds at the constructions with
  the projection and the messages' finders wherever the binding's bound does (`couplingTarget_withProject`).
* **What the joint law's support holds** (`jointG_reaches`): its tape is one through every stop the projection read,
  each cut fixed at a stop has there, on that tape, the message the stop decoded (`Verity.Game.Pos.modeOn_eq`), and
  the proof files are the decoding of the play's output on it.
-/

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Refine (roundOf)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The parameters -/

/-- **The decoders a projection reads**: the registration, from the record's most likely message (`none` where the
record's cut fixes none); `Commit`'s roots and a round's submission at `J` tables, from the most likely message of each
round whose cut is fixed at a stop (`none` at the others); and the proof files, from the play's output. -/
structure Decoders (R : ℕ) (Reg : Type) where
  reg : Option Claims.Msg → Reg
  root : (Fin R → Option Claims.Msg) → Array ByteArray
  sub : (J : ℕ) → (Fin R → Option Claims.Msg) → Option (SubmitJ J)
  proofs : Exchange → Array ByteArray

/-- **The coin map a projection reads**, at a strategy `σ` of the period's game `g`: the widening of the draw's
sample (`T`); the period's tape on the draw's sample and the widening (`start`); and at a stop, the period's tape from
there with the live link points (`link`) or a round's live coins (`round`) written in. -/
structure CoinMap (g : Game Exchange) (σ : Strategy g) (Ω : Type) (R mPts : ℕ) where
  T : Type
  [fT : Fintype T]
  [nT : Nonempty T]
  start : Ω × T → CoinTape g σ
  link : (p : Pos g σ Claims.Msg (Fin R)) → (Fin 2 → Fin mPts → 𝔽) → CoinTape p.g' p.s' → CoinTape p.g' p.s'
  round : (p : Pos g σ Claims.Msg (Fin R)) → (J : ℕ) → (s : SubmitJ J) → (Fin s.n → 𝔽) → CoinTape p.g' p.s' →
    CoinTape p.g' p.s'

attribute [instance] CoinMap.fT CoinMap.nT

/-- The widening a coin map reads. -/
def CoinMap.widening {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ} (M : CoinMap g σ Ω R mPts) :
    Widening :=
  ⟨M.T⟩

/-! ## The projection on the live session -/

section Session

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : Decoders R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ)

/-- **One round's projection** at the stop `q`, at the submission `o` decoded there: the play's end, with the proof
files decoded from the play's output on the tape, or the submission and then `next` on its coins. -/
noncomputable def stepP (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R))
    (next : (s : SubmitJ J) → (c : Fin s.n → 𝔽) → FlockSoundness.Game.Strategy
      ((liveRoundsJ dg J N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))).bind (KLJ rootB pts))) :
    Option (SubmitJ J) → FlockSoundness.Game.Strategy ((liveRoundsJ dg J (N + 1) rs).bind (KLJ rootB pts))
  | none => ⟨none, ⟨D.proofs (CoinTape.run g σ q.tape), PUnit.unit⟩⟩
  | some s => ⟨some s, next s⟩

/-- **The rounds' projection** from the position `p`: at the next stop, the decoded submission; on its coins, written
into the tape there, the next round; after the last, the proof files decoded from the play's output on the tape. -/
noncomputable def roundsP (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) →
      FlockSoundness.Game.Strategy ((liveRoundsJ dg J N rs).bind (KLJ rootB pts))
  | 0, _, p => ⟨D.proofs (CoinTape.run g σ p.tape), PUnit.unit⟩
  | N + 1, rs, p => stepP D dg J rootB pts N rs p.toStop
    (fun s c => roundsP rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
      (p.toStop.rewrite (M.round p.toStop J s c)).consume) (D.sub J p.toStop.modes)

/-- **One round's joint law** at the stop `q`, at the submission `o` decoded there: the play's end beside the tape, or
the round's coins and then `next`. -/
noncomputable def stepW {β : Type} (fin : LiveOutJ J mPts → β) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R))
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β × CoinTape g σ)) :
    Option (SubmitJ J) → Game (β × CoinTape g σ)
  | none => .ret (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ q.tape)⟩, q.tape)
  | some s => .coin (Fin s.n → 𝔽) (next s)

/-- One round's joint law's strategy: it has no message to send. -/
def stepWS {β : Type} (fin : LiveOutJ J mPts → β) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R))
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β × CoinTape g σ)) (w : ∀ s c, Strategy (next s c)) :
    (o : Option (SubmitJ J)) → Strategy (stepW D J fin rootB pts rs q next o)
  | none => PUnit.unit
  | some s => w s

/-- **The rounds' joint law** from the position `p`: the live session's outcome, relabelled by `fin`, beside the
period's tape, over the rounds' live coins. -/
noncomputable def roundsW {β : Type} (fin : LiveOutJ J mPts → β) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → Game (β × CoinTape g σ)
  | 0, rs, p => .ret (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.tape)⟩, p.tape)
  | N + 1, rs, p => stepW D J fin rootB pts rs p.toStop
    (fun s c => roundsW fin rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
      (p.toStop.rewrite (M.round p.toStop J s c)).consume) (D.sub J p.toStop.modes)

/-- The rounds' joint law's strategy: it has no message to send. -/
noncomputable def roundsWS {β : Type} (fin : LiveOutJ J mPts → β) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → (p : Pos g σ Claims.Msg (Fin R)) →
      Strategy (roundsW D M dg J fin rootB pts N rs p)
  | 0, _, _ => PUnit.unit
  | N + 1, rs, p => stepWS D J fin rootB pts rs p.toStop _ (fun _ _ => roundsWS fin rootB pts N _ _)
    (D.sub J p.toStop.modes)

/-- **The session's projection** from the position `p`: at the next stop, `Commit`'s decoded roots; on the link points,
written into the tape there, the rounds. -/
noncomputable def sessionP (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    FlockSoundness.Game.Strategy (liveSession dg J mPts N) :=
  ⟨D.root p.toStop.modes, fun pts => roundsP D M dg J (D.root p.toStop.modes) pts N (fun _ _ => [])
    (p.toStop.rewrite (M.link p.toStop pts)).consume⟩

/-- **The session's joint law** from the position `p`. -/
noncomputable def sessionW {β : Type} (fin : LiveOutJ J mPts → β) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    Game (β × CoinTape g σ) :=
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts => roundsW D M dg J fin (D.root p.toStop.modes) pts N (fun _ _ => [])
    (p.toStop.rewrite (M.link p.toStop pts)).consume

/-- The session's joint law's strategy. -/
noncomputable def sessionWS {β : Type} (fin : LiveOutJ J mPts → β) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    Strategy (sessionW D M dg J fin N p) :=
  fun pts => roundsWS D M dg J fin _ pts N _ _

/-- **One round's projection has its events' probabilities under one round's joint law**, wherever `next`'s do. -/
theorem prob_stepP {β : Type} (fin : LiveOutJ J mPts → β) (E : β → Prop) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R))
    (nextP : (s : SubmitJ J) → (c : Fin s.n → 𝔽) → FlockSoundness.Game.Strategy
      ((liveRoundsJ dg J N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))).bind (KLJ rootB pts)))
    (nextW : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β × CoinTape g σ))
    (h : ∀ s c (w : Strategy (nextW s c)), FlockSoundness.Game.prob (fun o => E (fin o))
      ((liveRoundsJ dg J N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))).bind (KLJ rootB pts)) (nextP s c) =
        prob (fun o => E o.1) (nextW s c) w) :
    ∀ (o : Option (SubmitJ J)) (w : Strategy (stepW D J fin rootB pts rs q nextW o)),
      FlockSoundness.Game.prob (fun o => E (fin o)) ((liveRoundsJ dg J (N + 1) rs).bind (KLJ rootB pts))
          (stepP D dg J rootB pts N rs q nextP o) =
        prob (fun o => E o.1) (stepW D J fin rootB pts rs q nextW o) w
  | none, _ => rfl
  | some s, w => Finset.sum_congr rfl fun c _ => congrArg (HMul.hMul _) (h s c (w c))

/-- **The rounds' projection has its events' probabilities under the rounds' joint law.** -/
theorem prob_roundsP {β : Type} (fin : LiveOutJ J mPts → β) (E : β → Prop) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (w : Strategy (roundsW D M dg J fin rootB pts N rs p)),
      FlockSoundness.Game.prob (fun o => E (fin o)) ((liveRoundsJ dg J N rs).bind (KLJ rootB pts))
          (roundsP D M dg J rootB pts N rs p) =
        prob (fun o => E o.1) (roundsW D M dg J fin rootB pts N rs p) w
  | 0, _, _, _ => rfl
  | N + 1, rs, p, w =>
    prob_stepP D dg J fin E rootB pts N rs p.toStop _ _ (fun _ _ w => prob_roundsP fin E rootB pts N _ _ w)
      (D.sub J p.toStop.modes) w

/-- **The session's projection has its events' probabilities under the session's joint law.** -/
theorem prob_sessionP {β : Type} (fin : LiveOutJ J mPts → β) (E : β → Prop) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R))
    (w : Strategy (sessionW D M dg J fin N p)) :
    FlockSoundness.Game.prob (fun o => E (fin o)) (liveSession dg J mPts N) (sessionP D M dg J N p) =
      prob (fun o => E o.1) (sessionW D M dg J fin N p) w :=
  Finset.sum_congr rfl fun pts _ => congrArg (HMul.hMul _) (prob_roundsP D M dg J fin E _ pts N _ _ (w pts))

end Session

/-! ## The projection on the live audit -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- The link points' count at the session. -/
abbrev Live.mP (A : Live) : ℕ :=
  FlockSoundness.Discharge.Exec.mPtsOf A.D.unroll.g (A.kd / FlockSoundness.Discharge.Composed.nTab A.I)

/-- The tables at a draw and a registration. -/
noncomputable abbrev Live.J (A : Live) (S : Finset (Fin A.n)) (R : A.Reg) : ℕ :=
  (FlockSoundness.Discharge.ZkHidden.planZHJP (Reg := A.Reg) A.dj A.y₀ S R).length

/-- The live audit's outcome at the session. -/
abbrev Live.Out (A : Live) : Type := A.Reg × Finset (Fin A.n) × FlockSoundness.Discharge.Exec.Transcript

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The rounds' cuts at the reading `rd`**, where the projection stops. -/
noncomputable def roundCuts {R : ℕ} (rd : Reading R) : Fin R → DigestCut Claims.Msg (V.game θ F c) :=
  fun j => (roundCut V θ F c rd j).cut

/-- **The registration the projection sends**: the decoding of the record's most likely message. -/
noncomputable def regOf (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) : A.Reg :=
  D.reg ((recordCut V θ F c rd).cut.modeHere σ)

/-- **The projection** of `σ` at the session `A`, the reading `rd`, the decoders `D` and the coin map `M`: the
registration decoded from the record's most likely message; on the draw's widened sample `ω`, the session's projection
from the play's start on `M`'s tape at `ω`, at the rounds' cuts. -/
noncomputable def projectOf (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) :
    FlockSoundness.Game.Strategy (A.extend M.widening).game :=
  ⟨regOf V θ F c A rd σ D, fun ω =>
    FlockSoundness.Game.Strategy.map _ _ (FlockSoundness.Game.Strategy.map _ _
      (sessionP D M A.dg (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) A.N
        (Pos.root (roundCuts V θ F c rd) (M.start ω))))⟩

/-- **The joint law the projection simulates**: over the live audit's coins, the projected play's outcome beside the
period's tape `σ` is played on. -/
noncomputable def jointG (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) :
    Game (A.Out × CoinTape (V.game θ F c) σ) :=
  .coin (A.extend M.widening).L.Ω fun ω =>
    sessionW D M A.dg (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D))
      (fun o => (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D) o)) A.N
      (Pos.root (roundCuts V θ F c rd) (M.start ω))

/-- The joint law's strategy: it has no message to send. -/
noncomputable def jointS (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg)
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) : Strategy (jointG V θ F c A rd σ D M) :=
  fun _ => sessionWS D M _ _ _ _ _

/-- **The projected play's events have their probabilities under the joint law.** -/
theorem prob_projectOf (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (E : A.Out → Prop)
    (w : Strategy (jointG V θ F c A rd σ D M)) :
    FlockSoundness.Game.prob E (A.extend M.widening).game (projectOf V θ F c A rd σ D M) =
      prob (fun o => E o.1) (jointG V θ F c A rd σ D M) w := by
  refine Finset.sum_congr rfl fun ω _ => congrArg (HMul.hMul _) ?_
  exact ((FlockSoundness.Discharge.Exec.prob_strategy_map _ _ _ _).trans
    (FlockSoundness.Discharge.Exec.prob_strategy_map _ _ _ _)).trans (prob_sessionP D M A.dg _ _ E A.N _ (w ω))

/-! ## The logged coupling at the projection -/

/-- **The coin map's premise**: at every event `Q` of the period's tape, the joint law's tape is within the hiding
term of the uniform tape. -/
def Couples (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg)
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    prob (fun o => Q o.2) (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M) + A.hidingTerm

/-- **The decoders' premise**, at the composite `k` and the property `P` of the play: on the joint law's support, a
tape whose play's sampled verdict accepts, that misses at no round and not at the record, and whose play `P` holds of,
is beside an outcome flock-verify `--zk` accepts at `own`. -/
def Decodes (k : Verity.Registration.Composite) (P : Exchange → Prop) (A : Live) (rd : Reading A.rounds)
    (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) :
    Prop :=
  ∀ o, PlayReaches (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M) o →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2) ∧ NoMiss V θ F c rd σ o.2 ∧
      P (CoinTape.run (V.game θ F c) σ o.2) →
    FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own o.1

/-- **The leg's constructions with the projection** at the readings `rd`, the decoders `D` and the coin maps `M`: the
coins are `M`'s widening and the projection `projectOf`; `K`'s finders and forger. -/
noncomputable def Constructions.withProject (K : Constructions V) (rd : Readings)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (M : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) : Constructions V :=
  { K with
    coins := fun A θ F c σ => (M A θ F c σ).widening
    project := fun A θ F c σ => projectOf V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ) }

variable (k : Verity.Registration.Composite)

omit θ F c in
/-- **(b), the logged coupling, at the projection**: at the constructions with the projection, the logged coupling
holds at the session `A`, the property `P`, the parameters `θ` and the committed files `F` wherever the coin map's
premise and the decoders' hold at each commitment and strategy. -/
theorem loggedCoupling_withProject (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (M : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (A : Live) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (hC : ∀ c σ, Couples V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ))
    (hD : ∀ c σ, Decodes V θ F c k P A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ)) :
    LoggedCoupling V k rd (K.withProject V rd D M) A P θ F := fun c σ => by
  have h := pr_le_of_joint (jointG V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ))
    (jointS V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ))
    (fun x => V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x ∧
      P (CoinTape.run (V.game θ F c) σ x))
    (FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own) A.hidingTerm (hC c σ _) (hD c σ)
  rw [← prob_projectOf V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ)
    (FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own)] at h
  exact h

omit θ F c in
/-- **The coupling at the projection**: at the constructions with the projection and the messages' finders at the
readings `rd`, `CouplingTarget` holds at the session `A` wherever the binding's bound holds at one property `P` of the
play and the coin map's premise and the decoders' hold at `P`. -/
theorem couplingTarget_withProject (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (M : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (A : Live) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (ha : LoggedBound V k (K.withProject V rd D M) P θ F)
    (hC : ∀ c σ, Couples V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ))
    (hD : ∀ c σ, Decodes V θ F c k P A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ)) :
    CouplingTarget V k ((K.withProject V rd D M).withMessages rd) A θ F :=
  couplingTarget_of_logged V k rd _ A P θ F ha (loggedCoupling_withProject V k rd K D M A P θ F hC hD)

end Verity.Sampled
