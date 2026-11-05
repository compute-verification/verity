module

public import Proofs.Properties.Verity.Sampled.Coupling
public import Proofs.Core.Game.InterleaveBridge

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
* **The coin map's premise, off local ones** (`couples_of_preserves`, `couplingTarget_withProject_of_preserves`):
  `Couples` holds wherever the coin map writes each round's coins and the link points in as the tape's own
  (`CoinMap.RoundPreserves`, `CoinMap.LinkPreserves`), its start leaves the tape from the first stop fresh given the
  draw (`FirstStopFresh`), and its start is within the hiding term of the uniform tape (`StartClose`). Over a uniform
  tape from any position, the rounds' and the session's joint laws give an event of the tape its share of the tapes
  through the position (`avg_roundsW_tape`, `avg_linkW_tape`), from `Verity.Game.Pos.avg_toStop` and the tape law's
  indifference to everything but the stops (`prob_roundsW_tape`).
* **What the joint law's support holds** (`jointG_reaches`, `RoundsReach`): each outcome it reaches is a trace of the
  projection's stops, the submissions decoded there and their coins, ending on the outcome's tape with the proof files
  decoded from its play. That tape is one through every stop (`RoundsReach.through`), and each stop's decoded modes
  are the rounds' cuts' most likely messages on it (`RoundsReach.toStop_modes`, from `Verity.Game.Pos.modes_eq` at the
  most likely message's reading `DigestCut.modeReading`). The registration is the record's on every tape
  (`regOf_eq`).
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

variable {α V ι : Type}

section Readings

variable [Nonempty V]

namespace DigestCut

/-- **The miss as a reading of cuts** (`missOn`). -/
def missReading : CutReading α V Prop := ⟨fun K s x => K.missOn s x, fun _ _ _ => rfl, fun _ _ _ => rfl⟩

omit [Nonempty V] in
/-- **The revealed message as a reading of cuts** (`revealOn`). -/
def revealReading : CutReading α V (Option V) := ⟨fun K s x => K.revealOn s x, fun _ _ _ => rfl, fun _ _ _ => rfl⟩

/-- **The most likely message as a reading of cuts** (`modeOn`). -/
noncomputable def modeReading : CutReading α V (Option V) :=
  ⟨fun K s x => K.modeOn s x, fun _ _ _ => rfl, fun _ _ _ => rfl⟩

omit [Nonempty V] in
theorem isHere_iff {g : Game α} {K : DigestCut V g} : K.isHere = true ↔ ∃ rv, K = .here rv := by
  cases K with
  | here rv => exact ⟨fun _ => ⟨rv, rfl⟩, fun _ => rfl⟩
  | skip => exact ⟨fun h => by simp [isHere] at h, fun ⟨_, h⟩ => by cases h⟩
  | send K => exact ⟨fun h => by simp [isHere] at h, fun ⟨_, h⟩ => by cases h⟩
  | coin K => exact ⟨fun h => by simp [isHere] at h, fun ⟨_, h⟩ => by cases h⟩

theorem modeHere_of_isHere_false {g : Game α} {K : DigestCut V g} (s : Strategy g) (h : K.isHere = false) :
    K.modeHere s = none := by
  cases K with
  | here rv => simp [isHere] at h
  | skip => rfl
  | send K => rfl
  | coin K => rfl

end DigestCut

namespace Pos

variable {g : Game α} {σ : Strategy g}

/-- **A stop's most likely messages, on every later tape**: at a position that keeps the invariant, each cut fixed
there has, on every tape through any later position, the most likely message the position reads; every other cut
reads none there. -/
theorem modes_eq {Ks₀ : ι → DigestCut V g} {p q : Pos g σ V ι} (h : p.Inv Ks₀) (hpq : p.Le q)
    (y : CoinTape q.g' q.s') :
    p.modes = fun i => if (p.Ks i).isHere then (Ks₀ i).modeOn σ (q.back y) else none := by
  funext i
  by_cases hi : (p.Ks i).isHere = true
  · obtain ⟨rv, hK⟩ := DigestCut.isHere_iff.mp hi
    obtain ⟨z, hz⟩ := read_here h hpq hK DigestCut.modeReading y
    simp only [hi, ↓reduceIte]
    show (p.Ks i).modeHere p.s' = _
    rw [hK]
    exact hz.symm
  · simp only [Bool.not_eq_true] at hi
    simp only [hi, Bool.false_eq_true, ↓reduceIte]
    exact DigestCut.modeHere_of_isHere_false _ hi

end Pos

end Readings

end Verity.Game

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

/-- **The coin map writes a round's live coins in as the tape's own**: at every stop, table count and submission, a
uniform tape from the stop with a uniform round's coins written in is a uniform tape from there. -/
def CoinMap.RoundPreserves {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ} (M : CoinMap g σ Ω R mPts) :
    Prop :=
  ∀ (q : Pos g σ Claims.Msg (Fin R)) (J : ℕ) (s : SubmitJ J) (G : CoinTape q.g' q.s' → ℝ≥0∞),
    avg (fun c : Fin s.n → 𝔽 => avg fun z : CoinTape q.g' q.s' => G (M.round (q.withTape z) J s c z)) = avg G

/-- **The coin map writes the live link points in as the tape's own**: at every stop, a uniform tape from the stop
with uniform link points written in is a uniform tape from there. -/
def CoinMap.LinkPreserves {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ} (M : CoinMap g σ Ω R mPts) :
    Prop :=
  ∀ (q : Pos g σ Claims.Msg (Fin R)) (G : CoinTape q.g' q.s' → ℝ≥0∞),
    avg (fun pts : Fin 2 → Fin mPts → 𝔽 => avg fun z : CoinTape q.g' q.s' => G (M.link (q.withTape z) pts z)) =
      avg G

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

/-! ### The joint law's support -/

/-- **The rounds' joint law's support, as a trace** from the position `p`: the play's end after the last round or at
a stop whose decoding submits nothing, beside the tape there and with the proof files decoded from its play; or a
submission decoded at the stop, any of its coins, and the trace from the tape with them written in. -/
inductive RoundsReach {β : Type} (fin : LiveOutJ J mPts → β) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ℕ → (Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → β × CoinTape g σ → Prop
  | zero (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) :
      RoundsReach fin rootB pts 0 rs p (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.tape)⟩, p.tape)
  | stop (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (h : D.sub J p.toStop.modes = none) :
      RoundsReach fin rootB pts (N + 1) rs p
        (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.toStop.tape)⟩, p.toStop.tape)
  | round (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (s : SubmitJ J)
      (c : Fin s.n → 𝔽) (o : β × CoinTape g σ) (h : D.sub J p.toStop.modes = some s)
      (ho : RoundsReach fin rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
        (p.toStop.rewrite (M.round p.toStop J s c)).consume o) :
      RoundsReach fin rootB pts (N + 1) rs p o

theorem reaches_stepW {β : Type} (fin : LiveOutJ J mPts → β) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R))
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β × CoinTape g σ)) (w : ∀ s c, Strategy (next s c)) :
    ∀ (o' : Option (SubmitJ J)) (o : β × CoinTape g σ),
      PlayReaches (stepW D J fin rootB pts rs q next o') (stepWS D J fin rootB pts rs q next w o') o →
        (o' = none ∧ o = (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ q.tape)⟩, q.tape)) ∨
          ∃ s c, o' = some s ∧ PlayReaches (next s c) (w s c) o
  | none, _, h => Or.inl ⟨rfl, h⟩
  | some s, _, ⟨c, h⟩ => Or.inr ⟨s, c, rfl, h⟩

/-- **Every outcome the rounds' joint law reaches has a trace.** -/
theorem roundsReach_of_reaches {β : Type} (fin : LiveOutJ J mPts → β) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (o : β × CoinTape g σ),
      PlayReaches (roundsW D M dg J fin rootB pts N rs p) (roundsWS D M dg J fin rootB pts N rs p) o →
        RoundsReach D M dg J fin rootB pts N rs p o
  | 0, rs, p, o, h => by
    obtain rfl : o = _ := h
    exact .zero rs p
  | N + 1, rs, p, o, h => by
    rcases reaches_stepW D J fin rootB pts rs p.toStop
        (fun s c => roundsW D M dg J fin rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
          (p.toStop.rewrite (M.round p.toStop J s c)).consume)
        (fun _ _ => roundsWS D M dg J fin rootB pts N _ _) (D.sub J p.toStop.modes) o h with
      ⟨h0, rfl⟩ | ⟨s, c, hs, h'⟩
    · exact .stop N rs p h0
    · exact .round N rs p s c o hs (roundsReach_of_reaches fin rootB pts N _ _ o h')

variable {D M dg J} in
/-- **A trace's tape is one through its start, and through a later position that keeps the invariant.** -/
theorem RoundsReach.through {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {Ks₀ : Fin R → DigestCut Claims.Msg g} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J fin rootB pts N rs p o) (hp : p.Inv Ks₀) :
    ∃ q : Pos g σ Claims.Msg (Fin R), p.Le q ∧ q.Inv Ks₀ ∧ o.2 = q.tape := by
  induction hr with
  | zero rs p => exact ⟨p, Pos.Le.refl p, hp, rfl⟩
  | stop N rs p _ =>
    obtain ⟨hl, _, hi⟩ := Pos.toStop_spec hp
    exact ⟨p.toStop, hl, hi, rfl⟩
  | round N rs p s c o _ _ ih =>
    obtain ⟨hl, _, hi⟩ := Pos.toStop_spec hp
    obtain ⟨q, hq, hqi, ho⟩ := ih (Pos.inv_consume (Pos.inv_rewrite hi _))
    exact ⟨q, (hl.trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans hq, hqi, ho⟩

variable {D M dg J} in
/-- **The modes each stop of a trace decodes are the cuts' on the trace's tape**: at the trace's first stop, each cut
fixed there has, on the tape the trace ends on, the most likely message the stop decoded; every other cut decoded
none there. -/
theorem RoundsReach.toStop_modes {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {Ks₀ : Fin R → DigestCut Claims.Msg g} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J fin rootB pts (N + 1) rs p o) (hp : p.Inv Ks₀) :
    p.toStop.modes = fun i => if (p.toStop.Ks i).isHere then (Ks₀ i).modeOn σ o.2 else none := by
  obtain ⟨_, _, hi⟩ := Pos.toStop_spec hp
  cases hr with
  | stop _ _ _ _ => exact Pos.modes_eq hi (Pos.Le.refl _) _
  | round _ _ _ s c _ _ ho =>
    obtain ⟨q, hq, _, hoq⟩ := ho.through (Pos.inv_consume (Pos.inv_rewrite hi _))
    rw [hoq]
    exact Pos.modes_eq hi (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq) _

/-! ### The joint law's tape -/

open Classical in
/-- **An event of the period's tape, at a position**: its indicator on the whole tape there. -/
noncomputable def tapeInd (Q : CoinTape g σ → Prop) (q : Pos g σ Claims.Msg (Fin R)) : ℝ≥0∞ :=
  if Q q.tape then 1 else 0

theorem tapeInd_le_one (Q : CoinTape g σ → Prop) (q : Pos g σ Claims.Msg (Fin R)) : tapeInd Q q ≤ 1 := by
  unfold tapeInd
  split_ifs <;> simp

theorem resample_tapeInd_le_one (Q : CoinTape g σ → Prop) (q : Pos g σ Claims.Msg (Fin R)) :
    Pos.resample (tapeInd Q) q ≤ 1 :=
  (avg_mono fun _ => tapeInd_le_one Q _).trans_eq (avg_const 1)

theorem tapeInd_toStop (Q : CoinTape g σ → Prop) (q : Pos g σ Claims.Msg (Fin R)) :
    tapeInd Q q.toStop = tapeInd Q q := by
  unfold tapeInd
  rw [Pos.toStop_tape]

/-- **One round's joint law's tape reads neither the outcome's relabelling nor the session's values**: an event of the
tape has the same probability at any two of them, wherever the next rounds' tapes do. -/
theorem prob_stepW_tape {β β' : Type} (fin : LiveOutJ J mPts → β) (fin' : LiveOutJ J mPts → β')
    (rootB rootB' : Array ByteArray) (pts pts' : Fin 2 → Fin mPts → 𝔽) (rs rs' : Fin J → Fin 2 → List Flock.Round)
    (Q : CoinTape g σ → Prop) (q : Pos g σ Claims.Msg (Fin R))
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β × CoinTape g σ)) (w : ∀ s c, Strategy (next s c))
    (next' : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β' × CoinTape g σ)) (w' : ∀ s c, Strategy (next' s c))
    (h : ∀ s c, prob (fun o => Q o.2) (next s c) (w s c) = prob (fun o => Q o.2) (next' s c) (w' s c)) :
    ∀ o : Option (SubmitJ J),
      prob (fun o => Q o.2) (stepW D J fin rootB pts rs q next o) (stepWS D J fin rootB pts rs q next w o) =
        prob (fun o => Q o.2) (stepW D J fin' rootB' pts' rs' q next' o) (stepWS D J fin' rootB' pts' rs' q next' w' o)
  | none => rfl
  | some s => Finset.sum_congr rfl fun c _ => congrArg (HMul.hMul _) (h s c)

/-- **The rounds' joint law's tape reads neither the outcome's relabelling nor the session's values.** -/
theorem prob_roundsW_tape {β β' : Type} (fin : LiveOutJ J mPts → β) (fin' : LiveOutJ J mPts → β')
    (Q : CoinTape g σ → Prop) :
    ∀ (N : ℕ) (rootB rootB' : Array ByteArray) (pts pts' : Fin 2 → Fin mPts → 𝔽)
      (rs rs' : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)),
      prob (fun o => Q o.2) (roundsW D M dg J fin rootB pts N rs p) (roundsWS D M dg J fin rootB pts N rs p) =
        prob (fun o => Q o.2) (roundsW D M dg J fin' rootB' pts' N rs' p) (roundsWS D M dg J fin' rootB' pts' N rs' p)
  | 0, _, _, _, _, _, _, _ => rfl
  | N + 1, rootB, rootB', pts, pts', rs, rs', p =>
    prob_stepW_tape D J fin fin' rootB rootB' pts pts' rs rs' Q p.toStop _ _ _ _
      (fun _ _ => prob_roundsW_tape fin fin' Q N _ _ _ _ _ _ _) (D.sub J p.toStop.modes)

/-- **The rounds' joint law's tape is uniform from where it starts**: wherever the coin map writes each round's coins
in as the tape's own, an event of the tape has, over a uniform tape from the position `p`, its probability under the
rounds' joint law as its share of the tapes through `p`. -/
theorem avg_roundsW_tape (hR : M.RoundPreserves) {β : Type} (fin : LiveOutJ J mPts → β) (Q : CoinTape g σ → Prop) :
    ∀ (N : ℕ) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (rs : Fin J → Fin 2 → List Flock.Round)
      (p : Pos g σ Claims.Msg (Fin R)),
      avg (fun z : CoinTape p.g' p.s' => prob (fun o => Q o.2) (roundsW D M dg J fin rootB pts N rs (p.withTape z))
          (roundsWS D M dg J fin rootB pts N rs (p.withTape z))) =
        avg fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z)
  | 0, _, _, _, _ => rfl
  | N + 1, rootB, pts, rs, p => by
    let nxt : (q : Pos g σ Claims.Msg (Fin R)) → (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β × CoinTape g σ) :=
      fun q s c => roundsW D M dg J fin rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
        (q.rewrite (M.round q J s c)).consume
    let Ψ : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q => prob (fun o => Q o.2)
      (stepW D J fin rootB pts rs q (nxt q) (D.sub J q.modes))
      (stepWS D J fin rootB pts rs q (nxt q) (fun _ _ => roundsWS D M dg J fin rootB pts N _ _) (D.sub J q.modes))
    have hstop : ∀ (q : Pos g σ Claims.Msg (Fin R)) (o : Option (SubmitJ J)),
        avg (fun z : CoinTape q.g' q.s' => prob (fun o => Q o.2) (stepW D J fin rootB pts rs (q.withTape z)
          (nxt (q.withTape z)) o) (stepWS D J fin rootB pts rs (q.withTape z) (nxt (q.withTape z))
            (fun _ _ => roundsWS D M dg J fin rootB pts N _ _) o)) =
          avg fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z) := fun q o => by
      cases o with
      | none => rfl
      | some s =>
        let G : CoinTape q.g' q.s' → ℝ≥0∞ := fun z => prob (fun o => Q o.2)
          (roundsW D M dg J fin rootB pts N rs (q.consume.withTape z))
          (roundsWS D M dg J fin rootB pts N rs (q.consume.withTape z))
        calc _ = avg (fun z : CoinTape q.g' q.s' => avg fun c : Fin s.n → 𝔽 =>
              G (M.round (q.withTape z) J s c z)) :=
              congrArg avg (funext fun z => Finset.sum_congr rfl fun c _ => congrArg (HMul.hMul _)
                (prob_roundsW_tape D M dg J fin fin Q N rootB rootB pts pts _ rs _))
          _ = avg fun c : Fin s.n → 𝔽 => avg fun z : CoinTape q.g' q.s' => G (M.round (q.withTape z) J s c z) :=
              avg_comm _
          _ = avg G := hR q J s G
          _ = avg fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z) :=
              avg_roundsW_tape hR fin Q N rootB pts rs q.consume
    calc _ = avg (fun z : CoinTape p.g' p.s' => Ψ (p.withTape z).toStop) := rfl
      _ = avg fun z : CoinTape p.g' p.s' => Pos.resample Ψ (p.withTape z).toStop := Pos.avg_toStop Ψ p
      _ = avg fun z : CoinTape p.g' p.s' => Pos.resample (tapeInd Q) (p.withTape z).toStop :=
          congrArg avg (funext fun z => hstop _ (D.sub J (p.withTape z).toStop.modes))
      _ = avg fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z).toStop := (Pos.avg_toStop (tapeInd Q) p).symm
      _ = avg fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z) :=
          congrArg avg (funext fun z => tapeInd_toStop Q _)

/-- **The session's joint law's tape is uniform from its first stop**: wherever the coin map writes the link points and
each round's coins in as the tape's own, an event of the tape has, over a uniform tape from the stop `q`, its
probability under the session's joint law from there as its share of the tapes through `q`. -/
theorem avg_linkW_tape (hL : M.LinkPreserves) (hR : M.RoundPreserves) {β : Type} (fin : LiveOutJ J mPts → β)
    (Q : CoinTape g σ → Prop) (N : ℕ) (q : Pos g σ Claims.Msg (Fin R)) :
    Pos.resample (fun q' => avg fun pts : Fin 2 → Fin mPts → 𝔽 => prob (fun o => Q o.2)
        (roundsW D M dg J fin (D.root q'.modes) pts N (fun _ _ => []) (q'.rewrite (M.link q' pts)).consume)
        (roundsWS D M dg J fin (D.root q'.modes) pts N (fun _ _ => []) (q'.rewrite (M.link q' pts)).consume)) q =
      Pos.resample (tapeInd Q) q := by
  let G : CoinTape q.g' q.s' → ℝ≥0∞ := fun z => prob (fun o => Q o.2)
    (roundsW D M dg J fin (D.root q.modes) 0 N (fun _ _ => []) (q.consume.withTape z))
    (roundsWS D M dg J fin (D.root q.modes) 0 N (fun _ _ => []) (q.consume.withTape z))
  calc _ = avg (fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
        G (M.link (q.withTape z) pts z)) :=
        congrArg avg (funext fun z => congrArg avg (funext fun pts =>
          prob_roundsW_tape D M dg J fin fin Q N _ _ pts 0 _ _ _))
    _ = avg fun pts : Fin 2 → Fin mPts → 𝔽 => avg fun z : CoinTape q.g' q.s' => G (M.link (q.withTape z) pts z) :=
        avg_comm _
    _ = avg G := hL q G
    _ = Pos.resample (tapeInd Q) q := avg_roundsW_tape D M dg J hR fin Q N _ 0 _ q.consume

/-- **An event of the tape under the session's joint law** from `p` is its mean over the link points at the first
stop. -/
theorem prob_sessionW_tape {β : Type} (fin : LiveOutJ J mPts → β) (Q : CoinTape g σ → Prop) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) :
    prob (fun o => Q o.2) (sessionW D M dg J fin N p) (sessionWS D M dg J fin N p) =
      avg fun pts : Fin 2 → Fin mPts → 𝔽 => prob (fun o => Q o.2)
        (roundsW D M dg J fin (D.root p.toStop.modes) pts N (fun _ _ => [])
          (p.toStop.rewrite (M.link p.toStop pts)).consume)
        (roundsWS D M dg J fin (D.root p.toStop.modes) pts N (fun _ _ => [])
          (p.toStop.rewrite (M.link p.toStop pts)).consume) :=
  rfl

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

/-- **The registration the projection sends is the decoding of the record's most likely message on every tape.** -/
theorem regOf_eq (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg)
    (x : CoinTape (V.game θ F c) σ) :
    regOf V θ F c A rd σ D = D.reg ((recordCut V θ F c rd).cut.modeOn σ x) :=
  rfl

/-- **The projection's first stop** on the widened sample `ω`: where `Commit`'s roots are decoded. -/
noncomputable def firstStop (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (ω : (A.extend M.widening).L.Ω) :
    Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) :=
  (Pos.root (roundCuts V θ F c rd) (M.start ω)).toStop

/-- **What the joint law's support holds**: each outcome it reaches has a widened sample `ω` and link points `pts`
with a trace of the rounds from the first stop, with `pts` written in, ending on the outcome. The roots sent are the
decoding of the first stop's modes, which are the rounds' cuts' most likely messages on the outcome's tape. -/
theorem jointG_reaches (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    {o : A.Out × CoinTape (V.game θ F c) σ} (h : PlayReaches (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M) o) :
    ∃ (ω : (A.extend M.widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽),
      RoundsReach D M A.dg (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D))
        (fun t => (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
          A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D) t))
        (D.root (firstStop V θ F c A rd σ M ω).modes) pts A.N (fun _ _ => [])
        ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o ∧
      (firstStop V θ F c A rd σ M ω).modes = fun i =>
        if ((firstStop V θ F c A rd σ M ω).Ks i).isHere then (roundCuts V θ F c rd i).modeOn σ o.2 else none := by
  obtain ⟨ω, pts, h'⟩ := h
  have hr := roundsReach_of_reaches D M A.dg _ _ _ pts A.N _ _ o h'
  refine ⟨ω, pts, hr, ?_⟩
  obtain ⟨_, _, hi⟩ := Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))
  obtain ⟨q, hq, _, hoq⟩ := hr.through (Pos.inv_consume (Pos.inv_rewrite hi _))
  rw [hoq]
  exact Pos.modes_eq hi (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq) _

/-! ## The logged coupling at the projection -/

/-- **The coin map's premise**: at every event `Q` of the period's tape, the joint law's tape is within the hiding
term of the uniform tape. -/
def Couples (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg)
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    prob (fun o => Q o.2) (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M) + A.hidingTerm

/-- **The coin map's start leaves the tape from the first stop fresh**: over the widened sample, any reading of the
draw and the projection's first stop has the mean of its average over the tapes from the stop. -/
def FirstStopFresh (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) : Prop :=
  ∀ G : Finset (Fin A.n) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞,
    avg (fun ω : (A.extend M.widening).L.Ω => G ((A.extend M.widening).L.draw ω) (firstStop V θ F c A rd σ M ω)) =
      avg fun ω : (A.extend M.widening).L.Ω =>
        Pos.resample (G ((A.extend M.widening).L.draw ω)) (firstStop V θ F c A rd σ M ω)

/-- **The coin map's start is within the hiding term of the uniform tape**: over the widened sample, the period's tape
the start gives is, at every reading of it bounded by 1, within the hiding term of the uniform tape. -/
def StartClose (A : Live) (σ : Strategy (V.game θ F c)) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) : Prop :=
  ∀ H : CoinTape (V.game θ F c) σ → ℝ≥0∞, (∀ x, H x ≤ 1) →
    avg H ≤ avg (fun ω : (A.extend M.widening).L.Ω => H (M.start ω)) + A.hidingTerm

theorem ofReal_pr_eq_prCoin {C : Type} [Fintype C] [Nonempty C] (Q : C → Prop) :
    ENNReal.ofReal (Claims.Pr Q) = prCoin Q := by
  classical
  rw [prCoin_eq_card, Claims.Pr, ENNReal.ofReal_div_of_pos (Nat.cast_pos.2 Fintype.card_pos), ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]

/-- **The coin map's premise, off its local ones**: the joint law's tape is within the hiding term of the uniform tape
at every event wherever the coin map writes the link points and each round's coins in as the tape's own, its start
leaves the tape from the first stop fresh, and its start is within the hiding term of the uniform tape. -/
theorem couples_of_preserves (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (hR : M.RoundPreserves)
    (hL : M.LinkPreserves) (hF : FirstStopFresh V θ F c A rd σ M) (hS : StartClose V θ F c A σ M) :
    Couples V θ F c A rd σ D M := fun Q => by
  let R₀ := regOf V θ F c A rd σ D
  let G : Finset (Fin A.n) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞ := fun dr q =>
    avg fun pts : Fin 2 → Fin A.mP → 𝔽 => prob (fun o => Q o.2)
      (roundsW D M A.dg (A.J dr R₀) (fun t => (R₀, dr, A.wr dr R₀ t)) (D.root q.modes) pts A.N
        (fun _ _ => []) (q.rewrite (M.link q pts)).consume)
      (roundsWS D M A.dg (A.J dr R₀) (fun t => (R₀, dr, A.wr dr R₀ t)) (D.root q.modes) pts A.N
        (fun _ _ => []) (q.rewrite (M.link q pts)).consume)
  have hG : prob (fun o => Q o.2) (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M) =
      avg fun ω : (A.extend M.widening).L.Ω => Pos.resample (tapeInd Q) (firstStop V θ F c A rd σ M ω) := by
    calc prob (fun o => Q o.2) (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M)
        = avg fun ω : (A.extend M.widening).L.Ω => prob (fun o => Q o.2)
            (sessionW D M A.dg (A.J ((A.extend M.widening).L.draw ω) R₀)
              (fun t => (R₀, (A.extend M.widening).L.draw ω, A.wr ((A.extend M.widening).L.draw ω) R₀ t)) A.N
              (Pos.root (roundCuts V θ F c rd) (M.start ω)))
            (sessionWS D M A.dg (A.J ((A.extend M.widening).L.draw ω) R₀)
              (fun t => (R₀, (A.extend M.widening).L.draw ω, A.wr ((A.extend M.widening).L.draw ω) R₀ t)) A.N
              (Pos.root (roundCuts V θ F c rd) (M.start ω))) := rfl
      _ = avg fun ω : (A.extend M.widening).L.Ω =>
            G ((A.extend M.widening).L.draw ω) (firstStop V θ F c A rd σ M ω) :=
          congrArg avg (funext fun ω => prob_sessionW_tape D M A.dg _ _ Q A.N _)
      _ = avg fun ω : (A.extend M.widening).L.Ω =>
            Pos.resample (G ((A.extend M.widening).L.draw ω)) (firstStop V θ F c A rd σ M ω) := hF G
      _ = _ := congrArg avg (funext fun ω => avg_linkW_tape D M A.dg _ hL hR _ Q A.N _)
  have hU : prCoin Q = avg fun x : CoinTape (V.game θ F c) σ =>
      Pos.resample (tapeInd Q) (Pos.root (roundCuts V θ F c rd) x).toStop := by
    refine Eq.trans ?_ (Pos.avg_walk (tapeInd Q) (V.game θ F c) σ (roundCuts V θ F c rd) id)
    refine congrArg avg (funext fun x => ?_)
    show _ = tapeInd Q (Pos.root (roundCuts V θ F c rd) x).toStop
    rw [tapeInd, Pos.toStop_tape]
    rfl
  rw [ofReal_pr_eq_prCoin, hU, hG]
  exact hS _ fun x => resample_tapeInd_le_one Q _

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

omit θ F c in
/-- **The coupling at the projection, off the coin map's local premises**: `couplingTarget_withProject` with the coin
map's premise discharged by `couples_of_preserves` at each commitment and strategy. -/
theorem couplingTarget_withProject_of_preserves (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (M : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (A : Live) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (ha : LoggedBound V k (K.withProject V rd D M) P θ F)
    (hR : ∀ c σ, (M A θ F c σ).RoundPreserves) (hL : ∀ c σ, (M A θ F c σ).LinkPreserves)
    (hF : ∀ c σ, FirstStopFresh V θ F c A (rd A θ F c) σ (M A θ F c σ))
    (hS : ∀ c σ, StartClose V θ F c A σ (M A θ F c σ))
    (hD : ∀ c σ, Decodes V θ F c k P A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ)) :
    CouplingTarget V k ((K.withProject V rd D M).withMessages rd) A θ F :=
  couplingTarget_withProject V k rd K D M A P θ F ha
    (fun c σ => couples_of_preserves V θ F c A _ σ _ _ (hR c σ) (hL c σ) (hF c σ) (hS c σ)) hD

end Verity.Sampled
