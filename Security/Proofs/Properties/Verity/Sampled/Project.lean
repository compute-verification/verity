module

public import Proofs.Properties.Verity.Sampled.Coupling
public import Proofs.Core.Game.InterleaveBridge

/-!
# Sampled proofs' coupling: the projection

`Constructions.project`, the strategy of the live audit a strategy `σ` of the period's game is read as: it simulates
`σ` (`Verity.Game.Pos`, in `Proofs.Core.Game.InterleaveBridge`). It registers the decoding of the record's most likely
message; on the sample's widened sample it starts a tape of the period's game (the coin map's `start`); at each stop, the
first node of `σ`'s play on the tape where a round's cut (`roundCut`) fixes a message, it sends the decoding of the most
likely message there (`Commit`'s roots at the first stop, a round's submission at each later one) and writes the live
coin it gets back into the tape there (the coin map's `link` and `round`); after the last round it sends the proof files
decoded from the play's output on the tape.

* **The parameters** (`Decoders`, `CoinMap`): the decoders, and the coin map with the widening of the sample's sample
  it reads. They are the projection's two inputs: what each stop's message decodes to, and how the live coins become
  the period's.
* **The projection** (`projectOf`, `Constructions.withProject`), and **the joint law** it simulates (`jointG`, `jointS`):
  the projected play's outcome beside the period's tape it plays `σ` on, over the live audit's coins. The projected
  play's events have their probabilities under it (`prob_projectOf`).
* **The two premises** the logged coupling takes at it, off events at the stops that the joint law is held off
  (`LinkHold` at the first stop's link points, `RoundHold` at each later stop's round coins): the joint law's tape off
  them is, at every event, within the hiding term of the uniform tape (`Couples`, the coin map's), and on the joint
  law's support off them an accepted play `P` holds of that misses nowhere has an outcome flock-verify `--zk` accepts
  (`Decodes`, the decoders'). Off them the logged coupling holds at the projection (`loggedCoupling_withProject`), so
  `CouplingTarget` holds at the constructions with the projection and the messages' finders wherever the binding's
  bound does (`couplingTarget_withProject`). Where the held-off events are rare, `Couples` carries their probability
  (`Proofs.Properties.Verity.Sampled.CoinMap`).
* **The joint law's tape off the held-off events** (`roundsOff`, `sessionOff`, `jointOff`): an event of the period's
  tape under the joint law, the plays through a held-off event counted out. Wherever the event implies `E` of the
  outcome on the support off them, it is at most `E`'s probability under the joint law (`roundsOff_le_prob`,
  `sessionOff_le_prob`, `jointOff_le_prob`).
* **What the joint law's support off them holds** (`ReachesOff`, `RoundsReach`): each outcome is a trace of the
  projection's stops, the submissions decoded there and their coins off the held-off events, ending on the outcome's
  tape with the proof files decoded from its play. That tape is one through every stop (`RoundsReach.through`), and
  each stop's decoded modes are the rounds' cuts' most likely messages on it (`RoundsReach.toStop_modes`,
  `firstStop_modes`, from `Verity.Game.Pos.modes_eq` at the most likely message's reading `DigestCut.modeReading`).
  The registration is the record's on every tape (`regOf_eq`).
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

/-- **The coin map a projection reads**, at a strategy `σ` of the period's game `g`: the widening of the sample's
sample (`T`); the period's tape on the sample's sample and the widening (`start`); and at a stop, the period's tape from
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

/-- **An event at a stop's round coins that the joint law is held off**: of the stop's node, the table count, the
submission decoded there and the round's live coins. -/
abbrev RoundHold : Type 1 := Game Exchange → (j : ℕ) → (s : SubmitJ j) → (Fin s.n → 𝔽) → Prop

/-- **An event at the first stop's link points that the joint law is held off**: of the stop's node and the live link
points. -/
abbrev LinkHold (mPts : ℕ) : Type 1 := Game Exchange → (Fin 2 → Fin mPts → 𝔽) → Prop

/-! ## The projection on the live session -/

section Session

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : Decoders R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ) (B : RoundHold)

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

/-! ### The joint law's support off a held-off event -/

/-- **The rounds' joint law's support off `B`, as a trace** from the position `p`: the play's end after the last round
or at a stop whose decoding submits nothing, beside the tape there and with the proof files decoded from its play; or
a submission decoded at the stop, any of its coins `B` doesn't hold of there, and the trace from the tape with them
written in. -/
inductive RoundsReach {β : Type} (fin : LiveOutJ J mPts → β) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ℕ → (Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → β × CoinTape g σ → Prop
  | zero (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) :
      RoundsReach fin rootB pts 0 rs p (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.tape)⟩, p.tape)
  | stop (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (h : D.sub J p.toStop.modes = none) :
      RoundsReach fin rootB pts (N + 1) rs p
        (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.toStop.tape)⟩, p.toStop.tape)
  | round (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (s : SubmitJ J)
      (c : Fin s.n → 𝔽) (o : β × CoinTape g σ) (h : D.sub J p.toStop.modes = some s) (hB : ¬ B p.toStop.g' J s c)
      (ho : RoundsReach fin rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
        (p.toStop.rewrite (M.round p.toStop J s c)).consume o) :
      RoundsReach fin rootB pts (N + 1) rs p o

variable {D M dg J B} in
/-- **A trace's tape is one through its start, and through a later position that keeps the invariant.** -/
theorem RoundsReach.through {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {Ks₀ : Fin R → DigestCut Claims.Msg g} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs p o) (hp : p.Inv Ks₀) :
    ∃ q : Pos g σ Claims.Msg (Fin R), p.Le q ∧ q.Inv Ks₀ ∧ o.2 = q.tape := by
  induction hr with
  | zero rs p => exact ⟨p, Pos.Le.refl p, hp, rfl⟩
  | stop N rs p _ =>
    obtain ⟨hl, _, hi⟩ := Pos.toStop_spec hp
    exact ⟨p.toStop, hl, hi, rfl⟩
  | round N rs p s c o _ _ _ ih =>
    obtain ⟨hl, _, hi⟩ := Pos.toStop_spec hp
    obtain ⟨q, hq, hqi, ho⟩ := ih (Pos.inv_consume (Pos.inv_rewrite hi _))
    exact ⟨q, (hl.trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans hq, hqi, ho⟩

variable {D M dg J B} in
/-- **The modes each stop of a trace decodes are the cuts' on the trace's tape**: at the trace's first stop, each cut
fixed there has, on the tape the trace ends on, the most likely message the stop decoded; every other cut decoded
none there. -/
theorem RoundsReach.toStop_modes {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {Ks₀ : Fin R → DigestCut Claims.Msg g} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts (N + 1) rs p o) (hp : p.Inv Ks₀) :
    p.toStop.modes = fun i => if (p.toStop.Ks i).isHere then (Ks₀ i).modeOn σ o.2 else none := by
  obtain ⟨_, _, hi⟩ := Pos.toStop_spec hp
  cases hr with
  | stop _ _ _ _ => exact Pos.modes_eq hi (Pos.Le.refl _) _
  | round _ _ _ s c _ _ _ ho =>
    obtain ⟨q, hq, _, hoq⟩ := ho.through (Pos.inv_consume (Pos.inv_rewrite hi _))
    rw [hoq]
    exact Pos.modes_eq hi (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq) _

variable {D M dg J B} in
/-- **A trace's outcome is `fin` of the live session's outcome** at the trace's roots and link points: with the rounds
the submissions answered along it leave, in order, and the proof files decoded from the play on the tape it ends on.
Each answered submission is decoded at the stop after a position the trace passes through, which keeps the
invariant; its coins are ones `B` doesn't hold of there, and the rest of the trace runs from that position. -/
theorem RoundsReach.out {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)}
    {o : β × CoinTape g σ} (hr : RoundsReach D M dg J B fin rootB pts N rs p o) :
    ∃ as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽))),
      o.1 = fin ⟨rootB, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2)) rs,
          D.proofs (CoinTape.run g σ o.2)⟩ ∧
      ∀ a ∈ as, (∀ Ks₀ : Fin R → DigestCut Claims.Msg g, p.Inv Ks₀ → a.1.Inv Ks₀) ∧
        D.sub J a.1.toStop.modes = some a.2.1 ∧
        ¬ B a.1.toStop.g' J a.2.1 a.2.2 ∧ ∃ N' rs', RoundsReach D M dg J B fin rootB pts (N' + 1) rs' a.1 o := by
  induction hr with
  | zero rs p => exact ⟨[], rfl, fun _ h => by simp at h⟩
  | stop N rs p _ => exact ⟨[], rfl, fun _ h => by simp at h⟩
  | round N rs p s c o hs hB ho ih =>
    obtain ⟨as, h1, h2⟩ := ih
    refine ⟨(p, ⟨s, c⟩) :: as, h1, fun a ha => ?_⟩
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨fun _ h => h, hs, hB, N, rs, .round N rs p s c o hs hB ho⟩
    · obtain ⟨hi, h3, h4, h5⟩ := h2 a ha
      exact ⟨fun Ks₀ h => hi Ks₀ (Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec h).2.2 _)), h3, h4, h5⟩

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

open Classical in
/-- **One stop's share of an event of the tape, off `B`**, at the submission decoded there: the event at the play's end
there; else the mean over the round's coins of `e` from the tape with them written in, those `B` holds of at the stop
counted out. -/
noncomputable def stopOff (Q : CoinTape g σ → Prop) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (q : Pos g σ Claims.Msg (Fin R)) : Option (SubmitJ J) → ℝ≥0∞
  | none => tapeInd Q q
  | some s => avg fun c : Fin s.n → 𝔽 => if B q.g' J s c then 0 else e (q.rewrite (M.round q J s c)).consume

/-- **An event of the tape under the rounds' joint law off `B`** from the position `p`: its probability there, the
plays through a stop whose round's coins `B` holds of counted out. It reads neither the outcome's relabelling nor the
session's values. -/
noncomputable def roundsOff (Q : CoinTape g σ → Prop) : ℕ → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | 0, p => tapeInd Q p
  | N + 1, p => stopOff M J B Q (roundsOff Q N) p.toStop (D.sub J p.toStop.modes)

open Classical in
/-- **An event of the tape under the session's joint law off `BL` and `B`** from the position `p`: its mean over the
link points at the first stop of the rounds' share off `B`, the link points `BL` holds of there counted out. -/
noncomputable def sessionOff (BL : LinkHold mPts) (Q : CoinTape g σ → Prop) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    ℝ≥0∞ :=
  avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
    if BL p.toStop.g' pts then 0 else roundsOff D M J B Q N (p.toStop.rewrite (M.link p.toStop pts)).consume

theorem roundsOff_le_one (Q : CoinTape g σ → Prop) : ∀ (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)),
    roundsOff D M J B Q N p ≤ 1
  | 0, p => tapeInd_le_one Q p
  | N + 1, p => by
    show stopOff M J B Q (roundsOff D M J B Q N) p.toStop (D.sub J p.toStop.modes) ≤ 1
    cases D.sub J p.toStop.modes with
    | none => exact tapeInd_le_one Q _
    | some s =>
      refine (avg_mono fun c => ?_).trans_eq (avg_const 1)
      split_ifs
      · exact zero_le_one
      · exact roundsOff_le_one Q N _

open Classical in
variable {D M dg J B} in
/-- **One stop's share off `B` is at most the event's probability under one round's joint law** wherever, on the
support off `B`, the event of the tape implies `E` of the outcome: at the play's end there, and after each of the
round's coins `B` doesn't hold of, wherever `e` there is at most `E`'s probability under `next`. -/
theorem stopOff_le_prob {β : Type} (fin : LiveOutJ J mPts → β) (Q : CoinTape g σ → Prop) (E : β → Prop)
    (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (rs : Fin J → Fin 2 → List Flock.Round)
    (q : Pos g σ Claims.Msg (Fin R)) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β × CoinTape g σ)) (w : ∀ s c, Strategy (next s c)) :
    ∀ o : Option (SubmitJ J),
      (o = none → Q q.tape → E (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ q.tape)⟩)) →
      (∀ s, o = some s → ∀ c, ¬ B q.g' J s c →
        e (q.rewrite (M.round q J s c)).consume ≤ prob (fun o => E o.1) (next s c) (w s c)) →
      stopOff M J B Q e q o ≤ prob (fun o => E o.1) (stepW D J fin rootB pts rs q next o)
        (stepWS D J fin rootB pts rs q next w o)
  | none, h0, _ => by
    show tapeInd Q q ≤ if E (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ q.tape)⟩) then 1 else 0
    unfold tapeInd
    by_cases hQ : Q q.tape
    · rw [ite_eq_left hQ, ite_eq_left (h0 rfl hQ)]
    · rw [ite_eq_right hQ]
      exact zero_le
  | some s, _, hs => avg_mono fun c => by
    split_ifs with hB
    · exact zero_le
    · exact hs s rfl c hB

open Classical in
variable {D M dg J B} in
/-- **The rounds' share off `B` is at most the event's probability under the rounds' joint law** wherever, on the
support off `B` (`RoundsReach`), the event of the tape implies `E` of the outcome. -/
theorem roundsOff_le_prob {β : Type} (fin : LiveOutJ J mPts → β) (Q : CoinTape g σ → Prop) (E : β → Prop)
    (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)),
      (∀ o, RoundsReach D M dg J B fin rootB pts N rs p o → Q o.2 → E o.1) →
      roundsOff D M J B Q N p ≤ prob (fun o => E o.1) (roundsW D M dg J fin rootB pts N rs p)
        (roundsWS D M dg J fin rootB pts N rs p)
  | 0, rs, p, h => by
    show tapeInd Q p ≤ if E (fin ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.tape)⟩) then 1 else 0
    unfold tapeInd
    by_cases hQ : Q p.tape
    · rw [ite_eq_left hQ, ite_eq_left (h _ (.zero rs p) hQ)]
    · rw [ite_eq_right hQ]
      exact zero_le
  | N + 1, rs, p, h =>
    stopOff_le_prob fin Q E rootB pts rs p.toStop (roundsOff D M J B Q N) _
      (fun _ _ => roundsWS D M dg J fin rootB pts N _ _) (D.sub J p.toStop.modes)
      (fun h0 hQ => h _ (.stop N rs p h0) hQ)
      (fun s hs c hB => roundsOff_le_prob fin Q E rootB pts N _ _ fun o ho => h o (.round N rs p s c o hs hB ho))

variable {D M dg J B} in
/-- **The session's share off `BL` and `B` is at most the event's probability under the session's joint law** wherever,
on the support off them, the event of the tape implies `E` of the outcome. -/
theorem sessionOff_le_prob {β : Type} (fin : LiveOutJ J mPts → β) (BL : LinkHold mPts) (Q : CoinTape g σ → Prop)
    (E : β → Prop) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R))
    (h : ∀ pts, ¬ BL p.toStop.g' pts → ∀ o, RoundsReach D M dg J B fin (D.root p.toStop.modes) pts N (fun _ _ => [])
      (p.toStop.rewrite (M.link p.toStop pts)).consume o → Q o.2 → E o.1) :
    sessionOff D M J B BL Q N p ≤ prob (fun o => E o.1) (sessionW D M dg J fin N p) (sessionWS D M dg J fin N p) :=
  avg_mono fun pts => by
    split_ifs with hL
    · exact zero_le
    · exact roundsOff_le_prob fin Q E _ pts N _ _ (h pts hL)

end Session

/-! ## The projection on the live audit -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- The link points' count at the session. -/
abbrev Live.mP (A : Live) : ℕ :=
  FlockSoundness.Discharge.Exec.mPtsOf A.D.unroll.g (A.kd / FlockSoundness.Discharge.Composed.nTab A.I)

/-- The tables at a sample and a registration. -/
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
registration decoded from the record's most likely message; on the sample's widened sample `ω`, the session's projection
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

/-- **The joint law's support off `BL` and `B`**: the outcomes with a widened sample `ω` and link points `pts` that
`BL` doesn't hold of at the first stop, with a trace off `B` of the rounds from the first stop, `pts` written in,
ending on the outcome. -/
def ReachesOff (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg)
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP) (B : RoundHold)
    (o : A.Out × CoinTape (V.game θ F c) σ) : Prop :=
  ∃ (ω : (A.extend M.widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽), ¬ BL (firstStop V θ F c A rd σ M ω).g' pts ∧
    RoundsReach D M A.dg (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) B
      (fun t => (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D) t))
      (D.root (firstStop V θ F c A rd σ M ω).modes) pts A.N (fun _ _ => [])
      ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o

/-- **The first stop's modes on a trace from it**: the rounds' cuts' most likely messages on the tape the trace ends
on, at each cut fixed there. -/
theorem firstStop_modes (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (B : RoundHold)
    (ω : (A.extend M.widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽) {J : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β}
    {rootB : Array ByteArray} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts A.N (fun _ _ => [])
      ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o) :
    (firstStop V θ F c A rd σ M ω).modes = fun i =>
      if ((firstStop V θ F c A rd σ M ω).Ks i).isHere then (roundCuts V θ F c rd i).modeOn σ o.2 else none := by
  obtain ⟨_, _, hi⟩ := Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))
  obtain ⟨q, hq, _, hoq⟩ := hr.through (Pos.inv_consume (Pos.inv_rewrite hi _))
  rw [hoq]
  exact Pos.modes_eq hi (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq) _

/-- **An outcome on the joint law's support off `BL` and `B` is the writer's** at the sample of a widened sample, the
decoded registration and a live session's outcome: the roots decoded at the first stop, link points `BL` doesn't hold
of there, the rounds the trace from the first stop leaves (`RoundsReach.out`), and the proof files decoded from the play
on the tape beside the outcome. -/
theorem reachesOff_out (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP)
    (B : RoundHold) {o : A.Out × CoinTape (V.game θ F c) σ} (h : ReachesOff V θ F c A rd σ D M BL B o) :
    ∃ (ω : (A.extend M.widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽)
      (reps : Fin (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) → Fin 2 → List Flock.Round),
      ¬ BL (firstStop V θ F c A rd σ M ω).g' pts ∧
      o.1 = (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)
          ⟨D.root (firstStop V θ F c A rd σ M ω).modes, pts, reps, D.proofs (CoinTape.run (V.game θ F c) σ o.2)⟩) := by
  obtain ⟨ω, pts, hL, hr⟩ := h
  obtain ⟨_, h1, -⟩ := hr.out
  exact ⟨ω, pts, _, hL, h1⟩

variable {V θ F c} in
/-- **Off the misses, a stop decodes the message the play reveals there**: at the stop after a position of a trace that
keeps the invariant at the rounds' cuts, where the tape the trace ends on misses at no round and its play reveals `w`
at the cut of a round fixed at the stop, the stop's most likely message for that round is `w`. -/
theorem RoundsReach.modes_noMiss {A : Live} {rd : Reading A.rounds} {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {B : RoundHold} {J N : ℕ}
    {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin A.mP → 𝔽}
    {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds)}
    {o : β × CoinTape (V.game θ F c) σ} (hr : RoundsReach D M A.dg J B fin rootB pts (N + 1) rs p o)
    (hp : p.Inv (roundCuts V θ F c rd)) (hn : NoMiss V θ F c rd σ o.2) {i : Fin A.rounds}
    (hi : (p.toStop.Ks i).isHere = true) {w : Claims.Msg}
    (hw : (roundCut V θ F c rd i).cut.revealOn σ o.2 = some w) : p.toStop.modes i = some w := by
  rw [hr.toStop_modes hp]
  dsimp only
  rw [ite_eq_left hi]
  by_contra hne
  exact hn.1 i ((DigestCut.missOn_iff _ _ _).2 ⟨w, hw, hne⟩)

/-- **Off the misses, the first stop decodes the message the play reveals there** (`firstStop_modes`). -/
theorem firstStop_modes_noMiss (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (B : RoundHold)
    (ω : (A.extend M.widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽) {J : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β}
    {rootB : Array ByteArray} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts A.N (fun _ _ => [])
      ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o)
    (hn : NoMiss V θ F c rd σ o.2) {i : Fin A.rounds} (hi : ((firstStop V θ F c A rd σ M ω).Ks i).isHere = true)
    {w : Claims.Msg} (hw : (roundCut V θ F c rd i).cut.revealOn σ o.2 = some w) :
    (firstStop V θ F c A rd σ M ω).modes i = some w := by
  rw [firstStop_modes V θ F c A rd σ D M B ω pts hr]
  dsimp only
  rw [ite_eq_left hi]
  by_contra hne
  exact hn.1 i ((DigestCut.missOn_iff _ _ _).2 ⟨w, hw, hne⟩)

/-- **An event of the period's tape under the joint law off `BL` and `B`**: over the widened sample, the session's
share off them from the play's start (`sessionOff`). -/
noncomputable def jointOff (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP)
    (B : RoundHold) (Q : CoinTape (V.game θ F c) σ → Prop) : ℝ≥0∞ :=
  avg fun ω : (A.extend M.widening).L.Ω =>
    sessionOff D M (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) B BL Q A.N
      (Pos.root (roundCuts V θ F c rd) (M.start ω))

/-- **The joint law off `BL` and `B` gives an event of the tape at most `E`'s probability** wherever, on its support
off them (`ReachesOff`), the event implies `E` of the outcome. -/
theorem jointOff_le_prob (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP)
    (B : RoundHold) (Q : CoinTape (V.game θ F c) σ → Prop) (E : A.Out → Prop)
    (h : ∀ o, ReachesOff V θ F c A rd σ D M BL B o → Q o.2 → E o.1) :
    jointOff V θ F c A rd σ D M BL B Q ≤
      prob (fun o => E o.1) (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M) :=
  avg_mono fun ω => sessionOff_le_prob _ BL Q E A.N _ fun pts hL o ho => h o ⟨ω, pts, hL, ho⟩

/-! ## The logged coupling at the projection -/

/-- **The coin map's premise, off `BL` and `B`**: at every event `Q` of the period's tape, the joint law's tape off them
is within the hiding term of the uniform tape. -/
def Couples (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg)
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP) (B : RoundHold) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    jointOff V θ F c A rd σ D M BL B Q + A.hidingTerm

/-- **The coin map's start leaves the tape from the first stop fresh**: over the widened sample, any reading of the
sample and the projection's first stop has the mean of its average over the tapes from the stop. -/
def FirstStopFresh (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) : Prop :=
  ∀ G : Finset (Fin A.n) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞,
    avg (fun ω : (A.extend M.widening).L.Ω => G ((A.extend M.widening).L.draw ω) (firstStop V θ F c A rd σ M ω)) =
      avg fun ω : (A.extend M.widening).L.Ω =>
        Pos.resample (G ((A.extend M.widening).L.draw ω)) (firstStop V θ F c A rd σ M ω)

theorem ofReal_pr_eq_prCoin {C : Type} [Fintype C] [Nonempty C] (Q : C → Prop) :
    ENNReal.ofReal (Claims.Pr Q) = prCoin Q := by
  classical
  rw [prCoin_eq_card, Claims.Pr, Claims.prUniform, ENNReal.ofReal_div_of_pos (Nat.cast_pos.2 Fintype.card_pos), ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]

/-- **The decoders' premise**, at the composite `k` and the property `P` of the play: on the joint law's support off
`BL` and `B`, a tape whose play's sampled verdict accepts, that misses at no round and not at the record, and whose play
`P` holds of, is beside an outcome flock-verify `--zk` accepts at `own`. -/
def Decodes (k : Verity.Registration.Composite) (P : Exchange → Prop) (A : Live) (rd : Reading A.rounds)
    (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (BL : LinkHold A.mP) (B : RoundHold) : Prop :=
  ∀ o, ReachesOff V θ F c A rd σ D M BL B o →
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
holds at the session `A`, the parameters `θ`, the committed files `F` and the cut `P` wherever, at each commitment and
strategy, off its held-off events `BL` and `B`, the coin map's premise and the decoders' hold. -/
theorem loggedCoupling_withProject (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (M : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (P : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (c : ByteArray) → Strategy (V.game θ F c) → LinkHold A.mP)
    (B : (c : ByteArray) → Strategy (V.game θ F c) → RoundHold)
    (hC : ∀ c σ, Couples V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ) (BL c σ) (B c σ))
    (hD : ∀ c σ, Decodes V θ F c k (P c σ) A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ) (BL c σ) (B c σ)) :
    LoggedCoupling V k rd (K.withProject V rd D M) A θ F P := fun c σ => by
  have h := (hC c σ fun x => V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧
      NoMiss V θ F c (rd A θ F c) σ x ∧ P c σ (CoinTape.run (V.game θ F c) σ x)).trans
    (add_le_add (jointOff_le_prob V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ) (BL c σ) (B c σ) _
      (FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own) (hD c σ)) le_rfl)
  rw [← prob_projectOf V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ)
    (FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own)] at h
  exact h

omit θ F c in
/-- **The coupling at the projection**: at the constructions with the projection and the messages' finders at the
readings `rd`, `CouplingTarget` holds at the session `A` wherever the binding's bound holds at one property `P` of the
play, the digest's at one property `Q` at each commitment and strategy, and, at each commitment and strategy off its
held-off events `BL` and `B`, the coin map's premise and the decoders' hold at the two. -/
theorem couplingTarget_withProject (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (M : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (A : Live) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (c : ByteArray) → Strategy (V.game θ F c) → LinkHold A.mP)
    (B : (c : ByteArray) → Strategy (V.game θ F c) → RoundHold) (ha : LoggedBound V k (K.withProject V rd D M) P θ F)
    (hq : DigestBound V k (K.withProject V rd D M) A θ F Q)
    (hC : ∀ c σ, Couples V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ) (BL c σ) (B c σ))
    (hD : ∀ c σ, Decodes V θ F c k (fun o => P o ∧ Q c σ o) A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ) (BL c σ)
      (B c σ)) :
    CouplingTarget V k ((K.withProject V rd D M).withMessages rd) A θ F :=
  couplingTarget_of_logged V k rd _ A P θ F Q ha hq
    (loggedCoupling_withProject V k rd K D M A θ F (fun c σ o => P o ∧ Q c σ o) BL B hC hD)

end Verity.Sampled
