module

public import Proofs.Properties.Verity.Sampled.Window
public import Proofs.Properties.Verity.Sampled.Project

/-!
# Sampled proofs' coupling over a window: the projection through several sessions, in order

The projection of `Proofs.Properties.Verity.Sampled.Project`, from one live session to a window's sessions in the
members' order (`JointAuditV5.sessions`, `seqG`): a strategy `σ` of the period's game is read as a strategy of the
window's audit. It registers the decoding of the record's most likely message; on the sample's widened coins it starts
a tape of the period's game; and it walks `σ`'s play on the tape through each member's session in turn. In a member's
session it sends, at the session's first stop, the decoding of `Commit`'s roots, writes the live link points into the
tape there, and at each later stop sends the round's decoded submission and writes the round's live coins in; at the
session's end it sends the proof files decoded from the play's output on the tape, and the next member's session starts
from the position the walk is at. A session that ends early (a stop that decodes no submission) leaves the walk at
that stop, which the next session starts from.

* **A session, then what follows it** (`roundsThen`, `stepK`, `roundsK`, `sessionK`): the session's projection of
  `Project` (`stepP`, `roundsP`, `sessionP`), with the strategy of what follows the session taken at the position the
  session's walk ends at. Its joint law (`stepWK`, `roundsWK`, `sessionWK`), its share off the held-off events
  (`stopOffK`, `roundsOffK`, `sessionOffK`) and its support off them (`RoundsReachK`, `SessionReachK`) are `Project`'s
  with the same continuation.
* **The parameters** (`WindowCoinMap`): one widening and one start for the window, and each member's link points and
  round coins written at its own stops (`WindowCoinMap.member`). The decoders are a member's each, by its place in the
  window.
* **The sessions in order** (`seqP`, `seqW`, `seqOff`, `SeqReach`): member `i`'s session at its decoders and coin map,
  then the rest from where it ends.
* **The window's projection** (`windowP`) and **the joint law** it simulates (`windowG`): the projected play's
  outcome beside the period's tape; the projected play's events have their probabilities under it (`prob_windowP`).
* **The two premises** off the held-off events (`WindowCouples`, the coin map's; `WindowDecodes`, the decoders'), the
  logged coupling from them (`windowLogged`), and the coupling at the window with the messages' finders
  (`windowCouplingTarget_of_logged`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Refine (roundOf)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## A session, then what follows it -/

section SessionK

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : Decoders R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ) (B : RoundHold) {τ γ : Type}
  (wr : LiveOutJ J mPts → τ) (K : τ → Game γ)

/-- **A session's rounds, its writer, then what follows the session**: the game the rounds' projection plays from
the rounds `rs` so far, at the session's roots and link points. -/
abbrev roundsThen (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) : Game γ :=
  (((liveRoundsJ dg J N rs).bind (KLJ rootB pts)).map wr).bind K

/-- **One round's projection, then what follows the session** (`stepP`): at the stop `q`, at the submission decoded
there: the session's end, with the proof files decoded from the play's output on the tape and then `fin` at the
session's transcript, or the submission and then `next` on its coins. -/
noncomputable def stepK (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (fin : (a : τ) → Strategy (K a))
    (next : (s : SubmitJ J) → (c : Fin s.n → 𝔽) →
      Strategy (roundsThen dg J wr K rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))) :
    Option (SubmitJ J) → Strategy (roundsThen dg J wr K rootB pts (N + 1) rs)
  | none => ⟨none, ⟨D.proofs (CoinTape.run g σ q.tape),
      fin (wr ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ q.tape)⟩)⟩⟩
  | some s => ⟨some s, next s⟩

/-- **The rounds' projection, then what follows the session** (`roundsP`) from the position `p`: at the next stop, the
decoded submission; on its coins, written into the tape there, the next round; after the last, the proof files decoded
from the play's output on the tape, and `next` from the position the walk is at. -/
noncomputable def roundsK (next : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a)) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) →
      Strategy (roundsThen dg J wr K rootB pts N rs)
  | 0, rs, p => ⟨D.proofs (CoinTape.run g σ p.tape),
      next p (wr ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.tape)⟩)⟩
  | N + 1, rs, p => stepK D dg J wr K rootB pts N rs p.toStop (next p.toStop)
    (fun s c => roundsK next rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
      (p.toStop.rewrite (M.round p.toStop J s c)).consume) (D.sub J p.toStop.modes)

/-- **The session's projection, then what follows it** (`sessionP`) from the position `p`: at the next stop,
`Commit`'s decoded roots; on the link points, written into the tape there, the rounds; then `next` from the position
the session's walk ends at. -/
noncomputable def sessionK (next : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a)) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Strategy (((liveSession dg J mPts N).map wr).bind K) :=
  ⟨D.root p.toStop.modes, fun pts => roundsK D M dg J wr K next (D.root p.toStop.modes) pts N (fun _ _ => [])
    (p.toStop.rewrite (M.link p.toStop pts)).consume⟩

/-- **One round's joint law, then what follows the session** (`stepW`). -/
noncomputable def stepWK {ο : Type} (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (fin : τ → Game ο)
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game ο) : Option (SubmitJ J) → Game ο
  | none => fin (wr ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ q.tape)⟩)
  | some s => .coin (Fin s.n → 𝔽) (next s)

/-- **The rounds' joint law, then what follows the session** (`roundsW`) from the position `p`: over the rounds' live
coins, `next` at the position the walk ends at and the session's transcript. -/
noncomputable def roundsWK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → Game ο
  | 0, rs, p => next p (wr ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.tape)⟩)
  | N + 1, rs, p => stepWK D J wr rootB pts rs p.toStop (next p.toStop)
    (fun s c => roundsWK next rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
      (p.toStop.rewrite (M.round p.toStop J s c)).consume) (D.sub J p.toStop.modes)

/-- **The session's joint law, then what follows it** (`sessionW`) from the position `p`. -/
noncomputable def sessionWK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Game ο :=
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts => roundsWK D M dg J wr next (D.root p.toStop.modes) pts N (fun _ _ => [])
    (p.toStop.rewrite (M.link p.toStop pts)).consume

/-- **The session's projection, then what follows it, has its events' probabilities under the session's joint law**
(`prob_sessionP`), wherever what follows has. -/
theorem prob_sessionK (E : γ → Prop) (nextP : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a))
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game (γ × CoinTape g σ))
    (h : ∀ q a (w : Strategy (nextW q a)), prob E (K a) (nextP q a) = prob (fun o => E o.1) (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (sessionWK D M dg J wr nextW N p)) :
    prob E (((liveSession dg J mPts N).map wr).bind K) (sessionK D M dg J wr K nextP N p) =
      prob (fun o => E o.1) (sessionWK D M dg J wr nextW N p) w := by
  sorry

open Classical in
/-- **One stop's share off `B`, then what follows the session** (`stopOff`): `fin` at the session's end there; else
the mean over the round's coins of `e` from the tape with them written in, those `B` holds of at the stop counted
out. -/
noncomputable def stopOffK (fin e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (q : Pos g σ Claims.Msg (Fin R)) :
    Option (SubmitJ J) → ℝ≥0∞
  | none => fin q
  | some s => avg fun c : Fin s.n → 𝔽 => if B q.g' J s c then 0 else e (q.rewrite (M.round q J s c)).consume

/-- **The rounds' share off `B`, then what follows the session** (`roundsOff`) from the position `p`: `e` at the
position the walk ends at, the plays through a stop whose round's coins `B` holds of counted out. -/
noncomputable def roundsOffK (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) : ℕ → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | 0, p => e p
  | N + 1, p => stopOffK M J B e (roundsOffK e N) p.toStop (D.sub J p.toStop.modes)

open Classical in
/-- **The session's share off `BL` and `B`, then what follows it** (`sessionOff`) from the position `p`. -/
noncomputable def sessionOffK (BL : LinkHold mPts) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : ℝ≥0∞ :=
  avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
    if BL p.toStop.g' pts then 0 else roundsOffK D M J B e N (p.toStop.rewrite (M.link p.toStop pts)).consume

/-- **The rounds' joint law's support off `B`, then what follows the session, as a trace** (`RoundsReach`) from the
position `p`: the session's end after the last round or at a stop whose decoding submits nothing, and then `next`'s
trace from the position there at the session's transcript; or a submission decoded at the stop, any of its coins `B`
doesn't hold of there, and the trace from the tape with them written in. -/
inductive RoundsReachK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    ℕ → (Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → ο → Prop
  | zero (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (o : ο)
      (hn : next p (wr ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.tape)⟩) o) :
      RoundsReachK next rootB pts 0 rs p o
  | stop (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (o : ο)
      (h : D.sub J p.toStop.modes = none)
      (hn : next p.toStop (wr ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.toStop.tape)⟩) o) :
      RoundsReachK next rootB pts (N + 1) rs p o
  | round (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (s : SubmitJ J)
      (c : Fin s.n → 𝔽) (o : ο) (h : D.sub J p.toStop.modes = some s) (hB : ¬ B p.toStop.g' J s c)
      (ho : RoundsReachK next rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
        (p.toStop.rewrite (M.round p.toStop J s c)).consume o) :
      RoundsReachK next rootB pts (N + 1) rs p o

/-- **The session's joint law's support off `BL` and `B`, then what follows it**, from the position `p`: link points
`BL` doesn't hold of at the session's first stop, and the rounds' trace from the tape with them written in. -/
def SessionReachK {ο : Type} (BL : LinkHold mPts) (next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) (o : ο) : Prop :=
  ∃ pts : Fin 2 → Fin mPts → 𝔽, ¬ BL p.toStop.g' pts ∧
    RoundsReachK D M dg J B wr next (D.root p.toStop.modes) pts N (fun _ _ => [])
      (p.toStop.rewrite (M.link p.toStop pts)).consume o

end SessionK

/-! ## The parameters over a window -/

/-- **The coin map a window's projection reads**, at a strategy `σ` of the period's game `g`: the widening of the
sample's coins (`T`) and the period's tape on the sample's coins and the widening (`start`), once for the window; and,
for the member at place `i`, the period's tape from one of its session's stops with its live link points (`link`) or
a round's live coins (`round`) written in. -/
structure WindowCoinMap (g : Game Exchange) (σ : Strategy g) (Ω : Type) (R : ℕ) where
  T : Type
  [fT : Fintype T]
  [nT : Nonempty T]
  start : Ω × T → CoinTape g σ
  link : ℕ → (p : Pos g σ Claims.Msg (Fin R)) → (mPts : ℕ) → (Fin 2 → Fin mPts → 𝔽) → CoinTape p.g' p.s' →
    CoinTape p.g' p.s'
  round : ℕ → (p : Pos g σ Claims.Msg (Fin R)) → (J : ℕ) → (s : SubmitJ J) → (Fin s.n → 𝔽) → CoinTape p.g' p.s' →
    CoinTape p.g' p.s'

attribute [instance] WindowCoinMap.fT WindowCoinMap.nT

/-- The widening a window's coin map reads. -/
def WindowCoinMap.widening {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ} (M : WindowCoinMap g σ Ω R) :
    Widening :=
  ⟨M.T⟩

/-- **The coin map of the member at place `i`**, at its session's `mPts` link points. -/
def WindowCoinMap.member {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ} (M : WindowCoinMap g σ Ω R)
    (i mPts : ℕ) : CoinMap g σ Ω R mPts where
  T := M.T
  fT := M.fT
  nT := M.nT
  start := M.start
  link p := M.link i p mPts
  round := M.round i

/-- **The events at the members' first stops' link points that the joint law is held off**, by the member's place. -/
abbrev WindowLinkHold : Type 1 := ℕ → (mPts : ℕ) → LinkHold mPts

/-- **The events at the members' later stops' round coins that the joint law is held off**, by the member's place. -/
abbrev WindowRoundHold : Type 1 := ℕ → RoundHold

/-! ## The sessions, in order -/

section Seq

open FlockSoundness.Discharge.Exec (Transcript mPtsOf)

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} (D : ℕ → Decoders R Reg)
  (M : WindowCoinMap g σ Ω R) (BL : WindowLinkHold) (B : WindowRoundHold) (S : Finset (Fin nJ)) (Rg : Reg)

/-- A member's session's link points' count. -/
abbrev memberPts (m : Flock.Guarantees.MemberV5 Reg nJ) : ℕ :=
  mPtsOf m.D.unroll.g (m.kd / FlockSoundness.Discharge.Composed.nTab m.I)

/-- A member's session's tables at the sample `S` and the registration `Rg`. -/
noncomputable abbrev memberTabs (m : Flock.Guarantees.MemberV5 Reg nJ) : ℕ :=
  (FlockSoundness.Discharge.ZkHidden.planZHJP m.dj m.y₀ (m.part S) Rg).length

/-- **The sessions' projection, in the members' order** from the position `p`: the member at place `i`'s session at
its decoders and coin map on its part of the sample, then the rest from the position its walk ends at. -/
noncomputable def seqP : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    Strategy (FlockSoundness.Game.seqG (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l)
  | _, [], _ => PUnit.unit
  | i, m :: l, p =>
    sessionK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun a => (FlockSoundness.Game.seqG
        (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l).map (a :: ·))
      (fun q _ => Strategy.map _ _ (seqP (i + 1) l q)) m.N p

/-- **The sessions' joint law, in the members' order** from the position `p`: the members' transcripts beside the
period's tape, over the sessions' live coins. -/
noncomputable def seqW : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    Game (List Transcript × CoinTape g σ)
  | _, [], p => .ret ([], p.tape)
  | i, m :: l, p =>
    sessionWK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun q a => (seqW (i + 1) l q).map fun o => (a :: o.1, o.2)) m.N p

/-- **An event of the period's tape under the sessions' joint law off the held-off events** from the position `p`:
each member's session's share off its own, then the rest's from the position its walk ends at. -/
noncomputable def seqOff (Q : CoinTape g σ → Prop) :
    (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | _, [], p => tapeInd Q p
  | i, m :: l, p =>
    sessionOffK (D i) (M.member i (memberPts m)) (memberTabs S Rg m) (B i) (BL i (memberPts m))
      (fun q => seqOff Q (i + 1) l q) m.N p

/-- **The sessions' joint law's support off the held-off events** from the position `p`: each member's session's
trace off its own, ending at its transcript, then the rest's from the position its walk ends at. -/
def SeqReach : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    List Transcript × CoinTape g σ → Prop
  | _, [], p, o => o = ([], p.tape)
  | i, m :: l, p, o =>
    SessionReachK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (B i) (m.wr (m.part S) Rg)
      (BL i (memberPts m)) (fun q a o' => ∃ ts, SeqReach (i + 1) l q (ts, o'.2) ∧ o'.1 = a :: ts) m.N p o

/-- **The sessions' projection has its events' probabilities under the sessions' joint law.** -/
theorem prob_seqP (E : List Transcript → Prop) :
    ∀ (i : ℕ) (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (p : Pos g σ Claims.Msg (Fin R))
      (w : Strategy (seqW D M S Rg i l p)),
      prob E (FlockSoundness.Game.seqG (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l)
          (seqP D M S Rg i l p) =
        prob (fun o => E o.1) (seqW D M S Rg i l p) w := by
  sorry

/-- **The sessions' joint law off the held-off events gives an event of the tape at most `E`'s probability**
wherever, on its support off them (`SeqReach`), the event implies `E` of the members' transcripts. -/
theorem seqOff_le_prob (Q : CoinTape g σ → Prop) (E : List Transcript → Prop) :
    ∀ (i : ℕ) (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (p : Pos g σ Claims.Msg (Fin R))
      (w : Strategy (seqW D M S Rg i l p)),
      (∀ o, SeqReach D M BL B S Rg i l p o → Q o.2 → E o.1) →
      seqOff D M BL B S Rg Q i l p ≤ prob (fun o => E o.1) (seqW D M S Rg i l p) w := by
  sorry

end Seq

/-! ## The projection on the window's audit -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The registration the window's projection sends**: the first member's decoding of the record's most likely
message, the joint registration every member's session reads. -/
noncomputable def windowReg (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) : ByteArray :=
  (D 0).reg ((recordCut V θ F c rd).cut.modeHere σ)

/-- **The window's projection** of `σ` at the window `W`, `t` SHA-512 calls, the reading `rd`, the members' decoders
`D` and the coin map `M`: the registration decoded from the record's most likely message; on the sample's widened
coins `ω`, the sessions' projection in the members' order from the play's start on `M`'s tape at `ω`, at the rounds'
cuts. -/
noncomputable def windowP (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) :
    FlockSoundness.Game.Strategy ((W.extend M.widening).joint t).game :=
  ⟨windowReg V θ F c W rd σ D, fun ω =>
    FlockSoundness.Game.Strategy.map _ _
      (seqP D M ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0 ((W.extend M.widening).joint t).ms
        (Pos.root (roundCuts V θ F c rd) (M.start ω)))⟩

/-- **The joint law the window's projection simulates**: over the window's audit's coins, the projected play's outcome
beside the period's tape `σ` is played on. -/
noncomputable def windowG (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) :
    Game (W.Out × CoinTape (V.game θ F c) σ) :=
  .coin (W.extend M.widening).L.Ω fun ω =>
    (seqW D M ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0 ((W.extend M.widening).joint t).ms
      (Pos.root (roundCuts V θ F c rd) (M.start ω))).map
      fun o => ((windowReg V θ F c W rd σ D, (W.extend M.widening).L.draw ω, o.1), o.2)

/-- **The window's projected play's events have their probabilities under the joint law.** -/
theorem prob_windowP (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (E : W.Out → Prop)
    (w : Strategy (windowG V θ F c W t rd σ D M)) :
    FlockSoundness.Game.prob E ((W.extend M.widening).joint t).game (windowP V θ F c W t rd σ D M) =
      prob (fun o => E o.1) (windowG V θ F c W t rd σ D M) w := by
  sorry

/-- **The joint law's support off the held-off events**: the outcomes at the decoded registration and the sample of
some widened coins `ω`, with the sessions' trace off them from the play's start on `M`'s tape at `ω`, ending on the
outcome's transcripts and tape. -/
def WindowReaches (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) (o : W.Out × CoinTape (V.game θ F c) σ) : Prop :=
  ∃ ω : (W.extend M.widening).L.Ω, o.1.1 = windowReg V θ F c W rd σ D ∧
    o.1.2.1 = (W.extend M.widening).L.draw ω ∧
    SeqReach D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω)) (o.1.2.2, o.2)

/-- **An event of the period's tape under the joint law off the held-off events**: over the widened coins, the
sessions' share off them from the play's start (`seqOff`). -/
noncomputable def windowOff (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) (Q : CoinTape (V.game θ F c) σ → Prop) : ℝ≥0∞ :=
  avg fun ω : (W.extend M.widening).L.Ω =>
    seqOff D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) Q 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω))

/-- **The joint law off the held-off events gives an event of the tape at most `E`'s probability** wherever, on its
support off them (`WindowReaches`), the event implies `E` of the outcome. -/
theorem windowOff_le_prob (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) (Q : CoinTape (V.game θ F c) σ → Prop) (E : W.Out → Prop)
    (h : ∀ o, WindowReaches V θ F c W t rd σ D M BL B o → Q o.2 → E o.1)
    (w : Strategy (windowG V θ F c W t rd σ D M)) :
    windowOff V θ F c W t rd σ D M BL B Q ≤ prob (fun o => E o.1) (windowG V θ F c W t rd σ D M) w := by
  sorry

/-! ## The logged coupling at the window's projection -/

/-- **The coin map's premise at the window, off the held-off events**: at every event `Q` of the period's tape, the
joint law's tape off them is within the hiding term of the uniform tape. -/
def WindowCouples (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    windowOff V θ F c W t rd σ D M BL B Q + W.hidingTerm

/-- **The decoders' premise at the window**, at the composite `k` and the property `P` of the play: on the joint
law's support off the held-off events, a tape whose play's sampled verdict accepts, that misses at no round and not at
the record, and whose play `P` holds of, is beside an outcome the window's audit accepts: flock-verify `--zk` accepts
the transcript of every member with a proof block in the sample (`JointAuditV5.Accepts`). -/
def WindowDecodes (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ℕ → Decoders W.rounds ByteArray)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold) : Prop :=
  ∀ o, WindowReaches V θ F c W t rd σ D M BL B o →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2) ∧ NoMiss V θ F c rd σ o.2 ∧
      P (CoinTape.run (V.game θ F c) σ o.2) →
    ((W.extend M.widening).joint t).Accepts o.1

/-- **The logged coupling at the window's projection**: an accepted play `P` holds of, on a tape that misses at no
round and not at the record, is at most as likely as the window's audit's acceptance at the projected strategy, plus
the hiding term, wherever, off the held-off events, the coin map's premise and the decoders' hold. -/
theorem windowLogged (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ℕ → Decoders W.rounds ByteArray)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold)
    (hC : WindowCouples V θ F c W t rd σ D M BL B) (hD : WindowDecodes V θ F c k P W t rd σ D M BL B) :
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c rd σ x ∧
          P (CoinTape.run (V.game θ F c) σ x)) ≤
      FlockSoundness.Game.prob (fun o => ((W.extend M.widening).joint t).Accepts o)
          ((W.extend M.widening).joint t).game (windowP V θ F c W t rd σ D M) +
        W.hidingTerm := by
  sorry

/-! ## The coupling at the window -/

/-- A reading of each window's rounds: every member's session's, in the members' order. -/
abbrev WindowReadings := (W : Window) → ByteArray → Array ByteArray → ByteArray → Reading W.rounds

/-- **A round's two-continuation finder at the window**: the round's cut's, at `2t + 2h` SHA-512 calls on every
coin. -/
noncomputable def windowMsgFinder (rd : WindowReadings) (W : Window) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) (j : Fin W.rounds) : Claims.Finder :=
  (roundCut V θ F c (rd W θ F c) j).cut.finder σ (2 * V.cost θ F c σ + 2 * W.h)

/-- **The record's two-continuation finder at the window**: the record's cut's, at `2t + 2h` SHA-512 calls on every
coin. -/
noncomputable def windowRecordFinder (rd : WindowReadings) (W : Window) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) : Claims.Finder :=
  (recordCut V θ F c (rd W θ F c)).cut.finder σ (2 * V.cost θ F c σ + 2 * W.h)

omit θ F c in
theorem windowMsgFinder_calls (rd : WindowReadings) : ∀ (W : Window) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) (j : Fin W.rounds) (x : (windowMsgFinder V rd W θ F c σ j).S),
    ((windowMsgFinder V rd W θ F c σ j).cost x : ℝ) ≤ W.qM (V.cost θ F c σ) := fun W θ F c σ j _ => by
  show ((2 * V.cost θ F c σ + 2 * W.h : ℕ) : ℝ) ≤ W.qM (V.cost θ F c σ)
  simp [Window.qM]

omit θ F c in
theorem windowRecordFinder_calls (rd : WindowReadings) : ∀ (W : Window) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) (x : (windowRecordFinder V rd W θ F c σ).S),
    ((windowRecordFinder V rd W θ F c σ).cost x : ℝ) ≤ W.qM (V.cost θ F c σ) := fun W θ F c σ _ => by
  show ((2 * V.cost θ F c σ + 2 * W.h : ℕ) : ℝ) ≤ W.qM (V.cost θ F c σ)
  simp [Window.qM]

/-- **The window's constructions with the projection and the messages' finders** at the readings `rd`, the members'
decoders `D` and the coin maps `M`: the coins are `M`'s widening, the projection `windowP`, the rounds' and the
record's finders their cuts'; `K`'s read finder, forger and bound digests' finder. -/
noncomputable def WindowConstructions.withProject (K : WindowConstructions V) (rd : WindowReadings)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      ℕ → Decoders W.rounds ByteArray)
    (M : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) : WindowConstructions V :=
  { K with
    coins := fun W θ F c σ => (M W θ F c σ).widening
    project := fun W θ F c σ => windowP V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ)
    msgFinder := windowMsgFinder V rd
    recordFinder := windowRecordFinder V rd
    msg_calls := windowMsgFinder_calls V rd
    record_calls := windowRecordFinder_calls V rd }

variable (k : Verity.Registration.Composite)

omit θ F c in
/-- **The coupling at the window's projection**: at the constructions with the projection and the messages' finders
at the readings `rd`, `WindowCouplingTarget` holds at the window `W` wherever (a) an accepted play `P` doesn't hold
of is at most as likely as the read finder's collision plus the forger's forgery, (a') one `Q` at its commitment and
strategy doesn't hold of is at most as likely as the square root of the bound digests' finder's success, and, at each
commitment and strategy off its held-off events, the coin map's premise and the decoders' hold at the two. -/
theorem windowCouplingTarget_of_logged (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      ℕ → Decoders W.rounds ByteArray)
    (M : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (W : Window) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (c : ByteArray) → Strategy (V.game θ F c) → WindowLinkHold)
    (B : (c : ByteArray) → Strategy (V.game θ F c) → WindowRoundHold)
    (ha : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hC : ∀ c σ, WindowCouples V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ) (BL c σ)
      (B c σ))
    (hD : ∀ c σ, WindowDecodes V θ F c k (fun o => P o ∧ Q c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
      (M W θ F c σ) (BL c σ) (B c σ)) :
    WindowCouplingTarget V k (K.withProject V rd D M) W θ F := by
  sorry

end Verity.Sampled
