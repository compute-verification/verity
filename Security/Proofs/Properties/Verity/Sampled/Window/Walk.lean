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
session's end it sends the member's proof files, which its decoders give from the position the walk is at; it then
consumes the stop after it, the session's closing round's, with nothing written in (a closing round releases no
coin), and the next member's session starts from the position past it. A session that ends early (a stop that
decodes no submission) leaves the walk at that stop, which is consumed the same way. A member with no proof block in
the sample has no statement, so the period's play holds no session of its own: the projection plays an idle session
for it, which consumes no stop, and the next member's session starts from the same position.

* **A member's decoders** (`PosDecoders`): `Decoders`, with the proof files a function of the position the member's
  session's walk ends at. The window's audit takes a member's files before the next member's coins, and the period's
  game takes every member's after every coin, so a member's files are decoded from what the play has fixed by that
  position; the files of the play on the tape there (`Decoders.atEnd`) are the play's own only at the last member.
* **A session, then what follows it** (`roundsThen`, `stepK`, `roundsK`, `sessionK`): the session's projection of
  `Project` (`stepP`, `roundsP`, `sessionP`), with the strategy of what follows the session taken at the position the
  session's walk ends at. Its joint law (`stepWK`, `roundsWK`, `sessionWK`), its share off the held-off events
  (`stopOffK`, `roundsOffK`, `sessionOffK`) and its support off them (`RoundsReachK`, `SessionReachK`) are `Project`'s
  with the same continuation.
* **A member's session, or none** (`memberK`, `memberWK`, `memberOffK`, `MemberReachK`): the session's where the
  sample holds one of the member's proof blocks, with what follows it taken past the session's closing round's stop
  (`Pos.consume` of the stop after the session's walk); else an idle session (`idleOut`, `skipK`, `skipWK`), whose
  joint law flips only its link points and whose share and support are what follows the member's, from the same
  position.
* **The parameters** (`WindowCoinMap`): one widening and one start for the window, and each member's link points and
  round coins written at its own stops (`WindowCoinMap.member`). The decoders are a member's each, by its place in the
  window.
* **The sessions in order** (`seqP`, `seqW`, `seqOff`, `SeqReach`): member `i`'s session at its decoders and coin map
  where the sample holds one of its proof blocks (`memberDrawn`), then the rest from where it ends.
* **The window's projection** (`windowP`) and **the joint law** it simulates (`windowG`): the projected play's
  outcome beside the period's tape; the projected play's events have their probabilities under it (`prob_windowP`).
* **The two premises** off the held-off events (`WindowCouples`, the coin map's; `WindowDecodes`, the decoders'), the
  logged coupling from them (`windowLogged`), and the coupling at the window with the messages' finders
  (`windowCouplingTarget_of_logged`).
-/

@[expose] public section

namespace Verity.Game.Pos

variable {α V ι : Type} {g : Game α} {σ : Strategy g}

/-- **A stop is its own next stop**: the walk from a node at which a cut is fixed, or from the play's end, stays
there. -/
theorem toStop_of_stops (q : Pos g σ V ι) (h : (∃ i, (q.Ks i).isHere) ∨ ∃ a, q.g' = .ret a) : q.toStop = q := by
  obtain ⟨g', s', Ks, x, back⟩ := q
  show walk g' s' Ks x back = _
  cases g' with
  | ret a => rfl
  | send M k =>
    rcases h with h | ⟨a, ha⟩
    · rw [walk, ite_eq_left_of_eq_true _ _ (eq_true h)]
    · cases ha
  | coin C k =>
    rcases h with h | ⟨a, ha⟩
    · rw [walk, ite_eq_left_of_eq_true _ _ (eq_true h)]
    · cases ha

/-- **The next stop's next stop is itself.** -/
theorem toStop_toStop (p : Pos g σ V ι) : p.toStop.toStop = p.toStop := toStop_of_stops _ p.toStop_stops

end Verity.Game.Pos

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Refine (roundOf)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## A member's decoders -/

/-- **The decoders a window's projection reads for one member**, at a strategy `σ` of the period's game `g`: the
registration, `Commit`'s roots and a round's submission as `Decoders` has them, and the member's session's proof
files, from the position its walk ends at (`files`). -/
structure PosDecoders (g : Game Exchange) (σ : Strategy g) (R : ℕ) (Reg : Type) where
  reg : Option Claims.Msg → Reg
  root : (Fin R → Option Claims.Msg) → Array ByteArray
  sub : (J : ℕ) → (Fin R → Option Claims.Msg) → Option (SubmitJ J)
  files : Pos g σ Claims.Msg (Fin R) → Array ByteArray

/-- **One session's decoders as a member's**: the proof files at a position are those of the play's output on the
tape there. -/
def Decoders.atEnd {g : Game Exchange} {σ : Strategy g} {R : ℕ} {Reg : Type} (D : Decoders R Reg) :
    PosDecoders g σ R Reg where
  reg := D.reg
  root := D.root
  sub := D.sub
  files q := D.proofs (CoinTape.run g σ q.tape)

/-! ## A session, then what follows it -/

section SessionK

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : PosDecoders g σ R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ) (B : RoundHold) {τ γ : Type}
  (wr : LiveOutJ J mPts → τ) (K : τ → Game γ)

/-- **A session's rounds, its writer, then what follows the session**: the game the rounds' projection plays from
the rounds `rs` so far, at the session's roots and link points. -/
abbrev roundsThen (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) : Game γ :=
  (((liveRoundsJ dg J N rs).bind (KLJ rootB pts)).map wr).bind K

/-- **One round's projection, then what follows the session** (`stepP`): at the stop `q`, at the submission decoded
there: the session's end, with the member's proof files at the stop and then `fin` at the session's transcript, or
the submission and then `next` on its coins. -/
noncomputable def stepK (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (fin : (a : τ) → Strategy (K a))
    (next : (s : SubmitJ J) → (c : Fin s.n → 𝔽) →
      Strategy (roundsThen dg J wr K rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))) :
    Option (SubmitJ J) → Strategy (roundsThen dg J wr K rootB pts (N + 1) rs)
  | none => ⟨none, ⟨D.files q,
      fin (wr ⟨rootB, pts, rs, D.files q⟩)⟩⟩
  | some s => ⟨some s, next s⟩

/-- **The rounds' projection, then what follows the session** (`roundsP`) from the position `p`: at the next stop, the
decoded submission; on its coins, written into the tape there, the next round; after the last, the member's proof
files at the position the walk is at, and `next` from there. -/
noncomputable def roundsK (next : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a)) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) →
      Strategy (roundsThen dg J wr K rootB pts N rs)
  | 0, rs, p => ⟨D.files p,
      next p (wr ⟨rootB, pts, rs, D.files p⟩)⟩
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
  | none => fin (wr ⟨rootB, pts, rs, D.files q⟩)
  | some s => .coin (Fin s.n → 𝔽) (next s)

/-- **The rounds' joint law, then what follows the session** (`roundsW`) from the position `p`: over the rounds' live
coins, `next` at the position the walk ends at and the session's transcript. -/
noncomputable def roundsWK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → Game ο
  | 0, rs, p => next p (wr ⟨rootB, pts, rs, D.files p⟩)
  | N + 1, rs, p => stepWK D J wr rootB pts rs p.toStop (next p.toStop)
    (fun s c => roundsWK next rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
      (p.toStop.rewrite (M.round p.toStop J s c)).consume) (D.sub J p.toStop.modes)

/-- **The session's joint law, then what follows it** (`sessionW`) from the position `p`. -/
noncomputable def sessionWK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Game ο :=
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts => roundsWK D M dg J wr next (D.root p.toStop.modes) pts N (fun _ _ => [])
    (p.toStop.rewrite (M.link p.toStop pts)).consume

/-- One round's joint law's strategy: what follows the session's, at its end. -/
def stepWKS {ο : Type} (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (fin : τ → Game ο)
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game ο) (wf : ∀ a, Strategy (fin a))
    (w : ∀ s c, Strategy (next s c)) : (o : Option (SubmitJ J)) → Strategy (stepWK D J wr rootB pts rs q fin next o)
  | none => wf _
  | some s => w s

/-- The rounds' joint law's strategy: what follows the session's, at its end. -/
noncomputable def roundsWKS {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (wn : ∀ q a, Strategy (next q a)) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → (p : Pos g σ Claims.Msg (Fin R)) →
      Strategy (roundsWK D M dg J wr next rootB pts N rs p)
  | 0, _, _ => wn _ _
  | N + 1, rs, p => stepWKS D J wr rootB pts rs p.toStop (next p.toStop) _ (wn p.toStop)
    (fun _ _ => roundsWKS next wn rootB pts N _ _) (D.sub J p.toStop.modes)

/-- The session's joint law's strategy: what follows the session's, at its end. -/
noncomputable def sessionWKS {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (wn : ∀ q a, Strategy (next q a)) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    Strategy (sessionWK D M dg J wr next N p) :=
  fun pts => roundsWKS D M dg J wr next wn _ pts N _ _

/-- **One round's projection, then what follows the session, has its events' probabilities under one round's joint
law** (`prob_stepP`), wherever what follows the session and `next` have. -/
theorem prob_stepK (E : γ → Prop) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (finP : (a : τ) → Strategy (K a))
    (finW : τ → Game (γ × CoinTape g σ))
    (nextP : (s : SubmitJ J) → (c : Fin s.n → 𝔽) →
      Strategy (roundsThen dg J wr K rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))))
    (nextW : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (γ × CoinTape g σ))
    (hf : ∀ a (w : Strategy (finW a)), prob E (K a) (finP a) = prob (fun o => E o.1) (finW a) w)
    (h : ∀ s c (w : Strategy (nextW s c)),
      prob E (roundsThen dg J wr K rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))) (nextP s c) =
        prob (fun o => E o.1) (nextW s c) w) :
    ∀ (o : Option (SubmitJ J)) (w : Strategy (stepWK D J wr rootB pts rs q finW nextW o)),
      prob E (roundsThen dg J wr K rootB pts (N + 1) rs) (stepK D dg J wr K rootB pts N rs q finP nextP o) =
        prob (fun o => E o.1) (stepWK D J wr rootB pts rs q finW nextW o) w
  | none, w => hf _ w
  | some s, w => Finset.sum_congr rfl fun c _ => congrArg (HMul.hMul _) (h s c (w c))

/-- **The rounds' projection, then what follows the session, has its events' probabilities under the rounds' joint
law** (`prob_roundsP`), wherever what follows has. -/
theorem prob_roundsK (E : γ → Prop) (nextP : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a))
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game (γ × CoinTape g σ))
    (h : ∀ q a (w : Strategy (nextW q a)), prob E (K a) (nextP q a) = prob (fun o => E o.1) (nextW q a) w)
    (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (w : Strategy (roundsWK D M dg J wr nextW rootB pts N rs p)),
      prob E (roundsThen dg J wr K rootB pts N rs) (roundsK D M dg J wr K nextP rootB pts N rs p) =
        prob (fun o => E o.1) (roundsWK D M dg J wr nextW rootB pts N rs p) w
  | 0, _, p, w => h p _ w
  | N + 1, rs, p, w =>
    prob_stepK D dg J wr K E rootB pts N rs p.toStop (nextP p.toStop) (nextW p.toStop) _ _ (h p.toStop)
      (fun _ _ w => prob_roundsK E nextP nextW h rootB pts N _ _ w) (D.sub J p.toStop.modes) w

/-- **The session's projection, then what follows it, has its events' probabilities under the session's joint law**
(`prob_sessionP`), wherever what follows has. -/
theorem prob_sessionK (E : γ → Prop) (nextP : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a))
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game (γ × CoinTape g σ))
    (h : ∀ q a (w : Strategy (nextW q a)), prob E (K a) (nextP q a) = prob (fun o => E o.1) (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (sessionWK D M dg J wr nextW N p)) :
    prob E (((liveSession dg J mPts N).map wr).bind K) (sessionK D M dg J wr K nextP N p) =
      prob (fun o => E o.1) (sessionWK D M dg J wr nextW N p) w :=
  Finset.sum_congr rfl fun pts _ => congrArg (HMul.hMul _)
    (prob_roundsK D M dg J wr K E nextP nextW h _ pts N _ _ (w pts))

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
      (hn : next p (wr ⟨rootB, pts, rs, D.files p⟩) o) :
      RoundsReachK next rootB pts 0 rs p o
  | stop (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (o : ο)
      (h : D.sub J p.toStop.modes = none)
      (hn : next p.toStop (wr ⟨rootB, pts, rs, D.files p.toStop⟩) o) :
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

open Classical in
variable {D M J B wr} in
/-- **One stop's share off `B`, then what follows the session, is at most the event's probability under one round's
joint law** (`stopOff_le_prob`), wherever the session's end's share is at most its probability under what follows, and
after each of the round's coins `B` doesn't hold of, `e` there is at most its probability under `next`. -/
theorem stopOffK_le_prob {ο : Type} (E : ο → Prop) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R))
    (fin e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (finW : τ → Game ο)
    (nextW : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game ο) :
    ∀ (o : Option (SubmitJ J)) (w : Strategy (stepWK D J wr rootB pts rs q finW nextW o)),
      (o = none → ∀ w : Strategy (finW (wr ⟨rootB, pts, rs, D.files q⟩)),
        fin q ≤ prob E (finW (wr ⟨rootB, pts, rs, D.files q⟩)) w) →
      (∀ s, o = some s → ∀ c (w : Strategy (nextW s c)), ¬ B q.g' J s c →
        e (q.rewrite (M.round q J s c)).consume ≤ prob E (nextW s c) w) →
      stopOffK M J B fin e q o ≤ prob E (stepWK D J wr rootB pts rs q finW nextW o) w
  | none, w, h0, _ => h0 rfl w
  | some s, w, _, hs => avg_mono fun c => by
    split_ifs with hB
    · exact zero_le
    · exact hs s rfl c (w c) hB

variable {D M dg J B wr} in
/-- **The rounds' share off `B`, then what follows the session, is at most the event's probability under the rounds'
joint law** (`roundsOff_le_prob`) wherever, on the support off `B` (`RoundsReachK`), `H` implies `E`: at each position
the session ends at, with what follows the session at most `E`'s probability there wherever `H` implies `E` on its own
support. -/
theorem roundsOffK_le_prob {ο : Type} (E H : ο → Prop) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (nextR : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop)
    (hn : ∀ q a (w : Strategy (nextW q a)), (∀ o, nextR q a o → H o → E o) → e q ≤ prob E (nextW q a) w)
    (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (w : Strategy (roundsWK D M dg J wr nextW rootB pts N rs p)),
      (∀ o, RoundsReachK D M dg J B wr nextR rootB pts N rs p o → H o → E o) →
      roundsOffK D M J B e N p ≤ prob E (roundsWK D M dg J wr nextW rootB pts N rs p) w
  | 0, rs, p, w, h => hn p _ w fun o ho => h o (.zero rs p o ho)
  | N + 1, rs, p, w, h =>
    stopOffK_le_prob E rootB pts rs p.toStop e (roundsOffK D M J B e N) (nextW p.toStop) _ (D.sub J p.toStop.modes) w
      (fun h0 w => hn p.toStop _ w fun o ho => h o (.stop N rs p o h0 ho))
      (fun s hs c w hB => roundsOffK_le_prob E H e nextW nextR hn rootB pts N _ _ w
        fun o ho => h o (.round N rs p s c o hs hB ho))

open Classical in
variable {D M dg J B wr} in
/-- **The session's share off `BL` and `B`, then what follows it, is at most the event's probability under the
session's joint law** (`sessionOff_le_prob`) wherever, on the support off them (`SessionReachK`), `H` implies `E`. -/
theorem sessionOffK_le_prob {ο : Type} (BL : LinkHold mPts) (E H : ο → Prop)
    (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (nextR : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop)
    (hn : ∀ q a (w : Strategy (nextW q a)), (∀ o, nextR q a o → H o → E o) → e q ≤ prob E (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (sessionWK D M dg J wr nextW N p))
    (h : ∀ o, SessionReachK D M dg J B wr BL nextR N p o → H o → E o) :
    sessionOffK D M J B BL e N p ≤ prob E (sessionWK D M dg J wr nextW N p) w :=
  avg_mono fun pts => by
    split_ifs with hL
    · exact zero_le
    · exact roundsOffK_le_prob E H e nextW nextR hn _ pts N _ _ (w pts) fun o ho => h o ⟨pts, hL, ho⟩

/-! ### A member's session, or none -/

/-- **An idle session's outcome** at the link points `pts`: no root, no round and no proof file. It is the outcome
the projection gives the session of a member with no proof block in the sample; such a member has no statement, so
nothing reads it. -/
def idleOut (pts : Fin 2 → Fin mPts → 𝔽) : LiveOutJ J mPts :=
  ⟨#[], pts, fun _ _ => [], #[]⟩

/-- **An idle session's rounds, then what follows the session**: no submission and no proof file, then `fin` at the
session's transcript. It reads nothing of the period's play. -/
noncomputable def idleK (fin : (a : τ) → Strategy (K a)) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Strategy (roundsThen dg J wr K rootB pts N rs)
  | 0, rs => ⟨#[], fin (wr ⟨rootB, pts, rs, #[]⟩)⟩
  | _ + 1, rs => ⟨none, ⟨#[], fin (wr ⟨rootB, pts, rs, #[]⟩)⟩⟩

/-- **An idle session, then what follows it**: `Commit` with no root, no round and no proof file, then `fin` at the
idle outcome on the session's link points. -/
noncomputable def skipK (fin : (a : τ) → Strategy (K a)) (N : ℕ) :
    Strategy (((liveSession dg J mPts N).map wr).bind K) :=
  ⟨#[], fun pts => idleK dg J wr K fin #[] pts N (fun _ _ => [])⟩

/-- **An idle session's joint law, then what follows it**: over the idle session's link points, `fin` at the idle
outcome on them. -/
noncomputable def skipWK {ο : Type} (fin : τ → Game ο) : Game ο :=
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts => fin (wr (idleOut J pts))

/-- **An idle session's rounds, then what follows the session, have the events' probabilities of what follows.** -/
theorem prob_idleK (E : γ → Prop) (fin : (a : τ) → Strategy (K a)) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round),
      prob E (roundsThen dg J wr K rootB pts N rs) (idleK dg J wr K fin rootB pts N rs) =
        prob E (K (wr ⟨rootB, pts, rs, #[]⟩)) (fin (wr ⟨rootB, pts, rs, #[]⟩))
  | 0, _ => rfl
  | _ + 1, _ => rfl

/-- **An idle session, then what follows it, has its events' probabilities under the idle session's joint law**,
wherever what follows has. -/
theorem prob_skipK (E : γ → Prop) (finP : (a : τ) → Strategy (K a)) (finW : τ → Game (γ × CoinTape g σ))
    (h : ∀ a (w : Strategy (finW a)), prob E (K a) (finP a) = prob (fun o => E o.1) (finW a) w) (N : ℕ)
    (w : Strategy (skipWK J wr finW)) :
    prob E (((liveSession dg J mPts N).map wr).bind K) (skipK dg J wr K finP N) =
      prob (fun o => E o.1) (skipWK J wr finW) w :=
  Finset.sum_congr rfl fun pts _ => congrArg (HMul.hMul _)
    ((prob_idleK dg J wr K E finP #[] pts N fun _ _ => []).trans (h (wr (idleOut J pts)) (w pts)))

/-- **A member's projection, then what follows it**: its session's (`sessionK`) where the sample holds one of its
proof blocks (`true`), then what follows from the position past the stop after the session's walk, its closing
round's, consumed with nothing written in (the closing round releases no coin); else an idle session (`skipK`), which
consumes no stop of the period's play, and what follows from the same position. A member with no proof block in the
sample has no statement, so the period's play holds no session of its own. -/
noncomputable def memberK (next : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a)) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Bool → Strategy (((liveSession dg J mPts N).map wr).bind K)
  | true => sessionK D M dg J wr K (fun q => next q.toStop.consume) N p
  | false => skipK dg J wr K (next p) N

/-- **A member's joint law, then what follows it**: its session's (`sessionWK`) where the sample holds one of its
proof blocks, then what follows from the position past the session's closing round's stop; else the idle session's
(`skipWK`), and what follows from the same position. -/
noncomputable def memberWK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Bool → Game ο
  | true => sessionWK D M dg J wr (fun q => next q.toStop.consume) N p
  | false => skipWK J wr (next p)

/-- A member's joint law's strategy: what follows the member's, at its end. -/
noncomputable def memberWKS {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (wn : ∀ q a, Strategy (next q a)) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    (b : Bool) → Strategy (memberWK D M dg J wr next N p b)
  | true => sessionWKS D M dg J wr (fun q => next q.toStop.consume) (fun q => wn q.toStop.consume) N p
  | false => fun _ => wn p _

/-- **A member's projection, then what follows it, has its events' probabilities under the member's joint law**,
wherever what follows has. -/
theorem prob_memberK (E : γ → Prop) (nextP : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a))
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game (γ × CoinTape g σ))
    (h : ∀ q a (w : Strategy (nextW q a)), prob E (K a) (nextP q a) = prob (fun o => E o.1) (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ (b : Bool) (w : Strategy (memberWK D M dg J wr nextW N p b)),
      prob E (((liveSession dg J mPts N).map wr).bind K) (memberK D M dg J wr K nextP N p b) =
        prob (fun o => E o.1) (memberWK D M dg J wr nextW N p b) w
  | true, w => prob_sessionK D M dg J wr K E (fun q => nextP q.toStop.consume) (fun q => nextW q.toStop.consume)
      (fun q => h q.toStop.consume) N p w
  | false, w => prob_skipK dg J wr K E (nextP p) (nextW p) (h p) N w

/-- **A member's share off `BL` and `B`, then what follows it**: its session's (`sessionOffK`) where the sample holds
one of its proof blocks, then what follows from the position past the session's closing round's stop; else what
follows, from the same position. -/
noncomputable def memberOffK (BL : LinkHold mPts) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Bool → ℝ≥0∞
  | true => sessionOffK D M J B BL (fun q => e q.toStop.consume) N p
  | false => e p

/-- **A member's joint law's support off `BL` and `B`, then what follows it**, from the position `p`: its session's
(`SessionReachK`) where the sample holds one of its proof blocks, then `next`'s trace from the position past the
session's closing round's stop; else any link points of the idle session, and `next`'s trace from `p` at the idle
outcome on them. -/
def MemberReachK {ο : Type} (BL : LinkHold mPts) (next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) (o : ο) : Bool → Prop
  | true => SessionReachK D M dg J B wr BL (fun q => next q.toStop.consume) N p o
  | false => ∃ pts : Fin 2 → Fin mPts → 𝔽, next p (wr (idleOut J pts)) o

variable {D M dg J B wr} in
/-- **A member's share off `BL` and `B`, then what follows it, is at most the event's probability under the member's
joint law** wherever, on the support off them (`MemberReachK`), `H` implies `E`. -/
theorem memberOffK_le_prob {ο : Type} (BL : LinkHold mPts) (E H : ο → Prop)
    (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (nextR : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop)
    (hn : ∀ q a (w : Strategy (nextW q a)), (∀ o, nextR q a o → H o → E o) → e q ≤ prob E (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ (b : Bool) (w : Strategy (memberWK D M dg J wr nextW N p b)),
      (∀ o, MemberReachK D M dg J B wr BL nextR N p o b → H o → E o) →
      memberOffK D M J B BL e N p b ≤ prob E (memberWK D M dg J wr nextW N p b) w
  | true, w, h => sessionOffK_le_prob BL E H (fun q => e q.toStop.consume) (fun q => nextW q.toStop.consume)
      (fun q => nextR q.toStop.consume) (fun q => hn q.toStop.consume) N p w h
  | false, w, h =>
    (avg_const (C := Fin 2 → Fin mPts → 𝔽) (e p)).symm.le.trans
      (avg_mono fun pts => hn p (wr (idleOut J pts)) (w pts) fun o ho => h o ⟨pts, ho⟩)

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

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} (D : ℕ → PosDecoders g σ R Reg)
  (M : WindowCoinMap g σ Ω R) (BL : WindowLinkHold) (B : WindowRoundHold) (S : Finset (Fin nJ)) (Rg : Reg)

/-- A member's session's link points' count. -/
abbrev memberPts (m : Flock.Guarantees.MemberV5 Reg nJ) : ℕ :=
  mPtsOf m.D.unroll.g (m.kd / FlockSoundness.Discharge.Composed.nTab m.I)

/-- A member's session's tables at the sample `S` and the registration `Rg`. -/
noncomputable abbrev memberTabs (m : Flock.Guarantees.MemberV5 Reg nJ) : ℕ :=
  (FlockSoundness.Discharge.ZkHidden.planZHJP m.dj m.y₀ (m.part S) Rg).length

open Classical in
/-- **Whether the sample `S` holds one of the member's proof blocks**: whether the member has a statement at `S`, and
so a session in the period's play. -/
noncomputable def memberDrawn (m : Flock.Guarantees.MemberV5 Reg nJ) : Bool :=
  decide (m.part S).Nonempty

open Classical in
variable {S} in
/-- A member is drawn exactly where its part of the sample is not empty. -/
theorem memberDrawn_iff {m : Flock.Guarantees.MemberV5 Reg nJ} : memberDrawn S m = true ↔ (m.part S).Nonempty :=
  decide_eq_true_iff

variable {S} in
/-- A member that is not drawn has an empty part of the sample. -/
theorem not_nonempty_of_memberDrawn {m : Flock.Guarantees.MemberV5 Reg nJ} (h : memberDrawn S m = false) :
    ¬ (m.part S).Nonempty :=
  fun hne => Bool.false_ne_true (h.symm.trans (memberDrawn_iff.2 hne))

/-- **The sessions' projection, in the members' order** from the position `p`: the member at place `i`'s session at
its decoders and coin map on its part of the sample, then the rest from the position its walk ends at. A member with
no proof block in the sample plays an idle session, and the rest starts from the same position (`memberK`). -/
noncomputable def seqP : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    Strategy (FlockSoundness.Game.seqG (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l)
  | _, [], _ => PUnit.unit
  | i, m :: l, p =>
    memberK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun a => (FlockSoundness.Game.seqG
        (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l).map (a :: ·))
      (fun q _ => Strategy.map _ _ (seqP (i + 1) l q)) m.N p (memberDrawn S m)

/-- **The sessions' joint law, in the members' order** from the position `p`: the members' transcripts beside the
period's tape, over the sessions' live coins. -/
noncomputable def seqW : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    Game (List Transcript × CoinTape g σ)
  | _, [], p => .ret ([], p.tape)
  | i, m :: l, p =>
    memberWK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun q a => (seqW (i + 1) l q).map fun o => (a :: o.1, o.2)) m.N p (memberDrawn S m)

/-- **An event of the period's tape under the sessions' joint law off the held-off events** from the position `p`:
each member's session's share off its own, then the rest's from the position its walk ends at. -/
noncomputable def seqOff (Q : CoinTape g σ → Prop) :
    (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | _, [], p => tapeInd Q p
  | i, m :: l, p =>
    memberOffK (D i) (M.member i (memberPts m)) (memberTabs S Rg m) (B i) (BL i (memberPts m))
      (fun q => seqOff Q (i + 1) l q) m.N p (memberDrawn S m)

/-- **The sessions' joint law's support off the held-off events** from the position `p`: each member's session's
trace off its own, ending at its transcript, then the rest's from the position its walk ends at; at a member with no
proof block in the sample, an idle outcome's transcript and the rest's from the same position. -/
def SeqReach : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    List Transcript × CoinTape g σ → Prop
  | _, [], p, o => o = ([], p.tape)
  | i, m :: l, p, o =>
    MemberReachK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (B i) (m.wr (m.part S) Rg)
      (BL i (memberPts m)) (fun q a o' => ∃ ts, SeqReach (i + 1) l q (ts, o'.2) ∧ o'.1 = a :: ts) m.N p o
      (memberDrawn S m)

/-- The sessions' joint law's strategy: it has no message to send. -/
noncomputable def seqWS : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) →
    (p : Pos g σ Claims.Msg (Fin R)) → Strategy (seqW D M S Rg i l p)
  | _, [], _ => PUnit.unit
  | i, m :: l, p =>
    memberWKS (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun q a => (seqW D M S Rg (i + 1) l q).map fun o => (a :: o.1, o.2))
      (fun q _ => Strategy.map _ _ (seqWS (i + 1) l q)) m.N p (memberDrawn S m)

/-- **The sessions' projection has its events' probabilities under the sessions' joint law.** -/
theorem prob_seqP : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (E : List Transcript → Prop) (i : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (seqW D M S Rg i l p)),
    prob E (FlockSoundness.Game.seqG (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l)
        (seqP D M S Rg i l p) =
      prob (fun o => E o.1) (seqW D M S Rg i l p) w
  | [], _, _, _, _ => rfl
  | m :: l, E, i, p, w =>
    prob_memberK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun a => (FlockSoundness.Game.seqG
        (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l).map (a :: ·)) E
      (fun q _ => Strategy.map _ _ (seqP D M S Rg (i + 1) l q))
      (fun q a => (seqW D M S Rg (i + 1) l q).map fun o => (a :: o.1, o.2))
      (fun q a w => (FlockSoundness.Discharge.Exec.prob_strategy_map _ _ _ _).trans
        ((prob_seqP l (fun ts => E (a :: ts)) (i + 1) q (Strategy.ofMap _ _ w)).trans
          (prob_map (fun o : List Transcript × CoinTape g σ => E o.1)
            (fun o : List Transcript × CoinTape g σ => (a :: o.1, o.2)) _ w).symm))
      m.N p (memberDrawn S m) w

open Classical in
/-- **The sessions' joint law off the held-off events gives an event of the tape at most `E`'s probability**
wherever, on its support off them (`SeqReach`), the event implies `E` of the members' transcripts. -/
theorem seqOff_le_prob (Q : CoinTape g σ → Prop) : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ))
    (E : List Transcript → Prop) (i : ℕ) (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (seqW D M S Rg i l p)),
    (∀ o, SeqReach D M BL B S Rg i l p o → Q o.2 → E o.1) →
    seqOff D M BL B S Rg Q i l p ≤ prob (fun o => E o.1) (seqW D M S Rg i l p) w
  | [], E, _, p, _, h => by
    show tapeInd Q p ≤ if E [] then 1 else 0
    unfold tapeInd
    by_cases hQ : Q p.tape
    · rw [ite_eq_left hQ, ite_eq_left (h ([], p.tape) rfl hQ)]
    · rw [ite_eq_right hQ]
      exact zero_le
  | m :: l, E, i, p, w, h =>
    memberOffK_le_prob (D := D i) (M := M.member i (memberPts m)) (dg := m.dg) (J := memberTabs S Rg m) (B := B i)
      (wr := m.wr (m.part S) Rg) (BL i (memberPts m)) (fun o => E o.1) (fun o => Q o.2)
      (fun q => seqOff D M BL B S Rg Q (i + 1) l q)
      (fun q a => (seqW D M S Rg (i + 1) l q).map fun o => (a :: o.1, o.2))
      (fun q a o' => ∃ ts, SeqReach D M BL B S Rg (i + 1) l q (ts, o'.2) ∧ o'.1 = a :: ts)
      (fun q a w hq => (seqOff_le_prob Q l (fun ts => E (a :: ts)) (i + 1) q (Strategy.ofMap _ _ w)
          fun o ho hQ => hq (a :: o.1, o.2) ⟨o.1, ho, rfl⟩ hQ).trans_eq
        (prob_map (fun o : List Transcript × CoinTape g σ => E o.1)
            (fun o : List Transcript × CoinTape g σ => (a :: o.1, o.2)) _ w).symm)
      m.N p (memberDrawn S m) w h

end Seq

/-! ## The projection on the window's audit -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **A window's members' decoders**, by a member's place in the window, at a strategy `σ` of the period's game. -/
abbrev WindowDecoders (W : Window) (σ : Strategy (V.game θ F c)) : Type 1 :=
  ℕ → PosDecoders (V.game θ F c) σ W.rounds ByteArray

/-- **The registration the window's projection sends**: the first member's decoding of the record's most likely
message, the joint registration every member's session reads. -/
noncomputable def windowReg (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) : ByteArray :=
  (D 0).reg ((recordCut V θ F c rd).cut.modeHere σ)

/-- **The window's projection** of `σ` at the window `W`, `t` SHA-512 calls, the reading `rd`, the members' decoders
`D` and the coin map `M`: the registration decoded from the record's most likely message; on the sample's widened
coins `ω`, the sessions' projection in the members' order from the play's start on `M`'s tape at `ω`, at the rounds'
cuts. -/
noncomputable def windowP (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) :
    FlockSoundness.Game.Strategy ((W.extend M.widening).joint t).game :=
  ⟨windowReg V θ F c W rd σ D, fun ω =>
    FlockSoundness.Game.Strategy.map _ _
      (seqP D M ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0 ((W.extend M.widening).joint t).ms
        (Pos.root (roundCuts V θ F c rd) (M.start ω)))⟩

/-- **The joint law the window's projection simulates**: over the window's audit's coins, the projected play's outcome
beside the period's tape `σ` is played on. -/
noncomputable def windowG (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) :
    Game (W.Out × CoinTape (V.game θ F c) σ) :=
  .coin (W.extend M.widening).L.Ω fun ω =>
    (seqW D M ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0 ((W.extend M.widening).joint t).ms
      (Pos.root (roundCuts V θ F c rd) (M.start ω))).map
      fun o => ((windowReg V θ F c W rd σ D, (W.extend M.widening).L.draw ω, o.1), o.2)

/-- **The window's projected play's events have their probabilities under the joint law.** -/
theorem prob_windowP (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (E : W.Out → Prop)
    (w : Strategy (windowG V θ F c W t rd σ D M)) :
    FlockSoundness.Game.prob E ((W.extend M.widening).joint t).game (windowP V θ F c W t rd σ D M) =
      prob (fun o => E o.1) (windowG V θ F c W t rd σ D M) w := by
  refine Finset.sum_congr rfl fun ω _ => congrArg (HMul.hMul _) ?_
  exact (FlockSoundness.Discharge.Exec.prob_strategy_map _ _ _ _).trans
    ((prob_seqP D M ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D)
        ((W.extend M.widening).joint t).ms
        (fun ts => E (windowReg V θ F c W rd σ D, (W.extend M.widening).L.draw ω, ts)) 0
        (Pos.root (roundCuts V θ F c rd) (M.start ω)) (Strategy.ofMap _ _ (w ω))).trans
      (prob_map (fun o : W.Out × CoinTape (V.game θ F c) σ => E o.1)
        (fun o : List FlockSoundness.Discharge.Exec.Transcript × CoinTape (V.game θ F c) σ =>
          ((windowReg V θ F c W rd σ D, (W.extend M.widening).L.draw ω, o.1), o.2)) _ (w ω)).symm)

/-- The joint law's strategy: it has no message to send. -/
noncomputable def windowGS (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) :
    Strategy (windowG V θ F c W t rd σ D M) :=
  fun ω => Strategy.map _ _ (seqWS D M ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0
    ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω)))

/-- **The joint law's support off the held-off events**: the outcomes at the decoded registration and the sample of
some widened coins `ω`, with the sessions' trace off them from the play's start on `M`'s tape at `ω`, ending on the
outcome's transcripts and tape. -/
def WindowReaches (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) (o : W.Out × CoinTape (V.game θ F c) σ) : Prop :=
  ∃ ω : (W.extend M.widening).L.Ω, o.1.1 = windowReg V θ F c W rd σ D ∧
    o.1.2.1 = (W.extend M.widening).L.draw ω ∧
    SeqReach D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω)) (o.1.2.2, o.2)

/-- **An event of the period's tape under the joint law off the held-off events**: over the widened coins, the
sessions' share off them from the play's start (`seqOff`). -/
noncomputable def windowOff (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) (Q : CoinTape (V.game θ F c) σ → Prop) : ℝ≥0∞ :=
  avg fun ω : (W.extend M.widening).L.Ω =>
    seqOff D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) Q 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω))

/-- **The joint law off the held-off events gives an event of the tape at most `E`'s probability** wherever, on its
support off them (`WindowReaches`), the event implies `E` of the outcome. -/
theorem windowOff_le_prob (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) (Q : CoinTape (V.game θ F c) σ → Prop) (E : W.Out → Prop)
    (h : ∀ o, WindowReaches V θ F c W t rd σ D M BL B o → Q o.2 → E o.1)
    (w : Strategy (windowG V θ F c W t rd σ D M)) :
    windowOff V θ F c W t rd σ D M BL B Q ≤ prob (fun o => E o.1) (windowG V θ F c W t rd σ D M) w :=
  avg_mono fun ω =>
    (seqOff_le_prob D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) Q
        ((W.extend M.widening).joint t).ms
        (fun ts => E (windowReg V θ F c W rd σ D, (W.extend M.widening).L.draw ω, ts)) 0
        (Pos.root (roundCuts V θ F c rd) (M.start ω)) (Strategy.ofMap _ _ (w ω))
        fun o ho hQ => h ((windowReg V θ F c W rd σ D, (W.extend M.widening).L.draw ω, o.1), o.2)
          ⟨ω, rfl, rfl, ho⟩ hQ).trans_eq
      (prob_map (fun o : W.Out × CoinTape (V.game θ F c) σ => E o.1)
        (fun o : List FlockSoundness.Discharge.Exec.Transcript × CoinTape (V.game θ F c) σ =>
          ((windowReg V θ F c W rd σ D, (W.extend M.widening).L.draw ω, o.1), o.2)) _ (w ω)).symm

/-! ## The logged coupling at the window's projection -/

/-- **The coin map's premise at the window, off the held-off events**: at every event `Q` of the period's tape, the
joint law's tape off them is within the hiding term of the uniform tape. -/
def WindowCouples (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    windowOff V θ F c W t rd σ D M BL B Q + W.hidingTerm

/-- **The decoders' premise at the window**, at the composite `k` and the property `P` of the play: on the joint
law's support off the held-off events, a tape whose play's sampled verdict accepts, that misses at no round and not at
the record, and whose play `P` holds of, is beside an outcome the window's audit accepts: flock-verify `--zk` accepts
the transcript of every member with a proof block in the sample (`JointAuditV5.Accepts`). -/
def WindowDecodes (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : WindowDecoders V θ F c W σ)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold) : Prop :=
  ∀ o, WindowReaches V θ F c W t rd σ D M BL B o →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2) ∧ NoMiss V θ F c rd σ o.2 ∧
      P (CoinTape.run (V.game θ F c) σ o.2) →
    ((W.extend M.widening).joint t).Accepts o.1

/-- **The logged coupling at the window's projection**: an accepted play `P` holds of, on a tape that misses at no
round and not at the record, is at most as likely as the window's audit's acceptance at the projected strategy, plus
the hiding term, wherever, off the held-off events, the coin map's premise and the decoders' hold. -/
theorem windowLogged (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : WindowDecoders V θ F c W σ)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold)
    (hC : WindowCouples V θ F c W t rd σ D M BL B) (hD : WindowDecodes V θ F c k P W t rd σ D M BL B) :
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c rd σ x ∧
          P (CoinTape.run (V.game θ F c) σ x)) ≤
      FlockSoundness.Game.prob (fun o => ((W.extend M.widening).joint t).Accepts o)
          ((W.extend M.widening).joint t).game (windowP V θ F c W t rd σ D M) +
        W.hidingTerm := by
  have h := (hC fun x => V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c rd σ x ∧
      P (CoinTape.run (V.game θ F c) σ x)).trans
    (add_le_add (windowOff_le_prob V θ F c W t rd σ D M BL B
      (fun x => V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c rd σ x ∧
        P (CoinTape.run (V.game θ F c) σ x))
      (fun o => ((W.extend M.widening).joint t).Accepts o) hD (windowGS V θ F c W t rd σ D M)) le_rfl)
  rw [← prob_windowP V θ F c W t rd σ D M (fun o => ((W.extend M.widening).joint t).Accepts o)] at h
  exact h

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
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
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
/-- **The messages' bound at a window's constructions whose two-continuation finders are the cuts'**
(`windowMsgFinder`, `windowRecordFinder`): an event of the period's game is at most as likely as the event on the
tapes that miss at no round and not at the record, plus `WindowCouplingTarget`'s messages terms. -/
theorem prob_le_windowMessages (rd : WindowReadings) (K : WindowConstructions V)
    (hM : K.msgFinder = windowMsgFinder V rd) (hR : K.recordFinder = windowRecordFinder V rd) (W : Window)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)) (E : Exchange → Prop) :
    prob E (V.game θ F c) σ ≤
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          E (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x) +
        (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder W θ F c σ j).Pr sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.recordFinder W θ F c σ).Pr sha512Msg))) := by
  rw [hM, hR]
  exact prob_le_noMiss_add V θ F c (rd W θ F c) σ _ E

omit θ F c in
/-- **The coupling at the window off the messages' misses** (`couplingTarget_of_noMiss`): at constructions whose
two-continuation finders are the cuts', `WindowCouplingTarget` holds once the sampled verdict on the tapes that miss at
no round and not at the record is at most as likely as the window's audit's acceptance at the projected strategy, plus
the square root of the bound digests' finder's success, the hiding term, the read finder's collision and the
forger's forgery. -/
theorem windowCouplingTarget_of_noMiss (rd : WindowReadings) (K : WindowConstructions V)
    (hM : K.msgFinder = windowMsgFinder V rd) (hR : K.recordFinder = windowRecordFinder V rd) (W : Window)
    (θ : ByteArray) (F : Array ByteArray)
    (h : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x) ≤
        FlockSoundness.Game.prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
            ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
          (ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)) + W.hidingTerm +
            ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
            ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))) :
    WindowCouplingTarget V k K W θ F := fun c σ =>
  (prob_le_windowMessages V rd K hM hR W θ F c σ _).trans <| (add_le_add (h c σ) le_rfl).trans_eq (by ring)

omit θ F c in
/-- **The accepted plays that miss nowhere, off the binding, the digest and the logged coupling, at the window**
(`noMiss_le_of_logged`): each is one `P` and `Q` hold of, bounded by `hb`, or one `P` doesn't hold of, bounded by
`ha`, or one `Q` doesn't, bounded by `hq`. -/
theorem windowNoMiss_le_of_logged (rd : WindowReadings) (K : WindowConstructions V) (W : Window)
    (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (ha : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hb : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x ∧
            (P (CoinTape.run (V.game θ F c) σ x) ∧ Q c σ (CoinTape.run (V.game θ F c) σ x))) ≤
        FlockSoundness.Game.prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
            ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
          W.hidingTerm)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) :
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x) ≤
      FlockSoundness.Game.prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
          ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
        (ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)) + W.hidingTerm +
          ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify)) := by
  have hsplit : Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x) ≤
      Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x ∧
            (P (CoinTape.run (V.game θ F c) σ x) ∧ Q c σ (CoinTape.run (V.game θ F c) σ x))) +
        (Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
            V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ ¬ P (CoinTape.run (V.game θ F c) σ x)) +
          Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
            V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧
              ¬ Q c σ (CoinTape.run (V.game θ F c) σ x))) := by
    refine (Claims.Pr_mono fun x hx => ?_).trans ((pr_or_le _ _).trans (add_le_add le_rfl (pr_or_le _ _)))
    by_cases hp : P (CoinTape.run (V.game θ F c) σ x)
    · by_cases hQ : Q c σ (CoinTape.run (V.game θ F c) σ x)
      · exact Or.inl ⟨hx.1, hx.2, hp, hQ⟩
      · exact Or.inr (Or.inr ⟨hx.1, hQ⟩)
    · exact Or.inr (Or.inl ⟨hx.1, hp⟩)
  have hnot := ha c σ
  rw [prob_eq_ofReal_pr] at hnot
  have hnq := hq c σ
  rw [prob_eq_ofReal_pr] at hnq
  refine ((ENNReal.ofReal_le_ofReal hsplit).trans
    (ENNReal.ofReal_add_le.trans (add_le_add le_rfl ENNReal.ofReal_add_le))).trans ?_
  refine (add_le_add (hb c σ) (add_le_add hnot hnq)).trans_eq ?_
  ring

omit θ F c in
/-- **The coupling at the window's projection**: at the constructions with the projection and the messages' finders
at the readings `rd`, `WindowCouplingTarget` holds at the window `W` wherever (a) an accepted play `P` doesn't hold
of is at most as likely as the read finder's collision plus the forger's forgery, (a') one `Q` at its commitment and
strategy doesn't hold of is at most as likely as the square root of the bound digests' finder's success, and, at each
commitment and strategy off its held-off events, the coin map's premise and the decoders' hold at the two. -/
theorem windowCouplingTarget_of_logged (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
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
    WindowCouplingTarget V k (K.withProject V rd D M) W θ F :=
  windowCouplingTarget_of_noMiss V k rd (K.withProject V rd D M) rfl rfl W θ F
    (windowNoMiss_le_of_logged V k rd (K.withProject V rd D M) W P θ F Q ha hq fun c σ =>
      windowLogged V θ F c k (fun o => P o ∧ Q c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (M W θ F c σ) (BL c σ) (B c σ) (hC c σ) (hD c σ))

end Verity.Sampled
