module

public import Proofs.Properties.Verity.Sampled.Window.Walk
public import Proofs.Properties.Verity.Sampled.CoinMap

/-!
# A window's one sample, its members' parts, and the coin map over the window

The period's one sample covers a window's population of all its members' proof blocks; each member's session proves
the member's part of it (`Member.part`), and the window's projection hands each member's session that part
(`seqP`). What the leg reads of the one sample and of the coin map over the members' sessions:

* **The members' parts** (`Member.mem_part`, `Window.Partitions`, `Window.sum_part_card`): a member's part of a
  sample is its proof blocks the sample holds, and where the members' places partition the population the parts'
  sizes add up to the sample's.
* **The marginals** (`Window.avg_design`, `Window.drawMiss_extend`, `Window.eps_extend`,
  `Window.sessionDelta_extend`): member `k`'s sampling design in Flock's window is its part of the window's one
  sample, a mean of a reading of the part over the sample's coins; the members' knowledge and binding terms read only
  that, so the window's error is the same at the window widened by the coins a projection reads.
* **The coin map's premise within a term** (`stopErrK`, `roundsErrK`, `sessionErrK`, `seqErr`, `windowErr`,
  `WindowStartWithin`, `WindowFirstStopFresh`, `WindowCouplesWithin`, `windowCouples_of_close`): `CoinMap`'s
  (`couples_of_close`), over the members' sessions in order. Wherever each member's coin map writes its link points
  in within `γL` of the tape's own and each round's coins within `γ`, the start leaves the tape from the first stop
  fresh and is within `ε₀` of the uniform tape, the joint law's tape off the held-off events is within `ε₀` and the
  stops' term, every member's session's summed along the walk, of the uniform tape at every event; so the coin map's
  premise holds wherever that is at most the hiding term (`windowCouples_of_within`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Refine (roundOf)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The members' parts of the one sample -/

/-- **A member's part of a sample is its proof blocks the sample holds.** -/
theorem Member.mem_part {nJ : ℕ} (m : Member nJ) (S : Finset (Fin nJ)) (u : Fin m.n) :
    u ∈ m.part S ↔ m.emb u ∈ S := by
  sorry

/-- **The members' places partition the window's population**: each member's proof blocks sit at distinct places,
no two members share a place, and every place is some member's. -/
structure Window.Partitions (W : Window) : Prop where
  inj : ∀ m ∈ W.ms, Function.Injective m.emb
  disjoint : W.ms.Pairwise fun m m' => ∀ u u', m.emb u ≠ m'.emb u'
  cover : ∀ v : Fin W.nJ, ∃ m ∈ W.ms, ∃ u, m.emb u = v

/-- **The members' parts of a sample have its size between them**, where the members' places partition the
population. -/
theorem Window.sum_part_card (W : Window) (h : W.Partitions) (S : Finset (Fin W.nJ)) :
    (W.ms.map fun m => (m.part S).card).sum = S.card := by
  sorry

/-! ## The marginals -/

/-- **Member `k`'s sampling design is its part of the window's one sample**: a mean of a reading of the member's
sample over its design's coins is the mean, over the window's sample's coins, of the reading at the member's part. -/
theorem Window.avg_design (W : Window) (t : ℕ) (k : Fin (W.joint t).ms.length)
    (G : Finset (Fin ((W.joint t).ms.get k).n) → ℝ≥0∞) :
    avg (fun ω : ((W.joint t).design k).Ω => G (((W.joint t).design k).draw ω)) =
      avg fun ω : W.L.Ω => G (((W.joint t).ms.get k).part (W.L.draw ω)) := by
  sorry

/-- The sample's term doesn't read the widening. -/
theorem Window.drawMiss_extend (W : Window) (S : Widening) : (W.extend S).drawMiss = W.drawMiss := by
  sorry

/-- **The members' terms read only their parts of the one sample**: the members' knowledge and binding terms summed
are the same at the window widened by the coins `S`. -/
theorem Window.eps_extend (W : Window) (S : Widening) (t : ℕ) : ((W.extend S).joint t).eps = (W.joint t).eps := by
  sorry

/-- **The window's error doesn't read the widening** (`Live.sessionDelta_extend`). -/
theorem Window.sessionDelta_extend (W : Window) (S : Widening) (t : ℕ) :
    (W.extend S).sessionDelta t = W.sessionDelta t := by
  unfold Window.sessionDelta
  rw [W.drawMiss_extend S, W.eps_extend S t]

/-! ## The coin map's premise within a term, over the members' sessions -/

section SessionErr

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : Decoders R Reg) (J : ℕ)
  (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞)

/-- **One stop's error, then what follows the session** (`stopErr`): `fin` at the session's end there; `γ` there and
the next rounds' error over the tapes from the stop where a submission is decoded. -/
noncomputable def stopErrK (fin e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (q : Pos g σ Claims.Msg (Fin R)) :
    Option (SubmitJ J) → ℝ≥0∞
  | none => fin q
  | some s => γ q J s + Pos.resample e q.consume

/-- **The rounds' error, then what follows the session** (`roundsErr`) from the position `p`: `γ` at each stop a
uniform tape from `p` reaches, and `e` at the position the session's walk ends at. -/
noncomputable def roundsErrK (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) : ℕ → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | 0, p => e p
  | N + 1, p => stopErrK J γ e (roundsErrK e N) p.toStop (D.sub J p.toStop.modes)

/-- **The session's error, then what follows it**, from the position `p`: `γL` at the session's first stop and the
rounds' error over the tapes from it. -/
noncomputable def sessionErrK (γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) : ℝ≥0∞ :=
  γL p.toStop + Pos.resample (roundsErrK D J γ e N) p.toStop.consume

end SessionErr

section SeqErr

variable {g : Game Exchange} {σ : Strategy g} {Reg : Type} {R nJ : ℕ} (D : ℕ → Decoders R Reg)
  (γ : ℕ → Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞)
  (γL : ℕ → (mPts : ℕ) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (S : Finset (Fin nJ)) (Rg : Reg)

/-- **The sessions' error, in the members' order** from the position `p`: each member's session's, at its own `γ`
and `γL`, then the rest's from the position its walk ends at. -/
noncomputable def seqErr : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    ℝ≥0∞
  | _, [], _ => 0
  | i, m :: l, p =>
    sessionErrK (D i) (memberTabs S Rg m) (γ i) (γL i (memberPts m)) (fun q => seqErr (i + 1) l q) m.N p

end SeqErr

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The window's coin map's premise within `ε`, off the held-off events** (`CouplesWithin`). -/
def WindowCouplesWithin (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) (ε : ℝ≥0∞) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    windowOff V θ F c W t rd σ D M BL B Q + ε

/-- **The window's coin map's premise at a term within the hiding term.** -/
theorem windowCouples_of_within {W : Window} {t : ℕ} {rd : Reading W.rounds} {σ : Strategy (V.game θ F c)}
    {D : ℕ → Decoders W.rounds ByteArray} {M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds} {BL : WindowLinkHold}
    {B : WindowRoundHold} {ε : ℝ≥0∞} (h : WindowCouplesWithin V θ F c W t rd σ D M BL B ε)
    (hε : ε ≤ W.hidingTerm) : WindowCouples V θ F c W t rd σ D M BL B := fun Q =>
  (h Q).trans (add_le_add le_rfl hε)

/-- **The window's coin map's start is within `ε₀` of the uniform tape** (`StartWithin`). -/
def WindowStartWithin (W : Window) (σ : Strategy (V.game θ F c))
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (ε₀ : ℝ≥0∞) : Prop :=
  ∀ H : CoinTape (V.game θ F c) σ → ℝ≥0∞, (∀ x, H x ≤ 1) →
    avg H ≤ avg (fun ω : (W.extend M.widening).L.Ω => H (M.start ω)) + ε₀

/-- **The window's projection's first stop** on the widened coins `ω`: where the first member's `Commit`'s roots are
decoded. -/
noncomputable def windowFirstStop (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (ω : (W.extend M.widening).L.Ω) :
    Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) :=
  (Pos.root (roundCuts V θ F c rd) (M.start ω)).toStop

/-- **The window's coin map's start leaves the tape from the first stop fresh** (`FirstStopFresh`): over the widened
coins, any reading of the sample and the projection's first stop has the mean of its average over the tapes from the
stop. -/
def WindowFirstStopFresh (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) : Prop :=
  ∀ G : Finset (Fin W.nJ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞,
    avg (fun ω : (W.extend M.widening).L.Ω =>
        G ((W.extend M.widening).L.draw ω) (windowFirstStop V θ F c W rd σ M ω)) =
      avg fun ω : (W.extend M.widening).L.Ω =>
        Pos.resample (G ((W.extend M.widening).L.draw ω)) (windowFirstStop V θ F c W rd σ M ω)

/-- **The stops' term over the window**: over the widened coins, every member's session's error along the walk from
the play's start, in the members' order (`seqErr`). -/
noncomputable def windowErr (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (γ : ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞)
    (γL : ℕ → (mPts : ℕ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞) : ℝ≥0∞ :=
  avg fun ω : (W.extend M.widening).L.Ω =>
    seqErr D γ γL ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω))

open Classical in
/-- **The window's coin map's premise within a term, off its local ones** (`couples_of_close`): the joint law's tape
off the held-off events is within the start's term and the stops' of the uniform tape at every event, wherever each
member's coin map writes its link points in within `γL` of the tape's own and each round's coins within `γ`, the start
leaves the tape from the first stop fresh, and the start is within `ε₀` of the uniform tape. The stops' term carries,
beside each member's `γL` and `γ`, the probability of its held-off events at each stop. -/
theorem windowCouples_of_close (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {γ : ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞}
    {γL : ℕ → (mPts : ℕ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞} {ε₀ : ℝ≥0∞}
    (BL : WindowLinkHold) (B : WindowRoundHold) (hR : ∀ i mPts, (M.member i mPts).RoundClose (γ i))
    (hL : ∀ i mPts, (M.member i mPts).LinkClose (γL i mPts)) (hF : WindowFirstStopFresh V θ F c W rd σ M)
    (hS : WindowStartWithin V θ F c W σ M ε₀) :
    WindowCouplesWithin V θ F c W t rd σ D M BL B (ε₀ + windowErr V θ F c W t rd σ D M
      (fun i q j s => γ i q j s + prCoin (B i q.g' j s)) (fun i mPts q => γL i mPts q + prCoin (BL i mPts q.g'))) := by
  sorry

end Verity.Sampled
