module

public import Proofs.Properties.Verity.Sampled.Window.Draw
public import Proofs.Properties.Verity.Sampled.CouplingAt

/-!
# Sampled proofs' coupling over a window: the stops' term within the hiding term

The window's stops' term (`windowErr`) charges each member's stops their openings' gaps and empty classes, in the
members' order. Each is within the class error of the node the stop is at (`openErr`), the class errors along a play
of the period's game are within the potential from no tree open (`avg_pathCharge_game_le`), and that potential is the
hiding term however many sessions the play runs. So the one-session bound (`couplesErr_le_of_charge`) holds over a
window as it stands:

* **The sessions' error within the charge** (`avg_roundsErrK_le`, `sessionErrK_le`, `seqErr_le`): where each stop's
  term is within `Γ` at the stop's node, a session's error, then what follows it, is within the charge along the play
  from the session's first stop, averaged over the tapes from it; what follows a session is the next member's, so the
  members' errors in order are too.
* **The stops' term within the charge along a play** (`windowErr_le_of_charge`): over a start that leaves the tape
  from the first stop fresh and is the uniform tape.
* **Off the empty classes at their counts** (`windowSampleMap_couples_at`, `windowErr_sampleMap_le_at`): the
  sample-coupled coin map's premise off `linkEmptyAt` and `roundEmptyAt` at every member's stops holds within the
  stops' term, and that term is within the hiding term at every strategy.
* **The coupling** (`windowCouplingTarget_of_sample_at`): `windowCouplingTarget_of_logged` at the sample-coupled coin
  maps off those events, with no bound on the stops' term left.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The sessions' error within the charge -/

/-- **The rounds' error, then what follows the session, within the charge** (`avg_roundsErr_le`): where `γ` at each
stop the projection reaches from the cuts `Ks₀`, at each submission decoded there, is within `Γ` at the stop's node,
and what follows the session, `e`, at a position the projection reaches is within the charge from the position's next
stop averaged over the tapes from that stop, the rounds' error from a position the projection reaches is within the
charge from the next stop, over the tapes from the position; the play's end costs nothing. -/
theorem avg_roundsErrK_le {g : Game Exchange} {σ : Strategy g} {R : ℕ} {Reg : Type} (D : PosDecoders g σ R Reg)
    (J : ℕ) (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞) (Γ : Game Exchange → ℝ≥0∞)
    (hret : ∀ a, Γ (.ret a) = 0) (Ks₀ : Fin R → DigestCut Claims.Msg g)
    (hγ : ∀ q, Pos.Reached Ks₀ q → ∀ s, D.sub J q.modes = some s → γ q J s ≤ Γ q.g')
    {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞}
    (he : ∀ q, Pos.Reached Ks₀ q → e q ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) q.toStop) (N : ℕ) :
    ∀ p, Pos.Reached Ks₀ p → Pos.resample (roundsErrK D J γ e N) p ≤ Pos.resample (Pos.stopCharge Γ) p := by
  have key : ∀ (G : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (p : Pos g σ Claims.Msg (Fin R)),
      (∀ z : CoinTape p.g' p.s', G (p.withTape z) ≤
        Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) (p.withTape z).toStop) →
      Pos.resample G p ≤ Pos.resample (Pos.stopCharge Γ) p := fun G p h =>
    calc Pos.resample G p
        ≤ avg (fun z : CoinTape p.g' p.s' =>
            Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) (p.withTape z).toStop) := avg_mono h
      _ = avg (fun z : CoinTape p.g' p.s' =>
            pathCharge Γ (p.withTape z).toStop.g' (p.withTape z).toStop.s' (p.withTape z).toStop.x) :=
          (Pos.avg_toStop (fun r => pathCharge Γ r.g' r.s' r.x) p).symm
      _ = Pos.resample (Pos.stopCharge Γ) p := rfl
  induction N with
  | zero =>
    intro p hp
    refine key _ p fun z => ?_
    show e (p.withTape z) ≤ _
    exact he _ (.withTape z hp)
  | succ N ih =>
    intro p hp
    refine key _ p fun z => ?_
    show stopErrK J γ e (roundsErrK D J γ e N) (p.withTape z).toStop (D.sub J (p.withTape z).toStop.modes) ≤ _
    cases h : D.sub J (p.withTape z).toStop.modes with
    | none =>
      have h0 := he _ (.toStop (.withTape z hp))
      rw [Pos.toStop_toStop] at h0
      exact h0
    | some s =>
      calc γ (p.withTape z).toStop J s + Pos.resample (roundsErrK D J γ e N) (p.withTape z).toStop.consume
          ≤ Γ (p.withTape z).toStop.g' + Pos.resample (Pos.stopCharge Γ) (p.withTape z).toStop.consume :=
            add_le_add (hγ _ (.toStop (.withTape z hp)) s h) (ih _ (.consume (.toStop (.withTape z hp))))
        _ = Pos.resample (fun r => Γ r.g' + Pos.stopCharge Γ r.consume) (p.withTape z).toStop :=
            (Pos.resample_add_consume Γ (Pos.stopCharge Γ) _).symm
        _ ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) (p.withTape z).toStop :=
            avg_mono fun _ => Pos.add_stopCharge_consume_le Γ hret _

/-- **A session's error, then what follows it, within the charge**: where `γL` at each stop the projection reaches
and `γ` at each submission decoded there are within `Γ` at the stop's node, and what follows the session is within
the charge from its position's next stop averaged over the tapes from that stop, so is the session's error from a
position the projection reaches. -/
theorem sessionErrK_le {g : Game Exchange} {σ : Strategy g} {R : ℕ} {Reg : Type} (D : PosDecoders g σ R Reg)
    (J : ℕ) (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞) (Γ : Game Exchange → ℝ≥0∞)
    (hret : ∀ a, Γ (.ret a) = 0) (Ks₀ : Fin R → DigestCut Claims.Msg g)
    (hγ : ∀ q, Pos.Reached Ks₀ q → ∀ s, D.sub J q.modes = some s → γ q J s ≤ Γ q.g')
    (γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (hγL : ∀ q, Pos.Reached Ks₀ q → γL q ≤ Γ q.g')
    {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞}
    (he : ∀ q, Pos.Reached Ks₀ q → e q ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) q.toStop) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) (hp : Pos.Reached Ks₀ p) :
    sessionErrK D J γ γL e N p ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) p.toStop :=
  calc sessionErrK D J γ γL e N p
      = γL p.toStop + Pos.resample (roundsErrK D J γ e N) p.toStop.consume := rfl
    _ ≤ Γ p.toStop.g' + Pos.resample (Pos.stopCharge Γ) p.toStop.consume :=
        add_le_add (hγL _ (.toStop hp)) (avg_roundsErrK_le D J γ Γ hret Ks₀ hγ he N _ (.consume (.toStop hp)))
    _ = Pos.resample (fun r => Γ r.g' + Pos.stopCharge Γ r.consume) p.toStop :=
        (Pos.resample_add_consume Γ (Pos.stopCharge Γ) _).symm
    _ ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) p.toStop :=
        avg_mono fun _ => Pos.add_stopCharge_consume_le Γ hret _

/-- **What follows a stop consumed with nothing written in, within the charge**: where what follows is within the
charge from its position's next stop averaged over the tapes from that stop, what follows past a stop the projection
reaches, over the tapes from the stop, is within the charge from the stop, which leaves the stop's own node out. -/
theorem resample_consume_le {g : Game Exchange} {σ : Strategy g} {R : ℕ} (Γ : Game Exchange → ℝ≥0∞)
    (hret : ∀ a, Γ (.ret a) = 0) (Ks₀ : Fin R → DigestCut Claims.Msg g) {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞}
    (he : ∀ q, Pos.Reached Ks₀ q → e q ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) q.toStop)
    (s : Pos g σ Claims.Msg (Fin R)) (hs : Pos.Reached Ks₀ s) :
    Pos.resample e s.consume ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) s :=
  calc Pos.resample e s.consume
      ≤ avg (fun z : CoinTape s.consume.g' s.consume.s' =>
          Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) (s.consume.withTape z).toStop) :=
        avg_mono fun z => he _ (.withTape z (.consume hs))
    _ = avg (fun z : CoinTape s.consume.g' s.consume.s' =>
          pathCharge Γ (s.consume.withTape z).toStop.g' (s.consume.withTape z).toStop.s'
            (s.consume.withTape z).toStop.x) :=
        (Pos.avg_toStop (fun r => pathCharge Γ r.g' r.s' r.x) s.consume).symm
    _ = Pos.resample (Pos.stopCharge Γ) s.consume := rfl
    _ ≤ Γ s.g' + Pos.resample (Pos.stopCharge Γ) s.consume := le_add_self
    _ = Pos.resample (fun r => Γ r.g' + Pos.stopCharge Γ r.consume) s :=
        (Pos.resample_add_consume Γ (Pos.stopCharge Γ) s).symm
    _ ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) s :=
        avg_mono fun _ => Pos.add_stopCharge_consume_le Γ hret _

/-- **A member's error, then what follows it, within the charge**: its session's (`sessionErrK_le`) where the sample
holds one of its proof blocks, with what follows taken past the session's closing round's stop
(`resample_consume_le`); else what follows, which is within the charge from the same position. -/
theorem memberErrK_le {g : Game Exchange} {σ : Strategy g} {R : ℕ} {Reg : Type} (D : PosDecoders g σ R Reg)
    (J : ℕ) (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞) (Γ : Game Exchange → ℝ≥0∞)
    (hret : ∀ a, Γ (.ret a) = 0) (Ks₀ : Fin R → DigestCut Claims.Msg g)
    (hγ : ∀ q, Pos.Reached Ks₀ q → ∀ s, D.sub J q.modes = some s → γ q J s ≤ Γ q.g')
    (γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (hγL : ∀ q, Pos.Reached Ks₀ q → γL q ≤ Γ q.g')
    {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞}
    (he : ∀ q, Pos.Reached Ks₀ q → e q ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) q.toStop) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) (hp : Pos.Reached Ks₀ p) :
    ∀ b : Bool, memberErrK D J γ γL e N p b ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) p.toStop
  | true => sessionErrK_le D J γ Γ hret Ks₀ hγ γL hγL (e := fun q => Pos.resample e q.toStop.consume)
      (fun q hq => resample_consume_le Γ hret Ks₀ he q.toStop (.toStop hq)) N p hp
  | false => he p hp

/-- **The sessions' error, in the members' order, within the charge**: where each member's `γL` at each stop the
projection reaches and its `γ` at each submission decoded there are within `Γ` at the stop's node, the members'
errors from a position the projection reaches are within the charge from the position's next stop, averaged over the
tapes from that stop. -/
theorem seqErr_le {g : Game Exchange} {σ : Strategy g} {Reg : Type} {R nJ : ℕ} (D : ℕ → PosDecoders g σ R Reg)
    (γ : ℕ → Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞)
    (γL : ℕ → (mPts : ℕ) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (S : Finset (Fin nJ)) (Rg : Reg)
    (Γ : Game Exchange → ℝ≥0∞) (hret : ∀ a, Γ (.ret a) = 0) (Ks₀ : Fin R → DigestCut Claims.Msg g)
    (hγ : ∀ i J q, Pos.Reached Ks₀ q → ∀ s, (D i).sub J q.modes = some s → γ i q J s ≤ Γ q.g')
    (hγL : ∀ i mPts q, Pos.Reached Ks₀ q → γL i mPts q ≤ Γ q.g') :
    ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ) (p : Pos g σ Claims.Msg (Fin R)), Pos.Reached Ks₀ p →
      seqErr D γ γL S Rg i l p ≤ Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) p.toStop
  | [], _, _, _ => zero_le
  | m :: l, i, p, hp =>
    memberErrK_le (D i) (memberTabs S Rg m) (γ i) Γ hret Ks₀ (hγ i _) (γL i (memberPts m)) (hγL i _)
      (e := fun q => seqErr D γ γL S Rg (i + 1) l q)
      (fun q hq => seqErr_le D γ γL S Rg Γ hret Ks₀ hγ hγL l (i + 1) q hq) m.N p hp (memberDrawn S m)

variable (e : 𝔽 ≃ (Fin 128 → Bool))

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-! ## The stops' term -/

/-- **The window's stops' term within the charge along a play** (`couplesErr_le_of_charge`): where each member's
`γL` at each stop the projection reaches and its `γ` at each submission decoded there are within `Γ` at the stop's
node, the play's end costs nothing, the start leaves the tape from the first stop fresh and is the uniform tape, the
stops' term over the window is within the charge along a play over a uniform tape. -/
theorem windowErr_le_of_charge (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {γ : ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞}
    {γL : ℕ → (mPts : ℕ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞} {b : ℝ≥0∞}
    (Γ : Game Exchange → ℝ≥0∞) (hret : ∀ a, Γ (.ret a) = 0) (hF : WindowFirstStopFresh V θ F c W rd σ M)
    (hS : ∀ H : CoinTape (V.game θ F c) σ → ℝ≥0∞,
      avg (fun ω : (W.extend M.widening).L.Ω => H (M.start ω)) = avg H)
    (hγ : ∀ i J q, Pos.Reached (roundCuts V θ F c rd) q → ∀ s, (D i).sub J q.modes = some s → γ i q J s ≤ Γ q.g')
    (hγL : ∀ i mPts q, Pos.Reached (roundCuts V θ F c rd) q → γL i mPts q ≤ Γ q.g')
    (hb : avg (pathCharge Γ (V.game θ F c) σ) ≤ b) :
    windowErr V θ F c W t rd σ D M γ γL ≤ b := by
  unfold windowErr
  calc avg (fun ω : (W.extend M.widening).L.Ω =>
        seqErr D γ γL ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0
          ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω)))
      ≤ avg (fun ω : (W.extend M.widening).L.Ω =>
          Pos.resample (fun r => pathCharge Γ r.g' r.s' r.x) (windowFirstStop V θ F c W rd σ M ω)) :=
        avg_mono fun ω => seqErr_le D γ γL _ _ Γ hret (roundCuts V θ F c rd) hγ hγL _ 0 _ (.root (M.start ω))
    _ = avg (fun ω : (W.extend M.widening).L.Ω =>
          pathCharge Γ (windowFirstStop V θ F c W rd σ M ω).g' (windowFirstStop V θ F c W rd σ M ω).s'
            (windowFirstStop V θ F c W rd σ M ω).x) :=
        (hF fun _ r => pathCharge Γ r.g' r.s' r.x).symm
    _ ≤ avg (fun ω : (W.extend M.widening).L.Ω => pathCharge Γ (V.game θ F c) σ (M.start ω)) :=
        avg_mono fun ω => Pos.pathCharge_walk_le Γ (V.game θ F c) σ (roundCuts V θ F c rd) (M.start ω) id
    _ = avg (pathCharge Γ (V.game θ F c) σ) := hS _
    _ ≤ b := hb

/-- **From no tree open, the potential is the window's hiding term** (`treePotential_nil`). -/
theorem Window.treePotential_nil (W : Window) : treePotential e [] = W.hidingTerm := by
  have h : ENNReal.ofReal (1 / 2 ^ 257) = ((2 : ℝ≥0∞) ^ 257)⁻¹ := by
    rw [one_div, ENNReal.ofReal_inv_of_pos (by positivity), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_ofNat]
  rw [treePotential, treesErr, leafTerm, h, Window.hidingTerm, maxLeaves]
  simp only [List.map_nil, List.sum_nil, zero_add, treeLines, Nat.sub_zero, Nat.cast_mul, div_eq_mul_inv]
  ring

/-- **The sample-coupled start over the window averages any reading of the tape to its uniform average**, wherever
the sample's coins are uniform (`avg_sampleMap_start`). -/
theorem avg_windowSampleMap_start (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (label : String) (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H)
    (H : CoinTape (V.game θ F c) σ → ℝ≥0∞) :
    avg (fun ω : (W.extend (windowSampleMap e V θ F c W rd σ label after κ).widening).L.Ω =>
      H ((windowSampleMap e V θ F c W rd σ label after κ).start ω)) = avg H := by
  refine Eq.symm (Eq.trans ?_ (avg_prod (C := W.Ω) (D := CoinTape (V.game θ F c) σ) _).symm)
  exact (avg_fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m H).symm.trans
    (hκ fun κ' => avg fun t =>
      H (fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m κ' t)).symm

/-- **The sample-coupled coin map's premise over the window, off the empty classes at their counts**
(`sampleMap_couples_at`): the joint law's tape off `linkEmptyAt` and `roundEmptyAt` at every member's stops is within
the openings' gaps and empty classes at the stops' counts over the stops of the uniform tape, wherever the sample's
coins are uniform. -/
theorem windowSampleMap_couples_at (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (label : String) (after : ByteArray) {m : ℕ}
    (κ : W.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    WindowCouplesWithin V θ F c W t rd σ D (windowSampleMap e V θ F c W rd σ label after κ)
      (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
      (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1))
      (windowErr V θ F c W t rd σ D (windowSampleMap e V θ F c W rd σ label after κ)
        (fun _ q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
          openEmptyAt (finish (V.signer θ) θ F c σ.1) q.g' s.n (roundBits e : (Fin s.n → 𝔽) → List Bool))
        (fun _ mPts q =>
          openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin mPts → 𝔽) → List Bool) +
            openEmptyAt (finish (V.signer θ) θ F c σ.1) q.g' (mPts + mPts)
              (linkBits e : (Fin 2 → Fin mPts → 𝔽) → List Bool))) := by
  have h := windowCouples_of_close V θ F c W t rd σ D _
    (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
    (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1))
    (windowOpenMap_roundClose e (finish (V.signer θ) θ F c σ.1) (CoinTape (V.game θ F c) σ)
      (sampleStart V θ F c rd σ label after κ))
    (windowOpenMap_linkClose e (finish (V.signer θ) θ F c σ.1) (CoinTape (V.game θ F c) σ)
      (sampleStart V θ F c rd σ label after κ))
    (windowSampleMap_fresh e V θ F c W rd σ label after κ) (windowSampleMap_start e V θ F c W rd σ label after κ hκ)
  rwa [zero_add] at h

/-- **The window's stops' term at the sample-coupled coin map, with each stop's empty class at the count of coins the
stop releases, is within the hiding term** at every strategy (`couplesErr_sampleMap_le_at`): each stop's gap and empty
class at its count are within its node's class error at any count the frame asks for, and the class errors along a
play are within the potential from no tree open, however many members' sessions the play runs. -/
theorem windowErr_sampleMap_le_at (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (label : String) (after : ByteArray) {m : ℕ}
    (κ : W.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    windowErr V θ F c W t rd σ D (windowSampleMap e V θ F c W rd σ label after κ)
      (fun _ q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
        openEmptyAt (finish (V.signer θ) θ F c σ.1) q.g' s.n (roundBits e : (Fin s.n → 𝔽) → List Bool))
      (fun _ mPts q =>
        openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin mPts → 𝔽) → List Bool) +
          openEmptyAt (finish (V.signer θ) θ F c σ.1) q.g' (mPts + mPts)
            (linkBits e : (Fin 2 → Fin mPts → 𝔽) → List Bool)) ≤
      W.hidingTerm :=
  windowErr_le_of_charge V θ F c W t rd σ D _ (openErr e (finish (V.signer θ) θ F c σ.1))
    (fun _ => openErr_none e openingOf_ret) (windowSampleMap_fresh e V θ F c W rd σ label after κ)
    (avg_windowSampleMap_start e V θ F c W rd σ label after κ hκ)
    (fun _ _ _ _ s _ => openGap_add_openEmptyAt_le_openErr e s.n)
    (fun _ mPts _ _ => openGap_add_openEmptyAt_le_openErr_link e mPts)
    ((avg_pathCharge_game_le e V θ F c σ).trans_eq (W.treePotential_nil e))

variable (k : Verity.Registration.Composite)

omit θ F c in
/-- **The coupling at the window's projection with the sample-coupled coin maps, off the empty classes at their
counts** (`couplingTarget_withProject_of_sample_at`): `windowCouplingTarget_of_logged` at `windowSampleMaps` and the
cut `P` and `Q`, off `linkEmptyAt` and `roundEmptyAt` at every member's stops, whose coin map's premise holds within
the stops' term (`windowSampleMap_couples_at`), within the hiding term at every strategy
(`windowErr_sampleMap_le_at`), wherever the window's coins are uniform. -/
theorem windowCouplingTarget_of_sample_at (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {m : ℕ}
    (κ : (W : Window) → W.Ω → Fin m → Fin 512 → Bool) (W : Window) (P : Exchange → Prop) (θ : ByteArray)
    (F : Array ByteArray) (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ W ω)) = avg H)
    (ha : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hD : ∀ c σ, WindowDecodes V θ F c k (fun o => P o ∧ Q c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
      (windowSampleMaps e V rd label after κ W θ F c σ) (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
      (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1))) :
    WindowCouplingTarget V k (K.withProject V rd D (windowSampleMaps e V rd label after κ)) W θ F :=
  windowCouplingTarget_of_logged V k rd K D (windowSampleMaps e V rd label after κ) W P θ F Q
    (fun c σ _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
    (fun c σ _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) ha hq
    (fun c σ => windowCouples_of_within V θ F c (windowSampleMap_couples_at e V θ F c W (V.cost θ F c σ)
      (rd W θ F c) σ (D W θ F c σ) (label θ F c σ) (after θ F c σ) (κ W) hκ)
      (windowErr_sampleMap_le_at e V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (label θ F c σ)
        (after θ F c σ) (κ W) hκ)) hD

end Verity.Sampled
