module

public import Proofs.Properties.Verity.Sampled.Project
public import Proofs.Properties.Verity.Sampled.Members.Landing

/-!
# The session's projection, leaving the position it ends at

The session's projection (`sessionP`, in `Proofs.Properties.Verity.Sampled.Project`) walks a play of the period's game
from a position to the live session's end and keeps nothing of where it ends. A window's sessions are played one after
another on one play of the period's game, so each starts from the position the one before ends at:

* `stepL`, `roundsL`, `sessionL`: the round's, the rounds' and the session's projection as landings
  (`Verity.Game.Landing`), leaving the position the play is at when the session ends: after the last round's coins, or
  at the stop whose decoding submits nothing. They play the projections (`strat_roundsL`, `strat_sessionL`).
* `lands_roundsL`, `lands_sessionL`: where they end. The outcome and the position's tape are those a trace of the rounds
  ends on (`RoundsReach`, off no event and at the live session's own outcome), so what the one-session leg proves of a
  trace holds of each session; the position is a later one than the start and keeps the invariant the start does.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Refine (roundOf)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : Decoders R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ)

/-- **One round's landing** at the stop `q`, at the submission decoded there (`stepP`): the play's end, leaving the
stop, or the submission and then `next` on its coins. -/
noncomputable def stepL (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R))
    (next : (s : SubmitJ J) → (c : Fin s.n → 𝔽) → Landing (Pos g σ Claims.Msg (Fin R))
      ((liveRoundsJ dg J N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))).bind (KLJ rootB pts))) :
    Option (SubmitJ J) → Landing (Pos g σ Claims.Msg (Fin R)) ((liveRoundsJ dg J (N + 1) rs).bind (KLJ rootB pts))
  | none => ⟨none, ⟨D.proofs (CoinTape.run g σ q.tape), q⟩⟩
  | some s => ⟨some s, next s⟩

/-- **The rounds' landing** from the position `p` (`roundsP`), leaving the position the rounds end at. -/
noncomputable def roundsL (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) →
      Landing (Pos g σ Claims.Msg (Fin R)) ((liveRoundsJ dg J N rs).bind (KLJ rootB pts))
  | 0, _, p => ⟨D.proofs (CoinTape.run g σ p.tape), p⟩
  | N + 1, rs, p => stepL D dg J rootB pts N rs p.toStop
    (fun s c => roundsL rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
      (p.toStop.rewrite (M.round p.toStop J s c)).consume) (D.sub J p.toStop.modes)

/-- **The session's landing** from the position `p` (`sessionP`), leaving the position the session ends at. -/
noncomputable def sessionL (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    Landing (Pos g σ Claims.Msg (Fin R)) (liveSession dg J mPts N) :=
  ⟨D.root p.toStop.modes, fun pts => roundsL D M dg J (D.root p.toStop.modes) pts N (fun _ _ => [])
    (p.toStop.rewrite (M.link p.toStop pts)).consume⟩

/-- The rounds' landing plays the rounds' projection. -/
theorem strat_roundsL (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)),
      Landing.strat _ (roundsL D M dg J rootB pts N rs p) = roundsP D M dg J rootB pts N rs p
  | 0, _, _ => rfl
  | N + 1, rs, p => by
    show Landing.strat _ (stepL D dg J rootB pts N rs p.toStop _ (D.sub J p.toStop.modes)) =
      stepP D dg J rootB pts N rs p.toStop _ (D.sub J p.toStop.modes)
    cases D.sub J p.toStop.modes with
    | none => rfl
    | some s => exact congrArg (Sigma.mk (some s)) (funext fun c => strat_roundsL rootB pts N _ _)

/-- **The session's landing plays the session's projection.** -/
theorem strat_sessionL (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    Landing.strat _ (sessionL D M dg J N p) = sessionP D M dg J N p :=
  congrArg (Sigma.mk (D.root p.toStop.modes)) (funext fun pts => strat_roundsL D M dg J _ pts N _ _)

/-- **Where the rounds' landing ends**: on an outcome and a position whose tape a trace of the rounds from `p` ends on
(`RoundsReach`, off no event, at the live session's own outcome); the position is a later one than `p` and keeps the
invariant `p` does. -/
theorem lands_roundsL (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (o : LiveOutJ J mPts)
      (q : Pos g σ Claims.Msg (Fin R)), Landing.Lands _ (roundsL D M dg J rootB pts N rs p) o q →
        RoundsReach D M dg J (fun _ _ _ _ => False) id rootB pts N rs p (o, q.tape) ∧
          ∀ Ks₀ : Fin R → DigestCut Claims.Msg g, p.Inv Ks₀ → p.Le q ∧ q.Inv Ks₀
  | 0, rs, p, o, q, h => by
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨.zero _ _, fun _ hp => ⟨Pos.Le.refl _, hp⟩⟩
  | N + 1, rs, p, o, q, h => by
    change Landing.Lands _ (stepL D dg J rootB pts N rs p.toStop _ (D.sub J p.toStop.modes)) o q at h
    cases hs : D.sub J p.toStop.modes with
    | none =>
      rw [hs] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨.stop N rs p hs, fun _ hp => ⟨(Pos.toStop_spec hp).1, (Pos.toStop_spec hp).2.2⟩⟩
    | some s =>
      rw [hs] at h
      obtain ⟨c, h⟩ := h
      obtain ⟨hr, hi⟩ := lands_roundsL rootB pts N _ _ o q h
      refine ⟨.round N rs p s c _ hs (fun h => h) hr, fun Ks₀ hp => ?_⟩
      obtain ⟨hl, _, hi'⟩ := Pos.toStop_spec hp
      obtain ⟨hl', hq⟩ := hi Ks₀ (Pos.inv_consume (Pos.inv_rewrite hi' _))
      exact ⟨(hl.trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans hl', hq⟩

/-- **Where the session's landing ends**: at link points, on an outcome and a position whose tape a trace of the rounds
ends on, from the first stop with the link points written in, at the roots decoded there; the position is a later one
than `p` and keeps the invariant `p` does. -/
theorem lands_sessionL (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) (o : LiveOutJ J mPts) (q : Pos g σ Claims.Msg (Fin R))
    (h : Landing.Lands _ (sessionL D M dg J N p) o q) :
    ∃ pts : Fin 2 → Fin mPts → 𝔽,
      RoundsReach D M dg J (fun _ _ _ _ => False) id (D.root p.toStop.modes) pts N (fun _ _ => [])
          (p.toStop.rewrite (M.link p.toStop pts)).consume (o, q.tape) ∧
        ∀ Ks₀ : Fin R → DigestCut Claims.Msg g, p.Inv Ks₀ → p.Le q ∧ q.Inv Ks₀ := by
  obtain ⟨pts, h⟩ := h
  obtain ⟨hr, hi⟩ := lands_roundsL D M dg J _ pts N _ _ o q h
  refine ⟨pts, hr, fun Ks₀ hp => ?_⟩
  obtain ⟨hl, _, hi'⟩ := Pos.toStop_spec hp
  obtain ⟨hl', hq⟩ := hi Ks₀ (Pos.inv_consume (Pos.inv_rewrite hi' _))
  exact ⟨(hl.trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans hl', hq⟩

end Verity.Sampled
