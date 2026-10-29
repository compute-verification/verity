import Proofs.Flock.Claims.KS.PlainFirst

/-!
# The third bound of the plain mixture, from a simulation of the runs

`PlainLiveWrPAt`'s third clause bounds the average over a tape of the live session's probability of a self-clash
(`LiveSelfClash`) by the probability of a one-run event of `V false`'s coins. With the deployed coins as the tape and
`plainCoins` as the coins, `plainMix_le_of_run` gives that form for any live event `pL` and deployed event `E` from the
pointwise implication alone: wherever `pL` holds at the live outcome of `P'` at the tape `(ω, u)` and the live coins
`r'`, `E` holds at `plainCoins (u, r')` (`liveMix_le_Pr`, `Pr_mergeAt`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **A deployed event bounds a mixture of live strategies at `plainCoins`**: where the live event `pL` at the live
outcome of `P'` at the tape `(ω, u)` and the live coins `r'` gives `E` at `plainCoins (u, r')`, the average over the
deployed coins `u` of the live session's probability of `pL` against `P'`'s strategies is at most `Pr[E]`. -/
theorem plainMix_le_of_run {Ω : Type} (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts)
    (P' : Claims.Prover (Ω × (Fin ((V false).k x) → Coin))) (hk : N x + 2 = (V false).k x)
    (n : Fin ((V false).k x) → ℕ) (pL : LiveOutJ 1 mPts → Prop) (E : (Fin ((V false).k x) → Coin) → Prop)
    (h : ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
      E (plainCoins hk n (u, r'))) :
    (∑ u, (prob pL (liveSession dg 1 mPts (N x))
      (liveStrat (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card (Fin ((V false).k x) → Coin) ≤ Claims.Pr E :=
  liveMix_le_Pr (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x ω pL E (plainCoins hk n)
    (@Pr_mergeAt _ _ _ 𝔽 _ _ hk n E) h

end FlockDeploy.KS
