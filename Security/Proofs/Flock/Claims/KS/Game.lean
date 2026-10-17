import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Claims.Bridge

/-!
# Escapes at a game's verifier

At the public-coin verifier of a game (`Bridge.ofGame`), the verifier's `H` log is the decision's calls on the game's
outcome (`Bridge.run_ofGame`), so an escape event is tied to the verifier's log (`Escapes.tied`, `Escapes.tied₁`) as
soon as it gives a collision among the decision's calls: on the outcomes of two runs forked at `j`
(`tiedColl_ofGame`), or of one run (`tiedColl_ofGame₁`).
-/

namespace Claims

open FlockSoundness FlockSoundness.Game Bridge

theorem tiedColl_ofGame {β Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type} (P : Prover Ω)
    (k : Msg → ℕ) (g : Msg → Game β) (I : ∀ x, Iface Coin (g x)) (d : Msg → β → OC Bool) (x : Msg) (ω : Ω)
    (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x)) (j : Fin (k x))
    (r r' : Fin (k x) → Coin)
    (h : TiedColl H (((d x (outcome H P k g I x ω r)).eval H).2 ++
      ((d x (outcome H P k g I x ω (forkCoins j r r'))).eval H).2)) :
    TiedColl H ((run H P (ofGame k g I d) x ω r).vLog ++
      (run H P (ofGame k g I d) x ω (forkCoins j r r')).vLog) := by
  have h2 : (run H P (ofGame k g I d) x ω (forkCoins j r r')).vLog =
      ((d x (outcome H P k g I x ω (forkCoins j r r'))).eval H).2 :=
    (run_ofGame H P k g I d x ω _ hR).2.1
  simp only [(run_ofGame H P k g I d x ω r hR).2.1, h2]
  exact h

theorem tiedColl_ofGame₁ {β Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type} (P : Prover Ω)
    (k : Msg → ℕ) (g : Msg → Game β) (I : ∀ x, Iface Coin (g x)) (d : Msg → β → OC Bool) (x : Msg) (ω : Ω)
    (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x)) (r : Fin (k x) → Coin)
    (h : TiedColl H ((d x (outcome H P k g I x ω r)).eval H).2) :
    TiedColl H (run H P (ofGame k g I d) x ω r).vLog := by
  rw [(run_ofGame H P k g I d x ω r hR).2.1]
  exact h

end Claims
