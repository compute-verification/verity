module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.Claims.Recursion.LastBindsR

/-!
# The Flock instance's last-coin hypothesis (F8)

`Proto.LastCoinFree` at the element-level Flock (`Proto.ofFlock`) is `DE`'s `Bound.LastStepFree`: one statement
(`lastCoinFree_ofFlock`). So it holds at every `H` at each bound record whose `DE` steps as the deployed reader
(`RecDeploy.lastCoinFree_ofFlock_R`, from `RecDeploy.lastStepFree_of_R`) or the deployed plain verifier
(`RecDeploy.lastCoinFree_ofFlock_V`, from `RecDeploy.lastStepFree_of_V`): L1's last-coin hypothesis holds at the
deployment with no premise.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **F8 is `Bound.LastStepFree`**: the element-level Flock's last step reads no last coin wherever `DE`'s does. -/
theorem lastCoinFree_ofFlock {Coin Col T : Type} {H : Msg → Msg} {Bd : Bound Coin Col T} (h : Bd.LastStepFree H)
    (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) : (Proto.ofFlock Bd decI Cp).LastCoinFree H :=
  h

end Claims.RecGeneric

namespace RecDeploy

open Claims (Msg)
open Claims.Recursion Claims.RecGeneric
open FlockDeploy (Coin FinalP)

/-- **F8 at the deployed reader**: at every bound record whose `DE` steps as `FlockDeploy.VR rd`, at every `H`. -/
theorem lastCoinFree_ofFlock_R {Col T : Type} [DecidableEq Col] (H : Msg → Msg) (rd : Msg → Option FinalP)
    (Bd : Bound Coin Col T) (hV : Bd.DE.V = FlockDeploy.VR rd) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) :
    (Proto.ofFlock Bd decI Cp).LastCoinFree H :=
  lastCoinFree_ofFlock (lastStepFree_of_R H rd Bd hV) decI Cp

/-- **F8 at the deployed plain verifier**: at every bound record whose `DE` steps as `FlockDeploy.V false`, at every
`H`. -/
theorem lastCoinFree_ofFlock_V {Col T : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (hV : Bd.DE.V = FlockDeploy.V false) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) :
    (Proto.ofFlock Bd decI Cp).LastCoinFree H :=
  lastCoinFree_ofFlock (lastStepFree_of_V H Bd hV) decI Cp

end RecDeploy
