import Proofs.Flock.C3.RegLevels.SrcMaps
import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Soundness.Assumptions.Inner

/-!
# n14: R's Ligerito stage's coins, as the verifier holds R's stream

`lgSteps_of_lig` makes R's Ligerito's coin steps, at `MapsLeft`'s weave, `stacked/rep0`'s rounds of coins (`rep0`)
from the ring switches' `k₂` on. `rep0` reads the verifier's hold of that stream (`C3.coins`, `commits.stream?`'s
coins), so they are `StageHeld`'s first two parts at `stacked/rep0`, which `laterCoins_of_held`'s `h5` reads:

* `commits_c3Of`, `heldCoins_rep0`: at the `C3` the verifier builds of its hold `cs` (`c3Of`), the coins it holds of
  `stacked/rep0` at round `q` are `rep0`'s round `q`;
* `heldStage_of_lgCoins`: a stage whose coin steps are `rep0`'s rounds from `k₂` on has them as the coins the verifier
  holds of `stacked/rep0` at the rounds from `k₂` to `rep0`'s round count (`StageHeld`'s first two parts at `ix = id`).

Left for `h5`: `StageHeld`'s last two parts, that R's record's round count is `rep0`'s and the history's rounds from
the stage's start draw `stacked/rep0`'s rounds in order.
-/

namespace FlockC3.RegLevels

open FlockSoundness.Assumptions (heldCoins)
open FlockSoundness.Discharge.PlaysCoin (coinArr)

/-- **The `C3` the verifier builds of its hold holds its commits** (`c3Of`). -/
theorem commits_c3Of {ish : Flock.RecVStar.Inner} {cs : Flock.RecVStar.InnerCommits} {c3 : Flock.RecCoins.C3}
    (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3) : c3.commits = cs := by
  unfold Flock.Guarantees.c3Of at hc3
  cases hsp : ish.sparse with
  | none => rw [hsp] at hc3; cases hc3
  | some sp => rw [hsp] at hc3; cases hc3; rfl

/-- **`rep0` is the verifier's hold of `stacked/rep0`**: at the `C3` it builds of its hold `cs`, the coins it holds of
the stream at round `q` are `rep0`'s round `q` (none where it holds no such stream, and `rep0` is empty). -/
theorem heldCoins_rep0 {ish : Flock.RecVStar.Inner} {cs : Flock.RecVStar.InnerCommits} {c3 : Flock.RecCoins.C3}
    (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3) (unit : String) (q : ℕ) :
    heldCoins (.ok cs) s!"{Flock.RecVStar.STACKED}/rep0" q = (FlockC3.Opening.rep0 c3 unit)[q]? := by
  have hc := commits_c3Of hc3
  cases hst : cs.stream? s!"{Flock.RecVStar.STACKED}/rep0" with
  | none =>
    have hr : FlockC3.Opening.rep0 c3 unit = #[] := by
      unfold FlockC3.Opening.rep0 Flock.RecCoins.C3.coins
      rw [hc, hst]
      rfl
    rw [hr]
    show (cs.stream? s!"{Flock.RecVStar.STACKED}/rep0").bind (·.coins[q]?) = _
    rw [hst]
    rfl
  | some st =>
    rw [FlockC3.Opening.rep0_eq (c := st.coins) (by unfold Flock.RecCoins.C3.coins; rw [hc, hst]; rfl)]
    show (cs.stream? s!"{Flock.RecVStar.STACKED}/rep0").bind (·.coins[q]?) = _
    rw [hst]
    rfl

/-- **R's Ligerito stage's coins are the verifier's hold of `stacked/rep0`** (`StageHeld`'s first two parts at
`ix = id`): a stage whose coin steps are `rep0`'s rounds from `k₂` on (`lgSteps_of_lig`'s, at `MapsLeft`'s weave) has
them as the coins `g q := rep0[q]!`, `q` over the rounds from `k₂` to `rep0`'s round count, each the coins the verifier
holds of the stream at round `q` (`heldCoins_rep0`). -/
theorem heldStage_of_lgCoins {ish : Flock.RecVStar.Inner} {cs : Flock.RecVStar.InnerCommits}
    {c3 : Flock.RecCoins.C3} (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3) {unit : String}
    {ts : List FlockSoundness.Game.Step} {k₂ : ℕ}
    (hts : ts.filterMap Sum.getRight? =
      ((List.range' k₂ ((FlockC3.Opening.rep0 c3 unit).size - k₂)).map fun i =>
        (FlockC3.Opening.rep0 c3 unit)[i]!).map (coinArr FlockSoundness.Refine.fOf)) :
    ts.filterMap Sum.getRight? = (List.range' k₂ ((FlockC3.Opening.rep0 c3 unit).size - k₂)).map
        (coinArr FlockSoundness.Refine.fOf ∘ fun q => (FlockC3.Opening.rep0 c3 unit)[q]!) ∧
      ∀ q c, heldCoins (.ok cs) s!"{Flock.RecVStar.STACKED}/rep0" q = some c →
        (FlockC3.Opening.rep0 c3 unit)[q]! = c := by
  refine ⟨by rw [hts, List.map_map], fun q c hq => ?_⟩
  rw [heldCoins_rep0 hc3 unit q] at hq
  rw [getElem!_def, hq]

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.heldStage_of_lgCoins
