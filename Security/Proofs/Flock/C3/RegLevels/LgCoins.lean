import Proofs.Flock.C3.RegLevels.SrcMaps
import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Soundness.Assumptions.Inner

/-!
# n14: `MapsLeft`'s Ligerito coins, from the coins the verifier holds of R's stream

`MapsLeft`'s fourth conjunct asks R's Ligerito's coin steps to be `stacked/rep0`'s rounds of coins (`rep0`) from the
ring switches' `k₂` on. `rep0` reads the verifier's hold of that stream (`C3.coins`, `commits.stream?`'s coins), so of
`StageHeld`'s four parts at `stacked/rep0` the held one is by definition:

* `commits_c3Of`, `heldCoins_rep0`: at the `C3` the verifier builds of its hold `cs` (`c3Of`), the coins it holds of
  `stacked/rep0` at round `q` are `rep0`'s round `q`;
* `lgCoins_of_held`: a stage whose coin steps are the coins `g q`, `q` over the rounds from `k₂` to `rep0`'s round
  count, each the coins the verifier holds of `stacked/rep0` at round `q` (`StageHeld`'s first two parts at `ix = id`),
  has `MapsLeft`'s coin steps.

Left for R's Ligerito: its weave (its coin rounds are those from `k₂` to R's record's round count, which is `rep0`'s),
and `StageHeld`'s draws, which `laterCoins_of_held`'s `h5` reads and `MapsLeft` does not.
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

/-- **`MapsLeft`'s Ligerito coins from the verifier's hold of `stacked/rep0`**: where a stage's coin steps are the coins
`g q`, `q` over the rounds from `k₂` to `rep0`'s round count (`hts`), each the coins the verifier holds of the stream
at round `q` (`hd`), they are `rep0`'s rounds from `k₂` on (`heldCoins_rep0`). -/
theorem lgCoins_of_held {ish : Flock.RecVStar.Inner} {cs : Flock.RecVStar.InnerCommits} {c3 : Flock.RecCoins.C3}
    (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3) {unit : String} {ts : List FlockSoundness.Game.Step}
    {g : ℕ → Array Flock.F128} {k₂ : ℕ}
    (hts : ts.filterMap Sum.getRight? = (List.range' k₂ ((FlockC3.Opening.rep0 c3 unit).size - k₂)).map
      (coinArr FlockSoundness.Refine.fOf ∘ g))
    (hd : ∀ q c, heldCoins (.ok cs) s!"{Flock.RecVStar.STACKED}/rep0" q = some c → g q = c) :
    ts.filterMap Sum.getRight? =
      ((List.range' k₂ ((FlockC3.Opening.rep0 c3 unit).size - k₂)).map fun i =>
        (FlockC3.Opening.rep0 c3 unit)[i]!).map (coinArr FlockSoundness.Refine.fOf) := by
  rw [hts, List.map_map]
  refine List.map_congr_left fun q hq => ?_
  have hq' : q < (FlockC3.Opening.rep0 c3 unit).size := by
    have := List.mem_range'_1.1 hq
    omega
  show coinArr FlockSoundness.Refine.fOf (g q) = coinArr FlockSoundness.Refine.fOf ((FlockC3.Opening.rep0 c3 unit)[q]!)
  rw [hd q _ ((heldCoins_rep0 hc3 unit q).trans (Array.getElem?_eq_getElem hq')),
    getElem!_pos (FlockC3.Opening.rep0 c3 unit) q hq']

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.heldCoins_rep0
#print axioms FlockC3.RegLevels.lgCoins_of_held
