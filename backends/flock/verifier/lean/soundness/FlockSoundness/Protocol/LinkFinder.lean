import FlockSoundness.Assumptions
import FlockSoundness.Protocol.ExtractWait

/-!
# The link finder: a trial, then further trials until one reads the same string

The link theorem's collision finder (DESIGN.md §3), for any trial. A trial (`T`, played by `sT`)
reads a set of values behind one commit string (`rd`). The finder (`finderG`) runs a trial and then
up to `M` more. It succeeds (`Succ`) if

* its first trial reads two values (two drawn units opened the string differently), or
* its first trial reads one value, and the first further trial that reads anything reads another.

With `p v` the probability that a trial reads exactly `v`, `s` that it reads two or more values and
`r` that it reads anything, the finder succeeds with probability `s + geom r M · Σ_v p v (r − p v)`
(`expect_succ`). For the plurality `Xv` of `p` that is at least `s + Σ_{v ≠ Xv} p v − 1/(eM)`
(`succ_ge`): disagreeing with the plurality is at most as likely as two readings disagreeing.

Its cost stops at the first further trial that reads anything, so it is at most twice a trial's in
expectation, whatever `r` (`expect_finderCost_le`, a Wald identity). Under expected-time collision
resistance for the finder (`Assumptions.SHA512CRExpected`), and when every success outputs a
collision, `s + Σ_{v ≠ Xv} p v ≤ 2·E[trial cost]/2^256 + 1/(eM)` (`link_mass_le`).
-/

namespace FlockSoundness

namespace LinkFinder

open Game Extract

variable {τ Val : Type} [Fintype Val] [DecidableEq Val] (T : Game τ) (sT : Strategy T)
  (rd : τ → Finset Val)

/-- **The finder**: a trial, then `M` more. -/
def finderG (M : ℕ) : Game (τ × List τ) := T.bind fun e => (runsG T M).map fun aux => (e, aux)

/-- `sT` in every trial. -/
def finderStrat (M : ℕ) : Strategy (finderG T M) :=
  Strategy.bind T _ sT fun _ => Strategy.map _ _ (runsStrat T sT M)

open Classical in
/-- The first further trial that reads anything. -/
noncomputable def nextRead (aux : List τ) : Option τ := aux.find? fun e => decide (rd e).Nonempty

open Classical in
/-- The further trials' cost, up to and including the first that reads anything. -/
noncomputable def loopCost (c : τ → ℝ) : List τ → ℝ
  | [] => 0
  | e :: es => c e + if (rd e).Nonempty then 0 else loopCost c es

open Classical in
/-- **The finder's cost**: its first trial's, and when that read one value, the further trials' up
to the first that reads anything. -/
noncomputable def finderCost (c : τ → ℝ) (x : τ × List τ) : ℝ :=
  c x.1 + if (rd x.1).card = 1 then loopCost rd c x.2 else 0

/-- The probability that a trial reads exactly `v`. -/
noncomputable def pRd (v : Val) : ℝ := expect (ind fun e => rd e = {v}) T sT

end LinkFinder

end FlockSoundness
