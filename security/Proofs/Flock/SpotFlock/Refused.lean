module

public import Proofs.Flock.SpotFlock.Coins
public import Proofs.Flock.Bridge.QsRows.FirstReads
public import Proofs.Flock.Claims.Sampling.Select.Second

/-!
# SPOT[Flock]'s second stage refuses an honest run with probability at most `k_d·2⁻⁶⁴`

The second stage (`Spot.Select.selectSecond`) refuses where the replay sample and the cells are not a second stage's
(`secondProblem`), where its law does not parse (`Spot.Draw.flockLaw`), where a cell's sample does not end on the
stage's bytes (`Spot.Draw.drawUnits`), or where the sample object it builds is not well formed under its law
(`Spot.Draw.checkDraw`). At a replay sample and cells `secondProblem` passes, the second and fourth never happen, and
the third happens with probability at most `k_d·2⁻⁶⁴` on a uniform stream that holds every cell's first read, `k_d` the
`|sample|·Σ_t k_t` proof blocks the second stage samples.

* `secondRead q ns ks`: the bytes the cells of a replay sample of `q` read first, in all.
* `lawOn_second`: the second stage's law is a sampling design on the cells that samples `k_d` proof blocks (`LawOn`).
* `flockLaw_second_ok`, `selectSecond_of_drawUnits`: the law parses, and a sample that ends is accepted.
* `selectSecond_refused_le`: on `len` uniform bytes, `secondRead ≤ len`, the second stage is refused with probability
  at most `k_d·2⁻⁶⁴`.
* `secondDraw_refused_le`: the same of the code's second stage (`secondDraw`) on its `stage` uniform coins, at every
  first-stage coin whose replay sample the second stage takes.
-/

@[expose] public section

namespace SpotFlock.Refused

open Claims Claims.Sampling SamplingDeploy Properties.Spot Finset FlockSoundness FlockSoundness.Game
open scoped ENNReal

/-- **The cells' sizes** over a replay sample of `q`, in the second stage's order: template by template, `q` cells of
`n_t` proof blocks. -/
def cellSizes (q : ℕ) (ns : Array ℕ) : List ℕ := (List.range ns.size).flatMap fun t => List.replicate q (ns.getD t 0)

/-- **The bytes the second stage's cells read first, in all**, over a replay sample of `q`: each cell of template `t`
reads `2·k_t·width n_t + 64`. -/
def secondRead (q : ℕ) (ns ks : Array ℕ) : ℕ :=
  q * ((List.range ns.size).map fun t => 2 * ks.getD t 0 * Flock.Draw.width (ns.getD t 0) + 64).sum

/-- **The second stage's law is a sampling design on the cells** that samples `|sample|·Σ_t k_t` proof blocks. -/
theorem lawOn_second {sample ns ks : Array ℕ} (hp : Spot.Select.secondProblem sample ns ks = none) {m : ℕ}
    {parsed : Flock.Draw.Law} (h : Spot.Draw.flockLaw (Spot.Select.secondLaw sample ns ks) m = .ok parsed) :
    Flock.Bridge.LawOn parsed (cellSizes sample.size ns) (sample.size * ks.toList.sum) := sorry

/-- The second stage's stages read `secondRead` bytes first. -/
theorem firstRead_second {sample ns ks : Array ℕ} (hp : Spot.Select.secondProblem sample ns ks = none) {m : ℕ}
    {parsed : Flock.Draw.Law} (h : Spot.Draw.flockLaw (Spot.Select.secondLaw sample ns ks) m = .ok parsed) :
    Flock.Bridge.firstRead parsed m = secondRead sample.size ns ks := sorry

/-- **The second stage's law parses** at a replay sample and cells `secondProblem` passes. -/
theorem flockLaw_second_ok {sample ns ks : Array ℕ} (hp : Spot.Select.secondProblem sample ns ks = none) :
    ∃ parsed, Spot.Draw.flockLaw (Spot.Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) = .ok parsed :=
  sorry

/-- **A second stage whose sample ends is accepted**: the sample object `Spot.Draw.drawFirstRead` builds passes
`Spot.Draw.checkDraw`. -/
theorem selectSecond_of_drawUnits {sample ns ks : Array ℕ} (hp : Spot.Select.secondProblem sample ns ks = none)
    {parsed : Flock.Draw.Law}
    (hl : Spot.Draw.flockLaw (Spot.Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) = .ok parsed)
    {src : ByteArray} {units : Array ℕ} {read : ℕ}
    (hu : Spot.Draw.drawUnits parsed (sample.size * ns.toList.sum) src = .ok (units, read)) :
    ∃ draw, Spot.Select.selectSecond sample ns ks src = .ok (draw, read) := sorry

/-- **The second stage is refused with probability at most `k_d·2⁻⁶⁴`** on `len` uniform bytes that hold every cell's
first read, at a replay sample and cells `secondProblem` passes. -/
theorem selectSecond_refused_le {sample ns ks : Array ℕ} (hp : Spot.Select.secondProblem sample ns ks = none)
    (len : ℕ) (hlen : secondRead sample.size ns ks ≤ len) :
    prCoin (fun f : Fin len → Fin 256 => (Spot.Select.selectSecond sample ns ks (bytesOf f)).isOk = false) ≤
      ((sample.size * ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := sorry

/-- **SPOT[Flock]'s second stage is refused with probability at most `k_d·2⁻⁶⁴`** on its `stage` uniform coins, at
every first-stage coin whose replay sample the second stage takes, where the coins' `stage · 2^20` bytes hold every
cell's first read. -/
theorem secondDraw_refused_le (h : Held) (c₀ : FlockDeploy.Coin)
    (hp : Spot.Select.secondProblem (h.reg.sample (src c₀)) h.reg.ns h.reg.ks = none)
    (hlen : secondRead (h.reg.sample (src c₀)).size h.reg.ns h.reg.ks ≤ h.stage * L) :
    prCoin (fun c₁ : Stage h => secondDraw h c₀ c₁ = none) ≤
      (((h.reg.sample (src c₀)).size * h.reg.ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := sorry

end SpotFlock.Refused
