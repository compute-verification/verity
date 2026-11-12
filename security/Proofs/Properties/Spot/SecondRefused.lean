module

public import Proofs.Flock.SpotFlock.Coins
public import Proofs.Flock.SpotFlock.Drawn
public import Proofs.Properties.Spot.SecondFormed
public import Proofs.Flock.Claims.Sampling.Select.SecondFirst

/-!
# SPOT[Flock]'s second stage refuses an honest run with probability at most `K · Σ_t k_t · 2⁻⁶⁴`

The second stage (`Spot.Select.selectSecond`) refuses where the replay sample and the cells are not a second stage's
(`secondProblem`), where its sampling design is not well formed (`Spot.Draw.flockLaw`, `Spot.Draw.checkDraw`), or
where a cell's sample does not end on the stage's bytes (`Spot.Draw.drawUnits`). The first two read no coin of the
second stage, and
the second never happens where `secondProblem` passes (`Properties.Spot.SecurityProofs.secondFormed`). The third
happens with probability at most `k_d · 2⁻⁶⁴` on a uniform stream that holds every cell's first read, `k_d` the
`|sample| · Σ_t k_t` proof blocks the second stage samples (`Properties.Spot.SecurityProofs.selectSecond_refused_le`).

The code's second stage reads the `stage · 2^20` bytes of its `stage` coins (`stageSrc`), uniform on uniform coins
(`stageEquiv`), so the bound is the code's:

* `secondDraw_refused_le`: at a first-stage coin whose replay sample `secondProblem` passes, where the stage's bytes
  hold every cell's first read (`secondRead ≤ stage · 2^20`), the code's second stage (`secondDraw`) refuses with
  probability at most `k_d · 2⁻⁶⁴` over its coins.
* `secondProblem_first`: a first stage of count `K` (`firstK`) that accepts hands the second stage `K` replay blocks,
  strictly ascending, so `secondProblem` passes there at cells it passes on some replay sample: its other tests read
  the cells alone.
* `secondDraw_refused_le_first`: so at every first-stage coin the first stage accepts, the code's second stage refuses
  with probability at most `K · Σ_t k_t · 2⁻⁶⁴`, where `stage · 2^20` bytes hold the `K` replay blocks' first reads.
* `refused_after_first_le`, `secondDraw_none_le`: over both stages' coins, the first stage accepts and the second
  refuses with at most that probability, and the code samples no second stage with at most that probability more
  than the first stage refuses with.

Where the first stage refuses, the replay sample is empty and the second stage refuses on every coin.
-/

@[expose] public section

namespace SpotFlock

open Properties.Spot Properties.Spot.SecurityProofs FlockSoundness FlockSoundness.Game SamplingDeploy
open scoped ENNReal

/-- **A first stage of count `K` proves strictly ascending units**, at least one. -/
theorem select_subset_ascending {g : Reg} {K : ℕ} (hK : firstK g = some K) {src : ByteArray} {draw : Lean.Json}
    {proved : Array ℕ} {read : ℕ}
    (hs : Spot.Select.select g.rule g.program g.query g.table g.closure src = .ok (draw, proved, read)) :
    1 ≤ K ∧ proved.toList.Pairwise (· < ·) := by
  obtain ⟨hc, hne, n, named, hn, hl, hf⟩ := firstK_some hK
  refine ⟨(flockLaw_subset _ _ _ hf).2.1, ?_⟩
  obtain ⟨n', named', derived, hn', hl', hder, units, hpu, hcase⟩ := select_proved _ _ _ _ _ _ _ _ _ hs
  rw [hn] at hn'
  obtain rfl := Except.ok.inj hn'
  rw [hl] at hl'
  obtain rfl := Except.ok.inj hl'
  obtain rfl : derived = none := deriveLaw_subset _ _ _ _ _ (flockLaw_subset _ _ _ hf).1 hder
  rw [hpu hc]
  rcases hcase with ⟨hall, -⟩ | ⟨-, parsed, read', hp, hu⟩
  · exact absurd hall hne
  · have hp' : Spot.Draw.flockLaw named n = .ok parsed := hp
    rw [hf] at hp'
    obtain rfl : parsed = .subset K := (Except.ok.inj hp').symm
    have hu' : Spot.Draw.drawUnits (.subset K) n src = .ok (units, read') := hu
    exact drawUnits_subset_ascending hu'

/-- **Where a first stage of count `K` accepts, `secondProblem` passes** at cells it passes on some replay sample: the
replay sample is the first stage's `K` units, strictly ascending, and `secondProblem`'s other tests read the cells
alone. -/
theorem secondProblem_first {g : RegTwo} {K : ℕ} (hK : firstK g.first = some K) {src : ByteArray} {draw : Lean.Json}
    {proved : Array ℕ} {read : ℕ}
    (hs : Spot.Select.select g.first.rule g.first.program g.first.query g.first.table g.first.closure src =
      .ok (draw, proved, read))
    {s' : Array ℕ} (hc : Spot.Select.secondProblem s' g.ns g.ks = none) :
    (g.sample src).size = K ∧ Spot.Select.secondProblem (g.sample src) g.ns g.ks = none := by
  have he : g.sample src = proved := by
    unfold RegTwo.sample
    rw [hs]
  obtain ⟨hK1, hasc⟩ := select_subset_ascending hK hs
  have hsz : proved.size = K := select_subset_size hK hs
  rw [he]
  exact ⟨hsz, secondProblem_of_ascending (by omega) hasc hc⟩

variable (h : Held)

/-- The code's second stage returns no sample only where `selectSecond` refuses the stage's bytes. -/
theorem secondDraw_none {c₀ : FlockDeploy.Coin} {c₁ : Stage h} (hn : secondDraw h c₀ c₁ = none) :
    ¬ ∃ draw read, Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (bytesOf (stageByte c₁)) =
      .ok (draw, read) := by
  rintro ⟨draw, read, hok⟩
  unfold secondDraw at hn
  rw [show stageSrc c₁ = bytesOf (stageByte c₁) from rfl, hok] at hn
  exact absurd hn (by simp)

/-- **SPOT[Flock]'s second stage refuses with probability at most `k_d · 2⁻⁶⁴`** over its `stage` uniform coins, at a
first-stage coin whose replay sample `secondProblem` passes, where the coins' `stage · 2^20` bytes hold every cell's
first read. -/
theorem secondDraw_refused_le (c₀ : FlockDeploy.Coin)
    (hp : Spot.Select.secondProblem (h.reg.sample (src c₀)) h.reg.ns h.reg.ks = none)
    (hlen : secondRead (h.reg.sample (src c₀)).size h.reg.ns h.reg.ks ≤ h.stage * L) :
    prCoin (fun c₁ : Stage h => secondDraw h c₀ c₁ = none) ≤
      (((h.reg.sample (src c₀)).size * h.reg.ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  let E : (Fin (h.stage * L) → Fin 256) → Prop := fun f => ¬ ∃ draw read,
    Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (bytesOf f) = .ok (draw, read)
  calc prCoin (fun c₁ : Stage h => secondDraw h c₀ c₁ = none)
      ≤ prCoin (fun c₁ : Stage h => E (stageByte c₁)) := prCoin_mono fun _ hn => secondDraw_none h hn
    _ = prCoin E := FlockSoundness.ZK.prCoin_congr_equiv
        ((stageEquiv h.stage).trans (Equiv.funUnique (Fin 1) (Fin (h.stage * L) → Fin 256))) E
    _ ≤ _ := selectSecond_refused_le hp _ hlen

/-- **The first stage accepts** round 0's coin: `Select.select` samples on its bytes. -/
def firstAccepts (c₀ : FlockDeploy.Coin) : Prop :=
  ∃ r, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table
    h.reg.first.closure (src c₀) = .ok r

/-- **Where the first stage accepts, SPOT[Flock]'s second stage refuses with probability at most
`K · Σ_t k_t · 2⁻⁶⁴`** over its `stage` uniform coins: a first stage of count `K`, cells `secondProblem` passes on some
replay sample, and `stage · 2^20` bytes that hold the first reads of `K` replay blocks' cells. -/
theorem secondDraw_refused_le_first {K : ℕ} (hK : firstK h.reg.first = some K) {s' : Array ℕ}
    (hc : Spot.Select.secondProblem s' h.reg.ns h.reg.ks = none)
    (hlen : secondRead K h.reg.ns h.reg.ks ≤ h.stage * L) (c₀ : FlockDeploy.Coin) (ha : firstAccepts h c₀) :
    prCoin (fun c₁ : Stage h => secondDraw h c₀ c₁ = none) ≤
      ((K * h.reg.ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  obtain ⟨⟨draw, proved, read⟩, hs⟩ := ha
  obtain ⟨hsz, hp⟩ := secondProblem_first hK hs hc
  have hb := secondDraw_refused_le h c₀ hp (by rw [hsz]; exact hlen)
  rw [hsz] at hb
  exact hb

/-- **Over both stages' coins, the first stage accepts and the second refuses with probability at most
`K · Σ_t k_t · 2⁻⁶⁴`**, under `secondDraw_refused_le_first`'s conditions. -/
theorem refused_after_first_le {K : ℕ} (hK : firstK h.reg.first = some K) {s' : Array ℕ}
    (hc : Spot.Select.secondProblem s' h.reg.ns h.reg.ks = none)
    (hlen : secondRead K h.reg.ns h.reg.ks ≤ h.stage * L) :
    prCoin (fun c : FlockDeploy.Coin × Stage h => firstAccepts h c.1 ∧ secondDraw h c.1 c.2 = none) ≤
      ((K * h.reg.ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  rw [FlockSoundness.ZK.prCoin_prod_split fun c₀ (c₁ : Stage h) => firstAccepts h c₀ ∧ secondDraw h c₀ c₁ = none]
  refine (avg_mono fun c₀ => ?_).trans (avg_const _).le
  by_cases ha : firstAccepts h c₀
  · exact (prCoin_mono fun _ hx => hx.2).trans (secondDraw_refused_le_first h hK hc hlen c₀ ha)
  · exact (prCoin_mono (q := fun _ => False) fun _ hx => ha hx.1).trans (prCoin_false.le.trans zero_le)

/-- **Over both stages' coins, SPOT[Flock]'s code samples no second stage with probability at most the first stage's
chance of refusing and `K · Σ_t k_t · 2⁻⁶⁴`**, under `secondDraw_refused_le_first`'s conditions. -/
theorem secondDraw_none_le {K : ℕ} (hK : firstK h.reg.first = some K) {s' : Array ℕ}
    (hc : Spot.Select.secondProblem s' h.reg.ns h.reg.ks = none)
    (hlen : secondRead K h.reg.ns h.reg.ks ≤ h.stage * L) :
    prCoin (fun c : FlockDeploy.Coin × Stage h => secondDraw h c.1 c.2 = none) ≤
      prCoin (fun c₀ : FlockDeploy.Coin => ¬ firstAccepts h c₀) +
        ((K * h.reg.ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  classical
  refine (prCoin_mono (q := fun c : FlockDeploy.Coin × Stage h =>
    ¬ firstAccepts h c.1 ∨ (firstAccepts h c.1 ∧ secondDraw h c.1 c.2 = none)) fun c hn => ?_).trans
    ((Verity.Game.prCoin_or_le _ _).trans (add_le_add ?_ (refused_after_first_le h hK hc hlen)))
  · by_cases ha : firstAccepts h c.1
    · exact Or.inr ⟨ha, hn⟩
    · exact Or.inl ha
  · exact (FlockSoundness.ZK.prCoin_fst fun c₀ : FlockDeploy.Coin => ¬ firstAccepts h c₀).le

end SpotFlock
