module

public import Proofs.Flock.SpotFlock.Coins
public import Proofs.Flock.SpotFlock.Drawn
public import Proofs.Properties.Spot.SecondFormed
public import Proofs.Flock.Claims.Sampling.Select.SecondFirst

/-! SPOT[Flock]'s second stage refuses an honest run with probability at most `K · Σ_t k_t · 2⁻⁶⁴`. -/

@[expose] public section

namespace SpotFlock

open Properties.Spot Properties.Spot.SecurityProofs FlockSoundness FlockSoundness.Game SamplingDeploy
open scoped ENNReal

theorem select_subset_ascending {g : Reg} {K : ℕ} (hK : firstK g = some K) {src : ByteArray} {draw : Lean.Json}
    {proved : Array ℕ} {read : ℕ}
    (hs : Spot.Select.select g.rule g.program g.query g.table g.closure src = .ok (draw, proved, read)) :
    1 ≤ K ∧ proved.toList.Pairwise (· < ·) := by
  obtain ⟨hc, hne, n, named, hn, hl, hf⟩ := firstK_some hK
  refine ⟨(flockSamplingDesign_subset _ _ _ hf).2.1, ?_⟩
  obtain ⟨n', named', derived, hn', hl', hder, units, hpu, hcase⟩ := select_proved _ _ _ _ _ _ _ _ _ hs
  rw [hn] at hn'
  obtain rfl := Except.ok.inj hn'
  rw [hl] at hl'
  obtain rfl := Except.ok.inj hl'
  obtain rfl : derived = none := deriveSamplingDesign_subset _ _ _ _ _ (flockSamplingDesign_subset _ _ _ hf).1 hder
  rw [hpu hc]
  rcases hcase with ⟨hall, -⟩ | ⟨-, parsed, read', hp, hu⟩
  · exact absurd hall hne
  · have hp' : Spot.Draw.flockSamplingDesign named n = .ok parsed := hp
    rw [hf] at hp'
    obtain rfl : parsed = .subset K := (Except.ok.inj hp').symm
    have hu' : Spot.Draw.sampleUnits (.subset K) n src = .ok (units, read') := hu
    exact drawUnits_subset_ascending hu'

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

theorem secondDraw_none {c₀ : FlockDeploy.Coin} {c₁ : Stage h} (hn : secondDraw h c₀ c₁ = none) :
    ¬ ∃ draw read, Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (bytesOf (stageByte c₁)) =
      .ok (draw, read) := by
  rintro ⟨draw, read, hok⟩
  unfold secondDraw at hn
  rw [show stageSrc c₁ = bytesOf (stageByte c₁) from rfl, hok] at hn
  exact absurd hn (by simp)

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

def firstAccepts (c₀ : FlockDeploy.Coin) : Prop :=
  ∃ r, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table
    h.reg.first.closure (src c₀) = .ok r

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
