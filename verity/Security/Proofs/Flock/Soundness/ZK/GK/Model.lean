import Proofs.Flock.Soundness.ZK.GK.Sim

/-!
# Zero knowledge: the model Theorem GK reads

`Model` is the analysis's description of one session against a deterministic `V*` (fixed tape): its real view `real ω`,
the dummy run `dummy ω` (the protocol with a dummy witness), the simulator's rewind `rwS v₀ σ` from a completed view
`v₀`, hybrid 1's rewind `rwR v₀ ω` (the real witness, against the coins of `v₀`), completion `comp`, the coin conflict
`bad` (`conflicts` of `GK/Extract.lean` on the views' openings), and `V*`'s view before its final message (`pre`) with
completion read from it (`compP`) and an incomplete view rebuilt from it (`inc`). Its fields `comp_eq`, `inc_real`,
`inc_dummy` and `bind_real` are facts about the protocol's runs; `RewindClose` (Lemma B, one rewind) and `PrefinalClose`
(Lemma C) are its two distances, taken as hypotheses at `δ_B` and `δ_C`. The simulator (`sim`), the two hybrids and the
collision finder are `GK/Sim.lean`'s `gk` on the model's runs.

This file and its imports build on Mathlib alone, as do the extractor (`GK/Extract.lean`) and the simulator
(`GK/Sim.lean`); `GK/Theorem.lean` proves the bound.
-/

namespace FlockSoundness.ZK.GK

open Finset Game

/-- **One session against a deterministic `V*`**, as the hybrid argument reads it. The simulator uses only `dummy`, `rwS`,
`comp` and `bad`; `real`, `rwR`, `pre`, `compP` and `inc` are the analysis's. -/
structure Model (V P Ω Ωd Ωs : Type) where
  /-- `V*`'s view of a real run, on the prover's coins `ω`. -/
  real : Ω → V
  /-- `V*`'s view of a run with a dummy witness. -/
  dummy : Ωd → V
  /-- The simulator's rewind from the completed view `v₀`: fresh dummy openings for the coins `V*` opened in `v₀`. -/
  rwS : V → Ωs → V
  /-- Hybrid 1's rewind from `v₀`: a run with the real witness against the coins of `v₀`. -/
  rwR : V → Ω → V
  /-- `V*` completed the run: it opened its coins and the openings verify. -/
  comp : V → Bool
  /-- A coin conflict between two views: an opening of the second disagrees with one of the first at the same position,
  and both reach the committed root. -/
  bad : V → V → Bool
  /-- `V*`'s view before its final message. -/
  pre : V → P
  /-- Whether `V*` completes, read from its view before its final message. -/
  compP : P → Bool
  /-- An incomplete view, rebuilt from its part before `V*`'s final message. -/
  inc : P → V
  comp_eq : ∀ v, comp v = compP (pre v)
  inc_real : ∀ ω, comp (real ω) = false → real ω = inc (pre (real ω))
  inc_dummy : ∀ ω, comp (dummy ω) = false → dummy ω = inc (pre (dummy ω))
  /-- A real-witness rewind in which `V*` opens the first run's coins is the real run. -/
  bind_real : ∀ ω₀, comp (dummy ω₀) = true → ∀ ω, bad (dummy ω₀) (rwR (dummy ω₀) ω) = false →
    rwR (dummy ω₀) ω = real ω

variable {V P Ω Ωd Ωs : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ωd] [Nonempty Ωd] [Fintype Ωs] [Nonempty Ωs]

namespace Model

variable (m : Model V P Ω Ωd Ωs)

/-- **Lemma C's distance**: every event of `V*`'s view before its final message has probabilities within `δ_C` under
the real and the dummy witness (`Session.adaptive_prefinal_hm96`). -/
def PrefinalClose (δC : ℝ) : Prop :=
  ∀ U : P → Prop, |avgR (fun ω => ind U (m.pre (m.real ω))) - avgR (fun ω => ind U (m.pre (m.dummy ω)))| ≤ δC

/-- **Lemma B's distance, one rewind**: from every completed dummy view, every event of the simulator's rewind has
probability within `δ_B` of the real-witness rewind's (`Session.session_shvzk_hm96`). -/
def RewindClose (δB : ℝ) : Prop :=
  0 ≤ δB ∧ ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ U : V → Prop,
    |avgR (fun σ => ind U (m.rwS (m.dummy ω₀) σ)) - avgR (fun ω => ind U (m.rwR (m.dummy ω₀) ω))| ≤ δB

/-- **The simulator** (`gk`), with threshold `t` and cap `M`. -/
def sim (t M : ℕ) : Rand (Out V × ℕ) := gk Ωd Ωs m.dummy m.rwS m.comp m.bad t M

/-- Hybrid 1: the simulator with real-witness rewinds. -/
def hyb₁ (t M : ℕ) : Rand (Out V × ℕ) := gk Ωd Ω m.dummy m.rwR m.comp m.bad t M

/-- Hybrid 2: rewinds are fresh real runs, and the coins are never checked. -/
def hyb₂ (t M : ℕ) : Rand (Out V × ℕ) := gk Ωd Ω m.dummy (fun _ => m.real) m.comp (fun _ _ => false) t M

/-- **The collision finder**: the simulator, its `bind` outcome read by `pick`. -/
def finder (pick : V → V → Pair) (t M : ℕ) : Rand Pair := (m.sim t M).map fun o => Out.pickBy pick o.1

/-- The dummy run's completion rate. -/
noncomputable abbrev pD : ℝ := rate Ωd fun ω => m.comp (m.dummy ω)

/-- The real run's completion rate. -/
noncomputable abbrev pR : ℝ := rate Ω fun ω => m.comp (m.real ω)

end Model

end FlockSoundness.ZK.GK
