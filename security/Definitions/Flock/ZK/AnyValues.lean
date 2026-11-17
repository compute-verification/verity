module

public import Definitions.Core.Game
public import Definitions.Flock.ZK.SHVZK

/-! A table's release behind the ZK server's check, simulated per outcome at every witness carrying the public region
data, with no premise that its values pass the constraint. -/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- **Simulated per outcome**: at every outcome `s` and every event `E`, the released bytes `rel` jointly with the
outcome `out = s` are no likelier than the simulator's output at `s`, plus `ε s`. -/
def PerOutcome {Ω St Ωs V : Type} [Fintype Ω] [Fintype Ωs] (out : Ω → St) (rel : Ω → V) (S : St → Ωs → V)
    (ε : St → ℝ≥0∞) : Prop :=
  ∀ s (E : V → Prop), prCoin (fun ω => out ω = s ∧ E (rel ω)) ≤ prCoin (fun σ => E (S s σ)) + ε s

/-- **`X` is simulated by `Y` within `ε`**, one-sided: no event is more than `ε` likelier under `X`'s randomness than
under `Y`'s. -/
def Within {Ω Ωs V : Type} [Fintype Ω] [Fintype Ωs] (X : Ω → V) (Y : Ωs → V) (ε : ℝ≥0∞) : Prop :=
  ∀ E : V → Prop, prCoin (fun ω => E (X ω)) ≤ prCoin (fun σ => E (Y σ)) + ε

def Splits {Ω St V : Type} (rel : Ω → V) (out : Ω → St) (B : St → Ω → V) : Prop :=
  ∀ ω, rel ω = B (out ω) ω

def AgreesWhere {Ω V : Type} (go : Ω → Bool) (view d : Ω → V) : Prop :=
  ∀ ω, go ω = true → view ω = d ω

def gated {Ω V P : Type} (go : Ω → Bool) (view : Ω → V) (cut : Ω → P) (ω : Ω) : V ⊕ P :=
  if go ω then .inl (view ω) else .inr (cut ω)

def gatedSim {Ωf Ωc V P : Type} (simFull : Ωf → V) (simCut : Ωc → P) (b : Bool) (σ : Ωf × Ωc) : V ⊕ P :=
  if b then .inl (simFull σ.1) else .inr (simCut σ.2)

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

/-- **The coins are non-degenerate for the table**: the prover's rank check passes, both paddings reach every vector of
values at their opened positions, each rep's `W^r` is a bijection and each `β^r` is nonzero, so `S_shvzk`'s counting
goes through. -/
def Simulable : Prop :=
  T.Rank ∧ T.PadOnto ∧ (∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) ∧ T.PadsOnto ∧ ∀ r, T.β r ≠ 0

def RegPublic (w : W) : Prop := T.sReg w = T.pubReg

def HoldsAt (w : W) (ω : T.Rand) : Prop :=
  ∀ r : Fin 2, ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i = T.theta r (T.view w ω).pre

def Gated (go : T.Rand → Bool) (w : W) : Prop :=
  ∀ ω, go ω = true → T.HoldsAt w ω

noncomputable def release (go : T.Rand → Bool) (w : W) : T.Rand → T.View ⊕ T.PreView :=
  gated go (T.view w) (T.preView w)

end Table

namespace Release

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] {J : ℕ} (T : Fin J → Table F K W Salt Seed D Hi VL)
  [CharP F 2] {St : Type}

noncomputable def bytesAt (full : St → Fin J → Bool) (w : Fin J → W) (s : St) (ω : ∀ j, (T j).Rand) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((T j).view (w j) (ω j)) else .inr ((T j).preView (w j) (ω j))

noncomputable def simAt (full : St → Fin J → Bool) (w₀ : Fin J → W) (s : St) (σ : ∀ j, (T j).SimRand × (T j).Rand) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((T j).sim (w₀ j) (σ j).1) else .inr ((T j).preView (w₀ j) (σ j).2)

def Gated (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool) (w : Fin J → W) : Prop :=
  ∀ ω j, full (out ω) j = true → (T j).HoldsAt (w j) (ω j)

end Release

end FlockSoundness.ZK
