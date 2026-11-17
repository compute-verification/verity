module

import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Init.Data.Repr
public import Proofs.Flock.Soundness.Discharge.Layout.Scope
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-! A circuit for the statements no file binds (`defaultCircuit`). -/

@[expose] public section

namespace FlockC3.VStarList

def defaultRows : Array (Array ℕ) := (Array.replicate 128 #[]).push #[128]

def defaultBuild : Except String Flock.Net :=
  Flock.Net.ofRows "unit" 129 128 #[⟨0, 1, #[]⟩] #[⟨0, 1, #[]⟩] defaultRows defaultRows

theorem defaultBuild_isOk : defaultBuild.isOk = true := by decide +kernel

def defaultNet : Flock.Net :=
  match defaultBuild with
  | .ok n => n
  | .error _ => default

theorem defaultBuild_eq : defaultBuild = .ok defaultNet := by
  have h := defaultBuild_isOk
  unfold defaultNet
  cases hx : defaultBuild with
  | error e => rw [hx] at h; cases h
  | ok n => rfl

/-- `vstarParsesScoped_of_list` needs some circuit where no file binds a statement: one net `unit` of one
input word and no computed row, which `Layout.scopeOk` accepts. -/
def defaultCircuit : Flock.Circuit :=
  { (default : Flock.Circuit) with nets := #[("unit", defaultNet)], unitNet := 0, outNet := 0 }

theorem defaultCircuit_built : FlockSoundness.Built defaultCircuit.unit :=
  ⟨_, _, _, _, _, _, _, defaultBuild_eq⟩

theorem defaultCircuit_scope : FlockSoundness.Discharge.Layout.scopeOk defaultCircuit = true := by
  rw [← FlockSoundness.Discharge.FailClosed.scopeOk_eq]
  decide +kernel

end FlockC3.VStarList
