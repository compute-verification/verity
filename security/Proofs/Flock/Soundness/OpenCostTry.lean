module

public import Definitions.Flock.Compiled
public import Definitions.Flock.Accounting.Padded

@[expose] public section

namespace FlockSoundness.OpenCostTry

open FlockSoundness.Accounting FlockSoundness.Model

def q0 (m : ℕ) : ℕ :=
  if m = 25 then 277 else if m = 26 then 242 else if m = 27 then 229 else if m ≤ 31 then 219 else 218

def zkSched (m : ℕ) : Schedule := ((fast100 m).getD default).withQueries₀ (q0 m)

def openOne (l : Level) : ℕ := 3 + pathLen l

def hOpenMax (m logN : ℕ) : ℕ := max (((zkSched m).levels.map openOne).foldr max 0) (3 + logN)

def hSelfScan (m logN : ℕ) : ℕ :=
  2 * (((zkSched m).levels.map fun l => l.queries * openOne l).sum + 168 * (3 + logN) + 3)

theorem log_pads_decide : Nat.log 2 4096 = 12 ∧ Nat.log 2 8192 = 13 := by decide

theorem log_pads_kernel : Nat.log 2 4096 = 12 ∧ Nat.log 2 8192 = 13 := by decide +kernel

theorem hOpenMax_decide : ∀ m < 36, ∀ logN < 14, 25 ≤ m → hOpenMax m logN ≤ 19 := by decide

theorem hOpenMax_kernel : ∀ m < 36, ∀ logN < 14, 25 ≤ m → hOpenMax m logN ≤ 19 := by decide +kernel

theorem hSelfScan_kernel : ∀ m < 36, ∀ logN < 14, 25 ≤ m → hSelfScan m logN ≤ 23830 := by decide +kernel

theorem hOpenMax_rfl : ((List.range 11).all fun i => hOpenMax (25 + i) 13 ≤ 19) = true := by rfl

end FlockSoundness.OpenCostTry
