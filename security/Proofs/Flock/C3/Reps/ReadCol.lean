module

public import Proofs.Flock.VBridge.ResLayer
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

/-! The opening unit reads its row and sibling columns (`ReadCol`). -/

@[expose] public section

namespace FlockC3.Reps

open FlockSoundness FlockSoundness.Audit
open FlockVBridge.OfVStar (ReadCol)

def OpenRowRead (LANES H : ℕ) : Prop :=
  ∀ {net : Flock.Net} (hU : Built net),
    Flock.HmNets.netOf (Flock.RecOpen.unitName LANES H) (Flock.RecOpen.unitLaid LANES H) = .ok net →
      net.unitLog ≤ 32 → ∀ c < 128 * LANES, ReadCol (Rows.ofNet hU) c

def OpenSibRead (LANES H : ℕ) : Prop :=
  ∀ {net : Flock.Net} (hU : Built net),
    Flock.HmNets.netOf (Flock.RecOpen.unitName LANES H) (Flock.RecOpen.unitLaid LANES H) = .ok net →
      Flock.RecOpen.valid LANES H = true → net.unitLog ≤ 32 →
        ∀ h < H, ∀ i < 512, ReadCol (Rows.ofNet hU) (128 * LANES + 512 * h + i)

def OpenReadCol (LANES H : ℕ) : Prop :=
  ∀ {net : Flock.Net} (hU : Built net),
    Flock.HmNets.netOf (Flock.RecOpen.unitName LANES H) (Flock.RecOpen.unitLaid LANES H) = .ok net →
      Flock.RecOpen.valid LANES H = true → net.unitLog ≤ 32 →
        ∀ c < 128 * LANES + 512 * H, ReadCol (Rows.ofNet hU) c

theorem openReadCol_of {LANES H : ℕ} (hr : OpenRowRead LANES H) (hs : OpenSibRead LANES H) : OpenReadCol LANES H := by
  intro net hU hn hv hu c hc
  by_cases hcr : c < 128 * LANES
  · exact hr hU hn hu c hcr
  · have h1 : (c - 128 * LANES) / 512 < H := by omega
    have h2 : (c - 128 * LANES) % 512 < 512 := Nat.mod_lt _ (by norm_num)
    have h3 : c = 128 * LANES + 512 * ((c - 128 * LANES) / 512) + (c - 128 * LANES) % 512 := by omega
    have := hs hU hn hv hu _ h1 _ h2
    rwa [← h3] at this

end FlockC3.Reps
