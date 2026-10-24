import Proofs.Accounting.TS.Session.Lanes.Lane
import Definitions.Pouw.H1Zy.Chain
import Definitions.Pouw.H1Zy.Dequant

/-!
# A lane's part of `LeafChecks` at its Definition's own chain and dequant

`Lane`'s `laneOk_of_def` and `leafChecks_of_computes` hold at any chain words and any dequant. Here they are at the
Definition's own: `laneChain`, `Pouw.H1Zy.chainWords` at the salt, position and roots `decLane` reads (`h1zy.lane_partial`'s
fold, `Definitions.Pouw.H1Zy.Chain`), and `Pouw.H1Zy.dequantW` (`PouwZyDequant_v1`, `Definitions.Pouw.H1Zy.Dequant`).
Both are held to the Python by `Proofs.Pouw.H1Zy.Vectors`.

* `partOf_laneChain`: `LeafChecks`' `part` at `laneChain` is `h1zy.lane_partial`: the fold's words where an admit bit is
  set, zero words for a dead leaf.
* `dqOf_dequantW`: its `dq` at `dequantW` is `h1zy.dequant` on the s32 and f32 words.
* `laneOk_of_def_h1`, `leafChecks_h1`: `laneOk_of_def` and `leafChecks_of_computes` with neither parameter.
-/

namespace Accounting.TS.Lanes

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge PouwWindow.Lanes

/-- **A lane's chain words** (`h1zy.lane_partial`'s fold, ungated): `Pouw.H1Zy.chainWords` at the public inputs' salt,
position and roots. -/
def laneChain {K nw : ℕ} (P : LanePub) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8) (b : Fin 8 → Fin K → BitVec 8) :
    Fin nw → BitVec 32 :=
  Pouw.H1Zy.chainWords P.salt P.position P.roots l a b

/-- A lane's rows' codes as `lanePartial` reads them: by slot, each code's s8 encoding. -/
def codesOf {K n : ℕ} (a : Fin n → Row K) : List (List ℕ) :=
  List.ofFn fun s => List.ofFn fun k => (BitVec.ofInt 8 ((a s).codes k)).toNat

/-- **`part` at the lane's chain is `h1zy.lane_partial`**: word `w` of `Pouw.H1Zy.lanePartial` on the rows' codes where
some admit bit is set, `0` for a dead leaf. -/
theorem partOf_laneChain {K nw : ℕ} (P : LanePub) (l : Fin 32) (a : Fin 4 → Row K) (b : Fin 8 → Row K)
    (w : Fin nw) :
    partOf laneChain P l a b w =
      if ∃ i, P.admit i = true then
        BitVec.ofNat 32 ((Pouw.H1Zy.lanePartial K P.salt P.position P.roots l.val (codesOf a) (codesOf b)).getD w.val 0)
      else 0 := by
  unfold partOf laneChain Pouw.H1Zy.chainWords codesOf Pouw.H1Zy.mask32
  by_cases h : ∃ i, P.admit i = true
  · have ht : Pouw.H1Zy.tree Pouw.H1Zy.orBit false (List.ofFn P.admit) = true := by
      obtain ⟨i, hi⟩ := h
      cases e : Pouw.H1Zy.tree Pouw.H1Zy.orBit false (List.ofFn P.admit)
      · exact absurd hi (by rw [(Pouw.H1Zy.tree_orBit _).1 e _ (List.mem_ofFn.2 ⟨i, rfl⟩)]; simp)
      · rfl
    simp only [ht, h, ↓reduceIte]
  · have ht : Pouw.H1Zy.tree Pouw.H1Zy.orBit false (List.ofFn P.admit) = false := by
      refine (Pouw.H1Zy.tree_orBit _).2 fun x hx => ?_
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
      cases e : P.admit i
      · rfl
      · exact absurd ⟨i, e⟩ h
    simp only [ht, h, Bool.false_eq_true, ↓reduceIte]

theorem dequant_lt (z sa sw : ℕ) : Pouw.H1Zy.dequant z sa sw < 2 ^ 16 := by
  unfold Pouw.H1Zy.dequant Verity.Fp32.f32ToBf16RnBits Verity.Fp32.BF16_NAN_F2FP
  dsimp only
  split
  · norm_num
  · exact lt_of_le_of_lt Nat.and_le_right (by norm_num)

/-- **`dq` at `dequantW` is `h1zy.dequant`** on z's s32 word and the scales' f32 words. -/
theorem dqOf_dequantW (z sa sw : ℤ) :
    dqOf Pouw.H1Zy.dequantW [z, sa, sw] =
      (Pouw.H1Zy.dequant (BitVec.ofInt 32 z).toNat (BitVec.ofInt 32 sa).toNat (BitVec.ofInt 32 sw).toNat : ℤ) := by
  simp only [dqOf, Pouw.H1Zy.dequantW, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (dequant_lt _ _ _)]

/-- **Claim 3 of `LeafChecks` at a lane, at the Definition's own chain and dequant**: where
`PouwH1LaneZy_v1{PouwH1Tile_v1}`'s `ok` on the lane's words is `true`, `decLane`'s view meets `LaneOk` at
`dqOf dequantW` and `partOf laneChain`. -/
theorem laneOk_of_def_h1 {K nw : ℕ} (T : Tile) (rd : String → ℕ → Option Bool) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17)
    (h : Pouw.H1Zy.laneOk (laneChain (K := K) (nw := nw)) Pouw.H1Zy.dequantW (laneIn K nw T rd) = true) :
    LaneOk (dqOf Pouw.H1Zy.dequantW) (partOf laneChain) (decLane K nw T rd) :=
  laneOk_of_def T laneChain Pouw.H1Zy.dequantW rd hK32 hK h

/-- **`LeafChecks` at the Definition's own chain and dequant**: `LaneComputes` and `CombineComputes` at `laneChain` and
`dequantW` give `Accounting.Assumptions.TS.LeafChecks` at `dqOf dequantW` and `partOf laneChain`. -/
theorem leafChecks_h1 {K nw : ℕ} (T : Tile) (sess : Option (Fin 32) → RowsSession) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17)
    (hL : LaneComputes T (laneChain (K := K) (nw := nw)) Pouw.H1Zy.dequantW sess)
    (hC : CombineComputes sess (decC nw T)) :
    Accounting.Assumptions.TS.LeafChecks (dqOf Pouw.H1Zy.dequantW) (partOf laneChain) sess (decL K nw T)
      (decC nw T) :=
  leafChecks_of_computes T laneChain Pouw.H1Zy.dequantW sess hK32 hK hL hC

end Accounting.TS.Lanes
