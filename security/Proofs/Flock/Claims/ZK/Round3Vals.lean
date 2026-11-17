module

public import Proofs.Flock.Claims.ZK.FirstRefusal
public import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init

/-! The masked zerocheck's first round, the model's values against the executable's. -/

@[expose] public section

namespace FlockComplete.Replay.Zc

open Flock FlockLevel3 FlockSoundness.Discharge.Composed.Zerocheck

local notation "𝔽" => FlockLevel3.GF128

theorem r1ab_eq (tr : Trace) (n : ℕ) (r : Array F128) (hrs : r.size = n) (c : ℕ → 𝔽) (rho : Array F128)
    (z : F128) : r1ab tr n r = (honest n (bitF tr.az) (bitF tr.bz) c r rho z).round1ab := by
  unfold r1ab
  simp only [honest]
  congr 1
  funext i
  apply eq_of_of
  rw [of_sumF, of_val]
  refine Finset.sum_congr rfl fun x hx => ?_
  rw [of_mul, of_mul, of_eqTable _ x (by rw [hrs]; exact Finset.mem_range.mp hx), hrs, of_skipHat, of_skipHat,
    ← mul_assoc]
  rfl

end FlockComplete.Replay.Zc

namespace FlockSoundness.Discharge.Composed.Zerocheck

open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128

theorem skipAt_congr {t t' : ℕ → 𝔽} {N : ℕ} (h : ∀ i < 64 * N, t i = t' i) (ζ : 𝔽) {x : ℕ} (hx : x < N) :
    skipAt t ζ x = skipAt t' ζ x := by
  unfold skipAt
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [h _ (by have := s.isLt; omega)]

theorem honest_round1ab_congr (n : ℕ) {a b a' b' : ℕ → 𝔽} (c c' : ℕ → 𝔽) (r rho rho' : Array F128) (z z' : F128)
    (ha : ∀ i < 64 * 2 ^ n, a i = a' i) (hb : ∀ i < 64 * 2 ^ n, b i = b' i) :
    (honest n a b c r rho z).round1ab = (honest n a' b' c' r rho' z').round1ab := by
  simp only [honest]
  refine Array.map_congr_left fun i _ => ?_
  refine congrArg _ (Finset.sum_congr rfl fun x hx => ?_)
  rw [skipAt_congr ha _ (Finset.mem_range.mp hx), skipAt_congr hb _ (Finset.mem_range.mp hx)]

theorem honest_round1c_congr (n : ℕ) (a b a' b' : ℕ → 𝔽) {c c' : ℕ → 𝔽} (r rho rho' : Array F128) (z z' : F128)
    (hc : ∀ i < 64 * 2 ^ n, c i = c' i) :
    (honest n a b c r rho z).round1c = (honest n a' b' c' r rho' z').round1c := by
  simp only [honest]
  refine Array.map_congr_left fun i _ => ?_
  refine congrArg _ (Finset.sum_congr rfl fun x hx => ?_)
  rw [skipAt_congr hc _ (Finset.mem_range.mp hx)]

end FlockSoundness.Discharge.Composed.Zerocheck

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128

theorem sixtyFour_mul_pow {mPts : ℕ} (Tz : TabZK 𝔽 mPts) : 64 * 2 ^ (Tz.S.m - 6) = 2 ^ Tz.S.m := by
  rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.add_sub_of_le (six_le Tz)]

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m)

theorem zcH_round1ab (tr : FlockComplete.Trace) (hrs : (rF Tz st c r).size = Tz.S.m - 6)
    (ha : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.az i = aT Tz zw i)
    (hb : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.bz i = bT Tz zw i) :
    (zcH Tz st c r zw).round1ab = FlockComplete.Replay.Zc.r1ab tr (Tz.S.m - 6) (rF Tz st c r) := by
  rw [FlockComplete.Replay.Zc.r1ab_eq tr _ _ hrs (cT Tz zw) (rhoF Tz st c r) (zcF Tz st c r)]
  unfold zcH
  exact Zerocheck.honest_round1ab_congr _ _ _ _ _ _ _ _
    (fun i hi => (ha i (hi.trans_eq (sixtyFour_mul_pow Tz))).symm)
    (fun i hi => (hb i (hi.trans_eq (sixtyFour_mul_pow Tz))).symm)

theorem zcH_round1c (tr : FlockComplete.Trace) (hrs : (rF Tz st c r).size = Tz.S.m - 6)
    (hz : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.z i = cT Tz zw i) :
    (zcH Tz st c r zw).round1c = FlockComplete.Replay.Zc.r1c tr (Tz.S.m - 6) (rF Tz st c r) := by
  rw [FlockComplete.Replay.Zc.r1c_eq tr _ _ hrs (aT Tz zw) (bT Tz zw) (rhoF Tz st c r) (zcF Tz st c r)]
  unfold zcH
  exact Zerocheck.honest_round1c_congr _ _ _ _ _ _ _ _ _ _
    (fun i hi => (hz i (hi.trans_eq (sixtyFour_mul_pow Tz))).symm)

theorem honestAt_r1ab_exe (tr : FlockComplete.Trace) (hrs : (rF Tz st c r).size = Tz.S.m - 6)
    (ha : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.az i = aT Tz zw i)
    (hb : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.bz i = bT Tz zw i)
    {L : ℕ} (R : Fin 2 → Fin L → 𝔽) {i : ℕ} (hi : i < 64) :
    honestAt Tz st c r zw R ((layOf Tz st).r1ab i) =
      fOf (FlockComplete.Replay.Zc.r1ab tr (Tz.S.m - 6) (rF Tz st c r))[i]! := by
  rw [honestAt_r1ab Tz st c r zw R i hi, zcH_round1ab Tz st c r zw tr hrs ha hb]

theorem honestAt_r1c_exe (tr : FlockComplete.Trace) (hrs : (rF Tz st c r).size = Tz.S.m - 6)
    (hz : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.z i = cT Tz zw i)
    {L : ℕ} (R : Fin 2 → Fin L → 𝔽) {i : ℕ} (hi : i < 64) :
    honestAt Tz st c r zw R ((layOf Tz st).r1c i) =
      fOf (FlockComplete.Replay.Zc.r1c tr (Tz.S.m - 6) (rF Tz st c r))[i]! := by
  rw [honestAt_r1c Tz st c r zw R i hi, zcH_round1c Tz st c r zw tr hrs hz]

end FlockSoundness.Discharge.Composed

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge

namespace Masked

theorem outerAt_eq {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {s spec sts}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, sts)) (j r i : ℕ) :
    outerAt c j r i = slotAt c (repBase x (2 * j + r) + 1) i := by
  unfold outerAt slotAt
  rw [outerSlot_eq hsx j r]

end Masked

end Claims.ZKMal
