import Proofs.Flock.Claims.ZK.FirstRefusal
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
import Proofs.Flock.Soundness.Discharge.Composed.Inner

/-!
# The masked zerocheck's first round, the model's values against the executable's

Round `3` of the masked session absorbs the padded `round1_ab` and `round1_c`, which a cut holds at layout indices
`i` and `64 + i` (`Zk.PadLayout.r1ab`, `r1c`). Toward `SimPiop`'s zerocheck:

* `Replay.Zc.r1ab_eq`: the executable's `round1_ab` is the soundness analysis's honest one on the trace's bits, the twin
  of `Replay.Zc.r1c_eq`.
* `Zerocheck.honest_round1ab_congr`, `honest_round1c_congr`: the honest first round reads its tables below `64 · 2^n`
  only (`skipAt_congr`).
* `Composed.zcH_round1ab`, `zcH_round1c`: so the draw's clear first round is the executable's on a trace whose bits are
  the model's tables, at the draw's zerocheck vector `rF`; `honestAt_r1ab`, `honestAt_r1c`: the clear values a cut
  masks at layout indices `i` and `64 + i` (`innerOf`'s `vals`).
* `Masked.outerAt_eq`: a rep's outer coin at the opened coins is the coin of the round after its first
  (`repBase + 1`), the one the session's second zerocheck squeeze reads.
-/

namespace FlockComplete.Replay.Zc

open Flock FlockLevel3 FlockSoundness.Discharge.Composed.Zerocheck

local notation "𝔽" => FlockLevel3.GF128

/-- **The prover's `round1_ab` is the honest one** on the trace's `A·z` and `B·z` bits. -/
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

/-- A skip row at block-row `x < N` reads its table below `64 · N` only. -/
theorem skipAt_congr {t t' : ℕ → 𝔽} {N : ℕ} (h : ∀ i < 64 * N, t i = t' i) (ζ : 𝔽) {x : ℕ} (hx : x < N) :
    skipAt t ζ x = skipAt t' ζ x := by
  unfold skipAt
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [h _ (by have := s.isLt; omega)]

/-- **The honest `round1_ab` reads the A- and B-side tables below `64 · 2^n` only.** -/
theorem honest_round1ab_congr (n : ℕ) {a b a' b' : ℕ → 𝔽} (c c' : ℕ → 𝔽) (r rho rho' : Array F128) (z z' : F128)
    (ha : ∀ i < 64 * 2 ^ n, a i = a' i) (hb : ∀ i < 64 * 2 ^ n, b i = b' i) :
    (honest n a b c r rho z).round1ab = (honest n a' b' c' r rho' z').round1ab := by
  simp only [honest]
  congr 1
  funext i
  congr 1
  refine Finset.sum_congr rfl fun x hx => ?_
  rw [skipAt_congr ha _ (Finset.mem_range.mp hx), skipAt_congr hb _ (Finset.mem_range.mp hx)]

/-- **The honest `round1_c` reads the witness table below `64 · 2^n` only.** -/
theorem honest_round1c_congr (n : ℕ) (a b a' b' : ℕ → 𝔽) {c c' : ℕ → 𝔽} (r rho rho' : Array F128) (z z' : F128)
    (hc : ∀ i < 64 * 2 ^ n, c i = c' i) :
    (honest n a b c r rho z).round1c = (honest n a' b' c' r rho' z').round1c := by
  simp only [honest]
  congr 1
  funext i
  congr 1
  refine Finset.sum_congr rfl fun x hx => ?_
  rw [skipAt_congr hc _ (Finset.mem_range.mp hx)]

end FlockSoundness.Discharge.Composed.Zerocheck

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128

theorem sixtyFour_mul_pow {mPts : ℕ} (Tz : TabZK 𝔽 mPts) : 64 * 2 ^ (Tz.S.m - 6) = 2 ^ Tz.S.m := by
  rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.add_sub_of_le (six_le Tz)]

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m)

/-- **The draw's clear `round1_ab` is the executable's** on a trace whose `A·z`, `B·z` bits are the model's A- and
B-side tables, at the draw's zerocheck vector. -/
theorem zcH_round1ab (tr : FlockComplete.Trace) (hrs : (rF Tz st c r).size = Tz.S.m - 6)
    (ha : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.az i = aT Tz zw i)
    (hb : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.bz i = bT Tz zw i) :
    (zcH Tz st c r zw).round1ab = FlockComplete.Replay.Zc.r1ab tr (Tz.S.m - 6) (rF Tz st c r) := by
  rw [FlockComplete.Replay.Zc.r1ab_eq tr _ _ hrs (cT Tz zw) (rhoF Tz st c r) (zcF Tz st c r)]
  unfold zcH
  exact Zerocheck.honest_round1ab_congr _ _ _ _ _ _ _ _
    (fun i hi => (ha i (hi.trans_eq (sixtyFour_mul_pow Tz))).symm)
    (fun i hi => (hb i (hi.trans_eq (sixtyFour_mul_pow Tz))).symm)

/-- **The draw's clear `round1_c` is the executable's** on a trace whose `z` bits are the model's witness table. -/
theorem zcH_round1c (tr : FlockComplete.Trace) (hrs : (rF Tz st c r).size = Tz.S.m - 6)
    (hz : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.z i = cT Tz zw i) :
    (zcH Tz st c r zw).round1c = FlockComplete.Replay.Zc.r1c tr (Tz.S.m - 6) (rF Tz st c r) := by
  rw [FlockComplete.Replay.Zc.r1c_eq tr _ _ hrs (aT Tz zw) (bT Tz zw) (rhoF Tz st c r) (zcF Tz st c r)]
  unfold zcH
  exact Zerocheck.honest_round1c_congr _ _ _ _ _ _ _ _ _ _
    (fun i hi => (hz i (hi.trans_eq (sixtyFour_mul_pow Tz))).symm)

theorem honestAt_lt64 {L : ℕ} (R : Fin 2 → Fin L → 𝔽) {k : ℕ} (hk : k < 64) :
    honestAt Tz st c r zw R k = fOf (zcH Tz st c r zw).round1ab[k]! := by
  simp only [honestAt, hk, ↓reduceIte]

theorem honestAt_lt128 {L : ℕ} (R : Fin 2 → Fin L → 𝔽) {k : ℕ} (hk : 64 ≤ k) (hk' : k < 128) :
    honestAt Tz st c r zw R k = fOf (zcH Tz st c r zw).round1c[k - 64]! := by
  simp only [honestAt, show ¬ k < 64 by omega, hk', ↓reduceIte]

/-- **The clear value at layout index `i < 64`** (`round1_ab`'s) is the executable's `round1_ab` at `i`. -/
theorem honestAt_r1ab (tr : FlockComplete.Trace) (hrs : (rF Tz st c r).size = Tz.S.m - 6)
    (ha : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.az i = aT Tz zw i)
    (hb : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.bz i = bT Tz zw i)
    {L : ℕ} (R : Fin 2 → Fin L → 𝔽) {i : ℕ} (hi : i < 64) :
    honestAt Tz st c r zw R i = fOf (FlockComplete.Replay.Zc.r1ab tr (Tz.S.m - 6) (rF Tz st c r))[i]! := by
  rw [honestAt_lt64 Tz st c r zw R hi, zcH_round1ab Tz st c r zw tr hrs ha hb]

/-- **The clear value at layout index `64 + i`** (`round1_c`'s) is the executable's `round1_c` at `i`. -/
theorem honestAt_r1c (tr : FlockComplete.Trace) (hrs : (rF Tz st c r).size = Tz.S.m - 6)
    (hz : ∀ i < 2 ^ Tz.S.m, FlockComplete.Replay.Zc.bitF tr.z i = cT Tz zw i)
    {L : ℕ} (R : Fin 2 → Fin L → 𝔽) {i : ℕ} (hi : i < 64) :
    honestAt Tz st c r zw R (64 + i) = fOf (FlockComplete.Replay.Zc.r1c tr (Tz.S.m - 6) (rF Tz st c r))[i]! := by
  rw [honestAt_lt128 Tz st c r zw R (by omega) (by omega), Nat.add_sub_cancel_left,
    zcH_round1c Tz st c r zw tr hrs hz]

end FlockSoundness.Discharge.Composed

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge

namespace Masked

theorem outerSlot_eq {x : Msg} {s spec sts} (hsx : FlockDeploy.setupOf true x = some (s, spec, sts)) (j r : ℕ) :
    outerSlot x j r = repBase x (2 * j + r) + 1 := by
  simp only [outerSlot, repBase, hsx]

/-- **A rep's outer coin is read at the round after its first**: at a live setup, table `j`'s rep `r`'s outer coin
`i` is element `i` of the coin at `repBase + 1`, the round of the session's `rOuter` squeeze. -/
theorem outerAt_eq {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {s spec sts}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, sts)) (j r i : ℕ) :
    outerAt c j r i = slotAt c (repBase x (2 * j + r) + 1) i := by
  unfold outerAt slotAt
  rw [outerSlot_eq hsx j r]

end Masked

end Claims.ZKMal
