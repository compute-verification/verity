import Proofs.Flock.Soundness.Discharge.ZkSession.Session
import Proofs.Flock.Soundness.Accounting.PaddedNumbers

/-!
# The `--zk` table's bound at C-Flock ZK's parameters (live/PROTOCOL.md §5, §7)

`TabZK.bound` is `padLinkError + (padRepError + εInner)²`. At C-Flock ZK's parameters it is at most `2^-203`
(`TabZK.bound_le_m1`):
* the padded level 0 at every `m` the verifier serves, `25 ≤ m ≤ 35`: `padLinkError + padRepError² ≤ 2^-205`
  (`padTableError_le_m1`);
* the inner proof over `F` of order `2^128` with 168 queries, `N ≤ 2^13` and radius `δ ≥ 7/16`:
  `εInner = (N + 3)/|F| + (1 − δ)^168 ≤ 2^-114` (`εInner_le_m1`);
* `(a + b)² ≤ 2a² + 2b²` puts them together: `2·2^-205 + 2·2^-228 ≤ 2^-203`.

The radius: the pads code `RS[F, dom, L]` with `8L ≤ N` (C-Flock ZK's `N = 8·next_pow2(L)`) has relative unique-decoding
radius `(1 − L/N)/2 ≥ 7/16` (`relUDR_ge_m1`, from ArkLib's `relativeUniqueDecodingRadius_RS_eq`), so `δ = 7/16` meets
`TabZK.hδ`. The positions: the protocol reduces each query word to `lo(w) mod N` (§5), and for `N = 2^b`, `b ≤ 64`, that
map meets `TabZK.hpos` (`posLo_uniform`, from `Arith.Correct.lo_balanced`).

`zk_flock_count_execOS_m1`: the `--zk` count curve with every planned table at C-Flock ZK's parameters,
`miss(K) + 2^-203 + δ_link`.
-/

namespace FlockSoundness.Discharge.ZkSession

open Game Model Accounting Audit
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

/-! ## The inner proof's positions and radius -/

/-- The inner proof's query position for `N = 2^b`: a query word's `lo` reduced mod `N` (live/PROTOCOL.md §5). -/
def posLo (A : Arith F K) (b : ℕ) (x : F) : Fin (2 ^ b) := ⟨A.lo x % 2 ^ b, Nat.mod_lt _ (Nat.two_pow_pos b)⟩

omit [DecidableEq F] [DecidableEq K] in
/-- **The protocol's positions are uniform**: `Pr[lo(w) mod 2^b ∈ D] = |D|/2^b`. -/
theorem posLo_uniform (A : Arith F K) (hA : A.Correct) (b : ℕ) (hb : b ≤ 64) (D : Finset (Fin (2 ^ b))) :
    prCoin (fun x => posLo A b x ∈ D) ≤ (D.card : ℝ≥0∞) / ((2 ^ b : ℕ) : ℝ≥0∞) := by
  classical
  refine le_of_eq ?_
  rw [prCoin_eq_card]
  have hfib : (Finset.univ.filter fun x : F => posLo A b x ∈ D).card =
      ∑ t ∈ D, (Finset.univ.filter fun x : F => A.lo x % 2 ^ b = t.val).card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := posLo A b) (t := D) (fun x hx => by simpa using hx)]
    refine Finset.sum_congr rfl fun t ht => ?_
    congr 1
    ext x
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨_, h⟩
      exact congrArg Fin.val h
    · intro h
      have hx : posLo A b x = t := Fin.ext h
      exact ⟨hx ▸ ht, hx⟩
  rw [hfib, Finset.sum_congr rfl fun t _ => hA.lo_balanced b hb t.val t.isLt, Finset.sum_const, smul_eq_mul,
    hA.card_F]
  have h2 : (2 : ℕ) ^ 128 = 2 ^ b * 2 ^ (128 - b) := by rw [← pow_add]; congr 1; omega
  rw [h2, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
  push_cast
  rw [ENNReal.mul_div_mul_right _ _ (by positivity) (by simp)]

omit [Fintype F] in
/-- **The pads code's radius**: with `8L ≤ N`, `RS[F, dom, L]` decodes uniquely up to a fraction `7/16`. -/
theorem relUDR_ge_m1 {N L : ℕ} (dom : Fin N ↪ F) [NeZero L] (h : 8 * L ≤ N) :
    (7 / 16 : ℝ≥0) ≤ Code.relativeUniqueDecodingRadius (ReedSolomon.code dom L : Set (Fin N → F)) := by
  have hL : L ≤ Fintype.card (Fin N) := by rw [Fintype.card_fin]; omega
  rw [ReedSolomon.relativeUniqueDecodingRadius_RS_eq hL, Fintype.card_fin]
  have hN : (0 : ℝ≥0) < N := by
    have := NeZero.pos L
    exact_mod_cast (show 0 < N by omega)
  have hLN : (L : ℝ≥0) / N ≤ 1 / 8 := by
    rw [div_le_div_iff₀ hN (by norm_num), one_mul]
    exact_mod_cast (show L * 8 ≤ N by omega)
  have h78 : (7 / 8 : ℝ≥0) ≤ 1 - (L : ℝ≥0) / N := by
    have h18 : (1 / 8 : ℝ≥0) ≤ 1 := by rw [div_le_one (by norm_num)]; norm_num
    rw [le_tsub_iff_right (hLN.trans h18)]
    calc (7 / 8 : ℝ≥0) + L / N ≤ 7 / 8 + 1 / 8 := by gcongr
      _ = 1 := by norm_num
  calc (7 / 16 : ℝ≥0) = (7 / 8) / 2 := by norm_num
    _ ≤ (1 - (L : ℝ≥0) / N) / 2 := by gcongr

omit [Field F] [DecidableEq F] in
/-- **The inner proof's error at C-Flock ZK's parameters**: `εInner ≤ 2^-114` over `F` of order `2^128`, with 168
queries, `N ≤ 2^13` and `δ ≥ 7/16`. -/
theorem εInner_le_m1 {N : ℕ} (hF : Fintype.card F = 2 ^ 128) (hN : N ≤ 2 ^ 13) {δ : ℝ≥0} (hδ : 7 / 16 ≤ δ) :
    εInner N F 168 δ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(114 : ℝ))) := by
  unfold εInner
  rw [hF]
  have h1 : (N + 3 : ℝ≥0∞) / ((2 ^ 128 : ℕ) : ℝ≥0∞) = ENNReal.ofReal ((N + 3 : ℝ) / 2 ^ 128) := by
    rw [ENNReal.ofReal_div_of_pos (by positivity)]
    congr 1
    · rw [ENNReal.ofReal_add (by positivity) (by norm_num), ENNReal.ofReal_natCast]
      norm_num
    · rw [show ((2 : ℝ) ^ 128) = ((2 ^ 128 : ℕ) : ℝ) by norm_num, ENNReal.ofReal_natCast]
  have h2 : ((1 - δ : ℝ≥0) : ℝ≥0∞) ^ 168 = ENNReal.ofReal (((1 - δ : ℝ≥0) : ℝ) ^ 168) := by
    rw [ENNReal.ofReal_pow (NNReal.coe_nonneg _), ENNReal.ofReal_coe_nnreal]
  rw [h1, h2, ← ENNReal.ofReal_add (by positivity) (by positivity)]
  refine ENNReal.ofReal_le_ofReal ?_
  have hδ' : ((1 - δ : ℝ≥0) : ℝ) ≤ 9 / 16 := by
    have : (1 - δ : ℝ≥0) ≤ 9 / 16 := by
      rw [tsub_le_iff_right]
      calc (1 : ℝ≥0) = 9 / 16 + 7 / 16 := by norm_num
        _ ≤ 9 / 16 + δ := by gcongr
    exact_mod_cast this
  have hN' : (N : ℝ) ≤ 2 ^ 13 := by exact_mod_cast hN
  calc (N + 3 : ℝ) / 2 ^ 128 + ((1 - δ : ℝ≥0) : ℝ) ^ 168 ≤ (2 ^ 13 + 3) / 2 ^ 128 + (9 / 16) ^ 168 := by
        gcongr
    _ ≤ (2 : ℝ) ^ (-(114 : ℝ)) := by
      rw [Real.rpow_neg (by norm_num), show (114 : ℝ) = ((114 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
      norm_num

/-! ## One table -/

/-- A `--zk` table at C-Flock ZK's parameters: the padded level 0 of `m1Schedule m`, `25 ≤ m ≤ 35`, with C-Flock ZK's
padding and two extra lanes on a statement in `fast100`'s range, and the inner proof with 168 queries, `N ≤ 2^13`
and `δ ≥ 7/16`. -/
def TabZK.M1 (T : TabZK F mPts) : Prop :=
  ∃ m, 25 ≤ m ∧ m ≤ 35 ∧ T.sch = m1Schedule m ∧ T.t = m1Pad m ∧ T.e = 2 ∧ (T.S.shape mPts).InRange ∧
    T.N ≤ 2 ^ 13 ∧ T.q = 168 ∧ 7 / 16 ≤ T.δ

/-- **A `--zk` table's bound at C-Flock ZK's parameters** is at most `2^-203`. -/
theorem TabZK.bound_le_m1 (hF : Fintype.card F = 2 ^ 128) (T : TabZK F mPts) (hT : T.M1) :
    T.bound ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(203 : ℝ))) := by
  obtain ⟨m, h1, h2, hsch, ht, he, hst, hN, hq, hδ⟩ := hT
  unfold TabZK.bound
  set r := padRepError T.sch (T.S.shape mPts) T.t T.e
  set l := padLinkError T.sch (T.S.shape mPts) T.t
  set b := εInner T.N F T.q T.δ with hbdef
  have hl : 0 ≤ l := by simp only [l, padLinkError]; positivity
  have hsq : (ENNReal.ofReal r + b) ^ 2 ≤ 2 * (ENNReal.ofReal r ^ 2 + b ^ 2) := by
    have := ENNReal.rpow_add_le_mul_rpow_add_rpow (ENNReal.ofReal r) b (p := 2) (by norm_num)
    rw [ENNReal.rpow_two, ENNReal.rpow_two, ENNReal.rpow_two] at this
    norm_num at this
    exact this
  have hr2 : ENNReal.ofReal r ^ 2 ≤ ENNReal.ofReal (r ^ 2) := by
    rcases le_or_gt 0 r with h | h
    · rw [ENNReal.ofReal_pow h]
    · rw [ENNReal.ofReal_of_nonpos h.le]
      simp
  have hpt : l + r ^ 2 ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
    have := padTableError_le_m1 m h1 h2 _ hst
    rw [← hsch, ← ht, ← he] at this
    exact this
  have hb : b ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(114 : ℝ))) := by
    rw [hbdef, hq]
    exact εInner_le_m1 hF hN hδ
  have hpos : (0 : ℝ) ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by positivity
  have hpos' : (0 : ℝ) ≤ (2 : ℝ) ^ (-(114 : ℝ)) := by positivity
  calc ENNReal.ofReal l + (ENNReal.ofReal r + b) ^ 2
      ≤ ENNReal.ofReal l + 2 * (ENNReal.ofReal (r ^ 2) + b ^ 2) := by
        gcongr
        exact hsq.trans (by gcongr)
    _ = ENNReal.ofReal l + 2 * ENNReal.ofReal (r ^ 2) + 2 * b ^ 2 := by ring
    _ ≤ 2 * ENNReal.ofReal l + 2 * ENNReal.ofReal (r ^ 2) + 2 * b ^ 2 := by
        gcongr
        exact le_mul_of_one_le_left zero_le one_le_two
    _ = 2 * ENNReal.ofReal (l + r ^ 2) + 2 * b ^ 2 := by
        rw [ENNReal.ofReal_add hl (sq_nonneg r)]
        ring
    _ ≤ 2 * ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) + 2 * ENNReal.ofReal ((2 : ℝ) ^ (-(114 : ℝ))) ^ 2 := by
        gcongr
    _ = ENNReal.ofReal (2 * (2 : ℝ) ^ (-(205 : ℝ)) + 2 * ((2 : ℝ) ^ (-(114 : ℝ))) ^ 2) := by
        rw [ENNReal.ofReal_add (by positivity) (by positivity), ENNReal.ofReal_mul (by norm_num),
          ENNReal.ofReal_mul (by norm_num), ENNReal.ofReal_pow hpos', ENNReal.ofReal_ofNat]
    _ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(203 : ℝ))) := by
        refine ENNReal.ofReal_le_ofReal ?_
        rw [Real.rpow_neg (by norm_num), Real.rpow_neg (by norm_num), Real.rpow_neg (by norm_num),
          show (205 : ℝ) = ((205 : ℕ) : ℝ) by norm_num, show (114 : ℝ) = ((114 : ℕ) : ℝ) by norm_num,
          show (203 : ℝ) = ((203 : ℕ) : ℝ) by norm_num, Real.rpow_natCast, Real.rpow_natCast, Real.rpow_natCast]
        norm_num

/-! ## The audit at C-Flock ZK's parameters -/

section Audit

variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- With every planned table's bound at most `c`, so is the knowledge term. -/
theorem ksAvgZK_le {c : ℝ≥0∞} (h : ∀ S R (j : Fin (planZ S R).length), ((planZ S R).get j).bound ≤ c) (L : Law n)
    (R : Reg) : ksAvgZK planZ L R ≤ c :=
  (avg_mono fun _ => iSup_le fun j => h _ _ j).trans (avg_const c).le

/-- **The `--zk` count curve at the live draw, every table at C-Flock ZK's parameters**:
`Pr[accept ∧ at least K wrong units] ≤ miss(K) + 2^-203 + δ_link`. -/
theorem zk_flock_count_execOS_m1 (hA : A.Correct) (hM1 : ∀ S R (j : Fin (planZ S R).length), ((planZ S R).get j).M1)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    (Xc : (R : Reg) → Cont (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZK A planZ) R → (Fin C.N → Bool))
    {δlink : (R : Reg) → Cont (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZK A planZ) R → ℝ≥0∞}
    (hlink : (analysisZK A P planZ tab (decodeOfPlace A P planZ tab place) Xc
      (loweringSoundZK_of_place A P planZ tab place)).LinkSound δlink)
    (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZK A planZ))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (Xc (reg σ) (cont σ))).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZK A planZ)) σ ≤
      (Law.stratified σs ks hks).miss K + ENNReal.ofReal ((2 : ℝ) ^ (-(203 : ℝ))) + δlink (reg σ) (cont σ) :=
  (zk_flock_count_execOS A P planZ tab hA σs ks hks St hS Kd pop Lb src hA3 place Xc hlink K σ).trans
    (add_le_add (add_le_add le_rfl (ksAvgZK_le planZ (fun S R j => (planZ S R |>.get j).bound_le_m1 hA.card_F
      (hM1 S R j)) _ _)) le_rfl)

end Audit

end FlockSoundness.Discharge.ZkSession
