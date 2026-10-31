import Proofs.Flock.VBridge.Runs.ZcLc
import Proofs.Flock.Soundness.Refine.HoloRepConverse

/-!
# The holo rep's program from E's zerocheck and lincheck

`holoProg_of_zclc` is `Runs.prog_of_zclc` at a holo shape: `holoProg_of_checks` with its zerocheck and lincheck checks
from `zclc_of_residuals_dec`. A holo record keeps `z_partial`'s round, which issued no coins, so the rep's rounds are
not `NoEmpty`; E reads them through `CoinSource.ofRounds`, which skips the empty ones (`ofRounds_filter`). The checks are
taken at the transcript's rounds with their empty ones filtered out (`R'`), whose rounds below `z_partial`'s are the
transcript's (`filter_getElem!_of`: every one of them issued coins, `HoloLcAnswered` and the zerocheck's counts), and
read back at the transcript's.
-/

namespace FlockC3.HoloValues

open Flock
open FlockSoundness.Refine (lStep RsRounds zcOf lcOf rsT rsSws encs wOf HoloLcAnswered holoRepProg
  holoProg_of_checks RoundsWF)
open Flock.RecAlgebra
open FlockVBridge.Compiled (RepData AlgebraHolds AccHolds)
open FlockVBridge.Runs (rhoOf tsOf zclc_of_residuals_dec rows_coins)

theorem list_filter_getElem!_of {α : Type} [Inhabited α] (p : α → Bool) :
    ∀ (l : List α) (K : ℕ), (∀ j < K, j < l.length ∧ p l[j]! = true) → ∀ j < K, (l.filter p)[j]! = l[j]!
  | [], _, _, _, _ => rfl
  | a :: l, K, h, j, hj => by
      have ha : p a = true := by simpa using (h 0 (by omega)).2
      rw [List.filter_cons_of_pos ha]
      cases j with
      | zero => rfl
      | succ j =>
          have := list_filter_getElem!_of p l (K - 1) (fun i hi => by
            have := h (i + 1) (by omega)
            simpa using this) j (by omega)
          simpa using this

/-- Filtering keeps the first `K` elements where each of them passes. -/
theorem filter_getElem!_of {α : Type} [Inhabited α] (p : α → Bool) (xs : Array α) (K : ℕ)
    (h : ∀ j < K, j < xs.size ∧ p xs[j]! = true) : ∀ j < K, (xs.filter p)[j]! = xs[j]! := by
  intro j hj
  have e : xs.filter p = (xs.toList.filter p).toArray := by apply Array.ext'; simp
  have e2 : ∀ (ys : Array α) (i : ℕ), ys[i]! = ys.toList[i]! := fun ys i => by
    rcases ys with ⟨l⟩; exact List.getElem!_toArray
  rw [e, e2, List.toList_toArray, e2]
  exact list_filter_getElem!_of p xs.toList K (fun i hi => by rw [← e2]; simpa using h i hi) j hj

theorem map_range_congr {β : Type} (n : ℕ) {f g : ℕ → β} (h : ∀ i < n, f i = g i) :
    (Array.range n).map f = (Array.range n).map g := by
  apply Array.ext (by simp) fun i h₁ h₂ => ?_
  simp only [Array.size_map, Array.size_range] at h₁
  simp [h i h₁]

theorem foldl_congr_mem {σ β : Type} (f g : σ → β → σ) : ∀ (l : List β) (init : σ),
    (∀ s, ∀ j ∈ l, f s j = g s j) → l.foldl f init = l.foldl g init
  | [], _, _ => rfl
  | j :: l, init, h => by
      rw [List.foldl_cons, List.foldl_cons, h init j (by simp)]
      exact foldl_congr_mem f g l _ fun s j' hj' => h s j' (by simp [hj'])

/-- A round of well-formed rounds that issued coins is one of them, and its coins are not empty. -/
theorem coins_ne_of_n {R : Array Flock.Round} (hwf : RoundsWF R) {j : ℕ} (h : (R[j]!).n ≠ 0) :
    j < R.size ∧ decide ((R[j]!).coins.size ≠ 0) = true := by
  have hj : j < R.size := by
    by_contra hge
    rw [getElem!_neg R j hge] at h
    exact h rfl
  exact ⟨hj, decide_eq_true (by rw [FlockSoundness.Refine.RoundsWF.get hwf j]; exact h)⟩

/-- `rhoOf` reads only the zerocheck's residual rounds. -/
theorem rhoOf_congr {R : Array Flock.Round} {t : Transcript} {m : ℕ} {P : ByteArray}
    (h : ∀ i < m - 6, R[t.k + 3 + i]! = t.rounds[t.k + 3 + i]!) : rhoOf ⟨R, t.k, P⟩ m = rhoOf t m := by
  unfold rhoOf
  refine map_range_congr _ fun i hi => ?_
  try dsimp only
  rw [h i hi]

/-- `tsOf` reads only the lincheck's rounds. -/
theorem tsOf_congr {R : Array Flock.Round} {k₀ : ℕ} {P : ByteArray} {t : Transcript} {k₁ kLog : ℕ}
    (h : ∀ j < kLog - 6, R[k₁ + 2 + j]! = t.rounds[k₁ + 2 + j]!) : tsOf ⟨R, k₀, P⟩ k₁ kLog = tsOf t k₁ kLog := by
  unfold tsOf
  refine map_range_congr _ fun j hj => ?_
  try dsimp only
  rw [h j hj]

/-- **The points `verifierRows` reads at a holo record** (`rows_coins`): every round below `z_partial`'s issued coins,
so they are the record's coins there, as at plain, at any extra claims and comb. -/
theorem holoRows_coins {sh : Shape} {rounds : Array (Array F128)} {xs : Array Claim} {cb v : Array F128}
    {info : Info Lin} {kappa : Array (Array (Array F128))} (hrows : verifierRows sh rounds xs cb = .ok (v, info, kappa))
    (hm : 14 ≤ sh.m) {t : Transcript} (ht : t.k = 0) (hd : rounds = t.rounds.map (·.coins))
    (hpos : ∀ j < t.k + 3 + (sh.m - 6) + 2 + (sh.kLog - 6),
      j < t.rounds.size ∧ decide ((t.rounds[j]!).coins.size ≠ 0) = true) :
    info.zc.r = pinned ++ (t.rounds[t.k + 1]!).coins ∧ info.zc.z = (t.rounds[t.k + 2]!).coins.getD 0 0 ∧
      info.zc.rho = rhoOf t sh.m ∧ info.lc.fold.ts = tsOf t (t.k + 3 + (sh.m - 6)) sh.kLog ∧
      info.lc.fold.alpha = (t.rounds[t.k + 3 + (sh.m - 6)]!).coins.getD 0 0 ∧
      info.lc.fold.rhoIn = info.zc.rho.extract 0 (sh.kLog - 6) ∧ info.lc.rIn = info.lc.fold.ts.reverse := by
  obtain ⟨R', hR'⟩ : ∃ R', R' = t.rounds.filter (fun x => decide (x.coins.size ≠ 0)) := ⟨_, rfl⟩
  have hget : ∀ j < t.k + 3 + (sh.m - 6) + 2 + (sh.kLog - 6), R'[j]! = t.rounds[j]! := by
    rw [hR']
    exact filter_getElem!_of _ t.rounds _ hpos
  have hd' : rounds.filter (·.size ≠ 0) = R'.map (·.coins) := by
    rw [hd, Array.filter_map, hR']
    rfl
  have hrows' : verifierRows sh (rounds.filter (·.size ≠ 0)) xs cb =
      verifierRows sh rounds xs cb := by
    unfold verifierRows
    rw [ofRounds_filter]
  obtain ⟨hr, hz, hrho, hα, -, -, hrin, hts, -, hrIn⟩ :=
    rows_coins (hrows'.trans hrows) hm (t := ⟨R', t.k, t.pending⟩) ht hd' (noEmpty_filter rounds)
  dsimp only at hr hz hrho hα hts
  rw [hget (t.k + 1) (by omega)] at hr
  rw [hget (t.k + 2) (by omega)] at hz
  rw [hget (t.k + 3 + (sh.m - 6)) (by omega)] at hα
  exact ⟨hr, hz, hrho.trans (rhoOf_congr fun i hi => hget (t.k + 3 + i) (by omega)),
    hts.trans (tsOf_congr fun j hj => hget (t.k + 3 + (sh.m - 6) + 2 + j) (by omega)), hα, hrin, hrIn⟩

/-- **The holo rep's program from E's zerocheck and lincheck.** `holoProg_of_checks` with its zerocheck and lincheck
checks from `zclc_of_residuals_dec`, at the transcript's coins, where the lincheck's rounds end at `comb + h`
(`decoded_comb_eq`); the frames, counts, ring switching (from round `k₁ + 2 + (k_log − 6) + 2`, past `z_partial`'s
round and `r'`'s) and Ligerito stay as `holoProg_of_checks` takes them. The zerocheck ends at round `k₁`. -/
theorem holoProg_of_zclc (sh : Shape) (d : RepData) (halg : AlgebraHolds sh d) (hacc : AccHolds sh d)
    (st : Setup) (sch : Flock.Schedule) (p : Proof) (extras : Array Claim) (t : Transcript) (ht : t.k = 0)
    (hd : d.rounds = t.rounds.map (·.coins)) (hwf : RoundsWF t.rounds) (hm : st.m = sh.m) (hm14 : 14 ≤ st.m)
    (hkL : st.fold.kLog = sh.kLog)
    (hzp : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) p.zerocheck)
    (hlp : LcProofOf (decoded sh d.w d.comb d.cons) p.lincheck)
    (hfoldPts : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
        .ok d.comb)
    -- the zerocheck's frames and counts
    (hn₀ : (t.rounds[t.k]!).n = 6) (hn₁ : (t.rounds[t.k + 1]!).n = st.m - 13) (hn₂ : (t.rounds[t.k + 2]!).n = 1)
    (hnr : ∀ i < st.m - 6, (t.rounds[t.k + 3 + i]!).n = 1)
    (ha₀ : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
      .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6]))
    (ha₁ : AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (st.m - 13)]))
    (ha₂ : AnswersL (t.rounds[t.k + 2]!)
      (encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1]))
    (har : ∀ i < st.m - 6, AnswersL (t.rounds[t.k + 3 + i]!)
      (encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1]))
    -- the lincheck's frames and counts, with hidden units
    (hlc : HoloLcAnswered ⟨t.rounds, t.k + 3 + (st.m - 6),
      ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck (hOfW sh d.w))
    -- ring switching, at the transcript's points
    {claims : Array Claim}
    (hcl : claims = Flock.abcClaims st.m st.kLog
      (zcOf p.zerocheck (pinned ++ (t.rounds[t.k + 1]!).coins) ((t.rounds[t.k + 2]!).coins.getD 0 0) (rhoOf t st.m))
      (lcOf p.lincheck ((t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1]!).coins.getD 0 0)
        (tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog)) ++ extras)
    {rpp : ℕ → Array F128} {gamma : Array F128} (hrs : p.ringSwitches.size = claims.size)
    (hshape : ∀ k < claims.size,
      p.ringSwitches[k]!.size = 128 ∧ claims[k]!.skip.size = 64 ∧ claims[k]!.x.size = st.m - 6)
    (hval : ∀ k < claims.size, dot (wOf claims[k]!) p.ringSwitches[k]! = claims[k]!.value)
    (hR : RsRounds ⟨t.rounds, t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2, ByteArray.empty⟩ p.ringSwitches
      claims.size rpp gamma)
    -- Ligerito
    {t₄ : Transcript}
    (hlg : Flock.ligerito st.merkle sch (rsT claims p.ringSwitches rpp gamma) p.cap p.ligerito
      (Flock.basis (rsSws claims rpp gamma))
      ⟨t.rounds, t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2 + claims.size + 1, ByteArray.empty⟩ =
        .ok ((), t₄)) :
    holoRepProg st sch p (hOfW sh d.w) extras t = .ok ((), t₄) ∧
      ∃ zc₁ t₁, Flock.bindAndZerocheck st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck t =
        .ok (zc₁, t₁) ∧ t₁.k = t.k + 3 + (st.m - 6) := by
  have e₁ : (t.rounds[t.k + 3 + (st.m - 6)]!).n = 1 := hlc.alpha.1
  have e₂ : (t.rounds[t.k + 3 + (st.m - 6) + 1]!).n = 1 := hlc.beta.1
  have e₃ : ∀ j < st.fold.kLog - 6, (t.rounds[t.k + 3 + (st.m - 6) + 2 + j]!).n = 1 := fun j hj =>
    (hlc.rounds j hj).1
  have hpos : ∀ j < t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6), (t.rounds[j]!).n ≠ 0 := by
    intro j hj
    by_cases h3 : j < t.k + 3
    · rcases (show j = t.k ∨ j = t.k + 1 ∨ j = t.k + 2 by omega) with rfl | rfl | rfl
      · rw [hn₀]; omega
      · rw [hn₁]; omega
      · rw [hn₂]; omega
    by_cases h4 : j < t.k + 3 + (st.m - 6)
    · rw [show j = t.k + 3 + (j - (t.k + 3)) by omega, hnr (j - (t.k + 3)) (by omega)]; omega
    by_cases h5 : j < t.k + 3 + (st.m - 6) + 2
    · rcases (show j = t.k + 3 + (st.m - 6) ∨ j = t.k + 3 + (st.m - 6) + 1 by omega) with rfl | rfl
      · rw [e₁]; omega
      · rw [e₂]; omega
    rw [show j = t.k + 3 + (st.m - 6) + 2 + (j - (t.k + 3 + (st.m - 6) + 2)) by omega,
      e₃ (j - (t.k + 3 + (st.m - 6) + 2)) (by omega)]
    omega
  obtain ⟨R', hR'⟩ : ∃ R', R' = t.rounds.filter (fun x => decide (x.coins.size ≠ 0)) := ⟨_, rfl⟩
  have hget : ∀ j < t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6), R'[j]! = t.rounds[j]! := by
    rw [hR']
    exact filter_getElem!_of _ t.rounds _ fun j hj => coins_ne_of_n hwf (hpos j hj)
  have hd' : d.rounds.filter (·.size ≠ 0) = R'.map (·.coins) := by
    rw [hd, Array.filter_map, hR']
    rfl
  have hrows : ∀ ex cb, verifierRows sh (d.rounds.filter (·.size ≠ 0)) ex cb = verifierRows sh d.rounds ex cb := by
    intro ex cb
    unfold verifierRows
    rw [ofRounds_filter]
  obtain ⟨hpc, hfold, hlfold, hcomb, hsize⟩ := zclc_of_residuals_dec sh
    { d with rounds := d.rounds.filter (·.size ≠ 0) }
    (by
      obtain ⟨v, info, kappa, S, rs, h1, h2⟩ := halg
      exact ⟨v, info, kappa, S, rs, (hrows _ _).trans h1, h2⟩)
    (fun v info kappa h => hacc v info kappa ((hrows _ _).symm.trans h)) st p ⟨R', t.k, t.pending⟩ ht hd'
    (noEmpty_filter d.rounds) hm hm14 hkL (fun v info kappa h => hzp v info kappa ((hrows _ _).symm.trans h)) hlp
    (fun v info kappa h => hfoldPts v info kappa ((hrows _ _).symm.trans h))
  dsimp only at hpc hfold hlfold hcomb hsize
  have hrho : rhoOf ⟨R', t.k, t.pending⟩ st.m = rhoOf t st.m :=
    rhoOf_congr fun i hi => hget (t.k + 3 + i) (by omega)
  have hts : tsOf ⟨R', t.k, t.pending⟩ (t.k + 3 + (st.m - 6)) st.fold.kLog =
      tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog :=
    tsOf_congr fun j hj => hget (t.k + 3 + (st.m - 6) + 2 + j) (by omega)
  rw [hget (t.k + 2) (by omega)] at hpc
  rw [hget (t.k + 1) (by omega), hget (t.k + 2) (by omega), hrho] at hfold
  rw [hget (t.k + 3 + (st.m - 6)) (by omega), hget (t.k + 3 + (st.m - 6) + 1) (by omega), hts,
    decoded_comb_eq] at hlfold
  rw [hget (t.k + 3 + (st.m - 6)) (by omega), hget (t.k + 3 + (st.m - 6) + 1) (by omega), hget (t.k + 2) (by omega),
    hrho, hts] at hcomb
  have hlfold' : (List.range' 0 (st.fold.kLog - 6)).foldl
        (fun s j => lStep p.lincheck.rounds j s (t.rounds[t.k + 3 + (st.m - 6) + 2 + j]!).coins)
        ((t.rounds[t.k + 3 + (st.m - 6)]!).coins.getD 0 0 * p.zerocheck.finalA + p.zerocheck.finalB +
          (t.rounds[t.k + 3 + (st.m - 6) + 1]!).coins.getD 0 0, #[]) =
      (dot (addH d.comb (hOfW sh d.w)) p.lincheck.zPartial, tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog) := by
    rw [← hlfold]
    refine foldl_congr_mem _ _ _ _ fun s j hj => ?_
    rw [List.mem_range'_1] at hj
    rw [hget (t.k + 3 + (st.m - 6) + 2 + j) (by omega)]
  have hC : ∀ i < st.m - 6, (R'[t.k + 3 + i]!).coins.getD 0 0 = (t.rounds[t.k + 3 + i]!).coins.getD 0 0 :=
    fun i hi => by rw [hget (t.k + 3 + i) (by omega)]
  obtain ⟨t₁, h₁, -, hk₁, -⟩ := bindAndZerocheck_of_checks st.m hm14 st.digest
    (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck t hn₀ hn₁ hn₂ hnr ha₀ ha₁ ha₂ har
    (C := fun i => (R'[t.k + 3 + i]!).coins) rfl rfl hC hpc hfold
  exact ⟨holoProg_of_checks st sch p (hOfW sh d.w) extras t hm14 hn₀ hn₁ hn₂ hnr ha₀ ha₁ ha₂ har
    (C := fun i => (R'[t.k + 3 + i]!).coins) rfl rfl hC hpc hfold (k₁ := t.k + 3 + (st.m - 6)) rfl hlc
    (comb := d.comb) hlfold' hcomb hsize (k₂ := t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2) rfl hcl hrs
    hshape hval hR hlg, _, t₁, h₁, hk₁⟩

end FlockC3.HoloValues
