import Proofs.Flock.C3.HoloValues.Read
import Proofs.Flock.VBridge.Sparse.Plays

/-!
# A rep's sparse public values are its outputs' (`SparseRepTie`'s pub half)

`pubOf_eq_holoPub`: the sparse phase's public values at the coins `c` (`pubOf`) are `holoPub`'s at a rep's outputs
where `c`'s `z`, `alpha` and the first `k_log − 6` coordinates of its `x_rest` and `y_rest` are the outputs' (`ρ`, and
`ts` reversed), and its `U` is `holoPub`'s sum over the slots' units at those coordinates.
`pub_of_unit`: at a rep's unit (`RepUnit`), the points check 5's `checkSparse` reads off `verifierRows` at the unit's
rounds (`info.zc.z`, `info.lc.fold.alpha`, `info.lc.fold.rhoIn`, `info.lc.rIn`) are the coins its play's outputs read
(`rep_of_unit_coins`, `holoRows_coins`, every round below `z_partial`'s issuing coins: `hpos_of_unit`).
-/

namespace FlockC3.Sparse

noncomputable section

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Model
open Flock.RecAlgebra (verifierRows)
open FlockVBridge.Sparse (SCoins pubOf)
open FlockVBridge.Runs (rhoOf tsOf map_range_get)
open FlockC3.HoloValues (RepUnit rep_of_unit_coins holoRows_coins)

/-- **`pubOf` is `holoPub`** where `c`'s points are the outputs' coordinate by coordinate. -/
theorem pubOf_eq_holoPub {S : Model.Statement} {sh : SparseShape} (h6 : 6 ≤ sh.ell) (hell : sh.ell ≤ S.kLog)
    {units : Fin (2 ^ sh.sLog) → Finset ℕ} {zc : Model.ZcOut 𝔽 S.m} {lc : HoloLcOut 𝔽 S.m (S.kLog - 6)} {c : SCoins}
    (hz : fOf c.z = zc.z) (ha : fOf c.alpha = lc.alpha)
    (hx : ∀ j (hj : j < S.kLog - 6), fOf c.xRest[j]! = zc.ρ ⟨j, by have := S.hk; omega⟩)
    (hy : ∀ j (hj : j < S.kLog - 6), fOf c.yRest[j]! = lc.ts ⟨S.kLog - 6 - 1 - j, by omega⟩)
    (hU : ∀ σ : Fin (2 ^ sh.sLog), fOf c.U[σ.val]! = ∑ u ∈ units σ,
      Model.eqAt (fun i : Fin (S.kLog - sh.ell) => fOf c.xRest[sh.ell - 6 + i.val]!) u *
        Model.eqAt (fun i : Fin (S.kLog - sh.ell) => fOf c.yRest[sh.ell - 6 + i.val]!) u) :
    pubOf sh c = Model.holoPub S sh units zc lc := by
  have hx' : ∀ j, j < S.kLog - 6 → fOf c.xRest[j]! =
      if h : j < S.kLog - 6 then zc.ρ ⟨j, by have := S.hk; omega⟩ else 0 := fun j hj => by
    rw [dite_eq_left hj, hx j hj]
  have hy' : ∀ j, j < S.kLog - 6 → fOf c.yRest[j]! =
      if h : j < S.kLog - 6 then lc.ts ⟨S.kLog - 6 - 1 - j, by omega⟩ else 0 := fun j hj => by
    rw [dite_eq_left hj, hy j hj]
  unfold pubOf Model.holoPub
  congr 1
  · funext j; exact hx' j.val (by omega)
  · funext j; exact hy' j.val (by omega)
  · funext σ
    rw [hU σ]
    refine Finset.sum_congr rfl fun u _ => ?_
    congr 2
    · funext i; exact hx' _ (by omega)
    · funext i; exact hy' _ (by omega)

/-- **The layout factors are `holoPub`'s `U`**: slot `σ`'s factor is `eq(x_hi, q) eq(y_hi, q)` at its unit
`q = pos0 + σ` below the range's count and 0 past it (`layoutFactors`), and `x_hi`, `y_hi` are the coordinates of
`k_log − 6` points past `ELL − 6` (`eqTable_extract`); the range ends inside both tables, so its units are below
`2^(k_log − ELL)`. -/
theorem layoutFactors_U {sh : RecSparse.Shape} {xs ys U : Array Flock.F128} {rng : Flock.Range} {K : ℕ}
    (h : RecCoins.layoutFactors sh xs ys rng = .ok U) (hx : xs.size = K - 6) (hy : ys.size = K - 6)
    (h6 : 6 ≤ sh.ell) (hell : sh.ell ≤ K) (units : Fin (2 ^ sh.sLog) → Finset ℕ)
    (hunits : ∀ σ : Fin (2 ^ sh.sLog),
      units σ = (if σ.val < rng.count then ({rng.pos0 + σ.val} : Finset ℕ) else ∅).filter (· < 2 ^ (K - sh.ell)))
    (σ : Fin (2 ^ sh.sLog)) : fOf U[σ.val]! = ∑ u ∈ units σ,
      Model.eqAt (fun i : Fin (K - sh.ell) => fOf xs[sh.ell - 6 + i.val]!) u *
        Model.eqAt (fun i : Fin (K - sh.ell) => fOf ys[sh.ell - 6 + i.val]!) u := by
  unfold RecCoins.layoutFactors at h
  dsimp only at h
  split at h
  · cases h
  rename_i hc
  obtain rfl := (Except.ok.inj h).symm
  have hk : RecSparse.PRE_ROUNDS = 6 := rfl
  have hxe : xs.extract (sh.ell - RecSparse.PRE_ROUNDS) xs.size =
      xs.extract (sh.ell - 6) (sh.ell - 6 + (K - sh.ell)) := by rw [hk, hx]; congr 1; omega
  have hye : ys.extract (sh.ell - RecSparse.PRE_ROUNDS) ys.size =
      ys.extract (sh.ell - 6) (sh.ell - 6 + (K - sh.ell)) := by rw [hk, hy]; congr 1; omega
  have hle : rng.pos0 + rng.count ≤ 2 ^ (K - sh.ell) := by
    have hle : rng.pos0 + rng.count ≤
        min (eqTable (xs.extract (sh.ell - RecSparse.PRE_ROUNDS) xs.size)).size
          (eqTable (ys.extract (sh.ell - RecSparse.PRE_ROUNDS) ys.size)).size := by
      by_contra hn
      exact hc (by simp [Nat.lt_of_not_le hn])
    rw [hxe, hye] at hle
    simp only [FlockLevel3.eqTable_size, Array.size_extract, hx, hy] at hle
    rwa [show min (sh.ell - 6 + (K - sh.ell)) (K - 6) - (sh.ell - 6) = K - sh.ell by omega, min_self] at hle
  have eqF : ∀ (r : Array Flock.F128), r.size = K - 6 → ∀ b < 2 ^ (K - sh.ell),
      fOf (eqTable (r.extract (sh.ell - 6) (sh.ell - 6 + (K - sh.ell))))[b]! =
        Model.eqAt (fun i : Fin (K - sh.ell) => fOf r[sh.ell - 6 + i.val]!) b := fun r hr b hb => by
    show FlockLevel3.GF128.of (eqTable (r.extract (sh.ell - 6) (sh.ell - 6 + (K - sh.ell))))[b]! = _
    rw [eqTable_extract r _ _ (by omega) b hb, Finset.prod_range]
    rfl
  have hσS : σ.val < sh.S := σ.isLt
  rw [hunits σ]
  by_cases hσ : σ.val < rng.count
  · rw [ite_eq_left hσ, Finset.filter_singleton, ite_eq_left (by omega), Finset.sum_singleton,
      getElem!_pos _ _ (by simp; omega), Array.getElem_append_left (by simp; omega)]
    simp only [Array.getElem_map, Array.getElem_range]
    rw [fOf_mul, hxe, hye, eqF xs hx _ (by omega), eqF ys hy _ (by omega)]
  · rw [ite_eq_right hσ, Finset.filter_empty, Finset.sum_empty, getElem!_pos _ _ (by simp; omega),
      Array.getElem_append_right (by simp; omega)]
    simp only [Array.getElem_replicate]
    exact fOf_zero

section Unit

variable {S : Model.Statement} {sch₀ : Accounting.Schedule} {extra : List (Model.Claim 𝔽 S.m)}
  {cap₀ : Model.Cap Refine.Dig} {os : Model.Opens 𝔽 Refine.Dig} {r : ℕ} {ts : List Game.Step}

/-- **A rep's unit issues coins at every round below `z_partial`'s**: the zerocheck's (`n₀` … `nr`) and the
lincheck's (`HoloLcAnswered`), each of `n` coins (`RoundsWF`), inside the record (`zp_lt`). -/
theorem hpos_of_unit (U : RepUnit S sch₀ extra cap₀ os r ts) :
    ∀ j < U.t.k + 3 + (S.m - 6) + 2 + (S.kLog - 6),
      j < U.t.rounds.size ∧ decide ((U.t.rounds[j]!).coins.size ≠ 0) = true := by
  have h22 := Refine.exec_fast100_ge U.fast
  have hm := U.mS
  have hk := U.fold.kLog
  have k0 := U.k0
  have hlt : U.t.k + 3 + (U.st.m - 6) + 2 + (U.st.fold.kLog - 6) < U.t.rounds.size := U.lc.zp_lt
  intro j hj
  have hj' : j < U.t.rounds.size := by omega
  refine ⟨hj', decide_eq_true ?_⟩
  rw [U.wf j hj']
  rw [k0] at hj hlt
  by_cases a0 : j = 0
  · subst a0; have := U.n₀; rw [k0] at this; omega
  by_cases a1 : j = 1
  · subst a1; have := U.n₁; rw [k0, Nat.zero_add] at this; omega
  by_cases a2 : j = 2
  · subst a2; have := U.n₂; rw [k0, Nat.zero_add] at this; omega
  by_cases a3 : j < 3 + (U.st.m - 6)
  · have := U.nr (j - 3) (by omega)
    rw [k0, show 0 + 3 + (j - 3) = j by omega] at this; omega
  by_cases a4 : j = 3 + (U.st.m - 6)
  · have e : (U.t.rounds[U.t.k + 3 + (U.st.m - 6)]!).n = 1 := U.lc.alpha.1
    rw [k0, Nat.zero_add, ← a4] at e; omega
  by_cases a5 : j = 3 + (U.st.m - 6) + 1
  · have e : (U.t.rounds[U.t.k + 3 + (U.st.m - 6) + 1]!).n = 1 := U.lc.beta.1
    rw [k0, Nat.zero_add, ← a5] at e; omega
  · have e : (U.t.rounds[U.t.k + 3 + (U.st.m - 6) + 2 + (j - (3 + (U.st.m - 6) + 2))]!).n = 1 :=
      (U.lc.rounds (j - (3 + (U.st.m - 6) + 2)) (by omega)).1
    rw [k0, Nat.zero_add, show 3 + (U.st.m - 6) + 2 + (j - (3 + (U.st.m - 6) + 2)) = j by omega] at e; omega

/-- **The sparse public values at check 5's points are the rep's outputs'**: at a rep's unit, where `verifierRows`
accepts the unit's rounds at a shape of the statement's `m` and `k_log` (any extra claims and comb, as check 5's
`Inner.rep` runs it), and `c`'s `z`, `alpha`, `x_rest` and `y_rest` are the points it returns, the unit's play's
outputs have `pubOf c` for `holoPub`'s values, wherever `c`'s `U` is `holoPub`'s sum at `c`'s coordinates. -/
theorem pub_of_unit (hm : 13 ≤ S.m) (U : RepUnit S sch₀ extra cap₀ os r ts) {ash : Flock.Shape}
    {xs : Array Flock.Claim} {cb v : Array Flock.F128} {info : RecAlgebra.Info RecAlgebra.Lin}
    {kappa : Array (Array (Array Flock.F128))} (hrows : verifierRows ash U.d.rounds xs cb = .ok (v, info, kappa))
    (ham : ash.m = S.m) (hak : ash.kLog = S.kLog) {sh : SparseShape} (h6 : 6 ≤ sh.ell) (hell : sh.ell ≤ S.kLog)
    (units : Fin (2 ^ sh.sLog) → Finset ℕ) {c : SCoins} (hcz : c.z = info.zc.z) (hca : c.alpha = info.lc.fold.alpha)
    (hcx : c.xRest = info.lc.fold.rhoIn) (hcy : c.yRest = info.lc.rIn)
    (hU : ∀ σ : Fin (2 ^ sh.sLog), fOf c.U[σ.val]! = ∑ u ∈ units σ,
      Model.eqAt (fun i : Fin (S.kLog - sh.ell) => fOf c.xRest[sh.ell - 6 + i.val]!) u *
        Model.eqAt (fun i : Fin (S.kLog - sh.ell) => fOf c.yRest[sh.ell - 6 + i.val]!) u) :
    ∃ zc lc pend, Game.PlaysT (Model.holoRepC (D := Refine.Dig) execArith r S sch₀ hm extra) ts
        (some (zc, lc), pend) ∧ pend.verdict os = true ∧
      (∀ sl, Model.OpensOKL H512 Refine.enc512 Refine.leafPlain cap₀ r pend os sl) ∧
      pubOf sh c = Model.holoPub S sh units zc lc := by
  obtain ⟨zc, lc, pend, hT, hv, hz, hρ, hα, hts, ho⟩ := rep_of_unit_coins hm U
  have h22 := Refine.exec_fast100_ge U.fast
  have hmS := U.mS
  have hSk := S.hk
  obtain ⟨-, hz', hrho, hts', hα', hrin, hrIn⟩ := holoRows_coins hrows (by omega) U.k0 U.coins
    (by rw [ham, hak]; exact hpos_of_unit U)
  rw [ham] at hrho hts' hα'
  rw [hak] at hrin hts'
  refine ⟨zc, lc, pend, hT, hv, ho, pubOf_eq_holoPub h6 hell (by rw [hcz, hz', hz]) (by rw [hca, hα', hα])
    (fun j hj => ?_) (fun j hj => ?_) hU⟩
  · rw [hcx, hrin, hrho, FlockLevel3.extract_get! _ 0 _ j (by omega) (by simp [rhoOf]; omega), Nat.zero_add,
      rhoOf, map_range_get _ _ (by omega), hρ ⟨j, _⟩]
  · have hT : (tsOf U.t (U.t.k + 3 + (S.m - 6)) S.kLog).size = S.kLog - 6 := by simp [tsOf]
    rw [hcy, hrIn, hts', getElem!_pos _ j (by rw [Array.size_reverse, hT]; omega), Array.getElem_reverse,
      ← getElem!_pos (tsOf U.t (U.t.k + 3 + (S.m - 6)) S.kLog) _ (by rw [hT]; omega), hT, tsOf,
      map_range_get _ _ (by omega), hts ⟨S.kLog - 6 - 1 - j, by omega⟩]

end Unit

end

end FlockC3.Sparse

#print axioms FlockSoundness.Refine.zerocheck_refines_coins
#print axioms FlockSoundness.Refine.holoLincheck_refines_coins
#print axioms FlockSoundness.Refine.holoRep_refines_coins
#print axioms FlockSoundness.Refine.holoRep_refines
#print axioms FlockVBridge.Runs.rows_coins
#print axioms FlockC3.HoloValues.holoRows_coins
#print axioms FlockC3.HoloValues.rep_of_unit_coins
#print axioms FlockC3.Sparse.pubOf_eq_holoPub
#print axioms FlockC3.Sparse.layoutFactors_U
#print axioms FlockC3.Sparse.hpos_of_unit
#print axioms FlockC3.Sparse.pub_of_unit
