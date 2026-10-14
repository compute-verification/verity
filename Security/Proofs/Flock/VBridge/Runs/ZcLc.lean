import Proofs.Flock.VBridge.Compiled
import Proofs.Flock.Soundness.Refine.RepConverse

/-!
# `Runs`' zerocheck and lincheck checks from E

`zclc_of_residuals` gives `Refine.prog_of_checks`' zerocheck checks (`hpc`, `hfold`) and lincheck checks (`hlfold`,
`hcomb`, `hsize`) at a rep whose values meet `AlgebraHolds` and `AccHolds`, from E's `zerocheck_of_residuals` and
`lincheck_of_residuals`, on a transcript from round 0 whose rounds' coins are the rep's. The points are the
transcript's coins: `r`, `z`, `ρ = rhoOf`, the residual rounds' coins, and from round `k₁ = 3 + (m − 6)` the lincheck's
`α`, `β` and challenges `ts = tsOf`. `zclc_of_residuals_dec` gives the same checks at either shape, the lincheck's
rounds ending at the comb the rep holds (`decoded`, which at a holo shape adds w's `h`); `zclc_of_residuals` is it at a
plain shape, where that comb is `d.comb`.

`rows_coins` is what makes the two meet: the points `verifierRows` returns (`info`, read by `Coins.ofRounds` from the
rounds' coins) are those coins (rec_algebra.py:587–617, `RecAlgebra.zerocheck` and `RecAlgebra.lincheck`).
-/

namespace FlockVBridge.Runs

open Flock
open FlockSoundness.Refine (lStep LcAnswered RsRounds repProg zcOf lcOf rsT rsSws prog_of_checks encs wOf)
open Flock.RecAlgebra
open FlockVBridge.Compiled (RepData AlgebraHolds AccHolds)

/-- The zerocheck's residual-round challenges on `t` (Piop.lean:90–98): round `t.k + 3 + i`'s first coin. -/
def rhoOf (t : Transcript) (m : ℕ) : Array F128 :=
  (Array.range (m - 6)).map fun i => (t.rounds[t.k + 3 + i]!).coins.getD 0 0

/-- The lincheck's round challenges on `t` from round `k₁` (Piop.lean:134–142): round `k₁ + 2 + j`'s first coin. -/
def tsOf (t : Transcript) (k₁ kLog : ℕ) : Array F128 :=
  (Array.range (kLog - 6)).map fun j => (t.rounds[k₁ + 2 + j]!).coins.getD 0 0

theorem map_range_get (n : ℕ) (f : ℕ → F128) {i : ℕ} (hi : i < n) : ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_pos _ i (by simpa using hi)]
  simp

/-! ## The coins `verifierRows` reads -/

theorem coin_of_round {R : Array Round} {k n : ℕ} {c : Array F128} (hne : NoEmpty (R.map (·.coins)))
    (h : (Coins.ofRounds (R.map (·.coins))).round k n = .ok c) : (R[k]!).coins = c ∧ c.size = n := by
  have h' : (R.map (·.coins))[k]? = some c ∧ c.size = n := by
    simp only [Coins.ofRounds, filter_noEmpty hne] at h
    split at h
    next r hr =>
      split at h
      next hs => cases h; exact ⟨hr, hs⟩
      next => cases h
    next => cases h
  obtain ⟨h1, hn⟩ := h'
  obtain ⟨hk, he⟩ := Array.getElem?_eq_some_iff.1 h1
  rw [Array.size_map] at hk
  refine ⟨?_, hn⟩
  rw [getElem!_pos R k hk, ← he, Array.getElem_map]

theorem coins_at {R : Array Round} {k k' n : ℕ} {c : Array F128} (hne : NoEmpty (R.map (·.coins)))
    (h : (Coins.ofRounds (R.map (·.coins))).round k n = .ok c) (hk : k = k') : (R[k']!).coins = c := by
  subst hk
  exact (coin_of_round hne h).1

theorem coin_one {R : Array Round} {k k' : ℕ} {c : Array F128} (hne : NoEmpty (R.map (·.coins)))
    (h : (Coins.ofRounds (R.map (·.coins))).round k 1 = .ok c) (hk : k = k') :
    (R[k']!).coins.getD 0 0 = c[0]! := by
  subst hk
  obtain ⟨hc, hn⟩ := coin_of_round hne h
  rw [hc, Array.getD_eq_getD_getElem?, getElem!_pos c 0 (by omega), Array.getElem?_eq_getElem (by omega),
    Option.getD_some]

/-- `verifierRows`' points are those of `run` on the symbolic messages at the rounds' coins. -/
theorem run_of_rows {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128}
    {info : Info Lin} {kappa : Array (Array (Array F128))}
    (h : verifierRows sh rounds extras comb = .ok (v, info, kappa)) :
    ∃ lins k, run sh (symbolic sh) extras 0 (Coins.ofRounds rounds) = .ok ((lins, info), k) := by
  unfold verifierRows at h
  cases hr : run sh (symbolic sh) extras 0 (Coins.ofRounds rounds) with
  | error e =>
    have hl : residualLins sh (Coins.ofRounds rounds) extras = .error e := by
      show Prod.fst <$> run sh (symbolic sh) extras 0 (Coins.ofRounds rounds) = _
      rw [hr]; rfl
    rw [hl] at h; cases h
  | ok q =>
    obtain ⟨⟨lins, info'⟩, k⟩ := q
    have hl : residualLins sh (Coins.ofRounds rounds) extras = .ok (lins, info') := by
      show Prod.fst <$> run sh (symbolic sh) extras 0 (Coins.ofRounds rounds) = _
      rw [hr]; rfl
    rw [hl] at h
    simp only [bind, Except.bind] at h
    cases hb : build sh lins with
    | error e => rw [hb] at h; cases h
    | ok b =>
      rw [hb] at h
      simp only at h
      split at h
      · cases h
      · cases h
        exact ⟨lins, k, rfl⟩

/-- **The coins `verifierRows` reads.** At rounds whose coins are a transcript's from round 0, the points
`verifierRows` returns are the transcript's coins: the zerocheck's `r` (`pinned` and round 1), `z` (round 2) and `ρ`
(rounds `3 + i`), and from round `k₁ = 3 + (m − 6)` the lincheck's `α`, `β`, challenges `ts` (rounds `k₁ + 2 + j`) and
`r'` (round `k₁ + 2 + (k_log − 6)`), with the fold's skip weights and `ρ_in` at the zerocheck's points
(rec_algebra.py:587–617). -/
theorem rows_coins {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128}
    {info : Info Lin} {kappa : Array (Array (Array F128))}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hm : 14 ≤ sh.m) {t : Transcript}
    (ht : t.k = 0) (hd : rounds = t.rounds.map (·.coins)) (hne : NoEmpty rounds) :
    info.zc.r = pinned ++ (t.rounds[t.k + 1]!).coins ∧ info.zc.z = (t.rounds[t.k + 2]!).coins.getD 0 0 ∧
      info.zc.rho = rhoOf t sh.m ∧
      info.lc.fold.alpha = (t.rounds[t.k + 3 + (sh.m - 6)]!).coins.getD 0 0 ∧
      info.lc.fold.beta = (t.rounds[t.k + 3 + (sh.m - 6) + 1]!).coins.getD 0 0 ∧
      info.lc.fold.lw = skipWeights info.zc.z ∧ info.lc.fold.rhoIn = info.zc.rho.extract 0 (sh.kLog - 6) ∧
      info.lc.fold.ts = tsOf t (t.k + 3 + (sh.m - 6)) sh.kLog ∧
      info.lc.rPrime = (t.rounds[t.k + 3 + (sh.m - 6) + 2 + (sh.kLog - 6)]!).coins.getD 0 0 := by
  subst hd
  obtain ⟨lins, k, hrun⟩ := run_of_rows hrows
  obtain ⟨-, _, _, _, _, k1, _, _, h0, h1, -, -, -⟩ := run_stages hrun
  obtain ⟨c₀, c₁, c₂, -, hc₁, hc₂, hr, hz, hrs, hk1, hrho⟩ := zerocheck_coins hm h0
  obtain ⟨cα, cβ, cr, hcα, hcβ, hcr, hα, hβ, hrp, hlw, hrin, -, -, htsz, -, hts⟩ := lincheck_coins h1
  have hzs : (symbolic sh).zc.size = sh.m - 6 := by simp [symbolic]
  have hls : (symbolic sh).lc.size = sh.kLog - 6 := by simp [symbolic]
  rw [hzs] at hrs hk1 hrho
  rw [hls] at hcr htsz hts
  subst hk1
  refine ⟨?_, ?_, ?_, ?_, ?_, hlw, hrin, ?_, ?_⟩
  · rw [hr, coins_at hne (k' := t.k + 1) hc₁ (by omega)]
  · rw [hz, coin_one hne (k' := t.k + 2) hc₂ (by omega)]
  · apply Array.ext
    · rw [hrs]; simp [rhoOf]
    · intro j hj₁ hj₂
      obtain ⟨c, hc, he⟩ := hrho j (by omega)
      rw [← getElem!_pos info.zc.rho j hj₁, ← getElem!_pos (rhoOf t sh.m) j hj₂, he, rhoOf,
        map_range_get (sh.m - 6) _ (i := j) (by omega)]
      exact (coin_one hne (k' := t.k + 3 + j) hc (by omega)).symm
  · rw [hα, coin_one hne (k' := t.k + 3 + (sh.m - 6)) hcα (by omega)]
  · rw [hβ, coin_one hne (k' := t.k + 3 + (sh.m - 6) + 1) hcβ (by omega)]
  · apply Array.ext
    · rw [htsz]; simp [tsOf]
    · intro j hj₁ hj₂
      obtain ⟨c, hc, he⟩ := hts j (by omega)
      rw [← getElem!_pos info.lc.fold.ts j hj₁, ← getElem!_pos (tsOf t (t.k + 3 + (sh.m - 6)) sh.kLog) j hj₂, he,
        tsOf, map_range_get (sh.kLog - 6) _ (i := j) (by omega)]
      exact (coin_one hne (k' := t.k + 3 + (sh.m - 6) + 2 + j) hc (by omega)).symm
  · rw [hrp, coin_one hne (k' := t.k + 3 + (sh.m - 6) + 2 + (sh.kLog - 6)) hcr (by omega)]

/-! ## The checks -/

/-- **`prog_of_checks`' zerocheck and lincheck checks from E, at either shape.** At a rep `d` meeting E's preconditions
(`AlgebraHolds`, `AccHolds`), on a transcript `t` from round 0 whose rounds' coins are the rep's, with the setup's
`m` and fold's `k_log` the shape's, a proof whose zerocheck and lincheck messages are those `d` holds (`ZcProofOf`,
`LcProofOf`, with V*'s `v_C` for `final_c`) and whose comb is the circuit fold at `verifierRows`' fold points passes
`prog_of_checks`' checks `hpc`, `hfold` (Piop.lean:84, 98), `hlfold`, `hcomb`, `hsize` (Piop.lean:146–148) at the
transcript's coins: `r = pinned ++ coins(1)`, `z = coins(2)[0]`, `ρ = rhoOf t m`, `C i = coins(3 + i)`, from
`k₁ = t.k + 3 + (m − 6)` the challenges `ts = tsOf t k₁ k_log`, and `comb = d.comb`; the lincheck's rounds end at the
comb `d` holds (`decoded`), which at a holo shape adds w's `h` (`decoded_comb_get`). -/
theorem zclc_of_residuals_dec (sh : Shape) (d : RepData) (halg : AlgebraHolds sh d) (hacc : AccHolds sh d)
    (st : Setup) (p : Proof) (t : Transcript) (ht : t.k = 0) (hd : d.rounds = t.rounds.map (·.coins))
    (hne : NoEmpty d.rounds) (hm : st.m = sh.m) (hm14 : 14 ≤ st.m) (hkL : st.fold.kLog = sh.kLog)
    (hzp : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) p.zerocheck)
    (hlp : LcProofOf (decoded sh d.w d.comb d.cons) p.lincheck)
    (hfoldPts : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
        .ok d.comb) :
    dot (lagrange nodesL den6Inv ((t.rounds[t.k + 2]!).coins.getD 0 0)) p.zerocheck.round1c = p.zerocheck.finalC ∧
      zcFold (pinned ++ (t.rounds[t.k + 1]!).coins) p.zerocheck (fun i => (t.rounds[t.k + 3 + i]!).coins) (st.m - 6)
          (zcStart p.zerocheck ((t.rounds[t.k + 2]!).coins.getD 0 0)) =
        (p.zerocheck.finalA * p.zerocheck.finalB, rhoOf t st.m) ∧
      (List.range' 0 (st.fold.kLog - 6)).foldl
          (fun s j => lStep p.lincheck.rounds j s (t.rounds[t.k + 3 + (st.m - 6) + 2 + j]!).coins)
          ((t.rounds[t.k + 3 + (st.m - 6)]!).coins.getD 0 0 * p.zerocheck.finalA + p.zerocheck.finalB +
            (t.rounds[t.k + 3 + (st.m - 6) + 1]!).coins.getD 0 0, #[]) =
        (dot (decoded sh d.w d.comb d.cons).comb p.lincheck.zPartial, tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog) ∧
      st.fold.folded ((t.rounds[t.k + 3 + (st.m - 6)]!).coins.getD 0 0)
          ((t.rounds[t.k + 3 + (st.m - 6) + 1]!).coins.getD 0 0) (skipWeights ((t.rounds[t.k + 2]!).coins.getD 0 0))
          ((rhoOf t st.m).extract 0 (st.fold.kLog - 6)) (tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog) = .ok d.comb ∧
      d.comb.size = 64 := by
  obtain ⟨v, info, kappa, S, rs, hrows, hS, hw, hc, hres, hzero⟩ := halg
  have hacc' := hacc v info kappa hrows
  have hzp' := hzp v info kappa hrows
  obtain ⟨hr, hz, hrho, hα, hβ, hlw, hrin, hts, -⟩ := rows_coins hrows (by omega) ht hd hne
  obtain ⟨hpc, hzf⟩ := zerocheck_of_residuals hrows hS hw hc hacc' hres hzero hzp'
    (C := fun i => (t.rounds[t.k + 3 + i]!).coins) (fun i hi => by
      rw [hrho, rhoOf, map_range_get _ _ hi])
  have hlf := lincheck_of_residuals_dec hrows hS hw hc hacc' hres hzero hlp
    (C := fun j => (t.rounds[t.k + 3 + (sh.m - 6) + 2 + j]!).coins) (fun j hj => by
      rw [hts, tsOf, map_range_get _ _ hj])
  rw [hm, hkL]
  refine ⟨?_, ?_, ?_, ?_, hc⟩
  · rw [← hz]; exact hpc
  · rw [← hr, ← hz, ← hrho]; exact hzf
  · rw [hzp'.fa, hzp'.fb, ← hα, ← hβ, ← hts]; exact hlf
  · rw [← hα, ← hβ, ← hz, ← hrho, ← hts, ← hlw, ← hrin]; exact hfoldPts v info kappa hrows

/-- **`prog_of_checks`' zerocheck and lincheck checks from E**, at a plain shape: `zclc_of_residuals_dec`, where the
comb `d` holds is `d.comb` (`decoded_comb`). -/
theorem zclc_of_residuals (sh : Shape) (d : RepData) (halg : AlgebraHolds sh d) (hacc : AccHolds sh d)
    (st : Setup) (p : Proof) (t : Transcript) (ht : t.k = 0) (hd : d.rounds = t.rounds.map (·.coins))
    (hne : NoEmpty d.rounds) (hm : st.m = sh.m) (hm14 : 14 ≤ st.m) (hkL : st.fold.kLog = sh.kLog)
    (hzp : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) p.zerocheck)
    (hlp : LcProofOf (decoded sh d.w d.comb d.cons) p.lincheck) (hpl : sh.holo = false)
    (hfoldPts : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
        .ok d.comb) :
    dot (lagrange nodesL den6Inv ((t.rounds[t.k + 2]!).coins.getD 0 0)) p.zerocheck.round1c = p.zerocheck.finalC ∧
      zcFold (pinned ++ (t.rounds[t.k + 1]!).coins) p.zerocheck (fun i => (t.rounds[t.k + 3 + i]!).coins) (st.m - 6)
          (zcStart p.zerocheck ((t.rounds[t.k + 2]!).coins.getD 0 0)) =
        (p.zerocheck.finalA * p.zerocheck.finalB, rhoOf t st.m) ∧
      (List.range' 0 (st.fold.kLog - 6)).foldl
          (fun s j => lStep p.lincheck.rounds j s (t.rounds[t.k + 3 + (st.m - 6) + 2 + j]!).coins)
          ((t.rounds[t.k + 3 + (st.m - 6)]!).coins.getD 0 0 * p.zerocheck.finalA + p.zerocheck.finalB +
            (t.rounds[t.k + 3 + (st.m - 6) + 1]!).coins.getD 0 0, #[]) =
        (dot d.comb p.lincheck.zPartial, tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog) ∧
      st.fold.folded ((t.rounds[t.k + 3 + (st.m - 6)]!).coins.getD 0 0)
          ((t.rounds[t.k + 3 + (st.m - 6) + 1]!).coins.getD 0 0) (skipWeights ((t.rounds[t.k + 2]!).coins.getD 0 0))
          ((rhoOf t st.m).extract 0 (st.fold.kLog - 6)) (tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog) = .ok d.comb ∧
      d.comb.size = 64 := by
  have h := zclc_of_residuals_dec sh d halg hacc st p t ht hd hne hm hm14 hkL hzp hlp hfoldPts
  have hc := h.2.2.2.2
  rwa [decoded_comb sh d.cons hpl hc] at h

/-- **The rep's program from E's zerocheck and lincheck.** `prog_of_checks` with its zerocheck and lincheck checks from
`zclc_of_residuals`, at the transcript's coins; the frames, counts, ring switching and Ligerito stay as
`prog_of_checks` takes them, at those points. -/
theorem prog_of_zclc (sh : Shape) (d : RepData) (halg : AlgebraHolds sh d) (hacc : AccHolds sh d)
    (st : Setup) (sch : Flock.Schedule) (p : Proof) (extras : Array Claim) (t : Transcript) (ht : t.k = 0)
    (hd : d.rounds = t.rounds.map (·.coins)) (hne : NoEmpty d.rounds) (hm : st.m = sh.m) (hm14 : 14 ≤ st.m)
    (hkL : st.fold.kLog = sh.kLog) (hpl : sh.holo = false)
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
    -- the lincheck's frames and counts
    (hlc : LcAnswered ⟨t.rounds, t.k + 3 + (st.m - 6),
      ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck)
    -- ring switching, at the transcript's points
    {claims : Array Claim}
    (hcl : claims = Flock.abcClaims st.m st.kLog
      (zcOf p.zerocheck (pinned ++ (t.rounds[t.k + 1]!).coins) ((t.rounds[t.k + 2]!).coins.getD 0 0) (rhoOf t st.m))
      (lcOf p.lincheck ((t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6)]!).coins.getD 0 0)
        (tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog)) ++ extras)
    {rpp : ℕ → Array F128} {gamma : Array F128} (hrs : p.ringSwitches.size = claims.size)
    (hshape : ∀ k < claims.size,
      p.ringSwitches[k]!.size = 128 ∧ claims[k]!.skip.size = 64 ∧ claims[k]!.x.size = st.m - 6)
    (hval : ∀ k < claims.size, dot (wOf claims[k]!) p.ringSwitches[k]! = claims[k]!.value)
    (hR : RsRounds ⟨t.rounds, t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1, ByteArray.empty⟩ p.ringSwitches
      claims.size rpp gamma)
    -- Ligerito
    {t₄ : Transcript}
    (hlg : Flock.ligerito st.merkle sch (rsT claims p.ringSwitches rpp gamma) p.cap p.ligerito
      (Flock.basis (rsSws claims rpp gamma))
      ⟨t.rounds, t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + claims.size + 1, ByteArray.empty⟩ =
        .ok ((), t₄)) :
    repProg st sch p extras t = .ok ((), t₄) := by
  obtain ⟨hpc, hfold, hlfold, hcomb, hsize⟩ :=
    zclc_of_residuals sh d halg hacc st p t ht hd hne hm hm14 hkL hzp hlp hpl hfoldPts
  exact prog_of_checks st sch p extras t hm14 hn₀ hn₁ hn₂ hnr ha₀ ha₁ ha₂ har (r := _) (z := _)
    (rho := rhoOf t st.m) (C := fun i => (t.rounds[t.k + 3 + i]!).coins) rfl rfl (fun _ _ => rfl) hpc hfold
    (k₁ := t.k + 3 + (st.m - 6)) rfl hlc (ts := tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog) (comb := d.comb)
    hlfold hcomb hsize (k₂ := t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1) rfl hcl hrs hshape hval hR hlg

end FlockVBridge.Runs
