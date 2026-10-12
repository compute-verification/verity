import Proofs.Flock.VBridge.Compiled
import Proofs.Flock.VBridge.Algebra.RingSwitch
import Proofs.Flock.Soundness.Refine.RepConverse

/-!
# Ring switching's checks at a rep, from E

`rs_of_residuals` gives `Refine.prog_of_checks`' ring-switching hypotheses at a rep from E's
`ringSwitch_of_residuals`, when the rep's coins are its stream's (`d.rounds` the stream's rounds' coins) and the proof
carries the messages w holds (`decoded`):

* the claims ring switching reads (Verify.lean:49) are `claimsF`, V*'s claims at the rep's values, on the zerocheck's
  and the lincheck's points `verifierRows` returns, the lincheck's `r'` the coin of its stream round;
* the ring switches have the verifier's shape and pass its value check (Opening.lean:76–80);
* ring switching's rounds issued the coins `verifierRows` read: claim `k`'s `r''_k`, then the batching coins `γ`
  (Opening.lean:81, 87), from `ringSwitch_spec` and `Coins.ofRounds`. The rounds' coin counts and the bytes they answer
  are hypotheses, facts about the record V* holds;
* the batched value `T` (Opening.lean:88) is the one w holds, and the switched claims (Opening.lean:89) are
  `verifierRows`', the two Ligerito reads.
-/

namespace FlockVBridge.Runs

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockVBridge.Compiled

/-- `Coins.ofRounds` on a record's coins, none of its rounds empty, reads round `k`'s, when it has `n`
(RecAlgebra.lean:40–46). -/
theorem ofRounds_round {rs : Array Round} {k n : ℕ} {c : Array F128} (hne : NoEmpty (rs.map (·.coins)))
    (h : (Coins.ofRounds (rs.map (·.coins))).round k n = .ok c) : k < rs.size ∧ (rs[k]!).coins = c ∧ c.size = n := by
  by_cases hk : k < rs.size
  · have e : (rs.map (·.coins))[k]? = some rs[k].coins := by simp [hk]
    simp only [Coins.ofRounds, filter_noEmpty hne, e] at h
    by_cases hs : rs[k].coins.size = n
    · simp only [hs, ↓reduceIte] at h
      cases h
      exact ⟨hk, by rw [getElem!_pos rs k hk], hs⟩
    · simp [hs] at h
  · have e : (rs.map (·.coins))[k]? = none := by simp [hk]
    simp only [Coins.ofRounds, filter_noEmpty hne, e] at h
    cases h

theorem getD_zero_of_size {c : Array F128} (h : c.size = 1) : c.getD 0 0 = c[0]! := by
  rw [getElem!_pos c 0 (by omega)]
  simp [Array.getD, h]

theorem pair_append_get {P : Claim → Prop} {a b : Claim} {extras : Array Claim} (ha : P a) (hb : P b)
    (he : ∀ j < extras.size, P extras[j]!) : ∀ k < 2 + extras.size, P (#[a, b] ++ extras)[k]! := by
  intro k hk
  rw [getElem!_pos _ k (by simp; omega)]
  match k, hk with
  | 0, _ => rw [Array.getElem_append_left (by simp)]; exact ha
  | 1, _ => rw [Array.getElem_append_left (by simp)]; exact hb
  | j + 2, hk =>
    have e := he j (by omega)
    rw [getElem!_pos extras j (by omega)] at e
    rw [Array.getElem_append_right (by simp)]
    simpa using e

/-- **Ring switching's checks at a rep** (Verify.lean:49, Opening.lean:66–90). When E's preconditions hold at the
rep's values `d` (`ringSwitch_of_residuals`), the rep's coins are its stream's (`t.rounds`, from round 0), and the
proof carries the messages w holds (V*'s `v_C` for `final_c`, w's `z_partial` and ring switches), then at the
zerocheck's and the lincheck's points `verifierRows` returns, `r''_k` and `γ` its ring switching's coins, and the
claims `claimsF`:
* the claims are the verifier's (`abcClaims` at the zerocheck's and the lincheck's outputs, the lincheck's `r'` its
  stream round's coin) followed by the extras;
* there is one ring switch per claim, each of 128 words, and each claim has 64 skip weights and `m − 6` coordinates
  (the extras' shapes are a hypothesis);
* each claim passes the value check on its ring switch;
* the ring-switching rounds from `k₂` issued `r''_k` and `γ`, answering the bytes framed before them (hypotheses);
* the batched sum is the `T` w holds, and the switched claims are `verifierRows`'. -/
theorem rs_of_residuals {sh : Shape} {d : RepData} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {res : Array F128}
    (hrows : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64)
    (hacc : ∀ r < 2, valOf sh d.w d.comb d.cons (.acc1 r) + valOf sh d.w d.comb d.cons (.acc0 r) =
      kapSum kappa r d.cons)
    (hres : residuals S d.w v = .ok res) (hzero : ∀ x ∈ res, x = 0)
    {st : Setup} {p : Proof} {extras : Array Claim} {t : Transcript}
    (hm : 14 ≤ st.m) (hmS : st.m = sh.m) (hkS : st.kLog = sh.kLog) (hfS : st.fold.kLog = sh.kLog)
    (hkm : st.kLog ≤ st.m) (hxs : d.extras = extras)
    (hext : ∀ j < extras.size, extras[j]!.skip.size = 64 ∧ extras[j]!.x.size = st.m - 6)
    (hco : d.rounds = t.rounds.map (·.coins)) (hne : NoEmpty d.rounds) (ht : t.k = 0) (hpl : sh.holo = false)
    (hfc : p.zerocheck.finalC = info.zc.vc.eval (valOf sh d.w d.comb d.cons))
    (hzp : p.lincheck.zPartial = (decoded sh d.w d.comb d.cons).zp)
    (hrsw : p.ringSwitches = (decoded sh d.w d.comb d.cons).rs)
    {k₁ k₂ : ℕ} (hk₁ : k₁ = t.k + 3 + (st.m - 6)) (hk₂ : k₂ = k₁ + 2 + (st.fold.kLog - 6) + 1)
    (hrn : ∀ k < 2 + extras.size, (t.rounds[k₂ + k]!).n = 7)
    (hra : ∀ k < 2 + extras.size,
      (t.rounds[k₂ + k]!).answers (rsFramed (rsPending ByteArray.empty k) p.ringSwitches[k]!))
    (hgn : (t.rounds[k₂ + (2 + extras.size)]!).n = 2 + extras.size)
    (hga : (t.rounds[k₂ + (2 + extras.size)]!).answers (ByteArray.empty ++ Transcript.header 4 2 (2 + extras.size))) :
    claimsF sh (valOf sh d.w d.comb d.cons) info extras =
        abcClaims st.m st.kLog (zcOf p.zerocheck info.zc.r info.zc.z info.zc.rho)
          (lcOf p.lincheck ((t.rounds[k₁ + 2 + (st.fold.kLog - 6)]!).coins.getD 0 0) info.lc.fold.ts) ++ extras ∧
      p.ringSwitches.size = (claimsF sh (valOf sh d.w d.comb d.cons) info extras).size ∧
      (∀ k < (claimsF sh (valOf sh d.w d.comb d.cons) info extras).size,
        p.ringSwitches[k]!.size = 128 ∧ (claimsF sh (valOf sh d.w d.comb d.cons) info extras)[k]!.skip.size = 64 ∧
          (claimsF sh (valOf sh d.w d.comb d.cons) info extras)[k]!.x.size = st.m - 6) ∧
      (∀ k < (claimsF sh (valOf sh d.w d.comb d.cons) info extras).size,
        dot (wOf (claimsF sh (valOf sh d.w d.comb d.cons) info extras)[k]!) p.ringSwitches[k]! =
          (claimsF sh (valOf sh d.w d.comb d.cons) info extras)[k]!.value) ∧
      RsRounds ⟨t.rounds, k₂, ByteArray.empty⟩ p.ringSwitches
        (claimsF sh (valOf sh d.w d.comb d.cons) info extras).size (fun k => (info.rs.sws[k]!).rpp) info.rs.gamma ∧
      rsT (claimsF sh (valOf sh d.w d.comb d.cons) info extras) p.ringSwitches (fun k => (info.rs.sws[k]!).rpp)
          info.rs.gamma = valOf sh d.w d.comb d.cons .T ∧
      rsSws (claimsF sh (valOf sh d.w d.comb d.cons) info extras) (fun k => (info.rs.sws[k]!).rpp) info.rs.gamma =
        info.rs.sws := by
  subst hxs
  obtain ⟨hcl0, hval, hT⟩ := ringSwitch_of_residuals hpl hrows hS hw hc hacc hres hzero
  obtain ⟨rs', k, hrun, hres'⟩ := residuals_eq_run d.w d.comb d.cons hpl hrows hS hw hc hacc
  rw [hres] at hres'
  cases hres'
  rw [hco] at hrun hne
  obtain ⟨hsc, r0, r1, rsr, lgr, k1, k2, k3, h0, h1, h2, -, -⟩ := run_stages hrun
  obtain ⟨-, -, hszc⟩ := decoded_sizes sh d.w d.comb d.cons
  have hslc := decoded_lc_size sh d.w d.comb d.cons
  obtain ⟨hsrs, hsrs'⟩ := decoded_rs_size sh d.w d.comb d.cons
  obtain ⟨_, c₁, _, -, hc₁, -, hr, -, hrho, hk1, -⟩ := zerocheck_coins (by omega : 14 ≤ sh.m) h0
  obtain ⟨_, _, cr, -, -, hcr, -, -, hrp, -, -, hrIn, hw0, hts, hk2, -⟩ := lincheck_coins h1
  obtain ⟨_, -, -, hgam, -, -, hsws, hsw⟩ := ringSwitch_spec (by simp [claimsOf]) h2
  simp only [Info.map, ZcPts.map, LcPts.map] at hr hrho hrp hrIn hw0 hts
  simp only [Info.map, RsPts.map] at hgam hsws hsw
  obtain ⟨-, hc₁', hc₁s⟩ := ofRounds_round hne hc₁
  obtain ⟨-, hcr', hcrs⟩ := ofRounds_round hne hcr
  obtain ⟨hgsz, hgam', -⟩ := ofRounds_round hne hgam
  have hAs : (claimsOf sh.m sh.kLog (info.zc.map (Lin.eval (valOf sh d.w d.comb d.cons)))
      (info.lc.map (Lin.eval (valOf sh d.w d.comb d.cons))) d.extras).size = 2 + d.extras.size := by
    simp [claimsOf]
  have hFs := claimsF_size sh (valOf sh d.w d.comb d.cons) info d.extras
  have hkk : k₂ = k2 := by omega
  have hrp' : (t.rounds[k₁ + 2 + (st.fold.kLog - 6)]!).coins.getD 0 0 = info.lc.rPrime := by
    rw [show k₁ + 2 + (st.fold.kLog - 6) = k1 + 2 + (decoded sh d.w d.comb d.cons).lc.size by omega, hcr',
      getD_zero_of_size hcrs, hrp]
  have hcl : claimsF sh (valOf sh d.w d.comb d.cons) info d.extras =
      abcClaims st.m st.kLog (zcOf p.zerocheck info.zc.r info.zc.z info.zc.rho)
        (lcOf p.lincheck ((t.rounds[k₁ + 2 + (st.fold.kLog - 6)]!).coins.getD 0 0) info.lc.fold.ts) ++ d.extras := by
    rw [hrp', hmS, hkS]
    have hw1 : Lin.eval (valOf sh d.w d.comb d.cons) info.lc.w =
        dot (skipWeights info.lc.rPrime) p.lincheck.zPartial := by
      rw [hzp, ← dotV_eq_dot, hw0]
    have hex : (d.extras.map fun e => (⟨Val.cst e.value, e.skip, e.x⟩ : AClaim F128)).map AClaim.toClaim =
        d.extras := by
      rw [Array.map_map]
      exact Array.map_id'' (fun e => rfl) _
    unfold claimsF claimsOf abcClaims
    rw [Array.map_append, hex]
    simp only [zcOf, ZcPts.map, LcPts.map, hw1, hfc, hrIn, List.map_toArray, List.map_cons, List.map_nil,
      AClaim.toClaim]
  refine ⟨hcl, by rw [hrsw, hsrs, hFs, hsc], fun j hj => ?_, fun j hj => ?_, ?_, ?_, ?_⟩
  · rw [hFs] at hj
    refine ⟨by rw [hrsw]; exact hsrs' j (by omega), ?_⟩
    rw [hcl]
    refine pair_append_get (P := fun c => c.skip.size = 64 ∧ c.x.size = st.m - 6)
      ⟨skipWeights_size _, ?_⟩ ⟨skipWeights_size _, ?_⟩ hext j hj
    · simp only [zcOf, Array.size_append, Array.size_reverse, Array.size_extract, hts, hrho, hslc, hszc]
      omega
    · show info.zc.r.size = st.m - 6
      rw [hr, Array.size_append, FlockLevel3.pinned_size, hc₁s]
      omega
  · rw [hrsw]
    exact hval j (by rw [hcl0]; exact hj)
  · rw [hFs]
    refine ⟨?_, hrn, fun j hj => ?_, hra, hgn, ?_, hga⟩
    · show k₂ + (2 + d.extras.size) < t.rounds.size
      rw [hkk, ← hAs]
      exact hgsz
    · obtain ⟨c, hc', -, -, h4⟩ := hsw j (by rw [hAs]; exact hj)
      obtain ⟨-, hc'', -⟩ := ofRounds_round hne hc'
      have hjs : j < info.rs.sws.size := by rw [hsws, hAs]; exact hj
      rw [Array.getElem?_eq_getElem hjs, Option.some.injEq] at h4
      show (t.rounds[k₂ + j]!).coins = (info.rs.sws[j]!).rpp
      rw [getElem!_pos info.rs.sws j hjs, h4, hkk]
      exact hc''
    · show (t.rounds[k₂ + (2 + d.extras.size)]!).coins = info.rs.gamma
      rw [hkk, ← hAs]
      exact hgam'
  · rw [hT]
    unfold rsT
    rw [← hcl0, hrsw]
  · apply Array.ext
    · simp only [rsSws, Array.size_map, Array.size_range, hsws, hFs, hAs]
    · intro j h1 h2
      have hj : j < 2 + d.extras.size := by simpa [rsSws, hFs] using h1
      obtain ⟨c, -, -, -, h4⟩ := hsw j (by rw [hAs]; exact hj)
      rw [Array.getElem?_eq_getElem h2, Option.some.injEq] at h4
      simp only [rsSws, Array.getElem_map, Array.getElem_range]
      rw [getElem!_pos info.rs.sws j h2, h4,
        getElem!_pos (claimsF sh (valOf sh d.w d.comb d.cons) info d.extras) j (by rw [hFs]; exact hj)]
      simp only [claimsF, Array.getElem_map, AClaim.toClaim]

end FlockVBridge.Runs
