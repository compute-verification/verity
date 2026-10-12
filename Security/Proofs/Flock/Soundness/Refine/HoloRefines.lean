import Proofs.Flock.Soundness.Refine.HoloRep

/-!
# One rep with hidden units refines the model's `holoRepC` (PROTOCOL.md §9–§14, L4′)

`holoLincheck_refines` is `lincheck_refines` at `holoLincheckT`: the model's `holoLincheck`, played with the lincheck's
messages and `h` (`holoLcMsgs`) and the coins of the rounds consumed but `z_partial`'s, which issued none, returns the
executable's `ab` claim and `h`. `holoRep_refines` composes it with `zerocheck_refines`, `opening_refines` and
`ligerito_accepts_opening` into `holoRepC`, as `rep_refines` composes `repC`, on a run of `holoRepProg`.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128

/-- **The lincheck with hidden units refines the model's** (§10 with L4′). If `holoLincheckT` succeeds from `t` to
`t'`, on a circuit fold realizing `S` and on a zerocheck output that is the model's `zc'`, it consumed `k_log − 6 + 4`
rounds, the one after the rounds (`z_partial`'s) issued no coins, and the model's `holoLincheck` on `zc'`, played with
the proof's lincheck messages and `h` and the coins of the other rounds, returns the executable's `ab` claim, read in
the model's field, with `h` as the hidden part's fold. -/
theorem holoLincheck_refines (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S)
    (zc : Flock.ZcOut) (zc' : Model.ZcOut 𝔽 S.m) (hzc : ZcRel S.m zc zc') (p : LincheckProof) (hpart : Array F128)
    (t t' : Transcript) (lc : LcOut) (hwf : RoundsWF t.rounds)
    (h : holoLincheckT c zc p hpart t = .ok (lc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + 2 + (S.kLog - 6) + 2 ∧ (t.rounds[t.k + 2 + (S.kLog - 6)]!).n = 0 ∧
      ∃ out : HoloLcOut 𝔽 S.m (S.kLog - 6),
        Plays fOf (Model.holoLincheck execArith S zc') (holoLcMsgs S.kLog p hpart)
          (consumed t.rounds t.k (t.k + 2 + (S.kLog - 6)) ++ consumed t.rounds (t.k + 2 + (S.kLog - 6) + 1) t'.k)
          (some out) ∧
        ClaimRel S.m (abcClaims S.m S.kLog zc lc)[0]! out.ab ∧ out.h = vecOf 64 hpart := by
  unfold holoLincheckT at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨lt₁, n₁, t₂, r₂, kk₂, h⟩ := sample_bind h
  obtain ⟨lt₂, n₂, t₃, r₃, kk₃, h⟩ := sample_bind h
  obtain ⟨s, t₅, hloop, h⟩ := bind_ok h
  dsimp only at h
  rcases hw : c.folded (t₁.rounds[t₁.k]!.coins.getD 0 0) (t₂.rounds[t₂.k]!.coins.getD 0 0) (skipWeights zc.z)
      (zc.rho.extract 0 (c.kLog - 6)) s.2 with err | w
  · rw [hw] at h
    obtain ⟨_, _, h₀, _⟩ := bind_ok h
    cases h₀
  · rw [hw] at h
    simp only [pure_bind] at h
    obtain ⟨hsize, h⟩ := check_bind h
    obtain ⟨t₆, k₆, h⟩ := absorbSlice_bind h
    obtain ⟨lt₆, hzn, tz, hzr, hzk, h⟩ := closeEmpty_bind h
    obtain ⟨t₇, k₇, h⟩ := absorbSlice_bind h
    obtain ⟨hfinal, h⟩ := check_bind h
    obtain ⟨lt₈, n₈, t₈, r₈, kk₈, h⟩ := sample_bind h
    obtain ⟨rfl, rfl⟩ := pure_ok h
    have hkl : c.kLog = S.kLog := hc.kLog
    have hk6 : 6 ≤ S.kLog := S.hk6
    have hkm : S.kLog ≤ S.m := S.hk
    rw [hkl] at hloop hw
    -- the circuit's 64 entries: the fold, with `β` at the pin, halved at the rounds' challenges
    obtain ⟨v, hv, hvsz, hwv⟩ := hc.folded _ _ _ _ _ w hw
    rw [hc.pin] at hwv
    -- the rounds: one round consumed per step
    rw [forIn_range_eq] at hloop
    obtain ⟨hr₅, hk₅, hn₅, hs⟩ := forIn_rounds _ (lStep p.rounds) 1 (by
        intro j st tt b tt' hb
        obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
        obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
        obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
        obtain ⟨rfl, rfl⟩ := pure_ok hb
        have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
        have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
        refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
        rw [← e1, ← e2]
        rfl) (S.kLog - 6) 0 t₃.k _ t₃ t₅ s (by omega) hloop
    -- the rounds and the counter, in terms of `t`
    have hr₂ : t₂.rounds = t.rounds := by rw [r₂, k₁.2]
    have hk₂ : t₂.k = t.k + 1 := by rw [kk₂, k₁.1]
    have hr₃ : t₃.rounds = t.rounds := by rw [r₃, hr₂]
    have hk₃ : t₃.k = t.k + 2 := by rw [kk₃, hk₂]
    have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, hr₅, hr₃]
    have hk₆ : t₆.k = t.k + 2 + (S.kLog - 6) := by rw [k₆.1, hk₅, hk₃]
    have hr₇ : t₇.rounds = t.rounds := by rw [k₇.2, hzr, hr₆]
    have hk₇ : t₇.k = t.k + 2 + (S.kLog - 6) + 1 := by rw [k₇.1, hzk, hk₆]
    set cα := (t₁.rounds[t₁.k]!).coins with hcα
    set cβ := (t₂.rounds[t₂.k]!).coins with hcβ
    set cr := (t₇.rounds[t₇.k]!).coins with hcr
    set C : ℕ → Array F128 := fun i => (t₃.rounds[t₃.k + i]!).coins with hC
    have hwf₁ : RoundsWF t₁.rounds := k₁.2 ▸ hwf
    have hwf₂ : RoundsWF t₂.rounds := hr₂ ▸ hwf
    have hwf₃ : RoundsWF t₃.rounds := hr₃ ▸ hwf
    have hwf₇ : RoundsWF t₇.rounds := hr₇ ▸ hwf
    have hszα : cα.size = 1 := by rw [hcα, RoundsWF.get hwf₁, n₁]
    have hszβ : cβ.size = 1 := by rw [hcβ, RoundsWF.get hwf₂, n₂]
    have hszr : cr.size = 1 := by rw [hcr, RoundsWF.get hwf₇, n₈]
    have hCsz : ∀ i ∈ List.range' 0 (S.kLog - 6), (C i).size = 1 := fun i hi => by
      rw [hC]; simp only; rw [RoundsWF.get hwf₃, hn₅ i hi]
    have hkt : t'.k = t.k + 2 + (S.kLog - 6) + 2 := by omega
    have hrt : t'.rounds = t.rounds := by rw [r₈, hr₇]
    have hcoins : consumed t.rounds t.k (t.k + 2 + (S.kLog - 6)) ++
        consumed t.rounds (t.k + 2 + (S.kLog - 6) + 1) t'.k =
          cα :: cβ :: ((List.range' 0 (S.kLog - 6)).map C ++ [cr]) := by
      rw [hkt, consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by omega),
        consumed_append t.rounds (show t.k + 1 ≤ t.k + 2 by omega) (by omega),
        consumed_succ, consumed_succ, consumed_succ, consumed_eq_map, hcα, hcβ, hcr, hC, k₁.2, k₁.1, hr₂, hk₂,
        hr₃, hk₃, hr₇, hk₇]
      simp
    have hzn' : (t.rounds[t.k + 2 + (S.kLog - 6)]!).n = 0 := by rw [← hk₆, ← hr₆]; exact hzn
    -- the loop's output
    set s₀ : F128 × Array F128 := (cα.getD 0 0 * zc.vA + zc.vB + cβ.getD 0 0, #[]) with hs₀
    have htsF := lStep_fold p.rounds C (List.range' 0 (S.kLog - 6)) s₀
    rw [← hs] at htsF
    set tsL := (List.range' 0 (S.kLog - 6)).map fun i => (C i).getD 0 0 with htsL
    have htsL_len : tsL.length = S.kLog - 6 := by simp [htsL]
    have hcombF : w = tsL.foldl (fun c t => halve t c) (v.modify S.pin.val fun x => x + cβ.getD 0 0) := by
      rw [hwv, htsF, hs₀]; simp
    have hs₀sz : (v.modify S.pin.val fun x => x + cβ.getD 0 0).size = 2 ^ S.kLog := by simp [hvsz]
    have hcombFsz : w.size = 64 := eq_of_beq hsize
    have hrho : zc.rho.size = S.m - 6 := hzc.rho_size
    -- the column vector: the executable's is the model's `comb`
    have hcomb : ∀ col < 2 ^ S.kLog, fOf (v.modify S.pin.val fun x => x + cβ.getD 0 0)[col]! =
        Model.comb execArith S zc'.z (fOf (cα.getD 0 0)) (fOf (cβ.getD 0 0))
          (fun j => zc'.ρ ⟨j.val, by omega⟩) col := by
      intro col hcol
      have hfr := hc.fold (cα.getD 0 0) (skipWeights zc.z) (zc.rho.extract 0 (S.kLog - 6)) v (skipWeights_size _)
        (by simp; omega) hv ⟨col, hcol⟩
      rw [modify_get! v S.pin.val col _ (by omega), Model.comb, dite_eq_left hcol]
      have hsum : fOf v[col]! = ∑ i : Fin (2 ^ S.kLog), execArith.lagS zc'.z ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
          Model.eqAt (fun j : Fin (S.kLog - 6) => zc'.ρ ⟨j.val, by omega⟩) (i.val / 64) *
            (fOf (cα.getD 0 0) * ((S.A₀ i ⟨col, hcol⟩).val : 𝔽) + ((S.B₀ i ⟨col, hcol⟩).val : 𝔽)) := by
        rw [hfr]
        have hρ : (vecOf (S.kLog - 6) (zc.rho.extract 0 (S.kLog - 6))) =
            fun j : Fin (S.kLog - 6) => zc'.ρ ⟨j.val, by omega⟩ := by
          funext j
          rw [hzc.rho]
          simp only [vecOf]
          rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by omega), Array.getElem_extract]
          simp
        rw [hρ, hzc.z]
        rfl
      by_cases hp : S.pin.val = col
      · rw [ite_eq_left hp, ite_eq_left hp.symm, fOf_add, hsum]
      · rw [ite_eq_right hp, ite_eq_right (Ne.symm hp), add_zero, hsum]
    have hfold : ∀ (ρIn : Fin (S.kLog - 6) → 𝔽), ρIn = (fun j => zc'.ρ ⟨j.val, by omega⟩) → ∀ i < 64,
        Model.foldAll (List.ofFn fun j : Fin (S.kLog - 6) => fOf s.2[s₀.2.size + j.val]!) (2 ^ S.kLog)
          (Model.comb execArith S zc'.z (fOf (cα.getD 0 0)) (fOf (cβ.getD 0 0)) ρIn) i = fOf w[i]! := by
      intro ρIn hρIn i hi
      subst hρIn
      have hof : (List.ofFn fun j : Fin (S.kLog - 6) => fOf s.2[s₀.2.size + j.val]!) = tsL.map fOf :=
        ofFn_getElem! (S.kLog - 6) s.2 tsL htsF htsL_len
      have hi' : i < (v.modify S.pin.val fun x => x + cβ.getD 0 0).size / 2 ^ tsL.length := by
        rw [hs₀sz, htsL_len, Nat.pow_div (by omega) (by norm_num), show S.kLog - (S.kLog - 6) = 6 by omega]; exact hi
      rw [hof, hcombF, foldl_halve tsL _ i hi', hs₀sz]
      symm
      refine foldAll_congr (tsL.map fOf) (2 ^ S.kLog) _ _ (fun c hc' => ?_) i ?_
      · rw [hcomb c hc']
      · rw [List.length_map, htsL_len, Nat.pow_div (by omega) (by norm_num),
          show S.kLog - (S.kLog - 6) = 6 by omega]; exact hi
    have hfinE : dot (RecAlgebra.addH w hpart) p.zPartial = s.1 := eq_of_beq hfinal
    have hαM : fOf cα[0]! = fOf (cα.getD 0 0) := by rw [getElem!_eq_getD]
    have hβM : fOf cβ[0]! = fOf (cβ.getD 0 0) := by rw [getElem!_eq_getD]
    have hrM : fOf cr[0]! = fOf (cr.getD 0 0) := by rw [getElem!_eq_getD]
    have hclaim0 : fOf (cα.getD 0 0) * zc'.va + zc'.vb + fOf (cβ.getD 0 0) = fOf s₀.1 := by
      rw [hs₀, hzc.va, hzc.vb]; simp only [fOf_add, fOf_mul]
    have hres := lincheckRounds_plays p.rounds C (S.kLog - 6) 0 s₀ hCsz
    rw [← hs, ← hclaim0] at hres
    have htsF_size : s.2.size = S.kLog - 6 := by rw [htsF, hs₀]; simp [htsL_len]
    refine ⟨hrt, hkt, hzn', ?out, ?plays, ?rel, ?hh⟩
    case plays =>
      rw [hcoins]
      unfold Model.holoLincheck
      refine Plays.bind' (ms₂ := holoLcMsgs S.kLog p hpart)
        (cs₂ := cβ :: ((List.range' 0 (S.kLog - 6)).map C ++ [cr])) (Plays.drawOne cα hszα) ?_ (by simp) (by simp)
      refine Plays.bind' (ms₂ := holoLcMsgs S.kLog p hpart) (cs₂ := (List.range' 0 (S.kLog - 6)).map C ++ [cr])
        (Plays.drawOne cβ hszβ) ?_ (by simp) (by simp)
      simp only [hαM, hβM]
      refine Plays.bind' (ms₂ := [⟨Fin 64 → 𝔽, vecOf 64 p.zPartial⟩, ⟨Fin 64 → 𝔽, vecOf 64 hpart⟩]) (cs₂ := [cr])
        hres ?_ (by simp [holoLcMsgs, lcMsgs, List.range_eq_range']) (by simp)
      refine Plays.bind' (ms₂ := [⟨Fin 64 → 𝔽, vecOf 64 hpart⟩]) (cs₂ := [cr])
        (Plays.recv _ (vecOf 64 p.zPartial)) ?_ (by simp) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := [cr]) (Plays.recv _ (vecOf 64 hpart)) ?_ (by simp) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawOne cr hszr) ?_ (by simp) (by simp)
      simp only [hrM]
      split
      · exact Plays.pure (ι := fOf) _
      · rename_i hne
        exfalso
        apply hne
        rw [← hfinE, RecAlgebra.fOf_dot_holo]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [hfold _ rfl i.val i.isLt]
        rfl
    case rel =>
      show ClaimRel S.m (⟨dot (skipWeights (cr.getD 0 0)) p.zPartial, skipWeights (cr.getD 0 0),
        s.2.reverse ++ zc.rho.extract (S.kLog - 6) (S.m - 6)⟩ : Flock.Claim) _
      have hs₀e : s₀.2.size = 0 := by rw [hs₀]; rfl
      refine ⟨by rw [fOf_dot _ _ 64 (skipWeights_size _)]; rfl, skipWeights_size _, fun _ => rfl,
        by simp [htsF_size, hrho]; omega, fun i => ?_⟩
      simp only [hs₀e, Nat.zero_add]
      have hxs : (s.2.reverse ++ zc.rho.extract (S.kLog - 6) (S.m - 6)).size = S.m - 6 := by
        simp [htsF_size, hrho]; omega
      rw [getElem!_pos _ i.val (by rw [hxs]; exact i.isLt), Array.getElem_append]
      by_cases hi : i.val < S.kLog - 6
      · have hi' : i.val < s.2.reverse.size := by simp [htsF_size, hi]
        rw [dite_eq_left hi, dite_eq_left hi', Array.getElem_reverse, getElem!_pos _ _ (by omega)]
        congr 2
        omega
      · have hi' : ¬ i.val < s.2.reverse.size := by simp [htsF_size, hi]
        rw [dite_eq_right hi, dite_eq_right hi', Array.getElem_extract, hzc.rho, getElem!_pos _ _ (by omega)]
        congr 2
        simp [htsF_size]
        omega
    case hh => rfl

/-- **One rep with hidden units refines the model's** (§9–§14 with L4′). When `holoRepProg` returns from `t` to `t'`,
on a statement whose `m`, `k_log` and circuit fold are the model's `S`, with the extra claims `extras` the model's
`extra` claim by claim, and on a schedule whose dimensions and levels are the model's and whose levels' consistency
coins are not one word, one round `kz` in between issued no coins (`z_partial`'s). At the proof's openings `o`, the
model's `holoRepC`, played with the proof's messages and `h` (`holoRepMsgs`) and the coins of every round from `t.k` to
`t'.k` but `kz`, returns the zerocheck's and the lincheck's outputs, `h` the lincheck's, and a result whose verdict
is true, whose caps are the proof's and whose requests passed the executable's Merkle checks. -/
theorem holoRep_refines {D : Type} (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m) (h15 : 15 ≤ S.m)
    (sch : Flock.Schedule) (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (extra : List (Model.Claim 𝔽 S.m)) (extras : Array Claim) (hlenx : extras.size = extra.length)
    (hcx : ∀ (j : ℕ) (hj : j < extra.length), ClaimRel S.m extras[j]! extra[j])
    (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1) (hschm : sch.m = S.m)
    (p : Proof) (hpart : Array F128) (r : ℕ) (t t' : Transcript) (hwf : RoundsWF t.rounds)
    (hrun : holoRepProg st sch p hpart extras t = .ok ((), t'))
    (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p.ligerito (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf) :
    t'.rounds = t.rounds ∧ ∃ kz, t.k ≤ kz ∧ kz < t'.k ∧ (t.rounds[kz]!).n = 0 ∧
      ∃ (zc : Model.ZcOut 𝔽 S.m) (lc : HoloLcOut 𝔽 S.m (S.kLog - 6)) (pend : PendL 𝔽 D),
        Plays fOf (Model.holoRepC execArith r S (schedOf sch) hm extra)
          (holoRepMsgs st sch (2 + extra.length) p hpart capD)
          (consumed t.rounds t.k kz ++ consumed t.rounds (kz + 1) t'.k) (some (zc, lc), pend) ∧
        lc.h = vecOf 64 hpart ∧
        pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.ligerito.recursiveCaps[j]!) ∧
        (∀ req ∈ pend.reqs, ReqOK st.merkle sch p.cap p.ligerito req) ∧ pend.verdict o = true := by
  unfold holoRepProg at hrun
  obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨lc, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨⟨T, sws⟩, t₃, h₃, h₄⟩ := bind_ok hrun
  rw [hmS] at h₁
  obtain ⟨hr₁, hk₁, zc', hpl₁, hzc⟩ := zerocheck_refines S.m h15 _ _ _ _ _ zc hwf h₁
  obtain ⟨hr₂, hk₂, hz₂, lc', hpl₂, hab, hh⟩ := holoLincheck_refines S st.fold hfold zc zc' hzc p.lincheck hpart
    t₁ t₂ lc (hr₁ ▸ hwf) h₂
  rw [hmS, hkS] at h₃
  have hcsz : (abcClaims S.m S.kLog zc lc ++ extras).size = (lc'.ab :: Model.cClaim execArith zc' :: extra).length := by
    simp [abcClaims, hlenx]; omega
  obtain ⟨hr₃, hk₃, op, hpl₃, hop⟩ := opening_refines S.m (by omega) _ (lc'.ab :: Model.cClaim execArith zc' :: extra)
    hcsz (fun k hk => by
      rcases k with _ | _ | k
      · rw [getElem!_pos _ 0 (by simp [abcClaims]), Array.getElem_append_left (by simp [abcClaims])]
        rw [getElem!_pos _ 0 (by simp [abcClaims])] at hab
        exact hab
      · rw [getElem!_pos _ 1 (by simp [abcClaims]; omega), Array.getElem_append_left (by simp [abcClaims])]
        exact cClaim_rel zc zc' hzc
      · have hk' : k < extra.length := by simp at hk; omega
        have hsz2 : (abcClaims S.m S.kLog zc lc).size = 2 := rfl
        rw [getElem!_pos _ (k + 2) (by rw [Array.size_append, hsz2]; omega),
          Array.getElem_append_right (by rw [hsz2]; omega)]
        simp only [hsz2, Nat.add_sub_cancel]
        rw [← getElem!_pos extras k (by omega)]
        exact hcx k hk')
    (by simp [abcClaims]) (by rw [Array.size_append, show (abcClaims S.m S.kLog zc lc).size = 2 from rfl]; omega)
    p.ringSwitches t₂ t₃ T sws (hr₂ ▸ hr₁ ▸ hwf) h₃
  rw [← hschm] at hop
  obtain ⟨pend, hr₄, hk₄, hpl₄, hcaps, hreqs, hver⟩ := ligerito_accepts_opening st.merkle sch T sws _ op hop p.cap
    p.ligerito r capD o hdim hlv hα1 ho (hr₃ ▸ hr₂ ▸ hr₁ ▸ hwf) h₄
  have e2 : t₂.rounds = t.rounds := hr₂.trans hr₁
  have e3 : t₃.rounds = t.rounds := hr₃.trans e2
  have e4 : t'.rounds = t.rounds := hr₄.trans e3
  refine ⟨e4, t₁.k + 2 + (S.kLog - 6), by omega, by omega, by rw [← hr₁]; exact hz₂, zc', lc', pend, ?_, hh,
    hcaps, hreqs, hver⟩
  have hcs : consumed t.rounds t.k (t₁.k + 2 + (S.kLog - 6)) ++
      consumed t.rounds (t₁.k + 2 + (S.kLog - 6) + 1) t'.k =
        consumed t.rounds t.k t₁.k ++ ((consumed t₁.rounds t₁.k (t₁.k + 2 + (S.kLog - 6)) ++
          consumed t₁.rounds (t₁.k + 2 + (S.kLog - 6) + 1) t₂.k) ++
            (consumed t₂.rounds t₂.k t₃.k ++ consumed t₃.rounds t₃.k t'.k)) := by
    rw [hr₁, e2, e3, consumed_append t.rounds hk₁ (by omega : t₁.k ≤ t₁.k + 2 + (S.kLog - 6)),
      consumed_append t.rounds (by omega : t₁.k + 2 + (S.kLog - 6) + 1 ≤ t₂.k) (hk₃.trans hk₄),
      consumed_append t.rounds hk₃ hk₄]
    simp only [List.append_assoc]
  have hsz : (abcClaims S.m S.kLog zc lc ++ extras).size = 2 + extra.length := by
    simp [abcClaims, hlenx]
  rw [hsz] at hpl₃
  rw [hcs]
  unfold Model.holoRepC
  refine Plays.bind' (ms₂ := holoLcMsgs S.kLog p.lincheck hpart ++ opMsgs (2 + extra.length) p.ringSwitches ++
    ligMsgs p.ligerito capD sch) hpl₁ ?_ (by simp [holoRepMsgs, hmS, hkS]) rfl
  refine Plays.bind' (ms₂ := opMsgs (2 + extra.length) p.ringSwitches ++ ligMsgs p.ligerito capD sch) hpl₂ ?_
    (by simp) rfl
  refine Plays.bind' hpl₃ ?_ rfl rfl
  exact Plays.bind' (ms₂ := []) (cs₂ := []) hpl₄ (Plays.pure _) (by simp) (by simp)

end FlockSoundness.Refine
