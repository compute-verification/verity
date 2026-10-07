import Proofs.Flock.Soundness.Refine.Zerocheck
import Proofs.Flock.Soundness.Refine.Realizes
import Proofs.Flock.Level3.Folded

/-!
# The lincheck refines the model's (PROTOCOL.md §10)

`lincheck_refines`: if the executable's `lincheck` succeeds, on a statement whose circuit fold realizes the model
statement `S` (`FoldRealizes`) and on the zerocheck's output, the model's `lincheck`, played with the proof's lincheck
messages (`lcMsgs`) and the coins of the rounds consumed, returns `some ab`, and `ab` is the executable's `ab` claim
(§11) in the model's field.

After the rounds the executable takes the 64 entries left from the statement (`CircuitFold.folded`), which are the
column vector halved at each round's challenge (`halve`, `FoldRealizes.folded`); the model folds it after the rounds
(`foldAll`). `foldl_halve` shows the two agree on the 64 entries left.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript

local notation "𝔽" => FlockLevel3.GF128

/-! ## The column fold -/

theorem fOf_halve (t : F128) (comb : Array F128) (i : ℕ) (hi : i < comb.size / 2) :
    fOf (halve t comb)[i]! = Model.foldTop (fOf t) (comb.size / 2) (fun c => fOf comb[c]!) i := by
  simp only [halve, Model.foldTop]
  rw [getElem!_range_map _ _ i hi, fOf_add, fOf_mul, fOf_add]

/-- `foldAll` over `len` entries reads only the first `len`. -/
theorem foldAll_congr : ∀ (ts : List 𝔽) (len : ℕ) (v v' : ℕ → 𝔽), (∀ i < len, v i = v' i) →
    ∀ i < len / 2 ^ ts.length, Model.foldAll ts len v i = Model.foldAll ts len v' i
  | [], len, v, v', h, i, hi => by simp only [Model.foldAll]; exact h i (by simpa using hi)
  | t :: ts, len, v, v', h, i, hi => by
      simp only [Model.foldAll]
      refine foldAll_congr ts (len / 2) _ _ (fun j hj => ?_) i ?_
      · simp only [Model.foldTop]
        rw [h j (by omega), h (j + len / 2) (by omega)]
      · rwa [List.length_cons, pow_succ', ← Nat.div_div_eq_div_mul] at hi

/-- **The column fold.** Halving at the challenges `ts` in turn gives, on the entries left, the model's `foldAll`. -/
theorem foldl_halve : ∀ (ts : List F128) (comb : Array F128), ∀ i < comb.size / 2 ^ ts.length,
    fOf (ts.foldl (fun c t => halve t c) comb)[i]! =
      Model.foldAll (ts.map fOf) comb.size (fun c => fOf comb[c]!) i
  | [], comb, i, _ => by simp [Model.foldAll]
  | t :: ts, comb, i, hi => by
      rw [List.foldl_cons, List.map_cons, Model.foldAll]
      have hi' : i < (halve t comb).size / 2 ^ ts.length := by
        rw [halve_size, Nat.div_div_eq_div_mul, ← pow_succ']; simpa using hi
      rw [foldl_halve ts (halve t comb) i hi', halve_size]
      refine foldAll_congr (ts.map fOf) (comb.size / 2) _ _ (fun j hj => fOf_halve t comb j hj) i ?_
      rw [List.length_map]; rwa [halve_size] at hi'

/-! ## The rounds -/

/-- One lincheck round (L3) as the executable computes it, on the claim and the challenges so far. -/
def lStep (rounds : Array (F128 × F128)) (j : ℕ) (s : F128 × Array F128) (c : Array F128) : F128 × Array F128 :=
  let q := rounds[j]!
  let t := c.getD 0 0
  let e0 := s.1 + q.1
  let c1 := e0 + q.1 + q.2
  (q.2 * t * t + c1 * t + e0, s.2.push t)

theorem lStep_fold (rounds : Array (F128 × F128)) (C : ℕ → Array F128) :
    ∀ (l : List ℕ) (s : F128 × Array F128),
      (l.foldl (fun s i => lStep rounds i s (C i)) s).2 = s.2 ++ (l.map fun i => (C i).getD 0 0).toArray := by
  intro l
  induction l with
  | nil => intro s; simp
  | cons i l ih =>
      intro s
      rw [List.foldl_cons, ih (lStep rounds i s (C i))]
      simp [lStep]

theorem lStep_ts_get (s : Array F128) (l : List F128) (j : ℕ) (hj : j < l.length) :
    (s ++ l.toArray)[s.size + j]! = l[j] := by
  rw [getElem!_pos _ _ (by simp; omega), Array.getElem_append_right (by omega)]
  simp

/-- The model's lincheck round on the words of `lStep`. -/
theorem lincheck_round_step (rounds : Array (F128 × F128)) (a : ℕ) (s : F128 × Array F128) (c : Array F128) :
    (pairOf rounds[a]!).2 * fOf c[0]! ^ 2 +
        (fOf s.1 + (pairOf rounds[a]!).1 + (pairOf rounds[a]!).1 + (pairOf rounds[a]!).2) * fOf c[0]! +
        (fOf s.1 + (pairOf rounds[a]!).1) = fOf (lStep rounds a s c).1 := by
  simp only [lStep, pairOf, getElem!_eq_getD c 0, fOf_add, fOf_mul, pow_two, mul_assoc]

/-- **The lincheck rounds.** The model's `lincheckRounds`, fed round `i`'s messages `rounds[i]` and coins `C i`, ends
with the challenges and the claim of the fold of `lStep`. -/
theorem lincheckRounds_plays (rounds : Array (F128 × F128)) (C : ℕ → Array F128) :
    ∀ (n a : ℕ) (s : F128 × Array F128), (∀ i ∈ List.range' a n, (C i).size = 1) →
      Plays fOf (Model.lincheckRounds n (fOf s.1))
        ((List.range' a n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩) ((List.range' a n).map C)
        (fun j => fOf ((List.range' a n).foldl (fun s i => lStep rounds i s (C i)) s).2[s.2.size + j.val]!,
          fOf ((List.range' a n).foldl (fun s i => lStep rounds i s (C i)) s).1) := by
  intro n
  induction n with
  | zero =>
      intro a s _
      simp only [List.range'_zero, List.map_nil, List.foldl_nil]
      rw [Model.lincheckRounds]
      have e : (Fin.elim0 : Fin 0 → 𝔽) = fun j => fOf s.2[s.2.size + j.val]! := funext fun j => j.elim0
      rw [← e]
      exact Plays.pure _
  | succ n ih =>
      intro a s hC
      rw [Model.lincheckRounds]
      have hih := ih (a + 1) (lStep rounds a s (C a))
        (fun i hi => hC i (by rw [List.range'_succ]; exact List.mem_cons_of_mem _ hi))
      have hCa : (C a).size = 1 := hC a (by rw [List.range'_succ]; exact List.mem_cons_self)
      have hts := lStep_fold rounds C (List.range' (a + 1) n) (lStep rounds a s (C a))
      have hs1 : (lStep rounds a s (C a)).2.size = s.2.size + 1 := by simp [lStep]
      refine Plays.bind' (ms₂ := (List.range' (a + 1) n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩)
        (cs₂ := (List.range' a (n + 1)).map C) (Plays.recv _ (pairOf rounds[a]!)) ?_ ?_ ?_
      · refine Plays.bind' (ms₂ := (List.range' (a + 1) n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩)
          (cs₂ := (List.range' (a + 1) n).map C) (Plays.drawOne (C a) hCa) ?_ ?_ ?_
        · simp only []
          rw [lincheck_round_step rounds a s (C a)]
          refine Plays.bind' (ms₂ := []) (cs₂ := []) hih ?_ ?_ ?_
          · simp only [List.range'_succ, List.foldl_cons]
            convert Plays.pure (ι := fOf) _ using 2
            funext j
            refine Fin.cases ?_ (fun j => ?_) j
            · rw [Fin.cons_zero, Fin.val_zero, Nat.add_zero s.2.size, hts]
              simp only [lStep]
              rw [getElem!_pos _ s.2.size (by simp), Array.getElem_append_left (by simp), Array.getElem_push_eq,
                getElem!_eq_getD]
            · rw [Fin.cons_succ, hs1, Fin.val_succ, show s.2.size + (j.val + 1) = s.2.size + 1 + j.val by omega]
          all_goals simp
        all_goals simp [List.range'_succ]
      all_goals simp [List.range'_succ]

theorem ofFn_getElem! (n : ℕ) (a : Array F128) (l : List F128) (h : a = #[] ++ l.toArray) (hl : l.length = n) :
    List.ofFn (fun j : Fin n => fOf a[0 + j.val]!) = l.map fOf := by
  subst h
  refine List.ext_getElem (by simp [hl]) fun j h₁ h₂ => ?_
  rw [List.getElem_ofFn, List.getElem_map, Nat.zero_add]
  rw [getElem!_pos _ _ (by simp at h₁ ⊢; omega)]
  simp

/-- The executable's `c` claim is the model's (§11): `(P_c(z), L_s(z), r[6 … m−1])`. -/
theorem cClaim_rel {m : ℕ} (zc : Flock.ZcOut) (zc' : Model.ZcOut 𝔽 m) (hzc : ZcRel m zc zc') :
    ClaimRel m { value := zc.vC, skip := skipWeights zc.z, x := zc.rRest } (Model.cClaim execArith zc') :=
  ⟨hzc.vc, skipWeights_size _, fun s => by simp only [Model.cClaim, hzc.z]; rfl, hzc.r_size, hzc.r⟩

/-- **The lincheck refines the model's** (PROTOCOL.md §10). If the executable's lincheck succeeds from `t` to `t'`, on
a circuit fold realizing `S` and on a zerocheck output that is the model's `zc'`, the model's lincheck on `zc'`, played
with the proof's lincheck messages and the coins of the rounds consumed, returns the executable's `ab` claim, read in
the model's field. -/
theorem lincheck_refines (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S)
    (zc : Flock.ZcOut) (zc' : Model.ZcOut 𝔽 S.m) (hzc : ZcRel S.m zc zc') (p : LincheckProof)
    (t t' : Transcript) (lc : LcOut) (hwf : RoundsWF t.rounds)
    (h : Flock.lincheck c zc p t = .ok (lc, t')) :
    t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      ∃ ab, Plays fOf (Model.lincheck execArith S zc') (lcMsgs S.kLog p) (consumed t.rounds t.k t'.k) (some ab) ∧
        ClaimRel S.m (abcClaims S.m S.kLog zc lc)[0]! ab := by
  unfold Flock.lincheck at h
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
    obtain ⟨hfinal, h⟩ := check_bind h
    obtain ⟨lt₆, n₆, t₇, r₇, kk₇, h⟩ := sample_bind h
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
    set cα := (t₁.rounds[t₁.k]!).coins with hcα
    set cβ := (t₂.rounds[t₂.k]!).coins with hcβ
    set cr := (t₆.rounds[t₆.k]!).coins with hcr
    set C : ℕ → Array F128 := fun i => (t₃.rounds[t₃.k + i]!).coins with hC
    have hwf₁ : RoundsWF t₁.rounds := k₁.2 ▸ hwf
    have hwf₂ : RoundsWF t₂.rounds := hr₂ ▸ hwf
    have hwf₃ : RoundsWF t₃.rounds := hr₃ ▸ hwf
    have hwf₆ : RoundsWF t₆.rounds := hr₆ ▸ hwf
    have hszα : cα.size = 1 := by rw [hcα, RoundsWF.get hwf₁, n₁]
    have hszβ : cβ.size = 1 := by rw [hcβ, RoundsWF.get hwf₂, n₂]
    have hszr : cr.size = 1 := by rw [hcr, RoundsWF.get hwf₆, n₆]
    have hCsz : ∀ i ∈ List.range' 0 (S.kLog - 6), (C i).size = 1 := fun i hi => by
      rw [hC]; simp only; rw [RoundsWF.get hwf₃, hn₅ i hi]
    have hkt : t'.k = t.k + 2 + (S.kLog - 6) + 1 := by rw [kk₇, hk₆]
    have hrt : t'.rounds = t.rounds := by rw [r₇, hr₆]
    have hcoins : consumed t.rounds t.k t'.k = cα :: cβ :: ((List.range' 0 (S.kLog - 6)).map C ++ [cr]) := by
      rw [hkt, consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by omega),
        consumed_append t.rounds (show t.k + 1 ≤ t.k + 2 by omega) (by omega),
        consumed_append t.rounds (show t.k + 2 ≤ t.k + 2 + (S.kLog - 6) by omega) (by omega),
        consumed_succ, consumed_succ, consumed_succ, consumed_eq_map, hcα, hcβ, hcr, hC, k₁.2, k₁.1, hr₂, hk₂,
        hr₃, hk₃, hr₆, hk₆]
      simp
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
    have hfinE : dot w p.zPartial = s.1 := eq_of_beq hfinal
    have hαM : fOf cα[0]! = fOf (cα.getD 0 0) := by rw [getElem!_eq_getD]
    have hβM : fOf cβ[0]! = fOf (cβ.getD 0 0) := by rw [getElem!_eq_getD]
    have hrM : fOf cr[0]! = fOf (cr.getD 0 0) := by rw [getElem!_eq_getD]
    have hclaim0 : fOf (cα.getD 0 0) * zc'.va + zc'.vb + fOf (cβ.getD 0 0) = fOf s₀.1 := by
      rw [hs₀, hzc.va, hzc.vb]; simp only [fOf_add, fOf_mul]
    have hres := lincheckRounds_plays p.rounds C (S.kLog - 6) 0 s₀ hCsz
    rw [← hs, ← hclaim0] at hres
    have htsF_size : s.2.size = S.kLog - 6 := by rw [htsF, hs₀]; simp [htsL_len]
    refine ⟨hrt, by omega, ?ab, ?plays, ?rel⟩
    case plays =>
      rw [hcoins]
      unfold Model.lincheck
      refine Plays.bind' (ms₂ := lcMsgs S.kLog p) (cs₂ := cβ :: ((List.range' 0 (S.kLog - 6)).map C ++ [cr]))
        (Plays.drawOne cα hszα) ?_ (by simp) (by simp)
      refine Plays.bind' (ms₂ := lcMsgs S.kLog p) (cs₂ := (List.range' 0 (S.kLog - 6)).map C ++ [cr])
        (Plays.drawOne cβ hszβ) ?_ (by simp) (by simp)
      simp only [hαM, hβM]
      refine Plays.bind' (ms₂ := [⟨Fin 64 → 𝔽, vecOf 64 p.zPartial⟩]) (cs₂ := [cr]) hres ?_
        (by simp [lcMsgs, List.range_eq_range']) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := [cr]) (Plays.recv _ (vecOf 64 p.zPartial)) ?_ (by simp) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawOne cr hszr) ?_ (by simp) (by simp)
      simp only [hrM]
      split
      · exact Plays.pure (ι := fOf) _
      · rename_i hne
        exfalso
        apply hne
        rw [← hfinE, fOf_dot w p.zPartial 64 hcombFsz]
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

end FlockSoundness.Refine
