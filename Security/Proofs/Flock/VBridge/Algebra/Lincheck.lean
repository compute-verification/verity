import Proofs.Flock.VBridge.Algebra.Zerocheck
import Proofs.Flock.Soundness.Refine.Lincheck

/-!
# V*'s lincheck residual is zero only when the lincheck's checks pass

`lincheck_checks`: when `lincheck`'s residual `Σ_l comb[l]·z_partial[l] + claim` (rec_algebra.py:601–617) is zero over
F, a proof whose lincheck messages are the ones `lincheck` read ends the verifier's rounds (Piop.lean:134–142) at
`dot comb z_partial` with challenges `lincheck`'s `ts`, at its `α`, `β`: the `check` of Piop.lean:148, when `comb` is
the verifier's `CircuitFold.folded` at `lincheck`'s fold points. The rounds are the fold of `Refine.lStep`, which
`Refine.forIn_rounds` shows the verifier's loop computes. The other `check` (Piop.lean:146) is the comb's width:
`folded_size` reads it off level 3's `folded_eq` for a statement's circuit fold.

`lincheck_coins` says which rounds `lincheck` draws its points from, `lincheck_of_residuals_dec` is the stage at a rep,
and `lincheck_of_residuals` is it at a plain shape, where the comb w holds is the comb.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-! ## The comb's width -/

/-- **The comb's width.** A statement's circuit fold (`Stmt.circuitFold`, the verifier's `Setup.ofCircuit`), at
`k_log − 6` round challenges, has the 64 entries Piop.lean:146 checks. -/
theorem folded_size {st : Stmt} {alpha beta : F128} {lw rho ts w : Array F128}
    (h : st.circuitFold.folded alpha beta lw rho ts = .ok w) (hts : ts.size + 6 = st.c.kLog) : w.size = 64 := by
  obtain ⟨v, hv, rfl⟩ := FlockLevel3.folded_eq st alpha beta lw rho ts w h
  rw [← Array.foldl_toList, FlockLevel3.foldl_halve_size, Array.size_modify, FlockLevel3.fold_size st alpha lw rho v hv,
    Array.length_toList, ← hts, pow_add, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
  rfl

/-! ## The lincheck over values -/

section CoinSource

variable {α : Type} [Val α]

/-- The lincheck's rounds draw one round each, its first coin the round's challenge. -/
theorem lcRounds_coins :
    ∀ (l : List (α × α)) (claim : α) (ts : Array F128) (k k' : Nat) (src : CoinSource) (claim' : α) (ts' : Array F128),
      lcRounds l claim ts k src = .ok ((claim', ts'), k') →
      ts'.size = ts.size + l.length ∧ (∀ j < ts.size, ts'[j]! = ts[j]!) ∧ k' = k + l.length ∧
      ∀ j < l.length, ∃ c, src.round (k + j) 1 = .ok c ∧ ts'[ts.size + j]! = c[0]!
  | [], claim, ts, k, k', src, claim', ts', h => by
    cases cm_pure h
    exact ⟨rfl, fun _ _ => rfl, rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  | (q1, qinf) :: rest, claim, ts, k, k', src, claim', ts', h => by
    simp only [lcRounds] at h
    obtain ⟨t, k₁, h1, h⟩ := cm_bind h
    obtain ⟨c, hc, rfl, rfl⟩ := coinOne_ok h1
    obtain ⟨hsz, hpre, hk, hco⟩ := lcRounds_coins rest _ (ts.push c[0]!) _ _ src _ _ h
    simp only [Array.size_push, List.length_cons] at hsz hpre hk hco ⊢
    refine ⟨by omega, fun j hj => ?_, by omega, fun j hj => ?_⟩
    · rw [hpre j (by omega), getElem!_pos _ j (by simp; omega), Array.getElem_push_lt, getElem!_pos _ j hj]
    · rcases j with _ | j
      · refine ⟨c, by simpa using hc, ?_⟩
        rw [Nat.add_zero, hpre ts.size (by omega), getElem!_pos _ ts.size (by simp), Array.getElem_push_eq]
      · obtain ⟨c', hc', he⟩ := hco j (by omega)
        exact ⟨c', by rw [show k + (j + 1) = k + 1 + j by omega]; exact hc',
          by rw [show ts.size + (j + 1) = ts.size + 1 + j by omega]; exact he⟩

/-- **The lincheck's coins.** The rounds `lincheck` draws its points from: `α`, `β`, one round per lincheck round, its
challenge, then `r'`; and the points it hands on. -/
theorem lincheck_coins {kLog : Nat} {M : Messages α} {zc : ZcPts α} {k k' : Nat} {src : CoinSource} {r1 : α}
    {lc : LcPts α} (h : lincheck kLog M zc k src = .ok ((r1, lc), k')) :
    ∃ cα cβ cr, src.round k 1 = .ok cα ∧ src.round (k + 1) 1 = .ok cβ ∧ src.round (k + 2 + M.lc.size) 1 = .ok cr ∧
      lc.fold.alpha = cα[0]! ∧ lc.fold.beta = cβ[0]! ∧ lc.rPrime = cr[0]! ∧ lc.fold.lw = skipWeights zc.z ∧
      lc.fold.rhoIn = zc.rho.extract 0 (kLog - 6) ∧ lc.rIn = lc.fold.ts.reverse ∧
      lc.w = dotV (skipWeights lc.rPrime) M.zp ∧ lc.fold.ts.size = M.lc.size ∧ k' = k + 3 + M.lc.size ∧
      ∀ j < M.lc.size, ∃ c, src.round (k + 2 + j) 1 = .ok c ∧ lc.fold.ts[j]! = c[0]! := by
  unfold lincheck at h
  obtain ⟨a, k₁, h1, h⟩ := cm_bind h
  obtain ⟨b, k₂, h2, h⟩ := cm_bind h
  obtain ⟨⟨claim, ts⟩, k₃, h3, h⟩ := cm_bind h
  obtain ⟨acc, k₄, h4, h⟩ := cm_bind h
  obtain ⟨rp, k₅, h5, h⟩ := cm_bind h
  cases cm_pure h
  obtain ⟨_, -, he⟩ := liftE_ok h4
  cases he
  obtain ⟨cα, hα, rfl, rfl⟩ := coinOne_ok h1
  obtain ⟨cβ, hβ, rfl, rfl⟩ := coinOne_ok h2
  obtain ⟨hsz, -, hk, hco⟩ := lcRounds_coins M.lc.toList _ #[] _ _ src _ _ h3
  simp only [Array.length_toList, Array.size_empty, Nat.zero_add] at hsz hk hco
  obtain ⟨cr, hr, rfl, rfl⟩ := coinOne_ok h5
  subst hk
  refine ⟨cα, cβ, cr, hα, hβ, by rw [show k + 1 + 1 = k + 2 by omega] at hr; exact hr, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, hsz, by omega, fun j hj => ?_⟩
  obtain ⟨c, hc, he⟩ := hco j hj
  exact ⟨c, by rw [show k + 2 + j = k + 1 + 1 + j by omega]; exact hc, he⟩

end CoinSource

/-! ## The lincheck over F -/

theorem combDot_ok : ∀ (l : List (F128 × F128)) (acc : F128),
    combDot l acc = .ok (l.foldl (fun a cx => a + cx.1 * cx.2) acc)
  | [], _ => rfl
  | (c, z) :: rest, acc => combDot_ok rest (acc + c * z)

theorem foldl_zip_dot (a b : Array F128) :
    (a.toList.zip b.toList).foldl (fun acc cx => acc + cx.1 * cx.2) 0 = dot a b := by
  rw [dot_eq_foldl, foldl_zip_eq_range (fun acc c x => acc + c * x), Array.length_toList, Array.length_toList]
  simp only [Array.getElem!_toList]

/-- The lincheck's rounds (Piop.lean:134–142) on round `j`'s coins `C j`: the fold of `Refine.lStep` from `claim`. -/
def lcFold (p : LincheckProof) (C : Nat → Array F128) (n : Nat) (claim : F128) : F128 × Array F128 :=
  (List.range' 0 n).foldl (fun s j => lStep p.rounds j s (C j)) (claim, #[])

/-- A lincheck proof (`Proof.lean`) carrying messages `M`. -/
structure LcProofOf (M : Messages F128) (p : LincheckProof) : Prop where
  rounds : p.rounds = M.lc
  zp : p.zPartial = M.zp

/-- The lincheck's rounds over F are the fold of `Refine.lStep` on round `j`'s messages and on coins `C` whose first
entries are the challenges drawn. -/
theorem lcRounds_fold (rounds : Array (F128 × F128)) (C : Nat → Array F128) :
    ∀ (l : List (F128 × F128)) (i : Nat) (claim : F128) (ts : Array F128) (k k' : Nat) (src : CoinSource)
      (claim' : F128) (ts' : Array F128),
      lcRounds l claim ts k src = .ok ((claim', ts'), k') →
      (∀ j (hj : j < l.length), rounds[i + j]! = l[j]) →
      (∀ j < l.length, (C (i + j)).getD 0 0 = ts'[ts.size + j]!) →
      (List.range' i l.length).foldl (fun s j => lStep rounds j s (C j)) (claim, ts) = (claim', ts')
  | [], i, claim, ts, k, k', src, claim', ts', h, _, _ => by
    cases cm_pure h
    rfl
  | (q1, qinf) :: rest, i, claim, ts, k, k', src, claim', ts', h, hl, hC => by
    simp only [lcRounds] at h
    obtain ⟨t, k₁, h1, h⟩ := cm_bind h
    obtain ⟨c, -, rfl, rfl⟩ := coinOne_ok h1
    obtain ⟨-, hpre, -, -⟩ := lcRounds_coins rest _ (ts.push c[0]!) _ _ src _ _ h
    simp only [Array.size_push] at hpre
    have hlast : ts'[ts.size]! = c[0]! := by
      rw [hpre ts.size (by omega), getElem!_pos _ ts.size (by simp), Array.getElem_push_eq]
    simp only [List.length_cons] at hl hC ⊢
    rw [List.range'_succ, List.foldl_cons]
    have hq : rounds[i]! = (q1, qinf) := by simpa using hl 0 (by simp)
    have hC0 : (C i).getD 0 0 = c[0]! := by simpa [hlast] using hC 0 (by simp)
    have hstep : lStep rounds i (claim, ts) (C i) =
        (Val.add (Val.add (Val.sm (c[0]! * c[0]!) qinf) (Val.sm c[0]! (Val.add (Val.add (Val.add claim q1) q1) qinf)))
          (Val.add claim q1), ts.push c[0]!) := by
      simp only [lStep, hq, hC0, Val.add, Val.sm, Prod.mk.injEq, and_true]
      (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)
    rw [hstep]
    refine lcRounds_fold rounds C rest (i + 1) _ (ts.push c[0]!) _ _ src _ _ h (fun j hj => ?_) (fun j hj => ?_)
    · have e := hl (j + 1) (by omega)
      rw [show i + (j + 1) = i + 1 + j by omega, List.getElem_cons_succ] at e
      exact e
    · rw [Array.size_push, show i + 1 + j = i + (j + 1) by omega, show ts.size + 1 + j = ts.size + (j + 1) by omega]
      exact hC (j + 1) (by omega)

/-- **The lincheck's checks.** When `lincheck` over F returns residual zero, a proof carrying its messages ends the
verifier's rounds, on any coins `C` whose first entries are its challenges `ts`, from `α·v_A + v_B + β` at its `α`, `β`,
at `dot comb z_partial` with challenges `ts`: Piop.lean:148's check, `comb` the verifier's `CircuitFold.folded` at
`lincheck`'s fold points. -/
theorem lincheck_checks {kLog : Nat} {M : Messages F128} {zc : ZcPts F128} {k k' : Nat} {src : CoinSource} {r1 : F128}
    {lc : LcPts F128} (h : lincheck kLog M zc k src = .ok ((r1, lc), k')) (h1 : r1 = 0) {p : LincheckProof}
    (hp : LcProofOf M p) {C : Nat → Array F128} (hC : ∀ j < M.lc.size, (C j).getD 0 0 = lc.fold.ts[j]!) :
    lcFold p C M.lc.size (lc.fold.alpha * M.fa + M.fb + lc.fold.beta) = (dot M.comb p.zPartial, lc.fold.ts) := by
  unfold lincheck at h
  obtain ⟨a, k₁, -, h⟩ := cm_bind h
  obtain ⟨b, k₂, -, h⟩ := cm_bind h
  obtain ⟨⟨claim, ts⟩, k₃, h3, h⟩ := cm_bind h
  obtain ⟨acc, k₄, h4, h⟩ := cm_bind h
  obtain ⟨rp, k₅, -, h⟩ := cm_bind h
  cases cm_pure h
  obtain ⟨_, hacc, he⟩ := liftE_ok h4
  cases he
  rw [combDot_ok] at hacc
  cases hacc
  have hfold := lcRounds_fold p.rounds C M.lc.toList 0 _ #[] _ _ src _ _ h3
  simp only [Array.length_toList, Array.size_empty, Nat.zero_add] at hfold
  unfold lcFold
  refine Eq.trans (hfold (fun j hj => by rw [hp.rounds, getElem!_pos _ j (by simpa using hj), Array.getElem_toList])
    (fun j hj => hC j (by simpa using hj))) ?_
  have hz : dot M.comb M.zp = claim := by
    rw [← foldl_zip_dot, ← Array.toList_zip]
    exact f128_eq_of_add_eq_zero h1
  rw [hp.zp, hz]

/-! ## The lincheck at a rep -/

theorem decoded_lc_size (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) :
    (decoded sh w comb cons).lc.size = sh.kLog - 6 := by
  simp [decoded, Messages.map, symbolic]

theorem decoded_comb_size (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) :
    (decoded sh w comb cons).comb.size = 64 := by
  simp [decoded, Messages.map, symbolic]

/-- The comb's elements w holds: the comb's, plus w's `h` at a holo shape (`hOf`). -/
theorem decoded_comb_get (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) {l : Nat} (hl : l < 64) :
    (decoded sh w comb cons).comb[l]! = comb[l]! + hOf sh w l := by
  rw [getElem!_pos _ l (by rw [decoded_comb_size]; exact hl)]
  simp only [decoded, Messages.map, symbolic, Array.getElem_map, Array.getElem_range, Function.comp, Lin.eval_var,
    valOf]

theorem decoded_comb (sh : Shape) {w comb : Array F128} (cons : Nat → Nat → F128) (hpl : sh.holo = false)
    (hc : comb.size = 64) : (decoded sh w comb cons).comb = comb := by
  apply Array.ext
  · simp [decoded, Messages.map, symbolic, hc]
  · intro i h1 h2
    simp only [decoded, Messages.map, symbolic, Array.getElem_map, Array.getElem_range, Function.comp, Lin.eval_var,
      valOf]
    rw [hOf_plain hpl, f128_add_zero]
    exact getElem!_pos comb i h2

/-- **The lincheck at a rep, at either shape.** When `GfResiduals`' residuals at w, on the S `structureOf` builds and
the v `verifierRows` gives at the rep's coins, are all zero, a proof carrying the lincheck messages w holds (`decoded`)
ends the verifier's lincheck rounds at the fold points `verifierRows` returns, from the `v_A`, `v_B` w holds, at the
comb w holds dotted with `z_partial`: the comb, plus `h` at a holo shape (`decoded_comb_get`). -/
theorem lincheck_of_residuals_dec {sh : Shape} {w comb : Array F128} {cons : Nat → Nat → F128}
    {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {rs : Array F128}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum kappa r cons)
    (hres : residuals S w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0) {p : LincheckProof}
    (hp : LcProofOf (decoded sh w comb cons) p)
    {C : Nat → Array F128} (hC : ∀ j < sh.kLog - 6, (C j).getD 0 0 = info.lc.fold.ts[j]!) :
    lcFold p C (sh.kLog - 6)
        (info.lc.fold.alpha * (decoded sh w comb cons).fa + (decoded sh w comb cons).fb + info.lc.fold.beta) =
      (dot (decoded sh w comb cons).comb p.zPartial, info.lc.fold.ts) := by
  obtain ⟨rs', k, hrun, hres'⟩ := residuals_eq_run w comb cons hrows hS hw hc hacc
  rw [hres] at hres'
  cases hres'
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, -, -, h1, -, -, hrs⟩ := run_stages hrun
  have hl := decoded_lc_size sh w comb cons
  rw [← hl]
  exact lincheck_checks h1 (hzero r1 (by rw [hrs]; simp)) hp (fun j hj => hC j (by rwa [hl] at hj))

/-- **The lincheck at a rep.** `lincheck_of_residuals_dec` at a plain shape, where the comb w holds is the comb: the
rounds end at `dot comb z_partial`, Piop.lean:148's check for `comb` the verifier's `CircuitFold.folded` at those
points, which is the comb port. -/
theorem lincheck_of_residuals {sh : Shape} {w comb : Array F128} {cons : Nat → Nat → F128}
    {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {rs : Array F128} (hpl : sh.holo = false)
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum kappa r cons)
    (hres : residuals S w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0) {p : LincheckProof}
    (hp : LcProofOf (decoded sh w comb cons) p)
    {C : Nat → Array F128} (hC : ∀ j < sh.kLog - 6, (C j).getD 0 0 = info.lc.fold.ts[j]!) :
    lcFold p C (sh.kLog - 6)
        (info.lc.fold.alpha * (decoded sh w comb cons).fa + (decoded sh w comb cons).fb + info.lc.fold.beta) =
      (dot comb p.zPartial, info.lc.fold.ts) := by
  have h := lincheck_of_residuals_dec hrows hS hw hc hacc hres hzero hp hC
  rwa [decoded_comb sh cons hpl hc] at h

end Flock.RecAlgebra
