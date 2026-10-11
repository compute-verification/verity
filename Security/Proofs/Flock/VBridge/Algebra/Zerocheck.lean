import Proofs.Flock.VBridge.Algebra.Eval
import Proofs.Flock.Soundness.Refine.Zerocheck

/-!
# V*'s zerocheck residual is zero only when the zerocheck's checks pass

`zerocheck_checks`: when `zerocheck`'s residual `claim + v_A·v_B` (rec_algebra.py:587–600) is zero over F, the two
`check`s of the verifier's `bindAndZerocheck` (Piop.lean:84, 98) hold on a proof whose zerocheck messages are the ones
`zerocheck` read, at `zerocheck`'s points: `P_c(z) = final_c` when `final_c` is V*'s `v_C` (which no round absorbs),
and the residual rounds' claim is `final_a · final_b`. The rounds are the fold of `Refine.zStep`, which
`Refine.forIn_rounds` shows the verifier's loop computes, on any coins whose first entries are `zerocheck`'s challenges.

`zerocheck_coins` says which rounds `zerocheck` draws its points from, and `zerocheck_of_residuals` is the stage at a
rep: when `GfResiduals`' residuals on the S `structureOf` builds and the v `verifierRows` gives are all zero, the checks
hold on the messages w holds (`decoded`), at the points `verifierRows` returns.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-! ## `CM` -/

theorem cm_bind {A B : Type} {x : CM A} {f : A → CM B} {k : Nat} {src : Coins} {r : B × Nat}
    (h : (x >>= f) k src = .ok r) : ∃ a k₁, x k src = .ok (a, k₁) ∧ f a k₁ src = .ok r := by
  have e : (x >>= f) k src = (x k src).bind (fun p => f p.1 p.2 src) := rfl
  rw [e] at h
  cases hx : x k src with
  | error e => rw [hx] at h; cases h
  | ok p => rw [hx] at h; exact ⟨p.1, p.2, rfl, h⟩

theorem cm_pure {A : Type} {a : A} {k : Nat} {src : Coins} {r : A × Nat}
    (h : (Pure.pure a : CM A) k src = .ok r) : r = (a, k) := by
  cases h; rfl

theorem coinVec_ok {n k k' : Nat} {src : Coins} {v : Array F128} (h : coinVec n k src = .ok (v, k')) :
    (n = 0 ∧ v = #[] ∧ k' = k) ∨ (n ≠ 0 ∧ src.round k n = .ok v ∧ k' = k + 1) := by
  unfold coinVec at h
  split at h
  · rename_i hn; cases h; exact .inl ⟨hn, rfl, rfl⟩
  · rename_i hn
    change (match src.round k n with
      | .ok r => (set (k + 1) >>= fun _ => pure r : CM (Array F128))
      | .error e => (throw e : CM (Array F128))) k src = _ at h
    cases hr : src.round k n with
    | error e => rw [hr] at h; cases h
    | ok r => rw [hr] at h; cases h; exact .inr ⟨hn, rfl, rfl⟩

theorem coinOne_ok {k k' : Nat} {src : Coins} {x : F128} (h : coinOne k src = .ok (x, k')) :
    ∃ c, src.round k 1 = .ok c ∧ x = c[0]! ∧ k' = k + 1 := by
  obtain ⟨v, k₁, hv, h⟩ := cm_bind h
  cases cm_pure h
  rcases coinVec_ok hv with ⟨h1, -, -⟩ | ⟨-, hr, rfl⟩
  · cases h1
  · exact ⟨v, hr, rfl, rfl⟩

theorem liftE_ok {A : Type} {e : Except String A} {k : Nat} {src : Coins} {r : A × Nat}
    (h : liftE e k src = .ok r) : ∃ a, e = .ok a ∧ r = (a, k) := by
  cases e with
  | error e => cases h
  | ok a => cases h; exact ⟨a, rfl, rfl⟩

/-! ## `run`'s stages -/

section Stages

variable {α : Type} [Val α]

/-- `run`'s residuals are its stages': the zerocheck's, the lincheck's, the ring switches', Ligerito's and the absorbed
`T`'s, each stage run on the coins the one before left. -/
theorem run_stages {sh : Shape} {M : Messages α} {extras : Array Claim} {k k' : Nat} {src : Coins}
    {rs : Array α} {info : Info α} (h : run sh M extras k src = .ok ((rs, info), k')) :
    sh.claims = 2 + extras.size ∧ ∃ (r0 r1 : α) (rsr lgr : Array α) (k1 k2 k3 : Nat),
      zerocheck sh.m M k src = .ok ((r0, info.zc), k1) ∧
      lincheck sh.kLog M info.zc k1 src = .ok ((r1, info.lc), k2) ∧
      ringSwitch M (claimsOf sh.m sh.kLog info.zc info.lc extras) k2 src = .ok ((rsr, info.rs), k3) ∧
      ligerito sh.sched M (M.tAtom.getD info.rs.T) info.rs.sws k3 src = .ok ((lgr, info.lg), k') ∧
      rs = #[r0, r1] ++ rsr ++ lgr ++ #[Val.add (M.tAtom.getD info.rs.T) info.rs.T] := by
  unfold run at h
  split at h
  · cases h
  · rename_i hcl
    obtain ⟨⟨r0, zc⟩, k1, h1, h⟩ := cm_bind h
    obtain ⟨⟨r1, lc⟩, k2, h2, h⟩ := cm_bind h
    obtain ⟨⟨rsr, rsp⟩, k3, h3, h⟩ := cm_bind h
    obtain ⟨⟨lgr, lg⟩, k4, h4, h⟩ := cm_bind h
    cases cm_pure h
    exact ⟨Decidable.of_not_not hcl, r0, r1, rsr, lgr, k1, k2, k3, h1, h2, h3, h4, rfl⟩

end Stages

/-! ## The zerocheck over F -/

theorem f128_eq_of_add_eq_zero {x y : F128} (h : x + y = 0) : x = y := by
  have e : x = x + y + y := by
    apply fOf_injective
    simp only [fOf_add, add_assoc, CharTwo.add_self_eq_zero, add_zero]
  rw [e, h]
  apply fOf_injective
  simp only [fOf_add, fOf_zero, zero_add]

theorem foldl_zip_eq_range (f : F128 → F128 → F128 → F128) :
    ∀ (a b : List F128) (acc : F128),
      (a.zip b).foldl (fun acc cx => f acc cx.1 cx.2) acc =
        (List.range (min a.length b.length)).foldl (fun acc i => f acc a[i]! b[i]!) acc
  | [], _, acc => by simp
  | _ :: _, [], acc => by simp
  | x :: a, y :: b, acc => by
    rw [List.zip_cons_cons, List.foldl_cons, foldl_zip_eq_range f a b, List.length_cons, List.length_cons,
      Nat.add_min_add_right, List.range_succ_eq_map, List.foldl_cons, List.foldl_map]
    rfl

/-- `dotV` over F is the verifier's `dot`. -/
theorem dotV_eq_dot (cs xs : Array F128) : dotV cs xs = dot cs xs := by
  rw [dot_eq_foldl]
  unfold dotV
  rw [← Array.foldl_toList, Array.toList_zip]
  simp only [Val.add, Val.sm, Val.zero]
  rw [foldl_zip_eq_range (fun acc c x => acc + c * x), Array.length_toList, Array.length_toList]
  simp only [Array.getElem!_toList]

/-- The zerocheck's claim before its residual rounds (`pcomb + pc`, Piop.lean:84–88), at `z`. -/
def zcStart (p : ZerocheckProof) (z : F128) : F128 :=
  dot (combWeights z) ((Array.range 64).map fun i => p.round1ab[i]! + p.round1c[i]!) +
    dot (lagrange nodesL den6Inv z) p.round1c

/-- The residual rounds (Piop.lean:90–98) on round `i`'s coins `C i`: the fold of `Refine.zStep` from `claim`. -/
def zcFold (r : Array F128) (p : ZerocheckProof) (C : Nat → Array F128) (n : Nat) (claim : F128) :
    F128 × Array F128 :=
  (List.range' 0 n).foldl (fun s i => zStep r p.rounds i s (C i)) (claim, #[])

/-- A zerocheck proof (`Proof.lean`) carrying messages `M`, and `vc` for `final_c`. -/
structure ZcProofOf (M : Messages F128) (vc : F128) (p : ZerocheckProof) : Prop where
  ab : p.round1ab = M.ab
  c : p.round1c = M.c
  rounds : p.rounds = M.zc
  fa : p.finalA = M.fa
  fb : p.finalB = M.fb
  fc : p.finalC = vc

section Coins

variable {α : Type} [Val α]

/-- The residual rounds draw one round each, its first coin the round's challenge. -/
theorem zcRounds_coins (r : Array F128) :
    ∀ (l : List (α × α)) (i : Nat) (claim : α) (rho : Array F128) (k k' : Nat) (src : Coins) (claim' : α)
      (rho' : Array F128),
      zcRounds r l i claim rho k src = .ok ((claim', rho'), k') →
      rho'.size = rho.size + l.length ∧ (∀ j < rho.size, rho'[j]! = rho[j]!) ∧ k' = k + l.length ∧
      ∀ j < l.length, ∃ c, src.round (k + j) 1 = .ok c ∧ rho'[rho.size + j]! = c[0]!
  | [], i, claim, rho, k, k', src, claim', rho', h => by
    cases cm_pure h
    exact ⟨rfl, fun _ _ => rfl, rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  | (g1, ginf) :: rest, i, claim, rho, k, k', src, claim', rho', h => by
    simp only [zcRounds] at h
    obtain ⟨rr, k₁, h1, h⟩ := cm_bind h
    obtain ⟨c, hc, rfl, rfl⟩ := coinOne_ok h1
    obtain ⟨hsz, hpre, hk, hco⟩ := zcRounds_coins r rest (i + 1) _ (rho.push c[0]!) _ _ src _ _ h
    simp only [Array.size_push, List.length_cons] at hsz hpre hk hco ⊢
    refine ⟨by omega, fun j hj => ?_, by omega, fun j hj => ?_⟩
    · rw [hpre j (by omega), getElem!_pos _ j (by simp; omega), Array.getElem_push_lt, getElem!_pos _ j hj]
    · rcases j with _ | j
      · refine ⟨c, by simpa using hc, ?_⟩
        rw [Nat.add_zero, hpre rho.size (by omega), getElem!_pos _ rho.size (by simp), Array.getElem_push_eq]
      · obtain ⟨c', hc', he⟩ := hco j (by omega)
        exact ⟨c', by rw [show k + (j + 1) = k + 1 + j by omega]; exact hc',
          by rw [show rho.size + (j + 1) = rho.size + 1 + j by omega]; exact he⟩

/-- **The zerocheck's coins.** The rounds `zerocheck` draws its points from, when `r_outer` is not empty: `r_skip`
(6 coins, unused), `r_outer` (`m − 13`), `z`, then one round per residual round, its challenge. -/
theorem zerocheck_coins {m : Nat} (hm : 14 ≤ m) {M : Messages α} {k k' : Nat} {src : Coins} {r0 : α}
    {zc : ZcPts α} (h : zerocheck m M k src = .ok ((r0, zc), k')) :
    ∃ c₀ c₁ c₂, src.round k 6 = .ok c₀ ∧ src.round (k + 1) (m - 13) = .ok c₁ ∧ src.round (k + 2) 1 = .ok c₂ ∧
      zc.r = pinned ++ c₁ ∧ zc.z = c₂[0]! ∧ zc.rho.size = M.zc.size ∧ k' = k + 3 + M.zc.size ∧
      ∀ j < M.zc.size, ∃ c, src.round (k + 3 + j) 1 = .ok c ∧ zc.rho[j]! = c[0]! := by
  unfold zerocheck at h
  obtain ⟨c₀, k₁, h1, h⟩ := cm_bind h
  obtain ⟨c₁, k₂, h2, h⟩ := cm_bind h
  obtain ⟨z, k₃, h3, h⟩ := cm_bind h
  obtain ⟨⟨claim, rho⟩, k₄, h4, h⟩ := cm_bind h
  obtain ⟨ab, k₅, h5, h⟩ := cm_bind h
  cases cm_pure h
  obtain ⟨_, -, he⟩ := liftE_ok h5
  cases he
  rcases coinVec_ok h1 with ⟨h6, -⟩ | ⟨-, hr₀, rfl⟩
  · cases h6
  rcases coinVec_ok h2 with ⟨h7, -⟩ | ⟨-, hr₁, rfl⟩
  · omega
  obtain ⟨c₂, hr₂, rfl, rfl⟩ := coinOne_ok h3
  obtain ⟨hsz, -, hk, hco⟩ := zcRounds_coins _ M.zc.toList 0 _ #[] _ _ src _ _ h4
  simp only [Array.length_toList, Array.size_empty, Nat.zero_add] at hsz hk hco
  exact ⟨c₀, c₁, c₂, hr₀, hr₁, hr₂, rfl, rfl, hsz, hk, hco⟩

end Coins

/-- The residual rounds over F are the fold of `Refine.zStep` on round `i`'s messages and on coins `C` whose first
entries are the challenges drawn. -/
theorem zcRounds_fold (r : Array F128) (rounds : Array (F128 × F128)) (C : Nat → Array F128) :
    ∀ (l : List (F128 × F128)) (i : Nat) (claim : F128) (rho : Array F128) (k k' : Nat) (src : Coins)
      (claim' : F128) (rho' : Array F128),
      zcRounds r l i claim rho k src = .ok ((claim', rho'), k') →
      (∀ j (hj : j < l.length), rounds[i + j]! = l[j]) →
      (∀ j < l.length, (C (i + j)).getD 0 0 = rho'[rho.size + j]!) →
      (List.range' i l.length).foldl (fun s j => zStep r rounds j s (C j)) (claim, rho) = (claim', rho')
  | [], i, claim, rho, k, k', src, claim', rho', h, _, _ => by
    cases cm_pure h
    rfl
  | (g1, ginf) :: rest, i, claim, rho, k, k', src, claim', rho', h, hl, hC => by
    simp only [zcRounds] at h
    obtain ⟨rr, k₁, h1, h⟩ := cm_bind h
    obtain ⟨c, -, rfl, rfl⟩ := coinOne_ok h1
    obtain ⟨-, hpre, -, -⟩ := zcRounds_coins r rest (i + 1) _ (rho.push c[0]!) _ _ src _ _ h
    simp only [Array.size_push] at hpre
    have hlast : rho'[rho.size]! = c[0]! := by
      rw [hpre rho.size (by omega), getElem!_pos _ rho.size (by simp), Array.getElem_push_eq]
    simp only [List.length_cons] at hl hC ⊢
    rw [List.range'_succ, List.foldl_cons]
    have hg : rounds[i]! = (g1, ginf) := by simpa using hl 0 (by simp)
    have hC0 : (C i).getD 0 0 = c[0]! := by simpa [hlast] using hC 0 (by simp)
    have hstep : zStep r rounds i (claim, rho) (C i) =
        (Val.add (Val.add (Val.sm ((1 : F128) + c[0]!) (Val.sm (1 + r[i]!).inv (Val.add claim (Val.sm r[i]! g1))))
          (Val.sm c[0]! g1)) (Val.sm (c[0]! * ((1 : F128) + c[0]!)) ginf), rho.push c[0]!) := by
      simp only [zStep, hg, hC0, Val.add, Val.sm]
      congr 1
      (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_one]; ring)
    rw [hstep]
    refine zcRounds_fold r rounds C rest (i + 1) _ (rho.push c[0]!) _ _ src _ _ h (fun j hj => ?_) (fun j hj => ?_)
    · have e := hl (j + 1) (by omega)
      rw [show i + (j + 1) = i + 1 + j by omega, List.getElem_cons_succ] at e
      exact e
    · rw [Array.size_push, show i + 1 + j = i + (j + 1) by omega, show rho.size + 1 + j = rho.size + (j + 1) by omega]
      exact hC (j + 1) (by omega)

/-- **The zerocheck's checks.** When `zerocheck` over F returns residual zero, a proof carrying its messages, with V*'s
`v_C` for `final_c`, passes both `check`s of `bindAndZerocheck` at its points: `P_c(z) = final_c` (Piop.lean:84), and
on any coins `C` whose first entries are its challenges `ρ`, the residual rounds end at `final_a · final_b`
(Piop.lean:98) with challenges `ρ`. -/
theorem zerocheck_checks {m : Nat} {M : Messages F128} {k k' : Nat} {src : Coins} {r0 : F128} {zc : ZcPts F128}
    (h : zerocheck m M k src = .ok ((r0, zc), k')) (h0 : r0 = 0) {p : ZerocheckProof} (hp : ZcProofOf M zc.vc p)
    (hsab : M.ab.size = 64) (hsc : M.c.size = 64) (hszc : M.zc.size = m - 6)
    {C : Nat → Array F128} (hC : ∀ i < m - 6, (C i).getD 0 0 = zc.rho[i]!) :
    dot (lagrange nodesL den6Inv zc.z) p.round1c = p.finalC ∧
      zcFold zc.r p C (m - 6) (zcStart p zc.z) = (p.finalA * p.finalB, zc.rho) := by
  unfold zerocheck at h
  obtain ⟨_, k₁, -, h⟩ := cm_bind h
  obtain ⟨rOut, k₂, -, h⟩ := cm_bind h
  obtain ⟨z, k₃, -, h⟩ := cm_bind h
  obtain ⟨⟨claim, rho⟩, k₄, h4, h⟩ := cm_bind h
  obtain ⟨ab, k₅, h5, h⟩ := cm_bind h
  cases cm_pure h
  obtain ⟨ab', hab, he⟩ := liftE_ok h5
  cases he
  have hmv : ab = p.finalA * p.finalB := by
    cases hab; rw [hp.fa, hp.fb]
  have hfold := zcRounds_fold (pinned ++ rOut) p.rounds C M.zc.toList 0 _ #[] _ _ src _ _ h4
  simp only [Array.length_toList, Array.size_empty, Nat.zero_add] at hfold
  refine ⟨by rw [hp.fc, hp.c, dotV_eq_dot], ?_⟩
  unfold zcFold
  rw [← hszc]
  have hstart : zcStart p z =
      Val.add (dotV (combWeights z) (M.ab.zipWith Val.add M.c)) (dotV (lagrange nodesL den6Inv z) M.c) := by
    unfold zcStart
    rw [dotV_eq_dot, dotV_eq_dot, hp.ab, hp.c]
    congr 2
    apply Array.ext
    · simp [hsab, hsc]
    · intro i h1 h2
      simp only [Array.size_map, Array.size_range] at h1
      simp only [Array.getElem_map, Array.getElem_range, Array.getElem_zipWith, Val.add]
      rw [getElem!_pos _ i (by omega), getElem!_pos _ i (by omega)]
  rw [hstart, hfold (fun j hj => by rw [hp.rounds, getElem!_pos _ j (by simpa using hj),
      Array.getElem_toList]) (fun j hj => hC j (by simpa [hszc] using hj))]
  rw [← hmv, f128_eq_of_add_eq_zero h0]

/-! ## The zerocheck at a rep -/

theorem decoded_sizes (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) :
    (decoded sh w comb cons).ab.size = 64 ∧ (decoded sh w comb cons).c.size = 64 ∧
      (decoded sh w comb cons).zc.size = sh.m - 6 := by
  simp [decoded, Messages.map, symbolic]

/-- **The zerocheck at a rep.** When `GfResiduals`' residuals at w, on the S `structureOf` builds and the v
`verifierRows` gives at the rep's coins, are all zero, a proof carrying the zerocheck messages w holds (`decoded`), with
V*'s `v_C` for `final_c`, passes both `check`s of `bindAndZerocheck` at the points `verifierRows` returns. -/
theorem zerocheck_of_residuals {sh : Shape} {w comb : Array F128} {cons : Nat → Nat → F128}
    {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {rs : Array F128} (hpl : sh.holo = false)
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum kappa r cons)
    (hres : residuals S w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0) {p : ZerocheckProof}
    (hp : ZcProofOf (decoded sh w comb cons) (info.zc.vc.eval (valOf sh w comb cons)) p)
    {C : Nat → Array F128} (hC : ∀ i < sh.m - 6, (C i).getD 0 0 = info.zc.rho[i]!) :
    dot (lagrange nodesL den6Inv info.zc.z) p.round1c = p.finalC ∧
      zcFold info.zc.r p C (sh.m - 6) (zcStart p info.zc.z) = (p.finalA * p.finalB, info.zc.rho) := by
  obtain ⟨rs', k, hrun, hres'⟩ := residuals_eq_run w comb cons hpl hrows hS hw hc hacc
  rw [hres] at hres'
  cases hres'
  obtain ⟨-, r0, r1, rsr, lgr, k1, -, -, h0, -, -, -, hrs⟩ := run_stages hrun
  obtain ⟨hsab, hsc, hszc⟩ := decoded_sizes sh w comb cons
  exact zerocheck_checks h0 (hzero r0 (by rw [hrs]; simp)) hp hsab hsc hszc hC

end Flock.RecAlgebra
