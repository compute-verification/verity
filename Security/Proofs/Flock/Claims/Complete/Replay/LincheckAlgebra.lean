import Proofs.Flock.Claims.Complete.Replay.Satisfies
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Lincheck
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds

/-!
# The lincheck's algebra at the deployed relation

`LcAlgebra` at `Readable.relation`, from `RelS` alone: the honest prover's lincheck (`Lc.proof`) on the honest trace
passes the verifier's checks (§10).

* **The folded circuit** (`folded_ok`): `Stmt.folded` succeeds wherever `Stmt.fold` does (on a structured block
  `partialRanges` builds the slot types `foldRanges` built, `partialRanges_ok`), on the fold with `β` at the pin, halved
  at the round coins: the prover's `combT` at the last round, of width 64.
* **The start** (`start_eq`): `α·v_A + v_B + β = Σ_c comb[c]·zr[c]`. The fold is the statement's matrices at the
  zerocheck's point (`FoldRealizes.fold`, `comb`), the witness's folded columns are `hcol` (`zr0`), and `combSum_eq`
  reads `α·a + b + β·pin`, where `v_A`, `v_B` are the true `a`, `b` claims (`finalA_holds`: the trace's A side is
  `aVec`) and the pin's extension is `1` (the witness satisfies the statement).
* **The rounds** (`lcFold_claim`): each honest message moves the claim `Σ a·z` to the halved vectors' (`round_next`).
* `lcAlgebra`: at a good case the trace is the witness's bits (`traceOf_z`), its A and B sides are the statement's
  (`toZ_mulBlocks`, through `Case.model`'s `placedA = matA`), and `lcAlgebra_of` applies.
-/

namespace FlockLevel3

open Flock GF128

/-- Where `foldRanges` succeeds `partialRanges` does too: both look up or build the same slot types in the same
order. -/
theorem partialRanges_ok (st : Stmt) (alpha : F128) (lw rho rr : Array F128) : ∀ (rs : List Range)
    (bases : Array ((Kind × Nat) × Array F128)) (out res : Array F128),
    st.foldRanges alpha lw rho rs bases out = .ok res →
    ∀ out', ∃ res', st.partialRanges alpha lw rho rr rs bases out' = .ok res' := by
  intro rs
  induction rs with
  | nil => intro bases out res _ out'; exact ⟨out', rfl⟩
  | cons r rs ih =>
    intro bases out res h out'
    simp only [Stmt.foldRanges] at h
    simp only [Stmt.partialRanges]
    cases hf : bases.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
    | some kb =>
      rw [hf] at h
      obtain ⟨kb, b⟩ := kb
      simp only [bind, Except.bind, pure, Except.pure] at h ⊢
      exact ih _ _ _ h _
    | none =>
      rw [hf] at h
      cases hb : st.slotBase alpha lw rho (r.kind, r.slotLog) with
      | error e => rw [hb] at h; simp [bind, Except.bind] at h
      | ok b =>
        rw [hb] at h
        simp only [bind, Except.bind, pure, Except.pure] at h ⊢
        exact ih _ _ _ h _

/-- **`folded` succeeds wherever `fold` does**, on `fold`'s vector with `β` at the pin, halved at `ts` in turn. -/
theorem folded_ok (st : Stmt) (alpha beta : F128) (lw rho ts v : Array F128) (h : st.fold alpha lw rho = .ok v) :
    st.folded alpha beta lw rho ts = .ok (ts.foldl (fun c t => halve t c) (v.modify st.pin (· + beta))) := by
  suffices hw : ∃ w, st.folded alpha beta lw rho ts = .ok w by
    obtain ⟨w, hw⟩ := hw
    obtain ⟨v', hv', rfl⟩ := folded_eq st alpha beta lw rho ts w hw
    rw [h] at hv'
    obtain rfl := Except.ok.inj hv'
    exact hw
  unfold Stmt.folded
  split
  · simp only [Stmt.foldedPartial]
    rw [FlockSoundness.TmplFold.partialRangesT_eq st alpha lw rho ts.reverse _ #[] #[] _
      (by simp [FlockSoundness.TmplFold.CacheOk]) (by simp [FlockSoundness.TmplFold.CacheOk])]
    by_cases hck : (rho.size + 6 != st.c.kLog || lw.size != 64) = true
    · exfalso
      simp only [Stmt.fold] at h
      rw [ite_eq_left hck] at h
      simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    have hfr : ∃ res, st.foldRanges alpha lw rho st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0) = .ok res := by
      simp only [Stmt.fold] at h
      rw [ite_eq_right hck] at h
      cases hr : st.foldRanges alpha lw rho st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0) with
      | error e => rw [hr] at h; simp [bind, Except.bind] at h
      | ok res => exact ⟨res, rfl⟩
    obtain ⟨res, hr⟩ := hfr
    obtain ⟨res', hres'⟩ := partialRanges_ok st alpha lw rho ts.reverse _ #[] _ res hr (Array.replicate 64 0)
    rw [ite_eq_right hck]
    simp only [bind, Except.bind, pure, Except.pure, hres']
    exact ⟨_, rfl⟩
  · rw [h]
    exact ⟨_, rfl⟩

end FlockLevel3

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

namespace Lc

/-! ## The rounds -/

section Rounds

variable {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128} {cs : Challenges}

theorem combT_size (hf : full.size = 2 ^ c.kLog) : ∀ j, (combT c full cs j).size = 2 ^ c.kLog / 2 ^ j
  | 0 => by simp [combT, hf]
  | j + 1 => by
    show (halve (cs.τ j) (combT c full cs j)).size = _
    rw [halve_size, combT_size hf j, Nat.div_div_eq_div_mul, ← pow_succ]

theorem zrT_size : ∀ j, (zrT c m tr zc cs j).size = 2 ^ c.kLog / 2 ^ j
  | 0 => by simp [zrT, zr0]
  | j + 1 => by
    show (halve (cs.τ j) (zrT c m tr zc cs j)).size = _
    rw [halve_size, zrT_size j, Nat.div_div_eq_div_mul, ← pow_succ]

theorem lcFold_succ (p : LincheckProof) (C : ℕ → Array F128) (claim : F128) (j : ℕ) :
    Flock.RecAlgebra.lcFold p C (j + 1) claim = lStep p.rounds j (Flock.RecAlgebra.lcFold p C j claim) (C j) := by
  unfold Flock.RecAlgebra.lcFold
  rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]

theorem half_pow {K j : ℕ} (hj : j < K) : 2 ^ K / 2 ^ j = 2 * (2 ^ K / 2 ^ (j + 1)) := by
  rw [Nat.pow_div (by omega) (by norm_num), Nat.pow_div (by omega) (by norm_num),
    show K - j = K - (j + 1) + 1 by omega, pow_succ]
  ring

/-- **One honest round**: from the claim `Σ_{k<2H} a[k]·z[k]`, the verifier's step on the honest message `(q(1), q(∞))`
is the claim `Σ_{k<H}` of the vectors halved at the round's coin. -/
theorem round_eq (a z : Array F128) (H : ℕ) (ha : a.size = 2 * H) (hz : z.size = 2 * H) (t s : F128)
    (hs : GF128.of s = ∑ k ∈ Finset.range (2 * H), GF128.of a[k]! * GF128.of z[k]!) :
    GF128.of ((sumF (a.size / 2) fun k => (a[k]! + a[k + a.size / 2]!) * (z[k]! + z[k + a.size / 2]!)) * t * t +
        (s + (sumF (a.size / 2) fun k => a[k + a.size / 2]! * z[k + a.size / 2]!) +
          (sumF (a.size / 2) fun k => a[k + a.size / 2]! * z[k + a.size / 2]!) +
          (sumF (a.size / 2) fun k => (a[k]! + a[k + a.size / 2]!) * (z[k]! + z[k + a.size / 2]!))) * t +
        (s + (sumF (a.size / 2) fun k => a[k + a.size / 2]! * z[k + a.size / 2]!))) =
      ∑ k ∈ Finset.range H, GF128.of (halve t a)[k]! * GF128.of (halve t z)[k]! := by
  have hH : a.size / 2 = H := by omega
  have hHz : z.size / 2 = H := by omega
  have h1 : GF128.of (sumF (a.size / 2) fun k => a[k + a.size / 2]! * z[k + a.size / 2]!) =
      (Discharge.Composed.Lincheck.roundMsg H (fun k => GF128.of a[k]!) (fun k => GF128.of z[k]!)).1 := by
    simp only [Discharge.Composed.Lincheck.roundMsg, hH]
    rw [Zc.of_sumF]
    refine Finset.sum_congr rfl fun k _ => ?_
    show GF128.of (a[k + H]! * z[k + H]!) = _
    rw [of_mul]
  have h2 : GF128.of (sumF (a.size / 2) fun k => (a[k]! + a[k + a.size / 2]!) * (z[k]! + z[k + a.size / 2]!)) =
      (Discharge.Composed.Lincheck.roundMsg H (fun k => GF128.of a[k]!) (fun k => GF128.of z[k]!)).2 := by
    simp only [Discharge.Composed.Lincheck.roundMsg, hH]
    rw [Zc.of_sumF]
    refine Finset.sum_congr rfl fun k _ => ?_
    show GF128.of ((a[k]! + a[k + H]!) * (z[k]! + z[k + H]!)) = _
    rw [of_mul, of_add, of_add]
    ring
  have hfa : ∀ k ∈ Finset.range H, GF128.of (halve t a)[k]! * GF128.of (halve t z)[k]! =
      Model.foldTop (GF128.of t) H (fun k => GF128.of a[k]!) k *
        Model.foldTop (GF128.of t) H (fun k => GF128.of z[k]!) k := by
    intro k hk
    have hk' := Finset.mem_range.1 hk
    rw [halve_get t a k (by omega), halve_get t z k (by omega), hH, hHz]
    simp only [Model.foldTop]
    ring
  rw [Finset.sum_congr rfl hfa, ← Discharge.Composed.Lincheck.round_next H _ _ (GF128.of t)]
  simp only [of_add, of_mul]
  rw [h1, h2, hs]
  ring

/-- **The honest rounds**: after `j` rounds the verifier's claim, on the prover's messages, is `Σ a·z` of the circuit's
and the witness's columns halved at the first `j` round coins. -/
theorem lcFold_claim (hk : 6 ≤ c.kLog) (hf : full.size = 2 ^ c.kLog) (claim : F128)
    (h0 : GF128.of claim =
      ∑ k ∈ Finset.range (2 ^ c.kLog), GF128.of (combT c full cs 0)[k]! * GF128.of (zrT c m tr zc cs 0)[k]!) :
    ∀ j ≤ c.kLog - 6, GF128.of (Flock.RecAlgebra.lcFold (proof c m tr zc full cs) (fun j => #[cs.τ j]) j claim).1 =
      ∑ k ∈ Finset.range (2 ^ c.kLog / 2 ^ j), GF128.of (combT c full cs j)[k]! * GF128.of (zrT c m tr zc cs j)[k]!
  | 0, _ => by simpa [Flock.RecAlgebra.lcFold] using h0
  | j + 1, hj => by
    have ih := lcFold_claim hk hf claim h0 j (by omega)
    have h2H := half_pow (K := c.kLog) (j := j) (by omega)
    have hq : (proof c m tr zc full cs).rounds[j]! = msg c m tr zc full cs j := pushes_get _ _ j (by omega)
    have hg : (#[cs.τ j] : Array F128).getD 0 0 = cs.τ j := rfl
    rw [lcFold_succ]
    simp only [lStep, hq, hg, msg, combT, zrT]
    rw [h2H] at ih
    exact round_eq (combT c full cs j) (zrT c m tr zc cs j) (2 ^ c.kLog / 2 ^ (j + 1))
      (by rw [combT_size hf, h2H]) (by rw [zrT_size, h2H]) (cs.τ j) _ ih

theorem foldl_pushes (j : ℕ) :
    (pushes cs.τ j).foldl (fun acc t => halve t acc) (full.modify c.pin (· + cs.cb 0)) = combT c full cs j := by
  induction j with
  | zero => rfl
  | succ j ih => simp only [pushes, Array.foldl_push, ih, combT]

end Rounds

/-! ## The start -/

theorem bitF_toZ (b : Bool) : (if b then (1 : 𝔽) else 0) = ((Readable.toZ b).val : 𝔽) := by
  cases b
  · show (0 : 𝔽) = ((0 : ZMod 2).val : 𝔽)
    rw [show (0 : ZMod 2).val = 0 from rfl, Nat.cast_zero]
  · show (1 : 𝔽) = ((1 : ZMod 2).val : 𝔽)
    rw [show (1 : ZMod 2).val = 1 from rfl, Nat.cast_one]

/-- **The lincheck's algebra on a statement's trace** (§10). For a circuit fold `c` realizing `S` whose `folded` succeeds
where its `fold` does, a witness `w` of `S`'s pin and a trace whose bits are `w`'s and whose A and B sides are `aVec`,
`bVec`: at the zerocheck's output on that trace, wherever the circuit fold at the lincheck's `α` returns `full`, the
folded circuit is the prover's `combT` at the last round, of width 64, and the rounds' claim from `α·v_A + v_B + β`
ends at `dot comb z_partial`. -/
theorem lcAlgebra_of (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S)
    (hfd : ∀ (α β : F128) (lw ρ ts v : Array F128), c.fold α lw ρ = .ok v → ∃ w, c.folded α β lw ρ ts = .ok w)
    (m : ℕ) (hm : m = S.m) (w : Model.Witness S.m) (hpin : ∀ o, S.block w o S.pin = 1) (tr : Trace)
    (hz : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽))
    (ha : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.az a.val = ((aVec S w a).val : 𝔽))
    (hb : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.bz a.val = ((bVec S w a).val : 𝔽))
    (zcs : Zc.Challenges) (cs : Challenges) (full : Array F128)
    (hfold : c.fold (cs.ca 0) (skipWeights (Zc.out m zcs tr).z) ((Zc.out m zcs tr).rho.extract 0 (c.kLog - 6)) =
      .ok full) :
    ∃ comb, c.folded (cs.ca 0) (cs.cb 0) (skipWeights (Zc.out m zcs tr).z)
        ((Zc.out m zcs tr).rho.extract 0 (c.kLog - 6)) (pushes cs.τ (c.kLog - 6)) = .ok comb ∧ comb.size = 64 ∧
      (Flock.RecAlgebra.lcFold (proof c m tr (Zc.out m zcs tr) full cs) (fun j => #[cs.τ j]) (c.kLog - 6)
        (cs.ca 0 * (Zc.out m zcs tr).vA + (Zc.out m zcs tr).vB + cs.cb 0)).1 =
        dot comb (proof c m tr (Zc.out m zcs tr) full cs).zPartial := by
  subst hm
  have hk6 := S.hk6
  have hkm := S.hk
  have hK : c.kLog = S.kLog := hc.kLog
  set zc := Zc.out S.m zcs tr with hzc
  have hrho : zc.rho.size = S.m - 6 := pushes_size _ _
  obtain ⟨comb, hcomb⟩ := hfd (cs.ca 0) (cs.cb 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6))
    (pushes cs.τ (c.kLog - 6)) full hfold
  obtain ⟨v, hv, hvsz, hcv⟩ := hc.folded _ _ _ _ _ _ hcomb
  rw [hfold] at hv
  have hvf : v = full := (Except.ok.inj hv).symm
  rw [hvf] at hvsz hcv
  have hfull : full.size = 2 ^ c.kLog := by rw [hK]; exact hvsz
  have hcombT : comb = combT c full cs (c.kLog - 6) := by rw [hcv]; exact foldl_pushes _
  have h64 : 2 ^ c.kLog / 2 ^ (c.kLog - 6) = 64 := by
    rw [Nat.pow_div (by omega) (by norm_num), show c.kLog - (c.kLog - 6) = 6 by omega]
  refine ⟨comb, hcomb, by rw [hcombT, combT_size hfull, h64], ?_⟩
  have hρsz : (zc.rho.extract 0 (c.kLog - 6)).size = S.kLog - 6 := by
    rw [Array.size_extract, hrho, hK]; omega
  have hρv : vecOf (S.kLog - 6) (zc.rho.extract 0 (c.kLog - 6)) = fun j : Fin (S.kLog - 6) => fOf zc.rho[j.val]! := by
    funext j
    simp only [vecOf]
    rw [getElem!_pos _ _ (by rw [hρsz]; exact j.isLt), getElem!_pos _ _ (by rw [hrho]; omega),
      Array.getElem_extract]
    simp
  let zc' : Model.ZcOut 𝔽 S.m :=
    ⟨fOf zc.z, vecOf (S.m - 6) zc.rho, vecOf (S.m - 6) zc.rho, fOf zc.vA, fOf zc.vB, 0⟩
  have hU : ∀ k < 2 ^ S.kLog, GF128.of (combT c full cs 0)[k]! =
      Model.comb execArith S zc'.z (fOf (cs.ca 0)) (fOf (cs.cb 0)) (ρIn S zc') k := by
    intro k hk
    show GF128.of (full.modify c.pin (· + cs.cb 0))[k]! = _
    rw [modify_get! full c.pin k _ (by rw [hvsz]; exact hk), hc.pin]
    have hsum : GF128.of full[k]! = ∑ i : Fin (2 ^ S.kLog),
        execArith.lagS zc'.z ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
          Model.eqAt (ρIn S zc') (i.val / 64) *
            (fOf (cs.ca 0) * ((S.A₀ i ⟨k, hk⟩).val : 𝔽) + ((S.B₀ i ⟨k, hk⟩).val : 𝔽)) := by
      have hfr := hc.fold (cs.ca 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) full (skipWeights_size _)
        hρsz hfold ⟨k, hk⟩
      rw [hρv] at hfr
      exact hfr
    unfold Model.comb
    rw [dite_eq_left hk]
    by_cases hp : S.pin.val = k
    · rw [ite_eq_left hp, ite_eq_left hp.symm]
      show GF128.of (full[k]! + cs.cb 0) = _
      rw [of_add, hsum]
    · rw [ite_eq_right hp, ite_eq_right (Ne.symm hp), add_zero, hsum]
  have hlen : S.m - 6 - (c.kLog - 6) = S.nbl := by
    rw [hK]
    show S.m - 6 - (S.kLog - 6) = S.m - S.kLog
    omega
  have hE : ∀ o : Fin (2 ^ S.nbl), GF128.of (eqTable (zc.rho.extract (c.kLog - 6) (S.m - 6)))[o.val]! =
      Model.eqAt (zc'.ρOut S) o.val := by
    intro o
    rw [Zc.of_eqTable_extract zc.rho (c.kLog - 6) (S.m - 6) hrho.symm (by omega) o.val (by rw [hlen]; exact o.isLt)]
    unfold Discharge.Composed.Zerocheck.eqR Model.eqAt
    rw [hlen, ← Fin.prod_univ_eq_prod_range]
    refine Finset.prod_congr rfl fun j _ => ?_
    have hj : S.kLog - 6 + j.val < S.m - 6 := by
      have : j.val < S.m - S.kLog := j.isLt
      omega
    simp only [Model.ZcOut.ρOut, dite_eq_left hj, zc', vecOf, Discharge.Composed.Zerocheck.coin, hK]
  have hV : ∀ k < 2 ^ S.kLog, GF128.of (zrT c S.m tr zc cs 0)[k]! = hcol S (zc'.ρOut S) w k := by
    intro k hk
    show GF128.of (zr0 c S.m tr zc)[k]! = _
    unfold zr0
    rw [Discharge.Composed.Zerocheck.get_map_range _ (by rw [hK]; exact hk), Zc.of_sumF, eqTable_size,
      Array.size_extract, hrho, Nat.min_self, hlen, Finset.sum_range]
    unfold hcol
    rw [dite_eq_left hk]
    refine Finset.sum_congr rfl fun o _ => ?_
    have hidx : k + 2 ^ c.kLog * o.val = (S.addr o ⟨k, hk⟩).val := by
      simp only [Model.Statement.addr, hK]
      ring
    rw [← hz (S.addr o ⟨k, hk⟩), ← hE o, ← hidx]
    unfold Zc.bitF
    cases tr.z[k + 2 ^ c.kLog * o.val]!
    · rw [ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true, mul_zero]
      exact GF128.zero_eq.symm
    · rw [ite_eq_left rfl, ite_eq_left rfl, mul_one]
  have hsum := combSum_eq execArith S zc' w (fOf (cs.ca 0)) (fOf (cs.cb 0))
  have hA0 : (⟨zc'.va, execArith.lagS zc'.z, zc'.ρ⟩ : Model.Claim 𝔽 S.m).eval (aVec S w) =
      fOf (Discharge.Composed.Zerocheck.honest (S.m - 6) (Zc.bitF tr.az) (Zc.bitF tr.bz) (Zc.bitF tr.z) (zcs.r S.m)
        zc.rho zc.z).finalA :=
    Discharge.Composed.finalA_holds S.m (by omega) (Zc.bitF tr.az) (Zc.bitF tr.bz) (Zc.bitF tr.z) (aVec S w) ha
      (zcs.r S.m) zc.rho zc.z
  have hB0 : (⟨zc'.vb, execArith.lagS zc'.z, zc'.ρ⟩ : Model.Claim 𝔽 S.m).eval (bVec S w) =
      fOf (Discharge.Composed.Zerocheck.honest (S.m - 6) (Zc.bitF tr.az) (Zc.bitF tr.bz) (Zc.bitF tr.z) (zcs.r S.m)
        zc.rho zc.z).finalB :=
    Discharge.Composed.finalB_holds S.m (by omega) (Zc.bitF tr.az) (Zc.bitF tr.bz) (Zc.bitF tr.z) (bVec S w) hb
      (zcs.r S.m) zc.rho zc.z
  have hvA : fOf zc.vA = fOf (Discharge.Composed.Zerocheck.honest (S.m - 6) (Zc.bitF tr.az) (Zc.bitF tr.bz)
      (Zc.bitF tr.z) (zcs.r S.m) zc.rho zc.z).finalA :=
    Zc.of_tab (S.m - 6) zcs.z zcs.ρ tr.az zc.rho (pushes_get _ _) (S.m - 6) le_rfl 0 (by simp)
  have hvB : fOf zc.vB = fOf (Discharge.Composed.Zerocheck.honest (S.m - 6) (Zc.bitF tr.az) (Zc.bitF tr.bz)
      (Zc.bitF tr.z) (zcs.r S.m) zc.rho zc.z).finalB :=
    Zc.of_tab (S.m - 6) zcs.z zcs.ρ tr.bz zc.rho (pushes_get _ _) (S.m - 6) le_rfl 0 (by simp)
  have hP : pinMLE S w (zc'.ρOut S) = 1 := by
    unfold pinMLE
    simp only [hpin, show (1 : ZMod 2).val = 1 from rfl, Nat.cast_one, mul_one]
    rw [Fin.sum_univ_eq_sum_range (fun i => Model.eqAt (zc'.ρOut S) i)]
    exact sum_eqAt _ _
  rw [hA0, hB0, ← hvA, ← hvB, hP] at hsum
  have hterm : ∀ k ∈ Finset.range (2 ^ S.kLog),
      GF128.of (combT c full cs 0)[k]! * GF128.of (zrT c S.m tr zc cs 0)[k]! =
        Model.comb execArith S zc'.z (fOf (cs.ca 0)) (fOf (cs.cb 0)) (ρIn S zc') k * hcol S (zc'.ρOut S) w k :=
    fun k hk => by rw [hU k (Finset.mem_range.1 hk), hV k (Finset.mem_range.1 hk)]
  have h0 : GF128.of (cs.ca 0 * zc.vA + zc.vB + cs.cb 0) =
      ∑ k ∈ Finset.range (2 ^ c.kLog), GF128.of (combT c full cs 0)[k]! * GF128.of (zrT c S.m tr zc cs 0)[k]! := by
    rw [hK, Finset.sum_congr rfl hterm, hsum, of_add, of_add, of_mul, mul_one]
  have hfin := lcFold_claim (m := S.m) (tr := tr) (zc := zc) (cs := cs) (by omega) hfull _ h0 (c.kLog - 6) le_rfl
  rw [h64] at hfin
  apply Discharge.Composed.Zerocheck.eq_of_of
  rw [hfin, Discharge.Composed.ZerocheckMasked.of_dot_range _ _ 64 (by rw [hcombT, combT_size hfull, h64]), hcombT]
  rfl

end Lc

/-! ## At the deployed relation -/

variable {H : Claims.Msg → Claims.Msg}

/-- The A side of the trace of a witness's bits, as a bit of GF(2): `aVec` of the statement whose matrices are its
table's. -/
theorem toZ_mulBlocks (st : Stmt) (hwf : StmtWF st) (w : Claims.Flock.Wit)
    (M : Matrix (Fin (2 ^ st.c.kLog)) (Fin (2 ^ st.c.kLog)) (ZMod 2)) (entry : ℕ → ℕ → Bool)
    (he : ∀ r j : Fin (2 ^ st.c.kLog), entry r.val j.val = Readable.ofZ (M r j))
    (a : Fin (2 ^ (stmtOf st hwf (regionsOf st)).m)) :
    Readable.toZ (mulBlocks entry st.c.kLog (traceOf st w.hid.toArray 0).z)[a.val]! =
      (M.mulVec ((stmtOf st hwf (regionsOf st)).block (Readable.zOf 1 w 0 st.m)
        ⟨a.val / 2 ^ st.c.kLog, (stmtOf st hwf (regionsOf st)).block_index_lt a⟩))
        ⟨a.val % 2 ^ st.c.kLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩ := by
  have hk := hwf.hk
  have hzs : (traceOf st w.hid.toArray 0).z.size = 2 ^ st.m := by simp [traceOf]
  have ha : a.val < 2 ^ st.m := a.isLt
  rw [mulBlocks_get _ _ _ (by rw [hzs]; exact ha), Readable.toZ_xsum]
  simp only [Matrix.mulVec, dotProduct]
  rw [Finset.sum_range]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [he ⟨a.val % 2 ^ st.c.kLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩ r, Readable.toZ_ofZ,
    traceOf_z st w.hid (block_lt hk ha r.isLt)]
  show _ = _ * Readable.toZ (w.hid.getD ((a.val / 2 ^ st.c.kLog * 2 ^ st.c.kLog + r.val) * 1 + 0) false)
  rw [Nat.mul_one, Nat.add_zero]

/-- **The lincheck's algebra at the deployed relation**, from `RelS` alone. -/
theorem lcAlgebra (H : Claims.Msg → Claims.Msg) : LcAlgebra H Readable.relation := by
  intro o i hi r _ hfold
  obtain ⟨st, hsts, hwf, -, hmat, hsat⟩ := o.model
  have hsz : o.tabs.size = 1 := by simp [Run.tabs, FlockDeploy.tabsOf, hsts]
  obtain rfl : i = 0 := by omega
  have ht : o.tabs[0]! = Setup.ofCircuit st := by simp [Run.tabs, FlockDeploy.tabsOf, hsts]
  have hc : FoldRealizes o.tabs[0]!.fold (stmtOf st hwf (regionsOf st)) := by
    rw [ht]
    exact stmtOf_fold st hwf _
  have hfd : ∀ (α β : F128) (lw ρ ts v : Array F128), o.tabs[0]!.fold.fold α lw ρ = .ok v →
      ∃ w, o.tabs[0]!.fold.folded α β lw ρ ts = .ok w := by
    rw [ht]
    exact fun α β lw ρ ts v h => ⟨_, folded_ok st α β lw ρ ts v h⟩
  have hm : o.tabs[0]!.m = (stmtOf st hwf (regionsOf st)).m := by rw [ht]; rfl
  have htr := o.trace_one hsts
  have hzb : ∀ a : Fin (2 ^ (stmtOf st hwf (regionsOf st)).m),
      Zc.bitF (o.trace 0).z a.val = ((Readable.zOf 1 o.w 0 st.m a).val : 𝔽) := by
    intro a
    rw [htr]
    unfold Zc.bitF
    rw [traceOf_z st o.w.hid a.isLt, Lc.bitF_toZ]
    show _ = ((Readable.toZ (o.w.hid.getD (a.val * 1 + 0) false)).val : 𝔽)
    rw [Nat.mul_one, Nat.add_zero]
  have hab : ∀ a : Fin (2 ^ (stmtOf st hwf (regionsOf st)).m),
      Zc.bitF (o.trace 0).az a.val = ((aVec (stmtOf st hwf (regionsOf st)) (Readable.zOf 1 o.w 0 st.m) a).val : 𝔽) ∧
      Zc.bitF (o.trace 0).bz a.val = ((bVec (stmtOf st hwf (regionsOf st)) (Readable.zOf 1 o.w 0 st.m) a).val : 𝔽) := by
    intro a
    rw [htr]
    unfold Zc.bitF
    rw [Lc.bitF_toZ, Lc.bitF_toZ]
    refine ⟨congrArg (fun x : ZMod 2 => (x.val : 𝔽))
        (toZ_mulBlocks st hwf o.w (stmtOf st hwf (regionsOf st)).A₀ (entryA st) (fun r j => ?_) a),
      congrArg (fun x : ZMod 2 => (x.val : 𝔽))
        (toZ_mulBlocks st hwf o.w (stmtOf st hwf (regionsOf st)).B₀ (entryB st) (fun r j => ?_) a)⟩
    · show entryA st r.val j.val = Readable.ofZ (matA st r.val j.val)
      rw [entryA_eq st r.isLt j.isLt, ← (hmat r j).1]
      simp only [Readable.Tab.a, Readable.Tab.ofStmt, r.isLt, j.isLt, and_self, ↓reduceDIte]
    · show entryB st r.val j.val = Readable.ofZ (matB st r.val j.val)
      rw [entryB_eq st r.isLt j.isLt, ← (hmat r j).2]
      simp only [Readable.Tab.b, Readable.Tab.ofStmt, r.isLt, j.isLt, and_self, ↓reduceDIte]
  exact Lc.lcAlgebra_of _ _ hc hfd _ hm _ hsat.2.1 _ hzb (fun a => (hab a).1) (fun a => (hab a).2) _ _ _ hfold

/-- **The lincheck goal at the deployed relation**, at every rep the session walk aligns. -/
theorem lincheck_relation (H : Claims.Msg → Claims.Msg)
    (hzc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r) : Lincheck H Readable.relation :=
  lincheck_goal' H hzc hlc (lcAlgebra H)

end FlockComplete.Replay
