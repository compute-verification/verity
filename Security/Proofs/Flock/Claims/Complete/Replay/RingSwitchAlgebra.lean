import Proofs.Flock.Claims.Complete.Replay.LincheckAlgebra
import Proofs.Flock.Claims.Complete.Replay.RingSwitch
import Proofs.Flock.Soundness.LinkSound
import Proofs.Flock.Soundness.Discharge.Layout.Region
import Proofs.Flock.Soundness.Discharge.Composed.Inner.RingSwitch
import Proofs.Flock.Soundness.Discharge.Composed.Inner.LincheckReplay
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckGlue

/-!
# The ring switch's algebra at the deployed relation

`RsAlgebra` at `Readable.relation`, from `RelS` alone: every claim a rep opens has the value of the witness's extension
at its point, so its switched vector of the honest trace meets it (`switch_of`: `Σ_v w[v]·s[v]` is the claim's
evaluation, `claim_eval_eq` and `fOf_claimCheck`, with `s[v]` the honest `ŝ[v]`).

* **Lincheck's claim** (`ab_eval`): `w = Σ_s L_s(r')·zr[s]`, and `zr` is the witness's columns folded at the outer
  coins (`zr0_hcol`) and halved at the round coins (`foldAll_lcV`): `Σ_s L_s(r')·z_partial[s]`, the claim's evaluation
  (`sum_zPartialOf`).
* **Zerocheck's claim** (`c_claim_eval`): `P_c(ζ)` is the honest `final_c` (`c_eval`, on the prover's `round1_c`,
  `Zc.r1c_eq`), the claim's evaluation (`finalC_eq_eval`).
* **The extra claims** (`Region.regionClaim_holds`): a region claim at any link point evaluates to its value when the
  witness's region bits are the region's (`Satisfies`): the in-block part of its point vanishes off the region's
  addresses (`sum_low`), and on them it is the free coordinates' weight. Each extra claim is the model's
  (`regionClaim_rel'`) at link points of `m_pts` coordinates, which `Record.decodeWith` checks (S9,
  `decodeWith_points`).
-/

namespace FlockComplete.Replay.Region

open FlockSoundness FlockSoundness.Model

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] [CharP F 2]

theorem idxOf?_get {α : Type} [BEq α] [LawfulBEq α] {l : List α} (hl : l.Nodup) (t : Fin l.length) :
    l.idxOf? (l.get t) = some t.val := by
  rw [List.idxOf?_eq_some_iff]
  refine ⟨t.isLt, rfl, fun j hj h => ?_⟩
  have := (List.Nodup.get_inj_iff hl).1 (show l.get ⟨j, by omega⟩ = l.get t from h)
  exact Nat.ne_of_lt hj (congrArg Fin.val this)

omit [Fintype F] [DecidableEq F] in
theorem fixed_factor (b f : Bool) :
    (if b then (if f then (1 : F) else 0) else 1 + if f then 1 else 0) = if b = f then 1 else 0 := by
  cases b <;> cases f <;> simp [CharTwo.add_self_eq_zero]

variable (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (r : Fin mPts → F)

/-- A region point's in-block coordinates. -/
def lowPoint : Fin S.kLog → F := fun j => regionPoint S R ptLocal mPts r ⟨j.val, lt_of_lt_of_le j.isLt S.hk⟩

/-- The link coordinates a region's free bits read. -/
def freePoint : Fin R.free.length → F := fun t => if ht : t.val < mPts then r ⟨t.val, ht⟩ else 0

/-- The link coordinates a region's block bits read. -/
def highPoint : Fin S.nbl → F := fun i => if hc : ptLocal + i.val < mPts then r ⟨ptLocal + i.val, hc⟩ else 0

omit [Fintype F] [DecidableEq F] [CharP F 2] in
theorem lowPoint_free (hR : R.free.Nodup) (t : Fin R.free.length) :
    lowPoint S R ptLocal mPts r (R.free.get t) = freePoint S R mPts r t := by
  simp only [lowPoint, regionPoint, Fin.is_lt, ↓reduceDIte, Fin.eta, idxOf?_get hR t, freePoint]

omit [Fintype F] [DecidableEq F] [CharP F 2] in
theorem lowPoint_fixed (j : Fin S.kLog) (hj : j ∉ R.free) :
    lowPoint S R ptLocal mPts r j = if R.fixedBits j then 1 else 0 := by
  simp only [lowPoint, regionPoint, Fin.is_lt, ↓reduceDIte, Fin.eta, List.idxOf?_eq_none_iff.2 hj]

omit [Fintype F] [DecidableEq F] [CharP F 2] in
theorem regionPoint_high (j : Fin S.nbl) (h : S.kLog + j.val < S.m) :
    regionPoint S R ptLocal mPts r ⟨S.kLog + j.val, h⟩ = highPoint S ptLocal mPts r j := by
  simp only [regionPoint, show ¬ (S.kLog + j.val < S.kLog) by omega, ↓reduceDIte, Nat.add_sub_cancel_left,
    highPoint]

omit [Fintype F] [DecidableEq F] [CharP F 2] in
theorem regionValue_eq :
    regionValue S R ptLocal mPts r = ∑ b : Fin (2 ^ S.nbl), eqAt (highPoint S ptLocal mPts r) b.val *
      ∑ i ∈ Finset.range (2 ^ R.free.length), eqAt (freePoint S R mPts r) i * ((R.bits b i).val : F) := rfl

omit [Fintype F] in
/-- **The in-block weight of a region point**: over the block's addresses, it vanishes off the region's addresses
`R.inBlock i`, and at `R.inBlock i` it is the free coordinates' weight at `i`. -/
theorem sum_low (hR : R.free.Nodup) (g : ℕ → F) :
    ∑ c ∈ Finset.range (2 ^ S.kLog), eqAt (lowPoint S R ptLocal mPts r) c * g c =
      ∑ i ∈ Finset.range (2 ^ R.free.length), eqAt (freePoint S R mPts r) i * g (R.inBlock i) := by
  classical
  have hfix : ∀ (c : ℕ) (j : Fin S.kLog), j ∉ R.free →
      (if c.testBit j then lowPoint S R ptLocal mPts r j else 1 + lowPoint S R ptLocal mPts r j) =
        if c.testBit j = R.fixedBits j then 1 else 0 := fun c j hj => by
    rw [lowPoint_fixed S R ptLocal mPts r j hj]
    exact fixed_factor _ _
  have hfree : ∀ (i : ℕ) (t : Fin R.free.length), (R.inBlock i).testBit (R.free.get t) = i.testBit t :=
    fun i t => by rw [testBit_inBlock R hR i, idxOf?_get hR t]
  have hnot : ∀ j : Fin S.kLog, j ∉ Finset.univ.image R.free.get → j ∉ R.free := fun j hj hm => by
    obtain ⟨t, ht⟩ := List.mem_iff_get.1 hm
    exact hj (Finset.mem_image.2 ⟨t, Finset.mem_univ _, ht⟩)
  have h1 : ∀ i, eqAt (lowPoint S R ptLocal mPts r) (R.inBlock i) = eqAt (freePoint S R mPts r) i := by
    intro i
    have hsub : ∀ j ∈ (Finset.univ : Finset (Fin S.kLog)), j ∉ Finset.univ.image R.free.get →
        (if (R.inBlock i).testBit j then lowPoint S R ptLocal mPts r j else 1 + lowPoint S R ptLocal mPts r j) = 1 := by
      intro j _ hj
      have hj' := hnot j hj
      have htb := testBit_inBlock R hR i j
      rw [List.idxOf?_eq_none_iff.2 hj'] at htb
      rw [hfix _ j hj', ite_eq_left htb]
    unfold eqAt
    rw [← Finset.prod_subset (Finset.subset_univ _) hsub,
      Finset.prod_image fun t _ t' _ h => (List.Nodup.get_inj_iff hR).1 h]
    exact Finset.prod_congr rfl fun t _ => by rw [hfree i t, lowPoint_free S R ptLocal mPts r hR t]
  have h2 : ∀ c < 2 ^ S.kLog, eqAt (lowPoint S R ptLocal mPts r) c ≠ 0 →
      c ∈ (Finset.range (2 ^ R.free.length)).image R.inBlock := by
    intro c hc hne
    have hfx : ∀ j : Fin S.kLog, j ∉ R.free → c.testBit j = R.fixedBits j := by
      intro j hj
      by_contra hb
      apply hne
      unfold eqAt
      exact Finset.prod_eq_zero (Finset.mem_univ j) (by rw [hfix c j hj, ite_eq_right hb])
    let s : Finset ℕ := (Finset.univ.filter fun t : Fin R.free.length => c.testBit (R.free.get t)).image Fin.val
    have hmem : ∀ t : Fin R.free.length, t.val ∈ s ↔ c.testBit (R.free.get t) := by
      intro t
      simp only [s, Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · rintro ⟨t', ht', he⟩
        rwa [Fin.ext he] at ht'
      · exact fun h => ⟨t, h, rfl⟩
    have hlt : ∀ n ∈ s, n < R.free.length := by
      intro n hn
      obtain ⟨t, _, rfl⟩ := Finset.mem_image.1 hn
      exact t.isLt
    refine Finset.mem_image.2 ⟨∑ k ∈ s, 2 ^ k, Finset.mem_range.2 (Nat.lt_pow_two_of_testBit _ fun n hn => ?_), ?_⟩
    · rw [testBit_sum_two_pow, decide_eq_false_iff_not]
      exact fun h => absurd (hlt n h) (by omega)
    · apply Nat.eq_of_testBit_eq
      intro n
      by_cases hn : n < S.kLog
      · have htb := testBit_inBlock R hR (∑ k ∈ s, 2 ^ k) ⟨n, hn⟩
        simp only at htb
        rw [htb]
        split
        · rename_i t h
          obtain ⟨htl, hget, -⟩ := List.idxOf?_eq_some_iff.1 h
          have hg : (R.free.get ⟨t, htl⟩).val = n := by simp [hget]
          have := hmem ⟨t, htl⟩
          rw [hg] at this
          rw [testBit_sum_two_pow, Bool.eq_iff_iff, decide_eq_true_iff]
          exact this
        · rename_i h
          exact (hfx ⟨n, hn⟩ (List.idxOf?_eq_none_iff.1 h)).symm
      · rw [testBit_inBlock_ge R hR _ n (by omega),
          Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hc (Nat.pow_le_pow_right (by norm_num) (by omega)))]
  have h3 : Set.InjOn R.inBlock ↑(Finset.range (2 ^ R.free.length)) := by
    intro i hi i' hi' he
    have hi := Finset.mem_range.1 (Finset.mem_coe.1 hi)
    have hi' := Finset.mem_range.1 (Finset.mem_coe.1 hi')
    apply Nat.eq_of_testBit_eq
    intro n
    by_cases hn : n < R.free.length
    · have e1 : i.testBit n = (R.inBlock i).testBit (R.free.get ⟨n, hn⟩) := (hfree i ⟨n, hn⟩).symm
      have e2 : i'.testBit n = (R.inBlock i').testBit (R.free.get ⟨n, hn⟩) := (hfree i' ⟨n, hn⟩).symm
      rw [e1, e2, he]
    · rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hi (Nat.pow_le_pow_right (by norm_num) (by omega))),
        Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hi' (Nat.pow_le_pow_right (by norm_num) (by omega)))]
  have hsub : (Finset.range (2 ^ R.free.length)).image R.inBlock ⊆ Finset.range (2 ^ S.kLog) := by
    intro c hc
    obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hc
    exact Finset.mem_range.2 (Discharge.Layout.inBlock_lt R hR i)
  rw [← Finset.sum_subset (f := fun c => eqAt (lowPoint S R ptLocal mPts r) c * g c) hsub fun c hc hnc => ?_,
    Finset.sum_image h3]
  · exact Finset.sum_congr rfl fun i _ => by rw [h1 i]
  · by_contra hne
    exact hnc (h2 c (Finset.mem_range.1 hc) (left_ne_zero_of_mul hne))

/-- **A region claim holds at every link point** when the witness's region bits are the region's: its evaluation is
its value. -/
theorem regionClaim_holds (z : Witness S.m) (hR : R.free.Nodup)
    (hz : ∀ (o : Fin (2 ^ S.nbl)) (i : ℕ) (_hi : i < 2 ^ R.free.length) (h : R.inBlock i < 2 ^ S.kLog),
      S.block z o ⟨R.inBlock i, h⟩ = R.bits o i) :
    (regionClaim S R ptLocal mPts r).eval z = regionValue S R ptLocal mPts r := by
  have hN : S.kLog + S.nbl = S.m := by have := S.hk; simp only [Statement.nbl]; omega
  rw [regionClaim_eval, sum_addr, regionValue_eq]
  refine Finset.sum_congr rfl fun o _ => ?_
  let g : ℕ → F := fun c => if h : c < 2 ^ S.kLog then ((z (S.addr o ⟨c, h⟩)).val : F) else 0
  have e : ∀ c : Fin (2 ^ S.kLog), eqAt (regionPoint S R ptLocal mPts r) (S.addr o c).val * ((z (S.addr o c)).val : F) =
      eqAt (highPoint S ptLocal mPts r) o.val * (eqAt (lowPoint S R ptLocal mPts r) c.val * g c.val) := by
    intro c
    have ha : (S.addr o c).val = c.val + 2 ^ S.kLog * o.val := by simp only [Statement.addr]; ring
    have hhi : eqAt (fun j : Fin S.nbl => regionPoint S R ptLocal mPts r ⟨S.kLog + j.val, by omega⟩) o.val =
        eqAt (highPoint S ptLocal mPts r) o.val := by
      unfold eqAt
      exact Finset.prod_congr rfl fun j _ => by beta_reduce; rw [regionPoint_high]
    rw [ha, eqAt_split_at S.kLog S.nbl hN _ c.val o.val c.isLt, hhi, show g c.val = ((z (S.addr o c)).val : F) from
      dite_eq_left c.isLt]
    change eqAt (lowPoint S R ptLocal mPts r) c.val * eqAt (highPoint S ptLocal mPts r) o.val * _ = _
    ring
  rw [Finset.sum_congr rfl fun c _ => e c, ← Finset.mul_sum,
    Fin.sum_univ_eq_sum_range (fun c => eqAt (lowPoint S R ptLocal mPts r) c * g c), sum_low S R ptLocal mPts r hR g]
  congr 1
  refine Finset.sum_congr rfl fun i hi => ?_
  have hib := Discharge.Layout.inBlock_lt R hR i
  rw [show g (R.inBlock i) = ((z (S.addr o ⟨R.inBlock i, hib⟩)).val : F) from dite_eq_left hib,
    show z (S.addr o ⟨R.inBlock i, hib⟩) = R.bits o i from hz o i (Finset.mem_range.1 hi) hib]

end FlockComplete.Replay.Region

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## A switched vector meets its claim's evaluation -/

namespace Rs

theorem eval_congr {m : ℕ} (c c' : Model.Claim 𝔽 m) (hs : ∀ s, c.skip s = c'.skip s) (hx : ∀ i, c.x i = c'.x i)
    (w : Model.Witness m) : c.eval w = c'.eval w := by
  unfold Model.Claim.eval
  rw [funext hs, funext hx]

/-- **The honest `ŝ` of a claim**, on a trace whose bits are the witness's, has the claim's evaluation as its packed
weights' sum. -/
theorem switch_eval {m : ℕ} (hm : 7 ≤ m) (c : Claim) (hx : c.x.size = m - 6) (tr : Trace) (w : Model.Witness m)
    (hz : ∀ a : Fin (2 ^ m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) :
    fOf (dot (wOf c) (switchVec tr c)) = (Discharge.Composed.RingSwitch.claimOf m c).eval w := by
  show fOf (Zk.claimCheck c (switchVec tr c)) = _
  rw [Discharge.Composed.RingSwitch.fOf_claimCheck hm c, claim_eval_eq hm]
  refine Finset.sum_congr rfl fun v _ => ?_
  congr 1
  have hsz : (c.x.extract 1 c.x.size).size = m - 7 := by
    rw [Array.size_extract, Nat.min_self, hx]
    omega
  show GF128.of (switchVec tr c)[v.val]! = _
  unfold switchVec
  dsimp only
  rw [Discharge.Composed.Zerocheck.get_map_range _ v.isLt, Zc.of_sumF, eqTable_size, hsz, Finset.sum_range]
  unfold shat
  refine Finset.sum_congr rfl fun y _ => ?_
  have he : GF128.of (eqTable (c.x.extract 1 c.x.size))[y.val]! =
      Model.eqAt ((Discharge.Composed.RingSwitch.claimOf m c).tail hm) y.val := by
    rw [Zc.of_eqTable_extract c.x 1 c.x.size rfl (by omega) y.val
        (by rw [hx, show m - 6 - 1 = m - 7 by omega]; exact y.isLt),
      Discharge.Composed.RingSwitch.eqAt_range ((Discharge.Composed.RingSwitch.claimOf m c).tail hm)
        (fun i => fOf c.x[i + 1]!) fun j => rfl, hx, show m - 6 - 1 = m - 7 by omega]
    unfold Discharge.Composed.Zerocheck.eqR
    refine Finset.prod_congr rfl fun j _ => ?_
    rw [Nat.add_comm 1 j]
    rfl
  rw [← hz (addr128 hm y v), ← he]
  unfold Zc.bitF
  rw [show (addr128 hm y v).val = 128 * y.val + v.val from rfl]
  cases tr.z[128 * y.val + v.val]!
  · rw [ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true, mul_zero]
    exact GF128.zero_eq.symm
  · rw [ite_eq_left rfl, ite_eq_left rfl, mul_one]

/-- A claim whose evaluation is its value is met by its honest `ŝ`. -/
theorem switch_of {m : ℕ} (hm : 7 ≤ m) (c : Claim) (hx : c.x.size = m - 6) (tr : Trace) (w : Model.Witness m)
    (hz : ∀ a : Fin (2 ^ m), Zc.bitF tr.z a.val = ((w a).val : 𝔽))
    (hc : (Discharge.Composed.RingSwitch.claimOf m c).eval w = fOf c.value) :
    dot (wOf c) (switchVec tr c) = c.value :=
  Discharge.Composed.Zerocheck.eq_of_of ((switch_eval hm c hx tr w hz).trans hc)

end Rs

/-! ## Lincheck's claim -/

namespace Lc

theorem zrT_foldl (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (cs : Challenges) (j : ℕ) :
    (pushes cs.τ j).foldl (fun acc t => halve t acc) (zr0 c m tr zc) = zrT c m tr zc cs j := by
  induction j with
  | zero => rfl
  | succ j ih => simp only [pushes, Array.foldl_push, ih, zrT]

/-- **The witness's folded columns** (`zr0`): at column `k`, `Σ_o eq(ρ_out, o)·z[k + 2^kLog·o]`, at the zerocheck's
outer coins `ρ_out`. -/
theorem zr0_hcol (S : Model.Statement) (c : CircuitFold) (hK : c.kLog = S.kLog) (tr : Trace) (zc : ZcOut)
    (hrho : zc.rho.size = S.m - 6) (w : Model.Witness S.m)
    (hz : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (rOut : Fin S.nbl → 𝔽)
    (hr : ∀ j : Fin S.nbl, rOut j = fOf zc.rho[S.kLog - 6 + j.val]!) :
    ∀ k < 2 ^ S.kLog, fOf (zr0 c S.m tr zc)[k]! = hcol S rOut w k := by
  have hk6 := S.hk6
  have hkm := S.hk
  have hlen : S.m - 6 - (c.kLog - 6) = S.nbl := by
    rw [hK]
    show S.m - 6 - (S.kLog - 6) = S.m - S.kLog
    omega
  have hE : ∀ o : Fin (2 ^ S.nbl), GF128.of (eqTable (zc.rho.extract (c.kLog - 6) (S.m - 6)))[o.val]! =
      Model.eqAt rOut o.val := by
    intro o
    rw [Zc.of_eqTable_extract zc.rho (c.kLog - 6) (S.m - 6) hrho.symm (by omega) o.val (by rw [hlen]; exact o.isLt)]
    unfold Discharge.Composed.Zerocheck.eqR Model.eqAt
    rw [hlen, ← Fin.prod_univ_eq_prod_range]
    refine Finset.prod_congr rfl fun j _ => ?_
    rw [hr j, hK]
    rfl
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

/-- **Lincheck's claim is the witness's extension at its point**: `w = Σ_s L_s(r')·zr[s]` of the prover's last halved
columns is the evaluation of the claim `(L(r'), ρ_in' ‖ ρ_out)` (`ρ_in'` the round coins, last first). -/
theorem ab_eval (S : Model.Statement) (c : CircuitFold) (hK : c.kLog = S.kLog) (tr : Trace) (zc : ZcOut)
    (hrho : zc.rho.size = S.m - 6) (w : Model.Witness S.m)
    (hz : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (cs : Challenges) (v : F128) :
    fOf (dot (skipWeights (cs.cf 0)) (zrT c S.m tr zc cs (c.kLog - 6))) =
      (Discharge.Composed.RingSwitch.claimOf S.m
        { value := v, skip := skipWeights (cs.cf 0),
          x := (pushes cs.τ (c.kLog - 6)).reverse ++ zc.rho.extract (S.kLog - 6) (S.m - 6) }).eval w := by
  have h6 : 6 ≤ S.m := le_trans S.hk6 S.hk
  have hk6 := S.hk6
  have hts : (pushes cs.τ (c.kLog - 6)).size = S.kLog - 6 := by rw [pushes_size, hK]
  have hzs : (zr0 c S.m tr zc).size = 2 ^ S.kLog := by simp [zr0, hK]
  have h64 : 2 ^ S.kLog / 2 ^ (S.kLog - 6) = 64 := by
    rw [Nat.pow_div (by omega) (by norm_num), show S.kLog - (S.kLog - 6) = 6 by omega]
  rw [← Discharge.Composed.RingSwitch.sum_zPartialOf h6, fOf_dot _ _ 64 (skipWeights_size _)]
  refine Finset.sum_congr rfl fun i _ => congrArg₂ (· * ·) rfl ?_
  rw [← zrT_foldl, ← Array.foldl_toList,
    foldl_halve _ _ i (by rw [hzs, Array.length_toList, hts, h64]; exact i.isLt), hzs,
    foldAll_congr _ _ _ (Discharge.Composed.Lincheck.lcV S zc.rho w)
      (fun k hk => zr0_hcol S c hK tr zc hrho w hz (fun j => fOf zc.rho[S.kLog - 6 + j.val]!) (fun j => rfl) k hk) i
      (by rw [List.length_map, Array.length_toList, hts, h64]; exact i.isLt),
    Discharge.Composed.Lincheck.foldAll_lcV S w zc.rho _ hrho hts (cs.cf 0) i]
  rfl

end Lc

/-! ## Zerocheck's claim -/

/-- **Zerocheck's claim is the witness's extension at its point**: `P_c(ζ)` of the prover's `round1_c` is the
evaluation of the claim `(L(ζ), r)`. -/
theorem Zc.c_claim_eval {m : ℕ} (hm : 13 ≤ m) (zcs : Zc.Challenges) (tr : Trace) (w : Model.Witness m)
    (hz : ∀ a : Fin (2 ^ m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (v : F128) :
    fOf (Zc.out m zcs tr).vC =
      (Discharge.Composed.RingSwitch.claimOf m
        { value := v, skip := skipWeights (Zc.out m zcs tr).z, x := (Zc.out m zcs tr).rRest }).eval w := by
  have h := Discharge.Composed.ZerocheckMasked.finalC_eq_eval m (by omega) (Zc.bitF tr.az) (Zc.bitF tr.bz)
    (Zc.bitF tr.z) w hz (zcs.r m) (pushes zcs.ρ (m - 6)) zcs.z
  rw [← Discharge.Composed.Zerocheck.c_eval, ← Zc.r1c_eq tr (m - 6) (zcs.r m) (Zc.r_size m hm zcs)] at h
  exact h.trans (Rs.eval_congr _ _ (fun _ => rfl) (fun _ => rfl) w)

/-! ## The extra claims -/

theorem mem_right {α β : Type} {R : α → β → Prop} : ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ →
    ∀ {b : β}, b ∈ l₂ → ∃ a ∈ l₁, R a b
  | _, _, .nil, _, hb => by simp at hb
  | _, _, .cons hab h, _, hb => by
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨_, List.mem_cons_self, hab⟩
    · obtain ⟨a, ha, hr⟩ := mem_right h hb
      exact ⟨a, List.mem_cons_of_mem _ ha, hr⟩

theorem mem_of_forall₂ {α β γ : Type} {P : α → β → γ → Prop} {pts : List β} :
    ∀ {regs : List α} {ess : List (List γ)}, List.Forall₂ (fun a es => List.Forall₂ (P a) pts es) regs ess →
      ∀ {e : γ}, e ∈ ess.flatten → ∃ a ∈ regs, ∃ p ∈ pts, P a p e
  | _, _, .nil, _, he => by simp at he
  | _, _, .cons hes h, _, he => by
    rcases List.mem_append.1 (by simpa only [List.flatten_cons] using he) with he | he
    · obtain ⟨p, hp, hpe⟩ := mem_right hes he
      exact ⟨_, List.mem_cons_self, p, hp, hpe⟩
    · obtain ⟨a, ha, p, hp, hpe⟩ := mem_of_forall₂ h he
      exact ⟨a, List.mem_cons_of_mem _ ha, p, hp, hpe⟩

/-- **Every claim a rep opens is met by its honest `ŝ`**, for a statement's table, on a trace whose bits satisfy its
statement, at link points of `m_pts` coordinates. -/
theorem claims_switch (st : Stmt) (hwf : StmtWF st) (hrw : RegionsWF st) (hm : 22 ≤ st.m) (w : Model.Witness st.m)
    (hsat : (stmtOf st hwf (regionsOf st)).Satisfies w) (tr : Trace)
    (hz : ∀ a : Fin (2 ^ st.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (zcs : Zc.Challenges) (cs : Lc.Challenges)
    (points : Array (Array F128)) (hpts : ∀ p ∈ points.toList, p.size = st.mPts) :
    ∀ c ∈ abcClaims st.m st.c.kLog (Zc.out st.m zcs tr) (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) cs) ++
        val (st.extraClaims points), dot (wOf c) (switchVec tr c) = c.value := by
  have h7 : 7 ≤ st.m := by omega
  have hk := hwf.hk
  intro c hc
  rcases Array.mem_append.1 hc with hc | hc
  · simp only [abcClaims, List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl
    · refine Rs.switch_of h7 _ ?_ tr w hz ?_
      · show ((pushes cs.τ (st.c.kLog - 6)).reverse ++
          (pushes zcs.ρ (st.m - 6)).extract (st.c.kLog - 6) (st.m - 6)).size = st.m - 6
        rw [Array.size_append, Array.size_reverse, pushes_size, Array.size_extract, pushes_size, Nat.min_self]
        omega
      · exact (Lc.ab_eval (stmtOf st hwf (regionsOf st)) st.circuitFold rfl tr (Zc.out st.m zcs tr) (pushes_size _ _) w
          hz cs _).symm
    · exact Rs.switch_of h7 _ (Zc.r_size st.m (by omega) zcs) tr w hz (Zc.c_claim_eval (by omega) zcs tr w hz _).symm
  · cases hex : st.extraClaims points with
    | error e =>
      rw [hex, show val (Except.error e : Except String (Array Claim)) = #[] from rfl] at hc
      simp at hc
    | ok extras =>
      rw [hex] at hc
      obtain ⟨ess, hess, heq⟩ := extraClaims_ok st points extras hex
      have hc' : c ∈ ess.flatten := by
        rw [← heq]
        exact Array.mem_def.1 hc
      obtain ⟨reg, hreg, r, hr, v, hv, rfl⟩ := mem_of_forall₂ hess hc'
      have hrel := regionClaim_rel' st hwf (regionsOf st) reg (hrw.regions reg hreg) hrw.hm st.mPts hrw.mPts r
        (hpts r hr) (fun t => fOf r[t.val]!) (fun t => rfl) v hv
      have hRm : regionOf st.c.kLog (st.m - st.c.kLog) reg ∈ (stmtOf st hwf (regionsOf st)).regions :=
        List.mem_map_of_mem hreg
      have hhold := Region.regionClaim_holds (stmtOf st hwf (regionsOf st))
        (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL st.mPts (fun t => fOf r[t.val]!) w
        ((stmtOf_linkLayout st hwf hrw).1 _ hRm).1 (hsat.2.2 _ hRm)
      refine Rs.switch_of h7 _ hrel.x_size tr w hz ?_
      exact (Rs.eval_congr _ _ (fun s => (hrel.skip s).symm) (fun i => (hrel.x i).symm) w).trans
        (hhold.trans hrel.value)

/-! ## The record's link points -/

/-- The record's link points have `m_pts` coordinates: `Record.decodeWith` checks it (S9). -/
theorem decodeWith_points (hs : Hashes Flock.V) {spec : SessionSpec} {text : String} {sess : Session}
    (h : Record.decodeWith hs spec text = .ok sess) : ∀ p ∈ sess.points.toList, p.size = spec.mPts := by
  unfold Record.decodeWith Record.decodeJWith at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    exact (check_pts (by assumption)).2

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

theorem _root_.FlockComplete.Run.sess_points (o : Run H rel) {st : Stmt} (hspec : o.spec = st.spec) :
    ∀ p ∈ o.sess.points.toList, p.size = st.mPts := by
  unfold Run.sess
  cases h : Record.decodeWith o.hs o.spec o.record with
  | error e =>
    intro p hp
    rw [show (val (Except.error e : Except String Session)).points = #[] from rfl] at hp
    simp at hp
  | ok s =>
    have hp := decodeWith_points o.hs h
    rw [hspec] at hp
    exact hp

/-! ## At the deployed relation -/

theorem switch_all {claims : Array Claim} {tr : Trace} (h : ∀ c ∈ claims, dot (wOf c) (switchVec tr c) = c.value) :
    ∀ k < claims.size, dot (wOf claims[k]!) (switchVec tr claims[k]!) = claims[k]!.value := fun k hk => by
  rw [getElem!_pos claims k hk]
  exact h _ (Array.getElem_mem hk)

/-- **The ring switch's algebra at the deployed relation**, from `RelS` alone. -/
theorem rsAlgebra (H : Claims.Msg → Claims.Msg) : RsAlgebra H Readable.relation := by
  intro o i hi r _ k hk
  obtain ⟨st, hsts, hwf, hrw, -, hsat⟩ := o.model
  obtain ⟨st', hsts', hspec, h22, -⟩ := o.table
  obtain rfl : st = st' := by
    rw [hsts] at hsts'
    simpa using hsts'
  have hsz : o.tabs.size = 1 := by simp [Run.tabs, FlockDeploy.tabsOf, hsts]
  obtain rfl : i = 0 := by omega
  have ht : o.tabs[0]! = Setup.ofCircuit st := by simp [Run.tabs, FlockDeploy.tabsOf, hsts]
  have htr := o.trace_one hsts
  have hzb : ∀ a : Fin (2 ^ (stmtOf st hwf (regionsOf st)).m),
      Zc.bitF (o.trace 0).z a.val = ((Readable.zOf 1 o.w 0 st.m a).val : 𝔽) := by
    intro a
    rw [htr]
    unfold Zc.bitF
    rw [traceOf_z st o.w.hid a.isLt, Lc.bitF_toZ]
    show _ = ((Readable.toZ (o.w.hid.getD (a.val * 1 + 0) false)).val : 𝔽)
    rw [Nat.mul_one, Nat.add_zero]
  have hcl : o.claims 0 r = abcClaims st.m st.c.kLog (Zc.out st.m (o.zcCoins 0 r) (o.trace 0))
      (Lc.out st.circuitFold st.m (o.trace 0) (Zc.out st.m (o.zcCoins 0 r) (o.trace 0)) (o.lcCoins 0 r)) ++
        val (st.extraClaims o.sess.points) := by
    simp only [Run.claims, Run.lcOut, Run.zcOut, Run.extras, ht, Setup.ofCircuit]
  rw [hcl] at hk ⊢
  exact switch_all (claims_switch st hwf hrw h22 _ hsat (o.trace 0) hzb (o.zcCoins 0 r) (o.lcCoins 0 r) o.sess.points
    (o.sess_points hspec)) k hk

/-- **The ring-switch goal at the deployed relation**, at every rep the session walk aligns. -/
theorem ringSwitch_relation (H : Claims.Msg → Claims.Msg)
    (hzc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r)
    (hrs : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) :
    RingSwitch H Readable.relation :=
  ringSwitch_goal' H hzc hlc (lcAlgebra H) hrs (rsAlgebra H)

end FlockComplete.Replay
