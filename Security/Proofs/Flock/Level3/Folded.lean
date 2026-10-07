/-
Level 3: the lincheck's `Stmt.folded` (PROTOCOL.md §10). `partial_eq_halve_fold`: halving a vector of `2^n · 64` entries
at `ts` in turn (`halve`, the lincheck rounds' column fold) leaves at `s < 64` the sum `Σ_u eq(ts reversed, u) · v[s + 64·u]`.
`folded_eq`: when `Stmt.folded` succeeds, its vector is `Stmt.fold`'s with `β` at the pin, halved at `ts` in turn, whether
it read the block's structure (`foldedPartial`) or folded flat.
-/
import Proofs.Flock.Level3.FoldStmt

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

open GF128

/-! ### Sums, the equality tensor and the halving -/

theorem of_zero' : of (0 : F128) = 0 := zero_eq.symm

theorem sumUpTo_eq (n : ℕ) (f : ℕ → F128) : of (Stmt.sumUpTo n f) = ∑ i ∈ Finset.range n, of (f i) := by
  unfold Stmt.sumUpTo
  induction n with
  | zero => simp [of_zero']
  | succ n ih => rw [Nat.fold_succ, of_add, ih, Finset.sum_range_succ]

/-- `eq(r, u)` with `r` split after its first `a` coordinates. -/
theorem eqTable_split (r : Array F128) (a : ℕ) (ha : a ≤ r.size) (u : ℕ) (hu : u < 2 ^ r.size) :
    of (eqTable r)[u]! = of (eqTable (r.extract 0 a))[u % 2 ^ a]! * of (eqTable (r.extract a r.size))[u / 2 ^ a]! := by
  obtain ⟨b, hsz⟩ : ∃ b, r.size = a + b := ⟨r.size - a, by omega⟩
  rw [eqTable_eqAt r u hu]
  have e1 : eqAt (fun j : Fin r.size => of r[j.val]!) u = eqAt (fun j : Fin (a + b) => of r[j.val]!) u := by
    unfold eqAt; exact Fintype.prod_equiv (finCongr hsz) _ _ (fun j => rfl)
  rw [e1, eqAt_split a b]
  have hlsz : (r.extract 0 a).size = a := by simp; omega
  have hhsz : (r.extract a r.size).size = b := by simp; omega
  have hlo : u % 2 ^ a < 2 ^ (r.extract 0 a).size := by rw [hlsz]; exact Nat.mod_lt _ (Nat.two_pow_pos a)
  have hhi : u / 2 ^ a < 2 ^ (r.extract a r.size).size := by
    rw [hhsz, Nat.div_lt_iff_lt_mul (Nat.two_pow_pos a), ← pow_add, show b + a = r.size by omega]; exact hu
  rw [eqTable_eqAt _ _ hlo, eqTable_eqAt _ _ hhi]
  congr 1
  · unfold eqAt
    refine Fintype.prod_equiv (finCongr hlsz.symm) _ _ (fun j => ?_)
    have hj := j.isLt
    simp only [finCongr_apply, Fin.val_cast, Fin.val_castAdd]
    rw [extract_get! r 0 a j.val (by omega) (by omega), zero_add]
  · unfold eqAt
    refine Fintype.prod_equiv (finCongr hhsz.symm) _ _ (fun j => ?_)
    have hj := j.isLt
    simp only [finCongr_apply, Fin.val_cast, Fin.val_natAdd]
    rw [extract_get! r a r.size j.val (by omega) le_rfl]

theorem eqTable_push (r : Array F128) (t : F128) : eqTable (r.push t) = eqStep (eqTable r) t := by
  simp [eqTable]

theorem eqStep_lo (tb : Array F128) (x : F128) (u : ℕ) (hu : u < tb.size) :
    of (eqStep tb x)[u]! = of tb[u]! * (1 + of x) := by
  have hi : u < (eqStep tb x).size := by rw [eqStep_size]; omega
  rw [getElem!_pos _ u hi, eqStep_get tb x u hi (by omega), getElem!_pos tb u hu]
  simp [Nat.mod_eq_of_lt hu, Nat.div_eq_of_lt hu, eqFactor]

theorem eqStep_hi (tb : Array F128) (x : F128) (u : ℕ) (hu : u < tb.size) :
    of (eqStep tb x)[tb.size + u]! = of tb[u]! * of x := by
  have hi : tb.size + u < (eqStep tb x).size := by rw [eqStep_size]; omega
  have hm : (tb.size + u) % tb.size = u := by rw [Nat.add_mod_left, Nat.mod_eq_of_lt hu]
  have hd : (tb.size + u) / tb.size = 1 := by
    rw [Nat.add_div_left _ (by omega), Nat.div_eq_of_lt hu]
  rw [getElem!_pos (eqStep tb x) (tb.size + u) hi, eqStep_get tb x _ hi (by omega), getElem!_pos tb u hu]
  simp [hm, hd, eqFactor]

theorem halve_size (t : F128) (v : Array F128) : (halve t v).size = v.size / 2 := by simp [halve]

theorem halve_get (t : F128) (v : Array F128) (i : ℕ) (hi : i < v.size / 2) :
    of (halve t v)[i]! = (1 + of t) * of v[i]! + of t * of v[i + v.size / 2]! := by
  unfold halve
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range, of_add, of_mul, of_add]
  ring

/-- **The halvings' sum.** Halving `v` (`2^|l| · 64` entries) at `l` in turn, top bit first, leaves at `s < 64` the sum
`Σ_u eq(l reversed, u) · v[s + 64·u]`: the column marginal at the reversed challenges. -/
theorem partial_eq_halve_fold : ∀ (l : List F128) (v : Array F128), v.size = 2 ^ l.length * 64 → ∀ s < 64,
    of (l.foldl (fun c t => halve t c) v)[s]! =
      ∑ u ∈ Finset.range (2 ^ l.length), of (eqTable l.reverse.toArray)[u]! * of v[s + 64 * u]!
  | [], v, _, s, _ => by
      have e : eqTable (#[] : Array F128) = #[1] := rfl
      simp [e, ← one_eq]
  | t :: l, v, hv, s, hs => by
      have hv' : v.size = 2 ^ l.length * 64 * 2 := by rw [hv, List.length_cons, pow_succ]; ring
      have hh : (halve t v).size = 2 ^ l.length * 64 := by rw [halve_size, hv']; omega
      have hv2 : v.size / 2 = 2 ^ l.length * 64 := by omega
      rw [List.foldl_cons, partial_eq_halve_fold l (halve t v) hh s hs]
      have hrev : (t :: l).reverse.toArray = l.reverse.toArray.push t := by simp
      have htb : (eqTable l.reverse.toArray).size = 2 ^ l.length := by rw [eqTable_size]; simp
      rw [hrev, eqTable_push, List.length_cons, pow_succ, mul_two, Finset.sum_range_add, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun u hu => ?_
      rw [Finset.mem_range] at hu
      rw [halve_get t v (s + 64 * u) (by omega), hv2, eqStep_lo _ t u (by omega), ← htb,
        eqStep_hi _ t u (by omega), htb, show s + 64 * u + 2 ^ l.length * 64 = s + 64 * (2 ^ l.length + u) by ring]
      ring

theorem foldl_halve_size : ∀ (ts : List F128) (comb : Array F128),
    (ts.foldl (fun c t => halve t c) comb).size = comb.size / 2 ^ ts.length
  | [], comb => by simp
  | t :: ts, comb => by
      rw [List.foldl_cons, foldl_halve_size ts, halve_size, Nat.div_div_eq_div_mul, List.length_cons, pow_succ']

/-! ### Sums over blocks, windows and single entries -/

theorem sum_range_blocks (a b : ℕ) (f : ℕ → GF128) :
    ∑ u ∈ Finset.range (a * b), f u = ∑ q ∈ Finset.range a, ∑ j ∈ Finset.range b, f (q * b + j) := by
  induction a with
  | zero => simp
  | succ a ih => rw [Nat.succ_mul, Finset.sum_range_add, ih, Finset.sum_range_succ]

theorem sum_window (M p n : ℕ) (hpn : p + n ≤ M) (F : ℕ → GF128) :
    ∑ q ∈ Finset.range M, (if p ≤ q ∧ q < p + n then F q else 0) = ∑ k ∈ Finset.range n, F (p + k) := by
  have e : ∑ k ∈ Finset.range n, F (p + k) = ∑ k ∈ Finset.Ico p (p + n), F k := by
    rw [Finset.sum_Ico_eq_sum_range, Nat.add_sub_cancel_left]
  rw [e, ← Finset.sum_filter]
  congr 1
  ext q
  simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
  omega

/-- The halvings' weight on one column `c`: `Σ_u f(u)·[c = s + 64·u]` is `f(⌊c/64⌋)` when `c` is in entry `s`'s column. -/
theorem sum_spot (N s c : ℕ) (hs : s < 64) (f : ℕ → GF128) (x : GF128) :
    ∑ u ∈ Finset.range N, f u * (if c = s + 64 * u then x else 0) =
      if c < 2 ^ 6 * N ∧ c % 64 = s then f (c / 64) * x else 0 := by
  by_cases hm : c % 64 = s
  · have hiff : ∀ u, (c = s + 64 * u) ↔ c / 64 = u := fun u => by omega
    simp_rw [hiff, mul_ite, mul_zero]
    rw [Finset.sum_ite_eq]
    by_cases hc : c / 64 < N
    · rw [if_pos (Finset.mem_range.mpr hc), if_pos ⟨by omega, hm⟩]
    · rw [if_neg (by simpa using hc), if_neg (by omega)]
  · have hne : ∀ u, ¬ (c = s + 64 * u) := fun u => by omega
    simp [hne, hm]

theorem sum_list_comm {α : Type} (l : List α) (N : ℕ) (F : ℕ → α → GF128) :
    ∑ u ∈ Finset.range N, (l.map (F u)).sum = (l.map fun a => ∑ u ∈ Finset.range N, F u a).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.sum_cons, Finset.sum_add_distrib, ih]

theorem ofFn_get! (n : ℕ) (f : Fin n → F128) (i : ℕ) (hi : i < n) : (Array.ofFn f)[i]! = f ⟨i, hi⟩ := by
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_ofFn]

theorem array_ext_of (a b : Array F128) (hs : a.size = b.size) (h : ∀ i < b.size, of a[i]! = of b[i]!) : a = b := by
  refine Array.ext hs fun i h1 h2 => ?_
  have := h i h2
  rwa [getElem!_pos a i h1, getElem!_pos b i h2] at this

theorem list_mul_sum {α : Type} (a : GF128) (l : List α) (g : α → GF128) :
    a * (l.map g).sum = (l.map fun x => a * g x).sum := by
  induction l with
  | nil => simp
  | cons x l ih => simp only [List.map_cons, List.sum_cons, mul_add, ih]

/-! ### Disjoint ranges -/

theorem covers_iff (r : Range) (c : ℕ) :
    covers r c ↔ r.pos0 * 2 ^ r.slotLog ≤ c ∧ c < (r.pos0 + r.count) * 2 ^ r.slotLog := by
  unfold covers
  rw [Nat.le_div_iff_mul_le (Nat.two_pow_pos _), Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)]

/-- Two ranges with no column in common, as `Stmt.structured` requires of every pair. -/
def apart (a b : Range) : Prop :=
  (a.pos0 + a.count) * 2 ^ a.slotLog ≤ b.pos0 * 2 ^ b.slotLog ∨
    (b.pos0 + b.count) * 2 ^ b.slotLog ≤ a.pos0 * 2 ^ a.slotLog

theorem not_covers_of_apart {a b : Range} (h : apart a b) {c : ℕ} (ha : covers a c) : ¬ covers b c := by
  rw [covers_iff] at ha ⊢
  rcases h with h | h <;> omega

theorem structured_spec (st : Stmt) (h : st.structured = true) :
    6 ≤ st.c.kLog ∧
      (∀ r ∈ st.c.ranges.toList,
        6 ≤ r.slotLog ∧ r.slotLog ≤ st.c.kLog ∧ (r.pos0 + r.count) * 2 ^ r.slotLog ≤ 2 ^ st.c.kLog) ∧
      st.c.ranges.toList.Pairwise apart := by
  simp only [Stmt.structured, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- What range `r` leaves at a column it covers (`rangesVal`'s term). -/
def rangeTerm (st : Stmt) (alpha : F128) (lw rho : Array F128) (r : Range) (c : ℕ) : F128 :=
  (eHiFor rho r)[c / 2 ^ r.slotLog]! * (baseFor st alpha lw rho r)[c % 2 ^ r.slotLog]!

theorem rangesVal_none (st : Stmt) (alpha : F128) (lw rho : Array F128) :
    ∀ (rs : List Range) (c : ℕ) (v : F128), (∀ r ∈ rs, ¬ covers r c) → rangesVal st alpha lw rho rs c v = v
  | [], _, _, _ => rfl
  | r :: rs, c, v, h => by
      rw [rangesVal, if_neg (h r List.mem_cons_self)]
      exact rangesVal_none st alpha lw rho rs c v (fun r' hr' => h r' (List.mem_cons_of_mem _ hr'))

/-- On disjoint ranges, what they leave at `c` over zero is the sum of their terms there. -/
theorem rangesVal_sum (st : Stmt) (alpha : F128) (lw rho : Array F128) :
    ∀ (rs : List Range), rs.Pairwise apart → ∀ c, of (rangesVal st alpha lw rho rs c 0) =
      (rs.map fun r => if covers r c then of (rangeTerm st alpha lw rho r c) else 0).sum
  | [], _, c => by simp [rangesVal, of_zero']
  | r :: rs, hp, c => by
      rw [List.pairwise_cons] at hp
      rw [rangesVal, List.map_cons, List.sum_cons]
      by_cases hc : covers r c
      · simp only [if_pos hc]
        rw [rangesVal_none st alpha lw rho rs c _ (fun r' hr' => not_covers_of_apart (hp.1 r' hr') hc),
          List.sum_eq_zero (fun x hx => ?_), add_zero]
        · rfl
        obtain ⟨r', hr', rfl⟩ := List.mem_map.mp hx
        rw [if_neg (not_covers_of_apart (hp.1 r' hr') hc)]
      · simp only [if_neg hc, zero_add]
        exact rangesVal_sum st alpha lw rho rs hp.2 c

/-! ### One range's share of the 64 entries -/

/-- **A range's share.** After the halvings at `rr` reversed, what range `r` leaves (`eHi` at the slot times the base
at the column within it) contributes to entry `s` its base projected on `eq(rr[.. sl − 6], ·)`, times
`Σ_q eHi[q]·eq(rr[sl − 6 ..], q)` over its slots. -/
theorem range_share (r : Range) (rr eHi base : Array F128) (hsl : 6 ≤ r.slotLog) (hslk : r.slotLog - 6 ≤ rr.size)
    (hin : (r.pos0 + r.count) * 2 ^ r.slotLog ≤ 2 ^ (rr.size + 6)) (s : ℕ) (hs : s < 64) :
    ∑ u ∈ Finset.range (2 ^ rr.size), of (eqTable rr)[u]! *
        (if covers r (s + 64 * u) then
          of (eHi[(s + 64 * u) / 2 ^ r.slotLog]! * base[(s + 64 * u) % 2 ^ r.slotLog]!) else 0) =
      of (Stmt.sumUpTo r.count fun k => eHi[r.pos0 + k]! * (eqTable (rr.extract (r.slotLog - 6) rr.size))[r.pos0 + k]!) *
        of (Stmt.project base (eqTable (rr.extract 0 (r.slotLog - 6))))[s]! := by
  set h := r.slotLog - 6 with hh
  have hP : 2 ^ r.slotLog = 2 ^ h * 64 := by
    rw [show r.slotLog = h + 6 by omega, pow_add]; norm_num
  obtain ⟨b, hb⟩ : ∃ b, rr.size = h + b := ⟨rr.size - h, by omega⟩
  have hN : 2 ^ rr.size = 2 ^ b * 2 ^ h := by rw [hb, pow_add, mul_comm]
  have hwin : r.pos0 + r.count ≤ 2 ^ b := by
    have h2 : 2 ^ (rr.size + 6) = 2 ^ b * 2 ^ r.slotLog := by rw [pow_add, hN, hP]; ring
    rw [h2] at hin
    exact Nat.le_of_mul_le_mul_right hin (Nat.two_pow_pos _)
  have hlo : (eqTable (rr.extract 0 h)).size = 2 ^ h := by rw [eqTable_size]; simp; omega
  rw [hN, sum_range_blocks]
  have hterm : ∀ q ∈ Finset.range (2 ^ b), ∑ j ∈ Finset.range (2 ^ h), of (eqTable rr)[q * 2 ^ h + j]! *
      (if covers r (s + 64 * (q * 2 ^ h + j)) then
        of (eHi[(s + 64 * (q * 2 ^ h + j)) / 2 ^ r.slotLog]! * base[(s + 64 * (q * 2 ^ h + j)) % 2 ^ r.slotLog]!) else 0) =
      if r.pos0 ≤ q ∧ q < r.pos0 + r.count then
        of eHi[q]! * of (eqTable (rr.extract h rr.size))[q]! *
          ∑ j ∈ Finset.range (2 ^ h), of base[s + 64 * j]! * of (eqTable (rr.extract 0 h))[j]!
      else 0 := by
    intro q hq
    rw [Finset.mem_range] at hq
    have hj : ∀ j ∈ Finset.range (2 ^ h), of (eqTable rr)[q * 2 ^ h + j]! *
        (if covers r (s + 64 * (q * 2 ^ h + j)) then
          of (eHi[(s + 64 * (q * 2 ^ h + j)) / 2 ^ r.slotLog]! * base[(s + 64 * (q * 2 ^ h + j)) % 2 ^ r.slotLog]!)
          else 0) =
        if r.pos0 ≤ q ∧ q < r.pos0 + r.count then
          of eHi[q]! * of (eqTable (rr.extract h rr.size))[q]! * (of base[s + 64 * j]! * of (eqTable (rr.extract 0 h))[j]!)
        else 0 := by
      intro j hj
      rw [Finset.mem_range] at hj
      have hc : s + 64 * (q * 2 ^ h + j) = (s + 64 * j) + q * 2 ^ r.slotLog := by rw [hP]; ring
      have hlt : s + 64 * j < 2 ^ r.slotLog := by rw [hP]; omega
      have hdiv : (s + 64 * (q * 2 ^ h + j)) / 2 ^ r.slotLog = q := by
        rw [hc, Nat.add_mul_div_right _ _ (Nat.two_pow_pos _), Nat.div_eq_of_lt hlt, zero_add]
      have hmod : (s + 64 * (q * 2 ^ h + j)) % 2 ^ r.slotLog = s + 64 * j := by
        rw [hc, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hlt]
      have hcov : covers r (s + 64 * (q * 2 ^ h + j)) ↔ r.pos0 ≤ q ∧ q < r.pos0 + r.count := by
        unfold covers; rw [hdiv]
      have hu : q * 2 ^ h + j < 2 ^ rr.size := by
        rw [hN]; nlinarith
      rw [eqTable_split rr h (by omega) _ hu]
      have hm1 : (q * 2 ^ h + j) % 2 ^ h = j := by
        rw [add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
      have hm2 : (q * 2 ^ h + j) / 2 ^ h = q := by
        rw [add_comm, Nat.add_mul_div_right _ _ (Nat.two_pow_pos _), Nat.div_eq_of_lt hj, zero_add]
      rw [hm1, hm2]
      by_cases hw : r.pos0 ≤ q ∧ q < r.pos0 + r.count
      · rw [if_pos (hcov.mpr hw), if_pos hw, hdiv, hmod, of_mul]; ring
      · rw [if_neg (fun h' => hw (hcov.mp h')), if_neg hw, mul_zero]
    rw [Finset.sum_congr rfl hj]
    by_cases hw : r.pos0 ≤ q ∧ q < r.pos0 + r.count
    · simp only [if_pos hw]
      exact (Finset.mul_sum _ _ _).symm
    · simp only [if_neg hw, Finset.sum_const_zero]
  have hp : (Stmt.project base (eqTable (rr.extract 0 h)))[s]! =
      Stmt.sumUpTo (eqTable (rr.extract 0 h)).size fun u => base[s + 64 * u]! * (eqTable (rr.extract 0 h))[u]! := by
    unfold Stmt.project; rw [ofFn_get! 64 _ s hs]
  rw [Finset.sum_congr rfl hterm, sum_window _ _ _ hwin, hp, sumUpTo_eq, sumUpTo_eq, hlo]
  refine Eq.trans ?_ (Finset.sum_mul _ _ _).symm
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [of_mul]

/-! ### `partialRanges`, with the slot-type cache -/

/-- Range `r`'s share of entry `s` of `foldedPartial`, at the reversed challenges `rr`. -/
noncomputable def shareFor (st : Stmt) (alpha : F128) (lw rho rr : Array F128) (r : Range) (s : ℕ) : GF128 :=
  of (Stmt.sumUpTo r.count fun k =>
      (eHiFor rho r)[r.pos0 + k]! * (eqTable (rr.extract (r.slotLog - 6) rr.size))[r.pos0 + k]!) *
    of (Stmt.project (baseFor st alpha lw rho r) (eqTable (rr.extract 0 (r.slotLog - 6))))[s]!

/-- `partialRanges` adds each range's share, and wherever it succeeds `foldRanges` does too: both look up or build the
same slot types in the same order. -/
theorem partialRanges_get (st : Stmt) (alpha : F128) (lw rho rr : Array F128) : ∀ (rs : List Range)
    (bases : Array ((Kind × Nat) × Array F128)) (out res : Array F128),
    st.partialRanges alpha lw rho rr rs bases out = .ok res →
    (∀ kb ∈ bases, st.slotBase alpha lw rho kb.1 = .ok kb.2) → out.size = 64 →
    res.size = 64 ∧ (∀ s < 64, of res[s]! = of out[s]! + (rs.map fun r => shareFor st alpha lw rho rr r s).sum) ∧
      ∀ out', ∃ res', st.foldRanges alpha lw rho rs bases out' = .ok res' := by
  intro rs
  induction rs with
  | nil =>
    intro bases out res h _ hsz
    simp only [Stmt.partialRanges, pure, Except.pure] at h
    cases h
    refine ⟨hsz, fun s _ => by simp, fun out' => ⟨out', rfl⟩⟩
  | cons r rs ih =>
    intro bases out res h hinv hsz
    have hstep : ∀ (b : Array F128) (bases' : Array ((Kind × Nat) × Array F128)),
        baseFor st alpha lw rho r = b → (∀ kb ∈ bases', st.slotBase alpha lw rho kb.1 = .ok kb.2) →
        st.partialRanges alpha lw rho rr rs bases' (Array.ofFn (n := 64) fun s => out[s.val]! +
          Stmt.sumUpTo r.count (fun k => (eqTable (rho.extract (r.slotLog - 6) rho.size))[r.pos0 + k]! *
            (eqTable (rr.extract (r.slotLog - 6) rr.size))[r.pos0 + k]!) *
          (Stmt.project b (eqTable (rr.extract 0 (r.slotLog - 6))))[s.val]!) = .ok res →
        res.size = 64 ∧ (∀ s < 64, of res[s]! = of out[s]! + ((r :: rs).map fun r => shareFor st alpha lw rho rr r s).sum) ∧
          ∀ out', ∃ res', st.foldRanges alpha lw rho rs bases' out' = .ok res' := by
      intro b bases' hb hinv' h'
      obtain ⟨hsz', hget, hfr⟩ := ih bases' _ res h' hinv' (by simp)
      refine ⟨hsz', fun s hs => ?_, hfr⟩
      rw [hget s hs, ofFn_get! 64 _ s hs, List.map_cons, List.sum_cons, of_add, of_mul, shareFor, ← hb]
      simp only [eHiFor]
      ring
    simp only [Stmt.partialRanges] at h
    cases hf : bases.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
    | some kb =>
      rw [hf] at h
      simp only [bind, Except.bind, pure, Except.pure] at h
      have hkb := Array.find?_some hf
      have hmem := Array.mem_of_find?_eq_some hf
      have hkey : kb.1 = (r.kind, r.slotLog) := of_decide_eq_true hkb
      have hbase : baseFor st alpha lw rho r = kb.2 := by rw [baseFor, ← hkey, hinv kb hmem]
      obtain ⟨h1, h2, hfr⟩ := hstep kb.2 bases hbase hinv h
      refine ⟨h1, h2, fun out' => ?_⟩
      obtain ⟨res', hres'⟩ := hfr (Stmt.placeRange r (eqTable (rho.extract (r.slotLog - 6) rho.size)) kb.2 r.count 0 out')
      refine ⟨res', ?_⟩
      simp only [Stmt.foldRanges]
      rw [hf]
      simp only [bind, Except.bind, pure, Except.pure]
      exact hres'
    | none =>
      rw [hf] at h
      cases hb : st.slotBase alpha lw rho (r.kind, r.slotLog) with
      | error e => rw [hb] at h; simp [bind, Except.bind] at h
      | ok b =>
        rw [hb] at h
        simp only [bind, Except.bind, pure, Except.pure] at h
        have hbase : baseFor st alpha lw rho r = b := by rw [baseFor, hb]
        have hinv' : ∀ kb ∈ bases.push ((r.kind, r.slotLog), b), st.slotBase alpha lw rho kb.1 = .ok kb.2 := by
          intro kb hkb
          rcases Array.mem_push.mp hkb with h1 | h1
          · exact hinv kb h1
          · rw [h1]; exact hb
        obtain ⟨h1, h2, hfr⟩ := hstep b _ hbase hinv' h
        refine ⟨h1, h2, fun out' => ?_⟩
        obtain ⟨res', hres'⟩ := hfr (Stmt.placeRange r (eqTable (rho.extract (r.slotLog - 6) rho.size)) b r.count 0 out')
        refine ⟨res', ?_⟩
        simp only [Stmt.foldRanges]
        rw [hf, hb]
        simp only [bind, Except.bind, pure, Except.pure]
        exact hres'

/-! ### Δ and the pin -/

theorem addDelta_size (w : ℕ → F128) : ∀ (pairs : List (ℕ × ℕ)) (out : Array F128),
    (Stmt.addDelta w pairs out).size = out.size := by
  intro pairs
  induction pairs with
  | nil => intro out; rfl
  | cons ic pairs ih =>
    intro out
    have := ih (out.modify ic.2 (· + w ic.1))
    simp only [Stmt.addDelta, List.foldl_cons] at this ⊢
    rw [this]; simp

theorem fold_size (st : Stmt) (alpha : F128) (lw rho v : Array F128) (h : st.fold alpha lw rho = .ok v) :
    v.size = 2 ^ st.c.kLog := by
  simp only [Stmt.fold] at h
  split_ifs at h with hck
  · simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  cases hr : st.foldRanges alpha lw rho st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0) with
  | error e => rw [hr] at h; simp [bind, Except.bind] at h
  | ok res =>
    rw [hr] at h
    simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨hsz, -⟩ := foldRanges_get st alpha lw rho _ #[] _ res hr (by simp)
    rw [addDelta_size, addDelta_size, hsz, Array.size_replicate]

theorem addPart_size (n : ℕ) (eR : Array F128) (w : ℕ → F128) : ∀ (pairs : List (ℕ × ℕ)) (o : Array F128),
    (pairs.foldl (fun o ic => if ic.2 < n then o.modify (ic.2 % 64) (· + w ic.1 * eR[ic.2 / 64]!) else o) o).size =
      o.size := by
  intro pairs
  induction pairs with
  | nil => intro o; rfl
  | cons ic pairs ih => intro o; rw [List.foldl_cons, ih]; split_ifs <;> simp

/-- `foldedPartial`'s Δ: `(i, c)` adds `w(i)·eq(r, ⌊c/64⌋)` at `c mod 64` for each column `c` in the block. -/
theorem addPart_get (n : ℕ) (eR : Array F128) (w : ℕ → F128) : ∀ (pairs : List (ℕ × ℕ)) (o : Array F128) (s : ℕ),
    s < o.size →
    of (pairs.foldl (fun o ic => if ic.2 < n then o.modify (ic.2 % 64) (· + w ic.1 * eR[ic.2 / 64]!) else o) o)[s]! =
      of o[s]! + (pairs.map fun ic => if ic.2 < n ∧ ic.2 % 64 = s then of (w ic.1) * of eR[ic.2 / 64]! else 0).sum := by
  intro pairs
  induction pairs with
  | nil => intro o s _; simp
  | cons ic pairs ih =>
    intro o s hs
    rw [List.foldl_cons, ih _ s (by split_ifs <;> simpa using hs), List.map_cons, List.sum_cons]
    by_cases hn : ic.2 < n
    · rw [if_pos hn, modify_get! o _ s _ hs]
      by_cases hm : ic.2 % 64 = s
      · rw [if_pos hm, if_pos ⟨hn, hm⟩, of_add, of_mul]; ring
      · rw [if_neg hm, if_neg (fun h => hm h.2), zero_add]
    · rw [if_neg hn, if_neg (fun h => hn h.1), zero_add]

/-! ### `folded` -/

theorem foldedPartial_eq (st : Stmt) (alpha beta : F128) (lw rho ts w : Array F128) (hst : st.structured = true)
    (hts : ts.size + 6 = st.c.kLog) (h : st.foldedPartial alpha beta lw rho ts = .ok w) :
    ∃ v, st.fold alpha lw rho = .ok v ∧ w = ts.foldl (fun c t => halve t c) (v.modify st.pin (· + beta)) := by
  obtain ⟨_, hrs, hpw⟩ := structured_spec st hst
  simp only [Stmt.foldedPartial] at h
  by_cases hck : (rho.size + 6 != st.c.kLog || lw.size != 64) = true
  · rw [if_pos hck] at h
    simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  rw [if_neg hck] at h
  have hck' : rho.size + 6 = st.c.kLog ∧ lw.size = 64 := by simpa using hck
  cases hr : st.partialRanges alpha lw rho ts.reverse st.c.ranges.toList #[] (Array.replicate 64 0) with
  | error e => rw [hr] at h; simp [bind, Except.bind] at h
  | ok out =>
    rw [hr] at h
    simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨hosz, hoget, hfr⟩ := partialRanges_get st alpha lw rho ts.reverse _ #[] _ out hr (by simp) (by simp)
    obtain ⟨res, hres⟩ := hfr (Array.replicate (2 ^ st.c.kLog) 0)
    cases hv : st.fold alpha lw rho with
    | error e =>
      exfalso
      simp only [Stmt.fold] at hv
      split_ifs at hv
      rw [hres] at hv
      simp [bind, Except.bind, pure, Except.pure] at hv
    | ok v =>
      refine ⟨v, rfl, ?_⟩
      have hvsz := fold_size st alpha lw rho v hv
      have hN : 2 ^ st.c.kLog = 2 ^ 6 * 2 ^ ts.size := by rw [← hts, pow_add, mul_comm]
      have hv' : (v.modify st.pin (· + beta)).size = 2 ^ ts.toList.length * 64 := by
        rw [Array.size_modify, hvsz, hN, Array.length_toList]; ring
      have hrev : ts.toList.reverse.toArray = ts.reverse := Array.ext' (by simp)
      have hrsz : ts.reverse.size = ts.size := by simp
      have hRsz : (ts.toList.foldl (fun c t => halve t c) (v.modify st.pin (· + beta))).size = 64 := by
        rw [foldl_halve_size, hv', Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
      have hXsz : ∀ (wa wb : ℕ → F128), (st.db.toList.foldl (fun o ic => if ic.2 < 2 ^ st.c.kLog then
          o.modify (ic.2 % 64) (· + wb ic.1 * (eqTable ts.reverse)[ic.2 / 64]!) else o)
          (st.da.toList.foldl (fun o ic => if ic.2 < 2 ^ st.c.kLog then
            o.modify (ic.2 % 64) (· + wa ic.1 * (eqTable ts.reverse)[ic.2 / 64]!) else o) out)).size = 64 := by
        intro wa wb
        rw [addPart_size, addPart_size, hosz]
      rw [← Array.foldl_toList]
      refine array_ext_of _ _ ?_ (fun s hs => ?_)
      · rw [hRsz]
        split_ifs
        · rw [Array.size_modify]
          exact hXsz (fun i => alpha * (lw[i % 64]! * (eqTable rho)[i / 64]!)) (fun i => lw[i % 64]! * (eqTable rho)[i / 64]!)
        · exact hXsz (fun i => alpha * (lw[i % 64]! * (eqTable rho)[i / 64]!)) (fun i => lw[i % 64]! * (eqTable rho)[i / 64]!)
      rw [hRsz] at hs
      -- the halved side: each column's ranges, Δ entries and pin, weighted by `eq(ts reversed, ·)`
      rw [partial_eq_halve_fold ts.toList _ hv' s hs, hrev, Array.length_toList]
      have hterm : ∀ u ∈ Finset.range (2 ^ ts.size),
          of (eqTable ts.reverse)[u]! * of (v.modify st.pin (· + beta))[s + 64 * u]! =
            (st.c.ranges.toList.map fun r => of (eqTable ts.reverse)[u]! *
              (if covers r (s + 64 * u) then of (rangeTerm st alpha lw rho r (s + 64 * u)) else 0)).sum +
            (st.da.toList.map fun ic => of (eqTable ts.reverse)[u]! *
              (if ic.2 = s + 64 * u then of alpha * of (eIn lw rho ic.1) else 0)).sum +
            (st.db.toList.map fun ic => of (eqTable ts.reverse)[u]! *
              (if ic.2 = s + 64 * u then of (eIn lw rho ic.1) else 0)).sum +
            of (eqTable ts.reverse)[u]! * (if st.pin = s + 64 * u then of beta else 0) := by
        intro u hu
        rw [Finset.mem_range] at hu
        have hc : s + 64 * u < 2 ^ st.c.kLog := by rw [hN]; omega
        rw [modify_get! v _ _ _ (by rw [hvsz]; exact hc), ← list_mul_sum, ← list_mul_sum, ← list_mul_sum]
        have hfv := fold_get st alpha lw rho v hv _ hc
        rw [rangesVal_sum st alpha lw rho _ hpw] at hfv
        split_ifs with hp
        · rw [of_add, hfv]; simp only [rangeTerm]; ring
        · rw [hfv]; simp only [rangeTerm]; ring
      rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib,
        sum_list_comm, sum_list_comm, sum_list_comm, sum_spot _ _ _ hs]
      have hshare : (st.c.ranges.toList.map fun r => ∑ u ∈ Finset.range (2 ^ ts.size), of (eqTable ts.reverse)[u]! *
          (if covers r (s + 64 * u) then of (rangeTerm st alpha lw rho r (s + 64 * u)) else 0)) =
          st.c.ranges.toList.map fun r => shareFor st alpha lw rho ts.reverse r s := by
        refine List.map_congr_left fun r hr => ?_
        obtain ⟨h1, h2, h3⟩ := hrs r hr
        rw [← hrsz]
        exact range_share r ts.reverse (eHiFor rho r) (baseFor st alpha lw rho r) h1 (by rw [hrsz]; omega)
          (by rw [hrsz, hts]; exact h3) s hs
      have hspot : ∀ (x : ℕ × ℕ → GF128) (l : List (ℕ × ℕ)), (l.map fun ic => ∑ u ∈ Finset.range (2 ^ ts.size),
          of (eqTable ts.reverse)[u]! * (if ic.2 = s + 64 * u then x ic else 0)) =
          l.map fun ic => if ic.2 < 2 ^ st.c.kLog ∧ ic.2 % 64 = s then x ic * of (eqTable ts.reverse)[ic.2 / 64]! else 0 := by
        intro x l
        refine List.map_congr_left fun ic _ => ?_
        rw [sum_spot _ _ _ hs, hN]
        split_ifs <;> ring
      rw [hshare, hspot (fun ic => of alpha * of (eIn lw rho ic.1)), hspot (fun ic => of (eIn lw rho ic.1))]
      -- the 64 entries `foldedPartial` computed
      have hpinL : ∀ X : Array F128, X.size = 64 →
          of (if st.pin < 2 ^ st.c.kLog then X.modify (st.pin % 64) (fun x => x + beta * (eqTable ts.reverse)[st.pin / 64]!)
            else X)[s]! =
            of X[s]! + (if st.pin < 2 ^ st.c.kLog ∧ st.pin % 64 = s then
              of beta * of (eqTable ts.reverse)[st.pin / 64]! else 0) := by
        intro X hX
        by_cases h1 : st.pin < 2 ^ st.c.kLog
        · rw [if_pos h1, modify_get! X _ s _ (by omega)]
          by_cases h2 : st.pin % 64 = s
          · rw [if_pos h2, if_pos ⟨h1, h2⟩, of_add, of_mul]
          · rw [if_neg h2, if_neg (fun h => h2 h.2), add_zero]
        · rw [if_neg h1, if_neg (fun h => h1 h.1), add_zero]
      rw [hpinL _ (hXsz (fun i => alpha * (lw[i % 64]! * (eqTable rho)[i / 64]!)) (fun i => lw[i % 64]! * (eqTable rho)[i / 64]!))]
      have hYsz : ∀ (wa : ℕ → F128), (st.da.toList.foldl (fun o ic => if ic.2 < 2 ^ st.c.kLog then
          o.modify (ic.2 % 64) (· + wa ic.1 * (eqTable ts.reverse)[ic.2 / 64]!) else o) out).size = 64 := by
        intro wa
        rw [addPart_size, hosz]
      rw [addPart_get (2 ^ st.c.kLog) (eqTable ts.reverse) (fun i => lw[i % 64]! * (eqTable rho)[i / 64]!) _ _ s
        (lt_of_lt_of_eq hs (hYsz (fun i => alpha * (lw[i % 64]! * (eqTable rho)[i / 64]!))).symm)]
      rw [addPart_get (2 ^ st.c.kLog) (eqTable ts.reverse) (fun i => alpha * (lw[i % 64]! * (eqTable rho)[i / 64]!)) _ _ s
        (lt_of_lt_of_eq hs hosz.symm)]
      rw [hoget s hs, getElem!_pos _ s (by simpa using hs), Array.getElem_replicate, of_zero', zero_add]
      simp only [eIn, of_mul]
      rw [← hN, mul_comm (of (eqTable ts.reverse)[st.pin / 64]!) (of beta)]

/-- **The folded comb.** When `Stmt.folded` succeeds, its 64 entries are `Stmt.fold`'s vector with `β` added at the pin,
halved at `ts` in turn: whether it read the block's structure (`foldedPartial`) or folded flat. -/
theorem folded_eq (st : Stmt) (alpha beta : F128) (lw rho ts w : Array F128)
    (h : st.folded alpha beta lw rho ts = .ok w) :
    ∃ v, st.fold alpha lw rho = .ok v ∧ w = ts.foldl (fun c t => halve t c) (v.modify st.pin (· + beta)) := by
  unfold Stmt.folded at h
  by_cases hs : (st.structured && ts.size + 6 == st.c.kLog) = true
  · rw [if_pos hs] at h
    simp only [Bool.and_eq_true, beq_iff_eq] at hs
    exact foldedPartial_eq st alpha beta lw rho ts w hs.1 hs.2 h
  · rw [if_neg hs] at h
    cases hv : st.fold alpha lw rho with
    | error e => rw [hv] at h; simp [bind, Except.bind] at h
    | ok v =>
      rw [hv] at h
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨v, rfl, h.symm⟩

end FlockLevel3
