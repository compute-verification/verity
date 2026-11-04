/-
Level 3: the lincheck's `Stmt.folded` (`Flock.Piop`). `folded_eq`: when `Stmt.folded` succeeds, its vector is
`Stmt.fold`'s with `β` at the pin, halved at `ts` in turn, whether it read the block's structure (`foldedPartial`, a
templated unit's range from its template: `partialRangesT_eq`) or folded flat. The lemmas it reads are `FoldedBase`.
-/
import Proofs.Flock.Level3.FoldedBase
import Proofs.Flock.Soundness.TmplShare

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

open GF128

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
  rw [FlockSoundness.TmplFold.partialRangesT_eq st alpha lw rho ts.reverse _ #[] #[] _ (by simp [FlockSoundness.TmplFold.CacheOk])
    (by simp [FlockSoundness.TmplFold.CacheOk])] at h
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
