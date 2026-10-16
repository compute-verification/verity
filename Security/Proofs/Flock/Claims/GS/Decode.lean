import Proofs.Flock.Claims.GS.Interp
import Proofs.Flock.Claims.GS.RR
import Proofs.Flock.Claims.GS.Johnson

/-!
# The counted Guruswami–Sudan decoder for interleaved Reed–Solomon codes

`gsDecode C rf rd agree` decodes the interleaved Reed–Solomon code `C` (`IRS`: `s` rows of degree below `k`) from opened
columns `os`, computing on coefficient lists and tables only, and `gsCost` counts its field operations along the same
computation. With `N` opened columns and threshold `t = ⌈agree·N⌉` (`IRS.threshold`):

1. Each opened position's point `C.dom q` (`cols`, `dc` operations each), and a check that the points are distinct
   (`N²` comparisons) and that `(k+1)·N < t² ∧ t ≤ N` (`Reg`); outside that, `[]`.
2. Per row `r`: the Guruswami–Sudan interpolation `Q_r(X, Y)` through the row's `N` points with multiplicity `k + 1`
   (`interpTab`, Gaussian elimination, at most `20·N^9` operations, `interpCost_le`), written as a table, and its
   `Y`-roots of degree below `k` by Roth–Ruckenstein (`rr`, with the root finder `rf`) (`cands`).
3. The rows' candidates joined row by row, keeping the prefixes that agree with at least `t` opened columns (`kept`,
   `extendRows`), each test by Horner's rule (`jcC`). Every kept list after the first row has at most `N` prefixes
   (`length_kept_le`): two kept prefixes differ in a row, two distinct polynomials of degree below `k` agree on at most
   `k − 1` of the distinct points, and at `(k+1)·N < t²` Johnson's bound (`johnson_card_le`) caps the family at `N`.
4. Each full prefix read back as a message (`Reader`, `rd.cost` operations each).

On `s = 0` rows it returns the one message `rd.read []`.

**Completeness** (`mem_gsDecode`): on opened columns at distinct points, from the Johnson count `C.nMin agree` on, at
`agree > 0`, every message agreeing with at least `agree·N` of them is listed, the conclusion of
`IRS.listComplete_decode` on those inputs. A message's row agrees with its row's points at least `t` times, so the
interpolation vanishes on it (`interpPoly_eval_eq_zero`, from ArkLib's `GuruswamiSudan` multiplicity lemmas), so
Roth–Ruckenstein finds it (`mem_rr`), and its prefixes pass every pruning (`mem_extendRows`).

**Complexity** (`gsDecode_complexity`): at the input size `N·s·w` (the opened columns' field elements times the field's
width `w` in bits: the length of the opened data, the positions apart, which reach the decoder only through `C.dom`),
the count is at most `gsBound ρ β dc` at the size, with `ρ` the root finder's bound, `β` the reader's and `dc` the cost
of one point:

`20·X^9 + 171·X^7 + 9·X^6 + 12·X^4 + X^2 + dc·X + 1 + 3·X^2·ρ(3·X) + (X + 1)·β(X)`.

The interpolation's Gaussian elimination dominates: `N·(k+1)(k+2)/2` constraints in as many unknowns plus one, so
`O(N^3)` of each at `k < N`, cubed.
-/

namespace Claims.GS

open Polynomial

/-- **A counted reader** for the code `C`: a message from its rows' coefficient lists, in `cost` field operations, at
most `bound` at `s·k`. -/
structure Reader {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) where
  read : List (List F) → T
  read_eq : ∀ ps : List (List F), ps.length = s → (∀ p ∈ ps, p.length = C.k) →
    read ps = C.ofRows fun r => toPolyL (ps.getD r [])
  cost : ℕ
  bound : Polynomial ℕ
  cost_le : cost ≤ bound.eval (s * C.k)

/-- The opened columns at their points. -/
def cols {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (os : List (ℕ × (Fin s → F))) : List (F × (Fin s → F)) :=
  os.map fun p => (C.dom p.1, p.2)

/-- Row `r` of the opened columns: the points with the row's values. -/
def rowPts {F : Type} {s : ℕ} (cs : List (F × (Fin s → F))) (r : Fin s) : List (F × F) :=
  cs.map fun c => (c.1, c.2 r)

/-- The rows `ps` (coefficient lists) agree with the column `c` at the point `x`, by Horner's rule. -/
def fitsC {F : Type} [Field F] [DecidableEq F] {s : ℕ} (ps : List (List F)) (x : F) (c : Fin s → F) : Bool :=
  decide (∀ r : Fin s, r.val < ps.length → horner (ps.getD r []) x = c r)

/-- How many opened columns the rows `ps` agree with. -/
def jcC {F : Type} [Field F] [DecidableEq F] {s : ℕ} (ps : List (List F)) (cs : List (F × (Fin s → F))) : ℕ :=
  (cs.filter fun c => fitsC ps c.1 c.2).length

/-- Row `m`'s candidates: the `Y`-roots of degree below `k` of the row's interpolation polynomial (none past the last
row). -/
def cands {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) (k t : ℕ) (cs : List (F × (Fin s → F)))
    (m : ℕ) : List (List F) :=
  if h : m < s then rr rf (degX k t) (degY k t) (mkTab (degX k t) (degY k t) (interpTab k t (rowPts cs ⟨m, h⟩))) k
  else []

/-- The prefixes of `m` rows that agree with at least `t` opened columns. -/
def kept {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) (k t : ℕ) (cs : List (F × (Fin s → F)))
    (m : ℕ) : List (List (List F)) :=
  extendRows (cands rf k t cs) (fun ps => decide (t ≤ jcC ps cs)) m

/-- **The counted Guruswami–Sudan decoder.** -/
noncomputable def gsDecode {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (rf : RootFinder F)
    (rd : Reader C) (agree : ℝ) (os : List (ℕ × (Fin s → F))) : List T :=
  if s = 0 then [rd.read []]
  else if Reg C.k os.length (IRS.threshold agree os) ∧ ((cols C os).map Prod.fst).Nodup then
    (kept rf C.k (IRS.threshold agree os) (cols C os) s).map rd.read
  else []

/-- One pruning test on a prefix of `m + 1` rows: per opened column, `m + 1` Horner evaluations of `2k` operations and
a comparison each, and a count; then the comparison with `t`. -/
def keepCost (k N m : ℕ) : ℕ := N * ((m + 1) * (2 * k + 1) + 1) + 1

/-- Row `r`'s operations: the interpolation, writing its table (each entry looked up among the unknowns), and
Roth–Ruckenstein. -/
def rowCost {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) (k t : ℕ) (cs : List (F × (Fin s → F)))
    (r : Fin s) : ℕ :=
  interpCost k t (rowPts cs r) + (degX k t + 1) * (degY k t + 1) * (nvars k cs.length + 1) +
    rrCost rf (degX k t) (degY k t) (mkTab (degX k t) (degY k t) (interpTab k t (rowPts cs r))) k

/-- The pruning's operations: one test per kept prefix and candidate, row by row. -/
def pruneCost {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) (k t : ℕ)
    (cs : List (F × (Fin s → F))) : ℕ :=
  ∑ m ∈ Finset.range s, (kept rf k t cs m).length * (cands rf k t cs m).length * keepCost k cs.length m

/-- **`gsDecode`'s field operations**, along its computation, with `dc` operations per point. -/
noncomputable def gsCost {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (rf : RootFinder F)
    (rd : Reader C) (agree : ℝ) (dc : ℕ) (os : List (ℕ × (Fin s → F))) : ℕ :=
  if s = 0 then rd.cost
  else os.length * dc + os.length * os.length + 1 +
    if Reg C.k os.length (IRS.threshold agree os) ∧ ((cols C os).map Prod.fst).Nodup then
      ∑ r : Fin s, rowCost rf C.k (IRS.threshold agree os) (cols C os) r +
        pruneCost rf C.k (IRS.threshold agree os) (cols C os) +
        (kept rf C.k (IRS.threshold agree os) (cols C os) s).length * rd.cost
    else 0

/-- `gsDecode` with its count. -/
noncomputable def gsCounted {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (rf : RootFinder F)
    (rd : Reader C) (agree : ℝ) (dc : ℕ) (os : List (ℕ × (Fin s → F))) : List T × ℕ :=
  (gsDecode C rf rd agree os, gsCost C rf rd agree dc os)

/-! ## Structure of the pruning -/

theorem extendRows_spec {α : Type} (L : ℕ → List α) (keep : List α → Bool) :
    ∀ m, ∀ ps ∈ extendRows L keep m,
      ps.length = m ∧ (∀ i (hi : i < ps.length), ps[i] ∈ L i) ∧ (m ≠ 0 → keep ps = true)
  | 0, ps, h => by
    simp only [extendRows, List.mem_singleton] at h
    subst h
    simp
  | m + 1, ps, h => by
    rw [extendRows, List.mem_filter, List.mem_flatMap] at h
    obtain ⟨⟨qs, hqs, hps⟩, hk⟩ := h
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hps
    obtain ⟨hl, hi, -⟩ := extendRows_spec L keep m qs hqs
    refine ⟨by simp [hl], fun i hi' => ?_, fun _ => hk⟩
    rw [List.getElem_append]
    split_ifs with h1
    · exact hi i h1
    · simp only [List.length_append, List.length_singleton] at hi'
      have : i = qs.length := by omega
      subst this
      simpa [hl] using hp

theorem nodup_extendRows {α : Type} (L : ℕ → List α) (keep : List α → Bool) (hL : ∀ m, (L m).Nodup) :
    ∀ m, (extendRows L keep m).Nodup
  | 0 => by simp [extendRows]
  | m + 1 => by
    have ih := nodup_extendRows L keep hL m
    rw [extendRows]
    refine List.Nodup.filter _ (List.nodup_flatMap.2 ⟨fun qs _ => ?_, ?_⟩)
    · exact (hL m).map fun p p' h => by simpa using h
    · refine ih.imp_of_mem fun {a b} ha hb hab => ?_
      simp only [Function.onFun]
      intro x hxa hxb
      obtain ⟨p, -, rfl⟩ := List.mem_map.1 hxa
      obtain ⟨p', -, he⟩ := List.mem_map.1 hxb
      have hla := (extendRows_spec L keep m a ha).1
      have hlb := (extendRows_spec L keep m b hb).1
      exact hab (List.append_inj he (by omega)).1.symm

theorem cands_shape {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) (k t : ℕ)
    (cs : List (F × (Fin s → F))) (m : ℕ) :
    (cands rf k t cs m).Nodup ∧ ∀ p ∈ cands rf k t cs m, p.length = k := by
  unfold cands
  split_ifs with h
  · exact rr_shape rf _ k
  · simp

theorem kept_spec {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) (k t : ℕ)
    (cs : List (F × (Fin s → F))) (m : ℕ) (ps : List (List F)) (h : ps ∈ kept rf k t cs m) :
    ps.length = m ∧ (∀ i (hi : i < ps.length), ps[i].length = k) ∧ (m ≠ 0 → t ≤ jcC ps cs) := by
  obtain ⟨hl, hi, hk⟩ := extendRows_spec _ _ m ps h
  refine ⟨hl, fun i hi' => (cands_shape rf k t cs i).2 _ (hi i hi'), fun hm => ?_⟩
  simpa using hk hm

theorem nodup_kept {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) (k t : ℕ)
    (cs : List (F × (Fin s → F))) (m : ℕ) : (kept rf k t cs m).Nodup :=
  nodup_extendRows _ _ (fun i => (cands_shape rf k t cs i).1) m

/-! ## Completeness -/

theorem length_cols {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (os : List (ℕ × (Fin s → F))) :
    (cols C os).length = os.length := by
  simp [cols]

theorem cols_points {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (os : List (ℕ × (Fin s → F))) :
    (cols C os).map Prod.fst = os.map fun p => C.dom p.1 := by
  simp [cols]

theorem rowPts_points {F : Type} {s : ℕ} (cs : List (F × (Fin s → F))) (r : Fin s) :
    (rowPts cs r).map Prod.fst = cs.map Prod.fst := by
  simp [rowPts]

theorem length_rowPts {F : Type} {s : ℕ} (cs : List (F × (Fin s → F))) (r : Fin s) :
    (rowPts cs r).length = cs.length := by
  simp [rowPts]

/-- A message's row agrees with its row's points wherever the message agrees with an opened column. -/
theorem agreeCount_le_rowPts {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (M : T) (r : Fin s) :
    C.agreeCount M os ≤ ((rowPts (cols C os) r).filter fun p => decide ((C.rows M r).eval p.1 = p.2)).length := by
  unfold IRS.agreeCount rowPts cols
  rw [List.map_map, List.filter_map, List.length_map]
  refine (List.monotone_filter_right os fun p hp => ?_).length_le
  simp only [decide_eq_true_eq, Function.comp_apply] at hp ⊢
  rw [← hp]
  rfl

/-- In the regime the table of the interpolation polynomial reads back as that polynomial. -/
theorem tpoly_interp {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) :
    tpoly (degX k t) (degY k t) (mkTab (degX k t) (degY k t) (interpTab k t pts)) = interpPoly k t pts := by
  unfold tpoly interpPoly
  congr 1
  funext i j
  rw [tget_mkTab]
  split_ifs with hij
  · rfl
  · by_contra hne
    exact hij (interpTab_boxed h i j (Ne.symm hne))

theorem tget_interp {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) (i j : ℕ) :
    tget (mkTab (degX k t) (degY k t) (interpTab k t pts)) i j = interpTab k t pts i j := by
  rw [tget_mkTab]
  split_ifs with hij
  · rfl
  · by_contra hne
    exact hij (interpTab_boxed h i j (Ne.symm hne))

/-- **A message's row is among its row's candidates**, in the regime, on distinct points, when the message agrees with
at least `t` opened columns. -/
theorem mem_cands {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (rf : RootFinder F) {t : ℕ}
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (h : Reg C.k os.length t) (M : T)
    (hM : t ≤ C.agreeCount M os) (m : ℕ) (hm : m < s) :
    coeffList C.k (C.rows M ⟨m, hm⟩) ∈ cands rf C.k t (cols C os) m := by
  rw [cands, dite_eq_left hm]
  have hlen : (rowPts (cols C os) ⟨m, hm⟩).length = os.length := by rw [length_rowPts, length_cols]
  have h' : Reg C.k (rowPts (cols C os) ⟨m, hm⟩).length t := by rwa [hlen]
  have hnd' : ((rowPts (cols C os) ⟨m, hm⟩).map Prod.fst).Nodup := by rwa [rowPts_points, cols_points]
  refine mem_rr rf ?_ ?_ (C.natDegree_rows M _) ?_
  · rw [tpoly_interp h']
    exact interpPoly_ne_zero h'
  · intro i j hij
    rw [tget_interp h'] at hij
    exact interpTab_wdeg h' hij
  · rw [tpoly_interp h']
    exact interpPoly_eval_eq_zero hnd' h' (C.natDegree_rows M _)
      (hM.trans (agreeCount_le_rowPts C os M ⟨m, hm⟩))

/-- The prefix of a message's rows, as coefficient lists. -/
def rowsPre {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (M : T) (i : ℕ) : List F :=
  if h : i < s then coeffList C.k (C.rows M ⟨i, h⟩) else []

/-- **A message's prefixes agree with every opened column the message agrees with.** -/
theorem agreeCount_le_jcC {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (M : T) (m : ℕ) :
    C.agreeCount M os ≤ jcC ((List.range m).map (rowsPre C M)) (cols C os) := by
  unfold IRS.agreeCount jcC cols
  rw [List.filter_map, List.length_map]
  refine (List.monotone_filter_right os fun p hp => ?_).length_le
  simp only [decide_eq_true_eq] at hp
  simp only [Function.comp_apply, fitsC, decide_eq_true_eq, List.length_map, List.length_range]
  intro r hr
  rw [List.getD_eq_getElem _ _ (by simpa using hr), List.getElem_map, List.getElem_range, rowsPre, dite_eq_left r.isLt,
    horner_eq, toPolyL_coeffList _ _ (C.natDegree_rows M _), ← hp]
  rfl

/-- From the Johnson count on, the threshold is in the regime, when some message agrees with `agree·N` columns. -/
theorem reg_threshold {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) {agree : ℝ} (ha : 0 < agree)
    (os : List (ℕ × (Fin s → F))) (hN : C.nMin agree ≤ os.length) (M : T)
    (hM : agree * os.length ≤ C.agreeCount M os) : Reg C.k os.length (IRS.threshold agree os) := by
  have hJ := (johnsonCount_le_iff ha).1 hN
  have ht : agree * os.length ≤ (IRS.threshold agree os : ℝ) := Nat.le_ceil _
  have hc : (C.agreeCount M os : ℝ) ≤ os.length := by
    exact_mod_cast List.length_filter_le _ _
  refine ⟨?_, Nat.ceil_le.2 (hM.trans hc)⟩
  have hs := Real.sq_sqrt (show (0 : ℝ) ≤ (C.k + 1) * os.length by positivity)
  have h0 := Real.sqrt_nonneg ((C.k + 1) * os.length)
  have : ((C.k + 1 : ℕ) * os.length : ℝ) < (IRS.threshold agree os : ℝ) * (IRS.threshold agree os : ℝ) := by
    push_cast
    nlinarith
  exact_mod_cast this

/-- **The counted decoder is complete**: on opened columns at distinct points, from the Johnson count on, at
`agree > 0`, it lists every message that agrees with at least `agree·N` of them. -/
theorem mem_gsDecode {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (rf : RootFinder F)
    (rd : Reader C) {agree : ℝ} (ha : 0 < agree) (os : List (ℕ × (Fin s → F)))
    (hnd : (os.map fun p => C.dom p.1).Nodup) (hN : C.nMin agree ≤ os.length) (M : T)
    (hM : agree * os.length ≤ C.agreeCount M os) : M ∈ gsDecode C rf rd agree os := by
  unfold gsDecode
  by_cases hs : s = 0
  · rw [ite_eq_left hs, List.mem_singleton, rd.read_eq [] (by simp [hs]) (by simp)]
    conv_lhs => rw [← C.ofRows_rows M]
    congr 1
    funext r
    exact absurd r.isLt (by omega)
  rw [ite_eq_right hs]
  have hreg := reg_threshold C ha os hN M hM
  have hnd' : ((cols C os).map Prod.fst).Nodup := by rwa [cols_points]
  rw [ite_eq_left ⟨hreg, hnd'⟩]
  have ht : IRS.threshold agree os ≤ C.agreeCount M os := Nat.ceil_le.2 hM
  have hpre := mem_extendRows (cands rf C.k (IRS.threshold agree os) (cols C os))
    (fun ps => decide (IRS.threshold agree os ≤ jcC ps (cols C os))) (rowsPre C M) s
    (fun m hm => by
      rw [rowsPre, dite_eq_left hm]
      exact mem_cands C rf os hnd hreg M ht m hm)
    (fun m _ => decide_eq_true (ht.trans (agreeCount_le_jcC C os M m))) s le_rfl
  refine List.mem_map.2 ⟨_, hpre, ?_⟩
  rw [rd.read_eq _ (by simp) (fun p hp => ?_)]
  · conv_rhs => rw [← C.ofRows_rows M]
    congr 1
    funext r
    rw [List.getD_eq_getElem _ _ (by simp), List.getElem_map, List.getElem_range, rowsPre, dite_eq_left r.isLt,
      toPolyL_coeffList _ _ (C.natDegree_rows M _)]
  · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hp
    rw [rowsPre, dite_eq_left (List.mem_range.1 hi), length_coeffList]

end Claims.GS
