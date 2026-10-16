import Proofs.Flock.Claims.GS.Interp
import Proofs.Flock.Claims.GS.RR
import Proofs.Flock.Claims.GS.Johnson

/-!
# The counted Guruswami–Sudan decoder for interleaved Reed–Solomon codes

`gsDecode C rf rd agree` decodes the interleaved Reed–Solomon code `C` (`IRS`: `s` rows of degree below `k`) from opened
columns `os`, computing on coefficient lists and tables only, and `gsCost` counts its field operations (comparisons
included) along the same computation. With `N` opened columns and threshold `t = ⌈agree·N⌉` (`IRS.threshold`):

1. Each opened position's point `C.dom q` (`cols`, `dc` operations each), and a check that the points are distinct
   (`N²` comparisons) and that `(k+1)·N < t² ∧ t ≤ N` (`Reg`); outside that, `[]`.
2. Per row `r`: the Guruswami–Sudan interpolation `Q_r(X, Y)` through the row's `N` points with multiplicity `k + 1`
   (`interpTab`, Gaussian elimination, at most `20·N^9` operations, `interpCost_le`), written as a table, and its
   `Y`-roots of degree below `k` by Roth–Ruckenstein (`rr`, with the root finder `rf`) (`cands`).
3. The rows' candidates joined row by row, keeping the prefixes that agree with at least `t` opened columns (`kept`,
   `extendRows`), each test by Horner's rule (`jcC`). Every kept list has at most `N` prefixes
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

/-! ## The pruning's lists, by Johnson's bound -/

/-- Two distinct polynomials of degree below `k` agree on at most `k − 1` of distinct points. -/
theorem card_agree_le {F : Type} [Field F] [DecidableEq F] {N : ℕ} (x : Fin N → F) (hx : Function.Injective x)
    {f g : F[X]} (hfg : f ≠ g) {k : ℕ} (hf : f.natDegree < k) (hg : g.natDegree < k) :
    (Finset.univ.filter fun j => f.eval (x j) = g.eval (x j)).card ≤ k - 1 := by
  have hd : f - g ≠ 0 := sub_ne_zero.2 hfg
  calc (Finset.univ.filter fun j => f.eval (x j) = g.eval (x j)).card ≤ (f - g).roots.toFinset.card := by
        refine Finset.card_le_card_of_injOn x (fun j hj => ?_) (fun a _ b _ h => hx h)
        simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hj
        rw [Finset.mem_coe, Multiset.mem_toFinset, mem_roots hd, IsRoot.def, eval_sub, hj, sub_self]
    _ ≤ (f - g).roots.card := Multiset.toFinset_card_le _
    _ ≤ (f - g).natDegree := card_roots' _
    _ ≤ k - 1 := by
        have := natDegree_sub_le f g
        omega

theorem lt_of_mem_cands {F : Type} [Field F] [DecidableEq F] {s : ℕ} {rf : RootFinder F} {k t : ℕ}
    {cs : List (F × (Fin s → F))} {m : ℕ} {p : List F} (h : p ∈ cands rf k t cs m) : m < s := by
  unfold cands at h
  split_ifs at h with hm
  · exact hm
  · simp at h

/-- **Every kept list has at most `N` prefixes**, in the regime, on distinct points: Johnson's bound on the kept
prefixes' agreement sets. -/
theorem length_kept_le {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) {k t : ℕ}
    {cs : List (F × (Fin s → F))} (h : Reg k cs.length t) (hnd : (cs.map Prod.fst).Nodup) (m : ℕ) :
    (kept rf k t cs m).length ≤ cs.length := by
  have hN : 1 ≤ cs.length := by have := h.k_lt; omega
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · simpa [kept, extendRows] using hN
  rw [← List.toFinset_card_of_nodup (nodup_kept rf k t cs m)]
  have hdiff : ∀ ps ∈ (kept rf k t cs m).toFinset, ∀ ps' ∈ (kept rf k t cs m).toFinset, ps ≠ ps' →
      ∃ i, ∃ (hi : i < ps.length) (hi' : i < ps'.length), ps[i] ≠ ps'[i] := by
    intro ps hps ps' hps' hne
    rw [List.mem_toFinset] at hps hps'
    by_contra hc
    push Not at hc
    exact hne (List.ext_getElem (by rw [(kept_spec rf k t cs m ps hps).1, (kept_spec rf k t cs m ps' hps').1])
      fun i h1 h2 => hc i h1 h2)
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · refine (Finset.card_le_one.2 fun a ha b hb => ?_).trans hN
    by_contra hab
    obtain ⟨i, hi, hi', hne⟩ := hdiff a ha b hb hab
    rw [List.mem_toFinset] at ha hb
    exact hne (by rw [List.length_eq_zero_iff.1 ((kept_spec rf 0 t cs m a ha).2.1 i hi),
      List.length_eq_zero_iff.1 ((kept_spec rf 0 t cs m b hb).2.1 i hi')])
  have hx : Function.Injective fun j : Fin cs.length => (cs.get j).1 := by
    intro j j' hjj
    have := (List.Nodup.getElem_inj_iff hnd (i := j.val) (j := j'.val) (hi := by simp) (hj := by simp)).1
      (by simpa using hjj)
    exact Fin.ext this
  refine johnson_card_le _ (fun ps => Finset.univ.filter fun j => fitsC ps (cs.get j).1 (cs.get j).2 = true)
    (t := t) (a := k - 1) (fun ps hps => ?_) (fun ps hps ps' hps' hne => ?_)
    (by rw [show k - 1 + 2 = k + 1 by omega]; exact h.1) h.2
  · rw [List.mem_toFinset] at hps
    have e := card_filter_get (fun c : F × (Fin s → F) => fitsC ps c.1 c.2 = true) cs
    simp only [Bool.decide_eq_true] at e
    rw [e]
    exact (kept_spec rf k t cs m ps hps).2.2 (by omega)
  · obtain ⟨i, hi, hi', hdi⟩ := hdiff ps hps ps' hps' hne
    rw [List.mem_toFinset] at hps hps'
    have hik : ps[i].length = k := (kept_spec rf k t cs m ps hps).2.1 i hi
    have hik' : ps'[i].length = k := (kept_spec rf k t cs m ps' hps').2.1 i hi'
    have his : i < s := lt_of_mem_cands ((extendRows_spec _ _ m ps hps).2.1 i hi)
    have hpne : toPolyL ps[i] ≠ toPolyL ps'[i] := fun he => hdi (toPolyL_inj (by rw [hik, hik']) he)
    have hd : ∀ q : List F, q.length = k → (toPolyL q).natDegree < k := fun q hq => by
      have := natDegree_toPolyL_lt q (List.ne_nil_of_length_pos (by omega))
      rwa [hq] at this
    refine le_trans (Finset.card_le_card fun j hj => ?_) (card_agree_le _ hx hpne (hd _ hik) (hd _ hik'))
    simp only [Finset.mem_inter, Finset.mem_filter, Finset.mem_univ, true_and, fitsC, decide_eq_true_eq] at hj ⊢
    have h1 := hj.1 ⟨i, his⟩ hi
    have h2 := hj.2 ⟨i, his⟩ hi'
    rw [List.getD_eq_getElem _ _ hi, horner_eq] at h1
    rw [List.getD_eq_getElem _ _ hi', horner_eq] at h2
    rw [h1, h2]

/-! ## The count -/

theorem degY_le_degX (k t : ℕ) : degY k t ≤ degX k t := by
  unfold degY
  split_ifs
  · exact Nat.div_le_self _ _
  · exact le_rfl

/-- The `Y`-degree bound is below `3t`: `L + 1 ≤ 3t ≤ 3N`. -/
theorem degY_succ_le {k N t : ℕ} (h : Reg k N t) : degY k t + 1 ≤ 3 * N := by
  have ht := h.pos
  have hD := degX_succ h
  refine le_trans ?_ (Nat.mul_le_mul_left 3 h.2)
  unfold degY
  split_ifs with hk
  · have hlt : degX k t / (k - 1) < 3 * t := by
      rw [Nat.div_lt_iff_lt_mul (by omega)]
      have : (k + 1) * t ≤ 3 * t * (k - 1) := by
        rw [show 3 * t * (k - 1) = (3 * (k - 1)) * t by ring]
        exact Nat.mul_le_mul_right t (by omega)
      omega
    omega
  · have : (k + 1) * t ≤ 2 * t := Nat.mul_le_mul_right t (by omega)
    omega

/-- **One row's count**: at most `20·N^9 + 171·N^7 + 9·N^6 + 3·N²·ρ(3N)` in the regime. -/
theorem rowCost_le {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) {k t : ℕ}
    {cs : List (F × (Fin s → F))} (h : Reg k cs.length t) (r : Fin s) :
    rowCost rf k t cs r ≤ 20 * cs.length ^ 9 + 171 * cs.length ^ 7 + 9 * cs.length ^ 6 +
      3 * cs.length ^ 2 * rf.bound.eval (3 * cs.length) := by
  have hlen : (rowPts cs r).length = cs.length := length_rowPts cs r
  have hI := interpCost_le (show Reg k (rowPts cs r).length t by rwa [hlen])
  rw [hlen] at hI
  unfold rowCost
  generalize cs.length = N at h hI ⊢
  have hk := h.k_lt
  have hkN : k ≤ N := by omega
  have hD := degX_le h
  have hL := degY_succ_le h
  have hL4 : degY k t + 4 ≤ 6 * N := by omega
  have hR := rrCost_le rf (degY_le_degX k t) (interpTab k t (rowPts cs r)) k
  have hnv : nvars k N + 1 ≤ 3 * N ^ 3 := by
    have := neqs_le hk
    have : 1 ≤ N ^ 3 := Nat.one_le_pow _ _ (by omega)
    unfold nvars
    omega
  have hT : (degX k t + 1) * (degY k t + 1) * (nvars k N + 1) ≤ 9 * N ^ 6 := by
    calc _ ≤ (N * N) * (3 * N) * (3 * N ^ 3) := by gcongr
      _ = 9 * N ^ 6 := by ring
  have hρ : rf.bound.eval (degY k t + 1) ≤ rf.bound.eval (3 * N) := eval_mono_nat _ hL
  have hRR : k * ((degY k t + 1) * ((degX k t + 1) * (degY k t + 1) + rf.bound.eval (degY k t + 1) +
      childCost (degX k t) (degY k t))) ≤ 171 * N ^ 7 + 3 * N ^ 2 * rf.bound.eval (3 * N) := by
    unfold childCost
    have h57 : N ^ 5 ≤ N ^ 7 := Nat.pow_le_pow_right (by omega) (by norm_num)
    calc _ ≤ N * ((3 * N) * ((N * N) * (3 * N) + rf.bound.eval (3 * N) +
          (N * N) * (3 * N) * ((3 * N) * (6 * N)))) := by gcongr
      _ = 9 * N ^ 5 + 162 * N ^ 7 + 3 * N ^ 2 * rf.bound.eval (3 * N) := by ring
      _ ≤ _ := by omega
  omega

theorem keepCost_le {k N m s : ℕ} (hk : k + 1 < N) (hm : m < s) : keepCost k N m ≤ 4 * N ^ 2 * s := by
  unfold keepCost
  have h1 : (m + 1) * (2 * k + 1) + 1 ≤ 2 * N * s + 1 := by
    have : (m + 1) * (2 * k + 1) ≤ s * (2 * N) := Nat.mul_le_mul (by omega) (by omega)
    linarith
  have h2 : N * ((m + 1) * (2 * k + 1) + 1) ≤ N * (2 * N * s + 1) := Nat.mul_le_mul_left N h1
  have hN : 1 ≤ N := by omega
  have hs : 1 ≤ s := by omega
  have h3 : N ≤ N ^ 2 * s := by nlinarith
  have h4 : 1 ≤ N ^ 2 * s := by nlinarith
  nlinarith

/-- **The pruning's count**: at most `12·N^4·s²` in the regime, on distinct points. -/
theorem pruneCost_le {F : Type} [Field F] [DecidableEq F] {s : ℕ} (rf : RootFinder F) {k t : ℕ}
    {cs : List (F × (Fin s → F))} (h : Reg k cs.length t) (hnd : (cs.map Prod.fst).Nodup) :
    pruneCost rf k t cs ≤ 12 * cs.length ^ 4 * s ^ 2 := by
  unfold pruneCost
  have hk := h.k_lt
  have hterm : ∀ m ∈ Finset.range s, (kept rf k t cs m).length * (cands rf k t cs m).length *
      keepCost k cs.length m ≤ 12 * cs.length ^ 4 * s := by
    intro m hm
    rw [Finset.mem_range] at hm
    have h1 := length_kept_le rf h hnd m
    have h2 : (cands rf k t cs m).length ≤ 3 * cs.length := by
      rw [cands, dite_eq_left hm]
      exact (length_rr_le rf (degY_le_degX k t) _ k).trans (degY_succ_le h)
    have h3 := keepCost_le hk hm
    calc _ ≤ cs.length * (3 * cs.length) * (4 * cs.length ^ 2 * s) := by gcongr
      _ = 12 * cs.length ^ 4 * s := by ring
  calc _ ≤ (Finset.range s).card • (12 * cs.length ^ 4 * s) := Finset.sum_le_card_nsmul _ _ _ hterm
    _ = 12 * cs.length ^ 4 * s ^ 2 := by rw [Finset.card_range, smul_eq_mul]; ring

/-- **The complexity polynomial**, from the root finder's bound `ρ`, the reader's `β` and the cost `dc` of a point:
`20·X^9 + 171·X^7 + 9·X^6 + 12·X^4 + X^2 + dc·X + 1 + 3·X^2·ρ(3·X) + (X + 1)·β(X)`. -/
noncomputable def gsBound (ρ β : Polynomial ℕ) (dc : ℕ) : Polynomial ℕ :=
  C 20 * X ^ 9 + C 171 * X ^ 7 + C 9 * X ^ 6 + C 12 * X ^ 4 + X ^ 2 + C dc * X + 1 +
    C 3 * X ^ 2 * ρ.comp (C 3 * X) + (X + 1) * β

theorem eval_gsBound (ρ β : Polynomial ℕ) (dc Q : ℕ) :
    (gsBound ρ β dc).eval Q = 20 * Q ^ 9 + 171 * Q ^ 7 + 9 * Q ^ 6 + 12 * Q ^ 4 + Q ^ 2 + dc * Q + 1 +
      3 * Q ^ 2 * ρ.eval (3 * Q) + (Q + 1) * β.eval Q := by
  simp [gsBound, eval_comp]

/-- **`gsCost` is at most `gsBound` at `N·s·w`**, for every input. -/
theorem gsCost_le {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (rf : RootFinder F)
    (rd : Reader C) (agree : ℝ) (dc w : ℕ) (hw : 1 ≤ w) (os : List (ℕ × (Fin s → F))) :
    gsCost C rf rd agree dc os ≤ (gsBound rf.bound rd.bound dc).eval (os.length * s * w) := by
  rw [eval_gsBound]
  have hβ0 : ∀ Q : ℕ, rd.bound.eval Q ≤ (Q + 1) * rd.bound.eval Q := fun Q => Nat.le_mul_of_pos_left _ (by omega)
  unfold gsCost
  by_cases hs : s = 0
  · rw [ite_eq_left hs]
    subst hs
    have h1 := rd.cost_le
    rw [zero_mul] at h1
    have h2 := eval_mono_nat rd.bound (Nat.zero_le (os.length * 0 * w))
    have h3 := hβ0 (os.length * 0 * w)
    omega
  rw [ite_eq_right hs]
  generalize hQ : os.length * s * w = Q
  have hs1 : 1 ≤ s := by omega
  have hsN : os.length * s ≤ Q := by rw [← hQ]; exact Nat.le_mul_of_pos_right _ hw
  have hNQ : os.length ≤ Q := le_trans (Nat.le_mul_of_pos_right _ hs1) hsN
  have hsQ0 : 0 ≤ 3 * Q ^ 2 * rf.bound.eval (3 * Q) := Nat.zero_le _
  have hdc : os.length * dc ≤ dc * Q := by rw [mul_comm]; exact Nat.mul_le_mul_left dc hNQ
  have hNN : os.length * os.length ≤ Q ^ 2 := by rw [sq]; exact Nat.mul_le_mul hNQ hNQ
  split_ifs with hr
  swap
  · have := hβ0 Q
    omega
  obtain ⟨hreg, hnd⟩ := hr
  set t := IRS.threshold agree os
  have hlen : (cols C os).length = os.length := length_cols C os
  have hreg' : Reg C.k (cols C os).length t := by rwa [hlen]
  have hrow : ∑ r : Fin s, rowCost rf C.k t (cols C os) r ≤
      s * (20 * os.length ^ 9 + 171 * os.length ^ 7 + 9 * os.length ^ 6 +
        3 * os.length ^ 2 * rf.bound.eval (3 * os.length)) := by
    have := Finset.sum_le_card_nsmul Finset.univ (rowCost rf C.k t (cols C os)) _
      fun r _ => rowCost_le rf hreg' r
    rwa [Finset.card_univ, Fintype.card_fin, smul_eq_mul, hlen] at this
  have hpr := pruneCost_le rf hreg' hnd
  rw [hlen] at hpr
  have hkept := length_kept_le rf hreg' hnd s
  rw [hlen] at hkept
  have hrd : rd.cost ≤ rd.bound.eval Q := by
    refine rd.cost_le.trans (eval_mono_nat _ ?_)
    have := hreg.k_lt
    calc s * C.k ≤ os.length * s := by rw [mul_comm]; exact Nat.mul_le_mul_right s (by omega)
      _ ≤ Q := hsN
  generalize os.length = N at hsN hNQ hdc hNN hrow hpr hkept
  have hpow : ∀ e : ℕ, s * N ^ (e + 1) ≤ Q ^ (e + 1) := fun e => by
    calc s * N ^ (e + 1) = (N * s) * N ^ e := by ring
      _ ≤ Q * Q ^ e := Nat.mul_le_mul hsN (Nat.pow_le_pow_left hNQ e)
      _ = Q ^ (e + 1) := by ring
  have h9 : s * N ^ 9 ≤ Q ^ 9 := hpow 8
  have h7 : s * N ^ 7 ≤ Q ^ 7 := hpow 6
  have h6 : s * N ^ 6 ≤ Q ^ 6 := hpow 5
  have hρ : s * (3 * N ^ 2 * rf.bound.eval (3 * N)) ≤ 3 * Q ^ 2 * rf.bound.eval (3 * Q) := by
    calc s * (3 * N ^ 2 * rf.bound.eval (3 * N)) = 3 * (s * N ^ 2) * rf.bound.eval (3 * N) := by ring
      _ ≤ 3 * Q ^ 2 * rf.bound.eval (3 * Q) :=
        Nat.mul_le_mul (Nat.mul_le_mul_left 3 (hpow 1)) (eval_mono_nat _ (by omega))
  have h4 : 12 * N ^ 4 * s ^ 2 ≤ 12 * Q ^ 4 := by
    calc 12 * N ^ 4 * s ^ 2 = 12 * ((N * s) ^ 2 * N ^ 2) := by ring
      _ ≤ 12 * (Q ^ 2 * Q ^ 2) := Nat.mul_le_mul_left 12 (Nat.mul_le_mul (Nat.pow_le_pow_left hsN 2)
          (Nat.pow_le_pow_left hNQ 2))
      _ = 12 * Q ^ 4 := by ring
  have hrd' : (kept rf C.k t (cols C os) s).length * rd.cost ≤ (Q + 1) * rd.bound.eval Q :=
    Nat.mul_le_mul (by omega) hrd
  have hsplit : s * (20 * N ^ 9 + 171 * N ^ 7 + 9 * N ^ 6 + 3 * N ^ 2 * rf.bound.eval (3 * N)) =
      20 * (s * N ^ 9) + 171 * (s * N ^ 7) + 9 * (s * N ^ 6) +
        s * (3 * N ^ 2 * rf.bound.eval (3 * N)) := by ring
  omega

/-- **The counted decoder's complexity**: its count is at most `gsBound` at the input size `N·s·w`. -/
theorem gsDecode_complexity {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (rf : RootFinder F)
    (rd : Reader C) (agree : ℝ) (dc w : ℕ) (hw : 1 ≤ w) :
    DecoderComplexity (gsDecode C rf rd agree) (gsCounted C rf rd agree dc)
      (fun os => os.length * s * w) (gsBound rf.bound rd.bound dc) :=
  fun os => ⟨rfl, gsCost_le C rf rd agree dc w hw os⟩

end Claims.GS
