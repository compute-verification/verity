import Proofs.Properties.PoubCheck.Sort
import Proofs.Properties.Nci.Desc

/-!
# `Desc.novelWidth` is the width of its segments' gates

A piece of a segment names gates with their widths (`pieceGates`, `segGates`): a parameter's leaves are none, a row of a
piece of gates is its elements, and a segment of copies is each copy's pieces. `NovelWidthExact` says `Desc.novelWidth`
is the total width of the distinct gates its segments name, wherever no gate is named with two widths. It holds
(`novelWidthExact`), proved layer by layer:

* `distinctWidth_eq`: `coverStep`'s sweep over intervals sorted by start counts each gate once (`List.mergeSort` sorts,
  as Lean core proves).
* `widthIn_eq`: a component's width is its items' gates' width; a frame alone counts copies times one copy's
  (`Desc.sum_copies`).
* `sweep_eq`: `sweepStep`'s sweep over items sorted by start splits them into components no gate is in two of.
* `frameFold_inv`: frames of one start, period and count merge into one with the union of their gates (`Array.qsort`
  only permutes, `Properties.PoubCheck.qsort_toList_perm`).
* `copiesStep_inv`: a segment of copies gives its copies' intervals or one frame of them, with its gates.
* `tail_eq` and `novelWidth_sum` put them together, given `GateIvsExact` (`novelWidthExact_of`): `Desc.gateIvs` gives a
  piece list's gates as intervals, with their widths.
* `gateIvsExact_of`: `gateIvs` takes each width's pieces to `ivsOf`, so `GateIvsExact` holds where `IvsOfExact` does.
* `ivsOfExact_of`: `ivsOf` emits each periodic interval whose runs meet as one interval and sweeps the rest, sorted, merging
  those of one period and count that meet (`merge_spec`, `mergeFold_spec`), so `IvsOfExact` holds where `PisExact` does.
* `pisExact`: each case of `Piece.pis` is its piece's elements, one row's progression from its smallest element (`ap`), a
  grid whose rows (or columns) are runs as one periodic interval (`grid`), else row by row (`one_spec`).
-/

namespace NciCheck.Properties.SecurityProofs.Width

open NciCheck.Desc

/-- The gates of intervals `(lo, hi, w)`: `[lo, hi)` each. -/
def ivGates (L : List (ℕ × ℕ × ℕ)) : Finset ℕ := (L.flatMap fun x => List.range' x.1 (x.2.1 - x.1)).toFinset

theorem mem_ivGates {L : List (ℕ × ℕ × ℕ)} {a : ℕ} : a ∈ ivGates L ↔ ∃ x ∈ L, x.1 ≤ a ∧ a < x.2.1 := by
  simp only [ivGates, List.mem_toFinset, List.mem_flatMap, List.mem_range'_1]
  exact exists_congr fun x => and_congr_right fun _ => by omega

/-- Each interval's gates of the width `W` gives them. -/
def Agree (W : ℕ → ℕ) (L : List (ℕ × ℕ × ℕ)) : Prop := ∀ x ∈ L, ∀ a, x.1 ≤ a → a < x.2.1 → W a = x.2.2

/-- What `coverStep` keeps over the intervals `P` it has read, all starting at or before what comes next. -/
def CoverInv (W : ℕ → ℕ) (P : List (ℕ × ℕ × ℕ)) (st : ℕ × ℕ) : Prop :=
  st.1 = ∑ a ∈ ivGates P, W a ∧ (∀ a ∈ ivGates P, a < st.2) ∧
    ∀ lo, (∀ x ∈ P, x.1 ≤ lo) → ∀ a, lo ≤ a → a < st.2 → a ∈ ivGates P

theorem ivGates_append (P : List (ℕ × ℕ × ℕ)) (x : ℕ × ℕ × ℕ) :
    ivGates (P ++ [x]) = ivGates P ∪ Finset.Ico x.1 x.2.1 := by
  ext a
  simp only [Finset.mem_union, Finset.mem_Ico, mem_ivGates, List.mem_append, List.mem_singleton]
  constructor
  · rintro ⟨y, hy | rfl, h⟩
    · exact Or.inl ⟨y, hy, h⟩
    · exact Or.inr h
  · rintro (⟨y, hy, h⟩ | h)
    · exact ⟨y, Or.inl hy, h⟩
    · exact ⟨x, Or.inr rfl, h⟩

theorem coverStep_inv (W : ℕ → ℕ) (P : List (ℕ × ℕ × ℕ)) (st : ℕ × ℕ) (x : ℕ × ℕ × ℕ)
    (hP : ∀ y ∈ P, y.1 ≤ x.1) (hx : ∀ a, x.1 ≤ a → a < x.2.1 → W a = x.2.2) (h : CoverInv W P st) :
    CoverInv W (P ++ [x]) (coverStep st x) := by
  obtain ⟨h1, h2, h3⟩ := h
  have hsub : ∀ a, x.1 ≤ a → a < st.2 → a ∈ ivGates P := h3 x.1 hP
  unfold coverStep
  split_ifs with hc
  · have he : ivGates (P ++ [x]) = ivGates P := by
      rw [ivGates_append]
      refine Finset.union_eq_left.2 fun a ha => ?_
      rw [Finset.mem_Ico] at ha
      exact hsub a ha.1 (by omega)
    unfold CoverInv; rw [he]
    refine ⟨h1, h2, fun lo hlo a ha1 ha2 => h3 lo (fun y hy => hlo y (List.mem_append_left _ hy)) a ha1 ha2⟩
  · rw [not_le] at hc
    have he : ivGates (P ++ [x]) = ivGates P ∪ Finset.Ico (max x.1 st.2) x.2.1 := by
      rw [ivGates_append]
      ext a
      simp only [Finset.mem_union, Finset.mem_Ico]
      constructor
      · rintro (h | ⟨ha1, ha2⟩)
        · exact Or.inl h
        · by_cases hac : a < st.2
          · exact Or.inl (hsub a ha1 hac)
          · exact Or.inr ⟨by omega, ha2⟩
      · rintro (h | ⟨ha1, ha2⟩)
        · exact Or.inl h
        · exact Or.inr ⟨by omega, ha2⟩
    have hdisj : Disjoint (ivGates P) (Finset.Ico (max x.1 st.2) x.2.1) := by
      rw [Finset.disjoint_left]
      intro a ha ha'
      have := h2 a ha
      rw [Finset.mem_Ico] at ha'
      omega
    refine ⟨?_, ?_, ?_⟩
    · show st.1 + (x.2.1 - max x.1 st.2) * x.2.2 = _
      rw [he, Finset.sum_union hdisj, h1]
      congr 1
      rw [Finset.sum_congr rfl fun a ha => hx a (by rw [Finset.mem_Ico] at ha; omega)
        (by rw [Finset.mem_Ico] at ha; omega)]
      simp
    · intro a ha
      rw [he, Finset.mem_union, Finset.mem_Ico] at ha
      show a < x.2.1
      rcases ha with ha | ha
      · have := h2 a ha; omega
      · exact ha.2
    · intro lo hlo a ha1 ha2
      have hl : x.1 ≤ lo := hlo x (List.mem_append_right _ (List.mem_singleton_self x))
      rw [ivGates_append, Finset.mem_union, Finset.mem_Ico]
      exact Or.inr ⟨by omega, ha2⟩

theorem coverFold_inv (W : ℕ → ℕ) : ∀ (L P : List (ℕ × ℕ × ℕ)) (st : ℕ × ℕ),
    L.Pairwise (fun x y => x.1 ≤ y.1) → (∀ x ∈ P, ∀ y ∈ L, x.1 ≤ y.1) → Agree W L → CoverInv W P st →
      CoverInv W (P ++ L) (L.foldl coverStep st)
  | [], P, st, _, _, _, h => by simpa using h
  | x :: L, P, st, hs, hPL, hW, h => by
    rw [List.foldl_cons, show P ++ x :: L = (P ++ [x]) ++ L by simp]
    obtain ⟨hx, hL⟩ := List.pairwise_cons.1 hs
    refine coverFold_inv W L (P ++ [x]) _ hL (fun y hy z hz => ?_) (fun y hy => hW y (List.mem_cons_of_mem _ hy))
      (coverStep_inv W P st x (fun y hy => hPL y hy x (List.mem_cons_self ..))
        (hW x (List.mem_cons_self ..)) h)
    rcases List.mem_append.1 hy with hy | hy
    · exact hPL y hy z (List.mem_cons_of_mem _ hz)
    · rw [List.mem_singleton] at hy; subst hy; exact hx z hz

/-- **`distinctWidth` is the width of the union.** Of intervals whose gates have the widths `W` gives them, the total
width of the distinct gates. -/
theorem distinctWidth_eq (ivs : Array (ℕ × ℕ × ℕ)) (W : ℕ → ℕ) (hW : Agree W ivs.toList) :
    distinctWidth ivs = ∑ a ∈ ivGates ivs.toList, W a := by
  unfold distinctWidth
  set S := ivs.toList.mergeSort fun x y => decide (x.1 ≤ y.1) with hS
  have hperm : S.Perm ivs.toList := List.mergeSort_perm _ _
  have hsort : S.Pairwise (fun x y => x.1 ≤ y.1) := by
    have := List.pairwise_mergeSort (le := fun x y : ℕ × ℕ × ℕ => decide (x.1 ≤ y.1))
      (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
      (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) ivs.toList
    exact this.imp fun h => by simpa using h
  have hg : ivGates S = ivGates ivs.toList := by
    ext a; simp only [mem_ivGates]; exact exists_congr fun x => and_congr_left fun _ => hperm.mem_iff
  have := coverFold_inv W S [] (0, 0) hsort (by simp) (fun x hx => hW x (hperm.mem_iff.1 hx))
    ⟨by simp [ivGates], by simp [ivGates], fun _ _ a _ h => absurd h (Nat.not_lt_zero a)⟩
  rw [List.nil_append] at this
  rw [this.1, hg]


/-! ## The sweep over components -/

/-- The gates of items `(st, en, j)`, item `j`'s being `E j`. -/
def U (E : ℕ → Finset ℕ) (L : List (ℕ × ℕ × ℕ)) : Finset ℕ := L.toFinset.biUnion fun it => E it.2.2

theorem U_append (E : ℕ → Finset ℕ) (A B : List (ℕ × ℕ × ℕ)) : U E (A ++ B) = U E A ∪ U E B := by
  simp [U, List.toFinset_append, Finset.union_biUnion]

theorem mem_U {E : ℕ → Finset ℕ} {L : List (ℕ × ℕ × ℕ)} {a : ℕ} : a ∈ U E L ↔ ∃ it ∈ L, a ∈ E it.2.2 := by
  simp [U]

theorem U_map (E : ℕ → Finset ℕ) (C : List (ℕ × ℕ × ℕ)) :
    (C.map (·.2.2)).toFinset.biUnion E = U E C := by
  ext a
  simp only [U, Finset.mem_biUnion, List.mem_toFinset, List.mem_map]
  constructor
  · rintro ⟨j, ⟨it, hit, rfl⟩, ha⟩; exact ⟨it, hit, ha⟩
  · rintro ⟨it, hit, ha⟩; exact ⟨_, ⟨it, hit, rfl⟩, ha⟩

/-- What `sweepStep` keeps: the closed components `D` counted, the current one `C` its indices, every gate of `D` before
every start of `C`, and `C` reaching no further than the bound. -/
def SweepInv (W : ℕ → ℕ) (E : ℕ → Finset ℕ) (D C : List (ℕ × ℕ × ℕ)) (acc : ℕ × Array ℕ × ℕ) : Prop :=
  acc.1 = ∑ a ∈ U E D, W a ∧ acc.2.1.toList = C.map (·.2.2) ∧ (C = [] → D = []) ∧
    (∀ a ∈ U E D, ∀ c ∈ C, a < c.1) ∧ ∀ c ∈ C, c.2.1 ≤ acc.2.2

theorem sweepStep_inv (W : ℕ → ℕ) (E : ℕ → Finset ℕ) (width : Array ℕ → ℕ)
    (Q : ℕ × ℕ × ℕ → Prop) (hQ : ∀ c, Q c → ∀ a ∈ E c.2.2, c.1 ≤ a ∧ a < c.2.1)
    (hwidth : ∀ C : List (ℕ × ℕ × ℕ), (∀ c ∈ C, Q c) → width (C.map (·.2.2)).toArray = ∑ a ∈ U E C, W a)
    (D C : List (ℕ × ℕ × ℕ)) (acc : ℕ × Array ℕ × ℕ) (it : ℕ × ℕ × ℕ)
    (hb : ∀ c ∈ D ++ C ++ [it], Q c) (hs : ∀ c ∈ C, c.1 ≤ it.1)
    (h : SweepInv W E D C acc) :
    ∃ D' C', D' ++ C' = D ++ C ++ [it] ∧ SweepInv W E D' C' (sweepStep width acc it) := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  have hcomp : acc.2.1 = (C.map (·.2.2)).toArray := by rw [← h2]
  unfold sweepStep
  split_ifs with hc
  · simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq] at hc
    obtain ⟨hne, hge⟩ := hc
    have hCne : C ≠ [] := by rintro rfl; rw [hcomp] at hne; simp at hne
    obtain ⟨c0, hc0⟩ := List.exists_mem_of_ne_nil C hCne
    have hbC' : ∀ c ∈ C, Q c := fun c hc => hb c (by simp [hc])
    have hbC : ∀ c ∈ C, ∀ a ∈ E c.2.2, c.1 ≤ a ∧ a < c.2.1 := fun c hc => hQ c (hbC' c hc)
    refine ⟨D ++ C, [it], by simp, ?_, ?_, ?_, ?_, ?_⟩
    · show acc.1 + width acc.2.1 = _
      have hdisj : Disjoint (U E D) (U E C) := by
        rw [Finset.disjoint_left]
        intro a ha ha'
        obtain ⟨c, hc, hac⟩ := mem_U.1 ha'
        have := h4 a ha c hc
        have := (hbC c hc a hac).1
        omega
      rw [U_append, Finset.sum_union hdisj, h1, hcomp, hwidth C hbC']
    · simp
    · simp
    · intro a ha c hc
      rw [List.mem_singleton] at hc; subst hc
      rw [U_append, Finset.mem_union] at ha
      rcases ha with ha | ha
      · have := h4 a ha c0 hc0; have := hs c0 hc0; omega
      · obtain ⟨c, hc, hac⟩ := mem_U.1 ha
        have := (hbC c hc a hac).2; have := h5 c hc; omega
    · intro c hc; rw [List.mem_singleton] at hc; subst hc; exact le_refl _
  · refine ⟨D, C ++ [it], by simp, h1, by simp [h2], fun h => absurd h (by simp), ?_, ?_⟩
    · intro a ha c hc
      rcases List.mem_append.1 hc with hc | hc
      · exact h4 a ha c hc
      · rw [List.mem_singleton] at hc; subst hc
        by_cases hC : C = []
        · rw [h3 hC] at ha; simp [U] at ha
        · obtain ⟨c0, hc0⟩ := List.exists_mem_of_ne_nil C hC
          have := h4 a ha c0 hc0; have := hs c0 hc0; omega
    · intro c hc
      rcases List.mem_append.1 hc with hc | hc
      · exact le_trans (h5 c hc) (le_max_left _ _)
      · rw [List.mem_singleton] at hc; subst hc; exact le_max_right _ _

theorem sweepFold_inv (W : ℕ → ℕ) (E : ℕ → Finset ℕ) (width : Array ℕ → ℕ)
    (Q : ℕ × ℕ × ℕ → Prop) (hQ : ∀ c, Q c → ∀ a ∈ E c.2.2, c.1 ≤ a ∧ a < c.2.1)
    (hwidth : ∀ C : List (ℕ × ℕ × ℕ), (∀ c ∈ C, Q c) → width (C.map (·.2.2)).toArray = ∑ a ∈ U E C, W a) :
    ∀ (L D C : List (ℕ × ℕ × ℕ)) (acc : ℕ × Array ℕ × ℕ),
      L.Pairwise (fun x y => x.1 ≤ y.1) → (∀ c ∈ D ++ C, ∀ y ∈ L, c.1 ≤ y.1) →
      (∀ c ∈ D ++ C ++ L, Q c) → SweepInv W E D C acc →
      ∃ D' C', D' ++ C' = D ++ C ++ L ∧ SweepInv W E D' C' (L.foldl (sweepStep width) acc)
  | [], D, C, acc, _, _, _, h => ⟨D, C, by simp, h⟩
  | it :: L, D, C, acc, hs, hP, hb, h => by
    obtain ⟨hit, hL⟩ := List.pairwise_cons.1 hs
    obtain ⟨D', C', he, h'⟩ := sweepStep_inv W E width Q hQ hwidth D C acc it
      (fun c hc => hb c (by simp only [List.mem_append, List.mem_cons] at hc ⊢; tauto))
      (fun c hc => hP c (List.mem_append_right _ hc) it (List.mem_cons_self ..)) h
    have := sweepFold_inv W E width Q hQ hwidth L D' C' _ hL
      (fun c hc y hy => by
        rw [he] at hc
        rcases List.mem_append.1 hc with hc | hc
        · exact hP c hc y (List.mem_cons_of_mem _ hy)
        · rw [List.mem_singleton] at hc; subst hc; exact hit y hy)
      (fun c hc => hb c (by rw [he] at hc; simpa using hc)) h'
    obtain ⟨D'', C'', he', h''⟩ := this
    exact ⟨D'', C'', by rw [he', he]; simp, h''⟩

/-- **The sweep counts each component once.** Items sorted by start, each item's gates within its span: the sweep's total
(the last component counted after it) is the width of all their gates. -/
theorem sweep_eq (W : ℕ → ℕ) (E : ℕ → Finset ℕ) (width : Array ℕ → ℕ)
    (Q : ℕ × ℕ × ℕ → Prop) (hQ : ∀ c, Q c → ∀ a ∈ E c.2.2, c.1 ≤ a ∧ a < c.2.1)
    (hwidth : ∀ C : List (ℕ × ℕ × ℕ), (∀ c ∈ C, Q c) → width (C.map (·.2.2)).toArray = ∑ a ∈ U E C, W a)
    (L : List (ℕ × ℕ × ℕ)) (hs : L.Pairwise (fun x y => x.1 ≤ y.1))
    (hb : ∀ c ∈ L, Q c) :
    let fin := L.foldl (sweepStep width) (0, #[], 0)
    (if fin.2.1.isEmpty then fin.1 else fin.1 + width fin.2.1) = ∑ a ∈ U E L, W a := by
  intro fin
  obtain ⟨D, C, he, h1, h2, h3, h4, -⟩ := sweepFold_inv W E width Q hQ hwidth L [] [] (0, #[], 0) hs (by simp)
    (by simpa using hb) ⟨by simp [U], rfl, fun _ => rfl, by simp, by simp⟩
  simp only [List.nil_append] at he
  have hcomp : fin.2.1 = (C.map (·.2.2)).toArray := by rw [← h2]
  have hbC' : ∀ c ∈ C, Q c := fun c hc => hb c (by rw [← he]; simp [hc])
  have hbC : ∀ c ∈ C, ∀ a ∈ E c.2.2, c.1 ≤ a ∧ a < c.2.1 := fun c hc => hQ c (hbC' c hc)
  rw [← he, U_append]
  split_ifs with hc
  · have hC : C = [] := by
      rw [hcomp] at hc; simpa using hc
    have hD := h3 hC
    subst hC; subst hD; simpa [U] using h1
  · have hdisj : Disjoint (U E D) (U E C) := by
      rw [Finset.disjoint_left]
      intro a ha ha'
      obtain ⟨c, hc, hac⟩ := mem_U.1 ha'
      have := h4 a ha c hc
      have := (hbC c hc a hac).1
      omega
    rw [Finset.sum_union hdisj, ← h1, hcomp, hwidth C hbC']


/-! ## Frames -/

/-- A frame's intervals: `expandIn`'s for a frame item. -/
def frameIvs (x : Cl) : Array (ℕ × ℕ × ℕ) :=
  (Array.range x.cnt).flatMap fun m => x.res.map fun y => (x.lo + m * x.period + y.1, x.lo + m * x.period + y.2.1, y.2.2)

theorem expandIn_lt {ivs : Array (ℕ × ℕ × ℕ)} {frames : Array Cl} {j : ℕ} (h : j < ivs.size) :
    expandIn ivs frames j = #[ivs[j]!] := by
  simp [expandIn, h]

theorem expandIn_ge {ivs : Array (ℕ × ℕ × ℕ)} {frames : Array Cl} {j : ℕ} (h : ¬ j < ivs.size) :
    expandIn ivs frames j = frameIvs frames[j - ivs.size]! := by
  simp [expandIn, h, frameIvs]

theorem mem_frameIvs {x : Cl} {z : ℕ × ℕ × ℕ} : z ∈ (frameIvs x).toList ↔
    ∃ m < x.cnt, ∃ y ∈ x.res.toList, z = (x.lo + m * x.period + y.1, x.lo + m * x.period + y.2.1, y.2.2) := by
  simp only [frameIvs, Array.toList_flatMap, Array.toList_range, Array.toList_map, List.mem_flatMap, List.mem_range,
    List.mem_map]
  constructor
  · rintro ⟨m, hm, y, hy, rfl⟩; exact ⟨m, hm, y, hy, rfl⟩
  · rintro ⟨m, hm, y, hy, rfl⟩; exact ⟨m, hm, y, hy, rfl⟩

theorem mem_frame {x : Cl} {a : ℕ} : a ∈ ivGates (frameIvs x).toList ↔
    ∃ m < x.cnt, ∃ b ∈ ivGates x.res.toList, a = x.lo + m * x.period + b := by
  rw [mem_ivGates]
  constructor
  · rintro ⟨z, hz, h1, h2⟩
    obtain ⟨m, hm, y, hy, rfl⟩ := mem_frameIvs.1 hz
    exact ⟨m, hm, a - (x.lo + m * x.period), mem_ivGates.2 ⟨y, hy, by dsimp at h1 h2; omega⟩, by dsimp at h1; omega⟩
  · rintro ⟨m, hm, b, hb, rfl⟩
    obtain ⟨y, hy, h1, h2⟩ := mem_ivGates.1 hb
    exact ⟨_, mem_frameIvs.2 ⟨m, hm, y, hy, rfl⟩, by dsimp; omega, by dsimp; omega⟩

theorem agree_frame {W : ℕ → ℕ} {x : Cl} : Agree W (frameIvs x).toList ↔
    ∀ m < x.cnt, ∀ y ∈ x.res.toList, ∀ b, y.1 ≤ b → b < y.2.1 → W (x.lo + m * x.period + b) = y.2.2 := by
  constructor
  · intro h m hm y hy b h1 h2
    exact h _ (mem_frameIvs.2 ⟨m, hm, y, hy, rfl⟩) _ (by dsimp; omega) (by dsimp; omega)
  · intro h z hz a h1 h2
    obtain ⟨m, hm, y, hy, rfl⟩ := mem_frameIvs.1 hz
    dsimp at h1 h2 ⊢
    have := h m hm y hy (a - (x.lo + m * x.period)) (by omega) (by omega)
    rwa [show x.lo + m * x.period + (a - (x.lo + m * x.period)) = a by omega] at this

/-- A frame as `novelWidth` counts it: its intervals within its span, the span within its period, their widths `W`'s. -/
def FrameOk (W : ℕ → ℕ) (x : Cl) : Prop :=
  (∀ y ∈ x.res.toList, y.2.1 ≤ x.span) ∧ x.span ≤ x.period ∧ Agree W (frameIvs x).toList

theorem mem_ivGates_flatMap {β : Type} (L : List β) (f : β → List (ℕ × ℕ × ℕ)) (a : ℕ) :
    a ∈ ivGates (L.flatMap f) ↔ ∃ j ∈ L, a ∈ ivGates (f j) := by
  simp only [mem_ivGates, List.mem_flatMap]
  constructor
  · rintro ⟨z, ⟨j, hj, hz⟩, h⟩; exact ⟨j, hj, z, hz, h⟩
  · rintro ⟨j, hj, z, hz, h⟩; exact ⟨z, ⟨j, hj, hz⟩, h⟩

/-- Item `j`'s gates. -/
def itemGates (ivs : Array (ℕ × ℕ × ℕ)) (frames : Array Cl) (j : ℕ) : Finset ℕ := ivGates (expandIn ivs frames j).toList

theorem agree_expandIn {W : ℕ → ℕ} {ivs : Array (ℕ × ℕ × ℕ)} {frames : Array Cl} (hI : Agree W ivs.toList)
    (hF : ∀ x ∈ frames.toList, FrameOk W x) {j : ℕ} (hj : j < ivs.size + frames.size) :
    Agree W (expandIn ivs frames j).toList := by
  by_cases h : j < ivs.size
  · rw [expandIn_lt h]
    intro z hz
    have hz' : z = ivs[j]! := by simpa using hz
    rw [hz']
    exact hI _ (by rw [getElem!_pos ivs j h]; exact Array.getElem_mem_toList h)
  · rw [expandIn_ge h]
    exact (hF _ (by rw [getElem!_pos frames _ (by omega)]; exact Array.getElem_mem_toList _)).2.2

/-- **A component's width.** Of valid items, `widthIn` is the width of their gates: a frame alone by `sum_copies`, else by
`distinctWidth_eq`. -/
theorem widthIn_eq (W : ℕ → ℕ) (ivs : Array (ℕ × ℕ × ℕ)) (frames : Array Cl) (comp : Array ℕ)
    (hI : Agree W ivs.toList) (hF : ∀ x ∈ frames.toList, FrameOk W x)
    (hc : ∀ j ∈ comp.toList, j < ivs.size + frames.size) :
    widthIn ivs frames comp = ∑ a ∈ comp.toList.toFinset.biUnion (itemGates ivs frames), W a := by
  unfold widthIn
  split_ifs with h
  · simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    obtain ⟨h1, h2⟩ := h
    have hl : comp.toList = [comp[0]!] := by
      rcases comp with ⟨_ | ⟨j, _ | _⟩⟩ <;> simp_all
    set j := comp[0]! with hj
    have hjv : j < ivs.size + frames.size := hc j (by rw [hl]; simp)
    rw [hl]
    simp only [List.toFinset_cons, List.toFinset_nil, insert_empty_eq, Finset.singleton_biUnion]
    have hx : frames[j - ivs.size]! ∈ frames.toList := by
      rw [getElem!_pos frames _ (by omega)]; exact Array.getElem_mem_toList _
    set x := frames[j - ivs.size]! with hxdef
    obtain ⟨hres, hsp, hag⟩ := hF x hx
    have hE : itemGates ivs frames j = (Finset.range x.cnt).biUnion fun m => (ivGates x.res.toList).image (x.lo + m * x.period + ·) := by
      ext a
      rw [itemGates, expandIn_ge (by omega), ← hxdef, mem_frame]
      simp only [Finset.mem_biUnion, Finset.mem_range, Finset.mem_image]
      constructor
      · rintro ⟨m, hm, b, hb, rfl⟩; exact ⟨m, hm, b, hb, rfl⟩
      · rintro ⟨m, hm, b, hb, rfl⟩; exact ⟨m, hm, b, hb, rfl⟩
    have hag' := agree_frame.1 hag
    rw [hE, NciCheck.Properties.SecurityProofs.Desc.sum_copies (ivGates x.res.toList) x.lo x.period x.span x.cnt
      (fun b hb => by obtain ⟨y, hy, -, h2⟩ := mem_ivGates.1 hb; have := hres y hy; omega) hsp W
      (fun m hm b hb => by
        obtain ⟨y, hy, h1, h2⟩ := mem_ivGates.1 hb
        rw [hag' m hm y hy b h1 h2, ← hag' 0 (by omega) y hy b h1 h2]; simp)]
    rcases Nat.eq_zero_or_pos x.cnt with h0 | hpos
    · rw [h0]; simp
    · congr 1
      refine distinctWidth_eq x.res (fun b => W (x.lo + b)) fun y hy b h1 h2 => ?_
      have := hag' 0 hpos y hy b h1 h2; simpa using this
  · rw [distinctWidth_eq _ W]
    · congr 1
      ext a
      rw [Array.toList_flatMap, mem_ivGates_flatMap, Finset.mem_biUnion]
      simp [itemGates]
    · intro z hz
      rw [Array.toList_flatMap, List.mem_flatMap] at hz
      obtain ⟨j, hj, hz⟩ := hz
      exact agree_expandIn hI hF (hc j hj) z hz


theorem ivGates_append' (A B : List (ℕ × ℕ × ℕ)) : ivGates (A ++ B) = ivGates A ∪ ivGates B := by
  ext a; simp only [Finset.mem_union, mem_ivGates, List.mem_append]
  constructor
  · rintro ⟨z, hz | hz, h⟩
    · exact Or.inl ⟨z, hz, h⟩
    · exact Or.inr ⟨z, hz, h⟩
  · rintro (⟨z, hz, h⟩ | ⟨z, hz, h⟩)
    · exact ⟨z, Or.inl hz, h⟩
    · exact ⟨z, Or.inr hz, h⟩

/-- A frame's gates. -/
def FG (x : Cl) : Finset ℕ := ivGates (frameIvs x).toList

theorem frameStep_inv (W : ℕ → ℕ) (frames : Array Cl) (x : Cl) (hF : ∀ y ∈ frames.toList, FrameOk W y)
    (hx : FrameOk W x) :
    (∀ y ∈ (frameStep frames x).toList, FrameOk W y) ∧
      ∀ a, (∃ y ∈ (frameStep frames x).toList, a ∈ FG y) ↔ (∃ y ∈ frames.toList, a ∈ FG y) ∨ a ∈ FG x := by
  have hpush : (∀ y ∈ (frames.push x).toList, FrameOk W y) ∧
      ∀ a, (∃ y ∈ (frames.push x).toList, a ∈ FG y) ↔ (∃ y ∈ frames.toList, a ∈ FG y) ∨ a ∈ FG x := by
    refine ⟨fun y hy => ?_, fun a => ?_⟩
    · rw [Array.toList_push, List.mem_append, List.mem_singleton] at hy
      rcases hy with hy | rfl
      · exact hF y hy
      · exact hx
    · simp only [Array.toList_push, List.mem_append, List.mem_singleton]
      constructor
      · rintro ⟨y, hy | rfl, h⟩
        · exact Or.inl ⟨y, hy, h⟩
        · exact Or.inr h
      · rintro (⟨y, hy, h⟩ | h)
        · exact ⟨y, Or.inl hy, h⟩
        · exact ⟨x, Or.inr rfl, h⟩
  unfold frameStep
  split
  · rename_i y hy
    obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hy
    split_ifs with h
    · simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨hlo, hp⟩, hc⟩ := h
      set m : Cl := { y with span := max y.span x.span, res := y.res ++ x.res } with hm
      have hyF := hF y (by simp)
      have hys : ∀ z ∈ ys.toList, FrameOk W z := fun z hz => hF z (by simp [hz])
      obtain ⟨hyr, hys', hya⟩ := hyF
      obtain ⟨hxr, hxs, hxa⟩ := hx
      have hFm : FG m = FG y ∪ FG x := by
        ext a
        simp only [FG, Finset.mem_union, mem_frame, hm, Array.toList_append, ivGates_append', Finset.mem_union]
        rw [← hlo, ← hp, ← hc]
        constructor
        · rintro ⟨k, hk, b, hb | hb, rfl⟩
          · exact Or.inl ⟨k, hk, b, hb, rfl⟩
          · exact Or.inr ⟨k, hk, b, hb, rfl⟩
        · rintro (⟨k, hk, b, hb, rfl⟩ | ⟨k, hk, b, hb, rfl⟩)
          · exact ⟨k, hk, b, Or.inl hb, rfl⟩
          · exact ⟨k, hk, b, Or.inr hb, rfl⟩
      have hmF : FrameOk W m := by
        refine ⟨fun z hz => ?_, ?_, ?_⟩
        · simp only [hm, Array.toList_append, List.mem_append] at hz
          rcases hz with hz | hz
          · exact le_trans (hyr z hz) (le_max_left _ _)
          · exact le_trans (hxr z hz) (le_max_right _ _)
        · show max y.span x.span ≤ y.period
          rw [hp]; exact max_le (hp ▸ hys') hxs
        · rw [agree_frame]
          intro k hk z hz b h1 h2
          simp only [hm, Array.toList_append, List.mem_append] at hz
          rcases hz with hz | hz
          · exact agree_frame.1 hya k hk z hz b h1 h2
          · have := agree_frame.1 hxa k (by rw [← hc]; exact hk) z hz b h1 h2
            rwa [← hlo, ← hp] at this
      rw [Array.pop_push]
      refine ⟨fun z hz => ?_, fun a => ?_⟩
      · rw [Array.toList_push, List.mem_append, List.mem_singleton] at hz
        rcases hz with hz | rfl
        · exact hys z hz
        · exact hmF
      · simp only [Array.toList_push, List.mem_append, List.mem_singleton]
        constructor
        · rintro ⟨z, hz | rfl, h⟩
          · exact Or.inl ⟨z, Or.inl hz, h⟩
          · rw [hFm, Finset.mem_union] at h
            rcases h with h | h
            · exact Or.inl ⟨y, Or.inr rfl, h⟩
            · exact Or.inr h
        · rintro (⟨z, hz | rfl, h⟩ | h)
          · exact ⟨z, Or.inl hz, h⟩
          · exact ⟨m, Or.inr rfl, by rw [hFm]; exact Finset.mem_union_left _ h⟩
          · exact ⟨m, Or.inr rfl, by rw [hFm]; exact Finset.mem_union_right _ h⟩
    · exact hpush
  · exact hpush

theorem frameFold_inv (W : ℕ → ℕ) : ∀ (L : List Cl) (frames : Array Cl), (∀ y ∈ frames.toList, FrameOk W y) →
    (∀ x ∈ L, FrameOk W x) →
    (∀ y ∈ (L.foldl frameStep frames).toList, FrameOk W y) ∧
      ∀ a, (∃ y ∈ (L.foldl frameStep frames).toList, a ∈ FG y) ↔
        (∃ y ∈ frames.toList, a ∈ FG y) ∨ ∃ x ∈ L, a ∈ FG x
  | [], frames, hF, _ => ⟨hF, fun a => by simp⟩
  | x :: L, frames, hF, hL => by
    obtain ⟨h1, h2⟩ := frameStep_inv W frames x hF (hL x (List.mem_cons_self ..))
    obtain ⟨h3, h4⟩ := frameFold_inv W L _ h1 fun y hy => hL y (List.mem_cons_of_mem _ hy)
    refine ⟨h3, fun a => ?_⟩
    rw [List.foldl_cons, h4, h2]
    simp only [List.mem_cons]
    constructor
    · rintro ((h | h) | ⟨z, hz, h⟩)
      · exact Or.inl h
      · exact Or.inr ⟨x, Or.inl rfl, h⟩
      · exact Or.inr ⟨z, Or.inr hz, h⟩
    · rintro (h | ⟨z, rfl | hz, h⟩)
      · exact Or.inl (Or.inl h)
      · exact Or.inl (Or.inr h)
      · exact Or.inr ⟨z, hz, h⟩


/-- The frames' gates. -/
def FGs : List Cl → Finset ℕ
  | [] => ∅
  | x :: L => FG x ∪ FGs L

theorem mem_FGs {L : List Cl} {a : ℕ} : a ∈ FGs L ↔ ∃ x ∈ L, a ∈ FG x := by
  induction L with
  | nil => simp [FGs]
  | cons x L ih =>
    simp only [FGs, Finset.mem_union, ih, List.mem_cons]
    constructor
    · rintro (h | ⟨y, hy, h⟩)
      · exact ⟨x, Or.inl rfl, h⟩
      · exact ⟨y, Or.inr hy, h⟩
    · rintro ⟨y, rfl | hy, h⟩
      · exact Or.inl h
      · exact Or.inr ⟨y, hy, h⟩

theorem sorted_mergeSort (L : List (ℕ × ℕ × ℕ)) :
    (L.mergeSort fun x y => decide (x.1 ≤ y.1)).Pairwise (fun x y => x.1 ≤ y.1) := by
  have := List.pairwise_mergeSort (le := fun x y : ℕ × ℕ × ℕ => decide (x.1 ≤ y.1))
    (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
    (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) L
  exact this.imp fun h => by simpa using h

/-- **What follows the segments' step.** Intervals and frames whose widths are `W`'s: the frames sorted and merged, the
items swept by start and each component counted by `widthIn` give the width of all their gates. -/
theorem tail_eq (W : ℕ → ℕ) (ivs : Array (ℕ × ℕ × ℕ)) (cls : Array Cl) (lt : Cl → Cl → Bool)
    (hI : Agree W ivs.toList) (hF : ∀ x ∈ cls.toList, FrameOk W x) :
    let frames := (cls.qsort lt).foldl frameStep #[]
    let n := ivs.size
    let items := (ivs.mapIdx fun i x => (x.1, x.2.1, i)) ++
      (frames.mapIdx fun i x => (x.lo, x.lo + (x.cnt - 1) * x.period + x.span, n + i))
    let fin := (items.toList.mergeSort fun x y => x.1 ≤ y.1).foldl (sweepStep (widthIn ivs frames)) (0, #[], 0)
    (if fin.2.1.isEmpty then fin.1 else fin.1 + widthIn ivs frames fin.2.1) =
      ∑ a ∈ ivGates ivs.toList ∪ FGs cls.toList, W a := by
  intro frames n items fin
  have hperm := Properties.PoubCheck.qsort_toList_perm cls lt
  obtain ⟨hFr, hFg⟩ := frameFold_inv W (cls.qsort lt).toList #[] (by simp)
    (fun x hx => hF x (hperm.mem_iff.1 hx))
  rw [Array.foldl_toList] at hFr hFg
  have hitem : ∀ c, c ∈ items.toList ↔ (∃ i, ∃ h : i < ivs.size, c = (ivs[i].1, ivs[i].2.1, i)) ∨
      ∃ i, ∃ h : i < frames.size, c = (frames[i].lo, frames[i].lo + (frames[i].cnt - 1) * frames[i].period +
        frames[i].span, n + i) := by
    intro c
    simp only [items, Array.mem_toList_iff, Array.mem_append, Array.mem_mapIdx]
    constructor
    · rintro (⟨i, h, rfl⟩ | ⟨i, h, rfl⟩)
      · exact Or.inl ⟨i, h, rfl⟩
      · exact Or.inr ⟨i, h, rfl⟩
    · rintro (⟨i, h, rfl⟩ | ⟨i, h, rfl⟩)
      · exact Or.inl ⟨i, h, rfl⟩
      · exact Or.inr ⟨i, h, rfl⟩
  have hfi : ∀ i (h : i < frames.size), itemGates ivs frames (n + i) = FG frames[i] := by
    intro i h
    rw [itemGates, expandIn_ge (by omega), show n + i - ivs.size = i by omega, getElem!_pos frames i h]
    rfl
  have hii : ∀ i (h : i < ivs.size), ∀ a, a ∈ itemGates ivs frames i ↔ ivs[i].1 ≤ a ∧ a < ivs[i].2.1 := by
    intro i h a
    rw [itemGates, expandIn_lt h, getElem!_pos ivs i h, mem_ivGates]
    simp
  have hvalid : ∀ c ∈ items.toList, c.2.2 < ivs.size + frames.size := by
    intro c hc
    rcases (hitem c).1 hc with ⟨i, h, rfl⟩ | ⟨i, h, rfl⟩
    · dsimp; omega
    · dsimp; omega
  have hQ : ∀ c, c ∈ items.toList → ∀ a ∈ itemGates ivs frames c.2.2, c.1 ≤ a ∧ a < c.2.1 := by
    intro c hc a ha
    rcases (hitem c).1 hc with ⟨i, h, rfl⟩ | ⟨i, h, rfl⟩
    · exact (hii i h a).1 ha
    · dsimp at ha ⊢
      rw [hfi i h, FG, mem_frame] at ha
      obtain ⟨m, hm, b, hb, rfl⟩ := ha
      obtain ⟨y, hy, -, h2⟩ := mem_ivGates.1 hb
      have hres := (hFr _ (Array.getElem_mem_toList h)).1 y hy
      have : m * frames[i].period ≤ (frames[i].cnt - 1) * frames[i].period :=
        Nat.mul_le_mul_right _ (by omega)
      omega
  have hS := (List.mergeSort_perm items.toList fun x y => decide (x.1 ≤ y.1))
  have := sweep_eq W (itemGates ivs frames) (widthIn ivs frames) (· ∈ items.toList) hQ
    (fun C hC => by
      rw [widthIn_eq W ivs frames _ hI hFr (fun j hj => by
        simp only [List.mem_map] at hj
        obtain ⟨c, hc, rfl⟩ := hj
        exact hvalid c (hC c hc))]
      exact congrArg (fun S => ∑ a ∈ S, W a) (U_map _ C))
    _ (sorted_mergeSort items.toList) (fun c hc => hS.mem_iff.1 hc)
  refine this.trans (congrArg (fun S => ∑ a ∈ S, W a) ?_)
  ext a
  rw [mem_U, Finset.mem_union, mem_FGs]
  constructor
  · rintro ⟨c, hc, ha⟩
    rcases (hitem c).1 (hS.mem_iff.1 hc) with ⟨i, h, rfl⟩ | ⟨i, h, rfl⟩
    · exact Or.inl (mem_ivGates.2 ⟨ivs[i], Array.getElem_mem_toList h, (hii i h a).1 ha⟩)
    · dsimp at ha
      rw [hfi i h] at ha
      have h' := ((hFg a).1 ⟨_, Array.getElem_mem_toList h, ha⟩).resolve_left (by simp)
      obtain ⟨x, hx, hax⟩ := h'
      exact Or.inr ⟨x, hperm.mem_iff.1 hx, hax⟩
  · rintro (ha | ⟨x, hx, ha⟩)
    · obtain ⟨z, hz, h1, h2⟩ := mem_ivGates.1 ha
      obtain ⟨i, h, rfl⟩ := List.mem_iff_getElem.1 hz
      refine ⟨(ivs[i].1, ivs[i].2.1, i), hS.mem_iff.2 ((hitem _).2 (Or.inl ⟨i, h, rfl⟩)), ?_⟩
      exact (hii i h a).2 ⟨h1, h2⟩
    · obtain ⟨y, hy, hay⟩ := (hFg a).2 (Or.inr ⟨x, hperm.mem_iff.2 hx, ha⟩)
      obtain ⟨i, h, rfl⟩ := List.mem_iff_getElem.1 hy
      refine ⟨_, hS.mem_iff.2 ((hitem _).2 (Or.inr ⟨i, h, rfl⟩)), ?_⟩
      dsimp
      rw [hfi i h]; exact hay


/-! ## Segments -/

/-- A piece's gates with their widths: element `t` of row `r` is gate `base + r * rowStride + t * stride`, with one row
when it has none (as `Desc.Piece.pis` reads it); a parameter's leaves are no gates. -/
def pieceGates (q : Piece) : List (ℤ × ℕ) :=
  match q.tgt with
  | .gate w => (List.range (max q.rows 1)).flatMap fun (r : ℕ) => (List.range q.len).map fun (t : ℕ) =>
      (q.base + (r : ℤ) * q.rowStride + (t : ℤ) * q.stride, w)
  | .ref _ _ => []

def piecesGates (ps : Array Piece) : List (ℤ × ℕ) := ps.toList.flatMap pieceGates

/-- A segment's gates: its pieces', or each copy's when it has more than one (as `Desc.novelWidth` reads it). -/
def segGates (s : Seg) : List (ℤ × ℕ) :=
  if s.reps ≤ 1 then piecesGates s.ps else (List.range s.reps).flatMap fun m => piecesGates (s.copy m)

def segsGates (segs : Array Seg) : List (ℤ × ℕ) := segs.toList.flatMap segGates

/-- **`gateIvs` is a piece list's gates.** Of pieces whose gates are naturals, every gate of an interval `gateIvs`
gives is one of theirs with the interval's width, and each of theirs is in one of the intervals. -/
def GateIvsExact : Prop :=
  ∀ qs : Array Piece, (∀ x ∈ piecesGates qs, 0 ≤ x.1) →
    (∀ iv ∈ (gateIvs qs).toList, ∀ a, iv.1 ≤ a → a < iv.2.1 → ((a : ℤ), iv.2.2) ∈ piecesGates qs) ∧
    ∀ x ∈ piecesGates qs, ∃ iv ∈ (gateIvs qs).toList, iv.1 ≤ x.1.toNat ∧ x.1.toNat < iv.2.1

/-- Gates as naturals. -/
def gs (L : List (ℤ × ℕ)) : Finset ℕ := (L.map fun x => x.1.toNat).toFinset

theorem mem_gs {L : List (ℤ × ℕ)} {a : ℕ} : a ∈ gs L ↔ ∃ x ∈ L, a = x.1.toNat := by
  simp only [gs, List.mem_toFinset, List.mem_map]
  exact exists_congr fun x => and_congr_right fun _ => eq_comm

theorem gate_rep (hG : GateIvsExact) (qs : Array Piece) (hnn : ∀ x ∈ piecesGates qs, 0 ≤ x.1) (W : ℕ → ℕ)
    (hW : ∀ x ∈ piecesGates qs, W x.1.toNat = x.2) :
    Agree W (gateIvs qs).toList ∧ ivGates (gateIvs qs).toList = gs (piecesGates qs) := by
  obtain ⟨h1, h2⟩ := hG qs hnn
  refine ⟨fun iv hiv a ha1 ha2 => by simpa using hW _ (h1 iv hiv a ha1 ha2), ?_⟩
  ext a
  rw [mem_ivGates, mem_gs]
  constructor
  · rintro ⟨iv, hiv, ha1, ha2⟩
    exact ⟨_, h1 iv hiv a ha1 ha2, by simp⟩
  · rintro ⟨x, hx, rfl⟩
    obtain ⟨iv, hiv, h⟩ := h2 x hx
    exact ⟨iv, hiv, h⟩

theorem piecesGates_shift (δ : ℤ) (ps : Array Piece) :
    piecesGates (shiftGates δ ps) = (piecesGates ps).map fun x => (x.1 + δ, x.2) := by
  simp only [piecesGates, shiftGates, Array.toList_map, List.flatMap_map, List.map_flatMap]
  congr 1
  funext q
  rcases hq : q.tgt with p | w
  · simp [pieceGates, hq]
  · simp only [pieceGates, hq, List.map_flatMap, List.map_map]
    congr 1; funext r; congr 1; funext t
    simp only [Function.comp, Prod.mk.injEq, and_true]
    ring

theorem copy_gates (s : Seg) (m : ℕ) : piecesGates (s.copy m) = (piecesGates s.ps).map fun x => (x.1 + m * s.shift, x.2) :=
  piecesGates_shift _ _

theorem FGs_append (A B : List Cl) : FGs (A ++ B) = FGs A ∪ FGs B := by
  ext a; simp only [mem_FGs, Finset.mem_union, List.mem_append]
  constructor
  · rintro ⟨x, hx | hx, h⟩
    · exact Or.inl ⟨x, hx, h⟩
    · exact Or.inr ⟨x, hx, h⟩
  · rintro (⟨x, hx, h⟩ | ⟨x, hx, h⟩)
    · exact ⟨x, Or.inl hx, h⟩
    · exact ⟨x, Or.inr hx, h⟩

theorem foldl_min_le (L : List (ℕ × ℕ × ℕ)) : ∀ init, (L.foldl (fun m x => min m x.1) init ≤ init) ∧
    ∀ z ∈ L, L.foldl (fun m x => min m x.1) init ≤ z.1 := by
  induction L with
  | nil => intro init; simp
  | cons y L ih =>
    intro init
    obtain ⟨h1, h2⟩ := ih (min init y.1)
    refine ⟨le_trans h1 (min_le_left _ _), fun z hz => ?_⟩
    rcases List.mem_cons.1 hz with rfl | hz
    · exact le_trans h1 (min_le_right _ _)
    · exact h2 z hz

theorem le_foldl_max (L : List (ℕ × ℕ × ℕ)) : ∀ init, (init ≤ L.foldl (fun m x => max m x.2.1) init) ∧
    ∀ z ∈ L, z.2.1 ≤ L.foldl (fun m x => max m x.2.1) init := by
  induction L with
  | nil => intro init; simp
  | cons y L ih =>
    intro init
    obtain ⟨h1, h2⟩ := ih (max init y.2.1)
    refine ⟨le_trans (le_max_left _ _) h1, fun z hz => ?_⟩
    rcases List.mem_cons.1 hz with rfl | hz
    · exact le_trans (le_max_right _ _) h1
    · exact h2 z hz

theorem toNat_shift (g : ℤ) (hg : 0 ≤ g) (m p : ℕ) : (g + (m : ℤ) * (p : ℤ)).toNat = g.toNat + m * p := by
  obtain ⟨k, rfl⟩ := Int.eq_ofNat_of_zero_le hg
  rw [show (k : ℤ) + m * p = ((k + m * p : ℕ) : ℤ) by push_cast; ring, Int.toNat_natCast, Int.toNat_natCast]

/-- **One segment's step.** Its intervals and frames have `W`'s widths and their gates are the segment's, when it is one
of copies. -/
theorem copiesStep_inv (hG : GateIvsExact) (W : ℕ → ℕ) (s : Seg) (acc : Array (ℕ × ℕ × ℕ) × Array Cl)
    (hnn : ∀ x ∈ segGates s, 0 ≤ x.1) (hW : ∀ x ∈ segGates s, W x.1.toNat = x.2)
    (hI : Agree W acc.1.toList) (hF : ∀ x ∈ acc.2.toList, FrameOk W x) :
    Agree W (copiesStep acc s).1.toList ∧ (∀ x ∈ (copiesStep acc s).2.toList, FrameOk W x) ∧
      ivGates (copiesStep acc s).1.toList ∪ FGs (copiesStep acc s).2.toList =
        ivGates acc.1.toList ∪ FGs acc.2.toList ∪ (if s.reps ≤ 1 then ∅ else gs (segGates s)) := by
  unfold copiesStep
  by_cases hr : s.reps ≤ 1
  · simp only [hr, ↓reduceIte]; exact ⟨hI, hF, by simp⟩
  simp only [hr, ↓reduceIte]
  have hseg : segGates s = (List.range s.reps).flatMap fun m => piecesGates (s.copy m) := by
    simp [segGates, hr]
  have hcopy : ∀ m < s.reps, ∀ x ∈ piecesGates s.ps, ((x.1 + m * s.shift, x.2) : ℤ × ℕ) ∈ segGates s := by
    intro m hm x hx
    rw [hseg, List.mem_flatMap]
    exact ⟨m, List.mem_range.2 hm, by rw [copy_gates]; exact List.mem_map.2 ⟨x, hx, rfl⟩⟩
  have hnn0 : ∀ x ∈ piecesGates s.ps, 0 ≤ x.1 := fun x hx => by
    simpa using hnn _ (hcopy 0 (by omega) x hx)
  have hW0 : ∀ x ∈ piecesGates s.ps, W x.1.toNat = x.2 := fun x hx => by
    simpa using hW _ (hcopy 0 (by omega) x hx)
  obtain ⟨hrA, hrG⟩ := gate_rep hG s.ps hnn0 W hW0
  have hsound := (hG s.ps hnn0).1
  set r := gateIvs s.ps with hrdef
  by_cases he : r.isEmpty
  · simp only [he, ↓reduceIte]
    have hnil : piecesGates s.ps = [] := by
      rcases hl : piecesGates s.ps with _ | ⟨x, l⟩
      · rfl
      · exfalso
        obtain ⟨iv, hiv, -⟩ := (hG s.ps hnn0).2 x (by rw [hl]; exact List.mem_cons_self ..)
        rw [← hrdef] at hiv
        simp [Array.isEmpty_iff.1 (by simpa using he)] at hiv
    have : gs (segGates s) = ∅ := by
      rw [hseg]; ext a; simp [mem_gs, copy_gates, hnil]
    rw [this, Finset.union_empty]; exact ⟨hI, hF, rfl⟩
  simp only [he, ↓reduceIte, Bool.false_eq_true]
  split_ifs with hcase
  · -- the copies' intervals
    have hrep : ∀ m < s.reps, Agree W (gateIvs (s.copy m)).toList ∧
        ivGates (gateIvs (s.copy m)).toList = gs (piecesGates (s.copy m)) := by
      intro m hm
      refine gate_rep hG _ (fun x hx => hnn x ?_) W (fun x hx => hW x ?_) <;>
        (rw [hseg, List.mem_flatMap]; exact ⟨m, List.mem_range.2 hm, hx⟩)
    refine ⟨?_, hF, ?_⟩
    · intro z hz
      rw [Array.toList_append, List.mem_append] at hz
      rcases hz with hz | hz
      · exact hI z hz
      · rw [Array.toList_flatMap, List.mem_flatMap] at hz
        obtain ⟨m, hm, hz⟩ := hz
        rw [Array.toList_range, List.mem_range] at hm
        exact (hrep m hm).1 z hz
    · rw [Array.toList_append, ivGates_append']
      have : ivGates ((Array.range s.reps).flatMap fun m => gateIvs (s.copy m)).toList = gs (segGates s) := by
        ext a
        rw [Array.toList_flatMap, mem_ivGates_flatMap, hseg, mem_gs]
        simp only [Array.toList_range, List.mem_range, List.mem_flatMap]
        constructor
        · rintro ⟨m, hm, ha⟩
          rw [(hrep m hm).2, mem_gs] at ha
          obtain ⟨x, hx, rfl⟩ := ha
          exact ⟨x, ⟨m, hm, hx⟩, rfl⟩
        · rintro ⟨x, ⟨m, hm, hx⟩, rfl⟩
          exact ⟨m, hm, by rw [(hrep m hm).2, mem_gs]; exact ⟨x, hx, rfl⟩⟩
      rw [this]
      exact Finset.union_right_comm _ _ _
  · -- one frame
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_le, gt_iff_lt, not_lt] at hcase
    obtain ⟨hsh, hspan⟩ := hcase
    set lo := r.foldl (fun m x => min m x.1) r[0]!.1 with hlo
    set hi := r.foldl (fun m x => max m x.2.1) 0 with hhi
    have hlo' : ∀ z ∈ r.toList, lo ≤ z.1 := fun z hz => by
      rw [hlo, ← Array.foldl_toList]; exact (foldl_min_le r.toList _).2 z hz
    have hhi' : ∀ z ∈ r.toList, z.2.1 ≤ hi := fun z hz => by
      rw [hhi, ← Array.foldl_toList]; exact (le_foldl_max r.toList _).2 z hz
    set x : Cl := Cl.mk lo s.shift.toNat s.reps (hi - lo) (r.map fun y => (y.1 - lo, y.2.1 - lo, y.2.2)) with hx
    have hsht : (s.shift.toNat : ℤ) = s.shift := Int.toNat_of_nonneg (by omega)
    have hxlo : x.lo = lo := rfl
    have hxp : x.period = s.shift.toNat := rfl
    have hxs : x.span = hi - lo := rfl
    have hres : ∀ b, b ∈ ivGates x.res.toList ↔ lo + b ∈ ivGates r.toList := by
      intro b
      simp only [hx, mem_ivGates, Array.toList_map, List.mem_map]
      constructor
      · rintro ⟨_, ⟨z, hz, rfl⟩, h1, h2⟩
        exact ⟨z, hz, by dsimp at h1; omega, by dsimp at h2; omega⟩
      · rintro ⟨z, hz, h1, h2⟩
        have := hlo' z hz
        exact ⟨_, ⟨z, hz, rfl⟩, by dsimp; omega, by dsimp; omega⟩
    have hxF : FrameOk W x := by
      refine ⟨fun y hy => ?_, by simpa [hx] using hspan, ?_⟩
      · simp only [hx, Array.toList_map, List.mem_map] at hy
        obtain ⟨z, hz, rfl⟩ := hy
        have := hhi' z hz
        dsimp; rw [hxs]; omega
      · rw [agree_frame]
        intro m hm y hy b h1 h2
        simp only [hx, Array.toList_map, List.mem_map] at hy
        obtain ⟨z, hz, rfl⟩ := hy
        dsimp at h1 h2 ⊢
        have hlz := hlo' z hz
        have hmem := hsound z hz (lo + b) (by omega) (by omega)
        have := hW _ (hcopy m hm _ hmem)
        dsimp only at this
        rw [← this, ← hsht, toNat_shift _ (Int.natCast_nonneg _), Int.toNat_natCast, hxlo, hxp]
        congr 1
        omega
    have hFG : FG x = gs (segGates s) := by
      ext a
      rw [FG, mem_frame, mem_gs, hseg]
      simp only [List.mem_flatMap, List.mem_range, copy_gates, List.mem_map]
      constructor
      · rintro ⟨m, hm, b, hb, rfl⟩
        rw [hres, hrG, mem_gs] at hb
        obtain ⟨g, hg, hgb⟩ := hb
        have := hnn0 g hg
        refine ⟨_, ⟨m, hm, g, hg, rfl⟩, ?_⟩
        dsimp
        rw [← hsht, toNat_shift _ this, hxlo, hxp]
        omega
      · rintro ⟨_, ⟨m, hm, g, hg, rfl⟩, rfl⟩
        have := hnn0 g hg
        have hgr : g.1.toNat ∈ ivGates r.toList := by rw [hrG, mem_gs]; exact ⟨g, hg, rfl⟩
        obtain ⟨z, hz, h1, h2⟩ := mem_ivGates.1 hgr
        have := hlo' z hz
        refine ⟨m, hm, g.1.toNat - lo, by rw [hres, show lo + (g.1.toNat - lo) = g.1.toNat by omega]; exact hgr, ?_⟩
        dsimp
        rw [← hsht, toNat_shift _ (by assumption), hxlo, hxp]
        omega
    refine ⟨hI, fun y hy => ?_, ?_⟩
    · rw [Array.toList_push, List.mem_append, List.mem_singleton] at hy
      rcases hy with hy | rfl
      · exact hF y hy
      · exact hxF
    · rw [Array.toList_push, FGs_append, ← hFG]
      simp [FGs, Finset.union_assoc]


theorem gs_append (A B : List (ℤ × ℕ)) : gs (A ++ B) = gs A ∪ gs B := by
  simp [gs, List.toFinset_append]

theorem copiesFold_inv (hG : GateIvsExact) (W : ℕ → ℕ) : ∀ (L : List Seg) (acc : Array (ℕ × ℕ × ℕ) × Array Cl),
    (∀ s ∈ L, ∀ x ∈ segGates s, 0 ≤ x.1) → (∀ s ∈ L, ∀ x ∈ segGates s, W x.1.toNat = x.2) →
    Agree W acc.1.toList → (∀ x ∈ acc.2.toList, FrameOk W x) →
    Agree W (L.foldl copiesStep acc).1.toList ∧ (∀ x ∈ (L.foldl copiesStep acc).2.toList, FrameOk W x) ∧
      ivGates (L.foldl copiesStep acc).1.toList ∪ FGs (L.foldl copiesStep acc).2.toList =
        ivGates acc.1.toList ∪ FGs acc.2.toList ∪ gs ((L.filter fun s => !decide (s.reps ≤ 1)).flatMap segGates)
  | [], acc, _, _, hI, hF => ⟨hI, hF, by simp [gs]⟩
  | s :: L, acc, hnn, hW, hI, hF => by
    obtain ⟨h1, h2, h3⟩ := copiesStep_inv hG W s acc (hnn s (List.mem_cons_self ..)) (hW s (List.mem_cons_self ..)) hI hF
    obtain ⟨h4, h5, h6⟩ := copiesFold_inv hG W L _ (fun t ht => hnn t (List.mem_cons_of_mem _ ht))
      (fun t ht => hW t (List.mem_cons_of_mem _ ht)) h1 h2
    refine ⟨h4, h5, ?_⟩
    rw [List.foldl_cons, h6, h3, List.filter_cons]
    by_cases hr : s.reps ≤ 1
    · simp [hr]
    · simp only [hr, ↓reduceIte, decide_false, Bool.not_false, List.flatMap_cons, gs_append]
      simp only [Finset.union_assoc]

/-- **`novelWidth` is the width of the segments' gates**, whose widths are `W`'s, given `GateIvsExact`. -/
theorem novelWidth_sum (hG : GateIvsExact) (segs : Array Seg) (hnn : ∀ x ∈ segsGates segs, 0 ≤ x.1) (W : ℕ → ℕ)
    (hW : ∀ x ∈ segsGates segs, W x.1.toNat = x.2) :
    novelWidth segs = ∑ a ∈ gs (segsGates segs), W a := by
  have hmem : ∀ s ∈ segs.toList, ∀ x ∈ segGates s, x ∈ segsGates segs := fun s hs x hx =>
    List.mem_flatMap.2 ⟨s, hs, hx⟩
  set qs := (segs.filter (·.reps ≤ 1)).flatMap (·.ps) with hqs
  have hpq : piecesGates qs = (segs.toList.filter fun s => decide (s.reps ≤ 1)).flatMap segGates := by
    rw [hqs, piecesGates, Array.toList_flatMap, Array.toList_filter, List.flatMap_assoc]
    refine List.flatMap_congr fun s hs => ?_
    have : s.reps ≤ 1 := by simpa using (List.mem_filter.1 hs).2
    simp [segGates, this, piecesGates]
  have hq : ∀ x ∈ piecesGates qs, x ∈ segsGates segs := by
    intro x hx
    rw [hpq, List.mem_flatMap] at hx
    obtain ⟨s, hs, hx⟩ := hx
    exact hmem s (List.mem_filter.1 hs).1 x hx
  obtain ⟨hA0, hG0⟩ := gate_rep hG qs (fun x hx => hnn x (hq x hx)) W (fun x hx => hW x (hq x hx))
  obtain ⟨hI, hF, hU⟩ := copiesFold_inv hG W segs.toList (gateIvs qs, #[])
    (fun s hs x hx => hnn x (hmem s hs x hx)) (fun s hs x hx => hW x (hmem s hs x hx)) hA0 (by simp)
  rw [Array.foldl_toList] at hI hF hU
  refine (tail_eq W _ _ _ hI hF).trans (congrArg (fun S => ∑ a ∈ S, W a) ?_)
  rw [hU, hG0, hpq]
  simp only [FGs, Finset.union_empty]
  rw [← gs_append]
  ext a
  simp only [mem_gs, List.mem_append, List.mem_flatMap, List.mem_filter, segsGates]
  constructor
  · rintro ⟨x, (⟨s, ⟨hs, -⟩, hx⟩ | ⟨s, ⟨hs, -⟩, hx⟩), rfl⟩ <;> exact ⟨x, ⟨s, hs, hx⟩, rfl⟩
  · rintro ⟨x, ⟨s, hs, hx⟩, rfl⟩
    by_cases hr : s.reps ≤ 1
    · exact ⟨x, Or.inl ⟨s, ⟨hs, by simpa using hr⟩, hx⟩, rfl⟩
    · exact ⟨x, Or.inr ⟨s, ⟨hs, by simpa using hr⟩, hx⟩, rfl⟩

theorem dedup_sum (L : List (ℤ × ℕ)) : (L.dedup.map Prod.snd).sum = ∑ x ∈ L.toFinset, x.2 := by
  rw [← List.sum_toFinset _ (List.nodup_dedup L)]
  congr 1
  ext x
  simp only [List.mem_toFinset, List.mem_dedup]

/-- **`novelWidth` is the width of the union.** Of segments whose gates are naturals, none named with two widths,
`Desc.novelWidth` is the total width of the distinct gates they name. -/
def NovelWidthExact : Prop :=
  ∀ segs : Array Seg, (∀ x ∈ segsGates segs, 0 ≤ x.1) →
    (∀ x ∈ segsGates segs, ∀ y ∈ segsGates segs, x.1 = y.1 → x.2 = y.2) →
    novelWidth segs = ((segsGates segs).dedup.map Prod.snd).sum

/-- **`NovelWidthExact`, from `GateIvsExact`.** -/
theorem novelWidthExact_of (hG : GateIvsExact) : NovelWidthExact := by
  intro segs hnn hcons
  classical
  set L := segsGates segs
  let W : ℕ → ℕ := fun a => if h : ∃ x ∈ L, x.1.toNat = a then (Classical.choose h).2 else 0
  have hinj : ∀ x ∈ L, ∀ y ∈ L, x.1.toNat = y.1.toNat → x = y := by
    intro x hx y hy h
    have h1 : x.1 = y.1 := by have := hnn x hx; have := hnn y hy; omega
    exact Prod.ext h1 (hcons x hx y hy h1)
  have hW : ∀ x ∈ L, W x.1.toNat = x.2 := by
    intro x hx
    have h : ∃ y ∈ L, y.1.toNat = x.1.toNat := ⟨x, hx, rfl⟩
    simp only [W, h, ↓reduceDIte]
    obtain ⟨hy, hyx⟩ := Classical.choose_spec h
    rw [hinj _ hy x hx hyx]
  rw [novelWidth_sum hG segs hnn W hW, dedup_sum]
  have himg : gs L = L.toFinset.image fun x => x.1.toNat := by
    ext a; simp [gs]
  rw [himg, Finset.sum_image fun x hx y hy h => hinj x (by simpa using hx) y (by simpa using hy) h]
  exact Finset.sum_congr rfl fun x hx => hW x (by simpa using hx)

/-! ## `gateIvs`, from `ivsOf` -/

/-- A piece's elements: element `t` of row `r` is `base + r * rowStride + t * stride`, with one row when it has none. -/
def elems (q : Piece) : List ℤ :=
  (List.range (max q.rows 1)).flatMap fun (r : ℕ) => (List.range q.len).map fun (t : ℕ) =>
    q.base + (r : ℤ) * q.rowStride + (t : ℤ) * q.stride

theorem pieceGates_gate {q : Piece} {w : ℕ} (h : q.tgt = .gate w) : pieceGates q = (elems q).map fun e => (e, w) := by
  simp [pieceGates, h, elems, List.map_flatMap, Function.comp_def]

theorem pieceGates_ref {q : Piece} {p : Bool} {i : ℕ} (h : q.tgt = .ref p i) : pieceGates q = [] := by
  simp [pieceGates, h]

theorem tgt_beq_gate (t : Tgt) (w : ℕ) : (t == Tgt.gate w) = true ↔ t = .gate w := by
  cases t with
  | ref p i => exact ⟨fun h => absurd h (by change ¬ false = true; simp), fun h => by cases h⟩
  | gate w' => change (w' == w) = true ↔ _; simp

/-- **`ivsOf` is the pieces' elements.** Of pieces whose elements are naturals, every element of an interval
`ivsOf` gives is one of a piece's, and each of theirs is in one of the intervals. -/
def IvsOfExact : Prop :=
  ∀ qs : Array Piece, (∀ q ∈ qs.toList, ∀ e ∈ elems q, 0 ≤ e) →
    (∀ iv ∈ (ivsOf qs).toList, ∀ a, iv.1 ≤ a → a < iv.2 → ∃ q ∈ qs.toList, (a : ℤ) ∈ elems q) ∧
    ∀ q ∈ qs.toList, ∀ e ∈ elems q, ∃ iv ∈ (ivsOf qs).toList, iv.1 ≤ e.toNat ∧ e.toNat < iv.2

theorem mem_qsort_toList {α : Type} {xs : Array α} {lt : α → α → Bool} {a : α} :
    a ∈ (xs.qsort lt).toList ↔ a ∈ xs.toList :=
  (Properties.PoubCheck.qsort_toList_perm xs lt).mem_iff

theorem mem_gateIvs {qs : Array Piece} {z : ℕ × ℕ × ℕ} : z ∈ (gateIvs qs).toList ↔
    ∃ w, (∃ q ∈ qs.toList, q.tgt = .gate w) ∧ ∃ iv ∈ (ivsOf (qs.filter (·.tgt == .gate w))).toList, z = (iv.1, iv.2, w) := by
  unfold gateIvs
  simp only [Array.toList_flatMap, List.mem_flatMap, Array.toList_map, List.mem_map]
  simp only [List.mem_eraseDups]
  simp only [mem_qsort_toList]
  simp only [Array.mem_toList_iff, Array.mem_filterMap]
  constructor
  · rintro ⟨w, ⟨q, hq, h⟩, iv, hiv, rfl⟩
    refine ⟨w, ⟨q, hq, ?_⟩, iv, hiv, rfl⟩
    revert h; cases q.tgt <;> simp
  · rintro ⟨w, ⟨q, hq, h⟩, iv, hiv, rfl⟩
    exact ⟨w, ⟨q, hq, by rw [h]⟩, iv, hiv, rfl⟩

/-- **`GateIvsExact`, from `IvsOfExact`**: `gateIvs` takes each width's pieces to `ivsOf`. -/
theorem gateIvsExact_of (hV : IvsOfExact) : GateIvsExact := by
  intro qs hnn
  have hfil : ∀ w, ∀ q ∈ (qs.filter (·.tgt == .gate w)).toList, q ∈ qs.toList ∧ q.tgt = .gate w := by
    intro w q hq
    rw [Array.toList_filter, List.mem_filter, tgt_beq_gate] at hq
    exact hq
  have hnn' : ∀ w, ∀ q ∈ (qs.filter (·.tgt == .gate w)).toList, ∀ e ∈ elems q, 0 ≤ e := by
    intro w q hq e he
    obtain ⟨hq, ht⟩ := hfil w q hq
    exact hnn (e, w) (List.mem_flatMap.2 ⟨q, hq, by rw [pieceGates_gate ht]; exact List.mem_map.2 ⟨e, he, rfl⟩⟩)
  refine ⟨fun z hz a h1 h2 => ?_, fun x hx => ?_⟩
  · obtain ⟨w, -, iv, hiv, rfl⟩ := mem_gateIvs.1 hz
    obtain ⟨q, hq, ha⟩ := (hV _ (hnn' w)).1 iv hiv a h1 h2
    obtain ⟨hq, ht⟩ := hfil w q hq
    exact List.mem_flatMap.2 ⟨q, hq, by rw [pieceGates_gate ht]; exact List.mem_map.2 ⟨_, ha, rfl⟩⟩
  · obtain ⟨q, hq, hx⟩ := List.mem_flatMap.1 hx
    rcases ht : q.tgt with ⟨p, i⟩ | w
    · rw [pieceGates_ref ht] at hx; cases hx
    · rw [pieceGates_gate ht, List.mem_map] at hx
      obtain ⟨e, he, rfl⟩ := hx
      have hqf : q ∈ (qs.filter (·.tgt == .gate w)).toList := by
        rw [Array.toList_filter, List.mem_filter, tgt_beq_gate]; exact ⟨hq, ht⟩
      obtain ⟨iv, hiv, h⟩ := (hV _ (hnn' w)).2 q hqf e he
      exact ⟨(iv.1, iv.2, w), mem_gateIvs.2 ⟨w, ⟨q, hq, ht⟩, iv, hiv, rfl⟩, h⟩

/-! ## `ivsOf`, from `Piece.pis` -/

/-- `x`'s set: `⋃_{c < cnt} [lo + c·period, lo + c·period + len)`. -/
def inPI (x : PI) (a : ℕ) : Prop := ∃ c < x.cnt, ∃ t < x.len, a = x.lo + c * x.period + t

/-- A list of intervals' set. -/
def inIvs (L : List (ℕ × ℕ)) (a : ℕ) : Prop := ∃ iv ∈ L, iv.1 ≤ a ∧ a < iv.2

theorem inIvs_append {A B : List (ℕ × ℕ)} {a : ℕ} : inIvs (A ++ B) a ↔ inIvs A a ∨ inIvs B a := by
  simp only [inIvs, List.mem_append]
  constructor
  · rintro ⟨iv, hiv | hiv, h⟩
    · exact Or.inl ⟨iv, hiv, h⟩
    · exact Or.inr ⟨iv, hiv, h⟩
  · rintro (⟨iv, hiv, h⟩ | ⟨iv, hiv, h⟩)
    · exact ⟨iv, Or.inl hiv, h⟩
    · exact ⟨iv, Or.inr hiv, h⟩

/-- **A periodic interval whose runs meet is one interval.** -/
theorem inPI_run (x : PI) (hc : 0 < x.cnt) (h : x.cnt = 1 ∨ x.period ≤ x.len) (a : ℕ) :
    inPI x a ↔ x.lo ≤ a ∧ a < x.lo + (x.cnt - 1) * x.period + x.len := by
  constructor
  · rintro ⟨c, hc', t, ht, rfl⟩
    have : c * x.period ≤ (x.cnt - 1) * x.period := Nat.mul_le_mul_right _ (by omega)
    omega
  · rintro ⟨h1, h2⟩
    rcases h with h | h
    · rw [h] at h2
      exact ⟨0, by omega, a - x.lo, by simp at h2; omega, by simp; omega⟩
    · set d := a - x.lo
      rcases Nat.eq_zero_or_pos x.period with hp | hp
      · rw [hp] at h2
        exact ⟨0, hc, d, by simp at h2; omega, by simp [hp]; omega⟩
      · by_cases hq : d / x.period < x.cnt - 1
        · refine ⟨d / x.period, by omega, d % x.period, by have := Nat.mod_lt d hp; omega, ?_⟩
          have := Nat.div_add_mod d x.period
          rw [Nat.mul_comm] at this
          omega
        · have hle : (x.cnt - 1) * x.period ≤ d := by
            have := Nat.div_mul_le_self d x.period
            have : (x.cnt - 1) * x.period ≤ d / x.period * x.period := Nat.mul_le_mul_right _ (by omega)
            omega
          exact ⟨x.cnt - 1, by omega, d - (x.cnt - 1) * x.period, by omega, by omega⟩

theorem emitRuns (lo period len : ℕ) : ∀ (C : List ℕ) (out : Array (ℕ × ℕ)),
    (C.foldl (fun o c => o.push (lo + c * period, lo + c * period + len)) out).toList =
      out.toList ++ C.map fun c => (lo + c * period, lo + c * period + len)
  | [], out => by simp
  | c :: C, out => by rw [List.foldl_cons, emitRuns lo period len C]; simp

theorem emit_spec (out : Array (ℕ × ℕ)) (x : PI) (hc : 0 < x.cnt) (a : ℕ) :
    inIvs (PI.emit out x).toList a ↔ inIvs out.toList a ∨ inPI x a := by
  unfold PI.emit
  split_ifs with h
  · rw [Array.toList_push, inIvs_append, inPI_run x hc (Or.inr h)]
    simp [inIvs]
  · rw [← Array.foldl_toList, Array.toList_range, emitRuns, inIvs_append]
    refine or_congr_right ?_
    simp only [inIvs, List.mem_map, List.mem_range, inPI]
    constructor
    · rintro ⟨_, ⟨c, hc', rfl⟩, h1, h2⟩
      exact ⟨c, hc', a - (x.lo + c * x.period), by dsimp at h2; omega, by dsimp at h1; omega⟩
    · rintro ⟨c, hc', t, ht, rfl⟩
      exact ⟨_, ⟨c, hc', rfl⟩, by dsimp; omega, by dsimp; omega⟩

/-- **Two periodic intervals of one period and count, the second starting within the first's first run, are one.** -/
theorem merge_spec (c x : PI) (hp : c.period = x.period) (hn : c.cnt = x.cnt) (h1 : c.lo ≤ x.lo)
    (h2 : x.lo ≤ c.lo + c.len) (a : ℕ) :
    inPI { c with len := max c.len (x.lo + x.len - c.lo) } a ↔ inPI c a ∨ inPI x a := by
  simp only [inPI]
  constructor
  · rintro ⟨k, hk, t, ht, rfl⟩
    by_cases htc : t < c.len
    · exact Or.inl ⟨k, hk, t, htc, rfl⟩
    · refine Or.inr ⟨k, by omega, t - (x.lo - c.lo), by omega, ?_⟩
      rw [← hp]; omega
  · rintro (⟨k, hk, t, ht, rfl⟩ | ⟨k, hk, t, ht, rfl⟩)
    · exact ⟨k, hk, t, by omega, rfl⟩
    · refine ⟨k, by omega, x.lo - c.lo + t, by omega, ?_⟩
      rw [hp]; omega

theorem PI.le_lo {c x : PI} (h : PI.le c x = true) (hp : c.period = x.period) (hn : c.cnt = x.cnt) : c.lo ≤ x.lo := by
  simp only [PI.le, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  omega

/-- The current periodic interval's set, none before the first. -/
def inCur (st : Option PI) (a : ℕ) : Prop := match st with
  | some c => inPI c a
  | none => False

theorem mergeFold_spec : ∀ (L : List PI) (st : Array (ℕ × ℕ) × Option PI),
    L.Pairwise (fun x y => PI.le x y = true) → (∀ x ∈ L, 0 < x.cnt) →
    (∀ c, st.2 = some c → 0 < c.cnt ∧ ∀ z ∈ L, PI.le c z = true) →
    (∀ c, (L.foldl PI.mergeStep st).2 = some c → 0 < c.cnt) ∧
      ∀ a, inIvs (L.foldl PI.mergeStep st).1.toList a ∨ inCur (L.foldl PI.mergeStep st).2 a ↔
        inIvs st.1.toList a ∨ inCur st.2 a ∨ ∃ x ∈ L, inPI x a
  | [], st, _, _, hst => ⟨fun c hc => (hst c hc).1, fun a => by simp⟩
  | x :: L, st, hs, hpos, hst => by
    obtain ⟨hx, hL⟩ := List.pairwise_cons.1 hs
    have hxpos := hpos x (List.mem_cons_self ..)
    have key : (∀ c, (PI.mergeStep st x).2 = some c → 0 < c.cnt ∧ ∀ z ∈ L, PI.le c z = true) ∧
        ∀ a, inIvs (PI.mergeStep st x).1.toList a ∨ inCur (PI.mergeStep st x).2 a ↔
          inIvs st.1.toList a ∨ inCur st.2 a ∨ inPI x a := by
      unfold PI.mergeStep
      rcases hc : st.2 with _ | c
      · refine ⟨fun c' hc' => ?_, fun a => ?_⟩
        · simp only [Option.some.injEq] at hc'
          subst hc'; exact ⟨hxpos, hx⟩
        · simp [inCur]
      · obtain ⟨hcpos, hcle⟩ := hst c hc
        dsimp only
        split_ifs with hm
        · simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hm
          obtain ⟨⟨hp, hn⟩, h2⟩ := hm
          have h1 := PI.le_lo (hcle x (List.mem_cons_self ..)) hp hn
          refine ⟨fun c' hc' => ?_, fun a => ?_⟩
          · simp only [Option.some.injEq] at hc'
            subst hc'
            exact ⟨hcpos, fun z hz => by
              have := hcle z (List.mem_cons_of_mem _ hz); simpa [PI.le] using this⟩
          · simp only [inCur]
            rw [merge_spec c x hp hn h1 h2]
        · refine ⟨fun c' hc' => ?_, fun a => ?_⟩
          · simp only [Option.some.injEq] at hc'
            subst hc'; exact ⟨hxpos, hx⟩
          · simp only [inCur]
            rw [emit_spec _ _ hcpos]
            tauto
    obtain ⟨h1, h2⟩ := mergeFold_spec L (PI.mergeStep st x) hL (fun z hz => hpos z (List.mem_cons_of_mem _ hz)) key.1
    refine ⟨h1, fun a => ?_⟩
    rw [List.foldl_cons, h2, ← or_assoc, key.2 a]
    simp only [List.mem_cons]
    constructor
    · rintro ((h | h | h) | ⟨z, hz, h⟩)
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr ⟨x, Or.inl rfl, h⟩)
      · exact Or.inr (Or.inr ⟨z, Or.inr hz, h⟩)
    · rintro (h | h | ⟨z, rfl | hz, h⟩)
      · exact Or.inl (Or.inl h)
      · exact Or.inl (Or.inr (Or.inl h))
      · exact Or.inl (Or.inr (Or.inr h))
      · exact Or.inr ⟨z, hz, h⟩

theorem inPI_pos {x : PI} {a : ℕ} (h : inPI x a) : 0 < x.cnt ∧ 0 < x.len := by
  obtain ⟨c, hc, t, ht, -⟩ := h
  omega

theorem sorted_PI (L : List PI) : (L.mergeSort PI.le).Pairwise (fun x y => PI.le x y = true) :=
  List.pairwise_mergeSort
    (fun a b c hab hbc => by
      simp only [PI.le, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at *
      omega)
    (fun a b => by
      simp only [PI.le, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
      omega) L

/-- **`ivsOf`'s intervals are the union of its pieces' periodic intervals.** -/
theorem ivsOf_spec (qs : Array Piece) (a : ℕ) :
    inIvs (ivsOf qs).toList a ↔ ∃ x ∈ (qs.flatMap Piece.pis).toList, inPI x a := by
  unfold ivsOf
  dsimp only
  set xs := (qs.flatMap Piece.pis).filter fun x => !(x.cnt == 0 || x.len == 0) with hxs
  set out := (xs.filter fun x => x.cnt == 1 || x.len ≥ x.period).map
    fun x => (x.lo, x.lo + (x.cnt - 1) * x.period + x.len) with hout
  set sl := (xs.filter fun x => !(x.cnt == 1 || x.len ≥ x.period)).toList.mergeSort PI.le with hsl
  have hmemxs : ∀ x, x ∈ xs.toList ↔ x ∈ (qs.flatMap Piece.pis).toList ∧ 0 < x.cnt ∧ 0 < x.len := by
    intro x
    rw [hxs, Array.toList_filter, List.mem_filter]
    simp only [Bool.not_eq_true', Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq]
    exact and_congr_right fun _ => and_congr Nat.pos_iff_ne_zero.symm Nat.pos_iff_ne_zero.symm
  have hmemsl : ∀ x, x ∈ sl ↔ x ∈ xs.toList ∧ ¬ (x.cnt = 1 ∨ x.period ≤ x.len) := by
    intro x
    rw [hsl, (List.mergeSort_perm _ _).mem_iff, Array.toList_filter, List.mem_filter]
    simp only [Bool.not_eq_true', Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq, decide_eq_false_iff_not,
      not_le, not_or]
  have hout' : inIvs out.toList a ↔ ∃ x ∈ xs.toList, (x.cnt = 1 ∨ x.period ≤ x.len) ∧ inPI x a := by
    simp only [inIvs, hout, Array.toList_map, Array.toList_filter, List.mem_map, List.mem_filter, Bool.or_eq_true,
      beq_iff_eq, decide_eq_true_eq]
    constructor
    · rintro ⟨_, ⟨x, ⟨hx, hr⟩, rfl⟩, h⟩
      have hp := ((hmemxs x).1 hx).2
      exact ⟨x, hx, hr, (inPI_run x hp.1 hr a).2 h⟩
    · rintro ⟨x, hx, hr, h⟩
      have hp := ((hmemxs x).1 hx).2
      exact ⟨_, ⟨x, ⟨hx, hr⟩, rfl⟩, (inPI_run x hp.1 hr a).1 h⟩
  obtain ⟨hF1, hF2⟩ := mergeFold_spec sl (out, none) (sorted_PI _)
    (fun x hx => (((hmemxs x).1 ((hmemsl x).1 hx).1).2).1) (by simp)
  have hall : inIvs out.toList a ∨ inCur none a ∨ (∃ x ∈ sl, inPI x a) ↔
      ∃ x ∈ (qs.flatMap Piece.pis).toList, inPI x a := by
    rw [hout']
    simp only [inCur, false_or, hmemsl]
    constructor
    · rintro (⟨x, hx, -, h⟩ | ⟨x, ⟨hx, -⟩, h⟩) <;> exact ⟨x, ((hmemxs x).1 hx).1, h⟩
    · rintro ⟨x, hx, h⟩
      have hxs' := (hmemxs x).2 ⟨hx, inPI_pos h⟩
      by_cases hr : x.cnt = 1 ∨ x.period ≤ x.len
      · exact Or.inl ⟨x, hxs', hr, h⟩
      · exact Or.inr ⟨x, ⟨hxs', hr⟩, h⟩
  rw [← hall, ← hF2 a]
  split
  · rename_i o c heq
    rw [heq] at hF1 ⊢
    rw [emit_spec o c (hF1 c rfl) a]
    rfl
  · rename_i o heq
    rw [heq]
    simp [inCur]

/-- **`Piece.pis` is a piece's elements.** Of a piece whose elements are naturals, every element of a periodic
interval `pis` gives is one of the piece's, and each of the piece's is in one of them. -/
def PisExact : Prop :=
  ∀ q : Piece, (∀ e ∈ elems q, 0 ≤ e) →
    (∀ x ∈ q.pis.toList, ∀ a, inPI x a → ∃ e ∈ elems q, a = e.toNat) ∧
    ∀ e ∈ elems q, ∃ x ∈ q.pis.toList, inPI x e.toNat

/-- **`IvsOfExact`, from `PisExact`.** -/
theorem ivsOfExact_of (hP : PisExact) : IvsOfExact := by
  intro qs hnn
  refine ⟨fun iv hiv a h1 h2 => ?_, fun q hq e he => ?_⟩
  · obtain ⟨x, hx, ha⟩ := (ivsOf_spec qs a).1 ⟨iv, hiv, h1, h2⟩
    rw [Array.toList_flatMap, List.mem_flatMap] at hx
    obtain ⟨q, hq, hx⟩ := hx
    obtain ⟨e, he, rfl⟩ := (hP q (hnn q hq)).1 x hx a ha
    exact ⟨q, hq, by rw [Int.toNat_of_nonneg (hnn q hq e he)]; exact he⟩
  · obtain ⟨x, hx, ha⟩ := (hP q (hnn q hq)).2 e he
    exact (ivsOf_spec qs e.toNat).2 ⟨x, by rw [Array.toList_flatMap, List.mem_flatMap]; exact ⟨q, hq, hx⟩, ha⟩

/-! ## `Piece.pis` -/

/-- `Piece.pis`'s first element of a progression `b + i * s`, `i < k`: its smallest. -/
def rlo (s : ℤ) (k : ℕ) (b : ℤ) : ℕ := (if s ≥ 0 then b else b + ((k - 1 : ℕ) : ℤ) * s).toNat

/-- `Piece.pis`'s one row. -/
def one (q : Piece) : Array PI :=
  if q.len == 0 then #[]
  else if q.stride == 0 || q.len == 1 then #[⟨q.base.toNat, 1, 1, 1⟩]
  else
    let lo := (if q.stride > 0 then q.base else q.at (q.len - 1)).toNat
    if q.stride.natAbs == 1 then #[⟨lo, q.len, 1, 1⟩] else #[⟨lo, 1, q.stride.natAbs, q.len⟩]

theorem pis_eq (q : Piece) : q.pis =
    if q.rows ≤ 1 || q.len == 0 then one (q.row 0)
    else if q.stride.natAbs == 1 || q.len == 1 then
      #[⟨rlo q.rowStride q.rows (rlo q.stride q.len q.base : ℤ), q.len, q.rowStride.natAbs, q.rows⟩]
    else if q.rowStride.natAbs == 1 && q.stride.natAbs ≥ q.rows then
      #[⟨rlo q.stride q.len (rlo q.rowStride q.rows q.base : ℤ), q.rows, q.stride.natAbs, q.len⟩]
    else q.rowsOf.flatMap one := rfl

/-- **A progression is its smallest element on.** Of `b + i * s`, `i < k`, all naturals: they are
`rlo s k b + j * |s|`, `j < k`. -/
theorem ap (b s : ℤ) (k : ℕ) (hk : 0 < k) (hnn : ∀ i < k, 0 ≤ b + (i : ℤ) * s) (a : ℤ) :
    (∃ i < k, a = b + (i : ℤ) * s) ↔ ∃ j < k, a = (rlo s k b : ℤ) + (j : ℤ) * (s.natAbs : ℤ) := by
  unfold rlo
  rw [Int.natCast_natAbs]
  by_cases hs : 0 ≤ s
  · simp only [hs, ↓reduceIte]
    rw [Int.toNat_of_nonneg (by simpa using hnn 0 hk), abs_of_nonneg hs]
  · simp only [hs, ↓reduceIte]
    rw [Int.toNat_of_nonneg (hnn (k - 1) (by omega)), abs_of_neg (by omega)]
    constructor
    · rintro ⟨i, hi, rfl⟩
      refine ⟨k - 1 - i, by omega, ?_⟩
      push_cast [Nat.cast_sub (show 1 ≤ k by omega), Nat.cast_sub (show i ≤ k - 1 by omega)]
      ring
    · rintro ⟨j, hj, rfl⟩
      refine ⟨k - 1 - j, by omega, ?_⟩
      push_cast [Nat.cast_sub (show 1 ≤ k by omega), Nat.cast_sub (show j ≤ k - 1 by omega)]
      ring

theorem rlo_shift (b s δ : ℤ) (k : ℕ) (hk : 0 < k) (hnn : ∀ i < k, 0 ≤ b + (i : ℤ) * s)
    (hnn' : ∀ i < k, 0 ≤ b + δ + (i : ℤ) * s) : (rlo s k (b + δ) : ℤ) = rlo s k b + δ := by
  unfold rlo
  by_cases hs : 0 ≤ s
  · simp only [hs, ↓reduceIte]
    rw [Int.toNat_of_nonneg (by simpa using hnn' 0 hk), Int.toNat_of_nonneg (by simpa using hnn 0 hk)]
  · simp only [hs, ↓reduceIte]
    rw [Int.toNat_of_nonneg (hnn' (k - 1) (by omega)), Int.toNat_of_nonneg (hnn (k - 1) (by omega))]
    ring

/-- **A grid whose rows are runs is one periodic interval**: `b + j * s₂ + i * s₁` (`i < k₁`, `j < k₂`), each row a run
(`|s₁| = 1`, or one element). -/
theorem grid (b s₁ s₂ : ℤ) (k₁ k₂ : ℕ) (hk₁ : 0 < k₁) (hk₂ : 0 < k₂) (h : s₁.natAbs = 1 ∨ k₁ = 1)
    (hnn : ∀ j < k₂, ∀ i < k₁, 0 ≤ b + (j : ℤ) * s₂ + (i : ℤ) * s₁) (a : ℕ) :
    inPI ⟨rlo s₂ k₂ (rlo s₁ k₁ b : ℤ), k₁, s₂.natAbs, k₂⟩ a ↔
      ∃ j < k₂, ∃ i < k₁, (a : ℤ) = b + (j : ℤ) * s₂ + (i : ℤ) * s₁ := by
  have hrow : ∀ j < k₂, ∀ a' : ℤ, (∃ i < k₁, a' = b + (j : ℤ) * s₂ + (i : ℤ) * s₁) ↔
      ∃ t < k₁, a' = (rlo s₁ k₁ b : ℤ) + (j : ℤ) * s₂ + t := by
    intro j hj a'
    rw [ap (b + (j : ℤ) * s₂) s₁ k₁ hk₁ (hnn j hj) a',
      rlo_shift b s₁ _ k₁ hk₁ (fun i hi => by simpa using hnn 0 hk₂ i hi) (hnn j hj)]
    rcases h with h | h
    · rw [h]; simp
    · subst h
      constructor
      · rintro ⟨t, ht, rfl⟩
        obtain rfl : t = 0 := by omega
        exact ⟨0, by omega, by simp⟩
      · rintro ⟨t, ht, rfl⟩
        obtain rfl : t = 0 := by omega
        exact ⟨0, by omega, by simp⟩
  have hmin : ∀ j < k₂, 0 ≤ (rlo s₁ k₁ b : ℤ) + (j : ℤ) * s₂ := by
    intro j hj
    rw [← rlo_shift b s₁ _ k₁ hk₁ (fun i hi => by simpa using hnn 0 hk₂ i hi) (hnn j hj)]
    exact Int.natCast_nonneg _
  simp only [inPI]
  constructor
  · rintro ⟨c, hc, t, ht, rfl⟩
    obtain ⟨j, hj, hjc⟩ := (ap _ s₂ k₂ hk₂ hmin ((rlo s₂ k₂ (rlo s₁ k₁ b : ℤ) : ℤ) + (c : ℤ) * s₂.natAbs)).2 ⟨c, hc, rfl⟩
    refine ⟨j, hj, (hrow j hj _).2 ⟨t, ht, ?_⟩⟩
    rw [Int.natCast_natAbs] at hjc
    push_cast; linarith
  · rintro ⟨j, hj, i, hi, ha⟩
    obtain ⟨t, ht, hat⟩ := (hrow j hj a).1 ⟨i, hi, ha⟩
    obtain ⟨c, hc, hcj⟩ := (ap _ s₂ k₂ hk₂ hmin ((a : ℤ) - t)).1 ⟨j, hj, by linarith⟩
    refine ⟨c, hc, t, ht, ?_⟩
    rw [Int.natCast_natAbs] at hcj
    have : (a : ℤ) = ((rlo s₂ k₂ (rlo s₁ k₁ b : ℤ) + c * s₂.natAbs + t : ℕ) : ℤ) := by push_cast; linarith
    exact_mod_cast this

theorem one_spec (q : Piece) (hnn : ∀ t < q.len, 0 ≤ q.base + (t : ℤ) * q.stride) (a : ℕ) :
    (∃ x ∈ (one q).toList, inPI x a) ↔ ∃ t < q.len, (a : ℤ) = q.base + (t : ℤ) * q.stride := by
  unfold one
  by_cases h0 : q.len = 0
  · simp [h0]
  have hb := hnn 0 (by omega)
  simp only [Nat.cast_zero, zero_mul, add_zero] at hb
  simp only [show (q.len == 0) = false by simpa using h0, Bool.false_eq_true, ↓reduceIte]
  by_cases h1 : q.stride = 0 ∨ q.len = 1
  · simp only [show (q.stride == 0 || q.len == 1) = true by simpa using h1, ↓reduceIte]
    simp only [List.mem_singleton, exists_eq_left, inPI, Nat.lt_one_iff, exists_eq_left, zero_mul, add_zero]
    constructor
    · rintro rfl
      exact ⟨0, by omega, by simp [Int.toNat_of_nonneg hb]⟩
    · rintro ⟨t, ht, ha⟩
      have : (a : ℤ) = q.base := by
        rcases h1 with h1 | h1
        · rw [ha, h1]; simp
        · have : t = 0 := by omega
          rw [ha, this]; simp
      omega
  simp only [show (q.stride == 0 || q.len == 1) = false by simpa using h1, Bool.false_eq_true, ↓reduceIte]
  have hlo : (if q.stride > 0 then q.base else q.at (q.len - 1)).toNat = rlo q.stride q.len q.base := by
    unfold rlo Piece.at
    by_cases hs : 0 < q.stride
    · simp [hs, show (0 : ℤ) ≤ q.stride by omega]
    · simp [hs, show ¬ (0 : ℤ) ≤ q.stride by omega]
  rw [hlo, ap q.base q.stride q.len (by omega) hnn]
  by_cases h2 : q.stride.natAbs = 1
  · simp only [show (q.stride.natAbs == 1) = true by simpa using h2, ↓reduceIte]
    simp only [List.mem_singleton, exists_eq_left, inPI, h2]
    constructor
    · rintro ⟨c, hc, t, ht, rfl⟩
      exact ⟨t, ht, by push_cast; omega⟩
    · rintro ⟨j, hj, ha⟩
      exact ⟨0, by omega, j, hj, by push_cast at ha; omega⟩
  · simp only [show (q.stride.natAbs == 1) = false by simpa using h2, Bool.false_eq_true, ↓reduceIte,
      List.mem_singleton, exists_eq_left, inPI]
    constructor
    · rintro ⟨c, hc, t, ht, rfl⟩
      exact ⟨c, hc, by push_cast; omega⟩
    · rintro ⟨j, hj, ha⟩
      refine ⟨j, hj, 0, by omega, ?_⟩
      rw [Int.natCast_natAbs] at ha
      exact_mod_cast (show (a : ℤ) = ((rlo q.stride q.len q.base + j * q.stride.natAbs + 0 : ℕ) : ℤ) by
        push_cast; linarith)

theorem mem_elems {q : Piece} {e : ℤ} : e ∈ elems q ↔
    ∃ r < max q.rows 1, ∃ t < q.len, e = q.base + (r : ℤ) * q.rowStride + (t : ℤ) * q.stride := by
  unfold elems
  constructor
  · intro h
    obtain ⟨r, hr, h⟩ := List.mem_flatMap.1 h
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 h
    exact ⟨r, List.mem_range.1 hr, t, List.mem_range.1 ht, rfl⟩
  · rintro ⟨r, hr, t, ht, rfl⟩
    exact List.mem_flatMap.2 ⟨r, List.mem_range.2 hr, List.mem_map.2 ⟨t, List.mem_range.2 ht, rfl⟩⟩

theorem pis_spec (q : Piece) (hnn : ∀ e ∈ elems q, 0 ≤ e) (a : ℕ) :
    (∃ x ∈ q.pis.toList, inPI x a) ↔
      ∃ r < max q.rows 1, ∃ t < q.len, (a : ℤ) = q.base + (r : ℤ) * q.rowStride + (t : ℤ) * q.stride := by
  have hnn' : ∀ r < max q.rows 1, ∀ t < q.len, 0 ≤ q.base + (r : ℤ) * q.rowStride + (t : ℤ) * q.stride :=
    fun r hr t ht => hnn _ (mem_elems.2 ⟨r, hr, t, ht, rfl⟩)
  have hrow : ∀ r < max q.rows 1, ∀ a : ℕ, (∃ x ∈ (one (q.row r)).toList, inPI x a) ↔
      ∃ t < q.len, (a : ℤ) = q.base + (r : ℤ) * q.rowStride + (t : ℤ) * q.stride := by
    intro r hr a
    exact one_spec (q.row r) (fun t ht => hnn' r hr t ht) a
  rw [pis_eq]
  split_ifs with h1 h2 h3
  · simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at h1
    rw [hrow 0 (by omega) a]
    rcases h1 with h1 | h1
    · constructor
      · rintro ⟨t, ht, ha⟩; exact ⟨0, by omega, t, ht, by simpa using ha⟩
      · rintro ⟨r, hr, t, ht, ha⟩
        obtain rfl : r = 0 := by omega
        exact ⟨t, ht, by simpa using ha⟩
    · rw [h1]; simp
  · simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, not_or] at h1 h2
    simp only [List.mem_singleton, exists_eq_left]
    rw [grid q.base q.stride q.rowStride q.len q.rows (by omega) (by omega) h2
      (fun j hj i hi => hnn' j (by omega) i hi) a]
    rw [show max q.rows 1 = q.rows by omega]
  · simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, not_or, Bool.and_eq_true] at h1 h2 h3
    simp only [List.mem_singleton, exists_eq_left]
    rw [grid q.base q.rowStride q.stride q.rows q.len (by omega) (by omega) (Or.inl h3.1)
      (fun j hj i hi => by have := hnn' i (by omega) j hj; linarith) a]
    rw [show max q.rows 1 = q.rows by omega]
    constructor
    · rintro ⟨j, hj, i, hi, ha⟩; exact ⟨i, hi, j, hj, by linarith⟩
    · rintro ⟨i, hi, j, hj, ha⟩; exact ⟨j, hj, i, hi, by linarith⟩
  · simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, not_or] at h1
    simp only [Piece.rowsOf, Array.toList_flatMap, Array.toList_map, Array.toList_range, List.flatMap_map,
      List.mem_flatMap, List.mem_range]
    constructor
    · rintro ⟨x, ⟨r, hr, hx⟩, ha⟩
      obtain ⟨t, ht, h⟩ := (hrow r (by omega) a).1 ⟨x, hx, ha⟩
      exact ⟨r, by omega, t, ht, h⟩
    · rintro ⟨r, hr, t, ht, h⟩
      obtain ⟨x, hx, ha⟩ := (hrow r hr a).2 ⟨t, ht, h⟩
      exact ⟨x, ⟨r, by omega, hx⟩, ha⟩

/-- **`PisExact` holds.** -/
theorem pisExact : PisExact := by
  intro q hnn
  refine ⟨fun x hx a ha => ?_, fun e he => ?_⟩
  · obtain ⟨r, hr, t, ht, h⟩ := (pis_spec q hnn a).1 ⟨x, hx, ha⟩
    exact ⟨_, mem_elems.2 ⟨r, hr, t, ht, rfl⟩, by rw [← h]; simp⟩
  · obtain ⟨r, hr, t, ht, rfl⟩ := mem_elems.1 he
    have h0 := hnn _ he
    obtain ⟨x, hx, ha⟩ := (pis_spec q hnn _).2 ⟨r, hr, t, ht, Int.toNat_of_nonneg h0⟩
    exact ⟨x, hx, ha⟩

/-- **`NovelWidthExact`, from `PisExact`.** -/
theorem novelWidthExact_of_pis (hP : PisExact) : NovelWidthExact :=
  novelWidthExact_of (gateIvsExact_of (ivsOfExact_of hP))

/-- **`novelWidth` is the width of the union.** -/
theorem novelWidthExact : NovelWidthExact := novelWidthExact_of_pis pisExact

end NciCheck.Properties.SecurityProofs.Width
