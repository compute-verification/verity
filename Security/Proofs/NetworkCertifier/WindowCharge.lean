import Proofs.NetworkCertifier.AdviceV1
import Definitions.NetworkCertifier.Guarantees.WindowChargeKraft

/-!
# Advice v1's whole-window charge counts (Kraft), and the leak bound with it

`encard_charged_le` proves `Guarantees.EncardChargedLeWhole` for `WindowCharge.window` (`ChargeCount_v2{N, H, O}`),
and `bitsCount` gives `BitsCount` for its charge, so the leak bounds hold with it.

The argument is `AdviceV1`'s weights (`wt`: a state's width over `2^bits`), over a window's symbol sequence instead of
its decisions:
* `sum_owt_codeAll_le`: over sequences that pairwise **split** (`Split`: after a common prefix, their next symbols are
  of one total at disjoint intervals), the states the symbols lead to weigh at most the first. By induction on the
  length: the sequences' first symbols are one model's disjoint outcomes (`sum_owt_countStep_le`), and the tails after
  each split again.
* `syms_split`: two different windows' sequences from one carry split. A count comes before the values it counts, so
  where the counts differ the sequences split at the count; where they agree, they split at the first value that
  differs: a decision's two outcomes, an integer's two bins (`binned_split`), or a bin's two offsets
  (`uniform_split`). An order entry's rank step is `zigzag`, injective, and within its table's cap once the carry's
  rank is (`zigzag_le`).
* `sum_window_le`: one window's weights `2^(−charge)` sum to at most 1 from any carry (`finish_le`), and
  `AdviceV1.encard_runOf_le` counts the run.
-/

namespace NetTiming.WindowCharge

open Finset AdviceV1

/-! ### Sequences that split -/

/-- Two symbol sequences **split**: after a common prefix, their next symbols are of one total at disjoint
intervals `[cum, cum + freq)`. -/
def Split (s t : List Sym) : Prop :=
  ∃ (p : List Sym) (a b : Sym) (s' t' : List Sym), s = p ++ a :: s' ∧ t = p ++ b :: t' ∧ a.2.2 = b.2.2 ∧
    (a.1 + a.2.1 ≤ b.1 ∨ b.1 + b.2.1 ≤ a.1)

theorem Split.append {s t : List Sym} (h : Split s t) (u v : List Sym) : Split (s ++ u) (t ++ v) := by
  obtain ⟨p, a, b, s', t', rfl, rfl, h1, h2⟩ := h
  exact ⟨p, a, b, s' ++ u, t' ++ v, by simp, by simp, h1, h2⟩

theorem Split.prefix {s t : List Sym} (h : Split s t) (q : List Sym) : Split (q ++ s) (q ++ t) := by
  obtain ⟨p, a, b, s', t', rfl, rfl, h1, h2⟩ := h
  exact ⟨q ++ p, a, b, s', t', by simp, by simp, h1, h2⟩

theorem not_split_nil (t : List Sym) : ¬ Split [] t := by
  rintro ⟨p, a, b, s', t', h, -⟩
  cases p <;> simp at h

theorem split_cons_cons {x y : Sym} {s t : List Sym} (h : Split (x :: s) (y :: t)) :
    (x.2.2 = y.2.2 ∧ (x.1 + x.2.1 ≤ y.1 ∨ y.1 + y.2.1 ≤ x.1)) ∨ (x = y ∧ Split s t) := by
  obtain ⟨p, a, b, s', t', hs, ht, h1, h2⟩ := h
  cases p with
  | nil =>
    simp only [List.nil_append, List.cons.injEq] at hs ht
    obtain ⟨rfl, -⟩ := hs
    obtain ⟨rfl, -⟩ := ht
    exact Or.inl ⟨h1, h2⟩
  | cons c p =>
    simp only [List.cons_append, List.cons.injEq] at hs ht
    obtain ⟨rfl, rfl⟩ := hs
    obtain ⟨rfl, rfl⟩ := ht
    exact Or.inr ⟨rfl, p, a, b, s', t', rfl, rfl, h1, h2⟩

/-! ### Symbols, counted -/

theorem good_codeAll : ∀ (s : List Sym) {st st' : Tally}, Good st → codeAll st s = some st' → Good st'
  | [], st, st', hst, h => by cases h; exact hst
  | a :: s, st, st', hst, h => by
    simp only [codeAll, Option.bind_eq_some_iff] at h
    obtain ⟨s1, h1, h2⟩ := h
    exact good_codeAll s (countStep_spec hst h1).2.1 h2

theorem eq_singleton_of_nil {ι : Type*} {I : Finset ι} {f : ι → List Sym} {i : ι} (hi : i ∈ I) (h0 : f i = [])
    (hsp : ∀ i ∈ I, ∀ j ∈ I, i ≠ j → Split (f i) (f j)) : I = {i} :=
  eq_singleton_iff_unique_mem.2 ⟨hi, fun j hj => by
    by_contra hji
    have := hsp i hi j hj (Ne.symm hji)
    rw [h0] at this
    exact not_split_nil _ this⟩

/-- **Symbols, counted (Kraft)**: over sequences of at most `n` symbols that pairwise split, the states they lead to
weigh at most the first. -/
theorem sum_owt_codeAll_le {ι : Type*} (n : ℕ) : ∀ (I : Finset ι) (f : ι → List Sym) (st : Tally), Good st →
    (∀ i ∈ I, (f i).length ≤ n) → (∀ i ∈ I, ∀ j ∈ I, i ≠ j → Split (f i) (f j)) →
    ∑ i ∈ I, owt (codeAll st (f i)) ≤ wt st := by
  induction n with
  | zero =>
    intro I f st hst hlen hsp
    rcases I.eq_empty_or_nonempty with rfl | ⟨i, hi⟩
    · simpa using wt_nonneg st
    · have h0 : f i = [] := List.length_eq_zero_iff.1 (Nat.le_zero.1 (hlen i hi))
      rw [eq_singleton_of_nil hi h0 hsp, sum_singleton, h0]
      exact le_rfl
  | succ n ih =>
    intro I f st hst hlen hsp
    by_cases h0 : ∃ i ∈ I, f i = []
    · obtain ⟨i, hi, h0⟩ := h0
      rw [eq_singleton_of_nil hi h0 hsp, sum_singleton, h0]
      exact le_rfl
    push Not at h0
    classical
    let hd : ι → Sym := fun i => (f i).headD (0, 0, 0)
    let tl : ι → List Sym := fun i => (f i).tail
    have hf : ∀ i ∈ I, f i = hd i :: tl i := fun i hi => by
      obtain ⟨x, xs, hx⟩ := List.exists_cons_of_ne_nil (h0 i hi)
      simp [hd, tl, hx]
    rw [← sum_fiberwise_of_maps_to (g := hd) (t := I.image hd) (fun i hi => mem_image_of_mem _ hi)]
    have hfib : ∀ h ∈ I.image hd,
        ∑ i ∈ I with hd i = h, owt (codeAll st (f i)) ≤ owt (countStep st h.1 h.2.1 h.2.2) := by
      intro h _
      have hc : ∀ i ∈ I.filter (fun i => hd i = h),
          codeAll st (f i) = (countStep st h.1 h.2.1 h.2.2).bind fun st' => codeAll st' (tl i) := by
        intro i hi
        rw [mem_filter] at hi
        rw [hf i hi.1, hi.2]
        rfl
      rw [sum_congr rfl fun i hi => congrArg owt (hc i hi)]
      cases hs : countStep st h.1 h.2.1 h.2.2 with
      | none => simp
      | some st' =>
        obtain ⟨⟨hf1, -, -⟩, hg, -⟩ := countStep_spec hst hs
        simp only [Option.bind_some, owt_some]
        apply ih _ tl st' hg
        · intro i hi
          rw [mem_filter] at hi
          have := hlen i hi.1
          rw [hf i hi.1] at this
          simpa [tl] using this
        · intro i hi j hj hij
          rw [mem_filter] at hi hj
          have := hsp i hi.1 j hj.1 hij
          rw [hf i hi.1, hf j hj.1, hi.2, hj.2] at this
          rcases split_cons_cons this with ⟨-, hd'⟩ | ⟨-, hs'⟩
          · omega
          · exact hs'
    refine (sum_le_sum hfib).trans ?_
    rcases (I.image hd).eq_empty_or_nonempty with he | ⟨h₀, h₀m⟩
    · rw [he, sum_empty]
      exact wt_nonneg st
    have hpair : ∀ h ∈ I.image hd, ∀ h' ∈ I.image hd, h ≠ h' →
        h.2.2 = h'.2.2 ∧ (h.1 + h.2.1 ≤ h'.1 ∨ h'.1 + h'.2.1 ≤ h.1) := by
      intro h hm h' hm' hne
      obtain ⟨i, hi, rfl⟩ := mem_image.1 hm
      obtain ⟨j, hj, rfl⟩ := mem_image.1 hm'
      have := hsp i hi j hj fun e => hne (congrArg hd e)
      rw [hf i hi, hf j hj] at this
      rcases split_cons_cons this with h1 | ⟨e, -⟩
      · exact h1
      · exact absurd e hne
    have hT : ∀ h ∈ I.image hd, h.2.2 = h₀.2.2 := fun h hm => by
      by_cases e : h = h₀
      · rw [e]
      · exact (hpair h hm h₀ h₀m e).1
    calc ∑ h ∈ I.image hd, owt (countStep st h.1 h.2.1 h.2.2)
        = ∑ h ∈ I.image hd, owt (countStep st h.1 h.2.1 h₀.2.2) := sum_congr rfl fun h hm => by rw [hT h hm]
      _ ≤ wt st := sum_owt_countStep_le _ hst (fun h => h.1) (fun h => h.2.1) h₀.2.2
          fun i hi j hj hij => (hpair i hi j hj hij).2

/-! ### The values split -/

theorem uniform_of_le {u n : ℕ} (h : n ≤ MAX_TOTAL) : uniform u n = if 1 < n then [(u, 1, n)] else [] := by
  rw [uniform, ite_eq_left h]

theorem uniform_of_gt {u n : ℕ} (h : ¬ n ≤ MAX_TOTAL) :
    uniform u n = uniform (u / 2 ^ CHUNK) ((n - 1) / 2 ^ CHUNK + 1) ++ [(u % 2 ^ CHUNK, 1, 2 ^ CHUNK)] := by
  rw [uniform, ite_eq_right h]

/-- **A bin's offsets split**: two offsets of one uniform value. -/
theorem uniform_split (n : ℕ) : ∀ {u u' : ℕ}, u < n → u' < n → u ≠ u' → Split (uniform u n) (uniform u' n) := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro u u' hu hu' hne
    by_cases hn : n ≤ MAX_TOTAL
    · have h1 : 1 < n := by omega
      rw [uniform_of_le hn, uniform_of_le hn, ite_eq_left h1, ite_eq_left h1]
      exact ⟨[], (u, 1, n), (u', 1, n), [], [], rfl, rfl, rfl, by dsimp only; omega⟩
    · rw [uniform_of_gt hn, uniform_of_gt hn]
      have hm : (n - 1) / 2 ^ CHUNK + 1 < n := by simp only [CHUNK, MAX_TOTAL] at *; omega
      by_cases hq : u / 2 ^ CHUNK = u' / 2 ^ CHUNK
      · rw [hq]
        apply Split.prefix
        have hr : u % 2 ^ CHUNK ≠ u' % 2 ^ CHUNK := fun e => hne (by
          rw [← Nat.div_add_mod u (2 ^ CHUNK), ← Nat.div_add_mod u' (2 ^ CHUNK), hq, e])
        exact ⟨[], _, _, [], [], rfl, rfl, rfl, by dsimp only; omega⟩
      · apply Split.append
        have h1 := Nat.div_le_div_right (c := 2 ^ CHUNK) (show u ≤ n - 1 by omega)
        have h2 := Nat.div_le_div_right (c := 2 ^ CHUNK) (show u' ≤ n - 1 by omega)
        exact ih _ hm (by omega) (by omega) hq

theorem low_le_size (v : ℕ) : low v.size ≤ v := by
  unfold low
  split_ifs with h
  · exact Nat.zero_le _
  · exact Nat.lt_size.1 (by omega)

/-- **Two integers split** (`binned`): at their bins when those differ, else at their offsets. -/
theorem binned_split (t : Table) {hi v v' : ℕ} (hv : v ≤ hi) (hv' : v' ≤ hi) (hne : v ≠ v') :
    Split (binned t hi v) (binned t hi v') := by
  unfold binned
  by_cases hj : v.size = v'.size
  · have hl := low_le_size v
    have hl' := low_le_size v'
    have hs := Nat.lt_size_self v
    have hs' := Nat.lt_size_self v'
    rw [hj] at hl hs ⊢
    refine Split.prefix (uniform_split _ ?_ ?_ (by omega)) [_]
    · generalize 2 ^ v'.size = P at *
      generalize low v'.size = a at *
      omega
    · generalize 2 ^ v'.size = P at *
      generalize low v'.size = a at *
      omega
  · refine ⟨[], _, _, _, _, rfl, rfl, rfl, ?_⟩
    rcases Nat.lt_or_gt_of_ne hj with h | h
    · exact Or.inl (take_sum_add_getD_le _ h)
    · exact Or.inr (take_sum_add_getD_le _ h)

theorem le_countHi {n S : ℕ} (h : n ≤ S) : n ≤ countHi S := by
  have := Nat.lt_size_self S
  unfold countHi
  omega

theorem zigzag_le {r rank : ℕ} (hr : r ≤ INDEX_CAP) (hk : rank ≤ INDEX_CAP + 1) : zigzag r rank ≤ ZIG_CAP := by
  unfold zigzag ZIG_CAP
  split_ifs <;> omega

theorem zigzag_ne {r r' rank : ℕ} (h : r ≠ r') : zigzag r rank ≠ zigzag r' rank := by
  unfold zigzag
  split_ifs <;> omega

/-! ### Two windows split -/

theorem decSyms_split : ∀ (cy : AdviceV1.Carry) (ds ds' : List Bool) {p p' : List Sym × AdviceV1.Carry},
    ds.length = ds'.length → ds ≠ ds' → decSyms cy ds = some p → decSyms cy ds' = some p' → Split p.1 p'.1
  | _, [], [], _, _, _, hne, _, _ => absurd rfl hne
  | _, [], _ :: _, _, _, hl, _, _, _ => by simp at hl
  | _, _ :: _, [], _, _, hl, _, _, _ => by simp at hl
  | cy, b :: ds, b' :: ds', p, p', hl, hne, h, h' => by
    simp only [decSyms, Option.bind_eq_some_iff] at h h'
    obtain ⟨f, hf, h⟩ := h
    obtain ⟨f', hf', h'⟩ := h'
    rw [hf, Option.some.injEq] at hf'
    subst hf'
    by_cases hc : 1 ≤ f.1 ∧ 1 ≤ f.2 ∧ f.1 + f.2 ≤ MAX_TOTAL
    swap
    · rw [ite_eq_right hc] at h
      cases h
    rw [ite_eq_left hc] at h h'
    obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨q', hq', rfl⟩ := Option.map_eq_some_iff.1 h'
    by_cases hb : b = b'
    · subst hb
      exact Split.prefix (decSyms_split _ ds ds' (by simpa using hl) (fun e => hne (by rw [e])) hq hq') [_]
    · refine ⟨[], _, _, q.1, q'.1, rfl, rfl, rfl, ?_⟩
      cases b <;> cases b' <;> simp at hb ⊢

theorem hitSyms_split (bk : Book) : ∀ (hs hs' : List ℕ) {p p' : List Sym},
    hs.length = hs'.length → hs ≠ hs' → hitSyms bk hs = some p → hitSyms bk hs' = some p' → Split p p'
  | [], [], _, _, _, hne, _, _ => absurd rfl hne
  | [], _ :: _, _, _, hl, _, _, _ => by simp at hl
  | _ :: _, [], _, _, hl, _, _, _ => by simp at hl
  | s :: ss, s' :: ss', p, p', hl, hne, h, h' => by
    simp only [hitSyms] at h h'
    by_cases h1 : s ≤ HIT_CAP
    swap
    · rw [ite_eq_right h1] at h
      cases h
    by_cases h2 : s' ≤ HIT_CAP
    swap
    · rw [ite_eq_right h2] at h'
      cases h'
    rw [ite_eq_left h1] at h
    rw [ite_eq_left h2] at h'
    obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨q', hq', rfl⟩ := Option.map_eq_some_iff.1 h'
    by_cases hs : s = s'
    · subst hs
      exact Split.prefix (hitSyms_split bk ss ss' (by simpa using hl) (fun e => hne (by rw [e])) hq hq') _
    · exact (binned_split bk.hit h1 h2 hs).append q q'

theorem entrySyms_split (bk : Book) : ∀ (pos rank : ℕ) (es es' : List (ℕ × ℕ)) {p p' : List Sym × ℕ × ℕ},
    rank ≤ INDEX_CAP + 1 → es.length = es'.length → es ≠ es' →
    entrySyms bk pos rank es = some p → entrySyms bk pos rank es' = some p' → Split p.1 p'.1
  | _, _, [], [], _, _, _, _, hne, _, _ => absurd rfl hne
  | _, _, [], _ :: _, _, _, _, hl, _, _, _ => by simp at hl
  | _, _, _ :: _, [], _, _, _, hl, _, _, _ => by simp at hl
  | pos, rank, e :: es, e' :: es', p, p', hr, hl, hne, h, h' => by
    simp only [entrySyms] at h h'
    by_cases h1 : pos ≤ e.1 ∧ e.1 < 2 ^ 32 - 1 ∧ e.1 - pos ≤ GAP_CAP ∧ e.2 ≤ INDEX_CAP
    swap
    · rw [ite_eq_right h1] at h
      cases h
    by_cases h2 : pos ≤ e'.1 ∧ e'.1 < 2 ^ 32 - 1 ∧ e'.1 - pos ≤ GAP_CAP ∧ e'.2 ≤ INDEX_CAP
    swap
    · rw [ite_eq_right h2] at h'
      cases h'
    rw [ite_eq_left h1] at h
    rw [ite_eq_left h2] at h'
    obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨q', hq', rfl⟩ := Option.map_eq_some_iff.1 h'
    by_cases hp : e.1 = e'.1
    · by_cases hk : e.2 = e'.2
      · have he : e = e' := Prod.ext hp hk
        subst he
        apply Split.prefix
        apply Split.prefix
        exact entrySyms_split bk _ _ es es' (by omega) (by simpa using hl) (fun x => hne (by rw [x])) hq hq'
      · dsimp only
        rw [hp]
        apply Split.prefix
        exact (binned_split bk.index (zigzag_le h1.2.2.2 hr) (zigzag_le h2.2.2.2 hr) (zigzag_ne hk)).append _ _
    · exact (binned_split bk.gap h1.2.2.1 h2.2.2.1 (by omega)).append _ _

theorem window_eq {x y : Window} (h1 : x.ds = y.ds) (h2 : x.hits = y.hits) (h3 : x.order = y.order) : x = y := by
  cases x
  cases y
  simp only at h1 h2 h3
  subst h1 h2 h3
  rfl

/-- **Two windows split**: from one carry, two different windows' symbol sequences split. -/
theorem syms_split (bk : Book) (N H O : ℕ) (cy : Carry) {x y : Window} {p q : List Sym × Carry}
    (hx : x.ds.length ≤ N ∧ x.hits.length ≤ H ∧ x.order.length ≤ O)
    (hy : y.ds.length ≤ N ∧ y.hits.length ≤ H ∧ y.order.length ≤ O) (hr : cy.rank ≤ INDEX_CAP + 1) (hne : x ≠ y)
    (h : syms bk N H O cy x = some p) (h' : syms bk N H O cy y = some q) : Split p.1 q.1 := by
  simp only [syms, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h h'
  obtain ⟨d, hd, hs, hhs, e, he, rfl⟩ := h
  obtain ⟨d', hd', hs', hhs', e', he', rfl⟩ := h'
  dsimp only
  by_cases hdl : x.ds.length = y.ds.length
  swap
  · exact (binned_split _ (le_countHi hx.1) (le_countHi hy.1) hdl).append _ _
  rw [hdl]
  apply Split.prefix
  by_cases hds : x.ds = y.ds
  swap
  · exact (decSyms_split _ _ _ hdl hds hd hd').append _ _
  rw [hds, hd', Option.some.injEq] at hd
  subst hd
  apply Split.prefix
  by_cases hhl : x.hits.length = y.hits.length
  swap
  · exact (binned_split _ (le_countHi hx.2.1) (le_countHi hy.2.1) hhl).append _ _
  rw [hhl]
  apply Split.prefix
  by_cases hhe : x.hits = y.hits
  swap
  · exact (hitSyms_split bk _ _ hhl hhe hhs hhs').append _ _
  rw [hhe, hhs', Option.some.injEq] at hhs
  subst hhs
  apply Split.prefix
  apply Split.prefix
  by_cases hol : x.order.length = y.order.length
  swap
  · exact (binned_split _ (le_countHi hx.2.2) (le_countHi hy.2.2) hol).append _ _
  rw [hol]
  apply Split.prefix
  exact (entrySyms_split bk _ _ _ _ hr hol (fun ho => hne (window_eq hds hhe ho)) he he').append _ _

/-! ### One window, and the run -/

theorem window_isSome {bk : Book} {N H O : ℕ} {cy : Carry} {x : Window} (h : (window bk N H O cy x).isSome) :
    (x.ds.length ≤ N ∧ x.hits.length ≤ H ∧ x.order.length ≤ O ∧ cy.dec.run < 3 ∧ cy.rank ≤ INDEX_CAP + 1) ∧
      ∃ q, syms bk N H O cy x = some q := by
  unfold window at h
  split_ifs at h with hg
  · refine ⟨hg, ?_⟩
    cases hs : syms bk N H O cy x with
    | none =>
      rw [hs] at h
      simp at h
    | some q => exact ⟨q, rfl⟩
  · simp at h

/-- A window weighs, times `2^32`, at most the state its symbols lead to. -/
theorem stepWt_window_le (bk : Book) (N H O : ℕ) (cy : Carry) (x : Window) :
    stepWt (window bk N H O) cy x * 2 ^ 32 ≤ owt (codeAll Tally.fresh (((syms bk N H O cy x).map Prod.fst).getD [])) := by
  unfold stepWt
  cases hw : window bk N H O cy x with
  | none => simpa using owt_nonneg _
  | some r =>
    unfold window at hw
    split_ifs at hw
    simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at hw
    obtain ⟨q, hq, L, ⟨st, hst, hL⟩, rfl⟩ := hw
    simp only [hq, hst, Option.map_some, Option.getD_some, owt_some]
    exact finish_le (good_codeAll _ good_fresh hst) hL

/-- **One window (Kraft)**: over any finite set of windows, the weights `2^(−charge)` of those the window charge
doesn't refuse sum to at most 1, from any carry. -/
theorem sum_window_le (bk : Book) (N H O : ℕ) (cy : Carry) (A : Finset Window) :
    ∑ x ∈ A, stepWt (window bk N H O) cy x ≤ 1 := by
  classical
  let s : Window → List Sym := fun x => ((syms bk N H O cy x).map Prod.fst).getD []
  let B := A.filter fun x => (window bk N H O cy x).isSome
  have h1 : ∑ x ∈ A, stepWt (window bk N H O) cy x = ∑ x ∈ B, stepWt (window bk N H O) cy x := by
    refine (sum_filter_of_ne fun x _ h => ?_).symm
    by_contra hn
    apply h
    simp [stepWt, Option.not_isSome_iff_eq_none.1 hn]
  have h2 : ∑ x ∈ B, owt (codeAll Tally.fresh (s x)) ≤ wt Tally.fresh := by
    apply sum_owt_codeAll_le (B.sup fun x => (s x).length) B s _ good_fresh
    · exact fun x hx => le_sup (f := fun x => (s x).length) hx
    · intro x hx y hy hxy
      obtain ⟨hgx, qx, hqx⟩ := window_isSome (mem_filter.1 hx).2
      obtain ⟨hgy, qy, hqy⟩ := window_isSome (mem_filter.1 hy).2
      have := syms_split bk N H O cy ⟨hgx.1, hgx.2.1, hgx.2.2.1⟩ ⟨hgy.1, hgy.2.1, hgy.2.2.1⟩ hgx.2.2.2.2 hxy hqx hqy
      simpa [s, hqx, hqy] using this
  have h3 : (∑ x ∈ B, stepWt (window bk N H O) cy x) * 2 ^ 32 ≤ 1 * 2 ^ 32 := by
    rw [sum_mul, one_mul]
    calc _ ≤ _ := sum_le_sum fun x _ => stepWt_window_le bk N H O cy x
      _ ≤ wt Tally.fresh := h2
      _ = 2 ^ 32 := wt_fresh
  rw [h1]
  exact le_of_mul_le_mul_right h3 (by positivity)

theorem session_eq (bk : Book) (N H O : ℕ) (cy : Carry) (ws : List Window) :
    session bk N H O cy ws = runOf (window bk N H O) cy ws := by
  induction ws generalizing cy with
  | nil => rfl
  | cons x ws ih => simp only [session, runOf, ih]

/-- **Advice v1's whole-window charge counts (Kraft).** -/
theorem encard_charged_le (bk : Book) (N H O W c : ℕ) :
    {a : Fin W → Window | ∃ ls, charges bk N H O a = some ls ∧ ls.sum ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞) := by
  simpa only [charges, session_eq] using
    encard_runOf_le (window bk N H O) (sum_window_le bk N H O) bk.first W c

theorem sum_bits (bk : Book) (N H O : ℕ) {W : ℕ} (a : Streams bk N H O W) :
    ∑ w, bits bk N H O a w = ((charges bk N H O a.1).getD []).sum := by
  obtain ⟨ls, hls⟩ := Option.isSome_iff_exists.1 a.2
  simp only [bits, hls, Option.getD_some]
  have h : runOf (window bk N H O) bk.first (List.ofFn a.1) = some ls := by
    rw [← session_eq]
    exact hls
  exact sum_getD_eq_sum ls W (by rw [length_runOf _ _ _ h, List.length_ofFn])

/-- **`BitsCount` holds for advice v1's whole-window charge**, under any registered book. -/
theorem bitsCount (bk : Book) (N H O W : ℕ) : BitsCount (bits bk N H O (W := W)) := by
  intro c
  calc {a : Streams bk N H O W | ∑ w, bits bk N H O a w ≤ c}.encard
      = (Subtype.val '' {a : Streams bk N H O W | ∑ w, bits bk N H O a w ≤ c}).encard :=
        (Subtype.val_injective.encard_image _).symm
    _ ≤ {a : Fin W → Window | ∃ ls, charges bk N H O a = some ls ∧ ls.sum ≤ c}.encard := by
        apply Set.encard_le_encard
        rintro _ ⟨a, ha, rfl⟩
        obtain ⟨ls, hls⟩ := Option.isSome_iff_exists.1 a.2
        refine ⟨ls, hls, ?_⟩
        have := sum_bits bk N H O a
        rw [hls, Option.getD_some] at this
        rw [← this]
        exact ha
    _ ≤ _ := encard_charged_le bk N H O W c

end NetTiming.WindowCharge

namespace NetTiming

variable {L K S : ℕ} {In : Type}

/-- **The leak bound with advice v1's whole-window charge**: `encard_accSeqsWithin_le_constantRate`, its `BitsCount`
premise discharged by `WindowCharge.bitsCount`. -/
theorem encard_accSeqsWithin_le_constantRate_v1Whole {W : ℕ} (bk : WindowCharge.Book) (N H O : ℕ) (b : Fin W → ℕ)
    (T : ℕ) (ss : SyncSet) (frames : List In → WindowCharge.Streams bk N H O W → Fin L → List LogicalFrame)
    (op : ℕ → List (Obs L K S) → In) (n D m : ℕ) (hm : n ≤ m * T) :
    (AccSeqsWithin (WindowCharge.bits bk N H O) b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) :=
  encard_accSeqsWithin_le_constantRate (WindowCharge.bits bk N H O) (WindowCharge.bitsCount bk N H O W) b T ss frames
    op n D m hm

variable {Z Act Wire : Type}

/-- **The leak bound through the wire, with advice v1's whole-window charge**: `encard_decodableWithin_le_compiled`,
its `BitsCount` premise discharged by `WindowCharge.bitsCount`. -/
theorem encard_decodableWithin_le_compiled_v1Whole {Ω : Type} {W : ℕ} (I : Interaction L K S Z In Act Wire) (T : ℕ)
    (ss : SyncSet) (bk : WindowCharge.Book) (N H O : ℕ) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (WindowCharge.Streams bk N H O W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin (WindowCharge.bits bk N H O) b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n
        (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) :=
  encard_decodableWithin_le_compiled I T ss (WindowCharge.bits bk N H O) (WindowCharge.bitsCount bk N H O W) b
    compilerOf env hpre hdet n D m hm view hW2 hJ Zs hacc dec hdec

/-- **The leak bound per window, with advice v1's whole-window charge**:
`encard_decodableWithin_le_compiled_perWindow`, its `BitsCount` premise discharged by `WindowCharge.bitsCount`. -/
theorem encard_decodableWithin_le_compiled_perWindow_v1Whole {Ω : Type} {W : ℕ}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet) (bk : WindowCharge.Book) (N H O : ℕ) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (WindowCharge.Streams bk N H O W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin (WindowCharge.bits bk N H O) b
      (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T) (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z) :
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞) :=
  encard_decodableWithin_le_compiled_perWindow I T ss (WindowCharge.bits bk N H O)
    (WindowCharge.bitsCount bk N H O W) b compilerOf env hpre hdet D view hW2 hJ Zs hacc dec hdec

end NetTiming
