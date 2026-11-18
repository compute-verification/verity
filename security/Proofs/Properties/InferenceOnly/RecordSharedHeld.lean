module

public import Proofs.Properties.InferenceOnly.RecordShared
public import Proofs.Properties.InferenceOnly.RecordUnits

/-!
# Shared runs held: the proof

`SharedHeld`, from the check's `sharedHeld` stage. The root's ranges start where the ones before them end
(`evaluate_chained`), and their nodes are distinct (`evaluate_node_inj`), so a unit's index among its template's,
which `Properties.Record.UnitAt` counts over the ranges before its own, is the one `sharedPlan` counts over the
ranges that start before its own (`u0_eq`). Two runs `runsOf` admits are one run or share no leaf
(`runsOf_eqOrApart`): sorted, a run and the one after it are, so every two are (`before_of_sorted`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Flock.TemplateQuery (Desc TRange)

/-- The ranges `l` start at `base`, each where the one before it ends. -/
def Chained (base : Nat) (l : List TRange) : Prop :=
  ∀ j (h : j < l.length), l[j].first = base + ((l.take j).map (·.count)).sum

theorem chained_cons {base : Nat} {q : TRange} {l : List TRange} (h : Chained base (q :: l)) :
    q.first = base ∧ Chained (base + q.count) l := by
  have h0 := h 0 (by simp)
  rw [List.getElem_cons_zero] at h0
  refine ⟨by simpa using h0, fun j hj => ?_⟩
  have := h (j + 1) (by simp; omega)
  simp only [List.getElem_cons_succ, List.take_succ_cons, List.map_cons, List.sum_cons] at this
  rw [this]
  omega

theorem chained_ge {base : Nat} : ∀ {l : List TRange}, Chained base l → ∀ q ∈ l, base ≤ q.first
  | [], _, q, hq => by simp at hq
  | _ :: _, h, q, hq => by
    obtain ⟨h1, h2⟩ := chained_cons h
    rcases List.mem_cons.1 hq with rfl | hq
    · omega
    · have := chained_ge h2 q hq
      omega

theorem chained_push {base : Nat} {l : List TRange} {x : TRange} (h : Chained base l)
    (hx : x.first = base + (l.map (·.count)).sum) : Chained base (l ++ [x]) := by
  intro j hj
  rw [List.length_append, List.length_singleton] at hj
  rcases Nat.lt_or_ge j l.length with hj' | hj'
  · rw [List.getElem_append_left hj', List.take_append_of_le_length (Nat.le_of_lt hj')]
    exact h j hj'
  · have hjl : j = l.length := by omega
    subst hjl
    rw [List.getElem_append_right (Nat.le_refl _), List.take_left' rfl]
    simpa using hx

/-- **The root's ranges start at 0, each where the one before it ends.** -/
theorem evaluate_chained {d : Desc} {ts : Array String} {rs : Array TRange}
    (h : Flock.TemplateQuery.evaluate d ts = .ok rs) : Chained 0 rs.toList := by
  unfold Flock.TemplateQuery.evaluate at h
  cases hroot : d.def? d.root with
  | none =>
    simp only [hroot] at h
    cases h
  | some root =>
    simp only [hroot] at h
    obtain ⟨r, hloop, h⟩ := Json.bind_ok h
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
    have hinv := forIn_done
      (fun _ (s : Array TRange × Nat) => Chained 0 s.1.toList ∧ s.2 = (s.1.toList.map (·.count)).sum)
      ?_ _ [] (#[], 0) r ⟨fun j hj => absurd hj (by simp), rfl⟩ hloop
    · have hrs : rs = r.1 := by
        simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
        split at h
        · cases h
        · cases h
          rfl
      rw [hrs]
      exact hinv.1
    · clear hloop
      intro k s x done hi hx
      simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at hx
      split at hx
      · cases hx
        exact ⟨_, rfl, hi⟩
      · split at hx
        · cases hx
        · cases hx
          refine ⟨_, rfl, ?_, ?_⟩
          · rw [Array.toList_push]
            exact chained_push hi.1 (by rw [Nat.zero_add]; exact hi.2)
          · rw [Array.toList_push, List.map_append, List.sum_append, ← hi.2]
            simp

/-- **The root's ranges are of distinct nodes.** -/
theorem evaluate_node_inj {d : Desc} {ts : Array String} {rs : Array TRange}
    (h : Flock.TemplateQuery.evaluate d ts = .ok rs) {i i' : Nat} (hi : i < rs.size) (hi' : i' < rs.size)
    (he : rs[i].node = rs[i'].node) : i = i' := by
  obtain ⟨root, -, hnodes, -⟩ := evaluate_spec h
  have hnd : (rs.toList.map (·.node)).Nodup := by
    rw [hnodes]
    exact List.Nodup.filter _ List.nodup_range
  have hl : i < (rs.toList.map (·.node)).length := by simpa using hi
  have hl' : i' < (rs.toList.map (·.node)).length := by simpa using hi'
  exact (List.Nodup.getElem_inj_iff hnd (hi := hl) (hj := hl')).1 (by simpa using he)

theorem foldl_below_none (F : Nat) : ∀ (m : List TRange) (a : Nat), (∀ q ∈ m, F ≤ q.first) →
    m.foldl (fun a q => if q.first < F then a + q.count else a) a = a
  | [], _, _ => rfl
  | q :: m, a, h => by
    have hn := Nat.not_lt.2 (h q List.mem_cons_self)
    rw [List.foldl_cons]
    simp only [hn, ↓reduceIte]
    exact foldl_below_none F m a fun q' hq' => h q' (List.mem_cons_of_mem _ hq')

/-- Over chained ranges, the counts of the ranges of `T` that start before range `i` are the counts of those before
it: a range before `i` that starts where `i` does has no units. -/
theorem below_eq (T : TRange → Bool) : ∀ {base : Nat} {l : List TRange}, Chained base l →
    ∀ i (hi : i < l.length) (a : Nat),
      (l.filter T).foldl (fun a q => if q.first < l[i].first then a + q.count else a) a =
        ((l.take i).filter T).foldl (fun a q => a + q.count) a
  | _, [], _, i, hi, _ => absurd hi (by simp)
  | _, _ :: _, h, 0, _, a => by
    obtain ⟨h1, -⟩ := chained_cons h
    rw [List.take_zero, List.filter_nil, List.foldl_nil]
    apply foldl_below_none
    intro q' hq'
    rw [List.getElem_cons_zero, h1]
    exact chained_ge h q' (List.mem_filter.1 hq').1
  | _, q :: l, h, i + 1, hi, a => by
    obtain ⟨h1, h2⟩ := chained_cons h
    have hi' : i < l.length := by simpa using hi
    have hF := chained_ge h2 _ (List.getElem_mem hi')
    have ih := below_eq T h2 i hi'
    rw [List.getElem_cons_succ, List.take_succ_cons, List.filter_cons, List.filter_cons]
    cases hT : T q
    · simp only [Bool.false_eq_true, ↓reduceIte]
      exact ih a
    · simp only [↓reduceIte, List.foldl_cons]
      have : (if q.first < l[i].first then a + q.count else a) = a + q.count := by
        split <;> omega
      rw [this]
      exact ih (a + q.count)

/-- **A unit's index among its template's, as `sharedPlan` counts it**: the counts of the template's ranges that
start before the unit's range are those of the ranges before it (`Properties.Record.UnitAt`). -/
theorem u0_eq {rs : Array TRange} (hc : Chained 0 rs.toList) {i : Nat} (hi : i < rs.size) (tid : String) :
    (Spot.RecordText.rangesAt rs tid).foldl (fun a q => if q.first < rs[i].first then a + q.count else a) 0 =
      ((rs.extract 0 i).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 := by
  unfold Spot.RecordText.rangesAt
  rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_filter, Array.toList_filter, Array.toList_extract,
    List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero]
  have hl : i < rs.toList.length := by simpa using hi
  have := below_eq (fun r => r.template == tid) hc i hl 0
  rw [Array.getElem_toList] at this
  exact this

open Flock.Program (Prog Part)

/-- The order `runsOf` sorts runs in: by node, then by first leaf. -/
def runOrd (a b : (Nat × Nat × Nat) × Bool) : Bool := a.1.1 < b.1.1 || (a.1.1 == b.1.1 && a.1.2.1 ≤ b.1.2.1)

theorem runOrd_trans (a b c : (Nat × Nat × Nat) × Bool) (hab : runOrd a b = true) (hbc : runOrd b c = true) :
    runOrd a c = true := by
  simp only [runOrd, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hab hbc ⊢
  omega

theorem runOrd_total (a b : (Nat × Nat × Nat) × Bool) : (runOrd a b || runOrd b a) = true := by
  simp only [runOrd, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  omega

/-- Run `a` is run `b`, of another node, or ends where `b` starts or before. -/
def Before (a b : (Nat × Nat × Nat) × Bool) : Prop :=
  a.1 = b.1 ∨ a.1.1 ≠ b.1.1 ∨ a.1.2.1 + a.1.2.2 ≤ b.1.2.1

/-- Sorted by node and first leaf, runs each before the next are each before every later one. -/
theorem before_of_sorted : ∀ (l : List ((Nat × Nat × Nat) × Bool)), l.Pairwise (fun a b => runOrd a b = true) →
    ((l.zip l.tail).all fun (a, b) => a.1 == b.1 || a.1.1 != b.1.1 || a.1.2.1 + a.1.2.2 ≤ b.1.2.1) = true →
    l.Pairwise Before
  | [], _, _ => List.Pairwise.nil
  | [_], _, _ => List.pairwise_singleton _ _
  | x :: y :: rest, hs, ha => by
    rw [List.pairwise_cons] at hs
    obtain ⟨hx, hs'⟩ := hs
    simp only [List.tail_cons, List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at ha
    obtain ⟨hxy, ha'⟩ := ha
    have ih := before_of_sorted (y :: rest) hs' (by simpa using ha')
    refine List.pairwise_cons.2 ⟨fun z hz => ?_, ih⟩
    have hxy' : Before x y := by simpa [Before, or_assoc] using hxy
    rcases List.mem_cons.1 hz with rfl | hz'
    · exact hxy'
    · have hyz : Before y z := (List.pairwise_cons.1 ih).1 z hz'
      have hlxy := hx y List.mem_cons_self
      have hlyz := (List.pairwise_cons.1 hs').1 z hz'
      simp only [runOrd, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hlxy hlyz
      unfold Before at hxy' hyz ⊢
      rcases hxy' with hxy' | hxy'
      · rw [hxy']
        exact hyz
      · rcases hyz with hyz | hyz
        · rw [← hyz]
          exact Or.inr hxy'
        · right
          omega

/-- Two distinct elements of a list pairwise in a symmetric relation are in it. -/
theorem pairwise_mem {α : Type} {R : α → α → Prop} (hs : ∀ x y, R x y → R y x) :
    ∀ {l : List α}, l.Pairwise R → ∀ x ∈ l, ∀ y ∈ l, x ≠ y → R x y
  | [], _, x, hx, _, _, _ => by cases hx
  | _ :: _, h, x, hx, y, hy, hne => by
    rw [List.pairwise_cons] at h
    rcases List.mem_cons.1 hx with rfl | hx' <;> rcases List.mem_cons.1 hy with rfl | hy'
    · exact absurd rfl hne
    · exact h.1 y hy'
    · exact hs _ _ (h.1 x hx')
    · exact pairwise_mem hs h.2 x hx' y hy' hne

/-- **Two runs `runsOf` admits, each of some leaf, are one run or share no leaf.** -/
theorem runsOf_eqOrApart {P : Prog} {parts : List Part} {R : Runs} (h : runsOf P parts = some R)
    {q q' : Part} (hq : q ∈ parts) (hq' : q' ∈ parts) {a b : Nat × Nat × Nat} {ba bb : Bool}
    (ha : partRun P q = some (a, ba)) (hb : partRun P q' = some (b, bb)) (h0 : 0 < a.2.2) (h0' : 0 < b.2.2) :
    a = b ∨ a.1 ≠ b.1 ∨ a.2.1 + a.2.2 ≤ b.2.1 ∨ b.2.1 + b.2.2 ≤ a.2.1 := by
  unfold runsOf at h
  dsimp only at h
  split at h
  · rename_i hc
    rw [Bool.and_eq_true] at hc
    have hsorted := before_of_sorted _ (List.pairwise_mergeSort (le := runOrd) runOrd_trans runOrd_total _) hc.2
    have hmem : ∀ (x : Part) (r : Nat × Nat × Nat) (br : Bool), x ∈ parts → partRun P x = some (r, br) →
        0 < r.2.2 → (r, br) ∈ ((parts.filterMap (partRun P)).filter fun r => r.1.2.2 > 0).mergeSort runOrd := by
      intro x r br hx hr hr0
      rw [List.mem_mergeSort]
      exact List.mem_filter.2 ⟨List.mem_filterMap.2 ⟨x, hx, hr⟩, by simpa using hr0⟩
    by_cases hab : (a, ba) = (b, bb)
    · exact Or.inl (congrArg Prod.fst hab)
    · let E := fun (x y : (Nat × Nat × Nat) × Bool) =>
        x.1 = y.1 ∨ x.1.1 ≠ y.1.1 ∨ x.1.2.1 + x.1.2.2 ≤ y.1.2.1 ∨ y.1.2.1 + y.1.2.2 ≤ x.1.2.1
      have hR : ∀ x y, Before x y → E x y := by
        intro x y hxy
        unfold Before at hxy
        rcases hxy with hxy | hxy
        · exact Or.inl hxy
        · right
          omega
      have hsym : ∀ x y, E x y → E y x := by
        intro x y hxy
        rcases hxy with hxy | hxy
        · exact Or.inl hxy.symm
        · right
          omega
      exact pairwise_mem hsym (hsorted.imp fun {x y} => hR x y) _ (hmem q a ba hq ha h0) _
        (hmem q' b bb hq' hb h0') hab
  · cases h

end InferenceOnlyVerifier.Record.SecurityProofs
