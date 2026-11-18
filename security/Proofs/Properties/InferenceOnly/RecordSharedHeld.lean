module

public import Proofs.Properties.InferenceOnly.RecordShared
public import Proofs.Properties.InferenceOnly.RecordUnits
public import Proofs.Properties.InferenceOnly.RecordDeclared

/-!
# Shared runs held: the proof

`SharedHeld`, from the check's `sharedHeld` stage. The root's ranges start where the ones before them end
(`evaluate_chained`), and their nodes are distinct (`evaluate_node_inj`), so a unit's index among its template's,
which `Properties.Record.UnitAt` counts over the ranges before its own, is the one `sharedPlan` counts over the
ranges that start before its own (`u0_eq`). Two runs `runsOf` admits are one run or share no leaf
(`runsOf_eqOrApart`): sorted, a run and the one after it are, so every two are (`before_of_sorted`).

A record `read` accepts passed the stage (`finish_shared`). Where `heldFrom` holds a row from unit `u` to `e`, a
declaration that reads `p`, θ, covers the row at each of those units (`heldFrom_cover`), so, no two declarations
covering one row, it is what fills the row (`RecordDeclared.fillAt_of_covers`), and the row is held as that read
(`heldAt_read`).
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

open Spot.RecordText (Row Decl Fill inputsOf fillAt)

/-- **A record `read` accepts passed the `shared-unheld` stage**, over the declarations and the θ it holds. -/
theorem finish_shared {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) :
    c.f = f ∧ inputsOf f.ts c.W f.doc = .ok c.x ∧ sharedHeld f.P f.ts c.W c.x.decls c.isTheta = true := by
  unfold finish at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · cases h
  · split at h
    · split at h
      · cases h
      · split at h
        · cases h
        · rename_i x hx
          split at h
          · cases h
          · rename_i hs
            split at h
            · cases h
            split at h
            · cases h
            · split at h
              · cases h
              · cases h
                refine ⟨rfl, hx, ?_⟩
                show sharedHeld f.P f.ts _ x.decls (fun p => ((thetaOf f.T f.doc x).lookup p).getD false) = true
                simpa using hs
    · cases h

theorem read_shared {j tr : Lean.Json} {c : Case} (h : read j tr = .ok c) :
    inputsOf c.f.ts c.W c.f.doc = .ok c.x ∧ sharedHeld c.f.P c.f.ts c.W c.x.decls c.isTheta = true := by
  obtain ⟨f, pin, -, hc⟩ := read_front h
  obtain ⟨hf, hx, hs⟩ := finish_shared hc
  rw [hf]
  exact ⟨hx, hs⟩

/-- **A row filled by a registered read of `p`, θ, is held as that read.** -/
theorem heldAt_read {c : Case} {r : Row} {p : String} (hf : fillAt c.x.decls r = some (.read p))
    (ht : c.isTheta p = true) : c.heldAt r = some (.read p) := by
  simp [Case.heldAt, hf, Spot.RecordText.Inputs.declares, ht]

theorem cover_bounds {w : Circuits.Wire} {tid port : String} {u c : Nat}
    (hu : (w.indexOf tid u port c).isSome = true) : w.start ≤ u ∧ u < w.start + w.readerUnits := by
  unfold Circuits.Wire.indexOf at hu
  split at hu
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    exact ⟨hc.1.1.1.2, hc.1.1.2⟩
  · simp at hu

/-- A wire that covers a row at reader unit `u` covers it at every unit it reads. -/
theorem cover_at {w : Circuits.Wire} {tid port : String} {u v c : Nat}
    (hu : (w.indexOf tid u port c).isSome = true) (h1 : w.start ≤ v) (h2 : v < w.start + w.readerUnits) :
    (w.indexOf tid v port c).isSome = true := by
  unfold Circuits.Wire.indexOf at hu ⊢
  split at hu
  · rename_i hc
    split
    · rfl
    · rename_i hn
      apply absurd _ hn
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc ⊢
      exact ⟨⟨⟨⟨⟨hc.1.1.1.1.1, hc.1.1.1.1.2⟩, h1⟩, h2⟩, hc.1.2⟩, hc.2⟩
  · simp at hu

/-- **Where `heldFrom` holds a row, a registered read of `p`, θ, covers it at every unit it visits.** -/
theorem heldFrom_cover {ds : List Decl} {theta : String → Bool} {tid port : String} {c u0 e : Nat} {p : String}
    {g : List (Nat × Int)} {gb : Int} : ∀ fuel u, heldFrom ds theta tid port c u0 e p g gb fuel u = some true →
      ∀ v, u ≤ v → v < e → theta p = true ∧ ∃ d ∈ ds, (d.w.indexOf tid v port c).isSome = true ∧ d.fill = .read p := by
  intro fuel
  induction fuel with
  | zero =>
    intro u h v huv hve
    unfold heldFrom at h
    split at h
    · omega
    · cases h
  | succ fuel ih =>
    intro u h v huv hve
    unfold heldFrom at h
    split at h
    · omega
    · simp only at h
      split at h
      · rename_i d hd
        split at h
        · rename_i q hq
          split at h
          · cases h
          · rename_i hok
            simp only [Bool.not_eq_true', Bool.and_eq_false_iff, beq_iff_eq, decide_eq_true_eq, not_or,
              Bool.not_eq_false] at hok
            split at h
            · cases h
            · split at h
              · have hc := List.find?_some hd
                have hb := cover_bounds hc
                by_cases hv : v < min e (d.w.start + d.w.readerUnits)
                · refine ⟨?_, d, List.mem_of_find?_eq_some hd, cover_at hc (by omega) (by omega), ?_⟩
                  · simp_all
                  · rw [hq]
                    simp_all
                · exact ih _ h v (by omega) hve
              · simp_all
        · cases h
      · cases h

/-- The rows `sharedDecide` checks of the shared argument `s`: its port's cuts, 1 where it finds no port. -/
def sharedCuts (P : Prog) (rs : Array TRange) (ts : List Spot.RecordText.Tmpl) (W : Spot.Wires.Wiring) (s : Shared) :
    Nat :=
  ((rs.find? (·.node == s.node)).bind fun r => (ts.find? (·.tid == r.template)).bind fun t =>
    ((P.defs[t.defn]!).params[s.arg]?).map fun (port, _) => W.cutsOf r.template port).getD 1

/-- **What the stage decides**: where some run is shared, every row of every argument that reads one is held
(`sharedRow`), and two rows' marks of one input node are one. -/
theorem decide_rows {P : Prog} {ts : List Spot.RecordText.Tmpl} {W : Spot.Wires.Wiring} {ds : List Decl}
    {theta : String → Bool} (h : sharedDecide P ts W ds theta = some true) (hne : (sharedRuns P).isEmpty = false) :
    ∃ sh rs, sharedOf P = some sh ∧
      Flock.TemplateQuery.evaluate (Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs ∧
      ∀ s ∈ sh, ∀ c < sharedCuts P rs ts W s, ∃ m, sharedRow P rs ts W ds theta s c = some (some m) ∧
        ∀ s' ∈ sh, ∀ c' < sharedCuts P rs ts W s', ∀ m', sharedRow P rs ts W ds theta s' c' = some (some m') →
          m.1 = m'.1 → m = m' := by
  unfold sharedDecide at h
  simp only [hne, Bool.false_eq_true, ↓reduceIte] at h
  split at h
  · simp at h
  · simp at h
  · rename_i sh rs hsh hrs
    split at h
    · simp at h
    split at h
    · simp at h
    rename_i h2 h1
    have hall := Option.some.inj h
    have hmem : ∀ s ∈ sh, ∀ c < sharedCuts P rs ts W s, ∃ m, sharedRow P rs ts W ds theta s c = some (some m) := by
      intro s hs c hc
      match hr : sharedRow P rs ts W ds theta s c with
      | none =>
        exact absurd (List.any_eq_true.2 ⟨sharedRow P rs ts W ds theta s c, List.mem_map.2 ⟨(s, c),
          List.mem_flatMap.2 ⟨s, hs, List.mem_map.2 ⟨c, List.mem_range.2 hc, rfl⟩⟩, rfl⟩, by rw [hr]; rfl⟩) h1
      | some none =>
        exact absurd (List.any_eq_true.2 ⟨sharedRow P rs ts W ds theta s c, List.mem_map.2 ⟨(s, c),
          List.mem_flatMap.2 ⟨s, hs, List.mem_map.2 ⟨c, List.mem_range.2 hc, rfl⟩⟩, rfl⟩, by rw [hr]; rfl⟩) h2
      | some (some m) => exact ⟨m, rfl⟩
    refine ⟨sh, rs, hsh, hrs, fun s hs c hc => ?_⟩
    obtain ⟨m, hm⟩ := hmem s hs c hc
    refine ⟨m, hm, fun s' hs' c' hc' m' hm' he => ?_⟩
    have h1' := List.all_eq_true.1 hall m (List.mem_filterMap.2 ⟨sharedRow P rs ts W ds theta s c,
      List.mem_map.2 ⟨(s, c), List.mem_flatMap.2 ⟨s, hs, List.mem_map.2 ⟨c, List.mem_range.2 hc, rfl⟩⟩, rfl⟩,
      by rw [hm]; rfl⟩)
    have := List.all_eq_true.1 h1' m' (List.mem_filterMap.2 ⟨sharedRow P rs ts W ds theta s' c',
      List.mem_map.2 ⟨(s', c'), List.mem_flatMap.2 ⟨s', hs', List.mem_map.2 ⟨c', List.mem_range.2 hc', rfl⟩⟩, rfl⟩,
      by rw [hm']; rfl⟩)
    simpa [he] using this

/-- A step of `sharedOf`'s fold. -/
def sharedStep (P : Prog) (sh : List (Nat × Nat × Nat)) (reads : Part → Bool) (mapped : Nat → Nat → Nat → Bool) :
    Nat × Nat × List Part → Option (List Shared) → Option (List Shared) :=
  fun (k, j, ps) acc => acc.bind fun out => match ps with
    | [q] => match partRun P q with
      | some ((i, base, inner), _) =>
        if !sh.contains (i, base, inner) then some out
        else if mapped k j q.count then
          some ({ node := k, arg := j, input := i, base, inner, count := q.count } :: out)
        else none
      | none => some out
    | _ => if ps.any reads then none else some out

/-- `sharedOf`'s fold: an argument with a part over a shared run is that part alone, and is listed. -/
theorem shared_fold (P : Prog) (sh : List (Nat × Nat × Nat)) (reads : Part → Bool)
    (mapped : Nat → Nat → Nat → Bool) : ∀ (L : List (Nat × Nat × List Part)) (out : List Shared),
    L.foldr (sharedStep P sh reads mapped) (some []) = some out →
    ∀ k j ps, (k, j, ps) ∈ L → ∀ q ∈ ps, ∀ i base inner b, partRun P q = some ((i, base, inner), b) →
      sh.contains (i, base, inner) = true → reads q = true →
      ps = [q] ∧ ∃ s ∈ out, s.node = k ∧ s.arg = j ∧ s.input = i ∧ s.base = base ∧ s.inner = inner ∧
        s.count = q.count
  | [], _, _, _, _, _, hm, _, _, _, _, _, _, _, _, _ => by cases hm
  | (k0, j0, ps0) :: L, out, h, k, j, ps, hm, q, hq, i, base, inner, b, hr, hc, hrd => by
    rw [List.foldr_cons] at h
    cases hacc : L.foldr (sharedStep P sh reads mapped) (some []) with
    | none =>
      rw [hacc] at h
      cases h
    | some out' =>
      rw [hacc] at h
      simp only [sharedStep, Option.bind_some] at h
      have ih := shared_fold P sh reads mapped L out' hacc k j ps
      have hlift : ∀ (ss : List Shared), (∀ s ∈ out', s ∈ ss) →
          (ps = [q] ∧ ∃ s ∈ out', s.node = k ∧ s.arg = j ∧ s.input = i ∧ s.base = base ∧ s.inner = inner ∧
            s.count = q.count) →
          ps = [q] ∧ ∃ s ∈ ss, s.node = k ∧ s.arg = j ∧ s.input = i ∧ s.base = base ∧ s.inner = inner ∧
            s.count = q.count := fun ss hss ⟨h1, s, hs, h2⟩ => ⟨h1, s, hss s hs, h2⟩
      split at h
      · rename_i q'
        split at h
        · rename_i i' base' inner' b' hpr
          split at h
          · rename_i hnc
            cases h
            rcases List.mem_cons.1 hm with heq | hm'
            · cases heq
              rw [List.mem_singleton] at hq
              subst hq
              rw [hr] at hpr
              cases hpr
              simp only [List.contains_iff_mem] at hc
              simp at hnc
              exact absurd hc hnc
            · exact ih hm' q hq i base inner b hr hc hrd
          · split at h
            · cases h
              rcases List.mem_cons.1 hm with heq | hm'
              · cases heq
                rw [List.mem_singleton] at hq
                subst hq
                rw [hr] at hpr
                cases hpr
                exact ⟨rfl, _, List.mem_cons_self, rfl, rfl, rfl, rfl, rfl, rfl⟩
              · exact hlift _ (fun s hs => List.mem_cons_of_mem _ hs) (ih hm' q hq i base inner b hr hc hrd)
            · cases h
        · rename_i hpr
          cases h
          rcases List.mem_cons.1 hm with heq | hm'
          · cases heq
            rw [List.mem_singleton] at hq
            subst hq
            rw [hr] at hpr
            cases hpr
          · exact ih hm' q hq i base inner b hr hc hrd
      · split at h
        · cases h
        · rename_i hany
          cases h
          rcases List.mem_cons.1 hm with heq | hm'
          · cases heq
            exact absurd (List.any_eq_true.2 ⟨q, hq, hrd⟩) hany
          · exact ih hm' q hq i base inner b hr hc hrd

/-- **An argument of a root node with a part over a shared run is that part alone, and is one of `sharedOf`'s.** -/
theorem sharedOf_arg {P : Prog} {sh : List Shared} (h : sharedOf P = some sh) {k j : Nat}
    (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : InferenceOnlyVerifier.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hj : j < ((P.defs[P.root]!).body.nodes[k]!).args.size) {q : Part}
    (hq : q ∈ (((P.defs[P.root]!).body.nodes[k]!).args[j]!).seq.parts.toList) {i base inner : Nat} {b : Bool}
    (hr : partRun P q = some ((i, base, inner), b)) (hc : (sharedRuns P).contains (i, base, inner) = true) :
    (((P.defs[P.root]!).body.nodes[k]!).args[j]!).seq.parts.toList = [q] ∧
      ∃ s ∈ sh, s.node = k ∧ s.arg = j ∧ s.input = i ∧ s.base = base ∧ s.inner = inner ∧ s.count = q.count := by
  have h' : ((P.defs[P.root]!).body.nodes.toList.zipIdx.flatMap fun (nd, k) =>
      if InferenceOnlyVerifier.Desc.isInputNode P nd then [] else
        nd.args.toList.zipIdx.map fun (a, j) => (k, j, a.seq.parts.toList)).foldr
      (sharedStep P (sharedRuns P) (fun q => ((partRun P q).map ((sharedRuns P).contains ·.1)).getD false)
        (fun (k j c : Nat) => (((P.defs[P.root]!).body.nodes[k]?).map fun nd =>
          nd.form != .batch || (nd.args[j]?.bind (·.dims)).map (·.2) == some (c / nd.n)).getD false)) (some []) =
      some sh := h
  refine shared_fold P _ _ _ _ _ h' k j _ ?_ q hq i base inner b hr hc ?_
  · refine List.mem_flatMap.2 ⟨((P.defs[P.root]!).body.nodes[k]!, k),
      List.mem_zipIdx_iff_getElem?.2 (by simp [getElem!_pos (P.defs[P.root]!).body.nodes k hk, hk]), ?_⟩
    simp only [hu, Bool.false_eq_true, ↓reduceIte]
    exact List.mem_map.2 ⟨((((P.defs[P.root]!).body.nodes[k]!).args[j]!), j),
      List.mem_zipIdx_iff_getElem?.2 (by simp [getElem!_pos ((P.defs[P.root]!).body.nodes[k]!).args j hj, hj]), rfl⟩
  · show ((partRun P q).map ((sharedRuns P).contains ·.1)).getD false = true
    rw [hr]
    exact hc

end InferenceOnlyVerifier.Record.SecurityProofs
