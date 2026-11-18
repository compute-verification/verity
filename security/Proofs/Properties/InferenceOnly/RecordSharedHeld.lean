module

public import Proofs.Properties.InferenceOnly.RecordShared
public import Proofs.Properties.InferenceOnly.RecordUnits
public import Proofs.Properties.InferenceOnly.RecordDeclared
public import Proofs.Properties.InferenceOnly.RecordTemplates
public import Proofs.Properties.InferenceOnly.RecordLeafWide

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
    c.f = f ∧ inputsOf f.ts c.W f.doc = .ok c.x ∧ sharedHeld f.P f.ts c.W c.x.decls c.x.tables c.isTheta = true := by
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
                show sharedHeld f.P f.ts _ x.decls x.tables (fun p => ((thetaOf f.T f.doc x).lookup p).getD false) = true
                simpa using hs
    · cases h

theorem read_shared {j tr : Lean.Json} {c : Case} (h : read j tr = .ok c) :
    inputsOf c.f.ts c.W c.f.doc = .ok c.x ∧ sharedHeld c.f.P c.f.ts c.W c.x.decls c.x.tables c.isTheta = true := by
  obtain ⟨f, pin, -, hc⟩ := read_front h
  obtain ⟨hf, hx, hs⟩ := finish_shared hc
  rw [hf]
  exact ⟨hx, hs⟩

/-- **A row filled by a registered read of `p`, θ, is held as that read.** -/
theorem heldAt_read {c : Case} {r : Row} {p : String} (hf : fillAt c.x.decls r = some (.read p))
    (ht : c.isTheta p = true) : c.heldAt r = some (.read p) := by
  simp [Case.heldAt, hf, Spot.RecordText.Inputs.declares, ht]

/-- The reason a shared run's source holds a row as: its registered read, or its table read, of the table's
parameter. -/
def heldOf (c : Case) : Source → Spot.RecordText.Held
  | .read p _ => .read p
  | .table name root _ => .table name (((c.x.tables.find? (·.name == name)).map (·.param)).getD "") root

theorem sharesRun_heldOf (c : Case) : ∀ src, Properties.Record.SharesRun (heldOf c src)
  | .read _ _ => trivial
  | .table _ _ _ => trivial

/-- What a shared run's source says of a row it holds: what fills it, θ. -/
def SourceFill (c : Case) (r : Row) : Source → Prop
  | .read p _ => fillAt c.x.decls r = some (.read p) ∧ c.isTheta p = true
  | .table name root _ => fillAt c.x.decls r = some (.table name root) ∧ tableTheta c.x.tables c.isTheta name = true

/-- **A row its source fills is held as the source's reason.** -/
theorem heldAt_of_fill {c : Case} {r : Row} : ∀ {src : Source}, SourceFill c r src → c.heldAt r = some (heldOf c src)
  | .read _ _, ⟨hf, ht⟩ => heldAt_read hf ht
  | .table name root _, ⟨hf, ht⟩ => by
    unfold tableTheta at ht
    cases hx : c.x.tables.find? (·.name == name) with
    | none => simp [hx] at ht
    | some t =>
      simp only [hx, Option.map_some, Option.getD_some] at ht
      simp [heldOf, Case.heldAt, hf, Spot.RecordText.Inputs.declares, hx, ht]

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

/-- **Where `tableFrom` holds a row, a table read of `name`'s root (`root`), else its domain id, the table θ, covers
it at every unit it visits.** -/
theorem tableFrom_cover {ds : List Decl} {tables : List Spot.RecordText.Table} {theta : String → Bool}
    {tid port : String} {c e : Nat} {name : String} {root : Bool} : ∀ fuel u,
      tableFrom ds tables theta tid port c e name root fuel u = some true →
      ∀ v, u ≤ v → v < e → tableTheta tables theta name = true ∧
        ∃ d ∈ ds, (d.w.indexOf tid v port c).isSome = true ∧ d.fill = .table name root := by
  intro fuel
  induction fuel with
  | zero =>
    intro u h v huv hve
    unfold tableFrom at h
    split at h
    · omega
    · cases h
  | succ fuel ih =>
    intro u h v huv hve
    unfold tableFrom at h
    split at h
    · omega
    · simp only at h
      split at h
      · rename_i d hd
        split at h
        · rename_i name' root' hq
          split at h
          · rename_i hok
            simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hok
            obtain ⟨⟨⟨h1, h2⟩, hth⟩, hlt⟩ := hok
            subst h1
            subst h2
            have hc := List.find?_some hd
            have hb := cover_bounds hc
            by_cases hv : v < min e (d.w.start + d.w.readerUnits)
            · exact ⟨hth, d, List.mem_of_find?_eq_some hd, cover_at hc (by omega) (by omega), hq⟩
            · exact ih _ h v (by omega) hve
          · cases h
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
    {tables : List Spot.RecordText.Table} {theta : String → Bool} (h : sharedDecide P ts W ds tables theta = some true) (hne : (sharedRuns P).isEmpty = false) :
    ∃ sh rs, sharedOf P = some sh ∧
      Flock.TemplateQuery.evaluate (Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs ∧
      ∀ s ∈ sh, ∀ c < sharedCuts P rs ts W s, ∃ m, sharedRow P rs ts W ds tables theta s c = some (some m) ∧
        ∀ s' ∈ sh, ∀ c' < sharedCuts P rs ts W s', ∀ m', sharedRow P rs ts W ds tables theta s' c' = some (some m') →
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
    have hmem : ∀ s ∈ sh, ∀ c < sharedCuts P rs ts W s, ∃ m, sharedRow P rs ts W ds tables theta s c = some (some m) := by
      intro s hs c hc
      match hr : sharedRow P rs ts W ds tables theta s c with
      | none =>
        exact absurd (List.any_eq_true.2 ⟨sharedRow P rs ts W ds tables theta s c, List.mem_map.2 ⟨(s, c),
          List.mem_flatMap.2 ⟨s, hs, List.mem_map.2 ⟨c, List.mem_range.2 hc, rfl⟩⟩, rfl⟩, by rw [hr]; rfl⟩) h1
      | some none =>
        exact absurd (List.any_eq_true.2 ⟨sharedRow P rs ts W ds tables theta s c, List.mem_map.2 ⟨(s, c),
          List.mem_flatMap.2 ⟨s, hs, List.mem_map.2 ⟨c, List.mem_range.2 hc, rfl⟩⟩, rfl⟩, by rw [hr]; rfl⟩) h2
      | some (some m) => exact ⟨m, rfl⟩
    refine ⟨sh, rs, hsh, hrs, fun s hs c hc => ?_⟩
    obtain ⟨m, hm⟩ := hmem s hs c hc
    refine ⟨m, hm, fun s' hs' c' hc' m' hm' he => ?_⟩
    have h1' := List.all_eq_true.1 hall m (List.mem_filterMap.2 ⟨sharedRow P rs ts W ds tables theta s c,
      List.mem_map.2 ⟨(s, c), List.mem_flatMap.2 ⟨s, hs, List.mem_map.2 ⟨c, List.mem_range.2 hc, rfl⟩⟩, rfl⟩,
      by rw [hm]; rfl⟩)
    have := List.all_eq_true.1 h1' m' (List.mem_filterMap.2 ⟨sharedRow P rs ts W ds tables theta s' c',
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
      List.mem_zipIdx_iff_getElem?.2 (by simp [hk]), ?_⟩
    simp only [hu, Bool.false_eq_true, ↓reduceIte]
    exact List.mem_map.2 ⟨((((P.defs[P.root]!).body.nodes[k]!).args[j]!), j),
      List.mem_zipIdx_iff_getElem?.2 (by simp [hj]), rfl⟩
  · show ((partRun P q).map ((sharedRuns P).contains ·.1)).getD false = true
    rw [hr]
    exact hc

/-- A plan's template, port, first unit and count. -/
def Plan.at : Plan → String × String × Nat × Nat
  | .read tid port u0 n _ _ _ _ => (tid, port, u0, n)
  | .table tid port u0 n _ _ => (tid, port, u0, n)

/-- **What a row's plan reads**: its range, by the argument's node, its template and port, by the argument's index,
and its first unit (`u0_eq`) and count, the row one of the port's cuts. -/
theorem plan_of {P : Prog} {rs : Array TRange} {ts : List Spot.RecordText.Tmpl} {W : Spot.Wires.Wiring}
    {ds : List Decl} {s : Shared} {c : Nat} {pl : Plan} {tid port : String} {u0 n : Nat}
    (h : sharedPlan P rs ts W ds s c = some pl) (hat : pl.at = (tid, port, u0, n)) :
    ∃ r t ty, rs.find? (·.node == s.node) = some r ∧ ts.find? (·.tid == r.template) = some t ∧
      (P.defs[t.defn]!).params[s.arg]? = some (port, ty) ∧ tid = r.template ∧
      u0 = (Spot.RecordText.rangesAt rs r.template).foldl (fun a q => if q.first < r.first then a + q.count else a) 0 ∧
      n = r.count ∧ c < W.cutsOf r.template port := by
  cases hr : rs.find? (·.node == s.node) with
  | none => simp [sharedPlan, hr] at h
  | some r =>
    cases ht : ts.find? (·.tid == r.template) with
    | none => simp [sharedPlan, hr, ht] at h
    | some t =>
      cases hp : (P.defs[t.defn]!).params[s.arg]? with
      | none => simp [sharedPlan, hr, ht, hp] at h
      | some pp =>
        obtain ⟨port', ty⟩ := pp
        simp only [sharedPlan, hr, ht, hp, bind, Option.bind] at h
        refine ⟨r, t, ty, rfl, ht, ?_⟩
        cases hw : t.rows.lookup port' with
        | none => simp [hw] at h
        | some wv =>
          simp only [hw] at h
          generalize (Option.map (fun nd => Flock.Program.inputWidth? (P.defs[nd.fn]!).key)
            (P.defs[P.root]!).body.nodes[s.input]?).join = wo at h
          cases wo with
          | none => simp at h
          | some wn =>
            simp only at h
            split at h
            · simp at h
            · rename_i hif
              split at h <;> try (simp at h; done)
              simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not] at hif
              cases hfl : (‹Decl›).fill <;> simp only [hfl] at h <;> try (simp at h; done)
              · split at h <;> try (simp at h; done)
                simp only [Option.some.injEq] at h
                subst h
                simp only [Plan.at, Prod.mk.injEq] at hat
                obtain ⟨rfl, rfl, rfl, rfl⟩ := hat
                refine ⟨hp, rfl, rfl, rfl, ?_⟩
                simp_all
              · split at h <;> try (simp at h; done)
                simp only [Option.some.injEq] at h
                subst h
                simp only [Plan.at, Prod.mk.injEq] at hat
                obtain ⟨rfl, rfl, rfl, rfl⟩ := hat
                refine ⟨hp, rfl, rfl, rfl, ?_⟩
                simp_all

/-- **What a held row says**: its plan, `heldFrom` over the members' units and its mark, for a registered read;
its plan, `tableFrom` over them and its mark, for a table read. -/
theorem row_of {P : Prog} {rs : Array TRange} {ts : List Spot.RecordText.Tmpl} {W : Spot.Wires.Wiring}
    {ds : List Decl} {tables : List Spot.RecordText.Table} {theta : String → Bool} {s : Shared} {c : Nat}
    {m : Nat × Source} (h : sharedRow P rs ts W ds tables theta s c = some (some m)) :
    (∃ tid port u0 n p g gb o, sharedPlan P rs ts W ds s c = some (.read tid port u0 n p g gb o) ∧
      heldFrom ds theta tid port c u0 (u0 + n) p g gb (ds.length + 1) u0 = some true ∧ m = (s.input, .read p o)) ∨
    (∃ tid port u0 n name root, sharedPlan P rs ts W ds s c = some (.table tid port u0 n name root) ∧
      tableFrom ds tables theta tid port c (u0 + n) name root (ds.length + 1) u0 = some true ∧
      m = (s.input, .table name root s.base)) := by
  unfold sharedRow at h
  split at h
  · simp at h
  · rename_i tid port u0 n p g gb o hplan
    cases hh : heldFrom ds theta tid port c u0 (u0 + n) p g gb (ds.length + 1) u0 with
    | none => simp [hh] at h
    | some b =>
      cases b
      · simp [hh] at h
      · simp only [hh, Option.map_some, ↓reduceIte, Option.some.injEq] at h
        exact .inl ⟨tid, port, u0, n, p, g, gb, o, hplan, hh, h.symm⟩
  · rename_i tid port u0 n name root hplan
    cases hh : tableFrom ds tables theta tid port c (u0 + n) name root (ds.length + 1) u0 with
    | none => simp [hh] at h
    | some b =>
      cases b
      · simp [hh] at h
      · simp only [hh, Option.map_some, ↓reduceIte, Option.some.injEq] at h
        exact .inr ⟨tid, port, u0, n, name, root, hplan, hh, h.symm⟩

theorem find_tid {ts : List Spot.RecordText.Tmpl} (hnd : (ts.map (·.tid)).Nodup) {t : Spot.RecordText.Tmpl}
    (ht : t ∈ ts) : ts.find? (·.tid == t.tid) = some t := by
  induction ts with
  | nil => cases ht
  | cons a l ih =>
    rw [List.map_cons, List.nodup_cons] at hnd
    rw [List.find?_cons]
    rcases List.mem_cons.1 ht with rfl | ht'
    · simp
    · have hne : (a.tid == t.tid) = false := by
        rw [beq_eq_false_iff_ne]
        intro he
        exact hnd.1 (he ▸ List.mem_map.2 ⟨t, ht', rfl⟩)
      rw [hne]
      exact ih hnd.2 ht'

/-- **The range of a node is the one of it**, the root's ranges being of distinct nodes. -/
theorem find_node {d : Desc} {ts : Array String} {rs : Array TRange}
    (h : Flock.TemplateQuery.evaluate d ts = .ok rs) {i : Nat} (hi : i < rs.size) :
    rs.find? (·.node == rs[i].node) = some rs[i] := by
  rw [Array.find?_eq_some_iff_getElem]
  refine ⟨by simp, i, hi, rfl, fun j hj => ?_⟩
  have hne : ¬ rs[j].node = rs[i].node := fun he => by
    have := evaluate_node_inj h (Nat.lt_trans hj hi) hi he
    omega
  simpa using hne

/-- **A unit's template is its Definition's**: on a record `read` accepts, the template of a unit of `tid` that the
check finds by its id has the unit's Definition. -/
theorem unit_defn {j tr : Lean.Json} {c : Case} (h : read j tr = .ok c) {rs : Array TRange}
    (hrs : Flock.TemplateQuery.evaluate (Desc.ofProg c.f.P) (c.W.templates.map (·.1)).toArray = .ok rs)
    {tid : String} {u : Nat} {sc : Flock.Program.Scope} (hu : Properties.Record.UnitAt c.f.P rs tid u sc) :
    ∃ t, c.f.ts.find? (·.tid == tid) = some t ∧ t.defn = sc.d := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨hcf, -, -⟩ := finish_shared hc
  have hts : Spot.RecordText.templatesOf c.f.P c.f.doc = .ok c.f.ts := by
    rw [hcf]
    exact front_text hf
  obtain ⟨t, ht, htid, -, hd, -⟩ := Properties.Record.SecurityProofs.unit_tmpl (Properties.Record.SecurityProofs.readProg_of_read h) hts hrs hu
  refine ⟨t, ?_, hd⟩
  rw [← htid]
  exact find_tid (Properties.Record.SecurityProofs.tmpls_named hts).1 ht

/-- The parts `rowRuns` reads as runs (`runsOf`): every part of an argument of a root node that is no input node, but
an edge's and a strided part whose rows lie apart (`gatherRuns`). -/
def rowParts (P : Prog) : List Part :=
  let nodes := (P.defs[P.root]!).body.nodes
  let units := nodes.toList.filter fun n => !InferenceOnlyVerifier.Desc.isInputNode P n
  let edge := fun (q : Part) => match q with
    | .affine false i _ _ | .strided false i _ _ _ _ _ =>
      (nodes[i]?.map fun n => !InferenceOnlyVerifier.Desc.isInputNode P n).getD false
    | .explicit it => it.any fun r => !r.p &&
      (nodes[r.i]?.map fun n => !InferenceOnlyVerifier.Desc.isInputNode P n).getD false
    | .grid _ lanes => lanes.all fun l => !l.p &&
      (nodes[l.i]?.map fun n => !InferenceOnlyVerifier.Desc.isInputNode P n).getD false
    | _ => false
  (units.flatMap fun n => n.args.toList.flatMap fun a => a.seq.parts.toList.filter (!edge ·)).filter fun q =>
    !gathered P q

theorem rowRuns_runsOf {P : Prog} {R : Runs} (h : rowRuns P = some R) : runsOf P (rowParts P) = some R := by
  unfold rowRuns at h
  dsimp only at h
  split at h
  · split at h
    · unfold gatherRows at h
      dsimp only at h
      change Option.bind (runsOf P (rowParts P)) _ = some R at h
      cases hr : runsOf P (rowParts P) with
      | none => rw [hr] at h; cases h
      | some R0 =>
        rw [hr] at h
        dsimp only [Option.bind] at h
        split at h
        · cases h; rfl
        · cases h
    · rename_i hg
      have hg' : _ = false := Bool.eq_false_iff.2 hg
      have e := List.filter_eq_self.2 fun q hq => by
        have hq' := List.any_eq_false.1 hg' q hq
        show (!gathered P q) = true
        revert hq'
        cases gathered P q <;> simp
      exact (congrArg (runsOf P) e).trans h
  · cases h

/-- **A shared run is a run of some leaf of a part `runsOf` reads.** -/
theorem runsOf_shared {P : Prog} {parts : List Part} {R : Runs} (h : runsOf P parts = some R)
    {x : Nat × Nat × Nat} (hx : x ∈ R.shared) : 0 < x.2.2 ∧ ∃ q ∈ parts, ∃ b, partRun P q = some (x, b) := by
  unfold runsOf at h
  dsimp only at h
  split at h
  · cases h
    have hin : ∀ r ∈ ((parts.filterMap (partRun P)).filter fun r => r.1.2.2 > 0).mergeSort runOrd,
        0 < r.1.2.2 ∧ ∃ q ∈ parts, partRun P q = some r := by
      intro r hr
      obtain ⟨hrf, hpos⟩ := List.mem_filter.1 (List.mem_mergeSort.1 hr)
      obtain ⟨q, hq, hpq⟩ := List.mem_filterMap.1 hrf
      exact ⟨by simpa using hpos, q, hq, hpq⟩
    simp only [List.mem_append, List.mem_filterMap] at hx
    rcases hx with ⟨r, hr, hre⟩ | ⟨⟨a, b'⟩, hab, hre⟩
    · split at hre
      · cases hre
        obtain ⟨hpos, q, hq, hpq⟩ := hin r hr
        exact ⟨hpos, q, hq, r.2, hpq⟩
      · cases hre
    · split at hre
      · cases hre
        obtain ⟨hpos, q, hq, hpq⟩ := hin a (List.of_mem_zip hab).1
        exact ⟨hpos, q, hq, a.2, hpq⟩
      · cases hre
  · cases h

/-- **Two nonempty runs of parts `runsOf` reads that hold one leaf of one node are one run.** -/
theorem run_eq_of_gate {P : Prog} {parts : List Part} {R : Runs} (h : runsOf P parts = some R)
    {q q' : Part} (hq : q ∈ parts) (hq' : q' ∈ parts) {x y : Nat × Nat × Nat} {bx byy : Bool}
    (hx : partRun P q = some (x, bx)) (hy : partRun P q' = some (y, byy)) (hx0 : 0 < x.2.2) (hy0 : 0 < y.2.2)
    {jx jy : Nat} (hjx : x.2.1 ≤ jx ∧ jx < x.2.1 + x.2.2) (hjy : y.2.1 ≤ jy ∧ jy < y.2.1 + y.2.2)
    (hgate : x.1 = y.1 ∧ jx = jy) : x = y := by
  rcases runsOf_eqOrApart h hq hq' hx hy hx0 hy0 with e | hne | hle | hle
  · exact e
  · exact absurd hgate.1 hne
  · omega
  · omega

theorem take_count_le (l : List Part) (a : Nat) : ((l.take a).map Part.count).sum ≤ (l.map Part.count).sum := by
  conv_rhs => rw [← List.take_append_drop a l]
  rw [List.map_append, List.sum_append]
  omega

theorem take_count_mono (l : List Part) {a b : Nat} (h : a ≤ b) :
    ((l.take a).map Part.count).sum ≤ ((l.take b).map Part.count).sum := by
  have e : l.take a = (l.take b).take a := by rw [List.take_take, Nat.min_eq_left h]
  rw [e]
  exact take_count_le _ a

/-- **A reference of a sequence of parts lies in one part.** -/
theorem part_at {s : Flock.Program.Seq} (hs : s = Flock.Program.Seq.ofParts s.parts) {i i' k : Nat}
    (hi : i < s.parts.size) (hi' : i' < s.parts.size) (h1 : s.starts[i]! ≤ k) (h2 : k < s.starts[i]! + s.parts[i]!.count)
    (h1' : s.starts[i']! ≤ k) (h2' : k < s.starts[i']! + s.parts[i']!.count) : i = i' := by
  obtain ⟨-, -, hst, -⟩ := Properties.SecurityProofs.Walk.ofParts_spec s.parts
  rw [← hs] at hst
  rcases Nat.lt_trichotomy i i' with h | h | h
  · have hm := take_count_mono s.parts.toList (show i + 1 ≤ i' by omega)
    rw [Properties.SecurityProofs.Walk.sum_take_succ_map s.parts hi, ← hst i hi, ← hst i' hi'] at hm
    omega
  · exact h
  · have hm := take_count_mono s.parts.toList (show i' + 1 ≤ i by omega)
    rw [Properties.SecurityProofs.Walk.sum_take_succ_map s.parts hi', ← hst i hi, ← hst i' hi'] at hm
    omega

/-- **A part's references are references of the sequence.** -/
theorem start_lt_count {s : Flock.Program.Seq} (hs : s = Flock.Program.Seq.ofParts s.parts) {i o : Nat}
    (hi : i < s.parts.size) (ho : o < s.parts[i]!.count) : s.starts[i]! + o < s.count := by
  obtain ⟨-, -, hst, hcnt⟩ := Properties.SecurityProofs.Walk.ofParts_spec s.parts
  rw [← hs] at hst hcnt
  have hm := take_count_le s.parts.toList (i + 1)
  rw [Properties.SecurityProofs.Walk.sum_take_succ_map s.parts hi, ← hst i hi, ← hcnt] at hm
  omega

/-- **A shared run lies in its node's leaves.** -/
def SharedInNode : Prop :=
  ∀ (P : Prog), Properties.Record.SecurityProofs.ReadProg P → ∀ R, rowRuns P = some R →
    ∀ x ∈ R.shared, x.2.1 + x.2.2 ≤ ((P.defs[P.root]!).body.nodes[x.1]!).out.leaves

/-- **A unit's parameter leaf in a shared run is read by a part of its argument over that run.** A leaf of member `m`
of a unit node `k`, parameter `i`, that lies in a shared run `x` of `rowRuns`: the node has that argument, and one of
its parts is a run of `x` (`partRun`), strided parts and parts read more than once included. -/
def LeafPart : Prop :=
  ∀ (P : Prog), Properties.Record.SecurityProofs.ReadProg P → OwnRuns P → ∀ R, rowRuns P = some R →
    ∀ k, k < (P.defs[P.root]!).body.nodes.size →
      InferenceOnlyVerifier.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false →
      (((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch) →
      ∀ m, m < ((P.defs[P.root]!).body.nodes[k]!).members → ∀ i t g,
        i < (P.defs[(P.child P.rootScope k m).d]!).params.size →
        t < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!).2.leaves →
        Flock.Program.resolve P (P.child P.rootScope k m) (⟨true, i, t⟩ : Flock.Program.Ref) = .ok g →
        ∀ x ∈ R.shared, (P.defs[P.root]!).body.offsets[x.1]! + x.2.1 ≤ g →
          g < (P.defs[P.root]!).body.offsets[x.1]! + x.2.1 + x.2.2 →
          i < ((P.defs[P.root]!).body.nodes[k]!).args.size ∧
            ∃ q ∈ (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts.toList, ∃ b, partRun P q = some (x, b)

/-- **`LeafPart`, where shared runs lie in their nodes' leaves**: the shared run `x` is the run of a part at its own
place (`runsOf_shared`, `rowRuns_runsOf`), and the leaf lies in a part of its argument (`unit_leaf`): a run, which holds
one gate with `x`, so is `x` or apart from it (`node_of_gate`, `runs_shared`), or a strided part's row, apart from every
run of another place (`pieces_apart`). -/
theorem leafPart_of (hIN : SharedInNode) : LeafPart := by
  intro P hP hO R hR k hk hu hf m hm i t g hi ht hres x hx hlo hhi
  obtain ⟨hov, hlayb⟩ := Properties.Record.SecurityProofs.read_walk hP
  have hP' := hP
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP'
  obtain ⟨hroot, -, hL, -, -, -⟩ := Properties.SecurityProofs.Resolve.decoded_walk bytes P hdec hov hlayb
  obtain ⟨-, -, hnode⟩ := Properties.SecurityProofs.Resolve.decoded_root bytes P hdec k hk
  have hia : i < ((P.defs[P.root]!).body.nodes[k]!).args.size := by
    rw [hnode.1]
    exact hi
  obtain ⟨n, j, hleaf, hn, hj, ho, hg⟩ := Properties.SecurityProofs.Resolve.unit_leaf bytes P hdec hov hlayb hO hk hu
    hf hm hia ht ((P.defs[P.root]!).body.nodes[k]!).fn (P.child P.rootScope k m).off
  have hgn := hg g hres
  obtain ⟨hx0, q', hq', b', hpx⟩ := runsOf_shared (rowRuns_runsOf hR) hx
  unfold rowParts at hq'
  dsimp only at hq'
  obtain ⟨hq1, -⟩ := List.mem_filter.1 hq'
  obtain ⟨nd, hnd, hqn⟩ := List.mem_flatMap.1 hq1
  obtain ⟨a, ha, hqa⟩ := List.mem_flatMap.1 hqn
  obtain ⟨hndm, hux⟩ := List.mem_filter.1 hnd
  obtain ⟨kx, hkx, hke⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hndm)
  have hke' : (P.defs[P.root]!).body.nodes[kx]! = nd := by rw [getElem!_pos _ kx hkx]; exact hke
  subst hke'
  have hux' : InferenceOnlyVerifier.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[kx]!) = false := by simpa using hux
  obtain ⟨ix, hix, hie⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 ha)
  have hie' : ((P.defs[P.root]!).body.nodes[kx]!).args[ix]! = a := by rw [getElem!_pos _ ix hix]; exact hie
  subst hie'
  obtain ⟨qx, hqx, hqe⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 (List.mem_filter.1 hqa).1)
  have hqe' : (((P.defs[P.root]!).body.nodes[kx]!).args[ix]!).seq.parts[qx]! = q' := by
    rw [getElem!_pos _ qx hqx]; exact hqe
  subst hqe'
  have hrx : runOf P ((((P.defs[P.root]!).body.nodes[kx]!).args[ix]!).seq.parts[qx]!) = some x := by
    simp [runOf, hpx]
  obtain ⟨-, hxn, hxin, hxf⟩ := runOf_some hrx
  have hxl := hIN P hP R hR x hx
  obtain ⟨-, hc, hno⟩ := Properties.SecurityProofs.Resolve.decoded_root bytes P hdec x.1 hxn
  have hprim : (P.defs[((P.defs[P.root]!).body.nodes[x.1]!).fn]!).prim = true := by
    have h0 := hxin
    unfold Desc.isInputNode at h0
    exact (Bool.and_eq_true_iff.1 h0).1
  obtain ⟨hleaves, hg1, -⟩ := Properties.SecurityProofs.Resolve.input_node _ _ hc hno hprim hxf
  have hox : (P.defs[P.root]!).body.offsets[x.1 + 1]! =
      (P.defs[P.root]!).body.offsets[x.1]! + ((P.defs[P.root]!).body.nodes[x.1]!).out.leaves := by
    rw [(hL P.root hroot.1 hroot.2).2.2 x.1 hxn, hg1, hleaves, Nat.mul_one]
  obtain ⟨hn1, hj1⟩ := Properties.SecurityProofs.Resolve.node_of_gate P (hL P.root hroot.1) hroot.2 hxn hn
    (j := g - (P.defs[P.root]!).body.offsets[x.1]!) (by omega) hj hox ho (by omega)
  have hjx : x.2.1 ≤ j ∧ j < x.2.1 + x.2.2 := by omega
  rcases hleaf with ⟨qi, base, count, hq, hrun, h1, h2, hjj⟩ | ⟨qi, pc, hq, hpn, hpc, hpc1, hpc2, hpcpos, hplo, hphi, -⟩
  · have hpos := Properties.SecurityProofs.Resolve.run_pos hrun h1 h2
    have hjb : base ≤ j ∧ j < base + count := by
      rw [hjj]
      exact ⟨Nat.le_add_right _ _, Nat.add_lt_add_left (Nat.mod_lt _ hpos) _⟩
    obtain ⟨⟨y, b⟩, hpy⟩ : ∃ yb, partRun P ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[qi]!) = some yb := by
      cases hp : partRun P ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[qi]!) with
      | none => simp [runOf, hp] at hrun
      | some yb => exact ⟨yb, rfl⟩
    have hy : y = (n, base, count) := by simpa [runOf, hpy] using hrun
    subst hy
    have hmem : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[qi]! ∈
        (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts.toList := by
      rw [getElem!_pos _ qi hq]
      exact Array.mem_toList_iff.2 (Array.getElem_mem hq)
    by_cases hsame : k = kx ∧ i = ix ∧ qi = qx
    · obtain ⟨rfl, rfl, rfl⟩ := hsame
      exact ⟨hia, _, hmem, b', hpx⟩
    · rcases runs_shared hO hk hu hia hq hkx hux' hix hqx hrun hrx hpos hx0 hsame with ⟨e, -⟩ | hap
      · exact ⟨hia, _, hmem, b, by rw [← e]; exact hpy⟩
      · have hap' : n ≠ x.1 ∨ base + count ≤ x.2.1 ∨ x.2.1 + x.2.2 ≤ base := hap
        omega
  · exfalso
    by_cases hsame : k = kx ∧ i = ix ∧ qi = qx
    · obtain ⟨rfl, rfl, rfl⟩ := hsame
      rw [hpx] at hpn
      cases hpn
    · have hmx : (x, false) ∈ pieces P ((((P.defs[P.root]!).body.nodes[kx]!).args[ix]!).seq.parts[qx]!) := by
        simp [pieces, hpx]
      have hap := pieces_apart hO hk hu hia hq hkx hux' hix hqx hsame hpc hmx (Or.inl hpc2) hpcpos hx0
      have hap' : pc.1.1 ≠ x.1 ∨ pc.1.2.1 + pc.1.2.2 ≤ x.2.1 ∨ x.2.1 + x.2.2 ≤ pc.1.2.1 := hap
      omega

/-- **`SharedHeld`, from the stage, where a shared leaf is read by a part over its run** (`LeafPart`). -/
theorem sharedHeld_of (hNE : EdgeFreeReads) (h3 : LeafPart) : SharedHeld := by
  intro j tr c rs hread hrs g hg
  have hO : OwnRuns c.f.P := ⟨(program_of_read hread).2.2.2.2.1, hNE _ _ _ hread⟩
  obtain ⟨R, ⟨xi, xb, xn⟩, hR, hx, hlo, hhi⟩ := hg
  have hP := Properties.Record.SecurityProofs.readProg_of_read hread
  obtain ⟨hin, hsh⟩ := read_shared hread
  have hsr : sharedRuns c.f.P = R.shared := by
    unfold sharedRuns
    rw [hR]
    rfl
  have hne : (sharedRuns c.f.P).isEmpty = false := by
    rw [hsr]
    cases hs : R.shared with
    | nil => rw [hs] at hx; cases hx
    | cons => rfl
  obtain ⟨sh, rs', hsho, hrs', hrows⟩ := decide_rows (by simpa [sharedHeld] using hsh) hne
  have hrr : rs' = rs := by
    have := hrs'.symm.trans hrs
    cases this
    rfl
  subst hrr
  obtain ⟨hlist, hall⟩ := ranges_spec (Properties.Record.SecurityProofs.read_bodies hP).1 hP.2.2.2.2.2.2
    (Properties.Record.SecurityProofs.readProg_inputs hP) hrs
  have hrow : ∀ r l lo hi, Properties.Record.RowLeaf c.f.P rs' c.W r l g lo hi →
      ∃ s ∈ sh, s.input = xi ∧ ∃ cc < sharedCuts c.f.P rs' c.f.ts c.W s, ∀ m,
        sharedRow c.f.P rs' c.f.ts c.W c.x.decls c.x.tables c.isTheta s cc = some (some m) →
          m.1 = xi ∧ SourceFill c r m.2 := by
    intro r l lo hi hrl
    obtain ⟨sc, hU, pi, hpi, hport, hrw, hl, hres, -, -⟩ := hrl
    have hsz := Properties.Record.SecurityProofs.inRow_size (c.f.P.defs[sc.d]!) pi (c.W.cutsOf r.tid r.port) r.row hrw
    have ht : r.row * ((c.f.P.defs[sc.d]!).params[pi]!.2.leaves / c.W.cutsOf r.tid r.port) + l <
        ((c.f.P.defs[sc.d]!).params[pi]!).2.leaves := by
      have hl' : l < (c.f.P.defs[sc.d]!).params[pi]!.2.leaves / c.W.cutsOf r.tid r.port := by
        rw [← hsz]
        exact hl
      have h1 := Nat.mul_div_le (c.f.P.defs[sc.d]!).params[pi]!.2.leaves (c.W.cutsOf r.tid r.port)
      have h2 := Nat.mul_le_mul_right ((c.f.P.defs[sc.d]!).params[pi]!.2.leaves / c.W.cutsOf r.tid r.port)
        (show r.row + 1 ≤ c.W.cutsOf r.tid r.port from hrw)
      rw [Nat.add_one_mul] at h2
      omega
    obtain ⟨i, mm, hi, htm, hmm, hu, hsc⟩ := hU
    have hri : rs'[i]! = rs'[i] := getElem!_pos rs' i hi
    rw [hri] at htm hmm hsc
    have hmem : rs'[i] ∈ rs'.toList := Array.mem_toList_iff.2 (Array.getElem_mem hi)
    obtain ⟨hK, -, hfK, hcK⟩ := hall _ hmem
    have hkey : Flock.Program.isInputId
        (c.f.P.defs[((c.f.P.defs[c.f.P.root]!).body.nodes[rs'[i].node]!).fn]!).key = false := by
      have hn : rs'[i].node ∈ rs'.toList.map (·.node) := List.mem_map.2 ⟨_, hmem, rfl⟩
      rw [hlist] at hn
      simpa using (List.mem_filter.1 hn).2
    have hu' : InferenceOnlyVerifier.Desc.isInputNode c.f.P ((c.f.P.defs[c.f.P.root]!).body.nodes[rs'[i].node]!) =
        false := by
      unfold InferenceOnlyVerifier.Desc.isInputNode
      simp [hkey]
    subst hsc
    obtain ⟨hargs, q, hq, b, hpr⟩ := h3 c.f.P hP hO R hR _ hK hu' hfK mm (by rw [← hcK]; exact hmm) pi _ g hpi ht hres
      (xi, xb, xn) hx hlo hhi
    have hc' : (sharedRuns c.f.P).contains (xi, xb, xn) = true := by
      rw [hsr]
      exact List.contains_iff_mem.2 hx
    obtain ⟨-, s, hs, hsn, hsa, hsi, -, -, -⟩ := sharedOf_arg hsho hK hu' hargs hq hpr hc'
    obtain ⟨t, ht, htd⟩ := unit_defn hread hrs ⟨i, mm, hi, by rw [hri]; exact htm, by rw [hri]; exact hmm,
      hu, by rw [hri]⟩
    have hpar : (c.f.P.defs[t.defn]!).params[s.arg]? = some ((c.f.P.defs[t.defn]!).params[pi]!) := by
      rw [hsa, htd, getElem!_pos _ pi hpi]
      exact Array.getElem?_eq_getElem hpi
    have hpt : ((c.f.P.defs[t.defn]!).params[pi]!).1 = r.port := by
      rw [htd]
      exact hport
    have hfind : rs'.find? (·.node == s.node) = some rs'[i] := by
      rw [hsn]
      exact find_node hrs hi
    have hfindt : c.f.ts.find? (·.tid == rs'[i].template) = some t := by
      rw [htm]
      exact ht
    refine ⟨s, hs, hsi, r.row, ?_, fun mk hmk => ?_⟩
    · have hcuts : sharedCuts c.f.P rs' c.f.ts c.W s = c.W.cutsOf r.tid r.port := by
        unfold sharedCuts
        rw [hfind]
        simp only [Option.bind_some]
        rw [hfindt]
        simp only [Option.bind_some]
        rw [hpar]
        simp only [Option.map_some, Option.getD_some]
        rw [htm, hpt]
      rw [hcuts]
      exact hrw
    · have hcommon : ∀ {pl : Plan} {tid' port' : String} {u0 n : Nat}, sharedPlan c.f.P rs' c.f.ts c.W c.x.decls s r.row = some pl →
          pl.at = (tid', port', u0, n) → tid' = r.tid ∧ port' = r.port ∧ u0 ≤ r.unit ∧ r.unit < u0 + n := by
        intro pl tid' port' u0 n hplan hat
        obtain ⟨r', t', ty, hr', ht', hp', htid, hu0, hn, -⟩ := plan_of hplan hat
        rw [hfind] at hr'
        cases hr'
        rw [hfindt] at ht'
        cases ht'
        rw [hpar] at hp'
        have hport' : port' = r.port := by
          rw [← hpt, Option.some.inj hp']
        have hu0' := u0_eq (evaluate_chained hrs) hi rs'[i].template
        rw [htm] at hu0'
        rw [htm] at hu0
        refine ⟨by rw [htid, htm], hport', by omega, by omega⟩
      have hfa := (Properties.Record.SecurityProofs.inputsOf_decls hin).1
      rcases row_of hmk with ⟨tid', port', u0, n, p, gg, gb, o, hplan, hheld, rfl⟩ |
          ⟨tid', port', u0, n, name, root, hplan, hheld, rfl⟩
      · obtain ⟨rfl, rfl, h1, h2⟩ := hcommon hplan rfl
        obtain ⟨hth, d, hd, hcov, hfill⟩ := heldFrom_cover _ _ hheld r.unit h1 h2
        exact ⟨hsi, by rw [Properties.Record.SecurityProofs.fillAt_of_covers hfa hd hcov, hfill], hth⟩
      · obtain ⟨rfl, rfl, h1, h2⟩ := hcommon hplan rfl
        obtain ⟨hth, d, hd, hcov, hfill⟩ := tableFrom_cover _ _ hheld r.unit h1 h2
        exact ⟨hsi, by rw [Properties.Record.SecurityProofs.fillAt_of_covers hfa hd hcov, hfill], hth⟩
  by_cases hex : ∃ r l lo hi, Properties.Record.RowLeaf c.f.P rs' c.W r l g lo hi
  · obtain ⟨r0, l0, lo0, hi0, h0⟩ := hex
    obtain ⟨s0, hs0, -, cc0, hcc0, hm0⟩ := hrow _ _ _ _ h0
    obtain ⟨m0, hsr0, hagree⟩ := hrows s0 hs0 cc0 hcc0
    refine ⟨⟨heldOf c m0.2, sharesRun_heldOf c m0.2⟩, fun r l lo hi hrl => ?_⟩
    obtain ⟨s, hs, -, cc, hcc, hm⟩ := hrow _ _ _ _ hrl
    obtain ⟨m, hsr', -⟩ := hrows s hs cc hcc
    have he : m0 = m := hagree s hs cc hcc m hsr' (by rw [(hm0 m0 hsr0).1, (hm m hsr').1])
    rw [he]
    exact heldAt_of_fill (hm m hsr').2
  · exact ⟨⟨.read "", trivial⟩, fun r l lo hi hrl => absurd ⟨r, l, lo, hi, hrl⟩ hex⟩

/-- **`SharedInNode` holds**, on every root `rowRuns` reads: a shared run is the run of a part, and the part's last
reference, in that one part (`part_at`), is a leaf of the run's node (`BodyOk`). -/
theorem sharedInNode_holds : SharedInNode := by
  intro P hP R hR x hx
  obtain ⟨hov, hlayb⟩ := Properties.Record.SecurityProofs.read_walk hP
  have hP' := hP
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP'
  obtain ⟨hroot, hB, -, -, -, harg⟩ := Properties.SecurityProofs.Resolve.decoded_walk bytes P hdec hov hlayb
  obtain ⟨hx0, q', hq', b', hpx⟩ := runsOf_shared (rowRuns_runsOf hR) hx
  obtain ⟨xn, xb, xc⟩ := x
  dsimp only at hx0 ⊢
  unfold rowParts at hq'
  dsimp only at hq'
  obtain ⟨hq1, -⟩ := List.mem_filter.1 hq'
  obtain ⟨nd, hnd, hqn⟩ := List.mem_flatMap.1 hq1
  obtain ⟨a, ha, hqa⟩ := List.mem_flatMap.1 hqn
  obtain ⟨hndm, -⟩ := List.mem_filter.1 hnd
  obtain ⟨k, hk, hke⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hndm)
  have hke' : (P.defs[P.root]!).body.nodes[k]! = nd := by rw [getElem!_pos _ k hk]; exact hke
  subst hke'
  obtain ⟨i, hi, hie⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 ha)
  have hie' : ((P.defs[P.root]!).body.nodes[k]!).args[i]! = a := by rw [getElem!_pos _ i hi]; exact hie
  subst hie'
  obtain ⟨qi, hqi, hqe⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 (List.mem_filter.1 hqa).1)
  have hqe' : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[qi]! = q' := by
    rw [getElem!_pos _ qi hqi]; exact hqe
  subst hqe'
  have hs := harg P.root hroot.1 k hk i hi
  have hrx : runOf P ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[qi]!) = some (xn, xb, xc) := by
    simp [runOf, hpx]
  obtain ⟨hshape, -⟩ := runOf_some hrx
  have hcq : xc ≤ ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[qi]!).count := by
    rcases hshape with hqe | ⟨inner, cnt, hqe, -, -, hc⟩
    · rw [hqe]
      simp [Part.count]
    · rw [hqe]
      simp only [Part.count] at hc ⊢
      omega
  have hx' := start_lt_count hs hqi (o := xc - 1) (by omega)
  have hx'' : (((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq).starts[qi]! + (xc - 1)) <
      (Flock.Program.Seq.ofParts (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).count := by
    rw [← hs]
    exact hx'
  obtain ⟨q'', hq'', h1, h2, hget⟩ := Properties.SecurityProofs.Walk.seq_get _ hx''
  rw [← hs] at h1 h2 hget
  have e : qi = q'' := (part_at hs hq'' hqi h1 h2 (by omega) (by omega)).symm
  subst e
  rw [Nat.add_sub_cancel_left] at hget
  have hget' : (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
      ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.starts[qi]! + (xc - 1)) =
      ⟨false, xn, xb + (xc - 1)⟩ := by
    rw [hget]
    rcases hshape with hqe | ⟨inner, cnt, hqe, -, -, hc⟩
    · dsimp only at hqe
      rw [hqe]
      rfl
    · dsimp only at hqe hc
      rw [hqe, Properties.SecurityProofs.Resolve.strided_get, Nat.mod_eq_of_lt (by omega)]
  have hbody : Properties.SecurityProofs.Resolve.BodyOk P P.root := hB P.rootScope ⟨rfl, rfl, rfl⟩
  obtain ⟨-, hargs, -, -, -⟩ := hbody.1 k hk
  have href := (hargs i _ hx').2 (by rw [hget'])
  rw [hget'] at href
  have hj := href.2
  dsimp only at hj
  omega

/-- **`SharedHeld`, where every record `read` accepts is of a root with no edge.** -/
theorem sharedHeld_holds (hNE : EdgeFreeReads) : SharedHeld := sharedHeld_of hNE (leafPart_of sharedInNode_holds)

end InferenceOnlyVerifier.Record.SecurityProofs
