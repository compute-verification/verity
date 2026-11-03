import Proofs.Flock.Claims.Sampling.Select.Take
import SampledProofs.Slots

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

/-- A range list's units, in order. -/
def expand (l : List (ℕ × ℕ)) : List ℕ := l.flatMap fun r => List.range' r.1 (r.2 - r.1)

/-- Ranges `(template, first, count)` laid end to end from `a` to `b`. -/
def Tiles : List (String × ℕ × ℕ) → ℕ → ℕ → Prop
  | [], a, b => a = b
  | (_, lo, n) :: l, a, b => lo = a ∧ Tiles l (a + n) b

theorem flatMap_set_perm {α β : Type} (f : α → List β) (e : List β) :
    ∀ (l : List α) (i : ℕ) (hi : i < l.length) (y : α), f y = f l[i] ++ e →
      ((l.set i y).flatMap f).Perm (l.flatMap f ++ e)
  | [], _, hi, _, _ => absurd hi (by simp)
  | x :: xs, 0, _, y, hy => by
    simp only [List.set_cons_zero, List.flatMap_cons, List.getElem_cons_zero] at hy ⊢
    rw [hy, List.append_assoc, List.append_assoc]
    exact List.Perm.append_left _ List.perm_append_comm
  | x :: xs, i + 1, hi, y, hy => by
    simp only [List.set_cons_succ, List.flatMap_cons, List.getElem_cons_succ] at hy ⊢
    rw [List.append_assoc]
    exact (flatMap_set_perm f e xs i (by simpa using hi) y hy).append_left _

/-- The step of `Flock.Draw.strataOf`'s loop. -/
def strataStep (out : Array (String × Array (ℕ × ℕ))) (x : String × ℕ × ℕ) : Array (String × Array (ℕ × ℕ)) :=
  match x with
  | (t, lo, n) =>
    match out.findIdx? (·.1 == t) with
    | none => out.push (t, #[(lo, lo + n)])
    | some i =>
      let rs := out[i]!.2
      let rs := match rs.back? with
        | some (a, b) => if (b == lo) = true then rs.pop.push (a, lo + n) else rs.push (lo, lo + n)
        | none => rs.push (lo, lo + n)
      out.set! i (t, rs)

theorem strataOf_eq (L : Array (String × ℕ × ℕ)) : Flock.Draw.strataOf L = L.toList.foldl strataStep #[] := by
  unfold Flock.Draw.strataOf
  simp only [Id.run, bind_pure]
  rw [← Array.forIn_toList]
  generalize (#[] : Array (String × Array (ℕ × ℕ))) = out
  induction L.toList generalizing out with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨t, lo, n⟩ := x
    rw [List.forIn_cons, List.foldl_cons, ← ih]
    unfold strataStep
    dsimp only
    split
    next _ h => simp only [pure_bind, h]
    next _ i h => simp only [pure_bind, h]; rfl

/-- `strataOf`'s invariant: its strata's units are the units laid so far, each range ascending. -/
def StrataInv (out : Array (String × Array (ℕ × ℕ))) (a : ℕ) : Prop :=
  (out.toList.flatMap fun p => expand p.2.toList).Perm (List.range' 0 a) ∧
    ∀ p ∈ out.toList, ∀ r ∈ p.2.toList, r.1 ≤ r.2

theorem expand_push (rs : Array (ℕ × ℕ)) (r : ℕ × ℕ) : expand (rs.push r).toList = expand rs.toList ++ List.range' r.1 (r.2 - r.1) := by
  simp [expand]

theorem strataStep_inv (out : Array (String × Array (ℕ × ℕ))) (a : ℕ) (t : String) (n : ℕ) (h : StrataInv out a) :
    StrataInv (strataStep out (t, a, n)) (a + n) := by
  obtain ⟨hp, hr⟩ := h
  have hra : List.range' 0 a ++ List.range' a n = List.range' 0 (a + n) := by
    simpa using (List.range'_append (s := 0) (m := a) (n := n) (step := 1))
  unfold strataStep
  dsimp only
  split
  · refine ⟨?_, fun p hp' r hr' => ?_⟩
    · simp only [Array.toList_push, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil]
      rw [show expand (#[(a, a + n)] : Array (ℕ × ℕ)).toList = List.range' a n by simp [expand], ← hra]
      exact hp.append_right _
    · simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hp'
      rcases hp' with hp' | rfl
      · exact hr p hp' r hr'
      · simp only [List.mem_singleton] at hr'; subst hr'; omega
  · rename_i i hi
    have hlt : i < out.size := (Array.findIdx?_eq_some_iff_getElem.1 hi).1
    set rs := out[i]!.2 with hrs
    have hrs' : rs = out[i].2 := by rw [hrs, getElem!_pos out i hlt]
    have hmem : out[i] ∈ out.toList := Array.getElem_mem_toList hlt
    set rs' := (match rs.back? with
        | some (a', b) => if (b == a) = true then rs.pop.push (a', a + n) else rs.push (a, a + n)
        | none => rs.push (a, a + n)) with hrs''
    have hexp : expand rs'.toList = expand rs.toList ++ List.range' a n ∧ ∀ r ∈ rs'.toList, r.1 ≤ r.2 := by
      have hold : ∀ r ∈ rs.toList, r.1 ≤ r.2 := fun r hr' => hr _ hmem r (by rw [← hrs']; exact hr')
      rw [hrs'']
      split
      · rename_i a' b hb
        have hback : rs.toList = rs.pop.toList ++ [(a', b)] := by
          rw [Array.back?_eq_some_iff] at hb
          obtain ⟨ys, hys⟩ := hb
          rw [hys]; simp
        have hab : a' ≤ b := hold (a', b) (by rw [hback]; simp)
        split
        · rename_i hba
          have hba : b = a := by simpa using hba
          subst hba
          refine ⟨?_, fun r hr' => ?_⟩
          · rw [expand_push, hback]
            simp only [expand, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil,
              List.append_assoc]
            congr 1
            rw [show b + n - a' = (b - a') + n by omega]
            rw [← List.range'_append (s := a') (m := b - a') (n := n) (step := 1)]
            simp only [Nat.one_mul, show a' + (b - a') = b by omega]
          · simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr'
            rcases hr' with hr' | rfl
            · exact hold r (by rw [hback]; exact List.mem_append_left _ hr')
            · dsimp only; omega
        · refine ⟨by rw [expand_push]; simp, fun r hr' => ?_⟩
          simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr'
          rcases hr' with hr' | rfl
          · exact hold r hr'
          · dsimp only; omega
      · refine ⟨by rw [expand_push]; simp, fun r hr' => ?_⟩
        simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr'
        rcases hr' with hr' | rfl
        · exact hold r hr'
        · dsimp only; omega
    refine ⟨?_, fun p hp' r hr' => ?_⟩
    · rw [Array.set!_eq_setIfInBounds, Array.toList_setIfInBounds, ← hra]
      refine (flatMap_set_perm (fun p : String × Array (ℕ × ℕ) => expand p.2.toList) _ out.toList i
        (by simpa using hlt) (t, rs') ?_).trans (hp.append_right _)
      rw [Array.getElem_toList, ← hrs']
      exact hexp.1
    · rw [Array.set!_eq_setIfInBounds, Array.toList_setIfInBounds] at hp'
      rcases List.mem_or_eq_of_mem_set hp' with hp' | rfl
      · exact hr p hp' r hr'
      · exact hexp.2 r hr'

/-- **`strataOf` keeps the units**: over ranges laid end to end from `0` to `N`, its strata's units are `[0, N)`, each
once. -/
theorem strataOf_perm (L : Array (String × ℕ × ℕ)) (N : ℕ) (h : Tiles L.toList 0 N) :
    ((Flock.Draw.strataOf L).toList.flatMap fun p => expand p.2.toList).Perm (List.range' 0 N) := by
  rw [strataOf_eq]
  suffices ∀ (l : List (String × ℕ × ℕ)) (out : Array (String × Array (ℕ × ℕ))) (a : ℕ), StrataInv out a →
      Tiles l a N → StrataInv (l.foldl strataStep out) N from (this _ _ 0 ⟨by simp, by simp⟩ h).1
  intro l
  induction l with
  | nil => intro out a hI hT; cases hT; exact hI
  | cons x xs ih =>
    intro out a hI hT
    obtain ⟨t, lo, n⟩ := x
    obtain ⟨rfl, hT⟩ := hT
    exact ih _ _ (strataStep_inv out lo t n hI) hT

theorem tiles_push : ∀ (l : List (String × ℕ × ℕ)) (a b : ℕ) (t : String) (n : ℕ), Tiles l a b →
    Tiles (l ++ [(t, b, n)]) a (b + n)
  | [], _, _, _, _, h => by cases h; exact ⟨rfl, rfl⟩
  | (_, _, _) :: l, _, _, t, n, ⟨h1, h2⟩ => ⟨h1, tiles_push l _ _ t n h2⟩

theorem tiles_sum : ∀ (l : List (String × ℕ × ℕ)) (a b : ℕ), Tiles l a b → l.foldl (fun acc x => acc + x.2.2) a = b
  | [], _, _, h => h
  | (_, _, _) :: l, _, _, ⟨_, h2⟩ => tiles_sum l _ _ h2

/-- A `forIn` in `Except` keeps an invariant every step keeps. -/
theorem forIn_inv {α β ε : Type} (I : β → Prop) (f : α → β → Except ε (ForInStep β))
    (hf : ∀ a b, I b → ∀ s, f a b = .ok s → I s.value) :
    ∀ (l : List α) (b : β), I b → ∀ b', forIn l b f = .ok b' → I b'
  | [], b, hb, b', h => by
    simp only [List.forIn_nil] at h; obtain rfl := Except.ok.inj h; exact hb
  | a :: l, b, hb, b', h => by
    rw [List.forIn_cons] at h
    cases hs : f a b with
    | error e => rw [hs] at h; exact absurd h (by rw [bind_error]; simp)
    | ok s =>
      rw [hs, bind_ok] at h
      have := hf a b hb s hs
      cases s with
      | done c => obtain rfl := Except.ok.inj h; exact this
      | yield c => exact forIn_inv I f hf l c this b' h

/-- **The template query's ranges tile the population**: each node's units start where the last node's end, from `0`. -/
theorem evaluate_tiles (d : Flock.TemplateQuery.Desc) (ts : Array String) (rs : Array Flock.TemplateQuery.TRange)
    (h : Flock.TemplateQuery.evaluate d ts = .ok rs) :
    Tiles (rs.map fun r => (r.template, r.first, r.count)).toList 0 (Flock.TemplateQuery.population rs) := by
  unfold Flock.TemplateQuery.evaluate at h
  split at h
  · rename_i root _
    dsimp only at h
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
    have hsize : ([:root.nodes.size] : Std.Legacy.Range).size = root.nodes.size := by
      show (root.nodes.size - 0 + 1 - 1) / 1 = _
      omega
    rw [hsize, show ([:root.nodes.size] : Std.Legacy.Range).start = 0 from rfl,
      show ([:root.nodes.size] : Std.Legacy.Range).step = 1 from rfl] at h
    cases hl : forIn (m := Except String) (List.range' 0 root.nodes.size 1)
        ((#[] : Array Flock.TemplateQuery.TRange), 0) _ with
    | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨ranges, at_⟩ := y
      rw [hl, bind_ok] at h
      dsimp only at h
      have hinv : Tiles (ranges.map fun r => (r.template, r.first, r.count)).toList 0 at_ :=
        forIn_inv (fun (b : Array Flock.TemplateQuery.TRange × ℕ) =>
          Tiles (b.1.map fun (r : Flock.TemplateQuery.TRange) => (r.template, r.first, r.count)).toList 0 b.2) _
          (fun k b hb s hs => ?_) _
          ((#[] : Array Flock.TemplateQuery.TRange), 0) (by simp [Tiles]) _ hl
      · have hrs : rs = ranges := by
          split at h
          · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
          · exact (Except.ok.inj h).symm
        subst hrs
        have hpop : Flock.TemplateQuery.population rs = at_ := by
          unfold Flock.TemplateQuery.population
          rw [← Array.foldl_toList, ← tiles_sum _ 0 at_ hinv, Array.toList_map, List.foldl_map]
        rw [hpop]; exact hinv
      · obtain ⟨ranges₀, at₀⟩ := b
        dsimp only at hs hb
        split at hs
        · obtain rfl := Except.ok.inj hs; exact hb
        · split at hs
          · exact absurd hs (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
          · obtain rfl := Except.ok.inj hs
            simp only [ForInStep.value, Array.map_push, Array.toList_push]
            exact tiles_push _ 0 at₀ _ _ hb
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])

/-- The step of `Partition.callRanges`' fold. -/
def callStep (acc : Array Flock.TemplateQuery.TRange × ℕ) (x : ℕ × String × ℕ) : Array Flock.TemplateQuery.TRange × ℕ :=
  if x.2.2 == 0 then acc else (acc.1.push ⟨x.1, x.2.1, acc.2, x.2.2⟩, acc.2 + x.2.2)

theorem callStep_tiles : ∀ (l : List (ℕ × String × ℕ)) (acc : Array Flock.TemplateQuery.TRange × ℕ),
    Tiles (acc.1.map fun r => (r.template, r.first, r.count)).toList 0 acc.2 →
      Tiles ((l.foldl callStep acc).1.map fun r => (r.template, r.first, r.count)).toList 0 (l.foldl callStep acc).2
  | [], _, h => h
  | x :: l, acc, h => by
    rw [List.foldl_cons]
    refine callStep_tiles l _ ?_
    unfold callStep
    split
    · exact h
    · simp only [Array.map_push, Array.toList_push]
      exact tiles_push _ 0 acc.2 _ _ h

/-- **A Call query's ranges tile the population**: `callRanges` lays each Call with units where the last one ends. -/
theorem callRanges_tiles (cs : Array (ℕ × String × ℕ)) :
    Tiles ((Partition.callRanges cs).map fun r => (r.template, r.first, r.count)).toList 0
      (Flock.TemplateQuery.population (Partition.callRanges cs)) := by
  have heq : Partition.callRanges cs = (cs.toList.foldl callStep (#[], 0)).1 := by
    unfold Partition.callRanges; rw [Array.foldl_toList]; rfl
  have h := callStep_tiles cs.toList (#[], 0) (by simp [Tiles])
  have hpop : Flock.TemplateQuery.population (Partition.callRanges cs) = (cs.toList.foldl callStep (#[], 0)).2 := by
    unfold Flock.TemplateQuery.population
    rw [heq, ← Array.foldl_toList, ← tiles_sum _ 0 _ h, Array.toList_map, List.foldl_map]
  rw [hpop, heq]; exact h

/-- **The query's ranges tile the population**: a template query's (`evaluate_tiles`) or a Call query's
(`callRanges_tiles`). -/
theorem partition_evaluate_tiles (d : Program) (query : Lean.Json)
    (rs : Array Flock.TemplateQuery.TRange) (h : Partition.evaluate d query = .ok rs) :
    Tiles (rs.map fun r => (r.template, r.first, r.count)).toList 0 (Flock.TemplateQuery.population rs) := by
  unfold Partition.evaluate at h
  dsimp only at h
  split at h
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
  · split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · split at h
      · rename_i ts _
        cases hts : ts with
        | error e => rw [hts] at h; exact absurd h (by rw [bind_error]; simp)
        | ok ts' =>
          rw [hts, bind_ok] at h
          obtain ⟨rs', hrs', h⟩ := bind_eq_ok.1 h
          obtain ⟨_, -, h⟩ := bind_eq_ok.1 h
          obtain rfl := (Except.ok.inj h).symm
          exact evaluate_tiles d.desc ts' _ hrs'
      · obtain ⟨cs, _, h⟩ := bind_eq_ok.1 h
        obtain rfl := (Except.ok.inj h).symm
        exact callRanges_tiles cs

theorem list_mapM_ok {α β ε : Type} (f : α → Except ε β) :
    ∀ (xs : List α) (ys : List β), xs.mapM f = .ok ys → List.Forall₂ (fun a b => f a = .ok b) xs ys
  | [], ys, h => by
    simp only [List.mapM_nil] at h; obtain rfl := (Except.ok.inj h).symm; exact List.Forall₂.nil
  | x :: xs, ys, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | error e => rw [hx] at h; exact absurd h (by rw [bind_error]; simp)
    | ok y =>
      rw [hx, bind_ok] at h
      cases hxs : xs.mapM f with
      | error e => rw [hxs] at h; exact absurd h (by rw [bind_error]; simp)
      | ok ys' =>
        rw [hxs, bind_ok] at h
        obtain rfl := (Except.ok.inj h).symm
        exact List.Forall₂.cons hx (list_mapM_ok f xs ys' hxs)

theorem mapM_ok {α β ε : Type} (f : α → Except ε β) (xs : Array α) (ys : Array β) (h : xs.mapM f = .ok ys) :
    List.Forall₂ (fun a b => f a = .ok b) xs.toList ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; exact absurd h (by simp [Functor.map, Except.map])
  | ok ys' =>
    rw [hl] at h
    have : ys = ys'.toArray := (Except.ok.inj h).symm
    subst this
    exact list_mapM_ok f _ _ hl

theorem toList_expandArr (ranges : Array (ℕ × ℕ)) :
    (ranges.flatMap fun (r : ℕ × ℕ) => (Array.range (r.2 - r.1)).map (r.1 + ·)).toList = expand ranges.toList := by
  simp [expand, Array.toList_flatMap, List.range'_eq_map_range]

/-- The input class `Slots.layout` lays after the population: every gate of the root's input nodes, a proof block
each, or no stratum when there are none. -/
def inputClass (d : Program) : Array (Array Slots.ProofBlock) :=
  let inputs := (Slots.inputGates d.desc).map fun g => ({ unit := none, gates := (g, g + 1) } : Slots.ProofBlock)
  if inputs.isEmpty then #[] else #[inputs]

/-- A population stratum as `Slots.layout` lays it: its units in order, each one proof block. -/
def LaysOut (p : String × Array (ℕ × ℕ)) (s : Array Slots.ProofBlock) : Prop :=
  s.toList.map (·.unit) = (expand p.2.toList).map some

/-- **`Slots.layout`'s classes**: the strata of the evaluated query, each laid out unit by unit, then the input class. -/
theorem layout_ok (d : Program) (query : Lean.Json) (l : Slots.Layout)
    (h : Slots.layout d query = .ok l) :
    ∃ rs, Partition.evaluate d query = .ok rs ∧ ∃ pop : Array (Array Slots.ProofBlock),
      l = { classes := #[pop, inputClass d] } ∧
      List.Forall₂ LaysOut (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))).toList pop.toList := by
  unfold Slots.layout at h
  obtain ⟨rs, he, h⟩ := bind_eq_ok.1 h
  obtain ⟨gs, _, h⟩ := bind_eq_ok.1 h
  dsimp only at h
  obtain ⟨pop, hp, h⟩ := bind_eq_ok.1 h
  refine ⟨rs, he, pop, (Except.ok.inj h).symm, ?_⟩
  refine (mapM_ok _ _ _ hp).imp fun p s hs => ?_
  obtain ⟨t, ranges⟩ := p
  dsimp only at hs
  have hf := mapM_ok _ _ _ hs
  rw [toList_expandArr] at hf
  have key : ∀ (us : List ℕ) (bs : List Slots.ProofBlock), List.Forall₂ (fun u (b : Slots.ProofBlock) =>
      (match gs[u]? with
        | some g => pure { unit := some u, gates := g }
        | _ => throw (toString "unit " ++ toString u ++ toString " has no gates") : Except String Slots.ProofBlock) =
        .ok b) us bs → bs.map (·.unit) = us.map some := by
    intro us bs hF
    induction hF with
    | nil => rfl
    | cons hab _ ih =>
      rename_i u b us' bs'
      split at hab
      · obtain rfl := (Except.ok.inj hab).symm
        simp [ih]
      · exact absurd hab (by simp [throw, throwThe, MonadExceptOf.throw])
  exact key _ _ hf

/-- **The slots `checked` lists**: `v` is one when slot `v` holds a proof block whose unit is proved (or none, an
input). -/
theorem mem_checked (l : Slots.Layout) (proved : Array ℕ) (v : ℕ) :
    v ∈ Slots.checked l proved ↔ ∃ h : v < l.blocks.size, l.blocks[v].unit.all proved.contains = true := by
  unfold Slots.checked
  simp only [Array.mem_map, Array.mem_filter]
  constructor
  · rintro ⟨⟨b, w⟩, ⟨hz, hu⟩, rfl⟩
    dsimp only at hu ⊢
    rw [Array.mem_zipIdx_iff_getElem?] at hz
    dsimp only at hz
    obtain ⟨hv, rfl⟩ := Array.getElem?_eq_some_iff.1 hz
    exact ⟨hv, hu⟩
  · rintro ⟨hv, hu⟩
    exact ⟨(l.blocks[v], v), ⟨Array.mem_zipIdx_iff_getElem?.2 (Array.getElem?_eq_getElem hv), hu⟩, rfl⟩

end Properties.SampledProofs.SecurityProofs
