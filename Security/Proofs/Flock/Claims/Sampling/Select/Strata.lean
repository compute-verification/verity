import Proofs.Flock.Claims.Sampling.Select.Design

/-!
# The strata the audit publishes are the ones the draw ran (`Properties.SampledProofs.StrataDerived`)

* `ranges_read`, `size_read`, `stratum_read`: `Record.profileDraw` reads a stratum C-Flock's `Stratum.toJson` writes back
  as its template, its ranges' units and its `k`.
* `profileDraw_kind`, `profileDraw_stratified`, `profileDraw_work`: the profile's draw of a law object: `subset k` at
  its `k`, a Bernoulli draw, or the strata a stratified or work law's object writes.
* `flockLaw_subsetKind`, `deriveLaw_ofSubset`: a `subset` law object is the subset law and expands to none.
* `strataDerived`; `strataDerived_honest` and `strataDerived_nontrivial`, its two witnesses.
-/

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open Lean (Json)
open Claims.Sampling

theorem list_mapM_some {α β : Type} (f : α → Option β) :
    ∀ (xs : List α) (ys : List β), xs.mapM f = some ys → List.Forall₂ (fun a b => f a = some b) xs ys
  | [], ys, h => by
    simp only [List.mapM_nil] at h; obtain rfl := (Option.some.inj h).symm; exact List.Forall₂.nil
  | x :: xs, ys, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | none => rw [hx] at h; cases h
    | some y =>
      rw [hx] at h
      cases hxs : xs.mapM f with
      | none => rw [hxs] at h; cases h
      | some ys' =>
        rw [hxs] at h
        obtain rfl := (Option.some.inj h).symm
        exact List.Forall₂.cons hx (list_mapM_some f xs ys' hxs)

theorem array_mapM_some {α β : Type} (f : α → Option β) (xs : Array α) (ys : Array β) (h : xs.mapM f = some ys) :
    List.Forall₂ (fun a b => f a = some b) xs.toList ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | none => rw [hl] at h; cases h
  | some ys' =>
    rw [hl] at h
    have : ys = ys'.toArray := (Option.some.inj h).symm
    subst this
    exact list_mapM_some f _ _ hl

theorem eq_map_of_forall₂ {α β : Type} (f : α → β) {xs : List α} {ys : List β}
    (h : List.Forall₂ (fun a b => b = f a) xs ys) : ys = xs.map f := by
  induction h with
  | nil => rfl
  | cons hab _ ih => rw [hab, ih, List.map_cons]

/-! ## A stratum read back -/

/-- **`ranges?` reads the ranges `Stratum.toJson` writes**: each `[lo, hi]` as itself, where it reads them, with
`lo < hi`. -/
theorem ranges_read (us : Array (ℕ × ℕ)) (rs : Array (ℤ × ℤ))
    (h : Draw.ranges? (Json.arr (us.map fun p => Json.arr #[Lean.toJson p.1, Lean.toJson p.2])) = some rs) :
    List.Forall₂ (fun p r => r = ((p.1 : ℤ), (p.2 : ℤ)) ∧ p.1 < p.2) us.toList rs.toList := by
  simp only [Draw.ranges?] at h
  split at h
  · cases h
  · have hf := array_mapM_some _ _ _ h
    rw [Array.toList_map, List.forall₂_map_left_iff] at hf
    refine hf.imp fun {p r} hp => ?_
    change (if ((p.1 : ℤ) < (p.2 : ℤ)) then some ((p.1 : ℤ), (p.2 : ℤ)) else none) = some r at hp
    split at hp
    · exact ⟨(Option.some.inj hp).symm, by exact_mod_cast ‹((p.1 : ℤ) < (p.2 : ℤ))›⟩
    · cases hp

theorem foldl_read (F : ℤ → ℤ × ℤ → ℤ) (hF : ∀ a lo hi, F a (lo, hi) = a + (hi - lo)) (l : List (ℕ × ℕ))
    (rs : List (ℤ × ℤ)) (h : List.Forall₂ (fun p r => r = ((p.1 : ℤ), (p.2 : ℤ)) ∧ p.1 < p.2) l rs) :
    ∀ acc, rs.foldl F acc = acc + ((expand l).length : ℤ) := by
  induction h with
  | nil => intro acc; simp [expand]
  | @cons p r l rs hr _ ih =>
    intro acc
    obtain ⟨rfl, hlt⟩ := hr
    rw [List.foldl_cons, hF, ih]
    simp only [expand, List.flatMap_cons, List.length_append, List.length_range']
    omega

/-- **The size `profileDraw` reads**: the units of the ranges. -/
theorem size_read (F : ℤ → ℤ × ℤ → ℤ) (hF : ∀ a lo hi, F a (lo, hi) = a + (hi - lo)) (us : Array (ℕ × ℕ))
    (rs : Array (ℤ × ℤ)) (h : List.Forall₂ (fun p r => r = ((p.1 : ℤ), (p.2 : ℤ)) ∧ p.1 < p.2) us.toList rs.toList) :
    (rs.foldl F 0).toNat = rsSize us := by
  rw [← Array.foldl_toList, foldl_read F hF _ _ h, zero_add, Int.toNat_natCast]
  rfl

/-- What `profileDraw` reads of one stratum object `s` into the stratum `c`. -/
def StratumRead (s : Json) (c : Core.Stratum) : Prop :=
  ∃ (t : String) (rs : Array (ℤ × ℤ)) (F : ℤ → ℤ × ℤ → ℤ), (∀ a lo hi, F a (lo, hi) = a + (hi - lo)) ∧
    getD s "template" = .str t ∧ Draw.ranges? (getD s "units") = some rs ∧ nat? (getD s "k") = some c.k ∧
    c = { name := t, size := (rs.foldl F 0).toNat, k := c.k }

/-- **A stratum object read back**: its template, the units of its ranges and its `k`. -/
theorem stratum_read (s : Json) (c : Core.Stratum) (t : String) (us : Array (ℕ × ℕ)) (k : ℕ)
    (ht : getD s "template" = .str t)
    (hu : getD s "units" = Json.arr (us.map fun p => Json.arr #[Lean.toJson p.1, Lean.toJson p.2]))
    (hk : getD s "k" = Lean.toJson k) (h : StratumRead s c) :
    c = { name := t, size := rsSize us, k := k } := by
  obtain ⟨t', rs, F, hF, ht', hrs, hk', hc⟩ := h
  rw [ht] at ht'
  obtain rfl := Json.str.inj ht'
  rw [hu] at hrs
  rw [hk, nat_toJson] at hk'
  rw [hc, size_read F hF us rs (ranges_read us rs hrs), ← Option.some.inj hk']

/-! ## The profile's draw -/

/-- **The profile's draw of a law object**: `subset k` at the object's `k`, a Bernoulli draw, or strata, of a
`stratified` or `work` object. -/
theorem profileDraw_kind (law : Json) (draw : Core.SamplingDesign) (h : Record.profileDraw law = some draw) :
    (kindOf law = "subset" ∧ ∃ k, draw = .subset k ∧ nat? (getD law "k") = some k) ∨ (∃ p, draw = .bernoulli p) ∨
      ((kindOf law = "stratified" ∨ kindOf law = "work") ∧ ∃ strata, draw = .stratified strata) := by
  unfold Record.profileDraw at h
  split at h
  · rename_i heq
    obtain ⟨k, hk, rfl⟩ := Option.map_eq_some_iff.1 h
    exact Or.inl ⟨by simp [kindOf, heq], k, rfl, hk⟩
  · right; left
    split at h
    · rename_i p _
      cases hp : Draw.probability? p with
      | none => simp [hp] at h
      | some ab =>
        obtain ⟨a, b⟩ := ab
        cases hr : Core.Ratio.mk' a b with
        | none => simp [hp, hr] at h
        | some r => simp [hp, hr] at h; exact ⟨r, by subst h; rfl⟩
    · cases h
  all_goals first
    | cases h
    | (rename_i heq
       right; right
       refine ⟨by simp [kindOf, heq], ?_⟩
       split at h
       · obtain ⟨strata, -, h⟩ := Option.bind_eq_some_iff.1 h
         exact ⟨strata, (Option.some.inj h).symm⟩
       · cases h)

/-- **The strata the profile reads** of a `stratified` or `work` object: one per stratum object, in order, each read
back (`StratumRead`). -/
theorem profileDraw_strata (law : Json) (draw : Core.SamplingDesign) (ss : Array Json)
    (hk : kindOf law = "stratified" ∨ kindOf law = "work") (hs : getD law "strata" = .arr ss)
    (h : Record.profileDraw law = some draw) :
    ∃ strata, draw = .stratified strata ∧ List.Forall₂ StratumRead ss.toList strata.toList := by
  unfold Record.profileDraw at h
  split at h
  all_goals try (rename_i heq; simp [kindOf, heq] at hk; done)
  all_goals try cases h
  all_goals
    rw [hs] at h
    obtain ⟨strata, hm, h⟩ := Option.bind_eq_some_iff.1 h
    refine ⟨strata, (Option.some.inj h).symm, (array_mapM_some _ _ _ hm).imp fun {s c} hc => ?_⟩
    split at hc
    · rename_i t ht
      obtain ⟨rs, hrs, hc⟩ := Option.bind_eq_some_iff.1 hc
      obtain ⟨k, hk', hc⟩ := Option.bind_eq_some_iff.1 hc
      obtain rfl := (Option.some.inj hc).symm
      refine ⟨t, rs, _, ?_, ht, hrs, hk', rfl⟩
      intro a lo hi
      rfl
    · cases hc

theorem kindOf_stratified (k : ℕ) (ss : Array Flock.Draw.Stratum) :
    kindOf (Flock.Draw.Law.stratified k ss).toJson = "stratified" := rfl

theorem kindOf_work (k : ℕ) (ws : Array Flock.Draw.WorkStratum) :
    kindOf (Flock.Draw.Law.work k ws).toJson = "work" := rfl

/-- **The profile's draw of a stratified law's object**: its strata, each its template, the units of its ranges and its
`k`. -/
theorem profileDraw_stratified (k : ℕ) (ss : Array Flock.Draw.Stratum) (draw : Core.SamplingDesign)
    (h : Record.profileDraw (Flock.Draw.Law.stratified k ss).toJson = some draw) :
    ∃ st : Array Core.Stratum, draw = .stratified st ∧
      st.toList = ss.toList.map fun x => { name := x.template, size := rsSize x.units, k := x.k } := by
  obtain ⟨st, rfl, hF⟩ := profileDraw_strata _ draw (ss.map (·.toJson)) (Or.inl (kindOf_stratified k ss)) rfl h
  refine ⟨st, rfl, ?_⟩
  rw [Array.toList_map, List.forall₂_map_left_iff] at hF
  exact eq_map_of_forall₂ _ (hF.imp fun {x c} hc => stratum_read _ c x.template x.units x.k rfl rfl rfl hc)

/-- **The profile's draw of a work law's object**: the stratified draw of its strata, each its template, the units of
its ranges and its `k`. -/
theorem profileDraw_work (k : ℕ) (ws : Array Flock.Draw.WorkStratum) (draw : Core.SamplingDesign)
    (h : Record.profileDraw (Flock.Draw.Law.work k ws).toJson = some draw) :
    ∃ st : Array Core.Stratum, draw = .stratified st ∧
      st.toList = ws.toList.map fun x => { name := x.template, size := rsSize x.units, k := x.k } := by
  obtain ⟨st, rfl, hF⟩ := profileDraw_strata _ draw (ws.map (·.toJson)) (Or.inr (kindOf_work k ws)) rfl h
  refine ⟨st, rfl, ?_⟩
  rw [Array.toList_map, List.forall₂_map_left_iff] at hF
  exact eq_map_of_forall₂ _ (hF.imp fun {x c} hc => stratum_read _ c x.template x.units x.k rfl rfl rfl hc)

/-! ## A `subset` law object -/

/-- **The law `flockLaw` accepts, by its object's kind**: a `subset` object's subset law at its `k`, a `bernoulli`
object's Bernoulli law at the probability its `p` writes, or a `stratified` or `work` object. -/
theorem flockLaw_kind (law : Json) (n : ℕ) (parsed : Flock.Draw.Law) (h : Draw.flockLaw law n = .ok parsed) :
    (kindOf law = "subset" ∧ ∃ k, parsed = .subset k ∧ nat? (getD law "k") = some k) ∨
      (kindOf law = "bernoulli" ∧ ∃ s a b, getD law "p" = .str s ∧ Draw.probability? s = some (a, b) ∧
        parsed = .bernoulli a b) ∨ kindOf law = "stratified" ∨ kindOf law = "work" := by
  unfold Draw.flockLaw at h
  split at h
  next s hs =>
    have hk : kindOf law = s := by simp [kindOf, hs]
    dsimp only at h
    split at h
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      split at h
      · rename_i k hnat
        split at h
        · exact absurd h (throw_bind_ne _ _ _)
        · obtain rfl := (Except.ok.inj h).symm
          exact Or.inl ⟨by rwa [hk], _, rfl, hnat⟩
      · exact absurd h (throw_ne _ _)
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      split at h
      · rename_i ps hps
        split at h
        · rename_i a b hab
          obtain rfl := (Except.ok.inj h).symm
          exact Or.inr (Or.inl ⟨by rwa [hk], ps, a, b, hps, hab, rfl⟩)
        · exact absurd h (throw_ne _ _)
      · exact absurd h (throw_ne _ _)
    · exact Or.inr (Or.inr (Or.inl (by rwa [hk])))
    · exact Or.inr (Or.inr (Or.inr (by rwa [hk])))
    · exact absurd h (throw_ne _ _)
  next => exact absurd h (throw_ne _ _)

/-- **A `subset` object `flockLaw` accepts is the subset law at its `k`.** -/
theorem flockLaw_subsetKind (law : Json) (n : ℕ) (parsed : Flock.Draw.Law) (hk : kindOf law = "subset")
    (h : Draw.flockLaw law n = .ok parsed) : ∃ k, parsed = .subset k ∧ nat? (getD law "k") = some k := by
  rcases flockLaw_kind law n parsed h with ⟨-, h'⟩ | ⟨h', -⟩ | h' | h'
  · exact h'
  all_goals (rw [hk] at h'; exact absurd h' (by decide))

/-- **A `subset` object expands to none.** -/
theorem deriveLaw_ofSubset (d : Program) (query law : Json) (table : Option Json) (derived : Option Flock.Draw.Law)
    (hk : kindOf law = "subset") (h : Draw.deriveLaw d query law table = .ok derived) : derived = none := by
  rcases deriveLaw_cases d query law table derived h with ⟨rfl, -, -⟩ | ⟨S, k, law', -, rfl, -⟩
  · rfl
  · exfalso
    unfold Draw.deriveLaw at h
    split at h
    next s hs =>
      have hk' : s = "subset" := by rw [← hk]; simp [kindOf, hs]
      subst hk'
      dsimp only at h
      exact absurd (Except.ok.inj h) (by simp)
    next hne =>
      have : kindOf law = "" := by
        unfold kindOf; split
        · rename_i s hs; exact absurd hs (hne s)
        · rfl
      rw [this] at hk; exact absurd hk (by decide)

/-! ## The strata -/

theorem inputClass_sum (d : Program) :
    (strataSizes ((inputClass d).toList.map fun s => s.toList.map puGates)).sum = (Slots.inputGates d.desc).size := by
  simp only [inputClass]
  split
  · rename_i h
    have h0 : (Slots.inputGates d.desc).size = 0 := by simpa using h
    simp [strataSizes, h0]
  · simp [strataSizes]

/-- **The strata the audit publishes are the ones the draw ran** (`Properties.SampledProofs.StrataDerived`). -/
theorem strataDerived : StrataDerived := by
  intro rule d query table l D n named law draw strata hl hD hn hnamed hlaw hdraw hstrata
  have hwf := designWellFormed rule d query table l D hl hD
  obtain ⟨rs, he, pop, rfl, hlays⟩ := layout_ok d query l hl
  set S := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))
  obtain ⟨n', named', derived, hn', hnamed', hd, hDc⟩ := slotDesign_ok rule d query table D hD
  have e1 : n' = n := Except.ok.inj (hn'.symm.trans hn)
  rw [e1] at hnamed' hDc
  have e2 : named' = named := Except.ok.inj (hnamed'.symm.trans hnamed)
  rw [e2] at hDc hd
  have hlaw' : law = lawObj named derived := by
    unfold Draw.derive at hlaw
    rw [hd] at hlaw
    exact (Except.ok.inj hlaw).symm
  have hperm : (popUnits S).Perm (List.range' 0 (Flock.TemplateQuery.population rs)) :=
    strataOf_perm _ _ (partition_evaluate_tiles d query rs he)
  have hnU : n = (popUnits S).length := by
    rw [units_eq d query rs he] at hn
    rw [hperm.length_eq, List.length_range']
    exact (Except.ok.inj hn).symm
  have hns : pop.toList.map (·.size) = S.toList.map fun p => rsSize p.2 := (laysOut_units _ _ hlays).2
  have hsum : (S.toList.map fun p => rsSize p.2).sum = n := by rw [hnU, length_popUnits]
  obtain ⟨first, rfl, hfirst⟩ : ∃ first : OneStage, D = ⟨[first, .all]⟩ ∧
      first.strata (S.toList.map fun p => rsSize p.2) = some (strata.toList.map fun x => (x.size, x.k)) := by
    rcases hDc with ⟨rfl, rfl⟩ | ⟨hne, parsed, hp, rfl⟩
    · refine ⟨.all, rfl, ?_⟩
      have hnamed0 : named = Json.mkObj [("law", "subset"), ("k", Lean.toJson n)] := (Except.ok.inj hnamed).symm
      have hk : kindOf named = "subset" := by rw [hnamed0]; rfl
      obtain rfl := deriveLaw_ofSubset d query named table derived hk hd
      rw [hlaw'] at hdraw
      change Record.profileDraw named = some draw at hdraw
      rcases profileDraw_kind named draw hdraw with ⟨-, k, rfl, hk'⟩ | ⟨p, rfl⟩ | ⟨hk2, -⟩
      · rw [hnamed0, getD_k, nat_toJson] at hk'
        obtain rfl := Option.some.inj hk'
        simp only [Bound.strataOf, Option.some.injEq] at hstrata
        subst hstrata
        show some [((S.toList.map fun p => rsSize p.2).sum, (S.toList.map fun p => rsSize p.2).sum)] = _
        rw [hsum]
        rfl
      · simp [Bound.strataOf] at hstrata
      · rcases hk2 with h | h <;> rw [hk] at h <;> exact absurd h (by decide)
    · refine ⟨_, rfl, ?_⟩
      rcases deriveLaw_cases d query named table derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S', k, law', hS', rfl, hder⟩
      · rw [hlaw'] at hdraw
        change Record.profileDraw named = some draw at hdraw
        rcases profileDraw_kind named draw hdraw with ⟨hk, k, rfl, hk'⟩ | ⟨p, rfl⟩ | ⟨hk', -⟩
        · obtain ⟨k', rfl, hk''⟩ := flockLaw_subsetKind named n parsed hk hp
          rw [hk'] at hk''
          obtain rfl := Option.some.inj hk''
          simp only [Bound.strataOf, Option.some.injEq] at hstrata
          subst hstrata
          show some [((S.toList.map fun p => rsSize p.2).sum, k)] = _
          rw [hsum]
          rfl
        · simp [Bound.strataOf] at hstrata
        · rcases hk' with h | h
          · exact absurd h hk1
          · exact absurd h hk2
      · have e3 : S' = S := by
          rw [strata_eq d query rs he] at hS'
          exact (Except.ok.inj hS').symm
        rw [e3] at hder
        rw [hlaw'] at hdraw
        rcases hder with hder | ⟨t, tab, ht, hwork⟩
        · obtain rfl := derive_ok k _ law' hder
          obtain ⟨st, rfl, hst⟩ := profileDraw_stratified k _ draw hdraw
          simp only [Bound.strataOf, Option.some.injEq] at hstrata
          subst hstrata
          rw [hst]
          simp only [Option.getD_some, oneStageOf, OneStage.strata, Array.toList_map, List.map_map,
            Function.comp_def, Flock.Draw.stratumK, strataSize]
        · obtain ⟨ws, hF, rfl⟩ := deriveWork_ok k _ tab law' hwork
          obtain ⟨st, rfl, hst⟩ := profileDraw_work k _ draw hdraw
          simp only [Bound.strataOf, Option.some.injEq] at hstrata
          subst hstrata
          rw [hst]
          have hunits := units_of_forall₂ _ _ hF
          have hsz : S.toList.map (fun p => rsSize p.2) = ws.toList.map (·.size) := by
            have := congrArg (List.map rsSize) hunits
            simp only [List.map_map, Function.comp_def] at this
            rw [← this]
            simp [stratum_size]
          rw [hsz]
          simp only [Option.getD_some, oneStageOf, OneStage.strata, Array.toList_map, List.map_map,
            Function.comp_def, List.zip_map', totalWork_eq, Flock.Draw.workK, stratum_size]
  refine ⟨hwf, [strata.toList.map fun x => (x.size, x.k), [(Slots.inputGates d.desc).size].map fun m => (m, m)], rfl,
    fun p hp => ?_, by simp⟩
  simp only [slotPartition, List.map_cons, List.map_nil, List.zip_cons_cons, List.zip_nil_left, List.mem_cons,
    List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · simpa [strataSizes, Function.comp_def, hns] using hfirst
  · show some [((strataSizes _).sum, (strataSizes _).sum)] = _
    rw [inputClass_sum]

/-! ## The two witnesses -/

/-- **`StrataDerived`'s premises hold at the code** (the two-lemma rule's `_honest`): the `All` rule over two units
names `subset:2`, which `Draw.derive` leaves as it is, whose profile's draw is `subset 2`, read as one stratum of two
units drawing both; and on a layout's partition of a population class of two proof blocks and an input class of one,
those strata, then the input class drawn whole, are the strata draw of the design the `All` rule runs, both classes
`all`. -/
theorem strataDerived_honest :
    Select.Rule.all.lawOver 2 = .ok (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) ∧
      Draw.derive default Json.null (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) none =
        .ok (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) ∧
      Record.profileDraw (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) = some (.subset 2) ∧
      Bound.strataOf 2 (.subset 2) = some #[{ name := "unit", size := 2, k := 2 }] ∧
      StrataDrawn ⟨[.all, .all]⟩
        (slotPartition { classes := #[#[#[{ unit := some 0, gates := (0, 1) }, { unit := some 1, gates := (1, 2) }]],
          #[#[{ unit := none, gates := (2, 3) }]]] })
        (((#[{ name := "unit", size := 2, k := 2 }] : Array Core.Stratum).toList.map fun x => (x.size, x.k)) ++
          [1].map fun m => (m, m)) := by
  refine ⟨rfl, rfl, rfl, rfl, ?_⟩
  show StrataDrawn ⟨[.all, .all]⟩ ⟨[[[[0], [1]]], [[[2]]]]⟩ [(2, 2), (1, 1)]
  refine ⟨⟨rfl, fun p hp => ?_⟩, [[(2, 2)], [(1, 1)]], rfl, fun p hp => ?_, rfl⟩
  · simp only [List.zip_cons_cons, List.zip_nil_left, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> trivial
  · simp only [List.zip_cons_cons, List.zip_nil_left, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> rfl

/-- **`StrataDerived`'s conclusion is not trivial** (the two-lemma rule's `_nontrivial`): on that partition, the strata
of a `subset:1` draw over the two units, then the input class drawn whole, are not the strata draw of the design the
`All` rule runs, which draws both units. -/
theorem strataDerived_nontrivial :
    ¬ StrataDrawn ⟨[.all, .all]⟩
      (slotPartition { classes := #[#[#[{ unit := some 0, gates := (0, 1) }, { unit := some 1, gates := (1, 2) }]],
        #[#[{ unit := none, gates := (2, 3) }]]] })
      ((((Bound.strataOf 2 (.subset 1)).getD #[]).toList.map fun x => (x.size, x.k)) ++ [1].map fun m => (m, m)) := by
  show ¬ StrataDrawn ⟨[.all, .all]⟩ ⟨[[[[0], [1]]], [[[2]]]]⟩ [(2, 1), (1, 1)]
  rintro ⟨-, sss, hlen, hall, hflat⟩
  rcases sss with _ | ⟨a, _ | ⟨b, _ | ⟨c, sss⟩⟩⟩
  · simp at hlen
  · simp at hlen
  · have h1 := hall _ List.mem_cons_self
    simp [OneStage.strata, strataSizes] at h1
    subst h1
    simp at hflat
  · simp at hlen

end Properties.SampledProofs.SecurityProofs
