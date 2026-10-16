import Proofs.Flock.Claims.Sampling.Select.Bounds

/-!
# `Slots.select` misses a set of slots no more often than the claims' `P_a` (`Properties.SampledProofs.DrawMisses`)

* `misses_layout`: a draw that checks no proof unit at a slot of `B` leaves `B` no input slot, and proves no unit at a
  population slot of `B`.
* `pa_slots`: the design's `P_a` on `Slots.layout`'s partition, the population's first stage times the input class's.
* `drawMisses`: the draw is accepted and misses `B` with probability at most `missBound`.
-/

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit Finset Claims.Sampling

/-- The population's replay units, slot by slot. -/
def popRUs (pop : Array (Array Slots.RU)) : List Slots.RU := pop.toList.flatMap (·.toList)

theorem rus_toList (pop ic : Array (Array Slots.RU)) :
    (({ classes := #[pop, ic] } : Slots.Layout).rus).toList = popRUs pop ++ popRUs ic := by
  simp [Slots.Layout.rus, popRUs, List.flatten_append]
  rfl

/-- The population's units, slot by slot. -/
def popUnits (S : Array (String × Array (ℕ × ℕ))) : List ℕ := S.toList.flatMap fun p => expand p.2.toList

theorem laysOut_units : ∀ (S : List (String × Array (ℕ × ℕ))) (pop : List (Array Slots.RU)), List.Forall₂ LaysOut S pop →
    (pop.flatMap (·.toList)).map (·.unit) = (S.flatMap fun p => expand p.2.toList).map some ∧
      (∀ r ∈ pop.flatMap (·.toList), r.pus.size = 1) ∧ pop.map (·.size) = S.map fun p => rsSize p.2
  | [], [], .nil => ⟨rfl, fun r hr => absurd hr (by simp), rfl⟩
  | p :: S, s :: pop, .cons ⟨hu, hp⟩ h => by
    obtain ⟨ih1, ih2, ih3⟩ := laysOut_units S pop h
    refine ⟨by simp only [List.flatMap_cons, List.map_append, hu, ih1], fun r hr => ?_, ?_⟩
    · simp only [List.flatMap_cons, List.mem_append] at hr
      rcases hr with hr | hr
      · exact hp r hr
      · exact ih2 r hr
    · have : s.size = rsSize p.2 := by
        have := congrArg List.length hu
        simp only [List.length_map, Array.length_toList] at this
        exact this
      simp only [List.map_cons, this, ih3]

theorem inputClass_rus (d : Flock.TemplateQuery.Desc) : ∀ r ∈ popRUs (inputClass d), r.unit = none ∧ r.pus.size = 1 := by
  intro r hr
  unfold popRUs inputClass at hr
  dsimp only at hr
  split at hr
  · simp at hr
  · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, Array.toList_map, List.mem_map] at hr
    obtain ⟨g, -, rfl⟩ := hr
    exact ⟨rfl, rfl⟩

/-- **A draw that misses `B`**, over a layout of the population's strata `S` and an input class: no slot of `B` is an
input slot, and no unit at a population slot of `B` is proved. -/
theorem misses_layout (S : Array (String × Array (ℕ × ℕ))) (pop ic : Array (Array Slots.RU))
    (hlays : List.Forall₂ LaysOut S.toList pop.toList)
    (hic : ∀ r ∈ popRUs ic, r.unit = none ∧ r.pus.size = 1) (proved : Array ℕ) (B : Finset ℕ)
    (h : Misses B (Slots.checked { classes := #[pop, ic] } proved)) :
    window (popUnits S).length (popRUs ic).length B = ∅ ∧
      ∀ v ∈ window 0 (popUnits S).length B, (popUnits S).getD v 0 ∉ proved := by
  obtain ⟨hu, hp, -⟩ := laysOut_units S.toList pop.toList hlays
  replace hu : (popRUs pop).map (·.unit) = (popUnits S).map some := hu
  replace hp : ∀ r ∈ popRUs pop, r.pus.size = 1 := hp
  have hlen : (popRUs pop).length = (popUnits S).length := by
    rw [← List.length_map (f := fun r : Slots.RU => r.unit), hu, List.length_map]
  have hrus := rus_toList pop ic
  constructor
  · refine Finset.eq_empty_of_forall_notMem fun j hj => ?_
    have hjl : j < (popRUs ic).length := Finset.mem_range.1 (window_subset _ _ _ hj)
    refine h _ ((mem_window hjl).1 hj) 0 ((mem_checked _ _ _ _).2 ?_)
    have hv : (popUnits S).length + j < ({ classes := #[pop, ic] } : Slots.Layout).rus.size := by
      rw [← Array.length_toList, hrus, List.length_append]; omega
    have hr : ({ classes := #[pop, ic] } : Slots.Layout).rus[(popUnits S).length + j] = (popRUs ic)[j] := by
      rw [← Array.getElem_toList, List.getElem_of_eq hrus, List.getElem_append_right (by omega)]
      congr 1; omega
    refine ⟨hv, ?_, ?_⟩
    · rw [hr, (hic _ (List.getElem_mem _)).1]; rfl
    · rw [hr, (hic _ (List.getElem_mem _)).2]; omega
  · intro v hv hpv
    have hvl : v < (popUnits S).length := Finset.mem_range.1 (window_subset _ _ _ hv)
    rw [mem_window hvl, zero_add] at hv
    refine h v hv 0 ((mem_checked _ _ _ _).2 ?_)
    have hvr : v < ({ classes := #[pop, ic] } : Slots.Layout).rus.size := by
      rw [← Array.length_toList, hrus, List.length_append]; omega
    have hr : ({ classes := #[pop, ic] } : Slots.Layout).rus[v] = (popRUs pop)[v] := by
      rw [← Array.getElem_toList, List.getElem_of_eq hrus, List.getElem_append_left]
    have hunit : (popRUs pop)[v].unit = some ((popUnits S).getD v 0) := by
      have := List.getElem_of_eq hu (by rw [List.length_map]; omega : v < ((popRUs pop).map (·.unit)).length)
      rw [List.getElem_map, List.getElem_map] at this
      rw [List.getD_eq_getElem _ _ hvl]
      exact this
    refine ⟨hvr, ?_, ?_⟩
    · rw [hr, hunit]
      simpa using hpv
    · have := hp (popRUs pop)[v] (List.getElem_mem _)
      rw [hr, this]; omega

theorem slots_length (c : Array (Array Slots.RU)) :
    (c.toList.map fun s => s.toList.map fun r => r.pus.toList).flatten.length = (popRUs c).length := by
  simp [popRUs, List.length_flatten, List.length_flatMap, Function.comp_def]

/-- **The design's `P_a` on `Slots.layout`'s partition**: the population's first stage on its window of `B`, times the
input class's `all`. -/
theorem pa_slots (first : OneStage) (pop ic : Array (Array Slots.RU)) (B : Finset ℕ) :
    (⟨[⟨first, none⟩, ⟨.all, none⟩]⟩ : Design).Pa (slotPartition { classes := #[pop, ic] }) B =
      first.Pa (pop.toList.map (·.size)) (window 0 (popRUs pop).length B) *
        (if window (popRUs pop).length (popRUs ic).length B = ∅ then 1 else 0) := by
  unfold Design.Pa slotPartition
  simp only [List.map_cons, List.map_nil, classesPa, Rule.Pa_none (r := ⟨first, none⟩) rfl,
    Rule.Pa_none (r := ⟨.all, none⟩) rfl, strataSizes, slots_length, zero_add, mul_one]
  congr 2
  simp [Function.comp_def]

/-- `slotDesign`'s steps: the population, the rule's law and its expansion, and the design they give. -/
theorem slotDesign_ok (rule : Select.Rule) (d : Flock.TemplateQuery.Desc) (query : Lean.Json) (table : Option Lean.Json)
    (D : Design) (h : slotDesign rule d query table = .ok D) :
    ∃ n named derived, Partition.units d query = .ok n ∧ rule.lawOver n = .ok named ∧
      Draw.deriveLaw d query named table = .ok derived ∧
      ((rule = .all ∧ D = ⟨[⟨.all, none⟩, ⟨.all, none⟩]⟩) ∨
        (rule ≠ .all ∧ ∃ parsed, Draw.flockLaw (lawObj named derived) n = .ok parsed ∧
          D = ⟨[⟨oneStageOf (derived.getD parsed), none⟩, ⟨.all, none⟩]⟩)) := by
  unfold slotDesign at h
  obtain ⟨n, hn, h⟩ := bind_eq_ok.1 h
  obtain ⟨named, hnamed, h⟩ := bind_eq_ok.1 h
  obtain ⟨derived, hd, h⟩ := bind_eq_ok.1 h
  refine ⟨n, named, derived, hn, hnamed, hd, ?_⟩
  split at h
  · rw [pure_bind] at h
    exact Or.inl ⟨rfl, (Except.ok.inj h).symm⟩
  · rename_i hne
    obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
    rw [pure_bind] at h
    exact Or.inr ⟨fun e => hne e, parsed, hp, (Except.ok.inj h).symm⟩

theorem units_eq (d : Flock.TemplateQuery.Desc) (query : Lean.Json) (rs : Array Flock.TemplateQuery.TRange)
    (he : Partition.evaluate d query = .ok rs) : Partition.units d query = .ok (Flock.TemplateQuery.population rs) := by
  unfold Partition.units; rw [he]; rfl

theorem strata_eq (d : Flock.TemplateQuery.Desc) (query : Lean.Json) (rs : Array Flock.TemplateQuery.TRange)
    (he : Partition.evaluate d query = .ok rs) :
    Partition.strata d query = .ok (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))) := by
  unfold Partition.strata; rw [he]; rfl

theorem length_popUnits (S : Array (String × Array (ℕ × ℕ))) : (popUnits S).length = (S.toList.map fun p => rsSize p.2).sum := by
  simp [popUnits, rsSize, List.length_flatMap]

/-- The units at the slots of `W`, a set of as many units of `[0, n)` when the population's units are `[0, n)`, each once. -/
theorem image_units (U0 : List ℕ) (hnd : U0.Nodup) (hlt : ∀ x ∈ U0, x < U0.length) (W : Finset ℕ)
    (hW : W ⊆ range U0.length) :
    (W.image fun v => U0.getD v 0) ⊆ range U0.length ∧ (W.image fun v => U0.getD v 0).card = W.card := by
  refine ⟨fun u hu => ?_, card_image_of_injOn fun a ha b hb hab => ?_⟩
  · obtain ⟨v, hv, rfl⟩ := mem_image.1 hu
    have hvl := mem_range.1 (hW hv)
    rw [List.getD_eq_getElem _ _ hvl]
    exact mem_range.2 (hlt _ (List.getElem_mem _))
  · have ha' := mem_range.1 (hW ha)
    have hb' := mem_range.1 (hW hb)
    simp only [List.getD_eq_getElem _ _ ha', List.getD_eq_getElem _ _ hb'] at hab
    exact (List.Nodup.getElem_inj_iff hnd).1 hab

theorem units_of_forall₂ : ∀ (S : List (String × Array (ℕ × ℕ))) (ws : List Flock.Draw.WorkStratum),
    List.Forall₂ (fun p (w : Flock.Draw.WorkStratum) => w.units = p.2) S ws → ws.map (·.units) = S.map (·.2)
  | [], [], .nil => rfl
  | _ :: S, _ :: ws, .cons h hs => by simp only [List.map_cons, h, units_of_forall₂ S ws hs]

theorem totalWork_eq (ws : Array Flock.Draw.WorkStratum) :
    Flock.Draw.totalWork ws = (ws.toList.map fun s => s.size * s.work).sum := by
  unfold Flock.Draw.totalWork
  rw [← Array.foldl_toList]
  suffices ∀ (l : List Flock.Draw.WorkStratum) (a : ℕ),
      l.foldl (fun acc s => acc + s.work * s.size) a = a + (l.map fun s => s.size * s.work).sum from
    (this _ 0).trans (Nat.zero_add _)
  intro l
  induction l with
  | nil => simp
  | cons s l ih => intro a; simp only [List.foldl_cons, ih, List.map_cons, List.sum_cons, Nat.mul_comm s.work]; omega

/-- **The population's first stage misses `W`** with probability at most the claims' `P_a(W)` of the law it samples, on
the population's strata: the units `drawUnits` draws hold none at a slot of `W`. -/
theorem law_bound (d : Flock.TemplateQuery.Desc) (query : Lean.Json) (rs : Array Flock.TemplateQuery.TRange)
    (he : Partition.evaluate d query = .ok rs) (pop : Array (Array Slots.RU))
    (hlays : List.Forall₂ LaysOut (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))).toList
      pop.toList)
    (named : Lean.Json) (table : Option Lean.Json) (derived : Option Flock.Draw.Law) (parsed : Flock.Draw.Law)
    (hd : Draw.deriveLaw d query named table = .ok derived)
    (hp : Draw.flockLaw (lawObj named derived)
      (popUnits (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)))).length = .ok parsed)
    (W : Finset ℕ) (hW : W ⊆ range (popUnits (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)))).length)
    (L : ℕ) :
    Claims.Pr (fun src : Fin L → Fin 256 => ∃ units read,
      Draw.drawUnits (derived.getD parsed)
        (popUnits (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)))).length (bytesOf src) =
          .ok (units, read) ∧
        ∀ v ∈ W, (popUnits (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)))).getD v 0 ∉ units) ≤
      (oneStageOf (derived.getD parsed)).Pa (pop.toList.map (·.size)) W := by
  set S := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)) with hS
  have hperm : (popUnits S).Perm (List.range' 0 (Flock.TemplateQuery.population rs)) :=
    strataOf_perm _ _ (partition_evaluate_tiles d query rs he)
  have hlen : (popUnits S).length = Flock.TemplateQuery.population rs := by rw [hperm.length_eq, List.length_range']
  have hnd : (popUnits S).Nodup := hperm.nodup_iff.2 List.nodup_range'
  have hlt : ∀ x ∈ popUnits S, x < (popUnits S).length := fun x hx => by
    rw [hlen]; have := List.mem_range'.1 (hperm.subset hx); omega
  have hns : pop.toList.map (·.size) = S.toList.map fun p => rsSize p.2 := (laysOut_units _ _ hlays).2.2
  have hsum : (pop.toList.map (·.size)).sum = (popUnits S).length := by rw [hns, length_popUnits]
  rcases deriveLaw_cases d query named table derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S', k, law', hS', rfl, hlaw⟩
  · obtain ⟨hU, hcard⟩ := image_units _ hnd hlt W hW
    have hmiss : ∀ units : Array ℕ, (∀ v ∈ W, (popUnits S).getD v 0 ∉ units) →
        ∀ u ∈ W.image fun v => (popUnits S).getD v 0, u ∉ units := fun units hm u hu => by
      obtain ⟨v, hv, rfl⟩ := mem_image.1 hu; exact hm v hv
    rcases flockLaw_cases _ _ _ hp with ⟨k, rfl, hk⟩ | ⟨a, b, rfl, hb, hab⟩ | hk | hk
    · refine le_trans (Claims.Pr_mono fun src ⟨units, read, hu, hm⟩ => ⟨units, read, hu, hmiss units hm⟩)
        ((one_bound _ _ hU L).1 k hk) |>.trans_eq ?_
      simp only [Option.getD_none, oneStageOf, OneStage.Pa, hsum, hcard]
    · refine le_trans (Claims.Pr_mono fun src ⟨units, read, hu, hm⟩ => ⟨units, read, hu, hmiss units hm⟩)
        ((one_bound _ _ hU L).2 a b hb hab) |>.trans_eq ?_
      simp only [Option.getD_none, oneStageOf, OneStage.Pa, hcard]
    · exact absurd hk hk1
    · exact absurd hk hk2
  · have hS'eq : S' = S := by rw [strata_eq d query rs he] at hS'; exact (Except.ok.inj hS').symm
    subst hS'eq
    rcases hlaw with hder | ⟨t, tab, ht, hwork⟩
    · obtain rfl := derive_ok k _ law' hder
      refine le_trans (Claims.Pr_mono fun src ⟨units, read, hu, hm⟩ => ⟨units, read, hu, fun v hv => ?_⟩)
        (strata_bound _ _ (Or.inl ⟨k, rfl⟩) _ W L) |>.trans_eq ?_
      · rw [hns]
        simp [oneStageOf, OneStage.Pa, stratum_size, Flock.Draw.stratumK, strataSize, Function.comp_def]
      · simpa [popUnits, List.flatMap_map] using hm v hv
    · obtain ⟨ws, hF, rfl⟩ := deriveWork_ok k _ tab law' hwork
      have hunits := units_of_forall₂ _ _ hF
      have hsz : S.toList.map (fun p => rsSize p.2) = ws.toList.map (·.size) := by
        have := congrArg (List.map rsSize) hunits
        simp only [List.map_map, Function.comp_def] at this
        rw [← this]
        simp [stratum_size]
      refine le_trans (Claims.Pr_mono fun src ⟨units, read, hu, hm⟩ => ⟨units, read, hu, fun v hv => ?_⟩)
        (strata_bound _ _ (Or.inr ⟨k, _, rfl, rfl⟩) _ W L) |>.trans_eq ?_
      · rw [hns, hsz]
        simp only [Option.getD_some, oneStageOf, OneStage.Pa, Array.toList_map, List.map_map, Function.comp_def,
          List.zip_map', List.map_map, totalWork_eq, Flock.Draw.workK]
        rfl
      · have h1 : (ws.toList.map (·.units)).flatMap (fun u => expand u.toList) =
            (S.toList.map (·.2)).flatMap (fun u => expand u.toList) := by rw [hunits]
        simp only [List.flatMap_map] at h1
        simpa [popUnits, List.flatMap_map, Function.comp_def, h1] using hm v hv

end Properties.SampledProofs.SecurityProofs
