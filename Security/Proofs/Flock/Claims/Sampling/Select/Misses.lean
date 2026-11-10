module

public import Proofs.Flock.Claims.Sampling.Select.Bounds
public import Proofs.Flock.Claims.Sampling.Select.Steps
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Std.Tactic.BVDecide.Normalize.Prop
import VCVio.Prelude.Core

/-!
# `Slots.select` misses a set of slots no more often than the claims' `P_a` (`Properties.Spot.DrawMisses`)

* `slotPartition_strata`: `Slots.layout`'s partition is one class, in the registration's strata.
* `pa_layout`: the design's `P_a` on that partition, its one rule over the strata on `B`'s slots among the units.
* `law_bound`: the sampling design `Select.select` samples misses a set of units no more often than its `P_a` over the strata.
* `drawMisses`: the sample is accepted and misses `B` with probability at most `missBound`.
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit Finset Claims.Sampling

/-- The population's units, slot by slot. -/
def popUnits (S : Array (String × Array (ℕ × ℕ))) : List ℕ := S.toList.flatMap fun p => expand p.2.toList

/-- **`Slots.layout`'s partition**, where its strata `ns` cover the units: one class, of strata sizes `ns` and of
`Σ ns` proof blocks. -/
theorem slotPartition_strata (f : FilePart.FilePart) (ns : Array ℕ) (hn : f.n = ns.toList.sum) :
    ∃ c, slotPartition (Slots.layout f ns) = ⟨[c]⟩ ∧ strataSizes c = ns.toList ∧ c.flatten.length = ns.toList.sum := by
  have hG : ((List.range f.n).map fun u => puGates (Slots.blockOf f u)).length = ns.toList.sum := by simp [hn]
  exact ⟨_, slotPartition_layout f ns, List.map_splitLengths_length _ _ hG.ge,
    by rw [List.flatten_splitLengths _ _ hG.le, hG]⟩

/-- **The design's `P_a` on `Slots.layout`'s partition**: its one rule over the strata `ns`, on `B`'s slots among the
`Σ ns` units. -/
theorem pa_layout (first : OneStage) (f : FilePart.FilePart) (ns : Array ℕ) (hn : f.n = ns.toList.sum) (B : Finset ℕ) :
    (⟨[first]⟩ : Design).Pa (slotPartition (Slots.layout f ns)) B = first.Pa ns.toList (window 0 ns.toList.sum B) := by
  obtain ⟨c, hc, hs, hl⟩ := slotPartition_strata f ns hn
  rw [hc]
  simp only [Design.Pa, classesPa, hs, hl, mul_one]

/-- `slotDesign`'s steps: the population, the rule's law and its expansion, and the design they give. -/
theorem slotDesign_ok (rule : Select.Rule) (d : Program) (query : Lean.Json) (table : Option Lean.Json)
    (D : Design) (h : slotDesign rule d query table = .ok D) :
    ∃ n named derived, Partition.units d query = .ok n ∧ rule.lawOver n = .ok named ∧
      Draw.deriveLaw d query named table = .ok derived ∧
      ((rule = .all ∧ D = ⟨[.all]⟩) ∨
        (rule ≠ .all ∧ ∃ parsed, Draw.flockLaw (lawObj named derived) n = .ok parsed ∧
          D = ⟨[oneStageOf (derived.getD parsed)]⟩)) := by
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

theorem units_eq (d : Program) (query : Lean.Json) (rs : Array Flock.TemplateQuery.TRange)
    (he : Partition.evaluate d query = .ok rs) : Partition.units d query = .ok (Flock.TemplateQuery.population rs) := by
  unfold Partition.units; rw [he]; rfl

theorem strata_eq (d : Program) (query : Lean.Json) (rs : Array Flock.TemplateQuery.TRange)
    (he : Partition.evaluate d query = .ok rs) :
    Partition.strata d query = .ok (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))) := by
  unfold Partition.strata; rw [he]; rfl

theorem strata_ok (d : Program) (query : Lean.Json) (S : Array (String × Array (ℕ × ℕ)))
    (h : Partition.strata d query = .ok S) :
    ∃ rs, Partition.evaluate d query = .ok rs ∧ S = Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)) := by
  unfold Partition.strata at h
  obtain ⟨rs, he, h⟩ := bind_eq_ok.1 h
  exact ⟨rs, he, (Except.ok.inj h).symm⟩

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

/-- **The population's first stage misses `W`** with probability at most the claims' `P_a(W)` of the sampling design it samples, on
the population's strata: the units `drawUnits` samples hold none at a slot of `W`. -/
theorem law_bound (d : Program) (query : Lean.Json) (rs : Array Flock.TemplateQuery.TRange)
    (he : Partition.evaluate d query = .ok rs)
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
      (oneStageOf (derived.getD parsed)).Pa
        ((Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))).toList.map fun p => rsSize p.2) W := by
  set S := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)) with hS
  have hperm : (popUnits S).Perm (List.range' 0 (Flock.TemplateQuery.population rs)) :=
    strataOf_perm _ _ (partition_evaluate_tiles d query rs he)
  have hlen : (popUnits S).length = Flock.TemplateQuery.population rs := by rw [hperm.length_eq, List.length_range']
  have hnd : (popUnits S).Nodup := hperm.nodup_iff.2 List.nodup_range'
  have hlt : ∀ x ∈ popUnits S, x < (popUnits S).length := fun x hx => by
    rw [hlen]; have := List.mem_range'.1 (hperm.subset hx); omega
  have hsum : (S.toList.map fun p => rsSize p.2).sum = (popUnits S).length := (length_popUnits S).symm
  rcases deriveLaw_cases d query named table derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S', k, law', hS', rfl, hlaw⟩
  · obtain ⟨hU, hcard⟩ := image_units _ hnd hlt W hW
    have hmiss : ∀ units : Array ℕ, (∀ v ∈ W, (popUnits S).getD v 0 ∉ units) →
        ∀ u ∈ W.image fun v => (popUnits S).getD v 0, u ∉ units := fun units hm u hu => by
      obtain ⟨v, hv, rfl⟩ := mem_image.1 hu; exact hm v hv
    rcases flockLaw_cases _ _ _ hp with ⟨k, rfl, hk, -⟩ | ⟨a, b, rfl, hb, hab⟩ | hk | hk
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
      · simp [oneStageOf, OneStage.Pa, stratum_size, Flock.Draw.stratumK, strataSize, Function.comp_def]
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
      · rw [hsz]
        simp only [Option.getD_some, oneStageOf, OneStage.Pa, Array.toList_map, List.map_map, Function.comp_def,
          List.zip_map', List.map_map, totalWork_eq, Flock.Draw.workK]
        rfl
      · have h1 : (ws.toList.map (·.units)).flatMap (fun u => expand u.toList) =
            (S.toList.map (·.2)).flatMap (fun u => expand u.toList) := by rw [hunits]
        simp only [List.flatMap_map] at h1
        simpa [popUnits, List.flatMap_map, Function.comp_def, h1] using hm v hv

/-- **`Slots.select` misses `B` no more often than the claims' `P_a(B)`** (`Properties.Spot.DrawMisses`). -/
theorem drawMisses : DrawMisses := by
  intro rule cf d query table cm L B
  set E := fun src : Fin L → Fin 256 => (Slots.select rule cf d query table cm (bytesOf src)).isOk = true ∧
    Misses B (checkedOf (Slots.select rule cf d query table cm (bytesOf src))) with hE
  have hev : ∀ src, E src → ∃ l c, Slots.select rule cf d query table cm (bytesOf src) = .ok (l, c) ∧ Misses B c := by
    rintro src ⟨hok, hm⟩
    cases h : Slots.select rule cf d query table cm (bytesOf src) with
    | error e => rw [h] at hok; exact absurd hok (by simp [Except.isOk, Except.toBool])
    | ok x => rw [h] at hm; exact ⟨x.1, x.2, rfl, hm⟩
  have hzero : (∀ src, ¬ E src) → Claims.Pr E = 0 := fun h => by
    rw [Claims.Pr_congr fun src => iff_false_intro (h src), Claims.Pr_false]
  unfold missBound
  cases hl : Slots.fileLayout cf d query with
  | error e =>
    refine le_of_eq (hzero fun src hs => ?_)
    obtain ⟨l', c, hs, -⟩ := hev src hs
    have := (slots_select_ok _ _ _ _ _ _ _ _ _ hs).1
    rw [hl] at this
    exact absurd this (by simp)
  | ok l =>
  cases hD : slotDesign rule d query table with
  | error e =>
    refine le_of_eq (hzero fun src hs => ?_)
    obtain ⟨l', c, hs, -⟩ := hev src hs
    obtain ⟨-, draw, proved, read, hsel, -⟩ := slots_select_ok _ _ _ _ _ _ _ _ _ hs
    obtain ⟨n, named, derived, hn, hnamed, hd, _units, _hsub, hcase⟩ := select_ok _ _ _ _ _ _ _ _ _ hsel
    rcases hcase with ⟨rfl, -⟩ | ⟨hne, parsed, -, hp, -⟩
    · rw [slotDesign_all d query table n named derived hn hnamed hd] at hD
      exact absurd hD (by simp)
    · rw [slotDesign_law rule d query table n named derived parsed hne hn hnamed hd hp] at hD
      exact absurd hD (by simp)
  | ok D =>
  dsimp only
  obtain ⟨S₀, ns, hS₀, hns, rfl⟩ := fileLayout_ok cf d query l hl
  obtain ⟨hpop, hsz⟩ := sizes_ok S₀ ns hns
  obtain ⟨rs, he, rfl⟩ := strata_ok d query S₀ hS₀
  set S := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)) with hS
  have hpopU : popUnits S = List.range' 0 ns.toList.sum := hpop
  have hlenU : (popUnits S).length = ns.toList.sum := by rw [hpopU, List.length_range']
  have hfn : (FilePart.ofCircuit cf ns.toList.sum).n = ns.toList.sum := rfl
  have hperm : (popUnits S).Perm (List.range' 0 (Flock.TemplateQuery.population rs)) :=
    strataOf_perm _ _ (partition_evaluate_tiles d query rs he)
  obtain ⟨n, named, derived, hn, hnamed, hd, hDc⟩ := slotDesign_ok rule d query table D hD
  have hnU : n = (popUnits S).length := by
    rw [units_eq d query rs he] at hn
    rw [hperm.length_eq, List.length_range']
    exact (Except.ok.inj hn).symm
  subst hnU
  have hstep : ∀ src, E src →
      ∃ units : Array ℕ, (∀ v ∈ window 0 (popUnits S).length B, (popUnits S).getD v 0 ∉ units) ∧
        ((rule = .all ∧ units = Array.range (popUnits S).length) ∨
          (rule ≠ .all ∧ ∃ parsed read', Draw.flockLaw (lawObj named derived) (popUnits S).length = .ok parsed ∧
            Draw.drawUnits (derived.getD parsed) (popUnits S).length (bytesOf src) = .ok (units, read'))) := by
    intro src hs
    obtain ⟨l', c, hs, hm⟩ := hev src hs
    obtain ⟨hl', draw, proved, read, hsel, rfl⟩ := slots_select_ok _ _ _ _ _ _ _ _ _ hs
    rw [hl] at hl'
    obtain rfl := (Except.ok.inj hl')
    have hW0 : ∀ v ∈ window 0 (popUnits S).length B, (popUnits S).getD v 0 ∉ proved := by
      intro v hv hpv
      have hvl : v < (popUnits S).length := Finset.mem_range.1 (window_subset _ _ _ hv)
      rw [mem_window hvl, zero_add] at hv
      have hgv : (popUnits S).getD v 0 = v := by
        rw [hpopU]
        have hv' : v < (List.range' 0 ns.toList.sum).length := by rw [List.length_range']; omega
        rw [List.getD_eq_getElem _ _ hv']
        simp
      rw [hgv] at hpv
      exact hm v hv ((mem_checked_layout _ ns hfn.le proved v).2 ⟨by rw [hfn]; omega, hpv⟩)
    obtain ⟨n', named', derived', hn', hnamed', hd', units, hsub, hcase⟩ := select_ok _ _ _ _ _ _ _ _ _ hsel
    rw [hn] at hn'
    obtain rfl := Except.ok.inj hn'
    rw [hnamed] at hnamed'
    obtain rfl := Except.ok.inj hnamed'
    rw [hd] at hd'
    obtain rfl := Except.ok.inj hd'
    exact ⟨units, fun v hv hu => hW0 v hv (hsub _ hu), hcase⟩
  rcases hDc with ⟨hrule, rfl⟩ | ⟨hne, parsed, hp, rfl⟩
  · rw [pa_layout _ _ ns hfn, ← hlenU]
    by_cases hW0 : window 0 (popUnits S).length B = ∅
    · simp only [OneStage.Pa, hW0, ↓reduceIte]
      exact Claims.Pr_le_one _
    · simp only [OneStage.Pa, hW0, ↓reduceIte]
      refine le_of_eq (hzero fun src hs => ?_)
      obtain ⟨units, hW0', hcase⟩ := hstep src hs
      rcases hcase with ⟨-, rfl⟩ | ⟨hr, -⟩
      · obtain ⟨v, hv⟩ := Finset.nonempty_iff_ne_empty.2 hW0
        have hvl := Finset.mem_range.1 (window_subset _ _ _ hv)
        refine hW0' v hv (Array.mem_range.2 ?_)
        rw [List.getD_eq_getElem _ _ hvl]
        have := List.mem_range'.1 (hperm.subset (List.getElem_mem hvl))
        rw [hperm.length_eq, List.length_range']
        omega
      · exact absurd hrule hr
  · rw [pa_layout _ _ ns hfn, ← hlenU, show ns.toList = S.toList.map (fun p => rsSize p.2) from hsz]
    refine le_trans (Claims.Pr_mono fun src hs => ?_)
      (law_bound d query rs he named table derived parsed hd hp _ (window_subset _ _ _) L)
    obtain ⟨units, hW0, hcase⟩ := hstep src hs
    rcases hcase with ⟨hr, -⟩ | ⟨-, parsed', read', hp', hu⟩
    · exact absurd hr hne
    · rw [hp] at hp'
      obtain rfl := Except.ok.inj hp'
      exact ⟨units, read', hu, hW0⟩

end Properties.Spot.SecurityProofs
