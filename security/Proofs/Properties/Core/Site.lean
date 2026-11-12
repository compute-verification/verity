module

public import Mathlib.Data.List.Perm.Basic
public import Properties.Core.Site
public import Proofs.Core.Site

/-!
# The Site's code is the spec's: the proofs

Each translation of `Properties.Core.Site` is injective, the code's functions of the Site commute with them
(`kind?_site`, `held_site`, `far_site`, `host?_site`, `reached_site`), so each of a site entry's checks holds in code
exactly when the spec's holds of what it denotes (`fresh_iff` … `reached_iff`), and its refusal is the spec's; so is
the completeness a log is held to at its end (`complete_iff`). The check of a log then follows the spec's validity one
entry at a time (`go_says`), by `Verity.Registration.valid_append_singleton`.
-/

@[expose] public section

open scoped NNRat

namespace Properties.Site

open Core.Site.Registration (Digests Auditor Composite)

/-! ## The translations -/

theorem kind_eq_iff {k : Core.Site.Kind} {k' : Verity.Site.Kind} : kind k = k' ↔ k = ofKind k' := by
  cases k <;> cases k' <;> decide

theorem ofKind_kind (k : Core.Site.Kind) : ofKind (kind k) = k := by cases k <;> rfl

theorem kind_injective : Function.Injective kind := fun a b h => by
  rw [← ofKind_kind a, ← ofKind_kind b, h]

theorem owner_eq_iff {o : Core.Site.Owner} {o' : Verity.Site.Owner} : owner o = o' ↔ o = ofOwner o' := by
  cases o <;> cases o' <;> decide

theorem ofTier_tier (t : Core.Site.Tier) : ofTier (tier t) = t := by cases t <;> rfl

theorem ofOwner_owner (o : Core.Site.Owner) : ofOwner (owner o) = o := by cases o <;> rfl

theorem composite_eq_iff {c : Composite} {c' : Verity.Registration.Composite} :
    composite c = c' ↔ c = ofComposite c' := by
  cases c <;> cases c' <;> decide

theorem ofComposite_composite (c : Composite) : ofComposite (composite c) = c := by cases c <;> rfl

@[simp] theorem ofOperandClass_operandClass (c : Core.Site.OperandClass) : ofOperandClass (operandClass c) = c := by
  cases c <;> rfl

@[simp] theorem operandClass_ofOperandClass (c : Verity.OperandClass) : operandClass (ofOperandClass c) = c := by
  cases c <;> rfl

@[simp] theorem ofFigure_figure (q : Core.Site.Figure) : ofFigure (figure q) = q := rfl

@[simp] theorem figure_ofFigure (q : ℚ≥0) : figure (ofFigure q) = q := rfl

theorem ofDeviceFigures_deviceFigures (d : Core.Site.DeviceFigures) : ofDeviceFigures (deviceFigures d) = d := by
  cases d; simp [deviceFigures, ofDeviceFigures]

theorem deviceFigures_ofDeviceFigures (d : Verity.DeviceFigures) : deviceFigures (ofDeviceFigures d) = d := by
  cases d; simp [deviceFigures, ofDeviceFigures]

theorem ofSheet_sheet (k : Core.Site.Sheet) : ofSheet (sheet k) = k := by
  cases k; simp [sheet, ofSheet, ofDeviceFigures_deviceFigures]

theorem sheet_ofSheet (k : Verity.Site.Sheet) : sheet (ofSheet k) = k := by
  cases k; simp [sheet, ofSheet, deviceFigures_ofDeviceFigures]

theorem sheet_eq_iff {k : Core.Site.Sheet} {k' : Verity.Site.Sheet} : sheet k = k' ↔ k = ofSheet k' := by
  constructor
  · rintro rfl; rw [ofSheet_sheet]
  · rintro rfl; exact sheet_ofSheet k'

theorem ofShared_shared (p : Core.Site.Tier × Nat) : ofShared (shared p) = p := by
  cases p; simp only [shared, ofShared, ofTier_tier]

theorem figure_injective : Function.Injective figure := fun a b h => by
  rw [← ofFigure_figure a, ← ofFigure_figure b, h]

theorem shared_injective : Function.Injective shared := fun a b h => by
  rw [← ofShared_shared a, ← ofShared_shared b, h]

theorem ofLoc_loc (x : Core.Site.Loc) : ofLoc (loc x) = x := by
  cases x
  simp only [loc, ofLoc, ofKind_kind, ofOwner_owner, Option.map_map, Function.comp_def, ofTier_tier, Option.map_id',
    List.map_map, ofShared_shared, List.map_id', ofFigure_figure]

theorem ofLink_link (k : Core.Site.Link) : ofLink (link k) = k := rfl

theorem ofEntry_entry (e : Core.Site.Entry) : ofEntry (entry e) = e := by
  cases e
  simp only [entry, ofEntry, List.map_map, Function.comp_def, ofSheet_sheet, ofLoc_loc, ofLink_link, List.map_id']

theorem ofRegEntry_regEntry (e : Core.Site.Registration.Entry) : ofRegEntry (regEntry e) = e := by
  cases e
  simp only [regEntry, ofRegEntry, Verity.Registration.Entry.previous, Verity.Registration.Entry.added, ofEntry_entry]

theorem sheet_injective : Function.Injective sheet := fun a b h => by
  rw [← ofSheet_sheet a, ← ofSheet_sheet b, h]

theorem map_sheet_eq_some (o : Option Core.Site.Sheet) (k : Core.Site.Sheet) :
    o.map sheet = some (sheet k) ↔ o = some k :=
  (Option.map_injective sheet_injective).eq_iff (b := some k)

theorem refusal_injective : Function.Injective refusal := by
  intro a b h; cases a <;> cases b <;> first | rfl | cases h

@[simp] theorem loc_id (x : Core.Site.Loc) : (loc x).id = x.id := rfl
@[simp] theorem loc_kind (x : Core.Site.Loc) : (loc x).kind = kind x.kind := rfl
@[simp] theorem loc_parent (x : Core.Site.Loc) : (loc x).parent = x.parent := rfl
@[simp] theorem loc_owner (x : Core.Site.Loc) : (loc x).owner = owner x.owner := rfl
@[simp] theorem loc_sheet (x : Core.Site.Loc) : (loc x).sheet = x.sheet := rfl
@[simp] theorem loc_tier (x : Core.Site.Loc) : (loc x).tier = x.tier.map tier := rfl
@[simp] theorem loc_bytes (x : Core.Site.Loc) : (loc x).bytes = x.bytes := rfl
@[simp] theorem loc_key (x : Core.Site.Loc) : (loc x).key = x.key := rfl
@[simp] theorem loc_scheme (x : Core.Site.Loc) : (loc x).scheme = x.scheme := rfl
@[simp] theorem loc_bw (x : Core.Site.Loc) : (loc x).bw = x.bw := rfl
@[simp] theorem loc_chip (x : Core.Site.Loc) : (loc x).chip = x.chip := rfl
@[simp] theorem loc_shared (x : Core.Site.Loc) : (loc x).shared = x.shared.map shared := rfl
@[simp] theorem loc_drift (x : Core.Site.Loc) : (loc x).drift = x.drift.map figure := rfl
@[simp] theorem loc_timeUnit (x : Core.Site.Loc) : (loc x).timeUnit = x.timeUnit.map figure := rfl
@[simp] theorem loc_timebase (x : Core.Site.Loc) : (loc x).timebase = x.timebase := rfl
@[simp] theorem link_id (k : Core.Site.Link) : (link k).id = k.id := rfl
@[simp] theorem link_ends (k : Core.Site.Link) : (link k).ends = k.ends := rfl
@[simp] theorem link_β (k : Core.Site.Link) : (link k).β = k.β := rfl
@[simp] theorem link_unmediated (k : Core.Site.Link) : (link k).unmediated = k.unmediated := rfl
@[simp] theorem sheet_sku (k : Core.Site.Sheet) : (sheet k).sku = k.sku := rfl
@[simp] theorem sheet_mem (k : Core.Site.Sheet) : (sheet k).mem = k.mem := rfl
@[simp] theorem site_sheets (S : Core.Site) : (site S).sheets = S.sheets.map sheet := rfl
@[simp] theorem site_locs (S : Core.Site) : (site S).locs = S.locs.map loc := rfl
@[simp] theorem site_links (S : Core.Site) : (site S).links = S.links.map link := rfl
@[simp] theorem entry_sheets (e : Core.Site.Entry) : (entry e).sheets = e.sheets.map sheet := rfl
@[simp] theorem entry_locs (e : Core.Site.Entry) : (entry e).locs = e.locs.map loc := rfl
@[simp] theorem entry_links (e : Core.Site.Entry) : (entry e).links = e.links.map link := rfl
@[simp] theorem kind_auditor (k : Core.Site.Kind) : (kind k).auditor = k.auditor := by cases k <;> rfl
@[simp] theorem outside_eq : Verity.Site.outside = Core.Site.outside := rfl

@[simp] theorem digests_sheet (D : Digests) (k : Core.Site.Sheet) : (digests D).sheet (sheet k) = D.sheet k := by
  simp only [digests, ofSheet_sheet]

@[simp] theorem digests_entry (D : Digests) (e : Core.Site.Registration.Entry) :
    (digests D).entry (regEntry e) = D.entry e := by
  simp only [digests, ofRegEntry_regEntry]

@[simp] theorem digests_genesis (D : Digests) (c : Composite) : (digests D).genesis (composite c) = D.genesis c := by
  simp only [digests, ofComposite_composite]

@[simp] theorem auditor_key (A : Auditor) : (auditor A).key = A.key := rfl
@[simp] theorem auditor_scheme (A : Auditor) : (auditor A).scheme = A.scheme := rfl
@[simp] theorem auditor_timebases (A : Auditor) : (auditor A).timebases = A.timebases := rfl

/-- A lookup by id commutes with a translation of the values. -/
theorem lookup_map {α β : Type} (f : α → β) (i : String) (l : List (String × α)) :
    (l.map fun p => (p.1, f p.2)).lookup i = (l.lookup i).map f := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨j, a⟩ := p
    simp only [List.map_cons, List.lookup_cons, ih]
    cases i == j <;> rfl

theorem mem_lookup_map {α β : Type} (f : α → β) (i : String) (l : List (String × α)) (b : β) :
    b ∈ (l.map fun p => (p.1, f p.2)).lookup i ↔ ∃ a ∈ l.lookup i, f a = b := by
  rw [lookup_map]
  cases l.lookup i <;> simp

/-- Two conditionals agree when their conditions and their branches do. -/
theorem ite_iff {p q a b c d : Prop} [Decidable p] [Decidable q] (hp : p ↔ q) (h1 : a ↔ c) (h2 : b ↔ d) :
    (if p then a else b) ↔ if q then c else d := by
  by_cases h : q
  · rw [ite_eq_left h, ite_eq_left (hp.mpr h)]; exact h1
  · rw [ite_eq_right h, ite_eq_right (mt hp.mp h)]; exact h2

theorem regEntry_added (e : Core.Site.Registration.Entry) : (regEntry e).added = entry e.added := by cases e; rfl

theorem regEntry_previous (e : Core.Site.Registration.Entry) : (regEntry e).previous = e.previous := by cases e; rfl

theorem site_add (S : Core.Site) (e : Core.Site.Entry) : site (S.add e) = (site S).add (entry e) := by
  simp [site, entry, Core.Site.add, Verity.Site.add]

/-! ## The Site's functions commute with the translations -/

section Commute

variable (S : Core.Site)

theorem loc?_site (i : String) : (site S).loc? i = (S.loc? i).map loc := by
  simp [Verity.Site.loc?, Core.Site.loc?, List.find?_map, Function.comp_def]

theorem kind?_site (i : String) : (site S).kind? i = (S.kind? i).map kind := by
  unfold Verity.Site.kind? Core.Site.kind?
  by_cases h : i = Core.Site.outside
  · simp [h]; rfl
  · simp [h, loc?_site, Option.map_map, Function.comp_def]

theorem kind?_site_eq_some (i : String) (k : Verity.Site.Kind) :
    (site S).kind? i = some k ↔ S.kind? i = some (ofKind k) := by
  rw [kind?_site, Option.map_eq_some_iff]
  constructor
  · rintro ⟨a, ha, rfl⟩; rw [ofKind_kind]; exact ha
  · intro h; exact ⟨_, h, by rw [kind_eq_iff]⟩

theorem bind_kind?_site (o : Option String) : o.bind (site S).kind? = (o.bind S.kind?).map kind := by
  cases o <;> simp [kind?_site]

theorem map_kind_eq_some (o : Option Core.Site.Kind) (k : Verity.Site.Kind) :
    o.map kind = some k ↔ o = some (ofKind k) := by
  cases o <;> simp [kind_eq_iff]

/-- A search of the locations a code Site denotes is the search of its own, by the predicate's code form. -/
theorem find?_loc (p : Verity.Site.Loc → Bool) (q : Core.Site.Loc → Bool) (h : ∀ x, p (loc x) = q x) :
    ∀ l : List Core.Site.Loc, (l.map loc).find? p = (l.find? q).map loc
  | [] => rfl
  | x :: l => by
    simp only [List.map_cons, List.find?_cons, h, find?_loc p q h l]
    cases q x <;> rfl

theorem timeUnit_site : (site S).timeUnit = S.timeUnit.map figure := by
  unfold Verity.Site.timeUnit Core.Site.timeUnit
  rw [site_locs, find?_loc _ (fun x => decide (x.kind = .site)) (fun x => by simp [kind_eq_iff, ofKind])]
  cases S.locs.find? (fun x => decide (x.kind = .site)) <;> rfl

theorem known_site (i : String) : (site S).Known i ↔ S.Known i := by
  simp [Verity.Site.Known, Core.Site.Known, Function.comp_def]

theorem ids_site : (site S).ids = S.ids := by
  simp [Verity.Site.ids, Core.Site.ids, Function.comp_def]

theorem sheet?_site (D : Digests) (x : Core.Site.Loc) :
    (site S).sheet? (digests D).sheet (loc x) = (S.sheet? D.sheet x).map sheet := by
  simp [Verity.Site.sheet?, Core.Site.sheet?, List.find?_map, Function.comp_def]

theorem held_site (i : String) : (site S).held i = S.held i := by
  simp [Verity.Site.held, Core.Site.held, List.filter_map, Function.comp_def, kind_eq_iff, ofKind]
  rfl

theorem joins_site (k : Core.Site.Link) (a b : Verity.Site.Kind) :
    (site S).Joins (link k) a b ↔ S.Joins k (ofKind a) (ofKind b) := by
  simp [Verity.Site.Joins, Core.Site.Joins, kind?_site_eq_some]

theorem boundary_site (k : Core.Site.Link) : (site S).Boundary (link k) ↔ S.Boundary k := by
  simp [Verity.Site.Boundary, Core.Site.Boundary, joins_site, ofKind]

theorem certified_site (k : Core.Site.Link) : (site S).Certified (link k) ↔ S.Certified k := by
  simp [Verity.Site.Certified, Core.Site.Certified, kind?_site_eq_some, ofKind]

theorem far_site (i : String) : (site S).far i = S.far i := by
  simp [Verity.Site.far, Core.Site.far, List.filter_map, List.flatMap_map, Function.comp_def]
  rfl

theorem host?_site (x : Core.Site.Loc) : (site S).host? (loc x) = S.host? x := by
  simp [Verity.Site.host?, Core.Site.host?, kind?_site_eq_some, ofKind, loc?_site, Option.bind_map,
    Function.comp_def]

theorem reached_site : (site S).reached = S.reached := by
  simp [Verity.Site.reached, Core.Site.reached, List.filter_map, List.flatMap_map, Function.comp_def,
    kind?_site_eq_some, ofKind]
  rfl

end Commute

/-! ## A site entry's checks are the spec's -/

section Checks

variable (D : Digests) (A : Auditor) (c : Composite) (S : Core.Site) (e : Core.Site.Entry)

@[simp] theorem auditor_sheet (sku : String) : (auditor A).sheet sku = (A.sheet sku).map sheet := rfl

theorem parents_iff (o : Option Core.Site.Kind) (k : Core.Site.Kind) :
    o.map kind ∈ (kind k).parents ↔ o ∈ k.parents := by
  cases k <;> rcases o with _ | a <;> (try cases a) <;> decide

theorem perm_iff (l : List (Option Core.Site.Kind)) :
    (l.map (Option.map kind)).Perm [some .host, some .outside] ↔ l.Perm [some .host, some .outside] := by
  have : ([some .host, some .outside] : List (Option Verity.Site.Kind)) =
      [some .host, some .outside].map (Option.map kind) := rfl
  rw [this, List.map_perm_map_iff (Option.map_injective kind_injective)]

theorem fresh_iff : Verity.Site.Entry.Fresh (digests D) (site S) (entry e) ↔ Core.Site.Entry.Fresh D S e := by
  unfold Verity.Site.Entry.Fresh Core.Site.Entry.Fresh
  rw [← site_add, ids_site]
  simp [List.map_map, Function.comp_def]

theorem refs_iff : Verity.Site.Entry.Refs (site S) (entry e) ↔ Core.Site.Entry.Refs S e := by
  unfold Verity.Site.Entry.Refs Core.Site.Entry.Refs
  rw [← site_add]
  simp [known_site]

theorem kinds_iff : Verity.Site.Entry.Kinds (site S) (entry e) ↔ Core.Site.Entry.Kinds S e := by
  unfold Verity.Site.Entry.Kinds Core.Site.Entry.Kinds
  rw [← site_add]
  have hb : ∀ x : Core.Site.Loc,
      x.parent.bind (site (S.add e)).kind? = (x.parent.bind (S.add e).kind?).map kind := by
    intro x; cases x.parent <;> simp [kind?_site]
  have hf : ∀ i, ((site (S.add e)).far i).map (site (S.add e)).kind? =
      (((S.add e).far i).map (S.add e).kind?).map (Option.map kind) := by
    intro i; rw [far_site, List.map_map]; congr 1; funext j; simp [kind?_site]
  simp only [entry_locs, entry_links, site_locs, List.forall_mem_map, loc_parent, loc_kind, loc_tier, loc_id,
    link_ends, link_β, link_unmediated, hb, hf, parents_iff, perm_iff, Option.isSome_map, kind_eq_iff, ofKind]
  simp [Verity.Site.Kind.links, Core.Site.Kind.links, joins_site, ofKind]

theorem sheets_iff :
    Verity.Site.Entry.Sheets (digests D) (auditor A) (site S) (entry e) ↔ Core.Site.Entry.Sheets D A S e := by
  unfold Verity.Site.Entry.Sheets Core.Site.Entry.Sheets
  rw [← site_add]
  simp only [entry_sheets, entry_locs, List.forall_mem_map, auditor_sheet, sheet_sku, Option.isSome_map, loc_kind,
    loc_sheet, site_sheets, List.map_map, Function.comp_def, digests_sheet, kind_eq_iff]
  simp [ofKind]

@[simp] theorem coe_figure (q : Core.Site.Figure) : ((figure q : ℚ≥0) : ℚ) = q.1 := rfl

theorem within_iff (τ e x : Core.Site.Figure) :
    Verity.WithinTolerance (figure τ) (figure e) (figure x) ↔ Core.Site.WithinTolerance τ e x := by
  have hmax : ∀ q : ℚ, ((q.toNNRat : ℚ≥0) : ℚ) = max q 0 := fun _ => rfl
  unfold Verity.WithinTolerance Core.Site.WithinTolerance
  rw [← NNRat.coe_le_coe, ← NNRat.coe_le_coe]
  simp only [NNRat.coe_mul, NNRat.coe_add, NNRat.coe_one, NNRat.sub_def, hmax, coe_figure] <;> exact Iff.rfl

theorem pos_iff (q : Core.Site.Figure) : 0 < figure q ↔ 0 < q.1 := by
  rw [← NNRat.coe_pos, coe_figure]

theorem forall_class {P : Core.Site.OperandClass → Prop} :
    (∀ c : Verity.OperandClass, P (ofOperandClass c)) ↔ ∀ c ∈ Core.Site.OperandClass.all, P c := by
  constructor
  · intro h c _; simpa using h (operandClass c)
  · intro h c; exact h _ (Core.Site.OperandClass.mem_all _)

/-- The code's check of a device's figures is the spec's (`DeviceFigures.Accepted`). -/
theorem accepted_iff (est : Core.Site.DeviceEstimate) (d : Core.Site.DeviceFigures) :
    (deviceFigures d).Accepted ⟨deviceFigures est.figures, ⟨figure est.tolerance⟩⟩ ↔ d.Accepted est := by
  unfold Verity.DeviceFigures.Accepted Core.Site.DeviceFigures.Accepted
  refine and_congr Iff.rfl (and_congr (within_iff est.tolerance est.figures.clock d.clock) (and_congr ?_ ?_))
  · exact (forall_congr' fun c => within_iff est.tolerance (est.figures.peakRate (ofOperandClass c))
      (d.peakRate (ofOperandClass c))).trans
      (forall_class (P := fun c => Core.Site.WithinTolerance est.tolerance (est.figures.peakRate c) (d.peakRate c)))
  · exact (forall_congr' fun c => pos_iff (d.peakRate (ofOperandClass c))).trans
      (forall_class (P := fun c => 0 < (d.peakRate c).1))

theorem figures_iff : Verity.Site.Entry.Figures (auditor A) (entry e) ↔ Core.Site.Entry.Figures A e := by
  unfold Verity.Site.Entry.Figures Core.Site.Entry.Figures
  simp only [entry_sheets, List.forall_mem_map, auditor_sheet, sheet_sku, Option.forall_mem_map]
  exact forall₂_congr fun k _ => forall₂_congr fun a _ => and_congr (accepted_iff (A.estimate a) k.figures) Iff.rfl

theorem rates_iff : Verity.Site.Entry.Rates (auditor A) (site S) (entry e) ↔ Core.Site.Entry.Rates A S e := by
  unfold Verity.Site.Entry.Rates Core.Site.Entry.Rates
  rw [← site_add]
  simp only [entry_locs, List.forall_mem_map]
  refine forall₂_congr fun x _ => ite_iff ?_ ?_ Iff.rfl
  · rw [loc_parent, bind_kind?_site, map_kind_eq_some]
    exact and_congr (kind_eq_iff (k := x.kind) (k' := .memory)) Iff.rfl
  · have hm : (auditor A).memory.lookup (loc x).id = (A.memory.lookup x.id).map memoryFigures :=
      lookup_map memoryFigures x.id A.memory
    rw [hm]
    constructor
    · rintro ⟨f, hf, h⟩
      obtain ⟨g, hg, rfl⟩ := Option.map_eq_some_iff.mp (Option.mem_def.mp hf)
      exact ⟨g, Option.mem_def.mpr hg, h⟩
    · rintro ⟨g, hg, h⟩
      exact ⟨memoryFigures g, Option.mem_def.mpr (Option.map_eq_some_iff.mpr ⟨g, Option.mem_def.mp hg, rfl⟩), h⟩

theorem timed_iff : Verity.Site.Entry.Timed (site S) (entry e) ↔ Core.Site.Entry.Timed S e := by
  unfold Verity.Site.Entry.Timed Core.Site.Entry.Timed
  rw [← site_add, timeUnit_site]
  simp only [entry_locs, List.forall_mem_map]
  refine forall₂_congr fun x _ => ite_iff (kind_eq_iff (k := x.kind) (k' := .site)) ?_ ?_
  · rw [loc_timeUnit]
    constructor
    · rintro ⟨u, hu, hpos, hS⟩
      obtain ⟨v, hv, rfl⟩ := Option.map_eq_some_iff.mp (Option.mem_def.mp hu)
      exact ⟨v, Option.mem_def.mpr hv, (pos_iff v).mp hpos, Option.map_injective figure_injective hS⟩
    · rintro ⟨v, hv, hpos, hS⟩
      exact ⟨figure v, Option.mem_def.mpr (Option.map_eq_some_iff.mpr ⟨v, Option.mem_def.mp hv, rfl⟩),
        (pos_iff v).mpr hpos, by rw [hS]; rfl⟩
  · rw [loc_timeUnit, Option.map_eq_none_iff]

theorem chips_iff : Verity.Site.Entry.Chips (auditor A) (entry e) ↔ Core.Site.Entry.Chips A e := by
  unfold Verity.Site.Entry.Chips Core.Site.Entry.Chips
  simp only [entry_locs, List.forall_mem_map]
  refine forall₂_congr fun x _ => ite_iff (kind_eq_iff (k := x.kind) (k' := .host)) ?_ ?_
  · have hm : (auditor A).hosts.lookup (loc x).id = (A.hosts.lookup x.id).map hostFigures :=
      lookup_map hostFigures x.id A.hosts
    rw [hm]
    constructor
    · rintro ⟨f, hf, h1, h2, h3⟩
      obtain ⟨g, hg, rfl⟩ := Option.map_eq_some_iff.mp (Option.mem_def.mp hf)
      refine ⟨g, Option.mem_def.mpr hg, h1, (List.map_inj_right shared_injective).mp h2, ?_⟩
      rw [List.all_eq_true]
      intro p hp
      exact decide_eq_true (h3 (shared p) (List.mem_map_of_mem hp))
    · rintro ⟨g, hg, h1, h2, h3⟩
      refine ⟨hostFigures g, Option.mem_def.mpr (Option.map_eq_some_iff.mpr ⟨g, Option.mem_def.mp hg, rfl⟩), h1,
        congrArg (List.map shared) h2, ?_⟩
      intro p hp
      obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hp
      exact of_decide_eq_true (List.all_eq_true.mp h3 q hq)
  · rw [loc_chip, loc_shared, List.map_eq_nil_iff]

theorem capacity_iff :
    Verity.Site.Entry.Capacity (digests D) (site S) (entry e) ↔ Core.Site.Entry.Capacity D S e := by
  unfold Verity.Site.Entry.Capacity Core.Site.Entry.Capacity
  rw [← site_add]
  simp [kind_eq_iff, ofKind, sheet?_site, held_site]

theorem lt_one_iff (q : Core.Site.Figure) : figure q < 1 ↔ q.1 < 1 := by
  rw [← NNRat.coe_lt_coe, coe_figure, NNRat.coe_one]

/-- A clock's drift is below 1 in the spec exactly when the code's check of it passes. -/
theorem drift_iff (o : Option Core.Site.Figure) :
    (∀ q ∈ o.map figure, q < 1) ↔ (o.all fun (q : Core.Site.Figure) => decide (q.1 < 1)) = true := by
  cases o with
  | none => simp
  | some q =>
    constructor
    · intro h; exact decide_eq_true ((lt_one_iff q).mp (h (figure q) rfl))
    · intro h r hr
      obtain rfl : figure q = r := Option.some.inj hr
      exact (lt_one_iff q).mpr (of_decide_eq_true h)

theorem owners_iff : Verity.Site.Entry.Owners (auditor A) (entry e) ↔ Core.Site.Entry.Owners A e := by
  unfold Verity.Site.Entry.Owners Core.Site.Entry.Owners
  simp only [entry_locs, List.forall_mem_map]
  refine forall₂_congr fun x _ => and_congr ?_ (and_congr ?_
    (ite_iff (kind_eq_iff (k := x.kind) (k' := .auditorHost)) (and_congr Iff.rfl (and_congr ?_ ?_)) ?_))
  · refine imp_congr ?_ (and_congr (owner_eq_iff (o := x.owner) (o' := .auditor)) Iff.rfl)
    rw [loc_kind, kind_auditor]
  · refine imp_congr ?_ (and_congr (owner_eq_iff (o := x.owner) (o' := .prover)) Iff.rfl)
    rw [loc_kind, kind_auditor]
  · have hm : (auditor A).drift.lookup (loc x).id = (A.drift.lookup x.id).map figure := lookup_map figure x.id A.drift
    rw [hm, loc_drift]
    exact (Option.map_injective figure_injective).eq_iff
  · rw [loc_drift]; exact drift_iff x.drift
  · rw [loc_drift, Option.map_eq_none_iff]; exact Iff.rfl

theorem covered_iff :
    Verity.Site.Entry.Covered (composite c) (site S) (entry e) ↔ Core.Site.Entry.Covered c S e := by
  unfold Verity.Site.Entry.Covered Core.Site.Entry.Covered
  rw [← site_add]
  simp [composite_eq_iff, ofComposite, boundary_site, certified_site]

theorem reached_iff :
    Verity.Site.Entry.Reached (composite c) (site S) (entry e) ↔ Core.Site.Entry.Reached c S e := by
  unfold Verity.Site.Entry.Reached Core.Site.Entry.Reached
  rw [← site_add]
  simp [composite_eq_iff, ofComposite, kind_eq_iff, ofKind, host?_site, reached_site]

/-- A site entry's refusal in code is the spec's. -/
theorem siteRefusal : (Core.Site.Entry.refusal D A c S e).map refusal =
    Verity.Site.Entry.refusal (digests D) (auditor A) (composite c) (site S) (entry e) := by
  unfold Core.Site.Entry.refusal Verity.Site.Entry.refusal
  simp only [fresh_iff, refs_iff, kinds_iff, sheets_iff, figures_iff, rates_iff, timed_iff, chips_iff, capacity_iff,
    owners_iff, covered_iff, reached_iff]
  split_ifs <;> rfl

/-- The devices under a host of one SKU are as many in code as in the spec. -/
theorem cards_site (i sku : String) :
    ((site S).locs.filter fun d =>
        d.kind = .device ∧ d.parent = some i ∧ ((site S).sheet? (digests D).sheet d).map (·.sku) = some sku).length =
      (S.locs.filter fun d =>
        d.kind = .device ∧ d.parent = some i ∧ (S.sheet? D.sheet d).map (·.sku) = some sku).length := by
  rw [site_locs, List.filter_map, List.length_map]
  refine congrArg List.length (List.filter_congr fun d _ => decide_eq_decide.mpr ?_)
  simp [sheet?_site, kind_eq_iff, ofKind]

/-- A host's cards and ways out are registered in code exactly when they are in the spec. -/
theorem inventory_iff (i : String) (g : Core.Site.HostFigures) :
    ((∀ k ∈ (hostFigures g).cards, k.2 ≤ ((site S).locs.filter fun d =>
          d.kind = .device ∧ d.parent = some i ∧
            ((site S).sheet? (digests D).sheet d).map (·.sku) = some k.1).length) ∧
        ∀ w ∈ (hostFigures g).ways,
          ∃ k ∈ (site S).links, k.id = w.1 ∧ (k.ends.1 = i ∨ k.ends.2 = i) ∧ k.β = w.2) ↔
      (∀ k ∈ g.cards, k.2 ≤ (S.locs.filter fun d =>
          d.kind = .device ∧ d.parent = some i ∧ (S.sheet? D.sheet d).map (·.sku) = some k.1).length) ∧
        ∀ w ∈ g.ways, ∃ k ∈ S.links, k.id = w.1 ∧ (k.ends.1 = i ∨ k.ends.2 = i) ∧ k.β = w.2 := by
  refine and_congr (forall₂_congr fun k _ => ?_) (forall₂_congr fun w _ => ?_)
  · exact iff_of_eq (congrArg (k.2 ≤ ·) (cards_site D S i k.1))
  · constructor
    · rintro ⟨k, hk, h⟩
      obtain ⟨k', hk', rfl⟩ := List.mem_map.mp hk
      exact ⟨k', hk', h⟩
    · rintro ⟨k, hk, h⟩
      exact ⟨link k, List.mem_map_of_mem hk, h⟩

/-- **Completeness is the spec's**: a code Site is complete against the auditor's figures exactly when the Site it
denotes is. -/
theorem complete_iff :
    Verity.Registration.Complete (digests D) (auditor A) (site S) ↔ Core.Site.Registration.Complete D A S := by
  unfold Verity.Registration.Complete Core.Site.Registration.Complete
  constructor
  · intro hc h hh hk
    obtain ⟨hm, hg⟩ := hc (loc h) (List.mem_map_of_mem hh) ((kind_eq_iff (k := h.kind) (k' := .host)).mpr hk)
    refine ⟨fun f hf hfh => ?_, fun g hg' => ?_⟩
    · obtain ⟨y, hy, h1, h2, h3⟩ := hm (f.1, memoryFigures f.2)
        (List.mem_map_of_mem (f := fun p : String × Core.Site.MemoryFigures => (p.1, memoryFigures p.2)) hf) hfh
      obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hy
      exact ⟨x, hx, h1, (kind_eq_iff (k := x.kind) (k' := .memory)).mp h2, h3⟩
    · exact (inventory_iff D S h.id g).mp
        (hg (hostFigures g) ((mem_lookup_map hostFigures h.id A.hosts _).mpr ⟨g, hg', rfl⟩))
  · intro hc y hy hk
    obtain ⟨h, hh, rfl⟩ := List.mem_map.mp hy
    obtain ⟨hm, hg⟩ := hc h hh ((kind_eq_iff (k := h.kind) (k' := .host)).mp hk)
    refine ⟨fun f hf hfh => ?_, fun g' hg' => ?_⟩
    · obtain ⟨f', hf', rfl⟩ := List.mem_map.mp hf
      obtain ⟨x, hx, h1, h2, h3⟩ := hm f' hf' hfh
      exact ⟨loc x, List.mem_map_of_mem hx, h1, (kind_eq_iff (k := x.kind) (k' := .memory)).mpr h2, h3⟩
    · obtain ⟨g, hg'', rfl⟩ := (mem_lookup_map hostFigures h.id A.hosts g').mp hg'
      exact (inventory_iff D S h.id g).mpr (hg g hg'')

end Checks

/-! ## The registration's check is the spec's -/

theorem refuses : Refuses := by
  intro D A c S head e
  cases e with
  | site p s =>
    unfold Core.Site.Registration.Entry.refusal Verity.Registration.Entry.refusal
    simp only [regEntry, Verity.Registration.Entry.Follows, Verity.Registration.Entry.previous,
      Core.Site.Registration.Entry.previous, Core.Site.Registration.Entry.kindRefusal,
      Verity.Registration.Entry.kindRefusal, ne_eq]
    by_cases hp : p = head <;> simp [hp, siteRefusal, refusal]

theorem entryValid : EntryValid := by
  intro D A c S head e
  rw [← Verity.Registration.Entry.refusal_eq_none_iff, ← refuses, Option.map_eq_none_iff]

theorem fold : Fold := by
  intro l
  induction l with
  | nil => rfl
  | cons e l ih =>
    have h : Verity.Site.ofLog (log (e :: l)) = Verity.Site.ofLog ([regEntry e] ++ log l) := rfl
    rw [h, Verity.Site.ofLog_append, ← ih]
    cases e
    simp [site, Core.Site.ofLog, Verity.Site.ofLog, regEntry, entry, Verity.Registration.Entry.added,
      Core.Site.Registration.Entry.added]

theorem log_append (l l' : Core.Site.Registration) : log (l ++ l') = log l ++ log l' := List.map_append

theorem log_take (l : Core.Site.Registration) (i : Nat) : log (l.take i) = (log l).take i := List.map_take

theorem items_log : ∀ l : Core.Site.Registration, Verity.Registration.items (log l) = []
  | [] => rfl
  | .site _ _ :: l => items_log l

/-- A code log registers no item, and a site entry declares no day, so an entry it denotes registers none twice. -/
theorem once_regEntry (l : Core.Site.Registration) (e : Core.Site.Registration.Entry) :
    Verity.Registration.Entry.Once (log l) (regEntry e) := by
  cases e
  simp [Verity.Registration.Entry.Once, Verity.Registration.Entry.ItemsOnce, Verity.Registration.Entry.DayOnce,
    items_log, regEntry, Verity.Registration.Entry.items, Verity.Registration.Entry.declared?]

/-- What a check's result says of the log it checked: an accepted log is valid and its Site complete, with the spec's
Site of it; a refusal at `i` follows a valid prefix of `i` entries, and is the spec's refusal of the next, or
`inventory` at the log's end, where the log's Site is not complete. -/
def Says (D : Digests) (A : Auditor) (c : Composite) (l : Core.Site.Registration) :
    Except (Nat × Core.Site.Registration.Refusal) Core.Site → Prop
  | .ok S => (Verity.Registration.Valid (digests D) (auditor A) (composite c) (log l) ∧
        Verity.Registration.Complete (digests D) (auditor A) (Verity.Site.ofLog (log l))) ∧
      site S = Verity.Site.ofLog (log l)
  | .error (i, r) =>
      Verity.Registration.Valid (digests D) (auditor A) (composite c) (log (l.take i)) ∧
        ((∃ hi : i < l.length,
            Verity.Registration.Entry.refusal (digests D) (auditor A) (composite c)
              (Verity.Site.ofLog (log (l.take i))) (Verity.Registration.head (digests D) (composite c) (log (l.take i)))
              (regEntry l[i]) = some (refusal r)) ∨
          (i = l.length ∧ r = .inventory ∧
            ¬Verity.Registration.Complete (digests D) (auditor A) (Verity.Site.ofLog (log l))))

/-- **The check follows the spec one entry at a time**: from a valid prefix `p`, with its Site and head, the check of
the rest says what the spec says of the whole log. -/
theorem go_says (D : Digests) (A : Auditor) (c : Composite) :
    ∀ (rest p : Core.Site.Registration) (S : Core.Site) (h : String),
      site S = Verity.Site.ofLog (log p) → h = Verity.Registration.head (digests D) (composite c) (log p) →
      Verity.Registration.Valid (digests D) (auditor A) (composite c) (log p) →
      Says D A c (p ++ rest) (Core.Site.Registration.check.go D A c p.length S h rest) := by
  intro rest
  induction rest with
  | nil =>
    intro p S h hS _ hv
    have hc := complete_iff D A S
    rw [hS] at hc
    simp only [Core.Site.Registration.check.go, List.append_nil]
    split
    · next hcS => exact ⟨⟨hv, hc.mpr hcS⟩, hS⟩
    · next hcS => exact ⟨by simpa using hv, Or.inr ⟨rfl, rfl, fun h' => hcS (hc.mp h')⟩⟩
  | cons e rest ih =>
    intro p S h hS hh hv
    have hr := refuses D A c S h e
    simp only [Core.Site.Registration.check.go]
    split
    · next r hre =>
      refine ⟨by simpa [List.take_append_length] using hv, Or.inl ⟨by simp, ?_⟩⟩
      rw [hre] at hr
      simp only [List.take_append_length, List.getElem_append_right (Nat.le_refl _), Nat.sub_self,
        List.getElem_cons_zero]
      rw [← hS, ← hh, ← hr]; rfl
    · next hre =>
      rw [hre] at hr
      have hev : Verity.Registration.Entry.Valid (digests D) (auditor A) (composite c)
          (Verity.Site.ofLog (log p)) (Verity.Registration.head (digests D) (composite c) (log p)) (regEntry e) := by
        rw [← hS, ← hh, ← Verity.Registration.Entry.refusal_eq_none_iff, ← hr]; rfl
      have hv' : Verity.Registration.Valid (digests D) (auditor A) (composite c) (log (p ++ [e])) := by
        rw [log_append]; exact Verity.Registration.valid_append_singleton.mpr ⟨hv, hev, once_regEntry p e⟩
      have := ih (p ++ [e]) (S.add e.added) (D.entry e)
        (by
          rw [site_add, hS, log_append]
          show _ = Verity.Site.ofLog (log p ++ [regEntry e])
          rw [Verity.Site.ofLog_concat, regEntry_added])
        (by rw [log_append]; simp [log]) hv'
      simpa [List.append_assoc] using this

/-- What the check says of a whole log, from the empty prefix. -/
theorem check_says (D : Digests) (A : Auditor) (c : Composite) (l : Core.Site.Registration) :
    Says D A c l (Core.Site.Registration.check D A c l) := by
  have hs := go_says D A c l [] ⟨[], [], []⟩ (D.genesis c) rfl (by simp [log]) (by simp [log])
  simp only [List.nil_append, List.length_nil] at hs
  exact hs

theorem checks : Checks := by
  intro D A c l
  have hs := check_says D A c l
  revert hs
  cases Core.Site.Registration.check D A c l with
  | ok S => rintro ⟨hv, hS⟩; rw [ite_eq_left hv]; simp [Except.toOption, hS]
  | error ir =>
    obtain ⟨i, r⟩ := ir
    rintro ⟨_, hr⟩
    have hnot : ¬(Verity.Registration.Valid (digests D) (auditor A) (composite c) (log l) ∧
        Verity.Registration.Complete (digests D) (auditor A) (Verity.Site.ofLog (log l))) := by
      rintro ⟨hv, hcomp⟩
      rcases hr with ⟨hi, hr⟩ | ⟨_, _, hn⟩
      · have hp : Verity.Registration.Valid (digests D) (auditor A) (composite c) (log (l.take (i + 1))) :=
          Verity.Registration.Valid.of_prefix (by rw [log_take]; exact List.take_prefix _ _) hv
        rw [List.take_succ_eq_append_getElem hi, log_append] at hp
        have he := (Verity.Registration.valid_append_singleton.mp hp).2.1
        rw [← Verity.Registration.Entry.refusal_eq_none_iff] at he
        rw [he] at hr
        cases hr
      · exact hn hcomp
    rw [ite_eq_right hnot]; simp [Except.toOption]

theorem refusedAt : RefusedAt := by
  intro D A c l i r hc
  have hs := check_says D A c l
  rw [hc] at hs
  exact hs

/-! ## Witnesses

Each guarantee holds at `honestLog`, a host and its uplink registered for the remote composite, and its conclusion
fails at a check that accepts everything (for `refusedAt`, one that refuses everything): the code's check decides each
instance, and the guarantee carries it to the spec. -/

theorem honestEntry_refusal :
    honestEntry.refusal witnessDigests witnessAuditor .remote emptySite "genesis" = none := by decide

theorem staleEntry_refusal :
    staleEntry.refusal witnessDigests witnessAuditor .remote emptySite "genesis" = some .previous := by decide

theorem check_honestLog :
    Core.Site.Registration.check witnessDigests witnessAuditor .remote honestLog = .ok (Core.Site.ofLog honestLog) := by
  rfl

theorem check_staleLog :
    Core.Site.Registration.check witnessDigests witnessAuditor .remote staleLog = .error (1, .previous) := by rfl

theorem valid_honestLog :
    Verity.Registration.Valid (digests witnessDigests) (auditor witnessAuditor) (composite .remote) (log honestLog) ∧
      Verity.Registration.Complete (digests witnessDigests) (auditor witnessAuditor)
        (Verity.Site.ofLog (log honestLog)) := by
  have h := checks witnessDigests witnessAuditor .remote honestLog
  rw [check_honestLog] at h
  by_contra hv
  rw [ite_eq_right hv] at h
  simp [Except.toOption] at h

theorem not_valid_staleLog :
    ¬(Verity.Registration.Valid (digests witnessDigests) (auditor witnessAuditor) (composite .remote) (log staleLog) ∧
      Verity.Registration.Complete (digests witnessDigests) (auditor witnessAuditor)
        (Verity.Site.ofLog (log staleLog))) := by
  intro hv
  have h := checks witnessDigests witnessAuditor .remote staleLog
  rw [check_staleLog, ite_eq_left hv] at h
  simp [Except.toOption] at h

/-- `Refuses`' honest witness: the code accepts `honestEntry` against the empty Site at the genesis, and so does the
spec. -/
theorem refuses_honest :
    honestEntry.refusal witnessDigests witnessAuditor .remote emptySite "genesis" = none ∧
      Verity.Registration.Entry.refusal (digests witnessDigests) (auditor witnessAuditor) (composite .remote)
        (site emptySite) "genesis" (regEntry honestEntry) = none := by
  refine ⟨honestEntry_refusal, ?_⟩
  rw [← refuses, honestEntry_refusal]
  rfl

/-- `Refuses`' nontrivial witness: an entry check that accepts everything doesn't refuse `staleEntry` as the spec
does. -/
theorem refuses_nontrivial :
    ¬(acceptAllRefusal staleEntry).map refusal =
      Verity.Registration.Entry.refusal (digests witnessDigests) (auditor witnessAuditor) (composite .remote)
        (site emptySite) "genesis" (regEntry staleEntry) := by
  rw [← refuses, staleEntry_refusal]
  simp [acceptAllRefusal]

/-- `EntryValid`'s honest witness: the code accepts `honestEntry` against the empty Site at the genesis, and it is
valid there. -/
theorem entryValid_honest :
    honestEntry.refusal witnessDigests witnessAuditor .remote emptySite "genesis" = none ∧
      Verity.Registration.Entry.Valid (digests witnessDigests) (auditor witnessAuditor) (composite .remote)
        (site emptySite) "genesis" (regEntry honestEntry) :=
  ⟨honestEntry_refusal, (entryValid _ _ _ _ _ _).mp honestEntry_refusal⟩

/-- `EntryValid`'s nontrivial witness: an entry check that accepts everything accepts `staleEntry`, which is not
valid. -/
theorem entryValid_nontrivial :
    ¬(acceptAllRefusal staleEntry = none ↔
      Verity.Registration.Entry.Valid (digests witnessDigests) (auditor witnessAuditor) (composite .remote)
        (site emptySite) "genesis" (regEntry staleEntry)) := by
  intro h
  have hr := (entryValid _ _ _ _ _ _).mpr (h.mp rfl)
  rw [staleEntry_refusal] at hr
  cases hr

/-- `Checks`' honest witness: the code's check accepts `honestLog`, with its Site, and the log is valid and its Site
complete against the auditor's figures. -/
theorem checks_honest :
    Core.Site.Registration.check witnessDigests witnessAuditor .remote honestLog = .ok (Core.Site.ofLog honestLog) ∧
      Verity.Registration.Valid (digests witnessDigests) (auditor witnessAuditor) (composite .remote) (log honestLog) ∧
      Verity.Registration.Complete (digests witnessDigests) (auditor witnessAuditor)
        (Verity.Site.ofLog (log honestLog)) :=
  ⟨check_honestLog, valid_honestLog⟩

/-- `Checks`' nontrivial witness: a log check that accepts everything accepts `staleLog`, which is not valid. -/
theorem checks_nontrivial :
    ¬(acceptAllCheck staleLog).toOption.map site =
      if Verity.Registration.Valid (digests witnessDigests) (auditor witnessAuditor) (composite .remote)
          (log staleLog) ∧
        Verity.Registration.Complete (digests witnessDigests) (auditor witnessAuditor)
          (Verity.Site.ofLog (log staleLog))
      then some (Verity.Site.ofLog (log staleLog)) else none := by
  rw [ite_eq_right not_valid_staleLog]
  simp [acceptAllCheck, Except.toOption]

/-- `RefusedAt`'s honest witness: the code's check accepts `honestLog` and refuses `staleLog` at its second entry, as
not following the head. -/
theorem refusedAt_honest :
    Core.Site.Registration.check witnessDigests witnessAuditor .remote honestLog = .ok (Core.Site.ofLog honestLog) ∧
      Core.Site.Registration.check witnessDigests witnessAuditor .remote staleLog = .error (1, .previous) :=
  ⟨check_honestLog, check_staleLog⟩

/-- `RefusedAt`'s nontrivial witness: a log check that refuses everything at its first entry, as not following the
head, refuses `honestLog` with a refusal the spec doesn't give that entry, at every index of `honestLog` it may name. -/
theorem refusedAt_nontrivial :
    ∀ i : Fin honestLog.length,
      ¬Verity.Registration.Entry.refusal (digests witnessDigests) (auditor witnessAuditor) (composite .remote)
          (Verity.Site.ofLog (log (honestLog.take i)))
          (Verity.Registration.head (digests witnessDigests) (composite .remote) (log (honestLog.take i)))
          (regEntry honestLog[i]) = some (refusal .previous) := by
  rintro ⟨i, hi⟩
  obtain rfl : i = 0 := by simp only [honestLog, List.length_singleton] at hi; omega
  have h := refuses witnessDigests witnessAuditor .remote emptySite "genesis" honestEntry
  rw [honestEntry_refusal] at h
  show ¬Verity.Registration.Entry.refusal (digests witnessDigests) (auditor witnessAuditor) (composite .remote)
    (site emptySite) "genesis" (regEntry honestEntry) = _
  rw [← h]
  simp

end Properties.Site
