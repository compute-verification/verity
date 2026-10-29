import Mathlib.Data.List.Perm.Basic
import Properties.Core.Site
import Proofs.Core.Site

/-!
# The Site's code is the spec's: the proofs

Each translation of `Properties.Core.Site` is injective, the code's functions of the Site commute with them
(`kind?_site`, `held_site`, `far_site`, `host?_site`, `reached_site`), so each of a site entry's checks holds in code
exactly when the spec's holds of what it denotes (`fresh_iff` … `reached_iff`), and its refusal is the spec's. The
check of a log then follows the spec's validity one entry at a time (`go_says`), by
`Verity.Registration.valid_append_singleton`.
-/

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

theorem ofLoc_loc (x : Core.Site.Loc) : ofLoc (loc x) = x := by
  cases x
  simp only [loc, ofLoc, ofKind_kind, ofOwner_owner, Option.map_map, Function.comp_def, ofTier_tier, Option.map_id']

theorem ofLink_link (k : Core.Site.Link) : ofLink (link k) = k := rfl

theorem ofEntry_entry (e : Core.Site.Entry) : ofEntry (entry e) = e := by
  cases e
  simp only [entry, ofEntry, List.map_map, Function.comp_def, ofSheet_sheet, ofLoc_loc, ofLink_link, List.map_id']

theorem ofRegEntry_regEntry (e : Core.Site.Registration.Entry) : ofRegEntry (regEntry e) = e := by
  cases e; simp only [regEntry, ofRegEntry, ofEntry_entry]

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
@[simp] theorem link_id (k : Core.Site.Link) : (link k).id = k.id := rfl
@[simp] theorem link_ends (k : Core.Site.Link) : (link k).ends = k.ends := rfl
@[simp] theorem link_β (k : Core.Site.Link) : (link k).β = k.β := rfl
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
    link_ends, link_β, hb, hf, parents_iff, perm_iff, Option.isSome_map, kind_eq_iff, ofKind]
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

theorem capacity_iff :
    Verity.Site.Entry.Capacity (digests D) (site S) (entry e) ↔ Core.Site.Entry.Capacity D S e := by
  unfold Verity.Site.Entry.Capacity Core.Site.Entry.Capacity
  rw [← site_add]
  simp [kind_eq_iff, ofKind, sheet?_site, held_site]

theorem owners_iff : Verity.Site.Entry.Owners (auditor A) (entry e) ↔ Core.Site.Entry.Owners A e := by
  unfold Verity.Site.Entry.Owners Core.Site.Entry.Owners
  simp [owner_eq_iff, ofOwner]

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
  simp only [fresh_iff, refs_iff, kinds_iff, sheets_iff, figures_iff, capacity_iff, owners_iff, covered_iff,
    reached_iff]
  split_ifs <;> rfl

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

/-- What a check's result says of the log it checked: an accepted log is valid, with the spec's Site of it; a refusal
at `i` follows a valid prefix of `i` entries, and is the spec's refusal of the next. -/
def Says (D : Digests) (A : Auditor) (c : Composite) (l : Core.Site.Registration) :
    Except (Nat × Core.Site.Registration.Refusal) Core.Site → Prop
  | .ok S => Verity.Registration.Valid (digests D) (auditor A) (composite c) (log l) ∧
      site S = Verity.Site.ofLog (log l)
  | .error (i, r) => ∃ hi : i < l.length,
      Verity.Registration.Valid (digests D) (auditor A) (composite c) (log (l.take i)) ∧
        Verity.Registration.Entry.refusal (digests D) (auditor A) (composite c)
            (Verity.Site.ofLog (log (l.take i))) (Verity.Registration.head (digests D) (composite c) (log (l.take i)))
            (regEntry l[i]) = some (refusal r)

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
    simp only [Core.Site.Registration.check.go, List.append_nil]
    exact ⟨hv, hS⟩
  | cons e rest ih =>
    intro p S h hS hh hv
    have hr := refuses D A c S h e
    simp only [Core.Site.Registration.check.go]
    split
    · next r hre =>
      refine ⟨by simp, ?_, ?_⟩
      · simpa [List.take_append_length] using hv
      · rw [hre] at hr
        simp only [List.take_append_length, List.getElem_append_right (Nat.le_refl _), Nat.sub_self,
          List.getElem_cons_zero]
        rw [← hS, ← hh, ← hr]; rfl
    · next hre =>
      rw [hre] at hr
      have hev : Verity.Registration.Entry.Valid (digests D) (auditor A) (composite c)
          (Verity.Site.ofLog (log p)) (Verity.Registration.head (digests D) (composite c) (log p)) (regEntry e) := by
        rw [← hS, ← hh, ← Verity.Registration.Entry.refusal_eq_none_iff, ← hr]; rfl
      have hv' : Verity.Registration.Valid (digests D) (auditor A) (composite c) (log (p ++ [e])) := by
        rw [log_append]; exact Verity.Registration.valid_append_singleton.mpr ⟨hv, hev⟩
      have := ih (p ++ [e]) (S.add e.added) (D.entry e)
        (by
          rw [site_add, hS, log_append]
          show _ = Verity.Site.ofLog (log p ++ [regEntry e])
          rw [Verity.Site.ofLog_concat, regEntry_added])
        (by rw [log_append]; simp [log]) hv'
      simpa [List.append_assoc] using this

theorem checks : Checks := by
  intro D A c l
  have hs := go_says D A c l [] ⟨[], [], []⟩ (D.genesis c) rfl (by simp [log]) (by simp [log])
  simp only [List.nil_append, List.length_nil] at hs
  unfold Core.Site.Registration.check
  revert hs
  cases Core.Site.Registration.check.go D A c 0 ⟨[], [], []⟩ (D.genesis c) l with
  | ok S => rintro ⟨hv, hS⟩; simp [Except.toOption, hv, hS]
  | error ir =>
    obtain ⟨i, r⟩ := ir
    rintro ⟨hi, _, hr⟩
    have hnot : ¬Verity.Registration.Valid (digests D) (auditor A) (composite c) (log l) := by
      intro hv
      have hp : Verity.Registration.Valid (digests D) (auditor A) (composite c) (log (l.take (i + 1))) :=
        Verity.Registration.Valid.of_prefix (by rw [log_take]; exact List.take_prefix _ _) hv
      rw [List.take_succ_eq_append_getElem hi, log_append] at hp
      have he := (Verity.Registration.valid_append_singleton.mp hp).2
      rw [← Verity.Registration.Entry.refusal_eq_none_iff] at he
      rw [he] at hr
      cases hr
    simp [Except.toOption, hnot]

theorem refusedAt : RefusedAt := by
  intro D A c l i r hc
  have hs := go_says D A c l [] ⟨[], [], []⟩ (D.genesis c) rfl (by simp [log]) (by simp [log])
  simp only [List.nil_append, List.length_nil] at hs
  unfold Core.Site.Registration.check at hc
  rw [hc] at hs
  exact hs

end Properties.Site
