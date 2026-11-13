module

public import Definitions.Accounting.Registration.Guarantees.Witnesses
public import Proofs.Core.Site

/-!
# What registration establishes: proofs

`RegistrationSound`, one registered fact at a time, each by induction along the log:

* `Registration.Valid.sheets_accepted`: every sheet of a valid log's Site is accepted against the auditor's sheet for
  its SKU, since the site entry that added it passed `sheetDigest` (the auditor has a sheet for the SKU) and `figures`;
  and `Registration.Valid.devices_named`: every device names a registered sheet's digest, since `sheetDigest` checked it
  against the Site the entry extends, and the Site only grows. So every device's sheet lookup finds an accepted sheet
  (`Registration.Valid.devices`).
* `Registration.Valid.onsite`: a site entry's `uncertified` and `challengerReach` checks read the whole Site it extends
  the log's to, and an information entry or a declaration adds nothing to it (`Registration.Valid.covered`); and each
  memory leaf is in a host, its parent or its device's parent (`Registration.Valid.host?_isSome`).
* `Registration.Valid.checked`: every check of a site entry holds of the whole Site, since each site entry passed them
  against the Site before it and the checks of the older locations, links and sheets carry over to a Site that extends
  theirs (`Site.checked_add`); and `Registration.Valid.chained`: each entry passed its `chain` check.
* `Registration.Valid.theta_nodup`: the last entry's `duplicateInformation` check reads every item the log registers, a
  θ value by its name; and `Registration.Valid.declarations_nodup`: the log declares each day once
  (`Registration.Valid.days_nodup`).

`RegistrationSound_honest` and `RegistrationSound_nontrivial`, at the witness site of
`Definitions.Accounting.Registration.Guarantees.Witnesses`: its log passes the check by evaluation, and a log whose
device names no sheet finds no sheet for it.
-/

@[expose] public section

namespace Verity

namespace Site

/-- A device naming a sheet's digest finds a sheet of the Site under it. -/
theorem sheet?_of_mem {S : Site} {digest : Sheet → String} {x : Loc} (h : x.sheet ∈ S.sheets.map digest) :
    ∃ k, S.sheet? digest x = some k ∧ k ∈ S.sheets := by
  obtain ⟨k, hk, hkx⟩ := List.mem_map.mp h
  have hs : (S.sheets.find? (digest · == x.sheet)).isSome := List.find?_isSome.mpr ⟨k, hk, by simp [hkx]⟩
  obtain ⟨k', hk'⟩ := Option.isSome_iff_exists.mp hs
  exact ⟨k', hk', List.mem_of_find?_eq_some hk'⟩

@[simp] theorem asEntry_sheets (S : Site) : S.asEntry.sheets = S.sheets := rfl
@[simp] theorem asEntry_locs (S : Site) : S.asEntry.locs = S.locs := rfl
@[simp] theorem asEntry_links (S : Site) : S.asEntry.links = S.links := rfl
theorem add_sheets (S : Site) (e : Entry) : (S.add e).sheets = S.sheets ++ e.sheets := rfl
theorem add_locs (S : Site) (e : Entry) : (S.add e).locs = S.locs ++ e.locs := rfl
theorem add_links (S : Site) (e : Entry) : (S.add e).links = S.links ++ e.links := rfl

/-- The Site, as one site entry over the empty Site, is itself. -/
theorem empty_add_asEntry (S : Site) : (⟨[], [], []⟩ : Site).add S.asEntry = S := by
  cases S
  simp [Site.add, Site.asEntry]

/-- A link joining two kinds still joins them in a Site whose locations extend its. -/
theorem joins_of_prefix {S S' : Site} (h : S.locs <+: S'.locs) {k : Link} {a b : Kind} (hj : S.Joins k a b) :
    S'.Joins k a b := by
  rcases hj with hj | hj <;> simp only [Prod.mk.injEq] at hj
  · exact Or.inl (by rw [Site.kind?_of_prefix h hj.1, Site.kind?_of_prefix h hj.2])
  · exact Or.inr (by rw [Site.kind?_of_prefix h hj.1, Site.kind?_of_prefix h hj.2])

/-- The empty Site passes every check of a site entry. -/
theorem checked_empty {D : Registration.Digests} {A : Registration.Auditor} {c : Registration.Composite} :
    (⟨[], [], []⟩ : Site).Checked D A c := by
  simp [Site.Checked, Site.Entry.Valid, Site.Entry.Fresh, Site.Entry.Refs, Site.Entry.Kinds, Site.Entry.Sheets,
    Site.Entry.Figures, Site.Entry.Capacity, Site.Entry.Owners, Site.Entry.Covered, Site.Entry.Reached, Site.asEntry,
    Site.add, Site.ids]

/-- **A site entry valid against a Site that passes every check leaves one that does**: the checks over the whole Site
it extends to (fresh ids and sheets, certifiers' uplinks, devices' capacity, coverage, reach) are the entry's own, and
the rest hold of the old locations, links and sheets in the larger Site, whose kinds and names only grow. -/
theorem checked_add {D : Registration.Digests} {A : Registration.Auditor} {c : Registration.Composite} {S : Site}
    {e : Entry} (hS : S.Checked D A c) (he : e.Valid D A c S) : (S.add e).Checked D A c := by
  have hpre : S.locs <+: (S.add e).locs := ⟨e.locs, rfl⟩
  obtain ⟨eF, eR, eK, eS, eFi, eC, eO, eCo, eRe⟩ := he
  obtain ⟨sF, sR, sK, sS, sFi, sC, sO, -, -⟩ := hS
  simp only [Site.Entry.Fresh, Site.Entry.Refs, Site.Entry.Kinds, Site.Entry.Sheets, Site.Entry.Figures,
    Site.Entry.Capacity, Site.Entry.Owners, empty_add_asEntry, asEntry_sheets, asEntry_locs, asEntry_links]
    at sF sR sK sS sFi sC sO
  simp only [Site.Entry.Fresh, Site.Entry.Refs, Site.Entry.Kinds, Site.Entry.Sheets, Site.Entry.Figures,
    Site.Entry.Capacity, Site.Entry.Owners, Site.Entry.Covered, Site.Entry.Reached] at eF eR eK eS eFi eC eO eCo eRe
  simp only [Site.Checked, Site.Entry.Valid, Site.Entry.Fresh, Site.Entry.Refs, Site.Entry.Kinds, Site.Entry.Sheets,
    Site.Entry.Figures, Site.Entry.Capacity, Site.Entry.Owners, Site.Entry.Covered, Site.Entry.Reached,
    empty_add_asEntry, asEntry_sheets, asEntry_locs, asEntry_links]
  refine ⟨eF, ⟨fun x hx p hp => ?_, fun k hk => ?_⟩, ⟨fun x hx => ?_, fun k hk => ?_, eK.2.2⟩,
    ⟨fun k hk => ?_, fun x hx hd => ?_⟩, fun k hk => ?_, ⟨fun x hx hm => ?_, eC.2⟩, fun x hx => ?_, eCo, eRe⟩
  · rcases List.mem_append.mp hx with hx | hx
    · exact Site.known_of_prefix hpre (sR.1 x hx p hp)
    · exact eR.1 x hx p hp
  · rcases List.mem_append.mp hk with hk | hk
    · exact ⟨Site.known_of_prefix hpre (sR.2 k hk).1, Site.known_of_prefix hpre (sR.2 k hk).2⟩
    · exact eR.2 k hk
  · rcases List.mem_append.mp hx with hx | hx
    · obtain ⟨hp, ht⟩ := sK.1 x hx
      refine ⟨?_, ht⟩
      cases hpar : x.parent with
      | none => simpa [hpar] using hp
      | some p =>
        obtain ⟨q, hq⟩ := Option.isSome_iff_exists.mp
          (Site.kind?_isSome_of_known (sR.1 x hx p (by simp [hpar])))
        simpa [hpar, hq, Site.kind?_of_prefix hpre hq] using hp
    · exact eK.1 x hx
  · rcases List.mem_append.mp hk with hk | hk
    · obtain ⟨hne, ⟨p, hp, hj⟩, hβ⟩ := sK.2.1 k hk
      exact ⟨hne, ⟨p, hp, joins_of_prefix hpre hj⟩, hβ⟩
    · exact eK.2.1 k hk
  · rcases List.mem_append.mp hk with hk | hk
    · exact sS.1 k hk
    · exact eS.1 k hk
  · rcases List.mem_append.mp hx with hx | hx
    · simp only [add_sheets, List.map_append, List.mem_append]
      exact Or.inl (sS.2 x hx hd)
    · exact eS.2 x hx hd
  · rcases List.mem_append.mp hk with hk | hk
    · exact sFi k hk
    · exact eFi k hk
  · rcases List.mem_append.mp hx with hx | hx
    · exact sC.1 x hx hm
    · exact eC.1 x hx hm
  · rcases List.mem_append.mp hx with hx | hx
    · exact sO x hx
    · exact eO x hx

end Site

namespace Registration

variable {D : Digests} {A : Auditor} {c : Composite}

/-- **Every sheet of a valid log's Site is accepted** against the auditor's sheet for its SKU, figures and memory
figures alike. -/
theorem Valid.sheets_accepted {l : Registration} (h : Valid D A c l) :
    ∀ k ∈ (Site.ofLog l).sheets, ∃ a ∈ A.sheet k.sku,
      k.figures.Accepted (A.estimate a) ∧ k.bw = a.bw ∧ k.chip = a.chip ∧ k.mem = a.mem := by
  induction l using List.reverseRecOn with
  | nil => simp
  | append_singleton l e ih =>
    obtain ⟨hl, ⟨-, he⟩, -⟩ := valid_append_singleton.mp h
    rw [Site.ofLog_concat]
    intro k hk
    cases e with
    | information _ _ => exact ih hl k (by simpa [Entry.added] using hk)
    | declaration _ _ => exact ih hl k (by simpa [Entry.added] using hk)
    | site _ s =>
      simp only [Entry.added, Site.add, List.mem_append] at hk
      rcases hk with hk | hk
      · exact ih hl k hk
      · obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (he.2.2.2.1.1 k hk)
        exact ⟨a, ha, he.2.2.2.2.1 k hk a ha⟩

/-- **Every device of a valid log's Site names a registered sheet's digest.** -/
theorem Valid.devices_named {l : Registration} (h : Valid D A c l) :
    ∀ x ∈ (Site.ofLog l).locs, x.kind = .device → x.sheet ∈ (Site.ofLog l).sheets.map D.sheet := by
  induction l using List.reverseRecOn with
  | nil => simp
  | append_singleton l e ih =>
    obtain ⟨hl, ⟨-, he⟩, -⟩ := valid_append_singleton.mp h
    rw [Site.ofLog_concat]
    intro x hx hd
    cases e with
    | information _ _ => simpa [Entry.added] using ih hl x (by simpa [Entry.added] using hx) hd
    | declaration _ _ => simpa [Entry.added] using ih hl x (by simpa [Entry.added] using hx) hd
    | site _ s =>
      have hx' : x ∈ (Site.ofLog l).locs ∨ x ∈ s.locs := by simpa [Entry.added, Site.add] using hx
      rcases hx' with hx' | hx'
      · have := ih hl x hx' hd
        simp only [Entry.added, Site.add, List.map_append, List.mem_append]
        exact Or.inl this
      · exact he.2.2.2.1.2 x hx' hd

/-- **Every device of a valid log has accepted figures** (`View.Devices`). -/
theorem Valid.devices {l : Registration} (h : Valid D A c l) : (view D c l).Devices D A := by
  intro x hx hd
  obtain ⟨k, hk, hmem⟩ := Site.sheet?_of_mem (h.devices_named x hx hd)
  exact ⟨k, hk, h.sheets_accepted k hmem⟩

/-- **Onsite, a valid log's boundary links are certified, and each memory leaf's host, if it has one, is reached**: a
site entry's `uncertified` and `challengerReach` checks read the whole Site it extends the log's to. -/
theorem Valid.covered {l : Registration} (h : Valid D A c l) : c = .onsite →
    (∀ k ∈ (Site.ofLog l).links, (Site.ofLog l).Boundary k → (Site.ofLog l).Certified k) ∧
      ∀ x ∈ (Site.ofLog l).locs, x.kind = .memory → ∀ q ∈ (Site.ofLog l).host? x, q ∈ (Site.ofLog l).reached := by
  induction l using List.reverseRecOn with
  | nil => intro _; simp
  | append_singleton l e ih =>
    obtain ⟨hl, ⟨-, he⟩, -⟩ := valid_append_singleton.mp h
    intro hc
    cases e with
    | site _ s =>
      simp only [Site.ofLog_concat, Entry.added]
      exact ⟨he.2.2.2.2.2.2.2.1 hc, he.2.2.2.2.2.2.2.2 hc⟩
    | information _ _ => simpa [Site.ofLog_concat, Entry.added] using ih hl hc
    | declaration _ _ => simpa [Site.ofLog_concat, Entry.added] using ih hl hc

/-- **Every memory leaf of a valid log's Site is in a host**: its parent is a host, or a device whose parent is a host
(`Valid.parents`). -/
theorem Valid.host?_isSome {l : Registration} (h : Valid D A c l) {x : Site.Loc} (hx : x ∈ (Site.ofLog l).locs)
    (hm : x.kind = .memory) : ∃ q, (Site.ofLog l).host? x = some q := by
  obtain ⟨-, hp⟩ := h.parents x hx
  rw [hm] at hp
  simp only [Site.Kind.parents, List.mem_cons, List.not_mem_nil, or_false] at hp
  cases hpar : x.parent with
  | none => simp [hpar] at hp
  | some p =>
    rw [hpar, Option.bind_some] at hp
    rcases hp with hpk | hpk
    · have hpo : p ≠ Site.outside := by
        rintro rfl
        simp [Site.kind?] at hpk
      have hpk' := hpk
      unfold Site.kind? at hpk'
      rw [if_neg hpo] at hpk'
      obtain ⟨y, hy, hyk⟩ := Option.map_eq_some_iff.mp hpk'
      have hyl : y ∈ (Site.ofLog l).locs :=
        List.mem_of_find?_eq_some (show (Site.ofLog l).locs.find? (·.id == p) = some y from hy)
      obtain ⟨-, hyp⟩ := h.parents y hyl
      rw [hyk] at hyp
      simp only [Site.Kind.parents, List.mem_singleton] at hyp
      cases hq : y.parent with
      | none => simp [hq] at hyp
      | some q => exact ⟨q, by simp [Site.host?, hpar, hpk, hy, hq]⟩
    · exact ⟨p, by simp [Site.host?, hpar, hpk]⟩

/-- **Onsite, a valid log's boundary links are certified and its memory reached** (`View.Onsite`): each memory leaf is
in a host (`Valid.host?_isSome`), which the `challengerReach` check holds reached (`Valid.covered`). -/
theorem Valid.onsite {l : Registration} (h : Valid D A c l) : (view D c l).Onsite c := by
  intro hc
  obtain ⟨hk, hm⟩ := h.covered hc
  refine ⟨hk, fun x hx hxm => ?_⟩
  obtain ⟨q, hq⟩ := h.host?_isSome hx hxm
  exact ⟨q, hq, hm x hx hxm q hq⟩

/-- **The whole Site of a valid log passes every check of a site entry** (`Site.Checked`): each site entry passed them
against the Site before it (`Site.checked_add`), and an information entry or a declaration adds nothing to it. -/
theorem Valid.checked {l : Registration} (h : Valid D A c l) : (Site.ofLog l).Checked D A c := by
  induction l using List.reverseRecOn with
  | nil => exact Site.checked_empty
  | append_singleton l e ih =>
    obtain ⟨hl, ⟨-, he⟩, -⟩ := valid_append_singleton.mp h
    rw [Site.ofLog_concat]
    cases e with
    | site _ s => exact Site.checked_add (ih hl) he
    | information _ _ => simpa [Entry.added] using ih hl
    | declaration _ _ => simpa [Entry.added] using ih hl

/-- **A valid log is chained**: each entry's `previous` is the head of the log before it, its `chain` check. -/
theorem Valid.chained {l : Registration} (h : Valid D A c l) : Chained D c l := fun i hi => (h i hi).1.1

/-- θ's values, each under its item's key, are among the keys of the log's items. -/
theorem theta_keys_sublist (log : Registration) :
    List.Sublist (log.theta.map fun t => "theta " ++ t.value) (log.items.map Item.key) := by
  unfold theta
  generalize log.items = is
  induction is with
  | nil => simp
  | cons i is ih =>
    cases i with
    | theta t => simpa [Item.key] using ih.cons_cons ("theta " ++ t.value)
    | builder b => simpa [Item.key] using ih.cons "builder"

/-- **A valid log registers each θ value once.** -/
theorem Valid.theta_nodup {l : Registration} (h : Valid D A c l) : (l.theta.map (·.value)).Nodup := by
  induction l using List.reverseRecOn with
  | nil => simp [theta, items]
  | append_singleton l e _ =>
    have ho : e.ItemsOnce l := (valid_append_singleton.mp h).2.2.1
    have hk : ((items (l ++ [e])).map Item.key).Nodup := by
      simpa [Entry.ItemsOnce, items, List.flatMap_append] using ho
    have hs := hk.sublist (theta_keys_sublist _)
    rw [show (fun t : Theta => "theta " ++ t.value) = ("theta " ++ ·) ∘ (·.value) from rfl, ← List.map_map] at hs
    exact hs.of_map _

/-- **A valid log's declarations declare each day once.** -/
theorem Valid.declarations_nodup {l : Registration} (h : Valid D A c l) :
    ((l.filterMap Entry.declared?).map (·.day)).Nodup := by
  simpa [days, List.map_filterMap] using h.days_nodup

end Registration

end Verity

namespace Verity.SecurityProofs

open Verity Registration

theorem RegistrationSound : Guarantees.RegistrationSound := by
  intro D A c log h
  exact ⟨⟨h.devices, h.onsite, ⟨h.theta_nodup, h.declarations_nodup⟩, h.checked⟩, h.chained⟩

theorem RegistrationSound_honest : Guarantees.RegistrationSound_honest := by
  unfold Guarantees.RegistrationSound_honest
  exact ⟨by decide +kernel, by simp [Witness.log, Witness.entry, Site.ofLog, Entry.added], by decide +kernel⟩

theorem RegistrationSound_nontrivial : Guarantees.RegistrationSound_nontrivial := by
  unfold Guarantees.RegistrationSound_nontrivial
  rintro ⟨hd, -⟩
  obtain ⟨k, hk, -⟩ :=
    hd { id := "gpu9", kind := .device } (by simp [view, Witness.unnamed, Site.ofLog, Entry.added]) rfl
  simp [view, Witness.unnamed, Site.ofLog, Site.sheet?, Entry.added] at hk

end Verity.SecurityProofs
