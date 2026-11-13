module

public import Definitions.Core.Guarantees.Registration
public import Proofs.Core.Site

/-!
# What registration establishes: proofs

`RegistrationSound`, one registered fact at a time, each by induction along the log:

* `Registration.Valid.sheets_accepted`: every sheet of a valid log's Site is accepted against the auditor's sheet for its
  SKU, since the site entry that added it passed `sheetDigest` (the auditor has a sheet for the SKU) and `figures`; and
  `Registration.Valid.devices_named`: every device names a registered sheet's digest, since `sheetDigest` checked it
  against the Site the entry extends, and the Site only grows. So every device's sheet lookup finds an accepted sheet
  (`Registration.Valid.devices`).
* `Registration.Valid.onsite`: a site entry's `uncertified` and `challengerReach` checks read the whole Site it extends
  the log's to, and an information entry or a declaration adds nothing to it.
* `Registration.Valid.theta_nodup`: the last entry's `duplicateInformation` check reads every item the log registers, a
  θ value by its name; and `Registration.Valid.declarations_nodup`: the log declares each day once
  (`Registration.Valid.days_nodup`).

`RegistrationSound_honest`: the onsite example's log, with day 0 declared, is valid, at a Site with a device.
`RegistrationSound_nontrivial`: the view of a log with a device naming no sheet holds no registered facts, so the
conclusion fails for a check that accepts every log.
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

/-- **Onsite, a valid log's boundary links are certified and its memory reached** (`View.Onsite`). -/
theorem Valid.onsite {l : Registration} (h : Valid D A c l) : (view D c l).Onsite c := by
  unfold View.Onsite
  induction l using List.reverseRecOn with
  | nil => intro _; simp [view]
  | append_singleton l e ih =>
    obtain ⟨hl, ⟨-, he⟩, -⟩ := valid_append_singleton.mp h
    intro hc
    cases e with
    | site _ s =>
      simp only [view, Site.ofLog_concat, Entry.added]
      exact ⟨he.2.2.2.2.2.2.2.1 hc, he.2.2.2.2.2.2.2.2 hc⟩
    | information _ _ => simpa [view, Site.ofLog_concat, Entry.added] using ih hl hc
    | declaration _ _ => simpa [view, Site.ofLog_concat, Entry.added] using ih hl hc

/-- θ's values, each under its item's key, are among the keys of the log's items. -/
theorem theta_keys_sublist (log : Registration) :
    List.Sublist (log.theta.map fun t => "theta " ++ t.value) (log.items.map Item.key) := by
  unfold theta
  generalize log.items = is
  induction is with
  | nil => simp
  | cons i is ih =>
    cases i with
    | theta t => simpa [Item.key] using ih.cons₂ ("theta " ++ t.value)
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

/-- A log whose one entry registers a device naming no sheet. -/
def unnamed : Registration := [.site "genesis" { locs := [{ id := "gpu9", kind := .device }] }]

end Registration

end Verity

namespace Verity.SecurityProofs

open Verity Registration

theorem RegistrationSound : Guarantees.RegistrationSound := by
  intro D A c log h
  exact ⟨h.devices, h.onsite, h.theta_nodup, h.declarations_nodup⟩

/-- **`RegistrationSound`'s honest lemma: a valid log at a nonempty Site.** The onsite example's log, with day 0
declared, passes the check, and its Site has a device. -/
theorem RegistrationSound_honest :
    Valid Example.digests Example.auditor .onsite Example.declaredLog ∧
      Example.device "gpu0" "node0" Example.sheetA ∈ (Site.ofLog Example.declaredLog).locs :=
  ⟨Example.declaredLog_valid.1, by simp [Example.declaredLog, Example.onsite, Example.node, Site.ofLog, Entry.added]⟩

/-- **`RegistrationSound`'s nontrivial lemma: its conclusion fails at a check that accepts everything.** A log whose
device names no sheet: the device's sheet lookup finds none, so its view holds no registered facts. -/
theorem RegistrationSound_nontrivial :
    ¬(view Example.digests .onsite unnamed).Holds Example.digests Example.auditor .onsite := by
  rintro ⟨hd, -⟩
  obtain ⟨k, hk, -⟩ := hd { id := "gpu9", kind := .device } (by simp [view, unnamed, Site.ofLog, Entry.added]) rfl
  simp [view, unnamed, Site.ofLog, Site.sheet?, Entry.added] at hk

end Verity.SecurityProofs
