module

public import Mathlib.Data.List.Induction
public import Definitions.Core.Site
import all Init.Data.Repr
import all Init.Data.ToString.Basic

/-!
# The registration: its check, and the Site along it

* `Registration.Entry.refusal_eq_none_iff`: an entry is refused exactly when it is invalid, so a refusal names a check
  the entry fails, the first in `Entry.Valid`'s order (a site entry's: `Site.Entry.refusal_eq_none_iff`).
* `Registration.valid_append_singleton`: a log with one more entry is valid exactly when the log is and the entry is
  valid against it and registers no item and declares no day it registers or declares, so a log is checked one entry
  at a time (`Registration.valid_append_singleton_iff_refusal`, by the log's refusal); and every prefix of a valid log
  is valid (`Registration.Valid.of_prefix`).
* `Registration.Valid.days_nodup`: a valid log declares each day at most once; and a day's declaration stays the day's as
  the log grows (`Registration.declared_prefix`).
* `Site.ofLog_prefix`: the Site grows along the log, a prefix's sheets, locations and links being a prefix of every
  extension's; and what an id names in a prefix's Site it names in every extension's (`Site.loc?_ofLog_prefix`,
  `Site.kind?_ofLog_prefix`), so a later entry changes nothing already registered.
* `Registration.Valid.ids_nodup` and `Registration.Valid.parents`: a valid log registers each id once, and its Site is a
  tree, each location under a registered parent of a kind its own allows.
* A small onsite site and a later entry for each check, each refused by that check's name
  (`Registration.Example.onsite_refusals`), and the same entries on a remote site, which accepts the two that only an
  onsite site refuses (`Registration.Example.remote_refusals`); both sites complete against the auditor's figures, and
  one short of a leaf not (`Registration.Example.complete`); a device's model read off the Site
  (`Registration.Example.gpu0_device`); and the onsite site with day 0 declared, a second declaration of the day refused
  (`Registration.Example.declaration_refusals`).
-/

@[expose] public section

namespace Verity

open scoped NNRat

namespace Site

@[simp] theorem ofLog_nil : ofLog [] = ⟨[], [], []⟩ := rfl

theorem ofLog_append (l l' : Registration) :
    ofLog (l ++ l') = ⟨(ofLog l).sheets ++ (ofLog l').sheets, (ofLog l).locs ++ (ofLog l').locs,
      (ofLog l).links ++ (ofLog l').links⟩ := by
  simp [ofLog]

theorem ofLog_concat (l : Registration) (e : Registration.Entry) : ofLog (l ++ [e]) = (ofLog l).add e.added := by
  simp [ofLog, add]

@[simp] theorem add_empty (S : Site) : S.add {} = S := by
  cases S; simp [add]

/-- **The Site grows along the log**: a prefix's sheets, locations and links are a prefix of every extension's. -/
theorem ofLog_prefix {l₁ l₂ : Registration} (h : l₁ <+: l₂) :
    (ofLog l₁).sheets <+: (ofLog l₂).sheets ∧ (ofLog l₁).locs <+: (ofLog l₂).locs ∧
      (ofLog l₁).links <+: (ofLog l₂).links := by
  obtain ⟨t, rfl⟩ := h
  simp [ofLog_append, List.prefix_append]

theorem loc?_of_prefix {S S' : Site} (h : S.locs <+: S'.locs) {i : String} {x : Loc} (hx : S.loc? i = some x) :
    S'.loc? i = some x := by
  obtain ⟨t, ht⟩ := h
  simp only [loc?] at hx ⊢
  simp [← ht, List.find?_append, hx]

theorem kind?_of_prefix {S S' : Site} (h : S.locs <+: S'.locs) {i : String} {k : Kind}
    (hk : S.kind? i = some k) : S'.kind? i = some k := by
  unfold kind? at *
  split_ifs at * with hi
  · exact hk
  · cases hx : S.loc? i with
    | none => simp [hx] at hk
    | some x => simpa [loc?_of_prefix h hx, hx] using hk

theorem known_of_prefix {S S' : Site} (h : S.locs <+: S'.locs) {i : String} (hi : S.Known i) : S'.Known i := by
  rcases hi with hi | hi
  · exact Or.inl hi
  · exact Or.inr (List.Sublist.subset (h.sublist.map _) hi)

theorem kind?_isSome_of_known {S : Site} {i : String} (hi : S.Known i) : (S.kind? i).isSome := by
  unfold kind?
  split_ifs with ho
  · rfl
  · rcases hi with hi | hi
    · exact absurd hi ho
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hi
      simpa [loc?, List.find?_isSome] using ⟨x, hx, rfl⟩

/-- What an id names in a prefix's Site, it names in every extension's. -/
theorem loc?_ofLog_prefix {l₁ l₂ : Registration} (h : l₁ <+: l₂) {i : String} {x : Loc}
    (hx : (ofLog l₁).loc? i = some x) : (ofLog l₂).loc? i = some x :=
  loc?_of_prefix (ofLog_prefix h).2.1 hx

theorem kind?_ofLog_prefix {l₁ l₂ : Registration} (h : l₁ <+: l₂) {i : String} {k : Kind}
    (hk : (ofLog l₁).kind? i = some k) : (ofLog l₂).kind? i = some k :=
  kind?_of_prefix (ofLog_prefix h).2.1 hk

end Site

namespace Site.Entry

open Registration (Digests Auditor Composite)

variable {D : Digests} {A : Auditor} {c : Composite}

/-- **A site entry's refusal decides its validity**: it is refused exactly when it is invalid. -/
theorem refusal_eq_none_iff {S : Site} {e : Entry} : e.refusal D A c S = none ↔ e.Valid D A c S := by
  unfold refusal Valid
  split_ifs <;> simp_all

end Site.Entry

namespace Registration

variable {D : Digests} {A : Auditor} {c : Composite}

/-- **The refusal decides validity**: an entry is refused exactly when it is invalid. -/
theorem Entry.refusal_eq_none_iff {S : Site} {head : String} {e : Entry} :
    e.refusal D A c S head = none ↔ e.Valid D A c S head := by
  cases e with
  | site _ s =>
    unfold Entry.refusal Entry.Valid Entry.kindRefusal Entry.Checks
    split_ifs <;> simp_all [Site.Entry.refusal_eq_none_iff]
  | information _ _ =>
    unfold Entry.refusal Entry.Valid Entry.kindRefusal Entry.Checks
    split_ifs <;> simp_all
  | declaration _ _ =>
    unfold Entry.refusal Entry.Valid Entry.kindRefusal Entry.Checks
    split_ifs <;> simp_all

/-- **The log's refusal decides validity**: the log refuses its next entry exactly when the entry is invalid against it,
registers an item twice or declares a day it declares. -/
theorem refusal_eq_none_iff {l : Registration} {e : Entry} :
    refusal D A c l e = none ↔ e.Valid D A c (Site.ofLog l) (head D c l) ∧ e.Once l := by
  unfold refusal Entry.Once
  rw [Option.or_eq_none_iff, Entry.refusal_eq_none_iff]
  split_ifs <;> simp_all

@[simp] theorem days_nil : days [] = [] := rfl

theorem days_concat (l : Registration) (e : Entry) :
    days (l ++ [e]) = days l ++ (e.declared?.map (·.day)).toList := by
  cases h : e.declared? <;> simp [days, List.filterMap_append, h]

@[simp] theorem head_nil : head D c [] = D.genesis c := rfl

@[simp] theorem head_concat (l : Registration) (e : Entry) : head D c (l ++ [e]) = D.entry e := by
  simp [head]

@[simp] theorem valid_nil : Valid D A c [] := by
  simp [Valid]

/-- **A log is checked one entry at a time**: with one more entry it is valid exactly when it was, and the entry is
valid against it and registers no item it registers. -/
theorem valid_append_singleton {l : Registration} {e : Entry} :
    Valid D A c (l ++ [e]) ↔ Valid D A c l ∧ e.Valid D A c (Site.ofLog l) (head D c l) ∧ e.Once l := by
  unfold Valid
  constructor
  · intro h
    refine ⟨fun i hi => ?_, ?_⟩
    · have := h i (by simp; omega)
      simpa [List.getElem_append_left hi, List.take_append_of_le_length (Nat.le_of_lt hi)] using this
    · simpa using h l.length (by simp)
  · rintro ⟨h, he⟩ i hi
    rcases Nat.lt_or_ge i l.length with hi' | hi'
    · simpa [List.getElem_append_left hi', List.take_append_of_le_length (Nat.le_of_lt hi')] using h i hi'
    · obtain rfl : i = l.length := by simp at hi; omega
      simpa using he

/-- **Every prefix of a valid log is valid**: an entry valid against its prefix stays valid in every extension of the
log. -/
theorem Valid.of_prefix {l₁ l₂ : Registration} (h : l₁ <+: l₂) (hv : Valid D A c l₂) : Valid D A c l₁ := by
  obtain ⟨t, rfl⟩ := h
  unfold Valid at hv ⊢
  intro i hi
  have := hv i (by simp; omega)
  simpa [List.getElem_append_left hi, List.take_append_of_le_length (Nat.le_of_lt hi)] using this

/-- **A valid log with one more entry is valid exactly when the log refuses no entry**: the check of a log is its
refusal of each entry in turn. -/
theorem valid_append_singleton_iff_refusal {l : Registration} {e : Entry} :
    Valid D A c (l ++ [e]) ↔ Valid D A c l ∧ refusal D A c l e = none := by
  rw [valid_append_singleton, refusal_eq_none_iff]

/-- **A valid log declares each day at most once.** -/
theorem Valid.days_nodup {l : Registration} (h : Valid D A c l) : (days l).Nodup := by
  induction l using List.reverseRecOn with
  | nil => simp
  | append_singleton l e ih =>
    obtain ⟨hl, -, ho⟩ := valid_append_singleton.mp h
    have ho : e.DayOnce l := ho.2
    rw [days_concat, List.nodup_append]
    refine ⟨ih hl, ?_, ?_⟩
    · cases e.declared? <;> simp
    · intro a ha b hb
      cases hd : e.declared? with
      | none => simp [hd] at hb
      | some d =>
        have hn : d.day ∉ days l := ho d hd
        simp [hd] at hb
        rintro rfl
        exact hn (hb ▸ ha)

/-- **A day's declaration stays the day's**: once the log declares a circuit for day `j`, every extension of it declares
the same circuit for day `j`, and in a valid log it is the only declaration of the day (`Valid.days_nodup`). -/
theorem declared_prefix {l₁ l₂ : Registration} (h : l₁ <+: l₂) {j : ℕ} {d : Declared}
    (hd : declared l₁ j = some d) : declared l₂ j = some d := by
  obtain ⟨t, rfl⟩ := h
  unfold declared at hd ⊢
  rw [List.findSome?_append, hd]
  rfl

/-- **A valid log registers each id once**, the outside's among them. -/
theorem Valid.ids_nodup {l : Registration} (h : Valid D A c l) : (Site.ofLog l).ids.Nodup := by
  induction l using List.reverseRecOn with
  | nil => simp [Site.ids]
  | append_singleton l e ih =>
    rw [Site.ofLog_concat]
    obtain ⟨hl, ⟨-, he⟩, -⟩ := valid_append_singleton.mp h
    cases e with
    | site _ s => exact he.1.1
    | information _ _ => simpa [Entry.added] using ih hl
    | declaration _ _ => simpa [Entry.added] using ih hl

/-- **A valid log's Site is a tree**: each location's parent is registered, and of a kind the location's own allows. -/
theorem Valid.parents {l : Registration} (h : Valid D A c l) :
    ∀ x ∈ (Site.ofLog l).locs, (∀ p ∈ x.parent, (Site.ofLog l).Known p) ∧
      x.parent.bind (Site.ofLog l).kind? ∈ x.kind.parents := by
  induction l using List.reverseRecOn with
  | nil => simp
  | append_singleton l e ih =>
    obtain ⟨hl, ⟨-, he⟩, -⟩ := valid_append_singleton.mp h
    cases e with
    | information _ _ => simpa [Site.ofLog_concat, Entry.added] using ih hl
    | declaration _ _ => simpa [Site.ofLog_concat, Entry.added] using ih hl
    | site _ s => ?_
    have hpre : (Site.ofLog l).locs <+: ((Site.ofLog l).add s).locs := by simp [Site.add, List.prefix_append]
    rw [Site.ofLog_concat, Entry.added]
    intro x hx
    rcases List.mem_append.mp hx with hx | hx
    · obtain ⟨hk, hp⟩ := ih hl x hx
      refine ⟨fun p hp' => Site.known_of_prefix hpre (hk p hp'), ?_⟩
      cases hpar : x.parent with
      | none => simpa [hpar] using hp
      | some p =>
        obtain ⟨k, hk'⟩ := Option.isSome_iff_exists.mp (Site.kind?_isSome_of_known (hk p (by simp [hpar])))
        simpa [hpar, hk', Site.kind?_of_prefix hpre hk'] using hp
    · exact ⟨he.2.1.1 x hx, (he.2.2.1.1 x hx).1⟩

namespace Example

open Site (Sheet Loc Link Tier outside)

/-- Figures for an SKU: a tensor clock of `clock` Hz, and `rate` MACs a second at every operand class. -/
def stated (sku : String) (clock rate : ℚ≥0) : DeviceFigures := ⟨sku, clock, fun _ => rate⟩

/-- Two sheets, an SKU each. -/
def sheetA : Sheet := ⟨stated "sim-int8-a" 1000 64000, 3200, 2 ^ 25, 64 * 2 ^ 30⟩
def sheetB : Sheet := ⟨stated "sim-int8-b" 2000 256000, 6400, 2 ^ 26, 128 * 2 ^ 30⟩

/-- Digests that tell the cases apart: a sheet's SKU, bandwidth, clock and s8 rate, and an entry's head one mark longer
than the one it follows. -/
def digests : Digests :=
  ⟨fun k => k.sku ++ "/" ++ toString k.bw ++ "/" ++ toString k.figures.clock.num ++ "/" ++
      toString (k.figures.peakRate .uniformS8).num, (·.previous ++ "+"), fun _ => "genesis"⟩

/-- The scheme the example's keys are of. -/
def lms : String := "lms-sha512/n64-w1-h20/v0"

/-- The auditor's figures for the host `node0`: its on-chip state, its two cards, its DRAM's shared rate and its
uplink. -/
def node0Figures : Site.HostFigures :=
  { chip := 2 ^ 28
    cards := [("sim-int8-a", 1), ("sim-int8-b", 1)]
    shared := [(.dram, 2 ^ 38)]
    ways := [("uplink0", some 100000)] }

/-- The auditor's data sheets for the two SKUs and a tenth's tolerance around their figures; the keys of its certifier,
its challenger and its host `vm0`, each of the scheme `lms`; its figures for the host `node0` and for its DRAM and disk
leaves; its figures for a host `node1` and its DRAM, which a later entry registers; and `vm0`'s time base and its
clock's drift, one part in two thousand. -/
def auditor : Auditor :=
  { sheet := fun s => [sheetA, sheetB].find? (·.sku == s)
    tolerance := ⟨1 / 10⟩
    key := fun i => if i = "cert0" then some "key0" else if i = "chal0" then some "" else
      if i = "vm0" then some "key1" else none
    scheme := fun i => if i ∈ ["cert0", "chal0", "vm0"] then some lms else none
    memory := [("dram0", ⟨"node0", 2 ^ 37, 512 * 2 ^ 30⟩), ("nvme0", ⟨"node0", 2 ^ 35, 2 ^ 40⟩),
      ("dram1", ⟨"node1", 2 ^ 37, 1⟩)]
    hosts := [("node0", node0Figures), ("node1", { chip := 2 ^ 28 })]
    drift := [("vm0", 1 / 2000)]
    timebases := [("vm0", "tb0")] }

/-- A device under `parent` with sheet `k`. -/
def device (id parent : String) (k : Sheet) : Loc :=
  { id, kind := .device, parent := some parent, sheet := digests.sheet k }

/-- A memory leaf of tier `t` under the device `parent`, holding `bytes`. -/
def memory (id parent : String) (t : Tier) (bytes : ℕ) : Loc :=
  { id, kind := .memory, parent := some parent, tier := some t, bytes }

/-- A memory leaf of tier `t` directly under the host `parent`, read at `bw` bits a second and holding `bytes`. -/
def leaf (id parent : String) (t : Tier) (bw bytes : ℕ) : Loc :=
  { memory id parent t bytes with bw := some bw }

/-- A host under `site0` with `chip` bits of on-chip state. -/
def host (id : String) (chip : ℕ) (shared : List (Tier × ℕ) := []) : Loc :=
  { id, kind := .host, parent := some "site0", chip := some chip, shared }

/-- One of the auditor's own under `parent`, with its key, of the scheme `lms`. -/
def own (id : String) (kind : Site.Kind) (parent key : String) : Loc :=
  { id, kind, parent := some parent, owner := .auditor, key, scheme := lms }

/-- One node at a site whose unit of time is a millisecond, with two devices, each with its HBM, and the node's DRAM
and disk. -/
def node : List Loc :=
  [{ id := "site0", kind := .site, timeUnit := some (1 / 1000) }, host "node0" (2 ^ 28) [(.dram, 2 ^ 38)],
   device "gpu0" "node0" sheetA, memory "hbm0" "gpu0" .hbm (64 * 2 ^ 30),
   device "gpu1" "node0" sheetB, memory "hbm1" "gpu1" .hbm (128 * 2 ^ 30),
   leaf "dram0" "node0" .dram (2 ^ 37) (512 * 2 ^ 30), leaf "nvme0" "node0" .disk (2 ^ 35) (2 ^ 40)]

/-- A link between two ends at `β` bits a second. -/
def link (id : String) (ends : String × String) (β latency : ℕ) : Link := { id, ends, β := some β, latency }

/-- The onsite site's first entry: the node, its uplink split by a certifier, and a memory challenger linked to it. -/
def onsite : Entry := .site "genesis"
  { sheets := [sheetA, sheetB]
    locs := node ++ [own "cert0" .certifier "site0" "key0", own "chal0" .challenger "site0" ""]
    links := [link "uplink0" ("node0", "cert0") 100000 100000, link "uplink0-out" ("cert0", outside) 100000 0,
      link "chal0-link" ("chal0", "node0") 100000 1000000] }

/-- The remote site's first entry: the node and its uplink, which no certifier splits. -/
def remote : Entry := .site "genesis"
  { sheets := [sheetA, sheetB]
    locs := node
    links := [link "uplink0" ("node0", outside) 100000 100000] }

/-- A site entry following the first, or `previous`. -/
def next (locs : List Loc) (links : List Link := []) (sheets : List Sheet := []) (previous := "genesis+") : Entry :=
  .site previous ⟨sheets, locs, links⟩

/-- The auditor's host `vm0`, at the site, with its time base and its clock's drift. -/
def vm0 : Loc := { own "vm0" .auditorHost "site0" "key1" with timebase := some "tb0", drift := some (1 / 2000) }

/-- A later entry for each check, with the refusal it samples on the onsite site. -/
def later : List (Entry × Option Refusal) :=
  [(next [device "gpu2" "node0" sheetB, memory "hbm2" "gpu2" .hbm (128 * 2 ^ 30)], none),
   (next [device "gpu2" "node0" sheetB] (previous := "genesis"), some .previous),
   (next [device "gpu0" "node0" sheetA], some .duplicateId),
   (next [memory "hbm9" "gpu9" .hbm 1], some .unknownRef),
   (next [device "gpu9" "site0" sheetA], some .kind),
   (next [] [link "l9" ("node0", "node0") 1 1], some .kind),
   (next [{ vm0 with parent := some "node0" }], some .kind),
   (next [] [{ id := "u9", ends := ("node0", outside), β := none, latency := 1 }], some .kind),
   (next [] [{ id := "u9", ends := ("node0", outside), β := none, latency := 1, unmediated := some "no line rate" }],
     none),
   (next [] [] [{ sheetA with figures := stated "sim-int8-z" 1000 64000 }], some .sheetDigest),
   (next [] [] [{ sheetA with figures := stated "sim-int8-a" 1050 60000 }], none),
   (next [] [] [{ sheetA with figures := stated "sim-int8-a" 1200 64000 }], some .figures),
   (next [] [] [{ sheetA with figures := stated "sim-int8-a" 1000 0 }], some .figures),
   (next [] [] [{ sheetA with bw := 3201 }], some .figures),
   (next [leaf "dram9" "node0" .dram (2 ^ 37) 1], some .figures),
   (next [host "node9" (2 ^ 28)], some .figures),
   (next [{ id := "site1", kind := .site, timeUnit := some 1 }], some .figures),
   (next [{ device "gpu9" "node0" sheetA with sheet := "00" }], some .sheetDigest),
   (next [device "gpu9" "node0" sheetA, memory "hbm9" "gpu9" .hbm (64 * 2 ^ 30 + 1)], some .capacity),
   (next [device "gpu9" "node0" sheetA, memory "hbm9" "gpu9" .hbm 0], some .capacity),
   (next [own "cert9" .certifier "site0" "00"]
      [link "u9" ("node0", "cert9") 1 1, link "u9-out" ("cert9", outside) 1 0], some .owner),
   (next [{ device "gpu9" "node0" sheetA with owner := .auditor }], some .owner),
   (next [vm0], none),
   (next [{ vm0 with timebase := some "tb1" }], some .owner),
   (next [{ vm0 with drift := none }], some .owner),
   (next [] [link "u9" ("node0", outside) 1 1], some .uncertified),
   (next [host "node1" (2 ^ 28), leaf "dram1" "node1" .dram (2 ^ 37) 1] [link "n01" ("node0", "node1") 1 1],
     some .challengerReach)]

theorem onsite_valid : Valid digests auditor .onsite [onsite] := by decide +kernel

theorem remote_valid : Valid digests auditor .remote [remote] := by decide +kernel

/-- The remote site's first entry without the node's disk leaf, which the auditor has figures for. -/
def short : Entry := .site "genesis"
  { sheets := [sheetA, sheetB]
    locs := node.dropLast
    links := [link "uplink0" ("node0", outside) 100000 100000] }

/-- Both sites are complete against the auditor's figures: `node0`'s two leaves, its two cards and its uplink are
registered. Without its disk leaf the remote site is not. -/
theorem complete :
    Complete digests auditor (Site.ofLog [onsite]) ∧ Complete digests auditor (Site.ofLog [remote]) ∧
      ¬Complete digests auditor (Site.ofLog [short]) := by
  decide +kernel

/-- gpu0's device model, read off the onsite Site at its unit of time, a millisecond: its sheet's 64000 MACs a second
are 64 a tick and its 3200 bits a second from device memory 4 a tick, each rounded up, and its memory figures are its
sheet's, its device memory in bits. -/
theorem gpu0_device :
    ((Site.ofLog [onsite]).device digests.sheet auditor (device "gpu0" "node0" sheetA) 0 0
      (fun _ : Unit => 0)).map (fun d => (d.rate .tc, d.bw, d.chip, d.mem)) =
      some (64, 4, 2 ^ 25, 8 * (64 * 2 ^ 30)) := by
  decide +kernel

/-- Each check refuses its case on the onsite site, by its name. -/
theorem onsite_refusals :
    later.map (fun x => x.1.refusal digests auditor .onsite (Site.ofLog [onsite]) (head digests .onsite [onsite])) =
      later.map (·.2) := by
  decide +kernel

/-- The remote site refuses each case as the onsite site does, but accepts a bare uplink and a host no challenger
reaches. -/
theorem remote_refusals :
    later.map (fun x => x.1.refusal digests auditor .remote (Site.ofLog [remote]) (head digests .remote [remote])) =
      later.map (fun x => if x.2 = some .uncertified ∨ x.2 = some .challengerReach then none else x.2) := by
  decide +kernel

/-- Day 0's circuit: a program and its one block Definition. -/
def day0 : Declared := ⟨0, "program0", ["block0"]⟩

/-- The onsite site, then its declaration of day 0. -/
def declaredLog : Registration := [onsite, .declaration "genesis+" day0]

/-- The onsite site with day 0 declared is valid, and day 0's declared circuit is `day0`, a nonempty one. -/
theorem declaredLog_valid : Valid digests auditor .onsite declaredLog ∧ declared declaredLog 0 = some day0 ∧
    day0.blocks ≠ [] := by
  refine ⟨by decide +kernel, rfl, by simp [day0]⟩

/-- A second declaration of day 0 is refused, by `duplicateDeclaration`, and a declaration of day 1 is not; a
declaration that follows no head is refused by `previous` first. -/
theorem declaration_refusals :
    [Entry.declaration "genesis++" ⟨0, "program1", ["block0"]⟩, .declaration "genesis++" ⟨1, "program1", ["block1"]⟩,
      .declaration "genesis+" ⟨0, "program1", ["block0"]⟩].map (refusal digests auditor .onsite declaredLog) =
      [some .duplicateDeclaration, none, some .previous] := by
  decide +kernel

/-- A window within day 0 of a monitoring period cut into days runs day 0's declared program, and runs no other. -/
theorem runsDeclared_day0 (m : MonitoringPeriod) (cut : m.span.Cut 1) :
    RunsDeclared declaredLog m cut (cut.part 0) "program0" ∧ ¬RunsDeclared declaredLog m cut (cut.part 0) "program1" := by
  refine ⟨⟨0, ⟨le_refl _, le_refl _⟩, day0, rfl, rfl⟩, ?_⟩
  rintro ⟨j, -, d, hd, hp⟩
  obtain rfl : j = 0 := Subsingleton.elim _ _
  obtain rfl : d = day0 := Option.some_injective _ (hd.symm.trans rfl)
  simp [day0] at hp

end Example

end Registration

end Verity
