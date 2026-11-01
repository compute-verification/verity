import Mathlib.Data.List.Induction
import Definitions.Core.Site

/-!
# The registration: its check, and the Site along it

* `Registration.Entry.refusal_eq_none_iff`: an entry is refused exactly when it is invalid, so a refusal names a check
  the entry fails, the first in `Entry.Valid`'s order (a site entry's: `Site.Entry.refusal_eq_none_iff`).
* `Registration.valid_append_singleton`: a log with one more entry is valid exactly when the log is and the entry is
  valid against it, so a log is checked one entry at a time; and every prefix of a valid log is valid
  (`Registration.Valid.of_prefix`).
* `Site.ofLog_prefix`: the Site grows along the log, a prefix's sheets, locations and links being a prefix of every
  extension's; and what an id names in a prefix's Site it names in every extension's (`Site.loc?_ofLog_prefix`,
  `Site.kind?_ofLog_prefix`), so a later entry changes nothing already registered.
* `Registration.Valid.ids_nodup` and `Registration.Valid.parents`: a valid log registers each id once, and its Site is a
  tree, each location under a registered parent of a kind its own allows.
* A small onsite site and a later entry for each check, each refused by that check's name
  (`Registration.Example.onsite_refusals`), and the same entries on a remote site, which accepts the two that only an
  onsite site refuses (`Registration.Example.remote_refusals`).
-/

namespace Verity

namespace Site

@[simp] theorem ofLog_nil : ofLog [] = ⟨[], [], []⟩ := rfl

theorem ofLog_append (l l' : Registration) :
    ofLog (l ++ l') = ⟨(ofLog l).sheets ++ (ofLog l').sheets, (ofLog l).locs ++ (ofLog l').locs,
      (ofLog l).links ++ (ofLog l').links⟩ := by
  simp [ofLog]

theorem ofLog_concat (l : Registration) (e : Registration.Entry) : ofLog (l ++ [e]) = (ofLog l).add e.added := by
  simp [ofLog, add]

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

@[simp] theorem head_nil : head D c [] = D.genesis c := rfl

@[simp] theorem head_concat (l : Registration) (e : Entry) : head D c (l ++ [e]) = D.entry e := by
  simp [head]

@[simp] theorem valid_nil : Valid D A c [] := by
  simp [Valid]

/-- **A log is checked one entry at a time**: with one more entry it is valid exactly when it was, and the entry is
valid against it. -/
theorem valid_append_singleton {l : Registration} {e : Entry} :
    Valid D A c (l ++ [e]) ↔ Valid D A c l ∧ e.Valid D A c (Site.ofLog l) (head D c l) := by
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

/-- **A valid log registers each id once**, the outside's among them. -/
theorem Valid.ids_nodup {l : Registration} (h : Valid D A c l) : (Site.ofLog l).ids.Nodup := by
  rcases List.eq_nil_or_concat' l with rfl | ⟨l', e, rfl⟩
  · simp [Site.ids]
  · rw [Site.ofLog_concat]
    obtain ⟨-, -, he⟩ := valid_append_singleton.mp h
    cases e with
    | site _ s => exact he.1.1

/-- **A valid log's Site is a tree**: each location's parent is registered, and of a kind the location's own allows. -/
theorem Valid.parents {l : Registration} (h : Valid D A c l) :
    ∀ x ∈ (Site.ofLog l).locs, (∀ p ∈ x.parent, (Site.ofLog l).Known p) ∧
      x.parent.bind (Site.ofLog l).kind? ∈ x.kind.parents := by
  induction l using List.reverseRecOn with
  | nil => simp
  | append_singleton l e ih =>
    obtain ⟨hl, -, he⟩ := valid_append_singleton.mp h
    cases e with | site _ s => ?_
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

/-- Two sheets, an SKU each. -/
def sheetA : Sheet := ⟨"sim-int8-a", 64, 3200, 2 ^ 25, 64 * 2 ^ 30⟩
def sheetB : Sheet := ⟨"sim-int8-b", 128, 6400, 2 ^ 26, 128 * 2 ^ 30⟩

/-- Digests that tell the cases apart: a sheet's SKU and rate, and an entry's head one mark longer than the one it
follows. -/
def digests : Digests := ⟨fun k => k.sku ++ "/" ++ toString k.tc, (·.previous ++ "+"), fun _ => "genesis"⟩

/-- The auditor's data sheets for the two SKUs, and its certifier's and challenger's keys. -/
def auditor : Auditor :=
  ⟨fun s => [sheetA, sheetB].find? (·.sku == s), fun i => if i = "cert0" then some "key0" else
    if i = "chal0" then some "" else none⟩

/-- A device under `parent` with sheet `k`. -/
def device (id parent : String) (k : Sheet) : Loc :=
  { id, kind := .device, parent := some parent, sheet := digests.sheet k }

/-- A memory leaf of tier `t` under `parent`, holding `bytes`. -/
def memory (id parent : String) (t : Tier) (bytes : ℕ) : Loc :=
  { id, kind := .memory, parent := some parent, tier := some t, bytes }

/-- One node with two devices, each with its HBM, and the node's DRAM and disk. -/
def node : List Loc :=
  [{ id := "site0", kind := .site }, { id := "node0", kind := .host, parent := some "site0" },
   device "gpu0" "node0" sheetA, memory "hbm0" "gpu0" .hbm (64 * 2 ^ 30),
   device "gpu1" "node0" sheetB, memory "hbm1" "gpu1" .hbm (128 * 2 ^ 30),
   memory "dram0" "node0" .dram (512 * 2 ^ 30), memory "nvme0" "node0" .disk (2 ^ 40)]

/-- The onsite site's first entry: the node, its uplink split by a certifier, and a memory challenger linked to it. -/
def onsite : Entry := .site "genesis"
  { sheets := [sheetA, sheetB]
    locs := node ++
      [{ id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key := "key0" },
       { id := "chal0", kind := .challenger, parent := some "site0", owner := .auditor }]
    links := [⟨"uplink0", ("node0", "cert0"), 100000, 100000⟩, ⟨"uplink0-out", ("cert0", outside), 100000, 0⟩,
      ⟨"chal0-link", ("chal0", "node0"), 100000, 1000000⟩] }

/-- The remote site's first entry: the node and its uplink, which no certifier splits. -/
def remote : Entry := .site "genesis"
  { sheets := [sheetA, sheetB]
    locs := node
    links := [⟨"uplink0", ("node0", outside), 100000, 100000⟩] }

/-- A site entry following the first, or `previous`. -/
def next (locs : List Loc) (links : List Link := []) (sheets : List Sheet := []) (previous := "genesis+") : Entry :=
  .site previous ⟨sheets, locs, links⟩

/-- A later entry for each check, with the refusal it draws on the onsite site. -/
def later : List (Entry × Option Refusal) :=
  [(next [device "gpu2" "node0" sheetB, memory "hbm2" "gpu2" .hbm (128 * 2 ^ 30),
      memory "nvme1" "node0" .disk (2 ^ 40)], none),
   (next [device "gpu2" "node0" sheetB] (previous := "genesis"), some .previous),
   (next [device "gpu0" "node0" sheetA], some .duplicateId),
   (next [memory "hbm9" "gpu9" .hbm 1], some .unknownRef),
   (next [device "gpu9" "site0" sheetA], some .kind),
   (next [] [⟨"l9", ("node0", "node0"), 1, 1⟩], some .kind),
   (next [] [] [{ sheetA with tc := 2 ^ 20 }], some .sheetDigest),
   (next [{ device "gpu9" "node0" sheetA with sheet := "00" }], some .sheetDigest),
   (next [device "gpu9" "node0" sheetA, memory "hbm9" "gpu9" .hbm (64 * 2 ^ 30 + 1)], some .capacity),
   (next [memory "dram9" "node0" .dram 0], some .capacity),
   (next [{ id := "cert9", kind := .certifier, parent := some "site0", owner := .auditor, key := "00" }]
      [⟨"u9", ("node0", "cert9"), 1, 1⟩, ⟨"u9-out", ("cert9", outside), 1, 0⟩], some .owner),
   (next [{ id := "node9", kind := .host, parent := some "site0", owner := .auditor }], some .owner),
   (next [] [⟨"u9", ("node0", outside), 1, 1⟩], some .uncertified),
   (next [{ id := "node1", kind := .host, parent := some "site0" }, memory "dram1" "node1" .dram 1]
      [⟨"n01", ("node0", "node1"), 1, 1⟩], some .challengerReach)]

theorem onsite_valid : Valid digests auditor .onsite [onsite] := by decide

theorem remote_valid : Valid digests auditor .remote [remote] := by decide

/-- Each check refuses its case on the onsite site, by its name. -/
theorem onsite_refusals :
    later.map (fun x => x.1.refusal digests auditor .onsite (Site.ofLog [onsite]) (head digests .onsite [onsite])) =
      later.map (·.2) := by
  decide

/-- The remote site refuses each case as the onsite site does, but accepts a bare uplink and a host no challenger
reaches. -/
theorem remote_refusals :
    later.map (fun x => x.1.refusal digests auditor .remote (Site.ofLog [remote]) (head digests .remote [remote])) =
      later.map (fun x => if x.2 = some .uncertified ∨ x.2 = some .challengerReach then none else x.2) := by
  decide

end Example

end Registration

end Verity
