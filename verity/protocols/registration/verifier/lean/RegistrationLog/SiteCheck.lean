module

public import CoreProtocol.Site

/-!
# Registration's check on the Site

Each check a site entry passes against the Site before it (`Site.Entry.refusal`: `duplicateId`, `unknownRef`, `kind`,
`sheetDigest`, `figures`, `capacity`, `owner`, `uncertified`, `challengerReach`, in the spec's order), a registration
entry's refusal (`Site.Registration.Entry.refusal`, `previous` first), the inventory a log is held to at its end
(`Site.Registration.Complete`), the log's head (`Site.Registration.head`) and the check of a log
(`Site.Registration.check`). The Site's type, its views, the auditor's figures, the digests and the refusals' names are
the interface's (`CoreProtocol.Site`); `Properties.Core.Site` states that each check is the spec's.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core

namespace Site.Registration

/-- The log's head: the digest of its last entry, or of its composite's genesis when it is empty. -/
def head (D : Digests) (c : Composite) (log : Registration) : String :=
  match log.getLast? with
  | none => D.genesis c
  | some e => D.entry e

end Site.Registration

namespace Site.Entry

open Site.Registration (Digests Auditor Composite Refusal)

variable (D : Digests) (A : Auditor) (c : Composite) (S : Site) (e : Entry)

/-- `duplicateId`: no id is registered twice, the outside's among them, and no sheet twice. -/
def Fresh : Prop := (S.add e).ids.Nodup ∧ ((S.add e).sheets.map D.sheet).Nodup

/-- `unknownRef`: each new location's parent and each new link's ends name something registered. -/
def Refs : Prop :=
  (∀ x ∈ e.locs, ∀ p ∈ x.parent, (S.add e).Known p) ∧
    ∀ k ∈ e.links, (S.add e).Known k.ends.1 ∧ (S.add e).Known k.ends.2

/-- `kind`: parents and tiers by kind, so an auditor's device is at the site and holds only the auditor's devices; links
between kinds a link may join, each at a positive capacity or, where it says no certifier splits it, at none, and each
certifier splitting exactly one uplink. -/
def Kinds : Prop :=
  (∀ x ∈ e.locs, x.parent.bind (S.add e).kind? ∈ x.kind.parents ∧ (x.kind = .memory ↔ x.tier.isSome)) ∧
    (∀ k ∈ e.links, k.ends.1 ≠ k.ends.2 ∧ (∃ p ∈ Kind.links, (S.add e).Joins k p.1 p.2) ∧
      (∀ b ∈ k.β, 0 < b) ∧ (k.β = none → k.unmediated.isSome)) ∧
    ∀ x ∈ (S.add e).locs, x.kind = .certifier →
      (((S.add e).far x.id).map (S.add e).kind?).Perm [some .host, some .outside]

/-- `sheetDigest`: each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered
sheet's digest. -/
def Sheets : Prop :=
  (∀ k ∈ e.sheets, (A.sheet k.sku).isSome) ∧ ∀ x ∈ e.locs, x.kind = .accelerator → x.sheet ∈ (S.add e).sheets.map D.sheet

/-- `figures`: each new sheet is accepted against the auditor's sheet `a` for its SKU: its figures against `a`'s
estimate, and its memory figures exactly `a`'s. -/
def Figures : Prop :=
  ∀ k ∈ e.sheets, ∀ a ∈ A.sheet k.sku, k.figures.Accepted (A.estimate a) ∧ k.bw = a.bw ∧ k.chip = a.chip ∧ k.mem = a.mem

/-- `figures`, a host's memory: each new memory leaf directly under a host states the auditor's figures for it: the
host it is under, a rate that is not 0, and its bytes; every other new location states no rate. -/
def Rates : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .memory ∧ x.parent.bind (S.add e).kind? = some .host then
      ∃ f ∈ A.memory.lookup x.id, x.parent = some f.host ∧ x.bw = some f.bw ∧ 0 < f.bw ∧ x.bytes = f.bytes
    else x.bw = none

/-- `figures`, a host's own: each new host states the auditor's figures for it: its on-chip state in bits, and each
tier's shared rate, no shared rate 0; every other new location states neither. -/
def Chips : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .host then
      ∃ h ∈ A.hosts.lookup x.id, x.chip = some h.chip ∧ x.shared = h.shared ∧
        (h.shared.all fun (p : Tier × Nat) => decide (0 < p.2)) = true
    else x.chip = none ∧ x.shared = []

/-- `figures`, the unit of time: each new location of kind `site` states the Site's unit of time, which is not 0 and is
the one the Site states; every other new location states none. -/
def Timed : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .site then ∃ u ∈ x.timeUnit, 0 < u.1 ∧ (S.add e).timeUnit = some u else x.timeUnit = none

/-- `capacity`: each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory. -/
def Capacity : Prop :=
  (∀ x ∈ e.locs, x.kind = .memory → 0 < x.bytes) ∧
    ∀ x ∈ (S.add e).locs, x.kind = .accelerator → ∀ k ∈ (S.add e).sheet? D.sheet x, (S.add e).held x.id ≤ k.mem

/-- `owner`: each new certifier, challenger or auditor's device is the auditor's own, owned by it, with its key and its
key's scheme, one the check knows (`SCHEMES`); every other new location is the prover's and states no scheme; and each
new auditor's device states its time base, the auditor's for it, held as its key is, and the auditor's figure for its
clock, below 1 where there is one, and no other new location states a time base or a clock. -/
def Owners : Prop :=
  ∀ x ∈ e.locs,
    (x.kind.auditor = true → x.owner = .auditor ∧ A.key x.id = some x.key ∧ A.scheme x.id = some x.scheme ∧
      x.scheme ∈ SCHEMES) ∧
    (x.kind.auditor = false → x.owner = .prover ∧ x.scheme = "") ∧
    if x.kind = .auditorHost then
      (∃ t ∈ A.timebases.lookup x.id, x.timebase = some t) ∧ x.drift = A.drift.lookup x.id ∧
        (x.drift.all fun (q : Figure) => decide (q.1 < 1)) = true
    else x.timebase = none ∧ x.drift = none

/-- `uncertified`, onsite: every boundary link is certified, or says why no certifier splits it. -/
def Covered : Prop :=
  c = .onsite → ∀ k ∈ (S.add e).links, (S.add e).Boundary k → (S.add e).Certified k ∨ k.unmediated.isSome

/-- `challengerReach`, onsite: the host of every memory leaf is at an end of a memory challenger's link. -/
def Reached : Prop :=
  c = .onsite → ∀ x ∈ (S.add e).locs, x.kind = .memory → ∀ h ∈ (S.add e).host? x, h ∈ (S.add e).reached

instance instDecidableFresh : Decidable (Fresh D S e) := by unfold Fresh; infer_instance
instance instDecidableRefs : Decidable (Refs S e) := by unfold Refs; infer_instance
instance instDecidableKinds : Decidable (Kinds S e) := by unfold Kinds; infer_instance
instance instDecidableSheets : Decidable (Sheets D A S e) := by unfold Sheets; infer_instance
instance instDecidableFigures : Decidable (Figures A e) := by unfold Figures; infer_instance
instance instDecidableRates : Decidable (Rates A S e) := by unfold Rates; infer_instance
instance instDecidableTimed : Decidable (Timed S e) := by unfold Timed; infer_instance
instance instDecidableChips : Decidable (Chips A e) := by unfold Chips; infer_instance
instance instDecidableCapacity : Decidable (Capacity D S e) := by unfold Capacity; infer_instance
instance instDecidableOwners : Decidable (Owners A e) := by unfold Owners; infer_instance
instance instDecidableCovered : Decidable (Covered c S e) := by unfold Covered; infer_instance
instance instDecidableReached : Decidable (Reached c S e) := by unfold Reached; infer_instance

/-- A site entry's refusal: the first of its checks it fails, or `none` when it passes them all. -/
def refusal : Option Refusal :=
  if ¬Fresh D S e then some .duplicateId
  else if ¬Refs S e then some .unknownRef
  else if ¬Kinds S e then some .kind
  else if ¬Sheets D A S e then some .sheetDigest
  else if ¬(Figures A e ∧ Rates A S e ∧ Timed S e ∧ Chips A e) then some .figures
  else if ¬Capacity D S e then some .capacity
  else if ¬Owners A e then some .owner
  else if ¬Covered c S e then some .uncertified
  else if ¬Reached c S e then some .challengerReach
  else none

end Site.Entry

namespace Site.Registration

namespace Entry

variable (D : Digests) (A : Auditor) (c : Composite) (S : Site) (head : String)

/-- The refusal of the entry's kind: a site entry's. -/
def kindRefusal : Entry → Option Refusal
  | .site _ s => s.refusal D A c S

/-- **The refusal**: `previous` when the entry doesn't follow the log's head, else its kind's, or `none`. -/
def refusal (e : Entry) : Option Refusal :=
  if e.previous ≠ head then some .previous else kindRefusal D A c S e

end Entry

/-- `inventory`, of the Site a log folds to: under each of its hosts, every memory leaf the auditor has figures for is
registered, and so is every card it lists for the host, at least its count of devices whose sheet is of its SKU, and
every way out it lists, as a link under that id with an end at the host and the auditor's rate. A log is held to it at
its end and not entry by entry, so a later entry may add hardware under a registered host; a result at a head is over a
Site complete against the auditor's figures at that head. -/
def Complete (D : Digests) (A : Auditor) (S : Site) : Prop :=
  ∀ h ∈ S.locs, h.kind = .host →
    (∀ f ∈ A.memory, f.2.host = h.id →
      ∃ x ∈ S.locs, x.id = f.1 ∧ x.kind = .memory ∧ x.parent = some h.id) ∧
    ∀ g ∈ A.hosts.lookup h.id,
      (∀ k ∈ g.cards, k.2 ≤ (S.locs.filter fun d =>
        d.kind = .accelerator ∧ d.parent = some h.id ∧ (S.sheet? D.sheet d).map (·.sku) = some k.1).length) ∧
      ∀ w ∈ g.ways, ∃ k ∈ S.links, k.id = w.1 ∧ (k.ends.1 = h.id ∨ k.ends.2 = h.id) ∧ k.β = w.2

instance instDecidableComplete (D : Digests) (A : Auditor) (S : Site) : Decidable (Complete D A S) := by
  unfold Complete; infer_instance

/-- **The check of a log**: the Site it folds to; or the first entry refused, with its index and refusal; or, where
every entry passes and the Site is not complete against the auditor's figures (`Complete`), `inventory` at the log's
length. -/
def check (D : Digests) (A : Auditor) (c : Composite) (log : Registration) : Except (Nat × Refusal) Site :=
  go 0 ⟨[], [], []⟩ (D.genesis c) log
where
  go : Nat → Site → String → Registration → Except (Nat × Refusal) Site
    | i, S, _, [] => if Complete D A S then .ok S else .error (i, .inventory)
    | i, S, h, e :: rest =>
      match e.refusal D A c S h with
      | some r => .error (i, r)
      | none => go (i + 1) (S.add e.added) (D.entry e) rest

end Site.Registration

end Core
