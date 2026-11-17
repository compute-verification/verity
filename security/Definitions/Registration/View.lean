module

public import Definitions.Core.Site
public import Definitions.Compose.Joint

/-! What the theorems read of an accepted registration log and when the registered hardware is accounted for; `Registration.Valid` doesn't check `AccountedFor`, so no guarantee here concludes it. -/

@[expose] public section

namespace Verity

namespace Site

/-- **What a part of the registered hardware is of**: one of the device model's resources at the leaf's tier, or a
boundary link's egress, which is no resource of the device model. -/
inductive Accountable
  | resource (r : ResourceAccounting.Resource)
  | egress

/-- The site, a host and the auditor's devices hold no part of their own: a host has no compute rate, and its memory is
its memory leaves'. -/
def parts (S : Site) : Set (String × Accountable) :=
  {xr | (∃ x ∈ S.locs, x.kind = .accelerator ∧ xr = (x.id, .resource .compute)) ∨
    (∃ x ∈ S.locs, x.kind = .memory ∧ (xr = (x.id, .resource .capacity) ∨ xr = (x.id, .resource .bandwidth))) ∨
    ∃ k ∈ S.links, S.Boundary k ∧ xr = (k.id, .egress)}

def asEntry (S : Site) : Entry := ⟨S.sheets, S.locs, S.links⟩

/-- Each site entry is checked against the Site it extends, so this is what holds of the whole Site at a valid log's
head. -/
def Checked (D : Registration.Digests) (A : Registration.Auditor) (c : Registration.Composite) (S : Site) : Prop :=
  Entry.Valid D A c ⟨[], [], []⟩ S.asEntry

/-- `none` when no auditor's device holds `k` or more than one does; the key field of any other location is read by
nothing. -/
def holder (S : Site) (k : String) : Option Loc :=
  match S.locs.filter fun x => x.kind.auditor && x.key == k with
  | [x] => some x
  | _ => none

def hostLinks (S : Site) (i : String) : List Link :=
  S.links.filter fun k =>
    (k.ends.1 = i ∧ S.kind? k.ends.2 = some .host) ∨ (k.ends.2 = i ∧ S.kind? k.ends.1 = some .host)

def hostsOf (S : Site) (i : String) : List String :=
  (S.hostLinks i).map fun k => if k.ends.1 = i then k.ends.2 else k.ends.1

/-- A network attester reaches its links to a host, a memory challenger each location in a host its links join it to; no
other location reaches anything. -/
def Reaches (S : Site) (x : Loc) (i : String) : Prop :=
  (x.kind = .certifier ∧ i ∈ (S.hostLinks x.id).map (·.id)) ∨
    (x.kind = .challenger ∧ ∃ y ∈ S.locs, y.id = i ∧ ∃ h ∈ S.host? y, h ∈ S.hostsOf x.id)

end Site

namespace Registration

open Site (Sheet Loc)

/-- Shortest prefix first: the empty prefix's head, the composite's genesis, then one for each entry. -/
def heads (D : Digests) (c : Composite) (log : Registration) : List (ℕ × String) :=
  (List.range (log.length + 1)).map fun n => (n, head D c (log.take n))

/-- What the theorems read of a registration log, at its head; the code's `Core.Registration.View` is its copy. -/
structure View where
  site : Site
  theta : List Theta
  builder : Option Builder
  declarations : List Declared
  head : String
  heads : List (ℕ × String)

def view (D : Digests) (c : Composite) (log : Registration) : View where
  site := Site.ofLog log
  theta := log.theta
  builder := log.builder?
  declarations := log.filterMap Entry.declared?
  head := head D c log
  heads := heads D c log

/-- `h` is the head of a prefix of the log and `P` holds of that prefix's view. That the head names one prefix is the
code's (`Properties.Registration.HeadBinds`). -/
def AtHead (D : Digests) (c : Composite) (log : Registration) (h : String) (P : View → Prop) : Prop :=
  ∃ n, (n, h) ∈ (view D c log).heads ∧ P (view D c (log.take n))

/-- **A reading of a registered device's figures**: as its registered sheet states them, or as the bound reads them,
figure by figure the larger of those and the auditor's estimate (`DeviceFigures.bounding`). -/
inductive Reading
  | stated
  | bound
  deriving DecidableEq, Repr

namespace View

variable (v : View)

/-- `none` when `x` names no registered sheet or the auditor has no sheet for its SKU. `Site.device` reads `bound`. -/
def figures (digest : Sheet → String) (A : Auditor) (r : Reading) (x : Loc) : Option DeviceFigures :=
  (v.site.sheet? digest x).bind fun k => (A.sheet k.sku).map fun a =>
    match r with
    | .stated => k.figures
    | .bound => k.figures.bounding (A.estimate a)

/-- So both readings of each device's figures exist, and its device model (`Site.device`) is defined. -/
def Devices (D : Digests) (A : Auditor) : Prop :=
  ∀ x ∈ v.site.locs, x.kind = .accelerator →
    ∃ k ∈ v.site.sheet? D.sheet x, ∃ a ∈ A.sheet k.sku,
      k.figures.Accepted (A.estimate a) ∧ k.bw = a.bw ∧ k.chip = a.chip ∧ k.mem = a.mem

def Onsite (c : Composite) : Prop :=
  c = .onsite → (∀ k ∈ v.site.links, v.site.Boundary k → v.site.Certified k) ∧
    ∀ x ∈ v.site.locs, x.kind = .memory → ∃ h, v.site.host? x = some h ∧ h ∈ v.site.reached

/-- So a lookup by name or by day reads the one registered value or circuit. -/
def Once : Prop := (v.theta.map (·.value)).Nodup ∧ (v.declarations.map (·.day)).Nodup

def Holds (D : Digests) (A : Auditor) (c : Composite) : Prop :=
  v.Devices D A ∧ v.Onsite c ∧ v.Once ∧ v.site.Checked D A c

end View

def Chained (D : Digests) (c : Composite) (log : Registration) : Prop :=
  ∀ i (h : i < log.length), log[i].previous = head D c (log.take i)

inductive Accountant
  | pouw
  | pous
  | bandwidth
  | certifier
  deriving DecidableEq, Repr

def Accountant.part : Accountant → Site.Accountable
  | .pouw => .resource .compute
  | .pous => .resource .capacity
  | .bandwidth => .resource .bandwidth
  | .certifier => .egress

/-- `none` where the protocol's account names no witness, so the answer counts against the whole site. No answer names
a device. -/
abbrev Answer := Accountant × Option String

def Credits (S : Site) (a : Answer) (x : String) (r : Site.Accountable) : Prop :=
  a.1.part = r ∧ match a.2 with
    | none => True
    | some k => ∃ y ∈ S.holder k, S.Reaches y x

structure Lists where
  protocols : Composite → Finset Accountant

open Classical in
/-- In bytes; none when `x` names no registered sheet, which `View.Devices` refuses. -/
noncomputable def View.unaccounted (v : View) (D : Digests) (L : Lists) (as : List Answer) (k : Composite) (x : Loc) :
    ℕ :=
  match v.site.sheet? D.sheet x with
  | none => 0
  | some s => s.mem - ((v.site.locs.filter fun y => y.kind = .memory ∧ y.parent = some x.id ∧
      ∃ a ∈ as, a.1 ∈ L.protocols k ∧ Credits v.site a y.id (.resource .capacity)).map (·.bytes)).sum

/-- The account's check. That every part sits where an auditor's device can witness it is registration's
(`View.Onsite`), and that the list has a protocol for each resource is the composite's. -/
def AccountedFor (D : Digests) (L : Lists) (as : List Answer) (v : View) (k : Composite)
    (charged : List (String × Site.Accountable)) (unaccounted : List (String × ℕ)) : Prop :=
  charged.Nodup ∧
    (∀ x r, (x, r) ∈ v.site.parts → (∃ a ∈ as, a.1 ∈ L.protocols k ∧ Credits v.site a x r) ∨ (x, r) ∈ charged) ∧
    ∀ x ∈ v.site.locs, x.kind = .accelerator → 0 < v.unaccounted D L as k x → (x.id, v.unaccounted D L as k x) ∈ unaccounted

end Registration

end Verity
