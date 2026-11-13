module

public import Definitions.Core.Site
public import Definitions.Accounting.Joint

/-!
# What registration establishes, and what the protocols account for

Registration is checked before any challenge: the developer's log (`Verity.Registration`, `Definitions.Core.Site`) is
valid when each entry passes its checks against the log before it (`Registration.Valid`). This file states what the
theorems read of an accepted log and what its acceptance gives them, and the condition that every part of the registered
hardware is accounted for by some protocol of the composite. It defines no second log, Site, sheet, builder or θ: each
is the one `Definitions.Core.Site` and `Definitions.Core.Compiler` define.

* **The view** (`Registration.View`, `Registration.view`): what the theorems read of a log at its head, the Site, θ's
  values by their committed roots, the registered compiler `B` by its builder, each day's circuit by its program's
  digest, and the heads of the log's chain (`Registration.heads`). It is the spec's copy of the code's view
  (`Core.Registration.View`), which a protocol reads of an accepted log.
* **Prefix heads** (`Registration.heads`, `Registration.AtHead`): an answer names a head of the accepted log's chain,
  the head of one of its prefixes, and is read against what that prefix registers. No entry retires what an earlier
  one registers, so what a prefix registers a longer prefix registers too.
* **The readings** (`Registration.Reading`, `View.figures`): a registered device's figures as its sheet states them, or
  as the bound reads them, figure by figure the larger of those and the auditor's estimate (`DeviceFigures.bounding`),
  the reading `Site.device` builds the device model from.
* **The registered facts** (`View.Holds`): every registered device names a sheet registration accepted against the
  auditor's data sheet (`View.Devices`); onsite, every boundary link is certified and every memory leaf's host is
  reached by a memory challenger (`View.Onsite`); and θ registers each value once and the log declares each day once
  (`View.Once`), so a lookup by name or by day reads the one registered value or circuit.
* **The guarantee** (`Verity.Guarantees.RegistrationSound`, in `Definitions.Accounting.Registration.Guarantees`): a
  valid log's view holds the registered facts.
* **The parts of the registered hardware** (`Site.Accountable`, `Site.parts`): each device's compute, each memory leaf's
  memory held and memory read, the device model's resources (`Accounting.Resource`) at the leaf's tier, and each
  boundary link's egress, by the id that holds it.
* **What each protocol accounts for** (`Registration.Lists`): the composite's protocols, and for each the parts it
  accounts for at each id. Its values come from each protocol's account in code (`Core.Account`: each amount's `part`
  and `place`, whose leaves `Core.Account.Part.leaves` names). `Registration.AccountedFor` holds when every part of the
  registered hardware is accounted for by some protocol of the composite.

**Not covered.** `RegistrationSound` reads the spec's check (`Registration.Valid`); that the code's check decides it is
a property of the code (`Properties.Registration.Checks`). Nothing here ties a builder's digest to the compiler it
holds: the spec's builder holds its compiler (`Registration.Builder.B`), while the code's holds a digest and a cap.
What another protocol names of the registration (the circuit its parameters name, a network certifier's links and key)
joins two protocols, so it is stated in the composition's area (`Properties.Ties`), and nothing proves it.
`Registration.Valid` doesn't check `AccountedFor`, so no guarantee here concludes it. Hardware outside the Site
(unregistered hardware) and a host's own compute, which the Site registers no rate for, are parts of no Site.
-/

@[expose] public section

namespace Verity

namespace Site

/-- **What a part of the registered hardware is of**, at one registered id: one of the device model's resources
(`Accounting.Resource`), a device's compute or a memory leaf's bits read (`bandwidth`) or memory held (`capacity`) at
the leaf's tier, or a boundary link's egress. Egress is no resource of the device model: a resource there is what a
joint audit certifies by an audit of its own, and a joint run (`Accounting.Run`) carries an account, a window and a
verdict for each. The code's `Core.Account.Part` is the same parts, its memory and read parts by tier. -/
inductive Accountable
  | resource (r : Accounting.Resource)
  | egress

/-- **The parts the Site holds to account for**, by the id of the location or link that holds each: each device's
compute, each memory leaf's memory held and memory read, and each boundary link's egress (`Site.Boundary`). The site, a
host and the auditor's devices hold none of their own: a host has no compute rate, and its memory is its memory leaves'.
-/
def parts (S : Site) : Set (String × Accountable) :=
  {xr | (∃ x ∈ S.locs, x.kind = .device ∧ xr = (x.id, .resource .compute)) ∨
    (∃ x ∈ S.locs, x.kind = .memory ∧ (xr = (x.id, .resource .capacity) ∨ xr = (x.id, .resource .bandwidth))) ∨
    ∃ k ∈ S.links, S.Boundary k ∧ xr = (k.id, .egress)}

end Site

namespace Registration

open Site (Sheet Loc)

/-- **The heads of the log's chain**: the head (`head`) of each prefix of the log with the prefix's length, shortest
first: the empty prefix's, the composite's genesis, then one for each entry. -/
def heads (D : Digests) (c : Composite) (log : Registration) : List (ℕ × String) :=
  (List.range (log.length + 1)).map fun n => (n, head D c (log.take n))

/-- **What the theorems read of a registration log**, at its head: the Site, θ's registered values in order, the
registered compiler's builder, the days' declared circuits in order, its head, and the heads of its chain. The code's
`Core.Registration.View` is its copy. -/
structure View where
  site : Site
  theta : List Theta
  builder : Option Builder
  declarations : List Declared
  head : String
  heads : List (ℕ × String)

/-- **A log's view**: its Site (`Site.ofLog`), its θ (`Registration.theta`), its builder (`Registration.builder?`), its
declarations in the log's order, its head (`Registration.head`) and the heads of its chain (`Registration.heads`). -/
def view (D : Digests) (c : Composite) (log : Registration) : View where
  site := Site.ofLog log
  theta := log.theta
  builder := log.builder?
  declarations := log.filterMap Entry.declared?
  head := head D c log
  heads := heads D c log

/-- **At a head of the log's chain**: `h` is the head of a prefix of the log (`View.heads`), and `P` holds of that
prefix's view, so what `P` reads as registered is what the prefix registers. An answer that names the head `h` is read
so. -/
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

/-- **Device `x`'s figures at reading `r`**, for the sheet digest `digest` and the auditor `A`: its registered sheet's
figures, or those bounded by the auditor's estimate for the sheet's SKU; `none` when `x` names no registered sheet or
the auditor has no sheet for its SKU. `Site.device` reads `bound`. -/
def figures (digest : Sheet → String) (A : Auditor) (r : Reading) (x : Loc) : Option DeviceFigures :=
  (v.site.sheet? digest x).bind fun k => (A.sheet k.sku).map fun a =>
    match r with
    | .stated => k.figures
    | .bound => k.figures.bounding (A.estimate a)

/-- **Every registered device has accepted figures**: each device names a registered sheet (under the digests `D`) whose
SKU the auditor `A` has a data sheet for, whose figures registration accepted against the auditor's estimate
(`DeviceFigures.Accepted`) and whose memory figures are the auditor's. So both readings of its figures exist, and its
device model (`Site.device`) is defined. -/
def Devices (D : Digests) (A : Auditor) : Prop :=
  ∀ x ∈ v.site.locs, x.kind = .device →
    ∃ k ∈ v.site.sheet? D.sheet x, ∃ a ∈ A.sheet k.sku,
      k.figures.Accepted (A.estimate a) ∧ k.bw = a.bw ∧ k.chip = a.chip ∧ k.mem = a.mem

/-- **Onsite, the auditor's devices see the boundary and the memory**: for the composite `c` onsite, every boundary link
is certified, and the host of every memory leaf is at an end of a memory challenger's link. -/
def Onsite (c : Composite) : Prop :=
  c = .onsite → (∀ k ∈ v.site.links, v.site.Boundary k → v.site.Certified k) ∧
    ∀ x ∈ v.site.locs, x.kind = .memory → ∀ h ∈ v.site.host? x, h ∈ v.site.reached

/-- **Each value and each day once**: θ registers no value's name twice, and no day is declared twice, so a lookup by
name or by day reads the one registered value or circuit. -/
def Once : Prop := (v.theta.map (·.value)).Nodup ∧ (v.declarations.map (·.day)).Nodup

/-- **The registered facts the theorems read** hold of the view, for the digests `D`, the auditor `A` and the composite
`c`: its devices' figures (`Devices`), onsite coverage (`Onsite`), and each value and day once (`Once`). -/
def Holds (D : Digests) (A : Auditor) (c : Composite) : Prop := v.Devices D A ∧ v.Onsite c ∧ v.Once

end View

/-- **The composite's lists, and what each protocol on them accounts for**: the protocols `P` of each composite, and
the parts of the registered hardware protocol `p` of composite `k` accounts for at id `x` of Site `S`
(`accounts k p S x`). A deployment's values come from each protocol's account in code (`Core.Account`): `p` accounts for
part `r` at `x` when one of its amounts' parts is of `r`'s kind and its leaves at `S` (`Core.Account.Part.leaves`)
hold `x`. -/
structure Lists (P : Type) where
  protocols : Composite → Finset P
  accounts : Composite → P → Site → String → Set Site.Accountable

/-- **The registered hardware is accounted for** by composite `k`'s protocols: every part the view's Site holds
(`Site.parts`) is accounted for by some protocol on `k`'s list. It is about the list, before any challenge: whether a
protocol's answer then certifies the part is the accounting's, which charges at capacity a part no accepted answer
covers. -/
def AccountedFor {P : Type} (L : Lists P) (v : View) (k : Composite) : Prop :=
  ∀ x r, (x, r) ∈ v.site.parts → ∃ p ∈ L.protocols k, r ∈ L.accounts k p v.site x

end Registration

end Verity
