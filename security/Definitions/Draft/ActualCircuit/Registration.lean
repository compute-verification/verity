module

public import Definitions.Registration.Guarantees
public import Definitions.ResourceAccounting.Guarantees.RegisteredSite
public import Definitions.Draft.ActualCircuit.Epoch

/-!
# Registration's statement (draft)

Registration runs once, before the first epoch, and draws no challenge: its verifier accepts when the registration is
valid (`Verity.Registration.Valid`), a decision on registered information alone. Three statements:

* `RegistrationBinds`: what it establishes about the transcript, what the developer is bound to before the first
  epoch: no later entry changes the Site's sheets, locations and links, θ, the registered compiler or a day's declared
  circuit as registered. What is missing is the registered facts themselves (`Verity.Guarantees.RegistrationSound`),
  which the lock records as failing as stated at the present Site.
* `RegistrationSet`: the statement in the shape every protocol's has. Its set is all actual circuits: registration by
  itself says nothing about what the hardware computed.
* `RegistrationEgress`: the one property of the actual circuit that follows from it, under the physical premise that
  every wire runs on a registered link within its rate: `ResourceAccounting.Guarantees.EgressRegistered` at the Site
  registration folds (`Verity.Site.ofLog`). Its error is `0`, and its content is the premise's.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting Verity

/-- **What registration binds the developer to before the first epoch.** Whatever is appended after a registration,
the Site's sheets, locations and links, θ, the registered compiler and each day's declared circuit, as registered, are
unchanged and come first: the auditor holds the registration, and every later entry is appended to it. Missing: that a
valid registration's view holds the registered facts (`Verity.Guarantees.RegistrationSound`), which the lock records as
failing as stated at the present Site, so it is not in this conjunction. Not vacuous because: I cannot name one; it
holds of every registration by the definitions of `Site.ofLog` and the registration's readers, and its content is that
those readers take the first entry. -/
def RegistrationBinds : Prop :=
  ∀ (reg more : Registration),
    (Site.ofLog reg).sheets <+: (Site.ofLog (reg ++ more)).sheets ∧
    (Site.ofLog reg).locs <+: (Site.ofLog (reg ++ more)).locs ∧
    (Site.ofLog reg).links <+: (Site.ofLog (reg ++ more)).links ∧
    Registration.theta reg <+: Registration.theta (reg ++ more) ∧
    (∀ b ∈ Registration.builder? reg, Registration.builder? (reg ++ more) = some b) ∧
    ∀ j, ∀ d ∈ Registration.declared reg j, Registration.declared (reg ++ more) j = some d

/-- **Registration's statement, in the shape every protocol's has.** If the verifier accepts (the registration is
valid) then, except with probability `0` over any challenges, the actual circuit lies in the set of all actual
circuits. The set is all circuits: registration fixes registered information and the hardware description, and by
itself constrains nothing the hardware computed. Not vacuous because: I cannot name one; its conclusion holds of every
strategy and verifier, which is what it records. -/
def RegistrationSet : Prop :=
  ∀ {L : Type*} [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (D : Registration.Digests) (A : Registration.Auditor) (c : Registration.Composite) (reg : Registration),
    Sound Ω 𝓕 V (fun _ => Registration.Valid D A c reg) (Lies V fun _ => Set.univ) 0

/-- **Every wire runs on a registered link within its rate**, in every feasible run (the physical premise of
`ResourceAccounting.Guarantees.EgressRegistered`, `Placed.Links`, at the Site registration folds): for some assignment of
wires to links, the actual circuit, each device and memory leaf taken to its host, meets `Placed.Links` at the
registered links' ends and the rates `β`. New: a premise on the feasible set. -/
def RunsOnRegisteredLinks (reg : Registration) (β : Fin (Site.ofLog reg).links.length → ℕ)
    (𝓕 : Set (Actual.Run (Site.ofLog reg).Place)) : Prop :=
  ∀ R ∈ 𝓕, ∃ link : Fin R.A.N → Fin R.A.N → Fin (Site.ofLog reg).links.length,
    (R.A.onto (Site.ofLog reg).linkPlace).Links (Site.ofLog reg).placeEnds β link

/-- **The egress set** of the subtree at `i` of the registered Site: the actual circuits whose egress from the subtree
over every `[a, b)` is at most the registered capacity of the links with exactly one end in it times `b − a`. -/
def egressSet (reg : Registration) (i : String) (β : Fin (Site.ofLog reg).links.length → ℕ) :
    Set (Placed (Site.ofLog reg).Place) :=
  {A | ∀ a b : ℕ, A.egress ((Site.ofLog reg).placesIn ((Site.ofLog reg).under i)) (Set.Ico a b) ≤
    linkCapacity (Site.ofLog reg).placeEnds β ((Site.ofLog reg).placesIn ((Site.ofLog reg).under i)) * (b - a)}

/-- **The property of the actual circuit that follows from registration.** If the verifier accepts (the registration
is valid) then, except with probability `0`, the actual circuit lies in the egress set of the subtree at `i`
(`egressSet`), given that each link across the subtree's boundary is registered with a rate (`Site.RatedOn`), that the
subtree holds each host whole (`Site.HostsWhole`), and that every wire of every feasible run runs on a registered link
within its rate (`RunsOnRegisteredLinks`). The acceptance does no work: `EgressRegistered` takes no validity
hypothesis, so this holds at any acceptance, the registered Site being only where the premises are read. Not vacuous
because: I cannot name a strategy and verifier at which its premises hold and its conclusion fails, since it is
`EgressRegistered`, a theorem; its premises hold at a Site with one host and one rated uplink, and its content is
`RunsOnRegisteredLinks`, without which the conclusion fails for a feasible set holding a run that sends the outside a
word wider than the uplink's rate in one tick. -/
def RegistrationEgress : Prop :=
  ∀ (D : Registration.Digests) (A : Registration.Auditor) (c : Registration.Composite) (reg : Registration)
    (i : String) (β : Fin (Site.ofLog reg).links.length → ℕ) (Ω : Type) [Fintype Ω]
    (𝓕 : Set (Actual.Run (Site.ofLog reg).Place)) (V : Finset (Site.ofLog reg).Place),
    (Site.ofLog reg).RatedOn β
        ↑(boundary (Site.ofLog reg).placeEnds ((Site.ofLog reg).placesIn ((Site.ofLog reg).under i))) →
      (Site.ofLog reg).HostsWhole i → RunsOnRegisteredLinks reg β 𝓕 →
      Sound Ω 𝓕 V (fun _ => Registration.Valid D A c reg) (Lies V fun _ => egressSet reg i β) 0

end Draft.ActualCircuit
