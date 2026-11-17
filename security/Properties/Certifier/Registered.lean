module

public import NetworkCheck.Registered

/-! The network attester's link, read off the registered hardware, states its rate; nothing here is said of `S`'s
other links, which may state none, nor that the rate is positive or the link's true line rate. -/

@[expose] public section

namespace Properties.Certifier.Registered

open NetworkCheck (Registered)

/-- This is of the named link only; `S`'s other links, a certifier's other half to the outside and the links of
another certifier among them, may state none. -/
def RegisteredLinkRated : Prop :=
  ∀ (S : Core.Site) (head link key : String) (r : Registered),
    Registered.registered S head link key = .ok r →
      ∃ k ∈ S.links, k.id = link ∧ S.Joins k .certifier .host ∧ k.β = some r.β

/-- `RegisteredLinkRated`'s honest witness, unproved: `registered` reads some link off some Site. -/
def RegisteredLinkRatedHonest : Prop :=
  ∃ (S : Core.Site) (head link key : String) (r : Registered), Registered.registered S head link key = .ok r

def unrated (S : Core.Site) (link : String) : Core.Site :=
  { S with links := S.links.map fun k => if k.id = link then { k with β := none, unmediated := some "no line rate" }
      else k }

/-- `RegisteredLinkRated`'s nontrivial witness, unproved: a Site off which `registered` reads `link`, whose same link
stating no rate, with a reason no certifier splits it, is refused. -/
def RegisteredLinkRatedNontrivial : Prop :=
  ∃ (S : Core.Site) (head link key : String) (r : Registered),
    Registered.registered S head link key = .ok r ∧
      ∀ r', Registered.registered (unrated S link) head link key ≠ .ok r'

end Properties.Certifier.Registered
