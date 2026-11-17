module

public import NetworkCheck.Registered

/-!
# The network attester's link, read off the registered hardware: it states its rate

`NetworkCheck.Registered.registered S head link key` reads one link of a Site, the one named `link`, which a check of the
network attester's certificates is handed (`NetworkCheck.Span`'s parameters). It runs over that link alone and no other
link of `S`. `RegisteredLinkRated` states what its acceptance says of that link's rate; `RegisteredLinkRatedHonest` and
`RegisteredLinkRatedNontrivial` are its witnesses.

What these do not say: anything of another link of `S`, which may state no rate. The registration's check on a Site
(`Core.Site.Entry.Kinds`) accepts a link with no rate wherever the link carries `unmediated`, a network attester's own among
them, so a registered Site is not rated on every link. Nor that the rate is positive, which `registered` does not hold
for a Site the registration did not accept, nor that it is the link's true line rate.
-/

@[expose] public section

namespace Properties.Certifier.Registered

open NetworkCheck (Registered)

/-- **The named link states a rate.** For every Site `S`, registered or not, and every `head`, `link` and `key`: where
`registered` reads `link` off `S`, `S` has a link under that id (the first, where `S` gives an id twice), it joins a
network attester and a host, and it states a rate (`β` some), the one `registered` returns. So a link with no rate is
never read as one: `registered` refuses it by name. This is of the named link only; `S`'s other links, a network attester's
other half to the outside and the links of another certifier among them, may state none. -/
def RegisteredLinkRated : Prop :=
  ∀ (S : Core.Site) (head link key : String) (r : Registered),
    Registered.registered S head link key = .ok r →
      ∃ k ∈ S.links, k.id = link ∧ S.Joins k .certifier .host ∧ k.β = some r.β

/-- `RegisteredLinkRated`'s honest witness, unproved: `registered` reads some link off some Site. -/
def RegisteredLinkRatedHonest : Prop :=
  ∃ (S : Core.Site) (head link key : String) (r : Registered), Registered.registered S head link key = .ok r

/-- `S` with every link named `link` stating no rate and a reason no certifier splits it: a change the registration's
check on a Site accepts (`Core.Site.Entry.Kinds`). -/
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
