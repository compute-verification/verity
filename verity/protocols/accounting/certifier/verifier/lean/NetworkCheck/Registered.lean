module

public import CoreProtocol.Site
meta import CoreProtocol.Site

/-!
# The network certifier's link and key, read from the registered hardware

A check of the network certifier's certificates is handed a link's name and the certifier's key. `registered` reads both
off the registered hardware (`Core.Site`, the site of an accepted registration log's view): the link is one that joins
a network certifier and a host, the half of an uplink the certifier splits that is the host's boundary
(`Core.Site.Boundary`, `Core.Site.Certified`), and the key is the one that certifier registered. It returns what the
registration states of them (`Registered`): the link's id, its host end, the certifier's id and registered key, the
link's capacity `β` in bits a tick each way, and the head of the log the site was read at, which it carries and doesn't
read.

It refuses, by name:

* a link that is no link of the site;
* a link that doesn't join a network certifier and a host: the certifier's other half, to the outside, a host's bare
  uplink, a link between two hosts, a memory challenger's link;
* a certifier whose registered key is not lower-case hex, and a key that is not the certifier's registered key
  (`keyRefusal`).

**The registered key.** The registered key of a network certifier (`Core.Site.Loc.key`) is the string the checks' `key`
parameter is: its LMS public key, I ‖ T[1], in lower-case hex. `keyRefusal` is the one function that reads it.

`siteOf` reads the site as the log's view writes it (`Core.Site.toJson`: `{"links", "locs", "sheets"}`), each location
and link by the Site's own readers. The sheets are not read, since the reading reads no figure, so the Site it returns
has none. Nothing here holds the site to an accepted log: its caller does.

`registered.py` is the Python mirror, held to this on the same vectors by `tests/test_network_certifier_registered.py`.
-/

@[expose] public section

namespace NetworkCheck

open Lean (Json)
open Flock (V)

/-- What the registered hardware states of a network certifier's link: the link's id, its host end, the certifier's id
and registered key, the link's capacity `β` in bits a tick each way, and the head of the log it was read at. -/
structure Registered where
  head : String
  link : String
  host : String
  certifier : String
  key : String
  β : Nat
  deriving DecidableEq, Repr

namespace Registered

/-- The site of a registration log's view, as `Core.Site.toJson` writes it: its locations and links. Its sheets are not
read, and the Site has none. -/
def siteOf (j : Json) : V Core.Site := do
  Core.Site.exactly j ["links", "locs", "sheets"] "the site"
  let _ ← Flock.J.arr j "sheets" "the site"
  let locs ← (← Flock.J.arr j "locs" "the site").toList.mapM Core.Site.Loc.ofJson
  let links ← (← Flock.J.arr j "links" "the site").toList.mapM Core.Site.Link.ofJson
  pure ⟨[], locs, links⟩

/-- The network certifiers' links of `S`, by id, in the Site's order: those that join a network certifier and a host. -/
def certifierLinks (S : Core.Site) : List String := (S.links.filter fun k => S.Joins k .certifier .host).map (·.id)

/-- What a refusal of a link says a right one is: the network certifiers' links of `S`. -/
def rightLinks (S : Core.Site) : String :=
  s!"the network certifiers' links there are [{", ".intercalate (certifierLinks S)}]"

/-- A link's end as a refusal names it: its kind and id, the outside, or an id nothing registers. -/
def endName (S : Core.Site) (i : String) : String :=
  if i = Core.Site.outside then "the outside" else
    match S.kind? i with
    | some k => s!"{k.name} {i}"
    | none => s!"{i}, which is not registered"

/-- `s` is lower-case hex: an even, positive number of the digits `0`–`9` and `a`–`f`. -/
def lowerHex (s : String) : Bool :=
  !s.isEmpty && s.length % 2 == 0 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

/-- Why `key` is not the key of the network certifier `c`, or `none` when it is: `c`'s registered key is its LMS public
key in lower-case hex, and `key` is that string. The one reading of a certifier's registered key. -/
def keyRefusal (c : Core.Site.Loc) (key : String) : Option String :=
  if !lowerHex c.key then
    some s!"network certifier {c.id} registers the key {c.key}, which is not lower-case hex; a network certifier \
      registers its LMS public key in lower-case hex"
  else if c.key != key then
    some s!"the key {key} is not network certifier {c.id}'s registered key, {c.key}"
  else none

/-- **The network certifier's link and key, as registered.** `link` names a link of `S` that joins a network certifier
and a host, and `key` is that certifier's registered key (`keyRefusal`): the link's id, its host end, the certifier's id
and key and the link's `β`, with `head`, the head of the log `S` was read at, carried unread. A link that is no link of
`S`, one that doesn't join a network certifier and a host, and a key that is not the certifier's are each refused,
naming the network certifiers' links of `S` or the certifier's registered key. -/
def registered (S : Core.Site) (head link key : String) : V Registered := do
  let some k := S.links.find? (·.id == link)
    | throw s!"link {link}: no link of the registered hardware; {rightLinks S}"
  unless S.Joins k .certifier .host do
    throw s!"link {link}: it joins {endName S k.ends.1} and {endName S k.ends.2}, not a network certifier and a host; \
      {rightLinks S}"
  let (c, host) := if S.kind? k.ends.1 = some .certifier then k.ends else (k.ends.2, k.ends.1)
  let some x := S.loc? c | throw s!"link {link}: network certifier {c} is not registered"
  if let some why := keyRefusal x key then throw s!"link {link}: {why}"
  pure { head, link := k.id, host, certifier := c, key := x.key, β := k.β }

/-! ## The registration's example

The locations and links of the registration's example (`Core.Registration.Example`), restated: this module reads the
Site and not the registration. -/

section Example

open Core.Site (Loc Link outside)

/-- A key in the registered form: 48 bytes in lower-case hex. -/
private def key0 : String := String.ofList (List.replicate 96 'a')

/-- Another network certifier's. -/
private def key1 : String := String.ofList (List.replicate 96 'b')

/-- The example's locations, its certifier registering `key`: a host with a device and its memory, behind a network
certifier, and a memory challenger. -/
private def exampleLocs (key : String) : List Loc :=
  [{ id := "site0", kind := .site }, { id := "node0", kind := .host, parent := some "site0" },
   { id := "gpu0", kind := .device, parent := some "node0", sheet := "sheet0" },
   { id := "hbm0", kind := .memory, parent := some "gpu0", tier := some .hbm, bytes := 1 <<< 30 },
   { id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key },
   { id := "chal0", kind := .challenger, parent := some "site0", owner := .auditor, key := "k1" }]

/-- The example's links: the host's uplink, which the certifier splits, and the memory challenger's link. -/
private def exampleLinks : List Link :=
  [{ id := "up0", ends := ("node0", "cert0"), β := 100, latency := 1 },
   { id := "up1", ends := ("cert0", outside), β := 100, latency := 1 },
   { id := "mc0", ends := ("chal0", "node0"), β := 10, latency := 1 }]

/-- The example's site with its certifier's key in the registered form, and beside it a second host behind a second
network certifier, a link between the two hosts and the second host's bare uplink. -/
private def site : Core.Site :=
  ⟨[],
   exampleLocs key0 ++
     [{ id := "node1", kind := .host, parent := some "site0" },
      { id := "cert1", kind := .certifier, parent := some "site0", owner := .auditor, key := key1 }],
   exampleLinks ++
     [{ id := "up2", ends := ("cert1", "node1"), β := 40, latency := 1 },
      { id := "up3", ends := ("cert1", outside), β := 40, latency := 1 },
      { id := "hh0", ends := ("node0", "node1"), β := 400, latency := 1 },
      { id := "up4", ends := ("node1", outside), β := 10, latency := 1 }]⟩

private def refusal (S : Core.Site) (link key : String) : Option String :=
  match registered S "h" link key with
  | .ok _ => none
  | .error e => some e

private def links : String := "the network certifiers' links there are [up0, up2]"

-- the certifier's link, its ends in either order
#guard (registered site "h" "up0" key0).toOption == some ⟨"h", "up0", "node0", "cert0", key0, 100⟩
#guard (registered site "h" "up2" key1).toOption == some ⟨"h", "up2", "node1", "cert1", key1, 40⟩
-- a link the site doesn't have
#guard refusal site "up9" key0 == some s!"link up9: no link of the registered hardware; {links}"
-- the certifier's half to the outside, a bare uplink, a link between two hosts, the memory challenger's link
#guard refusal site "up1" key0 ==
  some s!"link up1: it joins certifier cert0 and the outside, not a network certifier and a host; {links}"
#guard refusal site "up4" key0 ==
  some s!"link up4: it joins host node1 and the outside, not a network certifier and a host; {links}"
#guard refusal site "hh0" key0 ==
  some s!"link hh0: it joins host node0 and host node1, not a network certifier and a host; {links}"
#guard refusal site "mc0" key0 ==
  some s!"link mc0: it joins challenger chal0 and host node0, not a network certifier and a host; {links}"
-- a link to an end nothing registers
#guard refusal { site with links := [{ id := "up0", ends := ("cert0", "node9"), β := 1, latency := 1 }] } "up0" key0 ==
  some "link up0: it joins certifier cert0 and node9, which is not registered, not a network certifier and a host; \
    the network certifiers' links there are []"
-- the outside is the outside, whatever is registered under its name
#guard refusal { site with locs := site.locs ++ [{ id := outside, kind := .host }],
                           links := { id := "up5", ends := ("cert0", outside), β := 9, latency := 1 } :: site.links }
    "up5" key0 ==
  some s!"link up5: it joins certifier cert0 and the outside, not a network certifier and a host; {links}"
-- a name the site gives twice: the first link and the first location of it
#guard (registered { site with locs := site.locs ++ [{ id := "cert0", kind := .certifier, key := key1 }],
                               links := site.links ++ [{ id := "up0", ends := ("node1", "cert1"), β := 7,
                                                         latency := 1 }] }
    "h" "up0" key0).toOption == some ⟨"h", "up0", "node0", "cert0", key0, 100⟩
-- another certifier's key, and a key no certifier registers
#guard refusal site "up0" key1 ==
  some s!"link up0: the key {key1} is not network certifier cert0's registered key, {key0}"
#guard refusal site "up2" "00" == some s!"link up2: the key 00 is not network certifier cert1's registered key, {key1}"
-- a registered key that is not lower-case hex: the example's own, an upper-case one, an odd one, none
#guard refusal ⟨[], exampleLocs "k0", exampleLinks⟩ "up0" "k0" ==
  some "link up0: network certifier cert0 registers the key k0, which is not lower-case hex; a network certifier \
    registers its LMS public key in lower-case hex"
#guard ["AB", "abc", "", "0g"].all fun bad =>
  (refusal ⟨[], exampleLocs bad, exampleLinks⟩ "up0" bad).isSome
#guard lowerHex "00ff19ae" && !lowerHex "00FF" && !lowerHex "0" && !lowerHex "" && !lowerHex "0x"
-- the site as the log's view writes it: its locations and links, no sheet
#guard (siteOf site.toJson).toOption == some site
#guard (siteOf (Json.mkObj [("links", .arr #[]), ("locs", .arr #[])])).toOption.isNone
#guard (siteOf (Json.mkObj [("links", .arr #[]), ("locs", .arr #[]), ("sheets", .null)])).toOption.isNone

end Example

end Registered

end NetworkCheck
