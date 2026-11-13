module

public import NetworkCheck.Period
public import CoreProtocol.Account
meta import NetworkCheck.Period
meta import CoreProtocol.Account

/-!
# What the network certifier's period check accounts for

`account` is the period check's `Core.Account` (`verity/certifier/v0`, `NetworkCheck.Period.protocol`): one amount, the
period's leak `K`, about the egress of the one link the period's parameters name, and no other part of the registered
hardware. `K` bounds the period's leak from above (the sum of `PeriodLeakAccepted`'s terms: the free inputs, the
incorrect proof blocks a sample in part leaves possible, the jitter, the schedules and the cells), so it is the link's
free amount, and the rest of the link's capacity over the span is verified. What a sample in part leaves unproved is
inside `K` (its `faulty_sets`, `faulty_bits` and `faulty_entries` terms), so the amount has no share; its error is
`eps`, the union of its audits' δ, which adds over a period's answers.

The account reads eight keys of an accepted answer. The check publishes three of them: `K`, `eps` and the profile's
`link`. It publishes none of the other five, so no answer it writes counts in `Core.Account.rows` until it does (the
guards below hold it to that):

* `registration_head`: the head of the registration the period's link and key were read at. The parameters hold none.
* `start_ms` and `span_ms`: the period's span in milliseconds from its certifier session's start. The answer publishes
  only `windows`, the count of its link-windows; the first one's number and a link-window's length are parameters (the
  length is the schedule's in the rows form, and no parameter in the clock form). A step of an account is of a fixed
  length, so the span is named in a fixed unit and not in link-windows.
* `session`: the certifier session the span is numbered in. Two sessions' spans cannot be shown disjoint.
* `rests`: the names (`Core.Account.nameOf`) of the answers the check reads as given and does not verify, each
  sampled proofs' one-stage (`oneStage`): each link-window's, and each compute audit's.

Nothing here ties the link an answer names to a network certifier of the registered hardware, or the key its
certificates verify under to that certifier's registered key (`NetworkCheck.Registered.registered` states both, and
the period check does not call it): `Core.Account.rows` finds the link by its name among the site's boundary links, so
a period under any key that names a host's bare uplink reads as that uplink's (the last guard).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace NetworkCheck.Period

open Lean (Json toJson)

/-- **The period check's account**: the period's leak `K`, in bits, the free amount of the egress of the link its
profile names, over the span the answer names in milliseconds of its certifier session, at the error `eps`, summed
over a period's answers. The registered capacity is the link's `β`, read from the registration and from no answer. A
period the check does not accept publishes nothing. `eps` names and does not price each audit's knowledge error and
collision errors. An answer rests on the one-stage answers its check read as given. -/
def account : Core.Account :=
  { protocol := protocolId
    head := some (.published "registration_head")
    amounts := [{ metric := "leak", part := .egress, place := some (.profile "link"), key := .published "K",
                  unit := Core.Account.bits, side := .free,
                  time := .steps (.run (.published "start_ms") (.published "span_ms")) ⟨1, 1000⟩,
                  series := some (.published "session"), error := .published "eps", pool := .sum,
                  uncertified := .refused,
                  unpriced := ["each audit's knowledge error", "each audit's collision errors"] }]
    rests := some ⟨.published "rests", [oneStage]⟩ }

#guard account.Formed
#guard account.protocol == protocol.id

namespace AccountExample

open Core.Registration.Example (sheet locs auditor fig)
open Core.Site (Loc Link Figure Sheet outside)

/-- The registration's example site with links of its own: host `node0` behind network certifier `cert0`, a second
host `node1` behind a second certifier `cert1`, and `node1`'s bare uplink, which no certifier splits. -/
private def site : Core.Site :=
  ⟨[sheet],
   locs ++ [{ id := "node1", kind := .host, parent := some "site0" },
            { id := "cert1", kind := .certifier, parent := some "site0", owner := .auditor, key := "k2" }],
   [{ id := "up0", ends := ("node0", "cert0"), β := 1000000, latency := 1 },
    { id := "up1", ends := ("cert0", outside), β := 1000000, latency := 1 },
    { id := "mc0", ends := ("chal0", "node0"), β := 10, latency := 1 },
    { id := "up2", ends := ("cert1", "node1"), β := 400000, latency := 1 },
    { id := "up3", ends := ("cert1", outside), β := 400000, latency := 1 },
    { id := "up4", ends := ("node1", outside), β := 10000, latency := 1 }]⟩

private def digest : Sheet → String := Core.Site.Registration.sha512.sheet

/-- A tick of one second. -/
private def tick : Figure := fig 1

/-- A period's public output in the rows form: four link-windows, each scan sampled whole at δ = 1/100, its tied
audit's free inputs 9,088 bits and the slot entries its incorrect proof blocks can set 896 bits. -/
private def published : Public :=
  rowsTerms #[{ units := 16, drawn := 9, delta := ⟨1, 100⟩, wrong := 2 }] ⟨32, 0, 300, 512, 8 * 112⟩
    (Array.replicate 4 (({ units := 64, drawn := 64, delta := ⟨1, 100⟩, wrong := 0 } : Facts), 9088, 8 * 112))
    #[] 1 4

/-- The period's accepted answer on `link`, its public output and profile as the check writes them. -/
private def period (link : String) : Core.Answer :=
  { protocol := protocolId, verdict := .accept, reason := none, digest := none,
    published := some (publicJson published), profile := some (profileJson ⟨link, 4⟩) }

private def today : Core.Answer := period "up0"

/-- An accepted answer of sampled proofs' one-stage. -/
private def audit (n : Nat) : Core.Answer :=
  { protocol := oneStage, verdict := .accept, reason := none, digest := none,
    published := some (Json.mkObj [("audit", toJson n)]), profile := some (Json.mkObj []) }

/-- Answer `a` with the keys the account reads and the check does not publish: the registration's head, the span
(`start` and `span`, in milliseconds of `session`) and the names of the answers `rests` it rests on. -/
private def named (a : Core.Answer) (start span : Nat) (rests : List Core.Answer) (session : String := "s0")
    (head : String := "h") : Core.Answer :=
  { a with published := a.published.map fun j => j.mergeObj (Json.mkObj [
      ("registration_head", .str head),
      ("rests", Json.arr (rests.filterMap fun b => (Core.Account.nameOf b).map Json.str).toArray),
      ("session", .str session), ("span_ms", toJson span), ("start_ms", toJson start)]) }

private def publishes (a : Core.Answer) (keys : List Core.Account.Key) : List Bool :=
  keys.map fun k => (k.read a).isSome

/-- What the account's amount reads of answer `a`: the place, the amount, the error, the ticks covered and the steps
named. -/
private def reading (a : Core.Answer) : List (Option (Option String × Nat × String × Nat × Option (Nat × Nat))) :=
  account.amounts.map fun m => (m.read tick a).map fun r => (r.place, r.amount, r.error.toString, r.ticks, r.steps)

/-- The egress rows of the site registered at head `h`, over a period of 600 ticks, from `answers` read by `account`:
each row's metric, place, standing, capacity, verified and free amounts, and error. -/
private def egress (answers : List Core.Answer) :
    List (String × String × String × Option Nat × Option Nat × Option Nat × String) :=
  ((Core.Account.rows site digest auditor "h" tick 600 [account] answers).filter fun r => r.part = .egress).map
    fun r => (r.metric, r.place, r.status.name, r.capacity, r.verified, r.free, r.error.toString)

private def charged (place : String) (β : Nat) :
    String × String × String × Option Nat × Option Nat × Option Nat × String :=
  ("not audited", place, "not audited: charged at capacity", some (β * 600), some 0, some (β * 600), "0")

#guard published.K == 4 * (9088 + 896) ∧ published.eps.toString == "1/25"
-- what the check publishes: the amount, its error and its link
#guard publishes today [.published "K", .published "eps", .profile "link"] == [true, true, true]
-- what it does not: the registration's head, the span, the session, and the answers it rests on
#guard publishes today [.published "registration_head", .published "start_ms", .published "span_ms",
    .published "session", .published "rests"] == [false, false, false, false, false]
-- so the amount reads its link, its bits and its error, and no span: it covers no tick
#guard reading today == [some (some "up0", 39936, "1/25", 0, none)]
-- and the answer counts for nothing: every boundary link is charged at capacity, as with no answer
#guard egress [] == [charged "up0, up2, up4" 1410000]
#guard egress [today, audit 0] == egress []
-- with the five keys, four link-windows of 60 s from the session's 120th second: the link's row, its capacity the
-- registration's β over the period's 600 ticks, its free amount K and the capacity of the 360 ticks not covered
#guard reading (named today 120000 240000 [audit 0]) == [some (some "up0", 39936, "1/25", 240, some (120000, 360000))]
#guard egress [named today 120000 240000 [audit 0], audit 0] ==
  [("leak", "up0", "certified", some 600000000, some 239960064, some 360039936, "1/25"), charged "up2, up4" 410000]
-- it rests on the one-stage answers it names: without one of them, beside another, or with no list, it counts for
-- nothing
#guard egress [named today 120000 240000 [audit 0]] == egress []
#guard egress [named today 120000 240000 [audit 0], audit 1] == egress []
#guard egress [named today 120000 240000 [audit 0], { audit 0 with verdict := .reject }] == egress []
-- as does an answer that names another registration's head
#guard egress [named today 120000 240000 [audit 0] "s0" "g", audit 0] == egress []
-- two periods of one session over disjoint spans: their bits, their errors and their cover add
#guard egress [named today 120000 240000 [audit 0], named today 360000 240000 [audit 0], audit 0] ==
  [("leak", "up0", "certified", some 600000000, some 479920128, some 120079872, "2/25"), charged "up2, up4" 410000]
-- two periods of two sessions: their bits and their errors add, and only one session's cover counts
#guard egress [named today 120000 240000 [audit 0], named today 360000 240000 [audit 0] "s1", audit 0] ==
  [("leak", "up0", "certified", some 600000000, some 239920128, some 360079872, "2/25"), charged "up2, up4" 410000]
-- two certified links: a row a link, each at its own capacity and its own error, and the bare uplink charged
#guard egress [named today 0 600000 [audit 0], named (period "up2") 0 600000 [audit 0] "s1", audit 0] ==
  [("leak", "up0", "certified", some 600000000, some 599960064, some 39936, "1/25"),
   ("leak", "up2", "certified", some 240000000, some 239960064, some 39936, "1/25"), charged "up4" 10000]
-- the certifier's half to the outside is no boundary link: an answer that names it has no row
#guard egress [named (period "up1") 0 600000 [audit 0], audit 0] == egress []
-- a bare uplink is one, and no certifier splits it: an answer that names it reads as its row
#guard egress [named (period "up4") 0 600000 [audit 0], audit 0] ==
  [("leak", "up4", "certified", some 6000000, some 5960064, some 39936, "1/25"), charged "up0, up2" 1400000]

end AccountExample

end NetworkCheck.Period
