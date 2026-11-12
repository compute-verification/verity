module

public import CoreProtocol.Account
public import NetworkCheck.Period
meta import CoreProtocol.Account
meta import NetworkCheck.Period

/-!
# What the network certifier's period check accounts for

The account (`Core.Account`) of `verity/certifier/v0`: one amount, `K`, the bits a covert sender in the server can move
through the network certifier over a certification period, a bound from above on the free bits of the link the period
is on, at the error `eps`. A value beside the protocol, which `Core.Account.rows` reads the period check's accepted
answers by.

The keys it names, and where the period check writes each:

* `K`, `eps` and `windows`, in the public output (`publicJson`): the bound, the probability it fails (the sum of its
  audits' δ over the period's execution phases) and the count of execution phases the period spans;
* `link`, in the profile (`profileJson`): the link the period is on;
* `registration`, in the profile: the registration's head. No answer holds it yet. The period check writes it once it
  takes the registered hardware as a parameter, and `link` is then a registered link's id;
* `certified`, in the public output: that the answer certifies its `K`. No answer holds it yet. The period check
  writes it, and only as true, once the one-stage answers its `K` rests on are verified, which it takes as given until
  then, and only for a site whose network certifier mediates the link's egress. An answer that certifies nothing
  holds no `certified`, and says why under a key of its own.

An answer that holds no `registration` counts for nothing, and one that holds no `certified` gives no reading, so until
the period check writes both the link's row reads "not audited: charged at capacity" (the guards below).

The answer states its span as a count of execution phases, neither their length nor which they are. The length is a
parameter here: `Schedule.phaseSeconds` in the rows form. A count alone shows no two spans disjoint, so of two
certification periods on one link only the longer covers its ticks; `accountNamed` is the account once the period
check publishes the index of its first execution phase.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace NetworkCheck.Period

open Lean (Json toJson)

/-- The seconds of one execution phase under schedule `s`: its `window` buckets, of `bucketMs` milliseconds each. -/
def Schedule.phaseSeconds (s : Schedule) : Core.Ratio := (Core.Ratio.mk' (s.window * s.bucketMs) 1000).getD ⟨0, 1⟩

/-- **The period check's account**, at execution phases of `seconds` seconds: `K`, in bits, bounds the link's free
bits from above over the answer's `windows` execution phases, at the error `eps`; a monitoring period's error is its
answers' summed. `eps` holds the audits' δ alone: the error of the bound it is the union of names three terms more,
which it does not price. -/
def account (seconds : Core.Ratio) : Core.Account :=
  { protocol := protocolId
    head := some (.profile "registration")
    amounts := [{ metric := "K", part := .link, place := some (.profile "link"), key := .published "K",
                  unit := Core.Account.bits, side := .free, time := .span (.published "windows") seconds,
                  error := .published "eps", pool := .sum, uncertified := .flagged (.published "certified"),
                  unpriced := ["each sampled audit's knowledge soundness error", "the commitment's binding",
                               "the network certifier's openings"] }] }

/-- The account once an answer names its span by the session's index of its first execution phase (the parameter
`first`, under the key `first`) and their count: two certification periods of one session that share no execution
phase then both cover their ticks. -/
def accountNamed (seconds : Core.Ratio) : Core.Account :=
  { account seconds with amounts := (account seconds).amounts.map fun m =>
      { m with time := .steps (.run (.published "first") (.published "windows")) seconds } }

/-! ## The rows of one link -/

namespace AccountExample

open Core.Registration.Example (sheet locs links auditor fig)

/-- The registration's example site: its one boundary link is `up0`, between the host and the network certifier, at
100 bits a tick. -/
private def site : Core.Site := ⟨[sheet], locs, links⟩
/-- A tick of a millisecond, and a period of an hour. -/
private def tick : Core.Site.Figure := fig 1 1000
private def ticks : Nat := 3600000
private def minute : Core.Ratio := defaultSchedule.phaseSeconds

/-- What the period check publishes on accepting one execution phase in the rows form: `K` = 9088 at `eps` = 1/100. -/
private def pub : Public :=
  rowsTerms #[{ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }] ⟨32, 0, 300, 512⟩
    #[({ units := 64, drawn := 64, delta := ⟨1, 100⟩, wrong := 0 }, 8 * 1136)] #[] 1 1

private def prof : Profile := { link := "up0", windows := pub.windows }

/-- The period check's accepted answer on `link`: its public output and profile as it writes them, and the keys
`published` and `profile` put into each. -/
private def answer (link : String) (published profile : List (String × Json) := []) : Core.Answer :=
  let add (j : Json) (kvs : List (String × Json)) : Json := kvs.foldl (fun j kv => j.setObjVal! kv.1 kv.2) j
  { protocol := protocolId, verdict := .accept, reason := none, digest := none,
    published := some (add (publicJson pub) published),
    profile := some (add (profileJson { prof with link }) profile) }

private def registered : List (String × Json) := [("registration", .str "h")]

/-- The account without its flag: the form of an answer that certifies its amount whenever it accepts. -/
private def unflagged (c : Core.Account) : Core.Account :=
  { c with amounts := c.amounts.map fun m => { m with uncertified := .refused } }

/-- The link's rows: the amount, the place, the standing, the capacity, the verified and the free bits, the error and
the count of answers read. -/
private def linkRows (c : Core.Account) (answers : List Core.Answer) :
    List (String × String × String × Option Nat × Option Nat × Option Nat × String × Nat) :=
  ((Core.Account.rows site Core.Site.Registration.sha512.sheet auditor "h" tick ticks [c] answers).filter
    fun r => r.part = .link).map fun r =>
      (r.metric, r.place, r.status.name, r.capacity, r.verified, r.free, r.error.toString, r.sessions)

private def notAudited : List (String × String × String × Option Nat × Option Nat × Option Nat × String × Nat) :=
  [("not audited", "up0", "not audited: charged at capacity", some 360000000, some 0, some 360000000, "0", 0)]

#guard defaultSchedule.phaseSeconds == ⟨60, 1⟩
#guard ({ defaultSchedule with window := 32 } : Schedule).phaseSeconds == ⟨16, 5⟩
#guard (account minute).Formed ∧ (accountNamed minute).Formed
-- the keys the account reads that the period check writes: `K`, `eps` and `windows` in its public output, `link` and
-- `windows` in its profile
#guard (["K", "eps", "windows"].map fun k => ((publicJson pub).getObjVal? k).toOption) ==
  [some (toJson (9088 : Nat)), some (.str "1/100"), some (toJson (1 : Nat))]
#guard (["link", "windows"].map fun k => ((profileJson prof).getObjVal? k).toOption) ==
  [some (.str "up0"), some (toJson (1 : Nat))]
-- and those it does not write yet, in neither. `certified` goes into the public output in the change after which the
-- one-stage answers the period check reads are verified, and in no other: this guard then goes with it
#guard ["certified", "first", "registration"].all fun k =>
  ((publicJson pub).getObjVal? k).toOption.isNone && ((profileJson prof).getObjVal? k).toOption.isNone
-- so the answer as the period check writes it counts for nothing, flagged or not: it names no registration's head
#guard linkRows (account minute) [] == notAudited
#guard linkRows (account minute) [answer "up0"] == notAudited
#guard linkRows (unflagged (account minute)) [answer "up0"] == notAudited
-- an answer that names the head and holds no `certified` gives no reading: the link stays not audited
#guard linkRows (account minute) [answer "up0" [] registered] == notAudited
-- without the flag it would make the link's row: `K` beside the capacity of the ticks the period does not cover, set
-- against the link's capacity, and two answers' `K` and `eps` summed, the longer span alone covering its ticks
#guard linkRows (unflagged (account minute)) [answer "up0" [] registered] ==
  [("K", "up0", "certified", some 360000000, some 5990912, some 354009088, "1/100", 1)]
#guard linkRows (unflagged (account minute)) [answer "up0" [] registered, answer "up0" [] registered] ==
  [("K", "up0", "certified", some 360000000, some 5981824, some 354018176, "1/50", 2)]
-- a link the registration does not hold makes no row, and the site's boundary link stays not audited
#guard linkRows (unflagged (account minute)) [answer "warden" [] registered] == notAudited
-- `certified` true: the row. False: `K` is dropped and every tick is charged at capacity, but in a row whose standing
-- still reads "certified", at the answers' error, so the period check never writes false. Anything but a boolean: no
-- reading
#guard linkRows (account minute) [answer "up0" [("certified", .bool true)] registered] ==
  [("K", "up0", "certified", some 360000000, some 5990912, some 354009088, "1/100", 1)]
#guard linkRows (account minute) [answer "up0" [("certified", .bool false)] registered] ==
  [("K", "up0", "certified", some 360000000, some 0, some 360000000, "1/100", 1)]
#guard linkRows (account minute) [answer "up0" [("certified", .str "answers read as given")] registered] == notAudited
-- spans named by their first execution phase: two that share none both cover their ticks, two of one span cover it
-- once, and an answer that names no first execution phase covers no tick
#guard linkRows (accountNamed minute)
    [answer "up0" [("certified", .bool true), ("first", toJson (0 : Nat))] registered,
     answer "up0" [("certified", .bool true), ("first", toJson (1 : Nat))] registered] ==
  [("K", "up0", "certified", some 360000000, some 11981824, some 348018176, "1/50", 2)]
#guard linkRows (accountNamed minute)
    [answer "up0" [("certified", .bool true), ("first", toJson (0 : Nat))] registered,
     answer "up0" [("certified", .bool true), ("first", toJson (0 : Nat))] registered] ==
  [("K", "up0", "certified", some 360000000, some 5981824, some 354018176, "1/50", 2)]
#guard linkRows (accountNamed minute) [answer "up0" [("certified", .bool true)] registered] ==
  [("K", "up0", "certified", some 360000000, some 0, some 360000000, "1/100", 1)]

end AccountExample

end NetworkCheck.Period
