module

public import CoreProtocol.Account
public import NetworkCheck.Period
meta import CoreProtocol.Account
meta import NetworkCheck.Period

/-!
# What the network certifier's period check accounts for

The account (`Core.Account`) of `verity/certifier/v0`: one amount, `K`, the bits a covert sender in the server can move
through the network certifier over a certification period, a bound from above on the free bits of the link the
certification period is on, at the error `eps`. A value beside the protocol, which `Core.Account.rows` reads the period
check's accepted answers by.

The keys it names. The period check writes three of them:

* `K` and `eps`, in the public output (`publicJson`): the bound, and the probability it fails, the sum of its audits'
  δ over the certification period's execution phases;
* `link`, in the profile (`profileJson`): the link the certification period is on.

It writes none of the other five, and the link's row reads "not audited: charged at capacity" until an answer holds
all of them (the guards below):

* `registration`, in the profile: the registration's head. The period check writes it in the change that passes it the
  registered site and holds `link` to a registered link's id, and in no other;
* `session`, in the public output: the certifier session, which the period check's parameters hold (`session`);
* `start_ms` and `span_ms`, in the public output: the certification period's span on that session's clock, its first
  millisecond and its length in milliseconds. The clock is the network certifier's count of its own ticks: millisecond
  0 is the start of the session's first execution phase, an execution phase is T ticks, and a tick is a whole number of
  milliseconds. So the certification period of the parameters' `windows` execution phases from the session's `first`
  has `start_ms = first · T · bucket_ms` and `span_ms = windows · T · bucket_ms`, and a step of the account is a
  millisecond whatever T and the tick are. In the rows form they are the schedule's (`window`, `bucket_ms`), which
  the parameters hold. In the clock form they are the link's grid's, its buckets and their `bucket_ms`, which each
  execution phase's proof takes as a public input and the parameters do not hold;
* `certified`, in the public output: that the answer certifies its `K`. The period check writes it only as true, and
  only when both hold: the one-stage answers its `K` rests on are verified, which it takes as given, and the site's
  network certifier mediates the link's egress. An answer that certifies nothing holds no `certified`, and says why
  under a key of its own. A configuration verifies the one-stage answers and ties them to the period check's answer by
  key, while `Core.Account.rows` reads each answer alone and holds none to a configuration's acceptance.

The series of the account's steps is the session. The network certifier counts each session's ticks from that
session's start, so two sessions' spans are on two clocks, and no span of one is shown disjoint from a span of the
other: two certification periods of one session that share no millisecond both cover their time, and of two sessions
on one link the one that covers more covers alone.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace NetworkCheck.Period

open Lean (Json toJson)

/-- **The period check's account**: `K`, in bits, bounds the link's free bits from above over the answer's span, the
`span_ms` milliseconds from `start_ms` on the clock of the certifier session `session`, at the error `eps`; a monitoring
period's error is its answers' summed. `eps` holds the audits' δ alone: the error of the bound it is the union of names
three terms more, which it does not price. -/
def account : Core.Account :=
  { protocol := protocolId
    head := some (.profile "registration")
    amounts := [{ metric := "K", part := .link, place := some (.profile "link"), key := .published "K",
                  unit := Core.Account.bits, side := .free,
                  time := .steps (.run (.published "start_ms") (.published "span_ms")) ⟨1, 1000⟩,
                  series := some (.published "session"), error := .published "eps", pool := .sum,
                  uncertified := .flagged (.published "certified"),
                  unpriced := ["each sampled audit's knowledge soundness error", "the commitment's binding",
                               "the network certifier's openings"] }] }

/-! ## The rows of one link -/

namespace AccountExample

open Core.Registration.Example (sheet locs links auditor fig)

/-- The registration's example site: its one boundary link is `up0`, between the host and the network certifier, at
100 bits a unit of the Site's time. -/
private def site : Core.Site := ⟨[sheet], locs, links⟩
/-- The Site's unit of time, a millisecond, and a monitoring period of an hour of them. -/
private def tick : Core.Site.Figure := fig 1 1000
private def ticks : Nat := 3600000

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

/-- The span of the `windows` execution phases from a session's `first` under schedule `s`, in milliseconds on the
session's clock. -/
private def span (first windows : Nat) (s : Schedule := defaultSchedule) : List (String × Json) :=
  [("start_ms", toJson (first * s.window * s.bucketMs)), ("span_ms", toJson (windows * s.window * s.bucketMs))]

/-- An answer on `up0` that names the head, its `session` and its span, and holds no `certified`. -/
private def spanned (first : Nat) (session : String := "s0") (windows : Nat := 1) (s : Schedule := defaultSchedule) :
    Core.Answer :=
  answer "up0" (("session", .str session) :: span first windows s) registered

/-- The same answer certified. -/
private def named (first : Nat) (session : String := "s0") (windows : Nat := 1) (s : Schedule := defaultSchedule) :
    Core.Answer :=
  answer "up0" (("certified", .bool true) :: ("session", .str session) :: span first windows s) registered

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

#guard account.Formed
-- the keys the account reads that the period check writes: `K` and `eps` in its public output, `link` in its profile
#guard (["K", "eps"].map fun k => ((publicJson pub).getObjVal? k).toOption) ==
  [some (toJson (9088 : Nat)), some (.str "1/100")]
#guard ((profileJson prof).getObjVal? "link").toOption == some (.str "up0")
-- and those it does not write, in neither. `certified` goes into the public output in the change after which the
-- one-stage answers the period check reads are verified, and in no other: this guard then goes with it
#guard ["certified", "registration", "session", "span_ms", "start_ms"].all fun k =>
  ((publicJson pub).getObjVal? k).toOption.isNone && ((profileJson prof).getObjVal? k).toOption.isNone
-- the span is in milliseconds whatever the schedule: an execution phase of 600 ticks of 100 ms, and one of 32
#guard span 1 2 == [("start_ms", toJson (60000 : Nat)), ("span_ms", toJson (120000 : Nat))]
#guard span 1 2 { defaultSchedule with window := 32 } ==
  [("start_ms", toJson (3200 : Nat)), ("span_ms", toJson (6400 : Nat))]
-- the answer as the period check writes it counts for nothing, flagged or not: it names no registration's head
#guard linkRows account [] == notAudited
#guard linkRows account [answer "up0"] == notAudited
#guard linkRows (unflagged account) [answer "up0"] == notAudited
-- the gate: an answer that names the head, its session and its span and holds no `certified` gives no reading, so the
-- link stays not audited
#guard linkRows account [spanned 0] == notAudited
#guard linkRows account [spanned 0, spanned 1, spanned 0 "s1"] == notAudited
-- without the flag it would make the link's row: `K` beside the capacity of the time its span does not cover, set
-- against the link's capacity
#guard linkRows (unflagged account) [spanned 0] ==
  [("K", "up0", "certified", some 360000000, some 5990912, some 354009088, "1/100", 1)]
-- a link the registration does not hold makes no row, and the site's boundary link stays not audited
#guard linkRows (unflagged account) [answer "up9" (("session", .str "s0") :: span 0 1) registered] == notAudited
-- `certified` true: the row. False: `K` is dropped and all the time is charged at capacity, but in a row whose
-- standing still reads "certified", at the answers' error, so the period check never writes false. Anything but a
-- boolean: no reading
#guard linkRows account [named 0] ==
  [("K", "up0", "certified", some 360000000, some 5990912, some 354009088, "1/100", 1)]
#guard linkRows account [answer "up0" (("certified", .bool false) :: ("session", .str "s0") :: span 0 1) registered] ==
  [("K", "up0", "certified", some 360000000, some 0, some 360000000, "1/100", 1)]
#guard linkRows account
    [answer "up0" (("certified", .str "answers read as given") :: ("session", .str "s0") :: span 0 1) registered] ==
  notAudited
-- one account reads both schedules: an execution phase of 32 ticks covers its 3200 ms
#guard linkRows account [named 0 "s0" 1 { defaultSchedule with window := 32 }] ==
  [("K", "up0", "certified", some 360000000, some 310912, some 359689088, "1/100", 1)]
-- two certification periods of one session that share no millisecond both cover their time; two of one span cover it
-- once, as do the milliseconds two spans share; and each answer's `K` and `eps` count every time
#guard linkRows account [named 0, named 1] ==
  [("K", "up0", "certified", some 360000000, some 11981824, some 348018176, "1/50", 2)]
#guard linkRows account [named 0, named 0] ==
  [("K", "up0", "certified", some 360000000, some 5981824, some 354018176, "1/50", 2)]
#guard linkRows account [named 0 "s0" 2, named 1] ==
  [("K", "up0", "certified", some 360000000, some 11981824, some 348018176, "1/50", 2)]
-- two sessions do not add: each counts its ticks from its own start, so no span of one is shown disjoint from a span
-- of the other. Both `K` count, and one session's span alone covers its time
#guard linkRows account [named 0, named 1 "s1"] ==
  [("K", "up0", "certified", some 360000000, some 5981824, some 354018176, "1/50", 2)]
#guard linkRows account [named 0, named 1, named 2 "s1"] ==
  [("K", "up0", "certified", some 360000000, some 11972736, some 348027264, "3/100", 3)]
-- an answer that names no start, no length or no session covers no time
#guard linkRows account
    [answer "up0" [("certified", .bool true), ("session", .str "s0"), ("span_ms", toJson (60000 : Nat))] registered] ==
  [("K", "up0", "certified", some 360000000, some 0, some 360000000, "1/100", 1)]
#guard linkRows account
    [answer "up0" [("certified", .bool true), ("session", .str "s0"), ("start_ms", toJson (0 : Nat))] registered] ==
  [("K", "up0", "certified", some 360000000, some 0, some 360000000, "1/100", 1)]
#guard linkRows account [answer "up0" (("certified", .bool true) :: span 0 1) registered] ==
  [("K", "up0", "certified", some 360000000, some 0, some 360000000, "1/100", 1)]

end AccountExample

end NetworkCheck.Period
