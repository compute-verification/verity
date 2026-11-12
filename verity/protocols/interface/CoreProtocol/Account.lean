module

public import CoreProtocol.Example
public import CoreProtocol.Protocol
public import CoreProtocol.Ratio
public import CoreProtocol.Registration
public import CoreProtocol.Site
meta import CoreProtocol.Example
meta import CoreProtocol.Protocol
meta import CoreProtocol.Ratio
meta import CoreProtocol.Registration
meta import CoreProtocol.Site

/-!
# What a protocol accounts for, and the rows an auditor reads

A protocol's check (`Core.Protocol`) publishes its amounts under its own keys and in its own units. An `Account` is a
value beside a protocol, never a field of it: for each amount the protocol publishes (`Account.Amount`) it states

* the part of the registered hardware the amount is about (`Part`: the devices' compute, the memory of some tiers, the
  read rate of some tiers' memory, a link's egress), whose capacity is a function of the Site (`Part.capacity`), so a
  row's
  capacity is read from the registration and never copied from an answer;
* the form the amount is published in: its key, its unit, which side of the capacity it bounds (`Side`: the verified
  amount from below, or the free amount from above), a moment or the keys of its span (`Time`), the key of its
  error, the key of the share of the amount the error leaves out where the protocol reports one (ε at δ), how a
  period's error forms from its answers' (`Pool`), the key of the capacity where the answer carries one, and what an
  answer holds when nothing is certified (`Uncertified`);
* the terms the error names beside it and does not price (`unpriced`);

and, for the protocol, the key under which an accepted answer names the registration's head (`Account.head`), and the
answers its own rest on (`Account.rests`): those its check read as given and did not verify.

`rows` reads a site, the head of its registration, the protocols' answers and their accounts into the accounting rows.
An answer counts only when it accepts, names that head, and rests only on accepted answers `rows` is given. Each amount
certified of a registered resource gets a row, at each reading of the registration its capacity can be read at: the
capacity, the certified amount with its error, and the other, which follows.

* A flow (compute, reads, a link's bits) has one row for the period. Two answers' amounts add only where both name
  their spans (`Steps`), in one series of steps, and the spans share no step: a verified amount is the largest total
  over answers of one series whose spans are pairwise disjoint, or the largest single amount of an answer that names
  none, whichever is more. A free amount is its answers' summed plus the capacity of the ticks no certified answer's
  span covers, each tick's cover counted once: one series' named spans' steps, or the longest span that names none.
* An amount at a moment (memory) has one row for each accepted answer, a session, and one for the period, which states
  the least verified (or the most free) of its sessions and counts them (`Row.sessions`).
* A period's error, of a flow and of a moment alike, is its answers' errors as the amount's `Pool` says: summed, or
  the one answer's that states it once for the period.
* A row whose error is 1 or more is `Status.noBound`: it prints "no bound at this period" in place of a number.
* What no accepted answer covers is `Status.notAudited`: charged at capacity. So is the memory of a device's sheet
  that none of its leaves holds.

The comparison of the rows with what an auditor states it accepts is not here.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core

namespace Account

open Lean (Json)
open Site (Tier Figure Sheet)

/-- Where an accepted answer holds a value: under a key of its public output, or of its profile. -/
inductive Key where
  | published (key : String)
  | profile (key : String)
  deriving DecidableEq, Repr

/-- The value `a` holds under `k`: none when `a` is no accept, or holds no such key. -/
def Key.read (k : Key) (a : Answer) : Option Json :=
  match k with
  | .published key => a.published.bind fun j => (j.getObjVal? key).toOption
  | .profile key => a.profile.bind fun j => (j.getObjVal? key).toOption

/-- The tag an answer's name is hashed under. -/
def ANSWER : String := "verity/account/answer/v1"

/-- **An answer's name**: SHA-512 of the tag and its canonical JSON (`Answer.canonical`), in hex. -/
def nameOf (a : Answer) : Option String :=
  a.canonical.toOption.map fun s => Flock.hexOf (Flock.Sha512.hash (ANSWER.toUTF8.push 0 ++ s.toUTF8))

/-- **What an answer rests on**: answers its check read as given and did not verify. An accepted answer lists the name
(`nameOf`) of each under `key`, and each is to be an accepted answer of one of `protocols`. -/
structure Rests where
  key : Key
  protocols : List String
  deriving DecidableEq, Repr

/-- Every answer `a` rests on is among `answers`, accepted, and of a protocol `r` names. False when `a` holds no list
of names under `r`'s key. -/
def Rests.held (r : Rests) (a : Answer) (answers : List Answer) : Bool :=
  match (r.key.read a).bind fun j => j.getArr?.toOption.bind fun xs => xs.toList.mapM fun x => x.getStr?.toOption with
  | none => false
  | some names => names.all fun n => answers.any fun b =>
      b.verdict == .accept && r.protocols.contains b.protocol && nameOf b == some n

/-- A part of the registered hardware an amount is about: the devices' compute, in tensor-core MACs; the memory of some
tiers, in bits held; the read rate of some tiers' memory, in bits read from it; a boundary link's egress, in bits sent
over it out of the Site, whose capacity is the link's `β`, which is each way. -/
inductive Part where
  | compute
  | memory (tiers : List Tier)
  | read (tiers : List Tier)
  | egress
  deriving DecidableEq, Repr

namespace Part

/-- A flow is counted over a span of ticks; memory is held at a moment. -/
def flow : Part → Bool
  | .memory _ => false
  | _ => true

/-- The unit a part's rows are in. -/
def unit : Part → String
  | .compute => "MACs"
  | _ => "bits"

def name : Part → String
  | .compute => "compute"
  | .memory ts => "memory/" ++ "+".intercalate (ts.map fun t => t.name)
  | .read ts => "read/" ++ "+".intercalate (ts.map fun t => t.name)
  | .egress => "egress"

/-- Every part a site can hold, tier by tier. -/
def all : List Part :=
  [.compute, .memory [.hbm], .memory [.dram], .memory [.disk], .read [.hbm], .read [.dram], .read [.disk], .egress]

/-- Two parts are of one resource: both compute, both memory, both a read rate, or both a link's egress. -/
def same : Part → Part → Bool
  | .compute, .compute => true
  | .memory _, .memory _ => true
  | .read _, .read _ => true
  | .egress, .egress => true
  | _, _ => false

/-- The tiers a part names, where it names tiers. -/
def tiers : Part → Option (List Tier)
  | .memory ts => some ts
  | .read ts => some ts
  | _ => none

/-- A part names at least one tier where it names tiers. -/
def Formed (p : Part) : Prop := p.tiers ≠ some []

instance instDecidableFormed (p : Part) : Decidable p.Formed := by unfold Formed; infer_instance

/-- The ids of a site's leaves of a part: its devices, its tiers' memory leaves, or its boundary links. -/
def leaves (p : Part) (S : Site) : List String :=
  match p with
  | .compute => S.units
  | .memory ts => (S.locs.filter fun x => x.kind = .memory ∧ x.tier ∈ ts.map some).map (·.id)
  | .read ts => (S.locs.filter fun x => x.kind = .memory ∧ x.tier ∈ ts.map some).map (·.id)
  | .egress => (S.boundary none).map (·.id)

/-- The part's leaves at `place`: all of them when an answer names no place, a link by its id, and otherwise the leaves
at or under the location. -/
def leavesAt (p : Part) (S : Site) (place : Option String) : List String :=
  match place, p with
  | none, _ => p.leaves S
  | some i, .egress => (p.leaves S).filter (· == i)
  | some i, _ => (p.leaves S).filter fun x => (S.loc? x).any (S.within i (S.locs.length + 1))

/-- Device `d`'s tensor-core MACs a tick, the device model's (`Site.device`) under auditor `A`. -/
def tensorRate (S : Site) (digest : Sheet → String) (A : Site.Registration.Auditor) (d : String) : Option Nat :=
  (S.device digest A d 0 0 (fun (e : Empty) => nomatch e)).map fun m => m.rate .tc

/-- The reading a read rate's row prints where a host's disk leaf is among its leaves: the rate counts that host's whole
DRAM tier's rate once, since a disk leaf's bytes can be returned from memory (`Site.readRate`). -/
def READ_WITH_DRAM : String := "registered, the host's DRAM tier counted once beside its disk"

/-- **The registered capacity** of the leaves `xs` of a part, at each reading of the registration it can be read at: a
rate a tick for a flow, an amount for memory. Compute has two readings: `stated`, the registered sheets' own figures,
and `bound`, the larger of those and the auditor's estimate, which is the device model's rate. A read rate is the
`bw` of the sheets of the devices its leaves are under; no sheet states one for a leaf under a host, so the list is
empty where a leaf has none: the registration holds no rate for the part. -/
def capacity (p : Part) (S : Site) (digest : Sheet → String) (A : Site.Registration.Auditor) (ticks : Nat)
    (xs : List String) : List (String × Nat) :=
  match p with
  | .compute =>
    let stated : Option (List Nat) := xs.mapM fun d => (S.sheetOf digest d).bind fun k =>
      tensorRate S digest { A with sheet := fun _ => some k } d
    let bound : Option (List Nat) := xs.mapM fun d => tensorRate S digest A d
    (stated.map fun ns => ("stated", ns.sum * ticks)).toList ++ (bound.map fun ns => ("bound", ns.sum * ticks)).toList
  | .memory _ => [("registered", 8 * (xs.filterMap fun x => (S.loc? x).map fun y => y.bytes).sum)]
  | .read _ =>
    let disk : Bool := xs.any fun x => (S.loc? x).any fun (y : Site.Loc) => y.bw.isSome && decide (y.tier = some .disk)
    (((S.readRate digest xs).bind fun n => S.over n ticks).map fun n =>
      (if disk then READ_WITH_DRAM else "registered", n)).toList
  | .egress =>
    let stated : Option (List Nat) :=
      (xs.filterMap fun i => S.links.find? fun (k : Site.Link) => k.id == i).mapM fun (k : Site.Link) => k.β
    ((stated.bind fun bs => S.over bs.sum ticks).map fun n => ("registered", n)).toList

end Part

/-- A unit an amount is published in: its name, and how many of its part's unit (a bit, or a MAC of compute) one of it
is. -/
structure Measure where
  name : String
  base : Nat
  deriving DecidableEq, Repr

def bits : Measure := ⟨"bits", 1⟩
def bytes : Measure := ⟨"bytes", 8⟩
def mebibytes : Measure := ⟨"MiB", 8 * 2 ^ 20⟩
def macs : Measure := ⟨"MACs", 1⟩

/-- Which side of the capacity an amount bounds: the verified amount from below, or the free amount from above. -/
inductive Side where
  | verified
  | free
  deriving DecidableEq, Repr

/-- How an answer names the steps its span covers, so that two answers' spans can be shown disjoint: by the index of its
one step (an epoch); by the indices of its first and last steps, both included; or by the index of its first step and
the count of its steps (a list counts its entries). The indices are those of one series of steps (`Amount.series`). -/
inductive Steps where
  | index (key : Key)
  | range (first last : Key)
  | run (first count : Key)
  deriving DecidableEq, Repr

/-- When an amount holds: at a moment; over a span whose length the answer holds under `key`, as a count of steps of
`seconds` seconds each (a list counts its entries), without saying which steps; over a span whose steps, of `seconds`
seconds each, the answer names (`Steps`); or over a span the answer does not state, which covers no tick. Only spans
that name their steps can be shown disjoint, so only their amounts and their cover add (`Amount.rows`). -/
inductive Time where
  | moment
  | span (key : Key) (seconds : Ratio)
  | steps (name : Steps) (seconds : Ratio)
  | unstated
  deriving DecidableEq, Repr

/-- How a period's error forms from its accepted answers' errors: summed, or stated once by the one answer of the
period, which pools it. -/
inductive Pool where
  | sum
  | once
  deriving DecidableEq, Repr

/-- What an answer holds when its check certifies nothing: an accepted answer that says so under a boolean key, or one
whose amount is already at its safe end (nothing verified, or everything free), or no accepted answer at all, the check
refusing. -/
inductive Uncertified where
  | flagged (key : Key)
  | safe
  | refused
  deriving DecidableEq, Repr

/-- **The form of one amount a protocol publishes**, and the part of the registered hardware it is about. `metric` names
the amount among its protocol's; `place` is the key under which an answer names the location or the link it ran on, none
when it is about the whole site. `series` is the key under which an answer names the series its steps are numbered in
(a session whose steps are counted from its own start), none when every answer's steps are numbered in one series: two
answers that name different series cannot be shown disjoint. -/
structure Amount where
  metric : String
  part : Part
  place : Option Key := none
  key : Key
  unit : Measure
  side : Side
  time : Time
  series : Option Key := none
  share : Option Key := none
  error : Key
  pool : Pool := .sum
  capacity : Option Key := none
  uncertified : Uncertified
  unpriced : List String := []
  deriving DecidableEq, Repr

/-- A form `rows` reads: a part that names a tier where it names tiers, a moment exactly for memory, a share only beside
a verified amount, and a unit that is not empty. -/
def Amount.Formed (m : Amount) : Prop :=
  m.part.Formed ∧ (m.time = .moment ↔ m.part.flow = false) ∧ (m.share.isSome = true → m.side = .verified) ∧
    0 < m.unit.base

instance Amount.instDecidableFormed (m : Amount) : Decidable m.Formed := by unfold Amount.Formed; infer_instance

end Account

/-- **What a protocol accounts for**: its id, the key under which an accepted answer names the registration's head (none
while the protocol names no registration: `Account.rows` then holds its answers to none), the form of each amount it
publishes, and the answers its own rest on (`rests`, none for a protocol whose check verifies all it reads): an answer
that rests on one `Account.rows` was not given, or on one that does not accept, counts for nothing. -/
structure Account where
  protocol : String
  head : Option Account.Key
  amounts : List Account.Amount
  rests : Option Account.Rests := none
  deriving DecidableEq, Repr

namespace Account

open Lean (Json)
open Site (Tier Figure Sheet)

/-- Every amount's form is one `rows` reads, and no two have one name. -/
def Formed (c : Account) : Prop := (∀ m ∈ c.amounts, m.Formed) ∧ (c.amounts.map fun m => m.metric).Nodup

instance instDecidableFormed (c : Account) : Decidable c.Formed := by unfold Formed; infer_instance

/-- What one accepted answer certifies of one amount, in its part's unit: the place it names, whether it certifies
anything, the amount with its share taken off, its error, the ticks its span covers, the steps it names its span by
(the first, and the one after the last; none where it names none) with the series they are numbered in, and the
capacity it carries. -/
structure Reading where
  place : Option String
  certified : Bool
  amount : Nat
  error : Ratio
  ticks : Nat
  steps : Option (Nat × Nat) := none
  series : String := ""
  capacity : Option Nat
  deriving DecidableEq, Repr

private def natOf (j : Json) : Option Nat := j.getNat?.toOption

private def ratioOf (j : Json) : Option Ratio := j.getStr?.toOption.bind fun s => (Ratio.ofString s "").toOption

/-- The naturals under an amount's key: one, or a list of them. -/
private def natsOf : Json → Option (List Nat)
  | .arr xs => xs.toList.mapM natOf
  | j => (natOf j).map fun n => [n]

/-- A share: `null` where none is established. -/
private def shareOf : Json → Option (Option Ratio)
  | .null => some none
  | j => (ratioOf j).map some

/-- The shares under a share's key, one beside each amount. -/
private def sharesOf : Json → Option (List (Option Ratio))
  | .arr xs => xs.toList.mapM shareOf
  | j => (shareOf j).map fun e => [e]

/-- The count under a span's key: a natural, or the entries of a list. -/
private def countOf : Json → Option Nat
  | .arr xs => some xs.size
  | j => natOf j

/-- What a share `e` leaves of `n`, rounded down: nothing where no share is established. -/
private def kept (n : Nat) : Option Ratio → Nat
  | none => 0
  | some e => n * (e.den - e.num) / e.den

/-- The ticks of `count` steps of `seconds` seconds at ticks of `tick` seconds, rounded down; none at a tick of 0. -/
def ticksOf (count : Nat) (seconds : Ratio) (tick : Figure) : Nat :=
  count * seconds.num * tick.1.den / (seconds.den * tick.1.num.toNat)

/-- The ticks of `count` steps of `seconds` seconds at ticks of `tick` seconds, rounded up. -/
def ticksUp (count : Nat) (seconds : Ratio) (tick : Figure) : Nat :=
  let d : Nat := seconds.den * tick.1.num.toNat
  (count * seconds.num * tick.1.den + (d - 1)) / d

/-- The ticks `count` steps of `seconds` seconds cover when a clock that may run `drift` from real time, as a share of
it, counted them: the steps' length less that share, rounded down once. A clock that states no drift covers no tick. -/
def ticksCredited (count : Nat) (seconds : Ratio) (tick : Figure) (drift : Option Figure) : Nat :=
  match drift with
  | none => 0
  | some q =>
    count * seconds.num * (q.1.den - q.1.num.toNat) * tick.1.den / (seconds.den * q.1.den * tick.1.num.toNat)

/-- The ticks to charge for `count` steps of `seconds` seconds that a clock which may run `drift` from real time
counted: the steps' length over what the drift leaves of it, rounded up once. None where the clock states no drift, or
one of 1: such a time has no bound. -/
def ticksCharged (count : Nat) (seconds : Ratio) (tick : Figure) (drift : Option Figure) : Option Nat :=
  drift.bind fun q =>
    let d : Nat := seconds.den * (q.1.den - q.1.num.toNat) * tick.1.num.toNat
    if d = 0 then none else some ((count * seconds.num * q.1.den * tick.1.den + (d - 1)) / d)

/-- What a clock that may run `drift` from real time leaves of `ticks` ticks it counted, rounded down: nothing where it
states no drift. -/
def ticksKept (ticks : Nat) (drift : Option Figure) : Nat :=
  match drift with
  | none => 0
  | some q => ticks * (q.1.den - q.1.num.toNat) / q.1.den

/-- The steps of `seconds` seconds that surely end within `ticks` ticks when a clock that may run `drift` from real
time counts them: as many as, charged (`ticksCharged`), still fit. None where the clock states no drift. -/
def stepsWithin (ticks : Nat) (seconds : Ratio) (tick : Figure) (drift : Option Figure) : Nat :=
  match drift with
  | none => 0
  | some q =>
    ticks * tick.1.num.toNat * (q.1.den - q.1.num.toNat) * seconds.den / (tick.1.den * q.1.den * seconds.num)

/-- **Where a series of steps lies in the monitoring period**: step `first` of the series begins at the period's tick
`tick`, and `device` is the auditor's device that numbers the steps, on the clock of the auditor's host it is in, so
the drift the Site states for that host is the series' (`Site.drift?`). A series older than the period gives the step
it was at when the period began, at tick 0; one that began in the period gives its step 0, at the tick it began. -/
structure Origin where
  first : Nat
  tick : Nat
  device : String
  deriving DecidableEq, Repr

/-- The steps of `x` (the first, and the one after the last) that surely lie in a period of `ticks` ticks, for a series
placed at `o` on a clock that may run `drift` from real time: none before `o.first`, and none that might end past the
period's last tick (`stepsWithin`). None when no step is left. -/
def Origin.clip (o : Origin) (seconds : Ratio) (tick : Figure) (ticks : Nat) (drift : Option Figure)
    (x : Nat × Nat) : Option (Nat × Nat) :=
  let a : Nat := max x.1 o.first
  let b : Nat := min x.2 (o.first + stepsWithin (ticks - o.tick) seconds tick drift)
  if a < b then some (a, b) else none

/-- The period's ticks the steps `x` of a series placed at `o` can touch: none before the first, and none from the
second on. The start is the steps before it credited and the end the steps to it charged, so the true ticks lie
between whatever the clock's rate was. -/
def Origin.bounds (o : Origin) (seconds : Ratio) (tick : Figure) (ticks : Nat) (drift : Option Figure)
    (x : Nat × Nat) : Nat × Nat :=
  (min ticks (o.tick + ticksCredited (x.1 - o.first) seconds tick drift),
   min ticks (o.tick + (ticksCharged (x.2 - o.first) seconds tick drift).getD ticks))

/-- The steps `x` surely lie in the period: none is before `o.first`, and the last surely ends by the period's last
tick. -/
def Origin.inside (o : Origin) (seconds : Ratio) (tick : Figure) (ticks : Nat) (drift : Option Figure)
    (x : Nat × Nat) : Bool :=
  decide (o.first ≤ x.1) && decide (x.2 ≤ o.first + stepsWithin (ticks - o.tick) seconds tick drift)

/-- **A monitoring period, as `rows` reads it**: its length in ticks; every head the registration had in it, an answer
counting when it names one of them; and where each series of steps lies in it (`Origin`), by the series' name as an
answer writes it: the compact JSON under its amount's `series`, or the empty text where the amount names none. A series
with no origin covers no tick and adds no amount. The three are the auditor's: no answer states them. -/
structure Period where
  ticks : Nat
  heads : List String
  origin : String → Option Origin

/-- The steps answer `a` names its span by: its first, and the one after its last. None where it does not hold the
keys, or names no step. -/
def Steps.read (name : Steps) (a : Answer) : Option (Nat × Nat) :=
  match name with
  | .index k => ((k.read a).bind natOf).map fun i => (i, i + 1)
  | .range first last =>
    ((first.read a).bind natOf).bind fun i => ((last.read a).bind natOf).bind fun j =>
      if i ≤ j then some (i, j + 1) else none
  | .run first count =>
    ((first.read a).bind natOf).bind fun i => ((count.read a).bind countOf).bind fun n =>
      if 0 < n then some (i, i + n) else none

/-- **An answer read by an amount's form**: none when the answer does not hold what the form names. An answer that does
not name the steps its form asks for, or the series they are numbered in, is read as one that names no span: it covers
no tick. -/
def Amount.read (m : Amount) (tick : Figure) (a : Answer) : Option Reading := do
  let place : Option String ← match m.place with
    | none => some none
    | some k => (k.read a).bind fun j => j.getStr?.toOption.map some
  let ns : List Nat ← (m.key.read a).bind natsOf
  let es : List (Option Ratio) ← match m.share with
    | none => some (ns.map fun _ => some (Ratio.ofNat 0))
    | some k => (k.read a).bind sharesOf
  guard (es.length == ns.length)
  let error : Ratio ← (m.error.read a).bind ratioOf
  let certified : Bool ← match m.uncertified with
    | .flagged k => (k.read a).bind fun j => j.getBool?.toOption
    | _ => some true
  let series : Option String := match m.series with
    | none => some ""
    | some k => (k.read a).map fun j => j.compress
  let steps : Option (Nat × Nat) := match m.time, series with
    | .steps name _, some _ => name.read a
    | _, _ => none
  let ticks : Nat ← match m.time with
    | .moment => some 0
    | .unstated => some 0
    | .span k seconds => ((k.read a).bind countOf).map fun n => ticksOf n seconds tick
    | .steps _ seconds => some ((steps.map fun s => ticksOf (s.2 - s.1) seconds tick).getD 0)
  let capacity : Option Nat ← match m.capacity with
    | none => some none
    | some k => (k.read a).bind fun j => (natOf j).map fun n => some (n * m.unit.base)
  pure { place, certified, error, ticks, steps, capacity, series := series.getD "",
         amount := m.unit.base * ((ns.zip es).map fun x => kept x.1 x.2).sum }

/-- A row's standing: an amount certified, an error of 1 or more ("no bound at this period"), or a resource no accepted
answer covers ("not audited: charged at capacity"). -/
inductive Status where
  | certified
  | noBound
  | notAudited
  | unmediated
  deriving DecidableEq, Repr

def Status.name : Status → String
  | .certified => "certified"
  | .noBound => "no bound at this period"
  | .notAudited => "not audited: charged at capacity"
  | .unmediated => "egress not mediated: no bound"

/-- **An accounting row**: the amount (`metric`) of a part at a place, at one reading of the registration; the period's
row, or the row of its `session`-th accepted answer; how many accepted answers it reads; and the capacity, the verified
and the free amounts in the part's unit, with the error, the terms it names and does not price, and the premises it
rests on beyond its answers' own, by name (`premises`). A row that names a premise is no figure of record, and no total
of record counts it. An amount is none where there is no number to print: no bound, or no registered capacity. -/
structure Row where
  metric : String
  part : Part
  place : String
  reading : String
  session : Option Nat := none
  sessions : Nat := 0
  status : Status
  side : Option Side := none
  capacity : Option Nat
  verified : Option Nat
  free : Option Nat
  error : Ratio := ⟨0, 1⟩
  unpriced : List String := []
  premises : List String := []
  deriving DecidableEq, Repr

/-- **A drift taken on a premise**: how far the clock of the auditor's host `host` may run from real time, as a share
of it, where no data sheet bounds it and the Site states none for the host; `premise` is the name of the premise that
says so, as the row that rests on it prints it. It is the drift of every device of the auditor's in the host. The
three are the auditor's: no answer states them. -/
structure Premised where
  host : String
  drift : Figure
  premise : String
  deriving DecidableEq, Repr

private def whole : Ratio := ⟨1, 1⟩

private def plus (a b : Ratio) : Ratio := (Ratio.mk' (a.num * b.den + b.num * a.den) (a.den * b.den)).getD whole

/-- The largest total of the amounts of spans no two of which share a step. Each span is its first step, the step
after its last, and its amount. -/
private def disjointMost (xs : List (Nat × Nat × Nat)) : Nat :=
  let sorted : List (Nat × Nat × Nat) := xs.mergeSort fun a b => decide (a.2.1 ≤ b.2.1)
  let table : List (Nat × Nat) := sorted.foldl (fun t x =>
    let before : Nat := ((t.filter fun e => e.1 ≤ x.1).map fun e => e.2).foldl max 0
    let sofar : Nat := (t.map fun e => e.2).foldl max 0
    t ++ [(x.2.1, max sofar (x.2.2 + before))]) []
  (table.map fun e => e.2).foldl max 0

/-- The spans' union, as spans in order no two of which share a step or touch. Each span is its first step and the step
after its last. -/
private def merged (xs : List (Nat × Nat)) : List (Nat × Nat) :=
  (xs.mergeSort fun a b => decide (a.1 ≤ b.1)).foldl (fun (acc : List (Nat × Nat)) x =>
    match acc.getLast? with
    | some y => if x.1 ≤ y.2 then acc.dropLast ++ [(y.1, max y.2 x.2)] else acc ++ [x]
    | none => [x]) []

/-- The ticks two lists of spans of ticks can share: each pair's overlap, summed. -/
private def sharedTicks (xs ys : List (Nat × Nat)) : Nat :=
  (xs.flatMap fun a => ys.map fun b => min a.2 b.2 - max a.1 b.1).sum

/-- The ticks several series surely cover together. Each series gives the ticks it surely covers and the ticks its
spans can touch; the series' covers add, less what any two can share, and no less than the largest alone. -/
private def together (each : List (Nat × List (Nat × Nat))) : Nat :=
  let common : Nat :=
    (each.zipIdx.flatMap fun (a, i) => (each.drop (i + 1)).map fun b => sharedTicks a.2 b.2).sum
  max ((each.map fun e => e.1).foldl max 0) ((each.map fun e => e.1).sum - common)

/-- The row of amount `m` certified as `n` at `error`, against capacity `total`: the other amount follows, and neither
is printed at an error of 1 or more. A free amount is at most the capacity. -/
private def certifiedRow (m : Amount) (place reading : String) (total : Option Nat) (n : Nat) (error : Ratio)
    (sessions : Nat) (session : Option Nat) : Row :=
  let n : Nat := match m.side, total with
    | .free, some c => min n c
    | _, _ => n
  let other : Option Nat := total.map fun c => c - n
  let bounded : Bool := !whole.le error
  { metric := m.metric, part := m.part, place, reading, session, sessions, side := some m.side, error,
    status := if bounded then .certified else .noBound,
    capacity := total,
    verified := if bounded then (match m.side with | .verified => some n | .free => other) else none,
    free := if bounded then (match m.side with | .verified => other | .free => some n) else none,
    unpriced := m.unpriced }

/-- The rows of amount `m` at `place`, from its accepted answers' readings `rs`, at each reading of the capacity of the
leaves it is about. None when the site holds no such leaf.

A named span counts only where it surely lies in the period, which `origin` says for its series: a series with no
origin, one whose device has no drift under `clock`, a step before the period and a step that might end past its last
tick count for nothing. `clock` gives a device's drift: the Site's own for a row of record. A span that names no steps
cannot be placed, and that it is the period's is the caller's to hold; it covers its length less its clock's drift.

A flow's amounts add only over spans shown disjoint. Its verified amount is the largest total over certified answers
whose named spans lie whole in the period, are of one series and are pairwise disjoint, or the largest amount of one
that names no span, whichever is more: two answers of one span, or of overlapping spans, count the larger once, and an
answer that names no span, or a span of another series, adds to none. The total is sound because each span's amount is
a lower bound on what its own steps did, and what disjoint steps did adds: of three spans in a chain, the two ends add
where they outweigh the middle. Its free amount is the amounts of the certified answers with a step surely in the
period, summed, plus the capacity of the period's ticks they do not surely cover. A series' cover is its steps surely
in the period, joined and counted once, their length less its clock's drift, rounded down once a series. Two series'
covers add less the ticks their spans can share, each span taken at the widest its clock's drift allows (`together`).
The longest span that names no steps counts instead where it is more. Where the registration holds no capacity for the
ticks not covered it prints no amount.

An amount at a moment has a row for each session and one for the period: the least verified, or the most free, of its
sessions. The period's error is its answers' as the amount's `pool` says, for a flow and for a moment alike. -/
def Amount.rows (m : Amount) (S : Site) (digest : Sheet → String) (A : Site.Registration.Auditor) (tick : Figure)
    (ticks : Nat) (origin : String → Option Origin) (clock : String → Option Figure) (place : Option String)
    (rs : List Reading) : List Row :=
  let xs := m.part.leavesAt S place
  let totals : List (String × Option Nat) :=
    match m.part.capacity S digest A ticks xs with
    | [] => [("", none)]
    | cs => cs.map fun c => (c.1, some c.2)
  let at_ : String := place.getD ""
  let error : Ratio := match m.pool, rs with
    | .once, [r] => r.error
    | .once, _ => whole
    | .sum, _ => rs.foldl (fun e r => plus e r.error) ⟨0, 1⟩
  if xs.isEmpty then [] else totals.flatMap fun (reading, total) =>
    if m.part.flow then
      let good := rs.filter fun r => r.certified
      let unnamed : List Reading := good.filter fun r => r.steps.isNone
      let seconds : Ratio := match m.time with
        | .steps _ s => s
        | _ => ⟨0, 1⟩
      let series : List String := (good.map fun r => r.series).eraseDups
      let drift : String → Option Figure := fun s => (origin s).bind fun o => clock o.device
      let within : List (List (Nat × Nat × Nat)) := series.map fun s =>
        good.filterMap fun r => (origin s).bind fun o => r.steps.bind fun x =>
          if r.series = s ∧ o.inside seconds tick ticks (drift s) x then some (x.1, x.2, r.amount) else none
      let covers : List (Nat × List (Nat × Nat)) := series.map fun s =>
        match origin s with
        | none => (0, [])
        | some o =>
          let sure : List (Nat × Nat) :=
            (merged (good.filterMap fun r => if r.series = s then r.steps else none)).filterMap fun x =>
              o.clip seconds tick ticks (drift s) x
          (ticksCredited ((sure.map fun x => x.2 - x.1).sum) seconds tick (drift s),
           sure.map fun x => o.bounds seconds tick ticks (drift s) x)
      let reaching : List Reading := good.filter fun r =>
        ((origin r.series).bind fun o => r.steps.bind fun x => o.clip seconds tick ticks (drift r.series) x).isSome
      let cover : Nat :=
        max (together covers) ((unnamed.map fun r => ticksKept r.ticks (drift r.series)).foldl max 0)
      let uncovered : Nat := ticks - cover
      let charged : Nat := ((m.part.capacity S digest A uncovered xs).lookup reading).getD 0
      let n : Nat := match m.side with
        | .verified => max ((within.map disjointMost).foldl max 0) ((unnamed.map fun r => r.amount).foldl max 0)
        | .free => ((reaching ++ unnamed).map fun r => r.amount).sum + charged
      let row : Row := certifiedRow m at_ reading total n error rs.length none
      if m.side = .free ∧ total = none ∧ 0 < uncovered then
        [{ row with status := .notAudited, verified := none, free := none }]
      else [row]
    else
      let each : List (Nat × Ratio) := rs.map fun r =>
        let fits : Bool := r.certified && (r.capacity.all fun c => decide (c ≤ total.getD c))
        let rest : Nat := match r.capacity, total with
          | some c, some C => C - c
          | _, _ => 0
        match m.side, fits with
        | .verified, true => (r.amount, r.error)
        | .verified, false => (0, r.error)
        | .free, true => (r.amount + rest, r.error)
        | .free, false => (total.getD r.amount, r.error)
      let worst : Nat := match m.side with
        | .verified => each.foldl (fun n x => min n x.1) ((each.head?.map fun x => x.1).getD 0)
        | .free => each.foldl (fun n x => max n x.1) 0
      (each.zipIdx.map fun (x, i) => certifiedRow m at_ reading total x.1 x.2 1 (some i)) ++
        [certifiedRow m at_ reading total worst error rs.length none]

/-- **What the accepted answers certify**, amount by amount. An answer counts for an account when it names the
account's protocol, accepts, names one of `heads` under the account's key, and rests only on accepted answers among
`answers` (`Rests.held`); of such an answer, each amount of a form `rows` reads that the answer holds. -/
def readings (heads : List String) (tick : Figure) (accounts : List Account) (answers : List Answer) :
    List (Amount × Reading) :=
  accounts.flatMap fun c =>
    let named : List Answer := answers.filter fun a =>
      a.protocol == c.protocol && a.verdict == .accept &&
        ((c.head.bind fun k => (k.read a).bind fun j => j.getStr?.toOption).any fun h => heads.contains h) &&
        c.rests.all fun r => r.held a answers
    (c.amounts.filter fun m => m.Formed).flatMap fun m => named.filterMap fun a => (m.read tick a).map fun r => (m, r)

/-- The rows of what no accepted answer covers: for each part, the leaves under no reading's place, charged at capacity
over the period. Their capacity is none where the registration holds none. -/
def unaudited (S : Site) (digest : Sheet → String) (A : Site.Registration.Auditor) (ticks : Nat)
    (cs : List (Amount × Reading)) : List Row :=
  Part.all.flatMap fun p =>
    let covered : List String := (cs.filter fun x => x.1.part.same p).flatMap fun x => x.1.part.leavesAt S x.2.place
    let rest : List String := (p.leaves S).filter fun x => !covered.contains x
    let totals : List (String × Option Nat) :=
      match p.capacity S digest A ticks rest with
      | [] => [("", none)]
      | ks => ks.map fun k => (k.1, some k.2)
    if rest.isEmpty then [] else totals.map fun (reading, total) =>
      { metric := "not audited", part := p, place := ", ".intercalate rest, reading, status := .notAudited,
        capacity := total, verified := some 0, free := total }

/-- The rows of the device memory no leaf holds: a device's sheet states its memory, its memory leaves hold at most
that (`Site.Entry.Capacity`), and an answer names leaves, so the bytes of the sheet beyond the leaves' are charged at
capacity. -/
def unregistered (S : Site) (digest : Sheet → String) : List Row :=
  S.units.filterMap fun d => (S.sheetOf digest d).bind fun k =>
    let rest : Nat := 8 * (k.mem - S.held d)
    if rest = 0 then none
    else some { metric := "not audited", part := .memory [.hbm], place := d, reading := "registered",
                status := .notAudited, capacity := some rest, verified := some 0, free := some rest }

/-- **The accounting rows** of the site `S`, over the monitoring period `period` in the Site's units of time, from the
protocols' `answers` read by their `accounts`: the rows of each amount at each place its answers name, then the rows of
what none covers and of the device memory no leaf holds. `S` is the Site at the period's last head: the registration
only adds, so what an answer read at an earlier head of the period is in `S` unchanged. `digest` and `A` are the
registration's: the sheets' digest and the auditor, whose estimate compute's `bound` reading reads. No row where the
Site states no unit of time. Where a boundary link of the Site says no certifier splits it (`Site.mediated` is false),
bits leave by a way no answer counts, so every row of a link prints no amount, a certified one too, and says why.

A row of record reads a device's drift from the Site alone: the drift the auditor's host it is in states
(`Site.drift?`), and a device at the site, or in a host that states none, times nothing. `premised` lists drifts taken
on a premise, for auditor's hosts whose locations state none. A flow's amount with a series timed on a device in such a
host has, after its rows of record, the same rows with those series covered at the premised drifts; each names the
premises it rests on (`Row.premises`). The rows of record are as they are without the list, so a premised row adds to
none of them and replaces none. A premise for a host that is no auditor's host of the Site is refused, by the
premise's name: a premise is taken for a clock the registration holds, and no row is printed beside one for a clock it
does not. -/
def rows (S : Site) (digest : Sheet → String) (A : Site.Registration.Auditor) (period : Period)
    (accounts : List Account) (answers : List Answer) (premised : List Premised := []) : Except String (List Row) := do
  if let some q := premised.find? fun q => S.kind? q.host ≠ some .auditorHost then
    throw s!"premise {q.premise}: {q.host} is no auditor's host of the Site"
  let some tick := S.timeUnit | pure []
  let cs := readings period.heads tick accounts answers
  let groups : List (Amount × Option String) := (cs.map fun x => (x.1, x.2.place)).eraseDups
  let stated : String → Option Figure := S.drift?
  let taken : String → Option Premised := fun d =>
    if (stated d).isSome then none else (S.clock? d).bind fun h => premised.find? fun q => q.host == h
  let assumed : String → Option Figure := fun d => (stated d).orElse fun _ => (taken d).map fun q => q.drift
  let all : List Row :=
    (groups.flatMap fun g =>
        let rs : List Reading := (cs.filter fun x => x.1 = g.1 ∧ x.2.place = g.2).map fun x => x.2
        let names : List String :=
          ((rs.filter fun r => r.certified).filterMap fun r =>
            (period.origin r.series).bind fun o => (taken o.device).map fun q => q.premise).eraseDups
        g.1.rows S digest A tick period.ticks period.origin stated g.2 rs ++
          (if g.1.part.flow && !names.isEmpty then
            (g.1.rows S digest A tick period.ticks period.origin assumed g.2 rs).map fun r =>
              { r with premises := names }
          else [])) ++
      unaudited S digest A period.ticks cs ++ unregistered S digest
  pure <| if S.mediated then all
    else all.map fun r =>
      if r.part = .egress then { r with status := .unmediated, capacity := none, verified := none, free := none } else r

/-- A row as JSON, under the shared column names: `metric`, `verified`, `capacity`, `free` and `error`, with the part,
its unit, the place, the premises, the reading, the session, the count of answers, the standing and the terms not
priced. An amount with no number is `null`, and its row's `status` says why. -/
def Row.toJson (r : Row) : Json :=
  let cell (n : Option Nat) : Json := match n with
    | some n => Lean.toJson n
    | none => .null
  Json.mkObj [("capacity", cell r.capacity), ("error", .str r.error.toString), ("free", cell r.free),
    ("metric", .str r.metric), ("part", .str r.part.name), ("place", .str r.place),
    ("premises", Json.arr (r.premises.map Json.str).toArray), ("reading", .str r.reading),
    ("session", cell r.session), ("sessions", Lean.toJson r.sessions), ("status", .str r.status.name),
    ("unit", .str r.part.unit), ("unpriced", Json.arr (r.unpriced.map Json.str).toArray), ("verified", cell r.verified)]

/-! ## The example protocol's account, and the rows of one node -/

namespace Example

open Core.Registration.Example (sheet links auditor fig rated)

/-- **The example protocol's account** (`Core.Example.protocol`): the units it reports passed, each counted as one
tensor-core MAC of the site's devices, verified from below, with the error its profile states, once a period. It
states no span, no capacity and no registration's head, and a refusal publishes nothing. -/
def account : Account :=
  { protocol := Core.Example.protocol.id
    head := none
    amounts := [{ metric := "passed", part := .compute, key := .published "passed", unit := macs, side := .verified,
                  time := .unstated, error := .profile "delta", pool := .once,
                  uncertified := .refused, unpriced := ["the toy's units are not work"] }] }

/-- The rows, or none where `rows` refuses a premise. -/
private def rowsOf (S : Site) (digest : Sheet → String) (A : Site.Registration.Auditor) (period : Period)
    (accounts : List Account) (answers : List Answer) (premised : List Premised := []) : List Row :=
  (rows S digest A period accounts answers premised).toOption.getD []

/-- An auditor's host, at the site, whose clock may run `drift` from real time, or states none. -/
private def vm (id : String) (drift : Option Figure) : Site.Loc :=
  { id, kind := .auditorHost, parent := some "site0", owner := .auditor, key := "k2",
    scheme := Core.Registration.Example.lms, timebase := some "tb0", drift }

/-- The example's locations with its certifier in an auditor's host, `vm0`, which states `drift` for its clock. -/
private def hostedBy (drift : Option Figure) : List Site.Loc :=
  (Core.Registration.Example.locs.map fun x => if x.id == "cert0" then { x with parent := some "vm0" } else x) ++
    [vm "vm0" drift]

/-- The example's locations with a drift of `n / d` stated for its certifier's host. -/
private def drifting (n d : Nat) : List Site.Loc := hostedBy (some (fig n d))

/-- The example's locations, its certifier's host's clock taken at real time's rate. -/
private def locs : List Site.Loc := drifting 0 1

private def site : Site := ⟨[sheet], locs, links⟩
private def digest : Sheet → String := Site.Registration.sha512.sheet
private def tick : Figure := fig 1 1000000000
private def root : ByteArray := ⟨(List.replicate 32 (7 : UInt8)).toArray⟩
/-- The example protocol's accepted answer on two units: its check's public output and profile, as it writes them. -/
private def passed : Answer :=
  match Core.Example.check 2 root ⟨2, 2⟩ with
  | .accept pub prof =>
    { protocol := Core.Example.protocol.id, verdict := .accept, reason := none, digest := none,
      published := some (Core.Example.protocol.publicJson pub),
      profile := some (Core.Example.protocol.profileJson prof) }
  | _ =>
    { protocol := Core.Example.protocol.id, verdict := .reject, reason := none, digest := none, published := none,
      profile := none }

private def accept (protocol : String) (published profile : List (String × Json)) (head : String := "h") : Answer :=
  { protocol, verdict := .accept, reason := none, digest := none,
    published := some (Json.mkObj (("registration_head", .str head) :: published)),
    profile := some (Json.mkObj profile) }

/-- Toy accounts in the shapes a row takes: memory held, at a moment, from below; memory after a wipe, at a moment,
from above, with its own capacity and a flag; a link's bits, over a span named by its first step and its count, from
above, summed; work, over a span that is only counted, from below, with a share, pooled once; and work over a span
named by its first and last steps, from below. -/
private def held : Account :=
  { protocol := "toy/held", head := some (.published "registration_head")
    amounts := [{ metric := "held", part := .memory [.hbm], place := some (.profile "device"),
                  key := .published "s_bits", unit := bits, side := .verified, time := .moment,
                  error := .published "eps", uncertified := .flagged (.published "certified") }] }

private def wiped : Account :=
  { protocol := "toy/wiped", head := some (.published "registration_head")
    amounts := [{ metric := "after a wipe", part := .memory [.hbm, .dram, .disk], key := .published "foreign_bits",
                  unit := bits, side := .free, time := .moment, error := .published "eps",
                  capacity := some (.published "memory_bits"), uncertified := .flagged (.published "certified"),
                  unpriced := ["the keyed stream's PRF advantage"] }] }

private def sent : Account :=
  { protocol := "toy/sent", head := some (.published "registration_head")
    amounts := [{ metric := "sent", part := .egress, place := some (.profile "link"), key := .published "K",
                  unit := bits, side := .free,
                  time := .steps (.run (.published "start_ns") (.published "span_ns")) ⟨1, 1000000000⟩,
                  error := .published "eps", uncertified := .refused }] }

private def worked : Account :=
  { protocol := "toy/worked", head := some (.published "registration_head")
    amounts := [{ metric := "work", part := .compute, key := .published "work", unit := macs, side := .verified,
                  time := .span (.published "windows") ⟨1, 1000000⟩, share := some (.published "eps"),
                  error := .published "delta", pool := .once, uncertified := .safe }] }

private def ranged : Account :=
  { protocol := "toy/ranged", head := some (.published "registration_head")
    amounts := [{ metric := "audited", part := .compute, key := .published "work", unit := macs, side := .verified,
                  time := .steps (.range (.published "first") (.published "last")) ⟨1, 1000000⟩,
                  series := some (.profile "session"), error := .published "delta", uncertified := .safe }] }

/-- A read rate bounded from above over two tiers, one of them a host's, whose rate no sheet states. -/
private def spare : Account :=
  { protocol := "toy/spare", head := some (.published "registration_head")
    amounts := [{ metric := "unread", part := .read [.hbm, .dram], key := .published "K", unit := bits, side := .free,
                  time := .span (.published "span_ns") ⟨1, 1000000000⟩, error := .published "eps",
                  uncertified := .refused }] }

private def session (n : Nat) (eps : String) (certified : Bool := true) : Answer :=
  accept "toy/held" [("certified", .bool certified), ("eps", .str eps), ("s_bits", Lean.toJson n)]
    [("device", .str "gpu0")]

private def wipe (foreign memory : Nat) (certified : Bool := true) : Answer :=
  accept "toy/wiped" [("certified", .bool certified), ("eps", .str "1/1000"), ("foreign_bits", Lean.toJson foreign),
    ("memory_bits", Lean.toJson memory)] []

private def check (k ns : Nat) (eps : String) (head : String := "h") (start : Option Nat := some 0) : Answer :=
  accept "toy/sent" ([("K", Lean.toJson k), ("eps", .str eps), ("span_ns", Lean.toJson ns)] ++
    (start.map fun s => ("start_ns", Lean.toJson s)).toList) [("link", .str "up0")] head

/-- An audit of `work` MACs over the steps `span` names, first and last, numbered in `session`; over no named span
where it is none. -/
private def audit (work : Nat) (span : Option (Nat × Nat)) (session : String := "s0") : Answer :=
  accept "toy/ranged" ([("delta", .str "1/1000"), ("work", Lean.toJson work)] ++
    (span.map fun s => [("first", Lean.toJson s.1), ("last", Lean.toJson s.2)]).getD [])
    [("session", .str session)]

private def firstVerified (rs : List Row) : List (Option Nat × String) :=
  (rs.take 1).map fun r => (r.verified, r.error.toString)

private def unread (k ns : Nat) : Answer :=
  accept "toy/spare" [("K", Lean.toJson k), ("eps", .str "1/1000"), ("span_ns", Lean.toJson ns)] []

private def credit : Answer :=
  accept "toy/worked" [("delta", .str "1/100"), ("eps", Json.arr #[.str "1/4", .null]),
    ("windows", Json.arr #[Json.arr #[], Json.arr #[]]),
    ("work", Json.arr #[Lean.toJson (1000 : Nat), Lean.toJson (500 : Nat)])]
    []

/-- A period of `ticks` ticks at head `h`, every series' step 0 at its first tick unless `origin` says. -/
private def period (ticks : Nat := 1000) (heads : List String := ["h"])
    (origin : String → Option Origin := fun _ => some ⟨0, 0, "cert0"⟩) : Period := { ticks, heads, origin }

private def table (accounts : List Account) (answers : List Answer) (A : Site.Registration.Auditor := auditor)
    (ticks : Nat := 1000) : List Row :=
  rowsOf site digest A (period ticks) accounts answers

/-- The audits' rows over a period of 20,000 ticks, which holds twenty of their steps. -/
private def audited (answers : List Answer) : List Row := table [ranged] answers (ticks := 20000)

private def cells (rs : List Row) :
    List (String × String × String × Option Nat × Option Nat × Option Nat × String) :=
  rs.map fun r => (r.metric, r.place, r.reading, r.capacity, r.verified, r.free, r.error.toString)

/-- The example's site with a DRAM leaf under its host. -/
private def hosted : Site :=
  ⟨[sheet],
    locs ++ [{ id := "dram0", kind := .memory, parent := some "node0", tier := some .dram, bytes := 1 <<< 31 }],
    links⟩

private def registered (p : Part) (S : Site) : List (String × Nat) := p.capacity S digest auditor 1 (p.leaves S)

/-- The example's site with a DRAM leaf under its host that states its read rate, in bits a second. -/
private def hostRated : Site :=
  let leaf : Site.Loc :=
    { id := "dram0", kind := .memory, parent := some "node0", tier := some .dram, bytes := 1 <<< 31,
      bw := some 6554000000000 }
  ⟨[sheet], locs ++ [leaf], links⟩

/-- The example's site with a DRAM leaf and a disk leaf under its host, each stating its read rate. -/
private def cached : Site :=
  let dram : Site.Loc :=
    { id := "dram0", kind := .memory, parent := some "node0", tier := some .dram, bytes := 1 <<< 31,
      bw := some 6554000000000 }
  let disk : Site.Loc :=
    { id := "disk0", kind := .memory, parent := some "node0", tier := some .disk, bytes := 1 <<< 33,
      bw := some 2000000000000 }
  ⟨[sheet], locs ++ [dram, disk], links⟩

#guard registered (.read [.dram]) hostRated == [("registered", 6554)]
#guard registered (.read [.hbm, .dram]) hostRated == [("registered", 3200 + 6554)]
#guard registered (.read [.hbm]) hostRated == [("registered", 3200)]
-- a disk leaf's bytes can be returned from memory, so a part that holds a host's disk leaf counts the host's whole
-- DRAM tier's rate beside the disk's, once
#guard registered (.read [.dram]) cached == [("registered", 6554)]
#guard registered (.read [.disk]) cached == [(Part.READ_WITH_DRAM, 2000 + 6554)]
#guard registered (.read [.dram, .disk]) cached == [(Part.READ_WITH_DRAM, 6554 + 2000)]
#guard registered (.read [.hbm, .disk]) cached == [(Part.READ_WITH_DRAM, 3200 + 2000 + 6554)]
-- the row of a read part that holds a host's disk leaf says so in its reading
#guard ((rowsOf cached digest auditor period [] []).filter fun r => r.part = .read [.disk]).map
    (fun r => (r.reading, r.status, r.capacity)) ==
  [(Part.READ_WITH_DRAM, .notAudited, some ((2000 + 6554) * 1000))]
-- the device model's memory bandwidth a tick, from its sheet's bits a second at the Site's unit of time
#guard (site.device digest auditor "gpu0" 0 0 (fun (e : Empty) => nomatch e)).map (fun m => m.bw) == some 3200
-- a rate a second is rounded up once over the ticks it is read at: 1.31 Mbit/s over a millisecond of nanoseconds
#guard hostRated.over 1310000 1000000 == some 1310 ∧ hostRated.over 1310000 1 == some 1
-- the same link over a week of nanosecond ticks, covered but for one step of a thousandth of a second
#guard ticksOf 604799999 ⟨1, 1000⟩ (fig 1 1000000000) == 604799999000000
#guard hostRated.over 1310000 (604800 * 1000000000) == some 792288000000
#guard hostRated.over 1310000 (604800 * 1000000000 - ticksOf 604799999 ⟨1, 1000⟩ (fig 1 1000000000)) == some 1310
#guard ((rowsOf hostRated digest auditor period [spare] [unread 3000 400]).take 1).map
    (fun r => (r.status, r.capacity, r.verified, r.free)) ==
  [(.certified, some (9754 * 1000), some (9754 * 1000 - (3000 + 9754 * 600)), some (3000 + 9754 * 600))]

#guard account.Formed ∧ held.Formed ∧ wiped.Formed ∧ sent.Formed ∧ worked.Formed ∧ ranged.Formed
#guard account.amounts.all fun m => ¬ ({ m with time := .moment } : Amount).Formed
#guard account.amounts.all fun m => ¬ ({ m with part := .read [] } : Amount).Formed
-- a part over two tiers: their leaves' bytes summed; a leaf under a host has no registered read rate, so neither has a
-- part that holds one
#guard registered (.memory [.hbm, .dram]) hosted == [("registered", 8 * (2 ^ 30 + 2 ^ 31))]
#guard registered (.read [.hbm]) hosted == [("registered", 3200)]
#guard registered (.read [.hbm, .dram]) hosted == [] ∧ registered (.read [.dram]) hosted == []
#guard registered (.read [.disk]) hosted == [("registered", 0)]
-- a free amount with ticks no answer covers and no registered capacity to charge them at prints no amount; it prints
-- its amount, against no capacity, when its answers cover the period
#guard spare.Formed
#guard ((rowsOf hosted digest auditor period [spare] [unread 3000 400]).take 1).map
    (fun r => (r.status, r.capacity, r.verified, r.free)) == [(.notAudited, none, none, none)]
#guard ((rowsOf hosted digest auditor period [spare] [unread 3000 1000]).take 1).map
    (fun r => (r.status, r.capacity, r.verified, r.free)) == [(.certified, none, none, some 3000)]
-- the example protocol's own answer reads by its account: two units at its profile's error
#guard (account.amounts.map fun m => (m.read tick passed).map fun r => (r.amount, r.error.toString)) ==
  [some (2, "1/1000000")]
-- it names no registration, so it is held to none: nothing is certified, and every part of the site is charged, the
-- 63 GiB of the device's sheet that its one leaf does not hold among them
#guard cells (table [account] [passed]) ==
  [("not audited", "gpu0", "stated", some 4000000000, some 0, some 4000000000, "0"),
   ("not audited", "gpu0", "bound", some 4000000000, some 0, some 4000000000, "0"),
   ("not audited", "hbm0", "registered", some 8589934592, some 0, some 8589934592, "0"),
   ("not audited", "hbm0", "registered", some 3200000, some 0, some 3200000, "0"),
   ("not audited", "up0", "registered", some 100000, some 0, some 100000, "0"),
   ("not audited", "gpu0", "registered", some (8 * 63 * 2 ^ 30), some 0, some (8 * 63 * 2 ^ 30), "0")]
-- a device whose leaves hold all of its sheet's memory has no such row
#guard unregistered ⟨[{ sheet with mem := 1 <<< 30 }], locs.map (fun x =>
    if x.id = "gpu0" then { x with sheet := digest { sheet with mem := 1 <<< 30 } } else x), links⟩ digest == []
#guard (table [account] [passed]).all fun r => r.status = .notAudited
-- compute's capacity at each reading: the sheet's own figures, and the larger of those and the auditor's estimate
#guard ((table [] [] { auditor with sheet := fun _ => some (rated .zero 4200000000000000) }).take 2).map
    (fun r => (r.reading, r.capacity)) == [("stated", some 4000000000), ("bound", some 4200000000)]
-- an amount at a moment: a row for each session and one for the period, the least verified at the errors summed
#guard cells ((table [held] [session 6000000000 "1/100", session 5000000000 "1/50"]).take 3) ==
  [("held", "gpu0", "registered", some 8589934592, some 6000000000, some 2589934592, "1/100"),
   ("held", "gpu0", "registered", some 8589934592, some 5000000000, some 3589934592, "1/50"),
   ("held", "gpu0", "registered", some 8589934592, some 5000000000, some 3589934592, "3/100")]
#guard ((table [held] [session 6000000000 "1/100", session 5000000000 "1/50"]).take 3).map
    (fun r => (r.session, r.sessions)) == [(some 0, 1), (some 1, 1), (none, 2)]
-- sessions whose protocol says one answer pools the period's error state no error for the period, each still its own
#guard ((table [{ held with amounts := held.amounts.map fun m => { m with pool := .once } }]
    [session 6000000000 "1/100", session 5000000000 "1/50"]).take 3).map (fun r => r.status) ==
  [.certified, .certified, .noBound]
-- a session that certifies nothing verifies nothing, whatever amount it states
#guard cells ((table [held] [session 6000000000 "0" false]).take 1) ==
  [("held", "gpu0", "registered", some 8589934592, some 0, some 8589934592, "0")]
-- a free amount with its own capacity: the memory the wipe did not cover is free too, and a wipe of more memory than
-- is registered frees nothing
#guard cells ((table [wiped] [wipe 1000 8000000000]).take 1) ==
  [("after a wipe", "", "registered", some 8589934592, some 7999999000, some 589935592, "1/1000")]
#guard cells ((table [wiped] [wipe 1000 9000000000]).take 1) ==
  [("after a wipe", "", "registered", some 8589934592, some 0, some 8589934592, "1/1000")]
#guard ((table [wiped] [wipe 1000 8000000000]).take 1).map (fun r => r.unpriced) ==
  [["the keyed stream's PRF advantage"]]
-- a link: its checks' bits and errors summed, and the ticks no check covers charged at the link's capacity
#guard cells ((table [sent] [check 3000 400 "1/1000", check 2000 300 "1/1000" "h" (some 400)]).take 1) ==
  [("sent", "up0", "registered", some 100000, some 65000, some 35000, "1/500")]
-- a tick's cover counts once: two checks of one span cover it once, two that overlap cover the steps of both once,
-- and a check that names no span covers no tick
#guard cells ((table [sent] [check 3000 400 "1/1000", check 2000 400 "1/1000"]).take 1) ==
  [("sent", "up0", "registered", some 100000, some 35000, some 65000, "1/500")]
#guard cells ((table [sent] [check 3000 400 "1/1000", check 2000 300 "1/1000" "h" (some 200)]).take 1) ==
  [("sent", "up0", "registered", some 100000, some 45000, some 55000, "1/500")]
#guard cells ((table [sent] [check 3000 400 "1/1000", check 2000 300 "1/1000" "h" none]).take 1) ==
  [("sent", "up0", "registered", some 100000, some 35000, some 65000, "1/500")]
-- verified amounts add only over spans shown disjoint: two audits of one span, or of overlapping spans, count the
-- larger once; two of disjoint spans add; of three in a chain, the two ends add where they outweigh the middle
#guard firstVerified (audited [audit 700 (some (0, 9)), audit 500 (some (0, 9))]) == [(some 700, "1/500")]
#guard firstVerified (audited [audit 700 (some (0, 9)), audit 500 (some (5, 14))]) == [(some 700, "1/500")]
#guard firstVerified (audited [audit 700 (some (0, 9)), audit 500 (some (10, 19))]) == [(some 1200, "1/500")]
#guard firstVerified (audited [audit 700 (some (0, 9)), audit 900 (some (5, 14)), audit 500 (some (10, 19))])
  == [(some 1200, "3/1000")]
#guard firstVerified (audited [audit 700 (some (0, 9)), audit 1300 (some (5, 14)), audit 500 (some (10, 19))])
  == [(some 1300, "3/1000")]
-- an audit that names no span adds to none: beside one that names its span the larger counts, and of two the larger
#guard firstVerified (audited [audit 700 (some (0, 9)), audit 500 none]) == [(some 700, "1/500")]
#guard firstVerified (audited [audit 300 (some (0, 9)), audit 500 none]) == [(some 500, "1/500")]
#guard firstVerified (audited [audit 300 none, audit 500 none]) == [(some 500, "1/500")]
-- steps numbered in two sessions cannot be shown disjoint: the larger counts once
#guard firstVerified (audited [audit 700 (some (0, 9)), audit 500 (some (10, 19)) "s1"]) == [(some 700, "1/500")]
-- a span named by one step, an epoch; by its first and last steps; and one whose last step is before its first
#guard (Steps.index (.published "first")).read (audit 1 (some (7, 9))) == some (7, 8)
#guard (Steps.range (.published "first") (.published "last")).read (audit 1 (some (7, 9))) == some (7, 10)
#guard (Steps.range (.published "first") (.published "last")).read (audit 1 (some (9, 7))) == none
-- a check under another registration's head counts for nothing
#guard cells (table [sent] [check 3000 400 "1/1000" "g"]) == cells (table [] [])
-- errors that sum to 1 or more: no bound at this period
#guard ((table [sent] [check 3000 400 "1/2", check 2000 300 "1/2"]).take 1).map
    (fun r => (r.status, r.verified, r.free, r.error.toString)) == [(.noBound, none, none, "1")]
-- work: each window's share taken off, nothing where none is established, at the one answer's error
#guard cells ((table [worked] [credit]).take 2) ==
  [("work", "", "stated", some 4000000000, some 750, some 3999999250, "1/100"),
   ("work", "", "bound", some 4000000000, some 750, some 3999999250, "1/100")]
-- two answers that each pool a period's error state no error for the period
#guard ((table [worked] [credit, credit]).take 1).map (fun r => r.status) == [.noBound]
-- what the answers cover is not charged again
#guard cells ((table [held, sent, worked] [session 6000000000 "1/100", check 3000 1000 "1/1000", credit]).drop 5) ==
  [("not audited", "hbm0", "registered", some 3200000, some 0, some 3200000, "0"),
   ("not audited", "gpu0", "registered", some (8 * 63 * 2 ^ 30), some 0, some (8 * 63 * 2 ^ 30), "0")]
#guard ((table [sent] [check 3000 400 "1/1000"]).take 1).map (fun r => (r.toJson.compress)) ==
  ["{\"capacity\":100000,\"error\":\"1/1000\",\"free\":63000,\"metric\":\"sent\",\"part\":\"egress\"," ++
   "\"place\":\"up0\",\"premises\":[],\"reading\":\"registered\",\"session\":null,\"sessions\":1," ++
   "\"status\":\"certified\"," ++
   "\"unit\":\"bits\",\"unpriced\":[],\"verified\":37000}"]

/-- A read rate verified from below, in bytes, by an answer that names no span. -/
private def reads : Account :=
  { protocol := "toy/reads", head := some (.published "registration_head")
    amounts := [{ metric := "reads", part := .read [.hbm], key := .published "q_bytes", unit := bytes,
                  side := .verified, time := .unstated, error := .published "eps", uncertified := .safe }] }

private def read (q : Nat) : Answer := accept "toy/reads" [("eps", .str "1/1000"), ("q_bytes", Lean.toJson q)] []

-- reads verified from below: the bytes in bits, and the rest of the registered rate over the period is free; of two
-- answers that name no span the larger counts once
#guard reads.Formed
#guard cells ((table [reads] [read 1000]).take 1) ==
  [("reads", "", "registered", some 3200000, some 8000, some 3192000, "1/1000")]
#guard firstVerified (table [reads] [read 1000, read 700]) == [(some 8000, "1/500")]

/-- A link's bits whose check reads audits as given: its answer lists their names under `rests`. -/
private def resting : Account :=
  { sent with protocol := "toy/resting", rests := some ⟨.published "rests", ["toy/ranged"]⟩ }

private def rested (audits : List Answer) : Answer :=
  accept "toy/resting" [("K", Lean.toJson (3000 : Nat)), ("eps", .str "1/1000"), ("span_ns", Lean.toJson (400 : Nat)),
    ("start_ns", Lean.toJson (0 : Nat)),
    ("rests", Json.arr (audits.filterMap fun b => (nameOf b).map Json.str).toArray)] [("link", .str "up0")]

-- an answer that rests on an audit counts beside it; it counts for nothing without it, beside another audit, beside
-- the audit refused, or where it lists no names; one that rests on none counts
#guard cells ((table [resting] [rested [audit 300 none], audit 300 none]).take 1) ==
  [("sent", "up0", "registered", some 100000, some 37000, some 63000, "1/1000")]
#guard cells (table [resting] [rested [audit 300 none]]) == cells (table [] [])
#guard cells (table [resting] [rested [audit 300 none], audit 301 none]) == cells (table [] [])
#guard cells (table [resting] [rested [audit 300 none], { audit 300 none with verdict := .reject }]) ==
  cells (table [] [])
#guard cells (table [resting] [{ check 3000 400 "1/1000" with protocol := "toy/resting" }]) == cells (table [] [])
#guard cells ((table [resting] [rested []]).take 1) ==
  [("sent", "up0", "registered", some 100000, some 37000, some 63000, "1/1000")]
#guard (nameOf (audit 300 none)).map (·.length) == some 128

-- a span counts only where it lies in the period. Before it: no cover and no amount, the link charged at capacity
private def from_ (first : Nat) (at_ : Nat := 0) : Period :=
  period (origin := fun _ => some ⟨first, at_, "cert0"⟩)
private def linkRow (p : Period) (accounts : List Account) (answers : List Answer) :
    List (Status × Option Nat × Option Nat) :=
  ((rowsOf site digest auditor p accounts answers).take 1).map fun r => (r.status, r.capacity, r.free)
#guard linkRow period [sent] [check 3000 400 "1/1000"] == linkRow (from_ 0) [sent] [check 3000 400 "1/1000"]
#guard (linkRow (from_ 400) [sent] [check 3000 400 "1/1000"]).map (fun r => (r.1, r.2.1.isSome, r.2.2 == r.2.1)) ==
  [(.certified, true, true)]
-- across the period's start: only its steps in the period cover, and its whole amount adds
#guard (linkRow (from_ 300) [sent] [check 3000 400 "1/1000"]).map (·.2.2) ==
  (linkRow period [sent] [check 3000 100 "1/1000"]).map (·.2.2)
-- past the period's end: the same
#guard (linkRow (from_ 0 900) [sent] [check 3000 400 "1/1000"]).map (·.2.2) ==
  (linkRow period [sent] [check 3000 100 "1/1000"]).map (·.2.2)
-- a series with no origin covers nothing and adds nothing
#guard linkRow (period (origin := fun _ => none)) [sent] [check 3000 400 "1/1000"] ==
  linkRow (from_ 400) [sent] [check 3000 400 "1/1000"]
-- two series, each placed in the period: their cover adds where they share no tick, and a tick both cover counts once
private def keyed : Account :=
  { sent with protocol := "toy/keyed"
              amounts := sent.amounts.map fun m => { m with series := some (.published "key") } }
private def under (key : String) (k ns : Nat) : Answer :=
  accept "toy/keyed" [("K", Lean.toJson k), ("eps", .str "1/1000"), ("key", .str key), ("span_ns", Lean.toJson ns),
    ("start_ns", Lean.toJson (0 : Nat))] [("link", .str "up0")]
private def twoKeys (second : Nat) : Period :=
  period (origin := fun s =>
    if s == "\"k1\"" then some ⟨0, 0, "cert0"⟩
    else if s == "\"k2\"" then some ⟨0, second, "cert0"⟩ else none)
#guard (linkRow (twoKeys 500) [keyed] [under "k1" 3000 500, under "k2" 2000 500]).map (·.2.2) == [some 5000]
#guard (linkRow (twoKeys 400) [keyed] [under "k1" 3000 500, under "k2" 2000 500]).map (·.2.2) ==
  (linkRow period [sent] [check 5000 900 "1/1000"]).map (·.2.2)
-- an answer that names an earlier head of the period counts; one that names a head the period does not hold does not
#guard linkRow (period (heads := ["g", "h"])) [sent] [check 3000 400 "1/1000" "g"] ==
  linkRow period [sent] [check 3000 400 "1/1000"]
#guard linkRow (period (heads := ["h"])) [sent] [check 3000 400 "1/1000" "g"] == linkRow period [] []
-- the verified side: a span counts only whole in the period
#guard firstVerified (rowsOf site digest auditor (period 20000 (origin := fun _ => some ⟨5, 0, "cert0"⟩)) [ranged]
  [audit 700 (some (0, 9)), audit 500 (some (10, 19))]) == [(some 500, "1/500")]
#guard firstVerified (rowsOf site digest auditor (period 15000) [ranged]
  [audit 700 (some (0, 9)), audit 500 (some (10, 19))]) == [(some 700, "1/500")]

-- cover on a clock that may drift: none without a stated drift, the length less its share with one, rounded down
#guard ticksCredited 1000 ⟨1, 1000⟩ tick none == 0
#guard ticksCredited 1000 ⟨1, 1000⟩ tick (some (fig 3 5000)) == 999400000
#guard ticksCredited 1000 ⟨1, 1000⟩ tick (some (fig 0 1)) == ticksOf 1000 ⟨1, 1000⟩ tick
#guard ticksCredited 1 ⟨1, 1000⟩ tick (some (fig 1 3)) == 666666

-- a time charged on such a clock: none without a stated drift, the length over what the drift leaves, rounded up
#guard ticksCharged 1000 ⟨1, 1000⟩ tick none == none
#guard ticksCharged 1000 ⟨1, 1000⟩ tick (some (fig 3 5000)) == some 1000600361
#guard ticksCharged 1000 ⟨1, 1000⟩ tick (some (fig 0 1)) == some (ticksOf 1000 ⟨1, 1000⟩ tick)
#guard ticksCharged 1000 ⟨1, 1000⟩ tick (some (fig 1 1)) == none

-- the rows on a clock that may drift 1/1000. A span of 400 steps surely covers 399 ticks of 1000: 601 are charged
private def drifted (p : Period) (accounts : List Account) (answers : List Answer) (n : Nat := 1) (d : Nat := 1000) :
    List (Option Nat) :=
  ((rowsOf ⟨[sheet], drifting n d, links⟩ digest auditor p accounts answers).take 1).map fun r => r.free
#guard drifted period [sent] [check 3000 400 "1/1000"] == [some (3000 + 601 * 100)]
-- with no drift the rows are those of a clock at real time's rate
#guard drifted period [sent] [check 3000 400 "1/1000"] 0 1 ==
  (linkRow period [sent] [check 3000 400 "1/1000"]).map (·.2.2)
-- a device in a host that states no drift covers nothing: the link is charged at capacity, and its answer adds
-- nothing; a device at the site, in no host, the same
#guard ((rowsOf ⟨[sheet], hostedBy none, links⟩ digest auditor period [sent]
    [check 3000 400 "1/1000"]).take 1).map (fun r => (r.status, r.free == r.capacity)) == [(.certified, true)]
#guard ((rowsOf ⟨[sheet], Core.Registration.Example.locs, links⟩ digest auditor period [sent]
    [check 3000 400 "1/1000"]).take 1).map (fun r => (r.status, r.free == r.capacity)) == [(.certified, true)]
-- an origin that names a device the Site does not hold: the same
#guard ((rowsOf site digest auditor (period (origin := fun _ => some ⟨0, 0, "nobody"⟩)) [sent]
    [check 3000 400 "1/1000"]).take 1).map (fun r => (r.status, r.free == r.capacity)) == [(.certified, true)]
-- the period's end: of 1000 steps over a period of 1000 ticks only 999 surely end in it, and they cover 998 ticks
#guard drifted period [sent] [check 3000 1000 "1/1000"] == [some (3000 + 2 * 100)]
-- two series end to end: 499 and 498 ticks surely covered, less the one tick the two can share
#guard drifted (twoKeys 500) [keyed] [under "k1" 3000 500, under "k2" 2000 500] == [some (5000 + 4 * 100)]
-- one series in several spans loses its drift once, not once a span: 10 spans of 100 steps cover as one of 1000 does
private def spans (n len : Nat) : List Answer :=
  (List.range n).map fun i =>
    accept "toy/sent" [("K", Lean.toJson (10 : Nat)), ("eps", .str "1/1000"), ("span_ns", Lean.toJson len),
      ("start_ns", Lean.toJson (i * len))] [("link", .str "up0")]
#guard drifted (period 2000) [sent] (spans 10 100) == [some (100 + 1001 * 100)]
#guard drifted (period 2000) [sent] (spans 1 1000) == [some (10 + 1001 * 100)]
-- a Site that is not mediated: a boundary link says no certifier splits it, so every link's row prints no amount,
-- the certified link's too, and the rows of every other part are as they were
private def standin : Site :=
  ⟨[sheet], locs, links ++ [{ id := "eth0", ends := ("node0", Site.outside), β := some 400000000000, latency := 0,
                              unmediated := some "no certifier splits it" }]⟩
private def egress (S : Site) : List Row × List Row :=
  (rowsOf S digest auditor period [sent] [check 3000 400 "1/1000"]).partition fun r => r.part = .egress
#guard (egress standin).1.map (fun r => (r.place, r.status, r.capacity, r.verified, r.free)) ==
  [("up0", .unmediated, none, none, none), ("eth0", .unmediated, none, none, none)]
#guard (egress site).1.map (fun r => (r.place, r.status)) == [("up0", .certified)]
-- the same where the link states no capacity: the registration holds none for the part, and no row prints a number
private def unrated : Site :=
  ⟨[sheet], locs, links ++ [{ id := "eth0", ends := ("node0", Site.outside), β := none, latency := 0,
                              unmediated := some "no certifier splits it" }]⟩
#guard registered .egress unrated == [] ∧ (registered .egress standin).length == 1
#guard (egress unrated).1.map (fun r => (r.place, r.status, r.capacity, r.verified, r.free)) ==
  [("up0", .unmediated, none, none, none), ("eth0", .unmediated, none, none, none)]
#guard (egress unrated).2.map (fun r => (r.place, r.status, r.free)) ==
  (egress site).2.map (fun r => (r.place, r.status, r.free))
#guard (egress standin).2.map (fun r => (r.place, r.status, r.free)) ==
  (egress site).2.map (fun r => (r.place, r.status, r.free))
-- a drift taken on a premise. The example's certifier's host states no drift, so its row of record covers nothing:
-- the link is charged at capacity over the week. With 1/2000 premised for the host, the same row follows, covered,
-- and names the premise: of a week of steps of a thousandth of a second, 302,400 might end past the week, and what is
-- left covers its length less a 2000th; the 604.6488 seconds not covered are charged at capacity
private def week : Nat := 604800 * 1000000000
private def paced : Account :=
  { sent with amounts := sent.amounts.map fun m =>
      { m with time := .steps (.run (.published "start_ns") (.published "span_ns")) ⟨1, 1000⟩ } }
private def bare : Site := ⟨[sheet], hostedBy none, links⟩
private def onPremise : List Premised :=
  [{ host := "vm0", drift := fig 1 2000, premise := "ClockRateWithin(vm0, 5e-4)" }]
private def weekRows (S : Site) (premised : List Premised) : List Row :=
  (rowsOf S digest auditor (period week) [paced] [check 1000000000000000 604800000 "1/1000"] premised).filter
    fun r => r.metric = "sent"
#guard (weekRows bare []).map (fun r => (r.premises, r.capacity, r.verified, r.free)) ==
  [([], some 60480000000000000, some 0, some 60480000000000000)]
#guard (weekRows bare onPremise).map (fun r => (r.premises, r.capacity, r.verified, r.free)) ==
  [([], some 60480000000000000, some 0, some 60480000000000000),
   (["ClockRateWithin(vm0, 5e-4)"], some 60480000000000000, some 59419535120000000, some 1060464880000000)]
#guard ((weekRows bare onPremise).drop 1).map (fun r => r.toJson.compress) ==
  ["{\"capacity\":60480000000000000,\"error\":\"1/1000\",\"free\":1060464880000000,\"metric\":\"sent\"," ++
   "\"part\":\"egress\",\"place\":\"up0\",\"premises\":[\"ClockRateWithin(vm0, 5e-4)\"]," ++
   "\"reading\":\"registered\",\"session\":null,\"sessions\":1,\"status\":\"certified\"," ++
   "\"unit\":\"bits\",\"unpriced\":[],\"verified\":59419535120000000}"]
#guard stepsWithin week ⟨1, 1000⟩ tick (some (fig 1 2000)) == 604800000 - 302400
#guard ticksCredited (604800000 - 302400) ⟨1, 1000⟩ tick (some (fig 1 2000)) == week - 604648800000
-- the rows of record are the same rows with the list or without it, and every other row is as it was
#guard (rowsOf bare digest auditor (period week) [paced] [check 1000000000000000 604800000 "1/1000"] onPremise).filter
    (fun r => r.premises.isEmpty) ==
  rowsOf bare digest auditor (period week) [paced] [check 1000000000000000 604800000 "1/1000"]
-- a host whose location states a drift times its devices at it, of record, and no premise is read for it
#guard weekRows ⟨[sheet], drifting 1 2000, links⟩ onPremise ==
  ((weekRows bare onPremise).drop 1).map fun r => { r with premises := [] }
-- a premise for a host none of whose devices times a series of the amount adds no row
private def twoHosts : Site := ⟨[sheet], hostedBy none ++ [vm "vm1" none], links⟩
private def other : List Premised := [{ host := "vm1", drift := fig 1 2000, premise := "ClockRateWithin(vm1, 5e-4)" }]
#guard weekRows twoHosts other == weekRows bare [] ∧ weekRows twoHosts (other ++ onPremise) == weekRows bare onPremise
-- a premise for a host that is no auditor's host of the Site is refused by name, and no row is printed: one the Site
-- does not hold, the auditor's device, the prover's host; a certifier at the site is in no host, so no premise times
-- it and its row of record covers nothing
private def refused (S : Site) (host : String) : Option String :=
  match rows S digest auditor (period week) [paced] [check 1000000000000000 604800000 "1/1000"]
      [{ host, drift := fig 1 2000, premise := "ClockRateWithin(vm0, 5e-4)" }] with
  | .error e => some e
  | .ok _ => none
#guard refused bare "vm0" == none
#guard refused bare "vm1" == some "premise ClockRateWithin(vm0, 5e-4): vm1 is no auditor's host of the Site"
#guard (refused bare "cert0").isSome ∧ (refused bare "node0").isSome
private def atSite : Site := ⟨[sheet], Core.Registration.Example.locs, links⟩
#guard (refused atSite "vm0").isSome ∧ (refused atSite "cert0").isSome
#guard (weekRows atSite []).map (fun r => (r.premises, r.capacity, r.verified, r.free)) ==
  [([], some 60480000000000000, some 0, some 60480000000000000)]
-- what a drift leaves of ticks counted, and the steps that surely end in a period
#guard ticksKept 1000 none == 0
#guard ticksKept 1000 (some (fig 1 1000)) == 999
#guard stepsWithin 1000 ⟨1, 1000000000⟩ tick (some (fig 1 1000)) == 999
#guard stepsWithin 1000 ⟨1, 1000000000⟩ tick none == 0

end Example

end Account

end Core
