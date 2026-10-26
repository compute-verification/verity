import CertifierDevice.Frame
import CertifierDevice.Shaper

/-!
# The active network certifier as a pure state machine

`step : State → Event → State × List Action` is `active.py`'s `Proxy` on one boundary link, with its I/O taken out: an
`Event` is one thing the world did (a ticker woke, the link or a server connection gave bytes, a connection opened, was
refused, ended, or finished writing a request), carrying the clock's reading `now` (ns) when it happened, and the
`Action`s are what the proxy does in reply, in its order. Everything that decides a grid is the reference's: the
`Shaper` (egress, at r) and the `Bucketer` (ingress, at r_in) over `grid.Certifier`, eligibility and Σ_sync from
`NetworkCertifier`, and the online `Prover` where one is configured. `active.py`'s docstring is the spec; this is its
proxy event by event:

* **Ticks.** Tick t of a grid is due at `origin + t·τ`, and `windows · T` of them run, after which a grid's ticks are
  ignored. A tick read more than `slack` after its due time faults its certifier `monitor-unavailable`. An egress tick
  reads the link's backlog (`Event.tick`'s `backlog`): over a bucket's bytes it is `delivery-refused`, and over four the
  bucket isn't written at all. It then checks forwarding, ticks the shaper, writes the row and r − n padding frames,
  starts window w's online choice if it is tick w·T − lag, and publishes the window's record at its last bucket. An
  ingress tick ticks the bucketer, cuts each released `DATA` frame's payload into requests by their lengths
  (`Requests.feed`: the k-th request on the link is stream k) and hands its bytes to their stream (opening the stream's
  server connection at its first frame), records the stream's anchor at its request's last frame, and publishes the
  record at the window's last bucket.
* **Streams.** A stream's connection is `connect`ed at its first released frame. Until it opens (`connected`), its
  bytes wait; then each frame's are written in order (`writeUpstream`), and the request's last written waits for its
  drain (`drain := true`, answered by `delivered`), which is when the request reached the server. A refused connection
  is `delivery-refused` on ingress. The server's bytes are cut into events, each read at time s ready at bucket
  ⌊(s − origin)/τ_b⌋ + 1, and an event's frame is what follows its `id:` line (`eventBody`) on `canonicalStream`. An
  event without a position, whose body is too long for a frame, or on a stream with no released request is
  `malformed`, as are bytes left unframed at the stream's end. A frame the shaper hands at a bucket after its `h`
  (ready after it, or read after that bucket's tick) is late, and faults the window it is handed in
  `coverage-never-established`.
* **Late forwards.** At each egress tick, a released request not delivered by `origin + min(t·τ_in + slack, (a + 1)·τ_b)`
  faults its ingress window (if still open) and every egress window until its stream has ended and every frame read
  before the end is handed.
* **Order.** Within one event, actions are in the order the proxy's tasks take them: an ingress tick's `publish` comes
  before the connections and writes its frames start, which come in the order of the frames that started them (a new
  stream's `connect`, or the writes of a stream whose connection was waiting for a frame).
* **Online choices.** The online prover (`LaggedChoices`) chooses window w's sync right after egress tick w·T − lag
  (window 0's before the first tick, in `init`), on the frames ready by then, and the shaper declares it at tick w·T.
  Every choice is ready by its tick, as on the proxy's `InlineExecutor`.
* **Refusal.** The online prover refuses a choice that reads a frame ready no later than its anchor's bucket
  (`online.Refused`, which a request released by an ingress tick read before its due time can lead to): the egress
  window that declares it is faulted `monitor-unavailable` at its first tick, as a late tick is, and its sync is the
  choice made on the frames read.
* **The salt** (`Config.salt`, off by default). At PoUW window p's first ingress tick (its first certifier window of
  `pouwWindow`) the bucketer enqueues the certifier's own frame, `saltFrame p placeholder`, at the head of the queue,
  ahead of that bucket's arrivals and of any queue carried over, so it leaves in that bucket's slot 0. When the ingress
  certifier releases it, the proxy queues p's salt
  (`values[p]`, the certifier's coins, drawn outside the device) in that frame in place of the placeholder for a server
  connection of its own: the first opens it (`connectSalt`, answered `saltConnected` or `saltRefused`, which is
  `delivery-refused` on ingress), and each frame is written (`writeSalt`) once the one before it is drained
  (`saltDelivered`). Nothing is read from it. The reveal and its custody are outside the device.
* **The restart** (`initAt c start position`): a proxy back from maintenance, its first tick of each grid window
  `start`'s first bucket and the first egress frame it takes the Program's at `position`, carrying nothing from before.
* **A served link** (`Config.served`, off by default): the bucketer enqueues arrived frames only as whole requests of
  the profile's F_in frames that the admission rule admits (`Served.admits`) under the window's declared sync; the rest
  wait, in order, for the next window.
* **Ordinals and compact records** (`Config.ordinals`, `Config.compact`, each off by default). Under `ordinals` a
  stream's server connection opens with `ordinalHeader s`, written as it opens, before its request's bytes. Under
  `compact` (which takes `ordinals`, and egress frames of at least `compactRecordSize` bytes) the server's bytes are an
  HTTP response head, then (position, record) units (`CompactUnits.feed`), and each unit's frame is its record
  zero-filled to B (`compactCell`) at its position: a record that isn't one (`compactOrdinal`) or names another request
  than its stream's is `malformed`, as are bytes left uncut at the stream's end, an unfinished head included.
* **Ticks due together.** `step` takes one tick at a time, so a world whose ticks of both grids fall due at one
  boundary orders them; the proxy runs the ingress tick first (`active.TICK_ORDER`), as `Driver.run` does.

Events the proxy couldn't be waiting for (a second `connected`, bytes on a stream not open, a `delivered` with no write
waiting, anything after both grids' last ticks) change nothing. Python's floats and Lean's integers agree away from the
boundaries; the difftest keeps every reading at least 10 µs from one.
-/

namespace CertifierDevice

open NetworkCertifier (ClockSync SyncSet Reason)

/-- Which grid ticks. -/
inductive GridKind where
  | egress
  | ingress
  deriving BEq, DecidableEq, Repr

/-- One thing the world did, with the clock's reading `now` (ns) when it did it. -/
inductive Event where
  /-- The grid's ticker woke; `backlog` is the link's unwritten bytes then (read on egress only). -/
  | tick (grid : GridKind) (now backlog : Nat)
  /-- The link gave bytes; none is its end. -/
  | link (now : Nat) (bytes : ByteArray)
  /-- Stream `s`'s server connection opened. -/
  | connected (s now : Nat)
  /-- Stream `s`'s server refused the connection. -/
  | refused (s now : Nat)
  /-- Stream `s`'s server gave bytes; none is its end. -/
  | upstream (s now : Nat) (bytes : ByteArray)
  /-- The write to stream `s` that asked for a drain is drained: the request reached the server. -/
  | delivered (s now : Nat)
  /-- The salt connection opened. -/
  | saltConnected (now : Nat)
  /-- The server refused the salt connection. -/
  | saltRefused (now : Nat)
  /-- The salt connection's write is drained: the salt reached the server. -/
  | saltDelivered (now : Nat)

/-- What the proxy does. -/
inductive Action where
  /-- Write these bytes to the link (one bucket: its row's frames, then padding). -/
  | writeLink (bytes : ByteArray)
  /-- Open stream `s`'s server connection; the world answers `connected` or `refused`. -/
  | connect (s : Nat)
  /-- Write to stream `s`'s server connection; with `drain`, the world answers `delivered` once it is written. -/
  | writeUpstream (s : Nat) (bytes : ByteArray) (drain : Bool)
  /-- A window's record closed, with its grid's parameters. -/
  | publish (record : Record) (params : Params)
  /-- Open the salt connection to the server; the world answers `saltConnected` or `saltRefused`. -/
  | connectSalt
  /-- Write a salt frame on the salt connection; the world answers `saltDelivered` once it is written. -/
  | writeSalt (bytes : ByteArray)

/-- The online prover's configuration: its rule, envelope, lag (1 … T − 1 buckets) and the history of an earlier run. -/
structure ProverConfig where
  rule : Rule
  envelope : ClockSync
  lag : Nat
  prior : Array Past := #[]

/-- The proxy's salt (`Proxy(salt=…)`): one salt of `bits` bits per PoUW window of `pouwWindow` certifier windows, PoUW
window p's `values[p]`. They are the certifier's own coins (`active.py` draws p's at p's first ingress tick, the driver
all of them before its first tick); the device reads p's only once p's salt frame is released. -/
structure SaltSource where
  values : Array ByteArray
  bits : Nat := 512
  pouwWindow : Nat := 1

/-- `_draw_salt`'s frame: at a PoUW window's first certifier window `w`, the certifier's own frame of `B` bytes, the
placeholder's. -/
def SaltSource.first (s : SaltSource) (B w : Nat) : Option ByteArray :=
  if w % s.pouwWindow = 0 then saltFrame (w / s.pouwWindow) placeholder B else none

/-- `Proxy`'s parameters: both grids (windows of the same length, frames of at least a header), Σ_sync, the declared
syncs or else the online prover, how many windows run, the clock's time of bucket 0 and the slack, in ns, and the salt
and a served link's profile, each off by default. -/
structure Config where
  egress : Params
  ingress : Params
  syncSet : SyncSet
  syncs : Array ClockSync
  windows : Nat
  origin : Nat
  slack : Nat
  prover : Option ProverConfig := none
  egressLink : String := "responses"
  ingressLink : String := "requests"
  salt : Option SaltSource := none
  served : Option ServedProfile := none
  ordinals : Bool := false
  compact : Bool := false

/-- The slowest of Σ_sync: its largest δ and smallest ρ. -/
def Config.slowest (c : Config) : ClockSync :=
  let d := c.syncSet.deltas.foldl max 0
  let ρ := c.syncSet.rhos.foldl min (c.syncSet.rhos.headD 1)
  if h : 0 < ρ then ⟨d, ρ, h⟩ else ⟨d, 1, by decide⟩

/-- A served link's admission rule (`served.Admission`): at the egress allocation, under each window's declared sync,
or the slowest of Σ_sync under an online prover or past the declared syncs. -/
def Config.admission (c : Config) (p : ServedProfile) : Admission :=
  { profile := p, allocation := c.egress.allocation, syncs := if c.prover.isSome then #[] else c.syncs,
    slowest := c.slowest }

/-- `Proxy.__init__`'s checks, and what the frames need, for a proxy that starts at window `start`: positive parameters,
frames of at least a header (compact egress frames of at least a record, under `ordinals`), the two grids' windows of
one length, an online prover declaring every window's sync, with
a lag of 1 … T − 1 buckets, from window 0 on; a salt frame that fits an ingress frame, a salt of a positive whole number
of bytes for every PoUW window of the run, of a positive number of certifier windows; and a served link's profile, of
grids that share T and τ_b, counting the salt frame exactly when the salt is on (then at every window), whose window at
`start` holds its own frames, one request and its response under the admission's sync. -/
def Config.check (c : Config) (start : Nat := 0) : Except String Unit := do
  for p in [c.egress, c.ingress] do
    if [p.buckets, p.allocation, p.frameBytes, p.bucketMs, p.queueCapacity].any (· = 0) then
      throw "grid parameters must be positive"
  if c.compact then
    if !c.ordinals || c.egress.frameBytes < compactRecordSize then
      throw s!"compact records name their request by its ingress ordinal (ordinals) and fill egress frames of at \
        least {compactRecordSize} bytes"
  else if c.egress.frameBytes < headerSize then throw s!"a frame holds its {headerSize}-byte header"
  if c.ingress.frameBytes < headerSize then throw s!"a frame holds its {headerSize}-byte header"
  if c.egress.buckets * c.egress.bucketMs ≠ c.ingress.buckets * c.ingress.bucketMs then
    throw "the egress and ingress grids have windows of the same length"
  if let some p := c.prover then
    if !c.syncs.isEmpty then throw "an online prover declares every window's sync"
    if p.lag = 0 || p.lag ≥ c.egress.buckets then
      throw "the proxy's online prover chooses off its tick: a lag of 1 … T − 1 buckets"
  if let some s := c.served then
    if c.egress.buckets ≠ c.ingress.buckets || c.egress.bucketMs ≠ c.ingress.bucketMs then
      throw "a served link's grids share T and τ_b: its anchors are ingress buckets"
    if s.own ≠ (if c.salt.isSome then 1 else 0) || c.salt.any (·.pouwWindow ≠ 1) then
      throw "a served link's profile counts the salt frame (own 1) exactly when the salt is on, and then every window \
        has one (a PoUW window of one certifier window)"
  if let some s := c.salt then
    if c.ingress.frameBytes < saltFrameBytes then
      throw s!"a salt frame needs ingress frames of at least {saltFrameBytes} bytes"
    if s.bits = 0 || s.bits % 8 ≠ 0 then throw "a salt is a positive whole number of bytes"
    if s.pouwWindow = 0 then throw "a PoUW window is a positive number of certifier windows"
    if s.values.any (·.size ≠ s.bits / 8) || s.values.size * s.pouwWindow < c.windows then
      throw "the salt source gives every PoUW window of the run a salt of its bits"
  if start > c.windows then throw s!"a proxy starts at one of windows 0 … {c.windows}"
  if start > 0 && c.prover.isSome then
    throw "an online prover declares from window 0 on: a proxy back from maintenance takes syncs"
  if let some s := c.served then
    s.check
    if !admits #[] (s.own + s.requestFrames) ((c.admission s).sync start) s c.egress.allocation c.ingress.allocation
        c.ingress.buckets then
      throw "a served link's window holds its own frames, one request and its response under the admission's sync"

/-- Where a stream's server connection is. -/
inductive Phase where
  | connecting
  | open
  | closed
  deriving BEq, DecidableEq

/-- `active._Stream`: its request's anchor, ingress window and forwarding deadline, its frames read so far, the frames
waiting to be written, its event or unit cutter, and when its request was delivered and its ready bucket at its end. -/
structure Stream where
  phase : Phase := .connecting
  anchor : Option Nat := none
  count : Nat := 0
  out : Array (ByteArray × Bool) := #[]
  draining : Bool := false
  events : Events := {}
  units : CompactUnits := {}
  window : Option Nat := none
  deadline : Option Nat := none
  delivered : Option Nat := none
  ended : Option Int := none

/-- `_salt_upstream`'s connection: the salt frames waiting to be written, and whether a write waits for its drain. It is
written, never read. -/
structure SaltConn where
  phase : Phase := .connecting
  out : Array ByteArray := #[]
  draining : Bool := false

/-- The proxy's state. `frames` logs every egress frame read as (position, stream, index, anchor, ready). -/
structure State where
  config : Config
  shaper : Shaper
  bucketer : Bucketer
  streams : Std.HashMap Nat Stream := {}
  requests : Requests := {}
  linkOpen : Bool := true
  linkBuffer : ByteArray := ByteArray.empty
  released : Array Nat := #[]
  late : Array Nat := #[]
  frames : Array (Nat × Nat × Nat × Nat × Int) := #[]
  linkFramesRead : Nat := 0
  requestsReleased : Nat := 0
  saltConn : Option SaltConn := none
  saltsReleased : Nat := 0
  saltsDelivered : Nat := 0

def nsPerMs : Nat := 1000000

/-- The state before the first tick, window 0's choice started if any window runs. -/
def init (c : Config) : State :=
  let prover := c.prover.map fun p =>
    let l : LaggedChoices := { prover := Prover.make c.syncSet c.egress.buckets p.rule p.envelope p.prior p.lag }
    if c.windows > 0 then l.start else l
  { config := c,
    shaper := Shaper.make { link := c.egressLink, params := c.egress } c.syncs c.syncSet prover c.windows,
    bucketer := { certifier := { link := c.ingressLink, params := c.ingress },
                  first := match c.salt with
                    | some s => s.first c.ingress.frameBytes
                    | none => fun _ => none,
                  served := c.served.map c.admission } }

/-- The state before window `start`'s first tick, the first egress frame it takes the one at `position`: a proxy back
from maintenance (`Proxy(start=…, position=…)`), which carries nothing from before and has no online prover. -/
def initAt (c : Config) (start position : Nat) : State :=
  let st := init c
  if start = 0 then { st with shaper := { st.shaper with next := position } } else
  let shaper := Shaper.make { link := c.egressLink, params := c.egress, window := start } c.syncs c.syncSet none c.windows
  let certifier := { st.bucketer.certifier with window := start }
  { st with shaper := { shaper with t := start * c.egress.buckets, next := position },
            bucketer := { st.bucketer with certifier, t := start * c.ingress.buckets } }

namespace State

def ticks (st : State) (p : Params) : Nat := st.config.windows * p.buckets

/-- Both tickers have run all their ticks: the proxy has closed. -/
def done (st : State) : Bool :=
  st.shaper.t ≥ st.ticks st.config.egress && st.bucketer.t ≥ st.ticks st.config.ingress

/-- `_bucket_ready`: a frame read at `now` is ready at the first egress bucket that starts after it. -/
def bucketReady (st : State) (now : Nat) : Int :=
  ((now : Int) - st.config.origin) / (st.config.egress.bucketMs * nsPerMs : Nat) + 1

/-- The ingress bucket a link frame read at `now` is enqueued at: ⌈(now − origin)/τ_in⌉. -/
def bucketIn (st : State) (now : Nat) : Int :=
  -((-((now : Int) - st.config.origin)) / (st.config.ingress.bucketMs * nsPerMs : Nat))

/-- Whether a tick of a grid of bucket length `ms`, its t-th, read at `now` is later than the slack. -/
def isLate (st : State) (ms t now : Nat) : Bool :=
  (now : Int) - (st.config.origin + t * ms * nsPerMs : Nat) > st.config.slack

def stream (st : State) (s : Nat) : Stream := st.streams.getD s {}

def setStream (st : State) (s : Nat) (x : Stream) : State := { st with streams := st.streams.insert s x }

/-- A stream's pump: if its connection is open and no write waits for its drain, write what waits, in order, up to and
including its request's last frame while the request is undelivered, which then waits for its drain. -/
def pump (st : State) (s : Nat) : State × Array Action := Id.run do
  let mut x := st.stream s
  if x.phase != .open || x.draining then return (st, #[])
  let mut acts := #[]
  let mut taken := 0
  for (payload, last) in x.out do
    let drain := last && x.delivered.isNone
    acts := acts.push (.writeUpstream s payload drain)
    taken := taken + 1
    if drain then
      x := { x with draining := true }
      break
  x := { x with out := x.out.extract taken x.out.size }
  return (st.setStream s x, acts)

/-- `_settled`: the stream has ended upstream, and every frame read before its end is handed. -/
def settled (st : State) (s : Nat) : Bool :=
  match (st.stream s).ended with
  | none => false
  | some e => (st.shaper.t : Int) ≥ e && st.shaper.pending.toList.all fun (_, _, ready) => ready > e

/-- `_check_forwarding`. -/
def checkForwarding (st : State) (now : Nat) : State := Id.run do
  let mut st := st
  let mut waiting := #[]
  for s in st.released do
    let x := st.stream s
    let deadline := x.deadline.getD 0
    if x.delivered.isNone && now < deadline then waiting := waiting.push s
    else if x.delivered.isNone || x.delivered.getD 0 > deadline then
      if some st.bucketer.certifier.window == x.window then
        st := { st with bucketer := { st.bucketer with certifier := st.bucketer.certifier.doFault .deliveryRefused } }
      st := { st with late := st.late.push s }
  st := { st with released := waiting }
  st := { st with late := st.late.filter fun s => !st.settled s }
  if !st.late.isEmpty then st := { st with shaper := st.shaper.fault .deliveryRefused }
  return st

/-- `Proxy._egress`'s body at one tick: after tick w·T − lag it starts window w's choice. -/
def egressTick (st : State) (now backlog : Nat) : State × Array Action := Id.run do
  let p := st.config.egress
  let t := st.shaper.t
  let mut st := st
  if st.isLate p.bucketMs st.shaper.t now then st := { st with shaper := st.shaper.fault .monitorUnavailable }
  let bucketBytes := p.allocation * p.frameBytes
  if backlog > bucketBytes then st := { st with shaper := st.shaper.fault .deliveryRefused }
  st := st.checkForwarding now
  let (shaper, row, record) := st.shaper.tick
  st := { st with shaper }
  let mut acts := #[]
  if backlog ≤ 4 * bucketBytes then
    let n := rowCount row
    let bytes := (row.extract 0 n).foldl (fun acc f => acc ++ f.getD ByteArray.empty) ByteArray.empty
    acts := acts.push (.writeLink (bytes ++ ⟨Array.replicate ((p.allocation - n) * p.frameBytes) 0⟩))
  if let some l := st.shaper.prover then
    if (t + l.prover.lag) % p.buckets = 0 && (t + l.prover.lag) / p.buckets < st.config.windows then
      st := { st with shaper := { st.shaper with prover := some l.start } }
  if let some r := record then acts := acts.push (.publish r p)
  return (st, acts)

/-- What an ingress tick sets going, in the order its tasks run. -/
inductive Started where
  | connect (s : Nat)
  | pump (s : Nat)
  | saltConnect
  | saltPump

/-- The salt connection's writer: if it is open and no write waits for its drain, write the next salt frame, which
waits for its drain. -/
def saltPump (st : State) : State × Array Action :=
  match st.saltConn with
  | some x =>
    if x.phase == .open && !x.draining && 0 < x.out.size then
      ({ st with saltConn := some { x with out := x.out.extract 1 x.out.size, draining := true } },
        #[.writeSalt x.out[0]!])
    else (st, #[])
  | none => (st, #[])

/-- `_release_salt`: PoUW window p's salt frame queued for the server, its own frame being released; the first starts
the salt connection, and one waiting for a frame starts writing. -/
def releaseSalt (st : State) (p : Nat) : State × Option Started :=
  let value := (st.config.salt.bind (·.values[p]?)).getD ByteArray.empty
  let frame := (saltFrame p value (headerSize + 8 + value.size)).getD ByteArray.empty
  let st := { st with saltsReleased := st.saltsReleased + 1 }
  match st.saltConn with
  | none => ({ st with saltConn := some { out := #[frame] } }, some .saltConnect)
  | some x =>
    ({ st with saltConn := some { x with out := x.out.push frame } },
      if x.phase == .open && !x.draining && x.out.isEmpty then some .saltPump else none)

/-- `Proxy._ingress`'s body at one tick. -/
def ingressTick (st : State) (now : Nat) : State × Array Action := Id.run do
  let p := st.config.ingress
  let e := st.config.egress
  let mut st := st
  let t := st.bucketer.t
  if st.isLate p.bucketMs t now then
    st := { st with bucketer := { st.bucketer with certifier := st.bucketer.certifier.doFault .monitorUnavailable } }
  let (bucketer, row, record) := st.bucketer.tick
  st := { st with bucketer }
  let anchor := t * p.bucketMs / e.bucketMs
  let mut started := #[]
  for (slot, i) in (row.extract 0 (rowCount row)).zipIdx do
    if let some (_, w) := bucketer.own.find? (·.1 == i) then
      let salt := st.releaseSalt (w / (st.config.salt.map (·.pouwWindow)).getD 1)
      st := salt.1
      if let some x := salt.2 then started := started.push x
      continue
    let (kind, _, payload) := decodeFrame (slot.getD ByteArray.empty)
    if kind != kindData then continue
    let (requests, s, data, last) := st.requests.feed payload
    st := { st with requests }
    if !st.streams.contains s then
      st := st.setStream s {}
      started := started.push (Started.connect s)
    let mut x := st.stream s
    if x.phase == .open && !x.draining && x.out.isEmpty then started := started.push (.pump s)
    x := { x with out := x.out.push (data, last) }
    if last && x.anchor.isNone then
      x := { x with anchor := some anchor, window := some (t / p.buckets),
                    deadline := some (st.config.origin +
                      min (t * p.bucketMs * nsPerMs + st.config.slack) ((anchor + 1) * e.bucketMs * nsPerMs)) }
      st := { st with released := st.released.push s, requestsReleased := st.requestsReleased + 1 }
    st := st.setStream s x
  let mut acts := #[]
  if let some r := record then acts := acts.push (.publish r p)
  for x in started do
    match x with
    | .connect s => acts := acts.push (.connect s)
    | .pump s =>
      let (st', more) := st.pump s
      st := st'
      acts := acts ++ more
    | .saltConnect => acts := acts.push .connectSalt
    | .saltPump =>
      let (st', more) := st.saltPump
      st := st'
      acts := acts ++ more
  return (st, acts)

/-- `_read_link` at one read: whole frames are enqueued at the ingress bucket of the read; the rest waits. -/
def linkRead (st : State) (now : Nat) (data : ByteArray) : State := Id.run do
  if data.isEmpty then return { st with linkOpen := false }
  let size := st.config.ingress.frameBytes
  let bucket := st.bucketIn now
  let buffer := st.linkBuffer ++ data
  let whole := buffer.size - buffer.size % size
  let mut b := st.bucketer
  for i in [0:whole / size] do
    b := b.arrive (buffer.extract (i * size) ((i + 1) * size)) bucket
  return { st with bucketer := b, linkBuffer := buffer.extract whole buffer.size,
                   linkFramesRead := st.linkFramesRead + whole / size }

/-- `_egress_frame`: a frame at its position, with its stream's anchor and its index there; malformed on a stream with
no released request. -/
def egressFrame (st : State) (s position : Nat) (bytes : ByteArray) (ready : Int) : State :=
  let x := st.stream s
  match x.anchor with
  | some anchor =>
    let st := st.setStream s { x with count := x.count + 1 }
    { st with frames := st.frames.push (position, s, x.count, anchor, ready),
              shaper := st.shaper.arrive position ⟨bytes, anchor, x.count⟩ ready }
  | none => { st with shaper := st.shaper.fault .malformed }

/-- `_egress_event`: an event at its position, its body framed on `canonicalStream`. -/
def egressEvent (st : State) (s : Nat) (event : ByteArray) (ready : Int) : State :=
  match eventPosition event, encodeFrame kindData canonicalStream (eventBody event) st.config.egress.frameBytes with
  | some position, some bytes => st.egressFrame s position bytes ready
  | _, _ => { st with shaper := st.shaper.fault .malformed }

/-- `_egress_record`: a compact record at its position, zero-filled to a frame; one that isn't a record or names another
request than stream `s` is malformed. -/
def egressRecord (st : State) (s position : Nat) (record : ByteArray) (ready : Int) : State :=
  match compactOrdinal record == some s, compactCell record st.config.egress.frameBytes with
  | true, some bytes => st.egressFrame s position bytes ready
  | _, _ => { st with shaper := st.shaper.fault .malformed }

/-- The end of stream `s`'s server connection: bytes left uncut are malformed; the stream's end is its ready bucket. -/
def upstreamEnd (st : State) (s : Nat) (now : Nat) : State :=
  let x := st.stream s
  let st := if x.events.buffer.isEmpty && x.units.buffer.isEmpty then st
    else { st with shaper := st.shaper.fault .malformed }
  st.setStream s { x with phase := .closed, draining := false, ended := some (st.bucketReady now) }

/-- One read of stream `s`'s server connection: cut into events, or into compact units under `compact`. -/
def upstreamRead (st : State) (s : Nat) (now : Nat) (data : ByteArray) : State := Id.run do
  if data.isEmpty then return st.upstreamEnd s now
  let ready := st.bucketReady now
  let x := st.stream s
  if st.config.compact then
    let (units, cut) := x.units.feed data
    let mut st := st.setStream s { x with units }
    for (position, record) in cut do st := st.egressRecord s position record ready
    return st
  let (events, cut) := x.events.feed data
  let mut st := st.setStream s { x with events }
  for event in cut do st := st.egressEvent s event ready
  return st

/-- Stream `s`'s server connection opened: its ordinal header first, under `ordinals`, then what `pump` writes. -/
def opened (st : State) (s : Nat) : State × Array Action :=
  let (st', acts) := (st.setStream s { st.stream s with phase := .open }).pump s
  (st', if st.config.ordinals then #[.writeUpstream s (ordinalHeader s) false] ++ acts else acts)

end State

open State in
/-- **The device.** One event, the state it leaves and the proxy's actions, in order. -/
def step (st : State) (ev : Event) : State × List Action :=
  if st.done then (st, []) else
  let (st, acts) : State × Array Action := match ev with
    | .tick .egress now backlog =>
      if st.shaper.t < st.ticks st.config.egress then st.egressTick now backlog else (st, #[])
    | .tick .ingress now _ =>
      if st.bucketer.t < st.ticks st.config.ingress then st.ingressTick now else (st, #[])
    | .link now bytes => if st.linkOpen then (st.linkRead now bytes, #[]) else (st, #[])
    | .connected s _ =>
      if st.streams.contains s && (st.stream s).phase == .connecting then st.opened s else (st, #[])
    | .refused s now =>
      if st.streams.contains s && (st.stream s).phase == .connecting then
        let st := { st with bucketer := { st.bucketer with
          certifier := st.bucketer.certifier.doFault .deliveryRefused } }
        (st.setStream s { st.stream s with phase := .closed, ended := some (st.bucketReady now) }, #[])
      else (st, #[])
    | .upstream s now bytes =>
      if (st.stream s).phase == .open && st.streams.contains s then (st.upstreamRead s now bytes, #[]) else (st, #[])
    | .delivered s now =>
      let x := st.stream s
      if st.streams.contains s && x.phase == .open && x.draining then
        (st.setStream s { x with delivered := some now, draining := false }).pump s
      else (st, #[])
    | .saltConnected _ =>
      match st.saltConn with
      | some x => if x.phase == .connecting then { st with saltConn := some { x with phase := .open } }.saltPump
        else (st, #[])
      | none => (st, #[])
    | .saltRefused _ =>
      match st.saltConn with
      | some x =>
        if x.phase == .connecting then
          ({ st with bucketer := { st.bucketer with certifier := st.bucketer.certifier.doFault .deliveryRefused },
                     saltConn := some { x with phase := .closed } }, #[])
        else (st, #[])
      | none => (st, #[])
    | .saltDelivered _ =>
      match st.saltConn with
      | some x =>
        if x.phase == .open && x.draining then
          { st with saltConn := some { x with draining := false }, saltsDelivered := st.saltsDelivered + 1 }.saltPump
        else (st, #[])
      | none => (st, #[])
  (st, acts.toList)

end CertifierDevice
