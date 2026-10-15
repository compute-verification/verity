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
  ingress tick ticks the bucketer, hands each released `DATA` or `LAST` frame to its stream (opening the stream's server
  connection at its first frame), records the stream's anchor at its first `LAST`, and publishes the record at the
  window's last bucket.
* **Streams.** A stream's connection is `connect`ed at its first released frame. Until it opens (`connected`), its
  frames wait; then each is written in order (`writeUpstream`), and the first `LAST` written waits for its drain
  (`drain := true`, answered by `delivered`), which is when the request reached the server. A refused connection is
  `delivery-refused` on ingress. The server's bytes are cut into events, each read at time s ready at bucket
  ⌊(s − origin)/τ_b⌋ + 1. An event without a position, too long for a frame, or on a stream with no released request is
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

/-- The online prover's configuration: its rule, envelope, lag (1 … T − 1 buckets) and the history of an earlier run. -/
structure ProverConfig where
  rule : Rule
  envelope : ClockSync
  lag : Nat
  prior : Array Past := #[]

/-- `Proxy`'s parameters: both grids (windows of the same length, frames of at least a header), Σ_sync, the declared
syncs or else the online prover, how many windows run, the clock's time of bucket 0 and the slack, in ns. -/
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

/-- `Proxy.__init__`'s checks, and what the frames need: positive parameters, frames of at least a header, the two
grids' windows of one length, and an online prover declaring every window's sync, with a lag of 1 … T − 1 buckets. -/
def Config.check (c : Config) : Except String Unit := do
  for p in [c.egress, c.ingress] do
    if [p.buckets, p.allocation, p.frameBytes, p.bucketMs, p.queueCapacity].any (· = 0) then
      throw "grid parameters must be positive"
    if p.frameBytes < headerSize then throw s!"a frame holds its {headerSize}-byte header"
  if c.egress.buckets * c.egress.bucketMs ≠ c.ingress.buckets * c.ingress.bucketMs then
    throw "the egress and ingress grids have windows of the same length"
  if let some p := c.prover then
    if !c.syncs.isEmpty then throw "an online prover declares every window's sync"
    if p.lag = 0 || p.lag ≥ c.egress.buckets then
      throw "the proxy's online prover chooses off its tick: a lag of 1 … T − 1 buckets"

/-- Where a stream's server connection is. -/
inductive Phase where
  | connecting
  | open
  | closed
  deriving BEq, DecidableEq

/-- `active._Stream`: its request's anchor, ingress window and forwarding deadline, its frames read so far, the frames
waiting to be written, its event cutter, and when its request was delivered and its ready bucket at its end. -/
structure Stream where
  phase : Phase := .connecting
  anchor : Option Nat := none
  count : Nat := 0
  out : Array (ByteArray × Bool) := #[]
  draining : Bool := false
  events : Events := {}
  window : Option Nat := none
  deadline : Option Nat := none
  delivered : Option Nat := none
  ended : Option Int := none

/-- The proxy's state. `frames` logs every egress frame read as (position, stream, index, anchor, ready). -/
structure State where
  config : Config
  shaper : Shaper
  bucketer : Bucketer
  streams : Std.HashMap Nat Stream := {}
  linkOpen : Bool := true
  linkBuffer : ByteArray := ByteArray.empty
  released : Array Nat := #[]
  late : Array Nat := #[]
  frames : Array (Nat × Nat × Nat × Nat × Int) := #[]
  linkFramesRead : Nat := 0
  requestsReleased : Nat := 0

def nsPerMs : Nat := 1000000

/-- The state before the first tick, window 0's choice started if any window runs. -/
def init (c : Config) : State :=
  let prover := c.prover.map fun p =>
    let l : LaggedChoices := { prover := Prover.make c.syncSet c.egress.buckets p.rule p.envelope p.prior p.lag }
    if c.windows > 0 then l.start else l
  { config := c,
    shaper := Shaper.make { link := c.egressLink, params := c.egress } c.syncs c.syncSet prover c.windows,
    bucketer := { certifier := { link := c.ingressLink, params := c.ingress } } }

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
including the first `LAST` while its request is undelivered, which then waits for its drain. -/
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
  for slot in row.extract 0 (rowCount row) do
    let (kind, s, payload) := decodeFrame (slot.getD ByteArray.empty)
    if kind != kindData && kind != kindLast then continue
    if !st.streams.contains s then
      st := st.setStream s {}
      started := started.push (Started.connect s)
    let mut x := st.stream s
    if x.phase == .open && !x.draining && x.out.isEmpty then started := started.push (.pump s)
    x := { x with out := x.out.push (payload, kind == kindLast) }
    if kind == kindLast && x.anchor.isNone then
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

/-- `_egress_event`: an event at its position, framed with its stream's anchor and its index there. -/
def egressEvent (st : State) (s : Nat) (event : ByteArray) (ready : Int) : State :=
  let x := st.stream s
  let B := st.config.egress.frameBytes
  match eventPosition event, x.anchor, encodeFrame kindData s event B with
  | some position, some anchor, some bytes =>
    let st := st.setStream s { x with count := x.count + 1 }
    { st with frames := st.frames.push (position, s, x.count, anchor, ready),
              shaper := st.shaper.arrive position ⟨bytes, anchor, x.count⟩ ready }
  | _, _, _ => { st with shaper := st.shaper.fault .malformed }

/-- The end of stream `s`'s server connection: bytes left unframed are malformed; the stream's end is its ready bucket. -/
def upstreamEnd (st : State) (s : Nat) (now : Nat) : State :=
  let x := st.stream s
  let st := if x.events.buffer.isEmpty then st else { st with shaper := st.shaper.fault .malformed }
  st.setStream s { x with phase := .closed, draining := false, ended := some (st.bucketReady now) }

/-- One read of stream `s`'s server connection. -/
def upstreamRead (st : State) (s : Nat) (now : Nat) (data : ByteArray) : State := Id.run do
  if data.isEmpty then return st.upstreamEnd s now
  let ready := st.bucketReady now
  let x := st.stream s
  let (events, cut) := x.events.feed data
  let mut st := st.setStream s { x with events }
  for event in cut do st := st.egressEvent s event ready
  return st

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
      if st.streams.contains s && (st.stream s).phase == .connecting then
        (st.setStream s { st.stream s with phase := .open }).pump s
      else (st, #[])
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
  (st, acts.toList)

end CertifierDevice
