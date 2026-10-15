import CertifierDevice.Certifier
import CertifierDevice.Online

/-!
# The shaper and the bucketer: the reference's certifiers fed frames as they arrive

The executable forms of `active.py`'s `Shaper` (the honest shaper and the egress certifier: `schedule.run_shaper`'s
handing rule, `h(k) = max(eligible(k), h(k−1))` under the declared sync of the frame's anchor's window, then
`max(h(k), ready(k), last)`) and `Bucketer` (the ingress certifier, each frame enqueued at the bucket it arrived by).
-/

namespace CertifierDevice

open NetworkCertifier (ClockSync SyncSet Reason Status)

/-- A real egress frame as the shaper holds it: its B bytes, the anchor of its stream's request and its index there. -/
structure Frame where
  bytes : ByteArray
  anchor : Nat
  index : Nat

/-- `active.LaggedChoices` on an executor where every choice is ready by its tick (`InlineExecutor`): `start`, right
after tick w·T − lag (window 0's before the first tick), makes window w's choice (`Prover.choose`), and `declare` at
tick w·T declares it, or the envelope if none was started for w. A refused choice is kept (`refused`) and refuses the
window's declaration once. -/
structure LaggedChoices where
  prover : Prover
  next : Option (Nat × ClockSync) := none
  refused : Option Nat := none

namespace LaggedChoices

def arrive (l : LaggedChoices) (position anchor index : Nat) (ready : Int) : LaggedChoices :=
  { l with prover := l.prover.arrive position anchor index ready }

/-- `LaggedChoices.start`: window w's choice on what the prover has read by now; a refusal is kept for w. -/
def start (l : LaggedChoices) : LaggedChoices :=
  let w := l.prover.syncs.size
  let (prover, sync, refused) := l.prover.choose
  { prover, next := some (w, sync), refused := if refused then some w else l.refused }

/-- `LaggedChoices.declare`: window w's declaration, and whether it is refused (Python raises `Refused` once, and the
next call declares the choice made on what was read: the sync returned here). -/
def declare (l : LaggedChoices) : LaggedChoices × ClockSync × Bool :=
  let w := l.prover.syncs.size
  let refused := l.refused == some w
  let sync := match l.next with
    | some (v, s) => if v = w then s else l.prover.envelope
    | none => l.prover.envelope
  ({ prover := l.prover.declare sync, next := none, refused := if refused then none else l.refused }, sync, refused)

end LaggedChoices

/-- `active.Shaper`. `queued` is what the certifier's queue would hold had no fault dropped it. -/
structure Shaper where
  certifier : Certifier
  syncs : Array ClockSync
  syncSet : SyncSet
  prover : Option LaggedChoices
  windows : Nat
  pending : Std.HashMap Nat (Frame × Int) := {}
  next : Nat := 0
  h : Nat := 0
  last : Nat := 0
  t : Nat := 0
  queued : Nat := 0
  hands : Array Nat := #[]

namespace Shaper

/-- `_check_advice`: a window whose declared sync is outside Σ_sync never establishes coverage. -/
def checkAdvice (s : Shaper) : Shaper :=
  match s.syncs[s.certifier.window]? with
  | some sync => if inSet s.syncSet sync then s else { s with certifier := s.certifier.doFault .coverageNeverEstablished }
  | none => s

def make (certifier : Certifier) (syncs : Array ClockSync) (ss : SyncSet) (prover : Option LaggedChoices)
    (windows : Nat) : Shaper :=
  checkAdvice { certifier, syncs, syncSet := ss, prover, windows }

def fault (s : Shaper) (r : Reason) : Shaper := { s with certifier := s.certifier.doFault r }

/-- `Shaper.arrive`: a position already handed or held is malformed, one past Q waiting an overflow. -/
def arrive (s : Shaper) (position : Nat) (frame : Frame) (ready : Int) : Shaper :=
  if position < s.next || s.pending.contains position then s.fault .malformed
  else if s.pending.size ≥ s.certifier.params.queueCapacity then s.fault .overflow
  else { s with pending := s.pending.insert position (frame, ready),
                prover := s.prover.map (·.arrive position frame.anchor frame.index ready) }

/-- `Shaper.tick`: the prover's declaration at a window's first tick (a choice already made), then the frames due by the
handing rule, then the certifier's row, with the window's record at its last bucket. A frame handed at a bucket after
its `h` (late: ready after it, behind a late frame, or read after its tick) faults the window
`coverage-never-established` before it is enqueued. A window closing faulted with frames still queued faults the next
one with the same reason. A declaration the prover refuses faults the window `monitor-unavailable`. -/
def tick (s : Shaper) : Shaper × Slots × Option Record := Id.run do
  let T := s.certifier.params.buckets
  let mut s := s
  if let some p := s.prover then
    if s.t % T = 0 && s.t / T = s.syncs.size && s.syncs.size < s.windows then
      let (p, sync, refused) := p.declare
      if refused then s := s.fault .monitorUnavailable
      s := checkAdvice { s with prover := some p, syncs := s.syncs.push sync }
  for _ in [0:s.pending.size] do
    match s.pending.get? s.next with
    | none => break
    | some (frame, ready) =>
      match s.syncs[frame.anchor / T]? with
      | none => break
      | some sync =>
        let h := max (eligibleAt frame.anchor frame.index sync) s.h
        let hand := (max (max (h : Int) ready) s.last).toNat
        if hand > s.t then break
        let certifier := if h < s.t then s.certifier.doFault .coverageNeverEstablished else s.certifier
        s := { s with pending := s.pending.erase s.next, h, last := hand, next := s.next + 1, hands := s.hands.push hand,
                      certifier := certifier.enqueue frame.bytes, queued := s.queued + 1 }
  let (c, row) := s.certifier.tick
  s := { s with certifier := c, queued := s.queued - min s.queued c.params.allocation, t := s.t + 1 }
  if s.t % T ≠ 0 then return (s, row, none)
  let (c, record) := s.certifier.closeWindow
  s := { s with certifier := c }
  if s.queued > 0 then
    if let some r := Status.reason? record.status then s := s.fault r
  return (checkAdvice s, row, some record)

end Shaper

/-- `active.Bucketer`: frames wait in arrival order with the bucket they arrived by, and enter the certifier at its tick. -/
structure Bucketer where
  certifier : Certifier
  waiting : Array (Int × ByteArray) := #[]
  t : Nat := 0

namespace Bucketer

def arrive (b : Bucketer) (frame : ByteArray) (bucket : Int) : Bucketer := { b with waiting := b.waiting.push (bucket, frame) }

def tick (b : Bucketer) : Bucketer × Slots × Option Record := Id.run do
  let due := (b.waiting.takeWhile fun (x, _) => x ≤ b.t).size
  let c := (b.waiting.extract 0 due).foldl (fun c (_, f) => c.enqueue f) b.certifier
  let (c, row) := c.tick
  let b := { b with certifier := c, waiting := b.waiting.extract due b.waiting.size, t := b.t + 1 }
  if b.t % c.params.buckets ≠ 0 then return (b, row, none)
  let (c, record) := c.closeWindow
  return ({ b with certifier := c }, row, some record)

end Bucketer

end CertifierDevice
