import NetworkCertifier.Run
import NetworkCheck.Certificate

/-!
# One link's certifier: constant-rate, left-packed, FIFO and fail-closed

The executable form of `grid.py`'s `Certifier` and `Record` (W1–W6). A grid's parameters are `NetworkCheck.Params` (T, r,
B, τ_b, Q), the record's status is `NetworkCertifier.Status`, and a slot holds a frame's bytes or padding (`none`), as in
`grid.Row`.
-/

namespace CertifierDevice

open NetworkCertifier (Reason Status)

abbrev Params := NetworkCheck.Params

abbrev Slots := Array (Option ByteArray)

/-- `Reason`'s value, as `grid.Reason` spells it. -/
def reasonValue : Reason → String
  | .monitorUnavailable => "monitor-unavailable"
  | .overflow => "overflow"
  | .malformed => "malformed"
  | .dropped => "dropped"
  | .deliveryRefused => "delivery-refused"
  | .coverageNeverEstablished => "coverage-never-established"

/-- A status's reason, or none when complete. -/
def Status.reason? : Status → Option Reason
  | .complete => none
  | .notEstablished r => some r

/-- `grid.Record` (W5): the network certificate of one sampling block is exactly the grid and one status. -/
structure Record where
  link : String
  window : Nat
  grid : Array Slots
  status : Status

/-- `row_count`: the real frames of a row. -/
def rowCount (row : Slots) : Nat := (row.filter Option.isSome).size

/-- `grid.Certifier`: one link's active certifier, its queue, the rows of the open window and its first fault. -/
structure Certifier where
  link : String
  params : Params
  window : Nat := 0
  queue : Array ByteArray := #[]
  rows : Array Slots := #[]
  fault : Option Reason := none

namespace Certifier

/-- `fault`: the first reason is kept, and the queue is dropped, since nothing is forwarded uncertified. -/
def doFault (c : Certifier) (r : Reason) : Certifier :=
  if c.fault.isNone then { c with fault := some r, queue := #[] } else c

/-- `enqueue`: a frame of the wrong size is malformed, one past Q an overflow; a faulted window takes nothing. With
`first`, the frame goes to the head of the queue, ahead of every frame queued: the ingress certifier's own salt frame. -/
def enqueue (c : Certifier) (frame : ByteArray) (first : Bool := false) : Certifier :=
  if c.fault.isSome then c
  else if frame.size ≠ c.params.frameBytes then c.doFault .malformed
  else if c.queue.size ≥ c.params.queueCapacity then c.doFault .overflow
  else if first then { c with queue := #[frame] ++ c.queue }
  else { c with queue := c.queue.push frame }

/-- `tick`: bucket t releases min(queue, r) frames in queue order, then padding; a faulted window's row is padding. -/
def tick (c : Certifier) : Certifier × Slots :=
  let r := c.params.allocation
  if c.fault.isSome then
    let row := Array.replicate r none
    ({ c with rows := c.rows.push row }, row)
  else
    let n := min c.queue.size r
    let row := (c.queue.extract 0 n).map some ++ Array.replicate (r - n) none
    ({ c with queue := c.queue.extract n c.queue.size, rows := c.rows.push row }, row)

/-- `close_window`: the record, with one untimed status, and the next window; queued frames carry over. -/
def closeWindow (c : Certifier) : Certifier × Record :=
  let status := match c.fault with | none => Status.complete | some r => .notEstablished r
  ({ c with rows := #[], fault := none, window := c.window + 1 }, ⟨c.link, c.window, c.rows, status⟩)

end Certifier

end CertifierDevice
