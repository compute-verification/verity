import NetworkCheck.Period

/-!
# The relay's schedule: refusal, admission, the egress limit and the cut

The schedule the relay form of the network certifier runs (`verity/core/protocols/certifier/warden.py`'s `Warden`), as
executable Lean over the parameters the auditor reads (`NetworkCheck.Period.Schedule`, `warden.WardenParams`): what the
relay does with each connection, and for one it admits, the tick it admits it and the egress cells its lane sends. It
mirrors `Warden.handle`'s refusal, `Warden._tick`'s admission, `Warden._egress`'s limit and `Warden.run`'s cut, and
reads nothing of a cell's bytes. A connection enters as the first tick that finds it queued (`ready`, the Python's
`arrived` for a connection that arrives off a bucket boundary) and the index its lane's `END` or `FAIL` has, were the
lane neither capped nor cut (`last`), which its server's bytes and their times decide.

- A connection that finds `queue` connections queued is refused (`REFUSED`, with no ordinal, and its window withheld).
  One queued is admitted, in arrival order, at the first tick from its `ready` and its predecessor's admission at which
  fewer than `lanes` lanes are live (`admitAt`), if the run reaches that tick; otherwise the run's end cuts it queued, and
  every later connection it doesn't refuse.
- A lane admitted at `adm` sends cells `0 … last` (`last` its `END` or `FAIL`), at most `egressCells` of them (at cell
  `egressCells − 1` it ends or fails, limit), `r` a bucket from `adm + D`, until the run's end at `horizon` cuts it
  (`laneCells`). It is live through the bucket of its last cell, or to the run's end if the run cuts it before its first
  (`freeAt`).

**What it leaves out.** A connection the relay fails for its client's sake (a malformed cell, or ingress past its bound)
fails at once, before `adm + D` too, and frees its lane in that bucket; its lane is not one this schedule gives.
-/

namespace CertifierDevice.Relay

open NetworkCheck.Period (Schedule)

/-- A connection as the relay's schedule reads it: the first tick that finds it queued (`ready`), and the index of its
lane's `END` or `FAIL` were the lane neither capped nor cut (`last`). -/
structure Conn where
  ready : Nat
  last : Nat
  deriving Repr, DecidableEq

/-- An admitted connection's lane: the tick that admits it (`_Connection.admitted`) and the egress cells it sends
(`_Connection.egress_cells`, with its `END` or `FAIL`), the last its `END` or `FAIL` unless the run cut it. -/
structure Lane where
  adm : Nat
  cells : Nat
  deriving Repr, DecidableEq

/-- What the relay does with a connection: refuses it, admits it to a lane, or holds it queued until the run's end cuts
it. -/
inductive Decision where
  | refused
  | admitted (l : Lane)
  | cut
  deriving Repr, DecidableEq

/-- The cells a lane admitted at `adm` sends before the run's end at `horizon`: through its `END` or `FAIL` at `last`,
at most `egressCells` (`_egress`'s limit), and `r` a bucket from `adm + D` until the cut. -/
def laneCells (s : Schedule) (horizon adm last : Nat) : Nat :=
  min (min (last + 1) s.egressCells) (s.slots * (horizon - (adm + s.delay)))

/-- The first tick at which lane `l` is no longer live: the one after its last cell's bucket, or the run's end for a
lane the run cuts before its first cell. -/
def freeAt (s : Schedule) (horizon : Nat) (l : Lane) : Nat :=
  if l.cells = 0 then horizon else l.adm + s.delay + (l.cells + s.slots - 1) / s.slots

/-- The lanes of `frees` (each lane's `freeAt`) live at tick `t`. -/
def live (frees : List Nat) (t : Nat) : Nat :=
  (frees.filter (t < ·)).length

/-- The tick that admits the queue's head, which finds the lanes `frees` and goes no earlier than `base`: the first tick
from `base` at which fewer than `lanes` lanes are live. No lane is admitted between, so the live count only falls from
`base` on, at ticks lanes free: it is `base` or one of those. -/
def admitAt (s : Schedule) (frees : List Nat) (base : Nat) : Option Nat :=
  ((base :: frees).filter fun t => base ≤ t && live frees t < s.lanes).min?

/-- The relay's state between arrivals: each admitted lane's `freeAt`, each queued connection's admission tick (none
for one the run cuts queued), the last queued connection's (`prev`), and its decisions so far, the latest first. -/
structure State where
  frees : List Nat := []
  queued : List (Option Nat) := []
  prev : Nat := 0
  decisions : List Decision := []

/-- The connections still queued when one the tick `ready` first finds arrives: those no tick before `ready` admits. -/
def waiting (st : State) (ready : Nat) : Nat :=
  (st.queued.filter fun a => a.all (ready ≤ ·)).length

/-- The relay's decision on the next connection `c` to arrive, in a run that ends at `horizon`. -/
def step (s : Schedule) (horizon : Nat) (st : State) (c : Conn) : State :=
  if s.queue ≤ waiting st c.ready then { st with decisions := .refused :: st.decisions }
  else
    match admitAt s st.frees (max c.ready st.prev) with
    | some t =>
      if t < horizon then
        let l : Lane := { adm := t, cells := laneCells s horizon t c.last }
        { frees := freeAt s horizon l :: st.frees, queued := some t :: st.queued, prev := t,
          decisions := .admitted l :: st.decisions }
      else { st with queued := none :: st.queued, prev := horizon, decisions := .cut :: st.decisions }
    | none => { st with queued := none :: st.queued, prev := horizon, decisions := .cut :: st.decisions }

/-- **The relay's run** of `horizon` ticks (`windows · window`): its decision on each connection, in arrival order. -/
def run (s : Schedule) (horizon : Nat) (conns : List Conn) : List Decision :=
  (conns.foldl (step s horizon) {}).decisions.reverse

/-- The lanes a run admits, in admission order, which is the ordinals' (the queue is first in, first out). -/
def lanes (ds : List Decision) : List Lane :=
  ds.filterMap fun
    | .admitted l => some l
    | _ => none

end CertifierDevice.Relay
